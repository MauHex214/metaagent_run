"""Pass B — family-level LLM partitioning.

For each concept family of size ≥ 2, send its members to the LLM
together with a prompt describing the 5-class qualifier rules; the
LLM returns a partition of the family into equivalence groups. Groups
of size 1 (single-member, no merge needed) are allowed and common.

Concurrency is a ramp-up controller in the spirit of other steps:
start at a low level, raise by a fixed step every N seconds up to a
ceiling. Failures back off with jitter.

This module does NOT do Union-Find across families — that is done by
`family.apply_family_partitions` using the `{family_id: [[members],...]}`
mapping this module returns.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:  # tqdm should always be present in the env, fallback just in case
    tqdm = None  # type: ignore

try:
    from .family import Family
except ImportError:
    from family import Family  # type: ignore


LOGGER = logging.getLogger("step3.dedup_family")


# ═══════════════════════════════════════════════════════════════════
#  Prompt loading
# ═══════════════════════════════════════════════════════════════════

_PROMPT_CACHE: Dict[str, str] = {}


def load_prompt(prompt_version: str = "family_partition_v1") -> str:
    if prompt_version in _PROMPT_CACHE:
        return _PROMPT_CACHE[prompt_version]
    prompt_dir = Path(__file__).resolve().parent / "prompts"
    path = prompt_dir / f"{prompt_version}.txt"
    text = path.read_text(encoding="utf-8")
    _PROMPT_CACHE[prompt_version] = text
    return text


# ═══════════════════════════════════════════════════════════════════
#  Response parsing
# ═══════════════════════════════════════════════════════════════════

_STOP_SENTINEL = "</json>"


def _clean_json_payload(text: str) -> str:
    text = text.strip()
    if _STOP_SENTINEL in text:
        text = text.split(_STOP_SENTINEL, 1)[0].strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text).strip()
    return text


def _extract_groups(payload: Any) -> Optional[List[List[str]]]:
    if not isinstance(payload, dict):
        return None
    groups_raw = payload.get("groups")
    if not isinstance(groups_raw, list):
        return None
    out: List[List[str]] = []
    for g in groups_raw:
        if isinstance(g, dict):
            members = g.get("members")
        elif isinstance(g, list):
            members = g
        else:
            continue
        if not isinstance(members, list):
            continue
        clean = [str(m) for m in members if isinstance(m, (str, int))]
        if clean:
            out.append(clean)
    return out or None


def _reconcile_with_input(
    parsed_groups: List[List[str]],
    input_members: List[str],
) -> Tuple[List[List[str]], Dict[str, int]]:
    """Ensure every input member appears exactly once in the output.

    Returns (reconciled_groups, audit). `audit` reports how many
    members the LLM dropped, duplicated, or hallucinated.
    """
    input_set = set(input_members)
    seen: set = set()
    out_groups: List[List[str]] = []
    dropped = 0
    duplicated = 0
    hallucinated = 0
    for g in parsed_groups:
        kept: List[str] = []
        for m in g:
            if m not in input_set:
                hallucinated += 1
                continue
            if m in seen:
                duplicated += 1
                continue
            seen.add(m)
            kept.append(m)
        if kept:
            out_groups.append(kept)
    # Any input member the LLM didn't place → its own singleton group (fallback).
    for m in input_members:
        if m not in seen:
            dropped += 1
            out_groups.append([m])
    return out_groups, {
        "dropped": dropped,
        "duplicated": duplicated,
        "hallucinated": hallucinated,
    }


# ═══════════════════════════════════════════════════════════════════
#  Per-family LLM call
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FamilyPartitionResult:
    family_id: str
    anchor: str
    groups: List[List[str]]
    status: str                       # "ok" | "fallback" | "skip"
    audit: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def group_count(self) -> int:
        return len(self.groups)


def _singleton_result(family: Family, reason: str = "skip") -> FamilyPartitionResult:
    """When a family has 1 member or we decide to skip LLM, return each
    member as its own singleton group (the safe no-merge partition)."""
    groups = [[d.original] for d in family.members]
    return FamilyPartitionResult(
        family_id=family.id,
        anchor=family.anchor,
        groups=groups,
        status=reason,
    )


async def partition_family(
    family: Family,
    llm_client: Any,
    prompt_version: str = "family_partition_v1",
    max_retries: int = 3,
) -> FamilyPartitionResult:
    """Call the LLM once for this family; parse and reconcile.

    On repeated failure, returns a FALLBACK result where every member
    is its own singleton group — this is conservative (no merges) and
    preserves all members for Pass C attribution.
    """
    if family.size < 2:
        return _singleton_result(family, reason="skip")

    member_names = [d.original for d in family.members]

    system_prompt = load_prompt(prompt_version)
    user_msg = json.dumps(
        {"anchor_word": family.anchor, "members": member_names},
        ensure_ascii=False,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    last_error: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await llm_client.chat(
                messages=messages, max_retries=1, base_backoff=1.0,
            )
            if response is None:
                last_error = f"attempt {attempt}: empty response"
                continue

            payload_text = _clean_json_payload(response)
            try:
                parsed = json.loads(payload_text)
            except json.JSONDecodeError as exc:
                last_error = f"attempt {attempt}: JSON decode error {exc}"
                continue

            raw_groups = _extract_groups(parsed)
            if raw_groups is None:
                last_error = f"attempt {attempt}: no 'groups' array"
                continue

            reconciled, audit = _reconcile_with_input(raw_groups, member_names)
            return FamilyPartitionResult(
                family_id=family.id,
                anchor=family.anchor,
                groups=reconciled,
                status="ok",
                audit=audit,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = f"attempt {attempt}: {type(exc).__name__}: {exc}"
            LOGGER.warning("family %s attempt %d failed: %s",
                           family.id, attempt, exc)

    LOGGER.warning(
        "family %s: LLM failed after %d attempts (%s) — falling back to singletons",
        family.id, max_retries, last_error,
    )
    res = _singleton_result(family, reason="fallback")
    res.error = last_error
    return res


# ═══════════════════════════════════════════════════════════════════
#  Concurrent driver with ramp-up
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RampUpConfig:
    initial: int = 4
    step: int = 8
    ceiling: int = 32
    step_every_seconds: float = 30.0


def _safe_write(msg: str, pbar: Any) -> None:
    """Route a log-style message so it co-exists with a tqdm progress bar."""
    if pbar is not None and tqdm is not None:
        try:
            tqdm.write(msg)
            return
        except Exception:
            pass
    LOGGER.info("%s", msg)


async def run_family_partitioning(
    families: List[Family],
    llm_client: Any,
    prompt_version: str = "family_partition_v1",
    max_retries: int = 3,
    ramp: Optional[RampUpConfig] = None,
    progress_cb: Optional[Any] = None,
    show_progress: bool = True,
) -> Dict[str, FamilyPartitionResult]:
    """Dispatch per-family partition calls under a ramp-up controller.

    Returns {family_id: FamilyPartitionResult}.

    If tqdm is available and `show_progress` is True, a live progress
    bar is rendered on stderr. The bar auto-adjusts to a 5-second
    minimum refresh interval so it remains readable in log files when
    stderr is redirected.
    """
    ramp = ramp or RampUpConfig()
    families = list(families)
    total = len(families)
    results: Dict[str, FamilyPartitionResult] = {}
    completed = 0
    ok_count = 0
    fallback_count = 0
    current_limit = ramp.initial
    semaphore = asyncio.Semaphore(current_limit)
    lock = asyncio.Lock()
    start_ts = asyncio.get_event_loop().time()

    pbar: Any = None
    if show_progress and tqdm is not None and total > 0:
        is_tty = sys.stderr.isatty()
        pbar = tqdm(
            total=total, desc="Pass B", unit="fam",
            mininterval=0.5 if is_tty else 5.0,
            maxinterval=10.0 if is_tty else 60.0,
            smoothing=0.1, dynamic_ncols=True,
            ascii=not is_tty,
        )
        pbar.set_postfix(conc=current_limit, ok=0, fb=0, refresh=False)

    async def _ramp_task() -> None:
        nonlocal current_limit
        while completed < total:
            await asyncio.sleep(ramp.step_every_seconds)
            new_limit = min(current_limit + ramp.step, ramp.ceiling)
            if new_limit > current_limit:
                for _ in range(new_limit - current_limit):
                    semaphore.release()
                _safe_write(
                    f"[ramp-up] concurrency {current_limit} → {new_limit}", pbar,
                )
                current_limit = new_limit
                if pbar is not None:
                    pbar.set_postfix(
                        conc=current_limit, ok=ok_count, fb=fallback_count,
                        refresh=False,
                    )
            if current_limit >= ramp.ceiling:
                return

    async def _run_one(fam: Family) -> None:
        nonlocal completed, ok_count, fallback_count
        async with semaphore:
            res = await partition_family(
                fam, llm_client, prompt_version=prompt_version,
                max_retries=max_retries,
            )
        async with lock:
            results[fam.id] = res
            completed += 1
            if res.status == "ok":
                ok_count += 1
            elif res.status == "fallback":
                fallback_count += 1
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    conc=current_limit, ok=ok_count, fb=fallback_count,
                    refresh=False,
                )
            if progress_cb is not None:
                progress_cb(completed, total, res)

    ramp_task = asyncio.create_task(_ramp_task())
    try:
        await asyncio.gather(*[_run_one(f) for f in families])
    finally:
        ramp_task.cancel()
        try:
            await ramp_task
        except asyncio.CancelledError:
            pass
        if pbar is not None:
            pbar.close()

    elapsed = asyncio.get_event_loop().time() - start_ts
    LOGGER.info(
        "family partitioning complete: %d families in %.1fs; "
        "ok=%d fallback=%d skip=%d",
        total, elapsed,
        sum(1 for r in results.values() if r.status == "ok"),
        sum(1 for r in results.values() if r.status == "fallback"),
        sum(1 for r in results.values() if r.status == "skip"),
    )
    return results


# ═══════════════════════════════════════════════════════════════════
#  Export helper
# ═══════════════════════════════════════════════════════════════════

def results_to_partitions(
    results: Dict[str, FamilyPartitionResult],
) -> Dict[str, List[List[str]]]:
    """Strip to the shape expected by family.apply_family_partitions."""
    return {fid: r.groups for fid, r in results.items()}
