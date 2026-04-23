import asyncio
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from metaagent_run.core import AsyncLocalModelClient

from .config import PROMPT_VERSION, STOP_SENTINEL
from .prompt_builder import build_mapping_messages

LOGGER = logging.getLogger(__name__)
VALID_CONFIDENCES = {"high", "medium", "low"}


def load_final_fields(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict) and "final_schema" in data:
        fields = data["final_schema"]
        LOGGER.info("从 final_schema 加载字段 (%s)", path.name)
    elif isinstance(data, dict) and "synonym_groups" in data:
        # Step 3 的自动化产物不再单独写 final_schema——字段列表直接
        # 由 synonym_groups 的 canonical 键集派生（二者等价）。
        fields = list(data["synonym_groups"].keys())
        LOGGER.info(
            "从 synonym_groups 键集派生字段列表 (%s, %d canonicals)",
            path.name, len(fields),
        )
    elif isinstance(data, list):
        fields = data
        LOGGER.info("从纯列表加载字段 (%s)", path.name)
    else:
        raise ValueError(f"无法识别字段文件格式: {path}")
    return [str(item).strip() for item in fields if isinstance(item, str) and item.strip()]


def load_mixs_standards(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_pmid_index(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_pmid_env_index(path: Path) -> Dict[str, Dict[str, List[str]]]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_paper_env_map(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def compute_multi_env_pmids(paper_env_map: Dict[str, List[str]]) -> Set[str]:
    """识别属于多个有效子环境的 PMID，用于剔除多环境论文。"""
    from .config import VALID_SUB_ENVS
    multi = set()
    for pmid, envs in paper_env_map.items():
        valid = {e for e in envs if e in VALID_SUB_ENVS}
        if len(valid) > 1:
            multi.add(str(pmid))
    LOGGER.info("识别多环境论文: %d 篇", len(multi))
    return multi


def filter_pmid_index(
    pmid_index: Dict[str, List[str]],
    exclude_pmids: Set[str],
) -> Dict[str, List[str]]:
    """从 field->PMIDs 索引中剔除指定 PMID。"""
    filtered = {}
    for field, pmids in pmid_index.items():
        kept = [p for p in pmids if p not in exclude_pmids]
        if kept:
            filtered[field] = kept
    return filtered


def filter_pmid_env_index(
    pmid_env_index: Dict[str, Dict[str, List[str]]],
    exclude_pmids: Set[str],
) -> Dict[str, Dict[str, List[str]]]:
    """从 field->env->PMIDs 索引中剔除指定 PMID。"""
    filtered = {}
    for field, env_map in pmid_env_index.items():
        new_env_map = {}
        for env, pmids in env_map.items():
            kept = [p for p in pmids if p not in exclude_pmids]
            if kept:
                new_env_map[env] = kept
        if new_env_map:
            filtered[field] = new_env_map
    return filtered


def load_synonym_groups(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data.get("synonym_groups", {})


def load_mapping_review_decisions(path: Path) -> List[Dict[str, Any]]:
    """Load unified mapping review decisions (replaces tier_exclusion +
    mapping_veto + mapping_correction).

    File schema (mapping_review_decisions.json):
      {
        "decisions": [
          {"field": "canonical_name",
           "action": "EXCLUDE" | "FORCE_UNMAPPED" | "FORCE_SLOT",
           "target_slot": "(only for FORCE_SLOT) MIxS Slot_Name",
           "target_title": "(optional, FORCE_SLOT) display title",
           "reason": "reviewer note",
           "reviewer": "(optional) reviewer ID",
           "decided_at": "(optional) ISO date"
          },
          ...
        ]
      }

    Action semantics:
      EXCLUDE        — field is not environmental sample metadata.
                       Pre-LLM: excluded from LLM mapping (saves cost).
                       Post-LLM: any stale mapping is force-unmapped (cleans
                       checkpoint pollution).
      FORCE_UNMAPPED — field IS environmental metadata but has no MIxS
                       equivalent. Pre-LLM: still sent to LLM (in case LLM
                       is right). Post-LLM: LLM's slot stripped to UNMAPPED.
      FORCE_SLOT     — LLM mapped to wrong slot. Post-LLM: override to
                       reviewer-verified target_slot.
    """
    if not path.exists():
        LOGGER.warning("Mapping review decisions 不存在: %s（无人工决策）", path)
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    raw_decisions = data.get("decisions", [])
    # Compatibility: tolerate one level of nesting
    # ({"decisions": {"decisions": [...]}}) which AI tooling has been
    # observed to emit when wrapping the payload with its own metadata block.
    if isinstance(raw_decisions, dict):
        inner = raw_decisions.get("decisions")
        if isinstance(inner, list):
            LOGGER.warning(
                "Mapping review decisions %s nests the list one level deeper "
                "than the canonical schema; unwrapping for compatibility. "
                "Consider flattening at the source.",
                path,
            )
            raw_decisions = inner
        else:
            raw_decisions = []
    valid_actions = {"EXCLUDE", "FORCE_UNMAPPED", "FORCE_SLOT"}
    normalized: List[Dict[str, Any]] = []
    for entry in raw_decisions:
        if not isinstance(entry, dict):
            continue
        field = str(entry.get("field", "")).strip()
        action = str(entry.get("action", "")).strip().upper()
        if not field or action not in valid_actions:
            continue
        if action == "FORCE_SLOT" and not str(entry.get("target_slot", "")).strip():
            LOGGER.warning("FORCE_SLOT 缺少 target_slot 跳过: %s", field)
            continue
        normalized.append({
            "field": field,
            "action": action,
            "target_slot": str(entry.get("target_slot", "")).strip(),
            "target_title": str(entry.get("target_title", "")).strip(),
            "reason": str(entry.get("reason", "")).strip(),
            "reviewer": str(entry.get("reviewer", "")).strip(),
            "decided_at": str(entry.get("decided_at", "")).strip(),
        })
    by_action = {"EXCLUDE": 0, "FORCE_UNMAPPED": 0, "FORCE_SLOT": 0}
    for d in normalized:
        by_action[d["action"]] += 1
    LOGGER.info(
        "加载 Mapping review decisions: %d 条 (%s)  EXCLUDE=%d, FORCE_UNMAPPED=%d, FORCE_SLOT=%d",
        len(normalized), path.name,
        by_action["EXCLUDE"], by_action["FORCE_UNMAPPED"], by_action["FORCE_SLOT"],
    )
    return normalized


def get_pre_llm_excluded_fields(decisions: List[Dict[str, Any]]) -> Set[str]:
    """Extract canonicals to block from LLM mapping (action=EXCLUDE only)."""
    return {d["field"] for d in decisions if d["action"] == "EXCLUDE"}


def clean_llm_response(response: str) -> str:
    text = response.strip()
    if STOP_SENTINEL in text:
        text = text.split(STOP_SENTINEL, 1)[0].strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return text
    except Exception:
        pass

    start = text.find("[")
    if start == -1:
        return text

    in_string = False
    escape = False
    depth = 0
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1].strip()
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, list):
                        return candidate
                except Exception:
                    break
    return text


def validate_mapping_result(
    raw_result: Any,
    expected_fields: List[str],
    valid_slots: Set[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not isinstance(raw_result, list):
        LOGGER.warning("LLM 返回不是 list，类型: %s", type(raw_result))
        return [], list(expected_fields)

    expected_set = set(expected_fields)
    processed_fields: Set[str] = set()
    valid_items: List[Dict[str, Any]] = []

    for item in raw_result:
        if not isinstance(item, dict):
            continue

        field = item.get("field", "")
        if not isinstance(field, str) or not field.strip():
            continue
        field = field.strip()

        matched_field = field
        if field not in expected_set:
            field_lower = field.lower()
            matched = None
            for expected in expected_set:
                if expected.lower() == field_lower:
                    matched = expected
                    break
            if matched is None:
                continue
            matched_field = matched

        if matched_field in processed_fields:
            continue

        mixs_slot = str(item.get("mixs_slot", "UNMAPPED")).strip()
        mixs_title = str(item.get("mixs_title", "UNMAPPED")).strip()
        confidence = str(item.get("confidence", "medium")).lower().strip()
        reason = str(item.get("reason", "")).strip()

        if mixs_slot != "UNMAPPED" and mixs_slot not in valid_slots:
            slot_lower = mixs_slot.lower()
            matched_slot = None
            for valid_slot in valid_slots:
                if valid_slot.lower() == slot_lower:
                    matched_slot = valid_slot
                    break
            if matched_slot:
                mixs_slot = matched_slot
            else:
                mixs_slot = "UNMAPPED"
                mixs_title = "UNMAPPED"

        if confidence not in VALID_CONFIDENCES:
            confidence = "medium"

        valid_items.append(
            {
                "field": matched_field,
                "mixs_slot": mixs_slot,
                "mixs_title": mixs_title,
                "confidence": confidence,
                "reason": reason,
            }
        )
        processed_fields.add(matched_field)

    missing_fields = [field for field in expected_fields if field not in processed_fields]
    return valid_items, missing_fields


def apply_review_decisions_post_llm(
    mapped_results: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
    valid_slots: Set[str],
) -> List[Dict[str, Any]]:
    """Apply post-LLM stage of unified review decisions to mapped_results.

    Single function replaces apply_mapping_vetoes + apply_exclusion_as_veto
    + apply_mapping_corrections. Action semantics:

      EXCLUDE        — strip any LLM mapping (UNMAPPED, with audit fields).
                       Cleans up stale checkpoint mappings of pre-LLM-blocked
                       fields. Idempotent if field already UNMAPPED.
      FORCE_UNMAPPED — strip LLM's slot; field stays in mapped_results as
                       UNMAPPED with audit fields.
      FORCE_SLOT     — override LLM's slot with reviewer-verified target_slot.
                       Skipped if field already UNMAPPED (cannot re-promote
                       a blocked field; see priority rule below).

    Conflict resolution: if a field has multiple decisions, the most
    aggressive action wins (EXCLUDE > FORCE_UNMAPPED > FORCE_SLOT).
    This keeps the curation hierarchy monotonic.
    """
    if not decisions:
        LOGGER.info("Review decisions: 无规则 (空 decisions list)")
        return mapped_results

    # If multiple decisions exist for one field, keep the most aggressive
    priority = {"EXCLUDE": 0, "FORCE_UNMAPPED": 1, "FORCE_SLOT": 2}
    field_decision: Dict[str, Dict[str, Any]] = {}
    for d in decisions:
        existing = field_decision.get(d["field"])
        if existing is None or priority[d["action"]] < priority[existing["action"]]:
            field_decision[d["field"]] = d

    counts = {"EXCLUDE": 0, "FORCE_UNMAPPED": 0, "FORCE_SLOT": 0,
              "skipped_invalid_slot": 0, "skipped_no_change": 0}

    adjusted: List[Dict[str, Any]] = []
    for record in mapped_results:
        new_record = dict(record)
        field = str(record.get("field", ""))
        d = field_decision.get(field)
        if d is None:
            adjusted.append(new_record)
            continue

        action = d["action"]
        current_slot = new_record.get("mixs_slot")
        original_slot = current_slot or "UNMAPPED"
        original_title = new_record.get("mixs_title", "UNMAPPED")
        original_reason = new_record.get("reason", "")

        if action == "EXCLUDE":
            if current_slot == "UNMAPPED":
                counts["skipped_no_change"] += 1
            else:
                new_record.update({
                    "mixs_slot": "UNMAPPED",
                    "mixs_title": "UNMAPPED",
                    "reason": d.get("reason") or "Excluded as non-environmental sample metadata.",
                    "review_decision_applied": True,
                    "review_action": "EXCLUDE",
                    "review_reviewer": d.get("reviewer", ""),
                    "original_mixs_slot": original_slot,
                    "original_mixs_title": original_title,
                    "original_llm_reason": original_reason,
                })
                counts["EXCLUDE"] += 1

        elif action == "FORCE_UNMAPPED":
            if current_slot == "UNMAPPED":
                counts["skipped_no_change"] += 1
            else:
                new_record.update({
                    "mixs_slot": "UNMAPPED",
                    "mixs_title": "UNMAPPED",
                    "reason": d.get("reason") or "Reviewer-rejected mapping; no MIxS equivalent.",
                    "review_decision_applied": True,
                    "review_action": "FORCE_UNMAPPED",
                    "review_reviewer": d.get("reviewer", ""),
                    "original_mixs_slot": original_slot,
                    "original_mixs_title": original_title,
                    "original_llm_reason": original_reason,
                })
                counts["FORCE_UNMAPPED"] += 1

        elif action == "FORCE_SLOT":
            target = d.get("target_slot")
            if not target or target not in valid_slots:
                LOGGER.warning("FORCE_SLOT target invalid; skipped: %s -> %s", field, target)
                counts["skipped_invalid_slot"] += 1
            elif current_slot == "UNMAPPED":
                # Don't re-promote a field that is currently UNMAPPED;
                # if upstream upgrades made it UNMAPPED that intent wins.
                # (When the file is consistent this branch is unreachable
                # because we resolved conflicts up-front.)
                counts["skipped_no_change"] += 1
            elif current_slot == target:
                counts["skipped_no_change"] += 1
            else:
                new_record.update({
                    "mixs_slot": target,
                    "mixs_title": d.get("target_title") or target,
                    "reason": (d.get("reason") or "Reviewer-corrected slot mapping.")
                              + " (corrected from {})".format(original_slot),
                    "review_decision_applied": True,
                    "review_action": "FORCE_SLOT",
                    "review_reviewer": d.get("reviewer", ""),
                    "original_mixs_slot": original_slot,
                    "original_mixs_title": original_title,
                    "original_llm_reason": original_reason,
                })
                counts["FORCE_SLOT"] += 1

        adjusted.append(new_record)

    LOGGER.info(
        "Review decisions applied (post-LLM):  EXCLUDE=%d  FORCE_UNMAPPED=%d  FORCE_SLOT=%d  (skipped: invalid_slot=%d, no_change=%d)",
        counts["EXCLUDE"], counts["FORCE_UNMAPPED"], counts["FORCE_SLOT"],
        counts["skipped_invalid_slot"], counts["skipped_no_change"],
    )
    return adjusted


async def map_batch(
    fields_batch: List[str],
    llm_client: AsyncLocalModelClient,
    mixs_standards: List[Dict[str, str]],
    synonym_groups: Dict[str, List[str]],
    valid_slots: Set[str],
    max_retries: int = 3,
    request_interval: float = 0.5,
    prompt_version: str = PROMPT_VERSION,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    remaining_fields = list(fields_batch)
    all_mapped: List[Dict[str, Any]] = []

    for attempt in range(max_retries):
        if not remaining_fields:
            break

        messages = build_mapping_messages(
            remaining_fields,
            mixs_standards,
            synonym_groups,
            prompt_version=prompt_version,
        )
        response = await llm_client.chat(messages=messages, max_retries=3, base_backoff=3.0)

        if response is None:
            LOGGER.warning(
                "批次映射第 %d/%d 次：LLM 返回 None，%d 个字段待重试",
                attempt + 1,
                max_retries,
                len(remaining_fields),
            )
            await asyncio.sleep(request_interval * (2**attempt))
            continue

        response_clean = clean_llm_response(response)
        try:
            parsed = json.loads(response_clean)
        except json.JSONDecodeError as error:
            LOGGER.warning(
                "批次映射第 %d/%d 次：JSON 解析失败(%s)，%d 个字段待重试",
                attempt + 1,
                max_retries,
                error,
                len(remaining_fields),
            )
            await asyncio.sleep(request_interval * (2**attempt))
            continue

        valid_items, missing_fields = validate_mapping_result(
            parsed, remaining_fields, valid_slots
        )
        all_mapped.extend(valid_items)
        remaining_fields = missing_fields
        if not missing_fields:
            break
        await asyncio.sleep(request_interval)

    return all_mapped, remaining_fields


async def mapping_pipeline(
    all_fields: List[str],
    llm_client: AsyncLocalModelClient,
    mixs_standards: List[Dict[str, str]],
    synonym_groups: Dict[str, List[str]],
    batch_size: int = 15,
    max_retries_per_batch: int = 3,
    request_interval: float = 0.5,
    checkpoint_path: str = "mapping_checkpoint.json",
    resume: bool = True,
    prompt_version: str = PROMPT_VERSION,
    checkpoint_loader: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    checkpoint_saver: Optional[Callable[..., None]] = None,
    concurrency: int = 8,
    checkpoint_every: int = 5,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    valid_slots = {entry["Slot_Name"] for entry in mixs_standards}
    all_mapped: List[Dict[str, Any]] = []
    all_failed: List[str] = []

    if resume and checkpoint_loader is not None:
        checkpoint = checkpoint_loader(checkpoint_path)
        if checkpoint is not None:
            all_mapped = checkpoint.get("mapped_results", [])
            all_failed = checkpoint.get("failed_fields", [])

    already_done = {item["field"] for item in all_mapped} | set(all_failed)
    remaining = [field for field in all_fields if field not in already_done]
    if not remaining:
        LOGGER.info("所有字段已处理完毕（来自 Checkpoint）")
        return all_mapped, all_failed

    from tqdm.auto import tqdm

    batches = [remaining[index : index + batch_size] for index in range(0, len(remaining), batch_size)]
    LOGGER.info(
        "MIxS 映射启动: %d batches × batch_size=%d, concurrency=%d, checkpoint_every=%d",
        len(batches), batch_size, concurrency, checkpoint_every,
    )

    semaphore = asyncio.Semaphore(concurrency)
    state_lock = asyncio.Lock()
    progress = tqdm(total=len(batches), desc="MIxS 映射", unit="batch", dynamic_ncols=True)
    completed_batches = 0

    async def process_one(batch: List[str]) -> None:
        nonlocal completed_batches
        async with semaphore:
            mapped_items, failed_items = await map_batch(
                fields_batch=batch,
                llm_client=llm_client,
                mixs_standards=mixs_standards,
                synonym_groups=synonym_groups,
                valid_slots=valid_slots,
                max_retries=max_retries_per_batch,
                request_interval=request_interval,
                prompt_version=prompt_version,
            )
        async with state_lock:
            all_mapped.extend(mapped_items)
            all_failed.extend(failed_items)
            completed_batches += 1
            progress.update(1)
            progress.set_postfix_str(
                "mapped={} unmapped={} failed={}".format(
                    sum(1 for item in all_mapped if item["mixs_slot"] != "UNMAPPED"),
                    sum(1 for item in all_mapped if item["mixs_slot"] == "UNMAPPED"),
                    len(all_failed),
                )
            )
            should_save = (
                checkpoint_saver is not None
                and completed_batches % checkpoint_every == 0
            )
            if should_save:
                checkpoint_saver(
                    path=checkpoint_path,
                    mapped_results=all_mapped,
                    failed_fields=all_failed,
                    processed_count=len(all_mapped) + len(all_failed),
                    total_count=len(all_fields),
                )

    try:
        await asyncio.gather(*[process_one(batch) for batch in batches])
    finally:
        progress.close()

    if checkpoint_saver is not None:
        checkpoint_saver(
            path=checkpoint_path,
            mapped_results=all_mapped,
            failed_fields=all_failed,
            processed_count=len(all_mapped) + len(all_failed),
            total_count=len(all_fields),
        )

    return all_mapped, all_failed


# ═════════════ Reviewer-ready CSV queue generator ═════════════

_TENSION_STRONG_PATTERNS = [
    r"\bclosest\b", r"\bnot (?:an? )?exact(?:ly)? match\b",
    r"\bnot a perfect\b", r"\bno exact\b", r"\bdoes not exactly\b",
    r"\bmore specific\b", r"\bless specific\b", r"\bmore general\b",
    r"\bbroader\b(?! than)", r"\bnarrower\b",
    r"\bproxy\b", r"\bsurrogate\b",
    r"\balthough\b", r"\bthough not\b", r"\bdespite\b",
    r"\bnot a direct match\b", r"\bimperfect\b",
    r"\bfallback\b", r"\bgeneralized match\b",
]
_TENSION_WEAK_PATTERNS = [
    r"\bapproximat(?:ion|e|ely)\b", r"\bindirect(?:ly)?\b",
    r"\bpartial(?:ly)?\b", r"\brather than\b", r"\bhowever\b",
    r"\bmight\b", r"\bmay be\b", r"\bcould (?:be|potentially)\b",
    r"\blikely\b", r"\bperhaps\b",
    r"\bif (?:this refers|it is|assumed)\b", r"\bassuming\b",
    r"\bcontext is unclear\b", r"\bwithout (?:additional )?context\b",
    r"\bbest (?:match|fit|available)\b",
    r"\bfits (?:under|into|within)\b", r"\buncertain\b",
]


def _tension_level(reason: str) -> str:
    r = (reason or "").lower()
    for p in _TENSION_STRONG_PATTERNS:
        if re.search(p, r):
            return "strong"
    for p in _TENSION_WEAK_PATTERNS:
        if re.search(p, r):
            return "weak"
    return ""


def generate_review_queue_csv(
    output_path: Path,
    mapped_results: List[Dict[str, Any]],
    included_entries: List[Dict[str, Any]],
    synonym_groups: Dict[str, List[str]],
    pmid_index: Dict[str, List[str]],
) -> None:
    """Produce a reviewer-ready CSV covering all fields that will enter
    Step 5 extraction — both mapped (LLM-aligned to MIxS slot) and
    UNMAPPED-but-high-frequency (tier2 fields that passed the env-pct
    threshold). Fields below the env threshold are NOT in the CSV
    because they don't reach Step 5 — reviewing them has no impact.

    Two kinds of rows:
      (1) Mapped canonicals whose slot made it to Step 5 target list —
          reviewer judges whether LLM's slot assignment is correct.
      (2) UNMAPPED canonicals whose PMID qualified as an independent
          Step 5 target (e.g. season, area, station) — reviewer judges
          whether this field is truly environmental metadata or should
          be EXCLUDED (e.g. abundance sneaks in this way).

    Columns:
      priority             — 'high' | 'medium' | 'low'
      field                — canonical name (Step 3 key)
      total_pmid           — union over synonym-group aliases
      mixs_slot            — LLM's slot, or 'UNMAPPED' for (2)
      mixs_title           — MIxS display title (empty for UNMAPPED)
      target_category      — Universal / Shared / Signature (step-5 category)
      confidence           — LLM self-reported (empty for UNMAPPED rows)
      tension_flag         — 'strong' | 'weak' | '' (from reason regex)
      slot_contribs        — how many canonicals map to this slot
                             ('' for UNMAPPED rows since they are not
                             slot-aggregated)
      review_applied       — Y/N whether a review decision already
                             modified this entry
      reason               — LLM rationale (truncated 250 chars)

    Priority rule (both kinds):
      high   — total_pmid ≥ 100  OR  tension_flag='strong'
      medium — tension_flag='weak'  OR  confidence='low'
      low    — everything else
    """
    import csv
    from collections import Counter

    # Cache per-canonical PMID total
    pmid_cache: Dict[str, int] = {}

    def total_pmid(field: str) -> int:
        if field not in pmid_cache:
            members = synonym_groups.get(field, [field])
            u: Set[str] = set()
            for m in members:
                u.update(pmid_index.get(m, []))
            pmid_cache[field] = len(u)
        return pmid_cache[field]

    # Build two indexes from included entries:
    #   slot → (category, contributing_fields, mapped=True)
    #   canonical → (category) for UNMAPPED tier2 entries
    slot_to_entry: Dict[str, Dict[str, Any]] = {}
    unmapped_canonical_to_entry: Dict[str, Dict[str, Any]] = {}
    for e in included_entries:
        if e.get("mapped"):
            slot_to_entry[e.get("slot", "")] = e
        else:
            unmapped_canonical_to_entry[e.get("label", "")] = e

    # Slot contrib counts (only over kept-mapped rows)
    slot_contribs = Counter()
    for e in mapped_results:
        slot = e.get("mixs_slot")
        if slot and slot != "UNMAPPED":
            slot_contribs[slot] += 1

    priority_rank = {"high": 0, "medium": 1, "low": 2}
    rows: List[Dict[str, Any]] = []

    # ── (1) Mapped rows: LLM-mapped canonicals whose slot enters step 5 ──
    for e in mapped_results:
        slot = e.get("mixs_slot", "")
        if not slot or slot == "UNMAPPED":
            continue
        included_entry = slot_to_entry.get(slot)
        if included_entry is None:
            # slot was mapped but didn't make it past env-threshold;
            # reviewer has no reason to audit it — it won't enter step 5
            continue

        field = str(e.get("field", ""))
        pmid = total_pmid(field)
        reason = str(e.get("reason", ""))
        tension = _tension_level(reason)
        conf = str(e.get("confidence", "")).lower()

        if pmid >= 100 or tension == "strong":
            priority = "high"
        elif tension == "weak" or conf == "low":
            priority = "medium"
        else:
            priority = "low"

        rows.append({
            "priority": priority,
            "field": field,
            "total_pmid": pmid,
            "mixs_slot": slot,
            "mixs_title": e.get("mixs_title", ""),
            "target_category": included_entry.get("category", ""),
            "confidence": conf,
            "tension_flag": tension,
            "slot_contribs": slot_contribs.get(slot, 0),
            "review_applied": "Y" if e.get("review_decision_applied") else "N",
            "reason": (reason[:247] + "..." if len(reason) > 250 else reason),
        })

    # ── (2) UNMAPPED rows: tier2 canonicals that qualified as step-5 targets ──
    # Reviewer judges whether these "LLM couldn't find a MIxS match" fields
    # are really environmental metadata, or should be EXCLUDEd entirely.
    # Typical EXCLUDE candidates that slip through: abundance, biomass,
    # primary_prod etc. (analytical results passing frequency threshold).
    for field, entry in unmapped_canonical_to_entry.items():
        pmid = total_pmid(field) if field in synonym_groups or field in pmid_index else entry.get("total_pmid", 0)
        if pmid >= 100:
            priority = "high"
        else:
            priority = "medium"
        rows.append({
            "priority": priority,
            "field": field,
            "total_pmid": pmid,
            "mixs_slot": "UNMAPPED",
            "mixs_title": "",
            "target_category": entry.get("category", ""),
            "confidence": "",
            "tension_flag": "",
            "slot_contribs": "",
            "review_applied": "",
            "reason": "(LLM found no MIxS match; enters Step 5 as independent field — audit whether this is true environmental metadata)",
        })

    rows.sort(key=lambda r: (priority_rank[r["priority"]], -r["total_pmid"]))

    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "priority", "field", "total_pmid", "mixs_slot", "mixs_title",
            "target_category", "confidence", "tension_flag", "slot_contribs",
            "review_applied", "reason",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    by_priority = Counter(r["priority"] for r in rows)
    by_slot_type = Counter("UNMAPPED" if r["mixs_slot"] == "UNMAPPED" else "mapped" for r in rows)
    LOGGER.info(
        "Review queue CSV written: %s  rows=%d  (high=%d, medium=%d, low=%d)  types=(mapped=%d, unmapped=%d)",
        output_path, len(rows),
        by_priority["high"], by_priority["medium"], by_priority["low"],
        by_slot_type["mapped"], by_slot_type["UNMAPPED"],
    )
