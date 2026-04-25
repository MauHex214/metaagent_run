"""LLM-on-section extractability evaluator (Phase 7 redesign).

For each sampled section S with candidate_targets [T1, T2, ...]:
  - Fetch S's text from step5 input file (the 7.5GB segment universe)
  - Build prompt: section text + per-target alias list
  - LLM call (DeepSeek-V3 via step5's AsyncLocalModelClient)
  - Parse JSON: {target: value or null}
  - Record per (section, target) success boolean

Concurrency: bounded async (default 16). Checkpointing: each section's result
appended to a JSONL file as it completes — re-runs skip already-done sections.

CLI:
  python -m metaagent_run.steps.env_field_pipeline.phase7_validation.section_evaluator \\
      --concurrency 16
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from metaagent_run.core import AsyncLocalModelClient


def _parse_json_from_response(response: str) -> dict | None:
    """Strip optional markdown code-fence (```json ... ``` or ``` ... ```)
    then JSON-load. Fallback: find first '{' and matching last '}'.
    Returns dict or None."""
    if not response:
        return None
    s = response.strip()
    # Strip code fence
    if s.startswith("```"):
        # Drop first line (``` or ```json)
        nl = s.find("\n")
        if nl >= 0:
            s = s[nl + 1:]
        # Drop trailing ``` line
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    s = s.strip()
    # Try direct parse
    try:
        v = json.loads(s)
        if isinstance(v, dict):
            return v
    except Exception:
        pass
    # Fallback: substring between first '{' and last '}'
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            v = json.loads(s[i:j + 1])
            if isinstance(v, dict):
                return v
        except Exception:
            pass
    return None
from metaagent_run.steps.env_field_pipeline import config as ep_config
from metaagent_run.steps.env_field_pipeline.phase7_validation import (
    PHASE7_DIR,
    ensure_phase7_dir,
)
from metaagent_run.steps.step5 import config as step5_config

PROMPT_PATH = (Path(__file__).parent / "prompts" / "section_eval_v1.txt")
SAMPLED_SECTIONS_PATH = PHASE7_DIR / "sampled_sections.csv"
ENV6_V1B_PATH = ep_config.OUTPUT_DIR / "env6_extraction_targets_v1b.json"
SECTION_TEXT_INDEX_PATH = PHASE7_DIR / "section_text_index.json"
RESULTS_JSONL_PATH = PHASE7_DIR / "section_eval_results.jsonl"
RESULTS_CSV_PATH = PHASE7_DIR / "section_eval_results.csv"

DEFAULT_CONCURRENCY = 16
MAX_TEXT_CHARS = 8000   # truncate long section texts to fit prompt budget


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower().strip()).strip("_")


# ─── Section text index (built once, cached) ────────────────────────────

def build_section_text_index() -> dict:
    """Map (pmid, st, idx) → representative text segment from the 7.5GB
    step5 input. When multiple segments share (pmid, st, idx), pick the
    longest non-trivial one (skip pure-table titles, prefer body text)."""
    print(f"Loading {step5_config.RuntimeConfig.input_file} (7.5GB) ...",
          flush=True)
    input_path = ep_config.PROJECT_ROOT_DIR / "target_env_v1_relation_input.json"
    t0 = time.time()
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"  loaded {len(records):,} segments in {time.time()-t0:.1f}s",
          flush=True)

    # Filter to sections we actually need (sampled set)
    needed: set[tuple[str, str, int]] = set()
    with open(SAMPLED_SECTIONS_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            needed.add((row["pmid"], row["section_type"], int(row["section_index"])))
    print(f"  need text for {len(needed)} unique sections", flush=True)

    # Group all segments per (pmid, st, idx), pick longest text.
    seg_by_key: dict[tuple[str, str, int], str] = {}
    for r in records:
        pmid = str(r.get("pmid", ""))
        st = r.get("section_type", "")
        idx = int(r.get("index", 0))
        key = (pmid, st, idx)
        if key not in needed:
            continue
        text = (r.get("text") or "").strip()
        if not text:
            continue
        prev = seg_by_key.get(key, "")
        if len(text) > len(prev):
            seg_by_key[key] = text

    matched = len(seg_by_key)
    missing = len(needed) - matched
    print(f"  matched {matched}/{len(needed)} sections "
          f"({missing} missing — these have no body text in step5 input)",
          flush=True)

    out = {
        "version": 1,
        "n_sections": matched,
        # JSON keys must be strings → use "pmid|st|idx"
        "texts": {f"{p}|{s}|{i}": t for (p, s, i), t in seg_by_key.items()},
    }
    with open(SECTION_TEXT_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"Wrote {SECTION_TEXT_INDEX_PATH} "
          f"({SECTION_TEXT_INDEX_PATH.stat().st_size / 1e6:.1f} MB)",
          flush=True)
    return out


def load_section_text_index() -> dict[tuple[str, str, int], str]:
    if not SECTION_TEXT_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{SECTION_TEXT_INDEX_PATH} not found. Run with subcommand "
            f"`build-text-index` first."
        )
    with open(SECTION_TEXT_INDEX_PATH, "r", encoding="utf-8") as f:
        idx = json.load(f)
    out: dict[tuple[str, str, int], str] = {}
    for k, v in idx["texts"].items():
        p, s, i = k.split("|", 2)
        out[(p, s, int(i))] = v
    return out


# ─── env6_v1b alias map for prompt construction ─────────────────────────

def load_target_aliases() -> dict[str, list[str]]:
    """canonical → list of human-friendly aliases (incl. canonical name itself).
    Used to render the prompt's target_block."""
    with open(ENV6_V1B_PATH, "r", encoding="utf-8") as f:
        v1b = json.load(f)
    out: dict[str, set[str]] = defaultdict(set)
    for env_block in v1b.get("per_environment", {}).values():
        for fdef in env_block.get("fields", []):
            name = fdef["field"]
            out[name].add(name)
            for a in fdef.get("aliases") or []:
                if a:
                    out[name].add(a)
    return {t: sorted(s) for t, s in out.items()}


# ─── Prompt building + LLM call ─────────────────────────────────────────

def _render_prompt(section_text: str, candidate_targets: list[str],
                   target_aliases: dict[str, list[str]],
                   prompt_template: str) -> tuple[str, dict]:
    text = section_text
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + "\n... [truncated]"

    target_block_lines = []
    example_lines = []
    # Cap aliases per target to keep prompt size manageable
    MAX_ALIASES_SHOWN = 12
    for t in candidate_targets:
        aliases = target_aliases.get(t, [t])
        # Always lead with canonical name
        canonical = t
        other = [a for a in aliases if a != canonical]
        shown = [canonical] + other[:MAX_ALIASES_SHOWN]
        target_block_lines.append(f"- {canonical} ({', '.join(shown)})")
        example_lines.append(f'  "{canonical}": null')

    target_block = "\n".join(target_block_lines)
    example_keys = ",\n".join(example_lines)
    prompt = prompt_template.format(
        section_text=text,
        target_block=target_block,
        example_keys=example_keys,
    )
    return prompt, {"truncated": len(section_text) > MAX_TEXT_CHARS,
                    "n_targets": len(candidate_targets),
                    "prompt_chars": len(prompt)}


async def _evaluate_section(
    client: AsyncLocalModelClient,
    section_key: tuple[str, str, int],
    section_text: str,
    candidate_targets: list[str],
    target_aliases: dict[str, list[str]],
    prompt_template: str,
) -> dict:
    """Returns dict: {target: value_or_null} for all candidate_targets."""
    prompt, meta = _render_prompt(section_text, candidate_targets,
                                  target_aliases, prompt_template)
    messages = [
        {"role": "system",
         "content": "You are a metadata extraction assistant. Output strict JSON."},
        {"role": "user", "content": prompt},
    ]

    # Try once with normal call; on parse fail, retry with lower temp.
    parsed = None
    for attempt in range(2):
        try:
            response = await client.chat(messages,
                                         temperature_override=0.05 if attempt else 0.1)
        except Exception as e:
            response = None
        if not response:
            continue
        parsed = _parse_json_from_response(response)
        if isinstance(parsed, dict) and parsed:
            break

    out = {t: None for t in candidate_targets}
    if isinstance(parsed, dict):
        for t in candidate_targets:
            v = parsed.get(t)
            if v is None:
                continue
            v_str = str(v).strip()
            if not v_str or v_str.lower() in {"null", "none", "n/a", "na"}:
                continue
            out[t] = v_str
    return out


# ─── Driver ─────────────────────────────────────────────────────────────

async def run_evaluation(concurrency: int) -> None:
    ensure_phase7_dir()
    text_idx = load_section_text_index()
    target_aliases = load_target_aliases()
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

    # Resume support: read JSONL of completed sections
    done: set[tuple[str, str, int]] = set()
    if RESULTS_JSONL_PATH.exists():
        with open(RESULTS_JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add((rec["pmid"], rec["section_type"],
                              int(rec["section_index"])))
                except Exception:
                    pass
        print(f"Resuming: {len(done)} sections already done", flush=True)

    # Load sample list
    todo = []
    with open(SAMPLED_SECTIONS_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["pmid"], row["section_type"], int(row["section_index"]))
            if key in done:
                continue
            text = text_idx.get(key)
            if not text:
                continue
            cand = [t for t in row["candidate_targets"].split(";") if t]
            if not cand:
                continue
            todo.append((key, text, cand))
    print(f"Sections to evaluate: {len(todo)}", flush=True)

    cfg = step5_config.load_runtime_config()
    sem = asyncio.Semaphore(concurrency)
    completed = 0
    failed = 0
    t0 = time.time()
    write_lock = asyncio.Lock()

    async with AsyncLocalModelClient(
        base_url=cfg.base_url, model=cfg.model,
        temperature=cfg.temperature, max_tokens=cfg.max_tokens,
        api_key=cfg.api_key, stop_sentinel=cfg.stop_sentinel,
        api_style=cfg.api_style, azure_api_version=cfg.azure_api_version,
        azure_deployment=cfg.azure_deployment, auth_mode=cfg.auth_mode,
    ) as client:

        async def worker(item):
            nonlocal completed, failed
            (pmid, st, idx), text, cand = item
            async with sem:
                try:
                    result = await _evaluate_section(
                        client, (pmid, st, idx), text, cand,
                        target_aliases, prompt_template,
                    )
                    rec = {
                        "pmid": pmid, "section_type": st, "section_index": idx,
                        "n_candidate_targets": len(cand),
                        "extracted": result,
                    }
                    async with write_lock:
                        with open(RESULTS_JSONL_PATH, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            f.flush()
                except Exception as e:
                    failed += 1
                    print(f"[err] {pmid}/{st}/{idx}: {e}", flush=True)
                completed += 1
                if completed % 50 == 0:
                    elapsed = time.time() - t0
                    rate = completed / elapsed
                    remaining = (len(todo) - completed) / max(rate, 0.001)
                    print(f"  [{completed}/{len(todo)}] "
                          f"({rate:.1f}/s, ~{remaining/60:.1f}min remaining, "
                          f"{failed} failed)", flush=True)

        await asyncio.gather(*[worker(it) for it in todo])

    print(f"\nDone: {completed} evaluated, {failed} failed, "
          f"{time.time()-t0:.1f}s total", flush=True)

    # Materialize CSV from JSONL
    rows = []
    with open(RESULTS_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            for t, v in (rec.get("extracted") or {}).items():
                rows.append({
                    "pmid": rec["pmid"],
                    "section_type": rec["section_type"],
                    "section_index": rec["section_index"],
                    "target": t,
                    "extracted_value": v if v is not None else "",
                    "success": int(v is not None),
                })
    rows.sort(key=lambda r: (r["target"], r["pmid"], r["section_type"], r["section_index"]))
    with open(RESULTS_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pmid", "section_type", "section_index",
                                          "target", "extracted_value", "success"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {RESULTS_CSV_PATH} ({len(rows)} per-(section, target) rows)",
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser(prog="section_evaluator")
    sub = p.add_subparsers(dest="cmd", required=True)

    sb = sub.add_parser("build-text-index",
                         help="One-time: extract section texts from 7.5GB step5 input")
    sb.set_defaults(func=lambda a: build_section_text_index())

    se = sub.add_parser("evaluate", help="Run LLM evaluation on sampled sections")
    se.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    se.set_defaults(func=lambda a: asyncio.run(run_evaluation(a.concurrency)))

    args = p.parse_args()

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(line_buffering=True)
    args.func(args)


if __name__ == "__main__":
    main()
