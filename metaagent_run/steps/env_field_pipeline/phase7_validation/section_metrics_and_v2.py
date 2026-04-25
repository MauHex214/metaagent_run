"""Section-level metrics + v2 schema generation (Phase 7 redesign).

Replaces the paper-level metric. Each (section, target) pair from
section_eval_results.csv contributes one trial; success is the LLM-extracted
value being non-null.

  n_tries[T]   = # sections in sample where T was a candidate target
  n_success[T] = # sections where LLM returned non-null for T
  rate[T]      = success / tries
  decision[T]:
    n_tries < SECTION_MIN_TRIES → kept_observe (target_section_index lacked
                                  enough sections; rare in this design since
                                  broad pool covers 158/158 with ≥1 section)
    rate < FLOOR                → pruned (drop from v2)
    else                        → kept

Defaults: SECTION_MIN_TRIES=10, FLOOR=0.20 — same threshold semantics as
the paper-level pipeline. Tune as needed; sections are smaller evidence units
so a higher MIN_TRIES (e.g., 20) may be defensible.

CLI:
  python -m metaagent_run.steps.env_field_pipeline.phase7_validation.section_metrics_and_v2
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from metaagent_run.steps.env_field_pipeline import config as ep_config
from metaagent_run.steps.env_field_pipeline.phase7_validation import PHASE7_DIR

ENV6_V1B_PATH = ep_config.OUTPUT_DIR / "env6_extraction_targets_v1b.json"
RESULTS_CSV = PHASE7_DIR / "section_eval_results.csv"
SUCCESS_CSV = PHASE7_DIR / "extraction_success_section_level.csv"
ENV6_V2_PATH = ep_config.OUTPUT_DIR / "env6_extraction_targets_v2.json"
DIFF_CSV = ep_config.OUTPUT_DIR / "env6_v1b_to_v2_diff.csv"

DEFAULT_MIN_TRIES = 10
DEFAULT_FLOOR = 0.20
SAMPLE_FAILURE_LIMIT = 10


def aggregate() -> dict[str, dict]:
    """target → {n_tries, n_success, success_rate, sample_failure_pmids}."""
    tries: dict[str, set] = defaultdict(set)   # set of section_id strings
    success: dict[str, set] = defaultdict(set)
    failures: dict[str, list[str]] = defaultdict(list)

    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row["target"]
            sid = f"{row['pmid']}::{row['section_type']}::{row['section_index']}"
            tries[t].add(sid)
            if int(row["success"]) == 1:
                success[t].add(sid)
            else:
                if len(failures[t]) < SAMPLE_FAILURE_LIMIT:
                    failures[t].append(row["pmid"])

    out = {}
    for t in tries:
        nt = len(tries[t])
        ns = len(success[t])
        out[t] = {
            "n_tries": nt,
            "n_success": ns,
            "success_rate": (ns / nt) if nt else 0.0,
            "sample_failure_pmids": ";".join(failures[t]),
        }
    return out


def write_success_csv(stats: dict, env6_v1b: dict) -> None:
    # target_meta from env6_v1b (envs/tier/subtype) for output rows
    target_meta: dict[str, dict] = {}
    for env_name, env_block in env6_v1b.get("per_environment", {}).items():
        for fdef in env_block.get("fields", []):
            t = fdef["field"]
            entry = target_meta.setdefault(t, {
                "envs": set(), "tier": fdef.get("tier"),
                "subtype": fdef.get("subtype"),
            })
            entry["envs"].add(env_name)

    rows = []
    for t, meta in target_meta.items():
        s = stats.get(t, {"n_tries": 0, "n_success": 0,
                          "success_rate": 0.0, "sample_failure_pmids": ""})
        rows.append({
            "target": t,
            "tier": meta.get("tier"),
            "subtype": meta.get("subtype"),
            "envs": ";".join(sorted(meta.get("envs", []))),
            "n_tries": s["n_tries"],
            "n_success": s["n_success"],
            "success_rate": round(s["success_rate"], 4),
            "sample_failure_pmids": s["sample_failure_pmids"],
        })
    rows.sort(key=lambda r: (-r["n_tries"], r["target"]))
    with open(SUCCESS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {SUCCESS_CSV} ({len(rows)} target rows)")


def generate_v2(stats: dict, env6_v1b: dict, min_tries: int, floor: float) -> dict:
    diff_rows = []
    v2 = json.loads(json.dumps(env6_v1b))
    pruned: set[str] = set()

    for env_name, env_block in v2.get("per_environment", {}).items():
        kept_fields = []
        for fdef in env_block.get("fields", []):
            tname = fdef["field"]
            s = stats.get(tname, {"n_tries": 0, "n_success": 0,
                                  "success_rate": 0.0, "sample_failure_pmids": ""})
            nt, rate = s["n_tries"], s["success_rate"]
            if nt < min_tries:
                decision = "kept_observe"
            elif rate < floor:
                decision = "pruned"
            else:
                decision = "kept"
            diff_rows.append({
                "env": env_name, "target": tname,
                "tier": fdef.get("tier"),
                "n_tries": nt, "n_success": s["n_success"],
                "success_rate": round(rate, 4),
                "decision": decision,
                "sample_failure_pmids": s["sample_failure_pmids"],
            })
            if decision == "pruned":
                pruned.add(tname)
            else:
                kept_fields.append(fdef)
        env_block["fields"] = kept_fields

    meta = v2.setdefault("metadata", {})
    meta["v2_generated_at"] = datetime.now(timezone.utc).isoformat()
    meta["v2_pruning_rule"] = (
        f"section-level: n_tries >= {min_tries} AND success_rate < {floor}"
    )
    meta["v2_evaluation_unit"] = "section"
    meta["v2_pruned_unique_targets"] = sorted(pruned)
    meta["v2_value_level_precision_note"] = (
        "success_rate measures whether the LLM returned a non-null value from "
        "section text. Value-level correctness requires sampled human review."
    )

    with open(ENV6_V2_PATH, "w", encoding="utf-8") as f:
        json.dump(v2, f, ensure_ascii=False, indent=2)

    diff_rows.sort(key=lambda r: (r["decision"], -r["n_tries"], r["target"]))
    with open(DIFF_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(diff_rows[0].keys()))
        w.writeheader()
        w.writerows(diff_rows)

    n_pruned_rows = sum(1 for r in diff_rows if r["decision"] == "pruned")
    n_kept_observe = sum(1 for r in diff_rows if r["decision"] == "kept_observe")
    n_kept = sum(1 for r in diff_rows if r["decision"] == "kept")
    summary = {
        "v2_path": str(ENV6_V2_PATH),
        "diff_path": str(DIFF_CSV),
        "n_pruned_unique": len(pruned),
        "n_pruned_rows": n_pruned_rows,
        "n_kept_observe": n_kept_observe,
        "n_kept": n_kept,
    }
    return summary


def main() -> None:
    p = argparse.ArgumentParser(prog="section_metrics_and_v2")
    p.add_argument("--min-tries", type=int, default=DEFAULT_MIN_TRIES)
    p.add_argument("--success-floor", type=float, default=DEFAULT_FLOOR)
    args = p.parse_args()

    print(f"Reading {RESULTS_CSV} ...", flush=True)
    stats = aggregate()
    print(f"  {len(stats)} targets evaluated", flush=True)

    with open(ENV6_V1B_PATH, "r", encoding="utf-8") as f:
        env6_v1b = json.load(f)
    write_success_csv(stats, env6_v1b)

    summary = generate_v2(stats, env6_v1b, args.min_tries, args.success_floor)
    print()
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
