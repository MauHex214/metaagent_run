"""Generate env6_extraction_targets_v2.json by pruning targets that
extraction empirically fails to recover.

Pruning rule (Q1, briefing §2.4):
    n_tries ≥ MIN_TRIES  AND  success_rate < SUCCESS_RATE_FLOOR
Targets with n_tries < MIN_TRIES are kept (待观察 — insufficient data).

Reads:
  - env_field_pipeline_output/env6_extraction_targets.json (v1)
  - env_field_pipeline_output/phase7_validation/extraction_success_cumulative.csv

Writes:
  - env_field_pipeline_output/env6_extraction_targets_v2.json
  - env_field_pipeline_output/env6_v1_to_v2_diff.csv

LIMITATION (briefing §7.4): success_rate measures whether step5 produced a
non-null value, NOT whether the value is correct. Value-level precision
requires sampled human review before treating v2 as production-ready.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from metaagent_run.steps.env_field_pipeline import config as ep_config
from metaagent_run.steps.env_field_pipeline.phase7_validation import PHASE7_DIR

import os
# Default base = v1b (raw_key_expansion injected) if it exists, else v1.
_V1B = ep_config.OUTPUT_DIR / "env6_extraction_targets_v1b.json"
_V1 = ep_config.OUTPUT_DIR / "env6_extraction_targets.json"
ENV6_V1_PATH = Path(os.environ.get("PHASE7_ENV6_PATH",
                                   str(_V1B if _V1B.exists() else _V1)))
ENV6_V2_PATH = ep_config.OUTPUT_DIR / "env6_extraction_targets_v2.json"
DIFF_PATH = ep_config.OUTPUT_DIR / "env6_v1_to_v2_diff.csv"
SUCCESS_CSV = PHASE7_DIR / "extraction_success_cumulative.csv"

DEFAULT_MIN_TRIES = 10
DEFAULT_SUCCESS_FLOOR = 0.20


def load_per_target_stats() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(SUCCESS_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["target"]] = {
                "n_tries": int(row["n_tries"]),
                "n_success": int(row["n_success"]),
                "success_rate": float(row["success_rate"]),
                "sample_failure_pmids": row["sample_failure_pmids"],
            }
    return out


def decide(stats: dict, min_tries: int, success_floor: float) -> str:
    """Returns one of: 'kept', 'kept_observe', 'pruned'."""
    nt = stats["n_tries"]
    rate = stats["success_rate"]
    if nt < min_tries:
        return "kept_observe"
    if rate < success_floor:
        return "pruned"
    return "kept"


def generate(min_tries: int, success_floor: float) -> dict:
    with open(ENV6_V1_PATH, "r", encoding="utf-8") as f:
        v1 = json.load(f)
    stats = load_per_target_stats()

    diff_rows = []
    v2 = json.loads(json.dumps(v1))  # deep copy
    pruned_targets: set[str] = set()

    for env_name, env_block in v2.get("per_environment", {}).items():
        kept_fields = []
        for fdef in env_block.get("fields", []):
            tname = fdef["field"]
            s = stats.get(tname, {"n_tries": 0, "n_success": 0,
                                  "success_rate": 0.0, "sample_failure_pmids": ""})
            decision = decide(s, min_tries, success_floor)
            diff_rows.append({
                "env": env_name,
                "target": tname,
                "tier": fdef.get("tier"),
                "n_tries": s["n_tries"],
                "n_success": s["n_success"],
                "success_rate": s["success_rate"],
                "decision": decision,
                "sample_failure_pmids": s["sample_failure_pmids"],
            })
            if decision == "pruned":
                pruned_targets.add(tname)
            else:
                kept_fields.append(fdef)
        env_block["fields"] = kept_fields

    # Update metadata block
    meta = v2.setdefault("metadata", {})
    meta["v2_generated_at"] = datetime.now(timezone.utc).isoformat()
    meta["v2_pruning_rule"] = (
        f"n_tries >= {min_tries} AND success_rate < {success_floor}"
    )
    meta["v2_pruned_unique_targets"] = sorted(pruned_targets)
    meta["v2_value_level_precision_note"] = (
        "success_rate counts non-null LLM output, not value correctness; "
        "sampled human review required before production use."
    )

    with open(ENV6_V2_PATH, "w", encoding="utf-8") as f:
        json.dump(v2, f, ensure_ascii=False, indent=2)

    diff_rows.sort(key=lambda r: (r["decision"], -r["n_tries"], r["target"]))
    with open(DIFF_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(diff_rows[0].keys()))
        w.writeheader()
        w.writerows(diff_rows)

    summary = {
        "v2_path": str(ENV6_V2_PATH),
        "diff_path": str(DIFF_PATH),
        "n_pruned_unique": len(pruned_targets),
        "n_kept_observe": sum(1 for r in diff_rows if r["decision"] == "kept_observe"),
        "n_kept": sum(1 for r in diff_rows if r["decision"] == "kept"),
        "n_pruned_rows": sum(1 for r in diff_rows if r["decision"] == "pruned"),
    }
    return summary


def main() -> None:
    p = argparse.ArgumentParser(prog="schema_v2_generator")
    p.add_argument("--min-tries", type=int, default=DEFAULT_MIN_TRIES)
    p.add_argument("--success-floor", type=float, default=DEFAULT_SUCCESS_FLOOR)
    args = p.parse_args()
    summary = generate(args.min_tries, args.success_floor)
    print("Schema v2 generated.")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
