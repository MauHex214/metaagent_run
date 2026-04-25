"""Decide whether to stop iterating based on cumulative target coverage.

Reads coverage_per_round.csv (produced by extraction_metrics.py) and prints
SATURATED / CONTINUE plus the diagnostic that drove the call.

Stop criterion (briefing §7.2): two consecutive rounds where BOTH
delta_targets_tried/total_tried < 5% AND delta_targets_succeeded/total_succeeded < 5%.
Also requires ≥3 rounds of data so the first delta isn't load-bearing.
"""
from __future__ import annotations

import argparse
import csv
import sys

from metaagent_run.steps.env_field_pipeline.phase7_validation import PHASE7_DIR

COVERAGE_PATH = PHASE7_DIR / "coverage_per_round.csv"
DEFAULT_DELTA_THRESHOLD = 0.05
DEFAULT_CONSECUTIVE_ROUNDS = 2
DEFAULT_MIN_ROUNDS = 3


def check(
    delta_threshold: float = DEFAULT_DELTA_THRESHOLD,
    consecutive: int = DEFAULT_CONSECUTIVE_ROUNDS,
    min_rounds: int = DEFAULT_MIN_ROUNDS,
) -> dict:
    if not COVERAGE_PATH.exists():
        return {"status": "NO_DATA", "reason": f"{COVERAGE_PATH} not found"}

    with open(COVERAGE_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < min_rounds:
        return {"status": "CONTINUE",
                "reason": f"only {len(rows)} round(s); need ≥{min_rounds}",
                "rows": rows}

    # Compute relative deltas per round (skip first row which has 100% delta)
    flagged_rounds = []
    for r in rows[1:]:
        n_tried = int(r["targets_tried"]) or 1
        n_succ = int(r["targets_succeeded"]) or 1
        d_tried = int(r["delta_targets_tried"]) / n_tried
        d_succ = int(r["delta_targets_succeeded"]) / n_succ
        flagged_rounds.append({
            "round": int(r["round"]),
            "rel_delta_tried": round(d_tried, 4),
            "rel_delta_succ": round(d_succ, 4),
            "below_threshold": d_tried < delta_threshold and d_succ < delta_threshold,
        })

    # Look for `consecutive` flagged rounds at the tail
    tail = flagged_rounds[-consecutive:]
    if len(tail) >= consecutive and all(x["below_threshold"] for x in tail):
        return {"status": "SATURATED",
                "reason": f"last {consecutive} rounds had delta < {delta_threshold:.0%} on both axes",
                "tail": tail}
    return {"status": "CONTINUE",
            "reason": f"latest round delta exceeds {delta_threshold:.0%}",
            "tail": tail}


def main() -> None:
    p = argparse.ArgumentParser(prog="saturation_check")
    p.add_argument("--delta-threshold", type=float, default=DEFAULT_DELTA_THRESHOLD)
    p.add_argument("--consecutive", type=int, default=DEFAULT_CONSECUTIVE_ROUNDS)
    p.add_argument("--min-rounds", type=int, default=DEFAULT_MIN_ROUNDS)
    args = p.parse_args()

    result = check(args.delta_threshold, args.consecutive, args.min_rounds)
    print(f"STATUS: {result['status']}")
    print(f"  reason: {result['reason']}")
    if "tail" in result:
        for x in result["tail"]:
            print(f"  round {x['round']}: "
                  f"Δtried={x['rel_delta_tried']:.2%} "
                  f"Δsucc={x['rel_delta_succ']:.2%} "
                  f"{'✓' if x['below_threshold'] else '✗'}")

    sys.exit(0 if result["status"] == "SATURATED" else 1)


if __name__ == "__main__":
    main()
