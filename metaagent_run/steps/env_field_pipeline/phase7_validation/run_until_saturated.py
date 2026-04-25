"""Run phase7 rounds in a loop until saturation (or max_rounds hit).

Wraps run_iteration in a sequential loop. After each round:
  - Writes round_NN_summary.json (paper count, target counts, status)
  - Aborts if papers_with_samples == 0 (upstream sanity tripwire)
  - Calls saturation_check; stops if SATURATED
  - If --auto-v2 and saturated, generates env6_extraction_targets_v2.json

Designed for unattended runs via nohup. Output is line-buffered.

CLI:
  nohup python -m metaagent_run.steps.env_field_pipeline.phase7_validation.run_until_saturated \\
      --start-round 2 --max-rounds 4 --per-cell 7 \\
      --strata env,section,era --paper-concurrency 16 \\
      --auto-v2 > phase7_run.log 2>&1 &
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from argparse import Namespace
from datetime import datetime, timezone

from metaagent_run.steps.env_field_pipeline.phase7_validation import (
    PHASE7_DIR,
    ensure_phase7_dir,
)
from metaagent_run.steps.env_field_pipeline.phase7_validation import (
    extraction_metrics,
    run_iteration,
    saturation_check,
    schema_v2_generator,
)


def _round_summary_path(round_num: int) -> str:
    return str(PHASE7_DIR / f"round_{round_num:02d}_summary.json")


def _read_coverage_last_row() -> dict | None:
    import csv
    p = PHASE7_DIR / "coverage_per_round.csv"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def _write_round_summary(round_num: int, status: str, extra: dict) -> None:
    cov = _read_coverage_last_row() or {}
    payload = {
        "round": round_num,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "cumulative_papers": int(cov.get("cumulative_papers", 0) or 0),
        "papers_with_samples": int(cov.get("papers_with_samples", 0) or 0),
        "targets_tried": int(cov.get("targets_tried", 0) or 0),
        "targets_succeeded": int(cov.get("targets_succeeded", 0) or 0),
        **extra,
    }
    with open(_round_summary_path(round_num), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[summary] round {round_num}: {payload}", flush=True)


async def _run_one_round(args: Namespace, round_num: int) -> int:
    """Returns papers_with_samples for the round (sanity-check signal)."""
    sub = Namespace(
        round=round_num,
        per_cell=args.per_cell,
        strata=args.strata,
        seed=args.seed,
        paper_concurrency=args.paper_concurrency,
        exclude_prior_rounds=True,
        auto_v2=False,
        min_tries=schema_v2_generator.DEFAULT_MIN_TRIES,
        success_floor=schema_v2_generator.DEFAULT_SUCCESS_FLOOR,
    )
    rc = await run_iteration._async_main(sub)
    if rc != 0:
        raise RuntimeError(f"run_iteration returned non-zero: {rc}")
    cov = _read_coverage_last_row() or {}
    return int(cov.get("papers_with_samples", 0) or 0)


async def _async_main(args: Namespace) -> int:
    ensure_phase7_dir()
    print(f"=== run_until_saturated start ===", flush=True)
    print(f"  start_round={args.start_round}  max_rounds={args.max_rounds}",
          flush=True)
    print(f"  per_cell={args.per_cell}  strata={args.strata}  "
          f"paper_concurrency={args.paper_concurrency}", flush=True)

    final_status = "INCOMPLETE"
    last_round = args.start_round - 1

    for r in range(args.start_round, args.start_round + args.max_rounds):
        print(f"\n══════ Round {r} ══════", flush=True)
        try:
            pws = await _run_one_round(args, r)
        except Exception as e:
            print(f"[FATAL] round {r} failed: {e}", flush=True)
            _write_round_summary(r, "FAILED", {"error": str(e)})
            final_status = "FAILED"
            last_round = r
            break

        last_round = r

        if pws == 0:
            print(f"[ABORT] round {r} papers_with_samples=0 — upstream sanity tripwire",
                  flush=True)
            _write_round_summary(r, "ABORTED_ZERO_SAMPLES", {})
            final_status = "ABORTED"
            break

        sat = saturation_check.check()
        if sat["status"] == "SATURATED":
            _write_round_summary(r, "SATURATED", {"saturation": sat})
            print(f"[STOP] saturated at round {r}: {sat['reason']}", flush=True)
            final_status = "SATURATED"
            break
        else:
            _write_round_summary(r, "CONTINUE", {"saturation": sat})

    if final_status == "SATURATED" and args.auto_v2:
        print(f"\n══════ Generating schema v2 ══════", flush=True)
        target_meta, alias_to_target = extraction_metrics.load_target_catalog()
        summary = schema_v2_generator.generate(args.min_tries, args.success_floor)
        for k, v in summary.items():
            print(f"  {k}: {v}", flush=True)

    if final_status == "INCOMPLETE":
        # Hit max_rounds without saturating
        final_status = "MAX_ROUNDS_REACHED"
        print(f"\n[STOP] reached max_rounds={args.max_rounds} without saturation",
              flush=True)

    print(f"\n=== run_until_saturated end: {final_status} (last_round={last_round}) ===",
          flush=True)
    return 0 if final_status in ("SATURATED", "MAX_ROUNDS_REACHED") else 1


def main() -> None:
    p = argparse.ArgumentParser(prog="run_until_saturated")
    p.add_argument("--start-round", type=int, default=2,
                   help="round number to start from (default: 2, leaving round 1 as dry-run)")
    p.add_argument("--max-rounds", type=int, default=4,
                   help="hard cap on number of rounds to run (default: 4)")
    p.add_argument("--per-cell", type=int, default=7)
    p.add_argument("--strata", default="env,section,era")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--paper-concurrency", type=int, default=8)
    p.add_argument("--auto-v2", action="store_true",
                   help="If saturated, automatically generate schema v2")
    p.add_argument("--min-tries", type=int,
                   default=schema_v2_generator.DEFAULT_MIN_TRIES)
    p.add_argument("--success-floor", type=float,
                   default=schema_v2_generator.DEFAULT_SUCCESS_FLOOR)
    args = p.parse_args()

    # Force line buffering for nohup-friendliness
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(line_buffering=True)

    sys.exit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
