"""End-to-end orchestrator for one phase7 round.

Pipeline:
  1. stratified_sampler.sample_round  → sampled_papers_round_NN.csv
  2. subset_input                     → round_NN_step5_input.json
  3. step5.orchestrator.main_async    → round_NN_step5_output.json
  4. extraction_metrics               → extraction_success_round_NN.csv,
                                        extraction_success_cumulative.csv,
                                        coverage_per_round.csv
  5. saturation_check                 → SATURATED / CONTINUE
  6. (optional, if --auto-v2 and saturated) schema_v2_generator
                                       → env6_extraction_targets_v2.json

CLI:
  python -m metaagent_run.steps.env_field_pipeline.phase7_validation.run_iteration \\
      --round 1 --per-cell 2 --strata env,section --paper-concurrency 8

Defaults work for the dry-run (round 1, 16 strata × 2 = 32 papers, env×section).
For the formal 336-paper run: --per-cell 7 --strata env,section,era.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from metaagent_run.steps.env_field_pipeline import config as ep_config
from metaagent_run.steps.env_field_pipeline.phase7_validation import (
    ACCESSION_LIST_PATH,
    PHASE7_DIR,
    ensure_phase7_dir,
)
from metaagent_run.steps.env_field_pipeline.phase7_validation import (
    extraction_metrics,
    saturation_check,
    schema_v2_generator,
    stratified_sampler,
    subset_input,
)
from metaagent_run.steps.step5 import config as step5_config
from metaagent_run.steps.step5 import orchestrator as step5_orch

ENV6_PATH = ep_config.OUTPUT_DIR / "env6_extraction_targets.json"
ENV_TAG_PATH = ep_config.PROJECT_ROOT_DIR / "env_tag_v2_step4a_env_tag_output.json"


def _step1_sample(args: argparse.Namespace) -> Path:
    print(f"\n[1/5] Sampling round {args.round} ...", flush=True)
    pool = stratified_sampler.load_paper_pool_index()
    strata_dims = [s.strip() for s in args.strata.split(",") if s.strip()]
    excluded = (
        stratified_sampler.load_prior_round_pmids(args.round - 1)
        if args.exclude_prior_rounds and args.round > 1 else set()
    )
    samples = stratified_sampler.sample_round(
        pool_index=pool,
        per_cell=args.per_cell,
        strata_dims=strata_dims,
        excluded_pmids=excluded,
        seed=args.seed + args.round,
    )
    csv_path = stratified_sampler.save_round(samples, args.round)
    print(f"      {len(samples)} unique pmids → {csv_path.name}")
    return csv_path


def _step2_subset(args: argparse.Namespace) -> Path:
    print(f"\n[2/5] Subsetting step5 input ...", flush=True)
    pmids = subset_input.load_round_pmids(args.round)
    out = PHASE7_DIR / f"round_{args.round:02d}_step5_input.json"
    subset_input.subset_input(pmids, output_path=out)
    return out


async def _step3_step5(args: argparse.Namespace, input_path: Path) -> Path:
    print(f"\n[3/5] Running step5 (paper_concurrency={args.paper_concurrency}) ...",
          flush=True)
    output_path = PHASE7_DIR / f"round_{args.round:02d}_step5_output.json"
    cfg = step5_config.load_runtime_config(
        input_file=str(input_path),
        output_file=str(output_path),
        env_extraction_targets_file=str(ENV6_PATH),
        env_tag_file=str(ENV_TAG_PATH),
        accession_list_file=str(ACCESSION_LIST_PATH),
    )
    await step5_orch.main_async(
        input_file=str(input_path),
        output_file=str(output_path),
        paper_concurrency=args.paper_concurrency,
        runtime_config=cfg,
    )
    return output_path


def _step4_metrics(args: argparse.Namespace) -> None:
    print(f"\n[4/5] Computing metrics ...", flush=True)
    target_meta, alias_to_target = extraction_metrics.load_target_catalog()
    print(f"      {len(target_meta)} unique targets, "
          f"{len(alias_to_target)} alias entries")

    t_r, s_r, p_r = extraction_metrics.aggregate(
        [args.round], target_meta, alias_to_target,
    )
    out_r = PHASE7_DIR / f"extraction_success_round_{args.round:02d}.csv"
    extraction_metrics.write_success_csv(out_r, t_r, s_r, target_meta)
    print(f"      round {args.round}: {len(p_r)} papers, "
          f"{len(t_r)} targets tried, {len(s_r)} succeeded → {out_r.name}")

    rounds = extraction_metrics.discover_rounds()
    t_c, s_c, _ = extraction_metrics.aggregate(rounds, target_meta, alias_to_target)
    out_c = PHASE7_DIR / "extraction_success_cumulative.csv"
    extraction_metrics.write_success_csv(out_c, t_c, s_c, target_meta)
    print(f"      cumulative: {len(t_c)} tried, {len(s_c)} succeeded → {out_c.name}")

    cov_path = PHASE7_DIR / "coverage_per_round.csv"
    extraction_metrics.write_coverage_per_round(cov_path, target_meta, alias_to_target)
    print(f"      → {cov_path.name}")


def _step5_saturation_and_v2(args: argparse.Namespace) -> str:
    print(f"\n[5/5] Saturation check ...", flush=True)
    result = saturation_check.check()
    print(f"      STATUS: {result['status']} — {result['reason']}")
    if "tail" in result:
        for x in result["tail"]:
            mark = "✓" if x["below_threshold"] else "✗"
            print(f"        round {x['round']}: "
                  f"Δtried={x['rel_delta_tried']:.2%} "
                  f"Δsucc={x['rel_delta_succ']:.2%} {mark}")
    if result["status"] == "SATURATED" and args.auto_v2:
        print("\n[bonus] Generating schema v2 ...", flush=True)
        summary = schema_v2_generator.generate(args.min_tries, args.success_floor)
        for k, v in summary.items():
            print(f"        {k}: {v}")
    return result["status"]


async def _async_main(args: argparse.Namespace) -> int:
    ensure_phase7_dir()
    _step1_sample(args)
    input_path = _step2_subset(args)
    await _step3_step5(args, input_path)
    _step4_metrics(args)
    status = _step5_saturation_and_v2(args)
    return 0 if status in ("SATURATED", "CONTINUE") else 1


def main() -> None:
    p = argparse.ArgumentParser(prog="run_iteration")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--per-cell", type=int, default=2)
    p.add_argument("--strata", default="env,section",
                   help="comma-separated subset of {env,section,era}; "
                        "dry-run uses 'env,section', formal uses 'env,section,era'")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--paper-concurrency", type=int, default=8)
    p.add_argument("--exclude-prior-rounds", action="store_true",
                   help="Exclude pmids sampled in rounds < this one")
    p.add_argument("--auto-v2", action="store_true",
                   help="If saturated, also generate schema v2")
    p.add_argument("--min-tries", type=int,
                   default=schema_v2_generator.DEFAULT_MIN_TRIES)
    p.add_argument("--success-floor", type=float,
                   default=schema_v2_generator.DEFAULT_SUCCESS_FLOOR)
    args = p.parse_args()

    sys.exit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
