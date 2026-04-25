"""End-to-end target-stratified phase7 validation.

One-shot orchestrator (no rounds, no saturation iteration) — target-stratified
sampling guarantees per-target evidence in a single pass.

Pipeline:
  1. (assumed) inject_raw_key_expansion already produced env6_v1b
  2. (assumed) target_stratified_sampler already produced
     sampled_papers_target_stratified.csv
  3. subset_input → target_stratified_step5_input.json
  4. step5 with --env-targets env6_v1b → target_stratified_step5_output.json
  5. extraction_metrics (env6_v1b catalog) → extraction_success_cumulative.csv
  6. schema_v2_generator (env6_v1b base) → env6_extraction_targets_v2.json
                                          + env6_v1_to_v2_diff.csv

CLI:
  nohup python -m metaagent_run.steps.env_field_pipeline.phase7_validation.run_target_validation \\
      --paper-concurrency 16 > target_validation.log 2>&1 &
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
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
    schema_v2_generator,
    subset_input,
    target_stratified_sampler,
)
from metaagent_run.steps.step5 import config as step5_config
from metaagent_run.steps.step5 import orchestrator as step5_orch

ENV_TAG_PATH = ep_config.PROJECT_ROOT_DIR / "env_tag_v2_step4a_env_tag_output.json"
SAMPLED_CSV = PHASE7_DIR / "sampled_papers_target_stratified.csv"
STEP5_INPUT = PHASE7_DIR / "target_stratified_step5_input.json"
STEP5_OUTPUT = PHASE7_DIR / "target_stratified_step5_output.json"
SUCCESS_CSV = PHASE7_DIR / "extraction_success_cumulative.csv"


def _load_sampled_pmids() -> set[str]:
    if not SAMPLED_CSV.exists():
        raise FileNotFoundError(
            f"{SAMPLED_CSV} missing. Run target_stratified_sampler.sample first."
        )
    with open(SAMPLED_CSV, "r", encoding="utf-8") as f:
        return {row["pmid"] for row in csv.DictReader(f)}


def _step1_subset() -> Path:
    print("\n[1/4] Subsetting step5 input ...", flush=True)
    pmids = _load_sampled_pmids()
    print(f"      {len(pmids)} unique pmids from {SAMPLED_CSV.name}")
    subset_input.subset_input(pmids, output_path=STEP5_INPUT)
    return STEP5_INPUT


async def _step2_step5(args: argparse.Namespace, input_path: Path) -> Path:
    print(f"\n[2/4] Running step5 (paper_concurrency={args.paper_concurrency}) ...",
          flush=True)
    cfg = step5_config.load_runtime_config(
        input_file=str(input_path),
        output_file=str(STEP5_OUTPUT),
        env_extraction_targets_file=str(extraction_metrics.ENV6_PATH),
        env_tag_file=str(ENV_TAG_PATH),
        accession_list_file=str(ACCESSION_LIST_PATH),
    )
    await step5_orch.main_async(
        input_file=str(input_path),
        output_file=str(STEP5_OUTPUT),
        paper_concurrency=args.paper_concurrency,
        runtime_config=cfg,
    )
    return STEP5_OUTPUT


def _step3_metrics() -> None:
    """Adapt extraction_metrics aggregate() to the single target-stratified
    output (no per-round splits). Builds extraction_success_cumulative.csv
    from STEP5_OUTPUT directly."""
    print(f"\n[3/4] Computing metrics ...", flush=True)
    print(f"      env6 catalog: {extraction_metrics.ENV6_PATH}")
    target_meta, alias_to_target = extraction_metrics.load_target_catalog()
    print(f"      {len(target_meta)} unique targets, "
          f"{len(alias_to_target)} alias entries")

    # Reuse extraction_metrics primitives but with custom step5 output path.
    # aggregate() iterates per-round files; we shim by writing a fake
    # round_99 pair pointing at our single output, then calling aggregate.
    # Simpler: replicate the logic inline.
    sampled_pmids = _load_sampled_pmids()
    pmid_step2_keys = extraction_metrics.load_pmid_step2_keys(
        pool_index=None, pmids_filter=sampled_pmids,
    )

    from collections import defaultdict
    tries: dict[str, set[str]] = defaultdict(set)
    success: dict[str, set[str]] = defaultdict(set)

    for pmid, keys in pmid_step2_keys.items():
        for k in keys:
            t = alias_to_target.get(k)
            if t:
                tries[t].add(pmid)

    with open(STEP5_OUTPUT, "r", encoding="utf-8") as f:
        papers = json.load(f)
    n_papers_with_samples = 0
    for paper in papers:
        if paper.get("samples"):
            n_papers_with_samples += 1
        pmid = paper.get("pmid")
        for sample in paper.get("samples", []):
            for m in sample.get("metadata", []) or []:
                raw = extraction_metrics._normalize_key(m.get("raw_field", ""))
                if not raw:
                    continue
                t = alias_to_target.get(raw)
                if t:
                    success[t].add(pmid)

    extraction_metrics.write_success_csv(SUCCESS_CSV, tries, success, target_meta)
    print(f"      papers={len(papers)}, with_samples={n_papers_with_samples} "
          f"({100*n_papers_with_samples/max(1,len(papers)):.1f}%)")
    print(f"      targets tried={len(tries)}, succeeded={len(success)} → {SUCCESS_CSV.name}")


def _step4_v2(args: argparse.Namespace) -> None:
    print(f"\n[4/4] Generating schema v2 ...", flush=True)
    print(f"      base: {schema_v2_generator.ENV6_V1_PATH}")
    summary = schema_v2_generator.generate(args.min_tries, args.success_floor)
    for k, v in summary.items():
        print(f"      {k}: {v}")


async def _async_main(args: argparse.Namespace) -> int:
    ensure_phase7_dir()
    print(f"=== run_target_validation ===")
    print(f"  env6 catalog: {extraction_metrics.ENV6_PATH}")
    print(f"  paper sample: {SAMPLED_CSV}")
    print(f"  paper_concurrency: {args.paper_concurrency}")
    print(f"  prune rule: n_tries >= {args.min_tries} AND success_rate < {args.success_floor}")

    input_path = _step1_subset()
    await _step2_step5(args, input_path)
    _step3_metrics()
    _step4_v2(args)

    print(f"\n=== done ===")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="run_target_validation")
    p.add_argument("--paper-concurrency", type=int, default=16)
    p.add_argument("--min-tries", type=int,
                   default=schema_v2_generator.DEFAULT_MIN_TRIES)
    p.add_argument("--success-floor", type=float,
                   default=schema_v2_generator.DEFAULT_SUCCESS_FLOOR)
    args = p.parse_args()

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(line_buffering=True)

    sys.exit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
