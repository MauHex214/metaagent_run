"""Subset target_env_v1_relation_input.json to a given pmid set.

step5 has no built-in pmid filter, so for phase7 dry-runs and per-round
sampling we materialize a small subset input. The subset is segment-level
JSON in the same shape step5 expects.

CLI:
  python -m metaagent_run.steps.env_field_pipeline.phase7_validation.subset_input \\
      --round 1 \\
      [--input target_env_v1_relation_input.json] \\
      [--output env_field_pipeline_output/phase7_validation/round_01_step5_input.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from metaagent_run.steps.env_field_pipeline import config as ep_config
from metaagent_run.steps.env_field_pipeline.phase7_validation import (
    PHASE7_DIR,
    ensure_phase7_dir,
)

DEFAULT_INPUT = ep_config.PROJECT_ROOT_DIR / "target_env_v1_relation_input.json"


def load_round_pmids(round_num: int) -> set[str]:
    path = PHASE7_DIR / f"sampled_papers_round_{round_num:02d}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run sampler first.")
    with open(path, "r", encoding="utf-8") as f:
        return {row["pmid"] for row in csv.DictReader(f)}


def subset_input(
    pmids: set[str],
    input_path: Path = DEFAULT_INPUT,
    output_path: Path | None = None,
) -> Path:
    ensure_phase7_dir()
    if output_path is None:
        output_path = PHASE7_DIR / "subset_step5_input.json"

    print(f"Loading {input_path} ({input_path.stat().st_size / 1e9:.1f} GB) ...",
          flush=True)
    t0 = time.time()
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"  loaded {len(records):,} records in {time.time() - t0:.1f}s",
          flush=True)

    filtered = [r for r in records if r.get("pmid") in pmids]
    print(f"  filtered to {len(filtered):,} records "
          f"({len(filtered) / len(records) * 100:.2f}%) for {len(pmids)} pmids",
          flush=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False)
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)",
          flush=True)
    return output_path


def main() -> None:
    p = argparse.ArgumentParser(prog="subset_input")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=None,
                   help="default: phase7_validation/round_{NN}_step5_input.json")
    args = p.parse_args()

    pmids = load_round_pmids(args.round)
    out = args.output or PHASE7_DIR / f"round_{args.round:02d}_step5_input.json"
    subset_input(pmids, input_path=args.input, output_path=out)


if __name__ == "__main__":
    main()
