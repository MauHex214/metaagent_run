"""Stratified paper sampler for phase7 extraction validation.

Two stages:
  1. build-index: scan paper_env_map / pmid_year / step2 once, persist
     paper_pool_index.json with each pmid's envs / era / section_tiers.
  2. sample: pick min(per_cell, cell_size) papers per (env × section × era)
     stratum, dedup globally by pmid, exclude prior-round pmids.

CLI:
  python -m metaagent_run.steps.env_field_pipeline.phase7_validation.stratified_sampler build-index
  python -m metaagent_run.steps.env_field_pipeline.phase7_validation.stratified_sampler sample \\
      --round 1 --per-cell 2 --strata env,section --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

from metaagent_run.steps.env_field_pipeline import config as ep_config
from metaagent_run.steps.env_field_pipeline.phase7_validation import (
    ACCESSION_LIST_PATH,
    ERAS,
    PHASE7_DIR,
    SECTION_TIER_MAP,
    TARGET_ENVS,
    TIERS,
    ensure_phase7_dir,
    to_era,
)

POOL_INDEX_PATH = PHASE7_DIR / "paper_pool_index.json"
PMID_YEAR_PATH = ep_config.PROJECT_ROOT_DIR / "paper_down" / "pmid_year.txt"


# ─── Index build ────────────────────────────────────────────────────────

def _load_pmid_year() -> dict[str, int]:
    out: dict[str, int] = {}
    with open(PMID_YEAR_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            try:
                out[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return out


def _load_paper_envs() -> dict[str, list[str]]:
    with open(ep_config.PAPER_ENV_MAP, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        pmid: [e for e in envs if e in TARGET_ENVS]
        for pmid, envs in raw.items()
        if any(e in TARGET_ENVS for e in envs)
    }


def _load_accession_pmids() -> set[str]:
    """Set of pmids that have ≥1 verified accession in pmid_run-accession*.list.

    Last column is pmid; can be semicolon-separated when multiple papers share
    one accession (~23% of rows have multi-pmid)."""
    out: set[str] = set()
    with open(ACCESSION_LIST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue
            for p in parts[-1].split(";"):
                p = p.strip()
                if p:
                    out.add(p)
    return out


def _load_pmid_section_tiers() -> dict[str, set[str]]:
    """For each pmid, the set of normalized section tiers that contain at
    least one segment with metadata_keys_found ≥ 1 (i.e. step5 will actually
    invoke LLM on something in that tier)."""
    print(f"  loading {ep_config.STEP2_INPUT.name} ...", flush=True)
    with open(ep_config.STEP2_INPUT, "r", encoding="utf-8") as f:
        records = json.load(f)
    out: dict[str, set[str]] = defaultdict(set)
    for r in records:
        keys = r.get("metadata_keys_found")
        if not isinstance(keys, list) or len(keys) < 1:
            continue
        tier = SECTION_TIER_MAP.get(r.get("section_type", ""))
        if tier:
            out[r["pmid"]].add(tier)
    return out


def build_paper_pool_index() -> dict:
    """Pool = pmids in (target env ∩ metadata-bearing section ∩ has year ∩
    has ≥1 verified accession). The accession constraint is critical: step5's
    Phase A (Identity Resolution) gates on verified accessions, so papers
    without them produce 0 extracted samples and contribute no signal."""
    ensure_phase7_dir()
    print("Building paper pool index ...", flush=True)
    print("  loading paper_env_map ...", flush=True)
    paper_envs = _load_paper_envs()
    print("  loading pmid_year ...", flush=True)
    pmid_year = _load_pmid_year()
    print("  loading accession list ...", flush=True)
    acc_pmids = _load_accession_pmids()
    pmid_tiers = _load_pmid_section_tiers()

    papers: dict[str, dict] = {}
    for pmid, tiers in pmid_tiers.items():
        if pmid not in acc_pmids:
            continue
        envs = paper_envs.get(pmid)
        if not envs:
            continue
        year = pmid_year.get(pmid)
        if year is None:
            continue
        papers[pmid] = {
            "envs": envs,
            "year": year,
            "era": to_era(year),
            "section_tiers": sorted(tiers),
        }

    n_full_cells = len(TARGET_ENVS) * len(TIERS) * len(ERAS)
    cell_sizes_full: Counter = Counter()  # env × section × era
    cell_sizes_envxsec: Counter = Counter()  # env × section (dry-run)
    for pmid, info in papers.items():
        for e in info["envs"]:
            for t in info["section_tiers"]:
                cell_sizes_full[f"{e}|{t}|{info['era']}"] += 1
                cell_sizes_envxsec[f"{e}|{t}"] += 1

    index = {
        "version": 2,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_papers": len(papers),
        "constraint": "target_env ∩ metadata-bearing ∩ has_year ∩ has_accession",
        "eras": list(ERAS),
        "papers": papers,
        "cell_sizes_full": dict(cell_sizes_full),
        "cell_sizes_envxsec": dict(cell_sizes_envxsec),
    }
    with open(POOL_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"Wrote {POOL_INDEX_PATH} ({len(papers)} pmids, "
          f"{len(cell_sizes_full)}/{n_full_cells} full cells, "
          f"{len(cell_sizes_envxsec)}/{len(TARGET_ENVS)*len(TIERS)} env×section cells)",
          flush=True)
    return index


def load_paper_pool_index() -> dict:
    if not POOL_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{POOL_INDEX_PATH} not found. Run `build-index` subcommand first."
        )
    with open(POOL_INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Sampling ───────────────────────────────────────────────────────────

def _enumerate_strata(strata_dims: list[str]) -> list[tuple]:
    """Enumerate cell tuples for the given stratification dimensions."""
    axes = []
    if "env" in strata_dims:
        axes.append(TARGET_ENVS)
    if "section" in strata_dims:
        axes.append(TIERS)
    if "era" in strata_dims:
        axes.append(ERAS)
    return list(product(*axes))


def _paper_in_cell(info: dict, cell: tuple, strata_dims: list[str]) -> bool:
    """Does paper `info` qualify for `cell` under `strata_dims`?
    A paper qualifies for an env-stratified cell if it maps to that env;
    for a section-stratified cell if it has a passage of that tier;
    for an era-stratified cell if its era matches."""
    i = 0
    if "env" in strata_dims:
        if cell[i] not in info["envs"]:
            return False
        i += 1
    if "section" in strata_dims:
        if cell[i] not in info["section_tiers"]:
            return False
        i += 1
    if "era" in strata_dims:
        if cell[i] != info["era"]:
            return False
        i += 1
    return True


def sample_round(
    pool_index: dict,
    per_cell: int,
    strata_dims: list[str],
    excluded_pmids: set[str] | None,
    seed: int,
) -> list[dict]:
    """Pick papers per cell, dedup globally by pmid.

    Returns list of dicts: {pmid, year, era, envs, section_tiers, picked_in_cells}
    where picked_in_cells lists every cell the paper was selected for (a paper
    can be picked for multiple cells if it qualifies)."""
    excluded = excluded_pmids or set()
    rng = random.Random(seed)

    cells = _enumerate_strata(strata_dims)
    pmid_picked_cells: dict[str, list[str]] = defaultdict(list)

    for cell in cells:
        candidates = [
            pmid for pmid, info in pool_index["papers"].items()
            if pmid not in excluded and _paper_in_cell(info, cell, strata_dims)
        ]
        candidates.sort()  # deterministic order before shuffle
        rng.shuffle(candidates)
        picked = candidates[: min(per_cell, len(candidates))]
        cell_label = "|".join(cell)
        for pmid in picked:
            pmid_picked_cells[pmid].append(cell_label)

    samples = []
    for pmid in sorted(pmid_picked_cells):
        info = pool_index["papers"][pmid]
        samples.append({
            "pmid": pmid,
            "year": info["year"],
            "era": info["era"],
            "envs": info["envs"],
            "section_tiers": info["section_tiers"],
            "picked_in_cells": pmid_picked_cells[pmid],
        })
    return samples


def save_round(samples: list[dict], round_num: int) -> Path:
    ensure_phase7_dir()
    path = PHASE7_DIR / f"sampled_papers_round_{round_num:02d}.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pmid", "year", "era", "envs", "section_tiers", "picked_in_cells"])
        for s in samples:
            w.writerow([
                s["pmid"], s["year"], s["era"],
                ";".join(s["envs"]),
                ";".join(s["section_tiers"]),
                ";".join(s["picked_in_cells"]),
            ])
    return path


def load_prior_round_pmids(up_to_round: int) -> set[str]:
    """Union of pmids sampled in rounds 1 .. up_to_round (inclusive)."""
    out: set[str] = set()
    for r in range(1, up_to_round + 1):
        path = PHASE7_DIR / f"sampled_papers_round_{r:02d}.csv"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.add(row["pmid"])
    return out


# ─── CLI ────────────────────────────────────────────────────────────────

def _cli_build_index(_args: argparse.Namespace) -> None:
    build_paper_pool_index()


def _cli_sample(args: argparse.Namespace) -> None:
    pool = load_paper_pool_index()
    strata_dims = [s.strip() for s in args.strata.split(",") if s.strip()]
    for d in strata_dims:
        if d not in {"env", "section", "era"}:
            print(f"ERROR: unknown stratum dim '{d}'", file=sys.stderr)
            sys.exit(2)

    excluded = (
        load_prior_round_pmids(args.round - 1)
        if args.exclude_prior_rounds and args.round > 1 else set()
    )
    if excluded:
        print(f"Excluding {len(excluded)} pmids from prior rounds", flush=True)

    samples = sample_round(
        pool_index=pool,
        per_cell=args.per_cell,
        strata_dims=strata_dims,
        excluded_pmids=excluded,
        seed=args.seed + args.round,  # different seed per round
    )
    path = save_round(samples, args.round)

    # Quick stats
    cells_filled = Counter()
    for s in samples:
        for c in s["picked_in_cells"]:
            cells_filled[c] += 1
    expected_cells = len(_enumerate_strata(strata_dims))
    print(f"\nRound {args.round}: {len(samples)} unique pmids", flush=True)
    print(f"  strata={strata_dims}, per_cell={args.per_cell}, "
          f"expected cells={expected_cells}, filled cells={len(cells_filled)}")
    print(f"  total stratum-picks={sum(cells_filled.values())} "
          f"(< unique × cells_per_pmid because of dedup)")
    print(f"  written: {path}")


def main() -> None:
    p = argparse.ArgumentParser(prog="stratified_sampler")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_build = sub.add_parser("build-index")
    sp_build.set_defaults(func=_cli_build_index)

    sp_samp = sub.add_parser("sample")
    sp_samp.add_argument("--round", type=int, required=True)
    sp_samp.add_argument("--per-cell", type=int, default=5)
    sp_samp.add_argument("--strata", default="env,section,era",
                         help="comma-separated subset of {env,section,era}")
    sp_samp.add_argument("--seed", type=int, default=42)
    sp_samp.add_argument("--exclude-prior-rounds", action="store_true",
                         help="exclude pmids already sampled in rounds < this one")
    sp_samp.set_defaults(func=_cli_sample)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
