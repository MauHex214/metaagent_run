"""Target × env stratified sampler — guarantees per-target evidence.

Design rationale (per design discussion):
  Random env×section×era stratified sampling estimates
    P(extract T) = P(mention T) × P(extract T | mention T)
  which conflates two probabilities and chronically undersamples rare targets
  (47/158 env6 targets had n_tries=0 even after 1178 random papers).

  Target-stratified sampling estimates the conditional we actually care about:
    P(extract T | mention T)
  by sampling papers conditional on "they mention T". Each target gets at
  most N tries (limited by paper pool availability), so v2 schema kept/
  pruned decisions have meaningful evidence per target.

Algorithm:
  1. build_target_paper_index: scan step2 once; for each env6 canonical T
     (using v1b aliases), record set of pmids in the pool whose
     metadata_keys_found ∋ alias(T).
  2. For each (T, env) where T is listed in env6 for env AND pmid is in
     env-tagged paper_pool: sample min(N, |candidates|) papers.
  3. Global dedup. Save sampled CSV.

Cache: target_paper_index built once and saved (re-uses across runs unless
env6_v1b changes).

CLI:
  # Build cached index (one-time, ~30s)
  python -m metaagent_run.steps.env_field_pipeline.phase7_validation.target_stratified_sampler build-index

  # Sample
  python -m metaagent_run.steps.env_field_pipeline.phase7_validation.target_stratified_sampler sample \
      --per-cell 10 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from metaagent_run.steps.env_field_pipeline import config as ep_config
from metaagent_run.steps.env_field_pipeline.phase7_validation import (
    PHASE7_DIR,
    TARGET_ENVS,
    ensure_phase7_dir,
)
from metaagent_run.steps.env_field_pipeline.phase7_validation.stratified_sampler import (
    load_paper_pool_index,
)

ENV6_V1B_PATH = ep_config.OUTPUT_DIR / "env6_extraction_targets_v1b.json"
TARGET_INDEX_PATH = PHASE7_DIR / "target_paper_index.json"
SAMPLED_CSV = PHASE7_DIR / "sampled_papers_target_stratified.csv"
DEFAULT_PER_CELL = 10


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower().strip()).strip("_")


# ─── Index build ────────────────────────────────────────────────────────

def _load_v1b_aliases() -> dict[str, dict]:
    """Returns target_name → {envs: set, aliases_norm: set}."""
    with open(ENV6_V1B_PATH, "r", encoding="utf-8") as f:
        v1b = json.load(f)
    out: dict[str, dict] = {}
    for env_name, env_block in v1b.get("per_environment", {}).items():
        for fdef in env_block.get("fields", []):
            name = fdef["field"]
            entry = out.setdefault(name, {
                "envs": set(),
                "aliases_norm": set(),
                "tier": fdef.get("tier"),
                "subtype": fdef.get("subtype"),
            })
            entry["envs"].add(env_name)
            entry["aliases_norm"].add(_norm(name))
            for a in fdef.get("aliases") or []:
                na = _norm(a)
                if na:
                    entry["aliases_norm"].add(na)
    return out


def build_target_paper_index() -> dict:
    """Scan step2 once; build target → list of pmids whose any metadata_key
    normalizes to an alias of the target. Constrained to pmids in the
    paper pool (target_env ∩ has_metadata ∩ has_year ∩ has_acc)."""
    ensure_phase7_dir()
    print("Building target × paper index ...", flush=True)
    pool = load_paper_pool_index()
    pool_pmids = set(pool["papers"].keys())
    print(f"  paper pool: {len(pool_pmids)} pmids", flush=True)

    target_meta = _load_v1b_aliases()
    print(f"  env6_v1b: {len(target_meta)} unique canonicals", flush=True)

    # Inverted index: alias_norm → set(target_names)
    alias_to_targets: dict[str, set[str]] = defaultdict(set)
    for tname, meta in target_meta.items():
        for an in meta["aliases_norm"]:
            alias_to_targets[an].add(tname)

    print(f"  loading step2 ({ep_config.STEP2_INPUT.name}) ...", flush=True)
    with open(ep_config.STEP2_INPUT, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"  {len(records):,} records", flush=True)

    target_pmids: dict[str, set[str]] = defaultdict(set)
    for r in records:
        pmid = str(r.get("pmid", ""))
        if pmid not in pool_pmids:
            continue
        keys = r.get("metadata_keys_found")
        if not isinstance(keys, list):
            continue
        for k in keys:
            nk = _norm(k)
            if not nk:
                continue
            for tname in alias_to_targets.get(nk, ()):
                target_pmids[tname].add(pmid)

    # Stats
    tries_dist = Counter()
    for t, pmids in target_pmids.items():
        n = len(pmids)
        if n == 0: tries_dist["0"] += 1
        elif n < 10: tries_dist["1-9"] += 1
        elif n < 50: tries_dist["10-49"] += 1
        elif n < 200: tries_dist["50-199"] += 1
        else: tries_dist["200+"] += 1
    # Targets that never matched any pool paper
    for t in target_meta:
        if t not in target_pmids:
            tries_dist["0"] += 1

    print(f"\nPool coverage per target:")
    for k in ["0", "1-9", "10-49", "50-199", "200+"]:
        print(f"  {k:<8} {tries_dist.get(k, 0):>4} targets")

    out = {
        "version": 1,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "v1b_path": str(ENV6_V1B_PATH),
        "n_targets": len(target_meta),
        "n_targets_with_at_least_1_pool_paper": len(target_pmids),
        "target_pmids": {t: sorted(pmids) for t, pmids in target_pmids.items()},
        "target_envs": {t: sorted(meta["envs"]) for t, meta in target_meta.items()},
        "target_tier": {t: meta.get("tier") for t, meta in target_meta.items()},
        "target_subtype": {t: meta.get("subtype") for t, meta in target_meta.items()},
    }
    with open(TARGET_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"\nWrote {TARGET_INDEX_PATH}", flush=True)
    return out


def load_target_paper_index() -> dict:
    if not TARGET_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{TARGET_INDEX_PATH} not found. Run `build-index` subcommand first."
        )
    with open(TARGET_INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Sampling ───────────────────────────────────────────────────────────

def sample(per_cell: int, seed: int) -> list[dict]:
    """For each (target T, env E) where T applies in E and pool has papers:
      candidates = target_paper_index[T] ∩ env_papers[E]
      pick min(per_cell, |candidates|) random papers (seeded).

    Globally dedup. Each unique pmid carries a 'picked_in_cells' list of
    every (T, E) cell it satisfied for. Era + section_tier carried as
    metadata only (not stratification dimensions, per design discussion)."""
    rng = random.Random(seed)
    pool = load_paper_pool_index()
    target_idx = load_target_paper_index()
    target_pmids = target_idx["target_pmids"]
    target_envs = target_idx["target_envs"]

    # env → set(pmids) lookup
    env_pmids: dict[str, set[str]] = defaultdict(set)
    for pmid, info in pool["papers"].items():
        for e in info["envs"]:
            env_pmids[e].add(pmid)

    pmid_picked_cells: dict[str, list[str]] = defaultdict(list)
    cell_stats: dict[str, dict] = {}

    for tname in sorted(target_pmids):
        envs = target_envs.get(tname, [])
        for e in sorted(envs):
            cell_label = f"{tname}|{e}"
            t_pool = set(target_pmids[tname])
            e_pool = env_pmids.get(e, set())
            candidates = sorted(t_pool & e_pool)
            n_avail = len(candidates)
            n_pick = min(per_cell, n_avail)
            cell_stats[cell_label] = {"avail": n_avail, "picked": n_pick}
            if n_pick == 0:
                continue
            rng.shuffle(candidates)
            for pmid in candidates[:n_pick]:
                pmid_picked_cells[pmid].append(cell_label)

    samples = []
    for pmid in sorted(pmid_picked_cells):
        info = pool["papers"][pmid]
        samples.append({
            "pmid": pmid,
            "year": info["year"],
            "era": info["era"],
            "envs": info["envs"],
            "section_tiers": info["section_tiers"],
            "picked_in_cells": pmid_picked_cells[pmid],
            "n_cells": len(pmid_picked_cells[pmid]),
        })

    # Cell coverage stats
    cells_total = len(cell_stats)
    cells_filled = sum(1 for c in cell_stats.values() if c["picked"] > 0)
    cells_under = sum(1 for c in cell_stats.values() if c["picked"] < per_cell)
    print(f"\nSampling summary:")
    print(f"  per_cell quota: {per_cell}")
    print(f"  total (target × env) cells: {cells_total}")
    print(f"  filled cells (≥1 paper):    {cells_filled}")
    print(f"  cells below quota:          {cells_under}  "
          f"(can't reach {per_cell} due to limited candidates)")
    print(f"  cells with 0 candidates:    {cells_total - cells_filled}")
    print(f"  unique pmids sampled:       {len(samples)}")
    print(f"  total stratum-picks:        {sum(len(s['picked_in_cells']) for s in samples)}")
    return samples, cell_stats


def save(samples: list[dict], cell_stats: dict[str, dict],
         path: Path = SAMPLED_CSV) -> Path:
    ensure_phase7_dir()
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pmid", "year", "era", "envs", "section_tiers",
                    "n_cells", "picked_in_cells"])
        for s in samples:
            w.writerow([
                s["pmid"], s["year"], s["era"],
                ";".join(s["envs"]), ";".join(s["section_tiers"]),
                s["n_cells"],
                ";".join(s["picked_in_cells"]),
            ])
    # Cell stats sidecar (for debugging cell coverage)
    cell_stats_path = path.with_name(path.stem + "_cell_stats.json")
    with open(cell_stats_path, "w", encoding="utf-8") as f:
        json.dump(cell_stats, f, ensure_ascii=False, indent=2)
    print(f"  written: {path}")
    print(f"  cell stats: {cell_stats_path}")
    return path


# ─── CLI ────────────────────────────────────────────────────────────────

def _cli_build_index(_args: argparse.Namespace) -> None:
    build_target_paper_index()


def _cli_sample(args: argparse.Namespace) -> None:
    samples, cell_stats = sample(per_cell=args.per_cell, seed=args.seed)
    save(samples, cell_stats)


def main() -> None:
    p = argparse.ArgumentParser(prog="target_stratified_sampler")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_build = sub.add_parser("build-index")
    sp_build.set_defaults(func=_cli_build_index)

    sp_samp = sub.add_parser("sample")
    sp_samp.add_argument("--per-cell", type=int, default=DEFAULT_PER_CELL)
    sp_samp.add_argument("--seed", type=int, default=42)
    sp_samp.set_defaults(func=_cli_sample)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
