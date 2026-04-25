"""Section-level target × env stratified sampler.

For each (target T, env E) cell:
  candidates = target_section_index[T] ∩ {sections in env-tagged papers}
  sampled = min(N, |candidates|)

Global dedup by (pmid, section_type, section_index). Each unique section
carries the set of (T, E) cells it satisfies, so per-section LLM evaluation
can batch all candidate targets for that section in a single call.

Outputs:
  sampled_sections.csv with columns
    pmid, section_type, section_index, env, candidate_targets (semicolon-joined)

CLI:
  python -m metaagent_run.steps.env_field_pipeline.phase7_validation.section_sampler \\
      --per-cell 10 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from metaagent_run.steps.env_field_pipeline.phase7_validation import (
    PHASE7_DIR,
    ensure_phase7_dir,
)
from metaagent_run.steps.env_field_pipeline.phase7_validation.section_indexer import (
    BROAD_POOL_PATH,
    TARGET_SECTION_INDEX_PATH,
)

SAMPLED_PATH = PHASE7_DIR / "sampled_sections.csv"
CELL_STATS_PATH = PHASE7_DIR / "sampled_sections_cell_stats.json"
DEFAULT_PER_CELL = 10


def sample(per_cell: int, seed: int):
    print(f"Loading broad pool + target_section_index ...", flush=True)
    pool = json.load(open(BROAD_POOL_PATH, "r", encoding="utf-8"))
    idx = json.load(open(TARGET_SECTION_INDEX_PATH, "r", encoding="utf-8"))

    pool_papers = pool["papers"]   # pmid → {envs, year, era}
    target_sections = idx["target_sections"]
    target_envs = idx["target_envs"]

    # env → set of pmids in pool
    env_pmids: dict[str, set[str]] = defaultdict(set)
    for pmid, info in pool_papers.items():
        for e in info["envs"]:
            env_pmids[e].add(pmid)

    rng = random.Random(seed)
    # (pmid, st, idx) → set of cells it was picked for
    section_cells: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    cell_stats: dict[str, dict] = {}

    for tname in sorted(target_sections):
        envs = target_envs.get(tname, [])
        sections_for_t = target_sections[tname]   # list of [pmid, st, idx]
        for e in sorted(envs):
            cell_label = f"{tname}|{e}"
            e_pmids = env_pmids.get(e, set())
            candidates = [tuple(s) for s in sections_for_t if s[0] in e_pmids]
            n_avail = len(candidates)
            n_pick = min(per_cell, n_avail)
            cell_stats[cell_label] = {"avail": n_avail, "picked": n_pick}
            if n_pick == 0:
                continue
            # Deterministic shuffle then pick
            candidates_sorted = sorted(candidates)
            rng.shuffle(candidates_sorted)
            for s in candidates_sorted[:n_pick]:
                # s = (pmid, st, idx)  (idx is int already)
                section_cells[s].append(cell_label)

    # For each unique sampled section, derive candidate_targets = set of T whose
    # alias matches this section's step2 keys (= set of cells the section was
    # picked for, target component)
    samples = []
    for s_key in sorted(section_cells):
        pmid, st, sidx = s_key
        cells = section_cells[s_key]
        # candidate targets: extract target name from each cell label
        cand_targets = sorted({c.split("|", 1)[0] for c in cells})
        info = pool_papers.get(pmid, {})
        samples.append({
            "pmid": pmid,
            "section_type": st,
            "section_index": int(sidx),
            "envs": info.get("envs", []),
            "year": info.get("year"),
            "era": info.get("era"),
            "candidate_targets": cand_targets,
            "picked_cells": cells,
        })

    # Stats
    cells_total = len(cell_stats)
    cells_filled = sum(1 for c in cell_stats.values() if c["picked"] > 0)
    cells_under_quota = sum(
        1 for c in cell_stats.values()
        if 0 < c["picked"] < per_cell
    )
    cells_zero = cells_total - cells_filled
    print(f"\nSampling summary:")
    print(f"  per_cell quota:               {per_cell}")
    print(f"  total (T × E) cells:          {cells_total}")
    print(f"  filled cells (≥1 section):    {cells_filled}")
    print(f"  cells below quota:            {cells_under_quota}")
    print(f"  cells with 0 candidates:      {cells_zero}")
    print(f"  unique sampled sections:      {len(samples)}")
    print(f"  avg targets per section:      "
          f"{sum(len(s['candidate_targets']) for s in samples) / max(1, len(samples)):.1f}")
    return samples, cell_stats


def save(samples, cell_stats):
    ensure_phase7_dir()
    with open(SAMPLED_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pmid", "section_type", "section_index", "envs",
                    "year", "era", "n_candidate_targets", "candidate_targets",
                    "picked_cells"])
        for s in samples:
            w.writerow([
                s["pmid"], s["section_type"], s["section_index"],
                ";".join(s["envs"]), s["year"], s["era"],
                len(s["candidate_targets"]),
                ";".join(s["candidate_targets"]),
                ";".join(s["picked_cells"]),
            ])
    with open(CELL_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(cell_stats, f, ensure_ascii=False, indent=2)
    print(f"  written: {SAMPLED_PATH}")
    print(f"  cell stats: {CELL_STATS_PATH}")


def main() -> None:
    p = argparse.ArgumentParser(prog="section_sampler")
    p.add_argument("--per-cell", type=int, default=DEFAULT_PER_CELL)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    samples, cell_stats = sample(args.per_cell, args.seed)
    save(samples, cell_stats)


if __name__ == "__main__":
    main()
