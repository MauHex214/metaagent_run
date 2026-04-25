"""Section-level pool + target index — phase7's data layer.

Design (per redesign):
  Phase 7 evaluates "is this field extractable from text?" — pure text
  question, no acc binding required. So the pool drops the has_accession
  constraint, swelling from 3,426 → ~23,500 papers, which gives every
  target enough section evidence (eliminates the kept_observe limbo).

  Evaluation unit is SECTION, keyed by (pmid, section_type, section_index).
  step2's metadata_keys_found at this granularity is the "this section
  mentions these fields" signal. step5's segment-level text (multiple
  segments may collapse to one (pmid, st, idx) tuple) is fetched on demand.

Outputs (cached):
  broad_paper_pool.json
    - target_env ∩ has_year (no has_acc)
    - papers: pmid → {envs, year, era}
  target_section_index.json
    - For each env6_v1b canonical T:
        target_sections[T] = list of [pmid, sec_type, sec_idx] where
          step2_metadata_keys_found ∋ alias(T) AND pmid in broad pool

CLI:
  python -m metaagent_run.steps.env_field_pipeline.phase7_validation.section_indexer build
"""
from __future__ import annotations

import argparse
import json
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
    to_era,
)

PMID_YEAR_PATH = ep_config.PROJECT_ROOT_DIR / "paper_down" / "pmid_year.txt"
ENV6_V1B_PATH = ep_config.OUTPUT_DIR / "env6_extraction_targets_v1b.json"
BROAD_POOL_PATH = PHASE7_DIR / "broad_paper_pool.json"
TARGET_SECTION_INDEX_PATH = PHASE7_DIR / "target_section_index.json"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower().strip()).strip("_")


# ─── Broad paper pool (no has_acc) ──────────────────────────────────────

def _load_paper_envs() -> dict[str, list[str]]:
    with open(ep_config.PAPER_ENV_MAP, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        pmid: [e for e in envs if e in TARGET_ENVS]
        for pmid, envs in raw.items()
        if any(e in TARGET_ENVS for e in envs)
    }


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


def build_broad_pool() -> dict:
    """Pool = pmids in (target_env ∩ has_year). No has_acc."""
    print("Building broad paper pool (no has_acc) ...", flush=True)
    paper_envs = _load_paper_envs()
    pmid_year = _load_pmid_year()
    papers: dict[str, dict] = {}
    for pmid, envs in paper_envs.items():
        year = pmid_year.get(pmid)
        if year is None:
            continue
        papers[pmid] = {
            "envs": envs,
            "year": year,
            "era": to_era(year),
        }
    print(f"  broad pool: {len(papers)} pmids "
          f"(vs old narrow pool with has_acc: ~3,426)", flush=True)
    return {
        "version": 1,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "constraint": "target_env ∩ has_year (NO has_acc)",
        "n_papers": len(papers),
        "papers": papers,
    }


# ─── Target → section index ─────────────────────────────────────────────

def _load_v1b_aliases() -> dict[str, set[str]]:
    """canonical_name → set(normalized aliases including own name)."""
    with open(ENV6_V1B_PATH, "r", encoding="utf-8") as f:
        v1b = json.load(f)
    out: dict[str, set[str]] = {}
    for env_block in v1b.get("per_environment", {}).values():
        for fdef in env_block.get("fields", []):
            name = fdef["field"]
            entry = out.setdefault(name, set())
            entry.add(_norm(name))
            for a in fdef.get("aliases") or []:
                na = _norm(a)
                if na:
                    entry.add(na)
    return out


def _load_v1b_envs_per_target() -> dict[str, list[str]]:
    with open(ENV6_V1B_PATH, "r", encoding="utf-8") as f:
        v1b = json.load(f)
    out: dict[str, set[str]] = defaultdict(set)
    for env_name, env_block in v1b.get("per_environment", {}).items():
        for fdef in env_block.get("fields", []):
            out[fdef["field"]].add(env_name)
    return {t: sorted(s) for t, s in out.items()}


def build_target_section_index(broad_pool: dict) -> dict:
    """For each canonical T, list (pmid, st, idx) tuples whose step2
    metadata_keys_found includes any T alias AND pmid in broad pool."""
    pool_pmids = set(broad_pool["papers"].keys())
    print(f"  pool size: {len(pool_pmids)} pmids", flush=True)

    target_aliases = _load_v1b_aliases()
    print(f"  env6_v1b: {len(target_aliases)} canonicals", flush=True)
    target_envs = _load_v1b_envs_per_target()

    # Inverted: alias_norm → set(target_names)
    alias_to_targets: dict[str, set[str]] = defaultdict(set)
    for tname, aliases_norm in target_aliases.items():
        for an in aliases_norm:
            alias_to_targets[an].add(tname)

    print(f"  loading {ep_config.STEP2_INPUT.name} ...", flush=True)
    with open(ep_config.STEP2_INPUT, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"  {len(records):,} records", flush=True)

    # target → list of [pmid, st, idx] tuples
    target_sections: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    section_tuple_seen: set[tuple[str, str, str, int]] = set()

    for r in records:
        pmid = str(r.get("pmid", ""))
        if pmid not in pool_pmids:
            continue
        keys = r.get("metadata_keys_found")
        if not isinstance(keys, list):
            continue
        sec_type = r.get("section_type", "")
        sec_idx = int(r.get("index", 0))
        # Determine which targets this section signals for
        signaled: set[str] = set()
        for k in keys:
            nk = _norm(k)
            for tname in alias_to_targets.get(nk, ()):
                signaled.add(tname)
        # Add (pmid, st, idx) to each signaled target's section list
        # Dedup at (target, pmid, st, idx) level — same (st,idx) may have
        # multiple step2 records (collapsed) with different keys; the union
        # of all their signaled targets is what we want.
        for tname in signaled:
            key = (tname, pmid, sec_type, sec_idx)
            if key in section_tuple_seen:
                continue
            section_tuple_seen.add(key)
            target_sections[tname].append([pmid, sec_type, sec_idx])

    # Stats
    bucket = Counter()
    for tname in target_aliases:
        n = len(target_sections.get(tname, []))
        if n == 0: bucket["0"] += 1
        elif n < 10: bucket["1-9"] += 1
        elif n < 50: bucket["10-49"] += 1
        elif n < 200: bucket["50-199"] += 1
        elif n < 1000: bucket["200-999"] += 1
        else: bucket["1000+"] += 1

    print(f"\nSection coverage per target (broad pool):")
    for k in ["0", "1-9", "10-49", "50-199", "200-999", "1000+"]:
        print(f"  {k:<10} {bucket.get(k, 0):>4} targets")
    n_with_evidence = sum(1 for tname in target_aliases
                          if target_sections.get(tname))
    print(f"\n  {n_with_evidence}/{len(target_aliases)} targets have ≥1 section "
          f"({100*n_with_evidence/len(target_aliases):.1f}%)")

    return {
        "version": 1,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "v1b_path": str(ENV6_V1B_PATH),
        "broad_pool_n_papers": len(pool_pmids),
        "n_targets": len(target_aliases),
        "n_targets_with_sections": n_with_evidence,
        "target_sections": dict(target_sections),
        "target_envs": target_envs,
    }


def main() -> None:
    p = argparse.ArgumentParser(prog="section_indexer")
    p.add_argument("cmd", choices=["build"], help="build broad pool + target_section_index")
    args = p.parse_args()

    ensure_phase7_dir()
    pool = build_broad_pool()
    with open(BROAD_POOL_PATH, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False)
    print(f"Wrote {BROAD_POOL_PATH}", flush=True)

    idx = build_target_section_index(pool)
    with open(TARGET_SECTION_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False)
    print(f"Wrote {TARGET_SECTION_INDEX_PATH}", flush=True)


if __name__ == "__main__":
    main()
