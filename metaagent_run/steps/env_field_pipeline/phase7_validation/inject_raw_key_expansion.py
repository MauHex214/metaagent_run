"""Inject raw_key_expansion entries into env6 aliases → env6_v1b.

env6_extraction_targets.json (phase6 output) has aliases per canonical, but
those aliases come from the canonical→raw_key clustering of phase 0-4 only.
The raw_key_expansion_table.csv (~350 entries) — short forms (`k`→potassium,
`no3`→nitrate, `po4`→phosphate, etc.) and unicode/punctuation variants — was
NOT incorporated into env6 because phase6's pipeline didn't read it.

Effect: many step2 metadata_keys (e.g., `k` mentioned 251 times across 1.7M
records) match no env6 alias → step5 prompt doesn't include them as targets,
phase7 metric undercounts tries → schema v2 prune decisions become unstable
(e.g., `potassium` got pruned at tries=13/rate=15% in the prior run, but
adding `k` as alias would lift tries to ~40-50 and likely flip the call).

This script injects expansion rows into env6 by name-matching:
  raw_key_expansion row "(k, potassium)"
    → look up env6 canonical "potassium"
    → add "k" to its aliases

Output: env6_extraction_targets_v1b.json (alongside the v1 file as audit).
The v1b file is what step5 + phase7 should henceforth consume.

CLI:
  python -m metaagent_run.steps.env_field_pipeline.phase7_validation.inject_raw_key_expansion
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from metaagent_run.steps.env_field_pipeline import config as ep_config

EXPANSION_PATH = ep_config.PROJECT_ROOT_DIR.parents[1] / "Claude" / "metaagent_run" / "docs" / "raw_key_expansion_table.csv"
# Fallback: same file copied to a800 docs/
EXPANSION_FALLBACKS = [
    Path("/pf9550-bdp-A800/wuzhile/hydrosphere_meta/meta_agent0408/docs/raw_key_expansion_table.csv"),
    Path("/tmp/raw_key_expansion_table.csv"),
]

V1_PATH = ep_config.OUTPUT_DIR / "env6_extraction_targets.json"
V1B_PATH = ep_config.OUTPUT_DIR / "env6_extraction_targets_v1b.json"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower().strip()).strip("_")


def _resolve_expansion_path() -> Path:
    if EXPANSION_PATH.exists():
        return EXPANSION_PATH
    for p in EXPANSION_FALLBACKS:
        if p.exists():
            return p
    raise FileNotFoundError(
        "raw_key_expansion_table.csv not found. Tried:\n  "
        + "\n  ".join([str(EXPANSION_PATH)] + [str(p) for p in EXPANSION_FALLBACKS])
    )


def _load_expansion_table(path: Path) -> dict[str, set[str]]:
    """Returns normalized_raw_key → set of raw_keys that should map to it.
    Comments (lines starting with #) and blank rows are skipped."""
    out: dict[str, set[str]] = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)   # header
        for row in reader:
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 2:
                continue
            raw, normed = row[0].strip(), row[1].strip()
            if raw and normed:
                out[normed].add(raw)
    return dict(out)


def _build_canonical_index(env6: dict) -> dict[str, list[tuple[str, dict]]]:
    """Returns normalized_lookup_key → list of (env_name, field_dict).
    The lookup key covers both the canonical name and its existing aliases,
    so an expansion row like (po4, phosphate) can match either a canonical
    named 'phosphate' OR a canonical with 'phosphate' in its aliases."""
    idx: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for env_name, env_block in env6.get("per_environment", {}).items():
        for fdef in env_block.get("fields", []):
            keys = {fdef["field"]} | set(fdef.get("aliases") or [])
            for k in keys:
                nk = _norm(k)
                if nk:
                    idx[nk].append((env_name, fdef))
    return idx


def inject(v1_path: Path = V1_PATH, v1b_path: Path = V1B_PATH) -> dict:
    expansion_path = _resolve_expansion_path()
    print(f"Loading expansion table: {expansion_path}", flush=True)
    expansion = _load_expansion_table(expansion_path)
    print(f"  {len(expansion)} normalized targets, "
          f"{sum(len(v) for v in expansion.values())} raw_key entries",
          flush=True)

    print(f"Loading env6 v1: {v1_path}", flush=True)
    with open(v1_path, "r", encoding="utf-8") as f:
        env6 = json.load(f)
    canonical_idx = _build_canonical_index(env6)
    print(f"  {sum(len(env_block['fields']) for env_block in env6['per_environment'].values())} field rows across "
          f"{len(env6['per_environment'])} envs", flush=True)

    # Inject — for each expansion row, find matching env6 canonical(s) and add
    # the raw_key to each match's aliases (dedupe against existing).
    matched_normed = 0
    unmatched_normed: list[str] = []
    raw_keys_added = 0
    per_canonical_added: dict[str, set[str]] = defaultdict(set)

    for normed, raw_keys in expansion.items():
        matches = canonical_idx.get(_norm(normed))
        if not matches:
            unmatched_normed.append(normed)
            continue
        matched_normed += 1
        seen_field_ids = set()  # avoid double-add if same field appears in multiple envs (it does)
        for env_name, fdef in matches:
            field_id = id(fdef)
            if field_id in seen_field_ids:
                continue
            seen_field_ids.add(field_id)
            existing = set(_norm(a) for a in [fdef["field"]] + (fdef.get("aliases") or []))
            new_aliases = list(fdef.get("aliases") or [])
            for raw in raw_keys:
                if _norm(raw) not in existing:
                    new_aliases.append(raw)
                    existing.add(_norm(raw))
                    raw_keys_added += 1
                    per_canonical_added[fdef["field"]].add(raw)
            fdef["aliases"] = sorted(set(new_aliases))

    # Update metadata
    meta = env6.setdefault("metadata", {})
    meta["v1b_raw_key_expansion_injected"] = True
    meta["v1b_expansion_source"] = str(expansion_path)
    meta["v1b_matched_normed_targets"] = matched_normed
    meta["v1b_unmatched_normed_targets_count"] = len(unmatched_normed)
    meta["v1b_unmatched_normed_targets_sample"] = sorted(unmatched_normed)[:30]
    meta["v1b_raw_keys_added_total"] = raw_keys_added
    meta["v1b_top_canonicals_by_added_count"] = sorted(
        ((cn, len(adds)) for cn, adds in per_canonical_added.items()),
        key=lambda x: -x[1],
    )[:25]

    with open(v1b_path, "w", encoding="utf-8") as f:
        json.dump(env6, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {v1b_path}", flush=True)
    print(f"  matched normed targets:    {matched_normed} / {len(expansion)}", flush=True)
    print(f"  unmatched normed targets:  {len(unmatched_normed)}  "
          f"(samples: {sorted(unmatched_normed)[:8]})", flush=True)
    print(f"  raw_keys added (deduped):  {raw_keys_added}", flush=True)
    print(f"\nTop 10 canonicals by raw_keys added:")
    for cn, n in sorted(per_canonical_added.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  {cn:<30} +{n} aliases")
    return env6


def main() -> None:
    parser = argparse.ArgumentParser(prog="inject_raw_key_expansion")
    parser.add_argument("--input", type=Path, default=V1_PATH)
    parser.add_argument("--output", type=Path, default=V1B_PATH)
    args = parser.parse_args()
    inject(args.input, args.output)


if __name__ == "__main__":
    main()
