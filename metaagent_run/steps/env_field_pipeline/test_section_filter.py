"""Smoke-test the section-level target filter without calling LLM.

For 10 randomly sampled sections that have non-empty metadata_keys_found,
compare:
  - baseline: all tier1 + tier2 targets + aliases in prompt
  - optimized: only targets matched by the section's step2 keys

Measure target count, alias count, prompt block size.

Usage (on a800):
    python3 -m metaagent_run.steps.env_field_pipeline.test_section_filter
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from metaagent_run.steps.env_field_pipeline import config
from metaagent_run.steps.step5.metadata_extractor import (
    _build_alias_to_target_index,
    _filter_targets_by_section_keys,
    _format_targets_with_aliases,
)

PHASE6_JSON = config.OUTPUT_DIR / "env6_extraction_targets.json"
STEP2_JSON = config.STEP2_INPUT


def _load_env6_targets_per_env() -> dict:
    with open(PHASE6_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d["per_environment"]


def _extract_fields_for_env(env_fields: list[dict]) -> tuple[list[str], dict[str, list[str]]]:
    """Extract tier1 + tier2 names and aliases map from env6 per-environment block."""
    tier1 = [f["field"] for f in env_fields if f.get("tier") == 1]
    tier2 = [f["field"] for f in env_fields if f.get("tier") == 2]
    fields = tier1 + tier2
    aliases = {f["field"]: (f.get("aliases") or []) for f in env_fields}
    return fields, aliases


def main():
    print("Loading env6 extraction targets...")
    per_env = _load_env6_targets_per_env()
    # Use Open_ocean as the reference env (can cycle through all if needed)
    sample_env = "Open_ocean"
    fields, aliases = _extract_fields_for_env(per_env[sample_env]["fields"])
    print(f"  env={sample_env}: {len(fields)} targets, "
          f"{sum(len(v) for v in aliases.values())} total aliases")

    print("\nLoading step2 records...")
    with open(STEP2_JSON, "r", encoding="utf-8") as f:
        step2 = json.load(f)
    print(f"  {len(step2)} step2 records")

    # Filter to sections with non-empty metadata_keys_found
    sections_with_keys = [
        r for r in step2
        if isinstance(r.get("metadata_keys_found"), list)
        and len(r["metadata_keys_found"]) >= 2
    ]
    print(f"  {len(sections_with_keys)} sections with >= 2 metadata_keys_found")

    # Sample 10 random sections
    random.seed(42)
    samples = random.sample(sections_with_keys, min(10, len(sections_with_keys)))

    print("\n" + "=" * 90)
    print(f"{'pmid':<12} {'section':<18} {'idx':>4} {'keys':>5} {'targets(before/after)':>22} {'bytes(before/after)':>20}")
    print("=" * 90)

    total_before = 0
    total_after = 0
    total_unmatched: Counter = Counter()

    for s in samples:
        pmid = s.get("pmid", "")
        sec_type = s.get("section_type", "")
        sec_idx = int(s.get("index", 0))
        keys = s.get("metadata_keys_found", [])

        # Baseline
        baseline_str = _format_targets_with_aliases(fields, aliases)
        baseline_bytes = len(baseline_str.encode("utf-8"))

        # Optimized (filter)
        filtered_fields, filtered_aliases = _filter_targets_by_section_keys(
            fields, aliases, keys,
            pmid=pmid, section_type=sec_type, section_index=sec_idx,
        )
        opt_str = _format_targets_with_aliases(filtered_fields, filtered_aliases)
        opt_bytes = len(opt_str.encode("utf-8"))

        total_before += baseline_bytes
        total_after += opt_bytes

        # Detect unmatched
        alias_idx = _build_alias_to_target_index(fields, aliases)
        for k in keys:
            if str(k).lower().strip() not in alias_idx:
                total_unmatched[k] += 1

        print(f"{pmid[:12]:<12} {sec_type[:18]:<18} {sec_idx:>4} {len(keys):>5} "
              f"{len(fields)}/{len(filtered_fields):<18} "
              f"{baseline_bytes}/{opt_bytes}")

    print("=" * 90)
    print(f"\nTotal bytes over 10 sections:  baseline={total_before:,}  optimized={total_after:,}")
    if total_before:
        ratio = 100 * (1 - total_after / total_before)
        print(f"Reduction: {ratio:.1f}%")

    print(f"\nUnmatched step2 keys (across 10 sections): {sum(total_unmatched.values())} total, "
          f"{len(total_unmatched)} unique")
    if total_unmatched:
        print("Top 10 unmatched (these are candidates to add as phase6 aliases):")
        for k, c in total_unmatched.most_common(10):
            print(f"  {c}x  {k}")


if __name__ == "__main__":
    main()
