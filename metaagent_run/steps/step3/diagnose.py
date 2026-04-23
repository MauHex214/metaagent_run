"""Pass A diagnostic report for Step 3 (no LLM; seconds-level).

Usage (from the step3 package root on the server):
    python3 -m metaagent_run.steps.step3.diagnose

Produces `step3_pass_a_report.json` alongside the other step3 artefacts.

The report is the primary tool for iterating the rule set in `rules.py`:
it surfaces (a) what each rule matched, (b) which high-DF tokens are still
un-stripped and therefore candidates for new rules, (c) the head-word
coverage sensitivity table, and (d) orphan-key samples.

No LLM calls are made. The expected runtime on ~50K unique keys is seconds.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Package-relative imports (works when run as `python -m ...`).
try:
    from .config import RuntimeConfig, load_runtime_config
    from .schema import filter_discovered_entries
    from .stratification import build_flattened_occurrences
    from .rules import (
        Decomposed, decompose_all, rule_hit_summary, root_tokens,
    )
    from .headword import (
        compute_token_stats, induce_headwords, sensitivity_table,
        tau_coverage_curve, df_histogram, non_headword_summary,
        orphan_keys, orphan_buckets_by_token_count, top_tokens, zipf_points,
    )
except ImportError:
    # Fallback for local/script use — only works if running from
    # inside the step3 directory directly.
    from config import RuntimeConfig, load_runtime_config  # type: ignore
    from schema import filter_discovered_entries  # type: ignore
    from stratification import build_flattened_occurrences  # type: ignore
    from rules import (  # type: ignore
        Decomposed, decompose_all, rule_hit_summary, root_tokens,
    )
    from headword import (  # type: ignore
        compute_token_stats, induce_headwords, sensitivity_table,
        tau_coverage_curve, df_histogram, non_headword_summary,
        orphan_keys, orphan_buckets_by_token_count, top_tokens, zipf_points,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("step3.diagnose")


# ═══════════════════════════════════════════════════════════════════
#  Unstripped tail-token detection
# ═══════════════════════════════════════════════════════════════════

def find_unstripped_tail_tokens(
    decomposed: List[Decomposed],
    min_df: int = 20,
    sample_size: int = 8,
) -> List[Dict[str, Any]]:
    """Tokens that appear as the LAST token of many roots but are not
    stripped by any current rule — candidates for promotion to STAT /
    UNIT / AGGREGATION rule sets.

    We also report tokens that appear as the FIRST token of many roots
    but are not stripped — candidates for ISOTOPE / TAXON / AGGREGATION
    prefix rules.
    """
    last_tok_counter: Counter = Counter()
    first_tok_counter: Counter = Counter()
    last_tok_samples: Dict[str, List[str]] = {}
    first_tok_samples: Dict[str, List[str]] = {}

    for d in decomposed:
        toks = root_tokens(d)
        if not toks:
            continue
        last = toks[-1]
        first = toks[0]
        last_tok_counter[last] += 1
        first_tok_counter[first] += 1
        if len(last_tok_samples.get(last, [])) < sample_size:
            last_tok_samples.setdefault(last, []).append(d.original)
        if len(first_tok_samples.get(first, [])) < sample_size:
            first_tok_samples.setdefault(first, []).append(d.original)

    last_tail = [
        {
            "token": tok, "position": "suffix", "df": cnt,
            "sample_keys": last_tok_samples.get(tok, []),
        }
        for tok, cnt in last_tok_counter.most_common()
        if cnt >= min_df
    ]
    first_tail = [
        {
            "token": tok, "position": "prefix", "df": cnt,
            "sample_keys": first_tok_samples.get(tok, []),
        }
        for tok, cnt in first_tok_counter.most_common()
        if cnt >= min_df
    ]
    # Merge, sort by DF descending
    return sorted(last_tail + first_tail, key=lambda r: -r["df"])


# ═══════════════════════════════════════════════════════════════════
#  Root-collision analysis — how much did Pass A collapse?
# ═══════════════════════════════════════════════════════════════════

def root_collision_stats(decomposed: List[Decomposed]) -> Dict[str, Any]:
    """Count how many distinct original keys collapsed to the same root."""
    from collections import defaultdict
    root_to_originals: Dict[str, List[str]] = defaultdict(list)
    for d in decomposed:
        root_to_originals[d.root].append(d.original)
    multi_root_groups = {
        r: origs for r, origs in root_to_originals.items() if len(origs) > 1
    }
    collision_sizes = Counter(len(origs) for origs in root_to_originals.values())
    top_collisions = sorted(
        multi_root_groups.items(), key=lambda kv: -len(kv[1]),
    )[:30]
    return {
        "total_unique_originals": len(decomposed),
        "total_distinct_roots": len(root_to_originals),
        "roots_with_multiple_originals": len(multi_root_groups),
        "collision_size_histogram": {
            str(size): count for size, count in sorted(collision_sizes.items())
        },
        "top_30_collisions": [
            {"root": r, "collapsed_count": len(origs), "sample": origs[:10]}
            for r, origs in top_collisions
        ],
    }


# ═══════════════════════════════════════════════════════════════════
#  Top head-word families (preview family sizes)
# ═══════════════════════════════════════════════════════════════════

def preview_family_sizes(
    decomposed: List[Decomposed], headwords: set, top_k: int = 40,
) -> Dict[str, Any]:
    """For each headword h, count how many keys will be assigned to family h
    under multi-family membership (root token ∈ headwords).
    """
    sizes: Counter = Counter()
    for d in decomposed:
        for t in set(root_tokens(d)):
            if t in headwords:
                sizes[t] += 1
    sorted_sizes = sizes.most_common()
    p50 = p90 = p99 = 0
    max_family = ""
    max_size = 0
    if sorted_sizes:
        values = [v for _, v in sorted_sizes]
        values_sorted = sorted(values)
        n = len(values_sorted)
        p50 = values_sorted[int(n * 0.50)]
        p90 = values_sorted[min(int(n * 0.90), n - 1)]
        p99 = values_sorted[min(int(n * 0.99), n - 1)]
        max_family, max_size = sorted_sizes[0]
    return {
        "family_count": len(sizes),
        "p50": p50, "p90": p90, "p99": p99,
        "max_family": max_family,
        "max_size": max_size,
        "top_families": [
            {"headword": h, "size": s}
            for h, s in sorted_sizes[:top_k]
        ],
    }


# ═══════════════════════════════════════════════════════════════════
#  Main entry
# ═══════════════════════════════════════════════════════════════════

def load_keys_via_pipeline(
    config: RuntimeConfig,
) -> Tuple[List[str], Dict[str, Any]]:
    """Load discovery input → flatten → filter → return filtered keys.

    Reuses the existing step3 front-end so the diagnostic operates on the
    exact same key set that Phase 1 will later see.
    """
    LOGGER.info("Loading discovery input: %s", config.input_file)
    with config.input_file.open("r", encoding="utf-8") as f:
        items = json.load(f)
    LOGGER.info("  %d sections loaded", len(items))

    unique_keys, occurrences, key_to_evidence = build_flattened_occurrences(items)
    LOGGER.info("  unique_keys=%d, occurrences=%d", len(unique_keys), len(occurrences))

    raw_entries = [
        {"key": k, "evidence": key_to_evidence.get(k, "(step2)")}
        for k in sorted(unique_keys)
    ]
    review_pool: List[Dict[str, Any]] = []
    accepted = filter_discovered_entries(raw_entries, review_pool)
    LOGGER.info(
        "  after filter: accepted=%d, review_pool=%d",
        len(accepted), len(review_pool),
    )
    keys_after_filter = [entry["normalized_key"] for entry in accepted]
    preflight = {
        "sections_loaded": len(items),
        "unique_keys": len(unique_keys),
        "occurrences": len(occurrences),
        "after_filter": len(keys_after_filter),
        "review_pool": len(review_pool),
    }
    return keys_after_filter, preflight


def generate_report(
    config: Optional[RuntimeConfig] = None,
    coverage_target: float = 0.95,
    output_path: Optional[Path] = None,
    unstripped_min_df: int = 20,
) -> Dict[str, Any]:
    """Produce pass_a_report.json. Returns the dict for in-memory use."""
    config = config or load_runtime_config()
    keys, preflight = load_keys_via_pipeline(config)

    LOGGER.info("Decomposing %d keys via rules.decompose ...", len(keys))
    decomposed = decompose_all(keys)

    LOGGER.info("Computing token DF statistics ...")
    stats = compute_token_stats(decomposed)

    LOGGER.info("Solving head-word threshold (target=%.3f) ...", coverage_target)
    H, tau, coverage = induce_headwords(decomposed, coverage_target, stats=stats)

    LOGGER.info("Running sensitivity table ...")
    sens = sensitivity_table(
        decomposed, (0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.99),
        stats=stats,
    )
    curve = tau_coverage_curve(decomposed, stats=stats)

    LOGGER.info("Counting rule hits ...")
    rule_counts = rule_hit_summary(decomposed)

    LOGGER.info("Finding unstripped tail tokens ...")
    tails = find_unstripped_tail_tokens(
        decomposed, min_df=unstripped_min_df,
    )

    LOGGER.info("Computing root collision stats ...")
    collisions = root_collision_stats(decomposed)

    LOGGER.info("Previewing family sizes ...")
    fam_preview = preview_family_sizes(decomposed, H, top_k=40)

    LOGGER.info("Analyzing orphans ...")
    orphans = orphan_keys(decomposed, H)
    orphan_sample = [d.original for d in orphans[:30]]
    orphan_buckets = orphan_buckets_by_token_count(orphans)

    LOGGER.info("Assembling report ...")
    report: Dict[str, Any] = {
        "preflight": preflight,
        "corpus_summary": {
            "total_unique_keys_after_filter": len(decomposed),
            "total_distinct_root_tokens": stats.total_tokens,
            "total_roots": collisions["total_distinct_roots"],
            "roots_with_multiple_originals": collisions["roots_with_multiple_originals"],
        },
        "headword_induction": {
            "coverage_target": coverage_target,
            "threshold_tau": tau,
            "H_size": len(H),
            "coverage_achieved": coverage,
            "orphan_rate": 1.0 - coverage,
            "headwords_sorted_by_df": [
                {"token": t, "df": stats.df[t]}
                for t, _ in sorted(
                    ((t, stats.df[t]) for t in H),
                    key=lambda kv: (-kv[1], kv[0]),
                )
            ],
        },
        "sensitivity_table": sens,
        "tau_coverage_curve": curve,
        "df_histogram": df_histogram(stats),
        "zipf_topN": zipf_points(stats, max_rank=200),
        "non_headword_tail": non_headword_summary(stats, H),
        "rule_hit_counts": rule_counts,
        "rule_hit_totals": {
            name: sum(inner.values())
            for name, inner in rule_counts.items()
        },
        "unstripped_tail_tokens": tails,
        "root_collision": collisions,
        "family_size_preview": fam_preview,
        "orphans": {
            "count": len(orphans),
            "rate": len(orphans) / max(len(decomposed), 1),
            "buckets_by_token_count": orphan_buckets,
            "sample_keys": orphan_sample,
        },
        "top_tokens_overall": [
            {"token": t, "df": d} for t, d in top_tokens(stats, k=80)
        ],
    }

    # Write to disk
    if output_path is None:
        output_path = config.input_file.parent / "step3_pass_a_report.json"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    LOGGER.info("Report written to %s", output_path)
    return report


def _print_summary(report: Dict[str, Any]) -> None:
    """One-screen human-readable digest."""
    pre = report["preflight"]
    hw = report["headword_induction"]
    fs = report["family_size_preview"]
    orph = report["orphans"]
    print("=" * 60)
    print("Step 3 Pass A diagnostic summary")
    print("=" * 60)
    print(f"Input sections           : {pre['sections_loaded']}")
    print(f"Unique keys              : {pre['unique_keys']}")
    print(f"After static filter      : {pre['after_filter']}")
    print(f"Review pool              : {pre['review_pool']}")
    print("-" * 60)
    print(f"Distinct root tokens     : {report['corpus_summary']['total_distinct_root_tokens']}")
    print(f"Distinct roots           : {report['corpus_summary']['total_roots']}")
    print(f"Roots w/ aliases         : {report['corpus_summary']['roots_with_multiple_originals']}")
    print("-" * 60)
    print(f"Head-word threshold τ    : {hw['threshold_tau']}")
    print(f"Head-word set |H|        : {hw['H_size']}")
    print(f"Coverage achieved        : {hw['coverage_achieved']:.4f}")
    print(f"Orphan rate              : {hw['orphan_rate']:.4f}")
    print("-" * 60)
    print("Sensitivity table (target → τ):")
    for row in report["sensitivity_table"]:
        print(
            f"  target={row['coverage_target']:.2f}  "
            f"τ={row['tau']:>4}  |H|={row['H_size']:>5}  "
            f"coverage={row['coverage_achieved']:.4f}  "
            f"orphan={row['orphan_rate']:.4f}"
        )
    print("τ-coverage curve (direct):")
    for row in report["tau_coverage_curve"]:
        print(
            f"  τ={row['tau']:>4}  |H|={row['H_size']:>5}  "
            f"coverage={row['coverage']:.4f}  orphan={row['orphan_rate']:.4f}"
        )
    print("-" * 60)
    print(f"Family count (preview)   : {fs['family_count']}")
    print(f"Family size p50/p90/p99  : {fs['p50']}/{fs['p90']}/{fs['p99']}")
    print(f"Largest family           : {fs['max_family']} ({fs['max_size']} keys)")
    print("-" * 60)
    print(f"Orphan keys              : {orph['count']} ({orph['rate']:.4f})")
    print("-" * 60)
    print("Rule hit totals:")
    for name, tot in report["rule_hit_totals"].items():
        print(f"  {name:16s}: {tot}")
    print("-" * 60)
    print(f"Unstripped tail tokens (DF >= default threshold): "
          f"{len(report['unstripped_tail_tokens'])}")
    for row in report["unstripped_tail_tokens"][:15]:
        print(
            f"  [{row['position']}] {row['token']:25s} DF={row['df']:>5}  "
            f"sample={row['sample_keys'][:3]}"
        )
    print("=" * 60)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Step 3 Pass A diagnostic (no LLM)",
    )
    parser.add_argument(
        "--coverage-target", type=float, default=0.95,
        help="Target fraction of keys to cover by head words",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Report output path (default: step3_pass_a_report.json)",
    )
    parser.add_argument(
        "--unstripped-min-df", type=int, default=20,
        help="Minimum DF for a tail token to appear in the unstripped list",
    )
    args = parser.parse_args()

    config = load_runtime_config()
    report = generate_report(
        config=config,
        coverage_target=args.coverage_target,
        output_path=Path(args.output) if args.output else None,
        unstripped_min_df=args.unstripped_min_df,
    )
    _print_summary(report)


if __name__ == "__main__":
    main()
