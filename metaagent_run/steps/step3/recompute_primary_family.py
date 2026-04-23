"""Post-process Step 3 results using the 'most-specific single family' rule.

The multi-family Union-Find produced catastrophic cross-family transitive
contamination (e.g. a 4,876-member 'depth' super-cluster). This script
replays the existing LLM checkpoint under a stricter rule:

    For each key K, determine the ANCHOR word in K's root tokens with
    the LOWEST document frequency (the most-specific anchor). K is
    considered a 'primary member' of only that anchor's family (and the
    specific chunk within that family that contains K).

    For each family chunk F, F's LLM partition is FILTERED to keep only
    primary members of F. Union-Find then unions within-chunk only —
    cross-chunk cascades are impossible because a key contributes its
    equivalence edges from exactly one chunk.

No LLM re-run. Reads the existing step3_family_checkpoint.jsonl and the
step2 input, produces _primary-suffixed outputs alongside the originals
so both can be compared.
"""
import json
import logging
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, "/pf9550-bdp-A800/wuzhile/hydrosphere_meta/meta_agent0408")

from metaagent_run.steps.step3.config import load_runtime_config
from metaagent_run.steps.step3.orchestrator import (
    load_and_filter,
    load_family_checkpoint,
    build_pmid_count,
    build_attribution,
    collect_side_tags,
    build_key_to_canonical,
    _dump_json,
)
from metaagent_run.steps.step3.rules import decompose, root_tokens
from metaagent_run.steps.step3.headword import compute_token_stats
from metaagent_run.steps.step3.family import (
    build_families, split_all_families, find_orphans,
    PartitionKeyUnionFind, apply_family_partitions,
    select_canonicals_for_groups,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("step3.recompute")

OUT_SUFFIX = "_primary"


def _suffix_path(path: Path, suffix: str) -> Path:
    return path.with_name(path.stem + suffix + path.suffix)


def main() -> None:
    cfg = load_runtime_config()
    t0 = time.monotonic()

    # ---------- Rebuild Pass A + Pass 0 state (deterministic) ----------
    keys, occurrences, _ev, pre = load_and_filter(cfg.input_file)
    LOGGER.info("preflight: %s", pre)

    decomposed = [decompose(k) for k in keys]
    stats = compute_token_stats(decomposed)
    H = {t for t, df in stats.df.items() if df >= cfg.anchor_cutoff}
    LOGGER.info("anchors |H|=%d", len(H))

    families_raw = build_families(decomposed, H)
    families_all = split_all_families(families_raw, cfg.family_size_cap)
    LOGGER.info("families: %d anchors, %d chunks after split",
                len(families_raw), len(families_all))

    # ---------- Determine each key's primary anchor + chunk ----------
    # primary anchor = lowest-DF anchor in the key's root tokens.
    # primary chunk  = the specific chunk of that anchor containing this key.
    anchor_df = {a: stats.df[a] for a in H}

    # Build (anchor, key_original) -> chunk_id index
    member_to_chunk_by_anchor: Dict[str, Dict[str, str]] = defaultdict(dict)
    for fam in families_all:
        for d in fam.members:
            member_to_chunk_by_anchor[fam.anchor][d.original] = fam.id

    primary_anchor: Dict[str, str] = {}
    primary_chunk: Dict[str, str] = {}
    for d in decomposed:
        tokens_in_order = [t for t in root_tokens(d) if t in H]
        if not tokens_in_order:
            continue  # orphan — no primary
        # HEAD-NOUN heuristic:
        # In aquatic-metadata field names, the measured-quantity noun
        # (depth/flux/temperature/carbon/…) typically sits at the
        # rightmost position, preceded by modifier tokens
        # (water/sampling/benthic/organic/…). We therefore take the
        # RIGHTMOST token in H as the primary anchor. If the rightmost
        # is not in H, fall back to the highest-DF anchor (most
        # "well-known" concept marker in the corpus).
        pa = tokens_in_order[-1]
        primary_anchor[d.original] = pa
        chunk_id = member_to_chunk_by_anchor[pa].get(d.original)
        if chunk_id is not None:
            primary_chunk[d.original] = chunk_id

    LOGGER.info("primary assignment: %d keys with primary anchor, %d orphans",
                len(primary_anchor), len(decomposed) - len(primary_anchor))

    # Distribution of how many anchors a key has (for reference)
    num_anchors_hist: Counter = Counter()
    for d in decomposed:
        n = len([t for t in set(root_tokens(d)) if t in H])
        num_anchors_hist[n] += 1
    LOGGER.info("anchors-per-key histogram: %s",
                dict(sorted(num_anchors_hist.items())))

    # ---------- Load LLM checkpoint ----------
    ckpt = load_family_checkpoint(cfg.checkpoint_file)
    LOGGER.info("checkpoint records: %d", len(ckpt))

    # ---------- Filter each chunk's partition to primary members only ----
    filtered_partitions: Dict[str, List[List[str]]] = {}
    total_members_before = 0
    total_members_after = 0
    for chunk_id, res in ckpt.items():
        filtered_groups: List[List[str]] = []
        for group in res.groups:
            total_members_before += len(group)
            kept = [m for m in group if primary_chunk.get(m) == chunk_id]
            total_members_after += len(kept)
            if kept:
                filtered_groups.append(kept)
        filtered_partitions[chunk_id] = filtered_groups

    LOGGER.info(
        "member filtering: %d → %d (%.1f%% of LLM judgments kept)",
        total_members_before, total_members_after,
        100.0 * total_members_after / max(total_members_before, 1),
    )

    # ---------- Union-Find on filtered partitions ----------
    uf = PartitionKeyUnionFind(decomposed)
    apply_family_partitions(uf, filtered_partitions)
    groups = uf.groups()
    LOGGER.info("union-find: %d groups from %d keys (refused=%d)",
                len(groups), len(decomposed), uf.refused_count)

    # ---------- Canonical selection + attribution ----------
    accepted_set = {d.original for d in decomposed}
    pmid_count = build_pmid_count(occurrences, accepted_set)
    canonical_map = select_canonicals_for_groups(groups, pmid_count)
    key_to_canonical = build_key_to_canonical(canonical_map)
    field_pmid_index, field_pmid_env_index = build_attribution(
        occurrences, key_to_canonical
    )
    side_tags = collect_side_tags(decomposed, key_to_canonical)

    # ---------- Stats on result ----------
    alias_sizes = [len(a) for a in canonical_map.values()]
    alias_sizes.sort()
    size_hist: Dict[str, int] = defaultdict(int)
    for s in alias_sizes:
        bucket = ("1" if s == 1 else "2" if s == 2
                  else "3-5" if s <= 5 else "6-10" if s <= 10
                  else "11-20" if s <= 20 else "21-50" if s <= 50
                  else "51-100" if s <= 100 else ">100")
        size_hist[bucket] += 1

    stats_out = {
        "mode": "primary_family_only",
        "preflight": pre,
        "anchor_cutoff": cfg.anchor_cutoff,
        "family_size_cap": cfg.family_size_cap,
        "anchor_word_set_size": len(H),
        "anchors_per_key_histogram": dict(sorted(num_anchors_hist.items())),
        "llm_member_judgments_kept_pct": round(
            100.0 * total_members_after / max(total_members_before, 1), 2
        ),
        "unionfind_refused_count": uf.refused_count,
        "canonical_count": len(canonical_map),
        "alias_group_size_distribution": dict(size_hist),
        "orphan_keys": len(find_orphans(decomposed, H)),
        "elapsed_seconds": round(time.monotonic() - t0, 1),
    }

    # ---------- Write to _primary-suffixed paths ----------
    write_map = {
        cfg.output_file: canonical_map,
        cfg.field_pmid_index_file: field_pmid_index,
        cfg.field_pmid_env_index_file: field_pmid_env_index,
        cfg.side_tags_file: side_tags,
        cfg.pass_b_stats_file: stats_out,
    }
    for orig_path, data in write_map.items():
        new_path = _suffix_path(Path(orig_path), OUT_SUFFIX)
        _dump_json(new_path, data)
        LOGGER.info("wrote %s", new_path)

    print()
    print("Summary:")
    for k, v in stats_out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
