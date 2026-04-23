"""Step 3 orchestrator — new (Pass A / Pass 0 / Pass B / Pass C) pipeline.

Flow:
    Pass A  : structural decomposition via rules.decompose
    Pass 0  : anchor-word induction via headword.induce (DF ≥ cutoff)
    Pass B  : family-level LLM partitioning (dedup_family.run_family_partitioning)
              with concurrency ramp-up and per-family JSONL checkpoint
    Pass C  : Union-Find with hard partition-key (isotope, substance) constraint
              → canonical selection → attribution index build

Outputs (paths from RuntimeConfig):
    output_file               — synonym_groups JSON  {canonical: [aliases...]}
    field_pmid_index_file     — {raw_field: [pmid...]}  (alias-level; canonical
                                PMIDs computed downstream by union over members)
    field_pmid_env_index_file — {raw_field: {env: [pmid...]}}
    side_tags_file            — {canonical: {tag: distribution}}
    pass_b_stats_file         — family-level run stats
    checkpoint_file (JSONL)   — one record per family, enables resume
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from metaagent_run.core import AsyncLocalModelClient

from .config import RuntimeConfig, load_runtime_config
from .dedup_family import (
    FamilyPartitionResult,
    RampUpConfig,
    partition_family,
    run_family_partitioning,
    results_to_partitions,
)
from .family import (
    Family,
    PartitionKeyUnionFind,
    apply_family_partitions,
    auto_merge_sampling_context,
    build_families,
    drop_singleton_families,
    family_size_stats,
    find_orphans,
    select_canonicals_for_groups,
    split_all_families,
)
from .headword import compute_token_stats, induce_headwords
from .rules import Decomposed, decompose, root_tokens
from .schema import filter_discovered_entries
from .stratification import build_flattened_occurrences


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("step3.orchestrator")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════
#  Corpus preparation (Pass A + Pass 0)
# ═══════════════════════════════════════════════════════════════════

def load_and_filter(
    input_file: Path,
) -> Tuple[List[str], List[Tuple[str, str, str]], Dict[str, str], Dict[str, int]]:
    """Returns (keys_after_filter, occurrences, evidence, preflight_counts)."""
    LOGGER.info("Loading discovery input: %s", input_file)
    with input_file.open("r", encoding="utf-8") as f:
        items = json.load(f)
    unique_keys, occurrences, evidence = build_flattened_occurrences(items)
    raw_entries = [
        {"key": k, "evidence": evidence.get(k, "(step2)")}
        for k in sorted(unique_keys)
    ]
    review_pool: List[Dict[str, Any]] = []
    accepted = filter_discovered_entries(raw_entries, review_pool)
    keys = [e["normalized_key"] for e in accepted]
    preflight = {
        "sections_loaded": len(items),
        "unique_keys": len(unique_keys),
        "occurrences": len(occurrences),
        "after_filter": len(keys),
        "review_pool": len(review_pool),
    }
    LOGGER.info("preflight: %s", preflight)
    return keys, occurrences, evidence, preflight


# ═══════════════════════════════════════════════════════════════════
#  Checkpoint: per-family JSONL
# ═══════════════════════════════════════════════════════════════════

def _result_to_record(res: FamilyPartitionResult) -> Dict[str, Any]:
    return {
        "family_id": res.family_id,
        "anchor": res.anchor,
        "groups": res.groups,
        "status": res.status,
        "audit": res.audit,
        "error": res.error,
    }


def _record_to_result(rec: Dict[str, Any]) -> FamilyPartitionResult:
    return FamilyPartitionResult(
        family_id=rec["family_id"],
        anchor=rec.get("anchor", ""),
        groups=rec.get("groups", []),
        status=rec.get("status", "ok"),
        audit=rec.get("audit", {}),
        error=rec.get("error"),
    )


def load_family_checkpoint(
    checkpoint_file: Path,
) -> Dict[str, FamilyPartitionResult]:
    if not checkpoint_file.exists():
        return {}
    out: Dict[str, FamilyPartitionResult] = {}
    with checkpoint_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            res = _record_to_result(rec)
            out[res.family_id] = res
    LOGGER.info("Resumed %d family records from checkpoint", len(out))
    return out


def append_family_record(
    checkpoint_file: Path, res: FamilyPartitionResult,
) -> None:
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_result_to_record(res), ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════════
#  Pass C — Union-Find, canonical, attribution
# ═══════════════════════════════════════════════════════════════════

def build_pmid_count(
    occurrences: List[Tuple[str, str, str]],
    accepted_keys: Set[str],
) -> Dict[str, int]:
    """For each accepted normalized_key, count distinct PMIDs in which
    it was observed. Used for canonical tie-breaking."""
    pmids_by_key: Dict[str, Set[str]] = defaultdict(set)
    for (k, pmid, _env) in occurrences:
        if k in accepted_keys:
            pmids_by_key[k].add(str(pmid))
    return {k: len(v) for k, v in pmids_by_key.items()}


def build_attribution(
    occurrences: List[Tuple[str, str, str]],
    key_to_canonical: Dict[str, str],
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
    """Produce (field_pmid_index, field_pmid_env_index).

    Indexed by **raw field** (alias), not canonical. Downstream
    (step4-meta) computes canonical PMIDs by unioning across
    synonym_group members, enabling alias-level audits and retroactive
    corrections without information loss.

    field_pmid_index        : raw_field → sorted unique PMIDs
    field_pmid_env_index    : raw_field → env → sorted unique PMIDs

    Keys filtered out during Pass B/C (not in key_to_canonical) are
    dropped here, so the index size equals the number of accepted
    aliases across all canonicals.
    """
    pmid_by_field: Dict[str, Set[str]] = defaultdict(set)
    pmid_env_by_field: Dict[str, Dict[str, Set[str]]] = defaultdict(
        lambda: defaultdict(set),
    )
    for (k, pmid, env) in occurrences:
        if k not in key_to_canonical:
            continue  # key was filtered out (review_pool etc.)
        pmid_str = str(pmid)
        pmid_by_field[k].add(pmid_str)
        if env:
            pmid_env_by_field[k][env].add(pmid_str)
    return (
        {f: sorted(v) for f, v in pmid_by_field.items()},
        {
            f: {e: sorted(pmids) for e, pmids in env_map.items()}
            for f, env_map in pmid_env_by_field.items()
        },
    )


def collect_side_tags(
    decomposed: List[Decomposed],
    key_to_canonical: Dict[str, str],
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """For each canonical, report the distribution of side-tag values
    seen across its member keys. Structure:
        {canonical: {tag_name: {tag_value: count}}}
    Substance is reported per-element (each element in the set counts once).
    """
    out: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for d in decomposed:
        canon = key_to_canonical.get(d.original)
        if canon is None:
            continue
        for tag, value in d.side_tags.items():
            if value is None:
                continue
            if tag == "substance":
                for elem in value:
                    out[canon][tag][elem] += 1
            else:
                out[canon][tag][str(value)] += 1
    # Un-default and sort the inner dicts for stable on-disk form
    return {
        canon: {
            tag: dict(sorted(values.items(), key=lambda kv: -kv[1]))
            for tag, values in tags.items()
        }
        for canon, tags in out.items()
    }


def build_key_to_canonical(
    canonical_map: Dict[str, List[str]],
) -> Dict[str, str]:
    """Invert {canonical: [aliases]} into {alias: canonical}."""
    out: Dict[str, str] = {}
    for canon, aliases in canonical_map.items():
        for a in aliases:
            out[a] = canon
    return out


# ═══════════════════════════════════════════════════════════════════
#  Head-noun primary-family filter
# ═══════════════════════════════════════════════════════════════════
#
# Multi-family membership causes cross-family transitive contamination:
# a single LLM mis-merge in one family propagates through Union-Find
# into unrelated concept clusters (we observed a 4,876-member depth
# super-cluster containing temperature/date/light-level keys).
#
# Fix: restrict each key K's Union-Find contribution to exactly one
# family — the one whose anchor word is K's HEAD NOUN (the rightmost
# anchor token in K's root). For aquatic-metadata field names, the
# rightmost token is typically the measured quantity (depth / flux /
# temperature / carbon) while leading tokens are sampling-context
# modifiers (water / sampling / benthic / organic). Listening only to
# the head-noun family's LLM judgment prevents cross-family cascades.

def compute_primary_chunks(
    decomposed: List[Decomposed],
    headwords: set,
    families_all: List[Any],
) -> Dict[str, str]:
    """Return {key_original: chunk_id_of_primary_family}.

    Primary family = the chunk whose anchor is the key's RIGHTMOST root
    token in H. Keys with no anchor in H are orphans (absent from the
    return dict).
    """
    member_to_chunk_by_anchor: Dict[str, Dict[str, str]] = defaultdict(dict)
    for fam in families_all:
        for d in fam.members:
            member_to_chunk_by_anchor[fam.anchor][d.original] = fam.id

    primary_chunk: Dict[str, str] = {}
    for d in decomposed:
        tokens_in_order = [t for t in root_tokens(d) if t in headwords]
        if not tokens_in_order:
            continue
        pa = tokens_in_order[-1]  # head noun
        cid = member_to_chunk_by_anchor[pa].get(d.original)
        if cid is not None:
            primary_chunk[d.original] = cid
    return primary_chunk


def filter_partitions_by_primary(
    partition_map: Dict[str, List[List[str]]],
    primary_chunk: Dict[str, str],
) -> Tuple[Dict[str, List[List[str]]], Dict[str, int]]:
    """Drop members from each family-chunk's partition whose primary
    chunk is not this chunk. Returns (filtered_map, audit_counts)."""
    filtered: Dict[str, List[List[str]]] = {}
    members_before = 0
    members_after = 0
    for family_id, groups in partition_map.items():
        new_groups: List[List[str]] = []
        for group in groups:
            members_before += len(group)
            kept = [m for m in group if primary_chunk.get(m) == family_id]
            members_after += len(kept)
            if kept:
                new_groups.append(kept)
        filtered[family_id] = new_groups
    return filtered, {
        "members_before_filter": members_before,
        "members_after_filter": members_after,
    }


# ═══════════════════════════════════════════════════════════════════
#  Output writers
# ═══════════════════════════════════════════════════════════════════

def _dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_outputs(
    config: RuntimeConfig,
    canonical_map: Dict[str, List[str]],
    field_pmid_index: Dict[str, List[str]],
    field_pmid_env_index: Dict[str, Dict[str, List[str]]],
    side_tags: Dict[str, Dict[str, Dict[str, int]]],
    stats: Dict[str, Any],
) -> None:
    # Wrap canonical_map in the envelope shape consumed by Step 4-meta,
    # Step 5, Step 6, Step 7 — all of which do `data.get("synonym_groups")`.
    # `final_schema` is intentionally NOT written: it is derivable as
    # list(synonym_groups.keys()); step4-meta's loader falls back to
    # that derivation when final_schema is absent.
    schema_payload = {"synonym_groups": canonical_map}
    _dump_json(Path(config.output_file), schema_payload)
    _dump_json(Path(config.field_pmid_index_file), field_pmid_index)
    _dump_json(Path(config.field_pmid_env_index_file), field_pmid_env_index)
    _dump_json(Path(config.side_tags_file), side_tags)
    _dump_json(Path(config.pass_b_stats_file), stats)
    LOGGER.info("Wrote: %s", config.output_file)
    LOGGER.info("Wrote: %s", config.field_pmid_index_file)
    LOGGER.info("Wrote: %s", config.field_pmid_env_index_file)
    LOGGER.info("Wrote: %s", config.side_tags_file)
    LOGGER.info("Wrote: %s", config.pass_b_stats_file)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

async def main_async(
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    checkpoint_file: Optional[str] = None,
    runtime_config: Optional[RuntimeConfig] = None,
    review_file: Optional[str] = None,
) -> None:
    config = runtime_config or load_runtime_config()
    in_path = Path(input_file or config.input_file)
    out_path = Path(output_file or config.output_file)
    ckpt_path = Path(checkpoint_file or config.checkpoint_file)

    # Fail-fast: if --review was requested, make sure the decisions file
    # exists before spending time on Pass A/0/B/C. The apply step runs
    # after write_outputs, but we want the user to know immediately.
    review_path: Optional[Path] = Path(review_file) if review_file else None
    if review_path is not None and not review_path.exists():
        raise FileNotFoundError(
            f"--review decisions file does not exist: {review_path}"
        )

    t0 = time.monotonic()

    # ---------- Pass A + Pass 0 ----------
    keys, occurrences, evidence, preflight = load_and_filter(in_path)

    LOGGER.info("Pass A: decomposing %d keys", len(keys))
    decomposed = [decompose(k) for k in keys]

    LOGGER.info("Pass 0: computing token DF + inducing anchor words")
    stats = compute_token_stats(decomposed)
    H = {t for t, df in stats.df.items() if df >= config.anchor_cutoff}
    LOGGER.info(
        "anchor_cutoff=%d → |H|=%d (from %d distinct tokens)",
        config.anchor_cutoff, len(H), stats.total_tokens,
    )

    # ---------- Build families ----------
    LOGGER.info("Building families + splitting oversized (cap=%d)",
                config.family_size_cap)
    families_raw = build_families(decomposed, H)
    families_all = split_all_families(families_raw, config.family_size_cap)
    families_work = drop_singleton_families(families_all)
    LOGGER.info(
        "families: %d anchors, %d chunks after split, %d to run (dropped %d singletons)",
        len(families_raw), len(families_all),
        len(families_work), len(families_all) - len(families_work),
    )
    LOGGER.info("family size stats: %s", family_size_stats(families_work))

    orphans = find_orphans(decomposed, H)
    LOGGER.info("orphans (no anchor): %d keys", len(orphans))

    # ---------- Pass B (LLM) ----------
    cached = load_family_checkpoint(ckpt_path)
    to_run = [f for f in families_work if f.id not in cached]
    LOGGER.info("Pass B: %d families cached, %d to run", len(cached), len(to_run))

    def _progress_cb(completed: int, total: int, res: FamilyPartitionResult) -> None:
        # tqdm progress bar (inside run_family_partitioning) already renders
        # live progress; this callback is responsible only for persisting
        # the per-family record so a crash can resume from checkpoint.
        append_family_record(ckpt_path, res)

    if to_run:
        async with AsyncExitStack() as stack:
            llm = await stack.enter_async_context(AsyncLocalModelClient(
                base_url=config.base_url,
                model=config.model,
                api_key=config.api_key,
                api_style=config.api_style,
                azure_api_version=config.azure_api_version,
                azure_deployment=config.azure_deployment,
                auth_mode=config.auth_mode,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            ))
            ramp = RampUpConfig(
                initial=config.llm_concurrency_initial,
                step=config.llm_concurrency_step,
                ceiling=config.llm_concurrency_ceiling,
                step_every_seconds=config.llm_concurrency_step_every_seconds,
            )
            new_results = await run_family_partitioning(
                to_run, llm,
                prompt_version=config.prompt_version,
                max_retries=config.llm_max_retries_per_family,
                ramp=ramp,
                progress_cb=_progress_cb,
            )
        all_results: Dict[str, FamilyPartitionResult] = {**cached, **new_results}
    else:
        all_results = cached

    # ---------- Pass C ----------
    LOGGER.info("Pass C: Union-Find + canonical selection + attribution")
    partition_map = results_to_partitions(all_results)

    # Head-noun primary-family filter: each key's contribution to UF
    # comes from exactly one chunk (the one containing the key's head
    # noun). This prevents cross-family transitive contamination.
    LOGGER.info("  applying head-noun primary-family filter")
    primary_chunk = compute_primary_chunks(decomposed, H, families_all)
    partition_map, filter_audit = filter_partitions_by_primary(
        partition_map, primary_chunk,
    )
    kept_pct = 100.0 * filter_audit["members_after_filter"] / max(
        filter_audit["members_before_filter"], 1
    )
    LOGGER.info(
        "  member filter: %d → %d LLM judgments kept (%.1f%%)",
        filter_audit["members_before_filter"],
        filter_audit["members_after_filter"], kept_pct,
    )

    uf = PartitionKeyUnionFind(decomposed)
    apply_family_partitions(uf, partition_map)

    # Deterministic Class-1 SAMPLING-CONTEXT auto-merge — enforces the
    # dedup_v1 Class-1 rule for pairs the LLM partition stage missed
    # (typically due to chunk-boundary splits). Rule-based, domain-
    # stable (prefix list from dedup_v1.txt), corpus-agnostic.
    class1_stats = auto_merge_sampling_context(uf, decomposed)
    LOGGER.info(
        "Class-1 auto-merge: %d unions across %d clusters",
        class1_stats["unions_performed"], class1_stats["clusters_merged"],
    )

    groups = uf.groups()
    LOGGER.info(
        "union-find: %d groups from %d keys (refused unions across partition_key: %d)",
        len(groups), len(decomposed), uf.refused_count,
    )

    accepted_set = {d.original for d in decomposed}
    pmid_count = build_pmid_count(occurrences, accepted_set)
    canonical_map = select_canonicals_for_groups(groups, pmid_count)
    key_to_canonical = build_key_to_canonical(canonical_map)
    LOGGER.info("canonical vocabulary: %d entries", len(canonical_map))

    field_pmid_index, field_pmid_env_index = build_attribution(
        occurrences, key_to_canonical,
    )
    side_tags = collect_side_tags(decomposed, key_to_canonical)

    # ---------- Stats ----------
    status_counts = {"ok": 0, "fallback": 0, "skip": 0}
    audit_totals = {"dropped": 0, "duplicated": 0, "hallucinated": 0}
    for res in all_results.values():
        status_counts[res.status] = status_counts.get(res.status, 0) + 1
        for k, v in (res.audit or {}).items():
            audit_totals[k] = audit_totals.get(k, 0) + int(v)

    # Alias-group-size distribution (how "lumpy" is the vocabulary)
    alias_sizes = sorted([len(a) for a in canonical_map.values()])
    group_size_hist: Dict[str, int] = defaultdict(int)
    for s in alias_sizes:
        bucket = "1" if s == 1 else "2" if s == 2 else "3-5" if s <= 5 \
            else "6-10" if s <= 10 else "11-20" if s <= 20 \
            else "21-50" if s <= 50 else "51-100" if s <= 100 else ">100"
        group_size_hist[bucket] += 1

    stats_out = {
        "preflight": preflight,
        "anchor_cutoff": config.anchor_cutoff,
        "family_size_cap": config.family_size_cap,
        "family_membership_mode": "primary_head_noun",
        "anchor_word_set_size": len(H),
        "family_count": len(families_raw),
        "family_chunks_after_split": len(families_all),
        "families_run": len(families_work),
        "pass_b_family_status": status_counts,
        "pass_b_audit_totals": audit_totals,
        "orphan_keys": len(orphans),
        "primary_filter_members_before": filter_audit["members_before_filter"],
        "primary_filter_members_kept": filter_audit["members_after_filter"],
        "primary_filter_kept_pct": round(kept_pct, 2),
        "class1_auto_merge_unions": class1_stats["unions_performed"],
        "class1_auto_merge_clusters": class1_stats["clusters_merged"],
        "unionfind_refused_count": uf.refused_count,
        "canonical_count": len(canonical_map),
        "alias_group_size_distribution": dict(group_size_hist),
        "elapsed_seconds": round(time.monotonic() - t0, 1),
    }

    write_outputs(
        config=config,
        canonical_map=canonical_map,
        field_pmid_index=field_pmid_index,
        field_pmid_env_index=field_pmid_env_index,
        side_tags=side_tags,
        stats=stats_out,
    )

    # Persist Class-1 auto-merge log for auditing (Supplementary
    # material: lists every cluster that was programmatically merged
    # by the SAMPLING-CONTEXT rule, not by the LLM).
    if class1_stats["log"]:
        c1_path = Path(config.output_file).with_name("step3_class1_merge_log.json")
        c1_log = [
            {"core": list(core), "merged_roots": roots}
            for core, roots in class1_stats["log"]
        ]
        _dump_json(c1_path, {
            "summary": {
                "unions_performed": class1_stats["unions_performed"],
                "clusters_merged": class1_stats["clusters_merged"],
            },
            "merges": c1_log,
        })
        LOGGER.info("Wrote Class-1 merge log: %s", c1_path)

    # ---------- Optional: canonical review post-processing ----------
    if review_path is not None:
        from .canonical_review import run_review_postprocess
        audit_path = Path(config.output_file).with_name(
            "step3_canonical_review_audit.json"
        )
        review_summary = run_review_postprocess(
            output_file=Path(config.output_file),
            review_decisions_file=review_path,
            audit_file=audit_path,
        )
        LOGGER.info("Canonical review applied: %s", review_summary)

    LOGGER.info("Step 3 complete. %s", stats_out)
