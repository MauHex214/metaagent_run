"""Step5 orchestrator: paper-level pipeline (post-MIxS-decoupling refactor).

Phase A: Identity Resolution — resolve_identities() per paper
Phase B: Metadata Extraction
  - B1+B2: Table metadata extraction (structured tables)
  - B3: LLM per-section metadata extraction with identity map injection
Phase C2: Per-paper finalization → FinalSampleRecord

The earlier Phase C1 (LLM field-name → MIxS slot mapping) was removed: MIxS
alignment now happens once at the schema level (env_field_pipeline phase8),
not per-paper at extraction time.
"""

import asyncio
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm

from metaagent_run.core import AsyncLocalModelClient, load_json_items

from .config import RuntimeConfig, load_runtime_config
from .identity_resolver import resolve_identities
from .metadata_extractor import extract_table_metadata, extract_section_metadata
from .processor import _normalize_metadata_list
from .schemas import (
    FinalSampleRecord,
    IdentityMap,
    PaperContext,
    PaperOutput,
    SampleIdentity,
)
from .table_parser import StructuredTableParser
from .upstream_loader import UpstreamData, build_paper_context, build_identity_skeleton, load_upstream


# == Utility functions ==


def _group_by_pmid(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(item.get("pmid", "unknown"))].append(item)
    return dict(groups)


# == Debug output ==


def _init_debug_dir(output_file: str) -> str:
    """Create debug directory alongside output_file."""
    debug_dir = os.path.join(os.path.dirname(output_file) or ".", "debug")
    os.makedirs(debug_dir, exist_ok=True)
    return debug_dir


def _save_debug_json(debug_dir: str, filename: str, data: Any):
    """Save debug JSON file."""
    path = os.path.join(debug_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _save_paper_debug(
    debug_dir: str,
    pmid: str,
    paper_ctx: PaperContext,
    identity_map: IdentityMap,
    alias_to_accession: Dict[str, str],
    table_meta: Dict[str, List[str]],
    llm_meta: Dict[str, List[str]],
    samples_metadata: Dict[str, List[str]],
):
    """Save all intermediate results for a single paper."""
    debug_data = {
        "pmid": pmid,
        "phase_a_identity_map": {
            acc: {
                "formal_name": ident.formal_name,
                "aliases": ident.aliases,
                "parent_project": ident.parent_project,
                "environment": ident.environment,
            }
            for acc, ident in identity_map.items()
        },
        "phase_a_alias_to_accession": alias_to_accession,
        "phase_b_table_metadata": table_meta,
        "phase_b_llm_metadata": llm_meta,
        "merged_samples_metadata": samples_metadata,
        "paper_context": {
            "verified_accessions": sorted(paper_ctx.verified_accessions),
            "accession_to_env": paper_ctx.accession_to_env,
            "tier1_fields": paper_ctx.tier1_fields,
            "tier2_fields": paper_ctx.tier2_fields,
        },
    }
    path = os.path.join(debug_dir, "paper_%s.json" % pmid)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=2)


# == Checkpoint ==


def _load_checkpoint(path: str) -> Dict[str, PaperOutput]:
    done: Dict[str, PaperOutput] = {}
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                po = PaperOutput.model_validate(obj)
                done[po.pmid] = po
            except Exception:
                continue
    return done


def _save_checkpoint_line(path: str, paper: PaperOutput):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(paper.model_dump(), ensure_ascii=False) + "\n")
        f.flush()


# == Paper processing (Phase A + Phase B) ==


async def process_paper(
    pmid: str,
    sections: List[Dict],
    client: AsyncLocalModelClient,
    upstream: UpstreamData,
    config: RuntimeConfig,
    debug_dir: str = "",
) -> Dict[str, Any]:
    """Process a single paper through Phase A + Phase B."""

    # Phase 0: Build paper context (reuse existing)
    paper_ctx = build_paper_context(pmid, sections, upstream)

    if not paper_ctx.verified_accessions and not paper_ctx.accession_sections:
        # No accessions found - skip
        return {
            "pmid": pmid,
            "paper_ctx": paper_ctx,
            "identity_map": {},
            "samples_metadata": {},
            "stats": {
                "total_sections": len(sections),
                "accession_sections": 0,
                "metadata_sections": 0,
                "verified_accessions": 0,
                "identity_map_size": 0,
                "table_metadata_accessions": 0,
                "llm_metadata_accessions": 0,
            },
        }


    # Phase 0b: Build identity skeleton (programmatic — no LLM)
    skeleton_identity_map, skeleton_alias_to_accession = build_identity_skeleton(
        paper_ctx, upstream, max_samples=config.max_samples_for_skeleton
    )

    # Phase A: Alias Discovery (LLM enrichment of skeleton).
    # No acc-count threshold: paper-centric design relies on giant-table
    # filtering at section-ingestion time (build_paper_context) to keep the
    # acc set bounded by what the paper actually deposits.
    identity_map, alias_to_accession = await resolve_identities(
        client, paper_ctx, sections, upstream, config,
        skeleton_identity_map=skeleton_identity_map,
        skeleton_alias_to_accession=skeleton_alias_to_accession,
    )


    # Phase B1+B2: Table metadata extraction
    # Use paper_ctx.metadata_sections — already filtered for the 3 metadata-
    # bearing relations and giant tables (in build_paper_context). No need
    # to re-filter here; no need to pass max_cols (the giant-table cutoff is
    # owned by build_paper_context's GIANT_TABLE_MAX_COLS).
    table_parser = StructuredTableParser()
    table_meta = extract_table_metadata(
        paper_ctx.metadata_sections, identity_map, alias_to_accession,
        table_parser, paper_ctx, pmid,
    )

    # Phase B3: LLM metadata extraction per section (with identity map)
    llm_meta: Dict[str, List[str]] = {}

    # Filter out structured tables (already handled in B1+B2)
    llm_sections = [sec for sec in paper_ctx.metadata_sections
                    if not table_parser.is_structured_table(sec)]

    # Concurrent section extraction (bounded by section_concurrency)
    section_sem = asyncio.Semaphore(config.section_concurrency)

    async def _extract_one_section(sec):
        async with section_sem:
            return await extract_section_metadata(
                client, sec, identity_map, paper_ctx, config
            )

    if llm_sections:
        results = await asyncio.gather(*[_extract_one_section(sec) for sec in llm_sections])
        for sec_result in results:
            for acc, items in sec_result.items():
                if acc not in llm_meta:
                    llm_meta[acc] = []
                llm_meta[acc].extend(items)

    # Merge table + LLM metadata (keep ALL items, tag source, no field-level dedup)
    # Format: "source||field: value" where source is "table_parse" or "llm_extract"
    samples_metadata: Dict[str, List[str]] = {}
    for acc in identity_map:
        merged: List[str] = []
        # Table metadata
        for item in table_meta.get(acc, []):
            merged.append("table_parse||" + item)
        # LLM metadata
        for item in llm_meta.get(acc, []):
            merged.append("llm_extract||" + item)
        # Deduplicate exact same source+field+value
        seen: set = set()
        deduped: List[str] = []
        for item in merged:
            key = item.strip().lower()
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        if deduped:
            samples_metadata[acc] = deduped

    # Save debug
    if debug_dir:
        _save_paper_debug(
            debug_dir, pmid, paper_ctx, identity_map,
            alias_to_accession, table_meta, llm_meta, samples_metadata,
        )

    return {
        "pmid": pmid,
        "paper_ctx": paper_ctx,
        "identity_map": identity_map,
        "samples_metadata": samples_metadata,
        "stats": {
            "total_sections": len(sections),
            "accession_sections": len(paper_ctx.accession_sections),
            "metadata_sections": len(paper_ctx.metadata_sections),
            "verified_accessions": len(paper_ctx.verified_accessions),
            "identity_map_size": len(identity_map),
            "table_metadata_accessions": len(table_meta),
            "llm_metadata_accessions": len(llm_meta),
        },
    }


# == Phase C2: Finalize paper ==


def finalize_paper(
    intermediate: Dict[str, Any],
    upstream: UpstreamData,
) -> PaperOutput:
    """Phase C2: Final normalization."""
    pmid = intermediate["pmid"]
    paper_ctx = intermediate["paper_ctx"]
    identity_map = intermediate.get("identity_map", {})
    samples_metadata = intermediate.get("samples_metadata", {})

    final_samples: List[FinalSampleRecord] = []
    for acc, metadata_raw in samples_metadata.items():
        identity = identity_map.get(acc)
        if not identity:
            continue

        normalized = _normalize_metadata_list(metadata_raw)
        if not normalized:
            continue

        env = paper_ctx.accession_to_env.get(acc, identity.environment)

        final_samples.append(FinalSampleRecord(
            pmid=pmid,
            accession=acc,
            environment=env,
            labels=sorted(identity.all_names),
            metadata=normalized,
        ))

    stats = intermediate.get("stats", {})
    stats["final_samples"] = len(final_samples)

    return PaperOutput(
        pmid=pmid,
        environment=paper_ctx.environment,
        samples=final_samples,
        stats=stats,
    )


# == Output ==


def _write_final_output(path: str, papers: List[PaperOutput]):
    data = [p.model_dump() for p in papers]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _log_failure(path: str, pmid: str, error: str):
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "pmid": pmid,
        "error": error,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# == Main entry point ==


async def main_async(
    input_file: str,
    output_file: str,
    paper_concurrency: Optional[int] = None,
    runtime_config: Optional[RuntimeConfig] = None,
) -> None:
    config = runtime_config or load_runtime_config()
    resolved_concurrency = paper_concurrency or config.paper_concurrency

    try:
        items = load_json_items(input_file)
    except Exception as e:
        print("Failed to load input: %s" % e)
        return

    print("Loading upstream data...")
    upstream = load_upstream(
        relation_file=config.relation_file,
        accession_file=config.accession_file,
        accession_list_file=config.accession_list_file,
        expanded_metadata_file=config.expanded_metadata_file,
        env_tag_file=config.env_tag_file,
        env_extraction_targets_file=config.env_extraction_targets_file,
    )

    papers = _group_by_pmid(items)
    print("%d PMIDs, %d total sections" % (len(papers), len(items)))

    # Debug directory
    debug_dir = _init_debug_dir(output_file)
    print("Debug output: %s" % debug_dir)

    checkpoint_file = output_file + ".checkpoint.jsonl"
    done = _load_checkpoint(checkpoint_file)
    todo_pmids = [p for p in papers if p not in done]
    print("Checkpoint: %d done, %d remaining" % (len(done), len(todo_pmids)))

    if not todo_pmids:
        print("All papers already processed.")
        _write_final_output(output_file, list(done.values()))
        return

    semaphore = asyncio.Semaphore(resolved_concurrency)
    failed_count = 0
    failed_lock = asyncio.Lock()

    intermediates: Dict[str, Dict[str, Any]] = {}
    intermediates_lock = asyncio.Lock()

    pbar = tqdm(total=len(todo_pmids), desc="Phase A+B")

    async with AsyncLocalModelClient(
        base_url=config.base_url,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=config.api_key,
        stop_sentinel=config.stop_sentinel,
        api_style=config.api_style,
        azure_api_version=config.azure_api_version,
        azure_deployment=config.azure_deployment,
        auth_mode=config.auth_mode,
    ) as client:

        async def worker(pmid: str):
            nonlocal failed_count
            async with semaphore:
                try:
                    result = await process_paper(
                        pmid, papers[pmid], client, upstream, config,
                        debug_dir=debug_dir,
                    )
                    async with intermediates_lock:
                        intermediates[pmid] = result
                except Exception as e:
                    tqdm.write("Paper %s failed: %s" % (pmid, e))
                    async with failed_lock:
                        failed_count += 1
                        _log_failure(config.failed_log_file, pmid, str(e))
                finally:
                    pbar.update(1)

        await asyncio.gather(*(worker(pmid) for pmid in todo_pmids))
        pbar.close()

    # Phase C2: Per-paper finalization
    completed_papers: Dict[str, PaperOutput] = dict(done)
    for pmid, inter in intermediates.items():
        paper_output = finalize_paper(inter, upstream)
        completed_papers[pmid] = paper_output
        _save_checkpoint_line(checkpoint_file, paper_output)

    _write_final_output(output_file, list(completed_papers.values()))

    total_samples = sum(len(p.samples) for p in completed_papers.values())
    total_with_samples = sum(1 for p in completed_papers.values() if p.samples)
    print("Summary: %d papers, %d with samples, %d total samples" % (
        len(completed_papers), total_with_samples, total_samples,
    ))

    if failed_count:
        print("%d papers failed. See %s" % (failed_count, config.failed_log_file))
    print("Done. %d papers -> %s" % (len(completed_papers), output_file))
