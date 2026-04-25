# -*- coding: utf-8 -*-
"""Phase B — Metadata Extraction.

B1+B2: 从结构化表格提取 metadata 并映射到 accession。
B3: LLM per-section extraction with identity map injection.
"""

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

from metaagent_run.core import (
    LLMClientProtocol,
    backoff_with_jitter,
    detect_truncation,
    continue_json_until_ok,
    extract_json_from_response_with_repair,
    split_text_with_offsets,
)

from .config import RuntimeConfig
from .schemas import IdentityMap, PaperContext, SampleIdentity
from .table_parser import StructuredTableParser
from .upstream_loader import GIANT_TABLE_MAX_COLS, UpstreamData

LOGGER = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Prompt template loader
# ═══════════════════════════════════════════════════════════

def _load_prompt_template(name: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
    with open(os.path.join(prompt_dir, name), "r", encoding="utf-8") as f:
        return f.read()


# ═══════════════════════════════════════════════════════════
#  LLM call helper (follows processor.py patterns)
# ═══════════════════════════════════════════════════════════

async def _call_llm(
    client: LLMClientProtocol,
    messages: List[Dict[str, str]],
    temp: float,
    config: RuntimeConfig,
) -> Optional[str]:
    """调用 LLM（流式优先，fallback 非流式）。"""
    resp = await client.chat_streaming_with_signals(messages, temperature_override=temp)
    if resp:
        text = resp.get("text")
        if isinstance(text, str) and text.strip():
            status = detect_truncation(
                text, bool(resp.get("saw_done", False)),
                resp.get("finish_reason"),
                stop_sentinel=config.stop_sentinel,
            )
            if status != "ok":
                continued = await continue_json_until_ok(
                    client, messages, text, client.max_tokens,
                    max_tokens_cap=config.max_tokens_cap,
                    stop_sentinel=config.stop_sentinel,
                    max_rounds=config.continuation_max_rounds,
                )
                if continued:
                    return continued
            return text
    # fallback
    return await client.chat(messages, temperature_override=temp)


# ═══════════════════════════════════════════════════════════
#  Target fields + aliases formatter (route Y: surface-form hints)
# ═══════════════════════════════════════════════════════════

_ALIAS_MAX_PER_TARGET = 15

# Module-level log file for step2 metadata_keys that didn't match any phase6
# target alias. Audited periodically to grow phase6_schema.yaml aliases.
_UNMATCHED_KEYS_LOG = Path(os.environ.get(
    "STEP5_UNMATCHED_KEYS_LOG",
    str(Path.cwd() / "step5_unmatched_step2_keys.log"),
))


_ALIAS_NORM_RE = re.compile(r"[\s/\-]+")


def _normalize_key(s: str) -> str:
    """Normalize a raw_key / alias / target name for robust matching.

    Lowercase, strip, collapse {space, hyphen, forward slash} to underscore,
    trim repeated/leading/trailing underscores. Makes 'Chl-a', 'chl a',
    'chl/a' all match 'chl_a'.
    """
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = _ALIAS_NORM_RE.sub("_", s)
    # collapse repeated underscores and strip
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _build_alias_to_target_index(
    target_fields: List[str],
    aliases_map: Dict[str, List[str]],
) -> Dict[str, str]:
    """Reverse index normalized_alias -> canonical target name.

    Target name itself is indexed as its own alias so a section mentioning
    the canonical form directly still matches.
    """
    idx: Dict[str, str] = {}
    for t in target_fields:
        key = _normalize_key(t)
        if key:
            idx.setdefault(key, t)
        for a in aliases_map.get(t, []) or []:
            key = _normalize_key(str(a))
            if key and key not in idx:
                idx[key] = t
    return idx


def _filter_targets_by_section_keys(
    target_fields: List[str],
    aliases_map: Dict[str, List[str]],
    section_metadata_keys: List[str],
    pmid: str = "",
    section_type: str = "",
    section_index: int = 0,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Narrow target_fields to only those matched by step2 metadata_keys_found.

    If the section has no step2 keys (empty list), return the full sets
    unchanged — a conservative fallback so sections without upstream hints
    don't silently produce empty extraction.

    Unmatched step2 keys are appended to _UNMATCHED_KEYS_LOG for later
    phase6 alias auditing.
    """
    if not section_metadata_keys:
        return target_fields, aliases_map

    alias_to_target = _build_alias_to_target_index(target_fields, aliases_map)
    relevant: set = set()
    unmatched: List[str] = []
    for k in section_metadata_keys:
        key = _normalize_key(str(k))
        if not key:
            continue
        t = alias_to_target.get(key)
        if t is not None:
            relevant.add(t)
        else:
            unmatched.append(str(k))

    # Log unmatched for later review (non-blocking, best effort)
    if unmatched:
        try:
            with open(_UNMATCHED_KEYS_LOG, "a", encoding="utf-8") as lf:
                for u in unmatched:
                    lf.write(f"{pmid}\t{section_type}\t{section_index}\t{u}\n")
        except Exception:
            pass  # never block extraction on log writes

    if not relevant:
        # All step2 keys missed phase6 aliases — fallback to full set rather
        # than empty prompt (would force LLM to extract nothing).
        return target_fields, aliases_map

    # Preserve original ordering (tier1 first, then tier2)
    filtered_fields = [t for t in target_fields if t in relevant]
    filtered_aliases = {t: aliases_map[t] for t in filtered_fields if t in aliases_map}
    return filtered_fields, filtered_aliases


def _format_targets_with_aliases(
    target_fields: List[str],
    aliases_map: Dict[str, List[str]],
) -> str:
    """Render `Target Metadata Fields` section for the section_extract prompt.

    For each target, append up to _ALIAS_MAX_PER_TARGET alternative surface
    forms in parentheses so the LLM can recognize them as filling the
    canonical target. Aliases are deduped, lower-cased, and prefer shorter
    strings (they are typically more common).

    Format:
      - collection_date (also: sampling_date, date, year, month, ...)
      - water_depth (also: depth, sampling_depth, bottom_depth, ...)
      - ph
      ...
    """
    lines: List[str] = []
    for field in target_fields:
        aliases = aliases_map.get(field) or []
        # dedupe (case-insensitive, drop target itself)
        seen = {field.lower().strip()}
        picked: List[str] = []
        # Prefer shorter aliases first (usually more common in literature)
        for a in sorted(aliases, key=lambda s: (len(s), s.lower())):
            al = str(a).strip()
            if not al or al.lower() in seen:
                continue
            seen.add(al.lower())
            picked.append(al)
            if len(picked) >= _ALIAS_MAX_PER_TARGET:
                break
        if picked:
            lines.append(f"- {field} (also: {', '.join(picked)})")
        else:
            lines.append(f"- {field}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  Dynamic metadata keywords builder
# ═══════════════════════════════════════════════════════════

def build_metadata_keywords(
    target_fields: List[str],
    target_field_aliases: Dict[str, List[str]],
) -> Set[str]:
    """Build dynamic metadata keyword set from target field names + their
    aliases (sourced from env6 phase6 output). Keywords filter table column
    headers in the structured-table parser."""
    keywords: Set[str] = set()

    def _add_tokens(name: str) -> None:
        norm = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
        for token in norm.split("_"):
            if len(token) >= 2:
                keywords.add(token)

    for field in target_fields:
        _add_tokens(field)
    for aliases in target_field_aliases.values():
        for alias in aliases:
            _add_tokens(alias)

    _TOO_GENERIC = {
        "of", "or", "and", "the", "in", "on", "at", "to", "by", "is",
        "from", "for", "with", "not", "no", "an", "as",
        "total", "mean", "max", "min", "number", "value", "level",
        "analysis", "method", "type", "source", "data", "other",
    }
    keywords -= _TOO_GENERIC

    # Remove tokens that overlap with accession/label column keywords
    # to prevent metadata_keywords from intercepting label/accession columns
    from .table_parser import _ACC_COL_KEYWORDS, _LABEL_COL_KEYWORDS
    keywords -= _ACC_COL_KEYWORDS
    keywords -= _LABEL_COL_KEYWORDS

    return keywords


# ═══════════════════════════════════════════════════════════
#  Semantic label matching for transposed tables
# ═══════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════
#  Fuzzy matching helper
# ═══════════════════════════════════════════════════════════

def _fuzzy_match_label_to_accession(
    label: str,
    identity_map: IdentityMap,
    alias_to_accession: Dict[str, str],
) -> Optional[str]:
    """Try to match a table label to an accession via fuzzy name matching.

    When scores are tied, prefer sample-level accessions (SAMN/SRX/SRR/ERR)
    over project-level accessions (PRJ*), since table labels usually refer
    to individual samples rather than entire projects.
    """
    # Normalize label
    label_lower = label.strip().lower()

    # Exact match first (case-insensitive)
    lower_to_original = {k.lower(): k for k in alias_to_accession}
    if label_lower in lower_to_original:
        return alias_to_accession[lower_to_original[label_lower]]

    # Token-based fuzzy match
    label_tokens = set(re.sub(r'[^\w\s]', ' ', label_lower).split())
    if not label_tokens:
        return None

    best_acc = None
    best_score = 0.0
    best_is_project = True  # tie-breaker: prefer non-project

    for acc, identity in identity_map.items():
        is_project = acc.startswith("PRJ")
        for name in identity.all_names:
            name_tokens = set(re.sub(r'[^\w\s]', ' ', name.lower()).split())
            if not name_tokens:
                continue
            overlap = label_tokens & name_tokens
            score = len(overlap) / min(len(label_tokens), len(name_tokens))
            if score < 0.5:
                continue
            # Prefer higher score; on tie, prefer non-project accession
            if (score > best_score) or (score == best_score and best_is_project and not is_project):
                best_score = score
                best_acc = acc
                best_is_project = is_project

    return best_acc


# ═══════════════════════════════════════════════════════════
#  Identity context builder (grouped by parent_project)
# ═══════════════════════════════════════════════════════════

def _format_sample_line(acc: str, identity: "SampleIdentity") -> str:
    """Format a single sample identity line for the prompt."""
    names = [identity.formal_name] + identity.aliases if identity.formal_name else identity.aliases
    names_str = ", ".join(names) if names else acc
    return "  - %s: %s" % (acc, names_str)


def _build_grouped_identity_context(identity_map: IdentityMap) -> str:
    """Build identity context grouped by parent_project.

    When samples share a parent_project, they are shown under that project
    header so the LLM understands which samples belong to the same experiment
    group. Project-level accessions (PRJ*) that have child samples are marked
    as umbrella entries.
    """
    from collections import defaultdict

    # 1. Identify project-level accessions and group children
    project_accs = {
        acc for acc in identity_map if acc.startswith("PRJ")
    }
    children_by_project: Dict[str, List[str]] = defaultdict(list)
    orphans: List[str] = []

    for acc, identity in identity_map.items():
        if acc in project_accs:
            continue
        parent = identity.parent_project
        if parent and parent in project_accs:
            children_by_project[parent].append(acc)
        else:
            orphans.append(acc)

    lines: List[str] = []

    # 2. Render project groups
    for prj_acc in sorted(project_accs):
        prj_identity = identity_map[prj_acc]
        children = children_by_project.get(prj_acc, [])

        if children:
            # Project has child samples — mark it as umbrella
            lines.append("### Project %s (contains %d samples below)" % (prj_acc, len(children)))
            lines.append(_format_sample_line(prj_acc, prj_identity))
            for child_acc in sorted(children):
                lines.append(_format_sample_line(child_acc, identity_map[child_acc]))
        else:
            # Project with no child samples — standalone
            lines.append("### Project %s" % prj_acc)
            lines.append(_format_sample_line(prj_acc, prj_identity))
        lines.append("")  # blank line between groups

    # 3. Render orphan samples (no known parent project)
    if orphans:
        lines.append("### Other samples")
        for acc in sorted(orphans):
            lines.append(_format_sample_line(acc, identity_map[acc]))
        lines.append("")

    return "\n".join(lines).rstrip()


# ═══════════════════════════════════════════════════════════
#  Phase B1+B2: Table metadata extraction
# ═══════════════════════════════════════════════════════════

def _map_relations_to_accessions(
    relations: List[Any],
    identity_map: IdentityMap,
    alias_to_accession: Dict[str, str],
) -> Dict[str, List[str]]:
    """Map AtomicRelations to accessions, return {acc: ["field: value", ...]}."""
    mapped: Dict[str, List[str]] = {}
    for rel in relations:
        metadata = rel.metadata
        if not metadata:
            continue

        matched_acc = None
        if rel.accession:
            matched_acc = rel.accession
        elif rel.label:
            # 1. Exact match
            matched_acc = alias_to_accession.get(rel.label)
            if matched_acc is None:
                matched_acc = alias_to_accession.get(rel.label.lower())
            # 2. Fuzzy token match (fallback)
            if matched_acc is None:
                matched_acc = _fuzzy_match_label_to_accession(
                    rel.label, identity_map, alias_to_accession,
                )

        if matched_acc is None:
            continue

        if matched_acc not in mapped:
            mapped[matched_acc] = []
        mapped[matched_acc].extend(metadata)
    return mapped


def _count_mapped_metadata(mapped: Dict[str, List[str]]) -> int:
    """Count total metadata items across all accessions."""
    return sum(len(items) for items in mapped.values())


def extract_table_metadata(
    sections: List[Dict[str, Any]],
    identity_map: IdentityMap,
    alias_to_accession: Dict[str, str],
    table_parser: StructuredTableParser,
    paper_ctx: PaperContext,
    pmid: str,
    max_cols: int = GIANT_TABLE_MAX_COLS,
) -> Dict[str, List[str]]:
    """
    Phase B1+B2: Extract metadata from structured tables and map to accessions.

    Deterministic keyword-based approach (zero LLM calls). Callers from
    process_paper pass paper_ctx-filtered sections (already without giant
    tables); the inner max_cols check below is a defensive guard for
    out-of-pipeline callers that bypass build_paper_context.

    - Layer 1: normalize_table_text (in table_parser)
    - Layer 2: column count hard cutoff (> max_cols → skip)
    - Dual-run strategy: try both normal and transposed, keep whichever yields more metadata

    Returns: {accession: ["field: value", ...]}
    """
    result: Dict[str, List[str]] = {}
    target_fields = paper_ctx.tier1_fields + paper_ctx.tier2_fields

    metadata_keywords = build_metadata_keywords(
        target_fields, paper_ctx.target_field_aliases,
    )

    for section in sections:
        # Layer 1: is_structured_table uses normalize_table_text internally
        if not table_parser.is_structured_table(section):
            continue

        # Layer 2: column count hard cutoff
        n_cols = table_parser.count_columns(section)
        if n_cols > max_cols:
            LOGGER.info("[TableMeta] PMID %s: skipping table with %d columns (> %d)",
                        pmid, n_cols, max_cols)
            continue

        # Dual-run: try both normal and transposed, pick the one with more metadata
        normal_relations = table_parser.extract(
            section, target_fields, metadata_keywords,
            pmid=pmid, verified_accessions=paper_ctx.verified_accessions,
        )
        normal_mapped = _map_relations_to_accessions(
            normal_relations, identity_map, alias_to_accession,
        )

        transposed_relations = table_parser.extract_transposed(
            section, target_fields, metadata_keywords,
            pmid=pmid, verified_accessions=paper_ctx.verified_accessions,
        )
        transposed_mapped = _map_relations_to_accessions(
            transposed_relations, identity_map, alias_to_accession,
        )

        normal_count = _count_mapped_metadata(normal_mapped)
        transposed_count = _count_mapped_metadata(transposed_mapped)

        if transposed_count > normal_count:
            chosen = transposed_mapped
            LOGGER.info("[TableMeta] PMID %s: transposed wins (%d > %d metadata)",
                        pmid, transposed_count, normal_count)
        else:
            chosen = normal_mapped
            if normal_count > 0:
                LOGGER.info("[TableMeta] PMID %s: normal wins (%d >= %d metadata)",
                            pmid, normal_count, transposed_count)

        for acc, items in chosen.items():
            if acc not in result:
                result[acc] = []
            result[acc].extend(items)

    # Deduplicate metadata per accession
    for acc in result:
        seen: Set[str] = set()
        deduped: List[str] = []
        for item in result[acc]:
            normalized = item.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                deduped.append(item.strip())
        result[acc] = deduped

    LOGGER.info("[TableMeta] PMID %s: extracted table metadata for %d accessions",
                pmid, len(result))

    return result


# ═══════════════════════════════════════════════════════════
#  Phase B3 (v2): Per-section LLM extraction with identity map
# ═══════════════════════════════════════════════════════════

async def extract_section_metadata(
    client: LLMClientProtocol,
    section: Dict[str, Any],
    identity_map: IdentityMap,
    paper_ctx: PaperContext,
    config: RuntimeConfig,
) -> Dict[str, List[str]]:
    """
    Phase B3-LLM (v2): Extract metadata from ONE section for ALL accessions.

    The LLM receives the identity map and can directly assign metadata to accessions.

    Returns: {accession: ["field: value", ...]}
    """
    text = section.get("text", "")
    if not text or len(text.strip()) < 50:
        return {}

    sec_type = section.get("section_type", "")
    sec_idx = section.get("index", 0)

    # Build identity context for prompt — grouped by parent_project
    identity_context = _build_grouped_identity_context(identity_map)

    # Build target fields (with aliases block for path Y: surface-form hints)
    # Filter by step2 metadata_keys_found for this section to keep the prompt
    # focused on targets actually referenced here (big token saving).
    target_fields_all = paper_ctx.tier1_fields + paper_ctx.tier2_fields
    aliases_map_all = getattr(paper_ctx, "target_field_aliases", {}) or {}
    section_keys = (getattr(paper_ctx, "section_metadata_keys", {}) or {}).get(
        (sec_type, sec_idx), []
    )
    target_fields, aliases_map = _filter_targets_by_section_keys(
        target_fields_all,
        aliases_map_all,
        section_keys,
        pmid=paper_ctx.pmid,
        section_type=sec_type,
        section_index=sec_idx,
    )
    if len(target_fields) < len(target_fields_all):
        logger.info(
            "[%s %s:%d] target filter: %d step2 keys → %d targets (from %d)",
            paper_ctx.pmid, sec_type, sec_idx,
            len(section_keys), len(target_fields), len(target_fields_all),
        )
    target_fields_str = _format_targets_with_aliases(target_fields, aliases_map)

    # Split text into chunks instead of hard-truncating
    chunk_entries = split_text_with_offsets(text, chunk_size=config.text_chunk_size, overlap=config.text_overlap)
    if not chunk_entries:
        return {}

    template = _load_prompt_template(config.prompt_section_extract)
    merged_result: Dict[str, List[str]] = {}

    for chunk_idx, (chunk_text, _chunk_start, _chunk_end) in enumerate(chunk_entries):
        # Build prompt for this chunk
        prompt_text = template.format(
            identity_context=identity_context,
            target_fields=target_fields_str,
            section_type=sec_type,
            section_index=sec_idx,
            section_text=chunk_text,
        )

        messages = [
            {"role": "system", "content": "You are a bioinformatics expert specializing in metadata extraction from scientific literature."},
            {"role": "user", "content": prompt_text},
        ]

        # Call LLM with retry for this chunk
        last_error = None
        chunk_result: Optional[Dict[str, List[str]]] = None
        for attempt in range(config.retry_times):
            temp = config.retry_temps[min(attempt, len(config.retry_temps) - 1)]

            response_text = await _call_llm(client, messages, temp, config)
            if not response_text:
                last_error = "No response"
                await asyncio.sleep(backoff_with_jitter(attempt, base=config.backoff_base, cap=config.backoff_cap))
                continue

            parsed = extract_json_from_response_with_repair(
                response_text,
                stop_sentinel=config.stop_sentinel,
                target_keys=("results",),
                enable_p0=True, enable_p1=False,
            )

            parsed_dict = None
            if isinstance(parsed, dict):
                parsed_dict = parsed
            elif isinstance(parsed, list):
                for e in parsed:
                    if isinstance(e, dict) and "results" in e:
                        parsed_dict = e
                        break

            if not parsed_dict or "results" not in parsed_dict:
                last_error = "JSON parse failed"
                await asyncio.sleep(backoff_with_jitter(attempt, base=config.backoff_base, cap=config.backoff_cap))
                continue

            raw_results = parsed_dict.get("results", [])
            if not isinstance(raw_results, list):
                last_error = "'results' not a list"
                await asyncio.sleep(backoff_with_jitter(attempt, base=config.backoff_base, cap=config.backoff_cap))
                continue

            # Parse results: each entry has accession + metadata list
            chunk_result = {}
            for entry in raw_results:
                if not isinstance(entry, dict):
                    continue
                acc = str(entry.get("accession", "")).strip()
                if not acc or acc not in identity_map:
                    continue
                metadata = entry.get("metadata", [])
                if not isinstance(metadata, list):
                    continue
                items = []
                for m in metadata:
                    if isinstance(m, dict):
                        field = str(m.get("field", "")).strip()
                        value = str(m.get("value", "")).strip()
                        if field and value:
                            items.append("%s: %s" % (field, value))
                    elif isinstance(m, str) and ":" in m:
                        items.append(m.strip())
                if items:
                    chunk_result[acc] = chunk_result.get(acc, []) + items
            break

        if chunk_result is None:
            LOGGER.warning("[SectionMeta] %s::%s chunk%d: failed after %d attempts. Last: %s",
                           sec_type, sec_idx, chunk_idx, config.retry_times, last_error)
            continue

        # Merge chunk results into overall result
        for acc, items in chunk_result.items():
            if acc not in merged_result:
                merged_result[acc] = []
            merged_result[acc].extend(items)

    # Deduplicate metadata per accession
    for acc in merged_result:
        seen: Set[str] = set()
        deduped: List[str] = []
        for item in merged_result[acc]:
            normalized = item.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                deduped.append(item.strip())
        merged_result[acc] = deduped

    LOGGER.info("[SectionMeta] %s::%s: extracted metadata for %d accessions across %d chunks",
                sec_type, sec_idx, len(merged_result), len(chunk_entries))
    return merged_result

