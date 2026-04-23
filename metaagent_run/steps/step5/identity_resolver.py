# -*- coding: utf-8 -*-
"""Phase A — Alias Discovery (simplified).

从论文文本中发现 accession 的别名/标签，合并到程序化构建的骨架 identity_map 中。
formal_name、parent_project、environment 已由 Phase 0 (build_identity_skeleton) 程序化完成。
"""

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from metaagent_run.core import (
    LLMClientProtocol,
    backoff_with_jitter,
    detect_truncation,
    continue_json_until_ok,
    extract_json_from_response_with_repair,
)

from .config import RuntimeConfig
from .schemas import IdentityMap, PaperContext, SampleIdentity
from .upstream_loader import UpstreamData

LOGGER = logging.getLogger(__name__)

from metaagent_run.core import INSDC_ACCESSION_RE as ACC_PATTERN


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
#  Formatting helpers
# ═══════════════════════════════════════════════════════════

def _format_accession_sections(sections: List[Dict[str, Any]]) -> str:
    """Concatenate accession-bearing section texts."""
    parts = []
    for sec in sections:
        sec_type = sec.get("section_type", "")
        idx = sec.get("index", 0)
        text = sec.get("text", "")
        if not text or not isinstance(text, str):
            continue
        # Limit each section to ~4000 chars
        if len(text) > 4000:
            text = text[:4000] + "\n... [truncated]"
        parts.append("### Section: %s (index %s)\n%s" % (sec_type, idx, text))
    return "\n\n".join(parts) if parts else "(no accession sections found)"


def _format_skeleton_context(identity_map: IdentityMap) -> str:
    """Format skeleton identity map for LLM context."""
    if not identity_map:
        return "(no samples identified)"
    lines = []
    for acc in sorted(identity_map.keys()):
        ident = identity_map[acc]
        name = ident.formal_name or "(no DB name)"
        parent = " (under %s)" % ident.parent_project if ident.parent_project else ""
        lines.append("- %s%s: DB name = \"%s\"" % (acc, parent, name))
    return "\n".join(lines)


def _format_verified_accessions(accessions: Set[str]) -> str:
    """Format verified accessions as a list."""
    if not accessions:
        return "(none)"
    return ", ".join(sorted(accessions))


# ═══════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════

async def resolve_identities(
    client: LLMClientProtocol,
    paper_ctx: PaperContext,
    sections: List[Dict[str, Any]],
    upstream: UpstreamData,
    config: RuntimeConfig,
    skeleton_identity_map: Optional[IdentityMap] = None,
    skeleton_alias_to_accession: Optional[Dict[str, str]] = None,
) -> Tuple[IdentityMap, Dict[str, str]]:
    """
    Phase A: Alias Discovery.

    Discovers additional aliases from paper text and merges them into the
    programmatically built skeleton identity_map.

    Returns:
        identity_map: Dict[str, SampleIdentity] keyed by accession
        alias_to_accession: Dict[str, str] mapping all names to accession
    """
    pmid = paper_ctx.pmid

    # Graceful degradation: if no skeleton provided, return empty
    if skeleton_identity_map is None:
        skeleton_identity_map = {}
    if skeleton_alias_to_accession is None:
        skeleton_alias_to_accession = {}

    # 1. Collect section texts (accession-bearing + metadata-bearing)
    # Phase A needs both: accession sections define accession-label links,
    # metadata sections define label meanings (e.g., "free-living (FL)")
    all_context_sections = paper_ctx.accession_sections + paper_ctx.metadata_sections
    accession_sections_text = _format_accession_sections(all_context_sections)

    # 2. Format skeleton context (show LLM what we already know)
    skeleton_context = _format_skeleton_context(skeleton_identity_map)

    # 3. Format verified accessions
    verified_accessions_list = _format_verified_accessions(paper_ctx.verified_accessions)

    # 4. Build prompt
    template = _load_prompt_template(config.prompt_identity)
    prompt_text = template.format(
        accession_sections_text=accession_sections_text,
        skeleton_context=skeleton_context,
        verified_accessions_list=verified_accessions_list,
    )

    # 4b. Append Step 2 label hints
    step2_labels = paper_ctx.step2_labels
    if step2_labels:
        labels_hint = (
            "\n\n## Pre-discovered Labels (from relation extraction)\n"
            "The following labels/aliases were detected in this paper's text:\n"
            + ", ".join(step2_labels)
            + "\n\nUse these as clues — some may be aliases for the verified accessions above. "
            "But do NOT blindly assign them; verify from the paper excerpts which accession each label refers to.\n"
        )
        prompt_text = prompt_text + labels_hint

    messages = [
        {"role": "system", "content": "You are a bioinformatics expert specializing in sample identification."},
        {"role": "user", "content": prompt_text},
    ]

    # 5. Call LLM with retry logic
    raw_samples = None
    last_error = None

    for attempt in range(config.retry_times):
        temp = config.retry_temps[min(attempt, len(config.retry_temps) - 1)]

        response_text = await _call_llm(client, messages, temp, config)
        if not response_text:
            last_error = "No response"
            LOGGER.warning("[Identity] PMID %s attempt %d: %s", pmid, attempt + 1, last_error)
            await asyncio.sleep(backoff_with_jitter(
                attempt, base=config.backoff_base, cap=config.backoff_cap,
            ))
            continue

        # 6. Parse JSON response
        parsed = extract_json_from_response_with_repair(
            response_text,
            stop_sentinel=config.stop_sentinel,
            target_keys=("samples",),
            enable_p0=True, enable_p1=False,
        )
        parsed_dict = None
        if isinstance(parsed, dict):
            parsed_dict = parsed
        elif isinstance(parsed, list):
            for e in parsed:
                if isinstance(e, dict) and "samples" in e:
                    parsed_dict = e
                    break
            if parsed_dict is None and parsed:
                parsed_dict = {"samples": [e for e in parsed if isinstance(e, dict)]}

        if not parsed_dict or "samples" not in parsed_dict:
            last_error = "JSON parse failed or missing 'samples' key"
            LOGGER.warning("[Identity] PMID %s attempt %d: %s", pmid, attempt + 1, last_error)
            await asyncio.sleep(backoff_with_jitter(
                attempt, base=config.backoff_base, cap=config.backoff_cap,
            ))
            continue

        raw_samples = parsed_dict.get("samples", [])
        if not isinstance(raw_samples, list):
            last_error = "'samples' is not a list"
            LOGGER.warning("[Identity] PMID %s attempt %d: %s", pmid, attempt + 1, last_error)
            await asyncio.sleep(backoff_with_jitter(
                attempt, base=config.backoff_base, cap=config.backoff_cap,
            ))
            continue

        break  # success

    if raw_samples is None:
        LOGGER.warning("[Identity] PMID %s: LLM failed after %d attempts, using skeleton only. Last: %s",
                        pmid, config.retry_times, last_error)
        # Graceful degradation: return deep copy of skeleton
        return _deep_copy_skeleton(skeleton_identity_map, skeleton_alias_to_accession)

    # 7. Merge LLM-discovered aliases into skeleton
    identity_map, alias_to_accession = _merge_llm_aliases(
        raw_samples, paper_ctx, skeleton_identity_map, skeleton_alias_to_accession,
    )

    LOGGER.info("[Identity] PMID %s: resolved %d accessions, %d aliases",
                pmid, len(identity_map), len(alias_to_accession))

    return identity_map, alias_to_accession


# ═══════════════════════════════════════════════════════════
#  Merge LLM aliases into skeleton
# ═══════════════════════════════════════════════════════════

def _deep_copy_skeleton(
    skeleton_map: IdentityMap,
    skeleton_aliases: Dict[str, str],
) -> Tuple[IdentityMap, Dict[str, str]]:
    """Deep copy skeleton for safe return."""
    new_map: IdentityMap = {}
    for acc, ident in skeleton_map.items():
        new_map[acc] = SampleIdentity(
            accession=ident.accession,
            formal_name=ident.formal_name,
            aliases=list(ident.aliases),
            parent_project=ident.parent_project,
            environment=ident.environment,
        )
    return new_map, dict(skeleton_aliases)


def _merge_llm_aliases(
    raw_samples: List[Dict[str, Any]],
    paper_ctx: PaperContext,
    skeleton_identity_map: IdentityMap,
    skeleton_alias_to_accession: Dict[str, str],
) -> Tuple[IdentityMap, Dict[str, str]]:
    """
    Merge LLM-discovered aliases into the skeleton identity_map.

    Only merges aliases for accessions that exist in verified_accessions.
    formal_name, parent_project, environment are NOT overwritten (owned by skeleton).
    """
    # Deep copy skeleton
    identity_map, alias_to_accession = _deep_copy_skeleton(
        skeleton_identity_map, skeleton_alias_to_accession,
    )

    for entry in raw_samples:
        if not isinstance(entry, dict):
            continue
        acc = str(entry.get("accession", "")).strip()
        if not acc or acc not in paper_ctx.verified_accessions:
            continue
        if acc not in identity_map:
            continue

        # Extract aliases from LLM output
        aliases = entry.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = [str(aliases)] if aliases else []
        aliases = [str(a).strip() for a in aliases if str(a).strip()]

        # Merge aliases (deduplicated)
        identity = identity_map[acc]
        existing = set(identity.aliases)
        if identity.formal_name:
            existing.add(identity.formal_name)
        for alias in aliases:
            if alias not in existing and alias != acc:
                identity.aliases.append(alias)
                existing.add(alias)
                # Register in alias_to_accession
                alias_to_accession[alias] = acc
                alias_to_accession[alias.lower()] = acc

    # Re-aggregate BioProject aliases from enriched children
    # Only add to identity.aliases for display; do NOT overwrite alias_to_accession
    # (child mappings have higher priority than project-level mappings)
    for acc, identity in identity_map.items():
        if not acc.startswith("PRJ"):
            continue
        existing = set(identity.aliases)
        for child_acc, child_ident in identity_map.items():
            if child_ident.parent_project == acc and child_acc != acc:
                for alias in child_ident.aliases:
                    if alias not in existing:
                        identity.aliases.append(alias)
                        existing.add(alias)

    return identity_map, alias_to_accession
