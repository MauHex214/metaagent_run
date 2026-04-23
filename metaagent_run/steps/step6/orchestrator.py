"""Step6 主流程：load → flatten → group → resolve → write."""

import json
import logging
from collections import Counter
from typing import Any, Dict, List

from .config import RuntimeConfig
from .resolver import group_candidates, resolve_group
from .schemas import Candidate, ResolvedMetadataItem, Step6Output
from .upstream_loader import flatten_candidates, load_upstream

LOGGER = logging.getLogger(__name__)


def _build_stats(
    candidates: List[Candidate],
    groups: Dict,
    resolved: List[ResolvedMetadataItem],
) -> Dict[str, Any]:
    """构建运行统计。"""
    modality_counter = Counter(r.extraction_modality for r in resolved)
    section_counter = Counter(r.section_type for r in resolved)
    group_size_hist = Counter(len(v) for v in groups.values())
    alt_size_hist = Counter(len(r.alternate_sources) for r in resolved)
    canonical_namespace_counter = Counter(
        r.canonical_slot.split(":", 1)[0] if ":" in r.canonical_slot else "unknown"
        for r in resolved
    )
    # 候选有跨论文（不同 PMID）的 group 数量
    cross_paper_groups = sum(
        1 for v in groups.values() if len({c.pmid for c in v}) > 1
    )
    return {
        "total_candidates": len(candidates),
        "total_groups": len(groups),
        "resolved_count": len(resolved),
        "modality_selected_count": dict(modality_counter),
        "section_type_selected_count": dict(section_counter),
        "canonical_namespace_count": dict(canonical_namespace_counter),
        "group_size_distribution": dict(group_size_hist),
        "alternate_sources_size_distribution": dict(alt_size_hist),
        "cross_paper_group_count": cross_paper_groups,
    }


def run(cfg: RuntimeConfig) -> None:
    """主流程入口。"""
    LOGGER.info("Loading upstream data...")
    ud = load_upstream(cfg)

    LOGGER.info("Flattening candidates...")
    candidates = flatten_candidates(ud)
    if not candidates:
        LOGGER.warning("No candidates found. Exiting.")
        return
    LOGGER.info("  %d candidates total", len(candidates))

    LOGGER.info("Grouping by (raw_accession, canonical_slot)...")
    groups = group_candidates(candidates)
    LOGGER.info("  %d unique (raw_acc, slot) groups", len(groups))

    LOGGER.info("Resolving conflicts...")
    resolved = [
        resolve_group(key, members, cfg.max_alternate_sources)
        for key, members in groups.items()
    ]

    stats = _build_stats(candidates, groups, resolved)
    output = Step6Output(resolved_items=resolved, stats=stats)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / cfg.output_file
    stats_path = cfg.output_dir / cfg.stats_file
    out_path.write_text(
        json.dumps(output.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Wrote %s (%d resolved items)", out_path, len(resolved))
    LOGGER.info("Wrote %s", stats_path)
