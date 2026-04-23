"""Step6 仲裁器：按优先级链选出单一权威值。

仲裁规则（从高到低）：
  1. modality:     table_parse  >  llm_extract
  2. section_type: METHODS  >  其他    (仅 llm_extract 内部生效)
  3. pub_year:     早  >  晚；缺失视为 9999

PMID tiebreaker 已移除（设计文档 §4.4）。
"""

from collections import defaultdict
from typing import Dict, List, Tuple

from .config import (
    MAX_ALTERNATE_SOURCES,
    MODALITY_DEFAULT_PRIORITY,
    MODALITY_PRIORITY,
    PUB_YEAR_MISSING_VALUE,
    SECTION_DEFAULT_PRIORITY,
    SECTION_PRIORITY,
)
from .schemas import Candidate, ProvenanceRecord, ResolvedMetadataItem


def arbitration_key(c: Candidate) -> Tuple[int, int, int]:
    """按优先级链生成排序键（lexicographic ascending = 偏好）。

    返回 (modality_rank, section_rank, pub_year_rank) — 数值越小越优。
    """
    modality_rank = MODALITY_PRIORITY.get(
        c.extraction_modality, MODALITY_DEFAULT_PRIORITY,
    )
    section_rank = SECTION_PRIORITY.get(c.section_type, SECTION_DEFAULT_PRIORITY)
    pub_year_rank = c.pub_year if c.pub_year is not None else PUB_YEAR_MISSING_VALUE
    return (modality_rank, section_rank, pub_year_rank)


def group_candidates(
    candidates: List[Candidate],
) -> Dict[Tuple[str, str], List[Candidate]]:
    """按 (raw_accession, canonical_slot) 分组。"""
    groups: Dict[Tuple[str, str], List[Candidate]] = defaultdict(list)
    for c in candidates:
        groups[(c.raw_accession, c.canonical_slot)].append(c)
    return dict(groups)


def _to_provenance(c: Candidate) -> ProvenanceRecord:
    return ProvenanceRecord(
        pmid=c.pmid,
        value=c.value,
        raw_field=c.raw_field,
        source_file=c.source_file,
        section_type=c.section_type,
        paragraph_index=c.paragraph_index,
        extraction_modality=c.extraction_modality,
    )


def resolve_group(
    key: Tuple[str, str],
    candidates: List[Candidate],
    max_alternates: int = MAX_ALTERNATE_SOURCES,
) -> ResolvedMetadataItem:
    """对单组候选做仲裁。

    sorted_candidates[0] 为权威值；后续 max_alternates 条进入 alternate_sources。
    """
    sorted_cands = sorted(candidates, key=arbitration_key)
    top = sorted_cands[0]
    alternates = [
        _to_provenance(c) for c in sorted_cands[1 : 1 + max_alternates]
    ]
    return ResolvedMetadataItem(
        raw_accession=key[0],
        canonical_slot=key[1],
        raw_field=top.raw_field,
        value=top.value,
        authoritative_pmid=top.pmid,
        source_file=top.source_file,
        section_type=top.section_type,
        paragraph_index=top.paragraph_index,
        extraction_modality=top.extraction_modality,
        alternate_sources=alternates,
    )
