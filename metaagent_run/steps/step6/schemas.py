"""Step6 数据模型。

主要数据流：
  - Candidate: 内部 dataclass，扁平化 step5 metadata 后的单条记录
  - ProvenanceRecord: 对外 schema，落选候选的来源记录
  - ResolvedMetadataItem: 主输出 schema，仲裁选出的权威值
  - Step6Output: 顶层产物
"""

from dataclasses import dataclass, field as dc_field
from typing import List, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════
#  内部 — 扁平化候选
# ═══════════════════════════════════════════════════════════

@dataclass
class Candidate:
    """扁平化后单条 metadata 候选记录。

    内部使用，不对外暴露。pub_year 可以为 None（pmid_year.txt 中找不到）。
    """
    raw_accession: str
    canonical_slot: str        # mixs:xxx 或 internal:xxx
    raw_field: str             # 原始字段名（来自 step5 raw_field）
    value: str                 # 原始值字符串
    pmid: str
    pub_year: Optional[int] = None
    source_file: str = ""
    section_type: str = ""
    paragraph_index: int = -1
    extraction_modality: str = ""   # table_parse / llm_extract


# ═══════════════════════════════════════════════════════════
#  对外 schema
# ═══════════════════════════════════════════════════════════

class ProvenanceRecord(BaseModel):
    """一条证据的溯源信息（用于 alternate_sources）。"""
    pmid: str
    value: str
    raw_field: str
    source_file: str = ""
    section_type: str = ""
    paragraph_index: int = -1
    extraction_modality: str = ""


class ResolvedMetadataItem(BaseModel):
    """冲突解决后的一条权威记录。"""
    raw_accession: str
    canonical_slot: str
    raw_field: str
    value: str
    authoritative_pmid: str
    source_file: str = ""
    section_type: str = ""
    paragraph_index: int = -1
    extraction_modality: str = ""
    alternate_sources: List[ProvenanceRecord] = Field(default_factory=list)


class Step6Output(BaseModel):
    """step6 最终产物。"""
    resolved_items: List[ResolvedMetadataItem] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)
