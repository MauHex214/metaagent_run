"""Step5 数据模型：原子关系 → 样本记录 → 最终输出。"""

from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field



# ═══════════════════════════════════════════════════════════
#  Phase 0 — Paper 级上下文
# ═══════════════════════════════════════════════════════════

@dataclass
class PaperContext:
    """Per-paper context built from upstream products."""
    pmid: str = ""
    verified_accessions: Set[str] = dc_field(default_factory=set)   # step3 ∩ 外部DB ∩ 非Others
    accession_to_env: Dict[str, str] = dc_field(default_factory=dict)  # accession → env_tag
    accession_sections: List[Dict[str, Any]] = dc_field(default_factory=list)  # step2: accession-*
    metadata_sections: List[Dict[str, Any]] = dc_field(default_factory=list)   # step2: label-metadata
    environment: str = "unknown"             # paper 主导环境（投票）
    tier1_fields: List[str] = dc_field(default_factory=list)   # 所有目标环境 tier1 并集
    tier2_fields: List[str] = dc_field(default_factory=list)   # 所有目标环境 tier2 并集
    # Aliases per target field (surface-form synonyms collected from phase6 output).
    # Used by metadata_extractor to render targets+aliases into the section_extract prompt,
    # so the LLM can map alternative raw_key forms (e.g. "sampling_date") back to the
    # canonical target (e.g. "collection_date"). Keys cover both tier1 and tier2.
    target_field_aliases: Dict[str, List[str]] = dc_field(default_factory=dict)
    # Per-section metadata_keys_found from step2 output. Used to filter the
    # target_fields block down to only those targets actually referenced in
    # each section (matched via aliases), reducing prompt size.
    # Key: (section_type, index). Value: list of raw metadata key names.
    section_metadata_keys: Dict[Any, List[str]] = dc_field(default_factory=dict)
    step2_labels: List[str] = dc_field(default_factory=list)   # Step 2 discovered labels


# ═══════════════════════════════════════════════════════════
#  Layer 1 — Section 级原子关系（表格解析 / LLM 提取共用输出）
# ═══════════════════════════════════════════════════════════

class AtomicRelation(BaseModel):
    """一条从单个 section 中提取的最小关系单元。"""
    pmid: str
    section_key: str                          # "supplementary::2"
    relation_type: str                        # accession_label | accession_metadata | label_metadata | accession_label_metadata
    accession: Optional[str] = None
    label: Optional[str] = None
    metadata: List[str] = Field(default_factory=list)  # ["depth: 25.0 m", ...]

    source: str = "llm_extract"               # table_parse | llm_extract




# ═══════════════════════════════════════════════════════════
#  Layer 3 — 最终输出
# ═══════════════════════════════════════════════════════════

class NormalizedMetadataItem(BaseModel):
    mixs_slot: Optional[str] = None       # MIxS 标准 slot 名, None 表示未映射
    raw_field: str                         # 原始字段名
    value: str                             # 值
    source: str = ""                       # 来源: "table_parse" / "llm_extract"

class FinalSampleRecord(BaseModel):
    """最终产出的 accession 粒度记录。保留原始 accession 层级。"""
    pmid: str
    accession: str
    environment: str = "unknown"           # accession 级别的环境标注
    labels: List[str] = Field(default_factory=list)
    metadata: List[NormalizedMetadataItem] = Field(default_factory=list)

class PaperOutput(BaseModel):
    """一篇 paper 的完整输出。"""
    pmid: str
    environment: str = "unknown"           # paper 主导环境
    samples: List[FinalSampleRecord] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════
#  Phase A — Identity Resolution 输出
# ═══════════════════════════════════════════════════════════

@dataclass
class SampleIdentity:
    """Phase A 输出：单个样品的身份信息。"""
    accession: str = ""
    formal_name: str = ""             # DB sample_name 或论文中的提交名
    aliases: List[str] = dc_field(default_factory=list)  # 论文中的所有别名
    parent_project: str = ""          # 所属 BioProject accession
    environment: str = "unknown"      # 从 env_tag 获取

    @property
    def all_names(self) -> Set[str]:
        """所有名称集合，用于关键词检索。"""
        names = set()
        if self.formal_name:
            names.add(self.formal_name)
        names.update(self.aliases)
        return names


# Type alias for identity map
IdentityMap = Dict[str, SampleIdentity]  # accession → SampleIdentity
