"""Step7 数据模型。"""

from dataclasses import dataclass, field as dc_field
from typing import Any, List, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════
#  Value 归一化结果（内部）
# ═══════════════════════════════════════════════════════════

@dataclass
class NormalizedValue:
    """单条 value 归一化的产出。"""
    value_normalized: Any = None
    unit: str = ""
    value_type: str = ""              # measurement / number / integer / boolean / datetime / coordinate / range / enum / ontology_term / text / duration / compound / ratio
    normalize_status: str = "ok"      # ok / failed / out_of_preferred / no_ontology_match
    normalize_error: str = ""


# ═══════════════════════════════════════════════════════════
#  最终输出 schema
# ═══════════════════════════════════════════════════════════

class ProvenanceRecord(BaseModel):
    """落选候选的来源记录（与 step6 同结构）。"""
    pmid: str
    value: str
    raw_field: str
    source_file: str = ""
    section_type: str = ""
    paragraph_index: int = -1
    extraction_modality: str = ""


class FinalMetadataEntry(BaseModel):
    """单条 metadata 的最终输出。"""
    key: str                             # mixs:xxx 或 internal:xxx
    raw_field: str
    value_raw: str
    value_normalized: Any = None
    unit: str = ""
    value_type: str = ""
    normalize_status: str = "ok"
    normalize_error: str = ""
    source_level: str = ""               # biosample / sra_sample / sra_experiment / sra_run / bioproject / sra_study
    authoritative_pmid: str = ""
    source_file: str = ""
    section_type: str = ""
    paragraph_index: int = -1
    extraction_modality: str = ""
    alternate_sources: List[ProvenanceRecord] = Field(default_factory=list)


class BioSampleRecord(BaseModel):
    """以 BioSample 为主键的样本级记录。"""
    biosample_id: str
    parent_project: str = ""
    runs: List[str] = Field(default_factory=list)
    environment: str = "unknown"             # step4a 权威
    paper_dominant_env: str = ""             # step5 投票
    formal_name: str = ""
    aliases: List[str] = Field(default_factory=list)
    source_pmids: List[str] = Field(default_factory=list)
    metadata: List[FinalMetadataEntry] = Field(default_factory=list)


class Step7Output(BaseModel):
    samples: List[BioSampleRecord] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)
