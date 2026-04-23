from typing import Literal

from pydantic import BaseModel
from pydantic import Field


class PaperRelationInfo(BaseModel):
    has_accession_metadata: bool
    has_accession_label_metadata: bool
    has_accession_label: bool
    has_label_metadata: bool
    is_valid: bool
    build_mode: str


class DiscoveryStats(BaseModel):
    kept: int
    skipped_non_target_paper: int
    skipped_abstract: int
    skipped_no_metadata: int


RelationLiteral = Literal[
    "accession-label",
    "accession-metadata",
    "label-metadata",
    "accession-label-metadata",
    "unknown",
]


class RelationRecord(BaseModel):
    pmid: str
    source: str
    section_type: str
    index: int
    relation: RelationLiteral

    class Config:
        extra = "allow"


class LLMRelationItem(BaseModel):
    relation: RelationLiteral
    accessions_found: list[str] = Field(default_factory=list)
    labels_found: list[str] = Field(default_factory=list)
    metadata_keys_found: list[str] = Field(default_factory=list)
    evidence_quote: str = Field(default="")

    class Config:
        extra = "allow"


class LLMRelationOutput(BaseModel):
    results: list[LLMRelationItem]


class JudgementItem(BaseModel):
    pmid: str
    source: str
    section_type: str
    index: int
    relation: RelationLiteral
    accessions_found: list[str] = Field(default_factory=list)
    labels_found: list[str] = Field(default_factory=list)
    metadata_keys_found: list[str] = Field(default_factory=list)
    evidence_quote: str = Field(default="")

    class Config:
        extra = "allow"
