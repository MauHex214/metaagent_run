from typing import List, Literal

from pydantic import BaseModel, Field


EnvTagLiteral = Literal[
    "Open_ocean",
    "Coastal_waters",
    "Lake",
    "Wetlands",
    "Others",
]


class ValueField(BaseModel):
    value: EnvTagLiteral
    source_field: str
    reason: str


class SampleEntry(BaseModel):
    biosample_id: str
    organism: str
    env_tag: ValueField

    class Config:
        extra = "allow"


class ExtractionOutput(BaseModel):
    results: List[SampleEntry] = Field(default_factory=list)
