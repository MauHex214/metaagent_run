"""Step7 配置：路径与常量。"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, Tuple


# ── 输入文件 ────────────────────────────────────────────────
STEP6_OUTPUT_FILE: Final[str] = "step6_resolved_output.json"
STEP5_OUTPUT_FILE: Final[str] = "step5_output.json"     # 用于 paper_dominant_env
ACCESSION_LIST_FILE: Final[str] = "accession_list.tsv"
EXPANDED_METADATA_FILE: Final[str] = "pmid_run_merged_data_expanded.json"
ENV_TAG_FILE: Final[str] = "env_tag_v1_step4a_env_tag_output.json"
ENVO_TERMS_FILE: Final[str] = "envo_terms.json"
CDE_FILE: Final[str] = "cde_per_environment.json"

# CDE 构建子模块输入
ENV_EXTRACTION_TARGETS_FILE: Final[str] = (
    "step4_metadata_extend_output/step4b_env_extraction_targets.json"
)
SCHEMA_DISCOVERY_FILE: Final[str] = "step3_schema_discovery_result.json"
SCHEMA_CONTEXTS_FILE: Final[str] = "step3_schema_contexts.json"
MIXS_XLSX_FILE: Final[str] = "mixs/mixs_v6.xlsx"

# ── 输出文件 ────────────────────────────────────────────────
OUTPUT_FILE: Final[str] = "sample_level_metadata.json"
STATS_FILE: Final[str] = "step7_stats.json"

CDE_TIER1_AUTOGEN_FILE: Final[str] = "cde_tier1_autogen.json"
CDE_TIER2_SUGGESTIONS_FILE: Final[str] = "tier2_cde_suggestions.json"

# ── LLM 服务（与 step5 相同）────────────────────────────────
LLM_BASE_URL: Final[str] = "https://maas-cn-southwest-2.modelarts-maas.com/deepseek-v3"
LLM_MODEL: Final[str] = "DeepSeek-V3"
LLM_API_KEY_ENV: Final[str] = "DS_API_KEY"
LLM_TEMPERATURE: Final[float] = 0.1
LLM_MAX_TOKENS: Final[int] = 4096
LLM_API_STYLE: Final[str] = "openai"
LLM_AUTH_MODE: Final[str] = "bearer"
LLM_STOP_SENTINEL: Final[str] = "</json>"
LLM_MAX_TOKENS_CAP: Final[int] = 8192

# 重试参数
RETRY_TIMES: Final[int] = 3
RETRY_TEMPS: Final[Tuple[float, float, float]] = (0.1, 0.3, 0.5)
BACKOFF_BASE: Final[float] = 2.0
BACKOFF_CAP: Final[float] = 60.0
CONTINUATION_MAX_ROUNDS: Final[int] = 2

# CDE 构建
TIER2_LLM_BATCH_SIZE: Final[int] = 10        # 每批送 LLM 的 Tier 2 字段数
TIER2_EVIDENCE_PER_FIELD: Final[int] = 3     # 每字段附带证据片段数

# Hoist 优先级（数值越小越优）
HOIST_LEVEL_PRIORITY: Final[Dict[str, int]] = {
    "biosample": 0,        # SAMN/SAMEA/SAMD — 自有
    "sra_sample": 1,       # SRS/ERS/DRS
    "sra_experiment": 2,   # SRX/ERX/DRX
    "sra_run": 3,          # SRR/ERR/DRR
    "bioproject": 4,       # PRJNA/PRJEB/PRJDB — 最远
    "sra_study": 4,        # SRP/ERP/DRP / SRA — 同 bioproject 层级
}


@dataclass(frozen=True)
class RuntimeConfig:
    input_dir: Path
    output_dir: Path
    # 输入
    step6_output_file: str = STEP6_OUTPUT_FILE
    step5_output_file: str = STEP5_OUTPUT_FILE
    accession_list_file: str = ACCESSION_LIST_FILE
    expanded_metadata_file: str = EXPANDED_METADATA_FILE
    env_tag_file: str = ENV_TAG_FILE
    envo_terms_file: str = ENVO_TERMS_FILE
    cde_file: str = CDE_FILE
    # CDE 构建相关
    env_targets_file: str = ENV_EXTRACTION_TARGETS_FILE
    schema_discovery_file: str = SCHEMA_DISCOVERY_FILE
    schema_contexts_file: str = SCHEMA_CONTEXTS_FILE
    mixs_xlsx_file: str = MIXS_XLSX_FILE
    # 输出
    output_file: str = OUTPUT_FILE
    stats_file: str = STATS_FILE
    cde_tier1_autogen_file: str = CDE_TIER1_AUTOGEN_FILE
    cde_tier2_suggestions_file: str = CDE_TIER2_SUGGESTIONS_FILE
    # LLM
    llm_base_url: str = LLM_BASE_URL
    llm_model: str = LLM_MODEL
    llm_api_key: str = ""
    llm_temperature: float = LLM_TEMPERATURE
    llm_max_tokens: int = LLM_MAX_TOKENS
    llm_api_style: str = LLM_API_STYLE
    llm_auth_mode: str = LLM_AUTH_MODE
    llm_stop_sentinel: str = LLM_STOP_SENTINEL
    llm_max_tokens_cap: int = LLM_MAX_TOKENS_CAP
    retry_times: int = RETRY_TIMES
    retry_temps: Tuple[float, float, float] = RETRY_TEMPS
    backoff_base: float = BACKOFF_BASE
    backoff_cap: float = BACKOFF_CAP
    continuation_max_rounds: int = CONTINUATION_MAX_ROUNDS
    tier2_llm_batch_size: int = TIER2_LLM_BATCH_SIZE
    tier2_evidence_per_field: int = TIER2_EVIDENCE_PER_FIELD


def load_runtime_config(input_dir: str, output_dir: str = "") -> RuntimeConfig:
    return RuntimeConfig(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir or input_dir),
        llm_api_key=os.environ.get(LLM_API_KEY_ENV, ""),
    )
