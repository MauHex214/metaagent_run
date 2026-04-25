"""Step5_test 配置：LLM 参数、上游文件路径。"""

import os
from dataclasses import dataclass
from typing import Final, Tuple

# ── LLM 服务 ──────────────────────────────────────────────
BASE_URL: Final[str] = "https://maas-cn-southwest-2.modelarts-maas.com/deepseek-v3"
MODEL: Final[str] = "DeepSeek-V3"
API_STYLE: Final[str] = "openai"
AZURE_API_VERSION: Final[str] = "2023-05-15"   # 仅占位，无实际作用
AZURE_DEPLOYMENT: Final[str] = "gpt-4o"   # 仅占位，无实际作用
AUTH_MODE: Final[str] = "bearer"    # 使用 Bearer Token 认证

TEMPERATURE: Final[float] = 0.1
MAX_TOKENS: Final[int] = 2048
STOP_SENTINEL: Final[str] = "</json>"
MAX_TOKENS_CAP: Final[int] = 8192

# ── 重试 / 退避 ──────────────────────────────────────────
RETRY_TIMES: Final[int] = 3
RETRY_TEMPS: Final[Tuple[float, float, float]] = (0.1, 0.3, 0.5)
BACKOFF_BASE: Final[float] = 2.0
BACKOFF_CAP: Final[float] = 60.0
CONTINUATION_MAX_ROUNDS: Final[int] = 2

# ── 文本切分（Phase B3 段落级抽取） ────────────────────
TEXT_CHUNK_SIZE: Final[int] = 12000      # 每块最大字符数
TEXT_OVERLAP: Final[int] = 200           # 相邻块重叠字符数

# ── 表格解析 ──────────────────────────────────────────────
TABLE_MAX_COLS: Final[int] = 30         # 列数 > 此值直接跳过

# ── Identity 阈值 ───────────────────────────────────────
MAX_SAMPLES_FOR_SKELETON: Final[int] = 50   # BioSample 去重后 > 此值触发压缩
MAX_SAMPLES_FOR_PHASE_A: Final[int] = 100   # 压缩后仍 > 此值跳过 Phase A LLM

# ── 并发 ─────────────────────────────────────────────────
PAPER_CONCURRENCY: Final[int] = 8      # 同时处理的论文数
SECTION_CONCURRENCY: Final[int] = 8    # 单篇论文内 Phase B3 section 并发数

# ── Prompt 模板文件名 ────────────────────────────────────
PROMPT_IDENTITY: Final[str] = "step5_identity_v1.txt"
PROMPT_SECTION_EXTRACT: Final[str] = "step5_section_extract_v1.txt"


@dataclass(frozen=True)
class RuntimeConfig:
    # LLM
    base_url: str = BASE_URL
    model: str = MODEL
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS
    api_key: str = ""
    api_style: str = API_STYLE
    azure_api_version: str = AZURE_API_VERSION
    azure_deployment: str = AZURE_DEPLOYMENT
    auth_mode: str = AUTH_MODE
    stop_sentinel: str = STOP_SENTINEL
    max_tokens_cap: int = MAX_TOKENS_CAP
    # 重试
    retry_times: int = RETRY_TIMES
    retry_temps: Tuple[float, float, float] = RETRY_TEMPS
    backoff_base: float = BACKOFF_BASE
    backoff_cap: float = BACKOFF_CAP
    continuation_max_rounds: int = CONTINUATION_MAX_ROUNDS
    # 文本切分
    text_chunk_size: int = TEXT_CHUNK_SIZE
    text_overlap: int = TEXT_OVERLAP
    # 表格
    table_max_cols: int = TABLE_MAX_COLS
    # Identity 阈值
    max_samples_for_skeleton: int = MAX_SAMPLES_FOR_SKELETON
    max_samples_for_phase_a: int = MAX_SAMPLES_FOR_PHASE_A
    # 并发
    paper_concurrency: int = PAPER_CONCURRENCY
    section_concurrency: int = SECTION_CONCURRENCY
    # Prompt 模板
    prompt_identity: str = PROMPT_IDENTITY
    prompt_section_extract: str = PROMPT_SECTION_EXTRACT
    # 日志
    failed_log_file: str = "step5_failed.log"
    # ── 上游文件路径（运行时指定） ────────────────────────
    input_file: str = "target_env_v1_relation_input.json"
    output_file: str = "step5_output.json"
    # 上游产物
    relation_file: str = "relation_v1_step2_relation_output.json"            # step2 relation output
    accession_file: str = ""           # step3 accession output
    accession_list_file: str = "pmid_run-accession0418.list"      # 外部 DB 验证 accession list
    expanded_metadata_file: str = "pmid_run_merged_data_expanded.json"   # pmid_run_merged_data_expanded.json
    env_tag_file: str = "env_tag_v2_step4a_env_tag_output.json"             # step4 env_tag output
    env_extraction_targets_file: str = "env_field_pipeline_output/env6_extraction_targets.json"   # phase6 output


def load_runtime_config(**overrides) -> RuntimeConfig:
    defaults = dict(
        api_key=os.environ.get("ALL_API_KEY", ""),
        api_style=os.environ.get("METAAGENT_API_STYLE", API_STYLE),
        azure_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", AZURE_API_VERSION),
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", AZURE_DEPLOYMENT),
        auth_mode=os.environ.get("METAAGENT_AUTH_MODE", AUTH_MODE),
    )
    defaults.update({k: v for k, v in overrides.items() if v is not None})
    return RuntimeConfig(**defaults)
