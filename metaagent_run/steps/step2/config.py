import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final, FrozenSet, Tuple


STEP_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT_DIR: Final[Path] = STEP_DIR.parents[2]

STEP1_PROMPT_VERSION: Final[str] = "target_env_v1"
FULL_TEXT_JSON: Final[Path] = PROJECT_ROOT_DIR / f"{STEP1_PROMPT_VERSION}_relation_input.json"
PAPER_ENV_MAP_FILE: Final[Path] = PROJECT_ROOT_DIR / "paper_env_map.json"
RELATION_OUTPUT_FILE: Final[Path] = PROJECT_ROOT_DIR / "step2_relation_output.json"
PMID_YEAR_TXT: Final[Path] = PROJECT_ROOT_DIR / "paper_down" / "pmid_year.txt"

DISCOVERY_OUT: Final[Path] = PROJECT_ROOT_DIR / "step2_discovery_input.json"

RELATION_FAILED_LOG_FILE: Final[Path] = PROJECT_ROOT_DIR / "step2_failed.log"
RELATION_FILTERED_LOG_FILE: Final[Path] = PROJECT_ROOT_DIR / "step2_filtered.log.jsonl"

RELATION_PROMPT_VERSION: Final[str] = "relation_v1"

STOP_SENTINEL: Final[str] = "</json>"
MAX_TOKENS_CAP: Final[int] = 1024
MIN_TEXT_LENGTH: Final[int] = 200
TEXT_CHUNK_SIZE: Final[int] = 20000
TEXT_OVERLAP: Final[int] = 150
RETRY_TIMES: Final[int] = 3
RETRY_TEMPS: Final[tuple[float, float, float]] = (0.1, 0.05, 0.2)
BACKOFF_BASE: Final[float] = 2.0
BACKOFF_CAP: Final[float] = 10.0
CONTINUATION_MAX_ROUNDS: Final[int] = 2
FALLBACK_FROM_ATTEMPT: Final[int] = 1

RELATION_BATCH_SIZE: Final[int] = 1

MAX_CONCURRENCY: Final[int] = 8

# ── 表格 / 超长文本头部采样 ─────────────────────────────
# 表格类条目和超长 supplementary 文件只取前 N 字符送 LLM，
# 避免对结构化/重复数据做无意义的多 chunk 调用。
HEADER_SAMPLE_SIZE: Final[int] = 3000

# ── 多端点配置 ─────────────────────────────────────────
# 每个端点可独立配置 base_url, model, api_key_env, max_concurrency。
# 只配 1 个端点 = 单端点模式（等价于原行为）。
ENDPOINTS: Final[Tuple[Dict[str, str], ...]] = (
    {"base_url": "https://api.modelarts-maas.com", "model": "DeepSeek-V3",
     "api_key_env": "ALL_API_KEY", "max_concurrency": "8", "streaming_timeout": "90"},
    # {"base_url": "https://maas-cn-southwest-2.modelarts-maas.com/deepseek-v3", "model": "DeepSeek-V3",
    #  "api_key_env": "ALL_API_KEY", "max_concurrency": "8"},  # 与 global 共享限流配额，勿同时启用
    {"base_url": "https://uni-api.cstcloud.cn", "model": "deepseek-v3:671b",
     "api_key_env": "CAS_API_KEY", "max_concurrency": "16", "streaming_timeout": "180"},
)

## 使用AZURE OPENAI服务（记得替换API_KEY）
# BASE_URL: Final[str] = "https://southgene-openai-8.openai.azure.com"
# MODEL: Final[str] = "gpt-4o"
# API_STYLE: Final[str] = "azure"  # 可选 "openai" 或 "azure"
# AZURE_API_VERSION: Final[str] = "2023-05-15"
# AZURE_DEPLOYMENT: Final[str] = "gpt-4o"
# AUTH_MODE: Final[str] = "api-key"  # 可选 "api-key" 或 "bearer"

## 使用中国科技云（记得替换API_KEY）
# BASE_URL: Final[str] = "https://uni-api.cstcloud.cn"
# MODEL: Final[str] = "deepseek-v3:671b"
# API_STYLE: Final[str] = "openai"
# AZURE_API_VERSION: Final[str] = "2023-05-15"   # 仅占位，无实际作用
# AZURE_DEPLOYMENT: Final[str] = "gpt-4o"   # 仅占位，无实际作用
# AUTH_MODE: Final[str] = "bearer"    # 中国科技云使用 Bearer Token 认证

## DS官方模型服务（记得替换API_KEY）
BASE_URL: Final[str] = "https://api.modelarts-maas.com"
MODEL: Final[str] = "DeepSeek-V3"
API_STYLE: Final[str] = "openai"
AZURE_API_VERSION: Final[str] = "2023-05-15"   # 仅占位，无实际作用
AZURE_DEPLOYMENT: Final[str] = "gpt-4o"   # 仅占位，无实际作用
AUTH_MODE: Final[str] = "bearer"    # 中国科技云使用 Bearer Token 认证

TEMPERATURE: Final[float] = 0.1
MAX_TOKENS: Final[int] = 1024

EXCLUDED_SECTION_TYPES: Final[FrozenSet[str]] = frozenset(
    {
        "TITLE",
        "REF",
        "FIG",
        "COMP_INT",
        "AUTH_CONT",
        "REVIEW_INFO",
        "APPENDIX",
        "INTRO",
        "DISCUSS",
        "CONCL",
        "ACK_FUND",
    }
)


@dataclass(frozen=True)
class RuntimeConfig:
    full_text_json: Path
    relation_output_file: Path
    pmid_year_txt: Path
    paper_env_map_file: Path
    discovery_out: Path
    relation_failed_log_file: Path
    relation_filtered_log_file: Path
    relation_prompt_version: str
    stop_sentinel: str
    max_tokens_cap: int
    min_text_length: int
    retry_temps: tuple[float, float, float]
    retry_times: int
    text_chunk_size: int
    text_overlap: int
    excluded_section_types: FrozenSet[str]
    base_url: str
    model: str
    api_key: str
    api_style: str
    azure_api_version: str
    azure_deployment: str
    auth_mode: str
    temperature: float
    max_tokens: int
    max_concurrency: int
    backoff_base: float
    backoff_cap: float
    continuation_max_rounds: int
    fallback_from_attempt: int
    relation_batch_size: int
    header_sample_size: int
    endpoints: Tuple[Dict[str, str], ...]


def load_runtime_config() -> RuntimeConfig:
    api_style = os.environ.get("METAAGENT_API_STYLE", API_STYLE).strip().lower()
    if api_style not in {"openai", "azure"}:
        api_style = API_STYLE

    auth_mode = os.environ.get("METAAGENT_AUTH_MODE", AUTH_MODE).strip().lower()
    if auth_mode not in {"bearer", "api-key"}:
        auth_mode = AUTH_MODE

    return RuntimeConfig(
        full_text_json=FULL_TEXT_JSON,
        relation_output_file=RELATION_OUTPUT_FILE,
        pmid_year_txt=PMID_YEAR_TXT,
        paper_env_map_file=PAPER_ENV_MAP_FILE,
        discovery_out=DISCOVERY_OUT,
        relation_failed_log_file=RELATION_FAILED_LOG_FILE,
        relation_filtered_log_file=RELATION_FILTERED_LOG_FILE,
        relation_prompt_version=RELATION_PROMPT_VERSION,
        stop_sentinel=STOP_SENTINEL,
        max_tokens_cap=MAX_TOKENS_CAP,
        min_text_length=MIN_TEXT_LENGTH,
        retry_temps=RETRY_TEMPS,
        retry_times=RETRY_TIMES,
        text_chunk_size=TEXT_CHUNK_SIZE,
        text_overlap=TEXT_OVERLAP,
        excluded_section_types=EXCLUDED_SECTION_TYPES,
        base_url=BASE_URL,
        model=MODEL,
        api_key=os.environ.get("ALL_API_KEY", ""),
        api_style=api_style,
        azure_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", AZURE_API_VERSION),
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", AZURE_DEPLOYMENT),
        auth_mode=auth_mode,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        max_concurrency=MAX_CONCURRENCY,
        backoff_base=BACKOFF_BASE,
        backoff_cap=BACKOFF_CAP,
        continuation_max_rounds=CONTINUATION_MAX_ROUNDS,
        fallback_from_attempt=FALLBACK_FROM_ATTEMPT,
        relation_batch_size=RELATION_BATCH_SIZE,
        header_sample_size=HEADER_SAMPLE_SIZE,
        endpoints=ENDPOINTS,
    )
