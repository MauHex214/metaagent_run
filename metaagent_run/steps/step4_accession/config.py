import os
from dataclasses import dataclass
from typing import Final, Tuple

STOP_SENTINEL: Final[str] = "</json>"
MAX_TOKENS_CAP: Final[int] = 8192

RETRY_TEMPS: Final[Tuple[float, float, float]] = (0.1, 0.3, 0.5)
RETRY_TIMES: Final[int] = 3
FALLBACK_FROM_ATTEMPT: Final[int] = 1

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
BASE_URL: Final[str] = "https://maas-cn-southwest-2.modelarts-maas.com/deepseek-v3"
MODEL: Final[str] = "DeepSeek-V3"
API_STYLE: Final[str] = "openai"
AZURE_API_VERSION: Final[str] = "2023-05-15"   # 仅占位，无实际作用
AZURE_DEPLOYMENT: Final[str] = "gpt-4o"   # 仅占位，无实际作用
AUTH_MODE: Final[str] = "bearer"    # 中国科技云使用 Bearer Token 认证

TEMPERATURE: Final[float] = 0.1
MAX_TOKENS: Final[int] = 512

INPUT_FILE: Final[str] = "pmid_run_merged_data_expanded.json"
OUTPUT_FILE: Final[str] = "step4a_env_tag_output.json"
FAILED_LOG_FILE: Final[str] = "step4a_failed.log"
FILTERED_LOG_FILE: Final[str] = "step4a_filtered.log.jsonl"
NCBI_ENV_WHITELIST_FILE: Final[str] = "ncbi_taxonomy_env.list"
MAX_CONCURRENCY: Final[int] = 16
BATCH_SIZE: Final[int] = 2
PROMPT_VERSION: Final[str] = "env_tag_v2"

BACKOFF_BASE: Final[float] = 2.0
BACKOFF_CAP: Final[float] = 60.0
CONTINUATION_MAX_ROUNDS: Final[int] = 2


@dataclass(frozen=True)
class RuntimeConfig:
    stop_sentinel: str
    max_tokens_cap: int
    retry_temps: Tuple[float, float, float]
    retry_times: int
    fallback_from_attempt: int
    backoff_base: float
    backoff_cap: float
    continuation_max_rounds: int
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    api_key: str
    api_style: str
    azure_api_version: str
    azure_deployment: str
    auth_mode: str
    input_file: str
    output_file: str
    failed_log_file: str
    filtered_log_file: str
    ncbi_env_whitelist_file: str
    max_concurrency: int
    prompt_version: str
    batch_size: int


def load_runtime_config() -> RuntimeConfig:
    api_style = os.environ.get("METAAGENT_API_STYLE", API_STYLE).strip().lower()
    if api_style not in {"openai", "azure"}:
        api_style = API_STYLE

    auth_mode = os.environ.get("METAAGENT_AUTH_MODE", AUTH_MODE).strip().lower()
    if auth_mode not in {"bearer", "api-key"}:
        auth_mode = AUTH_MODE

    return RuntimeConfig(
        stop_sentinel=STOP_SENTINEL,
        max_tokens_cap=MAX_TOKENS_CAP,
        retry_temps=RETRY_TEMPS,
        retry_times=RETRY_TIMES,
        fallback_from_attempt=FALLBACK_FROM_ATTEMPT,
        backoff_base=BACKOFF_BASE,
        backoff_cap=BACKOFF_CAP,
        continuation_max_rounds=CONTINUATION_MAX_ROUNDS,
        base_url=BASE_URL,
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        api_key=os.environ.get("ALL_API_KEY", ""),
        api_style=api_style,
        azure_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", AZURE_API_VERSION),
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", AZURE_DEPLOYMENT),
        auth_mode=auth_mode,
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        failed_log_file=FAILED_LOG_FILE,
        filtered_log_file=FILTERED_LOG_FILE,
        ncbi_env_whitelist_file=NCBI_ENV_WHITELIST_FILE,
        max_concurrency=MAX_CONCURRENCY,
        prompt_version=PROMPT_VERSION,
        batch_size=BATCH_SIZE,
    )
