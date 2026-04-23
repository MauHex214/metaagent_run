import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final


STEP_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT_DIR: Final[Path] = STEP_DIR.parents[2]

FULL_TEXT_JSON: Final[Path] = PROJECT_ROOT_DIR / "oa_merge_env.json"
RELATION_INPUT_OUT: Final[Path] = PROJECT_ROOT_DIR / "relation_input.json"
PAPER_ENV_CACHE: Final[Path] = PROJECT_ROOT_DIR / "paper_env_map.json"

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
RETRY_TEMPS: Final[tuple[float, float, float]] = (0.1, 0.3, 0.5)
MAX_TOKENS: Final[int] = 64
MAX_CONCURRENCY: Final[int] = 16
LLM_RETRY_TIMES: Final[int] = 3
BACKOFF_BASE: Final[float] = 1.5
BACKOFF_CAP: Final[float] = 5.0
PROMPT_VERSION: Final[str] = "target_env_v1"
LLM_FAILURE_FALLBACK_DEFAULT: Final[str] = "Others"
FAILED_LOG_FILE: Final[Path] = PROJECT_ROOT_DIR / "failed_step1_llm.log"
FILTERED_LOG_FILE: Final[Path] = PROJECT_ROOT_DIR / "filtered_step1.log.jsonl"


@dataclass(frozen=True)
class RuntimeConfig:
    full_text_json: Path
    relation_input_out: Path
    paper_env_cache: Path
    base_url: str
    model: str
    api_key: str
    api_style: str
    azure_api_version: str
    azure_deployment: str
    auth_mode: str
    temperature: float
    retry_temps: tuple[float, float, float]
    max_tokens: int
    max_concurrency: int
    llm_retry_times: int
    backoff_base: float
    backoff_cap: float
    llm_failure_fallback: str
    prompt_version: str
    failed_log_file: Path
    filtered_log_file: Path


def load_runtime_config() -> RuntimeConfig:
    fallback = os.environ.get(
        "ABSTRACT_ENV_LLM_FAILURE_FALLBACK",
        LLM_FAILURE_FALLBACK_DEFAULT,
    ).strip()
    if fallback not in {"Others"}:
        fallback = LLM_FAILURE_FALLBACK_DEFAULT

    api_style = os.environ.get("METAAGENT_API_STYLE", API_STYLE).strip().lower()
    if api_style not in {"openai", "azure"}:
        api_style = API_STYLE

    auth_mode = os.environ.get("METAAGENT_AUTH_MODE", AUTH_MODE).strip().lower()
    if auth_mode not in {"bearer", "api-key"}:
        auth_mode = AUTH_MODE

    return RuntimeConfig(
        full_text_json=FULL_TEXT_JSON,
        relation_input_out=RELATION_INPUT_OUT,
        paper_env_cache=PAPER_ENV_CACHE,
        base_url=BASE_URL,
        model=MODEL,
        api_key=os.environ.get("ALL_API_KEY", ""),
        api_style=api_style,
        azure_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", AZURE_API_VERSION),
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", AZURE_DEPLOYMENT),
        auth_mode=auth_mode,
        temperature=TEMPERATURE,
        retry_temps=RETRY_TEMPS,
        max_tokens=MAX_TOKENS,
        max_concurrency=MAX_CONCURRENCY,
        llm_retry_times=LLM_RETRY_TIMES,
        backoff_base=BACKOFF_BASE,
        backoff_cap=BACKOFF_CAP,
        llm_failure_fallback=fallback,
        prompt_version=PROMPT_VERSION,
        failed_log_file=FAILED_LOG_FILE,
        filtered_log_file=FILTERED_LOG_FILE,
    )
