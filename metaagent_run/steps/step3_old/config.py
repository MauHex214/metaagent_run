import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final


STEP_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT_DIR: Final[Path] = STEP_DIR.parents[2]

PROMPT_VERSION: Final[str] = "model3_discovery_v1"

# DS官方模型服务（记得替换API_KEY）
BASE_URL: Final[str] = "https://maas-cn-southwest-2.modelarts-maas.com/deepseek-v3"
MODEL: Final[str] = "DeepSeek-V3"
API_STYLE: Final[str] = "openai"
AZURE_API_VERSION: Final[str] = "2023-05-15"   # 占位
AZURE_DEPLOYMENT: Final[str] = "gpt-4o"         # 占位
AUTH_MODE: Final[str] = "bearer"

TEMPERATURE: Final[float] = 0.1
MAX_TOKENS: Final[int] = 1024
STOP_SENTINEL: Final[str] = "</json>"

INPUT_FILE: Final[Path] = PROJECT_ROOT_DIR / "step2_discovery_input.json"
OUTPUT_FILE: Final[Path] = PROJECT_ROOT_DIR / "step3_schema_discovery_result.json"
CHECKPOINT_FILE: Final[Path] = PROJECT_ROOT_DIR / "step3_schema_checkpoint.json"
SATURATION_PLOT_FILE: Final[Path] = PROJECT_ROOT_DIR / "step3_saturation_curve.pdf"
FIELD_PMID_INDEX_FILE: Final[Path] = PROJECT_ROOT_DIR / "step3_field_pmid_index.json"
FIELD_PMID_ENV_INDEX_FILE: Final[Path] = PROJECT_ROOT_DIR / "step3_field_pmid_env_index.json"

# Phase 1 主循环：每处理 CHECKPOINT_STRIDE 个 unique key 落盘一次 checkpoint
# 并打一行日志 summary。注意这**不是** LLM 批量调用大小（dedup 仍然逐 key 串行
# 调 LLM）；这是 checkpoint / 日志 / IterationRecord 的步长。
CHECKPOINT_STRIDE: Final[int] = 50
REQUEST_INTERVAL: Final[float] = 0.3
LLM_MAX_RETRIES: Final[int] = 3
RESUME_FROM_CHECKPOINT: Final[bool] = True

# Evidence 消融开关：
#   USE_EVIDENCE_IN_DEDUP  — SemanticDeduplicator.resolve() 是否把段落 evidence 传给 LLM 精判
#                            True  → 用 Step 2 的 evidence_quote（按 METHODS/RESULTS/SUPPL/TABLE 优先级首见锁定）
#                            False → 传空串，dedup 仅基于 key 字符串 + 候选列表
#   USE_EVIDENCE_IN_FILTER — filter_discovered_entries() 是否用真实 evidence（影响 review_pool 记录的可读性）
#                            True  → 真实原文片段（审核时无需回查 Step 2）
#                            False → "(step2)" 占位符（满足非空校验但无信息量）
# 消融组合：(D=T,F=T) 完整 / (D=F,F=T) 只评审、dedup 不用 / (D=T,F=F) 只 dedup 不用审 / (D=F,F=F) 双盲基线
USE_EVIDENCE_IN_DEDUP: Final[bool] = True
USE_EVIDENCE_IN_FILTER: Final[bool] = True


@dataclass(frozen=True)
class RuntimeConfig:
    prompt_version: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    api_key: str
    api_style: str
    azure_api_version: str
    azure_deployment: str
    auth_mode: str
    stop_sentinel: str
    input_file: Path
    output_file: Path
    checkpoint_file: Path
    saturation_plot_file: Path
    field_pmid_index_file: Path
    field_pmid_env_index_file: Path
    checkpoint_stride: int
    request_interval: float
    resume_from_checkpoint: bool
    llm_max_retries: int
    use_evidence_in_dedup: bool
    use_evidence_in_filter: bool


def load_runtime_config() -> RuntimeConfig:
    api_style = os.environ.get("METAAGENT_API_STYLE", API_STYLE).strip().lower()
    if api_style not in {"openai", "azure"}:
        api_style = API_STYLE

    auth_mode = os.environ.get("METAAGENT_AUTH_MODE", AUTH_MODE).strip().lower()
    if auth_mode not in {"bearer", "api-key"}:
        auth_mode = AUTH_MODE

    return RuntimeConfig(
        prompt_version=PROMPT_VERSION,
        base_url=BASE_URL,
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        api_key=os.environ.get("ALL_API_KEY", ""),
        api_style=api_style,
        azure_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", AZURE_API_VERSION),
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", AZURE_DEPLOYMENT),
        auth_mode=auth_mode,
        stop_sentinel=STOP_SENTINEL,
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        checkpoint_file=CHECKPOINT_FILE,
        saturation_plot_file=SATURATION_PLOT_FILE,
        field_pmid_index_file=FIELD_PMID_INDEX_FILE,
        field_pmid_env_index_file=FIELD_PMID_ENV_INDEX_FILE,
        checkpoint_stride=CHECKPOINT_STRIDE,
        request_interval=REQUEST_INTERVAL,
        resume_from_checkpoint=RESUME_FROM_CHECKPOINT,
        llm_max_retries=LLM_MAX_RETRIES,
        use_evidence_in_dedup=USE_EVIDENCE_IN_DEDUP,
        use_evidence_in_filter=USE_EVIDENCE_IN_FILTER,
    )
