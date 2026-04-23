import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final


STEP_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT_DIR: Final[Path] = STEP_DIR.parents[2]

STOP_SENTINEL: Final[str] = "</json>"
PROMPT_VERSION: Final[str] = "mixs_mapping_v1"

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

TEMPERATURE: Final[float] = 0.05
MAX_TOKENS: Final[int] = 4096

FIELDS_FILE: Final[Path] = PROJECT_ROOT_DIR / "step3_schema_discovery_result.json"
MIXS_FILE: Final[Path] = PROJECT_ROOT_DIR / "mixs_standards.json"
PMID_INDEX_FILE: Final[Path] = PROJECT_ROOT_DIR / "step3_field_pmid_index.json"
PMID_ENV_INDEX_FILE: Final[Path] = PROJECT_ROOT_DIR / "step3_field_pmid_env_index.json"
PAPER_ENV_MAP_FILE: Final[Path] = PROJECT_ROOT_DIR / "paper_env_map.json"
DISCOVERY_FILE: Final[Path] = PROJECT_ROOT_DIR / "step3_schema_discovery_result.json"
OUTPUT_DIR: Final[Path] = PROJECT_ROOT_DIR / "step4_metadata_extend_output"
MAPPING_REVIEW_DECISIONS_FILE: Final[Path] = PROJECT_ROOT_DIR / "mapping_review_decisions.json"
VALID_SUB_ENVS: Final[frozenset[str]] = frozenset({"Open_ocean", "Coastal_waters", "Lake", "Wetlands"})

# ── Output file names ──
TIERED_OUTPUT_FILE: Final[str] = "step4b_tiered_extraction_targets.json"
ENV_TARGETS_OUTPUT_FILE: Final[str] = "step4b_env_extraction_targets.json"
SUPPLEMENTARY_LOW_FREQ_FILE: Final[str] = "step4b_supplementary_low_freq_fields.json"

# ── Intermediate products ──
MAPPING_RESULT_FILE: Final[str] = "step4b_mapping_result.json"
UNMAPPED_FIELDS_FILE: Final[str] = "step4b_unmapped_fields.json"
FREQUENCY_BY_MIXS_FILE: Final[str] = "step4b_frequency_by_mixs_term.json"
MAPPING_CHECKPOINT_FILE: Final[str] = "step4b_mapping_checkpoint.json"
REVIEW_QUEUE_CSV_FILE: Final[str] = "step4b_mapping_review_queue.csv"

# ── Visualization ──
ENV_FIELD_FREQUENCY_PDF: Final[str] = "step4b_env_field_frequency.pdf"
ENV_FIELD_HEATMAP_PDF: Final[str] = "step4b_env_field_heatmap.pdf"
ENV_FIELD_BARS_PDF: Final[str] = "step4b_env_field_bars.pdf"
SYNONYM_FANOUT_PDF: Final[str] = "step4b_synonym_fanout.pdf"

BATCH_SIZE: Final[int] = 1
MAX_RETRIES_PER_BATCH: Final[int] = 3
REQUEST_INTERVAL: Final[float] = 0.5
CONCURRENCY: Final[int] = 10
CHECKPOINT_EVERY: Final[int] = 5
TIER2_MIN_PMID: Final[int] = 3
TOP_N_FREQ: Final[int] = 50
TOP_N_FANOUT: Final[int] = 40
MIN_FANOUT: Final[int] = 2
RESUME_FROM_CHECKPOINT: Final[bool] = True

# ── Env-inclusion threshold for step5 extraction targets ───────────
# A field is a "step5 target" in env E if it is reported by at least
# MIN_ENV_PMID papers AND at least MIN_ENV_PCT % of env-E papers.
# The relative threshold adapts to env size (Wetlands n=3150 vs
# Open_ocean n=8208) so smaller envs are not over-represented.
# The older min_pmid=3 produced ~1964 env targets; the tighter pair
# produces ~450, appropriate for step5 per-paper extraction loops.
MIN_ENV_PMID: Final[int] = 5       # absolute safety floor
MIN_ENV_PCT: Final[float] = 0.5    # percentage of env papers (0.5 = 0.5%)


@dataclass(frozen=True)
class RuntimeConfig:
    stop_sentinel: str
    prompt_version: str
    base_url: str
    model: str
    api_key: str
    api_style: str
    azure_api_version: str
    azure_deployment: str
    auth_mode: str
    temperature: float
    max_tokens: int
    fields_file: Path
    mixs_file: Path
    pmid_index_file: Path
    pmid_env_index_file: Path
    paper_env_map_file: Path
    discovery_file: Path
    output_dir: Path
    mapping_review_decisions_file: Path
    batch_size: int
    max_retries_per_batch: int
    request_interval: float
    concurrency: int
    checkpoint_every: int
    tier2_min_pmid: int
    top_n_freq: int
    top_n_fanout: int
    min_fanout: int
    resume_from_checkpoint: bool
    min_env_pmid: int = MIN_ENV_PMID
    min_env_pct: float = MIN_ENV_PCT
    # ── Output file names ──
    tiered_output_file: str = TIERED_OUTPUT_FILE
    env_targets_output_file: str = ENV_TARGETS_OUTPUT_FILE
    supplementary_low_freq_file: str = SUPPLEMENTARY_LOW_FREQ_FILE
    mapping_result_file: str = MAPPING_RESULT_FILE
    unmapped_fields_file: str = UNMAPPED_FIELDS_FILE
    frequency_by_mixs_file: str = FREQUENCY_BY_MIXS_FILE
    mapping_checkpoint_file: str = MAPPING_CHECKPOINT_FILE
    review_queue_csv_file: str = REVIEW_QUEUE_CSV_FILE
    env_field_frequency_pdf: str = ENV_FIELD_FREQUENCY_PDF
    env_field_heatmap_pdf: str = ENV_FIELD_HEATMAP_PDF
    env_field_bars_pdf: str = ENV_FIELD_BARS_PDF
    synonym_fanout_pdf: str = SYNONYM_FANOUT_PDF


def load_runtime_config() -> RuntimeConfig:
    api_style = os.environ.get("METAAGENT_API_STYLE", API_STYLE).strip().lower()
    if api_style not in {"openai", "azure"}:
        api_style = API_STYLE

    auth_mode = os.environ.get("METAAGENT_AUTH_MODE", AUTH_MODE).strip().lower()
    if auth_mode not in {"bearer", "api-key"}:
        auth_mode = AUTH_MODE

    return RuntimeConfig(
        stop_sentinel=STOP_SENTINEL,
        prompt_version=PROMPT_VERSION,
        base_url=BASE_URL,
        model=MODEL,
        api_key=os.environ.get("ALL_API_KEY", ""),
        api_style=api_style,
        azure_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", AZURE_API_VERSION),
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", AZURE_DEPLOYMENT),
        auth_mode=auth_mode,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        fields_file=FIELDS_FILE,
        mixs_file=MIXS_FILE,
        pmid_index_file=PMID_INDEX_FILE,
        pmid_env_index_file=PMID_ENV_INDEX_FILE,
        paper_env_map_file=PAPER_ENV_MAP_FILE,
        discovery_file=DISCOVERY_FILE,
        output_dir=OUTPUT_DIR,
        mapping_review_decisions_file=MAPPING_REVIEW_DECISIONS_FILE,
        batch_size=BATCH_SIZE,
        max_retries_per_batch=MAX_RETRIES_PER_BATCH,
        request_interval=REQUEST_INTERVAL,
        concurrency=CONCURRENCY,
        checkpoint_every=CHECKPOINT_EVERY,
        tier2_min_pmid=TIER2_MIN_PMID,
        top_n_freq=TOP_N_FREQ,
        top_n_fanout=TOP_N_FANOUT,
        min_fanout=MIN_FANOUT,
        resume_from_checkpoint=RESUME_FROM_CHECKPOINT,
        min_env_pmid=MIN_ENV_PMID,
        min_env_pct=MIN_ENV_PCT,
        tiered_output_file=TIERED_OUTPUT_FILE,
        env_targets_output_file=ENV_TARGETS_OUTPUT_FILE,
        supplementary_low_freq_file=SUPPLEMENTARY_LOW_FREQ_FILE,
        mapping_result_file=MAPPING_RESULT_FILE,
        unmapped_fields_file=UNMAPPED_FIELDS_FILE,
        frequency_by_mixs_file=FREQUENCY_BY_MIXS_FILE,
        mapping_checkpoint_file=MAPPING_CHECKPOINT_FILE,
        review_queue_csv_file=REVIEW_QUEUE_CSV_FILE,
        env_field_frequency_pdf=ENV_FIELD_FREQUENCY_PDF,
        env_field_heatmap_pdf=ENV_FIELD_HEATMAP_PDF,
        env_field_bars_pdf=ENV_FIELD_BARS_PDF,
        synonym_fanout_pdf=SYNONYM_FANOUT_PDF,
    )
