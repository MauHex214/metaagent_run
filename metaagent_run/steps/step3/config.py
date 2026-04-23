"""Step 3 runtime configuration.

The new (Pass A / Pass 0 / Pass B / Pass C) pipeline adds fields for
anchor-word cutoff, family-size cap, and LLM concurrency ramp-up.
Existing fields from the older pipeline are preserved for backwards
compatibility (the old `SemanticDeduplicator` is no longer called).
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final


STEP_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT_DIR: Final[Path] = STEP_DIR.parents[2]

# ── Prompt ──────────────────────────────────────────────────────────
# Old prompt (kept for reference / ablation runs).
LEGACY_PROMPT_VERSION: Final[str] = "model3_discovery_v1"
# Pass B family-partition prompt (active).
PROMPT_VERSION: Final[str] = "family_partition_v1"

# ── LLM endpoint ────────────────────────────────────────────────────
BASE_URL: Final[str] = "https://maas-cn-southwest-2.modelarts-maas.com/deepseek-v3"
MODEL: Final[str] = "DeepSeek-V3"
API_STYLE: Final[str] = "openai"
AZURE_API_VERSION: Final[str] = "2023-05-15"
AZURE_DEPLOYMENT: Final[str] = "gpt-4o"
AUTH_MODE: Final[str] = "bearer"

TEMPERATURE: Final[float] = 0.1
MAX_TOKENS: Final[int] = 4096
STOP_SENTINEL: Final[str] = "</json>"

# ── I/O paths ───────────────────────────────────────────────────────
INPUT_FILE: Final[Path] = PROJECT_ROOT_DIR / "step2_discovery_input.json"
OUTPUT_FILE: Final[Path] = PROJECT_ROOT_DIR / "step3_schema_discovery_result.json"
CHECKPOINT_FILE: Final[Path] = PROJECT_ROOT_DIR / "step3_family_checkpoint.jsonl"
FIELD_PMID_INDEX_FILE: Final[Path] = PROJECT_ROOT_DIR / "step3_field_pmid_index.json"
FIELD_PMID_ENV_INDEX_FILE: Final[Path] = PROJECT_ROOT_DIR / "step3_field_pmid_env_index.json"
SIDE_TAGS_FILE: Final[Path] = PROJECT_ROOT_DIR / "step3_side_tags.json"
PASS_B_STATS_FILE: Final[Path] = PROJECT_ROOT_DIR / "step3_pass_b_stats.json"
PASS_A_REPORT_FILE: Final[Path] = PROJECT_ROOT_DIR / "step3_pass_a_report.json"
SATURATION_PLOT_FILE: Final[Path] = PROJECT_ROOT_DIR / "step3_saturation_curve.pdf"
# Canonical review post-processing (only applied when --review is passed).
CANONICAL_REVIEW_DECISIONS_FILE: Final[Path] = (
    PROJECT_ROOT_DIR / "canonical_review_decisions.json"
)

# ── Pass 0 / Pass A ─────────────────────────────────────────────────
ANCHOR_CUTOFF: Final[int] = 2   # minimum DF for a token to become an anchor word

# ── Pass B — family-partition ───────────────────────────────────────
FAMILY_SIZE_CAP: Final[int] = 80  # split families larger than this into chunks
LLM_MAX_RETRIES_PER_FAMILY: Final[int] = 3
LLM_CONCURRENCY_INITIAL: Final[int] = 8
LLM_CONCURRENCY_STEP: Final[int] = 8
LLM_CONCURRENCY_CEILING: Final[int] = 32
LLM_CONCURRENCY_STEP_EVERY_SECONDS: Final[float] = 10.0

# ── Legacy settings (no longer consulted by the active pipeline) ────
CHECKPOINT_STRIDE: Final[int] = 50
REQUEST_INTERVAL: Final[float] = 0.3
LLM_MAX_RETRIES: Final[int] = 3
RESUME_FROM_CHECKPOINT: Final[bool] = True
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
    side_tags_file: Path
    pass_b_stats_file: Path
    pass_a_report_file: Path
    canonical_review_decisions_file: Path
    # Pass 0 / Pass A
    anchor_cutoff: int
    # Pass B
    family_size_cap: int
    llm_max_retries_per_family: int
    llm_concurrency_initial: int
    llm_concurrency_step: int
    llm_concurrency_ceiling: int
    llm_concurrency_step_every_seconds: float
    # Legacy
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
        side_tags_file=SIDE_TAGS_FILE,
        pass_b_stats_file=PASS_B_STATS_FILE,
        pass_a_report_file=PASS_A_REPORT_FILE,
        canonical_review_decisions_file=CANONICAL_REVIEW_DECISIONS_FILE,
        anchor_cutoff=ANCHOR_CUTOFF,
        family_size_cap=FAMILY_SIZE_CAP,
        llm_max_retries_per_family=LLM_MAX_RETRIES_PER_FAMILY,
        llm_concurrency_initial=LLM_CONCURRENCY_INITIAL,
        llm_concurrency_step=LLM_CONCURRENCY_STEP,
        llm_concurrency_ceiling=LLM_CONCURRENCY_CEILING,
        llm_concurrency_step_every_seconds=LLM_CONCURRENCY_STEP_EVERY_SECONDS,
        checkpoint_stride=CHECKPOINT_STRIDE,
        request_interval=REQUEST_INTERVAL,
        resume_from_checkpoint=RESUME_FROM_CHECKPOINT,
        llm_max_retries=LLM_MAX_RETRIES,
        use_evidence_in_dedup=USE_EVIDENCE_IN_DEDUP,
        use_evidence_in_filter=USE_EVIDENCE_IN_FILTER,
    )
