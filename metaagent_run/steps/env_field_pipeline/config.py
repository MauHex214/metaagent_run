"""env_field_pipeline 共享配置。"""
from pathlib import Path
from typing import Final

STEP_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT_DIR: Final[Path] = STEP_DIR.parents[2]

# ── 输入 ──────────────────────────────────────────────────────────
STEP2_INPUT: Final[Path] = PROJECT_ROOT_DIR / "relation_v1_step2_relation_output.json"
PAPER_ENV_MAP: Final[Path] = PROJECT_ROOT_DIR / "paper_env_map.json"
DESIGN_REVIEW_DIR: Final[Path] = PROJECT_ROOT_DIR / "design_review_package"

# ── 输出根目录 ─────────────────────────────────────────────────────
OUTPUT_DIR: Final[Path] = PROJECT_ROOT_DIR / "env_field_pipeline_output"

# ── 各环节产出文件名 ────────────────────────────────────────────────
# Phase 0
PHASE0_OUTPUT: Final[Path] = OUTPUT_DIR / "env0_raw_key_env_vectors.csv"
PHASE0_STATS: Final[Path] = OUTPUT_DIR / "env0_stats.json"

# Phase 1
PHASE1_MAINSTREAM: Final[Path] = OUTPUT_DIR / "env1_mainstream_raw_keys.csv"
PHASE1_ORPHAN: Final[Path] = OUTPUT_DIR / "env1_orphan_pool_raw_keys.csv"
PHASE1_STATS: Final[Path] = OUTPUT_DIR / "env1_stats.json"

# Phase 2
PHASE2_KEPT: Final[Path] = OUTPUT_DIR / "env2_kept_raw_keys.csv"
PHASE2_EXCLUDED: Final[Path] = OUTPUT_DIR / "env2_excluded_raw_keys.csv"
PHASE2_EVAL_SPLIT: Final[Path] = OUTPUT_DIR / "env2_eval_split.json"
PHASE2_EVAL_REPORT: Final[Path] = OUTPUT_DIR / "env2_eval_report.json"

# Phase 3（新版结构化标注；4 槽位 family/subtype/qk/modifier_bag）
PHASE3_OUTPUT: Final[Path] = OUTPUT_DIR / "env3_structured_annotations.csv"
PHASE3_QK_COUNTS: Final[Path] = OUTPUT_DIR / "env3_quantity_kind_counts.csv"
# legacy alias（仍被 phase4 引用到旧输出 name；过渡期保留指向新输出）
PHASE3_QK_CANDIDATES: Final[Path] = OUTPUT_DIR / "env3_quantity_kind_counts.csv"

# Phase 4
PHASE4_CANONICALS: Final[Path] = OUTPUT_DIR / "env4_canonicals.csv"
PHASE4_MAPPING: Final[Path] = OUTPUT_DIR / "env4_canonical_to_raw_key.csv"
PHASE4_EDC_LOG: Final[Path] = OUTPUT_DIR / "env4_edc_verify_log.csv"

# Phase 5
PHASE5_THRESHOLD_CAL: Final[Path] = OUTPUT_DIR / "env5_threshold_calibration.csv"
PHASE5_MAIN_LIST: Final[Path] = OUTPUT_DIR / "env5_main_list.csv"
PHASE5_SIGNATURE_PREFIX: Final[str] = "env5_signature_"  # + {env}.csv
PHASE5_TRACE: Final[Path] = OUTPUT_DIR / "env5_full_traceability.csv"

# ── 环境约定 ────────────────────────────────────────────────────────
HYDRO_ENVS: Final[tuple] = ("Open_ocean", "Coastal_waters", "Lake", "Wetlands")

# ── LLM endpoint（与 step3/4 保持一致，DeepSeek-V3 西南二区专用端点）──
BASE_URL: Final[str] = "https://maas-cn-southwest-2.modelarts-maas.com/deepseek-v3"
MODEL: Final[str] = "DeepSeek-V3"
API_STYLE: Final[str] = "openai"
AUTH_MODE: Final[str] = "bearer"
TEMPERATURE: Final[float] = 0.05
MAX_TOKENS: Final[int] = 4096
STOP_SENTINEL: Final[str] = "</json>"

# LLM 并发（ramp-up）
LLM_CONCURRENCY_INITIAL: Final[int] = 8
LLM_CONCURRENCY_STEP: Final[int] = 8
LLM_CONCURRENCY_CEILING: Final[int] = 48
LLM_CONCURRENCY_STEP_EVERY_SECONDS: Final[float] = 15.0
LLM_MAX_RETRIES: Final[int] = 3


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
