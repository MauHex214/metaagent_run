"""Step6 配置：冲突解决模块的路径与常量。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final


# ── 输入文件 ────────────────────────────────────────────────
STEP5_OUTPUT_FILE: Final[str] = "step5_output.json"
ENV_EXTRACTION_TARGETS_FILE: Final[str] = (
    "step4_metadata_extend_output/step4b_env_extraction_targets.json"
)
SCHEMA_DISCOVERY_FILE: Final[str] = "step3_schema_discovery_result.json"
PMID_YEAR_FILE: Final[str] = "paper_down/pmid_year.txt"

# ── 输出文件 ────────────────────────────────────────────────
OUTPUT_FILE: Final[str] = "step6_resolved_output.json"
STATS_FILE: Final[str] = "step6_stats.json"

# ── 仲裁参数 ────────────────────────────────────────────────
MAX_ALTERNATE_SOURCES: Final[int] = 3   # 落选候选保留数

# modality 优先级：数值越小越优
MODALITY_PRIORITY: Final[Dict[str, int]] = {
    "table_parse": 0,
    "llm_extract": 1,
}
MODALITY_DEFAULT_PRIORITY: Final[int] = 99

# section_type 优先级：数值越小越优
# 注：table_parse 的 section 必为 TABLE，已由 modality 规则先选出
# 此表仅在 llm_extract 内部生效；METHODS 优于其他章节
SECTION_PRIORITY: Final[Dict[str, int]] = {
    "METHODS": 0,
    # 其余 RESULTS / DISCUSS / ABSTRACT / INTRO / etc. 统一归为默认
}
SECTION_DEFAULT_PRIORITY: Final[int] = 1

# pub_year 缺失时的排序值（视为未知年，排最后）
PUB_YEAR_MISSING_VALUE: Final[int] = 9999


@dataclass(frozen=True)
class RuntimeConfig:
    input_dir: Path
    output_dir: Path
    step5_output_file: str = STEP5_OUTPUT_FILE
    env_targets_file: str = ENV_EXTRACTION_TARGETS_FILE
    schema_discovery_file: str = SCHEMA_DISCOVERY_FILE
    pmid_year_file: str = PMID_YEAR_FILE
    output_file: str = OUTPUT_FILE
    stats_file: str = STATS_FILE
    max_alternate_sources: int = MAX_ALTERNATE_SOURCES


def load_runtime_config(input_dir: str, output_dir: str = "") -> RuntimeConfig:
    return RuntimeConfig(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir or input_dir),
    )
