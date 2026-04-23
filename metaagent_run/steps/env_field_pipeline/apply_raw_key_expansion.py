"""对 env2_kept_raw_keys.csv 应用 raw_key 展开表。

流程：
    1. 读 raw_key_expansion_table.csv（展开字典，lower-case 查表）
    2. 备份 env2_kept_raw_keys.csv → env2_kept_raw_keys.pre_expansion.csv
    3. 对每条 raw_key 做 lower-case 查表，命中则替换
    4. 保留原始字段到新列 raw_key_original
    5. 覆盖写回 env2_kept_raw_keys.csv

下游 phase3 改动：
    - prompt 输入字段：用 raw_key（展开后，更准确）
    - evidence 索引键：用 raw_key_original（与 step2 relation_output 的字段名一致）
"""
import logging
import shutil
from pathlib import Path

import pandas as pd

from metaagent_run.steps.env_field_pipeline import config

logger = logging.getLogger(__name__)

EXPANSION_TABLE: Path = config.OUTPUT_DIR / "raw_key_expansion_table.csv"
PHASE2_KEPT_BAK: Path = config.OUTPUT_DIR / "env2_kept_raw_keys.pre_expansion.csv"
PHASE2_EXCLUDED_BAK: Path = config.OUTPUT_DIR / "env2_excluded_raw_keys.pre_expansion.csv"


def _load_expansion_map() -> dict[str, str]:
    """读 expansion table，返回 lower_raw_key → normalized 的字典。"""
    if not EXPANSION_TABLE.exists():
        raise RuntimeError(f"Expansion table missing: {EXPANSION_TABLE}")
    df = pd.read_csv(EXPANSION_TABLE, comment="#")
    df = df[df["raw_key"].notna() & df["normalized_raw_key"].notna()]
    df["raw_key"] = df["raw_key"].astype(str).str.strip().str.lower()
    df["normalized_raw_key"] = df["normalized_raw_key"].astype(str).str.strip()
    df = df[df["raw_key"] != ""].drop_duplicates(subset=["raw_key"], keep="first")
    m = dict(zip(df["raw_key"], df["normalized_raw_key"]))
    logger.info("Loaded %d expansion entries (unique lower-case keys)", len(m))
    return m


def _normalize(raw_key: str, expansion: dict[str, str]) -> str:
    """若 raw_key.lower() 在表中则返回展开值，否则原样。"""
    if not isinstance(raw_key, str):
        return raw_key
    return expansion.get(raw_key.strip().lower(), raw_key)


def _apply_to_csv(path: Path, bak_path: Path, expansion: dict[str, str]) -> dict:
    """返回 {'total', 'expanded', 'unchanged'} 统计。"""
    df = pd.read_csv(path)
    if "raw_key_original" in df.columns:
        # 已有 original 列说明之前跑过展开，先从 original 恢复再重新应用（幂等）
        logger.info("%s already has raw_key_original column; restoring then re-applying",
                    path.name)
        df["raw_key"] = df["raw_key_original"]
        df = df.drop(columns=["raw_key_original"])

    if not bak_path.exists():
        shutil.copy(path, bak_path)
        logger.info("Backed up %s → %s", path.name, bak_path.name)

    df["raw_key_original"] = df["raw_key"]
    df["raw_key"] = df["raw_key_original"].apply(lambda k: _normalize(k, expansion))

    n_expanded = int((df["raw_key"] != df["raw_key_original"]).sum())
    logger.info("%s: %d rows; expanded %d; unchanged %d",
                path.name, len(df), n_expanded, len(df) - n_expanded)

    # 新列顺序：raw_key 保持在原位置，raw_key_original 紧跟其后
    cols = list(df.columns)
    if "raw_key" in cols and "raw_key_original" in cols:
        cols.remove("raw_key_original")
        idx = cols.index("raw_key")
        cols.insert(idx + 1, "raw_key_original")
        df = df[cols]

    df.to_csv(path, index=False)
    return {"total": len(df), "expanded": n_expanded, "unchanged": len(df) - n_expanded}


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config.ensure_output_dir()
    expansion = _load_expansion_map()

    kept_stats = _apply_to_csv(config.PHASE2_KEPT, PHASE2_KEPT_BAK, expansion)
    excluded_stats = _apply_to_csv(config.PHASE2_EXCLUDED, PHASE2_EXCLUDED_BAK, expansion)

    # 打印受影响的 raw_key top 20（按 pmid）
    kept = pd.read_csv(config.PHASE2_KEPT)
    changed = kept[kept["raw_key"] != kept["raw_key_original"]].copy()
    if len(changed):
        changed = changed.sort_values("total_pmid", ascending=False).head(30)
        logger.info("Top 30 expanded (kept pool):")
        for r in changed.itertuples(index=False):
            logger.info("  %-25s → %-35s  pmid=%d",
                        r.raw_key_original, r.raw_key, int(r.total_pmid))

    logger.info("=" * 60)
    logger.info("EXPANSION APPLY SUMMARY")
    logger.info("=" * 60)
    logger.info("kept:     total=%d expanded=%d",
                kept_stats["total"], kept_stats["expanded"])
    logger.info("excluded: total=%d expanded=%d",
                excluded_stats["total"], excluded_stats["expanded"])
    logger.info("Backups: %s, %s", PHASE2_KEPT_BAK.name, PHASE2_EXCLUDED_BAK.name)


if __name__ == "__main__":
    run()
