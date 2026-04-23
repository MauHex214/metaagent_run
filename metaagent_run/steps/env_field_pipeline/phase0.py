"""环节 0：raw key 环境向量提取。

输入：
    - step 2 段落记录（relation_v1_step2_relation_output.json）
    - PMID → sub_env 映射（paper_env_map.json）

输出：
    - env0_raw_key_env_vectors.csv
    - env0_stats.json（验证用统计）

规则：
    - 只统计 metadata_keys_found 非空的段落记录
    - sub_env 中的 "Others" 跳过（不计入任何环境向量维度、不计入 total_pmid）
    - 若 PMID 在 sub_env 数组中包含多个水圈环境，采用"双倍计数"：
      该 PMID 对每个所属环境的计数都 +1
    - raw key 不做任何 normalize，直接沿用 step 2 的字符串
"""
import json
import logging
from collections import defaultdict

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def run() -> None:
    config.ensure_output_dir()

    logger.info("Loading step2 records from %s", config.STEP2_INPUT)
    with open(config.STEP2_INPUT, "r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info("Loaded %d step2 records", len(records))

    logger.info("Loading paper_env_map from %s", config.PAPER_ENV_MAP)
    with open(config.PAPER_ENV_MAP, "r", encoding="utf-8") as f:
        paper_env_map = json.load(f)
    logger.info("Loaded %d paper env mappings", len(paper_env_map))

    hydro_envs = set(config.HYDRO_ENVS)

    # 去重的 (raw_key, env, pmid) 三元组
    triples: set[tuple[str, str, str]] = set()
    non_empty_records = 0
    missing_pmid_in_map = 0
    pmid_no_hydro_env = 0
    pmid_others_only = 0

    for r in records:
        keys = r.get("metadata_keys_found") or []
        if not keys:
            continue
        non_empty_records += 1
        pmid = r.get("pmid")
        envs = paper_env_map.get(pmid)
        if envs is None:
            missing_pmid_in_map += 1
            continue
        if not envs:
            pmid_no_hydro_env += 1
            continue
        if all(e == "Others" for e in envs):
            pmid_others_only += 1
            continue
        paper_hydro_envs = [e for e in envs if e in hydro_envs]
        if not paper_hydro_envs:
            pmid_no_hydro_env += 1
            continue
        for env in paper_hydro_envs:
            for key in keys:
                triples.add((key, env, pmid))

    logger.info(
        "Non-empty metadata records: %d; missing-pmid: %d; others-only: %d; "
        "no-hydro-env: %d; unique triples: %d",
        non_empty_records,
        missing_pmid_in_map,
        pmid_others_only,
        pmid_no_hydro_env,
        len(triples),
    )

    # 聚合为宽表
    df_trip = pd.DataFrame(list(triples), columns=["raw_key", "env", "pmid"])
    counts = (
        df_trip.groupby(["raw_key", "env"])["pmid"]
        .nunique()
        .unstack(fill_value=0)
    )
    for env in config.HYDRO_ENVS:
        if env not in counts.columns:
            counts[env] = 0
    counts = counts[list(config.HYDRO_ENVS)]
    counts.columns = [f"env_{c}" for c in counts.columns]
    env_cols = [f"env_{e}" for e in config.HYDRO_ENVS]
    counts["total_pmid"] = counts[env_cols].sum(axis=1)
    counts["n_envs_present"] = (counts[env_cols] > 0).sum(axis=1)
    counts = counts.reset_index().sort_values(
        ["total_pmid", "raw_key"], ascending=[False, True]
    )

    counts.to_csv(config.PHASE0_OUTPUT, index=False)
    logger.info("Wrote %d raw keys to %s", len(counts), config.PHASE0_OUTPUT)

    stats = _compute_stats(counts, non_empty_records, missing_pmid_in_map,
                           pmid_others_only, pmid_no_hydro_env, len(triples))
    with open(config.PHASE0_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("Wrote stats to %s", config.PHASE0_STATS)

    _print_verification(counts, stats)


def _compute_stats(df: pd.DataFrame, non_empty: int, missing: int,
                   others_only: int, no_hydro: int, triples: int) -> dict:
    env_cols = [f"env_{e}" for e in config.HYDRO_ENVS]
    total_pmid_sum = int(df["total_pmid"].sum())
    return {
        "step2_non_empty_records": non_empty,
        "records_missing_pmid_in_map": missing,
        "records_others_only": others_only,
        "records_no_hydro_env": no_hydro,
        "unique_raw_key_env_pmid_triples": triples,
        "total_raw_keys": int(len(df)),
        "total_pmid_sum_double_counted": total_pmid_sum,
        "raw_keys_total_pmid_ge_3": int((df["total_pmid"] >= 3).sum()),
        "raw_keys_total_pmid_ge_10": int((df["total_pmid"] >= 10).sum()),
        "raw_keys_n_envs_present": {
            str(k): int(v)
            for k, v in df["n_envs_present"].value_counts().sort_index().items()
        },
        "per_env_raw_key_count_gt0": {
            e: int((df[f"env_{e}"] > 0).sum()) for e in config.HYDRO_ENVS
        },
        "per_env_pmid_sum": {
            e: int(df[f"env_{e}"].sum()) for e in config.HYDRO_ENVS
        },
    }


def _print_verification(df: pd.DataFrame, stats: dict) -> None:
    logger.info("=" * 60)
    logger.info("PHASE 0 VERIFICATION")
    logger.info("=" * 60)
    for k, v in stats.items():
        logger.info("  %s: %s", k, v)

    logger.info("")
    logger.info("Top 20 raw keys by total_pmid:")
    logger.info("\n%s", df.head(20).to_string(index=False))

    logger.info("")
    logger.info("Sample raw-key checks:")
    samples = [
        ("chl_a", "Open_ocean 应占多数"),
        ("chla", "Open_ocean 应占多数"),
        ("chlorophyll_a", "Open_ocean 应占多数"),
        ("peat_depth", "Wetlands 应占多数"),
        ("water_table_depth", "Wetlands 应占多数"),
        ("mixed_layer_depth", "Open_ocean 应占绝对多数"),
        ("mld", "Open_ocean 应占绝对多数"),
        ("lake_depth", "Lake 应占多数"),
        ("secchi_depth", "Lake 应占多数"),
    ]
    for key, expect in samples:
        row = df[df["raw_key"] == key]
        if len(row) > 0:
            d = row.iloc[0].to_dict()
            logger.info("  %-20s (%s) -> %s", key, expect, d)
        else:
            logger.info("  %-20s (%s) -> NOT FOUND", key, expect)
