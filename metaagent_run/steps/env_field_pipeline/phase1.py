"""环节 1：低频 raw key 隔离。

规则：
    - total_pmid >= 3 → 主流（env1_mainstream_raw_keys.csv）
    - total_pmid < 3 → 孤儿池（env1_orphan_pool_raw_keys.csv），附 evidence_record_ids

evidence_record_ids 定义：
    - 复合 id 字符串 "pmid|source|section_type|index"
    - 每条孤儿 raw key 最多保留 3 条来源段落 id，便于事后人工查询
"""
import json
import logging

import pandas as pd

from . import config

logger = logging.getLogger(__name__)

EVIDENCE_CAP_PER_KEY = 3
ORPHAN_THRESHOLD = 3  # total_pmid < 3 → orphan


def _record_id(r: dict) -> str:
    return f"{r.get('pmid','')}|{r.get('source','')}|{r.get('section_type','')}|{r.get('index','')}"


def _build_orphan_evidence_index(orphan_keys: set[str]) -> dict[str, list[str]]:
    """扫一次 step2，为每个孤儿 raw key 收集最多 N 条段落 id。"""
    logger.info("Building evidence index for %d orphan raw keys", len(orphan_keys))
    with open(config.STEP2_INPUT, "r", encoding="utf-8") as f:
        records = json.load(f)
    index: dict[str, list[str]] = {k: [] for k in orphan_keys}
    full_count = 0
    for r in records:
        keys = r.get("metadata_keys_found") or []
        if not keys:
            continue
        rid = None  # 惰性生成
        for k in keys:
            if k in index and len(index[k]) < EVIDENCE_CAP_PER_KEY:
                if rid is None:
                    rid = _record_id(r)
                index[k].append(rid)
                if len(index[k]) == EVIDENCE_CAP_PER_KEY:
                    full_count += 1
    logger.info(
        "Orphan keys with evidence full (>=%d): %d",
        EVIDENCE_CAP_PER_KEY, full_count,
    )
    return index


def run() -> None:
    config.ensure_output_dir()

    logger.info("Loading env0 output: %s", config.PHASE0_OUTPUT)
    df = pd.read_csv(config.PHASE0_OUTPUT)
    logger.info("Loaded %d raw keys", len(df))

    mainstream = df[df["total_pmid"] >= ORPHAN_THRESHOLD].copy()
    orphan = df[df["total_pmid"] < ORPHAN_THRESHOLD].copy()
    assert len(mainstream) + len(orphan) == len(df), "split leak"
    assert (mainstream["total_pmid"] >= ORPHAN_THRESHOLD).all(), "mainstream violates threshold"

    logger.info("mainstream: %d; orphan: %d", len(mainstream), len(orphan))

    # Orphan evidence
    orphan_keys = set(orphan["raw_key"].tolist())
    ev_index = _build_orphan_evidence_index(orphan_keys)
    orphan["evidence_record_ids"] = orphan["raw_key"].map(
        lambda k: ";".join(ev_index.get(k, []))
    )

    mainstream.to_csv(config.PHASE1_MAINSTREAM, index=False)
    orphan.to_csv(config.PHASE1_ORPHAN, index=False)
    logger.info("Wrote mainstream to %s", config.PHASE1_MAINSTREAM)
    logger.info("Wrote orphan to %s", config.PHASE1_ORPHAN)

    stats = _compute_stats(df, mainstream, orphan)
    with open(config.PHASE1_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("Wrote stats to %s", config.PHASE1_STATS)

    _print_verification(stats, mainstream, orphan)


def _compute_stats(all_df: pd.DataFrame, main: pd.DataFrame, orp: pd.DataFrame) -> dict:
    all_pmid_sum = int(all_df["total_pmid"].sum())
    main_pmid_sum = int(main["total_pmid"].sum())
    orp_pmid_sum = int(orp["total_pmid"].sum())
    return {
        "threshold_total_pmid_ge": ORPHAN_THRESHOLD,
        "total_raw_keys": int(len(all_df)),
        "mainstream_count": int(len(main)),
        "orphan_count": int(len(orp)),
        "split_sum_matches": int(len(main) + len(orp)) == int(len(all_df)),
        "total_pmid_sum_all": all_pmid_sum,
        "total_pmid_sum_mainstream": main_pmid_sum,
        "total_pmid_sum_orphan": orp_pmid_sum,
        "mainstream_pmid_coverage_pct": round(
            100.0 * main_pmid_sum / max(1, all_pmid_sum), 2
        ),
        "orphan_total_pmid_breakdown": {
            str(k): int(v)
            for k, v in orp["total_pmid"].value_counts().sort_index().items()
        },
    }


def _print_verification(stats: dict, main: pd.DataFrame, orp: pd.DataFrame) -> None:
    logger.info("=" * 60)
    logger.info("PHASE 1 VERIFICATION")
    logger.info("=" * 60)
    for k, v in stats.items():
        logger.info("  %s: %s", k, v)
    logger.info("")
    logger.info("Top 10 mainstream by total_pmid:")
    logger.info("\n%s", main.head(10).to_string(index=False))
    logger.info("")
    logger.info("Random 20 orphan samples:")
    logger.info(
        "\n%s",
        orp.sample(min(20, len(orp)), random_state=42).to_string(index=False),
    )
