"""人工补丁层：step 5 跑完后把人审确认的字段合并下沉到所有 env5 产出。

适用场景：
    两个 target_field 语义上等价，但因 step3 的 qk 不同 或 subtype 不同，
    step5 的 Safety Net + coarse collapse 合并不到一起。
    人工在 env5_manual_merges.csv 里登记后，本脚本把合并信息应用到
    env5_main_list / env5_signature_* / env5_full_traceability。

幂等性：每次运行都先从 .bak_pre_patch 恢复原始文件，再重新应用补丁；
        所以修改 CSV 后重跑不会重复合并。

入口：
    python3 -m metaagent_run.steps.env_field_pipeline.new manual-patch
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

MANUAL_MERGES_CSV: Path = config.OUTPUT_DIR / "env5_manual_merges.csv"

# 被补丁的文件（主清单 + traceability + 4 份 signature）
PATCH_TARGETS: list[Path] = [
    config.PHASE5_MAIN_LIST,
    config.PHASE5_TRACE,
    *[config.OUTPUT_DIR / f"{config.PHASE5_SIGNATURE_PREFIX}{e}.csv"
      for e in config.HYDRO_ENVS],
]

ENV_COLS: list[str] = [f"env_{e}" for e in config.HYDRO_ENVS]
ENV_PCT_COLS: list[str] = [f"env_pct_{e}" for e in config.HYDRO_ENVS]


# ── 工具 ──────────────────────────────────────────────────────────
def _backup_or_restore(path: Path) -> Path:
    """首次跑备份；后续跑从 .bak_pre_patch 恢复（保证幂等）。"""
    bak = path.with_suffix(path.suffix + ".bak_pre_patch")
    if bak.exists():
        shutil.copy(bak, path)
        logger.info("Restored %s from %s", path.name, bak.name)
    else:
        shutil.copy(path, bak)
        logger.info("Backed up %s → %s", path.name, bak.name)
    return bak


def _recompute_metrics(row: pd.Series) -> pd.Series:
    """合并后 env 向量变了，重新算 H_norm / dominant_* / env_pct_*."""
    env_vec = np.array([float(row[c]) for c in ENV_COLS])
    total = env_vec.sum()
    if total <= 0:
        row["H_norm"] = 0.0
        row["dominant_env"] = config.HYDRO_ENVS[int(env_vec.argmax())]
        row["dominant_share"] = 0.0
        for c in ENV_PCT_COLS:
            row[c] = 0.0
        row["n_envs_present"] = 0
        return row
    probs = env_vec / total
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(probs > 0, np.log(probs), 0.0)
        H = -(probs * log_p).sum()
    row["H_norm"] = float(H / np.log(len(config.HYDRO_ENVS)))
    dom_idx = int(probs.argmax())
    row["dominant_env"] = config.HYDRO_ENVS[dom_idx]
    row["dominant_share"] = float(probs.max())
    for c, p in zip(ENV_PCT_COLS, probs):
        row[c] = round(float(p * 100), 2)
    row["n_envs_present"] = int((env_vec > 0).sum())
    return row


def _pipe_concat(a, b) -> str:
    a = "" if pd.isna(a) else str(a)
    b = "" if pd.isna(b) else str(b)
    parts = [p for p in (a, b) if p]
    return "|".join(parts)


def _merge_rows(host: pd.Series, src: pd.Series, src_name: str) -> pd.Series:
    """把 src 行累加进 host 行。返回新 host."""
    new = host.copy()
    # env 向量累加
    for c in ENV_COLS:
        new[c] = int(host[c]) + int(src[c])
    new["total_pmid"] = int(host["total_pmid"]) + int(src["total_pmid"])
    new["n_member_raw_keys"] = (
        int(host["n_member_raw_keys"]) + int(src["n_member_raw_keys"])
    )
    new["member_raw_keys"] = _pipe_concat(
        host.get("member_raw_keys"), src.get("member_raw_keys"))
    new["merged_canonicals"] = _pipe_concat(
        host.get("merged_canonicals"), src.get("merged_canonicals"))
    new["merged_bag_variants"] = _pipe_concat(
        host.get("merged_bag_variants"), src.get("merged_bag_variants"))
    # 重算指标
    new = _recompute_metrics(new)
    # 标注补丁痕迹
    prev = str(new.get("manual_patch_absorbed") or "")
    new["manual_patch_absorbed"] = _pipe_concat(prev, src_name)
    return new


# ── 主流程 ────────────────────────────────────────────────────────
def run() -> None:
    if not MANUAL_MERGES_CSV.exists():
        logger.warning(
            "No manual merges file found at %s — creating an empty "
            "template for you to edit.", MANUAL_MERGES_CSV,
        )
        template = pd.DataFrame([
            {"source_target_field": "",
             "absorbed_into": "",
             "reason": ""},
        ])
        template.to_csv(MANUAL_MERGES_CSV, index=False)
        return

    merges = pd.read_csv(MANUAL_MERGES_CSV)
    merges = merges[
        merges["source_target_field"].notna()
        & merges["absorbed_into"].notna()
        & (merges["source_target_field"].astype(str).str.strip() != "")
    ].copy()
    if not len(merges):
        logger.info("Manual merges file is empty. Nothing to do.")
        return
    logger.info("Found %d manual merge rules:", len(merges))
    for _, r in merges.iterrows():
        logger.info("  %s  →  %s   (%s)",
                    r["source_target_field"], r["absorbed_into"],
                    str(r.get("reason", ""))[:80])

    # 1. 恢复/备份 → 加载 main list
    for p in PATCH_TARGETS:
        if p.exists():
            _backup_or_restore(p)

    main = pd.read_csv(config.PHASE5_MAIN_LIST)
    trace = pd.read_csv(config.PHASE5_TRACE)

    # 若没有 manual_patch_absorbed 列则新增
    if "manual_patch_absorbed" not in main.columns:
        main["manual_patch_absorbed"] = ""

    # 2. 应用到主清单
    applied: list[dict] = []
    for _, mr in merges.iterrows():
        src_name = str(mr["source_target_field"]).strip()
        host_name = str(mr["absorbed_into"]).strip()

        src_mask = main["target_field_name"] == src_name
        host_mask = main["target_field_name"] == host_name
        if not src_mask.any():
            logger.warning(
                "  source '%s' not in main list — skip (maybe already patched?)",
                src_name,
            )
            continue
        if not host_mask.any():
            logger.warning(
                "  host '%s' not in main list — skip",
                host_name,
            )
            continue

        src_row = main[src_mask].iloc[0]
        host_row = main[host_mask].iloc[0]

        new_host = _merge_rows(host_row, src_row, src_name)
        main.loc[host_mask] = new_host.values
        main = main[~src_mask].reset_index(drop=True)
        applied.append({
            "source": src_name, "host": host_name,
            "source_pmid": int(src_row["total_pmid"]),
            "host_pmid_after": int(new_host["total_pmid"]),
            "source_canonicals": src_row["merged_canonicals"],
            "reason": str(mr.get("reason", "")),
        })
        logger.info(
            "  ✓ merged %s (pmid=%d) into %s (new pmid=%d, env_dist=%s%%)",
            src_name, src_row["total_pmid"], host_name,
            new_host["total_pmid"],
            {e: round(new_host[f"env_pct_{e}"], 1) for e in config.HYDRO_ENVS},
        )

    # 3. 重排序 + 写回
    main = main.sort_values("total_pmid", ascending=False).reset_index(drop=True)
    main.to_csv(config.PHASE5_MAIN_LIST, index=False)
    logger.info("Wrote patched main list (%d rows) → %s",
                len(main), config.PHASE5_MAIN_LIST)

    # 4. traceability 更新：把 source 的所有 raw_key 行改指向 host
    n_trace_updated = 0
    if "manual_patch_absorbed" not in trace.columns:
        trace["manual_patch_absorbed"] = ""
    for a in applied:
        mask = trace["target_field_name"] == a["source"]
        n_trace_updated += int(mask.sum())
        trace.loc[mask, "target_field_name"] = a["host"]
        trace.loc[mask, "collapse_reason"] = (
            trace.loc[mask, "collapse_reason"].astype(str)
            + "|manual_patch_to_target"
        )
        trace.loc[mask, "manual_patch_absorbed"] = a["source"]
    trace.to_csv(config.PHASE5_TRACE, index=False)
    logger.info("Wrote patched traceability (%d rows updated) → %s",
                n_trace_updated, config.PHASE5_TRACE)

    # 5. signature 文件：若 source 或 host 出现，同样处理
    # （本次两对都在主清单，signature 其实不会命中，但留兜底）
    for env in config.HYDRO_ENVS:
        sig_path = config.OUTPUT_DIR / f"{config.PHASE5_SIGNATURE_PREFIX}{env}.csv"
        if not sig_path.exists():
            continue
        sig = pd.read_csv(sig_path)
        if "manual_patch_absorbed" not in sig.columns:
            sig["manual_patch_absorbed"] = ""
        changed = False
        for a in applied:
            sm = sig["target_field_name"] == a["source"]
            if sm.any():
                changed = True
                # 若 host 不在该 sig 文件，把 source 重命名即可
                hm = sig["target_field_name"] == a["host"]
                if hm.any():
                    sr = sig[sm].iloc[0]
                    hr = sig[hm].iloc[0]
                    new_hr = _merge_rows(hr, sr, a["source"])
                    sig.loc[hm] = new_hr.values
                    sig = sig[~sm].reset_index(drop=True)
                else:
                    sig.loc[sm, "target_field_name"] = a["host"]
                    sig.loc[sm, "manual_patch_absorbed"] = a["source"]
        if changed:
            sig.to_csv(sig_path, index=False)
            logger.info("Updated signature %s", sig_path.name)

    # 6. 审计日志
    log_path = config.OUTPUT_DIR / "env5_manual_patch_log.csv"
    pd.DataFrame(applied).to_csv(log_path, index=False)
    logger.info("Wrote patch audit log → %s", log_path)

    logger.info("=" * 60)
    logger.info("MANUAL PATCH SUMMARY")
    logger.info("=" * 60)
    logger.info("Rules in CSV: %d   Applied: %d", len(merges), len(applied))
    logger.info("Main list:    %d rows (was %d)",
                len(main), len(main) + len(applied))
    logger.info("Traceability: %d rows updated in place", n_trace_updated)
    for a in applied:
        logger.info("  %s → %s  (pmid +%d)",
                    a["source"], a["host"], a["source_pmid"])
