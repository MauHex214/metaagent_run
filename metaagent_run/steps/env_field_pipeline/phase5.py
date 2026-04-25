"""环节 5：环境感知 collapse，产出主清单 + 4 份 Signature + 追溯表。

两个入口：
    5-calibrate  扫描 H × PMID × share 五维网格，输出阈值参考表，不产出清单
    5           读 env5_thresholds.json 应用阈值，产出所有清单

Rule A 主清单（两层 collapse）：
    第一层 Safety Net（总是执行）：
        按 (family, subtype, quantity_kind, modifier_bag) dropna=False 分组；
        同组若 ≥ 2 canonical 说明 phase4 漏网，合并之。
    第二层 Granularity Collapse（config.granularity_mode）：
        fine   每个 canonical 独立成 target_field
        coarse 按 (family, subtype, quantity_kind) 分组，PMID 最高的 canonical
               作为 target_field，其他成 synonyms（含 bag 变体信息）

Rule B Signature：
    按 category=Signature 的 canonical，按 dominant_env 分 4 组导出。

modifier_bag / family / subtype 字段始终保留原判，不因 collapse 改变。
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .phase4 import _pick_representative

logger = logging.getLogger(__name__)

# ── 路径 ───────────────────────────────────────────────────────────
PHASE5_THRESHOLD_CAL = config.PHASE5_THRESHOLD_CAL
PHASE5_MAIN_LIST = config.PHASE5_MAIN_LIST
PHASE5_SIGNATURE_PREFIX = config.PHASE5_SIGNATURE_PREFIX
PHASE5_TRACE = config.PHASE5_TRACE
PHASE5_THRESHOLDS = config.OUTPUT_DIR / "env5_thresholds.json"
PHASE5_CANONICAL_CLASSIFIED = config.OUTPUT_DIR / "env5_canonical_classified.csv"


# Final thresholds adopted after sensitivity analysis on cross_min ∈ {10, 30, 50}
# (see docs/env5_thresholds.reference.json for the canonical config and
# docs/phase6_decisions.md §H_norm/dom_share rationale).
#
# Decision rationale:
#   - pmid_universal_min=50: a universal field must have ≥50-paper coverage
#     across well-distributed environments (H_norm ≥ 0.85)
#   - pmid_cross_min=30:    chosen to retain numeric fields like
#     mixed_layer_depth, ice_thickness, secchi_disk_depth, sediment_mean_grain_size,
#     porewater_salinity (all PMID 30-49); cross_min=50 would drop ~150 numeric
#     fields, cross_min=10 admits too many low-coverage descriptive fields.
#   - pmid_signature_min=5: signature fields are inherently rare per environment
#     (e.g. peat_depth=22 in wetlands, lake_area=87 in lakes); 5 is a low but
#     defensible floor for "sub-environment-specific" claim.
#   - granularity_mode='coarse': merge bag variants (e.g. salinity / surface_salinity
#     / bottom_salinity) into a single target with `merged_bag_variants` log;
#     keeps the main list scale aligned with downstream step5 prompt budget.
DEFAULT_THRESHOLDS: dict[str, Any] = {
    "H_universal_min": 0.85,
    "pmid_universal_min": 50,
    "pmid_cross_min": 30,
    "H_signature_max": 0.4,
    "dominant_share_min": 0.7,
    "pmid_signature_min": 5,
    "granularity_mode": "coarse",  # fine | coarse
}


# ── 指标计算 ────────────────────────────────────────────────────────
def compute_env_metrics(df: pd.DataFrame) -> pd.DataFrame:
    env_cols = [f"env_{e}" for e in config.HYDRO_ENVS]
    totals = df[env_cols].sum(axis=1).replace(0, np.nan)
    probs = df[env_cols].div(totals, axis=0).fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(probs > 0, np.log(probs), 0.0)
        H = -(probs * log_p).sum(axis=1)
    H_norm = H / np.log(len(config.HYDRO_ENVS))
    dom_idx = probs.values.argmax(axis=1)
    dom_env = [config.HYDRO_ENVS[i] for i in dom_idx]
    dom_share = probs.values.max(axis=1)

    out = df.copy()
    out["H_norm"] = H_norm.values
    out["dominant_env"] = dom_env
    out["dominant_share"] = dom_share
    for i, e in enumerate(config.HYDRO_ENVS):
        out[f"env_pct_{e}"] = (probs.iloc[:, i] * 100).round(2)
    return out


def classify_canonicals(df: pd.DataFrame, thr: dict) -> pd.Series:
    """返回 category 列: Universal / Cross-biome common / Signature / Niche。"""
    H = df["H_norm"].values
    pmid = df["total_pmid"].values
    dom = df["dominant_share"].values

    is_universal = (H > thr["H_universal_min"]) & (pmid > thr["pmid_universal_min"])
    is_signature = (H < thr["H_signature_max"]) & \
                   (dom > thr["dominant_share_min"]) & \
                   (pmid > thr["pmid_signature_min"])
    # Cross-biome: H 中段，pmid 较高，不是 Universal / Signature
    is_cross = (H > thr["H_signature_max"]) & (H <= thr["H_universal_min"]) & \
               (pmid > thr["pmid_cross_min"]) & \
               (~is_universal) & (~is_signature)

    cats = np.full(len(df), "Niche", dtype=object)
    cats[is_universal] = "Universal"
    cats[is_cross] = "Cross-biome common"
    cats[is_signature] = "Signature"
    return pd.Series(cats, index=df.index)


# ── 5-calibrate ───────────────────────────────────────────────────
def calibrate() -> None:
    config.ensure_output_dir()
    can = pd.read_csv(config.PHASE4_CANONICALS)
    logger.info("Loaded %d canonicals", len(can))
    can = compute_env_metrics(can)

    rows = []
    H_univ_grid = [round(x, 2) for x in np.arange(0.70, 0.96, 0.05)]
    pmid_univ_grid = [20, 30, 50, 75, 100]
    H_sig_grid = [round(x, 2) for x in np.arange(0.30, 0.56, 0.05)]
    share_sig_grid = [0.60, 0.70, 0.80]
    pmid_sig_grid = [3, 5, 10]

    for H_univ in H_univ_grid:
        for pmid_univ in pmid_univ_grid:
            for H_sig in H_sig_grid:
                for share_sig in share_sig_grid:
                    for pmid_sig in pmid_sig_grid:
                        pmid_cross = max(pmid_sig, pmid_univ // 3)
                        thr = {
                            "H_universal_min": H_univ,
                            "pmid_universal_min": pmid_univ,
                            "pmid_cross_min": pmid_cross,
                            "H_signature_max": H_sig,
                            "dominant_share_min": share_sig,
                            "pmid_signature_min": pmid_sig,
                        }
                        cats = classify_canonicals(can, thr)
                        n_u = int((cats == "Universal").sum())
                        n_c = int((cats == "Cross-biome common").sum())
                        n_s = int((cats == "Signature").sum())
                        n_n = int((cats == "Niche").sum())
                        rows.append({
                            "H_universal_min": H_univ,
                            "pmid_universal_min": pmid_univ,
                            "pmid_cross_min": pmid_cross,
                            "H_signature_max": H_sig,
                            "dominant_share_min": share_sig,
                            "pmid_signature_min": pmid_sig,
                            "n_universal": n_u,
                            "n_cross_biome": n_c,
                            "n_signature": n_s,
                            "n_niche": n_n,
                            "n_main_list": n_u + n_c,
                            "n_signature_total": n_s,
                        })
    cal_df = pd.DataFrame(rows)
    cal_df.to_csv(PHASE5_THRESHOLD_CAL, index=False)
    logger.info("Wrote %d threshold combinations → %s",
                len(cal_df), PHASE5_THRESHOLD_CAL)

    # 默认阈值 preview
    default_cats = classify_canonicals(can, DEFAULT_THRESHOLDS)
    default_dist = {
        "Universal": int((default_cats == "Universal").sum()),
        "Cross-biome common": int((default_cats == "Cross-biome common").sum()),
        "Signature": int((default_cats == "Signature").sum()),
        "Niche": int((default_cats == "Niche").sum()),
    }

    # 写默认阈值 JSON（用户编辑后跑 5）
    if not PHASE5_THRESHOLDS.exists():
        with open(PHASE5_THRESHOLDS, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_THRESHOLDS, f, ensure_ascii=False, indent=2)
        logger.info("Default thresholds → %s", PHASE5_THRESHOLDS)

    logger.info("=" * 60)
    logger.info("PHASE 5 CALIBRATION")
    logger.info("=" * 60)
    logger.info("Total canonicals: %d", len(can))
    logger.info("Default threshold preview: %s", default_dist)
    logger.info(f"Default 主清单大小 (U + C): "
                f"{default_dist['Universal'] + default_dist['Cross-biome common']}")
    logger.info("")
    logger.info("H_univ=0.85, pmid_univ={20,30,50,75,100} 下的主清单大小:")
    default_row = cal_df[
        (cal_df.H_universal_min == 0.85) &
        (cal_df.H_signature_max == 0.40) &
        (cal_df.dominant_share_min == 0.70) &
        (cal_df.pmid_signature_min == 5)
    ].sort_values("pmid_universal_min")
    if len(default_row):
        logger.info("\n%s", default_row[[
            "pmid_universal_min", "pmid_cross_min",
            "n_universal", "n_cross_biome", "n_signature",
            "n_main_list"
        ]].to_string(index=False))
    logger.info("")
    logger.info(">> 请审阅 %s 后编辑 %s 定稿，然后运行",
                PHASE5_THRESHOLD_CAL, PHASE5_THRESHOLDS)
    logger.info(">>   python3 -m metaagent_run.steps.env_field_pipeline.new 5")


# ── Rule A: Safety Net + Granularity Collapse ────────────────────
def _apply_safety_net(can_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """第一层：按 (family, subtype, qk, bag) dropna=False 分组。
    同组多 canonical → 合并成一个（PMID 最高为代表，其他 canonical_id 存 safety_net_merged_ids）。
    返回合并后的 df，以及 {被吸收的 canonical_id: target canonical_id}。
    """
    absorbed_map: dict[str, str] = {}  # absorbed_cid -> target_cid
    groups = can_df.groupby(
        ["family", "subtype", "quantity_kind", "modifier_bag"],
        dropna=False, sort=False,
    )
    kept_rows = []
    safety_log = []
    env_cols = [f"env_{e}" for e in config.HYDRO_ENVS]

    for _, grp in groups:
        if len(grp) == 1:
            kept_rows.append(grp.iloc[0].to_dict())
            continue
        # 多 canonical → safety net
        target_idx = grp["total_pmid"].idxmax()
        target_row = grp.loc[target_idx].to_dict()
        # v3 代表名规则跨 canonical_name 选最清晰者（PMID 作 freq），
        # 身份仍以 PMID 最高那条为准（canonical_id/family/subtype/...）
        names = grp["canonical_name"].astype(str).tolist()
        freq = dict(zip(names, grp["total_pmid"].astype(int).tolist()))
        target_row["canonical_name"] = _pick_representative(names, freq)
        absorbed = grp[grp.index != target_idx]
        # 累加 env 向量、total_pmid、n_member_raw_keys
        for col in env_cols:
            target_row[col] = int(grp[col].sum())
        target_row["total_pmid"] = int(grp["total_pmid"].sum())
        target_row["n_envs_present"] = sum(
            1 for e in config.HYDRO_ENVS if target_row[f"env_{e}"] > 0
        )
        target_row["n_member_raw_keys"] = int(grp["n_member_raw_keys"].sum())
        target_row["member_raw_keys"] = "|".join(
            grp["member_raw_keys"].astype(str).tolist()
        )
        for aid in absorbed["canonical_id"]:
            absorbed_map[aid] = target_row["canonical_id"]
            safety_log.append({
                "absorbed_canonical": aid,
                "target_canonical": target_row["canonical_id"],
                "key": f"{target_row['family']}|{target_row['subtype']}|"
                       f"{target_row['quantity_kind']}|{target_row.get('modifier_bag','')}",
            })
        kept_rows.append(target_row)
    out = pd.DataFrame(kept_rows)
    logger.info("Safety-net: %d canonicals absorbed → %d target rows (was %d)",
                len(absorbed_map), len(out), len(can_df))
    return out, absorbed_map


def _apply_granularity_collapse(
    can_df: pd.DataFrame, mode: str,
) -> tuple[pd.DataFrame, dict]:
    """第二层 Rule A collapse（仅对 main list 使用；Signature 不走此步）。

    fine:   每个 canonical 直接成 target_field
    coarse: 按 (family, subtype, qk) 合并，PMID 最高代表，其他 bag 变体成 synonyms
    """
    bag_collapsed_map: dict[str, str] = {}
    env_cols = [f"env_{e}" for e in config.HYDRO_ENVS]

    if mode == "fine":
        out = can_df.copy()
        out["target_field_name"] = out["canonical_name"]
        out["merged_canonicals"] = out["canonical_id"]
        out["merged_bag_variants"] = out["modifier_bag"].fillna("").astype(str)
        out["granularity_mode"] = "fine"
        return out, bag_collapsed_map

    # coarse
    groups = can_df.groupby(
        ["family", "subtype", "quantity_kind"], sort=False,
    )
    kept = []
    for _, grp in groups:
        if len(grp) == 1:
            r = grp.iloc[0].to_dict()
            r["target_field_name"] = r["canonical_name"]
            r["merged_canonicals"] = r["canonical_id"]
            r["merged_bag_variants"] = str(r.get("modifier_bag") or "")
            r["granularity_mode"] = "coarse"
            kept.append(r)
            continue
        target_idx = grp["total_pmid"].idxmax()
        target_row = grp.loc[target_idx].to_dict()
        # v3 代表名规则：跨被合并 canonical 的 canonical_name 里挑最清晰者作 target_field_name
        # （例如 toc/total_organic_carbon 都入组时，挑 total_organic_carbon 而非 PMID 最大的 toc）
        names = grp["canonical_name"].astype(str).tolist()
        freq = dict(zip(names, grp["total_pmid"].astype(int).tolist()))
        target_row["target_field_name"] = _pick_representative(names, freq)
        target_row["granularity_mode"] = "coarse"
        target_row["merged_canonicals"] = "|".join(grp["canonical_id"].tolist())
        bag_variants = grp["modifier_bag"].fillna("").astype(str).tolist()
        target_row["merged_bag_variants"] = "|".join(
            v if v else "(empty)" for v in bag_variants
        )
        # 聚合
        for col in env_cols:
            target_row[col] = int(grp[col].sum())
        target_row["total_pmid"] = int(grp["total_pmid"].sum())
        target_row["n_member_raw_keys"] = int(grp["n_member_raw_keys"].sum())
        member_strs = grp["member_raw_keys"].astype(str).tolist()
        target_row["member_raw_keys"] = "|".join(member_strs)
        # 记录被合并
        absorbed = grp[grp.index != target_idx]
        for aid in absorbed["canonical_id"]:
            bag_collapsed_map[aid] = target_row["canonical_id"]
        kept.append(target_row)
    out = pd.DataFrame(kept)
    logger.info("Granularity coarse: %d canonicals absorbed via bag collapse → %d targets",
                len(bag_collapsed_map), len(out))
    return out, bag_collapsed_map


# ── 5 入口：主流程 ─────────────────────────────────────────────────
def run() -> None:
    config.ensure_output_dir()

    if not PHASE5_THRESHOLDS.exists():
        logger.warning("Threshold config not found, using defaults")
        thr = DEFAULT_THRESHOLDS.copy()
    else:
        with open(PHASE5_THRESHOLDS, "r", encoding="utf-8") as f:
            thr = json.load(f)
    # 补全缺失键
    for k, v in DEFAULT_THRESHOLDS.items():
        thr.setdefault(k, v)
    logger.info("Thresholds: %s", thr)
    mode = thr.get("granularity_mode", "fine")
    if mode not in {"fine", "coarse"}:
        raise ValueError(f"granularity_mode must be fine|coarse, got {mode}")
    logger.info("Granularity mode: %s", mode)

    can = pd.read_csv(config.PHASE4_CANONICALS)
    map_df = pd.read_csv(config.PHASE4_MAPPING)
    annot = pd.read_csv(config.PHASE3_OUTPUT)
    logger.info("canonicals=%d; mappings=%d; annotations=%d",
                len(can), len(map_df), len(annot))

    # 1. 指标
    can = compute_env_metrics(can)
    can["category"] = classify_canonicals(can, thr)
    # 写完整分类表（调试/论文附录用）
    can.to_csv(PHASE5_CANONICAL_CLASSIFIED, index=False)
    logger.info("Wrote canonical_classified (all %d) → %s",
                len(can), PHASE5_CANONICAL_CLASSIFIED)

    cat_dist = can["category"].value_counts().to_dict()
    logger.info("Category distribution: %s", cat_dist)

    # 2. Rule A 主清单
    main_candidates = can[
        can["category"].isin({"Universal", "Cross-biome common"})
    ].copy()
    logger.info("Main candidates (Universal + Cross-biome): %d", len(main_candidates))

    # Safety net
    main_after_sn, sn_map = _apply_safety_net(main_candidates)
    # Granularity collapse
    main_final, bag_map = _apply_granularity_collapse(main_after_sn, mode)
    main_final = main_final.sort_values("total_pmid", ascending=False).reset_index(drop=True)

    # 3. Rule B Signature
    sig_can = can[can["category"] == "Signature"].copy()
    sig_final, sig_sn_map = _apply_safety_net(sig_can) if len(sig_can) else (sig_can, {})
    # Signature 不做 granularity collapse（保留细粒度，签名字段本就稀有）
    sig_final["target_field_name"] = sig_final["canonical_name"]
    sig_final["merged_canonicals"] = sig_final["canonical_id"]
    sig_final["merged_bag_variants"] = sig_final.get("modifier_bag", "").fillna("").astype(str)
    sig_final["granularity_mode"] = "fine"

    # 4. 输出主清单
    main_cols = [
        "target_field_name", "canonical_id", "canonical_name",
        "family", "subtype", "quantity_kind", "modifier_bag",
        "category", "granularity_mode",
        "merged_canonicals", "merged_bag_variants",
        "H_norm", "dominant_env", "dominant_share",
        "total_pmid", "n_envs_present",
        *[f"env_{e}" for e in config.HYDRO_ENVS],
        *[f"env_pct_{e}" for e in config.HYDRO_ENVS],
        "mixs_slot", "mixs_alignment", "mixs_reasoning",
        "n_member_raw_keys", "member_raw_keys",
    ]
    main_cols = [c for c in main_cols if c in main_final.columns]
    main_final[main_cols].to_csv(PHASE5_MAIN_LIST, index=False)
    logger.info("Main list (%s) → %s  (%d rows)",
                mode, PHASE5_MAIN_LIST, len(main_final))

    # 5. Signature 分 4 文件
    sig_counts = {}
    for env in config.HYDRO_ENVS:
        sub = sig_final[sig_final["dominant_env"] == env].copy()
        sub = sub.sort_values("total_pmid", ascending=False)
        out_path = config.OUTPUT_DIR / f"{PHASE5_SIGNATURE_PREFIX}{env}.csv"
        sub[main_cols].to_csv(out_path, index=False)
        sig_counts[env] = len(sub)
        logger.info("Signature %s → %s (%d rows)", env, out_path, len(sub))

    # 6. 追溯表
    _build_traceability(
        can_df=can, map_df=map_df, annot_df=annot,
        main_final=main_final, sig_final=sig_final,
        sn_map=sn_map, bag_map=bag_map, sig_sn_map=sig_sn_map,
    )

    # 7. 验证
    _verify(can, main_final, sig_counts, mode)


def _build_traceability(
    can_df: pd.DataFrame,
    map_df: pd.DataFrame,
    annot_df: pd.DataFrame,
    main_final: pd.DataFrame,
    sig_final: pd.DataFrame,
    sn_map: dict, bag_map: dict, sig_sn_map: dict,
) -> None:
    """raw_key → canonical → target_field 的完整追溯。"""
    # canonical_id → target_field_name
    cid_to_target: dict[str, str] = {}
    for _, r in main_final.iterrows():
        # Fine: merged_canonicals = 单 canonical_id
        # Coarse: merged_canonicals = pipe 分隔
        for cid in str(r["merged_canonicals"]).split("|"):
            if cid:
                cid_to_target[cid] = r["target_field_name"]
    for _, r in sig_final.iterrows():
        for cid in str(r["merged_canonicals"]).split("|"):
            if cid:
                cid_to_target[cid] = r["target_field_name"]

    # safety-net 合并：吸收 canonical 的 raw_key 也映射到目标
    # 但实际 safety_net 已在 main_final 里把合并后的 canonical 成为代表；
    # 原 canonical_id 通过 sn_map 找到目标
    sn_resolve = dict(sn_map)   # absorbed -> target
    sn_resolve.update(sig_sn_map)

    # category & dom_env by canonical_id
    can_meta = dict(zip(can_df["canonical_id"], zip(
        can_df["category"], can_df["dominant_env"]
    )))
    can_subtype = dict(zip(can_df["canonical_id"], can_df["family"]))

    # annotations: raw_key → (subtype, qk, bag)
    annot_map = {
        r["raw_key"]: (r["subtype"], r["quantity_kind"],
                       r.get("modifier_bag") if pd.notna(r.get("modifier_bag")) else "")
        for _, r in annot_df.iterrows()
    }

    # map_df: raw_key → canonical_id, is_representative
    trace_rows = []
    for _, r in map_df.iterrows():
        rk = r["raw_key"]
        cid_original = r["canonical_id"]
        # 追溯路径：rk → canonical_id → (safety_net 吸收后的 id) → (bag collapse 吸收后的 id) → target
        cid_final = cid_original
        path_reasons: list[str] = []
        # step 4 合并
        if r.get("is_representative", 1) == 0:
            path_reasons.append("merged_to_canonical")
        else:
            path_reasons.append("identity")
        # safety net
        if cid_final in sn_resolve:
            cid_final = sn_resolve[cid_final]
            path_reasons.append("safety_net_to_target")
        # bag collapse
        if cid_final in bag_map:
            cid_final = bag_map[cid_final]
            path_reasons.append("bag_collapsed_to_target")

        target = cid_to_target.get(cid_final, "")
        if not target:
            # 该 canonical 不在主清单也不在 signature → Niche
            category_guess, _ = can_meta.get(cid_original, ("Niche", ""))
            category = category_guess
            target = ""  # 空 → 未进入任何清单
        else:
            category, _dom = can_meta.get(cid_final, can_meta.get(cid_original, ("Niche", "")))

        subtype, qk, bag = annot_map.get(rk, ("", "", ""))
        trace_rows.append({
            "raw_key": rk,
            "step3_family": can_subtype.get(cid_original, ""),
            "step3_subtype": subtype,
            "step3_qk": qk,
            "step3_bag": bag or "",
            "canonical_id": cid_original,
            "canonical_name": can_df.loc[
                can_df["canonical_id"] == cid_original, "canonical_name"
            ].iloc[0] if (can_df["canonical_id"] == cid_original).any() else "",
            "target_field_name": target,
            "collapse_reason": "|".join(path_reasons),
            "category": category,
            "dominant_env": can_meta.get(cid_final, can_meta.get(cid_original, ("", "")))[1],
        })
    trace_df = pd.DataFrame(trace_rows)
    trace_df.to_csv(PHASE5_TRACE, index=False)
    logger.info("Traceability → %s (%d rows)", PHASE5_TRACE, len(trace_df))


def _verify(can, main_final, sig_counts, mode):
    logger.info("=" * 60)
    logger.info("PHASE 5 VERIFICATION")
    logger.info("=" * 60)
    logger.info("Mode: %s", mode)
    cat_dist = can["category"].value_counts().to_dict()
    logger.info("Category dist: %s", cat_dist)
    logger.info("Main list rows: %d", len(main_final))
    logger.info("Signature by env: %s", sig_counts)
    total_sig = sum(sig_counts.values())
    logger.info("Signature total: %d", total_sig)

    # 主清单 PMID 覆盖率（pipeline 内）
    main_pmid = int(main_final["total_pmid"].sum())
    all_pmid_pipeline = int(can["total_pmid"].sum())
    coverage_pipeline = 100.0 * main_pmid / max(1, all_pmid_pipeline)
    logger.info("Main-list PMID coverage (pipeline): %.1f%% (%d / %d)",
                coverage_pipeline, main_pmid, all_pmid_pipeline)

    # 经典大头字段检查
    must_have = [
        "salinity", "temperature", "ph", "oxygen",
        "latitude", "longitude", "depth",
        "collection_date", "nitrate", "phosphate",
        "chlorophyll", "habitat", "sediment_type",
    ]
    logger.info("")
    logger.info("Canonical name / target field 命中检查:")
    targets = set(main_final["target_field_name"].astype(str).str.lower())
    for n in must_have:
        hits = [t for t in targets if n in t]
        flag = "✓" if hits else "✗"
        logger.info("  %s %s  →  %s", flag, n, hits[:3])
