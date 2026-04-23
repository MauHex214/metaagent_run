"""环节 6：渲染最终抽取清单（step5 消费）。

输入：
    - phase6_schema.yaml         机器可读决策源（人工维护）
    - env5_main_list.csv         phase5 主清单（含 canonical_id, member_raw_keys, merged_bag_variants, total_pmid）
    - env5_signature_*.csv       4 份 signature
    - env4_canonicals.csv        含 canonical_id → member_raw_keys
    - env3_final_annotations.csv raw_key → raw_key_original 映射（追溯用）

输出：
    - env6_extraction_targets.json   step5.upstream_loader 消费
    - env6_main_schema.csv           主清单审计表
    - env6_signature_schema.csv      signature 审计表
    - env6_excluded_trace.csv        R1-R4 剔除追溯

Schema aliases 自动从 env5 member_raw_keys + merged_bag_variants 收集 +
merge_from_sig / merge_from_subtype / extra_aliases 合并。

入口：
    python3 -m metaagent_run.steps.env_field_pipeline.new 6
    # 或直接：python3 -m metaagent_run.steps.env_field_pipeline.phase6
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import config

logger = logging.getLogger(__name__)

PHASE6_SCHEMA_YAML: Path = config.OUTPUT_DIR / "phase6_schema.yaml"
PHASE6_EXTRACTION_TARGETS: Path = config.OUTPUT_DIR / "env6_extraction_targets.json"
PHASE6_MAIN_SCHEMA: Path = config.OUTPUT_DIR / "env6_main_schema.csv"
PHASE6_SIGNATURE_SCHEMA: Path = config.OUTPUT_DIR / "env6_signature_schema.csv"
PHASE6_EXCLUDED_TRACE: Path = config.OUTPUT_DIR / "env6_excluded_trace.csv"


# ── IO helpers ──────────────────────────────────────────────────────
def _load_yaml() -> dict:
    if not PHASE6_SCHEMA_YAML.exists():
        raise RuntimeError(f"Schema YAML not found: {PHASE6_SCHEMA_YAML}")
    with open(PHASE6_SCHEMA_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_env5_main() -> pd.DataFrame:
    df = pd.read_csv(config.PHASE5_MAIN_LIST)
    # 索引：target_field_name 可重复（不同 subtype 下可能同名），用 (subtype, target_field_name)
    return df


def _load_env5_signatures() -> dict[str, pd.DataFrame]:
    out = {}
    for env in config.HYDRO_ENVS:
        p = config.OUTPUT_DIR / f"{config.PHASE5_SIGNATURE_PREFIX}{env}.csv"
        if p.exists():
            out[env] = pd.read_csv(p)
    return out


def _load_env3() -> pd.DataFrame:
    return pd.read_csv(config.PHASE3_OUTPUT)


def _load_env4_canonicals() -> pd.DataFrame:
    return pd.read_csv(config.PHASE4_CANONICALS)


# ── Alias 收集核心 ───────────────────────────────────────────────────
def _split_pipe(s: Any) -> list[str]:
    """phase5 里 member_raw_keys 用 '|' 拼接。"""
    if pd.isna(s):
        return []
    parts = [p.strip() for p in str(s).split("|") if p.strip() and p.strip() != "(empty)"]
    return parts


def _normalize_alias(a: str, exclude: str) -> str | None:
    """清理 alias 字面，排除等于 target 自身的。"""
    a = str(a).strip()
    if not a or a.lower() == exclude.lower():
        return None
    return a


def _collect_aliases(
    target: str,
    main_df: pd.DataFrame,
    sig_dfs: dict[str, pd.DataFrame],
    env3_df: pd.DataFrame,
    entry: dict,
) -> tuple[list[str], dict[str, Any]]:
    """返回 (aliases_list, metadata_dict)."""
    subtype = entry.get("subtype")
    merge_from = entry.get("merge_from", []) or []
    merge_from_sig = entry.get("merge_from_sig", {}) or {}
    merge_from_subtype = entry.get("merge_from_subtype", {}) or {}
    extra_aliases = entry.get("extra_aliases", []) or []
    force_split = entry.get("force_split_from_canonical")

    aliases: set[str] = set()
    canonical_ids: set[str] = set()
    total_pmid = 0
    n_env5_rows = 0
    env_vec = {f"env_{e}": 0 for e in config.HYDRO_ENVS}

    # (a) 从主清单 merge_from 的 target 行收集
    for t in merge_from:
        rows = main_df[
            (main_df["subtype"] == subtype)
            & (main_df["target_field_name"].astype(str) == t)
        ]
        for _, r in rows.iterrows():
            n_env5_rows += 1
            total_pmid += int(r["total_pmid"])
            for e in config.HYDRO_ENVS:
                env_vec[f"env_{e}"] += int(r.get(f"env_{e}", 0) or 0)
            if pd.notna(r.get("canonical_id")):
                canonical_ids.add(str(r["canonical_id"]))
            for rk in _split_pipe(r.get("member_raw_keys")):
                na = _normalize_alias(rk, target)
                if na:
                    aliases.add(na)
            for v in _split_pipe(r.get("merged_bag_variants")):
                if v and v != "(empty)":
                    # merged_bag_variants 是 bag 变体（如 surface/mean/bottom），不是 raw_key
                    # 不加入 aliases
                    pass
            # merged_canonicals = 合并的 canonical_id list
            for cid in _split_pipe(r.get("merged_canonicals")):
                canonical_ids.add(cid)

    # (b) 从 signature 收集（合入主清单的）
    for env, sig_targets in merge_from_sig.items():
        sdf = sig_dfs.get(env)
        if sdf is None:
            continue
        for t in sig_targets:
            rows = sdf[sdf["target_field_name"].astype(str) == t]
            for _, r in rows.iterrows():
                total_pmid += int(r["total_pmid"])
                if pd.notna(r.get("canonical_id")):
                    canonical_ids.add(str(r["canonical_id"]))
                for rk in _split_pipe(r.get("member_raw_keys")):
                    na = _normalize_alias(rk, target)
                    if na:
                        aliases.add(na)
                # 同时 aliases 加 signature target 本身名字
                na = _normalize_alias(t, target)
                if na:
                    aliases.add(na)

    # (c) 从其他 subtype 挪过来的 target
    for from_sub, ts in merge_from_subtype.items():
        for t in ts:
            rows = main_df[
                (main_df["subtype"] == from_sub)
                & (main_df["target_field_name"].astype(str) == t)
            ]
            for _, r in rows.iterrows():
                total_pmid += int(r["total_pmid"])
                if pd.notna(r.get("canonical_id")):
                    canonical_ids.add(str(r["canonical_id"]))
                for rk in _split_pipe(r.get("member_raw_keys")):
                    na = _normalize_alias(rk, target)
                    if na:
                        aliases.add(na)

    # (d) extra_aliases 手工添加
    for a in extra_aliases:
        na = _normalize_alias(a, target)
        if na:
            aliases.add(na)

    # (e) force_split_from_canonical（BOD/COD 特殊处理）
    if force_split:
        env5_target = force_split["env5_target"]
        match_raw_keys = [k.lower() for k in force_split.get("match_raw_keys", [])]
        rows = main_df[main_df["target_field_name"].astype(str) == env5_target]
        for _, r in rows.iterrows():
            # 遍历该 canonical 的所有 member_raw_keys；
            # 只挑字面匹配 match_raw_keys 的那些
            for rk in _split_pipe(r.get("member_raw_keys")):
                if rk.lower() in match_raw_keys:
                    na = _normalize_alias(rk, target)
                    if na:
                        aliases.add(na)
            # canonical_id 也记下来（但会被两个拆分 target 共享）
            if pd.notna(r.get("canonical_id")):
                canonical_ids.add(str(r["canonical_id"]))

    # 也把 raw_key_original 加入 alias（来自 env3 annotations）
    if env3_df is not None and "raw_key_original" in env3_df.columns:
        # 先找到本 target 所有 aliases 中字面在 env3 raw_key 列出现的
        # 从这些行取 raw_key_original 补充 aliases
        matched = env3_df[env3_df["raw_key"].astype(str).isin(aliases)]
        for _, r in matched.iterrows():
            rko = str(r.get("raw_key_original", ""))
            if rko:
                na = _normalize_alias(rko, target)
                if na:
                    aliases.add(na)

    meta = {
        "total_pmid": total_pmid,
        "n_env5_source_rows": n_env5_rows,
        "n_canonical_ids": len(canonical_ids),
        "canonical_ids": sorted(canonical_ids),
        "env_distribution": env_vec,
    }
    return sorted(aliases), meta


# ── Main list schema ─────────────────────────────────────────────────
def _render_main_list(
    schema: dict, main_df: pd.DataFrame,
    sig_dfs: dict[str, pd.DataFrame], env3_df: pd.DataFrame,
) -> list[dict]:
    rendered = []
    for entry in schema.get("main_list", []):
        target = entry["target"]
        subtype = entry.get("subtype", "")
        aliases, meta = _collect_aliases(target, main_df, sig_dfs, env3_df, entry)
        rendered.append({
            "target": target,
            "subtype": subtype,
            "aliases": aliases,
            "total_pmid": meta["total_pmid"],
            "n_canonical_ids": meta["n_canonical_ids"],
            "canonical_ids": meta["canonical_ids"],
            **meta["env_distribution"],
        })
    return rendered


# ── Signatures ───────────────────────────────────────────────────────
def _render_signatures(
    schema: dict, sig_dfs: dict[str, pd.DataFrame], env3_df: pd.DataFrame,
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {e: [] for e in config.HYDRO_ENVS}
    for env, entries in schema.get("signatures", {}).items():
        if env not in sig_dfs:
            logger.warning("Signature %s not in pipeline output, skipped", env)
            continue
        sdf = sig_dfs[env]
        for entry in entries:
            target = entry["target"]
            subtype = entry.get("from_subtype", "")
            merge_from = entry.get("merge_from", []) or [target]
            aliases: set[str] = set()
            canonical_ids: set[str] = set()
            total_pmid = 0
            for t in merge_from:
                rows = sdf[sdf["target_field_name"].astype(str) == t]
                for _, r in rows.iterrows():
                    total_pmid += int(r["total_pmid"])
                    if pd.notna(r.get("canonical_id")):
                        canonical_ids.add(str(r["canonical_id"]))
                    for rk in _split_pipe(r.get("member_raw_keys")):
                        na = _normalize_alias(rk, target)
                        if na:
                            aliases.add(na)
                    # 把合并的非主 target 名也加进去
                    if t != target:
                        na = _normalize_alias(t, target)
                        if na:
                            aliases.add(na)

            # raw_key_original 补充
            if env3_df is not None and "raw_key_original" in env3_df.columns:
                matched = env3_df[env3_df["raw_key"].astype(str).isin(aliases)]
                for _, r in matched.iterrows():
                    rko = str(r.get("raw_key_original", ""))
                    if rko:
                        na = _normalize_alias(rko, target)
                        if na:
                            aliases.add(na)

            out[env].append({
                "target": target,
                "subtype": subtype,
                "aliases": sorted(aliases),
                "total_pmid": total_pmid,
                "canonical_ids": sorted(canonical_ids),
            })
    return out


# ── Step5 JSON format ────────────────────────────────────────────────
def _render_step5_json(
    main_items: list[dict],
    sig_items: dict[str, list[dict]],
    schema: dict,
) -> dict:
    """与 step5.upstream_loader 的 env_target_fields schema 对齐：
       per_environment[env] = {"fields": [{field, tier, aliases}, ...]}
       tier1 = main_list (对所有 env 可见)
       tier2 = env signature
    """
    tier1_fields = [
        {
            "field": m["target"],
            "tier": 1,
            "subtype": m["subtype"],
            "aliases": m["aliases"],
            "total_pmid": m["total_pmid"],
        }
        for m in main_items
    ]

    per_env = {}
    for env in config.HYDRO_ENVS:
        tier2 = [
            {
                "field": s["target"],
                "tier": 2,
                "subtype": s["subtype"],
                "aliases": s["aliases"],
                "total_pmid": s["total_pmid"],
            }
            for s in sig_items.get(env, [])
        ]
        per_env[env] = {"fields": tier1_fields + tier2}

    # global_fields 兼容旧 step4b 格式（step5 部分逻辑回退到此）
    global_fields = {
        "Universal": [{"field": m["target"]} for m in main_items],
        "Signature": [
            {"field": s["target"], "env": env}
            for env, items in sig_items.items()
            for s in items
        ],
    }

    return {
        "metadata": {
            "generated_at": "phase6",
            "source": "phase6_schema.yaml + env5 pipeline products",
            "version": schema.get("version", 1),
            "n_tier1": len(tier1_fields),
            "n_tier2_per_env": {e: len(v) for e, v in sig_items.items()},
        },
        "global_fields": global_fields,
        "per_environment": per_env,
    }


# ── Main ─────────────────────────────────────────────────────────────
def run() -> None:
    config.ensure_output_dir()
    schema = _load_yaml()
    logger.info("Loaded schema: main=%d, sig=%s, excluded=%d",
                len(schema.get("main_list", [])),
                {e: len(v) for e, v in schema.get("signatures", {}).items()},
                len(schema.get("excluded", [])))

    main_df = _load_env5_main()
    sig_dfs = _load_env5_signatures()
    env3_df = _load_env3()
    logger.info("env5_main: %d rows; sigs: %s; env3: %d rows",
                len(main_df),
                {e: len(s) for e, s in sig_dfs.items()},
                len(env3_df))

    # Render
    main_items = _render_main_list(schema, main_df, sig_dfs, env3_df)
    sig_items = _render_signatures(schema, sig_dfs, env3_df)

    # Write audit CSVs
    main_rows = []
    for m in main_items:
        main_rows.append({
            "target": m["target"],
            "subtype": m["subtype"],
            "n_aliases": len(m["aliases"]),
            "aliases": "|".join(m["aliases"]),
            "total_pmid": m["total_pmid"],
            "n_canonical_ids": m["n_canonical_ids"],
            **{f"env_{e}": m.get(f"env_{e}", 0) for e in config.HYDRO_ENVS},
        })
    pd.DataFrame(main_rows).sort_values(
        "total_pmid", ascending=False
    ).to_csv(PHASE6_MAIN_SCHEMA, index=False)
    logger.info("Wrote main schema → %s (%d targets)",
                PHASE6_MAIN_SCHEMA, len(main_rows))

    sig_rows = []
    for env, items in sig_items.items():
        for s in items:
            sig_rows.append({
                "env": env,
                "target": s["target"],
                "subtype": s["subtype"],
                "n_aliases": len(s["aliases"]),
                "aliases": "|".join(s["aliases"]),
                "total_pmid": s["total_pmid"],
            })
    pd.DataFrame(sig_rows).sort_values(
        ["env", "total_pmid"], ascending=[True, False]
    ).to_csv(PHASE6_SIGNATURE_SCHEMA, index=False)
    logger.info("Wrote signature schema → %s (%d rows)",
                PHASE6_SIGNATURE_SCHEMA, len(sig_rows))

    # Excluded trace
    excl = schema.get("excluded", [])
    if excl:
        pd.DataFrame(excl).to_csv(PHASE6_EXCLUDED_TRACE, index=False)
        logger.info("Wrote excluded trace → %s (%d fields)",
                    PHASE6_EXCLUDED_TRACE, len(excl))

    # Step5 JSON
    step5_json = _render_step5_json(main_items, sig_items, schema)
    with open(PHASE6_EXTRACTION_TARGETS, "w", encoding="utf-8") as f:
        json.dump(step5_json, f, ensure_ascii=False, indent=2)
    logger.info("Wrote extraction targets → %s", PHASE6_EXTRACTION_TARGETS)

    # Summary
    logger.info("=" * 60)
    logger.info("PHASE 6 SUMMARY")
    logger.info("=" * 60)
    logger.info("Main list targets: %d", len(main_items))
    for env in config.HYDRO_ENVS:
        logger.info("  Signature %s: %d", env, len(sig_items.get(env, [])))
    logger.info("Excluded: %d (R1-R4)", len(excl))

    # Top 30 main by pmid
    sorted_main = sorted(main_items, key=lambda m: -m["total_pmid"])
    logger.info("")
    logger.info("Top 20 main targets by PMID:")
    for m in sorted_main[:20]:
        logger.info("  %6d  %-35s (%s, %d aliases)",
                    m["total_pmid"], m["target"], m["subtype"], len(m["aliases"]))

    # Alias overlap warning: target 自己重名
    names = [m["target"] for m in main_items]
    dups = {n for n in names if names.count(n) > 1}
    if dups:
        logger.warning("Duplicate target names in main_list: %s", dups)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()
