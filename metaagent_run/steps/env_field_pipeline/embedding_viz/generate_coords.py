"""Phase 3 语义空间可视化（LLM 规范化描述 → embedding → UMAP → plots）。

阶段（通过 STAGE 环境变量控制，默认 all）：
    describe: 取 evidence + LLM 扩写规范化描述（4,132 次调用）
    embed:    用本地 BGE-small 模型嵌入
    viz:      UMAP 降维 + 画图 + KNN 错位识别

输入：
    env3_final_annotations.csv       4,132 条带 family/subtype/qk/bag 标注
    step2 段落记录                    取 evidence 原文

输出（env_field_pipeline_output/viz/）：
    env3_field_descriptions.jsonl    描述（可复用的 checkpoint）
    env3_field_embeddings.npy        float32 (N, 384)
    env3_viz_coords.csv              UMAP 坐标 + 元数据 + 描述
    env3_viz_family.png              family 总图（4 色）
    env3_viz_{A|B|C}_*.png           按 family 拆分的子类图
    env3_viz_family_misplaced.csv    family 级错位字段
    env3_viz_subtype_misplaced.csv   subtype 级错位字段
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from metaagent_run.core import (
    AsyncLocalModelClient, backoff_with_jitter,
    extract_json_from_response_with_repair,
)
from metaagent_run.steps.env_field_pipeline import config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phase3_viz")

# ── 路径（跟随 config.OUTPUT_DIR；可通过环境变量 ENV_VIZ_DIR 覆盖）────
VIZ_DIR = Path(os.environ.get(
    "ENV_VIZ_DIR",
    str(config.OUTPUT_DIR / "viz"),
))
DESC_JSONL = VIZ_DIR / "env3_field_descriptions.jsonl"
EMBED_NPY = VIZ_DIR / "env3_field_embeddings.npy"
COORDS_CSV = VIZ_DIR / "env3_viz_coords.csv"
PLOT_FAMILY = VIZ_DIR / "env3_viz_family.png"
PLOT_A = VIZ_DIR / "env3_viz_A_physicochemical.png"
PLOT_B = VIZ_DIR / "env3_viz_B_env_categorical.png"
PLOT_C = VIZ_DIR / "env3_viz_C_spatiotemporal.png"
MISPLACED_FAMILY_CSV = VIZ_DIR / "env3_viz_family_misplaced.csv"
MISPLACED_SUBTYPE_CSV = VIZ_DIR / "env3_viz_subtype_misplaced.csv"

# ── 模型路径（默认 $META/models/bge-small-en-v1.5；MODEL_PATH env var 可覆盖）──
# PROJECT_ROOT_DIR 指向 meta_agent0408/（config.py 里 STEP_DIR.parents[2]）
MODEL_PATH = Path(os.environ.get(
    "MODEL_PATH",
    str(config.PROJECT_ROOT_DIR / "models" / "bge-small-en-v1.5"),
))

# ── 常量 ───────────────────────────────────────────────────────────
CONCURRENCY = 24
MAX_RETRIES = 3
EVIDENCE_CAP = 2
EVIDENCE_CHAR_CAP = 800   # 比 phase3 稍宽，留给扩写

# Evidence section tier（优先级：Tier 1 > Tier 2 > Tier 3）
# phase3 用 _is_text_section 只排除 TABLE；这里更严格：METHODS/RESULTS 优先
TIER1_KEYWORDS = ("method", "result")       # methods / results / Main text - METHODS / ...
TIER2_KEYWORDS = ("introduction", "discussion", "main text", "abstract")
# Tier 3 = 其他（table / supplementary / figure caption / legend / ...）


def _section_tier(section_type: str) -> int:
    st = (section_type or "").lower()
    for kw in TIER1_KEYWORDS:
        if kw in st:
            return 1
    for kw in TIER2_KEYWORDS:
        if kw in st:
            return 2
    return 3


def _build_evidence_index(raw_keys: set[str]) -> dict[str, list[dict]]:
    """按 Tier 1 → 2 → 3 顺序收集 evidence，每 key 最多 EVIDENCE_CAP 条。"""
    logger.info("Building evidence index for %d keys…", len(raw_keys))
    with open(config.STEP2_INPUT, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 三级桶，同 pmid 去重
    buckets: dict[str, dict[int, list[dict]]] = {
        k: {1: [], 2: [], 3: []} for k in raw_keys
    }
    seen_pmids: dict[str, set[str]] = {k: set() for k in raw_keys}

    for r in records:
        keys = r.get("metadata_keys_found") or []
        if not keys:
            continue
        pmid = r.get("pmid", "")
        tier = _section_tier(r.get("section_type", ""))
        for k in keys:
            if k not in buckets:
                continue
            if pmid in seen_pmids[k]:
                continue
            if len(buckets[k][tier]) >= EVIDENCE_CAP:
                continue
            quote = (r.get("evidence_quote") or "")[:EVIDENCE_CHAR_CAP]
            buckets[k][tier].append({
                "pmid": pmid, "quote": quote,
                "section": r.get("section_type", ""), "tier": tier,
            })
            seen_pmids[k].add(pmid)

    out: dict[str, list[dict]] = {}
    tier_stat = {1: 0, 2: 0, 3: 0, "none": 0}
    for k in raw_keys:
        combined: list[dict] = []
        for tier in (1, 2, 3):
            for e in buckets[k][tier]:
                if len(combined) >= EVIDENCE_CAP:
                    break
                combined.append(e)
        if not combined:
            tier_stat["none"] += 1
        else:
            tier_stat[combined[0]["tier"]] += 1
        out[k] = combined
    logger.info("Evidence primary tier: %s", tier_stat)
    return out


def _fmt_evidence(evs: list[dict]) -> str:
    if not evs:
        return "(no evidence)"
    lines = []
    for e in evs:
        q = (e["quote"] or "").strip().replace("\n", " ")
        if len(q) > EVIDENCE_CHAR_CAP:
            q = q[:EVIDENCE_CHAR_CAP] + "…"
        lines.append(f"[{e['pmid']} | {e['section']}] {q}")
    return "\n".join(lines)


# ── LLM prompt ─────────────────────────────────────────────────────
DESCRIBE_PROMPT = """\
You are helping build embeddings for environmental metadata fields.

Given a field from a hydrosphere (ocean / coastal / lake / wetland) research
paper, produce ONE concise English phrase (10-25 words) that normalizes its
concept, so that fields with the same meaning across papers yield similar
phrases.

Field information:
  raw_key:         {raw_key}
  family/subtype:  {family} / {subtype}
  quantity_kind:   {quantity_kind}
  modifier_bag:    {modifier_bag}

Evidence from source paper(s) [prioritizes methods/results sections]:
{evidence}

Guidelines:
  1. Ground your phrase in the evidence when it provides context; otherwise
     infer from raw_key and the structured slots.
  2. Expand abbreviations: doc→dissolved organic carbon; tp→total phosphorus;
     tds→total dissolved solids; mld→mixed-layer depth; sst→sea-surface
     temperature; etc.
  3. Specify the measured concept and, when relevant, the matrix (water /
     sediment / soil / atmosphere / observation event).
  4. Keep the phrase 10-25 words. No lists, no quotes, no extra text.

Return JSON:
{{
  "description": "the normalized phrase",
  "confidence": "high | medium | low"
}}
</json>"""


async def _describe_one(
    client: AsyncLocalModelClient, row: dict, evidence: list[dict],
) -> dict:
    prompt = DESCRIBE_PROMPT.format(
        raw_key=row["raw_key"],
        family=row["family"],
        subtype=row["subtype"],
        quantity_kind=row["quantity_kind"],
        modifier_bag=row.get("modifier_bag") or "",
        evidence=_fmt_evidence(evidence),
    )
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.chat(messages)
            if resp is None:
                await asyncio.sleep(backoff_with_jitter(attempt))
                continue
            parsed = extract_json_from_response_with_repair(
                resp, stop_sentinel=config.STOP_SENTINEL
            )
            if isinstance(parsed, list):
                parsed = next((x for x in parsed if isinstance(x, dict)), None)
            if not isinstance(parsed, dict):
                await asyncio.sleep(backoff_with_jitter(attempt))
                continue
            desc = str(parsed.get("description", "")).strip()
            if not desc:
                continue
            return {
                "raw_key": row["raw_key"],
                "description": desc,
                "conf_desc": str(parsed.get("confidence", "medium")).lower(),
                "evidence_tier": evidence[0]["tier"] if evidence else 0,
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("attempt %d failed for %s: %s",
                           attempt, row["raw_key"], e)
            await asyncio.sleep(backoff_with_jitter(attempt))
    # 兜底：用 raw_key + qk 构造默认描述
    return {
        "raw_key": row["raw_key"],
        "description": f"{row['quantity_kind']} ({row['raw_key']})",
        "conf_desc": "low",
        "evidence_tier": 0,
    }


def _load_desc_checkpoint() -> dict[str, dict]:
    out = {}
    if DESC_JSONL.exists():
        with open(DESC_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    out[d["raw_key"]] = d
                except Exception:
                    continue
        logger.info("Resumed %d descriptions from checkpoint", len(out))
    return out


async def stage_describe():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.PHASE3_OUTPUT)
    logger.info("Loaded %d annotations", len(df))

    raw_keys = set(df["raw_key"].astype(str).tolist())
    ev_idx = _build_evidence_index(raw_keys)

    done = _load_desc_checkpoint()
    todo_rows = [r for r in df.to_dict("records") if str(r["raw_key"]) not in done]
    logger.info("To describe: %d  (cached: %d)", len(todo_rows), len(done))

    api_key = os.environ.get("ALL_API_KEY")
    if not api_key:
        raise RuntimeError("ALL_API_KEY required")

    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    results: list[dict] = list(done.values())

    async with AsyncLocalModelClient(
        base_url=config.BASE_URL, model=config.MODEL,
        temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS,
        api_key=api_key, stop_sentinel=config.STOP_SENTINEL,
        api_style=config.API_STYLE, auth_mode=config.AUTH_MODE,
    ) as client:
        async def worker(row):
            async with sem:
                ev = ev_idx.get(str(row["raw_key"]), [])
                res = await _describe_one(client, row, ev)
                async with lock:
                    with open(DESC_JSONL, "a", encoding="utf-8") as f:
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
                return res

        start = time.time()
        tasks = [asyncio.create_task(worker(r)) for r in todo_rows]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            results.append(await coro)
            if (i + 1) % 500 == 0:
                rate = (i + 1) / max(1e-6, time.time() - start)
                logger.info("described %d/%d (%.1f/s)", i + 1, len(todo_rows), rate)

    logger.info("Describe done: %d results", len(results))


def stage_embed():
    logger.info("Loading sentence-transformers model …")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(MODEL_PATH), device="cpu")

    descs = _load_desc_checkpoint()
    df = pd.read_csv(config.PHASE3_OUTPUT)
    keys = df["raw_key"].astype(str).tolist()
    missing = [k for k in keys if k not in descs]
    if missing:
        raise RuntimeError(f"{len(missing)} keys without description; run describe first")

    texts = [descs[k]["description"] for k in keys]
    logger.info("Encoding %d texts …", len(texts))
    embs = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    np.save(EMBED_NPY, embs)
    logger.info("Embeddings saved → %s (shape=%s)", EMBED_NPY, embs.shape)

    # 附带写 index 文件（raw_key + description）
    idx_df = pd.DataFrame({
        "raw_key": keys,
        "description": texts,
        "conf_desc": [descs[k]["conf_desc"] for k in keys],
        "evidence_tier": [descs[k]["evidence_tier"] for k in keys],
    })
    idx_df.to_csv(VIZ_DIR / "env3_desc_index.csv", index=False)


def stage_viz():
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import umap
    from sklearn.neighbors import NearestNeighbors

    logger.info("Loading data …")
    df = pd.read_csv(config.PHASE3_OUTPUT)
    embs = np.load(EMBED_NPY)
    idx_df = pd.read_csv(VIZ_DIR / "env3_desc_index.csv")
    assert len(df) == len(embs) == len(idx_df)

    logger.info("UMAP 2D …")
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, random_state=42,
        metric="cosine", n_components=2, verbose=True,
    )
    coords = reducer.fit_transform(embs)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    df["description"] = idx_df["description"].values

    coords_df = df[[
        "raw_key", "family", "subtype", "quantity_kind",
        "modifier_bag", "description", "x", "y",
    ]].copy()
    coords_df.to_csv(COORDS_CSV, index=False)
    logger.info("Coords saved → %s", COORDS_CSV)

    # ── Plot 1: family 总图（4 色）──
    FAMILY_COLORS = {
        "A_physicochemical": "#d62728",
        "B_env_categorical": "#2ca02c",
        "C_spatiotemporal": "#1f77b4",
        "D_other": "#7f7f7f",
    }
    fig, ax = plt.subplots(figsize=(12, 10))
    for fam, sub_df in df.groupby("family"):
        ax.scatter(sub_df["x"], sub_df["y"],
                   s=8, alpha=0.55, c=FAMILY_COLORS.get(fam, "#000000"),
                   label=f"{fam} (n={len(sub_df)})")
    ax.set_title("UMAP of 4,132 env metadata fields — colored by family",
                 fontsize=14)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", markerscale=2, fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_FAMILY, dpi=140)
    plt.close(fig)
    logger.info("Saved %s", PLOT_FAMILY)

    # ── Plot 2-4: 按 family 拆分子类 ──
    def _plot_family(fam_key: str, out_path: Path):
        sub = df[df["family"] == fam_key]
        if len(sub) == 0:
            return
        subtypes = sorted(sub["subtype"].unique())
        cmap = cm.get_cmap("tab20", len(subtypes))
        colors = {st: cmap(i) for i, st in enumerate(subtypes)}

        fig, ax = plt.subplots(figsize=(12, 10))
        # 背景：全部点灰色
        ax.scatter(df["x"], df["y"], s=3, alpha=0.12, c="#cccccc")
        # 前景：该 family 按 subtype 染色
        for st, ss in sub.groupby("subtype"):
            ax.scatter(ss["x"], ss["y"], s=12, alpha=0.75,
                       c=[colors[st]], label=f"{st} (n={len(ss)})")
        ax.set_title(f"UMAP — {fam_key} ({len(sub)} fields) by subtype",
                     fontsize=14)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        ax.legend(loc="best", markerscale=2, fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        logger.info("Saved %s", out_path)

    _plot_family("A_physicochemical", PLOT_A)
    _plot_family("B_env_categorical", PLOT_B)
    _plot_family("C_spatiotemporal", PLOT_C)

    # ── KNN 错位识别 ──
    logger.info("KNN misplacement detection …")
    K = 5  # 除自身
    nbrs = NearestNeighbors(n_neighbors=K + 1, metric="cosine").fit(embs)
    _, idx = nbrs.kneighbors(embs)

    families = df["family"].values
    subtypes = df["subtype"].values
    raw_keys = df["raw_key"].values
    descriptions = df["description"].values

    family_miss: list[dict] = []
    subtype_miss: list[dict] = []
    for i in range(len(df)):
        nbr = idx[i][1:]
        same_fam = int(sum(1 for j in nbr if families[j] == families[i]))
        same_sub = int(sum(1 for j in nbr if subtypes[j] == subtypes[i]))
        dominant_fam_nbr = max(set(families[nbr].tolist()),
                               key=lambda f: sum(1 for j in nbr if families[j] == f))
        dominant_sub_nbr = max(set(subtypes[nbr].tolist()),
                               key=lambda s: sum(1 for j in nbr if subtypes[j] == s))
        if same_fam < 3:  # < 3/5
            family_miss.append({
                "raw_key": raw_keys[i],
                "current_family": families[i],
                "current_subtype": subtypes[i],
                "nbr_dominant_family": dominant_fam_nbr,
                "same_family_among_k5": same_fam,
                "description": descriptions[i],
            })
        if same_sub < 2 and families[i] != "D_other":  # < 2/5
            subtype_miss.append({
                "raw_key": raw_keys[i],
                "current_family": families[i],
                "current_subtype": subtypes[i],
                "nbr_dominant_subtype": dominant_sub_nbr,
                "same_subtype_among_k5": same_sub,
                "description": descriptions[i],
            })

    pd.DataFrame(family_miss).to_csv(MISPLACED_FAMILY_CSV, index=False)
    pd.DataFrame(subtype_miss).to_csv(MISPLACED_SUBTYPE_CSV, index=False)
    logger.info("Family-level misplaced: %d → %s",
                len(family_miss), MISPLACED_FAMILY_CSV)
    logger.info("Subtype-level misplaced: %d → %s",
                len(subtype_miss), MISPLACED_SUBTYPE_CSV)

    logger.info("=" * 60)
    logger.info("PHASE 3 VIZ SUMMARY")
    logger.info("=" * 60)
    logger.info("Total fields: %d", len(df))
    logger.info("Family-level misplacement (same_family_nbr < 3/5): %d (%.1f%%)",
                len(family_miss), 100 * len(family_miss) / len(df))
    logger.info("Subtype-level misplacement (same_subtype_nbr < 2/5): %d (%.1f%%)",
                len(subtype_miss), 100 * len(subtype_miss) / len(df))


def main():
    stage = os.environ.get("STAGE", "all").lower()
    if stage in ("describe", "all"):
        asyncio.run(stage_describe())
    if stage in ("embed", "all"):
        stage_embed()
    if stage in ("viz", "all"):
        stage_viz()


if __name__ == "__main__":
    main()
