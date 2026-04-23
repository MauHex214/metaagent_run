"""Phase 3 可视化 v2：分面板图 + 密度图 + kNN 硬验证。

复用 v1 产出：
    env3_field_embeddings.npy     (4132, 384)  bge-small 向量
    env3_viz_coords.csv            UMAP 2D 坐标 + 描述
    env3_final_annotations.csv    family/subtype/total_pmid

输出目录: env_field_pipeline_output/viz_v2/
"""
from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("viz_v2")

import os
# ── Paths（跟随 config.OUTPUT_DIR，可被环境变量覆盖） ──
from metaagent_run.steps.env_field_pipeline import config
BASE = Path(os.environ.get("ENV_OUTPUT_DIR", str(config.OUTPUT_DIR)))
VIZ_V1 = Path(os.environ.get("ENV_VIZ_DIR", str(BASE / "viz")))
OUT = Path(os.environ.get("ENV_VIZ_V2_DIR", str(BASE / "viz_v2")))
OUT.mkdir(parents=True, exist_ok=True)

# ── Config ──
K = 5   # kNN
FAMILY_ORDER = ["A_physicochemical", "B_env_categorical", "C_spatiotemporal", "D_other"]
FAM_COLORS = {
    "A_physicochemical": "#d62728",
    "B_env_categorical": "#2ca02c",
    "C_spatiotemporal":  "#1f77b4",
    "D_other":           "#7f7f7f",
}
FAM_LABEL = {  # shorter for titles
    "A_physicochemical": "A: physicochemical",
    "B_env_categorical": "B: env_categorical",
    "C_spatiotemporal":  "C: spatiotemporal",
    "D_other":           "D: other",
}

# ── Load data ──
logger.info("Loading data …")
coords = pd.read_csv(VIZ_V1 / "env3_viz_coords.csv")
embs = np.load(VIZ_V1 / "env3_field_embeddings.npy")
annot = pd.read_csv(BASE / "env3_final_annotations.csv")[
    ["raw_key", "total_pmid"]
]
# Post raw_key expansion, multiple raw_key_original rows can share the same
# normalized raw_key (e.g. no3 / no3- / no3− all → nitrate), so annot has
# duplicate raw_keys. Aggregate pmid before merging to avoid a Cartesian
# blow-up of coords from 4132 → 7406 rows.
annot = annot.groupby("raw_key", as_index=False)["total_pmid"].sum()
df = coords.merge(annot, on="raw_key", how="left")
assert len(df) == len(embs) == len(coords), (
    f"Row count mismatch after merge: df={len(df)} embs={len(embs)} coords={len(coords)}"
)
logger.info("df: %d rows × %d cols; embs shape: %s", len(df), len(df.columns), embs.shape)

XLIM = (df["x"].min() - 1, df["x"].max() + 1)
YLIM = (df["y"].min() - 1, df["y"].max() + 1)

df["_size"] = 10 + np.log(df["total_pmid"].clip(lower=1)) * 5


# ═════════════════════════════════════════════════════════════════
# 选项 A — 重画图
# ═════════════════════════════════════════════════════════════════

def plot_family_panels():
    fig, axes = plt.subplots(2, 2, figsize=(15, 13), sharex=True, sharey=True)
    for fam, ax in zip(FAMILY_ORDER, axes.flat):
        ax.scatter(df["x"], df["y"], s=3, alpha=0.05, c="#bbbbbb",
                   linewidths=0, rasterized=True)
        sub = df[df["family"] == fam]
        if len(sub):
            ax.scatter(sub["x"], sub["y"], s=sub["_size"], alpha=0.35,
                       c=FAM_COLORS[fam], linewidths=0, rasterized=True)
        pmid_sum = int(sub["total_pmid"].sum())
        ax.set_title(
            f"Family {FAM_LABEL[fam]}  (n={len(sub):,}, PMID={pmid_sum:,})",
            fontsize=13, color=FAM_COLORS[fam], fontweight="bold",
        )
        ax.set_xlim(XLIM); ax.set_ylim(YLIM)
        ax.grid(True, alpha=0.25)
    fig.suptitle("UMAP embedding of 4,132 env-metadata fields — by family",
                 fontsize=15, y=1.00)
    fig.supxlabel("UMAP-1"); fig.supylabel("UMAP-2")
    fig.tight_layout()
    out = OUT / "env3_viz_v2_family_panels.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_family_hexbin():
    # family → int
    fam_code = df["family"].map({
        "A_physicochemical": 1,
        "B_env_categorical": 2,
        "C_spatiotemporal":  3,
        "D_other":           4,
    }).values.astype(int)

    def dominant_or_zero(vals):
        # vals can be np array or list of int codes
        c = Counter(int(v) for v in vals)
        m, n = c.most_common(1)[0]
        return m if n > len(vals) / 2 else 0  # 0 = mixed

    # 5-color map: 0=mixed, 1=A, 2=B, 3=C, 4=D
    cmap = ListedColormap(["#ededed",
                            FAM_COLORS["A_physicochemical"],
                            FAM_COLORS["B_env_categorical"],
                            FAM_COLORS["C_spatiotemporal"],
                            FAM_COLORS["D_other"]])

    fig, ax = plt.subplots(figsize=(12, 10))
    hb = ax.hexbin(
        df["x"], df["y"], C=fam_code,
        reduce_C_function=dominant_or_zero,
        gridsize=50, cmap=cmap, vmin=-0.5, vmax=4.5,
        mincnt=1, linewidths=0.2, edgecolors="white",
    )
    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    ax.set_title("UMAP density hexbin — dominant family per cell "
                 "(>50% threshold, else 'mixed')", fontsize=13)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.grid(True, alpha=0.25)

    # 手动图例
    from matplotlib.patches import Patch
    handles = [
        Patch(color="#ededed", label="mixed (no family > 50%)"),
        Patch(color=FAM_COLORS["A_physicochemical"], label="A_physicochemical"),
        Patch(color=FAM_COLORS["B_env_categorical"], label="B_env_categorical"),
        Patch(color=FAM_COLORS["C_spatiotemporal"],  label="C_spatiotemporal"),
        Patch(color=FAM_COLORS["D_other"],           label="D_other"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=10)
    fig.tight_layout()
    out = OUT / "env3_viz_v2_family_hexbin.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)

    # 顺便返回主导族统计给 report 用
    hex_domain = [dominant_or_zero(hb.get_array()[i:i+1])
                   for i in range(len(hb.get_array()))]
    return None


def plot_subtype_panels(family: str, grid: tuple[int, int],
                        fname: str, figsize=(18, 12)):
    sub_all = df[df["family"] == family].copy()
    subtypes = (
        sub_all.groupby("subtype")["total_pmid"].sum()
        .sort_values(ascending=False).index.tolist()
    )
    rows, cols = grid
    fig, axes = plt.subplots(rows, cols, figsize=figsize,
                             sharex=True, sharey=True)
    axes_flat = axes.flat
    color = FAM_COLORS[family]

    for i, st in enumerate(subtypes):
        ax = axes_flat.__next__()
        # 同 family 其他 subtype 作底色
        ax.scatter(sub_all["x"], sub_all["y"], s=3, alpha=0.08,
                   c="#cccccc", linewidths=0, rasterized=True)
        cur = sub_all[sub_all["subtype"] == st]
        ax.scatter(cur["x"], cur["y"], s=cur["_size"], alpha=0.5,
                   c=color, linewidths=0, rasterized=True)
        pmid_sum = int(cur["total_pmid"].sum())
        ax.set_title(f"{st}\n(n={len(cur):,}, PMID={pmid_sum:,})", fontsize=11)
        ax.set_xlim(XLIM); ax.set_ylim(YLIM)
        ax.grid(True, alpha=0.25)

    # 剩余空格子隐藏
    for ax in axes_flat:
        ax.axis("off")

    fig.suptitle(
        f"{FAM_LABEL[family]} subtypes ({len(subtypes)} subtypes, "
        f"{len(sub_all):,} fields) — highlighted vs grey background",
        fontsize=14, color=color, fontweight="bold", y=1.00,
    )
    fig.supxlabel("UMAP-1"); fig.supylabel("UMAP-2")
    fig.tight_layout()
    out = OUT / fname
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


# ═════════════════════════════════════════════════════════════════
# 选项 B — kNN 硬验证（在 384 维 embedding 空间）
# ═════════════════════════════════════════════════════════════════

def run_knn_analysis():
    logger.info("kNN (k=%d, cosine) on %d × %d embeddings …", K, *embs.shape)
    nbrs = NearestNeighbors(n_neighbors=K + 1, metric="cosine").fit(embs)
    dist, idx = nbrs.kneighbors(embs)
    nbr_idx = idx[:, 1:K + 1]  # 去掉自身

    families = df["family"].values
    subtypes = df["subtype"].values
    raw_keys = df["raw_key"].values
    pmids = df["total_pmid"].values
    descs = df["description"].values

    # 预测族/子类 = 邻居众数
    def majority(vals):
        c = Counter(vals)
        return c.most_common(1)[0][0]

    pred_fam = np.array([majority(families[nbr_idx[i]]) for i in range(len(df))])
    pred_sub = np.array([majority(subtypes[nbr_idx[i]]) for i in range(len(df))])

    # 同族/同子类邻居计数
    same_fam_cnt = np.array([
        int(sum(1 for j in nbr_idx[i] if families[j] == families[i]))
        for i in range(len(df))
    ])
    same_sub_cnt = np.array([
        int(sum(1 for j in nbr_idx[i] if subtypes[j] == subtypes[i]))
        for i in range(len(df))
    ])

    # ── 族混淆矩阵 ──
    cm_fam = confusion_matrix(families, pred_fam, labels=FAMILY_ORDER)
    cm_df = pd.DataFrame(cm_fam, index=FAMILY_ORDER, columns=FAMILY_ORDER)
    cm_df["total"] = cm_df.sum(axis=1)
    cm_df["purity_pct"] = np.round(
        100 * np.diag(cm_fam) / cm_df["total"].clip(lower=1), 2
    )
    cm_df.to_csv(OUT / "env3_knn_family_confusion.csv")
    logger.info("Family confusion → env3_knn_family_confusion.csv")
    logger.info("Family purity:\n%s",
                cm_df[["total", "purity_pct"]].to_string())

    # ── 子类混淆矩阵 ──
    subtype_order = sorted(df["subtype"].unique())
    cm_sub = confusion_matrix(subtypes, pred_sub, labels=subtype_order)
    cm_sub_df = pd.DataFrame(cm_sub, index=subtype_order, columns=subtype_order)
    cm_sub_df["total"] = cm_sub_df.sum(axis=1)
    cm_sub_df["purity_pct"] = np.round(
        100 * np.diag(cm_sub) / cm_sub_df["total"].clip(lower=1), 2
    )
    cm_sub_df.to_csv(OUT / "env3_knn_subtype_confusion.csv")
    logger.info("Subtype confusion → env3_knn_subtype_confusion.csv")

    # ── 错位字段清单 ──
    fam_mis = []
    for i in range(len(df)):
        if same_fam_cnt[i] < 3:  # < 3/5
            dist_nbr = Counter(families[nbr_idx[i]])
            dist_str = " ".join(f"{k}:{v}" for k, v in dist_nbr.most_common())
            fam_mis.append({
                "raw_key": raw_keys[i],
                "current_family": families[i],
                "neighbor_majority_family": pred_fam[i],
                "same_family_count": int(same_fam_cnt[i]),
                "neighbor_distribution": dist_str,
                "total_pmid": int(pmids[i]),
                "description": descs[i],
            })
    fam_mis_df = pd.DataFrame(fam_mis).sort_values("total_pmid", ascending=False)
    fam_mis_df.to_csv(OUT / "env3_knn_family_misplaced.csv", index=False)
    logger.info("Family misplaced: %d → env3_knn_family_misplaced.csv", len(fam_mis_df))

    sub_mis = []
    for i in range(len(df)):
        if same_sub_cnt[i] < 2 and families[i] != "D_other":  # < 2/5, skip D
            dist_nbr = Counter(subtypes[nbr_idx[i]])
            dist_str = " ".join(f"{k}:{v}" for k, v in dist_nbr.most_common())
            sub_mis.append({
                "raw_key": raw_keys[i],
                "current_family": families[i],
                "current_subtype": subtypes[i],
                "neighbor_majority_subtype": pred_sub[i],
                "same_subtype_count": int(same_sub_cnt[i]),
                "neighbor_distribution": dist_str,
                "total_pmid": int(pmids[i]),
                "description": descs[i],
            })
    sub_mis_df = pd.DataFrame(sub_mis).sort_values("total_pmid", ascending=False)
    sub_mis_df.to_csv(OUT / "env3_knn_subtype_misplaced.csv", index=False)
    logger.info("Subtype misplaced: %d → env3_knn_subtype_misplaced.csv", len(sub_mis_df))

    # ── Subtype centroid distance ──
    centroids = {}
    for st in subtype_order:
        mask = (subtypes == st)
        c = embs[mask].mean(axis=0)
        norm = np.linalg.norm(c)
        centroids[st] = c / norm if norm > 0 else c
    C = np.stack([centroids[st] for st in subtype_order])
    sim = C @ C.T
    cen_dist = 1 - sim
    np.fill_diagonal(cen_dist, 0.0)  # 清零对角线显示噪声
    cen_df = pd.DataFrame(cen_dist, index=subtype_order, columns=subtype_order)
    cen_df.to_csv(OUT / "env3_subtype_centroid_distances.csv")
    logger.info("Subtype centroid distances → env3_subtype_centroid_distances.csv")

    return cm_df, cm_sub_df, fam_mis_df, sub_mis_df, cen_df, subtype_order


# ═════════════════════════════════════════════════════════════════
# 描述质量抽检（供报告）
# ═════════════════════════════════════════════════════════════════

def sample_descriptions(n=20):
    rng = random.Random(42)
    idx = rng.sample(range(len(df)), n)
    sample = df.iloc[idx][["raw_key", "family", "subtype",
                            "quantity_kind", "description"]]
    return sample


# ═════════════════════════════════════════════════════════════════
# 生成 Markdown 报告
# ═════════════════════════════════════════════════════════════════

def write_report(cm_fam, cm_sub, fam_mis, sub_mis, cen, subtype_order):
    lines: list[str] = []
    lines.append("# Phase 3 Embedding 可视化 v2 — 判定报告")
    lines.append("")
    lines.append("生成时间：auto")
    lines.append("")

    # 族纯度
    lines.append("## 1. Family 级 kNN 纯度")
    lines.append("")
    lines.append("k=5, cosine 距离，384 维 embedding 空间。")
    lines.append("")
    lines.append("| Family | n | purity% |")
    lines.append("|---|---|---|")
    for fam in FAMILY_ORDER:
        n = int(cm_fam.loc[fam, "total"])
        p = cm_fam.loc[fam, "purity_pct"]
        note = " (n=7 免评)" if fam == "D_other" else ""
        target = "≥80%"
        status = "✓" if (fam == "D_other") else ("✓" if p >= 80 else "⚠")
        lines.append(f"| `{fam}` | {n:,} | **{p:.1f}%**{note} {status} |")
    lines.append("")
    lines.append(f"目标：A/B/C 三族 ≥80%（D 族因样本过少 n=7 免评）。")
    lines.append("")

    # 族混淆矩阵（手写 MD）
    lines.append("### 族混淆矩阵（每行：true_family，列：predicted_family）")
    lines.append("")
    hdrs = list(cm_fam.columns)
    lines.append("| " + " | ".join(["true \\ pred"] + hdrs) + " |")
    lines.append("|" + "---|" * (len(hdrs) + 1))
    for idx_row, row in cm_fam.iterrows():
        cells = [str(int(row[c])) if c not in ("purity_pct",) else f"{row[c]:.1f}%"
                 for c in hdrs]
        lines.append(f"| `{idx_row}` | " + " | ".join(cells) + " |")
    lines.append("")

    # 子类纯度
    lines.append("## 2. Subtype 级 kNN 纯度")
    lines.append("")
    avg_purity = cm_sub["purity_pct"].mean()
    min_purity_row = cm_sub.sort_values("purity_pct").iloc[0]
    min_sub = min_purity_row.name
    min_p = min_purity_row["purity_pct"]
    min_n = int(min_purity_row["total"])

    lines.append(f"- **平均纯度**：{avg_purity:.1f}% （目标 ≥70%）"
                 + (" ✓" if avg_purity >= 70 else " ⚠"))
    lines.append(f"- **最低纯度子类**：`{min_sub}` purity={min_p:.1f}% (n={min_n}) "
                 + ("✓" if min_p >= 50 else "⚠（目标 ≥50%）"))
    lines.append("")

    # 子类纯度 TOP/BOTTOM 5
    sorted_sub = cm_sub.sort_values("purity_pct")
    lines.append("### Subtype 纯度 Bottom 8（可疑子类）")
    lines.append("")
    lines.append("| subtype | n | purity% |")
    lines.append("|---|---|---|")
    for st in sorted_sub.head(8).index:
        lines.append(f"| `{st}` | {int(cm_sub.loc[st,'total']):,} | "
                     f"{cm_sub.loc[st,'purity_pct']:.1f}% |")
    lines.append("")
    lines.append("### Subtype 纯度 Top 8（最清晰子类）")
    lines.append("")
    lines.append("| subtype | n | purity% |")
    lines.append("|---|---|---|")
    for st in sorted_sub.tail(8).iloc[::-1].index:
        lines.append(f"| `{st}` | {int(cm_sub.loc[st,'total']):,} | "
                     f"{cm_sub.loc[st,'purity_pct']:.1f}% |")
    lines.append("")

    # 最近的 subtype 对（可能合并候选）
    lines.append("## 3. Subtype centroid 最近对（可能合并候选）")
    lines.append("")
    n = len(subtype_order)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = subtype_order[i], subtype_order[j]
            # 同族对 vs 跨族对区分
            fam_a = [f for f in FAMILY_ORDER
                     if a in df[df["family"] == f]["subtype"].values][0] if (
                df[df["subtype"] == a]["family"].iloc[0] is not None) else "?"
            fam_b = df[df["subtype"] == b]["family"].iloc[0]
            fam_a = df[df["subtype"] == a]["family"].iloc[0]
            pairs.append({
                "subtype_a": a, "subtype_b": b,
                "family_a": fam_a, "family_b": fam_b,
                "distance": cen.iloc[i, j],
                "same_family": fam_a == fam_b,
            })
    pairs_df = pd.DataFrame(pairs).sort_values("distance")

    # Top 10 同族最近
    same_fam_top = pairs_df[pairs_df["same_family"]].head(10)
    lines.append("### 同 family 内最近 10 对（子类语义最接近，合并风险）")
    lines.append("")
    lines.append("| family | subtype_a | subtype_b | cosine_dist |")
    lines.append("|---|---|---|---|")
    for _, r in same_fam_top.iterrows():
        lines.append(f"| {r['family_a']} | `{r['subtype_a']}` | `{r['subtype_b']}` | "
                     f"**{r['distance']:.4f}** |")
    lines.append("")

    # 跨族最近
    cross_top = pairs_df[~pairs_df["same_family"]].head(5)
    lines.append("### 跨 family 最近 5 对（族级边界）")
    lines.append("")
    lines.append("| subtype_a (fam) | subtype_b (fam) | cosine_dist |")
    lines.append("|---|---|---|")
    for _, r in cross_top.iterrows():
        lines.append(f"| `{r['subtype_a']}` ({r['family_a']}) | "
                     f"`{r['subtype_b']}` ({r['family_b']}) | "
                     f"**{r['distance']:.4f}** |")
    lines.append("")

    # 距离统计：同族 vs 跨族
    same_median = pairs_df[pairs_df["same_family"]]["distance"].median()
    cross_median = pairs_df[~pairs_df["same_family"]]["distance"].median()
    lines.append(f"- **同族 subtype 对**（n={len(same_fam_top)+(pairs_df['same_family'].sum()-len(same_fam_top))}）中位数距离：**{same_median:.4f}**")
    lines.append(f"- **跨族 subtype 对**中位数距离：**{cross_median:.4f}**")
    lines.append(f"- 跨族中位数 {'>' if cross_median > same_median else '<'} 同族中位数 "
                 f"{'✓（族级区分 > 子类级）' if cross_median > same_median else '⚠（族内外距离颠倒）'}")
    lines.append("")

    # 错位高 PMID 字段
    lines.append("## 4. 错位高 PMID 字段 Top 10（family 级）")
    lines.append("")
    lines.append("| raw_key | pmid | current | nbr_majority | distribution | description |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in fam_mis.head(10).iterrows():
        desc = str(r["description"])[:80]
        lines.append(f"| `{r['raw_key']}` | {r['total_pmid']:,} | "
                     f"{r['current_family'][0]} | {r['neighbor_majority_family'][0]} | "
                     f"{r['neighbor_distribution']} | {desc} |")
    lines.append("")

    # 描述质量抽检
    lines.append("## 5. LLM 描述质量抽检（随机 20 条）")
    lines.append("")
    sample = sample_descriptions(20)
    lines.append("| raw_key | family/subtype | qk | description |")
    lines.append("|---|---|---|---|")
    for _, r in sample.iterrows():
        desc = str(r["description"])[:120]
        lines.append(f"| `{r['raw_key']}` | {r['family'][0]}/{r['subtype']} | "
                     f"{r['quantity_kind']} | {desc} |")
    lines.append("")

    # 结论
    lines.append("## 6. 结论")
    lines.append("")
    fam_pass = all(cm_fam.loc[f, "purity_pct"] >= 80 for f in FAMILY_ORDER if f != "D_other")
    sub_avg_pass = avg_purity >= 70
    sub_min_pass = min_p >= 50

    lines.append(f"- Family 纯度 ≥80% (A/B/C)：{'✅ 全部达标' if fam_pass else '⚠ 部分未达标'}")
    lines.append(f"- Subtype 平均纯度 ≥70%：{'✅' if sub_avg_pass else '⚠'} ({avg_purity:.1f}%)")
    lines.append(f"- Subtype 最低纯度 ≥50%：{'✅' if sub_min_pass else '⚠'} "
                 f"({min_sub}={min_p:.1f}%)")
    lines.append(f"- 族级区分度 > 子类级（跨族 > 同族中位数）："
                 f"{'✅' if cross_median > same_median else '⚠'}")
    lines.append("")
    if fam_pass and sub_avg_pass and sub_min_pass:
        lines.append("**判定：分类方案经 embedding 独立验证成立，可推进 step 4。**")
    else:
        lines.append("**判定：部分指标未达标，需进一步分析。** 见上文 Bottom 8 子类。")
    lines.append("")

    with open(OUT / "env3_viz_v2_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Report → env3_viz_v2_report.md")


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════

logger.info("============= OPTION A: 重画图 =============")
plot_family_panels()
plot_family_hexbin()
plot_subtype_panels("A_physicochemical", grid=(3, 4),
                    fname="env3_viz_v2_A_subtypes.png", figsize=(18, 14))
plot_subtype_panels("B_env_categorical", grid=(2, 2),
                    fname="env3_viz_v2_B_subtypes.png", figsize=(12, 10))
plot_subtype_panels("C_spatiotemporal", grid=(2, 4),
                    fname="env3_viz_v2_C_subtypes.png", figsize=(18, 10))

logger.info("============= OPTION B: kNN 分析 =============")
cm_fam, cm_sub, fam_mis, sub_mis, cen, subtype_order = run_knn_analysis()

logger.info("============= 报告 =============")
write_report(cm_fam, cm_sub, fam_mis, sub_mis, cen, subtype_order)

logger.info("=" * 60)
logger.info("ALL DONE. Outputs in %s", OUT)
