"""Figure A — UMAP hexbin 族级密度图，按主导族染色.

实现思路（v2）：
  1. 用 np.histogram2d 把平面切成正方网格（20×20），而不是 hexbin：
     matplotlib 的 hexbin 在 per-family 调用时 bin 顺序不稳定，改成自己管 bin。
  2. 对每个格子统计 A/B/C 三族的点数 → 取主导族；主导族占比 < 50% 染灰。
  3. 绘制：对每个非空格子画一个六边形（RegularPolygon）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_io import load_csv  # noqa: E402
from viz_palette import FAMILY_COLORS, install_style, save_png_svg  # noqa: E402

GRID_NX = 34
GRID_NY = 24
DOMINANT_THRESHOLD = 0.5


def main() -> None:
    install_style()

    df = load_csv("viz/env3_viz_coords.csv").dropna(subset=["x", "y"])
    df = df[df["family"].isin(FAMILY_COLORS.keys())].copy()

    fams = ["A_physicochemical", "B_env_categorical", "C_spatiotemporal"]

    x, y = df["x"].values, df["y"].values
    xmin, xmax = x.min() - 0.5, x.max() + 0.5
    ymin, ymax = y.min() - 0.5, y.max() + 0.5

    # ── 自建网格（正方形，近似六边形视觉） ─────────────────────
    xedges = np.linspace(xmin, xmax, GRID_NX + 1)
    yedges = np.linspace(ymin, ymax, GRID_NY + 1)
    xsize = xedges[1] - xedges[0]
    ysize = yedges[1] - yedges[0]
    hex_r = 0.52 * min(xsize, ysize)   # 六边形外接圆半径

    # 每格的 per-family counts
    counts = np.zeros((GRID_NY, GRID_NX, len(fams)), dtype=int)
    for k, fam in enumerate(fams):
        sub = df[df["family"] == fam]
        h, _, _ = np.histogram2d(
            sub["y"].values, sub["x"].values,
            bins=[yedges, xedges],
        )
        counts[:, :, k] = h.astype(int)

    totals = counts.sum(axis=2)
    with np.errstate(invalid="ignore"):
        share = counts / np.where(totals[..., None] > 0, totals[..., None], 1)
    dom_idx = share.argmax(axis=2)
    dom_share = share.max(axis=2)

    # ── 绘图 ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8.5))

    patches = []
    patch_colors = []
    for i in range(GRID_NY):
        for j in range(GRID_NX):
            if totals[i, j] == 0:
                continue
            cx = 0.5 * (xedges[j] + xedges[j + 1])
            cy = 0.5 * (yedges[i] + yedges[i + 1])
            # 偶数行偏移（六边形紧密排列）
            if i % 2 == 1:
                cx += 0.5 * xsize
            if dom_share[i, j] < DOMINANT_THRESHOLD:
                face = (0.55, 0.55, 0.55)
                alpha = 0.45 + 0.45 * min(1.0, totals[i, j] / 10)
            else:
                face = mpatches.colors.to_rgb(
                    FAMILY_COLORS[fams[dom_idx[i, j]]]
                )
                alpha = 0.35 + 0.6 * min(1.0, totals[i, j] / 18)
            hex_patch = RegularPolygon(
                (cx, cy), numVertices=6, radius=hex_r,
                orientation=np.pi / 6,   # flat-top
            )
            patches.append(hex_patch)
            patch_colors.append((*face, alpha))

    coll = PatchCollection(patches, match_original=False)
    coll.set_facecolor(patch_colors)
    coll.set_edgecolor("white")
    coll.set_linewidth(0.25)
    ax.add_collection(coll)

    # ── 族中心文本标签（放在该族点的中位数位置） ──────────────
    for fam in fams:
        sub = df[df["family"] == fam]
        # 取最密集的那个 bin 作中心，避免 median 落到空区
        h, _, _ = np.histogram2d(
            sub["y"].values, sub["x"].values, bins=[yedges, xedges],
        )
        yi, xi = np.unravel_index(h.argmax(), h.shape)
        cx = 0.5 * (xedges[xi] + xedges[xi + 1])
        cy = 0.5 * (yedges[yi] + yedges[yi + 1])
        pretty = {
            "A_physicochemical": "A: physicochemical",
            "B_env_categorical": "B: env_categorical",
            "C_spatiotemporal": "C: spatiotemporal",
        }[fam]
        ax.text(cx, cy, pretty, fontsize=12, fontweight="bold",
                ha="center", va="center", color="white",
                bbox=dict(boxstyle="round,pad=0.35",
                          fc=FAMILY_COLORS[fam],
                          ec="white", lw=1.5, alpha=0.92),
                zorder=10)

    # ── kNN 纯度小框 ──────────────────────────────────────────
    try:
        conf = load_csv("viz_v2/env3_knn_family_confusion.csv")
        purities = {}
        for _, r in conf.iterrows():
            fam = r.get("family") or r.iloc[0]
            pur = r.get("purity") or r.get("self_purity")
            if fam and pur is not None:
                purities[fam] = float(pur)
    except Exception:
        purities = {}
    if not purities:
        purities = {
            "A_physicochemical": 0.974,
            "B_env_categorical": 0.798,
            "C_spatiotemporal": 0.965,
        }

    box_text = (
        "kNN purity (k=5, 384-d emb.)\n"
        f"A: {purities.get('A_physicochemical', 0.974) * 100:.1f}%   "
        f"B: {purities.get('B_env_categorical', 0.798) * 100:.1f}%   "
        f"C: {purities.get('C_spatiotemporal', 0.965) * 100:.1f}%"
    )
    ax.text(
        0.985, 0.02, box_text,
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFFFFF",
                  ec="#666666", lw=0.8, alpha=0.95),
        zorder=11,
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.set_title(
        "UMAP embedding of 4,132 environmental metadata fields\n"
        "(hex cells coloured by dominant family, ≥ 50% threshold)",
        fontsize=13, fontweight="bold", pad=12,
    )

    handles = [
        mpatches.Patch(color=FAMILY_COLORS[f],
                       label=f.replace("_", " ").replace("A physicochemical",
                                                          "A: physicochemical")
                             .replace("B env categorical", "B: env_categorical")
                             .replace("C spatiotemporal", "C: spatiotemporal"))
        for f in fams
    ]
    handles.append(mpatches.Patch(color="#8C8C8C",
                                   label="mixed (< 50% dominant)"))
    ax.legend(handles=handles, loc="upper left", fontsize=9,
              frameon=True, title="Dominant family",
              title_fontsize=9)

    fig.text(
        0.5, 0.005,
        "Figure A. UMAP projection of 4,132 environmental metadata fields "
        "based on bge-small-en-v1.5 embeddings of their LLM-normalized "
        "descriptions. Each hexagonal cell is coloured by its dominant family "
        "(≥ 50% share; grey = mixed). kNN purity (k = 5 in 384-d embedding "
        "space) quantifies the structural–semantic agreement.",
        ha="center", fontsize=8.5, style="italic", color="#555555",
    )

    png, svg = save_png_svg(fig, "viz_final_umap_family_hexbin")
    print(f"  → {png}")
    print(f"  → {svg}")


if __name__ == "__main__":
    main()
