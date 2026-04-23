"""Figure A-sup — A 族 subtype UMAP 分面板（3×4 subplots, 11 subtype + legend）."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_io import load_csv  # noqa: E402
from viz_palette import FAMILY_COLORS, install_style, save_png_svg  # noqa: E402

A_SUBTYPES = [
    "nutrient_chemistry", "trace_chemistry", "temperature", "salinity",
    "oxygen", "ph_alkalinity", "chlorophyll_pigment", "conductivity_tds",
    "turbidity_transparency", "physical_env_driver", "other_chemistry",
]


def main() -> None:
    install_style()

    df = load_csv("viz/env3_viz_coords.csv").dropna(subset=["x", "y"])
    df_A = df[df["family"] == "A_physicochemical"].copy()

    xmin, xmax = df["x"].min() - 0.5, df["x"].max() + 0.5
    ymin, ymax = df["y"].min() - 0.5, df["y"].max() + 0.5

    ncols = 4
    nrows = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 10),
                             sharex=True, sharey=True)
    axes = axes.ravel()

    # subtype 着色：hsv 调色板里挑 11 个（避开太浅/太亮的）
    sub_colors = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    sub_colors = [sub_colors[i] for i in (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 5)]

    for i, sub in enumerate(A_SUBTYPES):
        ax = axes[i]
        # 背景：所有 A 点（灰色）
        ax.scatter(df_A["x"], df_A["y"], s=3, alpha=0.12,
                   color="#CCCCCC", linewidths=0)
        # 前景：该 subtype 的点
        sel = df_A[df_A["subtype"] == sub]
        if len(sel):
            ax.scatter(sel["x"], sel["y"], s=8, alpha=0.85,
                       color=sub_colors[i], edgecolors="white",
                       linewidths=0.15)
        ax.set_title(f"{sub}  (n={len(sel)})", fontsize=9.5,
                     fontweight="bold")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis="both", labelsize=7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#888888")

    # 第 12 格：A-family summary
    ax_sum = axes[11]
    ax_sum.scatter(df_A["x"], df_A["y"], s=4, alpha=0.35,
                   color=FAMILY_COLORS["A_physicochemical"], linewidths=0)
    ax_sum.set_title(f"all A-family  (n={len(df_A)})",
                     fontsize=9.5, fontweight="bold")
    ax_sum.set_xlim(xmin, xmax)
    ax_sum.set_ylim(ymin, ymax)
    ax_sum.tick_params(axis="both", labelsize=7)

    # 整图 labels
    for ax in axes[-ncols:]:
        ax.set_xlabel("UMAP-1", fontsize=9)
    for ax in axes[::ncols]:
        ax.set_ylabel("UMAP-2", fontsize=9)

    fig.suptitle(
        "A-family physicochemical subtypes in UMAP space\n"
        "(n = 2,096 fields, 11 subtypes)",
        fontsize=13.5, fontweight="bold", y=1.01,
    )
    fig.text(
        0.5, -0.01,
        "Figure A-sup. Each panel highlights one A-family subtype (coloured) "
        "against all A-family fields (grey). Subtypes such as oxygen, "
        "chlorophyll_pigment and temperature form clearly separated clusters, "
        "while nutrient_chemistry and trace_chemistry are broader — consistent "
        "with their chemical diversity.",
        ha="center", fontsize=8.5, style="italic", color="#555555",
    )
    plt.tight_layout(rect=(0, 0.005, 1, 0.965))
    png, svg = save_png_svg(fig, "viz_final_umap_A_subtypes")
    print(f"  → {png}")
    print(f"  → {svg}")


if __name__ == "__main__":
    main()
