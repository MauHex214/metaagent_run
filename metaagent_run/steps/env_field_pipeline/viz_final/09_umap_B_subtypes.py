"""Figure A-sup-B — B 族 subtype UMAP 分面板（2×3, 4 subtype + 1 all-B + 1 blank）."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_io import load_csv  # noqa: E402
from viz_palette import FAMILY_COLORS, install_style, save_png_svg  # noqa: E402

B_SUBTYPES = [
    "habitat_biome", "material_medium_type",
    "env_state_context", "other_categorical",
]


def main() -> None:
    install_style()

    df = load_csv("viz/env3_viz_coords.csv").dropna(subset=["x", "y"])
    df_B = df[df["family"] == "B_env_categorical"].copy()

    xmin, xmax = df["x"].min() - 0.5, df["x"].max() + 0.5
    ymin, ymax = df["y"].min() - 0.5, df["y"].max() + 0.5

    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 7),
                             sharex=True, sharey=True)
    axes = axes.ravel()

    # 色板：选 B 相关色系（青绿 → 蓝绿）
    sub_colors = plt.get_cmap("Set2")(np.linspace(0, 1, 8))
    sub_colors = [sub_colors[i] for i in (0, 1, 2, 3)]

    for i, sub in enumerate(B_SUBTYPES):
        ax = axes[i]
        ax.scatter(df_B["x"], df_B["y"], s=3, alpha=0.12,
                   color="#CCCCCC", linewidths=0)
        sel = df_B[df_B["subtype"] == sub]
        if len(sel):
            ax.scatter(sel["x"], sel["y"], s=10, alpha=0.85,
                       color=sub_colors[i], edgecolors="white",
                       linewidths=0.2)
        ax.set_title(f"{sub}  (n={len(sel)})", fontsize=10,
                     fontweight="bold")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis="both", labelsize=7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#888888")

    # 第 5 格：all-B
    ax_sum = axes[4]
    ax_sum.scatter(df_B["x"], df_B["y"], s=5, alpha=0.45,
                   color=FAMILY_COLORS["B_env_categorical"], linewidths=0)
    ax_sum.set_title(f"all B-family  (n={len(df_B)})",
                     fontsize=10, fontweight="bold")
    ax_sum.set_xlim(xmin, xmax)
    ax_sum.set_ylim(ymin, ymax)
    ax_sum.tick_params(axis="both", labelsize=7)

    # 第 6 格：留空
    axes[5].axis("off")

    for ax in axes[-ncols:]:
        if ax.axison:
            ax.set_xlabel("UMAP-1", fontsize=9)
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("UMAP-2", fontsize=9)

    fig.suptitle(
        "B-family environmental-categorical subtypes in UMAP space\n"
        "(n = 600 fields, 4 subtypes)",
        fontsize=13, fontweight="bold", y=1.0,
    )
    fig.text(
        0.5, -0.015,
        "Figure A-sup-B. Each panel highlights one B-family subtype "
        "(coloured) against all B-family fields (grey). habitat_biome and "
        "material_medium_type cluster together in the top-right region; "
        "env_state_context spreads more broadly, reflecting its catch-all "
        "role for trophic / land-use / climate descriptors.",
        ha="center", fontsize=8.5, style="italic", color="#555555",
    )
    plt.tight_layout(rect=(0, 0.005, 1, 0.94))
    png, svg = save_png_svg(fig, "viz_final_umap_B_subtypes")
    print(f"  → {png}")
    print(f"  → {svg}")


if __name__ == "__main__":
    main()
