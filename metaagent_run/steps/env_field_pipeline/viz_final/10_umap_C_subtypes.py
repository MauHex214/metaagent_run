"""Figure A-sup-C — C 族 subtype UMAP 分面板（3×3, 8 subtype + 1 all-C）."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_io import load_csv  # noqa: E402
from viz_palette import FAMILY_COLORS, install_style, save_png_svg  # noqa: E402

C_SUBTYPES = [
    "vertical_position", "time_point", "sampling_site", "geo_coord",
    "time_duration", "geo_region", "spatial_metric", "water_body_descriptor",
]


def main() -> None:
    install_style()

    df = load_csv("viz/env3_viz_coords.csv").dropna(subset=["x", "y"])
    df_C = df[df["family"] == "C_spatiotemporal"].copy()

    xmin, xmax = df["x"].min() - 0.5, df["x"].max() + 0.5
    ymin, ymax = df["y"].min() - 0.5, df["y"].max() + 0.5

    ncols, nrows = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 10),
                             sharex=True, sharey=True)
    axes = axes.ravel()

    # 色板：C 相关（蓝紫系）
    sub_colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))
    sub_colors = [sub_colors[i] for i in (0, 1, 2, 3, 4, 5, 6, 8)]

    for i, sub in enumerate(C_SUBTYPES):
        ax = axes[i]
        ax.scatter(df_C["x"], df_C["y"], s=3, alpha=0.12,
                   color="#CCCCCC", linewidths=0)
        sel = df_C[df_C["subtype"] == sub]
        if len(sel):
            ax.scatter(sel["x"], sel["y"], s=9, alpha=0.85,
                       color=sub_colors[i], edgecolors="white",
                       linewidths=0.2)
        ax.set_title(f"{sub}  (n={len(sel)})", fontsize=9.5,
                     fontweight="bold")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis="both", labelsize=7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#888888")

    # 第 9 格：all-C
    ax_sum = axes[8]
    ax_sum.scatter(df_C["x"], df_C["y"], s=4, alpha=0.40,
                   color=FAMILY_COLORS["C_spatiotemporal"], linewidths=0)
    ax_sum.set_title(f"all C-family  (n={len(df_C)})",
                     fontsize=9.5, fontweight="bold")
    ax_sum.set_xlim(xmin, xmax)
    ax_sum.set_ylim(ymin, ymax)
    ax_sum.tick_params(axis="both", labelsize=7)

    for ax in axes[-ncols:]:
        ax.set_xlabel("UMAP-1", fontsize=9)
    for ax in axes[::ncols]:
        ax.set_ylabel("UMAP-2", fontsize=9)

    fig.suptitle(
        "C-family spatiotemporal subtypes in UMAP space\n"
        "(n = 1,429 fields, 8 subtypes)",
        fontsize=13, fontweight="bold", y=1.0,
    )
    fig.text(
        0.5, -0.01,
        "Figure A-sup-C. Each panel highlights one C-family subtype "
        "(coloured) against all C-family fields (grey). time_point and "
        "geo_coord form dense, well-isolated clusters; sampling_site and "
        "geo_region bleed into each other as expected (both location "
        "descriptors at different scales); water_body_descriptor is the "
        "most diffuse.",
        ha="center", fontsize=8.5, style="italic", color="#555555",
    )
    plt.tight_layout(rect=(0, 0.005, 1, 0.95))
    png, svg = save_png_svg(fig, "viz_final_umap_C_subtypes")
    print(f"  → {png}")
    print(f"  → {svg}")


if __name__ == "__main__":
    main()
