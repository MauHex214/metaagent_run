"""Figure B — Top 30 主清单字段：左 条形图 + 右 环境热力图."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_io import load_csv  # noqa: E402
from viz_palette import FAMILY_COLORS, install_style, save_png_svg  # noqa: E402

MIXS_SYMBOLS: dict[str, str] = {
    "exact": "e", "subset": "s", "superset": "S",
    "partial": "p", "UNMAPPED": "U",
}


def main() -> None:
    install_style()

    df = load_csv("env5_main_list.csv")
    df = df.sort_values("total_pmid", ascending=False).head(30).reset_index(drop=True)

    env_cols = ["env_pct_Open_ocean", "env_pct_Coastal_waters",
                "env_pct_Lake", "env_pct_Wetlands"]
    env_labels = ["Open_ocean", "Coastal_waters", "Lake", "Wetlands"]

    fig = plt.figure(figsize=(13.5, 11))
    gs = GridSpec(1, 2, width_ratios=[1.8, 1.0], wspace=0.08)

    # ── 左：条形图 ─────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0, 0])
    y_pos = np.arange(len(df))[::-1]  # 顶部最大
    colors = df["family"].map(FAMILY_COLORS).fillna("#999999").tolist()

    ax_bar.barh(y_pos, df["total_pmid"].values,
                color=colors, edgecolor="#222222", linewidth=0.5)
    ax_bar.set_yticks(y_pos)
    # y-label 用 quantity_kind（而不是 target_field_name）——qk 是 step 3 结构化
    # 标注的概念层标签，粒度更规整；target_field_name 是 v3 规则从 raw_key 里
    # 挑出的表面形式，可能因 PMID 最高的 raw_key 恰好是别名而偏移
    ax_bar.set_yticklabels(df["quantity_kind"], fontsize=9)
    ax_bar.set_xlabel("Total PMID", fontsize=11)
    ax_bar.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.tick_params(axis="x", labelsize=9)
    ax_bar.set_xlim(0, df["total_pmid"].max() * 1.08)
    ax_bar.invert_yaxis = lambda: None  # 保留显示

    # MIxS 符号（条形尾部）
    for i, (_, row) in enumerate(df.iterrows()):
        y = y_pos[i]
        sym = MIXS_SYMBOLS.get(row["mixs_alignment"], "?")
        ax_bar.text(
            row["total_pmid"] * 1.01, y, sym,
            va="center", ha="left", fontsize=8.5,
            fontweight="bold",
            color="#333333" if sym != "U" else "#999999",
        )

    # 数字标签
    for i, (_, row) in enumerate(df.iterrows()):
        y = y_pos[i]
        ax_bar.text(
            row["total_pmid"] * 0.02, y, f"{int(row['total_pmid']):,}",
            va="center", ha="left", fontsize=8,
            color="white", fontweight="bold",
        )

    # ── 右：环境热力图 ─────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[0, 1])
    heat = df[env_cols].values
    cmap = LinearSegmentedColormap.from_list(
        "envheat", ["#FFFFFF", "#FFE4B5", "#FB8C00", "#BF360C"]
    )
    im = ax_heat.imshow(heat, cmap=cmap, aspect="auto", vmin=0, vmax=100)
    ax_heat.set_yticks(range(len(df)))
    ax_heat.set_yticklabels([])
    ax_heat.set_xticks(range(4))
    ax_heat.set_xticklabels(env_labels, rotation=30, ha="right", fontsize=9)
    ax_heat.tick_params(axis="y", length=0)
    # 格子内数字（只在 ≥ 10% 才写，避免视觉过载）
    for i in range(len(df)):
        for j in range(4):
            v = heat[i, j]
            if v >= 10:
                tc = "white" if v >= 55 else "#222222"
                ax_heat.text(j, i, f"{int(round(v))}", ha="center", va="center",
                             fontsize=7.5, color=tc)

    # colorbar
    cax = fig.add_axes([0.92, 0.12, 0.014, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Env share (%)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # 共享 y-order：右图 row i 对应左图同一字段（都是 0 在顶部）
    # matplotlib imshow 默认 origin='upper'，刚好一致

    # ── 图例（family + MIxS 符号） ────────────────────────────
    fam_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=FAMILY_COLORS["A_physicochemical"],
                      edgecolor="#222", label="A: physicochemical"),
        plt.Rectangle((0, 0), 1, 1, facecolor=FAMILY_COLORS["B_env_categorical"],
                      edgecolor="#222", label="B: env_categorical"),
        plt.Rectangle((0, 0), 1, 1, facecolor=FAMILY_COLORS["C_spatiotemporal"],
                      edgecolor="#222", label="C: spatiotemporal"),
    ]
    leg1 = ax_bar.legend(handles=fam_handles, loc="lower right",
                         frameon=True, fontsize=9, title="Family",
                         title_fontsize=9)
    ax_bar.add_artist(leg1)

    # MIxS 符号说明
    fig.text(
        0.01, 0.012,
        "MIxS alignment symbols:  e = exact   s = subset   S = superset   "
        "p = partial   U = UNMAPPED",
        fontsize=8.5, color="#444444",
    )

    fig.suptitle(
        "Top 30 main-list fields by PMID coverage",
        fontsize=14, fontweight="bold", y=0.98,
    )

    fig.text(
        0.5, 0.03,
        "Figure B. Top 30 fields in the main list by total PMID. "
        "Row label is the step-3 quantity_kind (normalized concept label); "
        "left: bar length is total PMID, colour encodes family "
        "(A red, B green, C blue). Right: percentage of PMID "
        "contributed by each of four hydrosphere environments.",
        ha="center", fontsize=8.5, style="italic", color="#555555",
    )

    plt.tight_layout(rect=(0.01, 0.05, 0.915, 0.96))
    png, svg = save_png_svg(fig, "viz_final_main_list_heatmap")
    print(f"  → {png}")
    print(f"  → {svg}")


if __name__ == "__main__":
    main()
