"""Figure B-Sig — 4 环境 Signature 字段对比（4 行 × 1 列）."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_io import load_csv  # noqa: E402
from viz_palette import ENV_COLORS, install_style, save_png_svg  # noqa: E402

ENV_DISPLAY: dict[str, str] = {
    "Open_ocean": "Open ocean",
    "Coastal_waters": "Coastal waters",
    "Lake": "Lake",
    "Wetlands": "Wetlands",
}

# 每环境展示条数（不够就全展示）— 减到 15 避免文字拥挤
ENV_TOPK: dict[str, int] = {
    "Open_ocean": 15,
    "Coastal_waters": 17,   # 本来就 17
    "Lake": 15,
    "Wetlands": 21,         # 本来就 21
}


def main() -> None:
    install_style()

    fig, axes = plt.subplots(4, 1, figsize=(13, 16),
                             gridspec_kw={"hspace": 0.60,
                                          "height_ratios": [15, 17, 15, 21]})

    for ax, env in zip(axes, ["Open_ocean", "Coastal_waters", "Lake", "Wetlands"]):
        df = load_csv(f"env5_signature_{env}.csv")
        total_n = len(df)
        topk = min(ENV_TOPK[env], total_n)
        df = df.sort_values("total_pmid", ascending=False).head(topk).reset_index(drop=True)

        y_pos = np.arange(len(df))[::-1]
        ax.barh(y_pos, df["total_pmid"].values,
                color=ENV_COLORS[env], edgecolor="#222222",
                linewidth=0.4, alpha=0.9, height=0.72)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["target_field_name"], fontsize=9.5)
        ax.set_xlabel("Total PMID", fontsize=10)
        ax.grid(axis="x", linestyle=":", alpha=0.35, zorder=0)
        ax.set_axisbelow(True)

        # 留更多右侧空间给 "dominance/PMID" 组合标签
        x_max = df["total_pmid"].max()
        ax.set_xlim(0, x_max * 1.30)
        for i, (_, row) in enumerate(df.iterrows()):
            y = y_pos[i]
            dom = row["dominant_share"] * 100
            pmid = int(row["total_pmid"])
            # 组合标签：PMID 数 + 百分比（在 bar 外右端）
            ax.text(row["total_pmid"] + x_max * 0.012, y,
                    f"{pmid:,}  ·  {int(round(dom))}%",
                    va="center", ha="left",
                    fontsize=8.2, color="#333333", fontweight="bold")

        if topk < total_n:
            title = (f"{ENV_DISPLAY[env]} ({total_n} signature fields, "
                     f"showing top {topk})")
        else:
            title = f"{ENV_DISPLAY[env]} ({total_n} signature fields)"
        ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left")
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle(
        "Signature fields per hydrosphere sub-environment",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.005,
        "Figure B-sig. Signature fields for each hydrosphere sub-environment, "
        "sorted by total PMID. Labels at each bar's right end read "
        "PMID · dominant_share (i.e., total PMID count and the fraction of "
        "that count contributed by the labelled environment).",
        ha="center", fontsize=8.5, style="italic", color="#555555",
    )

    plt.tight_layout(rect=(0.005, 0.022, 0.995, 0.975))
    png, svg = save_png_svg(fig, "viz_final_signature_4env_comparison")
    print(f"  → {png}")
    print(f"  → {svg}")


if __name__ == "__main__":
    main()
