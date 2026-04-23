"""Figure D — MIxS 对齐结果：左 族堆叠条形 + 右 UNMAPPED 子类 Top10."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_io import load_csv  # noqa: E402
from viz_palette import FAMILY_COLORS, install_style, save_png_svg  # noqa: E402

ALIGNMENTS = ["exact", "subset", "superset", "partial", "UNMAPPED"]
FAMILY_ORDER = [
    "A_physicochemical", "B_env_categorical",
    "C_spatiotemporal", "D_other",
]


def main() -> None:
    install_style()

    mlog = load_csv("env4_mixs_alignment_log.csv")
    can = load_csv("env4_canonicals.csv")

    # family 信息从 canonicals 取（mlog 其实也有，但稳妥起见 merge）
    merged = mlog.merge(
        can[["canonical_id", "family", "subtype"]],
        on="canonical_id", how="left", suffixes=("_log", "")
    )
    if "family_log" in merged.columns:
        merged = merged.drop(columns=["family_log"])
    if "subtype_log" in merged.columns:
        merged = merged.drop(columns=["subtype_log"])

    # ── 左图：alignment × family 堆叠条形图 ──────────────────
    pivot = (
        merged.groupby(["mixs_alignment", "family"])
        .size().unstack(fill_value=0)
    )
    pivot = pivot.reindex(index=ALIGNMENTS, columns=FAMILY_ORDER, fill_value=0)

    fig = plt.figure(figsize=(13.5, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(ALIGNMENTS))
    bottom = np.zeros(len(ALIGNMENTS))
    for fam in FAMILY_ORDER:
        vals = pivot[fam].values.astype(int)
        if vals.sum() == 0:
            continue
        ax1.bar(x, vals, bottom=bottom,
                color=FAMILY_COLORS[fam], edgecolor="#222222",
                linewidth=0.5, width=0.65,
                label=fam.split("_", 1)[1].replace("_", " "))
        # 段内数字（若够大）
        for xi, v, b in zip(x, vals, bottom):
            if v >= 25:
                ax1.text(xi, b + v / 2, str(int(v)), ha="center",
                         va="center", fontsize=8.5,
                         color="white", fontweight="bold")
        bottom += vals
    # 每个柱子顶部总数
    for xi, total in enumerate(pivot.sum(axis=1).values):
        ax1.text(xi, total + can.shape[0] * 0.006, f"{int(total)}",
                 ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color="#222222")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ALIGNMENTS, fontsize=10)
    ax1.set_ylabel("Number of canonicals", fontsize=11)
    ax1.grid(axis="y", linestyle=":", alpha=0.35)
    ax1.set_axisbelow(True)

    # 留出顶部空间以容纳顶部标签
    ax1.set_ylim(0, max(pivot.sum(axis=1).values) * 1.22)

    # Legend 放左上（原 exact 柱上方空白）— 顶部窄版
    ax1.legend(title="Family", fontsize=8.5, title_fontsize=9,
               loc="upper left", bbox_to_anchor=(0.02, 0.97),
               frameon=True, ncol=1)

    # 覆盖率做成 (a) 标题副行，避免任何遮挡
    total_aligned = int((pivot.loc[["exact", "subset", "superset", "partial"]]).values.sum())
    total = int(pivot.values.sum())
    cov_pct = 100.0 * total_aligned / total
    ax1.set_title(
        f"(a) Canonicals by MIxS alignment type (n=1,874)\n"
        f"Any alignment: {total_aligned} / {total} ({cov_pct:.1f}%)",
        fontsize=11.5, fontweight="bold", loc="left",
    )

    # ── 右图：UNMAPPED Top10 subtype ──────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    unmapped = merged[merged["mixs_alignment"] == "UNMAPPED"].copy()
    by_sub = (
        unmapped.groupby(["family", "subtype"])
        .size().reset_index(name="n")
        .sort_values("n", ascending=False).head(10)
    )
    # 反转（matplotlib barh 上下方向）
    by_sub = by_sub.iloc[::-1].reset_index(drop=True)

    y = np.arange(len(by_sub))
    colors = [FAMILY_COLORS[f] for f in by_sub["family"]]
    ax2.barh(y, by_sub["n"].values,
             color=colors, edgecolor="#222222", linewidth=0.4)
    ax2.set_yticks(y)
    labels = [
        f"{r.subtype}  [{r.family.split('_',1)[0]}]"
        for _, r in by_sub.iterrows()
    ]
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel("UNMAPPED canonicals", fontsize=11)
    ax2.set_title("(b) Top-10 subtypes with most UNMAPPED canonicals",
                  fontsize=11.5, fontweight="bold", loc="left")
    ax2.grid(axis="x", linestyle=":", alpha=0.35)
    ax2.set_axisbelow(True)

    # 数字标签
    for i, v in enumerate(by_sub["n"].values):
        ax2.text(v + 2.5, i, str(int(v)), va="center",
                 ha="left", fontsize=8.5, fontweight="bold")
    ax2.set_xlim(0, by_sub["n"].max() * 1.28)

    fig.suptitle(
        "MIxS alignment coverage and unmapped residue",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.015,
        "Figure D. MIxS alignment outcomes on 1,874 canonicals. "
        "(a) 48% have any correspondence to the 89-slot hydrosphere MIxS subset. "
        "(b) UNMAPPED canonicals concentrate in env_state_context, trace_chemistry "
        "and spatial_metric, marking where hydrosphere research extends beyond "
        "current MIxS coverage.",
        ha="center", fontsize=8.5, style="italic", color="#555555",
    )

    plt.tight_layout(rect=(0.005, 0.005, 0.995, 0.965))
    png, svg = save_png_svg(fig, "viz_final_mixs_alignment")
    print(f"  → {png}")
    print(f"  → {svg}")


if __name__ == "__main__":
    main()
