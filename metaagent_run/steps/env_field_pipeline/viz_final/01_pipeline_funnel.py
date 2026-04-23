"""Figure C — Pipeline 数据漏斗图 (v2 layout fix).

横向漏斗：Stage 0 → 1 → 2 → 3-4 → Stage 5 (三分栏 Main / Signature / Niche)。
PMID 覆盖率以 phase4 pipeline pmid (154,155) 为基准显示主线阶段；
stage 1 的 76.94% 显式标注为 "vs Stage 0"，避免口径混淆。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_io import load_csv, load_json  # noqa: E402
from viz_palette import CATEGORY_COLORS, install_style, save_png_svg  # noqa: E402


def main() -> None:
    install_style()

    s0 = load_json("env0_stats.json")
    s1 = load_json("env1_stats.json")
    s2 = load_json("env2_eval_report.json")
    c4 = load_csv("env4_canonicals.csv")
    c5 = load_csv("env5_canonical_classified.csv")
    m5 = load_csv("env5_main_list.csv")

    phase4_pmid = int(c4["total_pmid"].sum())
    main_pmid = int(m5["total_pmid"].sum())
    main_cov_pct = 100.0 * main_pmid / phase4_pmid

    cat_dist = c5["category"].value_counts().to_dict()
    n_universal = int(cat_dist.get("Universal", 0))
    n_cross = int(cat_dist.get("Cross-biome common", 0))
    n_sig = int(cat_dist.get("Signature", 0))
    n_niche = int(cat_dist.get("Niche", 0))
    n_main = int(len(m5))
    n_sig_files = 65 + 17 + 37 + 21

    # ── 布局 ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(17, 8.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # 5 段水平中心：前 4 段紧凑，Stage 5 拉开
    X = [7, 22, 37, 52, 80]
    W = 10.5
    Y_CENTER = 5.5
    bar_h_max = 3.2

    counts_main = [s0["total_raw_keys"], s1["mainstream_count"],
                   s2["kept"], len(c4)]
    max_count = max(counts_main)

    def bar_h(n: int) -> float:
        return bar_h_max * (n / max_count) ** 0.55

    stage_specs = [
        {
            "title": "Stage 0\nraw metadata keys",
            "count_fmt": f"{s0['total_raw_keys']:,}",
            "sub": f"{s0['total_pmid_sum_double_counted']:,} PMID×env triples",
            "count": s0["total_raw_keys"], "color": "#D6D6D6",
        },
        {
            "title": "Stage 1\nmainstream pool",
            "count_fmt": f"{s1['mainstream_count']:,}",
            "sub": f"{s1['mainstream_pmid_coverage_pct']:.2f}% PMID\n(vs Stage 0)",
            "count": s1["mainstream_count"], "color": "#BDBDBD",
        },
        {
            "title": "Stage 2\nenv-metadata kept",
            "count_fmt": f"{s2['kept']:,}",
            "sub": f"recall {s2['eval']['recall_on_historical_EXCLUDE']['rate_pct']:.1f}%\n"
                   f"FN {s2['eval']['false_negative_rate_pct']:.1f}%",
            "count": s2["kept"], "color": "#A8A8A8",
        },
        {
            "title": "Stage 3–4\ncanonicals",
            "count_fmt": f"{len(c4):,}",
            "sub": "4-slot annotation\n+ string clustering",
            "count": len(c4), "color": "#909090",
        },
    ]

    side_specs = [
        {
            "at_stage": 0,
            "label": f"{s1['orphan_count']:,} orphans\n({100.0 - s1['mainstream_pmid_coverage_pct']:.2f}% PMID)\narchived",
        },
        {
            "at_stage": 1,
            "label": f"{s2['excluded']:,} excluded\n(tools / IDs / QC)",
        },
    ]

    transition_reasons = [
        "PMID ≥ 3\nthreshold",
        "LLM binary\nis_env_metadata",
        "structured\nannotation",
        "bucket merge\n+ string cluster",
    ]

    # ── 绘制前 4 段 ───────────────────────────────────────────────
    for i, spec in enumerate(stage_specs):
        cx = X[i]
        h = bar_h(spec["count"])
        rect = Rectangle(
            (cx - W / 2, Y_CENTER - h / 2), W, h,
            facecolor=spec["color"], edgecolor="#333333",
            linewidth=1.2, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(cx, Y_CENTER + h / 2 + 1.15, spec["title"],
                ha="center", va="center", fontsize=10.5,
                fontweight="bold", zorder=4)
        ax.text(cx, Y_CENTER, spec["count_fmt"],
                ha="center", va="center", fontsize=15,
                fontweight="bold", color="white", zorder=5)
        ax.text(cx, Y_CENTER - h / 2 - 0.95, spec["sub"],
                ha="center", va="center", fontsize=8.2,
                color="#444444", zorder=4)

    # ── 阶段间主箭头（前 4 段之间） ─────────────────────────────
    for i in range(3):
        x0 = X[i] + W / 2
        x1 = X[i + 1] - W / 2
        arrow = FancyArrowPatch(
            (x0 + 0.2, Y_CENTER), (x1 - 0.2, Y_CENTER),
            arrowstyle="-|>", mutation_scale=18,
            linewidth=1.6, color="#2B2B2B", zorder=2,
        )
        ax.add_patch(arrow)
        mid_x = (x0 + x1) / 2
        ax.text(mid_x, Y_CENTER + 0.55, transition_reasons[i],
                ha="center", va="center", fontsize=8,
                color="#444444", style="italic", zorder=4)

    # ── 旁路箭头（虚线） ─────────────────────────────────────────
    for spec in side_specs:
        i = spec["at_stage"]
        x0 = X[i] + W / 2
        x_drop = X[i] + (X[i + 1] - X[i]) / 2
        arrow = FancyArrowPatch(
            (x0 - 1.5, Y_CENTER - bar_h(stage_specs[i]["count"]) / 2 - 0.1),
            (x_drop, 1.4),
            arrowstyle="-|>", mutation_scale=12,
            linewidth=1.1, color="#999999",
            linestyle="--", zorder=2,
        )
        ax.add_patch(arrow)
        ax.text(x_drop, 0.85, spec["label"],
                ha="center", va="top", fontsize=8,
                color="#666666", zorder=3)

    # ── Stage 3-4 → Stage 5 总箭头 ────────────────────────────
    s5_cx = X[4]
    box_w = 22
    box_h = 1.8
    gap = 0.5
    y_top = Y_CENTER + box_h + gap
    y_mid = Y_CENTER
    y_bot = Y_CENTER - box_h - gap

    # 主箭头分叉到三个 panel
    for y in (y_top, y_mid, y_bot):
        arrow = FancyArrowPatch(
            (X[3] + W / 2 + 0.2, Y_CENTER),
            (s5_cx - box_w / 2 - 0.2, y),
            arrowstyle="-|>", mutation_scale=15,
            linewidth=1.3, color="#2B2B2B", zorder=2,
        )
        ax.add_patch(arrow)
    ax.text(
        (X[3] + W / 2 + s5_cx - box_w / 2) / 2, Y_CENTER + 2.1,
        "H_norm × PMID\nclassification",
        ha="center", va="center", fontsize=8,
        color="#444444", style="italic", zorder=4,
    )

    # Stage 5 标题
    ax.text(s5_cx, y_top + box_h / 2 + 0.8, "Stage 5 outputs",
            ha="center", va="center", fontsize=11.5,
            fontweight="bold", zorder=4)

    # ── Stage 5 三分栏（明确左侧 label / 右侧数据） ──────────────
    s5_panels = [
        {
            "y": y_top,
            "label": f"{n_main}\nMain list",
            "detail": (
                f"Universal {n_universal} + Cross-biome {n_cross}\n"
                f"→ post Rule A collapse: {n_main}\n"
                f"PMID coverage: {main_cov_pct:.2f}%  (vs phase 4 pool)"
            ),
            "color": CATEGORY_COLORS["Universal"],
        },
        {
            "y": y_mid,
            "label": f"{n_sig_files}\nSignature",
            "detail": (
                f"from {n_sig} Signature-classified canonicals\n"
                f"OO 65  |  CW 17  |  LK 37  |  WL 21"
            ),
            "color": CATEGORY_COLORS["Signature"],
        },
        {
            "y": y_bot,
            "label": f"{n_niche}\nNiche",
            "detail": (
                "Low H_norm, low PMID.\n"
                "Preserved in env5_full_traceability.csv\n"
                "for downstream queries but not in main list."
            ),
            "color": CATEGORY_COLORS["Niche"],
        },
    ]

    for p in s5_panels:
        box = FancyBboxPatch(
            (s5_cx - box_w / 2, p["y"] - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor=p["color"], edgecolor="#333333",
            linewidth=1.1, zorder=3, alpha=0.92,
        )
        ax.add_patch(box)
        # 左侧：大标签
        ax.text(s5_cx - box_w / 2 + 1.0, p["y"], p["label"],
                ha="left", va="center", fontsize=13,
                fontweight="bold", color="white", zorder=4)
        # 右侧：详情（左对齐，独占右 2/3）
        ax.text(s5_cx - box_w / 2 + 7.5, p["y"], p["detail"],
                ha="left", va="center", fontsize=8.2,
                color="white", zorder=4)

    # ── 整图 title ───────────────────────────────────────────────
    ax.set_title(
        "Pipeline data flow: 53,068 raw keys → 443 target fields "
        "(93.23% PMID coverage)",
        pad=18, fontsize=14, fontweight="bold",
    )

    fig.text(
        0.5, 0.01,
        "Figure C. Pipeline data flow from 53,068 raw metadata keys to 443 "
        "final target fields. Numbers in solid bars are raw_key counts; "
        "dashed-arrow side branches denote records removed from the main "
        "pipeline but retained for audit. Main-list PMID coverage is "
        "relative to the post-phase-4 canonical pool (154,155 PMID×env).",
        ha="center", fontsize=8.5, style="italic", color="#555555",
    )

    fig.tight_layout(rect=(0.005, 0.04, 0.995, 0.96))
    png, svg = save_png_svg(fig, "viz_final_pipeline_funnel")
    print(f"  → {png}")
    print(f"  → {svg}")


if __name__ == "__main__":
    main()
