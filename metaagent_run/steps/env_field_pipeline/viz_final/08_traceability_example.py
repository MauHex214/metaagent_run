"""Figure F — Traceability 链示例（v3: salinity only, 展示 Step 3 → 4 → 5）.

单面板，讲清楚 3 个阶段的收敛：
  Step 3  每个 raw_key 被标成 4 槽 (family, subtype, qk, bag)
  Step 4  按 (family, subtype, qk, sorted(bag)) 分桶 + 字符串聚类 → 7 个 canonical
  Step 5  coarse granularity collapse：(family, subtype, qk) 相同的 canonical 合一个 target

核心信息：Step 5 把 7 个 bag-variant canonical 合并成 1 个 target_field=salinity
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_io import load_csv  # noqa: E402
from viz_palette import (CATEGORY_COLORS, FAMILY_COLORS, install_style,  # noqa: E402
                          save_png_svg)

TARGET = "salinity"
TARGET_FAMILY = "A_physicochemical"
# 每个 bag 组展示多少 raw_key 样本（主要为了避免图太长）
PER_BAG_CAP = 4


def main() -> None:
    install_style()

    main_df = load_csv("env5_main_list.csv")
    map_df = load_csv("env4_canonical_to_raw_key.csv")
    annot_df = load_csv("env3_final_annotations.csv")
    can_df = load_csv("env4_canonicals.csv")

    # Salinity target row
    r = main_df[main_df["target_field_name"] == TARGET].iloc[0]
    merged_cids = [c for c in str(r["merged_canonicals"]).split("|") if c]
    # 按 canonical_total_pmid 降序（c_00015 > c_00282 > ...）
    can_rows = (
        can_df[can_df["canonical_id"].isin(merged_cids)]
        .sort_values("total_pmid", ascending=False)
        .reset_index(drop=True)
    )

    # 每个 canonical 的成员 raw_keys（按 pmid 降序，截前 PER_BAG_CAP）
    rk_pmid = dict(zip(annot_df["raw_key"], annot_df["total_pmid"]))
    rk_bag = dict(zip(annot_df["raw_key"], annot_df["modifier_bag"]))
    can_members: dict[str, pd.DataFrame] = {}
    for cid in can_rows["canonical_id"]:
        sub = map_df[map_df["canonical_id"] == cid].copy()
        sub["pmid"] = sub["raw_key"].map(rk_pmid).fillna(0).astype(int)
        sub = sub.sort_values("pmid", ascending=False).reset_index(drop=True)
        can_members[cid] = sub

    # ── 布局 ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.set_xlim(0, 1)
    total_rows = sum(min(len(v), PER_BAG_CAP) + 1.3
                     for v in can_members.values())
    ax.set_ylim(total_rows + 3, -2.5)   # y 反向（0 在顶部），顶留 title，底留 step5 footer
    ax.axis("off")

    # ── 三阶段列头 ─────────────────────────────────────────────
    x_raw = 0.08
    x_can = 0.52
    x_tgt = 0.88

    header_y = -1.3
    ax.text(x_raw, header_y, "Step 3 — raw_key + 4-slot annotation",
            fontsize=12, fontweight="bold", color=FAMILY_COLORS[TARGET_FAMILY],
            ha="left", va="center")
    ax.text(x_can, header_y, "Step 4 — canonicals\n(by bucket key + string clustering)",
            fontsize=12, fontweight="bold", color="#444444",
            ha="center", va="center")
    ax.text(x_tgt, header_y, "Step 5 — target_field\n(coarse granularity collapse)",
            fontsize=12, fontweight="bold",
            color=CATEGORY_COLORS[r["category"]],
            ha="center", va="center")

    # ── 逐个 canonical 画一组 ─────────────────────────────────
    y_cursor = 0.4
    canonical_anchors = []   # 记录每个 canonical box 的 (y_center) 用于连线

    for idx, can_row in can_rows.iterrows():
        cid = can_row["canonical_id"]
        bag = can_row.get("modifier_bag")
        bag_str = str(bag) if pd.notna(bag) and str(bag) != "nan" else "(empty)"
        mem = can_members[cid]
        shown = mem.head(PER_BAG_CAP)
        n_total = len(mem)
        n_rows_this_group = len(shown)
        group_h = n_rows_this_group + 0.7
        group_top = y_cursor
        group_bot = y_cursor + group_h
        y_center = (group_top + group_bot) / 2

        # 组背景（浅色条带区分 bag 组）
        band_color = "#F8F8F8" if idx % 2 == 0 else "#EEEEEE"
        ax.add_patch(Rectangle(
            (0.005, group_top - 0.2), 0.99, group_h + 0.15,
            facecolor=band_color, edgecolor="none", zorder=0,
        ))

        # 左列：raw_keys
        for i, (_, row_m) in enumerate(shown.iterrows()):
            y = group_top + i + 0.4
            is_rep = int(row_m["is_representative"]) == 1
            rk = row_m["raw_key"]
            if len(rk) > 28:
                rk = rk[:25] + "…"
            raw_bag = rk_bag.get(row_m["raw_key"])
            raw_bag_str = (str(raw_bag) if pd.notna(raw_bag)
                           and str(raw_bag) != "nan" else "(none)")
            # raw_key name
            ax.text(x_raw, y, rk,
                    ha="left", va="center", fontsize=9,
                    fontweight="bold" if is_rep else "normal",
                    color=FAMILY_COLORS[TARGET_FAMILY] if is_rep
                    else "#333333")
            # 4-slot tag： (family=A / subtype=salinity / qk=salinity / bag=X)
            tag = f"  4-slot: A / salinity / salinity / bag={raw_bag_str}"
            ax.text(x_raw, y + 0.22, tag,
                    ha="left", va="center", fontsize=7,
                    color="#777777", style="italic")
            # PMID
            ax.text(x_raw + 0.25, y - 0.08, f"PMID {int(row_m['pmid']):,}",
                    ha="left", va="center", fontsize=7.5,
                    color="#555555")
        # 组底部 "... N more" 提示
        if n_total > PER_BAG_CAP:
            ax.text(x_raw, group_bot - 0.15,
                    f"... {n_total - PER_BAG_CAP} more raw_keys in this bucket",
                    ha="left", va="center", fontsize=7.5,
                    color="#999999", style="italic")

        # 中列：canonical box
        box_w = 0.20
        box_h = min(group_h * 0.88, 2.6)
        box_y0 = y_center - box_h / 2
        can_bag = can_row.get("modifier_bag")
        can_bag_str = (str(can_bag) if pd.notna(can_bag)
                       and str(can_bag) != "nan" else "(empty)")
        # 主 c_00015 用深色描边
        is_main_cid = (cid == r["canonical_id"])
        edge_color = FAMILY_COLORS[TARGET_FAMILY] if is_main_cid else "#AAAAAA"
        edge_lw = 1.8 if is_main_cid else 0.9
        box = FancyBboxPatch(
            (x_can - box_w / 2, box_y0), box_w, box_h,
            boxstyle="round,pad=0.012,rounding_size=0.008",
            facecolor="#FFFFFF", edgecolor=edge_color,
            linewidth=edge_lw, zorder=2,
        )
        ax.add_patch(box)
        ax.text(x_can, y_center - box_h * 0.32,
                f"{cid}", ha="center", va="center",
                fontsize=10, fontweight="bold",
                color=FAMILY_COLORS[TARGET_FAMILY] if is_main_cid else "#222222")
        ax.text(x_can, y_center - box_h * 0.08,
                f"{can_row['canonical_name']}",
                ha="center", va="center", fontsize=9)
        ax.text(x_can, y_center + box_h * 0.15,
                f"bag = {can_bag_str}",
                ha="center", va="center", fontsize=8,
                color="#555555", style="italic")
        ax.text(x_can, y_center + box_h * 0.36,
                f"n_raw={n_total}  PMID={int(can_row['total_pmid']):,}",
                ha="center", va="center", fontsize=7.5,
                color="#666666")

        canonical_anchors.append((y_center, cid))

        # Step 4 收敛箭头：每个 raw_key → 该 canonical
        for i in range(n_rows_this_group):
            y = group_top + i + 0.4
            ax.plot(
                [x_raw + 0.32, x_can - box_w / 2],
                [y, y_center],
                color="#BBBBBB", linewidth=0.5, zorder=1, alpha=0.7,
            )

        y_cursor = group_bot + 0.4

    # ── 右侧：合并的 target box + Step 5 箭头 ─────────────────
    tgt_w = 0.14
    tgt_h = 2.6
    tgt_cy = total_rows / 2 + 0.5
    tgt_color = CATEGORY_COLORS.get(r["category"], "#666666")
    tgt_box = FancyBboxPatch(
        (x_tgt - tgt_w / 2, tgt_cy - tgt_h / 2),
        tgt_w, tgt_h,
        boxstyle="round,pad=0.012,rounding_size=0.008",
        facecolor=tgt_color, edgecolor="#222222",
        linewidth=1.4, alpha=0.95, zorder=3,
    )
    ax.add_patch(tgt_box)
    ax.text(x_tgt, tgt_cy - 0.8, TARGET,
            ha="center", va="center", fontsize=13,
            fontweight="bold", color="white")
    ax.text(x_tgt, tgt_cy - 0.3, f"[{r['category']}]",
            ha="center", va="center", fontsize=9,
            color="white")
    ax.text(x_tgt, tgt_cy + 0.15,
            f"PMID: {int(r['total_pmid']):,}",
            ha="center", va="center", fontsize=9,
            color="white", fontweight="bold")
    ax.text(x_tgt, tgt_cy + 0.55,
            f"H_norm: {float(r['H_norm']):.2f}",
            ha="center", va="center", fontsize=9,
            color="white")
    ax.text(x_tgt, tgt_cy + 0.9,
            f"n_raw_keys: {int(r['n_member_raw_keys'])}",
            ha="center", va="center", fontsize=9,
            color="white")

    # Step 5 箭头：每个 canonical → target
    for yc, cid in canonical_anchors:
        arrow = FancyArrowPatch(
            (x_can + 0.20 / 2 + 0.005, yc),
            (x_tgt - tgt_w / 2 - 0.005, tgt_cy),
            arrowstyle="-|>", mutation_scale=12,
            linewidth=1.2, color="#333333", zorder=2,
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)

    # ── Step 5 高亮说明（本图最关键信息） ──────────────────────
    highlight_y = total_rows + 1.8
    highlight_text = (
        f"Step 5 coarse-granularity collapse:  "
        f"{len(merged_cids)} bag-variant canonicals  "
        f"→  1 target_field '{TARGET}'"
    )
    highlight_box = FancyBboxPatch(
        (0.08, highlight_y - 0.55), 0.84, 1.1,
        boxstyle="round,pad=0.02,rounding_size=0.015",
        facecolor="#FFF3CD", edgecolor="#E63946",
        linewidth=2.0, zorder=5,
    )
    ax.add_patch(highlight_box)
    ax.text(0.5, highlight_y - 0.15, highlight_text,
            ha="center", va="center", fontsize=12,
            fontweight="bold", color="#AA2020", zorder=6)
    ax.text(0.5, highlight_y + 0.25,
            "Same (family, subtype, quantity_kind) = (A, salinity, salinity);  "
            "bag variants (empty / surface / bottom / porewater / mean / range / total) "
            "unified under the highest-PMID canonical's representative name.",
            ha="center", va="center", fontsize=8.8,
            color="#555555", style="italic", zorder=6)

    # 整图 title
    ax.set_title(
        f"Traceability chain for target_field = '{TARGET}'\n"
        f"65 raw_keys → 7 canonicals → 1 target",
        fontsize=14, fontweight="bold", pad=12,
    )

    fig.text(
        0.5, 0.008,
        "Figure F. End-to-end traceability for the main-list target_field "
        "'salinity'. Step 3 assigns each raw_key a four-slot annotation "
        "(family / subtype / quantity_kind / modifier_bag). Step 4 groups "
        "raw_keys sharing the same four-tuple and collapses near-identical "
        "strings into canonicals — producing 7 bag-variant canonicals here. "
        "Step 5 coarse-granularity collapse then merges all 7 into a single "
        "target because they share (family, subtype, quantity_kind), differing "
        "only in bag. c_00015 (bag=empty, PMID 5,213) supplies the "
        "representative name.",
        ha="center", fontsize=8.5, style="italic", color="#555555",
    )

    plt.tight_layout(rect=(0, 0.025, 1, 0.965))
    png, svg = save_png_svg(fig, "viz_final_traceability_example")
    print(f"  → {png}")
    print(f"  → {svg}")


if __name__ == "__main__":
    main()
