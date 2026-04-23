"""Figure E — 阈值敏感性：左 Universal 2D 热图 + 右 Signature 3-panel."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_io import load_csv, load_json  # noqa: E402
from viz_palette import install_style, save_png_svg  # noqa: E402


def main() -> None:
    install_style()

    cal = load_csv("env5_threshold_calibration.csv")
    thr = load_json("env5_thresholds.json")

    # ── 左图：Universal 敏感性 ───────────────────────────────────
    # 固定 H_signature_max=0.40, share_sig=0.70, pmid_sig=5
    sub_u = cal[
        (cal["H_signature_max"] == 0.40)
        & (cal["dominant_share_min"] == 0.70)
        & (cal["pmid_signature_min"] == 5)
    ].copy()
    H_u_grid = sorted(sub_u["H_universal_min"].unique())
    pmid_u_grid = sorted(sub_u["pmid_universal_min"].unique())
    heat_u = np.zeros((len(H_u_grid), len(pmid_u_grid)), dtype=int)
    for i, h in enumerate(H_u_grid):
        for j, p in enumerate(pmid_u_grid):
            row = sub_u[
                (sub_u["H_universal_min"] == h)
                & (sub_u["pmid_universal_min"] == p)
            ]
            heat_u[i, j] = int(row["n_universal"].iloc[0]) if len(row) else 0

    # ── 右图（3-panel）：Signature 敏感性 ────────────────────────
    # 固定 pmid_univ=50, H_univ=0.85
    sub_s = cal[
        (cal["pmid_universal_min"] == 50)
        & (cal["H_universal_min"] == 0.85)
    ].copy()
    share_grid = sorted(sub_s["dominant_share_min"].unique())
    pmid_s_grid = sorted(sub_s["pmid_signature_min"].unique())
    H_s_panels = sorted(sub_s["H_signature_max"].unique())
    H_s_panels_display = [h for h in H_s_panels if h in (0.30, 0.40, 0.50)]
    heat_s = {}
    for h in H_s_panels_display:
        mat = np.zeros((len(share_grid), len(pmid_s_grid)), dtype=int)
        for i, s in enumerate(share_grid):
            for j, p in enumerate(pmid_s_grid):
                row = sub_s[
                    (sub_s["H_signature_max"] == h)
                    & (sub_s["dominant_share_min"] == s)
                    & (sub_s["pmid_signature_min"] == p)
                ]
                mat[i, j] = int(row["n_signature"].iloc[0]) if len(row) else 0
        heat_s[h] = mat

    # ── 布局 ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 6.3))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.0, 1.3], wspace=0.25,
        top=0.82, bottom=0.18,
    )
    gs_left = gs[0, 0].subgridspec(1, 1)
    ax_u = fig.add_subplot(gs_left[0, 0])

    im_u = ax_u.imshow(heat_u, cmap="YlOrRd", aspect="auto",
                       vmin=0, vmax=heat_u.max())
    ax_u.set_xticks(range(len(pmid_u_grid)))
    ax_u.set_xticklabels(pmid_u_grid, fontsize=9)
    ax_u.set_yticks(range(len(H_u_grid)))
    ax_u.set_yticklabels([f"{h:.2f}" for h in H_u_grid], fontsize=9)
    ax_u.set_xlabel("pmid_universal_min", fontsize=11)
    ax_u.set_ylabel("H_universal_min", fontsize=11)
    ax_u.set_title(
        "(a) Universal count sensitivity\n"
        "fixed H_sig=0.40, share=0.70, pmid_sig=5",
        fontsize=11.5, fontweight="bold", loc="left",
    )
    # 格子内数字
    for i in range(len(H_u_grid)):
        for j in range(len(pmid_u_grid)):
            v = heat_u[i, j]
            tc = "white" if v >= heat_u.max() * 0.55 else "#222222"
            ax_u.text(j, i, str(v), ha="center", va="center",
                      fontsize=8.5, color=tc, fontweight="bold")
    # 红星：当前选定阈值 (H=0.85, pmid=50) — 放到数字下方（cell 下半）
    try:
        i_star = H_u_grid.index(thr["H_universal_min"])
        j_star = pmid_u_grid.index(thr["pmid_universal_min"])
        star_y = i_star + 0.32   # 下移 32% cell 高
        # 白色光晕小星
        ax_u.plot(j_star, star_y, marker="*", markersize=18,
                  markerfacecolor="white", markeredgecolor="white",
                  markeredgewidth=0.5, zorder=4, alpha=0.95)
        ax_u.plot(j_star, star_y, marker="*", markersize=11,
                  markerfacecolor="#FF0000", markeredgecolor="#880000",
                  markeredgewidth=0.8, zorder=5)
    except ValueError:
        pass
    fig.colorbar(im_u, ax=ax_u, fraction=0.035, pad=0.02,
                 label="n_universal")

    # ── 右：Signature 3-panel ─────────────────────────────────
    gs_right = gs[0, 1].subgridspec(1, 3, wspace=0.15)
    axes_s = [fig.add_subplot(gs_right[0, k])
              for k in range(len(H_s_panels_display))]

    vmax_s = max(m.max() for m in heat_s.values())
    for k, h in enumerate(H_s_panels_display):
        mat = heat_s[h]
        im_s = axes_s[k].imshow(mat, cmap="YlOrRd", aspect="auto",
                                 vmin=0, vmax=vmax_s)
        axes_s[k].set_xticks(range(len(pmid_s_grid)))
        axes_s[k].set_xticklabels(pmid_s_grid, fontsize=9)
        axes_s[k].set_xlabel("pmid_sig", fontsize=10)
        if k == 0:
            axes_s[k].set_yticks(range(len(share_grid)))
            axes_s[k].set_yticklabels([f"{s:.2f}" for s in share_grid],
                                       fontsize=9)
            axes_s[k].set_ylabel("dominant_share_min", fontsize=11)
        else:
            axes_s[k].set_yticks(range(len(share_grid)))
            axes_s[k].set_yticklabels([])
        axes_s[k].set_title(f"H_sig ≤ {h:.2f}", fontsize=10,
                             fontweight="bold")

        for i in range(len(share_grid)):
            for j in range(len(pmid_s_grid)):
                v = mat[i, j]
                tc = "white" if v >= vmax_s * 0.55 else "#222222"
                axes_s[k].text(j, i, str(v), ha="center", va="center",
                               fontsize=8.5, color=tc, fontweight="bold")
        # 红星：当前 Signature 阈值 (H_sig=0.40, share=0.70, pmid=5) — 数字下方
        if abs(h - thr["H_signature_max"]) < 1e-6:
            try:
                i_star = share_grid.index(thr["dominant_share_min"])
                j_star = pmid_s_grid.index(thr["pmid_signature_min"])
                star_y = i_star + 0.32
                axes_s[k].plot(j_star, star_y, marker="*", markersize=18,
                               markerfacecolor="white", markeredgecolor="white",
                               markeredgewidth=0.5, zorder=4, alpha=0.95)
                axes_s[k].plot(j_star, star_y, marker="*", markersize=11,
                               markerfacecolor="#FF0000",
                               markeredgecolor="#880000",
                               markeredgewidth=0.8, zorder=5)
            except ValueError:
                pass

    # shared colorbar for panel b
    cb_s = fig.colorbar(im_s, ax=axes_s, fraction=0.025, pad=0.02,
                        label="n_signature")
    # title at top of the 3-panel strip — 上移留白
    fig.text(
        0.74, 0.93,
        "(b) Signature count sensitivity  (fixed H_univ=0.85, pmid_univ=50)",
        ha="center", fontsize=11.5, fontweight="bold",
    )

    fig.suptitle(
        "Threshold calibration sensitivity (1,620 grid combinations)",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.02,
        "Figure E. Threshold sensitivity over 1,620 grid combinations. "
        "Red stars mark the chosen thresholds (Universal: H>0.85 & PMID>50; "
        "Signature: H<0.40, share>0.70, PMID>5). The stars sit in locally "
        "stable regions of the heatmaps.",
        ha="center", fontsize=8.5, style="italic", color="#555555",
    )

    png, svg = save_png_svg(fig, "viz_final_threshold_calibration")
    print(f"  → {png}")
    print(f"  → {svg}")


if __name__ == "__main__":
    main()
