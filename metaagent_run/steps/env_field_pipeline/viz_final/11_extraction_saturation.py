"""Figure 11 — Phase 7 extraction saturation curves.

Two cumulative curves over rounds:
  Y1 (filled circles, solid line):   targets_tried   — exposure coverage
  Y2 (hollow circles, dashed line):  targets_succeeded — extractable coverage

X axis is cumulative paper count, with round numbers annotated above each
data point. Stop criterion (briefing §7.2): two consecutive rounds where
both relative deltas drop below 5% — drawn as a horizontal reference band
in the lower-right inset showing per-round Δ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_io import DATA_DIR  # noqa: E402
from viz_palette import install_style, save_png_svg  # noqa: E402

COVERAGE_PATH = DATA_DIR / "phase7_validation" / "coverage_per_round.csv"
DELTA_THRESHOLD = 0.05


def main() -> None:
    install_style()

    df = pd.read_csv(COVERAGE_PATH)
    if df.empty:
        print(f"No data at {COVERAGE_PATH}", file=sys.stderr)
        sys.exit(1)

    fig, (ax_main, ax_delta) = plt.subplots(
        2, 1, figsize=(8.5, 6.5), gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # ── Top: cumulative coverage curves ──────────────────────────────
    ax_main.plot(df["cumulative_papers"], df["targets_tried"],
                 marker="o", color="#1D3557", linewidth=2, markersize=8,
                 label="targets tried (≥1 step2 key match)")
    ax_main.plot(df["cumulative_papers"], df["targets_succeeded"],
                 marker="o", mfc="white", color="#E63946", linewidth=2,
                 markersize=8, linestyle="--",
                 label="targets succeeded (≥1 extracted value)")

    for _, row in df.iterrows():
        ax_main.annotate(f"R{row['round']}",
                         (row["cumulative_papers"], row["targets_tried"]),
                         textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=9, color="#1D3557")

    ax_main.set_ylabel("cumulative unique targets")
    ax_main.set_title("Phase 7 — extraction saturation across rounds")
    ax_main.legend(loc="lower right")
    ax_main.grid(True, alpha=0.3)

    # ── Bottom: per-round relative deltas + threshold band ───────────
    rel_d_tried = df["delta_targets_tried"] / df["targets_tried"].clip(lower=1)
    rel_d_succ = df["delta_targets_succeeded"] / df["targets_succeeded"].clip(lower=1)

    ax_delta.plot(df["cumulative_papers"], rel_d_tried * 100,
                  marker="o", color="#1D3557", linewidth=1.5,
                  label="Δ tried (% of total)")
    ax_delta.plot(df["cumulative_papers"], rel_d_succ * 100,
                  marker="o", mfc="white", color="#E63946", linewidth=1.5,
                  linestyle="--", label="Δ succ (% of total)")
    ax_delta.axhline(DELTA_THRESHOLD * 100, color="#999", linestyle=":",
                     linewidth=1)
    ax_delta.text(df["cumulative_papers"].iloc[-1], DELTA_THRESHOLD * 100,
                  f"  stop threshold ({DELTA_THRESHOLD:.0%})",
                  va="center", ha="left", fontsize=9, color="#666")
    ax_delta.set_xlabel("cumulative papers")
    ax_delta.set_ylabel("Δ per round (%)")
    ax_delta.legend(loc="upper right", fontsize=9)
    ax_delta.grid(True, alpha=0.3)
    ax_delta.set_ylim(bottom=0)

    fig.tight_layout()
    png, svg = save_png_svg(fig, "11_extraction_saturation")
    print(f"Wrote {png}")
    print(f"Wrote {svg}")


if __name__ == "__main__":
    main()
