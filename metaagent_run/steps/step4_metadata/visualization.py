"""
Step 4 Visualization — environment-characteristic metadata field analysis.

Produces 4 PDF figures:
  Fig A  env_field_frequency.pdf   Absolute PMID ranking + data funnel
  Fig B  env_field_heatmap.pdf     Universal MIxS backbone heatmap
  Fig C  env_field_bars.pdf        Per-env Signature (unequal panels)
  Fig D  synonym_fanout.pdf        Synonym fan-out for MIxS-mapped fields
"""

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

# ── visual constants ─────────────────────────────────────────────
ENVS = ["Open_ocean", "Coastal_waters", "Lake", "Wetlands"]
ENV_LABELS = {"Open_ocean": "Open Ocean", "Coastal_waters": "Coastal Waters",
              "Lake": "Lake",   "Wetlands": "Wetlands"}
ENV_COLORS = {"Open_ocean": "#2166ac", "Coastal_waters": "#4dac26",
              "Lake": "#b2182b",  "Wetlands": "#ff7f00"}

CAT_COLORS = {
    "Universal": "#6baed6",
    "Shared":    "#74c476",
    "Signature": "#e31a1c",
}

MAPPED_COLOR   = "#4393c3"
UNMAPPED_COLOR = "#f4a582"

FISHER_ALPHA = 0.05


def _fmt_label(entry: Dict[str, Any]) -> str:
    return entry["label"] + ("  \u2020" if not entry["mapped"] else "")


# ═════════════════════════════════════════════════════════════════
# Fig A: env_field_frequency.pdf — Absolute PMID ranking + funnel inset
# ═════════════════════════════════════════════════════════════════
def plot_env_field_frequency(
    all_entries: List[Dict[str, Any]],
    output_path: Path,
    total_papers: int,
    funnel_counts: Optional[Dict[str, int]] = None,
    min_pmid: int = 3,
    top_n: int = 40,
) -> None:
    """Ranked-by-absolute-count field chart with a data funnel on top.

    Design:
      * Primary visual = absolute number of papers (log-scale x axis).
      * No percentages — avoids denominator confusion entirely.
      * Data funnel sits as its own panel ABOVE the main chart so it
        cannot overlap the field-label annotations.
    """
    show = sorted(all_entries, key=lambda x: -x["total_pmid"])[:top_n]
    if not show:
        return

    labels = [_fmt_label(e) for e in show]
    pmids  = [e["total_pmid"] for e in show]
    colors = [MAPPED_COLOR if e["mapped"] else UNMAPPED_COLOR for e in show]

    # Layout: two stacked panels — top = funnel (small), bottom = main chart
    main_h = max(9.5, len(show) * 0.28)
    if funnel_counts:
        fig = plt.figure(figsize=(12, main_h + 1.8))
        gs = fig.add_gridspec(2, 1, height_ratios=[0.14, 1.0], hspace=0.18)
        ax_funnel = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[1, 0])
        _draw_funnel(ax_funnel, funnel_counts, base_total=total_papers)
    else:
        fig, ax = plt.subplots(figsize=(12, main_h))

    y_pos = list(range(len(labels)))
    ax.barh(y_pos, pmids, color=colors, edgecolor="gray", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()

    ax.set_xscale("log")
    xmin = max(1, min(pmids) * 0.5)
    xmax = max(pmids) * 5   # room for labels on the right
    ax.set_xlim(xmin, xmax)

    # Absolute counts only — no percentages
    for i, n in enumerate(pmids):
        ax.text(n * 1.07, i, "{:,}".format(n),
                va="center", fontsize=9, color="#111", fontweight="bold")

    ax.set_xlabel("Papers reporting the field  (log scale)", fontsize=10)
    ax.set_title(
        "Metadata Field Frequency — ranked by absolute publication count\n"
        "Analysis base: {:,} hydrosphere papers with \u22651 extracted metadata field"
        .format(total_papers),
        fontsize=11, fontweight="bold", loc="left", pad=8,
    )
    legend_elements = [
        Patch(facecolor=MAPPED_COLOR, edgecolor="gray", label="MIxS-aligned"),
        Patch(facecolor=UNMAPPED_COLOR, edgecolor="gray", label="Non-MIxS (\u2020)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="x", which="both", alpha=0.25, linewidth=0.4)

    fig.savefig(str(output_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    LOGGER.info("Saved %s", output_path.name)


def _draw_funnel(ax, counts: Dict[str, int], base_total: int) -> None:
    """Horizontal funnel as an independent top panel — no overlap with main."""
    steps = [
        ("Step 1 \u00b7 hydrosphere-classified",
            counts.get("classified", 0), "#9ecae1"),
        ("Step 1b \u00b7 single sub-environment",
            counts.get("single_env", 0), "#4292c6"),
        ("Step 2 \u00b7 \u22651 metadata extracted  \u2190 analysis base",
            counts.get("metadata_bearing", base_total), "#084594"),
    ]
    max_n = max(n for _, n, _ in steps) or 1
    y_pos = list(range(len(steps)))

    # bars drawn with left-anchor = 0; label on left (reverse-space), count on right
    for i, (label, n, color) in enumerate(steps):
        w = n / max_n
        ax.barh(i, w, color=color, height=0.62, edgecolor="#444", linewidth=0.5,
                left=0)
        # label to the left of bar (in reserved negative x-space)
        ax.text(-0.01, i, label, ha="right", va="center",
                fontsize=8.5, color="#222")
        # count immediately right of bar
        ax.text(w + 0.01, i, "n = {:,}".format(n),
                ha="left", va="center",
                fontsize=8.5, color="#111", fontweight="bold")

    ax.set_yticks([]); ax.set_xticks([])
    ax.set_xlim(-0.55, 1.25)
    ax.set_ylim(-0.5, len(steps) - 0.5)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        "Data funnel \u2014 corpus attrition from Step 1 classification to the analysis base",
        fontsize=9.5, fontweight="bold", loc="left", pad=3,
    )


# ═════════════════════════════════════════════════════════════════
# Fig B: env_field_heatmap.pdf — Universal MIxS backbone only
# ═════════════════════════════════════════════════════════════════
def plot_env_field_heatmap(
    included: List[Dict[str, Any]],
    output_path: Path,
    env_n: Dict[str, int],
    top_n: int = 20,  # now enforced — show only the most-reported backbone fields
) -> None:
    """Universal-backbone-only heatmap (MIxS-mapped, \u22653 envs), Top N.

    Design choice: a heatmap only works when it tells a SINGLE story.
    Previous version stuffed all 51 Universal fields into one panel; the
    bottom ~30 rows had pct 0-3% across the board, becoming a flat
    pale-yellow noise floor with no cross-environment discrimination.
    Here we cap at the top-N most-reported backbone fields (by total
    PMID), which is where meaningful environment-to-environment variation
    actually lives. Remainder available in tiered_extraction_targets.json.
    """
    backbone = [
        e for e in included
        if e["category"] == "Universal" and e.get("mapped")
    ]
    backbone.sort(key=lambda x: -x["total_pmid"])
    if not backbone:
        return

    n_total = len(backbone)
    backbone = backbone[:top_n]
    n_fields = len(backbone)
    pct_matrix = np.zeros((n_fields, len(ENVS)))
    for i, e in enumerate(backbone):
        for j, env in enumerate(ENVS):
            pct_matrix[i, j] = e[env + "_pct"]

    fig, ax = plt.subplots(figsize=(9, max(6, n_fields * 0.32)))
    vmax = max(60.0, float(pct_matrix.max()) * 1.05)
    im = ax.imshow(pct_matrix, cmap="YlGnBu", vmin=0, vmax=vmax, aspect="auto")

    # X (envs)
    ax.set_xticks(range(len(ENVS)))
    xlabels = ["{}\n(n={:,})".format(ENV_LABELS[e], env_n[e]) for e in ENVS]
    ax.set_xticklabels(xlabels, fontsize=9, fontweight="bold")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Y (fields)
    ax.set_yticks(range(n_fields))
    ax.set_yticklabels([e["label"] for e in backbone], fontsize=8)

    # Cell text: env_pct%
    for i in range(n_fields):
        for j in range(len(ENVS)):
            pct = pct_matrix[i, j]
            text_color = "white" if pct > vmax * 0.55 else "#222"
            ax.text(j, i, "{:.1f}%".format(pct), ha="center", va="center",
                    fontsize=7.5, color=text_color, fontweight="bold")

    cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cb.set_label("% of environment papers reporting the field", fontsize=9)

    title_extra = ""
    if n_total > top_n:
        title_extra = " | showing top {} of {} (by total PMID)".format(top_n, n_total)
    ax.set_title(
        "Hydrosphere Universal Metadata Backbone — Top {} most-reported MIxS fields\n"
        "Fields present (\u22650.5% of env papers) in \u22653 of 4 sub-environments"
        "{}   |   cell = % of env papers".format(n_fields, title_extra),
        fontsize=10, fontweight="bold", pad=16,
    )

    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    LOGGER.info("Saved %s", output_path.name)


# ═════════════════════════════════════════════════════════════════
# Fig C: env_field_bars.pdf — Per-env Signature only (unequal panels)
# ═════════════════════════════════════════════════════════════════
def plot_env_field_bars(
    included: List[Dict[str, Any]],
    output_path: Path,
    env_n: Dict[str, int],
    min_pmid: int = 5,        # kept for compat, unused in new design
    top_n_per_env: int = 15,  # panel cap (only applied if signature count exceeds)
) -> None:
    """Per-environment Signature-field panels — intentionally UNEQUAL.

    Key design change from previous version:
      * No \"extend with Shared/Universal\" backfill — if an environment
        has 2 signature fields, the panel shows 2 bars. Panel length is
        itself scientific information (\"this environment is less
        research-specific\").
      * No mixing with Universal/Shared — those have their own figure
        (Fig B). Avoids the earlier problem where Open-Ocean's panel
        looked similar to Lake's simply because both showed the same
        universal fields.
      * Panels use the environment's signature COLOR so each panel is
        visually distinct at a glance.
    """
    # Panel membership rule: a field belongs to env E's panel iff E is
    # in the field's `sig_envs` (strict: Fisher-significant AND enrichment
    # ≥ threshold used by classify_env_fields). This prevents fields that
    # merely have non-zero presence in E (e.g. Lake's conductivity present
    # in Open_ocean at enr=0.2) from cluttering the wrong env's panel.
    #
    # For Signatures classified by presence-count (no sig_envs), fall back
    # to the field's single envs_present entry so they still appear somewhere.
    sigs = [e for e in included if e["category"] == "Signature"]
    env_sigs = {env: [] for env in ENVS}
    for e in sigs:
        primary_envs = e.get("sig_envs") or e.get("envs_present", [])
        for env in primary_envs:
            if env in env_sigs:
                env_sigs[env].append(e)

    # Consistent x limit for cross-panel comparison
    all_pcts = []
    for env in ENVS:
        for e in env_sigs[env]:
            all_pcts.append(e[env + "_pct"])
    global_xmax = (max(all_pcts) * 1.45) if all_pcts else 5.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for idx, env in enumerate(ENVS):
        ax = axes[idx]
        fields_all = sorted(env_sigs[env],
                            key=lambda x: -x["enrichment"].get(env, 1.0))
        n_total = len(fields_all)
        fields = fields_all[:top_n_per_env]

        if not fields:
            ax.text(0.5, 0.5, "(no signature fields)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#888", style="italic")
            ax.set_title(
                "{}  (n={:,} papers)\n0 signature fields".format(
                    ENV_LABELS[env], env_n[env]),
                fontsize=11, fontweight="bold", color=ENV_COLORS[env],
            )
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ["top", "right", "bottom", "left"]:
                ax.spines[spine].set_visible(False)
            continue

        labels  = [_fmt_label(e) for e in fields]
        pcts    = [e[env + "_pct"] for e in fields]
        hatches = ["//" if not e["mapped"] else "" for e in fields]

        y_pos = list(range(len(labels)))
        bars = ax.barh(y_pos, pcts, color=ENV_COLORS[env],
                       edgecolor="gray", linewidth=0.4, alpha=0.88)
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)

        for i, e in enumerate(fields):
            p = e[env + "_pct"]
            n = e[env + "_n"]
            enr = e["enrichment"].get(env, 1.0)
            fp  = e["fisher_p"].get(env, 1.0)
            sig = "*" if (fp < FISHER_ALPHA and enr > 1.0) else ""
            txt = "{:.1f}%  n={}  \u00d7{:.1f}{}".format(p, n, enr, sig)
            ax.text(p + global_xmax * 0.012, i, txt,
                    va="center", fontsize=7.5, color="#333")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("% of env papers reporting the field", fontsize=9)
        ax.set_xlim(0, global_xmax)
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)

        suffix = ""
        if n_total > top_n_per_env:
            suffix = "  (top {} of {})".format(top_n_per_env, n_total)
        else:
            suffix = "  ({} fields)".format(n_total)
        ax.set_title(
            "{}  (n={:,} papers){}".format(ENV_LABELS[env], env_n[env], suffix),
            fontsize=11, fontweight="bold", color=ENV_COLORS[env],
        )

    legend_elements = [
        Patch(facecolor="#888", edgecolor="gray", label="Signature-field bar (env color)"),
        Patch(facecolor="white", edgecolor="gray", hatch="//",
              label="Non-MIxS field (\u2020)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor="gray")

    fig.suptitle(
        "Environment Signature Fields — fields strongly enriched (Fisher p<0.05, "
        "enrichment \u22652) in this environment\n"
        "Ranked by fold-enrichment within the sub-environment  |  "
        "\"n=\" = env paper count reporting the field  |  \"*\" = Fisher p<0.05\n"
        "Panel lengths are deliberately UNEQUAL — they reflect real cross-environment"
        " differences in research specificity.",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    LOGGER.info("Saved %s", output_path.name)


# ═════════════════════════════════════════════════════════════════
# Fig D: synonym_fanout.pdf — unchanged
# ═════════════════════════════════════════════════════════════════
def plot_synonym_fanout(
    tiered: Dict[str, Any],
    output_path: Path,
    min_fanout: int = 2,
    top_n: int = 40,
) -> None:
    """Synonym fan-out chart for MIxS-mapped fields."""
    items = []
    for t in tiered.get("tier1", []):
        n_raw = t.get("raw_field_count", 0)
        if n_raw < min_fanout:
            continue
        items.append(dict(
            label=t["mixs_title"],
            n_raw=n_raw,
            contributing=t.get("contributing_fields", []),
        ))

    items.sort(key=lambda x: -x["n_raw"])
    items = items[:top_n]
    if not items:
        return

    fig, ax = plt.subplots(figsize=(10, max(8, len(items) * 0.35)))
    y_pos = range(len(items))
    ax.barh(y_pos, [f["n_raw"] for f in items],
            color="#7fcdbb", edgecolor="gray", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f["label"] for f in items], fontsize=8)
    ax.invert_yaxis()

    for i, f in enumerate(items):
        names = ", ".join(f["contributing"][:5])
        if len(f["contributing"]) > 5:
            names += ", \u2026"
        ax.text(f["n_raw"] + 0.2, i,
                "{} synonyms   [{}]".format(f["n_raw"], names),
                va="center", fontsize=6, color="#555555")

    ax.set_xlabel("Number of raw field name variants", fontsize=10)
    ax.set_title(
        "Synonym Fan-out: Raw Field Names \u2192 Standardised MIxS Terms\n"
        "(Multiple naming conventions in literature collapse into one standard term)",
        fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    LOGGER.info("Saved %s", output_path.name)
