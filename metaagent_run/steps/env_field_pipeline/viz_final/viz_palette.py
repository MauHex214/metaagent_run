"""viz_final 共享配色 / 字体 / 输出约定。"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

# ── 色盘 ────────────────────────────────────────────────────────────
FAMILY_COLORS: dict[str, str] = {
    "A_physicochemical": "#E63946",   # 红
    "B_env_categorical": "#2A9D8F",   # 青绿
    "C_spatiotemporal": "#1D3557",    # 深蓝
    "D_other": "#6C757D",             # 灰
}

ENV_COLORS: dict[str, str] = {
    "Open_ocean": "#0D3B66",      # 深海蓝
    "Coastal_waters": "#2A9D8F",  # 青绿
    "Lake": "#5DADE2",            # 浅湖蓝
    "Wetlands": "#8B5A2B",        # 棕褐
}

CATEGORY_COLORS: dict[str, str] = {
    "Universal": "#264653",
    "Cross-biome common": "#2A9D8F",
    "Signature": "#E9C46A",
    "Niche": "#BDBDBD",
}

MIXS_ALIGN_COLORS: dict[str, str] = {
    "exact": "#1B7F3A",
    "subset": "#43A047",
    "superset": "#FB8C00",
    "partial": "#FDD835",
    "UNMAPPED": "#9E9E9E",
}


# ── 字体 / 尺寸 ────────────────────────────────────────────────────
def install_style() -> None:
    """每个脚本开头调用，统一字体和字号。"""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",  # Arial 的开源近亲，Linux 自带
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 15,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,   # embed fonts (editable in Illustrator)
        "svg.fonttype": "none",
    })


# ── 输出路径 ────────────────────────────────────────────────────────
OUTPUT_DIR: Path = Path(__file__).resolve().parents[4] / "env_field_pipeline_output" / "viz_final"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_png_svg(fig, stem: str) -> tuple[Path, Path]:
    """保存 PNG（300 DPI） + SVG（矢量）。返回两个路径。"""
    ensure_output_dir()
    png = OUTPUT_DIR / f"{stem}.png"
    svg = OUTPUT_DIR / f"{stem}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    return png, svg
