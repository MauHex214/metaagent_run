"""Step7 Value 归一化器。

策略（design_summary §5.3.1 方案 b）：
  - 只归一化单位名称，不跨量纲/跨化学换算
  - 保留原文单位维度
  - preferred_unit 在 CDE 内是允许列表（多单位 MIxS 字段）

13 种 normalizer，按 MIxS Value syntax 模式分发：
  passthrough, float_with_unit, float_only, integer, boolean, timestamp,
  lat_lon, ontology_term, range, enum, duration, compound_text_measurement, ratio
"""

import logging
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .schemas import NormalizedValue

LOGGER = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Unit synonym table（水圈领域常见单位）
# ═══════════════════════════════════════════════════════════

UNIT_SYNONYMS: Dict[str, str] = {
    # 长度
    "m": "meter", "meter": "meter", "meters": "meter",
    "metre": "meter", "metres": "meter",
    "km": "kilometer", "kilometer": "kilometer", "kilometers": "kilometer",
    "cm": "centimeter", "centimeter": "centimeter", "centimeters": "centimeter",
    "mm": "millimeter", "millimeter": "millimeter", "millimeters": "millimeter",
    "um": "micrometer", "µm": "micrometer", "μm": "micrometer",
    "micrometer": "micrometer", "micrometers": "micrometer",
    "micron": "micrometer", "microns": "micrometer",
    "nm": "nanometer", "nanometer": "nanometer", "nanometers": "nanometer",
    # 温度
    "c": "degC", "°c": "degC", "degc": "degC",
    "degree celsius": "degC", "degrees celsius": "degC", "celsius": "degC",
    "k": "K", "kelvin": "K",
    "f": "degF", "°f": "degF", "degf": "degF",
    "fahrenheit": "degF", "degree fahrenheit": "degF",
    # 浓度（质量/体积）
    "mg/l": "mg/L", "milligram per liter": "mg/L", "mg per l": "mg/L",
    "ug/l": "ug/L", "µg/l": "ug/L", "μg/l": "ug/L",
    "microgram per liter": "ug/L",
    "ng/l": "ng/L", "nanogram per liter": "ng/L",
    # 浓度（摩尔/体积）
    "um_conc": "uM",
    "umol/l": "uM", "µmol/l": "uM", "μmol/l": "uM",
    "micromole per liter": "uM",
    "umol/kg": "umol/kg", "µmol/kg": "umol/kg", "μmol/kg": "umol/kg",
    "micromole per kilogram": "umol/kg",
    "mmol/l": "mM", "millimole per liter": "mM",
    "mol/l": "M", "mole per liter": "M",
    "ppm": "ppm", "parts per million": "ppm",
    "ppb": "ppb", "parts per billion": "ppb",
    # 比例
    "%": "percent", "percent": "percent", "percentage": "percent",
    # 盐度
    "psu": "PSU", "practical salinity unit": "PSU",
    "pss": "PSU", "pss-78": "PSU", "pss78": "PSU",
    # pH
    "ph unit": "pH", "ph": "pH", "ph_unit": "pH",
    # 时间
    "s": "second", "sec": "second", "second": "second", "seconds": "second",
    "min": "minute", "minute": "minute", "minutes": "minute",
    "h": "hour", "hr": "hour", "hour": "hour", "hours": "hour",
    "d": "day", "day": "day", "days": "day",
    # 质量
    "g": "gram", "gram": "gram", "grams": "gram",
    "mg": "milligram", "milligram": "milligram", "milligrams": "milligram",
    "kg": "kilogram", "kilogram": "kilogram", "kilograms": "kilogram",
    "ton": "ton", "tons": "ton", "tonne": "ton", "tonnes": "ton",
    # 体积
    "l": "liter", "liter": "liter", "litre": "liter", "litres": "liter",
    "ml": "milliliter", "milliliter": "milliliter",
    "millilitre": "milliliter", "millilitres": "milliliter",
    # 压力
    "bar": "bar", "dbar": "dbar",
    "pa": "Pa", "kpa": "kPa", "mpa": "MPa",
    "atm": "atm", "atmosphere": "atm",
    # 密度
    "g/cm3": "g/cm3", "g/cm^3": "g/cm3",
    "kg/m3": "kg/m3",
    "g/ml": "g/mL", "g/l": "g/L",
    # 速度 / 流量
    "m/s": "m/s", "cm/s": "cm/s", "knot": "knot", "knots": "knot",
    # 光强 / PAR
    "uein/m2/s": "umol photons/m2/s",
    "umol/m2/s": "umol photons/m2/s",
    "µmol/m²/s": "umol photons/m2/s",
    "μmol/m²/s": "umol photons/m2/s",
    # 立方米/天 等环境衍生
    "g/m3": "g/m3", "mg/m3": "mg/m3",
    # 频率/计数
    "cells/ml": "cells/mL", "cells/l": "cells/L",
    "copies/ml": "copies/mL", "copies/l": "copies/L",
}


def canonicalize_unit(raw: str) -> str:
    """Return canonical unit name; if unknown, return lowercased & trimmed raw."""
    if raw is None:
        return ""
    key = raw.strip().lower()
    if not key:
        return ""
    return UNIT_SYNONYMS.get(key, key)


# ═══════════════════════════════════════════════════════════
#  Normalizer 实现
# ═══════════════════════════════════════════════════════════

def _passthrough(value_raw: str, cde_entry: dict, envo_index: dict) -> NormalizedValue:
    return NormalizedValue(
        value_normalized=value_raw.strip() if value_raw else "",
        value_type="text",
        normalize_status="ok",
    )


_NUMBER_UNIT_RE = re.compile(
    r"^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*(.*?)\s*$"
)


def _float_with_unit(value_raw: str, cde_entry: dict, envo_index: dict) -> NormalizedValue:
    if not value_raw:
        return NormalizedValue(value_type="measurement",
                               normalize_status="failed",
                               normalize_error="empty value")
    m = _NUMBER_UNIT_RE.match(value_raw)
    if not m:
        return NormalizedValue(value_type="measurement",
                               normalize_status="failed",
                               normalize_error="no leading number")
    try:
        number = float(m.group(1))
    except ValueError:
        return NormalizedValue(value_type="measurement",
                               normalize_status="failed",
                               normalize_error="bad number: %s" % m.group(1))
    raw_unit = m.group(2).strip()
    canonical_unit = canonicalize_unit(raw_unit)
    preferred_units = cde_entry.get("preferred_unit", []) or []
    preferred_canonical = [canonicalize_unit(u) for u in preferred_units]
    # 状态判定
    if not preferred_canonical:
        status = "ok"
    elif canonical_unit and canonical_unit in preferred_canonical:
        status = "ok"
    elif not canonical_unit:
        status = "failed"  # CDE 期望有单位但原文没有
        return NormalizedValue(
            value_normalized=number, unit="", value_type="measurement",
            normalize_status=status,
            normalize_error="unit missing in raw value; CDE expects %s" % preferred_canonical,
        )
    else:
        status = "out_of_preferred"
    return NormalizedValue(
        value_normalized=number,
        unit=canonical_unit,
        value_type="measurement",
        normalize_status=status,
    )


def _float_only(value_raw: str, cde_entry: dict, envo_index: dict) -> NormalizedValue:
    try:
        return NormalizedValue(
            value_normalized=float(value_raw.strip()),
            value_type="number",
            normalize_status="ok",
        )
    except (ValueError, AttributeError) as e:
        return NormalizedValue(value_type="number", normalize_status="failed",
                               normalize_error=str(e))


def _integer(value_raw: str, cde_entry: dict, envo_index: dict) -> NormalizedValue:
    try:
        # 容忍 "10.0" 这种写法 → 转 int
        return NormalizedValue(
            value_normalized=int(float(value_raw.strip())),
            value_type="integer",
            normalize_status="ok",
        )
    except (ValueError, AttributeError) as e:
        return NormalizedValue(value_type="integer", normalize_status="failed",
                               normalize_error=str(e))


def _boolean(value_raw: str, cde_entry: dict, envo_index: dict) -> NormalizedValue:
    s = (value_raw or "").strip().lower()
    if s in ("true", "yes", "1", "y", "t"):
        return NormalizedValue(value_normalized=True, value_type="boolean", normalize_status="ok")
    if s in ("false", "no", "0", "n", "f"):
        return NormalizedValue(value_normalized=False, value_type="boolean", normalize_status="ok")
    return NormalizedValue(value_type="boolean", normalize_status="failed",
                           normalize_error="not a boolean: %s" % value_raw)


def _timestamp(value_raw: str, cde_entry: dict, envo_index: dict) -> NormalizedValue:
    s = (value_raw or "").strip()
    if not s:
        return NormalizedValue(value_type="datetime", normalize_status="failed",
                               normalize_error="empty")
    # 简易日期解析：尝试常见格式（避免引入 dateutil 依赖）
    formats = [
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y",
        "%Y-%m", "%Y/%m", "%Y",
        "%b %Y", "%B %Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            return NormalizedValue(
                value_normalized=dt.isoformat(),
                value_type="datetime",
                normalize_status="ok",
            )
        except ValueError:
            continue
    return NormalizedValue(
        value_type="datetime",
        normalize_status="failed",
        normalize_error="unrecognized date format: %s" % s,
    )


_LAT_LON_DECIMAL_RE = re.compile(
    r"^\s*([+-]?\d+\.?\d*)\s*[,/\s]\s*([+-]?\d+\.?\d*)\s*$"
)
_LAT_LON_NSEW_RE = re.compile(
    r"^\s*(\d+\.?\d*)\s*°?\s*([NS])\s*[,/\s]+\s*(\d+\.?\d*)\s*°?\s*([EW])\s*$",
    re.IGNORECASE,
)


def _lat_lon(value_raw: str, cde_entry: dict, envo_index: dict) -> NormalizedValue:
    s = (value_raw or "").strip()
    # decimal pair
    m = _LAT_LON_DECIMAL_RE.match(s)
    if m:
        return NormalizedValue(
            value_normalized={"lat": float(m.group(1)), "lon": float(m.group(2))},
            value_type="coordinate", normalize_status="ok",
        )
    # NSEW format
    m = _LAT_LON_NSEW_RE.match(s)
    if m:
        lat = float(m.group(1)) * (-1 if m.group(2).upper() == "S" else 1)
        lon = float(m.group(3)) * (-1 if m.group(4).upper() == "W" else 1)
        return NormalizedValue(
            value_normalized={"lat": lat, "lon": lon},
            value_type="coordinate", normalize_status="ok",
        )
    return NormalizedValue(value_type="coordinate", normalize_status="failed",
                           normalize_error="unrecognized lat/lon: %s" % s)


def _ontology_term(value_raw: str, cde_entry: dict, envo_index: dict) -> NormalizedValue:
    """精确 + 模糊匹配；LLM 兜底暂不启用。"""
    s = (value_raw or "").strip().strip(".,;").lower()
    if not s:
        return NormalizedValue(value_type="ontology_term", normalize_status="failed",
                               normalize_error="empty")
    # 阶段 1：精确匹配（envo_index 是 name_lower → {id, name}）
    entry = envo_index.get(s)
    if entry:
        return NormalizedValue(
            value_normalized={"label": entry.get("name", ""), "id": entry.get("id", "")},
            value_type="ontology_term", normalize_status="ok",
        )
    # 阶段 2：token-overlap 模糊匹配（轻量版）
    s_tokens = set(re.split(r"[^a-z0-9]+", s))
    s_tokens.discard("")
    if not s_tokens:
        return NormalizedValue(
            value_normalized={"label": value_raw, "id": ""},
            value_type="ontology_term", normalize_status="no_ontology_match",
        )
    best_id, best_name, best_score = "", "", 0.0
    for name_lower, ent in envo_index.items():
        n_tokens = set(re.split(r"[^a-z0-9]+", name_lower))
        n_tokens.discard("")
        if not n_tokens:
            continue
        overlap = len(s_tokens & n_tokens)
        denom = max(len(s_tokens), len(n_tokens))
        score = overlap / denom
        if score > best_score and score >= 0.7:
            best_score = score
            best_id = ent.get("id", "")
            best_name = ent.get("name", "")
    if best_id:
        return NormalizedValue(
            value_normalized={"label": best_name, "id": best_id},
            value_type="ontology_term", normalize_status="ok",
        )
    return NormalizedValue(
        value_normalized={"label": value_raw, "id": ""},
        value_type="ontology_term", normalize_status="no_ontology_match",
    )


_RANGE_RE = re.compile(
    # unit 部分限制为非空格非逗号字符，避免吞掉后续逗号分隔的第二个 range
    r"^\s*([+-]?\d*\.?\d+)\s*(?:[-~–—]|to)\s*([+-]?\d*\.?\d+)\s*([^\s,]*)\s*"
)


def _range(value_raw: str, cde_entry: dict, envo_index: dict) -> NormalizedValue:
    s = (value_raw or "").strip()
    m = _RANGE_RE.match(s)
    if not m:
        return NormalizedValue(value_type="range", normalize_status="failed",
                               normalize_error="not a range: %s" % s)
    try:
        mn = float(m.group(1))
        mx = float(m.group(2))
    except ValueError:
        return NormalizedValue(value_type="range", normalize_status="failed",
                               normalize_error="bad range numbers")
    raw_unit = m.group(3).strip()
    canonical_unit = canonicalize_unit(raw_unit)
    # 检测原始值后是否还有逗号 + 内容（说明是多 range 形式，本次只取第一个）
    rest = s[m.end():].strip()
    status = "out_of_preferred" if rest.startswith(",") else "ok"
    err = "multi-range value, first range used: %r" % s if status == "out_of_preferred" else ""
    return NormalizedValue(
        value_normalized={"min": mn, "max": mx},
        unit=canonical_unit,
        value_type="range",
        normalize_status=status,
        normalize_error=err,
    )


_ENUM_OPTIONS_RE = re.compile(r"\[([^\]]+)\]")


def _enum(value_raw: str, cde_entry: dict, envo_index: dict) -> NormalizedValue:
    s = (value_raw or "").strip().lower()
    if not s:
        return NormalizedValue(value_type="enum", normalize_status="failed",
                               normalize_error="empty")
    syntax = cde_entry.get("value_syntax", "")
    m = _ENUM_OPTIONS_RE.search(syntax)
    if not m:
        # CDE 没指定枚举集 → 当文本接受
        return NormalizedValue(value_normalized=value_raw.strip(),
                               value_type="enum", normalize_status="ok")
    options = [o.strip().lower() for o in m.group(1).split("|")]
    if s in options:
        return NormalizedValue(value_normalized=s, value_type="enum",
                               normalize_status="ok")
    # 模糊：包含关系
    for opt in options:
        if opt in s or s in opt:
            return NormalizedValue(value_normalized=opt, value_type="enum",
                                   normalize_status="ok")
    return NormalizedValue(value_type="enum", normalize_status="failed",
                           normalize_error="not in [%s]" % "|".join(options))


def _duration(value_raw: str, cde_entry: dict, envo_index: dict) -> NormalizedValue:
    """简易 duration：'5 days', '24 hours' → ISO 8601 P5D / PT24H。"""
    s = (value_raw or "").strip().lower()
    m = re.match(r"^([+-]?\d+\.?\d*)\s*([a-z]+)?\s*$", s)
    if not m:
        return NormalizedValue(value_type="duration", normalize_status="failed",
                               normalize_error="unrecognized: %s" % s)
    try:
        n = float(m.group(1))
    except ValueError:
        return NormalizedValue(value_type="duration", normalize_status="failed",
                               normalize_error="bad number")
    unit = canonicalize_unit(m.group(2) or "")
    iso_map = {"second": "PT%fS", "minute": "PT%fM", "hour": "PT%fH",
               "day": "P%fD"}
    fmt = iso_map.get(unit)
    if not fmt:
        return NormalizedValue(value_type="duration", normalize_status="failed",
                               normalize_error="unsupported unit: %s" % unit)
    iso = (fmt % n).replace(".000000", "").replace(".0S", "S").replace(".0H", "H").replace(".0M", "M").replace(".0D", "D")
    return NormalizedValue(value_normalized=iso, value_type="duration",
                           normalize_status="ok")


def _compound_text_measurement(value_raw: str, cde_entry: dict, envo_index: dict) -> NormalizedValue:
    """{text};{float} {unit} 形式：分号切分，前半文本 + 后半 float_with_unit。"""
    s = (value_raw or "").strip()
    if ";" not in s:
        return NormalizedValue(value_normalized={"text": s},
                               value_type="compound", normalize_status="ok")
    text_part, _, num_part = s.partition(";")
    sub = _float_with_unit(num_part.strip(), cde_entry, envo_index)
    return NormalizedValue(
        value_normalized={
            "text": text_part.strip(),
            "value": sub.value_normalized,
            "unit": sub.unit,
        },
        value_type="compound",
        normalize_status=sub.normalize_status,
        normalize_error=sub.normalize_error,
    )


_RATIO_RE = re.compile(r"^\s*([+-]?\d*\.?\d+)\s*:\s*([+-]?\d*\.?\d+)\s*$")


def _ratio(value_raw: str, cde_entry: dict, envo_index: dict) -> NormalizedValue:
    s = (value_raw or "").strip()
    m = _RATIO_RE.match(s)
    if not m:
        return NormalizedValue(value_type="ratio", normalize_status="failed",
                               normalize_error="not a ratio: %s" % s)
    try:
        n1, n2 = float(m.group(1)), float(m.group(2))
    except ValueError:
        return NormalizedValue(value_type="ratio", normalize_status="failed",
                               normalize_error="bad numbers")
    return NormalizedValue(
        value_normalized={"n1": n1, "n2": n2},
        value_type="ratio", normalize_status="ok",
    )


# ═══════════════════════════════════════════════════════════
#  Dispatcher
# ═══════════════════════════════════════════════════════════

NORMALIZERS: Dict[str, Callable[[str, dict, dict], NormalizedValue]] = {
    "passthrough": _passthrough,
    "float_with_unit": _float_with_unit,
    "float_only": _float_only,
    "integer": _integer,
    "boolean": _boolean,
    "timestamp": _timestamp,
    "lat_lon": _lat_lon,
    "ontology_term": _ontology_term,
    "range": _range,
    "enum": _enum,
    "duration": _duration,
    "compound_text_measurement": _compound_text_measurement,
    "ratio": _ratio,
}


def normalize(
    value_raw: str,
    cde_entry: dict,
    envo_index: dict,
    fallback: str = "passthrough",
) -> NormalizedValue:
    """按 cde_entry['normalizer'] 分发；未知 normalizer 走 fallback。"""
    name = cde_entry.get("normalizer", fallback)
    fn = NORMALIZERS.get(name) or NORMALIZERS[fallback]
    return fn(value_raw or "", cde_entry, envo_index)
