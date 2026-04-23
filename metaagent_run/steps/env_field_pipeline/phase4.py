"""环节 4：三元组合并 + MIxS 对齐（基于新四槽位 schema）。

合并键：
    A 族: (subtype, quantity_kind, sorted(modifier_bag))
    B/C/D 族: (subtype, quantity_kind)   modifier_bag 永远为空，天然退化

流水线：
    1. 按合并键分桶
    2. 桶内字符串相似度聚类（Jaccard / 编辑距离 / 子串）
    3. EDC verify：桶内多分量 pair 经 LLM 判 merge / keep_separate
    4. 代表名选择、环境向量聚合
    5. MIxS 对齐：每个 canonical 独立 LLM 调用，选 89 水圈 slot 之一或 UNMAPPED

入口：
    4              完整流程
    4-rerun-mixs   只重跑 MIxS 对齐，不重建 canonical
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from metaagent_run.core import (
    AsyncLocalModelClient,
    backoff_with_jitter,
    extract_json_from_response_with_repair,
)

from . import config

logger = logging.getLogger(__name__)

# ── 常量 ────────────────────────────────────────────────────────────
CONCURRENCY = 24
MAX_RETRIES = 3
EVIDENCE_CAP = 2
EVIDENCE_CHAR_CAP = 400

# 字符串相似度阈值
JACCARD_THRESHOLD = 0.5
EDIT_DISTANCE_CAP = 2
EDIT_DISTANCE_RATIO = 0.3
SUBSTRING_LEN_RATIO = 0.5

# 字符串预处理：单位后缀剥离（谨慎；只剥末尾一次）
UNIT_SUFFIXES = (
    "_psu", "_ppt", "_ppm", "_ppb",
    "_mg_l", "_mgl", "_ug_l", "_ugl", "_ng_l", "_ngl",
    "_umol_l", "_umoll", "_mmol_l", "_mmoll", "_nmol_l", "_nmoll",
    "_umol_kg", "_umolkg", "_mg_kg", "_mgkg", "_ug_kg", "_ugkg",
    "_nmol_kg", "_nmolkg", "_g_kg", "_gkg",
    "_percent", "_pct", "_pc",
    "_m", "_cm", "_mm", "_km",
    "_degc", "_degk",
)
_TOKEN_SPLIT_RE = re.compile(r"[_]+|(?<=\D)(?=\d)|(?<=\d)(?=\D)")
_PAREN_RE = re.compile(r"\([^)]*\)")

PHASE4_EDC_CHECKPOINT: Path = config.OUTPUT_DIR / "env4_edc_verify.checkpoint.jsonl"
PHASE4_MIXS_CHECKPOINT: Path = config.OUTPUT_DIR / "env4_mixs_align.checkpoint.jsonl"

PHASE4_CANONICALS = config.PHASE4_CANONICALS                  # env4_canonicals.csv
PHASE4_MAPPING = config.PHASE4_MAPPING                         # env4_canonical_to_raw_key.csv
PHASE4_EDC_LOG = config.PHASE4_EDC_LOG                         # env4_edc_verify_log.csv
PHASE4_BUCKET_STATS = config.OUTPUT_DIR / "env4_bucket_stats.csv"
PHASE4_MIXS_LOG = config.OUTPUT_DIR / "env4_mixs_alignment_log.csv"
PHASE4_SINGLETONS = config.OUTPUT_DIR / "env4_singleton_canonicals.csv"

MIXS_SLOTS_FILE: Path = config.PROJECT_ROOT_DIR / "mixs_hydrosphere_slots.json"


# ── 字符串预处理 / 相似度 ──────────────────────────────────────────
def _preprocess(s: str) -> str:
    t = str(s or "").strip().lower()
    t = _PAREN_RE.sub("", t)
    # 剥常见单位后缀（只剥一次，尽量保守）
    for suf in UNIT_SUFFIXES:
        if t.endswith(suf) and len(t) > len(suf) + 2:  # 保留主干 ≥ 2 字符
            t = t[: -len(suf)]
            break
    t = re.sub(r"[_\s]+", "_", t).strip("_")
    return t


def _tokens(s: str) -> set[str]:
    return {t for t in _TOKEN_SPLIT_RE.split(s) if t}


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def _similar(a_raw: str, b_raw: str) -> bool:
    a, b = _preprocess(a_raw), _preprocess(b_raw)
    if a == b:
        return True
    ta, tb = _tokens(a), _tokens(b)
    if ta and tb:
        jac = len(ta & tb) / len(ta | tb)
        if jac >= JACCARD_THRESHOLD:
            return True
    short_len = min(len(a), len(b))
    cap = max(EDIT_DISTANCE_CAP, int(short_len * EDIT_DISTANCE_RATIO))
    if _levenshtein(a, b) <= cap:
        return True
    short, long = (a, b) if len(a) < len(b) else (b, a)
    if short and short in long and len(short) / len(long) >= SUBSTRING_LEN_RATIO:
        return True
    return False


# ── Union-Find ─────────────────────────────────────────────────────
class UF:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb

    def components(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for x in self.p:
            out[self.find(x)].append(x)
        return out


# ── Bucket key ─────────────────────────────────────────────────────
def _merge_key(row: pd.Series) -> tuple:
    bag_str = row.get("modifier_bag")
    if pd.notna(bag_str) and str(bag_str).strip():
        bag = tuple(sorted(str(bag_str).split("|")))
    else:
        bag = ()
    if row["family"] == "A_physicochemical":
        return (row["family"], row["subtype"], row["quantity_kind"], bag)
    # B/C/D 忽略 bag
    return (row["family"], row["subtype"], row["quantity_kind"], ())


# ── 代表名 v2：clarity > brevity ────────────────────────────────────
# 化学物质/环境概念全称白名单 — 命中则优先选为代表名
_FULL_NAME_KEYWORDS: frozenset[str] = frozenset([
    # 元素/金属全称
    "iron", "calcium", "sodium", "magnesium", "potassium",
    "copper", "zinc", "cadmium", "mercury", "lead", "aluminum",
    "chromium", "nickel", "manganese", "barium", "strontium",
    "arsenic", "selenium", "cobalt", "uranium", "silver",
    # 离子/化合物全称
    "chloride", "sulfate", "sulfide", "bromide", "fluoride",
    "phosphorus", "nitrogen", "carbon", "oxygen", "hydrogen", "argon",
    "ammonium", "ammonia", "nitrate", "nitrite", "phosphate",
    "silicate", "bicarbonate", "carbonate",
    # 水圈核心测量
    "chlorophyll", "alkalinity", "salinity", "temperature",
    "conductivity", "turbidity", "dissolved", "particulate",
    # 位置/时间
    "depth", "latitude", "longitude", "elevation", "altitude",
    "coordinates",  # 让 coordinates 胜过 latitude_longitude
    "collection", "sampling", "season", "date", "year",
    # 生境/分类
    "habitat", "sediment", "substrate", "vegetation", "biome",
    # 完整描述词
    "mixed_layer", "water_table", "apparent_oxygen", "potential_temperature",
    "soil_moisture", "kjeldahl", "organic_matter", "suspended",
    # 限定词
    "total", "inorganic", "organic",
])

# 通用学术缩写：即使 <4 字符也保留为代表名（不触发 too_short 惩罚）
_COMMON_ACRONYM_WHITELIST: frozenset[str] = frozenset([
    "ph",     # 酸碱度
    "bod",    # biological oxygen demand
    "cod",    # chemical oxygen demand
    "tds",    # total dissolved solids
    "tss",    # total suspended solids
])

_COMMON_ROOT_PREFIXES: tuple[str, ...] = (
    "sampling", "collection", "sample", "water", "sediment", "soil",
)

# 特殊字符：含之则惩罚（包括 unicode minus / en dash / em dash / 加号）
_SPECIAL_CHARS: str = "/\\|%@#()[]{}<>−–—+"

# 化学式缩写 — 同桶内若有全称则让位
# 注意：ph / ec / bod / cod / tds / tss 已在 _COMMON_ACRONYM_WHITELIST 里受保护，
# 不列入此处；且化学式惩罚仅在分数上起次级作用，全称关键词奖励更强。
_CHEM_FORMULAS: frozenset[str] = frozenset([
    # 含氢化合物 / 氧化物
    "nh4", "nh3",
    "no2", "no3", "no", "nox",
    "po4", "p2o5",
    "so4", "so3", "s2",
    "co2", "co3", "hco3",
    "sio2", "sio4",
    "ch4",
    "o2", "h2", "h2s", "h2o",
    # 常见阴离子符号
    "cl", "br", "f",
    # 元素符号（加入后 fe / ca / na / k / cu 等让位全称）
    "fe", "ca", "mg", "na", "k",
    "cu", "zn", "mn", "al", "pb", "hg", "cd", "as",
    "ni", "cr", "ba", "sr", "li", "mo", "co",
])


def _pick_representative(members: list[str], freq_map: dict[str, int]) -> str:
    """代表名选择规则 v2：clarity > brevity。

    按以下惩罚/奖励项综合排序（字典序最小胜出）：
        1. 特殊字符（斜杠/管道/百分号等）    惩罚
        2. 太短（<4 且不在通用缩写白名单）  惩罚
        3. 太长（>30）                      惩罚
        4. CamelCase                        惩罚
        5. 拼写异常（常见词根开头+缺下划线） 惩罚
        6. 含化学/科学全称关键词            奖励
        7. PMID 高优先
        8. 字符串短优先
        9. 字典序兜底
    """
    if not members:
        return ""
    candidates = [k for k in members if k and isinstance(k, str)]
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    # 预计算"有下划线规范版 peer"：若同桶中存在去掉下划线后等价的 peer，
    # 本名（无下划线版）被视为拼写异常 — 更通用，无需维护词根白名单。
    norm_map: dict[str, list[str]] = defaultdict(list)
    for n in candidates:
        norm_map[n.lower().replace("_", "")].append(n)
    has_underscored_peer: dict[str, bool] = {}
    for n in candidates:
        peers = norm_map[n.lower().replace("_", "")]
        has_underscored_peer[n] = (
            "_" not in n and any("_" in p for p in peers if p != n)
        )

    def score(name: str) -> tuple:
        pmid = int(freq_map.get(name, 0))
        name_lc = name.lower()
        normalized = name_lc.replace("-", "_").replace(" ", "_")

        # 惩罚项（0 好 1 差）
        has_special_char = any(c in name for c in _SPECIAL_CHARS)
        is_whitelisted_short = name_lc in _COMMON_ACRONYM_WHITELIST
        is_too_short = (len(name) < 4) and not is_whitelisted_short
        is_too_long = len(name) > 30
        is_camelcase = (
            name != name_lc and name != name.upper()
            and "_" not in name and len(name) >= 5
        )
        # 拼写异常 1：有规范的含下划线 peer（如 reef_type ↔ reeftype）
        is_missing_peer_underscore = has_underscored_peer.get(name, False)
        # 拼写异常 2：全小写 + 无下划线 + 长度 ≥ 7 + 以常见词根开头
        # （兜底规则：即使没有含下划线 peer 也能拦，如 samplingdate 即使桶里没 sampling_date）
        has_missing_underscore = (
            name.islower() and "_" not in name and len(name) >= 7
            and any(
                name.startswith(root) and len(name) > len(root) + 2
                for root in _COMMON_ROOT_PREFIXES
            )
        )
        # 化学式惩罚：nh4/no2/cl/fe/na/... 等同桶若有全称应让位
        name_stem = name_lc.strip("+-−–").split("_")[0].split("-")[0]
        is_chemical_formula = name_stem in _CHEM_FORMULAS

        # 奖励项（-1 好 0 差）— 含全称关键词
        has_full_word = any(kw in normalized for kw in _FULL_NAME_KEYWORDS)

        # 元组顺序决定优先级（字典序最小胜出）：
        #   1. 特殊字符  — 最高优先排除
        #   2. 白名单短词 — 强优先（让 ph 压过 sediment_ph）
        #   3. 化学式    — 让位全称（nh4→ammonium、cl→chloride、na→sodium）
        #   4-7. 其他惩罚项
        #   8. 全称关键词奖励
        #   9. PMID 高优先
        #   10. 短优先
        #   11. 字典序兜底
        return (
            int(has_special_char),
            -int(is_whitelisted_short),
            int(is_chemical_formula),
            int(is_too_short),
            int(is_too_long),
            int(is_camelcase),
            int(is_missing_peer_underscore or has_missing_underscore),
            -int(has_full_word),
            -pmid,
            len(name),
            name,
        )

    return sorted(candidates, key=score)[0]


def _self_test_pick_representative() -> None:
    """自测：覆盖关键 v2 规则转换 case。"""
    cases = [
        (["tn", "total_nitrogen", "total_n", "TN", "tdn"],
         {"tn": 498, "total_nitrogen": 250, "total_n": 80, "TN": 15, "tdn": 20},
         "total_nitrogen"),
        (["tp", "total_phosphorus", "total_p", "TP"],
         {"tp": 468, "total_phosphorus": 150, "total_p": 60, "TP": 9},
         "total_phosphorus"),
        (["mld", "mixed_layer_depth"],
         {"mld": 98, "mixed_layer_depth": 156},
         "mixed_layer_depth"),
        (["wtd", "water_table_depth"],
         {"wtd": 73, "water_table_depth": 42},
         "water_table_depth"),
        (["aou", "apparent_oxygen_utilization"],
         {"aou": 22, "apparent_oxygen_utilization": 5},
         "apparent_oxygen_utilization"),
        (["fe", "iron", "Fe"],
         {"fe": 170, "iron": 60, "Fe": 10},
         "iron"),
        (["cl", "chloride", "Cl"],
         {"cl": 110, "chloride": 30, "Cl": 5},
         "chloride"),
        (["date/time", "collection_date", "sampling_date"],
         {"date/time": 7227, "collection_date": 3000, "sampling_date": 500},
         "collection_date"),
        (["samplingdate", "sampling_date", "sample_date"],
         {"samplingdate": 5003, "sampling_date": 1200, "sample_date": 100},
         "sampling_date"),
        (["samplingdepth", "sampling_depth", "sampling_depths"],
         {"samplingdepth": 5945, "sampling_depth": 800, "sampling_depths": 50},
         "sampling_depth"),
        (["reeftype", "reef_type"],
         {"reeftype": 40, "reef_type": 5},
         "reef_type"),
        (["salinity", "sal", "water_salinity"],
         {"salinity": 4562, "sal": 152, "water_salinity": 82},
         "salinity"),
        (["ph", "pH", "water_ph"],
         {"ph": 3710, "pH": 500, "water_ph": 100},
         "ph"),  # ph 在白名单，不被 too_short 惩罚；且 PMID 最高
        (["nh4", "ammonium", "NH4", "nh4+"],
         {"nh4": 400, "ammonium": 120, "NH4": 30, "nh4+": 50},
         "ammonium"),
        # v2.1: 白名单强保护（压过含全称的候选）
        (["ph", "pH", "soil_ph", "water_ph", "sediment_ph", "seawater_ph"],
         {"ph": 2000, "pH": 500, "soil_ph": 600, "water_ph": 400,
          "sediment_ph": 300, "seawater_ph": 200},
         "ph"),
        # v2.1: 化学式 + 全称
        (["nh4", "nh4+", "ammonium", "ammonium_concentration", "nh4_n"],
         {"nh4": 254, "nh4+": 100, "ammonium": 298,
          "ammonium_concentration": 72, "nh4_n": 31},
         "ammonium"),
        # v2.1: 化学式 + unicode minus
        (["no2", "no2-", "no2−", "nitrite", "nitrite_concentration"],
         {"no2": 300, "no2-": 150, "no2−": 100,
          "nitrite": 200, "nitrite_concentration": 50},
         "nitrite"),
        (["cl", "cl-", "cl−", "chloride", "chloride_concentration"],
         {"cl": 201, "cl-": 50, "cl−": 30,
          "chloride": 109, "chloride_concentration": 20},
         "chloride"),
        (["na", "na+", "sodium", "sodium_concentration"],
         {"na": 135, "na+": 80, "sodium": 100, "sodium_concentration": 20},
         "sodium"),
        (["toc", "TOC", "total_organic_carbon", "organic_carbon", "oc"],
         {"toc": 413, "TOC": 24, "total_organic_carbon": 251,
          "organic_carbon": 180, "oc": 60},
         "total_organic_carbon"),
        (["coordinates", "gps_coordinates", "latitude_longitude", "lat_long"],
         {"coordinates": 500, "gps_coordinates": 150,
          "latitude_longitude": 80, "lat_long": 40},
         "coordinates"),
        (["habitat", "habitats", "habitat_type"],
         {"habitat": 800, "habitats": 100, "habitat_type": 300},
         "habitat"),
    ]
    ok = 0
    for members, pmid_map, expected in cases:
        actual = _pick_representative(members, pmid_map)
        if actual == expected:
            ok += 1
            print(f"  ✓ {expected}  (members={members})")
        else:
            print(f"  ✗ expected={expected}  got={actual}  members={members}")
    print(f"\n{ok}/{len(cases)} passed")


# ── Evidence ───────────────────────────────────────────────────────
def _is_text_section(section_type: str) -> bool:
    return "table" not in (section_type or "").lower()


def _build_evidence_index(raw_keys: set[str]) -> dict[str, list[dict]]:
    logger.info("Building evidence index for %d keys…", len(raw_keys))
    with open(config.STEP2_INPUT, "r", encoding="utf-8") as f:
        records = json.load(f)
    text_b: dict[str, list[dict]] = {k: [] for k in raw_keys}
    table_b: dict[str, list[dict]] = {k: [] for k in raw_keys}
    for r in records:
        keys = r.get("metadata_keys_found") or []
        if not keys:
            continue
        pmid = r.get("pmid", "")
        sec = r.get("section_type", "")
        is_text = _is_text_section(sec)
        for k in keys:
            if k not in text_b:
                continue
            bucket = text_b[k] if is_text else table_b[k]
            if len(bucket) >= EVIDENCE_CAP:
                continue
            if any(e["pmid"] == pmid for e in bucket):
                continue
            quote = (r.get("evidence_quote") or "")[:EVIDENCE_CHAR_CAP]
            bucket.append({"pmid": pmid, "quote": quote})
    out = {}
    for k in raw_keys:
        combined = list(text_b[k])
        seen = {e["pmid"] for e in combined}
        if len(combined) < EVIDENCE_CAP:
            for e in table_b[k]:
                if len(combined) >= EVIDENCE_CAP:
                    break
                if e["pmid"] in seen:
                    continue
                combined.append(e)
                seen.add(e["pmid"])
        out[k] = combined
    logger.info("Evidence index built")
    return out


def _fmt_evidence(evs: list[dict]) -> str:
    if not evs:
        return "(no evidence)"
    lines = []
    for e in evs:
        q = (e["quote"] or "").strip().replace("\n", " ")
        if len(q) > EVIDENCE_CHAR_CAP:
            q = q[:EVIDENCE_CHAR_CAP] + "…"
        lines.append(f"[{e['pmid']}] {q}")
    return "\n".join(lines)


# ── EDC verify prompt ──────────────────────────────────────────────
EDC_PROMPT = """\
You are a hydrosphere metadata curator. Two raw field names already share
the exact same (subtype, quantity_kind, modifier_bag) — their structured
meaning is identical. Judge ONLY whether their string forms represent the
**same concept** or different concepts.

Context (shared for both):
  subtype        : {subtype}
  quantity_kind  : {quantity_kind}
  modifier_bag   : {modifier_bag}

Field A: `{rep_a}`
  sibling raw_keys: {members_a}
  evidence: {evidence_a}

Field B: `{rep_b}`
  sibling raw_keys: {members_b}
  evidence: {evidence_b}

Decide:
  - "merge"          if A and B are the same measurement (abbreviation /
                     full-name / synonym / unit variant / spelling variant)
  - "keep_separate"  if they are different measurements whose 3-tuple just
                     happened to match (rare but possible)

================== STRICT ANTI-MERGE EXAMPLES (MUST keep_separate) ==================

Even when string forms look similar, these are DIFFERENT measurements:
  - `biochemical_oxygen_demand` (BOD) ≠ `chemical_oxygen_demand` (COD)
    → BOD quantifies biodegradable organic matter via microbial O2
      consumption; COD quantifies total oxidizable matter via chemical
      oxidation. Different protocols, different values.
  - `nitrate` ≠ `nitrite`
    → Different N species (NO3 vs NO2).
  - `total_nitrogen` ≠ `nitrate` or `nitrite`
    → Total vs single species.
  - `oxygen_concentration` (mg/L) ≠ `oxygen_saturation` (%)
    → Absolute vs relative measure.
  - `fluorescence` (raw RFU) ≠ `chlorophyll_a_concentration` (μg/L)
    → Signal vs calibrated quantity.
  - `surface_temperature` ≠ `bottom_temperature`
    → Different water-column positions (bag differs).

General principle:
  - Same measurement concept but different abbreviation / synonym / spelling
    → merge.
  - Different species, different measurement method, different water column
    position, or different target matrix → keep_separate.

Do NOT infer hidden chemical semantics — judge from string form, evidence
and the shared structured slots above.

Return JSON:
{{"decision": "merge" | "keep_separate", "reasoning": "one short sentence"}}
</json>"""


# ── MIxS alignment prompt ──────────────────────────────────────────
MIXS_PROMPT = """\
You are aligning a merged metadata canonical to the MIxS (hydrosphere-
relevant subset) standard slot vocabulary.

Canonical info:
  canonical_name   : {canonical_name}
  family / subtype : {family} / {subtype}
  quantity_kind    : {quantity_kind}
  modifier_bag     : {modifier_bag}
  sample members   : {member_samples}

MIxS water-sphere slot candidates (89 slots; choose ONE or UNMAPPED):
{slot_block}

Decision rules:
  - "exact"    canonical matches the slot concept precisely
  - "subset"   canonical is a MORE SPECIFIC sub-concept of the slot
               (e.g. mixed_layer_depth vs Water_depth)
  - "superset" canonical is a MORE GENERAL concept that covers the slot
               (rare; e.g. generic "nitrogen" vs slot tot_nitro)
  - "partial"  canonical partly overlaps but neither subset nor superset
               (e.g. "conductivity_specific" vs slot conduc — related
                but measurement convention differs)
  - "UNMAPPED" canonical falls entirely outside the 89-slot hydrosphere
               vocabulary (e.g. peat_depth, aragonite_saturation)

Do NOT invent slots. The slot name MUST appear verbatim in the list above,
or be exactly "UNMAPPED".

Return JSON:
{{
  "mixs_slot": "<slot_name or UNMAPPED>",
  "mixs_alignment": "exact | subset | superset | partial | UNMAPPED",
  "mixs_reasoning": "one short English sentence"
}}
</json>"""


def _load_mixs_slots() -> list[dict]:
    if not MIXS_SLOTS_FILE.exists():
        raise RuntimeError(f"MIxS slot file missing: {MIXS_SLOTS_FILE}")
    with open(MIXS_SLOTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_slot_block(slots: list[dict]) -> str:
    lines = []
    for s in slots:
        name = s["slot_name"]
        desc = (s.get("description") or "").strip()[:120]
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


# ── LLM helpers ────────────────────────────────────────────────────
def _load_checkpoint(path: Path, key: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                if key in d:
                    out[d[key]] = d
            except Exception:
                continue
    return out


async def _edc_verify_pair(
    client: AsyncLocalModelClient,
    bucket_key: tuple, rep_a: str, members_a: list[str],
    rep_b: str, members_b: list[str],
    ev_idx: dict[str, list[dict]],
) -> dict:
    ev_a = ev_idx.get(rep_a, [])
    ev_b = ev_idx.get(rep_b, [])
    fam, sub, qk, bag = bucket_key
    bag_str = "|".join(bag) if bag else ""
    prompt = EDC_PROMPT.format(
        subtype=sub, quantity_kind=qk, modifier_bag=bag_str or "(empty)",
        rep_a=rep_a,
        members_a=", ".join(members_a[:6]),
        evidence_a=_fmt_evidence(ev_a)[:400],
        rep_b=rep_b,
        members_b=", ".join(members_b[:6]),
        evidence_b=_fmt_evidence(ev_b)[:400],
    )
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.chat([{"role": "user", "content": prompt}])
            if resp is None:
                await asyncio.sleep(backoff_with_jitter(attempt))
                continue
            parsed = extract_json_from_response_with_repair(
                resp, stop_sentinel=config.STOP_SENTINEL
            )
            if isinstance(parsed, list):
                parsed = next((x for x in parsed if isinstance(x, dict)), None)
            if not isinstance(parsed, dict):
                await asyncio.sleep(backoff_with_jitter(attempt))
                continue
            d = str(parsed.get("decision", "")).strip().lower()
            if d not in {"merge", "keep_separate"}:
                d = "keep_separate"
            return {
                "bucket_family": fam, "bucket_subtype": sub,
                "bucket_qk": qk, "bucket_bag": bag_str,
                "rep_a": rep_a, "members_a": ";".join(members_a),
                "rep_b": rep_b, "members_b": ";".join(members_b),
                "decision": d,
                "reasoning": str(parsed.get("reasoning", "")).strip(),
                "pair_key": f"{fam}|{sub}|{qk}|{bag_str}|{rep_a}|{rep_b}",
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("EDC attempt %d failed for %s vs %s: %s",
                           attempt, rep_a, rep_b, e)
            await asyncio.sleep(backoff_with_jitter(attempt))
    return {
        "bucket_family": fam, "bucket_subtype": sub,
        "bucket_qk": qk, "bucket_bag": bag_str,
        "rep_a": rep_a, "members_a": ";".join(members_a),
        "rep_b": rep_b, "members_b": ";".join(members_b),
        "decision": "keep_separate",
        "reasoning": "LLM failed after retries",
        "pair_key": f"{fam}|{sub}|{qk}|{bag_str}|{rep_a}|{rep_b}",
    }


async def _mixs_align_one(
    client: AsyncLocalModelClient,
    canonical: dict, slot_block: str, valid_slots: set[str],
) -> dict:
    prompt = MIXS_PROMPT.format(
        canonical_name=canonical["canonical_name"],
        family=canonical["family"], subtype=canonical["subtype"],
        quantity_kind=canonical["quantity_kind"],
        modifier_bag=canonical.get("modifier_bag") or "(empty)",
        member_samples=", ".join(canonical["member_samples"][:5]),
        slot_block=slot_block,
    )
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.chat([{"role": "user", "content": prompt}])
            if resp is None:
                await asyncio.sleep(backoff_with_jitter(attempt))
                continue
            parsed = extract_json_from_response_with_repair(
                resp, stop_sentinel=config.STOP_SENTINEL
            )
            if isinstance(parsed, list):
                parsed = next((x for x in parsed if isinstance(x, dict)), None)
            if not isinstance(parsed, dict):
                await asyncio.sleep(backoff_with_jitter(attempt))
                continue
            slot = str(parsed.get("mixs_slot", "")).strip()
            alignment = str(parsed.get("mixs_alignment", "")).strip().lower()
            valid_alignments = {"exact", "subset", "superset", "partial", "unmapped"}
            if alignment not in valid_alignments:
                alignment = "unmapped"
            if alignment == "unmapped" or slot == "":
                slot = "UNMAPPED"
                alignment = "UNMAPPED"
            elif slot not in valid_slots:
                # LLM hallucinated a slot name → downgrade to UNMAPPED
                slot = "UNMAPPED"
                alignment = "UNMAPPED"
            else:
                alignment = alignment.lower() if alignment != "UNMAPPED" else "UNMAPPED"
            return {
                "canonical_id": canonical["canonical_id"],
                "canonical_name": canonical["canonical_name"],
                "mixs_slot": slot,
                "mixs_alignment": alignment,
                "mixs_reasoning": str(parsed.get("mixs_reasoning", "")).strip(),
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("MIxS attempt %d failed for %s: %s",
                           attempt, canonical["canonical_id"], e)
            await asyncio.sleep(backoff_with_jitter(attempt))
    return {
        "canonical_id": canonical["canonical_id"],
        "canonical_name": canonical["canonical_name"],
        "mixs_slot": "UNMAPPED",
        "mixs_alignment": "UNMAPPED",
        "mixs_reasoning": "LLM failed after retries",
    }


# ── 主流程 ──────────────────────────────────────────────────────────
def run() -> None:
    config.ensure_output_dir()
    df = pd.read_csv(config.PHASE3_OUTPUT)
    logger.info("Loaded %d annotations", len(df))

    # 清理：剔除 FAILED 标注
    df = df[df["family"] != "FAILED"].copy()
    logger.info("After dropping FAILED: %d rows", len(df))

    # 合并键
    df["_merge_key"] = df.apply(_merge_key, axis=1)
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, k in enumerate(df["_merge_key"]):
        buckets[k].append(i)
    logger.info("Total buckets: %d", len(buckets))

    # Freq map（按 total_pmid）
    # 注意：展开后多个 raw_key_original 可能映射到同一 raw_key（如 temp_c/t_c/t°c → temperature），
    # 此时 df 里同 raw_key 有多行，需要对 pmid 求和，否则 dict(zip) 按 raw_key 去重会丢信息。
    freq_map = df.groupby("raw_key")["total_pmid"].sum().astype(int).to_dict()

    # 字符串聚类（每桶内）
    bucket_components: dict[tuple, list[list[str]]] = {}
    n_single_bucket = 0
    for key, idxs in buckets.items():
        members = df.iloc[idxs]["raw_key"].tolist()
        uf = UF(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if _similar(members[i], members[j]):
                    uf.union(members[i], members[j])
        comps = list(uf.components().values())
        bucket_components[key] = comps
        if len(comps) == 1:
            n_single_bucket += 1
    logger.info("Single-component buckets: %d / %d", n_single_bucket, len(buckets))

    # EDC verify 目标
    edc_targets: list[tuple[tuple, list[str], list[str]]] = []
    for key, comps in bucket_components.items():
        if len(comps) < 2:
            continue
        # 对所有分量两两 pair（经济考量：只当至少一个是 singleton 时触发，
        # 否则两个大分量串在一起的风险高）
        has_singleton = any(len(c) == 1 for c in comps)
        if not has_singleton:
            continue
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                edc_targets.append((key, comps[i], comps[j]))
    logger.info("EDC verify targets: %d pairs", len(edc_targets))

    # 调用 EDC
    edc_results: list[dict] = []
    if edc_targets:
        ev_idx = _build_evidence_index(set(df["raw_key"]))
        done = _load_checkpoint(PHASE4_EDC_CHECKPOINT, key="pair_key")
        logger.info("EDC resumed from checkpoint: %d", len(done))
        edc_results = list(done.values())

        api_key = os.environ.get("ALL_API_KEY")
        if not api_key:
            raise RuntimeError("ALL_API_KEY required")

        todo = [
            (k, a, b) for (k, a, b) in edc_targets
            if f"{k[0]}|{k[1]}|{k[2]}|{('|'.join(k[3]) if k[3] else '')}|"
               f"{_pick_representative(a, freq_map)}|"
               f"{_pick_representative(b, freq_map)}" not in done
        ]

        async def _go():
            sem = asyncio.Semaphore(CONCURRENCY)
            lock = asyncio.Lock()
            async with AsyncLocalModelClient(
                base_url=config.BASE_URL, model=config.MODEL,
                temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS,
                api_key=api_key, stop_sentinel=config.STOP_SENTINEL,
                api_style=config.API_STYLE, auth_mode=config.AUTH_MODE,
            ) as client:
                async def worker(key, comp_a, comp_b):
                    async with sem:
                        rep_a = _pick_representative(comp_a, freq_map)
                        rep_b = _pick_representative(comp_b, freq_map)
                        res = await _edc_verify_pair(
                            client, key, rep_a, comp_a,
                            rep_b, comp_b, ev_idx
                        )
                        async with lock:
                            with open(PHASE4_EDC_CHECKPOINT, "a",
                                      encoding="utf-8") as f:
                                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                        return res

                tasks = [asyncio.create_task(worker(k, a, b))
                         for k, a, b in todo]
                start = time.time()
                for i, coro in enumerate(asyncio.as_completed(tasks)):
                    edc_results.append(await coro)
                    if (i + 1) % 50 == 0:
                        rate = (i + 1) / max(1e-6, time.time() - start)
                        logger.info("EDC progress %d/%d (%.1f/s)",
                                    i + 1, len(todo), rate)
        asyncio.run(_go())
        logger.info("EDC done: %d pairs", len(edc_results))

    # 应用 EDC merge 决策
    for r in edc_results:
        if r["decision"] != "merge":
            continue
        key = (r["bucket_family"], r["bucket_subtype"], r["bucket_qk"],
               tuple(r["bucket_bag"].split("|")) if r["bucket_bag"] else ())
        comps = bucket_components.get(key)
        if not comps:
            continue
        mem_a = set(r["members_a"].split(";"))
        mem_b = set(r["members_b"].split(";"))
        ia = next((i for i, c in enumerate(comps) if any(m in mem_a for m in c)), None)
        ib = next((i for i, c in enumerate(comps) if any(m in mem_b for m in c)), None)
        if ia is None or ib is None or ia == ib:
            continue
        merged = comps[ia] + comps[ib]
        bucket_components[key] = (
            [c for i, c in enumerate(comps) if i not in (ia, ib)] + [merged]
        )

    # 组装 canonical
    canonicals: list[dict] = []
    mapping: list[dict] = []
    bucket_stats: list[dict] = []
    cid = 0
    row_by_rk = {r["raw_key"]: r for _, r in df.iterrows()}

    for key, comps in bucket_components.items():
        fam, sub, qk, bag = key
        edc_pair_cnt = sum(
            1 for r in edc_results
            if r["bucket_family"] == fam and r["bucket_subtype"] == sub
            and r["bucket_qk"] == qk
            and r["bucket_bag"] == ("|".join(bag) if bag else "")
        )
        edc_merged_cnt = sum(
            1 for r in edc_results
            if r["bucket_family"] == fam and r["bucket_subtype"] == sub
            and r["bucket_qk"] == qk
            and r["bucket_bag"] == ("|".join(bag) if bag else "")
            and r["decision"] == "merge"
        )
        bucket_stats.append({
            "family": fam, "subtype": sub, "quantity_kind": qk,
            "modifier_bag": "|".join(bag) if bag else "",
            "n_raw_keys": sum(len(c) for c in comps),
            "n_canonicals": len(comps),
            "edc_pairs_tested": edc_pair_cnt,
            "edc_pairs_merged": edc_merged_cnt,
        })
        for comp in comps:
            cid += 1
            canonical_id = f"c_{cid:05d}"
            rep = _pick_representative(comp, freq_map)
            env_vec = {e: 0 for e in config.HYDRO_ENVS}
            tot_pmid = 0
            for m in comp:
                r = row_by_rk[m]
                for e in config.HYDRO_ENVS:
                    env_vec[e] += int(r[f"env_{e}"])
                tot_pmid += int(r["total_pmid"])
            n_envs = sum(1 for v in env_vec.values() if v > 0)
            canonicals.append({
                "canonical_id": canonical_id,
                "canonical_name": rep,
                "family": fam, "subtype": sub, "quantity_kind": qk,
                "modifier_bag": "|".join(bag) if bag else "",
                **{f"env_{e}": env_vec[e] for e in config.HYDRO_ENVS},
                "total_pmid": tot_pmid,
                "n_envs_present": n_envs,
                "n_member_raw_keys": len(comp),
                "member_raw_keys": "|".join(comp),
                "member_samples": sorted(comp, key=lambda m: -freq_map.get(m, 0))[:5],
            })
            for m in comp:
                mapping.append({
                    "canonical_id": canonical_id,
                    "raw_key": m,
                    "raw_key_pmid": int(freq_map.get(m, 0)),
                    "is_representative": int(m == rep),
                })
    logger.info("Total canonicals: %d (from %d raw_keys)",
                len(canonicals), len(df))

    # MIxS 对齐
    logger.info("=" * 60)
    logger.info("MIxS alignment stage")
    logger.info("=" * 60)
    _run_mixs_alignment(canonicals)

    # 写 CSV
    _write_outputs(canonicals, mapping, bucket_stats, edc_results)

    # 验证
    _print_verification(canonicals, mapping, df)


def _run_mixs_alignment(canonicals: list[dict]) -> None:
    slots = _load_mixs_slots()
    valid_slots = {s["slot_name"] for s in slots}
    slot_block = _fmt_slot_block(slots)
    logger.info("MIxS candidates: %d slots", len(slots))

    done = _load_checkpoint(PHASE4_MIXS_CHECKPOINT, key="canonical_id")
    logger.info("MIxS resumed: %d", len(done))

    for c in canonicals:
        cid = c["canonical_id"]
        if cid in done:
            c["mixs_slot"] = done[cid]["mixs_slot"]
            c["mixs_alignment"] = done[cid]["mixs_alignment"]
            c["mixs_reasoning"] = done[cid]["mixs_reasoning"]

    todo = [c for c in canonicals if "mixs_slot" not in c]
    logger.info("MIxS todo: %d / %d", len(todo), len(canonicals))
    if not todo:
        return

    api_key = os.environ.get("ALL_API_KEY")
    if not api_key:
        raise RuntimeError("ALL_API_KEY required")

    async def _go():
        sem = asyncio.Semaphore(CONCURRENCY)
        lock = asyncio.Lock()
        async with AsyncLocalModelClient(
            base_url=config.BASE_URL, model=config.MODEL,
            temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS,
            api_key=api_key, stop_sentinel=config.STOP_SENTINEL,
            api_style=config.API_STYLE, auth_mode=config.AUTH_MODE,
        ) as client:
            async def worker(c):
                async with sem:
                    res = await _mixs_align_one(client, c, slot_block, valid_slots)
                    async with lock:
                        with open(PHASE4_MIXS_CHECKPOINT, "a",
                                  encoding="utf-8") as f:
                            f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    return res

            tasks = [asyncio.create_task(worker(c)) for c in todo]
            start = time.time()
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                res = await coro
                # 找到对应 canonical 并赋值
                for c in canonicals:
                    if c["canonical_id"] == res["canonical_id"]:
                        c["mixs_slot"] = res["mixs_slot"]
                        c["mixs_alignment"] = res["mixs_alignment"]
                        c["mixs_reasoning"] = res["mixs_reasoning"]
                        break
                if (i + 1) % 100 == 0:
                    rate = (i + 1) / max(1e-6, time.time() - start)
                    logger.info("MIxS progress %d/%d (%.1f/s)",
                                i + 1, len(todo), rate)

    asyncio.run(_go())


def _write_outputs(canonicals, mapping, bucket_stats, edc_results):
    # 清理 member_samples 列（仅内存用，不写 CSV）
    canonical_rows = []
    for c in canonicals:
        d = {k: v for k, v in c.items() if k != "member_samples"}
        canonical_rows.append(d)
    can_df = pd.DataFrame(canonical_rows).sort_values(
        "total_pmid", ascending=False
    )
    can_df.to_csv(PHASE4_CANONICALS, index=False)
    logger.info("Wrote %d canonicals → %s", len(can_df), PHASE4_CANONICALS)

    map_df = pd.DataFrame(mapping).sort_values(
        ["canonical_id", "is_representative", "raw_key_pmid"],
        ascending=[True, False, False],
    )
    map_df.to_csv(PHASE4_MAPPING, index=False)
    logger.info("Wrote %d raw_key mappings → %s", len(map_df), PHASE4_MAPPING)

    bs_df = pd.DataFrame(bucket_stats).sort_values(
        "n_raw_keys", ascending=False
    )
    bs_df.to_csv(PHASE4_BUCKET_STATS, index=False)
    logger.info("Wrote bucket stats → %s", PHASE4_BUCKET_STATS)

    if edc_results:
        edc_df = pd.DataFrame(edc_results).drop(columns=["pair_key"], errors="ignore")
        edc_df.to_csv(PHASE4_EDC_LOG, index=False)
        logger.info("Wrote EDC log → %s (%d pairs)", PHASE4_EDC_LOG, len(edc_df))

    # MIxS 日志（完整 reasoning）
    mixs_rows = []
    for c in canonicals:
        mixs_rows.append({
            "canonical_id": c["canonical_id"],
            "canonical_name": c["canonical_name"],
            "family": c["family"], "subtype": c["subtype"],
            "quantity_kind": c["quantity_kind"],
            "modifier_bag": c["modifier_bag"],
            "mixs_slot": c.get("mixs_slot", ""),
            "mixs_alignment": c.get("mixs_alignment", ""),
            "mixs_reasoning": c.get("mixs_reasoning", ""),
        })
    pd.DataFrame(mixs_rows).to_csv(PHASE4_MIXS_LOG, index=False)
    logger.info("Wrote MIxS alignment log → %s", PHASE4_MIXS_LOG)

    # Singletons
    sing = can_df[can_df["n_member_raw_keys"] == 1].copy()
    sing.to_csv(PHASE4_SINGLETONS, index=False)
    logger.info("Wrote %d singletons → %s", len(sing), PHASE4_SINGLETONS)


def _print_verification(canonicals, mapping, df):
    logger.info("=" * 60)
    logger.info("PHASE 4 VERIFICATION")
    logger.info("=" * 60)
    logger.info("Total canonicals: %d", len(canonicals))
    logger.info("Total raw_keys in mapping: %d (should == %d)",
                len(mapping), len(df))

    size_cnt = Counter(c["n_member_raw_keys"] for c in canonicals)
    logger.info("Canonical size distribution (top): %s",
                sorted(size_cnt.items())[:8])
    logger.info("Mean members/canonical: %.2f",
                sum(c["n_member_raw_keys"] for c in canonicals) / max(1, len(canonicals)))
    logger.info("Singletons: %d (%.1f%%)",
                sum(1 for c in canonicals if c["n_member_raw_keys"] == 1),
                100 * sum(1 for c in canonicals if c["n_member_raw_keys"] == 1) / max(1, len(canonicals)))

    # MIxS 分布
    aligns = Counter(c.get("mixs_alignment", "") for c in canonicals)
    logger.info("MIxS alignment distribution: %s", dict(aligns))

    # 经典家族合并抽检
    logger.info("\n经典 canonical 抽检：")
    cn_to_c = {c["canonical_name"]: c for c in canonicals}
    rk_to_cid = {m["raw_key"]: m["canonical_id"] for m in mapping}

    for probe_rks in [
        ["salinity", "water_salinity", "sal"],
        ["dissolved_oxygen", "do", "o2"],
        ["depth", "sampling_depth"],
        ["temperature", "temp"],
        ["latitude", "lat"],
        ["collection_date", "sampling_date"],
        ["surface_salinity", "bottom_salinity"],
        ["total_phosphorus", "tp"],
    ]:
        cids = {rk: rk_to_cid.get(rk, "MISSING") for rk in probe_rks}
        logger.info("  %s → %s", probe_rks, cids)


# ── rerun_mixs 入口 ────────────────────────────────────────────────
def rerun_mixs() -> None:
    """仅重跑 MIxS 对齐；canonical 结构不变。"""
    config.ensure_output_dir()
    if not PHASE4_CANONICALS.exists():
        raise RuntimeError(f"Need {PHASE4_CANONICALS} first (run 4 once)")

    logger.info("Re-reading existing canonicals …")
    can_df = pd.read_csv(PHASE4_CANONICALS)
    map_df = pd.read_csv(PHASE4_MAPPING)

    # 构造 member_samples
    mapping_by_cid = defaultdict(list)
    for r in map_df.itertuples(index=False):
        mapping_by_cid[r.canonical_id].append((r.raw_key, int(r.raw_key_pmid)))

    canonicals: list[dict] = []
    for _, r in can_df.iterrows():
        cid = r["canonical_id"]
        members = sorted(mapping_by_cid[cid], key=lambda x: -x[1])
        canonicals.append({
            "canonical_id": cid,
            "canonical_name": r["canonical_name"],
            "family": r["family"], "subtype": r["subtype"],
            "quantity_kind": r["quantity_kind"],
            "modifier_bag": r.get("modifier_bag", "") if pd.notna(r.get("modifier_bag")) else "",
            "member_samples": [m for m, _ in members[:5]],
        })

    # 清 MIxS checkpoint
    if PHASE4_MIXS_CHECKPOINT.exists():
        PHASE4_MIXS_CHECKPOINT.unlink()
        logger.info("Cleared MIxS checkpoint")

    _run_mixs_alignment(canonicals)

    # 更新 canonicals CSV（只覆盖 mixs_* 列）
    mixs_map = {c["canonical_id"]: (c["mixs_slot"], c["mixs_alignment"],
                                     c["mixs_reasoning"]) for c in canonicals}
    can_df["mixs_slot"] = can_df["canonical_id"].map(lambda x: mixs_map[x][0])
    can_df["mixs_alignment"] = can_df["canonical_id"].map(lambda x: mixs_map[x][1])
    can_df["mixs_reasoning"] = can_df["canonical_id"].map(lambda x: mixs_map[x][2])
    can_df.to_csv(PHASE4_CANONICALS, index=False)
    logger.info("Updated %s with fresh MIxS alignments", PHASE4_CANONICALS)

    # 重写 MIxS log
    mixs_rows = []
    for _, r in can_df.iterrows():
        mixs_rows.append({
            "canonical_id": r["canonical_id"],
            "canonical_name": r["canonical_name"],
            "family": r["family"], "subtype": r["subtype"],
            "quantity_kind": r["quantity_kind"],
            "modifier_bag": r["modifier_bag"],
            "mixs_slot": r["mixs_slot"],
            "mixs_alignment": r["mixs_alignment"],
            "mixs_reasoning": r["mixs_reasoning"],
        })
    pd.DataFrame(mixs_rows).to_csv(PHASE4_MIXS_LOG, index=False)
    logger.info("Wrote MIxS alignment log → %s", PHASE4_MIXS_LOG)

    aligns = Counter(can_df["mixs_alignment"])
    logger.info("MIxS alignment distribution: %s", dict(aligns))


# ── rename_only 入口 ──────────────────────────────────────────────
def rename_only() -> None:
    """只重算 canonical_name（v2 规则），不重跑合并/EDC/MIxS。

    读 env4_canonicals.csv 和 env4_canonical_to_raw_key.csv，对每个 canonical
    按 v2 规则重选代表名，覆写 canonical_name 列 + is_representative 标记。
    Step 5 重跑时会自动继承新名字。
    """
    config.ensure_output_dir()
    if not PHASE4_CANONICALS.exists():
        raise RuntimeError(f"Need {PHASE4_CANONICALS} first")

    logger.info("Loading canonicals + mapping for rename-only …")
    can = pd.read_csv(PHASE4_CANONICALS)
    map_df = pd.read_csv(PHASE4_MAPPING)

    # raw_key → total_pmid (from phase3 output, used as freq_map)
    annot = pd.read_csv(config.PHASE3_OUTPUT)[["raw_key", "total_pmid"]]
    raw_key_to_pmid = dict(zip(annot["raw_key"].astype(str),
                                annot["total_pmid"].astype(int)))

    changed = 0
    new_names: list[str] = []
    for _, row in can.iterrows():
        members_str = str(row.get("member_raw_keys") or "")
        members = [m for m in members_str.split("|") if m]
        if not members:
            new_names.append(row["canonical_name"])
            continue
        new_rep = _pick_representative(members, raw_key_to_pmid)
        if not new_rep:
            new_rep = row["canonical_name"]
        if new_rep != row["canonical_name"]:
            changed += 1
        new_names.append(new_rep)
    can["canonical_name"] = new_names
    logger.info("Renamed %d / %d canonicals", changed, len(can))

    # 更新 is_representative 标记
    cid_to_rep = dict(zip(can["canonical_id"], can["canonical_name"]))
    map_df["is_representative"] = map_df.apply(
        lambda r: int(r["raw_key"] == cid_to_rep.get(r["canonical_id"], "")),
        axis=1,
    )

    can.to_csv(PHASE4_CANONICALS, index=False)
    map_df.to_csv(PHASE4_MAPPING, index=False)
    logger.info("Updated %s", PHASE4_CANONICALS)
    logger.info("Updated %s", PHASE4_MAPPING)

    # 同步更新 MIxS log 的 canonical_name 列（其他列不变）
    if PHASE4_MIXS_LOG.exists():
        mlog = pd.read_csv(PHASE4_MIXS_LOG)
        mlog["canonical_name"] = mlog["canonical_id"].map(cid_to_rep)
        mlog.to_csv(PHASE4_MIXS_LOG, index=False)
        logger.info("Updated canonical_name in %s", PHASE4_MIXS_LOG)

    # 同步 singleton CSV
    sing = can[can["n_member_raw_keys"] == 1].copy()
    sing.to_csv(PHASE4_SINGLETONS, index=False)

    # 抽样展示关键改名
    logger.info("")
    logger.info("改名抽样（查 TN/TP/MLD/AOU 等常见缩写）：")
    for probe in ["tn", "tp", "mld", "wtd", "aou", "fe", "cl", "nh4",
                   "samplingdate", "samplingdepth", "date/time", "reeftype"]:
        hit = can[can["member_raw_keys"].astype(str).str.contains(
            probe.replace("/", r"\/"), case=False, regex=True, na=False
        )]
        for _, r in hit.head(2).iterrows():
            members_preview = str(r["member_raw_keys"])[:60]
            logger.info("  [%s] %-35s → %s  (members: %s…)",
                        probe, "contained in", r["canonical_name"], members_preview)
