"""环节 3：结构化标注（family + subtype + quantity_kind + modifier_bag）。

对 env2_kept 中每条 raw key 做四槽位标注：

    family         4 选 1  (A_physicochemical / B_env_categorical /
                            C_spatiotemporal / D_other)
    subtype        按族取值，共 24 个子类（A=11, B=4, C=8, D=1）
    quantity_kind  LLM free-gen snake_case，不绑定清单
    modifier_bag   仅族 A；从 14 词固定词表中取多值；字面可见才加

产出：
    env3_structured_annotations.csv   每条 raw key 的完整标注
    env3_quantity_kind_counts.csv     quantity_kind 频次统计（为 3a-norm 做输入）
    env3_structured.checkpoint.jsonl  断点续跑
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import Counter
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
EVIDENCE_CHAR_CAP = 600

PHASE3_CHECKPOINT: Path = config.OUTPUT_DIR / "env3_structured.checkpoint.jsonl"


# ── Schema: 4 族 + 24 子类 ──────────────────────────────────────────
FAMILY_SCHEMA: dict[str, list[str]] = {
    "A_physicochemical": [
        "nutrient_chemistry", "temperature", "salinity", "trace_chemistry",
        "ph_alkalinity", "oxygen", "physical_env_driver",
        "chlorophyll_pigment", "conductivity_tds", "turbidity_transparency",
        "other_chemistry",
    ],
    "B_env_categorical": [
        "habitat_biome", "material_medium_type", "env_state_context",
        "other_categorical",
    ],
    "C_spatiotemporal": [
        "vertical_position", "time_point", "sampling_site", "geo_coord",
        "time_duration", "geo_region", "spatial_metric", "water_body_descriptor",
    ],
    "D_other": ["other"],
}
FAMILY_CHOICES: tuple[str, ...] = tuple(FAMILY_SCHEMA.keys())

# 同 family 下的 subtype fallback（若 LLM 选错子类，降级到该族的 catch-all）
_FAMILY_FALLBACK_SUBTYPE: dict[str, Optional[str]] = {
    "A_physicochemical": "other_chemistry",
    "B_env_categorical": "other_categorical",
    "C_spatiotemporal": None,    # 无 catch-all 子类 → 退到 D_other
    "D_other": "other",
}

# modifier_bag 固定词表（14 词）
MODIFIER_POSITION: frozenset[str] = frozenset(
    {"surface", "middle", "bottom", "porewater", "mix"}
)
MODIFIER_FORM: frozenset[str] = frozenset(
    {"dissolved", "particulate", "total", "organic", "inorganic"}
)
MODIFIER_STAT: frozenset[str] = frozenset({"mean", "max", "min", "range"})
MODIFIER_BAG_CHOICES: frozenset[str] = (
    MODIFIER_POSITION | MODIFIER_FORM | MODIFIER_STAT
)


# ── Few-shot（10-12 条合成示例） ───────────────────────────────────
FEW_SHOT_EXAMPLES: list[dict] = [
    # Family A — 物化测量，含 modifier 使用
    {"raw_key": "salinity",
     "family": "A_physicochemical", "subtype": "salinity",
     "quantity_kind": "salinity", "modifier_bag": []},
    {"raw_key": "surface_salinity",
     "family": "A_physicochemical", "subtype": "salinity",
     "quantity_kind": "salinity", "modifier_bag": ["surface"]},
    {"raw_key": "dissolved_oxygen",
     "family": "A_physicochemical", "subtype": "oxygen",
     "quantity_kind": "oxygen", "modifier_bag": ["dissolved"]},
    {"raw_key": "total_dissolved_phosphorus",
     "family": "A_physicochemical", "subtype": "nutrient_chemistry",
     "quantity_kind": "phosphorus", "modifier_bag": ["total", "dissolved"]},
    {"raw_key": "porewater_ph",
     "family": "A_physicochemical", "subtype": "ph_alkalinity",
     "quantity_kind": "ph", "modifier_bag": ["porewater"]},
    {"raw_key": "mean_annual_temperature",
     "family": "A_physicochemical", "subtype": "temperature",
     "quantity_kind": "temperature", "modifier_bag": ["mean"]},
    # Family B
    {"raw_key": "habitat_type",
     "family": "B_env_categorical", "subtype": "habitat_biome",
     "quantity_kind": "habitat_type", "modifier_bag": []},
    {"raw_key": "sediment_type",
     "family": "B_env_categorical", "subtype": "material_medium_type",
     "quantity_kind": "sediment_type", "modifier_bag": []},
    # Family C
    {"raw_key": "latitude",
     "family": "C_spatiotemporal", "subtype": "geo_coord",
     "quantity_kind": "latitude", "modifier_bag": []},
    {"raw_key": "mixed_layer_depth",
     "family": "C_spatiotemporal", "subtype": "vertical_position",
     "quantity_kind": "mixed_layer_depth", "modifier_bag": []},
    {"raw_key": "collection_date",
     "family": "C_spatiotemporal", "subtype": "time_point",
     "quantity_kind": "collection_date", "modifier_bag": []},
    {"raw_key": "lake_area",
     "family": "C_spatiotemporal", "subtype": "spatial_metric",
     "quantity_kind": "lake_area", "modifier_bag": []},
]


# ── Prompt ──────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
You are an expert curator of hydrosphere environmental metadata (open ocean,
coastal waters, lakes, wetlands). The given field has already passed the
is_env_metadata filter. Your task is to assign four structured slots.

===================== FAMILY + SUBTYPE SCHEMA (24 subtypes) =====================

family=A_physicochemical (physico-chemical measurements; **modifier_bag allowed**):
  - nutrient_chemistry     — N / P / Si / S / C nutrients, methane, organic matter
  - temperature            — temperature
  - salinity               — salinity
  - trace_chemistry        — metals / major ions / stable isotopes / redox / density / porosity / grain size
  - ph_alkalinity          — pH and alkalinity
  - oxygen                 — dissolved oxygen and oxygen demand (BOD / COD / AOU)
  - physical_env_driver    — light / current / wind / precipitation / humidity / water level / moisture
  - chlorophyll_pigment    — chlorophyll and pigments
  - conductivity_tds       — conductivity and total dissolved solids
  - turbidity_transparency — turbidity and transparency
  - other_chemistry        — catch-all for physicochemical measurements

family=B_env_categorical (categorical environmental descriptors; **modifier_bag must be []**):
  - habitat_biome          — habitat / biome / ecosystem type
  - material_medium_type   — sampled-medium type (sediment type / soil type / substrate / grain-size class)
  - env_state_context      — environmental state & landscape context (trophic level, water quality class, land use, climate, tidal state)
  - other_categorical      — catch-all for categorical descriptors

family=C_spatiotemporal (spatial / temporal identifiers; **modifier_bag must be []**):
  - vertical_position      — vertical measurements (depth / elevation / mixed_layer_depth / thermocline_depth / peat_depth)
  - time_point             — point-in-time (date / year / month / sampling_time)
  - sampling_site          — named sampling site (site / station / sampling_location)
  - geo_coord              — numeric coordinates (latitude / longitude)
  - time_duration          — duration or season (season / period / sampling_season)
  - geo_region             — named geographic region (country / region / basin / ocean)
  - spatial_metric         — area / distance / length / slope
  - water_body_descriptor  — descriptive water-body label (lake / ocean / river / surface / bottom)

family=D_other (catch-all; **modifier_bag must be []**):
  - other

========================= MODIFIER_BAG (family A only) =========================

Fixed vocabulary, 14 values:
  Position (5): surface | middle | bottom | porewater | mix
  Form     (5): dissolved | particulate | total | organic | inorganic
  Stat     (4): mean | max | min | range

**STRONG RULE — literal-only, no chemistry inference**:
  - Only include a modifier if its token (or its obvious abbreviation) is
    LITERALLY visible in the raw_key string.
  - Do NOT infer from chemistry/domain knowledge.

Examples:
  - "nitrate"                        → []             (nitrate IS chemically inorganic but the word "inorganic" is NOT in the string)
  - "total_phosphorus"               → [total]
  - "dissolved_inorganic_nitrogen"   → [dissolved, inorganic]
  - "surface_salinity"               → [surface]
  - "mean_annual_temperature"        → [mean]
  - "porewater_ph"                   → [porewater]

For B / C / D, modifier_bag MUST be [].

========================= QUANTITY_KIND (free-form) ===========================

Output a short snake_case label capturing the core measurement concept:
  - Base nouns: salinity / temperature / oxygen / phosphorus / nitrate / chlorophyll
  - Location/time: latitude / longitude / coordinates / depth / collection_date
  - Categorical: habitat_type / sediment_type / trophic_status
  - If you cannot determine: "other".

Do NOT append generic suffixes like "_value", "_concentration", "_amount",
"_level". Use the bare concept noun. For family C, use the field's own
meaning (e.g., "latitude", "mixed_layer_depth", "collection_date").

================================= FEW-SHOT ===================================

{few_shot_block}

========================= FIELD TO ANNOTATE =================================

- Field name:            {raw_key}
- Env PMID distribution: {env_pct}
- Total PMIDs:           {total_pmid}
- Evidence:
{evidence_block}

Return exactly one JSON object (terminated by </json>):
{{
  "family": "A_physicochemical | B_env_categorical | C_spatiotemporal | D_other",
  "subtype": "<one of the subtypes under the chosen family>",
  "quantity_kind": "<snake_case short label>",
  "modifier_bag": ["..."],
  "confidence": "high | medium | low",
  "reasoning": "one short English sentence"
}}
</json>"""


def _fmt_few_shot() -> str:
    lines = []
    for e in FEW_SHOT_EXAMPLES:
        mb = ("[" + ", ".join(e["modifier_bag"]) + "]") if e["modifier_bag"] else "[]"
        lines.append(
            f"- {e['raw_key']:34s} → family={e['family']}, "
            f"subtype={e['subtype']}, qk={e['quantity_kind']}, modifier_bag={mb}"
        )
    return "\n".join(lines)


# ── Evidence 构建（与 phase2 一致：优先非 table 段落） ──────────────
def _is_text_section(section_type: str) -> bool:
    st = (section_type or "").lower()
    return "table" not in st


def _build_evidence_index(raw_keys: set[str]) -> dict[str, list[dict]]:
    """每条 raw key 取最多 EVIDENCE_CAP 条 evidence，优先文本段落。"""
    logger.info("Building evidence index for %d keys…", len(raw_keys))
    with open(config.STEP2_INPUT, "r", encoding="utf-8") as f:
        records = json.load(f)

    text_bucket: dict[str, list[dict]] = {k: [] for k in raw_keys}
    table_bucket: dict[str, list[dict]] = {k: [] for k in raw_keys}

    for r in records:
        keys = r.get("metadata_keys_found") or []
        if not keys:
            continue
        pmid = r.get("pmid", "")
        section = r.get("section_type", "") or ""
        is_text = _is_text_section(section)
        for k in keys:
            if k not in text_bucket:
                continue
            bucket = text_bucket[k] if is_text else table_bucket[k]
            if len(bucket) >= EVIDENCE_CAP:
                continue
            if any(e["pmid"] == pmid for e in bucket):
                continue
            quote = (r.get("evidence_quote") or "")[:EVIDENCE_CHAR_CAP]
            bucket.append({"pmid": pmid, "quote": quote, "section": section})

    out: dict[str, list[dict]] = {}
    for k in raw_keys:
        combined = list(text_bucket[k])
        seen = {e["pmid"] for e in combined}
        if len(combined) < EVIDENCE_CAP:
            for e in table_bucket[k]:
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
        return "  (no evidence)"
    lines = []
    for e in evs:
        q = e["quote"].strip().replace("\n", " ")
        if len(q) > EVIDENCE_CHAR_CAP:
            q = q[:EVIDENCE_CHAR_CAP] + "…"
        lines.append(f"  [{e['pmid']}] {q}")
    return "\n".join(lines)


def _env_pct(row: pd.Series) -> str:
    total = max(1, int(row["total_pmid"]))
    parts = []
    for e in config.HYDRO_ENVS:
        pct = 100.0 * int(row[f"env_{e}"]) / total
        parts.append(f"{e}={pct:.0f}%")
    return ", ".join(parts)


def _load_checkpoint() -> dict[str, dict]:
    if not PHASE3_CHECKPOINT.exists():
        return {}
    out: dict[str, dict] = {}
    with open(PHASE3_CHECKPOINT, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                out[d["raw_key"]] = d
            except Exception:
                continue
    logger.info("Resumed %d checkpoint entries", len(out))
    return out


# ── 验证 / 清洗 LLM 输出 ────────────────────────────────────────────
_QK_CHAR_RE = re.compile(r"[^a-z0-9_]")


def _sanitize_qk(raw: str) -> str:
    s = str(raw or "").strip().lower()
    s = re.sub(r"[\s\-/]+", "_", s)
    s = _QK_CHAR_RE.sub("", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "other"


def _validate_parsed(parsed: dict, raw_key: str) -> dict:
    """把 LLM 输出规整为受约束的四槽位。"""
    family = str(parsed.get("family", "")).strip()
    subtype = str(parsed.get("subtype", "")).strip()
    qk = _sanitize_qk(parsed.get("quantity_kind", ""))
    mb_raw = parsed.get("modifier_bag", [])
    if not isinstance(mb_raw, list):
        mb_raw = []
    modifier_bag = [
        str(m).strip().lower() for m in mb_raw
        if isinstance(m, (str, int, float))
    ]
    modifier_bag = [m for m in modifier_bag if m in MODIFIER_BAG_CHOICES]

    # family 纠偏
    if family not in FAMILY_CHOICES:
        family = "D_other"
        subtype = "other"
        modifier_bag = []

    # subtype 纠偏：若子类不在当前 family 下，退到该 family 的 catch-all；
    # 若没有 catch-all（C 族），退到 D_other/other
    if subtype not in FAMILY_SCHEMA[family]:
        fb = _FAMILY_FALLBACK_SUBTYPE[family]
        if fb is not None:
            subtype = fb
        else:
            family = "D_other"
            subtype = "other"
            modifier_bag = []

    # modifier_bag 强约束：非 A 族必须空
    if family != "A_physicochemical":
        modifier_bag = []

    # 去重保持顺序
    seen = set()
    modifier_bag = [m for m in modifier_bag if not (m in seen or seen.add(m))]

    conf = str(parsed.get("confidence", "medium")).strip().lower()
    if conf not in {"high", "medium", "low"}:
        conf = "medium"

    return {
        "raw_key": raw_key,
        "family": family,
        "subtype": subtype,
        "quantity_kind": qk,
        "modifier_bag": modifier_bag,
        "confidence": conf,
        "reasoning": str(parsed.get("reasoning", "")).strip(),
    }


# ── LLM 调用 ────────────────────────────────────────────────────────
async def _classify_one(
    client: AsyncLocalModelClient,
    row: pd.Series,
    evidence: list[dict],
    few_shot_block: str,
) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        few_shot_block=few_shot_block,
        raw_key=row["raw_key"],
        env_pct=_env_pct(row),
        total_pmid=int(row["total_pmid"]),
        evidence_block=_fmt_evidence(evidence),
    )
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.chat(messages)
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
            return _validate_parsed(parsed, row["raw_key"])
        except Exception as e:  # noqa: BLE001
            logger.warning("attempt %d failed for %s: %s", attempt, row["raw_key"], e)
            await asyncio.sleep(backoff_with_jitter(attempt))
    # 兜底：失败
    return {
        "raw_key": row["raw_key"],
        "family": "D_other",
        "subtype": "other",
        "quantity_kind": "other",
        "modifier_bag": [],
        "confidence": "low",
        "reasoning": "FAILED: LLM call failed after retries",
    }


# ── 主入口 ──────────────────────────────────────────────────────────
def run() -> None:
    config.ensure_output_dir()

    logger.info("Loading env2_kept …")
    kept = pd.read_csv(config.PHASE2_KEPT)
    # 兼容：若未应用 raw_key_expansion，raw_key_original 列缺失，用 raw_key 自身填充
    if "raw_key_original" not in kept.columns:
        kept["raw_key_original"] = kept["raw_key"]
    limit = os.environ.get("PHASE3_LIMIT")
    if limit:
        kept = kept.head(int(limit)).copy()
        logger.info("PHASE3_LIMIT=%s → %d rows (smoke test)", limit, len(kept))
    logger.info("env2_kept: %d rows (expanded=%d)",
                len(kept), int((kept["raw_key"] != kept["raw_key_original"]).sum()))

    # evidence 用 raw_key_original 索引（step2 relation_output 里字段名是原始 raw_key）
    ev_idx = _build_evidence_index(set(kept["raw_key_original"].tolist()))

    done = _load_checkpoint()
    todo = [r for _, r in kept.iterrows() if r["raw_key"] not in done]
    logger.info("Main todo: %d (cached %d)", len(todo), len(done))

    api_key = os.environ.get("ALL_API_KEY")
    if not api_key:
        raise RuntimeError("ALL_API_KEY env var is required")

    results: list[dict] = list(done.values())
    few_shot_block = _fmt_few_shot()

    async def _go():
        sem = asyncio.Semaphore(CONCURRENCY)
        lock = asyncio.Lock()
        async with AsyncLocalModelClient(
            base_url=config.BASE_URL, model=config.MODEL,
            temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS,
            api_key=api_key, stop_sentinel=config.STOP_SENTINEL,
            api_style=config.API_STYLE, auth_mode=config.AUTH_MODE,
        ) as client:
            async def worker(row):
                async with sem:
                    # evidence 按原始 raw_key 查；prompt 字段名用展开后的 raw_key
                    ev = ev_idx.get(row["raw_key_original"], [])
                    res = await _classify_one(client, row, ev, few_shot_block)
                    async with lock:
                        with open(PHASE3_CHECKPOINT, "a", encoding="utf-8") as f:
                            f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    return res
            start = time.time()
            tasks = [asyncio.create_task(worker(r)) for r in todo]
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                results.append(await coro)
                if (i + 1) % 500 == 0:
                    rate = (i + 1) / max(1e-6, time.time() - start)
                    logger.info("Progress %d/%d (%.1f/s)", i + 1, len(todo), rate)

    asyncio.run(_go())

    # Merge results with kept dataframe
    res_df = pd.DataFrame(results)
    # modifier_bag 在 CSV 里用 "|" 连接；下游读取用 split("|") 恢复 list
    res_df["modifier_bag_str"] = res_df["modifier_bag"].apply(
        lambda x: "|".join(x) if isinstance(x, list) else ""
    )
    res_df = res_df.drop(columns=["modifier_bag"]).rename(
        columns={"modifier_bag_str": "modifier_bag"}
    )
    # 展开后可能多个原 raw_key 映射到同一新 raw_key（如 no3/no3-/no3− → nitrate）。
    # 并发 worker 会为每个 kept 行各跑一次 LLM → checkpoint 有同 raw_key 多条 →
    # merge 时笛卡尔积膨胀。这里按 raw_key 保留最后一条结果，1-to-many 回填到 kept。
    res_df = res_df.drop_duplicates(subset=["raw_key"], keep="last").reset_index(drop=True)

    # 只保留 env2 里需要的列做合并
    phase2_cols = [
        "raw_key", "raw_key_original",
        "env_Open_ocean", "env_Coastal_waters", "env_Lake",
        "env_Wetlands", "total_pmid", "n_envs_present",
    ]
    kept_subset = kept[[c for c in phase2_cols if c in kept.columns]].copy()
    merged = kept_subset.merge(res_df, on="raw_key", how="left")

    # 列顺序
    final_cols = phase2_cols + [
        "family", "subtype", "quantity_kind", "modifier_bag",
        "confidence", "reasoning",
    ]
    final_cols = [c for c in final_cols if c in merged.columns]
    merged[final_cols].to_csv(config.PHASE3_OUTPUT, index=False)
    logger.info("Wrote %d rows → %s", len(merged), config.PHASE3_OUTPUT)

    # 副产出：quantity_kind 频次（给 3a-norm 做输入）
    qk_counts = (
        merged["quantity_kind"].fillna("other").value_counts()
        .reset_index().rename(columns={"index": "quantity_kind", "count": "freq"})
    )
    qk_counts.columns = ["quantity_kind", "freq"]
    qk_counts.to_csv(config.PHASE3_QK_COUNTS, index=False)
    logger.info("Wrote %d unique qk → %s", len(qk_counts), config.PHASE3_QK_COUNTS)

    # ── Verification ────────────────────────────────────────────────
    fam_counts = merged["family"].value_counts().to_dict()
    fam_pmid = merged.groupby("family")["total_pmid"].sum().to_dict()
    total_pmid_all = int(merged["total_pmid"].sum())
    fam_pmid_pct = {
        k: round(100.0 * v / max(1, total_pmid_all), 2) for k, v in fam_pmid.items()
    }

    sub_counts = (
        merged.groupby(["family", "subtype"]).size()
        .reset_index(name="count")
        .sort_values(["family", "count"], ascending=[True, False])
    )

    conf_counts = merged["confidence"].value_counts().to_dict()
    low_pct = round(100.0 * conf_counts.get("low", 0) / max(1, len(merged)), 2)

    # modifier_bag 约束检查：B/C/D 必须空
    bad_mb_mask = (merged["family"] != "A_physicochemical") & \
                  (merged["modifier_bag"].astype(str).str.len() > 0)
    bad_mb_count = int(bad_mb_mask.sum())

    failed_mask = merged["reasoning"].astype(str).str.startswith("FAILED")
    failed_count = int(failed_mask.sum())

    stats = {
        "total_rows": int(len(merged)),
        "failed": failed_count,
        "family_count": fam_counts,
        "family_pmid_pct": fam_pmid_pct,
        "confidence_breakdown": conf_counts,
        "low_confidence_pct": low_pct,
        "bad_modifier_bag_count (non-A with non-empty bag, should be 0)": bad_mb_count,
        "unique_quantity_kind": int(len(qk_counts)),
        "top10_quantity_kind": qk_counts.head(10).to_dict(orient="records"),
    }

    logger.info("=" * 60)
    logger.info("PHASE 3 VERIFICATION")
    logger.info("=" * 60)
    logger.info(json.dumps(stats, ensure_ascii=False, indent=2))
    logger.info("")
    logger.info("Subtype 分布（前 20）:")
    logger.info("\n%s", sub_counts.head(20).to_string(index=False))
    logger.info("")
    logger.info("重点 case 抽检:")
    for k in [
        "salinity", "surface_salinity", "bottom_salinity",
        "dissolved_oxygen", "oxygen_saturation",
        "nitrate", "total_dissolved_phosphorus", "total_phosphorus",
        "mixed_layer_depth", "collection_date", "latitude", "longitude",
        "habitat", "sediment_type", "lake_area",
    ]:
        row = merged[merged["raw_key"] == k]
        if len(row):
            r = row.iloc[0]
            logger.info(
                "  %-35s family=%-20s subtype=%-26s qk=%-22s mb=%s",
                k, r.family, r.subtype, r.quantity_kind, r.modifier_bag or "[]",
            )
