"""环节 2：非环境 raw key 过滤（LLM 二分类）。

流程：
    1. 从 03b_mapping_review_decisions.json 提取 245 条 EXCLUDE、从
       02_mapping_full.csv 提取 1,275 条 LLM-mapped 作为正反面样本池。
    2. 与 env1 mainstream 取交集，分层随机构造 eval split：
        - few_shot_exclude: 35 条（EXCLUDE 分层采样）
        - few_shot_keep:    10 条（LLM-mapped 随机）
        - eval_exclude:     剩余 EXCLUDE（用于召回率）
        - eval_keep:        200 条 LLM-mapped（用于误杀率）
       所有四组互不相交；split 固化到 env2_eval_split.json 保证可复现。
    3. 对 mainstream 6,140 条 raw key 逐条跑 DeepSeek-V3 二分类：
        {is_env_metadata, category, reasoning}
    4. 输出 env2_kept / env2_excluded CSV，并在 eval 子集上计算 recall / FN。

Evidence 抽取：扫 step2 记录，对每条 raw key 取前 EVIDENCE_CAP 条含该 key 的段落
（整词精确匹配 metadata_keys_found），evidence_quote 截断到 EVIDENCE_CHAR_CAP 字符。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
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
FEW_SHOT_EXCLUDE_N = 35   # legacy: 仅在首次构建 eval_split 时用于分层采样
FEW_SHOT_KEEP_N = 10      # legacy
EVAL_KEEP_N = 200         # legacy
EVIDENCE_CAP = 2
EVIDENCE_CHAR_CAP = 600
MAX_RETRIES = 3
RANDOM_SEED = 42
CONCURRENCY = 24
CHECKPOINT_FLUSH_EVERY = 50

PHASE2_CHECKPOINT: Path = config.OUTPUT_DIR / "env2_classification.checkpoint.jsonl"

# Few-shot 改为硬编码合成示例（不再从 eval_split 读，避免与评测集泄漏）
# 口径与项目决定一致：sampling_tool 视为 KEEP，所以 filter_pore_size / station_id 放 KEEP 侧。
HARDCODED_FEW_SHOT_KEEP: list[dict] = [
    {"field": "salinity",
     "reason": "Water salinity is a physicochemical state of the sampled medium."},
    {"field": "tidal_stage",
     "reason": "Tidal stage is an environmental categorical descriptor at sampling time."},
    {"field": "collection_date",
     "reason": "Collection date records when sampling happened."},
    {"field": "filter_pore_size",
     "reason": "Filter pore size defines the fraction of the sampled medium (sampling_tool scope, kept)."},
    {"field": "station_id",
     "reason": "Sampling station identifier is kept as a location descriptor (project scope)."},
]
HARDCODED_FEW_SHOT_EXCLUDE: list[dict] = [
    {"field": "sampling_duration",
     "reason": "Sampling duration is a study-design parameter, not an in-situ environmental state."},
    {"field": "bacterial_abundance",
     "reason": "Bacterial abundance is a post-sampling analysis output, not environmental state."},
    {"field": "site_description",
     "reason": "Free-text site description is a descriptive field, not a structured metadata measurement."},
]

# legacy keyword map, only referenced by the offline eval_split builder.
CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("analysis_result", ["丰度", "计数", "拷贝数", "生物量", "覆盖度", "reads", "qPCR",
                         "菌落", "OTU", "ASV", "物种", "群落", "比值"]),
    ("identifier", ["编号", "航次", "数据集", "站点编号", "站点名称", "site_id",
                    "Cruise", "国家代码", "Project"]),
    ("processing", ["过滤", "孔径", "试剂", "提取", "保存", "采样工具", "样品保存",
                    "pore", "filter", "preservation", "库建", "测序", "reagent"]),
    ("study_design", ["实验", "模拟", "处理参数", "treatment", "experimental",
                      "施加", "添加", "研究设计", "培养"]),
    ("bio_object", ["体长", "体重", "size", "生物个体", "年龄", "性别", "身长",
                    "宿主", "prawn", "urchin", "stem", "植物", "生物属性", "龄"]),
    ("descriptive", ["描述", "自由文本", "综合描述", "定性", "注释"]),
]


def _cat_from_reason(reason: str) -> str:
    r = reason or ""
    for cat, kws in CATEGORY_RULES:
        if any(kw in r for kw in kws):
            return cat
    return "other"


# ── 数据加载 ────────────────────────────────────────────────────────

def _load_mainstream() -> pd.DataFrame:
    df = pd.read_csv(config.PHASE1_MAINSTREAM)
    logger.info("mainstream loaded: %d rows", len(df))
    return df


def _load_exclude_pool() -> list[dict]:
    path = config.DESIGN_REVIEW_DIR / "03b_mapping_review_decisions.json"
    with open(path, "r", encoding="utf-8") as f:
        all_dec = json.load(f)
    out = [d for d in all_dec if d.get("action") == "EXCLUDE"]
    logger.info("EXCLUDE decisions loaded: %d", len(out))
    return out


def _load_llm_mapped_pool() -> list[str]:
    path = config.DESIGN_REVIEW_DIR / "02_mapping_full.csv"
    df = pd.read_csv(path)
    df = df[df["mapping_type"] == "LLM-mapped"]
    fields = df["canonical"].dropna().astype(str).unique().tolist()
    logger.info("LLM-mapped positives loaded: %d", len(fields))
    return fields


def _is_text_section(section_type: str) -> bool:
    """table / supplementary 类段落通常只有字段-数值对，信息量低；
    METHODS / RESULTS / INTRODUCTION 等文本段落有更丰富的上下文。"""
    st = (section_type or "").lower()
    return "table" not in st


def _build_evidence_index(raw_keys: set[str]) -> dict[str, list[dict]]:
    """对每条 raw key 优先收集文本段落作为 evidence，不够再用 table 补。

    策略：一次扫描，分别累积 text bucket 和 table bucket（各上限 EVIDENCE_CAP），
    最后合并时优先文本；table 只在文本不够时作为 fallback。
    """
    logger.info("Building evidence index for %d keys (scanning step2)…", len(raw_keys))
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

    # 合并：文本优先，不够用 table 补（且去重 pmid）
    out: dict[str, list[dict]] = {}
    text_backed = table_backed = mixed = none_count = 0
    for k in raw_keys:
        combined = list(text_bucket[k])
        seen_pmids = {e["pmid"] for e in combined}
        if len(combined) < EVIDENCE_CAP:
            for e in table_bucket[k]:
                if len(combined) >= EVIDENCE_CAP:
                    break
                if e["pmid"] in seen_pmids:
                    continue
                combined.append(e)
                seen_pmids.add(e["pmid"])
        out[k] = combined
        if not combined:
            none_count += 1
        elif all("table" not in (e.get("section", "") or "").lower() for e in combined):
            text_backed += 1
        elif all("table" in (e.get("section", "") or "").lower() for e in combined):
            table_backed += 1
        else:
            mixed += 1
    logger.info("Evidence index built: text-only=%d, table-only=%d, mixed=%d, empty=%d",
                text_backed, table_backed, mixed, none_count)
    return out


# ── eval split 构造（固化到 json 以可复现） ─────────────────────────

def _build_eval_split(
    mainstream: pd.DataFrame,
    exclude_pool: list[dict],
    llm_mapped_pool: list[str],
) -> dict[str, Any]:
    mainstream_set = set(mainstream["raw_key"].tolist())

    excl_eligible = [d for d in exclude_pool if d["field"] in mainstream_set]
    logger.info("EXCLUDE ∩ mainstream = %d (of %d)", len(excl_eligible), len(exclude_pool))

    # 分层：按 reason 关键词归类
    buckets: dict[str, list[dict]] = defaultdict(list)
    for d in excl_eligible:
        buckets[_cat_from_reason(d.get("reason", ""))].append(d)
    logger.info("EXCLUDE category buckets: %s",
                {k: len(v) for k, v in buckets.items()})

    rng = random.Random(RANDOM_SEED)
    few_shot_exclude: list[dict] = []
    # 按桶比例抽取 few-shot
    total = len(excl_eligible)
    for cat, items in sorted(buckets.items()):
        per = max(1, round(len(items) / total * FEW_SHOT_EXCLUDE_N))
        rng.shuffle(items)
        take = min(per, len(items))
        few_shot_exclude.extend(items[:take])
    if len(few_shot_exclude) > FEW_SHOT_EXCLUDE_N:
        few_shot_exclude = few_shot_exclude[:FEW_SHOT_EXCLUDE_N]
    fs_excl_set = {d["field"] for d in few_shot_exclude}

    # eval_exclude = 剩余
    eval_exclude = [d for d in excl_eligible if d["field"] not in fs_excl_set]

    # keep pool
    keep_eligible = [f for f in llm_mapped_pool if f in mainstream_set]
    logger.info("LLM-mapped ∩ mainstream = %d", len(keep_eligible))
    rng2 = random.Random(RANDOM_SEED + 1)
    keep_shuffled = keep_eligible[:]
    rng2.shuffle(keep_shuffled)
    few_shot_keep = keep_shuffled[:FEW_SHOT_KEEP_N]
    eval_keep = keep_shuffled[FEW_SHOT_KEEP_N:FEW_SHOT_KEEP_N + EVAL_KEEP_N]

    split = {
        "seed": RANDOM_SEED,
        "few_shot_exclude": few_shot_exclude,
        "few_shot_keep": few_shot_keep,
        "eval_exclude": eval_exclude,
        "eval_keep": eval_keep,
        "stats": {
            "excl_eligible": len(excl_eligible),
            "keep_eligible": len(keep_eligible),
            "few_shot_exclude": len(few_shot_exclude),
            "few_shot_keep": len(few_shot_keep),
            "eval_exclude": len(eval_exclude),
            "eval_keep": len(eval_keep),
            "bucket_sizes": {k: len(v) for k, v in buckets.items()},
        },
    }
    with open(config.PHASE2_EVAL_SPLIT, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    logger.info("Eval split saved to %s", config.PHASE2_EVAL_SPLIT)
    return split


def _load_or_build_split(mainstream: pd.DataFrame) -> dict[str, Any]:
    if config.PHASE2_EVAL_SPLIT.exists():
        logger.info("Reusing existing eval split at %s", config.PHASE2_EVAL_SPLIT)
        with open(config.PHASE2_EVAL_SPLIT, "r", encoding="utf-8") as f:
            return json.load(f)
    return _build_eval_split(mainstream, _load_exclude_pool(), _load_llm_mapped_pool())


# ── Prompt ──────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
You are an expert curator of environmental metadata for hydrosphere studies
(open ocean, coastal waters, lakes, wetlands). Your task is to decide whether
the given field name represents **the environmental state of the sampled
medium** (is_env_metadata=true) or something else (is_env_metadata=false).

Context: the field was extracted by an earlier LLM from methods/sample
description sections of papers on aquatic environments. The field appears
alongside biosample accession records.

=========================== KEEP (is_env_metadata=true) ===========================

1. **physicochemical** — physico-chemical state of the sampled medium:
   - pH (incl. pH_total_scale, pH_nbs), temperature (water/sediment/ambient),
     salinity, conductivity, pressure, density, turbidity, redox potential.
   - Dissolved gases: dissolved oxygen (DO, O2), dissolved CO2.
   - Ions/nutrients: NH4, NO3, NO2, PO4, SiO3, HCO3, Na, K, Mg, Ca, Cl, SO4,
     and metals (Fe, Cu, Zn, As, Pb, Cd, Hg, etc.).
   - Aggregated nutrients: TN, TP, DIN, DIP, DON, DOP, DOC, POC, TOC,
     dissolved/particulate fractions, ammoniacal nitrogen.
   - **Lab-measured characterizations of the sampled matrix** (these still
     describe the environment's state, not the study organism):
       chlorophyll a/b, phaeopigments, fluorescence indices, in-vivo
       fluorescence, loss-on-ignition, sediment/soil organic matter,
       moisture content, organic carbon content, hydrocarbons (TPH, PAHs)
       in water/sediment as pollutants, particulate carbon.

2. **location** — spatial descriptors of where sampling occurred:
   - Coordinates in any form: latitude/longitude/decimalLatitude/lat_wgs84/
     coordinates_wgs84/geographical_position/geoposition/northing.
   - Depths: sampling depth, water depth, bottom depth, depth_below_seafloor,
     meters_below_sea_level, core depth, DCM depth, instrument depth,
     maximal/minimum depth.
   - Elevations/altitudes: above/below sea level, lake_elevation.
   - Named places and regions: locality, localities, station, station12,
     coastal_location, wetland_location, geographical_region, country,
     ocean_and_sea_region, capture_location.
   - Note: `country` as a descriptor is KEPT; pure identifier `country_code`
     is EXCLUDED (see below).

3. **time** — when sampling/observation happened:
   - Collection date, sampling date/time, observation date/time, timestamp,
     month, month_year, year, season, sampling_times, catch_date,
     date_year, date_of_measurement.

4. **env_categorical** — categorical descriptors of environmental state:
   - Tidal stage/phase/state/cycle/type/condition.
   - Habitat type, wetland type, ecosystem type, water type.
   - Sediment type, substrate/substratum, bottom sediment, grain size
     class, textural group/class, silt/clay, sorting, granulometry.
   - Weather, hydrodynamic conditions, trophic status, water quality class.

5. **sampling_tool** — reserve for parameters that define the fraction
   of environment being sampled (rare):
   - filter_pore_size / mesh_size / filter_sizes / fraction_upper when
     used to define the size fraction characterized (borderline — keep
     when the paper uses them to describe what part of the water column
     or biota-size fraction was sampled).

=========================== EXCLUDE (is_env_metadata=false) ===========================

1. **analysis_result** — measurements of the biological community being
   STUDIED (outcomes of the analysis, not the environment):
   - Microbial/biological counts and abundance: bacterial counts,
     viral counts/abundance, algal density, zooplankton/phytoplankton
     abundance, cell abundance, colony counts, E. coli / coliform /
     enterococci counts, heterotrophic bacteria counts, picoeukaryote
     abundance, individual counts.
   - Biomass of target organisms: biomass, aboveground/belowground
     biomass, cyanobacterial biomass.
   - Sequencing QC: reads, contigs, N50, ORFs, genome_assembly.
   - KEY DISTINCTION: if the measurement describes **the biota living
     in the sample** → EXCLUDE. If it describes **the physical/chemical
     state of the water/sediment/soil** → KEEP.

2. **identifier** — pure identifiers that do not describe environmental
   state:
   - sampleid, library name, accession IDs, country_code (as opposed
     to "country"), dataset names, cruise names, project IDs,
     source_material_identifier.

3. **processing** — sampling and sample-processing parameters:
   - Filter volume, filtered_volume, sample mass (mass_g), preservation
     method, DNA extraction, sequencing method/primers.
   - Sampling procedure/protocol/methods (process descriptions).
   - Sampling frequency (study-design cadence, not env state).

4. **study_design** — research design / experiment descriptors:
   - Experiment name, treatment, study_duration, geographic_coverage,
     anthropogenic_disturbances (a study descriptor, not an in-situ
     measurement), simulated/manipulated conditions.

5. **bio_object** — taxonomic/developmental/morphometric attributes of
   the study organism itself:
   - age, age_class, sex, life stage, individual body size, carapace
     length, stem length, prawn length, urchin size.
   - biota (as subject class label).

6. **descriptive** — free-text or highly generalized summary fields:
   - Free-text notes, comments, summary fields, qualitative observations.

7. **contaminated/ambiguous generalized fields (→ other)** — fields whose
   aliases mix incompatible semantics, or whose meaning cannot be pinned
   down from evidence:
   - e.g. `density` that aggregates bacterial density + human population
     density + physical density — too polluted to use as a single metadata.
   - e.g. `area` mixing geographic area + study area name + sampling
     surface.
   - e.g. heavy abbreviations like `map` (mean annual precipitation? map?)
     with insufficient evidence.
   - When a field's alias set shows severe semantic mixing, EXCLUDE with
     category=other.

=============================== BOUNDARY CASES ================================

- Abbreviations: infer the specific meaning from evidence. If genuinely
  ambiguous (e.g. 'do', 'mat', 'sd'), set confidence=low.
- Chlorophyll a/b/phaeopigments: KEEP (matrix chemistry, not a count).
- TOC / LOI / sediment OM / DOC / POC: KEEP (matrix chemistry).
- Fluorescence indices / in-vivo fluorescence: KEEP.
- "country" (descriptor) vs "country_code" (ID): KEEP vs EXCLUDE.
- "station", "station12": KEEP (location descriptors).
- Depth variants (sampling/water/below_seafloor/core/DCM/etc.): KEEP all
  — the next stage (4-tuple extraction) disambiguates them.
- If aliases are severely contaminated with unrelated semantics, EXCLUDE
  with category=other even if the "base" meaning would otherwise be KEPT.

=============================== FEW-SHOT EXAMPLES =============================

[KEEP — is_env_metadata=true]
{few_shot_keep}

[EXCLUDE — is_env_metadata=false]
{few_shot_exclude}

=============================== FIELD TO JUDGE ==============================

- Field name:            {raw_key}
- Env PMID distribution: {env_pct}
- Total PMIDs:           {total_pmid}
- Evidence paragraphs:
{evidence_block}

Return exactly one JSON object (terminated by </json>):
{{
  "is_env_metadata": true | false,
  "confidence": "high | medium | low",
  "reasoning": "one short English sentence"
}}
</json>"""


def _fmt_few_shot_exclude(items: list[dict]) -> str:
    lines = []
    for d in items:
        r = (d.get("reason") or "").strip().replace("\n", " ")
        if len(r) > 120:
            r = r[:117] + "…"
        lines.append(f"- {d['field']} — {r}")
    return "\n".join(lines)


def _fmt_few_shot_keep(items: list) -> str:
    """支持两种输入：list[str]（旧，仅字段名）或 list[dict]（新，含 reason）。"""
    lines = []
    for it in items:
        if isinstance(it, dict):
            r = (it.get("reason") or "").strip().replace("\n", " ")
            if len(r) > 140:
                r = r[:137] + "…"
            lines.append(f"- {it['field']} — {r}")
        else:
            lines.append(f"- {it}")
    return "\n".join(lines)


def _fmt_evidence(evs: list[dict]) -> str:
    if not evs:
        return "  (无可用 evidence)"
    out = []
    for e in evs:
        q = e["quote"].strip().replace("\n", " ")
        if len(q) > EVIDENCE_CHAR_CAP:
            q = q[:EVIDENCE_CHAR_CAP] + "…"
        out.append(f"  [{e['pmid']}] {q}")
    return "\n".join(out)


def _env_pct(row: pd.Series) -> str:
    total = max(1, int(row["total_pmid"]))
    parts = []
    for e in config.HYDRO_ENVS:
        pct = 100.0 * int(row[f"env_{e}"]) / total
        parts.append(f"{e}={pct:.0f}%")
    return ", ".join(parts)


def _build_prompt(row: pd.Series, evidence: list[dict],
                  few_shot_exclude: str, few_shot_keep: str) -> str:
    return PROMPT_TEMPLATE.format(
        few_shot_keep=few_shot_keep,
        few_shot_exclude=few_shot_exclude,
        raw_key=row["raw_key"],
        env_pct=_env_pct(row),
        total_pmid=int(row["total_pmid"]),
        evidence_block=_fmt_evidence(evidence),
    )


# ── LLM 调用 ────────────────────────────────────────────────────────

async def _classify_one(
    client: AsyncLocalModelClient,
    row: pd.Series,
    evidence: list[dict],
    fs_excl: str,
    fs_keep: str,
) -> Optional[dict]:
    prompt = _build_prompt(row, evidence, fs_excl, fs_keep)
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
            if parsed is None:
                await asyncio.sleep(backoff_with_jitter(attempt))
                continue
            if isinstance(parsed, list):
                # 模型偶尔回一个 array，取第一个 dict
                parsed = next((x for x in parsed if isinstance(x, dict)), None)
                if parsed is None:
                    await asyncio.sleep(backoff_with_jitter(attempt))
                    continue
            if not isinstance(parsed, dict):
                await asyncio.sleep(backoff_with_jitter(attempt))
                continue
            is_env = parsed.get("is_env_metadata")
            if isinstance(is_env, str):
                is_env = is_env.lower() == "true"
            return {
                "raw_key": row["raw_key"],
                "is_env_metadata": bool(is_env),
                "confidence": str(parsed.get("confidence", "medium")).lower(),
                "reasoning": parsed.get("reasoning", ""),
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("attempt %d failed for %s: %s", attempt, row["raw_key"], e)
            if os.environ.get("PHASE2_DEBUG"):
                try:
                    logger.warning("  raw resp[:800]: %r", (resp or "")[:800])
                except Exception:
                    pass
            await asyncio.sleep(backoff_with_jitter(attempt))
    return {
        "raw_key": row["raw_key"],
        "is_env_metadata": None,
        "confidence": "low",
        "reasoning": "FAILED: LLM call failed after retries",
    }


def _load_checkpoint() -> dict[str, dict]:
    if not PHASE2_CHECKPOINT.exists():
        return {}
    out = {}
    with open(PHASE2_CHECKPOINT, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                out[d["raw_key"]] = d
            except Exception:
                continue
    logger.info("Resumed %d checkpoint entries", len(out))
    return out


async def _run_llm_pipeline(
    mainstream: pd.DataFrame,
    evidence_idx: dict[str, list[dict]],
    split: dict[str, Any],  # kept for signature compat; few-shot now hardcoded
) -> list[dict]:
    # 使用硬编码合成示例，不再从 eval_split 读取（避免 few-shot 与评测集污染同源）
    fs_excl = _fmt_few_shot_exclude(HARDCODED_FEW_SHOT_EXCLUDE)
    fs_keep = _fmt_few_shot_keep(HARDCODED_FEW_SHOT_KEEP)
    logger.info("Few-shot (hardcoded): KEEP=%d, EXCLUDE=%d",
                len(HARDCODED_FEW_SHOT_KEEP), len(HARDCODED_FEW_SHOT_EXCLUDE))
    done = _load_checkpoint()
    results: list[dict] = list(done.values())

    todo_rows = [r for _, r in mainstream.iterrows() if r["raw_key"] not in done]
    logger.info("To process: %d (already done: %d)", len(todo_rows), len(done))

    api_key = os.environ.get("ALL_API_KEY", "")
    if not api_key:
        raise RuntimeError("ALL_API_KEY env var is required")

    sem = asyncio.Semaphore(CONCURRENCY)
    ckpt_lock = asyncio.Lock()

    async with AsyncLocalModelClient(
        base_url=config.BASE_URL,
        model=config.MODEL,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        api_key=api_key,
        stop_sentinel=config.STOP_SENTINEL,
        api_style=config.API_STYLE,
        auth_mode=config.AUTH_MODE,
    ) as client:

        async def worker(row: pd.Series) -> dict:
            async with sem:
                ev = evidence_idx.get(row["raw_key"], [])
                res = await _classify_one(client, row, ev, fs_excl, fs_keep)
                async with ckpt_lock:
                    with open(PHASE2_CHECKPOINT, "a", encoding="utf-8") as f:
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
                return res

        tasks = [asyncio.create_task(worker(r)) for r in todo_rows]
        start = time.time()
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            res = await coro
            results.append(res)
            if (i + 1) % 200 == 0:
                rate = (i + 1) / max(1e-6, time.time() - start)
                logger.info("Progress: %d/%d (%.1f/s)", i + 1, len(todo_rows), rate)

    return results


# ── 评测 ────────────────────────────────────────────────────────────

def _compute_eval_metrics(
    results: list[dict],
    split: dict[str, Any],
) -> dict[str, Any]:
    res_by_key = {r["raw_key"]: r for r in results}

    eval_excl_keys = [d["field"] for d in split["eval_exclude"]]
    eval_keep_keys = [f for f in split["eval_keep"]]

    def rates(keys: list[str], expected_is_env: bool):
        hits = 0
        missing = 0
        for k in keys:
            r = res_by_key.get(k)
            if r is None:
                missing += 1
                continue
            pred = r.get("is_env_metadata")
            if pred is None:
                missing += 1
                continue
            if pred == expected_is_env:
                hits += 1
        n = max(1, len(keys) - missing)
        return {
            "total": len(keys),
            "missing_or_failed": missing,
            "correct": hits,
            "rate_pct": round(100.0 * hits / n, 2),
        }

    recall = rates(eval_excl_keys, expected_is_env=False)
    precision_on_positives = rates(eval_keep_keys, expected_is_env=True)
    false_negative_rate_pct = round(100.0 - precision_on_positives["rate_pct"], 2)

    return {
        "recall_on_historical_EXCLUDE": recall,
        "correct_on_historical_KEEP": precision_on_positives,
        "false_negative_rate_pct": false_negative_rate_pct,
        "target_recall_pct": 85.0,
        "target_false_negative_pct": 5.0,
    }


# ── 主入口 ──────────────────────────────────────────────────────────

def run() -> None:
    config.ensure_output_dir()
    mainstream_full = _load_mainstream()
    split = _load_or_build_split(mainstream_full)
    limit = os.environ.get("PHASE2_LIMIT")
    if limit:
        n = int(limit)
        mainstream = mainstream_full.head(n).copy()
        logger.info("PHASE2_LIMIT=%d → smoke-test mode, processing %d rows", n, len(mainstream))
    else:
        mainstream = mainstream_full
    ev_idx = _build_evidence_index(set(mainstream["raw_key"].tolist()))

    results = asyncio.run(_run_llm_pipeline(mainstream, ev_idx, split))

    # 合并回 mainstream
    res_df = pd.DataFrame(results)
    merged = mainstream.merge(res_df, on="raw_key", how="left")

    kept = merged[merged["is_env_metadata"] == True].copy()  # noqa: E712
    excluded = merged[merged["is_env_metadata"] != True].copy()

    # excluded 侧重命名 reasoning → excluded_reason（仅列名差异）
    if "reasoning" in excluded.columns:
        excluded = excluded.rename(columns={"reasoning": "excluded_reason"})

    kept.to_csv(config.PHASE2_KEPT, index=False)
    excluded.to_csv(config.PHASE2_EXCLUDED, index=False)
    logger.info("Wrote %d kept → %s", len(kept), config.PHASE2_KEPT)
    logger.info("Wrote %d excluded → %s", len(excluded), config.PHASE2_EXCLUDED)

    # 失败兜底：is_env_metadata is None 或 reasoning 以 FAILED 开头
    failed_mask = merged["is_env_metadata"].isna() | \
                  merged["reasoning"].astype(str).str.startswith("FAILED", na=False)
    # low-confidence review bucket（便于人工兜底）
    low_conf = merged[merged["confidence"] == "low"].copy()
    low_conf_path = config.OUTPUT_DIR / "env2_low_confidence_review.csv"
    low_conf.to_csv(low_conf_path, index=False)
    logger.info("Low-confidence entries (review candidate): %d → %s",
                len(low_conf), low_conf_path)

    metrics = _compute_eval_metrics(results, split)
    report = {
        "total_classified": len(merged),
        "kept": int(len(kept)),
        "excluded": int(len(excluded)),
        "failed_or_missing": int(failed_mask.sum()),
        "confidence_breakdown": dict(Counter(merged["confidence"].astype(str))),
        "low_confidence_count": int(len(low_conf)),
        "eval": metrics,
    }
    with open(config.PHASE2_EVAL_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("PHASE 2 VERIFICATION")
    logger.info("=" * 60)
    logger.info(json.dumps(report, ensure_ascii=False, indent=2))
