"""环节 3a-norm：quantity_kind 规范化合并。

双入口：
    3-norm-propose: 跑规则层 + LLM 层 →
                    env3_qk_merge_rules.csv (自动 approved)
                    env3_norm_review_queue.csv (pending 人审)
    3-norm-apply:   读人审过的 review_queue →
                    env3_final_annotations.csv
                    （同时把 env3_structured_annotations.csv 备份为 .pre_norm
                    并覆盖为 final 内容，让下游 phase4 无改动直接生效）

规范名选择：最高频 → 最短 → 字典序。
跨 family 合并：直接拒绝。
modifier_bag / family / subtype 字段不改（决策 9=A）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
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
BATCH_SIZE_CAP = 60   # 单批 qk 上限，超过切多批
MIN_SUBTYPE_QK = 4    # 同 subtype 下 qk <= MIN_SUBTYPE_QK 跳过 LLM

# 规则层 — 修饰词后缀（纯修饰，剥离后不会误伤实义）
SUFFIXES = (
    "_concentration", "_concentrations",
    "_value", "_values",
    "_amount", "_amounts",
    "_measurement", "_measurements",
)
# 规则层 — 修饰词前缀
PREFIXES = (
    "concentration_of_", "content_of_", "level_of_",
    "total_", "dissolved_", "particulate_",
)

# Anti-examples（5 条，写入 LLM prompt 强制 LLM 遵守）
ANTI_EXAMPLES: list[tuple[str, str, str]] = [
    ("oxygen", "oxygen_saturation",
     "concentration (mg/L) vs saturation (%) — different measurements"),
    ("phosphorus_total", "phosphorus_particulate",
     "whole vs sub-fraction — must stay separate"),
    ("nitrate", "nitrogen_total",
     "single species (NO3) vs sum of all N forms — must stay separate"),
    ("water_depth", "sediment_depth",
     "different sampled matrix (water column vs sediment core)"),
    ("latitude", "longitude",
     "orthogonal coordinate axes — never merge"),
]

# 输出文件路径（补充在 config 里加常量；此处先用字符串）
PHASE3_MERGE_RULES: Path = config.OUTPUT_DIR / "env3_qk_merge_rules.csv"
PHASE3_REVIEW_QUEUE: Path = config.OUTPUT_DIR / "env3_norm_review_queue.csv"
PHASE3_FINAL: Path = config.OUTPUT_DIR / "env3_final_annotations.csv"
PHASE3_NORM_CHECKPOINT: Path = config.OUTPUT_DIR / "env3_norm_llm.checkpoint.jsonl"
PHASE3_STRUCTURED_BAK: Path = config.OUTPUT_DIR / "env3_structured_annotations.pre_norm.csv"


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


# ── 规则层 ─────────────────────────────────────────────────────────
def _normalize_variants(qk: str) -> set[str]:
    """返回 qk 在规则层下的可映射候选形式（含自身）。"""
    variants = {qk}
    # 后缀剥离
    for suf in SUFFIXES:
        if qk.endswith(suf):
            base = qk[:-len(suf)]
            if base:
                variants.add(base)
    # 前缀剥离
    for pre in PREFIXES:
        if qk.startswith(pre):
            base = qk[len(pre):]
            if base:
                variants.add(base)
    # 单复数折叠：去末尾 s（长度>3 防误伤）
    if qk.endswith("s") and len(qk) > 3 and not qk.endswith("ss"):
        variants.add(qk[:-1])
    # 介词换位：X_of_Y -> Y_X, X_Y
    if "_of_" in qk:
        left, _, right = qk.partition("_of_")
        if left and right:
            variants.add(f"{right}_{left}")
            variants.add(f"{left}_{right}")
    return variants


def _pick_representative(members: list[str], freq_map: dict[str, int]) -> str:
    """最高频 → 最短 → 字典序。"""
    return sorted(members, key=lambda m: (-freq_map.get(m, 0), len(m), m))[0]


def _rule_layer_merge(counts_df: pd.DataFrame) -> tuple[dict[str, str], list[dict]]:
    """
    返回:
        merge_map: dict[source_qk -> target_qk]（source 可能 == target）
        merge_log: 合并决策列表（仅列 source != target 的）
    """
    qk_set = set(counts_df["quantity_kind"].astype(str))
    freq_map = dict(zip(counts_df["quantity_kind"].astype(str),
                        counts_df["freq"].astype(int)))
    uf = UF(qk_set)
    for qk in qk_set:
        for v in _normalize_variants(qk):
            if v != qk and v in qk_set:
                uf.union(qk, v)
    clusters = uf.components()
    merge_map: dict[str, str] = {}
    merge_log: list[dict] = []
    for members in clusters.values():
        target = _pick_representative(members, freq_map)
        for m in members:
            merge_map[m] = target
            if m != target:
                merge_log.append({
                    "source_qk": m, "target_qk": target,
                    "freq_source": freq_map.get(m, 0),
                    "layer": "rule",
                    "reason": "rule-layer literal variant",
                })
    logger.info("Rule-layer: %d unique qk → %d clusters (%d merges)",
                len(qk_set), len(clusters), len(merge_log))
    return merge_map, merge_log


# ── LLM 层 prompt ──────────────────────────────────────────────────
LLM_BATCH_PROMPT = """\
You are reviewing a list of `quantity_kind` labels generated by an earlier
LLM-based annotator for hydrosphere environmental metadata. These are
free-form snake_case labels, many of which are synonyms or near-synonyms
that should be merged into one canonical label.

All labels below belong to the same family=**{family}** and
subtype=**{subtype}**.

================= STRICT ANTI-MERGE RULES (MUST NOT MERGE) =================

{anti_examples}

General principle:
  - If two labels measure the same physical/chemical/spatial/temporal quantity
    but differ only in surface form (e.g., ordering, pluralization, extra
    descriptive word), merge them.
  - If two labels measure different quantities (concentration vs saturation,
    whole vs fraction, different matrices, different axes), DO NOT merge —
    even if their names look similar.
  - When in doubt, DO NOT merge.

================= CANDIDATE QUANTITY_KINDS (family={family}, subtype={subtype}) =================

{qk_block}

================= OUTPUT =================

Return a JSON object:
{{
  "merge_proposals": [
    {{
      "target_name": "canonical label (pick from the members, or propose a new short snake_case)",
      "members": ["qk1", "qk2", ...],
      "reason": "one short English sentence"
    }}
  ]
}}

If no merges are warranted, return {{"merge_proposals": []}}.
</json>"""


def _fmt_anti_examples() -> str:
    lines = []
    for a, b, reason in ANTI_EXAMPLES:
        lines.append(f"- `{a}` ≠ `{b}`  —  {reason}")
    return "\n".join(lines)


def _fmt_qk_block(qks: list[dict]) -> str:
    """qks: list of {qk, freq, sample_raw_keys (list)}"""
    lines = []
    for item in qks:
        samples = ", ".join(item.get("sample_raw_keys", [])[:4])
        lines.append(
            f"- {item['qk']:35s} freq={item['freq']:3d}  examples: {samples}"
        )
    return "\n".join(lines)


async def _llm_propose_for_batch(
    client: AsyncLocalModelClient,
    family: str, subtype: str,
    qks: list[dict],
) -> list[dict]:
    if not qks:
        return []
    prompt = LLM_BATCH_PROMPT.format(
        family=family, subtype=subtype,
        anti_examples=_fmt_anti_examples(),
        qk_block=_fmt_qk_block(qks),
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
            proposals = parsed.get("merge_proposals", [])
            if not isinstance(proposals, list):
                return []
            # sanitize
            out = []
            valid_qks = {q["qk"] for q in qks}
            for p in proposals:
                if not isinstance(p, dict):
                    continue
                target = str(p.get("target_name", "")).strip().lower()
                members_raw = p.get("members", [])
                if not isinstance(members_raw, list):
                    continue
                members = [str(m).strip().lower() for m in members_raw
                           if isinstance(m, str)]
                members = [m for m in members if m in valid_qks]
                if len(members) < 2:
                    continue
                if not target:
                    target = members[0]
                out.append({
                    "target_name": target,
                    "members": members,
                    "reason": str(p.get("reason", "")).strip(),
                    "family": family,
                    "subtype": subtype,
                })
            return out
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM attempt %d failed for %s/%s: %s",
                           attempt, family, subtype, e)
            await asyncio.sleep(backoff_with_jitter(attempt))
    logger.error("LLM failed for %s/%s", family, subtype)
    return []


def _build_qk_to_samples(annot: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for r in annot.itertuples(index=False):
        if len(out[r.quantity_kind]) < 6:
            out[r.quantity_kind].append(r.raw_key)
    return out


# ── Propose 入口 ───────────────────────────────────────────────────
def propose() -> None:
    """跑规则层 + LLM 层，输出 merge_rules 和 review_queue（待人审）。"""
    config.ensure_output_dir()

    logger.info("Loading structured annotations and qk counts …")
    annot = pd.read_csv(config.PHASE3_OUTPUT)
    counts = pd.read_csv(config.PHASE3_QK_COUNTS)
    logger.info("annotations: %d rows; unique qk: %d", len(annot), len(counts))

    # 1. 规则层
    rule_merge_map, rule_merge_log = _rule_layer_merge(counts)

    # 保存规则层决策
    rule_df = pd.DataFrame(rule_merge_log) if rule_merge_log else pd.DataFrame(
        columns=["source_qk", "target_qk", "freq_source", "layer", "reason"]
    )
    rule_df.to_csv(PHASE3_MERGE_RULES, index=False)
    logger.info("Rule-layer merge rules → %s (%d)",
                PHASE3_MERGE_RULES, len(rule_df))

    # 2. 规则层 collapse 后的 qk 统计 + 按 (family, subtype) 分组
    # 用 rule_merge_map 把 annot 的 qk 替换到 target
    annot = annot.copy()
    annot["qk_rule"] = annot["quantity_kind"].map(
        lambda q: rule_merge_map.get(str(q), str(q))
    )
    # 按 (family, subtype, qk) 统计
    qk_fs_stats = (
        annot.groupby(["family", "subtype", "qk_rule"]).size()
        .reset_index(name="freq")
    )
    qk_to_samples = _build_qk_to_samples(
        annot.rename(columns={"quantity_kind": "_original_qk",
                              "qk_rule": "quantity_kind"})
    )

    # 3. 按 (family, subtype) 分批做 LLM
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for _, r in qk_fs_stats.iterrows():
        groups[(r["family"], r["subtype"])].append({
            "qk": r["qk_rule"],
            "freq": int(r["freq"]),
            "sample_raw_keys": qk_to_samples.get(r["qk_rule"], []),
        })

    # 过滤：同 subtype 下 qk <= MIN_SUBTYPE_QK 跳过
    llm_tasks: list[tuple[str, str, list[dict]]] = []
    skipped_small = 0
    for (family, subtype), qks in groups.items():
        if len(qks) <= MIN_SUBTYPE_QK:
            skipped_small += 1
            continue
        qks_sorted = sorted(qks, key=lambda x: -x["freq"])
        # 若超过 BATCH_SIZE_CAP，切分
        if len(qks_sorted) <= BATCH_SIZE_CAP:
            llm_tasks.append((family, subtype, qks_sorted))
        else:
            for i in range(0, len(qks_sorted), BATCH_SIZE_CAP):
                llm_tasks.append(
                    (family, subtype, qks_sorted[i:i + BATCH_SIZE_CAP])
                )
    logger.info("LLM batches: %d (skipped %d small subtypes with qk<=%d)",
                len(llm_tasks), skipped_small, MIN_SUBTYPE_QK)

    # 4. LLM 调用
    api_key = os.environ.get("ALL_API_KEY")
    if not api_key:
        raise RuntimeError("ALL_API_KEY env var is required")

    all_proposals: list[dict] = []

    async def _go():
        sem = asyncio.Semaphore(CONCURRENCY)
        lock = asyncio.Lock()
        # checkpoint: which (family, subtype, batch_i) done
        done_set: set[tuple[str, str, int]] = set()
        if PHASE3_NORM_CHECKPOINT.exists():
            with open(PHASE3_NORM_CHECKPOINT, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        done_set.add((d["family"], d["subtype"], d["batch_i"]))
                        all_proposals.extend(d.get("proposals", []))
                    except Exception:
                        continue
            logger.info("Resumed %d LLM batches from checkpoint", len(done_set))

        async with AsyncLocalModelClient(
            base_url=config.BASE_URL, model=config.MODEL,
            temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS,
            api_key=api_key, stop_sentinel=config.STOP_SENTINEL,
            api_style=config.API_STYLE, auth_mode=config.AUTH_MODE,
        ) as client:

            async def worker(fam, sub, i, qks):
                async with sem:
                    proposals = await _llm_propose_for_batch(client, fam, sub, qks)
                    async with lock:
                        with open(PHASE3_NORM_CHECKPOINT, "a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "family": fam, "subtype": sub, "batch_i": i,
                                "proposals": proposals,
                            }, ensure_ascii=False) + "\n")
                    return proposals

            tasks = []
            # 重新按 batch 分配 index
            batch_index_map: dict[tuple[str, str], int] = defaultdict(int)
            for fam, sub, qks in llm_tasks:
                i = batch_index_map[(fam, sub)]
                batch_index_map[(fam, sub)] = i + 1
                if (fam, sub, i) in done_set:
                    continue
                tasks.append(asyncio.create_task(worker(fam, sub, i, qks)))

            start = time.time()
            for k, coro in enumerate(asyncio.as_completed(tasks)):
                batch_proposals = await coro
                all_proposals.extend(batch_proposals)
                if (k + 1) % 10 == 0:
                    rate = (k + 1) / max(1e-6, time.time() - start)
                    logger.info("LLM batches done %d/%d (%.2f/s)",
                                k + 1, len(tasks), rate)

    asyncio.run(_go())
    logger.info("Total LLM proposals (pre-validation): %d", len(all_proposals))

    # 5. 验证：决策 6 — 跨 family 直接拒绝（已在单批内，天然同 family）
    #    但再做一层检查：member qk 的 family 是否一致
    qk_family_map: dict[str, set[str]] = defaultdict(set)
    for r in annot.itertuples(index=False):
        qk_family_map[r.qk_rule].add(r.family)

    validated: list[dict] = []
    rejected_cross_family = 0
    for p in all_proposals:
        families = set()
        for m in p["members"]:
            families |= qk_family_map.get(m, set())
        if len(families) > 1:
            rejected_cross_family += 1
            continue
        # 累计 freq_sum
        freq_sum = int(sum(
            qk_fs_stats[(qk_fs_stats["qk_rule"] == m) &
                        (qk_fs_stats["subtype"] == p["subtype"])]["freq"].sum()
            for m in p["members"]
        ))
        validated.append({
            "target_name": p["target_name"],
            "members": ";".join(p["members"]),
            "reason": p["reason"],
            "subtype": p["subtype"],
            "family": p["family"],
            "freq_sum": freq_sum,
            "decision": "pending",
            "custom_target": "",
        })
    logger.info("Cross-family rejected: %d; validated: %d",
                rejected_cross_family, len(validated))

    # 6. 去重（同组 members 不同顺序视为同一提议）
    seen: set[frozenset] = set()
    deduped: list[dict] = []
    for p in validated:
        mkey = frozenset(p["members"].split(";"))
        if mkey in seen:
            continue
        seen.add(mkey)
        deduped.append(p)
    logger.info("After dedup: %d proposals", len(deduped))

    # 7. 写 review queue（decision 默认 pending）
    rq_df = pd.DataFrame(deduped).sort_values(
        ["subtype", "freq_sum"], ascending=[True, False]
    ) if deduped else pd.DataFrame(columns=[
        "target_name", "members", "reason", "subtype", "family",
        "freq_sum", "decision", "custom_target",
    ])
    rq_df.to_csv(PHASE3_REVIEW_QUEUE, index=False)
    logger.info("Review queue → %s (%d proposals, decision=pending)",
                PHASE3_REVIEW_QUEUE, len(rq_df))

    # 总结
    logger.info("=" * 60)
    logger.info("PHASE 3a-NORM PROPOSE SUMMARY")
    logger.info("=" * 60)
    logger.info("Input unique qk: %d", len(counts))
    logger.info("Rule-layer merges: %d (auto-approved)", len(rule_df))
    logger.info("LLM proposals: %d (need human review)", len(rq_df))
    logger.info("")
    logger.info("下一步：人工编辑 %s 的 decision 列（approve/reject/custom），",
                PHASE3_REVIEW_QUEUE)
    logger.info("然后运行  python3 -m metaagent_run.steps.env_field_pipeline.new 3-norm-apply")


# ── Apply 入口 ─────────────────────────────────────────────────────
def apply_merges() -> None:
    """读规则层 + 人审过的 LLM 提议，应用到 annotations 生成 final。"""
    config.ensure_output_dir()

    if not PHASE3_MERGE_RULES.exists():
        raise RuntimeError(
            f"规则层决策文件不存在：{PHASE3_MERGE_RULES}\n"
            "请先运行  python3 -m metaagent_run.steps.env_field_pipeline.new 3-norm-propose"
        )
    if not PHASE3_REVIEW_QUEUE.exists():
        raise RuntimeError(
            f"Review queue 文件不存在：{PHASE3_REVIEW_QUEUE}"
        )

    rule_df = pd.read_csv(PHASE3_MERGE_RULES)
    rq_df = pd.read_csv(PHASE3_REVIEW_QUEUE)
    annot = pd.read_csv(config.PHASE3_OUTPUT)

    logger.info("rule merges: %d; review_queue: %d; annotations: %d",
                len(rule_df), len(rq_df), len(annot))

    # 1. 规则层 map
    merge_map: dict[str, str] = {}
    for r in rule_df.itertuples(index=False):
        merge_map[r.source_qk] = r.target_qk

    # 2. 人审通过的 LLM 提议
    # decision 语义：
    #   approve  → 合并原 members 到 target_name
    #   reject / pending / 其他 → 忽略（不合并）
    #   custom   → 合并 custom_target 里列出的子集/修改后的成员集，
    #              target 优先用原 target_name（若在 custom_target 里），
    #              否则 custom_target 列表第一个。
    APPROVE_VALUES = {"approve", "approved", "y", "yes", "t", "true", "1"}
    CUSTOM_VALUES = {"custom"}
    llm_applied = 0
    custom_applied = 0
    llm_log: list[dict] = []
    for r in rq_df.itertuples(index=False):
        decision = str(r.decision or "").strip().lower()
        if decision not in APPROVE_VALUES and decision not in CUSTOM_VALUES:
            continue
        if decision in CUSTOM_VALUES:
            custom_str = str(r.custom_target or "").strip()
            if not custom_str:
                logger.warning("Decision=custom but custom_target empty: target=%s", r.target_name)
                continue
            effective_members = [m.strip() for m in custom_str.split(";") if m.strip()]
            if len(effective_members) < 2:
                logger.warning("Decision=custom but <2 members after parsing, skip: %s",
                               r.target_name)
                continue
            original_target = str(r.target_name).strip()
            if original_target in effective_members:
                target = original_target
            else:
                target = effective_members[0]
            custom_applied += 1
            layer_tag = "llm_custom"
        else:
            target = str(r.target_name).strip()
            effective_members = [m.strip() for m in str(r.members).split(";") if m.strip()]
            llm_applied += 1
            layer_tag = "llm"
        for m in effective_members:
            if not m:
                continue
            # 若 m 已被规则层映射，串行合并到最终 target
            # （规则层先行，然后 LLM 层再 override）
            merge_map[m] = target
            llm_log.append({
                "source_qk": m, "target_qk": target,
                "freq_source": 0,
                "layer": layer_tag,
                "reason": str(r.reason or "")[:150],
            })
    logger.info("LLM-approved merges: %d (+ %d custom)", llm_applied, custom_applied)

    # 3. 路径压缩：source -> target -> target'（chain resolve）
    def resolve(q: str, seen=None) -> str:
        seen = seen or set()
        if q in seen:
            return q
        seen.add(q)
        if q in merge_map and merge_map[q] != q:
            return resolve(merge_map[q], seen)
        return q

    final_map: dict[str, str] = {k: resolve(k) for k in list(merge_map.keys())}

    # 4. 应用到 annotations（只改 quantity_kind 字段）
    before_unique = annot["quantity_kind"].nunique()
    annot["quantity_kind"] = annot["quantity_kind"].apply(
        lambda q: final_map.get(str(q), str(q))
    )
    after_unique = annot["quantity_kind"].nunique()
    logger.info("qk collapsed: %d → %d (%.1f%% reduction)",
                before_unique, after_unique,
                100.0 * (1 - after_unique / max(1, before_unique)))

    # 5. 写 final CSV
    annot.to_csv(PHASE3_FINAL, index=False)
    logger.info("Final annotations → %s (%d rows)", PHASE3_FINAL, len(annot))

    # 6. 备份 structured_annotations 并用 final 覆盖
    if not PHASE3_STRUCTURED_BAK.exists():
        shutil.copy(config.PHASE3_OUTPUT, PHASE3_STRUCTURED_BAK)
        logger.info("Backed up pre-norm: %s", PHASE3_STRUCTURED_BAK)
    shutil.copy(PHASE3_FINAL, config.PHASE3_OUTPUT)
    logger.info("Overwrote %s with final content (phase4 reads this path)",
                config.PHASE3_OUTPUT)

    # 7. 追加 LLM 层决策到 merge_rules（合并审计）
    if llm_log:
        all_rules = pd.concat(
            [rule_df, pd.DataFrame(llm_log)], ignore_index=True
        ).drop_duplicates(subset=["source_qk", "target_qk"])
        all_rules.to_csv(PHASE3_MERGE_RULES, index=False)
        logger.info("Updated merge_rules with LLM decisions: %d total rows",
                    len(all_rules))

    # 8. 更新 quantity_kind_counts
    new_counts = (
        annot["quantity_kind"].value_counts()
        .reset_index()
    )
    new_counts.columns = ["quantity_kind", "freq"]
    new_counts.to_csv(config.PHASE3_QK_COUNTS, index=False)
    logger.info("Updated qk counts (post-norm): %d unique qk",
                len(new_counts))

    # 总结
    logger.info("=" * 60)
    logger.info("PHASE 3a-NORM APPLY SUMMARY")
    logger.info("=" * 60)
    logger.info("qk collapsed:       %d → %d", before_unique, after_unique)
    logger.info("Rule merges:        %d", len(rule_df))
    logger.info("LLM approved:       %d", llm_applied)
    logger.info("LLM custom-target:  %d", custom_applied)
    logger.info("Final annotations:  %s", PHASE3_FINAL)
    logger.info("Backup (pre-norm):  %s", PHASE3_STRUCTURED_BAK)
    logger.info("Downstream path:    %s (overwritten with final)",
                config.PHASE3_OUTPUT)
