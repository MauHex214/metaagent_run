"""step3 主编排：两阶段无阈值管线。

Phase 0 — 质量过滤（无 LLM）：
    将 build_flattened_occurrences 产出的 unique_keys 包装为 raw_entries
    送入 filter_discovered_entries：
      * SCIENTIFIC_WHITELIST 命中 → 直接通过
      * 命中 VALUE_PATTERNS / INSTRUMENT_BLACKLIST / EXPERIMENTAL_PREFIX → 丢弃
      * 长度 >60 字符 / 含 4 位数字 → 进 review_pool，从 Phase 1 剔除
    Evidence 字段按 runtime_config.use_evidence_in_filter：
      True  → 用 key_to_evidence 的真实原文（review_pool 带上下文）
      False → "(step2)" 占位符

Phase 1 — 分组（含 LLM）：
    按字典序遍历 filtered_keys，每个 key 逐一调 SemanticDeduplicator.resolve(
    key, evidence=...) 归入 canonical 组（LLM 调用串行，不做批量）。
    evidence 按 runtime_config.use_evidence_in_dedup 决定（True=首见 section
    的 evidence_quote，False=空串）。每处理 checkpoint_stride 个 key 落一次盘
    并打一行 summary，同时追加一条 IterationRecord 到 history。

Phase 2 — 归因（无 LLM）：
    遍历扁平化产出的 (key, pmid, env) 条目，按 Phase 1 的分组结果
    查出 canonical，累积 field_pmid_index / field_pmid_env_index。
    review_pool 里的 key 若出现在 occurrences 中，其 PMID 贡献**不**计入索引
    （这些 key 在 Phase 1 被刻意排除，canonical 查找会落空）。

无饱和触发、无确认、无轻量扫描、无采样模式切换——filter 通过的全部 unique key 都参与分组。
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, cast

from tqdm.auto import tqdm

from metaagent_run.core import AsyncLocalModelClient

from .config import RuntimeConfig, load_runtime_config
from .prompt_builder import load_prompt_template
from .runtime import LOGGER
from .sampling import IterationRecord, iter_batches
from .schema import SemanticDeduplicator, filter_discovered_entries
from .storage import (
    load_checkpoint,
    save_checkpoint,
    save_field_pmid_index,
    save_field_pmid_env_index,
)
from .stratification import (
    VALID_SUB_ENVS,
    build_flattened_occurrences,
)
from .visualization import plot_saturation_curve

ItemRecord = dict[str, object]


def load_discovery_items(path: Path) -> list[ItemRecord]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层不是 list")
    return [item for item in data if isinstance(item, dict)]


async def schema_discovery_pipeline(
    items: list[ItemRecord],
    llm_client: AsyncLocalModelClient,
    runtime_config: RuntimeConfig,
) -> tuple[set[str], list[IterationRecord]]:
    # ── Step 1: 扁平化预处理 ──
    LOGGER.info("[Step 1] 扁平化 section → (key,pmid,env) 条目 + key_to_evidence")
    unique_keys, occurrences, key_to_evidence = build_flattened_occurrences(items)

    if not unique_keys:
        LOGGER.warning("唯一 key 列表为空，流水线直接结束")
        return set(), []

    # ── Step 1b: 质量过滤 + review_pool（Phase 0，无 LLM） ──
    # 移植自主线 step3/processor.py::llm_discover()：在 unique key 进 dedup
    # 之前做一道廉价的静态过滤，剔除明显的值/仪器名/实验条件前缀，并把
    # 长 key / 带 4 位数字的可疑 key 单独归入 review_pool 供人工审核。
    #
    # Evidence 处理按 USE_EVIDENCE_IN_FILTER flag：
    #   True  → 用 key_to_evidence 里的真实原文片段，让 review_pool 记录带上下文
    #   False → "(step2)" 占位符（满足 filter 的非空校验即可）
    raw_unique_key_count = len(unique_keys)
    review_pool: list[dict[str, object]] = []
    if runtime_config.use_evidence_in_filter:
        raw_entries = [
            {"key": key, "evidence": key_to_evidence.get(key) or "(step2)"}
            for key in unique_keys
        ]
    else:
        raw_entries = [{"key": key, "evidence": "(step2)"} for key in unique_keys]
    filtered_hits = filter_discovered_entries(
        cast(list[object], raw_entries),
        review_pool,
    )
    filtered_keys_sorted = sorted({hit["normalized_key"] for hit in filtered_hits})
    rejected_silent = raw_unique_key_count - len(filtered_keys_sorted) - len(review_pool)
    LOGGER.info(
        "[质量过滤] 原始 unique=%d → 通过=%d, review_pool=%d, 静默丢弃=%d (use_evidence_in_filter=%s)",
        raw_unique_key_count,
        len(filtered_keys_sorted),
        len(review_pool),
        rejected_silent,
        runtime_config.use_evidence_in_filter,
    )
    if review_pool:
        LOGGER.info(
            "[review_pool] 样例（前 5 条）: %s",
            [{"key": r["normalized_key"], "reason": r["reason"]} for r in review_pool[:5]],
        )
    unique_keys = filtered_keys_sorted

    if not unique_keys:
        LOGGER.warning("质量过滤后唯一 key 列表为空，流水线直接结束")
        return set(), []

    # ── Step 2: checkpoint 恢复（可选） ──
    processed_key_count = 0
    history: list[IterationRecord] = []
    deduplicator = SemanticDeduplicator()

    if runtime_config.resume_from_checkpoint:
        cp = load_checkpoint(str(runtime_config.checkpoint_file))
        if cp is not None:
            processed_key_count = int(cp.get("processed_key_count", 0) or 0)
            raw_history = cp.get("history", [])
            if isinstance(raw_history, list):
                history = [rec for rec in raw_history if isinstance(rec, dict)]
            raw_dedup = cp.get("deduplicator")
            if isinstance(raw_dedup, dict):
                deduplicator = SemanticDeduplicator.from_dict(
                    {str(k): v for k, v in raw_dedup.items()}
                )
            LOGGER.info(
                "[恢复] 从 checkpoint 恢复: processed=%d/%d, canonical=%d, history=%d",
                processed_key_count, len(unique_keys),
                deduplicator.canonical_size, len(history),
            )
        else:
            LOGGER.info("[恢复] 未找到 checkpoint，从头开始")

    deduplicator.set_llm_client(
        llm_client,
        request_interval=runtime_config.request_interval,
    )

    # ── Step 3: 分组主循环（Phase 1，含 LLM） ──
    LOGGER.info("=" * 60)
    LOGGER.info("[Phase 1] 分组主循环：唯一 key=%d, checkpoint_stride=%d",
                len(unique_keys), runtime_config.checkpoint_stride)
    LOGGER.info("=" * 60)

    unresolved_keys: list[str] = []
    n_total = len(unique_keys)
    remaining_count = n_total - processed_key_count

    if remaining_count <= 0:
        LOGGER.info("[Phase 1] 所有 unique key 已处理完成（由 checkpoint 恢复）")
    else:
        pbar = tqdm(
            total=remaining_count,
            desc="Phase 1 grouping",
            unit="key",
            dynamic_ncols=True,
            initial=0,
        )
        iteration = len(history)
        for batch_start, batch_keys in iter_batches(
            unique_keys, runtime_config.checkpoint_stride, start_index=processed_key_count,
        ):
            iter_new: set[str] = set()
            for key in batch_keys:
                # USE_EVIDENCE_IN_DEDUP=True → 传真实 evidence 辅助 LLM 精判
                #   False → 传空串，dedup 仅凭 key 字符串 + 候选列表判定
                dedup_evidence = (
                    key_to_evidence.get(key, "") if runtime_config.use_evidence_in_dedup else ""
                )
                canonical, is_new = await deduplicator.resolve(key, evidence=dedup_evidence)
                if canonical is None:
                    # LLM 调用失败（已内部重试过）→ 记账、不强行成新组
                    unresolved_keys.append(key)
                elif is_new:
                    iter_new.add(canonical)
                pbar.update(1)

            processed_key_count = batch_start + len(batch_keys)
            iteration += 1
            record: IterationRecord = {
                "iteration": iteration,
                "keys_processed_this_iter": len(batch_keys),
                "cumulative_keys_processed": processed_key_count,
                "cumulative_canonical_size": deduplicator.canonical_size,
                "new_canonical_keys_this_iter": len(iter_new),
            }
            history.append(record)

            LOGGER.info(
                "[iter %d] 处理 %d keys (累计 %d/%d), canonical=%d (+%d), 失败=%d",
                iteration, len(batch_keys), processed_key_count, n_total,
                deduplicator.canonical_size, len(iter_new), len(unresolved_keys),
            )
            pbar.set_postfix_str(
                f"canonical={deduplicator.canonical_size} new={len(iter_new)} fail={len(unresolved_keys)}"
            )

            save_checkpoint(
                checkpoint_path=str(runtime_config.checkpoint_file),
                processed_key_count=processed_key_count,
                history=history,
                deduplicator=deduplicator,
            )
        pbar.close()

    LOGGER.info("=" * 60)
    LOGGER.info("[Phase 1] 完成。canonical 组数=%d, 未解析 key=%d",
                deduplicator.canonical_size, len(unresolved_keys))
    if unresolved_keys:
        LOGGER.warning("[Phase 1] 未解析 key 样例（前 20 个）: %s",
                       sorted(unresolved_keys)[:20])

    # ── Step 4: 归因后处理（Phase 2，无 LLM） ──
    LOGGER.info("=" * 60)
    LOGGER.info("[Phase 2] 归因：回填 field_pmid_index / field_pmid_env_index")
    field_pmid_index: dict[str, set[str]] = defaultdict(set)
    field_pmid_env_index: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {env: set() for env in VALID_SUB_ENVS}
    )
    unmatched_occurrences = 0
    matched_occurrences = 0

    for key, pmid, env in occurrences:
        canonical = deduplicator.get_canonical(key)
        if canonical is None:
            unmatched_occurrences += 1
            continue
        field_pmid_index[canonical].add(pmid)
        if env in VALID_SUB_ENVS:
            field_pmid_env_index[canonical][env].add(pmid)
        matched_occurrences += 1

    LOGGER.info("[Phase 2] 归因完成：matched=%d, unmatched=%d (占 %.2f%%)",
                matched_occurrences, unmatched_occurrences,
                100.0 * unmatched_occurrences / max(len(occurrences), 1))

    # ── Step 5: 保存输出 ──
    save_field_pmid_index(
        str(runtime_config.field_pmid_index_file), field_pmid_index,
    )
    save_field_pmid_env_index(
        str(runtime_config.field_pmid_env_index_file), field_pmid_env_index,
    )

    alias_report = deduplicator.get_alias_report()
    all_groups = deduplicator.get_all_groups()
    if alias_report:
        LOGGER.info("[语义去重] %d 个 canonical 有多个别名", len(alias_report))

    result: dict[str, Any] = {
        "final_schema": sorted(list(deduplicator.canonical_keys)),
        "total_fields": deduplicator.canonical_size,
        "synonym_groups": {key: sorted(value) for key, value in all_groups.items()},
        "alias_report": {key: sorted(value) for key, value in alias_report.items()},
        "total_unique_keys_before_filter": raw_unique_key_count,
        "total_unique_keys": n_total,
        "filter_rejected_silent": rejected_silent,
        "total_key_occurrences": len(occurrences),
        "unresolved_keys": sorted(unresolved_keys),
        "unresolved_key_count": len(unresolved_keys),
        "attribution_matched_occurrences": matched_occurrences,
        "attribution_unmatched_occurrences": unmatched_occurrences,
        "total_iterations": len(history),
        "history": history,
        "review_pool": review_pool,
        "review_pool_size": len(review_pool),
        "field_pmid_index_path": str(runtime_config.field_pmid_index_file),
        "field_pmid_index_key_count": len(field_pmid_index),
        "field_pmid_index_total_pairs": sum(len(v) for v in field_pmid_index.values()),
        "field_pmid_env_index_path": str(runtime_config.field_pmid_env_index_file),
        "field_pmid_env_index_key_count": len(field_pmid_env_index),
        "field_pmid_env_index_total_pairs": sum(
            len(pmids)
            for env_map in field_pmid_env_index.values()
            for pmids in env_map.values()
        ),
        "deduplicator_state": deduplicator.to_dict(),
    }

    with runtime_config.output_file.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    LOGGER.info("[输出] 结果已保存至 %s", runtime_config.output_file)

    return deduplicator.canonical_keys, history


async def main_async(
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    checkpoint_file: Optional[str] = None,
    runtime_config: Optional[RuntimeConfig] = None,
    **_ignored,  # 接住老接口传来的 schema_contexts_file 等，不报错
) -> tuple[set[str], list[IterationRecord]]:
    config = runtime_config or load_runtime_config()
    # 允许命令行覆盖 I/O 路径
    config = RuntimeConfig(
        prompt_version=config.prompt_version,
        base_url=config.base_url,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=config.api_key,
        api_style=config.api_style,
        azure_api_version=config.azure_api_version,
        azure_deployment=config.azure_deployment,
        auth_mode=config.auth_mode,
        stop_sentinel=config.stop_sentinel,
        input_file=Path(input_file) if input_file else config.input_file,
        output_file=Path(output_file) if output_file else config.output_file,
        checkpoint_file=Path(checkpoint_file) if checkpoint_file else config.checkpoint_file,
        saturation_plot_file=config.saturation_plot_file,
        field_pmid_index_file=config.field_pmid_index_file,
        field_pmid_env_index_file=config.field_pmid_env_index_file,
        checkpoint_stride=config.checkpoint_stride,
        request_interval=config.request_interval,
        resume_from_checkpoint=config.resume_from_checkpoint,
        llm_max_retries=config.llm_max_retries,
        use_evidence_in_dedup=config.use_evidence_in_dedup,
        use_evidence_in_filter=config.use_evidence_in_filter,
    )
    load_prompt_template(config.prompt_version)
    items = load_discovery_items(config.input_file)
    LOGGER.info("加载 discovery 输入: %d 条 section", len(items))

    async with AsyncLocalModelClient(
        base_url=config.base_url,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=config.api_key,
        stop_sentinel=config.stop_sentinel,
        enable_thinking=False,
        api_style=config.api_style,
        azure_api_version=config.azure_api_version,
        azure_deployment=config.azure_deployment,
        auth_mode=config.auth_mode,
    ) as llm_client:
        final_schema, history = await schema_discovery_pipeline(
            items=items,
            llm_client=llm_client,
            runtime_config=config,
        )

    plot_saturation_curve(
        history=history,
        output_path=str(config.saturation_plot_file),
    )
    LOGGER.info("最终 canonical 字段数: %d", len(final_schema))
    return final_schema, history
