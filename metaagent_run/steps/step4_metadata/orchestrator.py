import logging
from pathlib import Path
from typing import Optional

from metaagent_run.core import AsyncLocalModelClient

from .analytics import (
    classify_env_fields,
    compute_env_coverage_counts,
    compute_env_paper_counts,
    compute_env_profile,
    compute_frequency,
    compute_tiered_output,
    compute_total_sampled_pmids,
    compute_unmapped_frequency,
    save_env_extraction_targets,
    save_supplementary_fields,
    save_tiered_output,
)
from .config import RuntimeConfig, load_runtime_config
from .processor import (
    apply_review_decisions_post_llm,
    compute_multi_env_pmids,
    filter_pmid_env_index,
    filter_pmid_index,
    generate_review_queue_csv,
    get_pre_llm_excluded_fields,
    load_final_fields,
    load_mapping_review_decisions,
    load_mixs_standards,
    load_paper_env_map,
    load_pmid_env_index,
    load_pmid_index,
    load_synonym_groups,
    mapping_pipeline,
)
from .prompt_builder import load_prompt_template
from .storage import load_checkpoint, save_checkpoint, save_outputs
from .visualization import (
    plot_env_field_bars,
    plot_env_field_frequency,
    plot_env_field_heatmap,
    plot_synonym_fanout,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


async def main_async(
    fields_file: Optional[str] = None,
    mixs_file: Optional[str] = None,
    pmid_index_file: Optional[str] = None,
    pmid_env_index_file: Optional[str] = None,
    paper_env_map_file: Optional[str] = None,
    discovery_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_retries: Optional[int] = None,
    request_interval: Optional[float] = None,
    no_resume: bool = False,
    exclusion_list: Optional[str] = None,
    mapping_veto_list: Optional[str] = None,
    tier2_min_pmid: Optional[int] = None,
    top_n_freq: Optional[int] = None,
    top_n_fanout: Optional[int] = None,
    min_fanout: Optional[int] = None,
    runtime_config: Optional[RuntimeConfig] = None,
) -> None:
    config = runtime_config or load_runtime_config()

    resolved_fields_file = Path(fields_file or config.fields_file)
    resolved_mixs_file = Path(mixs_file or config.mixs_file)
    resolved_pmid_index_file = Path(pmid_index_file or config.pmid_index_file)
    resolved_pmid_env_index_file = Path(pmid_env_index_file or config.pmid_env_index_file)
    resolved_paper_env_map_file = Path(paper_env_map_file or config.paper_env_map_file)
    resolved_discovery_file = Path(discovery_file or config.discovery_file)
    resolved_output_dir = Path(output_dir or config.output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_batch_size = batch_size if batch_size is not None else config.batch_size
    resolved_max_retries = max_retries if max_retries is not None else config.max_retries_per_batch
    resolved_request_interval = request_interval if request_interval is not None else config.request_interval
    resolved_resume = False if no_resume else config.resume_from_checkpoint
    # Single unified file replaces tier_exclusion_list + mapping_veto_list +
    # mapping_correction_list. The kwargs `exclusion_list` and
    # `mapping_veto_list` are kept as aliases for backward compatibility.
    resolved_review_decisions_file = Path(
        exclusion_list or mapping_veto_list or config.mapping_review_decisions_file
    )
    resolved_tier2_min_pmid = tier2_min_pmid if tier2_min_pmid is not None else config.tier2_min_pmid
    resolved_top_n_freq = top_n_freq if top_n_freq is not None else config.top_n_freq
    resolved_top_n_fanout = top_n_fanout if top_n_fanout is not None else config.top_n_fanout
    resolved_min_fanout = min_fanout if min_fanout is not None else config.min_fanout

    final_fields = load_final_fields(resolved_fields_file)
    mixs_standards = load_mixs_standards(resolved_mixs_file)
    pmid_index = load_pmid_index(resolved_pmid_index_file)
    pmid_env_index = load_pmid_env_index(resolved_pmid_env_index_file)
    paper_env_map = load_paper_env_map(resolved_paper_env_map_file)

    # ── 剔除多环境论文，确保 Fisher 列联表数据源一致 ──
    multi_env_pmids = compute_multi_env_pmids(paper_env_map)
    pmid_index = filter_pmid_index(pmid_index, multi_env_pmids)
    pmid_env_index = filter_pmid_env_index(pmid_env_index, multi_env_pmids)
    LOGGER.info(
        "剔除多环境论文后: pmid_index=%d fields, pmid_env_index=%d fields",
        len(pmid_index), len(pmid_env_index),
    )

    env_paper_counts = compute_env_paper_counts(paper_env_map)
    env_coverage_counts = compute_env_coverage_counts(pmid_env_index)
    synonym_groups = load_synonym_groups(resolved_discovery_file)
    # ── 加载统一人工 review decisions ──
    review_decisions = load_mapping_review_decisions(resolved_review_decisions_file)
    pre_llm_excluded = get_pre_llm_excluded_fields(review_decisions)
    # exclusion_entries kept for downstream tier output (backward compat)
    exclusion_entries = [
        {"field": d["field"], "reason": d.get("reason", ""), "category": "EXCLUDE"}
        for d in review_decisions if d["action"] == "EXCLUDE"
    ]
    load_prompt_template(config.prompt_version)

    # ── Pre-LLM 高频过滤：union over alias PMIDs ≥ tier2_min_pmid ──
    all_field_count = len(final_fields)
    high_freq_fields = []
    for field in final_fields:
        members = synonym_groups.get(field, [field])
        pmid_union = set()
        for member in members:
            pmid_union.update(pmid_index.get(member, []))
        if len(pmid_union) >= resolved_tier2_min_pmid:
            high_freq_fields.append(field)
    final_fields = high_freq_fields
    LOGGER.info(
        "高频过滤 (>=%d PMIDs): %d / %d canonical → %d 高频字段",
        resolved_tier2_min_pmid, len(final_fields), all_field_count, len(final_fields),
    )

    # ── Pre-LLM EXCLUDE: 应用 review decisions 的 EXCLUDE action ──
    pre_excl_count = len(final_fields)
    final_fields = [f for f in final_fields if f not in pre_llm_excluded]
    LOGGER.info(
        "Pre-LLM EXCLUDE: %d / %d → %d 进入 LLM mapping",
        pre_excl_count - len(final_fields), pre_excl_count, len(final_fields),
    )

    # ── LLM mapping ──────────────────────────────────────────
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
        mapped_results, failed_fields = await mapping_pipeline(
            all_fields=final_fields,
            llm_client=llm_client,
            mixs_standards=mixs_standards,
            synonym_groups=synonym_groups,
            batch_size=resolved_batch_size,
            max_retries_per_batch=resolved_max_retries,
            request_interval=resolved_request_interval,
            checkpoint_path=str(resolved_output_dir / config.mapping_checkpoint_file),
            resume=resolved_resume,
            prompt_version=config.prompt_version,
            checkpoint_loader=load_checkpoint,
            checkpoint_saver=save_checkpoint,
            concurrency=config.concurrency,
            checkpoint_every=config.checkpoint_every,
        )

    # ── Post-LLM 应用 review decisions（统一三层 curation） ──
    # 单一函数处理所有人工决策（EXCLUDE / FORCE_UNMAPPED / FORCE_SLOT），
    # 替代之前的 apply_mapping_vetoes + apply_exclusion_as_veto +
    # apply_mapping_corrections 三个函数。
    # 优先级：EXCLUDE > FORCE_UNMAPPED > FORCE_SLOT (单调，不会把已封锁字段提升)
    valid_mixs_slots = {entry["Slot_Name"] for entry in mixs_standards}
    mapped_results = apply_review_decisions_post_llm(
        mapped_results, review_decisions, valid_mixs_slots,
    )
    freq_data = compute_frequency(mapped_results, pmid_index, synonym_groups)
    unmapped_freq = compute_unmapped_frequency(mapped_results, pmid_index, synonym_groups)
    env_profiles = compute_env_profile(
        pmid_env_index,
        mapped_results,
        synonym_groups,
        env_coverage_counts,
    )
    total_sampled_pmids = compute_total_sampled_pmids(pmid_index)
    tiered = compute_tiered_output(
        freq_data=freq_data,
        unmapped_freq=unmapped_freq,
        mapped_results=mapped_results,
        total_sampled_pmids=total_sampled_pmids,
        tier2_min_pmid=resolved_tier2_min_pmid,
        exclusion_set=pre_llm_excluded,
        exclusion_entries=exclusion_entries,
        env_profiles=env_profiles,
        env_paper_counts=env_paper_counts,
        env_coverage_counts=env_coverage_counts,
    )

    # ── Save JSON outputs ────────────────────────────────────
    save_outputs(mapped_results, failed_fields, freq_data, mixs_standards, resolved_output_dir,
                  mapping_filename=config.mapping_result_file, unmapped_filename=config.unmapped_fields_file,
                  frequency_filename=config.frequency_by_mixs_file)
    save_tiered_output(tiered, resolved_output_dir, filename=config.tiered_output_file)

    # ── Environment classification + extraction targets ──────
    all_entries, included, excluded_low_freq, env_n, total_papers = classify_env_fields(
        tiered=tiered,
        exclusion_set=pre_llm_excluded,
        min_pmid=config.min_env_pmid,
        min_env_pct=config.min_env_pct,
    )

    # ── 产出 reviewer-ready CSV (基于 included,只含进 Step 5 target 的字段) ──
    # 每跑一次 Step 4-meta 自动更新; CSV 按 priority + total_pmid 排好序,
    # reviewer 从顶部往下审; 决策填入 mapping_review_decisions.json
    # 后重跑 post-processing 即生效。
    # 包含两类行:
    #   (1) Mapped: LLM 映射到某 slot 且该 slot 进入 step-5 target
    #   (2) UNMAPPED: LLM 无 MIxS 匹配,但 PMID 通过 env 阈值,作为独立字段进入 step-5
    generate_review_queue_csv(
        output_path=resolved_output_dir / config.review_queue_csv_file,
        mapped_results=mapped_results,
        included_entries=included,
        synonym_groups=synonym_groups,
        pmid_index=pmid_index,
    )

    save_env_extraction_targets(
        included=included,
        all_entries=all_entries,
        excluded_low_freq=excluded_low_freq,
        env_n=env_n,
        total_papers=total_papers,
        output_dir=resolved_output_dir,
        filename=config.env_targets_output_file,
        min_pmid=config.min_env_pmid,
        min_env_pct=config.min_env_pct,
    )
    save_supplementary_fields(
        excluded_low_freq=excluded_low_freq,
        output_dir=resolved_output_dir,
        filename=config.supplementary_low_freq_file,
    )

    # ── Visualization (4 figures) ────────────────────────────
    # Build a data-funnel summary for Fig A
    VALID_ENVS_LOCAL = {"Open_ocean", "Coastal_waters", "Lake", "Wetlands"}
    _n_classified = len(paper_env_map)
    _n_single_env = sum(
        1 for _, envs in paper_env_map.items()
        if len({e for e in envs if e in VALID_ENVS_LOCAL}) == 1
    )
    funnel_counts = {
        "classified": _n_classified,
        "single_env": _n_single_env,
        "metadata_bearing": total_papers,
    }
    plot_env_field_frequency(
        all_entries=all_entries,
        output_path=resolved_output_dir / config.env_field_frequency_pdf,
        total_papers=total_papers,
        funnel_counts=funnel_counts,
    )
    plot_env_field_heatmap(
        included=included,
        output_path=resolved_output_dir / config.env_field_heatmap_pdf,
        env_n=env_n,
    )
    plot_env_field_bars(
        included=included,
        output_path=resolved_output_dir / config.env_field_bars_pdf,
        env_n=env_n,
    )
    plot_synonym_fanout(
        tiered=tiered,
        output_path=resolved_output_dir / config.synonym_fanout_pdf,
        min_fanout=resolved_min_fanout,
        top_n=resolved_top_n_fanout,
    )
