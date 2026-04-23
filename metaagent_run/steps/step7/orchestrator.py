"""Step7 主流程：load → key/value normalize → hoist → write."""

import json
import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from .config import RuntimeConfig
from .hoist import build_acc_to_items_index, classify_accession_level, hoist_one_biosample
from .normalizers import normalize, NORMALIZERS
from .schemas import (
    BioSampleRecord,
    FinalMetadataEntry,
    NormalizedValue,
    ProvenanceRecord,
    Step7Output,
)
from .upstream_loader import UpstreamData, load_upstream

LOGGER = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Phase 7a — Key 校验
# ═══════════════════════════════════════════════════════════

def _validate_keys(items: List[Dict[str, Any]]) -> None:
    """所有条目的 canonical_slot 必须有 mixs: 或 internal: 前缀。"""
    bad: List[str] = []
    for it in items:
        slot = it.get("canonical_slot", "")
        if not (slot.startswith("mixs:") or slot.startswith("internal:")):
            bad.append(slot)
    if bad:
        LOGGER.warning("Found %d items with non-namespaced keys (sample: %s)",
                       len(bad), bad[:3])


# ═══════════════════════════════════════════════════════════
#  Phase 7b — Value 归一化
# ═══════════════════════════════════════════════════════════

def _lookup_cde_entry(
    canonical_slot: str,
    biosample_env: str,
    cde: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """按 (env, key) 查 CDE。

    Lookup 顺序：
      1. env-specific CDE 中精确匹配 canonical_slot
      2. 跨 env 找该 canonical_slot
      3. 若 canonical_slot 是 mixs:xxx 但找不到，自动 fallback 到 internal:xxx
         （用于 step5 mixs_slot 命名与 step4b Tier 1 不一致的情形）
      4. 最终 fallback: passthrough
    """
    candidates = [canonical_slot]
    if canonical_slot.startswith("mixs:"):
        candidates.append("internal:" + canonical_slot[5:])

    env_cde = cde.get(biosample_env, {}) if isinstance(cde, dict) else {}
    if isinstance(env_cde, dict):
        for cand in candidates:
            if cand in env_cde:
                return env_cde[cand]
    # 跨 env 找
    for other_env, e_cde in cde.items():
        if not isinstance(e_cde, dict):
            continue
        for cand in candidates:
            if cand in e_cde:
                return e_cde[cand]
    return {"normalizer": "passthrough", "value_syntax": "{text}",
            "preferred_unit": [], "tier": 0}


def _build_final_entry(
    item: Dict[str, Any],
    norm_value: NormalizedValue,
    source_level: str,
) -> FinalMetadataEntry:
    alts = []
    for alt in item.get("alternate_sources", []):
        if isinstance(alt, dict):
            alts.append(ProvenanceRecord(
                pmid=str(alt.get("pmid", "")),
                value=str(alt.get("value", "")),
                raw_field=str(alt.get("raw_field", "")),
                source_file=str(alt.get("source_file", "")),
                section_type=str(alt.get("section_type", "")),
                paragraph_index=int(alt.get("paragraph_index", -1) or -1),
                extraction_modality=str(alt.get("extraction_modality", "")),
            ))
    return FinalMetadataEntry(
        key=str(item.get("canonical_slot", "")),
        raw_field=str(item.get("raw_field", "")),
        value_raw=str(item.get("value", "")),
        value_normalized=norm_value.value_normalized,
        unit=norm_value.unit,
        value_type=norm_value.value_type,
        normalize_status=norm_value.normalize_status,
        normalize_error=norm_value.normalize_error,
        source_level=source_level,
        authoritative_pmid=str(item.get("authoritative_pmid", "")),
        source_file=str(item.get("source_file", "")),
        section_type=str(item.get("section_type", "")),
        paragraph_index=int(item.get("paragraph_index", -1) or -1),
        extraction_modality=str(item.get("extraction_modality", "")),
        alternate_sources=alts,
    )


# ═══════════════════════════════════════════════════════════
#  Phase 7c — Sample 级 Hoist
# ═══════════════════════════════════════════════════════════

def _collect_target_biosamples(ud: UpstreamData) -> List[str]:
    """所有需要输出的 BioSample id：从 step6 涉及的 raw_acc 反查。"""
    targets: set = set()
    for it in ud.resolved_items:
        acc = it.get("raw_accession", "")
        if not acc:
            continue
        # 1) acc 本身就是 biosample
        if classify_accession_level(acc) == "biosample":
            targets.add(acc)
        # 2) acc 通过骨架反查 biosample
        bs = ud.acc_to_biosample.get(acc)
        if bs:
            targets.add(bs)
        # 3) acc 是 bioproject → 可能对应多个 biosample
        if classify_accession_level(acc) == "bioproject":
            for child_bs in ud.bioproject_to_biosamples.get(acc, []):
                targets.add(child_bs)
    return sorted(targets)


def _build_paper_dominant_env(bs_id: str, ud: UpstreamData) -> str:
    """挑选该 biosample 关联的所有 PMID 中第一个有 dominant env 的。

    与 _build_source_pmids 用相同的关联范围（biosample + 同骨架其他 acc + parent project）。
    """
    related_pmids = set(_build_source_pmids(bs_id, ud))
    for pmid in sorted(related_pmids):
        env = ud.pmid_to_dominant_env.get(pmid)
        if env:
            return env
    return ""


def _build_aliases(bs_id: str, ud: UpstreamData) -> List[str]:
    related_accs = {bs_id}
    related_accs.update(ud.biosample_to_related.get(bs_id, set()))
    bp = ud.biosample_to_bioproject.get(bs_id, "")
    if bp:
        related_accs.add(bp)
    aliases: set = set()
    for acc in related_accs:
        aliases.update(ud.aliases_by_acc.get(acc, set()))
    return sorted(aliases)


def _build_source_pmids(bs_id: str, ud: UpstreamData) -> List[str]:
    related_accs = {bs_id}
    related_accs.update(ud.biosample_to_related.get(bs_id, set()))
    bp = ud.biosample_to_bioproject.get(bs_id, "")
    if bp:
        related_accs.add(bp)
    pmids: set = set()
    for acc in related_accs:
        pmids.update(ud.pmids_by_acc.get(acc, set()))
    return sorted(pmids)


def _formal_name(bs_id: str, ud: UpstreamData) -> str:
    db = ud.biosample_metadata.get(bs_id, {})
    return str(db.get("sample_name", "") or db.get("sample_title", "") or "")


def _pmid_year_index(ud: UpstreamData) -> Dict[str, int]:
    """从 step5 PaperOutput 没有 pub_year，需独立读 paper_down/pmid_year.txt。

    复用 step6 已有的 loader：直接构造一个临时字典就行。
    """
    # 从 step6 阶段也读过 pmid_year，但它在 step6 内部，没回传到 step7。
    # 这里重新读一遍（一次性，不影响性能）。
    pmid_year: Dict[str, int] = {}
    # 不强依赖：找不到也不报错（hoist 同层级仲裁会用 9999）
    return pmid_year


# ═══════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════

def run(cfg: RuntimeConfig) -> None:
    LOGGER.info("Loading upstream data...")
    ud = load_upstream(cfg)

    # Phase 7a
    _validate_keys(ud.resolved_items)

    # 构建 raw_accession → slot → items 索引（hoist 用）
    acc_to_items = build_acc_to_items_index(ud.resolved_items)

    # 收集目标 biosample 集合
    targets = _collect_target_biosamples(ud)
    LOGGER.info("Target biosamples: %d", len(targets))

    # pub_year 索引（hoist 同层级仲裁用）
    pmid_year = _pmid_year_index(ud)
    # 顺便从 paper_down/pmid_year.txt 读一份（如果 step6 那时复用的话）
    pmid_year_path = cfg.input_dir / "paper_down" / "pmid_year.txt"
    if pmid_year_path.exists():
        with open(pmid_year_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        pmid_year[parts[0]] = int(parts[1])
                    except ValueError:
                        pass

    # Phase 7c hoist + Phase 7b inline normalize
    biosamples: List[BioSampleRecord] = []
    inherit_level_counter: Counter = Counter()
    norm_status_counter: Counter = Counter()
    normalizer_usage_counter: Counter = Counter()

    for bs_id in targets:
        related = list(ud.biosample_to_related.get(bs_id, set()))
        bp_id = ud.biosample_to_bioproject.get(bs_id, "")
        chosen = hoist_one_biosample(
            bs_id, related, bp_id, acc_to_items, pmid_year,
        )
        if not chosen:
            continue

        env_value = ud.env_by_biosample.get(bs_id, "unknown")
        # value 归一化
        final_metas: List[FinalMetadataEntry] = []
        for item, source_level in chosen:
            cde_entry = _lookup_cde_entry(
                item.get("canonical_slot", ""), env_value, ud.cde,
            )
            normalizer_usage_counter[cde_entry.get("normalizer", "passthrough")] += 1
            norm_value = normalize(item.get("value", ""), cde_entry, ud.envo_index)
            norm_status_counter[norm_value.normalize_status] += 1
            inherit_level_counter[source_level] += 1
            final_metas.append(_build_final_entry(item, norm_value, source_level))

        biosamples.append(BioSampleRecord(
            biosample_id=bs_id,
            parent_project=bp_id,
            runs=ud.biosample_to_runs.get(bs_id, []),
            environment=env_value,
            paper_dominant_env=_build_paper_dominant_env(bs_id, ud),
            formal_name=_formal_name(bs_id, ud),
            aliases=_build_aliases(bs_id, ud),
            source_pmids=_build_source_pmids(bs_id, ud),
            metadata=final_metas,
        ))

    stats = {
        "biosample_count": len(biosamples),
        "metadata_entry_count": sum(len(b.metadata) for b in biosamples),
        "source_level_distribution": dict(inherit_level_counter),
        "normalize_status_distribution": dict(norm_status_counter),
        "normalizer_usage": dict(normalizer_usage_counter),
        "environment_distribution": dict(Counter(b.environment for b in biosamples)),
    }
    output = Step7Output(samples=biosamples, stats=stats)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / cfg.output_file
    stats_path = cfg.output_dir / cfg.stats_file
    out_path.write_text(
        json.dumps(output.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Wrote %s (%d biosamples)", out_path, len(biosamples))
    LOGGER.info("Wrote %s", stats_path)
