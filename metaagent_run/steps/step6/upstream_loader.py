"""Step6 上游加载：step5 + step4b + step3 + pmid_year。"""

import json
import logging
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .config import RuntimeConfig
from .schemas import Candidate

LOGGER = logging.getLogger(__name__)


@dataclass
class UpstreamData:
    """汇总所有步骤产物的只读容器。"""
    pmid_year: Dict[str, int] = dc_field(default_factory=dict)
    tier1_slots: Set[str] = dc_field(default_factory=set)        # MIxS slot 名集合（不带 mixs: 前缀）
    raw_field_to_canonical: Dict[str, str] = dc_field(default_factory=dict)
    paper_outputs: List[Dict[str, Any]] = dc_field(default_factory=list)


# ═══════════════════════════════════════════════════════════
#  Loader 实现
# ═══════════════════════════════════════════════════════════

def _load_pmid_year(path: Path) -> Dict[str, int]:
    """读 paper_down/pmid_year.txt：每行 '<pmid> <year>'。"""
    result: Dict[str, int] = {}
    if not path.exists():
        LOGGER.warning("pmid_year file not found: %s", path)
        return result
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                result[parts[0]] = int(parts[1])
            except ValueError:
                continue
    LOGGER.info("Loaded pmid_year: %d entries", len(result))
    return result


def _load_env_targets(path: Path) -> Set[str]:
    """读 step4b_env_extraction_targets.json，返回 Tier 1 MIxS slot 集合。

    结构：
      {
        "per_environment": {
          "Open_ocean": {"fields": [{"slot": "...", "tier": 1, "mapped": true, ...}, ...]},
          ...
        }
      }
    """
    if not path.exists():
        LOGGER.warning("env_targets file not found: %s", path)
        return set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    slots: Set[str] = set()
    per_env = data.get("per_environment", {})
    for env, env_data in per_env.items():
        if not isinstance(env_data, dict):
            continue
        for f in env_data.get("fields", []):
            if not isinstance(f, dict):
                continue
            if f.get("tier") == 1 and f.get("mapped"):
                slot = f.get("slot") or f.get("mixs_slot")
                if slot:
                    slots.add(str(slot))
    LOGGER.info("Loaded Tier 1 MIxS slots: %d", len(slots))
    return slots


def _load_synonym_groups(path: Path) -> Dict[str, str]:
    """读 step3_schema_discovery_result_review.json，构建 alias(lowercase) → canonical 反向映射。

    synonym_groups 结构: {canonical: [alias1, alias2, ...]}
    反向：每个 alias（含 canonical 自身）映射到 canonical。
    """
    if not path.exists():
        LOGGER.warning("schema_discovery file not found: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sg = data.get("synonym_groups", {})
    reverse: Dict[str, str] = {}
    for canonical, aliases in sg.items():
        canonical_lower = str(canonical).strip().lower()
        if canonical_lower:
            reverse[canonical_lower] = canonical_lower
        if isinstance(aliases, list):
            for a in aliases:
                a_lower = str(a).strip().lower()
                if a_lower:
                    reverse.setdefault(a_lower, canonical_lower)
    LOGGER.info("Built alias→canonical map: %d entries (from %d synonym groups)",
                len(reverse), len(sg))
    return reverse


def _load_step5_output(path: Path) -> List[Dict[str, Any]]:
    """读 step5_output.json。返回 List[PaperOutput-as-dict]。"""
    if not path.exists():
        raise FileNotFoundError("step5 output not found: %s" % path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("step5_output.json must be a JSON array")
    LOGGER.info("Loaded step5 output: %d papers", len(data))
    return data


def load_upstream(cfg: RuntimeConfig) -> UpstreamData:
    """统一入口。按 RuntimeConfig 解析所有上游路径。"""
    ud = UpstreamData()
    ud.pmid_year = _load_pmid_year(cfg.input_dir / cfg.pmid_year_file)
    ud.tier1_slots = _load_env_targets(cfg.input_dir / cfg.env_targets_file)
    ud.raw_field_to_canonical = _load_synonym_groups(
        cfg.input_dir / cfg.schema_discovery_file
    )
    ud.paper_outputs = _load_step5_output(cfg.input_dir / cfg.step5_output_file)
    return ud


# ═══════════════════════════════════════════════════════════
#  Canonical slot 归属判定
# ═══════════════════════════════════════════════════════════

def resolve_canonical_slot(
    mixs_slot: Optional[str],
    raw_field: str,
    raw_field_to_canonical: Dict[str, str],
    tier1_slots: Optional[Set[str]] = None,
) -> Optional[str]:
    """决定一条 metadata 的 canonical_slot 归属。

    优先级：
      1. step5 已映射 mixs_slot                → "mixs:{slot}"
      2. raw_field 在 step3 synonym_groups 内  → "internal:{canonical}"
      3. 没有 canonical 兜底                   → "internal:{raw_field_lower}"

    设计说明：step6 不做 Tier 校正（即不强制 mixs_slot 必须出现在 step4b
    Tier 1 集合内）。原因：
      - step4b Tier 1 集合的命名不一定对齐 MIxS 主表（如 step4b 用 'Water_depth'
        而 MIxS 主表用 'depth'），强制校正会把真正的 MIxS 标准 slot 误降级
      - step7 CDE lookup 会自动 fallback：mixs:xxx 找不到 → 试 internal:xxx
        → passthrough，已经覆盖了语义不一致的边界情况
      - 信任 step5 Phase C1 的语义映射结果（mixs_slot 字段）

    `tier1_slots` 参数保留是为了向后兼容，当前未使用。
    """
    rf_lower = raw_field.strip().lower()
    if mixs_slot:
        return "mixs:" + str(mixs_slot)
    canonical = raw_field_to_canonical.get(rf_lower)
    if canonical:
        return "internal:" + canonical
    # 兜底：用 raw_field 自身的小写形式
    return "internal:" + rf_lower if rf_lower else None


# ═══════════════════════════════════════════════════════════
#  扁平化：所有 PaperOutput → List[Candidate]
# ═══════════════════════════════════════════════════════════

def flatten_candidates(ud: UpstreamData) -> List[Candidate]:
    """把 step5_output 中所有 metadata 扁平化为 Candidate 列表。"""
    candidates: List[Candidate] = []
    skipped_no_slot = 0
    for paper in ud.paper_outputs:
        pmid = str(paper.get("pmid", ""))
        pub_year = ud.pmid_year.get(pmid)
        for sample in paper.get("samples", []):
            raw_acc = str(sample.get("accession", ""))
            if not raw_acc:
                continue
            for m in sample.get("metadata", []):
                if not isinstance(m, dict):
                    continue
                raw_field = str(m.get("raw_field", "")).strip()
                value = str(m.get("value", "")).strip()
                if not raw_field or not value:
                    continue
                canonical_slot = resolve_canonical_slot(
                    m.get("mixs_slot"), raw_field, ud.raw_field_to_canonical,
                    tier1_slots=ud.tier1_slots,
                )
                if canonical_slot is None:
                    skipped_no_slot += 1
                    continue
                # provenance fields default-friendly
                try:
                    paragraph_index = int(m.get("paragraph_index", -1))
                except (ValueError, TypeError):
                    paragraph_index = -1
                # 优先用 metadata 自带 pmid（step5 v2 已填），否则用 paper.pmid
                cand_pmid = str(m.get("pmid") or pmid)
                candidates.append(Candidate(
                    raw_accession=raw_acc,
                    canonical_slot=canonical_slot,
                    raw_field=raw_field,
                    value=value,
                    pmid=cand_pmid,
                    pub_year=pub_year,
                    source_file=str(m.get("source_file", "")),
                    section_type=str(m.get("section_type", "")),
                    paragraph_index=paragraph_index,
                    extraction_modality=str(m.get("source", "")),
                ))
    if skipped_no_slot:
        LOGGER.info("Skipped %d items with no resolvable canonical_slot", skipped_no_slot)
    LOGGER.info("Flattened candidates: %d", len(candidates))
    return candidates
