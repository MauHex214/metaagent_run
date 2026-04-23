"""Step7 Phase 7c — Sample 级 Hoist。

规则：
  对每个 BioSample（SAMN/SAMEA/SAMD）独立处理：
    1. SAMN 自身条目作为基础集
    2. 对 SAMN 缺失的每个 slot，按优先级 SRS > SRX > SRR > PRJNA 查找
    3. 同层级多候选时复用 step6 仲裁 (modality, section, pub_year)
    4. 找到即停止，标记 source_level
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .config import HOIST_LEVEL_PRIORITY
# MODALITY/SECTION 优先级直接从 step6 模块复用，避免重复定义
from metaagent_run.steps.step6.config import (
    MODALITY_PRIORITY,
    MODALITY_DEFAULT_PRIORITY,
    SECTION_PRIORITY,
    SECTION_DEFAULT_PRIORITY,
    PUB_YEAR_MISSING_VALUE,
)

LOGGER = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Accession 层级判定
# ═══════════════════════════════════════════════════════════

ACCESSION_LEVEL_PREFIXES: Dict[str, str] = {
    # BioProject
    "PRJNA": "bioproject", "PRJEB": "bioproject", "PRJDB": "bioproject",
    # BioSample
    "SAMN": "biosample", "SAMEA": "biosample", "SAMD": "biosample",
    # SRA Sample
    "SRS": "sra_sample", "ERS": "sra_sample", "DRS": "sra_sample",
    # SRA Experiment
    "SRX": "sra_experiment", "ERX": "sra_experiment", "DRX": "sra_experiment",
    # SRA Run
    "SRR": "sra_run", "ERR": "sra_run", "DRR": "sra_run",
    # SRA Study：用于辨识，hoist 时与 bioproject 同级处理
    "SRP": "sra_study", "ERP": "sra_study", "DRP": "sra_study",
    "SRA": "sra_study", "ERA": "sra_study", "DRA": "sra_study",
}


def classify_accession_level(acc: str) -> str:
    if not acc:
        return "unknown"
    # 按前缀长度从长到短匹配（避免 SRP 误命中 SR* 之类）
    for prefix in sorted(ACCESSION_LEVEL_PREFIXES.keys(), key=len, reverse=True):
        if acc.startswith(prefix):
            return ACCESSION_LEVEL_PREFIXES[prefix]
    return "unknown"


# ═══════════════════════════════════════════════════════════
#  仲裁 key（与 step6 一致，用于同层级多候选时的 tiebreak）
# ═══════════════════════════════════════════════════════════

def _arbitration_key_for_entry(entry: Dict, pmid_year: Dict[str, int]) -> Tuple[int, int, int]:
    """对一条 step6 ResolvedMetadataItem-as-dict 计算仲裁键。"""
    modality_rank = MODALITY_PRIORITY.get(
        entry.get("extraction_modality", ""), MODALITY_DEFAULT_PRIORITY,
    )
    section_rank = SECTION_PRIORITY.get(
        entry.get("section_type", ""), SECTION_DEFAULT_PRIORITY,
    )
    pmid = entry.get("authoritative_pmid", "")
    pub_year = pmid_year.get(pmid, PUB_YEAR_MISSING_VALUE)
    return (modality_rank, section_rank, pub_year)


# ═══════════════════════════════════════════════════════════
#  Hoist 主流程
# ═══════════════════════════════════════════════════════════

def build_acc_to_items_index(
    resolved_items: List[Dict],
) -> Dict[str, Dict[str, List[Dict]]]:
    """构建 raw_accession → canonical_slot → [items]（同 acc-slot 通常只一条；预留多条）。"""
    idx: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for item in resolved_items:
        acc = item.get("raw_accession", "")
        slot = item.get("canonical_slot", "")
        if not acc or not slot:
            continue
        idx[acc][slot].append(item)
    return idx


def hoist_one_biosample(
    biosample_id: str,
    related_accs: List[str],
    bioproject_id: str,
    acc_to_items: Dict[str, Dict[str, List[Dict]]],
    pmid_year: Dict[str, int],
) -> List[Tuple[Dict, str]]:
    """对单个 BioSample 执行 hoist，返回 [(item, source_level), ...]。"""
    # 桶装：level → slot → list[item]
    bucket: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))

    # 自己 = biosample 层
    for slot, items in acc_to_items.get(biosample_id, {}).items():
        bucket["biosample"][slot].extend(items)

    # 其他相关 accession（来自 accession_list 同行的 SRS/SRX/SRR）
    for acc in related_accs:
        if acc == biosample_id:
            continue
        level = classify_accession_level(acc)
        if level == "unknown":
            continue
        for slot, items in acc_to_items.get(acc, {}).items():
            bucket[level][slot].extend(items)

    # 父 bioproject
    if bioproject_id:
        for slot, items in acc_to_items.get(bioproject_id, {}).items():
            bucket["bioproject"][slot].extend(items)

    # 按继承顺序选 slot
    chosen: List[Tuple[Dict, str]] = []
    seen_slots: set = set()
    level_order = sorted(bucket.keys(), key=lambda lv: HOIST_LEVEL_PRIORITY.get(lv, 99))
    for level in level_order:
        for slot, items in bucket[level].items():
            if slot in seen_slots:
                continue
            # 同层级多候选 → 复用 step6 仲裁规则
            if len(items) == 1:
                chosen.append((items[0], level))
            else:
                top = sorted(items, key=lambda it: _arbitration_key_for_entry(it, pmid_year))[0]
                chosen.append((top, level))
            seen_slots.add(slot)

    return chosen
