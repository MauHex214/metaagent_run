import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from metaagent_run.core import INSDC_ACCESSION_RE

from .schemas import DiscoveryStats, PaperRelationInfo, RelationRecord


LOGGER = logging.getLogger(__name__)


SECTION_PRIORITY = {
    "ABSTRACT": 5,
    "METHODS": 4,
    "RESULTS": 3,
    "INTRO": 2,
    "DISCUSS": 2,
    "CONCL": 1,
    "TABLE": 0,
    "SUPPL": 0,
    "unknown": 0,
}

SECTION_ALIASES = {
    "abstract": "ABSTRACT",
    "intro": "INTRO",
    "introduction": "INTRO",
    "background": "INTRO",
    "method": "METHODS",
    "methods": "METHODS",
    "materials and methods": "METHODS",
    "materials & methods": "METHODS",
    "experimental": "METHODS",
    "result": "RESULTS",
    "results": "RESULTS",
    "finding": "RESULTS",
    "findings": "RESULTS",
    "discussion": "DISCUSS",
    "discuss": "DISCUSS",
    "conclusion": "CONCL",
    "conclusions": "CONCL",
    "concl": "CONCL",
    "supplement": "SUPPL",
    "supplementary": "SUPPL",
    "supporting": "SUPPL",
    "appendix": "SUPPL",
    "table": "TABLE",
    "tables": "TABLE",
}


def load_pmid_year_map(txt_file: Path) -> dict[str, str]:
    pmid_year_map: dict[str, str] = {}
    with txt_file.open("r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, 1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = re.split(r"\s+", stripped)
            if len(parts) < 2:
                LOGGER.warning("第%d行格式异常，已跳过: %s", line_num, stripped)
                continue
            pmid_year_map[parts[0].strip()] = parts[1].strip()
    return pmid_year_map


def get_pmid(item: dict[str, Any]) -> str:
    return str(item.get("pmid", "")).strip()


def normalize_section(raw: Any) -> str:
    text = str(raw).strip()
    if text.upper() in SECTION_PRIORITY:
        return text.upper()
    return SECTION_ALIASES.get(text.lower(), "unknown")


def build_relation_map(relation_items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    relation_map: dict[str, dict[str, Any]] = {}
    for item in relation_items:
        try:
            validated = RelationRecord.model_validate(item)
        except ValidationError:
            continue
        pmid = validated.pmid.strip()
        source = validated.source.strip()
        section_type = validated.section_type.strip()
        index = str(validated.index)
        relation = validated.relation.strip().lower()
        if pmid:
            relation_map[f"{pmid}__{source}__{section_type}__{index}"] = {
                "relation": relation,
                "accessions_found": item.get("accessions_found", []),
                "labels_found": item.get("labels_found", []),
                "metadata_keys_found": item.get("metadata_keys_found", []),
            }
    return relation_map


def classify_paper_relation_types(
    relation_map: dict[str, dict[str, Any]],
) -> dict[str, PaperRelationInfo]:
    pmid_relations: dict[str, set[str]] = defaultdict(set)
    for key, record in relation_map.items():
        pmid = key.split("__", 1)[0]
        relation = record["relation"]
        # For accession-bearing relations, only count if at least one
        # accession matches INSDC prefix (BioProject/BioSample/SRX/SRR/etc.)
        if "accession" in relation:
            accessions = record.get("accessions_found", [])
            if not any(INSDC_ACCESSION_RE.search(acc) for acc in accessions):
                continue  # skip: no valid INSDC accession in this paragraph
        pmid_relations[pmid].add(relation)

    info: dict[str, PaperRelationInfo] = {}
    for pmid, relations in pmid_relations.items():
        has_am = "accession-metadata" in relations
        has_alm = "accession-label-metadata" in relations
        has_al = "accession-label" in relations
        has_lm = "label-metadata" in relations
        has_c = has_al and has_lm
        is_valid = has_am or has_alm or has_c

        modes: list[str] = []
        if has_am:
            modes.append("A")
        if has_alm:
            modes.append("B")
        if has_c:
            modes.append("C")

        info[pmid] = PaperRelationInfo(
            has_accession_metadata=has_am,
            has_accession_label_metadata=has_alm,
            has_accession_label=has_al,
            has_label_metadata=has_lm,
            is_valid=is_valid,
            build_mode="".join(modes) if modes else "none",
        )
    return info


def build_discovery_input(
    full_items: list[dict[str, Any]],
    valid_target_pmids: set[str],
    relation_map: dict[str, dict[str, Any]],
    pmid_year_map: dict[str, str],
    paper_env_map: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], DiscoveryStats]:
    # 仍然计算 paper_relation_info，用于在每条 out_item 上附带 build_mode 供
    # 下游（Step 4-meta / 论文 Method）做诊断与 INSDC-deposit-subset 分层分析，
    # 但**不再作为 discovery 的保留门槛**。
    #
    # Step 3 的目标是发现水圈环境领域的「样本元数据字段全景」，若按 Step 5 抽取
    # 路径所需的 Mode A/B/C 论文级有效性 + relation 组合段落级条件过滤，会把
    # 字段视野窄化到"已提交 INSDC 的子集"，对 Lake / Wetlands / 早期研究不公平。
    # 因此此处的保留条件退化为两条：
    #   (1) pmid ∈ valid_target_pmids  —— 来自 Step 1 的四目标环境筛选
    #   (2) metadata_keys_found 非空   —— 段落确实承载元数据信号
    # 由 relation taxonomy 的互斥性，条件 (2) 等价于
    #   relation ∈ {accession-metadata, accession-label-metadata, label-metadata}
    # 即自动排除 accession-label（定义上无 C）与 unknown（整体无效）。
    paper_relation_info = classify_paper_relation_types(relation_map)

    discovery_items: list[dict[str, Any]] = []
    skip_non_target_paper = 0
    skip_abstract = 0
    skip_no_metadata = 0

    for item in full_items:
        pmid = get_pmid(item)
        if pmid not in valid_target_pmids:
            skip_non_target_paper += 1
            continue

        normalized_section = normalize_section(item.get("section_type", ""))
        if normalized_section == "ABSTRACT":
            skip_abstract += 1
            continue

        source = str(item.get("source", "")).strip()
        section_type_raw = str(item.get("section_type", "")).strip()
        index = str(item.get("index", "")).strip()
        para_key = f"{pmid}__{source}__{section_type_raw}__{index}"
        record = relation_map.get(para_key, {})
        if not isinstance(record, dict) or not record:
            # 精确键未命中，回退到 normalize 后的 section_type 二次查找，
            # 兼容抽取/构建阶段 section 口径不一致的历史遗留。
            normalized_para_key = (
                f"{pmid}__{source}__{normalize_section(section_type_raw)}__{index}"
            )
            record = relation_map.get(normalized_para_key, {})

        if not isinstance(record, dict):
            record = {}
        metadata_keys = record.get("metadata_keys_found", []) or []
        if not metadata_keys:
            skip_no_metadata += 1
            continue

        relation = str(record.get("relation", "")).lower()
        relation_info = paper_relation_info.get(pmid)
        out_item = {
            **item,
            "pub_year": pmid_year_map.get(pmid),
            "relation": relation,
            "section_type": normalized_section,
            "build_mode": relation_info.build_mode if relation_info is not None else "none",
            "sub_env": paper_env_map.get(pmid, ["Others"]),
            "accessions_found": record.get("accessions_found", []),
            "labels_found": record.get("labels_found", []),
            "metadata_keys_found": metadata_keys,
        }
        discovery_items.append(out_item)

    stats = DiscoveryStats(
        kept=len(discovery_items),
        skipped_non_target_paper=skip_non_target_paper,
        skipped_abstract=skip_abstract,
        skipped_no_metadata=skip_no_metadata,
    )

    build_mode_counter = Counter(
        str(item.get("build_mode", "none")) for item in discovery_items
    )
    LOGGER.info("保留段落build_mode分布: %s", dict(build_mode_counter))
    return discovery_items, stats
