"""Step7 上游加载：step6 + accession_list + DB metadata + env_tag + envo + CDE + step5."""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import RuntimeConfig

LOGGER = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  数据容器
# ═══════════════════════════════════════════════════════════

@dataclass
class UpstreamData:
    # step6 主输入
    resolved_items: List[Dict[str, Any]] = dc_field(default_factory=list)

    # accession 骨架（step7 hoist 用）
    # row: (run, bioproject, biosample, sra_study_old, sra_study, sra_sample, sra_experiment, pmid)
    accession_list_rows: List[Tuple[str, ...]] = dc_field(default_factory=list)
    # any accession → biosample_id
    acc_to_biosample: Dict[str, str] = dc_field(default_factory=dict)
    # biosample → bioproject
    biosample_to_bioproject: Dict[str, str] = dc_field(default_factory=dict)
    # biosample → set of related accessions（含 SRR/SRX/SRS/etc.）
    biosample_to_related: Dict[str, Set[str]] = dc_field(default_factory=lambda: defaultdict(set))
    # biosample → list of runs
    biosample_to_runs: Dict[str, List[str]] = dc_field(default_factory=lambda: defaultdict(list))
    # bioproject → list of biosamples
    bioproject_to_biosamples: Dict[str, List[str]] = dc_field(default_factory=lambda: defaultdict(list))

    # BioSample DB metadata（用于 formal_name 等）
    biosample_metadata: Dict[str, Dict[str, Any]] = dc_field(default_factory=dict)

    # step4a env_tag: biosample → env value
    env_by_biosample: Dict[str, str] = dc_field(default_factory=dict)

    # step5 PaperOutput-as-dict （取 paper_dominant_env + 别名）
    paper_outputs: List[Dict[str, Any]] = dc_field(default_factory=list)
    pmid_to_dominant_env: Dict[str, str] = dc_field(default_factory=dict)
    # raw_acc → set of aliases (LLM-discovered, from step5)
    aliases_by_acc: Dict[str, Set[str]] = dc_field(default_factory=lambda: defaultdict(set))
    # raw_acc → set of pmids that mentioned it
    pmids_by_acc: Dict[str, Set[str]] = dc_field(default_factory=lambda: defaultdict(set))

    # ENVO term index（lower_name → entry）
    envo_index: Dict[str, Dict[str, Any]] = dc_field(default_factory=dict)

    # CDE: env → key → entry
    cde: Dict[str, Dict[str, Dict[str, Any]]] = dc_field(default_factory=dict)


# ═══════════════════════════════════════════════════════════
#  Loaders
# ═══════════════════════════════════════════════════════════

def _load_step6(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError("step6 output not found: %s" % path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("resolved_items", []) if isinstance(data, dict) else data
    LOGGER.info("Loaded step6 resolved_items: %d", len(items))
    return items


def _load_accession_list(path: Path, ud: UpstreamData) -> None:
    """读 accession_list.tsv，构建 accession 骨架。

    每行 8 列：run / bioproject / biosample / sra_study_old / sra_study / sra_sample / sra_experiment / pmid
    """
    if not path.exists():
        LOGGER.warning("accession_list not found: %s", path)
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            run, bp, bs = parts[0].strip(), parts[1].strip(), parts[2].strip()
            sra_study_old = parts[3].strip()
            srp = parts[4].strip()
            srs = parts[5].strip()
            srx = parts[6].strip()
            pmid = parts[7].strip()
            row = (run, bp, bs, sra_study_old, srp, srs, srx, pmid)
            ud.accession_list_rows.append(row)
            if not bs:
                continue
            # biosample → 各级 accession
            for acc in (run, bp, bs, sra_study_old, srp, srs, srx):
                if acc:
                    ud.acc_to_biosample[acc] = bs
            ud.biosample_to_bioproject[bs] = bp
            for acc in (run, sra_study_old, srp, srs, srx):
                if acc:
                    ud.biosample_to_related[bs].add(acc)
            if run:
                if run not in ud.biosample_to_runs[bs]:
                    ud.biosample_to_runs[bs].append(run)
            if bp:
                if bs not in ud.bioproject_to_biosamples[bp]:
                    ud.bioproject_to_biosamples[bp].append(bs)
    LOGGER.info(
        "Loaded accession_list: %d rows, %d biosamples, %d bioprojects",
        len(ud.accession_list_rows),
        len(ud.biosample_to_bioproject),
        len(ud.bioproject_to_biosamples),
    )


def _load_biosample_metadata(path: Path, ud: UpstreamData) -> None:
    if not path.exists():
        LOGGER.warning("expanded_metadata not found: %s", path)
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        LOGGER.warning("expanded_metadata is not a list, skipping")
        return
    for item in data:
        bs = item.get("biosample_id", "")
        if bs:
            ud.biosample_metadata[bs] = item
    LOGGER.info("Loaded biosample_metadata: %d entries", len(ud.biosample_metadata))


def _load_env_tag(path: Path, ud: UpstreamData) -> None:
    if not path.exists():
        LOGGER.warning("env_tag file not found: %s", path)
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        LOGGER.warning("env_tag is not a list, skipping")
        return
    for item in data:
        bs = item.get("biosample_id", "")
        env_obj = item.get("env_tag", {})
        env_value = env_obj.get("value", "") if isinstance(env_obj, dict) else str(env_obj)
        if bs and env_value:
            ud.env_by_biosample[bs] = env_value
    LOGGER.info("Loaded env_tag: %d entries", len(ud.env_by_biosample))


def _load_step5_paper_dominant_env(path: Path, ud: UpstreamData) -> None:
    """读 step5_output.json，提取 pmid → environment 和 raw_acc 的别名/pmids。"""
    if not path.exists():
        LOGGER.warning("step5_output not found: %s (paper_dominant_env will be empty)", path)
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return
    ud.paper_outputs = data
    for paper in data:
        pmid = str(paper.get("pmid", ""))
        env = str(paper.get("environment", ""))
        if pmid and env:
            ud.pmid_to_dominant_env[pmid] = env
        for sample in paper.get("samples", []):
            acc = str(sample.get("accession", ""))
            if not acc:
                continue
            for label in sample.get("labels", []):
                if label:
                    ud.aliases_by_acc[acc].add(str(label))
            if pmid:
                ud.pmids_by_acc[acc].add(pmid)
    LOGGER.info(
        "Loaded step5 paper_outputs: %d papers, %d unique aliased accessions",
        len(data), len(ud.aliases_by_acc),
    )


def _load_envo_terms(path: Path, ud: UpstreamData) -> None:
    if not path.exists():
        LOGGER.warning("envo_terms not found: %s", path)
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 字典是 {ID: {id, name, def, ...}}；构建 name_lower → entry
    skipped = 0
    for term_id, entry in data.items():
        if not isinstance(entry, dict):
            continue
        name = entry.get("name", "")
        if isinstance(name, list):
            # 某些 OBO 条目的 name 是别名列表，取第一个非空字符串
            name = next((str(n) for n in name if n), "")
        if not isinstance(name, str) or not name:
            skipped += 1
            continue
        key = name.strip().lower()
        if key and key not in ud.envo_index:
            ud.envo_index[key] = {"id": str(entry.get("id", term_id)), "name": name}
    if skipped:
        LOGGER.info("Skipped %d envo entries with non-string name", skipped)
    LOGGER.info("Loaded envo_index: %d unique term names", len(ud.envo_index))


def _load_cde(path: Path, ud: UpstreamData) -> None:
    if not path.exists():
        LOGGER.warning(
            "CDE file not found: %s (will use passthrough for all values)", path,
        )
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 期望结构 {env: {key: cde_entry}}
    if isinstance(data, dict):
        ud.cde = data
    LOGGER.info(
        "Loaded CDE: %d environments, total entries=%d",
        len(ud.cde),
        sum(len(v) for v in ud.cde.values() if isinstance(v, dict)),
    )


def load_upstream(cfg: RuntimeConfig) -> UpstreamData:
    ud = UpstreamData()
    ud.resolved_items = _load_step6(cfg.input_dir / cfg.step6_output_file)
    _load_accession_list(cfg.input_dir / cfg.accession_list_file, ud)
    _load_biosample_metadata(cfg.input_dir / cfg.expanded_metadata_file, ud)
    _load_env_tag(cfg.input_dir / cfg.env_tag_file, ud)
    _load_step5_paper_dominant_env(cfg.input_dir / cfg.step5_output_file, ud)
    _load_envo_terms(cfg.input_dir / cfg.envo_terms_file, ud)
    _load_cde(cfg.input_dir / cfg.cde_file, ud)
    return ud
