"""统一加载所有上游 step 产物，构建 per-paper context。"""

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

LOGGER = logging.getLogger(__name__)


def _segment_hash(text: str) -> str:
    """Stable per-segment identifier for disambiguating multiple step2 records
    that share (section_type, index) but have different text. Used as the
    second-level key in accession_by_section / step2_metadata_keys_index so
    we can ask "what accessions did step2 detect in THIS exact segment" rather
    than "what accessions across all segments at (section_type, index)"."""
    return hashlib.md5((text or "").encode("utf-8", errors="replace")).hexdigest()


# ═══════════════════════════════════════════════════════════
#  数据容器
# ═══════════════════════════════════════════════════════════

@dataclass
class UpstreamData:
    """汇聚所有上游产物的只读索引。"""

    # step2: pmid → {(section_type, index) → relation_type}
    relation_index: Dict[str, Dict[Tuple[str, int], str]] = field(default_factory=dict)

    # step2: pmid → {(section_type, index) → [accession, ...]}
    # Replaces Step 3a (step3_accession) data source
    step2_accessions_index: Dict[str, Dict[Tuple[str, int], List[str]]] = field(default_factory=dict)

    # step2: pmid → {(section_type, index) → [label, ...]}
    step2_labels_index: Dict[str, Dict[Tuple[str, int], List[str]]] = field(default_factory=dict)

    # step2: pmid → {(section_type, index) → [metadata_key, ...]}
    # NOTE: step2 records overwrite by (section_type, index) key — multiple
    # text segments at the same key collapse, only the last record's data
    # survives. step5 input texts are not preserved on the step2 side, so
    # per-segment disambiguation isn't possible. Giant-table filtering at
    # build_paper_context time uses pre-computed (section_type, index) keys
    # of giant segments and drops the whole (st, idx) entries.
    step2_metadata_keys_index: Dict[str, Dict[Tuple[str, int], List[str]]] = field(default_factory=dict)
    accession_by_section: Dict[str, Dict[Tuple[str, int], List[str]]] = field(default_factory=dict)

    # 外部 DB 验证: accession → {biosample, bioproject, pmid}
    verified_mapping: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # pmid → set of verified accessions
    verified_acc_by_pmid: Dict[str, Set[str]] = field(default_factory=dict)

    # pmid_run_merged_data_expanded: biosample_id → metadata dict
    biosample_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # pmid → [biosample_id, ...]
    biosamples_by_pmid: Dict[str, List[str]] = field(default_factory=dict)

    # step4 env_tag: biosample_id → env value
    env_by_biosample: Dict[str, str] = field(default_factory=dict)

    # env_extraction_targets: env → [{field, tier, subtype, aliases, ...}, ...]
    env_target_fields: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    # 全局字段 (Universal/Shared/Signature)
    global_fields: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # ── 查询方法 ──────────────────────────────────────────

    def get_all_labels_for_pmid(self, pmid: str) -> List[str]:
        """Return all Step 2 discovered labels for this paper (deduplicated)."""
        all_labels = []
        for labels in self.step2_labels_index.get(pmid, {}).values():
            all_labels.extend(labels)
        seen = set()
        result = []
        for label in all_labels:
            if label not in seen:
                seen.add(label)
                result.append(label)
        return result

    def get_all_step2_accessions_for_pmid(self, pmid: str) -> Set[str]:
        """Return all Step 2 discovered accessions for this paper (deduplicated)."""
        all_accs: Set[str] = set()
        for accs in self.step2_accessions_index.get(pmid, {}).values():
            all_accs.update(accs)
        return all_accs

    def get_section_relation(self, pmid: str, section_type: str, index: int) -> str:
        return self.relation_index.get(pmid, {}).get((section_type, index), "unknown")

    def get_section_accessions(self, pmid: str, section_type: str, index: int) -> List[str]:
        return self.accession_by_section.get(pmid, {}).get((section_type, index), [])

    def get_section_metadata_keys(
        self, pmid: str, section_type: str, index: int
    ) -> List[str]:
        """Return the list of metadata_key names step2 discovered in this section."""
        return self.step2_metadata_keys_index.get(pmid, {}).get((section_type, index), [])

    def get_all_step3_accessions_for_pmid(self, pmid: str) -> Set[str]:
        """返回该 paper 中 step3 (=step2) 提取的所有 accession 跨所有段落并集。"""
        result: Set[str] = set()
        for accs in self.accession_by_section.get(pmid, {}).values():
            result.update(accs)
        return result

    def get_verified_accessions_for_pmid(self, pmid: str) -> Set[str]:
        """返回该 paper 中外部 DB 验证过的所有 accession。"""
        return self.verified_acc_by_pmid.get(pmid, set())

    def get_verified_for_pmid(self, pmid: str) -> Dict[str, Dict[str, str]]:
        """返回 {accession: {biosample, bioproject}} 只限该 pmid。"""
        result = {}
        for acc in self.verified_acc_by_pmid.get(pmid, set()):
            if acc in self.verified_mapping:
                result[acc] = self.verified_mapping[acc]
        return result

    def get_dominant_env(self, pmid: str) -> str:
        """从该 paper 的 biosample 环境标签中投票得到主导环境。"""
        counter: Dict[str, int] = defaultdict(int)
        for bs_id in self.biosamples_by_pmid.get(pmid, []):
            env = self.env_by_biosample.get(bs_id, "")
            if env and env != "Others":
                counter[env] += 1
        if not counter:
            return "unknown"
        return max(counter, key=counter.get)

    def get_target_fields(self, env: str) -> Tuple[List[str], List[str]]:
        """返回 (tier1_fields, tier2_fields) 名称列表。"""
        env_key = self._normalize_env_key(env)
        fields = self.env_target_fields.get(env_key, [])
        tier1 = [f["field"] for f in fields if f.get("tier") == 1]
        tier2 = [f["field"] for f in fields if f.get("tier") == 2]
        if not tier1:
            tier1 = [f["field"] for f in self.global_fields.get("Universal", [])]
        return tier1, tier2

    def get_sample_name_hints(
        self, pmid: str, verified_accessions: Optional[Set[str]] = None,
        max_hints: int = 10,
    ) -> List[Dict[str, str]]:
        """从 BioSample 元数据中提取 sample_name 线索。

        只提供 verified_accessions（step3 ∩ db）中 accession 对应的 BioSample。
        不提供骨架扩展产生的、论文中未提及的 BioSample。
        """
        if verified_accessions is None:
            verified_accessions = set()

        # Build set of BioSample IDs that are directly linked to verified accessions
        relevant_biosamples: Set[str] = set()
        for acc in verified_accessions:
            # acc itself might be a BioSample
            if acc in self.biosample_metadata:
                relevant_biosamples.add(acc)
            # acc might link to a BioSample via verified_mapping
            vm = self.verified_mapping.get(acc, {})
            biosample_id = vm.get("biosample", "")
            if biosample_id and biosample_id in self.biosample_metadata:
                relevant_biosamples.add(biosample_id)

        hints = []
        for bs_id in relevant_biosamples:
            env = self.env_by_biosample.get(bs_id, "")
            if env == "Others":
                continue
            meta = self.biosample_metadata.get(bs_id, {})
            name = meta.get("sample_name") or meta.get("sample_title") or ""
            if name and name.lower() not in ("missing", "not collected", "not applicable", "na"):
                hints.append({"biosample_id": bs_id, "sample_name": name})
            if len(hints) >= max_hints:
                break
        return hints

    @staticmethod
    def _normalize_env_key(env: str) -> str:
        mapping = {
            "open_ocean": "Open_ocean", "coastal_waters": "Coastal_waters",
            "wetlands": "Wetlands", "lake": "Lake",
            "ocean": "Open_ocean", "coastal": "Coastal_waters", "wetland": "Wetlands",
        }
        return mapping.get(env.lower(), env.lower())


# ═══════════════════════════════════════════════════════════
#  Phase 0 — build_paper_context
# ═══════════════════════════════════════════════════════════

_ACCESSION_BEARING_RELATIONS = frozenset({
    "accession-label", "accession-metadata", "accession-label-metadata",
})

# Sections that carry metadata signals (any of: bound to acc directly, bound
# to label that needs alias bridging, or both). Used by both Phase B1+B2
# (table parser) and Phase B3 (LLM text extractor) — neither should miss
# accession-metadata text segments.
_METADATA_BEARING_RELATIONS = frozenset({
    "accession-metadata", "accession-label-metadata", "label-metadata",
})

_TARGET_ENVS = frozenset({"Open_ocean", "Coastal_waters", "Lake", "Wetlands"})

# Single source of truth for the "giant table" cutoff. Sections classified as
# structured tables with > GIANT_TABLE_MAX_COLS columns are skipped wholesale:
# their accs don't enter verified_accessions, their step2 metadata_keys are
# dropped from section_metadata_keys, and they're not passed to either Phase
# B1+B2 (table extraction) or Phase B3 (LLM extraction). One number controls
# the entire policy — do NOT introduce a separate threshold downstream.
GIANT_TABLE_MAX_COLS: int = 50


def _filter_others_accessions(
    accessions: Set[str],
    upstream: UpstreamData,
) -> Set[str]:
    """过滤掉 env_tag 为 Others 的 accession。

    策略：
    - 对每个 accession，通过 verified_mapping 找到其 biosample_id
    - 查 biosample 的 env_tag，只保留目标环境（非 Others）
    - BioProject 级别：只在其下有至少一个目标 biosample 时才保留
    """
    # 构建目标 biosample 和 bioproject 集合
    target_biosamples: Set[str] = set()
    target_bioprojects: Set[str] = set()

    for bs_id, env_val in upstream.env_by_biosample.items():
        if env_val in _TARGET_ENVS:
            target_biosamples.add(bs_id)
            vm = upstream.verified_mapping.get(bs_id, {})
            bp = vm.get("bioproject", "")
            if bp:
                target_bioprojects.add(bp)

    result: Set[str] = set()
    for acc in accessions:
        vm = upstream.verified_mapping.get(acc, {})
        biosample = vm.get("biosample", "")
        bioproject = vm.get("bioproject", "")

        # accession 自身就是目标 biosample
        if acc in target_biosamples:
            result.add(acc)
            continue

        # accession 关联的 biosample 是目标环境
        if biosample and biosample in target_biosamples:
            result.add(acc)
            continue

        # accession 是 bioproject 且其下有目标 biosample
        if acc in target_bioprojects:
            result.add(acc)
            continue

        # accession 关联的 bioproject 有目标 biosample
        if bioproject and bioproject in target_bioprojects:
            result.add(acc)
            continue

        # Others 或未知环境 → 不保留

    return result


def _build_accession_to_env(
    accessions: Set[str],
    upstream: UpstreamData,
) -> Dict[str, str]:
    """构建 accession -> env_tag 映射。

    通过 verified_mapping 找到 accession 关联的 biosample，再查 env_tag。
    BioProject 级别取其下目标 biosample 的主导环境。
    """
    result: Dict[str, str] = {}
    for acc in accessions:
        vm = upstream.verified_mapping.get(acc, {})
        biosample = vm.get("biosample", "")

        # accession 自身就是 biosample
        if acc in upstream.env_by_biosample:
            env = upstream.env_by_biosample[acc]
            if env in _TARGET_ENVS:
                result[acc] = env
                continue

        # 通过关联 biosample 查环境
        if biosample and biosample in upstream.env_by_biosample:
            env = upstream.env_by_biosample[biosample]
            if env in _TARGET_ENVS:
                result[acc] = env
                continue

    return result


def _get_union_target_fields(
    envs: Set[str],
    upstream: UpstreamData,
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """获取多个环境的 tier1/tier2 target fields 并集 + 每个 target 的 aliases map。

    Returns:
        (tier1_sorted, tier2_sorted, aliases_map)
        aliases_map: field_name -> alias list (可能来自多个 env 的并集，去重去自身)
    """
    tier1_set: Set[str] = set()
    tier2_set: Set[str] = set()
    aliases_map: Dict[str, Set[str]] = {}

    for env in envs:
        env_key = upstream._normalize_env_key(env)
        fields = upstream.env_target_fields.get(env_key, [])
        for f in fields:
            fname = f["field"]
            if f.get("tier") == 1:
                tier1_set.add(fname)
            elif f.get("tier") == 2:
                tier2_set.add(fname)
            # collect aliases (phase6 extraction_targets.json has an "aliases" list)
            a = f.get("aliases") or []
            if isinstance(a, list) and a:
                aliases_map.setdefault(fname, set()).update(
                    str(x).strip() for x in a if str(x).strip() and str(x).strip() != fname
                )

    # 如果没有任何 tier1，fallback 到全局 Universal
    if not tier1_set:
        tier1_set = {f["field"] for f in upstream.global_fields.get("Universal", [])}

    aliases_out: Dict[str, List[str]] = {
        k: sorted(v) for k, v in aliases_map.items()
    }
    return sorted(tier1_set), sorted(tier2_set), aliases_out


def build_paper_context(
    pmid: str,
    sections: List[Dict[str, Any]],
    upstream: UpstreamData,
    giant_table_max_cols: int = GIANT_TABLE_MAX_COLS,
) -> "PaperContext":
    """为单篇 paper 构建完整上下文（0408 design, paper-centric）。

    Single-pass section iteration with early skip of giant tables (列数 > N).
    Giant tables are dropped entirely from all downstream processing — their
    accessions don't enter verified_accessions, their metadata isn't extracted,
    their step2_keys aren't used. This keeps reference-citation accessions
    (e.g., a 454-column TARA Ocean Table S5) out of the paper context, so
    Phase A LLM enrichment runs on the paper's actual deposits only.

    Sections without semicolon-joined giant-table cutoff:
      1. acc 收集: step2 检测到的 acc per non-giant section (paper-mentioned)
      2. env_tag_v2 过滤 (drop "Others")
      3. accession → env 映射
      4. target fields = 所有目标环境的并集
      5. Section 分组 = accession-bearing / metadata-bearing
    """
    from .schemas import PaperContext
    from .table_parser import StructuredTableParser

    table_parser = StructuredTableParser()

    # Pre-compute (section_type, index) keys that contain ANY giant-table
    # segment. step2 records collapse by (sec_type, idx), so for the dict-
    # based acc lookup we must drop the whole (st, idx) entry — we can't
    # know per-segment which accs came from the giant vs sibling segments.
    giant_keys: Set[Tuple[str, int]] = set()
    for sec in sections:
        if (table_parser.is_structured_table(sec)
                and table_parser.count_columns(sec) > giant_table_max_cols):
            giant_keys.add((sec.get("section_type", ""),
                            int(sec.get("index", 0))))

    verified_accessions: Set[str] = set()
    accession_sections: List[Dict[str, Any]] = []
    metadata_sections: List[Dict[str, Any]] = []
    section_metadata_keys: Dict[Any, List[str]] = {}

    pmid_acc_idx = upstream.accession_by_section.get(pmid, {})
    pmid_keys_idx = upstream.step2_metadata_keys_index.get(pmid, {})

    # Acc collection: iterate the (st, idx)-keyed dict, skip giant keys.
    # Required because the dict collapses; can't tell which segment's accs
    # survive on collision.
    for sec_key, accs in pmid_acc_idx.items():
        if sec_key in giant_keys:
            continue
        verified_accessions.update(accs)

    for sec_key, keys in pmid_keys_idx.items():
        if sec_key in giant_keys:
            continue
        if keys:
            section_metadata_keys[sec_key] = keys

    # Section classification — disjoint partition.
    # Per-segment giant-skip here (NOT (st, idx)-level) so non-giant siblings
    # of a giant table (e.g., a 6-col Table S1 sharing supplementary-2 with
    # a 454-col Table S5) still flow into B1+B2 / B3 input.
    for sec in sections:
        if (table_parser.is_structured_table(sec)
                and table_parser.count_columns(sec) > giant_table_max_cols):
            continue   # skip THIS giant section, keep siblings
        sec_type = sec.get("section_type", "")
        sec_idx = int(sec.get("index", 0))
        relation = upstream.get_section_relation(pmid, sec_type, sec_idx)
        if relation == "accession-label":
            accession_sections.append(sec)
        elif relation in _METADATA_BEARING_RELATIONS:
            metadata_sections.append(sec)

    # env_tag_v2 过滤 (step4_accession's LLM-inferred target-env set)
    verified_accessions = _filter_others_accessions(verified_accessions, upstream)

    # accession → env 映射
    accession_to_env = _build_accession_to_env(verified_accessions, upstream)

    # target fields = 所有目标环境的并集
    involved_envs = set(accession_to_env.values())
    if not involved_envs:
        dom_env = upstream.get_dominant_env(pmid)
        if dom_env != "unknown":
            involved_envs = {dom_env}
    tier1, tier2, target_aliases = _get_union_target_fields(involved_envs, upstream)

    env = upstream.get_dominant_env(pmid)
    step2_labels = upstream.get_all_labels_for_pmid(pmid)

    return PaperContext(
        pmid=pmid,
        verified_accessions=verified_accessions,
        accession_to_env=accession_to_env,
        accession_sections=accession_sections,
        metadata_sections=metadata_sections,
        environment=env,
        tier1_fields=tier1,
        tier2_fields=tier2,
        target_field_aliases=target_aliases,
        section_metadata_keys=section_metadata_keys,
        step2_labels=step2_labels,
    )


# ═══════════════════════════════════════════════════════════
#  Phase 0 — build_identity_skeleton (programmatic)
# ════════════════════════════════════════════════════��══════

_JUNK_SAMPLE_NAMES = frozenset({
    "missing", "not collected", "not applicable", "na", "n/a", "none", "",
})


def build_identity_skeleton(
    paper_ctx: "PaperContext",
    upstream: "UpstreamData",
    max_samples: int = 50,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Phase 0: 程序化构建 identity_map 骨架。

    通过 accession_list.tsv 骨架找到每个 verified accession 对应的
    BioSample，再从 pmid_run_merged_data 获取 sample_name，无需 LLM。

    当 verified_accessions 过多时（> max_samples），压缩到 BioSample + BioProject
    层级：将 SRR/ERR/SRX 等 run-level accession 合并到其 BioSample 层级。

    Returns:
        identity_map: Dict[str, SampleIdentity] keyed by accession
        alias_to_accession: Dict[str, str] mapping names to accession
    """
    from .schemas import SampleIdentity, IdentityMap

    verified = paper_ctx.verified_accessions

    # ── Check if compression is needed ──────────────────
    if len(verified) > max_samples:
        verified = _compress_to_biosample_level(verified, upstream)
        LOGGER.info("[Skeleton] PMID %s: compressed %d accessions -> %d (BioSample+BioProject level)",
                    paper_ctx.pmid, len(paper_ctx.verified_accessions), len(verified))

    identity_map: Dict[str, SampleIdentity] = {}

    for acc in verified:
        vm = upstream.verified_mapping.get(acc, {})
        biosample_id = vm.get("biosample", "")
        bioproject_id = vm.get("bioproject", "")

        # formal_name: from BioSample metadata sample_name (skip for BioProject)
        sample_name = ""
        if not acc.startswith("PRJ") and biosample_id and biosample_id in upstream.biosample_metadata:
            raw_name = upstream.biosample_metadata[biosample_id].get("sample_name", "")
            if raw_name and raw_name.strip().lower() not in _JUNK_SAMPLE_NAMES:
                sample_name = raw_name.strip()

        # parent_project: only for non-PRJ accessions
        parent_project = ""
        if not acc.startswith("PRJ") and bioproject_id:
            parent_project = bioproject_id

        # environment: try acc directly, then via linked biosample
        env = upstream.env_by_biosample.get(acc, "")
        if not env and biosample_id:
            env = upstream.env_by_biosample.get(biosample_id, "")
        if not env:
            env = paper_ctx.accession_to_env.get(acc, "unknown")

        identity_map[acc] = SampleIdentity(
            accession=acc,
            formal_name=sample_name,
            aliases=[],
            parent_project=parent_project,
            environment=env,
        )

    # BioProject aggregation: collect child sample_names as aliases
    for acc, identity in list(identity_map.items()):
        if not acc.startswith("PRJ"):
            continue
        child_names: List[str] = []
        for child_acc, child_ident in identity_map.items():
            if child_ident.parent_project == acc and child_acc != acc:
                if child_ident.formal_name:
                    child_names.append(child_ident.formal_name)
        existing = set(identity.aliases)
        for name in child_names:
            if name not in existing:
                identity.aliases.append(name)
                existing.add(name)

    # Build alias_to_accession (child samples first, then projects)
    alias_to_accession: Dict[str, str] = {}

    # Pass 1: BioProject accessions (lower priority for shared aliases)
    for acc, identity in identity_map.items():
        if not acc.startswith("PRJ"):
            continue
        alias_to_accession[acc] = acc
        if identity.formal_name:
            alias_to_accession[identity.formal_name] = acc
            alias_to_accession[identity.formal_name.lower()] = acc
        for alias in identity.aliases:
            alias_to_accession[alias] = acc
            alias_to_accession[alias.lower()] = acc

    # Pass 2: Child sample accessions (higher priority — overwrites project mappings)
    for acc, identity in identity_map.items():
        if acc.startswith("PRJ"):
            continue
        alias_to_accession[acc] = acc
        if identity.formal_name:
            alias_to_accession[identity.formal_name] = acc
            alias_to_accession[identity.formal_name.lower()] = acc
        for alias in identity.aliases:
            alias_to_accession[alias] = acc
            alias_to_accession[alias.lower()] = acc

    LOGGER.info("[Skeleton] PMID %s: built skeleton with %d accessions, %d aliases",
                paper_ctx.pmid, len(identity_map), len(alias_to_accession))

    return identity_map, alias_to_accession


def _compress_to_biosample_level(
    verified: Set[str],
    upstream: "UpstreamData",
) -> Set[str]:
    """Compress verified accessions to BioSample + BioProject level.

    Run-level accessions (SRR/ERR/DRR/SRX/ERX/DRX/SRS/ERS/DRS) are replaced
    by their linked BioSample ID. BioProject and BioSample accessions pass through.
    """
    compressed: Set[str] = set()
    for acc in verified:
        if acc.startswith("PRJ") or acc.startswith("SAM"):
            compressed.add(acc)
        else:
            # Run-level: replace with BioSample
            vm = upstream.verified_mapping.get(acc, {})
            biosample = vm.get("biosample", "")
            if biosample:
                compressed.add(biosample)
                # Also keep the BioProject
                bioproject = vm.get("bioproject", "")
                if bioproject:
                    compressed.add(bioproject)
            else:
                # No linked BioSample — keep original
                compressed.add(acc)
    return compressed


# ═══════════════════════════════════════════════════════════
#  加载函数
# ═══════════════════════════════════════════════════════════


def load_upstream(
    relation_file: str = "",
    accession_file: str = "",
    accession_list_file: str = "",
    expanded_metadata_file: str = "",
    env_tag_file: str = "",
    env_extraction_targets_file: str = "",
) -> UpstreamData:
    """一次性加载所有上游产物。各文件缺失时对应索引为空，不报错。"""
    ud = UpstreamData()

    # ── step2 relation ────────────────────────────────────
    if relation_file and Path(relation_file).exists():
        with open(relation_file, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            pmid = str(item.get("pmid", ""))
            rel = item.get("relation", "unknown")
            key = (item.get("section_type", ""), int(item.get("index", 0)))
            ud.relation_index.setdefault(pmid, {})[key] = rel
            accs = item.get("accessions_found", [])
            if accs:
                ud.step2_accessions_index.setdefault(pmid, {})[key] = accs
                ud.accession_by_section.setdefault(pmid, {})[key] = accs
            labels = item.get("labels_found", [])
            if labels:
                ud.step2_labels_index.setdefault(pmid, {})[key] = labels
            meta_keys = item.get("metadata_keys_found", [])
            if meta_keys:
                ud.step2_metadata_keys_index.setdefault(pmid, {})[key] = meta_keys
        LOGGER.info("Loaded step2 relation: %d records", len(items))

    # ── step3 accession (legacy path; rarely used in 0408) ──
    if accession_file and Path(accession_file).exists():
        with open(accession_file, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            accs = item.get("accessions", [])
            if not accs:
                continue
            pmid = str(item.get("pmid", ""))
            key = (item.get("section_type", ""), int(item.get("index", 0)))
            ud.accession_by_section.setdefault(pmid, {})[key] = accs
        LOGGER.info("Loaded step3 accession: %d records with accessions", sum(
            len(v) for v in ud.accession_by_section.values()
        ))

    # ── 外部 DB 验证 accession list ───────────────────────
    # Used as a horizontal lookup (acc → biosample/bioproject) for table-column
    # alias bridging. The last-column pmid is intentionally NOT split on ";":
    # in 0408 design, "this paper's accessions" comes from step2 detection in
    # paper text, not from reverse pmid lookup against this file. The legacy
    # verified_acc_by_pmid index is kept for backward compatibility but no
    # longer consulted by build_paper_context.
    if accession_list_file and Path(accession_list_file).exists():
        with open(accession_list_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 8:
                    continue
                run_acc, bioproject, biosample = parts[0], parts[1], parts[2]
                pmid = parts[-1]
                entry = {"biosample": biosample, "bioproject": bioproject, "pmid": pmid}
                ud.verified_mapping[run_acc] = entry
                if biosample not in ud.verified_mapping:
                    ud.verified_mapping[biosample] = entry
                if bioproject not in ud.verified_mapping:
                    ud.verified_mapping[bioproject] = entry
                ud.verified_acc_by_pmid.setdefault(pmid, set()).update(
                    [run_acc, biosample, bioproject]
                )
                for col in parts[3:-1]:
                    col = col.strip()
                    if col:
                        ud.verified_mapping.setdefault(col, entry)
                        ud.verified_acc_by_pmid[pmid].add(col)
        LOGGER.info("Loaded verified accessions: %d unique, %d PMIDs",
                     len(ud.verified_mapping), len(ud.verified_acc_by_pmid))

    # ── pmid_run_merged_data_expanded ─────────────────────
    if expanded_metadata_file and Path(expanded_metadata_file).exists():
        with open(expanded_metadata_file, "r", encoding="utf-8") as f:
            items = json.load(f)
        pmid_bs_index: Dict[str, Set[str]] = defaultdict(set)
        for item in items:
            bs_id = item.get("biosample_id", "")
            if not bs_id:
                continue
            ud.biosample_metadata[bs_id] = item
            bp_id = item.get("bioproject_id", "")
            for acc_id in (bs_id, bp_id):
                vm = ud.verified_mapping.get(acc_id, {})
                pmid = vm.get("pmid", "")
                if pmid:
                    pmid_bs_index[pmid].add(bs_id)
        ud.biosamples_by_pmid = {p: list(bs) for p, bs in pmid_bs_index.items()}
        LOGGER.info("Loaded expanded metadata: %d biosamples, %d PMIDs",
                     len(ud.biosample_metadata), len(ud.biosamples_by_pmid))

    # ── step4 env_tag ─────────────────────────────────────
    if env_tag_file and Path(env_tag_file).exists():
        with open(env_tag_file, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            bs_id = item.get("biosample_id", "")
            env_val = item.get("env_tag", {}).get("value", "")
            if bs_id and env_val:
                ud.env_by_biosample[bs_id] = env_val
        LOGGER.info("Loaded env_tag: %d biosamples", len(ud.env_by_biosample))

    # ── env_extraction_targets ────────────────────────────
    if env_extraction_targets_file and Path(env_extraction_targets_file).exists():
        with open(env_extraction_targets_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for env_name, env_data in data.get("per_environment", {}).items():
            env_name = ud._normalize_env_key(env_name)
            ud.env_target_fields[env_name] = env_data.get("fields", [])
        ud.global_fields = data.get("global_fields", {})
        LOGGER.info("Loaded env extraction targets: %d envs", len(ud.env_target_fields))

    return ud
