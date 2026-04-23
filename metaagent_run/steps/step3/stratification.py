"""扁平化预处理：把 step2 的 section 输出展开为 (key, pmid, env) 条目。

与原 stratification.py 的职责对比：
- 原：按 (sub_env × section_type × year_bin) 三维建分层索引，供 sampling.py 采样
- 新：做字段归一化 + 扁平化展开，产出三样东西：
    1. unique_keys     : 字典序排序的唯一 key 字符串列表（主循环对象）
    2. occurrences     : list[(normalized_key, pmid, sub_env_bin)]（后处理归因用）
    3. key_to_evidence : {normalized_key: evidence_str}
                         每 key 一条，按 section_type 优先级（METHODS>RESULTS>SUPPL>
                         TABLE>unknown）首见锁定。Step 2 的 evidence_quote 当源头，
                         不存在时退化到 text 前 N 字符，再不行用 "(step2)" 占位。

原来的三维分层逻辑完全退役；但各维度的归一化函数保留并导出，因为
论文语料描述图（env × section × year）仍然会用到。
"""
from collections import Counter

from .runtime import LOGGER

ItemRecord = dict[str, object]
KeyOccurrence = tuple[str, str, str]  # (normalized_key, pmid, sub_env_bin)

# Evidence 源头 section_type 优先级（ABSTRACT 在 Step 2 的 build_discovery_input
# 就被跳过了，不会进 step3 输入，所以此处不列 ABSTRACT）
_SECTION_PRIORITY_FOR_EVIDENCE = {
    "METHODS": 4,
    "RESULTS": 3,
    "SUPPL": 2,
    "TABLE": 1,
    "unknown": 0,
}
_SECTION_PRIORITY_FOR_EVIDENCE_REVERSE = {v: k for k, v in _SECTION_PRIORITY_FOR_EVIDENCE.items()}
_EVIDENCE_TEXT_MAX_CHARS = 500

VALID_SECTION_TYPES = {
    "ABSTRACT", "INTRO", "METHODS", "RESULTS",
    "DISCUSS", "CONCL", "SUPPL", "TABLE",
}

SECTION_ALIASES = {
    "abstract": "ABSTRACT",
    "intro": "INTRO", "introduction": "INTRO", "background": "INTRO",
    "method": "METHODS", "methods": "METHODS",
    "materials and methods": "METHODS", "materials & methods": "METHODS",
    "experimental": "METHODS",
    "result": "RESULTS", "results": "RESULTS",
    "finding": "RESULTS", "findings": "RESULTS",
    "discussion": "DISCUSS", "discuss": "DISCUSS",
    "conclusion": "CONCL", "conclusions": "CONCL", "concl": "CONCL",
    "supplement": "SUPPL", "supplementary": "SUPPL",
    "supporting": "SUPPL", "appendix": "SUPPL",
    "table": "TABLE", "tables": "TABLE",
}
VALID_SUB_ENVS = {"Open_ocean", "Coastal_waters", "Lake", "Wetlands"}
_SUB_ENV_LOOKUP = {env.lower(): env for env in VALID_SUB_ENVS}


def normalize_section_type(item: ItemRecord) -> str:
    sec = str(item.get("section_type", "")).strip()
    up = sec.upper()
    if up in VALID_SECTION_TYPES:
        return up
    return SECTION_ALIASES.get(sec.lower(), "unknown")


def normalize_year_bin(item: ItemRecord) -> str:
    year_value = item.get("pub_year", None)
    if year_value is None:
        return "unknown"
    try:
        year = int(str(year_value))
    except (TypeError, ValueError):
        return "unknown"
    if 2000 <= year < 2005: return "2000-2005"
    if 2005 <= year < 2010: return "2005-2010"
    if 2010 <= year < 2015: return "2010-2015"
    if 2015 <= year < 2020: return "2015-2020"
    if 2020 <= year <= 2026: return "2020-2026"
    return "unknown"


def normalize_sub_env(item: ItemRecord) -> str:
    raw_value = item.get("sub_env", [])
    if not isinstance(raw_value, list):
        return "unknown_env"
    valid = []
    for value in raw_value:
        canonical = _SUB_ENV_LOOKUP.get(str(value).strip().lower())
        if canonical is not None:
            valid.append(canonical)
    deduped = list(dict.fromkeys(valid))
    if len(deduped) == 1:
        return deduped[0]
    if len(deduped) > 1:
        return "multi_env"
    return "unknown_env"


def _normalize_key_string(raw: str) -> str:
    """复用原 processor 中的 key 归一化规则。"""
    return raw.strip().lower().replace(" ", "_").replace("-", "_")


def _pick_evidence_from_item(item: ItemRecord) -> str:
    """从 section 条目里抽一条 evidence 字符串。
    优先用 Step 2 的 evidence_quote（≤200 字符，已精炼），
    其次退化到 text 前 500 字符，都没有则 "(step2)" 占位。"""
    evq = str(item.get("evidence_quote") or "").strip()
    if evq:
        return evq
    text = str(item.get("text", "")).strip()
    if text:
        return text[:_EVIDENCE_TEXT_MAX_CHARS]
    return "(step2)"


def build_flattened_occurrences(
    items: list[ItemRecord],
) -> tuple[list[str], list[KeyOccurrence], dict[str, str]]:
    """扫描 section 级输入，产出 (unique_keys, occurrences, key_to_evidence)。

    过滤规则：
      - 丢弃 metadata_keys_found 为空的 section
      - 丢弃 sub_env_bin ∉ VALID_SUB_ENVS 的 section（multi_env / unknown_env）
      - 丢弃 pmid 为空的 section（无法归因）

    同时在每个 item 上就地写入归一化后的 section_type / year_bin / sub_env_bin，
    供论文语料描述图使用。

    key_to_evidence 构建策略：
      - 对每个 normalized_key，在所有出现它的 section 里按 section_type 优先级
        （METHODS>RESULTS>SUPPL>TABLE>unknown）挑最高者；同优先级下取首见。
      - 当前选中 section 的 evidence 来源按 _pick_evidence_from_item 决定。
      - 即便下游 USE_EVIDENCE_IN_DEDUP / USE_EVIDENCE_IN_FILTER 关闭，此 dict 仍然
        构建（orchestrator 会按 flag 决定是否使用），避免开关切换时要重跑前处理。

    返回：
      unique_keys     : 按字典序排序的唯一 key 字符串列表
      occurrences     : list[(normalized_key, pmid, sub_env_bin)]，可能含同 key 重复
      key_to_evidence : dict[normalized_key, evidence_str]
    """
    occurrences: list[KeyOccurrence] = []
    unique_set: set[str] = set()
    key_to_evidence: dict[str, str] = {}
    key_evidence_priority: dict[str, int] = {}  # 用于实现优先级覆盖

    skipped_no_metadata = 0
    skipped_no_pmid = 0
    skipped_multi_env = 0
    skipped_unknown_env = 0
    skipped_empty_key = 0

    for item in items:
        # 原地归一化（供外部画图）
        item["section_type"] = normalize_section_type(item)
        item["year_bin"] = normalize_year_bin(item)
        item["sub_env_bin"] = normalize_sub_env(item)

        raw_keys = item.get("metadata_keys_found")
        if not isinstance(raw_keys, list) or not raw_keys:
            skipped_no_metadata += 1
            continue

        pmid = str(item.get("pmid", "")).strip()
        if not pmid:
            skipped_no_pmid += 1
            continue

        sub_env_bin = str(item["sub_env_bin"])
        if sub_env_bin not in VALID_SUB_ENVS:
            if sub_env_bin == "multi_env":
                skipped_multi_env += 1
            else:
                skipped_unknown_env += 1
            continue

        section_key = str(item.get("section_type", "unknown"))
        current_priority = _SECTION_PRIORITY_FOR_EVIDENCE.get(section_key, 0)
        evidence_for_item = _pick_evidence_from_item(item)

        for raw in raw_keys:
            key = _normalize_key_string(str(raw))
            if not key:
                skipped_empty_key += 1
                continue
            occurrences.append((key, pmid, sub_env_bin))
            unique_set.add(key)
            # 首见或遇到更高优先级 section 时，更新 evidence
            if current_priority > key_evidence_priority.get(key, -1):
                key_to_evidence[key] = evidence_for_item
                key_evidence_priority[key] = current_priority

    unique_keys = sorted(unique_set)

    LOGGER.info("=" * 60)
    LOGGER.info("[扁平化] section 总数: %d", len(items))
    LOGGER.info("  跳过无 metadata_keys:  %d", skipped_no_metadata)
    LOGGER.info("  跳过无 pmid:          %d", skipped_no_pmid)
    LOGGER.info("  跳过 multi_env:       %d", skipped_multi_env)
    LOGGER.info("  跳过 unknown_env:     %d", skipped_unknown_env)
    LOGGER.info("  跳过空 key 实例:      %d", skipped_empty_key)
    LOGGER.info("[扁平化] 累计 (key,pmid,env) 条目: %d", len(occurrences))
    LOGGER.info("[扁平化] 唯一 key 字符串: %d", len(unique_keys))
    # evidence 源 section 优先级分布（诊断用）
    ev_prio_counter = Counter(_SECTION_PRIORITY_FOR_EVIDENCE_REVERSE.get(p, "unknown")
                              for p in key_evidence_priority.values())
    LOGGER.info("[扁平化] key_to_evidence 源 section 分布: %s", dict(ev_prio_counter))

    # 语料剖面统计（非算法使用，用于论文描述）
    sec_counter = Counter(str(item.get("section_type", "unknown")) for item in items)
    year_counter = Counter(str(item.get("year_bin", "unknown")) for item in items)
    env_counter = Counter(str(item.get("sub_env_bin", "unknown_env")) for item in items)
    LOGGER.info("[语料剖面] section_type: %s", dict(sec_counter))
    LOGGER.info("[语料剖面] year_bin:     %s", dict(year_counter))
    LOGGER.info("[语料剖面] sub_env_bin:  %s", dict(env_counter))
    LOGGER.info("=" * 60)

    return unique_keys, occurrences, key_to_evidence
