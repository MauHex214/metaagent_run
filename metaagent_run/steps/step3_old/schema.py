import asyncio
import json
import re
from collections import defaultdict
from typing import Any, Optional, TypedDict, cast

from .runtime import LOGGER, STOP_SENTINEL

MAX_ITEM_TEXT_CHARS = 5000
# Sentinel: _llm_match determined "no match" (distinct from failure/None)
LLM_NO_MATCH = "__LLM_NO_MATCH__"

MAX_CONTEXTS_PER_SCHEMA = 3
MAX_EVIDENCE_CHARS = 500

# Dedup prompt 版本（对应 prompts/dedup_v1.txt，由 prompt_builder.load_prompt_template 按需加载）
DEDUP_PROMPT_VERSION = "dedup_v1"

SCHEMA_CATEGORIES = {
    "Spatiotemporal": [
        "latitude", "longitude", "lat", "lon", "lng", "coord", "gps",
        "location", "site", "station", "region", "country", "city",
        "date", "time", "year", "month", "day", "season", "temporal",
        "collection_date", "sampling_date", "sampling_time",
    ],
    "Physical": [
        "temperature", "temp", "pressure", "density", "conductiv", "ec",
        "turbidity", "secchi", "transparency", "light", "par",
        "current", "wave", "tide", "flow", "velocity",
        "sst", "sss", "eh", "orp",
    ],
    "Chemical": [
        "salinity", "sal", "ph", "oxygen", "do", "dissolved", "nutrient",
        "nitrogen", "phosph", "carbon", "sulfur", "silica", "silicate",
        "nitrate", "no3", "nitrite", "no2", "ammoni", "nh4", "nh3",
        "phosphate", "po4", "sulfate", "so4", "h2s", "co2",
        "doc", "don", "dop", "dic", "poc", "pon", "pop",
        "toc", "ton", "tdn", "tn", "tp", "tdp",
        "din", "dip", "dom", "pom", "srp",
        "bod", "cod", "tss", "tds",
        "chlorophyll", "chl", "chla", "chl_a",
        "alkalinity", "hardness", "redox",
    ],
    "Biological": [
        "biome", "env_biome", "env_feature", "env_material",
        "habitat", "ecosystem", "vegetation", "species",
        "organism", "host", "isolation_source", "biomass",
        "abundance", "diversity", "community",
    ],
    "Sampling": [
        "depth", "sample", "filter", "fraction", "volume",
        "replicate", "size", "mesh", "pore", "method",
        "collector", "vessel", "bottle", "net", "trap",
        "transect", "quadrat", "core",
    ],
}

VALUE_PATTERNS = [
    re.compile(r"^\d+[\.,]?\d*$"),
    re.compile(r"^\d{1,2}[_\-/]\w+[_\-/]\d{2,4}$"),
    re.compile(r"^\d{4}[_\-]\d{2}[_\-]\d{2}$"),
    re.compile(r"^-?\d+[\.,]?\d*\s*°\s*[cfnneessw]", re.I),
    re.compile(r"^-?\d+[\.,]?\d*°"),
    re.compile(r"^[><≤≥±]\s*\d"),
    re.compile(r"^\d+[\.,]?\d*\s*[µuμmMgGkKlL][gGmMlLmM]?$"),
]

INSTRUMENT_BLACKLIST = {
    "zeiss", "sigma", "invitrogen", "thermo", "agilent", "illumina",
    "niskin", "qubit", "nanodrop", "bioanalyzer", "qiagen", "promega",
    "bd_biosciences", "beckman", "waters", "shimadzu",
}

# 实验室条件/生理特征前缀：在 filter 阶段整键排除（它们不是原位样本元数据，
# 而是实验操作或生物体固有属性，Step 2 的 VALUE EXCLUSION 规则理论上应该
# 挡住但 LLM 漏判时此处兜底）
EXPERIMENTAL_PREFIX_BLACKLIST = (
    "growth_",
    "incubation_",
    "optimum_",
    "storage_",
    "treatment_",
    "cultivation_",
    "induction_",
)

SCIENTIFIC_WHITELIST = {
    "ph", "doc", "don", "dop", "dic", "poc", "pon", "pop",
    "nh4", "no3", "no2", "po4", "so4", "h2s", "co2", "o2",
    "toc", "ton", "tdn", "tn", "tp", "tdp",
    "chl_a", "chla", "par", "ntu",
    "bod", "cod", "tss", "tds", "do", "ec", "sal",
    "sst", "sss", "orp", "eh", "dom", "pom", "din", "dip", "srp",
}

# 单复数等价映射：归一化时把特定复数形式缩回单数，
# 避免 `sample_number` 与 `sample_numbers` 被拆成两个 canonical
PLURAL_MAPPINGS = {
    "numbers": "number",
    "counts": "count",
    "samples": "sample",
    "measurements": "measurement",
    "concentrations": "concentration",
    "temperatures": "temperature",
    "depths": "depth",
    "salinities": "salinity",
    "locations": "location",
    "sites": "site",
    "stations": "station",
    "replicates": "replicate",
    "fractions": "fraction",
    "sources": "source",
    "volumes": "volume",
    "dates": "date",
}


# ═══════════════════════════════════════════════════════════════
# LLM 语义去重提示词
# ═══════════════════════════════════════════════════════════════
#
# DEDUP system prompt 现以 prompts/{DEDUP_PROMPT_VERSION}.txt 外置维护。
# 通过 prompt_builder.load_prompt_template() 按需加载并 lru_cache。
# 历史 DEDUP_USER_TEMPLATE 常量未被代码引用（user_msg 在 _llm_match 内直接
# 用 f-string 拼装），随硬编码 system prompt 一并移除。


# ═══════════════════════════════════════════════════════════════
# LLM 语义去重器
# ═══════════════════════════════════════════════════════════════


class SemanticDeduplicator:
    """基于 LLM 的语义去重器。

    新 key 出现时的处理流程（分层漏斗）：
      1. 精确匹配：key 已见过 → 直接返回其 canonical（零开销）
      2. Token 预过滤：提取 key 的 token，在倒排索引中查找共享
         token 的已有 key，通常将数百个 key 缩减为 5-20 个候选
      3. 短 key 领域扩展：对去掉下划线后 ≤5 字符的缩写型 key
         （如 ph、doc、sst），按 SCHEMA_CATEGORIES 领域类别扩展候选
      4. LLM 精判：将新 key + 候选列表（含别名）发给 LLM，由 LLM
         判断是否存在语义等价关系
      5. 保守策略：LLM 返回 null 或调用失败 → 一律视为新字段，
         宁可多保留也不错误合并
    """

    def __init__(self):
        self._groups: dict[int, set[str]] = {}
        self._group_canonical: dict[int, str] = {}
        self._key_to_group: dict[str, int] = {}
        self._next_group_id: int = 0
        # Token 倒排索引：token → {key1, key2, ...}
        self._token_index: dict[str, set[str]] = defaultdict(set)
        # LLM 客户端（外部注入，不参与序列化）
        self._llm_client: Any = None
        self._request_interval: float = 0.3

    # ── 外部接口 ─────────────────────────────────────────────

    def set_llm_client(self, llm_client: Any, request_interval: float = 0.3) -> None:
        """注入 LLM 客户端。未注入时 resolve 退化为"全部视为新字段"。"""
        self._llm_client = llm_client
        self._request_interval = request_interval

    @property
    def canonical_keys(self) -> set[str]:
        return set(self._group_canonical.values())

    @property
    def canonical_size(self) -> int:
        return len(self._groups)

    def get_canonical(self, key: str) -> Optional[str]:
        gid = self._key_to_group.get(key)
        if gid is None:
            return None
        return self._group_canonical.get(gid)

    # ── Canonical 选择策略 ────────────────────────────────────

    @staticmethod
    def _select_canonical(members: set[str]) -> str:
        """选择组内最具描述性的 key 作为 canonical（token 最多、最长）。"""
        def sort_key(key: str) -> tuple[int, int, str]:
            token_count = len([t for t in key.split("_") if t])
            return (-token_count, -len(key), key)
        return min(members, key=sort_key)

    # ── Token 索引 ───────────────────────────────────────────

    @staticmethod
    def _tokenize(key: str) -> list[str]:
        """拆分 key 为有意义的 token（≥2 字符）。"""
        return [t for t in key.lower().split("_") if t and len(t) >= 2]

    def _update_token_index(self, key: str) -> None:
        for token in self._tokenize(key):
            self._token_index[token].add(key)

    def _rebuild_token_index(self) -> None:
        self._token_index = defaultdict(set)
        for key in self._key_to_group:
            self._update_token_index(key)

    # ── 候选预过滤 ───────────────────────────────────────────

    @staticmethod
    def _categorize_key(key: str) -> Optional[str]:
        key_lower = key.lower()
        best_cat: Optional[str] = None
        best_score = 0
        for cat, patterns in SCHEMA_CATEGORIES.items():
            score = sum(1 for p in patterns if p in key_lower)
            if score > best_score:
                best_score = score
                best_cat = cat
        return best_cat if best_score > 0 else None

    def _find_candidates(self, key: str) -> dict[str, list[str]]:
        """Token 预过滤 + 短 key 领域扩展，返回 {canonical: sorted_members}。

        典型情况下将数百个已有 key 缩减至 5-20 个候选 canonical，
        使 LLM 精判的 prompt 长度可控且准确率高。
        """
        tokens = self._tokenize(key)
        candidate_gids: set[int] = set()

        # ---- Token 重叠 ----
        for token in tokens:
            for existing_key in self._token_index.get(token, set()):
                gid = self._key_to_group.get(existing_key)
                if gid is not None:
                    candidate_gids.add(gid)

        # ---- 短 key / 缩写型扩展 ----
        # 例如 "ph", "doc", "sst" 拆 token 后可能无法命中足够候选。
        # 策略 A：按 SCHEMA_CATEGORIES 领域类别拉取同类 key
        # 策略 B：对只有 1 个 token 的 key，检查它是否是已有 key 的
        #         首字母缩写或子串（如 sst → sea_surface_temperature）
        stripped = key.replace("_", "")
        if len(stripped) <= 5:
            key_cat = self._categorize_key(key)
            if key_cat:
                for existing_key, gid in self._key_to_group.items():
                    if self._categorize_key(existing_key) == key_cat:
                        candidate_gids.add(gid)

        # 缩写检测：如果 key 只有 1 个 token 且 ≤5 字符，
        # 检查已有 key 的首字母缩写是否匹配
        if len(tokens) == 1 and len(stripped) <= 5:
            abbr = stripped.lower()
            for existing_key, gid in self._key_to_group.items():
                existing_tokens = [t for t in existing_key.lower().split("_") if t]
                if len(existing_tokens) >= 2:
                    initials = "".join(t[0] for t in existing_tokens)
                    if initials == abbr:
                        candidate_gids.add(gid)

        # 兜底：如果 token 预过滤 + 类别 + 缩写都未找到候选，
        # 且 key 是科学白名单中的缩写（如 sst, par, ntu），
        # 则拉取 Physical + Chemical 类别的所有 key 让 LLM 判断
        if not candidate_gids and stripped.lower() in SCIENTIFIC_WHITELIST:
            for existing_key, gid in self._key_to_group.items():
                cat = self._categorize_key(existing_key)
                if cat in ("Physical", "Chemical"):
                    candidate_gids.add(gid)

        # ---- 组装结果，上限 30 个 canonical ----
        raw_result: dict[str, list[str]] = {}
        for gid in candidate_gids:
            canon = self._group_canonical[gid]
            raw_result[canon] = sorted(self._groups[gid])

        if len(raw_result) <= 30:
            return raw_result

        # 超过 30 个候选时，按 token 重叠度排序取 top-30
        tokens_set = set(tokens)

        def _overlap_score(item: tuple[str, list[str]]) -> tuple[int, str]:
            canon = item[0]
            overlap = len(tokens_set & set(self._tokenize(canon)))
            return (-overlap, canon)

        sorted_items = sorted(raw_result.items(), key=_overlap_score)
        return dict(sorted_items[:30])

    # ── LLM 语义判断 ─────────────────────────────────────────

    async def _llm_match(
        self,
        new_key: str,
        evidence: str,
        candidates: dict[str, list[str]],
    ) -> Optional[str]:
        """调用 LLM 判断 new_key 是否与候选中的某个 canonical 语义等价。

        返回匹配的 canonical key，或 None（无匹配 / 调用失败）。
        """
        if not self._llm_client or not candidates:
            return None

        # 构建候选列表文本
        lines: list[str] = []
        for canon, members in candidates.items():
            if len(members) > 1:
                aliases = [m for m in members if m != canon][:5]
                lines.append(f"- {canon} (aliases: {', '.join(aliases)})")
            else:
                lines.append(f"- {canon}")
        candidates_str = "\n".join(lines)

        # evidence 可能为空（取决于 USE_EVIDENCE_IN_DEDUP flag）→ 空时直接
        # 省略 Evidence 行，避免在 prompt 里残留空白行
        if evidence:
            user_msg = (
                f'New field: "{new_key}"\n'
                f'Evidence: "{evidence}"\n'
                f'Existing candidate fields:\n{candidates_str}'
            )
        else:
            user_msg = (
                f'New field: "{new_key}"\n'
                f'Existing candidate fields:\n{candidates_str}'
            )

        # 延迟 import 以避免循环依赖风险；lru_cache 由 load_prompt_template 内置
        from .prompt_builder import load_prompt_template
        messages = [
            {"role": "system", "content": load_prompt_template(DEDUP_PROMPT_VERSION)},
            {"role": "user", "content": user_msg},
        ]

        try:
            response = await self._llm_client.chat(
                messages=messages, max_retries=2, base_backoff=1.0,
            )
            if response is None:
                return None

            # 解析响应
            text = response.strip()
            if STOP_SENTINEL in text:
                text = text.split(STOP_SENTINEL, 1)[0].strip()
            text = re.sub(r"^\s*```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text).strip()

            result = json.loads(text)
            match = result.get("match")
            reason = result.get("reason", "")

            if match and match in candidates:
                LOGGER.info(
                    "[LLM去重] '%s' → '%s' (%s)", new_key, match, reason,
                )
                return match

            if match:
                # LLM 返回了不在候选列表中的 key — 忽略
                LOGGER.warning(
                    "[LLM去重] '%s' → '%s' 不在候选列表中，忽略", new_key, match,
                )
            # LLM 成功判定无匹配 — 返回哨兵值，区别于调用失败的 None
            return LLM_NO_MATCH

        except Exception as exc:
            LOGGER.warning("[LLM去重] '%s' 判断失败: %s", new_key, exc)
            return None

    # ── 核心 resolve ─────────────────────────────────────────

    async def resolve(self, key: str, evidence: str = "") -> tuple[Optional[str], bool]:
        """解析一个 key，返回 (canonical, is_new)。

        is_new=True 表示该 key 是全新字段（未与任何已有字段合并）。
        """
        # 1. 精确匹配
        if key in self._key_to_group:
            gid = self._key_to_group[key]
            return self._group_canonical[gid], False

        # 2. 预过滤候选
        candidates = self._find_candidates(key)

        # 3. LLM 精判
        matched_canonical: Optional[str] = None
        if candidates:
            matched_canonical = await self._llm_match(key, evidence, candidates)
            if self._llm_client:
                await asyncio.sleep(self._request_interval)

        # 4. 合并或新建
        if matched_canonical is not None:
            target_gid: Optional[int] = None
            for gid, canon in self._group_canonical.items():
                if canon == matched_canonical:
                    target_gid = gid
                    break
            if target_gid is not None:
                old_canonical = self._group_canonical[target_gid]
                self._groups[target_gid].add(key)
                self._key_to_group[key] = target_gid
                self._update_token_index(key)
                new_canonical = self._select_canonical(self._groups[target_gid])
                self._group_canonical[target_gid] = new_canonical
                if new_canonical != old_canonical:
                    LOGGER.info(
                        "[语义去重] canonical 更新: '%s' -> '%s' (因 '%s' 加入)",
                        old_canonical, new_canonical, key,
                    )
                return self._group_canonical[target_gid], False

        # 4b. LLM 调用失败（网络/超时） → 跳过，避免分裂
        if candidates and matched_canonical is None:
            LOGGER.debug(
                "[语义去重] LLM 调用失败，无法判定 '%s' 与候选 %s 的关系，跳过本次",
                key, candidates,
            )
            return None, False

        # 5. 无候选，或 LLM 成功判定不匹配任何候选 → 全新字段
        gid = self._next_group_id
        self._next_group_id += 1
        self._groups[gid] = {key}
        self._group_canonical[gid] = key
        self._key_to_group[key] = gid
        self._update_token_index(key)
        return key, True

    async def resolve_batch(
        self, hits: list["ExplicitSchemaHit"],
    ) -> tuple[set[str], set[str]]:
        """解析一批 hit，逐条调用 resolve 以保证顺序一致性。

        返回 (all_canonical_set, truly_new_canonical_set)。
        """
        all_canonical: set[str] = set()
        truly_new: set[str] = set()
        for hit in sorted(hits, key=lambda h: h["normalized_key"]):
            key = hit["normalized_key"]
            evidence = hit.get("evidence", "")
            canonical, is_new = await self.resolve(key, evidence)
            if canonical is None:
                continue  # LLM 无法判定，跳过
            all_canonical.add(canonical)
            if is_new:
                truly_new.add(canonical)
        return all_canonical, truly_new

    # ── 报告与序列化 ─────────────────────────────────────────

    def get_alias_report(self) -> dict[str, list[str]]:
        report: dict[str, list[str]] = {}
        for gid, members in self._groups.items():
            if len(members) > 1:
                report[self._group_canonical[gid]] = sorted(members)
        return report

    def get_all_groups(self) -> dict[str, list[str]]:
        return {
            self._group_canonical[gid]: sorted(members)
            for gid, members in self._groups.items()
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "groups": {
                self._group_canonical[gid]: sorted(members)
                for gid, members in self._groups.items()
            },
            "next_group_id": self._next_group_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SemanticDeduplicator":
        dedup = cls()
        groups_raw = data.get("groups", data.get("canonical_map", {}))
        groups_data = (
            cast(dict[str, object], groups_raw)
            if isinstance(groups_raw, dict)
            else {}
        )
        for canonical_hint, members in groups_data.items():
            member_set = (
                set(members) if isinstance(members, list) else {canonical_hint}
            )
            gid = dedup._next_group_id
            dedup._next_group_id += 1
            dedup._groups[gid] = member_set
            dedup._group_canonical[gid] = cls._select_canonical(member_set)
            for member in member_set:
                dedup._key_to_group[member] = gid
        next_group_id = data.get("next_group_id")
        if isinstance(next_group_id, int):
            dedup._next_group_id = max(dedup._next_group_id, int(next_group_id))
        dedup._rebuild_token_index()
        return dedup


# ═══════════════════════════════════════════════════════════════
# 以下为原有工具函数（未修改）
# ═══════════════════════════════════════════════════════════════


def categorize_schema_keys(keys: set[str]) -> dict[str, list[str]]:
    categorized: dict[str, list[str]] = {cat: [] for cat in SCHEMA_CATEGORIES}
    categorized["Other"] = []
    for key in sorted(keys):
        key_lower = key.lower()
        best_cat = None
        best_score = 0
        for cat, patterns in SCHEMA_CATEGORIES.items():
            score = sum(1 for pattern in patterns if pattern in key_lower)
            if score > best_score:
                best_score = score
                best_cat = cat
        if best_cat and best_score > 0:
            categorized[best_cat].append(key)
        else:
            categorized["Other"].append(key)
    return {cat: values for cat, values in categorized.items() if values}


def format_categorized_schema_for_prompt(
    current_schema: set[str],
    max_per_category: int = 80,
    max_total: int = 400,
) -> tuple[str, str]:
    if not current_schema:
        return "[]", ""
    categorized = categorize_schema_keys(current_schema)
    total_schema_size = len(current_schema)
    lines: list[str] = []
    total_shown = 0
    for cat, keys in categorized.items():
        if total_shown >= max_total:
            break
        show_n = min(len(keys), max_per_category, max_total - total_shown)
        total_shown += show_n
        truncation_note = (
            f" (showing {show_n}/{len(keys)})" if show_n < len(keys) else ""
        )
        lines.append(
            f"  [{cat}]{truncation_note}: "
            f"{json.dumps(keys[:show_n], ensure_ascii=False)}"
        )
    return "\n".join(lines), (
        f"(total {total_schema_size} known attributes "
        f"across {len(categorized)} categories)"
    )


def truncate_text_for_llm(text: str, max_chars: int = MAX_ITEM_TEXT_CHARS) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head_keep = int(max_chars * 0.7)
    tail_keep = max(
        max_chars - head_keep - len("\n\n...[TRUNCATED]...\n\n"), 0
    )
    truncated = (
        text[:head_keep]
        + "\n\n...[TRUNCATED]...\n\n"
        + (text[-tail_keep:] if tail_keep > 0 else "")
    )
    LOGGER.info(
        "输入文本过长，已截断: original_chars=%d truncated_chars=%d",
        len(text),
        len(truncated),
    )
    return truncated


def clean_llm_json_response(response: str) -> str:
    text = response.strip()
    if STOP_SENTINEL in text:
        text = text.split(STOP_SENTINEL, 1)[0].strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return text
    except Exception:
        pass
    start = text.find("[")
    if start == -1:
        return text
    in_string = False
    escape = False
    depth = 0
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1].strip()
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, list):
                        return candidate
                except Exception:
                    break
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        candidate = match.group(0).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return candidate
        except Exception:
            pass
    return text


def normalize_key(raw_key: str) -> Optional[str]:
    key = raw_key.strip().lower()
    key = re.sub(r"[^\w\s/\-,]", "", key)
    key = re.sub(r"[/,\s]+", "_", key)
    key = re.sub(r"[^a-z0-9_\-]", "", key)
    key = re.sub(r"_+", "_", key).strip("_-")
    if len(key) < 2:
        return None
    # 单复数归一：只替换确定安全的复数形式（白名单），避免把
    # coordinates/species 这种天然以 s 结尾的词误裁
    tokens = key.split("_")
    tokens = [PLURAL_MAPPINGS.get(t, t) for t in tokens]
    return "_".join(tokens)


def is_experimental_condition_key(normalized: str) -> bool:
    """返回 True 表示 key 以实验/生理条件前缀开头（应被 filter 排除）。"""
    return any(normalized.startswith(prefix) for prefix in EXPERIMENTAL_PREFIX_BLACKLIST)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def clip_evidence(text: str, max_chars: int = MAX_EVIDENCE_CHARS) -> str:
    text = normalize_whitespace(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def is_blacklisted_instrument_key(normalized: str) -> bool:
    if normalized in INSTRUMENT_BLACKLIST:
        return True
    tokens = [token for token in normalized.split("_") if token]
    if any(token in INSTRUMENT_BLACKLIST for token in tokens):
        return True
    for brand in INSTRUMENT_BLACKLIST:
        if normalized.startswith(brand):
            suffix = normalized[len(brand) :]
            if (
                suffix == ""
                or suffix.startswith("_")
                or re.match(r"^[0-9a-z_\-]+$", suffix)
            ):
                return True
    return False


class ExplicitSchemaHit(TypedDict):
    raw_key: str
    normalized_key: str
    evidence: str


def filter_discovered_entries(
    raw_entries: list[object], review_pool: list[dict[str, object]]
) -> list[ExplicitSchemaHit]:
    review_pool_keys = {
        (record.get("normalized_key"), record.get("evidence", ""))
        for record in review_pool
        if isinstance(record, dict)
    }
    accepted: list[ExplicitSchemaHit] = []
    seen_pairs: set[tuple[str, str]] = set()
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        raw_key = entry.get("key")
        evidence = entry.get("evidence")
        if not isinstance(raw_key, str) or not isinstance(evidence, str):
            continue
        normalized = normalize_key(raw_key)
        if normalized is None:
            continue
        evidence_clean = clip_evidence(evidence)
        if not evidence_clean:
            continue
        pair = (normalized, evidence_clean)
        if pair in seen_pairs:
            continue
        if normalized in SCIENTIFIC_WHITELIST:
            accepted.append(
                {
                    "raw_key": raw_key,
                    "normalized_key": normalized,
                    "evidence": evidence_clean,
                }
            )
            seen_pairs.add(pair)
            continue
        if any(pattern.search(normalized) for pattern in VALUE_PATTERNS):
            continue
        if is_blacklisted_instrument_key(normalized):
            continue
        if is_experimental_condition_key(normalized):
            # growth_/incubation_/optimum_/storage_/treatment_/... 前缀
            # 属于实验条件或生理特征，不是原位样本元数据，静默丢弃
            continue
        if len(normalized) > 60:
            if pair not in review_pool_keys:
                review_pool.append(
                    {
                        "raw_key": raw_key,
                        "normalized_key": normalized,
                        "evidence": evidence_clean,
                        "reason": f"key过长({len(normalized)}字符)",
                    }
                )
                review_pool_keys.add(pair)
            continue
        if re.search(r"\d{4}", normalized):
            if pair not in review_pool_keys:
                review_pool.append(
                    {
                        "raw_key": raw_key,
                        "normalized_key": normalized,
                        "evidence": evidence_clean,
                        "reason": "包含4位数字，可能携带年份",
                    }
                )
                review_pool_keys.add(pair)
            continue
        accepted.append(
            {
                "raw_key": raw_key,
                "normalized_key": normalized,
                "evidence": evidence_clean,
            }
        )
        seen_pairs.add(pair)
    return accepted
