import asyncio
import json
import logging
import re
from collections import Counter, defaultdict
from typing import Callable, Optional, cast

from metaagent_run.core import AsyncLocalModelClient, backoff_with_jitter
from tqdm import tqdm

from .config import RuntimeConfig
from .prompt_builder import build_prompt


LOGGER = logging.getLogger(__name__)


ParagraphItem = dict[str, object]
VALID_ENV_LABELS = {"Open_ocean", "Coastal_waters", "Lake", "Wetlands", "Others"}
TARGET_ENV_LABELS = {"Open_ocean", "Coastal_waters", "Lake", "Wetlands"}


AQUATIC_BROAD_SCREEN = re.compile(
    r"|".join(
        [
            r"\baquatic\b",
            r"\baqueous\b",
            r"\bsubaqueous\b",
            r"\bhydro\w*\b",
            r"\bseas?\b",
            r"\bocean\w*\b",
            r"\bmarine\b",
            r"\bpelagic\b",
            r"\bbenthic\b",
            r"\bneritic\b",
            r"\btidal\b",
            r"\bintertidal\b",
            r"\bsubtidal\b",
            r"\bcoast\w*\b",
            r"\bshore\w*\b",
            r"\bestuar\w*\b",
            r"\blagoons?\b",
            r"\bfjords?\b",
            r"\bbights?\b",
            r"\bgulfs?\b",
            r"\bbays?\b",
            r"\breefs?\b",
            r"\bseeps?\b",
            r"\bvents?\b",
            r"\blakes?\b",
            r"\blacustrine\b",
            r"\blimnol\w*\b",
            r"\breservoirs?\b",
            r"\bwetlands?\b",
            r"\bmarsh\w*\b",
            r"\bmangroves?\b",
            r"\bswamps?\b",
            r"\bbogs?\b",
            r"\bfens?\b",
            r"\bmires?\b",
            r"\briparian\b",
            r"\bsalin\w*\b",
            r"\bbrackish\b",
            r"\bbrines?\b",
            r"\bplankton\b",
            r"\bphytoplankton\b",
            r"\bzooplankton\b",
            r"\bbathymetr\w*\b",
            r"\bepipelagic\b",
            r"\bmesopelagic\b",
            r"\bbathypelagic\b",
            r"\boceanograph\w*\b",
            r"\banchialine\b",
            r"\babyssal\b",
            r"\bbathyal\b",
            # ── 补充：水圈相关但原列表遗漏的关键词 ──
            r"\bsediment\w*\b",
            r"\bfloodplain\w*\b",
            r"\beutrophi\w*\b",
            r"\bcyanobacteri\w*\b",
            r"\bpeat\w*\b",
            r"\briver\w*\b",
            r"\bstreams?\b",
            r"\bcreeks?\b",
            r"\bcatchments?\b",
            r"\bwatersheds?\b",
            r"\bgroundwater\b",
            r"\bsubsurface.{0,5}water\b",
            r"\bwater.{0,3}column\b",
            r"\bwater.{0,3}body\b",
            r"\bwater.{0,3}sample\b",
            r"\bfreshwater\b",
            r"\bseawater\b",
            r"\bponds?\b",
            r"\bdeltas?\b",
            r"\bwhale.{0,3}fall\w*\b",
            r"\bmudflats?\b",
            r"\bmud.{0,3}flats?\b",
            r"\bstratifi\w*\b"
        ]
    ),
    re.IGNORECASE,
)


def get_pmid(item: ParagraphItem) -> str:
    return str(item.get("pmid", "")).strip()


def normalize_section(raw: object) -> str:
    text = str(raw or "").strip()
    return text.upper() if text else "unknown"


def is_abstract_section(raw: object) -> bool:
    return normalize_section(raw) == "ABSTRACT"


def is_clearly_non_aquatic(text: str) -> bool:
    if not text:
        return True
    return AQUATIC_BROAD_SCREEN.search(text) is None


def extract_abstract_texts(full_items: list[ParagraphItem]) -> dict[str, str]:
    abstract_parts_by_pmid: dict[str, list[str]] = defaultdict(list)
    unique_pmids: set[str] = set()

    for item in full_items:
        pmid = get_pmid(item)
        if not pmid:
            continue
        unique_pmids.add(pmid)
        if not is_abstract_section(item.get("section_type", "")):
            continue
        text = str(item.get("text", "") or "").strip()
        if text:
            abstract_parts_by_pmid[pmid].append(text)

    abstract_text_map = {
        pmid: "\n\n".join(parts)
        for pmid, parts in abstract_parts_by_pmid.items()
        if parts
    }

    LOGGER.info("全量数据共 %d 篇文献，%d 个段落", len(unique_pmids), len(full_items))
    LOGGER.info(
        "有摘要段落的文献: %d/%d (%.1f%%)",
        len(abstract_text_map),
        len(unique_pmids),
        len(abstract_text_map) / len(unique_pmids) * 100 if unique_pmids else 0,
    )
    return abstract_text_map


def get_llm_failure_fallback(runtime_config: RuntimeConfig) -> str:
    fallback = runtime_config.llm_failure_fallback
    if fallback == "Others":
        return fallback
    LOGGER.warning(
        "环境变量 ABSTRACT_ENV_LLM_FAILURE_FALLBACK=%s 非法，回退使用 Others",
        fallback,
    )
    return "Others"


def normalize_env_labels(raw_value: object) -> list[str]:
    if not isinstance(raw_value, list):
        return []
    labels: list[str] = []
    seen: set[str] = set()
    for value in raw_value:
        label = str(value).strip()
        if label not in VALID_ENV_LABELS or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    if not labels:
        return []
    if "Others" in seen and any(label in TARGET_ENV_LABELS for label in labels):
        labels = [label for label in labels if label != "Others"]
    return labels


def parse_llm_env_labels(result: str) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for token in result.strip().split(","):
        label = token.strip()
        if label not in VALID_ENV_LABELS or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    if "Others" in seen and any(label in TARGET_ENV_LABELS for label in labels):
        labels = [label for label in labels if label != "Others"]
    return labels


async def llm_classify_target_env(
    text: str,
    pmid: str,
    llm_client: AsyncLocalModelClient,
    runtime_config: RuntimeConfig,
) -> tuple[list[str], bool]:
    messages = build_prompt(
        input_text=text[:2000],
        prompt_version=runtime_config.prompt_version,
    )

    for attempt in range(runtime_config.llm_retry_times):
        response = await llm_client.chat(messages, temperature_override=runtime_config.retry_temps[min(attempt, len(runtime_config.retry_temps) - 1)])
        if response is None:
            if attempt < runtime_config.llm_retry_times - 1:
                delay = backoff_with_jitter(
                    attempt,
                    base=runtime_config.backoff_base,
                    cap=runtime_config.backoff_cap,
                )
                tqdm.write(
                    f"⚠️ [Retry {attempt + 1}/{runtime_config.llm_retry_times}] PMID={pmid}: "
                    + f"No response (sleep {delay:.1f}s)"
                )
                await asyncio.sleep(delay)
            continue

        result = response.strip()
        labels = parse_llm_env_labels(result)
        if labels:
            return labels, False

        if attempt < runtime_config.llm_retry_times - 1:
            delay = backoff_with_jitter(
                attempt,
                base=runtime_config.backoff_base,
                cap=runtime_config.backoff_cap,
            )
            tqdm.write(
                f"⚠️ [Retry {attempt + 1}/{runtime_config.llm_retry_times}] PMID={pmid}: "
                + f"Unexpected output '{result[:60]}' (sleep {delay:.1f}s)"
            )
            await asyncio.sleep(delay)

    fallback = get_llm_failure_fallback(runtime_config)
    LOGGER.warning(
        "PMID=%s 的 LLM 分类连续失败 %d 次，使用回退值 %s",
        pmid,
        runtime_config.llm_retry_times,
        fallback,
    )
    return [fallback], True


async def build_paper_env_map(
    pmid_abstract_text: dict[str, str],
    llm_client: AsyncLocalModelClient,
    runtime_config: RuntimeConfig,
    initial_env_map: Optional[dict[str, list[str]]] = None,
    processed_pmids: Optional[set[str]] = None,
    on_checkpoint: Optional[Callable[[str, list[str]], None]] = None,
    on_filtered: Optional[Callable[[str, str, int], None]] = None,
    on_failed: Optional[Callable[[str, str, str], None]] = None,

) -> dict[str, list[str]]:
    cache_path = runtime_config.paper_env_cache
    paper_env_map: dict[str, list[str]] = {
        pmid: list(envs) for pmid, envs in (initial_env_map or {}).items()
    }
    processed = processed_pmids or set()

    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as file:
            cached_json: object = json.load(file)
        cached_env_map: dict[str, list[str]] = {}
        if isinstance(cached_json, dict):
            for key, value in cast(dict[object, object], cached_json).items():
                labels = normalize_env_labels(value)
                if labels:
                    cached_env_map[str(key)] = labels
        cached_valid_map = {
            pmid: env
            for pmid, env in cached_env_map.items()
            if pmid in pmid_abstract_text
        }
        for pmid, env in cached_valid_map.items():
            if pmid not in paper_env_map:
                paper_env_map[pmid] = env
        LOGGER.info("加载env缓存: %d 篇", len(paper_env_map))

    candidate_pmids: list[str] = []
    for pmid, text in pmid_abstract_text.items():
        if pmid in paper_env_map:
            continue
        if pmid in processed:
            continue
        if is_clearly_non_aquatic(text):
            paper_env_map[pmid] = ["Others"]
            if on_filtered is not None:
                on_filtered(pmid, "broad_screen_non_aquatic", len(text))
            if on_checkpoint is not None:
                on_checkpoint(pmid, ["Others"])
        else:
            candidate_pmids.append(pmid)

    LOGGER.info(
        "摘要粗筛: 候选=%d, 直接标other=%d",
        len(candidate_pmids),
        sum(1 for envs in paper_env_map.values() if envs == ["Others"]),
    )

    if candidate_pmids:
        LOGGER.info(
            "LLM分类: %d 篇摘要候选文献，并发=%d",
            len(candidate_pmids),
            runtime_config.max_concurrency,
        )
        resolved_max_concurrency = max(1, runtime_config.max_concurrency)
        current_concurrency = max(1, min(8, resolved_max_concurrency))
        semaphore = asyncio.Semaphore(current_concurrency)
        completed = 0
        fallback_count = 0

        async def ramp_up() -> None:
            nonlocal current_concurrency
            while current_concurrency < resolved_max_concurrency:
                await asyncio.sleep(15)
                increment = min(8, resolved_max_concurrency - current_concurrency)
                for _ in range(increment):
                    semaphore.release()
                current_concurrency += increment
                tqdm.write(f"🚀 Ramp Up: {current_concurrency}")

        async def classify_one(
            pmid: str,
        ) -> tuple[str, list[str], bool, Optional[Exception]]:
            async with semaphore:
                try:
                    env, used_fallback = await llm_classify_target_env(
                        pmid_abstract_text[pmid],
                        pmid,
                        llm_client,
                        runtime_config,
                    )
                    return pmid, env, used_fallback, None
                except Exception as error:
                    return pmid, "", False, error

        tasks = [classify_one(pmid) for pmid in candidate_pmids]
        ramp_task = asyncio.create_task(ramp_up())
        pbar = tqdm(total=len(candidate_pmids), desc="Classifying Abstracts")
        try:
            for coro in asyncio.as_completed(tasks):
                pmid, env, used_fallback, error = await coro
                if error is not None:
                    env = [get_llm_failure_fallback(runtime_config)]
                    used_fallback = True
                    tqdm.write(
                        f"❌ [Failed] PMID={pmid}: {type(error).__name__}: {error}"
                    )
                    if on_failed is not None:
                        on_failed(pmid, type(error).__name__, str(error))

                paper_env_map[pmid] = env
                if on_checkpoint is not None:
                    on_checkpoint(pmid, env)

                completed += 1
                if used_fallback:
                    fallback_count += 1
                    if on_failed is not None:
                        on_failed(
                            pmid,
                            "LLMClassificationFallback",
                            f"Used fallback value: {','.join(env)}",
                        )
                pbar.update(1)

                if completed % 100 == 0:
                    with cache_path.open("w", encoding="utf-8") as file:
                        json.dump(paper_env_map, file, ensure_ascii=False, indent=2)
        finally:
            ramp_task.cancel()
            pbar.close()

        if fallback_count:
            LOGGER.warning(
                "LLM 回退分类共触发 %d 次，当前回退值=%s",
                fallback_count,
                get_llm_failure_fallback(runtime_config),
            )

    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(paper_env_map, file, ensure_ascii=False, indent=2)

    env_counts: Counter[str] = Counter()
    multi_label_count = 0
    pure_other_count = 0
    for envs in paper_env_map.values():
        if envs == ["Others"]:
            pure_other_count += 1
        if len(envs) > 1:
            multi_label_count += 1
        for env in envs:
            env_counts[env] += 1
    total = len(paper_env_map)
    LOGGER.info("env分布: %s", dict(env_counts))
    LOGGER.info("纯 other 文献: %d/%d (%.1f%%)", pure_other_count, total, pure_other_count / total * 100 if total else 0)
    LOGGER.info("多标签文献: %d/%d (%.1f%%)", multi_label_count, total, multi_label_count / total * 100 if total else 0)
    return paper_env_map


def build_relation_input(
    full_items: list[ParagraphItem],
    valid_target_pmids: set[str],
    abstract_pmids: set[str],
) -> list[ParagraphItem]:
    valid_paper_pmids = {
        pmid
        for pmid in valid_target_pmids
        if pmid in abstract_pmids
    }
    LOGGER.info("最终有效文献（有摘要 AND target）: %d 篇", len(valid_paper_pmids))

    relation_input_items: list[ParagraphItem] = []
    skip_non_valid_paper = 0
    skip_abstract = 0

    for item in full_items:
        pmid = get_pmid(item)
        if pmid not in valid_paper_pmids:
            skip_non_valid_paper += 1
            continue

        if is_abstract_section(item.get("section_type", "")):
            skip_abstract += 1
            continue

        relation_input_items.append(
            {
                **item,
            }
        )

    LOGGER.info(
        "筛选结果: 保留=%d, 跳过(非有效文献)=%d, 跳过(Abstract)=%d",
        len(relation_input_items),
        skip_non_valid_paper,
        skip_abstract,
    )
    return relation_input_items


def ensure_paragraph_items(raw_items: list[dict[str, object]]) -> list[ParagraphItem]:
    normalized: list[ParagraphItem] = []
    for item in raw_items:
        normalized.append(dict(item))
    return normalized
