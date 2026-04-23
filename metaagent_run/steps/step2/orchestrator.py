import asyncio
from contextlib import AsyncExitStack
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional

from tqdm import tqdm

from metaagent_run.core import AsyncLocalModelClient, is_excluded_section, load_json_items

from .config import RuntimeConfig, load_runtime_config
from .processor import (
    build_discovery_input,
    build_relation_map,
    get_pmid,
    load_pmid_year_map,
)
from .prompt_builder import load_prompt_template
from .relation_processor import process_batch_relation_items, process_single_relation_item


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def build_item_key(pmid: Any, source: Any, section_type: Any, index: Any) -> str:
    return f"{pmid}::{source}::{section_type}::{index}"


def add_prompt_prefix_to_output_name(output_file: str, prompt_version: str) -> str:
    output_path = Path(output_file)
    prefix = f"{prompt_version}_"
    if output_path.name.startswith(prefix):
        return str(output_path)
    return str(output_path.with_name(f"{prefix}{output_path.name}"))


def load_checkpoint(checkpoint_file: Path) -> tuple[set[str], list[dict[str, Any]]]:
    processed_keys: set[str] = set()
    completed_results: list[dict[str, Any]] = []
    if not checkpoint_file.exists():
        return processed_keys, completed_results

    with checkpoint_file.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    continue
                pmid = record.get("pmid", "unknown")
                source = record.get("source", "unknown")
                section_type = record.get("section_type", "unknown")
                index = record.get("index", 0)
                processed_keys.add(build_item_key(pmid, source, section_type, index))
                completed_results.append(record)
            except (TypeError, ValueError):
                continue
    return processed_keys, completed_results


def load_filtered_keys(filtered_log_file: Path) -> set[str]:
    processed_keys: set[str] = set()
    if not filtered_log_file.exists():
        return processed_keys

    with filtered_log_file.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    continue
                pmid = record.get("pmid", "unknown")
                source = record.get("source", "unknown")
                section_type = record.get("section_type", "unknown")
                index = record.get("index", 0)
                processed_keys.add(build_item_key(pmid, source, section_type, index))
            except (TypeError, ValueError):
                continue
    return processed_keys


def append_filtered_record(
    filtered_log_file: Path,
    item: dict[str, Any],
    reason: str,
    min_text_length: int,
) -> None:
    text = item.get("text", "")
    text_length = len(text.strip()) if isinstance(text, str) else 0
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "pmid": item.get("pmid", "unknown"),
        "source": item.get("source", "unknown"),
        "section_type": item.get("section_type", "unknown"),
        "index": item.get("index", 0),
        "reason": reason,
        "text_length": text_length,
        "min_text_length": min_text_length,
    }
    with filtered_log_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()



def build_task_queue(
    items_list: list[dict[str, Any]],
    processed_keys: set[str],
    runtime_config: RuntimeConfig,
    filtered_log_file: Path,
) -> tuple[list[dict[str, Any]], int]:
    tasks_queue: list[dict[str, Any]] = []
    filtered_count = 0

    for entry in items_list:
        pmid = entry.get("pmid", "unknown")
        source = entry.get("source", "unknown")
        section_type = entry.get("section_type", "")
        index = entry.get("index", 0)
        key = build_item_key(pmid, source, section_type, index)
        if key in processed_keys:
            continue

        if isinstance(section_type, str) and is_excluded_section(
            section_type,
            runtime_config.excluded_section_types,
        ):
            append_filtered_record(
                filtered_log_file,
                entry,
                reason="excluded_section_type",
                min_text_length=runtime_config.min_text_length,
            )
            processed_keys.add(key)
            filtered_count += 1
            continue

        text_content = entry.get("text", "")
        if not isinstance(text_content, str):
            continue
        if len(text_content.strip()) < runtime_config.min_text_length:
            append_filtered_record(
                filtered_log_file,
                entry,
                reason="text_too_short",
                min_text_length=runtime_config.min_text_length,
            )
            processed_keys.add(key)
            filtered_count += 1
            continue

        tasks_queue.append({"item": entry, "key": key})

    return tasks_queue, filtered_count


async def run_relation_extraction(
    items_list: list[dict[str, Any]],
    relation_output_file: Path,
    max_concurrency: Optional[int],
    runtime_config: RuntimeConfig,
) -> Path:
    resolved_max_concurrency = (
        max_concurrency if max_concurrency is not None else runtime_config.max_concurrency
    )
    resolved_output = Path(
        add_prompt_prefix_to_output_name(
            str(relation_output_file),
            runtime_config.relation_prompt_version,
        )
    )
    checkpoint_file = Path(f"{resolved_output}.checkpoint.jsonl")

    checkpoint_keys, completed_results = load_checkpoint(checkpoint_file)
    filtered_keys = load_filtered_keys(runtime_config.relation_filtered_log_file)
    processed_keys = checkpoint_keys | filtered_keys

    tasks_queue, filtered_count = build_task_queue(
        items_list,
        processed_keys,
        runtime_config,
        runtime_config.relation_filtered_log_file,
    )
    LOGGER.info(
        "Checkpoint: checkpoint=%d, filtered=%d, queued=%d",
        len(checkpoint_keys),
        len(filtered_keys),
        len(tasks_queue),
    )
    if filtered_count:
        LOGGER.info("Filtered=%d, log=%s", filtered_count, runtime_config.relation_filtered_log_file)

    if not tasks_queue:
        with resolved_output.open("w", encoding="utf-8") as file:
            json.dump(completed_results, file, ensure_ascii=False, indent=2)
        return resolved_output

    load_prompt_template(runtime_config.relation_prompt_version)

    checkpoint_lock = asyncio.Lock()
    failed_lock = asyncio.Lock()

    async def save_checkpoint(item: dict[str, Any]) -> None:
        async with checkpoint_lock:
            with checkpoint_file.open("a", encoding="utf-8") as file:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
                file.flush()

    async def log_failure(task: dict[str, Any], reason: Exception, endpoint: str = "") -> None:
        item = task.get("item", {})
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "stage": "relation_extraction",
            "pmid": item.get("pmid", "unknown"),
            "source": item.get("source", "unknown"),
            "section_type": item.get("section_type", "unknown"),
            "index": item.get("index", 0),
            "prompt_version": runtime_config.relation_prompt_version,
            "model": runtime_config.model,
            "endpoint": endpoint,
            "error_type": type(reason).__name__,
            "error": str(reason),
        }
        async with failed_lock:
            with runtime_config.relation_failed_log_file.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()

    # ── 多端点 client 池 ──────────────────────────────────
    batch_size = runtime_config.relation_batch_size
    task_batches = [tasks_queue[i:i+batch_size] for i in range(0, len(tasks_queue), batch_size)]

    task_queue: asyncio.Queue[list[dict[str, Any]]] = asyncio.Queue()
    for batch in task_batches:
        task_queue.put_nowait(batch)

    pbar = tqdm(total=len(tasks_queue), desc="Relation Extraction")

    async with AsyncExitStack() as stack:
        clients: List[AsyncLocalModelClient] = []
        for ep in runtime_config.endpoints:
            c = await stack.enter_async_context(AsyncLocalModelClient(
                base_url=ep["base_url"],
                model=ep.get("model", runtime_config.model),
                temperature=runtime_config.temperature,
                max_tokens=runtime_config.max_tokens,
                api_key=os.environ.get(ep.get("api_key_env", "ALL_API_KEY"), ""),
                stop_sentinel=runtime_config.stop_sentinel,
                api_style=runtime_config.api_style,
                auth_mode=runtime_config.auth_mode,
                streaming_total_timeout=float(ep.get("streaming_timeout", "90")),
            ))
            clients.append(c)

        total_concurrency = sum(int(ep.get("max_concurrency", "8")) for ep in runtime_config.endpoints)
        LOGGER.info(
            "多端点启动: %d 个端点, 总并发=%d (%s)",
            len(clients),
            total_concurrency,
            " + ".join(f"{ep['base_url'].split('//')[1].split('/')[0]}×{ep.get('max_concurrency', '8')}" for ep in runtime_config.endpoints),
        )

        async def endpoint_worker(client_idx: int) -> None:
            client = clients[client_idx]
            ep_url = runtime_config.endpoints[client_idx]["base_url"]
            while not task_queue.empty():
                try:
                    batch = task_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    items = [t["item"] for t in batch]
                    keys = [str(t["key"]) for t in batch]
                    results = await process_batch_relation_items(client, items, keys, runtime_config)
                    for task, result in zip(batch, results):
                        if result:
                            completed_results.append(result)
                            await save_checkpoint(result)
                        else:
                            await log_failure(task, RuntimeError("No result returned (batch + fallback both failed)"), endpoint=ep_url)
                except Exception:
                    # 故障：换一个端点重试一次
                    fallback_idx = (client_idx + 1) % len(clients)
                    fallback_url = runtime_config.endpoints[fallback_idx]["base_url"]
                    try:
                        results = await process_batch_relation_items(clients[fallback_idx], items, keys, runtime_config)
                        for task, result in zip(batch, results):
                            if result:
                                completed_results.append(result)
                                await save_checkpoint(result)
                            else:
                                await log_failure(task, RuntimeError("Fallback also returned no result"), endpoint=fallback_url)
                    except Exception as error:
                        for task in batch:
                            await log_failure(task, error, endpoint=fallback_url)
                finally:
                    pbar.update(len(batch))

        all_workers = []
        for i, ep in enumerate(runtime_config.endpoints):
            concurrency = int(ep.get("max_concurrency", "8"))
            for _ in range(concurrency):
                all_workers.append(endpoint_worker(i))

        await asyncio.gather(*all_workers)
        pbar.close()

    with resolved_output.open("w", encoding="utf-8") as file:
        json.dump(completed_results, file, ensure_ascii=False, indent=2)

    if runtime_config.relation_failed_log_file.exists():
        LOGGER.warning("Relation extraction has failures, log=%s", runtime_config.relation_failed_log_file)
    return resolved_output


async def main_async(
    full_text_file: str,
    relation_file: str,
    pmid_year_file: str,
    discovery_output_file: str,
    run_relation: bool = True,
    relation_max_concurrency: Optional[int] = None,
    runtime_config: Optional[RuntimeConfig] = None,
) -> None:
    config = runtime_config or load_runtime_config()
    full_text_path = Path(full_text_file)
    relation_path = Path(relation_file)
    pmid_year_path = Path(pmid_year_file)
    discovery_output_path = Path(discovery_output_file)

    full_items = load_json_items(str(full_text_path))
    LOGGER.info("加载step1筛选输入段落: %d 条", len(full_items))
    with config.paper_env_map_file.open("r", encoding="utf-8") as file:
        raw_paper_env_map = json.load(file)
    if not isinstance(raw_paper_env_map, dict):
        raise ValueError(f"paper_env_map 顶层不是 dict: {config.paper_env_map_file}")
    paper_env_map = {
        str(pmid): [str(env).strip() for env in envs if str(env).strip()]
        for pmid, envs in raw_paper_env_map.items()
        if isinstance(pmid, str) and isinstance(envs, list)
    }
    valid_target_pmids = {
        pmid
        for pmid, envs in paper_env_map.items()
        if any(env in {"Open_ocean", "Coastal_waters", "Lake", "Wetlands"} for env in envs)
    }
    LOGGER.info("paper_env_map覆盖target文献: %d 篇", len(valid_target_pmids))

    if run_relation:
        used_relation_file = await run_relation_extraction(
            items_list=full_items,
            relation_output_file=relation_path,
            max_concurrency=relation_max_concurrency,
            runtime_config=config,
        )
    else:
        candidate = relation_path
        prefixed = Path(
            add_prompt_prefix_to_output_name(
                str(relation_path),
                config.relation_prompt_version,
            )
        )
        if candidate.exists():
            used_relation_file = candidate
        elif prefixed.exists():
            used_relation_file = prefixed
        else:
            raise FileNotFoundError("skip-relation模式下未找到relation文件。")

    LOGGER.info("使用relation文件: %s", used_relation_file)

    relation_items = load_json_items(str(used_relation_file))
    if not relation_items:
        raise RuntimeError("relation结果为空，无法继续构建discovery输入。")
    relation_map = build_relation_map(relation_items)
    if not relation_map:
        raise RuntimeError("relation结果缺少有效pmid/relation记录，无法继续。")
    LOGGER.info("加载relation标注: %d 条", len(relation_map))

    pmid_year_map = load_pmid_year_map(pmid_year_path)
    LOGGER.info("加载PMID-Year映射: %d 条", len(pmid_year_map))

    discovery_items, stats = build_discovery_input(
        full_items=full_items,
        valid_target_pmids=valid_target_pmids,
        relation_map=relation_map,
        pmid_year_map=pmid_year_map,
        paper_env_map=paper_env_map,
    )

    with discovery_output_path.open("w", encoding="utf-8") as file:
        json.dump(discovery_items, file, ensure_ascii=False, indent=2)

    LOGGER.info(
        "已保存discovery输入: %s (保留=%d, 跳过非目标环境文献=%d, 跳过摘要=%d, 跳过无metadata段落=%d)",
        discovery_output_path,
        stats.kept,
        stats.skipped_non_target_paper,
        stats.skipped_abstract,
        stats.skipped_no_metadata,
    )

    if os.path.exists(str(config.relation_failed_log_file)):
        LOGGER.warning("relation失败记录: %s", config.relation_failed_log_file)
