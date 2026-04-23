import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from metaagent_run.core import AsyncLocalModelClient

from .config import RuntimeConfig, load_runtime_config
from .processor import process_batch_items, process_single_item
from .prompt_builder import load_prompt_template


TARGET_ENVS: Set[str] = {"Open_ocean", "Coastal_waters", "Lake", "Wetlands"}


# ── NCBI metagenomes whitelist pre-filter ─────────────────────────────────
# 对应方法论 v4 §3.2：NCBI Taxonomy 的 `metagenomes` 根节点 + `ecological
# metagenomes` 子树作为白名单；organismal metagenomes 与物种学名样本在
# 进入 LLM 前被确定性地标记为 Others，直接实现 §2.1 criterion 4 "非宿主关联"
# 的 scope 边界。
# ──────────────────────────────────────────────────────────────────────────

def load_ncbi_env_whitelist(path: str) -> Set[str]:
    """加载 NCBI metagenome 白名单；自动展开节点名的单复数形式。

    文件里前两行是 NCBI 节点名（复数，如 `metagenomes` / `ecological metagenomes`），
    其余是单数的具体 organism 术语（如 `marine metagenome`）。实际 BioSample 的
    organism 字段取单数形式（如 `Metagenome`），所以对任何以 `s` 结尾的 term
    同时加入去 `s` 的单数形式。
    """
    terms: Set[str] = set()
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"NCBI env whitelist not found: {path}")
    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            term = line.strip().lower()
            if not term:
                continue
            terms.add(term)
            if term.endswith("s"):
                terms.add(term[:-1])
    return terms


def is_in_ncbi_env_whitelist(organism: Any, whitelist: Set[str]) -> bool:
    if not isinstance(organism, str):
        return False
    return organism.strip().lower() in whitelist


def _build_prefilter_record(item: Dict[str, Any]) -> Dict[str, Any]:
    """为被 pre-filter 排除的样本构造 Others 记录（结构与 LLM 输出一致）。"""
    return {
        "biosample_id": item.get("biosample_id", ""),
        "organism": item.get("organism", ""),
        "env_tag": {
            "value": "Others",
            "source_field": "organism",
            "reason": (
                "pre-filter: organism not in NCBI metagenomes + "
                "ecological_metagenomes whitelist"
            ),
        },
    }


def append_filtered_record(
    filtered_log_file: str,
    item: Dict[str, Any],
    reason: str,
) -> None:
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "biosample_id": item.get("biosample_id", ""),
        "organism": item.get("organism", ""),
        "reason": reason,
    }
    with open(filtered_log_file, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def _flatten_items(raw_data: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_data, dict):
        return [raw_data]
    if not isinstance(raw_data, list):
        raise ValueError("Input format error: expected JSON object or array.")

    items: List[Dict[str, Any]] = []
    for entry in raw_data:
        if isinstance(entry, dict):
            items.append(entry)
        elif isinstance(entry, list):
            items.extend(obj for obj in entry if isinstance(obj, dict))
    return items


def load_input_items(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as file:
        raw_data = json.load(file)
    return _flatten_items(raw_data)


def load_checkpoint(checkpoint_file: str) -> Tuple[Set[str], List[Dict[str, Any]]]:
    processed_ids: Set[str] = set()
    completed_results: List[Dict[str, Any]] = []

    if not os.path.exists(checkpoint_file):
        return processed_ids, completed_results

    with open(checkpoint_file, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = record.get("biosample_id")
                if isinstance(key, str) and key.strip():
                    processed_ids.add(key)
                    completed_results.append(record)
            except (TypeError, json.JSONDecodeError):
                continue

    return processed_ids, completed_results


def build_tasks_queue(
    items: List[Dict[str, Any]],
    processed_ids: Set[str],
    whitelist: Set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """返回 (llm_queue, prefiltered_records)。

    通过 NCBI 白名单匹配的样本进 LLM 队列；未匹配的样本在此直接构造 Others 记录，
    不进入 LLM。pre-filter 结果由调用方写入 checkpoint + filtered log。
    """
    queue: List[Dict[str, Any]] = []
    prefiltered: List[Dict[str, Any]] = []
    for item in items:
        key = item.get("biosample_id")
        if not isinstance(key, str) or not key.strip():
            continue
        if key in processed_ids:
            continue
        if is_in_ncbi_env_whitelist(item.get("organism"), whitelist):
            queue.append({"item": item, "key": key})
        else:
            prefiltered.append(item)
    return queue, prefiltered


def add_prompt_prefix_to_output_name(output_file: str, prompt_version: str) -> str:
    directory, filename = os.path.split(output_file)
    prefix = f"{prompt_version}_"
    if filename.startswith(prefix):
        return output_file
    prefixed = f"{prefix}{filename}"
    return os.path.join(directory, prefixed) if directory else prefixed


async def main_async(
    input_file: str,
    output_file: str,
    max_concurrency: Optional[int] = None,
    runtime_config: Optional[RuntimeConfig] = None,
) -> None:
    config = runtime_config or load_runtime_config()
    resolved_max_concurrency = (
        max_concurrency if max_concurrency is not None else config.max_concurrency
    )
    resolved_output_file = add_prompt_prefix_to_output_name(
        output_file,
        config.prompt_version,
    )

    failed_log_file = config.failed_log_file
    filtered_log_file = config.filtered_log_file
    checkpoint_file = f"{resolved_output_file}.checkpoint.jsonl"

    try:
        items = load_input_items(input_file)
    except Exception as error:
        print(f"❌ Read Error: {error}")
        return

    print(f"📋 Loaded {len(items)} items from {input_file}")

    try:
        load_prompt_template(config.prompt_version)
    except Exception as error:
        print(f"❌ Prompt config error: {error}")
        return

    try:
        whitelist = load_ncbi_env_whitelist(config.ncbi_env_whitelist_file)
    except Exception as error:
        print(f"❌ NCBI whitelist error: {error}")
        return
    print(f"📘 NCBI env whitelist: {len(whitelist)} terms (singular/plural expanded)")

    processed_ids, completed_results = load_checkpoint(checkpoint_file)
    print(f"🔄 Checkpoint: Loaded {len(processed_ids)} processed records.")

    tasks_queue, prefiltered = build_tasks_queue(items, processed_ids, whitelist)
    print(
        f"🧹 Pre-filter: {len(prefiltered)} samples → Others (organism not in NCBI whitelist)"
    )
    print(f"🚀 Queued {len(tasks_queue)} items for LLM processing.")

    # 把 pre-filter 结果直接写入 checkpoint + filtered log,结构与 LLM 产出一致
    for item in prefiltered:
        record = _build_prefilter_record(item)
        completed_results.append(record)
        with open(checkpoint_file, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        append_filtered_record(
            filtered_log_file, item, reason="organism_not_in_ncbi_env_whitelist"
        )

    if not tasks_queue:
        # 只有 pre-filter 输出,没有 LLM 任务,直接写最终文件(4 类-only)
        pure_results = [
            r for r in completed_results
            if r.get("env_tag", {}).get("value") in TARGET_ENVS
        ]
        others_count = len(completed_results) - len(pure_results)
        with open(resolved_output_file, "w", encoding="utf-8") as file:
            json.dump(pure_results, file, ensure_ascii=False, indent=2)
        print(
            f"✅ Done (pre-filter only). Saved {len(pure_results)} 4-class records to "
            f"{resolved_output_file}; {others_count} Others records archived in "
            f"{filtered_log_file}"
        )
        return

    checkpoint_lock = asyncio.Lock()
    failed_lock = asyncio.Lock()

    async def save_checkpoint(item: Dict[str, Any]) -> None:
        async with checkpoint_lock:
            with open(checkpoint_file, "a", encoding="utf-8") as file:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
                file.flush()

    async def log_failure(task: Dict[str, Any], reason: Exception) -> None:
        item = task.get("item", {})
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "biosample_id": item.get("biosample_id", "unknown"),
            "prompt_version": config.prompt_version,
            "model": config.model,
            "error_type": type(reason).__name__,
            "error": str(reason),
        }
        async with failed_lock:
            with open(failed_log_file, "a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()

    async with AsyncLocalModelClient(
        base_url=config.base_url,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=config.api_key,
        stop_sentinel=config.stop_sentinel,
        api_style=config.api_style,
        azure_api_version=config.azure_api_version,
        azure_deployment=config.azure_deployment,
        auth_mode=config.auth_mode,
    ) as client:
        current_concurrency = max(1, min(8, resolved_max_concurrency))
        semaphore = asyncio.Semaphore(current_concurrency)

        async def ramp_up() -> None:
            nonlocal current_concurrency
            while current_concurrency < resolved_max_concurrency:
                await asyncio.sleep(15)
                increment = min(4, resolved_max_concurrency - current_concurrency)
                for _ in range(increment):
                    semaphore.release()
                current_concurrency += increment
                tqdm.write(f"🚀 Ramp Up: {current_concurrency}")

        batch_size = config.batch_size
        task_batches = [
            tasks_queue[i:i + batch_size]
            for i in range(0, len(tasks_queue), batch_size)
        ]

        pbar = tqdm(total=len(tasks_queue), desc="Classifying Samples")

        async def worker(batch: List[Dict[str, Any]]) -> None:
            async with semaphore:
                try:
                    items = [t["item"] for t in batch]
                    keys = [str(t["key"]) for t in batch]
                    results = await process_batch_items(
                        client, items, keys, runtime_config=config
                    )
                    for task, result in zip(batch, results):
                        if result is not None:
                            completed_results.append(result)
                            await save_checkpoint(result)
                            # LLM 判定为 Others 的样本同步进 filtered log,
                            # 与 pre-filter Others 合并为单一 Others 档案
                            env_value = result.get("env_tag", {}).get("value", "")
                            if env_value == "Others":
                                append_filtered_record(
                                    filtered_log_file,
                                    task["item"],
                                    reason="llm_determined_others",
                                )
                except Exception as error:
                    for task in batch:
                        await log_failure(task, error)
                finally:
                    pbar.update(len(batch))

        ramp_task = asyncio.create_task(ramp_up())
        await asyncio.gather(*(worker(batch) for batch in task_batches))
        ramp_task.cancel()
        pbar.close()

    # 最终输出仅保留 4 类水圈环境样本;Others(pre-filter + LLM 判定)全部归档到
    # filtered log。checkpoint 保留全量记录用于 resume。
    pure_results = [
        r for r in completed_results
        if r.get("env_tag", {}).get("value") in TARGET_ENVS
    ]
    others_count = len(completed_results) - len(pure_results)
    with open(resolved_output_file, "w", encoding="utf-8") as file:
        json.dump(pure_results, file, ensure_ascii=False, indent=2)

    if os.path.exists(failed_log_file):
        print(f"⚠️ Warnings found. Check {failed_log_file} for failed tasks.")
    print(
        f"✅ Done. Saved {len(pure_results)} 4-class records to {resolved_output_file}; "
        f"{others_count} Others records archived in {filtered_log_file}"
    )
