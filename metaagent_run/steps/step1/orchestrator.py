import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Set, Tuple

from metaagent_run.core import AsyncLocalModelClient, load_json_items

from .config import RuntimeConfig, load_runtime_config
from .prompt_builder import load_prompt_template
from .processor import (
    build_paper_env_map,
    build_relation_input,
    ensure_paragraph_items,
    extract_abstract_texts,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def add_prompt_prefix_to_output_name(output_file: Path, prompt_version: str) -> Path:
    prefix = f"{prompt_version}_"
    filename = output_file.name
    if filename.startswith(prefix):
        return output_file
    return output_file.with_name(f"{prefix}{filename}")


def normalize_env_list(raw_value: object) -> list[str]:
    if not isinstance(raw_value, list):
        return []
    valid = {"Open_ocean", "Coastal_waters", "Lake", "Wetlands", "Others"}
    labels: list[str] = []
    seen: set[str] = set()
    for value in raw_value:
        label = str(value).strip()
        if label not in valid or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    if "Others" in seen and any(label != "Others" for label in labels):
        labels = [label for label in labels if label != "Others"]
    return labels


def load_checkpoint(checkpoint_file: Path) -> Tuple[Set[str], dict[str, list[str]]]:
    processed_keys: Set[str] = set()
    env_map: dict[str, list[str]] = {}

    if not checkpoint_file.exists():
        return processed_keys, env_map

    with checkpoint_file.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    continue
                pmid = str(record.get("pmid", "")).strip()
                env = normalize_env_list(record.get("env", []))
                if not pmid or not env:
                    continue
                processed_keys.add(pmid)
                env_map[pmid] = env
            except (TypeError, ValueError):
                continue

    return processed_keys, env_map


def load_filtered_keys(filtered_log_file: Path) -> Tuple[Set[str], dict[str, list[str]]]:
    processed_keys: Set[str] = set()
    env_map: dict[str, list[str]] = {}

    if not filtered_log_file.exists():
        return processed_keys, env_map

    with filtered_log_file.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    continue
                pmid = str(record.get("pmid", "")).strip()
                if not pmid:
                    continue
                processed_keys.add(pmid)
                env = normalize_env_list(record.get("env", []))
                if env:
                    env_map[pmid] = env
            except (TypeError, ValueError):
                continue

    return processed_keys, env_map


def append_checkpoint_record(checkpoint_file: Path, pmid: str, env: list[str]) -> None:
    record = {
        "pmid": pmid,
        "env": env,
    }
    with checkpoint_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def append_filtered_record(
    filtered_log_file: Path,
    pmid: str,
    reason: str,
    abstract_length: int,
    env: list[str],
) -> None:
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "pmid": pmid,
        "reason": reason,
        "abstract_length": abstract_length,
        "env": env,
    }
    with filtered_log_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def append_failed_record(
    failed_log_file: Path,
    pmid: str,
    prompt_version: str,
    model: str,
    error_type: str,
    error: str,
) -> None:
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "pmid": pmid,
        "prompt_version": prompt_version,
        "model": model,
        "error_type": error_type,
        "error": error,
    }
    with failed_log_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


async def main_async(runtime_config: Optional[RuntimeConfig] = None) -> None:
    config = runtime_config or load_runtime_config()

    try:
        load_prompt_template(config.prompt_version)
    except Exception as error:
        LOGGER.error("Prompt config error: %s", error)
        return

    resolved_output_file = add_prompt_prefix_to_output_name(
        config.relation_input_out,
        config.prompt_version,
    )
    checkpoint_file = Path(f"{resolved_output_file}.checkpoint.jsonl")
    failed_log_file = config.failed_log_file
    filtered_log_file = config.filtered_log_file

    checkpoint_keys, checkpoint_env_map = load_checkpoint(checkpoint_file)
    filtered_keys, filtered_env_map = load_filtered_keys(filtered_log_file)
    processed_keys = checkpoint_keys | filtered_keys

    LOGGER.info(
        "Checkpoint: 已加载 %d 条checkpoint, %d 条filtered",
        len(checkpoint_keys),
        len(filtered_keys),
    )

    try:
        raw_items = load_json_items(str(config.full_text_json))
    except Exception as error:
        LOGGER.error("Read input error: %s", error)
        return

    full_items = ensure_paragraph_items(raw_items)
    LOGGER.info("全量段落: %d 条", len(full_items))

    pmid_abstract_text = extract_abstract_texts(full_items)
    abstract_pmids = set(pmid_abstract_text)

    initial_env_map: dict[str, list[str]] = {}
    initial_env_map.update(filtered_env_map)
    initial_env_map.update(checkpoint_env_map)

    checkpoint_written: set[str] = set(initial_env_map)

    def on_checkpoint(pmid: str, env: list[str]) -> None:
        if pmid in checkpoint_written:
            return
        append_checkpoint_record(checkpoint_file, pmid, env)
        checkpoint_written.add(pmid)

    def on_filtered(pmid: str, reason: str, abstract_length: int) -> None:
        append_filtered_record(filtered_log_file, pmid, reason, abstract_length, ["Others"])

    def on_failed(pmid: str, error_type: str, error: str) -> None:
        append_failed_record(
            failed_log_file,
            pmid,
            config.prompt_version,
            config.model,
            error_type,
            error,
        )


    async with AsyncLocalModelClient(
        base_url=config.base_url,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=config.api_key,
        api_style=config.api_style,
        azure_api_version=config.azure_api_version,
        azure_deployment=config.azure_deployment,
        auth_mode=config.auth_mode,
    ) as llm_client:
        paper_env_map = await build_paper_env_map(
            pmid_abstract_text=pmid_abstract_text,
            llm_client=llm_client,
            runtime_config=config,
            initial_env_map=initial_env_map,
            processed_pmids=processed_keys,
            on_checkpoint=on_checkpoint,
            on_filtered=on_filtered,
            on_failed=on_failed,

        )

    valid_target_pmids = {
        pmid
        for pmid, envs in paper_env_map.items()
        if any(env in {"Open_ocean", "Coastal_waters", "Lake", "Wetlands"} for env in envs)
    }
    LOGGER.info("有效目标环境文献pmid集合: %d 篇", len(valid_target_pmids))

    relation_input_items = build_relation_input(
        full_items=full_items,
        valid_target_pmids=valid_target_pmids,
        abstract_pmids=abstract_pmids,
    )

    with resolved_output_file.open("w", encoding="utf-8") as file:
        json.dump(relation_input_items, file, ensure_ascii=False, indent=2)
    LOGGER.info("已保存relation输入: %s (%d 条)", resolved_output_file, len(relation_input_items))

    if os.path.exists(str(failed_log_file)):
        LOGGER.warning("检测到失败记录，请检查: %s", failed_log_file)
