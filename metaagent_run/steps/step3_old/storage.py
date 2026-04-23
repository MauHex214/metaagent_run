"""I/O：checkpoint + field_pmid_index / field_pmid_env_index 的保存恢复。

相比原 storage.py：
- 移除 schema_contexts 相关（新流程 evidence 全为空，context 无意义）
- 移除 save_saturation_report（无饱和判定）
- checkpoint 瘦身：只保留 processed_key_count / history / deduplicator
"""
import json
import os
from typing import Optional

from .runtime import LOGGER
from .sampling import IterationRecord
from .schema import SemanticDeduplicator


def save_checkpoint(
    checkpoint_path: str,
    processed_key_count: int,
    history: list[IterationRecord],
    deduplicator: SemanticDeduplicator,
) -> None:
    checkpoint = {
        "processed_key_count": int(processed_key_count),
        "history": list(history),
        "deduplicator": deduplicator.to_dict(),
    }
    with open(checkpoint_path, "w", encoding="utf-8") as file:
        json.dump(checkpoint, file, ensure_ascii=False, indent=2)
    LOGGER.info("[Checkpoint] 已保存到 %s（已处理 %d 个 unique key）",
                checkpoint_path, processed_key_count)


def load_checkpoint(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ── field_pmid_index: canonical → PMID 集合 ──────────────────


def save_field_pmid_index(
    path: str,
    field_pmid_index: dict[str, set[str]],
) -> None:
    serializable = {
        key: sorted(pmids)
        for key, pmids in sorted(field_pmid_index.items())
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(serializable, file, ensure_ascii=False, indent=2)
    total_pairs = sum(len(v) for v in field_pmid_index.values())
    LOGGER.info(
        "[FieldPmidIndex] 已保存到 %s（%d 个字段，%d 个 field-PMID 对）",
        path, len(field_pmid_index), total_pairs,
    )


def save_field_pmid_env_index(
    path: str,
    field_pmid_env_index: dict[str, dict[str, set[str]]],
) -> None:
    serializable = {
        key: {env: sorted(pmids) for env, pmids in sorted(env_map.items())}
        for key, env_map in sorted(field_pmid_env_index.items())
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(serializable, file, ensure_ascii=False, indent=2)
    total_pairs = sum(
        len(pmids)
        for env_map in field_pmid_env_index.values()
        for pmids in env_map.values()
    )
    LOGGER.info(
        "[FieldPmidEnvIndex] 已保存到 %s（%d 个字段，%d 个 field-env-PMID 对）",
        path, len(field_pmid_env_index), total_pairs,
    )
