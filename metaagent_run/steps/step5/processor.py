"""Step5_test 核心处理逻辑（精简版）。

仅保留 orchestrator 实际调用的两个函数:
- _normalize_metadata_list: Phase C2 metadata 归一化
- llm_normalize_fields: Phase C1 LLM fallback 字段映射
"""

import asyncio
from typing import Dict, List, Set

from tqdm import tqdm

from metaagent_run.core import (
    LLMClientProtocol,
    backoff_with_jitter,
    detect_truncation,
    continue_json_until_ok,
    extract_json_from_response_with_repair,
)

from .config import RuntimeConfig
from .schemas import NormalizedMetadataItem


# ═══════════════════════════════════════════════════════════
#  Phase C2 — Metadata 归一化
# ═══════════════════════════════════════════════════════════

def _normalize_metadata_list(
    raw_items: List[str],
    field_to_mixs: Dict[str, str],
) -> List[NormalizedMetadataItem]:
    """Normalize metadata list. Supports source-tagged format: 'source||field: value'."""
    results: List[NormalizedMetadataItem] = []
    seen: Set[str] = set()

    for item in raw_items:
        # Parse optional source tag: "table_parse||field: value"
        source = ""
        payload = item
        if "||" in item:
            source, _, payload = item.partition("||")
            source = source.strip()

        key, _, value = payload.partition(":")
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue

        # Deduplicate exact same source+field+value
        dedup_key = (source, key.lower(), value.lower())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        mixs_slot = field_to_mixs.get(key.lower())

        results.append(NormalizedMetadataItem(
            mixs_slot=mixs_slot,
            raw_field=key,
            value=value,
            source=source,
        ))

    return results


# ═══════════════════════════════════════════════════════════
#  Phase C1 — LLM 字段规范化 (fallback)
# ═══════════════════════════════════════════════════════════

async def llm_normalize_fields(
    client: LLMClientProtocol,
    raw_fields: Set[str],
    mixs_slots: List[str],
    config: RuntimeConfig,
) -> Dict[str, str]:
    """调用 LLM 将 raw_field 批量映射到 MIxS slot。

    仅用于 field_to_mixs 程序化展开未覆盖的 raw_field (fallback)。
    返回 {raw_field.lower(): mixs_slot} 字典，UNMAPPED 的不包含。
    """
    if not raw_fields or not mixs_slots:
        return {}

    from .prompt_builder import _load_system_prompt
    import json as _json

    system_prompt = _load_system_prompt(config.prompt_field_norm)

    payload = _json.dumps({
        "raw_fields": sorted(raw_fields),
        "mixs_slots": sorted(set(mixs_slots)),
    }, ensure_ascii=False, indent=2)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload},
    ]

    for attempt in range(config.retry_times):
        temp = config.retry_temps[min(attempt, len(config.retry_temps) - 1)]

        resp = await client.chat_streaming_with_signals(messages, temperature_override=temp)
        response_text = None
        if resp:
            text = resp.get("text")
            if isinstance(text, str) and text.strip():
                status = detect_truncation(
                    text, bool(resp.get("saw_done", False)),
                    resp.get("finish_reason"),
                    stop_sentinel=config.stop_sentinel,
                )
                if status != "ok":
                    continued = await continue_json_until_ok(
                        client, messages, text, client.max_tokens,
                        max_tokens_cap=config.max_tokens_cap,
                        stop_sentinel=config.stop_sentinel,
                        max_rounds=config.continuation_max_rounds,
                    )
                    response_text = continued if continued else text
                else:
                    response_text = text

        if not response_text:
            response_text = await client.chat(messages, temperature_override=temp)

        if not response_text:
            tqdm.write("⚠️ Field norm attempt %d: no response" % (attempt + 1))
            await asyncio.sleep(backoff_with_jitter(
                attempt, base=config.backoff_base, cap=config.backoff_cap,
            ))
            continue

        parsed = extract_json_from_response_with_repair(
            response_text,
            stop_sentinel=config.stop_sentinel,
            target_keys=("field", "mixs_slot"),
            enable_p0=True, enable_p1=False,
        )

        mapping_list = []
        if isinstance(parsed, list):
            mapping_list = [e for e in parsed if isinstance(e, dict)]
        elif isinstance(parsed, dict):
            mapping_list = [parsed]

        if not mapping_list:
            tqdm.write("⚠️ Field norm attempt %d: parse failed" % (attempt + 1))
            await asyncio.sleep(backoff_with_jitter(
                attempt, base=config.backoff_base, cap=config.backoff_cap,
            ))
            continue

        result: Dict[str, str] = {}
        for entry in mapping_list:
            field = str(entry.get("field", "")).strip()
            slot = str(entry.get("mixs_slot", "")).strip()
            if field and slot and slot != "UNMAPPED":
                result[field.lower()] = slot

        tqdm.write("✅ Field normalization: %d/%d fields mapped to MIxS slots" % (
            len(result), len(raw_fields),
        ))
        return result

    tqdm.write("❌ Field normalization failed after %d attempts" % config.retry_times)
    return {}
