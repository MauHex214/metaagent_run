import asyncio
from typing import Any, Dict, List, Optional

from pydantic import ValidationError
from tqdm import tqdm

from metaagent_run.core import (
    continue_json_until_ok,
    extract_json_from_response,
    LLMClientProtocol,
    backoff_with_jitter,
    detect_truncation,
)

from .config import RuntimeConfig, load_runtime_config
from .prompt_builder import build_prompt
from .schemas import ExtractionOutput, SampleEntry


def _sanitize_value(value: Any) -> Any:
    if value is None:
        return "Unknown"
    return value


def _sanitize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _sanitize_value(value) for key, value in item.items()}


def _normalize_identity_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _validate_identity_consistency(
    predicted: SampleEntry,
    input_item: Dict[str, Any],
) -> Optional[str]:
    expected_biosample_id = _normalize_identity_value(input_item.get("biosample_id"))
    expected_organism = _normalize_identity_value(input_item.get("organism"))

    got_biosample_id = _normalize_identity_value(predicted.biosample_id)
    got_organism = _normalize_identity_value(predicted.organism)

    if got_biosample_id != expected_biosample_id:
        return (
            "Identity mismatch: biosample_id "
            f"expected={expected_biosample_id!r}, got={got_biosample_id!r}"
        )
    if got_organism != expected_organism:
        return (
            "Identity mismatch: organism "
            f"expected={expected_organism!r}, got={got_organism!r}"
        )
    return None


def _select_result_by_identity(
    candidates: list[SampleEntry],
    input_item: Dict[str, Any],
) -> Optional[SampleEntry]:
    expected_biosample_id = _normalize_identity_value(input_item.get("biosample_id"))
    expected_organism = _normalize_identity_value(input_item.get("organism"))

    for candidate in candidates:
        candidate_biosample_id = _normalize_identity_value(candidate.biosample_id)
        candidate_organism = _normalize_identity_value(candidate.organism)
        if (
            candidate_biosample_id == expected_biosample_id
            and candidate_organism == expected_organism
        ):
            return candidate
    return None


async def process_single_item(
    client: LLMClientProtocol,
    item: Dict[str, Any],
    key_id: str,
    runtime_config: Optional[RuntimeConfig] = None,
) -> Dict[str, Any]:
    config = runtime_config or load_runtime_config()
    if not config.retry_temps:
        raise RuntimeError("Invalid runtime config: retry_temps is empty")

    raw_item = dict(item)
    clean_item = _sanitize_item(item)
    messages = build_prompt(clean_item, prompt_version=config.prompt_version)
    last_error: Optional[str] = None

    for attempt in range(config.retry_times):
        temp = config.retry_temps[min(attempt, len(config.retry_temps) - 1)]
        resp = await client.chat_streaming_with_signals(messages, temperature_override=temp)
        response_text: Optional[str] = None

        if resp:
            candidate_text = resp.get("text")
            if isinstance(candidate_text, str):
                response_text = candidate_text
                trunc_status = detect_truncation(
                    response_text,
                    bool(resp.get("saw_done", False)),
                    resp.get("finish_reason"),
                    stop_sentinel=config.stop_sentinel,
                )
                if trunc_status != "ok":
                    continued_text = await continue_json_until_ok(
                        client,
                        messages,
                        response_text,
                        client.max_tokens,
                        max_tokens_cap=config.max_tokens_cap,
                        stop_sentinel=config.stop_sentinel,
                        max_rounds=config.continuation_max_rounds,
                    )
                    if continued_text:
                        response_text = continued_text

        if not response_text and attempt >= config.fallback_from_attempt:
            response_text = await client.chat(messages, temperature_override=temp)

        if not response_text:
            last_error = "Network/API Error: No response received"
            delay = backoff_with_jitter(
                attempt,
                base=config.backoff_base,
                cap=config.backoff_cap,
            )
            tqdm.write(
                f"⚠️ [Retry {attempt + 1}/{config.retry_times}] {key_id}: "
                f"{last_error} (Sleeping {delay:.1f}s)"
            )
            await asyncio.sleep(delay)
            continue

        parsed = extract_json_from_response(
            response_text,
            stop_sentinel=config.stop_sentinel,
        )
        if parsed:
            if isinstance(parsed, dict):
                parsed = [parsed]

            if not isinstance(parsed, list):
                parsed = None

            if parsed is not None:
                parsed_dicts = [entry for entry in parsed if isinstance(entry, dict)]
                try:
                    validated_entries = [
                        SampleEntry.model_validate(entry) for entry in parsed_dicts
                    ]
                    validated = ExtractionOutput(results=validated_entries)
                    if validated.results:
                        selected_result: Optional[SampleEntry]
                        if len(validated.results) > 1:
                            selected_result = _select_result_by_identity(
                                validated.results,
                                raw_item,
                            )
                            if selected_result is None:
                                last_error = (
                                    "Multiple results returned but none matched input "
                                    "(biosample_id + organism)"
                                )
                                tqdm.write(
                                    f"⚠️ [Retry {attempt + 1}/{config.retry_times}] {key_id}: "
                                    f"{last_error}"
                                )
                                await asyncio.sleep(
                                    backoff_with_jitter(
                                        attempt,
                                        base=config.backoff_base,
                                        cap=config.backoff_cap,
                                    )
                                )
                                continue
                        else:
                            selected_result = validated.results[0]

                        consistency_error = _validate_identity_consistency(
                            selected_result,
                            raw_item,
                        )
                        if consistency_error is not None:
                            last_error = consistency_error
                            tqdm.write(
                                f"⚠️ [Retry {attempt + 1}/{config.retry_times}] {key_id}: "
                                f"{consistency_error}"
                            )
                            await asyncio.sleep(
                                backoff_with_jitter(
                                    attempt,
                                    base=config.backoff_base,
                                    cap=config.backoff_cap,
                                )
                            )
                            continue
                        return selected_result.model_dump()
                except ValidationError as error:
                    last_error = f"Pydantic Error: {error.json()}"
                    tqdm.write(
                        f"⚠️ [Retry {attempt + 1}/{config.retry_times}] {key_id}: "
                        f"Validation Failed ({last_error[:120]}...)"
                    )
            else:
                last_error = "JSON Parsing Failed"
                tqdm.write(
                    f"⚠️ [Retry {attempt + 1}/{config.retry_times}] {key_id}: "
                    "JSON Parsing Failed"
                )
        else:
            last_error = "JSON Parsing Failed"
            tqdm.write(
                f"⚠️ [Retry {attempt + 1}/{config.retry_times}] {key_id}: "
                "JSON Parsing Failed"
            )

        await asyncio.sleep(
            backoff_with_jitter(
                attempt,
                base=config.backoff_base,
                cap=config.backoff_cap,
            )
        )

    raise RuntimeError(
        f"Processing failed. Key={key_id}. Last Reason: {last_error}"
    )


async def process_batch_items(
    client: LLMClientProtocol,
    items: List[Dict[str, Any]],
    key_ids: List[str],
    runtime_config: Optional[RuntimeConfig] = None,
) -> List[Optional[Dict[str, Any]]]:
    """Process a batch of items in a single LLM call.

    Falls back to individual processing if the batch call fails.
    """
    config = runtime_config or load_runtime_config()
    if not config.retry_temps:
        raise RuntimeError("Invalid runtime config: retry_temps is empty")

    raw_items = [dict(item) for item in items]
    clean_items = [_sanitize_item(item) for item in items]
    messages = build_prompt(clean_items, prompt_version=config.prompt_version)
    last_error: Optional[str] = None

    for attempt in range(config.retry_times):
        temp = config.retry_temps[min(attempt, len(config.retry_temps) - 1)]
        resp = await client.chat_streaming_with_signals(messages, temperature_override=temp)
        response_text: Optional[str] = None

        if resp:
            candidate_text = resp.get("text")
            if isinstance(candidate_text, str):
                response_text = candidate_text
                trunc_status = detect_truncation(
                    response_text,
                    bool(resp.get("saw_done", False)),
                    resp.get("finish_reason"),
                    stop_sentinel=config.stop_sentinel,
                )
                if trunc_status != "ok":
                    continued_text = await continue_json_until_ok(
                        client,
                        messages,
                        response_text,
                        client.max_tokens,
                        max_tokens_cap=config.max_tokens_cap,
                        stop_sentinel=config.stop_sentinel,
                        max_rounds=config.continuation_max_rounds,
                    )
                    if continued_text:
                        response_text = continued_text

        if not response_text and attempt >= config.fallback_from_attempt:
            response_text = await client.chat(messages, temperature_override=temp)

        if not response_text:
            last_error = "Network/API Error: No response received"
            delay = backoff_with_jitter(
                attempt,
                base=config.backoff_base,
                cap=config.backoff_cap,
            )
            batch_keys = ", ".join(key_ids)
            tqdm.write(
                f"\u26a0\ufe0f [Batch Retry {attempt + 1}/{config.retry_times}] "
                f"keys=[{batch_keys}]: {last_error} (Sleeping {delay:.1f}s)"
            )
            await asyncio.sleep(delay)
            continue

        parsed = extract_json_from_response(
            response_text,
            stop_sentinel=config.stop_sentinel,
        )
        if parsed:
            if isinstance(parsed, dict):
                parsed = [parsed]

            if not isinstance(parsed, list):
                parsed = None

            if parsed is not None:
                parsed_dicts = [entry for entry in parsed if isinstance(entry, dict)]
                try:
                    validated_entries = [
                        SampleEntry.model_validate(entry) for entry in parsed_dicts
                    ]

                    # Match each input item to a validated result by identity
                    results: List[Optional[Dict[str, Any]]] = []
                    all_ok = True
                    for raw_item, key_id in zip(raw_items, key_ids):
                        matched = _select_result_by_identity(validated_entries, raw_item)
                        if matched is not None:
                            consistency_error = _validate_identity_consistency(matched, raw_item)
                            if consistency_error is None:
                                results.append(matched.model_dump())
                            else:
                                tqdm.write(
                                    f"\u26a0\ufe0f [Batch] {key_id}: {consistency_error}"
                                )
                                results.append(None)
                                all_ok = False
                        else:
                            tqdm.write(
                                f"\u26a0\ufe0f [Batch] {key_id}: No matching result in batch response"
                            )
                            results.append(None)
                            all_ok = False

                    if all_ok:
                        return results

                    # On final attempt, fall back to individual processing for failed items
                    if attempt == config.retry_times - 1:
                        for idx in range(len(results)):
                            if results[idx] is None:
                                try:
                                    single_result = await process_single_item(
                                        client, raw_items[idx], key_ids[idx], runtime_config=config
                                    )
                                    results[idx] = single_result
                                except Exception as single_err:
                                    tqdm.write(
                                        f"\u26a0\ufe0f [Fallback] {key_ids[idx]}: {single_err}"
                                    )
                                    results[idx] = None
                        return results

                    last_error = "Some items in batch did not match"
                except ValidationError as error:
                    last_error = f"Pydantic Error: {error.json()}"
                    tqdm.write(
                        f"\u26a0\ufe0f [Batch Retry {attempt + 1}/{config.retry_times}]: "
                        f"Validation Failed ({last_error[:120]}...)"
                    )
            else:
                last_error = "JSON Parsing Failed"
                tqdm.write(
                    f"\u26a0\ufe0f [Batch Retry {attempt + 1}/{config.retry_times}]: "
                    "JSON Parsing Failed"
                )
        else:
            last_error = "JSON Parsing Failed"
            tqdm.write(
                f"\u26a0\ufe0f [Batch Retry {attempt + 1}/{config.retry_times}]: "
                "JSON Parsing Failed"
            )

        await asyncio.sleep(
            backoff_with_jitter(
                attempt,
                base=config.backoff_base,
                cap=config.backoff_cap,
            )
        )

    # Batch completely failed - fall back to individual processing
    tqdm.write(
        f"\u26a0\ufe0f Batch failed after {config.retry_times} attempts. "
        f"Falling back to individual processing for {len(items)} items."
    )
    results = []
    for raw_item, key_id in zip(raw_items, key_ids):
        try:
            single_result = await process_single_item(
                client, raw_item, key_id, runtime_config=config
            )
            results.append(single_result)
        except Exception as single_err:
            tqdm.write(f"\u26a0\ufe0f [Fallback] {key_id}: {single_err}")
            results.append(None)
    return results
