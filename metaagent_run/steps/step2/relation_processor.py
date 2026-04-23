import asyncio
from typing import Any, Optional

from pydantic import ValidationError
from tqdm import tqdm

from metaagent_run.core import (
    ContentFilterError,
    LLMClientProtocol,
    backoff_with_jitter,
    continue_json_until_ok,
    detect_truncation,
    extract_json_from_response,
    is_excluded_section,
    split_text_with_offsets,
)

from .config import RuntimeConfig
from .prompt_builder import build_prompt
from .schemas import JudgementItem, LLMRelationItem, LLMRelationOutput

# Default unknown result template for items where the model returns no data.
_UNKNOWN_TEMPLATE: dict[str, Any] = {
    "relation": "unknown",
    "accessions_found": [],
    "labels_found": [],
    "metadata_keys_found": [],
    "evidence_quote": "",
}


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        normalized = item.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _is_table_like(item: dict[str, Any]) -> bool:
    """Detect table-like entries that benefit from header sampling rather than chunking.

    Covers:
    - section_type == TABLE (Main text tables)
    - Structured data files: .xlsx, .xls, .csv, .tsv
    - Table-containing documents: source contains "table" with .docx or .pdf extension
    """
    source = str(item.get("source", "")).lower()
    section_type = str(item.get("section_type", "")).upper()

    if section_type == "TABLE":
        return True
    if any(ext in source for ext in (".xlsx", ".xls", ".csv", ".tsv")):
        return True
    if "table" in source and any(ext in source for ext in (".docx", ".pdf")):
        return True
    return False


def _prepare_chunks(
    raw_text: str,
    item: dict[str, Any],
    key_id: str,
    runtime_config: RuntimeConfig,
) -> list[tuple[str, int, int]]:
    """Decide whether to use header sampling or full chunking.

    - Table-like entries: always header sample (column headers carry all category info).
    - Non-table entries exceeding text_chunk_size: header sample (long supplementary
      pdf/docx/txt — essentially structured data that doesn't benefit from multi-chunk).
    - Normal paragraphs: full text via split_text_with_offsets (in practice always 1 chunk).
    """
    use_header_sample = False

    if _is_table_like(item):
        use_header_sample = True
    elif len(raw_text) > runtime_config.text_chunk_size:
        use_header_sample = True

    if use_header_sample:
        sampled = raw_text[:runtime_config.header_sample_size]
        tqdm.write(
            f"📋 [HeaderSample] {key_id}: {len(raw_text)} chars → {len(sampled)} chars"
        )
        return [(sampled, 0, len(sampled))]

    return split_text_with_offsets(
        raw_text,
        chunk_size=runtime_config.text_chunk_size,
        overlap=runtime_config.text_overlap,
    )


async def process_single_relation_item(
    client: LLMClientProtocol,
    item: dict[str, Any],
    key_id: str,
    runtime_config: RuntimeConfig,
) -> Optional[dict[str, Any]]:
    if not runtime_config.retry_temps:
        raise RuntimeError("Invalid runtime config: retry_temps is empty")

    raw_text = item.get("text", "")
    if not isinstance(raw_text, str):
        tqdm.write(f"⏭️ [Skipped] {key_id}: Text is not a string")
        return None

    section_type = item.get("section_type", "")
    if not isinstance(section_type, str):
        raise ValueError(
            f"Invalid section_type for {key_id}: expected string, got {type(section_type).__name__}"
        )

    if not raw_text or len(raw_text.strip()) < runtime_config.min_text_length:
        tqdm.write(f"⏭️ [Skipped] {key_id}: Text too short ({len(raw_text)} chars)")
        return None

    if is_excluded_section(section_type, runtime_config.excluded_section_types):
        return None

    chunk_entries = _prepare_chunks(raw_text, item, key_id, runtime_config)
    if not chunk_entries:
        return None

    # Relation priority: higher index = more informative
    _RELATION_PRIORITY = {
        "unknown": 0,
        "label-metadata": 1,
        "accession-metadata": 2,
        "accession-label": 3,
        "accession-label-metadata": 4,
    }

    all_accessions: list[str] = []
    all_labels: list[str] = []
    all_metadata_keys: list[str] = []

    best_relation: str = "unknown"
    best_evidence_quote: str = ""
    best_priority: int = 0
    last_error: Optional[str] = None
    any_chunk_succeeded = False

    for chunk_idx, (chunk_text, chunk_start, _chunk_end) in enumerate(chunk_entries):
        messages = build_prompt(
            input_text=chunk_text,
            prompt_version=runtime_config.relation_prompt_version,
        )

        for attempt in range(runtime_config.retry_times):
            temp = runtime_config.retry_temps[min(attempt, len(runtime_config.retry_temps) - 1)]

            response_text: Optional[str] = None
            try:
                resp = await client.chat_streaming_with_signals(messages, temperature_override=temp)
            except ContentFilterError:
                tqdm.write(f"⛔ [ContentFilter] {key_id}::chunk{chunk_idx}: 403 sensitive content, skipping")
                any_chunk_succeeded = True  # Mark as handled, don't raise later
                resp = None
                break

            if resp:
                candidate_text = resp.get("text")
                if isinstance(candidate_text, str):
                    response_text = candidate_text
                    trunc_status = detect_truncation(
                        response_text,
                        bool(resp.get("saw_done", False)),
                        resp.get("finish_reason"),
                        stop_sentinel=runtime_config.stop_sentinel,
                    )
                    if trunc_status != "ok":
                        try:
                            cont_text = await continue_json_until_ok(
                                client,
                                messages,
                                response_text,
                                client.max_tokens,
                                max_tokens_cap=runtime_config.max_tokens_cap,
                                stop_sentinel=runtime_config.stop_sentinel,
                                max_rounds=runtime_config.continuation_max_rounds,
                            )
                        except ContentFilterError:
                            cont_text = None
                        if cont_text:
                            response_text = cont_text

            if not response_text and attempt >= runtime_config.fallback_from_attempt:
                try:
                    response_text = await client.chat(messages, temperature_override=temp)
                except ContentFilterError:
                    tqdm.write(f"⛔ [ContentFilter] {key_id}::chunk{chunk_idx}: 403 on fallback, skipping")
                    any_chunk_succeeded = True
                    break

            if not response_text:
                last_error = "Network/API Error: No response received"
                delay = backoff_with_jitter(
                    attempt,
                    base=runtime_config.backoff_base,
                    cap=runtime_config.backoff_cap,
                )
                tqdm.write(
                    f"⚠️ [Retry {attempt + 1}/{runtime_config.retry_times}] {key_id}::chunk{chunk_idx}: "
                    f"{last_error} (Sleeping {delay:.1f}s)"
                )
                await asyncio.sleep(delay)
                continue

            parsed = extract_json_from_response(
                response_text,
                stop_sentinel=runtime_config.stop_sentinel,
            )
            if parsed is None:
                last_error = "JSON Parsing Failed"
                await asyncio.sleep(
                    backoff_with_jitter(
                        attempt,
                        base=runtime_config.backoff_base,
                        cap=runtime_config.backoff_cap,
                    )
                )
                continue

            if isinstance(parsed, dict):
                parsed = [parsed]

            # Handle empty array: model decided nothing is relevant (e.g. unknown)
            if isinstance(parsed, list) and len(parsed) == 0:
                tqdm.write(f"ℹ️ [EmptyResult] {key_id}::chunk{chunk_idx}: LLM returned [], treating as unknown")
                any_chunk_succeeded = True
                break

            try:
                validated_items = [
                    LLMRelationItem.model_validate(entry)
                    for entry in parsed
                    if isinstance(entry, dict)
                ]
                relation_val = LLMRelationOutput(results=validated_items)
                if relation_val.results:
                    result0 = relation_val.results[0]

                    # Accumulate entity annotations (union across all chunks)
                    all_accessions.extend(result0.accessions_found or [])
                    all_labels.extend(result0.labels_found or [])
                    all_metadata_keys.extend(result0.metadata_keys_found or [])

                    # Keep the highest priority relation and its evidence_quote
                    priority = _RELATION_PRIORITY.get(result0.relation, 0)
                    if priority > best_priority:
                        best_priority = priority
                        best_relation = result0.relation
                        best_evidence_quote = result0.evidence_quote or ""

                    any_chunk_succeeded = True
                    break  # This chunk succeeded, no more retries
                else:
                    # Parsed OK but no valid items — treat as unknown success
                    tqdm.write(f"ℹ️ [EmptyValidation] {key_id}::chunk{chunk_idx}: parsed but 0 valid items, treating as unknown")
                    any_chunk_succeeded = True
                    break
            except ValidationError as error:
                last_error = f"Pydantic Error: {error.json()}"

            await asyncio.sleep(
                backoff_with_jitter(
                    attempt,
                    base=runtime_config.backoff_base,
                    cap=runtime_config.backoff_cap,
                )
            )

    # If no chunk produced a valid result, raise error
    if not any_chunk_succeeded:
        raise RuntimeError(
            f"Item {key_id} failed: no chunk produced a valid relation after retries. "
            f"Last Reason: {last_error}"
        )

    final_item = JudgementItem(
        pmid=str(item.get("pmid", "unknown")),
        source=str(item.get("source", "unknown")),
        section_type=section_type,
        index=int(item.get("index", 0)),
        relation=best_relation,
        accessions_found=_dedupe_preserve_order(all_accessions),
        labels_found=_dedupe_preserve_order(all_labels),
        metadata_keys_found=_dedupe_preserve_order(all_metadata_keys),
        evidence_quote=best_evidence_quote,
    )
    return final_item.model_dump()


async def process_batch_relation_items(
    client: LLMClientProtocol,
    items: list[dict[str, Any]],
    key_ids: list[str],
    runtime_config: RuntimeConfig,
) -> list[Optional[dict[str, Any]]]:
    """Process a batch of paragraphs in one LLM call.

    For each item we use only the first chunk (to keep prompt size manageable).
    If the batch call fails entirely, we fall back to calling each item
    individually via process_single_relation_item().
    """
    if not items:
        return []

    # ---- 1. Prepare texts (first chunk of each item) ---- #
    texts: list[str] = []
    valid_indices: list[int] = []          # indices into `items` that have usable text
    results: list[Optional[dict[str, Any]]] = [None] * len(items)

    for idx, (item, key_id) in enumerate(zip(items, key_ids)):
        raw_text = item.get("text", "")
        if not isinstance(raw_text, str):
            tqdm.write(f"⏭️ [Batch-Skipped] {key_id}: Text is not a string")
            continue

        section_type = item.get("section_type", "")
        if isinstance(section_type, str) and is_excluded_section(
            section_type, runtime_config.excluded_section_types
        ):
            continue

        if not raw_text or len(raw_text.strip()) < runtime_config.min_text_length:
            tqdm.write(f"⏭️ [Batch-Skipped] {key_id}: Text too short ({len(raw_text)} chars)")
            continue

        chunk_entries = _prepare_chunks(raw_text, item, key_id, runtime_config)
        if not chunk_entries:
            continue

        texts.append(chunk_entries[0][0])   # first chunk text only
        valid_indices.append(idx)

    if not texts:
        return results

    # ---- 2. Build batch prompt & call LLM with retries ---- #
    from .prompt_builder import build_batch_prompt  # local import to avoid circular

    messages = build_batch_prompt(
        input_texts=texts,
        prompt_version=runtime_config.relation_prompt_version,
    )

    parsed_list: Optional[list] = None

    for attempt in range(runtime_config.retry_times):
        temp = runtime_config.retry_temps[min(attempt, len(runtime_config.retry_temps) - 1)]

        response_text: Optional[str] = None
        try:
            resp = await client.chat_streaming_with_signals(messages, temperature_override=temp)
        except ContentFilterError:
            tqdm.write(f"⛔ [ContentFilter] Batch call hit 403, falling back to individual calls")
            break

        if resp:
            candidate_text = resp.get("text")
            if isinstance(candidate_text, str):
                response_text = candidate_text
                trunc_status = detect_truncation(
                    response_text,
                    bool(resp.get("saw_done", False)),
                    resp.get("finish_reason"),
                    stop_sentinel=runtime_config.stop_sentinel,
                )
                if trunc_status != "ok":
                    try:
                        cont_text = await continue_json_until_ok(
                            client,
                            messages,
                            response_text,
                            client.max_tokens,
                            max_tokens_cap=runtime_config.max_tokens_cap,
                            stop_sentinel=runtime_config.stop_sentinel,
                            max_rounds=runtime_config.continuation_max_rounds,
                        )
                    except ContentFilterError:
                        cont_text = None
                    if cont_text:
                        response_text = cont_text

        if not response_text and attempt >= runtime_config.fallback_from_attempt:
            try:
                response_text = await client.chat(messages, temperature_override=temp)
            except ContentFilterError:
                tqdm.write(f"⛔ [ContentFilter] Batch fallback chat hit 403, falling back to individual calls")
                break

        if not response_text:
            delay = backoff_with_jitter(
                attempt,
                base=runtime_config.backoff_base,
                cap=runtime_config.backoff_cap,
            )
            tqdm.write(
                f"⚠️ [Batch Retry {attempt + 1}/{runtime_config.retry_times}] "
                f"No response (Sleeping {delay:.1f}s)"
            )
            await asyncio.sleep(delay)
            continue

        parsed = extract_json_from_response(
            response_text,
            stop_sentinel=runtime_config.stop_sentinel,
        )

        if parsed is None:
            await asyncio.sleep(
                backoff_with_jitter(
                    attempt,
                    base=runtime_config.backoff_base,
                    cap=runtime_config.backoff_cap,
                )
            )
            continue

        # Normalise: the model may return a list or a single dict
        if isinstance(parsed, dict):
            parsed = [parsed]

        if isinstance(parsed, list) and len(parsed) == len(texts):
            parsed_list = parsed
            break

        # Model returned fewer results than expected.
        # Pad with unknown defaults so we don't needlessly retry or fallback.
        if isinstance(parsed, list) and len(parsed) < len(texts):
            tqdm.write(
                f"ℹ️ [Batch Pad] Expected {len(texts)} results, got {len(parsed)}. "
                f"Padding {len(texts) - len(parsed)} missing items as unknown."
            )
            parsed_list = list(parsed) + [dict(_UNKNOWN_TEMPLATE) for _ in range(len(texts) - len(parsed))]
            break

        # Length exceeds expected — take first N
        if isinstance(parsed, list) and len(parsed) > len(texts):
            tqdm.write(
                f"⚠️ [Batch Trim] Expected {len(texts)} results, got {len(parsed)}. Trimming."
            )
            parsed_list = parsed[:len(texts)]
            break

        await asyncio.sleep(
            backoff_with_jitter(
                attempt,
                base=runtime_config.backoff_base,
                cap=runtime_config.backoff_cap,
            )
        )

    # ---- 3. If batch call succeeded, validate each result ---- #
    if parsed_list is not None:
        for slot, vi in enumerate(valid_indices):
            item = items[vi]
            key_id = key_ids[vi]
            entry = parsed_list[slot]
            if not isinstance(entry, dict):
                continue

            try:
                validated = LLMRelationItem.model_validate(entry)
            except ValidationError:
                continue

            section_type = item.get("section_type", "")
            final_item = JudgementItem(
                pmid=str(item.get("pmid", "unknown")),
                source=str(item.get("source", "unknown")),
                section_type=str(section_type),
                index=int(item.get("index", 0)),
                relation=validated.relation,
                accessions_found=validated.accessions_found or [],
                labels_found=validated.labels_found or [],
                metadata_keys_found=validated.metadata_keys_found or [],
                evidence_quote=validated.evidence_quote or "",
            )
            results[vi] = final_item.model_dump()

        return results

    # ---- 4. Fallback: call each item individually ---- #
    tqdm.write("⚠️ [Batch] Batch call failed, falling back to individual calls")
    for vi in valid_indices:
        try:
            result = await process_single_relation_item(
                client,
                items[vi],
                key_ids[vi],
                runtime_config,
            )
            results[vi] = result
        except Exception as exc:
            tqdm.write(f"⚠️ [Batch-Fallback] {key_ids[vi]} failed: {exc}")

    return results
