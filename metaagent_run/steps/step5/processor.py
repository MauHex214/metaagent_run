"""Step5 metadata normalization (Phase C2).

Parses raw "field: value" strings emitted by Phase B1+B2 (table parser) and
Phase B3 (LLM extractor) into typed NormalizedMetadataItem records. No MIxS
slot mapping — that responsibility moved to a separate post-pipeline step
(see env_field_pipeline phase8). The earlier `llm_normalize_fields` (Phase C1)
is gone for the same reason.
"""

from typing import List, Set

from .schemas import NormalizedMetadataItem


def _normalize_metadata_list(raw_items: List[str]) -> List[NormalizedMetadataItem]:
    """Parse 'field: value' strings, optionally prefixed with 'source||', and
    dedup on (source, field_lower, value_lower)."""
    results: List[NormalizedMetadataItem] = []
    seen: Set[tuple] = set()

    for item in raw_items:
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

        dedup_key = (source, key.lower(), value.lower())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        results.append(NormalizedMetadataItem(
            raw_field=key,
            value=value,
            source=source,
        ))

    return results
