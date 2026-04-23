from collections.abc import Sequence
from typing import Optional

from pydantic import BaseModel, Field


class LLMEvidenceLocation(BaseModel):
    start_char: int = Field(...)
    end_char: int = Field(...)


class EvidenceSpan(BaseModel):
    quote: str = Field(...)
    start_char: int = Field(...)
    end_char: int = Field(...)
    coordinate_space: str = Field(default="document")


LIGHT_CORRECTION_MAX_CHARS = 50
_SENTENCE_END_CHARS = {".", "!", "?", ";", "。", "！", "？", "；", "\n"}


def build_not_provided_evidence() -> EvidenceSpan:
    return EvidenceSpan(
        quote="Not provided",
        start_char=-1,
        end_char=-1,
        coordinate_space="document",
    )


def validate_evidence_location_lightweight(
    has_positive: bool,
    evidence: LLMEvidenceLocation,
    text: str,
    min_chars: int,
    max_chars: int,
    max_correction_chars: int = LIGHT_CORRECTION_MAX_CHARS,
) -> Optional[LLMEvidenceLocation]:
    if not has_positive or not text:
        return None

    text_len = len(text)
    raw_start = evidence.start_char
    raw_end = evidence.end_char

    out_of_range_total = (
        max(0, -raw_start)
        + max(0, raw_start - text_len)
        + max(0, -raw_end)
        + max(0, raw_end - text_len)
    )
    if out_of_range_total > max_correction_chars:
        return None

    start_char = min(max(raw_start, 0), text_len)
    end_char = min(max(raw_end, 0), text_len)
    if end_char <= start_char:
        return None

    span_len = end_char - start_char
    if span_len < min_chars or span_len > max_chars:
        return None

    return LLMEvidenceLocation(start_char=start_char, end_char=end_char)


def build_document_evidence(
    raw_text: str,
    start_char: int,
    end_char: int,
    max_output_chars: int,
) -> EvidenceSpan:
    quote = raw_text[start_char:end_char][:max_output_chars]
    return EvidenceSpan(
        quote=quote,
        start_char=start_char,
        end_char=end_char,
        coordinate_space="document",
    )


def _iter_sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for index, char in enumerate(text):
        if char not in _SENTENCE_END_CHARS:
            continue
        end = index + 1
        if end > start and text[start:end].strip():
            spans.append((start, end))
        start = end

    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))
    if not spans and text:
        spans.append((0, len(text)))
    return spans


def find_two_stage_evidence_location(
    text: str,
    query_terms: Sequence[str],
    min_chars: int,
    max_chars: int,
    sentence_window_chars: int = 240,
) -> Optional[LLMEvidenceLocation]:
    if not text or max_chars <= 0:
        return None

    cleaned_terms: list[str] = []
    for term in query_terms:
        token = term.strip()
        if token and token not in cleaned_terms:
            cleaned_terms.append(token)
    if not cleaned_terms:
        return None

    text_len = len(text)
    lower_text = text.lower()
    lower_terms = [term.lower() for term in cleaned_terms]
    spans = _iter_sentence_spans(text)
    if not spans:
        return None

    anchor_positions = [lower_text.find(term) for term in lower_terms]
    anchor_positions = [pos for pos in anchor_positions if pos >= 0]
    anchor = min(anchor_positions) if anchor_positions else text_len // 2

    best_span: Optional[tuple[int, int]] = None
    best_key: Optional[tuple[int, int, int]] = None
    for span_start, span_end in spans:
        sentence_lower = lower_text[span_start:span_end]
        hit_count = sum(1 for term in lower_terms if term in sentence_lower)
        if hit_count <= 0:
            continue

        center = (span_start + span_end) // 2
        distance = abs(center - anchor)
        score_key = (hit_count, -distance, -(span_end - span_start))
        if best_key is None or score_key > best_key:
            best_key = score_key
            best_span = (span_start, span_end)

    if best_span is None:
        if anchor_positions:
            pos = anchor_positions[0]
            best_span = (pos, min(text_len, pos + max(min_chars, 1)))
        else:
            best_span = (0, min(text_len, max(min_chars, 1)))

    span_start, span_end = best_span
    if sentence_window_chars > 0:
        center = (span_start + span_end) // 2
        half = max(1, sentence_window_chars // 2)
        span_start = max(0, center - half)
        span_end = min(text_len, center + half)

    span_len = span_end - span_start
    if span_len > max_chars:
        center = (span_start + span_end) // 2
        half = max_chars // 2
        span_start = max(0, center - half)
        span_end = min(text_len, span_start + max_chars)
        span_start = max(0, span_end - max_chars)

    span_len = span_end - span_start
    if span_len < min_chars:
        need = min_chars - span_len
        left_expand = need // 2
        right_expand = need - left_expand
        span_start = max(0, span_start - left_expand)
        span_end = min(text_len, span_end + right_expand)

        span_len = span_end - span_start
        if span_len < min_chars and text_len >= min_chars:
            span_start = max(0, span_end - min_chars)
            span_end = min(text_len, span_start + min_chars)
            span_start = max(0, span_end - min_chars)

    span_len = span_end - span_start
    if span_len < min_chars or span_len > max_chars:
        return None

    return LLMEvidenceLocation(start_char=span_start, end_char=span_end)
