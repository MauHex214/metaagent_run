import json
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

STOP_SENTINEL = "</json>"

ParsedJson = Optional[Union[Dict[str, Any], List[Dict[str, Any]], List[Any]]]


def clean_json_text(text: str) -> str:
    if not text:
        return ""
    replacements = {"“": '"', "”": '"', "‘": "'", "’": "'", "，": ","}
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = re.sub(r"\}\s*\{", "},{", text)
    text = re.sub(r"\]\s*\[", "],[", text)
    return text


def _iter_top_level_json_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    stack: List[str] = []
    start_index: Optional[int] = None
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char in "[{":
            if not stack:
                start_index = index
            stack.append(char)
            continue

        if char in "]}":
            if not stack:
                continue
            top = stack[-1]
            if (top == "[" and char == "]") or (top == "{" and char == "}"):
                stack.pop()
                if not stack and start_index is not None:
                    spans.append((start_index, index + 1))
                    start_index = None

    return spans


def _strip_markdown_json_blocks(text: str) -> List[str]:
    candidates: List[str] = []
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE):
        content = match.group(1).strip()
        if content:
            candidates.append(content)
    return candidates


def _iter_balanced_blocks(s: str, open_ch: str, close_ch: str):
    n = len(s)
    for start in range(n):
        if s[start] != open_ch:
            continue
        depth = 0
        for i in range(start, n):
            c = s[i]
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    yield s[start : i + 1]
                    break


def _truncate_trailing_explanation(text: str) -> str:
    spans = _iter_top_level_json_spans(text)
    if not spans:
        return text
    start, end = spans[0]
    return text[start:end].strip()


def _remove_trailing_commas_token_aware(text: str) -> str:
    chars = list(text)
    keep = [True] * len(chars)
    in_string = False
    escaped = False

    for i, char in enumerate(chars):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char != ",":
            continue

        j = i + 1
        while j < len(chars) and chars[j].isspace():
            j += 1
        if j < len(chars) and chars[j] in "]}":
            keep[i] = False

    return "".join(ch for idx, ch in enumerate(chars) if keep[idx])


def _fix_bare_newlines_in_strings(text: str) -> str:
    result: List[str] = []
    in_string = False
    escaped = False

    for char in text:
        if in_string:
            if escaped:
                result.append(char)
                escaped = False
                continue
            if char == "\\":
                result.append(char)
                escaped = True
                continue
            if char == '"':
                result.append(char)
                in_string = False
                continue
            if char == "\n":
                result.append("\\n")
                continue
            if char == "\r":
                continue
            result.append(char)
            continue

        if char == '"':
            in_string = True
        result.append(char)

    return "".join(result)


def _auto_close_structures(text: str) -> str:
    stack: List[str] = []
    in_string = False
    escaped = False

    for char in text:
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char in "[{":
            stack.append(char)
            continue
        if char == "]" and stack and stack[-1] == "[":
            stack.pop()
            continue
        if char == "}" and stack and stack[-1] == "{":
            stack.pop()

    suffix = "".join("]" if token == "[" else "}" for token in reversed(stack))
    return text + suffix


def _merge_adjacent_json_objects(text: str) -> str:
    spans = _iter_top_level_json_spans(text)
    if len(spans) < 2:
        return text

    segments: List[str] = []
    for start, end in spans:
        segment = text[start:end].strip()
        if not segment or not segment.startswith("{"):
            return text
        segments.append(segment)
    return "[" + ",".join(segments) + "]"


def _count_target_keys_in_text(text: str, target_keys: Sequence[str]) -> int:
    lowered = text.lower()
    score = 0
    for key in target_keys:
        normalized = key.lower().strip()
        if not normalized:
            continue
        if f'"{normalized}"' in lowered or normalized in lowered:
            score += 1
    return score


def _count_target_keys_in_parsed(parsed: Any, target_keys: Sequence[str]) -> int:
    if isinstance(parsed, dict):
        return sum(1 for key in target_keys if key in parsed)
    if isinstance(parsed, list):
        best = 0
        for item in parsed:
            if isinstance(item, dict):
                best = max(best, _count_target_keys_in_parsed(item, target_keys))
        return best
    return 0


def _try_load_json(value: str) -> Optional[Any]:
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _parse_candidate_with_repair(content: str, enable_p0: bool, enable_p1: bool) -> Optional[Any]:
    raw = content.strip()
    if not raw:
        return None

    parsed = _try_load_json(raw)
    if parsed is not None:
        return parsed

    current = raw
    if enable_p0:
        current = _truncate_trailing_explanation(current)
        current = _remove_trailing_commas_token_aware(current)
        parsed = _try_load_json(current)
        if parsed is not None:
            return parsed

    if enable_p1:
        current = _fix_bare_newlines_in_strings(current)
        current = _merge_adjacent_json_objects(current)
        current = _auto_close_structures(current)
        current = _remove_trailing_commas_token_aware(current)
        parsed = _try_load_json(current)
        if parsed is not None:
            return parsed

    return None


def extract_json_from_response_with_repair(
    response: str,
    stop_sentinel: str = STOP_SENTINEL,
    target_keys: Optional[Sequence[str]] = None,
    enable_p0: bool = True,
    enable_p1: bool = False,
) -> ParsedJson:
    if not response:
        return None
    if stop_sentinel and stop_sentinel in response:
        response = response.split(stop_sentinel)[0]

    keys = tuple(k.strip() for k in (target_keys or ()) if k and k.strip())

    candidates: List[str] = []
    candidates.extend(_strip_markdown_json_blocks(response))

    merged_objects = _merge_adjacent_json_objects(response)
    if merged_objects.strip() and merged_objects.strip() != response.strip():
        candidates.append(merged_objects.strip())

    for start, end in _iter_top_level_json_spans(response):
        block = response[start:end].strip()
        if block:
            candidates.append(block)

    for block in _iter_balanced_blocks(response, "[", "]"):
        candidate = block.strip()
        if candidate:
            candidates.append(candidate)
    for block in _iter_balanced_blocks(response, "{", "}"):
        candidate = block.strip()
        if candidate:
            candidates.append(candidate)

    if response.strip():
        candidates.append(response.strip())

    deduped: List[str] = []
    seen: Set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)

    if keys:
        ranked = sorted(
            deduped,
            key=lambda candidate: (_count_target_keys_in_text(candidate, keys), len(candidate)),
            reverse=True,
        )
    else:
        ranked = deduped

    best_parsed: Optional[Any] = None
    best_score = -1

    for candidate in ranked:
        parsed = _parse_candidate_with_repair(candidate, enable_p0=enable_p0, enable_p1=enable_p1)
        if parsed is None:
            continue

        if not keys:
            if isinstance(parsed, dict):
                return [parsed]
            return parsed

        score = _count_target_keys_in_parsed(parsed, keys)
        if score > best_score:
            best_score = score
            best_parsed = parsed
        if score == len(keys):
            break

    if isinstance(best_parsed, dict):
        return [best_parsed]
    return best_parsed


def extract_json_from_response(response: str, stop_sentinel: str = STOP_SENTINEL) -> ParsedJson:
    if not response:
        return None
    if stop_sentinel and stop_sentinel in response:
        response = response.split(stop_sentinel)[0]

    cleaned = clean_json_text(response)

    block_match = re.search(r"```json\s*([\s\S]+?)\s*```", cleaned)
    if block_match:
        try:
            return json.loads(block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    array_match = re.search(r"\[[\s\S]*\]", cleaned)
    if array_match:
        content = array_match.group(0)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                return json.loads(re.sub(r",\s*\]", "]", content))
            except json.JSONDecodeError:
                pass

    obj_match = re.search(r"\{[\s\S]*\}", cleaned)
    if obj_match:
        content = obj_match.group(0)
        try:
            return [json.loads(content)]
        except json.JSONDecodeError:
            pass

    return None
