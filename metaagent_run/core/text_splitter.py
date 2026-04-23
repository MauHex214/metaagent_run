"""功能描述：按阈值切分长文本并保留重叠区域。

输入：
- `split_text_with_offsets(text)`，原始长文本字符串。

输出：
- `List[Tuple[str, int, int]]`，按策略切分后的文本块列表（含偏移量）。

各参数说明：
- text：待切分文本。
- TEXT_CHUNK_SIZE：每块最大长度。
- TEXT_OVERLAP：相邻块重叠长度，用于保留上下文。
"""

import re
from typing import List, Tuple

TEXT_CHUNK_SIZE = 10000
TEXT_OVERLAP = 200


def split_text_with_offsets(
    text: str, chunk_size: int = TEXT_CHUNK_SIZE, overlap: int = TEXT_OVERLAP
) -> List[Tuple[str, int, int]]:
    """Split text into overlapping chunks with character offset tracking.

    Returns list of (chunk_text, start_offset, end_offset) tuples.
    """
    if not text:
        return []
    if len(text) <= chunk_size:
        return [(text, 0, len(text))]

    chunks: List[Tuple[str, int, int]] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            lookahead = text[end : end + 100]
            match = re.search(r"[。\.\n]", lookahead)
            if match:
                end += match.end()
        chunks.append((text[start:end], start, end))
        if end >= text_length:
            break
        start = max(end - overlap, 0)
    return chunks
