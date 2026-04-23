"""功能描述：提供重试退避与截断判定工具。

输入：
- `backoff_with_jitter(attempt, base, cap)`。
- `detect_truncation(text, saw_done, finish_reason, stop_sentinel)`。

输出：
- `backoff_with_jitter` 返回 `float` 秒级等待时长。
- `detect_truncation` 返回状态字符串：transport_error / length_truncation / ok / content_truncation。

用法示例：
```python
from metaagent.retry_utils import backoff_with_jitter, detect_truncation

delay = backoff_with_jitter(attempt=1)
status = detect_truncation(text, saw_done, finish_reason)
```

各参数说明：
- attempt：当前重试轮次（从 0 开始）。
- base：指数退避基数。
- cap：退避最大上限。
- text：当前累计响应文本。
- saw_done：是否收到流式结束标记 `[DONE]`。
- finish_reason：模型返回的结束原因，可为 None。
- stop_sentinel：解析时使用的停止符。
"""

import random
from typing import Optional

from .json_utils import extract_json_from_response

STOP_SENTINEL = "</json>"


def backoff_with_jitter(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
    delay = min(cap, base * (2.0 ** attempt))
    return random.uniform(delay, delay * 1.5)


def detect_truncation(
    text: str,
    saw_done: bool,
    finish_reason: Optional[str],
    stop_sentinel: str = STOP_SENTINEL,
) -> str:
    if not saw_done and finish_reason is None:
        return "transport_error"
    if finish_reason == "length":
        return "length_truncation"
    if extract_json_from_response(text, stop_sentinel=stop_sentinel) is not None:
        return "ok"
    return "content_truncation"
