"""功能描述：处理流式输出截断后的续写补全。

输入：
- `continue_json_until_ok(client, original_messages, partial_text, start_max_tokens, max_tokens_cap, stop_sentinel)`。

输出：
- 返回补全后的文本字符串；若补全失败则返回当前累计文本。

用法示例：
```python
from metaagent.continuation import continue_json_until_ok

text = await continue_json_until_ok(client, messages, partial_text, max_tokens)
```

各参数说明：
- client：支持 `chat_streaming_with_signals` 的客户端实例。
- original_messages：原始请求消息数组。
- partial_text：已生成的部分文本。
- start_max_tokens：当前续写起始 token 上限。
- max_tokens_cap：续写 token 最大上限。
- stop_sentinel：续写完成判定时使用的停止符。
"""

from typing import Dict, List

from .protocols import LLMClientProtocol
from .retry_utils import detect_truncation

MAX_TOKENS_CAP = 1024
STOP_SENTINEL = "</json>"


async def continue_json_until_ok(
    client: LLMClientProtocol,
    original_messages: List[Dict[str, str]],
    partial_text: str,
    start_max_tokens: int,
    max_tokens_cap: int = MAX_TOKENS_CAP,
    stop_sentinel: str = STOP_SENTINEL,
    max_rounds: int = 2,
) -> str:
    max_tokens = start_max_tokens
    text = partial_text
    for _ in range(max_rounds):
        tail = text[-500:]
        cont_msgs = [
            original_messages[0],
            {"role": "user", "content": "Keep generating."},
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": (
                    "Continue EXACTLY from where you stopped.\n"
                    f"Last chars: {tail}"
                ),
            },
        ]
        max_tokens = min(int(max_tokens * 1.5), max_tokens_cap)
        resp = await client.chat_streaming_with_signals(
            cont_msgs, max_tokens_override=max_tokens
        )
        if not resp:
            break
        text += resp["text"]
        if (
            detect_truncation(
                text,
                resp["saw_done"],
                resp["finish_reason"],
                stop_sentinel=stop_sentinel,
            )
            == "ok"
        ):
            return text
    return text
