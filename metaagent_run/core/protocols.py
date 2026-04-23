"""功能描述：定义客户端可替换的协议接口。

输入：
- 实现该协议的客户端对象。

输出：
- 提供统一的 `chat` 与 `chat_streaming_with_signals` 调用契约。

用法示例：
```python
from metaagent.protocols import LLMClientProtocol

async def run(client: LLMClientProtocol) -> None:
    ...
```

各参数说明：
- messages：对话消息数组。
- temperature_override：单次调用温度覆盖值。
- max_tokens_override：流式调用 token 覆盖值。
"""

from typing import Dict, List, Optional, Protocol, TypedDict


class StreamingResponse(TypedDict):
    text: str
    saw_done: bool
    finish_reason: Optional[str]


class LLMClientProtocol(Protocol):
    max_tokens: int

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature_override: Optional[float] = None,
    ) -> Optional[str]:
        ...

    async def chat_streaming_with_signals(
        self,
        messages: List[Dict[str, str]],
        temperature_override: Optional[float] = None,
        max_tokens_override: Optional[int] = None,
    ) -> Optional[StreamingResponse]:
        ...
