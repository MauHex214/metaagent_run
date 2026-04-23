"""功能描述：封装本地异步 LLM 客户端（非流式与流式）。

输入：
- 构造参数：base_url、model、temperature、max_tokens、api_key、stop_sentinel。
- 调用参数：messages、temperature_override、max_tokens_override。

输出：
- `chat` 返回 `Optional[str]`。
- `chat_streaming_with_signals` 返回 `Optional[Dict[str, Any]]`，包含 text/saw_done/finish_reason。

用法示例：
```python
from metaagent.llm_client import AsyncLocalModelClient

async with AsyncLocalModelClient(base_url=url, model=name, api_key=key) as client:
    text = await client.chat(messages)
```

各参数说明：
- base_url：模型服务地址。
- model：模型标识。
- temperature：默认采样温度。
- max_tokens：默认最大生成 token。
- api_key：鉴权令牌，可为空。
- stop_sentinel：终止生成标记。
- messages：对话消息数组。
- temperature_override：单次调用温度覆盖值。
- max_tokens_override：流式调用 token 覆盖值。
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .protocols import StreamingResponse

STOP_SENTINEL = "</json>"


class ContentFilterError(Exception):
    """Raised when the API rejects input due to content filtering (HTTP 403)."""
    pass
LOGGER = logging.getLogger(__name__)

# Total wall-clock timeout (seconds) for a single streaming call.
# Prevents indefinite hang when the server sends keepalive empty lines
# that reset httpx's per-read timeout without delivering useful content.
STREAMING_TOTAL_TIMEOUT: float = 90.0


class AsyncLocalModelClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        stop_sentinel: str = STOP_SENTINEL,
        enable_thinking: bool = False,      # 关闭Qwen3.5模型的thinking功能时使用
        api_style: str = "openai",
        azure_api_version: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        auth_mode: str = "bearer",
        streaming_total_timeout: float = STREAMING_TOTAL_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None
        self.stop = [stop_sentinel]
        self.enable_thinking = enable_thinking      # 关闭Qwen3.5模型的thinking功能时使用
        self.api_style = api_style.lower()
        self.azure_api_version = azure_api_version
        self.azure_deployment = azure_deployment
        self.auth_mode = auth_mode.lower()
        self.streaming_total_timeout = streaming_total_timeout

    def _build_headers(self) -> Optional[Dict[str, str]]:
        if not self.api_key:
            return None
        if self.auth_mode == "api-key":
            return {"api-key": self.api_key}
        return {"Authorization": f"Bearer {self.api_key}"}

    def _build_chat_endpoint(self) -> Tuple[str, Optional[Dict[str, str]]]:
        if self.api_style == "azure":
            deployment = (self.azure_deployment or self.model).strip()
            if not deployment:
                return "/openai/deployments/chat/completions", None
            params = (
                {"api-version": self.azure_api_version}
                if self.azure_api_version
                else None
            )
            return f"/openai/deployments/{deployment}/chat/completions", params
        return "/v1/chat/completions", None

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        temperature_override: Optional[float],
        max_tokens: int,
        stream: bool,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": (
                temperature_override
                if temperature_override is not None
                else self.temperature
            ),
            "max_tokens": max_tokens,
            "stop": self.stop,
        }
        if stream:
            payload["stream"] = True

        if self.api_style != "azure":
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking
            }
        return payload

    @staticmethod
    def _response_excerpt(resp: httpx.Response) -> str:
        try:
            text = resp.text
        except Exception:
            return "<unable-to-read-response-body>"
        if not text:
            return "<empty-body>"
        text = text.strip().replace("\n", " ")
        return text[:500]

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._build_headers(),
            timeout=httpx.Timeout(connect=30.0, read=120.0, write=60.0, pool=300.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature_override: Optional[float] = None,
        max_retries: int = 5,
        base_backoff: float = 5.0,
    ) -> Optional[str]:
        client = self._client
        if client is None:
            return None

        payload = self._build_payload(
            messages=messages,
            temperature_override=temperature_override,
            max_tokens=self.max_tokens,
            stream=False,
        )
        for attempt in range(max_retries):
            try:
                endpoint, params = self._build_chat_endpoint()
                resp = await client.post(endpoint, json=payload, params=params)
                if resp.status_code == 429:
                    wait_time = base_backoff * (2 ** attempt)
                    try:
                        retry_after = float(resp.headers.get("Retry-After", wait_time))
                        wait_time = max(wait_time, retry_after)
                    except (TypeError, ValueError):
                        pass
                    LOGGER.warning(
                        "触发限流(429)，第 %d/%d 次重试，等待 %.1f 秒...",
                        attempt + 1,
                        max_retries,
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                if resp.status_code == 403:
                    LOGGER.warning(
                        "Content filtered (403), skipping: %s",
                        self._response_excerpt(resp),
                    )
                    raise ContentFilterError(self._response_excerpt(resp))
                if resp.status_code != 200:
                    LOGGER.error(
                        "LLM request failed: status=%s endpoint=%s params=%s body=%s",
                        resp.status_code,
                        endpoint,
                        params,
                        self._response_excerpt(resp),
                    )
                    return None
                return resp.json()["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                LOGGER.exception("LLM request exception: %s", exc)
                return None

        LOGGER.error("已达最大重试次数(%d)，放弃本次请求", max_retries)
        return None

    async def _stream_sse(
        self,
        payload: Dict[str, Any],
    ) -> Optional[StreamingResponse]:
        """Internal: execute the streaming HTTP call and parse SSE lines."""
        client = self._client
        if client is None:
            return None

        endpoint, params = self._build_chat_endpoint()
        async with client.stream("POST", endpoint, json=payload, params=params) as resp:
            if resp.status_code == 403:
                body = await resp.aread()
                excerpt = body.decode("utf-8", errors="ignore").strip().replace("\n", " ")
                LOGGER.warning("Content filtered (403), skipping: %s", excerpt[:500] if excerpt else "<empty-body>")
                raise ContentFilterError(excerpt[:500] if excerpt else "content_filter")
            if resp.status_code != 200:
                body = await resp.aread()
                excerpt = body.decode("utf-8", errors="ignore").strip().replace("\n", " ")
                LOGGER.error(
                    "LLM stream request failed: status=%s endpoint=%s params=%s body=%s",
                    resp.status_code,
                    endpoint,
                    params,
                    excerpt[:500] if excerpt else "<empty-body>",
                )
                return None

            content_buf: List[str] = []
            saw_done = False
            final_finish_reason: Optional[str] = None

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: ") :].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                try:
                    obj = json.loads(data)
                    choice = obj["choices"][0]
                    delta = choice.get("delta", {})
                    if "content" in delta:
                        content_buf.append(delta["content"])
                    if choice.get("finish_reason"):
                        final_finish_reason = choice["finish_reason"]
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    pass

            return {
                "text": "".join(content_buf),
                "saw_done": saw_done,
                "finish_reason": final_finish_reason,
            }

    async def chat_streaming_with_signals(
        self,
        messages: List[Dict[str, str]],
        temperature_override: Optional[float] = None,
        max_tokens_override: Optional[int] = None,
    ) -> Optional[StreamingResponse]:
        client = self._client
        if client is None:
            return None

        payload = self._build_payload(
            messages=messages,
            temperature_override=temperature_override,
            max_tokens=max_tokens_override or self.max_tokens,
            stream=True,
        )

        try:
            return await asyncio.wait_for(
                self._stream_sse(payload),
                timeout=self.streaming_total_timeout,
            )
        except asyncio.TimeoutError:
            LOGGER.warning(
                "Streaming call exceeded total timeout (%.0fs), aborting",
                self.streaming_total_timeout,
            )
            return None
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            LOGGER.exception("LLM stream request exception: %s", exc)
            return None
