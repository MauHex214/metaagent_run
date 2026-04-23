from typing import Any

from .config import RuntimeConfig, load_runtime_config


async def main_async(*args: Any, **kwargs: Any):
    from .orchestrator import main_async as _main_async

    return await _main_async(*args, **kwargs)


__all__ = ["main_async", "RuntimeConfig", "load_runtime_config"]
