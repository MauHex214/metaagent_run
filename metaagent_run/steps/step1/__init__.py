from .config import RuntimeConfig, load_runtime_config
from typing import Optional


async def main_async(runtime_config: Optional[RuntimeConfig] = None) -> None:
    from .orchestrator import main_async as _main_async

    return await _main_async(runtime_config=runtime_config)


__all__ = ["main_async", "RuntimeConfig", "load_runtime_config"]
