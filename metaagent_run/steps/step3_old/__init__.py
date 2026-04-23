from .config import RuntimeConfig, load_runtime_config
from .orchestrator import schema_discovery_pipeline
from .prompt_builder import build_discovery_messages, load_prompt_template


def main_async(*args, **kwargs):
    from .orchestrator import main_async as _main_async

    return _main_async(*args, **kwargs)


__all__ = [
    "RuntimeConfig",
    "build_discovery_messages",
    "load_prompt_template",
    "load_runtime_config",
    "main_async",
    "schema_discovery_pipeline",
]
