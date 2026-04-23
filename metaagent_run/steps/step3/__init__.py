from .config import RuntimeConfig, load_runtime_config


def main_async(*args, **kwargs):
    from .orchestrator import main_async as _main_async
    return _main_async(*args, **kwargs)


__all__ = [
    "RuntimeConfig",
    "load_runtime_config",
    "main_async",
]
