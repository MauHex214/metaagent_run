"""功能描述：对外暴露 metaagent 公共入口函数。

输入：无。

输出：
- 导出 `main_async`、`RuntimeConfig`、`load_runtime_config`。

用法示例：
```python
from metaagent import main_async
```

各参数说明：
- main_async(input_file, output_file, max_concurrency)：
  由 `metaagent_run.steps.step2.orchestrator` 提供的异步主流程函数。
- RuntimeConfig：运行期配置对象。
- load_runtime_config：从环境变量加载配置对象。
"""

from .steps.step2.config import RuntimeConfig, load_runtime_config


async def main_async(*args, **kwargs):
    from .steps.step2.orchestrator import main_async as _main_async

    return await _main_async(*args, **kwargs)

__all__ = [
    "main_async",
    "RuntimeConfig",
    "load_runtime_config",
]
