"""功能描述：统一段落过滤规则判断。

输入：
- `section_type: str`，待判断的段落类型。
- `excluded_section_types: set[str] | None`，可选覆盖过滤集合。

输出：
- `bool`，True 表示应过滤，False 表示可继续处理。

用法示例：
```python
from metaagent.section_filters import is_excluded_section

if is_excluded_section(section_type):
    return None
```

各参数说明：
- section_type：输入段落类型字符串，函数内部会转大写后匹配。
- excluded_section_types：未传入时使用默认全局过滤集合。
"""

from typing import AbstractSet, Optional

EXCLUDED_SECTION_TYPES = frozenset(
    {
        "TITLE",
        "REF",
        "FIG",
        "COMP_INT",
        "AUTH_CONT",
        "REVIEW_INFO",
        "APPENDIX",
    }
)


def is_excluded_section(
    section_type: str, excluded_section_types: Optional[AbstractSet[str]] = None
) -> bool:
    filters = excluded_section_types or EXCLUDED_SECTION_TYPES
    return section_type.upper() in filters
