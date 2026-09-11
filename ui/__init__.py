"""LabLens 界面层：设计令牌与可复用组件。

- ``theme``      设计令牌 + 全局 CSS
- ``components`` 纯函数式组件（输入数据 → 输出 HTML）

组件与业务逻辑分离，因此可以被单测直接覆盖，不需要起浏览器。
"""

from .components import (
    critical_alert,
    disclaimer,
    empty,
    flag_badge,
    hero,
    item_card,
    item_table,
    note,
    render,
    section,
    severity_of,
    stat_cards,
)
from .theme import COLORS, FLAG_STYLE, SEVERITY_STYLE, inject_css

__all__ = [
    "COLORS",
    "FLAG_STYLE",
    "SEVERITY_STYLE",
    "inject_css",
    "critical_alert",
    "disclaimer",
    "empty",
    "flag_badge",
    "hero",
    "item_card",
    "item_table",
    "note",
    "render",
    "section",
    "severity_of",
    "stat_cards",
]
