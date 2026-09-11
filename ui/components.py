"""LabLens 界面组件。

约定
----
- 所有动态文本一律经 ``esc()`` 转义。报告上的项目名来自 OCR，
  直接拼进 HTML 既会破坏版式，也是一个注入面。
- 组件只从 ``ui.theme`` 取色，不写字面量颜色。
- 每个组件是纯函数：输入数据 → 输出 HTML 字符串，``render()`` 负责落地。
  这样组件可以被单测覆盖（见 ``tests/test_ui.py``）。
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence

import streamlit as st

from core.schema import Flag, LabItem, RefSource, Severity

from .theme import COLORS, FLAG_STYLE, SEVERITY_STYLE

__all__ = [
    "esc",
    "render",
    "hero",
    "disclaimer",
    "section",
    "stat_cards",
    "critical_alert",
    "flag_badge",
    "item_table",
    "item_card",
    "empty",
    "note",
    "severity_of",
]

_C = COLORS


# --------------------------------------------------------------------------- #
def esc(value: object) -> str:
    """转义任意值为可安全嵌入 HTML 的文本。"""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def render(html_str: str) -> None:
    """渲染 HTML 片段。

    优先用 ``st.html``：它不做 Markdown 处理，因此不会把 HTML 里的缩进
    误当成代码块（``st.markdown`` 有这个坑）。
    """
    if not html_str:
        return
    if hasattr(st, "html"):
        st.html(html_str)
    else:  # pragma: no cover - 兼容老版本
        st.markdown(html_str, unsafe_allow_html=True)


def severity_of(item: LabItem) -> str:
    """把 LabItem 映射到视觉严重度键。"""
    return {
        Severity.CRITICAL: "critical",
        Severity.ABNORMAL: "abnormal",
        Severity.BORDERLINE: "borderline",
        Severity.NORMAL: "normal",
        Severity.UNKNOWN: "unknown",
    }.get(item.severity, "unknown")


# --------------------------------------------------------------------------- #
# 页面级组件
# --------------------------------------------------------------------------- #
def hero(title: str, subtitle: str, chips: Sequence[str] = ()) -> None:
    chip_html = "".join(f'<span class="ll-chip">{esc(c)}</span>' for c in chips)
    render(
        f'<div class="ll-hero">'
        f"<h1>{esc(title)}</h1>"
        f'<p class="ll-sub">{esc(subtitle)}</p>'
        f'<div class="ll-chips">{chip_html}</div>'
        f"</div>"
    )


_DISCLAIMER_ICON = (
    '<svg class="ll-ic" width="17" height="17" viewBox="0 0 24 24" fill="none" '
    'stroke="#B45309" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="10"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg>'
)

_CRITICAL_ICON = (
    '<svg class="ll-ic" width="19" height="19" viewBox="0 0 24 24" fill="none" '
    'stroke="#DC2626" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86'
    'a2 2 0 0 0-3.42 0z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>'
)


def disclaimer(text_html: str) -> None:
    """顶部常驻的 AI 生成内容 + 非医疗器械声明。

    ``text_html`` 允许含少量标记（``<b>``、``<code>``），调用方负责其内容可信。
    """
    render(
        f'<div class="ll-disclaimer">{_DISCLAIMER_ICON}'
        f'<div class="ll-tx">{text_html}</div></div>'
    )


def critical_alert(names_html: str) -> None:
    render(
        f'<div class="ll-alert critical">{_CRITICAL_ICON}'
        f'<div class="ll-tx"><span class="ll-title">检测到达到危急值阈值的项目</span>'
        f"{names_html}<br>危急值意味着该结果可能需要立即的医疗关注。"
        f"请<b>尽快</b>携带原始报告联系医生或前往医院，不要等待或自行处理。</div></div>"
    )


def section(title: str, hint: str = "") -> None:
    hint_html = f'<span class="ll-sec-h">{esc(hint)}</span>' if hint else ""
    render(f'<div class="ll-sec"><span class="ll-sec-t">{esc(title)}</span>{hint_html}</div>')


def empty(icon: str, text: str) -> None:
    render(
        f'<div class="ll-empty"><span class="ll-ei">{esc(icon)}</span>'
        f'<span class="ll-et">{esc(text)}</span></div>'
    )


def note(text: str) -> None:
    render(f'<div class="ll-note">{esc(text)}</div>')


# --------------------------------------------------------------------------- #
def stat_cards(cards: Iterable[dict]) -> None:
    """统计卡片行。每张卡片的左侧色条与数字颜色由严重度决定。"""
    cards = list(cards)
    parts = []
    for card in cards:
        style = SEVERITY_STYLE.get(card.get("tone", "unknown"), SEVERITY_STYLE["unknown"])
        unit = f'<span class="ll-u">{esc(card["unit"])}</span>' if card.get("unit") else ""
        parts.append(
            f'<div class="ll-card" style="--accent:{style["dot"]};--fg:{style["fg"]}">'
            f'<div class="ll-k">{esc(card["k"])}</div>'
            f'<div class="ll-v">{esc(card["v"])}{unit}</div>'
            f"</div>"
        )
    render(f'<div class="ll-cards c{min(len(cards), 5)}">{"".join(parts)}</div>')


def flag_badge(flag: Flag | str, text: str | None = None) -> str:
    """判定徽标。定性项目如果自带原文（如"阳性"），就显示原文而不是"偏高"。"""
    key = flag.value if isinstance(flag, Flag) else str(flag)
    label, tone = FLAG_STYLE.get(key, ("未判定", "unknown"))
    style = SEVERITY_STYLE[tone]
    return (
        f'<span class="ll-badge" style="background:{style["bg"]};color:{style["fg"]};'
        f'border:1px solid {style["bd"]}"><span class="dot"></span>{esc(text or label)}</span>'
    )


def _source_chip(source: RefSource) -> str:
    label = {
        RefSource.REPORT: "报告单",
        RefSource.KNOWLEDGE_BASE: "知识库",
        RefSource.NONE: "无",
    }[source]
    cls = "ll-src report" if source is RefSource.REPORT else "ll-src"
    return f'<span class="{cls}">{esc(label)}</span>'


def item_table(rows: Sequence[dict]) -> None:
    """指标明细表。

    ``rows`` 每项形如
    ``{"name","value","unit","specimen","ref","ref_source","flag_key","flag_text","basis"}``
    """
    head = (
        "<thead><tr>"
        "<th>项目</th><th>结果</th><th>单位</th><th>标本</th>"
        "<th>参考区间</th><th>区间来源</th><th>判定</th><th>依据</th>"
        "</tr></thead>"
    )
    body = []
    for r in rows:
        src = r.get("ref_source", RefSource.NONE)
        chip = _source_chip(src if isinstance(src, RefSource) else RefSource.NONE)
        body.append(
            "<tr>"
            f'<td class="rn">{esc(r.get("name"))}</td>'
            f'<td class="num">{esc(r.get("value"))}</td>'
            f'<td>{esc(r.get("unit") or "—")}</td>'
            f'<td>{esc(r.get("specimen") or "—")}</td>'
            f'<td class="num">{esc(r.get("ref") or "—")}</td>'
            f"<td>{chip}</td>"
            f'<td>{flag_badge(r.get("flag_key", "?"), r.get("flag_text"))}</td>'
            f'<td>{esc(r.get("basis") or "—")}</td>'
            "</tr>"
        )
    render(
        '<div class="ll-tbl-wrap"><table class="ll-tbl">'
        f"{head}<tbody>{''.join(body)}</tbody></table></div>"
    )


def item_card(row: dict) -> None:
    """单个异常项的卡片，含字段级溯源与判读依据。"""
    style = SEVERITY_STYLE.get(row.get("tone", "unknown"), SEVERITY_STYLE["unknown"])
    factors = row.get("factors") or []
    factors_html = (
        '<div class="ll-blk"><span class="ll-lb">常见相关因素</span><ul>'
        + "".join(f"<li>{esc(f)}</li>" for f in factors)
        + "</ul></div>"
        if factors
        else ""
    )
    trace = row.get("trace")
    trace_html = f'<div class="ll-trace">{trace}</div>' if trace else ""
    src = row.get("source_text")
    src_html = f'<div class="ll-src-text">原报告行：{esc(src)}</div>' if src else ""
    one_liner = row.get("one_liner")
    one_liner_html = (
        f'<div class="ll-blk"><span class="ll-lb">是什么</span>{esc(one_liner)}</div>'
        if one_liner
        else ""
    )
    render(
        f'<div class="ll-item" style="--accent:{style["dot"]};--fg:{style["fg"]}">'
        f'<div class="ll-hd">'
        f'<span class="ll-nm">{esc(row.get("name"))}</span>'
        f'<span class="ll-val">{esc(row.get("value"))}</span>'
        f'<span class="ll-ref">参考 {esc(row.get("ref") or "—")}</span>'
        f'{flag_badge(row.get("flag_key", "?"), row.get("flag_text"))}'
        f"</div>"
        f"{one_liner_html}"
        f'<div class="ll-blk"><span class="ll-lb">本次结果</span>{esc(row.get("what") or "—")}</div>'
        f"{factors_html}{trace_html}{src_html}"
        f"</div>"
    )
