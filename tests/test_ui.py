"""组件层单测。

``ui.components`` 刻意设计成纯函数（数据 → HTML 字符串），
所以可以直接断言输出，不需要起浏览器。
"""

from __future__ import annotations

import pytest

from core.schema import Flag, RefSource
from ui import components as C
from ui.theme import COLORS, FLAG_STYLE, SEVERITY_STYLE


# --------------------------------------------------------------------------- #
# 转义（这是安全与版式正确性的基础）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("白细胞计数", "白细胞计数"),
        ("<script>alert(1)</script>", "&lt;script&gt;alert(1)&lt;/script&gt;"),
        ('a"b', "a&quot;b"),
        ("a'b", "a&#x27;b"),
        ("a&b", "a&amp;b"),
    ],
)
def test_esc(raw: str, expected: str) -> None:
    assert C.esc(raw) == expected


def test_esc_handles_none() -> None:
    assert C.esc(None) == ""


def test_esc_handles_non_string() -> None:
    assert C.esc(6.2) == "6.2"
    assert C.esc(0) == "0"


# --------------------------------------------------------------------------- #
# 判定徽标
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("flag", ["CH", "H", "N", "L", "CL", "?"])
def test_flag_badge_covers_all_flags(flag: str) -> None:
    html = C.flag_badge(flag)
    assert "ll-badge" in html
    assert FLAG_STYLE[flag][0] in html


def test_flag_badge_uses_severity_colors() -> None:
    critical = C.flag_badge("CH")
    abnormal = C.flag_badge("H")
    assert SEVERITY_STYLE["critical"]["fg"] in critical
    assert SEVERITY_STYLE["abnormal"]["fg"] in abnormal
    assert critical != abnormal


def test_flag_badge_accepts_enum() -> None:
    assert "正常" in C.flag_badge(Flag.NORMAL)


def test_flag_badge_shows_custom_text_for_qualitative() -> None:
    """定性项目要显示"阳性"原文，而不是显示成"偏高"。"""
    html = C.flag_badge(Flag.HIGH, "阳性")
    assert "阳性" in html
    assert "偏高" not in html


def test_flag_badge_unknown_flag_does_not_crash() -> None:
    assert "未判定" in C.flag_badge("NOT_A_FLAG")


def test_flag_badge_escapes_text() -> None:
    assert "<script>" not in C.flag_badge(Flag.HIGH, "<script>x</script>")


# --------------------------------------------------------------------------- #
# 严重度映射
# --------------------------------------------------------------------------- #
def test_severity_style_keys_are_complete() -> None:
    for key in ("critical", "abnormal", "borderline", "normal", "unknown"):
        assert key in SEVERITY_STYLE
        for field in ("fg", "bg", "bd", "dot"):
            assert field in SEVERITY_STYLE[key]


def test_all_colors_are_hex() -> None:
    for name, value in COLORS.items():
        assert value.startswith("#") and len(value) == 7, name


# --------------------------------------------------------------------------- #
# 统计卡片
# --------------------------------------------------------------------------- #
def test_stat_cards_renders_values_and_tone(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(C, "render", lambda h: captured.update(html=h))
    C.stat_cards([
        {"k": "检验项目", "v": 26, "tone": "unknown"},
        {"k": "危急值", "v": 2, "tone": "critical"},
    ])
    html = captured["html"]
    assert "检验项目" in html and "26" in html
    assert "危急值" in html
    assert SEVERITY_STYLE["critical"]["dot"] in html
    assert html.count("ll-card") >= 2


def test_stat_cards_caps_grid_at_five_columns(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(C, "render", lambda h: captured.update(html=h))
    C.stat_cards([{"k": f"k{i}", "v": i, "tone": "unknown"} for i in range(5)])
    assert "ll-cards c5" in captured["html"]


def test_stat_cards_escapes_text(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(C, "render", lambda h: captured.update(html=h))
    C.stat_cards([{"k": "<b>x</b>", "v": "<i>1</i>", "tone": "unknown"}])
    assert "<b>x</b>" not in captured["html"]
    assert "&lt;b&gt;x&lt;/b&gt;" in captured["html"]


# --------------------------------------------------------------------------- #
# 指标表
# --------------------------------------------------------------------------- #
def test_item_table_renders_all_columns(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(C, "render", lambda h: captured.update(html=h))
    C.item_table([{
        "name": "白细胞计数", "value": "6.2", "unit": "10^9/L", "specimen": "全血",
        "ref": "3.5-9.5", "ref_source": RefSource.REPORT,
        "flag_key": "N", "flag_text": None, "basis": "WS/T 405-2012",
    }])
    html = captured["html"]
    for text in ("项目", "结果", "单位", "标本", "参考区间", "区间来源", "判定", "依据"):
        assert text in html
    assert "白细胞计数" in html and "6.2" in html and "3.5-9.5" in html
    assert "报告单" in html
    assert "WS/T 405-2012" in html


def test_item_table_marks_reference_source(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(C, "render", lambda h: captured.update(html=h))
    C.item_table([{
        "name": "x", "value": "1", "ref_source": RefSource.REPORT, "flag_key": "N",
    }])
    assert "ll-src report" in captured["html"]

    C.item_table([{
        "name": "x", "value": "1", "ref_source": RefSource.KNOWLEDGE_BASE, "flag_key": "N",
    }])
    assert "ll-src report" not in captured["html"]
    assert "知识库" in captured["html"]


def test_item_table_handles_missing_fields(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(C, "render", lambda h: captured.update(html=h))
    C.item_table([{"name": "只有名字"}])
    assert "只有名字" in captured["html"]
    assert "—" in captured["html"]  # 空字段用占位符而不是 None


def test_item_table_escapes_every_field(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(C, "render", lambda h: captured.update(html=h))
    C.item_table([{
        "name": "<img onerror=1>", "value": "<x>", "unit": "<u>",
        "specimen": "<s>", "ref": "<r>", "flag_key": "H", "basis": "<b>",
    }])
    html = captured["html"]
    for bad in ("<img", "<x>", "<u>", "<s>", "<r>", "<b>"):
        assert bad not in html, bad


# --------------------------------------------------------------------------- #
# 异常项卡片
# --------------------------------------------------------------------------- #
def test_item_card_includes_trace_and_source(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(C, "render", lambda h: captured.update(html=h))
    C.item_card({
        "name": "丙氨酸氨基转移酶", "value": "685 U/L", "ref": "9-50",
        "flag_key": "CH", "tone": "critical",
        "one_liner": "肝细胞内的一种酶。",
        "what": "明显高于参考上限。",
        "factors": ["病毒性肝炎", "药物性肝损伤"],
        "trace": "规则 <code>kb_interval</code>　·　跨过 <code>critical_high</code>",
        "source_text": "丙氨酸氨基转移酶 685 U/L 9-50 ↑",
    })
    html = captured["html"]
    assert "ll-item" in html
    assert SEVERITY_STYLE["critical"]["dot"] in html
    assert "肝细胞内的一种酶。" in html
    assert "病毒性肝炎" in html
    assert "critical_high" in html
    assert "原报告行" in html


def test_item_card_omits_empty_sections(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(C, "render", lambda h: captured.update(html=h))
    C.item_card({"name": "x", "value": "1", "tone": "normal", "what": "正常"})
    html = captured["html"]
    assert "常见相关因素" not in html
    assert "ll-trace" not in html
    assert "原报告行" not in html


# --------------------------------------------------------------------------- #
# 其它组件
# --------------------------------------------------------------------------- #
def test_hero_renders_title_subtitle_chips(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(C, "render", lambda h: captured.update(html=h))
    C.hero("标题", "副标题", ["标签一", "标签二"])
    html = captured["html"]
    assert "标题" in html and "副标题" in html
    assert html.count('class="ll-chip"') == 2


def test_disclaimer_passes_through_markup(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(C, "render", lambda h: captured.update(html=h))
    C.disclaimer("<b>AI 生成内容</b>")
    assert "ll-disclaimer" in captured["html"]
    assert "<b>AI 生成内容</b>" in captured["html"]


def test_critical_alert_has_icon_and_class(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(C, "render", lambda h: captured.update(html=h))
    C.critical_alert("<code>血钾 6.8</code>")
    html = captured["html"]
    assert "ll-alert critical" in html
    assert "<svg" in html
    assert "血钾 6.8" in html


def test_section_renders_hint(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(C, "render", lambda h: captured.update(html=h))
    C.section("指标明细", "共 26 项")
    assert "指标明细" in captured["html"]
    assert "共 26 项" in captured["html"]


def test_empty_and_note(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(C, "render", lambda h: captured.update(html=h))
    C.empty("🧪", "请先分析")
    assert "ll-empty" in captured["html"] and "请先分析" in captured["html"]
    C.note("提示文字")
    assert "ll-note" in captured["html"] and "提示文字" in captured["html"]


def test_render_skips_empty_string() -> None:
    C.render("")  # 不应抛异常
