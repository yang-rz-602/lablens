"""单元测试：输出护栏。

护栏拦截的是**法规红线内容**（诊断结论、用药与剂量、处方），
所以既要测"该拦的拦住了"，也要测"不该拦的没误伤"。
"""

from __future__ import annotations

import pytest

from core.guard import GUARD_RULES, sanitize_text, scan_text


# --------------------------------------------------------------------------- #
# 必须拦截
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "建议服用阿司匹林100mg。",
        "可以口服二甲双胍 0.5g。",
        "建议使用头孢克肟 100 mg。",
        "请注射胰岛素 10 单位。",
        "建议您自行加量。",
        "可以考虑停药观察。",
        "以下是您的用药方案。",
        "医生会给您开一个处方。",
    ],
)
def test_blocks_medication_advice(text: str) -> None:
    hits = scan_text(text)
    assert hits, f"未拦截用药建议：{text}"
    assert all(h.category == "处方与用药" for h in hits)


@pytest.mark.parametrize(
    "text",
    [
        "可以确诊为糖尿病。",
        "提示您患有肝炎。",
        "这说明您得了贫血。",
        "检查结果就是肿瘤的表现。",
    ],
)
def test_blocks_definitive_diagnosis(text: str) -> None:
    hits = scan_text(text)
    assert hits, f"未拦截诊断断言：{text}"
    assert any(h.category == "诊断断言" for h in hits)


@pytest.mark.parametrize(
    "text",
    [
        "这个结果一定是异常的。",
        "说明您的身体100%有问题。",
        "完全正常，无需就医。",
        "您可以放心，绝对健康。",
    ],
)
def test_blocks_overreach(text: str) -> None:
    assert scan_text(text), f"未拦截绝对化表述：{text}"


# --------------------------------------------------------------------------- #
# 不能误伤
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "白细胞计数偏高，常见相关因素包括细菌感染与应激状态。",
        "该结果也可能由剧烈运动、进食后采血等非疾病因素导致。",
        "建议携带原始报告就诊，由医生结合病史综合判断。",
        "肌酐是肌肉代谢的产物，主要由肾小球滤过排出。",
        "参考区间为 3.5-9.5×10^9/L，本次结果为 11.2。",
        "是否需要复查以及复查间隔，请与医生确认。",
        "如果您正在服用某些药物，可能会影响该指标，请告知医生。",
    ],
)
def test_does_not_flag_legitimate_text(text: str) -> None:
    hits = scan_text(text)
    assert not hits, f"误伤正常文本：{text} -> {[h.matched for h in hits]}"


# --------------------------------------------------------------------------- #
# sanitize
# --------------------------------------------------------------------------- #
def test_sanitize_removes_offending_sentence_keeps_rest() -> None:
    text = "白细胞偏高常见于细菌感染。建议服用阿莫西林 500mg。请携带报告就诊。"
    clean, hits = sanitize_text(text)
    assert "阿莫西林" not in clean
    assert "细菌感染" in clean
    assert "携带报告就诊" in clean
    assert hits


def test_sanitize_returns_hits_with_rule_names() -> None:
    _, hits = sanitize_text("建议服用布洛芬 200mg。")
    assert hits[0].rule in {r[0] for r in GUARD_RULES}
    assert hits[0].sentence


def test_sanitize_all_offending_returns_empty() -> None:
    clean, hits = sanitize_text("建议服用阿司匹林100mg。")
    assert clean == ""
    assert hits


def test_sanitize_handles_empty() -> None:
    assert sanitize_text(None) == ("", [])
    assert sanitize_text("") == ("", [])


def test_sanitize_preserves_clean_text() -> None:
    text = "这项指标偏高，建议结合病史综合判断。"
    clean, hits = sanitize_text(text)
    assert clean == text
    assert hits == []


def test_multiple_categories_reported() -> None:
    _, hits = sanitize_text("可以确诊为糖尿病。建议服用二甲双胍 0.5g。")
    cats = {h.category for h in hits}
    assert "处方与用药" in cats
    assert "诊断断言" in cats
