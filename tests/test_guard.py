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
# 回归：药名出现在「病因描述」里不得被误判成用药建议
# --------------------------------------------------------------------------- #
#: 逐字来自 data/corpus.jsonl（低钠的 lowering 相关因素）。
#: 「抗利尿激素」是 SIADH 这个疾病状态的一部分，不是被开出的药。
CAUSAL_DRUG_SENTENCE = (
    "也可见于大量出汗后仅补充水分、长期低盐饮食、稀释性低钠血症与"
    "输注不含氯的液体，以及使用部分利尿剂和抗利尿激素分泌异常状态。"
)


def test_drug_mention_in_causal_context_is_not_flagged() -> None:
    """回归：护栏曾把这句话整句删掉，导致知识库里合法的一段病因说明凭空消失。"""
    hits = scan_text(CAUSAL_DRUG_SENTENCE)
    assert not hits, f"误伤知识库原文：{[h.matched for h in hits]}"


def test_causal_drug_sentence_survives_sanitize() -> None:
    clean, hits = sanitize_text(CAUSAL_DRUG_SENTENCE)
    assert clean == CAUSAL_DRUG_SENTENCE, "合法医学说明被删掉了"
    assert hits == []


@pytest.mark.parametrize(
    "text",
    [
        "建议使用胰岛素。",
        "可以口服激素治疗。",
        "需要使用抗生素。",
    ],
)
def test_prescriptive_drug_mention_is_still_blocked(text: str) -> None:
    """同一批药名一旦带上指导语气必须继续拦——语境豁免不能变成放水。"""
    assert scan_text(text), f"带指导语气的药名建议被漏放：{text}"


def test_bare_drug_instruction_without_cues_is_still_blocked() -> None:
    """既无描述性线索、也无祈使线索的裸指令，按原强度拦截。"""
    assert scan_text("使用胰岛素。")


def test_dose_rules_are_not_exempted() -> None:
    """剂量类信号最硬，即使句子里有描述性措辞也不豁免。"""
    assert scan_text("可见于服用阿司匹林100mg后。")


# --------------------------------------------------------------------------- #
# 回归：注射类给药指令曾经整条漏放（动词表缺项，属原有缺陷）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "每天注射胰岛素两次。",
        "静滴青霉素。",
        "吸入激素。",
        "外用激素。",
    ],
)
def test_parenteral_instructions_are_blocked(text: str) -> None:
    """``drug_name_plus_action`` 原先的动词表缺了注射/静滴/静注/外用/吸入/含服，

    于是"静滴青霉素"这类**不带剂量**的给药指令完全没有规则覆盖。这里锁死不再复发。
    """
    assert scan_text(text), f"注射类给药指令被漏放：{text}"


def test_nominalized_drug_phrase_is_not_an_instruction() -> None:
    """「注射**的**胰岛素制剂」是名词短语（被注射的胰岛素），不是给药指令。

    判据是动词与药名之间是否夹了「的」：祈使句不会带它。
    这句逐字来自 data/corpus.jsonl（C肽条目）。
    """
    text = (
        "C肽是胰岛素原经酶切后释放的连接肽，与胰岛素以等摩尔比例由胰岛β细胞"
        "共同分泌入血，其半衰期较胰岛素长，且外源注射的胰岛素制剂中不含C肽。"
    )
    assert not scan_text(text), "把名词短语误判成了给药指令"


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
