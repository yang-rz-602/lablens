"""单元测试：规则引擎。

这是项目最核心的一组测试。规则引擎是整个系统里唯一做数值判断的地方，
所以它的行为必须被穷举钉死——尤其是"拒绝判定"这条路径。
"""

from __future__ import annotations

import pytest

from core.rules import Indicator, KnowledgeBase, build_report, judge_item
from core.schema import Flag, PatientContext, RawLabItem, RawLabReport, RefSource, Severity


# --------------------------------------------------------------------------- #
@pytest.fixture()
def kb() -> KnowledgeBase:
    return KnowledgeBase(
        [
            Indicator(
                canonical_name="wbc", name_zh="白细胞计数",
                aliases=["WBC", "白细胞", "白细胞计数", "WBCC"],
                unit_canonical="10^9/L", category="血常规",
                ref={"default": {"low": 3.5, "high": 9.5}},
                critical={"low": 1.5, "high": 30.0},
                source="WS/T 405-2012",
            ),
            Indicator(
                canonical_name="hgb", name_zh="血红蛋白",
                aliases=["HGB", "Hb", "血红蛋白"],
                unit_canonical="g/L", category="血常规",
                ref={"default": {"low": 115, "high": 175},
                     "M": {"low": 130, "high": 175},
                     "F": {"low": 115, "high": 150}},
                critical={"low": 50, "high": 200},
                source="WS/T 405-2012",
            ),
            Indicator(
                canonical_name="crea", name_zh="肌酐", aliases=["CREA", "Cr", "肌酐"],
                unit_canonical="μmol/L", category="肾功能",
                ref={"M": {"low": 57, "high": 97}, "F": {"low": 41, "high": 73},
                     "default": {"low": 41, "high": 97}},
                unit_conversions=[{"from": "mg/dL", "factor": 88.4}],
                source="WS/T 404.2-2012",
            ),
            Indicator(
                canonical_name="u_pro", name_zh="尿蛋白", aliases=["尿蛋白", "PRO", "蛋白质"],
                qualitative=True, category="尿常规",
                ref={"qualitative_normal": ["阴性", "-", "neg"]},
                source="临床检验操作规程",
            ),
            Indicator(
                canonical_name="odd_unit", name_zh="奇怪单位项目", aliases=["奇怪单位项目"],
                unit_canonical="mmol/L",
                ref={"default": {"low": 1.0, "high": 2.0}},
                unit_conversions=[],  # 刻意不登记换算因子
                source="测试",
            ),
        ],
        version="test",
    )


def _item(**kw) -> RawLabItem:
    return RawLabItem(**kw)


# --------------------------------------------------------------------------- #
# 参考区间优先级
# --------------------------------------------------------------------------- #
def test_report_reference_interval_wins_over_knowledge_base(kb: KnowledgeBase) -> None:
    """同一家医院可能用不同区间，报告上印的必须优先。"""
    # 知识库说 3.5-9.5，但这家医院印的是 4.0-10.0，结果 9.8
    item = judge_item(_item(raw_name="白细胞", result_raw="9.8", unit_raw="10^9/L",
                           ref_raw="4.0-10.0"), kb, PatientContext(sex="M"))
    assert item.ref_source is RefSource.REPORT
    assert (item.ref_low, item.ref_high) == (4.0, 10.0)
    assert item.flag is Flag.NORMAL, "按报告区间 9.8 在 4.0-10.0 内，应为正常"

    # 若同一结果没有报告区间，用知识库 3.5-9.5 判，就是偏高
    item2 = judge_item(_item(raw_name="白细胞", result_raw="9.8", unit_raw="10^9/L"), kb,
                       PatientContext(sex="M"))
    assert item2.ref_source is RefSource.KNOWLEDGE_BASE
    assert item2.flag is Flag.HIGH


def test_missing_reference_interval_refuses_to_judge(kb: KnowledgeBase) -> None:
    """核心机制：两边都没有区间时必须拒绝判定，而不是猜一个正常范围。"""
    item = judge_item(
        _item(raw_name="某个陌生指标", result_raw="12.3", unit_raw="U/L"),
        kb, PatientContext(sex="M"),
    )
    assert item.flag is Flag.UNKNOWN
    assert item.severity is Severity.UNKNOWN
    assert item.ref_source is RefSource.NONE
    assert any("拒绝判定" in n for n in item.rules_notes)


def test_report_interval_present_but_unknown_indicator_still_judges(kb: KnowledgeBase) -> None:
    """指标不在字典里，但报告印了区间——仍然可以判定。"""
    item = judge_item(
        _item(raw_name="未收录项目", result_raw="15", ref_raw="1-10"),
        kb, PatientContext(),
    )
    assert item.flag is Flag.HIGH
    assert item.ref_source is RefSource.REPORT


# --------------------------------------------------------------------------- #
# 定量判定
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [
        ("5.0", Flag.NORMAL),
        ("9.5", Flag.NORMAL),        # 恰好等于上限算正常
        ("9.6", Flag.HIGH),
        ("3.5", Flag.NORMAL),
        ("3.4", Flag.LOW),
        ("1.4", Flag.CRITICAL_LOW),  # 低于危急下限 1.5
        ("31.0", Flag.CRITICAL_HIGH),
    ],
)
def test_quantitative_thresholds(kb: KnowledgeBase, value: str, expected: Flag) -> None:
    item = judge_item(_item(raw_name="WBC", result_raw=value), kb, PatientContext(sex="M"))
    assert item.flag is expected


def test_borderline_is_flagged_not_alarming(kb: KnowledgeBase) -> None:
    """略高于上限应判为临界而不是异常，避免制造焦虑。"""
    item = judge_item(_item(raw_name="WBC", result_raw="9.7"), kb, PatientContext(sex="M"))
    assert item.flag is Flag.HIGH
    assert item.severity is Severity.BORDERLINE


def test_clearly_abnormal_severity(kb: KnowledgeBase) -> None:
    item = judge_item(_item(raw_name="WBC", result_raw="15.0"), kb, PatientContext(sex="M"))
    assert item.flag is Flag.HIGH
    assert item.severity is Severity.ABNORMAL


# --------------------------------------------------------------------------- #
# 性别特异区间
# --------------------------------------------------------------------------- #
def test_sex_specific_reference_interval(kb: KnowledgeBase) -> None:
    """血红蛋白 120 g/L：女性正常，男性偏低——不区分性别就会误判。"""
    female = judge_item(_item(raw_name="血红蛋白", result_raw="120"), kb, PatientContext(sex="F"))
    male = judge_item(_item(raw_name="血红蛋白", result_raw="120"), kb, PatientContext(sex="M"))
    assert female.flag is Flag.NORMAL
    assert male.flag is Flag.LOW


def test_unknown_sex_refuses_sex_specific_indicator(kb: KnowledgeBase) -> None:
    """参考区间按性别分层、报告又没写性别时，必须拒判而不是随便挑一个区间。

    用男性区间判女性会误报"偏低"，用男女并集判男性会漏报真实的偏低——
    两种都是静默错误，所以选择不做判定。
    """
    item = judge_item(_item(raw_name="血红蛋白", result_raw="120"), kb, PatientContext())
    assert item.flag is Flag.UNKNOWN
    assert item.decision_trace.rule == "refused_sex_required"
    assert any("性别" in n for n in item.rules_notes)


def test_non_sex_specific_indicator_judges_without_sex(kb: KnowledgeBase) -> None:
    """WBC 的区间不按性别分层，缺性别也应照常判定。"""
    item = judge_item(_item(raw_name="WBC", result_raw="5.0"), kb, PatientContext())
    assert item.flag is Flag.NORMAL


def test_report_interval_bypasses_sex_requirement(kb: KnowledgeBase) -> None:
    """报告自带区间时不需要性别——那份区间本来就是对这个受检者有效的。"""
    item = judge_item(
        _item(raw_name="血红蛋白", result_raw="120", ref_raw="115-150"), kb, PatientContext()
    )
    assert item.flag is Flag.NORMAL
    assert item.ref_source is RefSource.REPORT


# --------------------------------------------------------------------------- #
# 定性项目
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value,expected", [("阴性", Flag.NORMAL), ("-", Flag.NORMAL), ("neg", Flag.NORMAL)])
def test_qualitative_normal(kb: KnowledgeBase, value: str, expected: Flag) -> None:
    item = judge_item(_item(raw_name="尿蛋白", result_raw=value, ref_raw="阴性"), kb, PatientContext())
    assert item.flag is expected


def test_qualitative_positive_is_abnormal(kb: KnowledgeBase) -> None:
    item = judge_item(_item(raw_name="尿蛋白", result_raw="+", ref_raw="阴性"), kb, PatientContext())
    assert item.flag is Flag.HIGH
    assert item.severity is Severity.ABNORMAL
    assert item.value_text == "+"


def test_qualitative_weak_positive_is_borderline(kb: KnowledgeBase) -> None:
    item = judge_item(_item(raw_name="尿蛋白", result_raw="±", ref_raw="阴性"), kb, PatientContext())
    assert item.severity is Severity.BORDERLINE


# --------------------------------------------------------------------------- #
# 单位处理
# --------------------------------------------------------------------------- #
def test_unit_conversion_applied_when_using_kb_interval(kb: KnowledgeBase) -> None:
    """报告用 mg/dL，知识库区间是 μmol/L —— 必须换算后再判。"""
    item = judge_item(
        _item(raw_name="肌酐", result_raw="1.2", unit_raw="mg/dL"), kb, PatientContext(sex="M")
    )
    assert item.unit == "μmol/L"
    assert item.value == pytest.approx(106.08)
    assert item.flag is Flag.HIGH, "106.08 > 97，应为偏高"


def test_no_conversion_when_report_provides_interval(kb: KnowledgeBase) -> None:
    """报告自带区间时，区间与结果同单位，绝不能换算其中一个。"""
    item = judge_item(
        _item(raw_name="肌酐", result_raw="1.2", unit_raw="mg/dL", ref_raw="0.7-1.3"),
        kb, PatientContext(sex="M"),
    )
    assert item.value == pytest.approx(1.2), "不应被换算"
    assert (item.ref_low, item.ref_high) == (0.7, 1.3)
    assert item.flag is Flag.NORMAL


def test_unregistered_unit_conversion_refuses_to_judge(kb: KnowledgeBase) -> None:
    """单位对不上又没有换算因子时，宁可放弃判定也不猜。"""
    item = judge_item(
        _item(raw_name="奇怪单位项目", result_raw="5.0", unit_raw="mg/dL"),
        kb, PatientContext(),
    )
    assert item.flag is Flag.UNKNOWN
    assert any("放弃判定" in n for n in item.rules_notes)


# --------------------------------------------------------------------------- #
# 审计痕迹
# --------------------------------------------------------------------------- #
def test_conflict_with_reported_flag_is_recorded(kb: KnowledgeBase) -> None:
    """报告标了 ↑ 但我们按区间判为正常时，必须留下审计痕迹，而不是悄悄覆盖。"""
    item = judge_item(
        _item(raw_name="白细胞", result_raw="5.0", flag_raw="↑"), kb, PatientContext(sex="M")
    )
    assert item.flag is Flag.NORMAL
    assert item.flag_reported == "↑"
    assert any("建议人工复核" in n for n in item.rules_notes)


def test_bounded_result_gets_note(kb: KnowledgeBase) -> None:
    item = judge_item(_item(raw_name="WBC", result_raw="<3.0"), kb, PatientContext(sex="M"))
    assert item.flag is Flag.LOW
    assert any("界限符号" in n for n in item.rules_notes)


def test_extreme_bounded_result_hits_critical(kb: KnowledgeBase) -> None:
    """<0.01 这种带界限的极端值也要能触发危急值判定。"""
    item = judge_item(_item(raw_name="WBC", result_raw="<0.01"), kb, PatientContext(sex="M"))
    assert item.flag is Flag.CRITICAL_LOW


def test_decision_trace_records_which_interval_was_used(kb: KnowledgeBase) -> None:
    """判读依据必须可审计：用了哪条区间、来自哪里、跨过哪个界。"""
    item = judge_item(
        _item(raw_name="WBC", result_raw="12.0", ref_raw="3.5-9.5"), kb, PatientContext(sex="M")
    )
    trace = item.decision_trace
    assert trace is not None
    assert trace.rule == "report_interval"
    assert trace.interval_source == "报告单"
    assert trace.crossed == "ref_high"
    assert "12" in trace.outcome


def test_decision_trace_marks_kb_source_and_standard(kb: KnowledgeBase) -> None:
    item = judge_item(_item(raw_name="WBC", result_raw="5.0"), kb, PatientContext(sex="M"))
    trace = item.decision_trace
    assert trace is not None
    assert trace.rule == "kb_interval"
    assert trace.interval_source == "内置知识库"
    assert trace.interval_standard == "WS/T 405-2012"
    assert trace.crossed == "within"


def test_decision_trace_marks_refusal(kb: KnowledgeBase) -> None:
    item = judge_item(_item(raw_name="陌生指标", result_raw="5.0"), kb, PatientContext())
    assert item.decision_trace is not None
    assert item.decision_trace.rule == "refused_no_interval"


def test_decision_trace_marks_critical_crossing(kb: KnowledgeBase) -> None:
    item = judge_item(_item(raw_name="WBC", result_raw="35.0"), kb, PatientContext(sex="M"))
    assert item.decision_trace.crossed == "critical_high"


def test_qualitative_normal_synonyms_all_recognized(kb: KnowledgeBase) -> None:
    """报告印"阴性"、结果写"-"时不能误判为阳性。"""
    for value in ("阴性", "-", "neg", "阴性(-)"):
        item = judge_item(
            _item(raw_name="尿蛋白", result_raw=value, ref_raw="阴性"), kb, PatientContext()
        )
        assert item.flag is Flag.NORMAL, f"{value} 被误判为 {item.flag}"


# --------------------------------------------------------------------------- #
# 别名解析
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alias", ["WBC", "wbc", "白细胞", "白细胞计数", "白细胞计数(WBC)", "WBCC"])
def test_alias_resolution(kb: KnowledgeBase, alias: str) -> None:
    assert kb.resolve(alias) is not None
    assert kb.resolve(alias).canonical_name == "wbc"


def test_unknown_indicator_not_fuzzy_matched(kb: KnowledgeBase) -> None:
    """不许模糊匹配——把 A 项目的区间套到 B 项目上是静默错误。"""
    assert kb.resolve("白细胞介素6") is None
    assert kb.resolve("") is None
    assert kb.resolve(None) is None


# --------------------------------------------------------------------------- #
# 跨类别同名：靠报告类型收窄作用域
# --------------------------------------------------------------------------- #
@pytest.fixture()
def cross_category_kb() -> KnowledgeBase:
    """`白细胞` 在血常规与尿常规里是两个完全不同的指标。"""
    return KnowledgeBase(
        [
            Indicator(
                canonical_name="wbc", name_zh="白细胞计数", aliases=["WBC", "白细胞"],
                category="血常规", unit_canonical="10^9/L",
                ref={"default": {"low": 3.5, "high": 9.5}},
            ),
            Indicator(
                canonical_name="u_le", name_zh="尿白细胞酯酶", aliases=["LEU", "白细胞", "尿白细胞酯酶"],
                category="尿常规", qualitative=True,
                ref={"qualitative_normal": ["阴性", "-"]},
            ),
        ]
    )


def test_alias_conflict_is_recorded(cross_category_kb: KnowledgeBase) -> None:
    """共享的名字必须被记录下来，而不是悄悄归属先登记的那一方。"""
    conflicts = cross_category_kb.conflicts()
    assert any("wbc" in v and "u_le" in v for v in conflicts.values())


def test_report_type_disambiguates_blood_panel(cross_category_kb: KnowledgeBase) -> None:
    ind = cross_category_kb.resolve("白细胞", prefer_category="血常规")
    assert ind is not None and ind.canonical_name == "wbc"


def test_report_type_disambiguates_urine_panel(cross_category_kb: KnowledgeBase) -> None:
    """尿常规报告里的"白细胞"必须解析成尿白细胞酯酶，不能串到血常规的 WBC。"""
    ind = cross_category_kb.resolve("白细胞", prefer_category="尿常规")
    assert ind is not None and ind.canonical_name == "u_le"


def test_conflict_without_hint_falls_back_to_owner(cross_category_kb: KnowledgeBase) -> None:
    ind = cross_category_kb.resolve("白细胞")
    assert ind is not None and ind.canonical_name == "wbc"


def test_judge_item_threads_report_type(cross_category_kb: KnowledgeBase) -> None:
    """端到端：尿常规报告里的一项"白细胞"应被判成定性项目。"""
    item = judge_item(
        _item(raw_name="白细胞", result_raw="+", ref_raw="阴性"),
        cross_category_kb,
        PatientContext(sex="F"),
        report_type="尿常规",
    )
    assert item.canonical_name == "u_le"
    assert item.category == "尿常规"


# --------------------------------------------------------------------------- #
# 整份报告
# --------------------------------------------------------------------------- #
def test_build_report_end_to_end(kb: KnowledgeBase) -> None:
    raw = RawLabReport(
        report_type="血常规", hospital="测试医院", collected_at="2026-03-12",
        patient_sex="男", patient_age=28,
        items=[
            RawLabItem(raw_name="WBC", result_raw="6.2", ref_raw="3.5-9.5", page=1),
            RawLabItem(raw_name="HGB", result_raw="100", page=1),
            RawLabItem(raw_name="PLT", result_raw="20", ref_raw="125-350", page=1),
            RawLabItem(raw_name="陌生项目", result_raw="1.0", page=1),
        ],
    )
    report = build_report(raw, kb, model_used="test-model")

    assert report.report_id and len(report.report_id) == 16
    assert report.patient.sex == "M"
    assert str(report.collected_at) == "2026-03-12"
    assert len(report.items) == 4

    counts = report.summary_counts()
    assert counts["N"] == 1
    assert counts["L"] == 2      # HGB 100 偏低、陌生项目无区间
    assert counts["?"] == 1

    # HGB 100 未达危急下限 50，所以不是危急
    hgb = next(i for i in report.items if i.canonical_name == "hgb")
    assert hgb.flag is Flag.LOW
    assert hgb.severity is Severity.ABNORMAL

    # abnormal_items 排序：异常在前
    assert report.abnormal_items
    assert all(i.flag not in {Flag.NORMAL} for i in report.abnormal_items)


def test_critical_items_surface(kb: KnowledgeBase) -> None:
    raw = RawLabReport(
        patient_sex="女",
        items=[
            RawLabItem(raw_name="WBC", result_raw="35.0"),
            RawLabItem(raw_name="HGB", result_raw="130"),
        ],
    )
    report = build_report(raw, kb)
    crit = report.critical_items
    assert len(crit) == 1
    assert crit[0].canonical_name == "wbc"
    assert crit[0].flag is Flag.CRITICAL_HIGH


def test_duplicate_items_are_warned(kb: KnowledgeBase) -> None:
    raw = RawLabReport(
        patient_sex="男",
        items=[
            RawLabItem(raw_name="WBC", result_raw="6.0"),
            RawLabItem(raw_name="白细胞", result_raw="6.1"),
        ],
    )
    report = build_report(raw, kb)
    assert any("重复" in w for w in report.warnings)


def test_missing_demographics_warns(kb: KnowledgeBase) -> None:
    report = build_report(RawLabReport(items=[RawLabItem(raw_name="WBC", result_raw="6.0")]), kb)
    assert any("性别" in w for w in report.warnings)
    assert any("年龄" in w for w in report.warnings)
