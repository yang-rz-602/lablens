"""端到端界面测试（Streamlit AppTest）。

不需要浏览器：``streamlit.testing.v1.AppTest`` 在同进程里跑真实的 app 脚本，
可以直接断言"有没有报错""session_state 里有没有产出""危急值有没有被识别出来"。

对界面来说，"不抛异常"本身就是一条重要断言——Streamlit 的运行期异常
（例如某个控件参数在新版本里改了名）在静态检查里是看不出来的。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# 界面测试需要 streamlit；没装时干净跳过，而不是让整个测试套件报错
pytest.importorskip("streamlit", reason="界面测试需要 streamlit（pip install -e '.[app]'）")

from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
SAMPLES = json.loads((ROOT / "evals" / "synthetic" / "ground_truth.json").read_text(encoding="utf-8"))["cases"]

ACK_WORDING = "不构成诊断或治疗建议"


def _index_of(case_id: str) -> int:
    return next(i for i, c in enumerate(SAMPLES) if c["case_id"] == case_id)


@pytest.fixture()
def app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    return at


def _sample_selectbox(at: AppTest):
    return next(sb for sb in at.selectbox if sb.label == "选择示例")


def _analyze(at: AppTest, case_id: str) -> AppTest:
    """走一遍完整交互：选示例 → 勾选知情确认 → 开始分析。"""
    _sample_selectbox(at).set_value(_index_of(case_id))
    at.checkbox[0].check()
    at.run()
    at.button[0].click()
    at.run()
    return at


# --------------------------------------------------------------------------- #
# 冒烟
# --------------------------------------------------------------------------- #
def test_app_runs_without_exception(app: AppTest) -> None:
    assert not app.exception, [e.value for e in app.exception]


def test_has_all_expected_widgets(app: AppTest) -> None:
    assert len(app.radio) == 1
    assert app.radio[0].options == ["使用内置合成示例", "上传报告图片", "粘贴报告文本"]
    assert any(ACK_WORDING in cb.label for cb in app.checkbox), "缺少知情确认勾选框"
    assert any(b.label == "开始分析" for b in app.button)


def test_five_tabs_present(app: AppTest) -> None:
    labels = [t.label for t in app.tabs]
    for expect in ("输入报告", "指标与判读依据", "解读", "趋势", "关于"):
        assert any(expect in lb for lb in labels), f"缺少标签页 {expect}"


def test_sidebar_shows_knowledge_base_size(app: AppTest) -> None:
    text = "\n".join(m.value for m in app.markdown if isinstance(m.value, str))
    assert "指标字典" in text or "解释语料" in text


def test_analyze_is_gated_by_acknowledgement(app: AppTest) -> None:
    """未勾选知情确认时，按钮必须是禁用状态。

    AppTest 会拒绝点击禁用按钮（"Cannot update a disabled button widget"），
    所以直接断言 disabled，而不是尝试点击。
    """
    assert app.button[0].disabled is True
    assert "report" not in app.session_state


def test_empty_state_before_analysis(app: AppTest) -> None:
    assert "report" not in app.session_state
    assert not app.exception


# --------------------------------------------------------------------------- #
# 完整流程
# --------------------------------------------------------------------------- #
def test_full_offline_flow_produces_report_and_interpretation(app: AppTest) -> None:
    _analyze(app, "case_001")
    assert not app.exception, [e.value for e in app.exception]

    report = app.session_state["report"]
    interp = app.session_state["interp"]

    assert report is not None and len(report.items) >= 12
    assert interp is not None
    # 测试环境没有 API Key，必须自动降级到离线抽取式解读
    assert interp.used_llm is False
    assert interp.interpretation.disclaimer
    assert interp.interpretation.limitations


def test_sex_specific_items_judged_correctly(app: AppTest) -> None:
    """case_002 是女性血常规，血红蛋白必须按女性区间判。"""
    _analyze(app, "case_002")
    report = app.session_state["report"]
    hgb = next(i for i in report.items if i.canonical_name == "hgb")
    assert hgb.ref_source.value in {"report", "knowledge_base"}
    assert hgb.flag.value == "N"


@pytest.mark.parametrize(
    "case_id,canonical",
    [
        ("case_005", "plt"),
        ("case_007", "alt"),
        ("case_012", "k"),
        ("case_017", "ctni"),
    ],
)
def test_critical_cases_are_surfaced(app: AppTest, case_id: str, canonical: str) -> None:
    """4 个危急值 case 必须真的被判成危急值，并且在界面上有专门提示。"""
    _analyze(app, case_id)
    assert not app.exception
    report = app.session_state["report"]

    crit = {i.canonical_name for i in report.critical_items}
    assert canonical in crit, f"{case_id} 的 {canonical} 未判为危急值，实际 {crit}"
    assert report.critical_items


def test_knowledge_base_fills_missing_reference_interval(app: AppTest) -> None:
    """case_020 的 insulin 在原报告里没有印参考区间 → 必须由知识库兜底补上。

    这是设计的**第一优先级降级路径**：报告区间 > 知识库区间 > 拒判。
    只有在两边都没有时才拒判（那条路径由 test_rules.py 与 MCP 测试覆盖，
    因为内置示例里的指标都能在知识库里找到）。
    """
    _analyze(app, "case_020")
    assert not app.exception
    report = app.session_state["report"]
    insulin = next(i for i in report.items if i.canonical_name == "insulin")
    assert insulin.ref_source.value == "knowledge_base"
    assert insulin.ref_low is not None
    assert insulin.decision_trace.rule == "kb_interval"


def test_interpretation_reports_abstention_when_knowledge_missing(app: AppTest) -> None:
    _analyze(app, "case_001")
    interp = app.session_state["interp"]
    # 拒答信息必须能透传到界面（可能为空列表，但属性必须存在）
    assert isinstance(interp.abstained_names, list)
    assert isinstance(interp.guard_hits, list)


def test_output_guard_holds_for_every_sample(app: AppTest) -> None:
    """系统保证的是"输出通过护栏"，而不是"不含某个词"。

    知识库里会出现「小剂量阿司匹林等药物会干扰结果」这类正常的医学说明——
    它讲的是"哪些药影响检验"，不是给用户的用药建议，**不应该被拦**。
    所以这里用真正的护栏去检验输出，而不是维护一份关键词黑名单。
    """
    from core.guard import scan_text

    for case_id in ("case_007", "case_012", "case_020"):
        at = AppTest.from_file(str(APP), default_timeout=180)
        at.run()
        _analyze(at, case_id)
        interp = at.session_state["interp"].interpretation
        joined = " ".join(
            [interp.overview]
            + [f"{i.one_liner}{i.what_it_means}{''.join(i.possible_factors)}" for i in interp.items]
            + interp.next_steps
            + interp.questions_for_doctor
        )
        assert scan_text(joined) == [], f"{case_id} 的输出未通过合规护栏"

    # 对照组：护栏对真正的用药建议必须仍然拦截
    assert scan_text("建议服用阿司匹林100mg") != []
    assert scan_text("可以确诊为糖尿病") != []


# --------------------------------------------------------------------------- #
# 输入方式切换
# --------------------------------------------------------------------------- #
def test_switch_to_upload_mode_does_not_crash(app: AppTest) -> None:
    app.radio[0].set_value("上传报告图片")
    app.run()
    assert not app.exception
    assert any("脱敏" in c.value for c in app.caption), "上传模式应提示先脱敏"


def test_switch_to_text_mode_does_not_crash(app: AppTest) -> None:
    app.radio[0].set_value("粘贴报告文本")
    app.run()
    assert not app.exception
    assert app.text_area, "文本模式应有输入框"


def test_upload_without_api_key_shows_actionable_error(app: AppTest) -> None:
    """没有 Key 时上传图片应给出可执行的提示，而不是静默失败。"""
    app.radio[0].set_value("粘贴报告文本")
    app.run()
    app.text_area[0].set_value("白细胞计数 6.2 10^9/L 3.5-9.5")
    app.checkbox[0].check()
    app.run()
    app.button[0].click()
    app.run()
    assert not app.exception
    msgs = [e.value for e in app.error]
    assert any("API Key" in str(m) for m in msgs), msgs


# --------------------------------------------------------------------------- #
# 合规触点
# --------------------------------------------------------------------------- #
def test_acknowledgement_wording_is_required(app: AppTest) -> None:
    ack = next(cb for cb in app.checkbox if ACK_WORDING in cb.label)
    assert "科普" in ack.label
    assert "不能替代医生" in ack.label


def test_disclaimer_source_kept_in_sync() -> None:
    """界面横幅与 core 层免责声明必须都覆盖"非医疗器械 + 不构成诊断"。"""
    from core.explain import DISCLAIMER

    src = APP.read_text(encoding="utf-8")
    assert "DISCLAIMER_HTML" in src and "AI 生成内容" in src
    assert "不是医疗器械" in src or "非医疗器械" in src

    assert "不构成医学诊断" in DISCLAIMER
    assert "不构成用药" in DISCLAIMER
