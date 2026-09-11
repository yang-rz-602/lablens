"""LabLens Streamlit 界面。

界面承担的不只是交互，还有三个**合规动作**（依据调研结论刻意设计进流程，而不是塞在页脚）：

1. **输入前**：必须勾选确认"仅用于科普、不构成医疗建议、请勿上传含姓名/就诊号的完整报告"
2. **输出顶部**：常驻横幅"AI 生成内容 · 仅供科普参考 · 不构成诊断或治疗建议"
3. **输出底部**：附判读依据、来源与"如有异常请咨询医生"

另外，界面**并列展示**每一项的：原始值 / 单位 / 标本类型 / 参考区间 / 区间来源 / 判读依据。
调研结论很明确——商业产品卖的是"解读的信任感"，而**可验证的透明性**是它的替代品；
把原始值和规则摊开给用户看，是这个 demo 唯一诚实的说服力来源。

样式与组件在 ``ui/`` 下（设计令牌 + 纯函数组件），本文件只负责编排。

启动::

    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.explain import DISCLAIMER, interpret, render_ref, render_value  # noqa: E402
from core.providers import PRESETS, LLMError, OpenAICompatClient  # noqa: E402
from core.retriever import Retriever, load_corpus  # noqa: E402
from core.rules import KnowledgeBase, build_report, load_knowledge_base  # noqa: E402
from core.schema import (  # noqa: E402
    LabItem,
    LabReport,
    RawLabItem,
    RawLabReport,
    Severity,
)
from core.store import TrendStore  # noqa: E402
from ui import (  # noqa: E402
    critical_alert,
    disclaimer,
    empty,
    hero,
    inject_css,
    item_card,
    item_table,
    note,
    section,
    severity_of,
    stat_cards,
)

DATA_DIR = ROOT / "data"
SAMPLE_PATH = ROOT / "evals" / "synthetic" / "ground_truth.json"

CHIPS = [
    "数值判定 100% 确定性代码",
    "没有参考区间就拒判",
    "字段级溯源",
    "约束检索 + 拒答",
    "输出护栏（代码级）",
]

DISCLAIMER_HTML = (
    "<b>AI 生成内容 · 仅供科普参考</b>　本工具不是医疗器械，"
    "不提供诊断、用药或治疗建议，不能替代执业医师的判断。"
)

st.set_page_config(
    page_title="LabLens · 检验报告解读",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# --------------------------------------------------------------------------- #
# 资源（缓存）
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_kb() -> KnowledgeBase:
    return load_knowledge_base(DATA_DIR)


@st.cache_resource(show_spinner=False)
def get_retriever() -> Retriever:
    return Retriever(load_corpus(DATA_DIR / "corpus.jsonl"))


@st.cache_data(show_spinner=False)
def get_samples() -> list[dict]:
    if not SAMPLE_PATH.is_file():
        return []
    return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))["cases"]


# --------------------------------------------------------------------------- #
# 侧栏
# --------------------------------------------------------------------------- #
def sidebar() -> dict:
    with st.sidebar:
        st.markdown(
            '<div style="font-size:1.16rem;font-weight:680;letter-spacing:-.01em">'
            '🔬 LabLens</div>'
            '<div style="font-size:.78rem;color:#94A3B8;margin-bottom:1rem">'
            "中文检验报告智能体 · 技术演示</div>",
            unsafe_allow_html=True,
        )

        st.markdown("##### 模型")
        options = ["（不使用模型 · 离线抽取式解读）"] + [p.label for p in PRESETS.values()]
        keys: list[str | None] = [None] + list(PRESETS)
        choice = st.selectbox(
            "模型来源", options, index=1,
            help="未配置 API Key 时会自动退回离线模式——判读与检索不依赖模型，依然完整可用。",
            label_visibility="collapsed",
        )
        provider_key = keys[options.index(choice)]

        api_key = ""
        model = ""
        if provider_key:
            preset = PRESETS[provider_key]
            api_key = st.text_input(
                "API Key", type="password",
                placeholder=f"留空则用服务端的 {preset.api_key_env}",
                help="留空则使用服务端已配置的环境变量；两者都没有时回退到离线模式。",
            )
            if not api_key and os.environ.get(preset.api_key_env):
                st.caption(f"✅ 已检测到服务端环境变量 `{preset.api_key_env}`")
            model = st.text_input("模型名", value=preset.vision_model or preset.text_model)

        st.divider()
        st.markdown("##### 数据")
        st.checkbox(
            "启用本地趋势存储", value=False, key="enable_store",
            help="默认关闭：检验报告属于敏感个人信息，本工具默认不写入任何数据。"
                 "开启后仅在你本机保存数值与日期，不含姓名、医院与报告原文。",
        )
        if st.session_state.get("enable_store"):
            st.caption("⚠️ 已开启：仅写入本机 `lablens_trend.duckdb`，不上传。")

        st.divider()
        kb, retriever = get_kb(), get_retriever()
        stats = retriever.stats()
        st.markdown(
            f'<div style="font-size:.8rem;color:#475569;line-height:1.9">'
            f"指标字典 <b>{len(kb)}</b> 项<br>"
            f"解释语料 <b>{stats['chunks']}</b> 条 / {stats['indicators']} 个指标<br>"
            f"检索模式 <code>{stats['mode']}</code></div>",
            unsafe_allow_html=True,
        )
        with st.expander("知识库来源"):
            st.markdown(
                "- 参考区间：**WS/T 405-2012**（血细胞分析）、**WS/T 404 系列**（生化）等卫生行业标准\n"
                "- 解释语料：临床检验操作规程与检验医学教材整理\n"
                "- 编码标准：LOINC / UCUM\n\n"
                "⚠️ 各实验室仪器与试剂不同，参考区间本就不同。"
                "**本工具优先使用报告单自带区间，知识库仅作兜底。**"
            )

    return {"provider_key": provider_key, "api_key": api_key, "model": model}


def make_client(cfg: dict):
    """构造 LLM 客户端；失败或没有 Key 时返回 None（走离线模式）。

    凭证优先级：**界面上填的 > 服务端环境变量**。
    后者是部署到 ModelScope / Hugging Face 等平台时必须支持的路径——
    平台把 secrets 以环境变量注入，如果只读界面输入框，配了 secret 也不会生效。
    """
    if not cfg["provider_key"]:
        return None
    preset = PRESETS[cfg["provider_key"]]
    api_key = cfg["api_key"] or os.environ.get(preset.api_key_env, "")
    if not api_key:
        return None
    try:
        return OpenAICompatClient.from_preset(
            cfg["provider_key"], api_key=api_key, model=cfg["model"] or None
        )
    except LLMError:
        return None


# --------------------------------------------------------------------------- #
# 展示用的行构造
# --------------------------------------------------------------------------- #
def _basis(item: LabItem) -> str:
    t = item.decision_trace
    return t.interval_standard if t and t.interval_standard else ""


def _row(item: LabItem) -> dict:
    """把一项结果摊平——原始值、区间、来源、依据全部并列。"""
    return {
        "name": item.raw_name,
        "value": render_value(item),
        "unit": item.unit,
        "specimen": item.specimen,
        "ref": render_ref(item),
        "ref_source": item.ref_source,
        "flag_key": item.flag.value,
        "flag_text": item.value_text if item.value_text else None,
        "basis": _basis(item),
    }


def _trace_html(item: LabItem) -> str:
    from ui.components import esc

    t = item.decision_trace
    if t is None:
        return ""
    parts = [
        f"规则 <code>{esc(t.rule)}</code>",
        f"区间来源 {esc(t.interval_source or '无')}",
    ]
    if t.interval_standard:
        parts.append(f"依据标准 <code>{esc(t.interval_standard)}</code>")
    if t.interval_used:
        parts.append(f"使用区间 <code>{esc(t.interval_used)}</code>")
    if t.crossed:
        parts.append(f"跨过 <code>{esc(t.crossed)}</code>")
    html_out = "　·　".join(parts)
    if t.outcome:
        html_out += f"<br>{esc(t.outcome)}"
    return html_out


# --------------------------------------------------------------------------- #
# 各标签页
# --------------------------------------------------------------------------- #
def render_report_tab(report: LabReport) -> None:
    counts = report.summary_counts()

    stat_cards([
        {"k": "检验项目", "v": counts["total"], "tone": "unknown"},
        {"k": "危急值", "v": counts["CH"] + counts["CL"], "tone": "critical"},
        {"k": "偏高", "v": counts["H"], "tone": "abnormal"},
        {"k": "偏低", "v": counts["L"], "tone": "abnormal"},
        {"k": "未判定", "v": counts["?"], "tone": "unknown"},
    ])

    if report.critical_items:
        from ui.components import esc

        names = "　".join(
            f'<code>{esc(i.raw_name)} {esc(render_value(i))} {esc(i.unit or "")}</code>'
            for i in report.critical_items
        )
        critical_alert(names)

    if report.warnings:
        with st.expander(f"⚠️ 数据质量提示（{len(report.warnings)} 条）"):
            for w in report.warnings:
                st.markdown(f"- {w}")

    section("指标明细", "每一项都列出原始值、单位、标本类型、参考区间及其来源")
    order = {
        Severity.CRITICAL: 0, Severity.ABNORMAL: 1, Severity.BORDERLINE: 2,
        Severity.UNKNOWN: 3, Severity.NORMAL: 4,
    }
    rows = [_row(i) for i in sorted(report.items, key=lambda x: order[x.severity])]
    item_table(rows)

    with st.expander("以可排序表格查看 / 导出"):
        st.dataframe(rows, width="stretch", hide_index=True)

    abnormal = report.abnormal_items
    if abnormal:
        section("异常项的判读依据", "逐条溯源：用了哪条区间、来自哪里、跨过哪个界")
        for item in abnormal:
            item_card({
                "name": item.raw_name,
                "value": render_value(item),
                "ref": render_ref(item),
                "flag_key": item.flag.value,
                "flag_text": item.value_text if item.value_text else None,
                "tone": severity_of(item),
                "what": "；".join(item.rules_notes) if item.rules_notes else "—",
                "trace": _trace_html(item),
                "source_text": item.source_text,
            })


def render_interpretation_tab(result) -> None:
    interp = result.interpretation

    if interp.urgent_notice:
        from ui.components import esc
        from ui.components import render as _render

        _render(
            f'<div class="ll-alert critical" style="margin-top:.2rem">'
            f'<div class="ll-tx"><span class="ll-title">🚨 需要尽快处理</span>'
            f"{esc(interp.urgent_notice)}</div></div>"
        )

    if not result.used_llm:
        note(
            "当前为离线模式：解读由知识库原文抽取拼装而成，不存在幻觉风险，"
            "表述不如语言模型流畅。配置 API Key 后可获得更自然的解读。"
        )

    section("整体印象")
    st.markdown(interp.overview)

    section("逐项解读", f"共 {len(interp.items)} 项")
    for row in interp.items:
        item_card({
            "name": row.raw_name,
            "value": row.value_display or "",
            "ref": "",
            "flag_key": row.flag.value,
            "flag_text": None,
            "tone": {
                Severity.CRITICAL: "critical", Severity.ABNORMAL: "abnormal",
                Severity.BORDERLINE: "borderline", Severity.NORMAL: "normal",
                Severity.UNKNOWN: "unknown",
            }[row.severity],
            "one_liner": row.one_liner,
            "what": row.what_it_means,
            "factors": row.possible_factors,
            "source_text": row.source_text,
        })

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        section("建议向医生提出的问题")
        for q in interp.questions_for_doctor:
            st.markdown(f"- {q}")
    with col2:
        section("后续建议")
        for s in interp.next_steps:
            st.markdown(f"- {s}")

    if interp.limitations:
        with st.expander("本次解读的局限性（请务必阅读）", expanded=True):
            for lim in interp.limitations:
                st.markdown(f"- {lim}")

    if result.guard_hits:
        with st.expander(f"🛡️ 输出护栏已拦截 {len(result.guard_hits)} 处内容"):
            st.caption(
                "以下内容命中合规护栏（涉及诊断结论、用药或剂量），已从输出中剔除。"
                "依据《互联网诊疗监管细则（试行）》第 13、21 条。"
            )
            for h in result.guard_hits:
                st.markdown(f"- `{h.category}`：{h.sentence}")

    if result.abstained_names:
        st.warning(
            "以下指标缺少可靠解释依据，系统按设计**拒绝给出推测性解释**："
            + "、".join(result.abstained_names)
        )

    st.divider()
    st.caption(DISCLAIMER)


def render_trend_tab(report: LabReport) -> None:
    store = TrendStore(enabled=bool(st.session_state.get("enable_store")))
    if not store.enabled:
        note(
            "趋势存储默认关闭。检验报告属于《个人信息保护法》第 28 条的敏感个人信息，"
            "GB/T 39725-2020 亦要求健康医疗数据不宜存储在境外服务器。"
            "因此本工具默认不写入任何数据，进程退出即忘。"
            "如需追踪多次报告的趋势，请在左侧开启本地存储——数据只写入你本机文件。"
        )
        return

    saved = store.save(report)
    st.success(f"已保存 {saved} 项数值到本地（不含姓名、医院与报告原文）。")
    tracked = store.tracked_indicators()
    st.caption(f"本地共记录 {store.report_count()} 份报告、{len(tracked)} 个指标。")

    if tracked:
        labels = {f"{zh}（{canon}）": canon for canon, zh, n in tracked if n >= 1}
        pick = st.selectbox("选择指标查看趋势", list(labels))
        rows = store.as_rows(labels[pick])
        if len(rows) >= 2:
            st.line_chart({pick: [r["结果"] for r in rows]})
        st.dataframe(rows, width="stretch", hide_index=True)

    if st.button("清空本地趋势数据"):
        store.delete_all()
        st.success("已清空。")


def render_about_tab(kb: KnowledgeBase, retriever: Retriever) -> None:
    st.markdown(
        """
### 这个工具做什么、不做什么

**做**：把检验报告上的数值与它自己的参考区间逐项对照，判定偏高/偏低/危急，
再把结果翻译成普通人能读懂的话，并给出建议向医生提问的清单。

**不做**：不诊断疾病、不给用药或剂量建议、不做处方相关内容、不出具任何医疗文书。
"""
    )
    section("三条设计上的硬约束")
    st.markdown(
        """
**1. 数值判定完全去 LLM 化**
高低判断、危急值识别、单位换算全部由确定性代码完成，可单测、可复现、可审计。
依据：Meyer 等（*Frontiers in AI* 2025）在 24.6 万个参考区间上测得大模型自报区间的
下限变异系数达 **26.5%**；Lab-AI 实测 GPT-4-turbo 无检索时参考区间检索准确率
**仅 42.9%**。

**2. 没有参考区间就拒绝判断**
参考区间优先取自报告单本身（不同医院仪器试剂不同，区间本就不同），知识库只做兜底；
两者都没有时系统明确拒绝判定，而不是猜一个"正常范围"。
性别分层指标缺少性别信息时同样拒判——用任一性别的区间都会产生静默错误。

**3. 输出护栏是代码而非提示词**
诊断结论、用药与剂量、处方相关表述由正则确定性拦截
（《互联网诊疗监管细则（试行）》第 13、21 条），不依赖模型自觉。
"""
    )

    section("当前知识库")
    stats = retriever.stats()
    st.markdown(
        f"""
- 指标字典：**{len(kb)}** 项，参考区间来源标注到具体卫生行业标准
- 解释语料：**{stats['chunks']}** 条，按 `(指标, 维度)` 切分
- 检索方式：**约束检索**——先用规则引擎给出的指标名硬过滤，再在组内按维度优先级排序；
  检索不到可靠依据时返回"暂无依据"而不是硬编一段解释
"""
    )

    section("评测")
    st.markdown(
        """
见 `evals/RESULTS.md`。头条指标是**危急值漏报率**而不是平均准确率——
依据 Ramaswamy 等（*Nature Medicine* 2026）：加入客观检验数据后总体准确率从 54.6%
升到 77.9%，但急症漏判率反而升到 56.2%。**只报平均分是在掩盖尾部风险。**

当前：生产路径准确率 **100.0%**，危急值漏报 **0** 项（20 份合成报告 / 276 项）。
"""
    )

    section("定位")
    st.markdown(
        """
**技术演示，非医疗器械。** 依据《人工智能医用软件产品分类界定指导原则》
（国家药监局 2021 年第 47 号通告），处理对象为"检验检查报告结论"这类非医疗器械数据、
且不用于医疗用途的软件，不作为医疗器械管理。本工具不面向公众提供医疗服务，
未取得也不申请任何医疗器械注册。
"""
    )


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = sidebar()
    kb, retriever = get_kb(), get_retriever()

    hero(
        "LabLens · 中文检验报告解读",
        "数值判定 100% 由确定性代码完成，语言模型只负责把结果翻译成人话；"
        "没有参考区间时，系统拒绝判断。",
        CHIPS,
    )
    disclaimer(DISCLAIMER_HTML)

    tab_input, tab_report, tab_interp, tab_trend, tab_about = st.tabs(
        ["① 输入报告", "② 指标与判读依据", "③ 解读", "④ 趋势", "⑤ 关于"]
    )

    with tab_input:
        section("选择输入方式")
        mode = st.radio(
            "输入方式",
            ["使用内置合成示例", "上传报告图片", "粘贴报告文本"],
            horizontal=True,
            label_visibility="collapsed",
            captions=[
                "推荐：无需上传任何真实数据",
                "需配置模型 API Key",
                "需配置模型 API Key",
            ],
        )

        acknowledged = st.checkbox(
            "我已阅读并理解：本工具仅用于科普参考，不构成诊断或治疗建议，不能替代医生判断。"
        )

        payload: dict | None = None
        uploaded_images = None
        pasted_text = None

        if mode == "使用内置合成示例":
            samples = get_samples()
            if not samples:
                st.error("未找到合成示例数据（evals/synthetic/ground_truth.json）。")
            else:
                labels = [
                    f"{c['case_id']} · {c.get('scenario', '')}（{c.get('report_type', '')}）"
                    for c in samples
                ]
                idx = st.selectbox(
                    "选择示例", range(len(labels)), format_func=lambda i: labels[i]
                )
                payload = samples[idx]
                st.caption("合成数据由代码生成，不含任何真实个人信息。")
        elif mode == "上传报告图片":
            st.caption(
                "⚠️ 请先自行脱敏：**遮住姓名、就诊号、身份证号与条码**再上传。"
                "若使用云端模型，图片会发送到模型服务商。"
            )
            files = st.file_uploader(
                "上传检验报告图片", type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
            )
            if files:
                uploaded_images = [f.getvalue() for f in files]
                cols = st.columns(min(len(files), 3))
                for i, f in enumerate(files):
                    cols[i % len(cols)].image(f, caption=f.name, width="stretch")
        else:
            pasted_text = st.text_area(
                "粘贴报告文本", height=200,
                placeholder="白细胞计数  6.2  10^9/L  3.5-9.5\n血红蛋白    148  g/L     130-175",
            )

        st.write("")
        analyze = st.button("开始分析", type="primary", disabled=not acknowledged)
        if not acknowledged:
            st.caption("请先勾选上方的知情确认。")

        if analyze:
            with st.spinner("正在解析与判定…"):
                try:
                    run_pipeline(cfg, kb, retriever, payload, uploaded_images, pasted_text)
                except LLMError as exc:
                    st.error(f"模型调用失败：{exc}")
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)
            if st.session_state.get("report") is not None:
                st.success("分析完成，请查看「② 指标与判读依据」和「③ 解读」。")

    report: LabReport | None = st.session_state.get("report")
    interp_result = st.session_state.get("interp")

    with tab_report:
        if report is None:
            empty("🧪", "请先在「① 输入报告」中分析一份报告")
        else:
            render_report_tab(report)

    with tab_interp:
        if interp_result is None:
            empty("📋", "请先在「① 输入报告」中分析一份报告")
        else:
            render_interpretation_tab(interp_result)

    with tab_trend:
        if report is None:
            empty("📈", "请先分析一份报告")
        else:
            render_trend_tab(report)

    with tab_about:
        render_about_tab(kb, retriever)


# --------------------------------------------------------------------------- #
def run_pipeline(cfg, kb, retriever, payload, images, text) -> None:
    """跑完整流水线：抽取 → 判定 → 解读。"""
    client = make_client(cfg)

    if payload is not None:
        report = sample_to_report(payload, kb)
        model_used = "synthetic"
    else:
        if client is None:
            st.error("上传图片或粘贴文本需要配置模型 API Key。可改用内置合成示例离线体验。")
            return
        from core.extract import extract_report

        result = extract_report(client, images=images, text=text)
        report = build_report(result.raw, kb, model_used=result.model_used)
        model_used = result.model_used

    interp = interpret(report, client=client, retriever=retriever)

    st.session_state["report"] = report
    st.session_state["interp"] = interp
    st.session_state["model_used"] = model_used


def sample_to_report(case: dict, kb: KnowledgeBase) -> LabReport:
    """把合成示例转成 RawLabReport 再走正规流水线（不跳过规则引擎）。"""
    patient = case.get("patient") or {}
    items = []
    for it in case["items"]:
        value = it.get("value")
        items.append(
            RawLabItem(
                raw_name=it.get("name_zh") or it["canonical_name"],
                result_raw=f"{value:g}" if value is not None else str(it.get("value_text") or ""),
                unit_raw=it.get("unit"),
                ref_raw=it.get("ref_text"),
                page=1,
                source_text=(
                    f"{it.get('name_zh')}　{value if value is not None else it.get('value_text')}"
                    f"　{it.get('unit') or ''}　{it.get('ref_text') or ''}"
                ).strip(),
            )
        )
    raw = RawLabReport(
        report_type=case.get("report_type"),
        hospital=case.get("hospital"),
        collected_at=case.get("collected_at"),
        patient_sex=patient.get("sex"),
        patient_age=patient.get("age"),
        items=items,
    )
    return build_report(raw, kb, model_used="synthetic")


if __name__ == "__main__":
    main()
