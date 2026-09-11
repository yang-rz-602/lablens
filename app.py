"""LabLens Streamlit 界面。

界面承担的不只是交互，还有三个**合规动作**（依据调研结论刻意设计进流程，而不是塞在页脚）：

1. **输入前**：必须勾选确认"仅用于科普、不构成医疗建议、请勿上传含姓名/就诊号的完整报告"
2. **输出顶部**：常驻横幅"AI 生成内容 · 仅供科普参考 · 不构成诊断或治疗建议"
3. **输出底部**：附判读依据、来源与"如有异常请咨询医生"

另外，界面**并列展示**每一项的：原始值 / 单位 / 标本类型 / 参考区间 / 区间来源 / 判定依据。
调研结论很明确——商业产品卖的是"解读的信任感"，而**可验证的透明性**是替代品；
把原始值和规则摊开来给用户看，是这个 demo 唯一诚实的说服力来源。

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
from core.schema import Flag, LabItem, LabReport, RefSource, Severity  # noqa: E402
from core.store import TrendStore  # noqa: E402

DATA_DIR = ROOT / "data"
SAMPLE_PATH = ROOT / "evals" / "synthetic" / "ground_truth.json"
SAMPLE_IMAGE_DIR = ROOT / "examples"

FLAG_BADGE = {
    Flag.CRITICAL_HIGH: "🔴 危急偏高",
    Flag.HIGH: "🟠 偏高",
    Flag.NORMAL: "🟢 正常",
    Flag.LOW: "🟠 偏低",
    Flag.CRITICAL_LOW: "🔴 危急偏低",
    Flag.UNKNOWN: "⚪ 未判定",
}

DISCLAIMER_BANNER = (
    "**AI 生成内容 · 仅供科普参考**　本工具不是医疗器械，不提供诊断、用药或治疗建议，"
    "不能替代执业医师的判断。"
)

st.set_page_config(page_title="LabLens · 检验报告解读", page_icon="🔬", layout="wide")


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
    st.sidebar.title("🔬 LabLens")
    st.sidebar.caption("中文检验报告智能体 · 技术演示")

    st.sidebar.markdown("### 模型")
    options = ["（不使用模型 · 离线抽取式解读）"] + [p.label for p in PRESETS.values()]
    keys = [None] + list(PRESETS)
    choice = st.sidebar.selectbox("模型来源", options, index=1, help="未配置 API Key 时会自动退回离线模式")
    provider_key = keys[options.index(choice)]

    api_key = ""
    model = ""
    if provider_key:
        preset = PRESETS[provider_key]
        api_key = st.sidebar.text_input(
            f"API Key（{preset.api_key_env}）", type="password",
            help="留空则使用服务端已配置的环境变量；两者都没有时回退到离线模式。",
        )
        if not api_key and os.environ.get(preset.api_key_env):
            st.sidebar.caption(f"✅ 已检测到服务端环境变量 `{preset.api_key_env}`")
        model = st.sidebar.text_input("模型名", value=preset.vision_model or preset.text_model)

    st.sidebar.markdown("### 数据")
    st.sidebar.checkbox(
        "启用本地趋势存储", value=False, key="enable_store",
        help="默认关闭：检验报告属于敏感个人信息，本工具默认不写入任何数据。"
             "开启后仅在你本机保存数值与日期，不含姓名、医院与报告原文。",
    )
    if st.session_state.get("enable_store"):
        st.sidebar.caption("⚠️ 已开启本地存储：数据仅写入本机 `lablens_trend.duckdb`，不会上传。")

    st.sidebar.markdown("---")
    kb, retriever = get_kb(), get_retriever()
    stats = retriever.stats()
    st.sidebar.caption(
        f"指标字典 **{len(kb)}** 项 · 解释语料 **{stats['chunks']}** 条 "
        f"（{stats['indicators']} 个指标）"
    )
    with st.sidebar.expander("知识库来源"):
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
# 判定结果展示（透明性核心）
# --------------------------------------------------------------------------- #
def item_row(item: LabItem) -> dict:
    """把一项结果摊平成一行——原始值、区间、来源、依据全部并列。"""
    return {
        "项目": item.raw_name,
        "结果": render_value(item),
        "单位": item.unit or "—",
        "标本": item.specimen or "—",
        "参考区间": render_ref(item),
        "区间来源": {
            RefSource.REPORT: "报告单",
            RefSource.KNOWLEDGE_BASE: "知识库",
            RefSource.NONE: "无",
        }[item.ref_source],
        "判定": FLAG_BADGE[item.flag],
        "依据": (item.decision_trace.interval_standard
                 if item.decision_trace and item.decision_trace.interval_standard else ""),
    }


def render_report_tab(report: LabReport) -> None:
    counts = report.summary_counts()

    cols = st.columns(5)
    cols[0].metric("检验项目", counts["total"])
    cols[1].metric("危急值", counts["CH"] + counts["CL"])
    cols[2].metric("偏高", counts["H"])
    cols[3].metric("偏低", counts["L"])
    cols[4].metric("未判定", counts["?"])

    if report.critical_items:
        names = "、".join(f"{i.raw_name} {render_value(i)}" for i in report.critical_items)
        st.error(
            f"⚠️ **检测到达到危急值阈值的项目**：{names}\n\n"
            "危急值意味着该结果可能需要立即的医疗关注。请尽快携带原始报告联系医生或前往医院，"
            "不要等待或自行处理。"
        )

    if report.warnings:
        with st.expander(f"⚠️ 数据质量提示（{len(report.warnings)} 条）", expanded=False):
            for w in report.warnings:
                st.markdown(f"- {w}")

    st.markdown("#### 指标明细与判读依据")
    st.caption("每一项都列出原始值、单位、标本类型、参考区间及其来源。**参考区间优先取自报告单本身。**")

    order = {Severity.CRITICAL: 0, Severity.ABNORMAL: 1, Severity.BORDERLINE: 2,
             Severity.UNKNOWN: 3, Severity.NORMAL: 4}
    rows = [item_row(i) for i in sorted(report.items, key=lambda x: order[x.severity])]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    abnormal = report.abnormal_items
    if abnormal:
        with st.expander("查看异常项的判定依据（逐条溯源）", expanded=False):
            for item in abnormal:
                badge = FLAG_BADGE[item.flag]
                st.markdown(
                    f"**{item.raw_name}** — {render_value(item)}　{badge}　"
                    f"参考区间 {render_ref(item)}"
                )
                if item.decision_trace:
                    t = item.decision_trace
                    st.caption(
                        f"规则：`{t.rule}`　区间来源：{t.interval_source or '无'}"
                        f"　依据标准：{t.interval_standard or '—'}　跨过：{t.crossed or '—'}"
                    )
                    st.caption(f"判定说明：{t.outcome}")
                if item.source_text:
                    st.code(item.source_text, language=None)
                for note in item.rules_notes:
                    st.caption(f"· {note}")
                st.markdown("---")


def render_interpretation_tab(result) -> None:
    interp = result.interpretation

    if interp.urgent_notice:
        st.error(f"🚨 {interp.urgent_notice}")

    if not result.used_llm:
        st.info(
            "当前为**离线模式**：解读由知识库原文抽取拼装而成（不存在幻觉风险），"
            "表述不如语言模型流畅。配置 API Key 后可获得更自然的解读。"
        )

    st.markdown("#### 整体印象")
    st.write(interp.overview)

    st.markdown("#### 逐项解读")
    for item in interp.items:
        with st.expander(f"{FLAG_BADGE[item.flag]}　{item.raw_name}　{item.value_display or ''}",
                         expanded=item.severity in {Severity.CRITICAL, Severity.ABNORMAL}):
            st.markdown(f"**这是什么**：{item.one_liner}")
            st.markdown(f"**本次结果**：{item.what_it_means}")
            if item.possible_factors:
                st.markdown("**常见相关因素**")
                for f in item.possible_factors:
                    st.markdown(f"- {f}")
            if item.source_text:
                st.caption(f"溯源原文：`{item.source_text}`")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 建议向医生提出的问题")
        for q in interp.questions_for_doctor:
            st.markdown(f"- {q}")
    with col2:
        st.markdown("#### 后续建议")
        for s in interp.next_steps:
            st.markdown(f"- {s}")

    if interp.limitations:
        with st.expander("本次解读的局限性（请务必阅读）", expanded=True):
            for lim in interp.limitations:
                st.markdown(f"- {lim}")

    if result.guard_hits:
        with st.expander(f"🛡️ 输出护栏已拦截 {len(result.guard_hits)} 处内容", expanded=False):
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


def render_trend_tab(report: LabReport) -> None:
    store = TrendStore(enabled=bool(st.session_state.get("enable_store")))
    if not store.enabled:
        st.info(
            "趋势存储默认关闭。\n\n"
            "检验报告属于《个人信息保护法》第 28 条的敏感个人信息，"
            "GB/T 39725-2020 亦要求健康医疗数据不宜存储在境外服务器。"
            "因此本工具**默认不写入任何数据**，进程退出即忘。\n\n"
            "如需追踪多次报告的趋势，请在左侧开启本地存储——数据只写入你本机文件。"
        )
        return

    saved = store.save(report)
    st.success(f"已保存 {saved} 项数值到本地（不含姓名、医院与报告原文）。")

    tracked = store.tracked_indicators()
    st.caption(f"本地共记录 {store.report_count()} 份报告、{len(tracked)} 个指标。")

    if tracked:
        labels = {f"{zh}（{canon}）": canon for canon, zh, _ in tracked if _ >= 1}
        pick = st.selectbox("选择指标查看趋势", list(labels))
        rows = store.as_rows(labels[pick])
        if len(rows) >= 2:
            st.line_chart({pick: [r["结果"] for r in rows]})
        st.dataframe(rows, use_container_width=True, hide_index=True)

    if st.button("清空本地趋势数据"):
        store.delete_all()
        st.success("已清空。")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = sidebar()
    kb, retriever = get_kb(), get_retriever()

    st.title("🔬 LabLens · 中文检验报告解读")
    st.caption(
        "数值判定 100% 由确定性代码完成，语言模型只负责把结果翻译成人话；"
        "**没有参考区间时系统拒绝判断**。"
    )
    st.warning(DISCLAIMER_BANNER)

    tab_input, tab_report, tab_interp, tab_trend, tab_about = st.tabs(
        ["① 输入报告", "② 指标与判读依据", "③ 解读", "④ 趋势", "⑤ 关于"]
    )

    with tab_input:
        st.markdown("#### 选择输入方式")
        mode = st.radio(
            "输入方式",
            ["使用内置合成示例（推荐，无需上传任何数据）", "上传报告图片", "粘贴报告文本"],
            horizontal=False,
        )

        acknowledged = st.checkbox(
            "我已阅读并理解：本工具仅用于科普参考，不构成诊断或治疗建议，不能替代医生判断。"
        )

        payload: dict | None = None
        uploaded_images = None
        pasted_text = None

        if mode.startswith("使用内置"):
            samples = get_samples()
            if not samples:
                st.error("未找到合成示例数据（evals/synthetic/ground_truth.json）。")
            else:
                labels = [
                    f"{c['case_id']} · {c.get('scenario', '')}（{c.get('report_type', '')}）"
                    for c in samples
                ]
                idx = st.selectbox("选择示例", range(len(labels)), format_func=lambda i: labels[i])
                payload = samples[idx]
                st.caption("合成数据由代码生成，不含任何真实个人信息。")

        elif mode.startswith("上传"):
            st.caption(
                "⚠️ 请先自行脱敏：**遮住姓名、就诊号、身份证号与条码**再上传。"
                "若使用云端模型，图片会发送到模型服务商。"
            )
            files = st.file_uploader(
                "上传检验报告图片", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True
            )
            if files:
                uploaded_images = [f.getvalue() for f in files]
                for f in files:
                    st.image(f, caption=f.name, width=380)

        else:
            pasted_text = st.text_area("粘贴报告文本", height=220,
                                       placeholder="白细胞计数  6.2  10^9/L  3.5-9.5\n血红蛋白    148  g/L     130-175")

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
            st.info("请先在「① 输入报告」中分析一份报告。")
        else:
            render_report_tab(report)

    with tab_interp:
        if interp_result is None:
            st.info("请先在「① 输入报告」中分析一份报告。")
        else:
            render_interpretation_tab(interp_result)
            st.markdown("---")
            st.caption(DISCLAIMER)

    with tab_trend:
        if report is None:
            st.info("请先分析一份报告。")
        else:
            render_trend_tab(report)

    with tab_about:
        render_about(kb, retriever)


def render_about(kb: KnowledgeBase, retriever: Retriever) -> None:
    st.markdown(
        f"""
### 这个工具做什么、不做什么

**做**：把检验报告上的数值与它自己的参考区间逐项对照，判定偏高/偏低/危急，
再把结果翻译成普通人能读懂的话，并给出建议向医生提问的清单。

**不做**：不诊断疾病、不给用药或剂量建议、不做处方相关内容、不出具任何医疗文书。

### 三条设计上的硬约束

1. **数值判定完全去 LLM 化**
   高低判断、危急值识别、单位换算全部由确定性代码完成，可单测、可复现、可审计。
   依据：Meyer 等（*Frontiers in AI* 2025）在 24.6 万个参考区间上测得大模型自报区间
   的下限变异系数达 **26.5%**；Lab-AI 实测 GPT-4-turbo 无检索时参考区间检索准确率
   **仅 42.9%**。

2. **没有参考区间就拒绝判断**
   参考区间优先取自报告单本身（不同医院仪器试剂不同，区间本就不同），
   知识库只做兜底；两者都没有时系统明确拒绝判定，而不是猜一个"正常范围"。
   Meyer 等原文即建议"在用户未提供参考区间时，AI 应拒绝进行实验室解读"。

3. **输出护栏是代码而非提示词**
   诊断结论、用药与剂量、处方相关表述由正则确定性拦截
   （《互联网诊疗监管细则（试行）》第 13、21 条），不依赖模型自觉。

### 当前知识库

- 指标字典：**{len(kb)}** 项，参考区间来源标注到具体卫生行业标准
- 解释语料：**{retriever.stats()['chunks']}** 条，按 `(指标, 维度)` 切分
- 检索方式：**约束检索**——先用规则引擎给出的指标名硬过滤，再在组内按维度优先级排序；
  检索不到可靠依据时返回"暂无依据"而不是硬编一段解释

### 评测

见 `evals/RESULTS.md`。头条指标是**危急值漏报率**而不是平均准确率——
依据 Ramaswamy 等（*Nature Medicine* 2026）：加入客观检验数据后总体准确率从 54.6%
升到 77.9%，但急症漏判率反而升到 56.2%。**只报平均分是在掩盖尾部风险。**

### 本工具的定位

**技术演示，非医疗器械。** 依据《人工智能医用软件产品分类界定指导原则》
（国家药监局 2021 年第 47 号通告），处理对象为"检验检查报告结论"这类非医疗器械数据、
且不用于医疗用途的软件，不作为医疗器械管理。本工具不面向公众提供医疗服务，
未取得也不申请任何医疗器械注册。
"""
    )


def run_pipeline(cfg, kb, retriever, payload, images, text) -> None:
    """跑完整流水线：抽取 → 判定 → 解读。"""
    client = make_client(cfg)

    if payload is not None:
        # 合成示例：直接从结构化数据构造，跳过抽取层
        report = sample_to_report(payload, kb)
        model_used = "synthetic"
        used_llm = client
    else:
        if client is None:
            st.error("上传图片或粘贴文本需要配置模型 API Key。可改用内置合成示例离线体验。")
            return
        from core.extract import extract_report

        result = extract_report(client, images=images, text=text)
        report = build_report(result.raw, kb, model_used=result.model_used)
        model_used = result.model_used
        used_llm = client

    interp = interpret(report, client=used_llm, retriever=retriever)

    st.session_state["report"] = report
    st.session_state["interp"] = interp
    st.session_state["model_used"] = model_used


def sample_to_report(case: dict, kb: KnowledgeBase) -> LabReport:
    """把合成示例转成 RawLabReport 再走正规流水线（不跳过规则引擎）。"""
    from core.schema import RawLabItem, RawLabReport

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
                source_text=f"{it.get('name_zh')}　{value if value is not None else it.get('value_text')}"
                            f"　{it.get('unit') or ''}　{it.get('ref_text') or ''}".strip(),
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
