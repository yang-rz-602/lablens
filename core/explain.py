"""Stage 3：解释层。

输入契约是 ``LabReport``——也就是说，**进入这一层的每个指标都已经被代码判定过了**。
模型在这里的角色是"翻译"，不是"判断"：把已经确定的结果翻译成普通人能读懂的话。

三条硬约束（提示词 + 代码双重保障）
-----------------------------------
1. 只能依据检索到的证据说话，检索不到就明说"暂无可靠依据"，不许自己补
2. 不出诊断结论，不出现"您患有/确诊为"
3. **不给用药建议、不写剂量、不做处方相关表述**（法规红线）

第 2、3 条由 ``core.guard`` 在输出后做正则拦截，不依赖模型自觉。

另外：当某项因为"报告未提供参考区间"而被规则引擎拒绝判定时，解释层必须**明确告知
用户本系统不做判定**，而不是含糊带过。这是 Meyer 等（2025）的建议落在产品上的样子。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .guard import GuardHit, sanitize_text
from .providers import LLMClient, LLMError
from .retriever import RetrievalResult, Retriever
from .schema import (
    Flag,
    Interpretation,
    ItemInterpretation,
    LabItem,
    LabReport,
    Severity,
)

__all__ = ["InterpretationResult", "interpret", "DISCLAIMER", "EXPLAIN_SYSTEM_PROMPT", "render_value"]

DISCLAIMER = (
    "本结果由自动化程序生成，仅用于帮助理解检验报告中的数值含义，"
    "**不构成医学诊断、不构成用药或治疗建议、不能替代执业医师的判断**。"
    "检验结果的临床意义必须结合病史、症状、体征及其它检查由医生综合判断。"
    "如结果异常或有身体不适，请及时就诊。"
)

EXPLAIN_SYSTEM_PROMPT = """\
你是一位耐心、严谨的医学科普人员，负责把检验报告里已经判定好的结果翻译成普通人能读懂的话。

**你的输入里已经包含了程序的判定结果（偏高/偏低/危急/正常）和参考区间。这些判定是确定的，
你不需要复核、不需要重新计算、绝对不要给出与输入不同的判定。**

必须严格遵守：

1. **不做诊断。** 禁止出现"确诊""诊断为""您患有""得了""提示您患有××病"这类表述。
   只能说"这项指标与××有关""常见相关因素包括……""需结合临床综合判断"。
2. **不给用药建议。** 禁止提及任何药物名称、剂量、停药、加药、处方、治疗方案。
   这是法规红线（《互联网诊疗监管细则（试行）》第 13、21 条）。
3. **只依据提供的证据。** 每一项的解释必须来自该项下方给出的【依据】。
   如果某项标注为"无可用依据"，你必须如实说明"知识库中暂无该指标的可靠解释依据"，
   **绝对不允许用你自己的医学知识补一段**。这是最重要的规则。
4. **不制造焦虑。** 描述异常时同时说明"该结果也可能由非疾病因素导致"，并给出与医生沟通的方向。
5. **对"未判定"要诚实。** 如果某项的判定是"未判定（缺少参考区间）"，你必须明确告诉用户
   本系统无法判断，建议以检验科出具的参考区间为准，不得给出任何倾向性猜测。
6. 语言通俗，避免堆砌术语；面向没有医学背景的读者。每项解释控制在 120 字以内。

输出 JSON，结构如下：
{
  "overview": "整体印象，2-3 句话。说明共几项、几项异常、是否有需要尽快就医的情况。",
  "items": [
    {
      "canonical_name": "必须与输入中该项的 canonical_name 完全一致",
      "one_liner": "这个指标是什么，一句话",
      "what_it_means": "本次结果的含义（基于给出的依据，不构成诊断）",
      "possible_factors": ["常见相关因素1", "常见相关因素2"]
    }
  ],
  "questions_for_doctor": ["建议向医生提出的具体问题，3-5 条"],
  "next_steps": ["后续建议，例如多久复查、挂什么科、带什么资料"]
}

只输出 JSON，不要解释文字，不要 Markdown 围栏。"""


# --------------------------------------------------------------------------- #
def render_value(item: LabItem) -> str:
    """把指标结果渲染成人类可读的一行。

    定性项目显示原文（"阳性"），而不是显示成"偏高"——这是常见的体验错误。
    """
    if item.value is not None:
        dec = 2 if (item.value and abs(item.value) < 10) else 1
        num = f"{item.value:.{dec}f}".rstrip("0").rstrip(".")
        return f"{num} {item.unit or ''}".strip()
    if item.value_text:
        return item.value_text
    return "—"


def render_ref(item: LabItem) -> str:
    if item.ref_low is not None and item.ref_high is not None:
        return f"{item.ref_low:g}–{item.ref_high:g}"
    if item.ref_high is not None:
        return f"< {item.ref_high:g}"
    if item.ref_low is not None:
        return f"> {item.ref_low:g}"
    return item.ref_text or "—"


@dataclass
class InterpretationResult:
    interpretation: Interpretation
    retrieval: dict[str, RetrievalResult]
    guard_hits: list[GuardHit]
    used_llm: bool
    model_used: str | None = None

    @property
    def abstained_names(self) -> list[str]:
        return [k for k, v in self.retrieval.items() if v.abstained]


# --------------------------------------------------------------------------- #
def _select_targets(report: LabReport, max_items: int) -> list[LabItem]:
    """挑选需要解读的指标：危急值 > 异常 > 临界 > 未判定，最后补少量正常项。"""
    order = {
        Severity.CRITICAL: 0,
        Severity.ABNORMAL: 1,
        Severity.BORDERLINE: 2,
        Severity.UNKNOWN: 3,
        Severity.NORMAL: 4,
    }
    ranked = sorted(report.items, key=lambda i: (order[i.severity], -(abs(i.deviation or 0))))
    targets = [i for i in ranked if i.severity is not Severity.NORMAL][:max_items]
    if len(targets) < min(3, max_items):
        targets += [i for i in ranked if i.severity is Severity.NORMAL][: max_items - len(targets)]
    return targets


def _retrieve_evidence(
    targets: list[LabItem], retriever: Retriever | None
) -> dict[str, RetrievalResult]:
    out: dict[str, RetrievalResult] = {}
    if retriever is None:
        return out
    for item in targets:
        key = item.canonical_name or item.raw_name
        if key in out:
            continue
        if not item.canonical_name:
            out[key] = RetrievalResult(
                key, abstained=True, reason="该指标未在指标字典中识别，无法检索依据。"
            )
            continue
        out[key] = retriever.retrieve(item.canonical_name, flag=item.flag.value)
    return out


def _evidence_block(item: LabItem, res: RetrievalResult | None) -> str:
    lines = [
        f"指标：{item.canonical_name or item.raw_name}（{item.raw_name}）",
        f"结果：{render_value(item)}",
        f"参考区间：{render_ref(item)}（来源：{'报告单' if item.ref_source.value == 'report' else '内置知识库' if item.ref_source.value == 'knowledge_base' else '无'}）",
        f"程序判定：{item.flag.label_zh}",
    ]
    if item.rules_notes:
        lines.append("判定说明：" + "；".join(item.rules_notes))
    if res is None:
        lines.append("【依据】未启用检索，无可用依据。")
    elif res.abstained:
        lines.append(f"【依据】无可用依据（{res.reason}）→ 本项必须如实说明暂无可靠依据。")
    else:
        lines.append("【依据】\n" + res.as_context())
    return "\n".join(lines)


def build_explain_messages(report: LabReport, targets: list[LabItem],
                           retrieval: dict[str, RetrievalResult]) -> list[dict]:
    blocks = [_evidence_block(i, retrieval.get(i.canonical_name or i.raw_name)) for i in targets]
    counts = report.summary_counts()
    header = (
        f"报告类型：{report.report_type or '未标注'}；"
        f"受检者：{report.patient.sex or '未知'}性，{report.patient.age if report.patient.age is not None else '未知'}岁；"
        f"共 {counts['total']} 项，其中危急 {counts['CH'] + counts['CL']} 项、"
        f"偏高 {counts['H']} 项、偏低 {counts['L']} 项、未判定 {counts['?']} 项。"
    )
    if report.warnings:
        header += "\n需要注意：" + "；".join(report.warnings)

    user = (
        f"{header}\n\n"
        "以下是需要解读的指标及其依据（判定结果已由程序确定，请勿改动）：\n\n"
        + "\n\n---\n\n".join(blocks)
        + "\n\n请按约定的 JSON 结构输出解读。"
    )
    return [
        {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- #
def _extractive_fallback(
    report: LabReport, targets: list[LabItem], retrieval: dict[str, RetrievalResult]
) -> dict[str, Any]:
    """无 LLM 时的降级路径：直接从检索到的证据里抽取式拼装解读。

    这条路径不存在幻觉风险（每一句都来自语料原文），只是语言不够流畅。
    它的存在让整条流水线在**没有任何 API Key** 的情况下依然可用、可评测。
    """
    items: list[dict[str, Any]] = []
    for item in targets:
        key = item.canonical_name or item.raw_name
        res = retrieval.get(key)
        one_liner = what = ""
        factors: list[str] = []
        if res and not res.abstained and res.hits:
            by_dim = {h.chunk.dimension: h.chunk.text for h in res.hits}
            one_liner = by_dim.get("definition", "")
            if item.flag in {Flag.HIGH, Flag.CRITICAL_HIGH}:
                what = by_dim.get("high_meaning", "")
            elif item.flag in {Flag.LOW, Flag.CRITICAL_LOW}:
                what = by_dim.get("low_meaning", "")
            if not what:
                what = by_dim.get("clinical_note", "") or one_liner
            factors = [h.chunk.text for h in res.hits if h.chunk.dimension == "factors"][:2]
        if item.flag is Flag.UNKNOWN:
            what = ("报告与知识库均未提供该项目的参考区间，本系统不做判定。"
                    "参考区间因仪器与试剂而异，请以检验科出具的区间为准。")
        items.append(
            {
                "canonical_name": item.canonical_name,
                "raw_name": item.raw_name,
                "one_liner": one_liner or f"{item.raw_name}是本次检验中的一个项目。",
                "what_it_means": what or "知识库中暂无该指标的可靠解释依据，本系统不提供推测性说明。",
                "possible_factors": factors,
            }
        )

    counts = report.summary_counts()
    abnormal = counts["CH"] + counts["CL"] + counts["H"] + counts["L"]
    overview = (
        f"本次共解读 {counts['total']} 项指标，其中判定为异常 {abnormal} 项"
        f"（偏高 {counts['H'] + counts['CH']} 项、偏低 {counts['L'] + counts['CL']} 项）"
        f"，未判定 {counts['?']} 项。"
    )
    if counts["CH"] + counts["CL"]:
        overview += "存在达到危急值阈值的项目，请尽快联系医生。"
    elif abnormal == 0:
        overview += "各项均在参考区间内。"

    return {
        "overview": overview,
        "items": items,
        "questions_for_doctor": [
            "这些异常项需要复查吗？间隔多久合适？",
            "结合我的既往史和用药情况，这些结果需要特别关注哪一项？",
            "是否需要补充其它检查来进一步明确？",
        ],
        "next_steps": (
            ["携带原始报告就诊，由医生结合症状与病史综合判断。"]
            + (["存在危急值项目，建议尽快就医。"] if counts["CH"] + counts["CL"] else [])
        ),
    }


# --------------------------------------------------------------------------- #
def interpret(
    report: LabReport,
    client: LLMClient | None = None,
    retriever: Retriever | None = None,
    max_items: int = 10,
    model: str | None = None,
) -> InterpretationResult:
    """生成解读。

    ``client`` 为 None 时自动走抽取式降级路径，整个流程仍然可用。
    """
    targets = _select_targets(report, max_items)
    retrieval = _retrieve_evidence(targets, retriever)

    used_llm = client is not None
    model_used: str | None = None
    payload: dict[str, Any] | None = None

    if client is not None:
        messages = build_explain_messages(report, targets, retrieval)
        try:
            payload = client.chat_json(messages, model=model)
            model_used = model or getattr(client, "model", None)
        except LLMError:
            payload = None
            used_llm = False

    if not payload:
        payload = _extractive_fallback(report, targets, retrieval)

    # ---------------- 合并 + 护栏 ---------------- #
    by_key: dict[str, dict] = {}
    for row in payload.get("items") or []:
        if isinstance(row, dict) and row.get("canonical_name"):
            by_key[str(row["canonical_name"])] = row

    guard_hits: list[GuardHit] = []
    item_interpretations: list[ItemInterpretation] = []

    for item in targets:
        key = item.canonical_name or item.raw_name
        row = by_key.get(str(item.canonical_name or ""), {})

        one_liner, hits1 = sanitize_text(str(row.get("one_liner") or ""))
        what, hits2 = sanitize_text(str(row.get("what_it_means") or ""))
        guard_hits += hits1 + hits2

        factors: list[str] = []
        for f in row.get("possible_factors") or []:
            clean, fhits = sanitize_text(str(f))
            guard_hits += fhits
            if clean:
                factors.append(clean)

        res = retrieval.get(key)
        if not what:
            if item.flag is Flag.UNKNOWN:
                what = ("报告与知识库均未提供该项目的参考区间，本系统不做判定，"
                        "请以检验科出具的区间为准。")
            elif res is not None and res.abstained:
                what = "知识库中暂无该指标的可靠解释依据，本系统不提供推测性说明。"
            else:
                what = "暂无可用说明。"

        item_interpretations.append(
            ItemInterpretation(
                raw_name=item.raw_name,
                canonical_name=item.canonical_name,
                flag=item.flag,
                severity=item.severity,
                value_display=f"{render_value(item)}（参考 {render_ref(item)}）",
                one_liner=one_liner or f"{item.raw_name}是本次检验中的一个项目。",
                what_it_means=what,
                possible_factors=factors,
                source_page=item.page,
                source_text=item.source_text,
            )
        )

    overview, gh = sanitize_text(str(payload.get("overview") or ""))
    guard_hits += gh

    questions: list[str] = []
    for q in payload.get("questions_for_doctor") or []:
        clean, qh = sanitize_text(str(q))
        guard_hits += qh
        if clean:
            questions.append(clean)

    next_steps: list[str] = []
    for s in payload.get("next_steps") or []:
        clean, sh = sanitize_text(str(s))
        guard_hits += sh
        if clean:
            next_steps.append(clean)

    # ---------------- 局限性说明（这是可信度的一部分） ---------------- #
    limitations: list[str] = []
    if not used_llm:
        limitations.append("本次未启用语言模型，解读由知识库原文拼装而成，表述可能不够连贯。")
    abstained = [k for k, v in retrieval.items() if v.abstained]
    if abstained:
        limitations.append(
            f"以下指标缺少可靠的解释依据，已按设计拒绝给出推测性解释：{'、'.join(abstained)}。"
        )
    unknown = [i.raw_name for i in report.items if i.flag is Flag.UNKNOWN]
    if unknown:
        limitations.append(
            f"以下项目因缺少参考区间而未做判定：{'、'.join(unknown[:8])}"
            f"{' 等' if len(unknown) > 8 else ''}。参考区间因仪器与试剂而异，请以原报告为准。"
        )
    limitations.append(
        "本工具只解读报告数值本身，无法获知您的病史、症状、用药与其它检查结果，"
        "因此不能据此判断健康状况。"
    )

    critical = report.critical_items
    urgent = None
    if critical:
        names = "、".join(f"{i.raw_name}（{render_value(i)}）" for i in critical)
        urgent = (
            f"检测到达到危急值阈值的项目：{names}。"
            "危急值意味着该结果可能需要立即的医疗关注，请**尽快**携带原始报告联系医生或前往医院，"
            "不要等待或自行处理。"
        )

    return InterpretationResult(
        interpretation=Interpretation(
            overview=overview or "已完成解读。",
            items=item_interpretations,
            questions_for_doctor=questions,
            next_steps=next_steps,
            urgent_notice=urgent,
            limitations=limitations,
            disclaimer=DISCLAIMER,
        ),
        retrieval=retrieval,
        guard_hits=guard_hits,
        used_llm=used_llm,
        model_used=model_used,
    )
