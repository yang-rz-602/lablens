"""LabLens 数据模型。

三层数据契约（全部用 Pydantic 强校验）：

1. ``RawLabItem`` / ``RawLabReport`` —— LLM 抽取层的输出契约。
   只允许包含"报告上能直接看到的东西"，不允许模型自行推断或计算。
2. ``LabItem`` / ``LabReport`` —— 规则引擎处理后的领域对象。
   在原始字段之上追加 H/L 判定、危急值等级、偏离度、参考区间来源等**由代码计算**的字段。
3. ``Interpretation`` —— 解释层输出契约。输入只能是已判定的 ``LabReport``。

设计原则：LLM 只负责**抽取**和**解释**，数值比较与阈值判断 100% 由 Python 完成。
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "Flag",
    "Severity",
    "RefSource",
    "DecisionTrace",
    "PatientContext",
    "RawLabItem",
    "RawLabReport",
    "LabItem",
    "LabReport",
    "ItemInterpretation",
    "Interpretation",
    "EXTRACTION_JSON_SCHEMA",
]


# --------------------------------------------------------------------------- #
# 枚举
# --------------------------------------------------------------------------- #
class Flag(str, Enum):
    """单个指标的判定结果。"""

    CRITICAL_HIGH = "CH"  # 危急值-偏高
    HIGH = "H"  # 偏高
    NORMAL = "N"  # 正常
    LOW = "L"  # 偏低
    CRITICAL_LOW = "CL"  # 危急值-偏低
    UNKNOWN = "?"  # 无法判定（缺参考区间 / 非数值结果）

    @property
    def label_zh(self) -> str:
        return {
            "CH": "危急偏高",
            "H": "偏高",
            "N": "正常",
            "L": "偏低",
            "CL": "危急偏低",
            "?": "未判定",
        }[self.value]


class Severity(str, Enum):
    """用于排序和前端着色的严重度分级。"""

    CRITICAL = "critical"
    ABNORMAL = "abnormal"
    BORDERLINE = "borderline"
    NORMAL = "normal"
    UNKNOWN = "unknown"


class RefSource(str, Enum):
    """参考区间的来源，决定了结果的可信度与展示方式。"""

    REPORT = "report"  # 报告单自带的参考区间（最可信，优先使用）
    KNOWLEDGE_BASE = "knowledge_base"  # 内置知识库补全（次之）
    NONE = "none"


# --------------------------------------------------------------------------- #
# 患者上下文
# --------------------------------------------------------------------------- #
class PatientContext(BaseModel):
    """判定参考区间所必需的上下文。

    很多项目的参考区间按性别、年龄分层（如血红蛋白、肌酐、尿酸、ALP），
    缺少这部分信息时规则引擎会退化为"只用报告自带区间"。
    """

    sex: str | None = Field(default=None, description="M=男 / F=女 / None=未知")
    age: int | None = Field(default=None, ge=0, le=130)

    @field_validator("sex")
    @classmethod
    def _norm_sex(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = str(v).strip().upper()
        if s in {"M", "MALE", "男", "男性"}:
            return "M"
        if s in {"F", "FEMALE", "女", "女性"}:
            return "F"
        return None


# --------------------------------------------------------------------------- #
# 第 1 层：LLM 抽取契约
# --------------------------------------------------------------------------- #
class RawLabItem(BaseModel):
    """LLM 从报告图像/文本中抽取出的**单个**检验项目。

    字段名刻意用 ``result_raw`` 而不是 ``value``，提醒抽取层：这里放的是
    报告上原样印出来的东西，不是算出来的东西。
    """

    raw_name: str = Field(description="报告上原样印出的项目名称")
    result_raw: str | None = Field(
        default=None, description="报告上原样印出的结果，如 '13.2'、'阴性'、'±'、'<0.01'"
    )
    unit_raw: str | None = Field(default=None, description="报告上原样印出的单位")
    ref_raw: str | None = Field(
        default=None, description="报告上原样印出的参考区间，如 '3.5-9.5'、'阴性'、'<5.0'"
    )
    flag_raw: str | None = Field(
        default=None, description="报告自带的异常标记（↑ ↓ H L + 等），没有则为 None"
    )
    page: int | None = Field(default=None, description="该项出现在第几页（从 1 开始）")
    source_text: str | None = Field(
        default=None, description="该项所在行的原始文本，用于人工核对与溯源"
    )


class RawLabReport(BaseModel):
    """LLM 抽取层的完整输出契约。"""

    report_type: str | None = Field(
        default=None, description="报告类型，如 '血常规'、'生化全套'、'尿常规'"
    )
    hospital: str | None = Field(default=None, description="出具机构名称")
    collected_at: str | None = Field(
        default=None, description="采样/检验日期，ISO 格式 YYYY-MM-DD，无法确定则 None"
    )
    patient_sex: str | None = Field(default=None, description="报告上标注的性别")
    patient_age: int | None = Field(default=None, description="报告上标注的年龄，无法确定则 None")
    items: list[RawLabItem] = Field(default_factory=list, description="抽取到的全部检验项目")
    extraction_notes: list[str] = Field(
        default_factory=list, description="抽取过程中遇到的模糊/存疑之处，供人工复核"
    )


# --------------------------------------------------------------------------- #
# 第 2 层：规则引擎处理后的领域对象
# --------------------------------------------------------------------------- #
class DecisionTrace(BaseModel):
    """判读依据的结构化记录。

    回答的是一个审计问题：**"这个判定结论是怎么来的？"**

    只给结论不给依据的系统在医疗场景不可用——用户无法判断该不该信，也无法在
    结论可疑时定位问题。把"用了哪条区间、来自哪里、跨过的是哪个界"结构化记录下来，
    是让自动化判读变得**可审计**的最小代价。
    """

    rule: str = Field(
        description="report_interval / kb_interval / qualitative / "
        "refused_no_interval / refused_unit_mismatch / refused_no_value"
    )
    interval_source: str | None = Field(default=None, description="报告单 / 内置知识库")
    interval_standard: str | None = Field(default=None, description="区间依据的标准，如 WS/T 405-2012")
    interval_used: str | None = Field(default=None, description="实际参与判定的区间，如 3.5-9.5 10^9/L")
    crossed: str | None = Field(
        default=None, description="跨过的是哪个界：critical_high / ref_high / ref_low / critical_low / within"
    )
    outcome: str = Field(default="", description="判定结果的一句话说明")
    notes: list[str] = Field(default_factory=list)


class LabItem(BaseModel):
    """规则引擎处理后的单个指标。

    ``flag`` / ``severity`` / ``deviation`` 三个字段**永远由代码写入**，
    抽取层和解释层都不得修改它们。
    """

    raw_name: str
    canonical_name: str | None = None
    loinc: str | None = None
    category: str | None = None
    specimen: str | None = Field(
        default=None, description="标本类型（全血/血清/血浆/尿液），参考区间与它强相关"
    )

    value: float | None = None
    value_text: str | None = None  # 定性结果：阴性 / 阳性 / ±
    unit: str | None = None

    ref_low: float | None = None
    ref_high: float | None = None
    ref_text: str | None = None
    ref_source: RefSource = RefSource.NONE

    flag_reported: str | None = None

    # --- 以下由 core.rules 计算 ---
    flag: Flag = Flag.UNKNOWN
    severity: Severity = Severity.UNKNOWN
    deviation: float | None = Field(
        default=None,
        description="相对偏离度：超出上限时 (值-上限)/上限，低于下限时 (下限-值)/下限",
    )
    rules_notes: list[str] = Field(default_factory=list, description="规则引擎给出的判定说明")
    decision_trace: DecisionTrace | None = Field(
        default=None, description="结构化的判读依据，支持逐条审计"
    )

    # --- 溯源 ---
    page: int | None = None
    source_text: str | None = None


class LabReport(BaseModel):
    """一份完整的检验报告。"""

    report_id: str
    report_type: str | None = None
    hospital: str | None = None
    collected_at: date | None = None
    patient: PatientContext = Field(default_factory=PatientContext)
    items: list[LabItem] = Field(default_factory=list)

    extracted_at: datetime = Field(default_factory=datetime.now)
    model_used: str | None = None
    warnings: list[str] = Field(default_factory=list)

    # ---------------- 便捷视图 ---------------- #
    @property
    def abnormal_items(self) -> list[LabItem]:
        """所有非正常项，按严重度与偏离度排序（危急值永远在最前）。"""
        order = {
            Severity.CRITICAL: 0,
            Severity.ABNORMAL: 1,
            Severity.BORDERLINE: 2,
            Severity.UNKNOWN: 3,
            Severity.NORMAL: 4,
        }
        bad = [i for i in self.items if i.flag not in {Flag.NORMAL, Flag.UNKNOWN}]
        return sorted(bad, key=lambda i: (order[i.severity], -(abs(i.deviation or 0))))

    @property
    def critical_items(self) -> list[LabItem]:
        return [i for i in self.items if i.severity is Severity.CRITICAL]

    @property
    def unknown_items(self) -> list[LabItem]:
        return [i for i in self.items if i.flag is Flag.UNKNOWN]

    def summary_counts(self) -> dict[str, int]:
        counts = {f.value: 0 for f in Flag}
        for i in self.items:
            counts[i.flag.value] += 1
        counts["total"] = len(self.items)
        return counts


# --------------------------------------------------------------------------- #
# 第 3 层：解释层输出契约
# --------------------------------------------------------------------------- #
class ItemInterpretation(BaseModel):
    """对单个异常项的解读。"""

    raw_name: str
    canonical_name: str | None = None
    flag: Flag = Flag.UNKNOWN
    severity: Severity = Severity.UNKNOWN
    value_display: str | None = None
    one_liner: str = Field(description="一句话说明这个指标是什么")
    what_it_means: str = Field(description="本次结果的含义（不构成诊断）")
    possible_factors: list[str] = Field(
        default_factory=list, description="可能导致该结果的常见因素，供与医生讨论"
    )
    source_page: int | None = None
    source_text: str | None = None


class Interpretation(BaseModel):
    """一份报告的完整解读结果。"""

    overview: str = Field(description="整体印象：正常项 / 异常项数量与分布")
    items: list[ItemInterpretation] = Field(default_factory=list)
    questions_for_doctor: list[str] = Field(
        default_factory=list, description="建议向医生提出的具体问题"
    )
    next_steps: list[str] = Field(default_factory=list, description="后续建议（复查/就诊/生活方式）")
    urgent_notice: str | None = Field(
        default=None, description="出现危急值时的醒目提示，无危急值则为 None"
    )
    limitations: list[str] = Field(
        default_factory=list, description="本次解读的局限性与不确定性说明"
    )
    disclaimer: str = ""


# --------------------------------------------------------------------------- #
# 给 LLM 的 JSON Schema（用于 structured output / 提示词内嵌）
# --------------------------------------------------------------------------- #
EXTRACTION_JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "report_type": {"type": ["string", "null"]},
        "hospital": {"type": ["string", "null"]},
        "collected_at": {"type": ["string", "null"]},
        "patient_sex": {"type": ["string", "null"]},
        "patient_age": {"type": ["integer", "null"]},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["raw_name"],
                "properties": {
                    "raw_name": {"type": "string"},
                    "result_raw": {"type": ["string", "null"]},
                    "unit_raw": {"type": ["string", "null"]},
                    "ref_raw": {"type": ["string", "null"]},
                    "flag_raw": {"type": ["string", "null"]},
                    "page": {"type": ["integer", "null"]},
                    "source_text": {"type": ["string", "null"]},
                },
            },
        },
        "extraction_notes": {"type": "array", "items": {"type": "string"}},
    },
}


def make_report_id(*parts: str) -> str:
    """由报告的关键特征生成稳定短 ID，用于去重与趋势串联。"""
    h = hashlib.sha256("|".join(p or "" for p in parts).encode("utf-8")).hexdigest()
    return h[:16]
