"""规则引擎：把 LLM 抽取的裸数据变成可判定的结构化指标。

**这是整个系统里唯一允许做数值判断的地方。** 它不调用任何模型，全是纯函数，
因此可以被穷举单测、可以被审计、结果 100% 可复现。

参考区间的优先级（这一条是产品的核心机制，有文献背书）
------------------------------------------------------
1. **报告单自带的参考区间**（``RefSource.REPORT``）—— 最高优先级。不同医院用不同
   仪器和试剂，参考区间本来就不同，报告上印的才是对这份结果有效的。
2. **内置知识库**（``RefSource.KNOWLEDGE_BASE``）—— 仅当报告没印区间时兜底。
3. **两者都没有 → 拒绝判定**（``Flag.UNKNOWN``）。

第 3 条不是保守，是必须的。Meyer 等（Frontiers in Artificial Intelligence, 2025）
在 24.6 万个参考区间上测得大模型给出的区间下限平均变异系数达 26.5%，作者明确建议
在用户未提供参考区间时应**拒绝解读**。让模型"猜一个正常范围"是本项目最需要防的错误。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .schema import (
    DecisionTrace,
    Flag,
    LabItem,
    LabReport,
    PatientContext,
    RawLabItem,
    RawLabReport,
    RefSource,
    Severity,
    make_report_id,
)
from .units import (
    Bounded,
    convert_value,
    normalize_unit,
    parse_ref_text,
    parse_result,
    units_compatible,
)

__all__ = ["Indicator", "KnowledgeBase", "load_knowledge_base", "judge_item", "build_report"]

# 偏离度小于该比例时判为"临界"，前端用黄色提示而非红色
BORDERLINE_RATIO = 0.05


# --------------------------------------------------------------------------- #
# 知识库
# --------------------------------------------------------------------------- #
@dataclass
class Indicator:
    """一个检验项目的结构化定义。"""

    canonical_name: str
    name_zh: str = ""
    aliases: list[str] = field(default_factory=list)
    loinc: str | None = None
    category: str | None = None
    specimen: str | None = None
    unit_canonical: str | None = None
    decimals: int | None = None
    qualitative: bool = False
    ref: dict[str, Any] = field(default_factory=dict)
    critical: dict[str, Any] = field(default_factory=dict)
    unit_conversions: list[dict] = field(default_factory=list)
    source: str | None = None
    note: str | None = None

    # --- 参考区间查询 ---
    def ref_for(self, patient: PatientContext) -> tuple[float | None, float | None, list[str] | None]:
        """按性别/年龄取参考区间。返回 (low, high, qualitative_normal)。"""
        if self.qualitative:
            normals = self.ref.get("qualitative_normal")
            return None, None, list(normals) if normals else None

        node = self.ref or {}
        picked = None
        if patient.sex and patient.sex in node and isinstance(node[patient.sex], dict):
            picked = node[patient.sex]
        elif "default" in node and isinstance(node["default"], dict):
            picked = node["default"]
        else:
            # ref 直接写成 {low:.., high:..} 的简写形式
            if "low" in node or "high" in node:
                picked = node
        if not picked:
            return None, None, None
        return picked.get("low"), picked.get("high"), None

    def requires_sex(self) -> bool:
        """该指标的参考区间是否按性别分层且男女不同。

        用于回答一个真实存在的困境：报告没写性别时，该用哪个区间？
        我们的答案是——**不猜**。用男性区间判女性、或用男女并集判男性，
        两种做法都会产生静默错误（前者误报"偏低"，后者漏报真实的偏低）。
        所以缺少性别时拒绝判定，请用户补充性别。
        """
        if self.qualitative:
            return False
        node = self.ref or {}
        m, f = node.get("M"), node.get("F")
        if not isinstance(m, dict) or not isinstance(f, dict):
            return False
        return (m.get("low"), m.get("high")) != (f.get("low"), f.get("high"))

    def critical_for(self, patient: PatientContext) -> tuple[float | None, float | None]:
        node = self.critical or {}
        if not isinstance(node, dict) or not node:
            return None, None
        return node.get("low"), node.get("high")


def _iter_yaml_files(dir_path: Path) -> Iterable[Path]:
    if not dir_path.is_dir():
        return []
    return sorted(p for p in dir_path.glob("*.y*ml") if p.is_file())


class KnowledgeBase:
    """结构化指标字典 + 别名索引。

    数据来源文件放在 ``data/indicators_*.yaml``，刻意与代码分离，
    方便按卫生行业标准版本替换而不用改一行逻辑。
    """

    def __init__(self, indicators: Iterable[Indicator], version: str = "unknown") -> None:
        self.version = version
        self.indicators: dict[str, Indicator] = {i.canonical_name: i for i in indicators}
        self.alias_conflicts: dict[str, list[str]] = {}
        self._alias_index: dict[str, str] = {}

        # 第一遍：只登记"本体名"（canonical_name 与 name_zh）。
        # 本体名是该指标唯一的、不会与别人共享的名字，优先级最高。
        # 不这么做就会出现真实事故：`crp` 把 `超敏C反应蛋白` 登记为自己的别名之后，
        # `hscrp` 就再也解析不到了——系统会拿着 crp 的参考区间去判 hscrp 的结果，
        # 数字看着正常，结论却是错的，用户完全看不出来。
        for ind in self.indicators.values():
            for key_src in (ind.canonical_name, ind.name_zh):
                if not key_src:
                    continue
                key = self._norm_key(key_src)
                owner = self._alias_index.get(key)
                if owner is None:
                    self._alias_index[key] = ind.canonical_name
                elif owner != ind.canonical_name:
                    self._record_conflict(key, owner, ind.canonical_name)

        # 第二遍：登记别名，只填空位。撞名的**记录下来**而不是悄悄覆盖。
        for ind in self.indicators.values():
            for key_src in ind.aliases or []:
                if not key_src:
                    continue
                key = self._norm_key(key_src)
                owner = self._alias_index.get(key)
                if owner is None:
                    self._alias_index[key] = ind.canonical_name
                elif owner != ind.canonical_name:
                    self._record_conflict(key, owner, ind.canonical_name)

    def _record_conflict(self, key: str, owner: str, other: str) -> None:
        bucket = self.alias_conflicts.setdefault(key, [owner])
        if other not in bucket:
            bucket.append(other)

    def conflicts(self) -> dict[str, list[str]]:
        """返回被多个指标共享的名字，供数据维护时人工消歧。"""
        return dict(self.alias_conflicts)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _norm_key(s: str) -> str:
        """别名归一：去掉空白、括号、单位后缀、大小写差异。"""
        t = str(s).strip().lower()
        t = re.sub(r"[\s\u3000]+", "", t)
        t = re.sub(r"[（(].*?[)）]", "", t)
        t = t.replace("＊", "*").replace("×", "x")
        return t

    def __len__(self) -> int:
        return len(self.indicators)

    def get(self, canonical_name: str | None) -> Indicator | None:
        if not canonical_name:
            return None
        return self.indicators.get(canonical_name)

    def resolve(self, raw_name: str | None, prefer_category: str | None = None) -> Indicator | None:
        """把报告上的任意写法解析到规范指标。解析不到返回 None。

        解析不到时**不会**用模糊匹配硬凑——宁可让上层标为"未识别指标"，
        也不要把 A 项目的参考区间套到 B 项目上。

        ``prefer_category``（通常是报告类型）用于消解**跨类别同名**问题。
        真实例子：``白细胞`` 在血常规里指 WBC，在尿常规里指尿白细胞酯酶(u_le)，
        两者参考区间与含义完全不同。若只建一张扁平的全局别名表，
        尿常规报告里的"白细胞"会被解析成血常规的 WBC——一个典型且难以察觉的串指标。
        """
        if not raw_name:
            return None
        key = self._norm_key(raw_name)

        owner = self._alias_index.get(key)
        if owner is None:
            # 去掉常见后缀再试： "白细胞计数(WBC)" 已在 _norm_key 里去括号
            for suffix in ("测定", "检测", "定量", "浓度", "计数", "值"):
                if key.endswith(suffix) and key[: -len(suffix)] in self._alias_index:
                    key = key[: -len(suffix)]
                    owner = self._alias_index[key]
                    break
        if owner is None:
            return None

        # 名字被多个指标共享时，用报告类型把作用域收窄
        candidates = self.alias_conflicts.get(key)
        if candidates and prefer_category:
            hint = str(prefer_category).strip()
            for canon in candidates:
                ind = self.indicators.get(canon)
                if ind and ind.category and ind.category == hint:
                    return ind
            for canon in candidates:
                ind = self.indicators.get(canon)
                if ind and ind.category and (ind.category in hint or hint in ind.category):
                    return ind

        return self.indicators.get(owner)


def load_knowledge_base(data_dir: str | Path = "data") -> KnowledgeBase:
    """从 YAML 载入知识库。缺少 PyYAML 或目录不存在时返回空库（不抛异常）。"""
    path = Path(data_dir)
    indicators: list[Indicator] = []
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover
        return KnowledgeBase([], version="unavailable")

    for f in _iter_yaml_files(path):
        try:
            rows = yaml.safe_load(f.read_text(encoding="utf-8")) or []
        except Exception:  # noqa: BLE001 - 单个文件损坏不应拖垮整个库
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or not row.get("canonical_name"):
                continue
            known = {k: v for k, v in row.items() if k in Indicator.__dataclass_fields__}
            indicators.append(Indicator(**known))

    # 后加载的文件覆盖先加载的同名指标，方便本地覆盖标准区间
    merged: dict[str, Indicator] = {}
    for ind in indicators:
        merged[ind.canonical_name] = ind
    return KnowledgeBase(merged.values(), version=f"{len(merged)} indicators")


# --------------------------------------------------------------------------- #
# 单项判定
# --------------------------------------------------------------------------- #
def judge_item(
    raw: RawLabItem,
    kb: KnowledgeBase,
    patient: PatientContext | None = None,
    report_type: str | None = None,
) -> LabItem:
    """把一条抽取结果判定为 ``LabItem``。

    判定优先级：报告自带区间 > 知识库区间 > 拒绝判定。

    ``report_type`` 用于消解跨类别同名指标（如"白细胞"在血常规与尿常规中含义不同）。
    """
    patient = patient or PatientContext()
    notes: list[str] = []

    ind = kb.resolve(raw.raw_name, prefer_category=report_type)
    parsed = parse_result(raw.result_raw)

    item = LabItem(
        raw_name=raw.raw_name,
        canonical_name=ind.canonical_name if ind else None,
        loinc=ind.loinc if ind else None,
        category=ind.category if ind else None,
        specimen=ind.specimen if ind else None,
        value=parsed.value,
        value_text=parsed.value_text,
        unit=normalize_unit(raw.unit_raw) or (ind.unit_canonical if ind else None),
        flag_reported=raw.flag_raw,
        page=raw.page,
        source_text=raw.source_text,
    )
    if ind is None:
        notes.append(f"未在知识库中识别到指标「{raw.raw_name}」，仅按报告自带区间判定。")
    elif ind.note:
        # 知识库里记录的方法学陷阱（如 ckmb 质量法 vs 活性法、ddimer 的 DDU/FEU 差异）
        # 必须透传出来，否则用户看到的数字可能来自不可比的检测体系。
        notes.append(f"指标说明：{ind.note}")

    # ---------- 1. 确定参考区间 ----------
    report_ref = parse_ref_text(raw.ref_raw)
    kb_normals: list[str] | None = None
    if report_ref.usable:
        item.ref_low, item.ref_high = report_ref.low, report_ref.high
        kb_normals = list(report_ref.qualitative_normal) if report_ref.qualitative_normal else None
        item.ref_text = str(raw.ref_raw).strip()
        item.ref_source = RefSource.REPORT
    elif ind is not None:
        # 参考区间按性别分层、而报告又没写性别时：拒绝判定。
        # 用任一性别的区间都会产生静默错误——用男性区间判女性会误报"偏低"，
        # 用男女并集判男性会漏报真实的偏低。本项目的原则是宁可拒判也不猜。
        if ind.requires_sex() and not patient.sex:
            item.flag, item.severity = Flag.UNKNOWN, Severity.UNKNOWN
            notes.append(
                f"「{ind.name_zh or ind.canonical_name}」的参考区间按性别分层，"
                "但报告未提供性别信息。为避免用错区间，本系统不做判定——"
                "请补充性别后重新分析。"
            )
            item.rules_notes = notes
            item.decision_trace = _make_trace(
                item, ind, "refused_sex_required",
                outcome=f"缺少性别信息，而「{ind.name_zh or ind.canonical_name}」的区间按性别分层",
            )
            return item

        lo, hi, normals = ind.ref_for(patient)
        if lo is not None or hi is not None or normals:
            item.ref_low, item.ref_high = lo, hi
            kb_normals = normals
            item.ref_text = _format_ref(lo, hi, normals)
            item.ref_source = RefSource.KNOWLEDGE_BASE
            notes.append(
                f"报告未提供参考区间，已使用知识库区间"
                f"（来源：{ind.source or '知识库'}）。"
            )

    # 定性结果的"正常表述"有多种写法（阴性 / - / neg / 正常 / 未见）。
    # 报告往往只印其中一种，而知识库里登记的是完整集合，两者取并集——
    # 否则报告写"阴性"、结果是"-"时会被误判成阳性。
    if ind is not None and ind.qualitative:
        _, _, kb_qual = ind.ref_for(patient)
        if kb_qual:
            merged = list(kb_normals or [])
            for n in kb_qual:
                if n not in merged:
                    merged.append(n)
            kb_normals = merged

    # ---------- 2. 单位换算 ----------
    # 只有在参考区间来自知识库时才需要换算：报告自带的区间与结果天然同单位。
    # 若拿知识库的区间去比对报告单位下的数值，就是"苹果比橘子"，必然误判。
    if (
        item.ref_source is RefSource.KNOWLEDGE_BASE
        and ind is not None
        and parsed.value is not None
        and raw.unit_raw
    ):
        new_val, converted, conv_note = convert_value(
            parsed.value, raw.unit_raw, ind.unit_canonical, ind.unit_conversions
        )
        if converted:
            item.value, item.unit = new_val, ind.unit_canonical
        if conv_note:
            notes.append(conv_note)
        if not converted and not units_compatible(raw.unit_raw, ind.unit_canonical):
            item.flag, item.severity = Flag.UNKNOWN, Severity.UNKNOWN
            notes.append("结果单位与知识库单位不一致且知识库未登记换算因子，已放弃判定以避免误判。")
            item.rules_notes = notes
            item.decision_trace = _make_trace(
                item, ind, "refused_unit_mismatch",
                outcome=f"报告单位 {raw.unit_raw} 与知识库单位 {ind.unit_canonical} 不一致且无法换算，拒绝判定",
            )
            return item
    elif item.ref_source is RefSource.REPORT and ind is not None and raw.unit_raw:
        if not units_compatible(raw.unit_raw, ind.unit_canonical):
            notes.append(
                f"报告单位（{raw.unit_raw}）与知识库规范单位（{ind.unit_canonical}）不同，"
                "已直接采用报告自带的参考区间判定，未做任何换算。"
            )

    # ---------- 3. 定性项目 ----------
    if parsed.value_text is not None and parsed.value is None:
        item.flag, item.severity = _judge_qualitative(parsed.value_text, kb_normals, notes)
        item.rules_notes = notes
        item.decision_trace = _make_trace(
            item, ind, "qualitative",
            crossed="within" if item.flag is Flag.NORMAL else "qualitative_abnormal",
            outcome=f"定性结果「{parsed.value_text}」判为{item.flag.label_zh}",
        )
        return item

    # ---------- 4. 定量项目 ----------
    if parsed.value is None:
        item.flag, item.severity = Flag.UNKNOWN, Severity.UNKNOWN
        notes.append("未能从报告中解析出数值结果，无法判定。")
        item.rules_notes = notes
        item.decision_trace = _make_trace(
            item, ind, "refused_no_value", outcome="未从报告中解析出数值，拒绝判定"
        )
        return item

    if item.ref_low is None and item.ref_high is None:
        item.flag, item.severity = Flag.UNKNOWN, Severity.UNKNOWN
        notes.append(
            "报告与知识库均无可用参考区间，**拒绝判定**。"
            "参考区间因仪器与试剂而异，请以检验科出具的区间为准。"
        )
        item.rules_notes = notes
        item.decision_trace = _make_trace(
            item, ind, "refused_no_interval",
            outcome="报告与知识库均无可用参考区间，本系统不做判定",
        )
        return item

    crit_lo: float | None = None
    crit_hi: float | None = None
    if ind is not None:
        # 危急值阈值来自知识库，只有在**量纲一致**时才可用。
        #
        # 反例：红细胞压积，报告用 % (44.6)，知识库阈值用 L/L (0.60)。直接用会把
        # 44.6 > 0.60 判成"危急偏高"——一个量纲错配就能把完全正常的结果报成危急值。
        # 知识库路径下数值已被换算到规范单位，报告路径下则必须先确认单位兼容。
        unit_ok = (
            item.ref_source is RefSource.KNOWLEDGE_BASE
            or raw.unit_raw is None
            or item.ref_source is RefSource.NONE
            or units_compatible(item.unit, ind.unit_canonical)
        )
        if unit_ok:
            crit_lo, crit_hi = ind.critical_for(patient)
        else:
            notes.append(
                f"报告单位（{raw.unit_raw}）与知识库规范单位（{ind.unit_canonical}）不一致，"
                "为避免量纲错配造成危急值误报，本次不套用知识库的危急值阈值。"
            )

    item.flag, item.severity, item.deviation = _judge_quantitative(
        item.value, item.ref_low, item.ref_high, crit_lo, crit_hi, parsed.bound, notes
    )

    # 与报告自带标记交叉核对，不一致时留下审计痕迹（不覆盖我们的判定）
    if raw.flag_raw:
        reported = _norm_reported_flag(raw.flag_raw)
        if reported and reported != item.flag.value:
            notes.append(
                f"报告自带标记为「{raw.flag_raw}」，本系统按区间判定为「{item.flag.label_zh}」，"
                "建议人工复核。"
            )

    item.rules_notes = notes
    rule = {
        RefSource.REPORT: "report_interval",
        RefSource.KNOWLEDGE_BASE: "kb_interval",
    }.get(item.ref_source, "refused_no_interval")
    item.decision_trace = _make_trace(
        item, ind, rule,
        crossed=_crossed_from_flag(item.flag),
        outcome=(
            f"结果 {item.value:g} 与区间 "
            f"{_format_ref(item.ref_low, item.ref_high, None)} 比较，判为{item.flag.label_zh}"
        ),
    )
    return item


# --------------------------------------------------------------------------- #
# 判读依据（审计用）
# --------------------------------------------------------------------------- #
_SOURCE_LABEL = {
    RefSource.REPORT: "报告单",
    RefSource.KNOWLEDGE_BASE: "内置知识库",
    RefSource.NONE: None,
}

_CROSSED_BY_FLAG = {
    Flag.CRITICAL_HIGH: "critical_high",
    Flag.HIGH: "ref_high",
    Flag.NORMAL: "within",
    Flag.LOW: "ref_low",
    Flag.CRITICAL_LOW: "critical_low",
    Flag.UNKNOWN: None,
}


def _crossed_from_flag(flag: Flag) -> str | None:
    return _CROSSED_BY_FLAG.get(flag)


def _make_trace(
    item: LabItem,
    ind: Indicator | None,
    rule: str,
    crossed: str | None = None,
    outcome: str = "",
) -> DecisionTrace:
    """记录"这个判定是怎么来的"。

    用户和审计者需要能回答：用的是报告单的区间还是内置知识库的？依据是哪份标准？
    跨过的是参考上限还是危急值阈值？——没有这些信息，自动化判读就不可信。
    """
    unit = item.unit or (ind.unit_canonical if ind else "") or ""
    interval = _format_ref(item.ref_low, item.ref_high, None)
    return DecisionTrace(
        rule=rule,
        interval_source=_SOURCE_LABEL.get(item.ref_source),
        interval_standard=(
            ind.source if (ind is not None and item.ref_source is RefSource.KNOWLEDGE_BASE) else None
        ),
        interval_used=(f"{interval} {unit}".strip() or None),
        crossed=crossed,
        outcome=outcome,
        notes=list(item.rules_notes),
    )


def _format_ref(lo: float | None, hi: float | None, normals: list[str] | None) -> str:
    if normals:
        return "/".join(normals[:2])
    if lo is not None and hi is not None:
        return f"{lo:g}-{hi:g}"
    if hi is not None:
        return f"<{hi:g}"
    if lo is not None:
        return f">{lo:g}"
    return ""


def _norm_reported_flag(raw: str) -> str | None:
    s = str(raw).strip().upper()
    if s in {"H", "↑", "HIGH", "偏高"}:
        return Flag.HIGH.value
    if s in {"L", "↓", "LOW", "偏低"}:
        return Flag.LOW.value
    if s in {"N", "正常"}:
        return Flag.NORMAL.value
    return None


def _qualitative_tokens(value: str) -> set[str]:
    """把定性结果拆成记号集合。

    报告上同一个意思有大量写法：``阴性``、``-``、``(-)``、``阴性(-)``、``neg``。
    直接做字符串相等会漏掉带括号的组合写法，所以拆成记号再求交。
    """
    s = str(value).strip()
    paren = re.findall(r"[（(]([^)）]*)[)）]", s)
    base = re.sub(r"[（(][^)）]*[)）]", "", s).strip()
    tokens = {s, base, *paren}
    return {t.strip().lower() for t in tokens if t and t.strip()}


def _judge_qualitative(
    value_text: str, normals: list[str] | None, notes: list[str]
) -> tuple[Flag, Severity]:
    """定性结果判定（尿常规等）。"""
    normals = normals or ["阴性", "negative", "neg", "正常", "未见", "(-)", "-"]
    norm_set = {str(n).strip().lower() for n in normals}
    tokens = _qualitative_tokens(value_text)

    if tokens & norm_set:
        return Flag.NORMAL, Severity.NORMAL

    if tokens & {"弱阳性", "±", "(±)", "trace", "微量"}:
        notes.append("定性结果为弱阳性/微量，属临界范围，建议复查确认。")
        return Flag.HIGH, Severity.BORDERLINE

    notes.append(f"定性结果为「{str(value_text).strip()}」，判为异常（阳性）。")
    return Flag.HIGH, Severity.ABNORMAL


def _judge_quantitative(
    value: float,
    ref_low: float | None,
    ref_high: float | None,
    crit_low: float | None,
    crit_high: float | None,
    bound: str,
    notes: list[str],
) -> tuple[Flag, Severity, float | None]:
    """定量判定。危急值优先于普通异常，偏离度用于排序与临界识别。"""
    if bound != Bounded.EXACT:
        notes.append(
            f"结果带界限符号（{'<' if bound == Bounded.UPPER else '>'}），"
            "已按报告所给界限参与判定。"
        )

    if crit_high is not None and value > crit_high:
        dev = (value - (ref_high if ref_high is not None else value)) / (ref_high or 1) if ref_high else None
        notes.append(f"达到危急值阈值（>{crit_high:g}），建议尽快联系医生。")
        return Flag.CRITICAL_HIGH, Severity.CRITICAL, dev

    if crit_low is not None and value < crit_low:
        dev = ((ref_low if ref_low is not None else value) - value) / (ref_low or 1) if ref_low else None
        notes.append(f"达到危急值阈值（<{crit_low:g}），建议尽快联系医生。")
        return Flag.CRITICAL_LOW, Severity.CRITICAL, dev

    if ref_high is not None and value > ref_high:
        dev = (value - ref_high) / ref_high if ref_high else None
        sev = Severity.BORDERLINE if dev is not None and dev <= BORDERLINE_RATIO else Severity.ABNORMAL
        if sev is Severity.BORDERLINE:
            notes.append("略高于参考上限，属临界范围。")
        return Flag.HIGH, sev, dev

    if ref_low is not None and value < ref_low:
        dev = (ref_low - value) / ref_low if ref_low else None
        sev = Severity.BORDERLINE if dev is not None and dev <= BORDERLINE_RATIO else Severity.ABNORMAL
        if sev is Severity.BORDERLINE:
            notes.append("略低于参考下限，属临界范围。")
        return Flag.LOW, sev, -dev

    return Flag.NORMAL, Severity.NORMAL, 0.0


# --------------------------------------------------------------------------- #
# 整份报告
# --------------------------------------------------------------------------- #
def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    s = str(raw).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y年%m月%d日", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    m = re.search(r"(\d{4})\D{1,2}(\d{1,2})\D{1,2}(\d{1,2})", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def build_report(
    raw: RawLabReport,
    kb: KnowledgeBase,
    model_used: str | None = None,
) -> LabReport:
    """把一次抽取结果整体过一遍规则引擎，产出 ``LabReport``。"""
    patient = PatientContext(
        sex=raw.patient_sex,
        age=raw.patient_age if isinstance(raw.patient_age, int) else None,
    )
    collected = _parse_date(raw.collected_at)

    items = [judge_item(r, kb, patient, report_type=raw.report_type) for r in raw.items]

    warnings: list[str] = list(raw.extraction_notes or [])
    if patient.sex is None:
        warnings.append("报告未提供性别信息，性别特异的参考区间无法应用，已退回通用区间。")
    if patient.age is None:
        warnings.append("报告未提供年龄信息，年龄特异的参考区间无法应用。")
    if not items:
        warnings.append("未从报告中抽取到任何检验项目，请确认图片清晰度或手动录入。")

    dup = _find_duplicates(items)
    if dup:
        warnings.append(f"检测到疑似重复项目：{'、'.join(dup)}，请核对。")

    return LabReport(
        report_id=make_report_id(raw.hospital or "", str(collected or ""), raw.report_type or "",
                                 ",".join(i.raw_name for i in raw.items[:8])),
        report_type=raw.report_type,
        hospital=raw.hospital,
        collected_at=collected,
        patient=patient,
        items=items,
        model_used=model_used,
        warnings=warnings,
    )


def _find_duplicates(items: list[LabItem]) -> list[str]:
    seen: dict[str, int] = {}
    for i in items:
        key = i.canonical_name or i.raw_name
        seen[key] = seen.get(key, 0) + 1
    return [k for k, v in seen.items() if v > 1]
