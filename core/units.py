"""单位归一化与换算。

设计要点
--------
检验报告单上的单位写法极度混乱（``×10^9/L`` / ``10E9/L`` / ``10*9/L`` /
``10⁹/L`` / ``x10^9/L`` 是同一个东西），而单位错了会导致**数量级级别的误判**
（把 ``×10⁹/L`` 当成 ``×10¹²/L``，红细胞计数就直接差 1000 倍）。

因此这里做两件事：
1. ``normalize_unit`` —— 把各种写法映射到规范单位。只做**写法归一**，不做数值换算。
2. ``convert_value`` —— 按知识库中登记的换算因子做**数值换算**。

刻意不做的事：不猜测换算关系。知识库里没登记 ``from`` 单位时，宁可标记为
"单位不一致，未换算"，也不擅自乘一个系数——这是静默错误的温床。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "Bounded",
    "ParsedResult",
    "RefRange",
    "normalize_unit",
    "parse_result",
    "parse_ref_text",
    "convert_value",
    "units_compatible",
]

# --------------------------------------------------------------------------- #
# 单位写法归一
# --------------------------------------------------------------------------- #
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

# 规范单位 -> 该单位的所有已知写法（比较时会先做预处理：去空格、上标转普通、统一小写、μ/u 互通）
_UNIT_ALIASES: dict[str, tuple[str, ...]] = {
    "10^9/L": ("10^9/l", "10*9/l", "10e9/l", "10 9/l", "x10^9/l", "×10^9/l", "10^9/l",
               "10e9/l", "g/l*10^9", "10^9个/l", "×10⁹/L"),
    "10^12/L": ("10^12/l", "10*12/l", "10e12/l", "x10^12/l", "×10^12/l", "10^12/l"),
    "g/L": ("g/l", "g/liter", "克/升"),
    "mg/L": ("mg/l",),
    "mg/dL": ("mg/dl",),
    "μg/L": ("ug/l", "μg/l", "mcg/l", "ng/ml"),
    "ng/mL": ("ng/ml", "ng/ml"),
    "pg/mL": ("pg/ml",),
    "μmol/L": ("umol/l", "μmol/l", "µmol/l", "umol/l"),
    "mmol/L": ("mmol/l",),
    "U/L": ("u/l", "iu/l", "μ/l", "u/l"),
    "mIU/L": ("miu/l", "uiu/ml", "miu/ml", "μiu/ml"),
    "%": ("%", "％", "percent"),
    "fL": ("fl", "fl."),
    "pg": ("pg", "pg."),
    "s": ("s", "sec", "秒"),
    "mm/h": ("mm/h", "mm/hr", "mm/1h", "mm/hour"),
    "mL/min/1.73m2": (
        "ml/min/1.73m2", "ml/min/1.73m^2", "ml/min/1.73m²", "ml/min/1.73㎡",
        "ml/min/1.73m^2", "ml/min/1.73平方米",
    ),
    "mOsm/kg": ("mosm/kg",),
    "ratio": ("ratio", "比值", "-"),
    "": (),
}

# 反向索引
_ALIAS_TO_CANON: dict[str, str] = {}
for _canon, _aliases in _UNIT_ALIASES.items():
    _ALIAS_TO_CANON[_canon.lower()] = _canon
    for _a in _aliases:
        _ALIAS_TO_CANON[_a.lower()] = _canon


def _prep(unit: str) -> str:
    """把单位字符串压成可比较的形式。"""
    u = str(unit).strip().translate(_SUPERSCRIPT)
    u = u.replace("μ", "u").replace("µ", "u")  # 统一到 u，别名表里 μ 已同时登记
    u = re.sub(r"\s+", "", u)
    return u.lower()


def normalize_unit(raw: str | None) -> str | None:
    """把报告单上的单位写法归一为规范单位。无法识别时返回清理后的原文。

    >>> normalize_unit("×10⁹/L")
    '10^9/L'
    >>> normalize_unit("10E9/L")
    '10^9/L'
    >>> normalize_unit("umol/L")
    'μmol/L'
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # 先把上标、乘号、科学计数法统一成 ^ 形式，再查表
    t = s.translate(_SUPERSCRIPT)
    t = re.sub(r"\s+", "", t)
    t = t.replace("×", "x").replace("*", "^")
    t = re.sub(r"(?i)^x?10\^?(\d+)", r"10^\1", t)
    t = re.sub(r"(?i)10e(\d+)", r"10^\1", t)

    key = _prep(t)
    if key in _ALIAS_TO_CANON:
        return _ALIAS_TO_CANON[key]

    key2 = _prep(s)
    if key2 in _ALIAS_TO_CANON:
        return _ALIAS_TO_CANON[key2]

    return t or s


def units_compatible(a: str | None, b: str | None) -> bool:
    """判断两个单位是否可视为同一单位（已归一后比较）。"""
    if a is None or b is None:
        return a == b
    na, nb = normalize_unit(a), normalize_unit(b)
    if na == nb:
        return True
    return (na or "").split("(")[0] == (nb or "").split("(")[0]


# --------------------------------------------------------------------------- #
# 结果值解析
# --------------------------------------------------------------------------- #
class Bounded(str):
    """结果值的边界类型：精确值 / 小于某值 / 大于某值。

    报告上常见 ``<0.01``、``>1000`` 这类带界限的结果。直接当成 0.01 / 1000 会
    在临界判定上出错，所以要显式记录它是"有界"的。
    """

    EXACT = "exact"
    UPPER = "upper"  # < x 或 ≤ x
    LOWER = "lower"  # > x 或 ≥ x


@dataclass
class ParsedResult:
    value: float | None
    value_text: str | None
    bound: str = Bounded.EXACT
    raw: str | None = None

    @property
    def is_quantitative(self) -> bool:
        return self.value is not None

    @property
    def is_ambiguous_bound(self) -> bool:
        """带界限的结果在跨过参考区间时判定是确定的，在界限内则确定；
        但若界限值本身落在参考区间之外，判定仍然确定。只有界限恰好跨越
        参考区间边界时才真正模糊。此属性供上层保守处理。"""
        return self.bound != Bounded.EXACT


_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
_QUALITATIVE_TOKENS = {
    "阴性", "阳性", "弱阳性", "弱阳性(±)", "±", "+", "++", "+++", "++++",
    "negative", "positive", "neg", "pos", "正常", "异常", "未见", "未检出",
    "(-)", "(+)", "(++)", "(+++)", "trace", "微量",
}
_QUALITATIVE_NORMAL = {"阴性", "negative", "neg", "正常", "未见", "未检出", "(-)", "-"}


def _to_float(num: str) -> float | None:
    try:
        return float(num.replace(",", ".")) if num.count(",") == 1 and "." not in num else float(num.replace(",", ""))
    except (TypeError, ValueError):
        return None


def parse_result(raw: str | None) -> ParsedResult:
    """解析报告上的结果栏。

    >>> parse_result("13.2").value
    13.2
    >>> parse_result("<0.01").bound
    'upper'
    >>> parse_result("阴性").value_text
    '阴性'
    """
    if raw is None:
        return ParsedResult(None, None, Bounded.EXACT, None)
    s = str(raw).strip()
    if not s:
        return ParsedResult(None, None, Bounded.EXACT, None)

    s_norm = re.sub(r"\s+", "", s)

    # 定性结果优先（避免"阴性"里没有数字却走到数值分支）
    low = s_norm.lower()
    if low in _QUALITATIVE_TOKENS or s_norm in _QUALITATIVE_TOKENS:
        return ParsedResult(None, s_norm, Bounded.EXACT, s)
    if re.fullmatch(r"\(?[+\-±]{1,4}\)?", s_norm):
        return ParsedResult(None, s_norm, Bounded.EXACT, s)
    if re.fullmatch(r"(阴性|阳性|弱阳性|negative|positive|neg|pos|trace|微量)", low):
        return ParsedResult(None, s_norm, Bounded.EXACT, s)

    # 数值（含可选的比较符）
    m = re.match(r"^(?P<op>[<>≤≥]=?)?\s*(?P<num>-?\d+(?:[.,]\d+)?)", s_norm)
    if m:
        val = _to_float(m.group("num"))
        op = m.group("op") or ""
        bound = Bounded.EXACT
        if op.startswith("<") or op.startswith("≤"):
            bound = Bounded.UPPER
        elif op.startswith(">") or op.startswith("≥"):
            bound = Bounded.LOWER
        return ParsedResult(val, None, bound, s)

    # 兜底：整串拿不到数字就当定性
    return ParsedResult(None, s_norm, Bounded.EXACT, s)


# --------------------------------------------------------------------------- #
# 参考区间解析
# --------------------------------------------------------------------------- #
@dataclass
class RefRange:
    low: float | None
    high: float | None
    qualitative_normal: tuple[str, ...] | None = None
    kind: str = "range"  # range | upper_bound | lower_bound | qualitative | unknown

    @property
    def usable(self) -> bool:
        return self.kind != "unknown" and (
            self.qualitative_normal is not None or self.low is not None or self.high is not None
        )


_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212~～至-"


def parse_ref_text(raw: str | None) -> RefRange:
    """解析参考区间栏。

    支持：``3.5-9.5`` / ``3.5~9.5`` / ``<5.0`` / ``≤5.0`` / ``>1.0`` / ``阴性`` / ``0-10``

    >>> parse_ref_text("3.5-9.5").low
    3.5
    >>> parse_ref_text("<5.0").high
    5.0
    >>> parse_ref_text("阴性").kind
    'qualitative'
    """
    if raw is None:
        return RefRange(None, None, None, "unknown")
    s = str(raw).strip()
    if not s:
        return RefRange(None, None, None, "unknown")

    s_norm = re.sub(r"\s+", "", s)

    # 定性区间："阴性" / "阴性(-)" / "negative"
    if re.fullmatch(r"[（(]?(阴性|negative|neg|正常|未见)[）)]?([（(].*[）)])?", s_norm, re.I):
        return RefRange(None, None, ("阴性", "negative", "neg", "正常", "未见"), "qualitative")

    # 上限型：<5.0 / ≤5.0
    m = re.fullmatch(r"[<≤]=?(-?\d+(?:[.,]\d+)?)", s_norm)
    if m:
        return RefRange(None, _to_float(m.group(1)), None, "upper_bound")

    # 下限型：>1.0 / ≥1.0
    m = re.fullmatch(r"[>≥]=?(-?\d+(?:[.,]\d+)?)", s_norm)
    if m:
        return RefRange(_to_float(m.group(1)), None, None, "lower_bound")

    # 双侧区间：3.5-9.5（注意负号与连接符的区分）
    m = re.fullmatch(rf"(-?\d+(?:[.,]\d+)?)[{_DASHES}]+(-?\d+(?:[.,]\d+)?)", s_norm)
    if m:
        lo, hi = _to_float(m.group(1)), _to_float(m.group(2))
        if lo is not None and hi is not None:
            if lo > hi:  # 报告印反了，容错交换并交由上层告警
                lo, hi = hi, lo
            return RefRange(lo, hi, None, "range")

    # 形如 "3.5 - 9.5 10^9/L" —— 去掉尾部单位再试一次
    m = re.search(rf"(-?\d+(?:[.,]\d+)?)[{_DASHES}]+(-?\d+(?:[.,]\d+)?)", s_norm)
    if m:
        lo, hi = _to_float(m.group(1)), _to_float(m.group(2))
        if lo is not None and hi is not None:
            if lo > hi:
                lo, hi = hi, lo
            return RefRange(lo, hi, None, "range")

    return RefRange(None, None, None, "unknown")


# --------------------------------------------------------------------------- #
# 数值换算
# --------------------------------------------------------------------------- #
def convert_value(
    value: float | None,
    from_unit: str | None,
    to_unit: str | None,
    conversions: list[dict] | None = None,
) -> tuple[float | None, bool, str | None]:
    """按知识库登记的换算因子把结果值换算到规范单位。

    Returns
    -------
    (换算后的值, 是否发生了换算, 说明)
    若单位不需要换算，返回原值；若需要换算但知识库没有登记因子，
    **返回原值并给出说明**，绝不擅自猜测系数。
    """
    if value is None:
        return None, False, None

    fu, tu = normalize_unit(from_unit), normalize_unit(to_unit)
    if fu is None or tu is None or fu == tu:
        return value, False, None

    for conv in conversions or []:
        src = normalize_unit(conv.get("from"))
        if src is None or src != fu:
            continue
        try:
            factor = float(conv.get("factor", 1.0))
            offset = float(conv.get("offset", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        return value * factor + offset, True, f"单位换算：{fu} → {tu}（×{factor:g}）"

    return value, False, f"单位不一致（报告为 {fu}，规范单位为 {tu}）且知识库未登记换算因子，未做换算"
