"""单元测试：单位归一与解析。

这些用例直接对应真实报告单上的写法混乱，是判定正确性的第一道防线。
"""

from __future__ import annotations

import pytest

from core.units import (
    Bounded,
    convert_value,
    normalize_unit,
    parse_ref_text,
    parse_result,
    units_compatible,
)


# ---------------------------------------------------------------- 单位归一 #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("×10^9/L", "10^9/L"),
        ("10E9/L", "10^9/L"),
        ("10*9/L", "10^9/L"),
        ("10⁹/L", "10^9/L"),
        ("x10^9/L", "10^9/L"),
        ("10^9/L", "10^9/L"),
        ("×10¹²/L", "10^12/L"),
        ("g/L", "g/L"),
        ("g/l", "g/L"),
        ("umol/L", "μmol/L"),
        ("μmol/L", "μmol/L"),
        ("U/L", "U/L"),
        ("IU/L", "U/L"),
        ("%", "%"),
        ("fL", "fL"),
        ("mm/hr", "mm/h"),
    ],
)
def test_normalize_unit(raw: str, expected: str) -> None:
    assert normalize_unit(raw) == expected


def test_normalize_unit_none_and_blank() -> None:
    assert normalize_unit(None) is None
    assert normalize_unit("   ") is None


def test_units_compatible() -> None:
    assert units_compatible("×10^9/L", "10^9/L")
    assert units_compatible("g/L", "g/l")
    assert not units_compatible("mg/dL", "mmol/L")


# ---------------------------------------------------------------- 结果解析 #
@pytest.mark.parametrize(
    "raw,value,bound",
    [
        ("13.2", 13.2, Bounded.EXACT),
        ("6", 6.0, Bounded.EXACT),
        ("<0.01", 0.01, Bounded.UPPER),
        ("≤0.5", 0.5, Bounded.UPPER),
        (">1000", 1000.0, Bounded.LOWER),
        ("-1.5", -1.5, Bounded.EXACT),
    ],
)
def test_parse_result_numeric(raw: str, value: float, bound: str) -> None:
    r = parse_result(raw)
    assert r.value == pytest.approx(value)
    assert r.bound == bound
    assert r.value_text is None


@pytest.mark.parametrize("raw", ["阴性", "阳性", "±", "(+)", "++", "trace", "negative"])
def test_parse_result_qualitative(raw: str) -> None:
    r = parse_result(raw)
    assert r.value is None
    assert r.value_text is not None


def test_parse_result_empty() -> None:
    r = parse_result(None)
    assert r.value is None and r.value_text is None


# ---------------------------------------------------------------- 区间解析 #
def test_parse_ref_range() -> None:
    r = parse_ref_text("3.5-9.5")
    assert (r.low, r.high) == (3.5, 9.5)
    assert r.kind == "range"
    assert r.usable


def test_parse_ref_range_various_dashes() -> None:
    for text in ["3.5~9.5", "3.5～9.5", "3.5 — 9.5", "3.5至9.5"]:
        r = parse_ref_text(text)
        assert (r.low, r.high) == (3.5, 9.5), text


def test_parse_ref_upper_lower() -> None:
    assert parse_ref_text("<5.0").high == 5.0
    assert parse_ref_text("≤5.0").high == 5.0
    assert parse_ref_text(">1.0").low == 1.0
    assert parse_ref_text("≥1.0").low == 1.0


def test_parse_ref_qualitative() -> None:
    r = parse_ref_text("阴性")
    assert r.kind == "qualitative"
    assert r.qualitative_normal and "阴性" in r.qualitative_normal


def test_parse_ref_reversed_is_swapped() -> None:
    r = parse_ref_text("9.5-3.5")
    assert (r.low, r.high) == (3.5, 9.5)


def test_parse_ref_unknown() -> None:
    r = parse_ref_text("详见报告")
    assert not r.usable


# ---------------------------------------------------------------- 换算 #
def test_convert_value_registered_factor() -> None:
    conv = [{"from": "mg/dL", "factor": 88.4}]
    val, converted, note = convert_value(1.2, "mg/dL", "μmol/L", conv)
    assert converted
    assert val == pytest.approx(106.08)
    assert note and "换算" in note


def test_convert_value_same_unit_is_noop() -> None:
    val, converted, note = convert_value(5.0, "g/L", "g/L", [])
    assert (val, converted, note) == (5.0, False, None)


def test_convert_value_unknown_factor_does_not_guess() -> None:
    """未登记换算因子时必须原样返回并告警，绝不擅自乘系数。"""
    val, converted, note = convert_value(5.0, "mg/dL", "mmol/L", [])
    assert val == 5.0
    assert not converted
    assert note and "未登记换算因子" in note
