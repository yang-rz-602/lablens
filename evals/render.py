"""把结构化 ground truth 渲染成仿真检验报告（文本 / 图片）。

评测抽取层需要"报告长什么样"的输入。为了保证评测是**可复现且不含真实隐私**的，
这里不采集任何真实报告，而是用代码从 ground truth 反向渲染出排版逼真的报告单。

这样做的两个好处：
1. 评测集可以随代码一起开源，不涉及任何个人信息
2. ground truth 与渲染结果是同源的，字段级比对不需要人工标注
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

__all__ = ["render_text", "render_image", "available_font"]

_FAKE_NAMES = ["张**", "李**", "王**", "刘**", "陈**", "杨**", "赵**", "周**"]
_FAKE_DEPT = ["体检中心", "内科门诊", "健康管理中心", "全科医学科"]


def _row(item: dict[str, Any]) -> str:
    name = item.get("name_zh") or item["canonical_name"]
    value = item.get("value")
    if value is None:
        value_s = str(item.get("value_text") or "")
    else:
        value_s = f"{value:g}"
    unit = item.get("unit") or ""
    ref = item.get("ref_text") or ""
    flag = item.get("flag_raw") or ""
    return f"{name:<16}{value_s:>10}  {unit:<12}{ref:<18}{flag}"


def render_text(case: dict[str, Any], seed: int | None = None) -> str:
    """渲染成一份'看起来像真的'检验报告文本。"""
    rng = random.Random(seed if seed is not None else hash(case["case_id"]) & 0xFFFF)
    patient = case.get("patient") or {}
    sex = {"M": "男", "F": "女"}.get(patient.get("sex"), "")
    age = patient.get("age", "")

    header = [
        f"{case.get('hospital', 'XX医院'):^56}",
        f"{'检 验 报 告 单':^54}",
        "=" * 72,
        f"姓名：{rng.choice(_FAKE_NAMES)}    性别：{sex}    年龄：{age}    "
        f"样本号：{case['case_id'].replace('case_', '2026')}00{rng.randint(1, 9)}",
        f"科室：{rng.choice(_FAKE_DEPT)}    标本类型：{'尿液' if case.get('report_type') == '尿常规' else '全血/血清'}"
        f"    采样日期：{case.get('collected_at', '')}",
        f"检验项目：{case.get('report_type', '')}",
        "=" * 72,
        f"{'项目名称':<16}{'结果':>10}  {'单位':<12}{'参考区间':<18}提示",
        "-" * 72,
    ]
    body = [_row(i) for i in case["items"]]
    footer = [
        "-" * 72,
        "本报告仅对送检标本负责。如有疑问请于 7 日内与检验科联系。",
        f"报告时间：{case.get('collected_at', '')}    检验者：***    审核者：***",
    ]
    return "\n".join(header + body + footer)


# --------------------------------------------------------------------------- #
# 图片渲染（可选，需要 Pillow + 中文字体）
# --------------------------------------------------------------------------- #
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


def available_font() -> str | None:
    for f in _FONT_CANDIDATES:
        if Path(f).is_file():
            return f
    return None


def render_image(case: dict[str, Any], out_path: str | Path, seed: int | None = None) -> Path | None:
    """把报告渲染成 PNG 图片，用于评测视觉模型的抽取能力。

    缺少 Pillow 或中文字体时返回 None（不抛异常），调用方据此跳过图片评测。
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None

    font_path = available_font()
    if not font_path:
        return None

    text = render_text(case, seed=seed)
    lines = text.splitlines()

    font_size, pad, line_h = 18, 28, 26
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        return None

    width = 900
    height = pad * 2 + line_h * len(lines)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    y = pad
    for line in lines:
        draw.text((pad, y), line, fill=(20, 20, 20), font=font)
        y += line_h

    # 轻微添加扫描件质感（噪声），让视觉模型面对的不是完美排版
    rng = random.Random((seed if seed is not None else hash(case["case_id"])) & 0xFFFF)
    pixels = img.load()
    for _ in range(int(width * height * 0.004)):
        x, yy = rng.randrange(width), rng.randrange(height)
        r, g, b = pixels[x, yy]
        delta = rng.randint(-22, 8)
        pixels[x, yy] = (max(0, min(255, r + delta)), max(0, min(255, g + delta)), max(0, min(255, b + delta)))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out
