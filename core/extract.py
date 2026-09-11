"""Stage 1：把检验报告图像/文本抽取成结构化裸数据。

这一层的职责边界要极其清楚：**只抄，不算，不判断**。

具体约束（都写进了提示词，并由 JSON Schema 二次约束）：
- 参考区间必须原样抄报告上的，抄不到就 ``null``——绝不允许模型自己"想一个正常范围"
- 数值结果原样保留字符串形式（``<0.01``、``阴性``、``±``），解析交给 ``core.units``
- 不做任何"H/L/异常"判断，那是 ``core.rules`` 的职责
- 不认识的栏目可以跳过，但不能编造

为什么坚持"抄而不算"：Meyer 等（2025）在 24.6 万个参考区间上测得大模型自报区间的
下限平均变异系数达 26.5%。让模型"顺便判断一下高不高"是这个项目最容易翻车的地方。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .providers import LLMClient, LLMError, encode_image_data_url
from .schema import EXTRACTION_JSON_SCHEMA, RawLabReport

__all__ = ["ExtractionResult", "build_extraction_messages", "extract_report", "EXTRACTION_SYSTEM_PROMPT"]


EXTRACTION_SYSTEM_PROMPT = """\
你是一个医学检验报告结构化抽取器。你的唯一任务是把报告上**印出来的内容**照抄成 JSON。

严格遵守以下规则，违反任何一条都算任务失败：

1. **只抽取，不计算，不判断。** 你不需要、也不允许判断任何指标是否异常。
   不要输出"偏高""异常""正常"这类结论，也不要输出 H/L/箭头标记之外的任何评价。
2. **参考区间必须原样照抄。** 报告上写"3.5-9.5"就抄"3.5-9.5"，写"<5.0"就抄"<5.0"，
   写"阴性"就抄"阴性"。**如果报告上没有印参考区间，必须填 null。**
   绝对不允许你根据医学知识补一个"常见参考范围"——不同医院仪器试剂不同，区间本来就不同，
   你补的区间会导致完全错误的判定。
3. **结果值原样保留字符串。** "13.2"就写"13.2"，"<0.01"就写"<0.01"，"阴性"就写"阴性"，
   "±"就写"±"，"1+"就写"1+"。不要换算单位，不要转换格式，不要四舍五入。
4. **单位原样照抄**，不要归一化（"×10^9/L"就写"×10^9/L"）。
5. **不要编造。** 看不清的字段填 null。没把握的项目条数少一点没关系，编造条目是严重错误。
6. **不要遗漏。** 报告上有多少项目就抽多少条，包括看起来正常的项目。
7. 如果报告里有明显的姓名、身份证号、住址、电话等个人身份信息，**不要抽取到输出中**。

输出必须是一个 JSON 对象，结构如下：
{
  "report_type": "血常规 / 生化全套 / 尿常规 ...",
  "hospital": "出具机构名称或 null",
  "collected_at": "YYYY-MM-DD 或 null",
  "patient_sex": "男/女 或 null",
  "patient_age": 整数或 null,
  "items": [
    {
      "raw_name": "报告上原样印出的项目名",
      "result_raw": "原样结果字符串",
      "unit_raw": "原样单位字符串",
      "ref_raw": "原样参考区间字符串，没有则 null",
      "flag_raw": "报告自带的箭头或字母标记，没有则 null",
      "page": 1,
      "source_text": "该项目所在行的原始文本"
    }
  ],
  "extraction_notes": ["任何存疑之处，例如某行字迹不清"]
}

只输出 JSON，不要任何解释文字，不要 Markdown 代码围栏。"""


@dataclass
class ExtractionResult:
    raw: RawLabReport
    model_used: str
    used_vision: bool
    raw_response: dict[str, Any] | None = None


def build_extraction_messages(
    images: Sequence[str | Path | bytes] | None = None,
    text: str | None = None,
    hint: str | None = None,
) -> list[dict]:
    """构造抽取请求消息。支持纯图片、纯文本、或图文混合。"""
    user_content: list[dict] = [
        {
            "type": "text",
            "text": (
                "请把下面这份检验报告抽取成约定的 JSON 结构。"
                "记住：参考区间抄不到就填 null，不要用你的医学知识补。"
                + (f"\n\n补充提示：{hint}" if hint else "")
            ),
        }
    ]

    for img in images or []:
        user_content.append({"type": "image_url", "image_url": {"url": encode_image_data_url(img)}})

    if text:
        user_content.append({"type": "text", "text": f"\n\n报告文本内容如下：\n```\n{text}\n```"})

    return [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": user_content if len(user_content) > 1 else user_content[0]["text"],
        },
    ]


def extract_report(
    client: LLMClient,
    images: Sequence[str | Path | bytes] | None = None,
    text: str | None = None,
    hint: str | None = None,
    model: str | None = None,
) -> ExtractionResult:
    """执行 Stage 1 抽取。

    Raises
    ------
    LLMError
        模型调用失败，或返回内容无法构成合法的 ``RawLabReport``。
    """
    if not images and not text:
        raise LLMError("必须提供报告图片或报告文本。")

    messages = build_extraction_messages(images=images, text=text, hint=hint)
    payload = client.chat_json(messages, model=model)

    try:
        raw = RawLabReport.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - 统一转成 LLMError，交由上层展示
        raise LLMError(f"模型输出不符合抽取契约：{exc}") from exc

    if not raw.items:
        raise LLMError(
            "模型未抽取到任何检验项目。可能原因：图片不清晰、不是检验报告、"
            "或报告版式过于特殊。"
        )

    used_vision = bool(images)
    model_name = model or getattr(client, "vision_model", None) or getattr(client, "model", "unknown")
    return ExtractionResult(raw=raw, model_used=str(model_name), used_vision=used_vision)


def schema_for_prompt() -> str:
    """把 JSON Schema 渲染成可嵌入提示词的紧凑字符串（供自定义提示词使用）。"""
    return json.dumps(EXTRACTION_JSON_SCHEMA, ensure_ascii=False, indent=2)
