"""LabLens MCP Server —— 把"确定性判读层"暴露成 Agent 可调用的工具。

为什么这个文件值得单独存在
--------------------------
调研（GitHub 开源现状，2026-09）的结论是：医疗 MCP 生态**几乎全是 FHIR/EHR 数据通道**
（wso2/fhir-mcp-server、the-momentum/fhir-mcp-server 等），
**没有"临床判读工具"**——即"单位换算 + 参考区间 + 危急值 + 指标归一化"这一层。

而这一层恰恰**最适合用确定性代码实现**，也正是 LLM 最不该自己做的事：
让模型"顺手判断一下这个值高不高"是这个领域最典型的失败模式
（Meyer 2025：模型自报参考区间下限 CV 达 26.5%；Lab-AI：无检索时区间检索准确率仅 42.9%）。

所以这里把判读层封装成 MCP 工具，让任何 Agent 都能"调用一个可靠的判定器"，
而不是"相信模型心算"。

实现说明
--------
**零依赖**：手写 JSON-RPC 2.0 over stdio（MCP 的标准传输即换行分隔的 JSON），
不需要 `mcp` SDK。这样在任何 Python 环境里都能直接跑，也不会因为 SDK 版本变化而失效。

用法::

    python mcp_server.py          # 以 stdio 方式运行，供 MCP 客户端接入

Claude Desktop / 其他 MCP 客户端的配置片段::

    {
      "mcpServers": {
        "lablens": {
          "command": "python",
          "args": ["/absolute/path/to/mcp_server.py"]
        }
      }
    }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.explain import DISCLAIMER, interpret, render_ref, render_value  # noqa: E402
from core.providers import OpenAICompatClient  # noqa: E402
from core.retriever import Retriever, load_corpus  # noqa: E402
from core.rules import KnowledgeBase, build_report, load_knowledge_base  # noqa: E402
from core.schema import PatientContext, RawLabItem, RawLabReport  # noqa: E402

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "lablens", "version": "0.1.0"}

DATA_DIR = ROOT / "data"

_KB: KnowledgeBase | None = None
_RETRIEVER: Retriever | None = None


def kb() -> KnowledgeBase:
    global _KB
    if _KB is None:
        _KB = load_knowledge_base(DATA_DIR)
    return _KB


def retriever() -> Retriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever(load_corpus(DATA_DIR / "corpus.jsonl"))
    return _RETRIEVER


# --------------------------------------------------------------------------- #
# 工具实现
# --------------------------------------------------------------------------- #
def tool_list_indicators(category: str | None = None) -> dict:
    """列出知识库中收录的检验指标。"""
    rows = []
    for ind in sorted(kb().indicators.values(), key=lambda i: (i.category or "", i.canonical_name)):
        if category and ind.category != category:
            continue
        lo, hi, qual = ind.ref_for(PatientContext())
        rows.append(
            {
                "canonical_name": ind.canonical_name,
                "name_zh": ind.name_zh,
                "category": ind.category,
                "specimen": ind.specimen,
                "unit": ind.unit_canonical,
                "loinc": ind.loinc,
                "reference_interval": "定性" if ind.qualitative else _fmt(lo, hi),
                "source": ind.source,
            }
        )
    categories = sorted({i.category for i in kb().indicators.values() if i.category})
    return {"count": len(rows), "categories": categories, "indicators": rows}


def tool_lookup_indicator(name: str) -> dict:
    """按名称（支持中英文别名）查询单个指标的定义与参考区间。"""
    ind = kb().resolve(name)
    if ind is None:
        return {
            "found": False,
            "query": name,
            "message": f"知识库中未收录「{name}」。为避免套用错误区间，本工具不会做模糊匹配。",
        }
    result: dict[str, Any] = {
        "found": True,
        "canonical_name": ind.canonical_name,
        "name_zh": ind.name_zh,
        "aliases": ind.aliases,
        "loinc": ind.loinc,
        "category": ind.category,
        "specimen": ind.specimen,
        "unit": ind.unit_canonical,
        "qualitative": ind.qualitative,
        "reference_by_sex": ind.ref,
        "critical": ind.critical,
        "unit_conversions": ind.unit_conversions,
        "source": ind.source,
        "note": ind.note,
    }
    return result


def tool_judge_items(items: list[dict], sex: str | None = None, age: int | None = None) -> dict:
    """**核心工具**：对一批检验指标做确定性判读。

    只做数值与区间比较、单位换算、危急值识别，不调用任何语言模型，
    因此结果可复现、可审计。每个判定都带 ``decision_trace`` 说明它用了哪条区间。
    """
    patient = PatientContext(sex=sex, age=age)
    raw_items = [
        RawLabItem(
            raw_name=str(it.get("name") or it.get("raw_name") or ""),
            result_raw=None if it.get("result") is None else str(it.get("result")),
            unit_raw=it.get("unit"),
            ref_raw=it.get("reference"),
            page=it.get("page"),
            source_text=it.get("source_text"),
        )
        for it in items
    ]
    report = build_report(
        RawLabReport(patient_sex=patient.sex, patient_age=patient.age, items=raw_items),
        kb(),
        model_used="deterministic",
    )

    out_items = []
    for item in report.items:
        trace = item.decision_trace
        out_items.append(
            {
                "name": item.raw_name,
                "canonical_name": item.canonical_name,
                "value": item.value,
                "value_text": item.value_text,
                "unit": item.unit,
                "specimen": item.specimen,
                "reference_interval": render_ref(item),
                "reference_source": item.ref_source.value,
                "flag": item.flag.value,
                "flag_label": item.flag.label_zh,
                "severity": item.severity.value,
                "decision_trace": trace.model_dump() if trace else None,
                "notes": item.rules_notes,
                "source_text": item.source_text,
            }
        )

    counts = report.summary_counts()
    return {
        "items": out_items,
        "summary": counts,
        "critical_items": [
            {"name": i.raw_name, "value": render_value(i), "flag": i.flag.value}
            for i in report.critical_items
        ],
        "warnings": report.warnings,
        "disclaimer": DISCLAIMER,
    }


def tool_explain(items: list[dict], sex: str | None = None, age: int | None = None) -> dict:
    """在确定性判读的基础上生成通俗解读（检索知识库，无依据则拒答）。"""
    patient = PatientContext(sex=sex, age=age)
    raw_items = [
        RawLabItem(
            raw_name=str(it.get("name") or it.get("raw_name") or ""),
            result_raw=None if it.get("result") is None else str(it.get("result")),
            unit_raw=it.get("unit"),
            ref_raw=it.get("reference"),
        )
        for it in items
    ]
    report = build_report(
        RawLabReport(patient_sex=patient.sex, patient_age=patient.age, items=raw_items), kb()
    )

    try:
        client = OpenAICompatClient.from_env()
    except Exception:  # noqa: BLE001 - 没有凭证就走抽取式降级路径
        client = None

    result = interpret(report, client=client, retriever=retriever())
    interp = result.interpretation
    return {
        "overview": interp.overview,
        "urgent_notice": interp.urgent_notice,
        "items": [
            {
                "name": i.raw_name,
                "flag": i.flag.value,
                "one_liner": i.one_liner,
                "what_it_means": i.what_it_means,
                "possible_factors": i.possible_factors,
                "source_page": i.source_page,
            }
            for i in interp.items
        ],
        "questions_for_doctor": interp.questions_for_doctor,
        "next_steps": interp.next_steps,
        "limitations": interp.limitations,
        "abstained_indicators": result.abstained_names,
        "guard_hits": [h.category + "：" + h.sentence for h in result.guard_hits],
        "used_llm": result.used_llm,
        "disclaimer": interp.disclaimer,
    }


def tool_convert_unit(value: float, from_unit: str, to_unit: str) -> dict:
    """UCUM 风格的单位换算（只做知识库登记过的换算，不猜系数）。"""
    from core.units import convert_value, normalize_unit

    converted, changed, note = convert_value(value, from_unit, to_unit, [])
    return {
        "input": {"value": value, "unit": normalize_unit(from_unit)},
        "output": {"value": converted, "unit": normalize_unit(to_unit)},
        "converted": changed,
        "note": note or "两个单位已归一为同一单位，无需换算。",
        "hint": "跨项目换算需在该项目的 unit_conversions 中登记因子，可用 lookup_indicator 查询。",
    }


def _fmt(lo: float | None, hi: float | None) -> str:
    if lo is not None and hi is not None:
        return f"{lo:g}-{hi:g}"
    if hi is not None:
        return f"<{hi:g}"
    if lo is not None:
        return f">{lo:g}"
    return "未收录"


# --------------------------------------------------------------------------- #
# 工具注册表
# --------------------------------------------------------------------------- #
TOOLS: dict[str, dict] = {
    "judge_lab_items": {
        "fn": tool_judge_items,
        "description": (
            "对一批检验指标做确定性判读：计算偏高/偏低/危急值，处理单位换算与性别年龄特异区间。"
            "不调用语言模型，结果可复现可审计，每个判定都附带 decision_trace 说明依据。"
            "**当你需要判断某个检验值是否异常时，应当调用本工具而不是自己推算参考区间。**"
        ),
        "inputSchema": {
            "type": "object",
            "required": ["items"],
            "properties": {
                "items": {
                    "type": "array",
                    "description": "检验项目列表",
                    "items": {
                        "type": "object",
                        "required": ["name", "result"],
                        "properties": {
                            "name": {"type": "string", "description": "项目名称，支持中英文别名"},
                            "result": {"type": "string", "description": "结果值原文，如 '13.2'、'阴性'、'<0.01'"},
                            "unit": {"type": ["string", "null"], "description": "单位原文"},
                            "reference": {
                                "type": ["string", "null"],
                                "description": "报告单自带的参考区间，如 '3.5-9.5'。**有就一定要传**，优先于内置知识库。",
                            },
                            "page": {"type": ["integer", "null"]},
                            "source_text": {"type": ["string", "null"]},
                        },
                    },
                },
                "sex": {"type": ["string", "null"], "enum": ["M", "F", None],
                        "description": "性别，用于选择性别特异参考区间"},
                "age": {"type": ["integer", "null"], "description": "年龄"},
            },
        },
    },
    "explain_lab_result": {
        "fn": tool_explain,
        "description": (
            "在先做确定性判读的基础上，生成面向普通人的通俗解读，并给出建议向医生提问的清单。"
            "输出内容经过合规护栏过滤（不含诊断结论、用药与剂量建议）。"
            "知识库中无可靠依据的指标会被明确标注为拒绝解释。"
        ),
        "inputSchema": {
            "type": "object",
            "required": ["items"],
            "properties": {
                "items": {"type": "array", "items": {"type": "object"}},
                "sex": {"type": ["string", "null"], "enum": ["M", "F", None]},
                "age": {"type": ["integer", "null"]},
            },
        },
    },
    "lookup_indicator": {
        "fn": tool_lookup_indicator,
        "description": "查询某个检验指标的定义、别名、LOINC 编码、参考区间、危急值阈值与方法学注意事项。",
        "inputSchema": {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string", "description": "指标名称，支持中英文别名"}},
        },
    },
    "list_indicators": {
        "fn": tool_list_indicators,
        "description": "列出知识库收录的全部检验指标及其参考区间来源。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {"type": ["string", "null"], "description": "按类别过滤，如 '血常规'、'肝功能'"}
            },
        },
    },
    "convert_unit": {
        "fn": tool_convert_unit,
        "description": "检验单位换算（UCUM 风格）。只做已登记的换算，不会擅自猜测换算系数。",
        "inputSchema": {
            "type": "object",
            "required": ["value", "from_unit", "to_unit"],
            "properties": {
                "value": {"type": "number"},
                "from_unit": {"type": "string"},
                "to_unit": {"type": "string"},
            },
        },
    },
}


# --------------------------------------------------------------------------- #
# JSON-RPC over stdio
# --------------------------------------------------------------------------- #
def _ok(msg_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _err(msg_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def handle(msg: dict) -> dict | None:
    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        # 按规范回显客户端请求的协议版本，避免因版本差异被拒绝
        client_version = params.get("protocolVersion") or PROTOCOL_VERSION
        return _ok(msg_id, {
            "protocolVersion": client_version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": (
                "LabLens 提供**确定性**的检验报告判读能力。"
                "当需要判断检验值是否异常时，请调用 judge_lab_items，"
                "不要自己推算参考区间——不同实验室区间不同，模型自报区间不可靠。"
                "本服务只提供科普性解释，不构成诊断或治疗建议。"
            ),
        })

    if method in {"notifications/initialized", "initialized"}:
        return None  # 通知无需响应

    if method == "ping":
        return _ok(msg_id, {})

    if method == "tools/list":
        return _ok(msg_id, {
            "tools": [
                {"name": name, "description": spec["description"], "inputSchema": spec["inputSchema"]}
                for name, spec in TOOLS.items()
            ]
        })

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        spec = TOOLS.get(name)
        if spec is None:
            return _err(msg_id, -32602, f"未知工具：{name}")
        try:
            payload = spec["fn"](**args)
        except TypeError as exc:
            return _ok(msg_id, {
                "content": [{"type": "text", "text": f"参数错误：{exc}"}],
                "isError": True,
            })
        except Exception as exc:  # noqa: BLE001
            return _ok(msg_id, {
                "content": [{"type": "text", "text": f"执行失败：{type(exc).__name__}: {exc}"}],
                "isError": True,
            })
        return _ok(msg_id, {
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}],
            "isError": False,
        })

    if msg_id is None:
        return None
    return _err(msg_id, -32601, f"不支持的方法：{method}")


def serve(instream=None, outstream=None) -> None:
    """stdio 主循环：一行一个 JSON-RPC 消息。"""
    instream = instream or sys.stdin
    outstream = outstream or sys.stdout
    for line in instream:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = handle(msg)
        if response is not None:
            outstream.write(json.dumps(response, ensure_ascii=False) + "\n")
            outstream.flush()


if __name__ == "__main__":
    serve()
