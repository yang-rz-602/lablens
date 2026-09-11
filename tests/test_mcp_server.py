"""单元测试：MCP Server（JSON-RPC over stdio）。

MCP 层是把"确定性判读"暴露给 Agent 的出口，它本身必须可靠：
协议握手、工具枚举、参数校验、错误处理都要有明确行为。
"""

from __future__ import annotations

import io
import json

import mcp_server
from mcp_server import TOOLS, handle, serve


def call(method: str, params: dict | None = None, msg_id: int = 1) -> dict:
    return handle({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}})


def call_tool(name: str, arguments: dict) -> dict:
    resp = call("tools/call", {"name": name, "arguments": arguments})
    assert "result" in resp, resp
    payload = json.loads(resp["result"]["content"][0]["text"])
    return payload


# --------------------------------------------------------------------------- #
# 协议
# --------------------------------------------------------------------------- #
def test_initialize_echoes_client_protocol_version() -> None:
    resp = call("initialize", {"protocolVersion": "2025-06-18"})
    assert resp["result"]["protocolVersion"] == "2025-06-18"
    assert resp["result"]["serverInfo"]["name"] == "lablens"
    assert "tools" in resp["result"]["capabilities"]


def test_initialize_without_version_uses_default() -> None:
    resp = call("initialize", {})
    assert resp["result"]["protocolVersion"] == mcp_server.PROTOCOL_VERSION


def test_initialized_notification_has_no_response() -> None:
    assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_ping() -> None:
    assert call("ping")["result"] == {}


def test_tools_list_exposes_expected_tools() -> None:
    names = {t["name"] for t in call("tools/list")["result"]["tools"]}
    assert names == set(TOOLS)
    assert "judge_lab_items" in names


def test_every_tool_has_description_and_schema() -> None:
    for tool in call("tools/list")["result"]["tools"]:
        assert tool["description"]
        assert tool["inputSchema"]["type"] == "object"


def test_unknown_method_returns_error() -> None:
    resp = call("nonexistent/method")
    assert resp["error"]["code"] == -32601


def test_unknown_tool_returns_error() -> None:
    resp = call("tools/call", {"name": "nope", "arguments": {}})
    assert "error" in resp


def test_bad_arguments_reported_as_tool_error_not_crash() -> None:
    resp = call("tools/call", {"name": "lookup_indicator", "arguments": {"wrong": 1}})
    assert resp["result"]["isError"] is True


# --------------------------------------------------------------------------- #
# judge_lab_items —— 核心工具
# --------------------------------------------------------------------------- #
def test_judge_basic_normal() -> None:
    out = call_tool("judge_lab_items", {
        "items": [{"name": "白细胞计数", "result": "6.2", "unit": "10^9/L", "reference": "3.5-9.5"}],
        "sex": "M",
    })
    item = out["items"][0]
    assert item["flag"] == "N"
    assert item["reference_source"] == "report"


def test_judge_detects_critical_value() -> None:
    out = call_tool("judge_lab_items", {
        "items": [{"name": "钾", "result": "6.8", "unit": "mmol/L", "reference": "3.5-5.3"}],
        "sex": "M",
    })
    assert out["items"][0]["flag"] == "CH"
    assert out["items"][0]["severity"] == "critical"
    assert out["critical_items"]


def test_judge_uses_report_reference_over_knowledge_base() -> None:
    """报告自带区间必须优先——这是产品的核心机制。"""
    out = call_tool("judge_lab_items", {
        "items": [{"name": "白细胞计数", "result": "9.8", "unit": "10^9/L", "reference": "4.0-10.0"}],
        "sex": "M",
    })
    item = out["items"][0]
    assert item["reference_source"] == "report"
    assert item["flag"] == "N", "按报告区间 9.8 在 4.0-10.0 内"


def test_judge_refuses_without_reference() -> None:
    """没有参考区间时必须拒判，而不是猜一个。"""
    out = call_tool("judge_lab_items", {
        "items": [{"name": "某个不存在的指标", "result": "12.3", "unit": "U/L"}],
    })
    assert out["items"][0]["flag"] == "?"
    assert out["items"][0]["decision_trace"]["rule"] == "refused_no_interval"


def test_judge_sex_specific_interval() -> None:
    female = call_tool("judge_lab_items", {"items": [{"name": "血红蛋白", "result": "120"}], "sex": "F"})
    male = call_tool("judge_lab_items", {"items": [{"name": "血红蛋白", "result": "120"}], "sex": "M"})
    assert female["items"][0]["flag"] == "N"
    assert male["items"][0]["flag"] == "L"


def test_judge_qualitative() -> None:
    out = call_tool("judge_lab_items", {
        "items": [{"name": "尿蛋白", "result": "+", "reference": "阴性"}],
    })
    assert out["items"][0]["flag"] == "H"


def test_judge_carries_decision_trace() -> None:
    out = call_tool("judge_lab_items", {
        "items": [{"name": "白细胞计数", "result": "12.0", "unit": "10^9/L"}],
        "sex": "M",
    })
    trace = out["items"][0]["decision_trace"]
    assert trace["rule"] == "kb_interval"
    assert trace["interval_source"] == "内置知识库"
    assert trace["interval_standard"]
    assert trace["crossed"] == "ref_high"


def test_judge_includes_disclaimer() -> None:
    out = call_tool("judge_lab_items", {"items": [{"name": "白细胞计数", "result": "6.0"}]})
    assert "不构成医学诊断" in out["disclaimer"]


def test_judge_does_not_call_llm() -> None:
    """判读必须是纯确定性的：即使没有任何 API Key 也要能跑。"""
    import os

    saved = {k: os.environ.pop(k, None) for k in
             ("DASHSCOPE_API_KEY", "ZHIPU_API_KEY", "OPENAI_API_KEY", "LABLENS_BASE_URL")}
    try:
        out = call_tool("judge_lab_items", {"items": [{"name": "白细胞计数", "result": "6.0"}]})
        assert out["items"][0]["flag"] == "N"
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


# --------------------------------------------------------------------------- #
# 其它工具
# --------------------------------------------------------------------------- #
def test_lookup_indicator_found() -> None:
    out = call_tool("lookup_indicator", {"name": "WBC"})
    assert out["found"] is True
    assert out["canonical_name"] == "wbc"
    assert out["source"]


def test_lookup_indicator_not_found_does_not_fuzzy_match() -> None:
    out = call_tool("lookup_indicator", {"name": "白细胞介素6"})
    assert out["found"] is False
    assert "模糊匹配" in out["message"]


def test_list_indicators() -> None:
    out = call_tool("list_indicators", {})
    assert out["count"] > 50
    assert "血常规" in out["categories"]


def test_list_indicators_filtered() -> None:
    out = call_tool("list_indicators", {"category": "肝功能"})
    assert out["count"] > 0
    assert all(i["category"] == "肝功能" for i in out["indicators"])


def test_convert_unit_needs_registered_factor() -> None:
    out = call_tool("convert_unit", {"value": 1.2, "from_unit": "mg/dL", "to_unit": "μmol/L"})
    assert out["converted"] is False
    assert "未登记" in out["note"] or "无需换算" in out["note"] or out["note"]


def test_convert_unit_same_unit_is_noop() -> None:
    out = call_tool("convert_unit", {"value": 5.0, "from_unit": "g/L", "to_unit": "g/L"})
    assert out["converted"] is False


def test_explain_runs_offline_and_abstains() -> None:
    """解释工具在没有 API Key 时必须能降级运行，且对无依据指标拒答。"""
    import os

    saved = {k: os.environ.pop(k, None) for k in
             ("DASHSCOPE_API_KEY", "ZHIPU_API_KEY", "OPENAI_API_KEY", "LABLENS_BASE_URL")}
    try:
        out = call_tool("explain_lab_result", {
            "items": [{"name": "白细胞计数", "result": "12.0", "unit": "10^9/L"}],
            "sex": "M",
        })
        assert out["used_llm"] is False
        assert out["overview"]
        assert out["disclaimer"]
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


# --------------------------------------------------------------------------- #
# stdio 循环
# --------------------------------------------------------------------------- #
def test_serve_handles_newline_delimited_json() -> None:
    lines = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        "",  # 空行应被跳过
        "这不是 JSON",  # 坏行应被跳过而不是崩溃
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),  # 无响应
    ]) + "\n"

    out = io.StringIO()
    serve(instream=io.StringIO(lines), outstream=out)
    responses = [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]
    assert len(responses) == 2, "通知与坏行都不应产生响应"
    assert responses[0]["id"] == 1
    assert responses[1]["id"] == 2
