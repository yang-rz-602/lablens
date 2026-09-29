"""MCP 冒烟校验：解析 stdio 输出，断言握手、工具枚举与危急值判读都正常。

为什么不用 grep
---------------
MCP 的工具结果是把**业务 JSON 当作字符串**放在 ``result.content[0].text`` 里的。
于是那份 JSON 在原始 stdout 中是转义形态::

    "text": "{\\n  \\"items\\": [ … \\"flag\\": \\"CH\\" …"

而 ``grep '"flag": "CH"'`` 找的是字面量 ``"flag": "CH"``——``flag`` 后面紧跟的是
反斜杠再引号，所以**永远匹配不上**。这条断言因此从仓库建立起就一直红着
（git 历史里 8 次 CI 运行全部在 MCP 冒烟失败），而主测试矩阵一直是绿的。

改成解析之后，可以断言真正关心的语义，而不是碰字符串。

用法::

    python mcp_server.py < 请求.jsonl > mcp_out.jsonl
    python scripts/check_mcp_smoke.py mcp_out.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXPECTED_TOOLS = {
    "judge_lab_items",
    "explain_lab_result",
    "lookup_indicator",
    "list_indicators",
    "convert_unit",
}


def main(path: str = "mcp_out.jsonl") -> int:
    raw = Path(path).read_text(encoding="utf-8")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        print(f"[失败] {path} 是空的：MCP server 没有输出任何响应", file=sys.stderr)
        return 1

    try:
        replies = [json.loads(ln) for ln in lines]
    except json.JSONDecodeError as exc:
        print(f"[失败] 输出不是合法的换行分隔 JSON：{exc}", file=sys.stderr)
        return 1

    by_id = {r.get("id"): r for r in replies}

    # ---- 1) 握手 ----
    init = by_id.get(1, {}).get("result")
    if not init:
        print(f"[失败] 没有收到 initialize 的 result：{by_id.get(1)}", file=sys.stderr)
        return 1
    name = (init.get("serverInfo") or {}).get("name")
    if name != "lablens":
        print(f"[失败] serverInfo.name 期望 lablens，实际 {name!r}", file=sys.stderr)
        return 1
    if not init.get("instructions"):
        print("[失败] initialize 未返回 instructions（Agent 就不知道该怎么用这个工具）", file=sys.stderr)
        return 1

    # ---- 2) 工具枚举 ----
    listed = (by_id.get(2, {}).get("result") or {}).get("tools")
    if not listed:
        print(f"[失败] tools/list 没返回工具：{by_id.get(2)}", file=sys.stderr)
        return 1
    tools = {t.get("name") for t in listed}
    missing = EXPECTED_TOOLS - tools
    if missing:
        print(f"[失败] 缺少工具：{sorted(missing)}（实际 {sorted(tools)}）", file=sys.stderr)
        return 1

    # ---- 3) 核心：危急值判读（含字段级溯源）----
    call = by_id.get(3, {})
    content = (call.get("result") or {}).get("content") or []
    if not content:
        print(f"[失败] judge_lab_items 没有返回 content：{call}", file=sys.stderr)
        return 1
    payload = json.loads(content[0]["text"])

    item = (payload.get("items") or [{}])[0]
    checks = {
        "canonical_name == 'k'": item.get("canonical_name") == "k",
        "flag == 'CH'": item.get("flag") == "CH",
        "severity == 'critical'": item.get("severity") == "critical",
        "crossed == 'critical_high'": (item.get("decision_trace") or {}).get("crossed") == "critical_high",
        "critical_items 非空": bool(payload.get("critical_items")),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        print(f"[失败] 判读断言未通过：{failed}", file=sys.stderr)
        print(json.dumps(payload, ensure_ascii=False, indent=2)[:1200], file=sys.stderr)
        return 1

    print("MCP 冒烟通过：握手 + 工具枚举 + 危急值判读（flag=CH, critical_high, 含 decision_trace）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "mcp_out.jsonl"))
