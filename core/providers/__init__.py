"""LLM 传输层：只负责"把消息发出去、把文本拿回来"，不含任何业务逻辑。

设计取舍
--------
- 统一走 **OpenAI 兼容协议**。国内的通义千问（DashScope）、智谱 GLM、SiliconFlow、
  DeepSeek，以及本地的 vLLM / Ollama 都提供该协议，因此一套客户端就能全部覆盖，
  切换供应商只改 base_url 和模型名。
- **没有 API Key 也能跑**。``FixtureClient`` 让整条流水线（规则引擎 + 检索 + 解释 +
  评测）在离线状态下可运行，CI 里不依赖任何外部服务。这是刻意的：把产品的
  可验证性从"有没有额度"里解耦出来。
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "Preset",
    "PRESETS",
    "LLMClient",
    "OpenAICompatClient",
    "FixtureClient",
    "LLMError",
    "encode_image_data_url",
    "extract_json",
]


class LLMError(RuntimeError):
    """调用模型失败（网络、鉴权、限流、返回格式异常）。"""


# --------------------------------------------------------------------------- #
# 供应商预设
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    base_url: str
    vision_model: str
    text_model: str
    api_key_env: str
    docs: str = ""


PRESETS: dict[str, Preset] = {
    "dashscope": Preset(
        "dashscope", "通义千问 · 阿里云百炼",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen3-vl-plus", "qwen-plus", "DASHSCOPE_API_KEY",
        "https://help.aliyun.com/zh/model-studio/",
    ),
    "zhipu": Preset(
        "zhipu", "智谱 GLM",
        "https://open.bigmodel.cn/api/paas/v4",
        "glm-4v-plus", "glm-4-plus", "ZHIPU_API_KEY",
        "https://docs.bigmodel.cn/",
    ),
    "siliconflow": Preset(
        "siliconflow", "SiliconFlow 硅基流动",
        "https://api.siliconflow.cn/v1",
        "Qwen/Qwen2.5-VL-72B-Instruct", "Qwen/Qwen2.5-72B-Instruct", "SILICONFLOW_API_KEY",
        "https://docs.siliconflow.cn/",
    ),
    "openai": Preset(
        "openai", "OpenAI",
        "https://api.openai.com/v1",
        "gpt-4o", "gpt-4o-mini", "OPENAI_API_KEY",
        "https://platform.openai.com/docs/",
    ),
    "deepseek": Preset(
        "deepseek", "DeepSeek（无视觉，仅可用于解释层）",
        "https://api.deepseek.com/v1",
        "", "deepseek-chat", "DEEPSEEK_API_KEY",
        "https://api-docs.deepseek.com/",
    ),
    "local": Preset(
        "local", "本地 vLLM / Ollama",
        "http://localhost:8000/v1",
        "Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3-8B-Instruct", "LOCAL_LLM_API_KEY",
        "https://docs.vllm.ai/",
    ),
}


# --------------------------------------------------------------------------- #
# 接口
# --------------------------------------------------------------------------- #
class LLMClient(Protocol):
    """LLM 客户端接口。实现它即可接入任意后端。"""

    def chat(
        self,
        messages: Sequence[dict],
        model: str | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str: ...

    def chat_json(
        self,
        messages: Sequence[dict],
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict: ...


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def encode_image_data_url(source: str | Path | bytes, mime: str | None = None) -> str:
    """把图片编码成 data URL，供多模态接口使用。"""
    if isinstance(source, bytes):
        raw, mime_type = source, mime or "image/png"
    else:
        p = Path(source)
        raw = p.read_bytes()
        mime_type = mime or mimetypes.guess_type(p.name)[0] or "image/png"
    return f"data:{mime_type};base64,{base64.b64encode(raw).decode('ascii')}"


def extract_json(text: str) -> dict:
    """从模型输出里稳健地取出 JSON 对象。

    模型经常在 JSON 外包一层 ```json 围栏，或前后带一句解释。这里做容错，
    但仍要求结果必须是 JSON 对象——解析失败就抛错，让上层触发一次修复重试，
    而不是猜。
    """
    if not text:
        raise LLMError("模型返回为空。")
    s = text.strip()

    if s.startswith("```"):
        s = s.split("```")[1] if "```" in s[3:] else s[3:]
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
        s = s.strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 扫描第一个平衡的 {...}
    start = s.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = s.find("{", start + 1)

    raise LLMError(f"无法从模型输出中解析出 JSON 对象：{text[:200]!r}")


# --------------------------------------------------------------------------- #
# OpenAI 兼容实现
# --------------------------------------------------------------------------- #
@dataclass
class OpenAICompatClient:
    """基于 OpenAI 兼容协议的客户端（httpx 传输）。"""

    base_url: str
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    vision_model: str | None = None
    timeout: float = 120.0
    max_retries: int = 2
    temperature: float = 0.2
    extra_headers: dict[str, str] = field(default_factory=dict)

    # ---------------- 构造 ---------------- #
    @classmethod
    def from_preset(cls, key: str, api_key: str | None = None, **kw: Any) -> OpenAICompatClient:
        preset = PRESETS.get(key)
        if preset is None:
            raise LLMError(f"未知识别供应商：{key}。可用：{', '.join(PRESETS)}")
        key_val = api_key or os.environ.get(preset.api_key_env) or ""
        return cls(
            base_url=kw.pop("base_url", preset.base_url),
            api_key=key_val or None,
            model=kw.pop("model", preset.text_model),
            vision_model=kw.pop("vision_model", preset.vision_model or None),
            **kw,
        )

    @classmethod
    def from_env(cls, **kw: Any) -> OpenAICompatClient:
        """按环境变量自动探测可用的供应商，优先视觉能力完整的。"""
        order = os.environ.get("LABLENS_PROVIDER")
        keys = [order] if order else ["dashscope", "zhipu", "siliconflow", "openai", "deepseek"]
        for k in keys:
            if not k or k not in PRESETS:
                continue
            if os.environ.get(PRESETS[k].api_key_env):
                return cls.from_preset(k, **kw)
        if os.environ.get("LABLENS_BASE_URL"):
            return cls(
                base_url=os.environ["LABLENS_BASE_URL"],
                api_key=os.environ.get("LABLENS_API_KEY"),
                model=os.environ.get("LABLENS_MODEL", "gpt-4o-mini"),
                vision_model=os.environ.get("LABLENS_VISION_MODEL") or None,
                **kw,
            )
        raise LLMError(
            "未检测到任何可用的模型凭证。请设置 DASHSCOPE_API_KEY / ZHIPU_API_KEY 等环境变量，"
            "或设置 LABLENS_BASE_URL 指向本地推理服务。"
        )

    @property
    def supports_vision(self) -> bool:
        return bool(self.vision_model)

    # ---------------- 调用 ---------------- #
    def chat(
        self,
        messages: Sequence[dict],
        model: str | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise LLMError("缺少 httpx，请先 `pip install httpx`。") from exc

        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": list(messages),
            "temperature": self.temperature if temperature is None else temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = self.base_url.rstrip("/") + "/chat/completions"
        last_err: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, json=payload, headers=headers)
                if resp.status_code >= 400:
                    body = resp.text[:400]
                    # 部分供应商不支持 response_format，降级重试一次
                    if resp.status_code == 400 and "response_format" in body and json_mode:
                        payload.pop("response_format", None)
                        continue
                    raise LLMError(f"模型接口返回 {resp.status_code}：{body}")
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise LLMError(f"模型返回中没有 choices：{str(data)[:300]}")
                return (choices[0].get("message") or {}).get("content") or ""
            except LLMError:
                raise
            except Exception as exc:  # noqa: BLE001 - 网络类异常统一重试
                last_err = exc
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
        raise LLMError(f"调用模型失败（已重试 {self.max_retries} 次）：{last_err}")

    def chat_json(
        self,
        messages: Sequence[dict],
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict:
        """要求模型输出 JSON 对象；解析失败时追加一次"修复"请求。"""
        text = self.chat(messages, model=model, temperature=temperature, json_mode=True)
        try:
            return extract_json(text)
        except LLMError:
            repair = list(messages) + [
                {"role": "assistant", "content": text[:2000]},
                {
                    "role": "user",
                    "content": "上面的输出不是合法 JSON。请只输出一个合法的 JSON 对象，"
                               "不要任何解释文字、不要 Markdown 代码围栏。",
                },
            ]
            return extract_json(self.chat(repair, model=model, temperature=0))


# --------------------------------------------------------------------------- #
# 离线夹具客户端
# --------------------------------------------------------------------------- #
@dataclass
class FixtureClient:
    """离线夹具客户端：按顺序或按提示词关键字返回预置结果。

    用途：
    - CI 与单测中验证整条流水线，不依赖网络与额度
    - 没有 API Key 时演示完整界面
    """

    extraction: dict | None = None
    responses: list[str] = field(default_factory=list)
    _calls: int = 0

    def chat(
        self,
        messages: Sequence[dict],
        model: str | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        if json_mode and self.extraction is not None:
            return json.dumps(self.extraction, ensure_ascii=False)
        if self.responses:
            out = self.responses[self._calls % len(self.responses)]
            self._calls += 1
            return out
        return json.dumps(self.extraction or {"items": []}, ensure_ascii=False)

    def chat_json(
        self,
        messages: Sequence[dict],
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict:
        return dict(self.extraction or {"items": []})
