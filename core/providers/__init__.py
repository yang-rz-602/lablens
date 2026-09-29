"""LLM 传输层：基于 **LangChain ChatModel** 抽象，不再手写 HTTP 请求。

设计取舍
--------
- 统一走 **OpenAI 兼容协议**。国内的通义千问（DashScope）、智谱 GLM、SiliconFlow、
  DeepSeek，以及本地的 vLLM / Ollama 都提供该协议，因此用 LangChain 官方的
  ``ChatOpenAI`` 集成 + 自定义 ``base_url`` 就能全部覆盖，切换供应商只改
  ``base_url`` 和模型名（见 ``PRESETS``）。
- **没有 API Key 也能跑**。``FakeLLMClient`` 让整条流水线（规则引擎 + 检索 + 解释 +
  评测）在离线状态下可运行，CI 里不依赖任何外部服务。这是刻意的：把产品的
  可验证性从"有没有额度"里解耦出来。
- 对外仍然暴露一个极窄的 ``LLMClient`` 协议（``chat`` / ``chat_json``），
  这样 ``core.extract`` 与 ``core.explain`` 不需要知道底层是 LangChain 还是别的。
  换成 LangGraph、原生 SDK 或本地推理服务时，只需再实现一次这个协议。
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

__all__ = [
    "Preset",
    "PRESETS",
    "LLMClient",
    "LangChainLLMClient",
    "FakeLLMClient",
    "LLMError",
    "build_chat_model",
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
    # LangChain ``init_chat_model`` 的提供方标识；这里的供应商都兼容 OpenAI 协议
    provider: str = "openai"


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


def _to_lc_messages(messages: Sequence[dict]) -> list[BaseMessage]:
    """把 OpenAI 风格的消息 dict 转成 LangChain 消息对象。

    ``content`` 既可以是一个字符串，也可以是 OpenAI 的多模态内容块列表
    （``[{"type": "text", ...}, {"type": "image_url", ...}]``）——后者原样透传，
    LangChain 与各家多模态接口都认这个结构。
    """
    mapping: dict[str, type[BaseMessage]] = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }
    out: list[BaseMessage] = []
    for m in messages:
        role = str(m.get("role") or "user")
        content = m.get("content") or ""
        out.append(mapping.get(role, HumanMessage)(content=content))
    return out


def _message_text(message: Any) -> str:
    """从 LangChain 消息里取出纯文本。

    LangChain 1.x 的 ``AIMessage.text`` 会把内容块里的 text 片段拼起来；
    这里再做一层兜底，保证不同提供方返回块状 content 时也能拿到字符串。
    """
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return ""


# --------------------------------------------------------------------------- #
# LangChain 模型构建
# --------------------------------------------------------------------------- #
def build_chat_model(
    base_url: str,
    model: str,
    api_key: str | None = None,
    temperature: float = 0.2,
    timeout: float = 120.0,
    max_retries: int = 2,
    **kwargs: Any,
) -> BaseChatModel:
    """构建一个 LangChain ``ChatOpenAI`` 实例。

    所有内置供应商都提供 OpenAI 兼容端点，所以这里用 LangChain 官方的
    ``langchain-openai`` 集成 + 自定义 ``base_url`` 覆盖全部情形，
    而不是为每家写一套适配代码。
    """
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key or "not-needed",
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# LangChain 实现
# --------------------------------------------------------------------------- #
@dataclass
class LangChainLLMClient:
    """基于 LangChain ``BaseChatModel`` 的客户端。

    属性 ``model`` / ``vision_model`` 保持为**模型名字符串**（而不是模型对象），
    这是 ``core.extract`` 与 ``core.explain`` 依赖的契约，用于在结果里记录
    "这次是哪个模型干的"。真正的 LangChain 对象放在 ``chat_model``。
    """

    chat_model: BaseChatModel
    model: str = "unknown"
    vision_model: str | None = None
    temperature: float = 0.2
    timeout: float = 120.0
    max_retries: int = 2
    preset_key: str | None = None
    _cache: dict[str, BaseChatModel] = field(default_factory=dict, repr=False)

    # ---------------- 构造 ---------------- #
    @classmethod
    def from_preset(
        cls,
        key: str,
        api_key: str | None = None,
        model: str | None = None,
        **kw: Any,
    ) -> LangChainLLMClient:
        preset = PRESETS.get(key)
        if preset is None:
            raise LLMError(f"未知识别供应商：{key}。可用：{', '.join(PRESETS)}")
        key_val = api_key or os.environ.get(preset.api_key_env) or ""
        model_name = model or preset.text_model
        temperature = kw.pop("temperature", 0.2)
        timeout = kw.pop("timeout", 120.0)
        max_retries = kw.pop("max_retries", 2)
        chat_model = build_chat_model(
            base_url=kw.pop("base_url", preset.base_url),
            model=model_name,
            api_key=key_val or None,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            **kw,
        )
        return cls(
            chat_model=chat_model,
            model=model_name,
            vision_model=preset.vision_model or None,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            preset_key=key,
        )

    @classmethod
    def from_env(cls, **kw: Any) -> LangChainLLMClient:
        """按环境变量自动探测可用的供应商，优先视觉能力完整的。"""
        order = os.environ.get("LABLENS_PROVIDER")
        keys = [order] if order else ["dashscope", "zhipu", "siliconflow", "openai", "deepseek"]
        for k in keys:
            if not k or k not in PRESETS:
                continue
            if os.environ.get(PRESETS[k].api_key_env):
                return cls.from_preset(k, **kw)
        if os.environ.get("LABLENS_BASE_URL"):
            model_name = os.environ.get("LABLENS_MODEL", "gpt-4o-mini")
            temperature = kw.pop("temperature", 0.2)
            chat_model = build_chat_model(
                base_url=os.environ["LABLENS_BASE_URL"],
                model=model_name,
                api_key=os.environ.get("LABLENS_API_KEY"),
                temperature=temperature,
                **kw,
            )
            return cls(
                chat_model=chat_model,
                model=model_name,
                vision_model=os.environ.get("LABLENS_VISION_MODEL") or None,
                temperature=temperature,
            )
        raise LLMError(
            "未检测到任何可用的模型凭证。请设置 DASHSCOPE_API_KEY / ZHIPU_API_KEY 等环境变量，"
            "或设置 LABLENS_BASE_URL 指向本地推理服务。"
        )

    @property
    def supports_vision(self) -> bool:
        return bool(self.vision_model)

    def _model_for(self, model: str | None) -> BaseChatModel:
        """按需切换模型名；同名复用，不同名构建并缓存。"""
        if not model or model == self.model:
            return self.chat_model
        if model not in self._cache:
            if not self.preset_key:
                raise LLMError(f"该客户端未绑定供应商预设，无法切换到模型 {model}。")
            preset = PRESETS[self.preset_key]
            self._cache[model] = build_chat_model(
                base_url=preset.base_url,
                model=model,
                api_key=getattr(self.chat_model, "openai_api_key", None),
                temperature=self.temperature,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._cache[model]

    # ---------------- 调用 ---------------- #
    def chat(
        self,
        messages: Sequence[dict],
        model: str | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        chat_model = self._model_for(model)
        if temperature is not None:
            chat_model = chat_model.bind(temperature=temperature)
        if json_mode:
            chat_model = chat_model.bind(response_format={"type": "json_object"})
        try:
            result = chat_model.invoke(_to_lc_messages(messages))
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - 网络/鉴权/限流统一转成 LLMError
            raise LLMError(f"调用模型失败：{exc}") from exc
        return _message_text(result)

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
# 离线降级客户端
# --------------------------------------------------------------------------- #
@dataclass
class FakeLLMClient:
    """离线客户端：按顺序或按提示词关键字返回预置结果。

    用途：
    - CI 与单测中验证整条流水线，不依赖网络与额度
    - 没有 API Key 时演示完整界面

    底层用的是 LangChain 自带的假模型（``FakeListChatModel``），
    所以它和真实客户端走完全相同的消息转换与解析路径。
    """

    extraction: dict | None = None
    responses: list[str] = field(default_factory=list)
    model: str = "fake"
    vision_model: str | None = "fake-vision"

    def _fake_model(self, model: str | None) -> FakeListChatModel:
        pool = self.responses or [json.dumps(self.extraction or {"items": []}, ensure_ascii=False)]
        return FakeListChatModel(responses=pool)

    def chat(
        self,
        messages: Sequence[dict],
        model: str | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        if json_mode and self.extraction is not None:
            return json.dumps(self.extraction, ensure_ascii=False)
        return _message_text(self._fake_model(model).invoke(_to_lc_messages(messages)))

    def chat_json(
        self,
        messages: Sequence[dict],
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict:
        return dict(self.extraction or {"items": []})
