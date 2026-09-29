"""单元测试：LangChain LLM 传输层。

这里**不碰网络**：用 LangChain 自带的假模型（``FakeListChatModel``）把几条
"最容易悄悄写错"的路径覆盖掉——消息转换、多模态内容块透传、JSON 解析与修复重试。

改造成 LangChain 之后最需要守住的性质是：对外那个极窄的 ``LLMClient`` 协议
（``chat`` / ``chat_json``）行为不变，因为 ``core.extract`` 与 ``core.explain``
只认它。
"""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from core.providers import (
    PRESETS,
    FakeLLMClient,
    LangChainLLMClient,
    LLMError,
    _to_lc_messages,
    build_chat_model,
    encode_image_data_url,
    extract_json,
)

ALL_KEY_ENVS = [p.api_key_env for p in PRESETS.values()] + [
    "DASHSCOPE_API_KEY",
    "LABLENS_PROVIDER",
    "LABLENS_BASE_URL",
    "LABLENS_API_KEY",
    "LABLENS_MODEL",
    "LABLENS_VISION_MODEL",
]


@pytest.fixture()
def clean_env(monkeypatch):
    """清掉所有模型相关环境变量，保证用例不受开发机配置影响。"""
    for name in ALL_KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _client(*responses: str) -> LangChainLLMClient:
    return LangChainLLMClient(
        chat_model=FakeListChatModel(responses=list(responses)),
        model="fake-model",
        vision_model="fake-vision",
    )


# --------------------------------------------------------------------------- #
# 模型构建
# --------------------------------------------------------------------------- #
def test_build_chat_model_returns_langchain_model() -> None:
    m = build_chat_model(base_url="https://example.invalid/v1", model="some-model", api_key="k")
    assert isinstance(m, BaseChatModel)
    assert m.model_name == "some-model"
    assert m.openai_api_base == "https://example.invalid/v1"


def test_from_preset_maps_endpoint_and_models() -> None:
    c = LangChainLLMClient.from_preset("dashscope", api_key="sk-x", model="qwen-plus")
    assert isinstance(c.chat_model, BaseChatModel)
    assert c.model == "qwen-plus"
    assert c.vision_model == PRESETS["dashscope"].vision_model
    assert c.supports_vision is True
    assert c.chat_model.openai_api_base == PRESETS["dashscope"].base_url


def test_from_preset_defaults_to_preset_text_model() -> None:
    c = LangChainLLMClient.from_preset("zhipu", api_key="sk-x")
    assert c.model == PRESETS["zhipu"].text_model


def test_deepseek_preset_has_no_vision() -> None:
    c = LangChainLLMClient.from_preset("deepseek", api_key="sk-x")
    assert c.supports_vision is False
    assert c.vision_model is None


def test_from_preset_unknown_raises() -> None:
    with pytest.raises(LLMError, match="未知识别供应商"):
        LangChainLLMClient.from_preset("nope", api_key="sk-x")


def test_model_override_builds_and_caches(clean_env) -> None:
    c = LangChainLLMClient.from_preset("dashscope", api_key="sk-x", model="qwen-plus")
    other = c._model_for("qwen-max")
    assert other is not c.chat_model
    assert other.model_name == "qwen-max"
    # 同一个模型名第二次直接复用缓存，不再重建
    assert c._model_for("qwen-max") is other
    # 传 None 或同名时返回原始模型
    assert c._model_for(None) is c.chat_model
    assert c._model_for("qwen-plus") is c.chat_model


def test_model_override_without_preset_raises() -> None:
    c = LangChainLLMClient(chat_model=FakeListChatModel(responses=["x"]), model="fake")
    with pytest.raises(LLMError, match="未绑定供应商预设"):
        c._model_for("another")


# --------------------------------------------------------------------------- #
# 环境探测
# --------------------------------------------------------------------------- #
def test_from_env_picks_first_available_provider(clean_env) -> None:
    clean_env.setenv("ZHIPU_API_KEY", "sk-zhipu")
    c = LangChainLLMClient.from_env()
    assert c.preset_key == "zhipu"
    assert c.model == PRESETS["zhipu"].text_model


def test_from_env_respects_explicit_provider(clean_env) -> None:
    clean_env.setenv("ZHIPU_API_KEY", "sk-zhipu")
    clean_env.setenv("SILICONFLOW_API_KEY", "sk-sf")
    clean_env.setenv("LABLENS_PROVIDER", "siliconflow")
    c = LangChainLLMClient.from_env()
    assert c.preset_key == "siliconflow"


def test_from_env_falls_back_to_custom_base_url(clean_env) -> None:
    clean_env.setenv("LABLENS_BASE_URL", "http://localhost:8000/v1")
    clean_env.setenv("LABLENS_MODEL", "Qwen/Qwen3-8B-Instruct")
    c = LangChainLLMClient.from_env()
    assert c.preset_key is None
    assert c.model == "Qwen/Qwen3-8B-Instruct"
    assert c.chat_model.openai_api_base == "http://localhost:8000/v1"


def test_from_env_without_credentials_raises(clean_env) -> None:
    with pytest.raises(LLMError, match="未检测到任何可用的模型凭证"):
        LangChainLLMClient.from_env()


# --------------------------------------------------------------------------- #
# 消息转换
# --------------------------------------------------------------------------- #
def test_to_lc_messages_maps_roles() -> None:
    msgs = _to_lc_messages([
        {"role": "system", "content": "系统"},
        {"role": "user", "content": "用户"},
        {"role": "assistant", "content": "助手"},
    ])
    assert [type(m).__name__ for m in msgs] == ["SystemMessage", "HumanMessage", "AIMessage"]
    assert [m.content for m in msgs] == ["系统", "用户", "助手"]


def test_to_lc_messages_preserves_multimodal_blocks() -> None:
    """多模态 content 块必须原样透传，否则视觉抽取会静默退化成纯文本。"""
    blocks = [
        {"type": "text", "text": "看图"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    msgs = _to_lc_messages([{"role": "user", "content": blocks}])
    assert msgs[0].content == blocks


def test_to_lc_messages_unknown_role_defaults_to_user() -> None:
    msgs = _to_lc_messages([{"role": "tool", "content": "x"}])
    assert type(msgs[0]).__name__ == "HumanMessage"


# --------------------------------------------------------------------------- #
# 调用
# --------------------------------------------------------------------------- #
def test_chat_returns_text() -> None:
    assert _client("你好").chat([{"role": "user", "content": "hi"}]) == "你好"


def test_chat_json_parses_valid_output() -> None:
    c = _client('{"items": [1, 2]}')
    assert c.chat_json([{"role": "user", "content": "hi"}]) == {"items": [1, 2]}


def test_chat_json_repairs_invalid_output() -> None:
    """模型第一次没吐 JSON 时，应当追加一次修复请求而不是直接失败。"""
    c = _client("我不太确定该怎么回答", '{"items": []}')
    assert c.chat_json([{"role": "user", "content": "hi"}]) == {"items": []}


def test_chat_json_raises_when_repair_also_fails() -> None:
    c = _client("还是不是 JSON", "依然不是 JSON")
    with pytest.raises(LLMError, match="无法从模型输出中解析出 JSON"):
        c.chat_json([{"role": "user", "content": "hi"}])


def test_chat_wraps_backend_errors_as_llm_error() -> None:
    class Boom(FakeListChatModel):
        def _call(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("网络炸了")

    c = LangChainLLMClient(chat_model=Boom(responses=["x"]), model="boom")
    with pytest.raises(LLMError, match="调用模型失败"):
        c.chat([{"role": "user", "content": "hi"}])


# --------------------------------------------------------------------------- #
# JSON 容错解析
# --------------------------------------------------------------------------- #
def test_extract_json_plain() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_strips_markdown_fence() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_finds_embedded_object() -> None:
    assert extract_json('好的，结果如下：{"a": {"b": 2}} 以上。') == {"a": {"b": 2}}


def test_extract_json_handles_braces_inside_strings() -> None:
    assert extract_json('{"a": "含 { 花括号 } 的文本"}') == {"a": "含 { 花括号 } 的文本"}


def test_extract_json_rejects_non_object() -> None:
    with pytest.raises(LLMError):
        extract_json("[1, 2, 3]")


def test_extract_json_rejects_empty() -> None:
    with pytest.raises(LLMError, match="模型返回为空"):
        extract_json("")


# --------------------------------------------------------------------------- #
# 图片编码
# --------------------------------------------------------------------------- #
def test_encode_image_data_url_from_bytes() -> None:
    url = encode_image_data_url(b"\x89PNG", mime="image/png")
    assert url.startswith("data:image/png;base64,")


def test_encode_image_data_url_from_file(local_tmp) -> None:
    p = local_tmp / "shot.png"
    p.write_bytes(b"\x89PNG\r\n")
    url = encode_image_data_url(p)
    assert url.startswith("data:image/png;base64,")


# --------------------------------------------------------------------------- #
# 离线客户端
# --------------------------------------------------------------------------- #
def test_fake_llm_client_satisfies_protocol() -> None:
    c = FakeLLMClient(extraction={"items": [{"raw_name": "白细胞"}]})
    assert c.chat_json([{"role": "user", "content": "hi"}]) == {"items": [{"raw_name": "白细胞"}]}
    assert c.model == "fake"


def test_fake_llm_client_returns_text_via_langchain_fake_model() -> None:
    c = FakeLLMClient(responses=["纯文本回复"])
    assert c.chat([{"role": "user", "content": "hi"}]) == "纯文本回复"
