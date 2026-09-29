"""pytest 公共 fixture。"""

from __future__ import annotations

from pathlib import Path

import pytest

#: 会改变运行时行为、因而必须从测试环境里清掉的所有环境变量。
#:
#: 为什么要在测试里统一清空：这些变量一旦存在，`build_retriever()` 就会去调用
#: DashScope 的 embedding 接口给 574 条语料建索引（约 58 次请求）。
#: 开发机上恰好配了 Key，测试就会变成**慢、依赖网络、还真花钱**——
#: 而且失败原因会伪装成无关的超时。测试必须是密闭的。
_AMBIENT_ENV_VARS = (
    "DASHSCOPE_API_KEY",
    "ZHIPU_API_KEY",
    "SILICONFLOW_API_KEY",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "LOCAL_LLM_API_KEY",
    "LABLENS_PROVIDER",
    "LABLENS_BASE_URL",
    "LABLENS_API_KEY",
    "LABLENS_MODEL",
    "LABLENS_VISION_MODEL",
    "LABLENS_EMBED_MODEL",
    "LABLENS_RERANK_MODEL",
)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """让每个用例都从"没有任何凭证"的状态开始。

    需要凭证的用例自己用 ``monkeypatch.setenv`` 显式加上；
    这样"离线降级路径"是被真正测到的那条，而不是碰巧被开发机的 Key 掩盖掉。
    """
    for name in _AMBIENT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def local_tmp() -> Path:
    """项目内的临时目录。

    pytest 默认的 ``tmp_path`` 落在系统临时目录，在受限沙箱或只读环境里会被拒绝，
    导致与被测逻辑无关的测试失败。改用项目内目录，让测试在任何环境下都能跑。
    """
    d = Path(__file__).resolve().parent / ".tmp"
    d.mkdir(parents=True, exist_ok=True)
    for f in d.iterdir():
        if f.is_file():
            f.unlink()
    return d
