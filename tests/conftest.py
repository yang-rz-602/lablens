"""pytest 公共 fixture。"""

from __future__ import annotations

from pathlib import Path

import pytest


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
