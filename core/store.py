"""纵向趋势存储 —— **默认关闭**。

为什么默认关闭
--------------
检验报告属于 PIPL 第 28 条的敏感个人信息；GB/T 39725-2020 把真实检验报告划为
第 4 级数据，并要求"不宜将健康医疗数据在境外的服务器中存储"。对一个公开部署的
demo 来说，**最省事也最正确的合规设计就是不存**。

所以这里的策略是：

- ``enabled=False``（默认）：所有写入是 no-op，进程退出即忘
- ``enabled=True``：只在**用户本机**写一个本地文件，且只存数值与日期，
  **不存姓名、不存医院、不存报告原文、不存 source_text**
- 始终提供 ``delete_all()``，一键清空

这和"有趋势功能"并不冲突：用户在自己电脑上追踪自己的指标，
数据控制者就是用户本人；而把别人的报告收进服务器数据库才是风险所在。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .schema import Flag, LabItem, LabReport

__all__ = ["TrendPoint", "TrendStore"]

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS measurements (
    report_id     VARCHAR NOT NULL,
    collected_at  DATE,
    canonical_name VARCHAR NOT NULL,
    label         VARCHAR,
    value         DOUBLE,
    unit          VARCHAR,
    flag          VARCHAR,
    PRIMARY KEY (report_id, canonical_name)
);
"""


@dataclass
class TrendPoint:
    collected_at: date | None
    value: float
    unit: str | None
    flag: str
    report_id: str


class TrendStore:
    """基于 DuckDB 的本地趋势存储。

    Parameters
    ----------
    path:
        DuckDB 文件路径。``:memory:`` 表示仅进程内。
    enabled:
        是否真正落盘。``False`` 时所有写操作静默跳过、读操作返回空。
    """

    def __init__(self, path: str | Path = "lablens_trend.duckdb", enabled: bool = False) -> None:
        self.path = str(path)
        self.enabled = enabled
        self._con: Any = None
        if self.enabled:
            self._connect()

    # ------------------------------------------------------------------ #
    def _connect(self) -> None:
        try:
            import duckdb  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "启用趋势存储需要 duckdb：`pip install duckdb`。"
                "或者保持 enabled=False（默认），此时不会写入任何数据。"
            ) from exc
        self._con = duckdb.connect(self.path)
        self._con.execute(_SCHEMA)

    @property
    def available(self) -> bool:
        return self._con is not None

    # ------------------------------------------------------------------ #
    def save(self, report: LabReport) -> int:
        """保存一份报告的**数值型**指标。返回写入行数（未启用时为 0）。"""
        if not self.enabled or self._con is None:
            return 0

        rows = [
            (
                report.report_id,
                report.collected_at,
                item.canonical_name,
                item.raw_name,
                item.value,
                item.unit,
                item.flag.value,
            )
            for item in report.items
            if item.canonical_name and item.value is not None
        ]
        if not rows:
            return 0

        self._con.executemany(
            "INSERT OR REPLACE INTO measurements "
            "(report_id, collected_at, canonical_name, label, value, unit, flag) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        log.info("已保存 %d 条指标到本地趋势库（仅数值，无身份信息）", len(rows))
        return len(rows)

    # ------------------------------------------------------------------ #
    def series(self, canonical_name: str) -> list[TrendPoint]:
        """取某一指标的历史序列，按日期升序。"""
        if not self.enabled or self._con is None:
            return []
        try:
            rows = self._con.execute(
                "SELECT collected_at, value, unit, flag, report_id FROM measurements "
                "WHERE canonical_name = ? AND value IS NOT NULL "
                "ORDER BY collected_at NULLS LAST, report_id",
                [canonical_name],
            ).fetchall()
        except Exception:  # noqa: BLE001
            return []
        return [TrendPoint(r[0], r[1], r[2], r[3] or "?", r[4]) for r in rows]

    def tracked_indicators(self) -> list[tuple[str, str, int]]:
        """返回 (canonical_name, 最近一次的中文名, 记录次数)，按记录次数降序。"""
        if not self.enabled or self._con is None:
            return []
        try:
            rows = self._con.execute(
                "SELECT canonical_name, any_value(label), count(*) AS n FROM measurements "
                "GROUP BY canonical_name ORDER BY n DESC, canonical_name"
            ).fetchall()
        except Exception:  # noqa: BLE001
            return []
        return [(r[0], r[1] or r[0], int(r[2])) for r in rows]

    def report_count(self) -> int:
        if not self.enabled or self._con is None:
            return 0
        try:
            return int(self._con.execute("SELECT count(DISTINCT report_id) FROM measurements").fetchone()[0])
        except Exception:  # noqa: BLE001
            return 0

    # ------------------------------------------------------------------ #
    def delete_all(self) -> None:
        """清空本地趋势数据。"""
        if self._con is not None:
            self._con.execute("DELETE FROM measurements")
        log.info("本地趋势数据已清空。")

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    # ------------------------------------------------------------------ #
    def as_rows(self, canonical_name: str) -> list[dict]:
        """把序列转成可直接喂给表格/绘图组件的行列表。"""
        return [
            {
                "日期": p.collected_at,
                "结果": p.value,
                "单位": p.unit,
                "判定": Flag(p.flag).label_zh if p.flag in {f.value for f in Flag} else p.flag,
            }
            for p in self.series(canonical_name)
        ]


def flag_is_abnormal(item: LabItem) -> bool:
    return item.flag not in {Flag.NORMAL, Flag.UNKNOWN}
