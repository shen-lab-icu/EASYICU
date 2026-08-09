"""时间单位归一 - AUMC/eICU 偏移量（分钟）→ 小时的单一转换点。

EasyICU 内部统一以**小时**作为相对时间单位（与 R ricu 默认
``interval = hours(1L)`` 对齐）。但若干数据源的原始 / DuckDB 输出是分钟：

- **AUMC**: 原始 ``measuredat`` 是 Unix 毫秒；DuckDB 聚合路径输出
  ``measuredat_minutes``（分钟），datasource 层也把 ms→分钟。
- **eICU**: ``*offset`` 列本身就是相对入院的分钟数。

历史问题（datasource.py 旧 NOTE "这里输出的是分钟，与ricu(小时)不匹配"）：
分钟→小时的 ``/ 60.0`` 转换散落在 ``concept.py`` 至少两处——
批量多概念路径（_pre_aggregated 表）一处、``_align_time_to_admission`` 一处——
两条路径必须手工保持同步，漏改即产生时间单位不一致。

本模块把转换因子与转换函数收敛到一处，供所有路径调用，便于单测与审计。
契约：**每个时间列只能被转换一次**。批量路径与 ``_align_time_to_admission``
是互斥路径（前者产物标记 ``_pre_aggregated`` 后不再经过后者的分钟→小时转换），
不得对同一列重复调用，否则会得到 /3600 的二次缩放。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = [
    "ICU_TIME_FALLBACK_LIMIT_HOURS",
    "ICU_TIME_POST_DISCHARGE_HOURS",
    "ICU_TIME_PRE_ADMISSION_HOURS",
    "MINUTES_PER_HOUR",
    "minutes_to_hours",
    "minutes_to_hours_series",
]

#: 单一来源的转换因子。任何分钟→小时换算都应使用它，禁止再写裸 ``/ 60.0``。
MINUTES_PER_HOUR: float = 60.0
#: Longitudinal ICU exports retain one day of pre-admission context.
ICU_TIME_PRE_ADMISSION_HOURS: float = 24.0
#: Longitudinal ICU exports retain one day after recorded ICU discharge.
ICU_TIME_POST_DISCHARGE_HOURS: float = 24.0
#: Fail-safe bound for sources/stays without a usable ICU length of stay.
ICU_TIME_FALLBACK_LIMIT_HOURS: float = 366.0 * 24.0


def minutes_to_hours(value: float) -> float:
    """标量分钟→小时。"""
    return value / MINUTES_PER_HOUR


def minutes_to_hours_series(series: "pd.Series") -> "pd.Series":
    """pandas.Series 分钟→小时（保持 dtype，逐元素除以 60）。

    仅用于已确认为分钟单位的相对/绝对时间列（如 AUMC ``measuredat_minutes``、
    eICU ``*offset``）。调用方负责保证该列尚未被转换（见模块级契约）。
    """
    return series / MINUTES_PER_HOUR
