"""跨源池化聚合决策 - 单一来源的 ricu 一致性策略。

R ricu 的 ``change_interval`` 在把同一 (patient, hour) 桶内的数据聚合成
median/max 时，是把**所有源 / 所有 itemid 的原始值合并后做一次聚合**
(pooled aggregation)。

EasyICU 为性能默认让每个源在 DuckDB 里各自预聚合（per-source MEDIAN），
再 concat、由 ``change_interval`` 做二次聚合。当一个概念有 2+ 个数值源时，
这会得到 **median-of-medians**（≠ ricu 的 pooled median）。

历史上这个判断散落在 ``concept.py`` 里三段内联代码
(``_block_duckdb_value_transform`` / ``_block_duckdb_same_table`` /
``_block_duckdb_multi_numeric``)，任何新增多源概念都可能漏判而回归。
本模块把它收敛为**单一、可单测**的策略函数 ``compute_pooling_decision``，
``concept.py`` 只读结果、不再内联重算。

语义与原内联实现逐字等价（见 ``tests/test_ricu_alignment.py``）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

__all__ = ["PoolingDecision", "compute_pooling_decision", "should_pool_raw"]


# 被视为 "value_transform" 的内联回调形式：这些回调若在多个源上各自
# 内联到 DuckDB 聚合中，会先变换再 per-source median，破坏池化语义。
_PERCENT_AS_NUMERIC = "transform_fun(percent_as_numeric)"
_CONVERT_UNIT_PREFIX = "convert_unit("
_VALUE_TRANSFORM_MARKERS = ("set_val(", "fahr_to_cels")


def _is_value_transform_callback(callback: object) -> bool:
    """源回调是否属于 "值变换" 类（percent_as_numeric / convert_unit+set_val/fahr）。"""
    if not isinstance(callback, str):
        return False
    cb = callback.strip()
    if cb == _PERCENT_AS_NUMERIC:
        return True
    if cb.startswith(_CONVERT_UNIT_PREFIX) and any(m in cb for m in _VALUE_TRANSFORM_MARKERS):
        return True
    return False


@dataclass(frozen=True)
class PoolingDecision:
    """单个概念的 DuckDB 预聚合门控决策。

    三个字段与 ``concept.py`` 历史上的三个 ``_block_duckdb_*`` 标志一一对应，
    语义完全一致，仅集中到一处便于审计与测试。
    """

    #: 2+ 个源使用值变换回调 → 禁止把变换内联进 DuckDB（回退 Python 回调路径）。
    block_value_transform: bool
    #: 2+ 个源指向同一张表 → 禁止 DuckDB 预聚合，交给 change_interval 池化。
    block_same_table: bool
    #: ≥2 个普通数值源（含简单字符串回调，排除 rgx_itm）→ 禁止 DuckDB 预聚合。
    block_multi_numeric: bool

    @property
    def force_raw_pool(self) -> bool:
        """是否必须禁用 DuckDB 预聚合、让 change_interval 做单次池化聚合。

        对应 ``concept.py`` 中
        ``if _block_duckdb_same_table or _block_duckdb_multi_numeric:``。
        """
        return self.block_same_table or self.block_multi_numeric


def compute_pooling_decision(sources: Iterable[object]) -> PoolingDecision:
    """根据一个概念的源列表计算池化决策。

    源对象需暴露（duck-typed）以下属性，缺失时按 ``None`` 处理：
    - ``callback``: 源回调（``None`` / 字符串表达式 / 可调用对象）
    - ``table``: 源所在原始表名
    - ``class_name``: 源类别（如 ``"num_itm"`` / ``"rgx_itm"`` / ``"fun_itm"``）
    """
    sources = list(sources)

    n_value_transform = 0
    table_counts: dict = {}
    n_plain_numeric = 0

    for src in sources:
        callback = getattr(src, "callback", None)
        table = getattr(src, "table", None)
        class_name = getattr(src, "class_name", None)

        if _is_value_transform_callback(callback):
            n_value_transform += 1

        if table:
            table_counts[table] = table_counts.get(table, 0) + 1

        # 普通数值源：无回调或简单字符串回调，且非 rgx_itm（正则常量源较稀疏）。
        if class_name != "rgx_itm" and (callback is None or isinstance(callback, str)):
            n_plain_numeric += 1

    return PoolingDecision(
        block_value_transform=n_value_transform > 1,
        block_same_table=any(cnt > 1 for cnt in table_counts.values()),
        block_multi_numeric=n_plain_numeric >= 2,
    )


def should_pool_raw(sources: Iterable[object]) -> bool:
    """便捷封装：是否需要强制原始值池化（禁用 DuckDB 预聚合）。"""
    return compute_pooling_decision(sources).force_raw_pool
