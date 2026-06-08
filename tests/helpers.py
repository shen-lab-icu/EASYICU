"""ricu 对照测试的共享工具 (pytest fixtures / 断言)。

替代历史上的独立脚本 ``feature_compare.py`` —— 参考 ricu 的设计哲学，
把验证逻辑放进 pytest 而非独立 CLI。本模块提供两类能力：

1. **纯逻辑工具**（不依赖真实数据库）：``pooled_median`` /
   ``median_of_medians`` 用于演示并锁定"池化 median ≠ median-of-medians"
   这一核心不一致；``FakeSource`` 用于驱动 ``pooling.compute_pooling_decision``。
2. **ricu CSV 黄金基准加载**：``load_ricu_csv`` /
   ``require_ricu_fixtures``，在 ``tests/fixtures/ricu/`` 缺失时让用例
   ``pytest.skip``，从而 CI 在没有 ricu 导出时仍能跑纯逻辑部分。
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pytest

TESTS_DIR = Path(__file__).resolve().parent
RICU_FIXTURE_DIR = TESTS_DIR / "fixtures" / "ricu"


# --------------------------------------------------------------------------- #
# 纯逻辑工具
# --------------------------------------------------------------------------- #
class FakeSource:
    """概念源的最小 duck-typed 替身，驱动 pooling.compute_pooling_decision。

    只暴露 ``callback`` / ``table`` / ``class_name`` 三个属性。
    """

    def __init__(self, table=None, callback=None, class_name="num_itm"):
        self.table = table
        self.callback = callback
        self.class_name = class_name


def pooled_median(groups: Sequence[Iterable[float]]) -> float:
    """R ricu 行为：把所有源/组的原始值合并后取一次 median。"""
    pooled: List[float] = []
    for g in groups:
        pooled.extend(float(x) for x in g)
    return statistics.median(pooled)


def median_of_medians(groups: Sequence[Iterable[float]]) -> float:
    """EasyICU 旧默认行为：每组先 median，再对各组 median 取均值。

    用于在测试里显式锁定它与 ``pooled_median`` 的差异。
    """
    per_group = [statistics.median([float(x) for x in g]) for g in groups]
    return sum(per_group) / len(per_group)


# --------------------------------------------------------------------------- #
# ricu CSV 黄金基准
# --------------------------------------------------------------------------- #
def ricu_fixtures_available() -> bool:
    return RICU_FIXTURE_DIR.is_dir() and any(RICU_FIXTURE_DIR.glob("*.csv"))


def require_ricu_fixtures() -> None:
    """无 ricu CSV 黄金基准时跳过当前用例。"""
    if not ricu_fixtures_available():
        pytest.skip(
            f"ricu 黄金基准缺失: {RICU_FIXTURE_DIR} 无 *.csv。"
            "放入 ricu 导出的 CSV 后该用例自动启用。"
        )


def load_ricu_csv(name: str):
    """加载一个 ricu 导出的 CSV 黄金基准 (返回 pandas.DataFrame)。

    ``name`` 可带或不带 ``.csv`` 后缀。文件须位于 ``tests/fixtures/ricu/``。
    """
    import pandas as pd

    require_ricu_fixtures()
    fname = name if name.endswith(".csv") else f"{name}.csv"
    path = RICU_FIXTURE_DIR / fname
    if not path.exists():
        pytest.skip(f"ricu 黄金基准文件不存在: {path}")
    return pd.read_csv(path)


def assert_series_close(actual, expected, *, rtol: float = 1e-6, atol: float = 1e-6,
                        name: str = "value") -> None:
    """逐元素近似相等断言 (供 ricu 对照数值列使用)。"""
    import numpy as np

    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    assert a.shape == e.shape, f"{name}: 形状不一致 {a.shape} vs {e.shape}"
    both_nan = np.isnan(a) & np.isnan(e)
    close = np.isclose(a, e, rtol=rtol, atol=atol) | both_nan
    if not close.all():
        bad = np.where(~close)[0][:10]
        raise AssertionError(
            f"{name}: {len(bad)}+ 个元素与 ricu 不一致 (前若干 idx={list(bad)}): "
            f"actual={a[bad]}, expected={e[bad]}"
        )
