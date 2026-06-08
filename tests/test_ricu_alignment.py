"""与 R ricu 的一致性回归测试。

历史上 EasyICU 的"与 ricu 一致"只存在于代码注释里，没有任何自动化校验
(feature_compare.py 头部声称已迁移到本文件，但文件长期缺失)。本文件补上这层
地基，覆盖两个已知会改变数值结果的核心不一致：

1. **跨源池化 median**：ricu 把所有源原始值合并后取一次 median；EasyICU 默认
   每源预聚合会得到 median-of-medians。验证决策函数在多源时强制原始池化。
2. **AUMC 时间单位**：内部统一以小时为单位，分钟→小时的换算必须经单一来源。

纯逻辑用例总是运行；依赖 ricu CSV 黄金基准的用例在 tests/fixtures/ricu/ 缺失
时自动跳过 (见 helpers.require_ricu_fixtures)。
"""

from __future__ import annotations

import pytest

from helpers import (  # type: ignore  # tests dir on sys.path via conftest
    FakeSource,
    load_ricu_csv,
    median_of_medians,
    pooled_median,
    require_ricu_fixtures,
)

from easyicu.pooling import (
    PoolingDecision,
    compute_pooling_decision,
    should_pool_raw,
)
from easyicu.time_units import (
    MINUTES_PER_HOUR,
    minutes_to_hours,
    minutes_to_hours_series,
)

pytestmark = pytest.mark.ricu_parity


# --------------------------------------------------------------------------- #
# 1. 池化决策 (修复2: 替代散落的 _block_duckdb_* 内联逻辑)
# --------------------------------------------------------------------------- #
class TestPoolingDecision:
    def test_single_source_does_not_block(self):
        d = compute_pooling_decision([FakeSource(table="chartevents")])
        assert d == PoolingDecision(False, False, False)
        assert d.force_raw_pool is False
        assert should_pool_raw([FakeSource(table="chartevents")]) is False

    def test_two_sources_same_table_blocks(self):
        # AUMC o2sat: 同一 numericitems 表的两个 itemid 源
        sources = [
            FakeSource(table="numericitems"),
            FakeSource(table="numericitems"),
        ]
        d = compute_pooling_decision(sources)
        assert d.block_same_table is True
        assert d.force_raw_pool is True
        assert should_pool_raw(sources) is True

    def test_two_plain_numeric_different_tables_blocks(self):
        # MIMIC o2sat: chartevents + labevents 两张不同表
        sources = [
            FakeSource(table="chartevents"),
            FakeSource(table="labevents"),
        ]
        d = compute_pooling_decision(sources)
        assert d.block_same_table is False
        assert d.block_multi_numeric is True
        assert d.force_raw_pool is True

    def test_two_value_transform_sources_block_inline_transform(self):
        sources = [
            FakeSource(table="t1", callback="transform_fun(percent_as_numeric)"),
            FakeSource(table="t2", callback="convert_unit(fahr_to_cels)"),
        ]
        d = compute_pooling_decision(sources)
        assert d.block_value_transform is True

    def test_convert_unit_without_marker_is_not_value_transform(self):
        # convert_unit 但不含 set_val/fahr_to_cels → 不计为 value_transform
        sources = [
            FakeSource(table="t1", callback="convert_unit(2.0)"),
            FakeSource(table="t2", callback="convert_unit(3.0)"),
        ]
        d = compute_pooling_decision(sources)
        assert d.block_value_transform is False

    def test_rgx_itm_sources_excluded_from_plain_numeric(self):
        # rgx_itm 源不计入 plain numeric 计数
        sources = [
            FakeSource(table="t1", class_name="rgx_itm"),
            FakeSource(table="t2", class_name="rgx_itm"),
        ]
        d = compute_pooling_decision(sources)
        assert d.block_multi_numeric is False

    def test_callable_callback_excluded_from_plain_numeric(self):
        # 可调用对象回调（非 None、非 str）不计入 plain numeric
        sources = [
            FakeSource(table="t1", callback=lambda x: x),
            FakeSource(table="t2", callback=lambda x: x),
        ]
        d = compute_pooling_decision(sources)
        assert d.block_multi_numeric is False

    def test_decision_matches_legacy_inline_logic(self):
        """逐字复现旧内联逻辑，证明重构行为等价（防回归）。"""
        import random

        rng = random.Random(20260608)
        tables = [None, "chartevents", "labevents", "numericitems"]
        callbacks = [
            None,
            "transform_fun(percent_as_numeric)",
            "convert_unit(fahr_to_cels)",
            "convert_unit(set_val(0))",
            "convert_unit(2.0)",
            lambda x: x,
        ]
        classes = ["num_itm", "rgx_itm", "fun_itm"]

        def legacy(sources):
            n_vt = 0
            for s in sources:
                cb = s.callback
                if isinstance(cb, str):
                    c = cb.strip()
                    if c == "transform_fun(percent_as_numeric)" or (
                        c.startswith("convert_unit(")
                        and ("set_val(" in c or "fahr_to_cels" in c)
                    ):
                        n_vt += 1
            block_vt = n_vt > 1
            counts = {}
            for s in sources:
                if s.table:
                    counts[s.table] = counts.get(s.table, 0) + 1
            block_st = any(c > 1 for c in counts.values())
            n_plain = 0
            for s in sources:
                if s.class_name == "rgx_itm":
                    continue
                if s.callback is None or isinstance(s.callback, str):
                    n_plain += 1
            block_mn = n_plain >= 2
            return (block_vt, block_st, block_mn)

        for _ in range(500):
            n = rng.randint(1, 4)
            sources = [
                FakeSource(
                    table=rng.choice(tables),
                    callback=rng.choice(callbacks),
                    class_name=rng.choice(classes),
                )
                for _ in range(n)
            ]
            d = compute_pooling_decision(sources)
            assert (
                d.block_value_transform,
                d.block_same_table,
                d.block_multi_numeric,
            ) == legacy(sources)


# --------------------------------------------------------------------------- #
# 2. 池化 median 语义 (锁定核心不一致)
# --------------------------------------------------------------------------- #
class TestPooledMedianSemantics:
    def test_pooled_median_differs_from_median_of_medians(self):
        # 两个源同一时间桶：源A=[1,2,3], 源B=[100]
        # pooled median([1,2,3,100]) = 2.5
        # median-of-medians: median([1,2,3])=2, median([100])=100 → mean=51
        groups = [[1, 2, 3], [100]]
        assert pooled_median(groups) == 2.5
        assert median_of_medians(groups) == 51.0
        assert pooled_median(groups) != median_of_medians(groups)

    def test_multi_source_concept_forces_raw_pool(self):
        # 上述两源若分布在不同表 → 决策必须强制原始池化（禁用 per-source 预聚合）
        sources = [FakeSource(table="chartevents"), FakeSource(table="labevents")]
        assert should_pool_raw(sources) is True


# --------------------------------------------------------------------------- #
# 3. AUMC/eICU 时间单位 (修复3: 单一来源的分钟→小时换算)
# --------------------------------------------------------------------------- #
class TestTimeUnits:
    def test_minutes_per_hour_constant(self):
        assert MINUTES_PER_HOUR == 60.0

    def test_scalar_minutes_to_hours(self):
        assert minutes_to_hours(60.0) == 1.0
        assert minutes_to_hours(90.0) == 1.5
        assert minutes_to_hours(0.0) == 0.0

    def test_series_minutes_to_hours(self):
        import pandas as pd

        s = pd.Series([0.0, 60.0, 120.0, 30.0])
        out = minutes_to_hours_series(s)
        assert list(out) == [0.0, 1.0, 2.0, 0.5]

    def test_single_conversion_contract(self):
        # 契约：每列只转一次。二次调用会产生 /3600 的二次缩放（这正是要避免的 bug）。
        import pandas as pd

        s = pd.Series([60.0])
        once = minutes_to_hours_series(s)
        twice = minutes_to_hours_series(once)
        assert once.iloc[0] == 1.0
        assert twice.iloc[0] == pytest.approx(1.0 / 60.0)


# --------------------------------------------------------------------------- #
# 4. 窗口聚合覆盖 (修复4: 声明式化 gcs/sofa_cardio 等散落特例)
# --------------------------------------------------------------------------- #
class TestWindowAggregateOverrides:
    def test_gcs_force_overrides_even_explicit(self):
        from easyicu.concept import resolve_window_aggregate

        # force=True: 即使传入别的显式聚合也覆盖为 min
        assert resolve_window_aggregate("gcs", None) == "min"
        assert resolve_window_aggregate("gcs", "median") == "min"
        assert resolve_window_aggregate("gcs", "max") == "min"
        # 已等于目标方法时保持
        assert resolve_window_aggregate("gcs", "min") == "min"

    def test_sofa_cardio_only_when_unset(self):
        from easyicu.concept import resolve_window_aggregate

        # force=False: 仅在未确定 (None) 时取 max；显式聚合保持不变
        assert resolve_window_aggregate("sofa_cardio", None) == "max"
        assert resolve_window_aggregate("sofa2_cardio", None) == "max"
        assert resolve_window_aggregate("sofa_cardio", "median") == "median"

    def test_unregistered_concept_passthrough(self):
        from easyicu.concept import resolve_window_aggregate

        assert resolve_window_aggregate("hr", None) is None
        assert resolve_window_aggregate("hr", "median") == "median"
        # VASO_RATE 概念不在覆盖表中 → 走默认（None 透传，后续用 median）
        assert resolve_window_aggregate("norepi_rate", None) is None


# --------------------------------------------------------------------------- #
# 5. ricu CSV 黄金基准 (fixture-gated, 缺失时自动跳过)
# --------------------------------------------------------------------------- #
class TestRicuCsvParity:
    def test_fixture_harness_skips_cleanly_when_absent(self):
        # 该用例本身验证 harness：有 fixture 则加载并断言列存在，无则 skip。
        require_ricu_fixtures()
        df = load_ricu_csv("o2sat")
        assert len(df.columns) > 0
