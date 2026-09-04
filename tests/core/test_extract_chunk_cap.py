"""提取资源策略的通用上限和逐数据库、逐模块实测覆盖。

通用 ``auto_batch_size`` 保留最多三份的旧交互式约束；正式 release 优先使用逐模块
实测 profile，已知重模块可以超过三份。语义或依赖改变后，旧 profile 必须显式失效，
不能继续准入 one-shot。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from easyicu.runtime.memory_manager import (
    auto_batch_size,
    MAX_EXTRACT_CHUNKS,
    _ceil_div,
)
from easyicu.api import EXTRACT_MODULES
from easyicu.api.extraction import (
    _INVALIDATED_MEASURED_PROFILES,
    _MEASURED_BATCH_PROFILES,
    _MEASURED_ONESHOT_PROFILES,
    _adapt_stream_batch_size_from_first_batch,
    _interleave_stream_patient_ids,
    _next_stream_retry_batch_size,
    _extract_worker_env_setup,
    _resource_budget_execution_limits,
    _resolve_stream_batch_size,
    plan_extraction_resources,
    plan_module_extraction_resources,
)


# 覆盖 6 个库的真实规模（含 eICU 最大 20 万），以及最宽 / 最重的模块。
_DB_SIZES = {
    "eicu": 200_000,
    "miiv": 94_458,
    "mimic": 61_532,
    "hirid": 33_000,
    "sic": 27_000,
    "aumc": 23_000,
}
_MODULES = ["medications", "chemistry", "hematology", "vitals", "renal", "blood_gas"]
# 用固定的"物理总内存"预算，复现提取路径 (_run_module_extraction 传 total_ram)。
_STABLE_AVAIL_MB = 16 * 1024.0


def _n_chunks(total: int, batch_size):
    return 1 if not batch_size else _ceil_div(total, batch_size)


@pytest.mark.parametrize("db,total", sorted(_DB_SIZES.items()))
@pytest.mark.parametrize("module", _MODULES)
def test_never_exceeds_chunk_cap(db, total, module):
    concepts = EXTRACT_MODULES[module]
    bs = auto_batch_size(concepts, db, total, available_memory_mb=_STABLE_AVAIL_MB)
    n = _n_chunks(total, bs)
    assert n <= MAX_EXTRACT_CHUNKS, (
        f"{db}/{module} N={total} 切成了 {n} 份 (batch_size={bs}), "
        f"超过统一上限 {MAX_EXTRACT_CHUNKS}"
    )
    if bs is not None:
        # 分批时每份必须是正的、整千的，且确实小于全量
        assert bs > 0 and bs % 1000 == 0 and bs < total


def test_largest_db_at_most_three_chunks():
    """eICU 20 万患者：最宽模块也至多 3 份。"""
    for module in ("medications", "chemistry"):
        bs = auto_batch_size(
            EXTRACT_MODULES[module], "eicu", 200_000,
            available_memory_mb=_STABLE_AVAIL_MB,
        )
        assert _n_chunks(200_000, bs) <= 3


def test_eicu_heaviest_modules_use_three_batches_on_16gb():
    """Pin the established laptop-safe eICU contract for the heavy modules."""
    for module in ("medications", "chemistry"):
        bs = auto_batch_size(
            EXTRACT_MODULES[module],
            "eicu",
            200_000,
            available_memory_mb=_STABLE_AVAIL_MB,
        )
        assert _n_chunks(200_000, bs) == 3


def test_small_cohort_is_one_shot():
    """小队列 / 窄模块应一次性(返回 None)，不做无意义分批。"""
    bs = auto_batch_size(
        EXTRACT_MODULES["vitals"], "aumc", 5_000,
        available_memory_mb=_STABLE_AVAIL_MB,
    )
    assert bs is None


def test_cap_constant_is_three():
    assert MAX_EXTRACT_CHUNKS == 3


@pytest.mark.parametrize(
    ("database", "total", "available_gb", "expected"),
    [
        ("eicu", 200_859, 32, 67_000),
        ("eicu", 200_859, 16, 50_000),
        ("eicu", 200_859, 14, 43_000),
        ("eicu", 200_859, 11, 34_000),
        ("eicu", 200_859, 8, 25_000),
        ("eicu", 200_859, 6, 16_000),
        ("eicu", 200_859, 4, 8_000),
        ("miiv", 94_458, 24, 94_458),
        ("miiv", 94_458, 16, 67_000),
        ("miiv", 94_458, 8, 37_000),
        ("mimic", 61_532, 24, 61_532),
        ("mimic", 61_532, 16, 41_000),
        ("mimic", 61_532, 8, 20_000),
        ("hirid", 33_905, 16, 29_000),
        ("hirid", 33_905, 8, 14_000),
        ("aumc", 23_106, 24, 13_000),
        ("aumc", 23_106, 16, 9_000),
        ("aumc", 23_106, 8, 5_000),
        ("sic", 27_386, 16, 27_386),
    ],
)
def test_stream_batch_uses_current_available_memory(
    database, total, available_gb, expected
):
    assert (
        _resolve_stream_batch_size(
            database,
            total,
            available_memory_mb=available_gb * 1024,
        )
        == expected
    )


@pytest.mark.parametrize("available_gb", [8, 16])
@pytest.mark.parametrize(
    "database,total",
    [(database, _DB_SIZES[database]) for database in ("mimic", "miiv", "aumc")],
)
def test_sub24gb_high_risk_cohorts_require_a_measured_pilot(
    database, total, available_gb
):
    batch_size = _resolve_stream_batch_size(
        database,
        total,
        available_memory_mb=available_gb * 1024,
    )

    assert 5_000 <= batch_size < total


def test_lower_risk_cohort_stays_one_shot_when_calibrated_peak_fits():
    assert (
        _resolve_stream_batch_size(
            "sic",
            _DB_SIZES["sic"],
            available_memory_mb=16 * 1024,
        )
        == _DB_SIZES["sic"]
    )
    assert (
        _resolve_stream_batch_size(
            "hirid",
            _DB_SIZES["hirid"],
            available_memory_mb=20 * 1024,
        )
        == _DB_SIZES["hirid"]
    )


def test_guarded_one_shot_is_split_evenly_not_into_a_tiny_tail():
    assert _resolve_stream_batch_size(
        "mimic",
        60_000,
        available_memory_mb=24_500,
    ) == 30_000


def test_low_memory_initial_batches_are_calibrated_not_a_fixed_10k_tier():
    planned = {
        database: _resolve_stream_batch_size(
            database,
            total,
            available_memory_mb=8 * 1024,
        )
        for database, total in _DB_SIZES.items()
    }

    assert planned == {
        "eicu": 25_000,
        "miiv": 37_000,
        "mimic": 20_000,
        "hirid": 14_000,
        "sic": 16_000,
        "aumc": 5_000,
    }


@pytest.mark.parametrize(
    ("initial", "expected"),
    [(67_000, 50_000), (50_000, 35_000), (45_000, 30_000), (10_000, 5_000), (5_000, 5_000)],
)
def test_adaptive_worker_crash_retry_reduces_only_the_failed_module(
    initial, expected
):
    assert _next_stream_retry_batch_size(initial) == expected


def test_explicit_stream_batch_size_always_wins():
    assert (
        _resolve_stream_batch_size(
            "eicu",
            200_859,
            25_000,
            available_memory_mb=4 * 1024,
        )
        == 25_000
    )


def test_8gib_resource_budget_owns_lower_layer_worker_limits(monkeypatch):
    limits = _resource_budget_execution_limits(8 * 1024)

    assert limits == {
        "resource_budget_mb": 8192.0,
        "modeled_total_memory_gb": pytest.approx(11.428571),
        "parallel_max_workers": 2,
        "arrow_threads": 2,
        "duckdb_threads": 2,
        "duckdb_memory_limit_mb": 2048,
        "resolver_cache_budget_mb": 512,
    }

    # A reproducible explicit contract replaces host-sized defaults inside the
    # dedicated worker.  The worker process exits after extraction, so these
    # values cannot leak back to the calling application in production.
    monkeypatch.setenv("EASYICU_PARALLEL_MAX_WORKERS", "64")
    monkeypatch.setenv("EASYICU_DUCKDB_MEMORY_LIMIT", "4GB")
    _extract_worker_env_setup("/data/aumc", 8 * 1024)
    assert os.environ["EASYICU_RESOURCE_BUDGET_MB"] == "8192.0"
    assert os.environ["EASYICU_PARALLEL_MAX_WORKERS"] == "2"
    assert os.environ["EASYICU_ARROW_THREADS"] == "2"
    assert os.environ["EASYICU_DUCKDB_THREADS"] == "2"
    assert os.environ["EASYICU_DUCKDB_MEMORY_LIMIT"] == "2048MB"
    assert os.environ["EASYICU_CACHE_BUDGET_MB"] == "512"


def test_measured_miiv_blood_gas_uses_one_shot_with_2gib_available():
    """1.66-GiB measured peak + 10% headroom fits inside 2 GiB."""

    plan = plan_extraction_resources(
        "miiv",
        ["blood_gas"],
        94_458,
        available_memory_mb=2 * 1024,
    )

    assert plan.mode == "one_shot"
    assert plan.reason_code == "measured_profile_fast_path"
    assert plan.batch_size == 94_458
    assert plan.measured_peak_rss_mb == pytest.approx(1_658.219)
    assert plan.required_available_memory_mb == pytest.approx(1_824.041)
    assert plan.advisory is None
    assert plan.advisory_zh is None


def test_measured_eicu_profile_can_authorize_full_cohort_above_legacy_size_cap():
    plan = plan_extraction_resources(
        "eicu",
        ["blood_gas"],
        200_859,
        available_memory_mb=8 * 1024,
    )

    assert plan.mode == "one_shot"
    assert plan.reason_code == "measured_profile_fast_path"
    assert plan.batch_size == 200_859
    assert plan.measured_peak_rss_mb == pytest.approx(1_104.6)
    assert plan.advisory is None


def test_measured_eicu_respiratory_uses_fastest_verified_five_batches():
    plan = plan_extraction_resources(
        "eicu",
        ["respiratory"],
        200_859,
        available_memory_mb=8 * 1024,
    )

    assert plan.mode == "patient_batches"
    assert plan.reason_code == "measured_profile_fastest_safe_batch"
    assert plan.batch_size == 50_000
    assert _n_chunks(200_859, plan.batch_size) == 5
    assert plan.measured_peak_rss_mb == pytest.approx(6_252.8)
    assert plan.required_available_memory_mb == pytest.approx(6_878.08)
    assert plan.advisory is None
    assert plan.advisory_zh is None


def test_eicu_full_request_remains_guarded_after_execution_envelope_change():
    plan = plan_extraction_resources(
        "eicu",
        list(EXTRACT_MODULES),
        200_859,
        available_memory_mb=8 * 1024,
    )

    assert plan.reason_code == "invalidated_profile_memory_guard"
    assert plan.measured_peak_rss_mb is None
    assert plan.advisory


def test_mixed_batch_summary_includes_larger_oneshot_peak(monkeypatch):
    monkeypatch.setitem(
        _MEASURED_ONESHOT_PROFILES,
        "fixture",
        {"light": {"cohort_stays": 10_000, "peak_rss_mb": 7_000.0}},
    )
    monkeypatch.setitem(
        _MEASURED_BATCH_PROFILES,
        "fixture",
        {
            "heavy": {
                "cohort_stays": 10_000,
                "batch_size": 5_000,
                "peak_rss_mb": 6_000.0,
            }
        },
    )

    plan = plan_extraction_resources(
        "fixture",
        ["light", "heavy"],
        10_000,
        available_memory_mb=8 * 1024,
    )

    assert plan.reason_code == "measured_profile_fastest_safe_batch"
    assert plan.batch_size == 5_000
    assert plan.measured_peak_rss_mb == 7_000.0
    assert plan.required_available_memory_mb == pytest.approx(7_700.0)


def test_invalidated_profiles_cannot_remain_in_measured_registries():
    for database, module in _INVALIDATED_MEASURED_PROFILES:
        assert module not in _MEASURED_ONESHOT_PROFILES.get(database, {})
        assert module not in _MEASURED_BATCH_PROFILES.get(database, {})


def test_eicu_mixed_request_keeps_each_measured_module_strategy_at_8gib():
    plans = plan_module_extraction_resources(
        "eicu",
        [
            "respiratory",
            "sepsis_shared",
            "sofa1_score",
            "sofa2_score",
            "sepsis3_sofa1",
            "sepsis3_sofa2",
        ],
        200_859,
        available_memory_mb=8 * 1024,
    )

    assert {module: plan.batch_size for module, plan in plans.items()} == {
        "respiratory": 50_000,
        "sepsis_shared": 200_859,
        "sofa1_score": 67_000,
        "sofa2_score": 25_000,
        "sepsis3_sofa1": 67_000,
        "sepsis3_sofa2": 25_000,
    }
    assert plans["sepsis_shared"].mode == "one_shot"
    assert plans["sofa2_score"].reason_code == (
        "invalidated_profile_memory_guard"
    )
    assert plans["sepsis3_sofa2"].reason_code == (
        "invalidated_profile_memory_guard"
    )
    assert plans["respiratory"].reason_code == (
        "measured_profile_fastest_safe_batch"
    )


def test_explicit_batch_override_still_applies_to_every_module():
    plans = plan_module_extraction_resources(
        "eicu",
        ["respiratory", "sepsis_shared"],
        200_859,
        requested_batch_size=5_000,
        available_memory_mb=8 * 1024,
    )

    assert {plan.batch_size for plan in plans.values()} == {5_000}
    assert {plan.reason_code for plan in plans.values()} == {
        "explicit_batch_size"
    }


def test_measured_miiv_full_module_set_uses_one_shot_at_8gib():
    plan = plan_extraction_resources(
        "miiv",
        list(EXTRACT_MODULES),
        94_458,
        available_memory_mb=8 * 1024,
    )

    assert plan.mode == "one_shot"
    assert plan.reason_code == "measured_profile_fast_path"
    assert plan.batch_size == 94_458
    assert plan.measured_peak_rss_mb == pytest.approx(7_362.0)
    assert plan.required_available_memory_mb == pytest.approx(8_098.2)
    assert plan.advisory is None
    assert plan.advisory_zh is None


def test_measured_miiv_renal_warns_only_below_its_one_shot_threshold():
    plan = plan_extraction_resources(
        "miiv",
        ["renal"],
        94_458,
        available_memory_mb=8_000,
    )

    assert plan.mode == "patient_batches"
    assert plan.reason_code == "measured_profile_insufficient_memory"
    assert plan.measured_peak_rss_mb == pytest.approx(7_362.0)
    assert plan.required_available_memory_mb == pytest.approx(8_098.2)
    assert plan.advisory
    assert plan.advisory_zh


def test_measured_mimic_vasopressors_use_one_shot_at_8gib():
    plan = plan_extraction_resources(
        "mimic",
        ["vasopressors"],
        61_532,
        available_memory_mb=8 * 1024,
    )

    assert plan.mode == "one_shot"
    assert plan.reason_code == "measured_profile_fast_path"
    assert plan.batch_size == 61_532
    assert plan.measured_peak_rss_mb == pytest.approx(7_415.4)
    assert plan.required_available_memory_mb == pytest.approx(8_156.94)
    assert plan.advisory is None


def test_measured_mimic_medications_use_fastest_verified_two_batches():
    plan = plan_extraction_resources(
        "mimic",
        ["medications"],
        61_532,
        available_memory_mb=8 * 1024,
    )

    assert plan.mode == "patient_batches"
    assert plan.reason_code == "measured_profile_fastest_safe_batch"
    assert plan.batch_size == 31_000
    assert plan.measured_peak_rss_mb == pytest.approx(7_236.8)
    assert plan.required_available_memory_mb == pytest.approx(7_960.48)
    assert plan.advisory is None


def test_mimic_full_module_request_remains_guarded_until_last_five_are_measured():
    plan = plan_extraction_resources(
        "mimic",
        list(EXTRACT_MODULES),
        61_532,
        available_memory_mb=8 * 1024,
    )

    assert plan.mode == "patient_batches"
    assert plan.reason_code == "unmeasured_profile_memory_guard"
    assert plan.advisory_zh


def test_measured_eicu_batch_shrinks_and_warns_below_verified_batch_threshold():
    plan = plan_extraction_resources(
        "eicu",
        ["respiratory"],
        200_859,
        available_memory_mb=6 * 1024,
    )

    assert plan.reason_code == "measured_profile_insufficient_memory"
    assert plan.batch_size == 40_000
    assert "fastest-batch threshold" in plan.advisory
    assert "最快批次门槛" in plan.advisory_zh
    assert "速度会变慢" in plan.advisory_zh


def test_measured_module_batches_and_warns_only_below_its_threshold():
    plan = plan_extraction_resources(
        "miiv",
        ["blood_gas"],
        94_458,
        available_memory_mb=1_800,
    )

    assert plan.mode == "patient_batches"
    assert plan.reason_code == "measured_profile_insufficient_memory"
    assert plan.batch_size == 48_000
    assert _n_chunks(94_458, plan.batch_size) == 2
    assert "1.78 GiB" in plan.advisory
    assert "1.78 GiB" in plan.advisory_zh
    assert "速度会变慢" in plan.advisory_zh


def test_unmeasured_module_cannot_borrow_a_light_module_fast_path():
    plan = plan_extraction_resources(
        "mimic",
        ["blood_gas", "other_scores"],
        61_532,
        available_memory_mb=2 * 1024,
    )

    assert plan.mode == "patient_batches"
    assert plan.reason_code == "unmeasured_profile_memory_guard"
    assert plan.required_available_memory_mb == 24 * 1024
    assert plan.advisory_zh


def test_explicit_batch_remains_authoritative_without_cleanup_advisory():
    plan = plan_extraction_resources(
        "miiv",
        ["blood_gas"],
        94_458,
        requested_batch_size=10_000,
        available_memory_mb=64 * 1024,
    )

    assert plan.mode == "patient_batches"
    assert plan.reason_code == "explicit_batch_size"
    assert plan.batch_size == 10_000
    assert plan.advisory is None


def test_first_measured_batch_can_grow_40k_to_67k():
    assert (
        _adapt_stream_batch_size_from_first_batch(
            40_000,
            observed_working_set_mb=2_000,
            available_memory_mb=8 * 1024,
            remaining_patients=160_859,
        )
        == 67_000
    )


def test_first_measured_batch_can_shrink_when_real_working_set_is_heavy():
    assert (
        _adapt_stream_batch_size_from_first_batch(
            40_000,
            observed_working_set_mb=8_000,
            available_memory_mb=8 * 1024,
            remaining_patients=160_859,
        )
        == 30_000
    )


def test_measured_batch_plan_never_exceeds_remaining_cohort():
    assert (
        _adapt_stream_batch_size_from_first_batch(
            40_000,
            observed_working_set_mb=1_000,
            available_memory_mb=8 * 1024,
            remaining_patients=12_345,
        )
        == 12_345
    )


def test_interleaved_stream_partition_preserves_exact_ids_and_batch_size():
    patient_ids = list(range(200_859))

    ordered, planned_batches = _interleave_stream_patient_ids(
        patient_ids,
        67_000,
    )

    assert planned_batches == 3
    assert len(ordered) == len(patient_ids)
    assert set(ordered) == set(patient_ids)
    assert len(set(ordered)) == len(patient_ids)
    assert [
        len(ordered[start : start + 67_000])
        for start in range(0, len(ordered), 67_000)
    ] == [67_000, 67_000, 66_859]


def test_interleaved_stream_partition_balances_source_order_density():
    patient_ids = list(range(200_859))
    ordered, _ = _interleave_stream_patient_ids(patient_ids, 67_000)
    batch_means = [
        sum(batch) / len(batch)
        for start in range(0, len(ordered), 67_000)
        if (batch := ordered[start : start + 67_000])
    ]
    sequential_means = [
        sum(batch) / len(batch)
        for start in range(0, len(patient_ids), 67_000)
        if (batch := patient_ids[start : start + 67_000])
    ]

    # A sequential split differs by ~67k in this monotone density proxy.  The
    # interleaved split samples the whole source-order range in every batch.
    assert max(batch_means) - min(batch_means) < (
        max(sequential_means) - min(sequential_means)
    ) / 100


def test_interleaved_stream_partition_is_noop_for_one_shot_cohort():
    patient_ids = [11, 12, 13]
    ordered, planned_batches = _interleave_stream_patient_ids(patient_ids, 10)
    assert ordered == patient_ids
    assert planned_batches == 1
