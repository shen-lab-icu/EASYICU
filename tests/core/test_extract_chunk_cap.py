"""提取分批的统一默认：任何库/模块，auto_batch_size 都不得切超过 MAX_EXTRACT_CHUNKS 份。

背景：ICU 队列即使 ~20 万患者（eICU 最大库），配合每模块流式落盘 + DuckDB 溢出落盘，
至多 3 份即可提取。历史上估算高估会把队列切成很多份，重复扫共享大表、拖慢数倍。
这是所有调用方共用的默认，不应由使用者每次手调 batch_size —— 故用测试锁死。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from easyicu.runtime.memory_manager import (
    auto_batch_size,
    MAX_EXTRACT_CHUNKS,
    _ceil_div,
)
from easyicu.api import EXTRACT_MODULES
from easyicu.api.extraction import (
    _adapt_stream_batch_size_from_first_batch,
    _interleave_stream_patient_ids,
    _next_stream_retry_batch_size,
    _resolve_stream_batch_size,
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
