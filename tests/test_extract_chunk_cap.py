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
