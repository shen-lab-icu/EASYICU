"""Urine gap/window thresholds are tolerant of float clocks (P1-3).

Synthetic raw rows only, loaded through the real per-database concept path
(MIIV datetimes, eICU minute offsets, SICdb second offsets).
"""

import numpy as np
import pandas as pd
import pytest

from easyicu.scores.urine_output import collection_intervals
from tests.support.urine_raw_source import _load, _urine_source

DATABASES = ("miiv", "eicu", "sic")
CONCEPTS = ["urine24", "uo_6h", "uo_12h", "uo_24h", "kdigo_uo"]
# Not whole hours or minutes; the first is 7 h 13 s.
SHIFTS_S = (7 * 3600 + 13, 1111, 3 * 3600 + 427, 86399)


def _series(database, tmp_path, seconds, amounts, shift_s, tag):
    stamps = (np.asarray(seconds, dtype=np.int64) + shift_s) / 3600.0
    source = _urine_source(tmp_path / tag, database, stamps, amounts)
    return _load(source, CONCEPTS)


def _gapped_records(extra_s=0):
    """Hourly quarter-past records, a gap of exactly 12 h (+extra_s), hourly again."""
    first = np.arange(31) * 3600 + 900
    second = first[-1] + 12 * 3600 + extra_s + np.arange(31) * 3600
    seconds = np.r_[first, second]
    amounts = np.where(np.arange(len(seconds)) % 5 == 0, 140.0, 60.0)
    return seconds, amounts


def _values(frame, column, id_col):
    return frame.sort_values([id_col, "charttime"])[column].to_numpy(dtype=float)


@pytest.mark.parametrize("database", DATABASES)
@pytest.mark.parametrize("extra_s", [0, 1])
def test_urine_concepts_do_not_depend_on_absolute_time_shift(tmp_path, database, extra_s):
    seconds, amounts = _gapped_records(extra_s)
    base = _series(database, tmp_path, seconds, amounts, 0, "base")
    for k, shift in enumerate(SHIFTS_S):
        shifted = _series(database, tmp_path, seconds, amounts, shift, f"shift{k}")
        for name in CONCEPTS:
            a, b = base[name].data, shifted[name].data
            id_col = a.columns[0]
            assert a.columns.tolist() == b.columns.tolist()
            for column in a.columns[2:]:
                if not pd.api.types.is_numeric_dtype(a[column]) or column.endswith("available_at"):
                    continue
                # The hourly grid moves by whole hours with the shift; the
                # non-missing values and their count must not change.
                x = _values(a, column, id_col)
                y = _values(b, column, id_col)
                assert np.isnan(x).sum() == np.isnan(y).sum(), (database, shift, name, column)
                if name == "urine24" and column == "urine24":
                    assert np.isfinite(x).any()
                np.testing.assert_allclose(
                    np.sort(x[np.isfinite(x)]), np.sort(y[np.isfinite(y)]),
                    rtol=0, atol=1e-9, err_msg=f"{database} {shift} {name} {column}",
                )


@pytest.mark.parametrize("database", DATABASES)
@pytest.mark.parametrize("shift_s", [0, *SHIFTS_S])
def test_exact_12h_gap_is_kept_and_one_second_more_is_rejected(tmp_path, database, shift_s):
    kept = _series(database, tmp_path, *_gapped_records(0), shift_s, "kept")["urine24"].data
    rejected = _series(database, tmp_path, *_gapped_records(1), shift_s, "rej")["urine24"].data
    # Windows spanning the gap are evaluable only when the 12 h gap is kept.
    assert int(kept.urine24.notna().sum()) == 50
    assert int(rejected.urine24.notna().sum()) == 15


@pytest.mark.parametrize("ends_h", [
    [0.1 + 0.2, 0.1 + 0.2 + 12.0],
    [7 + 13 / 3600, 7 + 13 / 3600 + 12.0],
    [1 / 3.0, 1 / 3.0 + 12.0],
    [5 * 0.1 + 7 / 3600, 5 * 0.1 + 7 / 3600 + 12.0],
])
def test_collection_intervals_keep_float_inexact_12h_and_reject_plus_one_second(ends_h):
    ends = np.asarray(ends_h)
    for source_is_rate in (True, False):
        valid = collection_intervals(ends, np.array([10.0, 10.0]), source_is_rate=source_is_rate)[3]
        assert valid[1], "an exact 12 h gap must be kept"
        later = ends + np.array([0.0, 1 / 3600])
        valid = collection_intervals(later, np.array([10.0, 10.0]), source_is_rate=source_is_rate)[3]
        assert not valid[1], "12 h + 1 s must be rejected"


def test_float_sweep_over_second_clocks_has_strict_failures_but_tolerant_boundary():
    starts = np.arange(0, 86400 * 3, 37)
    strict_failures = 0
    for s in starts:
        t0 = s / 3600.0
        t1 = (s + 12 * 3600) / 3600.0
        ends = np.array([t0, t0 + 12.0, t1])
        strict_failures += int(not (ends[1] - ends[0] <= 12.0)) + int(not (ends[2] - ends[0] <= 12.0))
        for later in (ends[1], ends[2]):
            pair = np.array([t0, later])
            assert collection_intervals(pair, np.array([5.0, 5.0]), source_is_rate=True)[3][1]
            plus = np.array([t0, (s + 12 * 3600 + 1) / 3600.0])
            assert not collection_intervals(plus, np.array([5.0, 5.0]), source_is_rate=True)[3][1]
    # The sweep must actually contain cases a strict comparison would reject.
    assert strict_failures > 0


def test_daily_volume_batch_keeps_float_inexact_12h_and_rejects_plus_one_second():
    from easyicu.scores.urine_output import daily_volume_batch

    def run(extra_s):
        for s in range(0, 3600, 7):
            first = s / 3600.0 + 1.0
            second = (s + extra_s) / 3600.0 + 13.0
            frame = pd.DataFrame({"stay_id": 1, "charttime": [first, second], "urine": [50.0, 50.0]})
            out = daily_volume_batch(frame, "stay_id", "charttime", "urine", 1.0, 24, 18, 12)
            yield out

    for out in run(0):
        assert out.urine24_covered_h.max() >= 1.0  # the 12 h interval carries coverage
    for out in run(1):
        assert out.urine24_covered_h.max() <= 1.0 + 1e-9  # 12 h + 1 s is a gap
