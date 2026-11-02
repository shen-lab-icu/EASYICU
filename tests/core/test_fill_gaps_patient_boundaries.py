"""A fast grid must never lend one patient's array slice to another."""

import numpy as np
import pandas as pd
import pytest

from easyicu.io.ts_utils import fill_gaps


def fill(frame, limits, **kwargs):
    return fill_gaps(
        frame, ["stay_id"], "charttime", pd.Timedelta(hours=1),
        limits=limits, method="none", **kwargs,
    ).sort_values(["stay_id", "charttime"]).reset_index(drop=True)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_outside_observations_keep_their_owner_and_timestamp(dtype):
    frame = pd.DataFrame({
        "stay_id": [1, 1, 1, 2, 2],
        "charttime": [1., 3., 10., 0., 8.],
        "score": [3., 1., 4., np.nan, 2.],
    })
    limits = pd.DataFrame({"stay_id": [1, 2], "start": [3., 0.], "end": [7., 8.]})
    together = fill(frame, limits, value_dtype=dtype)
    separate = pd.concat([
        fill(frame[frame.stay_id.eq(i)], limits[limits.stay_id.eq(i)], value_dtype=dtype)
        for i in (1, 2)
    ], ignore_index=True)
    pd.testing.assert_frame_equal(together, separate)
    assert together.loc[together.stay_id.eq(1), "charttime"].tolist() == [1., 3., 4., 5., 6., 7., 10.]
    assert together.loc[together.stay_id.eq(2) & together.charttime.eq(2), "score"].isna().all()
    assert together.score.dtype == np.dtype(dtype)


def test_missing_limit_id_uses_its_own_observation_range():
    frame = pd.DataFrame({"stay_id": [1, 1, 2, 2], "charttime": [0., 2., 4., 6.], "score": [1., 2., 3., 4.]})
    limits = pd.DataFrame({"stay_id": [1], "start": [0.], "end": [2.]})
    result = fill(frame, limits)
    assert result.loc[result.stay_id.eq(2), "charttime"].tolist() == [4., 5., 6.]
    assert result.loc[result.stay_id.eq(2), "score"].iloc[[0, 2]].tolist() == [3., 4.]


def test_invalid_supplied_limits_do_not_change_other_patients():
    frame = pd.DataFrame({'stay_id': [1, 1, 2, 2], 'charttime': [10., 12., 0., 3.], 'score': [4., 4., 1., 2.]})
    limits = pd.DataFrame({'stay_id': [1, 2], 'start': [10., 0.], 'end': [2., 3.]})
    result = fill(frame, limits)
    # Preserve the established invalid-limit contract (omit that group),
    # without transferring either of its records to a valid group.
    assert result.stay_id.eq(2).all()
    pd.testing.assert_frame_equal(result, fill(frame[frame.stay_id.eq(2)], limits[limits.stay_id.eq(2)]))


def test_fast_grid_matches_independent_union_reindex_for_shuffled_stays():
    rng = np.random.default_rng(90627)
    frames, limits, expected = [], [], []
    for stay in range(1, 45):
        times = np.sort(rng.choice(np.arange(-5, 21), size=8, replace=False)).astype(float)
        values = rng.normal(size=len(times))
        part = pd.DataFrame({"stay_id": stay, "charttime": times, "score": values})
        frames.append(part)
        lo, hi = 0., 12.
        limits.append(dict(stay_id=stay, start=lo, end=hi))
        index = pd.Index(np.arange(lo, hi + 1)).union(pd.Index(times))
        out = part.set_index("charttime").reindex(index)
        out["stay_id"] = stay
        expected.append(out.rename_axis("charttime").reset_index()[part.columns])
    frame = pd.concat(frames).sample(frac=1, random_state=47).reset_index(drop=True)
    actual = fill(frame, pd.DataFrame(limits).sample(frac=1, random_state=48))
    pd.testing.assert_frame_equal(actual, pd.concat(expected, ignore_index=True), check_dtype=False)


def test_non_numeric_merge_fallback_preserves_outside_observations():
    frame = pd.DataFrame({"stay_id": [1, 1, 2], "charttime": [3., 10., 0.], "label": ["a", "b", "c"]})
    # Multiple ID columns force the merge path instead of numeric scatter.
    frame["site"] = "site"
    limits = pd.DataFrame({"stay_id": [1, 2], "site": "site", "start": [3., 0.], "end": [7., 4.]})
    result = fill_gaps(frame, ["stay_id", "site"], "charttime", pd.Timedelta(hours=1), limits=limits, method="none")
    assert result.loc[result.stay_id.eq(1) & result.charttime.eq(10), "label"].item() == "b"
    assert result.loc[result.stay_id.eq(2) & result.charttime.eq(2), "label"].isna().all()
