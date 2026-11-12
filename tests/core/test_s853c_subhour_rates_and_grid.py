"""S853c: FiO2/HR/NEE in subhour_signals, minute infusion rates and an exact minute grid."""

import numpy as np
import pandas as pd
import pytest

from easyicu.concept.minute_resolution import (
    active_minute_rate_grid,
    grid_bin_hours,
    grid_floor_index,
    minute_rate_grid,
    MINUTE_RESOLUTION_KWARG,
)

MINUTE = pd.Timedelta(minutes=1)


def test_module_roster_appends_the_e3_signals():
    from easyicu.api.extraction import EXTRACT_MODULES
    from easyicu.concept.catalog import SUBHOUR_SIGNAL_CONCEPTS

    assert SUBHOUR_SIGNAL_CONCEPTS == (
        "map_invasive", "cvp", "peep_set", "peep_total", "hr", "fio2_chart", "norepi_equiv",
    )
    assert EXTRACT_MODULES["subhour_signals"] == list(SUBHOUR_SIGNAL_CONCEPTS)


def test_grid_points_floor_to_themselves():
    g = 1 / 60
    k = np.arange(2000)
    points = k * g
    # The plain floor moves k = 31, 62, 123, ... into the previous minute.
    assert (np.floor(points / g) != k).any()
    assert (grid_floor_index(points, g) == k).all()
    # Values strictly inside a bin still floor down.
    assert (grid_floor_index(points + g / 2, g) == k).all()
    assert (grid_bin_hours(k, MINUTE) == points).all()


def test_sql_floor_keeps_hourly_text_and_uses_seconds_below_an_hour():
    from easyicu.datasource import _epoch_hours_floor_sql

    assert _epoch_hours_floor_sql("s", 1.0) == "FLOOR(s / 3600.0 / 1.0) * 1.0"
    assert _epoch_hours_floor_sql("s", 60 / 60) == "FLOOR(s / 3600.0 / 1.0) * 1.0"
    assert _epoch_hours_floor_sql("s", 1 / 60) == f"FLOOR(s / 60.0) * {1 / 60}"


def test_pandas_floors_keep_sub_hour_grid_points():
    from easyicu.io.ts_utils import round_to_interval
    from easyicu.utils.compat import round_to_interval as compat_round

    points = pd.Series(np.arange(400) * (1 / 60))
    assert (compat_round(points, 1 / 60) == points).all()
    assert (round_to_interval(points, MINUTE, time_unit="hours") == points).all()
    hourly = pd.Series([0.0, 0.5, 1.999999999999, 2.0])
    assert compat_round(hourly, 1.0).tolist() == [0.0, 0.0, 1.0, 2.0]


def test_rate_grid_is_scoped_to_the_explicit_opt_in():
    assert active_minute_rate_grid() is None
    with minute_rate_grid({}, MINUTE):
        assert active_minute_rate_grid() is None
    with minute_rate_grid({MINUTE_RESOLUTION_KWARG: True}, pd.Timedelta(hours=1)):
        assert active_minute_rate_grid() is None
    with minute_rate_grid({MINUTE_RESOLUTION_KWARG: True}, MINUTE):
        assert active_minute_rate_grid() == MINUTE
    assert active_minute_rate_grid() is None


def test_hirid_minute_bins_spread_each_dose_over_its_interval():
    from easyicu.utils.callback_utils import _hirid_apportion_grid

    frame = pd.DataFrame({
        "patientid": [1, 1, 1],
        "givenat": [10.0 + 50 / 60, 11.0 + 5 / 60, 13.5],   # 10:50 start, 11:05 dose, 13:30 untimed
        "_prev_t": [np.nan, 10.0 + 50 / 60, np.nan],
        "givendose": [0.0, 15.0, 6.0],
    })
    out = _hirid_apportion_grid(frame.iloc[1:], "givenat", "givendose", MINUTE)
    timed = out.loc[out.givenat.lt(12)]
    # 15 over the 15 minutes 10:50..11:05: 1 per minute bin, bins 650..664.
    assert np.round(timed._time_bin * 60).astype(int).tolist() == list(range(650, 665))
    assert np.allclose(timed.givendose, 1.0)
    untimed = out.loc[out.givenat.eq(13.5)]
    # No usable interval: spread evenly over its record's clock hour, like the hourly bin.
    assert np.round(untimed._time_bin * 60).astype(int).tolist() == list(range(780, 840))
    assert np.isclose(untimed.givendose.sum(), 6.0)
    assert (out._time_bin == grid_bin_hours(np.round(out._time_bin * 60), MINUTE)).all()


def test_sicdb_minute_intervals_carry_the_rate_on_every_minute():
    from easyicu.concept.sicdb_sources import materialize_intervals

    frame = pd.DataFrame({"CaseID": [1], "Offset": [600.0], "OffsetDrugEnd": [600.0 + 150.0], "rate": [2.0]})
    hourly = materialize_intervals(frame, relative=True)
    minute = materialize_intervals(frame, relative=True, grid=MINUTE)
    assert hourly.Offset.tolist() == [0.0]
    # 10:00..12:30 overlaps minute bins 10, 11 and 12.
    assert np.round(minute.Offset * 60).astype(int).tolist() == [10, 11, 12]
    assert minute.rate.tolist() == [2.0, 2.0, 2.0]


@pytest.mark.parametrize("database", ["hirid", "sic"])
def test_minute_nee_averages_to_the_hourly_nee(tmp_path, database):
    from easyicu import load_concepts
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source

    generate_source(tmp_path / "src", database, stays=3)
    try:
        minute = load_concepts(concepts=["norepi_equiv"], database=database, data_path=str(tmp_path / "src"),
                               merge=True, interval=MINUTE, **{MINUTE_RESOLUTION_KWARG: True}, verbose=False)
        hourly = load_concepts(concepts=["norepi_equiv"], database=database, data_path=str(tmp_path / "src"),
                               merge=True, verbose=False)
    finally:
        clear_global_loader()
    ident = minute.columns[0]
    minutes = np.round(minute.charttime * 60).astype(int)
    assert minute.charttime.eq(grid_bin_hours(minutes, MINUTE)).all()
    # Contiguous minute rows inside each infusion (no minute lost to float flooring).
    assert minutes.groupby(minute[ident]).diff().dropna().isin([1]).mean() > 0.95
    per_hour = minute.assign(hour=minutes // 60).groupby([ident, "hour"]).norepi_equiv.sum() / 60
    expected = hourly.rename(columns={hourly.columns[0]: ident}).set_index([ident, "charttime"]).norepi_equiv
    expected.index = expected.index.set_names([ident, "hour"])
    joined = pd.concat([per_hour.rename("m"), expected.rename("h")], axis=1)
    assert joined.notna().all().all() and np.allclose(joined.m, joined.h)


def test_hirid_exact_minute_sample_keeps_its_minute(tmp_path):
    from easyicu import load_concepts
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / "src"
    generate_source(root, "hirid", stays=3)
    general = pd.read_parquet(next(p for p in (root / "general_table.parquet", root / "general.parquet") if p.exists()))
    path = root / "observations" / "1.parquet"
    table = pd.read_parquet(path)
    anchor = table.loc[table.variableid.eq(110)].iloc[[0]]
    patient = int(anchor.patientid.iloc[0])
    admitted = pd.to_datetime(general.loc[general.patientid.eq(patient), "admissiontime"].iloc[0])
    extra = pd.concat([anchor.assign(datetime=admitted + pd.Timedelta(minutes=m), value=70.0 + m / 1000)
                       for m in (31, 123, 124, 245)])
    pd.concat([table, extra], ignore_index=True).astype(table.dtypes.to_dict()).to_parquet(path, index=False)
    try:
        out = load_concepts(concepts=["map_invasive"], database="hirid", data_path=str(root), merge=True,
                            interval=MINUTE, **{MINUTE_RESOLUTION_KWARG: True}, verbose=False)
    finally:
        clear_global_loader()
    stay = out.loc[out.iloc[:, 0].eq(patient)]
    by_minute = dict(zip(np.round(stay.charttime * 60).astype(int), stay.map_invasive))
    assert {m: by_minute.get(m) for m in (31, 123, 124, 245)} == {
        m: pytest.approx(70.0 + m / 1000) for m in (31, 123, 124, 245)
    }
