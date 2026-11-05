"""Round-8 HiRID clocks: invented records only, through the public owners."""

from math import ceil, floor

import pandas as pd
import pytest

from easyicu.resources import load_dictionary
from tests.support.hirid_source import hirid_load, hirid_native, hirid_source_root


@pytest.mark.parametrize("prefix", ["norepi", "dobu", "epi"])
def test_r8_h1_five_minute_episodes_merge_on_the_source_clock(tmp_path, monkeypatch, prefix):
    from easyicu.utils.callback_utils import hirid_duration

    name = f"{prefix}_dur"
    item = load_dictionary()[name].sources["hirid"][0].ids[0]
    rows = []
    for patient, gap in ((1, 300), (2, 301)):
        for infusion, seconds, dose in ((1, 3, 0.), (1, 3603, 1.),
                                         (2, 3603 + gap, 0.), (2, 7203 + gap, 1.)):
            rows.append(dict(patientid=patient, givenat=seconds / 3600, pharmaid=item,
                             infusionid=infusion, givendose=dose, recordstatus=0,
                             doseunit="µg", route="iv-inf"))
    # Existing callback contracts also accept subsecond source timestamps.
    rows.extend(dict(patientid=3, givenat=t, pharmaid=item, infusionid=7,
                     givendose=dose, recordstatus=0, doseunit="µg", route="iv-inf")
                for t, dose in ((.001, 0.), (1., 1.)))
    frame = pd.DataFrame(rows)
    expected = {1: ([3 / 3600], [7500 / 3600]),
                2: ([3 / 3600, 3904 / 3600], [1., 1.]),
                3: ([.001], [.999])}

    def check(out, native=False):
        id_col, clock = ("stay_id", "charttime") if native else ("patientid", "givenat")
        for patient, (starts, durations) in expected.items():
            part = out.loc[out[id_col].eq(patient) & out[name].notna()].sort_values(clock)
            assert part[name].tolist() == pytest.approx(durations)
            assert part[clock].tolist() == pytest.approx(starts)

    source_offsets = pd.to_timedelta((frame.givenat * 3_600_000_000).round(), unit="us")
    for clock in (frame.givenat, source_offsets, pd.Timestamp("2150-01-01") + source_offsets):
        raw = frame.assign(givenat=clock)
        direct = hirid_duration(raw, concept_name=name, val_col="givendose",
                                index_col="givenat", grp_var="infusionid")
        # Absolute callback input historically uses its first observed time as origin.
        if pd.api.types.is_datetime64_any_dtype(clock):
            direct["givenat"] += 3 / 3600
        check(direct)
    root = hirid_source_root(tmp_path, "pharma", rows)
    origins = pd.read_parquet(root / "general.parquet").set_index("patientid").admissiontime
    pharma = pd.read_parquet(root / "pharma/1.parquet")
    pharma["givenat"] = pd.to_datetime(pharma.patientid.map(origins)) + source_offsets
    pharma.to_parquet(root / "pharma/1.parquet", index=False)
    for names in ([name], [name, "hr"]):
        check(hirid_load(root, names)[name])
    check(hirid_native(root, [name, "vaso_ind"], tmp_path / "native", monkeypatch, "medications"), native=True)


@pytest.mark.parametrize("seconds, states", [
    ([1800, 5400], [1., 6.]),
    ([1800, 7200], [1., 6.]),
    ([1800, 2700, 5400], [1., 3., 6.]),
    # Multiplying hours by 60 before differencing pushes the stop past 19 h.
    ([61457, 68400], [1., 6.]),
])
def test_r8_h2_categories_publish_only_positive_half_open_overlap(tmp_path, monkeypatch, seconds, states):
    from easyicu.utils.callback_utils import hirid_vent

    frame = pd.DataFrame({"patientid": 1, "datetime": [s / 3600 for s in seconds], "value": states})
    expected = {level: {float(h) for start, stop, state in zip(seconds[:-1], seconds[1:], states[:-1])
                        if state == code for h in range(start // 3600, ceil(stop / 3600))}
                for level, code in (("invasive", 1.), ("noninvasive", 3.))}

    def hours(out, level, *, native=False):
        clock = "charttime" if native else "datetime"
        selected = out.loc[out.mech_vent.astype("string").str.split("|").map(
            lambda levels: level in levels if isinstance(levels, list) else False)]
        if "dur_var" in selected:
            return {float(h) for start, duration in selected[[clock, "dur_var"]].itertuples(index=False, name=None)
                    for h in range(floor(start), ceil(start + duration)) if duration > 0}
        return set(selected[clock])

    direct = hirid_vent(frame, concept_name="mech_vent", val_col="value", padding_hours=0.)
    for level, code in (("invasive", 1.), ("noninvasive", 3.)):
        assert set(direct.loc[direct.mech_vent.eq(code), "datetime"]) == expected[level]
    root = hirid_source_root(tmp_path, "observations", frame.assign(variableid=15001552).to_dict("records"))
    origins = pd.read_parquet(root / "general.parquet").set_index("patientid").admissiontime
    observations = pd.read_parquet(root / "observations/1.parquet")
    observations["datetime"] = pd.to_datetime(observations.patientid.map(origins)) + pd.to_timedelta(seconds, unit="s")
    observations.to_parquet(root / "observations/1.parquet", index=False)
    for names in (["mech_vent"], ["mech_vent", "hr"]):
        out = hirid_load(root, names)["mech_vent"]
        for level in expected:
            assert hours(out, level) == expected[level]
    native = hirid_native(root, ["mech_vent", "vent_ind"], tmp_path / "native", monkeypatch, "respiratory")
    # The existing categorical aggregation publishes both levels for a mixed hour.
    assert hours(native, "invasive", native=True) == expected["invasive"]
    assert hours(native, "noninvasive", native=True) == expected["noninvasive"]


def test_r8_h3_noon_reset_survives_float_hours_and_matches_the_raw_ledger(tmp_path, monkeypatch):
    from easyicu.concept.hirid_fluid_counters import (
        INPUT_COUNTER_ID, OUTPUT_COUNTER_ID, _absolute_times, hirid_counter_frame,
    )

    origin = pd.Timestamp("2150-01-01 09:30:01")
    origins = pd.Series([origin], index=[1])
    clocks = pd.to_datetime(["2150-01-01 11:00:00", "2150-01-02 11:00:00", "2150-01-02 12:00:00",
                             "2150-01-02 13:00:00", "2150-01-02 14:00:00"])
    raw = pd.concat([pd.DataFrame({"patientid": 1, "variableid": item, "datetime": clocks, "value": values})
                     for item, values in ((INPUT_COUNTER_ID, [0., 600., 50., 200., 300.]),
                                          (OUTPUT_COUNTER_ID, [0., 200., 20., 50., 100.]))], ignore_index=True)
    relative = raw.assign(datetime=(raw.datetime - origin).dt.total_seconds() / 3600)
    pd.testing.assert_series_equal(_absolute_times(relative, "patientid", "datetime", origins),
                                   raw.datetime, check_names=False)

    expected = {}
    for name, net in (("total_input_ml", False), ("fluid_balance", True)):
        readings = raw if net else raw.loc[raw.variableid.eq(INPUT_COUNTER_ID)]
        exact = hirid_counter_frame(readings, concept_name=name, origins=origins,
                                    value_col="value", time_col="datetime", net_balance=net)
        exact["datetime"] = (exact.datetime - origin).dt.total_seconds() / 3600
        expected[name] = exact.set_index("datetime")[name]
        numeric = hirid_counter_frame(relative if net else relative.loc[relative.variableid.eq(INPUT_COUNTER_ID)],
                                      concept_name=name, origins=origins, value_col="value",
                                      time_col="datetime", net_balance=net)
        pd.testing.assert_frame_equal(numeric, exact, check_dtype=False, atol=1e-9, rtol=0)
    expected["fluid_balance_cumulative"] = expected["fluid_balance"].cumsum()
    assert expected["total_input_ml"].sum() == pytest.approx(900.)
    assert expected["fluid_balance_cumulative"].iloc[-1] == pytest.approx(600.)

    root = hirid_source_root(tmp_path, "observations", relative.to_dict("records"))
    general = pd.read_parquet(root / "general.parquet")
    general["admissiontime"] = pd.to_datetime(general.admissiontime)
    general.loc[general.patientid.eq(1), "admissiontime"] = origin
    general.to_parquet(root / "general.parquet", index=False)
    raw.to_parquet(root / "observations/1.parquet", index=False)
    names = list(expected)
    for requested in ([n] for n in names):
        out = hirid_load(root, requested)[requested[0]]
        clock = "datetime" if "datetime" in out else "charttime"
        actual = out.set_index(clock)[requested[0]].dropna()
        pd.testing.assert_series_equal(actual, expected[requested[0]], check_names=False,
                                       check_dtype=False, atol=1e-9, rtol=0)
    for name, out in hirid_load(root, names).items():
        clock = "datetime" if "datetime" in out else "charttime"
        pd.testing.assert_series_equal(out.set_index(clock)[name].dropna(), expected[name],
                                       check_names=False, check_dtype=False, atol=1e-9, rtol=0)
    native = hirid_native(root, names, tmp_path / "native", monkeypatch, "fluids")
    for name in names:
        # Native storage narrows values to float32 before the final wide merge.
        pd.testing.assert_series_equal(native.set_index("charttime")[name].dropna(),
                                       expected[name].astype("float32"),
                                       check_names=False, check_dtype=False, atol=1e-9, rtol=0)
