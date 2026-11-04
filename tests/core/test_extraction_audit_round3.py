"""Extraction audit round 3 (2026-11-04) regression tests."""

import types

import pandas as pd
import pytest


def test_r3_03_cancelled_eicu_orders_are_dropped_in_either_spelling():
    from easyicu.concept.callback_apply import _drop_excluded_statuses

    source = types.SimpleNamespace(
        params={"excluded_statuses": ["true"], "status_var": "drugordercancelled"}
    )
    raw_csv = pd.DataFrame({"drugordercancelled": ["Yes", "No", "yes", None], "dose": [1, 2, 3, 4]})
    converted = pd.DataFrame({"drugordercancelled": [True, False, True, None], "dose": [1, 2, 3, 4]})
    assert _drop_excluded_statuses(raw_csv, source)["dose"].tolist() == [2, 4]
    assert _drop_excluded_statuses(converted, source)["dose"].tolist() == [2, 4]


def _vaso_tables(rate_hours, rate_values, dur_start, dur_hours):
    from easyicu.table import ICUTable

    rate = ICUTable(
        pd.DataFrame({"patientid": [1] * len(rate_hours), "givenat": rate_hours, "norepi_rate": rate_values}),
        id_columns=["patientid"], index_column="givenat", value_column="norepi_rate",
    )
    dur = ICUTable(
        pd.DataFrame({"patientid": [1] * len(dur_start), "givenat": dur_start, "norepi_dur": dur_hours}),
        id_columns=["patientid"], index_column="givenat", value_column="norepi_dur",
    )
    return {"norepi_rate": rate, "norepi_dur": dur}


def _vaso60(tables, database="hirid"):
    from easyicu.concept.callbacks import ConceptCallbackContext, _callback_vaso60

    config = types.SimpleNamespace(name=database)
    ctx = ConceptCallbackContext(
        concept_name="norepi60", target=None, interval=pd.Timedelta(hours=1),
        resolver=None, data_source=types.SimpleNamespace(config=config), patient_ids=None,
    )
    return _callback_vaso60(tables, ctx).data


def test_r3_02_first_hour_of_an_episode_keeps_its_rate():
    # Hourly rates (HiRID/AUMC) sit on the floored hour; the episode starts at
    # 10.4 h. The 10 h rate used to fall before the window start.
    out = _vaso60(_vaso_tables([10.0, 11.0, 12.0], [0.1, 0.2, 0.3], [10.4], [2.5]))
    assert out["givenat"].tolist() == [10.0, 11.0, 12.0]
    assert out["norepi60"].tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_r3_04_duration_unit_is_never_inferred_from_magnitude():
    # One 8,784 h order span must not turn every duration into minutes.
    tables = _vaso_tables([0.0, 1.0, 2.0], [0.1, 0.1, 0.1], [0.0, 100.0], [3.0, 8784.0])
    out = _vaso60(tables, database="eicu")
    assert out["givenat"].tolist() == [0.0, 1.0, 2.0]


def test_r3_19_hirid_zero_doses_neither_rate_nor_extend_exposure():
    from easyicu.utils.callback_utils import hirid_duration, hirid_rate_kg

    frame = pd.DataFrame({
        "patientid": [1, 1, 1, 1, 2, 2],
        "infusionid": [7, 7, 7, 7, 8, 8],
        # start marker, two doses, a trailing zero record; stay 2 never doses
        "givenat": [10.5, 11.5, 12.5, 15.5, 3.0, 4.0],
        "givendose": [0.0, 30.0, 30.0, 0.0, 0.0, 0.0],
        "doseunit": ["µg"] * 6,
        "weight": [70.0] * 6,
    })
    duration = hirid_duration(frame, concept_name="norepi_dur", val_col="givendose",
                              index_col="givenat", grp_var="infusionid")
    assert duration["patientid"].tolist() == [1]
    # R5C-HIRID-003: the exact episode start is distinct from its hourly bin.
    assert duration["givenat"].tolist() == [10.5]
    assert duration["norepi_dur"].tolist() == [2.0]
    rate = hirid_rate_kg(frame, concept_name="norepi_rate", val_col="givendose", unit_col="doseunit",
                         grp_var="infusionid", index_col="givenat", interval_minutes=60.0)
    assert (rate["norepi_rate"] > 0).all()
    # Each dose is split over the hours since the previous record, so the
    # duration's first hour (10) has a rate (2026-11-04 audit round 4, R4-028).
    assert sorted(rate["givenat"].tolist()) == [10, 11, 12]


def test_r3_04_eicu_order_exposure_ends_at_unit_discharge():
    from easyicu.concept.eicu_vasopressor_orders import order_exposure_durations

    orders = pd.DataFrame({
        "patientunitstayid": [1, 1, 2],
        "drugstartoffset": [60.0, 600.0, 0.0],
        "drugstopoffset": [527_040.0, 900.0, 120.0],
    })
    out = order_exposure_durations(
        orders, concept_name="norepi_dur", infused_stays=(), discharge_offsets={1: 300.0, 2: 1_000.0}
    )
    # stay 1: the 8,784 h order stops at discharge (4 h); the order starting
    # after discharge is dropped. Stay 2 is unchanged.
    assert out["patientunitstayid"].tolist() == [1, 2]
    assert out["norepi_dur"].tolist() == pytest.approx([4.0, 2.0])


@pytest.mark.parametrize("first_hour", [-6.0, 10.0])
def test_r3_09_urine24_grid_ends_at_the_last_urine_record(first_hour):
    from easyicu.concept.callbacks import _urine24_batch

    hours = [first_hour + h for h in range(20)]
    frame = pd.DataFrame({"stay": [1] * 20, "t": hours, "urine": [50.0] * 20})
    out = _urine24_batch(frame, "stay", "t", "urine", time_step=1.0, window_steps=24,
                         min_steps=12, cap_steps=12)
    # ricu ended the grid at the duration (19 h): a late first record lost
    # its tail, an early one padded zero urine past the last record.
    assert out["t"].min() == first_hour
    assert out["t"].max() == hours[-1]


def test_r3_17_driving_pressure_matches_peep_within_one_hour_of_relative_time():
    from easyicu.callbacks import driving_pressure

    plateau = pd.DataFrame({"stay_id": [1, 1], "charttime": [10.0, 20.0], "plateau_pres": [25.0, 25.0]})
    peep = pd.DataFrame({"stay_id": [1, 1], "charttime": [9.5, 50.0], "peep": [8.0, 8.0]})
    out = driving_pressure(plateau, peep, match_win=pd.Timedelta(hours=1))
    # The 20 h plateau has no PEEP within 1 h (the nearest is 30 h away).
    assert out["charttime"].tolist() == [10.0]
    assert out["driving_pres"].tolist() == [17.0]


def test_r3_18_derived_plateau_time_column_matches_item_level_peep_and_pip():
    from easyicu.callbacks import driving_pressure

    # The derived plateau_pres carries the merged 'charttime'; AUMC PEEP and
    # PIP keep 'measuredat_minutes' (native-v2 ventilator module failed on it).
    plateau = pd.DataFrame({"admissionid": [1, 1], "charttime": [10.0, 20.0], "plateau_pres": [25.0, 40.0]})
    peep = pd.DataFrame({"admissionid": [1, 1], "measuredat_minutes": [10.0, 20.0], "peep": [8.0, 8.0]})
    pip = pd.DataFrame({"admissionid": [1, 1], "measuredat_minutes": [10.0, 20.0], "pip": [30.0, 30.0]})
    out = driving_pressure(plateau, peep, match_win=pd.Timedelta(hours=1), pip=pip)
    # The 20 h plateau is above PIP + 1 and gives no driving pressure.
    assert out["charttime"].tolist() == [10.0]
    assert out["driving_pres"].tolist() == [17.0]


def test_r3_08_r4_018_mimic_metavision_one_year_mortality_is_defined(monkeypatch) -> None:
    from easyicu.scores import outcomes

    tables = {
        "icustays": pd.DataFrame({
            "SUBJECT_ID": [10, 20, 30, 40], "HADM_ID": [100, 200, 300, 400],
            "ICUSTAY_ID": [1, 2, 3, 4],
            "INTIME": ["2150-01-01 08:00"] * 4, "OUTTIME": ["2150-01-03 08:00"] * 4,
            "LOS": [2.0] * 4,
            "DBSOURCE": ["metavision", "metavision", "carevue", "carevue"],
        }),
        # A MetaVision survivor and a MetaVision death on day 40; the same for CareVue.
        "patients": pd.DataFrame({
            "SUBJECT_ID": [10, 20, 30, 40], "DOD": [None, "2150-02-10", None, "2150-02-10"],
        }),
        "admissions": pd.DataFrame({
            "HADM_ID": [100, 200, 300, 400], "DEATHTIME": [None] * 4,
            "DISCHTIME": ["2150-01-11 08:00"] * 4,
        }),
    }
    monkeypatch.setattr(outcomes, "_raw_table", lambda _db, _path, table: tables[table])

    result = outcomes.load_outcomes("mimic").set_index("icustay_id")

    # Round 4 (R4-018) reverted R3-08 for MIMIC-III: MetaVision deaths are
    # linked well beyond 90 days after discharge (out-of-hospital deaths up to
    # 1,734 days), so one-year mortality is defined as for CareVue.
    assert bool(result.loc[1, "mort_365d"]) is False
    assert bool(result.loc[2, "mort_365d"]) is True
    assert result.loc[2, "followup_days_365d"] == pytest.approx(40.0)  # date-level DOD
    assert bool(result.loc[1, "mort_90d"]) is False
    assert bool(result.loc[2, "mort_90d"]) is True
    # CareVue's four-year linkage reaches one year for both.
    assert bool(result.loc[3, "mort_365d"]) is False
    assert bool(result.loc[4, "mort_365d"]) is True


def _death_src(callback, index_var, value_var):
    return types.SimpleNamespace(
        callback=callback, index_var=index_var, value_var=value_var,
        sub_var=None, unit_var=None, table=None, ids=None,
    )


def test_r3_13_aumc_registry_death_is_never_before_icu_discharge() -> None:
    from easyicu.concept.callback_apply import _apply_callback
    from easyicu.scores import outcomes

    hour, day = 60, 24 * 60
    # Loader minutes: a ward death on the calendar day of discharge (dated 10 h
    # before it), a registry date 30 days before admission, and an ICU death.
    frame = pd.DataFrame({
        "admissionid": [1, 2, 3],
        "dateofdeath": [3 * day - 10 * hour, -30 * day, -2 * day],
        "death": [3 * day, 3 * day, 3 * day],
        "destination": ["Verpleegafdeling", "Huis", "Overleden"],
    })
    out = _apply_callback(frame, _death_src("aumc_death", "dateofdeath", "dischargedat"), "death").set_index("admissionid")
    assert out.loc[1, "death"] is True and out.loc[1, "dateofdeath"] == 3 * day
    assert out.loc[2, "death"] is False and out.loc[2, "dateofdeath"] == 3 * day
    assert out.loc[3, "death"] is True and out.loc[3, "dateofdeath"] == 3 * day

    ms = 86_400_000
    admissions = pd.DataFrame({
        "admissionid": [1, 2], "admittedat": [0, 0], "dischargedat": [3 * ms, 3 * ms],
        "destination": ["Verpleegafdeling", "Huis"],
        "dateofdeath": [3 * ms - 10 * 3_600_000, -30 * ms],
    })
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(outcomes, "_raw_table", lambda *_: admissions)
        days = outcomes._aumc_stay_death_days("aumc", None).set_index("_stay")
        result = outcomes.load_outcomes("aumc").set_index("admissionid")
    finally:
        mp.undo()
    assert days.loc[1, "days_to_death"] == 3.0
    assert pd.isna(days.loc[2, "days_to_death"])
    assert bool(result.loc[1, "mort_28d"]) is True
    assert bool(result.loc[2, "mort_28d"]) is False


def test_r3_15_sic_recorded_death_without_offset_is_a_death_at_icu_discharge(monkeypatch) -> None:
    from easyicu.concept.callback_apply import _apply_callback
    from easyicu.scores import outcomes

    cases = pd.DataFrame({
        "CaseID": [1, 2, 3, 4],
        # Deceased in hospital, deceased at ICU discharge, a survivor, and a
        # death recorded half an hour before the ICU transfer.
        "OffsetOfDeath": [None, None, None, 1_800],
        "HospitalDischargeType": [2028, 2026, 2026, 2028],
        "DischargeState": [2202, 2215, 2202, 2202],
        "ICUOffset": [3_600] * 4,
        "TimeOfStay": [3_600 + 5 * 3_600] * 4,
        "EstimatedSurvivalObservationTime": [3077] * 4,
    })
    frame = cases.rename(columns={"OffsetOfDeath": "death"})
    out = _apply_callback(frame, _death_src("sic_death", "OffsetOfDeath", "OffsetOfDeath"), "death").set_index("CaseID")
    assert out["death"].to_dict() == {1: True, 2: True, 3: False, 4: True}
    assert out["charttime"].to_dict() == {1: 5.0, 2: 5.0, 3: 5.0, 4: 0.0}

    monkeypatch.setattr(outcomes, "_raw_table", lambda *_: cases)
    result = outcomes.load_outcomes("sic").set_index("CaseID")
    assert [bool(result.loc[i, "mort_28d"]) for i in (1, 2, 3, 4)] == [True, True, False, True]


def test_r3_05_eicu_ventilation_evidence_becomes_windows_across_gaps_of_up_to_8h() -> None:
    from easyicu.concept.eicu_ventilation_windows import bridge_hours, care_intervals, setting_evidence

    charting = pd.DataFrame({
        "patientunitstayid": 1, "respchartoffset": [60, 60, 90, 90, 90, 120, 150, 180],
        "respchartvaluelabel": ["PEEP", "PEEP", "Tidal Volume (set)", "LPM O2", "RR (patient)", "B1: IPAP", "RT Vent On/Off", "RT Vent On/Off"],
        "respchartvalue": ["5", "0", "450", "4", "18", "12", "Continued", "Off"],
    })
    # PEEP is shared with NIV; it takes the invasive level of the nearby set
    # tidal volume (2026-11-04 audit round 4, R4-024).
    kept = setting_evidence(charting, label_column="respchartvaluelabel", value_column="respchartvalue",
                            id_column="patientunitstayid", time_column="respchartoffset")
    assert kept["respchartvalue"].tolist() == ["invasive", "invasive", "noninvasive", "invasive"]

    care = pd.DataFrame({
        "patientunitstayid": [1, 1, 2],
        "ventstartoffset": [0, -120, 600],
        "ventendoffset": [0, 150, 0],
        "priorventstartoffset": [0, 0, 0],
        "priorventendoffset": [0, 0, 0],
    })
    intervals = care_intervals(care, {1: 120})
    # The 0/0 placeholder and the start-only row carry no interval; a start
    # before admission is left-censored and the end clipped to discharge.
    assert intervals.to_dict("records") == [{"patientunitstayid": 1, "start_h": 0.0, "end_h": 2.0}]

    points = pd.DataFrame({"patientunitstayid": [1, 1, 1, 2], "hour": [10.0, 18.0, 30.0, 5.0]})
    hours = bridge_hours(points, intervals)
    by_stay = hours.groupby("patientunitstayid")["hour"].apply(list).to_dict()
    # 2 -> 10 and 10 -> 18 are gaps of 8 h; 18 -> 30 is not.
    assert by_stay[1] == list(range(0, 19)) + [30]
    assert by_stay[2] == [5]


def test_ricu_sweep_unmatched_vent_start_lasts_match_win_not_start_plus_match_win() -> None:
    from easyicu.concept.callbacks import ConceptCallbackContext, _callback_vent_ind
    from easyicu.table import ICUTable

    start = ICUTable(
        pd.DataFrame({"stay_id": [1], "charttime": [40.0], "vent_start": [True]}),
        id_columns=["stay_id"], index_column="charttime", value_column="vent_start",
    )
    ctx = ConceptCallbackContext(
        concept_name="vent_ind", target=None, interval=pd.Timedelta(hours=1),
        resolver=None, data_source=types.SimpleNamespace(config=types.SimpleNamespace(name="miiv")),
        patient_ids=None,
    )
    out = _callback_vent_ind({"vent_start": start}, ctx).data
    assert sorted(out["charttime"].tolist()) == [40.0 + h for h in range(7)]


def test_ricu_sweep_a_charted_rate_runs_at_most_max_len() -> None:
    from easyicu.utils.callback_utils import create_intervals

    data = pd.DataFrame({"icustay_id": [1, 1], "charttime": [0.0, 48.0], "rate": [5.0, 5.0]})
    out = create_intervals(data, by_cols=["icustay_id"], max_len=pd.Timedelta(hours=6))
    # 6 h cap minus the 1 h interval; the last chart gets the 1 h overhang minus 1 h.
    assert out["endtime"].tolist() == [5.0, 48.0]


def test_section4_wintbl_expansion_reads_vent_ind_durations_in_hours() -> None:
    from easyicu.concept.callbacks import _expand_wintbl_to_hourly

    windows = pd.DataFrame({"CaseID": [1], "starttime": [2.0], "dur_var": [3.0]})
    out = _expand_wintbl_to_hourly(windows, ["CaseID"], "starttime", "dur_var", "charttime")
    assert out["charttime"].tolist() == [2.0, 3.0, 4.0, 5.0]
