"""Extraction audit round 3 (2026-11-04): F-item and M-item regression tests."""

import json
import re
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import ConceptSource, _apply_callback

DATA = Path(__file__).resolve().parents[2] / "src" / "easyicu" / "data"


def _dictionary(name="concept-dict.json"):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def _ids(source):
    ids = source.get("ids")
    return set(ids if isinstance(ids, list) else [ids])


class _Tables:
    """A data source whose load_table serves fixed frames by table name."""

    def __init__(self, **tables):
        self.tables = tables
        self.loads = []

    def load_table(self, name, columns=None, filters=None, verbose=False):
        self.loads.append((name, filters))
        frame = self.tables[name]
        for spec in filters or []:
            frame = frame.loc[frame[spec.column].isin(spec.value)]
        return frame


# ── fluid_balance subtracts non-urine output ────────────────────────────


def _fluid_ctx():
    from easyicu.concept.callbacks import ConceptCallbackContext

    return ConceptCallbackContext(
        concept_name="fluid_balance", target=None, interval=pd.Timedelta(hours=1),
        resolver=None, data_source=None, patient_ids=None,
    )


def _hourly(name, rows):
    from easyicu.table import ICUTable

    frame = pd.DataFrame(rows, columns=["stay_id", "charttime", name])
    return ICUTable(frame, id_columns=["stay_id"], index_column="charttime", value_column=name)


def test_fluid_balance_subtracts_drain_gastric_and_stool_output():
    from easyicu.concept.callbacks import _callback_fluid_balance_hourly

    tables = {
        "total_input_ml": _hourly("total_input_ml", [(1, 0, 500.0), (1, 1, 100.0)]),
        "urine": _hourly("urine", [(1, 0, 200.0), (1, 2, 50.0)]),
        # Stay 2 has no recorded intake and gets no balance at all.
        "other_output_ml": _hourly("other_output_ml", [(1, 0, 150.0), (1, 3, 80.0), (2, 0, 40.0)]),
    }
    out = _callback_fluid_balance_hourly(tables, _fluid_ctx()).data
    assert out["stay_id"].tolist() == [1, 1, 1, 1]
    assert out["charttime"].tolist() == [0, 1, 2, 3]
    assert out["fluid_balance"].tolist() == [150.0, 100.0, -50.0, -80.0]


def test_fluid_balance_without_non_urine_sources_subtracts_urine_only():
    from easyicu.concept.callbacks import _callback_fluid_balance_hourly

    tables = {
        "total_input_ml": _hourly("total_input_ml", [(1, 0, 500.0)]),
        "urine": _hourly("urine", [(1, 0, 200.0)]),
    }
    out = _callback_fluid_balance_hourly(tables, _fluid_ctx()).data
    assert out["fluid_balance"].tolist() == [300.0]


def test_other_output_is_registered_for_every_input_minus_urine_database():
    d = _dictionary()
    balance = d["fluid_balance"]
    assert "other_output_ml" in balance["concepts"] and "other_output_ml" in balance["depends_on"]
    other = d["other_output_ml"]["sources"]
    assert set(other) == set(d["urine"]["sources"]) - {"hirid"}
    for db, sources in other.items():
        if db.startswith("eicu"):
            continue
        urine = set().union(*(_ids(s) for s in d["urine"]["sources"][db]))
        mine = set().union(*(_ids(s) for s in sources))
        assert not mine & urine, db
    # RRT removal, tube-feed residual and OR/PACU/pre-admission output stay out.
    for item in (226626, 226627, 226631, 226633, 227510, 227511, 40286, 43176, 40060):
        assert item not in set().union(*(_ids(s) for s in other["mimic"]))
    assert not {8805, 8806, 8808} & _ids(other["aumc"][0])
    assert _ids(other["sic"][0]) == {2322, 726}


def test_eicu_other_output_drops_urine_counts_totals_and_rrt_labels():
    from easyicu.concept.callback_apply import _drop_declared_row_exclusions

    source_spec = _dictionary()["other_output_ml"]["sources"]["eicu"][0]
    source = types.SimpleNamespace(params={"exclude_regex": source_spec["exclude_regex"]})
    prefix = "flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|"
    kept = ["Chest Tube A Amount", "Gastric (NG)", "Ventric #1", "Paracentesis Output", "Ileostomy"]
    # The plain 'Stool' label is an occurrence count (audit round 4, R4-003).
    dropped = ["Stool", "Urine", "Urine Count", "Stool Occurrence", "TOTAL OUT", "ChestTube Total", "CRRT Out",
               "Actual Patient Fluid Removal", "Hemofiltration", "CBI/Bladder Irr Out",
               "Tube Feeding Residual Discarded (mL)", "OR EBL", "Incontinent", "Effluent"]
    frame = pd.DataFrame({"cellpath": [prefix + label for label in kept + dropped]})
    out = _drop_declared_row_exclusions(frame, source)
    assert [path[len(prefix):] for path in out["cellpath"]] == kept
    assert re.match(source_spec["regex"], prefix + "Ileostomy")


# ── HiRID rates and doses ───────────────────────────────────────────────


def test_hirid_mass_rate_sums_concurrent_syringes_and_drops_zero_doses():
    from easyicu.utils.callback_utils import hirid_rate_mass

    frame = pd.DataFrame({
        "patientid": [1, 1, 1, 1],
        "datetime": [5.1, 5.6, 5.9, 6.2],
        "infusionid": [10, 11, 11, 11],
        "givendose": [60.0, 30.0, 0.0, 0.0],
        "doseunit": ["µg"] * 4,
        "recordstatus": 0,
    })
    out = hirid_rate_mass(
        frame, concept_name="fentanyl_rate", val_col="givendose", unit_col="doseunit",
        grp_var="infusionid", index_col="datetime", target_unit="mcg/hour",
    )
    # 90 µg given over hour 5 by two syringes; the zero-dose stop rows are no hour.
    assert out["datetime"].tolist() == [5]
    assert out["fentanyl_rate"].tolist() == pytest.approx([90.0])


def test_furosemide_equivalent_drops_zero_amounts():
    from easyicu.concept.diuretics import furosemide_equivalent

    frame = pd.DataFrame({"givendose": [0.0, 20.0], "doseunit": ["mg", "mg"]})
    out = furosemide_equivalent(
        frame, concept_name="fe", value_var="givendose", unit_var="doseunit",
        params={"fe_drug": "furosemide", "fe_route": "iv"},
    )
    assert out["fe"].tolist() == [20.0]


def test_hirid_pharma_boolean_sources_require_a_positive_dose():
    for name in ("concept-dict.json", "sofa2-dict.json"):
        for concept, spec in _dictionary(name).items():
            for source in (spec.get("sources") or {}).get("hirid", []):
                if source.get("table") == "pharma":
                    assert source.get("callback") != "transform_fun(set_val(TRUE))", (name, concept)


def test_hirid_output_counter_dip_that_rebounds_does_not_restart_the_day():
    from easyicu.concept.hirid_fluid_counters import _restart_aware_increments

    ids = np.array([1, 1, 1, 1])
    days = np.array([0, 0, 0, 0])
    values = np.array([400.0, 100.0, 500.0, 700.0])
    assert _restart_aware_increments(ids, days, values).tolist() == [400.0, 0.0, 100.0, 200.0]
    # A real restart (no rebound) still counts the new block from zero.
    values = np.array([400.0, 100.0, 150.0, 300.0])
    assert _restart_aware_increments(ids, days, values).tolist() == [400.0, 100.0, 50.0, 150.0]


# ── ventilation windows ─────────────────────────────────────────────────


def test_mech_vent_window_without_a_level_is_not_ventilation():
    from easyicu.concept.callbacks import _leveled_window_mask

    windows = pd.DataFrame({"mech_vent": ["invasive", None, "invasive"]})
    assert _leveled_window_mask(windows, "mech_vent").tolist() == [True, False, True]
    # Each row is decided on its own level (audit round 4, R4-066): a batch
    # whose windows all lack a level is not ventilation either.
    unleveled = pd.DataFrame({"mech_vent": [None, None]})
    assert _leveled_window_mask(unleveled, "mech_vent").tolist() == [False, False]


def test_sic_sedation_and_end_tidal_co2_attribute_controlled_settings():
    from easyicu.concept.sicdb_ventilation import HOUR, _sedated_controlled_hours

    settings = pd.DataFrame({
        "CaseID": [1, 1, 1, 2],
        "DataID": [2282, 2282, 2282, 2282],
        "Offset": [0, HOUR, 2 * HOUR, 0],
    })
    source = _Tables(
        cases=pd.DataFrame({"CaseID": [1, 2], "HospitalUnit": [3, 5], "ICUOffset": [0, 0]}),
        data_float_h=pd.DataFrame({
            "CaseID": [1, 1, 1, 1, 2],
            "DataID": [3059, 716, 716, 3058, 3057],
            "Offset": [0, HOUR, 2 * HOUR, 2 * HOUR, 0],
            "Val": [0.1, 35.0, 4.0, 1.0, 2.0],
        }),
    )
    out = _sedated_controlled_hours(settings, source, [1, 2], "CaseID", "Offset")
    # R5-SIC-015: opioid-only hour 0 is sensitivity; etCO2 35 supports hour 1.
    # etCO2 4 does not, and the NIV ward (unit 5) never does.
    assert sorted(zip(out["CaseID"], out["_h"])) == [(1, 1.0)]


# ── vasopressor indicator ───────────────────────────────────────────────


def test_vaso_ind_covers_the_hour_in_which_an_episode_ends():
    from easyicu.concept.callbacks import ConceptCallbackContext, _callback_vaso_ind
    from easyicu.table import ICUTable

    dur = ICUTable(
        pd.DataFrame({"stay_id": [1], "starttime": [10.8], "norepi_dur": [1.4]}),
        id_columns=["stay_id"], index_column="starttime", value_column="norepi_dur",
    )
    ctx = ConceptCallbackContext(
        concept_name="vaso_ind", target=None, interval=None,
        resolver=None, data_source=None, patient_ids=None,
    )
    out = _callback_vaso_ind({"norepi_dur": dur}, ctx).data
    # 10.8 h + 1.4 h ends at 12.2 h: hours 10, 11 and 12.
    assert sorted(np.floor(pd.to_numeric(out["starttime"])).astype(int).tolist()) == [10, 11, 12]


# ── value callbacks ─────────────────────────────────────────────────────


def test_sic_exclude_admission_years_drops_the_rows_of_those_cases():
    source = _Tables(cases=pd.DataFrame({"CaseID": [1, 2], "AdmissionYear": [2013, 2014]}))
    frame = pd.DataFrame({"CaseID": [1, 2, 2], "Val": [7900.0, 480.0, 510.0]})
    out = _apply_callback(
        frame, ConceptSource(callback="sic_exclude_admission_years(2013)", value_var="Val"),
        concept_name="tidal_vol", data_source=source,
    )
    assert out["Val"].tolist() == [480.0, 510.0]


def test_scale_repair_callbacks_split_at_their_threshold():
    frame = pd.DataFrame({"value": [0.45, 3.0, 450.0]})
    out = _apply_callback(
        frame, ConceptSource(callback="transform_fun(multiply_below(3, 1000))", value_var="value"),
        concept_name="tidal_vol",
    )
    assert out["value"].tolist() == pytest.approx([450.0, 3000.0, 450.0])
    frame = pd.DataFrame({"value": [1.1, 4.6, 0.2]})
    out = _apply_callback(
        frame, ConceptSource(callback="transform_fun(divide_above(2, 4))", value_var="value"),
        concept_name="cai",
    )
    assert out["value"].tolist() == pytest.approx([1.1, 1.15, 0.2])


def test_sum_at_same_time_counts_a_re_entered_component_once():
    frame = pd.DataFrame({
        "patientid": [1, 1, 1, 1],
        "datetime": [3.0, 3.0, 3.0, 3.0],
        "variableid": [24000550, 24000550, 24000557, 24000557],
        "neut": [80.0, 80.0, 5.0, 5.0],
    })
    out = _apply_callback(
        frame, ConceptSource(callback="sum_at_same_time", sub_var="variableid", index_var="datetime"),
        concept_name="neut",
    )
    assert out["neut"].tolist() == [85.0]


# ── row filters, bounds and mappings (full-dictionary M rows) ───────────


def test_require_regex_keeps_only_matching_rows_and_drops_empty_values():
    from easyicu.concept.callback_apply import _drop_declared_row_exclusions

    source = types.SimpleNamespace(params={"require_regex": {"value": "(?i)^\\s*(extubated|self extubation)\\s*$"}})
    frame = pd.DataFrame({"value": ["Extubated", "Fluid Bolus", None, "self extubation", "Intubated"]})
    assert _drop_declared_row_exclusions(frame, source)["value"].tolist() == ["Extubated", "self extubation"]
    with pytest.raises(KeyError):
        _drop_declared_row_exclusions(pd.DataFrame({"other": [1]}), source)


def test_eicu_differential_keeps_percent_rows_only():
    from easyicu.concept.callback_apply import _drop_declared_row_exclusions

    spec = next(s for s in _dictionary()["lymph"]["sources"]["eicu"] if s["table"] == "lab")
    source = types.SimpleNamespace(params={"require_regex": spec["require_regex"]})
    frame = pd.DataFrame({"labmeasurenameinterface": ["%", "% WBC", "1000/uL", None, "K/uL"], "labresult": [12, 11, 1.3, 1.1, 1.2]})
    assert _drop_declared_row_exclusions(frame, source)["labresult"].tolist() == [12, 11]


def test_vaso_ind_loads_its_durations_on_a_minute_grid():
    assert _dictionary()["vaso_ind"]["interval"] == "00:01:00"


def test_pressures_at_or_below_zero_are_not_measurements():
    d = _dictionary()
    for concept in ("map", "map_invasive", "map_nibp", "sbp", "dbp", "plateau_pres", "plateau_pres_raw"):
        assert d[concept]["min"] == 1, concept


def test_full_dictionary_m_mappings():
    d, s2 = _dictionary(), _dictionary("sofa2-dict.json")

    def ids(concept, db, dictionary=d):
        return set().union(*(_ids(s) for s in dictionary[concept]["sources"][db]))

    # MIMIC-III: unspecified-specimen gases, monitor SaO2 and CRRT circuit additives are out.
    assert not {3785, 3837} & ids("po2", "mimic") and not {3784, 3835} & ids("pco2", "mimic")
    assert 834 not in ids("sao2", "mimic") and 834 in ids("o2sat", "mimic")
    assert 227525 not in ids("calcium_iv", "mimic") and 227536 not in ids("potassium_iv", "mimic")
    assert 225165 not in ids("bicarbonate", "mimic")
    assert {223771} <= ids("pawp", "mimic") and {227549, 226541} <= ids("scvo2", "mimic")
    extubation = next(s for s in d["vent_end"]["sources"]["mimic"] if _ids(s) == {640})
    assert extubation["require_regex"]["value"]
    for concept, dictionary in (("adv_resp", s2), ("vent_mode", d), ("vent_breath_seq", d)):
        for db in ("mimic", "miiv"):
            for source in dictionary[concept]["sources"].get(db, []):
                if 223849 in _ids(source):
                    assert re.search(source["exclude_regex"]["value"], "Standby"), (concept, db)
    for source in d["dex"]["sources"]["mimic"]:
        if source["table"] == "inputevents_mv":
            assert source["excluded_statuses"] == ["Rewritten"]
    # MIMIC-IV: aerosol masks are not high-flow oxygen; non-systemic antibiotics are not exposure.
    hfnc = next(s for s in s2["adv_resp"]["sources"]["miiv"] if s.get("regex"))
    assert not re.search(hfnc["regex"], "High flow neb") and re.search(hfnc["regex"], "High flow nasal cannula")
    abx = next(s for s in d["abx"]["sources"]["miiv"] if s["table"] == "prescriptions")
    assert re.search(abx["exclude_regex"]["drug"], "Erythromycin Ophth Oint")
    assert not re.search(abx["exclude_regex"]["drug"], "Vancomycin")
    # eICU: per-item GCS forms, base deficit, oral formulations of IV drugs.
    assert "Eye Opening" in ids("egcs", "eicu") and "Score (Glasgow Coma Scale)" in ids("tgcs", "eicu")
    deficit = next(s for s in d["be"]["sources"]["eicu"] if _ids(s) == {"Base Deficit"})
    assert deficit["callback"] == "transform_fun(binary_op(`*`, -1))"
    amio = next(s for s in d["amiodarone"]["sources"]["eicu"] if s.get("sub_var") == "drugname" and s["table"] == "medication")
    assert re.search(amio["exclude_regex"]["drugname"], "AMIODARONE HCL 200 MG PO TABS")
    assert re.search(amio["exclude_regex"]["routeadmin"], "PO")
    # AUMC and SICdb.
    assert 8873 not in ids("vent_rate", "aumc") and 20955 in ids("co", "aumc")
    assert not {9204, 13024, 19663, 19664} & ids("samp", "aumc")
    assert 2282 not in ids("resp", "sic") and 231 in ids("neut", "sic")
    assert {1440, 1432} <= ids("delirium_tx_proxy", "sic", s2)


def test_delirium_positive_is_a_positive_overall_cam_icu_result_only():
    s2 = _dictionary("sofa2-dict.json")
    for db in ("mimic", "miiv"):
        sources = s2["delirium_positive"]["sources"][db]
        assert [_ids(s) for s in sources] == [{228332}]
        assert sources[0]["regex"] == "^Positive$" and sources[0]["val_var"] == "value"


def test_aumc_rates_come_from_running_infusions_only():
    from easyicu.concept.callback_apply import _drop_declared_row_exclusions

    spec = _dictionary()["fentanyl_rate"]["sources"]["aumc"][0]
    source = types.SimpleNamespace(params={"require_regex": spec["require_regex"], "exclude_regex": spec["exclude_regex"]})
    frame = pd.DataFrame({
        "iscontinuous": [1, 1, 0, 1],
        "action": ["Nieuwe spuit", "Flush", "Nieuwe toediening", "Snelheid veranderd"],
        "rate": [2.0, 60.0, 0.0, 1.5],
    })
    assert _drop_declared_row_exclusions(frame, source)["rate"].tolist() == [2.0, 1.5]


def test_a_filter_on_the_value_column_reads_the_renamed_concept_column():
    from easyicu.concept.callback_apply import _drop_declared_row_exclusions

    source = types.SimpleNamespace(value_var="value", params={"exclude_regex": {"value": r"(?i)^\s*standby\s*$"}})
    frame = pd.DataFrame({"vent_mode": ["CMV/ASSIST", "Standby", "CPAP/PSV"]})
    kept = _drop_declared_row_exclusions(frame, source, "vent_mode")
    assert kept["vent_mode"].tolist() == ["CMV/ASSIST", "CPAP/PSV"]
    with pytest.raises(KeyError):
        _drop_declared_row_exclusions(frame, source)


def test_device_and_score_channels_missing_from_the_winning_dictionary():
    cd, s2 = _dictionary(), _dictionary("sofa2-dict.json")
    labels = {label for s in s2["motor_response"]["sources"]["eicu"] for label in _ids(s)}
    assert {"Motor", "Best Motor Response", "Motor Response"} <= labels
    for d in (cd, s2):
        windows = [s for s in d["mech_circ_support"]["sources"]["aumc"] if s.get("table") == "processitems"]
        assert [(_ids(s), s["dur_var"]) for s in windows] == [({10430, 9164}, "stop")]
    assert 8865 in {i for s in cd["ps"]["sources"]["aumc"] for i in _ids(s)}


def test_sicdb_continuous_furosemide_amount_is_shared_over_its_running_hours():
    from easyicu.concept.diuretics import furosemide_equivalent

    spec = next(s for s in _dictionary()["loop_diuretic_furosemide_eq"]["sources"]["sic"] if s["fe_drug"] == "furosemide")
    params = {k: v for k, v in spec.items() if k.startswith("fe_")}
    frame = pd.DataFrame({
        "CaseID": [1, 1],
        "Offset": [0, 10_000],
        "OffsetDrugEnd": [9_000, 10_060],
        "IsSingleDose": [0, 1],
        "loop_diuretic_furosemide_eq": [0.05, 0.02],  # g
    })
    # R5-SIC-009: even the zero-origin fixture declares its ICU anchor.
    out = furosemide_equivalent(frame, concept_name="loop_diuretic_furosemide_eq", value_var="Amount", unit_var=None, params=params,
                               data_source=_Tables(cases=pd.DataFrame({"CaseID": [1], "ICUOffset": [0]})))
    pieces = out.sort_values("Offset")
    # 50 mg over 2.5 h: 20, 20, 10 mg at 0, 3600 and 7200 s; the 20 mg single dose stays at its start.
    assert pieces["Offset"].tolist() == pytest.approx([0, 1, 2, 10_000/3600])
    assert pieces["loop_diuretic_furosemide_eq"].tolist() == pytest.approx([20.0, 20.0, 10.0, 20.0])
    assert out["loop_diuretic_furosemide_eq"].sum() == pytest.approx(70.0)


def test_eicu_end_tidal_co2_reads_the_monitor_column():
    sources = _dictionary()["etco2"]["sources"]["eicu"]
    assert {"table": "vitalperiodic", "val_var": "etco2", "class_name": "col_itm"}.items() <= next(
        s for s in sources if s.get("table") == "vitalperiodic"
    ).items()


def test_eicu_time_merge_keeps_offsets_beside_an_empty_source_charttime():
    from easyicu.concept import _merge_eicu_time_columns

    # The empty bucketed source arrived as 'charttime'; the monitor rows as observationoffset.
    combined = pd.DataFrame({
        "patientunitstayid": [1, 1],
        "charttime": [np.nan, np.nan],
        "observationoffset": [65.0, 130.0],
        "etco2": [30.0, 34.0],
    })
    merged, changed = _merge_eicu_time_columns(combined, ["observationoffset", "respchartoffset"])
    assert changed and list(merged.columns).count("charttime") == 1
    assert merged["charttime"].tolist() == [65.0, 130.0]


def test_miiv_admission_type_is_the_service_active_at_icu_intime():
    from easyicu.concept.service_at_admission import service_at_admission

    spec = _dictionary()["adm"]["sources"]["miiv"][0]
    assert (spec["class_name"], spec["callback"]) == ("fun_itm", "service_at_admission")
    stays = pd.DataFrame({
        "stay_id": [1, 2, 3],
        "hadm_id": [10, 10, 20],
        "intime": pd.to_datetime(["2150-01-05", "2150-01-20", "2150-03-01"]),
    })
    services = pd.DataFrame({
        "hadm_id": [10, 10, 20],
        # Chosen days before the first ICU stay; changed between the two stays;
        # recorded only after the third stay began.
        "transfertime": ["2150-01-01 08:00", "2150-01-12 09:00", "2150-03-01 06:00"],
        "curr_service": ["MED", "CSURG", "TRAUM"],
    })
    out = service_at_admission(stays, services, service_map=spec["service_map"], stay_column="stay_id", value_column="adm")
    assert out.set_index("stay_id")["adm"].to_dict() == {1: "med", 2: "surg", 3: "surg"}
