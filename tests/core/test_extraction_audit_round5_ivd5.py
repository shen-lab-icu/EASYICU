"""Round-5 MIIV/MIMIC-III regressions, using only invented events."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import ConceptDictionary, ConceptResolver, ConceptSource, _apply_callback
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu import resources
from easyicu.resources import load_dictionary
from tests.support.mimic_synthetic_source import mimic_synthetic_source as _synthetic_source, mimic_synthetic_load as _load

# Resolve fixtures beside the imported package, so copying this test for a
# read-only base run exercises that base's dictionary as well as its callbacks.
DATA = Path(resources.__file__).parent / "data"


def _dictionary(name="concept-dict.json"):
    return json.loads((DATA / name).read_text())


def _spec(concept, db, table, *, dictionary="concept-dict.json"):
    return next(s for s in _dictionary(dictionary)[concept]["sources"][db] if s["table"] == table)


def _apply(spec, frame, name, **kwargs):
    return _apply_callback(frame, ConceptSource.from_mapping(spec), name, **kwargs)


def _ecmo(db):
    spec = _spec("ecmo", db, "chartevents", dictionary="sofa2-dict.json")
    items = [228193, 224660, 224660, 228193, 224660, 228193, 224660]
    values = ["0", "0", "off", "1", "4.5", None, "not applicable"]
    frame = pd.DataFrame({"stay_id": range(1, 8), "charttime": 0., "itemid": items,
                          "ecmo": values, "value": values, "valuenum": pd.to_numeric(values, errors="coerce")})
    out = _apply(spec, frame, "ecmo")
    positive = out.loc[out.ecmo.eq(True), "stay_id"].tolist()
    assert positive == [4, 5]
    # A zero pump-flow observation must not claim that an established course ended.
    assert not out.loc[out.itemid.eq(224660) & out.stay_id.eq(2), "ecmo"].eq(False).any()
    assert _dictionary("sofa2-dict.json")["ecmo"]["aggregate"] == "max"
    from easyicu.scores.sofa2 import sofa2_resp
    active = pd.Series([i in positive for i in range(1, 8)])
    score = sofa2_resp(pafi=pd.Series([500.] * 7), ecmo=active)
    assert score.tolist() == [0, 0, 0, 4, 4, 0, 0]
    if db == "miiv":
        states = pd.DataFrame({"stay_id": range(11, 16), "charttime": 0.,
                               "itemid": [229268] * 4 + [229266], "value": ["VV", "VA", "VAV", "Unknown", "Femoral"],
                               "ecmo": ["VV", "VA", "VAV", "Unknown", "Femoral"]})
        assert _apply(spec, states, "ecmo").stay_id.tolist() == [11, 12, 13]
    # The default dictionary and the SOFA-2 overlay share positive semantics.
    assert _spec("ecmo", db, "chartevents")["callback"] == spec["callback"]


def test_r5c_iv_005_ecmo_needs_positive_evidence():
    _ecmo("miiv")


def test_r5c_iii_005_ecmo_needs_positive_evidence():
    _ecmo("mimic")


class _Items:
    def __init__(self, labels):
        self.labels = labels

    def load_table(self, table, **kwargs):
        assert table == "d_items"
        return pd.DataFrame({"itemid": list(self.labels), "label": list(self.labels.values())})


def _abx(db):
    import re
    from easyicu.scores.sepsis import susp_inf
    spec = _spec("abx", db, "prescriptions")
    antibacterial = ["Ceftriaxone", "Meropenem", "Imipenem-Cilastatin", "Ertapenem",
                     "Linezolid", "Daptomycin", "Colistin", "Cefazolin", "Vancomycin"]
    for drug in antibacterial:
        assert re.search(spec["regex"], drug, re.I), drug
    for drug in ["Fluconazole", "Acyclovir", "Ganciclovir", "Amphotericin B", "Caspofungin"]:
        assert not re.search(spec["regex"], drug, re.I), drug
    inputs = _spec("abx", db, "inputevents" if db == "miiv" else "inputevents_mv")
    assert inputs["antibacterial_regex"] == spec["regex"]
    ids = inputs["ids"][:4]
    labels = dict(zip(ids, ["Ceftriaxone", "Fluconazole", "Acyclovir", "Vancomycin"]))
    rows = pd.DataFrame({"stay_id": 1, "starttime": [0., 1., 2., 3.], "itemid": ids,
                         "abx": [100., 100., 100., 0.], "amount": [100., 100., 100., 0.],
                         "rate": [np.nan] * 4, "statusdescription": "FinishedRunning"})
    out = _apply(inputs, rows, "abx", data_source=_Items(labels))
    assert out.loc[out.abx.eq(True), "starttime"].tolist() == [0.]
    # Both drug channels feed the existing antibiotic/culture suspicion owner.
    antibiotic = pd.DataFrame({"stay_id": [1], "time": out.loc[out.abx.eq(True), "starttime"], "abx": True})
    culture = pd.DataFrame({"stay_id": [1], "time": [2.], "samp": True})
    result = susp_inf(antibiotic, culture, id_cols=["stay_id"], index_col="time")
    assert result["susp_inf"].tolist() == [True]
    assert result["time"].tolist() == [0.]


def test_r5c_iv_006_antibiotic_channels_are_antibacterial():
    _abx("miiv")


def test_r5c_iii_006_antibiotic_channels_are_antibacterial():
    _abx("mimic")


def test_r5c_iv_009_lab_priority_keeps_independent_lab_samples(tmp_path):
    from easyicu.concept.source_channel import select_preferred_channel, stamp_rank, validate_sources
    dictionary = load_dictionary()
    for name in ["crea", "hgb", "plt", "wbc", "ph", "sao2"]:
        definition = dictionary.get(name)
        sources = definition.sources["miiv"]
        assert validate_sources(definition, sources)
        frames = []
        for source in sources:
            is_lab = source.table == "labevents"
            assert source.params["source_channel"] == ("lab" if is_lab else "chart")
            rows = pd.DataFrame({"stay_id": [1, 1], "charttime": [0., 2.] if is_lab else [3., 12.],
                                 "storetime": [1., 4.] if is_lab else [5., 13.], name: [10., 10.]})
            frames.append(stamp_rank(rows, definition, source))
        result = select_preferred_channel(pd.concat(frames), value_column=name, id_columns=["stay_id"],
                                          time_column="charttime", window_hours=6).sort_values("charttime")
        assert result.charttime.tolist() == [0., 2., 12.]
        assert result.storetime.tolist() == [1., 4., 13.]


def test_r5c_iv_015_aspirin_combination_retains_aspirin_ingredient():
    spec = _spec("aspirin", "miiv", "prescriptions")
    frame = pd.DataFrame({"drug": ["Aspirin", "Butalbital-Aspirin-Caffeine", "Butalbital-Aspirin-Codeine"],
                          "aspirin": ["Aspirin", "Butalbital-Aspirin-Caffeine", "Butalbital-Aspirin-Codeine"]})
    assert _apply(spec, frame, "aspirin").aspirin.tolist() == [True] * 3


def test_r5c_iv_016_diuretic_dedup_requires_route_dose_and_drug_evidence():
    from easyicu.concept.emar_diuretics import drop_inputevent_duplicates
    t = pd.Timestamp("2150-02-01 10:00")
    doses = pd.DataFrame({"hadm_id": 1, "charttime": [t] * 5, "_route": ["oral", "iv", "iv", "iv", "iv"],
                          "_dose": [40., 40., 80., 40., 40.], "_unit": "mg",
                          "emar_id": list("abcde")})
    inputs = pd.DataFrame({"hadm_id": [1], "itemid": [221794], "starttime": [t], "amount": [40.], "amountuom": ["mg"],
                           "orderid": [123], "statusdescription": ["FinishedRunning"]})
    # A real input copy can suppress only one matching IV administration.
    kept = drop_inputevent_duplicates(doses, inputs, key="hadm_id", time_col="charttime", drug="furosemide")
    assert kept.index.tolist() == [0, 2, 3, 4]
    rewritten = inputs.assign(statusdescription="Rewritten")
    assert len(drop_inputevent_duplicates(doses, rewritten, key="hadm_id", time_col="charttime", drug="furosemide")) == 5
    # Another drug, route or mass cannot erase the dose.
    assert len(drop_inputevent_duplicates(doses, inputs.assign(itemid=229639), key="hadm_id", time_col="charttime", drug="furosemide")) == 5
    assert len(drop_inputevent_duplicates(doses.iloc[[0, 2]], inputs, key="hadm_id", time_col="charttime", drug="furosemide")) == 2


@pytest.mark.parametrize("index", [[20, 30, 40], [0, 0, 0]])
def test_r5c_iv_016_one_input_cannot_erase_repeated_emar_doses_with_duplicate_indices(index):
    from easyicu.concept.emar_diuretics import drop_inputevent_duplicates
    t = pd.Timestamp("2150-02-01 10:00")
    doses = pd.DataFrame({"hadm_id": [1, 1, 1], "charttime": t, "_route": "iv", "_dose": 40., "_unit": "mg",
                          "emar_id": ["", "   ", "known-event"]}, index=index)
    inputs = pd.DataFrame({"hadm_id": [1], "itemid": 221794, "starttime": t, "amount": 40., "amountuom": "mg"})
    out = drop_inputevent_duplicates(doses, inputs, key="hadm_id", time_col="charttime", drug="furosemide")
    assert out.index.tolist() == index[1:]
    assert out.emar_id.tolist() == ["   ", "known-event"]


def test_r5c_iii_009_inactive_modes_do_not_extend_ventilation():
    spec = _spec("mech_vent", "mimic", "chartevents")
    values = ["AC/VC", "AC/VC", "Standby", "Ambient", "Off", "AC/VC", "AC/VC"]
    frame = pd.DataFrame({"icustay_id": 1, "charttime": [0., 1., 6., 12., 18., 24., 25.], "itemid": 223849,
                          "mech_vent": values, "value": values, "error": 0})
    out = _apply(spec, frame, "mech_vent")
    assert out.charttime.tolist() == [0., 24.]
    # An inactive mode must not erase concurrent valid settings.
    valid = pd.DataFrame({"icustay_id": 1, "charttime": [0., 6.], "itemid": 220339,
                          "mech_vent": ["5", "5"], "value": ["5", "5"], "error": 0})
    with_settings = _apply(spec, pd.concat([frame, valid], ignore_index=True), "mech_vent")
    assert with_settings.loc[with_settings.charttime.eq(0), "dur_var"].item() == 360.


@pytest.mark.parametrize("database", ["mimic", "mimic_demo"])
def test_r5c_iii_009_vent_start_rejects_inactive_and_missing_mode_values(database):
    spec = _spec("vent_start", database, "chartevents")
    values = ["Standby", "Ambient", "Off", " stand by ", "Not Applicable", None, "", "AC/VC", "CPAP/PSV"]
    # The real loader renames the declared value column; never rely on a
    # second raw VALUE column being available to the row filter.
    frame = pd.DataFrame({"icustay_id": 1, "charttime": range(len(values)),
                          "itemid": 223849, "vent_start": values})
    assert spec.get("val_var") == "value"
    out = _apply(spec, frame, "vent_start")
    assert out.charttime.tolist() == [7, 8]


def test_r5c_iii_010_carevue_bipap_and_cpap_join_advanced_support():
    import re
    definition = _dictionary("sofa2-dict.json")["adv_resp"]
    sources = definition["sources"]["mimic"]
    device = next(s for s in sources if 467 in s.get("ids", []))
    for text in ["BiPAP", "bipap", "CPAP", "cpap mask"]:
        assert re.search(device["regex"], text)
    for text in ["Nasal cannula", "Aerosol-cool", "High flow neb"]:
        assert not re.search(device["regex"], text)
    assert definition["aggregate"] == "max"


def test_r5c_iii_011_arterial_chart_gases_fill_only_lab_gaps():
    from easyicu.concept.source_channel import validate_sources
    dictionary = load_dictionary()
    for name, ids in [("be", {776, 224828}), ("tco2", {777, 225698})]:
        definition = dictionary.get(name)
        sources = definition.sources["mimic"]
        charts = [s for s in sources if s.table == "chartevents"]
        assert set(charts[0].ids) == ids
        assert validate_sources(definition, sources)
        assert definition.source_channel_window_hours == 6
        labs = [s for s in sources if s.table == "labevents"]
        assert labs[0].callback == "arterial_specimen_only"
    from easyicu.concept.blood_gas_specimen import SPECIMEN_RULES
    assert SPECIMEN_RULES["mimic"].keep_unrecorded is False


def test_r5c_iii_012_other_vaso_fills_observed_intervals_and_keeps_bolus_points():
    spec = _spec("other_vaso", "mimic", "inputevents_cv", dictionary="sofa2-dict.json")
    frame = pd.DataFrame({"icustay_id": [1] * 4 + [2, 3, 4], "charttime": [0., 3., 4., 8., 0., 0., 0.],
                          "itemid": [30051] * 7, "other_vaso": [.1, .2, 0., 0., np.nan, np.nan, .1],
                          "rate": [.1, .2, 0., 0., np.nan, np.nan, .1], "amount": [np.nan] * 4 + [1., 0., 0.],
                          "stopped": [None, None, "Stopped", None, None, None, "Stopped"], "linkorderid": 1})
    out = _apply(spec, frame, "other_vaso")
    assert out.loc[out.icustay_id.eq(1) & out.other_vaso.eq(True), "charttime"].tolist() == [0., 1., 2., 3.]
    assert out.loc[out.icustay_id.eq(2), "other_vaso"].tolist() == [True]
    assert not out.loc[out.icustay_id.eq(3), "other_vaso"].eq(True).any()
    assert not out.loc[out.icustay_id.eq(4), "other_vaso"].eq(True).any()
    assert not out.loc[out.charttime.ge(4), "other_vaso"].eq(True).any()
    # Without a stop, the final observed positive setting still supports its hour.
    open_episode = _apply(spec, frame.iloc[:2], "other_vaso")
    assert open_episode.loc[open_episode.other_vaso.eq(True), "charttime"].tolist() == [0., 1., 2., 3.]
    assert _dictionary("sofa2-dict.json")["other_vaso"]["aggregate"] == "max"
    mv = _spec("other_vaso", "mimic", "inputevents_mv", dictionary="sofa2-dict.json")
    explicit = pd.DataFrame({"icustay_id": 1, "starttime": [0., 5., 10.], "endtime": [3., 8., 12.],
                             "itemid": 222315, "other_vaso": [.1, 0., np.nan], "rate": [.1, 0., np.nan],
                             "amount": [1., 1., 1.], "statusdescription": "Stopped"})
    mv_out = _apply(mv, explicit, "other_vaso")
    assert mv_out.loc[mv_out.other_vaso.eq(True), "starttime"].tolist() == [0., 1., 2., 10.]
    from easyicu.concept.callbacks import ConceptCallbackContext, _callback_vaso_ind
    from easyicu.table import ICUTable
    ctx = ConceptCallbackContext(concept_name="vaso_ind", target=None, interval=None,
                                 resolver=None, patient_ids=None,
                                 data_source=SimpleNamespace(config=SimpleNamespace(name="mimic")))
    active = out.loc[out.icustay_id.eq(1)]
    indicator = _callback_vaso_ind({"other_vaso": ICUTable(active, ["icustay_id"], "charttime", "other_vaso")}, ctx).data
    assert indicator.charttime.tolist() == [0., 1., 2., 3.]
    assert "other_vaso" in load_dictionary(include_sofa2=True).get("vaso_ind").sub_concepts


def test_r5c_iii_014_lab_fio2_fraction_survives_bounds():
    spec = _spec("fio2", "mimic", "labevents")
    out = _apply(spec, pd.DataFrame({"fio2": [.21, .5, 1., 40., 100.]}), "fio2")
    assert out.fio2.tolist() == [21., 50., 100., 40., 100.]
    from easyicu.callbacks import sofa_resp
    assert sofa_resp(pd.Series([100 / (out.fio2.iloc[1] / 100)])).item() == 2


DRUGS = {"calcium_iv": "Calcium Gluconate", "magnesium_iv": "Magnesium Sulfate",
         "pantoprazole": "Pantoprazole", "neostigmine": "Neostigmine", "rocuronium": "Rocuronium"}


def _ordered_drug(name):
    spec = _spec(name, "mimic", "prescriptions")
    import re
    assert re.search(spec["regex"], DRUGS[name], re.I)
    assert spec["exposure_kind"] == "ordered"
    assert spec["time_precision"] == "day"
    assert spec["route"] == "iv"
    frame = pd.DataFrame({"icustay_id": [1, 2, 3, 4, 5], "startdate": pd.to_datetime(["2150-01-01"] * 5),
                          "enddate": pd.to_datetime(["2150-01-05"] * 5), "drug": DRUGS[name],
                          name: DRUGS[name], "route": ["IV", "PO", "IV DRIP", "IVPB", "IV/PO"]})
    out = _apply(spec, frame, name)
    assert out.icustay_id.tolist() == [1, 3, 4]
    assert out.startdate.tolist() == [pd.Timestamp("2150-01-01 12:00")] * 3
    assert "dur_var" not in out and out[name].eq(True).all()
    assert out.attrs["medication_exposure"]["kind"] == "ordered"
    for administered in _dictionary()[name]["sources"]["mimic"]:
        if administered["table"].startswith("inputevents"):
            assert administered["exposure_kind"] == "administered"
            dose = pd.DataFrame({name: [0., np.nan, 2.], "amount": [0., np.nan, 2.],
                                 "rate": [0., .1, 0.], "statusdescription": "FinishedRunning"})
            assert _apply(administered, dose, name).index.tolist() == [1, 2]
    # A same-day documented administration supersedes the coarse order proxy.
    class Source:
        def load_table(self, table, **kwargs):
            assert table.startswith("inputevents")
            return pd.DataFrame({"icustay_id": [1], "charttime": [pd.Timestamp("2150-01-01 15:00")],
                                 "starttime": [pd.Timestamp("2150-01-01 15:00")], "amount": [1.],
                                 "rate": [1.], "statusdescription": ["FinishedRunning"]})
    dedup = _apply(spec, frame, name, data_source=Source())
    assert dedup.icustay_id.tolist() == ([3, 4] if spec.get("administered_sources") else [1, 3, 4])


def test_r5c_iii_015_calcium_order_proxy_is_day_level_and_intravenous():
    _ordered_drug("calcium_iv")


def test_r5c_iii_016_magnesium_order_proxy_is_day_level_and_intravenous():
    _ordered_drug("magnesium_iv")


def test_r5c_iii_017_pantoprazole_order_proxy_is_day_level_and_intravenous():
    _ordered_drug("pantoprazole")


def test_r5c_iii_018_neostigmine_order_proxy_is_day_level_and_intravenous():
    _ordered_drug("neostigmine")


def test_r5c_iii_019_rocuronium_order_proxy_is_day_level_and_intravenous():
    _ordered_drug("rocuronium")


@pytest.mark.parametrize("bucketed", [False, True])
def test_r5c_iii_021_fentanyl_weight_indexed_carevue_rate_is_mass_per_hour(tmp_path, bucketed):
    specs = _dictionary()["fentanyl_rate"]["sources"]["mimic"]
    spec = next(s for s in specs if 30308 in (s.get("ids") if isinstance(s.get("ids"), list) else [s.get("ids")]))
    frame = pd.DataFrame({"icustay_id": [1, 1, 2], "charttime": [0., 2., 0.], "itemid": 30308,
                          "fentanyl_rate": [2., 0., 2.], "rate": [2., 0., 2.], "rateuom": "mcgkghr",
                          "weight": [60., 60., np.nan], "stopped": [None, "Stopped", None], "linkorderid": 1})
    out = _apply(spec, frame, "fentanyl_rate")
    active = out[out.fentanyl_rate.gt(0)]
    assert active.icustay_id.tolist() == [1, 1]
    assert active.fentanyl_rate.tolist() == [120., 120.]
    assert active.charttime.tolist() == [0., 1.]
    assert out.attrs["mimic_mass_rate_audit"]["missing_weight_rows"] == 1
    # A stop marker can carry the previous positive setting, not a new dose.
    stopped = frame.copy()
    stopped.loc[stopped.stopped.eq("Stopped"), ["fentanyl_rate", "rate"]] = 2.
    stopped_out = _apply(spec, stopped, "fentanyl_rate")
    assert stopped_out.loc[stopped_out.fentanyl_rate.gt(0), "charttime"].tolist() == [0., 1.]
    rejected = _apply(spec, frame.iloc[:1].assign(rateuom="ml/hour"), "fentanyl_rate")
    assert not rejected.fentanyl_rate.gt(0).any()
    assert rejected.attrs["mimic_mass_rate_audit"]["unknown_unit_rows"] == 1
    chart = pd.DataFrame({"icustay_id": [1], "subject_id": [11], "hadm_id": [21],
                          "charttime": pd.Timestamp("2150-01-01"), "storetime": pd.Timestamp("2150-01-01"),
                          "itemid": 762, "valuenum": 60., "value": "60", "valueuom": "kg", "error": 0})
    cv = frame.drop(columns="weight").copy()
    cv["charttime"] = pd.Timestamp("2150-01-01") + pd.to_timedelta(cv.charttime, unit="h")
    cv = cv.drop(columns="fentanyl_rate")
    # A duplicate ordinary mass-rate channel must not double the hourly rate.
    ordinary = cv.loc[cv.icustay_id.eq(1)].copy()
    ordinary["itemid"], ordinary["rateuom"] = 30118, "mcghr"
    ordinary["rate"] *= 60.
    cv = pd.concat([cv, ordinary], ignore_index=True)
    source = _synthetic_source(tmp_path, "mimic", bucketed=bucketed, chartevents=chart, inputevents_cv=cv)
    result = ConceptResolver(load_dictionary()).load_concepts(
        ["fentanyl_rate"], source, merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1),
        patient_ids={"icustay_id": [1, 2]}, concept_workers=1, verbose=False,
    )["fentanyl_rate"].data
    assert result.loc[result.fentanyl_rate.gt(0), "fentanyl_rate"].tolist() == [120., 120.]
    assert set(result.loc[result.fentanyl_rate.gt(0), "icustay_id"]) == {1}


def test_r5c_iii_021_nullable_missing_weight_is_rejected_and_counted():
    spec = next(s for s in _dictionary()["fentanyl_rate"]["sources"]["mimic"] if s.get("ids") == [30308])
    frame = pd.DataFrame({"icustay_id": [1, 2, 3], "charttime": [0., 0., 0.], "itemid": 30308,
                          "fentanyl_rate": pd.Series([2., 2., 2.], dtype="Float64"),
                          "weight": pd.Series([60., pd.NA, 70.], dtype="Float64"),
                          "rateuom": pd.Series(["mcgkghr", "mcgkghr", pd.NA], dtype="string"),
                          "stopped": None, "linkorderid": [1, 2, 3]})
    out = _apply(spec, frame, "fentanyl_rate")
    assert out.loc[out.fentanyl_rate.gt(0), "icustay_id"].tolist() == [1]
    assert out.attrs["mimic_mass_rate_audit"] == {"missing_weight_rows": 1, "unknown_unit_rows": 1}


def test_r5c_iii_012_nullable_zero_rate_remains_explicit_false():
    spec = _spec("other_vaso", "mimic", "inputevents_cv", dictionary="sofa2-dict.json")
    frame = pd.DataFrame({"icustay_id": [1], "charttime": [0.], "itemid": 30051,
                          "other_vaso": pd.Series([0.], dtype="Float64"),
                          "amount": pd.Series([pd.NA], dtype="Float64"), "stopped": None})
    out = _apply(spec, frame, "other_vaso")
    assert out.other_vaso.tolist() == [False]


@pytest.mark.parametrize("database", ["mimic", "mimic_demo"])
def test_r5c_iii_030_order_weight_has_mass_units_without_changing_value(database):
    spec = _spec("weight", database, "inputevents_mv")
    frame = pd.DataFrame({"weight": [63.], "rateuom": ["mcg/kg/min"], "statusdescription": ["FinishedRunning"]})
    out = _apply(spec, frame, "weight", unit_column="rateuom")
    assert out.weight.tolist() == [63.]
    assert out.rateuom.tolist() == ["kg"]


def test_r5c_iv_021_legacy_sofa_stops_carrying_expired_components():
    from easyicu.callbacks import sofa_score
    frame = pd.DataFrame({"stay_id": 1, "time": [0., 30.], "sofa_resp": [4., np.nan]})
    liver = pd.DataFrame({"stay_id": [1], "time": [30.], "sofa_liver": [1.]})
    out = sofa_score({"sofa_resp": frame, "sofa_liver": liver}, keep_components=True)
    assert pd.isna(out.loc[out.time.eq(30), "sofa_resp"].item())
    assert out.loc[out.time.eq(30), "sofa"].item() == 1


def test_r5c_iv_022_hbco_declares_percent_unit():
    assert _dictionary()["hbco"]["unit"] == "%"


def test_r5c_iii_035_gas_documentation_matches_strict_specimen_rule():
    from easyicu.concept import blood_gas_specimen
    assert "record is kept, as in AUMCdb" not in blood_gas_specimen.__doc__


def _native_metadata(table, name, database):
    from easyicu.concept.export_metadata import build_export_file_metadata_binding
    clock = table.index_column
    # Native module publication normalizes both source identifiers and clocks.
    frame = table.data[[table.id_columns[0], clock, name]].rename(columns={table.id_columns[0]: "stay_id", clock: "charttime"})
    binding = build_export_file_metadata_binding(
        relative_path="synthetic.parquet", module="medications", frame=frame,
        concept_ids=[name], database=database, database_class_prefixes=(),
        dictionary=load_dictionary(include_sofa2=True),
    )
    return binding.columns[name].metadata


@pytest.mark.parametrize("bucketed", [False, True])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("database", ["miiv", "mimic"])
def test_ecmo_and_antibacterials_travel_through_real_resolver(tmp_path, database, bucketed, batch):
    start = pd.Timestamp("2150-01-01")
    id_col = "stay_id" if database == "miiv" else "icustay_id"
    chart = pd.DataFrame({id_col: [1, 2, 3], "subject_id": [11, 12, 13], "hadm_id": [21, 22, 23],
                          "charttime": start, "storetime": start, "itemid": [228193, 224660, 228193],
                          "value": ["0", "0", "1"], "valuenum": [0., 0., 1.], "valueuom": ""})
    table = "inputevents" if database == "miiv" else "inputevents_mv"
    inputs = pd.DataFrame({id_col: [1, 2, 3], "subject_id": [11, 12, 13], "hadm_id": [21, 22, 23],
                           "starttime": start + pd.Timedelta(hours=2),
                           "endtime": start + pd.Timedelta(hours=3), "itemid": [225798, 225837, 225838],
                           "rate": [1., 1., 0.], "amount": [100., 100., 0.], "rateuom": "mg/hour",
                           "amountuom": "mg", "statusdescription": "FinishedRunning", "linkorderid": [1, 2, 3]})
    labels = pd.DataFrame({"itemid": [225798, 225837, 225838], "label": ["Ceftriaxone", "Fluconazole", "Vancomycin"]})
    source = _synthetic_source(tmp_path, database, bucketed=bucketed, chartevents=chart, d_items=labels, **{table: inputs})
    result = _load(source, ["ecmo", "abx"], batch=batch)
    ecmo = result["ecmo"].data
    abx = result["abx"].data
    assert ecmo.loc[ecmo.ecmo.eq(True), id_col].tolist() == [3]
    assert abx.loc[abx.abx.eq(True), id_col].tolist() == [1]
    for name in ["ecmo", "abx"]:
        assert _native_metadata(result[name], name, database).source_lineage


@pytest.mark.parametrize("bucketed", [False, True])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("drug", list(DRUGS))
def test_order_proxy_and_carevue_support_use_real_resolver(tmp_path, bucketed, batch, drug):
    start = pd.Timestamp("2150-01-01")
    orders = pd.DataFrame({"subject_id": [11, 12, 13], "hadm_id": [21, 22, 23], "icustay_id": [1, 2, 3],
                           "startdate": start, "enddate": start + pd.Timedelta(days=2),
                           "drug": DRUGS[drug], "route": ["IV", "PO", "IV"], "dose_val_rx": "1", "dose_unit_rx": "g"})
    inputs = pd.DataFrame({"icustay_id": [1, 1, 1], "charttime": start + pd.to_timedelta([0., 3., 4.], unit="h"),
                           "itemid": 30051, "rate": [.1, .2, 0.], "amount": np.nan, "rateuom": "units/min",
                           "stopped": [None, None, "Stopped"], "linkorderid": 1})
    administered = _spec(drug, "mimic", "prescriptions")["administered_sources"]
    if administered:
        itemid = next(s["ids"][0] for s in administered if s["table"] == "inputevents_cv")
        dose = inputs.iloc[:1].assign(charttime=start + pd.Timedelta(hours=15), itemid=itemid, rate=0., amount=1.)
        inputs = pd.concat([inputs, dose], ignore_index=True)
    source = _synthetic_source(tmp_path, "mimic", bucketed=bucketed, prescriptions=orders, inputevents_cv=inputs)
    result = _load(source, [drug, "other_vaso", "vaso_ind"], batch=batch)
    exposure = result[drug].data
    vaso = result["other_vaso"].data
    positives = exposure.loc[exposure[drug].eq(True)]
    clock = result[drug].index_column
    # Executed stay 1 keeps its actual 15h dose; order-only stay 3 keeps noon.
    assert set(zip(positives.icustay_id, positives[clock])) == {(1, 15. if administered else 12.), (3, 12.)}
    assert vaso.loc[vaso.other_vaso.eq(True), "charttime"].tolist() == [0., 1., 2., 3.]
    indicator = result["vaso_ind"].data
    assert indicator.loc[indicator.vaso_ind.eq(True), result["vaso_ind"].index_column].tolist() == [0., 1., 2., 3.]
    lineage = _native_metadata(result[drug], drug, "mimic").source_lineage
    semantics = [dict(s.semantic_parameters) for s in lineage]
    assert any(s.get("exposure_kind") == '"ordered"' and s.get("time_precision") == '"day"' and s.get("route") == '"iv"' for s in semantics)
    if administered:
        assert any(s.get("exposure_kind") == '"administered"' for s in semantics)


@pytest.mark.parametrize("bucketed", [False, True])
@pytest.mark.parametrize("batch", [False, True])
def test_r5c_iii_009_all_ventilation_paths_keep_only_active_evidence(tmp_path, bucketed, batch):
    start = pd.Timestamp("2150-01-01")
    times = [0., 1., 6., 12., 18., 24., 25., 12., 12.]
    values = ["AC/VC", "AC/VC", "Standby", "Ambient", "Off", "AC/VC", "AC/VC", "Off", "Off"]
    chart = pd.DataFrame({"icustay_id": [1] * 7 + [2, 3], "subject_id": [11] * 7 + [12, 13],
                          "hadm_id": [21] * 7 + [22, 23], "itemid": 223849,
                          "charttime": start + pd.to_timedelta(times, unit="h"), "storetime": start,
                          "value": values, "valuenum": np.nan, "valueuom": "", "error": 0})
    procedure = pd.DataFrame({"icustay_id": [3], "subject_id": [13], "hadm_id": [23], "itemid": [225792],
                              "starttime": [start + pd.Timedelta(hours=6)], "endtime": [start + pd.Timedelta(hours=20)],
                              "value": [14.], "valueuom": ["hour"], "statusdescription": ["FinishedRunning"]})
    source = _synthetic_source(tmp_path, "mimic", bucketed=bucketed, chartevents=chart, procedureevents_mv=procedure)
    result = _load(source, ["mech_vent", "vent_start", "vent_ind"], batch=batch)
    starts = result["vent_start"].data
    assert starts.icustay_id.tolist() == [1] * 4
    assert starts[result["vent_start"].index_column].tolist() == [0., 1., 24., 25.]
    for name in ["mech_vent", "vent_ind"]:
        frame = result[name].data
        clock = result[name].index_column
        assert set(frame.icustay_id) == {1, 3}
        # Each explicit procedure is [0, 1) or [24, 25): the stop hour
        # carries no ventilation evidence in either projection (RI-03).
        assert frame.loc[frame.icustay_id.eq(1), clock].tolist() == [0., 24.]
        # Explicit invasive procedure evidence survives a concurrent Off mode.
        explicit = frame.loc[frame.icustay_id.eq(3), clock].tolist()
        assert explicit == [6.] if name == "mech_vent" else 12. in explicit


@pytest.mark.parametrize("bucketed", [False, True])
@pytest.mark.parametrize("database", ["miiv", "mimic"])
def test_r5c_iv_009_and_iii_011_channels_survive_single_and_batch_loading(tmp_path, bucketed, database):
    values = {"crea": 1.1, "hgb": 10., "plt": 100., "wbc": 10., "ph": 7.4, "sao2": 98.} if database == "miiv" else {"be": -2., "tco2": 25.}
    start = pd.Timestamp("2150-01-01")
    identifier = "stay_id" if database == "miiv" else "icustay_id"
    common = {identifier: 1, "subject_id": 11, "hadm_id": 21, "valueuom": "", "error": 0}
    labs, charts = [], []
    for n, (name, value) in enumerate(values.items(), 1):
        for table, hours, rows in [("labevents", [0., 2.], labs), ("chartevents", [3., 12.], charts)]:
            ids = _spec(name, database, table)["ids"]
            item = ids[0] if isinstance(ids, list) else ids
            for hour in hours:
                rows.append({**common, "itemid": item, "charttime": start + pd.Timedelta(hours=hour),
                             "storetime": start + pd.Timedelta(hours=hour + .25), "value": str(value),
                             "valuenum": value, "specimen_id": n * 100 + int(hour)})
        for hour in [0., 2.]:
            labs.append({**common, "itemid": 52033 if database == "miiv" else 50800,
                         "charttime": start + pd.Timedelta(hours=hour), "storetime": start + pd.Timedelta(hours=hour + .25),
                         "value": "ART." if database == "miiv" else "ART", "valuenum": np.nan, "specimen_id": n * 100 + int(hour)})
    source = _synthetic_source(tmp_path, database, bucketed=bucketed, labevents=pd.DataFrame(labs), chartevents=pd.DataFrame(charts))
    for batch in [False, True]:
        result = _load(source, list(values), batch=batch)
        for name, value in values.items():
            frame = result[name].data.sort_values(result[name].index_column)
            assert frame[result[name].index_column].tolist() == [0., 2., 12.], (name, frame)
            assert frame[name].tolist() == pytest.approx([value] * 3)


@pytest.mark.parametrize("bucketed", [False, True])
def test_r5c_iii_010_and_014_device_and_fractional_fio2_reach_respiratory_outputs(tmp_path, bucketed):
    start = pd.Timestamp("2150-01-01")
    chart = pd.DataFrame({"icustay_id": [1, 2, 3], "subject_id": [11, 12, 13], "hadm_id": [21, 22, 23],
                          "itemid": 467, "charttime": start, "storetime": start,
                          "value": ["bipap", "CPAP mask", "Nasal cannula"], "valuenum": np.nan, "valueuom": "", "error": 0})
    labs = pd.DataFrame({"icustay_id": 1, "subject_id": 11, "hadm_id": 21,
                         "itemid": [50816, 50821, 50800], "charttime": start, "storetime": start,
                         "value": ["0.5", "100", "ART"], "valuenum": [.5, 100., np.nan], "valueuom": ""})
    source = _synthetic_source(tmp_path, "mimic", bucketed=bucketed, chartevents=chart, labevents=labs)
    result = _load(source, ["fio2", "pafi", "adv_resp"])
    assert result["fio2"].data.fio2.tolist() == [50.]
    assert result["pafi"].data.pafi.tolist() == [200.]
    support = result["adv_resp"].data
    assert support.loc[support.adv_resp.eq(True), "icustay_id"].tolist() == [1, 2]


@pytest.mark.parametrize("bucketed", [False, True])
def test_r5c_iii_030_order_weight_is_kg_on_source_loading_paths(tmp_path, bucketed):
    start = pd.Timestamp("2150-01-01")
    inputs = pd.DataFrame({"icustay_id": [1], "subject_id": [11], "hadm_id": [21], "itemid": [221794],
                           "starttime": [start], "endtime": [start + pd.Timedelta(hours=1)],
                           "rate": [10.], "amount": [10.], "rateuom": ["mg/hour"], "amountuom": ["mg"],
                           "patientweight": [63.], "statusdescription": ["FinishedRunning"]})
    source = _synthetic_source(tmp_path, "mimic", bucketed=bucketed, inputevents_mv=inputs)
    table = _load(source, ["weight"])["weight"]
    assert table.data.weight.tolist() == [63.]
    if table.unit_column:
        assert table.data[table.unit_column].dropna().tolist() == ["kg"]


@pytest.mark.parametrize("bucketed", [False, True])
def test_r5c_iii_012_standalone_public_indicator_includes_adjunct_support(tmp_path, bucketed):
    import easyicu
    from easyicu.api.concepts import clear_global_loader
    start = pd.Timestamp("2150-01-01")
    inputs = pd.DataFrame({"icustay_id": 1, "subject_id": 11, "hadm_id": 21, "itemid": 30051,
                           "charttime": start + pd.to_timedelta([0., 3., 4.], unit="h"),
                           "rate": [.1, .2, 0.], "amount": np.nan, "rateuom": "units/min",
                           "stopped": [None, None, "Stopped"], "linkorderid": 1})
    _synthetic_source(tmp_path, "mimic", bucketed=bucketed, inputevents_cv=inputs)
    clear_global_loader()
    try:
        result = easyicu.load_concepts(
            ["vaso_ind"], database="mimic", data_path=tmp_path, patient_ids={"icustay_id": [1]},
            merge=False, r_compatible=False, verbose=False, concept_workers=1, parallel_workers=1,
        )["vaso_ind"]
        result = result.data if hasattr(result, "data") else result
        clock = next(c for c in ["charttime", "starttime", "time"] if c in result)
        assert result.loc[result.vaso_ind.eq(True), clock].tolist() == [0., 1., 2., 3.]
    finally:
        clear_global_loader()


def test_r5c_iii_012_compatibility_loader_selects_adjunct_definition(tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module("easyicu.load_concepts")
    source = _synthetic_source(tmp_path, "mimic")
    loader = module.ConceptLoader(source, data_path=str(tmp_path), low_memory=True)

    class SelectedDefinition(Exception):
        pass

    def selected_dictionary(*args, **kwargs):
        dictionary = load_dictionary(include_sofa2=kwargs.get("include_sofa2", False))
        assert "other_vaso" in dictionary.get("vaso_ind").sub_concepts
        raise SelectedDefinition

    monkeypatch.setattr(module, "load_dictionary", selected_dictionary)
    with pytest.raises(SelectedDefinition):
        loader.load_concepts(["vaso_ind"], patient_ids=[1], verbose=False)
