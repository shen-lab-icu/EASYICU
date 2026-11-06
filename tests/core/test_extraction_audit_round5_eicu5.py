"""Round-5 eICU regressions; all records below are invented fixtures."""

from tests.support.bucket_storage import move_fixture_parquet

from tests.support.bucket_storage import synthetic_data_source

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.concept.callback_apply import _apply_callback
from easyicu.concept.schema import ConceptSource
from easyicu.datasource import MissingTableSourceError
from tests.support.eicu_source import _eicu_source, _hourly

DATA = Path(__file__).parents[2] / "src/easyicu/data"
IV_DRUGS = "amiodarone bicarbonate calcium_iv diltiazem fentanyl labetalol levetiracetam magnesium_iv morphine other_vaso pantoprazole potassium_iv".split()
INFUSIONS = "amiodarone cisatracurium dexmedetomidine diltiazem esmolol fentanyl furosemide heparin insulin loop_diuretic midazolam milrinone nicardipine nitroglycerin octreotide other_vaso pantoprazole propofol".split()


def _definition(name):
    for file in ("concept-dict.json", "sofa2-dict.json"):
        d = json.loads((DATA / file).read_text())
        if name in d:
            return d[name]
    raise KeyError(name)


def _sources(name, table=None):
    rows = _definition(name).get("sources", {}).get("eicu", [])
    return [ConceptSource.from_mapping(s) for s in rows if table is None or s.get("table") == table]


def _fake(**tables):
    def load_table(name, **kwargs):
        if name not in tables:
            raise MissingTableSourceError(name)  # the source's absence signal
        return SimpleNamespace(data=tables[name].copy())
    return SimpleNamespace(config=SimpleNamespace(name="eicu"), load_table=load_table)


def _apply(name, source, frame, ds=None):
    frame = frame.copy()
    if name not in frame:
        value = source.value_var or {"medication": "dosage", "infusiondrug": "drugrate", "lab": "labresult", "intakeoutput": "cellvaluenumeric", "treatment": "treatmentstring", "patient": "admissionheight", "respiratorycharting": "respchartvalue"}[source.table]
        frame[name] = frame[value]
    return _apply_callback(frame, source, name, data_source=ds)


@pytest.mark.parametrize("removal", ["IABP removal", "intra-aortic balloon pump removal"])
def test_r5_eicu_004_removal_ends_only_its_own_support_interval(removal):
    s = _sources("mech_circ_support")[0]
    f = pd.DataFrame({"patientunitstayid": [1, 1, 2], "treatmentoffset": [0, 180, 0], "treatmentstring": ["cardiac|IABP insertion", "cardiac|" + removal, "cardiac|IABP removal"]})
    out = _apply("mech_circ_support", s, f)
    supported = out.loc[out.mech_circ_support.eq(True)]  # noqa: E712
    assert set(zip(supported.patientunitstayid, supported.treatmentoffset)) == {(1, 0), (1, 60), (1, 120)}
    assert not out.loc[out.patientunitstayid.eq(2), "mech_circ_support"].any()


def test_r5_eicu_005_dex_hicl_requires_parsed_dose_and_execution():
    ss = _sources("dex", "medication")
    fallback = next((s for s in ss if s.sub_var == "drughiclseqno" and 926 in s.ids), None)
    assert fallback is not None
    f = pd.DataFrame({"patientunitstayid": [1] * 6, "drugstartoffset": [0] * 6, "drugstopoffset": [120, 120, 120, 120, 120, 0], "dosage": ["25 g", "25000 mg", "50 mL", "one syringe", "50 mL", "50 mL"], "drugname": [""] * 6, "drugordercancelled": ["No", "No", "No", "No", "Yes", "No"], "routeadmin": ["IV"] * 6, "drughiclseqno": [926] * 6})
    infusion = pd.DataFrame({"patientunitstayid": [1, 1], "infusionoffset": [0, 120], "drugname": ["Dextrose 50% (ml/hr)"] * 2, "drugrate": [25, 0]})
    out = _apply("dex", fallback, f, _fake(infusiondrug=infusion))
    assert out.dex.tolist() == [125.0] * 6  # three parsed doses, two documented hours each
    assert _apply("dex", fallback, f, _fake()).empty



@pytest.mark.parametrize("dosage, parsed", [("25 Gm", True), ("25 Gram", True), ("25 GM", True), ("25 grams", True),
                                            ("25-50 mL", False), ("25", False), ("25 3", False)])
def test_r7_eicu_dex_reads_gram_spellings_and_keeps_ranges_unknown(dosage, parsed):
    # 2026-11-05 round-7 evidence: "Gm"/"Gram"/"GM" grams were unparsed D50 doses.
    fallback = next(s for s in _sources("dex", "medication") if s.sub_var == "drughiclseqno")
    f = pd.DataFrame({"patientunitstayid": [1], "drugstartoffset": [0], "drugstopoffset": [120], "dosage": [dosage],
                      "drugname": [""], "drugordercancelled": ["No"], "routeadmin": ["IV"], "drughiclseqno": [926]})
    infusion = pd.DataFrame({"patientunitstayid": [1, 1], "infusionoffset": [0, 120], "drugname": ["Dextrose 50% (ml/hr)"] * 2, "drugrate": [25, 0]})
    out = _apply("dex", fallback, f, _fake(infusiondrug=infusion))
    assert out.dex.tolist() == ([125.0, 125.0] if parsed else [])

def test_r5_eicu_006_raw_io_dedup_preserves_channels_and_entry_times(tmp_path):
    f = pd.DataFrame({"intakeoutputid": [1, 2, 3, 4], "intakeoutputoffset": [10] * 4, "intakeoutputentryoffset": [11, 11, 12, 11], "celllabel": ["Urine", "Urine", "Urine", "Voided Amount"], "cellvaluenumeric": [100.0] * 4, "cellvaluetext": [""] * 4})
    f["cellpath"] = "flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|" + f.celllabel
    ds = _eicu_source(tmp_path, "eicu", intakeoutput=f)
    raw = ds.load_table("intakeoutput", columns=["patientunitstayid", "intakeoutputoffset", "cellvaluenumeric"]).data
    assert len(raw) == 3
    assert raw.attrs["eicu_io_dedup_audit"]["duplicates_removed"] == 1
    assert _hourly(ds, "urine", "sum") == {0.0: 300.0}


@pytest.mark.parametrize("name", ["po2", "pco2", "ph", "sao2"])
def test_r5_eicu_009_same_offset_specimen_rejects_venous_and_conflicts(name):
    f = pd.DataFrame({"patientunitstayid": [1] * 5, "labresultoffset": [0, 60, 120, 180, 240], "labresult": [7.4] * 5})
    samples = pd.DataFrame({"patientunitstayid": [1] * 5, "labotheroffset": [0, 60, 120, 120, 240], "labothername": ["Sample Type"] * 5, "labothervaluetext": ["Venous", "Arterial", "Arterial", "Venous", "Capillary"], "labotherresult": [""] * 5})
    out = _apply(name, _sources(name, "lab")[0], f, _fake(customlab=samples))
    # Venous, capillary and conflicting samples drop; offset 180 has no sample:
    # paO2/paCO2 are arterial by name, the others lack a same-offset paO2/paCO2.
    label_defined = name in {"po2", "pco2"}
    assert out.labresultoffset.tolist() == ([60, 180] if label_defined else [60])
    assert out.specimen_status.tolist() == (["explicit_arterial", "label_defined_arterial"] if label_defined else ["explicit_arterial"])
    assert out.attrs["eicu_specimen_audit"]["excluded_venous"] == 3
    assert out.attrs["eicu_specimen_audit"]["excluded_unknown"] == (0 if label_defined else 1)


def test_r5_eicu_011_adult_height_keeps_raw_value_and_qc_reason():
    f = pd.DataFrame({"patientunitstayid": [1, 2, 3], "admissionheight": [69.0, 170.0, 90.0], "age": [60, 60, 5], "admissionweight": [85.0, 80.0, 15.0]})
    out = _apply("height", _sources("height")[0], f)
    assert np.isnan(out.height.iloc[0])
    assert out.height.iloc[1:].tolist() == [170.0, 90.0]
    assert out.admissionheight.iloc[0] == 69.0
    assert out.height_qc_reason.iloc[0] == "adult_height_outside_trusted_cm_domain"


@pytest.mark.parametrize("device", ["HFNC", "High Flow Nasal Cannula", "high-flow nasal cannula"])
def test_r5_eicu_013_hfnc_device_support_does_not_become_invasive(device):
    f = pd.DataFrame({"patientunitstayid": [1], "respchartoffset": [0], "respchartvaluelabel": ["O2 Device"], "respchartvalue": [device]})
    ss = [s for s in _sources("adv_resp", "respiratorycharting") if s.ids and "O2 Device" in s.ids]
    assert any(_apply("adv_resp", s, f).adv_resp.eq(True).any() for s in ss)  # noqa: E712
    s = next(s for s in _sources("mech_vent", "respiratorycharting") if s.ids and "O2 Device" in s.ids)
    assert _apply("mech_vent", s, f).empty


def test_r5_eicu_014_diagnosis_negation_is_row_local():
    texts = ["infectious diseases|pneumonia", "cardiac|noninfectious pericarditis", "GI|ischemic colitis", "GI|ulcerative colitis", "GI|Crohn disease enteritis", "ID|no evidence of infection", "GI|infectious colitis"]
    f = pd.DataFrame({"patientunitstayid": [1] * 7, "diagnosisoffset": [0] * 7, "diagnosisstring": texts})
    out = _apply("infection_icd", _sources("infection_icd")[0], f)
    assert out.diagnosisstring.tolist() == [texts[0], texts[6]]
    assert out.infection_icd.all()


def test_r5_eicu_015_input_separates_volume_cumulative_irrigation_and_rate():
    labels = ["I.V.", "24h Total Intake", "CBI/Bladder Irr In", "fluid (mL/hr)", "fluid (mL/hr)", "Nutrition Total"]
    f = pd.DataFrame({"patientunitstayid": [1] * 6, "intakeoutputoffset": [0, 0, 0, 0, 120, 0], "intakeoutputentryoffset": [0, 0, 0, 0, 120, 0], "celllabel": labels, "cellvaluenumeric": [100.0, 2400.0, 3000.0, 50.0, 0.0, 80.0]})
    f["cellpath"] = "flowsheet|Flowsheet Cell Labels|I&O|Intake (ml)|" + f.celllabel
    out = _apply("total_input_ml", _sources("total_input_ml")[0], f)
    assert out.total_input_ml.sum() == pytest.approx(280.0)
    assert set(out.intakeoutputoffset) == {0, 60}


def test_r5_eicu_020_fio2_sources_normalize_and_exclude_flow():
    for name, table in [("fio2_chart", "respiratorycharting"), ("fio2_lab", "lab")]:
        ss = _sources(name, table)
        assert ss
        if table == "lab":
            f = pd.DataFrame({"patientunitstayid": [1, 1], "labresultoffset": [0, 60], "labresult": [0.4, 40], "labmeasurenameinterface": ["%", "lpm"]})
        else:
            f = pd.DataFrame({"patientunitstayid": [1, 1], "respchartoffset": [0, 60], "respchartvalue": ["40%", "40 L/min"], "respchartvaluelabel": ["FiO2"] * 2})
        out = _apply(name, ss[0], f)
        assert out[name].dropna().tolist() == [40.0]
        assert out.attrs["eicu_fio2_source"] == ("lab" if table == "lab" else "chart")


def test_r5_eicu_021_loop_dose_uses_mass_rate_and_explicit_iv_order():
    ss = _sources("loop_diuretic_furosemide_eq", "infusiondrug")
    assert ss
    f = pd.DataFrame({"patientunitstayid": [1, 1, 2], "infusionoffset": [0, 120, 0], "drugname": ["Bumetanide (mg/hr)"] * 3, "drugrate": [1, 0, 1]})
    orders = pd.DataFrame({"patientunitstayid": [1, 2], "drugstartoffset": [0, 0], "drugstopoffset": [120, 120], "drugname": ["Bumetanide"] * 2, "routeadmin": ["IV", "unknown"], "drugordercancelled": ["No"] * 2})
    out = _apply("loop_diuretic_furosemide_eq", ss[0], f, _fake(medication=orders))
    assert out.loop_diuretic_furosemide_eq.sum() == pytest.approx(80.0)
    assert set(out.patientunitstayid) == {1}


def test_r5_eicu_022_mode_and_sequence_keep_ambiguous_breaths_unknown():
    f = pd.DataFrame({"patientunitstayid": [1] * 4, "respchartoffset": range(4), "respchartvalue": ["APVcmv", "SIMV", "Spontaneous", "AC/CMV"], "respchartvaluelabel": ["Ventilator Mode"] * 4})
    for name, expected in [("vent_mode", ["dual_adaptive", None, None, None]), ("vent_breath_seq", [None, "simv", "spontaneous", None])]:
        ss = _sources(name)
        assert ss
        out = _apply(name, ss[0], f)
        assert out[name].astype(object).where(out[name].notna(), None).tolist() == expected


@pytest.mark.parametrize("name", IV_DRUGS)
def test_r5_eicu_023_iv_exposure_requires_unambiguous_route(name):
    for s in _sources(name, "medication"):
        f = pd.DataFrame({"patientunitstayid": [1] * 8, "drugname": [""] * 8, "dosage": ["1 mg"] * 8, "drugordercancelled": ["No"] * 8, "routeadmin": ["IV", "intravenous", "PO/IV", "SC", "IM", "INTRAMUSCULAR", "PR", "unknown"]})
        out = _apply(name, s, f)
        assert out.loc[out[name].eq(True), "routeadmin"].tolist() == ["IV", "intravenous"]  # noqa: E712
        assert out.attrs["eicu_medication_audit"]["route_unknown"] >= 1


@pytest.mark.parametrize("kind", ["kg", "mass"])
def test_r5_eicu_026_rate_gap_never_outlives_six_hours(kind):
    from easyicu.utils.callback_utils import eicu_rate_kg_callback, eicu_rate_mass_callback

    f = pd.DataFrame({"patientunitstayid": [1, 1, 1], "infusionoffset": [0, 600, 660], "drugname": ["drug (mcg/kg/min)" if kind == "kg" else "drug (mg/hr)"] * 3, "rate": [1.0, 2.0, 0.0]})
    cb = eicu_rate_kg_callback(32) if kind == "kg" else eicu_rate_mass_callback("mg/hour")
    extra = {"weight_var": "patientweight"} if kind == "kg" else {}
    out = cb(f, val_var="rate", sub_var="drugname", concept_name="rate", **extra)
    assert not out.loc[out.infusionoffset.between(360, 540), "rate"].gt(0).any()
    assert out.loc[out.infusionoffset.eq(660), "rate"].item() == 0


def test_r5_eicu_028_positive_actual_patient_fluid_removal_is_rrt():
    ss = [s for s in _sources("rrt", "intakeoutput") if s.ids and "Actual Patient Fluid Removal" in s.ids]
    assert ss
    f = pd.DataFrame({"patientunitstayid": [1] * 3, "cellvaluenumeric": [100.0, 0.0, np.nan]})
    out = _apply("rrt", ss[0], f)
    assert out.rrt.iloc[0] == True  # noqa: E712
    assert out.rrt.iloc[1] == False  # noqa: E712
    assert pd.isna(out.rrt.iloc[2])
    assert _definition("rrt")["aggregate"] == "max"


def test_r5_eicu_034_shared_settings_use_cross_source_anchors_and_stop_barriers(tmp_path):
    settings = pd.DataFrame({"respchartoffset": [0, 60, 120, 180, 240, 300], "respchartvaluelabel": ["PEEP", "PEEP", "O2 Device", "PEEP", "RT Vent On/Off", "PEEP"], "respchartvalue": ["5", "5", "BiPAP", "5", "Stop", "5"]})
    care = pd.DataFrame({"airwaytype": ["Oral ETT"], "ventstartoffset": [0], "ventendoffset": [0], "priorventstartoffset": [0], "priorventendoffset": [0], "respcarestatusoffset": [0]})
    ds = _eicu_source(tmp_path, "eicu", respiratorycharting=settings, respiratorycare=care)
    out = _hourly(ds, "vent_ind")
    assert out.get(0.0) == True  # noqa: E712
    assert out.get(1.0) == True  # noqa: E712
    assert not any(out.get(h, False) for h in [2.0, 3.0, 4.0, 5.0])


@pytest.mark.parametrize("name", INFUSIONS)
def test_r5_eicu_037_infusion_exposure_requires_positive_rate(name):
    for s in _sources(name, "infusiondrug"):
        f = pd.DataFrame({"patientunitstayid": [1] * 5, "drugname": ["drug"] * 5, "drugrate": ["2", "0", "", "paused", "inf"], "infusionoffset": [0, 60, 120, 180, 240]})
        out = _apply(name, s, f)
        assert out[name].iloc[0] == True  # noqa: E712
        assert out[name].iloc[1] == False  # noqa: E712
        assert out[name].iloc[2:].isna().all()
        assert out.eicu_exposure_source.iloc[3] == "infusion_paused"
        assert out.attrs["eicu_medication_audit"]["positive"] == 1
        assert _definition(name)["aggregate"] == "max"


def test_r5_eicu_003_explicit_mgdl_ionized_calcium_converts_boundary():
    f = pd.DataFrame({"labresult": [2.25, 2.25, 1.1], "labmeasurenameinterface": ["mg/dL", "mmol/L", "mg/dL"]})
    out = _apply("cai", _sources("cai", "lab")[0], f)
    assert out.cai.tolist() == pytest.approx([2.25 / 4.008, 2.25, 1.1])


def test_r5_eicu_017_insulin_requires_units_or_row_concentration():
    f = pd.DataFrame({"patientunitstayid": [1] * 4, "infusionoffset": [0, 60, 120, 180], "drugname": ["insulin (units/hr)", "insulin (ml/hr)", "insulin (ml/hr)", "insulin ()"], "drugrate": [3.0] * 4, "drugamount": [None, 100.0, None, None], "volumeoffluid": [None, 50.0, None, None]})
    out = _apply("ins", _sources("ins")[0], f)
    assert out.ins.iloc[:2].tolist() == [3.0, 6.0]
    assert out.ins.iloc[2:].isna().all()


def test_r5_eicu_036_rate_weight_uses_past_unit_labelled_fallbacks():
    from easyicu.utils.callback_utils import eicu_rate_kg_callback

    f = pd.DataFrame({"patientunitstayid": [1, 2, 3, 4], "infusionoffset": [120] * 4, "drugname": ["drug (mcg/min)"] * 4, "drugrate": [8.0] * 4, "patientweight": [0.0, np.nan, np.nan, 40.0]})
    patient = pd.DataFrame({"patientunitstayid": [1, 2, 3, 4], "admissionweight": [np.nan, 0, np.nan, 80]})
    chart = pd.DataFrame({"patientunitstayid": [1, 1, 2, 3], "intakeoutputoffset": [60, 180, 60, 180], "celllabel": ["Bodyweight (kg)", "Bodyweight (kg)", "Bodyweight (lb)", "Bodyweight (kg)"], "cellvaluenumeric": [80.0, 40.0, 80 / .45359237, 80.0]})
    out = eicu_rate_kg_callback(32)(f, val_var="drugrate", sub_var="drugname", weight_var="patientweight", concept_name="rate", data_source=_fake(patient=patient, intakeoutput=chart))
    assert out.set_index("patientunitstayid").rate.to_dict() == pytest.approx({1: .1, 2: .1, 4: .2})


def test_dictionary_sources_reach_raw_loader_without_renaming_hazards(tmp_path):
    infusion = pd.DataFrame({"infusionoffset": [0, 60], "drugname": ["Amiodarone (mg/hr)"] * 2, "drugrate": ["2", "0"]})
    lab = pd.DataFrame({"labresultoffset": [0], "labname": ["FiO2"], "labresult": [0.4], "labmeasurenameinterface": ["%"]})
    resp = pd.DataFrame({"respchartoffset": [0], "respchartvaluelabel": ["FiO2"], "respchartvalue": ["60%"]})
    ds = _eicu_source(tmp_path, "eicu", infusiondrug=infusion, lab=lab, respiratorycharting=resp)
    assert _hourly(ds, "amiodarone") == {0.0: True, 1.0: False}
    assert _hourly(ds, "fio2_lab") == {0.0: 40.0}
    assert _hourly(ds, "fio2_chart") == {0.0: 60.0}
    assert _hourly(ds, "fio2") == {0.0: 40.0}


def test_r5_eicu_034_niv_splits_a_bridge_before_a_new_invasive_episode(tmp_path):
    resp = pd.DataFrame({"respchartoffset": [0, 180, 360], "respchartvaluelabel": ["O2 Device"] * 3, "respchartvalue": ["ETT", "BiPAP", "ETT"]})
    care = pd.DataFrame({"airwaytype": [""], "ventstartoffset": [0], "ventendoffset": [0], "priorventstartoffset": [0], "priorventendoffset": [0]})
    ds = _eicu_source(tmp_path, "eicu", respiratorycharting=resp, respiratorycare=care)
    out = _hourly(ds, "vent_ind")
    assert set(out) == {0.0, 1.0, 2.0, 6.0}


def test_r5_eicu_015_unclosed_rates_are_not_guessed_as_one_hour_volumes():
    f = pd.DataFrame({"patientunitstayid": [1], "intakeoutputoffset": [0], "cellpath": ["flowsheet|Flowsheet Cell Labels|I&O|Intake (ml)|IV (ml/hr)"], "cellvaluenumeric": [100.0]})
    assert _apply("total_input_ml", _sources("total_input_ml")[0], f).empty


def test_r5_eicu_021_loop_order_quantity_cannot_substitute_for_execution(tmp_path):
    order = pd.DataFrame({"drugstartoffset": [0], "drugstopoffset": [120], "drugname": ["Furosemide"], "drughiclseqno": [3660], "routeadmin": ["IV"], "dosage": ["40 mg"], "drugordercancelled": ["No"]})
    ds = _eicu_source(tmp_path, "eicu", medication=order)
    assert _hourly(ds, "loop_diuretic_furosemide_eq") == {}


def test_r5_eicu_021_loop_rejects_volume_rate_and_cancelled_route_evidence():
    source = _sources("loop_diuretic_furosemide_eq")[0]
    f = pd.DataFrame({"patientunitstayid": [1, 2], "infusionoffset": [0, 0], "drugname": ["Furosemide (ml/hr)", "Furosemide (mg/hr)"], "drugrate": [10.0, 10.0]})
    orders = pd.DataFrame({"patientunitstayid": [1, 2], "drugstartoffset": [0, 0], "drugstopoffset": [60, 60], "drugname": ["Furosemide"] * 2, "routeadmin": ["IV"] * 2, "drugordercancelled": ["No", "Yes"]})
    assert _apply("loop_diuretic_furosemide_eq", source, f, _fake(medication=orders)).empty


@pytest.mark.parametrize("offset", [1.0, pd.Timedelta(hours=1), pd.Timestamp("2020-01-01")])
def test_r5_eicu_009_missing_specimens_follow_the_r6_order_for_each_clock(offset):
    f = pd.DataFrame({"patientunitstayid": [1], "labresultoffset": [offset], "labresult": [80.0]})
    po2 = _apply("po2", _sources("po2")[0], f, _fake())
    assert po2.specimen_status.tolist() == ["label_defined_arterial"]
    ph = _apply("ph", _sources("ph")[0], f.assign(labresult=7.4), _fake())
    assert ph.empty
    assert ph.attrs["eicu_specimen_audit"]["excluded_unknown"] == 1


def test_r5_eicu_020_explicit_chart_flow_label_is_not_a_percentage():
    f = pd.DataFrame({"respchartvalue": [40], "respchartvaluelabel": ["FiO2 (L/min)"]})
    assert _apply("fio2_chart", _sources("fio2_chart")[0], f).empty


def test_r5_eicu_005_verified_hicl_dose_expands_only_its_documented_interval(tmp_path):
    medication = pd.DataFrame({"drugstartoffset": [0, 600], "drugstopoffset": [120, 720], "drugname": [""] * 2, "dosage": ["25 g"] * 2, "drughiclseqno": [926] * 2, "drugordercancelled": ["No"] * 2, "routeadmin": ["IV"] * 2})
    infusion = pd.DataFrame({"infusionoffset": [0, 120, 600, 720], "drugname": ["Dextrose 50% (ml/hr)"] * 4, "drugrate": [25, 0, 25, 0]})
    ds = _eicu_source(tmp_path, "eicu", medication=medication, infusiondrug=infusion)
    assert _hourly(ds, "dex") == {0.0: 125.0, 1.0: 125.0, 10.0: 125.0, 11.0: 125.0}


def test_r5_eicu_021_verified_loop_rate_is_an_hourly_amount_in_raw_loader(tmp_path):
    medication = pd.DataFrame({"drugstartoffset": [0], "drugstopoffset": [120], "drugname": ["Bumetanide"], "drugordercancelled": ["No"], "routeadmin": ["IV"]})
    infusion = pd.DataFrame({"infusionoffset": [0, 120], "drugname": ["Bumetanide (mg/hr)"] * 2, "drugrate": [1, 0]})
    ds = _eicu_source(tmp_path, "eicu", medication=medication, infusiondrug=infusion)
    assert _hourly(ds, "loop_diuretic_furosemide_eq", "sum") == {0.0: 40.0, 1.0: 40.0}


def test_r5_eicu_037_nullable_rate_keeps_unknown_without_a_boolean_cast_error():
    f = pd.DataFrame({"drugrate": pd.Series(["2", "0", pd.NA], dtype="string")})
    out = _apply("insulin", _sources("insulin", "infusiondrug")[0], f)
    assert out.insulin.iloc[0] == True  # noqa: E712
    assert out.insulin.iloc[1] == False  # noqa: E712
    assert pd.isna(out.insulin.iloc[2])


def test_r5_eicu_011_unknown_age_does_not_guess_an_adult_height_conversion():
    f = pd.DataFrame({"admissionheight": [69.0], "age": pd.Series([pd.NA], dtype="string")})
    out = _apply("height", _sources("height")[0], f)
    assert out.height.item() == 69.0
    assert out.admissionheight.item() == 69.0


def _review_io_source(tmp_path, layout):
    # Each channel has one true duplicate, plus separate entry times and text.
    rows = []
    for label, section, volume in [("Urine", "Output", 100), ("Drain", "Output", 20), ("I.V.", "Intake", 200)]:
        for entry, text in [(11, ""), (11, ""), (12, ""), (11, "independent")]:
            rows.append({"intakeoutputid": len(rows) + 1, "intakeoutputoffset": 10,
                         "intakeoutputentryoffset": entry, "celllabel": label,
                         "cellpath": f"flowsheet|Flowsheet Cell Labels|I&O|{section} (ml)|{label}",
                         "cellvaluenumeric": volume, "cellvaluetext": text})
    ds = _eicu_source(tmp_path, "eicu", intakeoutput=pd.DataFrame(rows))
    if layout != "file":
        folder = tmp_path / ("intakeoutput_bucket/bucket_id=0" if layout == "bucket" else "intakeoutput")
        folder.mkdir(parents=True)
        move_fixture_parquet(tmp_path / "intakeoutput.parquet", folder / "part.parquet")
    return ds


@pytest.mark.parametrize("layout", ["file", "flat", "bucket"])
def test_r5_eicu_006_all_io_consumers_deduplicate_before_hourly_sum(tmp_path, layout):
    from easyicu.concept import ConceptDictionary, ConceptResolver

    ds = _review_io_source(tmp_path, layout)
    names = ["urine", "total_input_ml", "other_output_ml", "fluid_balance", "fluid_balance_cumulative"]
    resolver = ConceptResolver(ConceptDictionary.from_json(DATA / "concept-dict.json"))
    tables = resolver.load_concepts(names, ds, merge=False, r_compatible=False,
                                   patient_ids={"patientunitstayid": [1]}, concept_workers=1,
                                   interval=pd.Timedelta(hours=1), verbose=False)
    for name, value in zip(names, [300, 600, 60, 240, 240]):
        assert tables[name].data[name].dropna().tolist() == [value], name
    assert _hourly(ds, "urine", "sum") == {0.0: 300.0}


def test_r5_eicu_006_duckdb_multi_scan_preserves_every_raw_identity_field(tmp_path):
    from easyicu.datasource import load_bucketed_table_multi_aggregated

    ds = _review_io_source(tmp_path, "bucket")
    out = load_bucketed_table_multi_aggregated(
        ds, "intakeoutput", {"urine": ["Urine"], "other_output_ml": ["Drain"]},
        value_column="cellvaluenumeric", time_col="intakeoutputoffset", itemid_col="celllabel",
        patient_ids=[1], agg_func="sum",
    )
    assert out.urine.tolist() == [300.0]
    assert out.other_output_ml.tolist() == [60.0]
    assert out.attrs["eicu_io_dedup_audit"]["duplicates_removed"] == 2


def test_r5_eicu_004_logical_pooling_is_max_in_each_dictionary():
    for path in [DATA / "concept-dict.json", DATA / "sofa2-dict.json"]:
        assert json.loads(path.read_text())["mech_circ_support"]["aggregate"] == "max"


def test_r5_eicu_004_partial_hour_support_reaches_the_hour_before_removal(tmp_path):
    treatment = pd.DataFrame({"treatmentoffset": [30, 90], "treatmentstring": ["IABP insertion", "IABP removal"]})
    ds = _eicu_source(tmp_path, "eicu", treatment=treatment)
    assert _hourly(ds, "mech_circ_support", "max") == {0.0: True, 1.0: True}


@pytest.mark.parametrize("endpoint", ["drugstartoffset", "drugstopoffset"])
def test_r5_eicu_021_missing_order_endpoint_cannot_confirm_route_interval(endpoint):
    f = pd.DataFrame({"patientunitstayid": [1], "infusionoffset": [0], "drugname": ["Furosemide (mg/hr)"], "drugrate": [10.0]})
    orders = pd.DataFrame({"patientunitstayid": [1], "drugstartoffset": [0.0], "drugstopoffset": [60.0], "drugname": ["Furosemide"], "routeadmin": ["IV"], "drugordercancelled": ["No"]})
    orders[endpoint] = np.nan
    assert _apply("loop_diuretic_furosemide_eq", _sources("loop_diuretic_furosemide_eq")[0], f, _fake(medication=orders)).empty


@pytest.mark.parametrize("diagnosis", ["no pneumonia", "sepsis ruled out", "without bacteremia"])
def test_r5_eicu_014_explicit_negation_of_named_infections_is_not_evidence(diagnosis):
    f = pd.DataFrame({"diagnosisstring": [diagnosis, "infectious diseases|pneumonia"]})
    out = _apply("infection_icd", _sources("infection_icd")[0], f)
    assert out.diagnosisstring.tolist() == ["infectious diseases|pneumonia"]


@pytest.mark.parametrize("database", ["eicu", "eicu_demo"])
def test_r5_eicu_005_hicl_fallback_cannot_override_a_conflicting_named_order(tmp_path, database):
    order = pd.DataFrame({"drugstartoffset": [0], "drugstopoffset": [60], "drugname": ["Dextrose 5%"], "drughiclseqno": [926], "routeadmin": ["IV"], "dosage": ["25 mL"], "drugordercancelled": ["No"]})
    infusion = pd.DataFrame({"infusionoffset": [0, 60], "drugname": ["Dextrose 50% (ml/hr)"] * 2, "drugrate": [25, 0]})
    ds = _eicu_source(tmp_path, database, medication=order, infusiondrug=infusion)
    assert _hourly(ds, "dex") == {}


def test_r5_eicu_034_niv_setting_cuts_cross_source_bridge(tmp_path):
    resp = pd.DataFrame({"respchartoffset": [0, 180, 360], "respchartvaluelabel": ["O2 Device", "B1: IPAP", "O2 Device"], "respchartvalue": ["ETT", "10", "ETT"]})
    ds = _eicu_source(tmp_path, "eicu", respiratorycharting=resp)
    assert set(_hourly(ds, "vent_ind")) == {0.0, 1.0, 2.0, 6.0}


def test_r5_eicu_034_later_same_hour_invasive_anchor_restarts_support(tmp_path):
    resp = pd.DataFrame({"respchartoffset": [0, 60, 90, 120], "respchartvaluelabel": ["O2 Device"] * 4, "respchartvalue": ["ETT", "BiPAP", "ETT", "ETT"]})
    ds = _eicu_source(tmp_path, "eicu", respiratorycharting=resp)
    assert set(_hourly(ds, "vent_ind")) == {0.0, 1.0, 2.0}


@pytest.mark.parametrize("name,drug", [
    ("norepi_rate", "Norepinephrine (mcg/kg/min)"), ("epi_rate", "Epinephrine (mcg/kg/min)"),
    ("dopa_rate", "Dopamine (mcg/kg/min)"), ("dobu_rate", "Dobutamine (mcg/kg/min)"),
    ("phn_rate", "Phenylephrine (mcg/kg/min)"), ("propofol_rate", "Propofol (mcg/kg/min)"),
    ("fentanyl_rate", "Fentanyl (mcg/hr)"), ("midazolam_rate", "Midazolam (mg/hr)"),
])
def test_r5_eicu_026_every_dictionary_rate_retains_six_hour_gap_and_zero_stop(tmp_path, name, drug):
    infusion = pd.DataFrame({"infusionoffset": [0, 600, 660], "drugname": [drug] * 3, "drugrate": [1.0, 2.0, 0.0]})
    ds = _eicu_source(tmp_path, "eicu", infusiondrug=infusion)
    out = _hourly(ds, name)
    assert set(out) == {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0}
    assert out[11.0] == 0


def test_r5_eicu_036_conflicting_unit_labelled_weights_are_not_arbitrarily_prioritized():
    from easyicu.concept.eicu_source_qc import rate_weights

    frame = pd.DataFrame({"patientunitstayid": [1], "infusionoffset": [120]})
    chart = pd.DataFrame({"patientunitstayid": [1, 1], "intakeoutputoffset": [60, 60],
                          "celllabel": ["Bodyweight (kg)", "Bodyweight (lb)"], "cellvaluenumeric": [80.0, 80.0]})
    assert np.isnan(rate_weights(frame, _fake(intakeoutput=chart))[0])


def test_r5_eicu_005_unknown_cancellation_status_does_not_verify_a_dose():
    frame = pd.DataFrame({"patientunitstayid": [1], "drugstartoffset": [0], "drugstopoffset": [60],
                          "dosage": ["25 mL"], "routeadmin": ["IV"], "drugordercancelled": [""]})
    infusion = pd.DataFrame({"patientunitstayid": [1, 1], "infusionoffset": [0, 60], "drugname": ["Dextrose 50% (ml/hr)"] * 2, "drugrate": [25, 0]})
    assert _apply("dex", _sources("dex", "medication")[0], frame, _fake(infusiondrug=infusion)).empty


@pytest.mark.parametrize("name", IV_DRUGS)
def test_r5_eicu_023_explicit_zero_order_amount_is_not_positive_exposure(name):
    frame = pd.DataFrame({"dosage": ["0 mg", "1 mg", ""], "routeadmin": ["IV"] * 3,
                          "drugname": [name] * 3, "drugordercancelled": ["No"] * 3})
    out = _apply(name, _sources(name, "medication")[0], frame)
    assert out[name].iloc[0] == False  # noqa: E712
    assert out[name].iloc[1] == True  # noqa: E712
    assert pd.isna(out[name].iloc[2])


def test_r5_eicu_006_registered_raw_loader_uses_all_nonprimary_fields():
    from easyicu.config import load_src_cfg
    from easyicu.datasource import ICUDataSource

    raw = pd.DataFrame({"patientunitstayid": [1] * 4, "intakeoutputid": [1, 2, 3, 4],
                        "intakeoutputoffset": [10] * 4, "intakeoutputentryoffset": [11, 11, 12, 11],
                        "cellvaluenumeric": [100] * 4, "cellpath": ["A", "A", "A", "B"]})
    source = ICUDataSource(load_src_cfg("eicu"), table_sources={"intakeoutput": lambda: raw.copy()})
    out = source.load_table("intakeoutput", columns=["patientunitstayid", "cellvaluenumeric"]).data
    assert len(out) == 3
    assert out.attrs["eicu_io_dedup_audit"]["duplicates_removed"] == 1


def test_r5_eicu_006_native_renal_module_publishes_deduplicated_balance(tmp_path, monkeypatch):
    import os
    import tempfile

    from easyicu.api.extraction import extract_database

    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    for name in ("TMPDIR", "EASYICU_DUCKDB_TEMP_DIR"):
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])
        else:
            monkeypatch.delenv(name, raising=False)
    _review_io_source(tmp_path / "source", "bucket")
    result = extract_database("eicu", data_path=str(tmp_path / "source"), output_dir=str(tmp_path / "out"),
                              modules=["outcome", "renal"], patient_ids={"patientunitstayid": [1]},
                              native_export_v2=True, verbose=False)
    assert not result["modules"]["renal"].get("errors")
    out = pd.read_parquet(tmp_path / "out/renal.parquet")
    assert out.urine.dropna().tolist() == [300.0]
    assert out.fluid_balance.dropna().tolist() == [240.0]


def test_r5_eicu_015_rate_intersections_are_vectorized_and_stop_exclusive(monkeypatch):
    from easyicu.concept.eicu_io import systemic_intake

    def no_row_loop(*args, **kwargs):
        raise AssertionError("I/O integration must not iterate over raw rows in Python")

    frame = pd.DataFrame({"patientunitstayid": [1] * 4, "intakeoutputoffset": [30, 90, 30, 90],
                          "cellpath": ["IV A (mL/hr)"] * 2 + ["IV B (mL/hr)"] * 2,
                          "cellvaluenumeric": [60, 0, 120, 0]})
    monkeypatch.setattr(pd.DataFrame, "iterrows", no_row_loop)
    out = systemic_intake(frame, "cellvaluenumeric")
    assert out.groupby("intakeoutputoffset").cellvaluenumeric.sum().to_dict() == {0: 90.0, 60: 90.0}


@pytest.mark.parametrize("duration,expected", [(360, 300.0), (361, 0.0)])
def test_r5_eicu_015_intake_rate_requires_a_next_record_within_six_hours(duration, expected):
    frame = pd.DataFrame({"patientunitstayid": [1, 1], "intakeoutputoffset": [0, duration],
                          "cellpath": ["Intake (ml)|IV (mL/hr)"] * 2, "cellvaluenumeric": [50, 0]})
    out = _apply("total_input_ml", _sources("total_input_ml")[0], frame)
    assert out.total_input_ml.sum() == expected


@pytest.mark.parametrize("name,drug", [("norepi_rate", "Norepinephrine"), ("midazolam_rate", "Midazolam")])
def test_r5_eicu_026_zero_volume_rate_stops_without_concentration(tmp_path, name, drug):
    unit = "mcg/kg/min" if name == "norepi_rate" else "mg/hr"
    frame = pd.DataFrame({"infusionoffset": [0, 60, 600], "drugname": [f"{drug} ({unit})", f"{drug} (ml/hr)", f"{drug} ({unit})"], "drugrate": [1.0, 0.0, 1.0]})
    ds = _eicu_source(tmp_path, "eicu", infusiondrug=frame)
    out = _hourly(ds, name)
    assert out[0] == 1
    assert not any(out.get(h, 0) > 0 for h in range(1, 10))
    assert out[1] == 0


def test_r5_eicu_014_suspected_infection_is_not_explicit_negation():
    frame = pd.DataFrame({"diagnosisstring": ["infectious diseases|suspected infection"]})
    assert _apply("infection_icd", _sources("infection_icd")[0], frame).infection_icd.tolist() == [True]


@pytest.mark.parametrize("middle", ["IABP weaning", "IABP removal planned"])
def test_r5_eicu_004_weaning_or_planned_removal_is_not_completed_removal(middle):
    frame = pd.DataFrame({"patientunitstayid": [1] * 3, "treatmentoffset": [0, 60, 180],
                          "treatmentstring": ["IABP insertion", middle, "IABP removal"]})
    out = _apply("mech_circ_support", _sources("mech_circ_support")[0], frame)
    assert set(out.loc[out.mech_circ_support.eq(True), "treatmentoffset"]) == {0, 60, 120}  # noqa: E712


def test_r5_eicu_034_optional_anchor_reads_declare_the_columns_they_need():
    from easyicu.concept.eicu_ventilation_windows import load_ventilation_anchors

    def load_table(name, *, columns, **kwargs):
        assert columns is not None
        return SimpleNamespace(data=pd.DataFrame(columns=columns))

    assert load_ventilation_anchors(SimpleNamespace(load_table=load_table), [1]).empty


@pytest.mark.parametrize("diagnosis", ["bacterial pericarditis", "viral enteritis", "bacterial cystitis"])
def test_r5_eicu_014_explicit_infectious_qualifiers_survive_the_inflammation_exclusion(diagnosis):
    frame = pd.DataFrame({"diagnosisstring": [diagnosis, "no " + diagnosis, diagnosis + " ruled out"]})
    out = _apply("infection_icd", _sources("infection_icd")[0], frame)
    assert out.diagnosisstring.tolist() == [diagnosis]
