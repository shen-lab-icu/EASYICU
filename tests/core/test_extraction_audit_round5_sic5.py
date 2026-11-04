"""Round-5 SICdb regressions; all records below are invented."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import ConceptSource, _apply_callback
import easyicu

# Resolve through the imported package so the copied tests exercise base bytes.
DATA = Path(easyicu.__file__).parent / "data"
H = 3600


def _dictionary(name="concept-dict.json"):
    dictionary = json.loads((DATA / name).read_text())
    if name == "concept-dict.json":
        for key, value in json.loads((DATA / "sofa2-dict.json").read_text()).items():
            dictionary.setdefault(key, value)
    return dictionary


class _Tables:
    config = SimpleNamespace(name="sic")

    def __init__(self, **tables):
        self.tables = tables

    def load_table(self, name, columns=None, filters=None, verbose=False):
        frame = self.tables[name].copy()
        for spec in filters or []:
            frame = frame.loc[frame[spec.column].isin(spec.value)]
        return frame if columns is None else frame[columns]


def _source(name, *, sofa=False, item=None):
    sources = _dictionary("sofa2-dict.json" if sofa else "concept-dict.json")[name].get("sources", {}).get("sic", [])
    if item is not None:
        sources = [s for s in sources if item in np.atleast_1d(s.get("ids", []))]
    assert sources, f"SICdb source missing: {name}/{item}"
    return ConceptSource.from_mapping({"index_var": "Offset", **sources[0]})


def _run(name, frame, *, source=None, data_source=None, sofa=False, item=None):
    source = source or _source(name, sofa=sofa, item=item)
    frame = frame.copy()
    value = source.value_var or {"medication": "Amount", "laboratory": "LaboratoryValue"}.get(source.table, "Val")
    if value in frame and name not in frame:
        frame = frame.rename(columns={value: name})
    return _apply_callback(frame, source, name, data_source=data_source)


def _clock(origin=1800):
    return _Tables(cases=pd.DataFrame({"CaseID": [1], "ICUOffset": [origin], "HospitalUnit": [3]}))


def test_r5_sic_002_d50_is_only_positive_single_dose_50_percent():
    source = _source("dextrose50")
    assert source.ids == [2148]  # 2045 is 33%, never D50.
    frame = pd.DataFrame({"CaseID": 1, "Offset": range(4), "Amount": [1., 1., 0., np.nan], "IsSingleDose": [0, 1, 1, 1]})
    out = _run("dextrose50", frame, source=source).set_index("Offset")
    assert 0 not in out.index
    assert out.loc[1, "dextrose50"] == 1
    assert out.loc[2, "dextrose50"] == 0
    assert pd.isna(out.loc[3, "dextrose50"])


@pytest.mark.parametrize("name,sofa,end", [("ett_gcs", False, "OffsetEnd"), ("mech_vent", False, "OffsetEnd"), ("adv_resp", True, "OffsetEnd"), ("other_vaso", True, "OffsetDrugEnd")])
def test_r5_sic_003_explicit_intervals_keep_each_positive_overlap_and_coverage(name, sofa, end):
    if name in {"adv_resp", "other_vaso"}:
        assert _dictionary("sofa2-dict.json")[name]["aggregate"] == "max"
    frame = pd.DataFrame({"CaseID": [1, 1, 1, 1], "Offset": [1800+H-60, 1800+3*H, 1800+5*H, np.nan], end: [1800+H+60, 1800+4*H, 1800+5*H, 6*H], "DataID": 720, "Amount": 1., "IsSingleDose": 0})
    out = _run(name, frame, sofa=sofa, data_source=_clock())
    assert out.Offset.tolist() == [0., 1., 3.]
    assert out.attrs["sicdb_interval_coverage"]["overlap_seconds"] == [60., 60., 3600.]
    assert out.attrs["sicdb_interval_coverage"]["interval_row"] == [0, 0, 1]
    assert out.attrs["sicdb_interval_coverage"]["overlap_start_seconds"] == [H-60., H, 3*H]
    assert out.attrs["sicdb_interval_coverage"]["overlap_stop_seconds"] == [H, H+60., 4*H]
    assert "dur_var" not in out  # already materialized, never expanded twice


def test_r5_sic_005_any_route_insulin_exposure_includes_lispro_and_nph():
    source = _source("insulin")
    assert {1557, 1848, 1961, 1962, 1964, 1975} <= set(source.ids)
    assert not {1964, 1975} & set(_source("ins").ids)


def test_r5_sic_009_infusion_amounts_follow_icu_hours_and_conserve_each_record():
    frame = pd.DataFrame({"CaseID": 1, "Offset": [1800+H-60, 1800+3*H+7], "OffsetDrugEnd": [1800+H+60, 1800+3*H+67], "Amount": [.012, .02], "IsSingleDose": [0, 1]})
    out = _run("loop_diuretic_furosemide_eq", frame, data_source=_clock()).sort_values("Offset")
    assert out.Offset.tolist() == [0., 1., 3.+7/H]
    assert out.loop_diuretic_furosemide_eq.tolist() == pytest.approx([6., 6., 20.])
    assert out.loop_diuretic_furosemide_eq.sum() == pytest.approx(32.)


def test_r5_sic_010_other_vaso_includes_positive_continuous_milrinone():
    assert _dictionary("sofa2-dict.json")["other_vaso"]["aggregate"] == "max"
    source = _source("other_vaso", sofa=True, item=1560)
    frame = pd.DataFrame({"CaseID": 1, "Offset": [1800]*3, "OffsetDrugEnd": [5400]*3, "Amount": [1., 0., 1.], "IsSingleDose": [0, 0, 1]})
    out = _run("other_vaso", frame, source=source, data_source=_clock())
    assert out.other_vaso.tolist() == [True]
    assert out.Offset.tolist() == [0.]


@pytest.mark.parametrize("name,item,amount,want", [("ins", 1557, 4., 4.), ("midazolam_rate", 1495, .006, 6.), ("dex", 2045, 3.3, 33.), ("dex", 2043, 20., 10.), ("dex", 2044, 20., 40.)])
def test_r5_sic_012_only_verified_continuous_doses_become_rates(name, item, amount, want):
    source = _source(name, item=item)
    frame = pd.DataFrame({"CaseID": 1, "DrugID": item, "Offset": [1800+H-60]*3, "OffsetDrugEnd": [1800+2*H-60]*3, "Amount": [amount, amount, 0.], "AmountPerMinute": 999., "IsSingleDose": [0, 1, 0]})
    ds = _clock()
    # Real SICdb shape: the preparation is in ReferenceValue, ReferenceName is "Fluid".
    ds.tables["d_references"] = pd.DataFrame({"ReferenceGlobalID": [item], "ReferenceUnit": ["ml" if item in {2043, 2044} else "g"], "ReferenceName": ["Fluid"], "ReferenceValue": [{2043: "Glukose 5%", 2044: "Glukose 20%"}.get(item, "synthetic")]})
    out = _run(name, frame, source=source, data_source=ds)
    assert out.Offset.tolist() == [0., 1.]
    assert out[name].tolist() == pytest.approx([want, want])


def test_r5_sic_013_fio2_source_columns_reuse_percent_conversion():
    for name, item in [("fio2_chart", 2283), ("fio2_lab", 684)]:
        source = _source(name, item=item)
        value = "Val" if name.endswith("chart") else "LaboratoryValue"
        out = _run(name, pd.DataFrame({value: [.4, 40.]}), source=source)
        assert out[name].tolist() == [40., 40.]


def test_r5_sic_015_opioid_only_is_sensitivity_with_rule_labels():
    from easyicu.concept.sicdb_ventilation import _sedated_controlled_hours, attributed_ventilation_windows

    settings = pd.DataFrame({"CaseID": 1, "DataID": 2282, "Offset": [0, H, 2*H]})
    ds = _Tables(cases=pd.DataFrame({"CaseID": [1], "HospitalUnit": [3], "ICUOffset": [0]}), data_float_h=pd.DataFrame({"CaseID": 1, "DataID": [3059, 3057, 716], "Offset": [0, H, 2*H], "Val": [.1, 1., 35.]}))
    out = _sedated_controlled_hours(settings, ds, [1], "CaseID", "Offset")
    assert out._h.tolist() == [1., 2.]
    assert out.sicdb_ventilation_rule.tolist() == ["propofol_controlled", "etco2_controlled"]
    assert out.attrs["sicdb_opioid_only_hours"] == [{"CaseID": 1, "_h": 0.0, "sicdb_ventilation_rule": "opioid_only_sensitivity"}]
    airway = pd.DataFrame({"CaseID": [1], "Offset": [H], "OffsetEnd": [2*H]})
    windows = attributed_ventilation_windows(settings, airway, concept_name="mech_vent", sedated_hours=out)
    rules = windows.attrs["sicdb_ventilation_attribution"]
    assert [r["sicdb_ventilation_rule"] for r in rules] == ["airway_setting_tolerance", "airway_setting_overlap", "airway_setting_tolerance"]
    assert windows.attrs["sicdb_ventilation_coverage_delta"]["additional_opioid_only_sensitivity_hours"] == 0


@pytest.mark.parametrize("name,raw,want", [("peep", 10., 10.*100/98.0665), ("peep_set", 10., 10.*100/98.0665), ("pip", 20., 20.*100/98.0665), ("ps", 10., 10.*100/98.0665), ("mean_airway_pres", 15., 15.*100/98.0665), ("compliance", 50., 50.*98.0665/100)])
def test_r5_sic_016_mbar_values_are_converted_before_canonical_bounds(name, raw, want):
    out = _run(name, pd.DataFrame({"Val": [raw]}))
    assert out[name].item() == pytest.approx(want)


def test_r5_sic_017_total_protein_pools_293_and_294_by_median():
    source = _source("total_protein", item=293)
    assert set(source.ids) == {293, 294}
    assert _dictionary()["total_protein"].get("aggregate", "median") == "median"
    assert source.params["source_label"] == "LaboratoryID"


@pytest.mark.parametrize("name,raw,want", [("norepi_rate", .000001, .0125), ("dobu_rate", .000001, .0125), ("epi_rate", .000001, .0125), ("phn_rate", .000001, .0125), ("dopa_rate", .000001, .0125), ("adh_rate", .03, .03)])
def test_r5_sic_018_exact_stop_is_excluded_partial_stop_is_kept(name, raw, want):
    frame = pd.DataFrame({"CaseID": 1, "Offset": [1800, 1800+H], "OffsetDrugEnd": [1800+H, 1800+2*H+60], "AmountPerMinute": [raw, raw*2], "IsSingleDose": 0, "weight": 80.})
    out = _run(name, frame, data_source=_clock())
    assert out.Offset.tolist() == [0., 1., 2.]
    assert out[name].tolist() == pytest.approx([want, want*2, want*2])


EXPOSURES = "abx aspirin bicarbonate calcium_iv cort delirium_tx_proxy dextrose50 enoxaparin esmolol fentanyl heparin insulin midazolam morphine neostigmine nitroglycerin pantoprazole phenytoin potassium_iv potassium_sparing_diuretic propofol rocuronium thiazide_diuretic warfarin".split()


@pytest.mark.parametrize("name", EXPOSURES)
def test_r5_sic_026_zero_is_false_missing_is_unknown_positive_wins(name):
    for source in _dictionary()[name]["sources"]["sic"]:
        if source["table"] != "medication":
            continue
        source = ConceptSource.from_mapping(source)
        frame = pd.DataFrame({"Amount": [0., np.nan, 1., -1., np.inf, -np.inf], "Offset": [0, 1, 0, 3, 4, 5], "IsSingleDose": 1})
        out = _run(name, frame, source=source)
        assert out[name].iloc[0] == 0
        assert pd.isna(out[name].iloc[1])
        assert out[name].iloc[2] == 1
        assert pd.isna(out[name].iloc[3])
        assert out[name].iloc[4:].isna().all()
        assert out.groupby("Offset")[name].max().loc[0] == 1
        assert _dictionary()[name].get("aggregate") == "max"


def test_r5_sic_001_raw_cvp_and_reasons_survive_the_conservative_evidence_filter():
    from easyicu.concept.callbacks import ConceptCallbackContext, _callback_cvp_artefact_filter
    from easyicu.table import ICUTable

    cvp = pd.DataFrame({"CaseID": 1, "Offset": [0., 1., 2., 3., 4., 5., 6.], "cvp_raw": [20., 40., 20., 38., 39., 60., 35.]})
    ds = _clock(0)
    ds.tables["data_float_h"] = pd.DataFrame({"CaseID": 1, "DataID": [702]*6+[705], "Offset": np.arange(7)*H, "Val": [70., 70., 70., 40., 80., 80., 80.]})
    ctx = ConceptCallbackContext(concept_name="cvp", target=None, interval=pd.Timedelta(hours=1), resolver=None, data_source=ds, patient_ids=None)
    table = ICUTable(cvp, id_columns=["CaseID"], index_column="Offset", value_column="cvp_raw")
    out = _callback_cvp_artefact_filter({"cvp_raw": table}, ctx).data
    assert out.Offset.tolist() == [0., 2., 4.]
    qc = out.attrs["sicdb_cvp_qc"]
    assert qc["cvp_raw"] == cvp.cvp_raw.tolist()
    assert qc["reason"] == ["kept", "isolated_spike", "kept", "arterial_diastolic_match", "kept", "above_50", "arterial_evidence_unavailable_cap_30"]
    from easyicu.resources import load_dictionary

    assert load_dictionary(source_name="sic")["cvp_raw"].maximum is None
    assert load_dictionary(source_name="sic")["cvp_raw"].minimum is None
    assert load_dictionary(source_name="miiv")["cvp_raw"].maximum == 50
    from easyicu.api.extraction import _enforce_native_export_concept_bounds

    for database, excluded in [("sic", 0), ("miiv", 2)]:
        raw = pd.DataFrame({"cvp_raw": [-8., 60.]})
        audit = _enforce_native_export_concept_bounds(raw, requested_concepts=["cvp_raw"], dictionary=load_dictionary(source_name=database), database=database)
        assert audit["cvp_raw"]["excluded_out_of_bounds"] == excluded
        assert raw.cvp_raw.isna().all() == bool(excluded)


def test_r5_sic_007_legacy_mass_rate_requires_the_same_icu_clock():
    frame = pd.DataFrame({"CaseID": [1], "Offset": [1800], "OffsetDrugEnd": [5400], "Amount": [.006], "IsSingleDose": [0]})
    source = ConceptSource.from_mapping({"table": "medication", "val_var": "Amount", "index_var": "Offset", "stop_var": "OffsetDrugEnd", "callback": 'sic_rate_mass(target_unit = "mg/hour")'})
    with pytest.warns(DeprecationWarning):
        out = _run("legacy", frame, source=source, data_source=_clock())
    assert out.Offset.tolist() == [0.]
    assert out.legacy.tolist() == pytest.approx([6.])


def test_r5_sic_008_public_sofa_expires_at_the_twenty_fourth_cell():
    from easyicu.callbacks import sofa_score

    frame = pd.DataFrame({"stay_id": 1, "time": [0., 24., 48.], "sofa_resp": [4., np.nan, 1.]})
    out = sofa_score({"sofa_resp": frame}, keep_components=True).set_index("time")
    assert pd.isna(out.loc[24., "sofa_resp"])
    assert out.loc[48., "sofa"] == 1


def test_r5_sic_011_minute_volume_keeps_raw_and_labels_standby_values():
    frame = pd.DataFrame({"Val": [.01, .99, 1., 7.]})
    out = _run("minute_vol", frame)
    assert out.minute_vol.isna().tolist() == [True, True, False, False]
    assert out.attrs["sicdb_minute_volume_qc"] == {"raw": [.01, .99, 1., 7.], "reason": ["below_analysis_min_1", "below_analysis_min_1", "kept", "kept"]}
    missing = _run("minute_vol", pd.DataFrame({"Val": [np.nan]}))
    assert missing.attrs["sicdb_minute_volume_qc"]["reason"] == ["missing"]


def test_r5_sic_003_public_resolver_preserves_boundary_hours_raw_qc_and_attribution(tmp_path):
    from easyicu.concept import ConceptResolver
    from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL
    from easyicu.config import load_src_cfg
    from easyicu.datasource import ICUDataSource
    from easyicu.resources import load_dictionary

    companions = {"cvp_qc", "minute_vol_raw", "minute_vol_qc", "mech_vent_rule", "mech_vent_opioid_sensitivity"}
    assert companions <= _dictionary().keys()
    assert companions | {"cvp_raw"} <= {c for group in CONCEPT_GROUPS_INTERNAL.values() for c in group}
    pd.DataFrame({"CaseID": [1, 2], "PatientID": [11, 12], "ICUOffset": [1800, 1800], "TimeOfStay": [20*H, 20*H], "HospitalUnit": [3, 3]}).to_parquet(tmp_path / "cases.parquet")
    pd.DataFrame({"CaseID": [1], "DataID": [720], "Offset": [1800+H-60], "OffsetEnd": [1800+H+60], "Data": ["synthetic"]}).to_parquet(tmp_path / "data_range.parquet")
    rows = [(1, 2018, 0, 40.), (1, 2018, 1, 42.), (1, 2018, 2, 60.), (1, 702, 0, 80.), (1, 702, 1, 80.), (1, 713, 0, .01), (1, 713, 1, 7.), (2, 2282, 1, 12.), (2, 3059, 1, .1)]
    events = pd.DataFrame(rows, columns=["CaseID", "DataID", "Offset", "Val"])
    events.Offset = 1800 + events.Offset*H
    events.to_parquet(tmp_path / "data_float_h.parquet")
    source = ICUDataSource(load_src_cfg("sic"), base_path=tmp_path, enable_cache=False)
    names = ["mech_vent", "vent_ind", "ett_gcs", "adv_resp", "cvp", "cvp_raw", "minute_vol", *sorted(companions)]
    out = ConceptResolver(load_dictionary(include_sofa2=True, source_name="sic")).load_concepts(
        names, source, merge=False, r_compatible=False, interval=pd.Timedelta(hours=1),
        patient_ids={"CaseID": [1, 2]}, concept_workers=1, verbose=False,
    )
    for name in ["mech_vent", "vent_ind", "ett_gcs", "adv_resp"]:
        frame = out[name].data
        clock = out[name].index_column
        positive = frame[name].notna() if name == "mech_vent" else frame[name].eq(True)
        assert frame.loc[positive, ["CaseID", clock]].values.tolist() == [[1., 0.], [1., 1.]], name
    coverage = out["mech_vent"].data.attrs["sicdb_interval_coverage"]
    assert coverage["source_index"] == [0, 0]
    assert coverage["overlap_start_seconds"] == [H-60., H]
    assert coverage["overlap_stop_seconds"] == [H, H+60.]
    assert out["cvp_raw"].data.cvp_raw.tolist() == [40., 42., 60.]
    assert out["cvp"].data.cvp.tolist() == [40., 42.]
    assert "above_50" in out["cvp_qc"].data.cvp_qc.tolist()
    assert out["minute_vol_raw"].data.minute_vol_raw.tolist() == [.01, 7.]
    assert "below_analysis_min_1" in out["minute_vol_qc"].data.minute_vol_qc.tolist()
    assert set(out["mech_vent_rule"].data.mech_vent_rule) == {"explicit_ett"}
    sensitivity = out["mech_vent_opioid_sensitivity"].data
    assert sensitivity.CaseID.tolist() == [2]
    assert sensitivity.mech_vent_opioid_sensitivity.tolist() == [True]


def test_r5_sic_012_unknown_glucose_unit_is_explicitly_unavailable():
    from easyicu.concept.errors import ConceptExtractionUnavailable

    ds = _clock()
    ds.tables["d_references"] = pd.DataFrame({"ReferenceGlobalID": [2045], "ReferenceUnit": ["unknown"], "ReferenceName": ["Fluid"], "ReferenceValue": ["Glukose 33%"]})
    frame = pd.DataFrame({"CaseID": [1], "DrugID": [2045], "Offset": [1800], "OffsetDrugEnd": [5400], "IsSingleDose": [0], "Amount": [1.]})
    with pytest.raises(ConceptExtractionUnavailable, match="reference amount unit"):
        _run("dex", frame, data_source=ds)


def test_r5_sic_012_unknown_glucose_concentration_is_never_guessed_from_amount():
    from easyicu.concept.errors import ConceptExtractionUnavailable

    ds = _clock()
    ds.tables["d_references"] = pd.DataFrame({"ReferenceGlobalID": [2043], "ReferenceUnit": ["g"], "ReferenceName": ["Fluid"], "ReferenceValue": ["Glukose solution"]})
    frame = pd.DataFrame({"CaseID": [1], "DrugID": [2043], "Offset": [1800], "OffsetDrugEnd": [5400], "IsSingleDose": [0], "Amount": [10.]})
    with pytest.raises(ConceptExtractionUnavailable, match="known concentration"):
        _run("dex", frame, data_source=ds)


def test_r5_sic_003_invalid_airway_intervals_do_not_attribute_settings():
    from easyicu.concept.sicdb_ventilation import attributed_ventilation_windows

    settings = pd.DataFrame({"CaseID": [1], "Offset": [0]})
    airway = pd.DataFrame({"CaseID": [1, 1, 1], "Offset": [0., 60., 0.], "OffsetEnd": [0., 0., np.inf]})
    assert attributed_ventilation_windows(settings, airway, concept_name="mech_vent").empty


def test_r5_sic_016_public_pressure_bounds_follow_conversion_and_protein_is_median(tmp_path):
    from easyicu.concept import ConceptResolver
    from easyicu.config import load_src_cfg
    from easyicu.datasource import ICUDataSource
    from easyicu.resources import load_dictionary

    pd.DataFrame({"CaseID": [1], "PatientID": [11], "ICUOffset": [21600], "TimeOfStay": [21600+20*H]}).to_parquet(tmp_path / "cases.parquet")
    pd.DataFrame({"CaseID": 1, "DataID": [2278, 2278, 2279, 2279, 3120, 3120], "Offset": [21600, 25200]*3, "Val": [39.5, 30., 59., 50., 203., 50.]}).to_parquet(tmp_path / "data_float_h.parquet")
    pd.DataFrame({"CaseID": [1, 1], "LaboratoryID": [293, 294], "Offset": [21600, 21600], "LaboratoryValue": [6., 8.], "LaboratoryType": [2296, 2296]}).to_parquet(tmp_path / "laboratory.parquet")
    ds = ICUDataSource(load_src_cfg("sic"), base_path=tmp_path, enable_cache=False)
    names = ["peep", "peep_set", "mean_airway_pres", "compliance", "total_protein"]
    out = ConceptResolver(load_dictionary(source_name="sic")).load_concepts(names, ds, merge=False, r_compatible=False, interval=pd.Timedelta(hours=1), concept_workers=1, verbose=False)
    for name in ["peep", "peep_set", "mean_airway_pres"]:
        assert len(out[name].data) == 1  # first value crosses the converted bound
    assert out["compliance"].data.compliance.tolist() == pytest.approx([203.*98.0665/100, 50.*98.0665/100])
    assert out["total_protein"].data.total_protein.tolist() == [7.]
    assert out["total_protein"].data.attrs["sicdb_source_labels"] == [293, 294]
    labelled = _run("total_protein", pd.DataFrame({"LaboratoryID": [293, 294], "LaboratoryValue": [6., 8.]}))
    assert labelled.attrs["sicdb_source_labels"] == [293, 294]


def test_r5_sic_015_unavailable_text_receipts_keep_nullable_string_dtype(tmp_path):
    from easyicu.concept import ConceptResolver
    from easyicu.config import load_src_cfg
    from easyicu.resources import load_dictionary

    ds = SimpleNamespace(config=load_src_cfg("eicu"))
    frame = ConceptResolver(load_dictionary())._load_single_concept("cvp_qc", ds, aggregator=False, patient_ids=None).data
    assert isinstance(frame.cvp_qc.dtype, pd.StringDtype)
    frame = frame.reindex([0])
    path = tmp_path / "synthetic-qc.parquet"
    frame.to_parquet(path)
    pd.testing.assert_series_equal(frame.cvp_qc, pd.read_parquet(path).cvp_qc)


def test_r5_sic_018_corrected_rates_feed_equivalents_sixty_minute_rates_and_scores(tmp_path):
    from easyicu.concept import ConceptResolver
    from easyicu.config import load_src_cfg
    from easyicu.datasource import ICUDataSource
    from easyicu.resources import load_dictionary

    pd.DataFrame({"CaseID": [1], "PatientID": [11], "ICUOffset": [1800], "TimeOfStay": [1800+20*H], "WeightOnAdmission": [80000.]}).to_parquet(tmp_path / "cases.parquet")
    raw = np.asarray([.01, .15])*80/1e6
    pd.DataFrame({"CaseID": 1, "DrugID": 1562, "Offset": [1800, 1800+H], "OffsetDrugEnd": [1800+H, 1800+2*H+60], "AmountPerMinute": raw, "Amount": raw*60, "IsSingleDose": 0}).to_parquet(tmp_path / "medication.parquet")
    pd.DataFrame({"CaseID": [1], "DataID": [703], "Offset": [1800], "Val": [80.]}).to_parquet(tmp_path / "data_float_h.parquet")
    ds = ICUDataSource(load_src_cfg("sic"), base_path=tmp_path, enable_cache=False)
    names = ["norepi_rate", "norepi_equiv", "norepi60", "sofa_cardio"]
    out = ConceptResolver(load_dictionary(include_sofa2=True, source_name="sic")).load_concepts(names, ds, merge=False, r_compatible=False, interval=pd.Timedelta(hours=1), concept_workers=1, verbose=False)
    for name in names:
        frame = out[name].data
        hour = pd.to_numeric(frame[out[name].index_column], errors="coerce")
        assert frame.loc[hour.eq(1), name].item() == pytest.approx(4. if name == "sofa_cardio" else .15)


def _review_source(tmp_path, events, *, bucketed=False):
    from easyicu.config import load_src_cfg
    from easyicu.datasource import ICUDataSource

    pd.DataFrame({"CaseID": [1], "PatientID": [11], "ICUOffset": [1800],
                  "TimeOfStay": [1800+20*H], "HospitalUnit": [3]}).to_parquet(tmp_path / "cases.parquet")
    path = tmp_path / "data_float_h.parquet"
    if bucketed:
        path = tmp_path / "data_float_h" / "part-0.parquet"
        path.parent.mkdir()
    events = pd.DataFrame(events, columns=["CaseID", "DataID", "Offset", "Val"])
    events.Offset = 1800 + events.Offset*H
    events.to_parquet(path)
    return ICUDataSource(load_src_cfg("sic"), base_path=tmp_path, enable_cache=False)


@pytest.mark.parametrize("selected_dictionary", [False, True])
@pytest.mark.parametrize("mode", ["single", "bucketed", "multi"])
def test_r5_sic_001_every_load_path_preserves_rejected_raw_values(tmp_path, monkeypatch, selected_dictionary, mode):
    from easyicu.concept import ConceptResolver
    from easyicu.resources import load_dictionary
    import easyicu.datasource as datasource

    ds = _review_source(tmp_path, [(1, 2018, 0, 60.), (1, 2018, 1, -8.),
                                   (1, 2282, 0, 20.), (1, 719, 0, 16.)], bucketed=mode != "single")
    fast_names = set()
    original = datasource.load_bucketed_table_multi_aggregated

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        fast_names.update(args[2])
        return result

    monkeypatch.setattr(datasource, "load_bucketed_table_multi_aggregated", record)
    dictionary = load_dictionary(source_name="sic" if selected_dictionary else None)
    resolver = ConceptResolver(dictionary)
    names = ["cvp_raw"] if mode != "multi" else ["cvp_raw", "vent_rate", "resp"]
    out = resolver.load_concepts(names, ds, merge=False, r_compatible=False,
                                interval=pd.Timedelta(hours=1), concept_workers=1, verbose=False)
    assert out["cvp_raw"].data.cvp_raw.tolist() == [60., -8.]
    if mode == "multi":
        assert fast_names == {"vent_rate", "resp"}
        assert out["vent_rate"].data.vent_rate.tolist() == [20.]
        assert out["resp"].data.resp.tolist() == [16.]


def test_r5_sic_001_qc_uses_the_same_hourly_values_as_cvp(tmp_path):
    from easyicu.concept import ConceptResolver
    from easyicu.resources import load_dictionary

    # The first source row resembles arterial pressure, but the hourly
    # median is 30. Its QC label must describe the published hourly value.
    rows = [(1, 2018, 0, 40.), (1, 2018, .1, 20.),
            (1, 702, 0, 40.), (1, 2018, 1, 60.), (1, 2018, 2, -8.)]
    ds = _review_source(tmp_path, rows)
    out = ConceptResolver(load_dictionary(source_name="sic")).load_concepts(
        ["cvp", "cvp_raw", "cvp_qc"], ds, merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), concept_workers=1, verbose=False,
    )
    assert out["cvp"].data.cvp.tolist() == [30.]
    assert out["cvp_qc"].data.cvp_qc.tolist() == ["kept", "above_50", "below_minus_5"]


def test_r5_sic_007_legacy_dispatch_keeps_continuous_filter_evidence():
    frame = pd.DataFrame({"CaseID": [1, 1], "Offset": [1800, 1800],
                          "OffsetDrugEnd": [5400, 5400], "Amount": [.006, .012], "IsSingleDose": [0, 1]})
    source = ConceptSource.from_mapping({"table": "medication", "val_var": "Amount", "index_var": "Offset",
                                        "stop_var": "OffsetDrugEnd", "extra_vars": ["IsSingleDose"],
                                        "callback": 'sic_rate_mass(target_unit = "mg/hour")'})
    with pytest.warns(DeprecationWarning):
        out = _run("legacy", frame, source=source, data_source=_clock())
    assert out.Offset.tolist() == [0.]
    assert out.legacy.tolist() == pytest.approx([6.])


def test_r5_sic_001_custom_dictionary_cannot_relax_other_database_publication_guard():
    from easyicu.api.extraction import _enforce_native_export_concept_bounds
    from easyicu.resources import load_dictionary

    dictionary = load_dictionary(source_name="miiv")
    dictionary["cvp_raw"].minimum = dictionary["cvp_raw"].maximum = None
    frame = pd.DataFrame({"cvp_raw": [-8., 60.], "cvp": [-8., 60.]})
    audit = _enforce_native_export_concept_bounds(frame, requested_concepts=["cvp_raw", "cvp"], dictionary=dictionary)
    assert audit["cvp_raw"]["excluded_out_of_bounds"] == 2
    assert audit["cvp"]["excluded_out_of_bounds"] == 2
    assert frame.isna().all().all()


def test_r5_sic_015_empty_hourly_airway_result_does_not_fall_back_to_six_hours():
    from easyicu.concept.callbacks import ConceptCallbackContext, _callback_vent_ind
    from easyicu.table import ICUTable

    frame = pd.DataFrame({"CaseID": pd.Series(dtype=int), "charttime": pd.Series(dtype=float),
                          "mech_vent": pd.Series(dtype="string")})
    ctx = ConceptCallbackContext(concept_name="vent_ind", target=None, interval=pd.Timedelta(hours=1),
                                 resolver=None, data_source=_clock(), patient_ids=None)
    table = ICUTable(frame, id_columns=["CaseID"], index_column="charttime", value_column="mech_vent")
    assert _callback_vent_ind({"mech_vent": table}, ctx).data.empty


@pytest.mark.parametrize("bucketed", [False, True])
def test_r5_sic_016_all_sic_pressure_channels_share_canonical_units_in_batches(tmp_path, monkeypatch, bucketed):
    from easyicu.concept import ConceptResolver
    from easyicu.resources import load_dictionary
    import easyicu.datasource as datasource

    rows = [(1, item, 0, value) for item, value in [(2278, 10.), (715, 20.), (2284, 12.),
            (2279, 15.), (3120, 50.), (2283, .4), (2282, 20.), (719, 18.)]]
    ds = _review_source(tmp_path, rows, bucketed=bucketed)
    calls = []
    original = datasource.load_bucketed_table_aggregated

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(kwargs.get("convert_unit_factor"))
        return result

    monkeypatch.setattr(datasource, "load_bucketed_table_aggregated", record)
    names = ["peep", "peep_set", "pip", "ps", "mean_airway_pres", "compliance", "fio2_chart", "vent_rate", "resp"]
    out = ConceptResolver(load_dictionary()).load_concepts(names, ds, merge=False, r_compatible=False,
                                                         interval=pd.Timedelta(hours=1), concept_workers=1, verbose=False)
    for name, raw in [("peep", 10.), ("peep_set", 10.), ("pip", 20.), ("ps", 12.), ("mean_airway_pres", 15.)]:
        assert out[name].data[name].tolist() == pytest.approx([raw*100/98.0665])
    assert out["compliance"].data.compliance.tolist() == pytest.approx([50.*98.0665/100])
    assert out["fio2_chart"].data.fio2_chart.tolist() == [40.]
    if bucketed:
        assert 100/98.0665 in calls and .980665 in calls


def test_r5_sic_026_public_medication_loads_keep_zero_unknown_and_positive_hours(tmp_path):
    from easyicu.concept import ConceptResolver
    from easyicu.resources import load_dictionary

    ds = _review_source(tmp_path, [(1, 3000, 0, 70.)])
    pd.DataFrame({"CaseID": 1, "DrugID": [1975]*4, "Offset": [1800, 1800+60, 1800+H, 1800+2*H],
                  "OffsetDrugEnd": [1860, 1920, 5460, 9060], "Amount": [0., 1., 0., np.nan],
                  "AmountPerMinute": 0., "IsSingleDose": 1}).to_parquet(tmp_path / "medication.parquet")
    out = ConceptResolver(load_dictionary()).load_concepts(
        ["insulin", "ins", "dextrose50"], ds, merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), concept_workers=1, verbose=False)
    values = out["insulin"].data.set_index(out["insulin"].index_column).insulin
    assert values.loc[0] == 1
    assert values.loc[1] == 0
    assert 2 not in values.index or pd.isna(values.loc[2])
    assert out["ins"].data.empty


def test_r5_sic_015_static_catalog_publishes_exact_audit_companions():
    from tests.support.node import run_node

    script = DATA.parent / "webserver/static/js/data-catalog.js"
    # Evaluate the shipped bootstrap, including its actual public names.
    code = 'global.window = {}; require(process.argv[1]); console.log(JSON.stringify(window));'
    result = run_node("node", code, str(script), check=True)
    catalog = json.loads(result.stdout)["EU_CATALOG"]
    expected = {
        "cvp_raw": ("vitals", "mmHg", 6), "cvp_qc": ("vitals", "category", 1),
        "minute_vol_raw": ("ventilator", "L/min", 1), "minute_vol_qc": ("ventilator", "category", 1),
        "mech_vent_rule": ("respiratory", "category", 1),
        "mech_vent_opioid_sensitivity": ("respiratory", "boolean", 1),
    }
    for name, (module, unit, databases) in expected.items():
        assert catalog["groupConcepts"][module].count(name) == 1
        assert catalog["dict"][name][2] == unit
        assert catalog["cov"][name] == databases
        assert catalog["conceptCoverage"][name] == {"kind": "audited", "databases": databases, "basis": "CONCEPT_DB_COVERAGE"}
    for name, databases in {"ins": 6, "dex": 6, "midazolam_rate": 6, "fio2_chart": 3, "fio2_lab": 3}.items():
        assert catalog["cov"][name] == databases


def test_r5_sic_001_unobserved_cvp_returns_empty_instead_of_breaking_module():
    from easyicu.concept.callbacks import ConceptCallbackContext, _callback_cvp_artefact_filter
    from easyicu.table import ICUTable

    table = ICUTable(pd.DataFrame({"CaseID": [1], "Offset": [0.], "dbp": [75.]}),
                     id_columns=["CaseID"], index_column="Offset", value_column="dbp")
    ctx = ConceptCallbackContext(concept_name="cvp", target=None, interval=pd.Timedelta(hours=1),
                                 resolver=None, data_source=_clock(), patient_ids=None)
    out = _callback_cvp_artefact_filter({"dbp": table}, ctx).data
    assert out.empty and "cvp" in out


def test_r5_sic_002_nullable_single_dose_flag_accepts_numeric_one():
    out = _run("dextrose50", pd.DataFrame({"Amount": [1., 1., 1.], "Offset": [0., 1., 2.],
                                         "IsSingleDose": [1., 0., np.nan]}))
    assert out.Offset.tolist() == [0.]
    assert out.dextrose50.tolist() == [True]


def test_r5_sic_011_hourly_qc_retains_flags_from_both_minute_volume_sources(tmp_path):
    from easyicu.concept import ConceptResolver
    from easyicu.resources import load_dictionary

    ds = _review_source(tmp_path, [(1, 713, 0, .01), (1, 2019, 0, 7.),
                                   (1, 713, 1, 10.), (1, 2019, 1, 12.)])
    out = ConceptResolver(load_dictionary()).load_concepts(
        ["minute_vol", "minute_vol_raw", "minute_vol_qc"], ds, merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), concept_workers=1, verbose=False)
    assert out["minute_vol"].data.minute_vol.tolist() == [7., 11.]
    assert out["minute_vol_raw"].data.minute_vol_raw.tolist() == [3.505, 11.]
    assert out["minute_vol_qc"].data.minute_vol_qc.tolist() == ["below_analysis_min_1|kept", "kept"]


def test_r5_sic_012_glucose_concentration_reads_the_reference_value_column():
    # As in SICdb 1.0.x: 2.5 % and 10 % glucose are named only in ReferenceValue.
    ds = _clock()
    ds.tables["d_references"] = pd.DataFrame({"ReferenceGlobalID": [2043, 2044], "ReferenceUnit": ["ml", "ml"], "ReferenceName": ["Fluid", "Fluid"], "ReferenceValue": ["Glukose 2,5%", "Glukose 10%"]})
    frame = pd.DataFrame({"CaseID": 1, "DrugID": [2043, 2044], "Offset": [1800+H-60]*2, "OffsetDrugEnd": [1800+2*H-60]*2, "Amount": [40., 20.], "IsSingleDose": [0, 0]})
    out = _run("dex", frame, source=_source("dex", item=2043), data_source=ds)
    # 40 ml of 2.5 % = 10 ml of 10 %; 20 ml of 10 % = 20 ml of 10 %.
    assert out.groupby("Offset")["dex"].sum().tolist() == pytest.approx([30., 30.])
