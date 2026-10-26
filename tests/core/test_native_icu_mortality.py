"""Named ICU outcomes on small, schema-faithful synthetic converted sources."""
from __future__ import annotations

import pandas as pd
import pytest

from tests.support.source_mortality import source


def test_sic_documented_dispositions_preserve_survivors_and_unknowns(tmp_path):
    from easyicu.scores.icu_mortality import load_icu_death

    root = source(tmp_path / "sic", "sic")
    rows = pd.read_parquet(root / "cases.parquet").iloc[[0] * 10].reset_index(drop=True)
    rows["CaseID"] = range(1, 11)
    rows["HospitalDischargeType"] = [2026, 2028, 2028, 2028, 2027, 9999, None, 2027, 9999, None]
    rows["OffsetOfDeath"] = [None, 90000, 90001, None, None, None, None, 90000, 90001, 90000]
    rows.to_parquet(root / "cases.parquet", index=False)
    result = load_icu_death("sic", root)
    assert result.icu_death.iloc[:3].notna().all()
    assert result.icu_death.iloc[:3].tolist() == [False, True, False]
    assert result.icu_death.iloc[3:].isna().all()
    assert result.icu_death_time_hours.dropna().tolist() == [24.0]


@pytest.mark.parametrize("database", ["mimic", "miiv", "eicu", "hirid", "aumc", "sic"])
def test_named_outcome_uses_icu_status_and_keeps_unknowns(tmp_path, database):
    from easyicu.scores.icu_mortality import load_icu_death

    frame = load_icu_death(database, source(tmp_path / database, database))
    assert frame.icu_death.iloc[:3].tolist() == [True, False, False]
    assert pd.isna(frame.icu_death.iloc[3])
    assert frame.attrs["outcome_meaning"] == "icu_death"
    assert "death" not in frame


def test_mimic_iii_upper_case_headers_read_like_the_data_source(tmp_path):
    from easyicu.scores.icu_mortality import load_icu_death

    frame = load_icu_death("mimic", source(tmp_path / "mimic", "mimic", upper=True))
    assert frame.icu_death.iloc[:3].tolist() == [True, False, False]
    assert pd.isna(frame.icu_death.iloc[3])
    assert frame.attrs["source_identifier"] == "icustay_id"


@pytest.mark.parametrize("database", ["eicu", "hirid", "aumc", "sic"])
def test_public_concept_api_selects_the_named_builder(tmp_path, database):
    from easyicu import load_concepts
    from easyicu.api.concepts import clear_global_loader

    try:
        frame = load_concepts(["icu_death"], database=database,
            data_path=source(tmp_path / database, database), merge=False,
            verbose=False, concept_workers=1, parallel_workers=1)
        if isinstance(frame, dict):
            frame = frame["icu_death"]
        assert frame.icu_death.iloc[:3].tolist() == [True, False, False]
        assert pd.isna(frame.icu_death.iloc[3])
    finally:
        clear_global_loader()


def test_sic_icu_window_includes_boundaries_but_not_pre_admission(tmp_path):
    from easyicu.scores.icu_mortality import load_icu_death

    root = source(tmp_path / "sic", "sic")
    frame = pd.read_parquet(root / "cases.parquet")
    frame["OffsetOfDeath"] = [3600, 3599, 90000, 90001]
    frame["HospitalDischargeType"] = 2028
    frame.to_parquet(root / "cases.parquet", index=False)
    assert load_icu_death("sic", root).icu_death.tolist() == [True, False, True, False]


def test_hirid_monitoring_values_do_not_encode_death(tmp_path):
    from easyicu.scores.icu_mortality import load_icu_death

    frame = load_icu_death("hirid", source(tmp_path / "hirid", "hirid"))
    assert frame.icu_death_time_hours.iloc[0] == 48
    assert frame.attrs["event_time_kind"] == "last_observation_proxy"
    assert frame.icu_death_time_hours.iloc[1:].isna().all()


@pytest.mark.parametrize("database,hours", [("eicu", 24), ("aumc", 100_000_000 / 3_600_000)])
def test_icu_discharge_time_is_a_proxy_only_for_icu_deaths(tmp_path, database, hours):
    from easyicu.databases.research_capabilities import research_capability_profile
    from easyicu.scores.icu_mortality import load_icu_death

    frame = load_icu_death(database, source(tmp_path / database, database))
    profile = research_capability_profile(database)
    native = next(item for item in profile.mortality_outcomes if item.meaning == "icu_death")
    assert native.event_time == profile.icu_discharge
    assert native.time_kind == frame.attrs["event_time_kind"] == "icu_discharge_proxy"
    assert frame.icu_death_time_hours.iloc[0] == pytest.approx(hours)
    assert frame.icu_death_time_hours.iloc[1:].isna().all()


@pytest.mark.parametrize("database,table,field", [
    ("aumc", "admissions", "destination"), ("sic", "cases", "ICUOffset"),
    ("eicu", "patient", "unitdischargestatus"), ("hirid", "general", "discharge_status"),
])
def test_absent_native_source_is_an_error_not_zero_mortality(tmp_path, database, table, field):
    from easyicu.scores.icu_mortality import load_icu_death

    root = source(tmp_path / database, database)
    frame = pd.read_parquet(root / f"{table}.parquet").drop(columns=[field])
    frame.to_parquet(root / f"{table}.parquet", index=False)
    with pytest.raises(ValueError, match="icu_death_source_field_missing"):
        load_icu_death(database, root)


@pytest.mark.parametrize("database", ["hirid", "aumc", "sic", "eicu"])
def test_native_status_reaches_normal_cohort_materialization(tmp_path, database):
    from easyicu.research_agent.cohort.materializer import materialize_cohort
    from easyicu.api.concepts import clear_global_loader

    try:
        rows, receipt = materialize_cohort(feature_concepts=[], static_concepts=["age"],
            outcome_concepts=["icu_death"], database=database,
            data_path=source(tmp_path / database, database), prefer_existing=False)
        assert rows.icu_death.iloc[:3].tolist() == [True, False, False]
        assert pd.isna(rows.icu_death.iloc[3])
        assert "death" not in rows
        assert receipt["outcome_concepts"] == ["icu_death"]
    finally:
        clear_global_loader()


def test_raw_materialization_refuses_a_conflicting_canonical_identifier(monkeypatch, tmp_path):
    from easyicu import api
    from easyicu.research_agent.cohort.materializer import _load_concept

    monkeypatch.setattr(api, "load_concepts", lambda *args, **kwargs: pd.DataFrame({
        "patientid": [1, 2], "stay_id": [2, 1], "age": [45, 65]}))
    with pytest.raises(RuntimeError) as error:
        _load_concept("raw", tmp_path, "age", "hirid", None, [])
    assert str(error.value.__cause__) == "source_icu_identifier_conflict"


def test_raw_materialization_accepts_equal_identifiers_with_different_integer_widths(monkeypatch, tmp_path):
    from easyicu import api
    from easyicu.research_agent.cohort.materializer import _load_concept

    frame = pd.DataFrame({"patientid": pd.Series([1, 2], dtype="int32"),
                          "stay_id": pd.Series([1, 2], dtype="int64"), "age": [45, 65]})
    monkeypatch.setattr(api, "load_concepts", lambda *args, **kwargs: frame)
    result = _load_concept("raw", tmp_path, "age", "hirid", None, [])
    assert result.stay_id.tolist() == [1, 2]


@pytest.mark.parametrize("database,identifier", [("aumc", "admissionid"), ("sic", "CaseID"), ("eicu", "patientunitstayid")])
def test_generic_death_retains_its_distinct_dictionary_meaning(tmp_path, database, identifier):
    from easyicu import load_concepts
    from easyicu.api.concepts import clear_global_loader

    try:
        root = source(tmp_path / database, database)
        frames = load_concepts(["death", "icu_death"], database=database, data_path=root,
                              merge=False, verbose=False, concept_workers=1, parallel_workers=1)
        assert frames["death"].set_index(identifier).loc[2, "death"]
        assert not frames["icu_death"].set_index(identifier).loc[2, "icu_death"]
    finally:
        clear_global_loader()


def test_public_icu_loader_does_not_swallow_missing_source_fields(tmp_path):
    from easyicu import load_concepts
    from easyicu.api.concepts import clear_global_loader

    root = source(tmp_path / "aumc", "aumc")
    path = root / "admissions.parquet"
    pd.read_parquet(path).drop(columns=["destination"]).to_parquet(path, index=False)
    try:
        with pytest.raises(ValueError, match="icu_death_source_field_missing"):
            load_concepts(["icu_death"], database="aumc", data_path=root,
                          merge=False, verbose=False, concept_workers=1, parallel_workers=1)
    finally:
        clear_global_loader()


def test_named_icu_status_rejects_duplicate_stay_identifiers(tmp_path):
    from easyicu.scores.icu_mortality import load_icu_death

    root = source(tmp_path / "hirid", "hirid")
    path = root / "general.parquet"
    rows = pd.read_parquet(path)
    rows.loc[1, "patientid"] = 1
    rows.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_icu_death("hirid", root)


def test_sic_static_native_index_is_kept_in_icu_hours():
    from types import SimpleNamespace
    from easyicu.concept import ConceptResolver

    resolver = ConceptResolver.__new__(ConceptResolver)
    frame = pd.DataFrame({"CaseID": [1, 2], "ICUOffset": [3600, 1800], "age": [45, 65]})
    cases = frame.assign(TimeOfStay=[90000, 90000])
    raw = SimpleNamespace(config=SimpleNamespace(name="sic"), load_table=lambda *args, **kwargs: SimpleNamespace(data=cases))
    result = resolver._align_time_to_admission(frame, raw,
                                              ["CaseID"], "ICUOffset")
    assert result["ICUOffset"].tolist() == [0., 0.]
    assert result["age"].tolist() == [45, 65]


@pytest.mark.parametrize("database", ["mimic", "miiv", "eicu", "hirid", "aumc", "sic"])
@pytest.mark.parametrize("keyed", [False, True])
def test_named_outcome_keeps_the_existing_patient_filter_contract(tmp_path, database, keyed):
    from easyicu.databases.profiles import get_database_profile
    from easyicu.scores.outcomes import load_outcomes

    identity = get_database_profile(database).stay_id_col
    selected = {identity: [2, 4]} if keyed else [2, 4]
    frame = load_outcomes(database, source(tmp_path / database, database), patient_ids=selected,
                          max_patients=1, output_concepts=["icu_death"])
    assert frame[identity].tolist() == [2]
    assert frame.icu_death.tolist() == [False]
