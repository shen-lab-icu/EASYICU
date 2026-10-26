"""Invented MIMIC-IV records exercise the shipped age mapping without ICU data."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept import ConceptResolver, _apply_callback
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource, _close_duckdb_connections
from easyicu.resources import load_dictionary


@pytest.fixture
def records():
    return {
        "patients": pd.DataFrame({
            "subject_id": [1, 2, 3], "anchor_age": [60, 91, 88],
            "anchor_year": [2150, 2150, 2150], "gender": ["F", "M", "F"],
        }),
        "icustays": pd.DataFrame({
            "subject_id": [1, 1, 2, 3], "hadm_id": [10, 10, 20, 30],
            "stay_id": [101, 102, 201, 301],
            "intime": pd.to_datetime(["2155-01-01", "2156-01-01", "2155-01-01", "2155-01-01"]),
            "outtime": pd.to_datetime(["2155-01-02", "2156-01-02", "2155-01-02", "2155-01-02"]),
        }),
        "admissions": pd.DataFrame({
            "subject_id": [1, 2, 3], "hadm_id": [10, 20, 30],
            "admittime": pd.to_datetime(["2154-12-31"] * 3),
            "dischtime": pd.to_datetime(["2156-01-03"] * 3),
        }),
    }


def _source(records, wrapped):
    def load_table(name, columns=None, filters=(), **kwargs):
        frame = records[name].copy()
        for spec in filters or ():
            frame = frame[frame[spec.column].isin(spec.value)]
        if columns:
            frame = frame[columns]
        return SimpleNamespace(data=frame) if wrapped else frame

    return SimpleNamespace(config=load_src_cfg("miiv"), load_table=load_table)


@pytest.mark.parametrize("database", ["miiv", "miiv_demo"])
def test_capability_declares_whole_year_resolution_and_the_privacy_top_code(database):
    from easyicu.databases.research_capabilities import research_capability_profile

    age = research_capability_profile(database).age
    assert age.kind == "capped_or_shifted"
    assert age.exact_through == 89
    assert "exact to whole years" in age.field.interpretation
    assert "year(icustays.intime) - anchor_year" in age.field.interpretation
    assert "remains 91, not an exact age" in age.field.interpretation


def test_demo_inherits_the_same_age_callback():
    config = SimpleNamespace(name="miiv_demo", class_prefix=["miiv"])
    assert load_dictionary()["age"].for_data_source(config)[0].callback == "transform_fun(miiv_age)"


@pytest.mark.parametrize("wrapped", [False, True])
def test_age_uses_each_icu_admission_year_and_preserves_the_anchor_top_code(records, wrapped):
    frame = records["icustays"][["subject_id", "stay_id"]].merge(
        records["patients"][["subject_id", "anchor_age"]], on="subject_id"
    ).rename(columns={"anchor_age": "age"})
    before = frame.copy(deep=True)
    result = _apply_callback(frame, load_dictionary()["age"].sources["miiv"][0],
                             concept_name="age", data_source=_source(records, wrapped))
    assert result.age.tolist() == [65, 66, 91, 93]
    pd.testing.assert_frame_equal(result.drop(columns="age"), before.drop(columns="age"))
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("column", ["anchor_year", "intime"])
def test_missing_age_evidence_never_falls_back_to_anchor_age_or_hospital_time(records, column):
    table = "patients" if column == "anchor_year" else "icustays"
    records[table].loc[0, column] = None
    frame = pd.DataFrame({"subject_id": [1], "stay_id": [101], "age": [60]})
    result = _apply_callback(frame, load_dictionary()["age"].sources["miiv"][0],
                             concept_name="age", data_source=_source(records, True))
    assert result.age.isna().all()


def test_nullable_missing_anchor_age_is_not_a_privacy_top_code(records):
    frame = pd.DataFrame({"subject_id": [1], "stay_id": [101],
                          "age": pd.Series([pd.NA], dtype="Int64")})
    result = _apply_callback(frame, load_dictionary()["age"].sources["miiv"][0],
                             concept_name="age", data_source=_source(records, True))
    assert result.age.isna().all()


def test_age_callback_requires_source_evidence():
    with pytest.raises(ValueError, match="anchor_year.*intime"):
        _apply_callback(pd.DataFrame({"stay_id": [101], "age": [60]}),
                        load_dictionary()["age"].sources["miiv"][0], concept_name="age")


@pytest.mark.parametrize("table", ["patients", "icustays"])
def test_ambiguous_source_keys_fail_closed(records, table):
    records[table] = pd.concat([records[table], records[table].iloc[:1]])
    with pytest.raises(pd.errors.MergeError):
        _apply_callback(pd.DataFrame({"subject_id": [1], "stay_id": [101], "age": [60]}),
                        load_dictionary()["age"].sources["miiv"][0], concept_name="age",
                        data_source=_source(records, True))


@pytest.mark.parametrize("stay_ids", [None, [102]])
def test_public_resolver_preserves_stay_selection_and_other_demographics(records, tmp_path, stay_ids):
    source = ICUDataSource(load_src_cfg("miiv"), base_path=tmp_path,
                           table_sources={name: (lambda frame=frame: frame.copy())
                                          for name, frame in records.items()})
    resolver = ConceptResolver(load_dictionary())
    try:
        result = resolver.load_concepts(
            ["age", "sex"], source, merge=True, patient_ids=stay_ids,
            verbose=False, concept_workers=1,
        ).sort_values("stay_id").reset_index(drop=True)
        assert result.stay_id.tolist() == ([102] if stay_ids else [101, 102, 201, 301])
        assert result.age.tolist() == ([66] if stay_ids else [65, 66, 91, 93])
        assert result.sex.tolist() == (["Female"] if stay_ids else ["Female", "Female", "Male", "Female"])
    finally:
        source.clear_cache()
        _close_duckdb_connections()


def test_parquet_public_load_changes_only_age_from_the_previous_mapping(records, tmp_path, monkeypatch):
    from easyicu.base import BaseICULoader

    monkeypatch.setenv("EASYICU_DUCKDB_THREADS", "1")
    for name, frame in records.items():
        frame.to_parquet(tmp_path / f"{name}.parquet", index=False)
    dictionary = json.loads((Path(__file__).parents[2] / "src/easyicu/data/concept-dict.json").read_text())
    dictionary["age"]["sources"]["miiv"][0].pop("callback", None)
    previous = tmp_path / "previous-dictionary.json"
    previous.write_text(json.dumps(dictionary))
    loaders = []
    try:
        results = []
        for dictionary_path in (previous, None):
            loader = BaseICULoader(data_path=tmp_path, database="miiv", dict_path=dictionary_path)
            loaders.append(loader)
            result = loader.load_concepts(["age", "sex"], merge=True, verbose=False,
                                          concept_workers=1, parallel_workers=1)
            results.append(result.sort_values("stay_id").reset_index(drop=True))
        old, new = results
        assert old.age.tolist() == [60, 60, 91, 88]
        assert new.age.tolist() == [65, 66, 91, 93]
        pd.testing.assert_frame_equal(old.drop(columns="age"), new.drop(columns="age"))
    finally:
        for loader in loaders:
            loader.datasource.clear_cache()
        _close_duckdb_connections()
