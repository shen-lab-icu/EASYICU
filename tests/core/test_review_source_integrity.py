"""Synthetic regression receipts for source identity and ascertainment."""

import numpy as np
import pandas as pd
import pytest

from easyicu.patient_filter import PatientFilter
from easyicu.scores import outcomes
from easyicu.io.data_load import load_difftime, load_id, TimeOriginError
from easyicu.datasource import ICUDataSource
from easyicu.resources import load_data_sources


def test_all_sixty_patient_shards_are_read(tmp_path):
    bucket = tmp_path / "patient_bucket"
    bucket.mkdir()
    for i in range(60):
        pd.DataFrame({"patientunitstayid": [i]}).to_parquet(bucket / f"{i:03}.parquet")
    result = PatientFilter("eicu", data_path=tmp_path)._read_table("patient")
    assert result.patientunitstayid.tolist() == list(range(60))


def test_negative_sepsis_requires_observed_negative_labels(monkeypatch):
    pf = PatientFilter("miiv", data_path="/unused")
    pf._demographics = pd.DataFrame({"patient_id": [1, 2, 3, 4, 5, 6]})
    monkeypatch.setattr("easyicu.api.load_sepsis3", lambda **kw: pd.DataFrame({
        "stay_id": [1, 2, 3, 5, 5, 6, 6], "sep3": [1, 0, None, 0, None, 0, 1],
    }))
    assert pf.filter(has_sepsis=False) == [2]
    assert pf.filter(has_sepsis=True) == [1, 6]


def test_empty_sepsis_result_is_not_a_negative_cohort(monkeypatch):
    pf = PatientFilter("miiv", data_path="/unused")
    pf._demographics = pd.DataFrame({"patient_id": [1, 2]})
    monkeypatch.setattr("easyicu.api.load_sepsis3", lambda **kw: pd.DataFrame())
    assert pf.filter(has_sepsis=False) == []


@pytest.mark.parametrize("table", ["patients", "admissions", "icustays"])
def test_mimic_demographic_join_rejects_duplicate_identity(tmp_path, table):
    frames = {
        "icustays": pd.DataFrame({"subject_id": [1], "hadm_id": [10], "stay_id": [100]}),
        "patients": pd.DataFrame({"subject_id": [1]}),
        "admissions": pd.DataFrame({"subject_id": [1], "hadm_id": [10]}),
    }
    frames[table] = pd.concat([frames[table]] * 2, ignore_index=True)
    for name, frame in frames.items():
        frame.to_parquet(tmp_path / f"{name}.parquet")
    with pytest.raises(ValueError, match="identity|unique|duplicate"):
        PatientFilter("miiv", data_path=tmp_path)._load_demographics()


def _mortality_tables(monkeypatch, dod, subjects=None):
    tables = {
        "icustays": pd.DataFrame({"subject_id": [1, 2], "hadm_id": [10, 20],
            "stay_id": [100, 200], "icustay_id": [100, 200],
            "intime": ["2180-01-01"] * 2, "los": [2, 2]}),
        "patients": pd.DataFrame({"subject_id": subjects or [1, 2], "dod": dod}),
    }
    monkeypatch.setattr(outcomes, "_raw_table", lambda db, path, name: tables[name].copy())


def test_unmatched_patient_is_unknown_not_365_day_survival(monkeypatch):
    _mortality_tables(monkeypatch, [None], [1])
    result = outcomes.load_outcomes("miiv").set_index("stay_id")
    assert not result.loc[100, "mort_365d"]
    assert pd.isna(result.loc[200, "mort_28d"])
    assert pd.isna(result.loc[200, "followup_days_365d"])


def test_nonempty_unparseable_death_date_is_rejected(monkeypatch):
    _mortality_tables(monkeypatch, [None, "broken-date"])
    with pytest.raises(ValueError, match="death|dod"):
        outcomes.load_outcomes("miiv")


def test_mortality_parses_date_and_timestamp_without_pandas_2_only_format(monkeypatch):
    _mortality_tables(monkeypatch, ["2180-01-03", "2180-01-04 00:00:00"])
    original = pd.to_datetime

    def pandas_15_api(*args, **kwargs):
        assert kwargs.get("format") not in {"mixed", "ISO8601"}
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "to_datetime", pandas_15_api)
    result = outcomes.load_outcomes("miiv")
    assert result["followup_days_28d"].tolist() == [2.0, 3.0]
    assert result["mort_28d"].all()


def test_mimic_iii_does_not_borrow_mimic_iv_null_followup_contract(monkeypatch):
    _mortality_tables(monkeypatch, [None, None])
    assert outcomes.load_outcomes("mimic")["mort_28d"].isna().all()


def test_demographics_preserves_io_failure(monkeypatch):
    from easyicu.api import convenience
    def denied(**kw):
        raise PermissionError("synthetic source permission denied")
    monkeypatch.setattr(convenience, "load_concepts", denied)
    with pytest.raises(PermissionError):
        convenience.load_demographics(verbose=False)


def _source_tables(tmp_path, origins=(0,)):
    config = load_data_sources().get("miiv")
    tables = {
        "icustays": pd.DataFrame({"subject_id": [1] * len(origins), "hadm_id": [10] * len(origins),
            "stay_id": [100] * len(origins),
            "intime": pd.to_datetime("2180-01-01") + pd.to_timedelta(list(origins), unit="h")}),
        "chartevents": pd.DataFrame({"stay_id": [100], "charttime": pd.to_datetime(["2180-01-01 13:00"])}),
    }
    paths = {}
    for name, frame in tables.items():
        for col in config.get_table(name).columns:
            if col not in frame:
                frame[col] = None
        paths[name] = tmp_path / f"{name}.parquet"
        frame.to_parquet(paths[name])
    return config, paths


def test_conflicting_time_origins_are_rejected_without_patient_ids(tmp_path):
    config, paths = _source_tables(tmp_path, (0, 12))
    source = ICUDataSource(config, base_path=tmp_path, table_sources=paths)
    with pytest.raises(TimeOriginError, match="conflicting") as caught:
        load_difftime("chartevents", src=source, id_hint="stay_id", time_vars=["charttime"])
    assert "100" not in str(caught.value)


@pytest.mark.parametrize("as_config", [True, False])
@pytest.mark.parametrize("remap", [True, False])
def test_source_selection_is_shared_by_observations_origin_and_id_map(tmp_path, as_config, remap):
    config, paths = _source_tables(tmp_path)
    opts = dict(src=config if as_config else "miiv", base_path=tmp_path,
        table_sources=paths, time_vars=["charttime"], cols=["stay_id", "charttime"])
    result = (load_id("chartevents", id_var="hadm_id", **opts) if remap else
        load_difftime("chartevents", id_hint="stay_id", **opts)).data
    assert result.charttime.iloc[0] == pd.Timedelta(hours=13)
    if remap:
        assert result.hadm_id.tolist() == [10]


def test_identical_duplicate_origins_are_safe(tmp_path):
    config, paths = _source_tables(tmp_path, (0, 0))
    source = ICUDataSource(config, base_path=tmp_path, table_sources=paths)
    assert len(load_difftime("chartevents", src=source, time_vars=["charttime"]).data) == 1
