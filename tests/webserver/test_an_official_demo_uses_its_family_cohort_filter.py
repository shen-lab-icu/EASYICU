"""Official demo extraction filters rows without relabelling the data source.

Synthetic tables exercise family dispatch and the native export boundary.
"""

from __future__ import annotations

from contextlib import contextmanager
import json

import pandas as pd
import pytest

import easyicu.api as api
from easyicu.webserver import dataio
from easyicu.patient_filter import PatientFilter, PatientFilterCriterionError


def _demo_tables(root, database):
    root.mkdir()
    if database == "eicu_demo":
        # Two adult stays belong to the same hospitalization. This intermediate
        # export must retain both: the source-bound hospital-first coordinate
        # is applied later by the research materializer, not the legacy filter.
        pd.DataFrame({
            "patientunitstayid": [11, 12, 13, 14],
            "patienthealthsystemstayid": [1, 1, 2, 3],
            "unitvisitnumber": [1, 2, 1, 1],
            "unittype": ["SDU/Step down", "MICU", "MICU", "MICU"],
            "age": ["65", "65", "17", None],
            "gender": ["Female", "Female", "Male", "Male"],
            "hospitaldischargestatus": ["Alive", "Alive", "Expired", "Unknown"],
            "unitdischargeoffset": [1440, 1440, 1440, 1440],
        }).to_parquet(root / "patient.parquet", index=False)
    else:
        pd.DataFrame({
            "icustay_id": [11, 12, 13, 14],
            "subject_id": [1, 1, 2, 3],
            "hadm_id": [1, 1, 2, 3],
            "intime": ["2020-01-01"] * 4,
            "los": [1.0] * 4,
        }).to_parquet(root / "icustays.parquet", index=False)
        pd.DataFrame({
            "subject_id": [1, 2, 3],
            "dob": ["1955-01-01", "2003-01-01", None],
            "gender": ["F", "M", "M"],
        }).to_parquet(root / "patients.parquet", index=False)
        pd.DataFrame({
            "subject_id": [1, 2, 3],
            "hadm_id": [1, 2, 3],
            "hospital_expire_flag": [0, 1, None],
        }).to_parquet(root / "admissions.parquet", index=False)


def _cohort(database):
    return {
        "preset": "all_icu", "age_min": 18, "exclude_readmissions": False,
        "observation_window_hours": 720,
        **({"first_icu_stay_per_hospitalization": True}
           if database == "eicu_demo" else {}),
    }


@pytest.mark.parametrize("database,id_column", [
    ("eicu_demo", "patientunitstayid"), ("mimic_demo", "icustay_id"),
])
def test_official_demo_uses_real_family_filter_and_counts_age_exclusions(
    tmp_path, database, id_column,
):
    raw = tmp_path / "raw"
    _demo_tables(raw, database)

    result = dataio._resolve_export_cohort(
        str(raw), database, _cohort(database), None, api,
    )

    assert result["patient_ids"] == {id_column: [11, 12]}
    report = result["cohort_report"]
    assert report["source_total"] == 4
    assert report["selected"] == 2
    age = report["demographic_steps"][0]
    assert age["criterion"] == "age"
    assert age["n_excluded"] == 2
    assert age["n_excluded_missing"] == 1
    assert result["cohort_contract"]["exclude_readmissions"] is False
    if database == "eicu_demo":
        assert result["cohort_contract"]["first_icu_stay_per_hospitalization"] is True


def test_unknown_demo_is_not_promoted_to_a_supported_family(tmp_path):
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="不支持的数据库: invented_demo"):
        dataio._resolve_export_cohort(
            str(tmp_path), "invented_demo", {"age_min": 18}, None,
            SimpleNamespace(get_id_col_for_database=lambda _: "stay_id"),
        )


def test_eicu_demo_export_keeps_demo_identity_and_hospital_first_contract(
    tmp_path, monkeypatch,
):
    raw = tmp_path / "raw"
    _demo_tables(raw, "eicu_demo")
    observed = []

    @contextmanager
    def keep_cache(**kwargs):
        observed.append(("cache", kwargs["database"]))
        yield

    def load_concepts(concepts, **kwargs):
        observed.append(("concepts", kwargs["database"]))
        assert kwargs["patient_ids"] == {"patientunitstayid": [11, 12]}
        values = {
            "age": [65.0, 65.0], "sex": ["Female", "Female"],
            "icu_unit_type": ["SDU/Step down", "MICU"], "death": [False, False],
        }
        return pd.DataFrame({
            "patientunitstayid": [11, 12],
            **{concept: values[concept] for concept in concepts},
        })

    monkeypatch.setattr(api, "keep_cache", keep_cache)
    monkeypatch.setattr(api, "load_concepts", load_concepts)

    class Job:
        events = []

        def emit(self, event):
            self.events.append(event)

    out = tmp_path / "export"
    result = dataio.make_export_runner(
        data_path=str(raw), database="eicu_demo",
        modules=["demographics", "outcome"],
        concepts={"demographics": ["age", "sex", "icu_unit_type"], "outcome": ["death"]},
        export_format="parquet", out_dir=str(out), cohort=_cohort("eicu_demo"),
    )(Job())

    manifest = json.loads((out / "_manifest.json").read_text())
    assert result["file_count"] == 2
    assert observed == [("cache", "eicu_demo"), ("concepts", "eicu_demo"), ("concepts", "eicu_demo")]
    assert manifest["database"] == "eicu_demo"
    assert manifest["data_path"] == str(raw)
    assert manifest["cohort_contract"]["first_icu_stay_per_hospitalization"] is True
    assert manifest["cohort_contract"]["exclude_readmissions"] is False
    assert manifest["cohort_report"]["selected"] == 2


@pytest.mark.parametrize("database,family", [("eicu_demo", "eicu"), ("mimic_demo", "mimic")])
def test_missing_age_never_becomes_a_top_coded_adult_and_censoring_remains_open(
    tmp_path, database, family,
):
    raw = tmp_path / "raw"
    _demo_tables(raw, database)
    table = raw / ("patient.parquet" if database == "eicu_demo" else "patients.parquet")
    frame = pd.read_parquet(table)
    if database == "eicu_demo":
        frame.loc[[0, 1], "age"] = "> 89"
    else:
        frame.loc[0, "dob"] = "1700-01-01"
    frame.to_parquet(table, index=False)
    patient_filter = PatientFilter(database=family, data_path=str(raw))
    selected = patient_filter.filter(age_min=18, return_dataframe=True)
    assert selected.patient_id.tolist() == [11, 12]
    assert selected.age_is_censored.tolist() == [True, True]
    assert selected.age_lower.tolist() == [90, 90]
    assert selected.age_upper.isna().all()
    step, = patient_filter.last_selection_steps()
    assert (step["n_before"], step["n_remaining"], step["n_excluded_missing"]) == (4, 2, 1)
    with pytest.raises(PatientFilterCriterionError) as refused:
        patient_filter.filter(age_min=91)
    assert refused.value.code == "patient_filter_grouped_age_indeterminate"
