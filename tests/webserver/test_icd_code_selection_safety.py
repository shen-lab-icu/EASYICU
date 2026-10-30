"""ICD selection uses recorded code boundaries, not free-text guesses."""
from pathlib import Path

import pandas as pd
import pytest

from easyicu.webserver import dataio


def test_eicu_matches_each_code_prefix_not_other_codes_or_diagnosis_text(tmp_path: Path):
    pd.DataFrame([
        {"patientunitstayid": 1, "icd9code": " 585.9, I50.9 ", "diagnosisstring": "unknown"},
        {"patientunitstayid": 2, "icd9code": "J15.0", "diagnosisstring": "I50.9 heart failure"},
        {"patientunitstayid": 3, "icd9code": "XI50.9", "diagnosisstring": "none"},
        {"patientunitstayid": 4, "icd9code": "E11.9, 038.9", "diagnosisstring": "none"},
        {"patientunitstayid": 5, "icd9code": "I50.90bad?", "diagnosisstring": "none"},
    ]).to_parquet(tmp_path / "diagnosis.parquet")
    included, excluded = dataio._match_eicu_icd_ids(tmp_path, ["I50"], ["038"])
    assert included == {1}
    assert excluded == {4}


def test_eicu_refuses_to_use_diagnosis_description_as_a_code_column(tmp_path: Path):
    pd.DataFrame({"patientunitstayid": [1], "diagnosisstring": ["I50.9"]}).to_parquet(tmp_path / "diagnosis.parquet")
    with pytest.raises(dataio.ExportCohortError, match="icd_code_column_missing"):
        dataio._match_eicu_icd_ids(tmp_path, ["I50"], [])


def test_eicu_unknown_diagnosis_status_does_not_mean_no_disease(tmp_path: Path):
    pd.DataFrame({"patientunitstayid": [1, 2, 3], "icd9code": ["I50.9", None, ""]}).to_parquet(tmp_path / "diagnosis.parquet")
    with pytest.raises(dataio.ExportCohortError, match="icd_diagnosis_status_unknown") as found:
        dataio._match_eicu_icd_ids(tmp_path, [], ["I50"], {1, 2, 3, 4})
    assert found.value.detail["unknown_stays"] == 3
    assert "patientunitstayid" not in str(found.value.detail)


def test_eicu_csv_leading_zeroes_are_preserved_for_code_prefix(tmp_path: Path):
    (tmp_path / "diagnosis.csv").write_text("patientunitstayid,icd9code\n1,038.9\n2,389.0\n")
    assert dataio._match_eicu_icd_ids(tmp_path, ["038"], [], {1, 2}) == ({1}, set())


def test_mimic_unknown_diagnosis_status_is_not_retained_as_negative(tmp_path: Path):
    pd.DataFrame({"stay_id": [1, 2, 3], "hadm_id": [10, 20, 30]}).to_parquet(tmp_path / "icustays.parquet")
    pd.DataFrame({"hadm_id": [10, 20], "icd_code": ["A419", None]}).to_parquet(tmp_path / "diagnoses_icd.parquet")
    with pytest.raises(dataio.ExportCohortError, match="icd_diagnosis_status_unknown") as found:
        dataio._match_mimic_icd_ids(tmp_path, "stay_id", [], ["A41"], {1, 2, 3})
    assert found.value.detail["unknown_stays"] == 2


def test_mimic_csv_icd9_leading_zeroes_are_preserved(tmp_path: Path):
    (tmp_path / "icustays.csv").write_text("stay_id,hadm_id\n1,10\n2,20\n")
    (tmp_path / "diagnoses_icd.csv").write_text("hadm_id,icd9_code\n10,0389\n20,3890\n")
    assert dataio._match_mimic_icd_ids(tmp_path, "stay_id", ["038"], [], {1, 2}) == ({1}, set())
