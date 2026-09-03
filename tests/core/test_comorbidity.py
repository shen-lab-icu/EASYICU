"""Tests for Quan 2005 Charlson / Elixhauser comorbidity coding.

The synthetic cases pin the code-set matching, the ICD-9/10 version
split, and the weighting/hierarchy rules. A real-data prevalence sanity
check (gated behind ``needs_real_data``) guards against gross code-table
regressions by asserting population prevalences stay in plausible bands.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from easyicu.scores.comorbidity import (
    CHARLSON_WEIGHTS,
    ELIXHAUSER_VW_WEIGHTS,
    flag_comorbidities,
)


def test_charlson_basic_flags_and_index():
    # id1: CHF (1); id3: cirrhosis(msld 3) + metastatic(6) + diabetes(1)
    df = pd.DataFrame({
        "id": [1, 1, 2, 3, 3, 3],
        "code": ["41401", "42833", "E785", "5723", "C7800", "E119"],
        "version": [9, 9, 10, 9, 10, 10],
    })
    out = flag_comorbidities(df, system="charlson").set_index("id")
    assert out.loc[1, "chf"] and out.loc[1, "charlson_index"] == 1
    assert out.loc[2, "charlson_index"] == 0          # lipidaemia is not Charlson
    assert out.loc[3, "charlson_index"] == 10         # 3 + 6 + 1
    assert out.loc[3, "msld"] and out.loc[3, "metacanc"]


def test_charlson_severity_hierarchy_suppresses_milder():
    # both mild + severe liver present -> only severe counts
    df = pd.DataFrame({
        "id": [1, 1],
        "code": ["5715", "5723"],   # mild liver (cirrhosis w/o portal htn) + msld
        "version": [9, 9],
    })
    out = flag_comorbidities(df, system="charlson").set_index("id")
    assert out.loc[1, "msld"]
    assert not out.loc[1, "mild_liver"]   # suppressed by hierarchy
    assert out.loc[1, "charlson_index"] == 3  # msld only, not 3+1


def test_icd9_vs_icd10_version_isolation():
    # 'I50' is ICD-10 CHF; as an ICD-9 code it must NOT match CHF.
    df = pd.DataFrame({"id": [1], "code": ["I50"], "version": [9]})
    out = flag_comorbidities(df, system="charlson").set_index("id")
    assert not out.loc[1, "chf"]
    df10 = pd.DataFrame({"id": [1], "code": ["I50"], "version": [10]})
    out10 = flag_comorbidities(df10, system="charlson").set_index("id")
    assert out10.loc[1, "chf"]


def test_elixhauser_van_walraven_score():
    # CHF (+7) and obesity (-4) -> VW = 3, count = 2
    df = pd.DataFrame({
        "id": [1, 1],
        "code": ["I50", "E66"],
        "version": [10, 10],
    })
    out = flag_comorbidities(df, system="elixhauser").set_index("id")
    assert out.loc[1, "chf"] and out.loc[1, "obesity"]
    assert out.loc[1, "elixhauser_vw"] == CHARLSON_WEIGHTS["chf"] * 0 + 7 - 4
    assert out.loc[1, "elixhauser_count"] == 2


def test_dotted_codes_are_normalised():
    df = pd.DataFrame({"id": [1], "code": ["428.0"], "version": [9]})
    out = flag_comorbidities(df, system="charlson").set_index("id")
    assert out.loc[1, "chf"]


def test_empty_input_returns_empty():
    out = flag_comorbidities(pd.DataFrame(columns=["id", "code", "version"]),
                             system="charlson")
    assert len(out) == 0


def test_weight_tables_cover_all_conditions():
    from easyicu.scores.comorbidity import CHARLSON, ELIXHAUSER
    assert set(CHARLSON) == set(CHARLSON_WEIGHTS)
    assert set(ELIXHAUSER) == set(ELIXHAUSER_VW_WEIGHTS)
    assert len(CHARLSON) == 17
    assert len(ELIXHAUSER) == 31


@pytest.mark.needs_real_data
def test_charlson_prevalence_on_real_mimiciv():
    root = os.environ.get("EASYICU_DATA_PATH", "")
    dx_path = Path(root) / "mimiciv" / "diagnoses_icd.parquet"
    if not dx_path.exists():
        pytest.skip("mimiciv diagnoses_icd.parquet not available")
    dx = pd.read_parquet(dx_path).rename(
        columns={"hadm_id": "id", "icd_code": "code", "icd_version": "version"})
    ch = flag_comorbidities(dx, system="charlson")
    # plausible bands from published MIMIC-IV comorbidity prevalences
    assert 1.0 <= ch["charlson_index"].mean() <= 3.5
    assert 0.08 <= ch["chf"].mean() <= 0.30
    assert 0.05 <= ch["malignancy"].mean() <= 0.20
    el = flag_comorbidities(dx, system="elixhauser")
    htn = (el["htn_unc"] | el["htn_comp"]).mean()
    assert 0.35 <= htn <= 0.65  # most prevalent


@pytest.mark.needs_real_data
def test_load_comorbidity_per_stay_real():
    from easyicu.scores.comorbidity import load_comorbidity

    out = load_comorbidity("miiv", system="charlson")
    assert not out.empty
    assert "stay_id" in out.columns and "charlson_index" in out.columns
    # one row per ICU stay (hadm->stay join must not duplicate)
    assert out["stay_id"].is_unique
    assert 1.5 <= out["charlson_index"].mean() <= 4.5  # ICU sicker than ward


@pytest.mark.needs_real_data
def test_load_comorbidity_na_for_hirid():
    from easyicu.scores.comorbidity import load_comorbidity

    # HiRID has no ICD diagnosis source -> empty, not an error.
    assert load_comorbidity("hirid", system="charlson").empty


@pytest.mark.needs_real_data
def test_charlson_via_load_concepts_real():
    from easyicu import load_concepts

    df = load_concepts(["charlson"], database="miiv")
    assert "charlson" in df.columns
    assert "stay_id" in df.columns
    assert df["charlson"].notna().any()


# --- eICU ICD-9/10 version split (2026-08-16 data review) ---

def test_eicu_v_codes_are_icd9_not_icd10() -> None:
    from easyicu.scores.comorbidity import _explode_eicu_codes

    long = _explode_eicu_codes(pd.Series(["427.31,V45.1"]))
    assert long["code"].tolist() == ["427.31", "V45.1"]
    assert long["version"].tolist() == [9, 9]

def test_eicu_true_icd10_letter_codes_are_not_reclassified_as_icd9() -> None:
    from easyicu.scores.comorbidity import _explode_eicu_codes

    long = _explode_eicu_codes(pd.Series(["I50.9,E11.9,E849.7,V45.1"]))

    assert long["code"].tolist() == ["I50.9", "E11.9", "E849.7", "V45.1"]
    assert long["version"].tolist() == [10, 10, 9, 9]
