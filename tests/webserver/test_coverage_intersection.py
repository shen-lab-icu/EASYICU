from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.webserver import cohort_review, extraction_filters
from easyicu.webserver.patient_drilldown import _bounded_covered_entities


def _fixture(tmp_path: Path):
    pd.DataFrame({"stay_id": [1, 2], "age": [60, 70]}).to_parquet(
        tmp_path / "demographics.parquet", index=False
    )
    pd.DataFrame({"stay_id": [2, 3], "value": [1.0, 2.0]}).to_parquet(
        tmp_path / "vitals.parquet", index=False
    )
    item = {
        "file": "vitals.parquet",
        "module": "vitals",
        "rows": 2,
        "columns": ["stay_id", "value"],
    }
    desc = {
        "path": str(tmp_path),
        "summary": {"stays": 2},
        "files": [
            {
                "file": "demographics.parquet",
                "module": "demographics",
                "rows": 2,
                "columns": ["stay_id", "age"],
            },
            item,
        ],
    }
    return item, desc


def test_all_web_coverage_surfaces_use_true_cohort_intersection(tmp_path: Path) -> None:
    item, desc = _fixture(tmp_path)
    cohort_ids = {"1", "2"}

    assert cohort_review._covered_entities(tmp_path, item, cohort_ids) == 1
    assert extraction_filters._coverage(
        tmp_path / "vitals.parquet", cohort_ids
    ) == (50.0, 1)
    assert _bounded_covered_entities(tmp_path, item, cohort_ids) == 1
    option = next(
        row for row in extraction_filters._module_options(desc)
        if row["module"] == "vitals"
    )
    assert option["coverage_pct"] == 50.0


def test_coverage_surfaces_canonicalize_source_stay_keys(tmp_path: Path) -> None:
    """eICU exports keep patientunitstayid; coverage must not degrade to unknown."""

    from easyicu.webserver import dataio
    from easyicu.webserver.patient_drilldown import _quality_payload

    pd.DataFrame({"patientunitstayid": [101, 102], "age": [61.0, 72.0]}).to_parquet(
        tmp_path / "demographics.parquet", index=False
    )
    pd.DataFrame(
        {"patientunitstayid": [101, 101], "charttime": [0.0, 2.0], "hr": [80.0, 91.0]}
    ).to_parquet(tmp_path / "vitals.parquet", index=False)
    desc = {
        "path": str(tmp_path),
        "summary": {"stays": 2},
        "files": [
            {
                "file": "demographics.parquet",
                "module": "demographics",
                "rows": 2,
                "columns": ["patientunitstayid", "age"],
            },
            {
                "file": "vitals.parquet",
                "module": "vitals",
                "rows": 2,
                "columns": ["patientunitstayid", "charttime", "hr"],
            },
        ],
    }

    assert dataio._fast_stay_ids(tmp_path, desc["files"]) == {"101", "102"}
    review = {row["module"]: row for row in cohort_review._coverage_payload(tmp_path, desc)}
    assert (review["demographics"]["covered_entities"], review["demographics"]["coverage_pct"]) == (2, 100.0)
    assert (review["vitals"]["covered_entities"], review["vitals"]["coverage_pct"]) == (1, 50.0)
    options = {row["module"]: row for row in extraction_filters._module_options(desc)}
    assert options["vitals"]["coverage_pct"] == 50.0
    quality = {row["module"]: row for row in _quality_payload(tmp_path, desc)}
    assert quality["vitals"]["coverage_pct"] == 50.0


def test_infection_catalog_does_not_claim_angus_definition() -> None:
    from easyicu.concept import catalog

    label = " ".join(catalog.CONCEPT_DICTIONARY["infection_icd"][:2])
    description = " ".join(catalog.CONCEPT_DESCRIPTIONS["infection_icd"])
    assert "Angus" not in label
    assert "not the Angus 2001 ICD-9 code list" in description

    # The composite that consumes infection_icd must not reintroduce the
    # ICD-code-list description it was corrected away from.
    susp_label = " ".join(catalog.CONCEPT_DICTIONARY["susp_inf"][:2])
    susp_description = " ".join(catalog.CONCEPT_DESCRIPTIONS["susp_inf"])
    assert "ICD" not in susp_label
    assert "ICD infection diagnosis codes" not in susp_description
    assert "ICD感染诊断码" not in susp_description
    assert "keyword match" in susp_description
