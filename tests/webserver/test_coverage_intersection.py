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


def test_infection_catalog_does_not_claim_angus_definition() -> None:
    from easyicu.concept import catalog

    label = " ".join(catalog.CONCEPT_DICTIONARY["infection_icd"][:2])
    description = " ".join(catalog.CONCEPT_DESCRIPTIONS["infection_icd"])
    assert "Angus" not in label
    assert "not the Angus 2001 ICD-9 code list" in description
