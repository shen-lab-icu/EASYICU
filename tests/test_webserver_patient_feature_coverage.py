from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import dataio
from easyicu.webserver import sources as source_store
from easyicu.webserver.app import app
from easyicu.webserver.patient_drilldown import coverage


@pytest.fixture(autouse=True)
def _isolated_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    coverage.clear_cache()


def _write_export(root: Path) -> Path:
    root.mkdir()
    demographics = pd.DataFrame(
        {"stay_id": [11, 12], "age": [61.0, 72.0], "sex": ["F", "M"]}
    )
    vitals = pd.DataFrame(
        {
            "stay_id": [11, 11, 12, 12],
            "charttime": [0.0, 4.0, 0.0, 3.0],
            "hr": [80.0, 91.0, 75.0, 82.0],
            "map": [70.0, 74.0, None, None],
            "temp": pd.Series([None, None, None, None], dtype="float64"),
        }
    )
    ventilator = pd.DataFrame(
        {
            "stay_id": [11, 11, 12],
            "charttime": [1.0, 5.0, 2.0],
            "vent_mode": ["volume", "pressure", "standby"],
        }
    )
    frames = {
        "demographics": demographics,
        "vitals": vitals,
        "ventilator": ventilator,
    }
    files = []
    for module, frame in frames.items():
        name = f"{module}.parquet"
        frame.to_parquet(root / name, index=False)
        concept_ids = [
            column for column in frame.columns if column not in {"stay_id", "charttime"}
        ]
        files.append(
            {
                "file": name,
                "module": module,
                "rows": len(frame),
                "concepts": len(concept_ids),
                "concept_ids": concept_ids,
            }
        )
    pd.DataFrame({"concept_id": ["hr", "map", "temp"]}).to_csv(
        root / "feature_definitions.csv", index=False
    )
    (root / "_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu_native_export_v2",
                "database": "miiv",
                "cohort_report": {"selected": 2},
                "files": files,
                "concept_availability": {
                    "structurally_unavailable": [
                        {
                            "concept_id": "vent_free_days_28",
                            "module": "outcome",
                            "reason_code": "outcome_concept_structurally_unavailable",
                            "supported_databases": ["eicu", "eicu_demo"],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    source_store.register_source(
        str(root), label="Coverage fixture", active=True, crossdb=True
    )
    return root


def _write_eicu_export(root: Path) -> Path:
    root.mkdir()
    frames = {
        "demographics": pd.DataFrame(
            {
                "patientunitstayid": [101, 102],
                "age": [61.0, 72.0],
                "sex": ["F", "M"],
            }
        ),
        "outcome": pd.DataFrame(
            {
                "patientunitstayid": [101, 102],
                "death": [0, 1],
                "los_icu": [2.0, 4.0],
            }
        ),
        "vitals": pd.DataFrame(
            {
                "patientunitstayid": [101, 101, 102],
                "charttime": [0.0, 2.0, 0.0],
                "hr": [80.0, 91.0, 75.0],
            }
        ),
    }
    files = []
    for module, frame in frames.items():
        name = f"{module}.parquet"
        frame.to_parquet(root / name, index=False)
        concept_ids = [
            column
            for column in frame.columns
            if column not in {"patientunitstayid", "charttime"}
        ]
        files.append(
            {
                "file": name,
                "module": module,
                "rows": len(frame),
                "concepts": len(concept_ids),
                "concept_ids": concept_ids,
            }
        )
    (root / "_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu_native_export_v2",
                "database": "eicu_demo",
                "cohort_report": {"selected": 2},
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    source_store.register_source(
        str(root), label="eICU identifier fixture", active=True, crossdb=True
    )
    return root


def _assert_no_direct_identifiers(value: Any) -> None:
    forbidden = {"stay_id", "subject_id", "hadm_id", "patientunitstayid"}
    if isinstance(value, dict):
        assert forbidden.isdisjoint(value)
        for child in value.values():
            _assert_no_direct_identifiers(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_direct_identifiers(child)


def test_export_description_excludes_feature_definition_metadata(
    tmp_path: Path,
) -> None:
    export = _write_export(tmp_path / "export")

    description = dataio.describe_export_source(str(export))

    assert description["summary"] == {
        "stays": 2,
        "modules": 3,
        "file_count": 3,
        "total_rows": 9,
    }
    assert {row["module"] for row in description["files"]} == {
        "demographics",
        "vitals",
        "ventilator",
    }


def test_feature_coverage_separates_observed_all_null_and_unsupported(
    tmp_path: Path,
) -> None:
    export = _write_export(tmp_path / "export")
    description = dataio.describe_export_source(str(export))

    payload = coverage.build_feature_coverage(export, description)
    rows = {
        row["feature"]: row
        for module in payload["modules"]
        for row in module["features"]
    }

    assert payload["summary"]["definitions"] == 281
    assert payload["summary"]["modules"] == 19
    assert rows["hr"]["status"] == "observed"
    assert rows["hr"]["non_null_count"] == 4
    assert rows["hr"]["trajectory_candidate"] is True
    assert rows["temp"]["status"] == "all_null"
    assert rows["temp"]["non_null_count"] == 0
    assert rows["vent_mode"]["status"] == "observed"
    assert rows["vent_mode"]["numeric"] is False
    assert rows["vent_free_days_28"]["status"] == "structurally_unavailable"
    assert rows["vent_free_days_28"]["reason_code"] == (
        "outcome_concept_structurally_unavailable"
    )
    assert rows["lact"]["status"] == "not_materialized"
    assert payload["provenance"]["patient_rows_returned"] is False


def test_drilldown_summary_uses_export_wide_catalog_coverage(
    tmp_path: Path,
) -> None:
    _write_export(tmp_path / "export")
    payload = TestClient(app).post("/api/patient-review/drilldown", json={}).json()

    loaded = payload["data_tables"]["loaded_summary"]
    assert loaded["review_features"] == 281
    assert loaded["module_count"] == 19
    assert loaded["observed_features"] == 5
    modules = {row["module"]: row for row in payload["data_tables"]["modules"]}
    assert modules["vitals"]["review_features"] == 12
    assert modules["vitals"]["observed_features"] == 2
    assert modules["vitals"]["dynamic_features"] == 2
    assert modules["ventilator"]["observed_features"] == 1


def test_lazy_feature_endpoint_is_bounded_and_pseudonymous(tmp_path: Path) -> None:
    _write_export(tmp_path / "export")
    client = TestClient(app)
    bootstrap = client.post("/api/patient-review/drilldown", json={})
    assert bootstrap.status_code == 200
    payload = bootstrap.json()
    entity = payload["entity_navigation"]["options"][0]

    response = client.post(
        "/api/patient-review/feature",
        json={
            "entity_ref": entity["ref"],
            "entity_ordinal": entity["ordinal"],
            "feature": "hr",
        },
    )
    category = client.post(
        "/api/patient-review/feature",
        json={
            "entity_ref": entity["ref"],
            "entity_ordinal": entity["ordinal"],
            "feature": "vent_mode",
        },
    )

    assert response.status_code == category.status_code == 200
    detail = response.json()
    assert detail["status"] == "numeric_trajectory"
    assert detail["signal"]["values"] == [80.0, 91.0]
    assert detail["signal"]["times"] == [0.0, 4.0]
    assert detail["signal"]["point_count"] == 2
    assert detail["privacy"]["max_points"] == 12
    category_detail = category.json()
    assert category_detail["status"] == "observed_categorical"
    assert category_detail["observation"]["observed_values"] == [
        "volume",
        "pressure",
    ]
    _assert_no_direct_identifiers(detail)
    _assert_no_direct_identifiers(category_detail)


def test_lazy_feature_endpoint_fails_closed_on_entity_or_feature_tampering(
    tmp_path: Path,
) -> None:
    _write_export(tmp_path / "export")
    client = TestClient(app)
    entity = client.post("/api/patient-review/drilldown", json={}).json()[
        "entity_navigation"
    ]["options"][0]

    tampered = client.post(
        "/api/patient-review/feature",
        json={
            "entity_ref": "ent_tampered",
            "entity_ordinal": entity["ordinal"],
            "feature": "hr",
        },
    )
    unknown = client.post(
        "/api/patient-review/feature",
        json={
            "entity_ref": entity["ref"],
            "entity_ordinal": entity["ordinal"],
            "feature": "not_a_feature",
        },
    )

    assert tampered.status_code == unknown.status_code == 400
    assert tampered.json()["detail"]["error"] == "entity_ref_ordinal_mismatch"
    assert unknown.json()["detail"] == {
        "error": "unknown_patient_feature",
        "feature": "not_a_feature",
    }


def test_eicu_patientunitstayid_is_canonicalized_at_review_boundary(
    tmp_path: Path,
) -> None:
    _write_eicu_export(tmp_path / "eicu_export")
    client = TestClient(app)

    bootstrap = client.post("/api/patient-review/drilldown", json={})

    assert bootstrap.status_code == 200
    payload = bootstrap.json()
    assert payload["source"]["database"] == "eicu_demo"
    assert payload["summary"]["entities"] == 2
    entity = payload["entity_navigation"]["options"][0]
    detail = client.post(
        "/api/patient-review/feature",
        json={
            "entity_ref": entity["ref"],
            "entity_ordinal": entity["ordinal"],
            "feature": "hr",
        },
    )
    assert detail.status_code == 200
    assert detail.json()["signal"]["values"] == [80.0, 91.0]
    _assert_no_direct_identifiers(payload)
    _assert_no_direct_identifiers(detail.json())
