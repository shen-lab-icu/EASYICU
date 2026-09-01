"""Focused contracts for bounded Patient Review navigation and lazy table pages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import patient_drilldown
from easyicu.webserver import sources as source_store
from easyicu.webserver.app import app
from easyicu.webserver.patient_drilldown import navigation


@pytest.fixture(autouse=True)
def _isolated_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(
        source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json"
    )
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])


def _write_export(root: Path, entities: int = 30) -> Path:
    root.mkdir()
    ids = list(range(1, entities + 1))
    tables = {
        "demographics": pd.DataFrame(
            {
                "stay_id": ids,
                "age": [40 + value for value in ids],
                "sex": ["F" if value % 2 else "M" for value in ids],
            }
        ),
        "outcome": pd.DataFrame(
            {
                "stay_id": ids,
                "death": [value % 7 == 0 for value in ids],
                "los_icu": [float(value % 5 + 1) for value in ids],
            }
        ),
        "sofa2_score": pd.DataFrame(
            {
                "stay_id": ids,
                "charttime": ["2026-01-01 00:00"] * entities,
                "sofa2": [float(value % 12) for value in ids],
            }
        ),
        "sepsis3_sofa2": pd.DataFrame(
            {
                "stay_id": ids,
                "sep3_sofa2": [value % 4 == 0 for value in ids],
            }
        ),
        "vitals": pd.DataFrame(
            {
                "stay_id": ids,
                "charttime": ["2026-01-01 00:00"] * entities,
                "hr": [70 + value for value in ids],
                "map": [60 + value for value in ids],
                "spo2": [96.0] * entities,
                "temp": [37.0] * entities,
            }
        ),
        "labs": pd.DataFrame(
            {
                "stay_id": ids,
                "charttime": ["2026-01-01 00:00"] * entities,
                "lact": [round(1.0 + value / 10, 1) for value in ids],
            }
        ),
    }
    files = []
    for module, frame in tables.items():
        file_name = f"{module}.csv"
        frame.to_csv(root / file_name, index=False)
        files.append({"file": file_name, "module": module, "rows": len(frame)})
    (root / "_manifest.json").write_text(
        json.dumps({"database": "miiv", "files": files}), encoding="utf-8"
    )
    source_store.register_source(
        str(root), label="Patient browse fixture", active=True, crossdb=True
    )
    return root


def _assert_no_direct_identifier_keys(value: Any) -> None:
    forbidden = {
        "stay_id",
        "subject_id",
        "hadm_id",
        "patientunitstayid",
        "patient_id",
    }
    if isinstance(value, dict):
        assert forbidden.isdisjoint(value)
        for child in value.values():
            _assert_no_direct_identifier_keys(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_direct_identifier_keys(child)


def test_entity_navigation_pages_are_bounded_and_pseudonymous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_export(tmp_path / "export")
    client = TestClient(app)

    bootstrap = client.post("/api/patient-review/drilldown", json={})
    second = client.post(
        "/api/patient-review/entities",
        json={"entity_page": 2, "entity_page_size": 12},
    )
    monkeypatch.setattr(navigation.secrets, "randbelow", lambda _count: 1)
    random_page = client.post(
        "/api/patient-review/entities", json={"random_page": True}
    )

    assert bootstrap.status_code == second.status_code == random_page.status_code == 200
    first_nav = bootstrap.json()["entity_navigation"]
    assert [row["ordinal"] for row in first_nav["options"]] == list(range(1, 13))
    navigation_page = second.json()["navigation"]
    assert navigation_page["page"] == 2
    assert navigation_page["page_size"] == 12
    assert navigation_page["page_count"] == 3
    assert navigation_page["total_entities"] == 30
    assert [row["ordinal"] for row in navigation_page["options"]] == list(
        range(13, 25)
    )
    assert all(row["ref"].startswith("ent_") for row in navigation_page["options"])
    assert random_page.json()["navigation"]["page"] == 2
    assert random_page.json()["navigation"]["randomized"] is True
    assert second.json()["privacy"]["max_entity_page_size"] == 24
    _assert_no_direct_identifier_keys(bootstrap.json())
    _assert_no_direct_identifier_keys(second.json())


def test_entity_detail_requires_matching_ref_and_ordinal(tmp_path: Path) -> None:
    _write_export(tmp_path / "export")
    client = TestClient(app)
    page = client.post(
        "/api/patient-review/entities",
        json={"entity_page": 2, "entity_page_size": 12},
    ).json()["navigation"]
    option = page["options"][0]

    detail = client.post(
        "/api/patient-review/entity",
        json={"entity_ref": option["ref"], "entity_ordinal": option["ordinal"]},
    )
    tampered = client.post(
        "/api/patient-review/entity",
        json={"entity_ref": "ent_tampered", "entity_ordinal": option["ordinal"]},
    )
    missing_ordinal = client.post(
        "/api/patient-review/entity", json={"entity_ref": option["ref"]}
    )

    assert detail.status_code == 200
    payload = detail.json()
    assert payload["selected"]["label"] == "Entity 13"
    assert payload["selected"]["demographics"] == {"age": 53.0, "sex": "F"}
    assert {row["key"]: row for row in payload["selected"]["signals"]}["hr"][
        "current"
    ] == 83.0
    assert payload["privacy"]["max_comparison_entities"] == 5
    assert tampered.status_code == 400
    assert tampered.json()["detail"]["error"] == "entity_ref_ordinal_mismatch"
    assert missing_ordinal.status_code == 400
    assert (
        missing_ordinal.json()["detail"]["error"]
        == "entity_ref_and_ordinal_required"
    )
    _assert_no_direct_identifier_keys(payload)


def test_patient_summary_omits_day_unit_when_los_is_unknown() -> None:
    cards = patient_drilldown._patient_summary_cards(
        {
            "demographics": {"age": 53, "sex": "F"},
            "scores": {},
            "outcomes": {"status": "Survived", "icu_los_days": None},
        }
    )

    assert cards[-1] == {"label": "ICU LOS", "value": "unknown", "tone": "neutral"}


def test_table_preview_endpoint_reads_only_requested_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_export(tmp_path / "export")
    calls: list[str] = []
    original = patient_drilldown._read_table_preview

    def record(path: Path, columns: list[str], nrows: int, offset: int = 0):
        calls.append(path.name)
        return original(path, columns, nrows, offset)

    monkeypatch.setattr(patient_drilldown, "_read_table_preview", record)
    client = TestClient(app)
    response = client.post(
        "/api/patient-review/table-preview",
        json={"table_module": "labs", "table_page": 2, "table_page_size": 1},
    )

    assert response.status_code == 200
    preview = response.json()["module_preview"]
    assert calls == ["labs.csv"]
    assert preview["module"] == "labs"
    assert preview["pagination"]["page"] == 2
    assert preview["rows"][0]["lact"] == 1.2
    assert preview["rows"][0]["entity"].startswith("ent_")
    assert response.json()["privacy"]["max_table_page_size"] == 100
    _assert_no_direct_identifier_keys(response.json())

    unknown = client.post(
        "/api/patient-review/table-preview",
        json={"table_module": "not_a_module"},
    )
    assert unknown.status_code == 400
    assert unknown.json()["detail"] == {
        "error": "unknown_table_module",
        "module": "not_a_module",
    }


def test_table_preview_read_failure_is_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_export(tmp_path / "export")

    def fail(*_args, **_kwargs):
        raise RuntimeError("/private/patient/path must not leave the server")

    monkeypatch.setattr(patient_drilldown, "_read_table_preview", fail)
    response = TestClient(app).post(
        "/api/patient-review/table-preview", json={"table_module": "labs"}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "error": "bounded_table_preview_read_failed",
        "module": "labs",
    }
    assert "/private/patient/path" not in response.text


def test_drilldown_bootstrap_reads_only_one_table_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_export(tmp_path / "export")
    calls: list[str] = []
    original = patient_drilldown._read_table_preview

    def record(path: Path, columns: list[str], nrows: int, offset: int = 0):
        calls.append(path.name)
        return original(path, columns, nrows, offset)

    monkeypatch.setattr(patient_drilldown, "_read_table_preview", record)
    response = TestClient(app).post("/api/patient-review/drilldown", json={})

    assert response.status_code == 200
    payload = response.json()
    assert calls == ["demographics.csv"]
    assert [row["module"] for row in payload["data_tables"]["table_previews"]] == [
        "demographics"
    ]
    labs = next(
        row for row in payload["data_tables"]["modules"] if row["module"] == "labs"
    )
    assert labs["review_status"] == "inventory_only"
    assert labs["preview_status"] == "available_on_demand"
