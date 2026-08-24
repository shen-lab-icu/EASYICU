"""Focused contracts for the conversational ICD cohort preview."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.webserver import dataio
from easyicu.webserver.copilot_data_workbench import (
    CopilotDataWorkbenchSnapshotStore,
)
from easyicu.webserver.pi_copilot import projections
from easyicu.webserver.pi_copilot import tools as tool_owner
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    PiSessionRecord,
    ToolExecutionContext,
)


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-icd-preview",
            project_id="project-icd-preview",
            binding=AuthorityBinding(
                study_context_id="study-icd-preview",
                study_revision=2,
            ),
        )
    )


def test_dataio_icd_preview_reuses_extraction_filter_without_returning_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import easyicu.api as api_module
    import easyicu.patient_filter as patient_filter_module

    pd.DataFrame({"stay_id": [1, 2, 3], "hadm_id": [10, 20, 30]}).to_parquet(
        tmp_path / "icustays.parquet", index=False
    )
    pd.DataFrame(
        {
            "hadm_id": [10, 20, 30, 30],
            "icd_code": ["A419", "J189", "A410", "R650"],
        }
    ).to_parquet(tmp_path / "diagnoses_icd.parquet", index=False)

    class FakePatientFilter:
        def __init__(self, **_: object) -> None:
            self._last_original_count = 3

        def filter(self, **_: object) -> pd.DataFrame:
            return pd.DataFrame({"patient_id": [1, 2, 3]})

    monkeypatch.setattr(patient_filter_module, "PatientFilter", FakePatientFilter)
    monkeypatch.setattr(
        api_module, "get_id_col_for_database", lambda _database: "stay_id"
    )

    preview = dataio.preview_export_cohort(
        str(tmp_path),
        "miiv",
        {
            "preset": "all_icu",
            "icd_enabled": True,
            "icd_include": ["A41"],
            "icd_exclude": ["R65"],
        },
    )

    assert preview["cohort_size"] == 1
    assert preview["cohort_report"]["selected_before_icd"] == 3
    assert preview["cohort_report"]["icd"]["include_matches"] == 2
    assert preview["cohort_report"]["icd"]["exclude_matches"] == 1
    assert preview["privacy"] == {
        "patient_ids_returned": False,
        "raw_rows_returned": False,
        "host_path_returned": False,
    }
    encoded = json.dumps(preview)
    assert str(tmp_path) not in encoded
    assert "patient_ids" not in preview
    assert "id_col" not in preview


def test_registered_export_icd_preview_resolves_manifest_source_without_returning_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_path = tmp_path / "raw"
    raw_path.mkdir()
    export_path = tmp_path / "export"
    export_path.mkdir()
    (export_path / "_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu_native_export_v2",
                "database": "miiv",
                "data_path": str(raw_path),
                "cohort_contract": {
                    "preset": "all_icu",
                    "age_min": 18,
                    "exclude_readmissions": False,
                },
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def preview(path: str, database: str, cohort: dict) -> dict:
        captured.update(path=path, database=database, cohort=cohort)
        return {
            "schema_version": "easyicu_export_cohort_preview_v1",
            "database": database,
            "cohort_size": 10,
            "cohort_contract": cohort,
            "cohort_report": {"source_total": 140, "selected": 10},
            "privacy": {
                "patient_ids_returned": False,
                "raw_rows_returned": False,
                "host_path_returned": False,
            },
        }

    monkeypatch.setattr(dataio, "preview_export_cohort", preview)

    result = dataio.preview_registered_export_icd_cohort(
        str(export_path),
        "miiv",
        {},
        include_codes=["A41"],
        exclude_codes=[],
    )

    assert captured == {
        "path": str(raw_path.resolve()),
        "database": "miiv",
        "cohort": {
            "preset": "all_icu",
            "age_min": 18,
            "exclude_readmissions": False,
            "icd_enabled": True,
            "icd_include": ["A41"],
            "icd_exclude": [],
        },
    }
    encoded = json.dumps(result)
    assert str(raw_path) not in encoded
    assert str(export_path) not in encoded


def test_registered_export_icd_preview_fails_closed_when_source_path_is_missing(
    tmp_path: Path,
) -> None:
    export_path = tmp_path / "export"
    export_path.mkdir()
    (export_path / "_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu_native_export_v2",
                "database": "miiv",
                "data_path": str(tmp_path / "missing"),
                "cohort_contract": {"preset": "all_icu"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(dataio.ExportCohortError) as exc_info:
        dataio.preview_registered_export_icd_cohort(
            str(export_path),
            "miiv",
            {},
            include_codes=["A41"],
            exclude_codes=[],
        )

    assert exc_info.value.error == "icd_preview_source_path_unavailable"
    assert str(tmp_path) not in json.dumps(exc_info.value.detail)


def test_copilot_icd_tool_preserves_bound_filters_and_returns_snapshot_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "mimic"
    source_path.mkdir()
    store = CopilotDataWorkbenchSnapshotStore(tmp_path / "snapshots")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        tool_owner,
        "_registered_source_choice",
        lambda _context, _source_id=None: {
            "id": "source-mimic",
            "label": "MIMIC-IV",
            "database": "miiv",
            "path": str(source_path),
        },
    )
    monkeypatch.setattr(
        tool_owner,
        "_bound_context",
        lambda _binding: {
            "id": "study-icd-preview",
            "cohort": {
                "preset": "adult_first",
                "age_min": 40,
                "age_max": 85,
                "exclude_readmissions": True,
            },
        },
    )

    def preview(
        path: str,
        database: str,
        cohort: dict,
        *,
        include_codes: list[str],
        exclude_codes: list[str],
    ) -> dict:
        captured.update(
            path=path,
            database=database,
            cohort=cohort,
            include_codes=include_codes,
            exclude_codes=exclude_codes,
        )
        return {
            "schema_version": "easyicu_export_cohort_preview_v1",
            "database": "miiv",
            "cohort_size": 37,
            "cohort_contract": cohort,
            "cohort_report": {
                "source_total": 100,
                "selected_before_icd": 52,
                "selected": 37,
                "applied_filters": ["demographics", "icd"],
                "icd": {
                    "enabled": True,
                    "include_tokens": ["A41"],
                    "exclude_tokens": ["R65"],
                    "include_matches": 45,
                    "exclude_matches": 8,
                },
            },
            "privacy": {
                "patient_ids_returned": False,
                "raw_rows_returned": False,
                "host_path_returned": False,
            },
        }

    monkeypatch.setattr(
        tool_owner.dataio, "preview_registered_export_icd_cohort", preview
    )
    monkeypatch.setattr(tool_owner, "CopilotDataWorkbenchSnapshotStore", lambda: store)

    result = tool_owner.execute_tool(
        "easyicu_preview_icd_cohort",
        {
            "source_id": "source-mimic",
            "include_codes": ["A41"],
            "exclude_codes": ["R65"],
        },
        _context(),
    )

    assert result["code"] == "easyicu_icd_cohort_preview_ready"
    assert result["details"]["cohort_size"] == 37
    assert captured["path"] == str(source_path)
    assert captured["database"] == "miiv"
    assert captured["cohort"] == {
        "preset": "adult_first",
        "age_min": 40,
        "age_max": 85,
        "exclude_readmissions": True,
    }
    assert captured["include_codes"] == ["A41"]
    assert captured["exclude_codes"] == ["R65"]
    assert str(source_path) not in json.dumps(result)
    resource = result["details"]["resource"]
    assert resource["kind"] == "data_workbench_snapshot"
    assert resource["view"] == "icd_cohort_preview"
    snapshot = store.load(
        project_id="project-icd-preview", digest=resource["snapshot_sha256"]
    )
    assert snapshot["payload"]["summary"]["cohort_size"] == 37
    assert snapshot["payload"]["cohort_report"]["icd"]["include_tokens"] == [
        "A41"
    ]


def test_icd_preview_resource_replays_and_uses_extraction_renderer() -> None:
    projected = projections._project_replay_resource(
        {
            "kind": "data_workbench_snapshot",
            "view": "icd_cohort_preview",
            "snapshot_sha256": "a" * 64,
            "label": "ICD cohort preview",
        }
    )
    assert projected == {
        "kind": "data_workbench_snapshot",
        "view": "icd_cohort_preview",
        "snapshot_sha256": "a" * 64,
        "label": "ICD cohort preview",
        "media_type": "application/json",
    }

    static = Path(__file__).parents[1] / "src/easyicu/webserver/static"
    embedded = (static / "js/screens-viz-embedded.js").read_text(encoding="utf-8")
    css = (static / "css/guided-pi-data-preview.css").read_text(encoding="utf-8")
    assert "icd_cohort_preview" in embedded
    assert "data-gpi-icd-flow" in embedded
    assert "return 'extraction'" in embedded
    assert ".gpi-icd-flow" in css

    node = Path(__file__).parents[1] / "src/easyicu/webserver/pi_copilot/node_app/src"
    main = (node / "main.mjs").read_text(encoding="utf-8")
    event_projection = (node / "event-projection.mjs").read_text(encoding="utf-8")
    assert 'name: "easyicu_preview_icd_cohort"' in main
    assert "For an ICD-defined cohort count" in main
    assert '"icd_cohort_preview"' in event_projection
