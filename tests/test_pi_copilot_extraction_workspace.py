"""Focused contract tests for Copilot's native Extraction workspace handoff."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.webserver import dataio
from easyicu.webserver.pi_copilot import projections
from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    PiSessionRecord,
    ToolExecutionContext,
)


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-extraction-workspace",
            binding=AuthorityBinding(
                study_context_id="study-extraction-workspace",
                study_revision=3,
            ),
        ),
        allowed_actions={"extract"},
    )


def test_incomplete_extraction_opens_path_free_native_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_module,
        "_bound_context",
        lambda binding: {
            "id": "study-extraction-workspace",
            "revision": 3,
            "data_source": {},
            "modules": [],
        },
    )
    monkeypatch.setattr(
        tool_module.sources,
        "load_registry",
        lambda: {"active_path": None, "sources": []},
    )

    result = tool_module.execute_tool("easyicu_start_extraction", {}, _context())

    assert result["code"] == "extraction_setup_incomplete"
    resource = result["details"]["resource"]
    assert resource == {
        "kind": "native_workspace",
        "route": "extraction",
        "state": "setup",
        "study_context_id": "study-extraction-workspace",
        "study_revision": 3,
        "label": "Data Extraction",
        "media_type": "application/vnd.easyicu.native-workspace",
    }
    assert "/private/" not in json.dumps(result).lower()


def test_submitted_extraction_workspace_carries_only_job_coordinate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_module,
        "_bound_context",
        lambda binding: {
            "id": "study-extraction-workspace",
            "revision": 3,
            "title": "Local extraction",
            "data_source": {
                "path": "/private/local/icu-source",
                "database": "miiv",
            },
            "modules": ["demographics", "vitals"],
            "cohort": {"preset": "adult_first"},
            "export_format": "parquet",
        },
    )
    monkeypatch.setattr(
        tool_module.sources,
        "load_registry",
        lambda: {"active_path": None, "sources": []},
    )
    from easyicu.webserver.routes import jobs as jobs_route

    monkeypatch.setattr(
        jobs_route,
        "jobs_extract",
        lambda body: {
            "job_id": "extract-job-42",
            "kind": "extract",
            "status": "running",
            "study_context_id": "study-extraction-workspace",
            "study_context_revision": 4,
        },
    )

    result = tool_module.execute_tool("easyicu_start_extraction", {}, _context())

    assert result["code"] == "easyicu_extraction_submitted"
    assert result["details"]["resource"]["job_id"] == "extract-job-42"
    assert result["details"]["resource"]["state"] == "running"
    assert "/private/local/icu-source" not in json.dumps(result)


def test_replay_projection_preserves_native_workspace_coordinates() -> None:
    projected = projections._project_replay_resource(
        {
            "kind": "native_workspace",
            "route": "extraction",
            "state": "running",
            "study_context_id": "study-extraction-workspace",
            "study_revision": 3,
            "job_id": "extract-job-42",
            "label": "Data Extraction",
        }
    )

    assert projected == {
        "kind": "native_workspace",
        "route": "extraction",
        "state": "running",
        "study_context_id": "study-extraction-workspace",
        "study_revision": 3,
        "label": "Data Extraction",
        "media_type": "application/vnd.easyicu.native-workspace",
        "job_id": "extract-job-42",
    }
    assert projections._project_replay_resource(
        {
            "kind": "native_workspace",
            "route": "settings",
            "state": "running",
            "study_context_id": "study-extraction-workspace",
            "study_revision": 3,
        }
    ) is None


def test_native_workspace_uses_extraction_owner_and_mimic_safe_recommendation() -> None:
    static_root = (
        Path(__file__).parents[1]
        / "src"
        / "easyicu"
        / "webserver"
        / "static"
    )
    extraction = (static_root / "js" / "screens-extraction.js").read_text()
    embedded = (static_root / "js" / "screens-extraction-embedded.js").read_text()
    preview = (static_root / "js" / "screens-guided-pi-preview.js").read_text()
    resources = (static_root / "js" / "screens-guided-pi-resources.js").read_text()

    assert "window.EU_EXTRACTION_NATIVE_OWNER" in extraction
    assert "window.EU_EXTRACTION_EMBEDDED_WORKSPACE" in embedded
    assert "Prepared export is ready — sync to Copilot" in embedded
    assert "recommendedUsesFirstStay" in extraction
    assert "['miiv', 'miii', 'mimic', 'mimiciv', 'mimiciii']" in extraction
    assert "preset: firstStay ? 'adult_first' : 'all_icu'" in extraction
    assert "exclude_readmissions: firstStay" in extraction
    assert "window.EU_EXTRACTION_EMBEDDED_WORKSPACE" in preview
    assert "typeof owner.mount === 'function'" in preview
    assert "native_workspace" in resources

    index = (static_root / "index.html").read_text()
    assert 'screens-extraction-embedded.js?v=20260823-extraction-workspace4' in index
    assert index.index("screens-extraction.js") < index.index("screens-extraction-embedded.js")


def test_mimic_adult_recommendation_does_not_request_unavailable_first_stay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.patient_filter as patient_filter_module

    captured: dict[str, object] = {}

    class FakePatientFilter:
        def __init__(self, **_: object) -> None:
            self._last_original_count = 2

        def filter(self, **kwargs: object) -> pd.DataFrame:
            captured.update(kwargs)
            return pd.DataFrame({"patient_id": [101, 102]})

    class FakeApi:
        @staticmethod
        def get_id_col_for_database(_: str) -> str:
            return "stay_id"

    monkeypatch.setattr(patient_filter_module, "PatientFilter", FakePatientFilter)

    resolved = dataio._resolve_export_cohort(
        "/private/mimic",
        "miiv",
        {
            "preset": "all_icu",
            "age_min": 18,
            "age_max": 100,
            "exclude_readmissions": False,
        },
        500,
        FakeApi(),
    )

    assert captured["age_min"] == 18
    assert captured["first_icu_stay"] is None
    assert resolved["cohort_size"] == 2
