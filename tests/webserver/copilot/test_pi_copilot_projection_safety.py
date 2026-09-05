"""Path, identifier, and secret projection contracts for Pi Copilot."""

from __future__ import annotations
import json
import pytest
from easyicu.webserver.pi_copilot.contracts import (
    PiCopilotError,
    PiSessionRecord,
    ToolExecutionContext,
)
from easyicu.webserver.pi_copilot.projections import (
    ensure_safe_projection,
    project_job,
    project_study_context,
    reject_sensitive_message,
)
from easyicu.webserver.pi_copilot.message_input import prepare_user_message
from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot import workflow as workflow_module


def test_phi_and_projection_boundaries_reject_rows_identifiers_and_paths() -> None:
    with pytest.raises(PiCopilotError, match="row-level"):
        reject_sensitive_message("Please inspect patient_id=12345")
    with pytest.raises(PiCopilotError) as raw_rows:
        ensure_safe_projection({"rows": [{"value": 1}]})
    assert raw_rows.value.code == "pi_projection_blocked"
    with pytest.raises(PiCopilotError) as raw_path:
        ensure_safe_projection({"path": "/private/export"})
    assert raw_path.value.code == "pi_projection_blocked"
    for unsafe_value in (
        "failed reading /Users/researcher/patient_12345/raw.csv",
        "Authorization: Bearer secret-token-value",
        'row fragment: {"subject_id": 12345, "value": 7.1}',
    ):
        with pytest.raises(PiCopilotError) as unsafe_string:
            ensure_safe_projection({"reason": unsafe_value})
        assert unsafe_string.value.code == "pi_projection_blocked"


def test_registered_local_data_path_is_kept_host_side() -> None:
    source = {
        "id": "src-mimic",
        "path": "/Volumes/research/easyicu/miiv",
        "label": "MIMIC-IV",
        "database": "miiv",
        "ok": True,
    }

    prepared = prepare_user_message(
        "研究 Sepsis-3，数据目录是 /Volumes/research/easyicu/miiv。请生成计划。",
        registered_sources=[source],
    )

    assert prepared.registered_source == source
    assert "/Volumes/" not in prepared.provider_message
    assert "EasyICU host-verified local data source: MIMIC-IV" in prepared.provider_message


def test_unregistered_local_path_is_not_forwarded_to_provider() -> None:
    with pytest.raises(PiCopilotError) as exc_info:
        prepare_user_message(
            "数据目录是 /Volumes/research/unregistered/raw。",
            registered_sources=[],
        )

    assert exc_info.value.code == "pi_message_local_path_unregistered"

    projected = project_study_context(
        {
            "id": "study-safe",
            "revision": 1,
            "question": "Aggregate lactate analysis",
            "primary_exposure": "lactate",
            "covariates": ["age", "sex"],
            "sensitivity_specs": [
                {
                    "spec_id": "landmark_24h",
                    "axis": "timing",
                    "strategy": "landmark",
                    "execution_variables": [],
                    "landmark_hours": 24,
                    "require_alive_at_landmark": True,
                    "exclude_negative_event_times": True,
                }
            ],
            "data_source": {"database": "mimiciv", "path": "/private/export"},
            "cohort": {"cohort_size": 140},
            "literature_authority": {
                "schema_version": "easyicu.web-literature-authority/2",
                "receipt_id": "lit_" + "a" * 24,
                "receipt_sha256": "b" * 64,
                "status": "searched",
                "result_count": 3,
                "searched_at": "2026-08-12T12:00:00+00:00",
                "study_configuration_sha256": "c" * 64,
            },
        }
    )
    assert "/private/export" not in json.dumps(projected)
    assert len(projected["data_source"]["path_digest"]) == 32
    assert projected["primary_exposure"] == "lactate"
    assert projected["covariates"] == ["age", "sex"]
    assert projected["sensitivity_specs"][0]["spec_id"] == "landmark_24h"
    assert projected["literature_authority"]["result_count"] == 3
    assert "/private/" not in json.dumps(projected)

    workflow_receipt = workflow_module.project_study_setup_receipt(
        {
            "id": "study-safe",
            "revision": 1,
            "question": "Aggregate lactate analysis",
            "data_source": {"database": "mimiciv", "path": "/private/export"},
        }
    )
    assert workflow_receipt.configuration["data_source"]["path_digest"] == projected[
        "data_source"
    ]["path_digest"]

    projected_job = project_job(
        {
            "id": "job-safe",
            "status": "failed",
            "cancel_reason": "/Users/reviewer/private.csv",
            "events": [
                {
                    "seq": 1,
                    "type": "progress",
                    "label": "patient_id=123",
                    "reason": "/private/raw.csv",
                }
            ],
        }
    )
    encoded_job = json.dumps(projected_job)
    assert "private.csv" not in encoded_job
    assert "patient_id" not in encoded_job
    assert projected_job["progress"][0]["reason_code"] is None


def test_validation_projection_is_owner_specific_and_value_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ToolExecutionContext(session=PiSessionRecord(session_id="pi-safe"))
    monkeypatch.setattr(
        tool_module,
        "_select_run",
        lambda context, requested_run_id=None: {
            "run_id": "run-safe",
            "project_dir": "/private/not-projected",
        },
    )
    monkeypatch.setattr(
        tool_module,
        "_run_review",
        lambda row: {
            "gate": {
                "status": "blocked",
                "reason": "/Users/reviewer/patient_123.csv",
                "checks": [
                    {"id": "numeric_evidence_missing", "passed": False},
                    {"id": "unsafe /private/path", "passed": False},
                ],
                "nested": {"raw": "patient_id=123"},
            },
            "readiness": {
                "status": "blocked",
                "reason": "Bearer hidden-secret",
                "non_human_failures": [
                    "evidence_not_ready",
                    "/private/source.csv",
                ],
            },
        },
    )

    result = tool_module.execute_tool(
        "easyicu_inspect_validation",
        {"run_id": "run-safe"},
        context,
    )

    encoded = json.dumps(result)
    assert "/Users" not in encoded
    assert "/private" not in encoded
    assert "patient_id" not in encoded
    assert "Bearer" not in encoded
    assert result["details"]["gate"]["failed_requirement_codes"] == [
        "numeric_evidence_missing"
    ]


def test_complete_tool_result_sanitizes_summary_and_authority_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ToolExecutionContext(session=PiSessionRecord(session_id="pi-safe"))
    monkeypatch.setattr(
        tool_module,
        "_select_run",
        lambda context, requested_run_id=None: {
            "run_id": "/Users/reviewer/private-run",
        },
    )

    with pytest.raises(PiCopilotError) as caught:
        tool_module.execute_tool("easyicu_inspect_run", {}, context)

    assert caught.value.code == "pi_projection_blocked"
