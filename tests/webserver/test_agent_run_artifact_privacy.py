from __future__ import annotations

import json

import pytest

from fastapi.testclient import TestClient

from easyicu.webserver import agent_runs, run_artifact_disclosure
from easyicu.webserver.app import app


@pytest.mark.parametrize('summary', [
    'No analysis step reads stay_id, icu_unit_type, age.',
    'Input unavailable at /Users/synthetic/private.csv',
    'Bearer synthetic-only-credential',
    'stay_id: synthetic-row-123',
    '',
])
def test_unsafe_review_prose_is_withheld_without_relaxing_disclosure(summary):
    projected = run_artifact_disclosure.browser_review_summary(summary)
    assert projected != summary
    assert run_artifact_disclosure.scan_browser_projection({'summary': projected})['passed']
    # A different unsafe field still blocks the entire browser package.
    assert not run_artifact_disclosure.scan_browser_projection({
        'summary': projected, 'other': 'stay_id: synthetic-row-123',
    })['passed']


def test_review_summary_projection_preserves_the_bound_request_and_stop():
    from easyicu.research_agent.orchestration.workflow import HumanReviewRequest

    request = HumanReviewRequest.create(
        kind='scientific_stop', authority_sha256='a' * 64,
        summary='No analysis step reads stay_id, icu_unit_type, age.',
        payload={'reason': 'question_requirement_not_covered', 'approval_allowed': False},
    )
    before = request.model_dump(mode='json')
    projection = {
        'summary': run_artifact_disclosure.browser_review_summary(request.summary),
        'authority_sha256': request.authority_sha256,
        'reason': request.payload['reason'],
        'approval_allowed': request.payload['approval_allowed'],
    }
    assert projection['approval_allowed'] is False
    assert projection['reason'] == 'question_requirement_not_covered'
    assert projection['authority_sha256'] == before['authority_sha256']
    assert request.model_dump(mode='json') == before


def test_a_safe_complete_plan_review_summary_remains_visible():
    summary = 'Review and explicitly approve the complete analysis plan before execution.'
    assert run_artifact_disclosure.browser_review_summary(summary) == summary


def test_review_api_withholds_unsafe_summary_and_keeps_approval_blocked(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
    from easyicu.webserver import agent_pipeline_runs
    from tests.webserver.copilot.research_workflow_fixtures import complete_study
    from tests.support.plan_review_fixtures import pending_plan_review as _pending

    pending = _pending(tmp_path)
    original = pending.requests[0]
    request = HumanReviewRequest.create(
        kind=original.kind, authority_sha256=original.authority_sha256,
        summary='No analysis step reads stay_id, icu_unit_type, age.',
        payload={**original.payload, 'reason': 'question_requirement_not_covered',
                 'approval_allowed': False},
    )
    pending = pending.model_copy(update={'requests': (request,)})
    entry = SimpleNamespace(
        pending=pending, study=complete_study(), credential_source='pi_verified',
        budget_mode='full_reviewed', provider={}, wrapper_dir=tmp_path,
    )
    monkeypatch.setattr(agent_pipeline_runs, '_PENDING_REVIEWS', {pending.run_id: entry})
    # Compare every existing review control with the former prose projection.
    # The minimal fixture has a stale scientific policy, which must keep its
    # higher-priority refusal rather than being changed by display redaction.
    with monkeypatch.context() as original_display:
        original_display.setattr(run_artifact_disclosure, 'browser_review_summary', lambda value: value)
        before = agent_pipeline_runs.pending_review(pending.run_id)['requests'][0]
    projected = agent_pipeline_runs.pending_review(pending.run_id)
    public_request = projected['requests'][0]
    assert public_request['summary'] != request.summary
    assert public_request['approval_allowed'] is False
    assert {key: value for key, value in public_request.items() if key != 'summary'} == {
        key: value for key, value in before.items() if key != 'summary'
    }
    assert public_request['review_id'] == request.review_id
    assert public_request['authority_sha256'] == request.authority_sha256
    assert run_artifact_disclosure.scan_browser_projection({'request': public_request})['passed']


def test_disclosure_owner_applies_both_browser_and_artifact_policies() -> None:
    browser_scan = run_artifact_disclosure.scan_browser_projection(
        {"run_context.json": {"reason": "/Users/reviewer/private.csv"}}
    )
    artifact_scan = run_artifact_disclosure.scan_artifact_payloads(
        {"figure_gallery.json": {"patient_rows": [{"age": 72}]}}
    )

    assert browser_scan["passed"] is False
    assert artifact_scan["passed"] is False
    blocked = run_artifact_disclosure.privacy_blocked_projection(
        run_context={"run_id": "run-safe", "study_id": "study-safe"},
        scan=browser_scan,
    )
    assert blocked["quality_gate.json"]["gate"]["reportable"] is False
    assert "/Users/reviewer" not in json.dumps(blocked)


def test_row_level_json_is_withheld_from_preview_download_and_bundle(tmp_path) -> None:
    artifact_path = tmp_path / "figure_gallery.json"
    artifact_path.write_text(
        json.dumps({"patient_rows": [{"stay_id": "synthetic-secret"}]}),
        encoding="utf-8",
    )

    preview = agent_runs.read_run_artifact(str(tmp_path), artifact_path.name)
    download = agent_runs.read_run_artifact_bytes(str(tmp_path), artifact_path.name)
    bundle = agent_runs.build_run_bundle(str(tmp_path))

    for result in (preview, download, bundle):
        assert result["ok"] is False
        assert result["error"] == "artifact_privacy_scan_failed"
        assert result["privacy_scan"]["passed"] is False
        assert "payload" not in result
        assert "content" not in result


def test_patient_rows_are_withheld_even_without_explicit_identifiers(tmp_path) -> None:
    artifact_path = tmp_path / "figure_gallery.json"
    artifact_path.write_text(
        json.dumps({"patient_rows": [{"age": 72, "sex": "F"}]}),
        encoding="utf-8",
    )

    result = agent_runs.read_run_artifact(str(tmp_path), artifact_path.name)

    assert result["ok"] is False
    assert result["error"] == "artifact_privacy_scan_failed"
    assert result["privacy_scan"]["row_level_markers"] == [
        {"path": "figure_gallery.json.patient_rows", "marker": "patient_rows"}
    ]


def test_row_level_json_routes_return_conflict_without_payload(tmp_path) -> None:
    artifact_path = tmp_path / "figure_gallery.json"
    artifact_path.write_text(
        json.dumps({"patient_rows": [{"stay_id": "synthetic-secret"}]}),
        encoding="utf-8",
    )
    client = TestClient(app)

    for endpoint in (
        "/api/agent-runs/artifact",
        "/api/agent-runs/download-artifact",
        "/api/agent-runs/download-bundle",
    ):
        response = client.post(
            endpoint,
            json={"project_dir": str(tmp_path), "artifact": artifact_path.name},
        )
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["error"] == "artifact_privacy_scan_failed"
        assert detail["privacy_scan"]["passed"] is False
        assert "synthetic-secret" not in response.text
