"""The HTTP walk never persists server text or acquires execution implicitly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.walk.contracts import ItemResult, Status
from tools.walk.http import HTTPFailure, HttpAPI
from tools.walk.report import ReceiptStore, redact, run_metrics
from tools.walk.runner import Options, WalkRunner
from tests.support.walk_http import SyntheticAPI, walk_options


def test_results_require_a_typed_status_and_a_reason_for_nonpasses():
    with pytest.raises(ValueError):
        ItemResult("environment", "health", "success", "", 0, 0, {})
    with pytest.raises(ValueError):
        ItemResult("environment", "health", Status.SKIPPED, "", 0, 0, {})
    assert ItemResult("environment", "health", Status.PASSED, "", 0, 0, {}).to_dict()["status"] == "passed"


@pytest.mark.parametrize("started,elapsed", [(float("nan"), 1), (0, float("inf")), (-1, 0), (0, -1)])
def test_result_timings_are_finite_and_nonnegative(started, elapsed):
    with pytest.raises(ValueError, match="timing"):
        ItemResult("environment", "health", Status.PASSED, "", started, elapsed, {})


def test_run_projection_keeps_names_and_counts_without_rows_or_error_text():
    payload = {"latest_run": {
        "present": True, "gate_status": "pass", "gate_reason_code": "analysis_complete",
        "artifact_refs": [
            {"artifact": "analysis.csv", "patient_id": "SECRET-PATIENT", "rows": [{"lact": 17}]},
            {"artifact": "manuscript.md"}, {"artifact": "evidence.json"},
            {"artifact": "patient_917355.csv"},
        ],
        "error": "subject_id=SECRET-PATIENT", "artifact_payloads": {"rows": [1]},
    }}
    result = run_metrics(payload)
    encoded = json.dumps(result)
    assert result["artifact_count"] == 4
    assert result["artifact_counts_by_type"] == {"csv": 2, "md": 1, "json": 1}
    assert result["report_present"] is True
    assert result["evidence_present"] is True
    assert "SECRET" not in encoded and "917355" not in encoded
    assert "patient_id" not in encoded and "artifact_payloads" not in encoded
    assert "17" not in encoded


@pytest.mark.parametrize("text", ["stay_id=987654", "subject_id: secret-X", "MRN 321999", "patient_321999.csv"])
def test_redaction_removes_identifier_values(text):
    assert "987654" not in redact(text)
    assert "secret-X" not in redact(text)
    assert "321999" not in redact(text)


@pytest.mark.parametrize("text", ["CaseID: 7654321", "patientid=7654321", "hospitaladmissionid 7654321"])
def test_source_specific_identifier_spellings_are_redacted(text):
    assert "7654321" not in redact(text)


def test_numeric_artifact_basename_cannot_leak_an_identifier():
    metrics = run_metrics({"latest_run": {"artifact_refs": [{"artifact": "7654321.csv"}]}})
    assert "7654321" not in json.dumps(metrics)


def test_resume_receipts_are_scoped_to_server_source_and_approval(tmp_path):
    store = ReceiptStore(tmp_path / "report", "server-A/source-A", resume=False)
    result = ItemResult("workflows", "one", Status.PASSED, "", 0, 1, {"artifact_count": 2})
    store.save(result, policy="plan-only")
    assert store.finished("workflows", "one", policy="plan-only") == result
    assert store.finished("workflows", "one", policy="approved") is None
    with pytest.raises(ValueError, match="scope"):
        ReceiptStore(tmp_path / "report", "server-B/source-A", resume=True)
    with pytest.raises(ValueError, match="resume"):
        ReceiptStore(tmp_path / "report", "server-A/source-A", resume=False)


def test_http_rejects_foreign_hosts_and_redirects_without_following_them():
    calls = []

    def exchange(method, url, body, timeout):
        calls.append(url)
        return 302, {"Location": "https://foreign.invalid/rows"}, b"patient_id=8675309"

    api = HttpAPI("http://localhost:8780", exchange=exchange)
    with pytest.raises(ValueError):
        api.request("GET", "https://foreign.invalid/rows")
    with pytest.raises(HTTPFailure) as error:
        api.request("GET", "/api/health")
    assert error.value.code == "http_redirect_refused"
    assert "8675309" not in str(error.value)
    assert calls == ["http://localhost:8780/api/health"]


def test_walk_stops_at_review_without_approval_and_resume_does_not_duplicate(tmp_path):
    api = SyntheticAPI()
    options = walk_options(tmp_path)
    report = WalkRunner(options, api=api).run()
    row = next(row for row in report["items"] if row["phase"] == "workflows")
    assert row["status"] == "skipped-with-reason"
    assert row["reason_code"] == "approval_flag_required"
    assert row["metrics"]["plan_valid"] is True
    assert not api.approved
    assert not any(path == "/api/jobs/agent-run-review" for _, path, _, _ in api.calls)
    assert "Describe synthetic ICU admission measurements?" in api.question
    previous_calls = len(api.calls)
    resumed = WalkRunner(walk_options(tmp_path, resume=True), api=api).run()
    assert len(api.sessions) == 1
    assert not any(method != "GET" for method, _, _, _ in api.calls[previous_calls:])
    assert [r for r in resumed["items"] if r["phase"] == "workflows"] == [
        r for r in report["items"] if r["phase"] == "workflows"]
    for path in options.out.rglob("*"):
        if path.is_file():
            assert "NEVER-SAVE-THIS" not in path.read_text()


def test_execution_requires_flag_and_current_digest_bound_offer(tmp_path):
    api = SyntheticAPI()
    report = WalkRunner(walk_options(tmp_path, approve_test_runs=True), api=api).run()
    assert api.approved
    row = next(row for row in report["items"] if row["phase"] == "workflows")
    assert row["status"] == "passed"
    approval = next(body for _, path, body, _ in api.calls if path == "/api/jobs/agent-run-review")
    assert approval["host_action"]["decision"]["review_authority_sha256"] == "b" * 64
    assert approval["host_action"]["project_id"] == "project-walk"


def test_foreign_session_is_refused_before_any_message_or_mutation(tmp_path):
    class ForeignAPI(SyntheticAPI):
        def request(self, method, path, body=None, **kwargs):
            result = super().request(method, path, body, **kwargs)
            if path == "/api/copilot/pi/sessions":
                result["session"]["project_id"] = "somebody-elses-project"
            return result

    api = ForeignAPI()
    report = WalkRunner(walk_options(tmp_path), api=api).run()
    row = next(row for row in report["items"] if row["phase"] == "workflows")
    assert row["status"] == "refused-with-reason"
    assert row["reason_code"] == "foreign_project_refused"
    assert not any(path.endswith(("/message", "/delete", "/rename")) for _, path, _, _ in api.calls)


def test_output_inside_repository_is_refused_before_any_api_call(tmp_path):
    api = SyntheticAPI()
    with pytest.raises(ValueError, match="outside"):
        WalkRunner(Options(source="demo", out=Path(__file__).resolve().parents[2] / "unsafe-walk"), api=api)
    assert api.calls == []
