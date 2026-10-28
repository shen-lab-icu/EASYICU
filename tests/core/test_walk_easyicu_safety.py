"""Fail-closed transport, persistence, deadline and execution boundaries."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tools.walk.cli import main
from tools.walk.contracts import ItemResult, Status
from tools.walk.http import HTTPFailure, HttpAPI, _NoRedirect
from tools.walk.report import ReceiptStore
from tools.walk.runner import WalkRunner
from tests.support.walk_http import SyntheticAPI, walk_options


@pytest.mark.parametrize("base", ["file:///tmp/data", "http://user:secret@localhost", "http://localhost/x", "http://localhost?key=secret"])
def test_base_cannot_include_credentials_or_non_http_locations(base):
    with pytest.raises(ValueError):
        HttpAPI(base)


@pytest.mark.parametrize("path", ["//foreign.invalid/api/health", "/api/../data", "/api/%2e%2e/data", "/api/health?next=secret"])
def test_request_paths_cannot_smuggle_redirects_or_traversal(path):
    api = HttpAPI("http://localhost", exchange=lambda *args: pytest.fail("transport must not be called"))
    with pytest.raises(ValueError):
        api.request("GET", path)


def test_response_error_text_is_never_used_as_a_reason():
    api = HttpAPI("http://localhost", exchange=lambda *args: (
        403, {}, b'{"detail":{"error":"patient_id=1234999","reason":"sensitive response"}}'))
    with pytest.raises(HTTPFailure) as caught:
        api.request("GET", "/api/health")
    assert str(caught.value) == "http_403"


@pytest.mark.parametrize("payload,code", [(b"[]", "http_object_required"), (b"<html>secret</html>", "http_invalid_json")])
def test_non_object_and_invalid_json_responses_have_only_stable_codes(payload, code):
    api = HttpAPI("http://localhost", exchange=lambda *args: (200, {}, payload))
    with pytest.raises(HTTPFailure) as caught:
        api.request("GET", "/api/health")
    assert str(caught.value) == code


def test_response_size_is_bounded_before_json_parsing(monkeypatch):
    from tools.walk import http

    monkeypatch.setattr(http, "MAX_RESPONSE_BYTES", 4)
    api = HttpAPI("http://localhost", exchange=lambda *args: (200, {}, b"sensitive"))
    with pytest.raises(HTTPFailure, match="http_response_too_large"):
        api.request("GET", "/api/health")


def test_default_opener_cannot_follow_a_redirect_or_use_environment_proxies(monkeypatch):
    import urllib.request

    monkeypatch.setattr(urllib.request, "getproxies", lambda: pytest.fail("environment proxies are forbidden"))
    HttpAPI("http://localhost")
    request = urllib.request.Request("http://localhost/api/health")
    assert _NoRedirect().redirect_request(request, None, 302, "", {}, "https://foreign.invalid/api") is None


def test_request_deadline_expires_before_transport(tmp_path):
    api = SyntheticAPI()
    runner = WalkRunner(walk_options(tmp_path), api=api)
    runner._deadline = 0
    with pytest.raises(TimeoutError):
        runner.request("GET", "/api/health")
    assert api.calls == []


def test_approval_flag_is_also_enforced_at_the_network_boundary(tmp_path):
    runner = WalkRunner(walk_options(tmp_path), api=SyntheticAPI())
    with pytest.raises(Exception, match="approval_flag_required"):
        runner.request("POST", "/api/jobs/agent-run-review", {})


@pytest.mark.parametrize("ambiguous", [False, True])
def test_source_must_be_configured_and_unambiguous_before_project_creation(tmp_path, ambiguous):
    class InvalidSourceAPI(SyntheticAPI):
        def request(self, method, path, body=None, **kwargs):
            payload = super().request(method, path, body, **kwargs)
            if path == "/api/workspaces/registry":
                payload["sources"] = ([{"id": "one", "label": "demo", "database": "sic", "ok": True},
                                       {"id": "two", "label": "demo", "database": "sic", "ok": True}]
                                      if ambiguous else [])
            return payload

    api = InvalidSourceAPI()
    report = WalkRunner(walk_options(tmp_path), api=api).run()
    assert report["items"][0]["reason_code"] == ("source_ambiguous" if ambiguous else "configured_source_not_found")
    assert not any(method != "GET" for method, _, _, _ in api.calls)


def test_resume_does_not_reapprove_an_execution_that_timed_out(tmp_path):
    class InterruptedAPI(SyntheticAPI):
        stalled = True

        def request(self, method, path, body=None, **kwargs):
            if path == "/api/jobs/review-job" and self.stalled:
                raise TimeoutError("patient_id=DO-NOT-LOG")
            return super().request(method, path, body, **kwargs)

    api = InterruptedAPI()
    options = walk_options(tmp_path, approve_test_runs=True)
    first = WalkRunner(options, api=api).run()
    assert first["items"][-1]["status"] == "timed-out"
    assert first["items"][-1]["metrics"]["plan_valid"] is True
    api.stalled = False
    second = WalkRunner(replace(options, resume=True), api=api).run()
    assert second["items"][-1]["status"] == "passed"
    assert sum(path == "/api/jobs/agent-run-review" for _, path, _, _ in api.calls) == 1
    assert "DO-NOT-LOG" not in (options.out / "walk_report.json").read_text()


def test_resume_reports_catalogue_discovery_failure_instead_of_reusing_old_success(tmp_path):
    class DisconnectedAPI(SyntheticAPI):
        disconnected = False

        def request(self, method, path, body=None, **kwargs):
            if path == "/api/capabilities" and self.disconnected:
                raise HTTPFailure(503, "catalogue_unavailable")
            return super().request(method, path, body, **kwargs)

    api = DisconnectedAPI()
    options = walk_options(tmp_path)
    WalkRunner(options, api=api).run()
    api.disconnected = True
    report = WalkRunner(replace(options, resume=True), api=api).run()
    assert report["items"][-1]["status"] == "refused-with-reason"
    assert report["items"][-1]["reason_code"] == "catalogue_unavailable"


def test_unbounded_polling_obeys_the_phase_deadline(tmp_path):
    class BusyAPI(SyntheticAPI):
        def request(self, method, path, body=None, **kwargs):
            result = super().request(method, path, body, **kwargs)
            if path.startswith("/api/copilot/pi/sessions/") and path.count("/") == 5:
                result["session"]["streaming"] = True
            return result

    options = replace(walk_options(tmp_path), timeouts={"workflows": 0.01}, poll_interval=0.001)
    report = WalkRunner(options, api=BusyAPI()).run()
    assert report["items"][-1]["status"] == "timed-out"
    assert report["items"][-1]["elapsed_seconds"] < 1


def test_missing_approval_offer_cannot_execute_even_with_flag(tmp_path):
    class NoOfferAPI(SyntheticAPI):
        def request(self, method, path, body=None, **kwargs):
            result = super().request(method, path, body, **kwargs)
            if path.endswith("/workflow"):
                result["host_decisions"] = {}
            return result

    api = NoOfferAPI()
    report = WalkRunner(walk_options(tmp_path, approve_test_runs=True), api=api).run()
    assert report["items"][-1]["reason_code"] == "current_host_decision_required"
    assert not api.approved


def test_network_boundary_refuses_non_candidate_plan_submission(tmp_path):
    runner = WalkRunner(walk_options(tmp_path), api=SyntheticAPI())
    with pytest.raises(Exception, match="candidate_plan_only_required"):
        runner.request("POST", "/api/jobs/agent-run", {"candidate_plan_only": False})


@pytest.mark.parametrize("method,path,body", [
    ("POST", "/api/settings", {"ai_enabled": True}),
    ("POST", "/api/copilot/pi/sessions/foreign/delete", {"project_id": "foreign", "confirm": True}),
    ("GET", "/api/copilot/pi/projects/foreign/workflow", None),
    ("GET", "/api/copilot/pi/projects/foreign/runs/run/artifacts/result_tables.json", None),
])
def test_network_boundary_only_allows_walk_owned_metadata_and_mutations(tmp_path, method, path, body):
    api = SyntheticAPI()
    runner = WalkRunner(walk_options(tmp_path), api=api)
    with pytest.raises(Exception, match="(?:walk_scope|unsafe_api_route)"):
        runner.request(method, path, body)
    assert api.calls == []


def test_output_symlink_into_repository_is_refused(tmp_path):
    from pathlib import Path

    link = tmp_path / "link"
    link.symlink_to(Path(__file__).resolve().parents[2], target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        WalkRunner(replace(walk_options(tmp_path), out=link), api=SyntheticAPI())


def test_resume_rejects_row_payload_in_state_instead_of_republishing_it(tmp_path):
    out = tmp_path / "walk"
    store = ReceiptStore(out, "scope", resume=False)
    state = json.loads(store.path.read_text())
    state["rows"] = [{"patient_id": "secret-patient"}]
    store.path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="state"):
        ReceiptStore(out, "scope", resume=True)


def test_receipts_reject_untyped_metrics_and_corruption(tmp_path):
    store = ReceiptStore(tmp_path / "walk", "scope", resume=False)
    result = ItemResult("mapping", "dictionary", Status.PASSED, "", 0, 0, {
        "mapped": 3, "rows": [{"patient_id": "secret"}], "artifact_names": ["stay_id-999999.csv"]})
    store.save(result, policy="plan-only")
    receipt = next(store.receipts.iterdir())
    assert "secret" not in receipt.read_text() and "999999" not in receipt.read_text()
    receipt.write_text("{broken")
    assert store.finished("mapping", "dictionary", policy="plan-only") is None


def test_cli_validates_phases_and_deadlines_without_running(tmp_path):
    for arguments in (("--only", "unknown"), ("--timeout-workflows", "nan"), ("--max-workflows", "0")):
        with pytest.raises(SystemExit) as caught:
            main(["--source", "demo", "--out", str(tmp_path / "walk"), *arguments],
                 runner_factory=lambda *_: pytest.fail("must validate before running"))
        assert caught.value.code == 2


def test_cli_reports_only_status_counts_and_returns_failure(tmp_path, capsys):
    class FailedRunner:
        def __init__(self, options):
            assert options.max_workflows == 2

        def run(self):
            return {"status_counts": {"passed": 1, "failed": 2}}

    assert main(["--source", "demo", "--out", str(tmp_path), "--max-workflows", "2"], runner_factory=FailedRunner) == 1
    assert capsys.readouterr().out == "failed: 2; passed: 1\n"
