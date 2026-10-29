"""Real HTTP response shapes, exercised with small synthetic walk fakes only."""

from dataclasses import replace
import json

import pytest

from tools.walk.contracts import ItemResult, Status, WalkStop
from tools.walk.conversations import _identity
from tools.walk.http import HTTPFailure
from tools.walk.report import ReceiptStore
from tools.walk.runner import WalkRunner
from tests.support.walk_http import PhaseAPI, walk_options


@pytest.mark.parametrize("database,parent", [("eicu_demo", "eicu"), ("mimic_demo", "mimic"), ("sic", "sic")])
def test_dictionary_uses_the_declared_database_family_not_the_source_label(tmp_path, database, parent):
    class DemoAPI(PhaseAPI):
        def request(self, method, path, body=None, **kwargs):
            payload = super().request(method, path, body, **kwargs)
            if path == "/api/workspaces/registry":
                payload["sources"][0].update(database=database, label="mimic_demo")
            if path == "/api/data-map":
                assert kwargs["params"].get("database") in {None, parent}
                payload["databases"][0]["database"] = parent
            if path.endswith("/projection"):
                assert kwargs["params"]["database"] == parent
                payload["databases"][0]["database"] = parent
            return payload

    options = replace(walk_options(tmp_path), only=("environment", "mapping"))
    report = WalkRunner(options, api=DemoAPI()).run()
    rows = {row["item_id"]: row for row in report["items"]}
    for item in ("source-capabilities", "dictionary-availability"):
        assert rows[item]["status"] == "passed", rows[item]
        assert rows[item]["metrics"]["database"] == database
        assert rows[item]["metrics"]["profile_database"] == parent
    assert rows["extraction-smoke"]["metrics"]["module_count"] == 19


@pytest.mark.parametrize("database", ["unknown_demo", "unknown", "eicu_demo"])
def test_missing_public_parent_is_skipped_before_a_filtered_metadata_request(tmp_path, database):
    class UnknownAPI(PhaseAPI):
        def request(self, method, path, body=None, **kwargs):
            payload = super().request(method, path, body, **kwargs)
            if path == "/api/workspaces/registry":
                payload["sources"][0]["database"] = database
            if path == "/api/data-map":
                assert "database" not in kwargs["params"]
            if path.endswith("/projection"):
                pytest.fail("a source without an advertised public parent must be skipped")
            return payload

    options = replace(walk_options(tmp_path), only=("environment", "mapping"))
    report = WalkRunner(options, api=UnknownAPI()).run()
    rows = {row["item_id"]: row for row in report["items"]}
    for item in ("source-capabilities", "dictionary-availability", "extraction-smoke"):
        assert rows[item]["status"] == "skipped-with-reason", rows[item]
        assert rows[item]["reason_code"] == "data_map_public_parent_not_reported"


def test_message_and_execution_job_ids_use_the_job_response_field_and_hex_contract(tmp_path):
    class JobAPI(PhaseAPI):
        def request(self, method, path, body=None, **kwargs):
            if path == "/api/jobs/234567abcdef":
                return {"id": "234567abcdef", "status": "done"}
            payload = super().request(method, path, body, **kwargs)
            if path.endswith("/message"):
                payload.update(job_id="012345abcdef", id="wrong/field", session_id=path.split("/")[-2])
            if path == "/api/jobs/agent-run-review":
                payload["job_id"] = "234567abcdef"
            return payload

    api = JobAPI()
    options = replace(walk_options(tmp_path, approve_test_runs=True), only=("literature", "workflows"))
    report = WalkRunner(options, api=api).run()
    assert all(row["status"] == "passed" for row in report["items"]), report
    state = json.loads((options.out / "walk_state.json").read_text())["items"]["simple-study"]
    assert state["message_job_id"] == "012345abcdef"
    assert state["execution_job_id"] == "234567abcdef"
    resumed = WalkRunner(replace(options, resume=True), api=api).run()
    assert resumed["status_counts"] == report["status_counts"]


def test_candidate_plan_reads_the_generated_job_id_from_the_submission_response(tmp_path):
    class PlanAPI(PhaseAPI):
        submitted = False

        def request(self, method, path, body=None, **kwargs):
            if path.startswith("/api/study-contexts/"):
                return {"context": {"id": path.rsplit("/", 1)[-1], "question": "Describe synthetic measurements?",
                                    "data_source": {"database": "sic", "path": "/synthetic/server-only"}}}
            if path == "/api/jobs/agent-run":
                assert body["candidate_plan_only"] is True
                self.submitted = True
                return {"job_id": "012345abcdef", "id": "wrong/field"}
            payload = super().request(method, path, body, **kwargs)
            if path.endswith("/workflow") and not self.submitted:
                payload["workflow"].update(next_action_code="provider_ready_to_generate_plan", plan_review_summary=None)
                payload["host_decisions"] = {"plan_transition": {"family": "plan_transition"}}
            return payload

    api = PlanAPI()
    options = walk_options(tmp_path)
    report = WalkRunner(options, api=api).run()
    assert report["items"][-1]["reason_code"] == "approval_flag_required", report
    assert api.submitted
    assert json.loads((options.out / "walk_state.json").read_text())["items"]["simple-study"]["plan_job_id"] == "012345abcdef"


@pytest.mark.parametrize("value", [None, "012345abcdef", "bad/id", "bad%2fid", "bad..id", "bad id", "bad?id"])
def test_project_session_and_run_identity_checks_stay_strict(value):
    with pytest.raises(WalkStop, match="server_identity_invalid"):
        _identity(value)


@pytest.mark.parametrize("field", ["project_id", "session_id", "study_context_id", "run_id"])
def test_hex_job_exception_cannot_relax_other_persisted_identities(tmp_path, field):
    store = ReceiptStore(tmp_path / "report", "scope", resume=False)
    store.state["items"]["synthetic"] = {field: "012345abcdef"}
    with pytest.raises(ValueError, match="resume_state_invalid"):
        store.flush()


@pytest.mark.parametrize("value", ["0bad", "012345abcde/", "012345abcde%", "012345abcde?", "../012345abcdef"])
def test_unsafe_job_ids_are_refused_before_persistence_or_url_use(tmp_path, value):
    class UnsafeJobAPI(PhaseAPI):
        def request(self, method, path, body=None, **kwargs):
            payload = super().request(method, path, body, **kwargs)
            if path.endswith("/message"):
                payload["job_id"] = value
            return payload

    options = walk_options(tmp_path)
    report = WalkRunner(options, api=UnsafeJobAPI()).run()
    assert report["items"][-1]["reason_code"] == "server_identity_invalid"
    assert value not in (options.out / "walk_state.json").read_text()


@pytest.mark.parametrize("busy_field", ["streaming", "active_message_job_id", "last_turn_status"])
def test_every_message_waits_for_the_owned_session_to_be_idle(tmp_path, monkeypatch, busy_field):
    class BusyAPI(PhaseAPI):
        polls = 0

        def request(self, method, path, body=None, **kwargs):
            if path.endswith("/message"):
                assert self.polls >= 3, "message was posted before the session became idle"
            payload = super().request(method, path, body, **kwargs)
            if method == "GET" and path.startswith("/api/copilot/pi/sessions/"):
                self.polls += 1
                payload["session"][busy_field] = (
                    {"streaming": True, "active_message_job_id": "prior-job", "last_turn_status": "running"}[busy_field]
                    if self.polls < 3 else {"streaming": False, "active_message_job_id": None, "last_turn_status": "done"}[busy_field]
                )
            return payload

    monkeypatch.setattr("tools.walk.runner.time.sleep", lambda _: None)
    api = BusyAPI()
    report = WalkRunner(walk_options(tmp_path), api=api).run()
    assert report["items"][-1]["reason_code"] == "approval_flag_required", report
    assert sum(path.endswith("/message") for _, path, _, _ in api.calls) == 1


def test_a_racing_busy_response_is_polled_then_retried(tmp_path, monkeypatch):
    class RaceAPI(PhaseAPI):
        attempts = 0

        def request(self, method, path, body=None, **kwargs):
            if path.endswith("/message"):
                self.attempts += 1
                if self.attempts == 1:
                    raise HTTPFailure(409, "pi_session_busy")
            return super().request(method, path, body, **kwargs)

    sleeps = []
    monkeypatch.setattr("tools.walk.runner.time.sleep", sleeps.append)
    api = RaceAPI()
    report = WalkRunner(walk_options(tmp_path), api=api).run()
    assert report["items"][-1]["reason_code"] == "approval_flag_required", report
    assert api.attempts == 2 and sleeps


def test_busy_before_send_is_bounded_by_the_item_deadline(tmp_path):
    class BusyAPI(PhaseAPI):
        def request(self, method, path, body=None, **kwargs):
            assert not path.endswith("/message"), "a busy session must never receive a message"
            payload = super().request(method, path, body, **kwargs)
            if method == "GET" and path.startswith("/api/copilot/pi/sessions/"):
                payload["session"]["last_turn_status"] = "running"
            return payload

    options = replace(walk_options(tmp_path), timeouts={"workflows": 0.02}, poll_interval=0.001)
    report = WalkRunner(options, api=BusyAPI()).run()
    assert report["items"][-1]["status"] == "timed-out", report
    assert report["items"][-1]["elapsed_seconds"] < 1


def test_workflow_waits_for_the_literature_turn_before_reusing_the_session(tmp_path, monkeypatch):
    class HandoffAPI(PhaseAPI):
        literature_read = False
        handoff_polls = 0

        def request(self, method, path, body=None, **kwargs):
            if self.literature_read and (path.endswith("/workflow") or path == "/api/jobs/agent-run-review"):
                assert self.handoff_polls >= 4, "workflow advanced before the previous turn became idle"
            payload = super().request(method, path, body, **kwargs)
            if path.endswith("/artifacts/literature_evidence.json"):
                self.literature_read = True
            if method == "GET" and path.startswith("/api/copilot/pi/sessions/") and self.literature_read:
                self.handoff_polls += 1
                payload["session"]["last_turn_status"] = "running" if self.handoff_polls < 4 else "done"
            return payload

    monkeypatch.setattr("tools.walk.runner.time.sleep", lambda _: None)
    api = HandoffAPI()
    options = replace(walk_options(tmp_path, approve_test_runs=True), only=("literature", "workflows"))
    report = WalkRunner(options, api=api).run()
    assert all(row["status"] == "passed" for row in report["items"]), report
    assert len(api.sessions) == 1
    assert sum(path.endswith("/message") for _, path, _, _ in api.calls) == 1


@pytest.mark.parametrize("binding", [
    {"primary_runner": "synthetic_runner", "capability_id": "synthetic_capability"},
    {"kernel_modules": ["easyicu.synthetic_module"]},
    {"kernel_modules": [], "primary_runner": "synthetic_runner", "capability_id": "synthetic_capability"},
])
def test_catalogue_reads_owner_links_and_component_runner_bindings(tmp_path, binding):
    class OwnersAPI(PhaseAPI):
        def request(self, method, path, body=None, **kwargs):
            payload = super().request(method, path, body, **kwargs)
            if path == "/api/capabilities":
                catalogue = payload["capabilities"]["method_skills"]
                catalogue["items"][0].update(execution_mode="deterministic_host", action_ids=["synthetic.method"])
                catalogue["items"].append({
                    "id": "synthetic-definition", "layer": "data_definition", "execution_mode": "owner_projection",
                    "use_action": {"kind": "open_owner", "owner": "synthetic_definitions", "entry": "synthetic_entry"},
                })
                catalogue["components"] = [{
                    "id": "synthetic.method", "execution_mode": "deterministic_host",
                    **binding,
                }]
            return payload

    options = replace(walk_options(tmp_path), only=("catalogue",))
    report = WalkRunner(options, api=OwnersAPI()).run()
    inventory = report["items"][-1]
    assert inventory["status"] == "passed", inventory
    assert inventory["metrics"]["unreported_owner_count"] == 0
    rows = {row["id"]: row for row in inventory["metrics"]["catalogue_items"]}
    assert rows["synthetic-definition"]["owner_status"] == "declared"
    assert rows["synthetic-definition"]["implementation"] == "owner_projection"
    assert rows["synthetic-definition"]["kernel_binding"] is False
    for item in ("synthetic.method", "simple-study"):
        assert rows[item]["kernel_binding"] is True
        assert rows[item]["implementation"] == "deterministic_host"


@pytest.mark.parametrize("mode,implementation", [
    ("deterministic_host", "deterministic"), ("agent_coded_with_host_gates", "llm_coded"),
])
def test_registered_method_ownership_does_not_require_a_deterministic_kernel(tmp_path, mode, implementation):
    class MethodAPI(PhaseAPI):
        def request(self, method, path, body=None, **kwargs):
            payload = super().request(method, path, body, **kwargs)
            if path == "/api/capabilities":
                payload["capabilities"]["method_skills"]["components"] = [{
                    "id": "synthetic.method", "method_family": "synthetic", "method_key": "method",
                    "execution_mode": mode, "implementation": implementation, "kernel_modules": [],
                }]
            return payload

    report = WalkRunner(replace(walk_options(tmp_path), only=("catalogue",)), api=MethodAPI()).run()
    inventory = report["items"][-1]
    assert inventory["status"] == "passed", inventory
    method = inventory["metrics"]["catalogue_items"][-1]
    assert method["owner_status"] == "declared"
    assert method["kernel_binding"] is False
    assert method["implementation"] == implementation


def test_owner_api_entry_is_a_declared_owner_without_a_kernel(tmp_path):
    class OwnerAPI(PhaseAPI):
        def request(self, method, path, body=None, **kwargs):
            payload = super().request(method, path, body, **kwargs)
            if path == "/api/capabilities":
                payload["capabilities"]["method_skills"]["items"].append({
                    "id": "synthetic-owner", "layer": "data_definition", "execution_mode": "owner_projection",
                    "use_action": {"kind": "open_owner", "owner": "synthetic_owner", "entry": "/api/jobs/synthetic"},
                })
            return payload

    report = WalkRunner(replace(walk_options(tmp_path), only=("catalogue",)), api=OwnerAPI()).run()
    inventory = report["items"][-1]
    assert inventory["status"] == "passed", inventory
    owner = inventory["metrics"]["catalogue_items"][-1]
    assert owner["owner_status"] == "declared"
    assert owner["kernel_binding"] is False
    assert owner["implementation"] == "owner_projection"


@pytest.mark.parametrize("registration,mode", [
    ({}, "deterministic_host"),
    ({"method_family": "synthetic"}, "deterministic_host"),
    ({"method_family": "other", "method_key": "unowned"}, "deterministic_host"),
    ({"method_family": "synthetic", "method_key": "unowned"}, "unknown"),
])
def test_an_execution_mode_alone_cannot_invent_a_missing_owner(tmp_path, registration, mode):
    class UnownedAPI(PhaseAPI):
        def request(self, method, path, body=None, **kwargs):
            payload = super().request(method, path, body, **kwargs)
            if path == "/api/capabilities":
                payload["capabilities"]["method_skills"]["components"] = [{
                    "id": "synthetic.unowned", "execution_mode": mode, "kernel_modules": [], **registration,
                }]
            return payload

    report = WalkRunner(replace(walk_options(tmp_path), only=("catalogue",)), api=UnownedAPI()).run()
    inventory = report["items"][-1]
    assert inventory["status"] == "failed", inventory
    assert inventory["reason_code"] == "catalogue_owner_binding_not_reported"
    assert inventory["metrics"]["unreported_owner_count"] == 1


@pytest.mark.parametrize("fault", ["transient", "persistent", "product", "cancelled"])
def test_failed_provider_turn_is_retried_once_and_remains_a_refusal(tmp_path, monkeypatch, fault):
    class ProviderAPI(PhaseAPI):
        attempts = 0

        def request(self, method, path, body=None, **kwargs):
            if path.endswith("/message"):
                self.attempts += 1
                assert self.attempts <= 2, "provider retries must be bounded"
            payload = super().request(method, path, body, **kwargs)
            if method == "GET" and path.startswith("/api/copilot/pi/sessions/") and self.attempts:
                failed = fault != "transient" or self.attempts == 1
                payload["session"].update(
                    last_turn_status=("cancelled" if fault == "cancelled" else "failed") if failed else "done",
                    last_turn_events=[{"type": "message_end", "stop_reason": "error" if failed else "complete",
                                       "error_code": "execution_step_raised" if fault == "product" else "pi_model_provider_unavailable",
                                       "text": "NEVER-SAVE-THIS"}],
                )
            return payload

    sleeps = []
    monkeypatch.setattr("tools.walk.runner.time.sleep", sleeps.append)
    api = ProviderAPI()
    options = replace(walk_options(tmp_path), only=("literature", "workflows"))
    report = WalkRunner(options, api=api).run()
    literature = next(row for row in report["items"] if row["phase"] == "literature")
    if fault == "transient":
        assert literature["status"] == "passed", literature
    else:
        assert literature["status"] == ("refused-with-reason" if fault == "persistent" else "failed")
        assert literature["reason_code"] == ("provider_unavailable" if fault == "persistent" else "conversation_turn_failed")
    assert api.attempts == (2 if fault in {"transient", "persistent"} else 1)
    assert bool(sleeps) == (fault in {"transient", "persistent"})
    if sleeps:
        assert sleeps == [1]
    for path in options.out.rglob("*.json"):
        assert "NEVER-SAVE-THIS" not in path.read_text()
    if fault == "persistent":
        WalkRunner(replace(options, resume=True), api=api).run()
        assert api.attempts == 2, "resume must preserve the exhausted provider retry budget"


@pytest.mark.parametrize("status", [500, 503])
@pytest.mark.parametrize("provider", [True, False])
def test_only_provider_http_failures_receive_provider_backoff(tmp_path, monkeypatch, status, provider):
    class HTTPProviderAPI(PhaseAPI):
        attempts = 0

        def request(self, method, path, body=None, **kwargs):
            if path.endswith("/message"):
                self.attempts += 1
                assert self.attempts <= 2, "provider retries must be bounded"
                raise HTTPFailure(status, "pi_model_provider_unavailable" if provider else "synthetic_product_failure")
            return super().request(method, path, body, **kwargs)

    sleeps = []
    monkeypatch.setattr("tools.walk.runner.time.sleep", sleeps.append)
    api = HTTPProviderAPI()
    options = replace(walk_options(tmp_path), only=("literature", "workflows") if provider else ("workflows",))
    report = WalkRunner(options, api=api).run()
    row = next(row for row in report["items"] if row["phase"] == ("literature" if provider else "workflows"))
    assert row["reason_code"] == ("provider_unavailable" if provider else "synthetic_product_failure")
    assert row["status"] == ("refused-with-reason" if provider or status == 503 else "failed")
    assert api.attempts == (2 if provider else 1)
    assert sleeps == ([1] if provider else [])


@pytest.mark.parametrize("history_only", [False, True])
def test_failed_plan_receipts_preserve_only_stable_run_failure_coordinates(tmp_path, history_only):
    class FailedPlanAPI(PhaseAPI):
        def request(self, method, path, body=None, **kwargs):
            payload = super().request(method, path, body, **kwargs)
            if path.endswith("/workflow"):
                failure = {"run_id": "run-failed", "gate_status": "blocked", "gate_reason_code": "pipeline_failed",
                           "gate_detail_code": "execution_step_raised", "gate_detail_cause_code": "synthetic_executor_refused",
                           "gate_detail_step_id": "synthetic_analysis", "gate_detail_runner_kind": "synthetic_phase",
                           "gate_detail_step_title": "NEVER-SAVE-THIS",
                           "error": "NEVER-SAVE-THIS", "rows": [{"patient_id": "NEVER-SAVE-THIS"}]}
                payload["workflow"]["next_action_code"] = "failed_pipeline_requires_fresh_plan"
                payload["host_decisions"] = {}
                if history_only:
                    payload["runs"] = [failure, {"run_id": "run-old", "gate_reason_code": "old_failure", "authoritative": True}]
                else:
                    payload["latest_run"] = {"present": True, **failure}
            return payload

    options = walk_options(tmp_path, approve_test_runs=True)
    api = FailedPlanAPI()
    report = WalkRunner(options, api=api).run()
    row = report["items"][-1]
    assert row["status"] == "refused-with-reason", row
    assert row["reason_code"] == "failed_pipeline_requires_fresh_plan"
    for field, value in {"gate_status": "blocked", "gate_reason_code": "pipeline_failed",
                         "gate_detail_code": "execution_step_raised", "gate_detail_cause_code": "synthetic_executor_refused",
                         "gate_detail_step_id": "synthetic_analysis", "gate_detail_runner_kind": "synthetic_phase"}.items():
        assert row["metrics"][field] == value
    assert not api.approved
    for path in options.out.rglob("*.json"):
        assert "NEVER-SAVE-THIS" not in path.read_text()
        assert "old_failure" not in path.read_text()


def test_failure_coordinate_projection_rejects_free_text_and_identifiers(tmp_path):
    store = ReceiptStore(tmp_path / "report", "scope", resume=False)
    result = ItemResult("workflows", "synthetic", Status.REFUSED, "pipeline_failed", 0, 0, {
        "gate_detail_code": "patient_id=NEVER-SAVE-THIS", "gate_detail_cause_code": "failure with private text",
        "gate_detail_step_id": "bad/coordinate", "gate_detail_step_title": "NEVER-SAVE-THIS",
    })
    store.save(result, policy="plan-only")
    restored = store.finished("workflows", "synthetic", policy="plan-only")
    assert restored.metrics == {"gate_detail_code": "not_reported", "gate_detail_cause_code": "not_reported",
                                "gate_detail_step_id": "not_reported"}
    assert "NEVER-SAVE-THIS" not in next(store.receipts.iterdir()).read_text()


def test_execution_failure_receipt_keeps_the_stable_step_and_cause(tmp_path):
    class FailedExecutionAPI(PhaseAPI):
        def request(self, method, path, body=None, **kwargs):
            payload = super().request(method, path, body, **kwargs)
            if path.endswith("/workflow") and self.approved:
                payload["latest_run"].update(
                    gate_status="blocked", gate_reason_code="pipeline_failed", gate_detail_code="execution_step_raised",
                    gate_detail_cause_code="synthetic_executor_refused", gate_detail_step_id="synthetic_analysis",
                    gate_detail_step_title="NEVER-SAVE-THIS", execution_complete=False,
                )
            return payload

    options = walk_options(tmp_path, approve_test_runs=True)
    report = WalkRunner(options, api=FailedExecutionAPI()).run()
    row = report["items"][-1]
    assert row["reason_code"] == "pipeline_failed", row
    assert row["metrics"]["gate_detail_step_id"] == "synthetic_analysis"
    assert row["metrics"]["gate_detail_cause_code"] == "synthetic_executor_refused"
    assert "NEVER-SAVE-THIS" not in (options.out / "walk_report.json").read_text()
