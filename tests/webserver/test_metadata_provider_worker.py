"""Guarded Provider transport, with loopback synthetic responses only."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import pytest
from types import SimpleNamespace

from easyicu.webserver.metadata_provider_worker import guarded_call
from easyicu.runtime.host_execution_lease import read_host_execution_receipt


@pytest.mark.parametrize(
    ("checkpoint", "scope", "acquisition", "expected"),
    [
        (False, None, None, True),
        (True, None, None, False),
        (True, SimpleNamespace(budget_mode="planner_canary", prepared_grouping=None),
         SimpleNamespace(kind="metadata_only_planning_catalog"), True),
        (False, SimpleNamespace(budget_mode="planner_canary", prepared_grouping=None), None, True),
        (True, SimpleNamespace(budget_mode="planner_canary", prepared_grouping=None),
         SimpleNamespace(kind="materialized_cohort"), False),
        (True, SimpleNamespace(budget_mode="planner_canary", prepared_grouping=object()),
         SimpleNamespace(kind="metadata_only_planning_catalog"), False),
        (True, SimpleNamespace(budget_mode="full_reviewed", prepared_grouping=None),
         SimpleNamespace(kind="metadata_only_planning_catalog"), False),
    ],
)
def test_guarded_planner_scope_distinguishes_metadata_from_patient_inputs(checkpoint, scope, acquisition, expected):
    from easyicu.webserver.metadata_provider_client import guarded_metadata_resume_scope

    assert guarded_metadata_resume_scope(has_checkpoint=checkpoint, scope=scope, acquisition=acquisition) is expected


def test_guarded_one_request_uses_loopback_only_and_cleans_worker(tmp_path):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"id": "fake", "object": "chat.completion", "created": 1,
                "model": "synthetic", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = {
            "provider": {"provider": "openai", "external": True},
            "environment": {"OPENAI_API_KEY": "synthetic-secret",
                "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
                "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"},
            "provider_identity": {}, "timeout": 10,
            "messages": [{"role": "user", "content": "synthetic metadata"}],
            "max_tokens": 8, "temperature": 0.1,
        }
        result = guarded_call(tmp_path / "guard", request, timeout=10)
        assert result.get("cause_type") is None, result.get("cause_type")
        assert result["ok"] is True and result["text"] == "ok", result
        assert len(calls) == 1
        receipt = read_host_execution_receipt(tmp_path / "guard")
        assert receipt["state"] == "quiesced"
        assert receipt["reason"] == "worker_completed"
        assert "synthetic-secret" not in json.dumps(receipt)
        assert "synthetic metadata" not in json.dumps(receipt)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_worker_timeout_preserves_unknown_usage_without_automatic_recovery(tmp_path, monkeypatch):
    import easyicu.research_agent.providers.llm  # noqa: F401
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.webserver import metadata_provider_client as client_owner
    from easyicu.webserver import metadata_provider_concurrency as slots, provider_adapter
    from easyicu.webserver.agent_pipeline_runs import _pipeline_failure_category, _record_pipeline_failure
    from easyicu.webserver.metadata_provider_client import MetadataProviderClient
    from easyicu.webserver.metadata_provider_worker import MetadataProviderWorkerTimeout
    from easyicu.webserver.provider_job_recovery import transient_provider_cause

    calls = []
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(1)
            self.rfile.read(int(self.headers["Content-Length"]))
            release.wait(30)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setattr(slots.state_paths, "state_root", lambda: tmp_path / "state")
        real_guarded_call = client_owner.guarded_call

        def slow_transport(lease_dir, request, **kwargs):
            # The guardian deadline fires while the synthetic SDK call is
            # still pending, so this tests the actual process timeout path.
            return real_guarded_call(lease_dir, {**request, "timeout": 30}, **kwargs)

        monkeypatch.setattr(client_owner, "guarded_call", slow_transport)
        environment = {"OPENAI_API_KEY": "synthetic-secret", "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                       "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
                       "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"}
        provider = {"provider": "openai", "external": True}
        client, public = provider_adapter.build_research_agent_provider_client(provider, environ=environment, request_timeout=10)
        book = ProviderHardStopLedger(path=tmp_path / "budget.json", task_ids=("task",),
            limits=provider_adapter.web_research_agent_hard_stop_limits("planner_canary"),
            batch_id="task", declaration_sha256="a" * 64)
        task = book.start_task("task")
        supervised = MetadataProviderClient(inner=client, task=task, provider=provider, environment=environment, public=public, timeout=8)
        wrapped = HardStopClient(supervised, role="planner", task=task)
        with pytest.raises(MetadataProviderWorkerTimeout) as caught:
            wrapped.complete([LLMMessage(role="user", content="synthetic metadata")])
        assert calls == [1]
        rows = book.snapshot()["tasks"][0]["calls"]
        assert len(rows) == 1 and rows[0]["state"] == "failed_usage_unknown"
        assert slots.get(rows[0]["attempt_id"]) is not None
        assert _pipeline_failure_category(caught.value) == "timeout"
        assert transient_provider_cause(caught.value) is None
        diagnostic = _record_pipeline_failure(wrapper_dir=tmp_path, study={"id": "synthetic"}, provider={},
            exc=caught.value, code="research_pipeline_execution_failed", execution_retry_id=None)
        recorded = json.loads((tmp_path / diagnostic).read_text())
        assert recorded["failure_type"] == "timeout"
        assert recorded["exception_types"] == ["MetadataProviderWorkerTimeout"]
        receipt = read_host_execution_receipt(tmp_path / ".metadata-provider-execution" / rows[0]["attempt_id"])
        assert receipt["state"] == "quiesced" and receipt["reason"] == "timeout"
    finally:
        release.set()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.mark.parametrize("output", [b"", b"{", b"[]", b"\xff"])
def test_bad_worker_ipc_is_not_reported_as_model_output_parse_error(output):
    from easyicu.webserver.metadata_provider_worker import _decode_worker_response, MetadataProviderWorkerResponseError
    from easyicu.research_agent.providers.structured_retry import safe_provider_error_category
    from easyicu.webserver.provider_job_recovery import transient_provider_cause

    with pytest.raises(MetadataProviderWorkerResponseError) as caught:
        _decode_worker_response(output)
    assert safe_provider_error_category(caught.value) == "error"
    assert transient_provider_cause(caught.value) is None


@pytest.mark.parametrize("first_status", [200, 500])
def test_budgeted_metadata_client_uses_durable_slots(tmp_path, monkeypatch, first_status):
    import easyicu.research_agent.providers.llm  # noqa: F401
    from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept
    from easyicu.research_agent.acquisition.foundation import DataFoundationAgent
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.webserver import metadata_provider_concurrency as slots, provider_adapter
    from easyicu.webserver import metadata_provider_client as client_owner
    from easyicu.webserver.metadata_provider_client import MetadataProviderClient

    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(1)
            self.rfile.read(int(self.headers["Content-Length"]))
            if len(calls) == 1 and first_status == 500:
                body = b'{"error":{"message":"synthetic unavailable"}}'
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = json.dumps({"id": "fake", "object": "chat.completion", "created": 1,
                "model": "synthetic", "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"selected_concepts":["age"],"inclusion_exclusion":[],"rationale":"synthetic"}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setattr(slots.state_paths, "state_root", lambda: tmp_path / "state")
        monkeypatch.setattr(client_owner.time, "sleep", lambda _seconds: None)
        environment = {"OPENAI_API_KEY": "synthetic-secret", "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                       "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
                       "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"}
        provider = {"provider": "openai", "external": True}
        client, public = provider_adapter.build_research_agent_provider_client(provider, environ=environment, request_timeout=10)
        book = ProviderHardStopLedger(path=tmp_path / "budget.json", task_ids=("task",),
            limits=provider_adapter.web_research_agent_hard_stop_limits("planner_canary"),
            batch_id="task", declaration_sha256="a" * 64)
        task = book.start_task("task")
        supervised = MetadataProviderClient(inner=client, task=task, provider=provider, environment=environment, public=public, timeout=10)
        wrapped = HardStopClient(supervised, role="acquisition", task=task)
        selection = DataFoundationAgent(wrapped).select_concepts(
            question="synthetic metadata", catalog=AvailableCatalog(source="fake", concepts=[CatalogConcept("age")]),
        )
        assert selection.selected_concepts == ["age"]
        rows = book.snapshot()["tasks"][0]["calls"]
        assert len(calls) == len(rows) == (1 if first_status == 200 else 2)
        assert rows[-1]["state"] == "completed"
        if first_status == 500:
            assert rows[0]["state"] == "failed_usage_unknown"
        assert all(slots.get(row["attempt_id"]) is None for row in rows)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_concurrent_metadata_slots_are_bounded_and_unknown_remains(tmp_path, monkeypatch):
    from easyicu.webserver import metadata_provider_concurrency as slots

    monkeypatch.setattr(slots.state_paths, "state_root", lambda: tmp_path / "state")
    monkeypatch.setenv("EASYICU_METADATA_PROVIDER_MAX_INFLIGHT", "2")
    barrier = threading.Barrier(4)
    def attempt(index):
        barrier.wait()
        try:
            slots.reserve(attempt_id=f"{index:032x}", account="account", task_id="task",
                          ledger_path=tmp_path / f"{index}.json", guard_path=tmp_path / f"guard-{index}",
                          worker_sha256="a" * 64)
            return True
        except RuntimeError:
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(attempt, range(4)))
    assert sum(results) == 2
    assert slots.has_capacity("account") is False
    accepted = [f"{index:032x}" for index, ok in enumerate(results) if ok]
    slots.release(accepted[0])
    assert slots.get(accepted[1]) is not None
    assert slots.has_capacity("account") is True


def test_planner_strict_schema_is_preserved_and_charged_to_planner(tmp_path, monkeypatch):
    import easyicu.research_agent.providers.llm  # noqa: F401
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.llm import LLMRouter
    from easyicu.research_agent.providers.protocol import LLMMessage, StructuredOutputRequest
    from easyicu.webserver import metadata_provider_concurrency as slots, provider_adapter
    from easyicu.webserver.metadata_provider_client import MetadataProviderClient

    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"id": "fake", "object": "chat.completion", "created": 1,
                "model": "synthetic", "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"value":"ok"}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setattr(slots.state_paths, "state_root", lambda: tmp_path / "state")
        env = {"OPENAI_API_KEY": "synthetic-secret", "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
               "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
               "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"}
        provider = {"provider": "openai", "external": True}
        client, public = provider_adapter.build_research_agent_provider_client(provider, environ=env, request_timeout=10, supports_strict_json_schema=True)
        book = ProviderHardStopLedger(path=tmp_path / "budget.json", task_ids=("task",),
            limits=provider_adapter.web_research_agent_hard_stop_limits("planner_canary"),
            batch_id="task", declaration_sha256="a" * 64)
        task = book.start_task("task")
        supervised = MetadataProviderClient(inner=client, task=task, provider=provider, environment=env, public=public, timeout=10)
        router = LLMRouter(default=client, planner=supervised)
        assert router.for_role("planner") is supervised
        assert router.for_role("coder") is client
        schema = StructuredOutputRequest.from_schema(name="plan", schema={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False})
        result = HardStopClient(supervised, role="planner", task=task).complete(
            [LLMMessage(role="user", content="synthetic metadata")], max_tokens=32, structured_output=schema)
        assert json.loads(result) == {"value": "ok"}
        assert seen[0]["response_format"] == schema.to_openai_response_format()
        row = book.snapshot()["tasks"][0]["calls"][0]
        assert row["role"] == "planner" and row["state"] == "completed"
        assert row["data_classes"] == ["text", "schema"]
        assert supervised.last_finish_reason == "stop" and supervised.last_transport_attempts == 1
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_guarded_recovery_probe_is_one_attempt_on_service_error(tmp_path, monkeypatch):
    import easyicu.research_agent.providers.llm  # noqa: F401
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
    from easyicu.research_agent.providers.hard_stop import HardStopClient, run_recovery_probe
    from easyicu.webserver import metadata_provider_client, metadata_provider_concurrency as slots, provider_adapter

    monkeypatch.setattr(slots.state_paths, "state_root", lambda: tmp_path / "state")
    sent = []
    def fail_once(*args, **kwargs):
        sent.append(1)
        return {"ok": False, "type": "InternalServerError", "status_code": 500}
    monkeypatch.setattr(metadata_provider_client, "guarded_call", fail_once)
    env = {"OPENAI_API_KEY": "synthetic-secret", "OPENAI_BASE_URL": "http://127.0.0.1:1/v1",
           "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
           "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"}
    provider = {"provider": "openai", "external": True}
    inner, public = provider_adapter.build_research_agent_provider_client(provider, environ=env, request_timeout=10)
    book = ProviderHardStopLedger(path=tmp_path / "budget.json", task_ids=("task",),
        limits=provider_adapter.web_research_agent_hard_stop_limits("planner_canary"),
        batch_id="task", declaration_sha256="a" * 64)
    task = book.start_task("task")
    guarded = metadata_provider_client.MetadataProviderClient(
        inner=inner, task=task, provider=provider, environment=env, public=public, timeout=10,
    )
    with pytest.raises(metadata_provider_client.MetadataProviderError) as error:
        run_recovery_probe(HardStopClient(guarded, role="provider_recovery_probe", task=task))
    from easyicu.webserver.provider_job_recovery import transient_provider_cause
    assert transient_provider_cause(error.value) == "provider_http_error"
    calls = book.snapshot()["tasks"][0]["calls"]
    assert len(sent) == len(calls) == 1
    assert calls[0]["role"] == "provider_recovery_probe"
    assert calls[0]["state"] == "failed_usage_unknown"
    assert slots.get(calls[0]["attempt_id"]) is None


def test_capacity_stop_does_not_send_or_release_existing_request(tmp_path, monkeypatch):
    import easyicu.research_agent.providers.llm  # noqa: F401
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.webserver import agent_pipeline_runs, metadata_provider_client, metadata_provider_concurrency as slots, provider_adapter
    from easyicu.webserver.stop_next_actions import next_actions_for_stop

    monkeypatch.setattr(slots.state_paths, "state_root", lambda: tmp_path / "state")
    monkeypatch.setenv("EASYICU_METADATA_PROVIDER_MAX_INFLIGHT", "1")
    sent = []
    monkeypatch.setattr(metadata_provider_client, "guarded_call", lambda *a, **k: sent.append(1))
    env = {"OPENAI_API_KEY": "synthetic-secret", "OPENAI_BASE_URL": "http://127.0.0.1:1/v1",
           "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
           "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"}
    provider = {"provider": "openai", "external": True}
    inner, public = provider_adapter.build_research_agent_provider_client(provider, environ=env, request_timeout=10)
    book = ProviderHardStopLedger(path=tmp_path / "budget.json", task_ids=("task",),
        limits=provider_adapter.web_research_agent_hard_stop_limits("planner_canary"),
        batch_id="task", declaration_sha256="a" * 64)
    task = book.start_task("task")
    client = metadata_provider_client.MetadataProviderClient(
        inner=inner, task=task, provider=provider, environment=env, public=public, timeout=10,
    )
    slots.reserve(attempt_id="old", account=slots.account_identity(public), task_id="old-task",
                  ledger_path=tmp_path / "old.json", guard_path=tmp_path / "old-guard", worker_sha256="a" * 64)
    with pytest.raises(slots.MetadataProviderCapacityExhausted) as caught:
        HardStopClient(client, role="planner", task=task).complete([LLMMessage(role="user", content="synthetic")])
    assert sent == []
    assert slots.get("old") is not None
    code = agent_pipeline_runs._pipeline_failure_code(caught.value)
    assert code == "research_pipeline_metadata_provider_capacity_unavailable"
    assert next_actions_for_stop(code) == []
    assert agent_pipeline_runs._pipeline_failure_code(RuntimeError("metadata_provider_concurrency_exhausted")) == "research_pipeline_execution_failed"
