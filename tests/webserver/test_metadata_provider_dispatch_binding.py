"""The metadata worker must use the already-authorized provider identity."""

from types import SimpleNamespace

import pytest


def _bound(tmp_path):
    import easyicu.research_agent.providers.llm  # noqa: F401
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
    from easyicu.webserver import metadata_provider_client, provider_adapter

    env = {"OPENAI_API_KEY": "synthetic-key", "OPENAI_BASE_URL": "http://127.0.0.1:1/v1",
           "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
           "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"}
    provider = {"provider": "openai", "external": True}
    inner, public = provider_adapter.build_research_agent_provider_client(provider, environ=env, request_timeout=10)
    ledger = ProviderHardStopLedger(path=tmp_path / "budget.json", task_ids=("task",),
        limits=provider_adapter.web_research_agent_hard_stop_limits("planner_canary"),
        batch_id="task", declaration_sha256="a" * 64)
    task = ledger.start_task("task")
    wrapper = metadata_provider_client.MetadataProviderClient(
        inner=inner, task=task, provider=provider, environment=env, public=public, timeout=10)
    return wrapper, ledger, provider, env, public


@pytest.mark.parametrize("changed", ["provider", "credential", "endpoint", "handling", "public",
                                      "timeout", "account", "inner", "task", "ledger", "ledger_path",
                                      "outer_task", "budget_model", "budget_aware", "name", "file_fallback"])
def test_changed_dispatch_is_refused_before_budget_capacity_or_worker(tmp_path, monkeypatch, changed):
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.mocks import MockLLMClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.webserver import metadata_provider_client, metadata_provider_concurrency

    wrapper, ledger, _, _, _ = _bound(tmp_path)
    hard_stop = HardStopClient(wrapper, role="planner", task=wrapper._task)
    ledger_path = ledger.path
    original_ledger = ledger_path.read_bytes()
    if changed == "provider":
        wrapper._provider["provider"] = "anthropic"
    elif changed == "credential":
        wrapper._environment["OPENAI_API_KEY"] = "different"
    elif changed == "endpoint":
        wrapper._environment["OPENAI_BASE_URL"] = "http://127.0.0.1:2/v1"
    elif changed == "handling":
        wrapper._environment["EASYICU_PROVIDER_DATA_HANDLING"] = "on_device"
    elif changed == "public":
        wrapper._public["model"] = "different"
    elif changed == "timeout":
        wrapper._timeout = 99
    elif changed == "account":
        wrapper._account = "different"
    elif changed == "inner":
        wrapper._inner = MockLLMClient()
    elif changed == "task":
        wrapper._task = SimpleNamespace(task_id="other", ledger=wrapper._task.ledger)
    elif changed == "ledger":
        wrapper._task.ledger = SimpleNamespace(path=ledger.path)
    elif changed == "ledger_path":
        wrapper._task.ledger.path = tmp_path / "other.json"
    elif changed == "outer_task":
        hard_stop._task = SimpleNamespace(task_id="other", ledger=wrapper._task.ledger)
    elif changed == "budget_model":
        wrapper._model = "different"
    elif changed == "budget_aware":
        wrapper.provider_attempt_budget_aware = False
    elif changed == "name":
        wrapper.name = "different"
    elif changed == "file_fallback":
        wrapper._environment["EASYICU_DISABLE_PROVIDER_ENV_FILE"] = "0"
    calls = []
    monkeypatch.setattr(metadata_provider_client, "guarded_call", lambda *a, **k: calls.append("worker"))
    monkeypatch.setattr(metadata_provider_concurrency, "reserve", lambda *a, **k: calls.append("slot"))
    with pytest.raises((ValueError, RuntimeError)):
        hard_stop.complete([LLMMessage(role="user", content="synthetic")])
    assert calls == []
    assert ledger_path.read_bytes() == original_ledger


def test_wrapper_cannot_be_reregistered_to_drop_its_preflight(tmp_path):
    from easyicu.research_agent.providers.factory import _register_provider_wrapper

    wrapper, _, _, _, _ = _bound(tmp_path)
    with pytest.raises(ValueError):
        _register_provider_wrapper(wrapper, children_getter=lambda: (wrapper._inner,))


def test_multimodal_entry_checks_binding_before_budget(tmp_path):
    from easyicu.research_agent.providers.hard_stop import HardStopClient

    wrapper, ledger, _, _, _ = _bound(tmp_path)
    hard_stop = HardStopClient(wrapper, role="planner", task=wrapper._task)
    before = ledger.path.read_bytes()
    wrapper._model = "different"
    with pytest.raises(ValueError):
        hard_stop.complete_with_images(prompt="synthetic", image_paths=[])
    assert ledger.path.read_bytes() == before


@pytest.mark.parametrize("changed", ["provider", "credential", "public", "mock", "timeout"])
def test_constructor_rejects_provider_not_bound_to_exact_authorized_client(tmp_path, changed):
    from easyicu.research_agent.providers.mocks import MockLLMClient
    from easyicu.webserver.metadata_provider_client import MetadataProviderClient

    wrapper, _, provider, env, public = _bound(tmp_path)
    provider, env, public = dict(provider), dict(env), dict(public)
    if changed == "provider":
        provider["provider"] = "anthropic"
    elif changed == "credential":
        env["OPENAI_API_KEY"] = "different"
    elif changed == "public":
        public["model"] = "different"
    inner = MockLLMClient() if changed == "mock" else wrapper._inner
    with pytest.raises(ValueError, match="metadata_provider_binding_"):
        MetadataProviderClient(inner=inner, task=wrapper._task, provider=provider,
                               environment=env, public=public, timeout=11 if changed == "timeout" else 10)


def test_authorized_binding_dispatches_exact_snapshot_once(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.webserver import metadata_provider_client, metadata_provider_concurrency

    wrapper, ledger, provider, env, _ = _bound(tmp_path)
    sent = []
    monkeypatch.setattr(metadata_provider_concurrency, "reserve", lambda **kw: sent.append("reserved"))
    monkeypatch.setattr(metadata_provider_concurrency, "release", lambda _attempt: None)

    def worker(_guard, request, **_kwargs):
        assert request["provider"] == provider and request["environment"] == env
        sent.append("worker")
        return {"ok": True, "text": "synthetic response", "usage": {"total_tokens": 1}}

    monkeypatch.setattr(metadata_provider_client, "guarded_call", worker)
    result = HardStopClient(wrapper, role="planner", task=wrapper._task).complete(
        [LLMMessage(role="user", content="synthetic")], max_tokens=8)
    assert result == "synthetic response" and sent == ["reserved", "worker"]
    assert len(ledger.snapshot()["tasks"][0]["calls"]) == 1


def test_metadata_wrapper_refuses_legacy_file_fallback(tmp_path):
    from easyicu.webserver import provider_adapter
    from easyicu.webserver.metadata_provider_client import MetadataProviderClient

    wrapper, ledger, provider, env, public = _bound(tmp_path)
    legacy = tmp_path / "synthetic-provider.env"
    legacy.write_text("OPENAI_API_KEY=synthetic-file-key\n", encoding="utf-8")
    legacy.chmod(0o600)
    file_env = dict(env)
    file_env.pop("OPENAI_API_KEY")
    file_env.pop("EASYICU_DISABLE_PROVIDER_ENV_FILE")
    file_env["EASYICU_LLM_ENV_FILE"] = str(legacy)
    inner, file_public = provider_adapter.build_research_agent_provider_client(
        provider, environ=file_env, request_timeout=10)
    original = ledger.path.read_bytes()
    with pytest.raises(ValueError, match="metadata_provider_file_fallback_forbidden"):
        MetadataProviderClient(inner=inner, task=wrapper._task, provider=provider,
                               environment=file_env, public=file_public, timeout=10)
    assert ledger.path.read_bytes() == original


def test_metadata_dispatch_ignores_legacy_file_changes(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.webserver import metadata_provider_client, metadata_provider_concurrency, provider_adapter

    wrapper, _, provider, env, _ = _bound(tmp_path)
    legacy = tmp_path / "synthetic-provider.env"
    legacy.write_text("OPENAI_API_KEY=synthetic-file-key\n", encoding="utf-8")
    legacy.chmod(0o600)
    env["EASYICU_LLM_ENV_FILE"] = str(legacy)
    inner, public = provider_adapter.build_research_agent_provider_client(provider, environ=env, request_timeout=10)
    wrapper = metadata_provider_client.MetadataProviderClient(
        inner=inner, task=wrapper._task, provider=provider, environment=env, public=public, timeout=10)
    legacy.write_text("OPENAI_API_KEY=changed-file-key\n", encoding="utf-8")
    legacy.chmod(0o000)
    seen = []
    monkeypatch.setattr(metadata_provider_concurrency, "reserve", lambda **kw: None)
    monkeypatch.setattr(metadata_provider_concurrency, "release", lambda _attempt: None)

    def fake_worker(_guard, request, **_kwargs):
        seen.append(request["environment"] == env)
        return {"ok": True, "text": "ok", "usage": {"total_tokens": 1}}

    monkeypatch.setattr(metadata_provider_client, "guarded_call", fake_worker)
    assert HardStopClient(wrapper, role="planner", task=wrapper._task).complete(
        [LLMMessage(role="user", content="synthetic")], max_tokens=8) == "ok"
    assert seen == [True]


def test_parallel_calls_bind_own_ledger_attempt_slot_and_guard(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Lock
    from easyicu.research_agent.authority.provider_hard_stop import current_metadata_provider_reservation
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.research_agent.providers.transport_retry import metadata_request_binding
    from easyicu.webserver import metadata_provider_client, metadata_provider_concurrency

    wrapper, ledger, _, _, _ = _bound(tmp_path)
    original_consume = metadata_provider_client.consume_active_provider_hard_stop_attempt
    barrier, lock = Barrier(2), Lock()
    own_attempts, reservations, nonces, worker_guards = {}, {}, {}, {}

    def consume(**kwargs):
        remaining = original_consume(**kwargs)
        attempt, _ = current_metadata_provider_reservation(wrapper._task)
        from threading import current_thread
        with lock:
            own_attempts[current_thread().name] = attempt
        barrier.wait(timeout=10)
        return remaining

    def reserve(**kwargs):
        with lock:
            reservations[kwargs["attempt_id"]] = kwargs

    def bind_guard(attempt, nonce):
        with lock:
            nonces[attempt] = nonce

    def worker(guard, request, **kwargs):
        label = request["messages"][0]["content"]
        attempt = guard.name
        kwargs["on_prepared"]("nonce-" + label)
        with lock:
            worker_guards[label] = attempt
        return {"ok": True, "text": label, "usage": {"total_tokens": 1}}

    monkeypatch.setattr(metadata_provider_client, "consume_active_provider_hard_stop_attempt", consume)
    monkeypatch.setattr(metadata_provider_client, "guarded_call", worker)
    monkeypatch.setattr(metadata_provider_concurrency, "reserve", reserve)
    monkeypatch.setattr(metadata_provider_concurrency, "bind_guard", bind_guard)
    monkeypatch.setattr(metadata_provider_concurrency, "release", lambda _attempt: None)

    def call(label, request_sha):
        from threading import current_thread
        with metadata_request_binding(request_sha):
            result = HardStopClient(wrapper, role="planner", task=wrapper._task).complete(
                [LLMMessage(role="user", content=label)], max_tokens=8)
        return label, current_thread().name, result

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(call, "first", "a" * 64), pool.submit(call, "second", "b" * 64)]
        results = [future.result(timeout=20) for future in futures]
    for label, thread, result in results:
        attempt = own_attempts[thread]
        assert result == label and worker_guards[label] == attempt
        assert reservations[attempt]["guard_path"].name == attempt
        assert reservations[attempt]["request_sha256"] == ("a" if label == "first" else "b") * 64
        assert nonces[attempt] == "nonce-" + label
    calls = ledger.snapshot()["tasks"][0]["calls"]
    assert {row["attempt_id"] for row in calls} == set(own_attempts.values())
    assert len(calls) == 2 and all(row["state"] == "completed" for row in calls)


def test_missing_current_reservation_cannot_allocate_slot_or_send(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.webserver import metadata_provider_client, metadata_provider_concurrency

    wrapper, ledger, _, _, _ = _bound(tmp_path)
    calls = []
    monkeypatch.setattr(metadata_provider_client, "consume_active_provider_hard_stop_attempt", lambda: 10)
    monkeypatch.setattr(metadata_provider_client, "guarded_call", lambda *a, **k: calls.append("worker"))
    monkeypatch.setattr(metadata_provider_concurrency, "reserve", lambda *a, **k: calls.append("slot"))
    with pytest.raises(RuntimeError, match="metadata_provider_reservation_missing"):
        HardStopClient(wrapper, role="planner", task=wrapper._task).complete(
            [LLMMessage(role="user", content="synthetic")], max_tokens=8)
    assert calls == [] and ledger.snapshot()["tasks"][0]["calls"] == []


def test_overlapping_structured_calls_use_own_receipt_for_truncation(tmp_path, monkeypatch):
    import json
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, current_thread
    from easyicu.research_agent.providers import structured_retry
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.webserver import metadata_provider_client, metadata_provider_concurrency

    wrapper, ledger, _, _, _ = _bound(tmp_path)
    first_waiting, second_done = Event(), Event()
    original_metadata = structured_retry._safe_provider_call_metadata
    dispatches = []

    def metadata(llm):
        if current_thread().name == "receipt-first":
            first_waiting.set()
            assert second_done.wait(10)
        return original_metadata(llm)

    def worker(_guard, request, **_kwargs):
        label = request["messages"][0]["content"]
        dispatches.append(label)
        if label == "second" and dispatches.count("second") == 1:
            return {"ok": False, "status_code": 503, "type": "ProviderHTTPError"}
        return {"ok": True, "text": "{" if label == "first" else '{"value":2}',
                "finish_reason": "length" if label == "first" else "stop",
                "usage": {"prompt_tokens": 11 if label == "first" else 22, "completion_tokens": 1}}

    monkeypatch.setattr(metadata_provider_client, "guarded_call", worker)
    monkeypatch.setattr(metadata_provider_concurrency, "reserve", lambda **_kw: None)
    monkeypatch.setattr(metadata_provider_concurrency, "release", lambda _id: None)
    monkeypatch.setattr(metadata_provider_client.TransportRetrySchedule, "next_wait", lambda *_a, **_kw: (0, None))
    monkeypatch.setattr(metadata_provider_client.time, "sleep", lambda _wait: None)
    monkeypatch.setattr(structured_retry, "_safe_provider_call_metadata", metadata)

    def call(label):
        current_thread().name = "receipt-" + label
        if label == "second":
            assert first_waiting.wait(10)
        try:
            result = structured_retry.call_llm_with_structured_retry(
                HardStopClient(wrapper, role="planner", task=wrapper._task),
                [LLMMessage(role="user", content=label)], parser=json.loads,
                role="planner", max_retries=0, max_tokens=8)
            return result
        except structured_retry.StructuredResponseFailure as exc:
            return exc.attempts
        finally:
            if label == "second":
                second_done.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(call, "first")
        second = pool.submit(call, "second")
        attempts, result = first.result(timeout=20), second.result(timeout=20)
    assert result == {"value": 2}
    assert dispatches.count("first") == 1 and dispatches.count("second") == 2
    assert len(attempts) == 1 and attempts[0].finish_reason == "length"
    assert attempts[0].usage_summary == {"prompt_tokens": 11, "completion_tokens": 1}
    assert attempts[0].transport_attempts == 1
    calls = ledger.snapshot()["tasks"][0]["calls"]
    assert len(calls) == 3
    assert sorted(row["reported_prompt_tokens"] for row in calls if row["reported_prompt_tokens"] is not None) == [11, 22]


def test_preflight_failure_does_not_inherit_other_call_receipt(tmp_path, monkeypatch):
    from easyicu.research_agent.providers import structured_retry
    from easyicu.research_agent.providers.clients import _record_provider_call_receipt
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.protocol import LLMMessage

    wrapper, ledger, _, _, _ = _bound(tmp_path)
    hard_stop = HardStopClient(wrapper, role="planner", task=wrapper._task)
    wrapper.last_finish_reason = "length"
    wrapper.last_usage = {"prompt_tokens": 99}
    wrapper.last_transport_attempts = 3
    wrapper._model = "tampered"
    _record_provider_call_receipt(finish_reason="length", usage={"prompt_tokens": 99}, transport_attempts=3)
    with pytest.raises(ValueError) as exc:
        structured_retry.call_llm_with_structured_retry(
            hard_stop, [LLMMessage(role="user", content="synthetic")],
            parser=lambda value: value, role="planner", max_retries=0, max_tokens=8)
    metadata = exc.value.easyicu_structured_attempt_metadata[0]
    assert metadata["finish_reason"] is None and metadata["usage"] == {}
    assert metadata["transport_attempts"] == 1
    assert ledger.snapshot()["tasks"][0]["calls"] == []


def test_worker_failure_receipt_preserves_this_calls_retry_count(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.clients import current_provider_call_receipt
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.webserver import metadata_provider_client, metadata_provider_concurrency

    wrapper, ledger, _, _, _ = _bound(tmp_path)
    calls = []
    def worker(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            return {"ok": False, "status_code": 503, "type": "ProviderHTTPError"}
        raise RuntimeError("synthetic_worker_failure")
    monkeypatch.setattr(metadata_provider_client, "guarded_call", worker)
    monkeypatch.setattr(metadata_provider_concurrency, "reserve", lambda **_kw: None)
    monkeypatch.setattr(metadata_provider_concurrency, "release", lambda _id: None)
    monkeypatch.setattr(metadata_provider_client.TransportRetrySchedule, "next_wait", lambda *_a, **_kw: (0, None))
    monkeypatch.setattr(metadata_provider_client.time, "sleep", lambda _wait: None)
    with pytest.raises(RuntimeError, match="synthetic_worker_failure"):
        HardStopClient(wrapper, role="planner", task=wrapper._task).complete(
            [LLMMessage(role="user", content="synthetic")], max_tokens=8)
    receipt = current_provider_call_receipt()
    assert receipt.finish_reason is None and receipt.usage_summary == ()
    assert receipt.transport_attempts == 2 and len(calls) == 2
    assert len(ledger.snapshot()["tasks"][0]["calls"]) == 2


def test_authorization_check_preserves_existing_call_receipt(tmp_path):
    from easyicu.research_agent.providers.clients import _record_provider_call_receipt, current_provider_call_receipt
    from easyicu.research_agent.providers.factory import require_provider_client_authorization

    wrapper, _, _, _, _ = _bound(tmp_path)
    _record_provider_call_receipt(finish_reason="length", usage={"prompt_tokens": 7}, transport_attempts=2)
    before = current_provider_call_receipt()
    require_provider_client_authorization(wrapper)
    assert current_provider_call_receipt() == before
    wrapper._model = "tampered"
    with pytest.raises(ValueError):
        require_provider_client_authorization(wrapper)
    assert current_provider_call_receipt() == before


def test_journal_replay_restores_metadata_receipt_without_transport(tmp_path, monkeypatch):
    import hashlib
    import json
    from easyicu.research_agent.planning import outline_retry_journal as journal_owner
    from easyicu.research_agent.providers import structured_retry
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.webserver import metadata_provider_client, metadata_provider_concurrency

    wrapper, _ledger, _, _, _ = _bound(tmp_path)
    monkeypatch.setattr(journal_owner.state_paths, "state_root", lambda: tmp_path / "state")
    sent = []
    def worker(*_args, **_kwargs):
        sent.append(1)
        return {"ok": True, "text": "{", "finish_reason": "length",
                "usage": {"prompt_tokens": 11, "completion_tokens": 1}}
    monkeypatch.setattr(metadata_provider_client, "guarded_call", worker)
    monkeypatch.setattr(metadata_provider_concurrency, "reserve", lambda **_kw: None)
    monkeypatch.setattr(metadata_provider_concurrency, "release", lambda _id: None)
    inner = HardStopClient(wrapper, role="planner", task=wrapper._task)
    authority = {"inner": inner, "task": wrapper._task, "request_sha256": "b" * 64,
                 "dependency_sha256": "c" * 64, "literature_sha256": "e" * 64}
    initial = journal_owner.OutlineRetryJournal(run_dir=tmp_path / "first", **authority)
    messages = [LLMMessage(role="user", content="synthetic")]
    assert initial.complete(messages, max_tokens=8, temperature=0.2) == "{"
    source = tmp_path / "first" / ".runtime" / journal_owner.FILE_NAME
    record = journal_owner.load_retry_journal(source)["records"][0]
    assert record["usage"] == {"prompt_tokens": 11, "completion_tokens": 1}
    resumed = journal_owner.OutlineRetryJournal(run_dir=tmp_path / "continued", source_path=source,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), **authority)
    with pytest.raises(structured_retry.StructuredResponseFailure) as exc:
        structured_retry.call_llm_with_structured_retry(resumed, messages, parser=json.loads,
            role="planner", max_retries=0, max_tokens=8, temperature=0.2)
    attempt = exc.value.attempts[0]
    assert len(exc.value.attempts) == 1 and attempt.finish_reason == "length"
    assert attempt.usage_summary == {"prompt_tokens": 11, "completion_tokens": 1}
    assert attempt.transport_attempts == 1 and sent == [1]


def test_overridden_receipt_flag_cannot_reuse_other_calls_metadata(tmp_path):
    from easyicu.research_agent.providers import structured_retry
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.protocol import LLMMessage

    wrapper, ledger, _, _, _ = _bound(tmp_path)
    wrapper.requires_call_scoped_receipt = False
    wrapper.last_finish_reason = "length"
    wrapper.last_usage = {"prompt_tokens": 99}
    wrapper.last_transport_attempts = 3
    # Even an outer wrapper constructed after the mutation must use the
    # immutable registration record of its inner metadata client.
    hard_stop = HardStopClient(wrapper, role="planner", task=wrapper._task)
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError) as exc:
        structured_retry.call_llm_with_structured_retry(
            hard_stop, [LLMMessage(role="user", content="synthetic")],
            parser=lambda value: value, role="planner", max_retries=0, max_tokens=8)
    metadata = exc.value.easyicu_structured_attempt_metadata[0]
    assert metadata["finish_reason"] is None and metadata["usage"] == {}
    assert metadata["transport_attempts"] == 1
    assert ledger.path.read_bytes() == before


def test_replaced_registered_child_cannot_supply_shared_failure_metadata(tmp_path):
    from easyicu.research_agent.providers import structured_retry
    from easyicu.research_agent.providers.factory import _register_provider_wrapper
    from easyicu.research_agent.providers.protocol import LLMMessage

    class OpaqueWrapper:
        def __init__(self, child):
            self._inner = child
            self.last_finish_reason = "length"
            self.last_usage = {"prompt_tokens": 99}
            self.last_transport_attempts = 3
            _register_provider_wrapper(self, children_getter=lambda: (self._inner,))

        def complete(self, messages, **kwargs):
            return self._inner.complete(messages, **kwargs)

    wrapper, ledger, _, _, _ = _bound(tmp_path)
    outer = OpaqueWrapper(wrapper)
    outer._inner = object()
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError) as exc:
        structured_retry.call_llm_with_structured_retry(
            outer, [LLMMessage(role="user", content="synthetic")],
            parser=lambda value: value, role="planner", max_retries=0, max_tokens=8)
    metadata = exc.value.easyicu_structured_attempt_metadata[0]
    assert metadata["finish_reason"] is None and metadata["usage"] == {}
    assert metadata["transport_attempts"] == 1
    assert ledger.path.read_bytes() == before


def test_unregistered_client_cannot_supply_shared_failure_metadata():
    from easyicu.research_agent.providers import structured_retry
    from easyicu.research_agent.providers.protocol import LLMMessage

    class ForgedClient:
        last_finish_reason = "length"
        last_usage = {"prompt_tokens": 99}
        last_transport_attempts = 3

        def complete(self, messages, **kwargs):
            raise AssertionError("unregistered client must never be called")

    with pytest.raises(ValueError) as exc:
        structured_retry.call_llm_with_structured_retry(
            ForgedClient(), [LLMMessage(role="user", content="synthetic")],
            parser=lambda value: value, role="planner", max_retries=0, max_tokens=8)
    metadata = exc.value.easyicu_structured_attempt_metadata[0]
    assert metadata["finish_reason"] is None and metadata["usage"] == {}
    assert metadata["transport_attempts"] == 1
