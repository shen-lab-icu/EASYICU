"""Direct HTTP-owner checks without ASGI sockets; all state is synthetic."""

from types import SimpleNamespace
from threading import RLock

from fastapi import HTTPException, Response
from pydantic import ValidationError
import pytest

from easyicu.webserver import jobs
from easyicu.webserver.attention_service import attention_for_service
from easyicu.webserver.attention_store import AttentionStore, AttentionUnavailable
from easyicu.webserver.pi_copilot.contracts import PiSessionRecord
from easyicu.webserver.pi_copilot.replay_store import PiConversationReplayStore
from easyicu.webserver.routes import attention
from tests.webserver.attention_fixtures import archive, result_ready


def service_at(home):
    record = PiSessionRecord(session_id="conversation-1", project_id="project-1")
    return SimpleNamespace(store_path=home / "sessions.json",
                           _lock=RLock(),
                           replay_store=PiConversationReplayStore(home / "replay"),
                           _read_records=lambda: [record])


def test_routes_deliver_ack_and_cache_no_metadata(tmp_path, monkeypatch):
    service = service_at(tmp_path)
    monkeypatch.setattr(attention, "get_pi_copilot_service", lambda: service)
    archive(service.replay_store, result=result_ready())
    response = Response()
    feed = attention.attention_feed(response, "browser-1")
    assert response.headers["cache-control"] == "no-store"
    assert len(feed["items"]) == 1
    body = attention.AttentionAcknowledgement(client_id="browser-1", sequences=[1])
    assert attention.acknowledge_attention(body, response)["acknowledged"] == [1]
    assert attention.attention_feed(Response(), "browser-1")["items"] == []


def test_attention_routes_are_registered_before_the_static_mount():
    from fastapi.routing import APIRoute
    from easyicu.webserver.app import app

    routes = {route.path: route for route in attention.router.routes if isinstance(route, APIRoute)}
    assert routes["/api/attention"].endpoint is attention.attention_feed
    assert routes["/api/attention"].methods == {"GET"}
    assert routes["/api/attention/ack"].endpoint is attention.acknowledge_attention
    assert routes["/api/attention/ack"].methods == {"POST"}
    indices = [index for index, route in enumerate(app.routes)
               if getattr(route, "original_router", None) is attention.router
               or (isinstance(route, APIRoute) and route.endpoint is attention.acknowledge_attention)]
    assert indices and max(indices) < len(app.routes) - 1
    assert app.routes[-1].name == "static"


def test_hosted_instances_keep_feed_and_client_acknowledgements_private(tmp_path, monkeypatch):
    alice, bob = service_at(tmp_path / "alice"), service_at(tmp_path / "bob")
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "alice"))
    archive(alice.replay_store, result=result_ready("alice-run"))
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "bob"))
    archive(bob.replay_store, result=result_ready("bob-run"))
    feed = attention_for_service(bob)
    assert [row["run_id"] for row in feed.page(client_id="same-browser")["items"]] == ["bob-run"]
    feed.acknowledge(client_id="same-browser", sequences=[1])
    with pytest.raises(HTTPException) as refused:
        attention_for_service(alice)
    assert refused.value.status_code == 403
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "alice"))
    assert len(attention_for_service(alice).page(client_id="same-browser")["items"]) == 1


def test_hosted_rejects_a_feed_redirected_to_another_home(tmp_path, monkeypatch):
    service = service_at(tmp_path / "alice")
    service.replay_store.attention = AttentionStore(tmp_path / "bob" / "attention.sqlite3")
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "alice"))
    with pytest.raises(HTTPException) as refused:
        attention_for_service(service)
    assert refused.value.status_code == 403
    assert not (tmp_path / "bob").exists()


def test_feed_repairs_a_crash_after_replay_commit(tmp_path, monkeypatch):
    service = service_at(tmp_path)
    store = service.replay_store.attention
    def unavailable(_payload):
        raise AttentionUnavailable("synthetic_disk_failure")
    with monkeypatch.context() as patch:
        patch.setattr(store, "observe_replay", unavailable)
        archive(service.replay_store, result=result_ready())
    assert store.page(client_id="browser-1")["items"] == []
    assert len(attention_for_service(service).page(client_id="browser-1")["items"]) == 1
    store.acknowledge(client_id="browser-1", sequences=[1])
    assert attention_for_service(service).page(client_id="browser-1")["items"] == []


def test_reconciliation_catches_a_settled_job_before_its_watcher(tmp_path, monkeypatch):
    service = service_at(tmp_path)
    service.replay_store.record_host_action(
        session_id="conversation-1", project_id="project-1", action_id="start",
        action_code="execute_plan", child_job_id="job-1",
    )
    job = jobs.Job("job-1", "agent-run")
    job.finish("failed", error="research_pipeline_execution_failed")
    monkeypatch.setattr(jobs, "MANAGER", SimpleNamespace(get=lambda key: job if key == job.id else None))
    item, = attention_for_service(service).page(client_id="browser-1")["items"]
    assert item["reason_code"] == "research_pipeline_execution_failed"


@pytest.mark.parametrize("changes", [
    {"client_id": "../other"}, {"client_id": ""}, {"client_id": "x" * 129},
    {"sequences": []}, {"sequences": [True]}, {"sequences": ["1"]},
    {"sequences": [0]}, {"sequences": [2**53]}, {"sequences": list(range(1, 102))},
    {"user_id": "other"}, {"project_id": "other"},
])
def test_ack_contract_accepts_only_bounded_client_and_sequences(changes):
    with pytest.raises(ValidationError):
        attention.AttentionAcknowledgement.model_validate({"client_id": "browser-1", "sequences": [1], **changes})


@pytest.mark.parametrize("changes", [
    {"client_id": "../other"}, {"after": -1}, {"after": True}, {"after": 2**53},
    {"limit": 0}, {"limit": 101}, {"limit": True},
])
def test_feed_owner_bounds_input_before_io(tmp_path, changes):
    store = AttentionStore(tmp_path / "attention.sqlite3")
    with pytest.raises(ValueError):
        store.page(**{"client_id": "browser-1", "after": 0, "limit": 100, **changes})
    assert not store.path.exists()


def test_storage_error_is_a_retryable_code_without_details(monkeypatch):
    def unavailable():
        raise AttentionUnavailable("synthetic details /private/example")
    monkeypatch.setattr(attention, "_feed", unavailable)
    with pytest.raises(HTTPException) as error:
        attention.attention_feed(Response(), "browser-1")
    assert error.value.status_code == 503
    assert error.value.detail == {"error": "attention_store_unavailable"}


def test_store_refuses_symlink_and_future_schema(tmp_path):
    import sqlite3

    path = tmp_path / "foreign.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=42")
    with pytest.raises(AttentionUnavailable, match="version_invalid"):
        AttentionStore(path).page(client_id="browser-1")
    alias = tmp_path / "alias.sqlite3"
    alias.symlink_to(path)
    with pytest.raises(AttentionUnavailable):
        AttentionStore(alias).page(client_id="browser-1")


def test_synchronous_append_delivery_repairs_from_its_owned_checkpoint(tmp_path, monkeypatch):
    from easyicu.webserver import research_append_run
    from easyicu.webserver.attention_service import publish_append_review
    from easyicu.webserver.pi_copilot.contracts import AuthorityBinding

    service = service_at(tmp_path)
    record = service._read_records()[0]
    record.binding = AuthorityBinding(run_id="append-" + "a" * 32, study_context_id="study-1")
    checkpoint = SimpleNamespace(run_id=record.binding.run_id, state="pending", created_at="2026-10-26T00:00:00+00:00")
    child = SimpleNamespace(checkpoint=checkpoint, record={"study": {"id": "study-1"}})
    monkeypatch.setattr(research_append_run, "append_run_for_id", lambda _run: child)
    def unavailable(_transitions):
        raise AttentionUnavailable("synthetic_delivery_failure")
    with monkeypatch.context() as patch:
        patch.setattr(service.replay_store.attention, "record", unavailable)
        publish_append_review(service, project_id="project-1", session_id="conversation-1", checkpoint=checkpoint)
    feed = attention_for_service(service)
    item, = feed.page(client_id="browser-1")["items"]
    assert (item["run_id"], item["kind"]) == (record.binding.run_id, "needs_review")
    feed.acknowledge(client_id="browser-1", sequences=[item["sequence"]])
    assert attention_for_service(service).page(client_id="browser-1")["items"] == []


def test_append_recovery_does_not_use_another_studys_checkpoint(tmp_path, monkeypatch):
    from easyicu.webserver import research_append_run
    from easyicu.webserver.pi_copilot.contracts import AuthorityBinding

    service = service_at(tmp_path)
    service._read_records()[0].binding = AuthorityBinding(run_id="append-" + "a" * 32, study_context_id="study-1")
    child = SimpleNamespace(record={"study": {"id": "another-study"}}, checkpoint=SimpleNamespace(
        state="pending", run_id="append-" + "a" * 32, created_at="2026-10-26T00:00:00Z",
    ))
    monkeypatch.setattr(research_append_run, "append_run_for_id", lambda _run: child)
    assert attention_for_service(service).page(client_id="browser-1")["items"] == []


def test_store_itself_refuses_another_home(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "alice"))
    store = AttentionStore(tmp_path / "bob" / "attention.sqlite3")
    with pytest.raises(HTTPException) as error:
        store.page(client_id="browser-1")
    assert error.value.status_code == 403
    assert not store.path.parent.exists()


def test_reconciliation_checks_session_ownership_before_reading_it(tmp_path, monkeypatch):
    service = service_at(tmp_path / "alice")
    service.store_path = tmp_path / "bob" / "sessions.json"
    service._read_records = lambda: pytest.fail("another user's sessions were read")
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "alice"))
    with pytest.raises(HTTPException) as error:
        attention_for_service(service)
    assert error.value.status_code == 403
