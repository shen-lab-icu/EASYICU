from __future__ import annotations

import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import agent_outputs
from easyicu.webserver import copilot_sessions
from easyicu.webserver import dataio
from easyicu.webserver import guided_sessions
from easyicu.webserver import jobs as job_store
from easyicu.webserver import patient_drilldown
from easyicu.webserver import settings as settings_store
from easyicu.webserver import sources as source_store
from easyicu.webserver import __main__ as web_cli
from easyicu.webserver.app import app
from easyicu.webserver.ideas import mining as idea_mining
from easyicu.webserver.jobs import Job, JobCapacityError, JobManager


def _write_large_workspace(root: Path, stays: int = 501) -> Path:
    root.mkdir()
    demographics = pd.DataFrame(
        {
            "stay_id": list(range(1, stays + 1)),
            "age": [60] * stays,
            "sex": ["F"] * stays,
        }
    )
    demographics.to_csv(root / "demographics.csv", index=False)
    vitals = pd.DataFrame(
        {
            "stay_id": list(range(1, stays + 1)),
            "charttime": ["2026-01-01T00:00:00Z"] * stays,
            "hr": [80] * stays,
            "unused_payload": ["not-needed"] * stays,
        }
    )
    vitals.to_csv(root / "vitals.csv", index=False)
    (root / "_manifest.json").write_text(
        json.dumps(
            {
                "database": "miiv",
                "files": [
                    {
                        "file": "demographics.csv",
                        "module": "demographics",
                        "rows": stays,
                    },
                    {"file": "vitals.csv", "module": "vitals", "rows": stays},
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_native_web_rejects_untrusted_host_and_remote_peer() -> None:
    bad_host = TestClient(
        app,
        base_url="http://attacker.invalid",
        client=("127.0.0.1", 50000),
    ).get("/api/health")
    remote_peer = TestClient(
        app,
        client=("192.168.1.20", 50000),
    ).get("/api/health")

    assert bad_host.status_code == 400
    assert remote_peer.status_code == 403


def test_native_web_accepts_bracketed_ipv6_loopback_host() -> None:
    response = TestClient(
        app,
        client=("::1", 50000),
    ).get("/api/health", headers={"Host": "[::1]:8765"})

    assert response.status_code == 200


def test_native_web_sets_defense_in_depth_headers_and_has_no_font_network_call() -> None:
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["permissions-policy"] == (
        "camera=(), microphone=(), geolocation=()"
    )
    csp = response.headers["content-security-policy"]
    assert "default-src 'self'" in csp
    assert "script-src 'self'" in csp
    assert "font-src 'self'" in csp
    assert "object-src 'none'" in csp
    assert "fonts.googleapis.com" not in response.text
    assert "fonts.gstatic.com" not in response.text


def test_native_web_cli_refuses_non_loopback_bind(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert web_cli.run_app("0.0.0.0", 8765) == 2
    assert "local-only" in capsys.readouterr().err


def test_settings_and_request_booleans_parse_false_strings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings_store, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(settings_store, "_CONFIG_PATH", tmp_path / "settings.json")
    client = TestClient(app)

    enabled = client.post("/api/settings", json={"ai_enabled": "true"})
    disabled = client.post("/api/settings", json={"ai_enabled": "false"})
    invalid = client.post(
        "/api/workspaces/register",
        json={"path": str(tmp_path), "active": "not-a-boolean"},
    )

    assert enabled.json()["ai_enabled"] is True
    assert disabled.json()["ai_enabled"] is False
    assert invalid.status_code == 400
    assert invalid.json()["detail"] == {
        "error": "invalid_boolean",
        "field": "active",
    }

    (tmp_path / "settings.json").write_text(
        json.dumps({"ai_enabled": "false"}), encoding="utf-8"
    )
    assert settings_store.load_settings()["ai_enabled"] is False


def test_network_false_string_cannot_enable_idea_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_network(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("network must stay disabled")

    monkeypatch.setattr(idea_mining, "_pubmed_esearch", fail_network)

    result = idea_mining.discover_literature(
        {"topic": "sepsis mortality", "allow_network": "false"}
    )

    assert result["status"] == "blocked_network_opt_in_required"
    assert result["privacy"]["network_calls"] == 0


def test_literature_query_prefers_typed_concepts_over_protocol_prose() -> None:
    query = idea_mining._discovery_queries(
        "Estimate the association in adult ICU stays.",
        "",
        {
            "exposure": (
                "Canonical EasyICU Sepsis-3: suspected infection plus "
                "traditional SOFA >=2 point increase, anchored to onset"
            ),
            "outcome": "In-hospital mortality",
            "exposure_concept": "sep3_sofa1",
            "outcome_concept": "death",
        },
    )[0]

    assert '"Sepsis-3"[Title/Abstract]' in query
    assert '"mortality"[Title/Abstract]' in query
    assert "Canonical EasyICU" not in query
    assert "anchored to onset" not in query


def test_export_cohort_false_strings_remain_false() -> None:
    cohort = dataio._normalize_export_cohort(
        {
            "preset": "all_icu",
            "icd_enabled": "false",
            "exclude_readmissions": "false",
        }
    )

    assert cohort["icd_enabled"] is False
    assert cohort["exclude_readmissions"] is False


def test_pdf_encoded_size_is_rejected_before_base64_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(idea_mining, "_MAX_PDF_BASE64_CHARS", 8)

    def fail_decode(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("oversized payload must not be decoded")

    monkeypatch.setattr(idea_mining.base64, "b64decode", fail_decode)

    with pytest.raises(idea_mining.IdeaMiningWebError) as exc_info:
        idea_mining.ingest_pdf_source(
            {"filename": "large.pdf", "content_base64": "A" * 9}
        )

    assert exc_info.value.detail["error"] == "pdf_too_large"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://10.1.2.3/private",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/admin",
    ],
)
def test_idea_url_fetch_rejects_non_public_targets_without_network(url: str) -> None:
    result = idea_mining._fetch_url_metadata(url)

    assert result["status"] == "unsafe_url"
    assert result["network_calls"] == 0


def test_idea_url_connection_uses_pinned_dns_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dns_answers = iter(
        [
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))],
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))],
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))],
        ]
    )
    dns_calls: list[str] = []

    def fake_getaddrinfo(host, port, **kwargs):  # type: ignore[no-untyped-def]
        dns_calls.append(str(host))
        return next(dns_answers)

    connected: list[tuple[str, int]] = []

    class FakeSocket:
        def settimeout(self, timeout):  # type: ignore[no-untyped-def]
            return None

        def connect(self, sockaddr):  # type: ignore[no-untyped-def]
            connected.append(sockaddr)

        def close(self) -> None:
            return None

    monkeypatch.setattr(idea_mining.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(idea_mining.socket, "socket", lambda *args: FakeSocket())

    target = idea_mining._resolve_public_http_target("http://paper.example/article")
    idea_mining._connect_resolved_addresses(target.addresses, timeout=1)

    assert dns_calls == ["paper.example"]
    assert connected == [("93.184.216.34", 80)]


def test_idea_url_redirect_is_re_resolved_and_private_hop_is_not_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_getaddrinfo(host, port, **kwargs):  # type: ignore[no-untyped-def]
        address = (
            "169.254.169.254" if str(host) == "169.254.169.254" else "93.184.216.34"
        )
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    class RedirectResponse:
        status = 302

        def getheader(self, name, default=None):  # type: ignore[no-untyped-def]
            return (
                "http://169.254.169.254/latest/meta-data/"
                if name.lower() == "location"
                else default
            )

        def close(self) -> None:
            return None

    request_count = 0

    def fake_request(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal request_count
        request_count += 1
        return RedirectResponse()

    monkeypatch.setattr(idea_mining.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(idea_mining, "_request_pinned_target", fake_request)
    req = idea_mining.request.Request("http://paper.example/article")
    target = idea_mining._resolve_public_http_target(req.full_url)

    with pytest.raises(idea_mining.UnsafeURL, match="non-public"):
        idea_mining._open_public_url(req, timeout=1, target=target)

    assert request_count == 1


def test_pinned_https_keeps_original_sni_and_host_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_socket = object()
    tls_socket = object()
    captured: dict = {}

    class FakeContext:
        def wrap_socket(self, sock, *, server_hostname):  # type: ignore[no-untyped-def]
            captured["raw_socket"] = sock
            captured["server_hostname"] = server_hostname
            return tls_socket

    class FakeHTTPResponse:
        status = 200

        def getheader(self, name, default=None):  # type: ignore[no-untyped-def]
            return default

        def close(self) -> None:
            return None

    class FakeHTTPSConnection:
        def __init__(self, host, port, **kwargs):  # type: ignore[no-untyped-def]
            captured["connection_host"] = host
            captured["connection_port"] = port
            self.sock = None

        def request(self, method, target, body=None, headers=None):  # type: ignore[no-untyped-def]
            captured["socket"] = self.sock
            captured["method"] = method
            captured["target"] = target
            captured["headers"] = headers

        def getresponse(self) -> FakeHTTPResponse:
            return FakeHTTPResponse()

        def close(self) -> None:
            return None

    target = idea_mining._ResolvedPublicTarget(
        url="https://paper.example:8443/article?id=1",
        scheme="https",
        hostname="paper.example",
        port=8443,
        request_target="/article?id=1",
        host_header="paper.example:8443",
        addresses=((socket.AF_INET, socket.SOCK_STREAM, 6, ("93.184.216.34", 8443)),),
    )
    monkeypatch.setattr(
        idea_mining,
        "_connect_resolved_addresses",
        lambda addresses, timeout: raw_socket,
    )
    monkeypatch.setattr(idea_mining.ssl, "create_default_context", FakeContext)
    monkeypatch.setattr(
        idea_mining.http.client,
        "HTTPSConnection",
        FakeHTTPSConnection,
    )

    response = idea_mining._request_pinned_target(
        target,
        idea_mining.request.Request(target.url),
        timeout=1,
    )
    response.close()

    assert captured["server_hostname"] == "paper.example"
    assert captured["connection_host"] == "paper.example"
    assert captured["socket"] is tls_socket
    assert captured["headers"]["Host"] == "paper.example:8443"


def test_workspace_summary_labels_bounded_500_stay_sample(tmp_path: Path) -> None:
    export = _write_large_workspace(tmp_path / "export")

    result = dataio.summarize_export_workspace(str(export))

    assert result["summary"]["stays"] == 501
    assert result["summary"]["total_stays"] == 501
    assert result["summary"]["sampled_stays"] == 500
    assert result["summary"]["snapshot_basis"] == "bounded_first_500_stays"
    assert result["cohort"]["total_stays"] == 501
    assert result["cohort"]["sampled_stays"] == 500


def test_agent_outputs_label_bounded_500_stay_sample(tmp_path: Path) -> None:
    export = _write_large_workspace(tmp_path / "export")
    workspace = dataio.summarize_export_workspace(str(export))

    artifacts = agent_outputs.build_agent_output_artifacts(
        export_path=str(export),
        source={"files": workspace["files"], "path": str(export)},
        summary=workspace["summary"],
        cohort=workspace["cohort"],
        quality=workspace["quality"],
    )

    assert artifacts
    for artifact in artifacts.values():
        assert artifact["sampling"] == {
            "total_entities": 501,
            "sampled_entities": 500,
            "sample_limit": 500,
            "snapshot_basis": "bounded_first_500_stays",
        }


def test_legacy_predictor_selection_has_no_sepsis_or_sofa_priority() -> None:
    selected = agent_outputs._select_predictor(
        {
            ("labs", "creatinine"): {"1": 0.8, "2": 1.1, "3": 1.7, "4": 2.0},
            ("sofa2_score", "sofa2"): {"1": 1.0, "2": 2.0, "3": 3.0},
        },
        {"1": False, "2": True, "3": False, "4": True},
    )

    assert selected is not None
    assert selected["module"] == "labs"
    assert selected["feature"] == "creatinine"


def test_workspace_and_agent_use_id_first_predicate_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = _write_large_workspace(tmp_path / "export")
    original = dataio._read_export_frame
    original_ids = dataio._read_stay_id_frame
    calls: list[dict] = []

    def spy(path, *, columns=None, stay_ids=None):  # type: ignore[no-untyped-def]
        calls.append(
            {
                "path": Path(path).name,
                "columns": list(columns) if columns is not None else None,
                "stay_ids": set(stay_ids) if stay_ids is not None else None,
            }
        )
        return original(path, columns=columns, stay_ids=stay_ids)

    def spy_ids(path, *, stay_ids=None):  # type: ignore[no-untyped-def]
        calls.append(
            {
                "path": Path(path).name,
                "columns": ["stay_id"],
                "stay_ids": set(stay_ids) if stay_ids is not None else None,
            }
        )
        return original_ids(path, stay_ids=stay_ids)

    monkeypatch.setattr(dataio, "_read_export_frame", spy)
    monkeypatch.setattr(dataio, "_read_stay_id_frame", spy_ids)
    workspace = dataio.summarize_export_workspace(str(export))

    assert calls[0]["columns"] == ["stay_id"]
    assert calls[0]["stay_ids"] is None
    assert all(call["columns"] is not None for call in calls)
    assert all(
        call["columns"] == ["stay_id"] or call["stay_ids"] is not None
        for call in calls[1:]
    )
    assert all(
        len(call["stay_ids"]) <= 500 for call in calls if call["stay_ids"] is not None
    )

    calls.clear()
    agent_outputs.build_agent_output_artifacts(
        export_path=str(export),
        source={"files": workspace["files"], "path": str(export)},
        summary=workspace["summary"],
        cohort=workspace["cohort"],
        quality=workspace["quality"],
    )
    assert calls[0]["columns"] == ["stay_id"]
    assert calls[0]["stay_ids"] is None
    assert all(call["stay_ids"] is not None for call in calls[1:])
    assert all(call["columns"] is not None for call in calls)
    assert all("charttime" not in (call["columns"] or []) for call in calls)


def test_parquet_export_reader_pushes_projection_and_stay_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "vitals.parquet"
    pd.DataFrame(
        {
            "stay_id": list(range(1, 1001)),
            "hr": list(range(1000)),
            "unused_payload": ["large"] * 1000,
        }
    ).to_parquet(path, index=False)

    def fail_full_pandas_read(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("pandas full parquet read must not be used")

    monkeypatch.setattr(pd, "read_parquet", fail_full_pandas_read)
    frame = dataio._read_export_frame(
        path,
        columns=["stay_id", "hr"],
        stay_ids={"2", "999"},
    )

    assert list(frame.columns) == ["stay_id", "hr"]
    assert set(frame["stay_id"].tolist()) == {2, 999}


def test_patient_preview_parquet_batch_failure_never_falls_back_to_full_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow.parquet as pq

    path = tmp_path / "vitals.parquet"
    pd.DataFrame({"stay_id": [1, 2], "hr": [80, 90]}).to_parquet(
        path, index=False
    )

    def fail_bounded_read(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("bounded parquet batch read failed")

    def fail_full_read(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("full parquet fallback must not be used")

    monkeypatch.setattr(pq, "ParquetFile", fail_bounded_read)
    monkeypatch.setattr(pd, "read_parquet", fail_full_read)

    with pytest.raises(RuntimeError, match="bounded parquet batch read failed"):
        patient_drilldown._read_table_preview(path, ["stay_id", "hr"], 1)


def test_extraction_folder_picker_uses_text_content_for_server_names() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "src/easyicu/webserver/static/js/screens-extraction.js"
    ).read_text(encoding="utf-8")

    assert "name.textContent = String(en.name || '')" in script
    assert "hint.textContent = String(en.hint)" in script
    assert "failure.textContent" in script
    assert "${en.name}" not in script
    assert "${en.hint}" not in script


def test_visualization_owner_surfaces_bounded_workspace_sample() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "src/easyicu/webserver/static/js/screens-viz.js"
    ).read_text(encoding="utf-8")

    assert "function workspaceSamplingNote(summary)" in script
    assert "sampled_stays" in script
    assert "total_stays" in script
    assert "snapshot_basis" in script


def test_same_second_guided_and_copilot_sessions_get_unique_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guided_sessions, "_CONFIG_DIR", tmp_path / "guided-cfg")
    monkeypatch.setattr(
        guided_sessions,
        "_CONFIG_PATH",
        tmp_path / "guided-cfg" / "sessions.json",
    )
    monkeypatch.setattr(guided_sessions, "_PROJECTS_ROOT", tmp_path / "guided-projects")
    monkeypatch.setattr(guided_sessions, "_now", lambda: "2026-07-10T12:00:00Z")
    monkeypatch.setattr(copilot_sessions, "_CONFIG_DIR", tmp_path / "copilot-cfg")
    monkeypatch.setattr(
        copilot_sessions,
        "_CONFIG_PATH",
        tmp_path / "copilot-cfg" / "sessions.json",
    )
    monkeypatch.setattr(
        copilot_sessions, "_PROJECTS_ROOT", tmp_path / "copilot-projects"
    )
    monkeypatch.setattr(copilot_sessions, "_now", lambda: "2026-07-10T12:00:00Z")

    guided_ids = {
        guided_sessions.create_guided_session({"context": {"route": "entry"}})[
            "session"
        ]["id"]
        for _ in range(2)
    }
    copilot_ids = {
        copilot_sessions.create_session({"context": {"route": "entry"}})["session"][
            "id"
        ]
        for _ in range(2)
    }

    assert len(guided_ids) == 2
    assert len(copilot_ids) == 2


def test_job_manager_retains_only_bounded_completed_jobs() -> None:
    manager = JobManager(max_completed=2)
    jobs = [manager.submit("quick", lambda _job, i=i: {"value": i}) for i in range(5)]
    deadline = time.time() + 2
    while any(job.status == "running" for job in jobs) and time.time() < deadline:
        time.sleep(0.01)

    retained = [job for job in jobs if manager.get(job.id) is not None]
    assert all(job.status == "done" for job in jobs)
    assert len(retained) == 2
    assert retained == jobs[-2:]


def test_job_manager_preserves_a_stable_typed_failure_code() -> None:
    manager = JobManager(max_completed=2)

    class TypedFailure(RuntimeError):
        code = "research_pipeline_provider_timeout"

    def fail(_job):  # type: ignore[no-untyped-def]
        raise TypedFailure("The provider timed out before analysis.")

    job = manager.submit("typed-failure", fail)
    deadline = time.time() + 2
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.01)

    assert job.status == "failed"
    assert job.error == (
        "research_pipeline_provider_timeout: "
        "The provider timed out before analysis."
    )


def test_job_manager_applies_running_job_backpressure() -> None:
    manager = JobManager(max_completed=2, max_running=1)
    release = threading.Event()
    started = threading.Event()

    def blocking_runner(_job):  # type: ignore[no-untyped-def]
        started.set()
        release.wait(timeout=2)
        return {"ok": True}

    first = manager.submit("blocking", blocking_runner)
    assert started.wait(timeout=1)
    with pytest.raises(JobCapacityError) as exc_info:
        manager.submit("rejected", lambda _job: {})
    assert exc_info.value.max_running == 1
    assert exc_info.value.running == 1
    release.set()
    deadline = time.time() + 2
    while first.status == "running" and time.time() < deadline:
        time.sleep(0.01)
    assert first.status == "done"


def test_job_endpoint_returns_429_when_local_capacity_is_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = JobManager(max_completed=2, max_running=1)
    release = threading.Event()
    started = threading.Event()

    def blocking_runner(_job):  # type: ignore[no-untyped-def]
        started.set()
        release.wait(timeout=2)
        return {"ok": True}

    first = manager.submit("blocking", blocking_runner)
    assert started.wait(timeout=1)
    monkeypatch.setattr(job_store, "MANAGER", manager)
    monkeypatch.setattr(
        dataio,
        "make_convert_runner",
        lambda path, database: lambda _job: {"ok": True},
    )

    response = TestClient(app).post(
        "/api/jobs/convert",
        json={"path": "/tmp/easyicu-test", "database": "miiv"},
    )
    release.set()
    deadline = time.time() + 2
    while first.status == "running" and time.time() < deadline:
        time.sleep(0.01)

    assert response.status_code == 429
    assert response.json()["detail"] == {
        "error": "job_capacity_exceeded",
        "running": 1,
        "max_running": 1,
        "reason": "Wait for a running local job to finish before retrying.",
    }


def test_job_events_replay_includes_terminal_end_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = JobManager(max_completed=2)
    monkeypatch.setattr(job_store, "MANAGER", manager)

    def runner(job):  # type: ignore[no-untyped-def]
        job.emit({"type": "progress", "phase": "testing"})
        return {"ok": True}

    job = manager.submit("sse-replay", runner)
    deadline = time.time() + 2
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.01)

    response = TestClient(app).get(f"/api/jobs/{job.id}/events")
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert events[0]["type"] == "progress"
    assert events[-1]["type"] == "end"
    assert events[-1]["status"] == "done"
    assert events[-1]["result"] == {"ok": True}


def test_job_cancel_and_terminal_transition_is_atomic() -> None:
    cancelled = Job("cancel-first", "test")
    assert cancelled.emit({"type": "progress", "phase": "loading"}) is True
    assert cancelled.request_cancel("test_cancel") is True
    assert cancelled.request_cancel("duplicate") is True
    assert cancelled.complete_from_runner({"partial": True}) is True

    snapshot = cancelled.snapshot()
    assert snapshot["status"] == "cancelled"
    assert [event["type"] for event in snapshot["events"]].count(
        "cancel_requested"
    ) == 1
    assert snapshot["events"][-1]["type"] == "end"
    assert snapshot["events"][-1]["status"] == "cancelled"
    assert [event["seq"] for event in snapshot["events"]] == list(
        range(len(snapshot["events"]))
    )
    assert cancelled.emit({"type": "progress", "phase": "late"}) is False
    assert cancelled.request_cancel("after_terminal") is False
    events, status = cancelled.events_since(1)
    assert status == "cancelled"
    assert events == snapshot["events"][1:]

    completed = Job("complete-first", "test")
    assert completed.complete_from_runner({"ok": True}) is True
    assert completed.request_cancel("too_late") is False
    assert completed.snapshot()["status"] == "done"


def test_job_cancel_callbacks_run_once_and_can_be_unregistered() -> None:
    job = Job("cancel-callbacks", "test")
    called: list[str] = []

    unregister_first = job.register_cancel_callback(
        lambda: called.append("first")
    )
    unregister_first()
    job.register_cancel_callback(lambda: called.append("second"))

    assert job.request_cancel("user_requested") is True
    assert job.request_cancel("duplicate") is True
    assert called == ["second"]

    job.register_cancel_callback(lambda: called.append("late"))
    assert called == ["second", "late"]


def test_job_cancel_callback_failure_does_not_block_other_callbacks() -> None:
    job = Job("cancel-callback-error", "test")
    called: list[str] = []

    def fail_callback() -> None:
        raise RuntimeError("callback failed")

    job.register_cancel_callback(fail_callback)
    job.register_cancel_callback(lambda: called.append("healthy"))

    assert job.request_cancel("user_requested") is True
    assert called == ["healthy"]


def test_source_registry_serializes_updates_and_writes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "cfg" / "sources.json"
    monkeypatch.setattr(source_store, "_CONFIG_DIR", config_path.parent)
    monkeypatch.setattr(source_store, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])

    def describe(path: str) -> dict:
        return {
            "ok": True,
            "label": Path(path).name,
            "database": "miiv",
            "modules": [],
            "files": [],
            "summary": {"stays": 0},
        }

    monkeypatch.setattr(source_store.dataio, "describe_export_source", describe)
    paths = [str(tmp_path / "export-a"), str(tmp_path / "export-b")]

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(source_store.register_source, paths))

    registry = source_store.load_registry()
    assert {source["path"] for source in registry["sources"]} == set(paths)
    assert json.loads(config_path.read_text(encoding="utf-8"))["sources"]
    assert not config_path.with_suffix(".json.tmp").exists()
