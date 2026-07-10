"""Security and provenance regressions for MCP HTTP and discovery launcher."""

from __future__ import annotations

import base64
import http.client
import http.server
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


@contextmanager
def _mcp_http_server(*, bearer_token=None, max_body_bytes=1024 * 1024):
    import easyicu.research_agent.mcp_server as mcp

    handler = mcp._make_sse_handler(
        bind_host="127.0.0.1",
        port=0,
        bearer_token=bearer_token,
        allowed_origins=["http://127.0.0.1"],
        max_body_bytes=max_body_bytes,
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _http_request(
    port: int, *, method="POST", path="/jsonrpc", body=b"{}", headers=None
):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    request_headers = dict(headers or {})
    if method == "POST":
        request_headers.setdefault("Content-Type", "application/json")
    connection.request(method, path, body=body, headers=request_headers)
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    return response.status, payload


def _initialize_body() -> bytes:
    return json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    ).encode("utf-8")


def test_mcp_loopback_environment_url_never_forwards_provider_secrets(
    ra, tmp_path, monkeypatch
):
    import easyicu.research_agent.mcp_server as mcp

    seen = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    class FakePipeline:
        def __init__(self, *, workdir, llm):
            pass

        def run(self, *, cohort, **kwargs):
            return SimpleNamespace(model_dump=lambda: {"status": "ok"})

    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8787/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "paid-openai-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "paid-openrouter-secret")
    monkeypatch.setattr(mcp, "OpenAIClient", FakeClient)
    monkeypatch.setattr(mcp, "ResearchAgentPipeline", FakePipeline)

    result = mcp.dispatch(
        "research_agent.run",
        {
            "question": "Inspect the cohort.",
            "cohort_path": str(tmp_path / "cohort.parquet"),
            "provider": "openai",
            "model": "local-model",
        },
    )

    assert result == {"status": "ok"}
    assert seen["base_url"] == "http://127.0.0.1:8787/v1"
    assert seen["api_key"] == "easyicu-local-noauth"
    assert seen["api_key"] not in {"paid-openai-secret", "paid-openrouter-secret"}


def test_mcp_http_originless_loopback_json_is_allowed():
    with _mcp_http_server() as port:
        status, payload = _http_request(port, body=_initialize_body())

    assert status == 200
    assert json.loads(payload)["result"]["serverInfo"]["name"] == (
        "easyicu-research-agent"
    )


@pytest.mark.parametrize("origin", ["https://evil.example", "null"])
def test_mcp_http_rejects_malicious_origin(origin):
    with _mcp_http_server() as port:
        status, _ = _http_request(
            port,
            body=_initialize_body(),
            headers={"Origin": origin},
        )

    assert status == 403


def test_mcp_http_rejects_simple_text_plain_post():
    with _mcp_http_server() as port:
        status, _ = _http_request(
            port,
            body=_initialize_body(),
            headers={"Content-Type": "text/plain"},
        )

    assert status == 415


@pytest.mark.parametrize("host", ["evil.example", "127.0.0.1:65535"])
def test_mcp_http_rejects_untrusted_host_or_wrong_port(host):
    with _mcp_http_server() as port:
        status, _ = _http_request(
            port,
            body=_initialize_body(),
            headers={"Host": host},
        )

    assert status == 400


def test_mcp_http_rejects_oversized_json_before_reading_body():
    with _mcp_http_server(max_body_bytes=8) as port:
        status, _ = _http_request(port, body=_initialize_body())

    assert status == 413


def test_mcp_http_bearer_auth_covers_jsonrpc_and_sse():
    token = "independent-mcp-token"
    with _mcp_http_server(bearer_token=token) as port:
        post_missing, _ = _http_request(port, body=_initialize_body())
        sse_missing, _ = _http_request(port, method="GET", path="/sse", body=None)
        post_ok, _ = _http_request(
            port,
            body=_initialize_body(),
            headers={"Authorization": f"Bearer {token}"},
        )

    assert post_missing == 401
    assert sse_missing == 401
    assert post_ok == 200


def test_mcp_remote_bind_requires_independent_token(monkeypatch):
    import easyicu.research_agent.mcp_server as mcp

    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")

    with pytest.raises(ValueError, match="non-loopback"):
        mcp._validate_sse_server_config("0.0.0.0", None)
    with pytest.raises(ValueError, match="must not reuse OPENAI_API_KEY"):
        mcp._validate_sse_server_config("0.0.0.0", "provider-secret")
    assert (
        mcp._validate_sse_server_config("0.0.0.0", "independent-mcp-token")
        == "independent-mcp-token"
    )


def test_discovery_launcher_loopback_never_forwards_provider_secrets(monkeypatch):
    import tools.run_discovery_to_manuscript as launcher

    seen = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8787/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "paid-openai-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "paid-openrouter-secret")
    monkeypatch.setattr(launcher, "OpenAIClient", FakeClient)

    launcher._build_data_foundation_llm(
        provider="openai",
        model="local-model",
        request_timeout=12.0,
    )

    assert seen["base_url"] == "http://localhost:8787/v1"
    assert seen["api_key"] == "easyicu-local-noauth"
    assert seen["api_key"] not in {"paid-openai-secret", "paid-openrouter-secret"}


def test_discovery_outcome_materialisation_uses_only_frozen_handoff_target():
    import tools.run_discovery_to_manuscript as launcher

    assert launcher._outcome_concepts_for_handoff(
        handoff_target="aki", requested=None
    ) == ("aki",)
    assert launcher._outcome_concepts_for_handoff(
        handoff_target="aki", requested="AKI"
    ) == ("aki",)
    with pytest.raises(SystemExit, match="frozen handoff target 'aki'"):
        launcher._outcome_concepts_for_handoff(handoff_target="aki", requested="death")
    with pytest.raises(SystemExit, match="exactly"):
        launcher._outcome_concepts_for_handoff(
            handoff_target="aki", requested="aki,death"
        )


def test_discovery_handoff_registration_blocks_existing_id_hash_mismatch(tmp_path):
    import tools.run_discovery_to_manuscript as launcher
    from easyicu.research_agent.evidence import EvidenceStore

    store = EvidenceStore(tmp_path / "run")
    source = tmp_path / "handoff.json"
    source.write_text('{"target_outcome":"aki"}', encoding="utf-8")
    original = launcher._register_file_exact(
        store,
        source_path=source,
        kind="log",
        description="handoff",
        evidence_id="discovery_handoff",
        producer="discovery_launcher",
        generation_mode="human_confirmed",
    )
    copied_path = store.root / original.relative_path
    source.write_text('{"target_outcome":"death"}', encoding="utf-8")

    with pytest.raises(ValueError, match="Evidence id collision"):
        launcher._register_file_exact(
            store,
            source_path=source,
            kind="log",
            description="handoff",
            evidence_id="discovery_handoff",
            producer="discovery_launcher",
            generation_mode="human_confirmed",
        )

    assert copied_path.read_text(encoding="utf-8") == '{"target_outcome":"aki"}'


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_discovery_story_exports_receive_closed_provenance(tmp_path):
    import tools.run_discovery_to_manuscript as launcher
    from easyicu.research_agent.evidence import EvidenceStore

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for filename in ("run_status.json", "evidence_audit.json", "numeric_audit.json"):
        _write_json(run_dir / filename, {"source": filename})
    evidence = EvidenceStore(run_dir)
    launcher._register_story_source_records(evidence=evidence, run_dir=run_dir)

    handoff = run_dir / "discovery_handoff.json"
    _write_json(handoff, {"target_outcome": "aki"})
    launcher._register_file_exact(
        evidence,
        source_path=handoff,
        kind="log",
        description="handoff",
        evidence_id="discovery_handoff",
        producer="discovery_launcher",
        generation_mode="human_confirmed",
    )

    figure_dir = run_dir / "publication_figures"
    figure_dir.mkdir()
    contract_path = figure_dir / "easyicu_discovery_story.figure_contract.json"
    _write_json(
        contract_path,
        {
            "figure_id": "easyicu_discovery_story",
            "source_data": [
                "discovery_handoff",
                "run_status",
                "evidence_audit",
                "numeric_audit",
            ],
            "panels": [
                {
                    "panel_id": "C",
                    "metadata": {"story_role": "primary_result"},
                    "evidence_ids": ["run_status"],
                }
            ],
        },
    )
    svg_path = figure_dir / "easyicu_discovery_story.svg"
    png_path = figure_dir / "easyicu_discovery_story.png"
    svg_path.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8"
    )
    png_path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
            "+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
    )

    records = launcher._register_story_figure_provenance(
        evidence=evidence,
        run_dir=run_dir,
        paths={"contract": contract_path, "svg": svg_path, "png": png_path},
    )

    assert records["script"].kind == "code"
    assert records["contract"].metadata["figure_id"] == "easyicu_discovery_story"
    for extension in ("svg", "png"):
        record = records[extension]
        assert record.kind == "figure"
        assert record.script_evidence_id == records["script"].evidence_id
        assert evidence.get(record.script_evidence_id).kind == "code"
        assert (
            record.metadata["contract_evidence_id"] == records["contract"].evidence_id
        )
        assert set(record.metadata["source_evidence_ids"]) <= set(record.inputs)
        assert record.metadata["inputs"] == record.inputs
