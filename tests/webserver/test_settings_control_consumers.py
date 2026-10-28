"""Settings contracts through real owners; no ASGI transport or network."""

from io import BytesIO
import json
from urllib.error import HTTPError

import pytest

from easyicu.webserver import capabilities, settings
from easyicu.webserver.routes import system


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_CONFIG_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(capabilities, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(capabilities, "_AUDIT_PATH", tmp_path / "audit.jsonl")


@pytest.mark.parametrize("enabled,authorized,expected", [(False, True, False), (True, False, False), (True, True, True)])
def test_literature_switch_and_search_consent_gate_both_preplan_sources(isolated, monkeypatch, enabled, authorized, expected):
    from easyicu.research_agent import literature
    from easyicu.research_agent.orchestration.profiles import get_submission_profile
    from easyicu.research_agent.schema import ResearchContext
    from easyicu.webserver.research_launch_runtime import _submission_profile_ref

    system.post_settings({"connector_pubmed_enabled": enabled})
    profile = get_submission_profile(_submission_profile_ref(budget_mode="first_result", live_pubmed=authorized, runner_kind="host"))
    calls = []

    class Agent:
        def __init__(self, *args, **kwargs):
            calls.append((kwargs["enable_pubmed"], kwargs["enable_openalex"]))

        def run(self, context):
            return literature.LiteratureBundle(research_question=context.research_question, citations=[])

    monkeypatch.setattr(literature, "LiteratureAgent", Agent)
    literature.build_preplan_literature_bundle(ResearchContext(research_question="Compare synthetic follow-up patterns.", cohort={"cohort_name": "Synthetic adults", "database": "synthetic", "n_stays": 0}, variables=[]),
                                              enable_pubmed=profile.pipeline_options().get("enable_pubmed", False))
    assert calls == [(expected, expected)]


@pytest.mark.parametrize("payload,available,reason", [
    (b"[]", True, "local_zotero_api_ready"),
    (b"<html>other app</html>", False, "local_zotero_api_invalid_response"),
    (b'{"app": "other"}', False, "local_zotero_api_invalid_response"),
    (b"[123]", False, "local_zotero_api_invalid_response"),
])
def test_zotero_toggle_reaches_probe_and_test_validates_the_api(isolated, monkeypatch, payload, available, reason):
    calls = []

    def urlopen(request, *, timeout):
        calls.append((request.full_url, timeout))
        return BytesIO(payload)

    monkeypatch.setattr(capabilities.urllib.request, "urlopen", urlopen)
    system.post_settings({"connector_zotero_enabled": False})
    assert system.post_capability_zotero_test()["status"]["status"] == "disabled"
    assert calls == []
    system.post_settings({"connector_zotero_enabled": True})
    status = system.post_capability_zotero_test()["status"]
    assert status["available"] is available
    assert status["reason"] == reason
    assert calls[0][0] == "http://127.0.0.1:23119/api/users/0/items?limit=1"
    assert len(calls) == 1


def test_zotero_http_failure_is_typed_and_hosted_never_probes_server_loopback(isolated, monkeypatch):
    def denied(*args, **kwargs):
        raise HTTPError("http://127.0.0.1", 403, "Denied", {}, None)

    monkeypatch.setattr(capabilities.urllib.request, "urlopen", denied)
    system.post_settings({"connector_zotero_enabled": True})
    status = system.post_capability_zotero_test()["status"]
    assert status["reason"] == "local_zotero_api_http_error"
    assert status["http_status"] == 403
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    assert system.post_capability_zotero_test()["status"]["reason"] == "zotero_desktop_only"


def test_zotero_reads_a_bounded_response_and_reports_unreachable(isolated, monkeypatch):
    class Response(BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 64_001, "probe must bound response reads"
            return super().read(size)

    monkeypatch.setattr(capabilities.urllib.request, "urlopen", lambda *a, **kw: Response(b" " * 64_001))
    system.post_settings({"connector_zotero_enabled": True})
    assert system.post_capability_zotero_test()["status"]["reason"] == "local_zotero_api_invalid_response"

    def unreachable(*args, **kwargs):
        raise ConnectionRefusedError

    monkeypatch.setattr(capabilities.urllib.request, "urlopen", unreachable)
    assert system.post_capability_zotero_test()["status"]["reason"] == "local_zotero_api_unavailable"


def test_audit_switch_controls_real_writes_and_count_is_bounded_events(isolated):
    system.post_settings({"tool_audit_enabled": False})
    assert not capabilities.record_tool_event("synthetic", {})["recorded"]
    assert system.post_capability_audit_events()["count"] == 0
    system.post_settings({"tool_audit_enabled": True})
    assert capabilities.record_tool_event("synthetic", {})["recorded"]
    assert system.post_capability_audit_events()["count"] == 1
    assert json.loads(capabilities._AUDIT_PATH.read_text())["event_type"] == "synthetic"


def test_reset_restores_real_switches_without_removing_extensions(isolated, tmp_path, monkeypatch):
    from easyicu.extensions import ExtensionRegistry

    monkeypatch.setenv("EASYICU_EXTENSION_HOME", str(tmp_path / "extensions"))
    registry = ExtensionRegistry()
    registry.install_skill("---\nname: synthetic-reset\ndescription: Explain synthetic inputs.\n---\nUse short paragraphs.\n")
    system.post_settings({"ai_enabled": True, "connector_pubmed_enabled": False, "density": "compact"})
    reset = system.post_settings_reset()
    assert reset["ai_enabled"] is reset["copilot_enabled"] is False
    assert reset["connector_pubmed_enabled"] is True
    assert reset["density"] == "comfortable"
    assert registry.list_public()["skills"][0]["name"] == "synthetic-reset"


def test_placeholder_switches_are_retired_and_status_stays_enforced(isolated):
    from fastapi import HTTPException

    for key in ("prompt_contracts_enabled", "remote_compute_enabled"):
        assert key not in settings.DEFAULTS
        assert key not in settings._COERCE
        with pytest.raises(HTTPException) as failure:
            system.post_settings({key: True})
        assert failure.value.detail["rejected"][0]["code"] == "retired_setting"
    assert capabilities.prompt_contract_status({"prompt_contracts_enabled": False})["status"] == "host_owned"
    assert capabilities.remote_compute_status({"remote_compute_enabled": True})["enabled"] is False
    assert capabilities.validate_compute_target({"compute_target": "hpc"})["ok"] is False


def test_mcp_counts_only_installed_allowlisted_tools_and_master_gates_consumer(isolated, tmp_path, monkeypatch):
    from easyicu.extensions import ExtensionRegistry
    from easyicu.webserver.pi_copilot import tools
    from easyicu.webserver.pi_copilot.contracts import PiSessionRecord, ToolExecutionContext

    monkeypatch.setenv("EASYICU_EXTENSION_HOME", str(tmp_path / "extensions"))
    registry = ExtensionRegistry()
    registry.install_mcp_server(name="synthetic-tools", url="http://127.0.0.1:9988/mcp", allowed_tools=["search"], enabled=True)
    context = ToolExecutionContext(session=PiSessionRecord(session_id="synthetic", extension_activation=registry.snapshot()), allowed_actions={"mcp_read"})
    calls = []
    monkeypatch.setattr(tools, "call_mcp_tool", lambda *args: calls.append(args) or {"ok": True})
    system.post_settings({"mcp_tools_enabled": False})
    assert capabilities.mcp_tool_policy()["allowed_tools"] == []
    assert capabilities.mcp_tool_policy()["blocked_tools"] == ["synthetic-tools.search"]
    assert tools.execute_tool("easyicu_call_mcp_tool", {"server": "synthetic-tools", "tool": "search", "arguments": {}}, context)["code"] == "extension_mcp_master_disabled"
    assert not calls
    system.post_settings({"mcp_tools_enabled": True})
    assert capabilities.mcp_tool_policy()["allowed_tools"] == ["synthetic-tools.search"]
    assert tools.execute_tool("easyicu_call_mcp_tool", {"server": "synthetic-tools", "tool": "search", "arguments": {}}, context)["code"] == "pi_extension_mcp_tool_completed"
    assert len(calls) == 1
    registry.set_enabled(kind="mcp", name="synthetic-tools", enabled=False)
    assert capabilities.mcp_tool_policy()["allowed_tools"] == []
