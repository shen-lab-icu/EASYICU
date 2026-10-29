"""Declared browser observations remain version-bound and below science authority."""
from copy import deepcopy
import json

import pytest

from easyicu.webserver.pi_copilot.workspace_runtime_qa import (
    RuntimeQaError, RuntimeQaStore, SCHEMA, contract_from_html,
    runtime_qa_bridge, validate_contract, validate_report,
)

SHA = "a" * 64
OTHER = "b" * 64
CONTRACT = {"schema": SCHEMA, "checks": [
    {"id": "items", "kind": "text_number", "selector": ".records", "expected": 12},
    {"id": "export", "kind": "download_control", "selector": "button[data-export]"},
]}


def report(*, status="passed", download="pending", sha=SHA):
    return {"schema": SCHEMA, "source_sha256": sha, "status": "complete",
            "scientific_verified": True, "browser_download_verified": True,
            "csv_export_ready": True, "runtime_errors": 0, "checks": [
                {"id": "items", "kind": "text_number", "status": status,
                 "code": "number_matches", "observed": 12, "expected": 12},
                {"id": "export", "kind": "download_control", "status": download,
                 "code": "csv_export_not_yet_observed"},
            ]}


def test_contract_is_declarative_bounded_and_embedded_as_json_only():
    normalized = validate_contract(CONTRACT)
    html = '<script type="application/json" data-easyicu-runtime-qa>' + json.dumps(CONTRACT) + '</script>'
    assert contract_from_html(html) == normalized
    assert contract_from_html("<h1>Static only</h1>") is None
    with pytest.raises(RuntimeQaError):
        contract_from_html(html + html)
    with pytest.raises(RuntimeQaError):
        contract_from_html('<script data-easyicu-runtime-qa>anything</script>')
    for change in [{"eval": "alert(1)"}, {"checks": []}, {"schema": "unknown"}]:
        with pytest.raises(RuntimeQaError):
            validate_contract({**CONTRACT, **change})
    bad = deepcopy(CONTRACT)
    bad["checks"][0]["selector"] = "x" * 257
    with pytest.raises(RuntimeQaError):
        validate_contract(bad)
    bad["checks"][0]["selector"] = ".count"
    bad["checks"][0]["id"] = 17
    with pytest.raises(RuntimeQaError):
        validate_contract(bad)


def test_report_completion_recomputed_and_untrusted_claims_discarded():
    normalized = validate_report(report(), validate_contract(CONTRACT), SHA)
    assert normalized["status"] == "partial"
    assert normalized["coverage"] == ["download_control", "text_number"]
    assert normalized["scientific_verified"] is False
    assert normalized["browser_download_verified"] is False
    assert normalized["csv_export_ready"] is False
    claimed_csv = validate_report(report(download="passed"), validate_contract(CONTRACT), SHA)
    assert claimed_csv["status"] == "partial"
    assert claimed_csv["checks"][1]["status"] == "pending"
    failed = validate_report(report(status="failed"), validate_contract(CONTRACT), SHA)
    assert failed["status"] == "failed"
    missing = report(download="passed")
    missing["checks"] = missing["checks"][:1]
    assert validate_report(missing, validate_contract(CONTRACT), SHA)["status"] == "partial"
    missing["checks"] = []
    assert validate_report(missing, None, SHA)["status"] == "partial"
    missing["runtime_errors"] = 1
    assert validate_report(missing, None, SHA)["status"] == "failed"


@pytest.mark.parametrize("mutate", [
    lambda value: value.update(source_sha256=OTHER),
    lambda value: value["checks"][0].update(id="undeclared"),
    lambda value: value["checks"][0].update(id=[]),
    lambda value: value["checks"][0].update(observed=float("nan")),
    lambda value: value.update(warnings=[{}]),
    lambda value: value.update(runtime_errors=True),
])
def test_report_rejects_unbound_unbounded_or_nonfinite_observations(mutate):
    value = report()
    mutate(value)
    with pytest.raises(RuntimeQaError):
        validate_report(value, validate_contract(CONTRACT), SHA)


def test_store_nonce_project_file_hash_expiry_and_bounded_eviction():
    clock = [100.0]
    store = RuntimeQaStore(clock=lambda: clock[0], max_entries=2, ttl_seconds=10)
    request = store.begin("project", "page.html", SHA, CONTRACT)
    nonce = request["nonce"]
    assert store.request("project", "page.html", SHA, nonce) == validate_contract(CONTRACT)
    for project, file, sha, token in [
        ("other", "page.html", SHA, nonce), ("project", "other.html", SHA, nonce),
        ("project", "page.html", OTHER, nonce), ("project", "page.html", SHA, "文"),
    ]:
        with pytest.raises(RuntimeQaError, match="request_stale"):
            store.request(project, file, sha, token)
        with pytest.raises(RuntimeQaError, match="request_stale"):
            store.record(project, file, sha, token, report())
    assert store.read("project", "page.html", SHA)["status"] == "partial"
    store.begin("project", "second.html", SHA)
    store.begin("project", "third.html", SHA)
    with pytest.raises(RuntimeQaError):
        store.request("project", "page.html", SHA, nonce)
    current = store.begin("project", "page.html", SHA, CONTRACT)
    clock[0] = 111
    with pytest.raises(RuntimeQaError):
        store.request("project", "page.html", SHA, current["nonce"])
    assert store.read("project", "page.html", SHA)["code"] == "browser_runtime_pending"


def test_csv_ready_is_host_only_not_os_saved_and_reload_keeps_failure_count():
    store = RuntimeQaStore()
    nonce = store.begin("project", "page.html", SHA, CONTRACT)["nonce"]
    assert store.record("project", "page.html", SHA, nonce, report(status="failed"))["status"] == "failed"
    store.note_csv_ready("project", "page.html", OTHER)
    store.record("project", "page.html", SHA, nonce, report())
    ready = store.read("project", "page.html", SHA)
    assert ready["previous_failed_runs"] == 1 and ready["csv_export_ready"] is False
    store.note_csv_ready("project", "page.html", SHA)
    ready = store.read("project", "page.html", SHA)
    assert ready["status"] == "complete" and ready["csv_export_ready"] is True
    assert ready["checks"][1]["code"] == "csv_ready_to_save"
    assert ready["browser_download_verified"] is False and ready["scientific_verified"] is False
    new = store.begin("project", "page.html", SHA, CONTRACT)
    assert new["nonce"] != nonce
    assert store.read("project", "page.html", SHA)["previous_failed_runs"] == 1
    with pytest.raises(RuntimeQaError):
        store.record("project", "page.html", SHA, nonce, report())


def test_bridge_escapes_contract_and_adds_no_host_network_or_download_authority():
    contract = deepcopy(CONTRACT)
    contract["checks"][0]["selector"] = '[data-label="</script><script>bad</script>"]'
    child, wrapper = runtime_qa_bridge(SHA, OTHER, contract)
    assert child.count("</script>") == 1
    assert "\\u003c/script>" in child
    assert "event.source!==frame.contentWindow" in wrapper
    assert "event.origin!=='null'" in wrapper
    assert "fetch(" not in child + wrapper
    assert "eval(" not in child + wrapper
    assert "allow-downloads" not in child + wrapper


def test_runtime_readback_tool_requires_current_checked_bytes(tmp_path):
    from easyicu.webserver.pi_copilot.contracts import PiSessionRecord, ToolExecutionContext
    from easyicu.webserver.pi_copilot.tools import execute_tool
    context = ToolExecutionContext(
        data_handling="zero_retention_declared",
        session=PiSessionRecord(session_id="session", project_id="project", agent_mode="workspace"),
        workspace_root=tmp_path, allowed_actions={"workspace_write"}, runtime_qa_store=RuntimeQaStore(),
    )
    execute_tool("easyicu_write_project_file", {"file": "page.html", "content": "<h1>UI</h1>"}, context)
    checked = execute_tool("easyicu_check_project_file", {"file": "page.html"}, context)
    assert checked["details"]["verification_scope"] == "static_only"
    sha = checked["details"]["checked_sha256"]
    pending = execute_tool("easyicu_get_project_runtime_qa", {"file": "page.html", "checked_sha256": sha}, context)
    assert pending["details"]["runtime_qa"]["status"] == "partial"
    assert pending["details"]["runtime_qa"]["code"] == "browser_runtime_pending"
    from easyicu.webserver.pi_copilot.contracts import PiCopilotError
    with pytest.raises(PiCopilotError) as stale:
        execute_tool("easyicu_get_project_runtime_qa", {"file": "page.html", "checked_sha256": OTHER}, context)
    assert stale.value.code == "pi_workspace_preview_check_stale"
