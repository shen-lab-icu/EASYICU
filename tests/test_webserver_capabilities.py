from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from easyicu.webserver import capabilities
from easyicu.webserver import settings as settings_store
from easyicu.webserver.app import app


def _settings(**patch):
    return {**settings_store.DEFAULTS, **patch}


def test_capability_status_reflects_settings_and_tool_policy(monkeypatch) -> None:
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(
            connector_pubmed_enabled=False,
            connector_zotero_enabled=False,
            mcp_tools_enabled=False,
            remote_compute_enabled=False,
        ),
    )

    response = TestClient(app).get("/api/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["settings"]["connector_pubmed_enabled"] is False
    assert body["capabilities"]["pubmed_connector"]["status"] == "disabled"
    assert body["capabilities"]["zotero_connector"]["status"] == "disabled"
    mcp = body["capabilities"]["mcp_tools"]
    assert "agent_artifact_reader" in mcp["allowed_tools"]
    assert "pubmed_metadata_search" in mcp["blocked_tools"]
    assert body["capabilities"]["remote_compute"]["status"] == "disabled"


def test_capability_tool_check_blocks_unknown_and_external_tools(monkeypatch) -> None:
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(mcp_tools_enabled=False, connector_pubmed_enabled=True),
    )
    client = TestClient(app)

    external = client.post(
        "/api/capabilities/tool-check",
        json={"tool_id": "pubmed_metadata_search"},
    ).json()
    unknown = client.post(
        "/api/capabilities/tool-check",
        json={"tool_id": "made_up_tool"},
    ).json()
    local = client.post(
        "/api/capabilities/tool-check",
        json={"tool_id": "agent_artifact_reader"},
    ).json()

    assert external["allowed"] is False
    assert external["reason"] == "mcp_tools_enabled_false"
    assert unknown["allowed"] is False
    assert unknown["reason"] == "unknown_tool"
    assert local["allowed"] is True


def test_zotero_search_fails_closed_when_connector_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(connector_zotero_enabled=False),
    )

    response = TestClient(app).post(
        "/api/capabilities/zotero/search",
        json={"query": "sepsis", "limit": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["blocked"] is True
    assert body["status"]["reason"] == "connector_zotero_enabled_false"
    assert body["items"] == []


def test_zotero_connection_test_records_audit_event(
    tmp_path: Path, monkeypatch
) -> None:
    audit_path = tmp_path / "capability_tool_audit.jsonl"
    monkeypatch.setattr(capabilities, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(capabilities, "_AUDIT_PATH", audit_path)
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(
            connector_zotero_enabled=False,
            tool_audit_enabled=True,
        ),
    )

    response = TestClient(app).post("/api/capabilities/zotero/test", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["status"]["reason"] == "connector_zotero_enabled_false"
    events = capabilities.audit_events(limit=5)["events"]
    assert events[-1]["event_type"] == "zotero_connection_test"
    assert events[-1]["detail"]["available"] is False


def test_zotero_source_maps_item_into_idea_payload(
    tmp_path: Path, monkeypatch
) -> None:
    audit_path = tmp_path / "capability_tool_audit.jsonl"
    monkeypatch.setattr(capabilities, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(capabilities, "_AUDIT_PATH", audit_path)
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(
            connector_zotero_enabled=True,
            tool_audit_enabled=True,
        ),
    )
    monkeypatch.setattr(
        capabilities,
        "zotero_status",
        lambda settings=None: {
            "enabled": True,
            "available": True,
            "status": "available",
            "reason": "local_zotero_api_ready",
        },
    )

    response = TestClient(app).post(
        "/api/capabilities/zotero/source",
        json={
            "item": {
                "key": "ABC123",
                "title": "Early Vasopressors in Septic Shock",
                "journal": "Intensive Care Medicine",
                "year": "2026",
                "doi": "10.1000/example",
                "url": "https://example.org/paper",
                "abstract": "Early vasopressors may define a measurable ICU exposure.",
                "citation_key": "smith2026vasopressors",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["blocked"] is False
    assert body["item"]["journal"] == "Intensive Care Medicine"
    suggested = body["suggested_payload"]
    assert suggested["source_type"] == "zotero"
    assert suggested["source_origin"] == "zotero_desktop"
    assert suggested["source_origin_label"] == "Zotero Desktop"
    assert suggested["title"] == "Early Vasopressors in Septic Shock"
    assert suggested["topic"] == "Early Vasopressors in Septic Shock"
    assert suggested["doi"] == "10.1000/example"
    assert suggested["citation_key"] == "smith2026vasopressors"
    assert suggested["zotero_key"] == "ABC123"
    assert "Early vasopressors" in suggested["excerpt"]
    assert body["source_adapter"]["status"] == "literature_source_ready"
    assert body["source_adapter"]["source_origin"] == "zotero_desktop"
    assert body["source_adapter"]["display_status"] == (
        "Literature source ready / 文献来源已就绪"
    )
    assert body["privacy"]["full_text_stored"] is False
    events = capabilities.audit_events(limit=5)["events"]
    assert events[-1]["event_type"] == "zotero_source_selected"


def test_zotero_paste_import_builds_source_without_connector(
    tmp_path: Path, monkeypatch
) -> None:
    audit_path = tmp_path / "capability_tool_audit.jsonl"
    monkeypatch.setattr(capabilities, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(capabilities, "_AUDIT_PATH", audit_path)
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(
            connector_zotero_enabled=False,
            tool_audit_enabled=True,
        ),
    )

    response = TestClient(app).post(
        "/api/capabilities/zotero/import",
        json={
            "text": """@article{smith2026shock,
              title={Early Vasopressors in Septic Shock},
              journal={Intensive Care Medicine},
              year={2026},
              doi={10.1000/example},
              abstract={Early vasopressors may define a measurable ICU exposure.}
            }"""
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["blocked"] is False
    assert body["source_adapter"]["status"] == "literature_source_ready"
    assert body["source_adapter"]["source_origin"] == "pasted_literature"
    assert body["source_adapter"]["display_status"] == (
        "Literature source ready / 文献来源已就绪"
    )
    assert "no Zotero setup is required" in body["source_adapter"]["display_reason"]
    suggested = body["suggested_payload"]
    assert suggested["source_type"] == "zotero"
    assert suggested["source_origin"] == "pasted_literature"
    assert suggested["source_origin_label"] == "Pasted literature metadata"
    assert suggested["title"] == "Early Vasopressors in Septic Shock"
    assert suggested["journal"] == "Intensive Care Medicine"
    assert suggested["year"] == "2026"
    assert suggested["doi"] == "10.1000/example"
    assert suggested["citation_key"] == "smith2026shock"
    assert suggested["zotero_key"] == "smith2026shock"
    assert "Early vasopressors" in suggested["excerpt"]
    assert body["privacy"]["full_text_stored"] is False
    events = capabilities.audit_events(limit=5)["events"]
    assert events[-1]["event_type"] == "zotero_paste_import"


def test_tool_audit_ledger_respects_setting(
    tmp_path: Path, monkeypatch
) -> None:
    audit_path = tmp_path / "capability_tool_audit.jsonl"
    monkeypatch.setattr(capabilities, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(capabilities, "_AUDIT_PATH", audit_path)

    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(tool_audit_enabled=True),
    )
    recorded = capabilities.record_tool_event("unit_test_event", {"value": 1})
    assert recorded["recorded"] is True
    assert audit_path.exists()
    assert capabilities.audit_events(limit=10)["count"] == 1

    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(tool_audit_enabled=False),
    )
    skipped = capabilities.record_tool_event("unit_test_event", {"value": 2})
    assert skipped["recorded"] is False
    assert capabilities.audit_events(limit=10)["count"] == 1


def test_remote_compute_policy_blocks_non_local_targets(monkeypatch) -> None:
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(remote_compute_enabled=False),
    )

    blocked = capabilities.validate_compute_target({"compute_target": "hpc"})
    local = capabilities.validate_compute_target({"compute_target": "local"})

    assert blocked["ok"] is False
    assert blocked["error"] == "remote_compute_disabled"
    assert local["ok"] is True
    assert local["compute_target"] == "local"
