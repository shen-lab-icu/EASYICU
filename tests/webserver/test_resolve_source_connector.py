"""Every URL resolver entry honors the shared network switch."""

from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from easyicu.webserver import settings, capabilities
from easyicu.webserver.app import app
from easyicu.webserver.ideas import mining


@pytest.mark.parametrize("entry", ["route", "backend"])
@pytest.mark.parametrize("enabled", [False, True])
def test_resolve_source_checks_shared_connector(monkeypatch, entry, enabled):
    events = []
    monkeypatch.setattr(
        settings, "load_settings", lambda: {"connector_pubmed_enabled": enabled}
    )
    monkeypatch.setattr(capabilities, "record_tool_event", lambda *a: events.append(a))
    fetch = Mock(
        return_value={
            "status": "metadata_fetched",
            "network_calls": 1,
            "title": "Synthetic",
        }
    )
    monkeypatch.setattr(mining, "_fetch_url_metadata", fetch)
    body = {
        "source_type": "url",
        "url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
        "allow_network": True,
    }
    if entry == "route":
        response = TestClient(app).post("/api/ideas/resolve-source", json=body)
        assert response.status_code == 200
        payload = response.json()
    else:
        payload = mining.resolve_source(body)
    assert fetch.call_count == int(enabled)
    if not enabled:
        assert payload["connector_disabled_reason"] == "connector_pubmed_enabled_false"
        assert payload["source_adapter"]["network_calls"] == 0
        assert events and all(e[0] == "pubmed_connector_blocked" for e in events)


def test_disabled_connector_preserves_local_source_metadata(monkeypatch):
    monkeypatch.setattr(
        settings, "load_settings", lambda: {"connector_pubmed_enabled": False}
    )
    monkeypatch.setattr(capabilities, "record_tool_event", lambda *a: None)
    payload = mining.resolve_source({"source_type": "manual", "title": "Local paper"})
    assert payload["source_adapter"]["status"] == "metadata_ready"
    assert payload["privacy"]["network_calls"] == 0
