"""Transport contracts for the lead's Mac; ASGI stalls in the server sandbox."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from easyicu.webserver import sources
from easyicu.webserver.routes.data_map import router


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(sources, "_read_raw", lambda: {})
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as instance:
        yield instance


def test_filtered_metadata_serializes_as_json(client):
    response = client.get("/api/data-map", params={"database": "MIMIC-IV", "concept": "hr", "family": "survival"})
    assert response.status_code == 200
    result = response.json()
    assert result["schema_version"] == "easyicu.data-map/1"
    assert [row["database"] for row in result["databases"]] == ["miiv"]
    row, = result["databases"]
    assert row["real_data_acceptance"] == "not_accepted"
    assert "concepts" not in row["capabilities"]
    assert len(row["concepts"]) == len(row["studies"]) == 1


def test_unknown_database_has_a_stable_error(client):
    response = client.get("/api/data-map", params={"database": "synthetic_unknown"})
    assert response.status_code == 400
    assert response.json() == {"detail": {"error": "data_map_unknown_database"}}


@pytest.mark.parametrize("params", [
    {"concept_limit": 101}, {"source_limit": 0}, {"source_offset": -1}, {"database": "x" * 161},
])
def test_transport_enforces_bounds(client, params):
    assert client.get("/api/data-map", params=params).status_code == 422


def test_data_map_rejects_writes(client):
    assert client.post("/api/data-map", json={}).status_code == 405
