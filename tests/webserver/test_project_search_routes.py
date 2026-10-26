"""HTTP contract checks: run on a host where TestClient's sockets are available."""

from fastapi.testclient import TestClient

from easyicu.webserver.app import app
from tests.support.project_search import corpus as corpus


def test_search_http_filters_and_targets(corpus):
    client = TestClient(app)
    response = client.get("/api/search", params={"q": "improved", "scope": "tasks", "project": "orchard"})
    assert response.status_code == 200
    assert response.json()["hits"][0]["open"]["message_id"] == "a1"
    assert client.get("/api/search", params={"q": "improved", "project": "meadow"}).json()["hits"] == []


def test_search_http_rejects_invalid_bounds_and_scopes(corpus):
    client = TestClient(app)
    for params in ({"q": "x" * 201}, {"scope": "invalid"}, {"limit": 51}, {"limit": 0}):
        assert client.get("/api/search", params=params).status_code == 422
    assert client.get("/api/search").json()["hits"] == []
