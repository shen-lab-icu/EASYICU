"""Body-integer validation for list/search routes.

These routes used to run ``int(body.get(key) or N)`` directly, so a non-numeric
client value (``"abc"``) escaped FastAPI validation and raised ``ValueError``
inside the route handler, producing an HTTP 500. A malformed request body is a
client error and must answer 400 with the field named.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver.app import app

BAD_LIMIT_ENDPOINTS = [
    ("/api/ideas/history", {}),
    ("/api/ideas/agent-projects", {}),
    ("/api/ideas/discover", {"topic": "sepsis"}),
    ("/api/capabilities/audit-events", {}),
    ("/api/capabilities/zotero/search", {"query": "sepsis"}),
    ("/api/guided/drafts/list", {}),
    ("/api/guided/sessions/list", {}),
    ("/api/copilot/sessions/list", {}),
    ("/api/page-guide/sessions/list", {}),
    ("/api/agent-runs/history", {}),
]


@pytest.mark.parametrize("path,extra", BAD_LIMIT_ENDPOINTS)
def test_non_numeric_limit_answers_400_not_500(path: str, extra: dict) -> None:
    client = TestClient(app)
    payload = {**extra, "limit": "abc"}
    response = client.post(path, json=payload)
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail["error"] == "invalid_integer"
    assert detail["field"] == "limit"


@pytest.mark.parametrize("path,extra", BAD_LIMIT_ENDPOINTS)
def test_out_of_range_limit_answers_400(path: str, extra: dict) -> None:
    client = TestClient(app)
    response = client.post(path, json={**extra, "limit": 10_000})
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail["error"] == "integer_out_of_range"
    assert detail["field"] == "limit"
