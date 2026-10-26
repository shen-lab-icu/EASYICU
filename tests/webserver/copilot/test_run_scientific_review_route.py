"""A conversation reads a run's scientific review through its project binding."""
import pytest
from fastapi import HTTPException

from easyicu.webserver.pi_copilot import run_scientific_review as owner
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.routes import pi_run_scientific_review as route


class Service:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def _research_run_row(self, project_id, run_id):
        self.calls.append((project_id, run_id))
        if (project_id, run_id) not in self.rows:
            raise PiCopilotError("pi_research_run_not_found", "missing", status_code=404)
        return self.rows[(project_id, run_id)]


@pytest.fixture
def reads(monkeypatch):
    seen = []

    def read(project_dir, *, offset, limit):
        seen.append((project_dir, offset, limit))
        if project_dir.endswith("blocked"):
            return {"ok": False, "error": "scientific_review_privacy_blocked"}
        return {"ok": True, "review": {"run_id": "r1", "analysis_only": True}}

    monkeypatch.setattr("easyicu.webserver.scientific_review.read_scientific_review", read)
    return seen


def test_the_run_is_resolved_by_the_project_binding_not_by_a_client_path(reads):
    service = Service({("p1", "r1"): {"project_dir": "/home/runs/r1"}})
    result = owner.run_scientific_review(service, project_id="p1", run_id="r1", offset=50, limit=10)
    assert result == {"run_id": "r1", "review": {"run_id": "r1", "analysis_only": True}}
    assert service.calls == [("p1", "r1")]
    assert reads == [("/home/runs/r1", 50, 10)]


def test_a_run_outside_the_project_or_a_blocked_review_is_refused_by_code(reads, monkeypatch):
    service = Service({("p1", "r9"): {"project_dir": "/home/runs/blocked"}})
    monkeypatch.setattr(route, "get_pi_copilot_service", lambda: service)
    with pytest.raises(HTTPException) as missing:
        route.get_run_scientific_review("p2", "r1", 0, 50)
    assert (missing.value.status_code, missing.value.detail) == (404, {"error": "pi_research_run_not_found"})
    with pytest.raises(HTTPException) as blocked:
        route.get_run_scientific_review("p1", "r9", 0, 50)
    assert (blocked.value.status_code, blocked.value.detail) == (409, {"error": "scientific_review_privacy_blocked"})
    assert reads == [("/home/runs/blocked", 0, 50)]
