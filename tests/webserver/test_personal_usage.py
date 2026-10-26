"""Small synthetic ledger checks, with no provider or ASGI transport."""
import json

import pytest

from easyicu.research_agent.authority.provider_hard_stop import PROVIDER_HARD_STOP_SCHEMA
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.webserver.research_run_usage import research_run_usage


def write_ledger(root, calls):
    path = root / ".runtime/provider_hard_stop_ledger.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": PROVIDER_HARD_STOP_SCHEMA, "tasks": [{"status": "running", "calls": calls}]}
    payload["sha256"] = canonical_sha256(payload, trailing_newline=True)
    path.write_text(json.dumps(payload))
    return path


def call(state, tokens=100, cost=0.01):
    return {"state": state, "role": "planner", "accounted_tokens": tokens,
            "accounted_estimated_cost_usd": cost,
            "reported_total_tokens": 10 if state == "completed" else None,
            "reported_prompt_tokens": 8 if state == "completed" else None,
            "reported_completion_tokens": 2 if state == "completed" else None}


def test_failed_and_inflight_reservations_are_separate_from_completed_usage(tmp_path):
    path = write_ledger(tmp_path, [call("completed"), call("failed_usage_unknown", 200, 0.02), call("in_progress", 300, 0.03)])
    before = path.read_bytes()
    usage = research_run_usage(tmp_path)
    assert usage["failed_calls"] == 1
    assert usage["in_flight_calls"] == 1
    assert usage["by_state"]["completed"]["provider_reported_prompt_tokens"] == 8
    assert usage["by_state"]["failed"]["accounted_tokens"] == 200
    assert usage["by_state"]["in_flight"]["accounted_tokens"] == 300
    assert sum(row["calls"] for row in usage["by_state"].values()) == usage["calls"] == 3
    assert sum(row["estimated_cost_usd"] for row in usage["by_state"].values()) == pytest.approx(usage["estimated_cost_usd"])
    assert usage["estimated_cost_usd"] == pytest.approx(0.06)
    assert path.read_bytes() == before


@pytest.mark.parametrize("bound", ["MAX_USAGE_LEDGERS", "MAX_USAGE_BYTES", "MAX_USAGE_CALLS"])
def test_usage_read_budget_refuses_partial_totals(tmp_path, monkeypatch, bound):
    from easyicu.webserver import research_run_usage as owner
    write_ledger(tmp_path, [call("completed"), call("in_progress")])
    if bound == "MAX_USAGE_LEDGERS":
        (tmp_path / ".runtime/provider_hard_stop_retry_a.json").write_bytes((tmp_path / ".runtime/provider_hard_stop_ledger.json").read_bytes())
    monkeypatch.setattr(owner, bound, 1)
    assert research_run_usage(tmp_path)["accounting_complete"] is False
    assert research_run_usage(tmp_path)["estimated_cost_usd"] is None


def test_unknown_prices_are_not_free_calls(tmp_path):
    write_ledger(tmp_path, [call("completed", cost=None)])
    usage = research_run_usage(tmp_path)
    assert usage["accounting_complete"] is True
    assert usage["estimated_cost_usd"] is None
    assert usage["by_state"]["completed"]["estimated_cost_usd"] is None


def test_run_usage_reuses_project_owner_before_any_ledger_read(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from easyicu.webserver.pi_copilot.personal_usage import run_usage
    from easyicu.webserver.pi_copilot.contracts import PiCopilotError
    from easyicu.webserver import agent_runs
    path = write_ledger(tmp_path, [call("in_progress")])
    calls = []
    def owned(project, run):
        calls.append((project, run))
        if project != "project-a" or run != "run-a":
            raise PiCopilotError("pi_research_run_not_found", "Unknown run", status_code=404)
        return {"project_dir": str(tmp_path)}
    service = SimpleNamespace(_research_run_row=owned)
    assert run_usage(service, project_id="project-a", run_id="run-a")["usage"]["in_flight_calls"] == 1
    assert path.exists()
    monkeypatch.setattr(agent_runs, "read_run_usage", lambda _: pytest.fail("Unowned ledger read"))
    with pytest.raises(PiCopilotError):
        run_usage(service, project_id="project-b", run_id="run-a")
    assert calls == [("project-a", "run-a"), ("project-b", "run-a")]
