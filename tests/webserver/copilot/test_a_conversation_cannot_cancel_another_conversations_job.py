"""A Copilot conversation can only inspect, reattach to or cancel its own job.

11-03 external review: JobManager is process-global and ``easyicu_cancel``
took any model-supplied ``job_id``. One conversation could therefore cancel
another conversation's extraction or analysis, consuming its own cancel
grant. Only the session's bound job and its bound study's active job belong
to it.
"""

from __future__ import annotations

import pytest

from easyicu.webserver import jobs
from easyicu.webserver.jobs import Job
from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    PiSessionRecord,
    ToolExecutionContext,
)


class _Manager:
    def __init__(self, *items: Job) -> None:
        self._jobs = {item.id: item for item in items}

    def get(self, job_id: str):
        return self._jobs.get(job_id)


@pytest.fixture
def two_jobs(monkeypatch):
    mine, theirs, study_job = Job("job-mine", "pi_copilot"), Job("job-theirs", "extract"), Job("job-study", "agent")
    monkeypatch.setattr(jobs, "MANAGER", _Manager(mine, theirs, study_job))
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: {"active_job_id": "job-study"})
    session = PiSessionRecord(session_id="pi-a", binding=AuthorityBinding(active_job_id="job-mine"))
    context = ToolExecutionContext(
        data_handling="zero_retention_declared", session=session,
        allowed_actions=frozenset({"cancel"}),
    )
    return context, mine, theirs, study_job


@pytest.mark.parametrize("tool", ["easyicu_cancel", "easyicu_inspect_run", "easyicu_resume"])
def test_another_conversations_job_is_refused(two_jobs, tool):
    context, _mine, theirs, _study_job = two_jobs
    result = tool_module.execute_tool(tool, {"job_id": "job-theirs"}, context)
    assert result["code"] == "easyicu_job_not_bound_to_session"
    assert result["status"] == "blocked"
    assert not theirs.cancel_requested


def test_a_refused_target_does_not_consume_the_cancel_grant(two_jobs, monkeypatch):
    context, mine, _theirs, _study_job = two_jobs
    consumed = []
    monkeypatch.setattr(tool_module, "_consume_action", lambda _ctx, action: consumed.append(action))
    tool_module.execute_tool("easyicu_cancel", {"job_id": "job-theirs"}, context)
    assert consumed == []
    result = tool_module.execute_tool("easyicu_cancel", {}, context)
    assert result["code"] == "easyicu_job_cancel_requested"
    assert consumed == ["cancel"] and mine.cancel_requested


def test_the_bound_studys_active_job_still_belongs_to_the_conversation(two_jobs, monkeypatch):
    context, _mine, theirs, study_job = two_jobs
    monkeypatch.setattr(tool_module, "_consume_action", lambda _ctx, _action: None)
    result = tool_module.execute_tool("easyicu_cancel", {"job_id": "job-study"}, context)
    assert result["code"] == "easyicu_job_cancel_requested"
    assert study_job.cancel_requested and not theirs.cancel_requested
