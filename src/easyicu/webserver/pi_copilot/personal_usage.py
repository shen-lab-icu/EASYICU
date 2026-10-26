"""Usage readers reuse project/session ownership; they grant no execution."""
from __future__ import annotations

from typing import TYPE_CHECKING

from easyicu.webserver import agent_runs

if TYPE_CHECKING:
    from .service import PiCopilotService


def run_usage(service: PiCopilotService, *, project_id: str, run_id: str) -> dict:
    row = service._research_run_row(project_id, run_id)
    usage = agent_runs.read_run_usage(str(row.get("project_dir") or ""))
    return {"run_id": run_id, "usage": usage}


def conversation_usage(service: PiCopilotService, *, project_id: str, session_id: str) -> dict:
    record = service._scoped_record(session_id, project_id=project_id)
    state = service._ensure_open(record, transcript_limit=1)
    usage = state.get("shell_usage", {}).get("usage")
    return {"session_id": session_id, "scope": "conversation_model_calls", "usage": usage or {
        "accounting_complete": False, "reason_code": "conversation_usage_history_unavailable",
        "calls": None, "estimated_cost_usd": None,
    }}
