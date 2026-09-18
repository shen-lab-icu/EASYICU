"""Typed tool-error contracts for Pi Copilot."""

import pytest

from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import (
    PiCopilotError,
    PiSessionRecord,
    ToolExecutionContext,
)


def test_unknown_tool_arguments_and_missing_plan_keep_owner_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ToolExecutionContext(session=PiSessionRecord(session_id="pi-test"))
    with pytest.raises(PiCopilotError) as unknown:
        tool_module.execute_tool("easyicu_inspect_context", {"raw": True}, context)
    assert unknown.value.code == "pi_tool_unknown_arguments"

    monkeypatch.setattr(
        tool_module.agent_runs, "list_run_history", lambda **kwargs: {"runs": []}
    )
    missing = tool_module.execute_tool(
        "easyicu_inspect_step",
        {"step_id": "analysis"},
        context,
    )
    assert missing["code"] == "easyicu_plan_not_found"
