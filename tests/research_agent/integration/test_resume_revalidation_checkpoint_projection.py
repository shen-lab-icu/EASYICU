from __future__ import annotations

from easyicu.research_agent.execution.resume_revalidation import (
    _inline_history_checkpoint_payload,
)


def test_hydrated_external_history_projects_to_inline_only_checkpoint() -> None:
    source = {
        "step_attempt_history": [{"step_id": "01_model", "status": "ok"}],
        "step_attempt_history_ref": {
            "schema_version": "easyicu.step_attempt_history_ref/1",
            "relative_path": "evidence/history.jsonl",
        },
        "per_step_records": [{"step_id": "01_model", "status": "ok"}],
    }

    projected = _inline_history_checkpoint_payload(source)

    assert "step_attempt_history_ref" not in projected
    assert projected["step_attempt_history"] == source["step_attempt_history"]
    assert "step_attempt_history_ref" in source
