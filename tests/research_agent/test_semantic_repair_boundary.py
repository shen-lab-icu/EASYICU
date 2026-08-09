"""Semantic source mutations become attributable replanning findings."""

from __future__ import annotations

from threading import Lock

from easyicu.research_agent.repairs.semantic_boundary import (
    SemanticRepairEscalation,
    SemanticRepairRecorder,
)


def test_semantic_repair_recorder_is_typed_and_deduplicated() -> None:
    step_record = {}
    findings = []
    recorder = SemanticRepairRecorder(
        step_record=step_record,
        findings=findings,
        lock=Lock(),
        step_id="04_primary",
        attempt_id="run:04_primary:1",
    )
    escalation = SemanticRepairEscalation(
        repair_id="statsmodels_rank_safe_fit_v1",
        source="deterministic_runner_repair",
    )

    recorder(escalation)
    recorder(escalation)

    assert step_record["semantic_repair_escalations"] == [
        {
            "issue_code": "scientific_design_change_requires_replan",
            "repair_id": "statsmodels_rank_safe_fit_v1",
            "source": "deterministic_runner_repair",
            "action": "replan_or_human_review",
            "step_id": "04_primary",
            "attempt_id": "run:04_primary:1",
        }
    ]
    assert len(findings) == 1
    assert findings[0].validator == "deterministic_repair_boundary"
    assert findings[0].severity == "warning"
