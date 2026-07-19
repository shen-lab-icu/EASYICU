"""Structural contracts for step-local concept quarantine state."""

from __future__ import annotations

import ast
import inspect

from easyicu.research_agent.execution import phase as pipeline_execute
from easyicu.research_agent.execution import concept_audit as concept_audit_execution
from easyicu.research_agent.schema import ValidationFinding


def _execute_one_step_node() -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(inspect.getsource(pipeline_execute.run_execute_phase))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_execute_one_step"
    )


def test_quarantine_state_defaults_are_step_local() -> None:
    first = concept_audit_execution.ConceptQuarantineState()
    second = concept_audit_execution.ConceptQuarantineState()

    assert first.draft_active is False
    assert first.policy_superseded is False
    assert first.deterministic_revalidated is False
    assert first.pending_errors == []
    assert first.resumed_draft_used is False
    assert first.repair_materially_changed is False
    assert first.repair_succeeded is False
    assert first.superseded_by_fallback is False

    first.resumed_draft_used = True
    first.repair_materially_changed = True
    first.repair_succeeded = True
    first.superseded_by_fallback = True
    first.pending_errors.append(
        ValidationFinding(
            validator="test",
            severity="error",
            message="step-local sentinel",
        )
    )

    assert second.pending_errors == []
    assert second.resumed_draft_used is False
    assert second.repair_materially_changed is False
    assert second.repair_succeeded is False
    assert second.superseded_by_fallback is False


def test_mixed_quarantine_findings_retire_by_independent_exact_digest_proofs(
    monkeypatch,
) -> None:
    deterministic = ValidationFinding(
        validator="mechanical_code_preflight",
        severity="error",
        message="stale deterministic finding",
    )
    llm = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="stale policy finding",
    )
    reclassified = llm.model_copy(
        update={
            "severity": "warning",
            "detail": {"downgraded_reason": "current host policy proof"},
        }
    )

    def resolve_deterministic(**kwargs):
        assert kwargs["prior_errors"] == (deterministic,)
        return [{"validator": deterministic.validator, "proof": "current_gate"}]

    def resolve_policy(**kwargs):
        assert kwargs["prior_errors"] == (llm,)
        return [reclassified], [
            {
                "validator": llm.validator,
                "downgraded_reason": "current host policy proof",
            }
        ]

    monkeypatch.setattr(
        concept_audit_execution,
        "quarantined_deterministic_errors_resolved_by_current_gate",
        resolve_deterministic,
    )
    monkeypatch.setattr(
        concept_audit_execution,
        "quarantined_errors_superseded_by_current_policy",
        resolve_policy,
    )

    decision = concept_audit_execution._quarantine_retirement_decision(
        prior_errors=[deterministic, llm],
        current_findings=[],
        context=object(),
        script_text="value = 1\n",
        quarantined_script_sha256="digest",
    )

    assert decision.remaining_errors == ()
    assert decision.deterministic_provenance[0]["proof"] == "current_gate"
    assert decision.policy_reclassified_findings == (reclassified,)
    assert decision.policy_provenance[0]["downgraded_reason"]


def test_mixed_quarantine_retirement_keeps_every_unproved_subset(monkeypatch) -> None:
    deterministic = ValidationFinding(
        validator="mechanical_code_preflight",
        severity="error",
        message="still blocking deterministic finding",
    )
    llm = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="still blocking policy finding",
    )
    monkeypatch.setattr(
        concept_audit_execution,
        "quarantined_deterministic_errors_resolved_by_current_gate",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        concept_audit_execution,
        "quarantined_errors_superseded_by_current_policy",
        lambda **kwargs: None,
    )

    decision = concept_audit_execution._quarantine_retirement_decision(
        prior_errors=[deterministic, llm],
        current_findings=[],
        context=object(),
        script_text="value = 1\n",
        quarantined_script_sha256="digest",
    )

    assert decision.remaining_errors == (deterministic, llm)
    assert decision.deterministic_provenance == ()
    assert decision.policy_provenance == ()


def test_execute_phase_keeps_quarantine_lifecycle_on_one_state_object() -> None:
    execute_one_step = _execute_one_step_node()
    constructors = [
        node
        for node in ast.walk(execute_one_step)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ConceptQuarantineState"
    ]
    assert len(constructors) == 1

    name_ids = {
        node.id for node in ast.walk(execute_one_step) if isinstance(node, ast.Name)
    }
    assert name_ids.isdisjoint(
        {
            "resumed_quarantined_draft_used",
            "quarantined_repair_materially_changed",
            "quarantined_repair_succeeded",
            "quarantine_superseded_by_fallback",
        }
    )

    quarantine_attributes = {
        node.attr
        for node in ast.walk(execute_one_step)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "quarantine_state"
    }
    assert {
        "resumed_draft_used",
        "repair_materially_changed",
        "repair_succeeded",
        "superseded_by_fallback",
    } <= quarantine_attributes

    use_draft = next(
        node
        for node in ast.walk(execute_one_step)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_use_quarantined_draft"
    )
    assert not any(isinstance(node, ast.Nonlocal) for node in ast.walk(use_draft))
