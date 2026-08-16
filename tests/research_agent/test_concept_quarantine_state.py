"""Structural contracts for step-local concept quarantine state."""

from __future__ import annotations

import ast
import inspect

from easyicu.research_agent.execution import phase as pipeline_execute
from easyicu.research_agent.execution import phase_support as pipeline_execute_support
from easyicu.research_agent.execution import concept_audit as concept_audit_execution
from easyicu.research_agent.execution import concept_repair as concept_repair_execution
from easyicu.research_agent.execution.concept_reaudit import (
    DETERMINISTIC_CONCEPT_REAUDIT_BUDGET_ISSUE_CODE,
)
from easyicu.research_agent.schema import ValidationFinding


def _execute_one_step_node() -> ast.Module:
    source = (
        inspect.getsource(pipeline_execute._execute_step)
        + "\n"
        + inspect.getsource(pipeline_execute._step_prepare_execution_authority)
        + "\n"
        + inspect.getsource(pipeline_execute._step_run_concept_repair_phase)
    )
    return ast.parse(source)


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


def _provider_failure(*, step_id: str = "05_analysis") -> ValidationFinding:
    return ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="provider unavailable",
        detail={
            "issue_code": "llm_concept_audit_provider_failure",
            "step_id": step_id,
        },
    )


def _proven_reaudit_budget_failure(
    *, step_id: str = "05_analysis"
) -> ValidationFinding:
    return ValidationFinding(
        validator="provider_call_budget",
        severity="error",
        message="reserved final audit budget exhausted",
        detail={
            "issue_code": DETERMINISTIC_CONCEPT_REAUDIT_BUDGET_ISSUE_CODE,
            "step_id": step_id,
            "category": "concept_audit",
            "used": 9,
            "limit": 9,
        },
    )


def test_final_audit_continuation_requires_exact_transport_failure_quarantine() -> None:
    provider_failure = _provider_failure()

    assert concept_audit_execution._final_audit_continuation_allowed(
        reservation_status="attempted_incomplete",
        quarantine_findings=[provider_failure],
        step_id="05_analysis",
    )
    assert not concept_audit_execution._final_audit_continuation_allowed(
        reservation_status="completed",
        quarantine_findings=[provider_failure],
        step_id="05_analysis",
    )
    assert not concept_audit_execution._final_audit_continuation_allowed(
        reservation_status="attempted_incomplete",
        quarantine_findings=[_provider_failure(step_id="other_step")],
        step_id="05_analysis",
    )


def test_final_audit_continuation_refuses_mixed_or_invalid_response_findings() -> None:
    semantic = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="scientific issue",
        detail={"issue_code": "scientific_semantics_violation"},
    )
    invalid_response = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="invalid response",
        detail={
            "issue_code": "llm_concept_audit_response_invalid",
            "step_id": "05_analysis",
        },
    )

    assert not concept_audit_execution._final_audit_continuation_allowed(
        reservation_status="attempted_incomplete",
        quarantine_findings=[_provider_failure(), semantic],
        step_id="05_analysis",
    )
    assert not concept_audit_execution._final_audit_continuation_allowed(
        reservation_status="attempted_incomplete",
        quarantine_findings=[invalid_response],
        step_id="05_analysis",
    )


def test_provider_failure_is_deferred_only_before_reserved_final_audit() -> None:
    failure = _provider_failure()

    assert concept_audit_execution._defer_provider_failure_until_final_audit(
        include_llm=False,
        reserved_final_category="concept_audit",
        quarantine_findings=[failure],
        step_id="05_analysis",
    )
    assert not concept_audit_execution._defer_provider_failure_until_final_audit(
        include_llm=True,
        reserved_final_category="concept_audit",
        quarantine_findings=[failure],
        step_id="05_analysis",
    )
    assert not concept_audit_execution._defer_provider_failure_until_final_audit(
        include_llm=False,
        reserved_final_category=None,
        quarantine_findings=[failure],
        step_id="05_analysis",
    )


def test_proven_deterministic_reaudit_budget_failure_is_deferred_and_retired() -> None:
    failure = _proven_reaudit_budget_failure()

    assert concept_audit_execution._defer_provider_failure_until_final_audit(
        include_llm=False,
        reserved_final_category="concept_audit",
        quarantine_findings=[failure],
        step_id="05_analysis",
    )

    quarantine = concept_audit_execution.ConceptQuarantineState()
    quarantine.draft_active = True
    quarantine.pending_errors = [failure]
    assert concept_audit_execution._retire_completed_provider_failure_continuation(
        quarantine,
        step_id="05_analysis",
        fresh_findings=[],
    )
    assert quarantine.pending_errors == []
    assert quarantine.draft_active is False


def test_unmarked_provider_budget_failure_remains_blocking() -> None:
    failure = _proven_reaudit_budget_failure().model_copy(
        update={
            "detail": {
                "step_id": "05_analysis",
                "category": "concept_audit",
                "used": 9,
                "limit": 9,
            }
        }
    )

    assert not concept_audit_execution._defer_provider_failure_until_final_audit(
        include_llm=False,
        reserved_final_category="concept_audit",
        quarantine_findings=[failure],
        step_id="05_analysis",
    )


def test_successful_final_audit_retires_only_prior_provider_failure() -> None:
    quarantine = concept_audit_execution.ConceptQuarantineState()
    quarantine.draft_active = True
    quarantine.pending_errors = [_provider_failure()]
    warning = ValidationFinding(
        validator="llm_concept_auditor",
        severity="warning",
        message="nonblocking note",
    )

    assert concept_audit_execution._retire_completed_provider_failure_continuation(
        quarantine,
        step_id="05_analysis",
        fresh_findings=[warning],
    )
    assert quarantine.pending_errors == []
    assert quarantine.draft_active is False

    quarantine.pending_errors = [_provider_failure()]
    assert not concept_audit_execution._retire_completed_provider_failure_continuation(
        quarantine,
        step_id="05_analysis",
        fresh_findings=[_provider_failure()],
    )
    assert quarantine.pending_errors


def test_execute_phase_keeps_quarantine_lifecycle_on_one_state_object() -> None:
    execute_one_step = _execute_one_step_node()
    repair_loop = ast.parse(
        inspect.getsource(concept_repair_execution.run_concept_repair_loop)
    )
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
        for tree, owner_names in (
            (execute_one_step, {"quarantine_state"}),
            (repair_loop, {"quarantine"}),
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in owner_names
    }
    assert {
        "resumed_draft_used",
        "repair_materially_changed",
        "repair_succeeded",
        "superseded_by_fallback",
    } <= quarantine_attributes

    use_draft = next(
        node
        for node in ast.walk(
            ast.parse(inspect.getsource(pipeline_execute_support))
        )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_step_use_quarantined_draft"
    )
    assert not any(isinstance(node, ast.Nonlocal) for node in ast.walk(use_draft))


def test_execute_phase_defers_only_host_authorized_provider_failure_quarantine() -> (
    None
):
    execute_one_step = _execute_one_step_node()
    guarded_blocks = [
        node
        for node in ast.walk(execute_one_step)
        if isinstance(node, ast.If)
        and "quarantine_state.draft_active" in ast.unparse(node.test)
        and "quarantine_state.repair_succeeded" in ast.unparse(node.test)
    ]

    assert len(guarded_blocks) == 1
    assert "provider_failure_deferred" in ast.unparse(guarded_blocks[0].test)
