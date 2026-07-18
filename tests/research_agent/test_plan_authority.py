"""Contracts for the provider-free PlanAuthority candidate boundary."""

from __future__ import annotations

import ast
import inspect

import pytest

from easyicu.research_agent import pipeline_execute
from easyicu.research_agent.authority import plan_authority
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ResearchContext,
)


def _plan(*, intent: str = "Produce the locked result.") -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Test candidate plan authority.",
        steps=[
            AnalysisStep(
                step_id="01_completed",
                intent=intent,
                method="descriptive",
                expected_outputs=["table:locked_result"],
            ),
            AnalysisStep(
                step_id="02_future",
                intent="Run future analysis.",
                method="descriptive",
                expected_outputs=["table:future_result"],
            ),
        ],
    )


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Test candidate plan authority.",
        cohort=CohortDescriptor(
            cohort_name="test",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
    )


def _completed_record(plan: AnalysisPlan, *, status: str = "ok") -> dict:
    return {
        "step_id": "01_completed",
        "status": status,
        "analysis_request": {
            "step": plan.steps[0].model_dump(mode="json"),
        },
    }


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _identity_transforms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        plan_authority,
        "_preserve_primary_estimand_step_after_replan",
        lambda *, current, revised: (revised, []),
    )
    monkeypatch.setattr(
        plan_authority,
        "_preserve_figure_steps_after_replan",
        lambda *, current, revised: (revised, []),
    )
    monkeypatch.setattr(
        plan_authority,
        "_augment_report_typed_product_inputs",
        lambda *, plan: (plan, []),
    )
    monkeypatch.setattr(
        plan_authority,
        "_project_locked_robustness_specs_after_replan",
        lambda *, revised_plan, locked_specs: (revised_plan, None),
    )
    monkeypatch.setattr(
        plan_authority,
        "augment_trajectory_plan_products",
        lambda *, plan, context: (plan, []),
    )
    monkeypatch.setattr(
        plan_authority,
        "_augment_measurement_companion_inputs",
        lambda *, plan, context: (plan, []),
    )


def test_pipeline_execute_reexports_plan_authority_objects_with_identity() -> None:
    for name in plan_authority.__all__:
        assert getattr(pipeline_execute, name) is getattr(plan_authority, name)


def test_plan_authority_has_no_provider_registration_or_cohort_mutation() -> None:
    tree = ast.parse(inspect.getsource(plan_authority))
    imported_modules = {
        node.module or "" for node in tree.body if isinstance(node, ast.ImportFrom)
    }
    assert not any(
        module.endswith(("pipeline", "pipeline_execute", "agents", "evidence"))
        for module in imported_modules
    )
    normalizer = ast.parse(inspect.getsource(plan_authority.normalize_replan_candidate))
    identifiers = {
        node.id for node in ast.walk(normalizer) if isinstance(node, ast.Name)
    }
    attributes = {
        node.attr for node in ast.walk(normalizer) if isinstance(node, ast.Attribute)
    }
    assert identifiers.isdisjoint(
        {
            "ReplannerAgent",
            "EvidenceStore",
            "_register_plan_revision",
            "_resolve_cohort_definition",
            "_replan_state",
            "provider_budget",
        }
    )
    assert attributes.isdisjoint({"write_text", "register_file", "register"})


def test_candidate_transform_order_keeps_second_snapshot_restore() -> None:
    tree = ast.parse(inspect.getsource(plan_authority.normalize_replan_candidate))
    calls = sorted(
        (
            (node.lineno, _call_name(node))
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _call_name(node) is not None
        ),
        key=lambda item: item[0],
    )
    relevant = [
        name
        for _, name in calls
        if name
        in {
            "_preserve_completed_step_snapshots_after_replan",
            "_preserve_primary_estimand_step_after_replan",
            "_preserve_figure_steps_after_replan",
            "_augment_report_typed_product_inputs",
            "_cap_plan_preserving_figure_steps",
            "_project_locked_robustness_specs_after_replan",
            "augment_trajectory_plan_products",
            "_augment_measurement_companion_inputs",
        }
    ]
    assert relevant == [
        "_preserve_completed_step_snapshots_after_replan",
        "_preserve_primary_estimand_step_after_replan",
        "_preserve_figure_steps_after_replan",
        "_augment_report_typed_product_inputs",
        "_cap_plan_preserving_figure_steps",
        "_project_locked_robustness_specs_after_replan",
        "augment_trajectory_plan_products",
        "_augment_measurement_companion_inputs",
        "_preserve_completed_step_snapshots_after_replan",
    ]


def test_second_snapshot_restore_repairs_intermediate_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _plan()
    _identity_transforms(monkeypatch)

    def mutate_completed(*, plan: AnalysisPlan, context: ResearchContext):
        changed = plan.steps[0].model_copy(
            update={"intent": "Mutated after first guard."}
        )
        return plan.model_copy(update={"steps": [changed, *plan.steps[1:]]}), []

    monkeypatch.setattr(
        plan_authority,
        "_augment_measurement_companion_inputs",
        mutate_completed,
    )
    result = plan_authority.normalize_replan_candidate(
        current_plan=current,
        candidate_plan=current.model_copy(update={"revision": 2}),
        completed_records=[_completed_record(current)],
        context=_context(),
        max_total_steps=0,
        locked_robustness_specs=[],
    )

    assert result.plan.steps[0] == current.steps[0]
    assert any(
        finding.detail.get("reason") == "completed_step_snapshot_immutable"
        for finding in result.findings
    )


def test_latest_failed_checkpoint_does_not_freeze_older_success_snapshot() -> None:
    current = _plan()
    candidate = _plan(intent="New future request.").model_copy(update={"revision": 2})
    preserved, findings = (
        plan_authority._preserve_completed_step_snapshots_after_replan(
            current_plan=current,
            revised_plan=candidate,
            completed_records=[
                _completed_record(current),
                _completed_record(current, status="contract_failed"),
            ],
        )
    )

    assert preserved.steps[0].intent == "New future request."
    assert findings == []


def test_normalizer_passes_only_current_successful_ids_to_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _plan()
    _identity_transforms(monkeypatch)
    observed: dict[str, object] = {}

    def capture_cap(*, plan: AnalysisPlan, cap: int, protected_step_ids: list[str]):
        observed.update(cap=cap, protected=list(protected_step_ids))
        return plan, []

    monkeypatch.setattr(
        plan_authority,
        "_cap_plan_preserving_figure_steps",
        capture_cap,
    )
    plan_authority.normalize_replan_candidate(
        current_plan=current,
        candidate_plan=current.model_copy(update={"revision": 2}),
        completed_records=[
            _completed_record(current),
            {
                "step_id": "02_future",
                "status": "contract_failed",
            },
        ],
        context=_context(),
        max_total_steps=9,
        locked_robustness_specs=[],
    )

    assert observed == {"cap": 9, "protected": ["01_completed"]}


def test_normalizer_does_not_mutate_input_plans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _plan()
    candidate = current.model_copy(update={"revision": 2})
    current_before = current.model_dump(mode="json")
    candidate_before = candidate.model_dump(mode="json")
    _identity_transforms(monkeypatch)

    result = plan_authority.normalize_replan_candidate(
        current_plan=current,
        candidate_plan=candidate,
        completed_records=[],
        context=_context(),
        max_total_steps=0,
        locked_robustness_specs=[],
    )

    assert current.model_dump(mode="json") == current_before
    assert candidate.model_dump(mode="json") == candidate_before
    assert result.plan.revision == 2
    assert result.substantive is False


def test_result_is_typed_and_immutable(monkeypatch: pytest.MonkeyPatch) -> None:
    current = _plan()
    _identity_transforms(monkeypatch)
    result = plan_authority.normalize_replan_candidate(
        current_plan=current,
        candidate_plan=current,
        completed_records=[],
        context=_context(),
        max_total_steps=0,
        locked_robustness_specs=[],
    )

    assert isinstance(result, plan_authority.NormalizedPlanCandidate)
    assert isinstance(result.findings, tuple)
    with pytest.raises((AttributeError, TypeError)):
        result.substantive = True  # type: ignore[misc]
