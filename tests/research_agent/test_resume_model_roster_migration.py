"""Fail-closed migration for legacy resume plans without typed model rosters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import easyicu.research_agent.pipeline as pipeline_module
from easyicu.research_agent.context import build_research_context
from easyicu.research_agent.evidence import EvidenceStore
from easyicu.research_agent.pipeline import (
    LegacyResumePlanMigrationError,
    _load_compatible_resume_plan,
    _migrate_legacy_resume_model_requirements,
)
from easyicu.research_agent.runtime_artifacts import verified_run_evidence_path
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    PlannedModelRequirement,
)


def _legacy_plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Estimate an adjusted ICU association.",
        analysis_type="cohort_definition_sensitivity",
        revision=1,
        rationale="Planner-owned scientific rationale.",
        steps=[
            AnalysisStep(
                step_id="01_completed_description",
                intent="Describe the fixed analytic cohort.",
                inputs=["baseline"],
                expected_outputs=["table:description"],
                method="descriptive_baseline",
                icu_rule_refs=["fixed_rule"],
            ),
            AnalysisStep(
                step_id="02_remaining_closed",
                intent="Fit the planner-selected adjusted association models.",
                inputs=["exposure", "endpoint"],
                expected_outputs=[
                    "table:adjusted_association_estimates",
                    "table:analytic_population",
                ],
                method="adjusted_association_models",
                icu_rule_refs=["fixed_model_rule"],
            ),
            AnalysisStep(
                step_id="03_method_only_not_closed",
                intent="A different product remains agent-owned.",
                expected_outputs=["table:unrelated_product"],
                method="adjusted_association_models",
            ),
            AnalysisStep(
                step_id="04_product_only_not_closed",
                intent="The closed product name alone is insufficient.",
                expected_outputs=["table:adjusted_association_estimates"],
                method="descriptive_baseline",
            ),
        ],
    )


def _planner_requirement() -> PlannedModelRequirement:
    return PlannedModelRequirement(
        requirement_id="planner_selected_binary_endpoint",
        outcome="agent_selected_endpoint",
        outcome_type="binary",
        method_family="statsmodels_glm_binomial",
        exposure_source="agent_selected_transform",
        analysis_role="secondary",
        analysis_set="complete_case",
        required_for_step_success=True,
    )


def _plan_with_target_roster(plan: AnalysisPlan) -> AnalysisPlan:
    steps = []
    for step in plan.steps:
        if step.step_id == "02_remaining_closed":
            step = step.model_copy(
                update={"model_requirements": [_planner_requirement()]}
            )
        steps.append(step)
    return plan.model_copy(update={"steps": steps, "revision": plan.revision + 1})


def _evidence_with_base_plan(run_dir: Path, plan: AnalysisPlan) -> EvidenceStore:
    plan_path = run_dir / "analysis_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    evidence = EvidenceStore(run_dir)
    evidence.register_file(
        kind="log",
        description="Legacy analysis plan.",
        source_path=plan_path,
        evidence_id="analysis_plan",
        producer="planner",
        generation_mode="llm",
    )
    return evidence


def test_resume_migration_uses_replanner_roster_and_registers_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _legacy_plan()
    evidence = _evidence_with_base_plan(tmp_path, plan)
    base_bytes = (tmp_path / "analysis_plan.json").read_bytes()
    resume_state = {
        "plan_path": "analysis_plan.json",
        "per_step_records": [
            {"step_id": "01_completed_description", "status": "ok"},
            {"step_id": "02_remaining_closed", "status": "ok"},
        ],
    }
    calls = []

    class StubReplanner:
        def __init__(self, llm) -> None:
            calls.append(("client", llm))

        def run(self, **kwargs):
            calls.append(("run", kwargs))
            assert [
                record["step_id"]
                for record in kwargs["completed_step_records"]
            ] == ["01_completed_description"]
            assert "02_remaining_closed" in kwargs["directive"]
            return _plan_with_target_roster(kwargs["current_plan"])

    monkeypatch.setattr(pipeline_module, "ReplannerAgent", StubReplanner)
    roles = []

    revised, revision_path, target_ids = _migrate_legacy_resume_model_requirements(
        plan=plan,
        context=object(),  # The stub stands in for the planner LLM boundary.
        run_dir=tmp_path,
        resume_state=resume_state,
        resume_from_step_id="02_remaining_closed",
        role_resolver=lambda role: roles.append(role) or "planner-client",
        evidence=evidence,
        prompt_version="test-pack",
        llm_signature="test-planner",
    )

    assert roles == ["planner"]
    assert len(calls) == 2
    assert target_ids == ("02_remaining_closed",)
    assert revision_path == tmp_path / "analysis_plan_revision_2.json"
    assert revision_path.exists()
    assert (tmp_path / "analysis_plan.json").read_bytes() == base_bytes
    assert revised.steps[1].model_requirements == [_planner_requirement()]
    assert not revised.steps[2].model_requirements
    assert not revised.steps[3].model_requirements

    revision_evidence = evidence.get("analysis_plan_revision_2")
    assert revision_evidence is not None
    assert revision_evidence.producer == "replanner"
    assert revision_evidence.generation_mode == "llm"
    assert revision_evidence.metadata["reason"] == (
        "legacy_missing_model_requirements"
    )
    assert revision_evidence.metadata["target_step_ids"] == [
        "02_remaining_closed"
    ]

    selected, selected_path = _load_compatible_resume_plan(
        run_dir=tmp_path,
        resume_state=resume_state,
    )
    assert selected_path == verified_run_evidence_path(
        tmp_path,
        revision_evidence,
    )
    assert selected == revised


def test_resume_loader_ignores_mutable_live_plan_and_unregistered_revision(
    tmp_path: Path,
) -> None:
    plan = _legacy_plan()
    evidence = _evidence_with_base_plan(tmp_path, plan)
    immutable_record = evidence.get("analysis_plan")
    assert immutable_record is not None

    live_drift = plan.model_copy(
        update={
            "steps": [
                plan.steps[0].model_copy(update={"intent": "Mutable live drift."}),
                *plan.steps[1:],
            ]
        }
    )
    (tmp_path / "analysis_plan.json").write_text(
        live_drift.model_dump_json(indent=2),
        encoding="utf-8",
    )
    unregistered_revision = live_drift.model_copy(update={"revision": 99})
    (tmp_path / "analysis_plan_revision_99.json").write_text(
        unregistered_revision.model_dump_json(indent=2),
        encoding="utf-8",
    )

    selected, selected_path = _load_compatible_resume_plan(
        run_dir=tmp_path,
        resume_state={
            "plan_path": "analysis_plan_revision_99.json",
            "per_step_records": [
                {"step_id": "01_completed_description", "status": "ok"},
            ],
        },
    )

    assert selected == plan
    assert selected_path == verified_run_evidence_path(
        tmp_path,
        immutable_record,
    )


def test_resume_migration_real_replanner_parser_preserves_plan_shape(
    tmp_path: Path,
) -> None:
    plan = _legacy_plan()
    evidence = _evidence_with_base_plan(tmp_path, plan)
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "exposure": [0.1, 0.4, 0.8, 1.2],
            "endpoint": [0, 1, 0, 1],
            "baseline": [50, 61, 72, 83],
        }
    ).to_parquet(cohort_path, index=False)
    context = build_research_context(
        research_question=plan.research_question,
        cohort=cohort_path,
        cohort_name="migration_parser_test",
        database="synthetic",
        target_outcome="endpoint",
        primary_exposure="exposure",
    )
    llm_calls = []

    class PlanJSONLLM:
        name = "plan-json-llm"

        def complete(self, messages, *, max_tokens=4096, temperature=0.1):
            llm_calls.append((messages, max_tokens, temperature))
            return _plan_with_target_roster(plan).model_dump_json(indent=2)

    revised, revision_path, target_ids = _migrate_legacy_resume_model_requirements(
        plan=plan,
        context=context,
        run_dir=tmp_path,
        resume_state={
            "per_step_records": [
                {"step_id": "01_completed_description", "status": "ok"},
            ]
        },
        resume_from_step_id=None,
        role_resolver=lambda role: PlanJSONLLM() if role == "planner" else None,
        evidence=evidence,
        prompt_version="test-pack",
        llm_signature="test-planner",
    )

    assert len(llm_calls) == 1
    assert revision_path == tmp_path / "analysis_plan_revision_2.json"
    assert target_ids == ("02_remaining_closed",)
    assert revised.analysis_type == plan.analysis_type
    assert revised.rationale == plan.rationale
    assert revised.cohort == plan.cohort
    assert revised.robustness_specs == plan.robustness_specs
    for original_step, revised_step in zip(plan.steps, revised.steps):
        if original_step.step_id == "02_remaining_closed":
            assert revised_step.model_requirements == [_planner_requirement()]
            revised_step = revised_step.model_copy(update={"model_requirements": []})
        assert revised_step == original_step


def test_resume_migration_does_not_touch_completed_or_nonclosed_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _legacy_plan()
    evidence = _evidence_with_base_plan(tmp_path, plan)

    class ForbiddenReplanner:
        def __init__(self, _llm) -> None:
            raise AssertionError("completed/non-closed steps must not invoke the LLM")

    monkeypatch.setattr(pipeline_module, "ReplannerAgent", ForbiddenReplanner)
    unchanged, revision_path, target_ids = _migrate_legacy_resume_model_requirements(
        plan=plan,
        context=object(),
        run_dir=tmp_path,
        resume_state={
            "per_step_records": [
                {"step_id": "02_remaining_closed", "status": "ok"},
            ]
        },
        resume_from_step_id=None,
        role_resolver=lambda _role: object(),
        evidence=evidence,
        prompt_version="test-pack",
        llm_signature="test-planner",
    )

    assert unchanged is plan
    assert revision_path is None
    assert target_ids == ()
    assert not list(tmp_path.glob("analysis_plan_revision_*.json"))


@pytest.mark.parametrize("mutation", ["target_intent", "completed_roster"])
def test_resume_migration_rejects_out_of_scope_replanner_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    plan = _legacy_plan()
    evidence = _evidence_with_base_plan(tmp_path, plan)

    class DriftingReplanner:
        def __init__(self, _llm) -> None:
            pass

        def run(self, **kwargs):
            revised = _plan_with_target_roster(kwargs["current_plan"])
            steps = list(revised.steps)
            if mutation == "target_intent":
                steps[1] = steps[1].model_copy(update={"intent": "Changed intent."})
            else:
                steps[0] = steps[0].model_copy(
                    update={"model_requirements": [_planner_requirement()]}
                )
            return revised.model_copy(update={"steps": steps})

    monkeypatch.setattr(pipeline_module, "ReplannerAgent", DriftingReplanner)
    with pytest.raises(LegacyResumePlanMigrationError):
        _migrate_legacy_resume_model_requirements(
            plan=plan,
            context=object(),
            run_dir=tmp_path,
            resume_state={
                "per_step_records": [
                    {"step_id": "01_completed_description", "status": "ok"},
                ]
            },
            resume_from_step_id=None,
            role_resolver=lambda _role: object(),
            evidence=evidence,
            prompt_version="test-pack",
            llm_signature="test-planner",
        )

    assert not list(tmp_path.glob("analysis_plan_revision_*.json"))
    assert evidence.get("analysis_plan_revision_2") is None


def test_resume_migration_llm_failure_has_no_default_model_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _legacy_plan()
    evidence = _evidence_with_base_plan(tmp_path, plan)
    calls = []

    class FailingReplanner:
        def __init__(self, llm) -> None:
            calls.append(llm)

        def run(self, **_kwargs):
            raise RuntimeError("planner unavailable")

    monkeypatch.setattr(pipeline_module, "ReplannerAgent", FailingReplanner)
    with pytest.raises(
        LegacyResumePlanMigrationError,
        match="stopped without a default model",
    ):
        _migrate_legacy_resume_model_requirements(
            plan=plan,
            context=object(),
            run_dir=tmp_path,
            resume_state={"per_step_records": []},
            resume_from_step_id=None,
            role_resolver=lambda _role: "real-planner-client",
            evidence=evidence,
            prompt_version="test-pack",
            llm_signature="test-planner",
        )

    assert calls == ["real-planner-client"]
    assert all(not step.model_requirements for step in plan.steps)
    assert not list(tmp_path.glob("analysis_plan_revision_*.json"))
    assert evidence.get("analysis_plan_revision_2") is None
