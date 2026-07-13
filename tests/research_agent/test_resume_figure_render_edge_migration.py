from __future__ import annotations

import pytest

from easyicu.research_agent.evidence import EvidenceStore
from easyicu.research_agent.pipeline import (
    _migrate_legacy_resume_figure_render_edges,
)
from easyicu.research_agent.plan_utils import _render_only_figure_step_intent
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _legacy_plan(
    *,
    child_inputs: list[str] | None = None,
    child_intent: str | None = None,
    parent_outputs: list[str] | None = None,
    extra_steps: list[AnalysisStep] | None = None,
) -> AnalysisPlan:
    parent = AnalysisStep(
        step_id="01_define_cohort",
        intent="Define the locked analysis cohort and report attrition.",
        method="cohort_definition_and_attrition",
        inputs=["stay_id", "age"],
        expected_outputs=parent_outputs
        or ["artifact:adult_cohort", "table:cohort_flow"],
        icu_rule_refs=["adult_rule", "admission_anchor"],
    )
    figures = ["figure:cohort_flow"]
    child = AnalysisStep(
        step_id="01_define_cohort_figure",
        intent=child_intent
        or _render_only_figure_step_intent(
            source_step_id=parent.step_id,
            figure_outputs=figures,
        ),
        method=parent.method,
        inputs=child_inputs if child_inputs is not None else list(parent.inputs),
        expected_outputs=figures,
        icu_rule_refs=[*parent.icu_rule_refs, "visualization_rule"],
    )
    return AnalysisPlan(
        research_question="Describe a cohort with a source-backed flow figure.",
        analysis_type="descriptive_study",
        revision=3,
        rationale="Keep analysis and rendering separate.",
        steps=[parent, child, *(extra_steps or [])],
    )


def _resume_state(*step_ids: str) -> dict[str, object]:
    return {
        "per_step_records": [
            {"step_id": step_id, "status": "ok"} for step_id in step_ids
        ]
    }


def _migrate(
    tmp_path,
    plan: AnalysisPlan,
    *,
    completed: tuple[str, ...],
    resume_from: str = "01_define_cohort_figure",
):
    evidence = EvidenceStore(tmp_path)
    result = _migrate_legacy_resume_figure_render_edges(
        plan=plan,
        run_dir=tmp_path,
        resume_state=_resume_state(*completed),
        resume_from_step_id=resume_from,
        evidence=evidence,
        prompt_version="test",
        llm_signature="mock",
    )
    return (*result, evidence)


def test_resume_migrates_unique_parent_table_and_excludes_artifact(tmp_path) -> None:
    plan = _legacy_plan()

    migrated, revision_path, step_ids, evidence = _migrate(
        tmp_path,
        plan,
        completed=("01_define_cohort",),
    )

    assert revision_path == tmp_path / "analysis_plan_revision_4.json"
    assert step_ids == ("01_define_cohort_figure",)
    assert migrated.revision == 4
    assert migrated.steps[0] == plan.steps[0]
    assert migrated.steps[1].inputs == ["table:cohort_flow"]
    assert migrated.steps[1].method == "visualization"
    assert migrated.steps[1].intent == plan.steps[1].intent
    assert migrated.steps[1].expected_outputs == plan.steps[1].expected_outputs
    assert "artifact:adult_cohort" not in migrated.steps[1].inputs
    assert migrated.research_question == plan.research_question
    assert migrated.analysis_type == plan.analysis_type
    record = evidence.get("analysis_plan_revision_4")
    assert record is not None
    assert record.metadata["reason"] == "resume_legacy_figure_render_edges"
    assert record.metadata["target_step_ids"] == ["01_define_cohort_figure"]


def test_resume_migrates_all_unique_parent_tables_but_not_raw_artifacts(
    tmp_path,
) -> None:
    plan = _legacy_plan(
        parent_outputs=[
            "artifact:adult_cohort",
            "table:distribution",
            "table:measurement_audit",
            "log:analysis_notes",
        ]
    )

    migrated, revision_path, step_ids, _ = _migrate(
        tmp_path,
        plan,
        completed=("01_define_cohort",),
    )

    assert revision_path is not None
    assert step_ids == ("01_define_cohort_figure",)
    assert migrated.steps[1].inputs == [
        "table:distribution",
        "table:measurement_audit",
    ]


def test_resume_figure_edge_migration_is_idempotent(tmp_path) -> None:
    plan = _legacy_plan()
    evidence = EvidenceStore(tmp_path)
    migrated, _, _ = _migrate_legacy_resume_figure_render_edges(
        plan=plan,
        run_dir=tmp_path,
        resume_state=_resume_state("01_define_cohort"),
        resume_from_step_id="01_define_cohort_figure",
        evidence=evidence,
        prompt_version="test",
        llm_signature="mock",
    )

    unchanged, revision_path, step_ids = _migrate_legacy_resume_figure_render_edges(
        plan=migrated,
        run_dir=tmp_path,
        resume_state=_resume_state("01_define_cohort"),
        resume_from_step_id="01_define_cohort_figure",
        evidence=evidence,
        prompt_version="test",
        llm_signature="mock",
    )

    assert unchanged == migrated
    assert revision_path is None
    assert step_ids == ()


def test_resume_does_not_guess_when_legacy_inputs_are_not_exact_parent_inputs(
    tmp_path,
) -> None:
    for inputs in (["age"], ["stay_id", "table:cohort_flow"]):
        plan = _legacy_plan(child_inputs=list(inputs))
        unchanged, revision_path, step_ids, _ = _migrate(
            tmp_path / str(len(inputs)),
            plan,
            completed=("01_define_cohort",),
        )
        assert unchanged == plan
        assert revision_path is None
        assert step_ids == ()


def test_resume_does_not_guess_ambiguous_or_nonexact_render_product(tmp_path) -> None:
    plans = [
        _legacy_plan(
            parent_outputs=[
                "table:cohort_flow",
                "statistic:cohort_flow",
            ]
        ),
        _legacy_plan(parent_outputs=["artifact:cohort_flow_values"]),
        _legacy_plan(
            extra_steps=[
                AnalysisStep(
                    step_id="02_duplicate",
                    intent="Declare an ambiguous duplicate producer.",
                    method="descriptive_summary",
                    expected_outputs=["table:cohort_flow"],
                )
            ]
        ),
    ]
    for index, plan in enumerate(plans):
        unchanged, revision_path, step_ids, _ = _migrate(
            tmp_path / str(index),
            plan,
            completed=("01_define_cohort",),
        )
        assert unchanged == plan
        assert revision_path is None
        assert step_ids == ()


def test_resume_migrates_child_when_parent_is_rerun_in_same_resume_window(
    tmp_path,
) -> None:
    plan = _legacy_plan()

    migrated, revision_path, step_ids, _ = _migrate(
        tmp_path,
        plan,
        completed=(),
        resume_from="01_define_cohort",
    )

    assert revision_path is not None
    assert step_ids == ("01_define_cohort_figure",)
    assert migrated.steps[1].inputs == ["table:cohort_flow"]


def test_resume_does_not_migrate_child_before_explicit_resume_cut(tmp_path) -> None:
    later = AnalysisStep(
        step_id="02_later",
        intent="Run a later analysis step.",
        method="descriptive_summary",
        expected_outputs=["table:later"],
    )
    plan = _legacy_plan(extra_steps=[later])

    unchanged, revision_path, step_ids, _ = _migrate(
        tmp_path,
        plan,
        completed=("01_define_cohort",),
        resume_from="02_later",
    )

    assert unchanged == plan
    assert revision_path is None
    assert step_ids == ()


def test_resume_migrates_neither_unverified_parent_nor_completed_child(tmp_path) -> None:
    plan = _legacy_plan()
    unchanged, revision_path, step_ids, _ = _migrate(
        tmp_path / "unverified",
        plan,
        completed=(),
    )
    assert unchanged == plan
    assert revision_path is None
    assert step_ids == ()

    later = AnalysisStep(
        step_id="02_later",
        intent="Run a later analysis step.",
        method="descriptive_summary",
        expected_outputs=["table:later"],
    )
    completed_plan = _legacy_plan(extra_steps=[later])
    unchanged, revision_path, step_ids, _ = _migrate(
        tmp_path / "completed",
        completed_plan,
        completed=("01_define_cohort", "01_define_cohort_figure"),
        resume_from="02_later",
    )
    assert unchanged == completed_plan
    assert revision_path is None
    assert step_ids == ()


def test_resume_requires_full_framework_authored_legacy_intent(tmp_path) -> None:
    plan = _legacy_plan(
        child_intent=(
            "Render a publication figure from another step and remain "
            "rendering-only."
        )
    )

    unchanged, revision_path, step_ids, _ = _migrate(
        tmp_path,
        plan,
        completed=("01_define_cohort",),
    )

    assert unchanged == plan
    assert revision_path is None
    assert step_ids == ()


def _legacy_adjusted_effect_plan(*, figure_output: str, with_roster: bool) -> AnalysisPlan:
    requirements = (
        [
            {
                "requirement_id": "primary_death_model",
                "outcome": "death",
                "outcome_type": "binary",
                "method_family": "logistic_regression",
                "exposure_source": "exposure",
                "analysis_role": "primary",
                "analysis_set": "complete_case",
                "required_for_step_success": True,
            }
        ]
        if with_roster
        else []
    )
    parent = AnalysisStep(
        step_id="05_primary_adjusted_association",
        intent="Fit the Planner-owned adjusted association roster.",
        method="adjusted_association_models",
        inputs=["exposure", "death"],
        expected_outputs=[
            "table:adjusted_association_estimates",
            "artifact:primary_model_specification",
        ],
        icu_rule_refs=["one_record_per_stay"],
        model_requirements=requirements,
    )
    child = AnalysisStep(
        step_id="05_primary_adjusted_association_figure",
        intent=_render_only_figure_step_intent(
            source_step_id=parent.step_id,
            figure_outputs=[figure_output],
        ),
        method=parent.method,
        inputs=list(parent.inputs),
        expected_outputs=[figure_output],
        icu_rule_refs=[*parent.icu_rule_refs, "visualization_rule"],
    )
    return AnalysisPlan(
        research_question="Estimate a Planner-owned adjusted association.",
        analysis_type="association_study",
        revision=2,
        steps=[parent, child],
    )


def test_resume_uses_planner_model_roster_for_generic_primary_effect_edge(
    tmp_path,
) -> None:
    plan = _legacy_adjusted_effect_plan(
        figure_output="figure:primary_adjusted_effect",
        with_roster=True,
    )

    migrated, revision_path, step_ids, _ = _migrate(
        tmp_path,
        plan,
        completed=("05_primary_adjusted_association",),
        resume_from="05_primary_adjusted_association_figure",
    )

    assert revision_path is not None
    assert step_ids == ("05_primary_adjusted_association_figure",)
    assert migrated.steps[1].inputs == ["table:adjusted_association_estimates"]


@pytest.mark.parametrize(
    ("figure_output", "with_roster"),
    [
        ("figure:primary_adjusted_effect", False),
        ("figure:secondary_adjusted_effect", True),
        ("figure:primary_hr_forest", True),
    ],
)
def test_resume_does_not_infer_effect_semantics_beyond_planner_roster(
    tmp_path,
    figure_output: str,
    with_roster: bool,
) -> None:
    plan = _legacy_adjusted_effect_plan(
        figure_output=figure_output,
        with_roster=with_roster,
    )

    unchanged, revision_path, step_ids, _ = _migrate(
        tmp_path,
        plan,
        completed=("05_primary_adjusted_association",),
        resume_from="05_primary_adjusted_association_figure",
    )

    assert unchanged == plan
    assert revision_path is None
    assert step_ids == ()
