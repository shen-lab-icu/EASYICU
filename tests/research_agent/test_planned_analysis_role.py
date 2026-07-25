"""Planner-owned analysis-role contract tests."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.core import (
    PlannerAgent,
    _build_planner_user_prompt,
    _canonicalise_figure_output_alias,
    _is_untyped_figure_alias_output,
    _normalise_plan_payload,
    _validate_required_primary_result,
)
from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    PlannedAnalysisRole,
    PlannedModelRequirement,
    ResearchContext,
    StepRecord,
)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Estimate the prespecified study result.",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
    )


def _raw_plan(*, include_role: bool, role: str = "primary") -> str:
    step = {
        "step_id": "01_model",
        "intent": "Estimate the prespecified result.",
        "inputs": [],
        "expected_outputs": ["table:estimate"],
        "method": "descriptive",
    }
    if include_role:
        step["planned_analysis_role"] = role
    return json.dumps(
        {
            "research_question": "Estimate the prespecified study result.",
            "steps": [step],
            "rationale": "Use the declared analysis plan.",
        }
    )


def test_host_constructed_step_defaults_to_auxiliary() -> None:
    step = AnalysisStep(step_id="01_prepare", intent="Prepare typed inputs.")
    assert step.planned_analysis_role == "auxiliary"


def test_planned_analysis_role_is_part_of_the_public_schema_api() -> None:
    from easyicu import research_agent

    assert research_agent.PlannedAnalysisRole is PlannedAnalysisRole


@pytest.mark.parametrize("role", ["primary", "secondary", "sensitivity", "auxiliary"])
def test_analysis_step_accepts_each_typed_role(role: str) -> None:
    step = AnalysisStep(
        step_id="01_step",
        intent="Run the planned step.",
        planned_analysis_role=role,
    )
    assert step.planned_analysis_role == role


def test_analysis_step_rejects_unknown_role() -> None:
    with pytest.raises(ValidationError, match="planned_analysis_role"):
        AnalysisStep(
            step_id="01_step",
            intent="Run the planned step.",
            planned_analysis_role="headline",
        )


def test_analysis_plan_allows_zero_primary_steps() -> None:
    plan = AnalysisPlan(
        research_question="Prepare the research package.",
        steps=[AnalysisStep(step_id="01_prepare", intent="Prepare typed inputs.")],
    )
    assert not [step for step in plan.steps if step.planned_analysis_role == "primary"]


def test_analysis_plan_rejects_multiple_primary_steps() -> None:
    with pytest.raises(ValidationError, match="at most one step"):
        AnalysisPlan(
            research_question="Estimate one headline result.",
            steps=[
                AnalysisStep(
                    step_id="01_model",
                    intent="Estimate the headline result.",
                    planned_analysis_role="primary",
                ),
                AnalysisStep(
                    step_id="02_model",
                    intent="Estimate another headline result.",
                    planned_analysis_role="primary",
                ),
            ],
        )


def _association_context() -> ResearchContext:
    return ResearchContext(
        research_question="Estimate the adjusted association with mortality.",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
        primary_exposure="sealed_exposure",
        target_outcome="sealed_outcome",
    )


def _primary_association_step() -> AnalysisStep:
    return AnalysisStep(
        step_id="03_primary_association",
        planned_analysis_role="primary",
        intent="Fit the prespecified adjusted association.",
        inputs=["sealed_exposure", "sealed_outcome"],
        expected_outputs=["table:adjusted_association_estimates"],
        method="adjusted_association_models",
        model_requirements=[
            PlannedModelRequirement(
                requirement_id="primary_adjusted",
                outcome="sealed_outcome",
                outcome_type="binary",
                method_family="logistic_regression",
                exposure_source="sealed_exposure",
                analysis_role="primary",
                analysis_set="source_aware",
                required_for_step_success=True,
            )
        ],
    )


def test_result_bearing_association_rejects_secondary_only_plan() -> None:
    context = _association_context()
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="association_study",
        steps=[
            AnalysisStep(
                step_id="01_feasibility",
                planned_analysis_role="secondary",
                intent="Audit whether the requested estimate may be feasible.",
                expected_outputs=["table:estimand_feasibility"],
                method="feasibility_audit",
            )
        ],
    )

    with pytest.raises(ValueError, match="requires exactly one Planner-owned primary"):
        _validate_required_primary_result(plan=plan, context=context)


def test_association_feasibility_step_cannot_masquerade_as_primary_model() -> None:
    context = _association_context()
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="association_study",
        steps=[
            AnalysisStep(
                step_id="01_feasibility",
                planned_analysis_role="primary",
                intent="Audit whether the requested estimate may be feasible.",
                expected_outputs=["table:estimand_feasibility"],
                method="feasibility_audit",
            )
        ],
    )

    with pytest.raises(ValueError, match="adjusted_association_models"):
        _validate_required_primary_result(plan=plan, context=context)


def test_association_primary_model_must_use_exact_context_coordinates() -> None:
    context = _association_context()
    wrong = _primary_association_step().model_copy(
        update={
            "model_requirements": [
                _primary_association_step()
                .model_requirements[0]
                .model_copy(update={"exposure_source": "plausible_alias"})
            ]
        }
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="association_study",
        steps=[wrong],
    )

    with pytest.raises(ValueError, match="exact ResearchContext operational exposure"):
        _validate_required_primary_result(plan=plan, context=context)


def test_valid_primary_association_satisfies_headline_contract() -> None:
    context = _association_context()
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="association_study",
        steps=[_primary_association_step()],
    )

    _validate_required_primary_result(plan=plan, context=context)


def test_protocol_only_family_may_still_have_no_primary_result() -> None:
    context = _context().model_copy(
        update={
            "research_question": "Audit whether the requested data fields exist.",
            "primary_exposure": None,
            "target_outcome": None,
        }
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="data_quality_audit",
        steps=[
            AnalysisStep(
                step_id="01_audit",
                intent="Audit the sealed data inventory.",
                planned_analysis_role="auxiliary",
                expected_outputs=["table:data_quality"],
                method="data_quality_audit",
            )
        ],
    )

    _validate_required_primary_result(plan=plan, context=context)


@pytest.mark.parametrize(
    "outputs",
    [
        [],
        ["forest_plot.svg"],
        ["figure:forest_plot"],
        ["log:audit"],
        ["report:manuscript"],
        ["code:analysis"],
        ["test:quality"],
    ],
)
def test_analysis_plan_rejects_primary_without_typed_scientific_result(
    outputs: list[str],
) -> None:
    with pytest.raises(ValidationError, match="typed, non-rendering"):
        AnalysisPlan(
            research_question="Estimate one headline result.",
            steps=[
                AnalysisStep(
                    step_id="01_primary",
                    intent="Produce only a renderer or support artifact.",
                    planned_analysis_role="primary",
                    expected_outputs=outputs,
                )
            ],
        )


def test_step_record_role_is_typed_but_historical_none_remains_readable() -> None:
    historical = StepRecord(step_id="01_model", intent="Run model.")
    current = StepRecord(
        step_id="01_model",
        intent="Run model.",
        planned_analysis_role="primary",
    )
    assert historical.planned_analysis_role is None
    assert current.planned_analysis_role == "primary"
    with pytest.raises(ValidationError, match="planned_analysis_role"):
        StepRecord(
            step_id="01_model",
            intent="Run model.",
            planned_analysis_role="headline",
        )


def test_plan_normalizer_preserves_planned_analysis_role() -> None:
    normalized, dropped = _normalise_plan_payload(
        json.loads(_raw_plan(include_role=True, role="secondary"))
    )
    assert normalized["steps"][0]["planned_analysis_role"] == "secondary"
    assert not dropped["steps"]


def test_plan_normalizer_compiles_robustness_role_for_registered_method() -> None:
    raw = json.loads(_raw_plan(include_role=True, role="robustness"))
    raw["steps"][0]["method"] = "robustness_sensitivity"
    raw["steps"][0]["model_requirements"] = [
        {
            "requirement_id": "locked_sensitivity_refit",
            "outcome": "death",
            "outcome_type": "binary",
            "method_family": "logistic_regression",
            "exposure_source": "lact_max",
            "analysis_role": "robustness",
            "analysis_set": "complete_case",
            "required_for_step_success": False,
        }
    ]

    normalized, _dropped = _normalise_plan_payload(raw)

    assert normalized["steps"][0]["planned_analysis_role"] == "sensitivity"
    assert (
        normalized["steps"][0]["model_requirements"][0]["analysis_role"]
        == "sensitivity"
    )


def test_plan_normalizer_does_not_guess_robustness_role_for_other_method() -> None:
    raw = json.loads(_raw_plan(include_role=True, role="robustness"))
    raw["steps"][0]["method"] = "descriptive_summary"

    normalized, _dropped = _normalise_plan_payload(raw)

    assert normalized["steps"][0]["planned_analysis_role"] == "robustness"
    with pytest.raises(ValidationError, match="planned_analysis_role"):
        AnalysisPlan.model_validate(normalized)


def test_plan_normalizer_canonicalises_role_casing_and_whitespace() -> None:
    raw = json.loads(_raw_plan(include_role=True, role="  SenSitivity  "))

    normalized, _dropped = _normalise_plan_payload(raw)

    assert normalized["steps"][0]["planned_analysis_role"] == "sensitivity"


def test_planner_does_not_retry_registered_robustness_role_alias() -> None:
    raw = json.loads(_raw_plan(include_role=True, role="robustness"))
    raw["steps"][0]["method"] = "robustness_sensitivity"
    llm = PatternScriptedMockLLMClient([("ICU-AWARE RESEARCH PLAN", [json.dumps(raw)])])

    plan = PlannerAgent(llm).run(_context())

    assert plan.steps[0].planned_analysis_role == "sensitivity"
    assert len(llm.calls) == 1


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("fig:missingness_measurement", "figure:missingness_measurement"),
        ("plot:adjusted_associations", "figure:adjusted_associations"),
        ("chart:stage_outcomes", "figure:stage_outcomes"),
        ("heatmap:coverage", "figure:coverage"),
        ("fig: spaced_name ", "figure:spaced_name"),
        # already canonical / non-figure / non-typed tokens are untouched
        ("figure:primary_association", "figure:primary_association"),
        ("table:table_one", "table:table_one"),
        ("artifact:analysis_cohort", "artifact:analysis_cohort"),
        ("fig_adjusted_associations", "fig_adjusted_associations"),
        ("noseparator", "noseparator"),
    ],
)
def test_canonicalise_figure_output_alias(token: str, expected: str) -> None:
    assert _canonicalise_figure_output_alias(token) == expected


def test_canonicalise_figure_output_alias_passes_through_non_strings() -> None:
    sentinel = {"not": "a string"}
    assert _canonicalise_figure_output_alias(sentinel) is sentinel


def test_plan_normalizer_canonicalises_figure_output_aliases() -> None:
    raw = json.loads(_raw_plan(include_role=True))
    raw["steps"][0]["expected_outputs"] = [
        "table:missingness_audit",
        "fig:missingness_measurement",
    ]
    normalized, _dropped = _normalise_plan_payload(raw)
    assert normalized["steps"][0]["expected_outputs"] == [
        "table:missingness_audit",
        "figure:missingness_measurement",
    ]


@pytest.mark.parametrize(
    "token",
    [
        "fig_stage_outcomes",
        "fig_adjusted_associations",
        "fig_robustness",
        "figure_gallery",
        "plot_trend",
        "chart_overview",
        "heatmap_coverage",
    ],
)
def test_untyped_figure_alias_output_is_detected(token: str) -> None:
    assert _is_untyped_figure_alias_output(token) is True


@pytest.mark.parametrize(
    "token",
    [
        "figure:stage_outcomes",  # already typed
        "fig:missingness_measurement",  # typed alias
        "table:missingness_audit",  # non-figure typed
        "analysis_cohort",  # non-figure bare token
        "overview.png",  # legitimate bare image export
        "figuration_summary",  # 'figure' is not the underscore head
        "",
    ],
)
def test_untyped_figure_alias_output_ignores_non_malformed(token: str) -> None:
    assert _is_untyped_figure_alias_output(token) is False


def test_plan_normalizer_rejects_e3_shaped_no_colon_figure_output() -> None:
    """E3's planner emitted ``fig_stage_outcomes`` (underscore, no colon)."""

    raw = json.loads(_raw_plan(include_role=True))
    raw["steps"][0]["expected_outputs"] = [
        "table:stage_stratified_outcomes",
        "fig_stage_outcomes",
    ]
    with pytest.raises(ValueError, match="figure:stage_outcomes"):
        _normalise_plan_payload(raw)


def test_plan_normalizer_rejects_duplicate_figure_after_alias_collision() -> None:
    """``fig:x`` and ``figure:x`` both normalise to ``figure:x`` -> one figure."""

    raw = json.loads(_raw_plan(include_role=True))
    raw["steps"][0]["expected_outputs"] = [
        "fig:primary_association",
        "figure:primary_association",
    ]
    with pytest.raises(ValueError, match="more than one output alias"):
        _normalise_plan_payload(raw)


def test_plan_normalizer_rejects_case_only_figure_collision() -> None:
    """A case-only difference collapses to one physical figure identity."""

    raw = json.loads(_raw_plan(include_role=True))
    raw["steps"][0]["expected_outputs"] = [
        "figure:Forest",
        "figure:forest",
    ]
    with pytest.raises(ValueError, match="more than one output alias"):
        _normalise_plan_payload(raw)


def test_plan_normalizer_rejects_suffix_only_figure_collision() -> None:
    """``figure:forest`` and ``figure:forest.svg`` are the same physical figure."""

    raw = json.loads(_raw_plan(include_role=True))
    raw["steps"][0]["expected_outputs"] = [
        "figure:forest",
        "figure:forest.svg",
    ]
    with pytest.raises(ValueError, match="more than one output alias"):
        _normalise_plan_payload(raw)


def test_plan_normalizer_rejects_alias_and_suffix_figure_collision() -> None:
    """``fig:forest.png`` and ``figure:forest`` collapse to one identity."""

    raw = json.loads(_raw_plan(include_role=True))
    raw["steps"][0]["expected_outputs"] = [
        "fig:forest.png",
        "figure:forest",
    ]
    with pytest.raises(ValueError, match="more than one output alias"):
        _normalise_plan_payload(raw)


def test_plan_normalizer_allows_distinct_figures_including_suffix() -> None:
    """Distinct figure identities are preserved even when one carries a suffix."""

    raw = json.loads(_raw_plan(include_role=True))
    raw["steps"][0]["expected_outputs"] = [
        "figure:forest.svg",
        "figure:km_curve",
    ]
    normalized, _dropped = _normalise_plan_payload(raw)
    assert normalized["steps"][0]["expected_outputs"] == [
        "figure:forest.svg",
        "figure:km_curve",
    ]


def test_plan_normalizer_accepts_canonical_e3_figure_suite() -> None:
    """The typed form of the whole E3 figure suite passes and is preserved."""

    raw = json.loads(_raw_plan(include_role=True))
    raw["steps"][0]["expected_outputs"] = [
        "figure:stage_outcomes",
        "plot:adjusted_associations",
        "chart:robustness",
    ]
    normalized, _dropped = _normalise_plan_payload(raw)
    assert normalized["steps"][0]["expected_outputs"] == [
        "figure:stage_outcomes",
        "figure:adjusted_associations",
        "figure:robustness",
    ]


def test_planner_parse_requires_explicit_role_despite_schema_default() -> None:
    planner = PlannerAgent.__new__(PlannerAgent)
    planner.last_dropped_plan_keys = {"top_level": [], "steps": []}
    with pytest.raises(ValueError, match="must explicitly declare"):
        planner._parse(_raw_plan(include_role=False), _context())


def test_planner_parse_preserves_explicit_role() -> None:
    planner = PlannerAgent.__new__(PlannerAgent)
    planner.last_dropped_plan_keys = {"top_level": [], "steps": []}
    plan = planner._parse(_raw_plan(include_role=True, role="sensitivity"), _context())
    assert plan.steps[0].planned_analysis_role == "sensitivity"


def test_planner_run_retries_missing_role_and_feedback_names_contract() -> None:
    llm = PatternScriptedMockLLMClient(
        [
            (
                "ICU-AWARE RESEARCH PLAN",
                [
                    _raw_plan(include_role=False),
                    _raw_plan(include_role=True),
                ],
            )
        ]
    )
    plan = PlannerAgent(llm).run(_context())

    assert plan.steps[0].planned_analysis_role == "primary"
    assert len(llm.calls) == 2
    retry_feedback = llm.calls[1][0][-1].content
    assert "planned_analysis_role" in retry_feedback


def test_planner_prompt_defines_required_role_without_case_specific_terms() -> None:
    prompt = _build_planner_user_prompt(_context())
    assert "Every step MUST explicitly declare `planned_analysis_role`" in prompt
    assert "at most one step may be primary" in prompt
    assert '"planned_analysis_role": "auxiliary"' in prompt
    assert "exactly one materialised closed primary-cohort product" in prompt
    assert "`artifact:cohort_defined` is not a cohort dataset" in prompt
    assert "Do not impute the primary exposure or outcome" in prompt
    assert "fit every imputer/scaler only on the training split" in prompt
    assert "never use future observations to fill an earlier window" in prompt
