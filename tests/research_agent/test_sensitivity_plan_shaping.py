from __future__ import annotations

from easyicu.research_agent.contracts.association_execution import (
    association_binary_sensitivity_plan_verdict,
)
from easyicu.research_agent.planning.robustness_contract import RobustnessSpec
from easyicu.research_agent.planning.sensitivity_plan_shaping import (
    ensure_prespecified_sensitivity_steps,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    LiteratureDesignBinding,
    PlannedModelRequirement,
    ResearchContext,
    UserPreferences,
    VariableRole,
)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Is exposure associated with the binary outcome?",
        cohort=CohortDescriptor(
            cohort_name="adult ICU stays",
            database="miiv",
            n_stays=100,
            id_columns=["stay_id"],
        ),
        variables=[
            ConceptDescriptor(name="exposure", role=VariableRole.OTHER, dtype="int64"),
            ConceptDescriptor(name="outcome", role=VariableRole.OUTCOME, dtype="int64"),
            ConceptDescriptor(name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64"),
        ],
        target_outcome="outcome",
        primary_exposure="exposure",
        user_preferences=UserPreferences(
            covariates=["age"],
            sensitivity_specs=[
                {
                    "spec_id": "age_functional_form",
                    "axis": "functional_form",
                    "strategy": "restricted_cubic_spline",
                    "execution_variables": ["age"],
                },
                {
                    "spec_id": "complete_case_primary",
                    "axis": "missing_data",
                    "strategy": "complete_case",
                    "execution_variables": ["exposure", "outcome", "age"],
                },
            ],
        ),
    )


def _plan() -> AnalysisPlan:
    spline_binding = LiteratureDesignBinding(
        citation_key="spline_method",
        design_elements=["adjustment", "robustness"],
        application="Use the sealed spline method card for the functional-form check.",
    )
    primary = AnalysisStep(
        step_id="primary_adjusted_association",
        planned_analysis_role="primary",
        intent="Estimate the adjusted association.",
        method="adjusted_association_models",
        inputs=["exposure", "outcome", "age", "artifact:analysis_cohort"],
        expected_outputs=["table:adjusted_association_estimates"],
        sensitivity_spec_ids=["age_functional_form", "complete_case_primary"],
        literature_citation_keys=["spline_method"],
        literature_design_bindings=[spline_binding],
        model_requirements=[
            PlannedModelRequirement(
                requirement_id="primary",
                outcome="outcome",
                outcome_type="binary",
                method_family="statsmodels_logit_mle",
                exposure_source="exposure",
                analysis_role="primary",
                analysis_set="source_aware",
                covariates=["age"],
                model_terms=[
                    {
                        "name": "exposure",
                        "role": "exposure",
                        "coding": "binary",
                        "levels": ["0", "1"],
                        "reference_level": "0",
                        "transform": "treatment_contrast",
                    },
                    {
                        "name": "age",
                        "role": "covariate",
                        "coding": "continuous",
                        "transform": "identity",
                    },
                ],
            )
        ],
    )
    replay = AnalysisStep.model_validate(
        {
            "step_id": "robustness_grid",
            "planned_analysis_role": "sensitivity",
            "intent": "Replay the locked complete-case specification.",
            "method": "robustness_sensitivity",
            "inputs": [
                "artifact:analysis_cohort",
                "table:adjusted_association_estimates",
            ],
            "expected_outputs": ["table:robustness_matrix"],
            "sensitivity_spec_ids": [
                "age_functional_form",
                "complete_case_primary",
            ],
            "literature_citation_keys": ["spline_method"],
            "literature_design_bindings": [spline_binding.model_dump(mode="json")],
            "robustness_replay_spec": {
                "products": [
                    {
                        "product_id": "robustness_matrix",
                        "output": "robustness_matrix",
                    }
                ]
            },
        }
    )
    return AnalysisPlan(
        research_question="Is exposure associated with the binary outcome?",
        analysis_type="association_study",
        steps=[primary, replay],
        robustness_specs=[
            RobustnessSpec(
                spec_id="complete_case_primary",
                axis="missing",
                description="Locked complete-case replay.",
                missing_override={
                    "strategy": "complete_case",
                    "variables": ["exposure", "outcome", "age"],
                },
            )
        ],
    )


def test_missing_typed_functional_form_becomes_explicit_analysis_only_step() -> None:
    shaped, findings = ensure_prespecified_sensitivity_steps(
        plan=_plan(),
        context=_context(),
    )

    inserted = [
        step
        for step in shaped.steps
        if step.step_id == "sensitivity_age_functional_form"
    ]
    assert len(inserted) == 1
    step = inserted[0]
    assert step.method == "restricted_cubic_spline_sensitivity"
    assert step.sensitivity_spec_ids == ["age_functional_form"]
    assert step.scientific_capability == "association_freeform_v1"
    assert step.expected_outputs == ["table:sensitivity_age_functional_form"]
    assert "complete_case_primary" not in step.sensitivity_spec_ids
    verdict = association_binary_sensitivity_plan_verdict(
        step,
        plan_steps=shaped.steps,
    )
    assert verdict.claimed is True
    assert findings[0].detail["deterministic_method_adapter"] is False

    again, repeated = ensure_prespecified_sensitivity_steps(
        plan=shaped,
        context=_context(),
    )
    assert again == shaped
    assert repeated == []


def test_missing_linear_per_unit_becomes_explicit_analysis_only_step() -> None:
    context = _context()
    preferences = UserPreferences.model_validate(
        {
            **context.user_preferences.model_dump(mode="json"),
            "sensitivity_specs": [
                {
                    "spec_id": "exposure_linear_per_unit",
                    "axis": "functional_form",
                    "strategy": "linear_per_unit",
                    "execution_variables": ["exposure"],
                }
            ],
        }
    )
    context = context.model_copy(update={"user_preferences": preferences})

    shaped, findings = ensure_prespecified_sensitivity_steps(
        plan=_plan(),
        context=context,
    )

    step = next(
        item for item in shaped.steps
        if item.step_id == "sensitivity_exposure_linear_per_unit"
    )
    assert step.method == "linear_per_unit_sensitivity"
    assert step.expected_outputs == ["table:sensitivity_exposure_linear_per_unit"]
    assert findings[0].detail["strategy"] == "linear_per_unit"
