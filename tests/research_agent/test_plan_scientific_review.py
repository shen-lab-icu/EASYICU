"""Focused contracts for the digest-bound pre-approval review owner."""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.contracts.claim_ceiling import DescriptiveClaimContract
from easyicu.research_agent.agents.core import PlannerAgent
from easyicu.research_agent.contracts.model_terms import ModelTermSpec
from easyicu.research_agent.contracts.figure_plan import PlannedFigurePanelSpec
from easyicu.research_agent.literature import (
    CitationRecord,
    LiteratureBundle,
    LiteratureScreeningDecision,
    LiteratureSearchProvenance,
)
from easyicu.research_agent.planning.figure_strategy import (
    build_article_figure_strategy,
)
from easyicu.research_agent.planning.dependence_authority import (
    DependenceAuthorityError,
    bind_context_dependence_authority,
    context_dependence_authority,
)
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    repeated_unit_design_closed,
    render_agent_plan_revision_contract,
    render_plan_scientific_guardrails,
)
from easyicu.research_agent.reporting.article_contract import (
    build_article_analysis_contract,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ClinicalDefinitionReference,
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
        research_question=(
            "Among adult ICU stays, is a first-24-hour exposure associated "
            "with in-hospital mortality?"
        ),
        cohort=CohortDescriptor(
            cohort_name="adult ICU stays",
            database="miiv",
            n_patients=None,
            n_stays=94_458,
            inclusion_criteria=["adult ICU stays; retain ICU readmissions"],
            id_columns=["stay_id"],
            provenance={"analysis_unit": "icu_stay"},
        ),
        variables=[
            ConceptDescriptor(
                name="exposure",
                role=VariableRole.OTHER,
                dtype="int64",
                analysis_window="icu_admission[0,24]h",
                analysis_window_role="exposure_definition",
            ),
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="int64"),
            ConceptDescriptor(
                name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64"
            ),
        ],
        target_outcome="death",
        endpoint=EndpointSpec(
            name="death",
            kind="binary",
            absence_semantics="no_absent_rows",
            levels=[0, 1],
        ),
        primary_exposure="exposure",
        user_preferences=UserPreferences(
            covariates=["age"],
            covariate_selection="planner_selectable",
            timing_and_design="Audit timing and readmissions.",
            must_have_outputs="Execute timing and readmission sensitivity analyses.",
        ),
    )


def test_guardrails_accept_a_context_without_optional_user_preferences() -> None:
    context = _context().model_copy(update={"user_preferences": None})

    rendered = render_plan_scientific_guardrails(context)

    assert "PRE-APPROVAL SCIENTIFIC PLAN GUARDRAILS" in rendered


def _binding(key: str, element: str, application: str) -> LiteratureDesignBinding:
    return LiteratureDesignBinding(
        citation_key=key,
        design_elements=[element],
        application=application,
    )


def _plan(*, typed_bindings: bool = True) -> AnalysisPlan:
    keys = ["direct_2024", "strobe_2007", "durrleman_1989"]
    bindings = (
        [
            _binding(
                "direct_2024",
                "estimand",
                "Compare population and estimand prospectively while retaining the sealed EasyICU cohort.",
            ),
            _binding(
                "strobe_2007",
                "reporting",
                "Pre-specify reporting of eligibility, missingness, and adjusted estimates.",
            ),
            _binding(
                "durrleman_1989",
                "adjustment",
                "Use a non-linear age check instead of assuming linear log odds.",
            ),
        ]
        if typed_bindings
        else []
    )
    return AnalysisPlan(
        research_question=_context().research_question,
        analysis_type="association_study",
        steps=[
            AnalysisStep(
                step_id="primary_model",
                planned_analysis_role="primary",
                intent="Estimate the primary adjusted association.",
                inputs=["exposure", "death", "age"],
                expected_outputs=["table:adjusted_association_estimates"],
                method="adjusted_association_models",
                literature_citation_keys=keys,
                literature_design_bindings=bindings,
                model_requirements=[
                    PlannedModelRequirement(
                        requirement_id="primary",
                        outcome="death",
                        outcome_type="binary",
                        method_family="statsmodels_glm_binomial",
                        exposure_source="exposure",
                        analysis_role="primary",
                        analysis_set="source_aware",
                        covariates=["age"],
                        model_terms=[
                            ModelTermSpec(
                                name="exposure",
                                role="exposure",
                                coding="binary",
                                levels=["0", "1"],
                                reference_level="0",
                                transform="treatment_contrast",
                            ),
                            ModelTermSpec(
                                name="age",
                                role="covariate",
                                coding="continuous",
                                transform="identity",
                            ),
                        ],
                    )
                ],
            ),
            AnalysisStep(
                step_id="primary_figure",
                planned_analysis_role="auxiliary",
                intent="Render adjusted effect estimates.",
                inputs=["table:adjusted_association_estimates"],
                expected_outputs=["figure:primary_estimand"],
                method="visualization",
                figure_panels=[
                    PlannedFigurePanelSpec(
                        panel_id="primary_estimand",
                        figure_output="figure:primary_estimand",
                        article_role="primary_estimand",
                        chart_type="forest",
                        source_products=["table:adjusted_association_estimates"],
                    )
                ],
            ),
            AnalysisStep(
                step_id="missingness",
                planned_analysis_role="auxiliary",
                intent="Audit measurement availability.",
                expected_outputs=["table:missingness_audit"],
                method="measurement_audit",
            ),
        ],
    )


def test_formal_review_blocks_an_analysis_only_primary_capability() -> None:
    plan = _plan()
    freeform_primary = plan.steps[0].model_copy(
        update={
            "method": "custom_interaction_model",
            "expected_outputs": ["table:custom_interaction_estimates"],
            "model_requirements": [],
            "scientific_capability": "association_freeform_v1",
        }
    )
    plan = plan.model_copy(update={"steps": [freeform_primary, *plan.steps[1:]]})

    review = build_plan_scientific_review(
        context=_context(),
        plan=plan,
        require_reportable_capability=True,
    )

    assert review.approval_allowed is False
    assert review.status == "changes_required"
    assert review.facts["scientific_capability"]["claim_ceiling"] == "analysis_only"
    assert "SCIENTIFIC_CAPABILITY_NOT_REPORTABLE" in {
        finding.code for finding in review.findings
    }


def _literature() -> LiteratureBundle:
    query = "adult ICU exposure mortality observational cohort"
    return LiteratureBundle(
        research_question=_context().research_question,
        citations=[
            CitationRecord(
                key="direct_2024",
                title="A recent observational ICU study of the same exposure and outcome",
                year="2024",
                venue="Critical Care",
                relevance="Direct comparator for population, exposure, outcome, and estimand appraisal.",
                pmid="12345678",
            ),
            CitationRecord(
                key="strobe_2007",
                title="The STROBE statement",
                year="2007",
                relevance="Reporting guidance for observational studies.",
            ),
            CitationRecord(
                key="durrleman_1989",
                title="Flexible regression models with cubic splines",
                year="1989",
                relevance="Functional-form assessment for continuous covariates.",
            ),
        ],
        search_provenance=LiteratureSearchProvenance(
            curated_seed_count=2,
            sources_enabled=["pubmed"],
            sources_returning=["pubmed"],
            search_queries={"pubmed": [query]},
            record_queries={"direct_2024": [query]},
            search_conducted=True,
            searched_at="2026-08-12T12:00:00+00:00",
        ),
        screening_decisions=[
            LiteratureScreeningDecision(
                citation_key="direct_2024",
                source="pubmed",
                disposition="include",
                evidence_role="direct_comparator",
                rationale="Population, exposure, outcome, and observational design match.",
                population_match=True,
                exposure_match=True,
                outcome_match=True,
                design_excerpt_available=True,
            )
        ],
    )


def test_e1_like_plan_is_nonapprovable_for_clinical_timing_and_dependence() -> None:
    context = _context()
    review = build_plan_scientific_review(
        context=context,
        plan=_plan(),
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )

    codes = {item.code for item in review.findings}
    assert review.status == "changes_required"
    assert review.approval_allowed is False
    assert review.top_journal_candidate is False
    assert {
        "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED",
        "REPEATED_STAY_IDENTITY_UNAVAILABLE",
        "ADJUSTMENT_SET_NOT_USER_CONFIRMED",
        "CONTINUOUS_COVARIATE_FUNCTIONAL_FORM_UNCHECKED",
        "FIGURE_ROLE_COVERAGE_INCOMPLETE",
    } <= codes
    assert review.dimension_scores["icu_clinical_design"] == 0
    assert review.dimension_scores["figures"] < 100
    assert review.facts["novelty_status"] == "independent_review_required"
    assert "NOVELTY_POSITIONING_REVIEW_REQUIRED" in codes
    assert review.facts["novelty_review"]["prespecified_dimensions"] == [
        "population_and_setting",
        "exposure_definition_and_time_zero",
        "outcome_and_estimand",
        "analysis_and_robustness_route",
        "data_source_and_transportability",
        "clinical_decision_or_methodological_contribution",
    ]
    assert review.facts["figure_roles"]["missing_roles"] == [
        "data_quality",
        "descriptive_result",
        "robustness",
    ]
    assert review.facts["figure_roles"]["assessment_scope"] == (
        "planned_roles_only_not_rendered_visual_quality"
    )
    assert review.review_scope == "pre_execution_plan"
    assert review.rendered_outputs_assessed is False
    assert (
        "not assessed until execution"
        in (review.facts["score_interpretation"]["figures"])
    )


def test_free_text_cannot_create_a_required_sensitivity_axis() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                covariates=["age"],
                data_constraints=json.dumps(
                    {
                        "cohort": {
                            "review": (
                                "First ICU stay is not part of the current "
                                "sensitivity analysis."
                            )
                        }
                    }
                ),
                must_have_outputs=(
                    "Describe missing values; do not add a complete-case "
                    "sensitivity unless the user later requests one."
                ),
                sensitivity_specs=[],
            )
        }
    )

    review = build_plan_scientific_review(
        context=context,
        plan=_plan(),
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )

    assert review.facts["sensitivity"]["requested"] == []
    assert "REQUIRED_SENSITIVITY_IS_PROTOCOL_ONLY" not in {
        item.code for item in review.findings
    }


def test_free_text_cannot_authorize_cluster_robust_covariance() -> None:
    context = _context().model_copy(
        update={
            "cohort": _context().cohort.model_copy(
                update={
                    "id_columns": ["subject_id", "stay_id"],
                    "provenance": {"analysis_unit": "icu_stay"},
                }
            ),
            "user_preferences": UserPreferences(
                covariates=["age"],
                data_constraints=(
                    "Do not use patient-cluster-robust covariance in this analysis."
                ),
            ),
        }
    )

    bound = bind_context_dependence_authority(plan=_plan(), context=context)

    assert bound.steps[0].model_requirements[0].dependence is None


def test_identifier_name_alone_cannot_create_patient_group_authority() -> None:
    context = _context().model_copy(
        update={
            "cohort": _context().cohort.model_copy(
                update={
                    "id_columns": ["subject_id", "stay_id"],
                    "provenance": {"analysis_unit": "icu_stay"},
                }
            ),
            "user_preferences": UserPreferences(
                covariates=["age"],
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "cluster_unit": "patient",
                            "variance_estimator": "cluster_robust",
                        }
                    }
                ),
            ),
        }
    )

    assert context_dependence_authority(context) is None


def test_owner_issued_patient_id_role_creates_typed_identity_authority() -> None:
    context = _context().model_copy(
        update={
            "cohort": _context().cohort.model_copy(
                update={
                    "id_columns": ["subject_id", "stay_id"],
                    "provenance": {
                        "analysis_unit": "icu_stay",
                        "patient_id_columns": ["subject_id"],
                    },
                }
            ),
            "user_preferences": UserPreferences(
                covariates=["age"],
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "cluster_unit": "patient",
                            "variance_estimator": "cluster_robust",
                        }
                    }
                ),
            ),
        }
    )

    authority = context_dependence_authority(context)

    assert authority is not None
    assert authority.group_source == "subject_id"
    assert authority.group_derivation == "identity"
    assert authority.delimiter is None


def test_dependence_authority_preserves_the_typed_family_sibling_coordinate() -> None:
    """One closed analysis-design envelope can serve its two distinct owners."""

    context = _context().model_copy(
        update={
            "cohort": _context().cohort.model_copy(
                update={
                    "id_columns": ["subject_id", "stay_id"],
                    "provenance": {
                        "analysis_unit": "icu_stay",
                        "patient_id_columns": ["subject_id"],
                    },
                }
            ),
            "user_preferences": UserPreferences(
                covariates=[],
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_family": "descriptive_epidemiology",
                            "analysis_unit": "icu_stay",
                            "cluster_unit": "patient",
                            "variance_estimator": "cluster_robust",
                        }
                    }
                ),
            ),
        }
    )

    authority = context_dependence_authority(context)

    assert authority is not None
    assert authority.group_source == "subject_id"


def test_dependence_authority_still_rejects_an_unknown_envelope_coordinate() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_family": "descriptive_epidemiology",
                            "analysis_unit": "icu_stay",
                            "cluster_unit": "patient",
                            "variance_estimator": "cluster_robust",
                            "model_output_invented_field": True,
                        }
                    }
                )
            )
        }
    )

    with pytest.raises(
        DependenceAuthorityError,
        match="closed repeated-unit contract",
    ):
        context_dependence_authority(context)


def test_typed_descriptive_ceiling_avoids_temporal_inference_claim() -> None:
    plan = AnalysisPlan(
        research_question=_context().research_question,
        analysis_type="descriptive_study",
        steps=[
            AnalysisStep(
                step_id="descriptive_distribution",
                planned_analysis_role="primary",
                intent="Report only observed group distributions.",
                inputs=["cohort:analysis_set", "exposure", "death"],
                expected_outputs=["table:distribution_prevalence"],
                method="descriptive_distribution",
                descriptive_claim=DescriptiveClaimContract(
                    unresolved_limitations=(
                        "post_baseline_exposure_opportunity_unresolved",
                    )
                ),
            )
        ],
    )

    review = build_plan_scientific_review(
        context=_context(),
        plan=plan,
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(_context()),
    )

    assert "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED" not in {
        item.code for item in review.findings
    }
    assert review.facts["temporal_inference_required"] is False
    assert review.facts["descriptive_only_step_ids"] == ["descriptive_distribution"]


def test_registered_descriptive_capability_preserves_typed_claim_ceiling() -> None:
    step = _absolute_risk_distribution_step().model_copy(
        update={
            "scientific_capability": (
                "descriptive_exposure_outcome_distribution_v1"
            )
        }
    )
    plan = AnalysisPlan(
        research_question=_context().research_question,
        analysis_type="descriptive_study",
        steps=[step],
    )

    review = build_plan_scientific_review(
        context=_context(),
        plan=plan,
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(_context()),
    )

    assert review.facts["temporal_inference_required"] is False
    assert review.facts["descriptive_only_step_ids"] == [
        "absolute_risk_distribution"
    ]


def test_non_descriptive_capability_cannot_borrow_typed_claim_ceiling() -> None:
    step = _absolute_risk_distribution_step().model_copy(
        update={"scientific_capability": "association_freeform_v1"}
    )
    plan = AnalysisPlan(
        research_question=_context().research_question,
        analysis_type="descriptive_study",
        steps=[step],
    )

    review = build_plan_scientific_review(
        context=_context(),
        plan=plan,
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(_context()),
    )

    assert review.facts["temporal_inference_required"] is True
    assert review.facts["descriptive_only_step_ids"] == []


def _absolute_risk_distribution_step(*, descriptive: bool = True) -> AnalysisStep:
    return AnalysisStep(
        step_id="absolute_risk_distribution",
        planned_analysis_role="primary",
        intent="Report observed prevalence, absolute risks, and risk difference.",
        inputs=["cohort:analysis_set", "exposure", "death"],
        expected_outputs=["table:exposure_outcome_distribution"],
        method="descriptive",
        descriptive_claim=(
            DescriptiveClaimContract(
                unresolved_limitations=(
                    "post_baseline_exposure_opportunity_unresolved",
                )
            )
            if descriptive
            else None
        ),
        exposure_outcome_distribution_spec={
            "exposure": "exposure",
            "exposure_levels": [0, 1],
            "outcome": "death",
            "outcome_levels": [0, 1],
            "outcome_positive_value": 1,
            "level_match_policy": "exact_typed",
            "denominator_policy": "all_declared_rows",
            "missing_outcome_policy": "structural_absence_is_non_event",
            "risk_difference_contrast": {
                "reference_exposure_level": 0,
                "comparison_exposure_level": 1,
            },
            "confidence_level": 0.95,
        },
    )


def test_descriptive_absolute_risk_with_supporting_tables_does_not_invent_inference() -> (
    None
):
    plan = AnalysisPlan(
        research_question=_context().research_question,
        analysis_type="descriptive_study",
        steps=[
            _absolute_risk_distribution_step(),
            AnalysisStep(
                step_id="cohort_description",
                planned_analysis_role="secondary",
                intent="Describe the bound analytic denominator.",
                inputs=["cohort:analysis_set"],
                expected_outputs=["table:cohort_summary"],
                method="descriptive_summary",
            ),
            AnalysisStep(
                step_id="measurement_missingness",
                planned_analysis_role="secondary",
                intent="Audit variable availability.",
                inputs=["cohort:analysis_set"],
                expected_outputs=["table:missingness_audit"],
                method="measurement_audit",
            ),
        ],
    )

    review = build_plan_scientific_review(
        context=_context(),
        plan=plan,
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(_context()),
    )

    codes = {item.code for item in review.findings}
    assert "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED" not in codes
    assert review.facts["temporal_inference_required"] is False
    assert review.facts["descriptive_only_step_ids"] == ["absolute_risk_distribution"]
    assert "primary_analysis" not in review.facts["content_roles"]["missing_roles"]
    assert "primary_analysis" not in review.facts["content_roles"]["required_roles"]
    assert review.facts["content_roles"]["source_analysis_type"] == (
        "descriptive_epidemiology"
    )
    assert review.facts["robustness_readiness"] == {
        "status": "blocked",
        "reason": "no_typed_sensitivity_authority",
        "family": "descriptive",
        "family_requirements": [
            "denominator definition sensitivity",
            "missingness/measurement availability sensitivity",
        ],
        "required_axis_count": 2,
        "executable_axes": [],
        "declared_authority_ids": [],
        "effect_style_grid_required": False,
    }
    assert "ROBUSTNESS_AXES_TOO_NARROW" not in codes
    assert "ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED" in codes


def test_absolute_risk_difference_without_typed_ceiling_remains_inferential() -> None:
    plan = AnalysisPlan(
        research_question=_context().research_question,
        analysis_type="descriptive_study",
        steps=[_absolute_risk_distribution_step(descriptive=False)],
    )

    review = build_plan_scientific_review(
        context=_context(),
        plan=plan,
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(_context()),
    )

    assert "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED" in {
        item.code for item in review.findings
    }
    assert review.facts["temporal_inference_required"] is True


def test_descriptive_label_cannot_hide_an_association_without_temporal_design() -> None:
    base = _plan()
    primary = base.steps[0].model_copy(
        update={
            "descriptive_claim": DescriptiveClaimContract(
                unresolved_limitations=(
                    "post_baseline_exposure_opportunity_unresolved",
                )
            )
        }
    )
    plan = base.model_copy(update={"steps": [primary, *base.steps[1:]]})

    review = build_plan_scientific_review(
        context=_context(),
        plan=plan,
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(_context()),
    )

    assert "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED" in {
        item.code for item in review.findings
    }
    assert review.facts["temporal_inference_required"] is True


def test_bound_cluster_contract_closes_repeated_stay_design_without_prose() -> None:
    context = _context().model_copy(
        update={
            "cohort": _context().cohort.model_copy(
                update={
                    "id_columns": ["patient_stay_id"],
                    "provenance": {
                        "analysis_unit": "icu_stay",
                        "replacement_row_identity": {
                            "output_identity_column": "patient_stay_id",
                            "mapping_file_sha256": "a" * 64,
                            "patient_group_derivation": {
                                "algorithm": "prefix_before_:s",
                                "delimiter": ":s",
                            },
                        },
                    },
                }
            ),
            "user_preferences": UserPreferences(
                covariates=["age"],
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "cluster_unit": "patient",
                            "variance_estimator": "cluster_robust",
                        }
                    }
                ),
            ),
        }
    )
    base = _plan()
    primary = base.steps[0]
    requirement = primary.model_requirements[0].model_copy(
        update={"method_family": "statsmodels_logit_mle"}
    )
    plan = base.model_copy(
        update={
            "steps": [
                primary.model_copy(update={"model_requirements": [requirement]}),
                *base.steps[1:],
            ]
        }
    )

    bound = bind_context_dependence_authority(plan=plan, context=context)
    dependence = bound.steps[0].model_requirements[0].dependence

    assert dependence is not None
    assert dependence.variance_estimator == "cluster_robust"
    assert dependence.group_source == "patient_stay_id"
    assert dependence.group_derivation == "prefix_before_delimiter"
    assert dependence.delimiter == ":s"

    review = build_plan_scientific_review(
        context=context,
        plan=bound,
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )
    codes = {item.code for item in review.findings}
    assert "REPEATED_STAY_METHOD_NOT_DECLARED" not in codes
    assert review.facts["repeated_unit_design_executable"] is True


def test_host_binds_patient_dependence_into_descriptive_risk_product() -> None:
    context = _context().model_copy(
        update={
            "cohort": _context().cohort.model_copy(
                update={
                    "id_columns": ["patient_stay_id"],
                    "provenance": {
                        "analysis_unit": "icu_stay",
                        "replacement_row_identity": {
                            "output_identity_column": "patient_stay_id",
                            "mapping_file_sha256": "c" * 64,
                            "patient_group_derivation": {
                                "algorithm": "prefix_before_:s",
                                "delimiter": ":s",
                            },
                        },
                    },
                }
            ),
            "user_preferences": UserPreferences(
                covariates=["age"],
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "cluster_unit": "patient",
                            "variance_estimator": "cluster_robust",
                        }
                    }
                ),
            ),
        }
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_study",
        steps=[_absolute_risk_distribution_step()],
    )

    bound = bind_context_dependence_authority(plan=plan, context=context)
    spec = bound.steps[0].exposure_outcome_distribution_spec

    assert spec is not None and spec.dependence is not None
    assert spec.dependence.group_source == "patient_stay_id"
    assert spec.dependence.group_derivation == "prefix_before_delimiter"
    assert spec.dependence.delimiter == ":s"
    assert spec.repeated_unit_interval_method == "patient_cluster_robust_wald"
    review = build_plan_scientific_review(
        context=context,
        plan=bound,
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )
    codes = {item.code for item in review.findings}
    assert "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED" not in codes
    assert "REPEATED_STAY_METHOD_NOT_DECLARED" not in codes
    assert review.facts["repeated_unit_design_executable"] is True


def _traditional_table_one_step() -> AnalysisStep:
    return AnalysisStep(
        step_id="table_one",
        planned_analysis_role="auxiliary",
        intent="Describe the cohort by exposure group.",
        inputs=["cohort:analysis_set", "exposure", "age"],
        expected_outputs=["table:table_one"],
        method="descriptive",
        table_one_spec={
            "group_by": "exposure",
            "group_levels": [0, 1],
            "variables": [
                {
                    "name": "age",
                    "variable_kind": "continuous",
                    "summary": "median_iqr",
                    "test": "mann_whitney_or_kruskal",
                }
            ],
        },
    )


def test_host_turns_repeated_unit_table_one_into_descriptive_smd_only() -> None:
    context = _context().model_copy(
        update={
            "cohort": _context().cohort.model_copy(
                update={
                    "id_columns": ["patient_stay_id"],
                    "provenance": {
                        "analysis_unit": "icu_stay",
                        "replacement_row_identity": {
                            "output_identity_column": "patient_stay_id",
                            "mapping_file_sha256": "9" * 64,
                            "patient_group_derivation": {
                                "algorithm": "prefix_before_:s",
                                "delimiter": ":s",
                            },
                        },
                    },
                }
            ),
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_family": "descriptive_epidemiology",
                            "analysis_unit": "icu_stay",
                            "cluster_unit": "patient",
                            "variance_estimator": "cluster_robust",
                        }
                    }
                )
            ),
        }
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_study",
        steps=[_traditional_table_one_step()],
    )

    bound = bind_context_dependence_authority(plan=plan, context=context)
    table_one = bound.steps[0].table_one_spec

    assert table_one is not None
    assert table_one.schema_version == "easyicu.table_one/2"
    assert table_one.p_values_required is False
    assert table_one.p_value_adjustment == "not_applicable_repeated_units"
    assert {variable.test for variable in table_one.variables} == {
        "none_descriptive_smd_only"
    }
    assert repeated_unit_design_closed(context, bound) is True
    review = build_plan_scientific_review(
        context=context,
        plan=bound,
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )
    assert "TABLE_ONE_INDEPENDENT_TESTS_IGNORE_REPEATED_UNITS" not in {
        finding.code for finding in review.findings
    }


def test_scientific_review_blocks_unbound_independent_table_one_tests() -> None:
    context = _context().model_copy(
        update={
            "cohort": _context().cohort.model_copy(
                update={
                    "id_columns": ["patient_stay_id"],
                    "provenance": {
                        "analysis_unit": "icu_stay",
                        "replacement_row_identity": {
                            "output_identity_column": "patient_stay_id",
                            "mapping_file_sha256": "8" * 64,
                            "patient_group_derivation": {
                                "algorithm": "prefix_before_:s",
                                "delimiter": ":s",
                            },
                        },
                    },
                }
            ),
        }
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_study",
        steps=[_traditional_table_one_step()],
    )

    review = build_plan_scientific_review(
        context=context,
        plan=plan,
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )

    assert "TABLE_ONE_INDEPENDENT_TESTS_IGNORE_REPEATED_UNITS" in {
        finding.code for finding in review.findings
    }
    assert review.facts["repeated_unit_design_executable"] is False


def test_independent_unit_table_one_keeps_its_declared_tests() -> None:
    context = _context().model_copy(
        update={
            "cohort": _context().cohort.model_copy(
                update={
                    "n_patients": _context().cohort.n_stays,
                    "inclusion_criteria": [],
                    "provenance": {"analysis_unit": "patient"},
                }
            ),
            "user_preferences": None,
        }
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_study",
        steps=[_traditional_table_one_step()],
    )

    unchanged = bind_context_dependence_authority(plan=plan, context=context)

    assert unchanged.steps[0].table_one_spec == plan.steps[0].table_one_spec
    assert unchanged.steps[0].table_one_spec.p_values_required is True


def test_counts_only_authority_removes_all_uncertainty_before_review() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "variance_estimator": "none_counts_only",
                        }
                    }
                )
            )
        }
    )
    distribution_step = _absolute_risk_distribution_step()
    distribution_spec = distribution_step.exposure_outcome_distribution_spec
    assert distribution_spec is not None
    distribution_step = distribution_step.model_copy(
        update={
            "scientific_capability": (
                "descriptive_exposure_outcome_distribution_v1"
            ),
            "exposure_outcome_distribution_spec": distribution_spec.model_copy(
                update={"risk_difference_contrast": None}
            )
        }
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_study",
        steps=[distribution_step],
    )

    bound = bind_context_dependence_authority(plan=plan, context=context)

    distribution = bound.steps[0].exposure_outcome_distribution_spec
    assert distribution is not None
    assert distribution.schema_version == "easyicu.exposure_outcome_distribution/3"
    assert distribution.interval_method == "none_counts_only"
    assert distribution.repeated_unit_interval_method is None
    assert distribution.confidence_level is None
    assert distribution.dependence is None
    assert repeated_unit_design_closed(context, bound) is True


def test_counts_only_article_contract_does_not_require_forbidden_table_one() -> None:
    ordinary = _context()
    context = ordinary.model_copy(
        update={
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "variance_estimator": "none_counts_only",
                        }
                    }
                )
            )
        }
    )

    contract = build_article_analysis_contract(
        context,
        analysis_type="descriptive_epidemiology",
    )
    ordinary_contract = build_article_analysis_contract(
        ordinary,
        analysis_type="descriptive_epidemiology",
    )

    assert "baseline_context" not in contract.required_roles
    assert all(item.role != "baseline_context" for item in contract.requirements)
    assert all(
        item.module_id != "distribution_prevalence"
        for item in contract.requirements
    )
    expected_roles = {
        item.role
        for item in ordinary_contract.requirements
        if item.required
        and item.role != "baseline_context"
        and item.module_id != "distribution_prevalence"
    }
    assert set(contract.required_roles) == expected_roles


def test_counts_only_typed_primary_subsumes_generic_distribution_module() -> None:
    context = _context().model_copy(
        update={
            "research_question": (
                "Estimate exposure prevalence and observed outcome event rates "
                "by exposure among ICU stays."
            ),
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "variance_estimator": "none_counts_only",
                        }
                    }
                )
            ),
        }
    )

    contract = build_article_analysis_contract(
        context,
        analysis_type="descriptive_epidemiology",
    )

    assert "descriptive_result" in contract.required_roles
    assert "distribution" not in contract.required_roles
    assert "baseline_context" not in contract.required_roles


def test_counts_only_authority_rejects_inferential_table_one_tests() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "variance_estimator": "none_counts_only",
                        }
                    }
                )
            )
        }
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_study",
        steps=[_traditional_table_one_step()],
    )

    with pytest.raises(
        DependenceAuthorityError,
        match="forbids inferential Table One",
    ):
        bind_context_dependence_authority(plan=plan, context=context)


def test_counts_only_authority_accepts_descriptive_smd_table_one_and_report() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "variance_estimator": "none_counts_only",
                        }
                    }
                )
            )
        }
    )
    table_step = _traditional_table_one_step()
    table_spec = table_step.table_one_spec
    assert table_spec is not None
    table_payload = table_spec.model_dump(mode="python")
    table_payload.update(
        schema_version="easyicu.table_one/2",
        p_values_required=False,
        p_value_adjustment="not_applicable_repeated_units",
    )
    for variable in table_payload["variables"]:
        variable["test"] = "none_descriptive_smd_only"
    table_step = table_step.model_copy(
        update={
            "table_one_spec": type(table_spec).model_validate(table_payload),
        }
    )
    report_step = AnalysisStep(
        step_id="report",
        planned_analysis_role="auxiliary",
        intent="Render the counts-only report.",
        inputs=["table:table_one"],
        expected_outputs=["report:strobe_style_report"],
        method="feasibility_protocol",
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_study",
        steps=[table_step, report_step],
    )

    bound = bind_context_dependence_authority(plan=plan, context=context)

    assert bound.steps[0].table_one_spec == table_step.table_one_spec
    assert bound.steps[1] == report_step


def test_counts_only_authority_rejects_untyped_descriptive_summaries() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "variance_estimator": "none_counts_only",
                        }
                    }
                )
            )
        }
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_study",
        steps=[
            AnalysisStep(
                step_id="age_distribution",
                planned_analysis_role="auxiliary",
                intent="Summarize age by exposure.",
                method="descriptive_distribution",
                inputs=["artifact:analysis_cohort", "exposure", "age"],
                expected_outputs=["table:distribution_prevalence"],
            )
        ],
    )

    with pytest.raises(DependenceAuthorityError, match="permits only"):
        bind_context_dependence_authority(plan=plan, context=context)


def test_counts_only_audit_cannot_launder_a_prevalence_product() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "variance_estimator": "none_counts_only",
                        }
                    }
                )
            )
        }
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_study",
        steps=[
            AnalysisStep(
                step_id="laundered_prevalence",
                planned_analysis_role="secondary",
                intent="Mislabel a measurement-process audit as prevalence.",
                method="measurement_audit",
                inputs=["artifact:analysis_cohort", "exposure"],
                expected_outputs=["table:distribution_prevalence"],
                measurement_audit_spec={
                    "products": [
                        {
                            "product_id": "distribution_prevalence",
                            "audit": "measurement_process",
                        }
                    ]
                },
            )
        ],
    )

    with pytest.raises(DependenceAuthorityError, match="audit product names"):
        bind_context_dependence_authority(plan=plan, context=context)


def test_counts_only_spec_cannot_launder_a_p_value_output() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "variance_estimator": "none_counts_only",
                        }
                    }
                )
            )
        }
    )
    distribution = _absolute_risk_distribution_step().exposure_outcome_distribution_spec
    assert distribution is not None
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_study",
        steps=[
            AnalysisStep(
                step_id="laundered_test",
                planned_analysis_role="auxiliary",
                intent="Run a prohibited hypothesis test.",
                method="chi_square_test",
                inputs=["artifact:analysis_cohort", "exposure", "death"],
                expected_outputs=[
                    "table:exposure_outcome_distribution",
                    "statistic:p_value",
                ],
                exposure_outcome_distribution_spec=distribution.model_copy(
                    update={"risk_difference_contrast": None}
                ),
            )
        ],
    )

    with pytest.raises(DependenceAuthorityError, match="permits only"):
        bind_context_dependence_authority(plan=plan, context=context)


def test_counts_only_authority_rejects_a_risk_difference() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "variance_estimator": "none_counts_only",
                        }
                    }
                )
            )
        }
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_study",
        steps=[_absolute_risk_distribution_step()],
    )

    with pytest.raises(DependenceAuthorityError, match="forbids risk-difference"):
        bind_context_dependence_authority(plan=plan, context=context)


def test_host_binds_dependence_for_marginal_risks_even_without_a_contrast() -> None:
    context = _context().model_copy(
        update={
            "cohort": _context().cohort.model_copy(
                update={
                    "id_columns": ["patient_stay_id"],
                    "provenance": {
                        "analysis_unit": "icu_stay",
                        "replacement_row_identity": {
                            "output_identity_column": "patient_stay_id",
                            "mapping_file_sha256": "d" * 64,
                            "patient_group_derivation": {
                                "algorithm": "prefix_before_:s",
                                "delimiter": ":s",
                            },
                        },
                    },
                }
            ),
            "user_preferences": UserPreferences(
                covariates=["age"],
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "cluster_unit": "patient",
                            "variance_estimator": "cluster_robust",
                        }
                    }
                ),
            ),
        }
    )
    step = _absolute_risk_distribution_step()
    distribution = step.exposure_outcome_distribution_spec
    assert distribution is not None
    step = step.model_copy(
        update={
            "exposure_outcome_distribution_spec": distribution.model_copy(
                update={"risk_difference_contrast": None}
            )
        }
    )
    bound = bind_context_dependence_authority(
        plan=AnalysisPlan(
            research_question=context.research_question,
            analysis_type="descriptive_study",
            steps=[step],
        ),
        context=context,
    )

    spec = bound.steps[0].exposure_outcome_distribution_spec
    assert spec is not None and spec.risk_difference_contrast is None
    assert spec.dependence is not None
    assert spec.dependence.group_source == "patient_stay_id"
    review = build_plan_scientific_review(
        context=context,
        plan=bound,
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )
    assert review.facts["repeated_unit_design_executable"] is True


def test_one_clustered_step_cannot_mask_an_unclosed_scientific_model() -> None:
    context = _context().model_copy(
        update={
            "cohort": _context().cohort.model_copy(
                update={
                    "id_columns": ["patient_stay_id"],
                    "provenance": {
                        "analysis_unit": "icu_stay",
                        "replacement_row_identity": {
                            "output_identity_column": "patient_stay_id",
                            "mapping_file_sha256": "e" * 64,
                            "patient_group_derivation": {
                                "algorithm": "prefix_before_:s",
                                "delimiter": ":s",
                            },
                        },
                    },
                }
            ),
            "user_preferences": UserPreferences(
                covariates=["age"],
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "cluster_unit": "patient",
                            "variance_estimator": "cluster_robust",
                        }
                    }
                ),
            ),
        }
    )
    distribution = bind_context_dependence_authority(
        plan=AnalysisPlan(
            research_question=context.research_question,
            analysis_type="descriptive_study",
            steps=[_absolute_risk_distribution_step()],
        ),
        context=context,
    ).steps[0]
    unclosed_model = _plan().steps[0]
    plan = _plan().model_copy(update={"steps": [distribution, unclosed_model]})

    assert repeated_unit_design_closed(context, plan) is False
    review = build_plan_scientific_review(
        context=context,
        plan=plan,
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )
    assert "REPEATED_STAY_METHOD_NOT_DECLARED" in {
        finding.code for finding in review.findings
    }


def test_one_step_cannot_close_models_while_leaving_marginal_cis_unclosed() -> None:
    context = _context().model_copy(
        update={
            "cohort": _context().cohort.model_copy(
                update={
                    "id_columns": ["patient_stay_id"],
                    "provenance": {
                        "analysis_unit": "icu_stay",
                        "replacement_row_identity": {
                            "output_identity_column": "patient_stay_id",
                            "mapping_file_sha256": "f" * 64,
                            "patient_group_derivation": {
                                "algorithm": "prefix_before_:s",
                                "delimiter": ":s",
                            },
                        },
                    },
                }
            ),
            "user_preferences": UserPreferences(
                covariates=["age"],
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "cluster_unit": "patient",
                            "variance_estimator": "cluster_robust",
                        }
                    }
                ),
            ),
        }
    )
    bound_model = bind_context_dependence_authority(
        plan=_plan(),
        context=context,
    ).steps[0]
    distribution = _absolute_risk_distribution_step().exposure_outcome_distribution_spec
    assert distribution is not None and distribution.dependence is None
    mixed = bound_model.model_copy(
        update={"exposure_outcome_distribution_spec": distribution}
    )
    plan = _plan().model_copy(update={"steps": [mixed]})

    assert repeated_unit_design_closed(context, plan) is False


def test_planner_parse_binds_cluster_authority_into_the_plan_digest() -> None:
    context = _context().model_copy(
        update={
            "cohort": _context().cohort.model_copy(
                update={
                    "id_columns": ["patient_stay_id"],
                    "provenance": {
                        "analysis_unit": "icu_stay",
                        "replacement_row_identity": {
                            "output_identity_column": "patient_stay_id",
                            "mapping_file_sha256": "b" * 64,
                            "patient_group_derivation": {
                                "algorithm": "prefix_before_:s",
                                "delimiter": ":s",
                            },
                        },
                    },
                }
            ),
            "user_preferences": UserPreferences(
                covariates=["age"],
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "cluster_unit": "patient",
                            "variance_estimator": "cluster_robust",
                        }
                    }
                ),
            ),
        }
    )
    payload = _plan().model_dump(mode="json")
    payload["steps"].append(_traditional_table_one_step().model_dump(mode="json"))
    for step in payload["steps"]:
        step["literature_citation_keys"] = []
        step["literature_design_bindings"] = []
        if step["step_id"] == "missingness":
            step["measurement_audit_spec"] = {
                "products": [
                    {
                        "product_id": "missingness_audit",
                        "audit": "missingness_profile",
                    }
                ]
            }
    payload["steps"][0]["model_requirements"][0]["method_family"] = (
        "statsmodels_logit_mle"
    )
    raw = json.dumps(payload)

    parsed = PlannerAgent.__new__(PlannerAgent)._parse(
        raw,
        context,
        allowed_literature_citation_keys=[],
        direct_comparator_literature_keys=[],
    )

    assert parsed.steps[0].model_requirements[0].dependence is not None
    assert parsed.steps[0].model_requirements[0].dependence.group_source == (
        "patient_stay_id"
    )
    parsed_table_one = parsed.steps[-1].table_one_spec
    assert parsed_table_one is not None
    assert parsed_table_one.schema_version == "easyicu.table_one/2"
    assert parsed_table_one.p_values_required is False
    assert [variable.test for variable in parsed_table_one.variables] == [
        "none_descriptive_smd_only"
    ]


def test_typed_sensitivities_do_not_hide_unclosed_primary_designs() -> None:
    context = _context()
    context = context.model_copy(
        update={
            "variables": [
                *context.variables,
                ConceptDescriptor(
                    name="icu_readmission",
                    role=VariableRole.OTHER,
                    dtype="int64",
                ),
            ],
            "user_preferences": UserPreferences(
                covariates=["age"],
                covariate_selection="exact",
                covariate_rationales={
                    "age": "Age is a prespecified baseline confounder."
                },
                covariate_temporal_roles={"age": "baseline_static"},
                sensitivity_specs=[
                    {
                        "spec_id": "landmark_24h",
                        "axis": "timing",
                        "strategy": "landmark",
                        "landmark_hours": 24,
                        "require_alive_at_landmark": True,
                        "exclude_negative_event_times": True,
                    },
                    {
                        "spec_id": "non_readmission_only",
                        "axis": "repeated_stays",
                        "strategy": "non_readmission_restriction",
                        "execution_variables": ["icu_readmission"],
                    },
                ],
            ),
        }
    )
    base = _plan()
    plan = base.model_copy(
        update={
            "steps": [
                *base.steps,
                AnalysisStep(
                    step_id="landmark_24h",
                    planned_analysis_role="sensitivity",
                    intent="Re-estimate among stays alive at the prespecified landmark.",
                    inputs=["exposure", "death"],
                    expected_outputs=["table:landmark_sensitivity"],
                    method="landmark_analysis",
                    sensitivity_spec_ids=["landmark_24h"],
                ),
                AnalysisStep(
                    step_id="non_readmission_only",
                    planned_analysis_role="sensitivity",
                    intent="Re-estimate after the prespecified readmission restriction.",
                    inputs=["exposure", "death", "icu_readmission"],
                    expected_outputs=["table:non_readmission_sensitivity"],
                    method="non_readmission_restriction",
                    sensitivity_spec_ids=["non_readmission_only"],
                ),
            ]
        }
    )

    review = build_plan_scientific_review(
        context=context,
        plan=plan,
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )

    codes = {item.code for item in review.findings}
    assert "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED" in codes
    # The sensitivity restriction is executable, but it does not change the
    # repeated-stay analysis set of the primary model. Every relevant primary
    # executor must close its own dependence rather than borrowing this step.
    assert "REPEATED_STAY_IDENTITY_UNAVAILABLE" in codes
    assert "REPEATED_STAY_METHOD_NOT_DECLARED" not in codes
    assert "REQUIRED_SENSITIVITY_IS_PROTOCOL_ONLY" not in codes
    assert review.facts["sensitivity"]["executed_spec_ids"] == [
        "landmark_24h",
        "non_readmission_only",
    ]


def test_declared_and_materialized_time_anchor_mismatch_is_study_owned() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                covariates=["age"],
                timing_and_design='{"anchor":"event onset"}',
            )
        }
    )

    review = build_plan_scientific_review(
        context=context,
        plan=_plan(),
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )

    finding = next(
        item
        for item in review.findings
        if item.code == "PRIMARY_EXPOSURE_TIME_ANCHOR_MISMATCH"
    )
    assert finding.severity == "blocker"
    assert finding.remediation_route == "study_authority_change"
    assert finding.requires_user_authorization is True
    assert review.facts["primary_exposure_time_anchor_alignment"] == {
        "status": "mismatch",
        "primary_exposure": "exposure",
        "declared_anchor": "event_onset",
        "definition_anchor": "icu_admission",
        "observation_window_anchor": "icu_admission",
        "observation_window_role": "exposure_definition",
        "declared_source": "user_preferences.timing_and_design.anchor",
        "definition_source": "variables.exposure.analysis_window",
        "observation_window_source": "variables.exposure.analysis_window",
    }


def test_typed_literature_route_joins_to_sealed_source_evidence() -> None:
    context = _context()
    review = build_plan_scientific_review(
        context=context,
        plan=_plan(),
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )

    rows = review.facts["literature_design_bindings"]["steps"][0]["citations"]
    direct = next(row for row in rows if row["citation_key"] == "direct_2024")
    assert direct["design_elements"] == ["estimand"]
    assert "Compare population" in direct["application"]
    assert direct["source_excerpt"].startswith("Direct comparator")
    assert direct["binding_status"] == "typed_source_joined"


def test_citation_presence_without_design_binding_remains_a_major_finding() -> None:
    context = _context()
    review = build_plan_scientific_review(
        context=context,
        plan=_plan(typed_bindings=False),
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )

    finding = next(
        item
        for item in review.findings
        if item.code == "LITERATURE_DESIGN_ROUTE_NOT_EXPLICIT"
    )
    assert finding.severity == "major"
    assert review.facts["literature_design_bindings"]["unresolved_steps"] == [
        "primary_model"
    ]


def test_exact_adjustment_requires_user_bound_rationale_and_timing() -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                covariates=["age"],
                covariate_selection="exact",
            )
        }
    )
    incomplete = build_plan_scientific_review(
        context=context,
        plan=_plan(),
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )
    assert "ADJUSTMENT_RATIONALE_OR_TIMING_UNBOUND" in {
        item.code for item in incomplete.findings
    }

    complete_context = context.model_copy(
        update={
            "user_preferences": UserPreferences(
                covariates=["age"],
                covariate_selection="exact",
                covariate_rationales={
                    "age": "Age is a clinically plausible baseline confounder."
                },
                covariate_temporal_roles={"age": "baseline_static"},
            )
        }
    )
    complete = build_plan_scientific_review(
        context=complete_context,
        plan=_plan(),
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(complete_context),
    )
    assert "ADJUSTMENT_RATIONALE_OR_TIMING_UNBOUND" not in {
        item.code for item in complete.findings
    }


def test_explicit_distribution_figure_covers_descriptive_role_without_prose() -> None:
    context = _context()
    plan = _plan().model_copy(
        update={
            "steps": [
                *_plan().steps,
                AnalysisStep(
                    step_id="exposure_outcome_panel",
                    planned_analysis_role="auxiliary",
                    intent="Render the result.",
                    method="visualization",
                    inputs=["table:exposure_outcome_distribution"],
                    expected_outputs=["figure:exposure_outcome_panel"],
                    figure_panels=[
                        PlannedFigurePanelSpec(
                            panel_id="exposure_outcome_panel",
                            figure_output="figure:exposure_outcome_panel",
                            article_role="descriptive_result",
                            chart_type="prevalence_panel",
                            source_products=["table:exposure_outcome_distribution"],
                        )
                    ],
                ),
            ]
        }
    )
    review = build_plan_scientific_review(
        context=context,
        plan=plan,
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )

    assert "descriptive_result" in review.facts["figure_roles"]["covered_roles"]
    assert "descriptive_result" not in review.facts["figure_roles"]["missing_roles"]


def test_descriptive_hero_figure_must_descend_from_primary_result() -> None:
    context = _context()
    primary = AnalysisStep(
        step_id="primary_distribution",
        planned_analysis_role="primary",
        intent="Estimate the primary descriptive result.",
        method="descriptive",
        expected_outputs=["table:exposure_outcome_distribution"],
    )
    secondary = AnalysisStep(
        step_id="age_distribution",
        planned_analysis_role="secondary",
        intent="Describe age by group.",
        method="descriptive_distribution",
        expected_outputs=["table:distribution_prevalence"],
    )
    off_lineage_figure = AnalysisStep(
        step_id="age_distribution_figure",
        planned_analysis_role="auxiliary",
        intent="Render age by group.",
        method="visualization",
        inputs=["table:distribution_prevalence"],
        expected_outputs=["figure:age_distribution"],
        figure_panels=[
            PlannedFigurePanelSpec(
                panel_id="grouped_distribution",
                figure_output="figure:age_distribution",
                article_role="distribution",
                chart_type="point_range",
                source_products=["table:distribution_prevalence"],
            )
        ],
    )
    plan = AnalysisPlan(
        research_question="Describe the primary exposure and outcome.",
        analysis_type="descriptive_epidemiology",
        steps=[primary, secondary, off_lineage_figure],
    )
    strategy = build_article_figure_strategy(
        context,
        analysis_family="descriptive",
    )

    missing = build_plan_scientific_review(
        context=context,
        plan=plan,
        literature=_literature(),
        figure_strategy=strategy,
    )
    assert "distribution" in missing.facts["figure_roles"]["missing_roles"]

    on_lineage_figure = off_lineage_figure.model_copy(
        update={
            "step_id": "primary_distribution_figure",
            "inputs": ["table:exposure_outcome_distribution"],
            "expected_outputs": ["figure:primary_distribution"],
            "figure_panels": [
                PlannedFigurePanelSpec(
                    panel_id="primary_distribution",
                    figure_output="figure:primary_distribution",
                    article_role="distribution",
                    chart_type="prevalence_panel",
                    source_products=["table:exposure_outcome_distribution"],
                )
            ],
        }
    )
    closed_plan = plan.model_copy(
        update={"steps": [primary, secondary, off_lineage_figure, on_lineage_figure]}
    )
    closed = build_plan_scientific_review(
        context=context,
        plan=closed_plan,
        literature=_literature(),
        figure_strategy=strategy,
    )
    assert "distribution" in closed.facts["figure_roles"]["covered_roles"]
    assert "distribution" not in closed.facts["figure_roles"]["missing_roles"]


def test_generic_overview_inputs_do_not_infer_figure_roles_or_chart_breadth() -> None:
    context = _context()
    plan = _plan().model_copy(
        update={
            "steps": [
                *_plan().steps,
                AnalysisStep(
                    step_id="overview",
                    planned_analysis_role="auxiliary",
                    intent="Render a generic overview.",
                    method="visualization",
                    inputs=[
                        "table:cohort_flow",
                        "table:missingness_audit",
                        "table:exposure_outcome_distribution",
                    ],
                    expected_outputs=["figure:overview"],
                ),
            ]
        }
    )

    review = build_plan_scientific_review(
        context=context,
        plan=plan,
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )

    facts = review.facts["figure_roles"]
    assert facts["typed_panel_count"] == 1
    assert facts["covered_roles"] == ["primary_estimand"]
    assert facts["chart_types"] == ["forest"]
    assert facts["distinct_chart_types_complete"] is False
    codes = {item.code for item in review.findings}
    assert "FIGURE_ROLE_COVERAGE_INCOMPLETE" in codes
    assert "FIGURE_CHART_TYPES_TOO_NARROW" in codes


def test_revision_contract_contains_only_agent_owned_findings() -> None:
    context = _context()
    review = build_plan_scientific_review(
        context=context,
        plan=_plan(),
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )

    contract = render_agent_plan_revision_contract(review)

    assert "CONTINUOUS_COVARIATE_FUNCTIONAL_FORM_UNCHECKED" in contract
    assert "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED" not in contract
    assert "ADJUSTMENT_SET_NOT_USER_CONFIRMED" not in contract
    assert "preserve the exact research question" in contract


def test_automated_clinical_definition_does_not_impersonate_clinician_review() -> None:
    context = _context()
    variables = list(context.variables)
    variables[0] = variables[0].model_copy(
        update={
            "clinical_definition": ClinicalDefinitionReference(
                contract_id="test_definition_v1",
                definition="Test ICU phenotype",
                version="1",
                source_id="PMID:1",
                definition_time_anchor="icu_admission",
                status="source_bound_golden",
                validation_status=(
                    "automated_golden; independent_clinical_review_pending"
                ),
                canonical_definition=True,
                ascertainment_limitations=["Manual chart semantics not reviewed."],
                database_conformance={"miiv": "mapping_only"},
            )
        }
    )
    context = context.model_copy(update={"variables": variables})

    review = build_plan_scientific_review(
        context=context,
        plan=_plan(),
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )

    finding = next(
        item
        for item in review.findings
        if item.code == "CLINICAL_DEFINITION_INDEPENDENT_REVIEW_PENDING"
    )
    assert finding.severity == "major"
    assert finding.remediation_route == "independent_review"
    assert review.facts["clinical_definitions"][
        "independent_clinical_review_pending_contracts"
    ] == ["test_definition_v1"]
    assert "CLINICAL_DEFINITION_INDEPENDENT_REVIEW_PENDING" not in (
        render_agent_plan_revision_contract(review)
    )
    conformance = next(
        item
        for item in review.findings
        if item.code == "CLINICAL_DEFINITION_DATABASE_CONFORMANCE_NOT_ESTABLISHED"
    )
    assert conformance.severity == "major"
    assert conformance.remediation_route == "independent_review"
    assert review.facts["clinical_definitions"]["database_conformance_gaps"] == [
        {
            "variable": "exposure",
            "contract_id": "test_definition_v1",
            "database": "miiv",
            "conformance": "mapping_only",
        }
    ]
    assert "CLINICAL_DEFINITION_DATABASE_CONFORMANCE_NOT_ESTABLISHED" not in (
        render_agent_plan_revision_contract(review)
    )
