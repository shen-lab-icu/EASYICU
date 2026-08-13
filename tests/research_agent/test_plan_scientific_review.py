"""Focused contracts for the digest-bound pre-approval review owner."""

from __future__ import annotations

from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.contracts.model_terms import ModelTermSpec
from easyicu.research_agent.literature import (
    CitationRecord,
    LiteratureBundle,
    LiteratureScreeningDecision,
    LiteratureSearchProvenance,
)
from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    render_agent_plan_revision_contract,
    render_plan_scientific_guardrails,
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
            ConceptDescriptor(
                name="death", role=VariableRole.OUTCOME, dtype="int64"
            ),
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
        "REQUIRED_SENSITIVITY_IS_PROTOCOL_ONLY",
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
    assert "not assessed until execution" in (
        review.facts["score_interpretation"]["figures"]
    )


def test_typed_sensitivity_steps_close_timing_and_nonreadmission_authority() -> None:
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
    assert "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED" not in codes
    assert "REPEATED_STAY_IDENTITY_UNAVAILABLE" not in codes
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
        if item.code
        == "CLINICAL_DEFINITION_DATABASE_CONFORMANCE_NOT_ESTABLISHED"
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
