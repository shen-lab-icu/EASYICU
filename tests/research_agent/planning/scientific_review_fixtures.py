"""Shared typed study and descriptive-product fixtures for scientific review."""

from __future__ import annotations

from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.contracts.model_terms import ModelTermSpec
from easyicu.research_agent.contracts.figure_plan import PlannedFigurePanelSpec
from easyicu.research_agent.literature import (
    CitationRecord,
    LiteratureBundle,
    LiteratureScreeningDecision,
    LiteratureSearchProvenance,
)
from easyicu.research_agent.contracts.claim_ceiling import DescriptiveClaimContract
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    LiteratureDesignBinding,
    PlannedModelRequirement,
    CohortDescriptor,
    ConceptDescriptor,
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
            "missing_outcome_policy": "fail_closed",
            "risk_difference_contrast": {
                "reference_exposure_level": 0,
                "comparison_exposure_level": 1,
            },
            "confidence_level": 0.95,
        },
    )


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
