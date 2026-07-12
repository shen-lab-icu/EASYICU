"""Structured-纳排 contract: an analysis cohort must be expressed as typed
inclusion/exclusion predicates, not left as free-text step prose.

Regression for the E1 deepseek-v4-flash finding: the planner left
``plan.cohort`` empty (inclusion/exclusion = []) while defining the cohort
only in step intents, so the 纳排 was never materialised/enforced and the
primary model silently ran on the full universe.
"""

from __future__ import annotations

from easyicu.research_agent.cohort_schema import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
)
from easyicu.research_agent.plan_utils import (
    _cohort_definition_contract_findings,
    _cohort_definition_is_empty,
    _plan_expects_analysis_cohort,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _plan(
    cohort: CohortDefinition,
    *,
    step_intent: str,
    step_id: str,
    method: str = "descriptive",
    expected_outputs: list[str] | None = None,
) -> AnalysisPlan:
    return AnalysisPlan(
        research_question="q",
        cohort=cohort,
        steps=[
            AnalysisStep(
                step_id=step_id,
                intent=step_intent,
                method=method,
                expected_outputs=expected_outputs or [],
            )
        ],
    )


def test_empty_cohort_with_cohort_step_is_error():
    plan = _plan(
        CohortDefinition(name="primary"),  # no predicates
        step_id="01_cohort_definition",
        step_intent="Define the adult ICU cohort with LoS >= 1 day.",
        method="cohort_definition",
        expected_outputs=["table:cohort_flow"],
    )
    assert _plan_expects_analysis_cohort(plan)
    assert _cohort_definition_is_empty(plan)
    findings = _cohort_definition_contract_findings(plan)
    assert len(findings) == 1
    assert findings[0].validator == "cohort_contract"
    assert findings[0].severity == "error"


def test_structured_cohort_passes():
    cohort = CohortDefinition(
        name="primary",
        inclusion=(
            ConceptPredicate("age", TimeWindow("icu_admit", 0, 24), "first", ">=", 18),
        ),
    )
    plan = _plan(
        cohort,
        step_id="01_cohort_definition",
        step_intent="Define the adult ICU cohort.",
    )
    assert not _cohort_definition_is_empty(plan)
    assert _cohort_definition_contract_findings(plan) == []


def test_no_cohort_step_does_not_require_structured_cohort():
    """A legitimate whole-universe analysis (no cohort/eligibility step) must
    not be forced to declare a structured cohort."""
    plan = _plan(
        CohortDefinition(name="primary"),
        step_id="01_describe_universe",
        step_intent="Describe all ICU stays in the export.",
    )
    assert not _plan_expects_analysis_cohort(plan)
    assert _cohort_definition_contract_findings(plan) == []


def test_treatment_eligibility_bias_prose_is_not_a_cohort_owner():
    plan = _plan(
        CohortDefinition(name="primary"),
        step_id="05_primary_effect",
        step_intent=(
            "Estimate the adjusted effect and address treatment eligibility bias "
            "with inverse-probability weighting."
        ),
        method="iptw",
        expected_outputs=["statistic:adjusted_effect"],
    )
    assert not _plan_expects_analysis_cohort(plan)
    assert _cohort_definition_contract_findings(plan) == []
