"""A synthetic survival question with no declared design, shared by its tests.

Test modules may not import one another (``tests/governance/test_test_organization.py``).
The proposal planner, its scientific review and the Web compiler that declares
the reviewed design all start from this one study: renal replacement therapy
and 90-day mortality, which no benchmark item asks.  Zero patient rows.
"""

from __future__ import annotations

import json

from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
    candidate_analysis_types,
    select_progressive_variables,
)
from easyicu.research_agent.planning.family_spec import build_family_spec_request
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    TimeWindow,
    UserPreferences,
    VariableRole,
)

ALLOWED = ["strobe_2007", "record_2015"]
BINARY = {"n_unique": 2, "is_binary": True, "levels": [0, 1]}
QUESTION = (
    "Among adult ICU stays, is renal replacement therapy associated with the time to "
    "death within 90 days in a time-respecting survival analysis?"
)
AGE = {
    "name": "age", "coding": "continuous", "reference_level_index": None,
    "clinical_rationale": "Age precedes admission and drives both renal support and death.",
}
SEX = {
    "name": "sex", "coding": "binary", "reference_level_index": 0,
    "clinical_rationale": "Sex is fixed at admission and associated with both therapy use and death.",
}


def survival_context(**update) -> ResearchContext:
    base = ResearchContext(
        research_question=QUESTION,
        cohort=CohortDescriptor(
            cohort_name="synthetic_rrt", database="miiv", n_stays=0, id_columns=["stay_id"],
            outcome_columns=["mort_90d"],
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            ConceptDescriptor(name="rrt", description="renal replacement therapy",
                              role=VariableRole.INTERVENTION, dtype="float64", observed_domain=BINARY),
            ConceptDescriptor(name="mort_90d", description="Death by day 90 from ICU admission",
                              role=VariableRole.OUTCOME, dtype="float64", observed_domain=BINARY),
            ConceptDescriptor(name="followup_days_90d", description="90-day event or censoring time",
                              role=VariableRole.OTHER, dtype="float64", unit="days"),
            ConceptDescriptor(name="age", description="age", role=VariableRole.DEMOGRAPHIC,
                              dtype="float64", unit="years"),
            ConceptDescriptor(name="sex", description="sex", role=VariableRole.DEMOGRAPHIC,
                              dtype="float64", observed_domain=BINARY),
        ],
        target_outcome="mort_90d",
        primary_exposure="rrt",
        time_windows=[
            TimeWindow(name="icu_admission_0_24h", anchor="icu_admission", start_hours=0.0, end_hours=24.0)
        ],
        user_preferences=UserPreferences(inferred_analysis_family="survival"),
    )
    return base.model_copy(update=update)


def survival_request(context: ResearchContext, planning_contract_context: str = ""):
    return build_family_spec_request(
        context,
        analysis_types=candidate_analysis_types(context),
        variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=ALLOWED,
        required_primary_cohort_selection_mode="all_input_rows",
        planning_contract_context=planning_contract_context,
    )


def survival_spec(request, roster: list[dict], *, labels: dict | None = None) -> dict:
    keys = [*request.required_reader_label_keys, *request.level_label_keys, *(item["name"] for item in roster)]
    labels = labels if labels is not None else {key: f"Reader label for {key}" for key in keys}
    return {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": roster,
        "reader_display_labels": [{"key": key, "value": value} for key, value in labels.items()],
        "comparator_applications": [],
        "roster_decision_note": "Adjust for the baseline factors that drive both therapy and death.",
    }


def proposed_survival_plan(context: ResearchContext, spec: dict):
    """One scripted Planner answer compiled into the proposed-suite plan."""

    llm = ScriptedMockLLMClient([json.dumps(spec)])
    result = ProgressivePlannerAgent(llm).run_attempt(
        context, planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=ALLOWED, direct_comparator_literature_keys=[],
        enforce_article_contract=True, article_contract_context=context,
        planning_contract_context="", required_primary_cohort_selection_mode="all_input_rows",
    )
    return result.output, llm
