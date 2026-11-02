"""Synthetic typed population foundation and context fixtures."""
from __future__ import annotations
from typing import Any
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanFoundation
from easyicu.research_agent.schema import CohortDescriptor, ConceptDescriptor, ResearchContext, VariableRole


_QUESTION = "Among older adults with shock, which heart-rate patterns occur in the first day?"

def _context() -> ResearchContext:
    return ResearchContext(
        research_question=_QUESTION,
        cohort=CohortDescriptor(
            cohort_name="source_cohort",
            database="synthetic",
            n_stays=0,
            id_columns=["stay_id"],
            outcome_columns=["death"],
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            ConceptDescriptor(name="age_years", role=VariableRole.DEMOGRAPHIC, dtype="float64"),
            ConceptDescriptor(name="shock_flag", role=VariableRole.OTHER, dtype="float64"),
            ConceptDescriptor(name="heart_rate", role=VariableRole.VITAL, dtype="float64"),
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="float64"),
        ],
        target_outcome="death",
    )

def _predicate(concept: str, op: str, number: float, aggregation: str = "max") -> dict:
    return {
        "concept_id": concept,
        "anchor": "icu_admission",
        "start_offset_hours": 0,
        "end_offset_hours": 24,
        "aggregation": aggregation,
        "op": op,
        "value": {
            "mode": "number",
            "string_value": None,
            "number_value": number,
            "boolean_value": None,
            "string_list": [],
            "number_list": [],
        },
    }

def _foundation(cohort: dict[str, Any]) -> ProgressivePlanFoundation:
    return ProgressivePlanFoundation.model_validate(
        {"cohort": cohort, "display_labels": [], "robustness_intents": [], "know_how_decisions": []}
    )
