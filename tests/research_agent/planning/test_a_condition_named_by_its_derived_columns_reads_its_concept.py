"""A condition named by the columns derived from its concept reads the concept (B33b).

UX leg 3 on rc_a12: the reviewed candidate applied 「脓毒症 ICU 患者」 as its
event concept present in [0, 24) h.  The execution run planned again on the
prepared export, where the planner named the condition by the export's derived
columns (``x_max``, ``x_n``, ``x_measured``, ``x_first_time``).  Each was read
as a status of its own, the timestamp is none, the criterion was not applied,
and the run stopped on the caller's bound population.  Presence is a property
of the concept: each derived column now stands for the concept it derives
from.  A name with no other source keeps its own reading.  Synthetic contexts
with a concept other than sepsis.
"""

from __future__ import annotations

import json

from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)

_BINARY = {"n_unique": 2, "is_binary": True, "levels": [0, 1], "min": 0.0, "max": 1.0}
_COUNT = {"n_unique": 5, "is_binary": False, "min": 0.0, "max": 4.0}
_DAY = {"start_hours": 0, "end_hours": 24}
_WINDOW = "icu_admission[0,24]h"


def _derived(concept: str) -> list[ConceptDescriptor]:
    def column(suffix, role, dtype, domain, transform):
        return ConceptDescriptor(
            name=f"{concept}_{suffix}", role=role, dtype=dtype, observed_domain=domain,
            source_concept=concept, unit_normalization=transform, analysis_window=_WINDOW,
        )

    return [
        column("max", VariableRole.OTHER, "int64", _BINARY, "window_presence_max"),
        column("n", VariableRole.META, "int64", _COUNT, "window_event_count"),
        column("measured", VariableRole.META, "int64", _BINARY, "window_measured"),
        column("first_time", VariableRole.TIME, "float64", None, "window_first_time"),
    ]


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Which stays does the study include?",
        cohort=CohortDescriptor(cohort_name="synthetic", database="miiv", n_stays=140,
                                id_columns=["stay_id"], outcome_columns=["death"]),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="int64", observed_domain=_BINARY),
            ConceptDescriptor(name="stay_hours", role=VariableRole.OTHER, dtype="float64",
                              observed_domain={"n_unique": 90, "is_binary": False, "min": 1.0, "max": 400.0}),
            *_derived("mech_vent"),
            *_derived("vaso"),
        ],
        target_outcome="death",
        user_preferences=UserPreferences(data_constraints=json.dumps({"materialization_window": {
            "role": "outer_observation_window", "anchor": "ICU admission", "hours": 24.0}})),
    )


def _compiled(*names: str):
    spec = PopulationSpec.model_validate({"criteria": [{
        "id": "c1", "quote": "ventilated ICU patients", "source": "question", "kind": "condition_present",
        "concepts_all_of": list(names), "window": _DAY, "role": "include", "missing_policy": "exclude",
    }]})
    compiled = compile_population(spec, _context(), time_zero_hours=None)
    definition = compiled.cohort_definition()
    return compiled.criteria[0], definition.selection_mode, [
        (predicate.concept_id, predicate.op, predicate.value, predicate.aggregation) for predicate in definition.inclusion
    ]


def test_the_derived_columns_of_one_concept_apply_its_presence() -> None:
    item, mode, predicates = _compiled("mech_vent_first_time", "mech_vent_max", "mech_vent_measured", "mech_vent_n")

    assert item.applied, item.reason
    assert mode == "predicate_filtered"
    assert predicates == [("mech_vent", "==", 1, "max")]


def test_the_concept_and_one_derived_column_state_the_same_predicate() -> None:
    assert _compiled("mech_vent")[2] == _compiled("mech_vent_max")[2] == [("mech_vent", "==", 1, "max")]


def test_two_concepts_keep_one_predicate_each() -> None:
    item, _mode, predicates = _compiled("mech_vent_n", "vaso_first_time", "vaso_max")

    assert item.applied, item.reason
    assert predicates == [("mech_vent", "==", 1, "max"), ("vaso", "==", 1, "max")]


def test_a_column_with_no_other_source_keeps_its_own_reading() -> None:
    item, mode, predicates = _compiled("stay_hours")

    assert not item.applied
    assert item.reason == "population_condition_column_not_status"
    assert mode == "all_input_rows" and predicates == []


def _conflict(*names: str):
    from types import SimpleNamespace

    from easyicu.research_agent.planning.family_spec.request import caller_bound_population_conflict

    spec = PopulationSpec.model_validate({"criteria": [{
        "id": "c1", "quote": "ventilated ICU patients", "source": "question", "kind": "condition_present",
        "concepts_all_of": list(names), "window": _DAY, "role": "include", "missing_policy": "exclude",
    }]})
    context = _context()
    plan = SimpleNamespace(cohort=compile_population(spec, context).cohort_definition(), analysis_type="association_study")
    return caller_bound_population_conflict(plan, context=context, required_selection_mode="predicate_filtered")


def test_the_execution_plan_keeps_the_population_the_reviewed_candidate_bound() -> None:
    # The candidate bound its applied condition; the execution plan that names
    # it by derived columns now states the same population, so prepare runs.
    assert _conflict("mech_vent_first_time", "mech_vent_max", "mech_vent_measured", "mech_vent_n") is None


def test_a_plan_that_applies_no_condition_is_still_refused() -> None:
    conflict = _conflict("stay_hours")

    assert conflict is not None and "expected 'predicate_filtered', observed 'all_input_rows'" in conflict
