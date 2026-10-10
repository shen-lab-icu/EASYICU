"""A population criterion over a concept without values is not applied.

A predicate over a column that holds no value reads nothing: an inclusion keeps
no stay, an exclusion excludes none, while the plan states the criterion as
applied.  The population owner refuses it, with its own reason, whether the
planning context states that its source holds no value of the concept
(``contracts.concept_values``) or the context's rows show the column empty.
An exposure grouping reads its rules through the same owner, so it is refused
the same way, in its own code.  Synthetic contexts only.
"""

from __future__ import annotations

import json

from easyicu.research_agent.contracts.concept_values import CONCEPTS_WITHOUT_VALUES_KEY
from easyicu.research_agent.planning.exposure_group_compile import (
    compile_exposure_groupings,
)
from easyicu.research_agent.planning.exposure_group_spec import (
    read_stated_exposure_groupings,
)
from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    MissingnessProfile,
    ResearchContext,
    UserPreferences,
    VariableRole,
)

_BINARY = {"n_unique": 2, "is_binary": True, "levels": [0, 1]}
_DAY = {"start_hours": 0, "end_hours": 24}
_STAYS = 120


def _glucose(*, empty: bool) -> ConceptDescriptor:
    return ConceptDescriptor(
        name="glu_max",
        role=VariableRole.LAB,
        dtype="float64",
        unit="mg/dL",
        source_concept="glu",
        unit_normalization="window_numeric_max",
        analysis_window="icu_admission[0,24]h",
        valid_range=[0, 1000],
        missingness=MissingnessProfile(
            fraction_missing=1.0 if empty else 0.1,
            n_missing=_STAYS if empty else 12,
            n_total=_STAYS,
            missingness_severity="high" if empty else "medium",
            missingness_test="not_run",
        ),
    )


def _context(*, stated: tuple[str, ...] = (), empty: bool = False) -> ResearchContext:
    return ResearchContext(
        research_question="Which stays does the study include?",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="miiv",
            n_stays=_STAYS,
            id_columns=["stay_id"],
            outcome_columns=["death"],
            provenance=(
                {CONCEPTS_WITHOUT_VALUES_KEY: list(stated)} if stated else {}
            ),
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            ConceptDescriptor(
                name="death", role=VariableRole.OUTCOME, dtype="int64", observed_domain=_BINARY
            ),
            ConceptDescriptor(
                name="circ", role=VariableRole.OTHER, dtype="int64", observed_domain=_BINARY
            ),
            _glucose(empty=empty),
        ],
        target_outcome="death",
        user_preferences=UserPreferences(
            data_constraints=json.dumps(
                {
                    "materialization_window": {
                        "role": "outer_observation_window",
                        "anchor": "ICU admission",
                        "hours": 24.0,
                    }
                }
            )
        ),
    )


def _one(criterion: dict, context: ResearchContext):
    spec = PopulationSpec.model_validate(
        {"criteria": [{"id": "c1", "quote": "the stated criterion", "source": "question", **criterion}]}
    )
    (item,) = compile_population(spec, context, time_zero_hours=None).criteria
    return item


_EXCLUDED_CONDITION = {
    "kind": "condition_present",
    "concepts_all_of": ["circ"],
    "window": _DAY,
    "role": "exclude",
}
_HIGH_GLUCOSE = {
    "kind": "measurement",
    "concept": "glu",
    "summary": "max",
    "window": _DAY,
    "op": ">",
    "value": 180,
    "unit": "mg/dL",
    "role": "include",
}


def test_an_exclusion_over_a_concept_its_source_holds_no_value_of_is_not_applied() -> None:
    applied = _one(_EXCLUDED_CONDITION, _context())
    assert (applied.disposition, applied.side) == ("applied_by_plan", "exclusion")

    refused = _one(_EXCLUDED_CONDITION, _context(stated=("circ",)))

    assert (refused.disposition, refused.reason) == (
        "not_applied",
        "population_concept_without_values",
    )
    assert "'circ'" in refused.detail and refused.predicates == ()


def test_a_concept_the_source_states_without_values_is_refused_before_its_column() -> None:
    # The source's statement decides before any column of the concept is read.
    refused = _one(_HIGH_GLUCOSE, _context(stated=("glu",)))

    assert (refused.disposition, refused.reason) == (
        "not_applied",
        "population_concept_without_values",
    )
    assert "source lists 'glu'" in refused.detail


def test_a_criterion_over_a_column_no_row_holds_a_value_of_is_not_applied() -> None:
    assert _one(_HIGH_GLUCOSE, _context()).disposition == "applied_by_plan"

    refused = _one(_HIGH_GLUCOSE, _context(empty=True))

    assert (refused.disposition, refused.reason) == (
        "not_applied",
        "population_concept_without_values",
    )
    assert "'glu_max'" in refused.detail


def test_an_exposure_grouping_over_a_column_without_values_is_not_applied() -> None:
    groupings = read_stated_exposure_groupings(
        {
            "groupings": [
                {
                    "id": "x1",
                    "concept": "glu",
                    "window": _DAY,
                    "scale": "nominal",
                    "groups": [
                        {"id": "g1", "label": "hyperglycaemia",
                         "rule": {"summary": "max", "op": ">", "value": 180, "unit": "mg/dL"}},
                        {"id": "g2", "label": "no hyperglycaemia", "rule": "otherwise"},
                    ],
                    "unmeasured": {"handling": "own_group", "label": "no glucose on day one"},
                    "quote": "glucose above 180",
                    "source": "question",
                }
            ]
        }
    )

    (grouping,) = compile_exposure_groupings(groupings, _context(empty=True)).groupings

    assert (grouping.disposition, grouping.reason) == (
        "not_applied",
        "exposure_group_concept_without_values",
    )
