"""A trajectory design keeps the population criteria its plan does not apply.

A plan states each restriction the question places on whom the study
includes; one that no allowed cohort concept expresses is applied by nothing,
and the plan's cohort carries it so that its review reports it.  A trajectory
plan is reviewed again on the signed owners, whose cohort is the sealed
design's population, and the design kept only the predicates.  The criterion
was therefore reported on the plan that handed the question over and lost on
the plan a researcher approves.  The design now carries it, outside the
population it seals, and the plan on the signed owners states it.  Fixtures
are generic.
"""

from __future__ import annotations

from typing import Any

import pytest

from easyicu.research_agent.contracts.trajectory_design import (
    TRAJECTORY_PRIMARY_ACTION,
    TrajectoryDesignError,
    load_trajectory_design,
    normalize_trajectory_design,
    sealed_trajectory_authority_body,
    trajectory_population_design,
    trajectory_window_design,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.planning.cohort_contract import CohortDefinition
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    build_trajectory_scientific_runtime_authority,
)
from easyicu.webserver.pi_copilot.plan_decisions import compile_agent_plan_configuration
from easyicu.webserver.trajectory_runtime_projection import (
    WebScientificRuntimeProjectionError,
    validate_trajectory_design_declaration,
)

_OWNER = "TRAJECTORY_LONGITUDINAL_OWNER_NOT_SEALED"
_FINDING = "POPULATION_CRITERION_NOT_APPLIED"
_COORDINATES = ["sofa2_resp", "sofa2_cardio"]
_STUDY = {
    "cohort": {"preset": "all_icu"},
    "confirmations": {"feature_time_window": True},
    "sensitivity_specs": [],
}
_ROLES = {
    "stay_id": VariableRole.ID,
    "age": VariableRole.DEMOGRAPHIC,
    "sofa2_resp": VariableRole.ORDINAL_SCORE,
    "sofa2_cardio": VariableRole.ORDINAL_SCORE,
    "death": VariableRole.OUTCOME,
}
_QUESTION = (
    "Among adults after cardiac surgery, do first-24h physiologic trajectories "
    "form distinct classes?"
)
_ADULT = {
    "concept_id": "age",
    "time_window": {"anchor": "icu_admit", "start_offset_hours": 0, "end_offset_hours": 24},
    "aggregation": "first",
    "op": ">=",
    "value": 18,
}
_SURGERY = "after cardiac surgery"


def _design(**population: Any) -> dict:
    return normalize_trajectory_design(
        {
            "coordinate_concepts": _COORDINATES,
            "window_end_hours": 24,
            "grid_width_hours": 4,
            **({"population": population} if population else {}),
        }
    )


def _body(design: dict) -> dict:
    return sealed_trajectory_authority_body(
        load_trajectory_design(design), protocol_content_sha256="1" * 64
    )


def _authority(design: dict):
    return build_trajectory_scientific_runtime_authority(_body(design))


def _context() -> ResearchContext:
    return ResearchContext(
        research_question=_QUESTION,
        cohort=CohortDescriptor(
            cohort_name="trajectory_criteria_fixture", database="synthetic",
            n_stays=0, id_columns=["stay_id"], outcome_columns=["death"],
        ),
        variables=[
            ConceptDescriptor(name=name, role=role, dtype="float64")
            for name, role in _ROLES.items()
        ],
        target_outcome="death",
    )


def _handoff_review(cohort: dict):
    """The review of a plan that hands a trajectory question to the signed owners."""

    step = AnalysisStep(
        step_id="trajectory_solution",
        planned_analysis_role="primary",
        intent="Cluster the prespecified coordinates into trajectory classes.",
        inputs=["stay_id", *_COORDINATES, "artifact:analysis_cohort"],
        expected_outputs=["table:phenotype_assignments"],
        method="prespecified_trajectory_feature_clustering",
        scientific_action_id=TRAJECTORY_PRIMARY_ACTION,
    )
    plan = AnalysisPlan(
        research_question=_QUESTION,
        analysis_type="trajectory_clustering",
        steps=[step],
        cohort=cohort,
    )
    return build_plan_scientific_review(context=_context(), plan=plan, literature=None)


def _compile(facts) -> dict:
    return compile_agent_plan_configuration(
        study=dict(_STUDY),
        agent_plan={"steps": []},
        runtime_finding_codes=(_OWNER,),
        patient_cluster_available=False,
        review_facts=facts,
    ).patch


# The design owner -------------------------------------------------------


def test_a_design_keeps_the_criteria_beside_the_population_it_seals() -> None:
    design = _design(inclusion=[_ADULT], unapplied_criteria=[_SURGERY])

    assert design["population"] == {
        "inclusion": [_ADULT],
        "exclusion": [],
        "unapplied_criteria": [_SURGERY],
    }
    typed = load_trajectory_design(design)
    assert typed.population_unapplied_criteria == (_SURGERY,)
    # They select no stay: the sealed population is the predicates alone.
    assert typed.population_definition == {
        "name": "primary",
        "selection_mode": "predicate_filtered",
        "inclusion": [_ADULT],
        "exclusion": [],
    }
    assert normalize_trajectory_design(design) == design


def test_a_design_with_criteria_only_seals_no_population() -> None:
    design = _design(unapplied_criteria=[_SURGERY])

    assert design["population"] == {
        "inclusion": [],
        "exclusion": [],
        "unapplied_criteria": [_SURGERY],
    }
    typed = load_trajectory_design(design)
    assert typed.population_definition is None
    body = _body(design)
    assert "population" not in body
    assert body["unapplied_population_criteria"] == [_SURGERY]
    # A study that states a population filter is still refused: this design
    # seals none, so its owners keep every stay.
    with pytest.raises(WebScientificRuntimeProjectionError) as caught:
        validate_trajectory_design_declaration(
            {**_STUDY, "cohort": {"preset": "adult_all"}, "trajectory_design": design,
             "analysis_design": {"analysis_family": "trajectory_clustering",
                                 "analysis_unit": "icu_stay",
                                 "variance_estimator": "model_based"}}
        )
    assert caught.value.code == "web_trajectory_population_filter_unsupported"


@pytest.mark.parametrize("population", [{}, {"inclusion": [_ADULT]}])
def test_a_design_without_criteria_keeps_its_field_and_sealed_body(population: dict) -> None:
    design = _design(**population)
    body = _body(design)

    assert "unapplied_criteria" not in design.get("population", {})
    assert "unapplied_population_criteria" not in body
    authority = build_trajectory_scientific_runtime_authority(body)
    assert "unapplied_population_criteria" not in authority.model_dump(mode="json")
    assert "unapplied_population_criteria" not in authority.population_definition


@pytest.mark.parametrize("value", [_SURGERY, [""], ["  "], [3], {"criterion": _SURGERY}])
def test_unreadable_criteria_are_refused(value: Any) -> None:
    with pytest.raises(TrajectoryDesignError) as caught:
        _design(inclusion=[_ADULT], unapplied_criteria=value)

    assert caught.value.code == "study_trajectory_population_invalid"
    assert caught.value.field == "trajectory_design.population.unapplied_criteria"


# The plan's facts and the host's compile ----------------------------------


@pytest.mark.parametrize(
    ("cohort", "source"),
    [
        ({"name": "primary", "inclusion": [_ADULT]}, "plan"),
        ({"name": "primary", "selection_mode": "all_input_rows"}, "none"),
    ],
)
def test_the_plan_facts_publish_the_criteria(cohort: dict, source: str) -> None:
    window = trajectory_window_design([])
    with_criteria = CohortDefinition.from_dict(
        {**cohort, "unapplied_population_criteria": [_SURGERY]}
    )

    facts = trajectory_population_design(with_criteria, window, static_concepts={"age"})

    assert facts["source"] == source
    assert facts["unapplied_criteria"] == [_SURGERY]
    assert trajectory_population_design(
        CohortDefinition.from_dict(cohort), window, static_concepts={"age"}
    )["unapplied_criteria"] == []


def test_the_host_compiles_the_criteria_into_the_design() -> None:
    review = _handoff_review(
        {"name": "primary", "inclusion": [_ADULT], "unapplied_population_criteria": [_SURGERY]}
    )
    # The plan that hands the question over reports its criterion too.
    assert [item.code for item in review.findings].count(_FINDING) == 1
    population = review.facts["trajectory_representation"]["trajectory_population"]
    assert population["unapplied_criteria"] == [_SURGERY]

    design = _compile(review.facts)["trajectory_design"]

    assert design["population"] == {
        "inclusion": [_ADULT],
        "exclusion": [],
        "unapplied_criteria": [_SURGERY],
    }


def test_a_plan_that_selects_every_stay_keeps_its_criteria_without_a_population() -> None:
    review = _handoff_review(
        {
            "name": "primary",
            "selection_mode": "all_input_rows",
            "unapplied_population_criteria": [_SURGERY],
        }
    )

    design = _compile(review.facts)["trajectory_design"]

    assert design["population"] == {
        "inclusion": [],
        "exclusion": [],
        "unapplied_criteria": [_SURGERY],
    }
    assert load_trajectory_design(design).population_definition is None


def test_a_plan_without_criteria_compiles_the_design_as_before() -> None:
    review = _handoff_review({"name": "primary", "inclusion": [_ADULT]})

    design = _compile(review.facts)["trajectory_design"]

    assert design["population"] == {"inclusion": [_ADULT], "exclusion": []}


# The plan a researcher approves ----------------------------------------


@pytest.mark.parametrize(
    ("population", "mode"),
    [({"inclusion": [_ADULT]}, "predicate_filtered"), ({}, "all_input_rows")],
)
def test_the_plan_on_the_signed_owners_states_the_criteria(population: dict, mode: str) -> None:
    authority = _authority(_design(**population, unapplied_criteria=[_SURGERY]))
    draft = AnalysisPlan.model_validate(
        {
            **authority.development_execution_only_plan(
                research_question=_QUESTION
            ).model_dump(mode="json"),
            # The draft's own cohort is not an authority for the signed owners.
            "cohort": {"name": "primary", "selection_mode": "all_input_rows"},
        }
    )

    bound, _ = ScientificRuntimeAuthorities(
        trajectory=authority, current_case=None
    ).bind_plan(draft)

    assert bound.cohort.selection_mode == mode
    assert bound.cohort.unapplied_population_criteria == (_SURGERY,)
    authority.validate_plan(bound)
    review = build_plan_scientific_review(context=_context(), plan=bound, literature=None)
    [finding] = [item for item in review.findings if item.code == _FINDING]
    assert f"'{_SURGERY}'" in finding.message
    assert finding.severity == "major"
