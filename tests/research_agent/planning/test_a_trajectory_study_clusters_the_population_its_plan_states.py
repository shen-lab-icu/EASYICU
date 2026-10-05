"""A trajectory study clusters the population its plan states.

A trajectory question names whom it studies ("adults with sepsis"), and the
Planner states that population as typed cohort predicates.  The signed
fixed-window owner used to replace every plan's cohort with all input rows,
so such a study clustered every stay of the source universe and nothing said
so.  Now the review publishes the plan's population beside the coordinates
and the window, the Host compiles it into the design unchanged, and the
sealed authority binds exactly those predicates into the plan cohort the
pipeline materializes.  A predicate the owner cannot apply -- one counted
from another event, or settled after the trajectory window ends, which
would choose stays by what happens after the hours the classes describe --
is a stated limitation, never a dropped predicate.  Fixtures are generic.
"""

from __future__ import annotations

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
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    remediation_route_for_finding,
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
    TrajectoryScientificAuthorityError,
    build_trajectory_scientific_runtime_authority,
)
from easyicu.webserver.pi_copilot.plan_decisions import (
    PlanDecisionError,
    agent_plan_configuration_available,
    compile_agent_plan_configuration,
)
from easyicu.webserver.trajectory_runtime_projection import (
    validate_trajectory_design_declaration,
)

_OWNER = "TRAJECTORY_LONGITUDINAL_OWNER_NOT_SEALED"
_COORDINATES = ["sofa2_resp", "sofa2_cardio"]
_STUDY = {
    "cohort": {"preset": "all_icu"},
    "confirmations": {"feature_time_window": True},
    "sensitivity_specs": [],
}
_ROLES = {
    "stay_id": VariableRole.ID,
    "age": VariableRole.DEMOGRAPHIC,
    "sep3": VariableRole.OTHER,
    "sofa2_resp": VariableRole.ORDINAL_SCORE,
    "sofa2_cardio": VariableRole.ORDINAL_SCORE,
    "lact": VariableRole.LAB,
    "death": VariableRole.OUTCOME,
}
_QUESTION = (
    "Among adults with sepsis, do first-24h physiologic trajectories form "
    "distinct classes?"
)


def _predicate(concept, op, value, end, aggregation="max", anchor="icu_admit", start=0):
    return {
        "concept_id": concept,
        "time_window": {
            "anchor": anchor,
            "start_offset_hours": start,
            "end_offset_hours": end,
        },
        "aggregation": aggregation,
        "op": op,
        "value": value,
    }


_ADULT = _predicate("age", ">=", 18, 24, "first")
_SEPSIS = _predicate("sep3", "==", 1, 24, "any")


def _design(**population) -> dict:
    return normalize_trajectory_design(
        {
            "coordinate_concepts": _COORDINATES,
            "window_end_hours": 24,
            "grid_width_hours": 4,
            **({"population": population} if population else {}),
        }
    )


def _authority(design: dict):
    return build_trajectory_scientific_runtime_authority(
        sealed_trajectory_authority_body(
            load_trajectory_design(design), protocol_content_sha256="1" * 64
        )
    )


def _canonical(predicates) -> list[dict]:
    return [predicate.to_dict() for predicate in predicates]


# The design owner -------------------------------------------------------


def test_a_design_without_a_population_keeps_its_field_and_sealed_body() -> None:
    design = _design()
    authority = _authority(design)

    assert "population" not in design
    assert "population" not in sealed_trajectory_authority_body(
        load_trajectory_design(design), protocol_content_sha256="1" * 64
    )
    assert "population" not in authority.model_dump(mode="json")
    plan = authority.development_execution_only_plan(research_question=_QUESTION)
    assert plan.cohort.selection_mode == "all_input_rows"
    assert not plan.cohort.inclusion and not plan.cohort.exclusion


def test_a_population_settled_by_the_window_end_is_part_of_the_design() -> None:
    design = _design(inclusion=[_ADULT, _SEPSIS])

    assert design["population"] == {"inclusion": [_ADULT, _SEPSIS], "exclusion": []}
    typed = load_trajectory_design(design)
    assert typed.population_concepts == ("age", "sep3")
    assert typed.as_study_field() == design
    # Coordinates alone still reach the long panel; the population reads
    # stay-level columns of the same universe.
    assert typed.required_concepts == tuple(_COORDINATES)
    assert typed.population_definition == {
        "name": "primary",
        "selection_mode": "predicate_filtered",
        "inclusion": [_ADULT, _SEPSIS],
        "exclusion": [],
    }


@pytest.mark.parametrize(
    ("predicate", "reason"),
    [
        (
            _predicate("sep3", "==", 1, 72, "any"),
            "is settled after the 24-hour trajectory window ends",
        ),
        (
            _predicate("sep3", "==", 1, "inf", "any"),
            "is settled after the 24-hour trajectory window ends",
        ),
        (
            _predicate("sep3", "==", 1, 24, "any", anchor="hospital_admit"),
            "is not counted from ICU admission",
        ),
    ],
)
def test_a_population_the_owner_cannot_apply_is_refused_by_name(predicate, reason) -> None:
    with pytest.raises(TrajectoryDesignError) as raised:
        _design(inclusion=[_ADULT], exclusion=[predicate])

    assert raised.value.code == "study_trajectory_population_window_invalid"
    assert raised.value.field == "trajectory_design.population"
    assert "the exclusion predicate sep3 == 1" in str(raised.value)
    assert reason in str(raised.value)


@pytest.mark.parametrize(
    "population",
    [
        "adults with sepsis",
        {"inclusion": [_ADULT], "criteria": []},
        {"inclusion": _ADULT},
        {"inclusion": [{"concept_id": "age", "op": ">="}]},
    ],
)
def test_a_population_that_is_not_typed_predicates_is_refused(population) -> None:
    with pytest.raises(TrajectoryDesignError):
        normalize_trajectory_design(
            {"coordinate_concepts": _COORDINATES, "population": population}
        )


# The sealed authority ---------------------------------------------------


def test_the_sealed_population_is_the_plan_cohort_the_owners_analyze() -> None:
    authority = _authority(_design(inclusion=[_ADULT, _SEPSIS]))

    plan = authority.development_execution_only_plan(research_question=_QUESTION)

    assert plan.cohort.selection_mode == "predicate_filtered"
    assert _canonical(plan.cohort.inclusion) == [_ADULT, _SEPSIS]
    assert not plan.cohort.exclusion
    assert '"population"' in authority.planning_contract_context()
    authority.validate_plan(plan)


@pytest.mark.parametrize(
    "cohort",
    [
        {"name": "primary", "selection_mode": "all_input_rows"},
        {"name": "primary", "inclusion": [_ADULT]},
        {"name": "primary", "inclusion": [_ADULT, {**_SEPSIS, "value": 0}]},
        {"name": "primary", "inclusion": [_ADULT, _SEPSIS], "exclusion": [_ADULT]},
    ],
)
def test_a_plan_that_drifts_from_the_sealed_population_is_refused(cohort) -> None:
    authority = _authority(_design(inclusion=[_ADULT, _SEPSIS]))
    plan = authority.development_execution_only_plan(research_question=_QUESTION)

    drifted = AnalysisPlan.model_validate(
        {**plan.model_dump(mode="json"), "cohort": cohort}
    )
    with pytest.raises(TrajectoryScientificAuthorityError) as raised:
        authority.validate_plan(drifted)
    assert "analyze the sealed population" in str(raised.value)


def test_without_a_population_the_owners_still_refuse_predicates() -> None:
    authority = _authority(_design())
    plan = authority.development_execution_only_plan(research_question=_QUESTION)

    drifted = AnalysisPlan.model_validate(
        {**plan.model_dump(mode="json"), "cohort": {"name": "primary", "inclusion": [_ADULT]}}
    )
    with pytest.raises(TrajectoryScientificAuthorityError) as raised:
        authority.validate_plan(drifted)
    assert "select all input rows" in str(raised.value)


def test_an_authority_with_a_population_settled_after_its_window_is_not_sealed() -> None:
    body = sealed_trajectory_authority_body(
        load_trajectory_design(_design(inclusion=[_ADULT])),
        protocol_content_sha256="1" * 64,
    )
    body["population"] = {
        "inclusion": [_predicate("sep3", "==", 1, 48, "any")],
        "exclusion": [],
    }
    with pytest.raises(ValueError, match="settled after the 24-hour trajectory window"):
        build_trajectory_scientific_runtime_authority(body)


def test_binding_a_draft_that_names_the_owners_carries_the_sealed_population() -> None:
    authority = _authority(_design(inclusion=[_ADULT, _SEPSIS]))
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

    assert bound.cohort.selection_mode == "predicate_filtered"
    assert _canonical(bound.cohort.inclusion) == [_ADULT, _SEPSIS]


# The review -------------------------------------------------------------


def _review(cohort: dict | None, question: str = _QUESTION):
    context = ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(
            cohort_name="trajectory_population_fixture", database="synthetic",
            n_stays=0, id_columns=["stay_id"], outcome_columns=["death"],
        ),
        variables=[
            ConceptDescriptor(name=name, role=role, dtype="float64")
            for name, role in _ROLES.items()
        ],
        target_outcome="death",
    )
    step = AnalysisStep(
        step_id="trajectory_solution",
        planned_analysis_role="primary",
        intent="Cluster the prespecified coordinates into trajectory classes.",
        inputs=["stay_id", *_COORDINATES, "lact", "artifact:analysis_cohort"],
        expected_outputs=["table:phenotype_assignments"],
        method="prespecified_trajectory_feature_clustering",
        scientific_action_id=TRAJECTORY_PRIMARY_ACTION,
    )
    plan = AnalysisPlan(
        research_question=question,
        analysis_type="trajectory_clustering",
        steps=[step],
        cohort=cohort,
    )
    review = build_plan_scientific_review(context=context, plan=plan, literature=None)
    [finding] = [item for item in review.findings if item.code.startswith("TRAJECTORY_")]
    return review, finding


def _compile(facts):
    return compile_agent_plan_configuration(
        study=dict(_STUDY),
        agent_plan={"steps": []},
        runtime_finding_codes=(_OWNER,),
        patient_cluster_available=False,
        review_facts=facts,
    )


def test_the_review_names_the_population_and_the_host_compiles_it_unchanged() -> None:
    review, finding = _review({"name": "primary", "inclusion": [_ADULT, _SEPSIS]})

    assert finding.code == _OWNER
    assert (
        "in the population the plan states (2 inclusion predicates, each settled "
        "by the end of the trajectory window)"
    ) in finding.message
    # A time-varying predicate decided inside the window is disclosed; a
    # demographic is fixed at admission and is not.
    assert "Membership is decided inside the trajectory window by sep3 == 1" in (
        finding.message
    )
    assert "age >=" not in finding.message
    assert "this window and this population" in finding.remediation
    assert "analysis_plan.json.cohort" in finding.evidence_refs
    population = review.facts["trajectory_representation"]["trajectory_population"]
    assert population["source"] == "plan" and population["executable"] is True

    compiled = _compile(review.facts)
    design = compiled.patch["trajectory_design"]
    assert design["population"] == {"inclusion": [_ADULT, _SEPSIS], "exclusion": []}
    declared = validate_trajectory_design_declaration({**_STUDY, **compiled.patch})
    assert declared is not None and declared.population_concepts == ("age", "sep3")
    sealed = _authority(design).development_execution_only_plan(research_question=_QUESTION)
    assert _canonical(sealed.cohort.inclusion) == [_ADULT, _SEPSIS]


def test_a_plan_that_selects_every_stay_compiles_no_population() -> None:
    review, finding = _review({"name": "primary", "selection_mode": "all_input_rows"})

    assert finding.code == _OWNER
    # The compile clusters every stay; the finding says so rather than leave
    # a population the question names to look applied.
    assert (
        "in every input row: the plan's cohort states no population predicate" in finding.message
    )
    assert "analysis_plan.json.cohort" not in finding.evidence_refs
    population = review.facts["trajectory_representation"]["trajectory_population"]
    assert population["source"] == "none" and population["executable"] is True
    assert "population" not in _compile(review.facts).patch["trajectory_design"]


def test_a_population_the_owner_cannot_apply_is_a_stated_limitation_not_dropped() -> None:
    late = _predicate("sep3", "==", 1, 72, "any")
    review, finding = _review({"name": "primary", "inclusion": [_ADULT, late]})

    assert finding.code == "TRAJECTORY_REPRESENTATION_NOT_LONGITUDINAL"
    assert finding.severity == "major"
    assert remediation_route_for_finding(finding) == "agent_plan_revision"
    assert (
        "the inclusion predicate sep3 == 1 (any over 0–72 h from ICU admission) is "
        "settled after the 24-hour trajectory window ends"
    ) in finding.message
    assert "Keep the population the plan states" in finding.remediation
    facts = review.facts["trajectory_representation"]
    assert facts["coordinates_executable"] is True
    assert facts["trajectory_window"]["executable"] is True
    assert facts["executable"] is False
    assert facts["trajectory_population"]["inclusion"] == [_ADULT, late]
    assert not agent_plan_configuration_available(
        study=dict(_STUDY), agent_plan={"steps": []},
        runtime_finding_codes=(_OWNER,), review_facts=review.facts,
    )


def test_the_population_is_judged_against_the_window_the_question_states() -> None:
    within_72 = _predicate("sep3", "==", 1, 48, "any")

    _, short = _review({"name": "primary", "inclusion": [within_72]})
    _, long = _review(
        {"name": "primary", "inclusion": [within_72]},
        question="Do physiologic trajectories over the first 72 hours form classes?",
    )

    assert short.code == "TRAJECTORY_REPRESENTATION_NOT_LONGITUDINAL"
    assert long.code == _OWNER


def _published(population: dict) -> dict:
    return {
        "trajectory_representation": {
            "longitudinal_owner": None,
            "proposed_coordinates": list(_COORDINATES),
            "trajectory_window": {
                "source": "question", "executable": True,
                "window_start_hours": 0, "window_end_hours": 24, "grid_width_hours": 4,
            },
            "trajectory_population": population,
            "executable": True,
        }
    }


def test_the_compiled_population_is_the_published_fact_and_its_rules_still_hold() -> None:
    published = trajectory_population_design(
        {"inclusion": [_ADULT]}, trajectory_window_design([])
    )
    assert published["source"] == "plan" and published["executable"] is True
    design = _compile(_published(published)).patch["trajectory_design"]
    assert design["population"] == {"inclusion": [_ADULT], "exclusion": []}

    # A population the review did not find executable is not compiled.
    design = _compile(_published({**published, "executable": False})).patch["trajectory_design"]
    assert "population" not in design

    # A tampered or stale population still has to satisfy the design owner.
    with pytest.raises(PlanDecisionError) as raised:
        _compile(
            _published({**published, "inclusion": [_predicate("sep3", "==", 1, 72, "any")]})
        )
    assert raised.value.code == "agent_plan_trajectory_design_invalid"
