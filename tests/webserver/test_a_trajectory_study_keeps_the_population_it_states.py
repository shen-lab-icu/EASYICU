"""A trajectory study keeps the population it states.

A study states its population as a filter that Data Extraction applies: an
adult preset, an age or ICU-stay bound, or a concept-derived preset.  The
signed trajectory owners cluster the plan's cohort, and that cohort is
filtered since the design seals the population the reviewed plan applies.
The web owner still refused every stated filter, as it did when the signed
plan could only keep every stay, so the host could not compile a design for
such a study and its launch was refused.  The sealed suite's template, for
its part, stated only the host's typed bounds, which cannot express a
concept-derived filter.  Synthetic, case-neutral studies and contexts only.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import easyicu.research_agent.pipeline as _pipeline
from easyicu.research_agent.agents.family_spec_planner import FAMILY_SPEC_STRATEGY
from easyicu.research_agent.agents.progressive_planner import (
    ProgressivePlannerAgent,
    candidate_analysis_types,
    select_progressive_variables,
)
from easyicu.research_agent.contracts.trajectory_design import (
    load_trajectory_design,
    normalize_trajectory_design,
    sealed_trajectory_authority_body,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.planning import figure_plan_shaping as _figure_plan
from easyicu.research_agent.planning import final_plan_shape as _final_plan
from easyicu.research_agent.planning.dependence_authority import (
    bind_context_dependence_authority,
)
from easyicu.research_agent.planning.family_spec import build_family_spec_request
from easyicu.research_agent.planning.family_spec.contract import sealed_cohort_predicate
from easyicu.research_agent.planning.family_spec.request import (
    SEALED_TRAJECTORY_SUITE_MARKER,
    sealed_trajectory_suite_coordinates,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    build_trajectory_scientific_runtime_authority,
)
from easyicu.webserver.pi_copilot.plan_decisions import (
    PlanDecisionError,
    compile_agent_plan_configuration,
)
from easyicu.webserver.research_launch_scientific import (
    ResearchPipelineRunError,
    validate_analysis_design_for_execution,
)
from easyicu.webserver.scientific_runtime_projection import (
    WebScientificRuntimeProjectionError,
)
from easyicu.webserver.trajectory_runtime_projection import (
    validate_trajectory_design_declaration,
)

_REFUSED = "web_trajectory_population_filter_unsupported"
_OWNER = "TRAJECTORY_LONGITUDINAL_OWNER_NOT_SEALED"
_COORDINATES = ["sofa2_resp", "sofa2_cardio"]
_DESIGN = {"coordinate_concepts": _COORDINATES, "window_end_hours": 24, "grid_width_hours": 4}
_TRAJECTORY = {
    "analysis_family": "trajectory_clustering",
    "analysis_unit": "icu_stay",
    "variance_estimator": "model_based",
}
_STATED = [
    pytest.param({"preset": "adult_all"}, id="adult_preset"),
    pytest.param({"age_min": 18}, id="age_bound"),
    pytest.param({"min_icu_los_hours": 24}, id="icu_stay_bound"),
    pytest.param({"preset": "sepsis3", "observation_window_hours": 24}, id="concept_preset"),
]
# The setup pages' default.  It also keeps each patient's first ICU stay,
# which the compile and the launch prove from the bound source; these
# studies bind none.
_DEFAULT_ADULT = pytest.param({"preset": "adult_first"}, id="default_adult_preset")


def _predicate(concept, op, value, aggregation):
    return {
        "concept_id": concept,
        "time_window": {"anchor": "icu_admit", "start_offset_hours": 0, "end_offset_hours": 24},
        "aggregation": aggregation,
        "op": op,
        "value": value,
    }


_ADULT = _predicate("age", ">=", 18, "first")
_SEPSIS = _predicate("sep3", "==", 1, "any")
_POPULATION = {"inclusion": [_ADULT, _SEPSIS], "exclusion": []}
_PLANNED_POPULATION = {
    "inclusion": [_ADULT, _SEPSIS],
    "exclusion": [_predicate("age", ">", 89, "first")],
}


def _design(population=None) -> dict:
    return normalize_trajectory_design(
        {**_DESIGN, **({"population": population} if population else {})}
    )


def _study(cohort, *, population=None) -> dict:
    return {
        "analysis_design": dict(_TRAJECTORY),
        "trajectory_design": _design(population),
        "cohort": dict(cohort),
    }


def _authority(population=None):
    return build_trajectory_scientific_runtime_authority(
        sealed_trajectory_authority_body(
            load_trajectory_design(_design(population)), protocol_content_sha256="5" * 64
        )
    )


@pytest.mark.parametrize("cohort", [*_STATED, _DEFAULT_ADULT])
def test_a_stated_population_is_admitted_when_the_design_seals_one(cohort) -> None:
    declared = validate_trajectory_design_declaration(_study(cohort, population=_POPULATION))

    assert declared is not None and declared.population_concepts == ("age", "sep3")


@pytest.mark.parametrize("cohort", _STATED)
def test_the_launch_gate_admits_the_same_study(cohort) -> None:
    # The launch gate reads the same owner, so the study is not refused there.
    validate_analysis_design_for_execution(_study(cohort, population=_POPULATION))


@pytest.mark.parametrize("cohort", [*_STATED, _DEFAULT_ADULT])
def test_a_stated_population_is_still_refused_when_the_design_seals_none(cohort) -> None:
    study = _study(cohort)

    with pytest.raises(WebScientificRuntimeProjectionError) as declared:
        validate_trajectory_design_declaration(study)
    with pytest.raises(ResearchPipelineRunError) as launched:
        validate_analysis_design_for_execution(study)

    assert declared.value.code == launched.value.code == _REFUSED
    assert declared.value.details["stated_selection_mode"] == "predicate_filtered"
    assert "this design seals no population" in str(declared.value)


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


def _compile(cohort, population: dict):
    return compile_agent_plan_configuration(
        study={"cohort": dict(cohort), "confirmations": {"feature_time_window": True}},
        agent_plan={"steps": []},
        runtime_finding_codes=(_OWNER,),
        patient_cluster_available=False,
        review_facts=_published(population),
    )


@pytest.mark.parametrize("cohort", _STATED)
def test_the_host_compiles_the_design_of_a_study_that_states_its_population(cohort) -> None:
    compiled = _compile(
        cohort, {"source": "plan", "executable": True, **_POPULATION}
    )

    assert compiled.patch["trajectory_design"]["population"] == _POPULATION
    # A plan that selects every stay seals none, so the stated filter is
    # still refused rather than clustering every stay.
    with pytest.raises(PlanDecisionError) as raised:
        _compile(cohort, {"source": "none", "executable": True, "inclusion": [], "exclusion": []})
    assert raised.value.details["design_error_code"] == _REFUSED


def _disclosed(predicates) -> list[tuple]:
    return [
        (item.concept_id, item.anchor, item.end_offset_hours, item.aggregation, item.op,
         item.value.materialize())
        for item in predicates
    ]


def test_the_sealed_suite_discloses_its_population_to_planning() -> None:
    sealed = sealed_trajectory_suite_coordinates(
        _authority(
            {"inclusion": [_ADULT, _SEPSIS], "exclusion": [_predicate("age", ">", 89, "first")]}
        ).planning_contract_context()
    )
    unfiltered = sealed_trajectory_suite_coordinates(_authority().planning_contract_context())

    assert _disclosed(sealed.population_inclusion) == [
        ("age", "icu_admit", 24.0, "first", ">=", 18.0),
        ("sep3", "icu_admit", 24.0, "any", "==", 1.0),
    ]
    assert _disclosed(sealed.population_exclusion) == [("age", "icu_admit", 24.0, "first", ">", 89.0)]
    # A suite that seals no population serializes, and so digests, as before.
    assert not unfiltered.population_inclusion and not unfiltered.population_exclusion
    assert not any(key.startswith("population") for key in unfiltered.model_dump(mode="json"))


@pytest.mark.parametrize(
    ("op", "value", "mode"),
    [
        ("==", True, "boolean"),
        ("==", 1, "number"),
        ("==", "septic_shock", "string"),
        ("in", ["a", "b"], "string_list"),
        ("in", [1, 2], "number_list"),
        ("not_missing", None, "none"),
    ],
)
def test_a_disclosed_predicate_keeps_its_value_in_closed_form(op, value, mode) -> None:
    predicate = sealed_cohort_predicate(_predicate("flag", op, value, "any"))

    assert predicate.value.mode == mode
    assert predicate.value.materialize() == value


@pytest.mark.parametrize(
    "population",
    [
        pytest.param({"inclusion": [{"concept_id": "age", "op": ">="}]}, id="no_time_window"),
        pytest.param({"inclusion": ["age >= 18"]}, id="predicate_as_text"),
        pytest.param(["age >= 18"], id="population_not_an_object"),
    ],
)
def test_an_unreadable_sealed_population_makes_the_disclosure_unreadable(population) -> None:
    disclosure = {
        "sealed_representation_owner": "signed_trajectory_representation",
        "sealed_candidate_owner": "trajectory_candidate_selection",
        "coordinate_concepts": _COORDINATES,
        "window_hours": [0, 24],
        "grid_width_hours": 4,
        "candidate_cluster_counts": [2, 3],
        "representation_outputs": ["table:trajectory_representation"],
    }
    readable = SEALED_TRAJECTORY_SUITE_MARKER + "\n" + json.dumps(disclosure)
    unreadable = SEALED_TRAJECTORY_SUITE_MARKER + "\n" + json.dumps(
        {**disclosure, "population": population}
    )

    assert sealed_trajectory_suite_coordinates(readable) is not None
    # Planning on part of a sealed population would choose other stays.
    assert sealed_trajectory_suite_coordinates(unreadable) is None


_CITATIONS = ("strobe_2007", "record_2015")
_LABELS = {
    "sofa2_resp": "SOFA-2 respiratory score",
    "sofa2_cardio": "SOFA-2 cardiovascular score",
}


def _context() -> ResearchContext:
    """A metadata-only trajectory context whose population reads age and Sepsis-3."""

    provenance = {
        "analysis_unit": "icu_stay",
        "patient_identity_available": False,
        "stay_id_columns": ["stay_id"],
        "patient_id_columns": [],
        "evidence_stage": "metadata_only_planning",
        "patient_rows_read": False,
    }
    return ResearchContext(
        research_question=(
            "Among adults with sepsis, do organ-dysfunction trajectories over the "
            "first 24 h of an ICU stay cluster into distinct subgroups?"
        ),
        cohort=CohortDescriptor(
            cohort_name="trajectory_population_synthetic", database="miiv", n_stays=0,
            id_columns=["stay_id"], outcome_columns=[], provenance=provenance,
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            ConceptDescriptor(
                name="age", description="age at admission", role=VariableRole.DEMOGRAPHIC,
                dtype="float64",
            ),
            ConceptDescriptor(
                name="sep3", description="Sepsis-3 indicator", role=VariableRole.OTHER,
                dtype="float64", analysis_window="icu_admission[0,24]h",
                observed_domain={"n_unique": 2, "is_binary": True, "levels": [0, 1]},
            ),
            *[
                ConceptDescriptor(
                    name=concept, description=f"{concept} coordinate",
                    role=VariableRole.ORDINAL_SCORE, dtype="float64",
                    analysis_window="icu_admission[0,24]h",
                )
                for concept in _COORDINATES
            ],
        ],
        user_preferences=UserPreferences(
            inferred_analysis_family="trajectory_clustering",
            covariate_selection="planner_selectable",
        ),
    )


def _plan(mode: str):
    """Plan through the host as a run does: one labels call, then shape and bind."""

    context = _context()
    authorities = ScientificRuntimeAuthorities(
        trajectory=_authority(_PLANNED_POPULATION), current_case=None
    )
    request = build_family_spec_request(
        context,
        analysis_types=candidate_analysis_types(context),
        variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=_CITATIONS,
        required_primary_cohort_selection_mode=mode,
        planning_contract_context=authorities.planning_contract_context(),
    )
    payload = {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": [],
        "reader_display_labels": [
            {"key": key, "value": _LABELS[key]} for key in request.required_reader_label_keys
        ],
        "comparator_applications": [],
        "roster_decision_note": "The sealed suite owns every coordinate.",
    }
    llm = ScriptedMockLLMClient([json.dumps(payload)])
    draft = ProgressivePlannerAgent(llm).run_attempt(
        context,
        planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=_CITATIONS,
        direct_comparator_literature_keys=(),
        comparison_literature_keys=(),
        enforce_article_contract=True,
        article_contract_context=context,
        planning_contract_context=authorities.planning_contract_context(),
        required_primary_cohort_selection_mode=mode,
    ).output
    assert len(llm.calls) == 1
    findings: list = []
    plan = _pipeline._shape_fresh_plan(
        pipeline=SimpleNamespace(
            _scientific_runtime_authorities=authorities,
            _enable_publication_figure_skill=True,
            _max_total_steps=24,
        ),
        plan=draft, context=context, agent_context=context,
        long_trajectory_bound=False, findings=findings,
    )
    plan = bind_context_dependence_authority(plan=plan, context=context)
    bound, _ = authorities.bind_plan(plan)
    bound = _figure_plan.apply_runtime_bound_figure_contracts(bound, findings)
    authorities.validate_plan(bound)
    _final_plan.validate_final_plan_shape(bound)
    return draft, bound


def _cohort(cohort) -> tuple[str, list[dict], list[dict]]:
    return (
        cohort.selection_mode,
        [item.to_dict() for item in cohort.inclusion],
        [item.to_dict() for item in cohort.exclusion],
    )


_SEALED_COHORT = (
    "predicate_filtered",
    _PLANNED_POPULATION["inclusion"],
    _PLANNED_POPULATION["exclusion"],
)


def test_a_filtered_study_plans_the_population_its_suite_seals() -> None:
    # A concept-derived filter carries no typed bound for the template to state.
    draft, bound = _plan("predicate_filtered")

    assert _cohort(draft.cohort) == _SEALED_COHORT
    assert _cohort(bound.cohort) == _SEALED_COHORT


def test_an_unfiltered_study_keeps_its_template_cohort_and_the_suite_binds_the_population() -> None:
    draft, bound = _plan("all_input_rows")

    assert _cohort(draft.cohort) == ("all_input_rows", [], [])
    assert _cohort(bound.cohort) == _SEALED_COHORT
