"""A question about the classes of a measurement's trajectories plans on metadata.

A study planned on a registered source's metadata holds no rows yet; its run
materializes each stay's timed values as a long panel over the trajectory
design's own window.  A question asking into which classes one vital sign's
trajectories over a stated window divide still stopped before any plan:

- the family router read it as descriptive, because only "trajectory classes
  emerge" wording counted as discovering classes;
- the stated window was not read when its anchor was spaced ("入 ICU 后 48 小时");
- the host verified a longitudinal capability gap from the stay-summary
  planning catalog, which has no time column;
- the signed owner counted eligible windows on SOFA-2 components only and
  needed two coordinates, so one measured vital sign was no design.

Synthetic, case-neutral contexts only.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import easyicu.research_agent.pipeline as _pipeline
from easyicu.research_agent.agents.family_spec_planner import FAMILY_SPEC_STRATEGY
from easyicu.research_agent.agents.progressive_planner import (
    ProgressivePlannerAgent,
    candidate_analysis_types,
    select_progressive_variables,
)
from easyicu.research_agent.contracts.trajectory_design import (
    executable_trajectory_coordinates,
    load_trajectory_design,
    recorded_measurement_concept,
    sealed_trajectory_authority_body,
    trajectory_window_design,
    unrecorded_trajectory_coordinates,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.planning import figure_plan_shaping as _figure_plan
from easyicu.research_agent.planning import final_plan_shape as _final_plan
from easyicu.research_agent.planning.analysis_types import (
    infer_analysis_type_with_basis,
    strong_trajectory_clustering_framing,
)
from easyicu.research_agent.planning.capability_gap import (
    check_capability_gap,
    longitudinal_representation_facts,
)
from easyicu.research_agent.planning.dependence_authority import (
    bind_context_dependence_authority,
)
from easyicu.research_agent.planning.design_selection import ResearchDesignCandidate
from easyicu.research_agent.planning.family_spec import (
    FIXED_WINDOW_TRAJECTORY_FAMILY_ID,
    build_family_spec_request,
    family_template_id_for_context,
)
from easyicu.research_agent.planning.progressive_contract import ProgressiveCapabilityGap
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.research_context.temporal_semantics import (
    trajectory_window_statements,
)
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

_ZH_QUESTION = "入 ICU 后 48 小时的心率轨迹可分为哪几类？"
_EN_QUESTION = (
    "Into how many classes can heart-rate trajectories over the first 48 hours "
    "after ICU admission be divided?"
)
_ROLES = {
    "stay_id": VariableRole.ID,
    "hr": VariableRole.VITAL,
    "lact": VariableRole.LAB,
    "gcs": VariableRole.ORDINAL_SCORE,
    "death": VariableRole.OUTCOME,
}
_GAP = ProgressiveCapabilityGap(requirement="longitudinal_representation_unavailable")


def _context(
    question: str = _EN_QUESTION,
    *,
    names: tuple[str, ...] = ("hr",),
    evidence_stage: str = "metadata_only_planning",
) -> ResearchContext:
    return ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(
            cohort_name="trajectory_synthetic", database="miiv", n_stays=0,
            id_columns=["stay_id"], outcome_columns=[],
            provenance={
                "analysis_unit": "icu_stay",
                "patient_identity_available": False,
                "stay_id_columns": ["stay_id"],
                "patient_id_columns": [],
                "evidence_stage": evidence_stage,
                "patient_rows_read": False,
            },
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            *[
                ConceptDescriptor(
                    name=name, description=f"{name} measurement", role=_ROLES[name],
                    dtype="float64", source_concept=name,
                    analysis_window="icu_admission[0,24]h",
                )
                for name in names
            ],
        ],
    )


@pytest.mark.parametrize(
    "question",
    [
        _ZH_QUESTION,
        "入住 ICU 首日的生命体征和化验能分出哪几种表型？",
        "患者能分成几个亚型？",
        _EN_QUESTION,
        "Can heart-rate trajectories be classified into distinct groups?",
        "How many phenotypes emerge from first-day vital signs and laboratory values?",
    ],
)
def test_a_question_asking_for_classes_reads_as_class_discovery(question: str) -> None:
    assert strong_trajectory_clustering_framing(question.lower())
    family, basis = infer_analysis_type_with_basis(_context(question))
    assert (family.key, basis) == ("trajectory_clustering", "keyword")


@pytest.mark.parametrize(
    "question",
    [
        "已知的 ARDS 亚型在入 ICU 首日的死亡率有何不同？",
        "不同表型患者的院内死亡率有何不同？",
        "按乳酸分为高乳酸组和低乳酸组，比较死亡率。",
        "心率轨迹分为上升组和下降组时，死亡率是否不同？",
        "不进行轨迹分型，只描述入 ICU 后 48 小时的心率。",
        "Is first-day lactate, divided into tertiles, associated with death?",
        "Which phenotypes are associated with in-hospital mortality?",
    ],
)
def test_known_or_stated_groups_are_not_class_discovery(question: str) -> None:
    assert not strong_trajectory_clustering_framing(question.lower())


def test_a_spaced_anchor_states_the_trajectory_window() -> None:
    [statement] = trajectory_window_statements(_ZH_QUESTION)
    assert (statement.hours, statement.anchor) == (48.0, "icu_admission")
    design = trajectory_window_design((statement,))
    assert (design["window_start_hours"], design["window_end_hours"]) == (0, 48)
    assert design["executable"] is True
    # Another spaced anchor keeps its own event, which the owner does not count from.
    [other] = trajectory_window_statements("插管后 48 小时的心率轨迹可分为哪几类？")
    assert (other.hours, other.anchor) == (48.0, "插管")
    assert trajectory_window_design((other,))["executable"] is False


def test_recorded_measurements_are_coordinates_and_derived_variables_are_not() -> None:
    for name in ("hr", "map", "lact"):
        assert recorded_measurement_concept(name), name
    for name in ("gcs", "vaso_ind", "mech_vent", "sofa_resp", "age", "not_a_concept"):
        assert not recorded_measurement_concept(name), name
    assert executable_trajectory_coordinates(["hr"])
    assert executable_trajectory_coordinates(["hr", "lact"])
    assert not executable_trajectory_coordinates([])
    assert not executable_trajectory_coordinates(["gcs"])
    assert not executable_trajectory_coordinates(["hr", "gcs"])
    assert unrecorded_trajectory_coordinates(["hr", "gcs"]) == ("gcs",)
    # A design with a SOFA-2 component counts windows on its receipts, unchanged.
    assert executable_trajectory_coordinates(["sofa2_resp", "gcs"])
    assert unrecorded_trajectory_coordinates(["sofa2_resp", "gcs"]) == ()


@pytest.mark.parametrize("question", [_EN_QUESTION, _ZH_QUESTION])
def test_the_host_contradicts_a_longitudinal_gap_it_can_plan(question: str) -> None:
    context = _context(question, names=("hr", "gcs"))

    facts = longitudinal_representation_facts(context)
    check = check_capability_gap(_GAP, context=context)

    assert facts is not None
    assert facts["time_varying_variables"] == ["hr"]
    assert facts["requires_longitudinal_representation"] is True
    assert check.verification == "unverified"
    assert "0-48 h after ICU admission in 8-hour windows" in check.fact
    assert "phenotyping.trajectory_feature_clustering" in check.fact
    # A part no offered step expresses is stated, never dropped or invented.
    assert "cannot_prove" in check.fact


@pytest.mark.parametrize(
    "context",
    [
        pytest.param(_context(evidence_stage="prepared"), id="rows_already_prepared"),
        pytest.param(_context(names=("gcs",)), id="only_derived_variables"),
        pytest.param(
            _context("插管后 48 小时的心率轨迹可分为哪几类？"), id="window_from_another_event"
        ),
    ],
)
def test_the_gap_still_stands_where_the_host_cannot_plan_it(context: ResearchContext) -> None:
    assert longitudinal_representation_facts(context) is None
    assert check_capability_gap(_GAP, context=context).verification == "verified"


def _candidate(analysis_type: str, variables: list[str]) -> dict:
    return {
        "design_id": "single_variable_design",
        "analysis_type": analysis_type,
        "estimand": "Latent classes of the stated measurement's trajectories.",
        "time_zero": "ICU admission of each stay.",
        "observation_window": "0-48 h after ICU admission.",
        "primary_method": "Fixed-window latent class model",
        "required_variables": variables,
        "assumptions": ["Windows are aligned from ICU admission."],
        "novelty_positioning": "Descriptive classes only; no novelty is claimed.",
        "figure_role": "Class trajectories over the window.",
        "supports": "The classes the measurement forms.",
        "cannot_prove": "Causes of class membership.",
        "disposition": "selected",
        "decision_reason": "The question asks for the classes of one measurement.",
    }


def test_a_trajectory_design_may_name_one_measurement() -> None:
    ResearchDesignCandidate.model_validate(_candidate("trajectory_clustering", ["hr"]))
    with pytest.raises(ValidationError, match="require at least two variables"):
        ResearchDesignCandidate.model_validate(_candidate("association_study", ["hr"]))


def _authorities(coordinates: tuple[str, ...]) -> ScientificRuntimeAuthorities:
    design = load_trajectory_design(
        {
            "coordinate_concepts": list(coordinates),
            "descriptive_only_concepts": [],
            "window_start_hours": 0,
            "window_end_hours": 48,
            "grid_width_hours": 8,
        }
    )
    authority = build_trajectory_scientific_runtime_authority(
        sealed_trajectory_authority_body(design, protocol_content_sha256="e" * 64)
    )
    return ScientificRuntimeAuthorities(trajectory=authority, current_case=None)


def test_one_measured_coordinate_seals_the_signed_authority() -> None:
    authority = _authorities(("hr",)).trajectory

    assert authority.coordinate_concepts == ("hr",)
    assert authority.representation_columns == tuple(
        f"hr__h{start}_{start + 8}" for start in range(0, 48, 8)
    )


def test_the_family_route_plans_the_sealed_suite_on_one_measurement() -> None:
    context = _context(_ZH_QUESTION).model_copy(
        update={
            "user_preferences": UserPreferences(
                inferred_analysis_family="trajectory_clustering",
                covariate_selection="planner_selectable",
            )
        }
    )
    authorities = _authorities(("hr",))
    contract = authorities.planning_contract_context()
    types = candidate_analysis_types(context)
    assert (
        family_template_id_for_context(context, analysis_types=types, planning_contract_context=contract)
        == FIXED_WINDOW_TRAJECTORY_FAMILY_ID
    )
    citations = ("strobe_2007", "record_2015")
    request = build_family_spec_request(
        context, analysis_types=types, variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=citations,
        required_primary_cohort_selection_mode="all_input_rows",
        planning_contract_context=contract,
    )
    payload = {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": [],
        "reader_display_labels": [
            {"key": key, "value": "Heart rate (beats/min)"} for key in request.required_reader_label_keys
        ],
        "comparator_applications": [],
        "roster_decision_note": "The sealed suite owns every coordinate.",
    }
    llm = ScriptedMockLLMClient([json.dumps(payload)])
    draft = ProgressivePlannerAgent(llm).run_attempt(
        context, planner_strategy=FAMILY_SPEC_STRATEGY, allowed_literature_citation_keys=citations,
        direct_comparator_literature_keys=(), comparison_literature_keys=(),
        enforce_article_contract=True, article_contract_context=context,
        planning_contract_context=contract, required_primary_cohort_selection_mode="all_input_rows",
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

    assert draft.design_selection.selected.required_variables == ["stay_id", "hr"]
    assert any(step.method == "signed_fixed_window_trajectory_representation" for step in bound.steps)
    review = build_plan_scientific_review(
        context=context, plan=bound, literature=None, runtime_authority=None,
    )
    blockers = {finding.code for finding in review.findings if finding.severity == "blocker"}
    assert review.approval_allowed, blockers
