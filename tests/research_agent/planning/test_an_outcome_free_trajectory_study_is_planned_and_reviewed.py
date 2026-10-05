"""An outcome-free trajectory study is planned and reviewed without an outcome.

Trajectory classes are discovered without an outcome; an outcome the study has
is only described by frozen class.  Four owners still asked every trajectory
study for one:
- the family router sent it to free-form planning;
- the family request refused it;
- the plan review and the maturity audit asked the researcher for an endpoint
  the question never named.
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
from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.contracts.trajectory_design import (
    load_trajectory_design,
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
from easyicu.research_agent.planning.family_spec import (
    FIXED_WINDOW_TRAJECTORY_FAMILY_ID,
    build_family_spec_request,
    family_template_id_for_context,
)
from easyicu.research_agent.planning.family_spec.contract import FamilySpecRequest
from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    study_endpoint_required,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.reporting.scientific_maturity import (
    build_scientific_maturity_audit,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    build_trajectory_scientific_runtime_authority,
)

_COORDINATES = ("sofa2_resp", "sofa2_cardio", "sofa2_renal", "lact")
_LABELS = {
    "death": "In-hospital death",
    "sofa2_resp": "SOFA-2 respiratory score",
    "sofa2_cardio": "SOFA-2 cardiovascular score",
    "sofa2_renal": "SOFA-2 renal score",
    "lact": "Lactate (mmol/L)",
}
_CITATIONS = ("strobe_2007", "record_2015")
_ENDPOINT = "OUTCOME_DEFINITION_UNRESOLVED"


def _authorities() -> ScientificRuntimeAuthorities:
    design = load_trajectory_design(
        {"coordinate_concepts": list(_COORDINATES), "descriptive_only_concepts": []}
    )
    authority = build_trajectory_scientific_runtime_authority(
        sealed_trajectory_authority_body(design, protocol_content_sha256="e" * 64)
    )
    return ScientificRuntimeAuthorities(trajectory=authority, current_case=None)


def _context(*, outcome: str | None = None) -> ResearchContext:
    provenance = {
        "analysis_unit": "icu_stay",
        "patient_identity_available": False,
        "stay_id_columns": ["stay_id"],
        "patient_id_columns": [],
        "evidence_stage": "metadata_only_planning",
        "patient_rows_read": False,
    }
    variables = [
        ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
        *[
            ConceptDescriptor(
                name=concept, description=f"{concept} coordinate",
                role=VariableRole.ORDINAL_SCORE, dtype="float64",
                analysis_window="icu_admission[0,72]h",
            )
            for concept in _COORDINATES
        ],
    ]
    if outcome:
        variables.append(
            ConceptDescriptor(
                name=outcome, description="in-hospital mortality", role=VariableRole.OUTCOME,
                dtype="float64", observed_domain={"n_unique": 2, "is_binary": True, "levels": [0, 1]},
            )
        )
    return ResearchContext(
        research_question=(
            "Do organ-dysfunction trajectories over the first 72 h of an ICU stay "
            "cluster into distinct subgroups?"
        ),
        cohort=CohortDescriptor(
            cohort_name="trajectory_synthetic", database="miiv", n_stays=0,
            id_columns=["stay_id"], outcome_columns=[outcome] if outcome else [],
            provenance=provenance,
        ),
        variables=variables,
        target_outcome=outcome,
        endpoint=(
            EndpointSpec(name=outcome, kind="binary", absence_semantics="no_absent_rows", levels=[0, 1])
            if outcome
            else None
        ),
        user_preferences=UserPreferences(
            inferred_analysis_family="trajectory_clustering",
            covariate_selection="planner_selectable",
        ),
    )


def _request(context: ResearchContext, authorities: ScientificRuntimeAuthorities):
    return build_family_spec_request(
        context,
        analysis_types=candidate_analysis_types(context),
        variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=_CITATIONS,
        required_primary_cohort_selection_mode="all_input_rows",
        planning_contract_context=authorities.planning_contract_context(),
    )


def _plan(context: ResearchContext) -> tuple[AnalysisPlan, AnalysisPlan]:
    """Plan through the host as a run does: one labels call, then shape and bind.

    Returns the template's draft, which carries the design record, and the
    plan bound to the signed owners.
    """

    authorities = _authorities()
    request = _request(context, authorities)
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
        required_primary_cohort_selection_mode="all_input_rows",
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


def _describes_an_outcome(step) -> bool:
    return step.scientific_action_id == "phenotyping.outcome_by_cluster"


def _review_codes(context: ResearchContext, plan: AnalysisPlan) -> tuple[bool, set[str]]:
    review = build_plan_scientific_review(
        context=context, plan=plan, literature=None,
        figure_strategy=build_article_figure_strategy(context), runtime_authority=None,
    )
    return review.approval_allowed, {finding.code for finding in review.findings}


def test_the_sealed_suite_template_takes_a_study_without_an_outcome() -> None:
    context = _context()
    authorities = _authorities()
    types = candidate_analysis_types(context)

    assert (
        family_template_id_for_context(
            context, analysis_types=types,
            planning_contract_context=authorities.planning_contract_context(),
        )
        == FIXED_WINDOW_TRAJECTORY_FAMILY_ID
    )
    # Without the sealed suite, nothing is templated: free-form planning remains.
    assert family_template_id_for_context(context, analysis_types=types) is None
    request = _request(context, authorities)
    assert (request.outcome, request.outcome_levels, request.event_level_index) == ("", [], 0)


def test_a_request_without_an_outcome_carries_no_outcome_levels() -> None:
    request = _request(_context(), _authorities())
    payload = request.model_dump(mode="json")

    with pytest.raises(ValidationError, match="without an outcome has no outcome levels"):
        FamilySpecRequest.model_validate({**payload, "outcome_levels": ["0", "1"]})


def test_the_outcome_free_plan_describes_no_outcome_and_needs_no_endpoint() -> None:
    context = _context()

    draft, plan = _plan(context)

    for planned in (draft, plan):
        assert not any(_describes_an_outcome(step) for step in planned.steps)
        assert all(value for step in planned.steps for value in step.inputs)
    design = draft.design_selection.selected
    assert design.required_variables == ["stay_id", *_COORDINATES]
    assert "descriptive link" not in design.supports
    assert "names no outcome" in design.reviewable_plan[2]
    allowed, codes = _review_codes(context, plan)
    assert allowed, codes
    assert _ENDPOINT not in codes


def test_a_declared_outcome_is_still_described_and_defined() -> None:
    with_outcome = _context(outcome="death")

    draft, plan = _plan(with_outcome)

    assert any(_describes_an_outcome(step) for step in draft.steps)
    assert any(_describes_an_outcome(step) for step in plan.steps)
    assert study_endpoint_required(with_outcome, plan)
    # A declared outcome without its owner-issued definition is still the
    # researcher's decision.
    undefined = with_outcome.model_copy(update={"endpoint": None})
    assert _ENDPOINT in _review_codes(undefined, plan)[1]


def test_an_endpoint_is_required_by_a_declared_outcome_or_an_endpoint_result() -> None:
    _draft, plan = _plan(_context())
    no_outcome = _context()

    assert not study_endpoint_required(no_outcome, plan)
    assert study_endpoint_required(_context(outcome="death"), plan)
    for family in ("descriptive_epidemiology", "survival", "prediction_model"):
        assert study_endpoint_required(
            no_outcome, plan.model_copy(update={"analysis_type": family})
        ), family


def test_the_maturity_audit_asks_no_endpoint_of_an_outcome_free_study(tmp_path) -> None:
    _draft, plan = _plan(_context())
    no_outcome = _context()

    def codes(audit_plan: AnalysisPlan) -> set[str]:
        audit = build_scientific_maturity_audit(
            context=no_outcome, plan=audit_plan, run_dir=tmp_path
        )
        return {finding.code for finding in audit.findings}

    assert _ENDPOINT not in codes(plan)
    assert _ENDPOINT in codes(plan.model_copy(update={"analysis_type": "descriptive_epidemiology"}))
