"""A trajectory design names its coordinates only while its estimand holds them.

The fixed-window trajectory template wrote every coordinate label into the
selected design's estimand, a sentence the design contract bounds.  Seven
coordinates with ordinary descriptive labels exceeded it: the Planner's spec
passed, and the run ended in schema validation when the host built the design
from that spec.  The landmark, phenotyping and prediction templates already
name a roster only while it fits, and the plan lists it in full.  The
estimand also called every study's classes organ-dysfunction classes,
whatever its coordinates measured.  Synthetic, case-neutral fixtures.
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
from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.contracts.trajectory_design import (
    TrajectoryDesignError,
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
from easyicu.research_agent.planning.family_spec import build_family_spec_request
from easyicu.research_agent.planning.family_spec.contract import (
    SpecReaderLabel,
    design_field_max_length,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
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

_CITATIONS = ("strobe_2007", "record_2015")
_SHORT = {
    "hr": "Heart rate (beats/min)",
    "map": "Mean arterial pressure (mmHg)",
    "lact": "Lactate (mmol/L)",
}
_ORDINARY = {
    "hr": "Heart rate, highest value per window (beats/min)",
    "map": "Mean arterial pressure, lowest value per window (mmHg)",
    "sbp": "Systolic blood pressure, lowest value per window (mmHg)",
    "resp": "Respiratory rate, highest value per window (breaths/min)",
    "lact": "Lactate, highest value per window (mmol/L)",
    "crea": "Creatinine, highest value per window (mg/dL)",
    "plt": "Platelet count, lowest value per window (10^3/uL)",
}
#: The design owner's coordinate maximum, checked below rather than imported.
_MOST_COORDINATES = (
    "hr", "map", "sbp", "dbp", "resp", "temp", "o2sat", "lact",
    "crea", "bili", "plt", "wbc", "na", "k", "glu", "bun",
)


def _label_bound() -> int:
    return next(
        int(item.max_length)
        for item in SpecReaderLabel.model_fields["value"].metadata
        if getattr(item, "max_length", None) is not None
    )


def _longest(name: str) -> str:
    """A distinct reader label at the spec contract's own length bound."""

    text = f"{name}: worst value in each fixed window after ICU admission, as charted; "
    return (text * (_label_bound() // len(text) + 1))[: _label_bound() - 1].rstrip() + "."


def _authorities(coordinates) -> ScientificRuntimeAuthorities:
    design = load_trajectory_design(
        {
            "coordinate_concepts": list(coordinates),
            "descriptive_only_concepts": [],
            "window_end_hours": 24,
            "grid_width_hours": 4,
        }
    )
    authority = build_trajectory_scientific_runtime_authority(
        sealed_trajectory_authority_body(design, protocol_content_sha256="f" * 64)
    )
    return ScientificRuntimeAuthorities(trajectory=authority, current_case=None)


def _context(coordinates, *, outcome: str | None) -> ResearchContext:
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
                name=concept, description=f"{concept} coordinate", role=VariableRole.VITAL,
                dtype="float64", analysis_window="icu_admission[0,24]h",
            )
            for concept in coordinates
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
            "Do physiologic trajectories over the first 24 h of an ICU stay form distinct classes?"
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


def _plan(
    labels: dict[str, str], *, outcome: str | None, outcome_label: str = "In-hospital death"
) -> tuple[AnalysisPlan, AnalysisPlan]:
    """Plan through the host as a run does: one labels call, then shape and bind.

    Returns the template's draft, which carries the design record, and the
    plan bound to the signed owners.
    """

    coordinates = tuple(labels)
    context = _context(coordinates, outcome=outcome)
    authorities = _authorities(coordinates)
    disclosure = authorities.planning_contract_context()
    request = build_family_spec_request(
        context,
        analysis_types=candidate_analysis_types(context),
        variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=_CITATIONS,
        required_primary_cohort_selection_mode="all_input_rows",
        planning_contract_context=disclosure,
    )
    every_label = {**labels, **({outcome: outcome_label} if outcome else {})}
    payload = {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": [],
        "reader_display_labels": [
            {"key": key, "value": every_label[key]} for key in request.required_reader_label_keys
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
        planning_contract_context=disclosure,
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


def test_a_roster_that_fits_is_named_in_the_estimand_without_another_construct() -> None:
    draft, _bound = _plan(_SHORT, outcome=None)

    estimand = draft.design_selection.selected.estimand
    assert estimand.startswith(
        "Candidate trajectory classes of 3 coordinate concepts "
        f"({', '.join(_SHORT.values())}), measured over 0–24 h after ICU admission "
        "on a fixed 4 h grid"
    )
    # The coordinates here are vital signs and a laboratory value; the design
    # must not call their classes organ-dysfunction classes.
    assert "organ" not in estimand.casefold()


@pytest.mark.parametrize("outcome", [None, "death"])
def test_seven_ordinary_coordinate_labels_compile_with_a_bounded_estimand(outcome) -> None:
    assert len(", ".join(_ORDINARY.values())) > 300

    draft, _bound = _plan(_ORDINARY, outcome=outcome)

    design = draft.design_selection.selected
    assert len(design.estimand) <= design_field_max_length("estimand")
    assert "7 coordinate concepts, named in the plan, measured over 0–24 h" in design.estimand
    # The plan the researcher reviews still names every coordinate.
    assert all(label in design.reviewable_plan[1] for label in _ORDINARY.values())


def test_the_longest_roster_and_labels_the_contracts_allow_still_compile() -> None:
    with pytest.raises(TrajectoryDesignError):
        load_trajectory_design({"coordinate_concepts": [*_MOST_COORDINATES, "hgb"]})
    labels = {name: _longest(name) for name in _MOST_COORDINATES}
    assert all(len(value) >= _label_bound() - 2 for value in labels.values())

    draft, _bound = _plan(labels, outcome="death", outcome_label=_longest("death"))

    design = draft.design_selection.selected
    assert len(design.estimand) <= design_field_max_length("estimand")
    assert f"{len(_MOST_COORDINATES)} coordinate concepts, named in the plan" in design.estimand
    # The item is a sentence, so its first label is capitalized.
    named = design.reviewable_plan[1].casefold()
    assert all(label.casefold() in named for label in labels.values())
