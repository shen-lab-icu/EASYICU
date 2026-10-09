"""A Planner that cannot express what a question needs says so, and the host checks it.

When a design element the question needs fits no offered family, the outline
declares a capability gap and drafts no steps, instead of steps that quietly
answer another question.  The host checks the claim against the study's typed
context first: a claim the context contradicts goes back to the Planner as a
repair; a verified claim, or one the host has no evidence about, stops planning
with the requirement as its cause and writes no checkpoint.
"""

from __future__ import annotations

import json
from typing import get_args

import pandas as pd
import pytest

from easyicu.research_agent.agents.progressive_payload import (
    progressive_outline_structured_output_request,
)
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.planning.capability_gap import check_capability_gap
from easyicu.research_agent.planning.family_spec.request import (
    SEALED_TRAJECTORY_SUITE_MARKER,
)
from easyicu.research_agent.planning.progressive_contract import (
    CapabilityGapRequirement,
    ProgressiveCapabilityGap,
    ProgressivePlanCompileError,
    ProgressivePlanOutline,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.structured_retry import StructuredResponseFailure
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole
from easyicu.research_agent.trajectory.contract import (
    infer_fixed_window_trajectory_metadata,
)
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context,
    _foundation_payload,
    _materialization_payloads,
    _outline_payload,
)


def _claim(requirement: str, concept: str | None = None) -> ProgressiveCapabilityGap:
    return ProgressiveCapabilityGap(requirement=requirement, concept=concept)


def _gap_outline(requirement: str, concept: str | None = None) -> dict:
    return {
        **_outline_payload(),
        "steps": [],
        "capability_gap": requirement,
        "capability_gap_concept": concept,
        "rationale": "The question groups the exposure by clinical thresholds.",
    }


def _run(responses: list[dict], checkpoints: list | None = None):
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)
    return llm, agent.run(
        _context(),
        checkpoint_callback=(checkpoints.append if checkpoints is not None else None),
    )


def test_an_outline_drafts_steps_or_declares_a_gap_never_both() -> None:
    outline = _outline_payload()

    with pytest.raises(ValueError, match="declares a capability_gap and drafts none"):
        ProgressivePlanOutline.model_validate(
            {**outline, "capability_gap": "estimand_unsupported"}
        )
    with pytest.raises(ValueError, match="drafts steps, or declares"):
        ProgressivePlanOutline.model_validate({**outline, "steps": []})
    with pytest.raises(ValueError, match="only that requirement names one"):
        ProgressivePlanOutline.model_validate(
            {**outline, "capability_gap_concept": "age_years"}
        )
    # Only a thresholds gap names the variable to group, on an outline and in a claim.
    for requirement, concept in (
        ("estimand_unsupported", "age_years"),
        ("levels_from_thresholds_unavailable", None),
    ):
        with pytest.raises(ValueError, match="only that requirement names one"):
            ProgressivePlanOutline.model_validate(_gap_outline(requirement, concept))
        with pytest.raises(ValueError, match="only that requirement names one"):
            _claim(requirement, concept)


def test_an_outline_without_a_gap_keeps_its_digest() -> None:
    dumped = ProgressivePlanOutline.model_validate(_outline_payload()).model_dump(
        mode="json"
    )

    assert "capability_gap" not in dumped
    assert "capability_gap_concept" not in dumped


@pytest.mark.parametrize(
    ("requirement", "concept", "context_update", "verification"),
    [
        # A continuous variable with no closed levels cannot be compared by group.
        ("levels_from_thresholds_unavailable", "age_years", {}, "verified"),
        # A binary exposure already has two levels to compare.
        ("levels_from_thresholds_unavailable", "exposure_flag", {}, "unverified"),
        ("levels_from_thresholds_unavailable", "glucose_max", {}, "unverified"),
        ("multiple_sources_required", None, {}, "verified"),
        (
            "multiple_sources_required",
            None,
            {"cross_database_validation": ["eicu"]},
            "unverified",
        ),
        ("longitudinal_representation_unavailable", None, {}, "verified"),
        ("estimand_unsupported", None, {}, "unverifiable"),
        ("design_element_unsupported", None, {}, "unverifiable"),
    ],
)
def test_the_host_checks_each_claim_against_the_study(
    requirement, concept, context_update, verification
) -> None:
    context = _context().model_copy(update=context_update)

    check = check_capability_gap(_claim(requirement, concept), context=context)

    assert check.verification == verification
    assert check.fact


def test_a_longitudinal_claim_does_not_hold_for_a_study_with_a_trajectory() -> None:
    gap = _claim("longitudinal_representation_unavailable")
    context = _context()
    window = ConceptDescriptor(
        name="severity_state_h0_6",
        role=VariableRole.ORDINAL_SCORE,
        dtype="int64",
        is_ordinal=True,
        fixed_window_trajectory=infer_fixed_window_trajectory_metadata(
            column_name="severity_state_h0_6",
            values=pd.Series([0, 1, 2], dtype="int64"),
            source_scale="ordinal",
        ),
    )
    with_windows = context.model_copy(
        update={"variables": [*context.variables, window]}
    )
    sealed_suite = SEALED_TRAJECTORY_SUITE_MARKER + json.dumps(
        {
            "sealed_representation_owner": "signed_trajectory_representation",
            "sealed_candidate_owner": "signed_trajectory_candidates",
            "coordinate_concepts": ["sofa_cardio", "sofa_resp"],
            "window_hours": [0, 72],
            "grid_width_hours": 24,
            "candidate_cluster_counts": [2, 3],
            "representation_outputs": ["trajectory_classes"],
        }
    )

    assert check_capability_gap(gap, context=with_windows).verification == "unverified"
    assert (
        check_capability_gap(
            gap, context=context, planning_contract_context=sealed_suite
        ).verification
        == "unverified"
    )


def test_a_checked_gap_stops_planning_with_its_cause_and_no_checkpoint() -> None:
    checkpoints: list = []

    with pytest.raises(ProgressivePlanCompileError) as caught:
        _run(
            [_gap_outline("levels_from_thresholds_unavailable", "age_years")],
            checkpoints,
        )

    stop = caught.value
    assert stop.reason_code == "progressive_capability_gap"
    assert (
        stop.easyicu_safe_diagnostic["cause_code"]
        == "levels_from_thresholds_unavailable"
    )
    assert stop.path == "capability_gap"
    (finding,) = stop.details["findings"]
    assert (finding["requirement"], finding["concept"], finding["verification"]) == (
        "levels_from_thresholds_unavailable",
        "age_years",
        "verified",
    )
    assert checkpoints == []


def test_a_gap_the_host_cannot_check_stops_as_unverifiable() -> None:
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _run([_gap_outline("estimand_unsupported")])

    assert caught.value.details["findings"][0]["verification"] == "unverifiable"


def test_a_claim_the_study_contradicts_goes_back_to_the_planner() -> None:
    llm, plan = _run(
        [
            _gap_outline("levels_from_thresholds_unavailable", "exposure_flag"),
            _outline_payload(),
            _foundation_payload(),
            *_materialization_payloads(),
        ]
    )

    assert plan.steps
    repair = llm.calls[1][0][-1].content
    assert "progressive_capability_gap_claim_unverified" in repair or (
        "the declared capability gap does not hold" in repair
    )


def test_a_contradicted_claim_on_every_attempt_exhausts_the_repairs() -> None:
    claim = _gap_outline("levels_from_thresholds_unavailable", "exposure_flag")

    with pytest.raises(StructuredResponseFailure) as caught:
        _run([claim] * 4)

    assert getattr(caught.value.__cause__, "reason_code", None) == (
        "progressive_capability_gap_claim_unverified"
    )


def test_the_strict_transport_offers_two_small_nullable_gap_fields() -> None:
    request = progressive_outline_structured_output_request(
        analysis_types=("association_study", "descriptive_epidemiology"),
        variable_names=("age_years", "outcome_flag"),
        scientific_action_ids=(),
    )
    schema = json.loads(request.schema_json)
    gap = schema["properties"]["capability_gap"]["anyOf"]
    concept = schema["properties"]["capability_gap_concept"]["anyOf"]

    assert {"capability_gap", "capability_gap_concept"} <= set(schema["required"])
    assert {"type": "null"} in gap
    assert {"type": "null"} in concept
    assert [item["enum"] for item in gap if "enum" in item] == [
        list(get_args(CapabilityGapRequirement))
    ]
    # The host checks the concept against the study, so the transport repeats
    # no variable list for it.
    assert all("enum" not in item for item in concept)
    assert "ProgressiveCapabilityGap" not in schema.get("$defs", {})
    assert "minItems" not in schema["properties"]["steps"]


def test_the_outline_shape_lists_the_gap_and_when_to_use_it() -> None:
    prompt = ProgressivePlannerAgent.request_messages(_context())[1].content

    assert '"capability_gap":null,"capability_gap_concept":null' in prompt
    assert (
        "capability_gap and capability_gap_concept stay null unless the question "
        "needs a design element" in prompt
    )
    assert (
        "do not change the analysis type, population or exposure to work around it"
        in prompt
    )
