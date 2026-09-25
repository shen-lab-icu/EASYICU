"""A frequently unmeasured confounder keeps its rows instead of shrinking the cohort.

The landmark categorical family used to drop every row missing any selected
covariate.  On routinely collected data one often-unmeasured laboratory
confounder could remove most of the cohort before the fit, and the rows it
removed were the rows where nobody ordered the test.  The host now reads each
candidate's measured missing share and, where the family's estimator supports
it, keeps those rows as an explicit unmeasured state.  The Planner can neither
see nor set that decision.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.family_spec_planner import family_spec_user_prompt
from easyicu.research_agent.agents.progressive_payload import (
    parse_progressive_step_materialization,
    progressive_structured_output_request,
)
from easyicu.research_agent.planning.family_spec import (
    LANDMARK_SPLINE_FAMILY_ID,
    keeps_unmeasured_covariate_rows,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveModelTermIntent,
)
from easyicu.research_agent.schema import MissingnessProfile

from .family_spec_fixtures import (
    PLANNER_ROSTER,
    _context,
    _request,
    _run,
    _spec_payload,
)
from .progressive_planner_fixtures import _materialization_payloads


def _measured(**shares: float):
    context = _context(exact=False)
    variables = [
        variable.model_copy(
            update={
                "missingness": MissingnessProfile(
                    fraction_missing=shares[variable.name],
                    n_missing=round(shares[variable.name] * 1000),
                    n_total=1000,
                )
            }
        )
        if variable.name in shares
        else variable
        for variable in context.variables
    ]
    return context.model_copy(update={"variables": variables})


def test_a_candidate_states_its_missing_share_only_when_it_was_measured() -> None:
    plain = _request(_context(exact=False))
    measured = _request(_measured(severity_score_24h=0.4, age=0.02))

    # Unmeasured contexts produce the request they always did.
    assert "missing_share" not in json.dumps(plain.model_dump(mode="json"))
    assert measured.candidate("severity_score_24h").missing_share == 0.4
    assert measured.candidate("age").missing_share == 0.02
    assert measured.candidate("sex").missing_share is None
    assert measured.request_sha256 != plain.request_sha256


def test_a_frequently_unmeasured_confounder_keeps_its_rows_in_the_primary_model() -> None:
    context = _measured(severity_score_24h=0.4, age=0.02)
    request = _request(context)

    _llm, result = _run(
        context, [json.dumps(_spec_payload(request, adjustment_set=PLANNER_ROSTER))]
    )

    primary = result.output.steps[3].model_requirements[0]
    assert primary.covariates == ["age", "sex", "severity_score_24h"]
    # Below the threshold, age keeps the complete-row default.
    assert primary.missing_category_covariates() == ("severity_score_24h",)
    step_ids = [step.step_id for step in result.output.steps]
    assert "severity_score_24h_functional_form" in step_ids
    selected = result.output.design_selection.candidates[0]
    assert any("explicit unmeasured state" in item for item in selected.reviewable_plan)


def test_without_a_measured_share_the_plan_keeps_the_complete_row_default() -> None:
    context = _context(exact=False)
    request = _request(context)

    _llm, result = _run(
        context, [json.dumps(_spec_payload(request, adjustment_set=PLANNER_ROSTER))]
    )

    primary = result.output.steps[3].model_requirements[0]
    assert primary.baseline_missing_handling is None
    assert "baseline_missing_handling" not in json.dumps(
        primary.model_dump(mode="json")
    )


def test_only_the_family_whose_estimator_fits_the_state_is_told_about_it() -> None:
    request = _request(_measured(severity_score_24h=0.4))
    spline = request.model_copy(update={"family_id": LANDMARK_SPLINE_FAMILY_ID})

    prompt = family_spec_user_prompt(request, variable_descriptions={})
    spline_prompt = family_spec_user_prompt(spline, variable_descriptions={})

    assert keeps_unmeasured_covariate_rows(request)
    assert not keeps_unmeasured_covariate_rows(spline)
    assert '"missing_share": 0.4' in prompt
    assert "explicit unmeasured state" in prompt
    assert "explicit unmeasured state" not in spline_prompt


def test_the_planner_transport_never_offers_the_decision() -> None:
    request = progressive_structured_output_request(
        analysis_types=["association_study"],
        variable_names=["exposure", "outcome", "lab"],
        scientific_action_ids=["association.adjusted_association"],
    )

    assert "missing_handling" not in request.schema_json


def test_a_provider_that_writes_the_decision_is_normalized_not_obeyed() -> None:
    payload = next(
        item
        for item in _materialization_payloads()
        if any(term["role"] == "covariate" for term in item["step"]["model_terms"])
    )
    for term in payload["step"]["model_terms"]:
        if term["role"] == "covariate":
            term["missing_handling"] = "unmeasured_category"

    parsed = parse_progressive_step_materialization(json.dumps(payload))

    covariates = [term for term in parsed.step.model_terms if term.role == "covariate"]
    assert covariates
    assert all(term.missing_handling is None for term in covariates)


def test_the_exposure_cannot_keep_its_unmeasured_rows() -> None:
    with pytest.raises(ValueError, match="exposure cannot keep"):
        ProgressiveModelTermIntent(
            name="exposure",
            role="exposure",
            coding="continuous",
            missing_handling="unmeasured_category",
        )


@pytest.mark.parametrize(
    "shares,refit_varies",
    [
        # The primary keeps severity's unmeasured rows; the refit drops them.
        ({"severity_score_24h": 0.4, "age": 0.02}, True),
        # The primary already fits complete rows; the refit restates it.
        ({}, False),
    ],
)
def test_the_complete_case_refit_is_described_for_what_it_is(shares, refit_varies) -> None:
    context = _measured(**shares)
    request = _request(context)

    _llm, result = _run(
        context, [json.dumps(_spec_payload(request, adjustment_set=PLANNER_ROSTER))]
    )

    plan = result.output
    replay = next(step for step in plan.steps if step.step_id == "robustness_replay")
    (spec,) = plan.robustness_specs
    reviewable = " ".join(plan.design_selection.candidates[0].reviewable_plan)
    assert ("restates the primary analysis" in replay.intent) is not refit_varies
    assert ("documents the primary analysis" in spec.description) is not refit_varies
    assert ("complete-case reanalysis" in reviewable) is refit_varies
    assert ("itself the complete-case analysis" in reviewable) is not refit_varies
    if refit_varies:
        assert "unmeasured for" in replay.intent
