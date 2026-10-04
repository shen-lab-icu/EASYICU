"""A dose-response question needs an exposure that can carry a gradient.

A gradient needs at least three ordered exposure levels or a continuous
exposure.  Asked for the dose-response gradient of a yes/no exposure, the
Planner chose the nearest answerable question, the two-level association,
and nothing in review compared the plan with the question: the run went on to
answer a question nobody asked.  Review now reads the dose-response cues of
the question and, when the exposure has two levels, blocks the plan for the
researcher's decision instead of letting a two-level contrast stand in for a
gradient.  Synthetic contexts only.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.planning.analysis_types import requested_dose_response_cues
from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    plan_revision_blocker_codes,
    remediation_route_for_finding,
    requested_dose_response_on_two_levels,
)

from .family_spec_fixtures import _descriptive_context, _request, _run

CODE = "REQUESTED_DOSE_RESPONSE_NOT_ESTIMABLE"
ASKS = (
    "Among adult ICU stays, characterise the dose-response relationship of the "
    "phenotype with in-hospital mortality."
)


def _question(context, text: str):
    return context.model_copy(update={"research_question": text})


@pytest.mark.parametrize(
    "question, cues",
    [
        ("Characterise the dose-response gradient of the exposure.", ("dose response",)),
        ("Is there a dose–response relationship with mortality?", ("dose response",)),
        ("Describe the exposure-response curve for mortality.", ("exposure response",)),
        ("Is the association dose dependent?", ("dose dependent",)),
        ("暴露与院内死亡是否存在剂量反应关系？", ("剂量反应",)),
    ],
)
def test_the_question_names_a_dose_response_relationship(question, cues) -> None:
    assert requested_dose_response_cues(_question(_descriptive_context(), question)) == cues


@pytest.mark.parametrize(
    "question",
    [
        "Is the alveolar-arterial gradient associated with mortality?",
        "Is there a secular trend in mortality across admission years?",
        "Does the phenotype change over time?",
    ],
)
def test_other_gradients_and_trends_are_not_dose_response_cues(question) -> None:
    assert requested_dose_response_cues(_question(_descriptive_context(), question)) == ()


def test_a_two_level_exposure_cannot_carry_the_requested_gradient() -> None:
    context = _question(_descriptive_context(), ASKS)

    assert requested_dose_response_on_two_levels(context) == (("dose response",), ("0", "1"))


def test_a_continuous_exposure_can_carry_it() -> None:
    context = _question(_descriptive_context(), ASKS).model_copy(
        update={"primary_exposure": "score_first"}
    )

    assert requested_dose_response_on_two_levels(context) == ((), ())


def _descriptive_plan(context):
    request = _request(context, cohort_mode="all_input_rows")
    labels = {
        "phenotype_flag": "Phenotype in the first 24 h",
        "death": "In-hospital death",
        "age": "Age (years)",
        "sex": "Patient sex",
        "score_first": "Chronic disease score",
        "phenotype_flag=0": "Phenotype absent",
        "phenotype_flag=1": "Phenotype present",
    }
    payload = {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": [],
        "baseline_variables": ["age"],
        "reader_display_labels": [
            {"key": key, "value": labels[key]}
            for key in [*request.required_reader_label_keys, *request.level_label_keys]
        ],
        "comparator_applications": [
            {
                "citation_key": key,
                "application": (
                    f"Compare this description with {key} on population, exposure "
                    "definition, time zero, and estimand without claiming novelty."
                ),
            }
            for key in request.direct_comparator_literature_keys
        ],
        "roster_decision_note": "Descriptive family: baseline variables from host-timed candidates.",
    }
    _llm, result = _run(
        context, [json.dumps(payload)], required_primary_cohort_selection_mode="all_input_rows"
    )
    return result.output


def _review(context, plan):
    return build_plan_scientific_review(
        context=context, plan=plan, literature=None,
        figure_strategy=build_article_figure_strategy(context), runtime_authority=None,
    )


def test_the_review_blocks_the_plan_for_the_researchers_decision() -> None:
    plain = _descriptive_context()
    plan = _descriptive_plan(plain)
    asked = _question(plain, ASKS + " " + plain.research_question)

    review = _review(asked, plan)

    finding = next(item for item in review.findings if item.code == CODE)
    assert finding.severity == "blocker"
    assert finding.requires_user_authorization is True
    assert "two levels (0, 1)" in finding.message
    # No plan revision can answer it: the researcher decides.
    assert remediation_route_for_finding(finding) == "study_authority_change"
    assert CODE in plan_revision_blocker_codes(list(review.findings))
    assert review.approval_allowed is False
    assert CODE not in {item.code for item in _review(plain, plan).findings}
