"""A descriptive study whose exposure or outcome is sometimes unrecorded reports those rows.

The descriptive family template declared fail-closed missing-value policies.
On metadata no value is known to be missing, so the plan was accepted; the
prepared data then recorded stays without an age, and the plan compiled on
them was refused (``progressive_distribution_missingness_authority_invalid``):
on the hosted pilot every grouped descriptive study stopped at data
preparation.  The request now seals complete-case policies for a variable
with known missing values -- those rows leave the denominator and their
count travels in the table -- as the study-population occurrence does.  A
request without known missing values keeps its fail-closed plan and its
identity.  Synthetic context only.
"""

from __future__ import annotations

import json

from easyicu.research_agent.schema import MissingnessProfile

from .family_spec_fixtures import _descriptive_context, _request, _run


def _payload(request) -> dict:
    return {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": [],
        "baseline_variables": ["age", "sex"],
        "reader_display_labels": [
            {"key": key, "value": f"Reader label {index}"}
            for index, key in enumerate([*request.required_reader_label_keys, *request.level_label_keys])
        ],
        "comparator_applications": [
            {
                "citation_key": key,
                "application": (
                    f"Compare this description with {key} on population, exposure definition, "
                    "time zero, and estimand without copying its design or claiming novelty."
                ),
            }
            for key in request.direct_comparator_literature_keys
        ],
        "roster_decision_note": "Descriptive family: no adjustment; baseline variables chosen from host-timed candidates.",
    }


def _with_missing(**n_missing: int):
    context = _descriptive_context()
    return context.model_copy(
        update={
            "variables": [
                variable.model_copy(
                    update={
                        "missingness": MissingnessProfile(
                            fraction_missing=n_missing[variable.name] / 1000,
                            n_missing=n_missing[variable.name],
                            n_total=1000,
                        )
                    }
                )
                if variable.name in n_missing
                else variable
                for variable in context.variables
            ]
        }
    )


def _distribution(context):
    request = _request(context, cohort_mode="all_input_rows")
    _llm, result = _run(
        context, [json.dumps(_payload(request))], required_primary_cohort_selection_mode="all_input_rows"
    )
    (step,) = [step for step in result.output.steps if step.exposure_outcome_distribution_spec is not None]
    spec = step.exposure_outcome_distribution_spec
    return request, (spec.denominator_policy, spec.missing_exposure_policy, spec.missing_outcome_policy)


def test_rows_without_a_recorded_exposure_or_outcome_leave_the_denominator_with_their_count() -> None:
    request, policies = _distribution(_with_missing(phenotype_flag=40, death=3))

    assert policies == ("observed_outcome_rows", "exclude_from_denominator", "exclude_from_denominator")
    assert request.distribution_missing_policies is not None


def test_an_unrecorded_exposure_alone_keeps_every_outcome_row() -> None:
    _request_, policies = _distribution(_with_missing(phenotype_flag=40))

    assert policies == ("all_declared_rows", "exclude_from_denominator", "fail_closed")


def test_without_known_missing_values_the_plan_and_the_request_are_unchanged() -> None:
    request, policies = _distribution(_descriptive_context())

    assert policies == ("all_declared_rows", "fail_closed", "fail_closed")
    assert request.distribution_missing_policies is None
    assert "distribution_missing_policies" not in json.dumps(request.model_dump(mode="json"))
