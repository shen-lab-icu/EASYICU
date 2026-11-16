"""A descriptive study whose exposure or outcome is sometimes unrecorded reports those rows.

The descriptive family template declared fail-closed missing-value policies.
On metadata no value is known to be missing, so the plan was accepted; the
prepared data then recorded stays without an age, and the plan compiled on
them was refused (``progressive_distribution_missingness_authority_invalid``):
on the hosted pilot every grouped descriptive study stopped at data
preparation.  The request now seals complete-case policies for a variable
with known missing values -- those rows leave the denominator and their
count travels in the table -- as the study-population occurrence does.  A
request without known missing values can explicitly retain its fail-closed
plan. Metadata-only requests separately declare that missingness is unknown.
Synthetic context only.
"""

from __future__ import annotations

import json

import pytest

from tests.support.distribution_policies import missing_policies_for
from tests.support.distribution_policies import EXCLUDE_MISSING, STOP_MISSING

from easyicu.research_agent.contracts.exposure_group_rules import (
    grouping_level_labels,
    read_grouping_rules,
)
from easyicu.research_agent.planning.family_spec import build_descriptive_skeleton
from easyicu.research_agent.planning.family_spec.contract import FamilyPlanSpec
from easyicu.research_agent.schema import ConceptDescriptor, MissingnessProfile

from .family_spec_fixtures import _descriptive_context, _request, _run


def _payload(request) -> dict:
    return {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": [],
        "distribution_missing_policies": missing_policies_for(request),
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


def test_without_known_missing_values_an_explicit_stop_policy_is_preserved() -> None:
    request, policies = _distribution(_descriptive_context())

    assert policies == ("all_declared_rows", "fail_closed", "fail_closed")
    assert request.distribution_missing_policies is None
    assert request.distribution_missingness_unobserved
    assert "distribution_missing_policies" not in json.dumps(request.model_dump(mode="json"))


def _grouped_context(*, handling="exclude", scale="nominal"):
    context = _descriptive_context()
    parameters = {
        "concept": "age", "window": None, "scale": scale,
        "groups": [
            {"id": "g1", "rule": {"summary": "value", "op": "<", "value": 65.0, "unit": "years"}},
            {"id": "g2", "rule": "otherwise"},
        ],
        "unmeasured": handling, "source_columns": {"value": "age"},
    }
    rules = read_grouping_rules(parameters)
    labels = {"g1": "Younger", "g2": "Older", "gU": "No age recorded"}
    levels = list(range(1, len(rules.levels) + 1))
    grouping = ConceptDescriptor(
        name="age_band", description="Age at admission groups", role="other", dtype="category",
        source_concept="age_band", derived_from_concepts=["age"],
        concept_enrichment_degraded=True,
        observed_domain={"n_unique": len(levels), "levels": levels},
        is_ordinal=scale == "ordinal", ordinal_levels=levels if scale == "ordinal" else None,
    )
    return context.model_copy(update={
        "primary_exposure": grouping.name,
        "variables": [*context.variables, grouping],
        "cohort": context.cohort.model_copy(update={"provenance": {
            **context.cohort.provenance,
            "exposure_group_labels": [{
                "variable": grouping.name, "concept": "age", "scale": scale,
                "levels": list(grouping_level_labels(rules, {key: labels[key] for key in rules.levels})), "parameters": parameters,
            }],
        }}),
    })


def test_a_declared_group_exclusion_binds_the_policy_before_missingness_is_measured():
    request = _request(_grouped_context(), cohort_mode="all_input_rows")
    assert request.distribution_missing_policies.missing_exposure_policy == "exclude_from_denominator"
    assert request.distribution_missingness["age_band"].n_missing is None
    payload = _payload(request)
    payload["distribution_missing_policies"] = STOP_MISSING
    draft = build_descriptive_skeleton(request, FamilyPlanSpec.model_validate(payload))
    distribution = next(m.step for m in draft.materializations if m.step.module_id == "exposure_outcome_distribution")
    assert distribution.missing_exposure_policy == "exclude_from_denominator"
    assert distribution.missing_outcome_policy == "fail_closed"


@pytest.mark.parametrize("handling", ["own_group"])
def test_a_grouping_without_an_exclusion_does_not_authorize_dropping_rows(handling):
    request = _request(_grouped_context(handling=handling), cohort_mode="all_input_rows")
    assert request.distribution_missing_policies is None
    assert not request.distribution_missingness
    payload = {**_payload(request), "distribution_missing_policies": STOP_MISSING}
    draft = build_descriptive_skeleton(request, FamilyPlanSpec.model_validate(payload))
    distribution = next(m.step for m in draft.materializations if m.step.module_id == "exposure_outcome_distribution")
    assert distribution.missing_exposure_policy == "fail_closed"


@pytest.mark.parametrize("scale", ["ordinal", "nominal"])
@pytest.mark.parametrize("measured", [False, True])
def test_grouped_exposure_compiles_with_exclusion_even_when_the_planner_proposes_stop(scale, measured):
    context = _grouped_context(scale=scale)
    if measured:
        context = context.model_copy(update={"variables": [
            v.model_copy(update={"missingness": MissingnessProfile(
                fraction_missing=0.004, n_missing=4, n_total=1000,
            )}) if v.name == "age_band" else v for v in context.variables
        ]})
    request = _request(context, cohort_mode="all_input_rows")
    payload = {**_payload(request), "distribution_missing_policies": STOP_MISSING}
    _llm, result = _run(context, [json.dumps(payload)], required_primary_cohort_selection_mode="all_input_rows")
    spec = next(s.exposure_outcome_distribution_spec for s in result.output.steps if s.exposure_outcome_distribution_spec)
    assert spec.exposure == "age_band"
    assert spec.missing_exposure_policy == "exclude_from_denominator"
    assert spec.missing_outcome_policy == "fail_closed"


@pytest.mark.parametrize("name", ["phenotype_flag", "sex", "death"])
def test_known_missingness_binds_only_its_role_and_discloses_the_count(name):
    context = _with_missing(**{name: 7})
    if name == "sex":
        context = context.model_copy(update={"primary_exposure": "sex"})
    request = _request(context, cohort_mode="all_input_rows")
    payload = _payload(request)
    payload["distribution_missing_policies"] = STOP_MISSING
    payload["baseline_variables"] = ["age"]
    _llm, result = _run(context, [json.dumps(payload)], required_primary_cohort_selection_mode="all_input_rows")
    spec = next(s.exposure_outcome_distribution_spec for s in result.output.steps if s.exposure_outcome_distribution_spec)
    role = "outcome" if name == "death" else "exposure"
    other = "exposure" if role == "outcome" else "outcome"
    assert getattr(spec, f"missing_{role}_policy") == "exclude_from_denominator"
    assert getattr(spec, f"missing_{other}_policy") == "fail_closed"
    assert spec.denominator_policy == ("observed_outcome_rows" if role == "outcome" else "all_declared_rows")
    assert any("7" in text and name in text for text in result.output.design_selection.selected.reviewable_plan)


def test_binding_exposure_exclusion_preserves_a_separately_declared_outcome_exclusion():
    request = _request(_with_missing(phenotype_flag=7), cohort_mode="all_input_rows")
    payload = _payload(request)
    payload["distribution_missing_policies"] = EXCLUDE_MISSING
    draft = build_descriptive_skeleton(request, FamilyPlanSpec.model_validate(payload))
    distribution = next(m.step for m in draft.materializations if m.step.module_id == "exposure_outcome_distribution")
    assert distribution.missing_outcome_policy == "exclude_from_denominator"
    assert distribution.denominator_policy == "observed_outcome_rows"


@pytest.mark.parametrize("name", ["phenotype_flag", "death"])
def test_a_remaining_compiler_conflict_returns_to_the_spec_retry(monkeypatch, name):
    context = _with_missing(**{name: 7})
    request = _request(context, cohort_mode="all_input_rows").model_copy(update={
        "distribution_missingness": {}, "distribution_missing_policies": None,
    })
    monkeypatch.setattr(
        "easyicu.research_agent.agents.family_spec_planner.build_family_spec_request",
        lambda *args, **kwargs: request,
    )
    rejected = {**_payload(request), "distribution_missing_policies": STOP_MISSING}
    corrected = {**rejected, "distribution_missing_policies": EXCLUDE_MISSING}
    llm, result = _run(context, [json.dumps(rejected), json.dumps(corrected)], required_primary_cohort_selection_mode="all_input_rows")
    assert len(llm.calls) == 2
    retry = llm.calls[1][0][-1].content
    assert "progressive_distribution_missingness_authority_invalid" in retry
    assert f"missing_{'outcome' if name == 'death' else 'exposure'}_policy" in retry
    assert "7" in retry
    assert result.output is not None
