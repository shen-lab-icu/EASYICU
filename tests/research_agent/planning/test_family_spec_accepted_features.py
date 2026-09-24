"""Accepted primary inputs survive into the package-bound feature roster.

Dev9 M2 (static mortality model, 22 predictors) was reviewed on catalog
concepts.  After data preparation two of them, sex and admission type, were
string categories, which the prediction request never offered as features, so
the plan could not keep the accepted Table 1 and failed before execution
(``progressive_outline_accepted_baseline_incomplete``).  Nothing stopped a
Planner from dropping an accepted input either.  The fixtures here use generic
variables, not that study.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.agents.progressive_planner import (
    candidate_analysis_types,
    select_progressive_variables,
)
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.planning.accepted_analysis_inputs import (
    bind_analysis_inputs,
    candidate_analysis_inputs,
)
from easyicu.research_agent.planning.family_spec import (
    FamilySpecError,
    build_family_spec_request,
    validate_family_plan_spec,
)
from easyicu.research_agent.planning.family_spec.contract import spec_from_mapping
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole

from .family_spec_fixtures import (
    ALLOWED_CITATIONS,
    DIRECT_COMPARATORS,
    _phenotyping_context,
    _prediction_context,
    _prediction_payload,
)


def _with_string_categories(context):
    variables = [item for item in context.variables if item.name != "sex"]
    variables += [
        ConceptDescriptor(
            name="sex", description="patient sex", role=VariableRole.DEMOGRAPHIC,
            dtype="string", source_concept="sex",
            observed_domain={"n_unique": 2, "levels": ["Female", "Male"]},
        ),
        ConceptDescriptor(
            name="admission_type", description="admission type", role=VariableRole.DEMOGRAPHIC,
            dtype="string", source_concept="admission_type",
            observed_domain={"n_unique": 3, "levels": ["medical", "surgical", "other"]},
        ),
    ]
    return context.model_copy(update={"variables": variables})


def _request(context):
    return build_family_spec_request(
        context,
        analysis_types=candidate_analysis_types(context),
        variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=ALLOWED_CITATIONS,
        direct_comparator_literature_keys=DIRECT_COMPARATORS,
        comparison_literature_keys=DIRECT_COMPARATORS,
    )


def _accepting(context, *concepts):
    accepted = candidate_analysis_inputs(
        plan={"steps": [{
            "step_id": "primary_model", "planned_analysis_role": "primary",
            "inputs": [*concepts, "artifact:analysis_cohort"],
        }]},
        source_plan_sha256="c" * 64,
        selected_concepts=concepts,
        excluded=(),
    )
    return bind_analysis_inputs(context, accepted.model_dump(mode="json"))


def test_a_prediction_model_can_select_a_closed_string_category() -> None:
    request = _request(_with_string_categories(_prediction_context()))
    offered = {item.name: item for item in request.feature_candidates}

    assert offered["sex"].selectable and offered["sex"].allowed_codings == ["binary"]
    assert offered["admission_type"].selectable
    assert offered["admission_type"].allowed_codings == ["categorical"]
    # A phenotype's distance metric cannot encode a category.
    phenotyping = _request(_with_string_categories(_phenotyping_context()))
    assert not {
        item.name for item in phenotyping.feature_candidates if item.selectable
    } & {"sex", "admission_type"}


def test_every_accepted_input_stays_but_the_planner_picks_its_column() -> None:
    context = _accepting(
        _with_string_categories(_prediction_context()), "hr", "lact", "admission_type"
    )
    request = _request(context)

    assert {group.concept: group.columns for group in request.accepted_feature_groups} == {
        "hr": ["hr_max"], "lact": ["lactate_max"], "admission_type": ["admission_type"],
    }
    kept = spec_from_mapping(
        _prediction_payload(request, features=["hr_max", "lactate_max", "admission_type", "age"])
    )
    validate_family_plan_spec(kept, request)
    dropped = spec_from_mapping(
        _prediction_payload(request, features=["hr_max", "lactate_max", "age"])
    )
    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(dropped, request)
    assert caught.value.reason_code == "family_spec_accepted_input_missing"
    assert "admission_type" in str(caught.value)


def test_a_request_without_accepted_inputs_keeps_its_digest() -> None:
    request = _request(_prediction_context())
    payload = request.model_dump(mode="json")

    assert request.accepted_feature_groups == []
    payload.pop("accepted_feature_groups")
    assert request.request_sha256 == canonical_sha256(payload)


def test_an_accepted_input_the_family_cannot_fit_is_refused_before_the_provider() -> None:
    # A string category is an accepted predictor, but no phenotype feature.
    context = _accepting(_with_string_categories(_phenotyping_context()), "hr", "sex")

    with pytest.raises(FamilySpecError) as caught:
        _request(context)
    assert caught.value.reason_code == "family_spec_accepted_input_not_selectable"
    assert "sex" in str(caught.value)
