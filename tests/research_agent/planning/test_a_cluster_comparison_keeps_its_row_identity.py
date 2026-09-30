"""A cluster comparison keeps its row identity in a materialized run.

``phenotyping.outcome_by_cluster`` joins frozen class labels to the cohort on
the row identity its typed spec names, and that identity must be one of the
step's inputs.  In a materialized run the identity is a sealed navigation
coordinate, reserved from analysis fields.  The compiler keeps it an input of
the comparison step only, and the Planner's context gate admits it there only,
as that step's identity.  Synthetic, case-neutral cross-sectional context.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.cohort.schema import (
    CohortSchemaError,
    materialized_input_column_authority,
    validate_plan_typed_bindings_against_context,
)

from .family_spec_fixtures import (
    _phenotyping_context,
    _phenotyping_payload,
    _request,
    _run,
)


def _materialized(context, *, reserved=("stay_id",)):
    """The context as a materialized run seals it: coordinates are not bound."""

    names = [variable.name for variable in context.variables]
    return context.model_copy(
        update={
            "materialized_inputs": SimpleNamespace(
                cohort=SimpleNamespace(
                    cohort_columns=list(dict.fromkeys([*reserved, *names])),
                    column_bindings={
                        name: object() for name in names if name not in reserved
                    },
                )
            )
        }
    )


def _compiled(context):
    request = _request(context, cohort_mode=None)
    _llm, result = _run(
        context,
        [
            json.dumps(
                _phenotyping_payload(
                    request,
                    features=["hr_max", "lactate_max", "map_min"],
                    baseline=["age", "sex"],
                    membership=None,
                )
            )
        ],
        required_primary_cohort_selection_mode=None,
    )
    return result.output


def _step(plan, step_id):
    return next(step for step in plan.steps if step.step_id == step_id)


def _replace_step(plan, step_id, **update):
    return plan.model_copy(
        update={
            "steps": [
                step.model_copy(update=update) if step.step_id == step_id else step
                for step in plan.steps
            ]
        }
    )


def test_the_comparison_reads_its_reserved_identity_and_no_other_step_does() -> None:
    context = _materialized(_phenotyping_context())
    authority = materialized_input_column_authority(context)
    assert authority.reserved_navigation_coordinates == ("stay_id",)
    assert "stay_id" not in authority.executable_columns

    plan = _compiled(context)

    comparisons = [
        step for step in plan.steps if step.phenotype_comparison_spec is not None
    ]
    assert [step.step_id for step in comparisons] == ["cluster_characterization"]
    comparison = comparisons[0]
    assert comparison.scientific_action_id == "phenotyping.outcome_by_cluster"
    assert comparison.phenotype_comparison_spec.identity_column == "stay_id"
    assert comparison.inputs.count("stay_id") == 1
    assert [step.step_id for step in plan.steps if "stay_id" in step.inputs] == [
        "cluster_characterization"
    ]
    # The Planner's context gate accepted this plan; it holds on a second look.
    validate_plan_typed_bindings_against_context(plan=plan, context=context)


def test_sealing_the_identity_changes_no_comparison_input() -> None:
    context = _phenotyping_context()
    assert materialized_input_column_authority(context).sealed_columns == ()

    plain = _compiled(context)
    sealed = _compiled(_materialized(context))

    assert [step.step_id for step in sealed.steps] == [
        step.step_id for step in plain.steps
    ]
    for before, after in zip(plain.steps, sealed.steps):
        if after.step_id == "cluster_characterization":
            assert after.inputs == before.inputs
        else:
            # Every other step drops the reserved identity, and only it.
            assert after.inputs == [name for name in before.inputs if name != "stay_id"]


def test_another_step_naming_the_identity_is_still_refused() -> None:
    context = _materialized(_phenotyping_context())
    plan = _compiled(context)
    fit = _step(plan, "primary_cluster_solution")
    plan = _replace_step(plan, fit.step_id, inputs=[*fit.inputs, "stay_id"])

    with pytest.raises(CohortSchemaError, match="raw name 'stay_id'") as caught:
        validate_plan_typed_bindings_against_context(plan=plan, context=context)

    message = str(caught.value)
    assert "reserved for host navigation" in message
    # Only the other step's reference is refused, not the comparison's own.
    assert "locations={'step inputs': 1}" in message


def test_the_comparison_admits_no_other_reserved_coordinate() -> None:
    context = _materialized(
        _phenotyping_context(), reserved=("stay_id", "icu_admit_time")
    )
    assert materialized_input_column_authority(
        context
    ).reserved_navigation_coordinates == ("icu_admit_time", "stay_id")
    plan = _compiled(context)
    comparison = _step(plan, "cluster_characterization")
    plan = _replace_step(
        plan, comparison.step_id, inputs=[*comparison.inputs, "icu_admit_time"]
    )

    with pytest.raises(CohortSchemaError, match="raw name 'icu_admit_time'") as caught:
        validate_plan_typed_bindings_against_context(plan=plan, context=context)

    assert "reserved for host navigation" in str(caught.value)
    assert "raw name 'stay_id'" not in str(caught.value)


def test_an_identity_the_run_did_not_seal_is_not_admitted_by_its_name() -> None:
    context = _materialized(_phenotyping_context())
    plan = _compiled(context)
    comparison = _step(plan, "cluster_characterization")
    plan = _replace_step(
        plan,
        comparison.step_id,
        inputs=["row_key" if name == "stay_id" else name for name in comparison.inputs],
        phenotype_comparison_spec=comparison.phenotype_comparison_spec.model_copy(
            update={"identity_column": "row_key"}
        ),
    )

    with pytest.raises(CohortSchemaError, match="raw name 'row_key'") as caught:
        validate_plan_typed_bindings_against_context(plan=plan, context=context)

    assert "not an exact executable sealed cohort column" in str(caught.value)
    assert "raw name 'stay_id'" not in str(caught.value)
