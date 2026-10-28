"""The shared owners and their historical executor imports stay identical."""

from types import SimpleNamespace

import pytest

from easyicu.research_agent.contracts import artifact_consumption, figure_plan, product_identity
from easyicu.research_agent.execution.runners import (
    _shared,
    cohort_flow_figure_executor,
    typed_input_binding,
)


def test_typed_input_primitives_are_reexported_from_the_leaf_owner() -> None:
    assert typed_input_binding.load_typed_input is artifact_consumption.load_typed_input
    assert (
        typed_input_binding.contained_regular_file
        is artifact_consumption.contained_regular_file
    )
    assert typed_input_binding.TypedInputBindingError is artifact_consumption.TypedInputBindingError
    assert typed_input_binding.BoundTypedInput is artifact_consumption.BoundTypedInput


def test_cohort_flow_declaration_is_reexported_from_the_leaf_owner() -> None:
    assert (
        cohort_flow_figure_executor.cohort_flow_figure_executor_declares_step
        is figure_plan.cohort_flow_figure_executor_declares_step
    )


def test_runner_identity_helpers_are_reexported_from_the_leaf_owner() -> None:
    assert _shared.figure_product is product_identity.figure_product
    assert _shared.method_head is product_identity.method_head


@pytest.mark.parametrize(
    "update",
    [
        {"planned_analysis_role": "primary"},
        {"inputs": ["table:cohort_flow", "table:other"]},
        {"expected_outputs": ["figure:Bad_Name"]},
        {"input_consumption_contracts": []},
        {"input_consumption_contracts": [SimpleNamespace(input_key="table:cohort_flow", mode="single_row")]},
    ],
)
def test_shared_declaration_refuses_a_widened_or_incompatible_step(update) -> None:
    declaration = {
        "planned_analysis_role": "auxiliary",
        "method": " Visualization with recorded counts ",
        "inputs": ["table:cohort_flow"],
        "expected_outputs": ["figure:cohort_flow"],
        "input_consumption_contracts": [
            SimpleNamespace(input_key="table:cohort_flow", mode="all_rows")
        ],
        "model_requirements": [],
        "table_one_spec": None,
        "trajectory_stability_spec": None,
        "exposure_outcome_distribution_spec": None,
    }
    predicate = figure_plan.cohort_flow_figure_executor_declares_step
    assert predicate(SimpleNamespace(**declaration))
    assert not predicate(SimpleNamespace(**{**declaration, **update}))
