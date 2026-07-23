from __future__ import annotations

import pytest

from easyicu.research_agent.execution.failure_classification import (
    RuntimeFailureClass,
    classify_runtime_failure,
)


@pytest.mark.parametrize(
    "diagnostic",
    [
        (
            "ValueError: Planner-declared Table 1 groups are absent from "
            "exposure: [opaque]"
        ),
        ("TableOneContractError: A Planner-declared Table 1 group is " "empty"),
    ],
)
def test_empty_closed_comparison_routes_to_plan_data_contract(diagnostic: str) -> None:
    decision = classify_runtime_failure(
        run_log=diagnostic,
        timed_out=False,
        step_id="02_table_one",
        returncode=1,
    )
    assert decision is not None
    assert decision.step_updates["runtime_failure_class"] == (
        RuntimeFailureClass.PLAN_DATA_CONTRACT.value
    )
    assert decision.step_updates["llm_repair_used"] is False
    assert decision.finding.validator == "runtime_plan_data_contract"


def test_runtime_failure_classifier_does_not_relabel_code_errors() -> None:
    assert (
        classify_runtime_failure(
            run_log="NameError: name 'model_frame' is not defined",
            timed_out=False,
            step_id="02_table_one",
            returncode=1,
        )
        is None
    )


def test_timeout_is_not_relabelled_from_partial_table_one_log() -> None:
    assert (
        classify_runtime_failure(
            run_log="A Planner-declared Table 1 group is empty",
            timed_out=True,
            step_id="02_table_one",
            returncode=124,
        )
        is None
    )
