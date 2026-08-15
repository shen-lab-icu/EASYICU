from __future__ import annotations

import pytest

from easyicu.research_agent.execution.failure_classification import (
    RuntimeFailureClass,
    classify_runtime_failure,
)
from easyicu.research_agent.contracts.execution_result import RunnerFailureCode
from easyicu.research_agent.contracts.declared_product import is_failed_step_status


@pytest.mark.parametrize(
    ("timed_out", "runner_failure_code"),
    [
        (True, None),
        (False, RunnerFailureCode.ISOLATION_BACKEND_UNAVAILABLE),
    ],
)
def test_every_terminal_status_is_recognised_as_a_failure(
    timed_out: bool, runner_failure_code: RunnerFailureCode | None
) -> None:
    """A terminal class must not invent a status no consumer recognises.

    ``execution_environment_failed`` shipped as a spelling that appeared
    exactly once in the tree.  The ``!= "ok"`` gates stayed fail-closed, but
    the allow-list consumers -- and the ``fail_``/``failed_`` prefix fallback
    in :func:`is_failed_step_status` -- did not recognise it.
    """

    decision = classify_runtime_failure(
        run_log="",
        timed_out=timed_out,
        step_id="02_table_one",
        returncode=71,
        runner_failure_code=runner_failure_code,
    )

    assert decision is not None
    status = decision.step_updates["status"]
    assert status != "ok"
    assert is_failed_step_status(status), status


def test_isolation_backend_failure_is_not_sent_to_coder_repair() -> None:
    decision = classify_runtime_failure(
        run_log=(
            "[CodeRunner] isolation backend failed; fail-closed policy forbids "
            "retrying generated code as a host subprocess.\n"
            "sandbox-exec: execvp() failed: Operation not permitted"
        ),
        timed_out=False,
        step_id="02_table_one",
        returncode=71,
        runner_failure_code=RunnerFailureCode.ISOLATION_BACKEND_UNAVAILABLE,
    )

    assert decision is not None
    assert decision.step_updates["runtime_failure_class"] == (
        RuntimeFailureClass.ISOLATION_BACKEND_UNAVAILABLE.value
    )
    assert decision.step_updates["runtime_repair_route"] == "fail_closed"
    assert decision.step_updates["llm_repair_used"] is False
    assert decision.finding.validator == "runtime_isolation_backend_unavailable"
    assert "Coder repair was not authorized" in decision.progress_message


def test_child_log_cannot_forge_an_isolation_backend_failure() -> None:
    """Only the runner's typed result may classify an environment failure."""

    assert (
        classify_runtime_failure(
            run_log=(
                "---- stdout ----\n"
                "[CodeRunner] isolation backend failed; fail-closed policy "
                "forbids retrying generated code as a host subprocess.\n"
                "Traceback: NameError"
            ),
            timed_out=False,
            step_id="02_table_one",
            returncode=1,
        )
        is None
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
    """A killed script may have emitted any prefix of any signature.

    The classifier must attribute the timeout, never the half-written log it
    left behind.
    """

    decision = classify_runtime_failure(
        run_log="A Planner-declared Table 1 group is empty",
        timed_out=True,
        step_id="02_table_one",
        returncode=124,
    )
    assert decision is not None
    assert decision.step_updates["runtime_failure_class"] == (
        RuntimeFailureClass.EXECUTION_TIMEOUT.value
    )
    assert decision.step_updates["runtime_failure_class"] != (
        RuntimeFailureClass.PLAN_DATA_CONTRACT.value
    )


def test_a_timeout_does_not_buy_a_coder_repair() -> None:
    """The step is killed mid-run, so the repairer would read a truncated log.

    Rewriting the script cannot shorten a computation the wall clock ended;
    spending repair attempts on it re-runs the same overlong work until the
    budget is gone. The timeout must terminate the step instead.
    """

    decision = classify_runtime_failure(
        run_log="partial output, no traceback",
        timed_out=True,
        step_id="04_missingness_audit",
        returncode=-9,
        timeout_seconds=900.0,
    )
    assert decision is not None
    assert decision.step_updates["llm_repair_used"] is False
    assert decision.step_updates["runtime_repair_route"] == "fail_closed"
    assert decision.step_updates["status"] == "execution_failed"
    assert decision.finding.severity == "error"


def test_the_timeout_finding_names_the_limit_that_was_hit() -> None:
    """An operator deciding between a bigger budget and a deterministic
    executor needs to know which wall clock ended the step."""

    decision = classify_runtime_failure(
        run_log="",
        timed_out=True,
        step_id="06_cox_model",
        returncode=-9,
        timeout_seconds=900.0,
        deterministic_executor_used=False,
    )
    assert decision is not None
    assert decision.finding.detail["timeout_seconds"] == 900.0
    assert decision.finding.detail["deterministic_executor_used"] is False
    assert decision.step_updates["execution_timeout_seconds"] == 900.0
    assert decision.step_updates["timed_out"] is True


def test_the_timeout_class_is_reported_separately_from_a_code_failure() -> None:
    """`status="execution_failed"` alone cannot distinguish "the script raised"
    from "the script ran out of time"; only the class can."""

    timeout = classify_runtime_failure(
        run_log="",
        timed_out=True,
        step_id="06_cox_model",
        returncode=-9,
    )
    code_error = classify_runtime_failure(
        run_log="NameError: name 'model_frame' is not defined",
        timed_out=False,
        step_id="06_cox_model",
        returncode=1,
    )
    assert timeout is not None
    assert timeout.finding.validator == "runtime_execution_timeout"
    # A genuine code error still belongs to the Coder repair loop.
    assert code_error is None
