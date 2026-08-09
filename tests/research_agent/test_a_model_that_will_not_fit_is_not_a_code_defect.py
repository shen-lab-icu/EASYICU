"""The host asked the Coder to repair the host's own fitter.

`adjusted_association_executor` raises `AdjustedAssociationError: declared model
'...' could not be fitted as declared: logistic fit did not converge`. That is
the HOST's script, and it is correct: estimability is a property of the declared
model and the cohort, not of the code. No rewrite of that script can change it.

It fell through `classify_runtime_failure` into the generic repair loop anyway.
Measured 2026-08-02 over every recorded run: 3 steps hit it, all 3 spent LLM
repairs on the host's own script, and one of those repairs added a keyword the
host does not accept --

    TypeError: run_adjusted_association_from_env() got an unexpected keyword
    argument 'fit_kwargs'

-- so a statistical outcome was replaced by a TypeError and the real reason was
lost. canary42's E3 died there, and its death truncated the rest of the plan.

Same shape as `EXECUTION_TIMEOUT`, whose own comment records the same lesson:
correct host code, a cause no rewrite can fix, repair budget burned.
"""

from __future__ import annotations

from easyicu.research_agent.execution.failure_classification import (
    RuntimeFailureClass,
    classify_runtime_failure,
)

#: canary42's real stderr, trimmed to the two lines the classifier reads.
_REAL_LOG = (
    "Traceback (most recent call last):\n"
    '  File "/easyicu-analysis.py", line 104, in <module>\n'
    "easyicu.research_agent.execution.runners.adjusted_association_executor."
    "AdjustedAssociationError: declared model "
    "'primary_in_hospital_mortality_by_aki_stage' could not be fitted as "
    "declared: logistic fit did not converge; sex treatment-coded against "
    "'Female'; adm treatment-coded against 'med'\n"
)


def _classify(log: str, *, deterministic: bool):
    return classify_runtime_failure(
        run_log=log,
        timed_out=False,
        step_id="07_primary_adjusted_mortality_association",
        returncode=1,
        deterministic_executor_used=deterministic,
    )


def test_the_host_fitter_saying_not_estimable_is_a_closed_class():
    decision = _classify(_REAL_LOG, deterministic=True)

    assert decision is not None
    assert (
        decision.step_updates["runtime_failure_class"]
        == RuntimeFailureClass.DETERMINISTIC_MODEL_NOT_ESTIMABLE.value
    )


def test_it_spends_no_llm_repair():
    """The whole point: the repair budget is not burned on host code."""

    decision = _classify(_REAL_LOG, deterministic=True)

    assert decision.step_updates["llm_repair_used"] is False
    assert decision.step_updates["runtime_repair_route"] == "fail_closed"


def test_it_fails_the_step_and_never_upgrades_it():
    decision = _classify(_REAL_LOG, deterministic=True)

    assert decision.step_updates["status"] == "execution_failed"
    assert decision.step_updates["diagnostic_only"] is True


def test_the_same_words_in_an_agent_script_stay_repairable():
    """The clause that keeps this from swallowing real Coder defects.

    An agent-written script raising a similarly named error describes code the
    Coder CAN fix, and must keep its repair.
    """

    assert _classify(_REAL_LOG, deterministic=False) is None


def test_another_refusal_from_the_same_owner_is_not_this_class():
    """`AdjustedAssociationError` alone is not enough -- that owner refuses for
    several reasons, and only non-estimability is unfixable by rewriting."""

    other = (
        "easyicu.research_agent.execution.runners.adjusted_association_executor."
        "AdjustedAssociationError: declared model 'x' names a covariate the "
        "cohort does not contain\n"
    )

    assert _classify(other, deterministic=True) is None


def test_a_timeout_still_wins():
    """Ordering: a killed script's partial log may contain any prefix, so the
    timeout class must not be displaced by reading it."""

    decision = classify_runtime_failure(
        run_log=_REAL_LOG,
        timed_out=True,
        step_id="07",
        returncode=-9,
        timeout_seconds=900.0,
        deterministic_executor_used=True,
    )

    assert (
        decision.step_updates["runtime_failure_class"]
        == RuntimeFailureClass.EXECUTION_TIMEOUT.value
    )


def test_the_message_tells_the_planner_what_to_change():
    """A closed class that names no remedy just moves the dead end."""

    message = _classify(_REAL_LOG, deterministic=True).finding.message

    assert "Re-declare the model" in message
    assert "rather than editing the executor" in message


def test_the_other_two_classes_still_classify():
    """Adding a class must not shadow the ones already there."""

    empty_group = "A Planner-declared Table 1 group is empty for the declared level\n"
    decision = _classify(empty_group, deterministic=True)

    assert (
        decision.step_updates["runtime_failure_class"]
        == RuntimeFailureClass.PLAN_DATA_CONTRACT.value
    )


def test_an_unrelated_failure_is_still_repairable():
    assert (
        _classify("ZeroDivisionError: division by zero\n", deterministic=True) is None
    )


def test_the_words_alone_without_the_typed_error_are_not_this_class():
    """statsmodels prints "did not converge" from agent code too.

    Both halves are required: the typed error says the HOST's fitter refused,
    the phrase says it refused for estimability. Either alone would swallow a
    Coder defect that a rewrite really could fix -- found by mutation, which
    dropped the typed-error half and no test noticed.
    """

    agent_side = (
        "/easyicu-analysis.py:88: ConvergenceWarning: Maximum Likelihood "
        "optimization failed to converge; the model did not converge\n"
    )

    assert _classify(agent_side, deterministic=True) is None
