"""A declined free repair used to leave no trace at all.

fresh19 step ``07_primary_adjusted_association`` died on a TypeError whose
deterministic patch already existed and, applied by hand to that run's on-disk
``analysis.py`` and ``run.log``, produced the correct fix immediately. The step
nonetheless spent both LLM repairs and left 5 of 9 provider calls unused.

The run could not say whether the patch had been offered that failure and
declined, or never reached it: only the firing case wrote a key
(``step_record["runner_repair"]``). That ambiguity is what these tests close.

``test_a_decline_is_recorded`` and ``test_disabled_is_not_the_same_as_declined``
are the load-bearing ones -- they are the two states the run conflated.
"""

from __future__ import annotations

import ast
from pathlib import Path

from easyicu.research_agent.repairs.attempt_record import (
    ATTEMPT_KEY,
    declined_attempts,
    failure_signature,
    record_deterministic_runner_repair_attempt,
)

# The real tail of fresh19 steps/07_primary_adjusted_association/run.log.
_REAL_RUN_LOG = """                ---- stderr ----
                Traceback (most recent call last):
  File "/easyicu-analysis.py", line 188, in <module>
    age = checked_numeric(cohort, "age")
  File "/easyicu-analysis.py", line 80, in checked_numeric
    (original.notna() & converted.notna() & ~np.isfinite(converted.astype("float64"))).sum()
  File "/usr/local/lib/python3.11/site-packages/pandas/core/dtypes/astype.py", line 133, in _astype_nansafe
    return arr.astype(dtype, copy=True)
TypeError: float() argument must be a string or a real number, not 'StrictNumericInput'

[DockerRunner] container teardown confirmed before output collection
"""


def test_the_real_failure_signature_is_recovered() -> None:
    """The reader's first question: which failure was the patch asked about?"""

    signature = failure_signature(_REAL_RUN_LOG)

    assert signature is not None
    assert signature.startswith("TypeError: float() argument")
    assert "StrictNumericInput" in signature


def test_a_decline_is_recorded() -> None:
    """Previously invisible: a decline wrote nothing anywhere."""

    step_record: dict = {}
    record_deterministic_runner_repair_attempt(
        step_record,
        code="x = 1\n",
        run_log=_REAL_RUN_LOG,
        previous_repair=None,
        outcome="declined",
    )

    (attempt,) = step_record[ATTEMPT_KEY]
    assert attempt["outcome"] == "declined"
    assert attempt["repair_id"] is None
    assert "StrictNumericInput" in attempt["failure_signature"]
    assert len(attempt["code_sha256"]) == 64
    assert len(attempt["run_log_sha256"]) == 64


def test_disabled_is_not_the_same_as_declined() -> None:
    """ "The patch passed on this" and "the patch never ran" are different facts."""

    step_record: dict = {}
    record_deterministic_runner_repair_attempt(
        step_record,
        code="x = 1\n",
        run_log=_REAL_RUN_LOG,
        previous_repair=None,
        outcome="disabled",
    )

    assert step_record[ATTEMPT_KEY][0]["outcome"] == "disabled"
    assert declined_attempts(step_record) == []


def test_an_applied_attempt_names_the_patch() -> None:
    step_record: dict = {}
    record_deterministic_runner_repair_attempt(
        step_record,
        code="x = 1\n",
        run_log=_REAL_RUN_LOG,
        previous_repair=None,
        outcome="applied",
        repair_id="strict_numeric_input_result_projection_v1",
    )

    attempt = step_record[ATTEMPT_KEY][0]
    assert attempt["outcome"] == "applied"
    assert attempt["repair_id"] == "strict_numeric_input_result_projection_v1"


def test_attempts_accumulate_in_order_with_the_previous_repair() -> None:
    """The sequence is the point: a patch fires at most once per step.

    ``_deterministic_runner_repair`` gates every patch on
    ``previous_repair != <patch id>``. If an LLM repair reintroduces a defect
    the free fix already handled, the second decline is only visible here.
    """

    step_record: dict = {}
    record_deterministic_runner_repair_attempt(
        step_record,
        code="a\n",
        run_log=_REAL_RUN_LOG,
        previous_repair=None,
        outcome="applied",
        repair_id="patch_v1",
    )
    record_deterministic_runner_repair_attempt(
        step_record,
        code="b\n",
        run_log=_REAL_RUN_LOG,
        previous_repair="patch_v1",
        outcome="declined",
    )

    attempts = step_record[ATTEMPT_KEY]
    assert [a["sequence"] for a in attempts] == [1, 2]
    assert attempts[1]["previous_repair"] == "patch_v1"
    assert attempts[0]["code_sha256"] != attempts[1]["code_sha256"]
    assert len(declined_attempts(step_record)) == 1


def test_the_account_never_fails_its_subject() -> None:
    """A record of a failing step must not become a second failure."""

    class _Hostile(dict):
        def get(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    record_deterministic_runner_repair_attempt(
        _Hostile(),
        code=None,
        run_log=None,
        previous_repair=None,
        outcome="declined",
    )  # must not raise


def test_a_log_with_no_terminal_error_reports_nothing_rather_than_guessing() -> None:
    assert failure_signature("") is None
    assert failure_signature(None) is None
    assert failure_signature("all fine\nreturncode: 0\n") is None


def test_the_recorder_is_actually_wired_into_the_execute_phase() -> None:
    """A recorder nobody calls records nothing.

    Both branches must call it -- the enabled branch (applied/declined) and the
    disabled branch -- because "the flag was off" is exactly one of the states
    the run could not report.
    """

    source = Path(
        "src/easyicu/research_agent/execution/candidate_loop.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None)
        == "record_deterministic_runner_repair_attempt"
    ]

    assert len(calls) >= 2
    outcomes = {
        keyword.value.value
        for call in calls
        for keyword in call.keywords
        if keyword.arg == "outcome" and isinstance(keyword.value, ast.Constant)
    }
    assert "disabled" in outcomes
