"""The replay reported the locked cohort N under the one name nobody reads.

``CrossStepCohortLockValidator`` asks a fixed-cohort step to restate the
analytic N locked by a completed upstream step, and reads it by trying twenty
spellings in ``_COUNT_PATHS`` -- ``n_total``, ``cohort_n``, ``analytic_cohort_n``
and so on.  The deterministic robustness replay publishes exactly that number,
copied from its own primary contract, under ``analysis_cohort_n``.  That is not
one of the twenty.

So the gate read nothing, reported ``reported_summary_path: null`` and failed
the step closed -- for a summary that was carrying the correct value, equal to
the locked N, the whole time.

Measured over every recorded run: 15 summaries publish a cohort count ONLY as
``analysis_cohort_n``, invisible to the gate.  Where both an accepted spelling
and ``analysis_cohort_n`` are present they agree in 1 of 1 cases and disagree in
0 -- it is the same quantity, not a different one.

Both sibling deterministic producers already say ``n_total``:
``adjusted_association_executor`` and ``deterministic_missingness``.  The replay
was the odd one out, so it now says ``n_total`` too.  ``analysis_cohort_n``
stays, because the manuscript and figure layers read it.

Why this surfaced only now: ``_requires_fixed_cohort`` is a regex over the
Planner's free-text intent.  canary27's intent said "the locked complete-case
missingness sensitivity specification" and did not match; canary28's said "the
locked adult-only cohort" and did.  Identical artifacts, opposite verdicts --
recorded as a separate finding, not fixed here.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from easyicu.research_agent.audits.validators import CrossStepCohortLockValidator
from easyicu.research_agent.execution.runners import deterministic_robustness

_ACCEPTED = {".".join(path) for path in CrossStepCohortLockValidator._COUNT_PATHS}


def _summary_dict() -> ast.Dict:
    """The mapping the replay publishes, read from source.

    Assembling it for real needs a fitted cohort inside the sandbox; what has
    to hold is that the key is written at all and from the same expression as
    the count the replay already publishes.
    """

    tree = ast.parse(inspect.getsource(deterministic_robustness))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if "analysis_cohort_n" in keys:
            return node
    raise AssertionError("the replay summary no longer publishes a cohort count")


def _written() -> dict[str, str]:
    node = _summary_dict()
    return {
        key.value: ast.unparse(value)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def test_the_count_is_published_under_a_name_the_gate_reads() -> None:
    """The property that was false."""

    written = _written()
    readable = sorted(name for name in written if name in _ACCEPTED)
    assert readable, (
        "the replay publishes a cohort count under none of the names the "
        f"cohort-lock gate reads; it writes {sorted(written)[:8]}..."
    )


def test_it_is_the_same_number_the_replay_already_had() -> None:
    """Not a second, independently derived count.

    Two cohort counts computed two ways is how a step comes to report a
    number that disagrees with itself; the point is one value under two names.
    """

    written = _written()
    readable = [name for name in written if name in _ACCEPTED]
    for name in readable:
        assert written[name] == written["analysis_cohort_n"], (
            f"{name} is derived separately from analysis_cohort_n: "
            f"{written[name]!r} vs {written['analysis_cohort_n']!r}"
        )


def test_the_name_the_manuscript_layer_reads_is_still_there() -> None:
    """``analysis_cohort_n`` has three readers in the reporting and figure
    layers; adding a spelling must not cost them theirs."""

    assert "analysis_cohort_n" in _written()


# --- the gate, driven on the real shape ---------------------------------------


def _locked_parent(count: int) -> list[dict]:
    return [
        {
            "step_id": "06_primary_adjusted_association",
            "status": "ok",
            "step_summary": {"status": "ok", "n_total": count},
        }
    ]


class _Step:
    """The two fields the validator reads."""

    def __init__(self, intent: str) -> None:
        self.step_id = "10_robustness_replay"
        self.intent = intent


_FIXED_COHORT_INTENT = (
    "Re-estimate the locked adult-only cohort and complete-case sensitivity "
    "specifications without changing the primary exposure, outcome, or estimand."
)


def _summary(*, cohort_n: int, readable: bool) -> dict:
    summary = {"status": "ok", "analysis_cohort_n": cohort_n}
    if readable:
        summary["n_total"] = cohort_n
    return summary


def test_the_gate_passes_a_replay_that_reports_the_locked_count() -> None:
    findings = CrossStepCohortLockValidator().audit(
        step=_Step(_FIXED_COHORT_INTENT),
        step_summary=_summary(cohort_n=1000, readable=True),
        completed_step_records=_locked_parent(1000),
    )
    assert findings == []


def test_the_host_key_alone_is_now_enough() -> None:
    """The defect is closed at the reader, so the alias is no longer load-bearing.

    This used to assert the opposite -- that a summary carrying only
    ``analysis_cohort_n`` left the gate reporting "no count at all" -- because
    the fix of the day was to have the replay ALSO publish ``n_total``, a
    spelling the reader already accepted. Adding a second spelling for a value
    that already had one is how a key ends up meaning two things, and it did:
    one recorded summary used ``n_total`` for the number of variants compared,
    so the gate read an analysis cohort of 2 against a locked 1,000.

    Teaching the reader the host's own key instead fixes that and reaches 15
    further recorded summaries where the gate had no count at all and the value
    was sitting in the file the whole time.
    """

    findings = CrossStepCohortLockValidator().audit(
        step=_Step(_FIXED_COHORT_INTENT),
        step_summary=_summary(cohort_n=1000, readable=False),
        completed_step_records=_locked_parent(1000),
    )
    assert findings == []


def test_the_gate_keeps_its_teeth_without_the_alias() -> None:
    """Reading a new key is only right if a WRONG count is still visible there."""

    findings = CrossStepCohortLockValidator().audit(
        step=_Step(_FIXED_COHORT_INTENT),
        step_summary=_summary(cohort_n=812, readable=False),
        completed_step_records=_locked_parent(1000),
    )
    assert len(findings) == 1
    assert "812" in findings[0].message and "1000" in findings[0].message


def test_a_replay_that_really_changed_the_cohort_is_still_caught() -> None:
    """The gate must keep its teeth.

    Making the count readable is only right if a WRONG count is now visible
    too; otherwise this would trade a false alarm for a silent pass.
    """

    findings = CrossStepCohortLockValidator().audit(
        step=_Step(_FIXED_COHORT_INTENT),
        step_summary=_summary(cohort_n=812, readable=True),
        completed_step_records=_locked_parent(1000),
    )
    assert len(findings) == 1
    assert "812" in findings[0].message and "1000" in findings[0].message


# --- the recorded corpus ------------------------------------------------------

_CORPUS = Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


@pytest.mark.skipif(
    not _CORPUS.exists(), reason="recorded runs are not on this machine"
)
def test_the_two_names_never_disagree_in_recorded_runs() -> None:
    """Real bytes: publishing one as the other cannot invent a number.

    If any recorded summary carried an accepted spelling AND
    ``analysis_cohort_n`` with different values, they would be different
    quantities and this fix would be reporting the wrong one.
    """

    disagreements = []
    compared = 0
    for path in sorted(
        _CORPUS.glob("batch_*/*/aware/run_*/steps/*/outputs/step_summary.json")
    ):
        try:
            summary = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(summary, dict):
            continue
        theirs = CrossStepCohortLockValidator._as_count(
            summary.get("analysis_cohort_n")
        )
        if theirs is None:
            continue
        extracted = CrossStepCohortLockValidator._extract_count(summary)
        if extracted is None:
            continue
        compared += 1
        if extracted[0] != theirs:
            disagreements.append((path.parent.parent.name, extracted, theirs))

    if not compared:
        pytest.skip("no recorded summary carries both spellings")
    assert not disagreements, (
        "recorded summaries disagree between the accepted spelling and "
        f"analysis_cohort_n, so they are not one quantity: {disagreements[:5]}"
    )


@pytest.mark.skipif(
    not _CORPUS.exists(), reason="recorded runs are not on this machine"
)
def test_recorded_replays_the_gate_could_not_read_are_the_ones_this_fixes() -> None:
    """Real bytes: measure the reach.

    Recorded runs predate the fix, so summaries the gate cannot read are
    expected; what is asserted is that every one of them does hold the count
    under ``analysis_cohort_n``, which is what this fix republishes. A summary
    with no count anywhere would need a different fix and must not pass here
    silently.

    Scoped to THIS producer -- ``primary_model_replay`` is a mapping only the
    deterministic replay writes.  Measured over the corpus: 15 of its summaries,
    every one carrying ``analysis_cohort_n``.  The 6 Coder-authored robustness
    summaries report no cohort count under any name at all; that is a real and
    separate gap, recorded here rather than asserted away, and widening this
    population to cover them would fail on someone else's defect.
    """

    unexplained = []
    unreadable = 0
    for path in sorted(
        _CORPUS.glob("batch_*/*/aware/run_*/steps/*/outputs/step_summary.json")
    ):
        try:
            summary = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(summary, dict):
            continue
        if str(summary.get("analysis_family") or "") != "robustness_sensitivity":
            continue
        if not isinstance(summary.get("primary_model_replay"), dict):
            continue
        if CrossStepCohortLockValidator._extract_count(summary) is not None:
            continue
        unreadable += 1
        if (
            CrossStepCohortLockValidator._as_count(summary.get("analysis_cohort_n"))
            is None
        ):
            unexplained.append(path.parent.parent.name)

    if not unreadable:
        pytest.skip("every recorded robustness summary is already readable")
    assert not unexplained, (
        "recorded robustness summaries carry no cohort count under any name, "
        f"so republishing one would not reach them: {unexplained[:5]}"
    )
