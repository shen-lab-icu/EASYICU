"""A third reader demanded a field the contract never published.

``gates/plausibility_receipt`` exists so the instruction the Coder is given and
the gate that judges the answer are rendered from the same constants.  Its own
comment says so: "the instruction and the gate cannot disagree about a field
name -- which is the only part of the wording that has to match."

There was a third reader.  ``cohort_summary_executor._verified_plausibility_audit``
respelled the field names as literals and required a FOURTH one, ``compared_n``,
that no instruction has ever asked for.  A script that followed the published
contract exactly was refused, and on 2026-08-01 that killed the E1
cohort-summary step outright: ``RuntimeError: Plausibility receipt for age
lacks non-negative counts``, on a receipt whose three published counts were all
present and all zero.

Measured over the recorded corpus: 319 receipts.  241 carry ``compared_n`` --
every one of them written by the host's own rendered block -- and 78 do not,
written by hand to the published three-key shape.  All 78 were unacceptable to
a reader that had never told anyone it wanted a fourth field.

``compared_n`` is still checked when it is there.  It is the only test that can
catch a receipt flagging more values than it looked at, and dropping it for the
241 would lose real coverage.  It is simply not a requirement, because
requiring what the contract does not publish is the defect itself.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from easyicu.research_agent.execution.runners.cohort_summary_executor import (
    _verified_plausibility_audit,
)
from easyicu.research_agent.gates.plausibility_receipt import (
    RECEIPT_ABOVE_FIELD,
    RECEIPT_BELOW_FIELD,
    RECEIPT_COMPARED_FIELD,
    RECEIPT_CONTRACT_SENTENCE,
    RECEIPT_POLICY_FIELD,
    RECEIPT_POLICY_VALUE,
    RECEIPT_TOTAL_FIELD,
)

# What the 2026-08-01 E1 step actually wrote, and what the contract asks for.
PUBLISHED_SHAPE = {
    RECEIPT_POLICY_FIELD: RECEIPT_POLICY_VALUE,
    RECEIPT_BELOW_FIELD: 0,
    RECEIPT_ABOVE_FIELD: 0,
    RECEIPT_TOTAL_FIELD: 0,
}

# What the host's own rendered block writes.
HOST_SHAPE = {
    RECEIPT_POLICY_FIELD: RECEIPT_POLICY_VALUE,
    RECEIPT_BELOW_FIELD: 1,
    RECEIPT_ABOVE_FIELD: 2,
    RECEIPT_TOTAL_FIELD: 3,
    RECEIPT_COMPARED_FIELD: 10,
    "observed_n": 10,
}


def test_the_published_shape_is_accepted() -> None:
    """The property that was false, and that killed a step."""

    assert _verified_plausibility_audit(
        {"age": dict(PUBLISHED_SHAPE)}, expected_columns=["age"]
    ) == {"age": PUBLISHED_SHAPE}


def test_the_hosts_own_richer_shape_is_still_accepted() -> None:
    assert _verified_plausibility_audit(
        {"age": dict(HOST_SHAPE)}, expected_columns=["age"]
    ) == {"age": HOST_SHAPE}


def test_every_field_the_reader_requires_is_named_in_the_published_sentence() -> None:
    """The rule that keeps this from happening again.

    Anything the reader refuses a receipt for must appear in the sentence the
    Coder is given.  ``compared_n`` deliberately does not appear there -- and
    correspondingly must not be required.
    """

    for field in (RECEIPT_BELOW_FIELD, RECEIPT_ABOVE_FIELD, RECEIPT_TOTAL_FIELD):
        assert field in RECEIPT_CONTRACT_SENTENCE

    assert RECEIPT_COMPARED_FIELD not in RECEIPT_CONTRACT_SENTENCE
    assert (
        _verified_plausibility_audit(
            {"age": dict(PUBLISHED_SHAPE)}, expected_columns=["age"]
        )
        is not None
    ), "an unpublished field must never be a requirement"


def test_the_reader_respells_no_field_name() -> None:
    """It must read the constants, not its own copies of the strings.

    The literals are what let the two sides drift in the first place.
    """

    from easyicu.research_agent.execution.runners import cohort_summary_executor

    source = Path(cohort_summary_executor.__file__).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "_verified_plausibility_audit":
            continue
        literals = {
            child.value
            for child in ast.walk(node)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        }
        for field in (
            RECEIPT_BELOW_FIELD,
            RECEIPT_ABOVE_FIELD,
            RECEIPT_TOTAL_FIELD,
            RECEIPT_COMPARED_FIELD,
        ):
            assert field not in literals, f"{field} is respelled instead of imported"
        return
    raise AssertionError("_verified_plausibility_audit not found")


# --- what must still be refused ----------------------------------------------


@pytest.mark.parametrize(
    "record, why",
    [
        (
            {
                RECEIPT_BELOW_FIELD: 5,
                RECEIPT_ABOVE_FIELD: 6,
                RECEIPT_TOTAL_FIELD: 11,
                RECEIPT_COMPARED_FIELD: 3,
            },
            "flags more values than it compared",
        ),
        (
            {
                RECEIPT_BELOW_FIELD: 1,
                RECEIPT_ABOVE_FIELD: 1,
                RECEIPT_TOTAL_FIELD: 9,
            },
            "total is not the sum of its parts",
        ),
        (
            {
                RECEIPT_BELOW_FIELD: -1,
                RECEIPT_ABOVE_FIELD: 0,
                RECEIPT_TOTAL_FIELD: -1,
            },
            "a count cannot be negative",
        ),
        (
            {
                RECEIPT_BELOW_FIELD: True,
                RECEIPT_ABOVE_FIELD: 0,
                RECEIPT_TOTAL_FIELD: 1,
            },
            "a bool is not a count",
        ),
        (
            {RECEIPT_BELOW_FIELD: 0, RECEIPT_ABOVE_FIELD: 0},
            "a published field is missing",
        ),
        (
            {
                RECEIPT_BELOW_FIELD: 0,
                RECEIPT_ABOVE_FIELD: 0,
                RECEIPT_TOTAL_FIELD: 0,
                RECEIPT_COMPARED_FIELD: "ten",
            },
            "a stated compared_n that is not a count",
        ),
    ],
)
def test_a_broken_receipt_is_still_refused(record: dict, why: str) -> None:
    with pytest.raises(RuntimeError):
        _verified_plausibility_audit({"age": record}, expected_columns=["age"])


def test_a_receipt_that_misses_the_sealed_scope_is_still_refused() -> None:
    with pytest.raises(RuntimeError):
        _verified_plausibility_audit(
            {"age": dict(PUBLISHED_SHAPE)}, expected_columns=["age", "lactate"]
        )


def test_a_receipt_supplied_with_no_scope_is_still_refused() -> None:
    with pytest.raises(RuntimeError):
        _verified_plausibility_audit(
            {"age": dict(PUBLISHED_SHAPE)}, expected_columns=[]
        )


# --- the recorded corpus ------------------------------------------------------

_CORPUS = Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


_PUBLISHED_COUNTS = (RECEIPT_BELOW_FIELD, RECEIPT_ABOVE_FIELD, RECEIPT_TOTAL_FIELD)


def _recorded_receipts() -> list[dict]:
    found = []
    for path in sorted(
        _CORPUS.glob("batch_*/*/aware/run_*/steps/*/outputs/step_summary.json")
    ):
        try:
            summary = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        audit = summary.get("plausibility_audit") if isinstance(summary, dict) else None
        if not isinstance(audit, dict) or not audit:
            continue
        if all(isinstance(record, dict) for record in audit.values()):
            found.append(audit)
    return found


@pytest.mark.skipif(
    not _CORPUS.exists(), reason="recorded runs are not on this machine"
)
def test_every_recorded_receipt_that_states_the_published_counts_is_readable() -> None:
    """The load-bearing one: real bytes, both shapes, no exceptions.

    The population is exactly the receipts that state what the contract asks
    for.  Widening it past that would be asserting the reader accepts receipts
    the contract never sanctioned -- one recorded receipt carries an
    ``out_of_range_n`` with no ``below``/``above`` and a ``policy`` that is a
    dict rather than the declared string, and refusing THAT is correct.  The
    next test pins it, so the boundary is stated rather than skipped over.
    """

    readable = refused = without_compared = 0
    for audit in _recorded_receipts():
        if not all(
            all(field in record for field in _PUBLISHED_COUNTS)
            for record in audit.values()
        ):
            continue
        if any(RECEIPT_COMPARED_FIELD not in record for record in audit.values()):
            without_compared += 1
        try:
            _verified_plausibility_audit(audit, expected_columns=sorted(audit))
            readable += 1
        except RuntimeError:
            refused += 1

    if not readable and not refused:
        pytest.skip("no recorded step summary carries a plausibility receipt")
    assert without_compared, (
        "no recorded receipt omits compared_n; the corpus no longer exercises "
        "the shape this fix exists for"
    )
    assert refused == 0, (
        f"{refused} of {readable + refused} recorded receipts state every "
        "published count and are still refused"
    )


@pytest.mark.skipif(
    not _CORPUS.exists(), reason="recorded runs are not on this machine"
)
def test_a_recorded_receipt_missing_a_published_count_is_still_refused() -> None:
    """The other half: the fix must not have turned into "accept anything".

    If this stops finding a refusal, either the corpus changed or the reader
    has stopped enforcing the published fields.
    """

    incomplete = [
        audit
        for audit in _recorded_receipts()
        if not all(
            all(field in record for field in _PUBLISHED_COUNTS)
            for record in audit.values()
        )
    ]
    if not incomplete:
        pytest.skip("no recorded receipt omits a published count")
    for audit in incomplete:
        with pytest.raises(RuntimeError):
            _verified_plausibility_audit(audit, expected_columns=sorted(audit))
