"""A legitimately un-estimated statistic had no representation at all.

Two host readers of the same sidecar imposed contradictory requirements, so no
JSON document satisfied both:

* ``contracts/result_envelope.py::_first_finite_number`` -- a numeric key that
  is present must be a finite number. Absent is fine.
* ``execution/runners/robustness_figure_executor.py::_load_statistic`` -- the
  ``value`` key must be PRESENT (it raises "sidecar records no value"
  otherwise), and reads ``null`` as "bound, not estimated", a state it
  deliberately keeps distinct from "not bound".

Omit the key and the figure raises; write ``null`` and the step is blocked as
malformed. The producer had no third option.

It is not an edge case. MEASURED 2026-08-02 over the recorded corpus: 25 of 64
``complete_case_n.json`` sidecars carry ``value: null``, across four different
tasks (e1, e3, m1, m2), and each of those blocked its step. The commonest
cause is a locked robustness grid that declares no complete-case missing-data
specification -- there is genuinely no complete-case N, and reporting one would
be a fabrication.

The missing piece was vocabulary, not permission: the sidecar can now SAY it
was not estimated, and only that explicit declaration is honoured.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.contracts.result_envelope import _first_finite_number


def _probe(payload: dict) -> tuple[object, list[str]]:
    issues: list = []
    value = _first_finite_number(
        payload,
        ["value"],
        product_id="statistic:complete_case_n",
        field_name="value",
        issues=issues,
    )
    return value, [issue.code for issue in issues]


def test_a_declared_non_estimate_is_not_a_malformed_number() -> None:
    """The state the corpus is full of, and the one that had no representation."""

    value, codes = _probe(
        {
            "statistic": "complete_case_n",
            "value": None,
            "estimated": False,
            "not_estimated_reason": (
                "the locked robustness grid declares no complete_case "
                "missing-data specification"
            ),
        }
    )

    assert codes == [], (
        "a statistic that declares it was not estimated is making a statement, "
        "not emitting a broken number; blocking the step here is what killed "
        "25 of 64 recorded robustness sidecars"
    )
    assert value is None


def test_a_bare_null_with_no_declaration_still_fails() -> None:
    """The relaxation has to be narrow or it stops being a check.

    An undeclared null is indistinguishable from a producer that forgot to
    compute its own statistic, which is a real bug this check exists to catch.
    """

    _, codes = _probe({"statistic": "complete_case_n", "value": None})

    assert codes == ["invalid_statistic_numeric_field"]


@pytest.mark.parametrize(
    "declared",
    [True, "false", "no", 0, None],
    ids=["estimated-true", "string-false", "string-no", "zero", "explicit-none"],
)
def test_only_a_real_boolean_false_counts_as_the_declaration(declared: object) -> None:
    """``estimated`` must actually say false -- not merely be falsy or absent.

    A truthiness test would let ``0``, ``""`` and a stray ``None`` waive the
    numeric check, which is how a narrow exemption turns into a hole.
    """

    _, codes = _probe(
        {"statistic": "complete_case_n", "value": None, "estimated": declared}
    )

    assert codes == ["invalid_statistic_numeric_field"]


def test_a_real_number_is_unaffected_by_the_declaration_path() -> None:
    value, codes = _probe(
        {"statistic": "complete_case_n", "value": 41379, "estimated": True}
    )

    assert codes == []
    assert value == 41379


def test_a_non_numeric_value_still_fails_even_when_not_estimated() -> None:
    """``estimated: false`` waives a null, not any value the producer likes."""

    _, codes = _probe(
        {"statistic": "complete_case_n", "value": "41379", "estimated": False}
    )

    assert codes == ["invalid_statistic_numeric_field"]
