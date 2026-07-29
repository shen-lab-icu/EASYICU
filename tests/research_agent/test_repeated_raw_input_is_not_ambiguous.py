"""A repeated raw input is redundant, not ambiguous.

Live E1 blocker, 2026-07-29, ``run_20260729T071550_24e5e8``. The run completed
steps 00-05, then died with an uncaught::

    ValueError: Planner-declared raw inputs must be unique

killing the whole benchmark item (``item_exception``, ``BENCH_EXIT=5``) after
~14 minutes of real provider spend. The cause was step 06 declaring 37 inputs
whose last two repeated names already present earlier in the same list::

    ..., 'lact_max', 'lact_measured', 'lact_n', 'sep3_sofa2_n', 'lact_n'

Every contract in ``resolved_raw_input_contracts`` is a pure function of the
name -- both the V1 and V2 branches resolve it from ``name`` alone and store it
in a name-keyed dict -- so a second occurrence produces a byte-identical entry
and cannot make the manifest ambiguous. Four lines further down the same call
chain, ``raw_contract_inputs_for_step`` already applied exactly the opposite
policy to the cohort predicate columns it appends::

    if resolved_column not in names:
        names.append(resolved_column)

One call chain therefore treated a repeated name as harmless in one line and
fatal in another. The uniqueness check had no test, no true positive, and one
production caller; it was deleted rather than downgraded to a warning.

``test_the_repeat_changes_nothing_at_all`` is the load-bearing one: it is what
makes deleting the check safe rather than merely convenient.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.research_context.typed import (
    resolved_raw_input_contracts,
)
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
)

# The tail of the real step-06 declaration, duplicates included.
_REAL_TAIL = (
    "lact_max",
    "lact_measured",
    "lact_n",
    "sep3_sofa2_n",
    "lact_n",
)
_DISTINCT = ("lact_max", "lact_measured", "lact_n", "sep3_sofa2_n")


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="test",
        cohort=CohortDescriptor(
            cohort_name="t",
            database="miiv",
            n_patients=1,
            n_stays=1,
            id_columns=["stay_id"],
            outcome_columns=["death"],
        ),
        variables=[
            ConceptDescriptor(name=name, dtype="float")
            for name in ("lact_max", "lact_measured", "lact_n", "sep3_sofa2_n")
        ],
    )


def test_the_real_step_06_declaration_resolves() -> None:
    """The regression: this list crashed the run."""

    payload = resolved_raw_input_contracts(_context(), _REAL_TAIL)

    assert set(payload["contracts"]) == set(_DISTINCT)


def test_the_repeat_changes_nothing_at_all() -> None:
    """Byte-identical output proves the repeat carried no information.

    This is the whole justification for deleting the check rather than
    reporting the duplicate: there is nothing to report. If a future change
    ever makes a contract depend on anything other than the name, the digests
    diverge and this test fails -- which is exactly when a uniqueness rule
    would need to come back.
    """

    context = _context()
    with_repeat = resolved_raw_input_contracts(context, _REAL_TAIL)
    without_repeat = resolved_raw_input_contracts(context, _DISTINCT)

    assert with_repeat == without_repeat
    assert with_repeat["contracts_sha256"] == without_repeat["contracts_sha256"]


def test_first_occurrence_order_is_preserved() -> None:
    """Dedupe keeps the Planner's declared order, it does not sort."""

    payload = resolved_raw_input_contracts(
        _context(), ("sep3_sofa2_n", "lact_n", "sep3_sofa2_n")
    )

    assert list(payload["contracts"]) == ["sep3_sofa2_n", "lact_n"]


def test_typed_products_are_still_excluded() -> None:
    """``kind:name`` inputs stay under the manifest's own inputs authority."""

    payload = resolved_raw_input_contracts(
        _context(), ("artifact:analysis_cohort", "lact_n", "artifact:analysis_cohort")
    )

    assert list(payload["contracts"]) == ["lact_n"]


def test_an_undeclared_name_still_fails_closed() -> None:
    """The check that does have true positives is untouched.

    Relaxing "declared twice" must not relax "never declared at all".
    """

    with pytest.raises(ValueError, match="lacks a context descriptor"):
        resolved_raw_input_contracts(_context(), ("lact_n", "not_a_real_column"))
