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

The rule had TWO owners, and the first fix hit the copy that was not on the
failing path. ``run_20260729T084225_cf05f9`` raised the same class again from
``authority/typed_binding.py::_write_resolved_inputs_manifest``, which is what
the execute phase actually calls. That traceback only exists because a step that
raises is now recorded instead of ending the run silently -- before that, this
crash produced no manifest and no line number, which is how a copy came to be
fixed in place of the rule.

That run also shows the host manufacturing the repeat itself, so blaming the
Planner for it was wrong. Step ``05_missingness_event_timing_audit``:

    revision 1 (Planner)                13 inputs, no ``lact_n``
    revision 2 (host input closure)     18 inputs, ``lact_n`` appended
    revision 3 (Replanner)              39 inputs, ``lact_n`` twice

``close_measurement_companion_inputs`` appends registered ``_measured``/``_n``
companions to a step's public inputs; the replan then rewrote that step's
inputs, kept the appended tail verbatim, and re-declared one of the same names
in its own expanded body. Neither side is wrong, which is the point: refusing
the list turns an ordinary merge into a dead run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent.authority.typed_binding import (
    _write_resolved_inputs_manifest,
)
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


# ---------------------------------------------------------------------------
# The second owner: the one the execute phase actually calls.
# ---------------------------------------------------------------------------

# Step 05 of run_20260729T084225_cf05f9 exactly as revision 3 declared it. The
# tail from ``charlson_measured`` on is what the host's input closure appended
# to revision 2; ``lact_n`` at index 15 is the replan's own re-declaration.
_REAL_STEP_05 = [
    "artifact:analysis_cohort",
    "sep3_sofa2_max",
    "sep3_sofa2_measured",
    "sep3_sofa2_n",
    "sep3_sofa2_first_time",
    "susp_inf_first_time",
    "susp_inf_measured",
    "susp_inf_n",
    "death",
    "death_time",
    "age",
    "sex",
    "charlson_max",
    "lact_first",
    "lact_measured",
    "lact_n",
    "sofa2_measured",
    "sofa2_n",
    "charlson_measured",
    "charlson_n",
    "lact_n",
]


_COHORT_BINDING = {"evidence_id": "development_execution_cohort", "kind": "artifact"}


def _write(tmp_path: Path, declared: list, *, bindings=None) -> dict:
    if bindings is None:
        bindings = {
            name: _COHORT_BINDING
            for name in dict.fromkeys(declared)
            if isinstance(name, str) and ":" in name
        }
    path = _write_resolved_inputs_manifest(
        run_dir=tmp_path,
        step_id="05_missingness_event_timing_audit",
        planner_declared_inputs=declared,
        bindings=bindings,
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_real_step_05_declaration_writes_a_manifest(tmp_path: Path) -> None:
    """The regression: this list ended the run nine steps early."""

    payload = _write(tmp_path, _REAL_STEP_05)

    assert payload["step_id"] == "05_missingness_event_timing_audit"


def test_the_written_manifest_holds_each_name_once_in_declared_order(
    tmp_path: Path,
) -> None:
    declared = _write(tmp_path, _REAL_STEP_05)["planner_declared_inputs"]

    assert declared.count("lact_n") == 1
    # First occurrence wins, so the replan's own ordering survives and only the
    # closure's trailing repeat is dropped.
    assert declared.index("lact_n") == _REAL_STEP_05.index("lact_n")
    assert declared == [
        name for i, name in enumerate(_REAL_STEP_05) if name not in _REAL_STEP_05[:i]
    ]


def test_the_manifest_now_satisfies_its_reader_by_construction(
    tmp_path: Path,
) -> None:
    """The load-bearing one for this half.

    ``authority/typed_input_receipt.py::_binding_for_input`` re-checks exactly
    these two predicates on the written manifest and raises
    ``TypedInputReceiptError`` if either fails. Deduplicating at the single
    write point makes them true by construction, so that reader keeps its check
    without a second policy having to agree with this one.
    """

    declared = _write(tmp_path, _REAL_STEP_05)["planner_declared_inputs"]

    assert len(set(declared)) == len(declared)
    assert all(declared.count(name) == 1 for name in declared)


def test_a_repeat_does_not_change_the_manifest_at_all(tmp_path: Path) -> None:
    """Same justification as the first owner: the repeat carried no payload.

    The contrast that matters is the declaration the host would have written
    had the replan not repeated the name -- not one with every copy removed,
    which is a genuinely different declaration.
    """

    deduped = [
        name for i, name in enumerate(_REAL_STEP_05) if name not in _REAL_STEP_05[:i]
    ]

    assert _write(tmp_path, _REAL_STEP_05) == _write(tmp_path, deduped)


def test_an_empty_or_non_string_input_still_fails_closed(tmp_path: Path) -> None:
    """Relaxing "declared twice" must not relax "not a name at all"."""

    for bad in ("", "   ", None, 7):
        with pytest.raises(ValueError, match="non-empty strings"):
            _write(tmp_path, ["age", bad])


def test_an_unbound_typed_input_still_fails_closed(tmp_path: Path) -> None:
    """The check with real true positives is untouched."""

    with pytest.raises(ValueError, match="exact Planner-declared typed inputs"):
        _write(tmp_path, ["artifact:analysis_cohort", "age"], bindings={})
