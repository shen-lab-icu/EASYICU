"""An owner spelled three ways cannot claim a Planner who writes fifty.

``absolute_risk_context`` is a registered deterministic runner: it is in the
capability registry, advertised to the Planner, and wired into the execution
dispatcher.  Its ownership predicate carried a three-entry method allowlist --
``absolute_risk_context``, ``descriptive_context``, ``exposure_outcome_summary``.

MEASURED over 89 recorded steps promising ``table:absolute_risk_context``, the
Planner wrote NONE of those three.  It wrote ``descriptive`` 52 times,
``descriptive_binary_outcome_summary`` 7, ``prevalence_and_absolute_risk_summary``
5, then a tail of synonyms.  The owner claimed 0 of 89.  The Coder wrote the
table every time, so it had a different shape every run, and the figures drawn
over it died **46 times out of 47**.  The sibling table that a real owner does
write ran the other way: 40 of 51 figures over it passed.

So the allowlist did not protect anything.  Re-measured over all 1,965 recorded
plan steps, removing it claims 77 of the 89 and adds exactly ZERO steps that do
not promise the product: the clauses beside it -- no figure output, no effect
method or effect output, and a closed product set -- were already doing all of
the discrimination.

This file holds both halves in place: the owner must be reachable from the words
the Planner actually writes, and it must still refuse everything it refused
before.
"""

from __future__ import annotations

import collections
import json
import pathlib

import pytest

from easyicu.research_agent.execution import phase as execution_phase
from easyicu.research_agent.execution.phase import (
    _absolute_risk_context_runner_owns_step as owns_step,
)

_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")
_PRODUCT = "table:absolute_risk_context"

#: The method heads the Planner actually wrote, with their recorded counts.
#: Not one of them was in the allowlist this file removed.
_REAL_METHOD_HEADS = (
    ("descriptive", 52),
    ("descriptive_binary_outcome_summary", 7),
    ("prevalence_and_absolute_risk_summary", 5),
    ("descriptive_stage_stratified_outcomes", 2),
    ("descriptive_absolute_risk", 2),
    ("descriptive_prevalence_and_absolute_risk", 2),
    ("descriptive_binary_exposure_outcome_summary", 2),
    ("descriptive_ordinal_gradient", 1),
)


# ---------------------------------------------------------------------------
# Reachable from what the Planner writes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method,_count", _REAL_METHOD_HEADS)
def test_every_method_the_planner_really_wrote_is_owned(method: str, _count: int):
    """Each of these was recorded on a real step and refused by the allowlist."""

    assert owns_step(method, "04_absolute_risk_context", [_PRODUCT])


def test_the_three_allowlisted_spellings_still_work():
    """Removing a filter must not lose the cases it did admit."""

    for method in (
        "absolute_risk_context",
        "descriptive_context",
        "exposure_outcome_summary",
    ):
        assert owns_step(method, "04_context", [_PRODUCT])


def test_the_allowlist_constant_is_gone_not_merely_bypassed():
    """A constant left in place invites the clause back.

    Its removal is the assertion: nothing may consult a hard-coded method
    spelling to decide this owner's claim.
    """

    assert not hasattr(execution_phase, "_ABSOLUTE_RISK_CONTEXT_METHODS")


# ---------------------------------------------------------------------------
# Everything it refused before, it still refuses
# ---------------------------------------------------------------------------


def test_a_figure_step_is_still_refused():
    """This owner writes a table; the renderer beside it draws the figure."""

    assert not owns_step(
        "descriptive",
        "06_absolute_risk_context_figure",
        ["figure:absolute_risk_context"],
    )
    assert not owns_step(
        "absolute_risk_context", "06_fig", ["figure:absolute_risk_context"]
    )


def test_an_effect_bearing_step_is_still_refused():
    """The primary estimand stays agent-owned; this owner never fits a model."""

    assert not owns_step(
        "absolute_risk_context", "07_primary", ["table:adjusted_odds_ratio"]
    )
    assert not owns_step("descriptive", "07_primary", ["table:adjusted_odds_ratio"])
    for method in ("adjusted_association_models", "logistic_regression", "cox_model"):
        assert not owns_step(method, "07_primary", [_PRODUCT])


def test_a_step_owning_other_artefacts_is_still_refused():
    """A reconciliation step mentioning absolute risk owns different products.

    This is the case the removed allowlist appeared to cover; the closed
    product-set clause is what actually covers it, which is why removing the
    allowlist changed nothing here.
    """

    assert not owns_step(
        "data_quality_audit",
        "06_absolute_risk_context_reconciliation",
        [
            "table:absolute_risk_representation_reconciliation",
            "table:reconciled_absolute_risk",
            "log:representation_gap_notes",
        ],
    )
    assert not owns_step(
        "descriptive",
        "06_reconciliation",
        ["table:absolute_risk_representation_reconciliation"],
    )


def test_a_step_promising_nothing_this_owner_emits_is_refused():
    assert not owns_step("descriptive", "02_table_one", ["table:table_one"])
    assert not owns_step("descriptive", "01_cohort", ["artifact:analysis_cohort"])
    assert not owns_step("descriptive", "03_audit", [])


# ---------------------------------------------------------------------------
# Reachability and blast radius, re-measured on the recorded corpus
# ---------------------------------------------------------------------------


def test_the_corpus_claims_the_product_and_nothing_else():
    """Re-derives the 77-of-89 and the zero, rather than restating them.

    The zero is the load-bearing half: a predicate widened until it claims
    steps that never promised this product would be worse than the allowlist.
    """

    if not _CORPUS.exists():
        pytest.skip("recorded run corpus is not mounted")

    promised = collections.Counter()
    intruders = 0
    for path in _CORPUS.glob("batch_*/*/aware/run_*/analysis_plan.json"):
        try:
            plan = json.loads(path.read_text())
        except Exception:  # noqa: BLE001 - a malformed plan is not this subject
            continue
        for step in plan.get("steps", []):
            outputs = [str(value) for value in (step.get("expected_outputs") or [])]
            method = str(step.get("method") or "")
            claimed = owns_step(method, str(step.get("step_id") or ""), outputs)
            if _PRODUCT in outputs:
                promised["owned" if claimed else "declined"] += 1
            elif claimed:
                intruders += 1

    if not promised:
        pytest.skip("no recorded plan promises the product")
    assert promised["owned"] > 0, "the owner is still unreachable on real input"
    assert promised["owned"] > promised["declined"], promised
    assert intruders == 0, f"{intruders} steps not promising the product were claimed"


def test_the_recorded_methods_are_really_absent_from_any_allowlist():
    """Anchors the whole file: the mismatch was real, not hypothesised."""

    if not _CORPUS.exists():
        pytest.skip("recorded run corpus is not mounted")

    heads = collections.Counter()
    for path in _CORPUS.glob("batch_*/*/aware/run_*/analysis_plan.json"):
        try:
            plan = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        for step in plan.get("steps", []):
            outputs = [str(value) for value in (step.get("expected_outputs") or [])]
            if _PRODUCT not in outputs:
                continue
            head = str(step.get("method") or "").strip().lower().split(" with ", 1)[0]
            heads[head] += 1

    if not heads:
        pytest.skip("no recorded plan promises the product")
    # The single most common spelling must be one the old allowlist refused.
    most_common = heads.most_common(1)[0][0]
    assert most_common not in {
        "absolute_risk_context",
        "descriptive_context",
        "exposure_outcome_summary",
    }, most_common
