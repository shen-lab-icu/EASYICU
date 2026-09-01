"""The measurement that gave "did that change help?" a denominator.

Every canonical-9 run writes a different plan, so two consecutive runs of the
same case are not two samples of one experiment: fresh17 and fresh19 ran the
same benchmark case two hours apart and produced different step ids, different
bundles and different step counts.  ``tools/measure_executor_ownership.py``
scores a *fixed* corpus of recorded plans instead.

``test_the_same_audit_work_is_owned_only_when_it_is_declared_alone`` is the
load-bearing one: it holds the two real plans side by side and shows that the
identical scientific work crossed the owned/unowned line on nothing but how the
Planner bundled it.

``test_the_ledger_asks_the_selector_rather_than_the_predicates`` is the other:
a measurement that re-derives ownership from the owners' own predicates becomes
a second registry and drifts away from what the selector actually decides.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tools.measure_executor_ownership import (  # noqa: E402
    OwnershipLedger,
    load_plan,
    measure_plan,
    referenced_concept_ids,
    unowned_products,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "real_plan_steps_fresh17_fresh19.json"
_TOOL = Path("tools/measure_executor_ownership.py")

# The one step that carries the defect, in each run's own spelling.
_AUDIT_STEP = {
    "fresh17": "05_missingness_event_timing_audit",
    "fresh19": "06_missingness_event_timing_audit",
}


def _plans() -> dict[str, tuple[object, list, bool]]:
    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    plans = {}
    for entry in document["plans"]:
        raw = entry["plan"]
        plan, granted = load_plan(raw)
        assert plan is not None, f"{entry['label']} plan did not parse"
        plans[entry["label"]] = (plan, raw["steps"], granted)
    return plans


@pytest.fixture(scope="module")
def rows() -> dict[str, dict]:
    measured = {}
    for label, (plan, raw_steps, _granted) in _plans().items():
        measured[label] = {row.step_id: row for row in measure_plan(plan, raw_steps)}
    return measured


def test_the_same_audit_work_is_owned_only_when_it_is_declared_alone(rows) -> None:
    """The defect, stated over two real plans.

    fresh17 declared one audit product and the host owned the step: 0 provider
    calls, ok.  fresh19 declared the same product plus two more the runner
    cannot emit, and because ownership is all-or-nothing the host declined all
    three -- the step went to the LLM coder and was blocked with 6 of its 9
    provider calls unspent.

    When ownership becomes per-product this assertion has to change: fresh19's
    step should then be claimed for the product the host can emit, leaving only
    the remainder to the coder.  Updating it is the point, not a nuisance.
    """

    seventeen = rows["fresh17"][_AUDIT_STEP["fresh17"]]
    nineteen = rows["fresh19"][_AUDIT_STEP["fresh19"]]

    assert seventeen.method == nineteen.method
    assert seventeen.declared_products == ("table:missingness_measurement_audit",)
    assert nineteen.declared_products == (
        "table:missingness_measurement_audit",
        "table:event_timing_audit",
        "table:sofa2_component_completeness_audit",
    )

    assert seventeen.upper_owner == "declared_missingness_audit_products"
    assert nineteen.upper_owner is None

    # The product fresh19 shares with fresh17 is emittable on its own, so the
    # decline is about the bundle and not about that product.
    assert seventeen.declared_products[0] in nineteen.declared_products


def test_every_candidate_owner_declined_rather_than_none_being_consulted(rows) -> None:
    """ "No owner claimed it" and "no owner was asked" are different facts."""

    nineteen = rows["fresh19"][_AUDIT_STEP["fresh19"]]

    assert nineteen.upper_trace, "the selector recorded no consultation at all"
    assert {outcome for _kind, _matches, outcome in nineteen.upper_trace} == {
        "contract_declined"
    }
    assert any(
        kind == "missingness_audit" for kind, _matches, _outcome in nineteen.upper_trace
    )


def test_both_bounds_are_reported_never_only_the_flattering_one() -> None:
    """A recorded plan cannot say whether a receipt was owed at run time.

    Reporting only the no-receipt count would answer in the permissive
    direction for a fact the corpus does not carry.
    """

    ledger = OwnershipLedger()
    for _label, (plan, raw_steps, _granted) in _plans().items():
        ledger.steps.extend(measure_plan(plan, raw_steps, ledger=ledger))
        ledger.readable_plans += 1
    payload = ledger.as_dict()

    assert "claimed_upper" in payload and "claimed_lower" in payload
    assert payload["claimed_lower"] <= payload["claimed_upper"]
    assert "unknown" in payload["bounds"]
    for row in payload["steps"]:
        assert "upper_owner" in row and "lower_owner" in row


def test_an_unreadable_plan_is_counted_not_dropped() -> None:
    """A plan this build cannot parse is missing from the denominator."""

    payload = OwnershipLedger(readable_plans=3, unreadable_plans=2).as_dict()

    assert payload["corpus"]["unreadable_plans"] == 2
    assert payload["corpus"]["readable_plans"] == 3


def test_a_run_registered_concept_does_not_silently_drop_the_plan() -> None:
    """The bias that made the first corpus flatter than the truth.

    Cohort predicates validate ``concept_id`` against the dictionary plus a
    registry the pipeline fills *during* a run.  Re-read afterwards, fresh19's
    plan names ``icu_readmission`` in a robustness cohort override and fails --
    as did 100 of the 110 plans the first sweep dropped, concentrated in the
    robustness family.  Silently dropping them measured an easier corpus.
    """

    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    nineteen = next(e for e in document["plans"] if e["label"] == "fresh19")["plan"]

    assert "icu_readmission" in referenced_concept_ids(nineteen)

    plan, granted = load_plan(nineteen)

    assert plan is not None
    assert granted is True, "the grant must be reported, not assumed away"


def test_the_grant_is_scoped_and_does_not_leak_into_later_validation() -> None:
    """A hypothetical asked here must not become a real answer elsewhere.

    The probe id is unique to this test on purpose.  Asking about a concept
    the other tests also grant would pass whenever one of them leaked first --
    the test would be reporting test order, not scope.
    """

    from easyicu.research_agent.planning.cohort_contract import concept_id_exists

    probe = "ledger_scope_probe_concept"
    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    nineteen = next(e for e in document["plans"] if e["label"] == "fresh19")["plan"]
    renamed = json.loads(
        json.dumps(nineteen).replace('"icu_readmission"', f'"{probe}"')
    )
    assert probe in referenced_concept_ids(renamed)
    assert not concept_id_exists(probe)

    plan, granted = load_plan(renamed)

    assert plan is not None and granted is True
    assert not concept_id_exists(probe), "the grant outlived its own question"


def test_unowned_products_reports_what_no_owner_can_emit(rows) -> None:
    """The list that says where to add capability next."""

    counter = unowned_products(list(rows["fresh19"].values()))

    assert counter["table:adjusted_association_estimates"] == 1
    assert counter["table:event_timing_audit"] == 1
    # Claimed steps contribute nothing: their products are already emittable.
    assert "table:table_one" not in counter


def test_the_ledger_asks_the_selector_rather_than_the_predicates() -> None:
    """A second copy of the ownership predicates would drift from the selector.

    ``select_standard_executor`` applies gates *after* a contract matches -- a
    receipt an owner cannot discharge, a typed input scope it does not support
    -- so a tool that called ``*_owns_step`` directly would eventually report
    an owner the selector declined.
    """

    tree = ast.parse(_TOOL.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert "select_standard_executor" in imported
    assert not [name for name in imported if name.endswith("_owns_step")]


def test_the_fixture_records_where_it_came_from() -> None:
    """Recorded Planner output, not a hand-written plan."""

    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))

    # Not a count: real plans get appended here whenever a run teaches
    # something, and a literal would make every such addition look like a
    # regression. What must hold of every entry is that it came from a run.
    assert len(document["plans"]) >= 2
    assert len({entry["label"] for entry in document["plans"]}) == len(
        document["plans"]
    )
    for entry in document["plans"]:
        assert entry["run_id"].startswith("run_2026")
        assert len(entry["source_sha256"]) == 64
        assert "canonical9_runs" in entry["source_path"]
