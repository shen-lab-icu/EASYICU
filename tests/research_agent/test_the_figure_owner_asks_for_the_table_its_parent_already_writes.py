"""A renderer that can draw the figure must not lose it over one input string.

The missingness/measurement renderer builds a panel from each of two audit
tables, so it declines any step that names only one.  That decline was a bare
boolean: the selector recorded "contract declined" and the step fell through to
the Coder, which is how m1's ``09_missingness_audit_figure`` ended up with a
hand-written source-data table whose columns -- ``count``, ``denominator``,
``percentage``, ``statistic`` -- could not be traced back to any upstream
vector, and the step died ``contract_failed``.  Its parent had written BOTH
tables to disk.

MEASURED over 182 recorded plans, 50 figure steps name at least one audit
table:

* 16 name both; the renderer claims them and nothing here fires.
* 9 name one whose producing step ALSO produces the other.  One string on the
  figure step is the entire gap, the parent and its digest do not change, and
  those are what this reports.
* 31 name one whose sibling NO step in the plan produces.  Closing those means
  asking a parent for a different analysis -- a scientific choice this owner
  does not get to make -- so the verdict stays silent, the same boundary the
  distribution owner had to be narrowed to after canary33.

The renderer itself is NOT loosened: it still refuses to draw one panel from
one table.  What changed is that its refusal now names the field the Planner
can fill in, which is the difference between a replan and a dead figure.
"""

from __future__ import annotations

import collections
import json
import pathlib

import pytest

from easyicu.research_agent.execution.runners.missingness_measurement_figure_executor import (  # noqa: E501
    MEASUREMENT_PROCESS_AUDIT_INPUT,
    MISSINGNESS_MEASUREMENT_AUDIT_INPUT,
    MISSINGNESS_MEASUREMENT_FIGURE_INPUTS,
    missingness_measurement_figure_declaration_verdict,
    missingness_measurement_figure_executor_owns_step,
)
from easyicu.research_agent.execution.owner_declaration import (
    owner_declaration_plan_findings,
)
from easyicu.research_agent.schema import AnalysisPlan

_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


def _audit_step(*outputs: str) -> dict:
    return {
        "step_id": "03_missingness_and_measurement_audit",
        "intent": "Audit availability and measurement process.",
        "method": "measurement_missingness_audit",
        "planned_analysis_role": "auxiliary",
        "inputs": ["artifact:analysis_cohort", "bili_max"],
        "expected_outputs": list(outputs),
    }


def _figure_step(*inputs: str) -> dict:
    return {
        "step_id": "09_missingness_audit_figure",
        "intent": "Draw the availability and measurement-process panels.",
        "method": "visualization",
        "planned_analysis_role": "auxiliary",
        "inputs": list(inputs),
        "expected_outputs": ["figure:missingness_audit"],
    }


def _plan(*steps: dict) -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "research_question": "How complete are the audited variables?",
            "steps": [json.loads(json.dumps(step)) for step in steps],
            "rationale": "missingness figure declaration regression",
        }
    )


def _verdict(plan: AnalysisPlan, step_id: str = "09_missingness_audit_figure"):
    step = next(s for s in plan.steps if s.step_id == step_id)
    return missingness_measurement_figure_declaration_verdict(step, plan=plan)


# ---------------------------------------------------------------------------
# The case this exists for
# ---------------------------------------------------------------------------


def test_the_missing_input_is_named_when_the_parent_already_writes_it():
    plan = _plan(
        _audit_step(*MISSINGNESS_MEASUREMENT_FIGURE_INPUTS),
        _figure_step(MISSINGNESS_MEASUREMENT_AUDIT_INPUT),
    )

    verdict = _verdict(plan)

    assert verdict.declaration_is_incomplete
    assert verdict.missing_declarations == (MEASUREMENT_PROCESS_AUDIT_INPUT,)


def test_the_other_direction_is_reported_too():
    """Either table may be the one left out; neither is privileged."""

    plan = _plan(
        _audit_step(*MISSINGNESS_MEASUREMENT_FIGURE_INPUTS),
        _figure_step(MEASUREMENT_PROCESS_AUDIT_INPUT),
    )

    verdict = _verdict(plan)

    assert verdict.missing_declarations == (MISSINGNESS_MEASUREMENT_AUDIT_INPUT,)


def test_the_reason_names_the_step_that_already_produces_both():
    """The Planner has to know where the table comes from to declare it."""

    plan = _plan(
        _audit_step(*MISSINGNESS_MEASUREMENT_FIGURE_INPUTS),
        _figure_step(MISSINGNESS_MEASUREMENT_AUDIT_INPUT),
    )

    reason = _verdict(plan).reason

    assert "03_missingness_and_measurement_audit" in reason
    assert MEASUREMENT_PROCESS_AUDIT_INPUT in reason


# ---------------------------------------------------------------------------
# The boundary: three shapes that must stay silent
# ---------------------------------------------------------------------------


def test_a_step_naming_both_tables_reports_nothing():
    plan = _plan(
        _audit_step(*MISSINGNESS_MEASUREMENT_FIGURE_INPUTS),
        _figure_step(*MISSINGNESS_MEASUREMENT_FIGURE_INPUTS),
    )

    assert _verdict(plan).missing_declarations == ()


def test_a_sibling_no_step_produces_is_not_demanded():
    """The 25-case boundary.

    Adding an input naming an artifact nobody writes does not help; demanding
    the parent produce it is demanding a different analysis. This is the
    canary33 lesson, and dropping the clause is what a mutation removes.
    """

    plan = _plan(
        _audit_step(MISSINGNESS_MEASUREMENT_AUDIT_INPUT),
        _figure_step(MISSINGNESS_MEASUREMENT_AUDIT_INPUT),
    )

    verdict = _verdict(plan)

    assert verdict.missing_declarations == ()
    assert "no step in this plan produces" in verdict.reason


def test_two_separate_audit_steps_are_still_reportable():
    """CORRECTED 2026-08-04. The first version asserted the opposite.

    It read the renderer's docstring -- "the two digest-bound audit tables
    declared by one direct parent" -- and required one producer. No code
    requires that: ``owns_step`` asks only that both keys resolve to bindings
    carrying the columns it reads, and each binding is digest-pinned to its own
    producer. The narrow rule was mine, and it cost a real case at once:
    e2/verify20 planned the two audits as separate steps 04 and 05, its figure
    named one, this stayed silent, and the step fell to the Coder and died on
    source-data traceability.
    """

    other = dict(_audit_step(MEASUREMENT_PROCESS_AUDIT_INPUT))
    other["step_id"] = "05_measurement_process_audit"
    plan = _plan(
        _audit_step(MISSINGNESS_MEASUREMENT_AUDIT_INPUT),
        other,
        _figure_step(MISSINGNESS_MEASUREMENT_AUDIT_INPUT),
    )

    verdict = _verdict(plan)

    assert verdict.missing_declarations == (MEASUREMENT_PROCESS_AUDIT_INPUT,)
    # The reason must name the step that really produces it, not the other one.
    assert "05_measurement_process_audit" in verdict.reason


def test_a_step_this_owner_could_never_draw_is_not_asked():
    """A modelling step that happens to read an audit table is someone else's."""

    modelling = _figure_step(MISSINGNESS_MEASUREMENT_AUDIT_INPUT)
    modelling["method"] = "adjusted_association_models"
    modelling["expected_outputs"] = ["table:adjusted_association_estimates"]
    plan = _plan(_audit_step(*MISSINGNESS_MEASUREMENT_FIGURE_INPUTS), modelling)

    verdict = _verdict(plan)

    assert verdict.missing_declarations == ()
    assert "auxiliary visualization" in verdict.reason


def test_a_step_naming_neither_table_is_not_asked():
    plan = _plan(
        _audit_step(*MISSINGNESS_MEASUREMENT_FIGURE_INPUTS),
        _figure_step("artifact:analysis_cohort"),
    )

    assert _verdict(plan).missing_declarations == ()


def test_a_composite_publication_figure_is_not_claimed_as_an_audit_figure():
    """One audit panel does not turn a multi-source figure into this owner."""

    composite = _figure_step(
        "table:cohort_flow",
        MISSINGNESS_MEASUREMENT_AUDIT_INPUT,
        "table:adjusted_association_estimates",
        "table:robustness_summary",
    )
    composite["step_id"] = "primary_figure_suite"
    composite["expected_outputs"] = ["figure:primary_figure_suite"]
    plan = _plan(_audit_step(*MISSINGNESS_MEASUREMENT_FIGURE_INPUTS), composite)

    verdict = _verdict(plan, step_id="primary_figure_suite")

    assert verdict.missing_declarations == ()
    assert "sole input" in verdict.reason


# ---------------------------------------------------------------------------
# The renderer is not loosened, and the gate carries the verdict
# ---------------------------------------------------------------------------


def test_the_renderer_still_refuses_to_draw_one_panel_from_one_table():
    """The fix is in the decline, never in what the figure is allowed to be.

    A renderer that claimed a one-table step would draw a figure missing the
    panel the plan asked for, which is worse than the Coder drawing it.
    """

    plan = _plan(
        _audit_step(*MISSINGNESS_MEASUREMENT_FIGURE_INPUTS),
        _figure_step(MISSINGNESS_MEASUREMENT_AUDIT_INPUT),
    )
    step = next(s for s in plan.steps if s.step_id == "09_missingness_audit_figure")

    one_binding = {
        MISSINGNESS_MEASUREMENT_AUDIT_INPUT: {
            "columns": [
                "variable",
                "n_total",
                "eligible_n",
                "not_applicable_n",
                "measured_one_n",
                "value_missing_n",
                "value_missing_pct",
            ]
        }
    }

    assert not missingness_measurement_figure_executor_owns_step(
        step, resolved_bindings=one_binding
    )


def test_the_gate_turns_the_verdict_into_a_repairable_finding():
    """End-to-end through the real gate, not a re-derivation of the predicate."""

    plan = _plan(
        _audit_step(*MISSINGNESS_MEASUREMENT_FIGURE_INPUTS),
        _figure_step(MISSINGNESS_MEASUREMENT_AUDIT_INPUT),
    )

    findings = [
        finding
        for finding in owner_declaration_plan_findings(plan=plan)
        if finding.detail.get("step_id") == "09_missingness_audit_figure"
    ]

    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity == "error"
    assert finding.detail["reason"] == "owner_declaration_incomplete"
    assert finding.detail["analysis_kind"] == "missingness_measurement_figure"
    assert finding.detail["missing_declarations"] == [MEASUREMENT_PROCESS_AUDIT_INPUT]
    # The directive must not tell the Planner to change the science instead.
    assert "do not split or add a step" in finding.message
    assert "exposure, outcome, cohort" in finding.message


# ---------------------------------------------------------------------------
# Reachability, re-measured on the recorded corpus
# ---------------------------------------------------------------------------


def test_the_corpus_fires_only_where_the_plan_can_close_the_gap():
    """Re-derives the 9/16/25 split rather than restating it.

    A verdict that fired on the 25 would demand a different analysis; one that
    fired on none would be coverage that never runs.
    """

    if not _CORPUS.exists():
        pytest.skip("recorded run corpus is not mounted")

    fires: collections.Counter = collections.Counter()
    silent: collections.Counter = collections.Counter()
    for path in _CORPUS.glob("batch_*/*/aware/run_*/analysis_plan.json"):
        try:
            plan = AnalysisPlan.model_validate(json.loads(path.read_text()))
        except Exception:  # noqa: BLE001 - a malformed plan is not this subject
            continue
        producer = {}
        for candidate in plan.steps:
            for output in candidate.expected_outputs or []:
                producer.setdefault(str(output).strip(), candidate)
        for step in plan.steps:
            declared = {str(value).strip() for value in step.inputs or []}
            named = [k for k in MISSINGNESS_MEASUREMENT_FIGURE_INPUTS if k in declared]
            if not named:
                continue
            if len(named) == 2:
                shape = "both"
            else:
                absent = next(
                    k for k in MISSINGNESS_MEASUREMENT_FIGURE_INPUTS if k != named[0]
                )
                # The only question that matters: does ANY step produce the
                # sibling? If not, the plan cannot close the gap here.
                shape = (
                    "one_producible"
                    if producer.get(absent) is not None
                    else "one_no_sibling"
                )
            verdict = missingness_measurement_figure_declaration_verdict(
                step, plan=plan
            )
            (fires if verdict.missing_declarations else silent)[shape] += 1

    if not (fires or silent):
        pytest.skip("no recorded plan names an audit table on a figure step")
    assert fires["one_producible"] > 0, "the verdict never fires on real input"
    assert fires["both"] == 0, fires
    # The load-bearing half: a sibling nobody writes is never demanded.
    assert fires["one_no_sibling"] == 0, fires
    assert silent["both"] > 0 and silent["one_no_sibling"] > 0, silent


def test_the_recorded_m1_step_is_the_one_that_fires():
    """The blocker this was written for, read from its own run."""

    path = (
        _CORPUS
        / "batch_20260804_luna_miiv_FULL_e765745_verify15"
        / "m1_hepatobiliary_missingness"
        / "aware"
    )
    if not path.exists():
        pytest.skip("the verify15 run is not mounted")

    plans = sorted(path.glob("run_*/analysis_plan.json"))
    if not plans:
        pytest.skip("the recorded run carries no plan")
    plan = AnalysisPlan.model_validate(json.loads(plans[0].read_text()))

    verdict = _verdict(plan)

    assert verdict.missing_declarations == (MEASUREMENT_PROCESS_AUDIT_INPUT,)
