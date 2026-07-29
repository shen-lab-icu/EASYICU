"""An unresolvable raw input must block at plan time, not crash mid-run.

A step's typed ``kind:product`` inputs are validated before execution; its raw
column inputs were not validated at all. They were first read deep inside
``_execute_one_step``, which raises ``ValueError`` when a name has no context
descriptor -- and ``run_sequential`` does not wrap ``execute_step``::

    record = execute_step(step)        # run_coordination.py:72, no try/except

so that exception leaves the coordinator, the execute phase and
``pipeline.run``, killing the run with no sealed artifacts and no diagnosis.
fresh16 died exactly this way (``BENCH_EXIT=5``) after ~14 minutes of real
provider spend.

Sweeping 1,114 historical plan steps found 8 that would hard crash this way,
all one class: the Planner declaring a column the sealed context does not
carry. This gate asks the same question at plan time, where the existing
preflight already turns a finding into one focused replan directive.
"""

from __future__ import annotations

import ast
import pathlib

from easyicu.research_agent.gates.plan_declared_inputs import (
    declared_raw_input_plan_findings,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
)

_PHASE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src/easyicu/research_agent/execution/phase.py"
)


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
            for name in ("lact_max", "death", "death_time")
        ],
    )


def _plan(*steps: AnalysisStep) -> AnalysisPlan:
    return AnalysisPlan(
        research_question="test",
        steps=list(steps),
    )


def test_a_column_the_context_does_not_carry_is_reported() -> None:
    """The real historical class: the Planner invents a column name."""

    plan = _plan(
        AnalysisStep(
            step_id="01_define_analysis_cohort",
            intent="cohort",
            inputs=["lact_max", "sepsis3"],
        )
    )

    findings = declared_raw_input_plan_findings(plan=plan, context=_context())

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].detail["unresolvable_inputs"] == ["sepsis3"]


def test_the_offending_name_is_in_the_message_not_only_the_detail() -> None:
    """``detail`` never reaches a prompt; only ``message`` does."""

    plan = _plan(AnalysisStep(step_id="s1", intent="i", inputs=["lact_max", "sepsis3"]))

    message = declared_raw_input_plan_findings(plan=plan, context=_context())[0].message

    assert "sepsis3" in message
    # Cause first: the prompt projection clips the message from the tail.
    assert message.index("sepsis3") < 120


def test_a_fully_resolvable_step_is_silent() -> None:
    plan = _plan(AnalysisStep(step_id="s1", intent="i", inputs=["lact_max", "death"]))

    assert declared_raw_input_plan_findings(plan=plan, context=_context()) == []


def test_typed_products_are_not_treated_as_raw_columns() -> None:
    """``kind:name`` inputs stay under the typed DAG validator's authority."""

    plan = _plan(
        AnalysisStep(
            step_id="s1",
            intent="i",
            inputs=["artifact:analysis_cohort", "table:not_a_column", "death"],
        )
    )

    assert declared_raw_input_plan_findings(plan=plan, context=_context()) == []


def test_every_unresolvable_name_in_one_step_is_named() -> None:
    plan = _plan(
        AnalysisStep(step_id="s1", intent="i", inputs=["ghost_a", "death", "ghost_b"])
    )

    detail = declared_raw_input_plan_findings(plan=plan, context=_context())[0].detail

    assert detail["unresolvable_inputs"] == ["ghost_a", "ghost_b"]


def test_each_bad_step_gets_its_own_finding() -> None:
    plan = _plan(
        AnalysisStep(step_id="s1", intent="i", inputs=["ghost"]),
        AnalysisStep(step_id="s2", intent="i", inputs=["death"]),
        AnalysisStep(step_id="s3", intent="i", inputs=["ghost"]),
    )

    findings = declared_raw_input_plan_findings(plan=plan, context=_context())

    assert [f.detail["step_id"] for f in findings] == ["s1", "s3"]


def test_a_step_declaring_no_inputs_is_silent() -> None:
    plan = _plan(AnalysisStep(step_id="s1", intent="i"))

    assert declared_raw_input_plan_findings(plan=plan, context=_context()) == []


# ---------------------------------------------------------------------------
# Reachability: a correct predicate nobody wires in changes nothing.
# ---------------------------------------------------------------------------


def _phase_calls_to(function_name: str) -> int:
    tree = ast.parse(_PHASE.read_text(encoding="utf-8"))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", "")) == function_name
    )


def test_the_execute_phase_actually_calls_this_gate_twice() -> None:
    """Once as repairable preflight, once as the final pre-Coder block.

    The preflight alone would let a plan the replanner failed to repair run
    anyway; the final check alone would waste the one focused replan the
    existing gate already funds.
    """

    assert _phase_calls_to("declared_raw_input_plan_findings") == 2


def test_the_finding_participates_in_the_forced_replan() -> None:
    """It must reach both the directive join and the ``force`` decision."""

    source = _PHASE.read_text(encoding="utf-8")

    assert "declared_input_directive," in source
    assert "or declared_input_preflight" in source
