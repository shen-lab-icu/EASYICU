"""Plan-phase display-contract guards: a result-bearing plan must declare a
publication figure and an audit/robustness panel.

E1 run13 scored plan=0.6: deepseek's plan declared a table-one but no figure
step (its question never says "figure", so the question-gated guard skipped it)
and no audit panel — even though the publication-figure skill produces a figure
and the run produces robustness/data-quality evidence. The scorer reads
``analysis_plan.json``, so the fix declares both at plan phase:
``_ensure_publication_figure_step_in_plan(force=...)`` and
``_ensure_audit_panel_step_in_plan``.
"""

from __future__ import annotations

from easyicu.research_agent.evaluation_scorecard import score_plan
from easyicu.research_agent.icu_agent_bench import ICUAgentBenchTask
from easyicu.research_agent.plan_utils import (
    _ensure_audit_panel_step_in_plan,
    _ensure_publication_figure_step_in_plan,
    _step_declares_audit_panel,
    _step_produces_figure,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _table_only_plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question=(
            "Among adult ICU patients, what is the prevalence of Sepsis-3 and "
            "is it associated with in-hospital mortality after adjustment?"
        ),  # deliberately never says "figure" / "plot"
        steps=[
            AnalysisStep(
                step_id="01_table_one",
                intent="Summarise baseline characteristics.",
                expected_outputs=["table:table_one"],
            ),
            AnalysisStep(
                step_id="02_primary_model",
                intent="Fit the adjusted logistic regression.",
                expected_outputs=["table:regression_results"],
            ),
        ],
    )


class _Ctx:
    def __init__(self, question: str):
        self.research_question = question


def test_figure_guard_needs_force_when_question_is_silent():
    plan = _table_only_plan()
    ctx = _Ctx(plan.research_question)

    # Default (question-gated): the question never says "figure", so no step.
    unforced, findings = _ensure_publication_figure_step_in_plan(
        plan=plan, context=ctx, force=False
    )
    assert not any(_step_produces_figure(s) for s in unforced.steps)
    assert findings == []

    # Forced (a figure WILL be produced): declare it.
    forced, findings = _ensure_publication_figure_step_in_plan(
        plan=plan, context=ctx, force=True
    )
    assert any(_step_produces_figure(s) for s in forced.steps)
    assert findings and findings[0].validator == "plan_contract"


def test_audit_panel_guard_appends_then_idempotent():
    plan = _table_only_plan()
    ctx = _Ctx(plan.research_question)

    with_audit, findings = _ensure_audit_panel_step_in_plan(plan=plan, context=ctx)
    assert any(_step_declares_audit_panel(s) for s in with_audit.steps)
    assert findings and findings[0].validator == "plan_contract"

    # Idempotent: a plan that already declares an audit panel is left alone.
    again, findings2 = _ensure_audit_panel_step_in_plan(plan=with_audit, context=ctx)
    assert len(again.steps) == len(with_audit.steps)
    assert findings2 == []


def test_plan_dimension_lifts_after_both_guards():
    task = ICUAgentBenchTask(
        task_id="t1",
        kind="descriptive_association",
        title="Sepsis-3 mortality",
        objective="Prevalence and adjusted association with mortality.",
        difficulty="basic",  # non-advanced -> needs >= 1 result figure
    )
    gates = {"required_step_count": 2}

    before = score_plan(
        task,
        plan_steps=[s.model_dump() for s in _table_only_plan().steps],
        gates=gates,
    )
    assert before.signals["result_figure_count"] == 0
    assert before.signals["has_audit_panel"] is False

    plan = _table_only_plan()
    ctx = _Ctx(plan.research_question)
    plan, _ = _ensure_publication_figure_step_in_plan(plan=plan, context=ctx, force=True)
    plan, _ = _ensure_audit_panel_step_in_plan(plan=plan, context=ctx)

    after = score_plan(
        task,
        plan_steps=[s.model_dump() for s in plan.steps],
        gates=gates,
    )
    assert after.signals["result_figure_count"] >= 1
    assert after.signals["has_audit_panel"] is True
    assert after.subscore > before.subscore
