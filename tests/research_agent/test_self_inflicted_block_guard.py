"""Tests for the self-inflicted-block guard (真修 M2 — A layer).

Two halves are covered:

* the pure, deterministic decision functions in ``viability`` and the
  ``build_self_block_replan_directive`` trigger logic (cheap, no run); and
* that ``ReplannerAgent.run`` surfaces a runtime ``directive`` to the LLM.

Background: a ``mortality_prediction`` run self-paralysed — agent-invented
``viability_gate`` / ``modeling_block_registration`` steps registered
``modeling_blocked=true`` on a 74,829-row, 7,397-event cohort, so the modeling
step emitted a non-execution stub and step-level contract repair re-stubbed it
three times. The guard fires a *forced, directive-carrying* replan when a
model/estimation step self-blocks on a task-viable cohort — conditioned on
viability so a genuinely non-viable cohort still legitimately blocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from easyicu.research_agent.viability import (
    MIN_VIABLE_ROWS,
    assess_cohort_viability,
    step_requires_model_performance,
    step_summary_block_signal,
)
from easyicu.research_agent.execution.phase import build_self_block_replan_directive


# ---------------------------------------------------------------------------
# assess_cohort_viability
# ---------------------------------------------------------------------------


def _frame(n_rows: int, n_pred: int = 6, *, outcome_minority: int | None = None):
    data = {f"p{i}": np.ones(n_rows) for i in range(n_pred)}
    if outcome_minority is not None:
        y = np.zeros(n_rows, dtype=int)
        y[:outcome_minority] = 1
        data["death"] = y
    return pd.DataFrame(data)


def test_viability_true_on_populated_cohort():
    v = assess_cohort_viability(_frame(500), outcome=None)
    assert v.viable is True
    assert v.n_rows == 500
    assert v.well_populated_predictors >= 5
    assert "500 rows" in v.note


def test_viability_false_when_too_few_rows():
    v = assess_cohort_viability(_frame(MIN_VIABLE_ROWS - 1), outcome=None)
    assert v.viable is False
    assert v.note == ""


def test_viability_false_when_too_few_predictors():
    v = assess_cohort_viability(_frame(500, n_pred=4), outcome=None)
    assert v.viable is False


def test_viability_false_when_outcome_has_no_variation():
    df = _frame(500)
    df["death"] = 0  # single class -> not modellable
    v = assess_cohort_viability(df, outcome="death")
    assert v.viable is False


def test_viability_false_when_minority_events_too_few():
    v = assess_cohort_viability(_frame(500, outcome_minority=5), outcome="death")
    assert v.viable is False


def test_viability_true_with_outcome_minority_counted():
    v = assess_cohort_viability(_frame(500, outcome_minority=40), outcome="death")
    assert v.viable is True
    assert v.minority_events == 40
    assert "40 minority-class outcome events" in v.note


def test_viability_excludes_outcome_from_predictor_count():
    # 5 predictors + the outcome column: the outcome must not be counted as a
    # predictor, so this is below the predictor floor and not viable.
    df = _frame(500, n_pred=5, outcome_minority=40)
    v = assess_cohort_viability(df, outcome="death")
    assert v.well_populated_predictors == 5  # the 5 p* cols, not death
    assert v.viable is True
    df4 = _frame(500, n_pred=4, outcome_minority=40)
    assert assess_cohort_viability(df4, outcome="death").viable is False


# ---------------------------------------------------------------------------
# step_requires_model_performance / step_summary_block_signal
# ---------------------------------------------------------------------------


def test_step_requires_model_performance_matches_canonical_contract():
    assert step_requires_model_performance(["table:x", "statistic:auroc"]) is True
    assert step_requires_model_performance(["statistic:brier_score"]) is True
    assert step_requires_model_performance(["STATISTIC:AUROC"]) is True  # case-neutral


def test_step_requires_model_performance_false_for_association_outputs():
    assert step_requires_model_performance(["statistic:primary_or"]) is False
    assert step_requires_model_performance([]) is False


def test_block_signal_detects_blocked_and_modeling_blocked():
    assert (
        step_summary_block_signal({"execution_status": "blocked_non_execution"})
        == "blocked_non_execution"
    )
    assert (
        step_summary_block_signal(
            {"modeling_blocked": True, "modeling_block_reason": "manifest lost"}
        )
        == "manifest lost"
    )


def test_block_signal_silent_on_ok_or_crash():
    assert step_summary_block_signal({"execution_status": "ok"}) is None
    assert step_summary_block_signal({}) is None
    assert step_summary_block_signal({"error": "Traceback ..."}) is None


# ---------------------------------------------------------------------------
# build_self_block_replan_directive (the trigger)
# ---------------------------------------------------------------------------


def _model_step():
    return AnalysisStep(
        step_id="01_model_training",
        intent="Train and validate the mortality prediction model.",
        inputs=[],
        expected_outputs=["statistic:auroc", "statistic:brier_score"],
        method="prediction_model",
    )


def _viable():
    return assess_cohort_viability(_frame(500, outcome_minority=40), outcome="death")


def test_directive_fires_when_model_step_self_blocks_on_viable_cohort():
    directive = build_self_block_replan_directive(
        failed_step=_model_step(),
        failed_record={
            "status": "contract_failed",
            "step_summary": {"execution_status": "blocked_non_execution"},
        },
        completed_records=[],
        viability=_viable(),
    )
    assert directive is not None
    # Conditioned on viability in the text itself (impartiality red line):
    assert "task-viable" in directive
    assert "genuinely non-viable" in directive
    assert "Do NOT re-insert" in directive


def test_directive_fires_via_upstream_block_registration_record():
    # The failed model step's own summary may not carry the signal; an upstream
    # modeling-block-registration step does.
    directive = build_self_block_replan_directive(
        failed_step=_model_step(),
        failed_record={"status": "contract_failed", "step_summary": {}},
        completed_records=[
            {
                "step_id": "01d_block_registration",
                "step_summary": {"modeling_blocked": True},
            }
        ],
        viability=_viable(),
    )
    assert directive is not None


def test_directive_silent_when_cohort_not_viable():
    nonviable = assess_cohort_viability(_frame(MIN_VIABLE_ROWS - 1), outcome=None)
    directive = build_self_block_replan_directive(
        failed_step=_model_step(),
        failed_record={
            "status": "contract_failed",
            "step_summary": {"execution_status": "blocked_non_execution"},
        },
        completed_records=[],
        viability=nonviable,
    )
    assert directive is None  # blocking on non-viable data stays legitimate


def test_directive_silent_for_non_model_step():
    assoc_step = AnalysisStep(
        step_id="02_association",
        intent="Fit the association model.",
        inputs=[],
        expected_outputs=["statistic:primary_or"],
        method="bias_audit_association",
    )
    directive = build_self_block_replan_directive(
        failed_step=assoc_step,
        failed_record={
            "status": "contract_failed",
            "step_summary": {"execution_status": "blocked_non_execution"},
        },
        completed_records=[],
        viability=_viable(),
    )
    assert directive is None


def test_directive_silent_without_block_signal():
    # A hard crash (no deliberate block) must not be reframed as self-paralysis.
    directive = build_self_block_replan_directive(
        failed_step=_model_step(),
        failed_record={"status": "execution_failed", "step_summary": {}},
        completed_records=[
            {"step_id": "01_audit", "step_summary": {"execution_status": "ok"}}
        ],
        viability=_viable(),
    )
    assert directive is None


# ---------------------------------------------------------------------------
# ReplannerAgent surfaces the directive to the LLM
# ---------------------------------------------------------------------------


def test_replanner_injects_runtime_directive_into_prompt(ra):
    from easyicu.research_agent.schema import (
        CohortDescriptor,
        ConceptDescriptor,
        ResearchContext,
        VariableRole,
    )
    from easyicu.research_agent.agents.core import ReplannerAgent

    captured: dict = {}

    class CapturingLLM(ra.MockLLMClient):
        def complete(self, messages, **kwargs):
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            captured["user"] = user
            plan = AnalysisPlan(research_question="q", steps=[_model_step()])
            return plan.model_dump_json(indent=2)

    ctx = ResearchContext(
        research_question="Does first-day SOFA-2 predict ICU mortality?",
        cohort=CohortDescriptor(
            cohort_name="t", database="miiv", n_stays=500, n_patients=500
        ),
        variables=[
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="int64")
        ],
        target_outcome="death",
    )
    plan = AnalysisPlan(research_question="q", steps=[_model_step()])

    ReplannerAgent(CapturingLLM()).run(
        context=ctx,
        current_plan=plan,
        directive="OVERRIDE: cohort is task-viable; fit the model, do not block.",
    )
    assert "PRIORITY RUNTIME DIRECTIVE" in captured["user"]
    assert "do not block" in captured["user"]
    # The directive must precede the routine plan body so it cannot be buried.
    assert captured["user"].index("PRIORITY RUNTIME DIRECTIVE") < captured[
        "user"
    ].index("CURRENT PLAN:")


def test_replanner_without_directive_has_no_directive_block(ra):
    from easyicu.research_agent.schema import (
        CohortDescriptor,
        ConceptDescriptor,
        ResearchContext,
        VariableRole,
    )
    from easyicu.research_agent.agents.core import ReplannerAgent

    captured: dict = {}

    class CapturingLLM(ra.MockLLMClient):
        def complete(self, messages, **kwargs):
            captured["user"] = next(
                (m.content for m in reversed(messages) if m.role == "user"), ""
            )
            return AnalysisPlan(
                research_question="q", steps=[_model_step()]
            ).model_dump_json()

    ctx = ResearchContext(
        research_question="q",
        cohort=CohortDescriptor(
            cohort_name="t", database="miiv", n_stays=500, n_patients=500
        ),
        variables=[
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="int64")
        ],
        target_outcome="death",
    )
    ReplannerAgent(CapturingLLM()).run(
        context=ctx,
        current_plan=AnalysisPlan(research_question="q", steps=[_model_step()]),
    )
    assert "PRIORITY RUNTIME DIRECTIVE" not in captured["user"]


# ---------------------------------------------------------------------------
# End-to-end: the guard fires through the real run_execute_phase loop
# ---------------------------------------------------------------------------
#
# The pure-function + replanner-injection tests above prove the *trigger* and
# that the replanner *accepts* a directive. They do NOT prove the GLUE: that the
# execute loop, on a model step that self-blocks on a viable cohort, actually
# calls the guard, forces the directive-carrying replan, and re-drives the step.
# Two live M2 reruns (2026-06-12) did NOT reproduce the self-block (gpt5.4 emits
# a clean plan), so live reproduction is an unreliable validator — this
# deterministic test is the mechanism proof. The stub never recovers (a real LLM
# obeying the directive is what recovers in production), so the run lands in the
# honest diagnostic_only fallback after the bounded directed-replan budget.

_SELF_BLOCK_STUB = (
    "import json, os\n"
    "from pathlib import Path\n"
    "out = Path(os.environ['STEP_OUT_DIR']); out.mkdir(parents=True, exist_ok=True)\n"
    "summary = {'step_id': '01_model_training', "
    "'execution_status': 'blocked_non_execution', 'modeling_blocked': True, "
    "'modeling_block_reason': 'upstream viability gate reported artifacts unusable', "
    "'model_executed': False}\n"
    "json.dump(summary, open(out / 'step_summary.json', 'w'))\n"
    "print(json.dumps(summary))\n"
)


def test_directed_replan_fires_through_execute_loop(ra, synthetic_cohort, tmp_path):
    model_step = AnalysisStep(
        step_id="01_model_training",
        intent="Train and validate the mortality prediction model in one step.",
        inputs=[],
        expected_outputs=[
            "statistic:auroc",
            "statistic:brier_score",
            "table:model_performance_train_test",
        ],
        method="prediction_model",
    )
    fixed_plan = AnalysisPlan(
        research_question="Build an in-hospital mortality prediction model.",
        steps=[model_step],
    )

    class SelfBlockLLM(ra.MockLLMClient):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.replan_prompts: list[str] = []

        def complete(self, messages, **kwargs):
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            up = user.upper()
            is_coder = "PYTHON CODE FOR STEP" in up or (
                "PYTHON CODE" in up and ("WRITE" in up or "REPAIR" in up)
            )
            is_replan = "CURRENT PLAN:" in up and (
                "PROBE SUMMARY" in up or "REVISE" in up or "DIRECTIVE" in up
            )
            is_plan = (not is_replan) and (
                "ANALYSISPLAN SCHEMA" in up
                or "RESEARCH PLAN AS JSON" in up
                or "ICU-AWARE RESEARCH PLAN" in up
            )
            if is_replan:
                self.replan_prompts.append(user)
                # Echo the same plan -> a noop revision; the guard still re-drives
                # the model step because _maybe_replan returns the current plan.
                return fixed_plan.model_dump_json(indent=2)
            if is_plan:
                return fixed_plan.model_dump_json(indent=2)
            if is_coder and "01_model_training" in user:
                return _SELF_BLOCK_STUB  # self-block on every model-step attempt
            return super().complete(messages, **kwargs)

    llm = SelfBlockLLM()
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=llm)
    result = pipeline.run(
        question="Build an in-hospital mortality prediction model.",
        cohort=synthetic_cohort,
        cohort_name="m2_selfblock",
        database="synthetic",
        target_outcome="death",
    )

    import json as _json

    manifest = _json.loads(__import__("pathlib").Path(result.manifest_path).read_text())
    findings = manifest["findings"]

    directed = [
        f
        for f in findings
        if f.get("validator") == "replanner"
        and "self-blocked on a task-viable cohort" in (f.get("message") or "")
    ]
    # Mechanism: the loop detected the self-block on the viable cohort and fired
    # the directed replan — at least once, and bounded by the budget.
    assert directed, (
        "directed-replan guard did not fire through the execute loop on a "
        "self-blocking model step over a viable cohort"
    )
    assert len(directed) <= 2, f"directed replans exceeded the budget: {len(directed)}"

    # Wiring: the viability-conditioned directive actually reached the replanner
    # prompt (front-placed, with the impartiality red-line text) — so a relapse
    # would be the replanner ignoring the directive, not a wiring failure.
    assert any(
        "PRIORITY RUNTIME DIRECTIVE" in p
        and "task-viable" in p
        and "genuinely non-viable" in p
        for p in llm.replan_prompts
    ), "the directive never reached the replanner prompt (wiring gap)"
