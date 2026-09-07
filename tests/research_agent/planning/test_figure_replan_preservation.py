"""Figure replan preservation keeps exact producer, product and replay contracts."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_preserve_figure_steps_after_replan_re_attaches_dropped_figure_step(ra):
    """Regression: Replanner must not silently drop figure-producing steps.

    qwen3-coder-30b under naive arms (no ICU context) often returns a
    revised plan that rationalises away the figure step after the probe
    summary. Task contracts still require the figure artefact, so the
    pipeline must re-attach any dropped step whose ``expected_outputs``
    declare a figure/plot output.
    """
    from easyicu.research_agent.planning.figure_step_contract import (
        _preserve_figure_steps_after_replan,
        _step_produces_figure,
    )

    fig_step = ra.AnalysisStep(
        step_id="02_summary_figure",
        intent="Render publication-ready figure for the table-one summary.",
        expected_outputs=["figure:table_one_summary"],
    )
    table_step = ra.AnalysisStep(
        step_id="01_table_one",
        intent="Build descriptive Table 1.",
        expected_outputs=["table:table_one"],
    )
    current = ra.AnalysisPlan(
        research_question="describe the cohort",
        steps=[table_step, fig_step],
    )
    revised = ra.AnalysisPlan(
        research_question="describe the cohort",
        steps=[table_step],
        revision=2,
    )

    assert _step_produces_figure(fig_step) is True
    assert _step_produces_figure(table_step) is False

    preserved, findings = _preserve_figure_steps_after_replan(
        current=current,
        revised=revised,
    )

    preserved_ids = [s.step_id for s in preserved.steps]
    assert "02_summary_figure" in preserved_ids, (
        "dropped figure step must be re-attached to revised plan; "
        f"got steps={preserved_ids}"
    )
    assert any(
        f.severity == "warning" and "figure-producing" in f.message for f in findings
    )


def test_preserve_figure_steps_after_replan_no_op_when_figure_kept(ra):
    """No-op when the replanner kept all figure steps."""
    from easyicu.research_agent.planning.figure_step_contract import _preserve_figure_steps_after_replan

    fig_step = ra.AnalysisStep(
        step_id="02_summary_figure",
        intent="Render summary figure.",
        expected_outputs=["figure:table_one_summary"],
    )
    table_step = ra.AnalysisStep(
        step_id="01_table_one",
        intent="Build Table 1.",
        expected_outputs=["table:table_one"],
    )
    current = ra.AnalysisPlan(
        research_question="describe the cohort",
        steps=[table_step, fig_step],
    )
    revised = ra.AnalysisPlan(
        research_question="describe the cohort",
        steps=[table_step, fig_step],
        revision=2,
    )

    preserved, findings = _preserve_figure_steps_after_replan(
        current=current,
        revised=revised,
    )

    assert findings == []
    assert [s.step_id for s in preserved.steps] == [
        "01_table_one",
        "02_summary_figure",
    ]


def test_preserve_figure_steps_after_replan_restores_exact_parent_products(ra):
    """An echoed pre-split parent must not strand the preserved render child."""
    from easyicu.research_agent.planning.figure_step_contract import (
        _preserve_figure_steps_after_replan,
    )

    current_parent = ra.AnalysisStep(
        step_id="01_model_training",
        intent="Fit the agent-selected prediction model.",
        method="prediction_model",
        expected_outputs=[
            "statistic:auroc",
            "table:model_performance",
            "table:roc_curve",
        ],
    )
    current_figure = ra.AnalysisStep(
        step_id="01_model_training_figure",
        intent=("Render the publication figure declared by step '01_model_training'."),
        method="visualization",
        inputs=["table:model_performance", "table:roc_curve"],
        expected_outputs=["figure:discrimination_calibration"],
    )
    current = ra.AnalysisPlan(
        research_question="build a prediction model",
        steps=[current_parent, current_figure],
    )
    # The replanner echoes the original parent shape and drops the host-split
    # child. It did not choose a different method or producer.
    revised = ra.AnalysisPlan(
        research_question="build a prediction model",
        steps=[
            current_parent.model_copy(update={"expected_outputs": ["statistic:auroc"]})
        ],
        revision=2,
    )

    preserved, findings = _preserve_figure_steps_after_replan(
        current=current,
        revised=revised,
    )

    by_id = {step.step_id: step for step in preserved.steps}
    assert by_id["01_model_training"].expected_outputs == [
        "statistic:auroc",
        "table:model_performance",
        "table:roc_curve",
    ]
    assert "01_model_training_figure" in by_id
    assert any(
        (finding.detail or {}).get("reason")
        == "preserved_figure_parent_output_contract"
        for finding in findings
    )


def test_preserved_robustness_parent_outputs_update_the_owner_spec(ra):
    """Restoring a render edge must keep the deterministic owner in sync."""
    from easyicu.research_agent.planning.figure_step_contract import _preserve_figure_steps_after_replan
    from easyicu.research_agent.schema import RobustnessReplaySpec

    base_spec = RobustnessReplaySpec.model_validate(
        {
            "products": [
                {
                    "product_id": "robustness_matrix",
                    "output": "robustness_matrix",
                },
                {
                    "product_id": "robustness_summary",
                    "output": "robustness_summary",
                },
            ]
        }
    )
    full_spec = RobustnessReplaySpec.model_validate(
        {
            "products": [
                *base_spec.model_dump(mode="python")["products"],
                {"product_id": "primary_or", "output": "primary_effect"},
                {
                    "product_id": "complete_case_n",
                    "output": "complete_case_n",
                },
            ]
        }
    )
    current_parent = ra.AnalysisStep(
        step_id="05_robustness",
        planned_analysis_role="sensitivity",
        intent="Replay the locked robustness grid.",
        method="robustness_sensitivity",
        expected_outputs=[
            "table:robustness_matrix",
            "table:robustness_summary",
            "statistic:primary_or",
            "statistic:complete_case_n",
        ],
        robustness_replay_spec=full_spec,
    )
    current_figure = ra.AnalysisStep(
        step_id="05_robustness_figure",
        intent="Render the publication figure(s) declared by step '05_robustness'.",
        method="visualization",
        inputs=["statistic:primary_or", "statistic:complete_case_n"],
        expected_outputs=["figure:robustness_forest"],
    )
    current = ra.AnalysisPlan(
        research_question="Audit robustness.",
        steps=[current_parent, current_figure],
    )
    revised = ra.AnalysisPlan(
        research_question="Audit robustness.",
        steps=[
            current_parent.model_copy(
                update={
                    "expected_outputs": [
                        "table:robustness_matrix",
                        "table:robustness_summary",
                    ],
                    "robustness_replay_spec": base_spec,
                }
            )
        ],
        revision=2,
    )

    preserved, _findings = _preserve_figure_steps_after_replan(
        current=current,
        revised=revised,
    )

    parent = next(step for step in preserved.steps if step.step_id == "05_robustness")
    assert parent.robustness_replay_spec is not None
    mapped = {
        item.product_id: item.output for item in parent.robustness_replay_spec.products
    }
    assert mapped["primary_or"] == "primary_effect"
    assert mapped["complete_case_n"] == "complete_case_n"


def test_preserve_figure_steps_after_replan_does_not_invent_missing_parent(ra):
    """A dropped producer remains a typed-DAG error; preservation cannot guess."""
    from easyicu.research_agent.planning.figure_step_contract import (
        _preserve_figure_steps_after_replan,
    )

    current_parent = ra.AnalysisStep(
        step_id="01_model_training",
        intent="Fit the agent-selected prediction model.",
        expected_outputs=["table:model_performance"],
    )
    current_figure = ra.AnalysisStep(
        step_id="01_model_training_figure",
        intent=("Render the publication figure declared by step '01_model_training'."),
        method="visualization",
        inputs=["table:model_performance"],
        expected_outputs=["figure:discrimination_calibration"],
    )
    current = ra.AnalysisPlan(
        research_question="build a prediction model",
        steps=[current_parent, current_figure],
    )
    revised = ra.AnalysisPlan(
        research_question="build a prediction model",
        steps=[
            ra.AnalysisStep(
                step_id="02_other",
                intent="Retain an unrelated descriptive step.",
                expected_outputs=[],
            )
        ],
        revision=2,
    )

    preserved, findings = _preserve_figure_steps_after_replan(
        current=current,
        revised=revised,
    )

    assert [step.step_id for step in preserved.steps] == [
        "02_other",
        "01_model_training_figure",
    ]
    assert all(
        (finding.detail or {}).get("reason")
        != "preserved_figure_parent_output_contract"
        for finding in findings
    )
