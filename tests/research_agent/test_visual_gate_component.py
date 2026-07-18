"""Behavior tests for the typed Visual-QA gate component (batch 1a-2).

These lock the PURE, side-effect-free ``VisualGateResult`` /
``VisualRepairDecision`` seam extracted from the ``_execute_one_step`` VisualQA
block. The component collects the figure audit + classifies it and recommends a
repair; it must carry NO control flow — ``continue`` / ``return`` / step status /
budget / locks / evidence registration stay in the orchestration layer.

Written before the implementation (Codex constraint: behavior tests + state
table first). The state table these mirror lives in
``task_logs/20260718_batch1a2_visual_gate_typed_report.md`` §2.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.schema import ValidationFinding

# --- shared synthetic findings (reused from test_pipeline_execute_contract) ---


def _cosmetic_error() -> ValidationFinding:
    return ValidationFinding(
        validator="visual_qa",
        severity="error",
        message=(
            "SVG figure 'x.svg' has overlapping text elements; "
            "multi-panel labels, annotations or axis text need more spacing."
        ),
    )


def _hard_error() -> ValidationFinding:
    return ValidationFinding(
        validator="visual_qa",
        severity="error",
        message="Could not open figure 'x.png': truncated image file",
    )


def _vlm_error() -> ValidationFinding:
    return ValidationFinding(
        validator="vlm_visual_qa",
        severity="error",
        message="Panel B axis values do not match source data.",
    )


def _warning() -> ValidationFinding:
    return ValidationFinding(
        validator="visual_qa",
        severity="warning",
        message="Minor spacing advisory.",
    )


# =====================================================================
# Commit 1 — VisualGateResult / collect_visual_gate_result
# =====================================================================


def test_collect_visual_gate_result_does_not_run_when_disabled(tmp_path):
    from easyicu.research_agent.pipeline_execute import collect_visual_gate_result

    png = tmp_path / "fig.png"
    result = collect_visual_gate_result(
        enabled=False,
        step_figures=[png],
        step=object(),
        step_summary={},
    )
    assert result.ran is False
    assert result.findings == ()
    assert result.error_findings == ()
    assert result.has_errors is False
    assert result.has_blocking_errors is False
    assert result.was_demoted is False


def test_collect_visual_gate_result_does_not_run_without_figures():
    from easyicu.research_agent.pipeline_execute import collect_visual_gate_result

    result = collect_visual_gate_result(
        enabled=True,
        step_figures=[],
        step=object(),
        step_summary={},
    )
    assert result.ran is False
    assert result.findings == ()


def test_collect_visual_gate_result_assembles_findings_and_demotion(
    monkeypatch, tmp_path
):
    from easyicu.research_agent import pipeline_execute
    from easyicu.research_agent.pipeline_execute import collect_visual_gate_result

    cosmetic, hard, vlm = _cosmetic_error(), _hard_error(), _vlm_error()
    monkeypatch.setattr(
        pipeline_execute, "_expected_numeric_annotations_for_step", lambda **_: None
    )
    monkeypatch.setattr(
        pipeline_execute.VisualQAAuditor,
        "audit_with_expected",
        lambda self, *, figure_paths, expected_numeric_by_path=None: [
            cosmetic,
            hard,
            vlm,
        ],
    )

    result = collect_visual_gate_result(
        enabled=True,
        step_figures=[tmp_path / "fig.png"],
        step=object(),
        step_summary={},
    )

    assert result.ran is True
    assert result.findings == (cosmetic, hard, vlm)
    assert [f.message for f in result.error_findings] == [
        cosmetic.message,
        hard.message,
        vlm.message,
    ]
    # cosmetic demotes to warning; hard + vlm stay error and remain blocking.
    assert [f.message for f in result.blocking_errors] == [hard.message, vlm.message]
    assert result.has_errors is True
    assert result.has_blocking_errors is True
    assert result.was_demoted is True
    # demoted_findings preserves ordering and demotes only the cosmetic one.
    assert [f.severity for f in result.demoted_findings] == [
        "warning",
        "error",
        "error",
    ]


def test_collect_visual_gate_result_clean_when_only_warnings(monkeypatch, tmp_path):
    from easyicu.research_agent import pipeline_execute
    from easyicu.research_agent.pipeline_execute import collect_visual_gate_result

    warn = _warning()
    monkeypatch.setattr(
        pipeline_execute, "_expected_numeric_annotations_for_step", lambda **_: None
    )
    monkeypatch.setattr(
        pipeline_execute.VisualQAAuditor,
        "audit_with_expected",
        lambda self, *, figure_paths, expected_numeric_by_path=None: [warn],
    )

    result = collect_visual_gate_result(
        enabled=True,
        step_figures=[tmp_path / "fig.png"],
        step=object(),
        step_summary={},
    )

    assert result.ran is True
    assert result.has_errors is False
    assert result.has_blocking_errors is False
    assert result.was_demoted is False
    # In the clean path demotion is a no-op: demoted == raw findings.
    assert result.demoted_findings == result.findings


def test_collect_visual_gate_result_passes_numeric_expectations_for_svg_only(
    monkeypatch, tmp_path
):
    from easyicu.research_agent import pipeline_execute
    from easyicu.research_agent.pipeline_execute import collect_visual_gate_result

    captured = {}

    monkeypatch.setattr(
        pipeline_execute,
        "_expected_numeric_annotations_for_step",
        lambda **_: {"AUROC": 0.76},
    )

    def fake_audit(self, *, figure_paths, expected_numeric_by_path=None):
        captured["expected"] = expected_numeric_by_path
        return []

    monkeypatch.setattr(
        pipeline_execute.VisualQAAuditor, "audit_with_expected", fake_audit
    )

    svg = tmp_path / "fig.svg"
    png = tmp_path / "fig.png"
    collect_visual_gate_result(
        enabled=True,
        step_figures=[svg, png],
        step=object(),
        step_summary={},
    )

    # Only the SVG path receives numeric annotations (the PNG does not).
    assert set(captured["expected"]) == {str(svg)}
    assert captured["expected"][str(svg)] == {"AUROC": 0.76}


def test_collect_visual_gate_result_no_numeric_expectations_when_absent(
    monkeypatch, tmp_path
):
    from easyicu.research_agent import pipeline_execute
    from easyicu.research_agent.pipeline_execute import collect_visual_gate_result

    captured = {}

    monkeypatch.setattr(
        pipeline_execute, "_expected_numeric_annotations_for_step", lambda **_: None
    )

    def fake_audit(self, *, figure_paths, expected_numeric_by_path=None):
        captured["expected"] = expected_numeric_by_path
        return []

    monkeypatch.setattr(
        pipeline_execute.VisualQAAuditor, "audit_with_expected", fake_audit
    )

    collect_visual_gate_result(
        enabled=True,
        step_figures=[tmp_path / "fig.svg"],
        step=object(),
        step_summary={},
    )
    # No expected numerics -> None (never an empty dict), matching the original.
    assert captured["expected"] is None
