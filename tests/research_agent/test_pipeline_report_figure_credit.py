"""Gate crediting for figures salvaged by a rendering-only repair step.

A ``*_figure`` step can fail (its own runner emits no exports) yet be salvaged
by a later ``*_figure_repair`` step that renders the figure. That repair step is
not a required plan step, so ``execution_gate_status`` would otherwise keep the
original figure step in ``failed_steps`` and fail-close the whole run — even
though the figure deliverable exists on disk. These tests lock the credit rule
and, crucially, its guardrails against over-crediting.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from easyicu.research_agent.pipeline_report import execution_gate_status


def _plan(*step_ids: str):
    return SimpleNamespace(steps=[SimpleNamespace(step_id=s) for s in step_ids])


def _write_repair(
    run_dir: Path,
    repair_step_id: str,
    parent_step: str,
    *,
    status: str = "ok",
    rendering_only: bool = True,
    with_figure: bool = True,
) -> None:
    out = run_dir / "steps" / repair_step_id / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "step_summary.json").write_text(
        json.dumps(
            {
                "step_id": repair_step_id,
                "parent_step": parent_step,
                "status": status,
                "rendering_only": rendering_only,
                "figure_paths": {"png": str(out / "fig.png")} if with_figure else {},
            }
        )
    )
    if with_figure:
        (out / "fig.png").write_bytes(b"\x89PNG\r\n\x1a\n")


def test_failed_figure_is_credited_when_repair_rendered_it(tmp_path: Path):
    fig = "01_target_trial_protocol_and_cohort_flow_figure"
    _write_repair(tmp_path, "03c1_target_trial_protocol_figure_repair",
                  "01_target_trial_protocol_and_cohort_flow")
    records = [{"step_id": fig, "status": "execution_failed"}]

    # legacy (no run_dir) still fails — backward compatible
    legacy = execution_gate_status(plan=_plan(fig), per_step_records=records)
    assert legacy["execution_complete"] is False
    assert legacy["failed_steps"] == [{"step_id": fig, "status": "execution_failed"}]

    # with run_dir the repaired figure is credited
    credited = execution_gate_status(
        plan=_plan(fig), per_step_records=records, run_dir=tmp_path
    )
    assert credited["execution_complete"] is True
    assert credited["failed_steps"] == []
    assert credited["completed_step_count"] == 1


def test_unrelated_failure_still_blocks_even_with_a_repair_present(tmp_path: Path):
    # a repair exists for a figure, but a DIFFERENT real step failed
    _write_repair(tmp_path, "03c1_target_trial_protocol_figure_repair",
                  "01_target_trial_protocol_and_cohort_flow")
    records = [{"step_id": "02_primary_analysis", "status": "execution_failed"}]
    gate = execution_gate_status(
        plan=_plan("02_primary_analysis"), per_step_records=records, run_dir=tmp_path
    )
    assert gate["execution_complete"] is False
    assert gate["failed_steps"] == [
        {"step_id": "02_primary_analysis", "status": "execution_failed"}
    ]


def test_repair_without_a_rendered_figure_does_not_credit(tmp_path: Path):
    fig = "01_flow_figure"
    _write_repair(tmp_path, "03c1_flow_figure_repair", "01_flow", with_figure=False)
    gate = execution_gate_status(
        plan=_plan(fig),
        per_step_records=[{"step_id": fig, "status": "execution_failed"}],
        run_dir=tmp_path,
    )
    assert gate["execution_complete"] is False


def test_failed_repair_does_not_credit(tmp_path: Path):
    fig = "01_flow_figure"
    _write_repair(tmp_path, "03c1_flow_figure_repair", "01_flow", status="execution_failed")
    gate = execution_gate_status(
        plan=_plan(fig),
        per_step_records=[{"step_id": fig, "status": "execution_failed"}],
        run_dir=tmp_path,
    )
    assert gate["execution_complete"] is False


def test_parent_mismatch_does_not_credit_wrong_figure(tmp_path: Path):
    # repair rendered a figure for parent "01_flow", but the FAILED step is a
    # different figure — exact parent_step matching must not cross-credit.
    _write_repair(tmp_path, "03c1_flow_figure_repair", "01_flow")
    failed = "01_survival_analysis_figure"
    gate = execution_gate_status(
        plan=_plan(failed),
        per_step_records=[{"step_id": failed, "status": "execution_failed"}],
        run_dir=tmp_path,
    )
    assert gate["execution_complete"] is False
    assert gate["failed_steps"] == [
        {"step_id": failed, "status": "execution_failed"}
    ]


def test_non_rendering_only_repair_does_not_credit(tmp_path: Path):
    # a repair that re-ran analysis (not rendering_only) is not a pure figure
    # salvage and must not silently credit a failed figure step.
    fig = "01_flow_figure"
    _write_repair(tmp_path, "03c1_flow_figure_repair", "01_flow", rendering_only=False)
    gate = execution_gate_status(
        plan=_plan(fig),
        per_step_records=[{"step_id": fig, "status": "execution_failed"}],
        run_dir=tmp_path,
    )
    assert gate["execution_complete"] is False
