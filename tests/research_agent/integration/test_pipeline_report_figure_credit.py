"""Gate crediting for figures salvaged by a rendering-only repair step.

A ``*_figure`` step can fail (its own runner emits no exports) yet be salvaged
by a later ``*_figure_repair`` step that renders the figure. That repair step is
not a required plan step, so ``execution_gate_status`` would otherwise keep the
original figure step in ``failed_steps`` and fail-close the whole run — even
though the figure deliverable exists on disk. These tests lock the credit rule
and, crucially, its guardrails against over-crediting.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.reporting.readiness import execution_gate_status


def _plan(*step_ids: str):
    return SimpleNamespace(
        steps=[
            SimpleNamespace(
                step_id=s,
                intent="",
                method="visualization",
                expected_outputs=["figure:publication_figure"],
            )
            for s in step_ids
        ]
    )


def _write_repair(
    run_dir: Path,
    repair_step_id: str,
    parent_step: str,
    *,
    status: str = "ok",
    rendering_only: bool = True,
    with_figure: bool = True,
) -> dict:
    out = run_dir / "steps" / repair_step_id / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "step_id": repair_step_id,
        "source_step_id": parent_step,
        "status": status,
        "rendering_only": rendering_only,
        "figure_paths": {"png": str(out / "fig.png")} if with_figure else {},
    }
    (out / "step_summary.json").write_text(json.dumps(summary))
    if with_figure:
        (out / "fig.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return {
        "step_id": repair_step_id,
        "status": status,
        "step_summary": summary,
        "repair_target_step_id": f"{parent_step}_figure",
        "source_evidence_ids": [f"source_{parent_step}"],
    }


def _write_modern_authority(
    run_dir: Path,
    *,
    records: list[dict],
    evidence: list[dict],
) -> None:
    (run_dir / "manifest_partial.json").write_text(
        json.dumps({"per_step_records": records, "evidence": evidence}),
        encoding="utf-8",
    )


def _records_with_current_source(fig: str, repair: dict) -> list[dict]:
    source_step_id = str(repair["step_summary"]["source_step_id"])
    return [
        {
            "step_id": source_step_id,
            "status": "ok",
            "evidence_ids": list(repair["source_evidence_ids"]),
        },
        {"step_id": fig, "status": "execution_failed"},
        repair,
    ]


def _write_digest_bound_authority(
    run_dir: Path,
    *,
    fig: str,
    repair: dict,
) -> tuple[list[dict], Path]:
    source_step_id = str(repair["step_summary"]["source_step_id"])
    source_id = repair["source_evidence_ids"][0]
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    source_path = evidence_dir / f"{source_id}__source.csv"
    source_path.write_text("term,estimate\nexposure,1.2\n", encoding="utf-8")
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()

    output = run_dir / "steps" / repair["step_id"] / "outputs" / "fig.png"
    figure_id = "figure_fig_current"
    figure_path = evidence_dir / f"{figure_id}__fig.png"
    figure_path.write_bytes(output.read_bytes())
    figure_digest = hashlib.sha256(figure_path.read_bytes()).hexdigest()
    repair["evidence_ids"] = [figure_id]
    records = _records_with_current_source(fig, repair)
    _write_modern_authority(
        run_dir,
        records=records,
        evidence=[
            {
                "evidence_id": source_id,
                "kind": "table",
                "produced_by_step": source_step_id,
                "relative_path": str(source_path.relative_to(run_dir)),
                "sha256": source_digest,
            },
            {
                "evidence_id": figure_id,
                "kind": "figure",
                "produced_by_step": repair["step_id"],
                "relative_path": str(figure_path.relative_to(run_dir)),
                "sha256": figure_digest,
                "inputs": [source_id],
            },
        ],
    )
    return records, source_path


def test_failed_figure_is_credited_when_repair_rendered_it(tmp_path: Path):
    fig = "01_target_trial_protocol_and_cohort_flow_figure"
    repair = _write_repair(
        tmp_path,
        "03c1_target_trial_protocol_figure_repair",
        "01_target_trial_protocol_and_cohort_flow",
    )
    records = [{"step_id": fig, "status": "execution_failed"}, repair]

    # legacy (no run_dir) still fails — backward compatible
    legacy = execution_gate_status(plan=_plan(fig), per_step_records=records)
    assert legacy["execution_complete"] is False
    assert legacy["failed_steps"] == [{"step_id": fig, "status": "execution_failed"}]

    # With current digest-bound source + figure authority, the repair is credited.
    records, _source_path = _write_digest_bound_authority(
        tmp_path,
        fig=fig,
        repair=repair,
    )
    credited = execution_gate_status(
        plan=_plan(fig), per_step_records=records, run_dir=tmp_path
    )
    assert credited["execution_complete"] is True
    assert credited["failed_steps"] == []
    assert credited["completed_step_count"] == 1


def test_basename_relative_repair_export_is_resolved_from_its_outputs_dir(
    tmp_path: Path,
):
    fig = "01_flow_figure"
    repair = _write_repair(tmp_path, "03c1_flow_figure_repair", "01_flow")
    repair["step_summary"]["figure_paths"] = {"png": "fig.png"}
    records, _source_path = _write_digest_bound_authority(
        tmp_path,
        fig=fig,
        repair=repair,
    )

    gate = execution_gate_status(
        plan=_plan(fig),
        per_step_records=records,
        run_dir=tmp_path,
    )

    assert gate["execution_complete"] is True


def test_relative_repair_export_cannot_escape_its_outputs_dir(tmp_path: Path):
    fig = "01_flow_figure"
    repair = _write_repair(
        tmp_path,
        "03c1_flow_figure_repair",
        "01_flow",
        with_figure=False,
    )
    outside = tmp_path / "steps" / "03c1_flow_figure_repair" / "outside.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\n")
    repair["step_summary"]["figure_paths"] = {"png": "../outside.png"}

    gate = execution_gate_status(
        plan=_plan(fig),
        per_step_records=[{"step_id": fig, "status": "execution_failed"}, repair],
        run_dir=tmp_path,
    )

    assert gate["execution_complete"] is False


def test_repair_step_id_cannot_traverse_outside_steps_dir(tmp_path: Path):
    fig = "01_flow_figure"
    repair = _write_repair(tmp_path, "../outside_repair", "01_flow")
    repair["step_summary"]["figure_paths"] = {"png": "fig.png"}

    gate = execution_gate_status(
        plan=_plan(fig),
        per_step_records=[{"step_id": fig, "status": "execution_failed"}, repair],
        run_dir=tmp_path,
    )

    assert gate["execution_complete"] is False


def test_symlinked_repair_export_cannot_escape_its_outputs_dir(tmp_path: Path):
    fig = "01_flow_figure"
    repair = _write_repair(
        tmp_path,
        "03c1_flow_figure_repair",
        "01_flow",
        with_figure=False,
    )
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\n")
    link = tmp_path / "steps" / "03c1_flow_figure_repair" / "outputs" / "fig.png"
    try:
        link.symlink_to(outside)
    except OSError:
        return
    repair["step_summary"]["figure_paths"] = {"png": "fig.png"}

    gate = execution_gate_status(
        plan=_plan(fig),
        per_step_records=[{"step_id": fig, "status": "execution_failed"}, repair],
        run_dir=tmp_path,
    )

    assert gate["execution_complete"] is False


def test_modern_manifest_requires_active_figure_evidence_binding(tmp_path: Path):
    fig = "01_flow_figure"
    repair = _write_repair(tmp_path, "03c1_flow_figure_repair", "01_flow")
    repair["step_summary"]["figure_paths"] = {"png": "fig.png"}
    records = _records_with_current_source(fig, repair)
    _write_modern_authority(tmp_path, records=records, evidence=[])

    gate = execution_gate_status(
        plan=_plan(fig), per_step_records=records, run_dir=tmp_path
    )

    assert gate["execution_complete"] is False


def test_modern_manifest_credits_digest_bound_current_figure(tmp_path: Path):
    fig = "01_flow_figure"
    repair = _write_repair(tmp_path, "03c1_flow_figure_repair", "01_flow")
    repair["step_summary"]["figure_paths"] = {"png": "fig.png"}
    records, _source_path = _write_digest_bound_authority(
        tmp_path,
        fig=fig,
        repair=repair,
    )

    gate = execution_gate_status(
        plan=_plan(fig), per_step_records=records, run_dir=tmp_path
    )

    assert gate["execution_complete"] is True


@pytest.mark.parametrize(
    "expected_output",
    ["table:association_estimates", "table:publication_figure_summary"],
)
def test_render_repair_cannot_credit_failed_nonfigure_science_step(
    tmp_path: Path,
    expected_output: str,
) -> None:
    target_step_id = "02_primary_model"
    source_step_id = "01_source"
    repair = _write_repair(
        tmp_path,
        "09_detached_figure_repair",
        source_step_id,
    )
    repair["repair_target_step_id"] = target_step_id
    records, _source_path = _write_digest_bound_authority(
        tmp_path,
        fig=target_step_id,
        repair=repair,
    )
    plan = SimpleNamespace(
        steps=[
            SimpleNamespace(
                step_id=source_step_id,
                intent="Create the exposure table.",
                method="descriptive",
                expected_outputs=["table:exposure"],
            ),
            SimpleNamespace(
                step_id=target_step_id,
                intent=(
                    "Fit the primary model using the exposure declared by step "
                    f"'{source_step_id}'."
                ),
                method="mixed_effects_regression",
                expected_outputs=[expected_output],
            ),
        ]
    )

    gate = execution_gate_status(
        plan=plan,
        per_step_records=records,
        run_dir=tmp_path,
    )

    assert gate["execution_complete"] is False
    assert gate["failed_steps"] == [
        {"step_id": target_step_id, "status": "execution_failed"}
    ]


def test_modern_manifest_rejects_repair_when_source_evidence_is_tampered(
    tmp_path: Path,
):
    fig = "01_flow_figure"
    repair = _write_repair(tmp_path, "03c1_flow_figure_repair", "01_flow")
    repair["step_summary"]["figure_paths"] = {"png": "fig.png"}
    records, source_path = _write_digest_bound_authority(
        tmp_path,
        fig=fig,
        repair=repair,
    )
    source_path.write_text("term,estimate\nexposure,9.9\n", encoding="utf-8")

    gate = execution_gate_status(
        plan=_plan(fig), per_step_records=records, run_dir=tmp_path
    )

    assert gate["execution_complete"] is False


def test_modern_manifest_rejects_same_step_file_with_stale_digest(tmp_path: Path):
    fig = "01_flow_figure"
    repair = _write_repair(tmp_path, "03c1_flow_figure_repair", "01_flow")
    repair["step_summary"]["figure_paths"] = {"png": "fig.png"}
    evidence_id = "figure_fig_current"
    evidence_path = tmp_path / "evidence" / f"{evidence_id}__fig.png"
    evidence_path.parent.mkdir()
    evidence_path.write_bytes(b"new current figure")
    digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    repair["evidence_ids"] = [evidence_id]
    records = _records_with_current_source(fig, repair)
    _write_modern_authority(
        tmp_path,
        records=records,
        evidence=[
            {
                "evidence_id": evidence_id,
                "kind": "figure",
                "produced_by_step": repair["step_id"],
                "relative_path": str(evidence_path.relative_to(tmp_path)),
                "sha256": digest,
                "inputs": list(repair["source_evidence_ids"]),
            }
        ],
    )

    gate = execution_gate_status(
        plan=_plan(fig), per_step_records=records, run_dir=tmp_path
    )

    assert gate["execution_complete"] is False


def test_unrelated_failure_still_blocks_even_with_a_repair_present(tmp_path: Path):
    # a repair exists for a figure, but a DIFFERENT real step failed
    repair = _write_repair(
        tmp_path,
        "03c1_target_trial_protocol_figure_repair",
        "01_target_trial_protocol_and_cohort_flow",
    )
    records = [
        {"step_id": "02_primary_analysis", "status": "execution_failed"},
        repair,
    ]
    gate = execution_gate_status(
        plan=_plan("02_primary_analysis"), per_step_records=records, run_dir=tmp_path
    )
    assert gate["execution_complete"] is False
    assert gate["failed_steps"] == [
        {"step_id": "02_primary_analysis", "status": "execution_failed"}
    ]


def test_repair_without_a_rendered_figure_does_not_credit(tmp_path: Path):
    fig = "01_flow_figure"
    repair = _write_repair(
        tmp_path, "03c1_flow_figure_repair", "01_flow", with_figure=False
    )
    gate = execution_gate_status(
        plan=_plan(fig),
        per_step_records=[{"step_id": fig, "status": "execution_failed"}, repair],
        run_dir=tmp_path,
    )
    assert gate["execution_complete"] is False


def test_unledgered_stale_repair_file_does_not_credit_modern_run(tmp_path: Path):
    fig = "01_flow_figure"
    _write_repair(tmp_path, "03c1_flow_figure_repair", "01_flow")

    gate = execution_gate_status(
        plan=_plan(fig),
        per_step_records=[{"step_id": fig, "status": "execution_failed"}],
        run_dir=tmp_path,
    )

    assert gate["execution_complete"] is False
    assert gate["failed_steps"] == [{"step_id": fig, "status": "execution_failed"}]


def test_failed_repair_does_not_credit(tmp_path: Path):
    fig = "01_flow_figure"
    repair = _write_repair(
        tmp_path,
        "03c1_flow_figure_repair",
        "01_flow",
        status="execution_failed",
    )
    gate = execution_gate_status(
        plan=_plan(fig),
        per_step_records=[{"step_id": fig, "status": "execution_failed"}, repair],
        run_dir=tmp_path,
    )
    assert gate["execution_complete"] is False


def test_parent_mismatch_does_not_credit_wrong_figure(tmp_path: Path):
    # repair rendered a figure for parent "01_flow", but the FAILED step is a
    # different figure — exact parent_step matching must not cross-credit.
    repair = _write_repair(tmp_path, "03c1_flow_figure_repair", "01_flow")
    failed = "01_survival_analysis_figure"
    gate = execution_gate_status(
        plan=_plan(failed),
        per_step_records=[
            {"step_id": failed, "status": "execution_failed"},
            repair,
        ],
        run_dir=tmp_path,
    )
    assert gate["execution_complete"] is False
    assert gate["failed_steps"] == [{"step_id": failed, "status": "execution_failed"}]


def test_renderer_self_report_cannot_claim_unledgered_target(tmp_path: Path):
    fig = "01_flow_figure"
    repair = _write_repair(tmp_path, "09_unrelated_renderer", "01_flow")
    repair.pop("repair_target_step_id")

    gate = execution_gate_status(
        plan=_plan(fig),
        per_step_records=[{"step_id": fig, "status": "execution_failed"}, repair],
        run_dir=tmp_path,
    )

    assert gate["execution_complete"] is False


def test_non_rendering_only_repair_does_not_credit(tmp_path: Path):
    # a repair that re-ran analysis (not rendering_only) is not a pure figure
    # salvage and must not silently credit a failed figure step.
    fig = "01_flow_figure"
    repair = _write_repair(
        tmp_path,
        "03c1_flow_figure_repair",
        "01_flow",
        rendering_only=False,
    )
    gate = execution_gate_status(
        plan=_plan(fig),
        per_step_records=[{"step_id": fig, "status": "execution_failed"}, repair],
        run_dir=tmp_path,
    )
    assert gate["execution_complete"] is False


def test_string_rendering_only_flag_does_not_credit(tmp_path: Path):
    fig = "01_flow_figure"
    repair = _write_repair(tmp_path, "03c1_flow_figure_repair", "01_flow")
    repair["step_summary"]["rendering_only"] = "false"

    gate = execution_gate_status(
        plan=_plan(fig),
        per_step_records=[{"step_id": fig, "status": "execution_failed"}, repair],
        run_dir=tmp_path,
    )

    assert gate["execution_complete"] is False
