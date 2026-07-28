"""The pre-run coverage replay must never over-state ownership.

This tool exists to answer "will this plan actually get deterministic
executors?" before a run is paid for.  Its only real failure mode is
optimism: a report that counts an owner the run will decline green-lights
exactly the run that then falls through to the LLM coder, and it does so with
the authority of having been checked.

The E1 Step 02 shape is the concrete instance.  Its contract matches
``descriptive_cohort_summary``, but the selector declines it when the step
owes a host-verified plausibility receipt.  An earlier scan had no scope
object, could not see that gate, and called the step owned; the real run
recorded ``declined_receipt_required``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.owner_coverage_replay import main, replay_owner_coverage


def _plan_payload(*steps: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_question": "Estimate an association in an ICU cohort.",
        "robustness_specs": [],
        "steps": list(steps),
    }


def _receipt_gated_step() -> dict[str, Any]:
    """A cohort summary that a plausibility receipt would take away."""

    return {
        "step_id": "02_cohort_definition_summary",
        "intent": "Summarise the analysis cohort.",
        "method": "descriptive_cohort_summary",
        "planned_analysis_role": "auxiliary",
        "inputs": ["artifact:analysis_cohort", "sofa2_liver_max"],
        "expected_outputs": ["table:cohort_summary"],
    }


def _unowned_step() -> dict[str, Any]:
    return {
        "step_id": "09_bespoke_analysis",
        "intent": "Something outside every closed contract.",
        "method": "bespoke_method",
        "inputs": [],
        "expected_outputs": ["table:bespoke_product"],
    }


def _write(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "analysis_plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_receipt_gated_owner_is_reported_conditional_not_owned(
    tmp_path: Path,
) -> None:
    plan_path = _write(tmp_path, _plan_payload(_receipt_gated_step()))

    (row,) = replay_owner_coverage(plan_path)

    assert row.verdict == "conditional_on_receipt"
    assert row.analysis_kind == "descriptive_cohort_summary"
    assert "receipt" in row.note


def test_conditional_steps_are_not_counted_as_covered(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The headline number is what an operator acts on."""

    plan_path = _write(tmp_path, _plan_payload(_receipt_gated_step()))

    assert main([str(plan_path)]) == 0

    out = capsys.readouterr().out
    assert "owned outright      : 0/1" in out
    assert "conditional on gate : 1/1" in out
    assert "CONDITIONAL" in out


def test_a_step_no_owner_claims_is_reported_as_coder(tmp_path: Path) -> None:
    plan_path = _write(tmp_path, _plan_payload(_unowned_step()))

    (row,) = replay_owner_coverage(plan_path)

    assert row.verdict == "coder"
    assert row.analysis_kind is None


def test_an_unconditionally_owned_step_is_reported_owned(tmp_path: Path) -> None:
    """The audit contract does not owe a receipt, so it survives the gate."""

    step = {
        "step_id": "05_missingness_measurement_process_audit",
        "intent": "Audit measurement process.",
        "method": "data_quality_audit",
        "planned_analysis_role": "auxiliary",
        "inputs": ["artifact:analysis_cohort", "sep3_sofa2_measured"],
        "expected_outputs": [
            "table:missingness_measurement_audit",
            "table:measurement_process_audit",
        ],
    }
    plan_path = _write(tmp_path, _plan_payload(step))

    (row,) = replay_owner_coverage(plan_path)

    assert row.verdict == "owned"
    assert row.analysis_kind == "declared_missingness_audit_products"


def test_a_raising_step_degrades_to_coder_rather_than_killing_the_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scan that dies on one step tells the operator nothing about the rest."""

    from easyicu.research_agent.execution.runners import selection

    def _explode(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("selector exploded")

    monkeypatch.setattr(selection, "select_standard_executor", _explode)
    monkeypatch.setattr(
        "easyicu.research_agent.execution.runners.selection."
        "select_standard_executor",
        _explode,
    )
    plan_path = _write(tmp_path, _plan_payload(_unowned_step(), _receipt_gated_step()))

    rows = replay_owner_coverage(plan_path)

    assert len(rows) == 2
    assert all(row.verdict == "coder" for row in rows)
    assert all("RuntimeError" in row.note for row in rows)


def test_json_output_is_machine_readable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = _write(tmp_path, _plan_payload(_receipt_gated_step(), _unowned_step()))

    assert main([str(plan_path), "--json"]) == 0

    rows = json.loads(capsys.readouterr().out)
    assert [row["verdict"] for row in rows] == ["conditional_on_receipt", "coder"]
