"""Focused owner tests for the typed ordered-stratified adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.contracts.ordered_stratified import (
    ordered_stratified_numeric_findings,
    ordered_stratified_script_findings,
    ordered_stratified_structure_findings,
)
from easyicu.research_agent.execution.runners.ordered_stratified_executor import (
    ordered_stratified_executor_owns_step,
    run_ordered_stratified_from_env,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _step() -> AnalysisStep:
    return AnalysisStep.model_validate(
        {
            "step_id": "ordered_trend",
            "planned_analysis_role": "secondary",
            "intent": "Compare an ordered exposure across two declared outcomes.",
            "inputs": ["artifact:analysis_cohort", "severity", "death", "duration"],
            "expected_outputs": [
                "table:ordinal_trend_dose_response",
                "table:ordered_stratified_outcomes",
                "test:ordinal_trend",
            ],
            "method": "ordinal_stratified_descriptive_analysis",
            "scientific_action_id": "association.ordinal_trend",
            "ordered_stratified_spec": {
                "ordered_exposure": "severity",
                "ordered_levels": [0, 1, 2],
                "cochran_armitage_scores": [0, 1, 2],
                "binary_outcome": "death",
                "continuous_outcome": "duration",
                "trend_product": "table:ordinal_trend_dose_response",
            },
        }
    )


def _bind(tmp_path: Path, frame: pd.DataFrame) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / "ordered_trend" / "outputs"
    out_dir.mkdir(parents=True)
    cohort_path = run_dir / "cohort.parquet"
    frame.to_parquet(cohort_path, index=False)
    digest = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
    resolved = {
        "step_id": "ordered_trend",
        "inputs": {
            "artifact:analysis_cohort": {
                "relative_path": "cohort.parquet",
                "sha256": digest,
                "declared_kind": "artifact",
                "product": "analysis_cohort",
                "evidence_id": "ev-cohort",
                "identity_row": {
                    "input_key": "artifact:analysis_cohort",
                    "declared_kind": "artifact",
                    "product": "analysis_cohort",
                    "evidence_id": "ev-cohort",
                    "sha256": digest,
                },
                "product_contract": {
                    "columns": list(frame.columns),
                    "row_count": len(frame),
                },
            }
        },
    }
    resolved_path = run_dir / "resolved_inputs.json"
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
    return out_dir, resolved_path


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "severity": [0, 0, 1, 1, 2, 2, None],
            "death": [0, 1, 0, 1, 1, 1, 0],
            "duration": [1.0, 2.0, 2.0, 4.0, 5.0, 8.0, 9.0],
        }
    )


def test_typed_owner_executes_and_replays_without_coder(monkeypatch, tmp_path: Path) -> None:
    step = _step()
    out_dir, resolved_path = _bind(tmp_path, _frame())
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(resolved_path.parent))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(resolved_path))

    assert ordered_stratified_executor_owns_step(step)
    selected = select_standard_executor(
        step, plan=AnalysisPlan(research_question="Test", steps=[step])
    )
    assert selected is not None
    assert selected.analysis_kind == "ordered_stratified_analysis"

    summary = run_ordered_stratified_from_env(
        spec_payload=step.ordered_stratified_spec.model_dump(mode="json"),
        typed_cohort_input="artifact:analysis_cohort",
        analysis_role="secondary",
    )
    assert not ordered_stratified_structure_findings(step=step, step_summary=summary)
    assert not ordered_stratified_numeric_findings(
        cohort_path=resolved_path.parent / "cohort.parquet",
        step=step,
        out_dir=out_dir,
        step_summary=summary,
    )
    assert not ordered_stratified_script_findings(
        step=step, script_text=selected.code
    )


def test_typed_owner_fails_closed_on_undeclared_exposure_level(
    monkeypatch, tmp_path: Path
) -> None:
    frame = _frame()
    frame.loc[0, "severity"] = 9
    out_dir, resolved_path = _bind(tmp_path, frame)
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(resolved_path.parent))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(resolved_path))

    with pytest.raises(RuntimeError, match="outside the declared level set"):
        run_ordered_stratified_from_env(
            spec_payload=_step().ordered_stratified_spec.model_dump(mode="json"),
            typed_cohort_input="artifact:analysis_cohort",
            analysis_role="secondary",
        )


def test_method_label_without_typed_spec_does_not_select_owner() -> None:
    step = _step().model_copy(update={"ordered_stratified_spec": None})
    assert not ordered_stratified_executor_owns_step(step)
