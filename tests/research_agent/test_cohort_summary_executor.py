from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.cohort_summary_executor import (
    cohort_summary_executor_code,
    cohort_summary_executor_owns_step,
)
from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _step() -> AnalysisStep:
    return AnalysisStep(
        step_id="02_cohort_summary",
        planned_analysis_role="auxiliary",
        intent="Describe the exact closed cohort columns.",
        inputs=[
            "artifact:analysis_cohort",
            "age",
            "sex",
            "exposure",
            "outcome",
        ],
        expected_outputs=["table:cohort_summary"],
        method="descriptive_cohort_summary",
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [50.0, 60.0, 70.0, 80.0],
            "sex": ["Female", "Male", "Female", "Male"],
            "exposure": [0.0, 1.0, 0.0, 1.0],
            "outcome": [0, 0, 1, 1],
        }
    )


def _context() -> dict:
    return {
        "variables": [
            {
                "name": "age",
                "unit": "years",
                "observed_domain": {
                    "n_unique": 4,
                    "is_binary": False,
                    "min": 50.0,
                    "max": 80.0,
                },
            },
            {
                "name": "sex",
                "unit": None,
                "observed_domain": {
                    "n_unique": 2,
                    "is_binary": False,
                    "levels": ["Female", "Male"],
                },
            },
            {
                "name": "exposure",
                "unit": None,
                "observed_domain": {
                    "n_unique": 2,
                    "is_binary": True,
                    "levels": [0.0, 1.0],
                },
            },
            {
                "name": "outcome",
                "unit": None,
                "observed_domain": {
                    "n_unique": 2,
                    "is_binary": True,
                    "levels": [0, 1],
                },
            },
        ]
    }


def _bind_run(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cohort_path = run_dir / "analysis_cohort.parquet"
    frame = _frame()
    frame.to_parquet(cohort_path, index=False)
    manifest = {
        "inputs": {
            "artifact:analysis_cohort": {
                "relative_path": cohort_path.relative_to(run_dir).as_posix(),
                "sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
                "product_contract": {
                    "columns": list(frame.columns),
                    "row_count": len(frame),
                },
            }
        }
    }
    manifest_path = run_dir / "resolved_inputs.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    context_path = run_dir / "research_context.json"
    context_path.write_text(json.dumps(_context()), encoding="utf-8")
    out_dir = run_dir / "steps" / "02_cohort_summary" / "outputs"
    monkeypatch.setenv("EASYICU_RUN_DIR", str(run_dir))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(manifest_path))
    monkeypatch.setenv("EASYICU_RESEARCH_CONTEXT", str(context_path))
    monkeypatch.setenv("COHORT_PARQUET", str(cohort_path))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    return cohort_path, out_dir


def test_cohort_summary_executor_owns_only_closed_auxiliary_contract():
    step = _step()
    assert cohort_summary_executor_owns_step(step)

    # Canonical9 planners may use the generic method head, but the remaining
    # closed contract still uniquely identifies this mechanical summary.
    assert cohort_summary_executor_owns_step(
        step.model_copy(update={"method": "descriptive"})
    )
    assert not cohort_summary_executor_owns_step(
        step.model_copy(update={"planned_analysis_role": "primary"})
    )
    assert not cohort_summary_executor_owns_step(
        step.model_copy(
            update={
                "expected_outputs": [
                    "table:cohort_summary",
                    "statistic:adjusted_or",
                ]
            }
        )
    )
    assert not cohort_summary_executor_owns_step(
        step.model_copy(
            update={"inputs": [*step.inputs, "table:unrelated_parent"]}
        )
    )


def test_cohort_summary_is_selected_before_coder_and_declares_consumption():
    step = _step()
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Test", steps=[step]),
    )

    assert selection is not None
    assert selection.analysis_kind == "descriptive_cohort_summary"
    assert selection.selection_reason == "cohort_summary_contract_preflight"
    assert selection.consumed_input_keys == ("artifact:analysis_cohort",)
    assert audit_mechanical_code_contracts(selection.code, step) == []


def test_cohort_summary_executes_exact_metadata_levels_and_numeric_statistics(
    tmp_path,
    monkeypatch,
):
    _, out_dir = _bind_run(tmp_path, monkeypatch)
    step = _step()

    exec(compile(cohort_summary_executor_code(step), "<cohort-summary>", "exec"), {})

    table = pd.read_csv(out_dir / "cohort_summary.csv")
    summary = json.loads((out_dir / "step_summary.json").read_text("utf-8"))
    cohort_row = table[
        (table["variable"] == "__cohort__")
        & (table["statistic"] == "cohort_n")
    ].iloc[0]
    age_median = table[
        (table["variable"] == "age") & (table["statistic"] == "median")
    ].iloc[0]
    exposed = table[
        (table["variable"] == "exposure")
        & (table["statistic"] == "level_count")
        & (table["level"].astype(str) == "1.0")
    ].iloc[0]

    assert cohort_row["value"] == 4
    assert age_median["value"] == 65.0
    assert exposed["numerator"] == 2
    assert exposed["denominator"] == 4
    assert exposed["percentage"] == 50.0
    assert summary["status"] == "ok"
    assert summary["cohort_n"] == 4
    assert summary["adjusted_effect"] is None
    assert summary["output_files"] == {
        "table:cohort_summary": "cohort_summary.csv"
    }
    assert summary["source_row_count_reconciliation"] == {
        "source_rows": 4,
        "analyzed_rows": 4,
        "filtering_performed": False,
    }


def test_cohort_summary_rejects_digest_drift(tmp_path, monkeypatch):
    cohort_path, _out_dir = _bind_run(tmp_path, monkeypatch)
    cohort_path.write_bytes(cohort_path.read_bytes() + b"tamper")

    with pytest.raises(RuntimeError, match="digest verification failed"):
        exec(
            compile(
                cohort_summary_executor_code(_step()),
                "<cohort-summary>",
                "exec",
            ),
            {},
        )


def test_cohort_summary_rejects_missing_structured_metadata(tmp_path, monkeypatch):
    _cohort_path, _out_dir = _bind_run(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    context = _context()
    context["variables"] = [
        item for item in context["variables"] if item["name"] != "outcome"
    ]
    (run_dir / "research_context.json").write_text(
        json.dumps(context),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="lack structured metadata: outcome"):
        exec(
            compile(
                cohort_summary_executor_code(_step()),
                "<cohort-summary>",
                "exec",
            ),
            {},
        )
