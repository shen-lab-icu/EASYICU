from __future__ import annotations

import json

import pandas as pd
import pytest

from easyicu.research_agent.contracts.table_one import table_one_output_findings
from easyicu.research_agent.audits import StepSummaryIntegrityValidator
from easyicu.research_agent.authority.typed_binding import (
    _write_host_input_binding_receipts,
)
from easyicu.research_agent.execution.runners.table_one_executor import (
    table_one_executor_code,
    table_one_executor_owns_step,
)
from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _step(*, outputs: list[str] | None = None) -> AnalysisStep:
    return AnalysisStep(
        step_id="02_table_one",
        intent="Describe the locked analysis cohort by outcome.",
        inputs=[
            "artifact:analysis_cohort",
            "death",
            "age",
            "sex",
            "lact_max",
            "lact_measured",
            "lact_n",
        ],
        expected_outputs=outputs or ["table:table_one"],
        method="grouped baseline characteristics",
        table_one_spec={
            "group_by": "death",
            "group_levels": [0, 1],
            "variables": [
                {
                    "name": "age",
                    "variable_kind": "continuous",
                    "summary": "median_iqr",
                    "test": "mann_whitney_or_kruskal",
                },
                {
                    "name": "sex",
                    "variable_kind": "categorical",
                    "summary": "count_percent",
                    "test": "chi_square_with_fisher_exact_for_sparse_2x2",
                    "levels": ["Female", "Male"],
                },
                {
                    "name": "lact_max",
                    "variable_kind": "continuous",
                    "summary": "median_iqr",
                    "test": "mann_whitney_or_kruskal",
                },
                {
                    "name": "lact_measured",
                    "variable_kind": "categorical",
                    "summary": "count_percent",
                    "test": "chi_square_with_fisher_exact_for_sparse_2x2",
                    "levels": [0, 1],
                },
            ],
        },
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "death": [0, 0, 0, 1, 1, 1],
            "age": [50.0, 60.0, 70.0, 65.0, 75.0, 85.0],
            "sex": ["Female", "Male", "Female", "Male", "Female", "Male"],
            "lact_max": [1.1, None, 2.0, 3.0, 4.0, None],
            "lact_measured": [1, 0, 1, 1, 1, 0],
            "lact_n": [1, 0, 2, 1, 3, 0],
        }
    )


def test_table_one_executor_owns_only_the_closed_table_contract():
    assert table_one_executor_owns_step(_step())
    assert not table_one_executor_owns_step(
        _step(outputs=["table:table_one", "figure:table_one"])
    )


def test_table_one_executor_does_not_ignore_a_second_typed_artifact():
    step = _step()
    step.inputs.insert(1, "artifact:validated_measurement_analysis_set")

    assert not table_one_executor_owns_step(step)
    assert (
        select_standard_executor(
            step,
            plan=AnalysisPlan(research_question="Test", steps=[step]),
        )
        is None
    )


@pytest.mark.parametrize(
    "typed_input",
    [
        "artifact:validated_measurement_analysis_set",
        "dataset:validated_measurement_analysis_set",
        "cohort:validated_measurement_analysis_set",
        "table:validated_measurement_analysis_set",
    ],
)
def test_table_one_executor_refuses_subset_only_typed_input(typed_input: str):
    step = _step()
    step.inputs = [
        typed_input,
        "death",
        "age",
        "sex",
        "lact_max",
        "lact_measured",
        "lact_n",
    ]

    assert not table_one_executor_owns_step(step)
    assert (
        select_standard_executor(
            step,
            plan=AnalysisPlan(research_question="Test", steps=[step]),
        )
        is None
    )


def test_standard_executor_selects_table_one_before_any_coder_path():
    step = _step()
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Test", steps=[step]),
    )

    assert selection is not None
    assert selection.analysis_kind == "grouped_table_one"
    assert selection.selection_reason == "table_one_spec_preflight"
    assert audit_mechanical_code_contracts(selection.code, step) == []


def test_table_one_executor_code_passes_preflight_and_executes_exact_spec(
    tmp_path, monkeypatch
):
    step = _step()
    cohort_path = tmp_path / "cohort.parquet"
    out_dir = tmp_path / "outputs"
    _frame().to_parquet(cohort_path, index=False)
    monkeypatch.setenv("COHORT_PARQUET", str(cohort_path))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))

    code = table_one_executor_code(step)
    assert audit_mechanical_code_contracts(code, step) == []
    exec(compile(code, "<table-one-executor>", "exec"), {})

    table = pd.read_csv(out_dir / "table_one.csv")
    summary = json.loads((out_dir / "step_summary.json").read_text("utf-8"))
    assert table_one_output_findings(step=step, out_dir=out_dir) == []
    assert set(table["group"]) == {"Overall", "0", "1"}
    assert summary["cohort_n"] == 6
    assert summary["output_files"] == {"table:table_one": "table_one.csv"}
    assert summary["adjusted_effect"] is None
    assert summary["measurement_provenance_audit"] == {
        "source": "COHORT_PARQUET",
        "checks": [
            {
                "measured_column": "lact_measured",
                "count_column": "lact_n",
                "status": "checked",
                "comparison_n": 6,
                "invalid_pair_n": 0,
                "discordant_n": 0,
                "role": "audit_only",
            }
        ],
    }


def test_host_seals_standard_executor_input_and_measurement_receipts(
    tmp_path, monkeypatch
):
    step = _step()
    cohort_path = tmp_path / "cohort.parquet"
    out_dir = tmp_path / "outputs"
    _frame().to_parquet(cohort_path, index=False)
    monkeypatch.setenv("COHORT_PARQUET", str(cohort_path))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    exec(compile(table_one_executor_code(step), "<table-one-executor>", "exec"), {})

    binding = {
        "absolute_path": str(cohort_path),
        "evidence_id": "step01_analysis_cohort",
        "sha256": "a" * 64,
    }
    summary = _write_host_input_binding_receipts(
        out_dir=out_dir,
        step_summary=json.loads((out_dir / "step_summary.json").read_text("utf-8")),
        resolved_input_bindings={"artifact:analysis_cohort": binding},
        consumed_input_keys=("artifact:analysis_cohort",),
    )

    assert (
        StepSummaryIntegrityValidator().audit(
            step=step,
            step_summary=summary,
            resolved_input_bindings={"artifact:analysis_cohort": binding},
            cohort_path=cohort_path,
        )
        == []
    )


def test_host_receipt_never_marks_an_unconsumed_binding_loaded(tmp_path):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    cohort_path = tmp_path / "cohort.parquet"
    other_path = tmp_path / "other.parquet"
    _frame().to_parquet(cohort_path, index=False)
    _frame().to_parquet(other_path, index=False)
    bindings = {
        "artifact:analysis_cohort": {
            "absolute_path": str(cohort_path),
            "evidence_id": "cohort",
            "sha256": "a" * 64,
        },
        "artifact:other": {
            "absolute_path": str(other_path),
            "evidence_id": "other",
            "sha256": "b" * 64,
        },
    }

    summary = _write_host_input_binding_receipts(
        out_dir=out_dir,
        step_summary={"status": "ok"},
        resolved_input_bindings=bindings,
        consumed_input_keys=("artifact:analysis_cohort",),
    )

    assert [item["input_key"] for item in summary["input_bindings"]] == [
        "artifact:analysis_cohort"
    ]


def test_table_one_executor_does_not_silently_claim_a_figure_step():
    with pytest.raises(ValueError, match="not owned"):
        table_one_executor_code(_step(outputs=["table:table_one", "figure:table_one"]))
