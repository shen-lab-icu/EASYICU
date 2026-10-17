"""Deterministic secondary summaries for an explicit continuous-outcome roster."""

from __future__ import annotations

import json
from pathlib import Path
import textwrap
from typing import Any, Sequence

import pandas as pd

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...contracts.continuous_outcome_summary import (
    CONTINUOUS_OUTCOME_OUTPUT, CONTINUOUS_OUTCOME_SUMMARY,
    continuous_outcome_columns, continuous_outcome_summary_verdict,
)
from ...contracts.cohort_product_keys import sole_typed_cohort_input
from ...contracts.continuous_outcome_metadata import continuous_summary_columns
from ...schema import AnalysisStep, ConceptDescriptor
from .descriptive_distribution_executor import grouped_distribution_table
from .plausibility_receipt import render_standard_plausibility_receipt_code


def continuous_outcome_summary_executor_code(
    step: AnalysisStep, *, plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> str:
    if not continuous_outcome_summary_verdict(step).claimed:
        raise ValueError("step does not declare the continuous-outcome summary contract")
    grouping, *outcomes = continuous_outcome_columns(step)
    code = textwrap.dedent(f"""
        import json
        import os
        from pathlib import Path
        from easyicu.research_agent.execution.runners.typed_input_binding import load_step_cohort_frame
        from easyicu.research_agent.execution.runners.continuous_outcome_summary_executor import run_continuous_outcome_summary
        typed_cohort_input = {sole_typed_cohort_input(step)!r}
        frame, cohort_path = load_step_cohort_frame(typed_cohort_input=typed_cohort_input)
    """).strip()
    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
        if plausibility_scope.expected_columns:
            code += "\n" + render_standard_plausibility_receipt_code(plausibility_scope, frame_name="frame")
    code += "\n" + textwrap.dedent(f"""
        summary = run_continuous_outcome_summary(
            frame=frame, grouping_variable={grouping!r}, outcome_variables={outcomes!r},
            typed_cohort_input=typed_cohort_input, source_cohort=cohort_path,
            out_dir=Path(os.environ["STEP_OUT_DIR"]), run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
        )
    """).strip()
    if plausibility_scope is not None and plausibility_scope.expected_columns:
        code += '\nsummary["plausibility_audit"] = plausibility_audit'
    code += '\n(Path(os.environ["STEP_OUT_DIR"]) / "step_summary.json").write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False), encoding="utf-8")\nprint(json.dumps(summary, ensure_ascii=False, allow_nan=False))'
    return code


def run_continuous_outcome_summary(
    *, frame: pd.DataFrame, grouping_variable: str, outcome_variables: Sequence[str],
    typed_cohort_input: str, source_cohort: Path, out_dir: Path, run_dir: Path,
) -> dict[str, Any]:
    """Preserve each outcome's own denominator, unit, missingness and quantiles."""
    payload = json.loads((Path(run_dir) / "research_context.json").read_text("utf-8"))
    variables = {row["name"]: ConceptDescriptor.model_validate(row) for row in payload["variables"]}
    try:
        continuous_summary_columns(grouping=grouping_variable,
                                   raw_inputs=[grouping_variable, *outcome_variables], variables=variables)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    tables = []
    for outcome in outcome_variables:
        table, _levels = grouped_distribution_table(
            frame=frame, grouping_variable=grouping_variable,
            value_variable=outcome, run_dir=run_dir,
        )
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "continuous_outcome_summary.csv"
    combined.to_csv(table_path, index=False)
    return {
        "status": "ok", "analysis_status": "ok", "analysis_family": "descriptive",
        "analysis_role": "secondary", "analysis_set": "bound_typed_cohort",
        "interpretation_class": CONTINUOUS_OUTCOME_SUMMARY,
        "interpretation_ceiling": "descriptive_unadjusted_not_causal",
        "deterministic_standard_analysis": CONTINUOUS_OUTCOME_SUMMARY,
        "grouping_variable": grouping_variable, "outcome_variables": list(outcome_variables),
        "typed_cohort_input": typed_cohort_input, "source_cohort": Path(source_cohort).name,
        "source_row_count_reconciliation": {"source_rows": len(frame), "analyzed_rows": len(frame), "filtering_performed": False},
        "adjusted_effect": None, "uncertainty_interval": None, "inferential_tests": [],
        "output_files": {CONTINUOUS_OUTCOME_OUTPUT: table_path.name},
    }
