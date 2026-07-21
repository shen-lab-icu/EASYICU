"""Deterministic executor for one Planner-owned grouped Table 1.

The Planner owns the grouping variable, closed levels, row variables, summary
families, and comparison tests through ``AnalysisStep.table_one_spec``.  This
module only turns that closed declaration into a small sandbox script; it never
chooses a cohort, variable, level, summary, or statistical test.
"""

from __future__ import annotations

import textwrap

from ...schema import AnalysisStep

__all__ = ["table_one_executor_code", "table_one_executor_owns_step"]


def table_one_executor_owns_step(step: AnalysisStep) -> bool:
    """Return whether the exact output contract is fully host-executable."""

    outputs = {str(value or "").strip() for value in step.expected_outputs}
    return bool(
        step.table_one_spec is not None
        and "table:table_one" in outputs
        and not any(value.startswith("figure:") for value in outputs)
        and outputs == {"table:table_one"}
    )


def table_one_executor_code(step: AnalysisStep) -> str:
    """Return sandbox code for the exact Planner-owned Table 1 declaration."""

    if not table_one_executor_owns_step(step):
        raise ValueError("The step is not owned by the grouped Table 1 executor")
    assert step.table_one_spec is not None
    specification = step.table_one_spec.model_dump(mode="python")
    return textwrap.dedent(f"""
        import json
        import os
        from pathlib import Path

        import pandas as pd

        from easyicu.research_agent.methods.table_one import (
            build_grouped_table_one,
            table_one_spec_sha256,
        )

        table_one_spec = {specification!r}
        cohort_path = Path(os.environ["COHORT_PARQUET"])
        out_dir = Path(os.environ["STEP_OUT_DIR"])
        out_dir.mkdir(parents=True, exist_ok=True)

        frame = pd.read_parquet(cohort_path)
        table_one = build_grouped_table_one(frame, table_one_spec)
        table_path = out_dir / "table_one.csv"
        table_one.to_csv(table_path, index=False)

        summary = {{
            "step_id": {step.step_id!r},
            "status": "ok",
            "analysis_family": "grouped_table_one",
            "interpretation_class": "descriptive_baseline_characteristics",
            "method": "Planner-declared grouped Table 1 executed by the host SDK.",
            "cohort_path": str(cohort_path),
            "cohort_n": int(len(frame)),
            "group_by": str(table_one_spec["group_by"]),
            "group_levels": list(table_one_spec["group_levels"]),
            "variables": [item["name"] for item in table_one_spec["variables"]],
            "table_one_contract_sha256": table_one_spec_sha256(table_one_spec),
            "table_one_result_rows": int(len(table_one)),
            "table_one_path": table_path.name,
            "adjusted_effect": None,
            "output_files": {{"table:table_one": table_path.name}},
            "notes": [
                "Overall and comparison groups use the exact Planner-declared levels.",
                "Missing n (%) uses each displayed group's full denominator.",
                "P values use the exact Planner-declared test family without adjustment because this is a descriptive baseline table.",
                "No row, variable, level, or test was selected by the executor.",
            ],
        }}
        (out_dir / "step_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(json.dumps({{"grouped_table_one": "ok", "cohort_n": len(frame)}}))
        """).strip()
