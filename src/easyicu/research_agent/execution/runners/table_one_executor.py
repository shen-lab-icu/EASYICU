"""Deterministic executor for one Planner-owned grouped Table 1.

The Planner owns the grouping variable, closed levels, row variables, summary
families, and comparison tests through ``AnalysisStep.table_one_spec``.  This
module only turns that closed declaration into a small sandbox script; it never
chooses a cohort, variable, level, summary, or statistical test.
"""

from __future__ import annotations

import textwrap

from ...authority.table_one_binding import table_one_execution_spec
from ...icu_rules import companion_count_column_for_measured
from ...schema import AnalysisStep

__all__ = ["table_one_executor_code", "table_one_executor_owns_step"]


def _typed_cohort_input(step: AnalysisStep) -> str | None:
    """Return the sole typed row-membership authority, when supported."""

    typed_inputs = {
        str(value or "").strip()
        for value in step.inputs
        if ":" in str(value or "").strip()
    }
    if not typed_inputs:
        return None
    if len(typed_inputs) != 1:
        return ""
    input_key = next(iter(typed_inputs))
    kind, separator, product = input_key.partition(":")
    if separator and product and (
        kind == "cohort" or input_key == "artifact:analysis_cohort"
    ):
        return input_key
    return ""


def table_one_executor_owns_step(step: AnalysisStep) -> bool:
    """Return whether the exact output contract is fully host-executable."""

    outputs = {str(value or "").strip() for value in step.expected_outputs}
    typed_cohort_input = _typed_cohort_input(step)
    return bool(
        step.table_one_spec is not None
        and "table:table_one" in outputs
        and not any(value.startswith("figure:") for value in outputs)
        and outputs == {"table:table_one"}
        # No typed input means COHORT_PARQUET is the row authority.  Otherwise
        # the executor supports exactly one explicitly cohort-scoped product
        # (plus the historical artifact:analysis_cohort spelling) and loads
        # that digest-bound table rather than silently analysing another frame.
        and typed_cohort_input != ""
    )


def table_one_executor_code(step: AnalysisStep) -> str:
    """Return sandbox code for the exact Planner-owned Table 1 declaration."""

    if not table_one_executor_owns_step(step):
        raise ValueError("The step is not owned by the grouped Table 1 executor")
    specification_model = table_one_execution_spec(step)
    assert specification_model is not None
    specification = specification_model.model_dump(mode="python")
    typed_cohort_input = _typed_cohort_input(step)
    declared_inputs = {
        str(value).strip()
        for value in step.inputs
        if str(value).strip() and ":" not in str(value)
    }
    measurement_pairs = sorted(
        (measured_column, count_column)
        for measured_column in declared_inputs
        if (count_column := companion_count_column_for_measured(measured_column))
        is not None
        and count_column in declared_inputs
    )
    return textwrap.dedent(f"""
        import hashlib
        import json
        import os
        from pathlib import Path

        import pandas as pd

        from easyicu.research_agent.methods.table_one import (
            build_grouped_table_one,
            table_one_spec_sha256,
        )
        from easyicu.research_agent.methods.descriptive_inputs import (
            measurement_provenance_receipt,
        )

        table_one_spec = {specification!r}
        measurement_pairs = {measurement_pairs!r}
        typed_cohort_input = {typed_cohort_input!r}
        out_dir = Path(os.environ["STEP_OUT_DIR"])
        out_dir.mkdir(parents=True, exist_ok=True)

        def sha256_file(path):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        def load_typed_cohort(input_key):
            run_dir = Path(os.environ["EASYICU_RUN_DIR"]).resolve()
            manifest_path = Path(
                os.environ["EASYICU_RESOLVED_INPUTS_JSON"]
            ).resolve()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            inputs = manifest.get("inputs")
            if not isinstance(inputs, dict) or input_key not in inputs:
                raise RuntimeError(
                    "Missing exact typed cohort binding: %s" % input_key
                )
            binding = inputs[input_key]
            relative_path = binding.get("relative_path")
            expected_sha256 = binding.get("sha256")
            contract = binding.get("product_contract")
            if (
                not isinstance(relative_path, str)
                or not relative_path
                or not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or not isinstance(contract, dict)
            ):
                raise RuntimeError("Typed cohort binding is incomplete")
            cohort_path = (run_dir / relative_path).resolve()
            try:
                cohort_path.relative_to(run_dir)
            except ValueError as exc:
                raise RuntimeError(
                    "Typed cohort binding escapes EASYICU_RUN_DIR"
                ) from exc
            if not cohort_path.is_file():
                raise RuntimeError("Typed cohort binding does not name a file")
            if sha256_file(cohort_path) != expected_sha256:
                raise RuntimeError("Typed cohort digest verification failed")
            columns = contract.get("columns")
            row_count = contract.get("row_count")
            if (
                not isinstance(columns, list)
                or not columns
                or not all(isinstance(value, str) and value for value in columns)
                or len(set(columns)) != len(columns)
                or not isinstance(row_count, int)
                or isinstance(row_count, bool)
                or row_count < 0
            ):
                raise RuntimeError(
                    "Typed cohort product_contract is incomplete"
                )
            suffix = cohort_path.suffix.lower()
            if suffix in {{".parquet", ".pq"}}:
                frame = pd.read_parquet(cohort_path)
            elif suffix == ".csv":
                frame = pd.read_csv(cohort_path)
            elif suffix == ".tsv":
                frame = pd.read_csv(cohort_path, sep="\\t")
            else:
                raise RuntimeError("Typed cohort table format is unsupported")
            if list(frame.columns) != columns:
                raise RuntimeError(
                    "Typed cohort columns do not match product_contract"
                )
            if len(frame) != row_count:
                raise RuntimeError(
                    "Typed cohort row count does not match product_contract"
                )
            return frame, cohort_path

        if typed_cohort_input is None:
            cohort_path = Path(os.environ["COHORT_PARQUET"])
            frame = pd.read_parquet(cohort_path)
        else:
            frame, cohort_path = load_typed_cohort(typed_cohort_input)

        table_one = build_grouped_table_one(frame, table_one_spec)
        table_path = out_dir / "table_one.csv"
        table_one.to_csv(table_path, index=False)

        measurement_checks = [
            measurement_provenance_receipt(
                frame,
                measured_column=measured_column,
                count_column=count_column,
            )
            for measured_column, count_column in measurement_pairs
        ]

        summary = {{
            "step_id": {step.step_id!r},
            "status": "ok",
            "analysis_family": "grouped_table_one",
            "interpretation_class": "descriptive_baseline_characteristics",
            "method": "Planner-declared grouped Table 1 executed by the host SDK.",
            "cohort_path": str(cohort_path),
            "cohort_input_key": typed_cohort_input or "COHORT_PARQUET",
            "cohort_n": int(len(frame)),
            "group_by": str(table_one_spec["group_by"]),
            "group_levels": list(table_one_spec["group_levels"]),
            "variables": [item["name"] for item in table_one_spec["variables"]],
            "table_one_contract_sha256": table_one_spec_sha256(table_one_spec),
            "table_one_result_rows": int(len(table_one)),
            "table_one_path": table_path.name,
            "adjusted_effect": None,
            "output_files": {{"table:table_one": table_path.name}},
            "measurement_provenance_audit": {{
                "source": "COHORT_PARQUET",
                "checks": measurement_checks,
            }},
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
