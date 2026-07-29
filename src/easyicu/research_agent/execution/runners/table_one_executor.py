"""Deterministic executor for one Planner-owned grouped Table 1.

The Planner owns the grouping variable, closed levels, row variables, summary
families, and comparison tests through ``AnalysisStep.table_one_spec``.  This
module only turns that closed declaration into a small sandbox script; it never
chooses a cohort, variable, level, summary, or statistical test.
"""

from __future__ import annotations

import textwrap

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...authority.table_one_binding import table_one_execution_spec
from ...icu_rules import companion_count_column_for_measured
from ...schema import AnalysisStep, TABLE_ONE_CLOSED_OUTPUTS
from .plausibility_receipt import render_standard_plausibility_receipt_code
from .typed_input_binding import sole_typed_cohort_input as _typed_cohort_input

__all__ = ["table_one_executor_code", "table_one_executor_owns_step"]


def table_one_executor_owns_step(step: AnalysisStep) -> bool:
    """Return whether the exact output contract is fully host-executable."""

    outputs = {str(value or "").strip() for value in step.expected_outputs}
    typed_cohort_input = _typed_cohort_input(step)
    return bool(
        step.table_one_spec is not None
        and "table:table_one" in outputs
        and not any(value.startswith("figure:") for value in outputs)
        and outputs.issubset(TABLE_ONE_CLOSED_OUTPUTS)
        # No typed input means COHORT_PARQUET is the row authority.  Otherwise
        # the executor supports exactly one explicitly cohort-scoped product
        # (plus the historical artifact:analysis_cohort spelling) and loads
        # that digest-bound table rather than silently analysing another frame.
        and typed_cohort_input != ""
    )


def table_one_executor_code(
    step: AnalysisStep,
    *,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> str:
    """Return sandbox code for the exact Planner-owned Table 1 declaration."""

    if not table_one_executor_owns_step(step):
        raise ValueError("The step is not owned by the grouped Table 1 executor")
    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    specification_model = table_one_execution_spec(step)
    assert specification_model is not None
    specification = specification_model.model_dump(mode="python")
    typed_cohort_input = _typed_cohort_input(step)
    declared_outputs = {str(value or "").strip() for value in step.expected_outputs}
    emit_cohort_flow = "table:cohort_flow" in declared_outputs
    emit_source_reconciliation = (
        "log:source_row_count_reconciliation" in declared_outputs
    )
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
    plausibility_code = (
        render_standard_plausibility_receipt_code(
            plausibility_scope,
            frame_name="frame",
        )
        if plausibility_scope is not None
        else ""
    )
    plausibility_summary_entry = (
        '"plausibility_audit": plausibility_audit,'
        if plausibility_scope is not None and plausibility_scope.expected_columns
        else ""
    )
    rendered = textwrap.dedent(
        f"""
        import hashlib
        import json
        import os
        from pathlib import Path

        import pandas as pd
        import pyarrow.parquet as pq

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
        emit_cohort_flow = {emit_cohort_flow!r}
        emit_source_reconciliation = {emit_source_reconciliation!r}
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

        __EASYICU_STANDARD_PLAUSIBILITY_RECEIPT__

        table_one = build_grouped_table_one(frame, table_one_spec)
        table_path = out_dir / "table_one.csv"
        table_one.to_csv(table_path, index=False)

        raw_cohort_path = Path(os.environ["COHORT_PARQUET"])
        source_row_count = int(pq.ParquetFile(raw_cohort_path).metadata.num_rows)
        analyzed_row_count = int(len(frame))
        output_files = {{"table:table_one": table_path.name}}

        if emit_cohort_flow:
            stages = [
                {{
                    "stage": "COHORT_PARQUET rows",
                    "count": source_row_count,
                    "denominator": source_row_count,
                    "percentage": 100.0 if source_row_count else None,
                    "interpretation": "Host-bound source frame",
                }}
            ]
            if typed_cohort_input is not None:
                stages.append(
                    {{
                        "stage": "Typed cohort rows",
                        "count": analyzed_row_count,
                        "denominator": source_row_count,
                        "percentage": (
                            100.0 * analyzed_row_count / source_row_count
                            if source_row_count
                            else None
                        ),
                        "interpretation": (
                            "Digest-verified typed cohort membership; no "
                            "eligibility rule was added by the Table 1 executor"
                        ),
                    }}
                )
            stages.append(
                {{
                    "stage": "Table 1 analyzed rows",
                    "count": analyzed_row_count,
                    "denominator": analyzed_row_count,
                    "percentage": 100.0 if analyzed_row_count else None,
                    "interpretation": (
                        "All rows in the bound Table 1 frame; no rows were "
                        "added or removed by the executor"
                    ),
                }}
            )
            flow_path = out_dir / "cohort_flow.csv"
            pd.DataFrame(stages).to_csv(flow_path, index=False)
            output_files["table:cohort_flow"] = flow_path.name

        if emit_source_reconciliation:
            reconciliation_path = out_dir / "source_row_count_reconciliation.json"
            reconciliation = {{
                "schema_version": "easyicu.source_row_count_reconciliation/1",
                "source": "COHORT_PARQUET",
                "source_rows": source_row_count,
                "typed_cohort_input": typed_cohort_input,
                "typed_cohort_rows": analyzed_row_count,
                "final_analyzed_rows": analyzed_row_count,
                "typed_minus_source": analyzed_row_count - source_row_count,
                "final_minus_typed": 0,
                "table_one_filtering_performed": False,
                "denominator_policy": (
                    "All rows in the digest-verified typed cohort"
                    if typed_cohort_input is not None
                    else "All rows in COHORT_PARQUET"
                ),
            }}
            reconciliation_path.write_text(
                json.dumps(reconciliation, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            output_files[
                "log:source_row_count_reconciliation"
            ] = reconciliation_path.name

        measurement_checks = [
            measurement_provenance_receipt(
                frame,
                measured_column=measured_column,
                count_column=count_column,
            )
            for measured_column, count_column in measurement_pairs
        ]

        summary = {{
            {plausibility_summary_entry}
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
            "source_row_count_reconciliation": {{
                "source_rows": source_row_count,
                "analyzed_rows": analyzed_row_count,
                "table_one_filtering_performed": False,
            }},
            "output_files": output_files,
            "measurement_provenance_audit": {{
                "source": "COHORT_PARQUET",
                "checks": measurement_checks,
            }},
            "notes": [
                "Overall and comparison groups use the exact Planner-declared levels.",
                "Missing n (%) uses each displayed group's full denominator.",
                "P values use the exact Planner-declared test family without adjustment because this is a descriptive baseline table.",
                "For two-group tables, standardized differences use comparison minus reference over the equal-weight pooled standard deviation.",
                "No row, variable, level, or test was selected by the executor.",
            ],
        }}
        (out_dir / "step_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(json.dumps({{"grouped_table_one": "ok", "cohort_n": len(frame)}}))
        """
    )
    return rendered.replace(
        "__EASYICU_STANDARD_PLAUSIBILITY_RECEIPT__",
        plausibility_code,
    ).strip()
