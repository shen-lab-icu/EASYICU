"""Deterministic executor for one closed descriptive cohort-summary contract.

The Planner has already fixed the cohort and the columns to describe.  This
executor performs only mechanical summaries: cohort size, missingness, declared
categorical-level counts, and standard numeric distribution statistics.  It
does not choose variables, create groups from observed numeric values, fit a
model, or report an effect estimate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from ...schema import AnalysisStep

__all__ = [
    "cohort_summary_executor_code",
    "cohort_summary_executor_owns_step",
    "run_cohort_summary_from_env",
]


def _typed_cohort_input(step: AnalysisStep) -> str | None:
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


def _declared_columns(step: AnalysisStep) -> tuple[str, ...]:
    return tuple(
        str(value).strip()
        for value in step.inputs
        if str(value).strip() and ":" not in str(value).strip()
    )


def cohort_summary_executor_owns_step(step: AnalysisStep) -> bool:
    """Own only a complete, auxiliary, count/descriptive-only contract."""

    columns = _declared_columns(step)
    return bool(
        str(step.method or "").strip().casefold() == "descriptive_cohort_summary"
        and str(step.planned_analysis_role or "").strip().casefold() == "auxiliary"
        and list(step.expected_outputs or []) == ["table:cohort_summary"]
        and columns
        and len(columns) == len(set(columns))
        and _typed_cohort_input(step) != ""
        and not step.model_requirements
        and step.table_one_spec is None
        and step.trajectory_stability_spec is None
    )


def cohort_summary_executor_code(step: AnalysisStep) -> str:
    """Return a small sandbox entrypoint with the exact Planner column scope."""

    if not cohort_summary_executor_owns_step(step):
        raise ValueError("The step is not owned by the cohort-summary executor")
    return textwrap.dedent(f"""
        from easyicu.research_agent.execution.runners.cohort_summary_executor import (
            run_cohort_summary_from_env,
        )

        run_cohort_summary_from_env(
            declared_columns={_declared_columns(step)!r},
            typed_cohort_input={_typed_cohort_input(step)!r},
        )
        """).strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained_regular_file(path: Path, root: Path) -> Optional[Path]:
    root = root.resolve()
    candidate = Path(path)
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    cursor = candidate
    while cursor != root:
        if cursor.is_symlink():
            return None
        parent = cursor.parent
        if parent == cursor:
            return None
        cursor = parent
    if not candidate.is_file():
        return None
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    return resolved


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise RuntimeError("Typed cohort table format is unsupported")


def _load_typed_cohort(
    *,
    input_key: str,
    run_dir: Path,
    resolved_inputs_path: Path,
) -> tuple[pd.DataFrame, Path]:
    try:
        payload = json.loads(resolved_inputs_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("Resolved input manifest is unreadable") from exc
    inputs = payload.get("inputs") if isinstance(payload, dict) else None
    binding = inputs.get(input_key) if isinstance(inputs, dict) else None
    if not isinstance(binding, dict):
        raise RuntimeError(f"Missing exact typed cohort binding: {input_key}")
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
    candidate = run_dir / relative_path
    cohort_path = _contained_regular_file(candidate, run_dir)
    if cohort_path is None:
        raise RuntimeError("Typed cohort binding is not a contained regular file")
    if _sha256_file(cohort_path) != expected_sha256:
        raise RuntimeError("Typed cohort digest verification failed")
    columns = contract.get("columns")
    row_count = contract.get("row_count")
    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(value, str) and value for value in columns)
        or len(columns) != len(set(columns))
        or not isinstance(row_count, int)
        or isinstance(row_count, bool)
        or row_count < 0
    ):
        raise RuntimeError("Typed cohort product_contract is incomplete")
    frame = _read_frame(cohort_path)
    if list(frame.columns) != columns:
        raise RuntimeError("Typed cohort columns do not match product_contract")
    if len(frame) != row_count:
        raise RuntimeError("Typed cohort row count does not match product_contract")
    return frame, cohort_path


def _load_context(run_dir: Path) -> Dict[str, Any]:
    context_path = Path(
        os.environ.get("EASYICU_RESEARCH_CONTEXT")
        or run_dir / "research_context.json"
    )
    try:
        payload = json.loads(context_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("Research context is unreadable") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Research context is not an object")
    return payload


def _variable_metadata(context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    by_name: Dict[str, Dict[str, Any]] = {}
    for item in context.get("variables") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name or name in by_name:
            continue
        by_name[name] = item
    return by_name


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _row(
    *,
    variable: str,
    statistic: str,
    value: Any = None,
    level: Any = None,
    numerator: Optional[int] = None,
    denominator: Optional[int] = None,
    percentage: Optional[float] = None,
    unit: Any = None,
) -> Dict[str, Any]:
    return {
        "variable": variable,
        "statistic": statistic,
        "level": level,
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "percentage": percentage,
        "unit": unit,
        "source_column": variable if variable != "__cohort__" else None,
    }


def run_cohort_summary_from_env(
    *,
    declared_columns: Sequence[str],
    typed_cohort_input: str | None,
) -> Dict[str, Any]:
    """Execute exact descriptive summaries from the standard runner environment."""

    columns = tuple(str(value).strip() for value in declared_columns)
    if not columns or any(not value for value in columns) or len(columns) != len(
        set(columns)
    ):
        raise RuntimeError("Declared cohort-summary columns are not closed and unique")

    out_dir = Path(os.environ["STEP_OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(
        os.environ.get("EASYICU_RUN_DIR") or out_dir.parents[2]
    ).resolve()
    if typed_cohort_input is None:
        cohort_path = Path(os.environ["COHORT_PARQUET"]).resolve()
        frame = _read_frame(cohort_path)
    else:
        resolved_inputs_path = Path(
            os.environ["EASYICU_RESOLVED_INPUTS_JSON"]
        ).resolve()
        frame, cohort_path = _load_typed_cohort(
            input_key=typed_cohort_input,
            run_dir=run_dir,
            resolved_inputs_path=resolved_inputs_path,
        )
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise RuntimeError(
            "Declared cohort-summary columns are absent: "
            + ", ".join(missing_columns)
        )

    context = _load_context(run_dir)
    metadata_by_name = _variable_metadata(context)
    missing_metadata = [column for column in columns if column not in metadata_by_name]
    if missing_metadata:
        raise RuntimeError(
            "Declared cohort-summary columns lack structured metadata: "
            + ", ".join(missing_metadata)
        )

    cohort_n = int(len(frame))
    rows = [
        _row(
            variable="__cohort__",
            statistic="cohort_n",
            value=cohort_n,
            numerator=cohort_n,
            denominator=cohort_n,
            percentage=100.0 if cohort_n else None,
        )
    ]
    for column in columns:
        values = frame[column]
        metadata = metadata_by_name[column]
        unit = metadata.get("unit")
        nonmissing_n = int(values.notna().sum())
        missing_n = cohort_n - nonmissing_n
        rows.append(
            _row(
                variable=column,
                statistic="nonmissing_n",
                value=nonmissing_n,
                numerator=nonmissing_n,
                denominator=cohort_n,
                percentage=(100.0 * nonmissing_n / cohort_n if cohort_n else None),
                unit=unit,
            )
        )
        rows.append(
            _row(
                variable=column,
                statistic="missing_n",
                value=missing_n,
                numerator=missing_n,
                denominator=cohort_n,
                percentage=(100.0 * missing_n / cohort_n if cohort_n else None),
                unit=unit,
            )
        )

        domain = metadata.get("observed_domain")
        levels = domain.get("levels") if isinstance(domain, dict) else None
        if isinstance(levels, list) and levels:
            if any(
                isinstance(level, (dict, list, tuple, set)) or level is None
                for level in levels
            ):
                raise RuntimeError(
                    f"Structured categorical levels are invalid for {column}"
                )
            for level in levels:
                count = int(values.eq(level).fillna(False).sum())
                rows.append(
                    _row(
                        variable=column,
                        statistic="level_count",
                        level=level,
                        value=count,
                        numerator=count,
                        denominator=nonmissing_n,
                        percentage=(
                            100.0 * count / nonmissing_n if nonmissing_n else None
                        ),
                        unit=unit,
                    )
                )
            continue

        numeric = pd.to_numeric(values, errors="coerce")
        if int(numeric.notna().sum()) != nonmissing_n:
            raise RuntimeError(
                f"Non-numeric {column} lacks closed categorical levels in metadata"
            )
        observed = numeric.dropna()
        statistics = {
            "mean": observed.mean() if not observed.empty else None,
            "std": observed.std(ddof=1) if len(observed) > 1 else None,
            "median": observed.median() if not observed.empty else None,
            "q1": observed.quantile(0.25) if not observed.empty else None,
            "q3": observed.quantile(0.75) if not observed.empty else None,
            "minimum": observed.min() if not observed.empty else None,
            "maximum": observed.max() if not observed.empty else None,
        }
        for statistic, raw_value in statistics.items():
            rows.append(
                _row(
                    variable=column,
                    statistic=statistic,
                    value=_finite_number(raw_value),
                    denominator=nonmissing_n,
                    unit=unit,
                )
            )

    table_path = out_dir / "cohort_summary.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    summary = {
        "status": "ok",
        "analysis_family": "descriptive",
        "interpretation_class": "cohort_summary",
        "cohort_n": cohort_n,
        "declared_columns": list(columns),
        "typed_cohort_input": typed_cohort_input,
        "source_cohort": cohort_path.name,
        "source_row_count_reconciliation": {
            "source_rows": cohort_n,
            "analyzed_rows": cohort_n,
            "filtering_performed": False,
        },
        "adjusted_effect": None,
        "output_files": {"table:cohort_summary": table_path.name},
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return summary
