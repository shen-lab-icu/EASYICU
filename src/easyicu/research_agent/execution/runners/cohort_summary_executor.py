"""Deterministic executor for one closed descriptive cohort-summary contract.

The Planner has already fixed the cohort and the columns to describe.  This
executor performs only mechanical summaries: cohort size, missingness, declared
categorical-level counts, and standard numeric distribution statistics.  It
does not choose variables, create groups from observed numeric values, fit a
model, or report an effect estimate.
"""

from __future__ import annotations

import json
import math
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...schema import AnalysisStep
from .plausibility_receipt import render_standard_plausibility_receipt_code
from .typed_cohort_binding import load_step_cohort_frame, run_dir_from_env

__all__ = [
    "cohort_summary_executor_code",
    "cohort_summary_executor_owns_step",
    "load_cohort_summary_frame",
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
    if (
        separator
        and product
        and (kind == "cohort" or input_key == "artifact:analysis_cohort")
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
        str(step.method or "").strip().casefold()
        in {"descriptive_cohort_summary", "descriptive"}
        and str(step.planned_analysis_role or "").strip().casefold() == "auxiliary"
        and list(step.expected_outputs or []) == ["table:cohort_summary"]
        and columns
        and len(columns) == len(set(columns))
        and _typed_cohort_input(step) != ""
        and not step.model_requirements
        and step.table_one_spec is None
        and step.trajectory_stability_spec is None
    )


def cohort_summary_executor_code(
    step: AnalysisStep,
    *,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> str:
    """Return a small sandbox entrypoint with the exact Planner column scope.

    When the step owes a flag-only plausibility receipt the comparisons are
    rendered here rather than performed inside the imported host function.
    That is not decoration: the pre-execution obligation gate proves the
    obligation by locating a comparison against a bound read from the sealed
    contract in the source that will actually run, and a source that only
    calls a helper is ``not_attributable`` to it.  This executor used to
    decline every receipt-bearing step for exactly that reason, sending a step
    the host can compute exactly to the stochastic Coder instead.
    """

    if not cohort_summary_executor_owns_step(step):
        raise ValueError("The step is not owned by the cohort-summary executor")
    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    expected_columns = (
        tuple(plausibility_scope.expected_columns)
        if plausibility_scope is not None
        else ()
    )
    receipt_code = (
        render_standard_plausibility_receipt_code(
            plausibility_scope,
            frame_name="frame",
        )
        if plausibility_scope is not None
        else ""
    )
    if not expected_columns:
        return textwrap.dedent(
            f"""
            from easyicu.research_agent.execution.runners.cohort_summary_executor import (
                run_cohort_summary_from_env,
            )

            run_cohort_summary_from_env(
                declared_columns={_declared_columns(step)!r},
                typed_cohort_input={_typed_cohort_input(step)!r},
            )
            """
        ).strip()

    return textwrap.dedent(
        f"""
        import hashlib
        import json
        import os
        from pathlib import Path

        import pandas as pd

        from easyicu.research_agent.execution.runners.cohort_summary_executor import (
            load_cohort_summary_frame,
            run_cohort_summary_from_env,
        )

        declared_columns = {_declared_columns(step)!r}
        typed_cohort_input = {_typed_cohort_input(step)!r}

        frame, cohort_path = load_cohort_summary_frame(
            typed_cohort_input=typed_cohort_input,
        )

        {textwrap.indent(receipt_code, " " * 8).strip()}

        summary = run_cohort_summary_from_env(
            declared_columns=declared_columns,
            typed_cohort_input=typed_cohort_input,
            frame=frame,
            cohort_path=cohort_path,
            plausibility_expected_columns=plausibility_expected_columns,
            plausibility_audit=plausibility_audit,
            emit_step_summary=False,
        )
        summary["plausibility_audit"] = plausibility_audit
        out_dir = Path(os.environ["STEP_OUT_DIR"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "step_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        """
    ).strip()


def load_cohort_summary_frame(
    *,
    typed_cohort_input: str | None,
) -> tuple[pd.DataFrame, Path]:
    """Load the exact bound cohort once, for both the receipt and the summary.

    Kept as this module's published name; the rule itself is owned by
    ``typed_cohort_binding`` so every executor that reads a bound cohort
    verifies it identically.
    """

    return load_step_cohort_frame(typed_cohort_input=typed_cohort_input)


def _verified_plausibility_audit(
    audit: Optional[Dict[str, Any]],
    *,
    expected_columns: Sequence[str],
) -> Optional[Dict[str, Any]]:
    """Accept the rendered receipt only if it covers the exact sealed scope.

    The caller is host-rendered source, but the check stays here so the
    invariant travels with the summary this function writes rather than
    depending on every future caller having rendered the block correctly.
    """

    expected = tuple(str(value) for value in expected_columns)
    if not expected:
        if audit:
            raise RuntimeError(
                "A plausibility receipt was supplied for a step with no "
                "flag-only scope"
            )
        return None
    if not isinstance(audit, dict) or set(audit) != set(expected):
        raise RuntimeError("Plausibility receipt does not cover the exact sealed scope")
    for column, record in audit.items():
        if not isinstance(record, dict):
            raise RuntimeError(f"Plausibility receipt for {column} is untyped")
        below = record.get("below_minimum_n")
        above = record.get("above_maximum_n")
        total = record.get("out_of_range_n")
        compared = record.get("compared_n")
        counts = (below, above, total, compared)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts
        ):
            raise RuntimeError(
                f"Plausibility receipt for {column} lacks non-negative counts"
            )
        if total != below + above:
            raise RuntimeError(
                f"Plausibility receipt for {column} does not partition its total"
            )
        if total > compared:
            raise RuntimeError(
                f"Plausibility receipt for {column} flags more values than it "
                "compared"
            )
    return dict(audit)


def _load_context(run_dir: Path) -> Dict[str, Any]:
    context_path = Path(
        os.environ.get("EASYICU_RESEARCH_CONTEXT") or run_dir / "research_context.json"
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
    frame: Optional[pd.DataFrame] = None,
    cohort_path: Optional[Path] = None,
    plausibility_expected_columns: Sequence[str] = (),
    plausibility_audit: Optional[Dict[str, Any]] = None,
    emit_step_summary: bool = True,
) -> Dict[str, Any]:
    """Execute exact descriptive summaries from the standard runner environment.

    ``frame``/``cohort_path`` let the rendered entrypoint hand over the cohort
    it already loaded to compute the plausibility receipt, so the receipt and
    the summary describe the same bytes and the cohort is read once.

    ``emit_step_summary=False`` returns the summary without writing it, for the
    receipt-bearing entrypoint that must perform the write in its own source:
    the obligation gate proves delivery by locating the write of the receipt
    key into the host's ``step_summary.json``, and a write hidden inside an
    imported function is not something it can attribute.  Correctness of the
    receipt is still decided here -- ``_verified_plausibility_audit`` raises
    before any summary exists -- so moving the write does not move the
    authority.
    """

    columns = tuple(str(value).strip() for value in declared_columns)
    if (
        not columns
        or any(not value for value in columns)
        or len(columns) != len(set(columns))
    ):
        raise RuntimeError("Declared cohort-summary columns are not closed and unique")

    out_dir = Path(os.environ["STEP_OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = run_dir_from_env()
    verified_audit = _verified_plausibility_audit(
        plausibility_audit,
        expected_columns=plausibility_expected_columns,
    )
    if frame is None or cohort_path is None:
        frame, cohort_path = load_cohort_summary_frame(
            typed_cohort_input=typed_cohort_input,
        )
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise RuntimeError(
            "Declared cohort-summary columns are absent: " + ", ".join(missing_columns)
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
    if verified_audit is not None:
        summary["plausibility_audit"] = verified_audit
    if not emit_step_summary:
        return summary
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return summary
