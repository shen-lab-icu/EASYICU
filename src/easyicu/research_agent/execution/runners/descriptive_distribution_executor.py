"""Deterministic grouped summaries for one Planner-declared continuous variable.

This owner performs no variable selection and invents no numeric groups.  A
complete Planner contract supplies one typed cohort, one categorical grouping
column, and one continuous value column in that order.  The structured research
context supplies the closed group levels and the value unit.  The executor then
mechanically emits overall and partition rows with an explicit ``row_role`` so
downstream consumers cannot double-count the overall denominator.
"""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...schema import AnalysisStep
from .plausibility_receipt import render_standard_plausibility_receipt_code
from .typed_input_binding import sole_typed_cohort_input

__all__ = [
    "DESCRIPTIVE_DISTRIBUTION_ANALYSIS_KIND",
    "descriptive_distribution_executor_code",
    "descriptive_distribution_executor_owns_step",
    "run_descriptive_distribution_summary",
]


DESCRIPTIVE_DISTRIBUTION_ANALYSIS_KIND = "grouped_descriptive_distribution"
_SUPPORTED_METHODS = frozenset(
    {"descriptive_distribution", "descriptive_distribution_summary"}
)
_OUTPUT_COLUMNS = (
    "variable",
    "group",
    "row_role",
    "group_n",
    "n_nonmissing",
    "missing_n",
    "missing_pct",
    "median",
    "q25",
    "q75",
    "mean",
    "sd",
    "unit",
)


def _declared_columns(step: AnalysisStep) -> tuple[str, ...]:
    return tuple(
        str(value).strip()
        for value in step.inputs
        if str(value).strip() and ":" not in str(value).strip()
    )


def descriptive_distribution_executor_owns_step(step: AnalysisStep) -> bool:
    """Own the exact primary or auxiliary grouped-distribution contract."""

    columns = _declared_columns(step)
    return bool(
        str(step.method or "").strip().casefold() in _SUPPORTED_METHODS
        and str(step.planned_analysis_role or "").strip().casefold()
        in {"primary", "secondary", "auxiliary"}
        and list(step.expected_outputs or []) == ["table:distribution_prevalence"]
        and len(columns) == 2
        and len(set(columns)) == 2
        and sole_typed_cohort_input(step) is not None
        and not step.model_requirements
        and step.table_one_spec is None
        and step.trajectory_stability_spec is None
        and step.exposure_outcome_distribution_spec is None
        and step.cohort_definition_spec is None
        and step.measurement_audit_spec is None
        and step.robustness_replay_spec is None
    )


def descriptive_distribution_executor_code(
    step: AnalysisStep,
    *,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> str:
    """Render the attributable entrypoint for the declared grouping/value pair."""

    if not descriptive_distribution_executor_owns_step(step):
        raise ValueError("step is not owned by the descriptive-distribution executor")
    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    grouping_variable, value_variable = _declared_columns(step)
    typed_cohort_input = sole_typed_cohort_input(step)
    if typed_cohort_input is None:  # pragma: no cover - guarded by ownership
        raise ValueError("descriptive distribution requires a typed cohort")
    receipt_code = (
        render_standard_plausibility_receipt_code(
            plausibility_scope,
            frame_name="frame",
        )
        if plausibility_scope is not None and plausibility_scope.expected_columns
        else ""
    )
    prologue = textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.descriptive_distribution_executor import (
            run_descriptive_distribution_summary,
        )
        from easyicu.research_agent.execution.runners.typed_input_binding import (
            load_step_cohort_frame,
        )

        typed_cohort_input = {typed_cohort_input!r}
        frame, cohort_path = load_step_cohort_frame(
            typed_cohort_input=typed_cohort_input,
        )
        """
    ).strip()
    if receipt_code:
        prologue += "\n\n" + receipt_code.strip()
    prologue += "\n\n" + textwrap.dedent(
        f"""
        summary = run_descriptive_distribution_summary(
            frame=frame,
            grouping_variable={grouping_variable!r},
            value_variable={value_variable!r},
            typed_cohort_input=typed_cohort_input,
            source_cohort=cohort_path,
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
        )
        """
    ).strip()
    if receipt_code:
        prologue += '\nsummary["plausibility_audit"] = plausibility_audit'
    prologue += "\n" + textwrap.dedent(
        """
        out_dir = Path(os.environ["STEP_OUT_DIR"])
        (out_dir / "step_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        """
    ).strip()
    return prologue


def _context_metadata(run_dir: Path) -> dict[str, Mapping[str, Any]]:
    payload = json.loads((Path(run_dir) / "research_context.json").read_text("utf-8"))
    variables = payload.get("variables") if isinstance(payload, Mapping) else None
    if not isinstance(variables, list):
        raise RuntimeError("ResearchContext has no structured variable metadata")
    return {
        str(item.get("name")): item
        for item in variables
        if isinstance(item, Mapping) and str(item.get("name") or "")
    }


def _closed_levels(metadata: Mapping[str, Any]) -> tuple[Any, ...]:
    domain = metadata.get("observed_domain")
    levels = domain.get("levels") if isinstance(domain, Mapping) else None
    if not isinstance(levels, list) or not levels:
        raise RuntimeError("Grouping variable has no closed categorical levels")
    if any(
        value is None or isinstance(value, (dict, list, tuple, set))
        for value in levels
    ):
        raise RuntimeError("Grouping variable categorical levels are invalid")
    if len({str(value) for value in levels}) != len(levels):
        raise RuntimeError("Grouping variable categorical levels are not unique")
    return tuple(levels)


def _finite(value: Any, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Grouped distribution produced non-finite {field}")
    return number


def _summary_row(
    *,
    frame: pd.DataFrame,
    mask: pd.Series,
    variable: str,
    group: str,
    row_role: str,
    unit: str,
) -> dict[str, Any]:
    group_frame = frame.loc[mask]
    numeric = pd.to_numeric(group_frame[variable], errors="coerce")
    source_nonmissing = int(group_frame[variable].notna().sum())
    if int(numeric.notna().sum()) != source_nonmissing:
        raise RuntimeError("Continuous value column contains non-numeric observations")
    observed = numeric.dropna()
    group_n = int(mask.sum())
    missing_n = group_n - int(len(observed))
    if group_n < 1 or len(observed) < 2:
        raise RuntimeError("Every declared group needs at least two observed values")
    return {
        "variable": variable,
        "group": group,
        "row_role": row_role,
        "group_n": group_n,
        "n_nonmissing": int(len(observed)),
        "missing_n": missing_n,
        "missing_pct": 100.0 * missing_n / group_n,
        "median": _finite(observed.median(), field="median"),
        "q25": _finite(observed.quantile(0.25), field="q25"),
        "q75": _finite(observed.quantile(0.75), field="q75"),
        "mean": _finite(observed.mean(), field="mean"),
        "sd": _finite(observed.std(ddof=1), field="sd"),
        "unit": unit,
    }


def run_descriptive_distribution_summary(
    *,
    frame: pd.DataFrame,
    grouping_variable: str,
    value_variable: str,
    typed_cohort_input: str,
    source_cohort: Path,
    out_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """Compute exact overall and closed-level summaries from one bound cohort."""

    if grouping_variable == value_variable:
        raise RuntimeError("Grouping and value variables must differ")
    missing = sorted({grouping_variable, value_variable} - set(frame.columns))
    if missing:
        raise RuntimeError("Declared distribution columns are absent: " + ", ".join(missing))
    if frame.empty:
        raise RuntimeError("Descriptive distribution cohort is empty")

    metadata = _context_metadata(Path(run_dir))
    if grouping_variable not in metadata or value_variable not in metadata:
        raise RuntimeError("Declared distribution columns lack structured metadata")
    levels = _closed_levels(metadata[grouping_variable])
    grouping = frame[grouping_variable]
    invalid = grouping.notna() & ~grouping.isin(levels)
    if bool(invalid.any()):
        raise RuntimeError("Grouping values fall outside the closed metadata domain")
    unit = str(metadata[value_variable].get("unit") or "")

    rows = [
        _summary_row(
            frame=frame,
            mask=pd.Series(True, index=frame.index),
            variable=value_variable,
            group="Overall",
            row_role="overall",
            unit=unit,
        )
    ]
    partition_masks: list[pd.Series] = []
    for level in levels:
        mask = grouping.eq(level).fillna(False)
        partition_masks.append(mask)
        rows.append(
            _summary_row(
                frame=frame,
                mask=mask,
                variable=value_variable,
                group=str(level),
                row_role="exposure_level",
                unit=unit,
            )
        )
    if bool(grouping.isna().any()):
        missing_mask = grouping.isna()
        partition_masks.append(missing_mask)
        rows.append(
            _summary_row(
                frame=frame,
                mask=missing_mask,
                variable=value_variable,
                group="Missing",
                row_role="exposure_level",
                unit=unit,
            )
        )
    partition_total = sum(int(mask.sum()) for mask in partition_masks)
    if partition_total != len(frame):
        raise RuntimeError("Closed grouping levels do not partition the cohort")

    table = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "distribution_prevalence.csv"
    table.to_csv(table_path, index=False)
    return {
        "status": "ok",
        "analysis_status": "ok",
        "analysis_family": "descriptive",
        "interpretation_class": "grouped_descriptive_distribution",
        "deterministic_standard_analysis": DESCRIPTIVE_DISTRIBUTION_ANALYSIS_KIND,
        "grouping_variable": grouping_variable,
        "value_variable": value_variable,
        "group_levels": [str(value) for value in levels],
        "typed_cohort_input": typed_cohort_input,
        "source_cohort": Path(source_cohort).name,
        "source_row_count_reconciliation": {
            "source_rows": int(len(frame)),
            "analyzed_rows": int(len(frame)),
            "filtering_performed": False,
        },
        "adjusted_effect": None,
        "output_files": {"table:distribution_prevalence": table_path.name},
    }
