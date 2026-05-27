"""[Layer 3: Safe Analytical Runtime] Deterministic estimator adapter.

This module turns a typed robustness specification into a reproducible model
fit. It is deliberately narrow: statsmodels logistic and linear estimators,
simple deterministic missing-data policies, and row-level failure capture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from .cohort_schema import CohortDefinition, build_cohort
from .missing import apply_missing_strategy
from .robustness_panel import PRIMARY_SPEC_ID, RobustnessPanelRow, RobustnessSpec


EstimatorKind = Literal["logistic", "linear", "cox", "glm_poisson"]


@dataclass(frozen=True)
class EstimatorResult:
    point_estimate: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    se: Optional[float]
    n: int
    converged: bool
    notes: str = ""


def fit_estimator(*, cohort: Any, X: Any, y: Any, kind: EstimatorKind | str) -> EstimatorResult:
    """Fit a supported estimator and capture failures as non-converged results."""

    cohort_note = _cohort_trace_note(cohort)
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        import statsmodels.api as sm  # type: ignore
    except Exception as exc:  # pragma: no cover - project dependency guard
        return EstimatorResult(
            None,
            None,
            None,
            None,
            0,
            False,
            _join_notes(f"missing dependency: {exc}", cohort_note),
        )

    kind = str(kind or "logistic")
    if kind not in {"logistic", "linear"}:
        raise NotImplementedError(f"estimator kind {kind!r} is not implemented")

    x_df = pd.DataFrame(X).copy()
    y_series = pd.Series(y).copy()
    if len(x_df) != len(y_series):
        return EstimatorResult(
            None,
            None,
            None,
            None,
            0,
            False,
            _join_notes("X/y length mismatch", cohort_note),
        )
    combined = pd.concat([x_df.reset_index(drop=True), y_series.rename("__y__")], axis=1)
    combined = combined.dropna()
    n = int(len(combined))
    if n == 0:
        return EstimatorResult(
            None,
            None,
            None,
            None,
            0,
            False,
            _join_notes("no complete rows", cohort_note),
        )
    x_df = combined.drop(columns=["__y__"])
    y_series = combined["__y__"]
    if x_df.shape[1] == 0:
        return EstimatorResult(
            None,
            None,
            None,
            None,
            n,
            False,
            _join_notes("no predictor columns", cohort_note),
        )
    if n <= x_df.shape[1] + 2:
        return EstimatorResult(
            None,
            None,
            None,
            None,
            n,
            False,
            _join_notes("sample size too small for deterministic estimator", cohort_note),
        )

    try:
        x_const = sm.add_constant(x_df.astype(float), has_constant="add")
        coefficient_name = next(col for col in x_const.columns if col != "const")
        if kind == "linear":
            result = sm.OLS(y_series.astype(float), x_const).fit()
            coef = float(result.params[coefficient_name])
            ci_low, ci_high = _conf_interval_for(result, coefficient_name)
            se = _float_or_none(result.bse[coefficient_name])
            converged = bool(math.isfinite(coef))
            return EstimatorResult(coef, ci_low, ci_high, se, n, converged, cohort_note)

        if len(set(y_series.astype(int).tolist())) < 2:
            return EstimatorResult(
                None,
                None,
                None,
                None,
                n,
                False,
                _join_notes("binary outcome has fewer than two classes", cohort_note),
            )
        result = sm.Logit(y_series.astype(float), x_const).fit(disp=False, maxiter=100)
        coef = float(result.params[coefficient_name])
        ci_low, ci_high = _conf_interval_for(result, coefficient_name)
        se = _float_or_none(result.bse[coefficient_name])
        point = float(np.exp(coef))
        low = float(np.exp(ci_low)) if ci_low is not None else None
        high = float(np.exp(ci_high)) if ci_high is not None else None
        converged = bool(getattr(result, "mle_retvals", {}).get("converged", True))
        return EstimatorResult(point, low, high, se, n, converged, cohort_note)
    except Exception as exc:
        return EstimatorResult(
            None,
            None,
            None,
            None,
            n,
            False,
            _join_notes(str(exc), cohort_note),
        )


def fit_robustness_rows_from_records(
    *,
    specs: Sequence[RobustnessSpec],
    per_step_records: Sequence[Dict[str, Any]],
    primary_cohort: Optional[CohortDefinition] = None,
) -> Tuple[List[RobustnessPanelRow], List[str]]:
    """Auto-fit robustness rows from a coder-emitted estimator payload.

    The payload is intentionally explicit and local to ``step_summary``:

    ``{"estimator_adapter": {"data": [...], "exposure": "x", "outcome": "y"}}``

    Coder-emitted ``robustness_rows`` remain accepted as a fallback, but adapter
    rows win for duplicate ``spec_id`` values and a warning is returned.
    """

    payload_record = _find_estimator_payload(per_step_records)
    if payload_record is None:
        return [], []
    record, payload = payload_record
    warnings: List[str] = []
    declared_ids = _declared_robustness_ids(per_step_records)
    try:
        data = _load_payload_dataframe(payload)
    except Exception as exc:
        return [], [f"estimator_adapter payload could not be loaded: {exc}"]

    exposure = str(payload.get("exposure") or "").strip()
    outcome = str(payload.get("outcome") or "").strip()
    if not exposure or not outcome:
        return [], ["estimator_adapter requires exposure and outcome columns"]
    kind = str(payload.get("estimator_kind") or "logistic")
    default_missing = str(payload.get("missing_strategy") or "complete_case")
    outcome_columns = payload.get("outcome_columns") or {}
    if not isinstance(outcome_columns, dict):
        outcome_columns = {}

    rows: List[RobustnessPanelRow] = []
    row_specs: List[Tuple[str, str, Optional[RobustnessSpec]]] = [
        (PRIMARY_SPEC_ID, "primary", None),
        *[(spec.spec_id, spec.axis, spec) for spec in specs],
    ]
    for spec_id, axis, spec in row_specs:
        row = _fit_one_row(
            spec_id=spec_id,
            axis=axis,
            spec=spec,
            data=data,
            primary_cohort=primary_cohort,
            exposure=exposure,
            outcome=outcome,
            outcome_columns=outcome_columns,
            kind=kind,
            default_missing=default_missing,
            evidence_id=str(record.get("step_summary_evidence_id") or ""),
        )
        rows.append(row)
        if spec_id in declared_ids:
            warnings.append(
                f"estimator adapter row for {spec_id!r} overrides coder-emitted "
                "robustness_rows with the same spec_id"
            )
    return rows, warnings


def _fit_one_row(
    *,
    spec_id: str,
    axis: str,
    spec: Optional[RobustnessSpec],
    data: Any,
    primary_cohort: Optional[CohortDefinition],
    exposure: str,
    outcome: str,
    outcome_columns: Dict[str, Any],
    kind: str,
    default_missing: str,
    evidence_id: str,
) -> RobustnessPanelRow:
    cohort_definition = (
        spec.cohort_override
        if spec is not None and spec.cohort_override is not None
        else primary_cohort
    )
    if cohort_definition is None:
        cohort_definition = CohortDefinition(name="primary")
    try:
        cohort_df = build_cohort(cohort_definition, data=data)
        outcome_column = _outcome_column_for(spec, outcome, outcome_columns)
        missing_strategy = _missing_strategy_for(spec, default_missing)
        needed = [exposure, outcome_column]
        missing_columns = [column for column in needed if column not in cohort_df.columns]
        if missing_columns:
            raise KeyError("missing estimator column(s): " + ", ".join(missing_columns))
        model_df = apply_missing_strategy(cohort_df[needed], missing_strategy)
        result = fit_estimator(
            cohort=cohort_definition,
            X=model_df[[exposure]],
            y=model_df[outcome_column],
            kind=kind,
        )
        return RobustnessPanelRow(
            spec_id=spec_id,
            axis=axis,
            n=result.n,
            point_estimate=result.point_estimate,
            ci_low=result.ci_low,
            ci_high=result.ci_high,
            se=result.se,
            evidence_id=evidence_id,
            converged=result.converged,
            notes=result.notes,
        )
    except Exception as exc:
        return RobustnessPanelRow(
            spec_id=spec_id,
            axis=axis,
            n=0,
            point_estimate=None,
            ci_low=None,
            ci_high=None,
            se=None,
            evidence_id=evidence_id,
            converged=False,
            notes=str(exc),
        )


def _find_estimator_payload(
    per_step_records: Sequence[Dict[str, Any]]
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for record in per_step_records:
        summary = record.get("step_summary")
        if not isinstance(summary, dict):
            continue
        payload = summary.get("estimator_adapter")
        if isinstance(payload, dict):
            return record, payload
    return None


def _load_payload_dataframe(payload: Dict[str, Any]):
    import pandas as pd  # type: ignore

    if "data" in payload:
        return pd.DataFrame(payload["data"])
    if "data_path" in payload:
        path = Path(str(payload["data_path"]))
        if path.suffix.lower() == ".json":
            return pd.read_json(path)
        return pd.read_csv(path)
    raise ValueError("estimator_adapter requires data or data_path")


def _declared_robustness_ids(per_step_records: Sequence[Dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for record in per_step_records:
        summary = record.get("step_summary")
        if not isinstance(summary, dict):
            continue
        rows = summary.get("robustness_rows")
        if rows is None and isinstance(summary.get("robustness_panel"), dict):
            rows = summary["robustness_panel"].get("rows")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and row.get("spec_id"):
                out.add(str(row["spec_id"]))
    return out


def _outcome_column_for(
    spec: Optional[RobustnessSpec],
    default_outcome: str,
    outcome_columns: Dict[str, Any],
) -> str:
    if spec is None or not spec.outcome_override:
        return default_outcome
    override = spec.outcome_override
    if override.get("column"):
        return str(override["column"])
    target = str(override.get("target") or "")
    if target and target in outcome_columns:
        return str(outcome_columns[target])
    if target:
        return target
    return default_outcome


def _missing_strategy_for(spec: Optional[RobustnessSpec], default_missing: str) -> str:
    if spec is not None and spec.missing_override:
        return str(spec.missing_override.get("strategy") or default_missing)
    return default_missing


def _conf_interval_for(result: Any, coefficient_name: str) -> Tuple[Optional[float], Optional[float]]:
    ci = result.conf_int()
    try:
        low, high = ci.loc[coefficient_name]
    except AttributeError:
        names = list(result.params.index)
        idx = names.index(coefficient_name)
        low, high = ci[idx]
    return _float_or_none(low), _float_or_none(high)


def _float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _cohort_trace_note(cohort: Any) -> str:
    name = getattr(cohort, "name", None)
    if not name:
        return ""
    return f"cohort={name}"


def _join_notes(message: str, trace_note: str) -> str:
    if not trace_note:
        return message
    if not message:
        return trace_note
    return f"{message}; {trace_note}"


__all__ = [
    "EstimatorKind",
    "EstimatorResult",
    "fit_estimator",
    "fit_robustness_rows_from_records",
]
