"""[Layer 3: Safe Analytical Runtime] Deterministic estimator adapter.

This module turns a typed robustness specification into a reproducible model
fit. It is deliberately narrow: statsmodels logistic and linear estimators,
simple deterministic missing-data policies, and row-level failure capture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import re
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
    data: Any = None,
    cohort_path: Optional[Path] = None,
    context: Any = None,
    exposure: Optional[str] = None,
    outcome: Optional[str] = None,
) -> Tuple[List[RobustnessPanelRow], List[str]]:
    """Auto-fit robustness rows from a coder-emitted estimator payload.

    The payload is intentionally explicit and local to ``step_summary``:

    ``{"estimator_adapter": {"data": [...], "exposure": "x", "outcome": "y"}}``

    Coder-emitted ``robustness_rows`` remain accepted as a fallback, but adapter
    rows win for duplicate ``spec_id`` values and a warning is returned.
    """

    warnings: List[str] = []
    declared_ids = _declared_robustness_ids(per_step_records)
    payload_record = _find_estimator_payload(per_step_records)
    if payload_record is not None:
        record, payload = payload_record
        try:
            data = _load_payload_dataframe(payload)
        except Exception as exc:
            return [], [f"estimator_adapter payload could not be loaded: {exc}"]

        exposure = str(payload.get("exposure") or exposure or "").strip()
        outcome = str(payload.get("outcome") or outcome or "").strip()
        kind = str(payload.get("estimator_kind") or "logistic")
        default_missing = str(payload.get("missing_strategy") or "complete_case")
        outcome_columns = payload.get("outcome_columns") or {}
        if not isinstance(outcome_columns, dict):
            outcome_columns = {}
        evidence_id = str(record.get("step_summary_evidence_id") or "")
    else:
        try:
            data = _load_direct_dataframe(data=data, cohort_path=cohort_path)
        except Exception as exc:
            return [], [f"deterministic estimator adapter could not load cohort data: {exc}"]
        exposure = _infer_exposure_column(
            data=data,
            context=context,
            per_step_records=per_step_records,
            requested=exposure,
            outcome=outcome,
        )
        outcome = _infer_outcome_column(
            data=data,
            context=context,
            per_step_records=per_step_records,
            requested=outcome,
        )
        kind = "logistic"
        default_missing = "complete_case"
        outcome_columns = {}
        evidence_id = "robustness_panel"
        warnings.append(
            "deterministic estimator adapter used cohort parquet because no "
            "step_summary estimator_adapter payload was emitted"
        )

    if not exposure or not outcome:
        return [], warnings + ["estimator_adapter requires exposure and outcome columns"]

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
            context=context,
            evidence_id=evidence_id,
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
    context: Any,
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
        data_for_filter = _data_with_predicate_aliases(
            data=data,
            cohort_definition=cohort_definition,
            exposure=exposure,
            context=context,
        )
        cohort_df = build_cohort(cohort_definition, data=data_for_filter)
        outcome_column = _outcome_column_for(spec, outcome, outcome_columns)
        outcome_column = _resolve_column_alias(
            outcome_column,
            cohort_df.columns,
            context=context,
            excluded={exposure},
        ) or outcome_column
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
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if path.suffix.lower() == ".json":
            return pd.read_json(path)
        return pd.read_csv(path)
    raise ValueError("estimator_adapter requires data or data_path")


def _load_direct_dataframe(*, data: Any, cohort_path: Optional[Path]):
    import pandas as pd  # type: ignore

    if data is not None:
        return pd.DataFrame(data).copy()
    if cohort_path is None:
        raise ValueError("cohort_path is required when no estimator payload exists")
    path = Path(cohort_path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


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
    if override.get("concept_id"):
        return str(override["concept_id"])
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


def _infer_exposure_column(
    *,
    data: Any,
    context: Any,
    per_step_records: Sequence[Dict[str, Any]],
    requested: Optional[str],
    outcome: Optional[str],
) -> str:
    columns = list(getattr(data, "columns", []))
    excluded = {outcome} if outcome else set()
    for candidate in _candidate_values_from_records(
        per_step_records,
        ("primary_predictor", "predictor", "exposure", "primary_exposure"),
    ):
        resolved = _resolve_column_alias(candidate, columns, context=context, excluded=excluded)
        if resolved:
            return resolved
    if requested:
        resolved = _resolve_column_alias(requested, columns, context=context, excluded=excluded)
        if resolved:
            return resolved

    question = str(getattr(context, "research_question", "") or "")
    best: Tuple[int, str] = (0, "")
    for variable in getattr(context, "variables", []) or []:
        name = str(getattr(variable, "name", "") or "")
        role = str(getattr(variable, "role", "") or "").lower()
        if not name or role.endswith("outcome") or role.endswith("id") or role.endswith("time"):
            continue
        column = _resolve_column_alias(name, columns, context=context, excluded=excluded)
        if not column:
            continue
        score = _question_overlap_score(name, question)
        if "score" in name.lower() or "sofa" in name.lower():
            score += 2
        if score > best[0]:
            best = (score, column)
    if best[1]:
        return best[1]

    for column in columns:
        role = _context_role_for_column(column, context)
        if role in {"outcome", "id", "time", "demographic"}:
            continue
        if column not in excluded:
            return str(column)
    return ""


def _infer_outcome_column(
    *,
    data: Any,
    context: Any,
    per_step_records: Sequence[Dict[str, Any]],
    requested: Optional[str],
) -> str:
    columns = list(getattr(data, "columns", []))
    for candidate in (
        requested,
        getattr(context, "target_outcome", None),
        *_candidate_values_from_records(per_step_records, ("outcome", "target_outcome")),
    ):
        if not candidate:
            continue
        resolved = _resolve_column_alias(str(candidate), columns, context=context)
        if resolved:
            return resolved
    for variable in getattr(context, "variables", []) or []:
        if str(getattr(variable, "role", "") or "").lower().endswith("outcome"):
            resolved = _resolve_column_alias(str(getattr(variable, "name", "")), columns, context=context)
            if resolved:
                return resolved
    for fallback in ("death", "mortality", "outcome", "y"):
        resolved = _resolve_column_alias(fallback, columns, context=context)
        if resolved:
            return resolved
    return ""


def _candidate_values_from_records(
    per_step_records: Sequence[Dict[str, Any]], keys: Sequence[str]
) -> List[str]:
    out: List[str] = []
    for record in per_step_records:
        summary = record.get("step_summary")
        if not isinstance(summary, dict):
            continue
        for key in keys:
            for value in _find_nested_values(summary, key):
                if isinstance(value, str) and value.strip():
                    out.append(value.strip())
    return out


def _find_nested_values(value: Any, key: str) -> List[Any]:
    found: List[Any] = []
    if isinstance(value, dict):
        for k, v in value.items():
            if str(k) == key:
                found.append(v)
            found.extend(_find_nested_values(v, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(_find_nested_values(item, key))
    return found


def _data_with_predicate_aliases(
    *,
    data: Any,
    cohort_definition: CohortDefinition,
    exposure: str,
    context: Any,
) -> Any:
    predicates = [*cohort_definition.inclusion, *cohort_definition.exclusion]
    missing = [pred for pred in predicates if pred.concept_id not in data.columns]
    if not missing:
        return data
    out = data.copy()
    for pred in missing:
        alias = _resolve_column_alias(
            pred.concept_id,
            out.columns,
            context=context,
            preferred=exposure,
        )
        if alias:
            out[pred.concept_id] = out[alias]
    return out


def _resolve_column_alias(
    requested: str,
    columns: Sequence[Any],
    *,
    context: Any,
    excluded: Optional[set[str]] = None,
    preferred: Optional[str] = None,
) -> Optional[str]:
    excluded = excluded or set()
    requested = str(requested or "")
    if not requested:
        return None
    string_columns = [str(c) for c in columns]
    if requested in string_columns and requested not in excluded:
        return requested
    normalised = _normalise_token(requested)
    for column in string_columns:
        if column in excluded:
            continue
        if _normalise_token(column) == normalised:
            return column
    if preferred and preferred in string_columns and preferred not in excluded:
        if _name_matches_concept(preferred, requested):
            return preferred

    context_names = [str(getattr(v, "name", "") or "") for v in getattr(context, "variables", []) or []]
    candidates = [name for name in context_names if name in string_columns]
    candidates.extend(column for column in string_columns if column not in candidates)
    scored: List[Tuple[int, str]] = []
    for column in candidates:
        if column in excluded:
            continue
        score = _alias_score(requested, column)
        if score > 0:
            scored.append((score, column))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def _context_role_for_column(column: str, context: Any) -> str:
    for variable in getattr(context, "variables", []) or []:
        if str(getattr(variable, "name", "") or "") == column:
            return str(getattr(variable, "role", "") or "").lower().split(".")[-1]
    return ""


def _question_overlap_score(name: str, question: str) -> int:
    nq = _normalise_token(question)
    score = 0
    for token in _name_tokens(name):
        if token and token in nq:
            score += len(token)
    return score


def _name_matches_concept(name: str, concept: str) -> bool:
    return _alias_score(concept, name) > 0


def _alias_score(requested: str, column: str) -> int:
    req = _normalise_token(requested)
    col = _normalise_token(column)
    if not req or not col:
        return 0
    if req == col:
        return 100
    score = 0
    if req in col or col in req:
        score += min(len(req), len(col))
    req_tokens = set(_name_tokens(requested))
    col_tokens = set(_name_tokens(column))
    score += 8 * len(req_tokens & col_tokens)
    return score


def _name_tokens(value: str) -> List[str]:
    raw = re.split(r"[^a-zA-Z0-9]+", str(value).lower())
    tokens = [token for token in raw if token and token not in {"admission", "adm", "window"}]
    compact = _normalise_token(value)
    if compact and compact not in tokens:
        tokens.append(compact)
    return tokens


def _normalise_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


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
