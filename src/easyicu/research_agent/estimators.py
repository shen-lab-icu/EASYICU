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
from .methods.missing import apply_missing_strategy
from .pipeline_primary_effect import _extract_primary_effect_payload_from_records
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


def _robust_design(x_const: Any, *, keep: Sequence[str]) -> Tuple[Any, List[str]]:
    """Drop degenerate predictors so the fit is not rank-deficient.

    A logistic/linear fit raises "Singular matrix" when the design is
    rank-deficient — most commonly a constant column or two perfectly collinear
    columns (e.g. a missing-indicator that is constant once its variable has been
    imputed). This greedily keeps an independent column set, always preserving
    the columns in ``keep`` (the intercept and the coefficient of interest), and
    drops the linearly dependent remainder. General and deterministic: it removes
    only columns that carry no independent information, never an analytical
    choice. Returns the reduced design and the list of dropped column names.
    """
    import numpy as np  # type: ignore

    cols = list(x_const.columns)
    keep_set = [c for c in keep if c in cols]
    # 1) zero-variance columns (except the ones we must keep) carry no info.
    variances = x_const.var(axis=0, ddof=0)
    zero_var = [
        c
        for c in cols
        if c not in keep_set and not (float(variances.get(c, 0.0)) > 0.0)
    ]
    working = x_const.drop(columns=zero_var)
    # 2) greedily build a full-rank set, prioritising the must-keep columns so a
    # dependent *other* column is dropped rather than the exposure/intercept.
    ordered = keep_set + [c for c in working.columns if c not in keep_set]
    kept: List[str] = []
    matrix: Optional[Any] = None
    rank = 0
    for col in ordered:
        vec = working[col].to_numpy(dtype=float).reshape(-1, 1)
        trial = vec if matrix is None else np.hstack([matrix, vec])
        trial_rank = int(np.linalg.matrix_rank(trial))
        if trial_rank > rank:
            kept.append(col)
            matrix = trial
            rank = trial_rank
    dropped = [c for c in cols if c not in kept]
    return x_const[kept], dropped


def fit_estimator(
    *, cohort: Any, X: Any, y: Any, kind: EstimatorKind | str
) -> EstimatorResult:
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
    combined = pd.concat(
        [x_df.reset_index(drop=True), y_series.rename("__y__")], axis=1
    )
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
            _join_notes(
                "sample size too small for deterministic estimator", cohort_note
            ),
        )

    try:
        x_const = sm.add_constant(x_df.astype(float), has_constant="add")
        coefficient_name = next(col for col in x_const.columns if col != "const")
        # Drop rank-deficient (constant / perfectly collinear) predictors so a
        # degenerate design does not dead-end the fit with "Singular matrix".
        x_const, dropped = _robust_design(x_const, keep=["const", coefficient_name])
        drop_note = (
            f"dropped collinear/constant predictors: {', '.join(dropped)}"
            if dropped
            else ""
        )
        fit_note = _join_notes(drop_note, cohort_note) if drop_note else cohort_note
        if kind == "linear":
            result = sm.OLS(y_series.astype(float), x_const).fit()
            coef = float(result.params[coefficient_name])
            ci_low, ci_high = _conf_interval_for(result, coefficient_name)
            se = _float_or_none(result.bse[coefficient_name])
            converged = bool(math.isfinite(coef))
            return EstimatorResult(coef, ci_low, ci_high, se, n, converged, fit_note)

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
        try:
            result = sm.Logit(y_series.astype(float), x_const).fit(
                disp=False, maxiter=100
            )
            coef = float(result.params[coefficient_name])
            ci_low, ci_high = _conf_interval_for(result, coefficient_name)
            se = _float_or_none(result.bse[coefficient_name])
            point = float(np.exp(coef))
            low = float(np.exp(ci_low)) if ci_low is not None else None
            high = float(np.exp(ci_high)) if ci_high is not None else None
            converged = bool(getattr(result, "mle_retvals", {}).get("converged", True))
            if not math.isfinite(coef):
                raise ValueError("non-finite logistic coefficient")
            return EstimatorResult(point, low, high, se, n, converged, fit_note)
        except Exception as mle_exc:
            # Unpenalised MLE diverges under (quasi-)separation. A ridge-penalised
            # fit is the standard robust fallback: it yields a finite point
            # estimate (a Wald CI is not reliable under penalisation, so it is
            # reported as None rather than fabricated).
            reg = sm.Logit(y_series.astype(float), x_const).fit_regularized(
                alpha=1.0, L1_wt=0.0, disp=False, maxiter=200
            )
            coef = float(reg.params[coefficient_name])
            if not math.isfinite(coef):
                raise mle_exc
            return EstimatorResult(
                float(np.exp(coef)),
                None,
                None,
                None,
                n,
                False,
                _join_notes(
                    f"penalised (ridge) fallback after MLE failed: {mle_exc}",
                    fit_note,
                ),
            )
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
    run_dir: Optional[Path] = None,
) -> Tuple[List[RobustnessPanelRow], List[str]]:
    """Auto-fit the robustness-panel rows for a run.

    Row sourcing, in priority order:

    1. **Primary row** — the step's *validated* primary estimate
       (``step_summary.primary_or`` / CI / n), never a re-fit, so the panel
       headline matches the manuscript-facing primary effect. Falls back to a
       primary re-fit only when no step reports a primary.
    2. **Variant rows** — re-fit per spec, adjusted for the primary model's
       recovered covariate set (``run_dir``) so each variant reports the
       exposure's effect on the same footing as the primary rather than as a
       bare unadjusted single-predictor fit.

    Data source for the re-fits: an explicit ``step_summary.estimator_adapter``
    payload (``{"data": [...], "exposure": "x", "outcome": "y"}``) *if a coder
    emits one* — no current prompt does, so in practice the operative path
    infers the exposure/outcome and loads the cohort parquet. Coder-emitted
    ``robustness_rows`` remain accepted; adapter/step rows win for duplicate
    ``spec_id`` values and a warning is returned.
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
            return [], [
                f"deterministic estimator adapter could not load cohort data: {exc}"
            ]
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
        return [], warnings + [
            "estimator_adapter requires exposure and outcome columns"
        ]

    rows: List[RobustnessPanelRow] = []
    # Recover the primary model's adjustment set so robustness *variants* are fit
    # on the same footing as the (now step-sourced, adjusted) primary effect
    # rather than as bare unadjusted single-predictor re-fits. Empty when the
    # covariate set cannot be recovered (e.g. no run_dir) — variants then stay
    # unadjusted, i.e. the previous behaviour.
    covariates = _recover_primary_covariates(
        run_dir,
        per_step_records=per_step_records,
        exposure=exposure,
        outcome=outcome,
        available_columns=getattr(data, "columns", ()),
    )
    if covariates:
        warnings.append(
            "robustness variants adjusted for the primary model covariates: "
            + ", ".join(covariates)
        )
    primary_row = _primary_row_from_step_records(per_step_records)
    primary_measure = _primary_effect_measure_from_records(per_step_records)
    if (
        primary_row is not None
        and primary_measure == "HR"
        and kind not in ("cox", "cox_ph")
    ):
        # The primary estimand is a hazard ratio (survival design), but the
        # estimator adapter can only refit logistic/linear variants (OR/beta) — a
        # DIFFERENT estimand on a DIFFERENT scale. Appending them would fabricate
        # a misleading mixed-measure "robustness range" (HR 1.82 primary vs OR
        # ~1.0 refit variants that look non-robust only because they measure a
        # different quantity). Report the primary hazard ratio alone; survival
        # robustness belongs to the deterministic Cox runner's own sensitivity
        # outputs, not an OR refit. Case-neutral: fires only when the primary is
        # an HR and no Cox variant estimator is available.
        rows.append(primary_row)
        if specs:
            warnings.append(
                "skipped logistic robustness variants for a hazard-ratio primary "
                "estimand (odds-ratio refits are not valid Cox hazard-ratio "
                "robustness variants); reporting the primary hazard ratio only"
            )
        return rows, warnings
    if primary_row is not None:
        rows.append(primary_row)
        if PRIMARY_SPEC_ID in declared_ids:
            warnings.append(
                "step primary estimate overrides coder-emitted robustness_rows "
                "with spec_id 'primary'"
            )
        row_specs: List[Tuple[str, str, Optional[RobustnessSpec]]] = [
            (spec.spec_id, spec.axis, spec) for spec in specs
        ]
    else:
        row_specs = [
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
            covariates=covariates,
        )
        rows.append(row)
        if spec_id in declared_ids:
            warnings.append(
                f"estimator adapter row for {spec_id!r} overrides coder-emitted "
                "robustness_rows with the same spec_id"
            )
    return rows, warnings


def _primary_row_from_step_records(
    per_step_records: Sequence[Dict[str, Any]],
) -> Optional[RobustnessPanelRow]:
    payload = _extract_primary_effect_payload_from_records(per_step_records)
    if not payload or payload.get("primary_or") is None:
        return None
    sample_size = payload.get("sample_size")
    n = int(sample_size) if isinstance(sample_size, int) else 0
    # ``primary_or`` holds the primary effect ratio whatever its measure (OR for a
    # logistic design, HR for a survival design). Record the measure in the notes
    # so the panel/writer does not silently label a Cox hazard ratio as an odds
    # ratio. Threading a real primary row here also STOPS the logistic-OR refit
    # fallback from overriding a survival estimand (H1 fix3j: real HR 1.82 was
    # buried under a refit near-null ~1.0 OR).
    measure = str(payload.get("effect_measure") or "").strip()
    notes = (
        f"Primary analysis estimate ({measure}) from step_summary."
        if measure
        else "Primary analysis estimate from step_summary."
    )
    return RobustnessPanelRow(
        spec_id=PRIMARY_SPEC_ID,
        axis="primary",
        n=n,
        point_estimate=_float_or_none(payload.get("primary_or")),
        ci_low=_float_or_none(payload.get("primary_ci_low")),
        ci_high=_float_or_none(payload.get("primary_ci_high")),
        se=None,
        evidence_id=str(payload.get("evidence_id") or ""),
        converged=True,
        notes=notes,
    )


def _primary_effect_measure_from_records(
    per_step_records: Sequence[Dict[str, Any]],
) -> Optional[str]:
    """The measure ("HR" / "OR") of the selected primary effect, or ``None``."""
    payload = _extract_primary_effect_payload_from_records(per_step_records)
    return str((payload or {}).get("effect_measure") or "").strip() or None


def _recover_primary_covariates(
    run_dir: Optional[Path],
    *,
    per_step_records: Sequence[Dict[str, Any]],
    exposure: Optional[str],
    outcome: Optional[str],
    available_columns: Any,
) -> List[str]:
    """The primary model's adjustment-set column names, or ``[]``.

    Prefers an explicit covariate declaration in the selected primary step's
    summary, then falls back to the conservative code parser used by the
    overadjustment check. It intentionally does not treat generic effect-summary
    CSV ``term`` columns as covariates: those rows can list sensitivity model
    focal terms rather than the primary adjustment set.
    """
    if run_dir is None:
        return []
    from .plan_utils import _covariate_names_from_code

    base = Path(run_dir)
    payload = _extract_primary_effect_payload_from_records(per_step_records)
    step_id = str((payload or {}).get("step_id") or "").strip()
    names = _explicit_primary_covariates_from_records(
        per_step_records,
        step_id=step_id,
    )
    search_dirs: List[Path] = []
    if step_id:
        for candidate in (
            base / "steps" / step_id / "outputs",
            base / "steps" / step_id,
        ):
            if candidate.exists():
                search_dirs.append(candidate)
    if not search_dirs and base.exists():
        search_dirs.append(base)

    if not names:
        for directory in search_dirs:
            names = _covariate_names_from_code(directory)
            if names:
                break
    try:
        available = {str(column) for column in list(available_columns)}
    except TypeError:
        available = set()
    excluded = {str(exposure or ""), str(outcome or "")}
    return [name for name in names if name in available and name not in excluded]


def _explicit_primary_covariates_from_records(
    per_step_records: Sequence[Dict[str, Any]],
    *,
    step_id: str,
) -> List[str]:
    if not step_id:
        return []
    for record in per_step_records or []:
        if str(record.get("step_id") or "") != step_id:
            continue
        summary = record.get("step_summary")
        if not isinstance(summary, dict):
            return []
        return _explicit_covariate_names(summary)
    return []


def _explicit_covariate_names(payload: Dict[str, Any]) -> List[str]:
    candidates: List[Any] = [
        payload.get("model_covariates"),
        payload.get("covariates"),
        payload.get("adjustment_covariates"),
        payload.get("adjustment_cols"),
        payload.get("confounders"),
    ]
    model_metadata = payload.get("model_metadata")
    if isinstance(model_metadata, dict):
        candidates.extend(
            [
                model_metadata.get("model_covariates"),
                model_metadata.get("covariates"),
                model_metadata.get("adjustment_covariates"),
                model_metadata.get("adjustment_cols"),
                model_metadata.get("confounders"),
            ]
        )
    for value in candidates:
        names = _string_sequence(value)
        if names:
            return names
    return []


def _string_sequence(value: Any) -> List[str]:
    if not isinstance(value, (list, tuple)):
        return []
    names: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in names:
            names.append(text)
    return names


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
    covariates: Sequence[str] = (),
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
        outcome_column = (
            _resolve_column_alias(
                outcome_column,
                cohort_df.columns,
                context=context,
                excluded={exposure},
            )
            or outcome_column
        )
        missing_strategy = _missing_strategy_for(spec, default_missing)
        # Adjust for the primary model's covariates (when supplied and present)
        # so the variant reports the exposure's *adjusted* effect on the same
        # footing as the primary — fit_estimator reports the first non-const
        # column, so the exposure must lead the design matrix.
        present_covariates = [
            column
            for column in covariates
            if column in cohort_df.columns and column not in (exposure, outcome_column)
        ]
        needed = [exposure, outcome_column, *present_covariates]
        missing_columns = [
            column for column in needed if column not in cohort_df.columns
        ]
        if missing_columns:
            raise KeyError("missing estimator column(s): " + ", ".join(missing_columns))
        model_df = apply_missing_strategy(cohort_df[needed], missing_strategy)
        result = fit_estimator(
            cohort=cohort_definition,
            X=model_df[[exposure, *present_covariates]],
            y=model_df[outcome_column],
            kind=kind,
        )
        notes = result.notes
        if present_covariates:
            notes = _join_notes(
                "adjusted for " + ", ".join(present_covariates), result.notes
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
            notes=notes,
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
    per_step_records: Sequence[Dict[str, Any]],
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
        resolved = _resolve_column_alias(
            candidate, columns, context=context, excluded=excluded
        )
        if resolved:
            return resolved
    if requested:
        resolved = _resolve_column_alias(
            requested, columns, context=context, excluded=excluded
        )
        if resolved:
            return resolved

    question = str(getattr(context, "research_question", "") or "")
    best: Tuple[int, str] = (0, "")
    for variable in getattr(context, "variables", []) or []:
        name = str(getattr(variable, "name", "") or "")
        role = str(getattr(variable, "role", "") or "").lower()
        if (
            not name
            or role.endswith("outcome")
            or role.endswith("id")
            or role.endswith("time")
        ):
            continue
        column = _resolve_column_alias(
            name, columns, context=context, excluded=excluded
        )
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
        *_candidate_values_from_records(
            per_step_records, ("outcome", "target_outcome")
        ),
    ):
        if not candidate:
            continue
        resolved = _resolve_column_alias(str(candidate), columns, context=context)
        if resolved:
            return resolved
    for variable in getattr(context, "variables", []) or []:
        if str(getattr(variable, "role", "") or "").lower().endswith("outcome"):
            resolved = _resolve_column_alias(
                str(getattr(variable, "name", "")), columns, context=context
            )
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

    context_names = [
        str(getattr(v, "name", "") or "")
        for v in getattr(context, "variables", []) or []
    ]
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
    tokens = [
        token for token in raw if token and token not in {"admission", "adm", "window"}
    ]
    compact = _normalise_token(value)
    if compact and compact not in tokens:
        tokens.append(compact)
    return tokens


def _normalise_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _conf_interval_for(
    result: Any, coefficient_name: str
) -> Tuple[Optional[float], Optional[float]]:
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
