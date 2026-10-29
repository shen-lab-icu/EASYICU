"""Compile Planner-declared model terms into a numeric design matrix.

This is the sole execution owner for variable coding shared by the adjusted
association and survival primary executors, and by the weight models of the
target trial executor, which have covariates only. It validates the observed
domain against the closed declaration and preserves missing values for the
caller's declared missing-data policy; it never chooses a reference or
dtype-based encoding.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..canonical_json import canonical_sha256
from ..contracts.model_retention import estimable_missing_category_rows
from ..contracts.model_terms import (
    DataPreparationPlan,
    ModelTermSpec,
    PreparationRefusal,
    level_identity_class,
    level_spelling,
)
from ..methods.descriptive_inputs import require_observed_columns
from ..methods.missing import (
    aggregate_windows,
    apply_outliers,
    database_preparation_status,
    source_has_raw_values,
    standardize_columns,
)
from ..methods.missing_data import ImputationRefusal, multiple_impute, rubin_pool


class ModelTermCompilationError(ValueError):
    """A typed model-term declaration cannot be bound to its source frame."""

    owner = "easyicu.host.model_term_compiler_v1"
    phase = "model_matrix_compilation"

    def __init__(self, reason_code: str, message: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {message}")


@dataclass(frozen=True, slots=True)
class CompiledModelMatrix:
    """Numeric design plus exact source lineage for every generated column."""

    design: Any
    exposure_columns: Tuple[str, ...]
    source_by_design_column: Mapping[str, str]


def _contrast_column(name: str, level: str) -> str:
    return f"{name}__is_{level}"


def _validate_observed_levels(series: Any, term: ModelTermSpec) -> Any:
    # Normalize pandas nullable scalars before using the dependency-neutral
    # spelling contract; missing values must never become an observed level.
    values = series.astype("object").where(series.notna(), None)
    keys = values.map(level_spelling)

    # A spelling reached by more than one type class is two source categories
    # wearing one name. Treatment coding would merge them into a single
    # contrast and the closed-domain check above would see nothing wrong,
    # because it compares spellings to spellings.
    classes_by_spelling: Dict[str, set] = {}
    for raw in values.unique().tolist():
        spelling = level_spelling(raw)
        if not spelling:
            continue
        classes_by_spelling.setdefault(spelling, set()).add(level_identity_class(raw))
    ambiguous = sorted(
        spelling
        for spelling, classes in classes_by_spelling.items()
        if len(classes) > 1
    )
    if ambiguous:
        raise ModelTermCompilationError(
            "model_term_level_identity_ambiguous",
            f"{term.name!r} holds values of more than one type that share a "
            "declared level spelling, so distinct categories would be coded as "
            "one: "
            + ", ".join(
                f"{spelling!r} ({'/'.join(sorted(classes_by_spelling[spelling]))})"
                for spelling in ambiguous
            ),
        )

    observed = {value for value in keys.unique().tolist() if value}
    declared = set(term.levels or ())
    unexpected = sorted(observed - declared)
    absent = sorted(declared - observed)
    if unexpected:
        raise ModelTermCompilationError(
            "model_term_observed_level_undeclared",
            f"{term.name!r} contains undeclared level(s): "
            + ", ".join(repr(item) for item in unexpected),
        )
    if absent:
        raise ModelTermCompilationError(
            "model_term_declared_level_absent",
            f"{term.name!r} has no observed row for declared level(s): "
            + ", ".join(repr(item) for item in absent),
        )
    return keys


def compile_model_terms(
    frame: Any,
    *,
    terms: Sequence[ModelTermSpec],
    exposure: Optional[str],
) -> CompiledModelMatrix:
    """Compile one exact term roster without inferring scientific choices.

    ``exposure`` names the roster's one exposure term; ``None`` declares a
    roster of covariates only, which may then hold no exposure term.
    """

    import pandas as pd

    roster = tuple(
        item if isinstance(item, ModelTermSpec) else ModelTermSpec.model_validate(item)
        for item in terms
    )
    names = [item.name for item in roster]
    if len(names) != len(set(names)):
        raise ModelTermCompilationError(
            "model_term_source_repeated", "model term source names must be unique"
        )
    exposures = [item for item in roster if item.role == "exposure"]
    if exposure is None:
        if exposures:
            raise ModelTermCompilationError(
                "model_term_exposure_mismatch",
                "a covariate-only roster declares no exposure term",
            )
    elif len(exposures) != 1 or exposures[0].name != exposure:
        raise ModelTermCompilationError(
            "model_term_exposure_mismatch",
            "the compiled roster must contain one exact declared exposure",
        )
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ModelTermCompilationError(
            "model_term_source_missing",
            "declared model term column(s) are absent: " + ", ".join(missing),
        )

    require_observed_columns(frame, names)
    design = pd.DataFrame(index=frame.index)
    source_by_column: Dict[str, str] = {}
    exposure_columns: list[str] = []
    for term in roster:
        source = frame[term.name]
        generated: list[str] = []
        if term.coding == "continuous":
            numeric = pd.to_numeric(source, errors="coerce")
            conversion_loss = source.notna() & numeric.isna()
            if bool(conversion_loss.any()):
                raise ModelTermCompilationError(
                    "model_term_numeric_conversion_loss",
                    f"continuous term {term.name!r} contains non-numeric values",
                )
            finite = numeric.dropna().map(lambda value: math.isfinite(float(value)))
            if not bool(finite.all()):
                raise ModelTermCompilationError(
                    "model_term_nonfinite",
                    f"continuous term {term.name!r} contains non-finite values",
                )
            design[term.name] = numeric.astype(float)
            generated = [term.name]
        elif term.coding == "ordinal_linear":
            keys = _validate_observed_levels(source, term)
            mapping = {
                level: float(index) for index, level in enumerate(term.levels or ())
            }
            encoded = keys.map(mapping)
            encoded = encoded.mask(keys.eq(""))
            design[term.name] = encoded.astype(float)
            generated = [term.name]
        else:
            keys = _validate_observed_levels(source, term)
            unobserved = keys.eq("")
            for level in term.contrast_levels:
                name = _contrast_column(term.name, level)
                design[name] = (keys == level).astype(float).mask(unobserved)
                generated.append(name)

        if not generated:
            raise ModelTermCompilationError(
                "model_term_generated_no_columns",
                f"term {term.name!r} generated no estimable design column",
            )
        for column in generated:
            source_by_column[column] = term.name
        if term.role == "exposure":
            exposure_columns.extend(generated)

    return CompiledModelMatrix(
        design=design,
        exposure_columns=tuple(exposure_columns),
        source_by_design_column=dict(source_by_column),
    )


@dataclass(frozen=True, slots=True)
class MissingCategoryTerm:
    """One covariate whose unmeasured rows a model keeps as their own state."""

    covariate: str
    indicator: str
    coding: str
    #: The observed median a continuous or ordinal covariate is filled with;
    #: ``None`` for a binary or categorical one, whose contrasts are zero.
    fill_value: Optional[float]
    n_missing: int

    def public(self) -> Dict[str, Any]:
        return {
            "covariate": self.covariate,
            "term": self.indicator,
            "coding": self.coding,
            "fill_value": self.fill_value,
            "n_missing": self.n_missing,
        }


@dataclass(frozen=True, slots=True)
class UnmeasuredRowsDropped:
    """A listed covariate whose unmeasured state could not be estimated.

    Its unmeasured rows leave the fit, as under the default policy; the
    reason says which part of the estimability rule failed.
    """

    covariate: str
    n_missing: int
    reason_code: str

    def public(self) -> Dict[str, Any]:
        return {
            "covariate": self.covariate,
            "n_missing": self.n_missing,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True)
class PrimaryModelRows:
    """The rows one declared model is fitted on, and their complete design."""

    #: Fitting rows only, indexed like the source frame; no value is missing.
    design: Any
    exposure_columns: Tuple[str, ...]
    source_by_design_column: Mapping[str, str]
    missing_category_terms: Tuple[MissingCategoryTerm, ...] = ()
    unmeasured_rows_dropped: Tuple[UnmeasuredRowsDropped, ...] = ()

    @property
    def availability_columns(self) -> Tuple[str, ...]:
        return tuple(item.indicator for item in self.missing_category_terms)


def _unmeasured_indicator(term: ModelTermSpec) -> str:
    # A binary or categorical covariate gains a level, spelled like its other
    # contrasts; a continuous or ordinal one gains an indicator beside it.
    if term.coding in {"binary", "categorical"}:
        return _contrast_column(term.name, "unmeasured")
    return f"{term.name}__unmeasured"


def primary_model_rows(
    frame: Any,
    *,
    terms: Sequence[ModelTermSpec],
    exposure: Optional[str],
    outcome: str,
    missing_category_covariates: Sequence[str] = (),
    term_groups: Optional[Mapping[str, str]] = None,
) -> PrimaryModelRows:
    """Select, and where declared fill, the rows one declared model uses.

    Without a declared policy these are the complete rows every fit already
    used: the outcome and every design column present.  A covariate listed in
    ``missing_category_covariates`` no longer removes its unmeasured rows.  A
    continuous or ordinal one is filled with its observed median over the
    fitting rows and gains an ``<name>__unmeasured`` indicator; a binary or
    categorical one gains an ``<name>__is_unmeasured`` level.  The exposure,
    the outcome and every unlisted covariate still have to be observed.  A
    state too small, or too uniform in its outcome, to be estimated is not
    kept: its rows leave the fit and ``unmeasured_rows_dropped`` says why
    (``contracts.model_retention.estimable_missing_category_rows``).

    ``term_groups`` maps generated terms (a spline basis) to the covariate
    they represent, so a listed covariate expanded into several continuous
    columns is kept as one state: its columns are filled with 0 there, which
    its ``<name>__unmeasured`` indicator absorbs, and the same rule decides.

    ``exposure=None`` selects the rows of a model with covariates only
    (:func:`covariate_model_rows`).
    """

    roster = [
        item if isinstance(item, ModelTermSpec) else ModelTermSpec.model_validate(item)
        for item in terms
    ]
    require_observed_columns(frame, (outcome,))
    compiled = compile_model_terms(frame, terms=roster, exposure=exposure)
    if outcome not in frame.columns:
        raise ModelTermCompilationError(
            "model_term_source_missing", f"the declared outcome {outcome!r} is absent"
        )
    groups = {str(key): str(value) for key, value in (term_groups or {}).items()}
    by_term = {item.name: item for item in roster}
    if any(
        key not in by_term
        or by_term[key].role != "covariate"
        or by_term[key].coding != "continuous"
        or value in by_term
        for key, value in groups.items()
    ):
        raise ModelTermCompilationError(
            "missing_category_group_invalid",
            "a term group must map continuous covariate terms to a covariate "
            "that is not itself a term",
        )
    members: Dict[str, list[ModelTermSpec]] = {}
    for item in roster:
        members.setdefault(groups.get(item.name, item.name), []).append(item)
    listed = list(dict.fromkeys(str(name).strip() for name in missing_category_covariates))
    for name in listed:
        if name not in members:
            raise ModelTermCompilationError(
                "missing_category_covariate_undeclared",
                f"{name!r} is not a declared model term",
            )
        if any(item.role == "exposure" for item in members[name]):
            raise ModelTermCompilationError(
                "missing_category_exposure_refused",
                f"the exposure {name!r} cannot be kept as unmeasured",
            )
    columns_by_source: Dict[str, list[str]] = {}
    for column, source in compiled.source_by_design_column.items():
        columns_by_source.setdefault(groups.get(source, source), []).append(column)

    design = compiled.design
    admitted = frame[outcome].notna()
    for source, columns in columns_by_source.items():
        if source not in listed:
            admitted &= design[columns].notna().all(axis=1)
    unmeasured_by_name = {
        name: design[columns_by_source[name]].isna().any(axis=1) for name in listed
    }
    n_admitted = int(admitted.sum())
    for name in listed:
        if n_admitted and int((unmeasured_by_name[name] & admitted).sum()) == n_admitted:
            raise ModelTermCompilationError(
                "missing_category_covariate_unobserved",
                f"{name!r} is unmeasured on every fitting row",
            )
    keep, not_estimable = estimable_missing_category_rows(
        rows=admitted, unmeasured=unmeasured_by_name, outcome=frame[outcome]
    )
    fitted = design.loc[keep].copy()
    source_by_design_column = dict(compiled.source_by_design_column)
    kept: list[MissingCategoryTerm] = []
    patterns: Dict[bytes, str] = {}
    for name in listed:
        if name in not_estimable:
            continue
        group = members[name]
        term = group[0]
        columns = columns_by_source[name]
        unmeasured = fitted[columns].isna().any(axis=1)
        n_missing = int(unmeasured.sum())
        if n_missing == 0:
            continue
        if n_missing == len(fitted):
            raise ModelTermCompilationError(
                "missing_category_covariate_unobserved",
                f"{name!r} is unmeasured on every fitting row",
            )
        indicator = (
            f"{name}__unmeasured" if len(group) > 1 else _unmeasured_indicator(term)
        )
        if indicator in fitted.columns or indicator in frame.columns:
            raise ModelTermCompilationError(
                "missing_category_level_collision"
                if term.coding in {"binary", "categorical"}
                else "missing_category_term_collision",
                f"{indicator!r} already names a column or declared level",
            )
        pattern = unmeasured.to_numpy().tobytes()
        if pattern in patterns:
            raise ModelTermCompilationError(
                "missing_category_indicator_collinear",
                f"{name!r} and {patterns[pattern]!r} are unmeasured on exactly "
                "the same rows, so their indicators cannot both be estimated",
            )
        patterns[pattern] = name
        fill: Optional[float] = None
        coding = str(term.coding)
        if len(group) > 1:
            # A basis is filled at 0; the indicator absorbs any constant.
            coding = "continuous_basis"
            fill = 0.0
            fitted.loc[unmeasured, columns] = 0.0
        elif term.coding in {"continuous", "ordinal_linear"}:
            (column,) = columns
            fill = float(fitted.loc[~unmeasured, column].median())
            fitted[column] = fitted[column].fillna(fill)
        else:
            fitted.loc[unmeasured, columns] = 0.0
        fitted[indicator] = unmeasured.astype(float)
        source_by_design_column[indicator] = name
        kept.append(
            MissingCategoryTerm(
                covariate=name,
                indicator=indicator,
                coding=coding,
                fill_value=fill,
                n_missing=n_missing,
            )
        )
    return PrimaryModelRows(
        design=fitted,
        exposure_columns=compiled.exposure_columns,
        source_by_design_column=source_by_design_column,
        missing_category_terms=tuple(kept),
        unmeasured_rows_dropped=tuple(
            UnmeasuredRowsDropped(covariate=name, n_missing=n_missing, reason_code=reason)
            for name, (n_missing, reason) in sorted(not_estimable.items())
        ),
    )


def covariate_model_rows(
    frame: Any,
    *,
    terms: Sequence[ModelTermSpec],
    outcome: str,
    missing_category_covariates: Sequence[str] = (),
) -> PrimaryModelRows:
    """The rows and complete design of a model with covariates only.

    A weight model of a target trial adjusts for baseline covariates and has
    no exposure term; its rows follow the same rules as a primary model's,
    with ``outcome`` the 0/1 event the model's estimability is judged on.
    """

    return primary_model_rows(
        frame,
        terms=terms,
        exposure=None,
        outcome=outcome,
        missing_category_covariates=missing_category_covariates,
    )


def prepare_analysis_frame(frame, plan, *, run_dir=None):
    plan = (
        plan
        if isinstance(plan, DataPreparationPlan)
        else DataPreparationPlan.model_validate(plan)
    )
    raw_variables, status = set(), {}
    if not (plan.outliers or plan.windows or plan.unit_harmonisation):
        return frame.copy(), {"outliers": [], "windows": [], "unit_harmonisation": {}}
    if run_dir is not None:
        from easyicu.concept.provenance import ExportDataProvenance

        context = json.loads((Path(run_dir) / "research_context.json").read_text())
        captured = (context.get("cohort", {}).get("provenance") or {}).get(
            "data_provenance"
        )
        provenance = ExportDataProvenance.model_validate(captured) if captured else None
        for variable in context.get("variables", []):
            name = variable["name"]
            status[name] = database_preparation_status(
                provenance, variable.get("source_concept")
            )
            # Numeric materialized aggregates are not raw measurements. The
            # selected operation may run only on a captured raw-value column.
            is_raw = (
                variable.get("analysis_window") is None
                and variable.get("aggregation_default") is None
                and name == variable.get("source_concept")
            )
            if is_raw and source_has_raw_values(
                status[name], context.get("cohort", {}).get("database")
            ):
                raw_variables.add(name)
    prepared, outliers = apply_outliers(
        frame, plan.outliers, raw_variables=raw_variables
    )
    prepared, windows = aggregate_windows(
        prepared, plan.windows, raw_variables=raw_variables
    )
    return prepared, {
        "outliers": outliers,
        "windows": windows,
        "unit_harmonisation": status if plan.unit_harmonisation else {},
    }


def write_preparation_receipt(
    out_dir, *, plan, execution, missing_data=None, sensitivity=None
):
    """One step-local provenance owner; never rewrite immutable extraction metadata."""
    plan = (
        plan
        if isinstance(plan, DataPreparationPlan)
        else DataPreparationPlan.model_validate(plan)
    )
    missing = plan.missing_data
    from ..contracts.executed_method_design import (
        EXECUTED_METHOD_DESIGN_SCHEMA_VERSION,
        DataPreparationDesign,
    )

    design = DataPreparationDesign(
        schema_version=EXECUTED_METHOD_DESIGN_SCHEMA_VERSION,
        design_kind="data_preparation",
        preparation=plan,
        imputation_fit_scope="training_partition_only"
        if (missing_data or execution).get("fit_scope") == "training_partition_only"
        else "analysis_cohort",
        sensitivity_status=(sensitivity or {}).get("status"),
    )
    method = design.reader_text()
    payload = {
        "schema_version": "easyicu.data_preparation_provenance/1",
        "plan": plan.model_dump(mode="json"),
        "plan_sha256": canonical_sha256(plan.model_dump(mode="json")),
        "execution": execution,
        "missing_data": missing_data,
        "sensitivity": sensitivity,
        "methods_text": method,
        "executed_method_design": design.model_dump(mode="json"),
        "reporting_checklist": [
            {
                "item": "STROBE 12c",
                "status": "reported" if missing else "not_assessed",
                "method": method,
            },
            {
                "item": "TRIPOD+AI 7",
                "status": "reported" if missing else "not_assessed",
                "method": method,
            },
        ],
        "claim_ceiling": "analysis_only",
    }
    path = Path(out_dir) / "data_preparation_provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2))
    return payload


def _pool_association(fits, *, focal_term, kind):
    from ..methods.ordered_trends import OutcomeFit
    from ..robustness.estimators import EstimatorTerm

    if any(not fit.converged or fit.separation_detected for fit in fits):
        raise ImputationRefusal(
            "mi_analysis_fit_failed",
            "Every imputed analysis must converge without separation",
        )
    names = [term.term for term in fits[0].terms]
    if any(
        [term.term for term in fit.terms] != names or fit.n != fits[0].n for fit in fits
    ):
        raise ImputationRefusal(
            "mi_analysis_design_mismatch",
            "All imputations must fit the same rows and terms",
        )
    exponential = kind != "linear"
    pooled, terms = {}, []
    coefficients = []
    for index, name in enumerate(names):
        rows = [fit.terms[index] for fit in fits]
        if any(
            row.se is None or not math.isfinite(row.se) or row.se <= 0
            for row in rows
        ):
            raise ImputationRefusal(
                "mi_analysis_variance_unavailable",
                "Every imputed coefficient requires a finite positive within-imputation variance",
            )
        if any(
            row.estimate is None or not math.isfinite(row.estimate)
            or (exponential and row.estimate <= 0) for row in rows
        ):
            raise ImputationRefusal("mi_analysis_fit_failed", "An imputed coefficient is not poolable")
        estimates = [
            math.log(row.estimate) if exponential else row.estimate for row in rows
        ]
        coefficients.append(estimates)
        value = rubin_pool(
            estimates, [row.se**2 for row in rows],
            complete_df=min(
                fit.cluster_count - 1 if fit.variance_estimator == "cluster_robust"
                else fit.diagnostics["residual_df"] if isinstance(fit, OutcomeFit)
                else fit.n - len(names) for fit in fits
            ),
        )
        pooled[name] = value.to_json()
        scale = math.exp if exponential else float
        terms.append(
            EstimatorTerm(
                name,
                rows[0].source_variable,
                scale(value.estimate),
                scale(value.ci_low),
                scale(value.ci_high),
                value.standard_error,
            )
        )
    focal = terms[names.index(focal_term)]
    covariance = ()
    if all(getattr(fit, "coefficient_covariance", ()) for fit in fits):
        within = np.mean([fit.coefficient_covariance for fit in fits], axis=0)
        between = np.atleast_2d(np.cov(np.asarray(coefficients), ddof=1))
        covariance = tuple(map(tuple, within + (1 + 1 / len(fits)) * between))
    changes = dict(
        ci_low=focal.ci_low,
        ci_high=focal.ci_high,
        se=focal.se,
        terms=tuple(terms),
        notes="; ".join(dict.fromkeys([
            "Rubin-pooled stochastic chained-equation multiple imputation",
            *(fit.notes for fit in fits if fit.notes),
        ])),
    )
    if isinstance(fits[0], OutcomeFit):
        changes.update(estimate=focal.estimate, diagnostics={
            "multiple_imputation": {
                "diagnostics_scope": "per_imputation",
                "imputations": [fit.diagnostics for fit in fits],
            },
            "proportional_odds": {
                "violated": any(fit.diagnostics.get("proportional_odds", {}).get("violated", False) for fit in fits),
                "scope": "any_imputation",
            },
        })
    else:
        changes.update(point_estimate=focal.estimate, coefficient_covariance=covariance)
    return replace(fits[0], **changes), pooled


def fit_association_missing_data(
    frame,
    *,
    plan,
    terms,
    exposure,
    outcome,
    focal_term,
    kind,
    variance_estimator="model_based",
    cluster_groups=None,
    outcome_definition=None,
    fit_function=None,
):
    """Primary and complementary sensitivity use the same signed model roster."""
    from ..robustness.estimators import fit_estimator

    missing = plan.missing_data
    needed = [outcome, *[term.name for term in terms], *missing.auxiliaries]
    if outcome_definition is not None and outcome_definition.count:
        needed.extend(outcome_definition.input_columns)
        if missing.strategy == "multiple_imputation" and any(
            frame[column].isna().any() for column in outcome_definition.input_columns
        ):
            raise ImputationRefusal(
                "mi_analysis_coordinate_unobserved", "Count exposure/offset coordinates must be observed",
            )
    model_frame = frame[list(dict.fromkeys(needed))]
    imputed = (
        multiple_impute(model_frame, plan=missing, outcome=outcome)
        if missing.strategy == "multiple_imputation"
        else None
    )

    def fit(data):
        compiled = compile_prepared_terms(
            data, plan=plan, terms=terms, exposure=exposure, outcome=outcome
        )
        fitting_function = fit_function or fit_estimator
        options = {"definition": outcome_definition, "frame": data} if outcome_definition is not None else {"cohort": None}
        result = fitting_function(
            **options,
            X=compiled.design,
            y=data.loc[compiled.design.index, outcome],
            term=focal_term,
            kind=kind,
            source_by_design_column=compiled.source_by_design_column,
            variance_estimator=variance_estimator,
            cluster_groups=cluster_groups.loc[compiled.design.index]
            if cluster_groups is not None
            else None,
        )
        return compiled, result

    try:
        cc_compiled, cc_fit = fit(model_frame)
    except ValueError:
        if missing.strategy == "complete_case":
            raise
        from ..robustness.estimators import EstimatorResult

        cc_compiled, cc_fit = (
            None,
            EstimatorResult(
                None,
                None,
                None,
                None,
                0,
                False,
                "complete-case sensitivity not estimable",
            ),
        )
    sensitivity = {
        "strategy": "complete_case",
        "status": "fitted" if cc_fit.converged else "not_estimable",
        "n": cc_fit.n,
        "estimate": cc_fit.point_estimate,
        "ci_low": cc_fit.ci_low,
        "ci_high": cc_fit.ci_high,
    }
    try:
        if imputed is None:
            imputed = multiple_impute(model_frame, plan=missing, outcome=outcome)
        try:
            products = [fit(data) for data in imputed.datasets]
        except (ValueError, NotImplementedError) as exc:
            if isinstance(exc, PreparationRefusal):
                raise
            raise ImputationRefusal(
                "mi_analysis_fit_failed", "An imputed analysis cannot fit the declared model",
            ) from exc
        if any(product[1].n != len(model_frame) for product in products):
            raise ImputationRefusal(
                "mi_rows_dropped", "Imputed analyses may not drop incomplete rows"
            )
        pooled, pooling = _pool_association(
            [item[1] for item in products], focal_term=focal_term, kind=kind
        )
    except PreparationRefusal as exc:
        if missing.strategy == "multiple_imputation":
            raise
        return (
            cc_compiled,
            cc_fit,
            None,
            {
                "strategy": "multiple_imputation",
                "status": "not_estimable",
                "reason_code": exc.reason_code,
            },
        )
    diagnostics = {**imputed.diagnostics, "pooling": pooling}
    if missing.strategy == "complete_case":
        return (
            cc_compiled,
            cc_fit,
            diagnostics,
            {
                "strategy": "multiple_imputation",
                "status": "fitted",
                "n": pooled.n,
                "estimate": pooled.point_estimate,
                "ci_low": pooled.ci_low,
                "ci_high": pooled.ci_high,
            },
        )
    return products[0][0], pooled, diagnostics, sensitivity


def compile_prepared_terms(frame, *, plan, terms, exposure, outcome):
    """Compile declared references and expand selected covariate spline bases."""

    choices = {item.variable: item for item in plan.scaling_coding} if plan else {}
    for term in terms:
        choice = choices.get(term.name)
        if (
            choice
            and choice.reference_level is not None
            and choice.reference_level != term.reference_level
        ):
            raise PreparationRefusal(
                "preparation_coding_conflict",
                "The model reference must agree with the reviewed preparation choice",
            )
        if choice and choice.spline_knots and term.name == exposure:
            raise PreparationRefusal(
                "preparation_exposure_spline_requires_curve",
                "An exposure spline requires the dedicated dose-response result contract",
            )
    transformed, _, _ = standardize_columns(
        frame, frame, plan.scaling_coding if plan else ()
    )
    compiled = primary_model_rows(
        transformed, terms=terms, exposure=exposure, outcome=outcome
    )
    design = compiled.design.copy()
    source_map = dict(compiled.source_by_design_column)
    for name, choice in choices.items():
        if not choice.spline_knots:
            continue
        if name not in design:
            raise PreparationRefusal(
                "preparation_spline_type_unsupported",
                "Spline expansion requires a declared continuous covariate",
            )
        from ..methods.rcs_dose_response import rcs_basis

        basis = rcs_basis(
            transformed.loc[design.index, name].to_numpy(), knots=choice.spline_knots
        )
        design = design.drop(columns=name)
        source_map.pop(name, None)
        for index in range(basis.as_array().shape[1]):
            column = f"{name}__rcs_{index}"
            design[column] = basis.as_array()[:, index]
            source_map[column] = name
    return replace(compiled, design=design, source_by_design_column=source_map)


__all__ = [
    "CompiledModelMatrix",
    "MissingCategoryTerm",
    "ModelTermCompilationError",
    "PrimaryModelRows",
    "UnmeasuredRowsDropped",
    "compile_model_terms",
    "covariate_model_rows",
    "primary_model_rows",
]
