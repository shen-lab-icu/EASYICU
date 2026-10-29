"""Stochastic chained-equation multiple imputation and missing-data sensitivity.

``multiple_impute`` samples independent seeded PMM/logistic/polytomous chains;
``rubin_pool`` combines coefficient uncertainty with finite-sample degrees of
freedom and FMI. Only aggregate diagnostics are artifact payloads; completed
frames and training donor models remain local to the analysis owner.

``ridge_single_impute`` is the explicitly named deterministic single-imputation
helper. ``mice_impute`` remains a typed, deprecated alias of that helper and
never claims multiple-imputation uncertainty. Tipping-point sensitivity is
retained for callers that explicitly ask to vary missing predictor values.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import t

from ..contracts.model_terms import (
    ImputationVariable,
    MissingDataPlan,
    PreparationRefusal,
    level_spelling,
)

# ---------------------------------------------------------------------------
# Deterministic single imputation (legacy public names retained)
# ---------------------------------------------------------------------------


@dataclass
class RidgeSingleImputationResult:
    column: str
    n_imputed: int
    imputed_mean: float
    imputed_std: float
    n_iterations: int
    converged: bool

    def to_json(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "n_imputed": self.n_imputed,
            "imputed_mean": self.imputed_mean,
            "imputed_std": self.imputed_std,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "method": "deterministic_ridge_single_imputation",
            "multiple_imputation": False,
        }


def _ridge_fit(X: Any, y: Any, alpha: float = 1.0) -> Any:
    """Small ridge-regression helper without sklearn."""
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size or y.size == 0:
        raise ValueError("ridge fit requires aligned observed rows")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("ridge fit requires finite observed values")
    Xt = X.T
    k = X.shape[1]
    beta = np.linalg.solve(Xt @ X + alpha * np.eye(k), Xt @ y)
    return beta


def ridge_single_impute(
    *,
    column: str,
    target: Sequence[float],
    predictors: Sequence[Sequence[float]],
    max_iter: int = 10,
    tol: float = 1e-4,
) -> Tuple[List[float], RidgeSingleImputationResult]:
    """Single ridge fit on observed targets, predicting only missing targets.

    ``max_iter`` and ``tol`` are retained for call compatibility; the fit is
    closed form, not iterative. ``converged`` denotes a finite completed fit,
    and ``imputed_std`` is the spread of filled values, not inferential error.
    """
    if np is None:
        raise RuntimeError("ridge_single_impute requires numpy")
    y = np.asarray(target, dtype=float)
    X = np.asarray(predictors, dtype=float)
    if y.ndim != 1 or X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("target must be 1D and predictors 2D with matching rows")
    if not isinstance(max_iter, int) or max_iter < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("max_iter and tol must be positive")
    if np.isinf(y).any() or not np.isfinite(X).all():
        raise ValueError("observed target and all predictors must be finite")
    missing = np.isnan(y)
    if not (~missing).any():
        raise ValueError("imputation requires at least one observed target")
    if not missing.any():
        return list(y), RidgeSingleImputationResult(
            column=column,
            n_imputed=0,
            imputed_mean=float("nan"),
            imputed_std=float("nan"),
            n_iterations=0,
            converged=True,
        )
    y_filled = y.copy()
    X_design = np.column_stack([np.ones(X.shape[0]), X])
    beta = _ridge_fit(X_design[~missing], y[~missing])
    pred = X_design[missing] @ beta
    if not np.isfinite(pred).all():
        raise ValueError("ridge imputation produced non-finite predictions")
    y_filled[missing] = pred
    result = RidgeSingleImputationResult(
        column=column,
        n_imputed=int(missing.sum()),
        imputed_mean=float(np.mean(pred)),
        imputed_std=float(np.std(pred)),
        n_iterations=1,
        converged=True,
    )
    return list(y_filled), result


class LegacySingleImputationWarning(DeprecationWarning):
    """The legacy MICE name supplies no multiple-imputation uncertainty."""

    reason_code = "legacy_single_imputation_not_mi"


MICEImputationResult = RidgeSingleImputationResult


def mice_impute(**kwargs) -> Tuple[List[float], RidgeSingleImputationResult]:
    """Deprecated alias; use ridge_single_impute or the real MI kernel."""
    warnings.warn(
        "mice_impute is deterministic single imputation; use ridge_single_impute "
        "for that operation, or methods.missing_data.multiple_impute for MI.",
        LegacySingleImputationWarning, stacklevel=2,
    )
    return ridge_single_impute(**kwargs)


# ---------------------------------------------------------------------------
# Tipping-point analysis
# ---------------------------------------------------------------------------


@dataclass
class TippingPointResult:
    column: str
    baseline_or: Optional[float]
    grid: List[float] = field(default_factory=list)
    or_by_imputed_value: List[Optional[float]] = field(default_factory=list)
    tipping_point: Optional[float] = None
    note: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "baseline_or": self.baseline_or,
            "grid": list(self.grid),
            "or_by_imputed_value": list(self.or_by_imputed_value),
            "tipping_point": self.tipping_point,
            "note": self.note,
        }


def _logistic_or(x: Any, y: Any) -> Optional[float]:
    """Single-predictor logistic OR via IRLS. Returns None on failure."""
    if np is None:
        return None
    n = len(x)
    if n < 30 or len(np.unique(y)) < 2:
        return None
    X = np.column_stack([np.ones(n), x])
    beta = np.zeros(2)
    for _ in range(50):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -40, 40)))
        W = p * (1 - p)
        XtWX = X.T @ (W[:, None] * X)
        try:
            inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            return None
        z = eta + (y - p) / np.maximum(W, 1e-9)
        new_beta = inv @ (X.T @ (W * z))
        if np.max(np.abs(new_beta - beta)) < 1e-6:
            beta = new_beta
            break
        beta = new_beta
    return float(math.exp(beta[1]))


def tipping_point_analysis(
    *,
    predictor_column: str,
    predictor_values: Sequence[float],
    outcome: Sequence[int],
    missing_mask: Sequence[bool],
    grid: Optional[Sequence[float]] = None,
) -> TippingPointResult:
    """Sweep the imputed value for the predictor's missing rows.

    For each value ``v`` in ``grid`` (defaults to
    ``[-4σ, -2σ, 0, +2σ, +4σ]`` around the observed mean), set all
    ``missing_mask == True`` rows to ``v`` and refit a tiny logistic
    regression of ``outcome ~ predictor``. The tipping point is the
    smallest absolute ``v`` at which the OR crosses 1.0.
    """
    if np is None:
        return TippingPointResult(
            column=predictor_column, baseline_or=None, note="numpy unavailable"
        )
    x = np.asarray(predictor_values, dtype=float)
    y = np.asarray(outcome, dtype=int)
    m = np.asarray(missing_mask, dtype=bool)
    if not m.any():
        return TippingPointResult(
            column=predictor_column,
            baseline_or=_logistic_or(x, y),
            note="no missing rows; tipping-point analysis skipped",
        )
    obs = x[~m]
    if len(obs) < 30:
        return TippingPointResult(
            column=predictor_column,
            baseline_or=None,
            note="too few observed rows for tipping-point analysis",
        )
    mean = float(np.mean(obs))
    sigma = float(np.std(obs)) or 1.0
    if grid is None:
        grid = [mean - 4 * sigma, mean - 2 * sigma, mean, mean + 2 * sigma, mean + 4 * sigma]
    else:
        grid = list(grid)
    ors: List[Optional[float]] = []
    baseline_or = _logistic_or(x[~m], y[~m])
    tipping: Optional[float] = None
    for v in grid:
        x_v = x.copy()
        x_v[m] = v
        orv = _logistic_or(x_v, y)
        ors.append(orv)
        if baseline_or is not None and orv is not None:
            if tipping is None and (
                (baseline_or > 1 and orv < 1) or (baseline_or < 1 and orv > 1)
            ):
                tipping = float(v)
    return TippingPointResult(
        column=predictor_column,
        baseline_or=baseline_or,
        grid=list(grid),
        or_by_imputed_value=ors,
        tipping_point=tipping,
    )


class ImputationRefusal(PreparationRefusal):
    pass


@dataclass(frozen=True)
class ColumnEncoding:
    name: str
    levels: tuple | None
    mean: float = 0.0
    scale: float = 1.0


def encodings_for(frame: pd.DataFrame, variables: dict[str, ImputationVariable]):
    encodings = []
    for name in frame:
        observed = frame[name].dropna()
        if len(observed) < 10:
            raise ImputationRefusal(
                "mi_insufficient_observations",
                "At least ten observed values are required per variable",
            )
        spec = variables.get(name)
        categorical = spec is not None and spec.method != "pmm"
        if categorical or not pd.api.types.is_numeric_dtype(observed):
            if spec is not None and spec.method == "pmm":
                raise ImputationRefusal(
                    "mi_variable_type_unsupported", "PMM requires a numeric variable"
                )
            actual = tuple(sorted(observed.unique(), key=str))
            if spec and spec.levels:
                by_spelling = {level_spelling(value): value for value in actual}
                if len(by_spelling) != len(actual) or set(spec.levels) != set(
                    by_spelling
                ):
                    raise ImputationRefusal(
                        "mi_variable_type_unsupported",
                        "Categorical values must match the declared level identities",
                    )
                levels = tuple(by_spelling[value] for value in spec.levels)
            else:
                levels = actual
            if not observed.isin(levels).all() or len(levels) < 2:
                raise ImputationRefusal(
                    "mi_variable_type_unsupported",
                    "Categorical values must match a closed level set",
                )
            if spec and spec.method == "logistic" and len(levels) != 2:
                raise ImputationRefusal(
                    "mi_variable_type_unsupported",
                    "Logistic imputation requires two levels",
                )
            if spec and spec.method == "polytomous" and len(levels) < 3:
                raise ImputationRefusal(
                    "mi_variable_type_unsupported",
                    "Polytomous imputation requires at least three observed training levels",
                )
            encodings.append(ColumnEncoding(name, levels))
        else:
            values = observed.to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ImputationRefusal(
                    "mi_nonfinite_value", "Observed numeric values must be finite"
                )
            scale = float(values.std())
            encodings.append(
                ColumnEncoding(name, None, float(values.mean()), scale or 1.0)
            )
    return tuple(encodings)


def design(frame: pd.DataFrame, encodings: tuple[ColumnEncoding, ...], target: str):
    columns = [np.ones(len(frame))]
    for item in encodings:
        if item.name == target:
            continue
        values = frame[item.name]
        if item.levels is None:
            columns.append((values.to_numpy(dtype=float) - item.mean) / item.scale)
        else:
            if not values.isin(item.levels).all():
                raise ImputationRefusal(
                    "mi_unseen_category", "Scoring values lie outside training levels"
                )
            columns.extend(
                values.eq(level).to_numpy(dtype=float) for level in item.levels[1:]
            )
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise ImputationRefusal(
            "mi_nonfinite_value", "Imputation design must be finite"
        )
    return matrix


@dataclass(frozen=True)
class ConditionalDraw:
    target: str
    method: str
    encodings: tuple[ColumnEncoding, ...]
    coefficients: np.ndarray
    donor_predictions: np.ndarray | None = None
    donor_values: np.ndarray | None = None
    levels: tuple = ()

    def sample(self, frame, rows, rng, donors):
        matrix = design(frame.loc[rows], self.encodings, self.target)
        if self.method == "pmm":
            predictions = matrix @ self.coefficients
            values = np.empty(len(predictions))
            k = min(donors, len(self.donor_values))
            # Bound the temporary distance matrix independently of cohort size.
            for start in range(0, len(predictions), 64):
                distance = np.abs(
                    predictions[start : start + 64, None] - self.donor_predictions
                )
                nearest = np.argpartition(distance, k - 1, axis=1)[:, :k]
                chosen = nearest[
                    np.arange(len(nearest)), rng.integers(k, size=len(nearest))
                ]
                values[start : start + len(nearest)] = self.donor_values[chosen]
            return values
        eta = matrix @ self.coefficients
        if eta.ndim == 1:
            eta = eta[:, None]
        probabilities = softmax(np.column_stack([np.zeros(len(matrix)), eta]), axis=1)
        sampled = (
            rng.random(len(matrix))[:, None] > np.cumsum(probabilities, axis=1)
        ).sum(axis=1)
        return np.asarray(self.levels)[np.minimum(sampled, len(self.levels) - 1)]


def fit_conditional(frame, observed, spec, encodings, rng):
    matrix = design(frame, encodings, spec.name)
    x = matrix[observed]
    if len(x) < max(10, x.shape[1] + 3):
        raise ImputationRefusal(
            "mi_insufficient_observations",
            "Too few observed values for the declared conditional model",
        )
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ImputationRefusal(
            "mi_rank_deficient", "Conditional design is rank deficient"
        )
    target = frame.loc[observed, spec.name]
    try:
        if spec.method == "pmm":
            y = target.to_numpy(dtype=float)
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            df = len(y) - x.shape[1]
            residual_ss = float(np.square(y - x @ beta).sum())
            if residual_ss <= np.finfo(float).eps:
                raise ImputationRefusal(
                    "mi_degenerate_conditional",
                    "Conditional model has no residual uncertainty",
                )
            sigma2 = residual_ss / rng.chisquare(df)
            draw = rng.multivariate_normal(
                beta, sigma2 * np.linalg.inv(x.T @ x), check_valid="raise"
            )
            return ConditionalDraw(spec.name, spec.method, encodings, draw, x @ beta, y)
        import statsmodels.api as sm

        encoding = next(item for item in encodings if item.name == spec.name)
        levels = encoding.levels
        if levels is None or min(int(target.eq(level).sum()) for level in levels) < 5:
            raise ImputationRefusal(
                "mi_insufficient_category_support",
                "Each declared category requires at least five observed values",
            )
        y = pd.Categorical(target, categories=levels).codes
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = sm.Logit(y, x) if spec.method == "logistic" else sm.MNLogit(y, x)
            fitted = model.fit(disp=False, maxiter=100)
        if caught or not fitted.mle_retvals.get("converged", False):
            raise ImputationRefusal(
                "mi_conditional_fit_failed",
                "Categorical conditional did not converge without warnings",
            )
        beta, covariance = np.asarray(fitted.params), np.asarray(fitted.cov_params())
        if (
            not np.isfinite(covariance).all()
            or not np.isfinite(beta).all()
            or np.max(np.abs(beta)) > 25
        ):
            raise ImputationRefusal(
                "mi_conditional_fit_failed",
                "Categorical conditional is separated or nonfinite",
            )
        draw = rng.multivariate_normal(
            beta.ravel(order="F"), covariance, check_valid="raise"
        )
        return ConditionalDraw(
            spec.name,
            spec.method,
            encodings,
            draw.reshape(beta.shape, order="F"),
            levels=levels,
        )
    except ImputationRefusal:
        raise
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
        raise ImputationRefusal(
            "mi_conditional_fit_failed",
            "The declared conditional model could not be sampled",
        ) from exc


@dataclass(frozen=True)
class RubinResult:
    estimate: float
    within_variance: float
    between_variance: float
    total_variance: float
    df: float
    fmi: float
    standard_error: float
    ci_low: float
    ci_high: float
    p_value: float

    def to_json(self):
        return {
            key: value if math.isfinite(value) else None
            for key, value in asdict(self).items()
        }


def rubin_pool(estimates, variances, *, complete_df=math.inf):
    """Pool on the coefficient (not odds/hazard ratio) scale; 95% t intervals.

    With complete-data residual df, use Barnard–Rubin small-sample adjustment.
    ``variances`` are squared coefficient standard errors, never their SDs.
    """
    q, u = np.asarray(estimates, float), np.asarray(variances, float)
    if (
        q.ndim != 1
        or u.shape != q.shape
        or len(q) < 2
        or not np.isfinite(q).all()
        or not np.isfinite(u).all()
        or (u <= 0).any()
        or math.isnan(complete_df)
        or complete_df <= 0
    ):
        raise ImputationRefusal(
            "mi_pool_invalid",
            "Pooling requires at least two aligned finite estimates and positive variances",
        )
    m = len(q)
    mean, within, between = float(q.mean()), float(u.mean()), float(q.var(ddof=1))
    added = (1 + 1 / m) * between
    total = within + added
    r = added / within
    df_old = (m - 1) * (1 + 1 / r) ** 2 if r else math.inf
    df = df_old
    if math.isfinite(complete_df):
        observed_df = (
            (complete_df + 1) / (complete_df + 3) * complete_df * (1 - added / total)
        )
        df = 1 / (1 / df_old + 1 / observed_df)
    fmi = (r + 2 / (df + 3)) / (r + 1)
    se = math.sqrt(total)
    critical = float(t.ppf(0.975, df))
    return RubinResult(
        mean,
        within,
        between,
        total,
        df,
        fmi,
        se,
        mean - critical * se,
        mean + critical * se,
        float(2 * t.sf(abs(mean / se), df)),
    )


def _summary(values):
    values = pd.Series(values).dropna()
    if pd.api.types.is_numeric_dtype(values):
        return {
            "count": len(values),
            "mean": float(values.mean()) if len(values) else None,
            "sd": float(values.std(ddof=0)) if len(values) else None,
        }
    return {
        "count": len(values),
        "levels": {
            str(key): int(count) for key, count in values.value_counts().items()
        },
    }


def _initialize(frame, rng):
    completed = frame.copy(deep=True)
    for name in frame:
        missing = frame[name].isna()
        if missing.any():
            if isinstance(completed[name].dtype, pd.CategoricalDtype):
                completed[name] = completed[name].astype(object)
            completed.loc[missing, name] = rng.choice(
                frame.loc[~missing, name].to_numpy(), size=int(missing.sum())
            )
    return completed


@dataclass
class MultipleImputationResult:
    datasets: tuple[pd.DataFrame, ...]
    diagnostics: dict
    _training: pd.DataFrame = field(repr=False)
    _draws: tuple = field(repr=False)
    _plan: MissingDataPlan = field(repr=False)
    _outcome: str = field(repr=False)

    def transform(self, frame: pd.DataFrame):
        """Score using training donors/parameters; held-out outcomes are forbidden.

        The outcome is latent at scoring. Its conditional is sampled from the
        training model, never populated with the validation labels. No fitting,
        vocabulary discovery or donor selection uses validation observations.
        """
        if self._outcome in frame:
            raise ImputationRefusal(
                "mi_fold_outcome_leakage",
                "Held-out outcomes must not enter the imputation transform",
            )
        if set(frame) != set(self._training) - {self._outcome}:
            raise ImputationRefusal(
                "mi_scoring_columns_mismatch",
                "Scoring predictors must match the training imputation model",
            )
        if set(frame.columns[frame.isna().any()]) - {
            item.name for item in self._plan.variables
        }:
            raise ImputationRefusal(
                "mi_method_missing",
                "Every missing scoring predictor needs a declared conditional",
            )
        scoring = frame.reindex(columns=self._training.columns).copy()
        results = []
        for chain, sweeps in enumerate(self._draws):
            rng = np.random.default_rng(
                np.random.SeedSequence([self._plan.seed, chain, 991])
            )
            completed = scoring.copy()
            missing = completed.isna()
            for name in completed:
                if missing[name].any():
                    completed.loc[missing[name], name] = rng.choice(
                        self._training[name].dropna().to_numpy(),
                        size=int(missing[name].sum()),
                    )
                completed[name] = completed[name].astype(self._training[name].dtype)
            for draws in sweeps * self._plan.iterations:
                for draw in draws:
                    mask = missing[draw.target]
                    if mask.any():
                        completed.loc[mask, draw.target] = draw.sample(
                            completed, mask, rng, self._plan.donors
                        )
            if completed.isna().any().any():
                raise ImputationRefusal(
                    "mi_scoring_incomplete", "Scoring imputation left missing values"
                )
            results.append(completed.drop(columns=self._outcome))
        return tuple(results)


def multiple_impute(
    frame: pd.DataFrame,
    *,
    plan: MissingDataPlan,
    outcome: str,
    fit_for_prediction: bool = False,
) -> MultipleImputationResult:
    """Run independent stochastic FCS chains with the outcome and auxiliaries.

    Missing outcomes are refused for inference. For survival the runner supplies
    the event indicator and Nelson–Aalen cumulative hazard as predictors.
    Every missing column must have a declared method; no dropped-column fallback.
    """
    if outcome not in frame or frame[outcome].isna().any():
        raise ImputationRefusal(
            "mi_outcome_missing",
            "The observed outcome must be included in the imputation model",
        )
    if not frame.columns.is_unique or not frame.index.is_unique:
        raise ImputationRefusal(
            "mi_identity_invalid", "Imputation columns and row indices must be unique"
        )
    variables = {item.name: item for item in plan.variables}
    if outcome in variables:
        raise ImputationRefusal(
            "mi_outcome_imputation_unsupported",
            "Inference requires an observed outcome",
        )
    if not set(variables).issubset(frame) or not set(plan.auxiliaries).issubset(frame):
        raise ImputationRefusal(
            "mi_column_missing", "A declared imputation variable or auxiliary is absent"
        )
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ImputationRefusal(
            "mi_nonfinite_value", "Infinite numeric values cannot enter imputation"
        )
    missing = frame.isna()
    if set(frame.columns[missing.any()]) - set(variables):
        raise ImputationRefusal(
            "mi_method_missing",
            "Every missing predictor needs a declared imputation method",
        )
    encodings = encodings_for(frame, variables)
    conditionals = list(plan.variables)
    if fit_for_prediction:
        observed = frame[outcome]
        method = "logistic" if set(observed.unique()) == {0, 1} else "pmm"
        conditionals.append(ImputationVariable(name=outcome, method=method))
        # Outcome is categorical for its scoring conditional, even though fully
        # observed in training. It remains a predictor of the training draws.
        variables[outcome] = conditionals[-1]
        encodings = encodings_for(frame, variables)
    datasets, chain_draws, traces = [], [], []
    for chain, stream in enumerate(np.random.SeedSequence(plan.seed).spawn(plan.m)):
        rng = np.random.default_rng(stream)
        completed = _initialize(frame, rng)
        sweeps = []
        for iteration in range(plan.iterations):
            draws = []
            for spec in conditionals:
                mask = missing[spec.name]
                if not mask.any() and not fit_for_prediction:
                    continue
                draw = fit_conditional(
                    completed, ~mask.to_numpy(), spec, encodings, rng
                )
                if mask.any():
                    completed.loc[mask, spec.name] = draw.sample(
                        completed, mask, rng, plan.donors
                    )
                draws.append(draw)
            if fit_for_prediction:
                # Keep the last frozen conditional draws, rather than retaining
                # donors for every burn-in sweep on a large cohort.
                sweeps = [tuple(draws)]
            traces.append(
                {
                    "chain": chain + 1,
                    "iteration": iteration + 1,
                    "variables": {
                        name: _summary(completed.loc[missing[name], name])
                        for name in variables
                        if missing[name].any()
                    },
                }
            )
        datasets.append(completed)
        chain_draws.append(tuple(sweeps))
    patterns = missing.astype(int).groupby(list(frame.columns), dropna=False).size()
    diagnostics = {
        "method": "stochastic_fcs_pmm_logistic_polytomous",
        "backend": "python",
        "m": plan.m,
        "seed": plan.seed,
        "iterations": plan.iterations,
        "donors": plan.donors,
        "methods": {name: spec.method for name, spec in variables.items()},
        "auxiliaries": list(plan.auxiliaries),
        "outcome_in_model": outcome,
        "n": len(frame),
        "convergence": traces,
        "missingness_patterns": [
            {
                "missing_variables": [name for name, bit in zip(frame, key) if bit],
                "count": int(count),
            }
            for key, count in patterns.items()
        ],
        "distributions": {
            name: {
                "observed": _summary(frame[name]),
                "imputed": _summary(
                    pd.concat([data.loc[missing[name], name] for data in datasets])
                ),
            }
            for name in variables
            if missing[name].any()
        },
    }
    return MultipleImputationResult(
        tuple(datasets), diagnostics, frame.copy(), tuple(chain_draws), plan, outcome
    )


__all__ = [
    "RidgeSingleImputationResult",
    "LegacySingleImputationWarning",
    "ridge_single_impute",
    "MICEImputationResult",
    "TippingPointResult",
    "mice_impute",
    "tipping_point_analysis",
    "ImputationRefusal",
    "MultipleImputationResult",
    "multiple_impute",
    "rubin_pool",
]
