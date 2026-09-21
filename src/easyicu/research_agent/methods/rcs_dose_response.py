"""Deterministic restricted cubic spline (natural spline) dose-response kernel.

A restricted cubic spline (RCS; Harrell 2001) models a continuous exposure
``x`` with ``k`` knots ``t_1 < ... < t_k`` as an intercept plus a linear term
in ``x`` plus ``k - 2`` nonlinear basis functions

.. math::

    S_j(x) = \\frac{(x - t_j)^3_+ - (x - t_{k-1})^3_+ \\lambda_j
        + (x - t_k)^3_+ \\mu_j}{(t_k - t_1)^2},
    \\qquad j = 1, \\dots, k - 2,

where ``(u)_+ = max(u, 0)``,
``lambda_j = (t_k - t_j) / (t_k - t_{k-1})`` and
``mu_j = (t_{k-1} - t_j) / (t_k - t_{k-1})``.
The construction forces the fitted function to be exactly linear outside
the boundary knots (constant spline contribution below ``t_1``, linear
above ``t_k``); this is the natural-spline boundary constraint, enforced
mathematically by the basis definition, not approximately by a penalty.

Hand-roll justification (see :mod:`easyicu.research_agent.methods` rules 1
and 3): ``patsy.cr`` is a natural cubic spline in the mgcv parameterization
with its own knot convention, so it cannot serve a caller that must report
Harrell quantile knots and audit the basis term by term -- the
boundary-linearity claim, the knot placement, the fail-closed gates (no
global RNG, pinned seeds) and the digest-bound typed receipts all live
here, implemented directly in numpy. Cross-checks against ``patsy.cr`` at
identical knots assert span equality (see the kernel tests), not
parameterization equality.

Knot-selection policy (callers must report the knot locations alongside any
number this kernel returns; the typed results always carry ``knots``):

* Explicit ``knots`` take precedence and are validated fail-closed: 3--7
  knots, all finite, strictly increasing (duplicates rejected), and every
  knot inside the observed ``[min(x), max(x)]`` range (out-of-range knots
  are refused because tail linearity would then be anchored where no data
  were seen).
* Otherwise ``n_knots`` (default 4, allowed 3--7) knots are placed at the
  Harrell recommended quantiles of ``x``: 3 -> (0.10, 0.50, 0.90);
  4 -> (0.05, 0.35, 0.65, 0.95); 5 -> (0.05, 0.275, 0.50, 0.725, 0.95);
  6 -> (0.05, 0.23, 0.41, 0.59, 0.77, 0.95);
  7 -> (0.025, 0.1833, 0.3417, 0.50, 0.6583, 0.8167, 0.975).
  Quantile knots that are not finite or not strictly increasing (e.g. a
  discrete exposure with fewer distinct values than knots) are refused
  fail-closed instead of being silently deduplicated.

Determinism contract: the basis is pure numpy; the gaussian fit is OLS via
``numpy.linalg.lstsq``; the binomial fit is unpenalised sklearn logistic
regression with a pinned ``random_state``.  No global RNG is consumed, so
rerunning with identical inputs yields byte-identical
:meth:`RCSFitResult.to_json` output.

Fail-closed inputs: non-finite exposures/outcomes/covariates, empty inputs,
length mismatches, illegal knot counts/positions, rank-deficient designs,
non-binary or single-class binomial outcomes, unconverged logistic fits,
singular covariances, and out-of-range grid/level arguments all raise
:class:`RCSError`.

Claim ceiling: ``analysis_only``.  A smooth dose-response curve describes
association under the reported knot placement; it is not a causal finding
without a preregistered protocol and independent review.

References
----------
Harrell FE. *Regression Modeling Strategies*. Springer, 2001, ch. 2
(restricted cubic splines).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence
import warnings

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

from ..canonical_json import canonical_sha256

TOOL_VERSION = "1.0.0"
DEFAULT_RANDOM_STATE = 0
DEFAULT_N_KNOTS = 4
MIN_KNOTS = 3
MAX_KNOTS = 7
DEFAULT_GRID_POINTS = 100

#: Harrell recommended knot quantiles keyed by knot count.
KNOT_QUANTILES: Dict[int, tuple[float, ...]] = {
    3: (0.10, 0.50, 0.90),
    4: (0.05, 0.35, 0.65, 0.95),
    5: (0.05, 0.275, 0.50, 0.725, 0.95),
    6: (0.05, 0.23, 0.41, 0.59, 0.77, 0.95),
    7: (0.025, 0.1833, 0.3417, 0.50, 0.6583, 0.8167, 0.975),
}


class RCSError(ValueError):
    """An RCS input or configuration the kernel refuses to run on."""


def _truncated_cube(values: np.ndarray) -> np.ndarray:
    positive = np.where(values > 0.0, values, 0.0)
    return positive * positive * positive


def _rcs_columns(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Return the ``(n, k - 1)`` RCS design columns ``[x, S_1, ...]``."""
    n_knots = knots.shape[0]
    first, before_last, last = knots[0], knots[-2], knots[-1]
    scale = (last - first) ** 2
    columns = [np.asarray(x, dtype=float)]
    for position in range(n_knots - 2):
        knot_j = knots[position]
        lam = (last - knot_j) / (last - before_last)
        mu = (before_last - knot_j) / (last - before_last)
        basis = (
            _truncated_cube(x - knot_j)
            - _truncated_cube(x - before_last) * lam
            + _truncated_cube(x - last) * mu
        ) / scale
        columns.append(np.asarray(basis, dtype=float))
    return np.column_stack(columns)


def _column_names(n_knots: int) -> tuple[str, ...]:
    return ("x",) + tuple(f"s{index}" for index in range(1, n_knots - 1))


def _coerce_float_vector(values: object, *, field: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError):
        raise RCSError(f"{field} must be a numeric one-dimensional vector") from None
    if result.shape[0] == 0:
        raise RCSError(f"{field} must be non-empty")
    if not np.all(np.isfinite(result)):
        raise RCSError(f"{field} must contain only finite values (no NaN/inf)")
    return result


def _require_knot_count(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RCSError(f"{field} must be an int in [{MIN_KNOTS}, {MAX_KNOTS}]")
    result = int(value)
    if result < MIN_KNOTS or result > MAX_KNOTS:
        raise RCSError(f"{field} must be an int in [{MIN_KNOTS}, {MAX_KNOTS}]")
    return result


def _validate_knots(
    knots: np.ndarray, *, x_min: float, x_max: float, origin: str
) -> np.ndarray:
    if knots.ndim != 1:
        raise RCSError(f"{origin} knots must be one-dimensional")
    if knots.shape[0] < MIN_KNOTS or knots.shape[0] > MAX_KNOTS:
        raise RCSError(
            f"{origin} knots need {MIN_KNOTS}..{MAX_KNOTS} entries, "
            f"got {knots.shape[0]}"
        )
    if not np.all(np.isfinite(knots)):
        raise RCSError(f"{origin} knots must all be finite (no NaN/inf)")
    if not bool(np.all(np.diff(knots) > 0.0)):
        raise RCSError(
            f"{origin} knots must be strictly increasing "
            "(duplicates and reversals refused)"
        )
    if float(knots[0]) < x_min or float(knots[-1]) > x_max:
        raise RCSError(
            f"{origin} knots [{float(knots[0])}, {float(knots[-1])}] "
            f"fall outside the observed exposure range [{x_min}, {x_max}]"
        )
    return knots


@dataclass(frozen=True)
class RCSBasis:
    """Typed RCS basis: knot placement plus the evaluated design columns.

    The evaluation points travel with the basis so any consumer can prove
    the columns match the knots: ``__post_init__`` re-evaluates
    :func:`_rcs_columns` and refuses hand-assembled matrices (e.g. a
    rescaled column smuggled in with the original knots, which fits
    identically but predicts a different curve).
    """

    knots: tuple[float, ...]
    column_names: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]
    n_samples: int
    knot_source: str
    knot_quantiles: tuple[float, ...]
    x_points: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        points = np.asarray(self.x_points, dtype=float).ravel()
        if points.shape[0] != int(self.n_samples) or points.shape[0] != len(
            self.matrix
        ):
            raise RCSError("RCSBasis evaluation points do not match its matrix")
        expected = _rcs_columns(
            points, np.asarray(list(self.knots), dtype=float)
        )
        actual = np.asarray(
            [[float(value) for value in row] for row in self.matrix],
            dtype=float,
        )
        if expected.shape != actual.shape or not bool(
            np.array_equal(expected, actual)
        ):
            raise RCSError(
                "RCSBasis columns do not match re-evaluation at its knots; "
                "build bases with rcs_basis instead of assembling them by hand"
            )
        if tuple(self.column_names) != tuple(
            _column_names(len(list(self.knots)))
        ):
            raise RCSError("RCSBasis column names do not match its knots")

    def to_json(self) -> Dict[str, Any]:
        return {
            "tool_version": TOOL_VERSION,
            "knots": [float(value) for value in self.knots],
            "column_names": list(self.column_names),
            "matrix": [[float(value) for value in row] for row in self.matrix],
            "n_samples": int(self.n_samples),
            "knot_source": self.knot_source,
            "knot_quantiles": [float(value) for value in self.knot_quantiles],
            "x_points": [float(value) for value in self.x_points],
        }

    def as_array(self) -> np.ndarray:
        """Return the design columns as a float ``(n, k - 1)`` array."""
        return np.asarray(self.matrix, dtype=float)


@dataclass(frozen=True)
class RCSFitResult:
    """Typed RCS fit: coefficients, covariance, and information criteria."""

    family: str
    coef_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    std_errors: tuple[float, ...]
    vcov: tuple[tuple[float, ...], ...]
    knots: tuple[float, ...]
    covariate_names: tuple[str, ...]
    covariate_profile: tuple[float, ...]
    n_samples: int
    n_params: int
    df_resid: int
    rss: float | None
    loglik: float | None
    aic: float
    bic: float
    random_state: int
    basis_sha256: str = ""
    claim_ceiling: str = "analysis_only"
    method: str = "restricted_cubic_spline"
    limitations: tuple[str, ...] = (
        "analysis_only: association under the reported knot placement, "
        "not a causal dose-response finding",
        "knot count and positions are modelling choices and must be "
        "reported alongside any estimate",
        "linearity outside the boundary knots is an imposed constraint, "
        "not an empirical finding",
        "binomial standard errors and intervals are Wald/delta-method "
        "approximations",
    )

    def to_json(self) -> Dict[str, Any]:
        return {
            "tool_version": TOOL_VERSION,
            "family": self.family,
            "coef_names": list(self.coef_names),
            "coefficients": [float(value) for value in self.coefficients],
            "std_errors": [float(value) for value in self.std_errors],
            "vcov": [[float(value) for value in row] for row in self.vcov],
            "knots": [float(value) for value in self.knots],
            "covariate_names": list(self.covariate_names),
            "covariate_profile": [float(value) for value in self.covariate_profile],
            "n_samples": int(self.n_samples),
            "n_params": int(self.n_params),
            "df_resid": int(self.df_resid),
            "rss": None if self.rss is None else float(self.rss),
            "loglik": None if self.loglik is None else float(self.loglik),
            "aic": float(self.aic),
            "bic": float(self.bic),
            "random_state": int(self.random_state),
            "basis_sha256": str(self.basis_sha256),
            "claim_ceiling": self.claim_ceiling,
            "method": self.method,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class RCSNonlinearityResult:
    """Joint Wald test that all nonlinear spline terms are zero."""

    statistic: float
    df: int
    p_value: float
    nonlinear_terms: tuple[str, ...]
    n_samples: int
    claim_ceiling: str = "analysis_only"
    method: str = "joint_wald_chi2"

    def to_json(self) -> Dict[str, Any]:
        return {
            "tool_version": TOOL_VERSION,
            "statistic": float(self.statistic),
            "df": int(self.df),
            "p_value": float(self.p_value),
            "nonlinear_terms": list(self.nonlinear_terms),
            "n_samples": int(self.n_samples),
            "claim_ceiling": self.claim_ceiling,
            "method": self.method,
        }


@dataclass(frozen=True)
class RCSCurveResult:
    """Equidistant dose-response curve with delta-method intervals."""

    grid: tuple[float, ...]
    predicted: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    std_errors: tuple[float, ...]
    level: float
    response: str
    knots: tuple[float, ...]
    covariate_profile: tuple[float, ...]
    n_samples: int
    claim_ceiling: str = "analysis_only"
    method: str = "restricted_cubic_spline"

    def to_json(self) -> Dict[str, Any]:
        return {
            "tool_version": TOOL_VERSION,
            "grid": [float(value) for value in self.grid],
            "predicted": [float(value) for value in self.predicted],
            "lower": [float(value) for value in self.lower],
            "upper": [float(value) for value in self.upper],
            "std_errors": [float(value) for value in self.std_errors],
            "level": float(self.level),
            "response": self.response,
            "knots": [float(value) for value in self.knots],
            "covariate_profile": [float(value) for value in self.covariate_profile],
            "n_samples": int(self.n_samples),
            "claim_ceiling": self.claim_ceiling,
            "method": self.method,
        }


def result_sha256(
    result: RCSBasis | RCSFitResult | RCSNonlinearityResult | RCSCurveResult,
) -> str:
    """Return the canonical digest of one RCS typed result."""

    if not isinstance(
        result, (RCSBasis, RCSFitResult, RCSNonlinearityResult, RCSCurveResult)
    ):
        raise TypeError(
            "result_sha256 requires an RCSBasis, RCSFitResult, "
            "RCSNonlinearityResult or RCSCurveResult"
        )
    return canonical_sha256(result.to_json())


def rcs_basis(
    x: object,
    knots: Sequence[float] | None = None,
    n_knots: int = DEFAULT_N_KNOTS,
) -> RCSBasis:
    """Build the RCS design columns for exposure vector ``x``.

    Pass explicit ``knots`` (validated fail-closed against the observed
    range of ``x``) or let ``n_knots`` knots be placed at the Harrell
    recommended quantiles.  The returned columns are ``[x, s1, ...]`` (no
    intercept; :func:`rcs_fit` adds it).
    """

    values = _coerce_float_vector(x, field="x")
    x_min, x_max = float(values.min()), float(values.max())
    if knots is not None:
        try:
            knot_values = np.asarray(list(knots), dtype=float).ravel()
        except (TypeError, ValueError):
            raise RCSError("knots must be a numeric sequence") from None
        checked = _validate_knots(
            knot_values, x_min=x_min, x_max=x_max, origin="explicit"
        )
        source = "explicit"
        used_quantiles: tuple[float, ...] = ()
    else:
        count = _require_knot_count(n_knots, field="n_knots")
        used_quantiles = KNOT_QUANTILES[count]
        try:
            knot_values = np.quantile(values, np.asarray(used_quantiles, dtype=float))
        except (TypeError, ValueError) as exc:
            raise RCSError(f"quantile knots could not be computed: {exc}") from exc
        checked = _validate_knots(
            np.asarray(knot_values, dtype=float).ravel(),
            x_min=x_min,
            x_max=x_max,
            origin="quantile",
        )
        source = "quantile"
    matrix = _rcs_columns(values, checked)
    if not np.all(np.isfinite(matrix)):
        raise RCSError("RCS basis evaluation produced non-finite values")
    names = _column_names(checked.shape[0])
    return RCSBasis(
        knots=tuple(float(value) for value in checked),
        column_names=names,
        matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        n_samples=int(values.shape[0]),
        knot_source=source,
        knot_quantiles=tuple(float(value) for value in used_quantiles),
        x_points=tuple(float(value) for value in values.tolist()),
    )


def _coerce_covariates(covariates: object, n_samples: int) -> tuple[np.ndarray, tuple[str, ...]]:
    if covariates is None:
        return np.zeros((n_samples, 0), dtype=float), ()
    if isinstance(covariates, dict):
        raise RCSError("covariates must be an array-like matrix, not a mapping")
    try:
        import pandas as pd  # local import: pandas is optional for callers
    except ImportError:
        frame = None
    else:
        frame = covariates if isinstance(covariates, pd.DataFrame) else None
    if frame is not None:
        if frame.empty or frame.shape[0] != n_samples:
            raise RCSError(
                "covariates must align with the outcome "
                f"(rows {frame.shape[0]} != {n_samples})"
            )
        names = tuple(str(column) for column in frame.columns)
        try:
            values = frame.to_numpy(dtype=float)
        except (TypeError, ValueError):
            raise RCSError("covariates must be fully numeric") from None
    else:
        try:
            values = np.asarray(covariates, dtype=float)
        except (TypeError, ValueError):
            raise RCSError("covariates must be a numeric matrix") from None
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[0] != n_samples:
            raise RCSError(
                "covariates must align with the outcome "
                f"(shape {values.shape} incompatible with n={n_samples})"
            )
        names = tuple(f"c{index}" for index in range(values.shape[1]))
    if values.shape[1] == 0:
        return np.zeros((n_samples, 0), dtype=float), ()
    if not np.all(np.isfinite(values)):
        raise RCSError("covariates must contain only finite values (no NaN/inf)")
    if any(not name.strip() for name in names):
        raise RCSError("covariate names must all be non-blank")
    if len(set(names)) != len(names):
        raise RCSError("covariate names must be unique")
    return np.asarray(values, dtype=float), names


def _resolve_spline_matrix(
    x_spline: object, knots: Sequence[float] | None, n_samples: int
) -> tuple[np.ndarray, tuple[str, ...], tuple[float, ...]]:
    # Raw matrices are refused outright: a matrix detached from its knots
    # cannot prove its parameterization, and a rescaled column fits
    # identically while predicting a different curve. Only a validated
    # RCSBasis (whose __post_init__ re-evaluates the columns) is accepted.
    if knots is not None:
        raise RCSError(
            "knots must travel inside the RCSBasis; build it with rcs_basis "
            "instead of passing knots alongside the design"
        )
    if not isinstance(x_spline, RCSBasis):
        raise RCSError(
            "X_spline must be an RCSBasis from rcs_basis; raw matrices "
            "cannot prove the parameterization they claim"
        )
    matrix = x_spline.as_array()
    if matrix.shape[0] != n_samples:
        raise RCSError(
            f"X_spline rows {matrix.shape[0]} do not match y length {n_samples}"
        )
    return matrix, x_spline.column_names, x_spline.knots


def _ols_fit(
    design: np.ndarray, response: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    # LAPACK can set benign sticky FP flags (e.g. divide-by-zero inside
    # gelsd) that numpy would otherwise misattribute to a later matmul;
    # suppress that numerics noise for the whole helper while the explicit
    # rank/finiteness gates below stay fail-closed (same treatment as the
    # mediation kernel's OLS helpers).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        coef, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
        coef = np.asarray(coef, dtype=float)
        if rank < design.shape[1]:
            raise RCSError("regression design is rank-deficient; cannot identify effects")
        n, p = design.shape
        resid = response - design @ coef
        rss = float(resid @ resid)
        dof = n - p
        if dof < 1:
            raise RCSError("insufficient residual degrees of freedom for inference")
        sigma2 = rss / dof
        try:
            vcov = sigma2 * np.linalg.inv(design.T @ design)
        except np.linalg.LinAlgError as exc:
            raise RCSError("OLS covariance is singular") from exc
    if not np.all(np.isfinite(coef)) or not np.all(np.isfinite(vcov)):
        raise RCSError("OLS fit produced non-finite estimates")
    return coef, np.asarray(vcov, dtype=float), rss


def _logistic_fit(
    design: np.ndarray, response: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, float]:
    features = design[:, 1:]
    try:
        model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            random_state=seed,
            max_iter=5000,
        )
        model.fit(features, response)
    except Exception as exc:
        raise RCSError(f"logistic spline model failed to fit: {exc}") from exc
    n_iter = int(np.max(np.atleast_1d(model.n_iter_)))
    if n_iter >= 5000:
        raise RCSError("logistic spline model failed to converge; refusing to report")
    intercept = float(np.ravel(model.intercept_)[0])
    slopes = np.asarray(np.ravel(model.coef_), dtype=float)
    coef = np.concatenate([[intercept], slopes])
    if coef.shape[0] != design.shape[1] or not np.all(np.isfinite(coef)):
        raise RCSError("logistic spline model produced non-finite coefficients")
    linear = design @ coef
    positive = 1.0 / (1.0 + np.exp(-np.clip(linear, -500.0, 500.0)))
    weights = positive * (1.0 - positive)
    try:
        vcov = np.linalg.inv((design * weights[:, None]).T @ design)
    except np.linalg.LinAlgError as exc:
        raise RCSError(
            "logistic spline covariance is singular "
            "(near-separation or rank loss)"
        ) from exc
    if not np.all(np.isfinite(vcov)):
        raise RCSError("logistic spline covariance is non-finite")
    with np.errstate(divide="raise"):
        try:
            clipped = np.clip(positive, 1e-300, 1.0)
            loglik = float(
                np.sum(response * np.log(clipped) + (1.0 - response) * np.log1p(-positive))
            )
        except FloatingPointError as exc:
            raise RCSError("logistic log-likelihood is non-finite") from exc
    if not np.isfinite(loglik):
        raise RCSError("logistic log-likelihood is non-finite")
    return coef, np.asarray(vcov, dtype=float), loglik


def rcs_fit(
    y: object,
    X_spline: object,
    covariates: object = None,
    *,
    family: str = "gaussian",
    knots: Sequence[float] | None = None,
    covariate_names: Sequence[str] | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> RCSFitResult:
    """Fit the RCS dose-response model for outcome ``y``.

    ``X_spline`` must be an :class:`RCSBasis` from :func:`rcs_basis`
    (which carries its knots and proves its columns by re-evaluation).
    Raw matrices are refused: a matrix detached from its knots cannot
    prove its parameterization, and a rescaled column fits identically
    while predicting a different curve. ``covariates`` are optional
    additional linear adjusters. ``family="gaussian"`` fits OLS;
    ``family="binomial"`` fits unpenalised sklearn logistic regression
    with the pinned ``random_state``.
    """

    normalized = str(family or "").strip().lower()
    if normalized not in {"gaussian", "binomial"}:
        raise RCSError(f"family must be 'gaussian' or 'binomial', got {family!r}")
    if isinstance(random_state, bool) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise RCSError("random_state must be an integer")
    seed = int(random_state)
    if seed < 0:
        raise RCSError("random_state must be non-negative")

    y_values = _coerce_float_vector(y, field="y")
    n = int(y_values.shape[0])
    spline_matrix, spline_names, knot_tuple = _resolve_spline_matrix(
        X_spline, knots, n
    )
    cov_values, cov_names = _coerce_covariates(covariates, n)
    if covariate_names is not None:
        override = [str(name) for name in list(covariate_names)]
        if len(override) != cov_values.shape[1]:
            raise RCSError(
                f"covariate_names length {len(override)} does not match "
                f"covariates columns {cov_values.shape[1]}"
            )
        if any(not name.strip() for name in override):
            raise RCSError("covariate names must all be non-blank")
        if len(set(override)) != len(override):
            raise RCSError("covariate names must be unique")
        cov_names = tuple(override)
    overlap = set(spline_names) & set(cov_names)
    if overlap:
        raise RCSError(
            "covariate names collide with spline columns: " + ",".join(sorted(overlap))
        )

    if normalized == "binomial":
        if not np.isin(y_values, [0.0, 1.0]).all():
            raise RCSError("binomial outcome must take values in {0, 1}")
        if not (y_values == 1.0).any() or not (y_values == 0.0).any():
            raise RCSError("binomial outcome needs both classes observed")

    design = np.column_stack(
        [np.ones(n), spline_matrix] + ([cov_values] if cov_values.shape[1] > 0 else [])
    )
    coef_names = ("intercept",) + tuple(spline_names) + tuple(cov_names)
    if design.shape[0] - design.shape[1] < 1:
        raise RCSError("insufficient residual degrees of freedom for inference")

    if normalized == "gaussian":
        coef, vcov, rss = _ols_fit(design, y_values)
        loglik: float | None = -0.5 * n * (
            np.log(2.0 * np.pi) + 1.0 + np.log(rss / n)
        )
        aic = float(-2.0 * loglik + 2.0 * design.shape[1])
        bic = float(-2.0 * loglik + design.shape[1] * np.log(n))
    else:
        coef, vcov, loglik_value = _logistic_fit(design, y_values, seed)
        rss = None
        loglik = float(loglik_value)
        aic = float(-2.0 * loglik + 2.0 * design.shape[1])
        bic = float(-2.0 * loglik + design.shape[1] * np.log(n))

    variances = np.diag(vcov)
    if np.any(variances < 0.0) or not np.all(np.isfinite(variances)):
        raise RCSError("fit covariance has invalid variances")
    std_errors = tuple(float(np.sqrt(max(value, 0.0))) for value in variances)
    profile = (
        tuple(float(cov_values.mean(axis=0)[index]) for index in range(cov_values.shape[1]))
        if cov_values.shape[1] > 0
        else ()
    )
    basis_digest = canonical_sha256(
        {
            "knots": [float(value) for value in knot_tuple],
            "matrix": spline_matrix.tolist(),
        }
    )
    return RCSFitResult(
        family=normalized,
        coef_names=coef_names,
        coefficients=tuple(float(value) for value in coef),
        std_errors=std_errors,
        vcov=tuple(tuple(float(value) for value in row) for row in vcov),
        knots=knot_tuple,
        basis_sha256=basis_digest,
        covariate_names=tuple(cov_names),
        covariate_profile=profile,
        n_samples=n,
        n_params=int(design.shape[1]),
        df_resid=int(design.shape[0] - design.shape[1]),
        rss=None if rss is None else float(rss),
        loglik=None if loglik is None else float(loglik),
        aic=float(aic),
        bic=float(bic),
        random_state=seed,
    )


def nonlinearity_wald_test(fit: RCSFitResult) -> RCSNonlinearityResult:
    """Joint Wald chi-square test that all nonlinear spline terms are zero.

    The linear exposure term ``x`` stays in the null model; only
    ``s1..s{k-2}`` are tested, mirroring the OLS Wald covariance use in the
    mediation kernel (``sigma2 * inv(X'X)`` for gaussian; the inverse
    observed-information covariance for binomial).
    """

    if not isinstance(fit, RCSFitResult):
        raise TypeError("nonlinearity_wald_test requires an RCSFitResult")
    nonlinear = [
        name
        for name in fit.coef_names
        if name.startswith("s") and name[1:].isdigit()
    ]
    if not nonlinear:
        raise RCSError("fit carries no nonlinear spline terms to test")
    positions = [fit.coef_names.index(name) for name in nonlinear]
    coefs = np.asarray(fit.coefficients, dtype=float)[positions]
    vcov = np.asarray(fit.vcov, dtype=float)
    sub = vcov[np.ix_(positions, positions)]
    if not np.all(np.isfinite(sub)):
        raise RCSError("nonlinearity covariance block is non-finite")
    if float(np.max(np.abs(sub))) <= 0.0:
        # Exact fit: the covariance block is all zeros.  Any nonzero
        # nonlinear coefficient is then real signal, not sampling noise.
        if bool(np.any(np.abs(coefs) > 1e-8)):
            return RCSNonlinearityResult(
                statistic=float("inf"),
                df=len(nonlinear),
                p_value=0.0,
                nonlinear_terms=tuple(nonlinear),
                n_samples=int(fit.n_samples),
            )
        return RCSNonlinearityResult(
            statistic=0.0,
            df=len(nonlinear),
            p_value=1.0,
            nonlinear_terms=tuple(nonlinear),
            n_samples=int(fit.n_samples),
        )
    try:
        statistic = float(coefs @ np.linalg.solve(sub, coefs))
    except np.linalg.LinAlgError as exc:
        raise RCSError("nonlinearity covariance block is singular") from exc
    if not np.isfinite(statistic) or statistic < 0.0:
        raise RCSError("nonlinearity Wald statistic is non-finite")
    p_value = float(stats.chi2.sf(statistic, len(nonlinear)))
    if not np.isfinite(p_value):
        raise RCSError("nonlinearity Wald p-value is non-finite")
    return RCSNonlinearityResult(
        statistic=statistic,
        df=len(nonlinear),
        p_value=p_value,
        nonlinear_terms=tuple(nonlinear),
        n_samples=int(fit.n_samples),
    )


def predict_curve(
    fit: RCSFitResult,
    x_min: float,
    x_max: float,
    n_grid: int = DEFAULT_GRID_POINTS,
    level: float = 0.95,
    covariate_profile: Sequence[float] | None = None,
) -> RCSCurveResult:
    """Predict the dose-response curve on an equidistant grid.

    Covariates are held at ``covariate_profile`` (default: the estimation
    covariate means stored on the fit).  Intervals are delta-method
    Wald intervals: on the identity scale for gaussian fits, and on the
    probability scale (via the expit derivative) for binomial fits.
    """

    if not isinstance(fit, RCSFitResult):
        raise TypeError("predict_curve requires an RCSFitResult")
    try:
        lo, hi = float(x_min), float(x_max)
    except (TypeError, ValueError):
        raise RCSError("x_min and x_max must be finite numbers") from None
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise RCSError("x_min and x_max must be finite numbers")
    if not hi > lo:
        raise RCSError("x_max must be strictly greater than x_min")
    if isinstance(n_grid, bool) or not isinstance(n_grid, (int, np.integer)):
        raise RCSError("n_grid must be an integer >= 2")
    grid_count = int(n_grid)
    if grid_count < 2:
        raise RCSError("n_grid must be an integer >= 2")
    try:
        confidence = float(level)
    except (TypeError, ValueError):
        raise RCSError("level must lie strictly between 0 and 1") from None
    if not 0.0 < confidence < 1.0:
        raise RCSError("level must lie strictly between 0 and 1")

    if covariate_profile is None:
        profile = np.asarray(fit.covariate_profile, dtype=float)
    else:
        try:
            profile = np.asarray(list(covariate_profile), dtype=float).ravel()
        except (TypeError, ValueError):
            raise RCSError("covariate_profile must be numeric") from None
    if profile.shape[0] != len(fit.covariate_names):
        raise RCSError(
            f"covariate_profile length {profile.shape[0]} does not match "
            f"fit covariates {len(fit.covariate_names)}"
        )
    if profile.shape[0] > 0 and not np.all(np.isfinite(profile)):
        raise RCSError("covariate_profile must contain only finite values")

    grid = np.linspace(lo, hi, grid_count)
    spline_part = _rcs_columns(grid, np.asarray(fit.knots, dtype=float))
    design = np.column_stack(
        [np.ones(grid_count), spline_part]
        + ([np.tile(profile, (grid_count, 1))] if profile.shape[0] > 0 else [])
    )
    beta = np.asarray(fit.coefficients, dtype=float)
    vcov = np.asarray(fit.vcov, dtype=float)
    eta = design @ beta
    variances = np.einsum("ij,jk,ik->i", design, vcov, design)
    if not np.all(np.isfinite(variances)) or bool(np.any(variances < -1e-8)):
        raise RCSError("curve variances are non-finite")
    se_eta = np.sqrt(np.clip(variances, 0.0, None))
    z_value = float(stats.norm.ppf(0.5 + confidence / 2.0))
    if not np.isfinite(z_value):
        raise RCSError("curve critical value is non-finite")

    if fit.family == "gaussian":
        predicted = eta
        se_response = se_eta
        lower = eta - z_value * se_eta
        upper = eta + z_value * se_eta
        response = "identity"
    else:
        predicted = 1.0 / (1.0 + np.exp(-np.clip(eta, -500.0, 500.0)))
        se_response = predicted * (1.0 - predicted) * se_eta
        lower = np.clip(predicted - z_value * se_response, 0.0, 1.0)
        upper = np.clip(predicted + z_value * se_response, 0.0, 1.0)
        response = "probability"
    if not all(np.all(np.isfinite(part)) for part in (predicted, lower, upper)):
        raise RCSError("curve prediction produced non-finite values")
    return RCSCurveResult(
        grid=tuple(float(value) for value in grid),
        predicted=tuple(float(value) for value in predicted),
        lower=tuple(float(value) for value in lower),
        upper=tuple(float(value) for value in upper),
        std_errors=tuple(float(value) for value in se_response),
        level=confidence,
        response=response,
        knots=fit.knots,
        covariate_profile=tuple(float(value) for value in profile),
        n_samples=int(fit.n_samples),
    )


__all__ = [
    "DEFAULT_GRID_POINTS",
    "DEFAULT_N_KNOTS",
    "DEFAULT_RANDOM_STATE",
    "KNOT_QUANTILES",
    "MAX_KNOTS",
    "MIN_KNOTS",
    "TOOL_VERSION",
    "RCSBasis",
    "RCSCurveResult",
    "RCSError",
    "RCSFitResult",
    "RCSNonlinearityResult",
    "nonlinearity_wald_test",
    "predict_curve",
    "rcs_basis",
    "rcs_fit",
    "result_sha256",
]
