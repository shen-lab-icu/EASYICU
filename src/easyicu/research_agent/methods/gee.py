"""Marginal regression via Generalised Estimating Equations (analysis only).

Thin wrapper around :class:`statsmodels.genmod.generalized_estimating_equations.GEE`
(wrap-vs-rewrite rule: the package owns the estimator; this kernel only pins
the call, validates inputs fail-closed, and normalises a typed result plus a
canonical digest). Two outcome families are supported:

* ``family="binomial"`` → ``sm.families.Binomial()`` (logit link);
* ``family="gaussian"`` → ``sm.families.Gaussian()`` (identity link).

The within-cluster dependence is one of ``"exchangeable"``, ``"ar1"`` or
``"independence"``, mapped to the corresponding ``statsmodels`` covariance
structure. Standard errors are the robust sandwich estimator (the GEE
default ``cov_type="robust"``); p-values and confidence intervals use the
large-sample Normal approximation. Cluster labels are passed through
explicitly — the kernel never invents a grouping.

Determinism contract: GEE is an iteratively-reweighted fit from fixed
starting values with no RNG involved, so rerunning with identical inputs
yields byte-identical :meth:`GEEResult.to_json` output.

Fail-closed inputs (all raise :class:`GEEError`, a ``ValueError``):

* a ``family`` or ``cov_struct`` name outside the closed sets above;
* a non-binary outcome under ``family="binomial"`` (values must be exactly
  0/1 with both classes observed), or a constant outcome;
* fewer than two distinct clusters (a single cluster has no between-cluster
  variation for the sandwich variance to measure);
* ``time`` missing for ``cov_struct="ar1"`` (AR(1) needs an explicit
  within-cluster ordering), or ``time`` supplied for any other structure;
* length mismatches, non-finite values, or more parameters than observations.

Claim ceiling: ``analysis_only``. GEE estimates a population-averaged effect
under a working correlation assumption; it is not reportable evidence of a
causal effect without a preregistered protocol and independent review.

References
----------
Liang KY, Zeger SL. "Longitudinal data analysis using generalized linear
models." *Biometrika* 1986;73(1):13-22.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..canonical_json import canonical_sha256

TOOL_VERSION = "1.0.0"

_FAMILIES = ("binomial", "gaussian")
_COV_STRUCTS = ("exchangeable", "ar1", "independence")


class GEEError(ValueError):
    """A GEE input or configuration the kernel refuses to run on."""


def _require_ci_level(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise GEEError("ci_level must be a number in (0, 1)")
    level = float(value)
    if not 0.0 < level < 1.0:
        raise GEEError("ci_level must be strictly between 0 and 1")
    return level


def _coerce_outcome(y: object, family: str) -> np.ndarray:
    if isinstance(y, (pd.Series, pd.DataFrame)):
        if isinstance(y, pd.DataFrame) and y.shape[1] != 1:
            raise GEEError("y must be a one-dimensional outcome vector")
        values = y.to_numpy(dtype=float).ravel()
    else:
        try:
            values = np.asarray(y, dtype=float).ravel()
        except (TypeError, ValueError):
            raise GEEError(
                "y must be a numeric one-dimensional outcome vector"
            ) from None
    if values.shape[0] == 0:
        raise GEEError("y must be non-empty")
    if not np.all(np.isfinite(values)):
        raise GEEError("y must contain only finite values (no NaN/inf)")
    if family == "binomial":
        if not np.isin(values, [0.0, 1.0]).all():
            raise GEEError("binomial outcome must take values in {0, 1}")
        if not (values == 1.0).any() or not (values == 0.0).any():
            raise GEEError("binomial outcome needs both classes observed")
    elif float(np.std(values)) == 0.0:
        raise GEEError("gaussian outcome has no variation; effects are unidentified")
    return values


def _coerce_design(
    X: object, n_samples: int, feature_names: Sequence[str] | None
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
        if X.empty or X.shape[1] == 0:
            raise GEEError("X must be a non-empty design matrix")
        non_numeric = [
            str(column)
            for column, dtype in X.dtypes.items()
            if not pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
        ]
        if non_numeric:
            raise GEEError(
                "X must be fully numeric; non-numeric columns: "
                + ",".join(sorted(non_numeric))
            )
        frame = frame.astype(float)
        columns = [str(column) for column in frame.columns]
        if feature_names is not None:
            given = [str(name) for name in list(feature_names)]
            if len(given) != frame.shape[1]:
                raise GEEError(
                    f"feature_names length {len(given)} does not match "
                    f"X features {frame.shape[1]}"
                )
            frame.columns = given
            columns = given
    else:
        try:
            values = np.asarray(X, dtype=float)
        except (TypeError, ValueError):
            raise GEEError("X must be a numeric 2D array or DataFrame") from None
        if values.ndim != 2:
            raise GEEError("X must be a numeric 2D array or DataFrame")
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise GEEError("X must be a non-empty design matrix")
        if not np.all(np.isfinite(values)):
            raise GEEError("X must contain only finite values (no NaN/inf)")
        if feature_names is None:
            columns = [f"x{i}" for i in range(values.shape[1])]
        else:
            columns = [str(name) for name in list(feature_names)]
            if len(columns) != values.shape[1]:
                raise GEEError(
                    f"feature_names length {len(columns)} does not match "
                    f"X features {values.shape[1]}"
                )
        frame = pd.DataFrame(values, columns=columns)
    if frame.shape[0] != n_samples:
        raise GEEError(
            f"X rows {frame.shape[0]} do not match y length {n_samples}"
        )
    if any(not name.strip() for name in columns):
        raise GEEError("feature_names must all be non-blank")
    if len(set(columns)) != len(columns):
        raise GEEError("feature_names must be unique")
    return frame, tuple(columns)


def _coerce_clusters(clusters: object, n_samples: int) -> np.ndarray:
    labels = np.asarray(list(clusters), dtype=object)
    if labels.ndim != 1 or labels.shape[0] != n_samples:
        raise GEEError("clusters must be one-dimensional with the same length as y")
    distinct = set(labels.tolist())
    if len(distinct) < 2:
        raise GEEError(
            "GEE needs at least two distinct clusters; "
            f"got {len(distinct)} (a single cluster leaves the "
            "sandwich variance with no between-cluster variation)"
        )
    return labels


@dataclass(frozen=True)
class GEEResult:
    """Typed GEE fit output; JSON form is the digest input."""

    var_names: tuple[str, ...]
    coef: tuple[float, ...]
    se: tuple[float, ...]
    z: tuple[float, ...]
    p: tuple[float, ...]
    ci_low: tuple[float, ...]
    ci_high: tuple[float, ...]
    family: str
    cov_struct: str
    ci_level: float
    n_obs: int
    n_clusters: int
    converged: bool
    claim_ceiling: str = "analysis_only"
    method: str = "gee_statsmodels_robust"

    def to_json(self) -> Dict[str, Any]:
        return {
            "var_names": list(self.var_names),
            "coef": [float(value) for value in self.coef],
            "se": [float(value) for value in self.se],
            "z": [float(value) for value in self.z],
            "p": [float(value) for value in self.p],
            "ci_low": [float(value) for value in self.ci_low],
            "ci_high": [float(value) for value in self.ci_high],
            "family": self.family,
            "cov_struct": self.cov_struct,
            "ci_level": float(self.ci_level),
            "n_obs": int(self.n_obs),
            "n_clusters": int(self.n_clusters),
            "converged": bool(self.converged),
            "claim_ceiling": self.claim_ceiling,
            "method": self.method,
        }


def result_sha256(result: GEEResult) -> str:
    """Return the canonical digest of one GEE result."""

    if not isinstance(result, GEEResult):
        raise TypeError("result_sha256 requires a GEEResult")
    return canonical_sha256(result.to_json())


def fit_gee(
    y: object,
    X: object,
    clusters: object,
    *,
    family: str = "binomial",
    cov_struct: str = "exchangeable",
    feature_names: Sequence[str] | None = None,
    add_intercept: bool = True,
    time: object = None,
    ci_level: float = 0.95,
) -> GEEResult:
    """Fit a marginal GEE model and report robust Wald statistics.

    Parameters
    ----------
    y, X, clusters:
        Aligned outcome vector, numeric design matrix (DataFrame or 2D
        array), and explicit cluster labels (one per row).
    family:
        ``"binomial"`` (logit link) or ``"gaussian"`` (identity link).
    cov_struct:
        ``"exchangeable"``, ``"ar1"`` or ``"independence"``.
    add_intercept:
        Prepend a ``"const"`` column unless one is already present.
    time:
        Required for ``cov_struct="ar1"`` (within-cluster ordering);
        refused for any other structure so an ignored argument can never
        silently change meaning.
    """

    family_name = str(family or "").strip().lower()
    if family_name not in _FAMILIES:
        raise GEEError(f"family must be one of {_FAMILIES}, got {family!r}")
    struct_name = str(cov_struct or "").strip().lower()
    if struct_name not in _COV_STRUCTS:
        raise GEEError(f"cov_struct must be one of {_COV_STRUCTS}, got {cov_struct!r}")
    level = _require_ci_level(ci_level)

    y_values = _coerce_outcome(y, family_name)
    n = int(y_values.shape[0])
    exog_frame, _ = _coerce_design(X, n, feature_names)
    group_labels = _coerce_clusters(clusters, n)

    time_values = None
    if struct_name == "ar1":
        if time is None:
            raise GEEError(
                'cov_struct="ar1" requires an explicit time ordering; '
                "pass time=... (statsmodels will not invent one)"
            )
        try:
            parsed_time = np.asarray(list(time), dtype=float).ravel()
        except (TypeError, ValueError):
            raise GEEError("time must be a numeric vector") from None
        if parsed_time.shape[0] != n or not np.all(np.isfinite(parsed_time)):
            raise GEEError("time must be finite with the same length as y")
        time_values = parsed_time
    elif time is not None:
        raise GEEError(
            f"time is only meaningful with cov_struct='ar1', not {struct_name!r}; "
            "refusing an argument that would be silently ignored"
        )

    if add_intercept:
        exog_frame = sm.add_constant(exog_frame, has_constant="skip")
    names = [str(column) for column in exog_frame.columns]
    if len(set(names)) != len(names):
        raise GEEError("design columns must be uniquely named")
    if n <= len(names):
        raise GEEError(
            f"need more observations ({n}) than parameters ({len(names)})"
        )

    if family_name == "binomial":
        family_obj = sm.families.Binomial()
    else:
        family_obj = sm.families.Gaussian()
    if struct_name == "exchangeable":
        cov_obj = sm.cov_struct.Exchangeable()
    elif struct_name == "ar1":
        cov_obj = sm.cov_struct.Autoregressive()
    else:
        cov_obj = sm.cov_struct.Independence()

    try:
        fitted = sm.GEE(
            y_values,
            np.asarray(exog_frame, dtype=float),
            groups=group_labels,
            time=time_values,
            family=family_obj,
            cov_struct=cov_obj,
        ).fit()
    except Exception as exc:
        raise GEEError(f"statsmodels GEE fit failed: {exc}") from exc
    if not bool(fitted.converged):
        raise GEEError("statsmodels GEE fit did not converge; refusing to report")

    coef = np.asarray(fitted.params, dtype=float).ravel()
    stderr = np.asarray(fitted.bse, dtype=float).ravel()
    zvalues = np.asarray(fitted.tvalues, dtype=float).ravel()
    pvalues = np.asarray(fitted.pvalues, dtype=float).ravel()
    intervals = np.asarray(
        fitted.conf_int(alpha=1.0 - level), dtype=float
    ).reshape(len(names), 2)
    if not (
        np.all(np.isfinite(coef))
        and np.all(np.isfinite(stderr))
        and np.all(np.isfinite(zvalues))
        and np.all(np.isfinite(pvalues))
        and np.all(np.isfinite(intervals))
    ):
        raise GEEError("statsmodels GEE fit produced non-finite statistics")
    if np.any(stderr <= 0.0):
        raise GEEError("statsmodels GEE fit produced non-positive standard errors")

    return GEEResult(
        var_names=tuple(names),
        coef=tuple(float(value) for value in coef),
        se=tuple(float(value) for value in stderr),
        z=tuple(float(value) for value in zvalues),
        p=tuple(float(value) for value in pvalues),
        ci_low=tuple(float(value) for value in intervals[:, 0]),
        ci_high=tuple(float(value) for value in intervals[:, 1]),
        family=family_name,
        cov_struct=struct_name,
        ci_level=level,
        n_obs=n,
        n_clusters=len(set(group_labels.tolist())),
        converged=True,
    )


__all__ = [
    "TOOL_VERSION",
    "GEEError",
    "GEEResult",
    "fit_gee",
    "result_sha256",
]
