"""Linear mixed model with a random intercept (analysis only).

Thin wrapper around :class:`statsmodels.regression.mixed_linear_model.MixedLM`
(wrap-vs-rewrite rule: the package owns the estimator; this kernel only pins
the call, validates inputs fail-closed, and normalises a typed result plus a
canonical digest). The model for a continuous outcome ``y`` in group ``g`` is::

    y_ig = x_ig @ beta + u_g + e_ig,  u_g ~ N(0, tau^2),  e_ig ~ N(0, sigma^2),

i.e. a random intercept per group and fixed slopes. Only this specification
is offered: random slopes need a separate validated kernel.

Determinism contract: ``MixedLM`` maximises the (restricted) likelihood from
fixed starting values with a deterministic optimiser and no RNG, so
rerunning with identical inputs yields byte-identical
:meth:`MixedEffectsResult.to_json` output.

Fail-closed inputs (all raise :class:`MixedEffectsError`, a ``ValueError``):

* fewer than five distinct groups. With a handful of groups the
  between-group variance ``tau^2`` is not identified in practice: the
  estimate piles onto the ``tau^2 = 0`` boundary, its standard error is
  meaningless, and the fixed-effect standard errors that depend on it
  cannot be trusted (multilevel guidance commonly asks for at least ten
  groups — Maas & Hox 2005 — so five is already a floor, not a target).
  The kernel refuses instead of warning because a warning is trivially
  ignored downstream while the numbers keep their authoritative shape;
* non-finite inputs, length mismatches, a constant outcome, or more fixed
  parameters than observations;
* an optimiser that reports non-convergence (reporting unconverged
  variance components as if they were estimates would be fabrication).

Claim ceiling: ``analysis_only``. Variance-component estimates are not
reportable evidence of heterogeneity without a preregistered protocol and
independent review.

References
----------
Laird NM, Ware JH. "Random-effects models for longitudinal data."
*Biometrics* 1982;38(4):963-974.
Maas CJM, Hox JJ. "Sufficient sample sizes for multilevel modeling."
*Methodology* 2005;1(3):86-92.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..canonical_json import canonical_sha256

TOOL_VERSION = "1.0.0"

#: Minimum number of groups. See the module docstring for why this is a
#: refusal rather than a warning.
MIN_GROUPS = 5


class MixedEffectsError(ValueError):
    """A mixed-effects input or configuration the kernel refuses to run on."""


def _require_ci_level(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise MixedEffectsError("ci_level must be a number in (0, 1)")
    level = float(value)
    if not 0.0 < level < 1.0:
        raise MixedEffectsError("ci_level must be strictly between 0 and 1")
    return level


def _coerce_outcome(y: object) -> np.ndarray:
    if isinstance(y, (pd.Series, pd.DataFrame)):
        if isinstance(y, pd.DataFrame) and y.shape[1] != 1:
            raise MixedEffectsError("y must be a one-dimensional outcome vector")
        values = y.to_numpy(dtype=float).ravel()
    else:
        try:
            values = np.asarray(y, dtype=float).ravel()
        except (TypeError, ValueError):
            raise MixedEffectsError(
                "y must be a numeric one-dimensional outcome vector"
            ) from None
    if values.shape[0] == 0:
        raise MixedEffectsError("y must be non-empty")
    if not np.all(np.isfinite(values)):
        raise MixedEffectsError("y must contain only finite values (no NaN/inf)")
    if float(np.std(values)) == 0.0:
        raise MixedEffectsError("y has no variation; effects are unidentified")
    return values


def _coerce_design(
    X: object, n_samples: int, feature_names: Sequence[str] | None
) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        # Check dtypes explicitly: astype(float) alone would attempt coercion
        # instead of refusing non-numeric input fail-closed.
        non_numeric = [
            str(column)
            for column, dtype in X.dtypes.items()
            if not pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
        ]
        if non_numeric:
            raise MixedEffectsError(
                "X must be fully numeric; non-numeric columns: "
                + ",".join(sorted(non_numeric))
            )
        frame = X.copy().astype(float)
        if X.empty or X.shape[1] == 0:
            raise MixedEffectsError("X must be a non-empty design matrix")
        if feature_names is not None:
            given = [str(name) for name in list(feature_names)]
            if len(given) != frame.shape[1]:
                raise MixedEffectsError(
                    f"feature_names length {len(given)} does not match "
                    f"X features {frame.shape[1]}"
                )
            frame.columns = given
    else:
        try:
            values = np.asarray(X, dtype=float)
        except (TypeError, ValueError):
            raise MixedEffectsError("X must be a numeric 2D array or DataFrame") from None
        if values.ndim != 2:
            raise MixedEffectsError("X must be a numeric 2D array or DataFrame")
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise MixedEffectsError("X must be a non-empty design matrix")
        if not np.all(np.isfinite(values)):
            raise MixedEffectsError("X must contain only finite values (no NaN/inf)")
        if feature_names is None:
            columns = [f"x{i}" for i in range(values.shape[1])]
        else:
            columns = [str(name) for name in list(feature_names)]
            if len(columns) != values.shape[1]:
                raise MixedEffectsError(
                    f"feature_names length {len(columns)} does not match "
                    f"X features {values.shape[1]}"
                )
        frame = pd.DataFrame(values, columns=columns)
    if frame.shape[0] != n_samples:
        raise MixedEffectsError(
            f"X rows {frame.shape[0]} do not match y length {n_samples}"
        )
    names = [str(column) for column in frame.columns]
    if any(not name.strip() for name in names):
        raise MixedEffectsError("feature_names must all be non-blank")
    if len(set(names)) != len(names):
        raise MixedEffectsError("feature_names must be unique")
    return frame


def _coerce_groups(groups: object, n_samples: int) -> np.ndarray:
    labels = np.asarray(list(groups), dtype=object)
    if labels.ndim != 1 or labels.shape[0] != n_samples:
        raise MixedEffectsError("groups must be one-dimensional with the same length as y")
    n_groups = len(set(labels.tolist()))
    if n_groups < MIN_GROUPS:
        raise MixedEffectsError(
            f"mixed-effects estimation needs at least {MIN_GROUPS} groups, "
            f"got {n_groups}: with fewer groups the random-intercept variance "
            "is not identified (it piles onto the zero boundary) and the "
            "fixed-effect standard errors cannot be trusted"
        )
    return labels


@dataclass(frozen=True)
class MixedEffectsResult:
    """Typed random-intercept fit output; JSON form is the digest input."""

    var_names: tuple[str, ...]
    coef: tuple[float, ...]
    se: tuple[float, ...]
    z: tuple[float, ...]
    p: tuple[float, ...]
    ci_low: tuple[float, ...]
    ci_high: tuple[float, ...]
    group_var: float
    resid_scale: float
    loglike: float
    reml: bool
    ci_level: float
    n_obs: int
    n_groups: int
    converged: bool
    claim_ceiling: str = "analysis_only"
    method: str = "mixedlm_random_intercept_statsmodels"

    def to_json(self) -> Dict[str, Any]:
        return {
            "var_names": list(self.var_names),
            "coef": [float(value) for value in self.coef],
            "se": [float(value) for value in self.se],
            "z": [float(value) for value in self.z],
            "p": [float(value) for value in self.p],
            "ci_low": [float(value) for value in self.ci_low],
            "ci_high": [float(value) for value in self.ci_high],
            "group_var": float(self.group_var),
            "resid_scale": float(self.resid_scale),
            "loglike": float(self.loglike),
            "reml": bool(self.reml),
            "ci_level": float(self.ci_level),
            "n_obs": int(self.n_obs),
            "n_groups": int(self.n_groups),
            "converged": bool(self.converged),
            "claim_ceiling": self.claim_ceiling,
            "method": self.method,
        }


def result_sha256(result: MixedEffectsResult) -> str:
    """Return the canonical digest of one mixed-effects result."""

    if not isinstance(result, MixedEffectsResult):
        raise TypeError("result_sha256 requires a MixedEffectsResult")
    return canonical_sha256(result.to_json())


def fit_mixed_effects(
    y: object,
    X: object,
    groups: object,
    *,
    feature_names: Sequence[str] | None = None,
    add_intercept: bool = True,
    reml: bool = True,
    ci_level: float = 0.95,
) -> MixedEffectsResult:
    """Fit a random-intercept linear mixed model and report Wald statistics.

    Parameters
    ----------
    y, X, groups:
        Aligned continuous outcome, numeric design matrix (DataFrame or 2D
        array), and explicit group labels (one per row).
    add_intercept:
        Prepend a ``"const"`` column unless one is already present.
    reml:
        Restricted maximum likelihood (default) versus maximum likelihood.
    """

    level = _require_ci_level(ci_level)
    y_values = _coerce_outcome(y)
    n = int(y_values.shape[0])
    exog_frame = _coerce_design(X, n, feature_names)
    group_labels = _coerce_groups(groups, n)

    if add_intercept:
        exog_frame = sm.add_constant(exog_frame, has_constant="skip")
    names = [str(column) for column in exog_frame.columns]
    n_fixed = len(names)
    if n <= n_fixed + 1:
        raise MixedEffectsError(
            f"need more observations ({n}) than fixed parameters ({n_fixed}) plus "
            "the variance components"
        )

    try:
        fitted = sm.MixedLM(
            y_values,
            np.asarray(exog_frame, dtype=float),
            groups=group_labels,
        ).fit(reml=bool(reml))
    except MixedEffectsError:
        raise
    except Exception as exc:
        raise MixedEffectsError(f"statsmodels MixedLM fit failed: {exc}") from exc
    if not bool(fitted.converged):
        raise MixedEffectsError(
            "statsmodels MixedLM fit did not converge; refusing to report"
        )
    if getattr(fitted.cov_re, "shape", (0, 0)) != (1, 1):
        raise MixedEffectsError(
            "expected a scalar random-intercept covariance; refusing an "
            "unexpected random-effects structure"
        )

    coef = np.asarray(fitted.fe_params, dtype=float).ravel()
    stderr = np.asarray(fitted.bse_fe, dtype=float).ravel()
    zvalues = np.asarray(fitted.tvalues, dtype=float).ravel()[:n_fixed]
    pvalues = np.asarray(fitted.pvalues, dtype=float).ravel()[:n_fixed]
    intervals = np.asarray(fitted.conf_int(alpha=1.0 - level), dtype=float)[:n_fixed, :]
    group_var = float(np.asarray(fitted.cov_re, dtype=float).ravel()[0])
    resid_scale = float(fitted.scale)
    loglike = float(fitted.llf)
    if not (
        np.all(np.isfinite(coef))
        and np.all(np.isfinite(stderr))
        and np.all(np.isfinite(zvalues))
        and np.all(np.isfinite(pvalues))
        and np.all(np.isfinite(intervals))
        and np.isfinite(group_var)
        and np.isfinite(resid_scale)
        and np.isfinite(loglike)
    ):
        raise MixedEffectsError("statsmodels MixedLM fit produced non-finite statistics")
    if np.any(stderr <= 0.0):
        raise MixedEffectsError(
            "statsmodels MixedLM fit produced non-positive standard errors"
        )
    if group_var < 0.0 or resid_scale <= 0.0:
        raise MixedEffectsError(
            "statsmodels MixedLM fit produced an inadmissible variance component"
        )

    return MixedEffectsResult(
        var_names=tuple(names),
        coef=tuple(float(value) for value in coef),
        se=tuple(float(value) for value in stderr),
        z=tuple(float(value) for value in zvalues),
        p=tuple(float(value) for value in pvalues),
        ci_low=tuple(float(value) for value in intervals[:, 0]),
        ci_high=tuple(float(value) for value in intervals[:, 1]),
        group_var=group_var,
        resid_scale=resid_scale,
        loglike=loglike,
        reml=bool(reml),
        ci_level=level,
        n_obs=n,
        n_groups=len(set(group_labels.tolist())),
        converged=True,
    )


__all__ = [
    "TOOL_VERSION",
    "MIN_GROUPS",
    "MixedEffectsError",
    "MixedEffectsResult",
    "fit_mixed_effects",
    "result_sha256",
]
