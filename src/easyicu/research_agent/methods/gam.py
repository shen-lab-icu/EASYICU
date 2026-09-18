"""Generalised additive model via penalised B-splines (analysis only).

Thin wrapper around :class:`statsmodels.gam.api.GLIMGam`
(:class:`~statsmodels.gam.generalized_additive_model.GLIMGam`, smoother
:class:`~statsmodels.gam.api.BSplines`) — wrap-vs-rewrite rule: the package
owns the penalised estimator; this kernel only pins the call (B-spline
smoother, configurable ``df`` per smooth term, Gaussian/Binomial family),
validates inputs fail-closed, and normalises a typed result plus a canonical
digest. Each named smooth term ``s(x_j)`` enters additively::

    g(E[y]) = x_linear @ beta + sum_j s_j(x_j),

with ``g`` the identity (``family="gaussian"``) or logit
(``family="binomial"``) link. Per-term effective degrees of freedom are the
penalised-fit EDFs reported by statsmodels, summed over each term's basis
columns (``edf = 1`` per column on the unpenalised ``alpha=0`` path).

Determinism contract: knot placement (quantiles) and the penalised-IRLS
optimiser are deterministic with no RNG involved, so rerunning with
identical inputs yields byte-identical :meth:`GAMResult.to_json` output.

Fail-closed inputs (all raise :class:`GAMError`, a ``ValueError``):

* a ``family`` name outside ``{"gaussian", "binomial"}``;
* a non-binary outcome under ``family="binomial"`` (values must be exactly
  0/1 with both classes observed), or a constant outcome;
* a ``df`` that is not a positive integer per smooth term (cubic splines
  need room for the basis: ``df`` must exceed ``degree``), an unknown
  smooth variable, a constant smooth variable, or duplicate smooth names;
* a negative ``alpha`` penalty or a non-finite input;
* length mismatches, or more total parameters than observations;
* a fit that reports non-convergence or non-finite statistics.

Claim ceiling: ``analysis_only``. Smooth dose-response shapes are not
reportable evidence of nonlinearity without a preregistered protocol and
independent review.

References
----------
Hastie TJ, Tibshirani RJ. *Generalized Additive Models.* Chapman & Hall,
1990.
Wood SN. *Generalized Additive Models: An Introduction with R.* 2nd ed.
Chapman & Hall/CRC, 2017.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.gam.api import BSplines, GLMGam

from ..canonical_json import canonical_sha256

TOOL_VERSION = "1.0.0"

_FAMILIES = ("gaussian", "binomial")


class GAMError(ValueError):
    """A GAM input or configuration the kernel refuses to run on."""


def _require_ci_level(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise GAMError("ci_level must be a number in (0, 1)")
    level = float(value)
    if not 0.0 < level < 1.0:
        raise GAMError("ci_level must be strictly between 0 and 1")
    return level


def _coerce_outcome(y: object, family: str) -> np.ndarray:
    if isinstance(y, (pd.Series, pd.DataFrame)):
        if isinstance(y, pd.DataFrame) and y.shape[1] != 1:
            raise GAMError("y must be a one-dimensional outcome vector")
        values = y.to_numpy(dtype=float).ravel()
    else:
        try:
            values = np.asarray(y, dtype=float).ravel()
        except (TypeError, ValueError):
            raise GAMError(
                "y must be a numeric one-dimensional outcome vector"
            ) from None
    if values.shape[0] == 0:
        raise GAMError("y must be non-empty")
    if not np.all(np.isfinite(values)):
        raise GAMError("y must contain only finite values (no NaN/inf)")
    if family == "binomial":
        if not np.isin(values, [0.0, 1.0]).all():
            raise GAMError("binomial outcome must take values in {0, 1}")
        if not (values == 1.0).any() or not (values == 0.0).any():
            raise GAMError("binomial outcome needs both classes observed")
    elif float(np.std(values)) == 0.0:
        raise GAMError("gaussian outcome has no variation; effects are unidentified")
    return values


def _coerce_frame(
    value: object,
    n_samples: int,
    names: Sequence[str] | None,
    *,
    field: str,
    allow_none: bool = False,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if value is None:
        if allow_none:
            return pd.DataFrame(index=range(n_samples)), ()
        raise GAMError(f"{field} must be provided")
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        resolved = [str(column) for column in frame.columns]
        if names is not None:
            given = [str(name) for name in list(names)]
            if len(given) != frame.shape[1]:
                raise GAMError(
                    f"{field} names length {len(given)} does not match "
                    f"columns {frame.shape[1]}"
                )
            frame.columns = given
            resolved = given
    else:
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise GAMError(f"{field} must be a numeric 2D array or DataFrame") from None
        if array.ndim != 2:
            raise GAMError(f"{field} must be a numeric 2D array or DataFrame")
        if names is None:
            resolved = [f"{field}{i}" for i in range(array.shape[1])]
        else:
            resolved = [str(name) for name in list(names)]
            if len(resolved) != array.shape[1]:
                raise GAMError(
                    f"{field} names length {len(resolved)} does not match "
                    f"columns {array.shape[1]}"
                )
        frame = pd.DataFrame(array, columns=resolved)
    if frame.shape[0] != n_samples:
        raise GAMError(f"{field} rows {frame.shape[0]} do not match y length {n_samples}")
    non_numeric = [
        str(column)
        for column, dtype in frame.dtypes.items()
        if not pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
    ]
    if non_numeric:
        raise GAMError(
            f"{field} must be fully numeric; non-numeric columns: "
            + ",".join(sorted(non_numeric))
        )
    block = frame.to_numpy(dtype=float)
    if block.size and not np.all(np.isfinite(block)):
        raise GAMError(f"{field} must contain only finite values (no NaN/inf)")
    if any(not name.strip() for name in resolved):
        raise GAMError(f"{field} names must all be non-blank")
    if len(set(resolved)) != len(resolved):
        raise GAMError(f"{field} names must be unique")
    return frame.astype(float), tuple(resolved)


@dataclass(frozen=True)
class GAMLinearTerm:
    """One parametric (linear) coefficient with Wald statistics."""

    name: str
    coef: float
    se: float
    z: float
    p: float
    ci_low: float
    ci_high: float

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "coef": float(self.coef),
            "se": float(self.se),
            "z": float(self.z),
            "p": float(self.p),
            "ci_low": float(self.ci_low),
            "ci_high": float(self.ci_high),
        }


@dataclass(frozen=True)
class GAMSmoothTerm:
    """One penalised smooth term with its effective degrees of freedom."""

    variable: str
    df: int
    degree: int
    n_basis: int
    edf: float
    basis_names: tuple[str, ...]

    def to_json(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "df": int(self.df),
            "degree": int(self.degree),
            "n_basis": int(self.n_basis),
            "edf": float(self.edf),
            "basis_names": list(self.basis_names),
        }


@dataclass(frozen=True)
class GAMResult:
    """Typed GAM fit output; JSON form is the digest input."""

    linear_terms: tuple[GAMLinearTerm, ...]
    smooth_terms: tuple[GAMSmoothTerm, ...]
    edf_total: float
    family: str
    alpha: float
    ci_level: float
    n_obs: int
    converged: bool
    claim_ceiling: str = "analysis_only"
    method: str = "glmgam_bsplines_statsmodels"

    def to_json(self) -> Dict[str, Any]:
        return {
            "linear_terms": [term.to_json() for term in self.linear_terms],
            "smooth_terms": [term.to_json() for term in self.smooth_terms],
            "edf_total": float(self.edf_total),
            "family": self.family,
            "alpha": float(self.alpha),
            "ci_level": float(self.ci_level),
            "n_obs": int(self.n_obs),
            "converged": bool(self.converged),
            "claim_ceiling": self.claim_ceiling,
            "method": self.method,
        }


def result_sha256(result: GAMResult) -> str:
    """Return the canonical digest of one GAM result."""

    if not isinstance(result, GAMResult):
        raise TypeError("result_sha256 requires a GAMResult")
    return canonical_sha256(result.to_json())


def _resolve_df(
    df: int | Mapping[str, int],
    smooth_names: Sequence[str],
    degree: int,
) -> list[int]:
    if isinstance(df, Mapping):
        try:
            resolved = [df[name] for name in smooth_names]
        except KeyError as exc:
            raise GAMError(f"df is missing the smooth variable {exc}") from exc
    else:
        if isinstance(df, bool) or not isinstance(df, (int, np.integer)):
            raise GAMError("df must be an integer or a mapping of variable to integer")
        resolved = [int(df)] * len(smooth_names)
    for name, value in zip(smooth_names, resolved):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise GAMError(f"df for smooth variable {name!r} must be an integer")
        if int(value) <= int(degree):
            raise GAMError(
                f"df for smooth variable {name!r} must exceed degree "
                f"({degree}); got {value}"
            )
    return [int(value) for value in resolved]


def fit_gam(
    y: object,
    smooth_X: object,
    X_linear: object | None = None,
    *,
    smooth_names: Sequence[str] | None = None,
    linear_names: Sequence[str] | None = None,
    df: int | Mapping[str, int] = 4,
    degree: int = 3,
    family: str = "gaussian",
    alpha: float = 0.0,
    add_intercept: bool = True,
    ci_level: float = 0.95,
) -> GAMResult:
    """Fit a penalised B-spline GAM and report parametric + smooth terms.

    Parameters
    ----------
    y:
        Outcome vector (binary 0/1 for ``family="binomial"``).
    smooth_X:
        DataFrame (or 2D array with ``smooth_names``) of variables that
        enter through penalised B-spline smooths, one column per term.
    X_linear:
        Optional DataFrame (or 2D array with ``linear_names``) of
        parametric terms; ``None`` means intercept only.
    df:
        Basis dimension per smooth term: a single int or a mapping
        ``variable -> int``. Must exceed ``degree``.
    alpha:
        Non-negative smoothing penalty (``0`` = unpenalised regression
        spline; larger values shrink each smooth toward a line).
    """

    family_name = str(family or "").strip().lower()
    if family_name not in _FAMILIES:
        raise GAMError(f"family must be one of {_FAMILIES}, got {family!r}")
    level = _require_ci_level(ci_level)
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):
        raise GAMError("degree must be an integer")
    degree_value = int(degree)
    if degree_value < 1:
        raise GAMError("degree must be a positive integer")
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError):
        raise GAMError("alpha must be a non-negative finite float") from None
    if not np.isfinite(alpha_value) or alpha_value < 0.0:
        raise GAMError("alpha must be a non-negative finite float")

    y_values = _coerce_outcome(y, family_name)
    n = int(y_values.shape[0])
    smooth_frame, smooth_resolved = _coerce_frame(
        smooth_X, n, smooth_names, field="smooth_X"
    )
    if len(smooth_resolved) == 0:
        raise GAMError("smooth_X must hold at least one smooth variable")
    overlap = set(smooth_resolved)
    linear_frame, linear_resolved = _coerce_frame(
        X_linear, n, linear_names, field="X_linear", allow_none=True
    )
    if overlap & set(linear_resolved):
        raise GAMError(
            "variables cannot enter both linearly and smoothly: "
            + ",".join(sorted(overlap & set(linear_resolved)))
        )
    for name in smooth_resolved:
        column = smooth_frame[name].to_numpy(dtype=float)
        if float(np.std(column)) == 0.0:
            raise GAMError(
                f"smooth variable {name!r} is constant; a spline of a "
                "constant is unidentified"
            )

    df_list = _resolve_df(df, list(smooth_resolved), degree_value)

    if add_intercept:
        linear_frame = sm.add_constant(linear_frame, has_constant="skip")
    linear_final = [str(column) for column in linear_frame.columns]

    try:
        smoother = BSplines(
            np.asarray(smooth_frame, dtype=float),
            df=df_list,
            degree=[degree_value] * len(smooth_resolved),
            variable_names=list(smooth_resolved),
        )
    except GAMError:
        raise
    except Exception as exc:
        raise GAMError(f"could not build the B-spline smoother: {exc}") from exc

    basis_counts: list[int] = []
    for pos in range(len(smooth_resolved)):
        try:
            block = smoother.smoothers[pos].transform(
                np.asarray(smooth_frame, dtype=float)[:, pos]
            )
            basis_counts.append(int(np.shape(block)[1]))
        except Exception as exc:
            raise GAMError(
                f"could not evaluate the spline basis for "
                f"{smooth_resolved[pos]!r}: {exc}"
            ) from exc
    n_params = len(linear_final) + int(np.sum(basis_counts))
    if n <= n_params:
        raise GAMError(
            f"need more observations ({n}) than total parameters ({n_params})"
        )

    family_obj = (
        sm.families.Binomial() if family_name == "binomial" else sm.families.Gaussian()
    )
    try:
        fitted = GLMGam(
            y_values,
            exog=np.asarray(linear_frame, dtype=float),
            smoother=smoother,
            alpha=alpha_value,
            family=family_obj,
        ).fit()
    except GAMError:
        raise
    except Exception as exc:
        raise GAMError(f"statsmodels GLMGam fit failed: {exc}") from exc
    if not bool(fitted.converged):
        raise GAMError("statsmodels GLMGam fit did not converge; refusing to report")

    params = np.asarray(fitted.params, dtype=float).ravel()
    stderr = np.asarray(fitted.bse, dtype=float).ravel()
    zvalues = np.asarray(fitted.tvalues, dtype=float).ravel()
    pvalues = np.asarray(fitted.pvalues, dtype=float).ravel()
    intervals = np.asarray(fitted.conf_int(alpha=1.0 - level), dtype=float)
    edf = np.asarray(fitted.edf, dtype=float).ravel()
    # Param names come from our own frames: fitted.params may be a bare
    # ndarray when exog was passed positionally, so the smoother's recorded
    # column names (aligned with its transform output) are authoritative.
    basis_names = [str(name) for name in list(smoother.col_names)]
    if len(basis_names) != int(np.sum(basis_counts)):
        raise GAMError("spline basis names do not match the basis columns")
    if not (
        params.shape[0] == n_params
        and stderr.shape[0] == n_params
        and zvalues.shape[0] == n_params
        and pvalues.shape[0] == n_params
        and intervals.shape == (n_params, 2)
        and edf.shape[0] == n_params
    ):
        raise GAMError("statsmodels GLMGam fit returned an unexpected result shape")
    if not (
        np.all(np.isfinite(params))
        and np.all(np.isfinite(stderr))
        and np.all(np.isfinite(zvalues))
        and np.all(np.isfinite(pvalues))
        and np.all(np.isfinite(intervals))
        and np.all(np.isfinite(edf))
    ):
        raise GAMError("statsmodels GLMGam fit produced non-finite statistics")
    if np.any(stderr[: len(linear_final)] <= 0.0):
        raise GAMError("statsmodels GLMGam fit produced non-positive standard errors")

    linear_terms: list[GAMLinearTerm] = []
    for pos, name in enumerate(linear_final):
        linear_terms.append(
            GAMLinearTerm(
                name=name,
                coef=float(params[pos]),
                se=float(stderr[pos]),
                z=float(zvalues[pos]),
                p=float(pvalues[pos]),
                ci_low=float(intervals[pos, 0]),
                ci_high=float(intervals[pos, 1]),
            )
        )
    smooth_terms: list[GAMSmoothTerm] = []
    param_offset = len(linear_final)
    name_offset = 0
    for var, count in zip(smooth_resolved, basis_counts):
        edf_slice = slice(param_offset, param_offset + count)
        name_slice = slice(name_offset, name_offset + count)
        smooth_terms.append(
            GAMSmoothTerm(
                variable=str(var),
                df=int(df_list[list(smooth_resolved).index(var)]),
                degree=degree_value,
                n_basis=int(count),
                edf=float(np.sum(edf[edf_slice])),
                basis_names=tuple(basis_names[name_slice]),
            )
        )
        param_offset += count
        name_offset += count

    return GAMResult(
        linear_terms=tuple(linear_terms),
        smooth_terms=tuple(smooth_terms),
        edf_total=float(np.sum(edf)),
        family=family_name,
        alpha=alpha_value,
        ci_level=level,
        n_obs=n,
        converged=True,
    )


__all__ = [
    "TOOL_VERSION",
    "GAMError",
    "GAMLinearTerm",
    "GAMSmoothTerm",
    "GAMResult",
    "fit_gam",
    "result_sha256",
]
