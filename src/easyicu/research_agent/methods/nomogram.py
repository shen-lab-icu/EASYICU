"""Nomogram rendering data for fitted logistic/Cox models (analysis only).

Hand-rolled computation (wrap-vs-rewrite rule, reason 1: no package owns a
"points scale" quantity — the nomogram is a deterministic re-expression of
an already fitted linear predictor, so this kernel takes coefficients as
*input* and never fits anything). Plotting is deliberately out of scope:
this module returns the points scale per variable plus the total-points to
predicted-probability mapping; drawing them is the figure owner's job.

Construction. For variables ``x_j`` with coefficients ``β_j`` over ranges
``[lo_j, hi_j]`` and reference values ``r_j`` (defaults: the end with the
smaller contribution, so points are non-negative), let
``R_j = |β_j| (hi_j - lo_j)`` and ``R_max = max_j R_j``. Points are::

    points_j(x) = 100 β_j (x - r_j) / R_max,

so the highest-range variable spans exactly 0–100 and every other variable
is scaled proportionally. With ``T(x) = sum_j points_j`` and the linear
predictor ``LP(x)``, the identity ``LP(x) = base_lp + T(x)/scale`` holds
*exactly* (``scale = 100/R_max``, ``base_lp = LP(refs)``), and predicted
probabilities follow the model link:

* ``model="logistic"``: ``P = expit(intercept + sum β_j x_j)``;
* ``model="cox"``: event probability by the horizon,
  ``P = 1 - S0^exp(LP - center)`` with ``LP = sum β_j x_j`` (no intercept;
  the baseline survival ``S0`` belongs to ``LP = center``, default 0).

:func:`nomogram_predict` inverts nothing by interpolation: it evaluates the
same closed form, so ``points -> total -> probability`` agrees with the
model's own arithmetic to floating-point precision (pinned by a test).

Determinism contract: pure arithmetic over fixed linspace grids; no RNG is
involved, so rerunning with identical inputs yields byte-identical
:meth:`NomogramResult.to_json` output.

Fail-closed inputs (all raise :class:`NomogramError`, a ``ValueError``):

* length mismatch between variable names, coefficients and ranges;
* non-finite coefficients, bounds, references, intercept or baseline
  quantities; blank or duplicated variable names;
* a range with ``lo >= hi`` or a reference outside ``[lo, hi]``;
* an unknown ``model`` name; an ``intercept`` on a Cox model or a
  ``baseline_survival`` on a logistic one (either would be silently
  ignored); a Cox ``baseline_survival`` outside ``(0, 1)``;
* all-zero effects (``R_max = 0`` leaves the points scale undefined).

Claim ceiling: ``analysis_only``. A nomogram re-presents a fitted model; it
validates nothing about discrimination, calibration, or clinical utility.

Reference
---------
Harrell FE Jr. *Regression Modeling Strategies.* 2nd ed. Springer, 2015
(Chapter 10: nomograms).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from scipy.special import expit

from ..canonical_json import canonical_sha256

TOOL_VERSION = "1.0.0"

_MODELS = ("logistic", "cox")


class NomogramError(ValueError):
    """A nomogram input the kernel refuses to render data for."""


def _require_names(var_names: object) -> tuple[str, ...]:
    try:
        names = tuple(str(name) for name in list(var_names))  # type: ignore[arg-type]
    except TypeError:
        raise NomogramError("var_names must be a sequence of variable names") from None
    if not names:
        raise NomogramError("var_names must be non-empty")
    if any(not name.strip() for name in names):
        raise NomogramError("var_names must all be non-blank")
    if len(set(names)) != len(names):
        raise NomogramError("var_names must be unique")
    return names


def _require_coefs(coefs: object, n: int) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in list(coefs))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise NomogramError("coefs must be a numeric sequence") from None
    if len(values) != n:
        raise NomogramError(
            f"coefs length {len(values)} does not match var_names length {n}"
        )
    if not all(np.isfinite(values)):
        raise NomogramError("coefs must all be finite (no NaN/inf)")
    return values


def _require_ranges(
    ranges: object, names: Sequence[str]
) -> tuple[tuple[float, float], ...]:
    if isinstance(ranges, Mapping):
        try:
            pairs = [ranges[name] for name in names]
        except KeyError as exc:
            raise NomogramError(f"ranges is missing variable {exc}") from exc
    else:
        try:
            pairs = list(ranges)  # type: ignore[arg-type]
        except TypeError:
            raise NomogramError(
                "ranges must be a mapping or an aligned sequence of (lo, hi)"
            ) from None
        if len(pairs) != len(names):
            raise NomogramError(
                f"ranges length {len(pairs)} does not match var_names length {len(names)}"
            )
    resolved: list[tuple[float, float]] = []
    for name, pair in zip(names, pairs):
        try:
            lo, hi = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, IndexError):
            raise NomogramError(
                f"ranges for {name!r} must be a (lo, hi) pair"
            ) from None
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise NomogramError(f"ranges for {name!r} must be finite")
        if not lo < hi:
            raise NomogramError(
                f"ranges for {name!r} need lo < hi, got ({lo}, {hi})"
            )
        resolved.append((lo, hi))
    return tuple(resolved)


def _require_references(
    reference: object,
    names: Sequence[str],
    coefs: Sequence[float],
    bounds: Sequence[tuple[float, float]],
) -> tuple[float, ...]:
    if reference is None:
        return tuple(
            lo if coef > 0.0 else (hi if coef < 0.0 else lo)
            for coef, (lo, hi) in zip(coefs, bounds)
        )
    if isinstance(reference, Mapping):
        try:
            raw = [reference[name] for name in names]
        except KeyError as exc:
            raise NomogramError(f"reference is missing variable {exc}") from exc
    else:
        try:
            raw = list(reference)  # type: ignore[arg-type]
        except TypeError:
            raise NomogramError(
                "reference must be a mapping, an aligned sequence, or None"
            ) from None
        if len(raw) != len(names):
            raise NomogramError(
                f"reference length {len(raw)} does not match "
                f"var_names length {len(names)}"
            )
    resolved: list[float] = []
    for name, value, (lo, hi) in zip(names, raw, bounds):
        try:
            point = float(value)
        except (TypeError, ValueError):
            raise NomogramError(f"reference for {name!r} must be numeric") from None
        if not np.isfinite(point):
            raise NomogramError(f"reference for {name!r} must be finite")
        if not lo <= point <= hi:
            raise NomogramError(
                f"reference for {name!r} ({point}) lies outside [{lo}, {hi}]"
            )
        resolved.append(point)
    return tuple(resolved)


def _probability_from_lp(
    lp: float, *, model: str, baseline_survival: float, center: float
) -> float:
    if model == "logistic":
        return float(expit(lp))
    with np.errstate(over="ignore", invalid="ignore"):
        relative = float(np.exp(lp - center))
    if not np.isfinite(relative):
        return 1.0 if relative > 0.0 else 0.0
    survival = float(baseline_survival) ** relative
    return float(1.0 - survival)


@dataclass(frozen=True)
class NomogramVariableTable:
    """Points lookup grid for one variable: aligned value/points pairs."""

    variable: str
    values: tuple[float, ...]
    points: tuple[float, ...]

    def to_json(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "values": [float(value) for value in self.values],
            "points": [float(value) for value in self.points],
        }


@dataclass(frozen=True)
class NomogramResult:
    """Typed nomogram rendering data; JSON form is the digest input."""

    model: str
    var_names: tuple[str, ...]
    coefs: tuple[float, ...]
    intercept: float | None
    baseline_survival: float | None
    center: float
    ranges: tuple[tuple[float, float], ...]
    references: tuple[float, ...]
    points_per_unit_lp: float
    base_lp: float
    total_max: float
    variable_tables: tuple[NomogramVariableTable, ...]
    total_points: tuple[float, ...]
    total_lp: tuple[float, ...]
    total_prob: tuple[float, ...]
    n_grid: int
    claim_ceiling: str = "analysis_only"
    method: str = "nomogram_points_scale"

    def to_json(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "var_names": list(self.var_names),
            "coefs": [float(value) for value in self.coefs],
            "intercept": None if self.intercept is None else float(self.intercept),
            "baseline_survival": (
                None
                if self.baseline_survival is None
                else float(self.baseline_survival)
            ),
            "center": float(self.center),
            "ranges": [[float(lo), float(hi)] for lo, hi in self.ranges],
            "references": [float(value) for value in self.references],
            "points_per_unit_lp": float(self.points_per_unit_lp),
            "base_lp": float(self.base_lp),
            "total_max": float(self.total_max),
            "variable_tables": [table.to_json() for table in self.variable_tables],
            "total_points": [float(value) for value in self.total_points],
            "total_lp": [float(value) for value in self.total_lp],
            "total_prob": [float(value) for value in self.total_prob],
            "n_grid": int(self.n_grid),
            "claim_ceiling": self.claim_ceiling,
            "method": self.method,
        }


def result_sha256(result: NomogramResult) -> str:
    """Return the canonical digest of one nomogram result."""

    if not isinstance(result, NomogramResult):
        raise TypeError("result_sha256 requires a NomogramResult")
    return canonical_sha256(result.to_json())


def build_nomogram(
    var_names: object,
    coefs: object,
    ranges: object,
    *,
    model: str = "logistic",
    intercept: float = 0.0,
    reference: object = None,
    baseline_survival: float | None = None,
    center: float = 0.0,
    n_grid: int = 25,
) -> NomogramResult:
    """Build points-scale rendering data for a fitted linear predictor.

    Parameters
    ----------
    var_names, coefs, ranges:
        Aligned variable names, fitted coefficients, and ``(lo, hi)``
        display ranges (mapping or aligned sequence).
    model:
        ``"logistic"`` (needs ``intercept``) or ``"cox"`` (needs
        ``baseline_survival`` at ``LP = center``; refuses ``intercept``).
    reference:
        Points-zero anchor per variable (mapping, aligned sequence, or
        ``None`` for the minimum-contribution end).
    n_grid:
        Points per lookup grid (at least 2).
    """

    model_name = str(model or "").strip().lower()
    if model_name not in _MODELS:
        raise NomogramError(f"model must be one of {_MODELS}, got {model!r}")
    if isinstance(n_grid, bool) or not isinstance(n_grid, (int, np.integer)):
        raise NomogramError("n_grid must be an integer")
    grid = int(n_grid)
    if grid < 2:
        raise NomogramError("n_grid must be at least 2")

    names = _require_names(var_names)
    beta = _require_coefs(coefs, len(names))
    bounds = _require_ranges(ranges, names)
    refs = _require_references(reference, names, beta, bounds)

    try:
        intercept_value = float(intercept)
    except (TypeError, ValueError):
        raise NomogramError("intercept must be numeric") from None
    if not np.isfinite(intercept_value):
        raise NomogramError("intercept must be finite")
    try:
        center_value = float(center)
    except (TypeError, ValueError):
        raise NomogramError("center must be numeric") from None
    if not np.isfinite(center_value):
        raise NomogramError("center must be finite")

    baseline_value: float | None = None
    if model_name == "logistic":
        if baseline_survival is not None:
            raise NomogramError(
                "baseline_survival is a Cox quantity; refusing an argument "
                "that would be silently ignored on a logistic model"
            )
        if center_value != 0.0:
            raise NomogramError(
                "center is a Cox quantity; refusing an argument that would "
                "be silently ignored on a logistic model"
            )
    else:
        if intercept_value != 0.0:
            raise NomogramError(
                "a Cox linear predictor carries no intercept (it lives in "
                "the baseline survival); pass baseline_survival instead"
            )
        if baseline_survival is None:
            raise NomogramError(
                "model='cox' requires baseline_survival (survival at the "
                "horizon for LP = center)"
            )
        try:
            baseline_value = float(baseline_survival)
        except (TypeError, ValueError):
            raise NomogramError("baseline_survival must be numeric") from None
        if not 0.0 < baseline_value < 1.0:
            raise NomogramError(
                f"baseline_survival must lie in (0, 1), got {baseline_survival!r}"
            )

    spans = [abs(b) * (hi - lo) for b, (lo, hi) in zip(beta, bounds)]
    scale_max = max(spans)
    if not np.isfinite(scale_max) or scale_max <= 0.0:
        raise NomogramError(
            "all-zero effects leave the points scale undefined "
            "(max |coef| * range is 0)"
        )
    scale = 100.0 / scale_max

    def _linear_predictor(values: Sequence[float]) -> float:
        total = intercept_value if model_name == "logistic" else 0.0
        return float(total + sum(b * v for b, v in zip(beta, values)))

    base_lp = _linear_predictor(refs)

    variable_tables: list[NomogramVariableTable] = []
    max_points = 0.0
    for name, b, (lo, hi), ref in zip(names, beta, bounds, refs):
        grid_values = np.linspace(lo, hi, grid)
        grid_points = tuple(float(scale * b * (v - ref)) for v in grid_values.tolist())
        if min(grid_points) < -1e-9:
            raise NomogramError(  # pragma: no cover - guarded by default references
                f"points for {name!r} go negative; pass an explicit reference "
                "at the minimum-contribution end"
            )
        max_points += float(max(grid_points))
        variable_tables.append(
            NomogramVariableTable(
                variable=name,
                values=tuple(float(v) for v in grid_values.tolist()),
                points=grid_points,
            )
        )

    totals = np.linspace(0.0, max_points, grid)
    total_lp = tuple(float(base_lp + t / scale) for t in totals.tolist())
    total_prob = tuple(
        _probability_from_lp(
            lp,
            model=model_name,
            baseline_survival=1.0 if baseline_value is None else baseline_value,
            center=center_value,
        )
        for lp in total_lp
    )
    if any(not np.isfinite(p) or not 0.0 <= p <= 1.0 for p in total_prob):
        raise NomogramError("probability mapping left [0, 1]; refusing to report")

    return NomogramResult(
        model=model_name,
        var_names=names,
        coefs=beta,
        intercept=intercept_value if model_name == "logistic" else None,
        baseline_survival=baseline_value,
        center=center_value,
        ranges=bounds,
        references=refs,
        points_per_unit_lp=float(scale),
        base_lp=float(base_lp),
        total_max=float(max_points),
        variable_tables=tuple(variable_tables),
        total_points=tuple(float(t) for t in totals.tolist()),
        total_lp=total_lp,
        total_prob=total_prob,
        n_grid=grid,
    )


def nomogram_predict(
    result: NomogramResult, values: Mapping[str, float] | Sequence[float]
) -> tuple[float, float, float]:
    """Evaluate ``(total_points, linear_predictor, probability)`` for one case.

    Accepts a mapping ``variable -> value`` or an aligned sequence. Values
    outside the display ranges are refused rather than extrapolated.
    """

    if not isinstance(result, NomogramResult):
        raise TypeError("nomogram_predict requires a NomogramResult")
    if isinstance(values, Mapping):
        try:
            ordered = [values[name] for name in result.var_names]
        except KeyError as exc:
            raise NomogramError(f"values are missing variable {exc}") from exc
    else:
        try:
            ordered = list(values)
        except TypeError:
            raise NomogramError("values must be a mapping or an aligned sequence") from None
        if len(ordered) != len(result.var_names):
            raise NomogramError(
                f"values length {len(ordered)} does not match "
                f"var_names length {len(result.var_names)}"
            )
    numeric: list[float] = []
    for name, raw, (lo, hi) in zip(result.var_names, ordered, result.ranges):
        try:
            point = float(raw)
        except (TypeError, ValueError):
            raise NomogramError(f"value for {name!r} must be numeric") from None
        if not np.isfinite(point):
            raise NomogramError(f"value for {name!r} must be finite")
        if not lo <= point <= hi:
            raise NomogramError(
                f"value for {name!r} ({point}) lies outside the display "
                f"range [{lo}, {hi}]; refusing to extrapolate"
            )
        numeric.append(point)
    scale = result.points_per_unit_lp
    total = float(
        sum(
            scale * b * (v - r)
            for b, v, r in zip(result.coefs, numeric, result.references)
        )
    )
    lp = float(result.base_lp + total / scale)
    prob = _probability_from_lp(
        lp,
        model=result.model,
        baseline_survival=1.0 if result.baseline_survival is None else result.baseline_survival,
        center=result.center,
    )
    return total, lp, prob


__all__ = [
    "TOOL_VERSION",
    "NomogramError",
    "NomogramVariableTable",
    "NomogramResult",
    "build_nomogram",
    "nomogram_predict",
    "result_sha256",
]
