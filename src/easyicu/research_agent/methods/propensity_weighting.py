"""Deterministic propensity-score kernel: PS estimation, 1:1 matching, IPTW.

This module is a **deterministic statistical kernel** in the
``research_agent.methods`` owner boundary. It prepares confounding-adjustment
inputs — propensity scores, matched index pairs, and inverse-probability
weights — plus balance/weight diagnostics. It performs **no effect
estimation**: no outcome is accepted, no treatment effect (ATE/ATT/RD/OR/HR)
is computed. Effect estimation belongs to the downstream owner.

Scope ceiling: ``analysis_only``. Every result carries
``analysis_only=True`` and a ``note`` stating it is not reportable evidence
and must not be promoted or inventoried (promotion/inventory are centrally
owned). There is no ``reportable`` flag anywhere in this module.

Determinism contract
--------------------
All randomness is seeded. ``estimate_propensity_scores`` fits
``sklearn.linear_model.LogisticRegression`` with a fixed ``random_state`` and
``C`` (``solver="lbfgs"``, ``max_iter=1000``). ``match_nearest_neighbor``
orders treated units with ``numpy.random.default_rng(random_state)`` and
breaks distance ties by ascending original index, so two runs with identical
inputs and parameters produce byte-identical ``to_dict()`` payloads (and
identical ``digest`` values). ``compute_iptw_weights`` is closed form.

Fail-closed inputs
------------------
``treatment`` must be one-dimensional and strictly binary (0/1, accepting
bool/0.0/1.0 spellings); ``covariates`` must be a finite 2D matrix with one
row per subject (a 1D vector is read as a single covariate). Any missing
(``NaN``/``None``), non-finite (``inf``), length-mismatched, single-class, or
non-binary input raises ``ValueError`` (via :class:`PropensityWeightingError`)
instead of returning a valid-looking result.

Statistical definitions
-----------------------
* Propensity score: ``P(T=1 | X)`` from logistic regression on ``X``.
* Standardized mean difference (per covariate)::

      SMD = (mean_treated - mean_control) / sqrt((s_t^2 + s_c^2) / 2)

  with sample variances (``ddof=1``). ``None`` when undefined (fewer than two
  observations per side, or zero pooled variance with unequal means).
* 1:1 matching: greedy nearest neighbour on the propensity-score scale
  without replacement. ``caliper`` is the maximum absolute PS distance for a
  pair (``None`` disables the caliper). Complexity is
  ``O(n_treated * n_control)`` distance evaluations, computed with a
  vectorised per-treated argmin (identical pair output to the explicit
  scan, C-level speed: ~22k rows match in ~0.1 s).
* Stabilized IPTW weights with marginal treated rate ``p`` and clipped score
  ``e = clip(ps, eps, 1 - eps)``::

      w = p / e            if treated
      w = (1 - p) / (1 - e) if control

  Raw (unstabilized) weights use ``p = 1``. Optional ``truncate_quantiles``
  ``(lo, hi)`` clips weights at those quantiles of the weight distribution.
* Effective sample size:: ``ESS = (sum w)^2 / sum w^2``.

Digest binding
--------------
Each result carries a ``digest``: the SHA-256 hex of the canonical JSON
(sorted keys, compact separators) of its validated inputs plus parameters,
mirroring the digest habit of neighbouring kernels (e.g. Table 1 spec
digests). Recomputing over the same inputs reproduces the digest.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

if TYPE_CHECKING:
    from ..schema import AnalysisStep, ValidationFinding

from ..contracts.diagnostic_specs import (
    BALANCE_ACTION, BALANCE_KIND, BALANCE_OUTPUTS,
    CovariateBalanceResult, CovariateBalanceRow, CovariateBalanceSpec,
    covariate_balance_spec_sha256,
)
from ..canonical_json import canonical_sha256
from ..contracts.model_terms import level_identity_class, level_spelling

DEFAULT_C = 1.0
DEFAULT_RANDOM_STATE = 42
DEFAULT_CALIPER = 0.1
_PS_EPS = 1e-6

ANALYSIS_ONLY_NOTE = (
    "analysis_only: balance/weight diagnostics only; no treatment effect "
    "estimated. Not reportable evidence; promotion/inventory are centrally "
    "owned. The downstream owner computes effects from the matched sample "
    "or weights."
)


class PropensityWeightingError(ValueError):
    """Fail-closed rejection of invalid propensity-weighting inputs."""


# ---------------------------------------------------------------------------
# Canonical digest helper
# ---------------------------------------------------------------------------


def _canonical_digest(payload: Dict[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Input coercion (fail closed)
# ---------------------------------------------------------------------------


def _coerce_treatment(treatment: Any) -> np.ndarray:
    """Return validated int 0/1 treatment vector or raise."""
    if isinstance(treatment, pd.Series):
        if treatment.isna().any():
            raise PropensityWeightingError("treatment must not contain missing values")
        values = treatment.to_numpy()
    elif isinstance(treatment, pd.DataFrame):
        raise PropensityWeightingError("treatment must be one-dimensional")
    else:
        try:
            values = np.asarray(treatment)
        except (TypeError, ValueError) as exc:
            raise PropensityWeightingError(f"treatment is not array-like: {exc}") from exc
    if values.ndim != 1:
        raise PropensityWeightingError("treatment must be one-dimensional")
    if values.size < 2:
        raise PropensityWeightingError("treatment needs at least two subjects")
    kind = values.dtype.kind
    if kind in ("U", "S"):
        raise PropensityWeightingError("treatment must be numeric binary 0/1")
    if kind == "O":
        # Object arrays arise from mixed/None inputs: accept only genuine
        # numbers, rejecting strings, None, and other non-numeric spellings.
        cleaned: List[float] = []
        for element in values.tolist():
            if isinstance(element, bool):
                cleaned.append(float(element))
            elif isinstance(element, (int, float, np.integer, np.floating)):
                cleaned.append(float(element))
            else:
                raise PropensityWeightingError(
                    "treatment must be binary 0/1 without missing values"
                )
        as_float = np.asarray(cleaned, dtype=float)
    elif kind == "b":
        out = values.astype(int)
        unique = set(out.tolist())
        if unique != {0, 1}:
            raise PropensityWeightingError(
                "treatment must contain both treated (1) and control (0) subjects"
            )
        return out
    elif kind in ("i", "u", "f"):
        as_float = values.astype(float)
    else:
        raise PropensityWeightingError("treatment must be numeric binary 0/1")
    if not np.all(np.isfinite(as_float)):
        raise PropensityWeightingError(
            "treatment must be finite without missing values"
        )
    allowed = (as_float == 0.0) | (as_float == 1.0)
    if not bool(np.all(allowed)):
        raise PropensityWeightingError("treatment must be binary (only 0/1)")
    out = as_float.astype(int)
    unique = set(out.tolist())
    if unique != {0, 1}:
        raise PropensityWeightingError(
            "treatment must contain both treated (1) and control (0) subjects"
        )
    return out


def _coerce_covariates(
    covariates: Any,
    n: int,
    covariate_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Return validated finite (n, k) matrix plus names, or raise."""
    inferred: Optional[List[str]] = None
    if isinstance(covariates, pd.DataFrame):
        if covariates.isna().any().any():
            raise PropensityWeightingError("covariates must not contain missing values")
        inferred = [str(c) for c in covariates.columns.tolist()]
        try:
            matrix = covariates.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise PropensityWeightingError(
                f"covariates must be numeric: {exc}"
            ) from exc
    else:
        try:
            matrix = np.asarray(covariates, dtype=float)
        except (TypeError, ValueError) as exc:
            raise PropensityWeightingError(
                f"covariates must be a numeric matrix: {exc}"
            ) from exc
    if matrix.ndim == 1:
        # A single covariate supplied as a flat vector.
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise PropensityWeightingError("covariates must be two-dimensional")
    if matrix.shape[0] != n:
        raise PropensityWeightingError(
            f"covariates has {matrix.shape[0]} rows but treatment has {n}"
        )
    if matrix.shape[1] < 1:
        raise PropensityWeightingError("covariates needs at least one column")
    if not np.all(np.isfinite(matrix)):
        raise PropensityWeightingError(
            "covariates must be finite without missing values"
        )
    k = matrix.shape[1]
    if covariate_names is None:
        names = inferred if inferred is not None else [f"x{i}" for i in range(k)]
    else:
        names = [str(name) for name in list(covariate_names)]
    if len(names) != k:
        raise PropensityWeightingError(
            f"covariate_names has {len(names)} entries but covariates has {k} columns"
        )
    if len(set(names)) != len(names) or any(name == "" for name in names):
        raise PropensityWeightingError("covariate_names must be unique and non-empty")
    return np.ascontiguousarray(matrix, dtype=float), names


def _coerce_propensity_scores(scores: Any, n: int) -> np.ndarray:
    """Return validated PS vector in [0, 1], or raise."""
    if isinstance(scores, PropensityScoreResult):
        values = np.asarray(scores.propensity_scores, dtype=float)
    else:
        if isinstance(scores, (pd.Series, pd.DataFrame)):
            raw = scores.to_numpy(dtype=float)
            values = np.asarray(raw, dtype=float).reshape(-1)
        else:
            try:
                values = np.asarray(scores, dtype=float)
            except (TypeError, ValueError) as exc:
                raise PropensityWeightingError(
                    f"propensity_scores must be numeric: {exc}"
                ) from exc
            if values.ndim != 1:
                raise PropensityWeightingError("propensity_scores must be one-dimensional")
    if values.shape[0] != n:
        raise PropensityWeightingError(
            f"propensity_scores has {values.shape[0]} entries but treatment has {n}"
        )
    if not np.all(np.isfinite(values)):
        raise PropensityWeightingError("propensity_scores must be finite")
    if bool(np.any(values < 0.0) or np.any(values > 1.0)):
        raise PropensityWeightingError("propensity_scores must lie in [0, 1]")
    return np.ascontiguousarray(values, dtype=float)


def _check_random_state(random_state: Any) -> int:
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise PropensityWeightingError("random_state must be an integer")
    if not 0 <= random_state <= 2**32 - 1:
        raise PropensityWeightingError("random_state must be in [0, 2**32 - 1]")
    return random_state


def _check_C(value: Any) -> float:  # noqa: N802 - sklearn spells it C
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PropensityWeightingError("C must be a positive number")
    c = float(value)
    if not math.isfinite(c) or c <= 0.0:
        raise PropensityWeightingError("C must be positive and finite")
    return c


def _check_caliper(caliper: Any) -> Optional[float]:
    if caliper is None:
        return None
    if isinstance(caliper, bool) or not isinstance(caliper, (int, float)):
        raise PropensityWeightingError("caliper must be a positive number or None")
    value = float(caliper)
    if not math.isfinite(value) or value <= 0.0:
        raise PropensityWeightingError("caliper must be positive and finite")
    return value


def _check_truncate_quantiles(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    try:
        lo, hi = (float(value[0]), float(value[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise PropensityWeightingError(
            "truncate_quantiles must be a (lo, hi) pair"
        ) from exc
    if (
        not math.isfinite(lo)
        or not math.isfinite(hi)
        or not 0.0 < lo < hi < 1.0
    ):
        raise PropensityWeightingError(
            "truncate_quantiles must satisfy 0 < lo < hi < 1"
        )
    return (lo, hi)


# ---------------------------------------------------------------------------
# Balance diagnostics
# ---------------------------------------------------------------------------


def standardized_mean_difference(
    treated_values: Sequence[float],
    control_values: Sequence[float],
) -> Optional[float]:
    """Standardized mean difference of one covariate (treated − control).

    Uses sample variances (``ddof=1``) with equal-weight pooling. Returns
    ``None`` when the SMD is undefined: either side has fewer than two
    observations, any value is non-finite, or the pooled variance is zero
    while the means differ (returns ``0.0`` when both means agree exactly).
    """
    try:
        a = np.asarray(list(treated_values), dtype=float)
        b = np.asarray(list(control_values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise PropensityWeightingError(
            f"SMD inputs must be numeric: {exc}"
        ) from exc
    if a.ndim != 1 or b.ndim != 1 or a.size < 2 or b.size < 2:
        return None
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return None
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    pooled = (float(np.var(a, ddof=1)) + float(np.var(b, ddof=1))) / 2.0
    if not math.isfinite(pooled) or pooled < 0.0:
        return None
    if pooled == 0.0:
        return 0.0 if mean_a == mean_b else None
    value = (mean_a - mean_b) / math.sqrt(pooled)
    return float(value) if math.isfinite(value) else None


@dataclass(frozen=True)
class BalanceRow:
    """Love-table-style balance row for one covariate."""

    covariate: str
    mean_treated_before: float
    mean_control_before: float
    smd_before: Optional[float]
    mean_treated_after: Optional[float]
    mean_control_after: Optional[float]
    smd_after: Optional[float]
    n_matched_pairs: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "covariate": self.covariate,
            "mean_treated_before": self.mean_treated_before,
            "mean_control_before": self.mean_control_before,
            "smd_before": self.smd_before,
            "mean_treated_after": self.mean_treated_after,
            "mean_control_after": self.mean_control_after,
            "smd_after": self.smd_after,
            "n_matched_pairs": self.n_matched_pairs,
        }

    def to_json(self) -> Dict[str, Any]:
        return self.to_dict()


def _balance_rows(
    names: Sequence[str],
    matrix: np.ndarray,
    treated: np.ndarray,
    matched_treated: np.ndarray,
    matched_control: np.ndarray,
) -> List[BalanceRow]:
    rows: List[BalanceRow] = []
    n_pairs = int(matched_treated.shape[0])
    for j, name in enumerate(names):
        col = matrix[:, j]
        before = standardized_mean_difference(col[treated == 1], col[treated == 0])
        if n_pairs > 0:
            after = standardized_mean_difference(
                col[matched_treated], col[matched_control]
            )
            mean_t_after: Optional[float] = float(np.mean(col[matched_treated]))
            mean_c_after: Optional[float] = float(np.mean(col[matched_control]))
        else:
            after = None
            mean_t_after = None
            mean_c_after = None
        rows.append(
            BalanceRow(
                covariate=str(name),
                mean_treated_before=float(np.mean(col[treated == 1])),
                mean_control_before=float(np.mean(col[treated == 0])),
                smd_before=before,
                mean_treated_after=mean_t_after,
                mean_control_after=mean_c_after,
                smd_after=after,
                n_matched_pairs=n_pairs,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Typed results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropensityScoreResult:
    """Estimated propensity scores with their deterministic binding."""

    propensity_scores: List[float] = field(default_factory=list)
    covariate_names: List[str] = field(default_factory=list)
    n: int = 0
    n_treated: int = 0
    n_control: int = 0
    treated_rate: float = 0.0
    C: float = DEFAULT_C  # noqa: N815 - sklearn spells it C
    random_state: int = DEFAULT_RANDOM_STATE
    intercept: float = 0.0
    coefficients: List[float] = field(default_factory=list)
    digest: str = ""
    method: str = "logistic_regression_lbfgs"
    analysis_only: bool = True
    note: str = ANALYSIS_ONLY_NOTE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "propensity_scores": list(self.propensity_scores),
            "covariate_names": list(self.covariate_names),
            "n": self.n,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "treated_rate": self.treated_rate,
            "C": self.C,
            "random_state": self.random_state,
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "digest": self.digest,
            "method": self.method,
            "analysis_only": self.analysis_only,
            "note": self.note,
        }

    def to_json(self) -> Dict[str, Any]:
        return self.to_dict()


@dataclass(frozen=True)
class MatchingResult:
    """Greedy 1:1 nearest-neighbour matched pairs plus balance diagnostics."""

    pairs: List[Tuple[int, int]] = field(default_factory=list)
    matched_treated_indices: List[int] = field(default_factory=list)
    matched_control_indices: List[int] = field(default_factory=list)
    n_pairs: int = 0
    n_unmatched_treated: int = 0
    n_unmatched_control: int = 0
    caliper: Optional[float] = DEFAULT_CALIPER
    random_state: int = DEFAULT_RANDOM_STATE
    balance: List[BalanceRow] = field(default_factory=list)
    digest: str = ""
    method: str = "nearest_neighbor_1to1_without_replacement"
    analysis_only: bool = True
    note: str = ANALYSIS_ONLY_NOTE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pairs": [[int(t), int(c)] for t, c in self.pairs],
            "matched_treated_indices": [int(i) for i in self.matched_treated_indices],
            "matched_control_indices": [int(i) for i in self.matched_control_indices],
            "n_pairs": self.n_pairs,
            "n_unmatched_treated": self.n_unmatched_treated,
            "n_unmatched_control": self.n_unmatched_control,
            "caliper": self.caliper,
            "random_state": self.random_state,
            "balance": [row.to_dict() for row in self.balance],
            "digest": self.digest,
            "method": self.method,
            "analysis_only": self.analysis_only,
            "note": self.note,
        }

    def to_json(self) -> Dict[str, Any]:
        return self.to_dict()


@dataclass(frozen=True)
class IPTWResult:
    """Stabilized IPTW weights with distribution summary and ESS."""

    weights: List[float] = field(default_factory=list)
    stabilized: bool = True
    truncate_quantiles: Optional[List[float]] = None
    weight_min: float = 0.0
    weight_p50: float = 0.0
    weight_max: float = 0.0
    weight_mean: float = 0.0
    ess: float = 0.0
    ess_treated: float = 0.0
    ess_control: float = 0.0
    n: int = 0
    digest: str = ""
    method: str = "iptw"
    analysis_only: bool = True
    note: str = ANALYSIS_ONLY_NOTE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": list(self.weights),
            "stabilized": self.stabilized,
            "truncate_quantiles": (
                None if self.truncate_quantiles is None else list(self.truncate_quantiles)
            ),
            "weight_min": self.weight_min,
            "weight_p50": self.weight_p50,
            "weight_max": self.weight_max,
            "weight_mean": self.weight_mean,
            "ess": self.ess,
            "ess_treated": self.ess_treated,
            "ess_control": self.ess_control,
            "n": self.n,
            "digest": self.digest,
            "method": self.method,
            "analysis_only": self.analysis_only,
            "note": self.note,
        }

    def to_json(self) -> Dict[str, Any]:
        return self.to_dict()


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def estimate_propensity_scores(
    treatment: Any,
    covariates: Any,
    *,
    covariate_names: Optional[Sequence[str]] = None,
    C: float = DEFAULT_C,  # noqa: N803 - sklearn spells it C
    random_state: int = DEFAULT_RANDOM_STATE,
) -> PropensityScoreResult:
    """Estimate ``P(T=1 | X)`` with logistic regression (deterministic).

    Parameters
    ----------
    treatment:
        Binary 0/1 vector (numpy/pandas/list spellings accepted).
    covariates:
        Finite numeric matrix with one row per subject (pandas DataFrame
        accepted; a flat vector counts as a single covariate).
    covariate_names:
        Optional column names; defaults to DataFrame columns or
        ``x0..x{k-1}``.
    C:
        Inverse regularization strength, fixed per call (default 1.0).
    random_state:
        Fixed seed recorded on the result (default 42).
    """
    treated = _coerce_treatment(treatment)
    matrix, names = _coerce_covariates(covariates, treated.shape[0], covariate_names)
    c = _check_C(C)
    seed = _check_random_state(random_state)

    model = LogisticRegression(C=c, random_state=seed, solver="lbfgs", max_iter=1000)
    model.fit(matrix, treated)
    scores = np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)
    scores = np.clip(scores, 0.0, 1.0)

    n_treated = int(np.sum(treated == 1))
    n = int(treated.shape[0])
    # The digest binds inputs, parameters AND the estimated scores: changing
    # any output must change the digest (review finding: digests that cover
    # inputs alone cannot serve as origin-output receipts).
    digest = _canonical_digest(
        {
            "kind": "propensity_scores",
            "treatment": treated.tolist(),
            "covariates": matrix.tolist(),
            "covariate_names": names,
            "C": c,
            "random_state": seed,
            "propensity_scores": [float(v) for v in scores.tolist()],
        }
    )
    return PropensityScoreResult(
        propensity_scores=[float(v) for v in scores.tolist()],
        covariate_names=names,
        n=n,
        n_treated=n_treated,
        n_control=n - n_treated,
        treated_rate=float(n_treated / n),
        C=c,
        random_state=seed,
        intercept=float(model.intercept_[0]),
        coefficients=[float(v) for v in np.asarray(model.coef_[0]).tolist()],
        digest=digest,
    )


def match_nearest_neighbor(
    treatment: Any,
    covariates: Any,
    propensity_scores: Any,
    *,
    covariate_names: Optional[Sequence[str]] = None,
    caliper: Optional[float] = DEFAULT_CALIPER,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> MatchingResult:
    """Greedy 1:1 nearest-neighbour matching without replacement.

    Treated units are visited in ``default_rng(random_state)`` permutation
    order; each takes the closest available control on the absolute
    propensity-score scale (ties broken by ascending original index) when
    that distance is within ``caliper`` (``None`` disables the caliper).
    Balance rows report pre/post means and SMDs per covariate in
    Love-table row form. No outcome is involved and none is estimated.
    """
    treated = _coerce_treatment(treatment)
    matrix, names = _coerce_covariates(covariates, treated.shape[0], covariate_names)
    scores = _coerce_propensity_scores(propensity_scores, treated.shape[0])
    limit = _check_caliper(caliper)
    seed = _check_random_state(random_state)

    treated_idx = np.where(treated == 1)[0]
    control_idx = np.where(treated == 0)[0]
    visit_order = np.random.default_rng(seed).permutation(treated_idx)
    available = np.asarray(sorted(int(i) for i in control_idx.tolist()), dtype=int)

    # Vectorised equivalent of the original Python-level greedy scan (kept
    # semantics, C-level speed): visit order and caliper rule unchanged;
    # ``argmin`` returns the first minimum in ascending index order, which
    # is exactly the old strict-``<`` tie-break. Pair output is identical.
    pairs: List[Tuple[int, int]] = []
    for cand in (int(i) for i in visit_order.tolist()):
        if available.shape[0] == 0:
            continue
        distances = np.abs(scores[available] - float(scores[cand]))
        if limit is not None:
            within = np.where(distances <= limit)[0]
            if within.shape[0] == 0:
                continue
            best = int(available[within[np.argmin(distances[within])]])
        else:
            best = int(available[int(np.argmin(distances))])
        pairs.append((cand, best))
        available = available[available != best]

    # Canonical pair order (ascending treated index) keeps the payload stable
    # regardless of visit order; determinism still flows from the seed.
    pairs = sorted(pairs, key=lambda pair: (pair[0], pair[1]))
    matched_treated = np.asarray([p[0] for p in pairs], dtype=int)
    matched_control = np.asarray([p[1] for p in pairs], dtype=int)
    rows = _balance_rows(names, matrix, treated, matched_treated, matched_control)
    # The digest binds the matched pairs and balance diagnostics, not just
    # the inputs: re-pairing the same inputs must produce a new digest.
    digest = _canonical_digest(
        {
            "kind": "nearest_neighbor_matching",
            "treatment": treated.tolist(),
            "covariates": matrix.tolist(),
            "covariate_names": names,
            "propensity_scores": [float(v) for v in scores.tolist()],
            "caliper": limit,
            "random_state": seed,
            "pairs": [[int(a), int(b)] for a, b in pairs],
            "matched_treated_indices": [int(i) for i in matched_treated.tolist()],
            "matched_control_indices": [int(i) for i in matched_control.tolist()],
            "balance": [row.to_dict() for row in rows],
        }
    )
    return MatchingResult(
        pairs=pairs,
        matched_treated_indices=[int(i) for i in matched_treated.tolist()],
        matched_control_indices=[int(i) for i in matched_control.tolist()],
        n_pairs=len(pairs),
        n_unmatched_treated=int(treated_idx.shape[0] - len(pairs)),
        n_unmatched_control=int(control_idx.shape[0] - len(pairs)),
        caliper=limit,
        random_state=seed,
        balance=rows,
        digest=digest,
    )


def adjustment_suite_digest(*, matched: "MatchingResult", iptw: "IPTWResult") -> str:
    """One origin digest covering the full adjustment suite (review finding).

    A Tool Card that declares matched pairs, balance diagnostics AND IPTW
    weights as outputs must bind all of them: an origin over the matching
    result alone lets weight changes pass silently. Both results must come
    from the same inputs (enforced by comparing their embedded input
    digests is left to the caller; this function binds the two payloads).
    """

    if not isinstance(matched, MatchingResult):
        raise PropensityWeightingError("matched must be a MatchingResult")
    if not isinstance(iptw, IPTWResult):
        raise PropensityWeightingError("iptw must be an IPTWResult")
    return _canonical_digest(
        {
            "kind": "confounding_adjustment_suite",
            "matching": matched.to_json(),
            "iptw": iptw.to_json(),
        }
    )


def _effective_sample_size(weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    squares = float(np.sum(weights * weights))
    if squares <= 0.0 or not math.isfinite(total) or not math.isfinite(squares):
        return 0.0
    return float(total * total / squares)


def compute_iptw_weights(
    treatment: Any,
    propensity_scores: Any,
    *,
    stabilized: bool = True,
    truncate_quantiles: Optional[Sequence[float]] = None,
    eps: float = _PS_EPS,
) -> IPTWResult:
    """Stabilized IPTW weights with optional quantile truncation.

    ``truncate_quantiles=(lo, hi)`` clips weights at those quantiles of the
    weight distribution (``None`` disables truncation). Summaries report
    ``min``/``p50``/``max`` plus the effective sample size (overall and per
    arm). No outcome is involved and none is estimated.
    """
    treated = _coerce_treatment(treatment)
    scores = _coerce_propensity_scores(propensity_scores, treated.shape[0])
    if isinstance(stabilized, bool) is False:
        raise PropensityWeightingError("stabilized must be a bool")
    bounds = _check_truncate_quantiles(truncate_quantiles)
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise PropensityWeightingError("eps must be a positive number")
    eps_value = float(eps)
    if not math.isfinite(eps_value) or not 0.0 < eps_value < 0.5:
        raise PropensityWeightingError("eps must satisfy 0 < eps < 0.5")

    clipped = np.clip(scores, eps_value, 1.0 - eps_value)
    n = int(treated.shape[0])
    treated_rate = float(np.sum(treated == 1) / n)
    if stabilized:
        weight_t = treated_rate / clipped
        weight_c = (1.0 - treated_rate) / (1.0 - clipped)
    else:
        weight_t = 1.0 / clipped
        weight_c = 1.0 / (1.0 - clipped)
    weights = np.where(treated == 1, weight_t, weight_c).astype(float)
    if not np.all(np.isfinite(weights)):
        raise PropensityWeightingError("IPTW weights must be finite")

    if bounds is not None:
        low = float(np.quantile(weights, bounds[0]))
        high = float(np.quantile(weights, bounds[1]))
        weights = np.clip(weights, low, high)

    treated_w = weights[treated == 1]
    control_w = weights[treated == 0]
    # The digest binds the computed weights, not just the inputs.
    digest = _canonical_digest(
        {
            "kind": "iptw_weights",
            "treatment": treated.tolist(),
            "propensity_scores": [float(v) for v in scores.tolist()],
            "stabilized": bool(stabilized),
            "truncate_quantiles": None if bounds is None else [bounds[0], bounds[1]],
            "eps": eps_value,
            "weights": [float(v) for v in weights.tolist()],
        }
    )
    return IPTWResult(
        weights=[float(v) for v in weights.tolist()],
        stabilized=bool(stabilized),
        truncate_quantiles=None if bounds is None else [bounds[0], bounds[1]],
        weight_min=float(np.min(weights)),
        weight_p50=float(np.quantile(weights, 0.5)),
        weight_max=float(np.max(weights)),
        weight_mean=float(np.mean(weights)),
        ess=_effective_sample_size(weights),
        ess_treated=_effective_sample_size(treated_w),
        ess_control=_effective_sample_size(control_w),
        n=n,
        digest=digest,
    )


__all__ = [
    "ANALYSIS_ONLY_NOTE",
    "DEFAULT_C",
    "DEFAULT_CALIPER",
    "DEFAULT_RANDOM_STATE",
    "adjustment_suite_digest",
    "BalanceRow",
    "IPTWResult",
    "MatchingResult",
    "PropensityScoreResult",
    "PropensityWeightingError",
    "compute_iptw_weights",
    "estimate_propensity_scores",
    "match_nearest_neighbor",
    "standardized_mean_difference",
]


def _numeric(series: pd.Series, name: str, *, missing: bool) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & numeric.isna()
    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    if bool(invalid.any()) or bool(np.isinf(values).any()):
        raise ValueError(f"covariate_balance_invalid_numeric: {name}")
    if not missing and bool(np.isnan(values).any()):
        raise ValueError(f"covariate_balance_missing_numeric: {name}")
    return values


def _moments(values: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    positive = weights > 0
    n = int(values.size)
    weighted_n = int(positive.sum())
    # Scaling avoids overflow in ESS and makes it invariant to weight scale.
    scaled = weights[positive] / weights[positive].max() if weighted_n else np.array([])
    total = float(weights.sum())
    if not math.isfinite(total):
        raise ValueError("covariate_balance_weight_sum_nonfinite")
    return {
        "n": n,
        "positive": weighted_n,
        "zero": n - weighted_n,
        "weight_sum": total,
        "ess": float(scaled.sum() ** 2 / np.square(scaled).sum())
        if weighted_n
        else 0.0,
        "mean": float(np.mean(values)) if n else None,
        "variance": float(np.var(values, ddof=1)) if n > 1 else None,
        "weighted_mean": float(np.average(values[positive], weights=scaled))
        if weighted_n
        else None,
    }


def _contrast(
    mean_t: float | None, mean_c: float | None, denominator: float | None
) -> tuple[float | None, str]:
    if mean_t is None or mean_c is None or denominator is None:
        return None, "insufficient_observations"
    if denominator == 0:
        return (
            (0.0, "constant_equal")
            if mean_t == mean_c
            else (None, "zero_variance_different_means")
        )
    return (float((mean_t - mean_c) / denominator), "estimated")


def compute_covariate_balance(
    frame: pd.DataFrame, spec: CovariateBalanceSpec | Mapping[str, Any]
) -> CovariateBalanceResult:
    """One unique declared unit per row; no refitting or row deduplication."""
    spec = CovariateBalanceSpec.model_validate(spec)
    required = {
        spec.unit_id_column,
        spec.treatment_column,
        *(item.column for item in spec.covariates),
    }
    if spec.weight_column is not None:
        required.add(spec.weight_column)
    if not frame.columns.is_unique or not required.issubset(frame.columns):
        raise ValueError("covariate_balance_missing_or_duplicate_columns")
    if frame.empty:
        raise ValueError("covariate_balance_empty_input")
    ids = frame[spec.unit_id_column]
    if (
        ids.isna().any()
        or ids.astype(str).str.strip().eq("").any()
        or ids.duplicated().any()
        or ids.astype(str).duplicated().any()
    ):
        raise ValueError("covariate_balance_analysis_unit_not_unique")
    treatment = _numeric(
        frame[spec.treatment_column], spec.treatment_column, missing=False
    )
    if set(treatment.tolist()) != {0.0, 1.0}:
        raise ValueError("covariate_balance_treatment_requires_both_binary_groups")
    weights = np.ones(len(frame))
    if spec.weight_column is not None:
        weights = _numeric(frame[spec.weight_column], spec.weight_column, missing=False)
        if bool((weights < 0).any()):
            raise ValueError(
                f"covariate_balance_negative_weights: n={int((weights < 0).sum())}"
            )
        if any(not bool((weights[treatment == group] > 0).any()) for group in (0, 1)):
            raise ValueError("covariate_balance_empty_weighted_group")
    group_counts = {group: int((treatment == group).sum()) for group in (0, 1)}
    rows: list[CovariateBalanceRow] = []
    for covariate in spec.covariates:
        source = frame[covariate.column]
        if covariate.kind == "categorical":
            observed = source.notna().to_numpy()
            spellings: dict[str, str] = {}
            for value in source.dropna().tolist():
                spelling, identity_class = (
                    level_spelling(value),
                    level_identity_class(value),
                )
                if isinstance(value, str) and value != value.strip():
                    raise ValueError(
                        f"covariate_balance_noncanonical_level: {covariate.column}"
                    )
                if spelling in spellings and spellings[spelling] != identity_class:
                    raise ValueError(
                        f"covariate_balance_ambiguous_level_types: {covariate.column}"
                    )
                spellings[spelling] = identity_class
            labels = source.map(
                lambda value: level_spelling(value) if pd.notna(value) else None
            )
            if not set(labels.dropna()).issubset(set(covariate.levels)):
                raise ValueError(
                    f"covariate_balance_undeclared_levels: {covariate.column}"
                )
            components = [
                (level, labels.eq(level).to_numpy(dtype=float))
                for level in covariate.levels
            ]
        else:
            values = _numeric(source, covariate.column, missing=True)
            observed = ~np.isnan(values)
            if covariate.kind == "binary" and not set(values[observed]).issubset(
                {0.0, 1.0}
            ):
                raise ValueError(
                    f"covariate_balance_binary_invalid: {covariate.column}"
                )
            components = [(None, values)]
        for level, values in components:
            masks = {group: observed & (treatment == group) for group in (0, 1)}
            moments = {
                group: _moments(values[masks[group]], weights[masks[group]])
                for group in (0, 1)
            }
            t, c = moments[1], moments[0]
            denominator = (
                math.sqrt((t["variance"] + c["variance"]) / 2)
                if t["variance"] is not None and c["variance"] is not None
                else None
            )
            _, before_status = _contrast(t["mean"], c["mean"], denominator)
            before = standardized_mean_difference(values[masks[1]], values[masks[0]])
            after, after_status = _contrast(
                t["weighted_mean"], c["weighted_mean"], denominator
            )
            if spec.weight_column is None:
                after, after_status = None, "not_requested"
            elif not t["positive"] or not c["positive"]:
                after, after_status = None, "empty_weighted_group"
            threshold = spec.abs_smd_threshold
            rows.append(
                CovariateBalanceRow(
                    row_role="category_indicator"
                    if covariate.kind == "categorical"
                    else "covariate",
                    covariate=covariate.column,
                    kind=covariate.kind,
                    level=level,
                    n_treated=t["n"],
                    n_control=c["n"],
                    missing_treated_n=group_counts[1] - t["n"],
                    missing_control_n=group_counts[0] - c["n"],
                    positive_weight_treated_n=t["positive"],
                    positive_weight_control_n=c["positive"],
                    zero_weight_treated_n=t["zero"],
                    zero_weight_control_n=c["zero"],
                    weight_sum_treated=t["weight_sum"],
                    weight_sum_control=c["weight_sum"],
                    ess_treated=t["ess"],
                    ess_control=c["ess"],
                    mean_treated_before=t["mean"],
                    mean_control_before=c["mean"],
                    variance_treated_before=t["variance"],
                    variance_control_before=c["variance"],
                    smd_denominator=denominator,
                    smd_before=before,
                    smd_after=after,
                    mean_treated_after=t["weighted_mean"]
                    if spec.weight_column
                    else None,
                    mean_control_after=c["weighted_mean"]
                    if spec.weight_column
                    else None,
                    status_before=before_status,
                    status_after=after_status,
                    exceeds_threshold_before=abs(before) > threshold
                    if before is not None and threshold is not None
                    else None,
                    exceeds_threshold_after=abs(after) > threshold
                    if after is not None and threshold is not None
                    else None,
                )
            )
    return CovariateBalanceResult(
        specification_sha256=covariate_balance_spec_sha256(spec),
        analysis_unit=spec.analysis_unit,
        input_n=len(frame),
        treated_n=group_counts[1],
        control_n=group_counts[0],
        zero_weight_n=int((weights == 0).sum()),
        rows=tuple(rows),
    )


# Shared ownership and output replay for the descriptive balance diagnostic.
from pathlib import Path
from ..contracts.cohort_product_keys import sole_typed_cohort_input
from ..contracts.ownership_verdict import OwnershipVerdict

def covariate_balance_executor_verdict(step: AnalysisStep) -> OwnershipVerdict:
    spec = getattr(step, "covariate_balance_spec", None)
    if step.scientific_action_id != BALANCE_ACTION and spec is None:
        return OwnershipVerdict.wrong_shape(
            BALANCE_KIND, reason="different scientific action"
        )
    if spec is None:
        return OwnershipVerdict.incomplete_declaration(
            BALANCE_KIND,
            missing=("covariate_balance_spec",),
            reason="Declare the exact covariates, units, treatment and existing weights.",
        )
    required = {
        spec.unit_id_column,
        spec.treatment_column,
        *(item.column for item in spec.covariates),
    }
    if spec.weight_column:
        required.add(spec.weight_column)
    cohort = sole_typed_cohort_input(step)
    valid = (
        step.scientific_action_id == BALANCE_ACTION
        and cohort is not None
        and len(step.inputs) == len(set(step.inputs))
        and {key for key in step.inputs if ":" in key} == {cohort}
        and {key for key in step.inputs if ":" not in key} == required
        and tuple(step.expected_outputs) == BALANCE_OUTPUTS
        and step.planned_analysis_role in {"secondary", "auxiliary"}
        and not step.model_requirements
        and not step.family_primary_result_requirement
        and not any(
            getattr(step, name, None) is not None
            for name in type(step).model_fields
            if name.endswith("_spec") and name != "covariate_balance_spec"
        )
    )
    if not valid:
        return OwnershipVerdict.incomplete_declaration(
            BALANCE_KIND,
            missing=("covariate_balance_step_contract",),
            reason="Declare one typed cohort, the exact spec columns and balance table/receipt outputs without a model refit.",
        )
    return OwnershipVerdict.claim(
        BALANCE_KIND, reason="exact declared descriptive balance diagnostic"
    )


def covariate_balance_executor_owns_step(step: AnalysisStep) -> bool:
    return covariate_balance_executor_verdict(step).claimed


def _check_unit_contract(
    binding: Mapping[str, Any], spec: CovariateBalanceSpec
) -> None:
    contract = binding.get("product_contract", {})
    if contract.get("row_identity_column", spec.unit_id_column) != spec.unit_id_column:
        raise ValueError("covariate_balance_input_identity_contract_mismatch")
    if contract.get("analysis_unit", spec.analysis_unit) != spec.analysis_unit:
        raise ValueError("covariate_balance_input_analysis_unit_mismatch")


def covariate_balance_output_findings(
    *,
    step: AnalysisStep,
    step_summary: Mapping[str, Any],
    context: Any = None,
    resolved_input_bindings: Mapping[str, Mapping[str, Any]] | None = None,
    out_dir: Path | None = None,
) -> list[ValidationFinding]:
    from ..contracts.artifact_consumption import load_typed_input, sha256_file
    from ..schema import ValidationFinding
    import pandas as pd

    if step.scientific_action_id != BALANCE_ACTION:
        return []
    try:
        if (
            not covariate_balance_executor_owns_step(step)
            or out_dir is None
            or resolved_input_bindings is None
        ):
            raise ValueError(
                "balance validation requires the exact declared step and verified inputs"
            )
        spec = step.covariate_balance_spec
        if context is not None and spec.unit_id_column not in context.cohort.id_columns:
            raise ValueError("balance identity is not a declared cohort identity")
        cohort = load_typed_input(
            input_key=sole_typed_cohort_input(step),
            run_dir=Path(out_dir).resolve().parents[2],
            resolved_inputs={
                "step_id": step.step_id,
                "inputs": dict(resolved_input_bindings),
            },
            step_id=step.step_id,
            require_consumption_contract=True,
            minimum_row_count=1,
        )
        _check_unit_contract(cohort.binding, spec)
        expected = compute_covariate_balance(cohort.frame, spec).model_dump(mode="json")
        receipt_path = Path(out_dir) / "covariate_balance_receipt.json"
        table_path = Path(out_dir) / "covariate_balance.csv"
        if receipt_path.is_symlink() or table_path.is_symlink():
            raise ValueError("balance outputs must be regular local files")
        receipt = json.loads(receipt_path.read_text("utf-8"))
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("refit_performed") is not False
        ):
            raise ValueError(
                "balance receipt must preserve its typed no-refit contract"
            )
        expected_csv = pd.DataFrame(expected["rows"]).to_csv(index=False)
        receipt_metadata = {
            "schema_version": "easyicu.covariate_balance_receipt/1",
            "variance_convention": "equal_weight_pooled_unweighted_sample_variance_ddof1_fixed_before_after",
            "ess_formula": "(sum(w))^2/sum(w^2)",
            "authority_scope": "analysis_only",
            "refit_performed": False,
        }
        if any(receipt.get(key) != value for key, value in receipt_metadata.items()):
            raise ValueError(
                "balance receipt changes its calculation or authority contract"
            )
        if (
            receipt.get("specification") != spec.model_dump(mode="json")
            or canonical_sha256(receipt.get("result")) != canonical_sha256(expected)
            or receipt.get("result_sha256") != canonical_sha256(expected)
            or receipt.get("source_cohort_sha256") != cohort.sha256
            or receipt.get("output_sha256") != sha256_file(table_path)
            or table_path.read_text("utf-8") != expected_csv
        ):
            raise ValueError(
                "balance result or provenance differs from independent cohort replay"
            )
        if (
            step_summary.get("step_id") != step.step_id
            or step_summary.get("deterministic_standard_analysis") != BALANCE_KIND
            or step_summary.get("specification_sha256")
            != expected["specification_sha256"]
            or step_summary.get("result_sha256") != canonical_sha256(expected)
            or step_summary.get("source_cohort_sha256") != cohort.sha256
            or step_summary.get("n_total") != expected["input_n"]
            or step_summary.get("analysis_unit") != spec.analysis_unit
            or step_summary.get("authority_scope") != "analysis_only"
            or step_summary.get("paper_authorization_allowed") is not False
            or step_summary.get("refit_performed") is not False
            or step_summary.get("output_files")
            != dict(zip(BALANCE_OUTPUTS, (table_path.name, receipt_path.name)))
        ):
            raise ValueError("balance step summary differs from its sealed result")
        expected_summary = {
            "scientific_action_id": BALANCE_ACTION,
            "claim_ceiling": "analysis_only",
            "analysis_family": "causal_emulation",
            "status": "ok",
            "analysis_status": "ok",
            "interpretation": "measured_covariate_balance_descriptive_no_causal_effect",
        }
        if any(
            step_summary.get(key) != value for key, value in expected_summary.items()
        ):
            raise ValueError("balance summary changes its action or interpretation")
        return []
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        return [
            ValidationFinding(
                validator="covariate_balance_output",
                severity="error",
                message=str(exc),
                detail={
                    "step_id": step.step_id,
                    "reason_code": "covariate_balance_output_mismatch",
                },
            )
        ]
