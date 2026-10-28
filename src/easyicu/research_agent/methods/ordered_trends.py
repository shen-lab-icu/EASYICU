"""Validated primitives for trends across ordered independent groups.

This module deliberately stops below the research-workflow layer.  It does not
choose variables, define a cohort, label clinical categories, write a result
table, or decide what belongs in a manuscript.  Those remain agent decisions.
The functions here only provide reusable, deterministic calculations that an
agent-authored analysis can call and that a validator can independently replay.

Two tests are exposed, plus ``normal_tail_probability`` for a Wald statistic:

``cochran_armitage_trend``
    A score test for a binary outcome across ordered groups.  Its scores are an
    explicit modelling choice: consecutive ranks are the default, but callers
    may pass clinically justified scores.  Consequently this test must not be
    described as using order alone.

``jonckheere_terpstra_trend``
    The Jonckheere--Terpstra test for an ordered shift in an ordinal or
    continuous outcome.  It uses individual observations, counts cross-group
    concordant/discordant pairs, gives response ties half credit in ``J``, and
    uses the tie-corrected asymptotic null variance.  Spearman correlation is
    not a substitute for this calculation.

Very small normal-tail probabilities are retained on the log scale.  The
machine-readable ``p_value`` is bounded below by ``1e-300`` so CSV/JSON output
never states ``p=0``; ``log_p_value`` and ``p_value_reporting`` preserve the
actual tail information.

References
----------
Jonckheere AR. A distribution-free k-sample test against ordered alternatives.
Biometrika. 1954;41:133-145.

Terpstra TJ. The asymptotic normality and consistency of Kendall's test against
trend, when ties are present in one ranking. Indagationes Mathematicae.
1952;14:327-333.

SAS/STAT FREQ documentation, Jonckheere--Terpstra test details:
https://support.sas.com/documentation/cdl/en/procstat/67528/HTML/default/
procstat_freq_details75.htm
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import norm
from scipy import stats
from scipy.special import expit
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial, Poisson
from statsmodels.miscmodels.ordinal_model import OrderedModel

Alternative = Literal["two-sided", "greater", "less"]
_MIN_REPORTED_P = 1e-300
_LOG_MIN_REPORTED_P = math.log(_MIN_REPORTED_P)


@dataclass(frozen=True)
class WilsonInterval:
    """Wilson score interval for a binomial proportion."""

    event_n: int
    n: int
    estimate: float
    ci_low: float
    ci_high: float
    alpha: float
    method: str = "Wilson score interval"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV-friendly record."""

        return asdict(self)


@dataclass(frozen=True)
class OrderedTrendResult:
    """Common machine-readable result returned by both ordered trend tests."""

    test_name: str
    input_n: int
    n: int
    excluded_n: int
    group_order: tuple[Any, ...]
    group_sizes: tuple[int, ...]
    alternative: Alternative
    statistic: float
    statistic_type: str
    z_statistic: float
    p_value: float
    p_value_reporting: str
    log_p_value: float
    negative_log10_p: float
    p_value_bounded: bool
    effect_size: float
    effect_size_name: str
    implementation: str
    score_scheme: str | None = None
    scores: tuple[float, ...] | None = None
    chi_square: float | None = None
    expected_statistic: float | None = None
    variance: float | None = None
    pair_count: int | None = None
    concordance_minus_discordance: float | None = None
    n_unique_outcomes: int | None = None
    outcome_tie_group_count: int | None = None
    tie_correction: bool = False
    continuity_correction: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV-friendly record."""

        payload = asdict(self)
        payload["group_order"] = list(self.group_order)
        payload["group_sizes"] = list(self.group_sizes)
        if self.scores is not None:
            payload["scores"] = list(self.scores)
        return payload


def _validate_alternative(alternative: str) -> Alternative:
    normalised = str(alternative).strip().lower()
    if normalised not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    return normalised  # type: ignore[return-value]


def _normal_tail(z_statistic: float, alternative: Alternative) -> dict[str, Any]:
    """Stable normal-tail probability plus a nonzero serialization value."""

    z = float(z_statistic)
    if not math.isfinite(z):
        raise ValueError("z_statistic must be finite")
    if alternative == "two-sided":
        log_p = min(0.0, math.log(2.0) + float(log_ndtr(-abs(z))))
    elif alternative == "greater":
        log_p = float(log_ndtr(-z))
    else:
        log_p = float(log_ndtr(z))

    bounded = log_p < _LOG_MIN_REPORTED_P
    p_value = _MIN_REPORTED_P if bounded else float(math.exp(log_p))
    if bounded:
        reporting = f"<{_MIN_REPORTED_P:.0e}"
    elif p_value < 0.001:
        reporting = f"{p_value:.3e}"
    else:
        reporting = f"{p_value:.6g}"
    return {
        "p_value": p_value,
        "p_value_reporting": reporting,
        "log_p_value": log_p,
        "negative_log10_p": -log_p / math.log(10.0),
        "p_value_bounded": bounded,
    }


def normal_tail_probability(
    z_statistic: float, alternative: str = "two-sided"
) -> dict[str, Any]:
    """The bounded normal-tail p-value record for one Wald statistic.

    The same serialization the trend tests use: a nonzero ``p_value`` with
    its ``log_p_value`` and display form, so a model-based trend p-value is
    reported exactly like a test-based one.
    """

    return _normal_tail(z_statistic, _validate_alternative(alternative))


def wilson_interval(
    event_n: int,
    n: int,
    *,
    alpha: float = 0.05,
) -> WilsonInterval:
    """Return a two-sided Wilson score interval for ``event_n / n``.

    Unlike a Wald interval, the Wilson interval stays inside ``[0, 1]`` and is
    well behaved at zero events or at all events.  An empty denominator is not
    estimable and therefore raises rather than fabricating a zero-width result.
    """

    if isinstance(event_n, bool) or isinstance(n, bool):
        raise TypeError("event_n and n must be integer counts")
    if int(event_n) != event_n or int(n) != n:
        raise ValueError("event_n and n must be integer counts")
    event_n = int(event_n)
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    if event_n < 0 or event_n > n:
        raise ValueError("event_n must be between 0 and n")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    z = float(norm.ppf(1.0 - float(alpha) / 2.0))
    estimate = event_n / n
    z2_over_n = z * z / n
    denominator = 1.0 + z2_over_n
    centre = (estimate + z2_over_n / 2.0) / denominator
    half_width = (
        z
        * math.sqrt(estimate * (1.0 - estimate) / n + z * z / (4.0 * n * n))
        / denominator
    )
    return WilsonInterval(
        event_n=event_n,
        n=n,
        estimate=float(estimate),
        ci_low=float(max(0.0, centre - half_width)),
        ci_high=float(min(1.0, centre + half_width)),
        alpha=float(alpha),
    )


def cochran_armitage_trend(
    event_counts: Sequence[int],
    totals: Sequence[int],
    *,
    scores: Sequence[float] | None = None,
    group_order: Sequence[Any] | None = None,
    alternative: Alternative = "two-sided",
) -> OrderedTrendResult:
    """Cochran--Armitage score test for a binary ordered-group trend.

    ``event_counts`` and ``totals`` contain one entry per ordered group.  If
    ``scores`` is omitted, consecutive scores ``0, 1, ..., k-1`` are used.  The
    caller must report that spacing assumption; it is not an order-only test.
    """

    alternative = _validate_alternative(alternative)
    events = np.asarray(list(event_counts), dtype=float)
    ns = np.asarray(list(totals), dtype=float)
    if events.ndim != 1 or ns.ndim != 1 or events.size != ns.size:
        raise ValueError("event_counts and totals must be one-dimensional and aligned")
    if events.size < 2:
        raise ValueError("at least two ordered groups are required")
    if not np.all(np.isfinite(events)) or not np.all(np.isfinite(ns)):
        raise ValueError("event_counts and totals must be finite")
    if not np.all(events == np.floor(events)) or not np.all(ns == np.floor(ns)):
        raise ValueError("event_counts and totals must be integer counts")
    if np.any(ns <= 0) or np.any(events < 0) or np.any(events > ns):
        raise ValueError("each total must be positive and contain its event count")

    if scores is None:
        score_values = np.arange(events.size, dtype=float)
        score_scheme = "consecutive_ordinal_ranks"
    else:
        score_values = np.asarray(list(scores), dtype=float)
        score_scheme = (
            "consecutive_ordinal_ranks"
            if np.array_equal(score_values, np.arange(events.size, dtype=float))
            else "prespecified_numeric_scores"
        )
    if score_values.shape != events.shape or not np.all(np.isfinite(score_values)):
        raise ValueError("scores must contain one finite value per ordered group")
    if np.any(np.diff(score_values) <= 0):
        raise ValueError(
            "scores must be strictly increasing in the declared group order"
        )

    if group_order is None:
        order = tuple(range(int(events.size)))
    else:
        order = tuple(group_order)
        if len(order) != events.size or len(set(order)) != len(order):
            raise ValueError("group_order must contain one unique label per group")

    n_total = float(ns.sum())
    event_total = float(events.sum())
    if event_total <= 0.0 or event_total >= n_total:
        raise ValueError("the pooled binary outcome must contain events and non-events")
    pooled_risk = event_total / n_total
    weighted_score_sum = float(np.dot(ns, score_values))
    observed_score_sum = float(np.dot(events, score_values))
    expected_score_sum = pooled_risk * weighted_score_sum
    # The usual Cochran-Armitage null conditions on both table margins.  The
    # N/(N-1) finite-population factor is therefore required; omitting it gives
    # the related unconditional binomial score variance, not the standard CA
    # test implemented by statsmodels/SAS.
    score_variance = (
        pooled_risk
        * (1.0 - pooled_risk)
        * (
            float(np.dot(ns, score_values * score_values))
            - weighted_score_sum * weighted_score_sum / n_total
        )
        * n_total
        / (n_total - 1.0)
    )
    if not math.isfinite(score_variance) or score_variance <= 0.0:
        raise ValueError("Cochran-Armitage variance is zero or non-finite")

    z_statistic = (observed_score_sum - expected_score_sum) / math.sqrt(score_variance)
    tail = _normal_tail(z_statistic, alternative)
    n_int = int(n_total)
    return OrderedTrendResult(
        test_name="Cochran-Armitage trend test",
        input_n=n_int,
        n=n_int,
        excluded_n=0,
        group_order=order,
        group_sizes=tuple(int(value) for value in ns.tolist()),
        alternative=alternative,
        statistic=float(z_statistic),
        statistic_type="z",
        z_statistic=float(z_statistic),
        chi_square=float(z_statistic * z_statistic),
        p_value=float(tail["p_value"]),
        p_value_reporting=str(tail["p_value_reporting"]),
        log_p_value=float(tail["log_p_value"]),
        negative_log10_p=float(tail["negative_log10_p"]),
        p_value_bounded=bool(tail["p_value_bounded"]),
        effect_size=float(z_statistic / math.sqrt(n_int)),
        effect_size_name="signed_z_over_sqrt_n",
        expected_statistic=float(expected_score_sum),
        variance=float(score_variance),
        implementation="easyicu.methods.ordered_trends.cochran_armitage_trend.v1",
        score_scheme=score_scheme,
        scores=tuple(float(value) for value in score_values.tolist()),
    )


def _is_numeric_label(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _resolve_group_order(
    observed_groups: list[Any], group_order: Sequence[Any] | None
) -> tuple[Any, ...]:
    unique_observed = list(dict.fromkeys(observed_groups))
    if group_order is None:
        if not all(_is_numeric_label(value) for value in unique_observed):
            raise ValueError(
                "group_order is required for non-numeric group labels; "
                "alphabetical order is not a scientific ordering"
            )
        return tuple(sorted(unique_observed, key=float))

    order = tuple(group_order)
    if len(order) < 2 or len(set(order)) != len(order):
        raise ValueError("group_order must contain at least two unique labels")
    missing = [value for value in unique_observed if value not in order]
    empty = [value for value in order if value not in unique_observed]
    if missing or empty:
        raise ValueError(
            "group_order must match the observed groups exactly; "
            f"missing={missing!r}, empty={empty!r}"
        )
    return order


def _falling_factorial_2(counts: np.ndarray) -> float:
    return float(np.sum(counts * (counts - 1.0)))


def _falling_factorial_3(counts: np.ndarray) -> float:
    return float(np.sum(counts * (counts - 1.0) * (counts - 2.0)))


def jonckheere_terpstra_trend(
    values: Sequence[float],
    groups: Sequence[Any],
    *,
    group_order: Sequence[Any] | None = None,
    alternative: Alternative = "two-sided",
) -> OrderedTrendResult:
    """Tie-corrected asymptotic Jonckheere--Terpstra ordered trend test.

    Missing/non-finite outcomes and missing group labels are removed pairwise.
    String labels require an explicit ``group_order``.  Numeric labels may be
    sorted numerically when no order is supplied, but callers are encouraged to
    pass the design order explicitly.  The returned ``J`` statistic counts an
    earlier-group value below a later-group value as one and a cross-group tie
    as one half.
    """

    alternative = _validate_alternative(alternative)
    raw_values = list(values)
    raw_groups = list(groups)
    if len(raw_values) != len(raw_groups):
        raise ValueError("values and groups must share length")
    if not raw_values:
        raise ValueError("values and groups are empty")

    clean_values: list[float] = []
    clean_groups: list[Any] = []
    for raw_value, raw_group in zip(raw_values, raw_groups):
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        group_missing = raw_group is None
        if not group_missing:
            try:
                group_missing = bool(np.asarray(raw_group != raw_group).item())
            except (TypeError, ValueError):
                group_missing = False
        if math.isfinite(value) and not group_missing:
            clean_values.append(value)
            clean_groups.append(raw_group)

    if len(clean_values) < 2:
        raise ValueError("fewer than two finite observations remain")
    order = _resolve_group_order(clean_groups, group_order)
    grouped = [
        np.sort(
            np.asarray(
                [
                    value
                    for value, group in zip(clean_values, clean_groups)
                    if group == label
                ],
                dtype=float,
            )
        )
        for label in order
    ]
    group_sizes = np.asarray([len(group) for group in grouped], dtype=float)
    if len(grouped) < 2 or np.any(group_sizes <= 0):
        raise ValueError("at least two non-empty ordered groups are required")

    # J = sum over earlier/later groups of wins + half cross-group ties.
    # Search in each sorted earlier group for every later observation, avoiding
    # an O(n^2) materialisation of all pairs.
    j_statistic = 0.0
    for earlier_index, earlier_values in enumerate(grouped[:-1]):
        for later_values in grouped[earlier_index + 1 :]:
            below = np.searchsorted(earlier_values, later_values, side="left")
            at_or_below = np.searchsorted(earlier_values, later_values, side="right")
            j_statistic += float(np.sum(below + 0.5 * (at_or_below - below)))

    n = len(clean_values)
    pair_count_float = (float(n * n) - float(np.dot(group_sizes, group_sizes))) / 2.0
    pair_count = int(round(pair_count_float))
    expected_j = pair_count / 2.0
    t_statistic = 2.0 * j_statistic - pair_count

    _, tie_counts_int = np.unique(
        np.asarray(clean_values, dtype=float), return_counts=True
    )
    tie_counts = tie_counts_int.astype(float)
    n2 = float(n * (n - 1))
    group2 = _falling_factorial_2(group_sizes)
    tie2 = _falling_factorial_2(tie_counts)
    variance_j = (n2 - group2) * (n2 - tie2) / (8.0 * n2)
    if n >= 3:
        n3 = float(n * (n - 1) * (n - 2))
        group3 = _falling_factorial_3(group_sizes)
        tie3 = _falling_factorial_3(tie_counts)
        variance_j += (n3 - group3) * (n3 - tie3) / (36.0 * n3)
    if not math.isfinite(variance_j) or variance_j <= 0.0:
        raise ValueError(
            "Jonckheere-Terpstra variance is zero or non-finite; "
            "the ordered trend is not estimable"
        )

    z_statistic = t_statistic / math.sqrt(4.0 * variance_j)
    tail = _normal_tail(z_statistic, alternative)
    return OrderedTrendResult(
        test_name="Jonckheere-Terpstra trend test",
        input_n=len(raw_values),
        n=n,
        excluded_n=len(raw_values) - n,
        group_order=order,
        group_sizes=tuple(int(value) for value in group_sizes.tolist()),
        alternative=alternative,
        statistic=float(j_statistic),
        statistic_type="J",
        z_statistic=float(z_statistic),
        p_value=float(tail["p_value"]),
        p_value_reporting=str(tail["p_value_reporting"]),
        log_p_value=float(tail["log_p_value"]),
        negative_log10_p=float(tail["negative_log10_p"]),
        p_value_bounded=bool(tail["p_value_bounded"]),
        effect_size=float(t_statistic / pair_count),
        effect_size_name="cross_group_concordance_minus_discordance",
        expected_statistic=float(expected_j),
        variance=float(variance_j),
        pair_count=pair_count,
        concordance_minus_discordance=float(t_statistic),
        n_unique_outcomes=int(len(tie_counts)),
        outcome_tie_group_count=int(np.sum(tie_counts > 1)),
        tie_correction=True,
        continuity_correction=False,
        implementation="easyicu.methods.ordered_trends.jonckheere_terpstra_trend.v1",
    )


__all__ = [
    "OutcomeFit",
    "OutcomeTerm",
    "fit_outcome_model",
    "proportional_odds_score",
    "OrderedTrendResult",
    "WilsonInterval",
    "cochran_armitage_trend",
    "jonckheere_terpstra_trend",
    "normal_tail_probability",
    "wilson_interval",
]


@dataclass(frozen=True)
class OutcomeTerm:
    term: str
    source_variable: str
    estimate: float
    ci_low: float
    ci_high: float
    se: float


@dataclass(frozen=True)
class OutcomeFit:
    estimate: float
    ci_low: float
    ci_high: float
    se: float
    n: int
    terms: tuple[OutcomeTerm, ...]
    diagnostics: dict
    variance_estimator: str
    cluster_count: int | None
    notes: str = ""
    converged: bool = True
    n_events: None = None
    separation_detected: None = None

    @property
    def point_estimate(self):
        return self.estimate


def proportional_odds_score(model, fitted, groups=None) -> dict:
    """LM score for unequal slopes, with OPG nuisance projection.

    For K categories and p slopes there are p(K-2) restrictions. Per-row
    derivatives are of log(P(Y=k)) wrt each cumulative logit. Threshold and
    common-slope scores are projected out before the quadratic score test.
    With declared groups, independent group sums supply the score covariance.
    """
    x = np.asarray(model.exog, dtype=float)
    y = np.asarray(model.endog, dtype=int)
    n, p = x.shape
    thresholds = model.transform_threshold_params(fitted.params)[1:-1]
    cdf = expit(thresholds[None, :] - (x @ np.asarray(fitted.params)[:p])[:, None])
    probabilities = np.diff(np.column_stack((np.zeros(n), cdf, np.ones(n))), axis=1)
    if np.any(probabilities <= 0):
        raise ValueError("proportional-odds score probabilities are numerically degenerate")
    j = np.arange(len(thresholds))
    eta_scores = cdf * (1 - cdf) * (
        (y[:, None] == j) / probabilities[:, :-1]
        - (y[:, None] == j + 1) / probabilities[:, 1:]
    )
    nuisance = np.column_stack((eta_scores, -x * eta_scores.sum(axis=1)[:, None]))
    extra = (-eta_scores[:, :-1, None] * x[:, None, :]).reshape(n, -1)
    df = extra.shape[1]
    if df == 0:
        return {"method": "proportional_odds_score", "df": 0, "statistic": 0.0,
                "p_value": 1.0, "violated": False}
    efficient = extra - nuisance @ np.linalg.lstsq(nuisance, extra, rcond=None)[0]
    if groups is not None:
        efficient = pd.DataFrame(efficient).groupby(np.asarray(groups), sort=False).sum().to_numpy()
    covariance = efficient.T @ efficient
    if np.linalg.matrix_rank(covariance) < df:
        raise ValueError("proportional-odds score test is not estimable for this design")
    score = efficient.sum(axis=0)
    statistic = max(0.0, float(score @ np.linalg.solve(covariance, score)))
    p_value = float(stats.chi2.sf(statistic, df))
    return {"method": "proportional_odds_score", "df": df, "statistic": statistic,
            "p_value": p_value, "violated": p_value < 0.05,
            "covariance": "cluster_robust" if groups is not None else "opg"}


def fit_outcome_model(
    *, X: pd.DataFrame, y: pd.Series, term: str, kind: str,
    ordered_levels=(), exposure_time=None, log_offset=None,
    variance_estimator: str = "model_based", cluster_groups=None,
    source_by_design_column: Mapping[str, str] | None = None,
) -> OutcomeFit:
    """Fit exactly the declared family, returning ratios with 95% Wald CIs.

    OR > 1 means greater odds of a higher declared category. Counts use NB2
    (Var(Y)=mu+alpha*mu²) or Poisson with log link. Offsets never enter as fitted
    predictors. The caller supplies the exact complete-case model matrix.
    """
    x = X.astype(float).copy()
    if (len(y) != len(x) or not X.index.equals(y.index) or term not in x
            or len(y) <= x.shape[1] + 3 or not np.isfinite(x.to_numpy()).all()
            or y.isna().any() or np.linalg.matrix_rank(x) != x.shape[1]):
        raise ValueError("finite aligned full-rank complete model rows and focal term required")
    if variance_estimator not in {"model_based", "cluster_robust"}:
        raise ValueError("unsupported covariance contract")
    fit_args: dict[str, Any] = {"disp": False, "maxiter": 300}
    groups = None
    if variance_estimator == "cluster_robust":
        if cluster_groups is None or len(cluster_groups) != len(x):
            raise ValueError("declared cluster groups required for every model row")
        if isinstance(cluster_groups, pd.Series) and not cluster_groups.index.equals(x.index):
            raise ValueError("cluster groups must align with exact model rows")
        groups = np.asarray(cluster_groups)
        if pd.isna(groups).any() or len(pd.unique(groups)) < 2:
            raise ValueError("at least two observed clusters required")
        fit_args.update(cov_type="cluster", cov_kwds={"groups": groups})
    elif cluster_groups is not None:
        raise ValueError("groups require an explicit cluster covariance contract")
    diagnostics = {}
    if kind == "proportional_odds":
        if exposure_time is not None or log_offset is not None:
            raise ValueError("offsets belong to count outcomes")
        from ..contracts.model_terms import level_spelling
        tokens = y.map(level_spelling)
        levels = list(ordered_levels)
        if len(levels) < 3 or len(set(levels)) != len(levels) or set(tokens) != set(levels):
            raise ValueError("every declared ordered level must be observed, with no undeclared levels")
        encoded = pd.Series(pd.Categorical(tokens, categories=levels, ordered=True), index=y.index)
        model = OrderedModel(encoded, x, distr="logit")
        fit = model.fit(method="bfgs", **fit_args)
        diagnostics["proportional_odds"] = proportional_odds_score(model, fit, groups)
        design_names = list(x.columns)
    elif kind in {"count_poisson", "negative_binomial"}:
        counts = np.asarray(y, dtype=float)
        if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
            raise ValueError("count outcome must contain finite nonnegative integers")
        if exposure_time is not None and log_offset is not None:
            raise ValueError("declare exposure time or log offset, never both")
        for values in (exposure_time, log_offset):
            if isinstance(values, pd.Series) and not values.index.equals(x.index):
                raise ValueError("offset/exposure time must align with exact model rows")
        if exposure_time is not None:
            times = np.asarray(exposure_time, dtype=float)
            if times.shape != counts.shape or not np.isfinite(times).all() or (times <= 0).any():
                raise ValueError("exposure time must be finite and strictly positive for every model row")
            offset = np.log(times)
        elif log_offset is not None:
            offset = np.asarray(log_offset, dtype=float)
            if offset.shape != counts.shape or not np.isfinite(offset).all():
                raise ValueError("log offset must be finite for every model row")
        else:
            offset = np.zeros(len(y))
        design = sm.add_constant(x, has_constant="raise")
        if np.linalg.matrix_rank(design) != design.shape[1]:
            raise ValueError("count model design must have full rank including intercept")
        poisson = Poisson(counts, design, offset=offset).fit(**fit_args)
        mu = np.asarray(poisson.predict())
        z = float(np.sum((counts - mu) ** 2 - counts) / np.sqrt(2 * np.sum(mu ** 2)))
        if groups is not None:
            dispersion_scores = pd.Series((counts - mu) ** 2 - counts).groupby(groups, sort=False).sum()
            z = float(dispersion_scores.sum() / np.sqrt(np.sum(dispersion_scores ** 2)))
        diagnostics["dispersion"] = {
            "method": "poisson_nb2_score", "statistic": z,
            "p_value": float(stats.norm.sf(z)), "overdispersed": bool(z > stats.norm.ppf(0.95)),
            "pearson_ratio": float(np.sum((counts - mu) ** 2 / mu) / (len(y) - design.shape[1])),
            "covariance": variance_estimator,
        }
        fit = (NegativeBinomial(counts, design, offset=offset, loglike_method="nb2").fit(**fit_args)
               if kind == "negative_binomial" else poisson)
        diagnostics["offset"] = "log_exposure" if exposure_time is not None else "log_offset" if log_offset is not None else "unit_exposure"
        if kind == "negative_binomial":
            diagnostics["alpha"] = float(fit.params["alpha"])
        design_names = list(design.columns)
    else:
        raise ValueError("unsupported outcome model kind")
    if not bool(fit.mle_retvals.get("converged", False)):
        raise ValueError("declared outcome model did not converge")
    intervals = np.asarray(fit.conf_int())
    params, errors = np.asarray(fit.params), np.asarray(fit.bse)
    source = dict(source_by_design_column or {})
    terms = tuple(OutcomeTerm(name, source.get(name, name), float(np.exp(params[i])),
                              float(np.exp(intervals[i, 0])), float(np.exp(intervals[i, 1])), float(errors[i]))
                  for i, name in enumerate(design_names))
    if not all(np.isfinite([row.estimate, row.ci_low, row.ci_high, row.se]).all() for row in terms):
        raise ValueError("declared outcome model returned non-finite effects or intervals")
    primary = next(row for row in terms if row.term == term)
    diagnostics["aic"] = float(fit.aic)
    diagnostics["effect_direction"] = "higher_ordered_category" if kind == "proportional_odds" else "higher_event_rate"
    violated = diagnostics.get("proportional_odds", {}).get("violated", False)
    return OutcomeFit(primary.estimate, primary.ci_low, primary.ci_high, primary.se, len(y), terms,
                      diagnostics, variance_estimator, len(pd.unique(groups)) if groups is not None else None,
                      "Proportional-odds assumption violated; common OR is model-dependent." if violated else "")
