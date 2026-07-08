"""[Layer 3: Safe Analytical Runtime] Proportional-hazards diagnostic.

The survival PRIMARY effect in this package is a Cox proportional-hazards
model, whose single reported hazard ratio (HR) is only meaningful if the
effect of each covariate is *constant over follow-up*. When a covariate's
effect changes with time (e.g. an exposure that hurts early but helps late,
or two survival curves that cross), one HR averages two opposite regimes and
is misleading. This module tests that assumption; it does **not** itself
estimate an effect.

Method — Grambsch & Therneau (1994)
-----------------------------------
Fit the Cox model, then look at the **scaled Schoenfeld residuals**. At each
event time the Schoenfeld residual for covariate ``j`` is the difference
between the covariate value of the subject who failed and the risk-set
weighted mean::

    r_j(t_i) = x_j(i) - sum_{k in R(t_i)} x_j(k) * w_k(t_i)

where ``w_k`` are the Cox partial-likelihood risk weights. Grambsch &
Therneau scale these by the estimated information matrix so that, under the
proportional-hazards null, the *scaled* residual has expectation equal to the
true (constant) coefficient and no trend in time.

The test regresses the scaled residual on a transform ``g(t)`` of event time
and asks whether the slope differs from zero::

    beta_j(t) ~= beta_j + theta_j * g(t)         H0: theta_j = 0

The per-covariate statistic is a chi-square with 1 df; a small p-value flags a
**time-varying effect** (PH violated). The ``global`` row is a **family-wise
(Bonferroni) summary** across covariates: its ``p_value`` is
``min(1, k * min_j p_j)`` for ``k`` covariates and its ``test_statistic`` is
that of the most-significant covariate. This is a deliberately conservative
"does ANY covariate violate PH" test. It is NOT the joint Grambsch-Therneau
chi-square (``sum_j chi2_j`` on ``k`` df): summing the marginal chi-squares
ignores the off-diagonal covariance of the scaled Schoenfeld residuals across
covariates and equals the joint statistic only when they are uncorrelated —
false for any adjusted ICU Cox model, where it can inflate OR deflate the
global in either direction (verified against R ``survival::cox.zph``). A true
joint GT global needs the full residual covariance (not exposed by lifelines'
per-covariate ``proportional_hazard_test``) and is a future refinement. The
time transform ``g(t)`` matters: ``"km"`` (1 minus the Kaplan-Meier survival,
the lifelines/survival default), ``"rank"``, ``"log"``, or ``"identity"``.

Reference
---------
Grambsch, P. M. & Therneau, T. M. (1994). "Proportional hazards tests and
diagnostics based on weighted residuals." *Biometrika* 81(3), 515-526.
Implemented here via ``lifelines`` (``CoxPHFitter`` +
``lifelines.statistics.proportional_hazard_test``), a curated, importable
dependency. A correct from-scratch scaled-Schoenfeld test requires the full
risk-set information matrix and is error-prone, so rather than ship a
questionable numpy fallback this module raises a clear, actionable
:class:`PHTestUnavailableError` when ``lifelines`` is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


class PHTestUnavailableError(RuntimeError):
    """Raised when the proportional-hazards test cannot run.

    Derived from :class:`RuntimeError` so callers that catch ``RuntimeError``
    (the pipeline's degradation ladder) treat a missing ``lifelines`` install
    as a recoverable "capability unavailable" signal rather than a crash.
    """


def _require_lifelines():
    """Import lifelines lazily, raising an actionable error if it is absent.

    Guarded so that ``import easyicu.research_agent.ph_schoenfeld`` never fails
    at module import time on a pandas+numpy+scipy-only install. The cost of a
    wrong from-scratch Schoenfeld test is a silently mis-stated PH verdict, so
    we refuse to fake it: callers get a clear install instruction instead.
    """

    try:
        from lifelines import CoxPHFitter
        from lifelines.statistics import proportional_hazard_test
    except Exception as exc:  # pragma: no cover - exercised only without lifelines
        raise PHTestUnavailableError(
            "The proportional-hazards (Schoenfeld) diagnostic requires the "
            "'lifelines' package, which is not installed. Install it with "
            "`pip install lifelines` (or `pip install easyicu[methods]`). "
            "This module deliberately does not ship a from-scratch fallback "
            "because a subtly wrong scaled-Schoenfeld test would mis-state the "
            "proportional-hazards verdict for the survival primary model."
        ) from exc
    return CoxPHFitter, proportional_hazard_test


# Time transforms lifelines accepts for the residual-vs-time regression.
_VALID_TIME_TRANSFORMS = ("km", "rank", "log", "identity", "all")


@dataclass(frozen=True)
class PHTestResult:
    """Result of a Grambsch-Therneau proportional-hazards test.

    ``table`` is a per-covariate + ``global`` DataFrame with columns
    ``covariate``, ``test_statistic`` (chi-square) and ``p_value``. The
    convenience accessors read that table so downstream code never re-parses it.
    """

    table: "pd.DataFrame"
    duration_col: str
    event_col: str
    covariates: Sequence[str]
    time_transform: str = "km"
    notes: str = ""
    _global_label: str = field(default="global", repr=False)

    def violated(self, alpha: float = 0.05) -> List[str]:
        """Covariates whose per-covariate PH test is significant at ``alpha``.

        The ``global`` row is excluded — it is the joint test, not a covariate.
        A returned name means that covariate's effect is time-varying, so a
        single hazard ratio for it should not be reported without a caveat.
        """

        out: List[str] = []
        for _, row in self.table.iterrows():
            name = str(row["covariate"])
            if name == self._global_label:
                continue
            p = row["p_value"]
            if p is not None and float(p) < float(alpha):
                out.append(name)
        return out

    def global_p_value(self) -> float:
        """The family-wise (Bonferroni ``global``) p-value across covariates."""

        mask = self.table["covariate"] == self._global_label
        if not bool(mask.any()):
            raise KeyError("no 'global' row present in PH test table")
        return float(self.table.loc[mask, "p_value"].iloc[0])

    def is_violated(self, alpha: float = 0.05) -> bool:
        """True if the family-wise PH assumption is rejected at ``alpha``.

        Uses the conservative Bonferroni ``global`` p-value, so this is a
        "does any covariate violate PH" verdict, not a joint chi-square test.
        """

        return self.global_p_value() < float(alpha)


def ph_test(
    df: "pd.DataFrame",
    duration_col: str,
    event_col: str,
    covariates: Sequence[str],
    time_transform: str = "km",
) -> "pd.DataFrame":
    """Test the Cox proportional-hazards assumption (Grambsch & Therneau 1994).

    Parameters
    ----------
    df:
        One row per subject. Must contain ``duration_col`` (follow-up time),
        ``event_col`` (1 = event observed, 0 = censored) and every name in
        ``covariates``.
    duration_col, event_col:
        Column names for follow-up time and the event indicator.
    covariates:
        The covariates entered into the Cox model and tested individually.
    time_transform:
        ``g(t)`` used to regress the scaled Schoenfeld residuals on time. One of
        ``"km"`` (default, 1 - Kaplan-Meier), ``"rank"``, ``"log"``,
        ``"identity"``.

    Returns
    -------
    pandas.DataFrame
        One row per covariate plus a final ``global`` row, with columns
        ``covariate``, ``test_statistic`` (per-covariate chi-square, 1 df) and
        ``p_value``. A small per-covariate ``p_value`` flags a time-varying
        effect (PH assumption violated). The ``global`` row is a conservative
        family-wise (Bonferroni) summary — ``p = min(1, k * min_j p_j)`` — not
        the joint Grambsch-Therneau chi-square (see the module docstring).

    Raises
    ------
    PHTestUnavailableError
        If ``lifelines`` is not importable.
    ValueError
        On bad column names, empty covariates, or an unknown ``time_transform``.
    """

    import pandas as pd

    CoxPHFitter, proportional_hazard_test = _require_lifelines()

    if time_transform not in _VALID_TIME_TRANSFORMS:
        raise ValueError(
            f"time_transform must be one of {_VALID_TIME_TRANSFORMS}, "
            f"got {time_transform!r}"
        )
    covariates = list(covariates)
    if not covariates:
        raise ValueError("covariates must be a non-empty sequence")

    needed = [duration_col, event_col, *covariates]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    fit_df = df.loc[:, needed].copy()

    fitter = CoxPHFitter()
    fitter.fit(fit_df, duration_col=duration_col, event_col=event_col)

    results = proportional_hazard_test(
        fitter, fit_df, time_transform=time_transform
    )
    summary = results.summary  # rows indexed by covariate (and transform)

    rows: List[dict] = []
    per_cov: List[tuple] = []  # (stat, p) for covariates with a finite p
    for cov in covariates:
        stat, p = _lookup_covariate(summary, cov)
        rows.append(
            {
                "covariate": cov,
                "test_statistic": stat,
                "p_value": p,
            }
        )
        if p is not None and p == p:  # finite / not NaN
            per_cov.append((stat, float(p)))

    # Family-wise (Bonferroni) global: the p-value is min(1, k * min_j p_j) and
    # the reported statistic is that of the most-significant covariate. This is a
    # conservative "does ANY covariate violate PH" test. It is deliberately NOT
    # the joint Grambsch-Therneau chi-square (sum of marginal chi-squares on k
    # df): that sum ignores the off-diagonal covariance of the scaled Schoenfeld
    # residuals and equals the joint statistic only for uncorrelated covariates
    # (never true for an adjusted Cox model). A true joint global needs the full
    # residual covariance, which lifelines' per-covariate test does not expose.
    if per_cov:
        k = len(per_cov)
        min_stat, min_p = min(per_cov, key=lambda sp: sp[1])
        global_stat = min_stat
        global_p = float(min(1.0, k * min_p))
    else:
        global_stat, global_p = float("nan"), float("nan")
    rows.append(
        {
            "covariate": "global",
            "test_statistic": global_stat,
            "p_value": global_p,
        }
    )

    return pd.DataFrame(rows, columns=["covariate", "test_statistic", "p_value"])


def _lookup_covariate(summary, covariate: str):
    """Pull ``(test_statistic, p_value)`` for one covariate from the summary.

    lifelines indexes the summary by covariate, or by a ``(covariate,
    transform)`` MultiIndex when multiple transforms are requested. This handles
    both and returns ``(None, None)`` if the covariate is absent.
    """

    idx = summary.index
    row = None
    try:
        if getattr(idx, "nlevels", 1) > 1:
            sub = summary.xs(covariate, level=0)
            row = sub.iloc[0]
        else:
            row = summary.loc[covariate]
    except (KeyError, IndexError):
        return None, None

    stat = _first_present(row, ("test_statistic", "test_stat", "chi2", "statistic"))
    p = _first_present(row, ("p", "p_value", "pvalue"))
    return (
        None if stat is None else float(stat),
        None if p is None else float(p),
    )


def _first_present(row, keys: Sequence[str]):
    for key in keys:
        if key in row.index:
            return row[key]
    return None


def run_ph_test(
    df: "pd.DataFrame",
    duration_col: str,
    event_col: str,
    covariates: Sequence[str],
    time_transform: str = "km",
) -> PHTestResult:
    """Run :func:`ph_test` and wrap it in a :class:`PHTestResult`.

    The convenience entry point when a caller wants ``.violated()`` /
    ``.global_p_value()`` rather than the raw DataFrame.
    """

    table = ph_test(
        df,
        duration_col=duration_col,
        event_col=event_col,
        covariates=covariates,
        time_transform=time_transform,
    )
    return PHTestResult(
        table=table,
        duration_col=duration_col,
        event_col=event_col,
        covariates=list(covariates),
        time_transform=time_transform,
    )


__all__ = [
    "PHTestResult",
    "PHTestUnavailableError",
    "ph_test",
    "run_ph_test",
]
