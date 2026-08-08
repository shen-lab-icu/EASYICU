"""Fairness / subgroup analysis (O24).

Builds a deterministic subgroup forest-style summary for the primary
effect estimate. For every subgroup column we:

1. stratify the cohort by the subgroup values (age buckets for
   continuous variables, raw levels for categorical ones),
2. refit a minimal logistic(outcome ~ predictor) on each stratum,
3. fit a logistic with a predictor × subgroup interaction term on
   the pooled data and compute the interaction p-value,
4. emit a CSV + MD suitable for a paper forest plot.

Pure numpy. The module registers nothing with the pipeline on its
own; the pipeline glue lives in :mod:`pipeline` under the new
``enable_fairness_subgroups`` flag.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SubgroupEstimate:
    subgroup_column: str
    subgroup_value: str
    n: int
    odds_ratio: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    p_value: Optional[float]

    def to_json(self) -> Dict[str, Any]:
        return {
            "subgroup_column": self.subgroup_column,
            "subgroup_value": self.subgroup_value,
            "n": self.n,
            "odds_ratio": self.odds_ratio,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "p_value": self.p_value,
        }


@dataclass
class SubgroupAnalysisResult:
    predictor: str
    outcome: str
    multiplicity_family_id: str
    estimates: List[SubgroupEstimate] = field(default_factory=list)
    interaction_pvalues: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "predictor": self.predictor,
            "outcome": self.outcome,
            "multiplicity_family_id": self.multiplicity_family_id,
            "estimates": [e.to_json() for e in self.estimates],
            "interaction_pvalues": dict(self.interaction_pvalues),
            "notes": list(self.notes),
        }

    # -- persistence ----------------------------------------------------

    def write_csv(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "hypothesis_id",
                    "multiplicity_family_id",
                    "analysis_role",
                    "row_type",
                    "subgroup_column",
                    "subgroup_value",
                    "n",
                    "odds_ratio",
                    "ci_lower",
                    "ci_upper",
                    "p_value",
                ]
            )
            for e in self.estimates:
                writer.writerow(
                    [
                        f"subgroup_effect:{e.subgroup_column}:{e.subgroup_value}",
                        self.multiplicity_family_id,
                        "secondary",
                        "stratum_effect",
                        e.subgroup_column,
                        e.subgroup_value,
                        e.n,
                        e.odds_ratio if e.odds_ratio is not None else "",
                        e.ci_lower if e.ci_lower is not None else "",
                        e.ci_upper if e.ci_upper is not None else "",
                        e.p_value if e.p_value is not None else "",
                    ]
                )
            for column, p_value in sorted(self.interaction_pvalues.items()):
                writer.writerow(
                    [
                        f"subgroup_interaction:{column}",
                        self.multiplicity_family_id,
                        "secondary",
                        "interaction",
                        column,
                        "__interaction__",
                        "",
                        "",
                        "",
                        "",
                        p_value,
                    ]
                )
        return path

    def write_markdown(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Subgroup / fairness analysis (O24)",
            "",
            f"Primary predictor: **{self.predictor}**",
            f"Outcome: **{self.outcome}**",
            "",
            "| Subgroup | Level | N | OR | 95% CI | p | interaction p |",
            "|---|---|---|---|---|---|---|",
        ]
        for e in self.estimates:
            ci = (
                f"{e.ci_lower:.2f} – {e.ci_upper:.2f}"
                if e.ci_lower is not None and e.ci_upper is not None
                else "—"
            )
            lines.append(
                "| {sc} | {sv} | {n} | {or_:.2f} | {ci} | {p:.3g} | {ip} |".format(
                    sc=e.subgroup_column,
                    sv=e.subgroup_value,
                    n=e.n,
                    or_=e.odds_ratio if e.odds_ratio is not None else float("nan"),
                    ci=ci,
                    p=e.p_value if e.p_value is not None else float("nan"),
                    ip=(
                        f"{self.interaction_pvalues.get(e.subgroup_column, float('nan')):.3g}"
                        if e.subgroup_column in self.interaction_pvalues
                        else "—"
                    ),
                )
            )
        if self.notes:
            lines += ["", "## Notes", ""]
            for n in self.notes:
                lines.append(f"- {n}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Core fit helpers
# ---------------------------------------------------------------------------


def _fit_logistic_or(
    x: Any, y: Any, *, minimum_n: int
) -> Optional[Tuple[float, float, float, float]]:
    """Return (OR, ci_lo, ci_hi, p_value) for logistic(y ~ x)."""
    if np is None:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(x)
    if n < minimum_n or len(np.unique(y)) < 2:
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
    try:
        inv = np.linalg.inv(X.T @ (W[:, None] * X))
    except np.linalg.LinAlgError:
        return None
    se = float(math.sqrt(max(0.0, inv[1, 1])))
    if se == 0.0:
        return None
    coef = float(beta[1])
    lo = coef - 1.96 * se
    hi = coef + 1.96 * se
    z_stat = coef / se
    p_val = math.erfc(abs(z_stat) / math.sqrt(2.0))
    return math.exp(coef), math.exp(lo), math.exp(hi), p_val


def _fit_interaction_p(
    x: Any, y: Any, s: Any
) -> Optional[float]:
    """Return interaction p-value for y ~ x + s + x*s (binary or categorical s)."""
    if np is None:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    # s is arbitrary label; dummy-code it and drop the first level.
    uniq = sorted(set(s))
    if len(uniq) < 2:
        return None
    levels = uniq[1:]
    dummies = np.column_stack(
        [np.asarray([1.0 if v == lv else 0.0 for v in s]) for lv in levels]
    )
    interactions = dummies * x[:, None]
    X = np.column_stack([np.ones(len(x)), x, dummies, interactions])
    beta = np.zeros(X.shape[1])
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
    # Wald test for the interaction block: coefficients at positions
    # 2 + len(levels) .. end correspond to interactions.
    k_levels = len(levels)
    start = 2 + k_levels
    int_coefs = beta[start:]
    try:
        inv = np.linalg.inv(X.T @ (W[:, None] * X))
    except np.linalg.LinAlgError:
        return None
    cov = inv[start:, start:]
    try:
        chi2 = float(int_coefs @ np.linalg.solve(cov, int_coefs))
    except np.linalg.LinAlgError:
        return None
    # chi2 with df = k_levels; survival function via math.gammainc.
    # math provides lower regularised incomplete gamma only indirectly;
    # compute upper tail via series for small df (k <= 5).
    df = k_levels
    p_val = _upper_chi2(df, chi2)
    return p_val


def _upper_chi2(df: int, x: float) -> float:
    """Upper-tail CDF of chi2(df) evaluated at x."""
    # Use the series expansion of the regularised incomplete gamma
    # function for small df. For df <= 10 and x <= 50 this is plenty
    # accurate for p-value reporting.
    if x <= 0:
        return 1.0
    # Regularised lower incomplete gamma P(df/2, x/2).
    a = df / 2.0
    z = x / 2.0
    # Series form from Numerical Recipes; capped at 200 terms.
    term = 1.0 / a
    total = term
    for n in range(1, 200):
        term *= z / (a + n)
        total += term
        if abs(term) < 1e-12 * abs(total):
            break
    lower = total * math.exp(-z + a * math.log(z) - math.lgamma(a))
    return max(0.0, 1.0 - lower)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def _bucketise_continuous(values: Any, n_buckets: int = 4) -> List[str]:
    """Turn a continuous column into quantile-label strings."""
    if np is None:
        return [str(v) for v in values]
    arr = np.asarray(values, dtype=float)
    valid = ~np.isnan(arr)
    if valid.sum() < n_buckets:
        return [str(v) for v in arr]
    qs = np.quantile(arr[valid], np.linspace(0, 1, n_buckets + 1))
    labels = []
    for v in arr:
        if math.isnan(v):
            labels.append("missing")
            continue
        idx = int(np.searchsorted(qs[1:-1], v, side="right"))
        labels.append(f"q{idx + 1}")
    return labels


def run_subgroup_analysis(
    *,
    cohort_df: Any,
    predictor: str,
    outcome: str,
    subgroup_columns: Sequence[str],
    continuous_buckets: int = 4,
    minimum_axis_n: int = 50,
    minimum_stratum_n: int = 30,
    multiplicity_family_id: str,
) -> SubgroupAnalysisResult:
    """Fit the primary OR per subgroup + interaction p-values."""
    if np is None:
        return SubgroupAnalysisResult(
            predictor=predictor,
            outcome=outcome,
            multiplicity_family_id=multiplicity_family_id,
            notes=["numpy unavailable"],
        )
    if predictor not in cohort_df.columns or outcome not in cohort_df.columns:
        return SubgroupAnalysisResult(
            predictor=predictor,
            outcome=outcome,
            multiplicity_family_id=multiplicity_family_id,
            notes=[f"predictor/outcome not in cohort: {predictor}, {outcome}"],
        )
    result = SubgroupAnalysisResult(
        predictor=predictor,
        outcome=outcome,
        multiplicity_family_id=multiplicity_family_id,
    )
    for col in subgroup_columns:
        if col not in cohort_df.columns:
            result.notes.append(f"{col}: not in cohort; skipped")
            continue
        sub = cohort_df.dropna(subset=[predictor, outcome, col])
        if len(sub) < minimum_axis_n:
            result.notes.append(
                f"{col}: fewer than {minimum_axis_n} rows after dropna; skipped"
            )
            continue
        raw_values = sub[col].to_list()
        if sub[col].dtype.kind in {"i", "u", "f"} and len(set(raw_values)) > 6:
            levels = _bucketise_continuous(raw_values, n_buckets=continuous_buckets)
        else:
            levels = [str(v) for v in raw_values]
        unique_levels = sorted(set(levels))
        x = sub[predictor].astype(float).to_numpy()
        y = sub[outcome].astype(int).to_numpy()
        s_arr = np.asarray(levels)
        for level in unique_levels:
            mask = s_arr == level
            if mask.sum() < minimum_stratum_n:
                result.estimates.append(
                    SubgroupEstimate(
                        subgroup_column=col,
                        subgroup_value=level,
                        n=int(mask.sum()),
                        odds_ratio=None,
                        ci_lower=None,
                        ci_upper=None,
                        p_value=None,
                    )
                )
                continue
            fit = _fit_logistic_or(
                x[mask], y[mask], minimum_n=minimum_stratum_n
            )
            if fit is None:
                result.estimates.append(
                    SubgroupEstimate(
                        subgroup_column=col,
                        subgroup_value=level,
                        n=int(mask.sum()),
                        odds_ratio=None,
                        ci_lower=None,
                        ci_upper=None,
                        p_value=None,
                    )
                )
                continue
            or_, lo, hi, p = fit
            result.estimates.append(
                SubgroupEstimate(
                    subgroup_column=col,
                    subgroup_value=level,
                    n=int(mask.sum()),
                    odds_ratio=or_,
                    ci_lower=lo,
                    ci_upper=hi,
                    p_value=p,
                )
            )
        int_p = _fit_interaction_p(x, y, levels)
        if int_p is not None:
            result.interaction_pvalues[col] = int_p
    return result


__all__ = [
    "SubgroupAnalysisResult",
    "SubgroupEstimate",
    "run_subgroup_analysis",
]
