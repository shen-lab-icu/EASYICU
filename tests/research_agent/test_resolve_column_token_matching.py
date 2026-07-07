"""resolve_column must match a candidate as a whole TOKEN, not a raw substring.

The raw-substring fallback silently returned the wrong column across the figure
renderers (false-pass audit, 2026-07-07): the bare 'n' size candidate matched
'media(n)'/'mea(n)', 'surv' matched 'survival_time' (colliding the time and
survival-probability axes), etc. Token-boundary matching keeps the intended
partial matches ('lactate' in 'mean_lactate', 'n' in 'n_total') while dropping
the traps.
"""

from __future__ import annotations

import pandas as pd

from easyicu.research_agent.figures.base import _appears_as_token, resolve_column


def test_token_boundaries_reject_substring_traps():
    # traps the audit found
    assert not _appears_as_token("n", "median")
    assert not _appears_as_token("n", "mean")
    assert not _appears_as_token("surv", "survival_time")
    assert not _appears_as_token("or", "score")
    assert not _appears_as_token("hr", "chart")


def test_token_boundaries_keep_legitimate_matches():
    assert _appears_as_token("n", "n_total")
    assert _appears_as_token("lactate", "mean_lactate")
    assert _appears_as_token("hr", "hr_first")
    assert _appears_as_token("silhouette", "mean_silhouette")
    assert _appears_as_token("follow_up", "follow_up_time")
    assert _appears_as_token("ci_low", "ci_low")


def test_resolve_column_size_no_longer_grabs_median():
    # the exact phenotype WIDE-path bug (finding #7): bare 'n' had matched median
    df = pd.DataFrame(columns=["cluster", "median", "mean", "sd", "n_total"])
    assert resolve_column(df, ["n", "size", "count"]) == "n_total"


def test_resolve_column_exact_still_wins_first():
    df = pd.DataFrame(columns=["n_total", "n"])
    # exact 'n' beats token match on 'n_total'
    assert resolve_column(df, ["n"]) == "n"


def test_resolve_column_returns_none_when_no_token_match():
    df = pd.DataFrame(columns=["median", "mean", "sd"])
    assert resolve_column(df, ["n", "size", "count"]) is None


def test_resolve_column_survival_axes_do_not_collide():
    # 'surv' must not resolve the duration column; the explicit full-form
    # candidates still resolve the real columns.
    df = pd.DataFrame(columns=["survival_time", "survival_prob"])
    time_col = resolve_column(df, ["time", "duration", "survival_time"])
    prob_col = resolve_column(df, ["survival_prob", "survival_probability", "surv"])
    assert time_col == "survival_time"
    assert prob_col == "survival_prob"
    assert time_col != prob_col
