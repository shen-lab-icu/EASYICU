"""Interpretation-free facts derived from one sealed cohort snapshot.

This leaf owns facts that can be recomputed from the exact staged table bytes.
It deliberately does not assign analysis roles, exposures, outcomes, methods,
covariates, or estimands.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


def observed_domain_for_series(series: pd.Series) -> Optional[Dict[str, Any]]:
    """Return the canonical domain actually observed in one physical column.

    ``levels`` is the exact set of values seen in this sealed cohort.  It is an
    observation, not a codebook: a study that needs a declared category list --
    one that includes a level this cohort happens not to contain -- must get it
    from the Planner, and ``allowed_values_basis`` says which of the two a
    consumer is holding.

    A numeric column gets its level set on the same terms as a text one.  The
    first version emitted ``levels`` for any text column with at most eight
    distinct values but, among numerics, only for a 0/1 binary -- so whether a
    variable had a usable category list depended on how it happened to be
    stored.  Every ordinal clinical score is stored as a number, and a real run
    died on exactly that: E3's ``aki_stage_max`` (KDIGO stage, 0/1/2/3 across
    all 93,762 rows) reached the generated code with no ``allowed_values``, and
    the code refused -- correctly, because the host's own guidance forbids
    recovering a closed category list from prose, from the broader context, or
    from the loaded frame.  One missing list, two dead steps: the primary
    association raised, and its figure was faulted for not enforcing the four
    stages or displaying the zero-count one.

    Values must be integral to qualify.  A continuous measurement is not a
    category set merely because a small cohort saw few of its values; requiring
    integrality costs nothing on real cohorts (measured across three: not one
    non-integral column has eight or fewer distinct values) and keeps a sparse
    lab result from arriving as a closed domain.  The eight-value cap does the
    rest: on the full E3 cohort ``aki_stage_max`` has 4 distinct values and
    ``aki_stage_first_time`` -- an hour offset, no kind of category -- has 25.
    """

    nonnull = series.dropna()
    if len(nonnull) == 0:
        return None
    n_unique = int(nonnull.nunique())
    domain: Dict[str, Any] = {
        "n_unique": n_unique,
        "is_constant": n_unique <= 1,
        # A binary fact is numeric {0, 1}; two-level categorical labels remain
        # categorical and are surfaced below without reinterpretation.
        "is_binary": False,
    }
    if pd.api.types.is_numeric_dtype(nonnull):
        try:
            domain["min"] = float(nonnull.min())
            domain["max"] = float(nonnull.max())
        except (TypeError, ValueError):
            pass
        if n_unique <= 2:
            try:
                values = {
                    int(value)
                    for value in nonnull.unique()
                    if float(value).is_integer()
                }
                domain["is_binary"] = values.issubset({0, 1}) and bool(values)
            except (TypeError, ValueError):
                domain["is_binary"] = False
        if domain["is_binary"] and n_unique == 2:
            if pd.api.types.is_bool_dtype(nonnull):
                domain["levels"] = [False, True]
            elif pd.api.types.is_integer_dtype(nonnull):
                domain["levels"] = [0, 1]
            else:
                domain["levels"] = [0.0, 1.0]
        elif 2 <= n_unique <= 8:
            # Two observed values at minimum.  One value is not a domain: a
            # column where only ``1`` appeared cannot be told apart from a
            # binary whose other level this cohort happens to miss, which is
            # the rule ``test_single_observed_binary_value_does_not_invent_
            # missing_level`` already fixed for the binary branch.
            try:
                observed = list(nonnull.unique())
                if pd.api.types.is_bool_dtype(nonnull):
                    domain["levels"] = sorted(bool(value) for value in observed)
                elif all(float(value).is_integer() for value in observed):
                    domain["levels"] = (
                        sorted(int(value) for value in observed)
                        if pd.api.types.is_integer_dtype(nonnull)
                        else sorted(float(value) for value in observed)
                    )
            except (TypeError, ValueError):
                pass
    elif n_unique <= 8:
        try:
            domain["levels"] = sorted(str(value) for value in nonnull.unique())
        except (TypeError, ValueError):
            pass
    return domain


__all__ = ["observed_domain_for_series"]
