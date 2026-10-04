"""A Cox fit that did not converge is not a survival result.

lifelines reports complete separation and Newton-Raphson non-convergence as
ConvergenceWarning and still returns coefficients: a separated covariate's
coefficient diverges (about -19, with a standard error near 1,600, on a
synthetic probe) while the exposure row stays finite.  The signed survival
suite checked only that the exposure row was finite, so it reported an
adjusted hazard ratio from a model that had not converged.  Both of its Cox
fits now refuse such a model and name lifelines' reason.

Synthetic study and seeded synthetic rows only (renal replacement therapy and
90-day mortality).
"""

from __future__ import annotations

import pytest

from tests.support.survival_sealed import run_signed_suite, sealed_survival, synthetic_survival_rows


def test_a_covariate_that_separates_the_deaths_stops_the_suite(tmp_path):
    _context, authority = sealed_survival(tmp_path)
    rows = synthetic_survival_rows()
    # Every death is in one sex: that covariate separates the events.
    rows.loc[rows["mort_90d"] == 1, "sex"] = 0.0

    # The adjusted model is fitted first and refuses in its own name; the
    # piecewise model, fitted after it, has its own test.
    with pytest.raises(ValueError, match="^landmark survival Cox model did not converge: Column sex"):
        run_signed_suite(authority, rows, tmp_path / "out")


def test_the_unseparated_suite_still_runs(tmp_path):
    _context, authority = sealed_survival(tmp_path)

    summary = run_signed_suite(authority, synthetic_survival_rows(), tmp_path / "out")

    assert summary["status"] == "ok"
