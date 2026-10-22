"""A wide estimate table's preview keeps the columns that name its primary contrast.

The browser's result summary reads the adjusted-association estimate table
from the run's bounded preview (twelve columns).  The producer writes
``is_primary_contrast``, ``reference_level`` and ``contrast`` last, so the
preview dropped them and every answer card fell back to a bare "OR 2.014"
with neither its groups nor its interval (pilot1 fixed set A2, 10-22).
"""

from __future__ import annotations

from easyicu.webserver import agent_pipeline_runs

# The producer's header, verbatim (pilot1 A2).
ESTIMATES = (
    "fit_status,estimate,ci_low,ci_high,effect_scale,exposure,model_id,requirement_id,"
    "outcome,covariates,estimator_kind,analysis_role,analysis_set,n,n_events,standard_error,"
    "variance_estimator,cluster_count,notes,exposure_level,reference_level,contrast,"
    "is_primary_contrast"
).split(",")


def _kept(headers: list[str]) -> list[str]:
    return [headers[index] for index in agent_pipeline_runs._table_preview_indices(headers)]


def test_the_estimate_preview_names_its_primary_contrast() -> None:
    kept = _kept(ESTIMATES)

    assert len(kept) == 12
    assert {
        "fit_status", "estimate", "ci_low", "ci_high", "effect_scale", "is_primary_contrast",
        "exposure", "exposure_level", "reference_level", "contrast", "analysis_role", "n",
    } == set(kept)


def test_another_wide_table_keeps_its_own_columns() -> None:
    headers = [f"column_{index}" for index in range(20)]
    other = ["row_role", *headers]

    assert _kept(headers) == headers[:12]
    assert _kept(other)[0] == "row_role"
