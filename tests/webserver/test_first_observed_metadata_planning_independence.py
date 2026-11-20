"""The review's live verifier does not override a first-stay setting at planning.

Every planning run reviews its plan inside ``host_first_observed_review_scope``.
On a metadata-only planning cohort (no rows, no materializer yet) the verifier
has nothing to observe; it must not answer "repeated stays possible" for it,
which made the review ask to retain all stays while the study's first-stay
setting keeps one stay per patient (S1639).  The host first-stay coordinate
then decides.  Outside metadata-only planning a missing materialization still
counts as possible repeats.  Synthetic cohorts only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.planning.dependence_authority import (
    FIRST_ICU_STAY_RESTRICTION_SCHEMA,
    repeat_units_possible,
)
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.webserver.first_observed_run_guard import host_first_observed_review_scope


def _context(tmp_path: Path, **provenance):
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": pd.Series([], dtype="int64"), "death": pd.Series([], dtype="int64")}).to_parquet(cohort)
    built = build_research_context(
        research_question="Is a higher first-day creatinine associated with hospital death?",
        cohort=cohort, cohort_name="synthetic", database="miiv", target_outcome="death",
        id_columns=("stay_id",), outcome_columns=("death",),
    )
    context = getattr(built, "context", built)
    merged = {**(context.cohort.provenance or {}), "analysis_unit": "icu_stay", **provenance}
    return context.model_copy(update={
        "cohort": context.cohort.model_copy(update={"provenance": merged}),
        "cohort_parquet": str(cohort.resolve()),
    })


_RESTRICTION = {
    "first_icu_stay_restriction": {
        "schema_version": FIRST_ICU_STAY_RESTRICTION_SCHEMA,
        "coordinate_sha256": "d" * 64,
    }
}


def test_a_first_stay_setting_at_metadata_planning_rules_out_repeats_in_review(tmp_path) -> None:
    context = _context(tmp_path, evidence_stage="metadata_only_planning", **_RESTRICTION)

    assert repeat_units_possible(context) is False
    with host_first_observed_review_scope(study_context_id="study-synthetic"):
        assert repeat_units_possible(context) is False


def test_metadata_planning_without_the_setting_still_allows_repeats(tmp_path) -> None:
    context = _context(tmp_path, evidence_stage="metadata_only_planning")

    with host_first_observed_review_scope(study_context_id="study-synthetic"):
        assert repeat_units_possible(context) is True


@pytest.mark.parametrize("stage", [None, "materialized"])
def test_outside_metadata_planning_a_missing_materialization_still_counts_as_repeats(
    tmp_path, stage
) -> None:
    extra = {"evidence_stage": stage} if stage else {}
    context = _context(tmp_path, **extra, **_RESTRICTION)

    with host_first_observed_review_scope(study_context_id="study-synthetic"):
        assert repeat_units_possible(context) is True
