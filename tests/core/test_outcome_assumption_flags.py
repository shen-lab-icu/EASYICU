"""Fixed-horizon mortality that rests on an undocumented follow-up assumption is flagged.

Round-12 ruling. AmsterdamUMCdb documents ``dateofdeath`` "if applicable
(non-null)" and no follow-up horizon, so an AUMC survivor known only by the
assumed one-year follow-up is assumption-based. SICdb documents its
EstimatedSurvivalObservationTime categories ("6 Months", "1 Year"), so those
survivors are not; only a survivor that reaches a horizon through the one-day
grace is. Values do not change. Synthetic tables only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.concept_output_sources import COMPOSITE_CONCEPT_OUTPUT_SOURCES
from easyicu.outcome_availability import (
    FOLLOWUP_OUTCOME_DATABASES,
    OUTCOME_CONCEPT_SUPPORTED_DATABASES,
)
from easyicu.scores import outcomes
from tests.support.sic_references import sic_raw_tables

FLAGS = ("mort_28d_assumption_based", "mort_90d_assumption_based", "mort_365d_assumption_based")
DAY_MS = 86_400_000


def test_sic_flags_only_horizons_reached_through_the_grace_day(monkeypatch) -> None:
    cases = pd.DataFrame({
        "CaseID": [1, 2, 3, 4],
        "ICUOffset": [0, 3_600, 3_600, 3_600],
        "TimeOfStay": [90_000] * 4,
        "OffsetOfDeath": [None, None, 3_600 + 5 * 86_400, None],
        "EstimatedSurvivalObservationTime": [3077, 3077, 3077, None],
    })
    monkeypatch.setattr(outcomes, "_raw_table", sic_raw_tables(cases))
    result = outcomes.load_outcomes("sic").set_index("CaseID")

    # Case 1 is followed the full documented year from its ICU origin.
    assert not result.loc[1, list(FLAGS)].any()
    # Case 2's ICU transfer is 1 h after admission: the year is 365 d - 1 h,
    # so only the one-day grace makes it a one-year survivor.
    assert bool(result.loc[2, "mort_365d"]) is False
    assert result.loc[2, list(FLAGS)].tolist() == [False, False, True]
    # A recorded death is documented.
    assert not result.loc[3, list(FLAGS)].any()
    # An empty category (SICdb 1.0.8 survivors) without an admission year
    # keeps the six-month minimum: 28 and 90 days rest on that assumption,
    # one year is unknown and carries no flag.
    assert result.loc[4, list(FLAGS[:2])].tolist() == [True, True]
    assert pd.isna(result.loc[4, FLAGS[2]])


def test_aumc_survivors_known_only_by_the_assumed_year_are_flagged(monkeypatch) -> None:
    admissions = pd.DataFrame({
        "admissionid": [1, 2, 3],
        "admittedat": [0, 0, 0],
        "dischargedat": [2 * DAY_MS, 2 * DAY_MS, 2 * DAY_MS],
        "dateofdeath": [None, 40 * DAY_MS, 400 * DAY_MS],
        "destination": ["Huis", "Huis", "Huis"],
    })
    monkeypatch.setattr(outcomes, "_raw_table", lambda *_: admissions)
    result = outcomes.load_outcomes("aumc").set_index("admissionid")

    assert result.loc[1, ["mort_28d", "mort_90d", "mort_365d"]].tolist() == [False, False, False]
    assert result.loc[1, list(FLAGS)].tolist() == [True, True, True]
    # Deaths on day 40 and day 400 are recorded, so no horizon is assumed.
    assert result.loc[2, ["mort_28d", "mort_90d"]].tolist() == [False, True]
    assert not result.loc[2, list(FLAGS)].any()
    assert result.loc[3, "mort_365d"] == False  # noqa: E712
    assert not result.loc[3, list(FLAGS)].any()


@pytest.mark.parametrize("flag", FLAGS)
def test_flags_are_outcome_loader_outputs_where_follow_up_exists(flag) -> None:
    assert COMPOSITE_CONCEPT_OUTPUT_SOURCES[flag] == "outcomes_loader"
    assert OUTCOME_CONCEPT_SUPPORTED_DATABASES[flag] == FOLLOWUP_OUTCOME_DATABASES
    from easyicu.api.extraction import EXTRACT_MODULES

    assert flag in EXTRACT_MODULES["outcome"]
