"""A mortality horizon needs its survivors followed to it.

A fixed horizon (``mort_28d`` with its paired ``followup_days_28d``) is defined
for a survivor only by follow-up to the horizon.  MIMIC-III and AmsterdamUMCdb
record dates of death but no follow-up of survivors, so their horizons were
defined for the dead only: every survivor was missing, and an analysis of
28-day mortality kept the dead alone.  The availability owner nevertheless
reported the horizons there, and the planning menu offered them.

The owner now names the databases that follow their survivors (MIMIC-IV and
SICdb, with their demo copies).  The loader opens no table elsewhere, and it
refuses a horizon for which no stay without a recorded death has follow-up.
Synthetic tables and packaged metadata only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.outcome_availability import (
    FIXED_HORIZON_MORTALITY_ENDPOINTS,
    structural_outcome_unavailability,
)
from easyicu.research_agent.acquisition.catalog import build_database_capability_catalog
from easyicu.research_agent.concept_availability import explain_concept_availability
from easyicu.scores import outcomes

_HORIZON_CONCEPTS = tuple(
    concept
    for endpoint in FIXED_HORIZON_MORTALITY_ENDPOINTS.values()
    for concept in (endpoint.event_concept, endpoint.followup_concept)
)
_UNFOLLOWED = ("mimic", "mimic_demo", "aumc")
_FOLLOWED = ("miiv", "miiv_demo", "sic", "sic_demo")


@pytest.mark.parametrize("database", _UNFOLLOWED)
@pytest.mark.parametrize("concept", _HORIZON_CONCEPTS)
def test_a_horizon_is_unavailable_where_survivors_are_not_followed(
    concept: str, database: str
) -> None:
    assert structural_outcome_unavailability(concept, database) is not None
    cell = explain_concept_availability(concept=concept, database=database)

    assert cell.status == "blocked"
    assert cell.available is False
    assert cell.structural_unavailable is True


@pytest.mark.parametrize("database", _UNFOLLOWED)
def test_the_planning_menu_does_not_offer_a_horizon_there(database: str) -> None:
    offered = {
        item.concept_id for item in build_database_capability_catalog(database).concepts
    }

    assert not set(_HORIZON_CONCEPTS) & offered


@pytest.mark.parametrize("database", _UNFOLLOWED)
def test_the_loader_opens_no_table_there(monkeypatch, database: str) -> None:
    def no_tables(*args, **kwargs):
        pytest.fail("a database without survivor follow-up must not open a table")

    monkeypatch.setattr(outcomes, "_raw_table", no_tables)

    assert outcomes.load_outcomes(database).empty


def _sic_cases(followup_codes=None) -> pd.DataFrame:
    """Two survivors (cases 1 and 3) and a death on day 5 (case 2)."""

    cases = {
        "CaseID": [1, 2, 3],
        "TimeOfStay": [90_000, 90_000, 90_000],
        "ICUOffset": [3_600, 3_600, 3_600],
        "OffsetOfDeath": [None, 3_600 + 5 * 86_400, None],
    }
    if followup_codes is not None:
        cases["EstimatedSurvivalObservationTime"] = followup_codes
    return pd.DataFrame(cases)


def test_a_horizon_without_followup_for_any_survivor_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(outcomes, "_raw_table", lambda *_args: _sic_cases())

    with pytest.raises(ValueError, match="no follow-up"):
        outcomes.load_outcomes("sic")


def test_a_survivor_with_followup_keeps_the_horizon(monkeypatch) -> None:
    monkeypatch.setattr(
        outcomes, "_raw_table", lambda *_args: _sic_cases([3077, None, None])
    )

    result = outcomes.load_outcomes("sic").set_index("CaseID")

    assert bool(result.loc[1, "mort_28d"]) is False
    assert result.loc[1, "followup_days_28d"] == 28.0
    assert bool(result.loc[2, "mort_28d"]) is True
    # A survivor without follow-up stays unknown; the others keep the horizon.
    assert pd.isna(result.loc[3, "mort_28d"])


@pytest.mark.parametrize("database", _FOLLOWED)
def test_databases_that_follow_their_survivors_keep_the_horizons(
    database: str,
) -> None:
    for concept in _HORIZON_CONCEPTS:
        assert structural_outcome_unavailability(concept, database) is None
        cell = explain_concept_availability(concept=concept, database=database)
        assert cell.status == "full", (concept, database, cell.reason)
    offered = {
        item.concept_id for item in build_database_capability_catalog(database).concepts
    }
    assert set(_HORIZON_CONCEPTS) <= offered
