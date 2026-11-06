"""A mortality horizon needs its survivors followed to it.

A fixed horizon (``mort_28d`` with its paired ``followup_days_28d``) is defined
for a survivor only by follow-up to the horizon; a horizon defined for the dead
alone keeps only the patients who died.  Each source states its follow-up:
MIMIC-IV one year after the last discharge, MIMIC-III 90 days (MetaVision) or
four years (CareVue) after the hospital discharge, AmsterdamUMCdb one year from
ICU admission (registry deaths are dated years after discharge), SICdb its
survival observation category.  eICU and HiRID record in-hospital death only:
the loader opens no table there, and it refuses a horizon for which no stay
without a recorded death has follow-up.
Synthetic tables and packaged metadata only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.support.sic_references import sic_raw_tables

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
_UNFOLLOWED = ("eicu", "hirid")
_FOLLOWED = ("miiv", "miiv_demo", "mimic", "mimic_demo", "aumc", "sic", "sic_demo")


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
    monkeypatch.setattr(outcomes, "_raw_table", sic_raw_tables(_sic_cases()))

    with pytest.raises(ValueError, match="no follow-up"):
        outcomes.load_outcomes("sic")


def test_a_survivor_with_followup_keeps_the_horizon(monkeypatch) -> None:
    monkeypatch.setattr(
        outcomes, "_raw_table", sic_raw_tables(_sic_cases([3077, None, None]))
    )

    result = outcomes.load_outcomes("sic").set_index("CaseID")

    assert bool(result.loc[1, "mort_28d"]) is False
    assert result.loc[1, "followup_days_28d"] == 28.0
    assert bool(result.loc[2, "mort_28d"]) is True
    # A survivor with an empty follow-up category and no admission year has
    # an unknown category (SICdb 1.0.8 policy): no horizon is reached.
    assert pd.isna(result.loc[3, "mort_28d"])
    assert bool(result.loc[3, "mort_28d_assumption_based"]) is True
    assert pd.isna(result.loc[3, "mort_365d"])


@pytest.mark.parametrize("origin", [None, float("nan"), float("inf"), -1])
def test_sic_unknown_care_transfer_origin_never_becomes_an_event_or_survivor(monkeypatch, origin) -> None:
    cases = pd.DataFrame({
        "CaseID": [1, 2, 3, 4, 5],
        "ICUOffset": [origin, origin, 3600, 3600, 0],
        "TimeOfStay": [90000] * 5,
        "OffsetOfDeath": [5 * 86400, None, None, float("inf"), 5 * 86400],
        "EstimatedSurvivalObservationTime": [3077] * 5,
    }, index=[11, 12, 13, 14, 15])
    monkeypatch.setattr(outcomes, "_raw_table", sic_raw_tables(cases))
    base = outcomes._sic_stay_death_days("sic", None).set_index("_stay")
    assert base.loc[[1, 2], ["days_to_death", "los_days", "followup_days"]].isna().all().all()
    assert pd.isna(base.loc[4, "days_to_death"])
    assert pd.isna(base.loc[4, "followup_days"])
    result = outcomes.load_outcomes("sic").set_index("CaseID")
    for patient in (1, 2, 4):
        assert pd.isna(result.loc[patient, "mort_28d"])
        assert pd.isna(result.loc[patient, "followup_days_28d"])
    assert not bool(result.loc[3, "mort_28d"])
    assert bool(result.loc[5, "mort_28d"])
    assert result.loc[5, "followup_days_28d"] == 5.0


def test_sic_followup_uses_conservative_admission_clock_bound(monkeypatch) -> None:
    cases = pd.DataFrame({
        "CaseID": [1, 2, 3, 4, 5], "ICUOffset": [0, 3600, 0, 3600, 181 * 86400],
        "TimeOfStay": [90000, 90000, 90000, 90000, 182 * 86400],
        "OffsetOfDeath": [None] * 5,
        "EstimatedSurvivalObservationTime": [3077, 3077, 3076, 3076, 3076],
    })
    monkeypatch.setattr(outcomes, "_raw_table", sic_raw_tables(cases))
    base = outcomes._sic_stay_death_days("sic", None).set_index("_stay")
    assert base.loc[1, "followup_days"] == 365
    assert base.loc[2, "followup_days"] == pytest.approx(365 - 1 / 24)
    assert base.loc[3, "followup_days"] == 180
    assert base.loc[4, "followup_days"] == pytest.approx(180 - 1 / 24)
    assert pd.isna(base.loc[5, "followup_days"])
    result = outcomes.load_outcomes("sic").set_index("CaseID")
    assert bool(result.loc[1, "mort_365d"]) is False
    # An ICU transfer an hour after admission leaves the one-year category an
    # hour short of the horizon; that sub-day shortfall keeps the survivor.
    assert bool(result.loc[2, "mort_365d"]) is False
    assert pd.isna(result.loc[3, "mort_365d"])
    assert result.loc[2, "followup_days_90d"] == 90


def _mimic3_tables(table: str) -> pd.DataFrame:
    """MetaVision survivor (1), CareVue survivor (2), post-discharge death (3)."""

    if table == "icustays":
        return pd.DataFrame({
            "SUBJECT_ID": [10, 20, 30], "HADM_ID": [100, 200, 300],
            "ICUSTAY_ID": [1, 2, 3],
            "INTIME": ["2150-01-01 08:00"] * 3, "OUTTIME": ["2150-01-03 08:00"] * 3,
            "LOS": [2.0] * 3, "DBSOURCE": ["metavision", "carevue", "metavision"],
        })
    if table == "patients":
        return pd.DataFrame({
            "SUBJECT_ID": [10, 20, 30], "DOD": [None, None, "2150-02-10"],
        })
    if table == "admissions":
        return pd.DataFrame({
            "HADM_ID": [100, 200, 300], "DEATHTIME": [None, None, None],
            "DISCHTIME": ["2150-01-11 08:00"] * 3,
        })
    raise AssertionError(table)


def test_mimic3_survivors_are_followed_to_their_system_bound(monkeypatch) -> None:
    monkeypatch.setattr(outcomes, "_raw_table", lambda _db, _path, table: _mimic3_tables(table))

    result = outcomes.load_outcomes("mimic").set_index("icustay_id")

    # MetaVision: the 4 x 365-day follow-up bound covers both systems
    # (extraction audit round 4, R4-018, reversing R3-08).
    assert bool(result.loc[1, "mort_90d"]) is False
    assert bool(result.loc[1, "mort_365d"]) is False
    # CareVue: four years after discharge.
    assert bool(result.loc[2, "mort_365d"]) is False
    # A death after discharge counts: day 40 is past 28 and within 90.
    assert bool(result.loc[3, "mort_28d"]) is False
    assert bool(result.loc[3, "mort_90d"]) is True


def test_aumc_times_an_icu_death_at_discharge_and_follows_survivors(monkeypatch) -> None:
    day = 86_400_000
    admissions = pd.DataFrame({
        "admissionid": [1, 2, 3],
        "admittedat": [0, 0, 0],
        "dischargedat": [3 * day, 3 * day, 3 * day],
        "destination": ["Overleden", "Overleden", "Huis"],
        # An ICU death with a registry date before ICU discharge, one without
        # any date, and a survivor.
        "dateofdeath": [2 * day, None, None],
    })
    monkeypatch.setattr(outcomes, "_raw_table", lambda *_: admissions)

    result = outcomes.load_outcomes("aumc").set_index("admissionid")

    assert result.loc[1, "followup_days_28d"] == 3.0
    assert bool(result.loc[2, "mort_28d"]) is True
    assert result.loc[2, "followup_days_28d"] == 3.0
    assert bool(result.loc[3, "mort_365d"]) is False


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
