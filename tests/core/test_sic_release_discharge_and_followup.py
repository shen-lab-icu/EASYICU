"""SICdb 1.0.6 and 1.0.8: discharge types by label, follow-up by policy."""

import numpy as np
import pandas as pd
import pytest

from easyicu.scores import outcomes
from easyicu.source_death_times import (
    SIC_REFERENCE_UNAVAILABLE,
    SicReferenceUnavailable,
    sic_discharge_codes,
    sic_recorded_death,
)
from tests.support.sic_references import sic_raw_tables, sic_references


def test_each_release_names_its_own_death_and_survival_codes():
    old = sic_discharge_codes(sic_references("1.0.6"))
    assert old.deceased == {2028} and old.survived == {2026}
    new = sic_discharge_codes(sic_references("1.0.8"))
    assert new.deceased == {2028, 3130}
    # Transfer, discharge against advice and the internal transfers are alive.
    assert new.survived == {2026, 3129, 3131, 3132, 3133, 3134}
    assert new.icu_deceased == {2215}
    assert new.followup_days == {3076: 180.0, 3077: 365.0}


def test_recorded_death_uses_the_release_codes():
    codes = sic_discharge_codes(sic_references("1.0.8"))
    hospital = pd.Series([3130, 3131, 3129, None, 2028])
    state = pd.Series([2202, 2202, 2215, 2215, 2202])
    assert sic_recorded_death(hospital, state, codes).tolist() == [True, False, True, True, True]


@pytest.mark.parametrize("references", [
    pd.DataFrame({"ReferenceGlobalID": [1], "ReferenceValue": ["x"]}),
    sic_references().loc[lambda f: f.ReferenceName != "DischargeType"],
    sic_references().loc[lambda f: f.ReferenceValue != "Deceased"],
    sic_references().loc[lambda f: f.ReferenceValue != "verstorben"],
])
def test_references_that_cannot_name_the_codes_fail_closed(references):
    with pytest.raises(SicReferenceUnavailable, match=SIC_REFERENCE_UNAVAILABLE):
        sic_discharge_codes(references)


def _cases_108() -> pd.DataFrame:
    """SICdb 1.0.8: ESOT is recorded for deaths only."""
    return pd.DataFrame({
        "CaseID": [1, 2, 3, 4, 5, 6],
        "ICUOffset": [0] * 6,
        "TimeOfStay": [86_400] * 6,
        "OffsetOfDeath": [None, None, None, None, 10 * 86_400, 200 * 86_400],
        "HospitalDischargeType": [3129, 3131, 3129, 3129, 3130, 3129],
        "DischargeState": [2202] * 6,
        "AdmissionYear": [2015, 2021, 2030, None, 2021, 2016],
        "EstimatedSurvivalObservationTime": [None, None, None, None, 3076, 3077],
    })


def test_1_0_8_followup_is_rebuilt_from_the_admission_year(monkeypatch):
    monkeypatch.setattr(outcomes, "_raw_table", sic_raw_tables(_cases_108(), "1.0.8"))
    base = outcomes._sic_stay_death_days("sic", None).set_index("_stay")
    assert base.loc[[1, 2, 3, 4], "followup_days"].tolist() == [365.0, 180.0, 180.0, 180.0]
    assert base.loc[[1, 2, 3, 4], "followup_assumed"].tolist() == [True] * 4
    assert base.loc[[5, 6], "followup_assumed"].tolist() == [False, False]

    result = outcomes.load_outcomes("sic").set_index("CaseID")
    # One-year category from 2013-2020: every horizon is reached, on the rebuilt category.
    assert bool(result.loc[1, "mort_365d"]) is False
    assert bool(result.loc[1, "mort_365d_assumption_based"]) is True
    # 2021, another year, or no year: 28 and 90 days are reached, one year is not.
    for case in (2, 3, 4):
        assert bool(result.loc[case, "mort_90d"]) is False
        assert pd.isna(result.loc[case, "mort_365d"])
    assert bool(result.loc[5, "mort_28d"]) is True
    assert bool(result.loc[5, "mort_28d_assumption_based"]) is False
    assert bool(result.loc[6, "mort_365d"]) is True


def test_the_rebuilt_category_reproduces_every_recorded_one():
    """Every recorded 1.0.6 / 1.0.8 ESOT follows the admission-year rule."""
    years = pd.Series([2013, 2016, 2020, 2021, 2021])
    recorded = pd.Series([365.0, 365.0, 365.0, 180.0, 180.0])
    rebuilt = outcomes.sic_followup_from_admission_year(years)
    assert rebuilt.tolist() == recorded.tolist()


def test_the_six_month_policy_gives_every_empty_category_the_minimum(monkeypatch):
    monkeypatch.setattr(outcomes, "_raw_table", sic_raw_tables(_cases_108(), "1.0.8"))
    base = outcomes._sic_stay_death_days("sic", None, followup_policy="six_month_minimum")
    assert base.set_index("_stay").loc[[1, 2, 3, 4], "followup_days"].tolist() == [180.0] * 4
    with pytest.raises(ValueError, match="follow-up policy"):
        outcomes.sic_followup_from_admission_year(pd.Series([2015]), policy="one_year")


def test_1_0_6_followup_keeps_its_recorded_category(monkeypatch):
    cases = _cases_108().assign(
        HospitalDischargeType=[2026, 2026, 2026, 2026, 2028, 2026],
        EstimatedSurvivalObservationTime=[3077, 3076, 3077, 3076, 3076, 3077],
    )
    monkeypatch.setattr(outcomes, "_raw_table", sic_raw_tables(cases, "1.0.6"))
    base = outcomes._sic_stay_death_days("sic", None).set_index("_stay")
    assert base["followup_days"].tolist() == [365.0, 180.0, 365.0, 180.0, 180.0, 365.0]
    assert not base["followup_assumed"].any()
    assert np.isfinite(base.loc[5, "days_to_death"])
