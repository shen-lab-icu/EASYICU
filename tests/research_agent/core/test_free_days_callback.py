"""Synthetic concept/callback coverage across source identities."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.concept.callbacks import ConceptCallbackContext, execute_concept_callback
from easyicu.concept.free_days import FreeDaysDerivationError
from easyicu.table import ICUTable, WinTbl


@pytest.mark.parametrize("database", ["miiv", "mimic", "eicu", "aumc", "hirid"])
@pytest.mark.parametrize("concept", ["mech_vent", "vent_ind", "icu_stay", "vaso_ind"])
def test_callback_uses_verified_intervals_and_followup_across_sources(database, concept):
    support = WinTbl(pd.DataFrame({
        "stay_id": [1, 1, 2], "time": pd.to_timedelta([0, 24, 0], unit="h"),
        "duration": pd.to_timedelta([48, 48, 24], unit="h"), concept: [1, 1, 1],
    }), id_vars=["stay_id"], index_var="time", dur_var="duration")
    followup = ICUTable(pd.DataFrame({
        "stay_id": [1, 2, 3], "death_hours": [np.nan, 500, np.nan],
        "followup_hours": [672, 500, 672], "support_observed": [True, True, True],
    }), id_columns=["stay_id"])
    ctx = ConceptCallbackContext(
        concept_name="free_days", target=None, interval=None, resolver=None,
        data_source=SimpleNamespace(config=SimpleNamespace(name=database)), patient_ids=None,
        kwargs={"horizon_days": 28, "support_concept": concept, "death_score": -1},
    )
    result = execute_concept_callback("free_days", {concept: support, "free_days_followup": followup}, ctx)
    assert result.data["free_days"].tolist() == [25, -1, 28]
    audit = result.data.attrs["free_days_derivation_audit"]
    assert audit["by_death_status"]["died_by_horizon"]["n"] == 1
    with pytest.raises(FreeDaysDerivationError) as absent:
        execute_concept_callback("free_days", {"free_days_followup": followup}, ctx)
    assert absent.value.database == database
    assert absent.value.code == "support_intervals_unavailable"
    followup.data.loc[2, "support_observed"] = False
    with pytest.raises(FreeDaysDerivationError, match="support_intervals_unavailable"):
        execute_concept_callback("free_days", {concept: support, "free_days_followup": followup}, ctx)


def test_combined_organ_support_unions_components_and_requires_each_source():
    def support(concept, start):
        return WinTbl(pd.DataFrame({"stay_id": [1], "time": pd.to_timedelta([start], unit="h"),
                                    "duration": pd.to_timedelta([48], unit="h"), concept: [1]}),
                      id_vars=["stay_id"], index_var="time", dur_var="duration")
    tables = {
        "mech_vent": support("mech_vent", 0), "vaso_ind": support("vaso_ind", 24),
        "free_days_followup": ICUTable(pd.DataFrame({"stay_id": [1], "death_hours": [np.nan],
                                                     "followup_hours": [672], "support_observed": [True]}),
                                      id_columns=["stay_id"]),
    }
    ctx = ConceptCallbackContext(
        concept_name="free_days", target=None, interval=None, resolver=None,
        data_source=SimpleNamespace(config=SimpleNamespace(name="synthetic")), patient_ids=None,
        kwargs={"horizon_days": 28, "support_concept": ["mech_vent", "vaso_ind"], "death_score": 0},
    )
    assert execute_concept_callback("free_days", tables, ctx).data["free_days"].tolist() == [25]
    tables.pop("vaso_ind")
    with pytest.raises(FreeDaysDerivationError, match="support_intervals_unavailable"):
        execute_concept_callback("free_days", tables, ctx)
