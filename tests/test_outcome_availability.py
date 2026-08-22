from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from easyicu.outcome_availability import (
    OUTCOME_CONCEPT_SUPPORTED_DATABASES,
    structural_outcome_unavailability,
)


def test_database_specific_outcome_support_is_directional() -> None:
    mimic_vfd = structural_outcome_unavailability("vent_free_days_28", "miiv")
    eicu_vfd = structural_outcome_unavailability("vent_free_days_28", "eicu_demo")
    eicu_mortality = structural_outcome_unavailability("mort_28d", "eicu_demo")
    mimic_mortality = structural_outcome_unavailability("mort_28d", "miiv")
    mimic_followup = structural_outcome_unavailability("followup_days_28d", "miiv")
    eicu_followup = structural_outcome_unavailability("followup_days_28d", "eicu")
    mimic_icu_free = structural_outcome_unavailability("icu_free_days_28", "miiv")
    mimic_readmission = structural_outcome_unavailability("icu_readmission", "mimic")

    assert mimic_vfd is not None
    assert mimic_vfd.reason_code == "outcome_concept_structurally_unavailable"
    assert mimic_vfd.supported_databases == ()
    assert eicu_vfd is not None
    assert eicu_vfd.supported_databases == ()
    assert eicu_mortality is not None
    assert mimic_mortality is None
    assert mimic_followup is None
    assert eicu_followup is not None
    assert mimic_icu_free is not None
    assert mimic_icu_free.supported_databases == ()
    assert mimic_readmission is not None
    assert mimic_readmission.supported_databases == ()


def test_non_outcome_concepts_are_not_reclassified() -> None:
    assert structural_outcome_unavailability("hr", "miiv") is None


def test_outcome_availability_contract_is_frozen() -> None:
    receipt = structural_outcome_unavailability("vent_free_days_28", "miiv")
    assert receipt is not None
    with pytest.raises(FrozenInstanceError):
        receipt.database = "eicu"  # type: ignore[misc]
    with pytest.raises(TypeError):
        OUTCOME_CONCEPT_SUPPORTED_DATABASES["hr"] = frozenset({"miiv"})  # type: ignore[index]
