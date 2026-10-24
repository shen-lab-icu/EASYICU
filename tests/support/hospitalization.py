"""Shared synthetic fixtures; no recorded research inputs."""
from __future__ import annotations

from copy import deepcopy
from easyicu.research_agent.acquisition.first_icu_stay import SELECTION_POLICY
from easyicu.research_agent.intake.materialized_metadata import FIRST_ICU_STAY_PER_HOSPITALIZATION_RESTRICTION_SCHEMA as SCHEMA

def receipt():
    return {
        "schema_version": SCHEMA, "coordinate_sha256": "a" * 64,
        "stays_before": 10, "stays_after": 4,
        "non_first_icu_stays_removed": 3, "non_icu_stays_removed": 1,
        "unknown_eligibility_stays_removed": 2,
        "excluded_reason_counts": {
            "non_first_icu_stay": 3, "non_icu_unit": 1,
            "hospitalization_identifier_missing_or_invalid": 2,
        },
        "authority_coordinates": {
            **deepcopy(SELECTION_POLICY), "private_source_path": "/synthetic/private",
        },
    }
