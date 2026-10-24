"""Path-free population counts from the verified preparation owner."""

from collections.abc import Mapping
from pathlib import Path
import re

from easyicu.research_agent.intake.materialized_metadata import (
    load_verified_materialized_cohort_authority,
)

_REASONS = frozenset({
    "hospitalization_identifier_missing_or_invalid",
    "hospitalization_prior_unit_coverage_incomplete",
    "hospitalization_unit_type_missing_or_unknown",
    "hospitalization_unit_ordinal_duplicate",
    "hospitalization_unit_ordinal_missing_or_invalid",
    "non_first_icu_stay", "non_icu_unit",
})


def prepared_population_summary(acquisition):
    """Expose counts only when the sealed cohort and restriction reconcile."""
    path = getattr(acquisition, "universe_path", None)
    if path is None:
        return None
    try:
        verified = load_verified_materialized_cohort_authority(Path(path))
        if verified is None:
            return None
        # This restriction is sealed as a producer parameter. Older selectors
        # also carry an unsealed convenience copy; that copy is not authority.
        restriction = verified.authority.producer_parameters.get(
            "first_icu_stay_per_hospitalization_restriction"
        )
        if not isinstance(restriction, Mapping) or restriction.get("schema_version") != (
            "easyicu.first_icu_stay_per_hospitalization_restriction/1"
        ):
            return None
        before, after = restriction.get("stays_before"), restriction.get("stays_after")
        reasons = restriction.get("excluded_reason_counts")
        coordinate = restriction.get("coordinate_sha256")
        authority = restriction.get("authority_coordinates")
        if (
            not isinstance(reasons, Mapping) or set(reasons) - _REASONS
            or any(type(n) is not int or not 0 <= n <= 2**53 - 1
                   for n in (before, after, *reasons.values()))
            or before - after != sum(reasons.values())
            or after != verified.authority.cohort_rows
            or not isinstance(coordinate, str) or not re.fullmatch(r"[a-f0-9]{64}", coordinate)
            or not isinstance(authority, Mapping)
            or authority.get("coordinate_sha256") != coordinate
            or authority.get("establishes_patient_independence") is not False
        ):
            return None
        return {
            "schema_version": "easyicu.prepared_population_summary/1",
            "analysis_unit": "icu_stay",
            "selection": "first_icu_stay_per_hospitalization",
            "stays_before": before, "stays_after": after,
            "excluded_reason_counts": dict(reasons),
            "materialized_authority_sha256": verified.reference.sha256,
            "coordinate_sha256": coordinate,
            "establishes_patient_independence": False,
        }
    except (OSError, ValueError, TypeError):
        return None
