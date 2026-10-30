"""Producer-side clock declaration for a freshly created SIC Web export."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from easyicu.databases.time_origin import SIC_CARE_UNIT_ORIGIN


def sic_export_clock(*, database: str, file_bindings: Sequence, time_window_authority: Mapping,
                     data_provenance: Mapping) -> dict | None:
    """Declare the native SIC clock only when current output metadata supports it.

    This is a source-clock declaration, not proof that raw source files were
    consumed, an ICU-only interval, or approval for a scientific substitution.
    Existing exports and non-SIC exports are never rewritten by this helper.
    """
    if str(database).strip().lower() != "sic":
        return None
    from easyicu.concept.provenance import ExportDataProvenance
    from easyicu.webserver.time_origin_source_authority import is_case_duration_provenance_snapshot

    provenance = ExportDataProvenance.model_validate(data_provenance)
    if sum(is_case_duration_provenance_snapshot(snapshot) for snapshot in provenance.snapshots) != 1:
        # Partial case projections cannot establish the reviewed source clock.
        return None
    coordinates = [coordinate for binding in file_bindings for coordinate in binding.time_coordinates]
    if any(binding.identity_column not in {"CaseID", "stay_id"} for binding in file_bindings):
        raise ValueError("sic_export_identity_metadata_mismatch")
    if any(c.origin != SIC_CARE_UNIT_ORIGIN or c.unit != "h" for c in coordinates):
        raise ValueError("sic_export_time_metadata_mismatch")
    if not coordinates:
        return None
    if not any(c.column == "charttime" for c in coordinates):
        raise ValueError("sic_export_time_metadata_missing_charttime")
    if (time_window_authority.get("required") is not True
            or time_window_authority.get("source") != "outcome.los_icu"
            or type(time_window_authority.get("bounded_stays")) is not int
            or time_window_authority["bounded_stays"] <= 0):
        raise ValueError("sic_export_time_window_authority_missing")
    return {
        "time_origin": SIC_CARE_UNIT_ORIGIN,
        "time_unit": "h",
        "clinical_scope_note": (
            "SICdb source clock is first transfer to intensive or intermediate care; "
            "case closure is not a verified ICU discharge."
        ),
    }
