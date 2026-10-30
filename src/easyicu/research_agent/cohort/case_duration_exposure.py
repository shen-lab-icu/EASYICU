"""Verify the source-bound, half-open exposure window for SIC case analysis."""
from pathlib import Path

from easyicu.research_agent.intake.materialized_metadata import implementation_bundle_sha256


def require_case_duration_exposure_materialization(verified, authority, *, source_concept=None):
    from easyicu.research_agent.cohort import materializer

    owner = Path(materializer.__file__).resolve()
    expected = implementation_bundle_sha256((
        owner, owner.with_name("primitives.py"), owner.parents[1] / "intake" / "materialized_metadata.py",
        owner.parents[2] / "concept" / "metadata_projection.py",
    ))
    if (verified.authority.producer != "cohort_materializer"
            or verified.authority.producer_implementation_sha256 != expected):
        raise ValueError("case_duration_exposure_implementation_unverified")
    matches = [file.columns[authority.exposure_column] for file in verified.sidecar.files
               if authority.exposure_column in file.columns]
    if source_concept is not None and (
        len(matches) != 1 or matches[0].metadata.source_concept != source_concept
    ):
        raise ValueError("case_duration_exposure_source_concept_mismatch")
    window = matches[0].derivation_window if len(matches) == 1 else None
    if (window is None or window.origin != authority.selected_origin
            or window.start_hours != authority.exposure_window_start_hours
            or window.end_hours != authority.exposure_window_end_hours):
        raise ValueError("case_duration_exposure_window_unverified")
