"""Source-bound hospital status for legacy Web research acquisitions."""

from pathlib import Path

from easyicu.research_agent.acquisition.foundation import AcquisitionResult
from easyicu.research_agent.acquisition.hospital_outcome_materialization import (
    materialize_hospital_status_acquisition,
)
from .raw_source_authority import resolve_raw_mimic_iv_source_binding
from .scientific_runtime_projection import WebScientificRuntimeProjectionError


def materialize_web_hospital_status(
    acquisition: AcquisitionResult, *, export_path: Path, database: str
) -> AcquisitionResult:
    if (
        database not in {"miiv", "mimic_iv", "mimic-iv"}
        or "death" not in acquisition.materialized_columns
    ):
        return acquisition
    # Native inputs retain their sealed source authority; this adapter is not
    # permitted to silently extend or replace signed cohort lineage.
    if acquisition.cohort_authority_ref is not None:
        return acquisition
    binding = resolve_raw_mimic_iv_source_binding(
        export_path=export_path, database=database
    )
    if binding is None:
        raise WebScientificRuntimeProjectionError(
            "web_hospital_status_authority_required",
            "Legacy hospital mortality requires verified admission-to-stay status, not event absence.",
            details={"owner": "source_runtime"},
        )
    return materialize_hospital_status_acquisition(
        acquisition,
        status=binding.materialize_hospital_mortality_status(),
        raw_source_receipt=binding.public_receipt(),
    )
