"""Attach verified hospital follow-up to a legacy one-stay cohort.

This transformation preserves its parent artifacts and explicitly accounts for
every rejected follow-up row. It never substitutes ICU length of stay.
"""

from __future__ import annotations

from dataclasses import replace
import io
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..authority.filesystem import AnchoredDirectory
from ..canonical_json import canonical_json_bytes, sha256_bytes, sha256_file
from .foundation import AcquisitionResult
from .hospital_mortality_followup import HospitalMortalityFollowup


def materialize_hospital_followup_acquisition(
    acquisition: AcquisitionResult,
    *,
    followup: HospitalMortalityFollowup,
    raw_source_receipt: Mapping[str, Any],
) -> AcquisitionResult:
    if acquisition.cohort_authority_ref is not None:
        raise ValueError("hospital_followup_native_lineage_extension_required")
    if acquisition.universe_path is None or acquisition.provenance_path is None:
        raise ValueError("hospital_followup_source_cohort_required")
    path = Path(acquisition.universe_path)
    with AnchoredDirectory.open(path.parent) as directory:
        payload = directory.read_bytes(path.name, max_bytes=128 * 1024 * 1024)
    frame = pd.read_parquet(io.BytesIO(payload))
    provenance = json.loads(Path(acquisition.provenance_path).read_text())
    if "stay_id" in frame:
        source_ids = frame["stay_id"]
    elif (provenance.get("replacement_row_identity") or {}).get(
        "output_identity_column"
    ) == "patient_stay_id":
        parsed = (
            frame["patient_stay_id"]
            .astype("string")
            .str.extract(r"^p[0-9]+:s([0-9]+)$", expand=False)
        )
        if parsed.isna().any():
            raise ValueError("hospital_followup_source_identity_invalid")
        source_ids = pd.to_numeric(parsed, errors="raise").astype("int64")
    else:
        raise ValueError("hospital_followup_source_identity_unbound")
    if source_ids.isna().any() or source_ids.duplicated().any():
        raise ValueError("hospital_followup_source_identity_invalid")
    declared = set(followup.frame["stay_id"]) | set(followup.exclusions["stay_id"])
    if not set(source_ids).issubset(declared):
        raise ValueError("hospital_followup_source_coverage_incomplete")
    known = source_ids.isin(followup.frame["stay_id"])
    selected = frame.loc[known].copy()
    joined = followup.frame.set_index("stay_id").loc[source_ids.loc[known]]
    selected["death"] = joined["hospital_death"].to_numpy()
    for column in ("death_time_hours", "hospital_followup_time_hours"):
        selected[column] = joined[column].to_numpy()
    receipt = {
        "schema_version": "easyicu.hospital_followup_materialization/1",
        "source_cohort_sha256": sha256_bytes(payload),
        "raw_source": dict(raw_source_receipt),
        "followup": dict(followup.receipt),
        "source_stays": len(frame),
        "analysis_stays": len(selected),
        "excluded_stays": int((~known).sum()),
        "event_time_column": "death_time_hours",
        "observation_duration_column": "hospital_followup_time_hours",
        "unit": "hours",
        "source_metadata_kind": "legacy_untyped",
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    output = path.parent / "hospital_followup_cohort.parquet"
    output_provenance = path.parent / "hospital_followup_cohort.provenance.json"
    if output.exists() or output_provenance.exists():
        raise ValueError("hospital_followup_artifact_exists")
    selected.to_parquet(output, index=False)
    provenance.update(
        {
            "n_rows": len(selected),
            "n_patients": len(selected),
            "columns": list(selected),
            "hospital_followup_materialization": receipt,
        }
    )
    output_provenance.write_bytes(canonical_json_bytes(provenance))
    return replace(
        acquisition,
        universe_path=output,
        provenance_path=output_provenance,
        materialized_columns=tuple(selected.columns),
    )


__all__ = ["materialize_hospital_followup_acquisition"]
