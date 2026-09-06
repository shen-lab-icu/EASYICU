"""Attach verified hospital status or follow-up to a legacy one-stay cohort.

This transformation preserves its parent artifacts and explicitly accounts for
every rejected follow-up row. It never substitutes ICU length of stay.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..authority.filesystem import AnchoredDirectory
from ..canonical_json import canonical_json_bytes, sha256_bytes, sha256_file
from ..contracts.dependence import (
    PlannedDependenceRequirement,
    PatientGroupResolutionError,
    resolve_patient_groups,
)
from .foundation import AcquisitionResult
from .hospital_mortality_followup import HospitalMortalityFollowup
from ...hospital_mortality import HospitalMortalityStatus


def _cohort_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(
            frame.reset_index(drop=True), index=False
        ).values.tobytes()
    ).hexdigest()


def _verified_patient_count(
    frame: pd.DataFrame,
    *,
    replacement_row_identity: object,
) -> int | None:
    if not isinstance(replacement_row_identity, Mapping):
        return None
    source = replacement_row_identity.get("output_identity_column")
    derivation = replacement_row_identity.get("patient_group_derivation")
    if (
        not isinstance(source, str)
        or source not in frame.columns
        or derivation != {"algorithm": "prefix_before_:s", "delimiter": ":s"}
    ):
        return None
    try:
        groups = resolve_patient_groups(
            frame[source].tolist(),
            requirement=PlannedDependenceRequirement(
                group_source=source,
                group_derivation="prefix_before_delimiter",
                delimiter=":s",
            ),
        )
    except PatientGroupResolutionError as exc:
        raise ValueError("hospital_followup_patient_grouping_invalid") from exc
    return groups.cluster_count


def _source_cohort(acquisition: AcquisitionResult):
    if acquisition.cohort_authority_ref is not None:
        raise ValueError("hospital_followup_native_lineage_extension_required")
    if acquisition.universe_path is None or acquisition.provenance_path is None:
        raise ValueError("hospital_followup_source_cohort_required")
    path = Path(acquisition.universe_path)
    with AnchoredDirectory.open(path.parent) as directory:
        payload = directory.read_bytes(path.name, max_bytes=128 * 1024 * 1024)
    frame = pd.read_parquet(io.BytesIO(payload))
    provenance_path = Path(acquisition.provenance_path)
    with AnchoredDirectory.open(provenance_path.parent) as directory:
        provenance = json.loads(
            directory.read_bytes(provenance_path.name, max_bytes=4 * 1024 * 1024)
        )
    if (
        provenance.get("schema_version") != "easyicu.cohort_materializer/1"
        or provenance.get("cohort_file_sha256") != sha256_bytes(payload)
        or provenance.get("cohort_file_size") != len(payload)
        or provenance.get("columns") != list(frame)
        or provenance.get("n_stays_after_inclusion_exclusion") != len(frame)
    ):
        raise ValueError("hospital_outcome_source_receipt_mismatch")
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
    return path, frame, provenance, source_ids, payload


def materialize_hospital_followup_acquisition(
    acquisition: AcquisitionResult,
    *,
    followup: HospitalMortalityFollowup,
    raw_source_receipt: Mapping[str, Any],
) -> AcquisitionResult:
    path, frame, provenance, source_ids, payload = _source_cohort(acquisition)
    declared = set(followup.frame["stay_id"]) | set(followup.exclusions["stay_id"])
    if not set(source_ids).issubset(declared):
        raise ValueError("hospital_followup_source_coverage_incomplete")
    known = source_ids.isin(followup.frame["stay_id"])
    selected = frame.loc[known].drop(columns=["death_time"], errors="ignore").copy()
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
        "invalidated_parent_columns": [c for c in ("death_time",) if c in frame],
    }
    return _publish_derivation(
        acquisition,
        path=path,
        selected=selected,
        provenance=provenance,
        stem="hospital_followup_cohort",
        receipt_key="hospital_followup_materialization",
        receipt=receipt,
    )


def materialize_hospital_status_acquisition(
    acquisition: AcquisitionResult,
    *,
    status: HospitalMortalityStatus,
    raw_source_receipt: Mapping[str, Any],
) -> AcquisitionResult:
    """Replace an unverified status without applying clock-based exclusions."""
    path, frame, provenance, source_ids, payload = _source_cohort(acquisition)
    if tuple(status.frame.columns) != ("stay_id", "hospital_death"):
        raise ValueError("hospital_status_output_schema_invalid")
    if status.frame.stay_id.isna().any() or status.frame.stay_id.duplicated().any():
        raise ValueError("hospital_status_output_identity_invalid")
    if not set(source_ids).issubset(set(status.frame.stay_id)):
        raise ValueError("hospital_status_source_coverage_incomplete")
    # A parent's old event-time alias was derived from the rejected status
    # mapping. It must not survive as apparently corroborating evidence.
    invalidated = [
        c
        for c in ("death_time", "death_time_hours", "hospital_followup_time_hours")
        if c in frame
    ]
    selected = frame.drop(columns=invalidated).copy()
    selected["death"] = pd.array(
        status.frame.set_index("stay_id").loc[source_ids, "hospital_death"],
        dtype="boolean",
    )
    receipt = {
        "schema_version": "easyicu.hospital_status_materialization/1",
        "source_cohort_sha256": sha256_bytes(payload),
        "raw_source": dict(raw_source_receipt),
        "status": dict(status.receipt),
        "source_stays": len(frame),
        "analysis_stays": len(selected),
        "excluded_stays": 0,
        "unknown_status_stays": int(selected.death.isna().sum()),
        "invalidated_parent_columns": invalidated,
        "source_metadata_kind": "legacy_untyped",
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    return _publish_derivation(
        acquisition,
        path=path,
        selected=selected,
        provenance=provenance,
        stem="hospital_status_cohort",
        receipt_key="hospital_status_materialization",
        receipt=receipt,
    )


def _publish_derivation(
    acquisition, *, path, selected, provenance, stem, receipt_key, receipt
):
    output = path.parent / f"{stem}.parquet"
    # The ResearchContext intake owner discovers legacy materialization receipts
    # through the canonical ``<stem>_provenance.json`` sibling selector.  Keep
    # this derived cohort inside that contract instead of writing a visually
    # similar sidecar that downstream planning cannot see.
    output_provenance = path.parent / f"{stem}_provenance.json"
    if output.exists() or output_provenance.exists():
        raise ValueError("hospital_followup_artifact_exists")
    selected.to_parquet(output, index=False)
    replacement_row_identity = provenance.get("replacement_row_identity")
    if isinstance(replacement_row_identity, Mapping):
        replacement_row_identity = dict(replacement_row_identity)
        replacement_row_identity["mapped_cohort_rows"] = len(selected)
    provenance.update(
        {
            "n_rows": len(selected),
            "n_patients": _verified_patient_count(
                selected,
                replacement_row_identity=replacement_row_identity,
            ),
            "n_stays_after_inclusion_exclusion": len(selected),
            "columns": list(selected),
            "cohort_sha256": _cohort_sha256(selected),
            "cohort_file_sha256": sha256_file(output),
            "cohort_file_size": output.stat().st_size,
            **(
                {"replacement_row_identity": replacement_row_identity}
                if replacement_row_identity is not None
                else {}
            ),
            receipt_key: receipt,
        }
    )
    output_provenance.write_bytes(canonical_json_bytes(provenance))
    return replace(
        acquisition,
        universe_path=output,
        provenance_path=output_provenance,
        materialized_columns=tuple(selected.columns),
    )


__all__ = [
    "materialize_hospital_followup_acquisition",
    "materialize_hospital_status_acquisition",
]
