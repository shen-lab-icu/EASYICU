"""Attach verified hospital status or follow-up to a one-stay cohort.

Legacy (untyped) cohorts are rewritten here with a receipt in the sibling
provenance selector.  Typed cohorts that carry a sealed authority are extended
through the typed-lineage owner instead, which publishes a parent-bound child
authority; this module never downgrades or forges that lineage.  Both paths
preserve their parent artifacts, account for every rejected follow-up row and
never substitute ICU length of stay.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

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


def _typed_parent_stay_axis(path: Path, identity: str) -> pd.Series | None:
    """Map the deriver's stay ids onto a patient-grouped parent's identities.

    A patient-grouped cohort keys its rows by ``p<patient>:s<stay>`` (the same
    reading the legacy path applies above), so renaming ``stay_id`` alone would
    leave every follow-up row unmatched.  ``None`` means the parent is keyed by
    the stay itself and the rename is the whole mapping.
    """

    if identity != "patient_stay_id":
        return None
    with AnchoredDirectory.open(path.parent) as directory:
        payload = directory.read_bytes(path.name, max_bytes=128 * 1024 * 1024)
    values = pd.read_parquet(io.BytesIO(payload), columns=[identity])[
        identity
    ].astype("string")
    parsed = values.str.extract(r"^p[0-9]+:s([0-9]+)$", expand=False)
    if parsed.isna().any():
        raise ValueError("hospital_followup_source_identity_invalid")
    stays = pd.to_numeric(parsed, errors="raise").astype("int64")
    if stays.duplicated().any():
        raise ValueError("hospital_followup_source_identity_invalid")
    return pd.Series(values.to_numpy(dtype=object), index=stays.to_numpy())


def _onto_parent_identity(
    frame: pd.DataFrame, *, identity: str, stay_axis: pd.Series | None
) -> pd.DataFrame:
    mapped = frame.rename(columns={"stay_id": identity})
    if stay_axis is None:
        return mapped
    # Rows for stays outside this cohort have no grouped identity to carry;
    # the owner still checks that every parent row is covered.
    stays = pd.to_numeric(mapped[identity], errors="raise").astype("int64")
    kept = stays.isin(stay_axis.index)
    mapped = mapped.loc[kept].copy()
    mapped[identity] = stays[kept].map(stay_axis).to_numpy(dtype=object)
    return mapped


def _materialize_typed_hospital_followup(
    acquisition: AcquisitionResult,
    *,
    followup: HospitalMortalityFollowup,
    raw_source_receipt: Mapping[str, Any],
) -> AcquisitionResult:
    """Extend a typed (sealed) cohort through the typed-lineage owner.

    The cohort authority owner publishes the child and re-verifies it against
    the parent parquet; this adapter only maps the deriver's ``stay_id`` axis
    onto the sealed identity column and forwards the two receipts.  A sealed
    trajectory is bound to the universe it was cut from, which is no longer
    the analysis universe, so it follows the child (``_followup_trajectory``).
    """
    from ..intake.materialized_metadata import (
        load_verified_materialized_cohort_authority,
        publish_hospital_followup_materialized_cohort,
    )
    from ..intake.materialized_trajectory import (
        load_verified_materialized_trajectory_authority,
        materialized_trajectory_provenance_path,
    )

    if acquisition.universe_path is None or acquisition.cohort_authority_ref is None:
        raise ValueError("hospital_followup_source_cohort_required")
    path = Path(acquisition.universe_path)
    verified = load_verified_materialized_cohort_authority(
        path, expected_authority=acquisition.cohort_authority_ref
    )
    if verified is None:
        raise ValueError("hospital_followup_source_cohort_required")
    # Verified before anything is written: an unsealed trajectory cannot be
    # bound to the child, and a sealed one must describe this parent.
    source_trajectory = None
    if acquisition.trajectory_path is not None:
        if acquisition.trajectory_authority_ref is None:
            raise ValueError("hospital_followup_trajectory_authority_required")
        source_trajectory = load_verified_materialized_trajectory_authority(
            Path(acquisition.trajectory_path),
            expected_authority=acquisition.trajectory_authority_ref,
            expected_universe_authority=acquisition.cohort_authority_ref,
        )
    identity = verified.authority.identity_column
    stay_axis = _typed_parent_stay_axis(path, identity)
    followup_frame = _onto_parent_identity(
        followup.frame, identity=identity, stay_axis=stay_axis
    )[[identity, "hospital_death", "death_time_hours", "hospital_followup_time_hours"]]
    exclusions = _onto_parent_identity(
        followup.exclusions.rename(columns={"reason_code": "reason"}),
        identity=identity,
        stay_axis=stay_axis,
    )[[identity, "reason"]]
    target = path.parent / "hospital_followup_cohort.parquet"
    trajectory_target = path.parent / "hospital_followup_cohort_trajectory.parquet"
    if (
        target.exists()
        or (path.parent / "hospital_followup_cohort_provenance.json").exists()
        or trajectory_target.exists()
        or materialized_trajectory_provenance_path(trajectory_target).exists()
    ):
        raise ValueError("hospital_followup_artifact_exists")
    published = publish_hospital_followup_materialized_cohort(
        path,
        target,
        followup=followup_frame,
        exclusions=exclusions,
        followup_receipt=dict(followup.receipt),
        raw_source_receipt=dict(raw_source_receipt),
        producer_implementation_sha256=sha256_file(Path(__file__)),
        producer_parameters={"adapter": "hospital_outcome_materialization"},
        expected_parent_authority=acquisition.cohort_authority_ref,
    )
    if published is None:  # pragma: no cover - typed parent verified above
        raise ValueError("hospital_followup_source_cohort_required")
    extended = replace(
        acquisition,
        universe_path=target,
        provenance_path=path.parent / "hospital_followup_cohort_provenance.json",
        cohort_authority_path=path.parent / published.reference.file,
        cohort_authority_ref=published.reference,
        materialized_columns=tuple(published.authority.cohort_columns),
    )
    if source_trajectory is None:
        return extended
    child_trajectory = _followup_trajectory(
        source_trajectory,
        source_path=Path(acquisition.trajectory_path),
        child_path=target,
        child=published,
        target=trajectory_target,
    )
    return replace(
        extended,
        trajectory_path=trajectory_target,
        trajectory_provenance_path=materialized_trajectory_provenance_path(
            trajectory_target
        ),
        trajectory_authority_path=trajectory_target.parent
        / child_trajectory.reference.file,
        trajectory_authority_ref=child_trajectory.reference,
    )


def _followup_trajectory(
    source: Any,
    *,
    source_path: Path,
    child_path: Path,
    child: Any,
    target: Path,
) -> Any:
    """Republish the retained stays' trajectory rows bound to the follow-up child.

    Only the trajectory owner publishes a trajectory, and only onto the exact
    universe it describes.  Retained rows keep their source order; a concept
    left without rows is available but unobserved, as the materializer records
    it.  The parent's source receipts are carried unchanged and the
    restriction is recorded next to them.
    """
    from ..intake import materialized_trajectory as trajectory_owner
    from ..intake.materialized_metadata import implementation_bundle_sha256

    authority = source.authority
    with AnchoredDirectory.open(source_path.parent) as directory:
        trajectory = pq.read_table(
            io.BytesIO(
                directory.read_bytes(
                    source_path.name,
                    max_bytes=authority.trajectory_size,
                    expected_size=authority.trajectory_size,
                    expected_sha256=authority.trajectory_sha256,
                )
            )
        )
    with AnchoredDirectory.open(child_path.parent) as directory:
        retained = pq.read_table(
            io.BytesIO(
                directory.read_bytes(
                    child_path.name,
                    max_bytes=child.authority.cohort_size,
                    expected_size=child.authority.cohort_size,
                    expected_sha256=child.authority.cohort_sha256,
                )
            ),
            columns=[child.authority.identity_column],
        ).column(0)
    identity = trajectory.column(authority.identity_column)
    kept = trajectory.filter(
        pc.is_in(identity, value_set=retained.combine_chunks().cast(identity.type))
    )
    observed = set(kept.column(authority.concept_column).to_pylist())
    available = {*authority.materialized_concepts, *authority.available_unobserved_concepts}
    materialized = [c for c in authority.requested_concepts if c in observed]
    unobserved = [
        c for c in authority.requested_concepts if c in available and c not in observed
    ]
    window = (
        (authority.window.start_hours, authority.window.end_hours)
        if authority.window is not None
        else None
    )
    frame = kept.to_pandas()
    recorded = authority.to_dict()
    parameters = {
        "database": child.sidecar.source_database,
        "requested_concepts": list(authority.requested_concepts),
        "materialized_concepts": materialized,
        "available_unobserved_concepts": unobserved,
        "unavailable_concepts": list(authority.unavailable_concepts),
        "window": list(window) if window is not None else None,
        "bound_universe_authority_sha256": child.reference.sha256,
        **{
            key: recorded["producer_parameters"][key]
            for key in ("bounds_violation_policy", "source_bounds_exclusions")
            if key in recorded["producer_parameters"]
        },
    }
    provenance = {
        **recorded["semantic_provenance"],
        "n_rows": kept.num_rows,
        "n_stays": len(set(kept.column(authority.identity_column).to_pylist())),
        "trajectory_concepts_materialized": materialized,
        "available_unobserved_concepts": unobserved,
        "trajectory_sha256": _cohort_sha256(frame),
        "hospital_followup_restriction": {
            "source_trajectory_authority_sha256": source.reference.sha256,
            "source_trajectory_sha256": authority.trajectory_sha256,
            "source_universe_authority_sha256": (
                authority.bound_universe_authority.sha256
            ),
            "source_rows": authority.trajectory_rows,
            "source_stays": authority.trajectory_stays,
        },
    }
    return trajectory_owner.publish_materialized_trajectory_authority(
        frame,
        target,
        bound_universe_path=child_path,
        bound_universe=child,
        requested_concepts=authority.requested_concepts,
        materialized_concepts=materialized,
        available_unobserved_concepts=unobserved,
        unavailable_concepts=authority.unavailable_concepts,
        window=window,
        semantic_provenance=provenance,
        producer_implementation_sha256=implementation_bundle_sha256(
            (Path(__file__), Path(trajectory_owner.__file__))
        ),
        producer_parameters=parameters,
    )


def materialize_hospital_followup_acquisition(
    acquisition: AcquisitionResult,
    *,
    followup: HospitalMortalityFollowup,
    raw_source_receipt: Mapping[str, Any],
) -> AcquisitionResult:
    if acquisition.cohort_authority_ref is not None:
        return _materialize_typed_hospital_followup(
            acquisition, followup=followup, raw_source_receipt=raw_source_receipt
        )
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
