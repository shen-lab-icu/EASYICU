"""Source-bound legacy-export bridge to opaque counting-process artifacts.

Wide cohort rows and longitudinal intervals are different products: one stay
is never counted as several patients. The source artifacts remain untouched.
Native typed exports fail closed until this transformation has their own
column-lineage extension; their authority must never be stripped or forged.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..authority.filesystem import AnchoredDirectory
from ..canonical_json import canonical_json_bytes, sha256_bytes, sha256_file
from ..contracts.time_varying_exposure import (
    TIME_VARYING_INPUT_METADATA_KEY,
    TimeVaryingExposureSpecification,
)
from .foundation import AcquisitionResult
from .hospital_mortality_followup import HospitalMortalityFollowup
from .patient_grouping import PatientGroupingBinding
from .time_varying_execution_input import build_time_varying_execution_input
from .time_varying_exposure import build_early_running_max_exposure_panel


class TimeVaryingMaterializationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _read_frame(path: Path) -> tuple[pd.DataFrame, str]:
    with AnchoredDirectory.open(path.parent) as directory:
        payload = directory.read_bytes(path.name, max_bytes=512 * 1024 * 1024)
    return pd.read_parquet(io.BytesIO(payload)), sha256_bytes(payload)


def _wide_cohort(
    frame: pd.DataFrame,
    specification: TimeVaryingExposureSpecification,
    *,
    exposure_column: str,
) -> pd.DataFrame:
    final = (
        frame.groupby("analysis_stay_index", sort=False).tail(1).reset_index(drop=True)
    )
    cohort = pd.DataFrame(
        {
            "patient_stay_id": "p"
            + final["analysis_cluster_index"].astype(str)
            + ":s"
            + final["analysis_stay_index"].astype(str),
            "death": final["hospital_death"].to_numpy(dtype="int8"),
            "hospital_followup_time_hours": final["interval_stop_hours"].to_numpy(),
            "death_time_hours": np.where(
                final["hospital_death"].eq(1), final["interval_stop_hours"], np.nan
            ),
            exposure_column: final["exposure_running_max_when_observed"]
            .where(final["exposure_unmeasured_indicator"].eq(0))
            .to_numpy(),
        }
    )
    for column in specification.baseline_columns:
        encoding = specification.baseline_categorical_encodings.get(column)
        cohort[column] = (
            final[column].to_numpy()
            if encoding is None
            else np.where(
                final[encoding.output_column].eq(1),
                encoding.positive_level,
                encoding.negative_level,
            )
        )
    return cohort


def materialize_time_varying_acquisition(
    acquisition: AcquisitionResult,
    *,
    specification: TimeVaryingExposureSpecification,
    hospital_followup: HospitalMortalityFollowup,
    raw_source_receipt: Mapping[str, Any],
    patient_grouping: PatientGroupingBinding,
    exposure_column: str,
) -> AcquisitionResult:
    """Publish a separate one-stay cohort and digest-bound interval input."""
    if (
        acquisition.cohort_authority_ref is not None
        or acquisition.trajectory_authority_ref is not None
    ):
        raise TimeVaryingMaterializationError(
            "time_varying_native_lineage_extension_required",
            "Native typed exports require a typed counting-process lineage extension; authority cannot be downgraded.",
        )
    if acquisition.universe_path is None or acquisition.trajectory_path is None:
        raise TimeVaryingMaterializationError(
            "time_varying_source_trajectory_required",
            "A source cohort and timestamped trajectory are required.",
        )
    baseline, cohort_digest = _read_frame(Path(acquisition.universe_path))
    trajectory, trajectory_digest = _read_frame(Path(acquisition.trajectory_path))
    if "stay_id" not in baseline or baseline["stay_id"].duplicated().any():
        raise TimeVaryingMaterializationError(
            "time_varying_source_identity_invalid",
            "Source cohort must have one row per stay.",
        )
    followup = hospital_followup.frame.loc[
        hospital_followup.frame["stay_id"].isin(baseline["stay_id"])
    ].copy()
    accounted = set(followup["stay_id"]) | set(hospital_followup.exclusions["stay_id"])
    if not set(baseline["stay_id"]).issubset(accounted):
        raise TimeVaryingMaterializationError(
            "time_varying_followup_coverage_incomplete",
            "Every source stay needs follow-up or an explicit exclusion reason.",
        )
    panel = build_early_running_max_exposure_panel(
        trajectory,
        followup,
        exposure_concept=specification.exposure_concept,
    )
    model = build_time_varying_execution_input(
        panel.panel,
        baseline,
        patient_grouping,
        baseline_columns=specification.baseline_columns,
        missingness_policy=specification.missingness_policy,
        baseline_categorical_encodings={
            key: value.model_dump(mode="json")
            for key, value in specification.baseline_categorical_encodings.items()
        },
    )
    cohort = _wide_cohort(model.frame, specification, exposure_column=exposure_column)
    receipt = {
        "schema_version": "easyicu.time_varying_materialization/1",
        "specification_sha256": specification.sha256,
        "specification": specification.model_dump(mode="json"),
        "source_cohort_sha256": cohort_digest,
        "source_trajectory_sha256": trajectory_digest,
        "raw_source": dict(raw_source_receipt),
        "followup": dict(hospital_followup.receipt),
        "exposure_panel": dict(panel.receipt),
        "execution_input": dict(model.receipt),
        "source_stays": int(len(baseline)),
        "analysis_stays": int(len(cohort)),
        "excluded_stays": int(len(baseline) - len(cohort)),
        "source_metadata_kind": "legacy_untyped",
        "claim_ceiling": "analysis_only",
    }
    root = Path(acquisition.universe_path).parent
    cohort_path, input_path = (
        root / "time_varying_cohort.parquet",
        root / "time_varying_intervals.parquet",
    )
    provenance_path = root / "time_varying_cohort_provenance.json"
    trajectory_provenance_path = root / "time_varying_intervals_provenance.json"
    if any(
        path.exists()
        for path in (
            cohort_path,
            input_path,
            provenance_path,
            trajectory_provenance_path,
        )
    ):
        raise TimeVaryingMaterializationError(
            "time_varying_artifact_exists",
            "Derived input paths already exist; do not overwrite a previous run.",
        )
    cohort.to_parquet(cohort_path, index=False)
    receipt["analysis_cohort_sha256"] = sha256_file(cohort_path)
    table = pa.Table.from_pandas(model.frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[TIME_VARYING_INPUT_METADATA_KEY.encode()] = canonical_json_bytes(receipt)
    pq.write_table(table.replace_schema_metadata(metadata), input_path)
    # Preserve the materializer provenance shape consumed by intake, but
    # explicitly replace the row count, columns and identity with this owner.
    provenance = json.loads(Path(acquisition.provenance_path).read_text())
    provenance.update(
        {
            "n_rows": len(cohort),
            "n_stays_after_inclusion_exclusion": len(cohort),
            "columns": list(cohort.columns),
            "cohort_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(
                    cohort.reset_index(drop=True), index=False
                ).values.tobytes()
            ).hexdigest(),
            "cohort_file_sha256": sha256_file(cohort_path),
            "cohort_file_size": cohort_path.stat().st_size,
            "time_varying_materialization": receipt,
        }
    )
    parameters = dict(provenance.get("producer_parameters") or {})
    parameters.update(
        {
            "identity_column": "patient_stay_id",
            "replacement_row_identity": {
                "mapping_file_sha256": patient_grouping.mapping_sha256,
                "output_identity_column": "patient_stay_id",
                "mapped_cohort_rows": len(cohort),
                "patient_group_derivation": {
                    "algorithm": "prefix_before_:s",
                    "delimiter": ":s",
                },
                "encoding": "per_run_opaque_patient_and_stay_indices",
                "authority_coordinates": dict(patient_grouping.authority_coordinates),
            },
        }
    )
    provenance["producer_parameters"] = parameters
    provenance["identity_column"] = "patient_stay_id"
    provenance["replacement_row_identity"] = parameters["replacement_row_identity"]
    provenance["n_patients"] = len(cohort)
    provenance_path.write_bytes(canonical_json_bytes(provenance))
    trajectory_provenance_path.write_bytes(
        canonical_json_bytes(
            {
                **receipt,
                "artifact_sha256": sha256_file(input_path),
                "artifact_kind": "opaque_counting_process_intervals",
            }
        )
    )
    return replace(
        acquisition,
        universe_path=cohort_path,
        provenance_path=provenance_path,
        trajectory_path=input_path,
        trajectory_provenance_path=trajectory_provenance_path,
        materialized_columns=tuple(cohort.columns),
        analysis_columns={
            "death": "death",
            specification.exposure_concept: exposure_column,
            **{column: column for column in specification.baseline_columns},
        },
        note="Source-bound time-updated analysis-only input; raw identities remain local.",
    )


__all__ = ["TimeVaryingMaterializationError", "materialize_time_varying_acquisition"]
