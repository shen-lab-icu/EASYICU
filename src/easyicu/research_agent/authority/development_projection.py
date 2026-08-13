"""Host-owned typed-input projection for non-paper development samples.

The Planner's typed product remains bound to its full-cohort producer.  For a
development run only, this module selects the exact registered sample bytes as
the physical execution input and records both the declared parent evidence and
the locked-cohort parent.  It never promotes the sample to paper authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from ..contracts.primary_cohort import locked_primary_cohort_product
from .evidence_store import sha256_of_file
from .runtime_artifacts import verified_run_evidence_path
from .typed_input_receipt import typed_input_row_identity_sha256

DEVELOPMENT_COHORT_EVIDENCE_ID = "development_execution_cohort"
DEVELOPMENT_INPUT_PROJECTION_SCHEMA = "easyicu.development_input_projection/1"
DEVELOPMENT_PRIMARY_COHORT_CONFIRMATION_ROLE = (
    "post_qc_development_sample_for_primary_cohort_confirmation"
)


@dataclass(frozen=True, slots=True)
class DevelopmentInputProjection:
    """Verified physical child plus durable parent/row-identity coordinates."""

    evidence_record: Any
    verified_path: Path
    authority_payload: Mapping[str, Any]
    row_identity_contract: Mapping[str, Any]


def _record_field(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def resolve_development_input_projection(
    *,
    declared_input: str,
    parent_evidence_id: str,
    parent_sha256: str,
    parent_produced_by_step: str,
    parent_verified_path: Path,
    evidence_records: Sequence[Any],
    run_dir: Path,
    authoritative_cohort_path: Optional[Path],
    development_sample: Optional[Any],
    locked_cohort_name: object = None,
) -> Optional[DevelopmentInputProjection]:
    """Return the exact development child, or ``None`` on any ambiguity.

    ``declared_input`` is the raw Planner token, because the primary-cohort
    identity is owned by ``declared_product`` and cannot be recovered from the
    canonical ``kind:product`` pair.  ``locked_cohort_name`` is the plan's own
    cohort name, needed for the same reason: a Planner that declared the
    population under that name declared *this* population, and a surface that
    cannot see it skips the projection and hands the step the full cohort
    while the run-level plane still mounts and reports the sample.
    """

    if (
        development_sample is None
        or locked_primary_cohort_product(
            declared_input, locked_cohort_name=locked_cohort_name
        )
        is None
    ):
        return None
    try:
        selected_cohort_path = Path(authoritative_cohort_path or "").resolve()
        sample_source_path = Path(development_sample.cohort_path).resolve()
        sample_manifest_path = Path(development_sample.manifest_path)
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if selected_cohort_path != sample_source_path:
        return None
    sample_record = next(
        (
            candidate
            for candidate in evidence_records
            if str(_record_field(candidate, "evidence_id") or "")
            == DEVELOPMENT_COHORT_EVIDENCE_ID
        ),
        None,
    )
    if sample_record is None or str(_record_field(sample_record, "kind") or "") != (
        "table"
    ):
        return None
    sample_verified_path = verified_run_evidence_path(run_dir, sample_record)
    sample_metadata = _record_field(sample_record, "metadata")
    if sample_verified_path is None or not isinstance(sample_metadata, Mapping):
        return None
    sample_sha256 = str(_record_field(sample_record, "sha256") or "")
    try:
        manifest_sha256 = sha256_of_file(sample_manifest_path)
        exact_projection = (
            sample_sha256 == str(development_sample.sample_sha256)
            and sha256_of_file(sample_source_path) == sample_sha256
            and str(sample_metadata.get("sample_cohort_sha256") or "") == sample_sha256
            and str(sample_metadata.get("parent_cohort_sha256") or "")
            == str(development_sample.parent_sha256)
            and str(sample_metadata.get("sample_manifest_sha256") or "")
            == manifest_sha256
            and sample_metadata.get("paper_authority") is False
        )
    except (OSError, TypeError, ValueError):
        return None
    if not exact_projection:
        return None

    identity_column = str(development_sample.identity_column or "")
    try:
        identity_frame = pd.read_parquet(
            sample_verified_path,
            columns=[identity_column],
        )
    except (OSError, TypeError, ValueError, ImportError):
        return None
    if (
        not identity_column
        or identity_column not in identity_frame.columns
        or identity_frame[identity_column].isna().any()
        or identity_frame[identity_column].astype("string").duplicated().any()
        or len(identity_frame) != int(development_sample.selected_rows)
    ):
        return None

    # The sample was drawn from the locked cohort, but the input it replaces
    # is whatever the declared producer emitted.  If that producer narrowed
    # the population, substituting the sample would hand the consumer rows its
    # own producer excluded, so require the substitution to be a real subset
    # of the declared parent rather than assuming the two agree.
    try:
        parent_identities = pd.read_parquet(
            parent_verified_path,
            columns=[identity_column],
        )
    except (OSError, TypeError, ValueError, ImportError):
        return None
    if identity_column not in parent_identities.columns:
        return None
    sampled = set(identity_frame[identity_column].astype("string"))
    if not sampled.issubset(set(parent_identities[identity_column].astype("string"))):
        return None

    authority_payload = {
        "schema_version": DEVELOPMENT_INPUT_PROJECTION_SCHEMA,
        "paper_authority": False,
        "projection_kind": "ordered_subset_of_locked_analysis_cohort",
        "declared_parent_input": {
            "evidence_id": parent_evidence_id,
            "sha256": parent_sha256,
            "produced_by_step": parent_produced_by_step,
        },
        "locked_parent_cohort_sha256": str(development_sample.parent_sha256),
        "sample_manifest_sha256": manifest_sha256,
        "selected_positions_sha256": str(development_sample.selected_positions_sha256),
        "seed": int(development_sample.seed),
        "selected_rows": int(development_sample.selected_rows),
    }
    row_identity_contract = {
        "row_identity_column": identity_column,
        "row_count": int(len(identity_frame)),
        "row_identity_sha256": typed_input_row_identity_sha256(
            identity_frame[identity_column]
        ),
    }
    return DevelopmentInputProjection(
        evidence_record=sample_record,
        verified_path=sample_verified_path,
        authority_payload=authority_payload,
        row_identity_contract=row_identity_contract,
    )


__all__ = [
    "DEVELOPMENT_COHORT_EVIDENCE_ID",
    "DEVELOPMENT_INPUT_PROJECTION_SCHEMA",
    "DevelopmentInputProjection",
    "resolve_development_input_projection",
]
