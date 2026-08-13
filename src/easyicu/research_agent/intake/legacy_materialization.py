"""Verification leaf for legacy cohort-materializer provenance receipts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from .materialized_metadata import MaterializedMetadataError


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_legacy_materialization_provenance(
    cohort_path: Union[str, Path],
    *,
    cohort: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, Any]]:
    """Return a digest-bound legacy cohort-materializer receipt, when present.

    Modern typed materializations carry per-column derivation windows in their
    sealed column-metadata authority. Older export packages cannot produce that
    sidecar, but the cohort materializer still writes an adjacent
    ``<stem>_provenance.json`` receipt containing the exact cohort window and
    file binding. This loader accepts only that closed schema and verifies it
    against the selected parquet before temporal metadata reaches the Agent.

    A non-materializer sidecar is outside this compatibility contract and
    returns ``None``. A claimed legacy receipt that is malformed or mismatched
    fails closed.
    """

    selected = Path(cohort_path).expanduser().resolve()
    provenance_path = selected.with_name(f"{selected.stem}_provenance.json")
    if not provenance_path.exists():
        return None
    if provenance_path.is_symlink() or not provenance_path.is_file():
        raise MaterializedMetadataError(
            "legacy materialization provenance must be a regular file"
        )
    try:
        raw_bytes = provenance_path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializedMetadataError(
            "legacy materialization provenance is unreadable"
        ) from exc
    if not isinstance(payload, dict):
        raise MaterializedMetadataError(
            "legacy materialization provenance must be a JSON object"
        )
    if payload.get("schema_version") != "easyicu.cohort_materializer/1":
        return None

    required = {
        "cohort_window_hours",
        "feature_concepts",
        "outcome_concepts",
        "static_concepts",
        "n_stays_after_inclusion_exclusion",
        "columns",
        "cohort_sha256",
        "cohort_file_sha256",
        "cohort_file_size",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise MaterializedMetadataError(
            "legacy materialization provenance lacks required fields: "
            + ", ".join(missing)
        )

    frame = cohort if cohort is not None else pd.read_parquet(selected)
    if not isinstance(frame, pd.DataFrame):
        raise MaterializedMetadataError("legacy materialization cohort is not tabular")
    if payload.get("columns") != list(frame.columns):
        raise MaterializedMetadataError(
            "legacy materialization provenance column order does not match cohort"
        )
    if payload.get("n_stays_after_inclusion_exclusion") != int(len(frame)):
        raise MaterializedMetadataError(
            "legacy materialization provenance row count does not match cohort"
        )
    expected_file_sha = payload.get("cohort_file_sha256")
    expected_file_size = payload.get("cohort_file_size")
    if (
        not isinstance(expected_file_sha, str)
        or len(expected_file_sha) != 64
        or isinstance(expected_file_size, bool)
        or not isinstance(expected_file_size, int)
        or expected_file_size < 0
        or selected.stat().st_size != expected_file_size
        or _sha256_file(selected) != expected_file_sha
    ):
        raise MaterializedMetadataError(
            "legacy materialization provenance file binding does not match cohort"
        )

    window = payload.get("cohort_window_hours")
    if (
        not isinstance(window, list)
        or len(window) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in window
        )
        or not all(np.isfinite(float(value)) for value in window)
        or float(window[0]) > float(window[1])
    ):
        raise MaterializedMetadataError(
            "legacy materialization provenance has an invalid cohort window"
        )
    for key in ("feature_concepts", "outcome_concepts", "static_concepts"):
        values = payload.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise MaterializedMetadataError(
                f"legacy materialization provenance has invalid {key}"
            )

    replacement = payload.get("replacement_row_identity")
    if replacement is not None:
        if not isinstance(replacement, dict):
            raise MaterializedMetadataError(
                "legacy replacement-row identity must be an object"
            )
        output_column = replacement.get("output_identity_column")
        mapping_sha256 = replacement.get("mapping_file_sha256")
        mapped_rows = replacement.get("mapped_cohort_rows")
        derivation = replacement.get("patient_group_derivation")
        coordinates = replacement.get("authority_coordinates")
        if (
            not isinstance(output_column, str)
            or output_column not in frame.columns
            or not isinstance(mapping_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", mapping_sha256) is None
            or isinstance(mapped_rows, bool)
            or not isinstance(mapped_rows, int)
            or mapped_rows != int(len(frame))
            or not isinstance(derivation, dict)
            or derivation
            != {"algorithm": "prefix_before_:s", "delimiter": ":s"}
            or not isinstance(coordinates, dict)
            or coordinates.get("schema_version")
            != "easyicu.patient_grouping_runtime_authority/1"
            or not isinstance(coordinates.get("authority_ref"), str)
            or not coordinates.get("authority_ref")
            or coordinates.get("provider_visible_values") is not False
        ):
            raise MaterializedMetadataError(
                "legacy replacement-row identity authority is invalid"
            )

    verified = dict(payload)
    verified["provenance_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    return verified


__all__ = ["load_verified_legacy_materialization_provenance"]
