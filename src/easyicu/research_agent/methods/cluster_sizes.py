"""Counts of existing cluster labels on a declared unique unit roster."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..contracts.cluster_sizes import (
    ClusterSizeResult,
    ClusterSizeRow,
    ClusterSizeSpec,
    cluster_size_spec_sha256,
)
from ..contracts.model_terms import level_identity_class, level_spelling


def _identities(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise ValueError("cluster_sizes_identity_missing")
    identities = series.astype(str)
    if (
        series.duplicated().any()
        or identities.str.strip().eq("").any()
        or identities.duplicated().any()
    ):
        raise ValueError("cluster_sizes_analysis_unit_not_unique")
    return identities


def compute_cluster_sizes(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    spec: ClusterSizeSpec | Mapping[str, Any],
) -> ClusterSizeResult:
    """Connect by key and count units, never fit clusters or count repeated rows."""
    spec = ClusterSizeSpec.model_validate(spec)
    if not frame.columns.is_unique or spec.unit_id_column not in frame:
        raise ValueError("cluster_sizes_unit_column_missing_or_duplicate")
    if not assignments.columns.is_unique or not {
        spec.assignment_unit_id_column,
        spec.label_column,
    }.issubset(assignments.columns):
        raise ValueError("cluster_sizes_assignment_columns_missing_or_duplicate")
    if frame.empty:
        raise ValueError("cluster_sizes_empty_input")
    ids = _identities(frame[spec.unit_id_column])
    assignment_ids = _identities(assignments[spec.assignment_unit_id_column])
    if not set(assignment_ids).issubset(set(ids)):
        raise ValueError("cluster_sizes_assignment_outside_cohort")
    spellings: dict[str, str] = {}
    for value in assignments[spec.label_column].dropna().tolist():
        spelling, identity_class = level_spelling(value), level_identity_class(value)
        if isinstance(value, str) and value != value.strip():
            raise ValueError("cluster_sizes_noncanonical_label")
        if spelling in spellings and spellings[spelling] != identity_class:
            raise ValueError("cluster_sizes_ambiguous_label_types")
        spellings[spelling] = identity_class
    source_labels = assignments[spec.label_column].map(
        lambda value: level_spelling(value) if pd.notna(value) else None
    )
    if not set(source_labels.dropna()).issubset(spec.expected_clusters):
        raise ValueError("cluster_sizes_undeclared_labels")
    mapping = pd.Series(source_labels.to_numpy(), index=assignment_ids)
    labels = ids.map(mapping)
    missing = int(labels.isna().sum())
    if missing and spec.missing_label_policy == "error":
        raise ValueError(f"cluster_sizes_missing_labels: n={missing}")
    assigned = int(labels.notna().sum())
    if assigned == 0:
        raise ValueError("cluster_sizes_no_assigned_units")
    rows = []
    for cluster in spec.expected_clusters:
        n = int(labels.eq(cluster).sum())
        proportion = n / assigned
        near_empty = None
        if spec.near_empty_n is not None or spec.near_empty_fraction is not None:
            near_empty = bool(
                (spec.near_empty_n is not None and n < spec.near_empty_n)
                or (
                    spec.near_empty_fraction is not None
                    and proportion < spec.near_empty_fraction
                )
            )
        rows.append(
            ClusterSizeRow(
                cluster=cluster,
                n_units=n,
                proportion_assigned=proportion,
                proportion_all_units=n / len(ids),
                is_empty=n == 0,
                is_near_empty=near_empty,
            )
        )
    observed = sum(row.n_units > 0 for row in rows)
    return ClusterSizeResult(
        specification_sha256=cluster_size_spec_sha256(spec),
        analysis_unit=spec.analysis_unit,
        input_n=len(ids),
        assignment_rows_n=len(assignments),
        assigned_n=assigned,
        missing_label_n=missing,
        unmatched_unit_n=int((~ids.isin(mapping.index)).sum()),
        matched_missing_label_n=int(assignments[spec.label_column].isna().sum()),
        observed_cluster_n=observed,
        degenerate_single_cluster=observed == 1,
        rows=tuple(rows),
    )
