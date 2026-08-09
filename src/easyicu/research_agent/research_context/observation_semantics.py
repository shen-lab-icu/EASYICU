"""Compile typed observation semantics from verified cohort representations.

Owner responsibility: distinguish raw nulls from event absence or conditional
non-applicability.  This module does not choose exposures, outcomes, cohorts,
or models.  It only accepts representations that the existing source-status
validators can prove from the locked frame.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ..methods.source_status import (
    reconcile_binary_event_presence,
    reconcile_conditional_event_time,
)
from ..schema import ConceptDescriptor, ObservationSemantics

__all__ = ["compile_observation_semantics"]


def _severity(fraction: float) -> str:
    if fraction == 0.0 or fraction < 0.05:
        return "low"
    if fraction < 0.30:
        return "medium"
    return "high"


def _representative_columns(frame: pd.DataFrame, base: str) -> tuple[str, ...]:
    candidates = (
        base,
        f"{base}_first",
        f"{base}_max",
        f"{base}_min",
        f"{base}_last",
    )
    return tuple(column for column in candidates if column in frame.columns)


def _semantic_profile(
    descriptor: ConceptDescriptor,
    *,
    raw_n_missing: int,
    eligible_n: int,
    not_applicable_n: int,
    n_missing: int,
    note: str,
) -> ConceptDescriptor:
    profile = descriptor.missingness
    if profile is None:
        return descriptor
    fraction = n_missing / eligible_n if eligible_n else 0.0
    return descriptor.model_copy(
        update={
            "missingness": profile.model_copy(
                update={
                    "fraction_missing": fraction,
                    "n_missing": n_missing,
                    "raw_n_missing": raw_n_missing,
                    "eligible_n": eligible_n,
                    "not_applicable_n": not_applicable_n,
                    "missingness_severity": _severity(fraction),
                    "missingness_test": "not_applicable_typed_observation_semantics",
                    "missingness_test_p_value": None,
                    "notes": note,
                }
            )
        }
    )


def _positive_only_event_updates(
    frame: pd.DataFrame,
    descriptors: dict[str, ConceptDescriptor],
) -> dict[str, ConceptDescriptor]:
    updates: dict[str, ConceptDescriptor] = {}
    measured_columns = sorted(
        column for column in frame.columns if column.endswith("_measured")
    )
    for measured_column in measured_columns:
        base = measured_column[: -len("_measured")]
        count_column = f"{base}_n"
        if count_column not in frame.columns:
            continue
        for representative_column in _representative_columns(frame, base):
            representative = pd.to_numeric(
                frame[representative_column],
                errors="coerce",
            )
            observed_levels = set(representative.dropna().unique().tolist())
            if (
                representative.notna().all()
                or not observed_levels
                or not observed_levels.issubset({1, 1.0})
            ):
                continue
            try:
                result = reconcile_binary_event_presence(
                    frame,
                    count_column=count_column,
                    measured_column=measured_column,
                    representative_column=representative_column,
                )
            except ValueError:
                continue
            if not result.audit["event_absent_n"]:
                continue
            descriptor = descriptors.get(representative_column)
            if descriptor is None:
                continue
            semantics = ObservationSemantics(
                kind="positive_only_event",
                event_count_column=count_column,
                measured_column=measured_column,
                representative_column=representative_column,
            )
            raw_n_missing = int(frame[representative_column].isna().sum())
            note = (
                "Raw nulls are reconciled event-absent rows, not unmeasured "
                "values; count, measured flag, and positive-only representative "
                "agree for every row."
            )
            updated = _semantic_profile(
                descriptor,
                raw_n_missing=raw_n_missing,
                eligible_n=len(frame),
                not_applicable_n=0,
                n_missing=0,
                note=note,
            )
            updates[representative_column] = updated.model_copy(
                update={
                    "observation_semantics": semantics,
                    "missingness_semantics": note,
                }
            )
    return updates


def _conditional_event_time_updates(
    frame: pd.DataFrame,
    descriptors: dict[str, ConceptDescriptor],
) -> dict[str, ConceptDescriptor]:
    updates: dict[str, ConceptDescriptor] = {}
    for descriptor in descriptors.values():
        if (
            descriptor.name not in frame.columns
            or descriptor.unit_normalization != "first_truthy_event_time"
            or not descriptor.source_concept
        ):
            continue
        candidates = [
            candidate
            for candidate in descriptors.values()
            if candidate.name != descriptor.name
            and candidate.name in frame.columns
            and candidate.source_concept == descriptor.source_concept
            and isinstance(candidate.observed_domain, dict)
            and candidate.observed_domain.get("is_binary") is True
            and candidate.missingness is not None
            and candidate.missingness.n_missing == 0
        ]
        candidates.sort(
            key=lambda candidate: (
                candidate.name != descriptor.source_concept,
                candidate.name,
            )
        )
        if not candidates:
            continue
        event_status_column = candidates[0].name
        try:
            result = reconcile_conditional_event_time(
                frame,
                event_status_column=event_status_column,
                event_time_column=descriptor.name,
            )
        except ValueError:
            continue
        audit = result.audit
        raw_n_missing = int(frame[descriptor.name].isna().sum())
        note = (
            f"{descriptor.name} is applicable only when "
            f"{event_status_column}=1; event-negative rows are not missing."
        )
        updated = _semantic_profile(
            descriptor,
            raw_n_missing=raw_n_missing,
            eligible_n=int(audit["eligible_event_n"]),
            not_applicable_n=int(audit["not_applicable_event_absent_n"]),
            n_missing=int(audit["missing_event_time_n"]),
            note=note,
        )
        caveats = list(updated.clinical_caveats)
        if audit["before_origin_n"]:
            caveat = (
                f"{audit['before_origin_n']} observed {descriptor.name} values "
                "precede the declared time origin and require a study-specific "
                "temporal protocol."
            )
            if caveat not in caveats:
                caveats.append(caveat)
        temporal_resolution = str(descriptor.temporal_resolution or "")
        relative_time = (
            temporal_resolution.removeprefix("relative to ")
            if temporal_resolution.startswith("relative to ")
            else ""
        )
        time_origin, separator, time_unit = relative_time.rpartition(" in ")
        updates[descriptor.name] = updated.model_copy(
            update={
                "observation_semantics": ObservationSemantics(
                    kind="conditional_event_time",
                    event_status_column=event_status_column,
                    representative_column=descriptor.name,
                    time_origin=time_origin or None if separator else None,
                    time_unit=time_unit or None if separator else None,
                ),
                "missingness_semantics": note,
                "clinical_caveats": caveats,
            }
        )
    return updates


def compile_observation_semantics(
    *,
    frame: pd.DataFrame,
    descriptors: Sequence[ConceptDescriptor],
) -> list[ConceptDescriptor]:
    """Return descriptors enriched only by mechanically verified semantics."""

    by_name = {descriptor.name: descriptor for descriptor in descriptors}
    updates = _positive_only_event_updates(frame, by_name)
    by_name.update(updates)
    updates.update(_conditional_event_time_updates(frame, by_name))
    return [updates.get(descriptor.name, descriptor) for descriptor in descriptors]
