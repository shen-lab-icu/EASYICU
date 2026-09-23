"""Run the declared cross-concept cohort derivations.

The declaration lives in
:mod:`easyicu.research_agent.contracts.host_derivations`; this module owns the
computation for each declared identifier.  It reads already-loaded per-concept
frames, never the source files, so the materializer keeps a single place where
concepts are resolved, bounded and typed.

Each derivation returns exactly one row per requested stay, with exactly the
columns its declaration publishes.  A stay with no source rows in the window is
returned as an unknown reading, never dropped and never coerced to a negative.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import pandas as pd

from easyicu.concept.metadata_projection import ConceptColumnRole

from ..contracts.host_derivations import (
    STRICT_KDIGO_DERIVATION_ID,
    HostDerivation,
    HostDerivationError,
)


def _join_source_rows(
    frames: Mapping[str, pd.DataFrame],
    *,
    concepts: Sequence[str],
    id_column: str,
    time_column: str,
) -> pd.DataFrame:
    """Rebuild the source rows from one frame per concept.

    The concepts of one declared derivation come from the same physical module,
    so an outer join on the identity/time coordinate restores the original rows
    without inventing any.
    """

    joined: pd.DataFrame | None = None
    for concept in concepts:
        frame = frames.get(concept)
        if frame is None:
            raise HostDerivationError(
                f"host derivation input {concept!r} was not loaded"
            )
        if id_column not in frame.columns:
            raise HostDerivationError(
                f"host derivation input {concept!r} has no {id_column!r} column"
            )
        if time_column not in frame.columns:
            raise HostDerivationError(
                f"host derivation input {concept!r} has no {time_column!r} column; "
                "a cross-concept window derivation needs the source time axis"
            )
        if concept not in frame.columns:
            raise HostDerivationError(
                f"host derivation input {concept!r} lost its value column"
            )
        part = frame[[id_column, time_column, concept]]
        if bool(part.duplicated(subset=[id_column, time_column]).any()):
            # The concepts of one derivation share a physical file, so a
            # repeated coordinate means the rows cannot be rejoined into the
            # source rows the reading assumes.  Keeping the first value would
            # silently choose one of two contradictory measurements.
            raise HostDerivationError(
                f"host derivation input {concept!r} has repeated "
                f"({id_column}, {time_column}) coordinates"
            )
        joined = (
            part
            if joined is None
            else joined.merge(part, on=[id_column, time_column], how="outer")
        )
    if joined is None:  # pragma: no cover - a declaration always names concepts
        raise HostDerivationError("host derivation read no concepts")
    return joined


def _strict_kdigo(
    *,
    frames: Mapping[str, pd.DataFrame],
    derivation: HostDerivation,
    identities: Sequence[object],
    id_column: str,
    time_column: str,
    window: tuple[float, float],
) -> pd.DataFrame:
    from ...scores.aki_strict import summarize_strict_kdigo_window

    rows = _join_source_rows(
        frames,
        concepts=derivation.source_concepts,
        id_column=id_column,
        time_column=time_column,
    )
    summary = summarize_strict_kdigo_window(
        rows,
        id_column=id_column,
        time_column=time_column,
        window_start_hours=float(window[0]),
        window_end_hours=float(window[1]),
    )
    return summary


_IMPLEMENTATIONS: Mapping[str, Callable[..., pd.DataFrame]] = {
    STRICT_KDIGO_DERIVATION_ID: _strict_kdigo,
}


def materialize_host_derivation(
    derivation: HostDerivation,
    *,
    frames: Mapping[str, pd.DataFrame],
    identities: Sequence[object],
    id_column: str,
    time_column: str,
    window: tuple[float, float],
) -> pd.DataFrame:
    """Compute one declared derivation for exactly the requested stays."""

    implementation = _IMPLEMENTATIONS.get(derivation.derivation_id)
    if implementation is None:
        raise HostDerivationError(
            f"declared host derivation {derivation.derivation_id!r} has no "
            "registered implementation"
        )
    summary = implementation(
        frames=frames,
        derivation=derivation,
        identities=identities,
        id_column=id_column,
        time_column=time_column,
        window=window,
    )
    declared = list(derivation.output_columns)
    missing = [column for column in declared if column not in summary.columns]
    if missing:
        raise HostDerivationError(
            f"{derivation.derivation_id} did not publish: " + ", ".join(missing)
        )
    # Every cohort stay gets a row.  For a stay the derivation never saw, a
    # value column stays unknown -- that is the whole point of the reading --
    # while a count or a measurement status is a real negative, because the
    # window was searched and was empty.
    spine = pd.DataFrame({id_column: pd.Series(list(identities)).drop_duplicates()})
    result = spine.merge(
        summary[[id_column, *declared]], on=id_column, how="left", validate="one_to_one"
    )
    for output in derivation.outputs:
        if output.role is ConceptColumnRole.COUNT:
            result[output.column] = (
                pd.to_numeric(result[output.column], errors="coerce")
                .fillna(0)
                .astype("int64")
            )
        elif output.role is ConceptColumnRole.MEASUREMENT_STATUS:
            result[output.column] = (
                result[output.column]
                .astype("object")
                .map(lambda value: 0 if value is None or value is pd.NA else int(bool(value)))
                .astype("int64")
            )
    return result


__all__ = ["materialize_host_derivation"]
