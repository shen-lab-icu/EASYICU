"""Materialize an analysis cohort directly from the EasyICU data layer.

This is the trusted bridge that lets the research agent's *first step* be
real cohort extraction + inclusion/exclusion, rather than relying on a
hand-built cohort parquet. It runs **outside** the network-isolated
analysis sandbox (it needs data access), and produces the per-stay cohort
parquet the sandbox then consumes.

Two sources, auto-detected:

* **converted database** (a ``data_path`` to a ricu-style prepared MIMIC/eICU
  directory) -> concepts are extracted for all patients via
  :func:`easyicu.api.load_concepts`;
* **existing EasyICU export package** (native ``_manifest.json`` or legacy
  ``easyicu_export_manifest.json`` with manifest-listed Parquet/CSV/XLSX
  members) -> concepts are read from disk, *no re-extraction* (the "user
  already has the data, just filter" path).

For every time-series concept it emits a wide per-stay summary
(``<c>_max/_min/_mean/_first/_n/_measured``) over a window, matching the
data-quality-gate input contract (see ``docs/qc_eligibility_gate_design_v1``),
plus timing columns ``<c>_first_time/_last_time`` carrying the ``charttime``
(hours from ICU admission) of the first/last recorded value. The timing
columns are what make exposure-timing questions answerable — e.g. the first
``charttime`` where ``norepi_rate`` is recorded is the vasopressor initiation
time, so "early vs delayed" exposure groups can be constructed from the wide
cohort without re-reading the raw event stream.
Each outcome is emitted as a whole-stay binary ``<outcome>`` and, when its
source carries a timestamp, an event time ``<outcome>_time`` (e.g.
``death_time`` = time-of-death in hours from ICU admission, NaN when the event
never occurred) so survival models and immortal-time guards are possible
instead of being blocked by a timeless binary outcome.
For every concept named by a CTAS cohort predicate it additionally emits a
bare ``<concept_id>`` column carrying the predicate's declared aggregation, so
:func:`easyicu.research_agent.cohort.schema.build_cohort` can apply the
inclusion/exclusion (纳排) deterministically and auditably.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import stat
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import uuid

import numpy as np
import pandas as pd

from .primitives import (
    ID_COL,
    TIME_COL,
    first_nonnull as _first_nonnull,
    merge_left as _merge_left,
    window as _window,
)
from .schema import CohortDefinition, build_cohort
from ..intake.export_package import (
    ExportPackageError,
    ExportPackage,
    is_export_package,
    open_export_package,
    read_exported_concept,
    require_canonical_time_projection,
    verify_export_package,
)
from ..intake.materialized_metadata import MaterializedColumnMetadataCollector
from ..intake.materialized_metadata import (
    MaterializedMetadataError,
    implementation_bundle_sha256,
    load_verified_materialized_cohort_authority,
    prepare_real_directory,
)
from ..intake.materialized_trajectory import (
    publish_materialized_trajectory_authority,
)
from easyicu.concept.metadata_projection import ConceptColumnRole

Window = Tuple[float, float]
_FALSE_TOKENS = {"", "0", "false", "f", "no", "n", "none", "nan", "na", "null", "off"}
_STRICT_EVENT_FALSE_TOKENS = {"0", "false", "f", "no", "n", "off"}
_STRICT_EVENT_TRUE_TOKENS = {"1", "true", "t", "yes", "y", "on"}


def _truthy_series(values: pd.Series) -> pd.Series:
    """Coerce mixed boolean-like concept values to conservative truth flags."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)

    numeric = pd.to_numeric(values, errors="coerce")
    normalized = values.astype("string").str.strip().str.lower()
    missing = values.isna() | normalized.isna()

    out = pd.Series(False, index=values.index, dtype=bool)
    numeric_known = numeric.notna()
    out.loc[numeric_known] = numeric.loc[numeric_known] != 0

    text_known = ~numeric_known & ~missing
    out.loc[text_known] = ~normalized.loc[text_known].isin(_FALSE_TOKENS)
    return out


def _strict_event_status_series(values: pd.Series, *, concept: str) -> pd.Series:
    """Decode a typed event status without laundering arbitrary values.

    The legacy path intentionally remains permissive.  A typed
    ``EVENT_STATUS`` binding, however, is an authority claim that the physical
    values are binary.  Accept only booleans, exact numeric 0/1, and canonical
    textual spellings of those two levels; preserve physical nulls as ``False``
    for the existing whole-stay/event-summary callers.
    """

    out = pd.Series(False, index=values.index, dtype=bool)
    for index, value in values.items():
        if pd.isna(value):
            continue
        if isinstance(value, (bool, np.bool_)):
            out.at[index] = bool(value)
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            if np.isfinite(numeric) and numeric in {0.0, 1.0}:
                out.at[index] = bool(int(numeric))
                continue
            raise MaterializedMetadataError(
                f"typed event concept {concept!r} contains a non-binary numeric value"
            )
        token = str(value).strip().lower()
        if token in _STRICT_EVENT_TRUE_TOKENS:
            out.at[index] = True
            continue
        if token in _STRICT_EVENT_FALSE_TOKENS:
            out.at[index] = False
            continue
        raise MaterializedMetadataError(
            f"typed event concept {concept!r} contains an unrecognised status value"
        )
    return out


def _require_finite_numeric(
    values: pd.Series,
    *,
    original: pd.Series,
    concept: str,
    purpose: str,
) -> pd.Series:
    """Fail closed on lossy or non-finite typed numeric conversion."""

    newly_invalid = original.notna() & values.isna()
    if bool(newly_invalid.any()):
        raise MaterializedMetadataError(
            f"typed value {purpose} {concept!r} has lossy numeric coercion"
        )
    nonfinite = values.notna() & ~np.isfinite(values.astype(float))
    if bool(nonfinite.any()):
        raise MaterializedMetadataError(
            f"typed value {purpose} {concept!r} contains a non-finite value"
        )
    return values


def _enforce_sealed_numeric_bounds(
    values: pd.Series,
    *,
    bounds: object,
    concept: str,
    label: str,
) -> None:
    """Fail closed when a range-preserving value leaves sealed bounds."""

    if bounds is None:
        return
    minimum = getattr(bounds, "minimum", None)
    maximum = getattr(bounds, "maximum", None)
    outside = pd.Series(False, index=values.index)
    if minimum is not None:
        outside |= values.notna() & (values < float(minimum))
    if maximum is not None:
        outside |= values.notna() & (values > float(maximum))
    if bool(outside.any()):
        raise MaterializedMetadataError(
            f"typed value {concept!r} is outside sealed {label}"
        )


def _outside_numeric_bounds(values: pd.Series, *, bounds: object) -> pd.Series:
    outside = pd.Series(False, index=values.index)
    if bounds is None:
        return outside
    minimum = getattr(bounds, "minimum", None)
    maximum = getattr(bounds, "maximum", None)
    if minimum is not None:
        outside |= values.notna() & (values < float(minimum))
    if maximum is not None:
        outside |= values.notna() & (values > float(maximum))
    return outside


def _bounded_typed_numeric(
    values: pd.Series,
    *,
    original: pd.Series,
    metadata: object,
    concept: str,
    purpose: str,
    bounds_violation_policy: str = "reject",
    bounds_violation_counts: Optional[dict[str, int]] = None,
) -> pd.Series:
    numeric = _require_finite_numeric(
        values,
        original=original,
        concept=concept,
        purpose=purpose,
    )
    bounds = getattr(metadata, "extraction_bounds", None)
    outside = _outside_numeric_bounds(numeric, bounds=bounds)
    if bounds_violation_policy not in {"reject", "exclude_with_receipt"}:
        raise MaterializedMetadataError("unsupported source bounds violation policy")
    if bool(outside.any()) and bounds_violation_policy == "exclude_with_receipt":
        if bounds_violation_counts is None:
            raise MaterializedMetadataError(
                "bounds exclusion policy requires a provenance receipt"
            )
        bounds_violation_counts[concept] = bounds_violation_counts.get(
            concept, 0
        ) + int(outside.sum())
        numeric = numeric.mask(outside)
    _enforce_sealed_numeric_bounds(
        numeric,
        bounds=bounds,
        concept=concept,
        label="extraction bounds",
    )
    return numeric


def _normalize_typed_output_domain(
    frame: pd.DataFrame,
    *,
    collector: MaterializedColumnMetadataCollector,
) -> tuple[pd.DataFrame, List[str]]:
    """Validate every typed output against its sealed physical-role contract."""

    event_columns: List[str] = []
    for column in collector.owned_columns:
        if column not in frame.columns:
            continue
        binding = collector.binding_for_output(column)
        if binding is None:  # pragma: no cover - owned_columns comes from bindings
            raise MaterializedMetadataError(
                f"typed output {column!r} lost its metadata binding"
            )
        metadata = binding.metadata
        role = metadata.role
        values = frame[column]
        if role is ConceptColumnRole.EVENT_STATUS:
            decoded = _strict_event_status_series(values, concept=column)
            frame[column] = decoded.astype(int)
            event_columns.append(column)
            continue
        if role is ConceptColumnRole.MEASUREMENT_STATUS:
            numeric = _require_finite_numeric(
                pd.to_numeric(values, errors="coerce"),
                original=values,
                concept=column,
                purpose="measurement-status output",
            ).fillna(0)
            if bool((~numeric.isin([0, 1])).any()):
                raise MaterializedMetadataError(
                    f"typed measurement-status output {column!r} is not binary"
                )
            frame[column] = numeric.astype(int)
            continue
        if role is ConceptColumnRole.EVENT_FRACTION:
            numeric = _require_finite_numeric(
                pd.to_numeric(values, errors="coerce"),
                original=values,
                concept=column,
                purpose="event-fraction output",
            ).fillna(0.0)
            if bool(((numeric < 0) | (numeric > 1)).any()):
                raise MaterializedMetadataError(
                    f"typed event-fraction output {column!r} is outside [0, 1]"
                )
            frame[column] = numeric.astype(float)
            event_columns.append(column)
            continue
        if role is ConceptColumnRole.COUNT:
            numeric = _require_finite_numeric(
                pd.to_numeric(values, errors="coerce"),
                original=values,
                concept=column,
                purpose="count output",
            ).fillna(0)
            if bool(((numeric < 0) | (numeric % 1 != 0)).any()):
                raise MaterializedMetadataError(
                    f"typed count output {column!r} is not a non-negative integer"
                )
            frame[column] = numeric.astype("int64")
            continue
        if role is ConceptColumnRole.NUMERIC_AGGREGATE:
            frame[column] = _bounded_typed_numeric(
                pd.to_numeric(values, errors="coerce"),
                original=values,
                metadata=metadata,
                concept=column,
                purpose="numeric aggregate output",
            )
            continue
        if role in {
            ConceptColumnRole.FIRST_OBSERVATION_TIME,
            ConceptColumnRole.LAST_OBSERVATION_TIME,
            ConceptColumnRole.EVENT_TIME,
        }:
            frame[column] = _require_finite_numeric(
                pd.to_numeric(values, errors="coerce"),
                original=values,
                concept=column,
                purpose="numeric output",
            )
            continue
        if role is ConceptColumnRole.VALUE and (
            metadata.canonical_unit is not None
            or metadata.extraction_bounds is not None
            or metadata.analysis_plausibility_range is not None
        ):
            frame[column] = _bounded_typed_numeric(
                pd.to_numeric(values, errors="coerce"),
                original=values,
                metadata=metadata,
                concept=column,
                purpose="physical value output",
            )
    return frame, event_columns


def _is_positive_only_boolean(series: pd.Series) -> bool:
    """True iff every non-NA value is boolean ``True`` (the positive level).

    Such a column is a sparse *event indicator*: the event is recorded as
    ``True`` and its absence is left as NA. A numeric column (e.g. a SOFA
    score) or a 0/1 column never matches, so the normalisation below cannot
    corrupt a measured variable.
    """
    nonnull = series.dropna()
    if nonnull.empty:
        return False
    return all(
        (v is True) or (isinstance(v, np.bool_) and bool(v)) for v in nonnull.unique()
    )


# Summary suffixes emitted by `_summarize_timeseries` for a time-series concept.
_EVENT_SUMMARY_SUFFIXES = ("_max", "_min", "_mean", "_first")
_SEMANTIC_PROVENANCE_KEYS = (
    "schema_version",
    "source_mode",
    "export_authority",
    "database",
    "cohort_window_hours",
    "feature_concepts",
    "outcome_concepts",
    "static_concepts",
    "cohort_definition",
    "n_stays_extracted",
    "n_stays_after_inclusion_exclusion",
    "unavailable_concepts",
    "event_indicator_columns_normalized",
    "declared_positive_only_event_concepts",
    "source_bounds_violation_policy",
    "source_bounds_exclusions",
    "columns",
    "cohort_sha256",
    "cohort_file_sha256",
    "cohort_file_size",
)


def _semantic_materialization_provenance(
    provenance: Mapping[str, object],
) -> dict[str, object]:
    missing = [key for key in _SEMANTIC_PROVENANCE_KEYS if key not in provenance]
    if missing:
        raise MaterializedMetadataError(
            "materialization provenance lacks semantic keys: " + ", ".join(missing)
        )
    return {key: provenance[key] for key in _SEMANTIC_PROVENANCE_KEYS}


def _normalize_event_indicator_columns(wide: pd.DataFrame) -> List[str]:
    """Decode sparse boolean event concepts so NA means "did not occur" (0).

    A concept like ``sep3_sofa2`` records only the positive event (``True``);
    a stay with no event has NA in ``<c>_max`` / ``<c>_first`` etc. Left as-is
    that NA reads as *measurement-missing* (e.g. an agent computing 66 %
    missingness and discarding the exposure), when it is really *structural
    absence* — the event did not happen. For any concept whose representative
    summary column is positive-only boolean, fill the negative level: integer
    summaries (max/min/first) become a clean 0/1 indicator and the mean
    becomes the within-window event fraction (0 when absent). Returns the list
    of normalised columns for provenance. Purely a representation decode — it
    never imputes a *measured* value (see ``_is_positive_only_boolean``).
    """
    normalized: List[str] = []
    bases: set[str] = set()
    for col in wide.columns:
        for suffix in _EVENT_SUMMARY_SUFFIXES:
            if col.endswith(suffix):
                bases.add(col[: -len(suffix)])
    for base in sorted(bases):
        probe = next(
            (
                f"{base}{s}"
                for s in ("_max", "_first", "_min")
                if f"{base}{s}" in wide.columns
            ),
            None,
        )
        if probe is None or not _is_positive_only_boolean(wide[probe]):
            continue
        for suffix in _EVENT_SUMMARY_SUFFIXES:
            col = f"{base}{suffix}"
            if col not in wide.columns:
                continue
            if suffix == "_mean":
                wide[col] = pd.to_numeric(wide[col], errors="coerce").fillna(0.0)
            else:
                # ``== True`` on a pandas *nullable* boolean/Int column keeps
                # <NA> in the result, which ``astype(int)`` rejects ("cannot
                # convert NA to integer"). Absence is the negative level for a
                # positive-only event indicator (docstring: "0 when absent"), so
                # fill the NA before the cast. Harmless no-op on numpy dtypes,
                # where NaN/None already compare False.
                wide[col] = (wide[col] == True).fillna(False).astype(int)  # noqa: E712
            normalized.append(col)
    return normalized


def _normalize_declared_positive_only_event_concepts(
    wide: pd.DataFrame,
    *,
    concepts: Sequence[str],
) -> list[str]:
    """Decode host-declared sparse positive event summaries to explicit 0/1."""

    normalized: list[str] = []
    for concept in concepts:
        summary_columns = [
            f"{concept}{suffix}"
            for suffix in _EVENT_SUMMARY_SUFFIXES
            if f"{concept}{suffix}" in wide.columns
        ]
        if not summary_columns:
            raise MaterializedMetadataError(
                f"declared positive-only event {concept!r} has no summary columns"
            )
        for column in summary_columns:
            numeric = pd.to_numeric(wide[column], errors="coerce")
            invalid = wide[column].notna() & numeric.isna()
            nonbinary = numeric.notna() & ~numeric.isin([0.0, 1.0])
            if bool(invalid.any()) or bool(nonbinary.any()):
                raise MaterializedMetadataError(
                    f"declared positive-only event {concept!r} is not binary"
                )
            wide[column] = numeric.fillna(0.0)
            normalized.append(column)
    return normalized


def _any_truthy(values: pd.Series) -> bool:
    return bool(_truthy_series(values).any())


def _all_truthy(values: pd.Series) -> bool:
    truth = _truthy_series(values.dropna())
    return bool(len(truth) > 0 and truth.all())


_AGG_FUNCS = {
    "max": lambda s: s.max(),
    "min": lambda s: s.min(),
    "mean": lambda s: s.mean(),
    "median": lambda s: s.median(),
    "sum": lambda s: s.sum(),
    "count": lambda s: s.count(),
    "first": _first_nonnull,
    "last": lambda s: _first_nonnull(s.iloc[::-1]),
    "any": _any_truthy,
    "all": _all_truthy,
}


def _coerce_int_stay(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or ID_COL not in df.columns:
        return df

    def parse(value: object) -> int:
        if isinstance(value, (bool, np.bool_)) or pd.isna(value):
            raise MaterializedMetadataError("stay identity must be an exact integer")
        if isinstance(value, (int, np.integer)):
            parsed = int(value)
        elif isinstance(value, str):
            if re.fullmatch(r"-?(0|[1-9][0-9]*)", value) is None or value == "-0":
                raise MaterializedMetadataError(
                    "stay identity string must be a canonical integer"
                )
            parsed = int(value)
        elif isinstance(value, (float, np.floating)):
            numeric = float(value)
            if (
                not np.isfinite(numeric)
                or not numeric.is_integer()
                or abs(numeric) >= 2**53
            ):
                raise MaterializedMetadataError(
                    "floating stay identity is not exactly representable"
                )
            parsed = int(numeric)
        else:
            raise MaterializedMetadataError("stay identity must be an exact integer")
        if parsed < np.iinfo(np.int64).min or parsed > np.iinfo(np.int64).max:
            raise MaterializedMetadataError("stay identity exceeds int64 bounds")
        return parsed

    out = df.copy()
    out[ID_COL] = pd.Series(
        (parse(value) for value in out[ID_COL].tolist()),
        index=out.index,
        dtype="int64",
    )
    return out


def _is_export_dir(path: Path) -> bool:
    return is_export_package(path)


def _resolve_source(
    data_path: Union[str, Path], prefer_existing: bool
) -> Tuple[str, Path]:
    root = Path(data_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"data_path does not exist: {root}")
    if _is_export_dir(root):
        return "export", root
    return "converted", root


def _timing_columns(w: pd.DataFrame, concept: str) -> pd.DataFrame:
    """Per-stay ``<c>_first_time`` / ``<c>_last_time`` for one concept.

    The time index (``charttime``, hours from ICU admission) of the FIRST and
    LAST *recorded* (non-null) value of ``concept`` inside the window. Computed
    on the raw column **before** any presence-coercion, so a categorical event's
    ``_first_time`` is its true onset, not the window start.

    This is what makes timing-dependent questions answerable: e.g. the first
    ``charttime`` where ``norepi_rate`` is recorded IS the vasopressor
    initiation time, so "early vs delayed" exposure can be constructed. Without
    it the wide summary only exposes magnitude (``_max/_min/_mean/_first``) and
    an agent wrongly concludes no row-level timing exists and BLOCKs the study.

    A stay with no recorded value is absent here -> the column is NaN after the
    left-merge, which honestly reads as "never measured / event never occurred",
    i.e. no onset time.
    """
    if TIME_COL not in w.columns:
        return pd.DataFrame(columns=[ID_COL])
    recorded = w.loc[w[concept].notna(), [ID_COL, TIME_COL]]
    if recorded.empty:
        return pd.DataFrame(columns=[ID_COL])
    recorded = recorded.sort_values([ID_COL, TIME_COL])
    g = recorded.groupby(ID_COL)[TIME_COL]
    first_t = g.first()
    last_t = g.last()
    return pd.DataFrame(
        {
            ID_COL: first_t.index,
            f"{concept}_first_time": first_t.to_numpy(),
            f"{concept}_last_time": last_t.to_numpy(),
        }
    )


def _load_concept(
    source_mode: str,
    root: Union[Path, ExportPackage],
    concept: str,
    database: str,
    patient_ids: Optional[Sequence[int]],
    unavailable: List[str],
) -> pd.DataFrame:
    """Load one concept from an export package or a converted database.

    Shared by the wide-summary path and the long-trajectory path so they read
    the source identically. Appends to ``unavailable`` and returns an empty
    stay-keyed frame when the concept is absent (fail-soft); re-raises other
    errors with concept context.
    """
    try:
        if source_mode == "export":
            if isinstance(root, ExportPackage):
                require_canonical_time_projection(root, concept)
            return _coerce_int_stay(read_exported_concept(root, concept))
        from ...api import load_concepts  # local import: heavy module

        return _coerce_int_stay(
            load_concepts(
                [concept],
                database=database,
                data_path=str(root),
                patient_ids=patient_ids,
            )
        )
    except ExportPackageError:
        raise
    except KeyError:
        unavailable.append(concept)
        return pd.DataFrame(columns=[ID_COL])
    except Exception as exc:  # noqa: BLE001 - add concept context before failing closed
        raise RuntimeError(f"failed to load concept {concept!r}") from exc


def _export_authority_provenance(
    package: Optional[ExportPackage],
) -> Optional[Dict[str, Any]]:
    if package is None:
        return None
    return {
        "manifest": package.manifest_path.name,
        "manifest_kind": package.manifest_kind,
        # This is a source-manifest fact, already covered by manifest_sha256.  It
        # is copied into materialization provenance so a paper-facing gate can
        # distinguish an official typed export from a structural retrofit without
        # reopening a mutable export directory.
        "seal_kind": package.source_seal_kind,
        "manifest_sha256": package.manifest_sha256,
        "authority_sha256": package.authority_sha256,
        "export_format": package.export_format,
        "feature_definitions_sha256": package.feature_definitions_sha256,
        "missing_selected_concepts": list(package.missing_selected_concepts),
        "files": [
            {
                "relative_path": item.relative_path,
                "sha256": item.identity.sha256,
                "rows": item.rows,
                "id_column": item.id_column,
                "time_column": item.time_column,
                "time_columns": list(item.time_columns),
            }
            for item in package.files
        ],
    }


def _summarize_timeseries(
    df: pd.DataFrame, concept: str, window: Window
) -> pd.DataFrame:
    """Per-stay summary columns for one time-series concept over ``window``."""
    out, _presence_encoded = _summarize_timeseries_with_representation(
        df, concept, window
    )
    return out


def _summarize_timeseries_with_representation(
    df: pd.DataFrame,
    concept: str,
    window: Window,
    *,
    source_role: Optional[ConceptColumnRole] = None,
) -> tuple[pd.DataFrame, bool]:
    """Return the summary plus whether values were encoded as event presence."""

    w = _window(df, window[0], window[1])
    if w.empty or concept not in w.columns:
        return pd.DataFrame(columns=[ID_COL]), False
    # Timing (onset/last-record time) is taken from the RAW non-null values
    # before the presence-coercion below, so a categorical event keeps its true
    # onset charttime rather than the window start.
    timing = _timing_columns(w, concept)
    # The wide summary emits numeric _max/_min/_mean. A concept stored as object
    # (e.g. a ventilation status or a vasopressor drug name) cannot be reduced
    # with max/mean and would raise. Coerce: if any value parses as a number the
    # concept is numeric-stored-as-text (use the numeric view; unparseable rows
    # honestly become NaN/measurement-missing); otherwise it is a categorical /
    # event concept and we summarise its PRESENCE (1 = recorded in window), so
    # `_max` reads as "ever" and `_mean` as the within-window event fraction.
    col = w[concept]
    if source_role is ConceptColumnRole.EVENT_STATUS:
        encoded = pd.Series(np.nan, index=col.index, dtype=float)
        nonnull = col.notna()
        encoded.loc[nonnull] = _strict_event_status_series(
            col.loc[nonnull], concept=concept
        ).astype(float)
        w = w.assign(**{concept: encoded})
        presence_encoded = True
    elif source_role is ConceptColumnRole.VALUE:
        if pd.api.types.is_bool_dtype(col):
            raise MaterializedMetadataError(
                f"typed value concept {concept!r} cannot be summarized as a boolean"
            )
        numeric = pd.to_numeric(col, errors="coerce")
        nonnull = col.notna()
        if bool((numeric.notna() == nonnull).all()):
            numeric = _require_finite_numeric(
                numeric,
                original=col,
                concept=concept,
                purpose="concept",
            )
            w = w.assign(**{concept: numeric})
            presence_encoded = False
        elif not bool(numeric.notna().any()):
            # A typed VALUE may legitimately be categorical (for example
            # invasive/noninvasive ventilation).  The wide materializer owns a
            # numeric summary contract, so encode recorded presence while the
            # raw typed trajectory remains available when category detail is
            # scientifically required.
            w = w.assign(**{concept: _truthy_series(col).astype(float)})
            presence_encoded = True
        else:
            raise MaterializedMetadataError(
                f"typed value concept {concept!r} mixes numeric and categorical values"
            )
    else:
        presence_encoded = bool(pd.api.types.is_bool_dtype(col))
    if (
        source_role is None
        and not pd.api.types.is_numeric_dtype(col)
        and not pd.api.types.is_bool_dtype(col)
    ):
        numeric = pd.to_numeric(col, errors="coerce")
        if numeric.notna().any():
            w = w.assign(**{concept: numeric})
        else:
            w = w.assign(**{concept: _truthy_series(col).astype(float)})
            presence_encoded = True
    grp = w.groupby(ID_COL)[concept]
    out = grp.agg(["max", "min", "mean", "count"]).reset_index()
    out.columns = [
        ID_COL,
        f"{concept}_max",
        f"{concept}_min",
        f"{concept}_mean",
        f"{concept}_n",
    ]
    first = (
        w.sort_values([ID_COL, TIME_COL])
        .groupby(ID_COL)[concept]
        .apply(_first_nonnull)
        .reset_index()
    )
    first.columns = [ID_COL, f"{concept}_first"]
    out = out.merge(first, on=ID_COL, how="left")
    out[f"{concept}_measured"] = (out[f"{concept}_n"].fillna(0) > 0).astype(int)
    if not timing.empty:
        out = out.merge(timing, on=ID_COL, how="left")
    return out, presence_encoded


def _predicate_column(
    df: pd.DataFrame,
    concept: str,
    window: Window,
    aggregation: str,
    *,
    source_role: Optional[ConceptColumnRole] = None,
) -> pd.DataFrame:
    """A bare ``<concept>`` column carrying the CTAS-declared aggregation."""
    if TIME_COL not in df.columns:
        # static concept: one value per stay
        cols = [ID_COL, concept] if concept in df.columns else [ID_COL]
        return df[cols].drop_duplicates(ID_COL).copy()
    w = _window(df, window[0], window[1])
    if w.empty or concept not in w.columns:
        return pd.DataFrame(columns=[ID_COL, concept])
    agg_key = (aggregation or "").lower()
    if agg_key not in _AGG_FUNCS:
        raise ValueError(f"unsupported cohort predicate aggregation: {aggregation!r}")
    fn = _AGG_FUNCS[agg_key]
    if source_role is ConceptColumnRole.VALUE:
        if agg_key in {"any", "all"}:
            raise MaterializedMetadataError(
                f"typed value predicate {concept!r} cannot use {agg_key!r}"
            )
        if agg_key != "count":
            if pd.api.types.is_bool_dtype(w[concept]):
                raise MaterializedMetadataError(
                    f"typed value predicate {concept!r} cannot aggregate a boolean"
                )
            numeric = _require_finite_numeric(
                pd.to_numeric(w[concept], errors="coerce"),
                original=w[concept],
                concept=concept,
                purpose="predicate",
            )
            w = w.assign(**{concept: numeric})
    elif source_role is ConceptColumnRole.EVENT_STATUS:
        if agg_key in {"median", "sum"}:
            raise MaterializedMetadataError(
                f"typed event predicate {concept!r} cannot use {agg_key!r}"
            )
        if agg_key != "count":
            encoded = pd.Series(np.nan, index=w.index, dtype=float)
            nonnull = w[concept].notna()
            encoded.loc[nonnull] = _strict_event_status_series(
                w.loc[nonnull, concept], concept=concept
            ).astype(float)
            w = w.assign(**{concept: encoded})
    if TIME_COL in w.columns:
        w = w.sort_values([ID_COL, TIME_COL])
    out = w.groupby(ID_COL)[concept].apply(fn).reset_index()
    out.columns = [ID_COL, concept]
    return out


def _static_column(
    df: pd.DataFrame,
    concept: str,
    *,
    source_role: Optional[ConceptColumnRole] = None,
) -> pd.DataFrame:
    cols = [ID_COL, concept] if concept in df.columns else [ID_COL]
    selected = df[cols].copy()
    if source_role is None or concept not in selected.columns:
        return selected.drop_duplicates(ID_COL).copy()
    conflicts = selected.groupby(ID_COL, dropna=True)[concept].nunique(dropna=True)
    if bool((conflicts > 1).any()):
        raise MaterializedMetadataError(
            f"typed static concept {concept!r} has conflicting stay-level values"
        )
    return (
        selected.groupby(ID_COL, dropna=True)[concept]
        .apply(_first_nonnull)
        .reset_index()
    )


def _binary_event_column(
    df: pd.DataFrame,
    concept: str,
    *,
    source_role: Optional[ConceptColumnRole] = None,
) -> pd.DataFrame:
    """Whole-stay binary: 1 if the stay has any event for ``concept`` (e.g. death)."""
    if ID_COL not in df.columns:
        return pd.DataFrame(columns=[ID_COL, concept])
    if concept not in df.columns:
        if source_role is not None:
            raise MaterializedMetadataError(
                f"typed outcome source {concept!r} is missing its physical column"
            )
        ids = pd.Series(df[ID_COL].dropna().unique(), name=ID_COL)
        return pd.DataFrame({ID_COL: ids, concept: 1})

    if source_role is not None and source_role is not ConceptColumnRole.EVENT_STATUS:
        raise MaterializedMetadataError(
            f"typed outcome {concept!r} is not authorized as an event status"
        )

    work = df[[ID_COL, concept]].dropna(subset=[ID_COL]).copy()
    if work.empty:
        return pd.DataFrame(columns=[ID_COL, concept])
    event = (
        _strict_event_status_series(work[concept], concept=concept)
        if source_role is ConceptColumnRole.EVENT_STATUS
        else _truthy_series(work[concept])
    )
    out = (
        pd.DataFrame({ID_COL: work[ID_COL], concept: event.astype(int)})
        .groupby(ID_COL, dropna=True)[concept]
        .max()
        .reset_index()
    )
    return out


def _event_time_column(
    df: pd.DataFrame,
    concept: str,
    *,
    source_role: Optional[ConceptColumnRole] = None,
) -> pd.DataFrame:
    """Per-stay ``<concept>_time``: the ``charttime`` of the event itself.

    ``_binary_event_column`` collapses an outcome to a whole-stay 0/1 and drops
    its time index. For an event concept whose source carries a timestamp (e.g.
    ``death`` is indexed by ``deathtime``), that time IS the time-of-event in
    hours from ICU admission. Surfacing it as ``<concept>_time`` (NaN when the
    event never occurred) is what lets a downstream analysis guard against
    immortal-time bias or fit a survival model — without it an agent sees only a
    binary outcome and must block any timing-aware effect estimate.

    Symmetric to ``_timing_columns`` for features. Returns an empty frame when
    the source has no usable time index (purely stay-level derived flags).
    """
    if source_role is not None and source_role is not ConceptColumnRole.EVENT_STATUS:
        raise MaterializedMetadataError(
            f"typed outcome {concept!r} cannot produce an event time"
        )
    if (
        TIME_COL not in df.columns
        or concept not in df.columns
        or ID_COL not in df.columns
    ):
        return pd.DataFrame(columns=[ID_COL])
    work = df[[ID_COL, TIME_COL, concept]].dropna(subset=[ID_COL]).copy()
    if work.empty:
        return pd.DataFrame(columns=[ID_COL])
    event = (
        _strict_event_status_series(work[concept], concept=concept)
        if source_role is ConceptColumnRole.EVENT_STATUS
        else _truthy_series(work[concept])
    )
    work = work[event & work[TIME_COL].notna()]
    if work.empty:
        return pd.DataFrame(columns=[ID_COL])
    out = (
        work.sort_values([ID_COL, TIME_COL])
        .groupby(ID_COL)[TIME_COL]
        .first()
        .reset_index()
    )
    out.columns = [ID_COL, f"{concept}_time"]
    return out


def _hash_df(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _canonical_stem(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or Path(value).name != value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise MaterializedMetadataError(
            "materialization stem must be one path component"
        )
    return value


def _atomic_write_provenance(
    path: Path,
    payload: Mapping[str, object],
    *,
    canonical: bool,
) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    raw = json.dumps(
        dict(payload),
        indent=2,
        ensure_ascii=False,
        sort_keys=canonical,
        allow_nan=not canonical,
    ).encode("utf-8")
    fd: Optional[int] = None
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write while publishing cohort provenance")
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd is not None:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def materialize_cohort(
    *,
    feature_concepts: Sequence[str],
    database: str = "miiv",
    data_path: Union[str, Path],
    cohort_definition: Optional[CohortDefinition] = None,
    cohort_window: Window = (0.0, 24.0),
    outcome_concepts: Sequence[str] = ("death",),
    static_concepts: Sequence[str] = ("age", "sex", "los_icu"),
    patient_ids: Optional[Sequence[int]] = None,
    prefer_existing: bool = True,
    bounds_violation_policy: str = "reject",
    positive_only_event_concepts: Sequence[str] = (),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build a per-stay analysis cohort from the EasyICU data layer.

    Returns ``(cohort_df, provenance)``. ``cohort_df`` is one row per ICU stay
    after applying ``cohort_definition`` (纳入排除); ``provenance`` records the
    source mode, concept list, window, attrition counts and hashes for audit.
    """
    cohort, provenance, _collector = _materialize_cohort_with_metadata(
        feature_concepts=feature_concepts,
        database=database,
        data_path=data_path,
        cohort_definition=cohort_definition,
        cohort_window=cohort_window,
        outcome_concepts=outcome_concepts,
        static_concepts=static_concepts,
        patient_ids=patient_ids,
        prefer_existing=prefer_existing,
        bounds_violation_policy=bounds_violation_policy,
        positive_only_event_concepts=positive_only_event_concepts,
    )
    return cohort, provenance


def _materialize_cohort_with_metadata(
    *,
    feature_concepts: Sequence[str],
    database: str,
    data_path: Union[str, Path],
    cohort_definition: Optional[CohortDefinition],
    cohort_window: Window,
    outcome_concepts: Sequence[str],
    static_concepts: Sequence[str],
    patient_ids: Optional[Sequence[int]],
    prefer_existing: bool,
    bounds_violation_policy: str,
    positive_only_event_concepts: Sequence[str],
) -> tuple[pd.DataFrame, Dict[str, Any], MaterializedColumnMetadataCollector]:
    t0 = time.time()
    source_mode, root = _resolve_source(data_path, prefer_existing)
    common = dict(
        feature_concepts=feature_concepts,
        database=database,
        cohort_definition=cohort_definition,
        cohort_window=cohort_window,
        outcome_concepts=outcome_concepts,
        static_concepts=static_concepts,
        patient_ids=patient_ids,
        source_mode=source_mode,
        root=root,
        t0=t0,
        bounds_violation_policy=bounds_violation_policy,
        positive_only_event_concepts=positive_only_event_concepts,
    )
    if source_mode == "export" and is_export_package(root):
        with open_export_package(root) as export_package:
            return _materialize_cohort_from_resolved_source(
                export_package=export_package,
                **common,
            )
    return _materialize_cohort_from_resolved_source(
        export_package=None,
        **common,
    )


def _materialize_cohort_from_resolved_source(
    *,
    feature_concepts: Sequence[str],
    database: str,
    cohort_definition: Optional[CohortDefinition],
    cohort_window: Window,
    outcome_concepts: Sequence[str],
    static_concepts: Sequence[str],
    patient_ids: Optional[Sequence[int]],
    source_mode: str,
    root: Path,
    export_package: Optional[ExportPackage],
    t0: float,
    bounds_violation_policy: str,
    positive_only_event_concepts: Sequence[str],
    verify_source_package: bool = True,
) -> tuple[pd.DataFrame, Dict[str, Any], MaterializedColumnMetadataCollector]:
    """Materialize from one already-resolved, explicitly owned source."""

    if (
        export_package is not None
        and export_package.database
        and export_package.database != database
    ):
        raise ExportPackageError(
            "requested database does not match export package authority",
            code="export_package_database_mismatch",
            manifest_path=export_package.manifest_path,
        )
    source_handle: Union[Path, ExportPackage] = export_package or root
    metadata_collector = MaterializedColumnMetadataCollector(export_package)
    if bounds_violation_policy not in {"reject", "exclude_with_receipt"}:
        raise MaterializedMetadataError("unsupported source bounds violation policy")
    bounds_violation_counts: dict[str, int] = {}

    unavailable: List[str] = []

    def load(concept: str) -> pd.DataFrame:
        loaded = _load_concept(
            source_mode, source_handle, concept, database, patient_ids, unavailable
        )
        if not metadata_collector.enabled or loaded.empty:
            return loaded
        loaded = loaded.copy()
        if TIME_COL in loaded.columns:
            loaded[TIME_COL] = _require_finite_numeric(
                pd.to_numeric(loaded[TIME_COL], errors="coerce"),
                original=loaded[TIME_COL],
                concept=concept,
                purpose="source time coordinate",
            )
        source_binding = metadata_collector.source_binding(concept)
        if source_binding is None:
            raise MaterializedMetadataError(
                f"typed materialization concept {concept!r} lost its source binding"
            )
        source_metadata = source_binding.metadata
        if source_metadata.role is ConceptColumnRole.EVENT_STATUS:
            # Validate every consumed status before any aggregation, but keep
            # physical nulls intact so ``count`` remains a measurement count.
            _strict_event_status_series(loaded[concept], concept=concept)
        elif source_metadata.role is ConceptColumnRole.VALUE and (
            source_metadata.canonical_unit is not None
            or source_metadata.extraction_bounds is not None
            or source_metadata.analysis_plausibility_range is not None
        ):
            loaded[concept] = _bounded_typed_numeric(
                pd.to_numeric(loaded[concept], errors="coerce"),
                original=loaded[concept],
                metadata=source_metadata,
                concept=concept,
                purpose="source physical value",
                bounds_violation_policy=bounds_violation_policy,
                bounds_violation_counts=bounds_violation_counts,
            )
        return loaded

    # concepts required by the CTAS cohort predicates (for 纳排)
    pred_specs: List[Tuple[str, Window, str, str]] = []
    if cohort_definition is not None:
        for pred in (*cohort_definition.inclusion, *cohort_definition.exclusion):
            tw = getattr(pred, "time_window", None)
            win: Window = (
                (float(tw.start_offset_hours), float(tw.end_offset_hours))
                if tw is not None
                else cohort_window
            )
            pred_specs.append(
                (
                    pred.concept_id,
                    win,
                    getattr(pred, "aggregation", "max"),
                    str(getattr(tw, "anchor", "icu_admission")),
                )
            )
    if metadata_collector.enabled:
        by_concept: dict[str, tuple[Window, str, str]] = {}
        for concept, window, aggregation, anchor in pred_specs:
            spec = (window, str(aggregation), anchor)
            previous = by_concept.get(concept)
            if previous is not None and previous != spec:
                raise MaterializedMetadataError(
                    f"typed predicate {concept!r} has multiple incompatible derivations"
                )
            by_concept[concept] = spec
        bounded_outcome_predicates = sorted(
            concept for concept in by_concept if concept in set(outcome_concepts)
        )
        if bounded_outcome_predicates:
            raise MaterializedMetadataError(
                "typed timed predicates cannot reuse whole-stay outcome columns: "
                + ", ".join(bounded_outcome_predicates)
            )

    static_set = list(dict.fromkeys(static_concepts))
    outcome_set = list(dict.fromkeys(outcome_concepts))
    feature_set = [c for c in dict.fromkeys(feature_concepts) if c not in static_set]
    declared_positive_only = tuple(positive_only_event_concepts)
    if len(declared_positive_only) != len(set(declared_positive_only)) or any(
        not isinstance(concept, str)
        or not concept
        or concept != concept.strip()
        or concept not in feature_set
        for concept in declared_positive_only
    ):
        raise MaterializedMetadataError(
            "positive-only event concepts must be unique materialized features"
        )

    # ---- base = every ICU stay (denominator); take from the first static concept
    base: Optional[pd.DataFrame] = None
    static_frames: List[pd.DataFrame] = []
    for c in static_set:
        df = load(c)
        source_role = metadata_collector.require_source_role(c)
        static_frame = _static_column(df, c, source_role=source_role)
        static_frames.append(static_frame)
        metadata_collector.add_static(c, output_columns=static_frame.columns)
        if base is None and ID_COL in df.columns:
            base = df[[ID_COL]].drop_duplicates().copy()
    if base is None:
        raise RuntimeError("Could not establish a stay-level base from static_concepts")

    frames: List[pd.DataFrame] = [*static_frames]

    # ---- time-series features -> wide per-stay summaries (over cohort_window)
    for c in feature_set:
        df = load(c)
        source_role = metadata_collector.require_source_role(c)
        if TIME_COL in df.columns:
            summary, _presence_encoded = _summarize_timeseries_with_representation(
                df,
                c,
                cohort_window,
                source_role=source_role,
            )
            frames.append(summary)
            metadata_collector.add_timeseries(
                c,
                output_columns=summary.columns,
                window=cohort_window,
            )
        else:
            static_frame = _static_column(df, c, source_role=source_role)
            frames.append(static_frame)
            metadata_collector.add_static(c, output_columns=static_frame.columns)

    # ---- outcomes -> whole-stay binary (a death after 24h still counts), plus
    # the event time (<c>_time, e.g. death_time = time-of-death hours from ICU
    # admission) when the source carries a timestamp, so timing-aware analyses
    # (immortal-time guards, survival models) are possible.
    for c in outcome_set:
        loaded = load(c)
        source_role = metadata_collector.require_source_role(c)
        event_column = _binary_event_column(
            loaded,
            c,
            source_role=source_role,
        )
        frames.append(event_column)
        event_time = _event_time_column(
            loaded,
            c,
            source_role=source_role,
        )
        if not event_time.empty:
            frames.append(event_time)
        metadata_collector.add_outcome(
            c,
            output_columns=tuple(event_column.columns) + tuple(event_time.columns),
        )

    # ---- bare predicate columns for 纳排 (skip concepts already materialised bare)
    produced_bare = set(static_set) | set(outcome_set)
    for concept, win, agg, anchor in pred_specs:
        if concept in produced_bare:
            continue
        loaded = load(concept)
        source_role = metadata_collector.require_source_role(concept)
        predicate_frame = _predicate_column(
            loaded,
            concept,
            win,
            agg,
            source_role=source_role,
        )
        frames.append(predicate_frame)
        metadata_collector.add_predicate(
            concept,
            output_columns=predicate_frame.columns,
            source_has_time=TIME_COL in loaded.columns,
            aggregation=agg,
            window=win,
            anchor=anchor,
        )
        produced_bare.add(concept)

    wide = _merge_left(base, frames)
    if metadata_collector.enabled:
        wide, event_indicator_columns = _normalize_typed_output_domain(
            wide,
            collector=metadata_collector,
        )
    else:
        for c in outcome_set:
            if c in wide.columns:
                wide[c] = wide[c].fillna(0).astype(int)
        # A stay absent from a sparse concept has zero measurements in the
        # legacy representation. Typed-v2 applies this only to owned columns.
        for col in wide.columns:
            if col.endswith("_n") or col.endswith("_measured"):
                wide[col] = wide[col].fillna(0)
                if col.endswith("_measured"):
                    wide[col] = wide[col].astype(int)
        event_indicator_columns = _normalize_event_indicator_columns(wide)
    event_indicator_columns = list(
        dict.fromkeys(
            [
                *event_indicator_columns,
                *_normalize_declared_positive_only_event_concepts(
                    wide,
                    concepts=declared_positive_only,
                ),
            ]
        )
    )
    n_all = int(len(wide))

    # ---- apply CTAS inclusion/exclusion (纳排), deterministic + auditable
    if cohort_definition is not None:
        cohort = build_cohort(cohort_definition, wide)
    else:
        cohort = wide
    n_after = int(len(cohort))
    if export_package is not None and verify_source_package:
        verify_export_package(export_package)

    provenance = {
        "schema_version": "easyicu.cohort_materializer/1",
        "source_mode": source_mode,
        "source": str(root),
        "export_authority": _export_authority_provenance(export_package),
        "database": database,
        "cohort_window_hours": list(cohort_window),
        "feature_concepts": list(feature_concepts),
        "outcome_concepts": list(outcome_concepts),
        "static_concepts": static_set,
        "cohort_definition": cohort_definition.to_dict() if cohort_definition else None,
        "n_stays_extracted": n_all,
        "n_stays_after_inclusion_exclusion": n_after,
        "unavailable_concepts": unavailable,
        "event_indicator_columns_normalized": event_indicator_columns,
        "declared_positive_only_event_concepts": list(declared_positive_only),
        "source_bounds_violation_policy": bounds_violation_policy,
        "source_bounds_exclusions": dict(sorted(bounds_violation_counts.items())),
        "columns": list(cohort.columns),
        "cohort_sha256": _hash_df(cohort.reset_index(drop=True)),
        "build_seconds": round(time.time() - t0, 2),
    }
    return cohort.reset_index(drop=True), provenance, metadata_collector


def build_trajectory_long(
    *,
    data_path: Union[str, Path],
    concepts: Sequence[str],
    database: str = "miiv",
    window: Optional[Window] = None,
    patient_ids: Optional[Sequence[int]] = None,
    prefer_existing: bool = True,
    bounds_violation_policy: str = "reject",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Long-format trajectory ``(stay_id, charttime, concept, value_num, value_str)``.

    The wide per-stay summary in :func:`materialize_cohort` decides the temporal
    aggregation (max/min/first over a fixed window) up front, BEFORE the agent
    sees the data — which is lossy for any question that needs a threshold-
    crossing onset (first time MAP < 65), an incident-after-exposure endpoint
    (first AKI after first PEEP), or a time-varying exposure / landmark design.
    This emits the per-timepoint series for the named concepts so the agent can
    construct those temporal features itself in-sandbox, instead of being forced
    through the baseline-summary lens.

    Only rows with a recorded (non-null) value are kept. ``value_num`` is the
    numeric view (NaN when the concept is categorical/unparseable); ``value_str``
    preserves the raw value. ``window`` (hours from ICU admission) bounds the
    series; ``None`` keeps the full available trajectory.
    """
    source_mode, root = _resolve_source(data_path, prefer_existing)
    common = dict(
        concepts=concepts,
        database=database,
        window=window,
        patient_ids=patient_ids,
        source_mode=source_mode,
        root=root,
        bounds_violation_policy=bounds_violation_policy,
    )
    if source_mode == "export" and is_export_package(root):
        with open_export_package(root) as export_package:
            return _build_trajectory_long_from_resolved_source(
                export_package=export_package,
                **common,
            )
    return _build_trajectory_long_from_resolved_source(
        export_package=None,
        **common,
    )


def _build_trajectory_long_from_resolved_source(
    *,
    concepts: Sequence[str],
    database: str,
    window: Optional[Window],
    patient_ids: Optional[Sequence[int]],
    source_mode: str,
    root: Path,
    export_package: Optional[ExportPackage],
    bounds_violation_policy: str,
    verify_source_package: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build trajectory rows from one already-resolved, explicitly owned source."""

    if (
        export_package is not None
        and export_package.database
        and export_package.database != database
    ):
        raise ExportPackageError(
            "requested database does not match export package authority",
            code="export_package_database_mismatch",
            manifest_path=export_package.manifest_path,
        )
    source_handle: Union[Path, ExportPackage] = export_package or root
    metadata_collector = MaterializedColumnMetadataCollector(export_package)
    if bounds_violation_policy not in {"reject", "exclude_with_receipt"}:
        raise MaterializedMetadataError("unsupported source bounds violation policy")
    bounds_violation_counts: dict[str, int] = {}
    unavailable: List[str] = []
    available_unobserved: List[str] = []
    frames: List[pd.DataFrame] = []
    materialized: List[str] = []
    for concept in dict.fromkeys(concepts):
        df = _load_concept(
            source_mode,
            source_handle,
            concept,
            database,
            patient_ids,
            unavailable,
        )
        if metadata_collector.enabled and not df.empty:
            df = df.copy()
            if TIME_COL in df.columns:
                df[TIME_COL] = _require_finite_numeric(
                    pd.to_numeric(df[TIME_COL], errors="coerce"),
                    original=df[TIME_COL],
                    concept=concept,
                    purpose="source time coordinate",
                )
            source_binding = metadata_collector.source_binding(concept)
            if source_binding is None:
                raise MaterializedMetadataError(
                    f"typed trajectory concept {concept!r} lost its source binding"
                )
            source_metadata = source_binding.metadata
            if source_metadata.role is ConceptColumnRole.EVENT_STATUS:
                _strict_event_status_series(df[concept], concept=concept)
            elif source_metadata.role is ConceptColumnRole.VALUE and (
                source_metadata.canonical_unit is not None
                or source_metadata.extraction_bounds is not None
                or source_metadata.analysis_plausibility_range is not None
            ):
                df[concept] = _bounded_typed_numeric(
                    pd.to_numeric(df[concept], errors="coerce"),
                    original=df[concept],
                    metadata=source_metadata,
                    concept=concept,
                    purpose="trajectory source physical value",
                    bounds_violation_policy=bounds_violation_policy,
                    bounds_violation_counts=bounds_violation_counts,
                )
        if TIME_COL not in df.columns or concept not in df.columns:
            if concept not in unavailable:
                unavailable.append(concept)
            continue
        w = _window(df, window[0], window[1]) if window is not None else df
        sub = w.loc[w[concept].notna(), [ID_COL, TIME_COL, concept]]
        if sub.empty:
            available_unobserved.append(concept)
            continue
        frames.append(
            pd.DataFrame(
                {
                    ID_COL: sub[ID_COL].to_numpy(),
                    TIME_COL: sub[TIME_COL].to_numpy(),
                    "concept": concept,
                    "value_num": pd.to_numeric(
                        sub[concept], errors="coerce"
                    ).to_numpy(),
                    "value_str": sub[concept].astype("string").to_numpy(),
                }
            )
        )
        materialized.append(concept)
    if frames:
        long_df = (
            pd.concat(frames, ignore_index=True)
            .sort_values([ID_COL, TIME_COL, "concept"])
            .reset_index(drop=True)
        )
    else:
        long_df = pd.DataFrame(
            columns=[ID_COL, TIME_COL, "concept", "value_num", "value_str"]
        )
    if export_package is not None and verify_source_package:
        verify_export_package(export_package)
    provenance = {
        "schema_version": "easyicu.cohort_trajectory/1",
        "source_mode": source_mode,
        "source": str(root),
        "export_authority": _export_authority_provenance(export_package),
        "database": database,
        "trajectory_window_hours": list(window) if window is not None else None,
        "trajectory_concepts_requested": list(dict.fromkeys(concepts)),
        "trajectory_concepts_materialized": materialized,
        "available_unobserved_concepts": available_unobserved,
        "unavailable_concepts": unavailable,
        "source_bounds_violation_policy": bounds_violation_policy,
        "source_bounds_exclusions": dict(sorted(bounds_violation_counts.items())),
        "n_rows": int(len(long_df)),
        "n_stays": int(long_df[ID_COL].nunique()) if len(long_df) else 0,
        "trajectory_sha256": _hash_df(long_df),
    }
    return long_df, provenance


def _materialize_with_open_export_package(
    package: ExportPackage,
    *,
    materialize_args: Mapping[str, Any],
) -> tuple[pd.DataFrame, Dict[str, Any], MaterializedColumnMetadataCollector]:
    """Reuse one verified package snapshot across sequential materializations."""

    requested = Path(materialize_args["data_path"]).expanduser().resolve(strict=True)
    if requested != package.root.resolve(strict=True):
        raise MaterializedMetadataError(
            "open export package does not match the requested data_path"
        )
    return _materialize_cohort_from_resolved_source(
        feature_concepts=materialize_args["feature_concepts"],
        database=materialize_args["database"],
        cohort_definition=materialize_args["cohort_definition"],
        cohort_window=materialize_args["cohort_window"],
        outcome_concepts=materialize_args["outcome_concepts"],
        static_concepts=materialize_args["static_concepts"],
        patient_ids=materialize_args["patient_ids"],
        source_mode="export",
        root=package.root,
        export_package=package,
        t0=time.time(),
        bounds_violation_policy=materialize_args["bounds_violation_policy"],
        positive_only_event_concepts=materialize_args["positive_only_event_concepts"],
        verify_source_package=False,
    )


def _trajectory_with_open_export_package(
    package: ExportPackage,
    *,
    concepts: Sequence[str],
    materialize_args: Mapping[str, Any],
    window: Optional[Window],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Build a trajectory from the same retained export-package snapshots."""

    return _build_trajectory_long_from_resolved_source(
        concepts=concepts,
        database=materialize_args["database"],
        window=window,
        patient_ids=materialize_args["patient_ids"],
        source_mode="export",
        root=package.root,
        export_package=package,
        bounds_violation_policy=materialize_args["bounds_violation_policy"],
        verify_source_package=False,
    )


def _exact_int_series(values: pd.Series, *, label: str) -> pd.Series:
    """Decode a structural identifier without lossy numeric coercion."""

    parsed = _coerce_int_stay(pd.DataFrame({ID_COL: values.reset_index(drop=True)}))[
        ID_COL
    ]
    parsed.name = values.name
    if parsed.isna().any():  # pragma: no cover - _coerce_int_stay rejects nulls
        raise MaterializedMetadataError(f"{label} contains a null identifier")
    return parsed


def _replace_row_identity_from_mapping(
    cohort: pd.DataFrame,
    *,
    mapping_path: Path,
    mapping_sha256: str,
    mapping_stay_column: str,
    mapping_patient_column: str,
    output_identity_column: str,
    authority_coordinates: Optional[Mapping[str, object]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Replace ``stay_id`` with a verified patient-grouped row identifier.

    The mapping is read through one non-symlink file descriptor and is used only
    for a deterministic one-to-one stay join.  The output remains unique per
    stay, as required by materialized-cohort authority, while the prefix before
    ``:s`` is stable for every stay owned by the same patient.  These host-only
    values are available to local analysis code but are not rendered into agent
    context or Provider prompts.
    """

    if (
        not mapping_path.is_absolute()
        or mapping_path.is_symlink()
        or mapping_path.suffix.lower() != ".parquet"
    ):
        raise MaterializedMetadataError(
            "replacement identity mapping must be an absolute regular Parquet file"
        )
    if (
        not isinstance(mapping_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", mapping_sha256) is None
    ):
        raise MaterializedMetadataError(
            "replacement identity mapping requires an exact sha256"
        )
    for label, value in (
        ("mapping stay column", mapping_stay_column),
        ("mapping patient column", mapping_patient_column),
        ("output identity column", output_identity_column),
    ):
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or "/" in value
            or "\\" in value
        ):
            raise MaterializedMetadataError(f"{label} is not canonical")
    if mapping_stay_column == mapping_patient_column:
        raise MaterializedMetadataError(
            "replacement identity mapping columns must be distinct"
        )
    if output_identity_column in cohort.columns and output_identity_column != ID_COL:
        raise MaterializedMetadataError(
            "replacement identity would overwrite an existing cohort column"
        )

    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            mapping_path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise MaterializedMetadataError(
                "replacement identity mapping must be a regular file"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != mapping_sha256:
                raise MaterializedMetadataError(
                    "replacement identity mapping digest mismatch"
                )
            handle.seek(0)
            table = pd.read_parquet(
                handle,
                columns=[mapping_stay_column, mapping_patient_column],
                engine="pyarrow",
            )
            after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise MaterializedMetadataError(
                "replacement identity mapping changed while being read"
            )
    except MaterializedMetadataError:
        raise
    except (OSError, ValueError) as exc:
        raise MaterializedMetadataError(
            "cannot read replacement identity mapping"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)

    mapping = table.rename(
        columns={
            mapping_stay_column: ID_COL,
            mapping_patient_column: output_identity_column,
        }
    )
    mapping[ID_COL] = _exact_int_series(
        mapping[ID_COL], label="replacement stay identity"
    )
    mapping[output_identity_column] = _exact_int_series(
        mapping[output_identity_column], label="replacement patient identity"
    )
    if mapping[ID_COL].duplicated().any():
        raise MaterializedMetadataError(
            "replacement identity mapping contains duplicate stay identifiers"
        )
    joined = cohort.merge(
        mapping,
        on=ID_COL,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if joined[output_identity_column].isna().any():
        raise MaterializedMetadataError(
            "replacement identity mapping does not cover every cohort stay"
        )
    patient_values = joined[output_identity_column].astype("int64")
    stay_values = joined[ID_COL].astype("int64")
    identity_values = "p" + patient_values.astype(str) + ":s" + stay_values.astype(str)
    if identity_values.duplicated().any():
        raise MaterializedMetadataError(
            "replacement identity mapping does not produce unique stay rows"
        )
    result = joined.drop(columns=[ID_COL, output_identity_column])
    result.insert(
        0,
        output_identity_column,
        identity_values,
    )
    coordinates = dict(authority_coordinates or {})
    return result, {
        "mapping_file_sha256": mapping_sha256,
        "mapping_file_size": int(before.st_size),
        "mapping_rows": int(len(mapping)),
        "mapping_stay_column": mapping_stay_column,
        "mapping_patient_column": mapping_patient_column,
        "output_identity_column": output_identity_column,
        "patient_group_derivation": {
            "algorithm": "prefix_before_:s",
            "delimiter": ":s",
        },
        "mapped_cohort_rows": int(len(result)),
        "authority_coordinates": coordinates,
    }


def materialize_to_parquet(
    output_dir: Union[str, Path],
    *,
    stem: str = "cohort",
    emit_trajectory: bool = False,
    trajectory_concepts: Optional[Sequence[str]] = None,
    trajectory_window: Optional[Window] = None,
    source_package: Optional[ExportPackage] = None,
    replacement_identity_path: Optional[Union[str, Path]] = None,
    replacement_identity_sha256: Optional[str] = None,
    replacement_identity_stay_column: str = ID_COL,
    replacement_identity_patient_column: Optional[str] = None,
    output_identity_column: Optional[str] = None,
    identity_authority_coordinates: Optional[Mapping[str, object]] = None,
    **kwargs: Any,
) -> Dict[str, Path]:
    """Materialize and write ``<stem>.parquet`` + ``<stem>_provenance.json``.

    When ``emit_trajectory`` is set, also writes ``<stem>_trajectory.parquet``
    (+ ``_provenance.json``): the long per-timepoint series for
    ``trajectory_concepts`` (default: the outcome + feature concepts), so the
    agent can build onsets / incident endpoints / landmark designs that the wide
    summary cannot express. Default off — existing callers are unaffected.
    """
    # Keep the public ``materialize_cohort`` argument contract.  This used to
    # forward ``**kwargs`` directly, so unknown/missing options failed rather
    # than silently selecting defaults after the typed-metadata bridge was
    # introduced.
    bound = inspect.signature(materialize_cohort).bind(**kwargs)
    bound.apply_defaults()
    materialize_args = bound.arguments
    stem = _canonical_stem(stem)
    out = prepare_real_directory(
        Path(output_dir).expanduser(), label="materialization output directory"
    )
    parquet_path = out / f"{stem}.parquet"
    prov_path = out / f"{stem}_provenance.json"
    if source_package is None:
        cohort, provenance, metadata_collector = _materialize_cohort_with_metadata(
            feature_concepts=materialize_args["feature_concepts"],
            database=materialize_args["database"],
            data_path=materialize_args["data_path"],
            cohort_definition=materialize_args["cohort_definition"],
            cohort_window=materialize_args["cohort_window"],
            outcome_concepts=materialize_args["outcome_concepts"],
            static_concepts=materialize_args["static_concepts"],
            patient_ids=materialize_args["patient_ids"],
            prefer_existing=materialize_args["prefer_existing"],
            bounds_violation_policy=materialize_args["bounds_violation_policy"],
            positive_only_event_concepts=materialize_args[
                "positive_only_event_concepts"
            ],
        )
    else:
        cohort, provenance, metadata_collector = _materialize_with_open_export_package(
            source_package,
            materialize_args=materialize_args,
        )
    identity_options = (
        replacement_identity_path,
        replacement_identity_sha256,
        replacement_identity_patient_column,
        output_identity_column,
    )
    if any(value is not None for value in identity_options) and not all(
        value is not None for value in identity_options
    ):
        raise MaterializedMetadataError(
            "replacement identity path, digest, patient column, and output column "
            "must be declared together"
        )
    identity_column = ID_COL
    identity_binding: Optional[dict[str, object]] = None
    if replacement_identity_path is not None:
        if emit_trajectory:
            raise MaterializedMetadataError(
                "replacement patient identity is not supported with stay trajectories"
            )
        cohort, identity_binding = _replace_row_identity_from_mapping(
            cohort,
            mapping_path=Path(replacement_identity_path).expanduser(),
            mapping_sha256=str(replacement_identity_sha256),
            mapping_stay_column=replacement_identity_stay_column,
            mapping_patient_column=str(replacement_identity_patient_column),
            output_identity_column=str(output_identity_column),
            authority_coordinates=identity_authority_coordinates,
        )
        identity_column = str(output_identity_column)
        provenance["columns"] = list(cohort.columns)
        provenance["cohort_sha256"] = _hash_df(cohort.reset_index(drop=True))
        provenance["replacement_row_identity"] = identity_binding
    producer_parameters = {
        "database": materialize_args["database"],
        "cohort_window": list(materialize_args["cohort_window"]),
        "feature_concepts": list(materialize_args["feature_concepts"]),
        "outcome_concepts": list(materialize_args["outcome_concepts"]),
        "static_concepts": list(materialize_args["static_concepts"]),
        "cohort_definition": (
            materialize_args["cohort_definition"].to_dict()
            if materialize_args["cohort_definition"] is not None
            else None
        ),
        "patient_ids": (
            list(materialize_args["patient_ids"])
            if materialize_args["patient_ids"] is not None
            else None
        ),
        "prefer_existing": bool(materialize_args["prefer_existing"]),
        "bounds_violation_policy": materialize_args["bounds_violation_policy"],
        "source_bounds_exclusions": provenance["source_bounds_exclusions"],
        "positive_only_event_concepts": list(
            materialize_args["positive_only_event_concepts"]
        ),
        "identity_column": identity_column,
        "replacement_row_identity": identity_binding,
    }
    if metadata_collector.enabled:
        _atomic_write_provenance(
            prov_path,
            {
                "schema_version": "easyicu.materialized_cohort_transaction/1",
                "materialized_authority_required": True,
                "column_metadata": None,
                "authority_transaction_state": "prepared",
            },
            canonical=True,
        )
    _atomic_write_parquet(cohort, parquet_path)
    provenance["cohort_file_sha256"] = _sha256_file(parquet_path)
    provenance["cohort_file_size"] = int(parquet_path.stat().st_size)
    descriptor = metadata_collector.seal_existing_cohort(
        cohort_path=parquet_path,
        identity_column=identity_column,
        source_database=materialize_args["database"],
        producer="cohort_materializer",
        producer_implementation_sha256=implementation_bundle_sha256(
            (
                Path(__file__),
                Path(__file__).resolve().parents[1]
                / "intake"
                / "materialized_metadata.py",
                Path(__file__).resolve().parents[2]
                / "concept"
                / "metadata_projection.py",
            )
        ),
        producer_parameters=producer_parameters,
        semantic_provenance=_semantic_materialization_provenance(provenance),
    )
    if descriptor is not None:
        provenance["column_metadata"] = descriptor
        provenance["materialized_authority_required"] = True
    _atomic_write_provenance(
        prov_path,
        provenance,
        canonical=descriptor is not None,
    )
    paths = {"parquet": parquet_path, "provenance": prov_path}
    if descriptor is not None:
        sidecar = descriptor["sidecar"]
        authority = descriptor["authority"]
        assert isinstance(sidecar, dict) and isinstance(authority, dict)
        paths["column_metadata"] = out / str(sidecar["file"])
        paths["cohort_authority"] = out / str(authority["file"])

    if emit_trajectory:
        concepts = trajectory_concepts
        if concepts is None:
            concepts = [
                *materialize_args["outcome_concepts"],
                *materialize_args["feature_concepts"],
            ]
        if source_package is None:
            long_df, traj_prov = build_trajectory_long(
                data_path=materialize_args["data_path"],
                concepts=concepts,
                database=materialize_args["database"],
                window=trajectory_window,
                patient_ids=materialize_args["patient_ids"],
                prefer_existing=materialize_args["prefer_existing"],
                bounds_violation_policy=materialize_args["bounds_violation_policy"],
            )
        else:
            long_df, traj_prov = _trajectory_with_open_export_package(
                source_package,
                concepts=concepts,
                materialize_args=materialize_args,
                window=trajectory_window,
            )
        # A trajectory bound to this universe may contain only identities the
        # universe owns.  This matters when the materializer applied a host-
        # declared cohort definition after reading the raw concept streams.
        universe_ids = set(cohort[ID_COL].tolist())
        long_df = long_df.loc[long_df[ID_COL].isin(universe_ids)].reset_index(drop=True)
        requested_trajectory_concepts = list(dict.fromkeys(concepts))
        source_available_concepts = {
            *traj_prov["trajectory_concepts_materialized"],
            *traj_prov["available_unobserved_concepts"],
        }
        observed_concepts = set(long_df["concept"].dropna().astype(str).tolist())
        materialized_concepts = [
            concept
            for concept in requested_trajectory_concepts
            if concept in observed_concepts
        ]
        available_unobserved_concepts = [
            concept
            for concept in requested_trajectory_concepts
            if concept in source_available_concepts and concept not in observed_concepts
        ]
        traj_prov["trajectory_concepts_materialized"] = materialized_concepts
        traj_prov["available_unobserved_concepts"] = available_unobserved_concepts
        traj_prov["n_rows"] = int(len(long_df))
        traj_prov["n_stays"] = int(long_df[ID_COL].nunique()) if len(long_df) else 0
        traj_prov["trajectory_sha256"] = _hash_df(long_df)
        traj_path = out / f"{stem}_trajectory.parquet"
        traj_prov_path = out / f"{stem}_trajectory_provenance.json"
        verified_cohort = load_verified_materialized_cohort_authority(parquet_path)
        if verified_cohort is not None:
            trajectory_parameters = {
                "database": materialize_args["database"],
                "requested_concepts": list(dict.fromkeys(concepts)),
                "materialized_concepts": list(
                    traj_prov["trajectory_concepts_materialized"]
                ),
                "available_unobserved_concepts": list(
                    traj_prov["available_unobserved_concepts"]
                ),
                "unavailable_concepts": list(traj_prov["unavailable_concepts"]),
                "window": list(trajectory_window) if trajectory_window else None,
                "bounds_violation_policy": materialize_args["bounds_violation_policy"],
                "source_bounds_exclusions": traj_prov["source_bounds_exclusions"],
                "bound_universe_authority_sha256": (verified_cohort.reference.sha256),
            }
            verified_trajectory = publish_materialized_trajectory_authority(
                long_df,
                traj_path,
                bound_universe_path=parquet_path,
                bound_universe=verified_cohort,
                requested_concepts=list(dict.fromkeys(concepts)),
                materialized_concepts=list(
                    traj_prov["trajectory_concepts_materialized"]
                ),
                available_unobserved_concepts=list(
                    traj_prov["available_unobserved_concepts"]
                ),
                unavailable_concepts=list(traj_prov["unavailable_concepts"]),
                window=trajectory_window,
                semantic_provenance=traj_prov,
                producer_implementation_sha256=implementation_bundle_sha256(
                    (
                        Path(__file__),
                        Path(__file__).resolve().parents[1]
                        / "intake"
                        / "materialized_trajectory.py",
                        Path(__file__).resolve().parents[1]
                        / "intake"
                        / "materialized_metadata.py",
                    )
                ),
                producer_parameters=trajectory_parameters,
            )
            paths["trajectory_authority"] = out / verified_trajectory.reference.file
        else:
            # Legacy/untyped materialization keeps its historical path-only
            # contract.  Typed inputs never take this branch.
            long_df.to_parquet(traj_path, index=False)
            traj_prov_path.write_text(
                json.dumps(traj_prov, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        paths["trajectory"] = traj_path
        paths["trajectory_provenance"] = traj_prov_path
    return paths
