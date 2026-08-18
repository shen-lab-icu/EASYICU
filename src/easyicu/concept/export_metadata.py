"""Producer-owned typed metadata for native EasyICU exports.

This module is deliberately independent of the Web server.  Both the browser/
HTTP export path and :func:`easyicu.api.extract_database` must seal exactly the
same physical-column contract while the producer still owns its frames.  Intake
only verifies this contract; it never infers concept identities from names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True, slots=True)
class ExportMetadataError(ValueError):
    """A producer cannot honestly bind a physical export column."""

    error: str
    detail: Dict[str, Any]

    def __init__(self, error: str, detail: Optional[Dict[str, Any]] = None):
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "detail", {"error": error, **(detail or {})})
        ValueError.__init__(self, error)


def _metadata_definition_for_export(concept_id: str, module: str, dictionary: Any):
    """Return dictionary metadata or a truthful catalog-only fallback."""

    definition = dictionary.get(concept_id)
    if definition is not None:
        return definition
    from easyicu.concept.catalog import CONCEPT_DESCRIPTIONS, CONCEPT_DICTIONARY
    from easyicu.concept.schema import ConceptDefinition

    _name_en, _name_zh, unit = CONCEPT_DICTIONARY.get(
        concept_id, (concept_id, concept_id, "")
    )
    description, _description_zh = CONCEPT_DESCRIPTIONS.get(concept_id, ("", ""))
    payload: Dict[str, Any] = {
        "description": description or None,
        "category": module or None,
        "sources": {},
    }
    if unit:
        payload["unit"] = [unit]
    if str(unit).strip().lower() == "boolean":
        # Derived catalog outputs such as mort_28d are physical event flags but
        # intentionally have no raw-source dictionary entry.  Preserve that
        # producer-owned logical type instead of falling back to num_cncpt and
        # subsequently rejecting an honest boolean column.
        payload["class_name"] = "lgl_cncpt"
    return ConceptDefinition.from_name_and_payload(concept_id, payload)


def _catalog_declares_event_semantics(concept_id: str) -> bool:
    """Return whether the public catalog declares a binary event output.

    Derived record concepts can retain ``rec_cncpt`` in the extraction
    dictionary because they carry event times while their materialized value
    column is still a Boolean status.  The catalog owns that public value
    domain; export metadata must not independently reinterpret the storage
    dtype or downgrade it to an untyped numeric value.
    """

    from easyicu.concept.catalog import CONCEPT_DICTIONARY

    _name_en, _name_zh, unit = CONCEPT_DICTIONARY.get(
        concept_id, (concept_id, concept_id, "")
    )
    return str(unit).strip().lower() == "boolean"


def concept_declares_event_status(concept_id: str, definition: Any = None) -> bool:
    """Return the concept owner's declared event-status semantics.

    Native export metadata and legacy catalog projection must use one policy.
    The physical dtype is deliberately not consulted: positive-only events can
    be stored as ``1/null`` and categorical booleans remain ordinary values.
    """

    class_name = getattr(definition, "class_name", None)
    return class_name == "lgl_cncpt" or (
        class_name != "fct_cncpt" and _catalog_declares_event_semantics(concept_id)
    )


def _export_identity_and_time_coordinates(
    columns: Sequence[str], database: str
) -> tuple[str, tuple[Any, ...], set[str]]:
    """Describe normalized structural coordinates without assigning a concept."""

    from easyicu.concept.metadata_sidecar import TimeCoordinate
    from easyicu.config import load_src_cfg
    from easyicu.database_config import (
        END_TIME_COLUMNS,
        START_TIME_COLUMNS,
        TIME_COLUMNS,
    )

    source_config = load_src_cfg(str(database).lower())
    icu_identifier = source_config.id_configs.get("icustay")
    preferred_ids = tuple(
        dict.fromkeys(
            (
                "stay_id",
                *((icu_identifier.id,) if icu_identifier is not None else ()),
                *(
                    item.id
                    for item in sorted(
                        source_config.id_configs.values(),
                        key=lambda item: item.position,
                        reverse=True,
                    )
                ),
            )
        )
    )
    identity = next(
        (candidate for candidate in preferred_ids if candidate in columns), None
    )
    if identity is None:
        raise ExportMetadataError(
            "column_metadata_identity_ambiguous",
            {"database": database, "candidates": []},
        )
    native_time_names = {
        value
        for value in (
            TIME_COLUMNS.get(str(database).lower(), ""),
            START_TIME_COLUMNS.get(str(database).lower(), ""),
            END_TIME_COLUMNS.get(str(database).lower(), ""),
        )
        if value and value not in {"charttime", "time"}
    }
    unprojected = sorted(native_time_names.intersection(columns))
    if unprojected:
        raise ExportMetadataError(
            "column_metadata_time_coordinate_unprojected",
            {"database": database, "columns": unprojected},
        )
    time_candidates = list(
        dict.fromkeys(c for c in ("charttime", "time") if c in columns)
    )
    coordinates = tuple(
        TimeCoordinate(column=column, origin="icu_admission", unit="h")
        for column in time_candidates
    )
    structural_ids = {candidate for candidate in preferred_ids if candidate in columns}
    return identity, coordinates, {*structural_ids, *time_candidates}


def _companion_projection(
    *, concept: str, column: str, series: Any, logical_concept: bool
) -> Optional[tuple[Any, Optional[Any], Optional[str]]]:
    """Apply the producer-owned, closed companion-output contract."""

    import pandas as pd

    from easyicu.concept.metadata_projection import ConceptColumnRole
    from easyicu.concept.metadata_sidecar import DerivationWindow

    escaped = re.escape(concept)
    match = re.fullmatch(
        rf"{escaped}_(first_time|last_time|time|measured|observed|available|count|n|max|min|mean|median|first|last)"
        r"(?:_([0-9]+(?:\.[0-9]+)?h))?",
        column,
    )
    if match is None:
        return None
    kind, raw_window = match.groups()
    window = (
        DerivationWindow(
            origin="icu_admission", start_hours=0.0, end_hours=float(raw_window[:-1])
        )
        if raw_window is not None
        else None
    )
    if kind in {"n", "count"}:
        return ConceptColumnRole.COUNT, window, "structural_count"
    if kind in {"measured", "observed", "available"}:
        transform = {
            "measured": "measurement_status",
            "observed": "owner_observed_status",
            "available": "owner_available_status",
        }[kind]
        return ConceptColumnRole.MEASUREMENT_STATUS, window, transform
    if kind in {"first_time", "last_time"}:
        role = (
            ConceptColumnRole.FIRST_OBSERVATION_TIME
            if kind == "first_time"
            else ConceptColumnRole.LAST_OBSERVATION_TIME
        )
        return role, window, "observation_time"
    if kind == "time":
        if not logical_concept or window is not None:
            return None
        return ConceptColumnRole.EVENT_TIME, None, "source_event_time"
    numeric = bool(
        getattr(series, "dtype", None) is not None
        and (
            pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)
        )
    )
    if not numeric:
        return None
    if pd.api.types.is_bool_dtype(series) and not logical_concept:
        raise ExportMetadataError(
            "column_metadata_value_domain_invalid",
            {"column": column, "role": ConceptColumnRole.NUMERIC_AGGREGATE.value},
        )
    if logical_concept:
        if kind not in {"first", "last", "max", "min", "mean"}:
            return None
        role = (
            ConceptColumnRole.EVENT_FRACTION
            if kind == "mean"
            else ConceptColumnRole.EVENT_STATUS
        )
    else:
        role = ConceptColumnRole.NUMERIC_AGGREGATE
    return role, window, f"aggregate:{kind}"


def _validate_metadata_series_domain(
    *, column: str, series: Any, role: Any, file_name: str
) -> None:
    """Prove mechanical physical-value semantics before sealing metadata."""

    import numpy as np
    import pandas as pd

    from easyicu.concept.metadata_projection import ConceptColumnRole

    checked_roles = {
        ConceptColumnRole.EVENT_STATUS,
        ConceptColumnRole.EVENT_FRACTION,
        ConceptColumnRole.MEASUREMENT_STATUS,
        ConceptColumnRole.COUNT,
        ConceptColumnRole.FIRST_OBSERVATION_TIME,
        ConceptColumnRole.LAST_OBSERVATION_TIME,
        ConceptColumnRole.EVENT_TIME,
    }
    if role not in checked_roles:
        return
    if role in {
        ConceptColumnRole.COUNT,
        ConceptColumnRole.FIRST_OBSERVATION_TIME,
        ConceptColumnRole.LAST_OBSERVATION_TIME,
        ConceptColumnRole.EVENT_TIME,
    } and pd.api.types.is_bool_dtype(getattr(series, "dtype", None)):
        raise ExportMetadataError(
            "column_metadata_value_domain_invalid",
            {"file": file_name, "column": column, "role": role.value},
        )
    if pd.api.types.is_datetime64_any_dtype(
        getattr(series, "dtype", None)
    ) or pd.api.types.is_timedelta64_dtype(getattr(series, "dtype", None)):
        raise ExportMetadataError(
            "column_metadata_value_domain_invalid",
            {"file": file_name, "column": column, "role": role.value},
        )
    nonmissing = series.notna()
    numeric = pd.to_numeric(series, errors="coerce")
    if bool((nonmissing & numeric.isna()).any()):
        raise ExportMetadataError(
            "column_metadata_value_domain_invalid",
            {"file": file_name, "column": column, "role": role.value},
        )
    values = numeric.loc[nonmissing].astype(float)
    if values.empty:
        return
    raw = values.to_numpy(dtype=float, copy=False)
    valid = bool(np.isfinite(raw).all())
    if role in {ConceptColumnRole.EVENT_STATUS, ConceptColumnRole.MEASUREMENT_STATUS}:
        valid = valid and bool(values.isin((0.0, 1.0)).all())
    elif role is ConceptColumnRole.EVENT_FRACTION:
        valid = valid and bool(((values >= 0.0) & (values <= 1.0)).all())
    elif role is ConceptColumnRole.COUNT:
        valid = (
            valid
            and bool((values >= 0.0).all())
            and bool(np.equal(raw, np.floor(raw)).all())
        )
    if not valid:
        raise ExportMetadataError(
            "column_metadata_value_domain_invalid",
            {"file": file_name, "column": column, "role": role.value},
        )


def _validate_time_coordinate_series(
    *, frame: Any, coordinates: Sequence[Any], file_name: str
) -> None:
    from easyicu.concept.metadata_projection import ConceptColumnRole

    for coordinate in coordinates:
        _validate_metadata_series_domain(
            column=coordinate.column,
            series=frame[coordinate.column],
            role=ConceptColumnRole.FIRST_OBSERVATION_TIME,
            file_name=file_name,
        )


def build_export_file_metadata_binding(
    *,
    relative_path: str,
    module: str,
    frame: Any,
    concept_ids: Sequence[str],
    database: str,
    database_class_prefixes: Sequence[str],
    dictionary: Any,
):
    """Bind a producer-owned physical output to typed metadata exactly once."""

    import pandas as pd

    from easyicu.concept.metadata_projection import (
        ColumnProjectionSpec,
        ConceptColumnRole,
        project_concept_column_metadata,
    )
    from easyicu.concept.metadata_sidecar import (
        ColumnMetadataBinding,
        ColumnMetadataFileBinding,
    )

    raw_columns = list(frame.columns)
    if any(not isinstance(value, str) or not value for value in raw_columns):
        raise ExportMetadataError(
            "column_metadata_physical_columns_invalid",
            {"file": relative_path, "reason": "non_string_or_empty"},
        )
    columns = list(raw_columns)
    if len(set(columns)) != len(columns):
        raise ExportMetadataError(
            "column_metadata_physical_columns_invalid",
            {"file": relative_path, "reason": "duplicate"},
        )
    identity, time_coordinates, structural = _export_identity_and_time_coordinates(
        columns, database
    )
    _validate_time_coordinate_series(
        frame=frame, coordinates=time_coordinates, file_name=relative_path
    )
    unresolved_columns = {column for column in columns if column not in structural}
    binding_specs: Dict[
        str, tuple[str, Any, Optional[Any], Optional[str], Optional[str]]
    ] = {}

    for concept in concept_ids:
        if concept in unresolved_columns:
            definition = _metadata_definition_for_export(concept, module, dictionary)
            physical_is_bool = pd.api.types.is_bool_dtype(frame[concept])
            concept_is_logical = concept_declares_event_status(concept, definition)
            categorical_boolean = (
                physical_is_bool and definition.class_name == "fct_cncpt"
            )
            if physical_is_bool and not (concept_is_logical or categorical_boolean):
                raise ExportMetadataError(
                    "column_metadata_value_domain_invalid",
                    {
                        "file": relative_path,
                        "column": concept,
                        "role": ConceptColumnRole.VALUE.value,
                    },
                )
            binding_specs[concept] = (
                concept,
                (
                    ConceptColumnRole.EVENT_STATUS
                    if concept_is_logical or categorical_boolean
                    else ConceptColumnRole.VALUE
                ),
                None,
                None,
                None,
            )
            unresolved_columns.remove(concept)

    for concept in concept_ids:
        definition = _metadata_definition_for_export(concept, module, dictionary)
        for column in sorted(tuple(unresolved_columns)):
            projected = _companion_projection(
                concept=concept,
                column=column,
                series=frame[column],
                logical_concept=concept_declares_event_status(concept, definition),
            )
            if projected is None:
                continue
            role, window, transform = projected
            binding_specs[column] = (
                concept,
                role,
                window,
                transform,
                (
                    transform.split(":", 1)[1]
                    if transform and transform.startswith("aggregate:")
                    else None
                ),
            )
            unresolved_columns.remove(column)

    bindings: Dict[str, Any] = {}
    for column, (
        concept,
        role,
        window,
        transform,
        aggregation,
    ) in binding_specs.items():
        definition = _metadata_definition_for_export(concept, module, dictionary)
        _validate_metadata_series_domain(
            column=column, series=frame[column], role=role, file_name=relative_path
        )
        spec_kwargs: Dict[str, Any] = {}
        if role in {
            ConceptColumnRole.FIRST_OBSERVATION_TIME,
            ConceptColumnRole.LAST_OBSERVATION_TIME,
            ConceptColumnRole.EVENT_TIME,
        }:
            spec_kwargs.update(time_origin="icu_admission", time_unit="h")
        metadata = project_concept_column_metadata(
            definition,
            spec=ColumnProjectionSpec(
                column_name=column,
                source_concept=concept,
                role=role,
                aggregation=aggregation,
                **spec_kwargs,
            ),
            source_database=database,
            source_database_class_prefixes=database_class_prefixes,
        )
        bindings[column] = ColumnMetadataBinding(
            metadata=metadata,
            derivation_window=window,
            representation_transform=transform,
        )
    return ColumnMetadataFileBinding(
        relative_path=relative_path,
        module=module,
        identity_column=identity,
        time_coordinates=time_coordinates,
        columns=bindings,
    )


def missing_primary_metadata_concepts(
    *, concept_plan: Dict[str, List[str]], file_bindings: Sequence[Any]
) -> List[str]:
    """Return selected concepts without one unambiguous typed primary column."""

    from easyicu.concept.metadata_projection import ConceptColumnRole

    primary_roles = {ConceptColumnRole.VALUE, ConceptColumnRole.EVENT_STATUS}
    missing: List[str] = []
    for module, concepts in concept_plan.items():
        module_bindings = [item for item in file_bindings if item.module == module]
        for concept in concepts:
            owned = [
                (column, binding)
                for file_binding in module_bindings
                for column, binding in file_binding.columns.items()
                if binding.metadata.source_concept == concept
                and binding.metadata.role in primary_roles
            ]
            exact = [column for column, _binding in owned if column == concept]
            if len(exact) != 1 and len(owned) != 1:
                missing.append(concept)
    return sorted(set(missing))


__all__ = [
    "ExportMetadataError",
    "build_export_file_metadata_binding",
    "concept_declares_event_status",
    "missing_primary_metadata_concepts",
]
