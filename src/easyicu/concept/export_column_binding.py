"""Public typed-metadata binding for EasyICU export columns.

Extracted verbatim from ``easyicu.webserver.dataio`` so both the native export
UI and repository-local paper infrastructure can bind physical export columns to
typed concept metadata WITHOUT importing the webserver (UI) layer. This is the
concept-layer producer contract for how a physical column maps to a typed
concept role / unit / time coordinate; intake consumes the sealed result and
never repeats this parsing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence


class TypedExportError(ValueError):
    """Raised when a physical export column cannot be typed honestly."""

    def __init__(self, error: str, detail: Optional[Dict[str, Any]] = None):
        self.error = error
        self.detail = {"error": error, **(detail or {})}
        super().__init__(error)


def _metadata_definition_for_export(concept_id: str, module: str, dictionary: Any):
    """Return dictionary metadata or a truthful catalog-only derived fallback."""

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
    return ConceptDefinition.from_name_and_payload(concept_id, payload)


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
        raise TypedExportError(
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
        raise TypedExportError(
            "column_metadata_time_coordinate_unprojected",
            {"database": database, "columns": unprojected},
        )
    time_candidates = [
        candidate for candidate in ("charttime", "time") if candidate in columns
    ]
    time_candidates = list(dict.fromkeys(time_candidates))
    coordinates = tuple(
        TimeCoordinate(
            column=column,
            origin="icu_admission",
            unit="h",
        )
        for column in time_candidates
    )
    structural_ids = {candidate for candidate in preferred_ids if candidate in columns}
    return identity, coordinates, {*structural_ids, *time_candidates}


def _companion_projection(
    *,
    concept: str,
    column: str,
    series: Any,
    logical_concept: bool,
) -> Optional[tuple[Any, Optional[Any], Optional[str]]]:
    """Apply the producer-owned, exact companion-output contract.

    This is not an intake fallback: it runs while the producer still owns the
    manifest concept binding.  Intake v2 consumes only the resulting exact
    records and never repeats this name parsing.
    """

    import pandas as pd

    from easyicu.concept.metadata_projection import ConceptColumnRole
    from easyicu.concept.metadata_sidecar import DerivationWindow

    escaped = re.escape(concept)
    match = re.fullmatch(
        rf"{escaped}_(first_time|last_time|measured|count|n|max|min|mean|median|first|last)"
        r"(?:_([0-9]+(?:\.[0-9]+)?h))?",
        column,
    )
    if match is None:
        return None
    kind, raw_window = match.groups()
    window = None
    if raw_window is not None:
        window = DerivationWindow(
            origin="icu_admission",
            start_hours=0.0,
            end_hours=float(raw_window[:-1]),
        )
    if kind in {"n", "count"}:
        return ConceptColumnRole.COUNT, window, "structural_count"
    if kind == "measured":
        return ConceptColumnRole.MEASUREMENT_STATUS, window, "measurement_status"
    if kind in {"first_time", "last_time"}:
        role = (
            ConceptColumnRole.FIRST_OBSERVATION_TIME
            if kind == "first_time"
            else ConceptColumnRole.LAST_OBSERVATION_TIME
        )
        return role, window, "observation_time"
    numeric = bool(
        getattr(series, "dtype", None) is not None
        and (
            pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)
        )
    )
    if not numeric:
        return None
    if pd.api.types.is_bool_dtype(series) and not logical_concept:
        raise TypedExportError(
            "column_metadata_value_domain_invalid",
            {
                "column": column,
                "role": ConceptColumnRole.NUMERIC_AGGREGATE.value,
            },
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
    *,
    column: str,
    series: Any,
    role: Any,
    file_name: str,
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
    }
    if role not in checked_roles:
        return
    if role in {
        ConceptColumnRole.COUNT,
        ConceptColumnRole.FIRST_OBSERVATION_TIME,
        ConceptColumnRole.LAST_OBSERVATION_TIME,
    } and pd.api.types.is_bool_dtype(getattr(series, "dtype", None)):
        raise TypedExportError(
            "column_metadata_value_domain_invalid",
            {"file": file_name, "column": column, "role": role.value},
        )
    if pd.api.types.is_datetime64_any_dtype(
        getattr(series, "dtype", None)
    ) or pd.api.types.is_timedelta64_dtype(getattr(series, "dtype", None)):
        raise TypedExportError(
            "column_metadata_value_domain_invalid",
            {"file": file_name, "column": column, "role": role.value},
        )
    nonmissing = series.notna()
    numeric = pd.to_numeric(series, errors="coerce")
    if bool((nonmissing & numeric.isna()).any()):
        raise TypedExportError(
            "column_metadata_value_domain_invalid",
            {"file": file_name, "column": column, "role": role.value},
        )
    values = numeric.loc[nonmissing].astype(float)
    if values.empty:
        return
    raw = values.to_numpy(dtype=float, copy=False)
    valid = bool(np.isfinite(raw).all())
    if role in {
        ConceptColumnRole.EVENT_STATUS,
        ConceptColumnRole.MEASUREMENT_STATUS,
    }:
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
        raise TypedExportError(
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


def _build_export_file_metadata_binding(
    *,
    relative_path: str,
    module: str,
    frame: Any,
    concept_ids: Sequence[str],
    database: str,
    database_class_prefixes: Sequence[str],
    dictionary: Any,
):
    """Bind producer-owned physical outputs to typed metadata exactly once."""

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
        raise TypedExportError(
            "column_metadata_physical_columns_invalid",
            {"file": relative_path, "reason": "non_string_or_empty"},
        )
    columns = list(raw_columns)
    if len(set(columns)) != len(columns):
        raise TypedExportError(
            "column_metadata_physical_columns_invalid",
            {"file": relative_path, "reason": "duplicate"},
        )
    identity, time_coordinates, structural = _export_identity_and_time_coordinates(
        columns, database
    )
    _validate_time_coordinate_series(
        frame=frame,
        coordinates=time_coordinates,
        file_name=relative_path,
    )
    value_columns = [column for column in columns if column not in structural]
    unresolved_columns = set(value_columns)
    binding_specs: Dict[
        str, tuple[str, Any, Optional[Any], Optional[str], Optional[str]]
    ] = {}

    # Exact physical names always win, including a real concept named ``foo_n``.
    for concept in concept_ids:
        if concept in unresolved_columns:
            definition = _metadata_definition_for_export(concept, module, dictionary)
            physical_is_bool = pd.api.types.is_bool_dtype(frame[concept])
            concept_is_logical = definition.class_name == "lgl_cncpt"
            categorical_boolean = (
                physical_is_bool and definition.class_name == "fct_cncpt"
            )
            if physical_is_bool and not (concept_is_logical or categorical_boolean):
                raise TypedExportError(
                    "column_metadata_value_domain_invalid",
                    {
                        "file": relative_path,
                        "column": concept,
                        "role": ConceptColumnRole.VALUE.value,
                    },
                )
            role = (
                ConceptColumnRole.EVENT_STATUS
                if concept_is_logical or categorical_boolean
                else ConceptColumnRole.VALUE
            )
            binding_specs[concept] = (
                concept,
                role,
                None,
                None,
                None,
            )
            unresolved_columns.remove(concept)

    # Then apply only the producer's closed companion contract.
    for concept in concept_ids:
        definition = _metadata_definition_for_export(concept, module, dictionary)
        for column in sorted(tuple(unresolved_columns)):
            projected = _companion_projection(
                concept=concept,
                column=column,
                series=frame[column],
                logical_concept=definition.class_name == "lgl_cncpt",
            )
            if projected is None:
                continue
            role, window, transform = projected
            aggregation = None
            if transform and transform.startswith("aggregate:"):
                aggregation = transform.split(":", 1)[1]
            binding_specs[column] = (
                concept,
                role,
                window,
                transform,
                aggregation,
            )
            unresolved_columns.remove(column)

    bindings: Dict[str, ColumnMetadataBinding] = {}
    for column, (
        concept,
        role,
        window,
        transform,
        aggregation,
    ) in binding_specs.items():
        definition = _metadata_definition_for_export(concept, module, dictionary)
        _validate_metadata_series_domain(
            column=column,
            series=frame[column],
            role=role,
            file_name=relative_path,
        )
        spec_kwargs: Dict[str, Any] = {}
        if role in {
            ConceptColumnRole.FIRST_OBSERVATION_TIME,
            ConceptColumnRole.LAST_OBSERVATION_TIME,
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


# Stable public names over the producer-owned binding functions.
metadata_definition_for_export = _metadata_definition_for_export
export_identity_and_time_coordinates = _export_identity_and_time_coordinates
companion_projection = _companion_projection
validate_metadata_series_domain = _validate_metadata_series_domain
validate_time_coordinate_series = _validate_time_coordinate_series
build_export_file_metadata_binding = _build_export_file_metadata_binding

__all__ = [
    "TypedExportError",
    "build_export_file_metadata_binding",
    "companion_projection",
    "export_identity_and_time_coordinates",
    "metadata_definition_for_export",
    "validate_metadata_series_domain",
    "validate_time_coordinate_series",
]
