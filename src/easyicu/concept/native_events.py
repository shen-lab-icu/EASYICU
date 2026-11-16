"""Typed native event tables, independent of the scalar concept grid.

An event table has its own row identity and time coordinates. It must never
enter the hourly/minute median, outer-merge or duplicate-key consolidation.
Its properties are typed physical fields, not new scalar dictionary concepts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class EventTableSpec:
    module: str
    fields: tuple[tuple[str, str], ...]
    domains: tuple[tuple[str, tuple[str, ...]], ...] = ()
    time_columns: tuple[str, ...] = ("start", "end")
    primary_key: tuple[str, ...] = ("stay_id", "event_index")
    row_grain: str = "one_row_per_native_administration"

    def empty(self) -> pd.DataFrame:
        return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in self.fields})

    def canonical(self, frame: pd.DataFrame) -> pd.DataFrame:
        if set(frame.columns) != {name for name, _ in self.fields}:
            raise ValueError(f"{self.module}: event schema mismatch")
        out = frame.astype(dict(self.fields))[[name for name, _ in self.fields]]
        if out.isna().any().any() or out.duplicated(list(self.primary_key)).any():
            raise ValueError(f"{self.module}: missing event fields or duplicate record identities")
        if any(not out[name].isin(values).all() for name, values in self.domains):
            raise ValueError(f"{self.module}: unrecognized event property code")
        return out


NATIVE_EVENT_MODULES: Mapping[str, EventTableSpec] = {
    "fluid_events": EventTableSpec("fluid_events", (
        ("stay_id", "int64"), ("event_index", "int64"), ("start", "float64"),
        ("end", "float64"), ("volume_ml", "float64"), ("fluid_class", "string"),
        ("mode", "string"), ("mode_evidence", "string"),
    ), domains=(
        ("fluid_class", ("crystalloid", "colloid", "blood_rbc", "blood_ffp", "blood_platelet", "parenteral")),
        ("mode", ("continuous", "bolus", "unknown")),
        ("mode_evidence", ("Continuous IV", "Continuous Med", "Bolus", "Drug Push", "iscontinuous=1",
                           "iscontinuous=0", "IsSingleDose=0", "IsSingleDose=1", "unrecognized_or_missing_mode")),
    )),
}


def event_metadata_binding(spec: EventTableSpec, *, relative_path: str, database: str):
    """Bind every event property and both clocks without scalar concept claims."""
    from .metadata_projection import ColumnProjectionSpec, ConceptColumnRole, project_concept_column_metadata
    from .metadata_sidecar import ColumnMetadataBinding, ColumnMetadataFileBinding, TimeCoordinate
    from .schema import ConceptDefinition
    from ..databases.time_origin import native_export_time_origin

    bindings = {}
    for name, dtype in spec.fields:
        if name == "stay_id" or name in spec.time_columns:
            continue
        definition = ConceptDefinition(
            name=name, sources={}, units=["mL"] if name == "volume_ml" else ["category"] if dtype == "string" else None,
            minimum=0.0 if dtype in {"int64", "float64"} else None,
            class_name="fct_cncpt" if dtype == "string" else "num_cncpt",
            levels=list(dict(spec.domains)[name]) if name in dict(spec.domains) else None,
            description=("Zero-based native administration ordinal within a stay" if name == "event_index"
                         else f"Native administration {name}"),
        )
        metadata = project_concept_column_metadata(
            definition, spec=ColumnProjectionSpec(
                column_name=name, source_concept=name,
                role=ConceptColumnRole.VALUE,
            ), source_database=database, source_database_class_prefixes=(),
        )
        bindings[name] = ColumnMetadataBinding(metadata)
    return ColumnMetadataFileBinding(
        relative_path=relative_path, module=spec.module, identity_column="stay_id",
        time_coordinates=tuple(TimeCoordinate(column=name, origin=native_export_time_origin(database), unit="h")
                               for name in spec.time_columns), columns=bindings,
    )
