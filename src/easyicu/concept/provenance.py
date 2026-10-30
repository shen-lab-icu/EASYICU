"""Frozen, metadata-only extraction provenance; never reconstruct old runs.

The snapshot records the resolver's selected configuration and conditional
callback rules. It makes no claim about which units occurred in patient rows.
Missing code descriptions remain missing; diagnosis text is not an ICD list.
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Callable, Iterator, Literal, Mapping, MutableMapping, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .expr_parser import _parse_binary_op, _split_arguments, _strip_quotes
from .metadata_projection import (
    ColumnProjectionSpec, ConceptColumnRole, project_concept_column_metadata,
)
from .schema import ConceptDictionary
from easyicu.config import DataSourceConfig

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ResolverInputSink = Callable[[ConceptDictionary, DataSourceConfig], None]
_resolver_sink: ContextVar[ResolverInputSink | None] = ContextVar("extraction_provenance_sink", default=None)


def metadata_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


class _Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class CodeProvenance(_Metadata):
    code: str
    description: str | None = None
    description_status: Literal["recorded", "not_recorded"] = "not_recorded"

    @model_validator(mode="after")
    def check_description(self):
        if (self.description is not None) != (self.description_status == "recorded"):
            raise ValueError("code description status mismatch")
        return self


class UnitConversionProvenance(_Metadata):
    rule: str
    source_unit_pattern: str | None = None
    target_unit: str | None = None
    factor: float | None = None
    offset: float | None = None
    application: Literal["source_callback_rule"] = "source_callback_rule"
    # callback_apply applies the rule to every row if no unit column exists.
    missing_unit_column: Literal["apply_to_all_rows"] = "apply_to_all_rows"
    blank_unit_matches: bool = False


class SourceProvenance(_Metadata):
    table: str | None = None
    selector_variable: str | None = None
    selector_regex: str | None = None
    item_ids: tuple[int | float | str, ...] = ()
    value_variable: str | None = None
    unit_variable: str | None = None
    time_variable: str | None = None
    duration_variable: str | None = None
    interval: str | None = None
    callback: str | None = None
    codes: tuple[CodeProvenance, ...] = ()
    unit_conversions: tuple[UnitConversionProvenance, ...] = ()


class ConceptProvenance(_Metadata):
    concept_id: str
    status: Literal["recorded", "not_recorded"]
    source_database: str
    dictionary_source_database: str | None = None
    definition: str | None = None
    definition_source: str | None = None
    definition_version: str | None = None
    canonical_unit: str | None = None
    derivation_callback: str | None = None
    derivation_inputs: tuple[str, ...] = ()
    sources: tuple[SourceProvenance, ...] = ()

    @model_validator(mode="after")
    def check_unrecorded(self):
        if self.status == "not_recorded" and any((
            self.sources, self.definition, self.definition_source, self.definition_version,
            self.canonical_unit, self.derivation_callback, self.derivation_inputs,
            self.dictionary_source_database,
        )):
            raise ValueError("unrecorded concept cannot contain invented provenance")
        return self


class CohortFilterProvenance(_Metadata):
    """Configuration values only, with no observed population counts."""

    field: str
    values: tuple[str | int | float | bool, ...]
    codes: tuple[CodeProvenance, ...] = ()


class DataProvenanceSnapshot(_Metadata):
    schema_version: Literal["easyicu.data-provenance-snapshot/1"] = "easyicu.data-provenance-snapshot/1"
    dictionary_sha256: Digest
    resolver_inputs_sha256: Digest
    implementation_sha256: Digest
    resource_inputs_sha256: Digest | None = None
    source_config_sha256: Digest | None = None
    source_database: str
    class_prefixes: tuple[str, ...] = ()
    concepts: tuple[ConceptProvenance, ...]
    cohort_filters: tuple[CohortFilterProvenance, ...] = ()
    resolver_parameters: tuple[CohortFilterProvenance, ...] = ()
    snapshot_sha256: Digest

    @model_validator(mode="after")
    def check_digest(self):
        payload = self.model_dump(mode="json", exclude={"snapshot_sha256"})
        if metadata_digest(payload) != self.snapshot_sha256:
            raise ValueError("data provenance snapshot digest mismatch")
        if any(concept.source_database != self.source_database for concept in self.concepts):
            raise ValueError("data provenance database mismatch")
        if len({concept.concept_id for concept in self.concepts}) != len(self.concepts):
            raise ValueError("duplicate provenance concept")
        return self


class ExportDataProvenance(_Metadata):
    schema_version: Literal["easyicu.export-data-provenance/1"] = "easyicu.export-data-provenance/1"
    snapshots: tuple[DataProvenanceSnapshot, ...]

    @model_validator(mode="after")
    def check_databases(self):
        if len({snapshot.source_database for snapshot in self.snapshots}) > 1:
            raise ValueError("export provenance database mismatch")
        return self


def _json_metadata(value: object) -> object:
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_metadata(item) for item in value]
    return value


def _conversions(callback: str | None) -> tuple[UnitConversionProvenance, ...]:
    expression = (callback or "").strip()
    combined = re.fullmatch(r"combine_callbacks\((.+)\)", expression, re.DOTALL)
    if combined:
        return tuple(rule for part in _split_arguments(combined[1]) for rule in _conversions(part))
    match = re.fullmatch(r"convert_unit\((.+)\)", expression, re.DOTALL)
    if not match:
        return ()
    arguments = _split_arguments(match[1])
    symbol, operand = _parse_binary_op(arguments[0])
    factor = offset = None
    if symbol in {"*", "/", "+", "-"} and isinstance(operand, (int, float)):
        if symbol == "*":
            factor, offset = float(operand), 0.0
        elif symbol == "/" and operand != 0:
            factor, offset = 1.0 / operand, 0.0
        elif symbol in {"+", "-"}:
            factor, offset = 1.0, float(operand) * (1 if symbol == "+" else -1)
    elif symbol == "fahr_to_cels":
        factor, offset = 5.0 / 9.0, -32.0 * 5.0 / 9.0
    return (UnitConversionProvenance(
        rule=expression,
        target_unit=_strip_quotes(arguments[1]) if len(arguments) > 1 else None,
        source_unit_pattern=_strip_quotes(arguments[2]) if len(arguments) > 2 else None,
        factor=factor, offset=offset,
        blank_unit_matches=symbol == "fahr_to_cels" or len(arguments) < 3,
    ),)


def resolver_implementation_digest() -> str:
    """Hash the extraction implementation bundle without installation paths.

    This binds executable source, not the bytes actually consumed from a raw
    database. In particular, it is not a producer receipt or a raw data digest.
    """
    root = Path(__file__).resolve().parents[1]
    # The resolver delegates to these owners for actual table reads, timing,
    # callbacks and derived scores. Omitting them would allow meaningful
    # extraction changes without changing the recorded implementation digest.
    packages = ("concept", "api", "table", "io", "scores", "utils")
    members = [path for package in packages for path in (root / package).rglob("*.py")]
    members.extend(root / name for name in (
        "base.py", "config.py", "clinical_contracts.py", "datasource.py", "load_concepts.py",
    ))
    return metadata_digest({
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(members)
    })


def record_resolver_inputs(
    parameters: MutableMapping[str, object], loader,
) -> None:
    """Observe the selected loader before reads, including an older cached dictionary."""
    sink = parameters.pop("_provenance_sink", None)
    if sink is None:
        sink = _resolver_sink.get()
    if sink is not None:
        if not callable(sink):
            raise TypeError("provenance sink must be callable")
        sink(loader.concept_resolver.dictionary, loader.datasource.config)


@contextmanager
def observing_resolver_inputs(sink: ResolverInputSink) -> Iterator[None]:
    token = _resolver_sink.set(sink)
    try:
        yield
    finally:
        _resolver_sink.reset(token)


def capture_data_provenance(
    dictionary: ConceptDictionary, *, concepts: Sequence[str], source_database: str,
    class_prefixes: Sequence[str] = (),
    cohort_filters: Sequence[CohortFilterProvenance] = (),
    resource_inputs_sha256: str | None = None,
    source_config_sha256: str | None = None,
    source_config: DataSourceConfig | None = None,
    resolver_parameters: Mapping[str, object] | None = None,
) -> DataProvenanceSnapshot:
    """Freeze selected resolver inputs, including dependencies, before extraction."""
    from easyicu.clinical_contracts import load_clinical_contracts

    database = source_database.strip().lower()
    # These are resolver configuration coordinates, not evidence that any
    # particular patient row was extracted or transformed. Never substitute
    # packaged defaults for a run's possibly customized source configuration.
    if source_config is not None:
        actual_sha = metadata_digest(source_config.model_dump(mode="json"))
        if source_config.name.strip().lower() != database or source_config_sha256 != actual_sha:
            raise ValueError("data provenance source configuration identity mismatch")
    prefixes = tuple(value.strip().lower() for value in class_prefixes)
    dictionary_payload = {
        key: _json_metadata(asdict(definition)) for key, definition in dictionary.items()
    }
    contracts = load_clinical_contracts()
    pending = list(dict.fromkeys(concepts))
    projected: dict[str, ConceptProvenance] = {}
    while pending:
        concept = pending.pop(0)
        if concept in projected:
            continue
        definition = dictionary.get(concept)
        if definition is None:
            projected[concept] = ConceptProvenance(
                concept_id=concept, status="not_recorded", source_database=database,
            )
            continue
        metadata = project_concept_column_metadata(
            definition, spec=ColumnProjectionSpec(
                column_name=concept, source_concept=concept, role=ConceptColumnRole.VALUE,
            ), source_database=database, source_database_class_prefixes=prefixes,
        )
        sources = []
        for source in definition.sources.get(metadata.dictionary_source_database, ()):
            if (database == "sic" and source.callback == "sic_case_landmark_fields"
                    and source_config is not None):
                from .sic_landmark import SIC_CASE_LANDMARK_COLUMNS, SIC_CASE_SOURCE_COLUMNS

                table = source_config.tables.get("cases")
                columns = table.columns if table is not None else None
                configured_names = {
                    str(value.get("name")) for value in (columns or {}).values()
                    if isinstance(value, dict) and value.get("name")
                }
                if concept in SIC_CASE_LANDMARK_COLUMNS and set(SIC_CASE_SOURCE_COLUMNS) <= configured_names:
                    # These are the callback's registered inputs, not a claim that
                    # any particular raw row was extracted or validated.
                    sources.extend(SourceProvenance(
                        table="cases", value_variable=column, callback=source.callback,
                    ) for column in SIC_CASE_SOURCE_COLUMNS)
                    continue
            table_config = (
                source_config.tables.get(source.table)
                if source_config is not None and source.table else None
            )
            defaults = table_config.defaults if table_config is not None else None
            descriptions = source.params.get("code_descriptions") or {}
            codes = ()
            if re.search(r"icd|(?:^|_)code$", source.sub_var or "", re.I):
                codes = tuple(CodeProvenance(
                    code=str(code), description=descriptions.get(str(code)) or None,
                    description_status="recorded" if descriptions.get(str(code)) else "not_recorded",
                ) for code in (source.ids or ()))
            sources.append(SourceProvenance(
                table=source.table, selector_variable=source.sub_var,
                selector_regex=source.regex, item_ids=tuple(source.ids or ()),
                value_variable=source.value_var or (defaults.val_var if defaults else None),
                unit_variable=source.unit_var or (defaults.unit_var if defaults else None),
                time_variable=source.index_var or (defaults.index_var if defaults else None),
                callback=source.callback,
                duration_variable=source.dur_var or (defaults.dur_var if defaults else None),
                interval=source.interval.isoformat() if source.interval is not None else None,
                codes=codes, unit_conversions=_conversions(source.callback),
            ))
        contract = contracts.get(definition.clinical_contract_id)
        forwarded = re.findall(r"fwd_concept\('(.+?)'\)", definition.callback or "")
        inputs = tuple(dict.fromkeys([
            *definition.sub_concepts, *definition.depends_on, *forwarded,
        ]))
        projected[concept] = ConceptProvenance(
            concept_id=concept, status="recorded", source_database=database,
            dictionary_source_database=metadata.dictionary_source_database,
            definition=contract.definition if contract else definition.description,
            definition_source=contract.source_id if contract else definition.definition_source,
            definition_version=contract.version if contract else definition.definition_version,
            canonical_unit=(definition.units or [None])[0],
            derivation_callback=definition.callback, derivation_inputs=inputs,
            sources=tuple(sources),
        )
        pending.extend(inputs)
    dictionary_sha = metadata_digest(dictionary_payload)
    implementation_sha = resolver_implementation_digest()
    payload = dict(
        schema_version="easyicu.data-provenance-snapshot/1",
        dictionary_sha256=dictionary_sha,
        implementation_sha256=implementation_sha,
        resource_inputs_sha256=resource_inputs_sha256,
        source_config_sha256=source_config_sha256,
        resolver_inputs_sha256=metadata_digest({
            "dictionary_sha256": dictionary_sha, "source_database": database,
            "class_prefixes": prefixes, "implementation_sha256": implementation_sha,
            "resource_inputs_sha256": resource_inputs_sha256,
            "source_config_sha256": source_config_sha256,
            "parameters": _json_metadata(resolver_parameters or {}),
            "requested_concepts": list(concepts),
            "clinical_contracts": _json_metadata({key: asdict(value) for key, value in contracts.items()}),
        }),
        source_database=database, class_prefixes=prefixes,
        concepts=[value.model_dump(mode="json") for value in projected.values()],
        cohort_filters=[value.model_dump(mode="json") for value in cohort_filters],
        resolver_parameters=[CohortFilterProvenance(field=key, values=(value,)).model_dump(mode="json")
                             for key, value in (resolver_parameters or {}).items()
                             if isinstance(value, (str, int, float, bool))],
    )
    return DataProvenanceSnapshot(**payload, snapshot_sha256=metadata_digest(payload))
