"""Concept dictionary data classes.

Extracted from :mod:`easyicu.concept` (2026-05-17) as part of the
Phase-1 split documented in CLAUDE.md. The three dataclasses below
(``ConceptSource`` / ``ConceptDefinition`` / ``ConceptDictionary``)
together with the legacy ``Concept`` alias used to live near the top
of ``concept.py`` (former lines ~215-495) plus the alias at the very
bottom of the file.

Why a separate module
---------------------
These are pure data containers. They depend on:

* :mod:`easyicu.config` for ``DataSourceConfig`` (only used by
  :meth:`ConceptDefinition.for_data_source`);
* :mod:`easyicu.concept_expr_parser` for ``_maybe_float`` /
  ``_maybe_int`` / ``_maybe_timedelta`` (coercion of JSON-loaded values
  in the ``from_*`` factory methods).

They do NOT depend on ``ConceptResolver``, on the load / aggregation
machinery in ``concept.py``, or on any IO. Keeping them in a small
module makes the schema reusable from downstream extractions (e.g.
the SOFA-2 standalone subset under ``其他文件/sofa2_core_code/``)
without dragging the whole resolver in.

Public surface
--------------
All names below are re-exported by :mod:`easyicu.concept`, so existing
``from easyicu.concept import ConceptSource, ConceptDefinition,
ConceptDictionary, Concept`` keeps working.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import pandas as pd

from .config import DataSourceConfig
from .concept_expr_parser import _maybe_float, _maybe_int, _maybe_timedelta


@dataclass
class ConceptSource:
    """Describe how to load a concept for a specific data source."""

    table: Optional[str] = None
    sub_var: Optional[str] = None
    ids: Optional[List[object]] = None
    value_var: Optional[str] = None
    unit_var: Optional[str] = None
    index_var: Optional[str] = None
    dur_var: Optional[str] = None  # 持续时间列，可能是duration或endtime
    regex: Optional[str] = None
    class_name: Optional[str] = None
    callback: Optional[str] = None
    interval: Optional[pd.Timedelta] = None
    target: Optional[str] = None
    params: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "ConceptSource":
        payload = dict(mapping)

        table = payload.pop("table", None)
        sub_var = payload.pop("sub_var", None)
        if isinstance(sub_var, bool):
            sub_var = None
        ids = payload.pop("ids", None)

        if ids is not None:
            if isinstance(ids, bool):
                ids_list = None
            elif isinstance(ids, (str, int, float)):
                ids_list = [ids]
            elif isinstance(ids, Iterable):
                ids_list = list(ids)
            else:
                raise TypeError("Concept source 'ids' must be scalar or iterable")
        else:
            ids_list = None

        value_var = payload.pop("value_var", payload.pop("val_var", None))
        if isinstance(value_var, bool):
            value_var = None
        unit_var = payload.pop("unit_var", payload.pop("unit", None))
        if isinstance(unit_var, bool):
            unit_var = None
        index_var = payload.pop("index_var", payload.pop("time_var", None))
        if isinstance(index_var, bool):
            index_var = None
        dur_var = payload.pop("dur_var", None)
        if isinstance(dur_var, bool):
            dur_var = None

        regex = payload.pop("regex", None)
        class_name = payload.pop("class", payload.pop("class_name", None))
        callback = payload.pop("callback", None)
        interval = payload.pop("interval", None)
        target = payload.pop("target", None)

        return cls(
            table=str(table) if table is not None else None,
            sub_var=str(sub_var) if sub_var is not None else None,
            ids=ids_list,
            value_var=str(value_var) if value_var is not None else None,
            unit_var=str(unit_var) if unit_var is not None else None,
            index_var=str(index_var) if index_var is not None else None,
            dur_var=str(dur_var) if dur_var is not None else None,
            regex=str(regex) if regex is not None else None,
            class_name=str(class_name) if class_name is not None else None,
            callback=str(callback) if callback is not None else None,
            interval=_maybe_timedelta(interval),
            target=str(target) if target is not None else None,
            params=payload,
        )


@dataclass
class ConceptDefinition:
    """Full description of a concept across multiple data sources."""

    name: str
    sources: Dict[str, List[ConceptSource]]
    units: Optional[List[str]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    description: Optional[str] = None
    category: Optional[str] = None
    target: Optional[str] = None
    interval: Optional[pd.Timedelta] = None
    aggregate: Optional[object] = None
    class_name: Optional[str] = None
    callback: Optional[str] = None
    sub_concepts: List[str] = field(default_factory=list)
    family: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    levels: Optional[List[object]] = None
    keep_components: Optional[bool] = None
    omop_id: Optional[int] = None

    @classmethod
    def from_name_and_payload(
        cls,
        name: str,
        payload: Mapping[str, object],
    ) -> "ConceptDefinition":
        raw_sources = payload.get("sources", {})
        sources: Dict[str, List[ConceptSource]] = {}
        for src_name, entries in raw_sources.items():
            sources[src_name] = [
                ConceptSource.from_mapping(entry) for entry in entries
            ]

        unit_value = payload.get("unit")
        if isinstance(unit_value, str):
            units: Optional[List[str]] = [unit_value]
        elif isinstance(unit_value, Iterable):
            units = [str(item) for item in unit_value]
        else:
            units = None

        raw_concepts = payload.get("concepts")
        if raw_concepts is None:
            sub_concepts: List[str] = []
        elif isinstance(raw_concepts, (list, tuple)):
            sub_concepts = [str(item) for item in raw_concepts]
        else:
            sub_concepts = [str(raw_concepts)]

        depends_raw = payload.get("depends_on", [])
        if isinstance(depends_raw, str):
            depends_list = [depends_raw]
        elif isinstance(depends_raw, Iterable):
            depends_list = [str(item) for item in depends_raw]
        else:
            depends_list = []

        return cls(
            name=name,
            sources=sources,
            units=units,
            minimum=_maybe_float(payload.get("min")),
            maximum=_maybe_float(payload.get("max")),
            description=payload.get("description"),
            category=payload.get("category"),
            target=payload.get("target"),
            interval=_maybe_timedelta(payload.get("interval")),
            aggregate=payload.get("aggregate"),
            class_name=payload.get("class") or payload.get("class_name"),
            callback=payload.get("callback"),
            sub_concepts=sub_concepts,
            levels=payload.get("levels"),
            keep_components=payload.get("keep_components"),
            omop_id=_maybe_int(payload.get("omopid")),
            family=payload.get("family"),
            depends_on=depends_list,
        )

    def for_data_source(self, config: DataSourceConfig) -> List[ConceptSource]:
        candidates: List[ConceptSource] = []
        keys = [config.name, *config.class_prefix]
        for key in keys:
            if key in self.sources:
                candidates.extend(self.sources[key])
        return candidates


class ConceptDictionary:
    """Container for all concept definitions."""

    def __init__(self, concepts: Mapping[str, ConceptDefinition]):
        self._concepts = dict(concepts)

    def __contains__(self, name: object) -> bool:
        return name in self._concepts

    def __getitem__(self, name: str) -> ConceptDefinition:
        return self._concepts[name]

    def get(self, name: str, default=None) -> Optional[ConceptDefinition]:
        """Get a concept by name, returning default if not found."""
        return self._concepts.get(name, default)

    def items(self):
        return self._concepts.items()

    def keys(self):
        return self._concepts.keys()

    def values(self):
        return self._concepts.values()

    def copy(self) -> "ConceptDictionary":
        """Create a shallow copy of this dictionary."""
        return ConceptDictionary(self._concepts.copy())

    def update(self, other: "ConceptDictionary") -> None:
        """Merge another dictionary into this one with per-concept granularity."""
        if not isinstance(other, ConceptDictionary):
            raise TypeError("Can only update from another ConceptDictionary")

        for name, incoming in other._concepts.items():
            if name not in self._concepts:
                self._concepts[name] = incoming
                continue

            current = self._concepts[name]

            merged_sources: Dict[str, List[ConceptSource]] = copy.deepcopy(current.sources)
            for source_name, entries in incoming.sources.items():
                merged_sources[source_name] = copy.deepcopy(entries)

            def _pick(new_value, old_value, *, allow_empty: bool = False):
                if allow_empty:
                    return copy.deepcopy(new_value) if new_value is not None else copy.deepcopy(old_value)
                if isinstance(new_value, list):
                    return copy.deepcopy(new_value) if new_value else copy.deepcopy(old_value)
                return new_value if new_value not in (None,) else old_value

            merged_definition = ConceptDefinition(
                name=name,
                sources=merged_sources,
                units=_pick(incoming.units, current.units, allow_empty=True),
                minimum=incoming.minimum if incoming.minimum is not None else current.minimum,
                maximum=incoming.maximum if incoming.maximum is not None else current.maximum,
                description=incoming.description if incoming.description is not None else current.description,
                category=incoming.category if incoming.category is not None else current.category,
                target=incoming.target if incoming.target is not None else current.target,
                interval=incoming.interval if incoming.interval is not None else current.interval,
                aggregate=incoming.aggregate if incoming.aggregate is not None else current.aggregate,
                class_name=incoming.class_name if incoming.class_name is not None else current.class_name,
                callback=incoming.callback if incoming.callback is not None else current.callback,
                sub_concepts=_pick(incoming.sub_concepts, current.sub_concepts),
                levels=_pick(incoming.levels, current.levels),
                keep_components=(
                    incoming.keep_components
                    if incoming.keep_components is not None
                    else current.keep_components
                ),
                omop_id=incoming.omop_id if incoming.omop_id is not None else current.omop_id,
                family=incoming.family if incoming.family is not None else current.family,
                depends_on=_pick(incoming.depends_on, current.depends_on, allow_empty=True),
            )

            self._concepts[name] = merged_definition

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ConceptDictionary":
        concepts = {
            name: ConceptDefinition.from_name_and_payload(name, definition)
            for name, definition in payload.items()
        }
        return cls(concepts)

    @classmethod
    def from_json(cls, file_path: str | Path) -> "ConceptDictionary":
        path = Path(file_path)
        with path.open("r", encoding="utf8") as handle:
            raw_dict = json.load(handle)
        return cls.from_payload(raw_dict)

    @classmethod
    def from_multiple_json(cls, file_paths: List[str | Path]) -> "ConceptDictionary":
        """从多个 JSON 文件加载概念字典并合并

        Args:
            file_paths: JSON 文件路径列表，后面的文件会覆盖前面的同名概念

        Returns:
            合并后的概念字典

        Examples:
            >>> dict1 = ConceptDictionary.from_multiple_json([
            ...     'data/concept-dict.json',
            ...     'data/sofa2-dict.json'
            ... ])
        """
        merged_payload = {}
        for file_path in file_paths:
            path = Path(file_path)
            with path.open("r", encoding="utf8") as handle:
                raw_dict = json.load(handle)
            # 合并，后面的覆盖前面的
            merged_payload.update(raw_dict)
        return cls.from_payload(merged_payload)


# Legacy alias kept for backward compatibility with code that imports
# ``Concept`` directly from :mod:`easyicu.concept`.
Concept = ConceptDefinition


__all__ = [
    "ConceptSource",
    "ConceptDefinition",
    "ConceptDictionary",
    "Concept",
]
