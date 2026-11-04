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
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import pandas as pd

from ..config import DataSourceConfig
from .. import hospital_mortality
from ..hospital_mortality import EICU_HOSPITAL_STATUS_BINDING
from .expr_parser import _maybe_float, _maybe_int, _maybe_timedelta


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
            raise TypeError(
                "Concept source field 'sub_var' must be str or None, got bool "
                f"({sub_var!r}); check the concept dictionary for a YAML "
                "'sub_var: true/false' typo"
            )
        ids = payload.pop("ids", None)

        if ids is not None:
            if isinstance(ids, bool):
                raise TypeError(
                    "Concept source field 'ids' must be scalar or iterable, got bool "
                    f"({ids!r}); check the concept dictionary for an 'ids: yes/no' "
                    "YAML typo"
                )
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
            raise TypeError(
                "Concept source field 'value_var' must be str or None, got bool "
                f"({value_var!r}); check the concept dictionary for a YAML "
                "'value_var: true/false' typo"
            )
        unit_var = payload.pop("unit_var", payload.pop("unit", None))
        if isinstance(unit_var, bool):
            raise TypeError(
                "Concept source field 'unit_var' must be str or None, got bool "
                f"({unit_var!r}); check the concept dictionary for a YAML "
                "'unit_var: true/false' typo"
            )
        index_var = payload.pop("index_var", payload.pop("time_var", None))
        if isinstance(index_var, bool):
            raise TypeError(
                "Concept source field 'index_var' must be str or None, got bool "
                f"({index_var!r}); check the concept dictionary for a YAML "
                "'index_var: true/false' typo"
            )
        dur_var = payload.pop("dur_var", None)
        if isinstance(dur_var, bool):
            raise TypeError(
                "Concept source field 'dur_var' must be str or None, got bool "
                f"({dur_var!r}); check the concept dictionary for a YAML "
                "'dur_var: true/false' typo"
            )

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


#: List fields whose empty overlay value inherits the base definition.
_OVERLAY_EMPTY_LIST_INHERITS = frozenset({"sub_concepts", "levels", "depends_on"})


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
    clinical_status: Optional[str] = None
    canonical_definition: Optional[bool] = None
    requires_explicit_opt_in: Optional[bool] = None
    definition_source: Optional[str] = None
    definition_version: Optional[str] = None
    clinical_contract_id: Optional[str] = None
    pre_admission_lookback_hours: Optional[float] = None
    #: Ranked measurement channels (best first); see ``source_channel``.
    source_channel_priority: Optional[List[str]] = None
    #: Select channels within +/- this many hours instead of the clock hour.
    source_channel_window_hours: Optional[float] = None

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

        # Bind the shipped eICU status source without changing the sealed
        # dictionary or overriding a caller's explicit source callback.
        if name == "death":
            for database in ("eicu", "eicu_demo"):
                for source in sources.get(database, []):
                    if (
                        source.table == "patient"
                        and source.value_var == "hospitaldischargestatus"
                        and source.callback == "transform_fun(comp_na(`==`, 'Expired'))"
                    ):
                        source.callback = EICU_HOSPITAL_STATUS_BINDING
                        source.params.update(hospital_mortality.EICU_HOSPITAL_STATUS_PARAMETERS)

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
            clinical_status=payload.get("clinical_status"),
            canonical_definition=payload.get("canonical_definition"),
            requires_explicit_opt_in=payload.get("requires_explicit_opt_in"),
            definition_source=payload.get("definition_source"),
            definition_version=payload.get("definition_version"),
            clinical_contract_id=payload.get("clinical_contract_id"),
            pre_admission_lookback_hours=_maybe_float(
                payload.get("pre_admission_lookback_hours")
            ),
            family=payload.get("family"),
            depends_on=depends_list,
            source_channel_priority=(
                [str(item) for item in payload["source_channel_priority"]]
                if payload.get("source_channel_priority")
                else None
            ),
            source_channel_window_hours=_maybe_float(
                payload.get("source_channel_window_hours")
            ),
        )

    def for_data_source(self, config: DataSourceConfig) -> List[ConceptSource]:
        # ricu 语义: 源按 class 继承链 "most-specific match" 解析 —— 取继承链中
        # 第一个有定义的 class 的源, 而非把所有 class 的源做并集 (union)。
        # 旧实现 union 会在 demo 配置(class_prefix 非空)上把父类源叠加到子类源:
        #   - 重复计数 (如 mimic_demo + mimic 都定义 vaso 的 cv/mv -> 同一源被加多次);
        #   - 引入错误的父类表 (mimic_demo class_prefix 误含 miiv -> 单数 inputevents,
        #     而 MIMIC-III demo 只有 inputevents_cv/_mv)。
        # 非 demo 配置 class_prefix 为空, keys=[name], 行为与旧实现完全一致 (零影响)。
        seen: set = set()
        for key in [config.name, *config.class_prefix]:
            if key in seen:
                continue
            seen.add(key)
            if key in self.sources:
                return list(self.sources[key])
        return []


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
        """Patch concepts, merging source definitions by database key."""
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

            # Every field merges by one rule, so a field added to the
            # definition can never be dropped by an overlay that only edits
            # another one (``pre_admission_lookback_hours`` was, 11-03 review).
            merged_fields: Dict[str, object] = {"name": name, "sources": merged_sources}
            for item in fields(ConceptDefinition):
                if item.name in merged_fields:
                    continue
                new_value = getattr(incoming, item.name)
                old_value = getattr(current, item.name)
                if item.name in _OVERLAY_EMPTY_LIST_INHERITS:
                    # An empty list in an overlay means "not specified".
                    merged_fields[item.name] = copy.deepcopy(new_value if new_value else old_value)
                else:
                    merged_fields[item.name] = copy.deepcopy(new_value if new_value is not None else old_value)
            merged_definition = ConceptDefinition(**merged_fields)

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
        """Load JSON dictionaries using the same per-concept patch contract.

        Args:
            file_paths: Ordered JSON paths; later definitions patch earlier ones.

        Returns:
            The merged concept dictionary.

        Examples:
            >>> dict1 = ConceptDictionary.from_multiple_json([
            ...     'data/concept-dict.json',
            ...     'data/sofa2-dict.json'
            ... ])
        """
        merged: ConceptDictionary | None = None
        for file_path in file_paths:
            path = Path(file_path)
            with path.open("r", encoding="utf8") as handle:
                raw_dict = json.load(handle)
            incoming = cls.from_payload(raw_dict)
            if merged is None:
                merged = incoming
            else:
                merged.update(incoming)
        return merged if merged is not None else cls({})


# Legacy alias kept for backward compatibility with code that imports
# ``Concept`` directly from :mod:`easyicu.concept`.
Concept = ConceptDefinition


__all__ = [
    "ConceptSource",
    "ConceptDefinition",
    "ConceptDictionary",
    "Concept",
]
