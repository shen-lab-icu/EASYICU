"""Research facts from packaged identifiers and concept mappings, never ICU rows.

A mapping is evidence of vocabulary coverage, not proof that a local extraction
or an execution adapter works. Source-specific meanings below cite the mapping
and its interpreting owner; the dictionary's generic ``death`` label is not an
endpoint definition. Consumers must use these facts instead of database lists.
"""
from __future__ import annotations

from functools import lru_cache
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .profiles import get_database_profile, get_packaged_registry


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceField(Fact):
    source: str
    table: str
    column: str
    interpretation: str | None = None


class AnalysisUnit(Fact):
    unit: Literal["patient", "hospitalization", "icu_stay"]
    source: str
    table: str
    column: str


class NumericRepresentation(Fact):
    kind: Literal["exact", "binned", "capped_or_shifted", "unavailable"]
    source: str
    field: SourceField | None = None
    bin_edges: tuple[float, ...] = ()
    bin_labels: tuple[str, ...] = ()
    bin_values: tuple[int, ...] = ()
    exact_through: float | None = None


class MortalityCapability(Fact):
    meaning: Literal["hospital_death", "icu_death", "death_within_72h_after_icu_discharge"]
    status: SourceField
    event_time: SourceField | None
    time_kind: Literal["recorded", "date", "last_observation_proxy", "unavailable"]


class ConceptAvailability(Fact):
    concept: str
    # None defers a registered recipe to concept(), which resolves its owner.
    mapped: bool | None
    defined: bool
    source: str
    fields: tuple[SourceField, ...] = ()
    timed: bool = False
    category: str | None = None
    derivation_status: Literal["full", "degraded", "blocked"] | None = None
    derivation_reason_code: str | None = None


class MonitoringItem(Fact):
    item_id: int
    concept: str
    source: str


# The death callback uses the LAST TIME of these monitoring measurements. Their
# values do not encode death. Both identities are present in the sealed dictionary.
HIRID_TIMING_ITEMS = (
    MonitoringItem(item_id=110, concept="map", source="concept-dict.json#/map/sources/hirid/0/ids"),
    MonitoringItem(item_id=200, concept="hr", source="concept-dict.json#/hr/sources/hirid/0/ids"),
)


class ResearchCapabilityProfile(Fact):
    database: str
    schema_database: str
    source_registered: bool
    units: tuple[AnalysisUnit, ...]
    mortality: MortalityCapability | None
    mortality_outcomes: tuple[MortalityCapability, ...] = ()
    age: NumericRepresentation
    weight: NumericRepresentation
    icu_admission: SourceField | None
    icu_discharge: SourceField | None
    hospital_discharge: SourceField | None
    time_representation: Literal["shifted_timestamp", "icu_offset_minutes", "global_offset_milliseconds", "admission_offset_seconds"]
    time_source: str
    hospital_los: bool
    icu_los: bool
    admission_type: bool
    concepts: tuple[ConceptAvailability, ...]

    @property
    def analysis_units(self) -> tuple[str, ...]:
        return tuple(item.unit for item in self.units)

    def concept(self, name: str) -> ConceptAvailability:
        matched = next((item for item in self.concepts if item.concept == name), None)
        if matched is not None and matched.mapped is None:
            status, reason = _derivation_status(name, self.schema_database)
            return matched.model_copy(update={"mapped": status == "full", "derivation_status": status,
                                              "derivation_reason_code": reason})
        return matched if matched is not None else _loader_output(name, self.schema_database)


@lru_cache(maxsize=512)
def _derivation_status(name: str, database: str) -> tuple[str, str | None]:
    from easyicu.research_agent.concept_availability import explain_concept_availability

    availability = explain_concept_availability(concept=name, database=database)
    return availability.status, availability.reason


@lru_cache(maxsize=512)
def _loader_output(name: str, database: str) -> ConceptAvailability:
    """Ask the existing composite registry only for an actually requested output."""
    from easyicu.concept_output_sources import COMPOSITE_CONCEPT_OUTPUT_SOURCES

    if name in COMPOSITE_CONCEPT_OUTPUT_SOURCES:
        status, reason = _derivation_status(name, database)
        # Partial derivability cannot be presented as an exact definition.
        return ConceptAvailability(concept=name, mapped=status == "full", defined=True,
            source=f"easyicu.concept_output_sources.COMPOSITE_CONCEPT_OUTPUT_SOURCES#{name}",
            derivation_status=status, derivation_reason_code=reason)
    return ConceptAvailability(concept=name, mapped=False, defined=False, source="concept-dict.json+sofa2-dict.json")


@lru_cache(maxsize=1)
def _dictionary():
    from easyicu.resources import load_dictionary

    return load_dictionary(include_sofa2=True)


def _field(config, table: str, column: str, *, source: str | None = None,
           interpretation: str | None = None) -> SourceField | None:
    """An interpretation cannot make an absent registry column available."""
    registered = config.tables.get(table)
    columns = registered.columns or {} if registered else {}
    names = {str(key).casefold() for key in columns}
    names.update(str(item.get("name", "")).casefold() for item in columns.values() if isinstance(item, dict))
    if column.casefold() not in names:
        return None
    return SourceField(source=source or f"data-sources.json#/{config.name}/tables/{table}/cols/{column}",
                       table=table, column=column, interpretation=interpretation)


def _identifier_field(config, unit: str, attribute: str) -> SourceField | None:
    identifier = config.id_configs.get(unit)
    column = getattr(identifier, attribute, None)
    if identifier is None or not column:
        return None
    # eICU unitadmitoffset is a loader-created zero, not a raw patient column.
    # The identifier registry explicitly declares it as the ICU origin.
    return SourceField(source=f"data-sources.json#/{config.name}/id_cfg/{unit}/{attribute}",
                       table=identifier.table, column=column)


def _concepts(config) -> tuple[ConceptAvailability, ...]:
    dictionary = _dictionary()
    seen: dict[str, ConceptAvailability] = {}

    def read(name: str, visiting: frozenset[str] = frozenset()) -> ConceptAvailability:
        if name in seen:
            return seen[name]
        origin = f"concept-dict.json+sofa2-dict.json#/{name}"
        definition = dictionary.get(name)
        if definition is None or name in visiting:
            return ConceptAvailability(concept=name, mapped=False, defined=definition is not None, source=origin)
        sources = definition.for_data_source(config)
        fields = []
        timed = False
        for item in sources:
            table = config.tables.get(item.table)
            if table is not None:
                column = item.value_var or table.defaults.val_var or item.sub_var or ""
                fields.append(SourceField(source=f"{origin}/sources/{config.name}", table=item.table, column=column))
                timed |= bool(item.index_var or table.defaults.index_var)
        dependencies = tuple(dict.fromkeys((*definition.sub_concepts, *definition.depends_on)))
        children = [read(child, visiting | {name}) for child in dependencies]
        mapped = bool(sources) if sources or not dependencies else None
        result = ConceptAvailability(concept=name, mapped=mapped, defined=True, source=origin,
                                     fields=tuple(fields), timed=timed or bool(children) and all(child.timed for child in children),
                                     category=definition.category)
        seen[name] = result
        return result

    mapped = {name: read(name) for name in sorted(dictionary.keys())}
    # Derived outcome availability has its own existing registry and builders.
    # It must not be confused with an absent raw dictionary mapping.
    from easyicu.outcome_availability import OUTCOME_CONCEPT_SUPPORTED_DATABASES

    for name, databases in OUTCOME_CONCEPT_SUPPORTED_DATABASES.items():
        mapped[name] = ConceptAvailability(
            concept=name, mapped=config.name in databases, timed=config.name in databases, defined=True,
            source=f"easyicu.outcome_availability.OUTCOME_CONCEPT_SUPPORTED_DATABASES#{name}", category="outcome",
        )
    return tuple(mapped[name] for name in sorted(mapped))


def _numeric(config, name: str) -> NumericRepresentation:
    definition = _dictionary()[name]
    entries = definition.for_data_source(config)
    source = f"concept-dict.json#/{name}/sources/{config.name}"
    if not entries:
        return NumericRepresentation(kind="unavailable", source=source)
    entry = entries[0]
    field = _field(config, entry.table, entry.value_var or entry.sub_var or "", source=source)
    labels = tuple(re.findall(r"`(\d+(?:-\d+|\+|-))`", entry.callback or ""))
    if labels:
        edges = tuple(float(label.split("-")[0].rstrip("+")) for label in labels if not label.endswith("-"))
        values = tuple(int(float(value)) for value in re.findall(r"`\d+(?:-\d+|\+|-)`\s*=\s*([\d.]+)", entry.callback or ""))
        return NumericRepresentation(kind="binned", source=source, field=field,
                                     bin_labels=labels, bin_edges=edges, bin_values=values)
    capped = name == "age" and (
        entry.value_var == "anchor_age" or "mimic_age" in (entry.callback or "")
        or "eicu_age" in (entry.callback or "")
    )
    return NumericRepresentation(kind="capped_or_shifted" if capped else "exact", source=source,
                                 field=field, exact_through=89 if capped else None)


def _mortality(config) -> MortalityCapability | None:
    entries = _dictionary()["death"].for_data_source(config)
    if not entries:
        return None
    item = entries[0]
    origin = f"concept-dict.json#/death/sources/{config.name}/0"
    interpretation = "easyicu.concept.callback_apply._apply_callback"
    if item.callback == "hirid_death":
        status = _field(config, "general", "discharge_status", source=origin, interpretation=interpretation)
        timing = _field(config, "observations", "datetime", source=origin, interpretation=interpretation)
        meaning, kind = "icu_death", "last_observation_proxy"
    elif item.callback == "aumc_death":
        status = _field(config, item.table, item.value_var, source=origin, interpretation=interpretation)
        timing = _field(config, item.table, item.index_var, source=origin, interpretation=interpretation)
        meaning, kind = "death_within_72h_after_icu_discharge", "date"
    elif item.callback == "sic_death":
        status = _field(config, item.table, "HospitalDischargeType", source=origin, interpretation=interpretation)
        timing = _field(config, item.table, item.index_var, source=origin, interpretation=interpretation)
        meaning, kind = "hospital_death", "recorded"
    else:
        status = _field(config, item.table, item.value_var, source=origin)
        # unitdischargeoffset indexes a hospital STATUS; it is not death time.
        timing = (_field(config, item.table, item.index_var, source=origin)
                  if item.index_var == "deathtime" else None)
        meaning, kind = "hospital_death", "recorded" if timing else "unavailable"
    return (MortalityCapability(meaning=meaning, status=status, event_time=timing, time_kind=kind)
            if status else None)


@lru_cache(maxsize=32)
def research_capability_profile(database: str) -> ResearchCapabilityProfile:
    """Describe a registered source, or miiv_demo's explicitly inherited schema.

    The IV demo is used by research callers but is absent from the packaged
    registry. Schema inheritance must not claim registration or runnable data.
    """
    registered = database != "miiv_demo"
    profile = get_database_profile(database if registered else "miiv")
    config = get_packaged_registry().get(profile.key)
    parent = profile.parent_key or profile.key
    unit_names = {"patient": "patient", "hadm": "hospitalization", "icustay": "icu_stay"}
    units = tuple(AnalysisUnit(unit=unit_names[name], source=f"data-sources.json#/{config.name}/id_cfg/{name}",
                               table=item.table, column=item.id)
                  for name, item in config.id_configs.items() if name in unit_names)
    start = _identifier_field(config, "icustay", "start")
    icu_end = _identifier_field(config, "icustay", "end")
    hospital_end = _identifier_field(config, "hadm", "end")
    mortality = _mortality(config)
    outcomes = [mortality] if mortality else []
    # eICU independently records ICU discharge status. A hospital status's
    # index is still not a recorded time of death for either endpoint.
    icu_status = _field(config, "patient", "unitdischargestatus")
    if icu_status is not None:
        outcomes.append(MortalityCapability(meaning="icu_death", status=icu_status, event_time=None, time_kind="unavailable"))
    # These named builders do not relabel the dictionary's generic death.
    native = {"aumc": ("admissions", "destination", None),
              "sic": ("cases", "OffsetOfDeath", "OffsetOfDeath"),
              "mimic": ("admissions", "deathtime", "deathtime"),
              "miiv": ("admissions", "deathtime", "deathtime")}.get(parent)
    if native is not None:
        table, column, timing = native
        status = _field(config, table, column, interpretation="easyicu.scores.icu_mortality.load_icu_death")
        if status is not None:
            outcomes.append(MortalityCapability(meaning="icu_death", status=status,
                event_time=_field(config, table, timing) if timing else None,
                time_kind="recorded" if timing else "unavailable"))
    concepts = _concepts(config)
    mapped = {item.concept for item in concepts if item.mapped}
    # These are source clock declarations, grounded in the registry's typed
    # columns and their loader conversion owners; they never infer a unit from
    # a numeric value. SIC uses primary-admission seconds, subtract ICUOffset.
    clocks = {"eicu": "icu_offset_minutes", "aumc": "global_offset_milliseconds", "sic": "admission_offset_seconds"}
    return ResearchCapabilityProfile(
        database=profile.key if registered else database, schema_database=config.name,
        source_registered=registered, units=units, mortality=mortality, mortality_outcomes=tuple(outcomes),
        age=_numeric(config, "age"), weight=_numeric(config, "weight"),
        icu_admission=start, icu_discharge=icu_end, hospital_discharge=hospital_end,
        time_representation=clocks.get(parent, "shifted_timestamp"),
        time_source=f"data-sources.json#/{config.name}/id_cfg/icustay;easyicu.database_config.TIME_IS_OFFSET",
        hospital_los="los_hosp" in mapped and hospital_end is not None,
        icu_los="los_icu" in mapped and icu_end is not None,
        admission_type="adm" in mapped, concepts=concepts,
    )
