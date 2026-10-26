"""Bounded metadata projection; mapping support is not execution acceptance.

The catalogue owns the concept vocabulary and units; db6a owns capability
decisions. Stored source registrations supply identity only, with no discovery,
manifest inspection, observed coverage, or patient rows. Family assessments
use the owner's minimum requirements, not an invented example study.
"""

from easyicu.concept import catalog
from easyicu.databases.profiles import get_database_profile, iter_database_profiles
from easyicu.databases.research_capabilities import ResearchCapabilityProfile, research_capability_profile
from easyicu.databases.research_requirements import StudyRequirements, assess_study, family_requirements
from easyicu.research_agent.planning.analysis_types import list_analysis_types
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver import sources
from easyicu.webserver.catalog_metadata import catalog_definitions, concept_audience
from easyicu.webserver.data_map_contracts import (
    ConceptCell, DatabaseCapabilities, DatabaseMapRow, DataMap, RegisteredSource, StudyCell,
)


class DataMapError(ValueError):
    """A bounded read selector cannot be resolved by its metadata owner."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _page(offset: int, limit: int) -> None:
    if type(offset) is not int or type(limit) is not int or not 0 <= offset <= 10000 or not 1 <= limit <= 100:
        raise DataMapError("data_map_invalid_page")


def _next_offset(offset: int, limit: int, total: int) -> int | None:
    return offset + limit if offset + limit < total else None


def _concept_cell(profile: ResearchCapabilityProfile, name: str, definitions: dict) -> ConceptCell:
    fact = profile.concept(name)
    representation = getattr(profile, name) if name in {"age", "weight"} else None
    substitute = (representation if fact.mapped and representation is not None
                  and representation.kind not in {"exact", "unavailable"} else None)
    # The catalogue calls death "In-hospital Mortality". Retain the capability
    # owner's actual endpoint, as the existing study-requirements owner does.
    if (name == "death" and fact.mapped and profile.mortality is not None
            and profile.mortality.meaning != "hospital_death"):
        substitute = profile.mortality
    status = "substituted" if substitute is not None else "available" if fact.mapped else "missing"
    return ConceptCell(
        concept=name, unit=catalog.CONCEPT_DICTIONARY[name][2],
        audience=concept_audience(name, definitions.get(name, {})),
        status=status, capability=fact, substitute=substitute,
        reason_code=(fact.derivation_reason_code or "concept_mapping_unavailable") if status == "missing" else None,
    )


def _study_cell(profile: ResearchCapabilityProfile, family: str) -> StudyCell:
    """Time-to-event cells assess the conversation's default ``death`` request.

    Generic death retains its source meaning and hospital follow-up requirement,
    even where a named ICU endpoint is also available. Use the conversation's
    requirements owner so the map cannot silently substitute ICU censoring.
    """
    from easyicu.webserver.research_capability_scope import study_requirements

    requirements = (study_requirements({"execution_concepts": {"outcome": "death"}}, family=family)
                    if "event_time" in family_requirements(family) else StudyRequirements(family=family))
    decision = assess_study(profile, requirements)
    return StudyCell(family=family, status="refused" if decision.status == "refusal" else decision.status,
                     decision=decision)


def _registered_sources() -> dict[str, tuple[RegisteredSource, ...]]:
    grouped: dict[str, list[RegisteredSource]] = {}
    for record in sources.stored_registration_metadata():
        identity = SourceIdentity.from_record({**record, "registration_id": record.get("id")})
        if identity is None:
            continue
        try:
            profile = get_database_profile(identity.database)
        except KeyError:
            continue
        demo = sources.is_demo_source(record)
        kind = "synthetic" if record.get("source_kind") == "synthetic" else "public_demo" if demo else "unrecorded"
        grouped.setdefault(profile.parent_key or profile.key, []).append(
            RegisteredSource(identity=identity, kind=kind, is_demo=demo),
        )
    return {key: tuple(sorted(rows, key=lambda row: row.identity.registration_id)) for key, rows in grouped.items()}


def build_data_map(
    *, database: str | None = None, concept: str | None = None, family: str | None = None,
    concept_offset: int = 0, concept_limit: int = 40, source_offset: int = 0, source_limit: int = 20,
) -> DataMap:
    """Read one concept page across public databases and minimum study families.

    Database aliases follow profiles; concept/family filters use catalogue keys.
    Source pages contain saved registration facts, not claims of present files,
    current data quality, full-database population, or real-data acceptance.
    """
    _page(concept_offset, concept_limit)
    _page(source_offset, source_limit)
    profiles = iter_database_profiles(public_only=True)
    if database is not None:
        try:
            selected = get_database_profile(database)
        except KeyError:
            raise DataMapError("data_map_unknown_database") from None
        if not selected.is_public:
            raise DataMapError("data_map_database_not_public")
        profiles = (selected,)
    concepts = sorted(catalog.CONCEPT_DICTIONARY)
    if concept is not None:
        if concept not in concepts:
            raise DataMapError("data_map_unknown_concept")
        concepts = [concept]
    families = [spec.key for spec in list_analysis_types()]
    if family is not None:
        if family not in families:
            raise DataMapError("data_map_unknown_family")
        families = [family]
    selected_concepts = concepts[concept_offset:concept_offset + concept_limit]
    definitions = catalog_definitions()
    registrations = _registered_sources()
    rows = []
    for profile in profiles:
        facts = research_capability_profile(profile.key)
        registered = registrations.get(profile.key, ())
        rows.append(DatabaseMapRow(
            database=profile.key, profile=profile,
            capabilities=DatabaseCapabilities.model_validate(facts.model_dump(exclude={"concepts"})),
            concepts=tuple(_concept_cell(facts, name, definitions) for name in selected_concepts),
            studies=tuple(_study_cell(facts, name) for name in families),
            sources=registered[source_offset:source_offset + source_limit], source_total=len(registered),
            source_next_offset=_next_offset(source_offset, source_limit, len(registered)),
        ))
    return DataMap(
        databases=tuple(rows), concept_total=len(concepts), concept_offset=concept_offset, concept_limit=concept_limit,
        concept_next_offset=_next_offset(concept_offset, concept_limit, len(concepts)),
        source_offset=source_offset, source_limit=source_limit,
    )
