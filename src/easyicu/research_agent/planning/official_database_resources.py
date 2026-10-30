"""Compatibility exports for versioned official database resource identities."""

from easyicu.databases.profiles import normalize_database_key as normalize_database_key

from .literature_contract import (
    SourceResourceIdentity as SourceResourceIdentity,
    OfficialResourceEntry as OfficialResourceEntry,
    OFFICIAL_RESOURCE_REGISTRIES as OFFICIAL_RESOURCE_REGISTRIES,
    CURRENT_OFFICIAL_SOURCE_REGISTRY_VERSION as CURRENT_OFFICIAL_SOURCE_REGISTRY_VERSION,
    canonical_resource_database as canonical_resource_database,
    match_official_source_identity as match_official_source_identity,
    _SICDB_REFERENCE as _SICDB_REFERENCE,
    _MIMIC_31_REFERENCE as _MIMIC_31_REFERENCE,
    _MIMIC_22_REFERENCE as _MIMIC_22_REFERENCE,
    _VERSION_1 as _VERSION_1,
    _VERSION_2 as _VERSION_2,
    _words as _words,
    _doi as _doi,
)
