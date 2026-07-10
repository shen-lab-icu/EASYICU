"""Canonical database metadata and identifier profiles."""

from .profiles import (
    DATABASE_ALIASES,
    DATABASE_ID_CONFIG,
    DATABASE_LABELS,
    DatabaseAliasesView,
    DatabaseIdConfigView,
    DatabaseLabelsView,
    DatabaseProfile,
    DatabaseProfileMetadata,
    get_database_profile,
    get_packaged_registry,
    iter_database_profiles,
    normalize_database_key,
    public_database_keys,
)

__all__ = [
    "DATABASE_ALIASES",
    "DATABASE_ID_CONFIG",
    "DATABASE_LABELS",
    "DatabaseAliasesView",
    "DatabaseIdConfigView",
    "DatabaseLabelsView",
    "DatabaseProfile",
    "DatabaseProfileMetadata",
    "get_database_profile",
    "get_packaged_registry",
    "iter_database_profiles",
    "normalize_database_key",
    "public_database_keys",
]
