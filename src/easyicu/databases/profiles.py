"""Typed, registry-backed metadata for supported ICU databases.

The packaged ``data-sources.json`` registry owns public display metadata and
identifier configuration.  This module turns those two pieces into one typed
profile without importing the resource loader until a profile is requested.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from easyicu.config import DataSourceConfig, DataSourceRegistry


class DatabaseProfileMetadata(BaseModel):
    """Display metadata declared by a public data-source registry entry."""

    display_name: str
    aliases: tuple[str, ...] = Field(default_factory=tuple)
    display_order: int

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("display_name")
    @classmethod
    def _require_display_name(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("database display_name must be non-empty")
        return text

    @field_validator("aliases", mode="before")
    @classmethod
    def _normalize_aliases(cls, value: object) -> tuple[str, ...]:
        return _coerce_aliases(value)


class DatabaseProfile(DatabaseProfileMetadata):
    """Canonical metadata and ICU-stay identifier contract for one database."""

    key: str
    stay_table: str
    stay_id_col: str
    is_public: bool = True
    parent_key: str | None = None

    model_config = ConfigDict(frozen=True)

    @field_validator("key", "stay_table", "stay_id_col")
    @classmethod
    def _require_text(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("database profile text fields must be non-empty")
        return text

    @field_validator("aliases", mode="before")
    @classmethod
    def _normalize_aliases(cls, value: object) -> tuple[str, ...]:
        return _coerce_aliases(value)


def _coerce_aliases(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("database profile aliases must be iterable") from exc
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        alias = str(item).strip()
        marker = alias.casefold()
        if alias and marker not in seen:
            seen.add(marker)
            out.append(alias)
    return tuple(out)


@lru_cache(maxsize=1)
def get_packaged_registry() -> DataSourceRegistry:
    """Load the packaged registry on first use, then reuse it process-wide."""

    # Local import is intentional: resources imports config, while config
    # re-exports the lazy compatibility mapping defined below.
    from easyicu.resources import load_data_sources

    return load_data_sources()


def iter_database_profiles(
    *,
    registry: DataSourceRegistry | None = None,
    public_only: bool = False,
) -> tuple[DatabaseProfile, ...]:
    """Return profiles ordered by display order and canonical key."""

    active_registry = registry or get_packaged_registry()
    configs = {config.name: config for config in active_registry}
    profiles: dict[str, DatabaseProfile] = {}

    for key, config in configs.items():
        raw = _declared_profile(config)
        if raw is None:
            continue
        profiles[key] = _build_declared_profile(config, raw)

    for key, config in configs.items():
        if key in profiles:
            continue
        if not key.endswith("_demo"):
            raise ValueError(
                f"public data source '{key}' must declare a database profile"
            )
        parent_key = key.removesuffix("_demo")
        parent = profiles.get(parent_key)
        if parent is None:
            raise ValueError(
                f"demo data source '{key}' has no profiled parent '{parent_key}'"
            )
        profiles[key] = _build_demo_profile(config, parent)

    ordered = tuple(
        sorted(profiles.values(), key=lambda row: (row.display_order, row.key))
    )
    if public_only:
        return tuple(profile for profile in ordered if profile.is_public)
    return ordered


def get_database_profile(
    database: str,
    *,
    registry: DataSourceRegistry | None = None,
) -> DatabaseProfile:
    """Resolve a canonical key, alias, or display label to one profile."""

    profiles = iter_database_profiles(registry=registry)
    lookup = _profile_lookup(profiles)
    raw = str(database).strip()
    for token in _lookup_tokens(raw):
        profile = lookup.get(token)
        if profile is not None:
            return profile
    raise KeyError(f"Unknown database profile '{database}'")


def normalize_database_key(
    database: str,
    *,
    registry: DataSourceRegistry | None = None,
) -> str:
    """Return the canonical registry key for a key, alias, or display name."""

    return get_database_profile(database, registry=registry).key


def public_database_keys(
    *, registry: DataSourceRegistry | None = None
) -> tuple[str, ...]:
    """Return public database keys in canonical display order."""

    return tuple(
        profile.key
        for profile in iter_database_profiles(registry=registry, public_only=True)
    )


class DatabaseIdConfigView(Mapping[str, Mapping[str, str]]):
    """Read-only, lazy compatibility view for the historical ID mapping."""

    def __init__(self, *, registry: DataSourceRegistry | None = None) -> None:
        self._registry = registry

    def __getitem__(self, key: str) -> Mapping[str, str]:
        profile = get_database_profile(key, registry=self._registry)
        return MappingProxyType(
            {"table": profile.stay_table, "id_col": profile.stay_id_col}
        )

    def __iter__(self) -> Iterator[str]:
        yield from (
            profile.key for profile in iter_database_profiles(registry=self._registry)
        )

    def __len__(self) -> int:
        return len(iter_database_profiles(registry=self._registry))


class DatabaseAliasesView(Mapping[str, tuple[str, ...]]):
    """Read-only, lazy aliases keyed by canonical database name."""

    def __init__(
        self,
        *,
        registry: DataSourceRegistry | None = None,
        public_only: bool = False,
    ) -> None:
        self._registry = registry
        self._public_only = public_only

    def __getitem__(self, key: str) -> tuple[str, ...]:
        profile = get_database_profile(key, registry=self._registry)
        if self._public_only and not profile.is_public:
            raise KeyError(key)
        return profile.aliases

    def __iter__(self) -> Iterator[str]:
        yield from (
            profile.key
            for profile in iter_database_profiles(
                registry=self._registry,
                public_only=self._public_only,
            )
        )

    def __len__(self) -> int:
        return len(
            iter_database_profiles(
                registry=self._registry,
                public_only=self._public_only,
            )
        )


class DatabaseLabelsView(Mapping[str, str]):
    """Read-only, lazy display labels keyed by canonical database name."""

    def __init__(
        self,
        *,
        registry: DataSourceRegistry | None = None,
        public_only: bool = True,
    ) -> None:
        self._registry = registry
        self._public_only = public_only

    def __getitem__(self, key: str) -> str:
        profile = get_database_profile(key, registry=self._registry)
        if self._public_only and not profile.is_public:
            raise KeyError(key)
        return profile.display_name

    def __iter__(self) -> Iterator[str]:
        yield from (
            profile.key
            for profile in iter_database_profiles(
                registry=self._registry,
                public_only=self._public_only,
            )
        )

    def __len__(self) -> int:
        return len(
            iter_database_profiles(
                registry=self._registry,
                public_only=self._public_only,
            )
        )


DATABASE_ID_CONFIG: Mapping[str, Mapping[str, str]] = DatabaseIdConfigView()
DATABASE_ALIASES: Mapping[str, tuple[str, ...]] = DatabaseAliasesView()
DATABASE_LABELS: Mapping[str, str] = DatabaseLabelsView()


def _declared_profile(config: DataSourceConfig) -> Mapping[str, Any] | None:
    declared = getattr(config, "profile", None)
    if declared is not None:
        return declared.model_dump(mode="python")
    raw = config.extra.get("profile")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError(f"database profile for '{config.name}' must be a mapping")
    return raw


def _build_declared_profile(
    config: DataSourceConfig,
    raw: Mapping[str, Any],
) -> DatabaseProfile:
    stay_table, stay_id_col = _stay_identifier(config)
    return DatabaseProfile(
        key=config.name,
        display_name=raw.get("display_name"),
        aliases=raw.get("aliases") or (),
        display_order=raw.get("display_order"),
        stay_table=stay_table,
        stay_id_col=stay_id_col,
        is_public=True,
    )


def _build_demo_profile(
    config: DataSourceConfig,
    parent: DatabaseProfile,
) -> DatabaseProfile:
    stay_table, stay_id_col = _stay_identifier(config)
    return DatabaseProfile(
        key=config.name,
        display_name=f"{parent.display_name} Demo",
        aliases=(config.name, config.name.replace("_", "-")),
        display_order=parent.display_order + 1000,
        stay_table=stay_table,
        stay_id_col=stay_id_col,
        is_public=False,
        parent_key=parent.key,
    )


def _stay_identifier(config: DataSourceConfig) -> tuple[str, str]:
    try:
        icustay = config.id_configs["icustay"]
    except KeyError as exc:
        raise ValueError(
            f"data source '{config.name}' must declare id_cfg.icustay"
        ) from exc
    table = str(icustay.table or "").strip()
    id_col = str(icustay.id or "").strip()
    if not table or not id_col:
        raise ValueError(
            f"data source '{config.name}' has incomplete id_cfg.icustay metadata"
        )
    return table, id_col


def _profile_lookup(
    profiles: tuple[DatabaseProfile, ...],
) -> dict[str, DatabaseProfile]:
    lookup: dict[str, DatabaseProfile] = {}
    for profile in profiles:
        for value in (profile.key, profile.display_name, *profile.aliases):
            for token in _lookup_tokens(value):
                existing = lookup.get(token)
                if existing is not None and existing.key != profile.key:
                    raise ValueError(
                        f"database alias '{value}' is shared by "
                        f"'{existing.key}' and '{profile.key}'"
                    )
                lookup[token] = profile
    return lookup


def _lookup_tokens(value: str) -> tuple[str, ...]:
    raw = str(value).strip().casefold()
    if not raw:
        return ()
    normalized = raw.replace("_", "-")
    return (raw,) if normalized == raw else (raw, normalized)


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
