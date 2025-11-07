"""Utility helpers for accessing packaged ricu configuration assets."""

from __future__ import annotations

import json
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Iterator, Optional, Sequence, Mapping, Any

from .concept import ConceptDictionary
from .config import DataSourceRegistry

_DATA_PACKAGE = "pyricu.data"


@contextmanager
def package_path(filename: str) -> Iterator[Path]:
    """Yield a filesystem path to a bundled data file."""

    resource = resources.files(_DATA_PACKAGE).joinpath(filename)
    if not resource.is_file():
        raise FileNotFoundError(f"{filename} not found in package data directory")

    with resources.as_file(resource) as extracted:
        yield Path(extracted)


def _resolve_external(
    filename: str,
    directories: Optional[Sequence[Path | str]],
) -> Optional[Path]:
    if not directories:
        return None
    for directory in directories:
        candidate = Path(directory) / filename
        if candidate.exists():
            return candidate
    return None


def _load_package_json(filename: str) -> object:
    resource = resources.files(_DATA_PACKAGE).joinpath(filename)
    if not resource.is_file():
        raise FileNotFoundError(f"{filename} not found in package data directory")
    text = resource.read_text(encoding="utf8")
    return json.loads(text)


def _load_json_payload(
    name: str,
    directories: Optional[Sequence[Path | str]] = None,
) -> Mapping[str, Any]:
    """Load JSON payload either from external overrides or bundled data."""

    external = resolve_resource(name, directories)
    if external is not None:
        with open(external, "r", encoding="utf8") as fh:
            return json.load(fh)

    filename = name if name.endswith(".json") else f"{name}.json"
    payload = _load_package_json(filename)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected mapping payload in {filename}, got {type(payload)!r}")
    return payload


def resolve_resource(
    name: str,
    directories: Optional[Sequence[Path | str]] = None,
    *,
    suffix: str = ".json",
) -> Path | None:
    """Resolve a JSON resource by name, searching custom directories first."""

    filename = name if name.endswith(suffix) else f"{name}{suffix}"
    return _resolve_external(filename, directories)


def load_data_sources(
    name: str = "data-sources",
    directories: Optional[Sequence[Path | str]] = None,
) -> DataSourceRegistry:
    """Load the packaged data source registry or a user supplied override."""

    external = resolve_resource(name, directories)
    if external is not None:
        return DataSourceRegistry.from_json(external)

    payload = _load_package_json(
        name if name.endswith(".json") else f"{name}.json"
    )
    return DataSourceRegistry.from_payload(payload)


def load_dictionary(
    name: str = "concept-dict",
    directories: Optional[Sequence[Path | str]] = None,
    *,
    extras: Optional[Sequence[str]] = None,
    include_sofa2: bool = False,
) -> ConceptDictionary:
    """Load the concept dictionary with optional overlays.

    Args:
        name: Base dictionary resource name (default ``concept-dict``).
        directories: Optional custom search directories (checked before bundled data).
        extras: Additional dictionary resource names to merge (later entries override).
        include_sofa2: Convenience flag to append the packaged ``sofa2-dict`` overlay.

    Returns:
        ConceptDictionary with merged definitions.
    """

    base_payload = dict(_load_json_payload(name, directories))

    extra_names: list[str] = list(extras or [])
    if include_sofa2 and "sofa2-dict" not in extra_names:
        extra_names.append("sofa2-dict")

    for extra in extra_names:
        overlay = _load_json_payload(extra, directories)
        base_payload.update(overlay)

    return ConceptDictionary.from_payload(base_payload)
