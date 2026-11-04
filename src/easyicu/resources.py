"""Utility helpers for accessing packaged ricu configuration assets."""

from __future__ import annotations

import json
import hashlib
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Iterator, Optional, Sequence, Mapping, Any

from .concept import ConceptDictionary
from .config import DataSourceRegistry

_DATA_PACKAGE = "easyicu.data"
SIC_CASE_OVERLAY_FILE = "sic-case-duration-overlay.v1.json"
SIC_CASE_OVERLAY_SHA256 = "fa3106b181e1c624a268aeb6d393b78abd4a630c9bbd8c25f552c789a06c9c9b"
SIC_CASE_OVERLAY_CONCEPTS = frozenset({
    "care_unit_to_case_closure_days", "death_during_case", "death_time_during_case_hours",
})


def _sic_case_overlay() -> Mapping[str, Any]:
    """Load the separately versioned SIC definitions only with exact identity."""
    resource = resources.files(_DATA_PACKAGE).joinpath(SIC_CASE_OVERLAY_FILE)
    data = resource.read_bytes()
    if hashlib.sha256(data).hexdigest() != SIC_CASE_OVERLAY_SHA256:
        raise ValueError("sic_case_overlay_identity_mismatch")
    payload = json.loads(data)
    if not isinstance(payload, dict) or set(payload) != SIC_CASE_OVERLAY_CONCEPTS:
        raise ValueError("sic_case_overlay_concept_mismatch")
    if any(not isinstance(value, dict) or set(value.get("sources", {})) != {"sic"}
           for value in payload.values()):
        raise ValueError("sic_case_overlay_source_mismatch")
    return payload


def _is_sic_source(source_name: Optional[str]) -> bool:
    if not source_name:
        return False
    from .databases.profiles import get_database_profile

    try:
        return get_database_profile(source_name).key == "sic"
    except (KeyError, ValueError):
        return False


def require_sic_case_overlay(dictionary: ConceptDictionary) -> None:
    """Reject attempts to replace a pinned SIC source definition in place."""
    expected = ConceptDictionary.from_payload(_sic_case_overlay())
    if any(dictionary.get(name) != expected.get(name) for name in SIC_CASE_OVERLAY_CONCEPTS):
        raise ValueError("sic_case_overlay_definition_mismatch")

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
    source_name: Optional[str] = None,
) -> ConceptDictionary:
    """Load the concept dictionary with optional overlays.

    The bundled default also declares computed outcome domains separately
    from its sealed raw-source mappings. Custom dictionaries remain explicit.

    Args:
        name: Base dictionary resource name (default ``concept-dict``).
        directories: Optional custom search directories (checked before bundled data).
        extras: Additional dictionary resources applied as per-concept patches.
            Same-name sources are merged by database; explicitly provided
            metadata fields override the base definition.
        include_sofa2: Convenience flag to append the packaged ``sofa2-dict`` overlay.
        source_name: Optional source identity. The packaged SIC case overlay is
            selected only for SIC; the historic base remains unchanged.

    Returns:
        ConceptDictionary with merged definitions.
    """

    payload = _load_json_payload(name, directories)
    is_packaged_default = name.removesuffix(".json") == "concept-dict" and resolve_resource(name, directories) is None
    if is_packaged_default and _is_sic_source(source_name):
        overlay = _sic_case_overlay()
        if set(payload) & set(overlay):
            raise ValueError("sic_case_overlay_concept_collision")
        payload = {**payload, **overlay}
    dictionary = ConceptDictionary.from_payload(payload)
    # Computed outputs have no raw-source mapping in the sealed dictionary.
    # Declare their domains before applying caller-supplied overlays.
    if is_packaged_default:
        derived = ConceptDictionary.from_payload(_load_package_json("derived-outcome-dict.json"))
        derived.update(dictionary)
        dictionary = derived

    extra_names: list[str] = list(extras or [])
    if (
        name in {"concept-dict", "concept-dict.json"}
        and not directories
        and "community-concept-sources" not in extra_names
    ):
        extra_names.insert(0, "community-concept-sources")
    if include_sofa2 and "sofa2-dict" not in extra_names:
        extra_names.append("sofa2-dict")

    for extra in extra_names:
        overlay = _load_json_payload(extra, directories)
        dictionary.update(ConceptDictionary.from_payload(overlay))

    if is_packaged_default and _is_sic_source(source_name):
        require_sic_case_overlay(dictionary)
        from .concept.sicdb_sources import preserves_raw_cvp

        if preserves_raw_cvp(dictionary.get("cvp_raw"), "sic"):
            dictionary["cvp_raw"].minimum = dictionary["cvp_raw"].maximum = None

    return dictionary
