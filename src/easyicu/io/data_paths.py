"""Shared local data path resolution helpers.

Production API and FastAPI paths need deterministic filesystem path resolution
without importing any UI runtime. Keep that pure logic here so core EasyICU
imports stay local and lightweight.
"""

from __future__ import annotations

import os
import re

from easyicu.databases.profiles import DATABASE_ALIASES, normalize_database_key

DEFAULT_DATABASE_VERSIONS = {
    "mimiciv": "3.1",
    "mimic-iv": "3.1",
    "miiv": "3.1",
    "eicu": "2.0.1",
    "eicu-crd": "2.0.1",
    "aumc": "1.0.2",
    "hirid": "1.1.1",
    "mimiciii": "1.4",
    "mimic-iii": "1.4",
    "sicdb": "1.0.6",
    "sic": "1.0.6",
    "nwicu": "0.1.0",
    "northwestern-icu": "0.1.0",
    "zhejiang_eicu": "OMIX005817",
    "jinhua": "OMIX007493-02",
    "zigong": "1.1",
}


def _path_looks_like_database(path: str) -> bool:
    """Return True when a path contains raw/prepared ICU database files."""
    if not os.path.isdir(path):
        return False
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    entries_lower = [entry.lower() for entry in entries]
    if any(entry.endswith(".parquet") for entry in entries_lower):
        return True
    known_dirs = {
        "hosp",
        "icu",
        "observations_bucket",
        "pharma_bucket",
        "observation_tables",
        "pharma_records",
        "reference_data",
    }
    if known_dirs & set(entries_lower):
        return True
    if any(entry.endswith(".csv") or entry.endswith(".csv.gz") for entry in entries_lower):
        return True
    return any(entry.endswith("_bucket") for entry in entries_lower)


def _latest_numeric_subdir(path: str) -> str | None:
    try:
        subdirs = [
            entry
            for entry in os.listdir(path)
            if os.path.isdir(os.path.join(path, entry)) and entry and entry[0].isdigit()
        ]
    except OSError:
        return None
    if not subdirs:
        return None
    def version_key(name: str) -> tuple[tuple[int, object], ...]:
        return tuple(
            (0, int(part)) if part.isdigit() else (1, part.casefold())
            for part in re.findall(r"\d+|[A-Za-z]+", name)
        )

    subdirs.sort(key=version_key, reverse=True)
    return os.path.join(path, subdirs[0])


def _normalized_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def find_database_path(root: str, db_name: str) -> str:
    """Resolve a database path from a root directory or direct database path."""
    try:
        database_key = normalize_database_key(db_name)
    except KeyError:
        database_key = db_name
    aliases = DATABASE_ALIASES.get(database_key, (db_name,))

    if os.path.isdir(root):
        root_basename = os.path.basename(os.path.normpath(root))
        accepted_labels = {
            _normalized_label(label) for label in (database_key, *aliases)
        }
        matched = _normalized_label(root_basename) in accepted_labels
        if matched:
            return _latest_numeric_subdir(root) or root
        if _path_looks_like_database(root):
            return root

    search_names = (database_key, *aliases)
    for alias in dict.fromkeys(search_names):
        direct_path = os.path.join(root, alias)
        if os.path.isdir(direct_path):
            return _latest_numeric_subdir(direct_path) or direct_path

        default_version = DEFAULT_DATABASE_VERSIONS.get(alias)
        if default_version:
            versioned_path = os.path.join(root, alias, default_version)
            if os.path.isdir(versioned_path):
                return versioned_path

    if os.path.isdir(root):
        try:
            entries = os.listdir(root)
        except OSError:
            entries = []
        for entry in entries:
            entry_path = os.path.join(root, entry)
            if not os.path.isdir(entry_path):
                continue
            if _normalized_label(entry) in {
                _normalized_label(label) for label in (database_key, *aliases)
            }:
                return _latest_numeric_subdir(entry_path) or entry_path

    raise FileNotFoundError(
        f"Could not resolve database {db_name!r} below data root {root!r}"
    )


__all__ = [
    "DATABASE_ALIASES",
    "DEFAULT_DATABASE_VERSIONS",
    "_path_looks_like_database",
    "find_database_path",
]
