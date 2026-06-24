"""Shared local data path resolution helpers.

The legacy Streamlit package still renders directory-picker UI, but production
API and FastAPI paths only need deterministic filesystem path resolution. Keep
that pure logic here so importing core EasyICU does not pull in Streamlit.
"""

from __future__ import annotations

import os


DATABASE_ALIASES = {
    "miiv": ["mimiciv", "mimic-iv", "miiv", "mimic_iv", "mimic-iv-3.1"],
    "eicu": ["eicu", "eicu-crd", "eicu_crd"],
    "aumc": ["aumc", "amsterdamumc", "amsterdam"],
    "hirid": ["hirid", "hi-rid"],
    "mimic": ["mimiciii", "mimic-iii", "mimic3", "mimic_iii"],
    "sic": ["sicdb", "sic", "sic-db"],
}

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
    subdirs.sort(reverse=True)
    return os.path.join(path, subdirs[0])


def find_database_path(root: str, db_name: str) -> str:
    """Resolve a database path from a root directory or direct database path."""
    aliases = DATABASE_ALIASES.get(db_name, [db_name])

    if os.path.isdir(root):
        root_basename = os.path.basename(os.path.normpath(root)).lower()
        matched = root_basename in aliases or any(
            alias in root_basename or root_basename.startswith(alias)
            for alias in aliases
        )
        if matched:
            return _latest_numeric_subdir(root) or root
        if _path_looks_like_database(root):
            return root

    for alias in aliases:
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
            entry_lower = entry.lower()
            if any(alias in entry_lower for alias in aliases):
                return _latest_numeric_subdir(entry_path) or entry_path

    return root


__all__ = [
    "DATABASE_ALIASES",
    "DEFAULT_DATABASE_VERSIONS",
    "_path_looks_like_database",
    "find_database_path",
]
