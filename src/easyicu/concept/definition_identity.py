"""The computational identity of concept definitions, for every result cache.

A cached concept is current only if the definitions that computed it are
unchanged. A path is not an identity: editing the dictionary at the same path
must miss the cache, and so must changing any field that can change a value
(bounds, units, interval, lookback, levels, sources, callbacks...). Each
cache layer asks this owner instead of choosing its own subset of fields.
"""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import hashlib
from importlib import resources
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

#: Bump when concept computation changes meaning without a dictionary change,
#: so results cached by an older implementation are not reused.
CONCEPT_COMPUTATION_VERSION = "2026-11-03.1"

DictPath = Union[str, Path, Sequence[Union[str, Path]], None]


def _canonical(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_canonical(item) for item in value]
        return sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return _canonical(value.tolist())
    return str(value)


def definition_fingerprint(dictionary: object) -> str:
    """SHA-256 over every field of every definition, in a canonical form."""

    payload = {
        str(name): _canonical(asdict(definition))
        for name, definition in sorted(dictionary.items(), key=lambda item: str(item[0]))
    }
    encoded = json.dumps(
        {"version": CONCEPT_COMPUTATION_VERSION, "definitions": payload},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _files(paths: Iterable[Path]) -> list[Path]:
    selected: list[Path] = []
    for path in paths:
        if path.is_dir():
            selected.extend(sorted(item for item in path.rglob("*.json") if item.is_file()))
        else:
            selected.append(path)
    return selected


def _digest_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256(CONCEPT_COMPUTATION_VERSION.encode("utf-8"))
    for path in _files(paths):
        digest.update(b"\0" + str(path).encode("utf-8") + b"\0")
        try:
            digest.update(path.read_bytes())
        except FileNotFoundError:
            digest.update(b"<missing>")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _packaged_dictionary_fingerprint() -> str:
    """Packaged data does not change within a process; hash it once."""

    digest = hashlib.sha256(CONCEPT_COMPUTATION_VERSION.encode("utf-8"))

    def visit(node, prefix: str) -> None:
        for child in sorted(node.iterdir(), key=lambda item: item.name):
            name = f"{prefix}/{child.name}"
            if child.is_dir():
                visit(child, name)
            elif child.name.endswith(".json"):
                digest.update(b"\0" + name.encode("utf-8") + b"\0")
                digest.update(child.read_bytes())

    visit(resources.files("easyicu.data"), "")
    return digest.hexdigest()


def dictionary_files_fingerprint(dict_path: DictPath) -> str:
    """Content identity of the dictionary files a loader reads.

    ``None`` selects the packaged dictionaries and their overlays. Files are
    read on every call, so an edit at the same path changes the identity.
    """

    if dict_path is None:
        return "packaged:" + _packaged_dictionary_fingerprint()
    if isinstance(dict_path, (str, Path)):
        selected = [Path(dict_path).expanduser().resolve()]
    else:
        selected = [Path(item).expanduser().resolve() for item in dict_path]
    return "files:" + _digest_files(selected)


def optional_dictionary_files_fingerprint(dict_path: object) -> Optional[str]:
    """As :func:`dictionary_files_fingerprint`, or ``None`` for an in-memory dictionary."""

    if dict_path is None or isinstance(dict_path, (str, Path)):
        return dictionary_files_fingerprint(dict_path)
    if isinstance(dict_path, (list, tuple)) and all(isinstance(item, (str, Path)) for item in dict_path):
        return dictionary_files_fingerprint(dict_path)
    if hasattr(dict_path, "items"):
        return "definitions:" + definition_fingerprint(dict_path)
    return None
