"""The computational identity of concept definitions, for every result cache.

A cached concept is current only if the definitions that computed it, and the
code that computed it, are unchanged. A path is not an identity: editing the
dictionary at the same path must miss the cache, and so must changing any
field that can change a value (bounds, units, interval, lookback, levels,
sources, callbacks...). Each cache layer asks this owner instead of choosing
its own subset of fields.

Code identity (P1-2, 11-11 review): :data:`CONCEPT_COMPUTATION_VERSION` alone
was not bumped for computation fixes after 11-03, so an upgraded install
served values cached by older code (HiRID fluid 580 mL cached vs 480 mL
recomputed). :func:`concept_computation_identity` therefore adds a digest of
the source files that compute concept values (:data:`CONCEPT_CODE_DIRECTORIES`
and :data:`CONCEPT_CODE_MODULES`), read from the imported package. The desktop
build copies the wheel's ``.py`` files next to its frozen modules, so the same
files are hashed there. Where no covered source file can be read (a
bytecode-only install) the digest falls back to the package version plus the
computation version: such an install then relies on the version bump.
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
#: so results cached by an older implementation are not reused. The code
#: digest below catches a missed bump wherever the sources are readable.
CONCEPT_COMPUTATION_VERSION = "2026-11-11.1"

#: Package subdirectories (relative to the ``easyicu`` package) whose ``.py``
#: files compute concept values: the resolver and its callbacks, scores, unit,
#: time and table helpers, source readers and the public loading API.
CONCEPT_CODE_DIRECTORIES = (
    "api",
    "concept",
    "databases",
    "io",
    "scores",
    "table",
    "utils",
)

#: Top-level modules the concept loader calls to compute values: data-source
#: reading, time/unit configuration, storage layout and value policies.
CONCEPT_CODE_MODULES = (
    "base.py",
    "callbacks.py",
    "callbacks_missing.py",
    "clinical_contracts.py",
    "concept_output_sources.py",
    "config.py",
    "database_config.py",
    "datasource.py",
    "hospital_mortality.py",
    "load_concepts.py",
    "outcome_availability.py",
    "patient_filter.py",
    "resources.py",
    "source_bounds.py",
    "source_death_times.py",
    "storage_binding.py",
    "storage_layout.py",
    "urine_weight_linkage.py",
    "value_failures.py",
)

DictPath = Union[str, Path, Sequence[Union[str, Path]], None]


def _package_root() -> Optional[Path]:
    """The imported ``easyicu`` package directory (``concept/..``), if known."""

    try:
        return Path(__file__).resolve().parent.parent
    except (NameError, OSError, RuntimeError):
        return None


def _concept_code_files(package_root: Path) -> list[tuple[str, Optional[Path]]]:
    """Sorted ``(package-relative posix name, path or None if absent)`` pairs."""

    found: dict[str, Optional[Path]] = {}
    for directory in CONCEPT_CODE_DIRECTORIES:
        base = package_root / directory
        if base.is_dir():
            for path in base.rglob("*.py"):
                if path.is_file():
                    found[path.relative_to(package_root).as_posix()] = path
    for module in CONCEPT_CODE_MODULES:
        path = package_root / module
        found[module] = path if path.is_file() else None
    return sorted(found.items())


def _package_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - stdlib on every supported Python
        return "unknown"
    try:
        return version("easyicu")
    except PackageNotFoundError:
        return "unknown"


def _code_digest(package_root: Optional[Path]) -> str:
    """Digest the covered source files under ``package_root``.

    Each entry contributes its length-prefixed package-relative name and
    bytes (or an absent/unreadable marker), in sorted order, so the value
    does not depend on the install location or directory listing order.
    """

    entries = _concept_code_files(package_root) if package_root is not None else []
    if not any(path is not None for _, path in entries):
        # No readable source (bytecode-only install): pin to what is known.
        fallback = json.dumps(
            {"package_version": _package_version(), "computation_version": CONCEPT_COMPUTATION_VERSION},
            sort_keys=True,
            separators=(",", ":"),
        )
        return "fallback:" + hashlib.sha256(fallback.encode("utf-8")).hexdigest()
    digest = hashlib.sha256(b"easyicu.concept_code/1\0")
    for name, path in entries:
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big") + encoded)
        if path is None:
            payload = b"<absent>"
        else:
            try:
                payload = path.read_bytes()
            except OSError as exc:
                payload = f"<unreadable:{type(exc).__name__}>".encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big") + payload)
    return "sha256:" + digest.hexdigest()


@lru_cache(maxsize=1)
def concept_code_digest() -> str:
    """Digest of the imported concept computation code, once per process."""

    return _code_digest(_package_root())


def concept_computation_identity() -> str:
    """The computation version plus the code digest: what every cache key binds."""

    return f"{CONCEPT_COMPUTATION_VERSION}+{concept_code_digest()}"


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
        {"version": concept_computation_identity(), "definitions": payload},
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
    digest = hashlib.sha256(concept_computation_identity().encode("utf-8"))
    for path in _files(paths):
        digest.update(b"\0" + str(path).encode("utf-8") + b"\0")
        try:
            digest.update(path.read_bytes())
        except FileNotFoundError:
            digest.update(b"<missing>")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _packaged_dictionary_fingerprint() -> str:
    """Packaged data does not change within a process; hash it once.

    The computation identity is bound by the caller, outside this cache.
    """

    digest = hashlib.sha256(b"easyicu.packaged_dictionaries/1")

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
    Every form binds :func:`concept_computation_identity`.
    """

    if dict_path is None:
        bound = f"{concept_computation_identity()}\0{_packaged_dictionary_fingerprint()}"
        return "packaged:" + hashlib.sha256(bound.encode("utf-8")).hexdigest()
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
