"""Readers refuse a bucket installation the seal tool has not published.

The seal tool (``tools/seal_buckets/seal_rebuild.py``) installs a root in two
steps. First it writes a durable pending marker ``.<install>.seal_PENDING``
beside the final name, then it renames the read-only staging tree onto that
name. It removes the marker only after the external PASS receipt is durable.
A root whose post-rename verification failed keeps a refusal marker,
``.<install>.seal_REFUSED``. A final name with either marker beside it is not
a published installation. The primary gate walks the complete configured
install closure before a run or datasource starts; read-time checks remain
as defense in depth.

``formal=True`` adds the acceptance half. The installation must carry the
seal tool's in-root receipt ``_SEAL_BUILD_RECEIPT.json`` with verdict
``PASS`` and no failure. A bucket must be the one that receipt sealed: its
``_BUCKET_MANIFEST.json`` sha256 must equal ``tables[<table>].manifest_sha256``.
In formal mode nothing falls back to an unsealed source.

A farm (the CareVue directory that links ``<table>_bucket`` to a sealed bucket
root) carries its own receipt with ``bucket_links`` instead of ``tables``. That
receipt is never accepted on its own: a farm bucket is bound through the
receipt of the realpath installation it links to.

The reader calls only the marker half (``formal=False``). The formal half is
for the formal run receipt to call per installation root
(:func:`assert_install_published`) or per bucket read
(:func:`assert_seal_published`).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .storage_layout import MANIFEST, StorageIntegrityError

# The seal tool's names (tools/seal_buckets/seal_rebuild.py, review-r13sealtool
# d09b1fcb3): PENDING_SUFFIX, REFUSAL_SUFFIX, IN_ROOT_RECEIPT.
PENDING_SUFFIX = ".seal_PENDING"
REFUSAL_SUFFIX = ".seal_REFUSED"
IN_ROOT_RECEIPT = "_SEAL_BUILD_RECEIPT.json"
BUCKET_SUFFIX = "_bucket"

UNPUBLISHED = "bucket_install_unpublished"
REFUSED = "bucket_install_refused"
UNACCEPTED = "bucket_install_unaccepted"


class SealPublicationError(StorageIntegrityError):
    """A reader met an installation the seal tool has not published or accepted.

    It is a :class:`StorageIntegrityError`, but deliberately not a
    ``BucketLayoutError``. A reader that catches a bucket layout error falls
    back to the flat source of the same installation, and an unpublished
    installation has no readable source at all.
    """

    def __init__(self, reason: str, install: Path, detail: str = ""):
        self.reason = reason
        self.install = install
        super().__init__(
            f"{reason}: {install.name}" + (f"; {detail}" if detail else "")
        )


@dataclass(frozen=True)
class SealPublicationClosure:
    """Start-time publication identity; walking never opens table contents.

    Every directory on a configured or linked path is a possible install.
    Counting these candidates avoids guessing install boundaries from names.
    """

    sha256: str
    install_count: int
    link_hops: int
    marker_hits: int = 0

    def to_dict(self) -> dict:
        return {
            "sha256": self.sha256,
            "install_count": self.install_count,
            "link_hops": self.link_hops,
            "marker_hits": self.marker_hits,
        }


def configured_publication_roots(
    config, base_path=None, table_sources=None
) -> list[Path]:
    """Include external configured files/datasets and registered Path loaders."""
    base = Path(base_path) if base_path is not None else Path.cwd()
    roots = [base] if base_path is not None else []
    for table in config.tables.values():
        paths = [entry.get("path") or entry.get("file") for entry in table.files]
        if table.dataset is not None:
            paths.append(table.dataset.path)
        for raw in paths:
            if raw:
                path = Path(raw)
                if not path.is_absolute():
                    path = base / path
                # The container covers every expansion of a configured glob.
                while any(c in str(path) for c in "*?["):
                    path = path.parent
                roots.append(path)
    roots.extend(
        Path(raw)
        for raw in (table_sources or {}).values()
        if isinstance(raw, (str, Path))
    )
    return roots


def assert_roots_published(roots) -> SealPublicationClosure:
    """Gate the full install closure before a run, datasource or resume reads.

    Resolve symlinks component by component, including intermediate farms and
    ancestors of file targets. Walk all subdirectories, even unused tables.
    Only directory entries, lstat/readlink and publication/provenance JSON are
    read. No table file, schema, receipt hash or data hash is opened here.
    Directory realpaths deduplicate cycles. Missing legacy paths retain their
    existing read-time handling; an unreadable closure fails closed.
    """
    return _publication_closure(roots)


def assert_configuration_outside_roots(path, roots) -> None:
    """Refuse configuration bytes inside any input or linked/provenance root.

    Ancestors used only for marker checks are not input directories: a seal
    work_root may legitimately hold both the installs and their configuration.
    Check containment before listing a reached directory or opening metadata.
    """
    _publication_closure(roots, configuration=Path(os.path.realpath(path)))


def _publication_closure(roots, *, configuration=None) -> SealPublicationClosure:
    pending = deque(Path(p) for p in roots if p is not None)
    checked: set[Path] = set()
    installs: set[Path] = set()
    visited: set[Path] = set()
    metadata: set[Path] = set()
    metadata_directories: set[Path] = set()
    links: set[Path] = set()
    metadata_names = {MANIFEST, "conversion_manifest.json", IN_ROOT_RECEIPT}

    def check(path: Path) -> None:
        if path in checked:
            return
        for suffix, reason in (
            (REFUSAL_SUFFIX, REFUSED),
            (PENDING_SUFFIX, UNPUBLISHED),
        ):
            try:
                _marker(path, suffix).lstat()
            except FileNotFoundError:
                pass
            else:
                raise SealPublicationError(reason, path)
        checked.add(path)

    def resolve(path: Path) -> Path | None:
        # Do not normpath before following links: link/.. refers to the
        # target's parent, and collapsing it would skip intermediate farms.
        if not path.is_absolute():
            path = Path.cwd() / path
        parts = deque(path.parts[1:])
        current = Path(path.anchor)
        chain: set[tuple[Path, tuple[str, ...]]] = set()
        hops = 0
        while parts:
            check(current)
            installs.add(current)
            # A linked table may enter an install below its root. Its parent
            # receipts can bind further installs even outside this subtree.
            if current not in metadata_directories:
                metadata_directories.add(current)
                for name in metadata_names:
                    candidate_metadata = current / name
                    if os.path.lexists(candidate_metadata):
                        pending.append(candidate_metadata)
            part = parts.popleft()
            if part == "..":
                current = current.parent
                continue
            candidate = current / part
            try:
                mode = candidate.lstat().st_mode
            except FileNotFoundError:
                # Check a not-yet-installed final name too (rename pending).
                check(candidate)
                return None
            if stat.S_ISLNK(mode):
                check(candidate)
                links.add(candidate)
                state = (candidate, tuple(parts))
                hops += 1
                if state in chain or hops > 256:
                    raise SealPublicationError(UNACCEPTED, candidate, "link cycle")
                chain.add(state)
                target = Path(os.readlink(candidate))
                if target.is_absolute():
                    current = Path(target.anchor)
                    target_parts = target.parts[1:]
                else:
                    target_parts = target.parts
                parts.extendleft(reversed(target_parts))
            else:
                current = candidate
        if current.is_dir():
            check(current)
            installs.add(current)
        return current

    def enqueue_metadata(path: Path) -> None:
        if path in metadata:
            return
        metadata.add(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError):
            # Layout validation remains the read owner's responsibility.
            return
        if not isinstance(payload, dict):
            return
        targets = []
        cwd_targets = []
        derived = payload.get("derived_from")
        if isinstance(derived, dict):
            if isinstance(derived.get("path"), str):
                targets.append((path.parent, derived["path"]))
                cwd_targets.append(derived["path"])
            for entry in (
                derived.get("files", [])
                if isinstance(derived.get("files"), list)
                else []
            ):
                if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                    targets.append((path.parent.parent, entry["path"]))
        bucket_links = payload.get("bucket_links")
        if isinstance(bucket_links, dict):
            for target in bucket_links.values():
                if isinstance(target, str):
                    targets.append((path.parent, target))
                    cwd_targets.append(target)
        for base, raw in targets:
            target = Path(raw)
            pending.append(target if target.is_absolute() else base / target)
        # Conversion-manifest and formal farm readers use Path(raw) directly.
        # Cover that legacy cwd binding as well as portable metadata-relative
        # paths. Bucket derived_from.files remain relative to the install.
        pending.extend(Path(raw) for raw in cwd_targets if not Path(raw).is_absolute())

    path = Path(".")
    try:
        while pending:
            requested = pending.popleft()
            path = resolve(requested)
            if path is None:
                continue
            if configuration is not None and (
                configuration == path
                or path.is_dir() and configuration.is_relative_to(path)
            ):
                raise SealPublicationError(
                    "formal_configuration_inside_input", path,
                    "configuration must be outside the publication closure",
                )
            if not path.is_dir():
                # External file targets bring their containing install into
                # the closure, including its other links and metadata.
                pending.append(path.parent)
                if requested.name in metadata_names or path.name in metadata_names:
                    enqueue_metadata(path)
                continue
            if path in visited:
                continue
            visited.add(path)
            with os.scandir(path) as entries:
                for entry in entries:
                    child = path / entry.name
                    if entry.is_symlink() or entry.is_dir(follow_symlinks=False):
                        pending.append(child)
                    elif entry.name in metadata_names:
                        enqueue_metadata(child)
    except OSError as exc:
        raise SealPublicationError(
            UNACCEPTED, path or Path("."), "cannot inspect publication closure"
        ) from exc
    # Marker state is explicit in the canonical identity. A marked closure
    # raises above and can never become an accepted start identity.
    records = [[str(p), False, False] for p in sorted(installs)]
    digest = hashlib.sha256(
        json.dumps(records, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SealPublicationClosure(digest, len(installs), len(links))


def _installs(root: Path) -> list[Path]:
    """All possible install ancestors on every hop, without opening tables.

    A table may be arbitrarily nested below the conversion manifest/receipt,
    including MIMIC-IV's icu/ and hosp/. Checking the full ancestor chain also
    covers legacy installs with neither metadata file and intermediate farms.
    Do not resolve the whole path at once: that loses intermediate links.
    """
    if not root.is_absolute():
        root = Path.cwd() / root
    parts = deque(root.parts[1:])
    current = Path(root.anchor)
    out: list[Path] = []
    seen: set[Path] = set()
    chain: set[tuple[Path, tuple[str, ...]]] = set()
    hops = 0

    def add(path):
        if path not in seen:
            seen.add(path)
            out.append(path)

    while parts:
        add(current)
        part = parts.popleft()
        if part == "..":
            current = current.parent
            continue
        candidate = current / part
        if candidate.is_symlink():
            add(candidate)
            state = (candidate, tuple(parts))
            hops += 1
            if state in chain or hops > 256:
                raise SealPublicationError(UNACCEPTED, candidate, "link cycle")
            chain.add(state)
            target = Path(os.readlink(candidate))
            if target.is_absolute():
                current = Path(target.anchor)
                target_parts = target.parts[1:]
            else:
                target_parts = target.parts
            parts.extendleft(reversed(target_parts))
        else:
            current = candidate
    if current.is_dir():
        add(current)
    return out


def _marker(install: Path, suffix: str) -> Path:
    return install.parent / f".{install.name}{suffix}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _passing_receipt(install: Path) -> dict[str, Any]:
    path = install / IN_ROOT_RECEIPT
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SealPublicationError(
            UNACCEPTED, install, f"no readable {IN_ROOT_RECEIPT}"
        ) from exc
    if not isinstance(receipt, dict) or receipt.get("verdict") != "PASS":
        raise SealPublicationError(UNACCEPTED, install, "receipt verdict is not PASS")
    if "failure" in receipt:
        raise SealPublicationError(UNACCEPTED, install, "receipt records a failure")
    return receipt


def _bind_bucket(install: Path, receipt: dict, root: Path, has_manifest: bool) -> None:
    """A receipt with ``tables``: the bucket read must be the one it sealed."""
    tables = receipt.get("tables")
    if not isinstance(tables, dict):
        raise SealPublicationError(UNACCEPTED, install, "receipt has no tables")
    table = (
        root.name[: -len(BUCKET_SUFFIX)] if root.name.endswith(BUCKET_SUFFIX) else None
    )
    record = tables.get(table) if table else None
    if not has_manifest:
        if record is not None:
            # The receipt sealed this table as a bucket; reading it any other
            # way is an unsealed fallback.
            raise SealPublicationError(
                UNACCEPTED, install, f"{root.name} is sealed but absent"
            )
        return
    if not isinstance(record, dict):
        raise SealPublicationError(UNACCEPTED, install, f"{root.name} is not sealed")
    if "failure" in record or record.get("qc_findings"):
        raise SealPublicationError(
            UNACCEPTED, install, f"{root.name} has failure entries"
        )
    bound = record.get("manifest_sha256")
    if not bound or _sha256(root / MANIFEST) != bound:
        raise SealPublicationError(
            UNACCEPTED, install, f"{root.name} manifest differs from the receipt"
        )


def _assert_accepted(root: Path) -> None:
    install = root.parent
    has_manifest = (root / MANIFEST).is_file()
    receipt = _passing_receipt(install)
    if "tables" in receipt:
        _bind_bucket(install, receipt, root, has_manifest)
        return
    links = receipt.get("bucket_links")
    if not isinstance(links, dict):
        raise SealPublicationError(
            UNACCEPTED, install, "receipt binds no tables or bucket links"
        )
    link = links.get(root.name)
    if link is None:
        # A farm receipt alone never accepts anything.
        raise SealPublicationError(
            UNACCEPTED, install, f"{root.name} is not a sealed link"
        )
    real = Path(os.path.realpath(root))
    if Path(os.path.realpath(link)) != real:
        raise SealPublicationError(
            UNACCEPTED, install, f"{root.name} does not resolve to its sealed link"
        )
    target = real.parent
    _bind_bucket(target, _passing_receipt(target), real, has_manifest)


def assert_seal_published(root: Path, *, formal: bool) -> None:
    """Refuse ``root`` unless its installation ``root.parent`` is published.

    ``root`` is a bucket directory or, for a flat read, any path directly
    inside the installation (the reader passes ``<source>/<table>_bucket``).
    A pending or refusal marker beside the installation, as given or as its
    realpath, is ``bucket_install_unpublished`` or ``bucket_install_refused``.
    Markers are tested with lstat, so a dangling marker link still counts.
    With ``formal=True`` the installation must also pass
    :func:`_assert_accepted` (``bucket_install_unaccepted``).
    """
    root = Path(root)
    for install in _installs(root):
        if os.path.lexists(_marker(install, REFUSAL_SUFFIX)):
            raise SealPublicationError(REFUSED, install)
        if os.path.lexists(_marker(install, PENDING_SUFFIX)):
            raise SealPublicationError(UNPUBLISHED, install)
    if formal:
        _assert_accepted(root)


def assert_install_published(install: Path, *, formal: bool) -> None:
    """:func:`assert_seal_published` for a whole installation root.

    The markers of ``install`` (as given and as its realpath) are checked.
    With ``formal=True``, every bucket the installation's receipt binds must
    match it. For a farm, every ``bucket_links`` entry is bound through the
    receipt of the installation it links to, and a farm without links is
    refused.
    """
    install = Path(install)
    assert_seal_published(install / IN_ROOT_RECEIPT, formal=False)
    if not formal:
        return
    receipt = _passing_receipt(install)
    if "tables" in receipt:
        names = [f"{t}{BUCKET_SUFFIX}" for t in (receipt.get("tables") or {})]
    else:
        names = list(receipt.get("bucket_links") or {})
    if not names:
        raise SealPublicationError(UNACCEPTED, install, "receipt binds no buckets")
    for name in names:
        assert_seal_published(install / name, formal=True)


def assert_source_published(source, table: str, *, formal: bool) -> None:
    """:func:`assert_seal_published` for a data source's own installation.

    It covers flat reads too: a flat table lives directly in the source root,
    the same installation as its buckets. An in-memory source has no
    installation.
    """
    if getattr(source, "base_path", None) is None:
        return
    assert_seal_published(
        Path(source.base_path) / f"{table}{BUCKET_SUFFIX}", formal=formal
    )
