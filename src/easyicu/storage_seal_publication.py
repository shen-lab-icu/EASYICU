"""Readers refuse a bucket installation the seal tool has not published.

The seal tool (``tools/seal_buckets/seal_rebuild.py``) installs a root in two
steps. First it writes a durable pending marker ``.<install>.seal_PENDING``
beside the final name, then it renames the read-only staging tree onto that
name. It removes the marker only after the external PASS receipt is durable.
A root whose post-rename verification failed keeps a refusal marker,
``.<install>.seal_REFUSED``. A final name with either marker beside it is not
a published installation, and every reader refuses it, at read time.

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


def _installs(root: Path) -> list[Path]:
    """The installation as given, its realpath, and the realpath of ``root``'s parent.

    The last one is the installation a symlinked bucket (a farm entry) really
    belongs to.
    """
    given = root.parent
    out: list[Path] = []
    for path in (
        given,
        Path(os.path.realpath(given)),
        Path(os.path.realpath(root)).parent,
    ):
        if path not in out:
            out.append(path)
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
