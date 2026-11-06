"""Which EasyICU code produced a result.

This module owns the code identity recorded in every native export's
``runtime_provenance``. The identity comes from one of three sources, in order:

``git_checkout``
    The imported package (symlinks resolved) is exactly ``<root>/src/easyicu``,
    ``<root>`` is the Git top level, its ``pyproject.toml`` names the EasyICU
    project, and Git tracks the imported package there (``__init__.py`` and
    this module). A package anywhere else inside a repository (vendored,
    installed into ``site-packages`` or a target directory under a checkout, a
    look-alike copy whose files Git does not track, or a symlink into any of
    these) is not that checkout and never borrows its commit.
``build_stamp``
    The package was built from a Git checkout and carries the
    ``_build_identity.json`` stamp written at build time (wheel, sdist, desktop
    bundle). The stamp records the commit of the build's own checkout.
``unknown``
    Neither applies, for example a source tree copied out of Git. The identity
    then says so with a reason code; it is never a silent ``None``. A formal
    run refuses it.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Dict, Optional

BUILD_IDENTITY_FILE = "_build_identity.json"
BUILD_IDENTITY_SCHEMA = "easyicu_build_identity_v1"

SOURCE_GIT_CHECKOUT = "git_checkout"
SOURCE_BUILD_STAMP = "build_stamp"
SOURCE_UNKNOWN = "unknown"

REASON_UNAVAILABLE = "code_identity_unavailable"
REASON_STAMP_INVALID = "code_identity_stamp_invalid"

PROJECT_NAME = "easyicu"
_TRACKED = ("src/easyicu/__init__.py", "src/easyicu/code_identity.py")

_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _git(repository: Path, *arguments: str, text: bool = True) -> Any:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=text,
        timeout=10,
    ).stdout


def _is_easyicu_project(root: Path) -> bool:
    try:
        with (root / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle).get("project") or {}
    except (OSError, ValueError):
        return False
    return project.get("name") == PROJECT_NAME


def git_checkout_identity(repository_root: Path) -> Optional[Dict[str, Any]]:
    """Return the identity of ``repository_root`` when it is an EasyICU checkout.

    ``repository_root`` must be the Git top level, hold the EasyICU project
    metadata, and track the package files that are imported from it.
    """

    root = Path(repository_root).resolve()
    if not _is_easyicu_project(root):
        return None
    try:
        top_level = Path(_git(root, "rev-parse", "--show-toplevel").strip()).resolve()
        if top_level != root:
            return None
        _git(root, "ls-files", "--error-unmatch", "--", *_TRACKED)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    try:
        commit = _git(root, "rev-parse", "HEAD").strip()
        status = _git(root, "status", "--short", "--untracked-files=all")
        diff = _git(root, "diff", "--binary", "HEAD", text=False)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if not _COMMIT.fullmatch(commit):
        return None
    dirty = bool(status.strip())
    return {
        "source": SOURCE_GIT_CHECKOUT,
        "commit": commit,
        "dirty": dirty,
        "diff_sha256": hashlib.sha256(diff).hexdigest() if dirty else None,
    }


def build_stamp_payload(repository_root: Path, *, version: str) -> Optional[Dict[str, Any]]:
    """The stamp a build writes, from the build's own checkout; ``None`` outside Git."""

    identity = git_checkout_identity(repository_root)
    if identity is None:
        return None
    return {
        "schema": BUILD_IDENTITY_SCHEMA,
        "version": str(version),
        "commit": identity["commit"],
        "dirty": identity["dirty"],
        "diff_sha256": identity["diff_sha256"],
    }


def _stamp_identity(package_root: Path) -> Optional[Dict[str, Any]]:
    path = Path(package_root) / BUILD_IDENTITY_FILE
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = None
    valid = (
        isinstance(payload, dict)
        and payload.get("schema") == BUILD_IDENTITY_SCHEMA
        and isinstance(payload.get("version"), str)
        and isinstance(payload.get("commit"), str)
        and _COMMIT.fullmatch(payload["commit"]) is not None
        and isinstance(payload.get("dirty"), bool)
        and "diff_sha256" in payload
        and (
            (payload["dirty"] and isinstance(payload.get("diff_sha256"), str)
             and _SHA256.fullmatch(payload["diff_sha256"]) is not None)
            or (not payload["dirty"] and payload.get("diff_sha256") is None)
        )
    )
    if not valid:
        return _unknown(REASON_STAMP_INVALID)
    return {
        "source": SOURCE_BUILD_STAMP,
        "commit": payload["commit"],
        "dirty": payload["dirty"],
        "diff_sha256": payload["diff_sha256"],
        "version": payload["version"],
    }


def _unknown(reason: str) -> Dict[str, Any]:
    return {
        "source": SOURCE_UNKNOWN,
        "commit": None,
        "dirty": None,
        "diff_sha256": None,
        "reason_code": reason,
    }


def resolve_code_identity(package_root: Path) -> Dict[str, Any]:
    """Resolve the identity of the imported ``easyicu`` package at ``package_root``."""

    root = Path(package_root).resolve()
    if root.name == PROJECT_NAME and root.parent.name == "src":
        identity = git_checkout_identity(root.parent.parent)
        if identity is not None:
            return identity
    stamp = _stamp_identity(root)
    if stamp is not None:
        return stamp
    return _unknown(REASON_UNAVAILABLE)


__all__ = [
    "BUILD_IDENTITY_FILE",
    "BUILD_IDENTITY_SCHEMA",
    "REASON_STAMP_INVALID",
    "REASON_UNAVAILABLE",
    "SOURCE_BUILD_STAMP",
    "SOURCE_GIT_CHECKOUT",
    "SOURCE_UNKNOWN",
    "build_stamp_payload",
    "git_checkout_identity",
    "resolve_code_identity",
]
