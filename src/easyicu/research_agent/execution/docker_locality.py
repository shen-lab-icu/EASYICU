"""Resolve a locally installed Docker executable when the service PATH is short.

A hosted EasyICU web service started from launchd, a GUI launcher or a login
item inherits a PATH such as ``/usr/bin:/bin:/usr/sbin:/sbin``. Homebrew and
Docker Desktop install their CLI outside every one of those directories, so
``shutil.which("docker")`` returns ``None`` inside the service even though the
host has a working installation, and the governed execution runtime reports
``docker_executable_missing`` for a tool that is present.

This is the same failure the PDF renderer already fixes for LaTeX
(``reporting/pdf_render.py``), so the directory list is deliberately the same
shape. It stays a leaf module with no EasyICU imports: every execution owner can
use it without adding an edge to the research-agent import graph.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Final, Optional

#: Standard install locations that a login shell has and a service process does
#: not. Checked in order, and only after PATH itself, so an explicitly installed
#: or already-active toolchain always wins over anything found here.
LOCAL_DOCKER_DIRS: Final[tuple[Path, ...]] = (
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
    Path.home() / ".docker" / "bin",
)

_EXECUTABLE_NAMES: Final[tuple[str, ...]] = ("docker", "podman")


def candidate_is_pathlike(name: str) -> bool:
    """Whether ``name`` already names a location rather than a command."""

    return "/" in name or Path(name).expanduser().is_absolute()


def _is_usable(candidate: Path) -> bool:
    return candidate.is_file() and os.access(candidate, os.X_OK)


def find_local_docker(names: tuple[str, ...] = _EXECUTABLE_NAMES) -> Optional[str]:
    """Return one standard-location executable, or ``None``.

    Never searches the filesystem: only the fixed candidate list is probed, so a
    mislabelled database directory cannot turn discovery into a scan.
    """

    for directory in LOCAL_DOCKER_DIRS:
        for name in names:
            candidate = directory / name
            try:
                if _is_usable(candidate):
                    return str(candidate)
            except OSError:
                continue
    return None


def resolve_docker_executable(
    requested: Optional[str] = None,
) -> Optional[str]:
    """Resolve one requested Docker-family executable, tolerating a short PATH.

    ``requested`` carries the caller's own precedence (an explicit argument,
    then ``EASYICU_DOCKER_EXECUTABLE``, then the bare ``"docker"`` name). An
    absolute path is honoured as given: a caller that named a binary should not
    have it silently replaced by a different installation. A bare name is looked
    up on PATH first so the environment the caller already has keeps authority,
    and the standard local directories are consulted only on a miss.
    """

    name = str(requested or "docker").strip() or "docker"
    resolved = shutil.which(name)
    if resolved is not None:
        return resolved
    try:
        if candidate_is_pathlike(name):
            candidate = Path(name).expanduser()
            return str(candidate) if _is_usable(candidate) else None
    except OSError:
        return None
    # Search additional locations for the selected command, not substitutes.
    # A wrapper may bind a particular context/socket; replacing it with docker
    # or podman would silently discard the caller's runtime selection.
    return find_local_docker((name,))


__all__ = [
    "LOCAL_DOCKER_DIRS",
    "candidate_is_pathlike",
    "find_local_docker",
    "resolve_docker_executable",
]
