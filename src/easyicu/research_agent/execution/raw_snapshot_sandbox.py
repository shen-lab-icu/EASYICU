"""Experimental macOS isolation for a worker over an approved input snapshot.

This is a containment primitive, not a research grant, immutable snapshot, or
record of files actually consumed. Callers must provide controlled runtime and
worker files, and arrange independent snapshot integrity and authorization.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _quote(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def _inside(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def _symlink_components(path: Path) -> tuple[Path, ...]:
    """List literal links needed to resolve one executable, without broad grants."""
    remaining = list(path.parts[1:])
    current = Path(path.anchor)
    links = []
    while remaining:
        current = current / remaining.pop(0)
        if current.is_symlink():
            links.append(current)
            if len(links) > 40:
                raise ValueError("python launcher has too many symlinks")
            target = Path(os.readlink(current))
            replacement = target if target.is_absolute() else current.parent / target
            replacement = Path(os.path.abspath(replacement))
            current = Path(replacement.anchor)
            remaining = list(replacement.parts[1:]) + remaining
    return tuple(links)


def run_snapshot_worker(
    *,
    snapshot: Path,
    output: Path,
    python: Path,
    worker: Path,
    runtime_roots: Sequence[Path],
    forbidden_source: Path | None = None,
    scratch: Path | None = None,
    arguments: Sequence[str] = (),
    timeout: float = 60,
) -> subprocess.CompletedProcess[str]:
    """Run a trusted worker with explicit read roots and no network.

    The output directory must be freshly created and empty. Snapshot and runtime
    are readable but not writable by the worker. Runtime roots must contain only
    approved runtime dependencies, never unapproved research data. No caller
    environment, arbitrary path-valued environment, or fallback is inherited.
    """
    sandbox = shutil.which("sandbox-exec")
    if sys.platform != "darwin" or sandbox is None:
        raise RuntimeError("macOS sandbox-exec is required; no unsafe fallback")
    snapshot = Path(snapshot).resolve(strict=True)
    output = Path(output).resolve(strict=True)
    scratch = Path(scratch).resolve(strict=True) if scratch is not None else output
    python_launcher = Path(os.path.abspath(Path(python).expanduser()))
    python = python_launcher.resolve(strict=True)
    python_launcher_dir = python_launcher.parent.resolve(strict=True)
    worker = Path(worker).resolve(strict=True)
    roots = tuple(Path(root).resolve(strict=True) for root in runtime_roots)
    if (not snapshot.is_dir() or not output.is_dir() or not scratch.is_dir()
            or any(output.iterdir()) or any(scratch.iterdir())):
        raise ValueError("snapshot and fresh empty output directories are required")
    if not python.is_file() or not worker.is_file() or not roots:
        raise ValueError("python, trusted worker, and runtime roots are required")
    if not all(root.is_dir() for root in roots):
        raise ValueError("runtime roots must be directories")
    if (not any(_inside(python, root) for root in roots)
            or not any(_inside(python_launcher_dir, root) for root in roots)):
        raise ValueError("python launcher and target must belong to approved runtime roots")
    if (_inside(snapshot, output) or _inside(output, snapshot)
            or _inside(snapshot, scratch) or _inside(scratch, snapshot)):
        raise ValueError("snapshot and output must be disjoint")
    if scratch != output and (_inside(scratch, output) or _inside(output, scratch)):
        raise ValueError("scratch and output must be disjoint")
    if any(
        _inside(snapshot, root) or _inside(root, snapshot)
        or _inside(output, root) or _inside(root, output) or _inside(scratch, root) or _inside(root, scratch)
        for root in roots
    ):
        raise ValueError("snapshot and output must be outside runtime roots")
    if _inside(worker, snapshot) or _inside(worker, output) or _inside(worker, scratch):
        raise ValueError("worker must be outside snapshot and output")

    system_roots = tuple(
        Path(path).resolve()
        for path in (
            "/System/Library", "/usr/lib", "/usr/share/locale",
            "/Library/Frameworks", "/private/var/db/timezone",
        )
        if Path(path).exists()
    )
    if any(
        _inside(snapshot, root) or _inside(root, snapshot)
        or _inside(output, root) or _inside(root, output) or _inside(scratch, root) or _inside(root, scratch)
        for root in system_roots
    ):
        raise ValueError("snapshot and output must be outside system roots")
    if forbidden_source is not None:
        forbidden_source = Path(forbidden_source).resolve(strict=True)
        if any(_inside(forbidden_source, root) or _inside(root, forbidden_source)
               for root in (*roots, *system_roots)) or _inside(worker, forbidden_source):
            raise ValueError("raw source must be outside runtime roots and worker path")
    read_roots = (*system_roots, *roots, snapshot, output, scratch)
    read_files = (Path("/"), worker, Path("/dev/null"), Path("/dev/random"), Path("/dev/urandom"),
                  *_symlink_components(python_launcher))
    ancestors = {parent for path in (*read_roots, *read_files) for parent in path.parents}
    rules = [
        "(version 1)", "(deny default)", "(allow process*)",
        "(allow sysctl-read)", "(allow mach-lookup)", "(allow ipc-posix-shm)",
    ]
    rules.extend(f'(allow file-read-metadata (literal "{_quote(p)}"))' for p in sorted(ancestors))
    rules.extend(f'(allow file-read* (subpath "{_quote(p)}"))' for p in read_roots)
    rules.extend(f'(allow file-read* (literal "{_quote(p)}"))' for p in read_files)
    rules.extend([f'(allow file-write* (subpath "{_quote(output)}"))', "(deny network*)"])
    rules.append(f'(allow file-write* (subpath "{_quote(scratch)}"))')
    return subprocess.run(
        [sandbox, "-p", "\n".join(rules), str(python_launcher), "-I", str(worker), *arguments],
        cwd=scratch,
        env={"HOME": str(scratch), "TMPDIR": str(scratch), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
