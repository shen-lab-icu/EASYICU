"""Auditable runtime identity + isolation-backend capability for the preflight.

Supervisor finding (2026-07-22): the batch-1 divergence between a green idle run
and a red run is NOT a wall-clock timeout.  It is the CodeRunner's macOS
``sandbox-exec`` confinement being *denied when nested* inside an already
sandboxed session: the step dies instantly with ``returncode=71`` and
``sandbox-exec: sandbox_apply: Operation not permitted`` (``timed_out=false``,
duration ~0.009s), and the runner currently reports that as a generic
``repair_failed`` rather than an explicit isolation-backend failure.

This module therefore records:

* an auditable *runtime identity* for the parent and a probe subprocess launched
  the way the CodeRunner launches steps (same interpreter, ``PYTHONNOUSERSITE``,
  worktree ``PYTHONPATH``), failing closed on any interpreter / worktree /
  dependency-version drift; and
* the *isolation backend* and whether it can actually apply here, so a nested
  sandbox denial is reported as ``isolation_backend_unavailable`` instead of
  masquerading as a code failure.  It never enables an unsafe host fallback.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import easyicu

# The dependency surface whose version drift would change real-stats behaviour.
AUDITED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",
    "pydantic",
)

# Mirror the CodeRunner's step interpreter: env["PYTHONPATH"] is the src root
# that owns the loaded research_agent package (runner.py sets parents[3]).
_WORKTREE_SRC = str(Path(easyicu.__file__).resolve().parents[1])

# Minimal profile mirroring the shape the CodeRunner passes to sandbox-exec.
_SANDBOX_PROBE_PROFILE = "(version 1)(allow default)"


def _package_versions() -> Dict[str, str]:
    import importlib.metadata as md

    out: Dict[str, str] = {}
    for name in AUDITED_PACKAGES:
        try:
            out[name] = md.version(name)
        except Exception:  # noqa: BLE001
            out[name] = "unavailable"
    return out


def parent_runtime_identity() -> Dict[str, object]:
    """Runtime identity of the current (test/parent) process."""

    return {
        "role": "parent",
        "executable": sys.executable,
        "python_version": sys.version.split()[0],
        "easyicu_file": str(Path(easyicu.__file__).resolve()),
        "worktree_src": _WORKTREE_SRC,
        "sys_path_head": list(sys.path[:4]),
        "packages": _package_versions(),
    }


_SUBPROCESS_PROBE = (
    "import json, sys, importlib.metadata as md\n"
    "import easyicu\n"
    f"names = {list(AUDITED_PACKAGES)!r}\n"
    "vers = {}\n"
    "for n in names:\n"
    "    try: vers[n] = md.version(n)\n"
    "    except Exception: vers[n] = 'unavailable'\n"
    "print(json.dumps({\n"
    "  'role': 'subprocess',\n"
    "  'executable': sys.executable,\n"
    "  'python_version': sys.version.split()[0],\n"
    "  'easyicu_file': easyicu.__file__,\n"
    "  'packages': vers,\n"
    "}))\n"
)


def subprocess_runtime_identity() -> Dict[str, object]:
    """Runtime identity of a probe subprocess launched like a CodeRunner step."""

    env = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": _WORKTREE_SRC,
    }
    proc = subprocess.run(  # noqa: S603 - fixed probe, no shell
        [sys.executable, "-c", _SUBPROCESS_PROBE],
        capture_output=True,
        text=True,
        timeout=60.0,
        env=env,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "preflight runtime probe subprocess failed: "
            f"{proc.stderr.strip() or proc.returncode}"
        )
    identity = json.loads(proc.stdout.strip().splitlines()[-1])
    identity["easyicu_file"] = str(Path(identity["easyicu_file"]).resolve())
    return identity


@dataclass
class IsolationCapability:
    """Whether the CodeRunner's isolation backend can actually apply here."""

    backend: str
    available: bool
    returncode: Optional[int] = None
    detail: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "backend": self.backend,
            "available": self.available,
            "returncode": self.returncode,
            "detail": self.detail,
        }


def probe_isolation_backend() -> IsolationCapability:
    """Probe the host isolation backend the subprocess runner would use.

    On macOS the CodeRunner confines every step with ``sandbox-exec``.  We run a
    trivial command under a minimal profile: a nested/denied sandbox returns
    non-zero with ``sandbox_apply: Operation not permitted`` (the supervisor's
    ``returncode=71``).  On non-macOS hosts the subprocess runner does not use
    ``sandbox-exec``; report the platform backend as available so the real gate
    still runs.
    """

    import shutil

    if sys.platform != "darwin":
        return IsolationCapability(
            backend=f"host_subprocess_{sys.platform}",
            available=True,
            detail="sandbox-exec confinement is macOS-only",
        )
    sandbox_exec = shutil.which("sandbox-exec")
    if not sandbox_exec:
        return IsolationCapability(
            backend="macos_sandbox_exec",
            available=False,
            detail="sandbox-exec not found on PATH",
        )
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sandbox_exec, "-p", _SANDBOX_PROBE_PROFILE, "/usr/bin/true"],
            capture_output=True,
            text=True,
            timeout=30.0,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:  # noqa: BLE001
        return IsolationCapability(
            backend="macos_sandbox_exec",
            available=False,
            detail=f"{type(exc).__name__}: {exc}",
        )
    if proc.returncode == 0:
        return IsolationCapability(
            backend="macos_sandbox_exec", available=True, returncode=0
        )
    return IsolationCapability(
        backend="macos_sandbox_exec",
        available=False,
        returncode=proc.returncode,
        detail=(proc.stderr or "").strip() or "sandbox-exec returned non-zero",
    )


# Signature of a nested-sandbox denial in a CodeRunner step record.
_SANDBOX_DENIED_MARKERS = (
    "sandbox_apply: operation not permitted",
    "sandbox-exec: sandbox_apply",
)


def step_isolation_unavailable(
    step_record: Dict[str, object],
    capability: "IsolationCapability",
) -> Optional[str]:
    """Classify a step failure as a nested-sandbox isolation-backend denial.

    Returns the denial detail only when **all** of the following hold — any
    other failure (including a script that legitimately exits 71) is a genuine
    execution failure, never an isolation outage:

    * the active isolation backend is ``macos_sandbox_exec`` (only that backend
      can be refused by a nested sandbox);
    * the capability probe itself came back *unavailable* (a working backend
      would have let the step run);
    * the step's persisted stderr carries the exact ``sandbox_apply`` denial.

    ``returncode == 71`` alone is deliberately **not** sufficient: a generated
    script may ``sys.exit(71)`` for its own reasons, and that must still be
    judged an execution failure.  The discriminator is the persisted sandbox
    denial marker, not the exit code.
    """

    if capability.backend != "macos_sandbox_exec":
        return None
    if capability.available:
        return None
    if step_record.get("timed_out"):
        return None
    stderr = str(step_record.get("stderr") or "").lower()
    if any(marker in stderr for marker in _SANDBOX_DENIED_MARKERS):
        return stderr.strip()[:400]
    return None


@dataclass
class RuntimeManifest:
    """Parent + subprocess identities, isolation capability, fail-closed verdict."""

    parent: Dict[str, object]
    subprocess: Dict[str, object]
    isolation: IsolationCapability
    mismatches: List[str] = field(default_factory=list)

    @property
    def compatible(self) -> bool:
        """No interpreter / worktree / dependency drift between parent + child."""

        return not self.mismatches

    @property
    def integration_ready(self) -> bool:
        """True only when the real subprocess gate may run *and* be trusted.

        Requires a compatible runtime (no drift) **and** an available isolation
        backend.  When false, the real-subprocess E1/E2/E3 gate must surface a
        structured :attr:`blocked_reason` — never a silent pass and never an
        unexplained skip.  It never permits an unsafe host fallback.
        """

        return self.compatible and self.isolation.available

    @property
    def blocked_reason(self) -> Optional[str]:
        """Structured reason the real subprocess gate cannot run, else None."""

        if not self.compatible:
            return "runtime_incompatible: " + "; ".join(self.mismatches)
        if not self.isolation.available:
            detail = self.isolation.detail or self.isolation.returncode
            return f"isolation_backend_unavailable: {self.isolation.backend} ({detail})"
        return None

    def as_dict(self) -> Dict[str, object]:
        return {
            "parent": self.parent,
            "subprocess": self.subprocess,
            "isolation": self.isolation.as_dict(),
            "mismatches": self.mismatches,
            "compatible": self.compatible,
            "integration_ready": self.integration_ready,
            "blocked_reason": self.blocked_reason,
        }


def build_runtime_manifest() -> RuntimeManifest:
    """Capture identities + isolation capability and diff load-bearing fields."""

    parent = parent_runtime_identity()
    child = subprocess_runtime_identity()
    mismatches: List[str] = []
    if child.get("executable") != parent.get("executable"):
        mismatches.append(
            f"interpreter: parent={parent.get('executable')} "
            f"subprocess={child.get('executable')}"
        )
    if child.get("easyicu_file") != parent.get("easyicu_file"):
        mismatches.append(
            f"easyicu root: parent={parent.get('easyicu_file')} "
            f"subprocess={child.get('easyicu_file')}"
        )
    for name in AUDITED_PACKAGES:
        pv = parent.get("packages", {}).get(name)  # type: ignore[union-attr]
        cv = child.get("packages", {}).get(name)  # type: ignore[union-attr]
        if pv != cv:
            mismatches.append(f"{name}: parent={pv} subprocess={cv}")
    return RuntimeManifest(
        parent=parent,
        subprocess=child,
        isolation=probe_isolation_backend(),
        mismatches=mismatches,
    )


def write_runtime_manifest(run_dir: Path, manifest: RuntimeManifest) -> Path:
    path = Path(run_dir) / "preflight_runtime_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


__all__ = [
    "AUDITED_PACKAGES",
    "IsolationCapability",
    "RuntimeManifest",
    "build_runtime_manifest",
    "parent_runtime_identity",
    "probe_isolation_backend",
    "step_isolation_unavailable",
    "subprocess_runtime_identity",
    "write_runtime_manifest",
]
