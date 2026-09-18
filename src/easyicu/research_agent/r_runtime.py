"""Shared local-R runtime rails: probe, fingerprint, hardened runner.

R enters EasyICU only through here plus one narrow adapter at a time.
``time_varying_exposure_cox`` was the first: it hand-rolled its subprocess
call.  New R adapters must call :func:`run_rscript` instead so timeouts,
warning promotion and output contracts behave identically everywhere.

What this module does NOT do: install R packages, reach the network, or
decide any scientific estimand.  R package installs stay operator-owned
(exactly like the Python ``methods`` extra); missing packages surface as
``None`` versions and fail closed at the calling adapter.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional, Sequence, Tuple

#: R packages the lane knows about. Present ones report versions; absent
#: ones report ``None`` and refuse at the calling adapter. This set mirrors
#: what ICU literature actually cites (mining 2026-09-18: R in 42.5% of
#: Methods sections) minus what no caller needs yet.
DECLARED_R_PACKAGES: Tuple[str, ...] = (
    "survival",
    "survminer",
    "mice",
    "lme4",
    "nlme",
    "mitools",
    "EValue",
)

#: High-frequency literature packages deliberately NOT installed. Installing
#: any of them is an operator decision (image rebuild + fingerprint
#: re-baseline), never a runtime download.
ABSENT_R_PACKAGES: Dict[str, str] = {
    "MatchIt": "not installed; propensity matching stays in methods/propensity_weighting.py",
    "WeightIt": "not installed; weighting stays in methods/propensity_weighting.py",
    "tableone": "not installed; Table One stays in methods/table_one.py",
    "rms": "not installed; no caller needs it yet",
    "cmprsk": "not installed; competing risks stay descriptive (methods/competing_risks.py)",
    "mediation": "not installed; mediation stays in methods/mediation.py",
    "geepack": "not installed; no caller needs it yet",
    "cobalt": "not installed; balance diagnostics stay in methods/propensity_weighting.py",
}

_R_VERSION_TIMEOUT_S = 60.0


class RRuntimeError(ValueError):
    """The local R runtime cannot satisfy the request."""


@dataclass(frozen=True)
class RRuntimeReceipt:
    """One probe of the local R runtime."""

    available: bool
    rscript_path: str = ""
    r_version: str = ""
    packages: Dict[str, Optional[str]] = field(default_factory=dict)
    reason_code: str = "r_runtime_ready"


@dataclass(frozen=True)
class RScriptResult:
    """Outcome of one hardened Rscript invocation (never raises)."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


_PROBE_RSCRIPT = r"""
cat(paste0("EASYICU_RVERSION=", as.character(getRversion()), "\n"))
for (pkg in commandArgs(trailingOnly=TRUE)) {
  version <- tryCatch(
    as.character(packageVersion(pkg)),
    error=function(e) "ABSENT"
  )
  cat(paste0("EASYICU_RPKG:", pkg, "=", version, "\n"))
}
"""


def _probe_once(rscript: str) -> RRuntimeReceipt:
    try:
        completed = subprocess.run(
            [rscript, "--vanilla", "-e", _PROBE_RSCRIPT, *DECLARED_R_PACKAGES],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=_R_VERSION_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return RRuntimeReceipt(available=False, reason_code="r_runtime_probe_failed")
    if completed.returncode != 0:
        return RRuntimeReceipt(available=False, reason_code="r_runtime_probe_failed")
    r_version = ""
    packages: Dict[str, Optional[str]] = {}
    for line in (completed.stdout or "").splitlines():
        if line.startswith("EASYICU_RVERSION="):
            r_version = line.split("=", 1)[1].strip()
        elif line.startswith("EASYICU_RPKG:"):
            rest = line.split(":", 1)[1]
            name, _, version = rest.partition("=")
            name, version = name.strip(), version.strip()
            packages[name] = None if version in ("", "ABSENT") else version
    if not r_version:
        return RRuntimeReceipt(available=False, reason_code="r_version_unreadable")
    for name in DECLARED_R_PACKAGES:
        packages.setdefault(name, None)
    return RRuntimeReceipt(
        available=True,
        rscript_path=rscript,
        r_version=r_version,
        packages=packages,
    )


@lru_cache(maxsize=1)
def _cached_probe(rscript: str) -> RRuntimeReceipt:
    return _probe_once(rscript)


def probe_r_runtime(*, refresh: bool = False) -> RRuntimeReceipt:
    """Probe the local Rscript runtime (cached; ``refresh=True`` re-probes)."""

    rscript = shutil.which("Rscript")
    if not rscript:
        return RRuntimeReceipt(
            available=False, reason_code="rscript_runtime_unavailable"
        )
    if refresh:
        _cached_probe.cache_clear()
    return _cached_probe(rscript)


def r_runtime_fingerprint(receipt: Optional[RRuntimeReceipt] = None) -> str:
    """Canonical fingerprint binding R version plus declared package versions."""

    checked = receipt if receipt is not None else probe_r_runtime()
    parts = [f"R/{checked.r_version or 'absent'}"]
    parts.extend(
        f"{name}/{checked.packages.get(name) or 'absent'}"
        for name in DECLARED_R_PACKAGES
    )
    return " ".join(parts)


def run_rscript(
    *,
    rscript: str,
    script: str,
    args: Sequence[str] = (),
    timeout_s: float = 120.0,
) -> RScriptResult:
    """Run one R script with hardened transport semantics (never raises).

    No network, no shell, fixed argv shape ``[rscript, --vanilla, -e,
    script, *args]``.  Timeouts, non-zero exits and unreadable output all
    surface as result fields; fail-closed mapping stays with the caller so
    each adapter keeps its own error codes.
    """

    if not str(rscript or "").strip():
        raise RRuntimeError("rscript path must be a non-empty string")
    if not str(script or "").strip():
        raise RRuntimeError("R script text must be non-empty")
    try:
        timeout_value = float(timeout_s)
    except (TypeError, ValueError) as exc:
        raise RRuntimeError(f"timeout_s must be numeric: {exc}") from exc
    if not timeout_value > 0:
        raise RRuntimeError("timeout_s must be positive")
    argv = [str(rscript), "--vanilla", "-e", str(script)]
    argv.extend(str(item) for item in (args or []))
    try:
        completed = subprocess.run(
            argv,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_value,
        )
    except subprocess.TimeoutExpired as exc:
        partial_out = exc.stdout
        partial_err = exc.stderr
        return RScriptResult(
            returncode=-1,
            stdout=partial_out if isinstance(partial_out, str) else "",
            stderr=partial_err if isinstance(partial_err, str) else "",
            timed_out=True,
        )
    except OSError as exc:
        raise RRuntimeError(f"Rscript execution failed to start: {exc}") from exc
    return RScriptResult(
        returncode=int(completed.returncode),
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        timed_out=False,
    )


__all__ = [
    "ABSENT_R_PACKAGES",
    "DECLARED_R_PACKAGES",
    "RRuntimeError",
    "RRuntimeReceipt",
    "RScriptResult",
    "probe_r_runtime",
    "r_runtime_fingerprint",
    "run_rscript",
]
