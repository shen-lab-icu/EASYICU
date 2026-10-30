"""Closed, host-neutral remediation for execution-runtime refusal codes."""

from __future__ import annotations

import json
import re
import subprocess
import sys

_RUNNER_UNAVAILABLE_REMEDIATION: dict[str, str] = {
    "host_runner_refuses_model_code": (
        "Declare a supported typed executor for every desktop step, or use "
        "the Docker profile for model-generated or repaired code."
    ),
    "host_runtime_unavailable": (
        "The desktop Python runtime is incomplete. Repair the EasyICU installation "
        "with its required analysis packages and psutil, then restart the app."
    ),
    "docker_daemon_unreachable": (
        "The Docker daemon is not responding. Start Docker (Docker Desktop, "
        "'colima start', ...) and retry."
    ),
    "docker_executable_missing": (
        "No 'docker' executable was found on PATH or in the standard local "
        "install locations. Install Docker, or set EASYICU_DOCKER_EXECUTABLE to "
        "the binary if it lives somewhere non-standard."
    ),
    "docker_image_missing": (
        "The pinned execution image is not present locally. Build or pull it "
        "with a live Docker daemon and retry."
    ),
    "docker_probe_failed": (
        "The Docker probe failed or returned an invalid response. Check the "
        "Docker context, permissions and runtime health; image absence was not established."
    ),
    "docker_workspace_unavailable": (
        "The execution workspace is not readable and writable inside Docker. "
        "Check Docker or Colima file sharing and workspace permissions, then retry."
    ),
    "host_sandbox_missing": (
        "macOS 'sandbox-exec' was not found, so no filesystem-isolating host "
        "fallback is available."
    ),
    "runner_image_kernel_mismatch": (
        "The execution image was built from different EasyICU execution-kernel "
        "source than this host is running. Rebuild the image from the current "
        "commit, verify it with tools/check_agent_runtime.py, point "
        "EASYICU_RUNNER_IMAGE at it, and restart the service."
    ),
    "runner_image_lock_mismatch": (
        "The execution image's requirements.lock differs from the lock this "
        "host pins. Rebuild the image from the current commit and restart the "
        "service."
    ),
}


def runner_unavailable_remediation(reason_code: str) -> str:
    """Return the constant, host-neutral remediation for one closed reason."""

    return _RUNNER_UNAVAILABLE_REMEDIATION.get(
        str(reason_code), "The configured execution backend is not usable."
    )


_DOCKER_DAEMON_UNREACHABLE_MARKERS = (
    "cannot connect to the docker daemon",
    "is the docker daemon running",
    "docker daemon is not running",
    "error during connect",
)


def classify_docker_failure(stderr: str, stdout: str) -> str:
    """Collapse one docker CLI failure into a closed reason code.

    A missing image and a stopped daemon are different problems with different
    fixes, and only the daemon's message distinguishes them -- but that message
    carries the host socket path. Classify here, return the code, drop the text.
    """

    text = f"{stderr}\n{stdout}".strip().lower()
    # These are explicit image-lookup results; image names themselves may
    # contain words such as tls or certificate without reporting a failure.
    if "no such image:" in text or "no such object:" in text:
        return "docker_image_missing"
    # Configuration and remote-endpoint failures must not select a different
    # execution backend, even when Docker appends a generic connection hint.
    if any(marker in text for marker in (
        "permission denied", "access is denied", "x509", "tls", "certificate",
        "unauthorized", "authentication", "forbidden", "tcp://", "ssh://",
    )) or re.search(r"https?://(?!%2f)", text):
        return "docker_probe_failed"
    if ("failed to connect to the docker api" in text and "unix" in text
            and any(cause in text for cause in ("no such file or directory", "connection refused"))):
        return "docker_daemon_unreachable"
    if any(marker in text for marker in _DOCKER_DAEMON_UNREACHABLE_MARKERS):
        return "docker_daemon_unreachable"
    return "docker_probe_failed"


def host_runtime_available(probe_timeout_seconds: float) -> bool:
    """Probe the app interpreter's typed-executor imports without a cohort."""
    from ..contracts.method_packages import BASELINE_PACKAGES

    names = [*BASELINE_PACKAGES, "psutil"]
    source = (
        "import importlib.util, sys; "
        f"sys.exit(0 if all(importlib.util.find_spec(n) for n in {names!r}) else 1)"
    )
    try:
        command = [sys.executable, "-c", source]
        frozen = bool(getattr(sys, "frozen", False))
        if frozen:
            from easyicu.runtime.host_worker import frozen_worker_command

            command = frozen_worker_command(
                sys.executable, "capabilities", "--names", json.dumps(names),
            )
        result = subprocess.run(
            command, capture_output=True,
            timeout=max(0.1, probe_timeout_seconds), check=False,
        )
        return result.returncode == 0 and (not frozen or set(names).issubset(json.loads(result.stdout)))
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError):
        return False
