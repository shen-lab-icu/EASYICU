"""Bounded preflight for the native runtime of the counting-process owner.

Probe the immutable selected image, not the host's R installation. No data or
host directories are mounted; the disposable probe has no network access.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
import shutil
import subprocess


@dataclass(frozen=True)
class NativeMethodRuntime:
    available: bool
    reason_code: str
    image_id: str | None = None
    r_version: str | None = None
    survival_version: str | None = None


@lru_cache(maxsize=8)
def _probe_survival(docker: str, image_id: str) -> NativeMethodRuntime:
    try:
        result = subprocess.run(
            [
                docker,
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,nosuid,size=16m",
                "--cap-drop=ALL",
                "--pids-limit",
                "64",
                "--entrypoint",
                "Rscript",
                image_id,
                "--vanilla",
                "-e",
                'suppressPackageStartupMessages(library(survival)); stopifnot(is.function(coxph)); cat(as.character(getRversion()), as.character(packageVersion("survival")), sep="|")',
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return NativeMethodRuntime(
            False, "time_varying_native_runtime_probe_failed", image_id
        )
    parts = result.stdout.strip().split("|")
    if result.returncode or len(parts) != 2 or not all(parts):
        return NativeMethodRuntime(
            False, "time_varying_r_survival_unavailable", image_id
        )
    return NativeMethodRuntime(True, "ready", image_id, *parts)


def probe_time_varying_native_runtime(image: str) -> NativeMethodRuntime:
    docker = shutil.which(os.environ.get("EASYICU_DOCKER_EXECUTABLE") or "docker")
    if docker is None:
        return NativeMethodRuntime(False, "docker_executable_missing")
    try:
        result = subprocess.run(
            [docker, "image", "inspect", image, "--format={{.Id}}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return NativeMethodRuntime(False, "docker_probe_failed")
    image_id = result.stdout.strip()
    if result.returncode or not image_id.startswith("sha256:"):
        return NativeMethodRuntime(False, "docker_image_unavailable")
    return _probe_survival(docker, image_id)


__all__ = ["NativeMethodRuntime", "probe_time_varying_native_runtime"]
