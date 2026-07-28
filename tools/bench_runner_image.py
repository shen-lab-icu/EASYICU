"""Require a benchmark launch to name the analysis runner image explicitly.

``DockerRunner`` resolves its image as ``EASYICU_RUNNER_IMAGE`` else its own
``DEFAULT_IMAGE``.  That fallback is right for a library -- a caller that does
not care gets something that works -- and wrong for a benchmark launch, where
the whole point is that the image matches the source under test.  On 2026-07-28
an E1 launch went out right after a fresh exact-SHA image was built, inherited
the stale default, and was stopped by the exact-SHA integrity gate at preflight.
Nothing was spent, but the run had to be relaunched.

The rule lives here, in the installed tree, rather than in the launcher script:
the launcher sits outside any git repository, so a check written there is
protected by nothing -- not review, not CI, not this test suite.  The shell
script calls this module and reports what it says.

Usage from a shell launcher::

    python -m tools.bench_runner_image || exit 2
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Mapping, Sequence

__all__ = [
    "RunnerImageError",
    "available_runner_images",
    "main",
    "resolve_required_runner_image",
]

ENV_VAR = "EASYICU_RUNNER_IMAGE"
_IMAGE_REPOSITORY = "easyicu-research-agent"


class RunnerImageError(RuntimeError):
    """The launch does not name a usable runner image."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def available_runner_images(
    *,
    docker: str | None = None,
) -> tuple[str, ...]:
    """Return locally built runner images, newest first, for the error text.

    A launch that is being refused should say what the operator could have
    passed.  Failing to list them is not itself an error: the refusal stands
    either way.
    """

    binary = docker or shutil.which("docker")
    if not binary:
        return ()
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                binary,
                "image",
                "ls",
                _IMAGE_REPOSITORY,
                "--format",
                "{{.Repository}}:{{.Tag}}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if completed.returncode != 0:
        return ()
    return tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())


def _image_exists_locally(image: str, *, docker: str | None = None) -> bool:
    binary = docker or shutil.which("docker")
    if not binary:
        # Without docker we cannot verify; the explicit-naming rule still held,
        # and the runtime will fail loudly on its own.
        return True
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [binary, "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    return completed.returncode == 0


def resolve_required_runner_image(
    env: Mapping[str, str] | None = None,
    *,
    verify_present: bool = True,
    docker: str | None = None,
) -> str:
    """Return the explicitly named runner image, or raise.

    Raises :class:`RunnerImageError` with ``reason_code``
    ``runner_image_not_declared`` when the variable is absent or blank, and
    ``runner_image_not_present`` when it names an image this host does not have.
    """

    source = os.environ if env is None else env
    image = str(source.get(ENV_VAR, "") or "").strip()
    if not image:
        available = available_runner_images(docker=docker)
        lines = [
            f"{ENV_VAR} is not set.",
            "",
            "  Refusing to launch rather than letting DockerRunner fall back to",
            "  its default image, which is almost never the one you just built.",
        ]
        if available:
            lines += ["", "  Locally built images:"]
            lines += [f"    {name}" for name in available]
        lines += ["", f"  Then: {ENV_VAR}=<image> <launcher> ..."]
        raise RunnerImageError("runner_image_not_declared", "\n".join(lines))
    if verify_present and not _image_exists_locally(image, docker=docker):
        raise RunnerImageError(
            "runner_image_not_present",
            f"{ENV_VAR}={image!r} is not a local image.\n"
            "  A typo here would otherwise surface much later, inside the run.",
        )
    return image


def main(argv: Sequence[str] | None = None) -> int:
    """Print the resolved image, or the refusal, and return the exit status."""

    try:
        print(resolve_required_runner_image())
    except RunnerImageError as exc:
        import sys

        print(f"bench launch refused [{exc.reason_code}]: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
