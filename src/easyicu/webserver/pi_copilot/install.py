"""Explicit installer for the pinned Pi Copilot Node runtime."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

PI_PACKAGE_VERSION = "0.84.1"
RUNTIME_FILES = (
    "package.json",
    "package-lock.json",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
)


def packaged_app_dir() -> Path:
    return Path(__file__).resolve().with_name("node_app")


def user_runtime_dir(*, home: Optional[Path] = None) -> Path:
    root = Path(home) if home is not None else Path.home()
    return root / ".easyicu" / "pi-agent" / "runtime" / PI_PACKAGE_VERSION


def runtime_is_installed(path: Path) -> bool:
    root = Path(path)
    return all((root / name).is_file() for name in RUNTIME_FILES) and all(
        candidate.is_file()
        for candidate in (
            root / "src" / "main.mjs",
            root
            / "node_modules"
            / "@earendil-works"
            / "pi-coding-agent"
            / "package.json",
        )
    )


def preferred_app_dir() -> Path:
    installed = user_runtime_dir()
    return installed if runtime_is_installed(installed) else packaged_app_dir()


def _installer_environment(source: Mapping[str, str]) -> dict[str, str]:
    allowed = {"PATH", "HOME", "TMPDIR", "TEMP", "LANG", "LC_ALL", "LC_CTYPE"}
    return {
        key: str(value)
        for key, value in source.items()
        if key in allowed or key.startswith("LC_")
    }


def install_runtime(
    *,
    destination: Optional[Path] = None,
    source: Optional[Path] = None,
    npm_binary: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Path:
    """Install the exact lockfile into a private, versioned user runtime."""

    source_dir = Path(source or packaged_app_dir()).resolve()
    target = Path(destination or user_runtime_dir()).resolve()
    if runtime_is_installed(target):
        return target
    if target.exists():
        raise RuntimeError(
            f"Pi runtime target exists but is incomplete: {target}. Remove it explicitly and retry."
        )
    for relative in (*RUNTIME_FILES, "src/main.mjs"):
        if not (source_dir / relative).is_file():
            raise RuntimeError(f"Packaged Pi runtime file is missing: {relative}")
    source_environment = os.environ if environ is None else environ
    npm = npm_binary or shutil.which("npm", path=source_environment.get("PATH"))
    if not npm:
        raise RuntimeError("npm is required to install the pinned Pi runtime")

    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{PI_PACKAGE_VERSION}-", dir=target.parent)
    )
    try:
        for name in RUNTIME_FILES:
            shutil.copy2(source_dir / name, staging / name)
        shutil.copytree(source_dir / "src", staging / "src")
        subprocess.run(
            [npm, "ci", "--ignore-scripts"],
            cwd=staging,
            env=_installer_environment(source_environment),
            check=True,
        )
        if not runtime_is_installed(staging):
            raise RuntimeError("npm completed without the pinned Pi dependency")
        staging.chmod(0o700)
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=None,
        help="Override the private versioned runtime directory.",
    )
    args = parser.parse_args(argv)
    installed = install_runtime(destination=args.runtime_dir)
    print(f"Installed pinned Pi Copilot runtime {PI_PACKAGE_VERSION} at {installed}")
    return 0


__all__ = [
    "PI_PACKAGE_VERSION",
    "install_runtime",
    "main",
    "packaged_app_dir",
    "preferred_app_dir",
    "runtime_is_installed",
    "user_runtime_dir",
]
