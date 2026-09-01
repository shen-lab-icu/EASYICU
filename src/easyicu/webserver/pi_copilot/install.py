"""Explicit installer for the pinned Pi Copilot Node runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

from easyicu.webserver import state_paths

PI_PACKAGE_VERSION = "0.84.1"
RUNTIME_FILES = (
    "package.json",
    "package-lock.json",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
)
RUNTIME_SOURCE_FILES = (
    "src/main.mjs",
    "src/event-projection.mjs",
    "src/shell-budget.mjs",
    "src/session-lifecycle.mjs",
    "src/skills/web-prototype/SKILL.md",
)
RUNTIME_MANIFEST_FILE = "runtime-manifest.json"
INSTALLATION_SCHEMA_VERSION = "easyicu.pi-runtime-installation/2"
_EXECUTABLE_SUFFIXES = frozenset({".js", ".mjs", ".cjs"})


def packaged_app_dir() -> Path:
    return Path(__file__).resolve().with_name("node_app")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_manifest(source: Optional[Path] = None) -> dict[str, object]:
    """Compile the content identity expected for one executable runtime."""

    root = Path(source or packaged_app_dir()).resolve()
    files = {}
    for relative in (*RUNTIME_FILES, *RUNTIME_SOURCE_FILES):
        candidate = root / relative
        if not candidate.is_file():
            raise RuntimeError(f"Packaged Pi runtime file is missing: {relative}")
        files[relative] = _sha256(candidate)
    try:
        lock = json.loads((root / "package-lock.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError("Packaged Pi runtime lockfile is invalid") from exc
    packages = lock.get("packages") if isinstance(lock, dict) else None
    if not isinstance(packages, dict):
        raise RuntimeError("Packaged Pi runtime lockfile has no packages map")
    pi_packages = {
        str(relative): str(metadata.get("version") or "")
        for relative, metadata in packages.items()
        if isinstance(metadata, dict)
        and str(relative).split("node_modules/")[-1].startswith("@earendil-works/pi-")
    }
    if not pi_packages or any(not version for version in pi_packages.values()):
        raise RuntimeError("Packaged Pi runtime has incomplete pinned Pi versions")
    identity = {
        "schema_version": "easyicu.pi-runtime-manifest/1",
        "pi_package_version": PI_PACKAGE_VERSION,
        "files": files,
        "pi_packages": dict(sorted(pi_packages.items())),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**identity, "runtime_manifest_sha256": digest}


def _installed_executable_hashes(
    root: Path,
) -> dict[str, str]:
    """Hash the complete installed production JavaScript dependency tree."""

    files: dict[str, str] = {}
    node_modules = root / "node_modules"
    if not node_modules.is_dir():
        raise RuntimeError("Installed Pi runtime has no production dependency tree")
    for candidate in sorted(node_modules.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.name != "package.json" and candidate.suffix not in _EXECUTABLE_SUFFIXES:
            continue
        files[candidate.relative_to(root).as_posix()] = _sha256(candidate)
    if not files:
        raise RuntimeError("Installed Pi runtime has no executable dependency files")
    return files


def _installation_manifest(
    source_manifest: Mapping[str, object],
    installed_root: Path,
    *,
    node_version: str,
) -> dict[str, object]:
    executable_files = _installed_executable_hashes(installed_root)
    return {
        **dict(source_manifest),
        "installation": {
            "schema_version": INSTALLATION_SCHEMA_VERSION,
            "node_version": node_version,
            "executable_files": executable_files,
        },
    }


PI_RUNTIME_REVISION = (
    f"{PI_PACKAGE_VERSION}-{str(runtime_manifest()['runtime_manifest_sha256'])[:12]}-install2"
)


def user_runtime_dir(*, home: Optional[Path] = None) -> Path:
    root = Path(home) if home is not None else state_paths.user_home()
    return root / ".easyicu" / "pi-agent" / "runtime" / PI_RUNTIME_REVISION


def runtime_is_installed(path: Path, *, source: Optional[Path] = None) -> bool:
    root = Path(path)
    try:
        expected = runtime_manifest(source)
        installed = json.loads(
            (root / RUNTIME_MANIFEST_FILE).read_text(encoding="utf-8")
        )
        if any(installed.get(key) != value for key, value in expected.items()):
            return False
        installation = installed.get("installation")
        if not isinstance(installation, dict):
            return False
        if installation.get("schema_version") != INSTALLATION_SCHEMA_VERSION:
            return False
        node_version = str(installation.get("node_version") or "").strip()
        if not node_version:
            return False
        for relative, digest in dict(expected["files"]).items():
            candidate = root / relative
            if not candidate.is_file() or _sha256(candidate) != digest:
                return False
        for relative, version in dict(expected["pi_packages"]).items():
            package_file = root / relative / "package.json"
            if not package_file.is_file():
                return False
            package = json.loads(package_file.read_text(encoding="utf-8"))
            if str(package.get("version") or "") != version:
                return False
        executable_files = installation.get("executable_files")
        if not isinstance(
            executable_files, dict
        ) or executable_files != _installed_executable_hashes(root):
            return False
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False
    return True


def packaged_runtime_is_complete(path: Optional[Path] = None) -> bool:
    """Validate packaged source plus installed exact Pi package versions."""

    root = Path(path or packaged_app_dir()).resolve()
    try:
        expected = runtime_manifest(root)
        for relative, version in dict(expected["pi_packages"]).items():
            package_file = root / relative / "package.json"
            if not package_file.is_file():
                return False
            package = json.loads(package_file.read_text(encoding="utf-8"))
            if str(package.get("version") or "") != version:
                return False
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False
    return True


def preferred_app_dir() -> Path:
    installed = user_runtime_dir()
    # An existing but invalid private runtime must remain visible to the
    # gateway's integrity gate; silently falling back would hide tampering.
    return installed if installed.exists() else packaged_app_dir()


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
    node_binary: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Path:
    """Install the exact lockfile into a private, versioned user runtime."""

    source_dir = Path(source or packaged_app_dir()).resolve()
    target = Path(destination or user_runtime_dir()).resolve()
    expected_manifest = runtime_manifest(source_dir)
    if runtime_is_installed(target, source=source_dir):
        return target
    if target.exists():
        raise RuntimeError(
            "pi_runtime_integrity_mismatch: Pi runtime target exists but does "
            f"not match the packaged content manifest: {target}. Remove it explicitly and retry."
        )
    source_environment = os.environ if environ is None else environ
    npm = npm_binary or shutil.which("npm", path=source_environment.get("PATH"))
    if not npm:
        raise RuntimeError("npm is required to install the pinned Pi runtime")
    node = node_binary or shutil.which("node", path=source_environment.get("PATH"))
    if not node:
        sibling = Path(npm).with_name("node")
        node = str(sibling) if sibling.is_file() else None
    if not node:
        raise RuntimeError("node is required to record the Pi runtime identity")

    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{PI_RUNTIME_REVISION}-", dir=target.parent)
    )
    try:
        for name in RUNTIME_FILES:
            shutil.copy2(source_dir / name, staging / name)
        shutil.copytree(source_dir / "src", staging / "src")
        subprocess.run(
            [npm, "ci", "--ignore-scripts", "--omit=dev"],
            cwd=staging,
            env=_installer_environment(source_environment),
            check=True,
        )
        node_result = subprocess.run(
            [node, "--version"],
            cwd=staging,
            env=_installer_environment(source_environment),
            check=True,
            capture_output=True,
            text=True,
        )
        node_version = str(node_result.stdout or "").strip().removeprefix("v")
        if not node_version:
            raise RuntimeError("node did not report a runtime version")
        installed_receipt = _installation_manifest(
            expected_manifest,
            staging,
            node_version=node_version,
        )
        (staging / RUNTIME_MANIFEST_FILE).write_text(
            json.dumps(installed_receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / RUNTIME_MANIFEST_FILE).chmod(0o600)
        # A custom source is useful for installer tests. Validate its exact
        # manifest directly before the atomic rename.
        installed_manifest = json.loads(
            (staging / RUNTIME_MANIFEST_FILE).read_text(encoding="utf-8")
        )
        if installed_manifest != installed_receipt:
            raise RuntimeError("Pi runtime manifest changed during installation")
        for relative, digest in dict(expected_manifest["files"]).items():
            if _sha256(staging / relative) != digest:
                raise RuntimeError(f"Pi runtime integrity mismatch: {relative}")
        for relative, version in dict(expected_manifest["pi_packages"]).items():
            package = json.loads(
                (staging / relative / "package.json").read_text(encoding="utf-8")
            )
            if str(package.get("version") or "") != version:
                raise RuntimeError(f"Pi runtime package version mismatch: {relative}")
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
    "PI_RUNTIME_REVISION",
    "RUNTIME_MANIFEST_FILE",
    "install_runtime",
    "main",
    "packaged_app_dir",
    "preferred_app_dir",
    "packaged_runtime_is_complete",
    "runtime_manifest",
    "runtime_is_installed",
    "user_runtime_dir",
]
