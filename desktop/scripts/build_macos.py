#!/usr/bin/env python3
"""Build the self-contained EasyICU macOS app and DMG."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
DESKTOP_ROOT = REPO_ROOT / "desktop"
BUILD_ROOT = DESKTOP_ROOT / ".build"
VENV_ROOT = BUILD_ROOT / "venv"
PYTHON_LOCK = DESKTOP_ROOT / "requirements-macos-arm64-py311.lock"
MIN_NODE = (22, 19, 0)


def _run(command: list[str], *, cwd: Path = REPO_ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _select_python() -> str:
    configured = str(os.environ.get("EASYICU_DESKTOP_PYTHON") or "").strip()
    candidates = [configured] if configured else []
    candidates.extend([sys.executable, "python3.11"])
    for candidate in candidates:
        executable = shutil.which(candidate) if not Path(candidate).is_absolute() else candidate
        if not executable:
            continue
        result = subprocess.run(
            [str(executable), "-c", "import sys; print(sys.version_info[:2] == (3, 11))"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip() == "True":
            return str(executable)
    raise RuntimeError("The macOS arm64 dependency lock requires Python 3.11")


def _venv_python() -> Path:
    return VENV_ROOT / "bin" / "python"


def _prepare_python_runtime(selected_python: str) -> Path:
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    if not _venv_python().exists():
        _run([selected_python, "-m", "venv", str(VENV_ROOT)])
    python = _venv_python()
    _run(
        [str(python), "-m", "pip", "install", "--require-hashes", "-r", str(PYTHON_LOCK)]
    )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            ".[webapp]",
        ]
    )
    _run([str(python), "-m", "pip", "check"])
    return python


def _node_version(node: Path) -> tuple[int, int, int]:
    result = subprocess.run(
        [str(node), "--version"], capture_output=True, text=True, check=True
    )
    parts = result.stdout.strip().removeprefix("v").split(".")
    if len(parts) < 3 or not all(part.isdigit() for part in parts[:3]):
        raise RuntimeError("The selected Node binary did not report a semantic version")
    return tuple(int(part) for part in parts[:3])


def _select_node() -> Path:
    configured = str(os.environ.get("EASYICU_DESKTOP_NODE") or "").strip()
    raw = configured or shutil.which("node") or ""
    if not raw:
        raise RuntimeError("Node 22.19+ is required to build the Copilot runtime")
    node = Path(raw).expanduser().resolve()
    if _node_version(node) < MIN_NODE:
        raise RuntimeError("Node 22.19+ is required to build the Copilot runtime")
    return node


def _node_license(node: Path) -> Path:
    for parent in [node.parent, *node.parents]:
        candidate = parent / "LICENSE"
        if candidate.is_file():
            return candidate
    raise RuntimeError("The selected Node distribution has no LICENSE file")


def _installed_node_app(python: Path) -> Path:
    result = subprocess.run(
        [
            str(python), "-c",
            "from importlib.util import find_spec; from pathlib import Path; "
            "print(Path(next(iter(find_spec('easyicu').submodule_search_locations))) "
            "/ 'webserver' / 'pi_copilot' / 'node_app')",
        ],
        cwd=BUILD_ROOT, capture_output=True, text=True, check=True,
    )
    return Path(result.stdout.strip())


def _prepare_node_runtime(node: Path, python: Path) -> None:
    _run(["npm", "ci", "--ignore-scripts", "--omit=dev"], cwd=_installed_node_app(python))
    resources = DESKTOP_ROOT / "src-tauri" / "resources"
    resources.mkdir(parents=True, exist_ok=True)
    target_name = "node.exe" if sys.platform == "win32" else "node"
    target = resources / target_name
    shutil.copy2(node, target)
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    shutil.copy2(_node_license(node), resources / "NODE_LICENSE")


def _build_backend(python: Path) -> Path:
    dist = BUILD_ROOT / "pyinstaller-dist"
    work = BUILD_ROOT / "pyinstaller-work"
    spec = BUILD_ROOT / "pyinstaller-spec"
    _run(
        [
            str(python),
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onedir",
            "--contents-directory",
            "_internal",
            "--name",
            "easyicu-backend",
            "--collect-submodules",
            "easyicu",
            "--collect-data",
            "easyicu",
            "--copy-metadata",
            "easyicu",
            "--hidden-import",
            "uvicorn.logging",
            "--hidden-import",
            "uvicorn.loops.auto",
            "--hidden-import",
            "uvicorn.protocols.http.auto",
            "--hidden-import",
            "uvicorn.protocols.websockets.auto",
            "--hidden-import",
            "uvicorn.lifespan.on",
            "--distpath",
            str(dist),
            "--workpath",
            str(work),
            "--specpath",
            str(spec),
            str(DESKTOP_ROOT / "backend_entry.py"),
        ]
    )
    backend_root = dist / "easyicu-backend"
    backend = backend_root / "easyicu-backend"
    if not backend.is_file():
        raise RuntimeError(f"PyInstaller did not create {backend}")
    resources = DESKTOP_ROOT / "src-tauri" / "resources"
    packaged_root = resources / "backend"
    if packaged_root.exists():
        shutil.rmtree(packaged_root)
    shutil.copytree(backend_root, packaged_root, symlinks=True)
    packaged = packaged_root / "easyicu-backend"
    packaged.chmod(packaged.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return packaged


def _build_tauri() -> None:
    _run(["npm", "ci"], cwd=DESKTOP_ROOT)
    _run(["npm", "run", "tauri", "--", "build", "--bundles", "app"], cwd=DESKTOP_ROOT)
    bundle_root = DESKTOP_ROOT / "src-tauri" / "target" / "release" / "bundle"
    app = bundle_root / "macos" / "EasyICU.app"
    if not app.is_dir():
        raise RuntimeError(f"Tauri did not create {app}")

    # Finder metadata copied into this generated bundle prevents codesigning.
    # Act on symlinks themselves so cleanup stays within the build artifact.
    _run(["xattr", "-crs", str(app)])
    identity = str(os.environ.get("APPLE_SIGNING_IDENTITY") or "-").strip() or "-"
    _run(["codesign", "--force", "--deep", "--sign", identity, str(app)])
    _run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)])

    dmg_dir = bundle_root / "dmg"
    dmg_dir.mkdir(parents=True, exist_ok=True)
    dmg = dmg_dir / "EasyICU_1.0.0_aarch64.dmg"
    dmg.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="easyicu-dmg-", dir=BUILD_ROOT) as raw:
        staging = Path(raw)
        shutil.copytree(app, staging / "EasyICU.app", symlinks=True)
        os.symlink("/Applications", staging / "Applications")
        _run(
            [
                "hdiutil",
                "create",
                "-volname",
                "EasyICU",
                "-srcfolder",
                str(staging),
                "-ov",
                "-format",
                "UDZO",
                str(dmg),
            ]
        )
    if not dmg.is_file():
        raise RuntimeError(f"hdiutil did not create {dmg}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="Prepare the frozen backend and bundled Node runtime without Tauri packaging.",
    )
    args = parser.parse_args(argv)
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("This dependency lock targets macOS Apple Silicon only")
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["EASYICU_HOME"] = str(BUILD_ROOT / "build-state")
    node = _select_node()
    python = _prepare_python_runtime(_select_python())
    _prepare_node_runtime(node, python)
    inputs = {
        "schema_version": "easyicu.desktop_build_inputs/1",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "source_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT, text=True,
        ).strip()),
        "python_version": subprocess.check_output(
            [str(python), "--version"], text=True
        ).strip(),
        "node_version": ".".join(map(str, _node_version(node))),
        "lock_sha256": {
            str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                PYTHON_LOCK, DESKTOP_ROOT / "package-lock.json",
                DESKTOP_ROOT / "src-tauri" / "Cargo.lock",
                REPO_ROOT / "src/easyicu/webserver/pi_copilot/node_app/package-lock.json",
            )
        },
    }
    (BUILD_ROOT / "build-inputs.json").write_text(
        json.dumps(inputs, indent=2) + "\n", encoding="utf-8"
    )
    backend = _build_backend(python)
    print(f"Frozen backend: {backend}")
    if not args.backend_only:
        _build_tauri()
        print(f"App bundles: {DESKTOP_ROOT / 'src-tauri' / 'target' / 'release' / 'bundle'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
