"""Installed-runtime build primitives shared by the desktop platform drivers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

MIN_NODE = (22, 19, 0)


def select_python(target: str) -> str:
    configured = str(os.environ.get("EASYICU_DESKTOP_PYTHON") or "").strip()
    candidates = ([configured] if configured else []) + [sys.executable, "python3.11"]
    for candidate in candidates:
        executable = shutil.which(candidate) if not Path(candidate).is_absolute() else candidate
        if not executable:
            continue
        result = subprocess.run(
            [str(executable), "-c", "import sys; print(sys.version_info[:2] == (3, 11))"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0 and result.stdout.strip() == "True":
            return str(executable)
    raise RuntimeError(f"The {target} dependency lock requires Python 3.11")


def prepare_python_runtime(selected_python, *, build_root, venv_root, python, lock, run):
    build_root.mkdir(parents=True, exist_ok=True)
    if not python.exists():
        run([selected_python, "-m", "venv", str(venv_root)])
    run([str(python), "-m", "pip", "install", "--require-hashes", "-r", str(lock)])
    # pip builds and installs a wheel, never an editable checkout. The locked
    # build requirements supply the backend without an unpinned build env.
    run([str(python), "-m", "pip", "install", "--no-deps", "--no-build-isolation", ".[webapp]"])
    run([str(python), "-m", "pip", "check"])
    return python


def node_version(node: Path) -> tuple[int, int, int]:
    result = subprocess.run(
        [str(node), "--version"], capture_output=True, text=True, check=True
    )
    parts = result.stdout.strip().removeprefix("v").split(".")
    if len(parts) < 3 or not all(part.isdigit() for part in parts[:3]):
        raise RuntimeError("The selected Node binary did not report a semantic version")
    return tuple(int(part) for part in parts[:3])


def select_node() -> Path:
    configured = str(os.environ.get("EASYICU_DESKTOP_NODE") or "").strip()
    raw = configured or shutil.which("node") or ""
    if not raw:
        raise RuntimeError("Node 22.19+ is required to build the Copilot runtime")
    node = Path(raw).expanduser().resolve()
    if node_version(node) < MIN_NODE:
        raise RuntimeError("Node 22.19+ is required to build the Copilot runtime")
    return node


def node_license(node: Path) -> Path:
    for parent in [node.parent, *node.parents]:
        candidate = parent / "LICENSE"
        if candidate.is_file():
            return candidate
    raise RuntimeError("The selected Node distribution has no LICENSE file")


def windows_npm(node: Path) -> list[str]:
    # Calling npm.cmd through cmd.exe breaks quoting for some user paths.
    # Use the CLI JavaScript from the very distribution we bundle instead.
    cli = node.parent / "node_modules" / "npm" / "bin" / "npm-cli.js"
    if not cli.is_file():
        raise RuntimeError(f"The selected Node distribution has no npm CLI: {cli}")
    return [str(node), str(cli)]


def installed_package(python: Path, build_root: Path) -> Path:
    result = subprocess.run(
        [str(python), "-I", "-X", "utf8", "-c",
         "from importlib.util import find_spec; from pathlib import Path; "
         "print(Path(next(iter(find_spec('easyicu').submodule_search_locations))))"],
        cwd=build_root, capture_output=True, encoding="utf-8", check=True,
    )
    return Path(result.stdout.strip())


def prepare_node_runtime(node, python, *, build_root, desktop_root, run, windows=False):
    npm = windows_npm(node) if windows else ["npm"]
    run([*npm, "ci", "--ignore-scripts", "--omit=dev"],
        cwd=installed_package(python, build_root) / "webserver/pi_copilot/node_app")
    resources = desktop_root / "src-tauri/resources"
    resources.mkdir(parents=True, exist_ok=True)
    target = resources / ("node.exe" if windows else "node")
    shutil.copy2(node, target)
    if not windows:
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    shutil.copy2(node_license(node), resources / "NODE_LICENSE")


def copy_python_sources(installed: Path, frozen: Path) -> None:
    """Preserve the wheel's bytes at the paths used by frozen __file__ values.

    PyInstaller --collect-data excludes Python sources. Kernel identity walks
    those sources (including transitive imports), so bytecode alone is not enough.
    """
    sources = sorted(installed.rglob("*.py"))
    if not sources or not (installed / "__init__.py").is_file():
        raise RuntimeError("The installed EasyICU wheel has no Python sources")
    for source in sources:
        target = frozen / source.relative_to(installed)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def build_backend(python, *, build_root, desktop_root, run, windows=False):
    dist = build_root / "pyinstaller-dist"
    command = [
        str(python), "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
        "--contents-directory", "_internal", "--name", "easyicu-backend",
        "--collect-submodules", "easyicu", "--collect-data", "easyicu",
        "--copy-metadata", "easyicu",
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.protocols.websockets.auto",
        "--hidden-import", "uvicorn.lifespan.on",
        "--distpath", str(dist), "--workpath", str(build_root / "pyinstaller-work"),
        "--specpath", str(build_root / "pyinstaller-spec"),
    ]
    if windows:
        command += ["--manifest", str(desktop_root / "windows-backend.manifest")]
    command.append(str(desktop_root / "backend_entry.py"))
    run(command)
    backend_root = dist / "easyicu-backend"
    name = "easyicu-backend.exe" if windows else "easyicu-backend"
    if not (backend_root / name).is_file():
        raise RuntimeError(f"PyInstaller did not create {backend_root / name}")
    packaged_root = desktop_root / "src-tauri/resources/backend"
    if packaged_root.exists():
        shutil.rmtree(packaged_root)
    shutil.copytree(backend_root, packaged_root, symlinks=not windows)
    copy_python_sources(installed_package(python, build_root), packaged_root / "_internal/easyicu")
    packaged = packaged_root / name
    if not windows:
        packaged.chmod(packaged.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return packaged


def verify_kernel_identity(python: Path, build_root: Path, packaged: Path) -> dict:
    result = subprocess.run(
        [str(python), "-I", "-c",
         "import json, sys; from pathlib import Path; import easyicu; "
         "from easyicu.research_agent.execution.kernel_identity import build_execution_kernel_identity as identity; "
         "installed = identity(Path(easyicu.__file__).parent); "
         "frozen = identity(Path(sys.argv[1])); "
         "print(json.dumps({'installed': installed.to_dict(), 'frozen': frozen.to_dict()}))",
         str(packaged.parent / "_internal/easyicu")],
        cwd=build_root, capture_output=True, text=True, check=True,
    )
    identities = json.loads(result.stdout)
    if identities["installed"] != identities["frozen"]:
        raise RuntimeError("Frozen execution kernel identity differs from the installed wheel")
    return identities["frozen"]


def build_inputs(*, repo_root, desktop_root, python, node, lock) -> dict:
    return {
        "schema_version": "easyicu.desktop_build_inputs/1",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip(),
        "source_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root, text=True,
        ).strip()),
        "python_version": subprocess.check_output([str(python), "--version"], text=True).strip(),
        "node_version": ".".join(map(str, node_version(node))),
        "lock_sha256": {
            str(path.relative_to(repo_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                lock, desktop_root / "package-lock.json", desktop_root / "src-tauri/Cargo.lock",
                repo_root / "src/easyicu/webserver/pi_copilot/node_app/package-lock.json",
            )
        },
    }
