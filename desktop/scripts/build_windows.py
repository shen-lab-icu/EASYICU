#!/usr/bin/env python3
"""Build an installed, per-user Windows x64 desktop runtime and NSIS installer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_common as common  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DESKTOP_ROOT = REPO_ROOT / "desktop"
BUILD_ROOT = Path(
    os.environ.get("EASYICU_DESKTOP_BUILD_DIR") or DESKTOP_ROOT / ".build/windows"
).expanduser().resolve()
VENV_ROOT = BUILD_ROOT / "venv"
PYTHON_LOCK = DESKTOP_ROOT / "requirements-windows-x64-py311.lock"
TARGET = "x86_64-pc-windows-msvc"
LOCK_COMMAND = (
    "uv pip compile pyproject.toml desktop/build-requirements.in "
    "--extra webapp --python-version 3.11 --python-platform x86_64-pc-windows-msvc "
    "--generate-hashes --no-emit-package easyicu "
    "--output-file desktop/requirements-windows-x64-py311.lock"
)


def _run(command: list[str], *, cwd: Path = REPO_ROOT, capture_output: bool = False) -> None:
    # A caller's source PYTHONPATH must not override the installed wheel.
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    subprocess.run(command, cwd=cwd, env=environment, check=True, capture_output=capture_output)


def _require_lock() -> None:
    content = PYTHON_LOCK.read_text(encoding="utf-8") if PYTHON_LOCK.is_file() else ""
    if ("EASYICU_WINDOWS_LOCK_PENDING" in content
            or not re.search(r"^\S+==\S+", content, re.MULTILINE)
            or not re.search(r"--hash=sha256:[0-9a-f]{64}\b", content)):
        raise RuntimeError(
            f"Windows dependency lock is missing or still a placeholder: {PYTHON_LOCK}\n"
            f"Generate and review it from the repository root before building:\n{LOCK_COMMAND}"
        )


def _check_target(python: str, node: Path) -> None:
    python_target = subprocess.check_output(
        [python, "-I", "-c", "import sys, struct, platform; "
         "print(sys.platform == 'win32' and sys.version_info[:2] == (3, 11) "
         "and struct.calcsize('P') == 8 and platform.machine().lower() in ('amd64', 'x86_64'))"],
        text=True,
    ).strip()
    node_target = subprocess.check_output(
        [str(node), "-p", "process.platform + ':' + process.arch"], text=True
    ).strip()
    if python_target != "True" or node_target != "win32:x64":
        raise RuntimeError("Windows builds require Python 3.11 x64 and Node x64 on Windows")


def _prepare_python_runtime(selected_python: str) -> Path:
    return common.prepare_python_runtime(
        selected_python, build_root=BUILD_ROOT, venv_root=VENV_ROOT,
        python=VENV_ROOT / "Scripts/python.exe", lock=PYTHON_LOCK, run=_run,
    )


def _build_tauri(node: Path) -> Path:
    npm = common.windows_npm(node)
    _run([*npm, "ci"], cwd=DESKTOP_ROOT)
    # Generate Windows assets from the existing artwork only on the build host.
    # Keep macOS's committed icons untouched.
    _run([*npm, "run", "tauri", "--", "icon", "src-tauri/icons/icon.png",
          "--output", "src-tauri/icons/windows"], cwd=DESKTOP_ROOT)
    target_root = Path(
        os.environ.get("CARGO_TARGET_DIR") or DESKTOP_ROOT / "src-tauri/target"
    ).expanduser().resolve()
    bundle = target_root / TARGET / "release/bundle/nsis"
    # Do not let a stale installer masquerade as this build's output.
    for old in bundle.glob("*.exe"):
        old.unlink()
    _run([*npm, "run", "tauri", "--", "build", "--target", TARGET,
          "--bundles", "nsis", "--", "--locked"], cwd=DESKTOP_ROOT)
    installers = sorted(bundle.glob("*.exe"))
    if len(installers) != 1:
        raise RuntimeError(f"Expected exactly one NSIS installer in {bundle}, got {len(installers)}")
    return installers[0]


def _plan(backend_only: bool) -> dict:
    try:
        _require_lock()
        status = "present (pip will verify every hash)"
    except RuntimeError:
        status = "pending"
    return {
        "target": TARGET,
        "installer": "nsis/currentUser",
        "lock_status": status,
        "lock_command": LOCK_COMMAND,
        "build_root": str(BUILD_ROOT),
        "steps": [
            "Require the Windows Python 3.11 hash lock and Windows x64 toolchains",
            "Create isolated venv; pip install --require-hashes; install EasyICU wheel; pip check",
            "npm ci --ignore-scripts --omit=dev in the installed Pi package; bundle Node and license",
            "Freeze backend_entry.py as onedir; preserve installed .py sources; compare kernel identities",
            *common.bundle_plan(windows=True),
            *([] if backend_only else ["Build Tauri NSIS installer with currentUser scope and locked Cargo inputs"]),
            "Write build-inputs.json with versions, lock hashes, kernel identity, pruning counts/bytes, paths and installer digest",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without running tools or writing files.")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(_plan(args.backend_only), indent=2))
        return 0
    _require_lock()
    if sys.platform != "win32":
        raise RuntimeError("Windows artifacts must be built on Windows x64")
    selected_python = common.select_python("Windows x64")
    node = common.select_node()
    _check_target(selected_python, node)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    receipt = BUILD_ROOT / "build-inputs.json"
    receipt.unlink(missing_ok=True)
    os.environ["EASYICU_HOME"] = str(BUILD_ROOT / "build-state")
    python = _prepare_python_runtime(selected_python)
    _check_target(str(python), node)
    common.prepare_node_runtime(
        node, python, build_root=BUILD_ROOT, desktop_root=DESKTOP_ROOT, run=_run, windows=True,
    )
    backend = common.build_backend(
        python, build_root=BUILD_ROOT, desktop_root=DESKTOP_ROOT, run=_run, windows=True,
    )
    identity = common.verify_kernel_identity(python, BUILD_ROOT, backend)
    bundle_inputs = common.prepare_bundle(
        desktop_root=DESKTOP_ROOT, build_root=BUILD_ROOT, run=_run, windows=True,
    )
    inputs = common.build_inputs(
        repo_root=REPO_ROOT, desktop_root=DESKTOP_ROOT, python=python, node=node, lock=PYTHON_LOCK,
    )
    inputs.update(target=TARGET, execution_kernel_identity=identity)
    inputs.update(bundle_inputs)
    receipt.write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
    if not args.backend_only:
        installer = _build_tauri(node)
        inputs["installer"] = {"name": installer.name, "sha256": hashlib.sha256(installer.read_bytes()).hexdigest()}
        print(f"Per-user installer: {installer}")
    receipt.write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen backend: {backend}\nBuild receipt: {receipt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
