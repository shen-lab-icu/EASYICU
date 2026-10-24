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
import subprocess
import sys
import tempfile

# Only the build driver imports these helpers; frozen imports use the installed wheel.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_common as common  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DESKTOP_ROOT = REPO_ROOT / "desktop"
BUILD_ROOT = Path(
    os.environ.get("EASYICU_DESKTOP_BUILD_DIR") or DESKTOP_ROOT / ".build"
).expanduser().resolve()
VENV_ROOT = BUILD_ROOT / "venv"
PYTHON_LOCK = DESKTOP_ROOT / "requirements-macos-arm64-py311.lock"
MIN_NODE = (22, 19, 0)


def _run(command: list[str], *, cwd: Path = REPO_ROOT, capture_output: bool = False) -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    subprocess.run(command, cwd=cwd, check=True, capture_output=capture_output, env=env)


def _select_python() -> str:
    return common.select_python("macOS arm64")


def _venv_python() -> Path:
    return VENV_ROOT / "bin" / "python"


def _app_version() -> str:
    """Read the release version from the Tauri manifest so artifact names track it."""
    config = json.loads(
        (DESKTOP_ROOT / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8")
    )
    version = str(config.get("version") or "").strip()
    if not version:
        raise RuntimeError("tauri.conf.json does not declare a version")
    return version


def _prepare_python_runtime(selected_python: str) -> Path:
    return common.prepare_python_runtime(
        selected_python, build_root=BUILD_ROOT, venv_root=VENV_ROOT,
        python=_venv_python(), lock=PYTHON_LOCK, run=_run,
    )


def _node_version(node: Path) -> tuple[int, int, int]:
    return common.node_version(node)


def _select_node() -> Path:
    return common.select_node()


def _node_license(node: Path) -> Path:
    return common.node_license(node)


def _installed_node_app(python: Path) -> Path:
    return common.installed_package(python, BUILD_ROOT) / "webserver/pi_copilot/node_app"


def _prepare_node_runtime(node: Path, python: Path) -> None:
    common.prepare_node_runtime(
        node, python, build_root=BUILD_ROOT, desktop_root=DESKTOP_ROOT, run=_run,
    )


def _build_backend(python: Path) -> Path:
    return common.build_backend(
        python, build_root=BUILD_ROOT, desktop_root=DESKTOP_ROOT, run=_run,
    )


def _build_tauri() -> Path:
    _run(["npm", "ci"], cwd=DESKTOP_ROOT)
    _run(["npm", "run", "tauri", "--", "build", "--bundles", "app", "--", "--locked"], cwd=DESKTOP_ROOT)
    target_root = Path(
        os.environ.get("CARGO_TARGET_DIR") or DESKTOP_ROOT / "src-tauri" / "target"
    ).expanduser().resolve()
    bundle_root = target_root / "release" / "bundle"
    app = bundle_root / "macos" / "EasyICU.app"
    if not app.is_dir():
        raise RuntimeError(f"Tauri did not create {app}")

    identity = str(os.environ.get("APPLE_SIGNING_IDENTITY") or "-").strip() or "-"

    dmg_dir = bundle_root / "dmg"
    dmg_dir.mkdir(parents=True, exist_ok=True)
    dmg = dmg_dir / f"EasyICU_{_app_version()}_aarch64.dmg"
    dmg.unlink(missing_ok=True)
    # File Provider can restore FinderInfo on bundles inside Documents even
    # after xattr cleanup. Sign the DMG payload in the system temporary area.
    with tempfile.TemporaryDirectory(prefix="easyicu-dmg-") as raw:
        staging = Path(raw)
        staged_app = staging / "EasyICU.app"
        shutil.copytree(app, staged_app, symlinks=True)
        # Act on symlinks themselves so cleanup stays within this artifact.
        _run(["xattr", "-crs", str(staged_app)])
        _run(["codesign", "--force", "--deep", "--sign", identity, str(staged_app)])
        _run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(staged_app)])
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
    return dmg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="Prepare the frozen backend and bundled Node runtime without Tauri packaging.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without running tools or writing files.")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"steps": [
            "Install hash-locked wheel and Pi dependencies; bundle Node and freeze backend",
            *common.bundle_plan(),
            *([] if args.backend_only else ["Build Tauri app and signed DMG"]),
            "Write build-inputs.json with pruning counts/bytes and bundle paths",
        ]}, indent=2))
        return 0
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("This dependency lock targets macOS Apple Silicon only")
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    receipt = BUILD_ROOT / "build-inputs.json"
    receipt.unlink(missing_ok=True)
    os.environ["EASYICU_HOME"] = str(BUILD_ROOT / "build-state")
    node = _select_node()
    python = _prepare_python_runtime(_select_python())
    _prepare_node_runtime(node, python)
    backend = _build_backend(python)
    identity = common.verify_kernel_identity(python, BUILD_ROOT, backend)
    bundle_inputs = common.prepare_bundle(
        desktop_root=DESKTOP_ROOT, build_root=BUILD_ROOT, run=_run,
    )
    inputs = common.build_inputs(
        repo_root=REPO_ROOT, desktop_root=DESKTOP_ROOT, python=python, node=node,
        lock=PYTHON_LOCK,
    )
    inputs.update(bundle_inputs)
    inputs.update(target="aarch64-apple-darwin", execution_kernel_identity=identity)
    receipt.write_text(
        json.dumps(inputs, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Frozen backend: {backend}")
    if not args.backend_only:
        installer = _build_tauri()
        with installer.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        inputs["installer"] = {"name": installer.name, "sha256": digest}
        receipt.write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
        print(f"Desktop build complete: {installer}\nBuild receipt: {receipt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
