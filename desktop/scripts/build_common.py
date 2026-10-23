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
import warnings

MIN_NODE = (22, 19, 0)
# MAX_PATH (260) minus a 100-character per-user installation root.
BUNDLE_PATH_BUDGET = 160
_OPTIONAL_NODE_DIRS = frozenset({
    "doc", "docs", "documentation", "test", "tests", "__tests__", "example", "examples",
})
_NODE_BUILD_SUFFIXES = (".d.ts", ".d.mts", ".d.cts", ".map")


def _runtime_targets(value, *, skip_types=False):
    """Include every runtime condition/array alternative, excluding type exports."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for child in value:
            yield from _runtime_targets(child, skip_types=skip_types)
    elif isinstance(value, dict):
        for key, child in value.items():
            if not skip_types or (key != "types" and not key.startswith("types@")):
                yield from _runtime_targets(child, skip_types=skip_types)


def _node_package_root(path: Path) -> bool:
    return path.parent.name == "node_modules" or (
        path.parent.name.startswith("@") and path.parent.parent.name == "node_modules"
    )


def _referenced_node_dirs(manifests: list[Path]) -> set[Path]:
    protected = set()
    for manifest in manifests:
        package = json.loads(manifest.read_text(encoding="utf-8"))
        for field in ("main", "exports", "bin"):
            for target in _runtime_targets(package.get(field), skip_types=(field == "exports")):
                # Normalize local main/bin paths before checking containment;
                # never resolve entries outside the package while pruning.
                relative = Path(os.path.normpath(target))
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError(f"Cannot safely prune non-local Node entry: {manifest}: {target}")
                matches = list(manifest.parent.glob(str(relative))) if "*" in target else [manifest.parent / relative]
                for match in matches:
                    resolved = [match, *(Path(str(match) + suffix) for suffix in (".js", ".json", ".node"))]
                    for entry in resolved:
                        if not entry.exists():
                            continue
                        for ancestor in (entry, *entry.parents):
                            if ancestor == manifest.parent:
                                break
                            if ancestor.name.lower() in _OPTIONAL_NODE_DIRS:
                                protected.add(ancestor)
    return protected


def prune_node_modules(staged_root: Path) -> dict[str, int]:
    """Prune all staged dependency copies, including nested node_modules.

    Runtime-referenced docs/tests/examples directories stay intact: their entry
    may import siblings or load assets. Type declarations and maps never execute.
    The installed wheel and checkout are deliberately outside this owner.
    """
    paths = [path for path in staged_root.rglob("*")
             if "node_modules" in path.relative_to(staged_root).parts]
    protected = _referenced_node_dirs([
        path for path in paths if path.name == "package.json" and path.is_file()
        and _node_package_root(path.parent)
    ])
    removed_dirs = {
        path for path in paths if path.is_dir() and not path.is_symlink()
        # A dependency can itself be named "docs" or "test". Package roots
        # are executable owners, including packages using Node's default main.
        and not _node_package_root(path)
        and (path.name.lower() == "dist-types" or (
            path.name.lower() in _OPTIONAL_NODE_DIRS
            and not any(parent in protected for parent in (path, *path.parents))
        ))
    }
    counts = {"files": 0, "bytes": 0}
    for path in sorted(paths, key=lambda p: len(p.parts), reverse=True):
        in_removed_dir = any(parent in removed_dirs for parent in (path, *path.parents))
        if path.is_symlink() or path.is_file():
            if in_removed_dir or path.name.lower().endswith(_NODE_BUILD_SUFFIXES):
                counts["files"] += 1
                counts["bytes"] += path.lstat().st_size
                path.unlink()
        elif in_removed_dir:
            path.rmdir()
    return counts


class BundlePathBudgetError(RuntimeError):
    """The staged installer payload exceeds the Windows runtime path budget."""

    code = "desktop_bundle_path_budget_exceeded"

    def __init__(self, report: dict):
        self.report = report
        longest = "\n".join(f"  {item['length']}: {item['path']}" for item in report["longest_paths"])
        super().__init__(f"{self.code}: {report['over_budget']} paths exceed "
                         f"{report['budget']} characters (100 reserved for install root).\n{longest}")


def check_bundle_paths(desktop_root: Path, report_file: Path, *, windows=False) -> dict:
    """Measure Tauri resource destinations relative to the NSIS install root."""
    tauri = desktop_root / "src-tauri"
    config = json.loads((tauri / "tauri.conf.json").read_text(encoding="utf-8"))
    resources = config["bundle"]["resources"]
    if windows:
        override = json.loads((tauri / "tauri.windows.conf.json").read_text(encoding="utf-8"))
        resources = override.get("bundle", {}).get("resources", resources)
    mapping = resources if isinstance(resources, dict) else {path: path for path in resources}
    lengths = {}
    for source, destination in mapping.items():
        root = tauri / source
        if not root.exists():
            raise RuntimeError(f"Missing staged bundle resource: {root}")
        for path in [root, *root.rglob("*")] if root.is_dir() else [root]:
            relative = (Path(destination) / path.relative_to(root)).as_posix()
            # Windows counts UTF-16 code units, including surrogate pairs.
            lengths[relative] = len(relative.encode("utf-16-le")) // 2
    if windows:
        binary = override.get("mainBinaryName", config.get("mainBinaryName", "easyicu-desktop"))
        for filename in (f"{binary}.exe", "uninstall.exe"):
            lengths[filename] = len(filename.encode("utf-16-le")) // 2
    ordered = sorted(lengths.items(), key=lambda item: (-item[1], item[0]))
    report = {
        "schema_version": "easyicu.desktop_bundle_paths/1",
        "budget": BUNDLE_PATH_BUDGET,
        "max_length": max(lengths.values(), default=0),
        "over_budget": sum(length > BUNDLE_PATH_BUDGET for length in lengths.values()),
        "longest_paths": [{"length": length, "path": path} for path, length in ordered[:10]],
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if report["over_budget"]:
        error = BundlePathBudgetError(report)
        if windows:
            raise error
        warnings.warn(str(error), RuntimeWarning, stacklevel=2)
    return report


def bundle_plan(*, windows=False) -> list[str]:
    return [
        "Stage Pi at short install resource p/; prune staged node_modules only: "
        "*.d.ts/*.d.mts/*.d.cts, dist-types/, *.map and unreferenced docs/tests/examples",
        f"{'Fail' if windows else 'Warn'} on installed relative paths over {BUNDLE_PATH_BUDGET} "
        "characters; reserve 100 for install root; write longest-paths.json (10 longest)",
        "Check gateway syntax and import the staged Pi agent, Pi AI and typebox entries with bundled Node",
    ]


def prepare_bundle(*, desktop_root, build_root, run, windows=False) -> dict:
    resources = desktop_root / "src-tauri/resources"
    original = resources / "backend/_internal/easyicu/webserver/pi_copilot/node_app"
    short = desktop_root / "src-tauri/p"
    if short.exists():
        shutil.rmtree(short)
    shutil.move(str(original), short)
    shutil.copy2(original.parent / "tool_catalog.json", short / "tool_catalog.json")
    pruned = prune_node_modules(resources)
    short_pruned = prune_node_modules(short)
    pruned = {key: pruned[key] + short_pruned[key] for key in pruned}
    report = check_bundle_paths(desktop_root, build_root / "longest-paths.json", windows=windows)
    bundled_node = resources / ("node.exe" if windows else "node")
    run([str(bundled_node), "--check", "src/main.mjs"], cwd=short)
    run([str(bundled_node), "--input-type=module", "-e",
         "await import('@earendil-works/pi-coding-agent'); "
         "await import('@earendil-works/pi-ai'); await import('typebox'); process.exit(0);"], cwd=short)
    return {"node_modules_pruned": pruned, "bundle_paths": report}


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
