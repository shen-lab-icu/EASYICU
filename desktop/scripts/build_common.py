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
_NODE_SOURCE_SUFFIXES = (".ts", ".mts", ".cts")
# Typed executors run generated scripts in a frozen child. Their import strings
# are not visible to PyInstaller's normal Python bytecode analysis. Importing
# these entry points also activates PyInstaller's numerical-library hooks.
_DESKTOP_ANALYSIS_IMPORTS = (
    "numpy", "pandas", "scipy.stats", "matplotlib.pyplot",
    # save_publication_figure chooses its export backend from the filename.
    # Agg supplies PNG/TIFF (and the PNG embedded in PPTX); vector exports
    # dynamically load separate modules even when pyplot uses Agg.
    "matplotlib.backends.backend_agg", "matplotlib.backends.backend_svg",
    "matplotlib.backends.backend_pdf", "PIL.TiffImagePlugin",
    "statsmodels.api", "statsmodels.formula.api",
    "sklearn", "pyarrow.parquet", "patsy", "psutil",
)


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


def _node_runtime_references(manifests: list[Path]) -> tuple[set[Path], set[Path]]:
    protected = set()
    typescript = set()
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
                        if entry.name.lower().endswith(_NODE_SOURCE_SUFFIXES):
                            typescript.add(manifest.parent)
                        for ancestor in (entry, *entry.parents):
                            if ancestor == manifest.parent:
                                break
                            if ancestor.name.lower() in _OPTIONAL_NODE_DIRS:
                                protected.add(ancestor)
    return protected, typescript


def prune_node_modules(staged_root: Path) -> dict[str, int]:
    """Prune all staged dependency copies, including nested node_modules.

    Only docs/tests/examples directly at package roots are optional. Nested
    names can be runtime code (for example dist/doc reached by require).
    Runtime-referenced directories stay intact: their entry may import siblings
    or load assets. Type declarations and maps never execute.
    Keep source siblings when a package actually exposes TypeScript at runtime.
    The installed wheel and checkout are deliberately outside this owner.
    """
    paths = [path for path in staged_root.rglob("*")
             if "node_modules" in path.relative_to(staged_root).parts]
    manifests = [
        path for path in paths if path.name == "package.json" and path.is_file()
        and _node_package_root(path.parent)
    ]
    packages = {path.parent for path in manifests}
    protected, typescript = _node_runtime_references(manifests)
    removed_dirs = {
        path for path in paths if path.is_dir() and not path.is_symlink()
        # A dependency can itself be named "docs" or "test". Package roots
        # are executable owners, including packages using Node's default main.
        and not _node_package_root(path)
        and (path.name.lower() == "dist-types" or (
            path.name.lower() in _OPTIONAL_NODE_DIRS
            and path.parent in packages
            and not any(parent in protected for parent in (path, *path.parents))
        ))
    }
    counts = {"files": 0, "bytes": 0}
    for path in sorted(paths, key=lambda p: len(p.parts), reverse=True):
        in_removed_dir = any(parent in removed_dirs for parent in (path, *path.parents))
        if path.is_symlink() or path.is_file():
            owner = next((parent for parent in path.parents if parent in packages), None)
            unused_source = path.name.lower().endswith(_NODE_SOURCE_SUFFIXES) and owner not in typescript
            if in_removed_dir or path.name.lower().endswith(_NODE_BUILD_SUFFIXES) or unused_source:
                counts["files"] += 1
                counts["bytes"] += path.lstat().st_size
                path.unlink()
        elif in_removed_dir:
            path.rmdir()
    return counts


def _node_package_digest(package: Path) -> str:
    """Compare duplicate payloads too: a matching version alone is insufficient."""
    digest = hashlib.sha256()
    for root, dirs, files in os.walk(package, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name != "node_modules")
        for name in sorted(files):
            path = Path(root) / name
            digest.update(path.relative_to(package).as_posix().encode("utf-8") + b"\0")
            if path.is_symlink():
                digest.update(os.readlink(path).encode("utf-8"))
            else:
                with path.open("rb") as handle:
                    digest.update(hashlib.file_digest(handle, "sha256").digest())
    return digest.hexdigest()


def hoist_node_modules(staged_root: Path) -> dict[str, int]:
    """Flatten only staged packages, without npm, lock edits or package renames.

    Existing top-level copies may replace byte-identical, same-version leaves.
    Private dependencies, version conflicts and bin shims stay in place. A move
    into an empty slot must preserve declared dependency and peer versions.
    Import smoke additionally exercises the resulting lazy-provider contexts.
    """
    packages = {
        path.parent: json.loads(path.read_text(encoding="utf-8"))
        for path in staged_root.rglob("package.json")
        if _node_package_root(path.parent) and not path.parent.is_symlink()
    }
    counts = {"moved": 0, "deduplicated": 0}

    def package_at(owner, name, *, skip=None):
        for parent in (owner, *owner.parents):
            if not parent.is_relative_to(staged_root):
                break
            candidate = parent / "node_modules" / name
            if candidate != skip and candidate in packages:
                return candidate
        return None

    def resolved(owner, name):
        package = packages.get(package_at(owner, name), {})
        return package.get("name"), package.get("version")

    changed = True
    while changed:
        changed = False
        for source in sorted(packages, key=lambda path: (-len(path.parts), path.as_posix())):
            manifest = packages[source]
            modules = source.parent.parent if source.parent.name.startswith("@") else source.parent
            target = staged_root / "node_modules" / source.relative_to(modules)
            if source == target or not manifest.get("version") or manifest.get("bin"):
                continue
            intervening = package_at(modules.parent, source.relative_to(modules), skip=source)
            if intervening is not None and intervening != target:
                continue
            duplicate = target in packages
            if duplicate:
                if ((manifest.get("name"), manifest["version"]) !=
                        (packages[target].get("name"), packages[target].get("version"))):
                    continue
                if any(path.is_relative_to(source / "node_modules") for path in packages):
                    continue
                if _node_package_digest(source) != _node_package_digest(target):
                    continue
            elif target.exists():
                continue
            dependencies = set(manifest.get("peerDependencies", {}))
            if not duplicate:
                dependencies |= set(manifest.get("dependencies", {})) | set(manifest.get("optionalDependencies", {}))
            if any(resolved(source, name) != resolved(target, name) for name in dependencies):
                continue
            if duplicate:
                shutil.rmtree(source)
                del packages[source]
                counts["deduplicated"] += 1
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), target)
                for old in [path for path in packages if path.is_relative_to(source)]:
                    packages[target / old.relative_to(source)] = packages.pop(old)
                counts["moved"] += 1
            changed = True
    return counts


class StagedPiRootError(RuntimeError):
    """Promoting Pi would change a declared runtime dependency version."""

    code = "desktop_pi_root_stage_failed"


def stage_pi_root(original: Path, staged_root: Path) -> dict[str, int]:
    """Make the agent the root package; retain its private dependency graph.

    Gateway imports use Node's package self-reference for the agent. Its other
    direct imports must retain their pinned versions. Fill empty dependency
    slots from the outer tree only after checking every selected package's
    dependencies and peers in the proposed layout. All checks precede moves.
    """
    packages = {
        path.parent: json.loads(path.read_text(encoding="utf-8"))
        for path in original.rglob("package.json") if _node_package_root(path.parent)
    }
    agent = original / "node_modules/@earendil-works/pi-coding-agent"
    if agent not in packages or staged_root.exists():
        raise StagedPiRootError(f"{StagedPiRootError.code}: missing agent or occupied staging root")
    outer = {}
    placements = {source: staged_root / source.relative_to(agent)
                  for source in packages if source.is_relative_to(agent)}

    def located(owner, name, tree, boundary):
        own = tree.get(owner, {})
        if own.get("name") == name and own.get("exports"):
            return owner
        for parent in (owner, *owner.parents):
            if not parent.is_relative_to(boundary):
                break
            candidate = parent / "node_modules" / name
            if candidate in tree:
                return candidate
        return None

    def identity(path, name, tree):
        manifest = tree.get(path)
        return (manifest.get("name", name), manifest.get("version")) if manifest is not None else None

    gateway = json.loads((original / "package.json").read_text(encoding="utf-8"))
    while True:
        proposed = {target: packages[source] for source, target in placements.items()}
        checks = [(source, target, packages[source]) for source, target in placements.items()]
        checks.append((original, staged_root, gateway))
        added = False
        for source, target, manifest in checks:
            names = set().union(*(manifest.get(field, {}) for field in (
                "dependencies", "optionalDependencies", "peerDependencies",
            )))
            for name in sorted(names):
                before_path = located(source, name, packages, original)
                after_path = located(target, name, proposed, staged_root)
                before = identity(before_path, name, packages)
                after = identity(after_path, name, proposed)
                if before == after:
                    continue
                if before_path is not None and after_path is None and not before_path.is_relative_to(agent):
                    parts = before_path.relative_to(original / "node_modules").parts
                    top = Path(*parts[:2 if parts[0].startswith("@") else 1])
                    outer[top] = original / "node_modules" / top
                    for path in packages:
                        if path.is_relative_to(outer[top]):
                            placements[path] = staged_root / path.relative_to(original)
                    added = True
                    break
                raise StagedPiRootError(
                    f"{StagedPiRootError.code}: {source.relative_to(original)} imports {name}: "
                    f"{before} would resolve to {after}"
                )
            if added:
                break
        if not added:
            break
    # Preserve third-party metadata and gateway source bytes independently.
    # Only top-level metadata collisions go under gateway/; no package scope
    # is introduced around src/main.mjs, which uses the agent self-reference.
    gateway_files = []
    for root, dirs, files in os.walk(original):
        if Path(root) == original:
            dirs[:] = [name for name in dirs if name != "node_modules"]
        for name in files:
            source = Path(root) / name
            relative = source.relative_to(original)
            existing = agent / relative
            if source.parent == original and existing.exists():
                relative = Path("gateway") / relative
                existing = agent / relative
            if existing.is_file() and existing.read_bytes() != source.read_bytes():
                raise StagedPiRootError(f"{StagedPiRootError.code}: gateway file collision: {relative}")
            gateway_files.append((source, relative))
    staged_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(agent), staged_root)
    for relative, source in sorted(outer.items()):
        target = staged_root / "node_modules" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), target)
    for source, relative in gateway_files:
        target = staged_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.rmtree(original)
    return {"promoted": 1, "outer_packages_retained": len(outer)}


class StagedPiImportError(RuntimeError):
    """The bundled Node cannot import a staged Pi entry or lazy adapter."""

    code = "desktop_pi_import_failed"


def check_staged_pi(staged_root: Path, node: Path, *, run) -> None:
    # Resolve from both actual importers using ESM conditions. The agent may
    # retain a private pi-ai version. In pi-ai 0.84.1, api/*.lazy.js imports
    # api/<transport>.js and lazyApi reads its stream/streamSimple exports
    # (also used by compat.js's BUILTIN_APIS registry). The gateway accepts
    # these five transports. Importing the public index cannot validate them.
    script = r"""
import { readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
let specifier = '@earendil-works/pi-coding-agent';
try {
    const gateway = pathToFileURL(resolve('src/main.mjs')).href;
    const agent = import.meta.resolve(specifier, gateway);
    await import(agent);
    specifier = 'typebox';
    await import(import.meta.resolve(specifier, gateway));
    const checked = new Set();
    for (const parent of [gateway, agent]) {
        specifier = '@earendil-works/pi-ai';
        const entry = import.meta.resolve(specifier, parent);
        if (checked.has(entry)) continue;
        checked.add(entry);
        await import(entry);
        for (const api of [
            'anthropic-messages', 'google-generative-ai', 'openai-completions',
            'openai-responses', 'openai-codex-responses',
        ]) {
            specifier = new URL(`./api/${api}.js`, entry).href;
            const implementation = await import(specifier);
            for (const name of ['stream', 'streamSimple']) {
                if (typeof implementation[name] !== 'function') throw new Error(`Missing callable ${name} export`);
            }
        }
        const providers = new URL('./providers/', entry);
        specifier = providers.href;
        const adapters = readdirSync(providers).filter(name => /\.(?:js|mjs|cjs)$/.test(name)).sort();
        for (const name of adapters) {
            specifier = new URL(name, providers).href;
            await import(specifier);
        }
    }
} catch (error) {
    console.error(`desktop_pi_import_failed: ${specifier}: ${error.stack || error}`);
    process.exit(1);
}
process.exit(0);
"""
    try:
        run([str(node), "--experimental-import-meta-resolve", "--input-type=module", "-e", script],
            cwd=staged_root, capture_output=True)
    except subprocess.CalledProcessError as error:
        detail = error.stderr or str(error)
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        if StagedPiImportError.code not in detail:
            detail = f"{StagedPiImportError.code}: {detail}"
        raise StagedPiImportError(detail.strip()) from error


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
        "Stage pi-coding-agent as the root package of short install resource p/; "
        "preserve declared dependency/peer versions and the gateway/skill resource owner; "
        "prune staged node_modules only: "
        "*.d.ts/*.d.mts/*.d.cts, dist-types/, *.map and unreferenced docs/tests/examples at package roots; "
        "prune *.ts/*.mts/*.cts when package runtime entries do not use TypeScript",
        "Hoist staged Pi dependencies; deduplicate only identical copies with matching versions; retain conflicts",
        f"{'Fail' if windows else 'Warn'} on installed relative paths over {BUNDLE_PATH_BUDGET} "
        "characters; reserve 100 for install root; write longest-paths.json (10 longest)",
        "Check gateway syntax and import the staged Pi agent, Pi AI, typebox and lazy provider adapters "
        "(including openai-completions) from gateway and agent contexts with bundled Node",
    ]


def prepare_bundle(*, desktop_root, build_root, run, windows=False) -> dict:
    resources = desktop_root / "src-tauri/resources"
    original = resources / "backend/_internal/easyicu/webserver/pi_copilot/node_app"
    short = desktop_root / "src-tauri/p"
    if short.exists():
        shutil.rmtree(short)
    pruned = prune_node_modules(resources)
    rooted = stage_pi_root(original, short)
    shutil.copy2(original.parent / "tool_catalog.json", short / "tool_catalog.json")
    hoisted = hoist_node_modules(short)
    report = check_bundle_paths(desktop_root, build_root / "longest-paths.json", windows=windows)
    bundled_node = resources / ("node.exe" if windows else "node")
    run([str(bundled_node), "--check", "src/main.mjs"], cwd=short)
    check_staged_pi(short, bundled_node, run=run)
    return {"node_modules_pruned": pruned, "node_modules_root_staged": rooted,
            "node_modules_hoisted": hoisted, "bundle_paths": report}


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
    if sys.platform == "darwin":
        check_macos_node_portability(node)
    return node


def check_macos_node_portability(node: Path) -> None:
    """The bundle copies one executable, so only macOS system libraries may link.

    Homebrew Node can run on the build machine while depending on libnode and
    other libraries outside the app. Reject it before an expensive freeze.
    """
    result = subprocess.run(
        ["/usr/bin/otool", "-L", str(node)], capture_output=True, text=True, check=True,
    )
    libraries = [line.strip().split(" (", 1)[0] for line in result.stdout.splitlines()[1:]
                 if line.strip()]
    external = [library for library in libraries
                if not library.startswith(("/usr/lib/", "/System/Library/"))]
    if not libraries or external:
        detail = ", ".join(external) or "no dynamic-library information"
        raise RuntimeError(
            f"desktop_node_not_portable: {node}: {detail}. "
            "Set EASYICU_DESKTOP_NODE to the node executable from an official standalone "
            "Node distribution (for example one installed by nvm); the app bundles only "
            "the executable and cannot use Homebrew libraries."
        )


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
        # HostRunner binds runtime versions into execution authority. Core
        # dependency metadata includes every baseline analysis package plus
        # transitive numerical dependencies; psutil belongs to the web extra.
        "--recursive-copy-metadata", "easyicu",
        "--copy-metadata", "psutil",
        # DBOS loads SQLite's SQLAlchemy dialect dynamically; its SDK imports
        # psycopg even for a local SQLite workflow database.
        "--copy-metadata", "dbos",
        "--collect-all", "psycopg_binary",
        "--hidden-import", "sqlalchemy.dialects.sqlite",
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.protocols.websockets.auto",
        "--hidden-import", "uvicorn.lifespan.on",
        "--distpath", str(dist), "--workpath", str(build_root / "pyinstaller-work"),
        "--specpath", str(build_root / "pyinstaller-spec"),
    ]
    for module in _DESKTOP_ANALYSIS_IMPORTS:
        command += ["--hidden-import", module]
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
