"""Synthetic dependency hoisting and lazy-provider import checks; no npm/network."""

import json
import shutil
import subprocess

import pytest

from tests.support.desktop_bundle import load_script, node_app, node_package, write_files
from tests.support.node import run_node


@pytest.mark.parametrize("directory", [
    "doc", "docs", "documentation", "test", "tests", "__tests__", "example", "examples",
])
@pytest.mark.parametrize("prefix", ["dist", "lib", "esm", "cjs", "build", "nested/runtime"])
def test_optional_directory_pruning_is_limited_to_package_root(tmp_path, directory, prefix):
    common = load_script("build_common")
    package = node_package(tmp_path / "node_modules/synthetic")
    write_files(package, {
        f"{directory}/guide.md": "Unused root documentation\n",
        f"{prefix}/{directory}/runtime.js": "export const value = 1;\n",
    })
    assert common.prune_node_modules(tmp_path) == {
        "files": 1, "bytes": len("Unused root documentation\n"),
    }
    assert not (package / directory).exists()
    assert (package / prefix / directory / "runtime.js").is_file()


def test_staged_pi_smoke_retains_transitively_required_dist_doc(tmp_path):
    common = load_script("build_common")
    source = node_app(tmp_path / "wheel/node_app")
    agent = node_package(source / "node_modules/@earendil-works/pi-coding-agent",
                         name="@earendil-works/pi-coding-agent", dependencies={"synthetic-yaml": "1.0.0"})
    yaml = node_package(agent / "node_modules/synthetic-yaml", name="synthetic-yaml",
                        type="commonjs", exports="./dist/index.js", main="./dist/index.js")
    write_files(agent, {"index.js": "import yaml from 'synthetic-yaml'; export const value = yaml.value;\n"})
    write_files(yaml, {
        "dist/index.js": "module.exports = require('./compose/composer.js');\n",
        "dist/compose/composer.js": "module.exports = require('../doc/directives.js');\n",
        "dist/doc/directives.js": "exports.value = 42;\n",
        "docs/guide.md": "Unused package documentation\n",
    })
    node = shutil.which("node")
    assert node
    common.check_staged_pi(source, node, run=run_command)
    desktop = tmp_path / "desktop"
    staged = desktop / "src-tauri/resources/backend/_internal/easyicu/webserver/pi_copilot/node_app"
    shutil.copytree(source, staged)
    write_files(staged.parent, {"tool_catalog.json": "{}"})
    shutil.copy2(node, desktop / "src-tauri/resources/node.exe")
    write_files(desktop / "src-tauri", {
        "tauri.conf.json": json.dumps({"bundle": {"resources": {"resources/backend": "resources/backend", "p": "p"}}}),
        "tauri.windows.conf.json": "{}",
    })
    result = common.prepare_bundle(desktop_root=desktop, build_root=tmp_path / "build", run=run_command, windows=True)
    packaged = desktop / "src-tauri/p/node_modules/synthetic-yaml"
    assert (packaged / "dist/doc/directives.js").read_bytes() == (yaml / "dist/doc/directives.js").read_bytes()
    assert not (packaged / "docs").exists()
    assert (yaml / "docs/guide.md").is_file()
    assert result["bundle_paths"]["over_budget"] == 0
    assert result["node_modules_pruned"]["files"] == 19


@pytest.mark.parametrize("suffix", ["ts", "mts", "cts"])
@pytest.mark.parametrize("field", ["main", "exports", "bin"])
def test_typescript_pruning_retains_runtime_entries_and_their_siblings(tmp_path, suffix, field):
    common = load_script("build_common")
    runtime = node_package(tmp_path / "node_modules/runtime", **{
        "exports": None, field: f"./src/entry.{suffix}",
    })
    unused = node_package(tmp_path / "node_modules/compiled")
    for package in (runtime, unused):
        write_files(package, {
            f"src/entry.{suffix}": "export const entry = 1;\n",
            f"src/helper.{suffix}": "export const helper = 2;\n",
            f"src/helper.d.{suffix}": "export declare const helper: number;\n",
            "src/entry.js.map": "{}\n",
        })
    common.prune_node_modules(tmp_path)
    assert (runtime / f"src/entry.{suffix}").is_file()
    assert (runtime / f"src/helper.{suffix}").is_file()
    assert not (unused / f"src/entry.{suffix}").exists()
    assert not (unused / f"src/helper.{suffix}").exists()
    assert (unused / "index.js").is_file()
    for package in (runtime, unused):
        assert not (package / f"src/helper.d.{suffix}").exists()
        assert not (package / "src/entry.js.map").exists()


def test_typescript_export_patterns_protect_only_the_owning_package(tmp_path):
    common = load_script("build_common")
    runtime = node_package(tmp_path / "node_modules/runtime", exports={
        "./*": {"types": "./types/*.d.ts", "import": "./src/*.ts"},
    })
    nested = node_package(runtime / "node_modules/compiled", exports={
        ".": {"types": "./src/unused.ts", "default": "./index.js"},
    })
    write_files(runtime, {"src/entry.ts": "export const value = 1;\n"})
    write_files(nested, {"src/unused.ts": "// unused source\n"})
    common.prune_node_modules(tmp_path)
    assert (runtime / "src/entry.ts").is_file()
    assert not (nested / "src/unused.ts").exists()


def test_hoisting_moves_unique_packages_and_deduplicates_identical_versions(tmp_path):
    common = load_script("build_common")
    source = tmp_path / "source"
    owner = node_package(source / "node_modules/owner", name="owner")
    node_package(source / "node_modules/shared", name="shared")
    node_package(owner / "node_modules/shared", name="shared")
    node_package(owner / "node_modules/@synthetic/sdk", name="@synthetic/sdk")
    staged = tmp_path / "staged"
    shutil.copytree(source, staged)
    assert common.hoist_node_modules(staged) == {"moved": 1, "deduplicated": 1}
    assert (staged / "node_modules/shared/index.js").is_file()
    assert (staged / "node_modules/@synthetic/sdk/index.js").is_file()
    assert not (staged / "node_modules/owner/node_modules/shared").exists()
    assert not (staged / "node_modules/owner/node_modules/@synthetic/sdk").exists()
    assert (source / "node_modules/owner/node_modules/@synthetic/sdk/index.js").is_file()
    assert common.hoist_node_modules(staged) == {"moved": 0, "deduplicated": 0}


@pytest.mark.parametrize("conflict", ["version", "bytes", "private_dependency"])
def test_hoisting_keeps_conflicting_copies_and_private_dependencies(tmp_path, conflict):
    common = load_script("build_common")
    node_package(tmp_path / "node_modules/shared", name="shared")
    owner = node_package(tmp_path / "node_modules/owner", name="owner")
    nested = node_package(owner / "node_modules/shared", name="shared",
                          version="2.0.0" if conflict == "version" else "1.0.0")
    if conflict == "bytes":
        write_files(nested, {"index.js": "export const value = 2;\n"})
    if conflict == "private_dependency":
        node_package(tmp_path / "node_modules/private", name="private", version="2.0.0")
        node_package(nested / "node_modules/private", name="private")
    common.hoist_node_modules(tmp_path)
    assert (nested / "index.js").is_file()
    if conflict == "private_dependency":
        assert (nested / "node_modules/private/index.js").is_file()


@pytest.mark.parametrize("field", ["dependencies", "optionalDependencies", "peerDependencies"])
def test_hoisting_retains_a_package_beside_its_incompatible_dependency(tmp_path, field):
    common = load_script("build_common")
    node_package(tmp_path / "node_modules/peer", name="peer", version="2.0.0")
    owner = node_package(tmp_path / "node_modules/owner", name="owner")
    nested = node_package(owner / "node_modules/adapter", name="adapter", **{field: {"peer": "1.0.0"}})
    node_package(owner / "node_modules/peer", name="peer")
    common.hoist_node_modules(tmp_path)
    assert (nested / "index.js").is_file()
    assert not (tmp_path / "node_modules/adapter").exists()


def test_hoisting_moves_a_package_whose_peer_is_already_available_at_the_root(tmp_path):
    common = load_script("build_common")
    node_package(tmp_path / "node_modules/peer", name="peer")
    owner = node_package(tmp_path / "node_modules/owner", name="owner", bin="index.js")
    nested = node_package(owner / "node_modules/adapter", name="adapter", peerDependencies={"peer": "1.0.0"})
    assert common.hoist_node_modules(tmp_path) == {"moved": 1, "deduplicated": 0}
    assert not nested.exists()
    assert (tmp_path / "node_modules/adapter/index.js").is_file()
    assert (owner / "index.js").is_file()


def test_hoisting_does_not_leap_over_an_intervening_version(tmp_path):
    common = load_script("build_common")
    owner = node_package(tmp_path / "node_modules/owner", name="owner")
    middle = node_package(owner / "node_modules/shared", name="shared", version="2.0.0")
    consumer = node_package(owner / "node_modules/consumer", name="consumer", bin="index.js")
    nearest = node_package(consumer / "node_modules/shared", name="shared")
    common.hoist_node_modules(tmp_path)
    assert (nearest / "index.js").is_file()
    assert not middle.exists()
    assert json.loads((tmp_path / "node_modules/shared/package.json").read_text())["version"] == "2.0.0"


def run_command(command, **kwargs):
    return subprocess.run(command, **kwargs, check=True, text=True, timeout=15)


@pytest.mark.parametrize("nested_ai", [False, True])
@pytest.mark.parametrize("adapter", ["anthropic", "google", "openai-completions", "openai-responses", "openai-codex-responses"])
def test_import_smoke_loads_lazy_providers_in_gateway_and_agent_contexts(tmp_path, nested_ai, adapter):
    common = load_script("build_common")
    app = node_app(tmp_path / "p")
    ai = app / "node_modules/@earendil-works/pi-ai"
    if nested_ai:
        private = app / "node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works/pi-ai"
        shutil.copytree(ai, private)
        ai = private
    node = shutil.which("node")
    assert node
    common.check_staged_pi(app, node, run=run_command)
    write_files(ai, {f"providers/{adapter}.js": "import 'synthetic-missing-peer';\n"})
    # The public entries still load: only importing the lazy adapter catches this.
    run_node(node, "await import('@earendil-works/pi-coding-agent'); await import('@earendil-works/pi-ai');",
             module=True, cwd=app, check=True, timeout=15)
    with pytest.raises(common.StagedPiImportError, match=f"desktop_pi_import_failed:.*{adapter}"):
        common.check_staged_pi(app, node, run=run_command)


@pytest.mark.parametrize("name", ["@earendil-works/pi-coding-agent", "@earendil-works/pi-ai", "typebox"])
def test_import_smoke_reports_public_entry_failures(tmp_path, name):
    common = load_script("build_common")
    app = node_app(tmp_path / "p")
    write_files(app / "node_modules" / name, {"index.js": "throw new Error('synthetic entry failure');\n"})
    node = shutil.which("node")
    assert node
    with pytest.raises(common.StagedPiImportError, match=f"desktop_pi_import_failed: {name}"):
        common.check_staged_pi(app, node, run=run_command)


@pytest.mark.parametrize("missing", [False, True])
def test_import_smoke_checks_gateway_adapters_even_with_a_healthy_private_ai(tmp_path, missing):
    common = load_script("build_common")
    app = node_app(tmp_path / "p")
    ai = app / "node_modules/@earendil-works/pi-ai"
    shutil.copytree(ai, app / "node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works/pi-ai")
    if missing:
        (ai / "providers/openai-completions.js").unlink()
    else:
        write_files(ai, {"providers/openai-completions.js": "throw new Error('synthetic root adapter failure');\n"})
    node = shutil.which("node")
    assert node
    with pytest.raises(common.StagedPiImportError, match="desktop_pi_import_failed:.*providers/openai-completions.js"):
        common.check_staged_pi(app, node, run=run_command)


def test_import_smoke_catches_a_peer_stranded_by_hoisting(tmp_path, monkeypatch):
    common = load_script("build_common")
    app = node_app(tmp_path / "p")
    owner = app / "node_modules/@earendil-works/pi-coding-agent"
    ai = owner / "node_modules/@earendil-works/pi-ai"
    shutil.copytree(app / "node_modules/@earendil-works/pi-ai", ai)
    adapter = node_package(owner / "node_modules/provider-sdk", name="provider-sdk", peerDependencies={"peer": "1.0.0"})
    node_package(owner / "node_modules/peer", name="peer")
    write_files(adapter, {"index.js": "import {value} from 'peer'; export {value};\n"})
    write_files(ai, {"providers/openai-completions.js": "export {value} from 'provider-sdk';\n"})
    node = shutil.which("node")
    assert node
    common.check_staged_pi(app, node, run=run_command)
    desktop = tmp_path / "desktop"
    staged = desktop / "src-tauri/resources/backend/_internal/easyicu/webserver/pi_copilot/node_app"
    staged.parent.mkdir(parents=True)
    shutil.move(str(app), staged)
    write_files(staged.parent, {"tool_catalog.json": "{}"})
    shutil.copy2(node, desktop / "src-tauri/resources/node.exe")
    write_files(desktop / "src-tauri", {
        "tauri.conf.json": json.dumps({"bundle": {"resources": {"p": "p"}}}),
        "tauri.windows.conf.json": "{}",
    })

    def broken_hoist(root):
        inaccessible = root / "gateway/node_modules/peer"
        inaccessible.parent.mkdir(parents=True)
        shutil.move(str(root / "node_modules/peer"), inaccessible)
        return {"moved": 1, "deduplicated": 0}

    monkeypatch.setattr(common, "hoist_node_modules", broken_hoist)
    with pytest.raises(common.StagedPiImportError, match="desktop_pi_import_failed:.*openai-completions"):
        common.prepare_bundle(desktop_root=desktop, build_root=tmp_path / "build", run=run_command, windows=True)
    assert (tmp_path / "build/longest-paths.json").is_file()


def test_import_smoke_uses_esm_import_conditions_for_private_pi_ai(tmp_path):
    common = load_script("build_common")
    app = node_app(tmp_path / "p")
    private = app / "node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works/pi-ai"
    shutil.copytree(app / "node_modules/@earendil-works/pi-ai", private)
    manifest = json.loads((private / "package.json").read_text())
    manifest["exports"] = {"import": "./index.js", "require": "./require.cjs"}
    write_files(private, {"package.json": json.dumps(manifest), "require.cjs": "throw new Error('wrong condition');\n"})
    node = shutil.which("node")
    assert node
    common.check_staged_pi(app, node, run=run_command)


def test_staging_deduplicates_generated_sdk_paths_below_the_unchanged_budget(tmp_path):
    common = load_script("build_common")
    source = node_app(tmp_path / "wheel/node_app")
    write_files(source.parent, {"tool_catalog.json": "{}"})
    sdk = node_package(source / "node_modules/@synthetic/provider-sdk", name="@synthetic/provider-sdk")
    prefix = "p/node_modules/@synthetic/provider-sdk/esm/models/operations/"
    relative = "esm/models/operations/" + "a" * (160 - len(prefix) - 3) + ".js"
    write_files(sdk, {relative: "export const value = 1;\n", relative.replace("esm/", "src/")[:-3] + ".ts": "// unused\n"})
    nested = source / "node_modules/@earendil-works/pi-coding-agent/node_modules/@synthetic/provider-sdk"
    shutil.copytree(sdk, nested)
    desktop = tmp_path / "desktop"
    staged = desktop / "src-tauri/resources/backend/_internal/easyicu/webserver/pi_copilot/node_app"
    shutil.copytree(source.parent, staged.parent)
    node = shutil.which("node")
    assert node
    bundled_node = desktop / "src-tauri/resources/node.exe"
    shutil.copy2(node, bundled_node)
    write_files(desktop / "src-tauri", {
        "tauri.conf.json": json.dumps({"bundle": {"resources": {"resources/backend": "resources/backend", "p": "p"}}}),
        "tauri.windows.conf.json": "{}",
    })
    result = common.prepare_bundle(desktop_root=desktop, build_root=tmp_path / "build", run=run_command, windows=True)
    assert result["node_modules_root_staged"]["promoted"] == 1
    assert result["node_modules_hoisted"] == {"moved": 0, "deduplicated": 0}
    assert result["node_modules_pruned"]["files"] == 20
    assert result["bundle_paths"]["max_length"] == 160
    assert (desktop / "src-tauri/p/node_modules/@synthetic/provider-sdk" / relative).is_file()
    assert not (desktop / "src-tauri/p" / nested.relative_to(source)).exists()
    assert (nested / relative).is_file()
    assert (sdk / (relative.replace("esm/", "src/")[:-3] + ".ts")).is_file()
    assert json.loads((tmp_path / "build/longest-paths.json").read_text()) == result["bundle_paths"]


@pytest.mark.parametrize("driver", ["build_windows", "build_macos"])
def test_build_drivers_retain_the_failing_import_specifier(driver, tmp_path):
    module = load_script(driver)
    app = node_app(tmp_path / "p")
    write_files(app, {"node_modules/@earendil-works/pi-ai/providers/google.js": "import 'synthetic-missing-peer';\n"})
    node = shutil.which("node")
    assert node
    with pytest.raises(module.common.StagedPiImportError, match="desktop_pi_import_failed:.*providers/google.js") as error:
        module.common.check_staged_pi(app, node, run=module._run)
    assert "synthetic-missing-peer" in str(error.value)


@pytest.mark.parametrize("driver", ["build_windows", "build_macos"])
def test_dry_run_explains_hoisting_typescript_and_lazy_provider_smoke(driver, capsys):
    module = load_script(driver)
    assert module.main(["--dry-run"]) == 0
    plan = capsys.readouterr().out.lower()
    assert "hoist" in plan and "version" in plan
    assert "at package roots" in plan
    assert "*.ts" in plan and "*.mts" in plan and "*.cts" in plan
    assert "provider" in plan and "openai-completions" in plan
