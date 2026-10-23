"""Offline pruning, runtime relocation and installed path budget contracts."""

import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from tests.support.desktop_bundle import DESKTOP, ROOT, load_script, node_app, write_files


@pytest.mark.parametrize("entry", [
    {"main": "docs/start"},
    {"main": "docs/nested/../start.js"},
    {"main": "docs"},
    {"exports": {".": {"import": "./docs/start.js", "require": "./docs/start.cjs"}}},
    {"exports": {"./feature/*": ["./docs/*.js", "./examples/*.mjs"]}},
    {"bin": "tests/cli.js"},
    {"bin": {"synthetic": "tests/cli.js"}},
    {"bin": {"types": "tests/cli.js"}},
])
def test_pruning_keeps_directories_referenced_by_runtime_entries(tmp_path, entry):
    common = load_script("build_common")
    package = tmp_path / "node_modules/synthetic"
    write_files(package, {
        "package.json": json.dumps(entry),
        "docs/start.js": "module.exports = 7;\n",
        "docs/start.cjs": "module.exports = 8;\n",
        "docs/index.js": "module.exports = 9;\n",
        "examples/feature.mjs": "export const value = 10;\n",
        "tests/cli.js": "// synthetic CLI\n",
        "dist-types/index.d.ts": "declare const value: number;\n",
        "docs/start.js.map": "{}\n",
    })
    common.prune_node_modules(tmp_path)
    if "bin" in entry:
        assert (package / "tests/cli.js").is_file()
        assert not (package / "docs").exists()
    else:
        assert (package / "docs/start.js").is_file()
        assert (package / "docs/start.cjs").is_file()
        assert (package / "docs/index.js").is_file()
    if "*" in json.dumps(entry):
        assert (package / "examples/feature.mjs").is_file()
    assert not (package / "dist-types").exists()
    assert not (package / "docs/start.js.map").exists()


def test_pruning_covers_nested_copies_counts_bytes_and_ignores_non_dependencies(tmp_path):
    common = load_script("build_common")
    source = node_app(tmp_path / "source/node_app")
    staged = tmp_path / "staged"
    shutil.copytree(source, staged / "pi")
    nested = staged / "pi/node_modules/@earendil-works/pi-coding-agent/node_modules/nested"
    write_files(nested, {
        "package.json": '{"main":"index.js","exports":{"types":"./index.d.mts","default":"./index.js"}}',
        "index.js": "module.exports = true;\n",
        "index.d.mts": "declare const value: boolean;\n",
        "index.d.cts": "declare const value: boolean;\n",
        "examples/a.js": "// example\n",
        "doc/readme.md": "Documentation\n",
        "__tests__/test.js": "// tests\n",
    })
    shutil.copytree(source, staged / "node-runtime/pi")
    write_files(staged, {"outside.d.ts": "keep", "docs/keep.md": "keep"})
    before = {p.relative_to(staged): p.stat().st_size for p in staged.rglob("*") if p.is_file()}
    result = common.prune_node_modules(staged)
    removed = [p for p in before if not (staged / p).exists()]
    assert result == {"files": len(removed), "bytes": sum(before[p] for p in removed)}
    assert result["files"] == 41
    assert (nested / "index.js").is_file()
    assert not (nested / "index.d.mts").exists()
    assert not (nested / "index.d.cts").exists()
    assert (staged / "outside.d.ts").is_file()
    assert (staged / "docs/keep.md").is_file()
    assert (source / "node_modules/typebox/index.d.ts").is_file()
    assert common.prune_node_modules(staged) == {"files": 0, "bytes": 0}


def test_dependency_package_names_are_not_mistaken_for_optional_directories(tmp_path):
    common = load_script("build_common")
    write_files(tmp_path, {
        "node_modules/docs/package.json": "{}",
        "node_modules/docs/index.js": "module.exports = true;\n",
        "node_modules/@synthetic/tests/package.json": "{}",
        "node_modules/@synthetic/tests/index.js": "module.exports = true;\n",
        "node_modules/@synthetic/tests/docs/unused.md": "Unused documentation\n",
    })
    assert common.prune_node_modules(tmp_path)["files"] == 1
    assert (tmp_path / "node_modules/docs/index.js").is_file()
    assert (tmp_path / "node_modules/@synthetic/tests/index.js").is_file()


def test_unused_fixture_manifests_are_pruned_without_parsing_or_protecting_them(tmp_path):
    common = load_script("build_common")
    package = tmp_path / "node_modules/synthetic"
    write_files(package, {
        "package.json": '{"main":"index.js"}',
        "index.js": "module.exports = true;\n",
        "docs/invalid/package.json": "{intentional invalid fixture",
        "tests/project/package.json": '{"main":"index.js"}',
        "tests/project/index.js": "// unused test fixture\n",
        "tests/project/node_modules/nested/package.json": '{"main":"index.js"}',
        "tests/project/node_modules/nested/index.js": "// unused nested fixture\n",
    })
    assert common.prune_node_modules(tmp_path)["files"] == 5
    assert (package / "index.js").is_file()
    assert not (package / "docs").exists()
    assert not (package / "tests").exists()


@pytest.mark.parametrize("absolute", [False, True])
def test_pruning_rejects_nonlocal_entries_before_deleting_files(tmp_path, absolute):
    common = load_script("build_common")
    target = str(tmp_path / "outside.js") if absolute else "../outside.js"
    write_files(tmp_path / "node_modules/synthetic", {
        "package.json": json.dumps({"main": target}),
        "docs/unused.md": "Documentation\n",
    })
    with pytest.raises(RuntimeError, match="non-local Node entry"):
        common.prune_node_modules(tmp_path)
    assert (tmp_path / "node_modules/synthetic/docs/unused.md").is_file()


@pytest.mark.parametrize("length", [160, 161])
@pytest.mark.parametrize("windows", [True, False])
def test_installed_path_budget_boundary_and_platform_policy(tmp_path, length, windows):
    common = load_script("build_common")
    desktop = tmp_path / "desktop"
    config = {"bundle": {"resources": {"resources/p/": "p/"}}}
    relative = "p/" + "a" * (length - 5) + ".js"
    write_files(desktop / "src-tauri", {
        "tauri.conf.json": json.dumps(config),
        "tauri.windows.conf.json": "{}",
        "resources/p/" + relative[2:]: "// synthetic\n",
    })
    report_file = tmp_path / "diagnostics/longest-paths.json"
    if length > 160 and windows:
        with pytest.raises(common.BundlePathBudgetError, match="desktop_bundle_path_budget_exceeded") as error:
            common.check_bundle_paths(desktop, report_file, windows=windows)
        assert relative in str(error.value)
    elif length > 160:
        with pytest.warns(RuntimeWarning, match="desktop_bundle_path_budget_exceeded"):
            common.check_bundle_paths(desktop, report_file, windows=windows)
    else:
        common.check_bundle_paths(desktop, report_file, windows=windows)
    report = json.loads(report_file.read_text())
    assert report["budget"] == 160
    assert report["max_length"] == length
    assert report["over_budget"] == int(length > 160)
    assert report["longest_paths"][0] == {"length": length, "path": relative}


def test_path_report_uses_resource_destinations_and_includes_ten_longest(tmp_path):
    common = load_script("build_common")
    desktop = tmp_path / "desktop"
    write_files(desktop / "src-tauri", {
        "tauri.conf.json": json.dumps({"bundle": {"resources": ["resources/mac"]}}),
        "tauri.windows.conf.json": json.dumps({"bundle": {"resources": {"resources/p/": "p/"}}}),
        **{"resources/p/" + "a" * (160 + i): "x" for i in range(12)},
    })
    report_file = tmp_path / "longest-paths.json"
    with pytest.raises(common.BundlePathBudgetError) as error:
        common.check_bundle_paths(desktop, report_file, windows=True)
    report = json.loads(report_file.read_text())
    assert len(report["longest_paths"]) == 10
    assert [p["length"] for p in report["longest_paths"]] == list(range(173, 163, -1))
    assert "resources/p" not in str(error.value)
    assert "a" * 171 in str(error.value)
    assert "a" * 162 in str(error.value)


def test_path_budget_counts_windows_utf16_units(tmp_path):
    common = load_script("build_common")
    desktop = tmp_path / "desktop"
    write_files(desktop / "src-tauri", {
        "tauri.conf.json": json.dumps({"bundle": {"resources": {"resources/p/": "p/"}}}),
        "tauri.windows.conf.json": "{}",
        "resources/p/" + "a" * 154 + "🩺.js": "x",
    })
    with pytest.raises(common.BundlePathBudgetError):
        common.check_bundle_paths(desktop, tmp_path / "longest-paths.json", windows=True)


def test_path_budget_includes_the_nsis_application_executable(tmp_path):
    common = load_script("build_common")
    desktop = tmp_path / "desktop"
    write_files(desktop / "src-tauri", {
        "tauri.conf.json": json.dumps({"bundle": {"resources": []}}),
        "tauri.windows.conf.json": json.dumps({"mainBinaryName": "a" * 157}),
    })
    with pytest.raises(common.BundlePathBudgetError):
        common.check_bundle_paths(desktop, tmp_path / "longest-paths.json", windows=True)


@pytest.mark.parametrize("broken_pi", [False, True])
def test_staging_shortens_pi_without_mutating_installed_wheel_and_checks_imports(tmp_path, broken_pi):
    common = load_script("build_common")
    source = node_app(tmp_path / "installed/easyicu/webserver/pi_copilot/node_app")
    if broken_pi:
        write_files(source, {"node_modules/@earendil-works/pi-coding-agent/index.js": "throw new Error('synthetic broken Pi');\n"})
    write_files(source.parent, {"tool_catalog.json": "{}"})
    desktop = tmp_path / "desktop"
    resources = desktop / "src-tauri/resources"
    staged = resources / "backend/_internal/easyicu/webserver/pi_copilot/node_app"
    shutil.copytree(source.parent, staged.parent)
    # This stable third-party name exceeds 160 with the old prefix even after pruning.
    runtime_file = ("node_modules/@earendil-works/pi-coding-agent/node_modules/"
                    "@aws-sdk/client-bedrock-runtime/dist-es/commands/"
                    "InvokeModelWithBidirectionalStreamCommand.js")
    write_files(staged, {runtime_file: "export const value = 1;\n"})
    write_files(desktop / "src-tauri", {
        "tauri.conf.json": json.dumps({"bundle": {"resources": {
            "resources/backend/": "resources/backend/", "p/": "p/",
        }}}),
        "tauri.windows.conf.json": "{}",
    })
    node = shutil.which("node")
    assert node, "These offline import tests require the installed Node executable"
    shutil.copy2(node, resources / "node.exe")
    calls = []

    def run(command, **kwargs):
        assert not (desktop / "src-tauri/p/node_modules/typebox/index.d.ts").exists()
        calls.append((command, kwargs))
        subprocess.run(command, **kwargs, check=True, timeout=15)

    if broken_pi:
        with pytest.raises(common.StagedPiImportError, match="desktop_pi_import_failed: @earendil-works/pi-coding-agent"):
            common.prepare_bundle(desktop_root=desktop, build_root=tmp_path / "build", run=run, windows=True)
        assert (tmp_path / "build/longest-paths.json").is_file()
        return
    result = common.prepare_bundle(desktop_root=desktop, build_root=tmp_path / "build", run=run, windows=True)
    assert not staged.exists()
    short = desktop / "src-tauri/p"
    assert (short / runtime_file).is_file()
    assert (short / "tool_catalog.json").read_text() == "{}"
    assert (short / "src/skills/web-prototype/SKILL.md").is_file()
    assert (source / "node_modules/typebox/index.d.ts").is_file()
    assert result["node_modules_pruned"]["files"] == 18
    assert result["node_modules_pruned"]["bytes"] > 0
    assert result["bundle_paths"]["max_length"] <= 160
    assert calls and all(call[1]["cwd"] == short for call in calls)
    assert all(call[0][0] == str(resources / "node.exe") for call in calls)
    assert any("@earendil-works/pi-coding-agent" in " ".join(call[0]) for call in calls)


@pytest.mark.parametrize("driver", ["build_windows", "build_macos"])
def test_both_dry_runs_explain_pruning_smoke_and_budget_without_build_tools(driver, tmp_path, monkeypatch, capsys):
    module = load_script(driver)
    monkeypatch.setattr(module, "BUILD_ROOT", tmp_path / "unused")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("process started"))
    assert module.main(["--dry-run"]) == 0
    plan = capsys.readouterr().out.lower()
    assert "prun" in plan and ".d.ts" in plan and ".map" in plan
    assert "160" in plan and "100" in plan
    assert "import" in plan
    assert ("warn" if driver == "build_macos" else "fail") in plan
    assert not list(tmp_path.iterdir())


def test_workflow_retains_path_report_when_build_fails():
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/desktop_windows_build.yml").read_text())
    assert (workflow.get("on") or workflow.get(True)) == {"workflow_dispatch": None}
    steps = workflow["jobs"]["installer"]["steps"]
    diagnostics = next(step for step in steps if step.get("with", {}).get("name") == "EasyICU-windows-smoke-diagnostics")
    assert diagnostics["if"] == "always()"
    assert "${{ env.EASYICU_DESKTOP_BUILD_DIR }}/longest-paths.json" in diagnostics["with"]["path"]


def test_tauri_resources_install_pi_at_the_short_root_on_both_platforms():
    for filename in ("tauri.conf.json", "tauri.windows.conf.json"):
        config = json.loads((DESKTOP / "src-tauri" / filename).read_text())
        assert "p" in config["bundle"]["resources"]


@pytest.mark.parametrize("frozen", [False, True])
def test_packaged_app_owner_resolves_the_short_desktop_resource_only_when_frozen(tmp_path, monkeypatch, frozen):
    import sys
    from easyicu.webserver.pi_copilot import install

    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "resources/backend/easyicu-backend.exe"))
    expected = tmp_path / "p" if frozen else Path(install.__file__).resolve().with_name("node_app")
    assert install.packaged_app_dir() == expected


@pytest.mark.parametrize("driver", ["build_windows", "build_macos"])
@pytest.mark.parametrize("length", [160, 161])
def test_drivers_gate_packaging_and_record_pruning_in_receipts(tmp_path, monkeypatch, driver, length):
    module = load_script(driver)
    windows = driver == "build_windows"
    desktop = tmp_path / "desktop"
    resources = desktop / "src-tauri/resources"
    monkeypatch.setattr(module, "BUILD_ROOT", tmp_path / "build")
    monkeypatch.setattr(module, "DESKTOP_ROOT", desktop)
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(module.sys, "platform", "win32" if windows else "darwin")
    if windows:
        monkeypatch.setattr(module, "_require_lock", lambda: None)
        monkeypatch.setattr(module, "_check_target", lambda *a: None)
    else:
        monkeypatch.setattr(module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(module.common, "select_node", lambda: Path("node"))
    monkeypatch.setattr(module.common, "select_python", lambda *a: "python")
    monkeypatch.setattr(module, "_prepare_python_runtime", lambda *a: Path("python"))
    monkeypatch.setattr(module.common, "prepare_node_runtime", lambda *a, **kw: None)
    monkeypatch.setattr(module.common, "verify_kernel_identity", lambda *a: {})
    monkeypatch.setattr(module.common, "build_inputs", lambda **kw: {"synthetic": True})

    def freeze(*args, **kwargs):
        staged = resources / "backend/_internal/easyicu/webserver/pi_copilot/node_app"
        node_app(staged)
        write_files(staged, {"a" * (length - 5) + ".js": "// synthetic\n"})
        write_files(staged.parent, {"tool_catalog.json": "{}"})
        return resources / "backend/easyicu-backend.exe"

    monkeypatch.setattr(module.common, "build_backend", freeze)
    commands = []
    monkeypatch.setattr(module, "_run", lambda command, **kw: commands.append(command))
    built = []
    installer = tmp_path / "synthetic.exe"
    installer.write_bytes(b"synthetic installer")

    def package(*args):
        assert (module.BUILD_ROOT / "longest-paths.json").is_file()
        receipt = json.loads((module.BUILD_ROOT / "build-inputs.json").read_text())
        assert receipt["node_modules_pruned"]["files"] == 18
        built.append(True)
        return installer

    monkeypatch.setattr(module, "_build_tauri", package)
    write_files(desktop / "src-tauri", {
        "tauri.conf.json": json.dumps({"bundle": {"resources": ["resources/backend", "p"]}}),
        "tauri.windows.conf.json": "{}",
    })
    if length == 161 and windows:
        with pytest.raises(module.common.BundlePathBudgetError):
            module.main([])
        assert not built
    else:
        if length == 161:
            with pytest.warns(RuntimeWarning, match="desktop_bundle_path_budget_exceeded"):
                assert module.main([]) == 0
        else:
            assert module.main([]) == 0
        receipt = json.loads((module.BUILD_ROOT / "build-inputs.json").read_text())
        assert receipt["node_modules_pruned"]["files"] == 18
        assert receipt["node_modules_pruned"]["bytes"] > 0
        assert receipt["bundle_paths"]["max_length"] == length
        assert built == [True]
        assert len(commands) == 2
    assert json.loads((module.BUILD_ROOT / "longest-paths.json").read_text())["max_length"] == length


def test_frozen_workspace_skill_uses_the_packaged_resource_owner(tmp_path, monkeypatch):
    from easyicu.webserver.pi_copilot import tools

    short = tmp_path / "p"
    write_files(short, {"src/skills/web-prototype/SKILL.md": "Synthetic relocated skill"})
    monkeypatch.setattr(tools, "packaged_app_dir", lambda: short)
    monkeypatch.setattr(tools, "_workspace_access", lambda *a, **kw: (None, None))
    context = SimpleNamespace(session=SimpleNamespace(binding=SimpleNamespace(model_dump=lambda **kw: {})))
    result = tools._load_skill(context, {"name": "web-prototype"})
    assert result["details"]["instructions"] == "Synthetic relocated skill"
