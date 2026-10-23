"""Offline root-package staging preserves Node resolution and runtime ownership."""

import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from tests.support.desktop_bundle import load_script, node_app, node_package, write_files


def root_fixture(tmp_path):
    source = node_app(tmp_path / "wheel/node_app")
    agent = source / "node_modules/@earendil-works/pi-coding-agent"
    ai = source / "node_modules/@earendil-works/pi-ai"
    shutil.copytree(ai, agent / "node_modules/@earendil-works/pi-ai")
    node_package(source / "node_modules/provider-sdk", name="provider-sdk", version="2.0.0")
    private = node_package(agent / "node_modules/provider-sdk", name="provider-sdk")
    write_files(ai, {"providers/openai-completions.js": "export {value} from 'provider-sdk';\n"})
    write_files(agent / "node_modules/@earendil-works/pi-ai", {
        "providers/openai-completions.js": "export {value} from 'provider-sdk';\n",
    })
    write_files(source, {
        "package.json": json.dumps({"type": "module", "dependencies": {
            "@earendil-works/pi-coding-agent": "1.0.0", "@earendil-works/pi-ai": "1.0.0", "typebox": "1.0.0",
        }}),
        "README.md": "Gateway documentation\n",
        "THIRD_PARTY_NOTICES.md": "Synthetic notices\n",
        "package-lock.json": json.dumps({"packages": {
            "node_modules/@earendil-works/pi-coding-agent": {"version": "1.0.0"},
            "node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works/pi-ai": {"version": "1.0.0"},
            "node_modules/@earendil-works/pi-ai": {"version": "1.0.0"},
        }}),
    })
    write_files(agent, {
        "README.md": "Agent documentation\n",
        "src/agent.mjs": "export const agent = true;\n",
    })
    return source, private


def test_root_staging_keeps_the_agent_manifest_dependencies_and_gateway(tmp_path):
    common = load_script("build_common")
    source, private = root_fixture(tmp_path)
    staged = tmp_path / "frozen/node_app"
    shutil.copytree(source, staged)
    short = tmp_path / "p"
    assert common.stage_pi_root(staged, short) == {"promoted": 1, "outer_packages_retained": 1}
    assert json.loads((short / "package.json").read_text())["name"] == "@earendil-works/pi-coding-agent"
    assert (short / "README.md").read_text() == "Agent documentation\n"
    assert (short / "gateway/README.md").read_text() == "Gateway documentation\n"
    assert (short / "gateway/package.json").read_bytes() == (source / "package.json").read_bytes()
    assert (short / "package-lock.json").read_bytes() == (source / "package-lock.json").read_bytes()
    assert (short / "src/main.mjs").is_file()
    assert (short / "src/agent.mjs").is_file()
    assert (short / "src/skills/web-prototype/SKILL.md").is_file()
    assert json.loads((short / "node_modules/provider-sdk/package.json").read_text())["version"] == "1.0.0"
    assert not (short / "node_modules/@earendil-works/pi-coding-agent").exists()
    assert not staged.exists()
    assert (private / "index.js").is_file()
    node = shutil.which("node")
    assert node
    common.check_staged_pi(short, node, run=lambda command, **kw: subprocess.run(
        command, check=True, text=True, timeout=15, **kw,
    ))


def test_root_staging_checks_gateway_dependency_versions_before_mutation(tmp_path):
    common = load_script("build_common")
    source, _ = root_fixture(tmp_path)
    private_ai = source / "node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works/pi-ai"
    manifest = json.loads((private_ai / "package.json").read_text())
    manifest["version"] = "2.0.0"
    write_files(private_ai, {"package.json": json.dumps(manifest)})
    with pytest.raises(RuntimeError, match="desktop_pi_root_stage_failed:.*@earendil-works/pi-ai"):
        common.stage_pi_root(source, tmp_path / "p")
    assert (source / "src/main.mjs").is_file()
    assert private_ai.is_dir()
    assert not (tmp_path / "p").exists()


@pytest.mark.parametrize("field", ["dependencies", "optionalDependencies", "peerDependencies"])
def test_root_staging_rejects_a_fallback_whose_dependency_would_change(tmp_path, field):
    common = load_script("build_common")
    source, _ = root_fixture(tmp_path)
    fallback = node_package(source / "node_modules/fallback", name="fallback", **{field: {"provider-sdk": "2.0.0"}})
    manifest = json.loads((source / "package.json").read_text())
    manifest["dependencies"]["fallback"] = "1.0.0"
    write_files(source, {"package.json": json.dumps(manifest)})
    with pytest.raises(RuntimeError, match="desktop_pi_root_stage_failed:.*provider-sdk"):
        common.stage_pi_root(source, tmp_path / "p")
    assert (fallback / "index.js").is_file()


@pytest.mark.parametrize("relative", ["src/main.mjs", "gateway/package.json"])
def test_root_staging_rejects_conflicting_gateway_source_before_mutation(tmp_path, relative):
    common = load_script("build_common")
    source, _ = root_fixture(tmp_path)
    write_files(source / "node_modules/@earendil-works/pi-coding-agent", {relative: "// different owner\n"})
    with pytest.raises(common.StagedPiRootError, match=f"gateway file collision: {relative}"):
        common.stage_pi_root(source, tmp_path / "p")
    assert (source / "src/main.mjs").read_text() == "export const gateway = true;\n"
    assert not (tmp_path / "p").exists()


def test_root_staging_rejects_an_agent_without_package_self_reference(tmp_path):
    common = load_script("build_common")
    source, _ = root_fixture(tmp_path)
    agent = source / "node_modules/@earendil-works/pi-coding-agent/package.json"
    manifest = json.loads(agent.read_text())
    del manifest["exports"]
    agent.write_text(json.dumps(manifest))
    with pytest.raises(common.StagedPiRootError, match="desktop_pi_root_stage_failed:.*pi-coding-agent"):
        common.stage_pi_root(source, tmp_path / "p")
    assert agent.is_file()
    assert not (tmp_path / "p").exists()


def test_root_staging_discards_unreachable_outer_packages_and_retains_required_fallbacks(tmp_path):
    common = load_script("build_common")
    source, _ = root_fixture(tmp_path)
    node_package(source / "node_modules/unused", name="unused", dependencies={"provider-sdk": "2.0.0"})
    node_package(source / "node_modules/typebox/node_modules/required-peer", name="required-peer")
    manifest = json.loads((source / "node_modules/typebox/package.json").read_text())
    manifest["peerDependencies"] = {"required-peer": "1.0.0"}
    write_files(source / "node_modules/typebox", {"package.json": json.dumps(manifest)})
    short = tmp_path / "p"
    common.stage_pi_root(source, short)
    assert not (short / "node_modules/unused").exists()
    assert (short / "node_modules/typebox/node_modules/required-peer/index.js").is_file()


@pytest.mark.parametrize("tamper", ["version", "name", "missing"])
def test_root_staged_runtime_validates_original_pins_and_metadata(tmp_path, tamper):
    from easyicu.webserver.pi_copilot import install

    common = load_script("build_common")
    source, _ = root_fixture(tmp_path)
    for relative in install.RUNTIME_SOURCE_FILES:
        if not (source / relative).is_file():
            write_files(source, {relative: "// synthetic gateway\n"})
    write_files(source.parent, {"tool_catalog.json": "{}"})
    expected = install.runtime_manifest(source)
    short = tmp_path / "p"
    common.stage_pi_root(source, short)
    write_files(short, {"tool_catalog.json": "{}"})
    assert install.runtime_manifest(short) == expected
    assert install.packaged_runtime_is_complete(short)
    private_ai = short / "node_modules/@earendil-works/pi-ai/package.json"
    manifest = json.loads(private_ai.read_text())
    if tamper == "missing":
        private_ai.unlink()
    else:
        manifest[tamper] = "2.0.0" if tamper == "version" else "different-package"
        private_ai.write_text(json.dumps(manifest))
    assert not install.packaged_runtime_is_complete(short)


def test_root_staging_shortens_conflicting_sdk_paths_with_the_budget_unchanged(tmp_path):
    common = load_script("build_common")
    source, private = root_fixture(tmp_path)
    prefix = "p/node_modules/@earendil-works/pi-coding-agent/node_modules/provider-sdk/dist-es/"
    relative = "dist-es/" + "a" * (171 - len(prefix) - 3) + ".js"
    write_files(private, {relative: "export const value = 1;\n"})
    desktop = tmp_path / "desktop"
    short = desktop / "src-tauri/p"
    common.stage_pi_root(source, short)
    write_files(desktop / "src-tauri", {
        "tauri.conf.json": json.dumps({"bundle": {"resources": {"p": "p"}}}),
        "tauri.windows.conf.json": "{}",
    })
    report = common.check_bundle_paths(desktop, tmp_path / "longest-paths.json", windows=True)
    assert report["budget"] == 160
    assert report["max_length"] == 171 - len("node_modules/@earendil-works/pi-coding-agent/")
    assert report["over_budget"] == 0
    assert (short / "node_modules/provider-sdk" / relative).is_file()


def test_root_staged_metadata_can_install_a_canonical_private_runtime(tmp_path, monkeypatch):
    from easyicu.webserver.pi_copilot import install

    common = load_script("build_common")
    source, _ = root_fixture(tmp_path)
    for relative in install.RUNTIME_SOURCE_FILES:
        if not (source / relative).is_file():
            write_files(source, {relative: "// synthetic gateway\n"})
    write_files(source.parent, {"tool_catalog.json": "{}"})
    # Both owners can ship a lockfile. Keep both without changing either.
    write_files(source / "node_modules/@earendil-works/pi-coding-agent", {"package-lock.json": "{}\n"})
    expected = install.runtime_manifest(source)
    short = tmp_path / "p"
    common.stage_pi_root(source, short)
    write_files(short, {"tool_catalog.json": "{}"})
    assert install.runtime_manifest(short) == expected

    def fake_run(command, *, cwd, **kwargs):
        if command[-1] == "--version":
            return SimpleNamespace(stdout="v22.19.0\n")
        for relative, version in expected["pi_packages"].items():
            node_package(cwd / relative, name=relative.split("node_modules/")[-1], version=version)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(install.subprocess, "run", fake_run)
    target = install.install_runtime(source=short, destination=tmp_path / "private", npm_binary="npm", node_binary="node", environ={})
    assert install.runtime_is_installed(target, source=short)
    assert (target / "package.json").read_bytes() == (short / "gateway/package.json").read_bytes()
    assert (target / "package-lock.json").read_bytes() == (short / "gateway/package-lock.json").read_bytes()


def test_gateway_reports_the_root_staged_dependency_through_the_resource_owner(tmp_path, monkeypatch):
    from easyicu.webserver.pi_copilot import gateway, install

    source, _ = root_fixture(tmp_path)
    for relative in install.RUNTIME_SOURCE_FILES:
        if not (source / relative).is_file():
            write_files(source, {relative: "// synthetic gateway\n"})
    short = tmp_path / "p"
    load_script("build_common").stage_pi_root(source, short)
    write_files(short, {"tool_catalog.json": "{}"})
    monkeypatch.setattr(gateway, "packaged_app_dir", lambda: short)
    client = gateway.PiGatewayClient(app_dir=short, session_dir=tmp_path / "sessions", environ={})
    monkeypatch.setattr(client, "_node_binary", lambda: "node")
    monkeypatch.setattr(client, "_node_version", lambda *args: (22, 19, 0))
    try:
        status = client.installation_status()
        assert status["dependency_installed"]
        assert status["entrypoint_available"]
        assert status["lockfile_present"]
        assert status["runtime_integrity_verified"]
    finally:
        client.close()


def test_locked_private_pi_paths_follow_safe_hoists_in_the_root_layout(tmp_path):
    from easyicu.webserver.pi_copilot import install

    short = tmp_path / "p"
    write_files(short, {"gateway/package.json": "{}"})
    ai = node_package(short / "node_modules/@earendil-works/pi-ai", name="@earendil-works/pi-ai")
    # A broken old copy's descendants are outside its original importer's
    # lookup chain and must never substitute for the missing package itself.
    node_package(short / "node_modules/consumer/node_modules/@earendil-works/pi-ai/node_modules/@earendil-works/pi-ai",
                 name="@earendil-works/pi-ai", version="2.0.0")
    locked = "node_modules/@earendil-works/pi-coding-agent/node_modules/consumer/node_modules/@earendil-works/pi-ai"
    assert install.pi_package_manifest(short, locked) == ai / "package.json"
    assert install.pi_package_manifest(short) == short / "package.json"
