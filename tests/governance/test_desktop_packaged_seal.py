"""The build writer and packaged reader agree on the final staged bytes."""
import json
import shutil
import subprocess

import pytest

from easyicu.webserver.pi_copilot import install
from tests.support.desktop_bundle import load_script, node_app, node_package, write_files


def _stage(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the staged import smoke")
    common = load_script("build_common")
    desktop = tmp_path / "desktop"
    source = desktop / "src-tauri/resources/backend/_internal/easyicu/webserver/pi_copilot/node_app"
    node_app(source)
    for relative in install.RUNTIME_SOURCE_FILES:
        if not (source / relative).is_file():
            write_files(source, {relative: "// synthetic gateway\n"})
    agent = source / "node_modules/@earendil-works/pi-coding-agent"
    manifest = json.loads((agent / "package.json").read_text())
    manifest["dependencies"] = {"provider-sdk": "1.0.0"}
    write_files(agent, {"package.json": json.dumps(manifest),
                        "dist/core/http-dispatcher.js": "export const ready = true;\n"})
    node_package(source / "node_modules/provider-sdk", name="provider-sdk")
    lock = {"packages": {f"node_modules/{name}": {"version": "1.0.0"} for name in (
        "@earendil-works/pi-coding-agent", "@earendil-works/pi-ai")}}
    write_files(source, {"package-lock.json": json.dumps(lock), "README.md": "synthetic\n",
                         "THIRD_PARTY_NOTICES.md": "synthetic\n"})
    write_files(source.parent, {"tool_catalog.json": "{}"})
    write_files(source.parent.parent, {"static/js/screens-guided-pi-research-draft.js": "module.exports = {};\n"})
    shutil.copy2(node, desktop / "src-tauri/resources/node.exe")
    write_files(desktop / "src-tauri", {"tauri.conf.json": json.dumps({"bundle": {"resources": {"p": "p"}}}),
                                        "tauri.windows.conf.json": "{}"})
    # The seal test does not test Node relocation.  A system Node may depend
    # on adjacent shared libraries; invoke that same original executable.
    def run_with_original_node(command, **kwargs):
        return subprocess.run([node, *command[1:]], check=True, timeout=20, **kwargs)

    common.prepare_bundle(desktop_root=desktop, build_root=tmp_path / "build", windows=True,
                          run=run_with_original_node)
    return desktop / "src-tauri/p"


def test_build_seal_detects_changes_additions_and_missing_files(tmp_path):
    root = _stage(tmp_path)
    assert install.packaged_runtime_is_complete(root)
    for relative in ("dist/core/http-dispatcher.js", "node_modules/provider-sdk/index.js"):
        path = root / relative
        data = path.read_bytes()
        path.write_bytes(data + b"\n// modified\n")
        assert not install.packaged_runtime_is_complete(root)
        path.unlink()
        assert not install.packaged_runtime_is_complete(root)
        path.write_bytes(data)
        assert install.packaged_runtime_is_complete(root)
    added = root / "node_modules/provider-sdk/added.wasm"
    added.write_bytes(b"added")
    assert not install.packaged_runtime_is_complete(root)
    added.unlink()
    assert install.packaged_runtime_is_complete(root)
    (root / install.PACKAGED_SEAL_FILE).unlink()
    assert not install.packaged_runtime_is_complete(root)


@pytest.mark.parametrize("target", ["missing", "outside"])
def test_seal_writer_rejects_dangling_or_escaping_links(tmp_path, target):
    common = load_script("build_common")
    root = tmp_path / "p"
    root.mkdir()
    external = tmp_path / "external"
    if target == "outside":
        external.write_text("not part of package")
    (root / "unsafe").symlink_to(external)
    with pytest.raises((OSError, RuntimeError)):
        common.seal_packaged_pi(root)
    assert not (root / install.PACKAGED_SEAL_FILE).exists()


def test_seal_writer_does_not_overwrite_an_existing_receipt(tmp_path):
    common = load_script("build_common")
    root = tmp_path / "p"
    root.mkdir()
    receipt = root / install.PACKAGED_SEAL_FILE
    receipt.write_text("existing", encoding="utf-8")
    with pytest.raises(RuntimeError, match="occupied"):
        common.seal_packaged_pi(root)
    assert receipt.read_text(encoding="utf-8") == "existing"
