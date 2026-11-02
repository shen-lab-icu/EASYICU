"""Exercise the packaged parser owner without any model or patient data."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from easyicu.webserver.pi_copilot import install
from tests.support.node import run_node


NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node is needed to verify runtime imports")


def test_private_install_binds_shared_parser_and_owned_modules(tmp_path, monkeypatch):
    source = install.packaged_app_dir()
    expected = install.runtime_manifest(source)
    real_run = subprocess.run

    def offline_install(command, **kwargs):
        if command[0] != "synthetic-npm":
            return real_run(command, **kwargs)
        root = Path(kwargs["cwd"])
        for relative, version in expected["pi_packages"].items():
            package = root / relative / "package.json"
            package.parent.mkdir(parents=True, exist_ok=True)
            package.write_text(json.dumps({"version": version}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(install.subprocess, "run", offline_install)
    target = install.install_runtime(destination=tmp_path / "private", source=source,
                                     npm_binary="synthetic-npm", node_binary=NODE)
    assert install.runtime_is_installed(target, source=source)
    shared = target / install.RUNTIME_SHARED_FILE
    assert shared.read_bytes() == (source.parent.parent / "static/js/screens-guided-pi-research-draft.js").read_bytes()
    for relative in (install.RUNTIME_SHARED_FILE, "src/draft-repair.mjs", "src/attached-skills.mjs"):
        assert relative in expected["files"]
        path = target / relative
        original = path.read_bytes()
        path.write_bytes(original + b"\n/* changed */\n")
        assert not install.runtime_is_installed(target, source=source)
        path.write_bytes(original)
    assert install.runtime_is_installed(target, source=source)
    result = run_node(NODE, "await import('./src/draft-repair.mjs')", module=True, cwd=target)
    assert result.returncode == 0, result.stderr
    shared.unlink()
    assert not install.runtime_is_installed(target, source=source)
    result = run_node(NODE, "await import('./src/draft-repair.mjs')", module=True, cwd=target)
    assert result.returncode != 0 and "MODULE_NOT_FOUND" in result.stderr


def test_staged_layout_uses_internal_cjs_and_never_falls_back(tmp_path):
    source = install.packaged_app_dir()
    stage = tmp_path / "p"
    (stage / "src").mkdir(parents=True)
    (stage / "gateway").mkdir()
    (stage / "gateway/package.json").write_text("{}", encoding="utf-8")
    (stage / "shared").mkdir()
    shutil.copy2(source / "src/draft-repair.mjs", stage / "src/draft-repair.mjs")
    shutil.copy2(source.parent.parent / "static/js/screens-guided-pi-research-draft.js",
                 stage / install.RUNTIME_SHARED_FILE)
    program = "await import('./src/draft-repair.mjs')"
    result = run_node(NODE, program, module=True, cwd=stage)
    assert result.returncode == 0, result.stderr
    (stage / install.RUNTIME_SHARED_FILE).unlink()
    result = run_node(NODE, program, module=True, cwd=stage)
    assert result.returncode != 0 and "MODULE_NOT_FOUND" in result.stderr


def test_development_manifest_and_node_both_prefer_internal_parser(tmp_path):
    source = install.packaged_app_dir()
    development = tmp_path / "webserver/pi_copilot/node_app"
    (development / "src").mkdir(parents=True)
    (development / "shared").mkdir()
    static = development.parent.parent / "static/js/screens-guided-pi-research-draft.js"
    static.parent.mkdir(parents=True)
    static.write_text("throw new Error('external parser must not load');", encoding="utf-8")
    shutil.copy2(source / "src/draft-repair.mjs", development / "src/draft-repair.mjs")
    internal = development / install.RUNTIME_SHARED_FILE
    internal.write_text("module.exports = {};", encoding="utf-8")
    assert install._packaged_source_file(development, install.RUNTIME_SHARED_FILE) == internal
    program = "await import('./src/draft-repair.mjs')"
    result = run_node(NODE, program, module=True, cwd=development)
    assert result.returncode == 0, result.stderr
    internal.unlink()
    assert install._packaged_source_file(development, install.RUNTIME_SHARED_FILE) == static
    result = run_node(NODE, program, module=True, cwd=development)
    assert result.returncode != 0 and "external parser must not load" in result.stderr


@pytest.mark.parametrize("obstacle", ["directory", "dangling_link", "shared_directory_link"])
def test_nonregular_shared_path_fails_closed_in_manifest_and_node(tmp_path, obstacle):
    source = install.packaged_app_dir()
    development = tmp_path / "webserver/pi_copilot/node_app"
    (development / "src").mkdir(parents=True)
    static = development.parent.parent / "static/js/screens-guided-pi-research-draft.js"
    static.parent.mkdir(parents=True)
    static.write_text("module.exports = {};", encoding="utf-8")
    shutil.copy2(source / "src/draft-repair.mjs", development / "src/draft-repair.mjs")
    shared = development / "shared"
    if obstacle == "shared_directory_link":
        other = tmp_path / "other"
        other.mkdir()
        shared.symlink_to(other, target_is_directory=True)
    else:
        shared.mkdir()
        parser = development / install.RUNTIME_SHARED_FILE
        if obstacle == "directory":
            parser.mkdir()
            (parser / "index.js").write_text("module.exports = {};", encoding="utf-8")
        else:
            parser.symlink_to(tmp_path / "absent")
    with pytest.raises(RuntimeError, match="shared"):
        install._packaged_source_file(development, install.RUNTIME_SHARED_FILE)
    result = run_node(NODE, "await import('./src/draft-repair.mjs')", module=True, cwd=development)
    assert result.returncode != 0 and "shared" in result.stderr
