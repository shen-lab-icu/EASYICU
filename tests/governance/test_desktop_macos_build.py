"""macOS build isolation and artifact provenance contracts."""

import hashlib
import json
from pathlib import Path

import pytest

from tests.support.desktop_bundle import load_script


def test_tauri_build_uses_the_locked_cargo_inputs(monkeypatch):
    module = load_script("build_macos")
    calls = []

    def stop_before_build(command, **kwargs):
        calls.append(command)
        if command[:2] == ["npm", "run"]:
            raise RuntimeError("synthetic build boundary")

    monkeypatch.setattr(module, "_run", stop_before_build)
    with pytest.raises(RuntimeError, match="synthetic build boundary"):
        module._build_tauri()
    assert calls[-1][-2:] == ["--", "--locked"]


def test_macos_build_does_not_inherit_a_source_python_runtime(monkeypatch):
    module = load_script("build_macos")
    monkeypatch.setenv("PYTHONPATH", "checkout/src")
    monkeypatch.setenv("PYTHONHOME", "another-python")
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: calls.append(kw))
    module._run(["python", "-m", "pip", "check"])
    assert "PYTHONPATH" not in calls[0]["env"]
    assert "PYTHONHOME" not in calls[0]["env"]


def test_macos_receipt_binds_frozen_kernel_and_exact_installer(tmp_path, monkeypatch):
    module = load_script("build_macos")
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(module, "BUILD_ROOT", tmp_path)
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path))
    monkeypatch.setattr(module, "_select_python", lambda: "python")
    monkeypatch.setattr(module, "_select_node", lambda: Path("node"))
    monkeypatch.setattr(module, "_prepare_python_runtime", lambda *a: Path("python"))
    monkeypatch.setattr(module, "_prepare_node_runtime", lambda *a: None)
    monkeypatch.setattr(module, "_build_backend", lambda *a: Path("backend"))
    monkeypatch.setattr(module.common, "prepare_bundle", lambda **k: {})
    monkeypatch.setattr(module.common, "build_inputs", lambda **k: {})
    identity = {"identity_sha256": "synthetic"}
    monkeypatch.setattr(module.common, "verify_kernel_identity", lambda *a: identity)
    installer = tmp_path / "EasyICU.dmg"
    installer.write_bytes(b"synthetic installer")
    monkeypatch.setattr(module, "_build_tauri", lambda: installer)

    assert module.main([]) == 0
    receipt_path = tmp_path / "build-inputs.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["target"] == "aarch64-apple-darwin"
    assert receipt["execution_kernel_identity"] == identity
    assert receipt["installer"] == {
        "name": installer.name, "sha256": hashlib.sha256(installer.read_bytes()).hexdigest(),
    }

    def refuse(*args):
        raise RuntimeError("kernel mismatch")

    monkeypatch.setattr(module.common, "verify_kernel_identity", refuse)
    monkeypatch.setattr(module, "_build_tauri", lambda: pytest.fail("unverified app packaged"))
    with pytest.raises(RuntimeError, match="kernel mismatch"):
        module.main([])
    assert not receipt_path.exists(), "failed rebuild must not retain a prior success receipt"
