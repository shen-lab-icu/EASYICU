"""Offline contracts for the Windows desktop distribution owner."""

import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import get_args

import pytest

from tests.support.desktop_bundle import DESKTOP, ROOT, load_script


def test_windows_dry_run_is_available_without_windows_or_dependencies(tmp_path):
    result = subprocess.run(
        [sys.executable, str(DESKTOP / "scripts/build_windows.py"), "--dry-run"],
        cwd=tmp_path, capture_output=True, text=True, check=False, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["target"] == "x86_64-pc-windows-msvc"
    assert plan["installer"] == "nsis/currentUser"
    assert plan["lock_status"] in {"pending", "present (pip will verify every hash)"}
    assert "--generate-hashes" in plan["lock_command"]
    assert not list(tmp_path.iterdir())


def test_missing_windows_lock_fails_before_running_any_build_step(tmp_path, monkeypatch):
    module = load_script("build_windows")
    monkeypatch.setattr(module, "PYTHON_LOCK", tmp_path / "absent.lock")
    monkeypatch.setattr(module, "_run", lambda *a, **k: pytest.fail("build ran"))
    with pytest.raises(RuntimeError, match="uv pip compile"):
        module.main([])


def test_live_build_requires_windows_even_when_a_lock_exists(monkeypatch):
    module = load_script("build_windows")
    monkeypatch.setattr(module, "_require_lock", lambda: None)
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.common, "select_python", lambda *a: pytest.fail("build started"))
    with pytest.raises(RuntimeError, match="built on Windows"):
        module.main([])


@pytest.mark.parametrize("content", ["", "# EASYICU_WINDOWS_LOCK_PENDING\n", "example==1.0\n", "example==1.0 --hash=sha256:invalid\n"])
def test_unresolved_or_unhashed_windows_lock_is_refused(tmp_path, monkeypatch, content):
    module = load_script("build_windows")
    lock = tmp_path / "windows.lock"
    lock.write_text(content)
    monkeypatch.setattr(module, "PYTHON_LOCK", lock)
    with pytest.raises(RuntimeError, match="uv pip compile"):
        module._require_lock()
    lock.write_text("example==1.0 --hash=sha256:" + "a" * 64 + "\n")
    module._require_lock()


def test_dry_run_has_no_process_or_filesystem_side_effects(tmp_path, monkeypatch, capsys):
    module = load_script("build_windows")
    monkeypatch.setattr(module, "BUILD_ROOT", tmp_path / "unused")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("process started"))
    assert module.main(["--dry-run", "--backend-only"]) == 0
    assert "Build Tauri" not in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_windows_venv_installs_hashes_then_a_noneditable_wheel(tmp_path, monkeypatch):
    module = load_script("build_windows")
    root = tmp_path / "临床 build space"
    monkeypatch.setattr(module, "BUILD_ROOT", root)
    monkeypatch.setattr(module, "VENV_ROOT", root / "venv")
    calls = []
    monkeypatch.setattr(module, "_run", lambda command, **kw: calls.append(command))
    python = module._prepare_python_runtime("Python 3.11/python.exe")
    assert python == root / "venv/Scripts/python.exe"
    assert calls == [
        ["Python 3.11/python.exe", "-m", "venv", str(root / "venv")],
        [str(python), "-m", "pip", "install", "--require-hashes", "-r", str(module.PYTHON_LOCK)],
        [str(python), "-m", "pip", "install", "--no-deps", "--no-build-isolation", ".[webapp]"],
        [str(python), "-m", "pip", "check"],
    ]


@pytest.mark.parametrize("python_target,node_target", [("False", "win32:x64"), ("True", "win32:arm64"), ("True", "linux:x64")])
def test_windows_lock_cannot_be_used_with_another_runtime_architecture(monkeypatch, python_target, node_target):
    module = load_script("build_windows")
    replies = iter([python_target, node_target])
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: next(replies))
    with pytest.raises(RuntimeError, match="Python 3.11 x64 and Node x64"):
        module._check_target("python.exe", Path("node.exe"))


def test_build_commands_do_not_inherit_source_pythonpath(monkeypatch):
    module = load_script("build_windows")
    monkeypatch.setenv("PYTHONPATH", "checkout/src")
    monkeypatch.setenv("PYTHONHOME", "another-python")
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    command = ["Python 临床/python.exe", "-m", "pip", "check"]
    module._run(command)
    assert calls[0][0] == command
    assert "PYTHONPATH" not in calls[0][1]["env"]
    assert "PYTHONHOME" not in calls[0][1]["env"]
    assert calls[0][1]["check"] is True


def fake_node(root):
    node = root / "Node 临床 space/node.exe"
    cli = node.parent / "node_modules/npm/bin/npm-cli.js"
    cli.parent.mkdir(parents=True)
    cli.write_text("// synthetic npm\n")
    node.write_bytes(b"synthetic node")
    (node.parent / "LICENSE").write_text("synthetic license")
    return node, cli


def test_pi_is_installed_in_the_wheel_and_bundled_node_has_its_license(tmp_path, monkeypatch):
    common = load_script("build_common")
    node, cli = fake_node(tmp_path)
    installed = tmp_path / "venv/site-packages/easyicu"
    monkeypatch.setattr(common, "installed_package", lambda *a: installed)
    calls = []
    common.prepare_node_runtime(
        node, Path("python.exe"), build_root=tmp_path, desktop_root=tmp_path / "desktop",
        run=lambda command, **kw: calls.append((command, kw)), windows=True,
    )
    assert calls == [([str(node), str(cli), "ci", "--ignore-scripts", "--omit=dev"],
                      {"cwd": installed / "webserver/pi_copilot/node_app"})]
    resources = tmp_path / "desktop/src-tauri/resources"
    assert (resources / "node.exe").read_bytes() == node.read_bytes()
    assert (resources / "NODE_LICENSE").read_text() == "synthetic license"
    cli.unlink()
    with pytest.raises(RuntimeError, match="npm CLI"):
        common.windows_npm(node)


@pytest.mark.parametrize("windows", [False, True])
def test_frozen_backend_preserves_installed_sources_and_kernel_identity(tmp_path, monkeypatch, windows):
    from matplotlib.backend_bases import get_registered_canvas_class
    from easyicu.research_agent.contracts.method_packages import BASELINE_PACKAGES
    from easyicu.research_agent.execution.kernel_identity import build_execution_kernel_identity
    from easyicu.research_agent.figures.publication import ExportFormat

    common = load_script("build_common")
    installed = tmp_path / "installed wheel/easyicu"
    payloads = {
        "__init__.py": "",
        "research_agent/execution/runners/demo.py": "from easyicu.helper import VALUE\n",
        "helper.py": "VALUE = 7\n",
        "research_agent/runner_image/requirements.lock": "synthetic==1\n",
    }
    for relative, payload in payloads.items():
        path = installed / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
    monkeypatch.setattr(common, "installed_package", lambda *a: installed)
    commands = []
    name = "easyicu-backend.exe" if windows else "easyicu-backend"

    def freeze(command):
        commands.append(command)
        dist = tmp_path / "pyinstaller-dist/easyicu-backend"
        target_lock = dist / "_internal/easyicu/research_agent/runner_image/requirements.lock"
        target_lock.parent.mkdir(parents=True)
        target_lock.write_text(payloads["research_agent/runner_image/requirements.lock"])
        (dist / name).write_bytes(b"synthetic executable")

    backend = common.build_backend(
        Path("python"), build_root=tmp_path, desktop_root=tmp_path / "desktop", run=freeze, windows=windows,
    )
    assert backend.name == name
    assert backend.read_bytes() == b"synthetic executable"
    frozen = backend.parent / "_internal/easyicu"
    assert (frozen / "helper.py").read_bytes() == (installed / "helper.py").read_bytes()
    assert build_execution_kernel_identity(installed) == build_execution_kernel_identity(frozen)
    assert "--onedir" in commands[0]
    assert "--paths" not in commands[0]
    options = list(zip(commands[0], commands[0][1:]))
    assert ("--recursive-copy-metadata", "easyicu") in options
    assert ("--copy-metadata", "psutil") in options
    assert ("--copy-metadata", "dbos") in options
    assert ("--collect-all", "psycopg_binary") in options
    hidden = {value for flag, value in options if flag == "--hidden-import"}
    assert {"dbos", "sqlalchemy.dialects.sqlite", "sqlalchemy.dialects.sqlite.pysqlite"} <= hidden
    assert "pyinstaller==6.22.2" in (DESKTOP / "requirements-windows-x64-py311.lock").read_text()
    # Validate the execution owner's current baseline, so adding a required
    # package cannot leave the frozen child unable to run a typed script.
    assert set(BASELINE_PACKAGES) <= {name.split(".")[0] for name in hidden}
    assert {"patsy", "psutil", "matplotlib.backends.backend_agg", "statsmodels.formula.api"} <= hidden
    # Exporters resolve backends dynamically by format, independently of the
    # active Agg canvas. Cover the figure contract, including PPTX's PNG image.
    for export_format in get_args(ExportFormat):
        canvas = get_registered_canvas_class("png" if export_format == "pptx" else export_format)
        assert canvas is not None
        assert canvas.__module__ in hidden, f"Missing frozen figure backend for {export_format}"
    assert "PIL.TiffImagePlugin" in hidden
    assert ("--manifest" in commands[0]) == windows


def test_source_copy_refuses_a_bytecode_only_package(tmp_path):
    common = load_script("build_common")
    with pytest.raises(RuntimeError, match="no Python sources"):
        common.copy_python_sources(tmp_path / "empty", tmp_path / "frozen")


def test_kernel_receipt_is_issued_only_for_equal_installed_and_frozen_identity(monkeypatch):
    common = load_script("build_common")
    identities = {"installed": {"identity_sha256": "same"}, "frozen": {"identity_sha256": "same"}}
    monkeypatch.setattr(common.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(identities)))
    assert common.verify_kernel_identity(Path("python"), Path("build"), Path("backend")) == identities["installed"]
    identities["frozen"]["identity_sha256"] = "different"
    with pytest.raises(RuntimeError, match="differs"):
        common.verify_kernel_identity(Path("python"), Path("build"), Path("backend"))


def test_tauri_uses_locked_windows_target_and_rejects_stale_installers(tmp_path, monkeypatch):
    module = load_script("build_windows")
    node, cli = fake_node(tmp_path)
    target = tmp_path / "cargo target"
    monkeypatch.setenv("CARGO_TARGET_DIR", str(target))
    bundle = target / module.TARGET / "release/bundle/nsis"
    bundle.mkdir(parents=True)
    stale = bundle / "stale.exe"
    stale.write_bytes(b"old")
    calls = []
    monkeypatch.setattr(module, "_run", lambda command, **kw: calls.append(command))
    with pytest.raises(RuntimeError, match="exactly one"):
        module._build_tauri(node)
    assert not stale.exists()
    assert calls[0] == [str(node), str(cli), "ci"]
    assert calls[-1][-7:] == ["build", "--target", module.TARGET, "--bundles", "nsis", "--", "--locked"]


def test_windows_main_writes_installer_digest_and_kernel_receipt(tmp_path, monkeypatch):
    import hashlib

    module = load_script("build_windows")
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(module, "BUILD_ROOT", tmp_path)
    monkeypatch.setattr(module, "_require_lock", lambda: None)
    monkeypatch.setattr(module.common, "select_python", lambda *a: "python.exe")
    monkeypatch.setattr(module.common, "select_node", lambda: Path("node.exe"))
    monkeypatch.setattr(module, "_check_target", lambda *a: None)
    monkeypatch.setattr(module, "_prepare_python_runtime", lambda *a: Path("python.exe"))
    monkeypatch.setattr(module.common, "prepare_node_runtime", lambda *a, **k: None)
    monkeypatch.setattr(module.common, "build_backend", lambda *a, **k: Path("backend.exe"))
    bundle_inputs = {"node_modules_pruned": {"files": 7, "bytes": 31}, "bundle_paths": {"max_length": 160}}
    monkeypatch.setattr(module.common, "prepare_bundle", lambda **k: bundle_inputs)
    monkeypatch.setattr(module.common, "verify_kernel_identity", lambda *a: {"identity_sha256": "synthetic"})
    monkeypatch.setattr(module.common, "build_inputs", lambda **k: {"lock_sha256": {"synthetic.lock": "digest"}})
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path))
    installer = tmp_path / "installer.exe"
    installer.write_bytes(b"synthetic installer")
    monkeypatch.setattr(module, "_build_tauri", lambda *a: installer)
    assert module.main([]) == 0
    receipt = json.loads((tmp_path / "build-inputs.json").read_text())
    assert receipt["target"] == module.TARGET
    assert receipt["execution_kernel_identity"] == {"identity_sha256": "synthetic"}
    assert receipt["installer"] == {"name": installer.name, "sha256": hashlib.sha256(installer.read_bytes()).hexdigest()}
    assert receipt["lock_sha256"] == {"synthetic.lock": "digest"}
    assert all(receipt[key] == value for key, value in bundle_inputs.items())
    monkeypatch.setattr(module, "_build_tauri", lambda *a: pytest.fail("backend-only built an installer"))
    assert module.main(["--backend-only"]) == 0
    backend_receipt = json.loads((tmp_path / "build-inputs.json").read_text())
    assert "installer" not in backend_receipt
    assert backend_receipt["execution_kernel_identity"] == receipt["execution_kernel_identity"]


def test_build_receipt_hashes_each_actual_lock(tmp_path, monkeypatch):
    import hashlib

    common = load_script("build_common")
    desktop = tmp_path / "desktop"
    paths = [
        desktop / "python.lock", desktop / "package-lock.json", desktop / "src-tauri/Cargo.lock",
        tmp_path / "src/easyicu/webserver/pi_copilot/node_app/package-lock.json",
    ]
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic lock {index}\n")
    monkeypatch.setattr(common, "node_version", lambda *a: (22, 19, 0))
    replies = iter(["a" * 40, " M synthetic.py", "Python 3.11.9"])
    monkeypatch.setattr(common.subprocess, "check_output", lambda *a, **k: next(replies))
    receipt = common.build_inputs(repo_root=tmp_path, desktop_root=desktop,
                                  python=Path("python"), node=Path("node"), lock=paths[0])
    assert receipt["source_commit"] == "a" * 40
    assert receipt["source_dirty"] is True
    assert receipt["python_version"] == "Python 3.11.9"
    assert receipt["node_version"] == "22.19.0"
    assert receipt["lock_sha256"] == {
        str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


def test_installed_package_probe_ignores_source_checkout_and_preserves_unicode(tmp_path, monkeypatch):
    common = load_script("build_common")
    expected = tmp_path / "临床 installed/easyicu"
    calls = []

    def probe(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=str(expected) + "\n")

    monkeypatch.setattr(common.subprocess, "run", probe)
    assert common.installed_package(Path("python.exe"), tmp_path) == expected
    assert calls[0][0][1:5] == ["-I", "-X", "utf8", "-c"]
    assert calls[0][1]["encoding"] == "utf-8"
    assert calls[0][1]["cwd"] == tmp_path


def test_windows_packaging_and_smoke_workflow_are_per_user():
    import xml.etree.ElementTree as ET
    import yaml

    config = json.loads((DESKTOP / "src-tauri/tauri.windows.conf.json").read_text())
    assert config["productName"] == "EasyICU Desktop"  # Different from the data directory.
    assert config["mainBinaryName"] == "easyicu-desktop"
    assert config["bundle"]["targets"] == ["nsis"]
    assert config["bundle"]["windows"]["nsis"]["installMode"] == "currentUser"
    assert config["bundle"]["windows"]["webviewInstallMode"]["type"] == "offlineInstaller"
    assert "resources/node.exe" in config["bundle"]["resources"]
    manifest = ET.fromstring((DESKTOP / "windows-backend.manifest").read_text())
    assert manifest.find(".//{*}requestedExecutionLevel").attrib["level"] == "asInvoker"
    assert manifest.find(".//{*}longPathAware").text == "true"
    controls = manifest.find(".//{*}dependentAssembly/{*}assemblyIdentity")
    assert controls is not None
    assert controls.attrib["name"] == "Microsoft.Windows.Common-Controls"
    assert controls.attrib["version"] == "6.0.0.0"
    build_rs = (DESKTOP / "src-tauri/build.rs").read_text()
    assert 'include_str!("../windows-backend.manifest")' in build_rs
    workflow = yaml.safe_load((ROOT / ".github/workflows/desktop_windows_build.yml").read_text())
    # PyYAML's YAML 1.1 resolver reads GitHub's 'on' as boolean True.
    triggers = workflow.get("on") or workflow.get(True)
    assert triggers["push"] == {"branches": ["codex/dev9-expansion-20260917"], "tags": ["v*"]}
    assert "workflow_dispatch" in triggers
    job = workflow["jobs"]["installer"]
    assert job["runs-on"] == "windows-latest"
    # GitHub refuses to parse a workflow whose job-level env reads a context
    # that exists only inside steps (runner, steps, job).
    job_env = json.dumps(job.get("env", {}))
    assert not re.search(r"\$\{\{[^}]*\b(runner|steps|job)\.", job_env)
    steps = job["steps"]
    assert any("smoke_windows.ps1" in step.get("run", "") for step in steps)
    artifact = next(step for step in steps if step.get("with", {}).get("name") == "EasyICU-windows-x64")
    assert "build-inputs.json" in artifact["with"]["path"]
    assert "nsis/*.exe" in artifact["with"]["path"]
    assert artifact.get("if", "success()") == "success()"


def test_acceptance_runtime_preparation_never_freezes_or_builds_an_installer(tmp_path, monkeypatch):
    module = load_script("build_windows")
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(module, "BUILD_ROOT", tmp_path)
    monkeypatch.setattr(module, "_require_lock", lambda: None)
    monkeypatch.setattr(module.common, "select_python", lambda *a: "python.exe")
    monkeypatch.setattr(module.common, "select_node", lambda: Path("node.exe"))
    monkeypatch.setattr(module, "_check_target", lambda *a: None)
    calls = []
    monkeypatch.setattr(module, "_prepare_python_runtime", lambda *a: calls.append(a) or Path("python.exe"))
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path))
    for name in ("prepare_node_runtime", "build_backend", "prepare_bundle", "build_inputs"):
        monkeypatch.setattr(module.common, name, lambda *a, **k: pytest.fail("artifact rebuilt"))
    monkeypatch.setattr(module, "_build_tauri", lambda *a: pytest.fail("installer rebuilt"))
    assert module.main(["--runtime-only"]) == 0
    assert calls == [("python.exe",)]
    assert not (tmp_path / "build-inputs.json").exists()


def test_smoke_requires_frozen_durable_startup_even_when_the_launcher_fails():
    smoke = (DESKTOP / "scripts/smoke_windows.ps1").read_text()
    assert "function Assert-FrozenBackendStartup" in smoke
    assertion = smoke.split("function Assert-FrozenBackendStartup", 1)[1].split("\n# The hosted runner", 1)[0]
    assert "if (-not $log.Contains('Application startup complete.'))" in assertion
    assert "if (-not $log.Contains('Durable dispatcher started (frozen=True)'))" in assertion
    assert "if ($failure.Success) { throw $failure.Value }" in assertion
    assert "throw 'desktop_backend_startup_incomplete'" in assertion
    assert "throw 'durable_executor_unavailable'" in assertion
    assert "finally {\n        Assert-FrozenBackendStartup" in smoke
    assert smoke.index("Assert-FrozenBackendStartup (Join-Path") < smoke.index("if (-not $result.healthy")


@pytest.mark.parametrize("code", ["durable_dependency_unavailable", "durable_launch_failed", "durable_executor_unavailable"])
def test_smoke_extracts_the_typed_failure_instead_of_the_traceback_module_name(code):
    smoke = (DESKTOP / "scripts/smoke_windows.ps1").read_text()
    pattern = re.search(r"\[regex\]::Match\(\$log, '([^']+)'\)", smoke).group(1)
    traceback = ('File "easyicu/webserver/durable_workflows.py", line 1, in launch\n'
                 f"easyicu.webserver.durable_workflows.DurableWorkflowError: {code}\n")
    assert re.search(pattern, traceback).group() == code
    assert re.search(pattern, "Application startup complete.\nDurable dispatcher started (frozen=True)") is None


def test_windows_headless_launch_checks_health_and_preserves_data_after_uninstall():
    rust = (DESKTOP / "src-tauri/src/windows.rs").read_text()
    shell = (DESKTOP / "src-tauri/src/lib.rs").read_text()
    smoke = (DESKTOP / "scripts/smoke_windows.ps1").read_text()
    assert 'var_os("LOCALAPPDATA")' in rust
    assert 'root.join("EasyICU")' in rust
    assert "windows::data_root()?" in shell
    assert '#[cfg(target_os = "macos")]\n        RunEvent::Reopen' in shell
    assert shell.index("windows::smoke_test()") < shell.index("let app = tauri::Builder")
    assert shell.index("windows::configure_webview_data()") < shell.index("let app = tauri::Builder")
    assert 'set_var("WEBVIEW2_USER_DATA_FOLDER", directory)' in rust
    assert 'data_root()?.join("runtime").join("webview2")' in rust
    assert "health_check(port, &token)" in rust
    assert "GET /api/catalog HTTP/1.1" in shell
    assert "X-EasyICU-Desktop-Token: {token}" in shell
    assert '(("127.0.0.1", 0))' in shell
    assert 'health_check(port, "invalid-smoke-token")' in rust
    assert "child.kill()" in rust and "child.wait()?" in rust
    assert ".creation_flags(0x08000000)" in rust
    assert '"PYTHONDONTWRITEBYTECODE", "1"' in rust
    assert "--smoke-test" in smoke
    assert '$env:PATH = "$env:SystemRoot\\System32;$env:SystemRoot"' in smoke
    assert '"/S /D=$installDir"' in smoke
    assert '"/S _?=$installDir"' in smoke
    assert "Test-Path -LiteralPath $stateSentinel" in smoke
    assert "Test-Path -LiteralPath $exportSentinel" in smoke
    assert "taskkill.exe" in smoke
