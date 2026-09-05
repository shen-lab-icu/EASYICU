import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[2]
DESKTOP = ROOT / "desktop"


def test_desktop_bundle_uses_one_private_runtime_and_local_loading_page():
    config = json.loads(
        (DESKTOP / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8")
    )

    assert config["identifier"] == "org.easyicu.desktop"
    assert config["app"]["withGlobalTauri"] is False
    assert config["app"]["windows"] == [
        {
            "label": "main",
            "title": "EasyICU",
            "url": "index.html",
            "width": 1440,
            "height": 940,
            "minWidth": 1080,
            "minHeight": 680,
            "center": True,
            "resizable": True,
        }
    ]
    assert config["bundle"]["resources"] == [
        "resources/backend",
        "resources/node",
        "resources/NODE_LICENSE",
    ]


def test_desktop_webview_has_no_shell_or_filesystem_permission():
    capabilities = json.loads(
        (DESKTOP / "src-tauri" / "capabilities" / "default.json").read_text(
            encoding="utf-8"
        )
    )
    assert capabilities["windows"] == ["main"]
    assert capabilities["permissions"] == ["core:default"]


def test_desktop_build_is_reproducible_and_source_checkout_independent():
    script = (DESKTOP / "scripts" / "build_macos.py").read_text(encoding="utf-8")
    backend = (DESKTOP / "backend_entry.py").read_text(encoding="utf-8")

    assert "--require-hashes" in script
    assert '"--onedir"' in script
    assert '"_internal"' in script
    assert '"--collect-data"' in script
    assert 'os.environ["EASYICU_HOME"]' in backend
    assert 'parser.add_argument("--parent-pid", required=True' in backend
    assert "psutil.pid_exists(parent_pid)" in backend
    assert "PROJECT_ROOT" not in backend
    assert '"--paths"' not in script


def test_desktop_installer_uses_the_lock_and_checks_dependencies(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "desktop_build_contract", DESKTOP / "scripts" / "build_macos.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "BUILD_ROOT", tmp_path)
    monkeypatch.setattr(module, "VENV_ROOT", tmp_path / "venv")
    python = module._venv_python()
    python.parent.mkdir(parents=True)
    python.touch()
    calls = []
    monkeypatch.setattr(module, "_run", lambda command, **kwargs: calls.append(command))

    assert module._prepare_python_runtime("python3.11") == python
    assert calls[0] == [
        str(python), "-m", "pip", "install", "--require-hashes",
        "-r", str(module.PYTHON_LOCK),
    ]
    assert "--no-deps" in calls[1]
    assert "--no-build-isolation" in calls[1]
    assert "-e" not in calls[1]
    assert calls[-1] == [str(python), "-m", "pip", "check"]


def test_desktop_lock_satisfies_declared_runtime_and_build_inputs():
    import re
    from packaging.requirements import Requirement

    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    lock = (DESKTOP / "requirements-macos-arm64-py311.lock").read_text()
    pins = dict(re.findall(r"^([a-zA-Z0-9_.-]+)==([^\s]+)", lock, re.MULTILINE))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    requirements = project["dependencies"] + project["optional-dependencies"]["webapp"]
    requirements += [
        line for line in (DESKTOP / "build-requirements.in").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    for raw in requirements:
        requirement = Requirement(raw)
        key = requirement.name.lower().replace("_", "-")
        assert key in pins, f"{key} is absent from the desktop lock"
        assert requirement.specifier.contains(pins[key]), raw
    assert "--hash=sha256:" in lock
