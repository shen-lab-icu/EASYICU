import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
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

    assert 'PYINSTALLER_VERSION = "6.22.2"' in script
    assert '"--onedir"' in script
    assert '"_internal"' in script
    assert '"--collect-data"' in script
    assert 'os.environ["EASYICU_HOME"]' in backend
    assert 'parser.add_argument("--parent-pid", required=True' in backend
    assert "psutil.pid_exists(parent_pid)" in backend
    assert "PROJECT_ROOT" not in backend
