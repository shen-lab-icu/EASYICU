"""Fresh-wheel installation contract for the pinned Pi runtime."""

from __future__ import annotations

import json
from pathlib import Path

from easyicu.scripts.extract_features import build_parser
from easyicu.webserver.pi_copilot.install import (
    install_runtime,
    runtime_is_installed,
)


def test_easyicu_cli_exposes_explicit_copilot_install() -> None:
    parsed = build_parser().parse_args(
        ["copilot", "install", "--runtime-dir", "/tmp/pi-test-runtime"]
    )

    assert parsed.command == "copilot"
    assert parsed.copilot_command == "install"
    assert parsed.runtime_dir == Path("/tmp/pi-test-runtime")


def test_pi_event_projection_is_in_both_distribution_manifests() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    relative = "pi_copilot/node_app/src/event-projection.mjs"
    assert f"src/easyicu/webserver/{relative}" in manifest
    assert f'"{relative}"' in pyproject


def test_installer_uses_lockfile_without_scripts_or_ambient_secrets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    (source / "src").mkdir(parents=True)
    (source / "package.json").write_text("{}", encoding="utf-8")
    (source / "package-lock.json").write_text("{}", encoding="utf-8")
    (source / "README.md").write_text("runtime", encoding="utf-8")
    (source / "THIRD_PARTY_NOTICES.md").write_text("MIT", encoding="utf-8")
    (source / "src" / "main.mjs").write_text("", encoding="utf-8")
    (source / "src" / "event-projection.mjs").write_text("", encoding="utf-8")
    calls = []

    def fake_run(command, *, cwd, env, check):
        calls.append((command, Path(cwd), dict(env), check))
        package = (
            Path(cwd)
            / "node_modules"
            / "@earendil-works"
            / "pi-coding-agent"
            / "package.json"
        )
        package.parent.mkdir(parents=True)
        package.write_text(json.dumps({"version": "0.84.1"}), encoding="utf-8")

    monkeypatch.setattr(
        "easyicu.webserver.pi_copilot.install.subprocess.run",
        fake_run,
    )
    target = install_runtime(
        destination=tmp_path / "runtime" / "0.84.1",
        source=source,
        npm_binary="/usr/local/bin/npm",
        environ={
            "PATH": "/usr/local/bin:/usr/bin",
            "HOME": str(tmp_path),
            "LANG": "en_US.UTF-8",
            "OPENAI_API_KEY": "must-not-leak",
            "DATABASE_PASSWORD": "must-not-leak",
        },
    )

    assert runtime_is_installed(target)
    command, cwd, child_env, check = calls[0]
    assert command == ["/usr/local/bin/npm", "ci", "--ignore-scripts"]
    assert cwd.parent == target.parent
    assert check is True
    assert "OPENAI_API_KEY" not in child_env
    assert "DATABASE_PASSWORD" not in child_env
