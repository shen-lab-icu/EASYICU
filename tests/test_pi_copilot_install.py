"""Fresh-wheel installation contract for the pinned Pi runtime."""

from __future__ import annotations

import json
from pathlib import Path

from easyicu.scripts.extract_features import build_parser
from easyicu.webserver.pi_copilot.install import (
    install_runtime,
    runtime_manifest,
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
    budget_relative = "pi_copilot/node_app/src/shell-budget.mjs"
    assert f"src/easyicu/webserver/{budget_relative}" in manifest
    assert f'"{budget_relative}"' in pyproject


def test_installer_uses_lockfile_without_scripts_or_ambient_secrets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    (source / "src").mkdir(parents=True)
    (source / "package.json").write_text("{}", encoding="utf-8")
    (source / "package-lock.json").write_text(
        json.dumps(
            {
                "packages": {
                    "node_modules/@earendil-works/pi-coding-agent": {
                        "version": "0.84.1"
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (source / "README.md").write_text("runtime", encoding="utf-8")
    (source / "THIRD_PARTY_NOTICES.md").write_text("MIT", encoding="utf-8")
    (source / "src" / "main.mjs").write_text("", encoding="utf-8")
    (source / "src" / "event-projection.mjs").write_text("", encoding="utf-8")
    (source / "src" / "shell-budget.mjs").write_text("", encoding="utf-8")
    calls = []

    class Result:
        stdout = "v24.11.0\n"

    def fake_run(
        command,
        *,
        cwd,
        env,
        check,
        capture_output=False,
        text=False,
    ):
        calls.append((command, Path(cwd), dict(env), check))
        if command[-1] == "--version":
            return Result()
        package = (
            Path(cwd)
            / "node_modules"
            / "@earendil-works"
            / "pi-coding-agent"
            / "package.json"
        )
        package.parent.mkdir(parents=True)
        package.write_text(json.dumps({"version": "0.84.1"}), encoding="utf-8")
        (package.parent / "dist").mkdir()
        (package.parent / "dist" / "index.js").write_text(
            "export const pinned = true;\n",
            encoding="utf-8",
        )
        return Result()

    monkeypatch.setattr(
        "easyicu.webserver.pi_copilot.install.subprocess.run",
        fake_run,
    )
    target = install_runtime(
        destination=tmp_path / "runtime" / "0.84.1",
        source=source,
        npm_binary="/usr/local/bin/npm",
        node_binary="/usr/local/bin/node",
        environ={
            "PATH": "/usr/local/bin:/usr/bin",
            "HOME": str(tmp_path),
            "LANG": "en_US.UTF-8",
            "OPENAI_API_KEY": "must-not-leak",
            "DATABASE_PASSWORD": "must-not-leak",
        },
    )

    assert runtime_is_installed(target, source=source)
    manifest = json.loads(
        (target / "runtime-manifest.json").read_text(encoding="utf-8")
    )
    assert all(
        manifest[key] == value for key, value in runtime_manifest(source).items()
    )
    assert manifest["installation"]["node_version"] == "24.11.0"
    executable_files = manifest["installation"]["executable_files"]
    assert (
        "node_modules/@earendil-works/pi-coding-agent/dist/index.js" in executable_files
    )
    command, cwd, child_env, check = calls[0]
    assert command == ["/usr/local/bin/npm", "ci", "--ignore-scripts"]
    assert cwd.parent == target.parent
    assert check is True
    assert "OPENAI_API_KEY" not in child_env
    assert "DATABASE_PASSWORD" not in child_env

    (
        target
        / "node_modules"
        / "@earendil-works"
        / "pi-coding-agent"
        / "dist"
        / "index.js"
    ).write_text(
        "export const tampered = true;\n",
        encoding="utf-8",
    )
    assert runtime_is_installed(target, source=source) is False

    (
        target
        / "node_modules"
        / "@earendil-works"
        / "pi-coding-agent"
        / "dist"
        / "index.js"
    ).write_text(
        "export const pinned = true;\n",
        encoding="utf-8",
    )
    (target / "src" / "main.mjs").write_text("tampered", encoding="utf-8")
    assert runtime_is_installed(target, source=source) is False

    before = runtime_manifest(source)["runtime_manifest_sha256"]
    (source / "src" / "main.mjs").write_text("changed source", encoding="utf-8")
    after = runtime_manifest(source)["runtime_manifest_sha256"]
    assert before != after
