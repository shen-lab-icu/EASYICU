from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from easyicu.research_agent.execution import kernel_identity
from easyicu.research_agent.execution.kernel_identity import (
    EXECUTION_KERNEL_IDENTITY_SCHEMA,
    RUNNER_IMAGE_KERNEL_MISMATCH_EXIT,
    RUNNER_IMAGE_LOCK_MISMATCH_EXIT,
    build_execution_kernel_identity,
    execution_kernel_relative_paths,
    runner_image_probe_source,
)


def test_execution_kernel_manifest_tracks_runtime_not_control_plane() -> None:
    package_root = Path(__file__).resolve().parents[3] / "src" / "easyicu"

    paths = execution_kernel_relative_paths(package_root)

    assert "research_agent/execution/runners/selection.py" in paths
    assert "research_agent/methods/__init__.py" in paths
    assert "research_agent/figures/__init__.py" in paths
    assert "research_agent/planning/robustness_contract.py" in paths
    assert "research_agent/execution/runner.py" not in paths
    assert "research_agent/agents/progressive_payload.py" not in paths
    assert "research_agent/providers/structured_diagnostics.py" not in paths
    assert "research_agent/planning/progressive_contract.py" not in paths


def test_execution_kernel_identity_separates_host_source_and_runner_inputs(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "easyicu"
    kernel_path = package_root / "kernel.py"
    host_path = package_root / "host_only.py"
    lock_path = package_root / "research_agent" / "runner_image" / "requirements.lock"
    lock_path.parent.mkdir(parents=True)
    kernel_path.write_text("VALUE = 1\n", encoding="utf-8")
    host_path.write_text("HOST = 1\n", encoding="utf-8")
    lock_path.write_text("numpy==2.0.0\n", encoding="utf-8")
    manifest = ("kernel.py",)

    initial = build_execution_kernel_identity(
        package_root,
        relative_paths=manifest,
    )
    host_path.write_text("HOST = 2\n", encoding="utf-8")
    after_host_change = build_execution_kernel_identity(
        package_root,
        relative_paths=manifest,
    )
    kernel_path.write_text("VALUE = 2\n", encoding="utf-8")
    after_kernel_change = build_execution_kernel_identity(
        package_root,
        relative_paths=manifest,
    )
    lock_path.write_text("numpy==2.1.0\n", encoding="utf-8")
    after_lock_change = build_execution_kernel_identity(
        package_root,
        relative_paths=manifest,
    )

    assert initial.schema_version == EXECUTION_KERNEL_IDENTITY_SCHEMA
    assert initial == after_host_change
    assert after_kernel_change.source_sha256 != initial.source_sha256
    assert after_kernel_change.identity_sha256 != initial.identity_sha256
    assert after_lock_change.source_sha256 == after_kernel_change.source_sha256
    assert after_lock_change.requirements_lock_sha256 != (
        after_kernel_change.requirements_lock_sha256
    )
    assert after_lock_change.identity_sha256 != after_kernel_change.identity_sha256


def test_the_runner_image_probe_reproves_the_identity_it_was_built_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the probe on a synthetic installed package, outside Docker."""

    package_root = tmp_path / "easyicu"
    lock_path = package_root / "research_agent" / "runner_image" / "requirements.lock"
    lock_path.parent.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    kernel_path = package_root / "kernel.py"
    kernel_path.write_text("VALUE = 1\n", encoding="utf-8")
    lock_path.write_text("numpy==2.0.0\n", encoding="utf-8")
    manifest = ("__init__.py", "kernel.py")
    identity = build_execution_kernel_identity(package_root, relative_paths=manifest)
    monkeypatch.setattr(kernel_identity, "RUNNER_IMAGE_LOCK_PATH", str(lock_path))
    source = runner_image_probe_source(identity, manifest)

    def probe() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", source],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin"},
            timeout=120,
        )

    matching = probe()
    lock_path.write_text("numpy==2.1.0\n", encoding="utf-8")
    other_lock = probe()
    lock_path.write_text("numpy==2.0.0\n", encoding="utf-8")
    kernel_path.write_text("VALUE = 2\n", encoding="utf-8")
    other_kernel = probe()
    kernel_path.unlink()
    missing_kernel = probe()

    assert matching.returncode == 0, matching.stderr
    rows = matching.stdout.splitlines()
    assert rows and all("==" in row for row in rows)
    assert rows == sorted(rows, key=lambda row: row.split("==", 1)[0].casefold())
    assert other_lock.returncode == RUNNER_IMAGE_LOCK_MISMATCH_EXIT
    assert "requirements.lock mismatch" in other_lock.stderr
    assert other_kernel.returncode == RUNNER_IMAGE_KERNEL_MISMATCH_EXIT
    assert "execution-kernel source mismatch" in other_kernel.stderr
    assert missing_kernel.returncode == RUNNER_IMAGE_KERNEL_MISMATCH_EXIT
    assert "execution-kernel file unavailable: kernel.py" in missing_kernel.stderr
