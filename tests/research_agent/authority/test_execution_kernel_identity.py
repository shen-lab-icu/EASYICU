from __future__ import annotations

import subprocess
import sys
import os
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


def _synthetic_kernel(tmp_path):
    root = tmp_path / "easyicu"
    seed = root / "research_agent/methods/counts.py"
    seed.parent.mkdir(parents=True)
    (root / "__init__.py").write_text("")
    seed.write_text("from easyicu.helper_a import count\n")
    (root / "helper_a.py").write_text("count = 1\n")
    (root / "helper_b.py").write_text("count = 2\n")
    lock = root / "research_agent/runner_image/requirements.lock"
    lock.parent.mkdir()
    lock.write_text("synthetic==1\n")
    return root, seed


def test_repeated_identity_checks_reuse_parsing_but_rehash_source(
    tmp_path, monkeypatch
):
    root, seed = _synthetic_kernel(tmp_path)
    parsed = []
    original = kernel_identity.ast.parse

    def parse(source, *args, **kwargs):
        if str(kwargs.get("filename", "")).startswith(str(root)):
            parsed.append(kwargs["filename"])
        return original(source, *args, **kwargs)

    monkeypatch.setattr(kernel_identity.ast, "parse", parse)
    first = build_execution_kernel_identity(root)
    first_parses = len(parsed)
    assert build_execution_kernel_identity(root) == first
    assert len(parsed) == first_parses, "unchanged sources were parsed again"
    # Neither size nor timestamp is a sufficient key for a source cache.
    before = seed.stat()
    seed.write_text("from easyicu.helper_b import count\n")
    os.utime(seed, ns=(before.st_atime_ns, before.st_mtime_ns))
    second = build_execution_kernel_identity(root)
    assert second.source_sha256 != first.source_sha256
    assert second.files_sha256 != first.files_sha256
    assert "helper_b.py" in execution_kernel_relative_paths(root)
    assert "helper_a.py" not in execution_kernel_relative_paths(root)
    # Non-import edits still change authority even with a warm parse cache.
    helper = root / "helper_b.py"
    helper.write_text("count = 3\n")
    third = build_execution_kernel_identity(root)
    assert third.source_sha256 != second.source_sha256
    assert third.files_sha256 == second.files_sha256


def test_warm_import_cache_resolves_added_and_removed_modules(tmp_path):
    root, seed = _synthetic_kernel(tmp_path)
    seed.write_text("from easyicu.optional import count\n")
    before = execution_kernel_relative_paths(root)
    helper = root / "optional.py"
    helper.write_text("count = 1\n")
    assert "optional.py" in execution_kernel_relative_paths(root)
    helper.unlink()
    assert execution_kernel_relative_paths(root) == before


def test_warm_import_cache_still_rejects_invalid_source_and_symlinks(tmp_path):
    root, seed = _synthetic_kernel(tmp_path)
    execution_kernel_relative_paths(root)
    seed.write_text("def broken(\n")
    with pytest.raises(
        kernel_identity.ExecutionKernelIdentityError, match="could not be parsed"
    ):
        execution_kernel_relative_paths(root)
    seed.unlink()
    seed.symlink_to(root / "helper_a.py")
    with pytest.raises(
        kernel_identity.ExecutionKernelIdentityError, match="not a regular file"
    ):
        execution_kernel_relative_paths(root)


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


def test_figure_renderers_do_not_bind_host_publication_workflows() -> None:
    package_root = Path(__file__).resolve().parents[3] / "src" / "easyicu"
    paths = set(execution_kernel_relative_paths(package_root))

    assert {
        "research_agent/figures/association_prior_outputs.py",
        "research_agent/figures/sensitivity_prior_outputs.py",
        "research_agent/figures/prior_output_support.py",
    } <= paths
    host_workflows = {
        "research_agent/reporting/publication_bundles.py",
        "research_agent/reporting/readiness.py",
        "research_agent/reporting/scientific_maturity.py",
        "research_agent/orchestration/exposure_grouping_phase.py",
        "research_agent/agents/exposure_grouping_planner.py",
        "research_agent/planning/progressive_compiler.py",
        "research_agent/providers/structured_retry.py",
        "research_agent/resources/coder.py",
    }
    assert not (paths & host_workflows), sorted(paths & host_workflows)


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
