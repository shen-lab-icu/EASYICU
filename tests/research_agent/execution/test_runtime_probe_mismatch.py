"""An image built from other kernel source fails with a named, fixable reason.

Dev9 M2 (job 0376779550ff) stopped with "The governed Research Agent operation
failed" and no cause: the shared checkout had landed kernel changes after the
runner image was built, so the in-image dependency probe refused the image.
That is a host-level fix -- rebuild the image -- and the failure has to say so
without carrying the probe's text across the web boundary, and without the
"start the container runtime" advice meant for a runtime that is down.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import easyicu.research_agent.execution.runner as runner_mod
from easyicu.research_agent.execution.kernel_identity import (
    build_execution_kernel_identity,
)
from easyicu.webserver import agent_pipeline_runs


class _Proc:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, probe_exit: int):
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}).to_parquet(cohort, index=False)
    scripts: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        del args, kwargs
        if cmd[1:3] == ["image", "inspect"]:
            return _Proc(
                stdout=json.dumps({"Id": "sha256:" + "a" * 64, "RepoDigests": []})
            )
        if cmd[1:3] == ["container", "inspect"]:
            return _Proc(returncode=1, stderr="Error: No such container")
        if "importlib.metadata" in " ".join(cmd):
            scripts.append(cmd[-1])
            return _Proc(
                stderr="EasyICU execution-kernel source mismatch: expected x, observed y",
                returncode=probe_exit,
            )
        return _Proc()

    monkeypatch.setattr(runner_mod.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(runner_mod.subprocess, "run", fake_run)
    runner = runner_mod.DockerRunner(workdir=tmp_path / "run", cohort_parquet=cohort)
    return runner, scripts


@pytest.mark.parametrize(
    ("probe_exit", "reason_code"),
    [(86, "runner_image_kernel_mismatch"), (87, "runner_image_lock_mismatch")],
)
def test_a_mismatched_image_is_a_named_runtime_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe_exit: int, reason_code: str
) -> None:
    runner, _scripts = _runner(tmp_path, monkeypatch, probe_exit=probe_exit)

    with pytest.raises(runner_mod.ExecutionRuntimeUnavailableError) as raised:
        runner._capture_runtime_provenance()

    error = raised.value
    assert error.reason_code == reason_code
    assert "Rebuild the image" in str(error)
    # The web receipt carries the owner's closed code, not the probe's text.
    assert agent_pipeline_runs._safe_pipeline_typed_failure(error) == {
        "owner": runner_mod.EXECUTION_RUNTIME_DIAGNOSTIC_OWNER,
        "reason_code": reason_code,
        "runner_kind": "docker",
        "exit_code": probe_exit,
    }
    assert (
        agent_pipeline_runs._pipeline_failure_code(error)
        == "research_pipeline_runner_image_mismatch"
    )


def test_resuming_an_approved_plan_names_the_image_mismatch_too() -> None:
    source = Path(agent_pipeline_runs.__file__).read_text(encoding="utf-8")
    _, _, resume = source.partition("def resume_research_pipeline(")
    resume = resume.partition("\ndef ")[0]

    assert "_RUNNER_IMAGE_MISMATCH_REASONS" in resume
    assert '"research_pipeline_runner_image_mismatch"' in resume
    assert '"research_pipeline_execution_runtime_unavailable"' in resume


def test_any_other_probe_failure_keeps_its_generic_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, _scripts = _runner(tmp_path, monkeypatch, probe_exit=1)

    with pytest.raises(RuntimeError, match="dependency capture failed") as raised:
        runner._capture_runtime_provenance()

    assert not isinstance(raised.value, runner_mod.ExecutionRuntimeUnavailableError)


def test_the_probe_script_exits_with_the_code_the_host_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the real generated probe outside Docker against this checkout."""

    package_root = Path(runner_mod.__file__).resolve().parents[2]
    real = build_execution_kernel_identity(package_root)
    runner, scripts = _runner(tmp_path, monkeypatch, probe_exit=86)
    with pytest.raises(runner_mod.ExecutionRuntimeUnavailableError):
        runner._capture_runtime_provenance()
    matching_kernel = scripts[-1]

    wrong = dataclasses.replace(real, source_sha256="0" * 64)
    monkeypatch.setattr(
        runner_mod, "build_execution_kernel_identity", lambda *_a, **_k: wrong
    )
    runner._cached_runtime_provenance = None
    runner._cached_runtime_requirements = None
    with pytest.raises(runner_mod.ExecutionRuntimeUnavailableError):
        runner._capture_runtime_provenance()
    mismatched_kernel = scripts[-1]
    # The fake Docker replaced the process-wide subprocess.run; run the
    # captured scripts for real.
    monkeypatch.undo()

    env = {"PYTHONPATH": str(package_root.parent), "PATH": "/usr/bin:/bin"}
    kernel = subprocess.run(
        [sys.executable, "-c", mismatched_kernel],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert kernel.returncode == runner_mod._PROBE_KERNEL_MISMATCH_EXIT
    assert "execution-kernel source mismatch" in kernel.stderr

    # This host has the kernel the script expects but no in-image lock file.
    lock = subprocess.run(
        [sys.executable, "-c", matching_kernel],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert lock.returncode == runner_mod._PROBE_LOCK_MISMATCH_EXIT
    assert "requirements.lock unavailable" in lock.stderr
