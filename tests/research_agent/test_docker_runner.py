"""DockerRunner contract tests (T3.1).

Real Docker is not available in CI / the dev sandbox, so these tests
mock ``subprocess.run`` and ``shutil.which`` to verify:

1. ``shutil.which("docker")`` is consulted at construction; missing
   binary raises a clean ``FileNotFoundError``.
2. The composed argv contains the right safety knobs:
   ``--rm``, ``--init``, ``--network=none``, RO cohort/run mounts, independent
   RW outputs-only mount, env injection, image trailer, and ``python -u``.
3. ``cohort_parquet`` is mounted read-only at ``/cohort.parquet``
   and ``COHORT_PARQUET`` points there.
4. Custom image, network, mounts, cpu/memory limits, user, and
   platform flow into the argv.
5. Subprocess timeout surfaces as ``RunResult(timed_out=True,
   returncode=-1)`` with the timeout message in the log.
6. The pipeline's ``runner_kind="docker"`` actually constructs a
   :class:`DockerRunner` (smoke check via ``_build_runner``).
"""

from __future__ import annotations

import json
import inspect
import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import pytest


def _make_cohort(tmp_path: Path) -> Path:
    p = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}).to_parquet(p, index=False)
    return p


class _FakeProc:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _install_fake_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    *,
    proc: Optional[_FakeProc] = None,
    raise_timeout: bool = False,
    stop_returncode: int = 0,
    kill_returncode: int = 0,
    rm_returncode: int = 0,
    wait_returncode: int = 0,
    container_inspect_returncode: int = 1,
    container_inspect_stderr: str = "Error: No such container",
    cidfile_value: Optional[str] = "c" * 64,
    run_exception: Optional[BaseException] = None,
    run_side_effect: Optional[Callable[[], None]] = None,
    captured: Optional[List[List[str]]] = None,
) -> None:
    """Replace runner.subprocess.run with a deterministic stub."""
    proc = proc or _FakeProc(stdout="hello-from-container\n")

    def fake_run(cmd, *args, **kwargs):
        if captured is not None:
            captured.append(list(cmd))
        if len(cmd) >= 3 and cmd[1:3] == ["image", "inspect"]:
            return _FakeProc(
                stdout=json.dumps(
                    {
                        "Id": "sha256:" + "a" * 64,
                        "RepoDigests": ["img@sha256:" + "b" * 64],
                    }
                )
            )
        if len(cmd) >= 3 and cmd[1:3] == ["container", "inspect"]:
            return _FakeProc(
                returncode=container_inspect_returncode,
                stderr=container_inspect_stderr,
            )
        if "importlib.metadata" in " ".join(cmd):
            return _FakeProc(
                stdout=(
                    "numpy==2.0.0\n"
                    "pandas==2.2.0\n"
                    "scipy==1.14.0\n"
                    "matplotlib==3.9.0\n"
                    "statsmodels==0.14.0\n"
                    "scikit-learn==1.5.0\n"
                    "pyarrow==23.0.0\n"
                    "seaborn==0.13.0\n"
                    "lifelines==0.30.0\n"
                )
            )
        if len(cmd) >= 2 and cmd[1] == "pull":
            return proc
        if len(cmd) >= 2 and cmd[1] == "stop":
            return _FakeProc(returncode=stop_returncode)
        if len(cmd) >= 2 and cmd[1] == "kill":
            return _FakeProc(returncode=kill_returncode)
        if len(cmd) >= 2 and cmd[1] == "rm":
            return _FakeProc(returncode=rm_returncode)
        if len(cmd) >= 2 and cmd[1] == "wait":
            return _FakeProc(returncode=wait_returncode)
        cidfile_args = [token for token in cmd if token.startswith("--cidfile=")]
        if run_side_effect is not None and cidfile_args:
            run_side_effect()
        if run_exception is not None and cidfile_args:
            raise run_exception
        if raise_timeout and cidfile_args:
            if cidfile_value is not None:
                Path(cidfile_args[0].split("=", 1)[1]).write_text(
                    cidfile_value, encoding="utf-8"
                )
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 0))
        # Mimic the side-effect a real container would have so the
        # caller's ``out_dir.iterdir()`` finds something.
        return proc

    # Patch the module-level subprocess (not the global one) so other
    # tests are unaffected.
    import easyicu.research_agent.execution.runner as runner_mod

    monkeypatch.setattr(runner_mod.subprocess, "run", fake_run)


def _force_docker_present(
    monkeypatch: pytest.MonkeyPatch, fake_path: str = "/usr/bin/docker"
) -> None:
    import easyicu.research_agent.execution.runner as runner_mod

    monkeypatch.setattr(runner_mod.shutil, "which", lambda _name: fake_path)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_missing_docker_binary_raises(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cohort = _make_cohort(tmp_path)
    import easyicu.research_agent.execution.runner as runner_mod

    monkeypatch.setattr(runner_mod.shutil, "which", lambda _n: None)

    with pytest.raises(FileNotFoundError, match="not found on PATH"):
        ra.DockerRunner(workdir=tmp_path / "run", cohort_parquet=cohort)


def test_constructor_resolves_docker_via_which(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch, "/opt/local/bin/docker")
    runner = ra.DockerRunner(workdir=tmp_path / "run", cohort_parquet=cohort)
    assert runner.docker_executable == "/opt/local/bin/docker"


def test_constructor_honours_environment_image(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    monkeypatch.setenv("EASYICU_RUNNER_IMAGE", "company/easyicu-ra:1.4")
    runner = ra.DockerRunner(workdir=tmp_path / "run", cohort_parquet=cohort)
    assert runner.image == "company/easyicu-ra:1.4"


def test_runner_authority_changes_when_same_tag_resolves_to_new_image(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    runner = ra.DockerRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort,
        image="company/easyicu-ra:latest",
        extra_env={"TRAJECTORY_PARQUET": str(tmp_path / "trajectory.parquet")},
    )
    trajectory = tmp_path / "trajectory.parquet"
    trajectory.write_bytes(b"trajectory-a")
    image_identity = ["sha256:" + "a" * 64, ()]
    monkeypatch.setattr(
        runner,
        "_inspect_image_identity",
        lambda: tuple(image_identity),
    )
    first = runner.authority_identity_sha256
    image_identity[0] = "sha256:" + "c" * 64
    second = runner.authority_identity_sha256
    trajectory.write_bytes(b"trajectory-b")
    third = runner.authority_identity_sha256

    assert first != second
    assert second != third


# ---------------------------------------------------------------------------
# build_command shape
# ---------------------------------------------------------------------------


def test_build_command_has_safety_knobs(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    runner = ra.DockerRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort,
        image="my/image:latest",
    )
    step_dir, script_path, out_dir = runner.prepare_step_dir("step_x")
    script_path.write_text("print('hi')\n", encoding="utf-8")
    cmd = runner.build_command(
        step_id="step_x",
        script_path=script_path,
        out_dir=out_dir,
    )
    joined = " ".join(cmd)

    assert cmd[0] == runner.docker_executable
    assert cmd[1] == "run"
    assert "--rm" in cmd
    assert "--init" in cmd
    assert "--network=none" in cmd
    assert "--workdir=/easyicu-run/steps/step_x" in cmd
    assert "--read-only" in cmd
    assert "--cap-drop=ALL" in cmd
    assert "--security-opt=no-new-privileges" in cmd
    if runner.user:
        assert f"--user={runner.user}" in cmd
    assert "my/image:latest" in cmd
    # Image must appear before "python" trailer.
    assert cmd.index("my/image:latest") < cmd.index("python")
    # The command runs the immutable script outside the read-only run-root
    # mount so Docker Desktop has no nested bind target to tear down.
    assert cmd[-3:] == [
        "python",
        "-u",
        "/easyicu-analysis.py",
    ]
    # Cohort mount: read-only, container path /cohort.parquet.
    assert any(
        "type=bind" in s and "readonly" in s and "target=/cohort.parquet" in s
        for s in cmd
    ), f"cohort mount missing in {joined}"
    # Run root is RO and the current attempt output is an independent RW bind.
    assert any(
        "type=bind" in s and "target=/easyicu-run" in s and "readonly" in s for s in cmd
    ), f"run-root mount missing in {joined}"
    assert any(
        "type=bind" in s and "target=/easyicu-step-output" in s and "readonly" not in s
        for s in cmd
    ), f"output mount missing in {joined}"
    assert not any(
        "type=bind" in s and s.endswith("target=/easyicu-run/steps/step_x") for s in cmd
    ), f"step directory must not be writable in {joined}"
    # Env injection.
    assert "-e" in cmd
    env_pairs = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-e"]
    env_dict = dict(p.split("=", 1) for p in env_pairs)
    assert env_dict["COHORT_PARQUET"] == "/cohort.parquet"
    assert env_dict["STEP_OUT_DIR"] == "/easyicu-step-output"
    assert env_dict["EASYICU_STEP_ID"] == "step_x"
    assert env_dict["EASYICU_RUN_DIR"] == "/easyicu-run"
    assert env_dict["MPLBACKEND"] == "Agg"
    assert env_dict["HOME"] == "/tmp"
    assert env_dict["MPLCONFIGDIR"] == "/tmp/matplotlib"


def test_build_command_maps_resolved_inputs_manifest_into_container(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    run_dir = tmp_path / "run"
    runner = ra.DockerRunner(workdir=run_dir, cohort_parquet=cohort)
    step_dir, script_path, out_dir = runner.prepare_step_dir("consume")
    script_path.write_text("print('hi')\n", encoding="utf-8")
    manifest = run_dir / "resolved_inputs" / "consume.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('{"schema_version":"1.0"}\n', encoding="utf-8")

    cmd = runner.build_command(
        step_id="consume",
        script_path=script_path,
        out_dir=out_dir,
        resolved_inputs_path=manifest,
    )

    env_pairs = [cmd[i + 1] for i, token in enumerate(cmd) if token == "-e"]
    env_dict = dict(pair.split("=", 1) for pair in env_pairs)
    assert env_dict["EASYICU_RESOLVED_INPUTS_JSON"] == (
        "/easyicu-run/resolved_inputs/consume.json"
    )


def test_run_owned_cohort_uses_one_canonical_container_path(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _force_docker_present(monkeypatch)
    run_dir = tmp_path / "run"
    cohort = run_dir / "evidence" / "analysis_cohort.parquet"
    cohort.parent.mkdir(parents=True)
    cohort.write_bytes(b"bound cohort")
    runner = ra.DockerRunner(workdir=run_dir, cohort_parquet=cohort)
    _, script_path, out_dir = runner.prepare_step_dir("consume")
    script_path.write_text("print('hi')\n", encoding="utf-8")

    cmd = runner.build_command(
        step_id="consume",
        script_path=script_path,
        out_dir=out_dir,
    )

    env_pairs = [cmd[i + 1] for i, token in enumerate(cmd) if token == "-e"]
    env_dict = dict(pair.split("=", 1) for pair in env_pairs)
    expected = "/easyicu-run/evidence/analysis_cohort.parquet"
    assert env_dict["COHORT_PARQUET"] == expected
    assert env_dict["COHORT_PATH"] == expected
    assert env_dict["EASYICU_COHORT_PATH"] == expected
    assert env_dict["EASYICU_COHORT_PARQUET"] == expected


def test_build_command_maps_digest_bound_authority_snapshot_into_container(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from easyicu.research_agent.execution.runner import (
        _capture_run_artifact_authority_snapshot,
    )

    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    run_dir = tmp_path / "run"
    runner = ra.DockerRunner(workdir=run_dir, cohort_parquet=cohort)
    step_dir, script_path, out_dir = runner.prepare_step_dir("consume")
    script_path.write_text("print('hi')\n", encoding="utf-8")
    (run_dir / "manifest_partial.json").write_text(
        json.dumps(
            {
                "checkpoint_sequence": 3,
                "per_step_records": [{"step_id": "01_primary", "status": "ok"}],
            }
        ),
        encoding="utf-8",
    )
    snapshot_path, snapshot_sha256, error = _capture_run_artifact_authority_snapshot(
        workdir=run_dir,
        step_dir=step_dir,
    )
    assert snapshot_path is not None
    assert snapshot_sha256
    assert error is None

    cmd = runner.build_command(
        step_id="consume",
        script_path=script_path,
        out_dir=out_dir,
        authority_snapshot_path=snapshot_path,
        authority_snapshot_sha256=snapshot_sha256,
    )

    env_pairs = [cmd[i + 1] for i, token in enumerate(cmd) if token == "-e"]
    env_dict = dict(pair.split("=", 1) for pair in env_pairs)
    assert env_dict["EASYICU_RUN_ARTIFACT_AUTHORITY_SNAPSHOT"] == (
        "/easyicu-run/steps/consume/.run_artifact_authority_snapshot.json"
    )
    assert env_dict["EASYICU_RUN_ARTIFACT_AUTHORITY_SNAPSHOT_SHA256"] == snapshot_sha256
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert set(snapshot["authority"]) == {
        "run_id",
        "checkpoint_sequence",
        "per_step_records",
        "evidence",
    }


def test_build_command_passes_through_advanced_flags(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    extra_mount_src = tmp_path / "extra"
    extra_mount_src.mkdir()
    runner = ra.DockerRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort,
        image="i:1",
        network="bridge",
        cpu_limit="1.5",
        memory_limit="2g",
        user="1000:1000",
        platform="linux/amd64",
        extra_mounts=[(str(extra_mount_src), "/easyicu-extra/extra", "ro")],
        extra_env={"PUBMED_API_KEY": "abc"},
    )
    step_dir, script_path, out_dir = runner.prepare_step_dir("step_y")
    script_path.write_text("print('hi')\n", encoding="utf-8")
    cmd = runner.build_command(
        step_id="step_y",
        script_path=script_path,
        out_dir=out_dir,
    )
    assert "--network=bridge" in cmd
    assert "--cpus=1.5" in cmd
    assert "--memory=2g" in cmd
    assert "--user=1000:1000" in cmd
    assert "--platform=linux/amd64" in cmd
    assert any(
        "type=bind" in s and f"source={extra_mount_src}" in s and "readonly" in s
        for s in cmd
    )
    env_pairs = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-e"]
    assert "PUBMED_API_KEY=abc" in env_pairs


@pytest.mark.parametrize(
    "target",
    [
        "/cohort.parquet",
        "/easyicu-run/manifest_partial.json",
        "/easyicu-inputs/forged",
        "/easyicu-extra",
    ],
)
def test_docker_runner_rejects_authority_shadowing_extra_mounts(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    source = tmp_path / "forged"
    source.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="below /easyicu-extra"):
        ra.DockerRunner(
            workdir=tmp_path / "run",
            cohort_parquet=cohort,
            image="i:1",
            extra_mounts=[(str(source), target, "ro")],
        )


@pytest.mark.parametrize(
    ("source_name", "target"),
    [
        ("safe", "/easyicu-extra/safe,target=/cohort.parquet"),
        ("source,with-comma", "/easyicu-extra/safe"),
        ("source=with-equals", "/easyicu-extra/safe"),
    ],
)
def test_docker_runner_rejects_mount_csv_injection(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_name: str,
    target: str,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    source = tmp_path / source_name
    source.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe mount syntax"):
        ra.DockerRunner(
            workdir=tmp_path / "run",
            cohort_parquet=cohort,
            image="i:1",
            extra_mounts=[(str(source), target, "ro")],
        )


def test_docker_runner_rejects_invalid_extra_env_key(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)

    with pytest.raises(ValueError, match="invalid environment key"):
        ra.DockerRunner(
            workdir=tmp_path / "run",
            cohort_parquet=cohort,
            image="i:1",
            extra_env={"COHORT_PARQUET=forged": "yes"},
        )


@pytest.mark.parametrize("kind", ["device", "fifo", "socket", "symlink", "root"])
def test_docker_runner_rejects_unsafe_auto_mounted_path_env(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    if kind == "device":
        candidate = Path("/dev/null")
    elif kind == "root":
        candidate = Path("/")
    elif kind == "fifo":
        candidate = tmp_path / "input.fifo"
        os.mkfifo(candidate)
    elif kind == "socket":
        candidate = Path("/tmp") / f"easyicu-{uuid.uuid4().hex}.sock"
        listener = socket.socket(socket.AF_UNIX)
        listener.bind(str(candidate))
    else:
        real = tmp_path / "real.txt"
        real.write_text("x", encoding="utf-8")
        candidate = tmp_path / "input-link"
        candidate.symlink_to(real)
    try:
        runner = ra.DockerRunner(
            workdir=tmp_path / "run",
            cohort_parquet=cohort,
            image="i:1",
            extra_env={"AUX_INPUT": str(candidate)},
        )
        _step_dir, script_path, out_dir = runner.prepare_step_dir("probe")
        script_path.write_text("print('ok')\n", encoding="utf-8")
        with pytest.raises(ValueError, match="real input|regular file|filesystem root"):
            runner.build_command(
                step_id="probe",
                script_path=script_path,
                out_dir=out_dir,
            )
    finally:
        if kind == "socket":
            listener.close()
            candidate.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "step_id",
    ["safe,target=cohort.parquet", "safe,readonly", "safe=target", "safe\nline"],
)
def test_docker_runner_rejects_step_id_mount_injection(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    step_id: str,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    runner = ra.DockerRunner(workdir=tmp_path / "run", cohort_parquet=cohort)

    with pytest.raises(ValueError, match="unsafe mount syntax"):
        runner.prepare_step_dir(step_id)


# ---------------------------------------------------------------------------
# run() integration with mocked subprocess
# ---------------------------------------------------------------------------


def test_run_invokes_subprocess_and_writes_log(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    captured: List[List[str]] = []
    _install_fake_subprocess(
        monkeypatch,
        proc=_FakeProc(stdout="container-said-hi\n", stderr="", returncode=0),
        captured=captured,
    )

    runner = ra.DockerRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort,
        image="img:0",
    )
    result = runner.run(step_id="probe", code="print('hi')\n")

    assert (
        len(captured) == 4
    ), "image inspect, metadata capture, run, and teardown confirmation are required"
    assert captured[0][1:3] == ["image", "inspect"]
    assert "importlib.metadata" in " ".join(captured[1])
    assert "EasyICU research-agent source mismatch" in " ".join(captured[1])
    immutable_id = "sha256:" + "a" * 64
    assert immutable_id in captured[1]
    assert "img:0" not in captured[1]
    cmd = captured[2]
    assert cmd[0] == runner.docker_executable and cmd[1] == "run"
    assert immutable_id in cmd
    assert "img:0" not in cmd
    assert not any("EASYICU_RUN_ARTIFACT_AUTHORITY_SNAPSHOT=" in token for token in cmd)
    cidfile_arg = next(token for token in cmd if token.startswith("--cidfile="))
    cidfile_path = Path(cidfile_arg.split("=", 1)[1])
    assert cidfile_path.parent == Path(tempfile.gettempdir())
    assert not cidfile_path.exists()

    assert result.succeeded
    assert result.outputs_safe_to_collect is True
    assert result.returncode == 0
    assert "container-said-hi" in result.stdout
    log_text = (result.cwd / "run.log").read_text(encoding="utf-8")
    assert "DockerRunner" in log_text
    assert "img:0" in log_text
    assert "sha256:" in log_text
    provenance = json.loads(
        (result.out_dir / "runner_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["image_id"] == "sha256:" + "a" * 64
    assert provenance["dependency_capture_method"] == (
        "importlib.metadata.distributions"
    )
    assert "lifelines" in provenance["method_capabilities"]
    assert "shap" not in provenance["method_capabilities"]
    requirements_text = (result.out_dir / "runner_requirements.lock.txt").read_text(
        encoding="utf-8"
    )
    assert "numpy==2.0.0" in requirements_text
    assert "# capture_method=importlib.metadata.distributions" in requirements_text
    assert "# research_agent_source_sha256=" in requirements_text
    assert (
        "# generated_by=easyicu.research_agent.execution.runner.DockerRunner"
        in requirements_text
    )
    import easyicu.research_agent.execution.method_capabilities as method_capabilities

    capability_block = method_capabilities.coder_method_capability_block()
    method_capabilities.set_runtime_capability_snapshot_provider(None)
    assert "* lifelines" in capability_block
    assert "* shap" not in capability_block
    # Script persisted to disk before run.
    assert result.script_path.read_text(encoding="utf-8") == "print('hi')\n"
    assert not (result.cwd / ".run_artifact_authority_snapshot.json").exists()


def test_docker_coder_capabilities_use_image_snapshot_before_first_step(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import easyicu.research_agent.execution.method_capabilities as method_capabilities

    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    captured: List[List[str]] = []
    _install_fake_subprocess(monkeypatch, captured=captured)
    monkeypatch.setattr(method_capabilities, "_importable", lambda _name: True)
    ra.DockerRunner(workdir=tmp_path / "run", cohort_parquet=cohort, image="img:tag")

    block = method_capabilities.coder_method_capability_block()
    method_capabilities.set_runtime_capability_snapshot_provider(None)

    assert len(captured) == 2
    assert captured[0][1:3] == ["image", "inspect"]
    assert "importlib.metadata" in " ".join(captured[1])
    assert "sha256:" + "a" * 64 in captured[1]
    assert "img:tag" not in captured[1]
    assert "* lifelines" in block
    assert "* shap" not in block


def test_runtime_provenance_timeout_tears_down_named_probe(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    captured: List[List[str]] = []

    def fake_run(cmd, *args, **kwargs):
        del args
        captured.append(list(cmd))
        if len(cmd) >= 3 and cmd[1:3] == ["image", "inspect"]:
            return _FakeProc(
                stdout=json.dumps({"Id": "sha256:" + "a" * 64, "RepoDigests": []})
            )
        if "importlib.metadata" in " ".join(cmd):
            raise subprocess.TimeoutExpired(
                cmd=cmd,
                timeout=kwargs.get("timeout", 0),
            )
        if len(cmd) >= 2 and cmd[1] in {"stop", "wait"}:
            return _FakeProc()
        raise AssertionError(cmd)

    import easyicu.research_agent.execution.runner as runner_module

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    run_dir = tmp_path / "run"
    runner = ra.DockerRunner(workdir=run_dir, cohort_parquet=cohort)

    with pytest.raises(
        RuntimeError,
        match="execution-runtime dependency capture timed out",
    ):
        runner._capture_runtime_provenance()

    capture_cmd = captured[1]
    assert capture_cmd[1] == "run"
    assert any(token.startswith("--cidfile=") for token in capture_cmd)
    container_name = next(
        token.removeprefix("--name=")
        for token in capture_cmd
        if token.startswith("--name=")
    )
    assert captured[2][1:3] == ["stop", "--timeout=5"]
    assert captured[2][-1] == container_name
    assert captured[3][1:] == ["wait", container_name]
    assert not list(run_dir.glob(".docker-runtime-provenance-*.sentinel"))
    assert not list(run_dir.glob(".docker-runtime-provenance-*.cid"))


def test_run_handles_timeout(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    captured: List[List[str]] = []
    _install_fake_subprocess(monkeypatch, raise_timeout=True, captured=captured)

    runner = ra.DockerRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort,
        timeout_seconds=0.001,
    )
    result = runner.run(step_id="slow", code="print('hi')\n")

    assert result.timed_out is True
    assert result.returncode == -1
    assert not result.succeeded
    log_text = (result.cwd / "run.log").read_text(encoding="utf-8")
    assert "timed_out: True" in log_text
    assert "DockerRunner] timed out" in result.stderr
    assert [cmd[1] for cmd in captured[-2:]] == ["stop", "wait"]
    assert captured[-2][2] == "--timeout=5"
    cidfile_arg = next(
        token for token in captured[-3] if token.startswith("--cidfile=")
    )
    assert not Path(cidfile_arg.split("=", 1)[1]).exists()
    assert not list((tmp_path / "run").glob("*.sentinel"))


def test_timeout_kills_when_graceful_stop_fails(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    captured: List[List[str]] = []
    _install_fake_subprocess(
        monkeypatch,
        raise_timeout=True,
        stop_returncode=1,
        captured=captured,
    )

    runner = ra.DockerRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort,
        timeout_seconds=0.001,
    )
    result = runner.run(step_id="slow", code="print('hi')\n")

    assert result.timed_out is True
    assert [cmd[1] for cmd in captured[-3:]] == ["stop", "kill", "wait"]


def test_timeout_force_removes_when_stop_and_kill_fail(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    captured: List[List[str]] = []
    _install_fake_subprocess(
        monkeypatch,
        raise_timeout=True,
        stop_returncode=1,
        kill_returncode=1,
        rm_returncode=0,
        captured=captured,
    )

    runner = ra.DockerRunner(workdir=tmp_path / "run", cohort_parquet=cohort)
    result = runner.run(step_id="slow", code="print('hi')\n")

    assert result.timed_out is True
    assert result.artefacts
    assert [cmd[1] for cmd in captured[-4:]] == ["stop", "kill", "rm", "wait"]
    assert captured[-2][2] == "--force"
    assert not list((tmp_path / "run").glob("*.sentinel"))


def test_nonzero_docker_return_collects_only_after_confirmed_teardown(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    captured: List[List[str]] = []
    _install_fake_subprocess(
        monkeypatch,
        proc=_FakeProc(stderr="container failed", returncode=2),
        captured=captured,
    )

    runner = ra.DockerRunner(workdir=tmp_path / "run", cohort_parquet=cohort)
    result = runner.run(step_id="failed", code="raise SystemExit(2)\n")

    assert result.returncode == 2
    assert result.timed_out is False
    assert result.outputs_safe_to_collect is True
    assert result.artefacts
    assert [cmd[1] for cmd in captured[-2:]] == ["stop", "wait"]
    assert not list((tmp_path / "run").glob("*.sentinel"))


def test_successful_docker_return_confirms_teardown_before_collecting_outputs(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A successful process exit is not itself proof that mounts are quiescent."""

    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    captured: List[List[str]] = []
    _install_fake_subprocess(
        monkeypatch,
        container_inspect_returncode=1,
        container_inspect_stderr="Error: No such container: completed",
        captured=captured,
    )

    runner = ra.DockerRunner(workdir=tmp_path / "run", cohort_parquet=cohort)
    result = runner.run(step_id="successful", code="print('done')\n")

    assert result.succeeded
    assert result.outputs_safe_to_collect is True
    assert result.artefacts
    assert captured[-1][1:3] == ["container", "inspect"]
    assert not list((tmp_path / "run").glob("*.sentinel"))


def test_successful_docker_return_hides_outputs_when_teardown_is_unconfirmed(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    captured: List[List[str]] = []
    _install_fake_subprocess(
        monkeypatch,
        stop_returncode=1,
        kill_returncode=1,
        rm_returncode=1,
        wait_returncode=1,
        container_inspect_returncode=0,
        captured=captured,
    )
    run_dir = tmp_path / "run"
    runner = ra.DockerRunner(workdir=run_dir, cohort_parquet=cohort)

    result = runner.run(step_id="successful", code="print('done')\n")

    assert result.returncode == 0
    assert result.succeeded is False
    assert result.outputs_safe_to_collect is False
    assert result.artefacts == []
    assert [cmd[1] for cmd in captured[-6:]] == [
        "container",
        "stop",
        "kill",
        "rm",
        "wait",
        "container",
    ]
    assert len(list(run_dir.glob(".docker-successful-*.sentinel"))) == 1


def test_nonzero_docker_return_hides_outputs_when_teardown_is_unconfirmed(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    captured: List[List[str]] = []
    _install_fake_subprocess(
        monkeypatch,
        proc=_FakeProc(stderr="docker transport failed", returncode=125),
        stop_returncode=1,
        kill_returncode=1,
        rm_returncode=1,
        wait_returncode=1,
        container_inspect_returncode=0,
        captured=captured,
    )
    run_dir = tmp_path / "run"
    runner = ra.DockerRunner(workdir=run_dir, cohort_parquet=cohort)

    result = runner.run(step_id="failed", code="print('unknown')\n")

    assert result.returncode == 125
    assert result.timed_out is False
    assert result.outputs_safe_to_collect is False
    assert result.artefacts == []
    assert [cmd[1] for cmd in captured[-5:]] == [
        "stop",
        "kill",
        "rm",
        "wait",
        "container",
    ]
    assert len(list(run_dir.glob(".docker-failed-*.sentinel"))) == 1


@pytest.mark.parametrize("cidfile_value", [None, "invalid-container-id"])
def test_timeout_uses_unique_name_when_cidfile_is_unavailable(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cidfile_value: Optional[str],
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    captured: List[List[str]] = []
    _install_fake_subprocess(
        monkeypatch,
        raise_timeout=True,
        cidfile_value=cidfile_value,
        captured=captured,
    )

    runner = ra.DockerRunner(workdir=tmp_path / "run", cohort_parquet=cohort)
    result = runner.run(step_id="slow", code="print('hi')\n")

    run_cmd = captured[-3]
    container_name = next(
        token.removeprefix("--name=")
        for token in run_cmd
        if token.startswith("--name=")
    )
    assert result.timed_out is True
    assert captured[-2][-1] == container_name
    assert not list((tmp_path / "run").glob("*.sentinel"))


def test_unconfirmed_timeout_hides_artifacts_and_retries_stale_cleanup(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    captured: List[List[str]] = []
    _install_fake_subprocess(
        monkeypatch,
        raise_timeout=True,
        stop_returncode=1,
        kill_returncode=1,
        rm_returncode=1,
        wait_returncode=1,
        container_inspect_returncode=0,
        captured=captured,
    )
    run_dir = tmp_path / "run"
    runner = ra.DockerRunner(workdir=run_dir, cohort_parquet=cohort)

    failed = runner.run(step_id="slow", code="print('hi')\n")

    assert failed.timed_out is True
    assert failed.artefacts == []
    assert failed.outputs_safe_to_collect is False
    assert not (failed.out_dir / "runner_requirements.lock.txt").exists()
    assert not (failed.out_dir / "runner_provenance.json").exists()
    assert failed.script_path.parent == run_dir
    assert failed.runner_log_path is not None
    assert failed.runner_log_path.parent == run_dir
    assert not failed.script_path.is_symlink()
    assert not failed.runner_log_path.is_symlink()
    assert not (failed.cwd / "run.log").exists()
    assert [cmd[1] for cmd in captured[-5:]] == [
        "stop",
        "kill",
        "rm",
        "wait",
        "container",
    ]
    sentinels = list(run_dir.glob(".docker-slow-*.sentinel"))
    assert len(sentinels) == 1
    assert sentinels[0].read_text(encoding="utf-8").startswith("name:easyicu-ra-")
    stale_output = failed.out_dir / "orphan-writer-output.csv"
    stale_output.write_text("unsafe\n", encoding="utf-8")

    retry_commands: List[List[str]] = []
    _install_fake_subprocess(monkeypatch, captured=retry_commands)
    succeeded = runner.run(step_id="slow", code="print('retry')\n")

    assert succeeded.succeeded
    assert [cmd[1] for cmd in retry_commands[:2]] == ["stop", "wait"]
    assert not stale_output.exists()
    assert not list(run_dir.glob(".docker-slow-*.sentinel"))
    assert not list(run_dir.glob(".docker-slow-*.cid"))


def test_stale_sentinel_for_absent_container_does_not_block_retry(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sentinel = run_dir / ".docker-slow-crashed.sentinel"
    sentinel.write_text("name:easyicu-ra-crashed\n", encoding="utf-8")
    captured: List[List[str]] = []
    _install_fake_subprocess(
        monkeypatch,
        stop_returncode=1,
        kill_returncode=1,
        rm_returncode=1,
        wait_returncode=1,
        container_inspect_returncode=1,
        container_inspect_stderr="Error: No such object: easyicu-ra-crashed",
        captured=captured,
    )

    runner = ra.DockerRunner(workdir=run_dir, cohort_parquet=cohort)
    result = runner.run(step_id="slow", code="print('retry')\n")

    assert result.succeeded
    assert [cmd[1] for cmd in captured[:5]] == [
        "stop",
        "kill",
        "rm",
        "wait",
        "container",
    ]
    assert not sentinel.exists()


def test_stale_cleanup_treats_step_id_glob_metacharacters_literally(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    own = run_dir / ".docker-probe*-own.sentinel"
    own.write_text("name:easyicu-ra-own\n", encoding="utf-8")
    unrelated = run_dir / ".docker-probe-other.sentinel"
    unrelated.write_text("name:easyicu-ra-unrelated\n", encoding="utf-8")
    _install_fake_subprocess(monkeypatch)

    runner = ra.DockerRunner(workdir=run_dir, cohort_parquet=cohort)
    result = runner.run(step_id="probe*", code="print('retry')\n")

    assert result.succeeded
    assert not own.exists()
    assert unrelated.exists()


@pytest.mark.parametrize(
    "run_exception", [OSError("docker failed"), KeyboardInterrupt()]
)
def test_host_interruption_preserves_cleanup_sentinel(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_exception: BaseException,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    _install_fake_subprocess(monkeypatch, run_exception=run_exception)
    run_dir = tmp_path / "run"
    runner = ra.DockerRunner(workdir=run_dir, cohort_parquet=cohort)

    with pytest.raises(type(run_exception)):
        runner.run(step_id="interrupted", code="print('hi')\n")

    sentinels = list(run_dir.glob(".docker-interrupted-*.sentinel"))
    assert len(sentinels) == 1
    assert sentinels[0].read_text(encoding="utf-8").startswith("name:easyicu-ra-")


def test_run_replaces_hostile_step_file_and_output_symlinks(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    run_dir = tmp_path / "run"
    external_script = tmp_path / "external-script.py"
    external_script.write_text("do not overwrite\n", encoding="utf-8")
    external_log = tmp_path / "external.log"
    external_log.write_text("do not overwrite\n", encoding="utf-8")
    external_outputs = tmp_path / "external-outputs"
    external_outputs.mkdir()
    external_marker = external_outputs / "keep.txt"
    external_marker.write_text("keep\n", encoding="utf-8")

    def replace_step_paths_with_symlinks() -> None:
        step_dir = run_dir / "steps" / "hostile"
        script_path = step_dir / "analysis.py"
        script_path.unlink()
        script_path.symlink_to(external_script)
        (step_dir / "run.log").symlink_to(external_log)
        shutil.rmtree(step_dir / "outputs")
        (step_dir / "outputs").symlink_to(
            external_outputs,
            target_is_directory=True,
        )

    _install_fake_subprocess(
        monkeypatch,
        run_side_effect=replace_step_paths_with_symlinks,
    )
    runner = ra.DockerRunner(workdir=run_dir, cohort_parquet=cohort)

    result = runner.run(step_id="hostile", code="print('safe')\n")

    assert result.succeeded
    assert result.outputs_safe_to_collect is True
    assert result.script_path.read_text(encoding="utf-8") == "print('safe')\n"
    assert not result.script_path.is_symlink()
    assert result.runner_log_path == result.cwd / "run.log"
    assert not result.runner_log_path.is_symlink()
    assert result.out_dir.is_dir()
    assert not result.out_dir.is_symlink()
    assert external_script.read_text(encoding="utf-8") == "do not overwrite\n"
    assert external_log.read_text(encoding="utf-8") == "do not overwrite\n"
    assert external_marker.read_text(encoding="utf-8") == "keep\n"
    assert all(not path.is_symlink() for path in result.artefacts)


def test_each_run_recreates_the_output_mount_directory(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    captured: List[List[str]] = []
    _install_fake_subprocess(monkeypatch, captured=captured)
    runner = ra.DockerRunner(workdir=tmp_path / "run", cohort_parquet=cohort)

    first = runner.run(step_id="repeat", code="print('first')\n")
    second = runner.run(step_id="repeat", code="print('second')\n")

    assert first.outputs_safe_to_collect is True
    assert second.outputs_safe_to_collect is True
    run_commands = [command for command in captured if "run" in command[:2]]
    output_mounts = [
        (
            command,
            entry.split(",target=", 1)[0].removeprefix("type=bind,source="),
            entry.split(",target=", 1)[1],
        )
        for command in run_commands
        for entry in command
        if ",target=/easyicu-step-output" in entry
    ]
    assert len(output_mounts) == 2
    assert output_mounts[0][1] != output_mounts[1][1]
    assert all("/.outputs-" in source for _command, source, _target in output_mounts)
    assert all(
        target == "/easyicu-step-output" for _command, _source, target in output_mounts
    )
    for command, _source, target in output_mounts:
        assert f"STEP_OUT_DIR={target}" in command
    assert not list((tmp_path / "run" / "steps" / "repeat").glob(".outputs-*"))


def test_each_run_executes_an_immutable_attempt_owned_script(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    captured: List[List[str]] = []
    _install_fake_subprocess(monkeypatch, captured=captured)
    runner = ra.DockerRunner(workdir=tmp_path / "run", cohort_parquet=cohort)

    runner.run(step_id="repeat", code="print('first')\n")
    runner.run(step_id="repeat", code="print('second')\n")

    run_commands = [command for command in captured if "run" in command[:2]]
    script_sources = [
        entry.split(",target=", 1)[0].removeprefix("type=bind,source=")
        for command in run_commands
        for entry in command
        if ",target=/easyicu-analysis.py" in entry
    ]
    assert len(script_sources) == 2
    assert script_sources[0] != script_sources[1]
    assert all(source.endswith(".analysis.py") for source in script_sources)
    assert all(
        not any(
            entry.startswith("type=bind") and ",target=/easyicu-run/" in entry
            for entry in command
        )
        for command in run_commands
    )
    assert not list((tmp_path / "run").glob(".docker-repeat-*.analysis.py"))


def test_run_rejects_symlinked_step_directory(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    run_dir = tmp_path / "run"
    steps_dir = run_dir / "steps"
    steps_dir.mkdir(parents=True)
    external_step = tmp_path / "external-step"
    external_step.mkdir()
    marker = external_step / "keep.txt"
    marker.write_text("keep\n", encoding="utf-8")
    (steps_dir / "hostile").symlink_to(external_step, target_is_directory=True)
    runner = ra.DockerRunner(workdir=run_dir, cohort_parquet=cohort)

    with pytest.raises(RuntimeError, match="requires a real directory"):
        runner.run(step_id="hostile", code="print('no')\n")

    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_pull_image_invoked_when_requested(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    captured: List[List[str]] = []
    _install_fake_subprocess(
        monkeypatch,
        proc=_FakeProc(stdout="ok\n"),
        captured=captured,
    )

    runner = ra.DockerRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort,
        image="img:1",
        pull_image=True,
    )
    runner.run(step_id="s", code="print('x')\n")

    # Pull is followed by image inspection, environment capture, then run.
    assert captured[0][:2] == [runner.docker_executable, "pull"]
    assert captured[0][2] == "img:1"
    assert captured[1][1:3] == ["image", "inspect"]
    assert "importlib.metadata" in " ".join(captured[2])
    assert captured[3][:2] == [runner.docker_executable, "run"]


def test_pull_precedes_and_binds_authority_image_identity(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    calls: List[List[str]] = []
    pulled = {"value": False}

    def fake_run(cmd, *args, **kwargs):
        del args, kwargs
        calls.append(list(cmd))
        if len(cmd) >= 2 and cmd[1] == "pull":
            pulled["value"] = True
            return _FakeProc()
        if len(cmd) >= 3 and cmd[1:3] == ["image", "inspect"]:
            image = "b" if pulled["value"] else "a"
            return _FakeProc(
                stdout=json.dumps({"Id": "sha256:" + image * 64, "RepoDigests": []})
            )
        if "importlib.metadata" in " ".join(cmd):
            return _FakeProc(
                stdout=(
                    "numpy==2\npandas==2\nscipy==1\nmatplotlib==3\n"
                    "statsmodels==0.14\nscikit-learn==1\npyarrow==23\n"
                )
            )
        raise AssertionError(cmd)

    import easyicu.research_agent.execution.runner as runner_module

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    runner = ra.DockerRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort,
        image="img:latest",
        pull_image=True,
    )

    authority = runner.authority_identity_sha256
    provenance, _requirements = runner._capture_runtime_provenance()

    assert calls[0][1] == "pull"
    assert calls[1][1:3] == ["image", "inspect"]
    assert provenance["image_id"] == "sha256:" + "b" * 64
    assert authority
    assert sum(call[1] == "pull" for call in calls) == 1


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------


def test_pipeline_runner_kind_docker_constructs_docker_runner(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort_path = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)

    pipe = ra.ResearchAgentPipeline(
        workdir=tmp_path / "ra",
        runner_kind="docker",
        runner_image="img:smoke",
    )
    runner = pipe._build_runner(
        run_dir=tmp_path / "ra" / "run", cohort_path=cohort_path
    )
    assert isinstance(runner, ra.DockerRunner)
    assert runner.manages_output_cleanup is True
    assert runner.image == "img:smoke"
    assert runner.network == "none"


def test_pipeline_runner_kind_default_uses_probed_safe_backend(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import easyicu.research_agent.pipeline as pipeline_mod

    cohort_path = _make_cohort(tmp_path)
    monkeypatch.setattr(
        pipeline_mod,
        "select_safe_runner_kind",
        lambda **_kwargs: "subprocess",
    )
    pipe = ra.ResearchAgentPipeline(workdir=tmp_path / "ra")
    runner = pipe._build_runner(
        run_dir=tmp_path / "ra" / "run", cohort_path=cohort_path
    )
    assert isinstance(runner, ra.CodeRunner)


def test_pipeline_runner_factory_overrides_kind(
    ra,
    tmp_path: Path,
):
    import easyicu.research_agent.execution.method_capabilities as method_capabilities

    cohort_path = _make_cohort(tmp_path)
    seen: Dict[str, Any] = {}

    def fake_factory(*, workdir, cohort_parquet, timeout_seconds, **kw):
        seen["workdir"] = workdir
        seen["cohort_parquet"] = cohort_parquet
        seen["timeout_seconds"] = timeout_seconds
        return "sentinel-runner"

    pipe = ra.ResearchAgentPipeline(
        workdir=tmp_path / "ra",
        runner_kind="docker",  # ignored when runner_factory is set
        runner_factory=fake_factory,
    )
    method_capabilities.set_runtime_capability_snapshot_provider(
        lambda: {"docker-only-capability"}
    )
    runner = pipe._build_runner(
        run_dir=tmp_path / "ra" / "run", cohort_path=cohort_path
    )
    assert runner == "sentinel-runner"
    assert seen["cohort_parquet"] == cohort_path
    assert method_capabilities.runtime_capability_snapshot() is None


def test_pipeline_unknown_runner_kind_raises(ra, tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown runner_kind"):
        ra.ResearchAgentPipeline(
            workdir=tmp_path / "ra",
            runner_kind="firecracker",
        )


def test_code_runner_validates_baseline_capabilities(ra, tmp_path: Path):
    cohort = _make_cohort(tmp_path)
    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort,
        allow_unsafe_host_fallback=True,
    )

    snapshot = runner.validate_runtime_capabilities()

    assert {"numpy", "pandas", "sklearn", "statsmodels"} <= set(snapshot)


def test_docker_runner_public_capability_preflight_uses_verified_image(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    _install_fake_subprocess(monkeypatch)
    runner = ra.DockerRunner(workdir=tmp_path / "run", cohort_parquet=cohort)

    snapshot = runner.validate_runtime_capabilities()

    assert {"numpy", "pandas", "sklearn", "statsmodels"} <= set(snapshot)


def test_pipeline_preflight_requires_custom_runner_capability_contract(
    ra,
    tmp_path: Path,
):
    cohort = _make_cohort(tmp_path)
    pipe = ra.ResearchAgentPipeline(
        workdir=tmp_path / "ra",
        runner_factory=lambda **_kwargs: object(),
    )

    with pytest.raises(RuntimeError, match="validate_runtime_capabilities"):
        pipe._preflight_execution_runtime(
            run_dir=tmp_path / "ra" / "run",
            cohort_path=cohort,
            target_outcome="death",
        )


def test_pipeline_runtime_preflight_precedes_plan_invocation(ra):
    pipeline_source = inspect.getsource(ra.ResearchAgentPipeline.run)

    assert pipeline_source.index("self._preflight_execution_runtime(") < (
        pipeline_source.index("def _plan_invoker()")
    )


def test_runtime_preflight_failure_spends_zero_llm_calls(
    ra,
    synthetic_cohort,
    tmp_path: Path,
):
    class CountingLLM(ra.MockLLMClient):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def complete(self, messages, **kwargs):
            self.calls += 1
            return super().complete(messages, **kwargs)

    class MissingRuntime:
        def validate_runtime_capabilities(self):
            raise RuntimeError("required runtime package is missing")

    llm = CountingLLM()
    pipe = ra.ResearchAgentPipeline(
        workdir=tmp_path / "ra",
        llm=llm,
        runner_factory=lambda **_kwargs: MissingRuntime(),
    )

    with pytest.raises(RuntimeError, match="required runtime package is missing"):
        pipe.run(
            question="Is exposure associated with mortality?",
            cohort=synthetic_cohort,
            cohort_name="runtime_preflight",
            database="synthetic",
            target_outcome="death",
        )

    assert llm.calls == 0


def test_runner_rebuild_restores_preflighted_capability_snapshot(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.execution.method_capabilities import (
        runtime_capability_snapshot,
    )

    cohort = _make_cohort(tmp_path)
    pipe = ra.ResearchAgentPipeline(
        workdir=tmp_path / "ra",
        runner_kind="subprocess",
    )
    expected = pipe._preflight_execution_runtime(
        run_dir=tmp_path / "ra" / "preflight",
        cohort_path=cohort,
        target_outcome="death",
    )

    pipe._build_runner(
        run_dir=tmp_path / "ra" / "execution",
        cohort_path=cohort,
        target_outcome="death",
    )

    assert runtime_capability_snapshot() == frozenset(expected)


def test_docker_runner_rebuild_reuses_preflight_receipt_without_second_capture(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    captured: List[List[str]] = []
    _force_docker_present(monkeypatch)
    _install_fake_subprocess(monkeypatch, captured=captured)
    pipe = ra.ResearchAgentPipeline(
        workdir=tmp_path / "ra",
        runner_kind="docker",
    )

    pipe._preflight_execution_runtime(
        run_dir=tmp_path / "ra" / "preflight",
        cohort_path=cohort,
        target_outcome="death",
    )
    rebuilt = pipe._build_runner(
        run_dir=tmp_path / "ra" / "execution",
        cohort_path=cohort,
        target_outcome="death",
    )
    rebuilt.runtime_capability_report()

    capture_calls = [cmd for cmd in captured if "importlib.metadata" in " ".join(cmd)]
    assert len(capture_calls) == 1


def test_docker_runner_rebuild_rejects_image_change_after_preflight(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    image_id = ["sha256:" + "a" * 64]

    def fake_run(cmd, *args, **kwargs):
        if len(cmd) >= 3 and cmd[1:3] == ["image", "inspect"]:
            return _FakeProc(stdout=json.dumps({"Id": image_id[0], "RepoDigests": []}))
        if "importlib.metadata" in " ".join(cmd):
            return _FakeProc(
                stdout=(
                    "numpy==2.0.0\npandas==2.2.0\nscipy==1.14.0\n"
                    "matplotlib==3.9.0\nstatsmodels==0.14.0\n"
                    "scikit-learn==1.5.0\npyarrow==23.0.0\n"
                )
            )
        return _FakeProc(stdout="ok\n")

    import easyicu.research_agent.execution.runner as runner_mod

    monkeypatch.setattr(runner_mod.subprocess, "run", fake_run)
    pipe = ra.ResearchAgentPipeline(
        workdir=tmp_path / "ra",
        runner_kind="docker",
    )
    pipe._preflight_execution_runtime(
        run_dir=tmp_path / "ra" / "preflight",
        cohort_path=cohort,
        target_outcome="death",
    )
    image_id[0] = "sha256:" + "c" * 64

    with pytest.raises(RuntimeError, match="changed after preflight"):
        pipe._build_runner(
            run_dir=tmp_path / "ra" / "execution",
            cohort_path=cohort,
            target_outcome="death",
        )


def test_build_command_mounts_explicit_external_path_env_read_only(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    trajectory = tmp_path / "external" / "trajectory.parquet"
    trajectory.parent.mkdir()
    trajectory.write_bytes(b"trajectory")
    _force_docker_present(monkeypatch)
    runner = ra.DockerRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort,
        extra_env={"TRAJECTORY_PARQUET": str(trajectory)},
    )
    _, script_path, out_dir = runner.prepare_step_dir("trajectory")
    script_path.write_text("print('ok')\n", encoding="utf-8")

    cmd = runner.build_command(
        step_id="trajectory", script_path=script_path, out_dir=out_dir
    )
    env_pairs = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-e"]
    env_dict = dict(pair.split("=", 1) for pair in env_pairs)

    assert env_dict["TRAJECTORY_PARQUET"].startswith("/easyicu-inputs/")
    assert str(trajectory) not in env_dict["TRAJECTORY_PARQUET"]
    assert any(
        f"source={trajectory}" in token
        and f"target={env_dict['TRAJECTORY_PARQUET']}" in token
        and "readonly" in token
        for token in cmd
    )


def test_docker_runner_rejects_step_id_path_escape(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    runner = ra.DockerRunner(workdir=tmp_path / "run", cohort_parquet=cohort)

    with pytest.raises(ValueError, match="single safe path component"):
        runner.run(step_id="../escape", code="print('no')\n")
