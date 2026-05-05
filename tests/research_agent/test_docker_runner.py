"""DockerRunner contract tests (T3.1).

Real Docker is not available in CI / the dev sandbox, so these tests
mock ``subprocess.run`` and ``shutil.which`` to verify:

1. ``shutil.which("docker")`` is consulted at construction; missing
   binary raises a clean ``FileNotFoundError``.
2. The composed argv contains the right safety knobs:
   ``--rm``, ``--init``, ``--network=none``, RO cohort mount, RW
   step mount, env injection, image trailer, and ``python -u``.
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

import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    captured: Optional[List[List[str]]] = None,
) -> None:
    """Replace runner.subprocess.run with a deterministic stub."""
    proc = proc or _FakeProc(stdout="hello-from-container\n")

    def fake_run(cmd, *args, **kwargs):
        if captured is not None:
            captured.append(list(cmd))
        if raise_timeout:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 0))
        # Mimic the side-effect a real container would have so the
        # caller's ``out_dir.iterdir()`` finds something.
        return proc

    # Patch the module-level subprocess (not the global one) so other
    # tests are unaffected.
    import easyicu.research_agent.runner as runner_mod
    monkeypatch.setattr(runner_mod.subprocess, "run", fake_run)


def _force_docker_present(monkeypatch: pytest.MonkeyPatch, fake_path: str = "/usr/bin/docker") -> None:
    import easyicu.research_agent.runner as runner_mod
    monkeypatch.setattr(runner_mod.shutil, "which", lambda _name: fake_path)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_missing_docker_binary_raises(ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cohort = _make_cohort(tmp_path)
    import easyicu.research_agent.runner as runner_mod
    monkeypatch.setattr(runner_mod.shutil, "which", lambda _n: None)

    with pytest.raises(FileNotFoundError, match="not found on PATH"):
        ra.DockerRunner(workdir=tmp_path / "run", cohort_parquet=cohort)


def test_constructor_resolves_docker_via_which(ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch, "/opt/local/bin/docker")
    runner = ra.DockerRunner(workdir=tmp_path / "run", cohort_parquet=cohort)
    assert runner.docker_executable == "/opt/local/bin/docker"


def test_constructor_honours_environment_image(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    monkeypatch.setenv("EASYICU_RUNNER_IMAGE", "company/easyicu-ra:1.4")
    runner = ra.DockerRunner(workdir=tmp_path / "run", cohort_parquet=cohort)
    assert runner.image == "company/easyicu-ra:1.4"


# ---------------------------------------------------------------------------
# build_command shape
# ---------------------------------------------------------------------------


def test_build_command_has_safety_knobs(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    runner = ra.DockerRunner(
        workdir=tmp_path / "run", cohort_parquet=cohort,
        image="my/image:latest",
    )
    step_dir, script_path, out_dir = runner.prepare_step_dir("step_x")
    script_path.write_text("print('hi')\n", encoding="utf-8")
    cmd = runner.build_command(
        step_id="step_x", script_path=script_path, out_dir=out_dir,
    )
    joined = " ".join(cmd)

    assert cmd[0] == runner.docker_executable
    assert cmd[1] == "run"
    assert "--rm" in cmd
    assert "--init" in cmd
    assert "--network=none" in cmd
    assert "--workdir=/workspace" in cmd
    assert "my/image:latest" in cmd
    # Image must appear before "python" trailer.
    assert cmd.index("my/image:latest") < cmd.index("python")
    # The command end is `python -u /workspace/analysis.py`.
    assert cmd[-3:] == ["python", "-u", "/workspace/analysis.py"]
    # Cohort mount: read-only, container path /cohort.parquet.
    assert any(
        "type=bind" in s and "readonly" in s and "target=/cohort.parquet" in s
        for s in cmd
    ), f"cohort mount missing in {joined}"
    # Step dir mount RW at /workspace.
    assert any(
        "type=bind" in s
        and "target=/workspace" in s
        and "readonly" not in s
        for s in cmd
    ), f"step mount missing in {joined}"
    # Env injection.
    assert "-e" in cmd
    env_pairs = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-e"]
    env_dict = dict(p.split("=", 1) for p in env_pairs)
    assert env_dict["COHORT_PARQUET"] == "/cohort.parquet"
    assert env_dict["STEP_OUT_DIR"] == "/workspace/outputs"
    assert env_dict["MPLBACKEND"] == "Agg"


def test_build_command_passes_through_advanced_flags(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
        extra_mounts=[(str(extra_mount_src), "/extra", "ro")],
        extra_env={"PUBMED_API_KEY": "abc"},
    )
    step_dir, script_path, out_dir = runner.prepare_step_dir("step_y")
    script_path.write_text("print('hi')\n", encoding="utf-8")
    cmd = runner.build_command(
        step_id="step_y", script_path=script_path, out_dir=out_dir,
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


# ---------------------------------------------------------------------------
# run() integration with mocked subprocess
# ---------------------------------------------------------------------------


def test_run_invokes_subprocess_and_writes_log(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
        workdir=tmp_path / "run", cohort_parquet=cohort, image="img:0",
    )
    result = runner.run(step_id="probe", code="print('hi')\n")

    assert len(captured) == 1, "subprocess.run should be called exactly once"
    cmd = captured[0]
    assert cmd[0] == runner.docker_executable and cmd[1] == "run"

    assert result.succeeded
    assert result.returncode == 0
    assert "container-said-hi" in result.stdout
    log_text = (result.cwd / "run.log").read_text(encoding="utf-8")
    assert "DockerRunner" in log_text
    assert "img:0" in log_text
    # Script persisted to disk before run.
    assert result.script_path.read_text(encoding="utf-8") == "print('hi')\n"


def test_run_handles_timeout(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    cohort = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)
    _install_fake_subprocess(monkeypatch, raise_timeout=True)

    runner = ra.DockerRunner(
        workdir=tmp_path / "run", cohort_parquet=cohort,
        timeout_seconds=0.001,
    )
    result = runner.run(step_id="slow", code="print('hi')\n")

    assert result.timed_out is True
    assert result.returncode == -1
    assert not result.succeeded
    log_text = (result.cwd / "run.log").read_text(encoding="utf-8")
    assert "timed_out: True" in log_text
    assert "DockerRunner] timed out" in result.stderr


def test_pull_image_invoked_when_requested(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
        workdir=tmp_path / "run", cohort_parquet=cohort,
        image="img:1", pull_image=True,
    )
    runner.run(step_id="s", code="print('x')\n")

    # The first invocation should be `docker pull img:1`.
    assert captured[0][:2] == [runner.docker_executable, "pull"]
    assert captured[0][2] == "img:1"
    # Then the actual `docker run` call.
    assert captured[1][:2] == [runner.docker_executable, "run"]


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------


def test_pipeline_runner_kind_docker_constructs_docker_runner(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    cohort_path = _make_cohort(tmp_path)
    _force_docker_present(monkeypatch)

    pipe = ra.ResearchAgentPipeline(
        workdir=tmp_path / "ra",
        runner_kind="docker",
        runner_image="img:smoke",
    )
    runner = pipe._build_runner(run_dir=tmp_path / "ra" / "run", cohort_path=cohort_path)
    assert isinstance(runner, ra.DockerRunner)
    assert runner.image == "img:smoke"
    assert runner.network == "none"


def test_pipeline_runner_kind_default_is_subprocess(
    ra, tmp_path: Path,
):
    cohort_path = _make_cohort(tmp_path)
    pipe = ra.ResearchAgentPipeline(workdir=tmp_path / "ra")
    runner = pipe._build_runner(run_dir=tmp_path / "ra" / "run", cohort_path=cohort_path)
    assert isinstance(runner, ra.CodeRunner)


def test_pipeline_runner_factory_overrides_kind(
    ra, tmp_path: Path,
):
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
    runner = pipe._build_runner(run_dir=tmp_path / "ra" / "run", cohort_path=cohort_path)
    assert runner == "sentinel-runner"
    assert seen["cohort_parquet"] == cohort_path


def test_pipeline_unknown_runner_kind_raises(ra, tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown runner_kind"):
        ra.ResearchAgentPipeline(
            workdir=tmp_path / "ra", runner_kind="firecracker",
        )
