"""CodeRunner provenance details."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def test_runner_records_real_duration(ra, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)

    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        timeout_seconds=10,
    )
    result = runner.run(
        step_id="duration_probe",
        code="from pathlib import Path\nimport os\nPath(os.environ['STEP_OUT_DIR'], 'ok.txt').write_text('ok')\n",
    )

    assert result.succeeded
    assert 0 <= result.duration_seconds < 10
    log_text = (result.cwd / "run.log").read_text(encoding="utf-8")
    assert "duration_seconds:" in log_text


def test_runner_build_command_defaults_to_network_isolation(ra, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)

    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
    )
    cmd = runner.build_command(script_path=Path("/tmp/demo.py"))
    joined = " ".join(cmd)
    assert "demo.py" in joined
    if "sandbox-exec" in joined:
        assert "deny network" in joined


def test_runner_retries_without_unshare_when_linux_namespace_is_unavailable(
    ra,
    tmp_path: Path,
    monkeypatch,
):
    import easyicu.research_agent.runner as runner_mod

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)

    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        timeout_seconds=10,
    )

    calls: list[list[str]] = []

    def _fake_run(cmd, *, cwd, env, capture_output, text, timeout, encoding, errors):
        calls.append(list(cmd))
        if cmd[0] == "unshare":
            return SimpleNamespace(
                stdout="",
                stderr="unshare: unshare failed: Operation not permitted",
                returncode=1,
            )
        Path(env["STEP_OUT_DIR"], "ok.txt").write_text("ok", encoding="utf-8")
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(
        runner,
        "build_command",
        lambda *, script_path: ["unshare", "-n", "--", "python", str(script_path)],
    )
    monkeypatch.setattr(runner_mod.sys, "platform", "linux")
    monkeypatch.setattr(runner_mod.subprocess, "run", _fake_run)

    result = runner.run(
        step_id="linux_unshare_fallback",
        code="print('ok')\n",
    )

    assert result.succeeded
    assert len(calls) == 2
    assert calls[0][0] == "unshare"
    assert calls[1][0].endswith("python")
    assert any(p.name == "ok.txt" for p in result.artefacts)
    assert "retrying without Linux network namespace isolation" in result.stderr


def test_runner_forces_single_thread_env_for_sandboxed_numeric_stacks(
    ra,
    tmp_path: Path,
    monkeypatch,
):
    import easyicu.research_agent.runner as runner_mod

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)

    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("MKL_NUM_THREADS", "8")
    captured_env = {}

    def _fake_run(cmd, *, cwd, env, capture_output, text, timeout, encoding, errors):
        captured_env.update(env)
        Path(env["STEP_OUT_DIR"], "ok.txt").write_text("ok", encoding="utf-8")
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(runner_mod.subprocess, "run", _fake_run)

    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        timeout_seconds=10,
    )
    result = runner.run(step_id="thread_env", code="print('ok')\n")

    assert result.succeeded
    assert captured_env["OMP_NUM_THREADS"] == "1"
    assert captured_env["MKL_NUM_THREADS"] == "1"
    assert captured_env["OPENBLAS_NUM_THREADS"] == "1"
    assert captured_env["JOBLIB_MULTIPROCESSING"] == "0"


def test_runner_retries_without_macos_sandbox_when_openmp_shm_is_blocked(
    ra,
    tmp_path: Path,
    monkeypatch,
):
    import easyicu.research_agent.runner as runner_mod

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)

    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        timeout_seconds=10,
    )

    calls: list[list[str]] = []

    def _fake_run(cmd, *, cwd, env, capture_output, text, timeout, encoding, errors):
        calls.append(list(cmd))
        if cmd[0] == "sandbox-exec":
            return SimpleNamespace(
                stdout="",
                stderr="OMP: Error #179: Function Can't open SHM2 failed:",
                returncode=-6,
            )
        Path(env["STEP_OUT_DIR"], "ok.txt").write_text("ok", encoding="utf-8")
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(
        runner,
        "build_command",
        lambda *, script_path: [
            "sandbox-exec",
            "-p",
            "(deny network*)",
            "python",
            str(script_path),
        ],
    )
    monkeypatch.setattr(runner_mod.sys, "platform", "darwin")
    monkeypatch.setattr(runner_mod.subprocess, "run", _fake_run)

    result = runner.run(step_id="macos_omp_fallback", code="print('ok')\n")

    assert result.succeeded
    assert len(calls) == 2
    assert calls[0][0] == "sandbox-exec"
    assert calls[1][0].endswith("python")
    assert any(p.name == "ok.txt" for p in result.artefacts)
    assert "retrying without sandbox-exec" in result.stderr


def test_runner_retries_without_macos_sandbox_when_stdio_is_blocked(
    ra,
    tmp_path: Path,
    monkeypatch,
):
    import easyicu.research_agent.runner as runner_mod

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)

    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        timeout_seconds=10,
    )

    calls: list[list[str]] = []

    def _fake_run(cmd, *, cwd, env, capture_output, text, timeout, encoding, errors):
        calls.append(list(cmd))
        if cmd[0] == "sandbox-exec":
            return SimpleNamespace(
                stdout="",
                stderr=(
                    "Fatal Python error: init_sys_streams: can't initialize sys standard streams\n"
                    "OSError: [Errno 9] Bad file descriptor"
                ),
                returncode=1,
            )
        Path(env["STEP_OUT_DIR"], "ok.txt").write_text("ok", encoding="utf-8")
        return SimpleNamespace(stdout="ok\n", stderr="", returncode=0)

    monkeypatch.setattr(
        runner,
        "build_command",
        lambda *, script_path: [
            "sandbox-exec",
            "-p",
            "(deny network*)",
            "python",
            str(script_path),
        ],
    )
    monkeypatch.setattr(runner_mod.sys, "platform", "darwin")
    monkeypatch.setattr(runner_mod.subprocess, "run", _fake_run)

    result = runner.run(step_id="macos_stdio_fallback", code="print('ok')\n")

    assert result.succeeded
    assert len(calls) == 2
    assert calls[0][0] == "sandbox-exec"
    assert calls[1][0].endswith("python")
    assert any(p.name == "ok.txt" for p in result.artefacts)
    assert "prevented Python stdio initialisation" in result.stderr
