"""CodeRunner provenance details."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def _is_python_executable(command: str) -> bool:
    return Path(command).name.startswith("python")


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


def test_code_runner_exposes_run_level_artifact_env(ra, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)
    run_dir = tmp_path / "run"

    runner = ra.CodeRunner(
        workdir=run_dir,
        cohort_parquet=cohort_path,
        timeout_seconds=10,
    )
    result = runner.run(
        step_id="env_probe",
        code=(
            "import json, os\n"
            "from pathlib import Path\n"
            "payload = {k: os.environ.get(k) for k in [\n"
            "  'EASYICU_RUN_DIR', 'EASYICU_EVIDENCE_DIR', 'EASYICU_MANIFEST_PARTIAL'\n"
            "]}\n"
            "Path(os.environ['STEP_OUT_DIR'], 'env.json').write_text(json.dumps(payload))\n"
        ),
    )

    assert result.succeeded
    payload = json.loads((result.out_dir / "env.json").read_text(encoding="utf-8"))
    assert payload["EASYICU_RUN_DIR"] == str(run_dir.resolve())
    assert payload["EASYICU_EVIDENCE_DIR"] == str((run_dir / "evidence").resolve())
    assert payload["EASYICU_MANIFEST_PARTIAL"] == str(
        (run_dir / "manifest_partial.json").resolve()
    )


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


def test_pipeline_runner_receives_target_outcome_env(ra, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "endpoint_x": [0]}).to_parquet(
        cohort_path, index=False
    )
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "work",
        enable_memory=False,
    )

    runner = pipeline._build_runner(
        run_dir=tmp_path / "run",
        cohort_path=cohort_path,
        target_outcome="endpoint_x",
    )

    assert runner.extra_env["OUTCOME_COL"] == "endpoint_x"


def test_pipeline_runner_auto_discovers_trajectory_sibling(ra, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "endpoint_x": [0]}).to_parquet(
        cohort_path, index=False
    )
    universe_path = tmp_path / "universe.parquet"
    pd.DataFrame({"stay_id": [1]}).to_parquet(universe_path, index=False)
    # sibling trajectory next to the universe
    (tmp_path / "universe_trajectory.parquet").write_bytes(b"x")

    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path / "work", enable_memory=False)
    runner = pipeline._build_runner(
        run_dir=tmp_path / "run",
        cohort_path=cohort_path,
        target_outcome="endpoint_x",
        universe_path=universe_path,
    )
    assert runner.extra_env["TRAJECTORY_PARQUET"] == str(
        tmp_path / "universe_trajectory.parquet"
    )


def test_materialise_cohort_carries_trajectory_sibling_then_runner_exposes_it(
    ra, tmp_path: Path
):
    # End-to-end of the staging fix: a universe parquet with a sibling
    # trajectory, staged into the run_dir, must carry the trajectory so the
    # runner's auto-discovery exposes TRAJECTORY_PARQUET.
    universe_dir = tmp_path / "universe"
    universe_dir.mkdir()
    src = universe_dir / "discovery_universe.parquet"
    pd.DataFrame({"stay_id": [1, 2]}).to_parquet(src, index=False)
    pd.DataFrame(
        {"stay_id": [1], "charttime": [3.0], "concept": ["map"], "value_num": [60.0]}
    ).to_parquet(universe_dir / "discovery_universe_trajectory.parquet", index=False)

    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path / "work", enable_memory=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cohort_path = pipeline._materialise_cohort(src, run_dir)

    assert (run_dir / "cohort_trajectory.parquet").exists()
    runner = pipeline._build_runner(
        run_dir=run_dir,
        cohort_path=cohort_path,
        target_outcome="aki",
        universe_path=cohort_path,  # how pipeline_execute wires it
    )
    assert runner.extra_env["TRAJECTORY_PARQUET"] == str(
        run_dir / "cohort_trajectory.parquet"
    )


def test_pipeline_runner_no_trajectory_env_when_sibling_absent(ra, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "endpoint_x": [0]}).to_parquet(
        cohort_path, index=False
    )
    universe_path = tmp_path / "universe.parquet"
    pd.DataFrame({"stay_id": [1]}).to_parquet(universe_path, index=False)

    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path / "work", enable_memory=False)
    runner = pipeline._build_runner(
        run_dir=tmp_path / "run",
        cohort_path=cohort_path,
        target_outcome="endpoint_x",
        universe_path=universe_path,
    )
    assert "TRAJECTORY_PARQUET" not in runner.extra_env


def test_pipeline_runner_preserves_explicit_outcome_env_override(ra, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "endpoint_x": [0]}).to_parquet(
        cohort_path, index=False
    )
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "work",
        enable_memory=False,
        runner_kwargs={"extra_env": {"OUTCOME_COL": "manual_endpoint"}},
    )

    runner = pipeline._build_runner(
        run_dir=tmp_path / "run",
        cohort_path=cohort_path,
        target_outcome="endpoint_x",
    )

    assert runner.extra_env["OUTCOME_COL"] == "manual_endpoint"


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
    assert _is_python_executable(calls[1][0])
    assert any(p.name == "ok.txt" for p in result.artefacts)
    assert "retrying without Linux network namespace isolation" in result.stderr
    assert result.isolation_degraded is True
    assert result.effective_isolation == "host_subprocess"
    assert "unshare" in (result.isolation_degradation_reason or "")


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
    assert _is_python_executable(calls[1][0])
    assert any(p.name == "ok.txt" for p in result.artefacts)
    assert "retrying without sandbox-exec" in result.stderr
    assert result.isolation_degraded is True
    assert result.effective_isolation == "host_subprocess"
    assert "shared memory" in (result.isolation_degradation_reason or "")


def test_runner_retries_without_macos_sandbox_when_profile_apply_is_denied(
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
    captured_env = {}

    def _fake_run(cmd, *, cwd, env, capture_output, text, timeout, encoding, errors):
        calls.append(list(cmd))
        captured_env.update(env)
        if cmd[0] == "sandbox-exec":
            return SimpleNamespace(
                stdout="",
                stderr="sandbox-exec: sandbox_apply: Operation not permitted",
                returncode=71,
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

    result = runner.run(step_id="macos_sandbox_apply_fallback", code="print('ok')\n")

    assert result.succeeded
    assert len(calls) == 2
    assert calls[0][0] == "sandbox-exec"
    assert _is_python_executable(calls[1][0])
    assert captured_env["MPLCONFIGDIR"].endswith(".matplotlib")
    assert any(p.name == "ok.txt" for p in result.artefacts)
    assert "could not apply its profile" in result.stderr
    assert result.isolation_degraded is True
    assert result.effective_isolation == "host_subprocess"
    assert "profile application" in (result.isolation_degradation_reason or "")


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
    assert _is_python_executable(calls[1][0])
    assert any(p.name == "ok.txt" for p in result.artefacts)
    assert "prevented Python stdio initialisation" in result.stderr
    assert result.isolation_degraded is True
    assert result.effective_isolation == "host_subprocess"
    assert "stdio" in (result.isolation_degradation_reason or "")
