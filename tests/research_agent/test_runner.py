"""CodeRunner provenance details."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def test_run_result_has_one_dependency_neutral_contract_owner():
    import easyicu.research_agent as research_agent
    from easyicu.research_agent.contracts.runtime import RunResult as ContractRunResult
    from easyicu.research_agent.execution.runner import RunResult as RunnerRunResult

    assert research_agent.RunResult is ContractRunResult
    assert RunnerRunResult is ContractRunResult


@pytest.fixture
def unauthorized_host_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Withdraw the suite-wide host-fallback grant for one test.

    ``tests/conftest.py`` sets ``EASYICU_ALLOW_UNSAFE_HOST_FALLBACK=1`` for
    every research-agent test so test-owned scripts still run when an outer
    CI/Codex sandbox prevents ``sandbox-exec`` from nesting.  That grant is
    right for tests that merely need generated code to execute, and wrong for
    the tests below, which *own* the fail-closed isolation boundary: inheriting
    it makes them assert a guard while the harness has already authorized
    exactly what the guard exists to refuse, so they fail whether or not the
    production default is intact.
    """

    monkeypatch.delenv("EASYICU_ALLOW_UNSAFE_HOST_FALLBACK", raising=False)


def _is_python_executable(command: str) -> bool:
    return Path(command).name.startswith("python")


def _skip_if_outer_macos_sandbox_denied(result) -> None:
    from easyicu.research_agent.execution.runner import (
        macos_sandbox_permission_denied,
    )

    if (
        result.effective_isolation == "macos_sandbox_exec"
        and not result.succeeded
        and macos_sandbox_permission_denied(result.stderr)
    ):
        pytest.skip(
            "isolation_backend_unavailable: outer macOS sandbox denied "
            "CodeRunner target execution"
        )


def test_output_cleanup_never_follows_untrusted_symlinks(tmp_path: Path):
    from easyicu.research_agent.pipeline import _clear_output_dir

    victim_dir = tmp_path / "victim"
    victim_dir.mkdir()
    victim_file = victim_dir / "must_survive.txt"
    victim_file.write_text("preserve", encoding="utf-8")
    step_dir = tmp_path / "run" / "steps" / "hostile"
    step_dir.mkdir(parents=True)
    out_dir = step_dir / "outputs"
    out_dir.symlink_to(victim_dir, target_is_directory=True)

    _clear_output_dir(out_dir)

    assert out_dir.is_dir()
    assert not out_dir.is_symlink()
    assert victim_file.read_text(encoding="utf-8") == "preserve"

    child_link = out_dir / "hostile_child"
    child_link.symlink_to(victim_dir, target_is_directory=True)
    _clear_output_dir(out_dir)

    assert not child_link.exists()
    assert victim_file.read_text(encoding="utf-8") == "preserve"


def test_runner_records_real_duration(ra, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)

    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        timeout_seconds=10,
        allow_unsafe_host_fallback=True,
    )
    result = runner.run(
        step_id="duration_probe",
        code="from pathlib import Path\nimport os\nPath(os.environ['STEP_OUT_DIR'], 'ok.txt').write_text('ok')\n",
    )

    assert result.succeeded
    assert 0 <= result.duration_seconds < 10
    log_text = (result.cwd / "run.log").read_text(encoding="utf-8")
    assert "duration_seconds:" in log_text


def test_code_runner_exposes_current_cohort_row_count(ra, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2, 3], "death": [0, 1, 0]}).to_parquet(
        cohort_path,
        index=False,
    )
    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        timeout_seconds=10,
        network_policy="allow",
        allow_unsafe_host_fallback=True,
    )

    result = runner.run(
        step_id="cohort_rows",
        code=(
            "import os\n"
            "from pathlib import Path\n"
            "Path(os.environ['STEP_OUT_DIR'], 'rows.txt').write_text("
            "os.environ['EASYICU_COHORT_ROWS'])\n"
        ),
    )

    assert result.succeeded
    assert (result.out_dir / "rows.txt").read_text(encoding="utf-8") == "3"


def test_code_runner_never_collects_generated_output_symlinks(ra, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)
    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        timeout_seconds=10,
        network_policy="allow",
        allow_unsafe_host_fallback=True,
    )
    result = runner.run(
        step_id="symlink_output",
        code=(
            "import os\n"
            "from pathlib import Path\n"
            "Path(os.environ['STEP_OUT_DIR'], 'forged.parquet').symlink_to("
            "os.environ['COHORT_PARQUET'])\n"
        ),
    )

    assert result.succeeded
    assert all(path.name != "forged.parquet" for path in result.artefacts)
    assert not (result.out_dir / "forged.parquet").exists()
    assert cohort_path.is_file()


def test_code_runner_control_writes_never_follow_planted_symlinks(ra, tmp_path: Path):
    # The macOS sandbox lets generated code write anywhere under the step dir,
    # and the step dir is reused across repair attempts. A prior attempt can
    # therefore leave analysis.py / run.log as symlinks pointing at a host file
    # outside the sandbox. The host must overwrite the *link* with a fresh
    # regular file, never write through it onto the victim.
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)

    victim = tmp_path / "victim_outside_sandbox.txt"
    victim.write_text("must survive", encoding="utf-8")

    workdir = tmp_path / "run"
    step_dir = workdir / "steps" / "hostile"
    step_dir.mkdir(parents=True)
    planted_script = step_dir / "analysis.py"
    planted_log = step_dir / "run.log"
    planted_script.symlink_to(victim)
    planted_log.symlink_to(victim)

    runner = ra.CodeRunner(
        workdir=workdir,
        cohort_parquet=cohort_path,
        timeout_seconds=10,
        allow_unsafe_host_fallback=True,
    )
    result = runner.run(
        step_id="hostile",
        code=(
            "import os\n"
            "from pathlib import Path\n"
            "Path(os.environ['STEP_OUT_DIR'], 'ok.txt').write_text('ok')\n"
        ),
    )

    assert result.succeeded
    # The victim host file was never written through either planted link.
    assert victim.read_text(encoding="utf-8") == "must survive"
    # Both control files are now real, single-hardlink regular files.
    for control in (planted_script, planted_log):
        assert control.is_file() and not control.is_symlink()
        assert control.stat().st_nlink == 1
    # And they hold their real content, not the victim's.
    assert "STEP_OUT_DIR" in planted_script.read_text(encoding="utf-8")
    assert "duration_seconds:" in planted_log.read_text(encoding="utf-8")


@pytest.mark.parametrize("unsafe_value", ["false", "0", "no", 0, 1])
def test_code_runner_rejects_non_bool_unsafe_host_fallback(
    ra, tmp_path: Path, unsafe_value
):
    # bool("false") is True: a quoted config value must not silently enable
    # unsafe host execution. Only True/False/None are accepted.
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1]}).to_parquet(cohort_path, index=False)
    with pytest.raises(TypeError, match="allow_unsafe_host_fallback"):
        ra.CodeRunner(
            workdir=tmp_path / "run",
            cohort_parquet=cohort_path,
            allow_unsafe_host_fallback=unsafe_value,
        )


def test_code_runner_authority_binds_extra_inputs_and_isolation(ra, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1]}).to_parquet(cohort_path, index=False)
    supplemental = tmp_path / "supplemental.csv"
    supplemental.write_text("x\n1\n", encoding="utf-8")
    first = ra.CodeRunner(
        workdir=tmp_path / "run-a",
        cohort_parquet=cohort_path,
        python_executable=sys.executable,
        extra_env={"SUPPLEMENTAL": str(supplemental)},
        allow_unsafe_host_fallback=False,
    )
    first_identity = first.authority_identity_sha256

    supplemental.write_text("x\n2\n", encoding="utf-8")
    changed_input = ra.CodeRunner(
        workdir=tmp_path / "run-b",
        cohort_parquet=cohort_path,
        python_executable=sys.executable,
        extra_env={"SUPPLEMENTAL": str(supplemental)},
        allow_unsafe_host_fallback=False,
    )
    changed_policy = ra.CodeRunner(
        workdir=tmp_path / "run-c",
        cohort_parquet=cohort_path,
        python_executable=sys.executable,
        extra_env={"SUPPLEMENTAL": str(supplemental)},
        allow_unsafe_host_fallback=True,
    )
    input_dir = tmp_path / "input-dir"
    input_dir.mkdir()
    (input_dir / "value.txt").write_text("one", encoding="utf-8")
    directory_before = ra.CodeRunner(
        workdir=tmp_path / "run-d",
        cohort_parquet=cohort_path,
        python_executable=sys.executable,
        extra_env={"INPUT_DIR": str(input_dir)},
    ).authority_identity_sha256
    (input_dir / "value.txt").write_text("two", encoding="utf-8")
    directory_after = ra.CodeRunner(
        workdir=tmp_path / "run-e",
        cohort_parquet=cohort_path,
        python_executable=sys.executable,
        extra_env={"INPUT_DIR": str(input_dir)},
    ).authority_identity_sha256

    assert first_identity != changed_input.authority_identity_sha256
    assert changed_input.authority_identity_sha256 != (
        changed_policy.authority_identity_sha256
    )
    assert directory_before != directory_after


def test_code_runner_authority_probe_failure_is_fail_closed(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1]}).to_parquet(cohort_path, index=False)
    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        python_executable=sys.executable,
    )
    import easyicu.research_agent.execution.runner as runner_module

    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="probe failed",
        ),
    )

    with pytest.raises(RuntimeError, match="authority probe failed"):
        _ = runner.authority_identity_sha256


def test_code_runner_exposes_run_level_artifact_env(ra, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)
    run_dir = tmp_path / "run"

    runner = ra.CodeRunner(
        workdir=run_dir,
        cohort_parquet=cohort_path,
        timeout_seconds=10,
        allow_unsafe_host_fallback=True,
    )
    result = runner.run(
        step_id="env_probe",
        code=(
            "import json, os\n"
            "from pathlib import Path\n"
            "payload = {k: os.environ.get(k) for k in [\n"
            "  'EASYICU_RUN_DIR', 'EASYICU_EVIDENCE_DIR', 'EASYICU_MANIFEST_PARTIAL',\n"
            "  'EASYICU_RUN_ARTIFACT_AUTHORITY_SNAPSHOT'\n"
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
    assert payload["EASYICU_RUN_ARTIFACT_AUTHORITY_SNAPSHOT"] is None
    assert not (
        run_dir / "steps" / "env_probe" / ".run_artifact_authority_snapshot.json"
    ).exists()


def test_code_runner_exposes_digest_bound_current_authority_snapshot(
    ra, tmp_path: Path
):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest_partial.json").write_text(
        json.dumps(
            {
                "checkpoint_sequence": 7,
                "per_step_records": [{"step_id": "01_primary", "status": "ok"}],
            }
        ),
        encoding="utf-8",
    )
    runner = ra.CodeRunner(
        workdir=run_dir,
        cohort_parquet=cohort_path,
        timeout_seconds=10,
        allow_unsafe_host_fallback=True,
    )

    result = runner.run(
        step_id="authority_probe",
        code=(
            "import hashlib, json, os\n"
            "from pathlib import Path\n"
            "from easyicu.research_agent.execution.runners.deterministic_robustness import (\n"
            "    _run_robustness_preflight_from_env,\n"
            ")\n"
            "path = Path(os.environ['EASYICU_RUN_ARTIFACT_AUTHORITY_SNAPSHOT'])\n"
            "raw = path.read_bytes()\n"
            "payload = {\n"
            "  'path': str(path),\n"
            "  'expected': os.environ['EASYICU_RUN_ARTIFACT_AUTHORITY_SNAPSHOT_SHA256'],\n"
            "  'observed': hashlib.sha256(raw).hexdigest(),\n"
            "  'snapshot': json.loads(raw),\n"
            "}\n"
            "Path(os.environ['STEP_OUT_DIR'], 'authority.json').write_text(json.dumps(payload))\n"
        ),
    )

    assert result.succeeded
    payload = json.loads(
        (result.out_dir / "authority.json").read_text(encoding="utf-8")
    )
    assert payload["expected"] == payload["observed"]
    assert payload["snapshot"]["checkpoint_sequence"] == 7
    assert payload["snapshot"]["authority"]["per_step_records"] == [
        {"step_id": "01_primary", "status": "ok"}
    ]
    assert set(payload["snapshot"]["authority"]) == {
        "run_id",
        "checkpoint_sequence",
        "per_step_records",
        "evidence",
    }
    assert Path(payload["path"]).parent == run_dir / "steps" / "authority_probe"


def test_code_runner_exposes_exact_resolved_inputs_manifest(ra, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1]}).to_parquet(cohort_path, index=False)
    run_dir = tmp_path / "run"
    manifest = run_dir / "resolved_inputs" / "consume.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('{"schema_version":"1.0","inputs":{}}\n', encoding="utf-8")
    runner = ra.CodeRunner(
        workdir=run_dir,
        cohort_parquet=cohort_path,
        timeout_seconds=10,
        allow_unsafe_host_fallback=True,
    )

    result = runner.run(
        step_id="consume",
        resolved_inputs_path=manifest,
        code=(
            "import os\n"
            "from pathlib import Path\n"
            "Path(os.environ['STEP_OUT_DIR'], 'path.txt').write_text("
            "os.environ['EASYICU_RESOLVED_INPUTS_JSON'])\n"
        ),
    )

    assert result.succeeded
    assert (result.out_dir / "path.txt").read_text(encoding="utf-8") == str(
        manifest.resolve()
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
        assert "(allow file-read*)" not in joined
        assert "(allow file-write*)" not in joined
        assert '(subpath "/opt/homebrew")' not in joined
        assert '(subpath "/usr/local")' not in joined
        step_root = runner.workdir / "steps" / "probe"
        profile = runner._macos_sandbox_profile(script_path=step_root / "analysis.py")
        assert f'(allow file-write* (subpath "{step_root}"))' in profile
        assert f'(allow file-write* (subpath "{runner.workdir}"))' not in profile


def test_code_runner_resolves_symlinked_python_parent(ra, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)
    runtime_dir = Path(sys.executable).parent.resolve(strict=True)
    runtime_link = tmp_path / "runtime-link"
    runtime_link.symlink_to(runtime_dir, target_is_directory=True)
    linked_python = runtime_link / Path(sys.executable).name

    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        python_executable=str(linked_python),
    )

    assert runner.python_executable == str(runtime_dir / Path(sys.executable).name)
    command = runner.build_command(script_path=tmp_path / "analysis.py")
    expected_launch_target = (
        str(Path(runner.python_executable).resolve())
        if Path(command[0]).name == "sandbox-exec"
        else runner.python_executable
    )
    assert command[-2] == expected_launch_target
    assert str(linked_python) not in command


def test_code_runner_preserves_bare_python_command(ra, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)

    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        python_executable="custom-python",
    )

    assert runner.python_executable == "custom-python"
    command = runner.build_command(script_path=tmp_path / "analysis.py")
    assert command[-2] == "custom-python"


def test_macos_sandbox_resolves_final_venv_symlink_without_leaving_venv(
    ra, tmp_path: Path, monkeypatch
):
    """The sandbox target is real while the selected venv remains authoritative."""

    import easyicu.research_agent.execution.runner as runner_mod

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(
        cohort_path, index=False
    )
    venv_prefix = tmp_path / "selected-venv"
    venv_bin = venv_prefix / "bin"
    venv_bin.mkdir(parents=True)
    (venv_prefix / "pyvenv.cfg").write_text(
        f"home = {Path(sys.executable).resolve().parent}\n",
        encoding="utf-8",
    )
    configured_python = venv_bin / "python"
    configured_python.symlink_to(Path(sys.executable).resolve())
    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        python_executable=str(configured_python),
        allow_unsafe_host_fallback=False,
    )
    calls: list[tuple[list[str], dict[str, str]]] = []

    def _fake_run(cmd, *, cwd, env, timeout):
        calls.append((list(cmd), dict(env)))
        Path(env["STEP_OUT_DIR"], "ok.txt").write_text("ok", encoding="utf-8")
        return SimpleNamespace(stdout="ok\n", stderr="", returncode=0)

    monkeypatch.setattr(runner_mod.sys, "platform", "darwin")
    monkeypatch.setattr(
        runner_mod.shutil,
        "which",
        lambda name: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None,
    )
    monkeypatch.setattr(
        runner_mod, "_run_capturing_with_descendant_reaping", _fake_run
    )

    result = runner.run(step_id="venv_symlink", code="print('ok')\n")

    assert result.succeeded
    assert result.effective_isolation == "macos_sandbox_exec"
    assert result.isolation_degraded is False
    assert len(calls) == 1
    command, child_env = calls[0]
    assert command[-2] == str(Path(sys.executable).resolve())
    assert command[-2] != str(configured_python)
    assert child_env["__PYVENV_LAUNCHER__"] == str(configured_python)
    assert child_env["PYTHONHOME"] == str(Path(sys.executable).resolve().parent.parent)
    assert f'(subpath "{venv_prefix}")' in command[2]
    assert f'(subpath "{Path(sys.executable).resolve().parent.parent}")' in command[2]


@pytest.mark.parametrize("reserved_key", ["PYTHONHOME", "__PYVENV_LAUNCHER__"])
def test_code_runner_rejects_host_owned_python_launch_env(
    ra, tmp_path: Path, reserved_key: str
):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(
        cohort_path, index=False
    )

    with pytest.raises(ValueError, match=reserved_key):
        ra.CodeRunner(
            workdir=tmp_path / "run",
            cohort_parquet=cohort_path,
            extra_env={reserved_key: "/attacker/runtime"},
        )


def test_code_runner_scrubs_secrets_and_reports_filesystem_degradation(
    ra, tmp_path: Path, monkeypatch, unauthorized_host_fallback
):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("file-secret", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")

    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        timeout_seconds=10,
    )
    result = runner.run(
        step_id="secret_probe",
        code=(
            "import json, os\n"
            "from pathlib import Path\n"
            f"outside = Path({str(outside)!r})\n"
            "try:\n"
            "    outside_value = outside.read_text()\n"
            "except Exception:\n"
            "    outside_value = None\n"
            "payload = {\n"
            "    'api_key': os.environ.get('OPENAI_API_KEY'),\n"
            "    'outside_value': outside_value,\n"
            "}\n"
            "Path(os.environ['STEP_OUT_DIR'], 'probe.json').write_text(json.dumps(payload))\n"
        ),
    )

    if result.effective_isolation == "blocked_fail_closed":
        assert result.returncode == 126
        assert not (result.out_dir / "probe.json").exists()
        return
    _skip_if_outer_macos_sandbox_denied(result)
    assert result.succeeded
    payload = json.loads((result.out_dir / "probe.json").read_text(encoding="utf-8"))
    assert payload["api_key"] is None
    assert result.isolation_degraded is False
    assert result.effective_isolation == "macos_sandbox_exec"
    assert payload["outside_value"] is None


def test_code_runner_default_does_not_retry_unavailable_sandbox_on_host(
    ra, tmp_path: Path, monkeypatch, unauthorized_host_fallback
):
    import easyicu.research_agent.execution.runner as runner_mod

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)
    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return SimpleNamespace(
            stdout="",
            stderr="sandbox-exec: sandbox_apply: Operation not permitted",
            returncode=71,
        )

    runner = ra.CodeRunner(workdir=tmp_path / "run", cohort_parquet=cohort_path)
    monkeypatch.setattr(
        runner,
        "build_command",
        lambda *, script_path: [
            "sandbox-exec",
            "-p",
            "(deny default)",
            "python",
            str(script_path),
        ],
    )
    monkeypatch.setattr(runner_mod.sys, "platform", "darwin")
    monkeypatch.setattr(runner_mod, "_run_capturing_with_descendant_reaping", _fake_run)

    result = runner.run(step_id="sandbox_abort", code="print('must not retry')\n")

    assert result.returncode == 71
    assert len(calls) == 2
    assert calls[0][0] == calls[1][0] == "sandbox-exec"
    assert "-c" in calls[1]
    assert result.effective_isolation == "macos_sandbox_exec"
    assert result.isolation_degraded is False
    assert "fail-closed policy" in result.stderr
    assert result.runner_failure_code is not None
    assert result.runner_failure_code.value == "isolation_backend_unavailable"


def test_code_runner_default_blocks_direct_host_execution(
    ra, tmp_path: Path, monkeypatch, unauthorized_host_fallback
):
    import easyicu.research_agent.execution.runner as runner_mod

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)
    runner = ra.CodeRunner(workdir=tmp_path / "run", cohort_parquet=cohort_path)
    monkeypatch.setattr(
        runner,
        "build_command",
        lambda *, script_path: [runner.python_executable, str(script_path)],
    )
    monkeypatch.setattr(
        runner_mod,
        "_run_capturing_with_descendant_reaping",
        lambda *args, **kwargs: pytest.fail("generated code must not run on host"),
    )

    result = runner.run(step_id="blocked_host", code="print('must not run')\n")

    assert result.returncode == 126
    assert result.effective_isolation == "blocked_fail_closed"
    assert result.isolation_degraded is False


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("sandbox-exec") is None,
    reason="requires the macOS sandbox-exec backend",
)
def test_macos_sandbox_executes_resolved_python_symlink(
    ra, tmp_path: Path, unauthorized_host_fallback
):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)
    runtime_dir = Path(sys.executable).parent.resolve(strict=True)
    runtime_link = tmp_path / "runtime-link"
    runtime_link.symlink_to(runtime_dir, target_is_directory=True)
    linked_python = runtime_link / Path(sys.executable).name
    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        python_executable=str(linked_python),
    )

    result = runner.run(
        step_id="symlinked_python",
        code=(
            "import os\n"
            "from pathlib import Path\n"
            "Path(os.environ['STEP_OUT_DIR'], 'ok.txt').write_text('ok')\n"
        ),
    )

    _skip_if_outer_macos_sandbox_denied(result)
    assert result.succeeded, result.stderr
    assert result.effective_isolation == "macos_sandbox_exec"
    assert (result.out_dir / "ok.txt").read_text(encoding="utf-8") == "ok"


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("sandbox-exec") is None,
    reason="requires the macOS sandbox-exec backend",
)
def test_macos_sandbox_imports_pandas_and_easyicu_but_confines_files(
    ra, tmp_path: Path, unauthorized_host_fallback
):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    run_dir = tmp_path / "run"
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    protected = evidence_dir / "protected.txt"
    protected.write_text("original", encoding="utf-8")
    runner = ra.CodeRunner(workdir=run_dir, cohort_parquet=cohort_path)

    result = runner.run(
        step_id="real_import_confinement",
        code=(
            "import json, os\n"
            "from pathlib import Path\n"
            "import pandas\n"
            "import easyicu\n"
            f"outside = Path({str(outside)!r})\n"
            f"protected = Path({str(protected)!r})\n"
            "def attempt(action):\n"
            "    try:\n"
            "        return action()\n"
            "    except Exception:\n"
            "        return None\n"
            "payload = {\n"
            "  'pandas': pandas.__version__,\n"
            "  'outside': attempt(outside.read_text),\n"
            "  'protected_write': attempt(lambda: protected.write_text('tampered')),\n"
            "}\n"
            "Path(os.environ['STEP_OUT_DIR'], 'probe.json').write_text(json.dumps(payload))\n"
        ),
    )

    _skip_if_outer_macos_sandbox_denied(result)
    assert result.succeeded, result.stderr
    assert result.effective_isolation == "macos_sandbox_exec"
    payload = json.loads((result.out_dir / "probe.json").read_text(encoding="utf-8"))
    assert payload["pandas"]
    assert payload["outside"] is None
    assert payload["protected_write"] is None
    assert protected.read_text(encoding="utf-8") == "original"


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("sandbox-exec") is None,
    reason="requires the macOS sandbox-exec backend",
)
def test_macos_sandbox_resolves_bound_run_file_without_exposing_sibling(
    ra, tmp_path: Path, unauthorized_host_fallback
):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(
        cohort_path, index=False
    )
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    run_dir = tmp_path / "run"
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    bound = evidence_dir / "bound.txt"
    bound.write_text("bound", encoding="utf-8")
    runner = ra.CodeRunner(workdir=run_dir, cohort_parquet=cohort_path)

    result = runner.run(
        step_id="realpath_confinement",
        code=(
            "import json, os\n"
            "from pathlib import Path\n"
            "root = Path(os.environ['EASYICU_RUN_DIR']).resolve(strict=True)\n"
            "bound = (root / 'evidence' / 'bound.txt').resolve(strict=True)\n"
            f"outside = Path({str(outside)!r})\n"
            "try:\n"
            "    outside_value = outside.read_text()\n"
            "except Exception:\n"
            "    outside_value = None\n"
            "payload = {\n"
            "    'bound': bound.read_text(),\n"
            "    'contained': bound.is_relative_to(root),\n"
            "    'outside': outside_value,\n"
            "}\n"
            "Path(os.environ['STEP_OUT_DIR'], 'probe.json').write_text(json.dumps(payload))\n"
        ),
    )

    _skip_if_outer_macos_sandbox_denied(result)
    assert result.succeeded, result.stderr
    assert result.effective_isolation == "macos_sandbox_exec"
    payload = json.loads((result.out_dir / "probe.json").read_text(encoding="utf-8"))
    assert payload == {"bound": "bound", "contained": True, "outside": None}


def test_pipeline_runner_receives_target_outcome_env(ra, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "endpoint_x": [0]}).to_parquet(
        cohort_path, index=False
    )
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "work",
        enable_memory=False,
        runner_kind="subprocess",
    )

    runner = pipeline._build_runner(
        run_dir=tmp_path / "run",
        cohort_path=cohort_path,
        target_outcome="endpoint_x",
    )

    assert runner.extra_env["OUTCOME_COL"] == "endpoint_x"


def test_pipeline_runner_does_not_discover_unstaged_trajectory_sibling(
    ra, tmp_path: Path
):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "endpoint_x": [0]}).to_parquet(
        cohort_path, index=False
    )
    universe_path = tmp_path / "universe.parquet"
    pd.DataFrame({"stay_id": [1]}).to_parquet(universe_path, index=False)
    # sibling trajectory next to the universe
    (tmp_path / "universe_trajectory.parquet").write_bytes(b"x")

    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "work",
        enable_memory=False,
        runner_kind="subprocess",
    )
    runner = pipeline._build_runner(
        run_dir=tmp_path / "run",
        cohort_path=cohort_path,
        target_outcome="endpoint_x",
        universe_path=universe_path,
    )
    assert "TRAJECTORY_PARQUET" not in runner.extra_env


def test_materialise_cohort_does_not_copy_unverified_trajectory_sibling(
    ra, tmp_path: Path
):
    # Cohort staging has no authority to discover/copy a mutable sibling.
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

    assert not (run_dir / "cohort_trajectory.parquet").exists()


def test_pipeline_runner_no_trajectory_env_when_sibling_absent(ra, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "endpoint_x": [0]}).to_parquet(
        cohort_path, index=False
    )
    universe_path = tmp_path / "universe.parquet"
    pd.DataFrame({"stay_id": [1]}).to_parquet(universe_path, index=False)

    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "work",
        enable_memory=False,
        runner_kind="subprocess",
    )
    runner = pipeline._build_runner(
        run_dir=tmp_path / "run",
        cohort_path=cohort_path,
        target_outcome="endpoint_x",
        universe_path=universe_path,
    )
    assert "TRAJECTORY_PARQUET" not in runner.extra_env


def test_pipeline_runner_rejects_explicit_outcome_env_override(ra, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "endpoint_x": [0]}).to_parquet(
        cohort_path, index=False
    )
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "work",
        enable_memory=False,
        runner_kwargs={"extra_env": {"OUTCOME_COL": "manual_endpoint"}},
    )

    with pytest.raises(ValueError, match="OUTCOME_COL"):
        pipeline._build_runner(
            run_dir=tmp_path / "run",
            cohort_path=cohort_path,
            target_outcome="endpoint_x",
        )


def test_pipeline_runner_rejects_universe_authority_override(ra, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "endpoint_x": [0]}).to_parquet(
        cohort_path, index=False
    )
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "work",
        enable_memory=False,
        runner_kwargs={
            "extra_env": {"EASYICU_UNIVERSE_PARQUET": str(tmp_path / "forged")}
        },
    )

    with pytest.raises(ValueError, match="EASYICU_UNIVERSE_PARQUET"):
        pipeline._build_runner(
            run_dir=tmp_path / "run",
            cohort_path=cohort_path,
            universe_path=cohort_path,
        )


def test_pipeline_runner_rejects_unsealed_typed_trajectory(ra, tmp_path: Path):
    from easyicu.research_agent.intake.materialized_metadata import (
        MaterializedCohortAuthorityRef,
        MaterializedMetadataError,
    )

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "endpoint_x": [0]}).to_parquet(
        cohort_path, index=False
    )
    (tmp_path / "cohort_trajectory.parquet").write_bytes(b"unsealed")
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path / "work", enable_memory=False)

    with pytest.raises(MaterializedMetadataError, match="exact sealed authority"):
        pipeline._build_runner(
            run_dir=tmp_path / "run",
            cohort_path=cohort_path,
            universe_path=cohort_path,
            universe_is_typed=True,
            universe_authority_ref=MaterializedCohortAuthorityRef(
                file="materialized_authority.json",
                sha256="1" * 64,
                size=1,
            ),
            trajectory_path=tmp_path / "cohort_trajectory.parquet",
        )


def test_code_runner_rejects_host_owned_output_override(ra, tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "endpoint_x": [0]}).to_parquet(
        cohort_path, index=False
    )

    with pytest.raises(ValueError, match="STEP_OUT_DIR"):
        ra.CodeRunner(
            workdir=tmp_path / "run",
            cohort_parquet=cohort_path,
            extra_env={"STEP_OUT_DIR": str(tmp_path / "forged")},
        )


@pytest.mark.parametrize(
    "key",
    ["COHORT_PARQUET=forged", "BAD-KEY", "9INVALID", "BAD\nKEY"],
)
def test_code_runner_rejects_invalid_extra_env_keys(
    ra,
    tmp_path: Path,
    key: str,
):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1]}).to_parquet(cohort_path, index=False)

    with pytest.raises(ValueError, match="invalid environment key"):
        ra.CodeRunner(
            workdir=tmp_path / "run",
            cohort_parquet=cohort_path,
            extra_env={key: "yes"},
        )


def test_runner_retries_without_unshare_when_linux_namespace_is_unavailable(
    ra,
    tmp_path: Path,
    monkeypatch,
):
    import easyicu.research_agent.execution.runner as runner_mod

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)

    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        timeout_seconds=10,
        allow_unsafe_host_fallback=True,
    )

    calls: list[list[str]] = []

    def _fake_run(cmd, *, cwd, env, timeout):
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
    monkeypatch.setattr(runner_mod, "_run_capturing_with_descendant_reaping", _fake_run)

    result = runner.run(
        step_id="linux_unshare_fallback",
        code="print('ok')\n",
    )

    assert result.succeeded
    assert len(calls) == 3
    assert calls[0][0] == calls[1][0] == "unshare"
    assert "-c" in calls[1]
    assert _is_python_executable(calls[2][0])
    assert any(p.name == "ok.txt" for p in result.artefacts)
    assert "unshare network isolation unavailable" in result.stderr
    assert result.isolation_degraded is True
    assert result.effective_isolation == "host_subprocess"
    assert "unshare" in (result.isolation_degradation_reason or "")


@pytest.mark.parametrize(
    ("diagnostic", "allow_unsafe_host_fallback"),
    [
        (
            "Traceback (most recent call last):\n"
            "  File 'script.py', line 1, in <module>\n"
            "NameError: name 'model_frame' is not defined",
            False,
        ),
        (
            "Traceback (most recent call last):\n"
            "  File 'script.py', line 1, in <module>\n"
            "OSError: [Errno 9] Bad file descriptor",
            True,
        ),
    ],
)
def test_sandboxed_child_code_error_remains_coder_repairable(
    ra,
    tmp_path: Path,
    monkeypatch,
    unauthorized_host_fallback,
    diagnostic: str,
    allow_unsafe_host_fallback: bool,
):
    """A working sandbox must not relabel the child's traceback as host failure."""

    import easyicu.research_agent.execution.runner as runner_mod
    from easyicu.research_agent.execution.failure_classification import (
        classify_runtime_failure,
    )

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(
        cohort_path, index=False
    )
    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        timeout_seconds=10,
        allow_unsafe_host_fallback=allow_unsafe_host_fallback,
    )

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
    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return SimpleNamespace(stdout="", stderr=diagnostic, returncode=1)

    monkeypatch.setattr(
        runner_mod, "_run_capturing_with_descendant_reaping", _fake_run
    )

    result = runner.run(step_id="child_name_error", code="print(model_frame)\n")
    run_log = result.runner_log_path.read_text(encoding="utf-8")

    assert len(calls) == 1
    assert "[CodeRunner] isolation backend failed" not in result.stderr
    assert diagnostic.splitlines()[-1] in result.stderr
    assert (
        classify_runtime_failure(
            run_log=run_log,
            timed_out=result.timed_out,
            step_id=result.step_id,
            returncode=result.returncode,
            runner_failure_code=result.runner_failure_code,
        )
        is None
    )


def test_child_diagnostic_cannot_authorize_unsafe_host_fallback(
    ra,
    tmp_path: Path,
    monkeypatch,
):
    """A host-owned probe, not generated stderr, decides sandbox fallback."""

    import easyicu.research_agent.execution.runner as runner_mod

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1]}).to_parquet(cohort_path, index=False)
    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        timeout_seconds=10,
        allow_unsafe_host_fallback=True,
    )
    calls: list[list[str]] = []

    def _fake_run(cmd, *, cwd, env, timeout):
        calls.append(list(cmd))
        if "-c" in cmd:
            return SimpleNamespace(
                stdout="easyicu-isolation-probe-ok\n", stderr="", returncode=0
            )
        return SimpleNamespace(
            stdout="",
            stderr="OMP: Error #179: Function Can't open SHM2 failed:",
            returncode=1,
        )

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
    monkeypatch.setattr(
        runner_mod, "_run_capturing_with_descendant_reaping", _fake_run
    )

    result = runner.run(step_id="child_spoof", code="raise RuntimeError('boom')\n")

    assert len(calls) == 2
    assert calls[0][0] == calls[1][0] == "sandbox-exec"
    assert result.returncode == 1
    assert result.isolation_degraded is False
    assert result.runner_failure_code is None


def test_an_unprobeable_command_keeps_the_child_failure_instead_of_crashing(
    ra,
    tmp_path: Path,
    monkeypatch,
):
    """An argv with no interpreter payload cannot be turned into a probe.

    ``subprocess`` raises ``IndexError`` for an empty argv, and the probe's
    ``(OSError, TimeoutExpired)`` handler does not catch it -- so building an
    empty probe command would replace a retained child failure with a crash.
    """

    import easyicu.research_agent.execution.runner as runner_mod

    assert (
        runner_mod._trusted_isolation_probe_command(
            ["sandbox-exec"], failure_kind="macos_permission"
        )
        is None
    )

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1]}).to_parquet(cohort_path, index=False)
    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        timeout_seconds=10,
        allow_unsafe_host_fallback=True,
    )
    calls: list[list[str]] = []

    def _fake_run(cmd, *, cwd, env, timeout):
        calls.append(list(cmd))
        if not cmd:  # pragma: no cover - the fix must prevent this call
            raise IndexError("list index out of range")
        return SimpleNamespace(
            stdout="",
            stderr="sandbox-exec: execvp() failed: Operation not permitted",
            returncode=1,
        )

    monkeypatch.setattr(
        runner, "build_command", lambda *, script_path: ["sandbox-exec"]
    )
    monkeypatch.setattr(runner_mod.sys, "platform", "darwin")
    monkeypatch.setattr(
        runner_mod, "_run_capturing_with_descendant_reaping", _fake_run
    )

    result = runner.run(step_id="short_argv", code="raise RuntimeError('boom')\n")

    # Exactly one call: the child. No probe was attempted, and nothing crashed.
    assert len(calls) == 1
    assert result.returncode == 1
    assert result.isolation_degraded is False
    assert result.runner_failure_code is None
    assert "trusted isolation probe did not complete" in result.stderr


def test_child_diagnostic_plus_probe_timeout_keeps_original_repair_route(
    ra,
    tmp_path: Path,
    monkeypatch,
):
    import easyicu.research_agent.execution.runner as runner_mod

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1]}).to_parquet(cohort_path, index=False)
    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        timeout_seconds=10,
        allow_unsafe_host_fallback=True,
    )
    calls: list[list[str]] = []

    def _fake_run(cmd, *, cwd, env, timeout):
        calls.append(list(cmd))
        if "-c" in cmd:
            raise subprocess.TimeoutExpired(cmd, timeout)
        return SimpleNamespace(
            stdout="",
            stderr="OMP: Error #179: Function Can't open SHM2 failed:",
            returncode=1,
        )

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
    monkeypatch.setattr(
        runner_mod, "_run_capturing_with_descendant_reaping", _fake_run
    )

    result = runner.run(step_id="probe_timeout", code="raise RuntimeError('boom')\n")

    assert len(calls) == 2
    assert result.returncode == 1
    assert result.timed_out is False
    assert result.isolation_degraded is False
    assert result.runner_failure_code is None
    assert "original child failure retained" in result.stderr


def test_runner_forces_single_thread_env_for_sandboxed_numeric_stacks(
    ra,
    tmp_path: Path,
    monkeypatch,
):
    import easyicu.research_agent.execution.runner as runner_mod

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)

    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("MKL_NUM_THREADS", "8")
    captured_env = {}

    def _fake_run(cmd, *, cwd, env, timeout):
        captured_env.update(env)
        Path(env["STEP_OUT_DIR"], "ok.txt").write_text("ok", encoding="utf-8")
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(runner_mod, "_run_capturing_with_descendant_reaping", _fake_run)

    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        timeout_seconds=10,
        allow_unsafe_host_fallback=True,
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
    import easyicu.research_agent.execution.runner as runner_mod

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)

    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        timeout_seconds=10,
        allow_unsafe_host_fallback=True,
    )

    calls: list[list[str]] = []

    def _fake_run(cmd, *, cwd, env, timeout):
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
    monkeypatch.setattr(runner_mod, "_run_capturing_with_descendant_reaping", _fake_run)

    result = runner.run(step_id="macos_omp_fallback", code="print('ok')\n")

    assert result.succeeded
    assert len(calls) == 3
    assert calls[0][0] == calls[1][0] == "sandbox-exec"
    assert "-c" in calls[1]
    assert _is_python_executable(calls[2][0])
    assert any(p.name == "ok.txt" for p in result.artefacts)
    assert "blocked numeric runtime shared memory" in result.stderr
    assert result.isolation_degraded is True
    assert result.effective_isolation == "host_subprocess"
    assert "shared memory" in (result.isolation_degradation_reason or "")


def test_runner_retries_without_macos_sandbox_when_profile_apply_is_denied(
    ra,
    tmp_path: Path,
    monkeypatch,
):
    import easyicu.research_agent.execution.runner as runner_mod

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)

    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        timeout_seconds=10,
        allow_unsafe_host_fallback=True,
    )

    calls: list[list[str]] = []
    captured_env = {}

    def _fake_run(cmd, *, cwd, env, timeout):
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
    monkeypatch.setattr(runner_mod, "_run_capturing_with_descendant_reaping", _fake_run)

    result = runner.run(step_id="macos_sandbox_apply_fallback", code="print('ok')\n")

    assert result.succeeded
    assert len(calls) == 3
    assert calls[0][0] == calls[1][0] == "sandbox-exec"
    assert "-c" in calls[1]
    assert _is_python_executable(calls[2][0])
    assert captured_env["MPLCONFIGDIR"].endswith(".matplotlib")
    assert any(p.name == "ok.txt" for p in result.artefacts)
    assert "could not apply its profile" in result.stderr
    assert result.isolation_degraded is True
    assert result.effective_isolation == "host_subprocess"
    assert "profile application" in (result.isolation_degradation_reason or "")


def test_runner_retries_without_macos_sandbox_when_target_exec_is_denied(
    ra,
    tmp_path: Path,
    monkeypatch,
):
    """Explicit development fallback also covers nested-sandbox execvp denial."""

    import easyicu.research_agent.execution.runner as runner_mod

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)
    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        timeout_seconds=10,
        allow_unsafe_host_fallback=True,
    )
    calls: list[list[str]] = []

    def _fake_run(cmd, *, cwd, env, timeout):
        calls.append(list(cmd))
        if cmd[0] == "sandbox-exec":
            return SimpleNamespace(
                stdout="",
                stderr=(
                    "sandbox-exec: execvp() of '/tmp/.venv/bin/python' failed: "
                    "Operation not permitted"
                ),
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
    monkeypatch.setattr(runner_mod, "_run_capturing_with_descendant_reaping", _fake_run)

    result = runner.run(step_id="macos_execvp_fallback", code="print('ok')\n")

    assert result.succeeded
    assert len(calls) == 3
    assert calls[0][0] == calls[1][0] == "sandbox-exec"
    assert "-c" in calls[1]
    assert _is_python_executable(calls[2][0])
    assert result.isolation_degraded is True
    assert result.effective_isolation == "host_subprocess"
    assert "could not apply its profile" in result.stderr


def test_runner_retries_without_macos_sandbox_when_stdio_is_blocked(
    ra,
    tmp_path: Path,
    monkeypatch,
):
    import easyicu.research_agent.execution.runner as runner_mod

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort_path, index=False)

    runner = ra.CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=cohort_path,
        timeout_seconds=10,
        allow_unsafe_host_fallback=True,
    )

    calls: list[list[str]] = []

    def _fake_run(cmd, *, cwd, env, timeout):
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
    monkeypatch.setattr(runner_mod, "_run_capturing_with_descendant_reaping", _fake_run)

    result = runner.run(step_id="macos_stdio_fallback", code="print('ok')\n")

    assert result.succeeded
    assert len(calls) == 3
    assert calls[0][0] == calls[1][0] == "sandbox-exec"
    assert "-c" in calls[1]
    assert _is_python_executable(calls[2][0])
    assert any(p.name == "ok.txt" for p in result.artefacts)
    assert "prevented Python stdio initialisation" in result.stderr
    assert result.isolation_degraded is True
    assert result.effective_isolation == "host_subprocess"
    assert "stdio" in (result.isolation_degradation_reason or "")
