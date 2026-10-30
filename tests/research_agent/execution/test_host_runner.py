"""Trusted desktop execution never grants host access to a coder candidate."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pandas as pd
import pytest


@pytest.fixture
def cohort(tmp_path):
    path = tmp_path / "cohort.parquet"
    pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}).to_parquet(path)
    return path


def test_host_runner_executes_only_explicit_host_code(cohort, tmp_path):
    from easyicu.research_agent.execution.host_runner import HostRunner
    from easyicu.research_agent.execution.step_execution import CodeOrigin

    runner = HostRunner(workdir=tmp_path, cohort_parquet=cohort)
    code = "import os\nfrom pathlib import Path\nPath(os.environ['STEP_OUT_DIR'], 'answer.txt').write_text('6')\n"
    for origin in (None, "model_generated", "repair", "recovered_capsule", "unknown"):
        result = runner.run(step_id="refused", code=code, code_origin=origin)
        assert not result.succeeded
        assert result.runner_failure_code.value == "host_runner_refuses_model_code"
        assert not (tmp_path / "steps/refused/outputs/answer.txt").exists()
    result = runner.run(
        step_id="typed", code=code, code_origin=CodeOrigin.HOST_RENDERED_EXECUTOR
    )
    assert result.succeeded, result.stderr
    assert (result.out_dir / "answer.txt").read_text() == "6"
    assert result.cwd == tmp_path
    assert not result.isolation_degraded
    assert result.runtime_provenance["runtime"] == "host"


def test_host_runner_refuses_invalid_wave_before_reserving_execution(
    cohort, tmp_path, monkeypatch,
):
    from easyicu.research_agent.execution import host_runner
    from easyicu.research_agent.execution.runner import WaveArtifactAuthority
    from easyicu.research_agent.execution.step_execution import CodeOrigin

    runner = host_runner.HostRunner(workdir=tmp_path, cohort_parquet=cohort)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid authority must not reserve or start execution")

    monkeypatch.setattr(host_runner, "prepare_host_execution", forbidden)
    with pytest.raises(ValueError, match="Wave authority receipt"):
        runner.run(
            step_id="typed", code_origin=CodeOrigin.HOST_RENDERED_EXECUTOR,
            wave_authority=WaveArtifactAuthority(error="missing"),
            code=("from easyicu.research_agent.execution.runners.deterministic_robustness "
                  "import _run_robustness_preflight_from_env\n"),
        )
    assert not (tmp_path / "steps").exists()


def test_host_uses_app_interpreter_windows_environment_and_run_font_cache(
    cohort, tmp_path, monkeypatch
):
    from easyicu.research_agent.execution.host_runner import HostRunner
    from easyicu.research_agent.execution.step_execution import CodeOrigin

    monkeypatch.setenv("SYSTEMROOT", "synthetic-system-root")
    monkeypatch.setenv("WINDIR", "synthetic-windows")
    monkeypatch.setenv("TEMP", str(tmp_path / "temp"))
    monkeypatch.setenv("UNRELATED_SECRET", "never-inherit")
    runner = HostRunner(workdir=tmp_path, cohort_parquet=cohort)
    code = """import json, os, sys
from pathlib import Path
cache = Path(os.environ['MPLCONFIGDIR'])
seen = (cache / 'sentinel').exists()
(cache / 'sentinel').write_text('cache')
Path(os.environ['STEP_OUT_DIR'], 'env.json').write_text(json.dumps({
    'python': sys.executable, 'cache': str(cache), 'seen': seen,
    'systemroot': os.environ.get('SYSTEMROOT'), 'windir': os.environ.get('WINDIR'),
    'temp': os.environ.get('TEMP'), 'secret': os.environ.get('UNRELATED_SECRET'),
    'host_only': os.environ.get('EASYICU_HOST_RENDERED_ONLY')}))
"""
    rows = []
    for step_id in ("one", "two"):
        result = runner.run(
            step_id=step_id, code=code, code_origin=CodeOrigin.HOST_RENDERED_EXECUTOR
        )
        assert result.succeeded, result.stderr
        rows.append(json.loads((result.out_dir / "env.json").read_text()))
    assert rows[0]["python"] == sys.executable
    assert rows[0]["cache"] == rows[1]["cache"]
    assert [row["seen"] for row in rows] == [False, True]
    assert rows[1]["systemroot"] == "synthetic-system-root"
    assert rows[1]["windir"] == "synthetic-windows"
    assert rows[1]["temp"] == os.environ["TEMP"]
    assert rows[1]["secret"] is None
    assert rows[1]["host_only"] == "1"


def test_host_runtime_receipt_is_bound_and_cannot_adopt_docker(cohort, tmp_path):
    from easyicu.research_agent.execution.host_runner import HostRunner
    from easyicu.research_agent.execution.budget_epoch import runtime_attempt_identity

    runner = HostRunner(workdir=tmp_path, cohort_parquet=cohort)
    capabilities = runner.validate_runtime_capabilities()
    receipt = runner.export_validated_runtime_bundle()
    assert receipt["schema"] == "easyicu.host_runtime_preflight/1"
    assert receipt["provenance"]["runtime"] == "host"
    assert len(runtime_attempt_identity(receipt).execution_kernel_identity_sha256) == 64
    assert runtime_attempt_identity(receipt).image_id is None
    second = HostRunner(workdir=tmp_path, cohort_parquet=cohort)
    second.adopt_validated_runtime_bundle(receipt)
    assert second.validate_runtime_capabilities() == capabilities
    with pytest.raises(ValueError):
        second.adopt_validated_runtime_bundle(
            {**receipt, "schema": "easyicu.docker_runtime_preflight/3"}
        )
    with pytest.raises(ValueError):
        second.adopt_validated_runtime_bundle(
            {
                **receipt,
                "provenance": {
                    **receipt["provenance"],
                    "python_executable": "another-python",
                },
            }
        )


def test_host_timeout_kills_detached_descendants(cohort, tmp_path):
    import psutil
    from easyicu.research_agent.execution.host_runner import HostRunner
    from easyicu.research_agent.execution.step_execution import CodeOrigin

    child_code = """import os, subprocess, sys, time
from pathlib import Path
grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
Path(os.environ['STEP_OUT_DIR'], 'grandchild.txt').write_text(str(grandchild.pid))
time.sleep(60)
"""
    code = f"""import os, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', {child_code!r}], start_new_session=True)
Path(os.environ['STEP_OUT_DIR'], 'pid.txt').write_text(str(child.pid))
time.sleep(60)
"""
    runner = HostRunner(workdir=tmp_path, cohort_parquet=cohort, timeout_seconds=1)
    result = runner.run(
        step_id="timeout", code=code, code_origin=CodeOrigin.HOST_RENDERED_EXECUTOR
    )
    assert result.timed_out
    pids = [
        int((result.out_dir / name).read_text())
        for name in ("pid.txt", "grandchild.txt")
    ]
    try:
        for pid in pids:
            assert (
                not psutil.pid_exists(pid)
                or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
            )
    finally:
        for pid in pids:
            if (
                psutil.pid_exists(pid)
                and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
            ):
                psutil.Process(pid).kill()


@pytest.mark.parametrize("denied", [None, "suspend", "children", "child", "parent"])
def test_terminate_tree_suspends_then_kills_children_and_parent(monkeypatch, denied):
    import psutil
    from types import SimpleNamespace
    from easyicu.research_agent.execution.host_runner import HostRunner

    calls = []

    def act(name):
        calls.append(name)
        if name == denied:
            raise psutil.AccessDenied(123)

    def children(*, recursive):
        assert recursive
        act("children")
        return [SimpleNamespace(kill=lambda: act("child"))]

    parent = SimpleNamespace(
        suspend=lambda: act("suspend"), children=children,
        kill=lambda: act("parent"),
    )
    monkeypatch.setattr(psutil, "Process", lambda pid: parent)
    HostRunner._terminate_tree(SimpleNamespace(pid=123))
    assert calls == (
        ["suspend", "children", "parent"] if denied == "children"
        else ["suspend", "children", "child", "parent"]
    )


def test_tree_cleanup_failure_still_reaps_parent_and_finishes_drains(
    tmp_path, monkeypatch
):
    import threading
    from easyicu.research_agent.execution import runner

    popen = runner.subprocess.Popen
    children = []

    def capture(*args, **kwargs):
        proc = popen(*args, **kwargs)
        children.append(proc)
        return proc

    def fail_cleanup(proc):
        raise RuntimeError("synthetic tree cleanup failure")

    monkeypatch.setattr(runner.subprocess, "Popen", capture)
    with pytest.raises(RuntimeError, match="synthetic tree cleanup failure"):
        runner._run_capturing_with_descendant_reaping(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=str(tmp_path),
            env=dict(os.environ),
            timeout=0.1,
            terminate_tree=fail_cleanup,
        )
    assert len(children) == 1 and children[0].poll() is not None
    assert not any(
        t.name.startswith(f"easyicu-runner-{children[0].pid}-")
        for t in threading.enumerate()
    )


def test_host_replay_policy_cannot_be_disabled_through_extra_env(cohort, tmp_path):
    from easyicu.research_agent.execution.host_runner import HostRunner

    with pytest.raises(ValueError, match="EASYICU_HOST_RENDERED_ONLY"):
        HostRunner(
            workdir=tmp_path,
            cohort_parquet=cohort,
            extra_env={"EASYICU_HOST_RENDERED_ONLY": "0"},
        )


def test_locked_request_provenance_reaches_host_and_refusal_preserves_outputs(
    cohort, tmp_path
):
    from easyicu.research_agent.execution.host_runner import HostRunner
    from easyicu.research_agent.execution.step_execution import (
        CodeOrigin,
        LockedStepExecutionRequest,
        StepExecutor,
    )

    runner = HostRunner(workdir=tmp_path, cohort_parquet=cohort)
    executor = StepExecutor(
        clear_output_dir=lambda path: pytest.fail("host owns cleanup")
    )
    out = tmp_path / "steps" / "typed" / "outputs"
    code = "import os\nfrom pathlib import Path\nPath(os.environ['STEP_OUT_DIR'], 'result').write_text('typed')"
    request = LockedStepExecutionRequest(
        step_id="typed",
        code=code,
        resolved_inputs_path=None,
        output_dir=out,
        code_origin=CodeOrigin.HOST_RENDERED_EXECUTOR,
    )
    assert executor.execute(runner=runner, request=request).succeeded
    unknown = LockedStepExecutionRequest(
        step_id="typed",
        code="raise AssertionError('must not run')",
        resolved_inputs_path=None,
        output_dir=out,
    )
    result = executor.execute(runner=runner, request=unknown)
    assert result.runner_failure_code.value == "host_runner_refuses_model_code"
    assert (out / "result").read_text() == "typed"


@pytest.mark.parametrize("returncode", [0, 1])
def test_host_availability_probes_application_interpreter(monkeypatch, returncode):
    from types import SimpleNamespace
    from easyicu.research_agent.execution import runner

    def probe(argv, **kwargs):
        assert argv[0] == sys.executable
        assert "psutil" in argv[-1]
        assert kwargs["timeout"] <= 5
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(runner.subprocess, "run", probe)
    result = runner.probe_runner_availability("host")
    assert result.available == (returncode == 0)
    assert result.reason_code == ("" if returncode == 0 else "host_runtime_unavailable")


def test_kernel_identity_on_installed_python_sources_without_git(tmp_path):
    from easyicu.research_agent.execution.kernel_identity import (
        build_execution_kernel_identity,
    )

    root = tmp_path / "site-packages" / "easyicu"
    owner = root / "research_agent" / "execution" / "runners" / "synthetic.py"
    owner.parent.mkdir(parents=True)
    (root / "__init__.py").write_text("")
    owner.write_text("VALUE = 17\n")
    lock = root / "research_agent" / "runner_image" / "requirements.lock"
    lock.parent.mkdir()
    lock.write_text("numpy==2.0\n")
    first = build_execution_kernel_identity(root)
    owner.write_text("VALUE = 18\n")
    second = build_execution_kernel_identity(root)
    assert first.identity_sha256 != second.identity_sha256
    assert first.requirements_lock_sha256 == second.requirements_lock_sha256
