"""Host workers clean up descendants before losing their parent identity."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from easyicu.research_agent.execution.host_runner import HostRunner
from easyicu.research_agent.execution.runner import _run_capturing_with_descendant_reaping
from easyicu.runtime import host_worker


@pytest.mark.parametrize("entry", ["source", "desktop"])
@pytest.mark.parametrize("completion", ["normal", "exception", "timeout"])
def test_host_completion_kills_detached_child_and_grandchild(tmp_path, entry, completion):
    step = tmp_path / "steps" / "synthetic"
    output = step / "outputs"
    output.mkdir(parents=True)
    child_code = """import os, subprocess, sys, time
from pathlib import Path
p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
    start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
Path(os.environ['STEP_OUT_DIR'], 'grandchild.pid').write_text(str(p.pid))
time.sleep(30)
"""
    ending = {
        "normal": "pass",
        "exception": "raise RuntimeError('synthetic failure')",
        "timeout": "time.sleep(30)",
    }[completion]
    source = f"""import os, subprocess, sys, time
from pathlib import Path
out = Path(os.environ['STEP_OUT_DIR'])
p = subprocess.Popen([sys.executable, '-c', {child_code!r}],
    start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
(out / 'child.pid').write_text(str(p.pid))
deadline = time.monotonic() + 3
while not (out / 'grandchild.pid').exists():
    if time.monotonic() >= deadline:
        raise RuntimeError('synthetic grandchild did not start')
    time.sleep(0.01)
{ending}
"""
    script = step / "analysis.py"
    script.write_text(source)
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    if entry == "source":
        runner = object.__new__(HostRunner)
        runner.python_executable = sys.executable
        command = runner.build_command(script_path=script)
        assert command[1:3] == ["-m", "easyicu.runtime.host_worker"]
    else:
        backend = Path(__file__).parents[3] / "desktop" / "backend_entry.py"
        command = [sys.executable, str(backend), host_worker.WORKER_FLAG,
                   "execute", "--script", str(script), "--sha256", digest]
    env = {**os.environ, "EASYICU_HOST_RENDERED_ONLY": "1", "EASYICU_RUN_DIR": str(tmp_path),
           "STEP_OUT_DIR": str(output), "PYTHONPATH": str(Path(host_worker.__file__).parents[2])}
    try:
        if completion == "timeout":
            with pytest.raises(subprocess.TimeoutExpired):
                _run_capturing_with_descendant_reaping(command, cwd=str(tmp_path), env=env,
                    timeout=2, terminate_tree=HostRunner._terminate_tree)
        else:
            result = _run_capturing_with_descendant_reaping(command, cwd=str(tmp_path), env=env,
                timeout=8, terminate_tree=HostRunner._terminate_tree)
            assert (result.returncode == 0) == (completion == "normal"), result.stderr
        for name in ("child.pid", "grandchild.pid"):
            pid = int((output / name).read_text())
            assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    finally:
        # A regression must not leave the synthetic sleepers behind.
        for marker in output.glob("*.pid"):
            try:
                child = psutil.Process(int(marker.read_text()))
                child.kill()
            except psutil.NoSuchProcess:
                pass


@pytest.mark.parametrize("exit_during_lookup", [False, True])
def test_parent_cleanup_never_signals_a_reaped_or_reused_pid(monkeypatch, exit_during_lookup):
    polls = iter([None, 0] if exit_during_lookup else [0])
    looked_up = []

    def process(pid):
        looked_up.append(pid)
        return SimpleNamespace(suspend=lambda: pytest.fail("PID no longer owned"))

    monkeypatch.setattr(psutil, "Process", process)
    HostRunner._terminate_tree(SimpleNamespace(pid=123, poll=lambda: next(polls)))
    assert looked_up == ([123] if exit_during_lookup else [])
