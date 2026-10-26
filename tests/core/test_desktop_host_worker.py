"""The frozen entry point supports analysis without becoming a Python shell."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from easyicu.runtime import host_worker


BACKEND_ENTRY = Path(__file__).parents[2] / "desktop/backend_entry.py"


def _entry(*args, **kwargs):
    return subprocess.run(
        [sys.executable, str(BACKEND_ENTRY), host_worker.WORKER_FLAG, *args],
        capture_output=True, text=True, timeout=20, **kwargs,
    )


def test_backend_probe_does_not_start_the_webserver():
    result = _entry("capabilities", "--names", json.dumps(["json", "definitely_absent_easyicu_package"]))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["json"]
    identity = _entry("identity", "--names", '["pip"]')
    assert identity.returncode == 0, identity.stderr
    assert json.loads(identity.stdout)["python_version"] == sys.version.split()[0]


@pytest.mark.parametrize("arguments", [
    ["-c", "print('untrusted')"], ["-m", "http.server"], ["execute", "--script", "relative.py", "--sha256", "a" * 64],
    ["capabilities", "--names", '["json;print(1)"]'],
])
def test_worker_rejects_interpreter_flags_and_invalid_coordinates(arguments):
    result = _entry(*arguments)
    assert result.returncode != 0
    assert "untrusted" not in result.stdout


def _script(tmp_path):
    script = tmp_path / "steps" / "typed" / "analysis.py"
    script.parent.mkdir(parents=True)
    output = script.parent / "outputs"
    output.mkdir()
    script.write_text("import os\nfrom pathlib import Path\nPath(os.environ['STEP_OUT_DIR'], 'answer.txt').write_text('6')\n")
    env = {**os.environ, "EASYICU_HOST_RENDERED_ONLY": "1", "EASYICU_RUN_DIR": str(tmp_path), "STEP_OUT_DIR": str(output)}
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    return script, output, env, digest


def test_frozen_entry_executes_exact_authorized_bytes(tmp_path):
    script, output, env, digest = _script(tmp_path)
    result = _entry("execute", "--script", str(script), "--sha256", digest, env=env, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (output / "answer.txt").read_text() == "6"


@pytest.mark.parametrize("change", ["bytes", "origin", "workspace", "output"])
def test_worker_does_not_execute_after_authorization_coordinates_change(tmp_path, change):
    script, output, env, digest = _script(tmp_path)
    if change == "bytes":
        script.write_text(script.read_text() + "# changed after parent decision\n")
    elif change == "origin":
        env.pop("EASYICU_HOST_RENDERED_ONLY")
    elif change == "workspace":
        env["EASYICU_RUN_DIR"] = str(tmp_path.parent)
    else:
        env["STEP_OUT_DIR"] = str(tmp_path)
    result = _entry("execute", "--script", str(script), "--sha256", digest, env=env, cwd=tmp_path)
    assert result.returncode != 0
    assert not (output / "answer.txt").exists()


def test_frozen_host_commands_use_only_fixed_worker_protocol(tmp_path, monkeypatch):
    from easyicu.research_agent.execution.host_runner import HostRunner
    from easyicu.research_agent.execution import step_execution

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    runner = object.__new__(HostRunner)
    runner.python_executable = "/fake/easyicu-backend"
    script, _, _, digest = _script(tmp_path)
    assert runner.build_command(script_path=script) == [
        runner.python_executable, host_worker.WORKER_FLAG, "execute", "--script", str(script), "--sha256", digest,
    ]
    for action in ("capabilities", "identity"):
        assert runner._runtime_probe_command(action, ["json"], "not executed") == [
            runner.python_executable, host_worker.WORKER_FLAG, action, "--names", '["json"]',
        ]
    calls = []

    def probe(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=command[-1].encode())

    monkeypatch.setattr(step_execution.subprocess, "run", probe)
    assert step_execution.host_runtime_available(5)
    assert calls[0][1:3] == [host_worker.WORKER_FLAG, "capabilities"]
    monkeypatch.setattr(step_execution.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=b'[]'))
    assert not step_execution.host_runtime_available(5)


def test_frozen_replay_keeps_primary_bytes_and_scoped_variant_outputs(tmp_path):
    script, _, env, digest = _script(tmp_path)
    replay_root = tmp_path / "steps" / "sensitivity" / "outputs" / "model_replays" / "variant"
    replay_output = replay_root / "outputs"
    replay_output.mkdir(parents=True)
    env.update(EASYICU_RUN_DIR=str(replay_root), STEP_OUT_DIR=str(replay_output))
    result = _entry("replay", "--script", str(script), "--sha256", digest, env=env, cwd=script.parent)
    assert result.returncode == 0, result.stderr
    assert (replay_output / "answer.txt").read_text() == "6"
    env["STEP_OUT_DIR"] = str(tmp_path / "unrelated")
    result = _entry("replay", "--script", str(script), "--sha256", digest, env=env, cwd=script.parent)
    assert result.returncode != 0
