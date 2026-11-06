"""Apply and recover share one lifecycle lock (Codex delta4 D4-RESEAL-RACE).

``recover()`` used to restore files and delete the journal without any mutex,
so a recovery started beside a running apply put the first file back while the
apply went on writing the others, and the apply's own rollback then found its
journal gone: a mix of old and new files with nothing left to recover from.

Every run here works on a copy of ``src`` in a temporary directory; the real
tree is only read. The second process is a real interpreter, so the lock is
exercised across processes, not just across file descriptors.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/reseal_concept_dict.py"
FILES = (
    "src/easyicu/data/concept-dict.LOCK.json",
    "src/easyicu/research_agent/orchestration/profiles.py",
    "src/easyicu/research_agent/providers/client_trust.py",
)
APPLY = dict(consent="test consent", label="test reseal", locked_at="2026-11-07T12:00:00Z",
             source_commit="0" * 40)

# Runs apply in its own process. MODE=pause: stop after the first file is
# replaced until GO appears. MODE=crash: die (no rollback, no cleanup) after
# the second file is replaced.
WORKER = r"""
import importlib.util, os, sys, time
from pathlib import Path
tool_path, root, mode, ready, go = sys.argv[1:6]
spec = importlib.util.spec_from_file_location("reseal_concept_dict", tool_path)
tool = importlib.util.module_from_spec(spec); sys.modules[spec.name] = tool; spec.loader.exec_module(tool)
real = tool._write_atomic
written = []
def hooked(path, raw):
    real(path, raw)
    if path.name == tool.RECOVERY_JOURNAL:
        return
    written.append(path.name)
    if mode == "pause" and len(written) == 1:
        Path(ready).write_text("ready")
        deadline = time.time() + 120
        while not Path(go).exists():
            if time.time() > deadline:
                os._exit(3)
            time.sleep(0.05)
    if mode == "crash" and len(written) == 2:
        os._exit(9)
tool._write_atomic = hooked
tool.apply(Path(root), consent="test consent", label="test reseal",
           locked_at="2026-11-07T12:00:00Z", source_commit="0" * 40)
print("applied")
"""


def _tool():
    spec = importlib.util.spec_from_file_location("reseal_concept_dict", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _digests(root):
    return {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in FILES}


@pytest.fixture
def root(tmp_path):
    shutil.copytree(ROOT / "src", tmp_path / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "static", "*.pyc"))
    path = tmp_path / "src/easyicu/data/concept-dict.json"
    data = json.loads(path.read_text())
    data["_comment_reseal_test"] = "changed by the test"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return tmp_path


def _worker(root: Path, mode: str):
    ready, go = root / "worker.ready", root / "worker.go"
    script = root / "worker.py"
    script.write_text(WORKER)
    proc = subprocess.Popen(
        [sys.executable, str(script), str(TOOL), str(root), mode, str(ready), str(go)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return proc, ready, go


def _wait_for(path: Path, proc, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while not path.exists():
        if proc.poll() is not None:
            raise AssertionError(f"worker exited early: {proc.returncode} {proc.stderr.read()[-500:]}")
        if time.time() > deadline:
            proc.kill()
            raise AssertionError("worker never reached the pause point")
        time.sleep(0.05)


def test_recover_beside_a_running_apply_refuses_and_touches_nothing(root):
    tool = _tool()
    proc, ready, go = _worker(root, "pause")
    try:
        _wait_for(ready, proc)
        mid_apply = _digests(root)  # one file new, two old
        assert (root / tool.RECOVERY_JOURNAL).exists()
        with pytest.raises(tool.ResealError) as caught:
            tool.recover(root)
        assert caught.value.reason == "reseal_active"
        # Nothing restored, the apply's journal kept.
        assert _digests(root) == mid_apply
        assert (root / tool.RECOVERY_JOURNAL).exists()
        # A second apply is refused for the same reason, before the journal check.
        with pytest.raises(tool.ResealError) as second:
            tool.apply(root, **APPLY)
        assert second.value.reason == "reseal_active"
    finally:
        go.write_text("go")
        out, err = proc.communicate(timeout=300)
    assert proc.returncode == 0, err[-800:]
    assert "applied" in out
    # The apply finished as its own transaction: every pin moved, nothing left behind.
    lock = json.loads((root / FILES[0]).read_text())
    assert lock["concept_dict_sha256"] == hashlib.sha256(
        (root / "src/easyicu/data/concept-dict.json").read_bytes()).hexdigest()
    assert not (root / tool.RECOVERY_JOURNAL).exists() and not (root / tool.RESEAL_LOCK).exists()


def test_a_crashed_apply_leaves_a_stale_lock_that_recovery_takes(root):
    tool = _tool()
    before = _digests(root)
    proc, _ready, _go = _worker(root, "crash")
    proc.communicate(timeout=300)
    assert proc.returncode == 9
    # The crash left both files behind and two of three replaced.
    assert (root / tool.RESEAL_LOCK).exists() and (root / tool.RECOVERY_JOURNAL).exists()
    assert _digests(root) != before
    # The dead holder's OS lock is gone: apply sees the journal, not a live reseal.
    with pytest.raises(tool.ResealError) as caught:
        tool.apply(root, **APPLY)
    assert caught.value.reason == "reseal_recovery_pending"
    assert sorted(tool.recover(root)["restored"]) == sorted(FILES)
    assert _digests(root) == before
    assert not (root / tool.RECOVERY_JOURNAL).exists() and not (root / tool.RESEAL_LOCK).exists()


def test_a_stale_lock_file_without_a_journal_does_not_block_a_reseal(root):
    tool = _tool()
    (root / tool.RESEAL_LOCK).write_text(json.dumps({"pid": 2 ** 22 + 7, "started_at": "x"}))
    result = tool.apply(root, **APPLY)
    assert result["verify"]["problems"] == []
    assert not (root / tool.RESEAL_LOCK).exists()


def test_recover_while_another_process_holds_the_lock_refuses(root):
    tool = _tool()
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import importlib.util,sys,time;"
         f"s=importlib.util.spec_from_file_location('t',{str(TOOL)!r});m=importlib.util.module_from_spec(s);"
         "sys.modules['t']=m;s.loader.exec_module(m);from pathlib import Path;"
         f"l=m._Lifecycle(Path({str(root)!r})).__enter__();print('held',flush=True);sys.stdin.read()"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        (root / tool.RECOVERY_JOURNAL).write_text(json.dumps({"files": []}))
        with pytest.raises(tool.ResealError) as caught:
            tool.recover(root)
        assert caught.value.reason == "reseal_active"
        assert (root / tool.RECOVERY_JOURNAL).exists()
    finally:
        holder.stdin.close()
        holder.wait(timeout=60)
    # Once the holder is gone the same recovery runs.
    assert tool.recover(root)["restored"] == []
    assert not (root / tool.RECOVERY_JOURNAL).exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX inode check")
def test_the_lock_file_is_removed_while_still_held(root):
    tool = _tool()
    with tool._Lifecycle(root) as held:
        inode = os.fstat(held.fd).st_ino
        assert os.stat(root / tool.RESEAL_LOCK).st_ino == inode
    assert not (root / tool.RESEAL_LOCK).exists()


def _dir_fsync_fault(monkeypatch, tool, code, *, skip_journal_dir_calls=0):
    """Make os.fsync on a directory raise ``code``; file fsyncs are untouched."""
    import stat

    real = os.fsync
    state = {"dir_calls": 0}

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            state["dir_calls"] += 1
            if state["dir_calls"] > skip_journal_dir_calls:
                raise OSError(code, os.strerror(code))
        return real(fd)

    monkeypatch.setattr(tool.os, "fsync", fsync)
    return state


def test_an_eio_on_a_directory_fsync_fails_the_reseal_and_restores(root, monkeypatch):
    import errno

    tool = _tool()
    before = _digests(root)
    # The journal's own directory fsync succeeds; the first file's fails, and
    # so does every later one (the restore reports them and keeps the journal).
    _dir_fsync_fault(monkeypatch, tool, errno.EIO, skip_journal_dir_calls=1)
    with pytest.raises(tool.ResealError) as caught:
        tool.apply(root, **APPLY)
    assert caught.value.reason == "reseal_rollback_incomplete"
    assert (root / tool.RECOVERY_JOURNAL).exists()
    monkeypatch.undo()
    tool = _tool()
    tool.recover(root)
    assert _digests(root) == before


def test_an_unsupported_directory_fsync_is_tolerated(root, monkeypatch):
    import errno

    tool = _tool()
    _dir_fsync_fault(monkeypatch, tool, errno.EINVAL)
    assert tool.apply(root, **APPLY)["verify"]["problems"] == []


def test_a_journal_that_could_not_be_made_durable_is_removed(root, monkeypatch):
    import errno

    tool = _tool()
    before = _digests(root)
    _dir_fsync_fault(monkeypatch, tool, errno.EIO)
    with pytest.raises(tool.ResealError) as caught:
        tool.apply(root, **APPLY)
    assert caught.value.reason == "reseal_not_started"
    assert _digests(root) == before
    assert not (root / tool.RECOVERY_JOURNAL).exists() and not (root / tool.RESEAL_LOCK).exists()
