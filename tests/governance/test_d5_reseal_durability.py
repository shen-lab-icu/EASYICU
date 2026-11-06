"""Reseal recovery and lock entry under faults that stay on (Codex delta5 B5-*).

B5-RECOVERY-DURABILITY: a rollback that rewrote a file's original bytes but
could not sync its directory leaves a journal. ``recover()`` used to see the
original digest, skip the write (and so the directory sync), delete the
journal and report success while the same fault was still on.

B5-LOCK-ENTRY: a failure after the lifecycle lock was taken but before its
descriptor was handed over (e.g. ENOSPC writing the lock metadata) leaked
the locked descriptor, so the same process could not reseal or recover
again (``reseal_active``) until it exited.

Every run works on a copy of ``src`` in a temporary directory; the real tree
is only read.
"""

from __future__ import annotations

import errno
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FILES = (
    "src/easyicu/data/concept-dict.LOCK.json",
    "src/easyicu/research_agent/orchestration/profiles.py",
    "src/easyicu/research_agent/providers/client_trust.py",
)
APPLY = dict(consent="test consent", label="test reseal", locked_at="2026-11-07T12:00:00Z",
             source_commit="0" * 40)


def _tool():
    spec = importlib.util.spec_from_file_location("reseal_concept_dict", ROOT / "tools/reseal_concept_dict.py")
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


def _persistent_dir_eio(monkeypatch, tool, *, skip=0):
    """Every directory fsync after the first ``skip`` raises EIO; file fsyncs work."""
    real = os.fsync
    calls = {"dir": 0}

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            calls["dir"] += 1
            if calls["dir"] > skip:
                raise OSError(errno.EIO, os.strerror(errno.EIO))
        return real(fd)

    monkeypatch.setattr(tool.os, "fsync", fsync)
    return calls


@pytest.mark.skipif(os.name == "nt", reason="no directory fsync on Windows")
def test_recovery_under_a_persistent_directory_fault_keeps_the_journal(root, monkeypatch):
    tool = _tool()
    before = _digests(root)
    # The journal's directory sync succeeds; every later one fails, so the
    # rollback rewrites the originals but cannot make them durable.
    _persistent_dir_eio(monkeypatch, tool, skip=1)
    with pytest.raises(tool.ResealError) as caught:
        tool.apply(root, **APPLY)
    assert caught.value.reason == "reseal_rollback_incomplete"
    assert _digests(root) == before  # visible bytes are already the originals
    journal = root / tool.RECOVERY_JOURNAL
    assert journal.exists()

    # Same fault, still on: recovery must not call this durable.
    with pytest.raises(tool.ResealError) as caught:
        tool.recover(root)
    assert caught.value.reason == "reseal_rollback_incomplete"
    assert journal.exists()

    monkeypatch.undo()
    tool = _tool()
    assert sorted(tool.recover(root)["restored"]) == sorted(FILES)
    assert _digests(root) == before
    assert not journal.exists() and not (root / tool.RESEAL_LOCK).exists()


@pytest.mark.parametrize("entry", ["apply", "recover"])
def test_a_failure_while_taking_the_lock_releases_it(root, monkeypatch, entry):
    tool = _tool()
    before = _digests(root)
    real = tool.os.write
    failed = []

    def write(fd, data):
        if not failed:  # the lock metadata is the first os.write of either entry
            failed.append(fd)
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        return real(fd, data)

    monkeypatch.setattr(tool.os, "write", write)
    if entry == "apply":
        with pytest.raises(OSError) as caught:
            tool.apply(root, **APPLY)
    else:
        with pytest.raises(OSError) as caught:
            tool.recover(root)
    assert caught.value.errno == errno.ENOSPC and failed
    assert _digests(root) == before
    monkeypatch.undo()
    # The same process can take the lock again at once.
    with tool._Lifecycle(root) as held:
        assert held.fd is not None
    assert not (root / tool.RESEAL_LOCK).exists()
