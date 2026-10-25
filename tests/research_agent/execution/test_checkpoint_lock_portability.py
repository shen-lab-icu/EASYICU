import os
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority import runtime_artifacts as artifacts
from easyicu.runtime import platform_compat as compat


def test_windows_checkpoint_locks_the_same_nonempty_byte_and_releases_on_error(tmp_path, monkeypatch):
    lock_path = tmp_path / ".manifest.checkpoint.lock"
    lock_path.write_bytes(b"\0")
    calls = []
    def locking(fd, mode, count):
        calls.append((mode, count, os.lseek(fd, 0, os.SEEK_CUR), os.fstat(fd).st_size))
    monkeypatch.setattr(compat, "fcntl", None)
    monkeypatch.setattr(compat, "msvcrt", SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=locking))
    with pytest.raises(RuntimeError, match="synthetic"):
        with artifacts._checkpoint_write_lock(tmp_path):
            raise RuntimeError("synthetic")
    assert calls == [(1, 1, 0, 1), (2, 1, 0, 1)]
    assert lock_path.read_bytes() == b"\0"


def test_checkpoint_cannot_silently_run_without_an_os_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(compat, "fcntl", None)
    monkeypatch.setattr(compat, "msvcrt", None)
    with pytest.raises(artifacts.RunArtifactAuthorityError, match="lock"):
        with artifacts._checkpoint_write_lock(tmp_path):
            pytest.fail("unlocked checkpoint authority")
