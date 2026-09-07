import os
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority import runtime_artifacts as artifacts


def test_windows_checkpoint_locks_the_same_nonempty_byte_and_releases_on_error(tmp_path, monkeypatch):
    calls = []
    def locking(fd, mode, count):
        calls.append((mode, count, os.lseek(fd, 0, os.SEEK_CUR), os.fstat(fd).st_size))
    monkeypatch.setattr(artifacts, "fcntl", None, raising=False)
    monkeypatch.setattr(artifacts, "msvcrt", SimpleNamespace(LK_LOCK=1, LK_UNLCK=2, locking=locking), raising=False)
    with pytest.raises(RuntimeError, match="synthetic"):
        with artifacts._checkpoint_write_lock(tmp_path):
            raise RuntimeError("synthetic")
    assert calls == [(1, 1, 0, 1), (2, 1, 0, 1)]


def test_checkpoint_cannot_silently_run_without_an_os_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "fcntl", None, raising=False)
    monkeypatch.setattr(artifacts, "msvcrt", None, raising=False)
    with pytest.raises(artifacts.RunArtifactAuthorityError, match="lock"):
        with artifacts._checkpoint_write_lock(tmp_path):
            pytest.fail("unlocked checkpoint authority")
