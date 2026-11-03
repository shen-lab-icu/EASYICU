"""Replay downloads and the handoff lease check run on Windows too.

11-03 external review: ``study_replay_io.read_snapshot`` used ``os.O_NOFOLLOW``,
``os.O_DIRECTORY`` and ``dir_fd`` directly, and the durable handoff lease
check used ``os.pread``; Windows has none of them, so a replay download and
an approved first-observed handoff raised AttributeError there. Both now go
through ``easyicu.runtime.platform_compat``, which pins directories on both
platforms. Synthetic files only.
"""

from __future__ import annotations

import os
from pathlib import Path
import re

import pytest

from easyicu.runtime import platform_compat
from easyicu.webserver import durable_research, study_replay_io
from easyicu.webserver.study_replay_policy import ReplayExportError

POSIX_ONLY = re.compile(r"os\.(?:O_NOFOLLOW|O_DIRECTORY|pread)\b|dir_fd=")


@pytest.mark.parametrize("module", [study_replay_io, durable_research])
def test_no_posix_only_file_call_outside_the_platform_owner(module):
    assert not POSIX_ONLY.findall(Path(module.__file__).read_text(encoding="utf-8"))


def test_a_positioned_read_keeps_the_offset_with_and_without_pread(monkeypatch, tmp_path):
    path = tmp_path / "lease.lock"
    path.write_text("4242\n", encoding="utf-8")
    descriptor = os.open(path, os.O_RDWR)
    try:
        os.lseek(descriptor, 3, os.SEEK_SET)
        assert platform_compat.read_at(descriptor, 64, 0) == b"4242\n"
        monkeypatch.delattr(os, "pread")
        assert platform_compat.read_at(descriptor, 64, 0) == b"4242\n"
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 3
    finally:
        os.close(descriptor)


def test_a_snapshot_reads_nested_files_and_still_refuses_links(tmp_path):
    root = tmp_path / "run"
    (root / "pipeline" / "a").mkdir(parents=True)
    (root / "pipeline" / "a" / "plan.json").write_bytes(b"{}")
    assert study_replay_io.read_snapshot(root, "pipeline/a/plan.json") == b"{}"
    assert study_replay_io.read_snapshot(root, "pipeline/a/missing.json", optional=True) is None
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.json").write_bytes(b"{}")
    (root / "linked").symlink_to(tmp_path / "outside", target_is_directory=True)
    (root / "pipeline" / "file_link.json").symlink_to(tmp_path / "outside" / "secret.json")
    for relative in ("linked/secret.json", "pipeline/file_link.json"):
        with pytest.raises(ReplayExportError) as refused:
            study_replay_io.read_snapshot(root, relative)
        assert refused.value.code == "replay_path_forbidden"
