"""Real host primitives, also run by the lead on windows-latest."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from easyicu.runtime import platform_compat as compat


def test_native_lock_excludes_another_process_and_releases_on_exit(tmp_path):
    path = tmp_path / "lock"
    script = """
import sys
from easyicu.runtime.platform_compat import lock_exclusive
with open(sys.argv[1], 'a+b') as handle:
    try:
        lock_exclusive(handle.fileno(), blocking=False)
    except BlockingIOError:
        print('held')
    else:
        print('acquired')
"""

    def contender():
        return subprocess.run(
            [sys.executable, "-c", script, str(path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout.strip()

    with path.open("a+b") as first:
        compat.lock_exclusive(first.fileno(), blocking=False)
        assert contender() == "held"
        compat.unlock_file(first.fileno())
    assert contender() == "acquired"
    # The preceding child exited while holding the lock.
    assert contender() == "acquired"


def test_native_authority_roundtrip(tmp_path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.authority.step_capsule import (
        put_content_blob,
        read_verified_content,
    )

    store = EvidenceStore(tmp_path)
    record = store.register_json(
        kind="log",
        description="Synthetic native test",
        payload={"synthetic": True},
        filename="synthetic.json",
        evidence_id="native",
    )
    assert EvidenceStore(tmp_path).records()[0].sha256 == record.sha256
    ref = put_content_blob(tmp_path, payload=b"synthetic", media_type="text/plain")
    assert read_verified_content(tmp_path, ref) == b"synthetic"
    with compat.SecureDirectory(tmp_path) as directory:
        assert directory.identity == compat._identity(tmp_path.stat())
        fd = directory.open("file", os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        try:
            os.write(fd, b"binary\r\n\x1a\n")
        finally:
            os.close(fd)
        directory.replace("file", "renamed")
        fd = directory.open("renamed", os.O_RDONLY)
        try:
            assert os.read(fd, 100) == b"binary\r\n\x1a\n"
        finally:
            os.close(fd)


def test_native_nofollow_rejects_parent_link(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "file").write_bytes(b"protected")
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Host lacks symlink privilege: {exc}")
    with pytest.raises(OSError):
        with compat.SecureDirectory(link):
            pytest.fail("directory link accepted")
    assert (real / "file").read_bytes() == b"protected"


@pytest.mark.skipif(os.name != "nt", reason="Win32 directory sharing semantics")
def test_windows_native_directory_cannot_be_replaced_while_pinned(tmp_path):
    selected = tmp_path / "selected"
    selected.mkdir()
    with compat.SecureDirectory(selected):
        with pytest.raises(OSError):
            selected.rename(tmp_path / "moved")
    selected.rename(tmp_path / "moved")
    assert Path(tmp_path / "moved").is_dir()
