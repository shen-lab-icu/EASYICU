"""Synthetic checks for the host portability boundary; no ASGI transport."""

import hashlib
import os
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

from easyicu.runtime import platform_compat as compat

pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX syscall emulation; native Windows checks live in test_platform_native.py",
)


@pytest.fixture(params=[False, True], ids=["posix", "windows"])
def host(request, monkeypatch):
    """Use real POSIX fds underneath a fake Win32/CRT syscall boundary.

    This exercises the Windows algorithm on Linux; native Windows handle and
    sharing semantics are additionally checked by windows-latest.
    """
    if request.param:
        native = compat.os
        facade = SimpleNamespace(**vars(native))
        facade.name = "nt"
        for name in ("O_NOFOLLOW", "O_DIRECTORY", "fchmod"):
            delattr(facade, name)

        def open_file(path, flags, mode=0o600, **kwargs):
            if kwargs.get("dir_fd") is not None or Path(path).is_dir():
                raise PermissionError("Windows cannot open a directory CRT fd")
            return native.open(path, flags, mode, **kwargs)

        facade.open = open_file

        def rename(source, target):
            native.link(source, target)  # Windows rename is no-clobber.
            native.unlink(source)

        facade.rename = rename
        monkeypatch.setattr(compat, "os", facade)
        monkeypatch.setattr(compat, "sys", SimpleNamespace(platform="win32"))

        def pin(path, stack):
            info = native.lstat(path)
            compat._reject_reparse(info)
            if not compat.stat.S_ISDIR(info.st_mode):
                raise compat.PlatformCapabilityError("directory_required")

        monkeypatch.setattr(compat, "_pin_windows_directory", pin)
        monkeypatch.setattr(
            compat,
            "_windows_handle_info",
            lambda fd: (0, *compat._identity(native.fstat(fd))),
        )
        import fcntl

        def locking(fd, mode, count):
            assert count == 1
            assert native.lseek(fd, 0, native.SEEK_CUR) == 0
            fcntl.flock(
                fd, fcntl.LOCK_UN if mode == 2 else fcntl.LOCK_EX | fcntl.LOCK_NB
            )

        monkeypatch.setattr(compat, "fcntl", None)
        monkeypatch.setattr(compat, "resource", None)
        monkeypatch.setattr(
            compat,
            "msvcrt",
            SimpleNamespace(
                LK_NBLCK=1, LK_UNLCK=2, locking=locking, get_osfhandle=lambda fd: fd
            ),
        )
    return request.param


@pytest.mark.parametrize("existing", [False, True])
def test_nofollow_create_race_does_not_create_outside_file(
    tmp_path, host, monkeypatch, existing
):
    if not host:
        pytest.skip("Windows lstat/open race")
    path, outside = tmp_path / "selected", tmp_path / "outside"
    if existing:
        path.write_bytes(b"original")
    original_open = compat.os.open

    def swap(selected, flags, mode=0o600, **kwargs):
        path.unlink(missing_ok=True)
        path.symlink_to(outside)
        return original_open(selected, flags, mode, **kwargs)

    monkeypatch.setattr(compat.os, "open", swap)
    with pytest.raises(OSError):
        compat.open_nofollow(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
    assert not outside.exists()


def test_checkpoint_roundtrip_without_directory_crt_fd(tmp_path, host, monkeypatch):
    from easyicu.research_agent.authority import runtime_artifacts

    monkeypatch.setattr(runtime_artifacts, "os", compat.os)
    target = tmp_path / "manifest.json"
    assert runtime_artifacts.write_run_checkpoint(target, {"synthetic": True}) == 1
    assert runtime_artifacts.write_run_checkpoint(target, {"synthetic": False}) == 2
    import json

    assert json.loads(target.read_text())["synthetic"] is False


def test_windows_relative_tree_digest_matches_posix_bytes():
    from easyicu.research_agent.authority.run_input import _tree_sha256

    raw = b"# synthetic implementation\n"
    path = SimpleNamespace(
        as_posix=lambda: "nested/owner.py",
        is_file=lambda: True,
        relative_to=lambda root: PureWindowsPath("nested/owner.py"),
        read_bytes=lambda: raw,
    )
    expected = hashlib.sha256(
        b"nested/owner.py" + len(raw).to_bytes(8, "big") + raw
    ).hexdigest()
    assert _tree_sha256(None, relative_paths=[path]) == expected


def test_platform_owner_provides_exclusive_lock(tmp_path, host):
    from easyicu.runtime import platform_compat as compat

    path = tmp_path / "synthetic.lock"
    with path.open("a+b") as first, path.open("a+b") as second:
        compat.lock_exclusive(first.fileno(), blocking=False)
        try:
            with pytest.raises(BlockingIOError):
                compat.lock_exclusive(second.fileno(), blocking=False)
        finally:
            compat.unlock_file(first.fileno())
        compat.lock_exclusive(second.fileno(), blocking=False)
        compat.unlock_file(second.fileno())


def test_nofollow_refuses_link_without_truncation(tmp_path, host):
    from easyicu.runtime import platform_compat as compat

    target = tmp_path / "target"
    target.write_bytes(b"synthetic original")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(OSError):
        compat.open_nofollow(link, os.O_WRONLY | os.O_TRUNC)
    assert target.read_bytes() == b"synthetic original"


def test_posix_tree_digest_keeps_legacy_bytes(tmp_path):
    from easyicu.research_agent.authority.run_input import _tree_sha256
    from easyicu.research_agent.authority.pipeline_cache import (
        _tree_sha256 as cache_tree,
    )

    (tmp_path / "nested").mkdir()
    for name in ("z.py", "nested/a.py", "nested-/b.py"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"# synthetic\n")
    digest = hashlib.sha256()
    paths = sorted(tmp_path.rglob("*.py"))
    legacy_cache = {}
    for path in paths:
        raw = path.read_bytes()
        digest.update(str(path.relative_to(tmp_path)).encode())
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        legacy_cache[str(path.relative_to(tmp_path))] = hashlib.sha256(raw).hexdigest()
    import json

    cache_digest = hashlib.sha256(
        json.dumps(
            legacy_cache,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    assert _tree_sha256(tmp_path) == digest.hexdigest()
    assert cache_tree(tmp_path, paths) == cache_digest


def test_secure_directory_publication_read_and_conflict(tmp_path, host):
    from easyicu.research_agent.authority.filesystem import (
        AnchoredDirectory,
        AuthorityFilesystemError,
    )

    with AnchoredDirectory.open(tmp_path) as directory:
        identity = directory.identity
        directory.replace_bytes("record.json", b"first")
        directory.replace_bytes("record.json", b"second")
        assert directory.read_bytes("record.json", max_bytes=20) == b"second"
        directory.publish_immutable_bytes("receipt", b"stable")
        directory.publish_immutable_bytes("receipt", b"stable")
        with pytest.raises(AuthorityFilesystemError, match="conflicts"):
            directory.publish_immutable_bytes("receipt", b"tamper")
        directory.assert_still_selected()
        assert directory.identity == identity
        with pytest.raises(AuthorityFilesystemError, match="already exists"):
            directory.replace_bytes("record.json", b"third", require_absent=True)
    assert list(tmp_path.glob("*.tmp")) == []


def test_evidence_store_roundtrip_and_immutable_conflict(tmp_path, host):
    from easyicu.research_agent.authority.evidence_store import (
        EvidenceStore,
        _atomic_write_once_bytes,
        EvidenceAuthorityIntegrityError,
        _atomic_copy_file,
        _exclusive_evidence_authority_lock,
    )

    store = EvidenceStore(tmp_path)
    record = store.register_json(
        kind="log",
        description="Synthetic portability record",
        payload={"synthetic": True},
        filename="synthetic.json",
        evidence_id="synthetic_portability",
    )
    assert record.relative_path.startswith("evidence/")
    restored = EvidenceStore(tmp_path)
    assert any(item.sha256 == record.sha256 for item in restored.records())
    target = tmp_path / record.relative_path
    original = target.read_bytes()
    _atomic_write_once_bytes(target, original, expected_root=tmp_path)
    with pytest.raises(EvidenceAuthorityIntegrityError):
        _atomic_write_once_bytes(target, b"tamper", expected_root=tmp_path)
    source = tmp_path / "synthetic.txt"
    source.write_bytes(b"synthetic copy")
    destination = target.parent / "copy"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    _atomic_copy_file(
        source, destination, expected_root=tmp_path, expected_sha256=digest
    )
    assert destination.read_bytes() == b"synthetic copy"
    with pytest.raises(EvidenceAuthorityIntegrityError, match="changed"):
        _atomic_copy_file(
            source,
            target.parent / "bad-copy",
            expected_root=tmp_path,
            expected_sha256="0" * 64,
        )
    # A second instance must use an OS lock as well as its process-local RLock.
    with _exclusive_evidence_authority_lock(target.parent, expected_root=tmp_path):
        with (target.parent / ".evidence_authority.lock").open("a+b") as contender:
            with pytest.raises(BlockingIOError):
                compat.lock_exclusive(contender.fileno(), blocking=False)


def test_linked_parent_is_refused_before_writing(tmp_path, host):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        with compat.SecureDirectory(link):
            pytest.fail("linked parent accepted")
    assert list(outside.iterdir()) == []


def test_open_nofollow_regular_and_truncate(tmp_path, host):
    path = tmp_path / "file"
    path.write_bytes(b"synthetic")
    fd = compat.open_nofollow(path, os.O_WRONLY | os.O_TRUNC)
    os.close(fd)
    assert path.read_bytes() == b""
    fd = compat.open_nofollow(path, os.O_RDONLY)
    os.close(fd)


def test_open_nofollow_detects_leaf_swap_before_truncate(tmp_path, host, monkeypatch):
    if not host:
        pytest.skip("Windows identity-comparison fallback")
    path = tmp_path / "file"
    path.write_bytes(b"original")
    open_original = compat.os.open

    def swap(selected, flags, mode=0o600, **kwargs):
        Path(selected).rename(tmp_path / "old")
        Path(selected).write_bytes(b"replacement")
        return open_original(selected, flags, mode, **kwargs)

    monkeypatch.setattr(compat.os, "open", swap)
    with pytest.raises(compat.PlatformCapabilityError) as exc:
        compat.open_nofollow(path, os.O_WRONLY | os.O_TRUNC)
    assert exc.value.reason == "file_selector_changed"
    assert path.read_bytes() == b"replacement"


def test_reparse_attributes_are_refused():
    info = SimpleNamespace(st_mode=0o100600, st_file_attributes=0x400)
    with pytest.raises(compat.PlatformCapabilityError) as exc:
        compat._reject_reparse(info)
    assert exc.value.reason == "reparse_point_refused"


def test_missing_lock_backend_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(compat, "fcntl", None)
    monkeypatch.setattr(compat, "msvcrt", None)
    with (tmp_path / "lock").open("a+b") as handle:
        with pytest.raises(compat.PlatformCapabilityError) as exc:
            compat.lock_exclusive(handle.fileno())
    assert exc.value.reason == "exclusive_lock_unavailable"


def test_windows_lock_retries_and_preserves_bytes_offset(tmp_path, host, monkeypatch):
    if not host:
        pytest.skip("Windows CRT retry contract")
    calls = []

    def locking(fd, mode, count):
        calls.append((mode, count, os.lseek(fd, 0, os.SEEK_CUR)))
        if len(calls) == 1:
            raise PermissionError(13, "synthetic contention")

    monkeypatch.setattr(compat.msvcrt, "locking", locking)
    monkeypatch.setattr(compat.time, "sleep", lambda seconds: None)
    path = tmp_path / "lock"
    path.write_bytes(b"receipt")
    with path.open("r+b") as handle:
        handle.seek(3)
        compat.lock_exclusive(handle.fileno())
        assert handle.tell() == 3
        compat.unlock_file(handle.fileno())
        assert handle.tell() == 3
    assert calls == [(1, 1, 0), (1, 1, 0), (2, 1, 0)]
    assert path.read_bytes() == b"receipt"


def test_resource_shim_and_rss_degradation(host, monkeypatch, caplog):
    monkeypatch.setattr(compat, "_DEGRADATIONS", set())
    calls = []
    fake = SimpleNamespace(
        RLIMIT_AS=7,
        RUSAGE_SELF=0,
        setrlimit=lambda *args: calls.append(args),
        # ru_maxrss is KiB on Linux and bytes on macOS.
        getrusage=lambda who: SimpleNamespace(
            ru_maxrss=4 * (1024 * 1024 if compat.sys.platform == "darwin" else 1024)
        ),
    )
    monkeypatch.setattr(compat, "resource", None if host else fake)
    assert compat.set_resource_limit("RLIMIT_AS", (10, 20)) is (not host)
    assert compat.peak_rss_mb() == (0.0 if host else 4.0)
    if host:
        assert compat.recorded_degradations() == (
            "peak_rss_unavailable",
            "resource_limits_unavailable",
        )
        assert "resource_limits_unavailable" in caplog.text
    else:
        assert calls == [(7, (10, 20))]


def test_fsync_and_replace_have_windows_durability_boundary(
    tmp_path, host, monkeypatch
):
    fsyncs = []
    original = compat.os.fsync

    def fsync(fd):
        fsyncs.append(fd)
        return original(fd)

    monkeypatch.setattr(compat.os, "fsync", fsync)
    compat.fsync_directory(tmp_path)
    assert len(fsyncs) == (0 if host else 1)
    source, target = tmp_path / "temporary", tmp_path / "target"
    source.write_bytes(b"complete")
    target.write_bytes(b"old")
    compat.atomic_replace(source, target)
    assert target.read_bytes() == b"complete"


@pytest.mark.parametrize("name", ["../escape", "x/y", "x\\y", "", ".."])
def test_secure_directory_rejects_non_components(tmp_path, name, host):
    with compat.SecureDirectory(tmp_path) as directory:
        with pytest.raises(compat.PlatformCapabilityError):
            directory.open(name, os.O_CREAT | os.O_WRONLY)


@pytest.mark.parametrize("name", ["file:stream", "NUL.txt", "COM1", "alias.", "alias "])
def test_windows_aliases_are_refused(tmp_path, name, host):
    if not host:
        pytest.skip("Windows filename aliases")
    with compat.SecureDirectory(tmp_path) as directory:
        with pytest.raises(compat.PlatformCapabilityError):
            directory.open(name, os.O_CREAT | os.O_WRONLY)


def test_additional_evidence_readers_refuse_links(tmp_path, host, monkeypatch):
    from easyicu.webserver import research_evidence_preview, run_step_trace

    target, link = tmp_path / "target", tmp_path / "link"
    target.write_bytes(b"synthetic")
    link.symlink_to(target)
    for module in (research_evidence_preview, run_step_trace):
        monkeypatch.setattr(module, "os", compat.os)
    with pytest.raises(research_evidence_preview.EvidencePreviewError):
        research_evidence_preview._read_regular_file(link, max_bytes=100)
    with pytest.raises(research_evidence_preview.EvidencePreviewError):
        research_evidence_preview._sha256_file(link)
    assert run_step_trace._read_bounded(link, 100) is None


def test_stop_record_missing_is_not_unreadable_on_windows(tmp_path, host, monkeypatch):
    from easyicu.research_agent.execution import executor_stop_record

    monkeypatch.setattr(executor_stop_record, "os", compat.os)
    assert executor_stop_record.read_executor_stop_record(
        tmp_path, expected_owner="synthetic"
    ) == (None, None)


def test_windows_directory_pin_denies_write_and_delete(tmp_path, monkeypatch):
    from contextlib import ExitStack
    import ctypes

    calls, closed = [], []

    def create(*args):
        calls.append(args)
        return 42

    kernel = SimpleNamespace(CreateFileW=create, CloseHandle=closed.append)
    monkeypatch.setattr(compat, "_windows_api", lambda: (ctypes, None, kernel))
    info = tmp_path.stat()
    monkeypatch.setattr(
        compat, "_windows_handle_info", lambda handle: (0x10, info.st_dev, info.st_ino)
    )
    with ExitStack() as stack:
        compat._pin_windows_directory(tmp_path, stack)
        assert closed == []
        assert calls[0][2] == 0x1
        assert calls[0][5] == 0x02200000
    assert closed == [42]


def test_windows_directory_pin_checks_native_identity(tmp_path, monkeypatch):
    from contextlib import ExitStack
    import ctypes

    closed = []
    kernel = SimpleNamespace(CreateFileW=lambda *args: 42, CloseHandle=closed.append)
    monkeypatch.setattr(compat, "_windows_api", lambda: (ctypes, None, kernel))
    info = tmp_path.stat()
    monkeypatch.setattr(
        compat,
        "_windows_handle_info",
        lambda handle: (0x10, info.st_dev, info.st_ino + 1),
    )
    with pytest.raises(
        compat.PlatformCapabilityError, match="directory_selector_changed"
    ):
        with ExitStack() as stack:
            compat._pin_windows_directory(tmp_path, stack)
    assert closed == [42]


def test_chmod_private_fd_and_identity(tmp_path, host):
    with (tmp_path / "private").open("a+b") as handle:
        compat.chmod_private_fd(handle.fileno())
        assert compat.descriptor_identity(handle.fileno()) == compat._identity(
            os.fstat(handle.fileno())
        )
        if not host:
            assert os.fstat(handle.fileno()).st_mode & 0o777 == 0o600


def test_secure_tree_creation_and_capsule_roundtrip(tmp_path, host):
    from easyicu.research_agent.authority.step_capsule import (
        put_content_blob,
        read_verified_content,
    )

    with compat.secure_directory_tree(tmp_path, ("a", "b"), create=True) as directory:
        assert directory.path == tmp_path / "a" / "b"
    ref = put_content_blob(tmp_path, payload=b"synthetic", media_type="text/plain")
    assert read_verified_content(tmp_path, ref) == b"synthetic"


def test_windows_cache_and_revision_tree_digests_match_posix(tmp_path, monkeypatch):
    from easyicu.research_agent.authority import pipeline_cache
    from easyicu.webserver import manuscript_repair

    raw = b"# synthetic tree\n"
    item = SimpleNamespace(
        as_posix=lambda: "nested/owner.py",
        is_file=lambda: True,
        is_symlink=lambda: False,
        relative_to=lambda root: PureWindowsPath("nested/owner.py"),
        read_bytes=lambda: raw,
    )
    root = SimpleNamespace(rglob=lambda pattern: [item])
    monkeypatch.setattr(
        pipeline_cache, "hash_file", lambda path: hashlib.sha256(raw).hexdigest()
    )
    expected = pipeline_cache._canonical_sha256(
        {"nested/owner.py": hashlib.sha256(raw).hexdigest()}
    )
    assert pipeline_cache._tree_sha256(root, [item]) == expected
    relative = b"nested/owner.py"
    digest = hashlib.sha256(
        len(relative).to_bytes(8, "big") + relative + hashlib.sha256(raw).digest()
    )
    assert manuscript_repair._source_fingerprint(root) == digest.hexdigest()


def test_revision_tree_digest_keeps_legacy_posix_bytes(tmp_path):
    from easyicu.webserver.manuscript_repair import _source_fingerprint

    for name in ("nested-/b.py", "nested/Z.py", "nested/a.py"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"# synthetic\n")
    expected = hashlib.sha256()
    for path in sorted(tmp_path.rglob("*")):
        if path.is_file():
            relative = str(path.relative_to(tmp_path)).encode()
            expected.update(len(relative).to_bytes(8, "big"))
            expected.update(relative)
            expected.update(hashlib.sha256(path.read_bytes()).digest())
    assert _source_fingerprint(tmp_path) == expected.hexdigest()


def test_concurrent_lockfile_creation_is_revalidated(tmp_path, host, monkeypatch):
    if not host:
        pytest.skip("Windows exclusive creation retry")
    path = tmp_path / "new-lock"
    original = compat.os.open
    calls = []

    def race(selected, flags, mode=0o600, **kwargs):
        calls.append(flags)
        if len(calls) == 1:
            path.write_bytes(b"other creator")
        return original(selected, flags, mode, **kwargs)

    monkeypatch.setattr(compat.os, "open", race)
    fd = compat.open_nofollow(path, os.O_RDWR | os.O_CREAT)
    try:
        assert os.read(fd, 100) == b"other creator"
    finally:
        os.close(fd)
    assert len(calls) == 2


def test_native_pinning_missing_fails_with_typed_reason(tmp_path, monkeypatch):
    import ctypes
    from contextlib import ExitStack

    monkeypatch.delattr(ctypes, "WinDLL", raising=False)
    with pytest.raises(compat.PlatformCapabilityError) as exc:
        with ExitStack() as stack:
            compat._pin_windows_directory(tmp_path, stack)
    assert exc.value.reason == "directory_pinning_unavailable"


def test_resource_rss_uses_bytes_on_macos(monkeypatch):
    monkeypatch.setattr(compat, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(
        compat,
        "resource",
        SimpleNamespace(
            RUSAGE_SELF=0,
            getrusage=lambda who: SimpleNamespace(ru_maxrss=2 * 1024 * 1024),
        ),
    )
    assert compat.peak_rss_mb() == 2.0


def test_posix_dataset_digest_keeps_legacy_relative_bytes(tmp_path):
    from easyicu.content_identity import data_path_fingerprint

    (tmp_path / "nested").mkdir()
    paths = [tmp_path / "nested" / "synthetic.csv", tmp_path / "root.json"]
    for path in paths:
        path.write_bytes(b"synthetic\n")
    expected = hashlib.sha256(str(tmp_path).encode())
    for path in sorted(paths, key=lambda item: str(item.relative_to(tmp_path))):
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        expected.update(f"{path.relative_to(tmp_path)}:{sha}\n".encode())
    assert data_path_fingerprint(tmp_path) == expected.hexdigest()
