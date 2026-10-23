"""Windows syscall emulation for Linux CI, scoped to EasyICU imports.

The Python interpreter and third-party libraries retain their real platform.
Only EasyICU sees os.name=nt/sys.platform=win32 and unavailable POSIX modules.
Native Win32 sharing/CRT semantics need the separate native tests on Windows.
"""

import builtins
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace


def emulate_windows(monkeypatch):
    if os.name == "nt":
        return
    import fcntl

    native_import = builtins.__import__
    facade = SimpleNamespace(**vars(os))
    facade.name = "nt"
    for name in (
        "O_NOFOLLOW",
        "O_DIRECTORY",
        "O_NONBLOCK",
        "O_CLOEXEC",
        "fchmod",
        "getuid",
        "geteuid",
        "getgid",
        "fork",
        "killpg",
        "setsid",
        "getpgrp",
        "uname",
    ):
        if hasattr(facade, name):
            delattr(facade, name)

    def open_file(path, flags, mode=0o600, **kwargs):
        if kwargs.get("dir_fd") is not None or Path(path).is_dir():
            raise PermissionError("Windows cannot open directory CRT descriptors")
        return os.open(path, flags, mode, **kwargs)

    def rename(source, target):
        os.link(source, target)
        os.unlink(source)

    def locking(fd, mode, count):
        assert count == 1
        assert os.lseek(fd, 0, os.SEEK_CUR) == 0
        fcntl.flock(fd, fcntl.LOCK_UN if mode == 2 else fcntl.LOCK_EX | fcntl.LOCK_NB)

    facade.open = open_file
    facade.rename = rename
    fake_sys = SimpleNamespace(**vars(sys))
    fake_sys.platform = "win32"
    fake_msvcrt = SimpleNamespace(
        LK_NBLCK=1,
        LK_UNLCK=2,
        locking=locking,
        get_osfhandle=lambda fd: fd,
    )

    def host_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = (globals or {}).get("__name__", "")
        if (module == "easyicu" or module.startswith("easyicu.")) and level == 0:
            if name == "os":
                return facade
            if name == "sys":
                return fake_sys
            if name in {"fcntl", "resource"}:
                raise ModuleNotFoundError(f"No Windows module named {name}")
            if name == "msvcrt":
                return fake_msvcrt
        return native_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", host_import)
    from easyicu.runtime import platform_compat as compat

    def pin(path, stack):
        info = os.lstat(path)
        compat._reject_reparse(info)
        if not stat.S_ISDIR(info.st_mode):
            raise compat.PlatformCapabilityError("directory_required")

    monkeypatch.setattr(compat, "_pin_windows_directory", pin)
    monkeypatch.setattr(
        compat,
        "_windows_handle_info",
        lambda fd: (
            0,
            *compat._identity(os.fstat(fd)),
        ),
    )


def smoke(*, health: bool):
    """Run in a fresh interpreter so cached EasyICU imports cannot mask defects."""
    from pytest import MonkeyPatch

    with MonkeyPatch.context() as patch:
        emulate_windows(patch)
        from easyicu.webserver.app import app
        from easyicu.runtime import platform_compat as compat
        from easyicu.research_agent.authority.evidence_store import EvidenceStore
        from easyicu.research_agent.authority.run_lock import (
            acquire_run_execution_lock,
            RunExecutionLockError,
        )
        from easyicu.webserver.deployment_lease import (
            acquire_single_process_lease,
            release_single_process_lease,
        )

        assert compat.is_windows()
        assert compat.fcntl is None and compat.resource is None
        assert app.title == "EasyICU"
        from easyicu.webserver.routes.system import router

        assert any(route.path == "/api/health" for route in router.routes)
        root = Path(os.environ["EASYICU_HOME"]) / "portability-smoke"
        root.mkdir(parents=True)
        store = EvidenceStore(root)
        record = store.register_json(
            kind="log",
            description="Synthetic smoke",
            payload={"synthetic": True},
            evidence_id="synthetic",
            filename="smoke.json",
        )
        assert EvidenceStore(root).records()[0].sha256 == record.sha256
        with acquire_run_execution_lock(workdir=root, run_id="synthetic"):
            try:
                acquire_run_execution_lock(workdir=root, run_id="synthetic")
            except RunExecutionLockError:
                pass
            else:
                raise AssertionError("concurrent run lock was admitted")
        acquire_single_process_lease(root / "deployment.lock")
        release_single_process_lease()
        if health:
            from fastapi.testclient import TestClient

            with TestClient(app) as client:
                response = client.get("/api/health")
                assert response.status_code == 200, response.text
                assert response.json()["status"] == "ok"
        print("windows import, evidence and locks: ok")


if __name__ == "__main__":
    smoke(health="--health" in sys.argv)
