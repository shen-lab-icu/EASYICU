"""Process fencing for trusted local execution, independent of scientific grants.

A parent holds the stable OS lock before mutating a run, then hands it to an
identified guardian. The durable nonce fences the lock-transfer interval. Only
the guardian holds the lock while a worker executes. Missing cleanup proof is
an unresolved execution, never permission to reuse the workspace or kill a PID.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import stat
import time
import uuid
from typing import Any

try:
    import psutil
except ImportError:  # Core/Docker installs do not require the webapp extra.
    psutil = None

from easyicu.runtime.platform_compat import SecureDirectory, chmod_private_fd, lock_exclusive, unlock_file

HOST_EXECUTION_SCHEMA = "easyicu.host-execution-lease/1"
_SCHEMA = HOST_EXECUTION_SCHEMA
_RECEIPT = "execution.json"
_PROOF = "worker-cleanup.json"
_STOP = "stop.json"
_STATES = {"prepared", "guardian", "running", "quiesced", "unresolved"}


class HostExecutionLeaseError(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


def _fail(reason: str) -> None:
    raise HostExecutionLeaseError("host_execution_" + reason)


def _require_psutil() -> None:
    if psutil is None:
        _fail("identity_unavailable")


def _nonce(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        _fail("nonce_invalid")
    return value


def _sha(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        _fail("script_binding_invalid")
    return value


def _identity(pid: int | None = None) -> dict[str, Any]:
    _require_psutil()
    process = psutil.Process(os.getpid() if pid is None else pid)
    return {"pid": process.pid, "created": process.create_time()}


def _valid_identity(value: Any) -> bool:
    return (isinstance(value, dict) and set(value) == {"pid", "created"}
            and type(value["pid"]) is int and value["pid"] > 0
            and type(value["created"]) in {int, float}
            and math.isfinite(value["created"]) and value["created"] > 0)


def _process(identity: dict[str, Any] | None) -> psutil.Process | None:
    if identity is None:
        return None
    _require_psutil()
    if not _valid_identity(identity):
        _fail("identity_invalid")
    try:
        process = psutil.Process(identity["pid"])
        if process.create_time() != identity["created"] or process.status() == psutil.STATUS_ZOMBIE:
            return None
        return process
    except psutil.NoSuchProcess:
        return None
    except psutil.AccessDenied:
        _fail("identity_unverifiable")


def _read(directory: SecureDirectory, name: str) -> dict[str, Any] | None:
    try:
        fd = directory.open(name, os.O_RDONLY)
    except FileNotFoundError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 16384:
            _fail("receipt_invalid")
        raw = os.read(fd, 16385)
    finally:
        os.close(fd)
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError):
        _fail("receipt_invalid")
    if not isinstance(value, dict):
        _fail("receipt_invalid")
    return value


def _read_receipt(directory: SecureDirectory) -> dict[str, Any] | None:
    value = _read(directory, _RECEIPT)
    if value is None:
        return None
    if (value.get("schema") != _SCHEMA or not isinstance(value.get("state"), str)
            or value["state"] not in _STATES
            or not _valid_identity(value.get("parent"))
            or type(value.get("spawn_requested")) is not bool
            or type(value.get("parent_complete")) is not bool):
        _fail("receipt_invalid")
    _nonce(value.get("nonce"))
    if value.get("script_sha256") is not None:
        _sha(value["script_sha256"])
    for role in ("guardian", "worker"):
        if value.get(role) is not None and not _valid_identity(value[role]):
            _fail("receipt_invalid")
    return value


def _write(directory: SecureDirectory, name: str, value: dict[str, Any]) -> None:
    directory.assert_still_selected()
    temporary = "." + name + "." + uuid.uuid4().hex
    fd = directory.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        chmod_private_fd(fd)
        raw = memoryview(json.dumps(value, sort_keys=True, allow_nan=False).encode())
        while raw:
            written = os.write(fd, raw)
            if written <= 0:
                _fail("receipt_write_failed")
            raw = raw[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    directory.replace(temporary, name)
    directory.fsync()


class _Lease:
    def __init__(self, path: Path, *, timeout: float = 0.0) -> None:
        self.path = Path(path).absolute()
        self._directory = SecureDirectory(self.path)
        self._fd: int | None = None
        self._locked = False
        try:
            self._fd = self._directory.open("execution.lock", os.O_RDWR | os.O_CREAT, 0o600)
            chmod_private_fd(self._fd)
            identity = os.fstat(self._fd)
            if not stat.S_ISREG(identity.st_mode):
                _fail("lock_invalid")
            deadline = time.monotonic() + min(max(0.0, timeout), 30.0)
            while True:
                try:
                    lock_exclusive(self._fd, blocking=False)
                    self._locked = True
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        _fail("active")
                    time.sleep(0.02)
            selected = self._directory.stat("execution.lock")
            if (identity.st_dev, identity.st_ino) != (selected.st_dev, selected.st_ino):
                _fail("lock_changed")
            self._directory.assert_still_selected()
        except BaseException:
            self.release()
            raise

    def release(self) -> None:
        if self._fd is not None:
            try:
                if self._locked:
                    unlock_file(self._fd)
            finally:
                os.close(self._fd)
                self._fd = None
                self._locked = False
        self._directory.close()

    def _owned(self) -> dict[str, Any]:
        if not self._locked:
            _fail("lease_released")
        record = _read_receipt(self._directory)
        if record is None or record["nonce"] != self.nonce:
            _fail("nonce_changed")
        return record

    @property
    def script_sha256(self) -> str:
        return _sha(self._owned().get("script_sha256"))


class HostLaunchLease(_Lease):
    nonce: str

    def bind_script(self, sha256: str) -> None:
        record = self._owned()
        digest = _sha(sha256)
        if record["state"] != "prepared" or record["spawn_requested"]:
            _fail("already_spawning")
        if record["script_sha256"] not in (None, digest):
            _fail("script_binding_changed")
        record["script_sha256"] = digest
        _write(self._directory, _RECEIPT, record)

    def mark_spawning(self) -> None:
        record = self._owned()
        _sha(record.get("script_sha256"))
        if record["state"] != "prepared" or record["spawn_requested"]:
            _fail("already_spawning")
        record["spawn_requested"] = True
        _write(self._directory, _RECEIPT, record)

    def bind_guardian(self, pid: int) -> None:
        record = self._owned()
        if record["state"] != "prepared" or not record["spawn_requested"] or record["guardian"] is not None:
            _fail("guardian_binding_invalid")
        guardian = psutil.Process(pid)
        if guardian.ppid() != os.getpid() or pid == os.getpid():
            _fail("guardian_binding_invalid")
        record["guardian"] = _identity(pid)
        _write(self._directory, _RECEIPT, record)

    def abort(self) -> None:
        record = self._owned()
        if record["state"] != "prepared" or record["spawn_requested"] or record["guardian"] is not None:
            _fail("abort_ambiguous")
        record.update(state="quiesced", reason="launch_aborted", parent_complete=True)
        _write(self._directory, _RECEIPT, record)


def _prepared_without_live_owner(record: dict[str, Any]) -> bool:
    """A late guardian cannot admit a child after its bound parent has died.

    Check both exact identities. No PID signalling or receipt mutation is
    needed; the next parent will replace the nonce while holding this lock.
    """
    return (record["state"] == "prepared" and record.get("worker") is None
            and _process(record["parent"]) is None
            and _process(record.get("guardian")) is None)


def prepare_host_execution(lease_dir: Path, *, script_sha256: str | None = None) -> HostLaunchLease:
    _require_psutil()
    path = Path(lease_dir).absolute()
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    lease = HostLaunchLease(path)
    try:
        prior = _read_receipt(lease._directory)
        if prior is not None and prior["state"] == "quiesced" and not prior["parent_complete"]:
            if _process(prior["parent"]) is not None:
                _fail("active")
        if prior is not None and prior["state"] != "quiesced":
            # A prepared guardian cannot start a worker before claiming the
            # lock and publishing guardian state. Revoking its nonce is safe
            # only after both known launching processes have disappeared.
            if not _prepared_without_live_owner(prior):
                _fail("unresolved")
        lease.nonce = uuid.uuid4().hex
        _write(lease._directory, _RECEIPT, {
            "schema": _SCHEMA, "nonce": lease.nonce, "state": "prepared",
            "script_sha256": _sha(script_sha256) if script_sha256 is not None else None,
            "parent": _identity(), "guardian": None, "worker": None,
            "spawn_requested": False, "parent_complete": False, "reason": None,
        })
        return lease
    except BaseException:
        lease.release()
        raise


class HostGuardianLease(_Lease):
    nonce: str

    def record_worker(self, pid: int) -> None:
        record = self._owned()
        if record["state"] != "guardian" or record["worker"] is not None:
            _fail("worker_binding_invalid")
        worker = psutil.Process(pid)
        if worker.ppid() != os.getpid() or pid == os.getpid():
            _fail("worker_binding_invalid")
        record.update(state="running", worker=_identity(pid))
        _write(self._directory, _RECEIPT, record)

    def parent_alive(self) -> bool:
        return _process(self._owned()["parent"]) is not None

    def stop_requested(self) -> bool:
        self._owned()
        value = _read(self._directory, _STOP)
        return value is not None and value.get("nonce") == self.nonce

    def worker_cleanup_confirmed(self) -> bool:
        return self.worker_completion_status() is not None

    def worker_completion_status(self) -> int | None:
        record = self._owned()
        proof = _read(self._directory, _PROOF)
        if (proof is not None and proof.get("nonce") == self.nonce
                and record.get("worker") is not None and proof.get("worker") == record["worker"]
                and type(proof.get("exitcode")) is int and 0 <= proof["exitcode"] <= 255):
            return proof["exitcode"]
        return None

    def terminate_worker_tree(self) -> bool:
        record = self._owned()
        worker = _process(record.get("worker"))
        if worker is None:
            # Even a published logical result is not proof that threads could
            # not spawn descendants between publication and abrupt death.
            return False
        pending, owned, seen = [worker], [], set()
        complete = True
        while pending:
            child = pending.pop()
            try:
                identity = (child.pid, child.create_time())
                if identity in seen:
                    continue
                seen.add(identity)
                # Freeze each spawning parent before enumerating children;
                # session/process-group changes do not escape this traversal.
                child.suspend()
                owned.append(child)
                pending.extend(child.children())
            except psutil.NoSuchProcess:
                complete = False  # A vanished parent may have orphaned children.
            except psutil.AccessDenied:
                complete = False
        for child in reversed(owned):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                complete = False
        _, alive = psutil.wait_procs(owned, timeout=2.0)
        for child in alive:
            try:
                if child.status() != psutil.STATUS_ZOMBIE:
                    complete = False
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                complete = False
        return complete

    def mark_quiesced(self, reason: str, *, cleanup_proven: bool) -> None:
        record = self._owned()
        if not cleanup_proven or _process(record.get("worker")) is not None:
            _fail("cleanup_unproven")
        self._finish(record, "quiesced", reason)

    def mark_unresolved(self, reason: str) -> None:
        self._finish(self._owned(), "unresolved", reason)

    def _finish(self, record: dict[str, Any], state: str, reason: str) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,95}", reason):
            _fail("reason_invalid")
        record.update(state=state, reason=reason)
        _write(self._directory, _RECEIPT, record)


def claim_host_execution(lease_dir: Path, nonce: str, *, timeout: float = 10.0) -> HostGuardianLease:
    _require_psutil()
    nonce = _nonce(nonce)
    lease = HostGuardianLease(lease_dir, timeout=timeout)
    lease.nonce = nonce
    try:
        record = lease._owned()
        if (record["state"] != "prepared" or not record["spawn_requested"]
                or record.get("guardian") != _identity()):
            _fail("guardian_binding_invalid")
        _sha(record.get("script_sha256"))
        if _process(record["parent"]) is None:
            # This guardian has not created a worker, so its own launch can be
            # safely settled without guessing about any process tree.
            lease.mark_quiesced("parent_lost_before_worker", cleanup_proven=True)
            _fail("parent_lost")
        record["state"] = "guardian"
        _write(lease._directory, _RECEIPT, record)
        return lease
    except BaseException:
        lease.release()
        raise


def verify_host_worker(lease_dir: Path, nonce: str, *, timeout: float = 10.0) -> str:
    nonce = _nonce(nonce)
    identity = _identity()
    deadline = time.monotonic() + min(max(0.0, timeout), 30.0)
    with SecureDirectory(Path(lease_dir)) as directory:
        while True:
            record = _read_receipt(directory)
            if record is None or record["nonce"] != nonce:
                _fail("nonce_changed")
            if _process(record.get("guardian")) is None or _process(record["parent"]) is None:
                _fail("parent_lost")
            if record["state"] == "running":
                if record.get("worker") != identity:
                    _fail("worker_binding_invalid")
                # A live PID and receipt are insufficient: execution also
                # requires the guardian's OS lease still to fence the run.
                try:
                    unlocked = _Lease(Path(lease_dir))
                except HostExecutionLeaseError as exc:
                    if exc.reason_code != "host_execution_active":
                        raise
                else:
                    unlocked.release()
                    _fail("guardian_lock_missing")
                return _sha(record.get("script_sha256"))
            if record["state"] != "guardian" or time.monotonic() >= deadline:
                _fail("worker_not_admitted")
            time.sleep(0.02)


def complete_host_worker_cleanup(lease_dir: Path, nonce: str, *, exitcode: int = 0) -> None:
    nonce = _nonce(nonce)
    if type(exitcode) is not int or not 0 <= exitcode <= 255:
        _fail("worker_exitcode_invalid")
    with SecureDirectory(Path(lease_dir)) as directory:
        record = _read_receipt(directory)
        if (record is None or record["nonce"] != nonce or record["state"] != "running"
                or record.get("worker") != _identity()):
            _fail("worker_binding_invalid")
        _write(directory, _PROOF, {"nonce": nonce, "worker": record["worker"], "exitcode": exitcode})


def request_host_execution_stop(lease_dir: Path, nonce: str) -> None:
    nonce = _nonce(nonce)
    with SecureDirectory(Path(lease_dir)) as directory:
        record = _read_receipt(directory)
        if record is None or record["nonce"] != nonce:
            _fail("nonce_changed")
        _write(directory, _STOP, {"nonce": nonce})


def read_host_execution_receipt(lease_dir: Path) -> dict[str, Any] | None:
    """Read the atomic nonce-bound lifecycle receipt, without claiming quiescence."""
    path = Path(lease_dir)
    if not path.exists():
        return None
    with SecureDirectory(path) as directory:
        return _read_receipt(directory)


def acknowledge_host_execution_parent(lease_dir: Path, nonce: str) -> bool:
    """Release the parent's post-worker output-collection fence, never an orphan."""
    nonce = _nonce(nonce)
    try:
        # A verifier briefly takes this same lock. Do not permanently strand
        # the parent's tail acknowledgement on that harmless contention.
        lease = _Lease(Path(lease_dir), timeout=2.0)
    except HostExecutionLeaseError as exc:
        if exc.reason_code == "host_execution_active":
            return False
        raise
    try:
        record = _read_receipt(lease._directory)
        if record is None or record["nonce"] != nonce:
            _fail("nonce_changed")
        if record["state"] != "quiesced" or record["parent"] != _identity():
            return False
        record["parent_complete"] = True
        _write(lease._directory, _RECEIPT, record)
        return True
    finally:
        lease.release()


def assert_host_execution_quiescent(
    lease_dir: Path, *, allow_missing: bool = True, wait_timeout: float = 0.0,
    allow_current_parent: bool = False,
) -> None:
    path = Path(lease_dir)
    deadline = time.monotonic() + min(max(0.0, wait_timeout), 30.0)
    while True:
        lease = None
        try:
            if path.is_symlink():
                _fail("receipt_invalid")
            if not path.exists():
                if allow_missing:
                    return
                _fail("receipt_missing")
            lease = _Lease(path)
            record = _read_receipt(lease._directory)
            if record is None:
                if allow_missing:
                    return
                _fail("receipt_missing")
            if record["state"] == "quiesced":
                if (record["parent_complete"] or _process(record["parent"]) is None
                        or (allow_current_parent and record["parent"] == _identity())):
                    return
                _fail("active")
            if _prepared_without_live_owner(record):
                return
            _fail("unresolved")
        except HostExecutionLeaseError as exc:
            if exc.reason_code not in {"host_execution_active", "host_execution_unresolved"} or time.monotonic() >= deadline:
                raise
        finally:
            if lease is not None:
                lease.release()
        time.sleep(0.02)
