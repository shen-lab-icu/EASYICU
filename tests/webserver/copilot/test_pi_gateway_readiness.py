"""Synthetic private sidecars only: no actual Pi sessions, Provider or host tools."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from easyicu.webserver.pi_copilot import gateway as owner
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.service import PiCopilotService


@pytest.fixture
def sidecar(tmp_path, monkeypatch):
    clients = []
    monkeypatch.setattr(owner, "STARTUP_READY_TIMEOUT_SECONDS", 0.6, raising=False)

    def build(*, delay=0.0, mode="ready"):
        app = tmp_path / f"app-{len(clients)}"
        (app / "src").mkdir(parents=True)
        # Python speaks only a tiny protocol fixture. It never imports Pi or
        # creates a research session, and gets no provider environment.
        (app / "src/main.mjs").write_text(f'''
import json,sys,time
mode={mode!r}
time.sleep({delay!r})
if mode == "exit": sys.exit(17)
seen=[]
for line in sys.stdin:
    request=json.loads(line)
    method=request["method"]
    seen.append(method)
    if mode == "silent" or (mode == "prompt_silent" and method == "session.prompt"): continue
    if mode == "invalid_json":
        print("not JSON", flush=True)
        continue
    if method == "runtime.status":
        result={{"gateway": "booting" if mode == "not_ready" else "ready"}}
    else:
        result={{"methods": seen.copy()}}
    print(json.dumps({{"protocol_version":request["protocol_version"],"kind":"response","request_id":request["request_id"],"ok":True,"result":result}}),flush=True)
''')
        gateway = owner.PiGatewayClient(
            app_dir=app, session_dir=app / "sessions", cwd=app / "workspace", environ={},
        )
        monkeypatch.setattr(gateway, "_node_binary", lambda: sys.executable)
        monkeypatch.setattr(gateway, "installation_status", lambda: dict.fromkeys([
            "node_available", "node_version_supported", "entrypoint_available",
            "dependency_installed", "lockfile_present", "runtime_integrity_verified",
        ], True))
        clients.append(gateway)
        return gateway

    yield build
    for gateway in clients:
        gateway.close()


def test_cold_sidecar_is_ready_before_short_state_deadline(sidecar):
    gateway = sidecar(delay=0.25)
    result = gateway.request("session.state", {"session_id": "synthetic"}, timeout=0.1)
    assert result["methods"] == ["runtime.status", "session.state"]
    diagnostic = gateway._startup_diagnostic
    assert diagnostic["status"] == "ready"
    assert 0.2 <= diagnostic["elapsed_seconds"] < 0.6
    result = gateway.request("session.state", {"session_id": "synthetic"}, timeout=0.1)
    assert result["methods"].count("runtime.status") == 1


def test_service_keeps_state_five_seconds_after_ready(sidecar, monkeypatch):
    gateway = sidecar()
    calls = []
    request = gateway._request_started

    def record(method, params=None, **kwargs):
        calls.append((method, kwargs.get("timeout")))
        return request(method, params, **kwargs)

    monkeypatch.setattr(gateway, "_request_started", record)
    PiCopilotService._ensure_open(
        SimpleNamespace(_conversation_gateway=lambda r: gateway),
        SimpleNamespace(session_id="synthetic"),
    )
    assert [m for m, _ in calls] == ["runtime.status", "session.state"]
    assert calls[1] == ("session.state", 5)


def test_silent_dead_gateway_has_one_bounded_attempt_no_action(sidecar, monkeypatch):
    gateway = sidecar(mode="silent")
    monkeypatch.setattr(owner, "STARTUP_READY_TIMEOUT_SECONDS", 0.2)
    writes = []
    write = gateway._write
    monkeypatch.setattr(gateway, "_write", lambda payload, **kwargs: (writes.append(payload["method"]), write(payload, **kwargs))[-1])
    start = time.monotonic()
    with pytest.raises(PiCopilotError) as caught:
        gateway.request("session.prompt", {"session_id": "synthetic", "message": "one turn"}, timeout=5)
    assert caught.value.code == "pi_gateway_startup_timeout"
    assert caught.value.details["phase"] == "startup_readiness"
    assert time.monotonic() - start < 0.5
    assert writes == ["runtime.status"]
    assert gateway._pending == {}
    assert gateway._ready_process is None


@pytest.mark.parametrize("mode,code", [
    ("exit", "pi_gateway_exited"),
    ("invalid_json", "pi_protocol_invalid_json"),
    ("not_ready", "pi_gateway_not_ready"),
])
def test_startup_fault_propagates_without_action(sidecar, monkeypatch, mode, code):
    gateway = sidecar(delay=0.05, mode=mode)
    writes = []
    write = gateway._write
    monkeypatch.setattr(gateway, "_write", lambda payload, **kwargs: (writes.append(payload["method"]), write(payload, **kwargs))[-1])
    with pytest.raises(PiCopilotError) as caught:
        gateway.request("session.prompt", {"session_id": "synthetic", "message": "one turn"}, timeout=5)
    assert caught.value.code == code
    assert caught.value.details["phase"] == "startup_readiness"
    assert writes == ["runtime.status"]
    assert gateway._ready_process is None
    if mode == "exit":
        assert caught.value.details["return_code"] == 17


def test_concurrent_first_requests_share_ready_receipt(sidecar):
    gateway = sidecar(delay=0.15)
    barrier = threading.Barrier(3)

    def read():
        barrier.wait()
        return gateway.request("session.state", {"session_id": "synthetic"}, timeout=0.1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(read) for _ in range(2)]
        barrier.wait()
        results = [f.result(timeout=1.5) for f in futures]
    assert all(r["methods"][0] == "runtime.status" for r in results)
    assert max(r["methods"].count("runtime.status") for r in results) == 1


def test_waiting_for_startup_lock_is_bounded(sidecar, monkeypatch):
    gateway = sidecar()
    monkeypatch.setattr(owner, "STARTUP_READY_TIMEOUT_SECONDS", 0.1)
    gateway._startup_lock.acquire()
    start = time.monotonic()
    try:
        with pytest.raises(PiCopilotError) as caught:
            gateway.request("session.state", {"session_id": "synthetic"}, timeout=5)
    finally:
        gateway._startup_lock.release()
    assert caught.value.code == "pi_gateway_startup_timeout"
    assert caught.value.details["phase"] == "startup_lock"
    assert time.monotonic() - start < 0.4
    assert gateway._process is None


def test_replaced_process_gets_new_handshake(sidecar):
    gateway = sidecar()
    first = gateway.request("session.state", {"session_id": "synthetic"}, timeout=0.1)
    old_process = gateway._process
    gateway.close()
    second = gateway.request("session.state", {"session_id": "synthetic"}, timeout=0.1)
    assert gateway._process is not old_process
    assert first["methods"] == second["methods"] == ["runtime.status", "session.state"]


def test_prompt_timeout_does_not_resubmit_action(sidecar, monkeypatch):
    gateway = sidecar(mode="prompt_silent")
    recovered = []
    writes = []
    write = gateway._write
    monkeypatch.setattr(gateway, "_write", lambda payload, **kwargs: (writes.append(payload["method"]), write(payload, **kwargs))[-1])
    monkeypatch.setattr(gateway, "_recover_timed_out_prompt", recovered.append)
    with pytest.raises(PiCopilotError) as caught:
        gateway.request("session.prompt", {"session_id": "synthetic", "message": "one turn"}, timeout=0.1)
    assert caught.value.code == "pi_gateway_timeout"
    assert writes == ["runtime.status", "session.prompt"]
    assert recovered == ["synthetic"]


class MemoryProcess:
    def __init__(self, callback):
        self.pid = 101
        self.stdin = self
        self.dead = False
        self.callback = callback
        self.methods = []

    def poll(self):
        return 0 if self.dead else None

    def write(self, raw):
        payload = json.loads(raw)
        self.methods.append(payload["method"])
        self.callback(payload)

    def flush(self):
        pass

    def close(self):
        self.dead = True

    def wait(self, timeout):
        return 0


@pytest.fixture
def memory_gateway(tmp_path, monkeypatch):
    gateway = owner.PiGatewayClient(
        app_dir=tmp_path, session_dir=tmp_path / "sessions", environ={},
    )
    # The memory fixtures replace only OS launch/write; exercise actual owner
    # locks, transport, ready identity, deadlines, cleanup and close methods.
    monkeypatch.setattr(gateway, "_spawn_process", lambda: None)
    monkeypatch.setattr(gateway, "_write_before_deadline", lambda process, encoded, deadline: process.stdin.write(encoded))
    yield gateway
    gateway.close()


@pytest.mark.parametrize("send_seconds,response_seconds,accepted", [
    (12, 14, False), (12, 2, True), (16, 0, False),
])
def test_absolute_ready_deadline_counts_send_and_rechecks_success(
    memory_gateway, monkeypatch, send_seconds, response_seconds, accepted,
):
    gateway = memory_gateway
    clock = SimpleNamespace(now=0.0)
    waits = []
    monkeypatch.setattr(owner, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(owner, "STARTUP_READY_TIMEOUT_SECONDS", 15.0)

    class Pending:
        def __init__(self, **kwargs):
            self.process = kwargs.get("process")
            self.error = None
            self.result = {"gateway": "ready"}
            self.done = SimpleNamespace(wait=self.wait)

        def wait(self, timeout):
            waits.append(timeout)
            # A late successful callback must not overrule the deadline.
            clock.now += response_seconds
            return True

    monkeypatch.setattr(owner, "_PendingRequest", Pending)

    def send(payload):
        clock.now += send_seconds

    process = MemoryProcess(send)
    gateway._process = process
    if accepted:
        assert gateway._start() is process
        assert gateway._ready_process is process
    else:
        with pytest.raises(PiCopilotError) as caught:
            gateway._start()
        assert caught.value.code == "pi_gateway_startup_timeout"
        assert gateway._ready_process is None
    assert waits == ([] if send_seconds > 15 else [3.0])
    assert process.methods == ["runtime.status"]
    assert gateway._pending == {}


def test_saturated_pipe_ready_write_is_bounded_and_never_sends_later(
    tmp_path, monkeypatch,
):
    gateway = owner.PiGatewayClient(app_dir=tmp_path, session_dir=tmp_path, environ={})
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    filled = 0
    try:
        while True:
            filled += os.write(write_fd, b"x" * 4096)
    except BlockingIOError:
        pass
    os.set_blocking(write_fd, True)
    process = MemoryProcess(lambda payload: None)
    process.stdin = os.fdopen(write_fd, "w")
    gateway._process = process
    monkeypatch.setattr(gateway, "_spawn_process", lambda: None)
    monkeypatch.setattr(owner, "STARTUP_READY_TIMEOUT_SECONDS", 0.1)
    try:
        start = time.monotonic()
        with pytest.raises(PiCopilotError) as caught:
            gateway.request("session.prompt", {"message": "once"}, timeout=5)
        assert caught.value.code == "pi_gateway_startup_timeout"
        assert time.monotonic() - start < 0.3
        assert os.get_blocking(write_fd) is True
        os.set_blocking(read_fd, False)
        drained = b""
        while True:
            try:
                drained += os.read(read_fd, 65536)
            except BlockingIOError:
                break
        assert drained == b"x" * filled
        # The call finished synchronously: no orphan sender wakes after drain.
        time.sleep(0.03)
        with pytest.raises(BlockingIOError):
            os.read(read_fd, 65536)
        assert gateway._pending == {}
    finally:
        gateway.close()
        os.close(read_fd)


@pytest.mark.parametrize("lock_name", ["_state_lock", "_write_lock"])
def test_transport_lock_queue_uses_ready_deadline(memory_gateway, monkeypatch, lock_name):
    gateway = memory_gateway
    gateway._process = MemoryProcess(lambda payload: None)
    monkeypatch.setattr(owner, "STARTUP_READY_TIMEOUT_SECONDS", 0.1)
    entered, release = threading.Event(), threading.Event()

    def hold():
        with getattr(gateway, lock_name):
            entered.set()
            assert release.wait(1)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(hold)
        assert entered.wait(1)
        try:
            start = time.monotonic()
            with pytest.raises(PiCopilotError) as caught:
                gateway._start()
            assert caught.value.code == "pi_gateway_startup_timeout"
            assert time.monotonic() - start < 0.3
        finally:
            release.set()
        future.result(timeout=1)
    assert gateway._pending == {}


def test_ready_queue_times_out_while_ordinary_pipe_writer_is_blocked(
    memory_gateway, monkeypatch,
):
    gateway = memory_gateway
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(owner, "STARTUP_READY_TIMEOUT_SECONDS", 0.1)

    def blocked_write(payload):
        entered.set()
        assert release.wait(1)

    process = MemoryProcess(blocked_write)
    gateway._process = process
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(gateway._write, {"method": "synthetic.response"})
        assert entered.wait(1)
        try:
            start = time.monotonic()
            with pytest.raises(PiCopilotError) as caught:
                gateway._start()
            assert caught.value.code == "pi_gateway_startup_timeout"
            assert time.monotonic() - start < 0.3
            assert process.methods == ["synthetic.response"]
        finally:
            release.set()
        future.result(timeout=1)


def test_expired_pending_cleanup_does_not_wait_again_on_state_lock(
    memory_gateway, monkeypatch,
):
    gateway = memory_gateway
    process = MemoryProcess(lambda payload: None)
    gateway._process = process
    monkeypatch.setattr(owner, "STARTUP_READY_TIMEOUT_SECONDS", 0.08)
    entered, release = threading.Event(), threading.Event()
    write = gateway._write

    def hold():
        with gateway._state_lock:
            entered.set()
            assert release.wait(1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        holders = []

        def block_after_registration(payload, **kwargs):
            assert gateway._pending
            holders.append(pool.submit(hold))
            assert entered.wait(1)
            return write(payload, **kwargs)

        monkeypatch.setattr(gateway, "_write", block_after_registration)
        future = pool.submit(gateway.request, "session.prompt", {"message": "once"})
        try:
            with pytest.raises(PiCopilotError) as caught:
                future.result(timeout=0.25)
            assert caught.value.code == "pi_gateway_startup_timeout"
            assert not release.is_set()
            assert gateway._pending == {}
            assert process.methods == []
        finally:
            release.set()
        for holder in holders:
            holder.result(timeout=1)
    assert gateway._ready_process is None


def test_prompt_cannot_cross_ready_process_generation(memory_gateway, monkeypatch):
    gateway = memory_gateway
    pause_a, resume_a, b_handshake_sent = threading.Event(), threading.Event(), threading.Event()
    a = MemoryProcess(lambda payload: None)
    gateway._process = gateway._ready_process = a
    status_ids = []

    def b_write(payload):
        if payload["method"] == "runtime.status":
            status_ids.append(payload["request_id"])
            b_handshake_sent.set()
        else:
            row = gateway._pending[payload["request_id"]]
            row.result = {"ready_at_write": gateway._ready_process is b}
            row.done.set()

    b = MemoryProcess(b_write)

    def spawn():
        if gateway._process is None or gateway._process.poll() is not None:
            gateway._process, gateway._ready_process = b, None

    monkeypatch.setattr(gateway, "_spawn_process", spawn)
    transport = gateway._request_started

    def pause(method, params=None, **kwargs):
        if method == "session.prompt":
            pause_a.set()
            assert resume_a.wait(1)
        return transport(method, params, **kwargs)

    monkeypatch.setattr(gateway, "_request_started", pause)
    with ThreadPoolExecutor(max_workers=2) as pool:
        old = pool.submit(gateway.request, "session.prompt", {"message": "once"}, timeout=1)
        assert pause_a.wait(1)
        gateway.close()
        new = pool.submit(gateway.request, "session.state", {}, timeout=1)
        assert b_handshake_sent.wait(1)
        resume_a.set()
        try:
            with pytest.raises(PiCopilotError) as caught:
                old.result(timeout=1)
            assert caught.value.code == "pi_gateway_process_changed"
            assert b.methods == ["runtime.status"]
        finally:
            row = gateway._pending[status_ids[0]]
            row.result = {"gateway": "ready"}
            row.done.set()
        assert new.result(timeout=1) == {"ready_at_write": True}
    assert b.methods == ["runtime.status", "session.state"]
    assert a.methods == []
    assert gateway._pending == {}


def test_late_old_reader_eof_cannot_fail_replacement_request(memory_gateway):
    gateway = memory_gateway
    entered, exit_reader = threading.Event(), threading.Event()
    old = MemoryProcess(lambda payload: None)
    new = MemoryProcess(lambda payload: None)

    def stdout():
        entered.set()
        assert exit_reader.wait(1)
        yield from ()

    old.stdout = stdout()
    gateway._process = old
    old_pending = owner._PendingRequest(process=old)
    new_pending = owner._PendingRequest(process=new)
    gateway._pending["old"] = old_pending
    with ThreadPoolExecutor(max_workers=1) as pool:
        reader = pool.submit(gateway._read_stdout)
        assert entered.wait(1)
        with gateway._state_lock:
            gateway._process = gateway._ready_process = new
        with gateway._pending_lock:
            gateway._pending["new"] = new_pending
        exit_reader.set()
        reader.result(timeout=1)
    assert old_pending.done.is_set()
    assert old_pending.error.code == "pi_gateway_exited"
    assert not new_pending.done.is_set()
    assert new_pending.error is None
    assert gateway._ready_process is new


def test_close_releases_real_full_pipe_business_writer(tmp_path, monkeypatch):
    gateway = owner.PiGatewayClient(app_dir=tmp_path, session_dir=tmp_path, environ={})
    # Private Python fixture holds its stdin open without reading. No Pi,
    # Provider, host tools, real session, credentials or patient data involved.
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; print('holding', flush=True); time.sleep(30)"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, env={},
    )
    assert process.stdout.readline().strip() == "holding"
    fd = process.stdin.fileno()
    os.set_blocking(fd, False)
    try:
        while True:
            os.write(fd, b"x" * 4096)
    except BlockingIOError:
        pass
    os.set_blocking(fd, True)
    gateway._process = gateway._ready_process = process
    entered = threading.Event()
    transport = gateway._write

    def record(payload, **kwargs):
        entered.set()
        return transport(payload, **kwargs)

    monkeypatch.setattr(gateway, "_write", record)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(gateway.request, "session.prompt", {"message": "synthetic once"})
        try:
            assert entered.wait(1)
            # A complete, saturated pipe forces this actual synchronous write
            # to retain the write lock until close terminates the private child.
            limit = time.monotonic() + 1
            while not gateway._write_lock.locked() and time.monotonic() < limit:
                time.sleep(0.001)
            assert gateway._write_lock.locked() and not writer.done()
            closer = pool.submit(gateway.close)
            closer.result(timeout=3)
            assert process.poll() is not None
            with pytest.raises(PiCopilotError) as caught:
                writer.result(timeout=1)
            assert caught.value.code in {"pi_gateway_pipe_closed", "pi_gateway_closed"}
            assert gateway._pending == {}
            assert gateway._process is None
        finally:
            # Ensure a red baseline can fail without leaking its blocked writer.
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)
    process.stdout.close()


def test_close_after_send_identity_check_still_pins_old_process(memory_gateway):
    gateway = memory_gateway
    at_write, release_write = threading.Event(), threading.Event()
    sent = []

    class PausedStream(io.StringIO):
        def write(self, raw):
            at_write.set()
            assert release_write.wait(1)
            result = super().write(raw)
            sent.append(raw)
            return result

    class ExitingProcess(MemoryProcess):
        def wait(self, timeout):
            self.dead = True
            return 0

    old = ExitingProcess(lambda payload: None)
    old.stdin = PausedStream()
    new = MemoryProcess(lambda payload: None)
    gateway._process = gateway._ready_process = old
    with ThreadPoolExecutor(max_workers=1) as pool:
        writer = pool.submit(gateway.request, "session.prompt", {"message": "once"})
        assert at_write.wait(1)
        try:
            # This is after _write's state-lock check, a later interleaving than
            # the existing A-ready -> close -> B-before-send regression.
            gateway.close()
            gateway._process, gateway._ready_process = new, None
            assert old.stdin.closed
        finally:
            release_write.set()
        with pytest.raises(PiCopilotError) as caught:
            writer.result(timeout=1)
        assert caught.value.code == "pi_gateway_pipe_closed"
    assert sent == []
    assert new.methods == []
    assert gateway._pending == {}
