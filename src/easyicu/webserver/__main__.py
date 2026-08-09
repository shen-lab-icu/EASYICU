"""Command-line entry point for the native EasyICU FastAPI WebApp."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from typing import Sequence
import urllib.request

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _runtime_dir() -> Path:
    """Per-user runtime directory for the PID and log files.

    Prefers $XDG_RUNTIME_DIR (already per-user and 0700). The /tmp fallback is
    namespaced by UID so two users on a shared host cannot collide in — or
    overwrite each other's — easyicu_webserver.pid.
    """

    override = os.environ.get("EASYICU_RUNTIME_DIR")
    if override:
        runtime_dir = Path(override).expanduser()
    else:
        xdg = os.environ.get("XDG_RUNTIME_DIR")
        base = Path(xdg).expanduser() if xdg else Path(tempfile.gettempdir())
        suffix = f"easyicu-{os.getuid()}" if hasattr(os, "getuid") else "easyicu"
        runtime_dir = base / suffix
    runtime_dir.mkdir(parents=True, exist_ok=True)
    try:
        runtime_dir.chmod(0o700)
    except OSError:
        pass
    return runtime_dir


def _pid_file() -> Path:
    return _runtime_dir() / "easyicu_webserver.pid"


def _log_file() -> Path:
    return _runtime_dir() / "easyicu_webserver.log"


def _probe_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


def _is_loopback_bind(host: str) -> bool:
    if str(host or "").strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(str(host).strip()).is_loopback
    except ValueError:
        return False


def _health_url(host: str, port: int) -> str:
    return f"http://{_probe_host(host)}:{port}/api/catalog"


def _health_check(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(_health_url(host, port), timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def _uvicorn_cmd(host: str, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "easyicu.webserver.app:app",
        "--host",
        host,
        "--port",
        str(port),
    ]


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("EASYICU_VERBOSE", "0")
    return env


def _process_create_time(pid: int) -> float | None:
    """Return the process start time, used to tell a recycled PID apart."""

    try:
        import psutil
    except ImportError:
        return None
    try:
        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _write_pid_record(pid_path: Path, pid: int, port: int) -> None:
    """Write the PID plus enough identity to validate it later, atomically."""

    record = {
        "pid": pid,
        "port": port,
        "create_time": _process_create_time(pid),
        "cmdline_marker": "easyicu.webserver.app:app",
    }
    tmp_path = pid_path.with_name(pid_path.name + f".{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(record), encoding="utf-8")
    os.replace(tmp_path, pid_path)


def _read_pid_record(pid_path: Path) -> dict:
    """Read the PID record, tolerating the legacy bare-integer format."""

    try:
        raw = pid_path.read_text(encoding="utf-8").strip()
    except OSError:
        return {}
    if not raw:
        return {}
    try:
        record = json.loads(raw)
    except ValueError:
        record = raw
    if isinstance(record, dict):
        return record
    # Legacy format: the file held a bare PID integer. Keep reading it — the
    # identity check still verifies the cmdline, it just has no start time.
    try:
        return {"pid": int(record)}
    except (TypeError, ValueError):
        return {}


def _pid_matches_record(pid: int, record: dict) -> bool:
    """True when the live process really is the server this record describes."""

    try:
        import psutil
    except ImportError:
        # Without psutil we cannot verify identity. Signalling an unverified
        # PID risks killing an unrelated process, so decline instead — the
        # port-based sweep below still stops a real server.
        return False
    try:
        proc = psutil.Process(pid)
        cmdline = " ".join(proc.cmdline() or [])
        live_create_time = float(proc.create_time())
    except Exception:
        return False

    if not _is_easyicu_webserver_cmdline(cmdline):
        return False
    recorded_create_time = record.get("create_time")
    if recorded_create_time is not None:
        # Sub-second tolerance: psutil rounds differently across platforms.
        if abs(live_create_time - float(recorded_create_time)) > 1.0:
            return False
    return True


def _is_easyicu_webserver_cmdline(cmdline: str) -> bool:
    normalized = (cmdline or "").replace("\\", "/").lower()
    return "uvicorn" in normalized and "easyicu.webserver.app:app" in normalized


def _find_easyicu_webserver_processes_on_port(port: int) -> list[int]:
    try:
        import psutil
    except ImportError:
        return []

    pids: list[int] = []
    try:
        inet_connections = list(psutil.net_connections(kind="inet"))
    except (psutil.AccessDenied, PermissionError):
        inet_connections = []
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                proc_connections = (
                    proc.net_connections(kind="inet")
                    if hasattr(proc, "net_connections")
                    else proc.connections(kind="inet")
                )
            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                PermissionError,
                NotImplementedError,
            ):
                continue

            cmdline = " ".join(proc.info.get("cmdline") or [])
            if not _is_easyicu_webserver_cmdline(cmdline):
                continue

            for conn in proc_connections:
                try:
                    if (
                        conn.laddr
                        and conn.laddr.port == port
                        and conn.status == psutil.CONN_LISTEN
                    ):
                        pids.append(int(proc.info["pid"]))
                except Exception:
                    continue
    else:
        for conn in inet_connections:
            try:
                if (
                    not conn.laddr
                    or conn.laddr.port != port
                    or conn.status != psutil.CONN_LISTEN
                ):
                    continue
            except Exception:
                continue

            pid = conn.pid
            if not pid:
                continue
            try:
                proc = psutil.Process(pid)
                cmdline = " ".join(proc.cmdline() or [])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if _is_easyicu_webserver_cmdline(cmdline):
                pids.append(int(pid))

    return list(dict.fromkeys(pids))


def run_app(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, *, background: bool = False
) -> int:
    if not _is_loopback_bind(host):
        print(
            "EasyICU WebApp is local-only because its filesystem APIs do not have "
            "remote authentication. Bind to 127.0.0.1, localhost, or ::1.",
            file=sys.stderr,
        )
        return 2
    cmd = _uvicorn_cmd(host, port)
    if background:
        log_path = _log_file()
        pid_path = _pid_file()
        with log_path.open("a", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                env=_runtime_env(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        _write_pid_record(pid_path, process.pid, port)
        print(f"Started EasyICU native WebApp in the background (PID: {process.pid})")
        print(f"Log file: {log_path}")
        return 0

    try:
        return subprocess.run(cmd, env=_runtime_env()).returncode
    except KeyboardInterrupt:
        return 130


def stop_app(port: int = DEFAULT_PORT) -> int:
    stopped = False
    pid_path = _pid_file()
    if pid_path.exists():
        record = _read_pid_record(pid_path)
        pid = int(record.get("pid") or 0)
        if pid:
            # A bare PID is not proof of identity: PIDs are recycled, and a
            # stale file plus a recycled PID means SIGTERM lands on whatever
            # unrelated process now holds that number. Verify the process is
            # actually our uvicorn before signalling it.
            if _pid_matches_record(pid, record):
                try:
                    os.kill(pid, signal.SIGTERM)
                    print(f"Stopped EasyICU native WebApp (PID: {pid})")
                    stopped = True
                except ProcessLookupError:
                    print(f"PID file points to a missing process: {pid}.")
            else:
                print(
                    f"Ignoring stale PID file: process {pid} is not this "
                    "EasyICU WebApp. Not signalling it."
                )
        pid_path.unlink(missing_ok=True)

    pids = _find_easyicu_webserver_processes_on_port(port)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            stopped = True
        except ProcessLookupError:
            continue

    if pids:
        print(
            f"Stopped {len(pids)} EasyICU native WebApp process{'es' if len(pids) != 1 else ''}."
        )
    elif not stopped:
        print("No EasyICU native WebApp process was found.")
    return 0


def status_app(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> int:
    if _health_check(host, port):
        print(f"EasyICU native WebApp is running: http://{_probe_host(host)}:{port}/")
        print(f"Runtime dir: {_runtime_dir()}")
        return 0
    print(f"EasyICU native WebApp is not healthy on port {port}.")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the native EasyICU FastAPI WebApp."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "stop", "status"],
        help="Action to perform.",
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help="Host interface to bind or probe."
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to bind or probe."
    )
    parser.add_argument(
        "--background", action="store_true", help="Run in the background."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run_app(args.host, args.port, background=args.background)
    if args.command == "stop":
        return stop_app(args.port)
    if args.command == "status":
        return status_app(args.host, args.port)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
