"""Frozen FastAPI sidecar entry point for EasyICU Desktop."""

from __future__ import annotations

import argparse
import logging
import ntpath
import os
from pathlib import Path
import sys
import threading
from typing import Callable, Sequence


def _windows_extended_path(raw: str) -> str:
    """Keep absolute drive/UNC paths usable beyond Win32's legacy MAX_PATH."""
    path = ntpath.normpath(raw)
    if path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path[2:]
    drive, tail = ntpath.splitdrive(path)
    if not drive or not tail.startswith("\\"):
        raise ValueError("Windows directory must be an absolute drive or UNC path")
    return "\\\\?\\" + path


def _absolute_directory(raw: str, *, name: str) -> Path:
    path = Path(str(raw or "").strip()).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    if sys.platform == "win32":
        path = Path(_windows_extended_path(str(path)))
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path.resolve()


def _configure_environment(
    *, state_dir: str, runtime_dir: str, session_token: str, node_bin: str | None
) -> None:
    state_root = _absolute_directory(state_dir, name="state-dir")
    runtime_root = _absolute_directory(runtime_dir, name="runtime-dir")
    token = str(session_token or "").strip()
    if len(token) < 32:
        raise ValueError("session-token must contain at least 32 characters")

    os.environ["EASYICU_HOME"] = str(state_root)
    os.environ["EASYICU_RUNTIME_DIR"] = str(runtime_root)
    os.environ["EASYICU_DESKTOP_SESSION_TOKEN"] = token
    os.environ["EASYICU_DEPLOYMENT"] = "desktop"
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("EASYICU_VERBOSE", "0")

    if node_bin:
        selected_node = Path(node_bin).expanduser()
        if not selected_node.is_absolute() or not selected_node.is_file():
            raise ValueError("node-bin must be an absolute executable path")
        os.environ["PATH"] = os.pathsep.join(
            [str(selected_node.parent), os.environ.get("PATH", "")]
        ).rstrip(os.pathsep)


def _parser() -> argparse.ArgumentParser:
    # The session token travels via the environment only. There is deliberately
    # no --session-token flag: process arguments are visible to other local
    # users via the process table, while the environment is inherited privately
    # from the desktop shell that generated the token.
    parser = argparse.ArgumentParser(description="EasyICU Desktop backend")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--node-bin")
    return parser


def _watch_parent_process(parent_pid: int, *, interval: float = 1.0,
                          request_shutdown: Callable[[], None] | None = None) -> threading.Event:
    if parent_pid <= 1 or parent_pid == os.getpid():
        raise ValueError("parent-pid must identify the desktop shell")
    if request_shutdown is None:
        raise ValueError("a graceful shutdown callback is required")
    stop = threading.Event()

    def monitor() -> None:
        import psutil

        try:
            parent = psutil.Process(parent_pid)
            # Process.is_running checks creation time too, protecting PID reuse.
            while parent.is_running():
                if stop.wait(interval):
                    return
        except psutil.NoSuchProcess:
            pass
        if not stop.is_set():
            request_shutdown()

    threading.Thread(
        target=monitor,
        name="easyicu-desktop-parent-watch",
        daemon=True,
    ).start()
    return stop


def _verify_durable_startup() -> None:
    """Exercise the running dispatcher in the sidecar, without submitting work."""
    from easyicu.webserver.durable_research import current_service
    from easyicu.webserver.durable_workflows import DurableWorkflowError

    service = current_service()
    if service is None:
        raise DurableWorkflowError("durable_executor_unavailable")
    # status requires a launched dispatcher and queries its actual DBOS store.
    # A fresh profile has no such workflow; no callback or model runs here.
    service.dispatcher.status("desktop-startup-probe")
    logging.getLogger("uvicorn.error").info(
        "Durable dispatcher started (frozen=%s)", bool(getattr(sys, "frozen", False)),
    )


def _register_startup_probe(app) -> None:
    """Run the durable probe after the Web app's own startup handlers.

    Starlette 1.x removed ``app.add_event_handler``; FastAPI's router keeps the
    startup list that ``app.on_event`` fills, and runs it in order.
    """
    app.router.add_event_handler("startup", _verify_durable_startup)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--easyicu-host-worker"]:
        from easyicu.runtime.host_worker import main as worker_main

        return worker_main(arguments[1:])
    args = _parser().parse_args(arguments)
    if not 1024 <= args.port <= 65535:
        raise ValueError("port must be between 1024 and 65535")
    session_token = os.environ.get("EASYICU_DESKTOP_SESSION_TOKEN", "")
    if not str(session_token).strip():
        raise ValueError(
            "EASYICU_DESKTOP_SESSION_TOKEN is not set; "
            "the desktop shell must launch the backend with a token"
        )
    _configure_environment(
        state_dir=args.state_dir,
        runtime_dir=args.runtime_dir,
        session_token=session_token,
        node_bin=args.node_bin,
    )
    import uvicorn

    from easyicu.webserver.app import app

    _register_startup_probe(app)
    server = uvicorn.Server(uvicorn.Config(
        app,
        host="127.0.0.1",
        port=args.port,
        access_log=False,
        log_level="info",
        timeout_graceful_shutdown=10,
    ))

    def request_shutdown() -> None:
        server.should_exit = True

    stop = _watch_parent_process(args.parent_pid, request_shutdown=request_shutdown)
    try:
        server.run()
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    # PyInstaller's multiprocessing hook dispatches spawn/resource-tracker
    # children before our application or analysis-worker argument parser.
    import multiprocessing

    multiprocessing.freeze_support()
    sys.exit(main())
