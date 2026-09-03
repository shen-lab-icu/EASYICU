"""Frozen FastAPI sidecar entry point for EasyICU Desktop."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import threading
import time
from typing import Sequence


def _absolute_directory(raw: str, *, name: str) -> Path:
    path = Path(str(raw or "").strip()).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
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
    parser = argparse.ArgumentParser(description="EasyICU Desktop backend")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--session-token")
    parser.add_argument("--node-bin")
    return parser


def _watch_parent_process(parent_pid: int, *, interval: float = 1.0) -> None:
    if parent_pid <= 1 or parent_pid == os.getpid():
        raise ValueError("parent-pid must identify the desktop shell")

    def monitor() -> None:
        import psutil

        while psutil.pid_exists(parent_pid):
            time.sleep(interval)
        os._exit(0)

    threading.Thread(
        target=monitor,
        name="easyicu-desktop-parent-watch",
        daemon=True,
    ).start()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1024 <= args.port <= 65535:
        raise ValueError("port must be between 1024 and 65535")
    _configure_environment(
        state_dir=args.state_dir,
        runtime_dir=args.runtime_dir,
        session_token=args.session_token
        or os.environ.get("EASYICU_DESKTOP_SESSION_TOKEN", ""),
        node_bin=args.node_bin,
    )
    _watch_parent_process(args.parent_pid)

    import uvicorn

    from easyicu.webserver.app import app

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        access_log=False,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
