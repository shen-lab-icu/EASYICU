#!/usr/bin/env python3
"""Install the macOS DMG into a temporary folder and prove the frozen backend starts.

The Windows shell has a headless ``--smoke-test`` mode; the macOS shell does
not, so this script launches the bundled backend with the arguments and token
header that ``lib.rs::start_backend`` uses. It checks readiness, rejection of
a wrong token and frozen durable startup. Every path lives under
``--profile-root``; no user data or clinical data is touched.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time

APP_IDENTIFIER = "org.easyicu.desktop"
READY_TIMEOUT_SECONDS = 300
TYPED_FAILURE = re.compile(r"(?m)(?<=DurableWorkflowError: )durable_[a-z0-9_]+\b")
RAISED = re.compile(r"(?m)^[A-Za-z_][\w.]*(?:Error|Exception)\b[^\r\n]*")


def startup_failure(log: str, launch_failure: str | None = None) -> str | None:
    """Name the most specific startup failure, as the Windows smoke does.

    The backend's typed failure comes first, then the exception it logged,
    then the launcher's own failure. ``None`` means the backend started.
    """
    typed = TYPED_FAILURE.search(log)
    if typed:
        return typed.group()
    if "Application startup complete." not in log:
        raised = RAISED.findall(log)
        if raised:
            return f"desktop_backend_startup_incomplete: {raised[-1]}"
        if launch_failure:
            return f"desktop_backend_startup_incomplete: {launch_failure}"
        return "desktop_backend_startup_incomplete"
    if "Durable dispatcher started (frozen=True)" not in log:
        return "durable_executor_unavailable"
    return launch_failure


def backend_command(app: Path, *, port: int, state: Path, runtime: Path, parent_pid: int) -> list[str]:
    resources = app / "Contents" / "Resources" / "resources"
    return [
        str(resources / "backend" / "easyicu-backend"),
        "--port", str(port),
        "--state-dir", str(state),
        "--runtime-dir", str(runtime),
        "--parent-pid", str(parent_pid),
        "--node-bin", str(resources / "node"),
    ]


def _status(port: int, token: str) -> int | None:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("GET", "/api/catalog", headers={"X-EasyICU-Desktop-Token": token})
        return connection.getresponse().status
    except OSError:
        return None
    finally:
        connection.close()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _install(dmg: Path, applications: Path) -> Path:
    applications.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="easyicu-dmg-mount-") as mount:
        subprocess.run(["hdiutil", "attach", "-nobrowse", "-readonly", "-noautoopen",
                        "-mountpoint", mount, str(dmg)], check=True)
        try:
            app = applications / "EasyICU.app"
            shutil.rmtree(app, ignore_errors=True)
            # ditto keeps the bundle's symlinks, metadata and signature intact.
            subprocess.run(["ditto", str(Path(mount) / "EasyICU.app"), str(app)], check=True)
        finally:
            subprocess.run(["hdiutil", "detach", mount], check=True)
    subprocess.run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)], check=True)
    return app


def _ready(process: subprocess.Popen, port: int, token: str) -> str | None:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            return f"backend exited before readiness: {code}"
        if _status(port, token) == 200:
            if _status(port, "invalid-smoke-token") == 200:
                return "backend accepted an unauthenticated readiness request"
            return None
        time.sleep(0.25)
    return "backend readiness timed out"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dmg", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    args = parser.parse_args(argv)
    profile = args.profile_root.expanduser().resolve()
    # The same folders Tauri's app_data_dir and app_log_dir give this identifier.
    data_root = profile / "Library" / "Application Support" / APP_IDENTIFIER
    state, runtime = data_root / "state", data_root / "runtime"
    logs = profile / "Library" / "Logs" / APP_IDENTIFIER
    for path in (state, runtime, logs):
        path.mkdir(parents=True, exist_ok=True)
    result_path = runtime / "desktop-smoke-result.json"
    result_path.unlink(missing_ok=True)

    app = _install(args.dmg.resolve(), profile / "Applications")
    port = _free_port()
    token = secrets.token_hex(32)
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    # Prove launch uses the bundled Python and Node, not the build machine's.
    env.update(HOME=str(profile), PATH="/usr/bin:/bin:/usr/sbin:/sbin",
               EASYICU_DESKTOP_SESSION_TOKEN=token)
    log_path = logs / "desktop-smoke.log"
    command = backend_command(app, port=port, state=state, runtime=runtime, parent_pid=os.getpid())
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        try:
            launch_failure = _ready(process, port, token)
            if launch_failure is None:
                result_path.write_text(
                    json.dumps({"healthy": True, "unauthenticated_rejected": True}) + "\n",
                    encoding="utf-8",
                )
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    text = log_path.read_text(encoding="utf-8", errors="replace")
    failure = startup_failure(text, launch_failure)
    if failure:
        # The synthetic profile's log holds no clinical data; show its end here.
        print("--- end of desktop-smoke.log ---")
        print("\n".join(text.splitlines()[-40:]))
        print(failure, file=sys.stderr)
        return 1
    print(f"macOS DMG smoke test passed: {app}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
