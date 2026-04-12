#!/usr/bin/env python3
"""Cross-platform bootstrap launcher for EasyICU.

This script is intended for source checkouts or GitHub ZIP downloads. It
creates a local virtual environment inside the repository, installs the webapp
dependencies on first run, and starts the EasyICU Streamlit service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import venv
import webbrowser

PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = PROJECT_ROOT / ".easyicu-runtime"
VENV_DIR = RUNTIME_DIR / "venv"
STAMP_FILE = RUNTIME_DIR / "install-stamp.json"
PYPROJECT_FILE = PROJECT_ROOT / "pyproject.toml"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8501


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env["EASYICU_RUNTIME_DIR"] = str(RUNTIME_DIR)
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("EASYICU_VERBOSE", "0")
    return env


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=_runtime_env(), check=True)


def _health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/_stcore/health"


def _wait_for_health(port: int, timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    url = _health_url(port)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def _current_install_state() -> dict[str, object]:
    return {
        "schema_version": 2,
        "pyproject_sha256": _hash_file(PYPROJECT_FILE),
        "python_major": sys.version_info.major,
        "python_minor": sys.version_info.minor,
        "install_mode": "editable",
    }


def _load_install_state() -> dict[str, object] | None:
    if not STAMP_FILE.exists():
        return None
    try:
        return json.loads(STAMP_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def ensure_virtualenv() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if _venv_python().exists():
        return

    print("Creating EasyICU runtime environment...")
    builder = venv.EnvBuilder(with_pip=True, clear=False, symlinks=os.name != "nt")
    builder.create(VENV_DIR)


def install_easyicu(force: bool = False) -> None:
    ensure_virtualenv()
    desired_state = _current_install_state()
    installed_state = _load_install_state()

    if not force and installed_state == desired_state and _runtime_package_is_usable():
        return

    python_bin = str(_venv_python())
    print("Installing EasyICU webapp dependencies...")
    _run([python_bin, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    subprocess.run(
        [python_bin, "-m", "pip", "uninstall", "-y", "easyicu"],
        env=_runtime_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    _run([python_bin, "-m", "pip", "install", "--upgrade", "-e", ".[webapp]"], cwd=PROJECT_ROOT)

    STAMP_FILE.write_text(json.dumps(desired_state, indent=2), encoding="utf-8")


def _runtime_package_is_usable() -> bool:
    python_path = _venv_python()
    if not python_path.exists():
        return False

    result = subprocess.run(
        [
            str(python_path),
            "-c",
            "import easyicu; import easyicu.webapp; print('ok')",
        ],
        env=_runtime_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def start_easyicu(
    host: str,
    port: int,
    *,
    force_reinstall: bool = False,
    foreground: bool = False,
    open_browser: bool = True,
) -> int:
    is_running = _wait_for_health(port, timeout=1)

    if force_reinstall and is_running:
        print("EasyICU is already running. Stopping it before reinstall...")
        stop_easyicu()
        is_running = False

    if is_running:
        url = f"http://{host}:{port}"
        print(f"EasyICU is already running at {url}")
        if open_browser:
            webbrowser.open(url)
        return 0

    install_easyicu(force=force_reinstall)

    python_bin = str(_venv_python())
    cmd = [
        python_bin,
        "-m",
        "easyicu.webapp",
        "run",
        "--host",
        host,
        "--port",
        str(port),
    ]

    if foreground:
        print("Starting EasyICU in the foreground...")
        return subprocess.run(cmd, env=_runtime_env()).returncode

    cmd.append("--background")
    print("Starting EasyICU in the background...")
    _run(cmd)

    if not _wait_for_health(port, timeout=60):
        log_path = RUNTIME_DIR / "easyicu_webapp.log"
        print(f"EasyICU did not become ready in time. Check {log_path}.", file=sys.stderr)
        return 1

    url = f"http://{host}:{port}"
    print(f"EasyICU is ready: {url}")
    if open_browser:
        webbrowser.open(url)
    return 0


def stop_easyicu() -> int:
    if not _venv_python().exists():
        print("EasyICU runtime is not installed yet.")
        return 1
    return subprocess.run(
        [str(_venv_python()), "-m", "easyicu.webapp", "stop"],
        env=_runtime_env(),
    ).returncode


def status_easyicu(port: int) -> int:
    if not _venv_python().exists():
        print("EasyICU runtime is not installed yet.")
        return 1
    return subprocess.run(
        [str(_venv_python()), "-m", "easyicu.webapp", "status", "--port", str(port)],
        env=_runtime_env(),
    ).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap and run EasyICU from a source checkout."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="start",
        choices=["start", "stop", "status", "install"],
        help="Action to perform.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host for the local web server.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port for the local web server.")
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Keep the Streamlit process attached to the current terminal.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser automatically after startup.",
    )
    parser.add_argument(
        "--force-reinstall",
        action="store_true",
        help="Reinstall the EasyICU package into the local runtime environment.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if sys.version_info < (3, 9):
        print("EasyICU requires Python 3.9 or newer.", file=sys.stderr)
        return 1

    if args.command == "install":
        install_easyicu(force=args.force_reinstall)
        print(f"EasyICU runtime is ready in {RUNTIME_DIR}")
        return 0

    if args.command == "start":
        return start_easyicu(
            args.host,
            args.port,
            force_reinstall=args.force_reinstall,
            foreground=args.foreground,
            open_browser=not args.no_browser,
        )

    if args.command == "stop":
        return stop_easyicu()

    if args.command == "status":
        return status_easyicu(args.port)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
