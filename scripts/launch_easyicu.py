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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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


def _run_capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        env=_runtime_env(),
        text=True,
        capture_output=True,
        check=False,
    )


def _is_easyicu_cmdline(cmdline: str) -> bool:
    normalized = (cmdline or "").replace("\\", "/").lower()
    return "streamlit" in normalized or "easyicu" in normalized


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


def _configured_pip_index(python_bin: str) -> tuple[str | None, str | None]:
    env = _runtime_env()
    for key in ("PIP_INDEX_URL", "UV_INDEX_URL"):
        value = env.get(key, "").strip()
        if value:
            return value, f"env:{key}"

    for scope_key in ("global.index-url", "user.index-url", "site.index-url"):
        result = _run_capture([python_bin, "-m", "pip", "config", "get", scope_key])
        value = (result.stdout or "").strip()
        if result.returncode == 0 and value:
            return value, f"pip config:{scope_key}"

    return None, None


def _print_pip_source_hint(python_bin: str) -> None:
    index_url, source = _configured_pip_index(python_bin)
    if index_url:
        print(f"Using pip index from {source}: {index_url}")
        return

    print("Using default pip index: https://pypi.org/simple")
    locale_hint = " ".join(
        filter(None, [
            os.environ.get("LANG", ""),
            os.environ.get("LC_ALL", ""),
            os.environ.get("LC_MESSAGES", ""),
        ])
    ).lower()
    if "zh" in locale_hint or "cn" in locale_hint:
        print(
            "No custom pip mirror is configured. If installation is slow, set PIP_INDEX_URL, "
            "for example: https://pypi.tuna.tsinghua.edu.cn/simple"
        )


def _stop_with_source_tree(port: int) -> int:
    stopper = (
        "import sys; "
        f"sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r}); "
        "from easyicu.webapp import stop_app; "
        f"stop_app(port={port})"
    )
    return subprocess.run([sys.executable, "-c", stopper], env=_runtime_env(), check=False).returncode


def _find_port_owners(port: int) -> list[dict[str, object]]:
    try:
        import psutil
    except ImportError:
        return []

    owners: list[dict[str, object]] = []
    for conn in psutil.net_connections(kind="inet"):
        try:
            if not conn.laddr or conn.laddr.port != port or conn.status != psutil.CONN_LISTEN:
                continue
        except Exception:
            continue

        pid = conn.pid
        if not pid:
            continue
        try:
            proc = psutil.Process(pid)
            cmdline = " ".join(proc.cmdline() or [])
            name = proc.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

        owners.append({
            "pid": pid,
            "name": name,
            "cmdline": cmdline,
            "is_easyicu": _is_easyicu_cmdline(cmdline),
        })

    unique: list[dict[str, object]] = []
    seen: set[int] = set()
    for owner in owners:
        pid = int(owner["pid"])
        if pid in seen:
            continue
        seen.add(pid)
        unique.append(owner)
    return unique


def _wait_until_port_state(port: int, *, should_be_free: bool, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        owners = _find_port_owners(port)
        is_free = len(owners) == 0
        if is_free == should_be_free:
            return True
        time.sleep(0.25)
    return len(_find_port_owners(port)) == 0 if should_be_free else len(_find_port_owners(port)) > 0


def _print_port_conflict_warning(port: int, owners: list[dict[str, object]]) -> None:
    print(f"Port {port} is still occupied.", file=sys.stderr)
    foreign = [owner for owner in owners if not bool(owner.get("is_easyicu"))]
    if foreign:
        print("EasyICU will not kill non-EasyICU processes automatically.", file=sys.stderr)
        print("Please stop the process below or choose a different --port.", file=sys.stderr)
        for owner in foreign[:5]:
            cmdline = str(owner.get("cmdline") or "").strip()
            if len(cmdline) > 180:
                cmdline = cmdline[:177] + "..."
            print(
                f"  - PID {owner['pid']} ({owner['name']}): {cmdline or '[command unavailable]'}",
                file=sys.stderr,
            )
        return

    print("An EasyICU-related process is still holding the port after shutdown.", file=sys.stderr)
    for owner in owners[:5]:
        cmdline = str(owner.get("cmdline") or "").strip()
        if len(cmdline) > 180:
            cmdline = cmdline[:177] + "..."
        print(
            f"  - PID {owner['pid']} ({owner['name']}): {cmdline or '[command unavailable]'}",
            file=sys.stderr,
        )


def _release_port_before_start(port: int) -> bool:
    runtime_exists = _venv_python().exists()
    port_busy = _wait_for_health(port, timeout=1)
    port_in_use = False

    if not port_busy:
        try:
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                port_in_use = sock.connect_ex(("127.0.0.1", port)) == 0
        except Exception:
            port_in_use = False

    if not runtime_exists and not port_busy and not port_in_use:
        return True

    print(f"Releasing EasyICU processes on port {port} before startup...")
    if runtime_exists:
        stop_easyicu(port)
    else:
        _stop_with_source_tree(port)
    _wait_until_port_state(port, should_be_free=True, timeout=5.0)

    owners = _find_port_owners(port)
    if owners:
        _print_port_conflict_warning(port, owners)
        return False
    return True


def install_easyicu(force: bool = False) -> None:
    ensure_virtualenv()
    desired_state = _current_install_state()
    installed_state = _load_install_state()

    if not force and installed_state == desired_state and _runtime_package_is_usable():
        return

    python_bin = str(_venv_python())
    print("Installing EasyICU webapp dependencies...")
    _print_pip_source_hint(python_bin)
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
    if not _release_port_before_start(port):
        return 1

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


def stop_easyicu(port: int = DEFAULT_PORT) -> int:
    if not _venv_python().exists():
        return _stop_with_source_tree(port)
    return subprocess.run(
        [str(_venv_python()), "-m", "easyicu.webapp", "stop", "--port", str(port)],
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
    parser.add_argument(
        "--pip-index-url",
        default="",
        help="Override pip index URL used to install the runtime environment.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.pip_index_url:
        os.environ["PIP_INDEX_URL"] = args.pip_index_url.strip()

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
        return stop_easyicu(args.port)

    if args.command == "status":
        return status_easyicu(args.port)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
