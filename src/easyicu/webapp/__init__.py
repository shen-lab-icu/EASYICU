"""EasyICU Web 应用模块。

基于 Streamlit 的交互式 ICU 数据分析界面。

使用方法:
    # 安装依赖
    pip install easyicu[webapp]

    # 启动应用
    easyicu-webapp

    # 或直接运行
    python -m easyicu.webapp run
"""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

try:
    import streamlit as st

    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


def _runtime_dir() -> Path:
    """Return the directory used for PID and log files."""
    override = os.environ.get("EASYICU_RUNTIME_DIR")
    runtime_dir = Path(override).expanduser() if override else Path(tempfile.gettempdir()) / "easyicu"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def _pid_file() -> Path:
    return _runtime_dir() / "easyicu_webapp.pid"


def _log_file() -> Path:
    return _runtime_dir() / "easyicu_webapp.log"


def check_dependencies():
    """检查 webapp 依赖是否安装。"""
    missing = []

    if not HAS_STREAMLIT:
        missing.append('streamlit')

    try:
        import plotly
    except ImportError:
        missing.append('plotly')

    if missing:
        raise ImportError(
            f"Missing dependencies for webapp: {', '.join(missing)}. "
            f"Install with: pip install easyicu[webapp]"
        )


def _health_check(port: int) -> bool:
    """检查服务是否健康。"""
    try:
        import urllib.request

        url = f"http://localhost:{port}/_stcore/health"
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def _is_port_in_use(port: int) -> bool:
    """检查端口是否被占用。"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(('localhost', port)) == 0


def _find_running_easyicu_processes():
    """Find orphaned EasyICU Streamlit processes when the PID file is missing."""
    try:
        import psutil
    except ImportError:
        return []

    pids = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

        normalized = cmdline.replace("\\", "/").lower()
        if "streamlit" in normalized and "easyicu/webapp/app.py" in normalized:
            pids.append(proc.info["pid"])
    return pids


def _find_processes_on_port(port: int):
    """Find processes listening on the target port, preferring Streamlit/EasyICU."""
    try:
        import psutil
    except ImportError:
        return []

    pids = []
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
            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError, NotImplementedError):
                continue

            cmdline = " ".join(proc.info.get("cmdline") or [])
            normalized = cmdline.replace("\\", "/").lower()
            if "streamlit" not in normalized and "easyicu" not in normalized:
                continue

            for conn in proc_connections:
                try:
                    if not conn.laddr or conn.laddr.port != port or conn.status != psutil.CONN_LISTEN:
                        continue
                except Exception:
                    continue
                pids.append(proc.info["pid"])
    else:
        for conn in inet_connections:
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
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

            normalized = cmdline.replace("\\", "/").lower()
            if "streamlit" in normalized or "easyicu" in normalized:
                pids.append(pid)
    return list(dict.fromkeys(pids))


def run_app(
    host: str = '0.0.0.0',
    port: int = 8501,
    debug: bool = False,
    daemon: bool = False,
    background: bool = False,
):
    """启动 EasyICU Web 应用。

    Args:
        host: 主机地址
        port: 端口号
        debug: 是否启用调试模式
        daemon: 守护模式，自动重启崩溃的服务
        background: 后台运行
    """
    check_dependencies()

    app_path = Path(__file__).parent / 'app.py'
    Path(__file__).parent / '.streamlit'
    child_env = os.environ.copy()
    child_env.setdefault("PYTHONUTF8", "1")
    child_env.setdefault("PYTHONIOENCODING", "utf-8")
    child_env.setdefault("EASYICU_VERBOSE", "0")

    cmd = [
        sys.executable,
        '-m',
        'streamlit',
        'run',
        str(app_path),
        '--server.address',
        host,
        '--server.port',
        str(port),
        '--server.headless',
        'true',
        '--server.runOnSave',
        'false',
        '--server.fileWatcherType',
        'none',
        '--browser.gatherUsageStats',
        'false',
        '--server.enableCORS',
        'false',
        '--server.enableXsrfProtection',
        'false',
        '--server.websocketPingInterval',
        '60',
        '--server.disconnectedSessionTTL',
        '3600',
    ]

    if not debug:
        cmd.extend(['--logger.level', 'warning'])

    if background:
        log_path = _log_file()
        pid_path = _pid_file()

        with log_path.open('a', encoding='utf-8') as log_file:
            process = subprocess.Popen(
                cmd,
                env=child_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        with pid_path.open('w', encoding='utf-8') as handle:
            handle.write(str(process.pid))

        print(f"✅ Started EasyICU in the background (PID: {process.pid})")
        print(f"   Log file: {log_path}")
        return

    if daemon:
        max_retries = 10
        retry_count = 0
        restart_delay = 5

        while retry_count < max_retries:
            print(f"🚀 Starting service... (attempt {retry_count + 1}/{max_retries})")

            process = subprocess.Popen(cmd, env=child_env)

            try:
                process.wait()

                if process.returncode == 0:
                    print("✅ Service exited normally")
                    break

                print(f"⚠️ Service exited unexpectedly (code: {process.returncode})")
                retry_count += 1
                print(f"⏳ Retrying in {restart_delay} seconds...")
                time.sleep(restart_delay)

            except KeyboardInterrupt:
                print("\n🛑 Received stop signal. Shutting down...")
                process.terminate()
                process.wait(timeout=5)
                break

        if retry_count >= max_retries:
            print(f"❌ Reached retry limit ({max_retries}). Exiting.")
            sys.exit(1)
    else:
        subprocess.run(cmd, env=child_env)


def stop_app(port: int = 8501):
    """停止 EasyICU Web 应用。"""
    pid_file = _pid_file()
    stopped_any = False

    if pid_file.exists():
        with pid_file.open('r', encoding='utf-8') as handle:
            pid = int(handle.read().strip())

        try:
            os.kill(pid, signal.SIGTERM)
            print(f"✅ Stopped service (PID: {pid})")
            stopped_any = True
        except ProcessLookupError:
            print(f"⚠️ PID file points to a missing process: {pid}. Continuing to scan for stale processes.")

        pid_file.unlink(missing_ok=True)

    pids = list(dict.fromkeys([
        *_find_running_easyicu_processes(),
        *_find_processes_on_port(port),
    ]))
    if pids:
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                stopped_any = True
            except ProcessLookupError:
                continue
        print(f"✅ Stopped service ({len(pids)} process{'es' if len(pids) != 1 else ''})")
    elif not stopped_any:
        print("⚠️ No running EasyICU service was found")


def status_app(port: int = 8501):
    """查看 EasyICU Web 应用状态。"""
    if _is_port_in_use(port):
        healthy = _health_check(port)
        print(f"✅ Service is running (port: {port})")
        print(f"   Health: {'OK' if healthy else 'Check failed'}")
        print(f"   URL: http://localhost:{port}")
        print(f"   Runtime dir: {_runtime_dir()}")
    else:
        print(f"❌ Service is not running (port: {port})")


__all__ = ['run_app', 'stop_app', 'status_app', 'check_dependencies']
