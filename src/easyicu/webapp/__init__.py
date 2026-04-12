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

        print(f"✅ 后台启动成功 (PID: {process.pid})")
        print(f"   日志文件: {log_path}")
        return

    if daemon:
        max_retries = 10
        retry_count = 0
        restart_delay = 5

        while retry_count < max_retries:
            print(f"🚀 启动服务... (尝试 {retry_count + 1}/{max_retries})")

            process = subprocess.Popen(cmd, env=child_env)

            try:
                process.wait()

                if process.returncode == 0:
                    print("✅ 服务正常退出")
                    break

                print(f"⚠️ 服务异常退出 (code: {process.returncode})")
                retry_count += 1
                print(f"⏳ {restart_delay}秒后重试...")
                time.sleep(restart_delay)

            except KeyboardInterrupt:
                print("\n🛑 收到停止信号，正在关闭...")
                process.terminate()
                process.wait(timeout=5)
                break

        if retry_count >= max_retries:
            print(f"❌ 重试次数已达上限 ({max_retries})，退出")
            sys.exit(1)
    else:
        subprocess.run(cmd, env=child_env)


def stop_app():
    """停止 EasyICU Web 应用。"""
    pid_file = _pid_file()

    if pid_file.exists():
        with pid_file.open('r', encoding='utf-8') as handle:
            pid = int(handle.read().strip())

        try:
            os.kill(pid, signal.SIGTERM)
            print(f"✅ 已停止服务 (PID: {pid})")
        except ProcessLookupError:
            print("⚠️ 服务未运行")

        pid_file.unlink(missing_ok=True)
        return

    pids = _find_running_easyicu_processes()
    if pids:
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
        print(f"✅ 已停止服务 ({len(pids)} 个进程)")
    else:
        print("⚠️ 未找到运行中的服务")


def status_app(port: int = 8501):
    """查看 EasyICU Web 应用状态。"""
    if _is_port_in_use(port):
        healthy = _health_check(port)
        print(f"✅ 服务运行中 (端口: {port})")
        print(f"   健康状态: {'正常' if healthy else '检查失败'}")
        print(f"   访问地址: http://localhost:{port}")
        print(f"   运行目录: {_runtime_dir()}")
    else:
        print(f"❌ 服务未运行 (端口: {port})")


__all__ = ['run_app', 'stop_app', 'status_app', 'check_dependencies']
