"""PyRICU Web 应用模块。

基于 Streamlit 的交互式 ICU 数据分析界面。

使用方法:
    # 安装依赖
    pip install pyricu[webapp]
    
    # 启动应用
    pyricu webapp
    
    # 或直接运行
    python -m pyricu.webapp
    
    # 守护模式（自动重启）
    python demo_webapp.py --daemon
"""

from typing import Optional
import time
import subprocess
import sys
import signal
import os
from pathlib import Path

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


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
            f"Install with: pip install pyricu[webapp]"
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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def run_app(
    host: str = 'localhost',
    port: int = 8501,
    debug: bool = False,
    daemon: bool = False,
    background: bool = False,
    low_memory: bool = False,
    workers: int = None,
):
    """启动 PyRICU Web 应用。
    
    Args:
        host: 主机地址
        port: 端口号
        debug: 是否启用调试模式
        daemon: 守护模式，自动重启崩溃的服务
        background: 后台运行
        low_memory: 低内存模式，减少内存占用（适用于 8GB 内存以下电脑）
        workers: 并行工作线程数，默认自动检测，设为 1 可减少内存占用
    """
    check_dependencies()
    
    app_path = Path(__file__).parent / 'app.py'
    config_dir = Path(__file__).parent / '.streamlit'
    
    # 设置环境变量传递配置给 app.py
    env = os.environ.copy()
    if low_memory:
        env['PYRICU_LOW_MEMORY'] = '1'
        print("💾 低内存模式已启用")
    if workers is not None:
        env['PYRICU_WORKERS'] = str(workers)
        print(f"🔧 并行工作线程数: {workers}")
    
    # 构建命令
    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        str(app_path),
        '--server.address', host,
        '--server.port', str(port),
        '--server.headless', 'true',
        '--server.runOnSave', 'false',
        '--server.fileWatcherType', 'none',  # 禁用文件监视，减少资源占用
        '--browser.gatherUsageStats', 'false',
    ]
    
    if not debug:
        cmd.extend(['--logger.level', 'warning'])
    
    # 后台运行
    if background:
        log_file = open('/tmp/pyricu_webapp.log', 'a')
        pid_file = '/tmp/pyricu_webapp.pid'
        
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))
        
        print(f"✅ 后台启动成功 (PID: {process.pid})")
        print(f"   日志文件: /tmp/pyricu_webapp.log")
        return
    
    # 守护模式
    if daemon:
        max_retries = 10
        retry_count = 0
        restart_delay = 5
        
        while retry_count < max_retries:
            print(f"🚀 启动服务... (尝试 {retry_count + 1}/{max_retries})")
            
            process = subprocess.Popen(cmd, env=env)
            
            try:
                # 等待进程退出
                process.wait()
                
                # 检查是否是正常退出
                if process.returncode == 0:
                    print("✅ 服务正常退出")
                    break
                else:
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
        # 普通模式
        subprocess.run(cmd, env=env)


def stop_app():
    """停止 PyRICU Web 应用。"""
    pid_file = '/tmp/pyricu_webapp.pid'
    
    if os.path.exists(pid_file):
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"✅ 已停止服务 (PID: {pid})")
        except ProcessLookupError:
            print("⚠️ 服务未运行")
        
        os.remove(pid_file)
    else:
        # 尝试通过进程名查找
        import subprocess
        result = subprocess.run(
            ['pkill', '-f', 'streamlit run.*app.py'],
            capture_output=True
        )
        if result.returncode == 0:
            print("✅ 已停止服务")
        else:
            print("⚠️ 未找到运行中的服务")


def status_app(port: int = 8501):
    """查看 PyRICU Web 应用状态。"""
    if _is_port_in_use(port):
        healthy = _health_check(port)
        print(f"✅ 服务运行中 (端口: {port})")
        print(f"   健康状态: {'正常' if healthy else '检查失败'}")
        print(f"   访问地址: http://localhost:{port}")
    else:
        print(f"❌ 服务未运行 (端口: {port})")


def main():
    """命令行入口点，支持参数解析。"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='PyRICU Web 应用 - ICU 数据分析界面',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  pyricu-webapp                      # 默认启动
  pyricu-webapp --low-memory         # 低内存模式（8GB 以下电脑）
  pyricu-webapp --workers 1          # 单线程模式（防止卡死）
  pyricu-webapp --port 8502          # 指定端口
  pyricu-webapp --low-memory --workers 1  # 最低资源模式
'''
    )
    
    parser.add_argument(
        '--host', type=str, default='localhost',
        help='服务器地址 (默认: localhost)'
    )
    parser.add_argument(
        '--port', type=int, default=8501,
        help='端口号 (默认: 8501)'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='启用调试模式'
    )
    parser.add_argument(
        '--daemon', action='store_true',
        help='守护模式，服务崩溃后自动重启'
    )
    parser.add_argument(
        '--background', action='store_true',
        help='后台运行'
    )
    parser.add_argument(
        '--low-memory', action='store_true', dest='low_memory',
        help='低内存模式：减少缓存、使用更小的数据块 (适用于 8GB 内存以下电脑)'
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='并行工作线程数 (默认: 自动检测，设为 1 可减少内存占用)'
    )
    
    args = parser.parse_args()
    
    run_app(
        host=args.host,
        port=args.port,
        debug=args.debug,
        daemon=args.daemon,
        background=args.background,
        low_memory=args.low_memory,
        workers=args.workers,
    )


__all__ = ['run_app', 'stop_app', 'status_app', 'check_dependencies', 'main']
