#!/usr/bin/env python
"""PyRICU Web 应用演示脚本。

本脚本启动 PyRICU 的 Streamlit Web 应用，提供交互式 ICU 数据分析界面。

功能特性:
    - 🏠 首页: 数据概览和快速开始指南
    - 📈 时序分析: 交互式时间序列可视化
    - 🏥 患者视图: 单患者多维度仪表盘
    - 📊 数据质量: 缺失率分析和数值分布检查
    - 💾 数据导出: CSV/Excel/Parquet 格式导出
    - 🎭 模拟数据: 无需真实数据即可体验所有功能

Usage:
    # 方式1: 使用此脚本
    python demo_webapp.py
    
    # 方式2: 使用此脚本 + 指定端口
    python demo_webapp.py --port 8502
    
    # 方式3: 直接使用命令行
    pyricu-webapp
    
    # 方式4: 使用 streamlit
    streamlit run ../src/pyricu/webapp/app.py

Requirements:
    pip install pyricu[webapp]
"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def check_dependencies():
    """检查 webapp 依赖是否安装。"""
    missing = []
    
    try:
        import streamlit
        print(f"✅ streamlit {streamlit.__version__} 已安装")
    except ImportError:
        missing.append('streamlit')
        print("❌ streamlit 未安装")
    
    try:
        import plotly
        print(f"✅ plotly {plotly.__version__} 已安装")
    except ImportError:
        missing.append('plotly')
        print("❌ plotly 未安装")
    
    if missing:
        print(f"\n请安装缺失的依赖:")
        print(f"  pip install {' '.join(missing)}")
        print(f"\n或安装完整的 webapp 依赖:")
        print(f"  pip install pyricu[webapp]")
        return False
    
    return True


def print_features():
    """打印功能说明。"""
    print("""
📋 功能说明:
─────────────────────────────────────────────────────────────
🏠 首页        数据概览、快速开始指南
📈 时序分析    交互式时间序列可视化，支持多患者对比
🏥 患者视图    单患者多维度仪表盘，3种展示模式
📊 数据质量    缺失率热力图、数值分布、时间覆盖分析
💾 数据导出    CSV/Excel/Parquet 格式，支持批量导出
─────────────────────────────────────────────────────────────

💡 快速开始:
   1. 在左侧栏勾选「使用模拟数据」
   2. 点击「生成模拟数据」按钮
   3. 探索各个功能标签页
""")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PyRICU Web 应用演示")
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Web 应用端口 (默认: 8501)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Web 应用主机地址 (默认: localhost)'
    )
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='不自动打开浏览器'
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='守护模式运行（自动重启）'
    )
    parser.add_argument(
        '--background',
        action='store_true',
        help='后台运行'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🏥 PyRICU Web 应用")
    print("=" * 60)
    print()
    print("本地 ICU 数据分析与可视化平台")
    print("所有数据处理在本地完成，不会上传到任何服务器")
    print()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 打印功能说明
    print_features()
    
    print(f"正在启动 Web 应用...")
    print(f"  地址: http://{args.host}:{args.port}")
    if args.daemon:
        print(f"  模式: 守护模式（自动重启）")
    if args.background:
        print(f"  模式: 后台运行")
    print()
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    print()
    
    # 启动应用
    from pyricu.webapp import run_app
    run_app(
        host=args.host, 
        port=args.port,
        daemon=args.daemon,
        background=args.background
    )


if __name__ == "__main__":
    main()
