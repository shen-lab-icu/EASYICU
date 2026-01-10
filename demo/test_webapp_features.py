#!/usr/bin/env python
"""PyRICU Webapp 功能测试脚本。

本脚本测试 webapp 的核心组件是否正常工作。
"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_imports():
    """测试所有必需的导入。"""
    print("=" * 60)
    print("🧪 测试导入...")
    print("=" * 60)
    
    errors = []
    
    # 测试 streamlit
    try:
        import streamlit as st
        print(f"✅ streamlit {st.__version__}")
    except ImportError as e:
        errors.append(f"streamlit: {e}")
        print(f"❌ streamlit: {e}")
    
    # 测试 plotly
    try:
        import plotly
        print(f"✅ plotly {plotly.__version__}")
    except ImportError as e:
        errors.append(f"plotly: {e}")
        print(f"❌ plotly: {e}")
    
    # 测试 pandas
    try:
        import pandas as pd
        print(f"✅ pandas {pd.__version__}")
    except ImportError as e:
        errors.append(f"pandas: {e}")
        print(f"❌ pandas: {e}")
    
    # 测试 numpy
    try:
        import numpy as np
        print(f"✅ numpy {np.__version__}")
    except ImportError as e:
        errors.append(f"numpy: {e}")
        print(f"❌ numpy: {e}")
    
    return len(errors) == 0


def test_mock_data_generation():
    """测试模拟数据生成。"""
    print()
    print("=" * 60)
    print("🧪 测试模拟数据生成...")
    print("=" * 60)
    
    try:
        from pyricu.webapp.app import generate_mock_data
        
        data, patient_ids = generate_mock_data(n_patients=5, hours=24)
        
        print(f"✅ 生成了 {len(data)} 个 Concepts")
        print(f"✅ 生成了 {len(patient_ids)} 个患者")
        
        for name, df in data.items():
            print(f"   - {name}: {len(df)} 条记录")
        
        return True
        
    except Exception as e:
        print(f"❌ 模拟数据生成失败: {e}")
        return False


def test_plotly_charts():
    """测试 Plotly 图表生成。"""
    print()
    print("=" * 60)
    print("🧪 测试 Plotly 图表...")
    print("=" * 60)
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
        import numpy as np
        
        # 测试折线图
        df = pd.DataFrame({
            'time': range(24),
            'value': np.random.randn(24).cumsum()
        })
        fig = px.line(df, x='time', y='value', title='Test Line Chart')
        print("✅ 折线图创建成功")
        
        # 测试直方图
        fig = px.histogram(df, x='value', title='Test Histogram')
        print("✅ 直方图创建成功")
        
        # 测试子图
        fig = make_subplots(rows=2, cols=1)
        fig.add_trace(go.Scatter(x=df['time'], y=df['value']), row=1, col=1)
        fig.add_trace(go.Bar(x=df['time'], y=df['value']), row=2, col=1)
        print("✅ 子图创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ Plotly 图表测试失败: {e}")
        return False


def test_webapp_components():
    """测试 webapp 组件。"""
    print()
    print("=" * 60)
    print("🧪 测试 Webapp 组件...")
    print("=" * 60)
    
    try:
        from pyricu.webapp.app import (
            init_session_state,
            generate_mock_data,
        )
        print("✅ 核心函数导入成功")
        
        # 测试模块结构
        from pyricu.webapp import run_app
        print("✅ run_app 函数导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ Webapp 组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试。"""
    print()
    print("🏥 PyRICU Webapp 功能测试")
    print("=" * 60)
    print()
    
    results = {
        '导入测试': test_imports(),
        '模拟数据生成': test_mock_data_generation(),
        'Plotly 图表': test_plotly_charts(),
        'Webapp 组件': test_webapp_components(),
    }
    
    print()
    print("=" * 60)
    print("📋 测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 所有测试通过！Webapp 可以正常运行。")
        print()
        print("启动 webapp:")
        print("  python demo_webapp.py")
        print()
        print("或直接运行:")
        print("  streamlit run ../src/pyricu/webapp/app.py")
    else:
        print("⚠️ 部分测试失败，请检查上述错误信息。")
    
    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
