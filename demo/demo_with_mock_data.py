#!/usr/bin/env python
"""PyRICU 可视化演示 - 使用模拟数据。

本脚本使用生成的模拟数据演示可视化功能，无需真实 ICU 数据。

Usage:
    python demo_with_mock_data.py
    
Requirements:
    pip install pyricu[viz]
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def generate_mock_data(n_patients=5, hours=72):
    """生成模拟 ICU 数据。"""
    print("📊 生成模拟 ICU 数据...")
    
    data = {}
    patient_ids = list(range(10001, 10001 + n_patients))
    
    np.random.seed(42)
    
    # 时间点（每小时）
    time_points = np.arange(0, hours, 1)
    
    # 生成心率数据
    hr_records = []
    for pid in patient_ids:
        base_hr = np.random.uniform(70, 90)
        for t in time_points:
            # 添加一些变化和噪声
            hr = base_hr + np.sin(t / 6) * 10 + np.random.normal(0, 5)
            hr_records.append({
                'stay_id': pid,
                'time': t,
                'hr': max(40, min(150, hr)),
            })
    data['hr'] = pd.DataFrame(hr_records)
    
    # 生成 MAP 数据
    map_records = []
    for pid in patient_ids:
        base_map = np.random.uniform(65, 85)
        for t in time_points:
            map_val = base_map + np.cos(t / 8) * 8 + np.random.normal(0, 4)
            map_records.append({
                'stay_id': pid,
                'time': t,
                'map': max(40, min(120, map_val)),
            })
    data['map'] = pd.DataFrame(map_records)
    
    # 生成 SBP 数据
    sbp_records = []
    for pid in patient_ids:
        base_sbp = np.random.uniform(110, 140)
        for t in time_points:
            sbp_val = base_sbp + np.sin(t / 5) * 15 + np.random.normal(0, 8)
            sbp_records.append({
                'stay_id': pid,
                'time': t,
                'sbp': max(70, min(200, sbp_val)),
            })
    data['sbp'] = pd.DataFrame(sbp_records)
    
    # 生成体温数据
    temp_records = []
    for pid in patient_ids:
        base_temp = np.random.uniform(36.5, 37.5)
        for t in time_points[::4]:  # 每4小时一次
            temp_val = base_temp + np.random.normal(0, 0.3)
            # 添加发热事件
            if np.random.random() < 0.1:
                temp_val += 1.5
            temp_records.append({
                'stay_id': pid,
                'time': t,
                'temp': max(35, min(41, temp_val)),
            })
    data['temp'] = pd.DataFrame(temp_records)
    
    # 生成呼吸频率
    resp_records = []
    for pid in patient_ids:
        base_resp = np.random.uniform(14, 18)
        for t in time_points:
            resp_val = base_resp + np.random.normal(0, 2)
            resp_records.append({
                'stay_id': pid,
                'time': t,
                'resp': max(8, min(40, resp_val)),
            })
    data['resp'] = pd.DataFrame(resp_records)
    
    # 生成 SpO2 数据
    spo2_records = []
    for pid in patient_ids:
        for t in time_points:
            spo2_val = 97 + np.random.normal(0, 2)
            # 偶尔低氧
            if np.random.random() < 0.05:
                spo2_val -= 10
            spo2_records.append({
                'stay_id': pid,
                'time': t,
                'spo2': max(80, min(100, spo2_val)),
            })
    data['spo2'] = pd.DataFrame(spo2_records)
    
    # 生成 SOFA 数据（包含组件）
    sofa_records = []
    for pid in patient_ids:
        for t in time_points[::6]:  # 每6小时
            sofa_resp = np.random.choice([0, 1, 2, 3, 4], p=[0.4, 0.3, 0.15, 0.1, 0.05])
            sofa_coag = np.random.choice([0, 1, 2, 3, 4], p=[0.5, 0.25, 0.15, 0.07, 0.03])
            sofa_liver = np.random.choice([0, 1, 2, 3, 4], p=[0.6, 0.2, 0.12, 0.05, 0.03])
            sofa_cardio = np.random.choice([0, 1, 2, 3, 4], p=[0.4, 0.25, 0.2, 0.1, 0.05])
            sofa_cns = np.random.choice([0, 1, 2, 3, 4], p=[0.5, 0.25, 0.15, 0.07, 0.03])
            sofa_renal = np.random.choice([0, 1, 2, 3, 4], p=[0.5, 0.25, 0.15, 0.07, 0.03])
            
            sofa_total = sofa_resp + sofa_coag + sofa_liver + sofa_cardio + sofa_cns + sofa_renal
            
            sofa_records.append({
                'stay_id': pid,
                'time': t,
                'sofa': sofa_total,
                'sofa_resp': sofa_resp,
                'sofa_coag': sofa_coag,
                'sofa_liver': sofa_liver,
                'sofa_cardio': sofa_cardio,
                'sofa_cns': sofa_cns,
                'sofa_renal': sofa_renal,
            })
    data['sofa'] = pd.DataFrame(sofa_records)
    
    # 生成 norepi_rate 数据
    norepi_records = []
    for pid in patient_ids:
        for t in time_points:
            # 只有部分时间有用药
            if 12 <= t <= 48 and np.random.random() < 0.7:
                rate = np.random.uniform(0.05, 0.3)
                norepi_records.append({
                    'stay_id': pid,
                    'time': t,
                    'norepi_rate': rate,
                })
    data['norepi_rate'] = pd.DataFrame(norepi_records) if norepi_records else pd.DataFrame(
        columns=['stay_id', 'time', 'norepi_rate']
    )
    
    # 生成肌酐数据
    crea_records = []
    for pid in patient_ids:
        base_crea = np.random.uniform(0.8, 1.2)
        for t in time_points[::8]:  # 每8小时
            crea_val = base_crea + np.random.normal(0, 0.2)
            crea_records.append({
                'stay_id': pid,
                'time': t,
                'crea': max(0.3, crea_val),
            })
    data['crea'] = pd.DataFrame(crea_records)
    
    # 生成胆红素数据
    bili_records = []
    for pid in patient_ids:
        base_bili = np.random.uniform(0.5, 1.5)
        for t in time_points[::12]:  # 每12小时
            bili_val = base_bili + np.random.normal(0, 0.3)
            bili_records.append({
                'stay_id': pid,
                'time': t,
                'bili': max(0.1, bili_val),
            })
    data['bili'] = pd.DataFrame(bili_records)
    
    print(f"✅ 生成了 {len(data)} 个 concepts 的模拟数据")
    for name, df in data.items():
        print(f"   - {name}: {len(df)} 条记录")
    
    return data, patient_ids[0]


def demo_timeline(data, patient_id, output_dir):
    """演示时序图。"""
    from pyricu.visualization import plot_timeline
    
    print("\n📈 绘制心率时序图...")
    fig = plot_timeline(
        data['hr'],
        patient_id=patient_id,
        title=f"Heart Rate - Patient {patient_id}",
    )
    output_path = output_dir / "demo_hr_timeline.html"
    fig.write_html(str(output_path))
    print(f"   保存到: {output_path}")
    
    return fig


def demo_vitals_panel(data, patient_id, output_dir):
    """演示生命体征面板。"""
    from pyricu.visualization import plot_vitals_panel
    
    print("\n📊 绘制生命体征面板...")
    vitals = {k: data[k] for k in ['hr', 'map', 'sbp', 'resp', 'temp', 'spo2'] if k in data}
    
    fig = plot_vitals_panel(
        vitals,
        patient_id=patient_id,
        title=f"Vital Signs Panel - Patient {patient_id}",
    )
    output_path = output_dir / "demo_vitals_panel.html"
    fig.write_html(str(output_path))
    print(f"   保存到: {output_path}")
    
    return fig


def demo_sofa_breakdown(data, patient_id, output_dir):
    """演示 SOFA 分解图。"""
    from pyricu.visualization import plot_sofa_breakdown
    
    print("\n📊 绘制 SOFA 分解堆叠图...")
    fig = plot_sofa_breakdown(
        data['sofa'],
        patient_id=patient_id,
        title=f"SOFA Score Breakdown - Patient {patient_id}",
        stacked=True,
    )
    output_path = output_dir / "demo_sofa_breakdown.html"
    fig.write_html(str(output_path))
    print(f"   保存到: {output_path}")
    
    return fig


def demo_sofa_trajectory(data, output_dir):
    """演示 SOFA 轨迹图。"""
    from pyricu.visualization import plot_sofa_trajectory
    
    print("\n📈 绘制 SOFA 轨迹图...")
    fig = plot_sofa_trajectory(
        data['sofa'],
        title="SOFA Score Trajectory (All Patients)",
        show_mean=True,
        show_ci=True,
    )
    output_path = output_dir / "demo_sofa_trajectory.html"
    fig.write_html(str(output_path))
    print(f"   保存到: {output_path}")
    
    return fig


def demo_missing_heatmap(data, output_dir):
    """演示缺失值热力图。"""
    from pyricu.visualization import plot_missing_heatmap
    
    print("\n📋 绘制缺失值热力图...")
    fig = plot_missing_heatmap(data, title="Missing Rate by Concept")
    output_path = output_dir / "demo_missing_heatmap.html"
    fig.write_html(str(output_path))
    print(f"   保存到: {output_path}")
    
    return fig


def demo_distribution(data, output_dir):
    """演示数值分布图。"""
    from pyricu.visualization import plot_concept_distribution
    
    print("\n📊 绘制心率分布图...")
    fig = plot_concept_distribution(data, 'hr')
    output_path = output_dir / "demo_hr_distribution.html"
    fig.write_html(str(output_path))
    print(f"   保存到: {output_path}")
    
    return fig


def demo_patient_dashboard(data, patient_id, output_dir):
    """演示患者仪表盘。"""
    from pyricu.visualization import PatientDashboard
    
    print("\n🏥 生成患者仪表盘...")
    dashboard = PatientDashboard(patient_id=patient_id, database='mock')
    dashboard.load_data(data)
    
    fig = dashboard.render_full_dashboard()
    output_path = output_dir / f"demo_patient_{patient_id}_dashboard.html"
    fig.write_html(str(output_path))
    print(f"   保存到: {output_path}")
    
    return fig


def main():
    print("=" * 60)
    print("🏥 PyRICU 可视化演示 (模拟数据)")
    print("=" * 60)
    
    # 检查依赖
    try:
        import plotly
        print(f"\n✅ plotly {plotly.__version__} 已安装")
    except ImportError:
        print("\n❌ plotly 未安装")
        print("   请运行: pip install pyricu[viz]")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(__file__).parent / "demo_output"
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 输出目录: {output_dir.absolute()}")
    
    # 生成模拟数据
    data, patient_id = generate_mock_data(n_patients=5, hours=72)
    
    print("\n" + "=" * 60)
    print("开始可视化演示")
    print("=" * 60)
    
    # 运行各项演示
    try:
        demo_timeline(data, patient_id, output_dir)
        demo_vitals_panel(data, patient_id, output_dir)
        demo_sofa_breakdown(data, patient_id, output_dir)
        demo_sofa_trajectory(data, output_dir)
        demo_missing_heatmap(data, output_dir)
        demo_distribution(data, output_dir)
        demo_patient_dashboard(data, patient_id, output_dir)
        
    except Exception as e:
        print(f"\n⚠️ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 总结
    print("\n" + "=" * 60)
    print("✅ 演示完成！")
    print("=" * 60)
    
    html_files = list(output_dir.glob("demo_*.html"))
    print(f"\n生成了 {len(html_files)} 个 HTML 文件:")
    for f in sorted(html_files):
        print(f"  📊 {f.name}")
    
    print(f"\n在浏览器中打开查看:")
    print(f"  file://{output_dir.absolute()}/")
    
    # 尝试自动打开第一个文件
    try:
        import webbrowser
        first_file = sorted(html_files)[0] if html_files else None
        if first_file:
            print(f"\n正在打开 {first_file.name}...")
            webbrowser.open(f"file://{first_file.absolute()}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
