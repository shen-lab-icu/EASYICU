#!/usr/bin/env python
"""EasyICU 可视化演示脚本。

本脚本演示如何使用 EasyICU 的可视化功能：
1. 加载 ICU 数据
2. 绘制时序图、SOFA 分解图等
3. 生成患者仪表盘
4. 保存图表到文件

Usage:
    python demo_visualization.py --data-path /path/to/ricu_data/miiv
    
Requirements:
    pip install easyicu[viz]
"""

import argparse
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def check_dependencies():
    """检查可视化依赖是否安装。"""
    try:
        import plotly
        print(f"✅ plotly {plotly.__version__} 已安装")
    except ImportError:
        print("❌ plotly 未安装，请运行: pip install plotly")
        return False
    
    try:
        import pandas
        print(f"✅ pandas {pandas.__version__} 已安装")
    except ImportError:
        print("❌ pandas 未安装")
        return False
    
    return True


def demo_timeseries(data, patient_id, output_dir):
    """演示时序数据可视化。"""
    from easyicu.visualization import plot_timeline, plot_vitals_panel
    
    print("\n" + "=" * 60)
    print("📈 时序数据可视化演示")
    print("=" * 60)
    
    # 1. 单个 concept 时序图
    if 'hr' in data:
        print("\n1. 绘制心率时序图...")
        fig = plot_timeline(
            data['hr'], 
            patient_id=patient_id,
            title=f"Heart Rate - Patient {patient_id}",
        )
        output_path = output_dir / "hr_timeline.html"
        fig.write_html(str(output_path))
        print(f"   保存到: {output_path}")
    
    # 2. 生命体征面板图
    vitals = {}
    for concept in ['hr', 'map', 'sbp', 'resp', 'temp', 'spo2']:
        if concept in data:
            vitals[concept] = data[concept]
    
    if vitals:
        print("\n2. 绘制生命体征面板图...")
        fig = plot_vitals_panel(
            vitals,
            patient_id=patient_id,
            title=f"Vital Signs Panel - Patient {patient_id}",
        )
        output_path = output_dir / "vitals_panel.html"
        fig.write_html(str(output_path))
        print(f"   保存到: {output_path}")


def demo_scores(data, patient_id, output_dir):
    """演示评分系统可视化。"""
    from easyicu.visualization import plot_sofa_breakdown, plot_sofa_trajectory
    
    print("\n" + "=" * 60)
    print("📊 评分系统可视化演示")
    print("=" * 60)
    
    if 'sofa' not in data:
        print("⚠️ SOFA 数据不可用，跳过评分可视化")
        return
    
    # 1. SOFA 分解图
    print("\n1. 绘制 SOFA 评分分解图...")
    fig = plot_sofa_breakdown(
        data['sofa'],
        patient_id=patient_id,
        title=f"SOFA Score Breakdown - Patient {patient_id}",
        stacked=True,
    )
    output_path = output_dir / "sofa_breakdown.html"
    fig.write_html(str(output_path))
    print(f"   保存到: {output_path}")
    
    # 2. SOFA 轨迹图（多患者）
    print("\n2. 绘制 SOFA 轨迹图...")
    fig = plot_sofa_trajectory(
        data['sofa'],
        title="SOFA Score Trajectory",
        show_mean=True,
        show_ci=True,
    )
    output_path = output_dir / "sofa_trajectory.html"
    fig.write_html(str(output_path))
    print(f"   保存到: {output_path}")


def demo_cohort(data, output_dir):
    """演示队列分析可视化。"""
    from easyicu.visualization import plot_missing_heatmap, plot_concept_distribution
    
    print("\n" + "=" * 60)
    print("📋 队列分析可视化演示")
    print("=" * 60)
    
    # 1. 缺失值热力图
    print("\n1. 绘制缺失值热力图...")
    fig = plot_missing_heatmap(data, title="Missing Rate by Concept")
    output_path = output_dir / "missing_heatmap.html"
    fig.write_html(str(output_path))
    print(f"   保存到: {output_path}")
    
    # 2. 数值分布图
    for concept in ['hr', 'map', 'bili', 'crea']:
        if concept in data:
            print(f"\n2. 绘制 {concept} 分布图...")
            fig = plot_concept_distribution(data, concept)
            output_path = output_dir / f"distribution_{concept}.html"
            fig.write_html(str(output_path))
            print(f"   保存到: {output_path}")
            break


def demo_patient_dashboard(data, patient_id, database, output_dir):
    """演示患者仪表盘。"""
    from easyicu.visualization import PatientDashboard, render_patient_report
    
    print("\n" + "=" * 60)
    print("🏥 患者仪表盘演示")
    print("=" * 60)
    
    print(f"\n为患者 {patient_id} 生成综合仪表盘...")
    
    try:
        dashboard = PatientDashboard(patient_id=patient_id, database=database)
        dashboard.load_data(data)
        
        fig = dashboard.render_full_dashboard()
        output_path = output_dir / f"patient_{patient_id}_dashboard.html"
        fig.write_html(str(output_path))
        print(f"   保存到: {output_path}")
        
    except Exception as e:
        print(f"⚠️ 仪表盘生成失败: {e}")
        
        # 使用简化版
        output_path = render_patient_report(
            patient_id=patient_id,
            data=data,
            database=database,
            output_format='html',
            output_path=str(output_dir / f"patient_{patient_id}_report.html"),
        )
        print(f"   保存到: {output_path}")


def load_sample_data(data_path, concepts, patient_ids=None, limit=100):
    """加载示例数据。"""
    from easyicu import load_concepts
    
    print(f"\n📂 从 {data_path} 加载数据...")
    print(f"   Concepts: {concepts}")
    
    data = load_concepts(
        data_path=data_path,
        concepts=concepts,
        patient_ids=patient_ids,
        verbose=True,
    )
    
    print(f"\n✅ 成功加载 {len(data)} 个 concepts")
    
    # 获取第一个患者 ID
    patient_id = None
    id_candidates = ['stay_id', 'hadm_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid']
    
    for concept_df in data.values():
        if hasattr(concept_df, 'columns'):
            for col in id_candidates:
                if col in concept_df.columns:
                    patient_ids_list = concept_df[col].unique()
                    if len(patient_ids_list) > 0:
                        patient_id = patient_ids_list[0]
                        print(f"   使用患者 ID: {patient_id}")
                        break
        if patient_id:
            break
    
    return data, patient_id


def main():
    parser = argparse.ArgumentParser(description="EasyICU 可视化演示")
    parser.add_argument(
        '--data-path', 
        type=str,
        default='/home/zhuhb/project/ricu_to_python/ricu_data/miiv',
        help='RICU 格式数据目录路径'
    )
    parser.add_argument(
        '--database',
        type=str,
        default='miiv',
        choices=['miiv', 'mimic', 'eicu', 'aumc', 'hirid'],
        help='数据库名称'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./demo_output',
        help='输出目录'
    )
    parser.add_argument(
        '--patient-id',
        type=int,
        default=None,
        help='指定患者 ID（可选）'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🏥 EasyICU 可视化演示")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 输出目录: {output_dir.absolute()}")
    
    # 检查数据路径
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"\n❌ 数据路径不存在: {data_path}")
        print("请指定正确的数据路径，例如:")
        print("  python demo_visualization.py --data-path /path/to/ricu_data/miiv")
        sys.exit(1)
    
    # 定义要加载的 concepts
    concepts = [
        # 生命体征
        'hr', 'map', 'sbp', 'dbp', 'resp', 'temp', 'spo2',
        # 实验室检查
        'bili', 'crea', 'lac', 'plt',
        # 血管活性药物
        'norepi_rate', 'epi_rate', 'dopa_rate',
        # 评分
        'sofa',
    ]
    
    # 加载数据
    try:
        data, patient_id = load_sample_data(
            data_path=str(data_path),
            concepts=concepts,
            patient_ids=[args.patient_id] if args.patient_id else None,
        )
        
        if args.patient_id:
            patient_id = args.patient_id
        
    except Exception as e:
        print(f"\n❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 运行演示
    try:
        demo_timeseries(data, patient_id, output_dir)
        demo_scores(data, patient_id, output_dir)
        demo_cohort(data, output_dir)
        demo_patient_dashboard(data, patient_id, args.database, output_dir)
        
    except Exception as e:
        print(f"\n⚠️ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 总结
    print("\n" + "=" * 60)
    print("✅ 演示完成！")
    print("=" * 60)
    
    html_files = list(output_dir.glob("*.html"))
    print(f"\n生成了 {len(html_files)} 个 HTML 文件:")
    for f in html_files:
        print(f"  📊 {f.name}")
    
    print(f"\n在浏览器中打开查看:")
    print(f"  file://{output_dir.absolute()}/")


if __name__ == "__main__":
    main()
