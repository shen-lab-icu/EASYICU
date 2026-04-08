#!/usr/bin/env python
"""EasyICU 快速上手示例。

最简化的演示脚本，展示如何在几行代码内使用 EasyICU。
"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# ============================================================
# 示例 1: 加载数据
# ============================================================
print("=" * 60)
print("示例 1: 加载 ICU 数据")
print("=" * 60)

from easyicu import load_concepts

# 定义数据路径（请修改为您的实际路径）
DATA_PATH = "/home/zhuhb/project/ricu_to_python/ricu_data/miiv"

# 加载心率和 SOFA 数据
print("\n加载 concepts: hr, map, sofa ...")
data = load_concepts(
    data_path=DATA_PATH,
    concepts=['hr', 'map', 'sofa'],
    verbose=True,
)

print(f"\n成功加载 {len(data)} 个 concepts:")
for name, df in data.items():
    print(f"  - {name}: {len(df)} 条记录")


# ============================================================
# 示例 2: 可视化 - 时序图
# ============================================================
print("\n" + "=" * 60)
print("示例 2: 绘制时序图")
print("=" * 60)

try:
    from easyicu.visualization import plot_timeline
    
    # 获取一个患者 ID
    hr_df = data['hr']
    patient_id = hr_df['stay_id'].iloc[0] if 'stay_id' in hr_df.columns else None
    
    if patient_id:
        print(f"\n为患者 {patient_id} 绘制心率时序图...")
        fig = plot_timeline(hr_df, patient_id=patient_id, title="Heart Rate")
        
        # 保存为 HTML
        output_path = Path("./demo_output/quick_hr.html")
        output_path.parent.mkdir(exist_ok=True)
        fig.write_html(str(output_path))
        print(f"保存到: {output_path}")
        
        # 如果在交互环境，可以直接显示
        # fig.show()
    else:
        print("⚠️ 未找到患者 ID")

except ImportError as e:
    print(f"⚠️ 可视化依赖未安装: {e}")
    print("   请运行: pip install easyicu[viz]")


# ============================================================
# 示例 3: 可视化 - SOFA 分解图
# ============================================================
print("\n" + "=" * 60)
print("示例 3: 绘制 SOFA 分解图")
print("=" * 60)

try:
    from easyicu.visualization import plot_sofa_breakdown
    
    if 'sofa' in data and patient_id:
        sofa_df = data['sofa']
        
        print(f"\n为患者 {patient_id} 绘制 SOFA 分解图...")
        fig = plot_sofa_breakdown(sofa_df, patient_id=patient_id)
        
        output_path = Path("./demo_output/quick_sofa.html")
        fig.write_html(str(output_path))
        print(f"保存到: {output_path}")

except ImportError:
    print("⚠️ 可视化依赖未安装")


# ============================================================
# 示例 4: 患者仪表盘
# ============================================================
print("\n" + "=" * 60)
print("示例 4: 生成患者仪表盘")
print("=" * 60)

try:
    from easyicu.visualization import PatientDashboard
    
    if patient_id:
        print(f"\n为患者 {patient_id} 生成综合仪表盘...")
        
        dashboard = PatientDashboard(patient_id=patient_id, database='miiv')
        dashboard.load_data(data)
        
        fig = dashboard.render_full_dashboard()
        
        output_path = Path("./demo_output/quick_dashboard.html")
        fig.write_html(str(output_path))
        print(f"保存到: {output_path}")

except ImportError:
    print("⚠️ 可视化依赖未安装")
except Exception as e:
    print(f"⚠️ 仪表盘生成失败: {e}")


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("✅ 快速上手演示完成！")
print("=" * 60)

output_dir = Path("./demo_output")
if output_dir.exists():
    html_files = list(output_dir.glob("quick_*.html"))
    if html_files:
        print(f"\n生成的文件 ({len(html_files)} 个):")
        for f in html_files:
            print(f"  📊 {f}")
        print(f"\n在浏览器中打开查看图表")

print("\n下一步:")
print("  1. 运行 demo_visualization.py 查看更多可视化示例")
print("  2. 运行 demo_webapp.py 启动交互式 Web 应用")
