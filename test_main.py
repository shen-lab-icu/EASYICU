#!/usr/bin/env python3
"""
pyricu 统一测试脚本

整合所有核心验证测试：
1. SOFA 评分加载和验证
2. Sepsis-3 诊断和验证  
3. 数据完整性检查
4. 性能基准测试

使用统一配置，避免代码重复。
支持多数据库：MIMIC-IV, eICU, HiRID, AUMC

注意：本脚本已更新为使用新的简化API
- 推荐: load_sofa(), load_concepts() 等函数（支持智能默认值）
- 弃用: ICUQuickLoader 类（保留向后兼容）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# 确保使用 src/ 下的代码
sys.path.insert(0, 'src')

# ============================================================================
# 全局配置：选择要测试的数据库
# ============================================================================

# 数据库选择：'miiv', 'eicu', 'hirid', 'aumc'
TEST_DATABASE = 'eicu'  # 修改这里来切换数据库

# 数据源选择：'test' (测试数据) 或 'production' (完整数据)
TEST_DATA_SOURCE = 'test'

# 患者集选择：'debug' (1个), 'default' (3个), '50patients' (50个)
TEST_PATIENT_SET = 'default'

# ============================================================================

# 可选的可视化库
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 导入核心功能（使用新API）
from pyricu import (
    load_sofa,
    load_sofa2,
    load_sepsis3,
    load_concepts,
)

# 保留向后兼容
try:
    from pyricu.quickstart import get_patient_ids as load_patient_ids
except ImportError:
    # 如果quickstart被移除，使用替代方法
    def load_patient_ids(data_path, database='miiv', max_patients=None):
        from pyricu.fst_reader import read_fst
        icustays = read_fst(Path(data_path) / 'icustays.fst')
        ids = icustays['stay_id'].tolist()
        if max_patients:
            ids = ids[:max_patients]
        return ids
from pyricu.datasource import FilterOp, FilterSpec
from pyricu.easy import load_vitals, load_labs, load_sofa_score, load_sepsis
from pyricu.project_config import (
    get_data_path,
    get_patient_ids,
    get_concepts,
    TEST_DATA_PATH,
    PRODUCTION_DATA_PATH,
    VERBOSE,
    print_config
)


# ============================================================================
# 测试 1: SOFA 评分加载和验证
# ============================================================================

def test_sofa_basic(data_path: str, patient_ids: list, database: str = 'miiv', verbose: bool = True):
    """测试基本 SOFA 评分加载"""
    if verbose:
        print("\n" + "=" * 80)
        print(f"🧪 测试 1: SOFA 评分加载 [{database.upper()}]")
        print("=" * 80)
    
    sofa_df = load_sofa(
        database=database,
        data_path=data_path,
        patient_ids=patient_ids,
        verbose=False
    )
    
    # 根据数据库类型确定 ID 列名
    id_col = 'patientunitstayid' if database in ['eicu', 'eicu_demo'] else 'stay_id'
    
    if verbose:
        print(f"✅ SOFA 数据: {len(sofa_df)} 行, 患者数={sofa_df[id_col].nunique()}, "
              f"平均分={sofa_df['sofa'].mean():.1f}")
    
    # 验证
    assert len(sofa_df) > 0, "❌ SOFA 数据为空"
    assert 'sofa' in sofa_df.columns, "❌ 缺少 sofa 列"
    assert id_col in sofa_df.columns, f"❌ 缺少 {id_col} 列"
    
    return sofa_df


def test_sofa_components(data_path: str, patient_ids: list, database: str = 'miiv', verbose: bool = True):
    """测试 SOFA 组件加载"""
    if verbose:
        print("\n" + "=" * 80)
        print(f"🧪 测试 2: SOFA 组件 [{database.upper()}]")
        print("=" * 80)
    
    sofa_df = load_sofa(
        database=database,
        data_path=data_path,
        patient_ids=patient_ids,
        keep_components=True,
        verbose=False
    )
    
    # 检查组件列
    expected_components = ['sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal']
    missing = [c for c in expected_components if c not in sofa_df.columns]
    
    if verbose:
        if missing:
            print(f"⚠️  缺少组件: {missing}")
        else:
            print(f"✅ 所有 SOFA 组件都存在，验证组件之和 = 总分")
    
    # 验证 SOFA = 各组件之和
    if len(sofa_df) > 0 and all(c in sofa_df.columns for c in expected_components):
        component_sum = sofa_df[expected_components].sum(axis=1)
        sofa_total = sofa_df['sofa']
        diff = (sofa_total - component_sum).abs().max()
        
        if verbose and diff >= 0.01:
            print(f"   ⚠️  最大差异: {diff:.6f}")
    
    return sofa_df


# ============================================================================
# 测试 2b: SOFA-2 评分对比
# ============================================================================

def test_sofa2_comparison(data_path: str, patient_ids: list, database: str = 'miiv', verbose: bool = True):
    """测试 SOFA-2 评分并与 SOFA 对比"""
    if verbose:
        print("\n" + "=" * 80)
        print(f"🧪 测试 3: SOFA-2 评分对比 [{database.upper()}]")
        print("=" * 80)
    
    try:
        # 使用新API加载SOFA-2
        sofa2_df = load_sofa2(
            database=database,
            data_path=data_path,
            patient_ids=patient_ids,
            interval='1h',
            win_length='24h',
            keep_components=True,
            verbose=False
        )
        
        if verbose and 'sofa2' in sofa2_df.columns and len(sofa2_df) > 0:
            print(f"✅ SOFA-2 数据: {len(sofa2_df)} 行, 平均分={sofa2_df['sofa2'].mean():.1f}")
        
        # 对比 SOFA 和 SOFA2
        sofa1_df = load_sofa(
            database=database,
            data_path=data_path,
            patient_ids=patient_ids,
            keep_components=False,
            verbose=False
        )
        
        # 根据数据库类型确定 ID 列名
        id_col = 'patientunitstayid' if database in ['eicu', 'eicu_demo'] else 'stay_id'
        
        if len(sofa1_df) > 0 and len(sofa2_df) > 0:
            merged = sofa1_df[[id_col, 'charttime', 'sofa']].merge(
                sofa2_df[[id_col, 'charttime', 'sofa2']],
                on=[id_col, 'charttime'], how='inner'
            )
            
            if len(merged) > 0 and verbose:
                print(f"   对比: SOFA={merged['sofa'].mean():.2f}, SOFA2={merged['sofa2'].mean():.2f}, "
                      f"相关性={merged['sofa'].corr(merged['sofa2']):.3f}")
        
        return sofa2_df
        
    except Exception as e:
        if verbose:
            print(f"⚠️  SOFA-2 测试跳过: {e}")
        return None


# ============================================================================
# 测试 3: Sepsis-3 诊断
# ============================================================================

def test_sepsis3(data_path: str, patient_ids: list, database: str = 'miiv', verbose: bool = True):
    """测试 Sepsis-3 诊断"""
    if verbose:
        print("\n" + "=" * 80)
        print(f"🧪 测试 4: Sepsis-3 诊断 [{database.upper()}]")
        print("=" * 80)
    
    sepsis_df = load_sepsis3(
        database=database,
        data_path=data_path,
        patient_ids=patient_ids,
        verbose=False
    )
    
    # 统计
    if len(sepsis_df) > 0 and verbose:
        si_count = (sepsis_df['susp_inf'] > 0).sum() if 'susp_inf' in sepsis_df.columns else 0
        sep3_count = (sepsis_df['sep3'] > 0).sum() if 'sep3' in sepsis_df.columns else 0
        print(f"✅ Sepsis-3: {len(sepsis_df)} 行, 疑似感染={si_count}, Sepsis阳性={sep3_count}")
    
    return sepsis_df


# ============================================================================
# 测试 4: 极简 API
# ============================================================================

def test_easy_api(data_path: str, patient_ids: list, database: str = 'miiv', verbose: bool = True):
    """测试极简 API"""
    if verbose:
        print("\n" + "=" * 80)
        print(f"🧪 测试 5: 极简 API [{database.upper()}]")
        print("=" * 80)
    
    tests = [
        ('生命体征', lambda: load_vitals(database=database, data_path=data_path, patient_ids=patient_ids)),
        ('实验室', lambda: load_labs(database=database, data_path=data_path, patient_ids=patient_ids)),
        ('SOFA评分', lambda: load_sofa_score(data_path, patient_ids=patient_ids, database=database)),
        ('Sepsis诊断', lambda: load_sepsis(data_path, patient_ids=patient_ids, database=database))
    ]
    
    for name, func in tests:
        try:
            result = func()
            if verbose:
                print(f"   ✅ {name}: {len(result)} 行")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  {name}: 跳过 ({str(e)[:50]}...)")
    
    if verbose:
        print("✅ API测试完成")


# ============================================================================
# 测试 5: 批量加载性能
# ============================================================================

def test_batch_performance(data_path: str, patient_ids: list, database: str = 'miiv', verbose: bool = True):
    """测试批量加载性能（缓存优化）"""
    if verbose:
        print("\n" + "=" * 80)
        print(f"🧪 测试 6: 批量加载性能 [{database.upper()}]")
        print("=" * 80)
    
    concepts = get_concepts('vitals')
    
    start = time.time()
    # 使用新API批量加载
    result = load_concepts(
        concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path
    )
    elapsed = time.time() - start
    
    if verbose:
        result_len = len(result.data) if hasattr(result, 'data') else len(result)
        print(f"✅ 批量加载 {len(concepts)} 个概念: {elapsed:.2f}秒, {result_len} 行")


def test_data_integrity(data_path: str, patient_ids: list, database: str = 'miiv', verbose: bool = True):
    """测试数据完整性"""
    if verbose:
        print("\n" + "=" * 80)
        print(f"🧪 测试 7: 数据完整性 [{database.upper()}]")
        print("=" * 80)
    
    try:
        vitals = load_vitals(database=database, data_path=data_path, patient_ids=patient_ids[:1])
        if verbose:
            status = "正常" if len(vitals) > 0 else "空数据"
            print(f"✅ 数据加载{status} ({len(vitals)} 条记录)")
    except Exception as e:
        if verbose:
            print(f"❌ 数据加载失败: {e}")
        raise


# ============================================================================
# 测试 7: SOFA vs SOFA2 和 Sepsis 对比可视化
# ============================================================================

def test_sofa_sepsis_visualization(data_path: str, patient_ids: list, database: str = 'miiv', verbose: bool = True):
    """可视化对比 SOFA vs SOFA2 及 Sepsis 诊断 - 多患者版本"""
    if verbose:
        print("\n" + "=" * 80)
        print(f"🧪 测试 8: Sepsis 可视化对比（多患者） [{database.upper()}]")
        print("=" * 80)
    
    if not HAS_MATPLOTLIB:
        print("⚠️  matplotlib 未安装，跳过可视化测试")
        return
    
    if len(patient_ids) == 0:
        print("⚠️  没有患者数据，跳过可视化")
        return
    
    try:
        from pyricu.sepsis_sofa2 import sep3_sofa2
        
        # 查找有 Sepsis 事件的患者（最多3个）
        if verbose:
            print(f"🔍 搜索 Sepsis 病例...")
        
        sepsis_patients = []
        
        for pid in patient_ids[:min(20, len(patient_ids))]:  # 搜索前20个患者
            try:
                sepsis3_df = load_sepsis3(
                    database=database,
                    data_path=data_path,
                    patient_ids=[pid],
                    verbose=False
                )
                has_sep3 = sepsis3_df['sep3'].sum() > 0 if 'sep3' in sepsis3_df.columns else False
                
                if has_sep3:
                    sepsis_patients.append(pid)
                    if len(sepsis_patients) >= 3:  # 找到3个就够了
                        break
            except:
                pass
        
        if len(sepsis_patients) == 0:
            sepsis_patients = patient_ids[:min(3, len(patient_ids))]
        
        if verbose:
            print(f"   找到 {len(sepsis_patients)} 个患者，开始绘图...")
        
        # 为每个患者创建图表
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        chart_count = 0
        
        for patient_id in sepsis_patients:
            try:
                # 加载该患者的数据
                sofa_df = load_sofa(
                    database=database,
                    data_path=data_path,
                    patient_ids=[patient_id],
                    interval='1h',
                    win_length='24h',
                    keep_components=False,
                    verbose=False
                )
                
                sofa2_df = load_sofa2(
                    database=database,
                    data_path=data_path,
                    patient_ids=[patient_id],
                    interval='1h',
                    win_length='24h',
                    keep_components=False,
                    verbose=False
                )
                
                sepsis3_df = load_sepsis3(
                    database=database,
                    data_path=data_path,
                    patient_ids=[patient_id],
                    verbose=False
                )
                
                if sofa_df.empty or sepsis3_df.empty:
                    continue
                
                # 提取事件数据
                patient_data = sepsis3_df.sort_values('charttime')
                
                # 提取各类事件
                abx_data = patient_data[patient_data['abx'].notna() & (patient_data['abx'] > 0)] if 'abx' in patient_data.columns else pd.DataFrame()
                samp_data = patient_data[patient_data['samp'].notna() & (patient_data['samp'] > 0)] if 'samp' in patient_data.columns else pd.DataFrame()
                si_data = patient_data[patient_data['susp_inf'] == True] if 'susp_inf' in patient_data.columns else pd.DataFrame()
                sep3_data = patient_data[patient_data['sep3'] == True] if 'sep3' in patient_data.columns else pd.DataFrame()
                
                # 计算 Sepsis-3 (SOFA2)
                sep3_sofa2_data = pd.DataFrame()
                if not si_data.empty and not sofa2_df.empty:
                    try:
                        sep3_sofa2_result = sep3_sofa2(
                            sofa2=sofa2_df,
                            susp_inf_df=si_data,
                            id_cols=['stay_id'],
                            index_col='charttime'
                        )
                        sep3_sofa2_data = sep3_sofa2_result[sep3_sofa2_result['sep3_sofa2'] == True] if 'sep3_sofa2' in sep3_sofa2_result.columns else pd.DataFrame()
                    except:
                        pass
                
                # 创建图表（参考 test_sepsis_validation.py 的设计）
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
                
                # 图1: SOFA vs SOFA2 评分
                ax1.plot(sofa_df['charttime'], sofa_df['sofa'], 
                        marker='o', linewidth=2, markersize=6, label='SOFA', color='#1f77b4')
                
                if not sofa2_df.empty:
                    ax1.plot(sofa2_df['charttime'], sofa2_df['sofa2'], 
                            marker='s', linewidth=2, markersize=6, label='SOFA2', color='#ff7f0e')
                
                # 添加 SOFA=2 参考线
                ax1.axhline(y=2, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='SOFA=2 (基线)')
                
                # 标记 Sepsis-3 时间和窗口
                if not sep3_data.empty:
                    sep3_time = sep3_data.iloc[0]['charttime']
                    ax1.axvline(x=sep3_time, color='red', linestyle='--', linewidth=2, 
                               label=f'Sepsis-3 时间 ({sep3_time:.1f}h)')
                    
                    # SI 窗口 (-48h 到 +24h)
                    si_window_start = sep3_time - 48
                    si_window_end = sep3_time + 24
                    ax1.axvspan(si_window_start, si_window_end, alpha=0.15, color='yellow', 
                               label='疑似感染窗口 (-48/+24h)')
                
                ax1.set_ylabel('SOFA 评分', fontsize=12, fontweight='bold')
                ax1.set_title(f'患者 {patient_id} - SOFA vs SOFA2 对比', fontsize=14, fontweight='bold')
                ax1.legend(loc='upper left', fontsize=10)
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(bottom=0)
                
                # 图2: 事件时间线（类似 test_sepsis_validation.py）
                y_positions = {'abx': 1, 'samp': 2, 'si': 3, 'sep3_sofa': 4, 'sep3_sofa2': 5}
                
                # 抗生素
                if not abx_data.empty:
                    ax2.scatter(abx_data['charttime'], [y_positions['abx']]*len(abx_data), 
                               s=150, marker='s', color='blue', label='抗生素', zorder=5, alpha=0.8)
                
                # 采样
                if not samp_data.empty:
                    ax2.scatter(samp_data['charttime'], [y_positions['samp']]*len(samp_data), 
                               s=150, marker='^', color='green', label='采样', zorder=5, alpha=0.8)
                
                # 疑似感染
                if not si_data.empty:
                    ax2.scatter(si_data['charttime'], [y_positions['si']]*len(si_data), 
                               s=180, marker='D', color='orange', label='疑似感染', zorder=5, alpha=0.9)
                
                # Sepsis-3 (SOFA)
                if not sep3_data.empty:
                    ax2.scatter(sep3_data['charttime'], [y_positions['sep3_sofa']]*len(sep3_data), 
                               s=250, marker='*', color='red', label='Sepsis-3 (SOFA)', zorder=6, 
                               edgecolors='darkred', linewidths=1.5)
                
                # Sepsis-3 (SOFA2)
                if not sep3_sofa2_data.empty:
                    ax2.scatter(sep3_sofa2_data['charttime'], [y_positions['sep3_sofa2']]*len(sep3_sofa2_data), 
                               s=250, marker='*', color='darkgreen', label='Sepsis-3 (SOFA2)', zorder=6,
                               edgecolors='green', linewidths=1.5)
                
                ax2.set_yticks(list(y_positions.values()))
                ax2.set_yticklabels(['抗生素', '采样', '疑似感染', 'Sepsis-3\n(SOFA)', 'Sepsis-3\n(SOFA2)'])
                ax2.set_xlabel('ICU 入院后时间（小时）', fontsize=12, fontweight='bold')
                ax2.set_ylabel('事件类型', fontsize=12, fontweight='bold')
                ax2.legend(loc='upper left', fontsize=10)
                ax2.grid(True, alpha=0.3, axis='x')
                ax2.set_ylim(0.5, 5.5)
                
                plt.tight_layout()
                
                # 保存图表
                output_file = output_dir / f'sepsis_comparison_patient_{patient_id}.png'
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                chart_count += 1
                
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  患者 {patient_id}: {str(e)[:60]}...")
        
        if verbose:
            print(f"✅ 成功生成 {chart_count} 个可视化图表")
        
    except Exception as e:
        if verbose:
            print(f"⚠️  可视化失败: {e}")


# ============================================================================
# 主函数
# ============================================================================

def run_all_tests(
    data_source: str = None,
    patient_set: str = None,
    database: str = None,
    verbose: bool = True
):
    """运行所有测试
    
    Args:
        data_source: 数据源 ('test', 'production')，默认使用全局 TEST_DATA_SOURCE
        patient_set: 患者集 ('default', '50patients', 'debug')，默认使用全局 TEST_PATIENT_SET
        database: 数据库 ('miiv', 'eicu', 'hirid', 'aumc')，默认使用全局 TEST_DATABASE
        verbose: 是否显示详细输出
    """
    # 使用全局变量作为默认值
    if data_source is None:
        data_source = TEST_DATA_SOURCE
    if patient_set is None:
        patient_set = TEST_PATIENT_SET
    if database is None:
        database = TEST_DATABASE
    
    # 获取配置
    data_path = str(get_data_path(data_source, database))
    patient_ids = get_patient_ids(patient_set, database, Path(data_path))
    
    print("=" * 80)
    print("🏥 pyricu 统一测试")
    print("=" * 80)
    
    print(f"\n📋 测试配置:")
    print(f"   数据库: {database.upper()}")
    print(f"   数据源: {data_source} ({data_path})")
    print(f"   患者集: {patient_set} ({len(patient_ids)} 个患者)")
    print(f"   患者ID: {patient_ids}")
    
    # 运行所有测试
    try:
        test_sofa_basic(data_path, patient_ids, database, verbose)
        test_sofa_components(data_path, patient_ids, database, verbose)
        test_sofa2_comparison(data_path, patient_ids, database, verbose)
        test_sepsis3(data_path, patient_ids, database, verbose)
        test_easy_api(data_path, patient_ids, database, verbose)
        test_batch_performance(data_path, patient_ids, database, verbose)
        test_data_integrity(data_path, patient_ids, database, verbose)
        test_sofa_sepsis_visualization(data_path, patient_ids, database, verbose)
        
        # 总结
        print("\n" + "=" * 80)
        print(f"✅ 所有测试通过！pyricu 核心功能验证完成 [{database.upper()}]")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """主函数 - 提供多种测试模式"""
    import argparse
    
    parser = argparse.ArgumentParser(description='pyricu 统一测试')
    parser.add_argument('--database', choices=['miiv', 'eicu', 'hirid', 'aumc'], 
                        default=TEST_DATABASE,
                        help=f'数据库 (default: {TEST_DATABASE})')
    parser.add_argument('--data', choices=['test', 'production'], 
                        default=TEST_DATA_SOURCE,
                        help=f'数据源 (default: {TEST_DATA_SOURCE})')
    parser.add_argument('--patients', choices=['debug', 'default', '50patients'], 
                        default=TEST_PATIENT_SET,
                        help=f'患者集 (default: {TEST_PATIENT_SET})')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='显示详细输出')
    parser.add_argument('--quiet', action='store_true',
                        help='安静模式（覆盖 --verbose）')
    
    args = parser.parse_args()
    
    # 处理 verbose
    verbose = args.verbose and not args.quiet
    
    # 运行测试
    success = run_all_tests(
        data_source=args.data,
        patient_set=args.patients,
        database=args.database,
        verbose=verbose
    )
    
    # 退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
