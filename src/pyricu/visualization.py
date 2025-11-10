"""
Visualization utilities for pyricu

提供简单易用的可视化函数，封装复杂的绘图逻辑。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Union

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_sepsis_timeline(
    patient_id: Union[int, float],
    database: str = 'miiv',
    data_path: Optional[str] = None,
    output_dir: str = 'output',
    show_sofa2: bool = True,
    verbose: bool = False
) -> Optional[Path]:
    """绘制单个患者的Sepsis-3诊断时间线图
    
    Args:
        patient_id: 患者ID
        database: 数据库名称 ('miiv', 'eicu', 'aumc', 'hirid')
        data_path: 数据路径（可选，默认使用测试数据）
        output_dir: 输出目录
        show_sofa2: 是否显示SOFA2曲线
        verbose: 是否显示详细信息
        
    Returns:
        保存的图片路径，如果失败返回None
        
    Example:
        >>> from pyricu.visualization import plot_sepsis_timeline
        >>> plot_sepsis_timeline(31629173, database='miiv')
    """
    if not HAS_MATPLOTLIB:
        print("❌ matplotlib未安装，无法绘图")
        return None
    
    from pyricu import load_sofa, load_sofa2, load_sepsis3, load_concepts
    
    try:
        # 加载数据
        sofa_df = load_sofa(
            database=database,
            data_path=data_path,
            patient_ids=[patient_id],
            interval='1h',
            win_length='24h',
            verbose=verbose
        )
        
        if sofa_df.empty:
            if verbose:
                print(f"⚠️  患者 {patient_id}: 无SOFA数据")
            return None
        
        # 加载SOFA2（可选）
        sofa2_df = pd.DataFrame()
        if show_sofa2:
            try:
                sofa2_df = load_sofa2(
                    database=database,
                    data_path=data_path,
                    patient_ids=[patient_id],
                    interval='1h',
                    win_length='24h',
                    verbose=verbose
                )
            except:
                pass
        
        # 加载Sepsis-3诊断
        try:
            sep3_df = load_sepsis3(
                database=database,
                data_path=data_path,
                patient_ids=[patient_id],
                verbose=verbose
            )
        except:
            sep3_df = pd.DataFrame()
        
        # 加载事件数据
        abx_df = _safe_load_concept('abx', database, data_path, [patient_id], verbose)
        samp_df = _safe_load_concept('samp', database, data_path, [patient_id], verbose)
        susp_inf_df = _safe_load_concept('susp_inf', database, data_path, [patient_id], verbose)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # 确定时间列
        time_col = _get_time_column(sofa_df)
        
        # 图1: SOFA评分曲线
        ax1.plot(sofa_df[time_col], sofa_df['sofa'], 
                marker='o', linewidth=2, markersize=6, label='SOFA', color='#1f77b4')
        
        if not sofa2_df.empty and 'sofa2' in sofa2_df.columns:
            time_col2 = _get_time_column(sofa2_df)
            ax1.plot(sofa2_df[time_col2], sofa2_df['sofa2'], 
                    marker='s', linewidth=2, markersize=6, label='SOFA2', color='#ff7f0e')
        
        # SOFA=2参考线
        ax1.axhline(y=2, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, 
                   label='SOFA=2 (基线)')
        
        # 标记Sepsis-3时间
        if not sep3_df.empty:
            sep3_time_col = _get_time_column(sep3_df)
            sep3_time = sep3_df.iloc[0][sep3_time_col]
            ax1.axvline(x=sep3_time, color='red', linestyle='--', linewidth=2, 
                       label=f'Sepsis-3 ({sep3_time:.1f}h)')
            
            # 疑似感染窗口
            si_window_start = sep3_time - 48
            si_window_end = sep3_time + 24
            ax1.axvspan(si_window_start, si_window_end, alpha=0.15, color='yellow', 
                       label='疑似感染窗口')
        
        ax1.set_ylabel('SOFA 评分', fontsize=12, fontweight='bold')
        ax1.set_title(f'患者 {patient_id} - Sepsis-3 诊断时间线', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # 图2: 事件时间线
        y_positions = {'abx': 1, 'samp': 2, 'si': 3, 'sep3': 4}
        
        # 抗生素
        abx_times = _extract_times(abx_df, ['starttime', 'charttime'])
        if len(abx_times) > 0:
            ax2.scatter(abx_times, [y_positions['abx']]*len(abx_times), 
                       s=150, marker='s', color='blue', label='抗生素', zorder=5, alpha=0.8)
        
        # 血培养
        samp_times = _extract_times(samp_df, ['charttime', 'starttime'])
        if len(samp_times) > 0:
            ax2.scatter(samp_times, [y_positions['samp']]*len(samp_times), 
                       s=150, marker='^', color='green', label='血培养', zorder=5, alpha=0.8)
        
        # 疑似感染
        si_times = _extract_times(susp_inf_df, ['starttime', 'charttime'], filter_col='susp_inf')
        if len(si_times) > 0:
            ax2.scatter(si_times, [y_positions['si']]*len(si_times), 
                       s=180, marker='D', color='orange', label='疑似感染', zorder=5, alpha=0.9)
        
        # Sepsis-3诊断
        if not sep3_df.empty:
            sep3_times = _extract_times(sep3_df, [sep3_time_col], filter_col='sep3')
            if len(sep3_times) > 0:
                ax2.scatter(sep3_times, [y_positions['sep3']]*len(sep3_times), 
                           s=250, marker='*', color='red', label='Sepsis-3', zorder=6, 
                           edgecolors='darkred', linewidths=1.5)
        
        ax2.set_yticks(list(y_positions.values()))
        ax2.set_yticklabels(['抗生素', '血培养', '疑似感染', 'Sepsis-3'])
        ax2.set_xlabel('ICU 入院后时间（小时）', fontsize=12, fontweight='bold')
        ax2.set_ylabel('事件类型', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_ylim(0.5, 4.5)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        output_file = output_path / f'sepsis_timeline_{database}_{patient_id}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"✅ 图表已保存: {output_file}")
        
        return output_file
        
    except Exception as e:
        if verbose:
            print(f"❌ 绘图失败: {e}")
        return None


def plot_sepsis_batch(
    patient_ids: List[Union[int, float]],
    database: str = 'miiv',
    data_path: Optional[str] = None,
    output_dir: str = 'output',
    max_patients: int = 10,
    verbose: bool = True
) -> List[Path]:
    """批量绘制多个患者的Sepsis-3时间线图
    
    Args:
        patient_ids: 患者ID列表
        database: 数据库名称
        data_path: 数据路径
        output_dir: 输出目录
        max_patients: 最大绘图患者数
        verbose: 是否显示详细信息
        
    Returns:
        成功保存的图片路径列表
        
    Example:
        >>> from pyricu.visualization import plot_sepsis_batch
        >>> plot_sepsis_batch([31629173, 33072499], database='miiv')
    """
    if not HAS_MATPLOTLIB:
        print("❌ matplotlib未安装，无法绘图")
        return []
    
    success_files = []
    patient_ids = patient_ids[:max_patients]
    
    if verbose:
        print(f"📊 批量绘制 {len(patient_ids)} 个患者的时间线图...")
    
    for i, pid in enumerate(patient_ids, 1):
        if verbose:
            print(f"   [{i}/{len(patient_ids)}] 患者 {pid}...", end=' ')
        
        result = plot_sepsis_timeline(
            patient_id=pid,
            database=database,
            data_path=data_path,
            output_dir=output_dir,
            verbose=False
        )
        
        if result:
            success_files.append(result)
            if verbose:
                print("✅")
        else:
            if verbose:
                print("⚠️  跳过")
    
    if verbose:
        print(f"\n✅ 成功生成 {len(success_files)} 个图表")
    
    return success_files


# ============================================================================
# 辅助函数
# ============================================================================

def _safe_load_concept(concept_name, database, data_path, patient_ids, verbose):
    """安全地加载概念，失败返回空DataFrame"""
    from pyricu import load_concepts
    try:
        return load_concepts(concept_name, database=database, data_path=data_path, 
                           patient_ids=patient_ids, verbose=verbose)
    except:
        return pd.DataFrame()


def _get_time_column(df):
    """自动识别时间列"""
    for col in ['charttime', 'starttime', 'measuredat', 'time']:
        if col in df.columns:
            return col
    # 如果都没有，返回第二列（假设第一列是ID）
    if len(df.columns) > 1:
        return df.columns[1]
    return df.columns[0]


def _extract_times(df, time_cols, filter_col=None):
    """从DataFrame提取时间点"""
    if df.empty:
        return []
    
    # 应用过滤
    if filter_col and filter_col in df.columns:
        df = df[df[filter_col].notna() & (df[filter_col] != False) & (df[filter_col] != 0)]
    
    # 查找时间列
    time_col = None
    for col in time_cols:
        if col in df.columns:
            time_col = col
            break
    
    if time_col is None:
        return []
    
    times = df[time_col].dropna().tolist()
    return times
