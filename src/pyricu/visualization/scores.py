"""评分系统可视化。

提供 SOFA、SIRS、qSOFA 等评分系统的可视化图表。
"""

from typing import Optional, Union, List, Dict, Any

import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from .utils import (
    prepare_timeseries_data,
    create_time_axis_label,
)


# SOFA 组件颜色
SOFA_COLORS = {
    'sofa_resp': '#e41a1c',      # 红色 - 呼吸
    'sofa_coag': '#377eb8',      # 蓝色 - 凝血
    'sofa_liver': '#4daf4a',     # 绿色 - 肝脏
    'sofa_cardio': '#984ea3',    # 紫色 - 心血管
    'sofa_cns': '#ff7f00',       # 橙色 - 神经系统
    'sofa_renal': '#a65628',     # 棕色 - 肾脏
}

SOFA_LABELS = {
    'sofa_resp': 'Respiratory',
    'sofa_coag': 'Coagulation',
    'sofa_liver': 'Liver',
    'sofa_cardio': 'Cardiovascular',
    'sofa_cns': 'CNS',
    'sofa_renal': 'Renal',
}


def plot_sofa_breakdown(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    patient_id: Optional[Any] = None,
    title: Optional[str] = None,
    show_total: bool = True,
    stacked: bool = True,
    height: int = 500,
    width: int = 1000,
) -> 'go.Figure':
    """绘制 SOFA 评分分解堆叠图。
    
    显示 SOFA 总分及各器官系统分量的时间变化。
    
    Args:
        data: SOFA 数据，可以是 DataFrame（包含分量列）或字典
        patient_id: 患者ID
        title: 图表标题
        show_total: 是否显示总分线
        stacked: 是否使用堆叠面积图
        height: 图表高度
        width: 图表宽度
        
    Returns:
        Plotly Figure 对象
        
    Example:
        >>> sofa = load_concepts(data_path, concepts=['sofa'], 
        ...                      patient_ids=[10001], keep_components=True)
        >>> fig = plot_sofa_breakdown(sofa['sofa'], patient_id=10001)
        >>> fig.show()
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Install with: pip install plotly")
    
    # 处理输入数据
    if isinstance(data, dict):
        # 合并所有 SOFA 组件
        dfs = []
        for key, df in data.items():
            if 'sofa' in key.lower():
                dfs.append(prepare_timeseries_data(df, patient_id))
        if dfs:
            data = pd.concat(dfs, axis=1)
            data = data.loc[:, ~data.columns.duplicated()]
    else:
        data = prepare_timeseries_data(data, patient_id)
    
    if isinstance(data, pd.Series):
        data = data.reset_index()
    
    # 确定时间列
    time_col = 'time' if 'time' in data.columns else data.index
    if isinstance(time_col, str):
        time_data = data[time_col]
    else:
        time_data = data.index
    
    # 找到 SOFA 组件列
    sofa_components = ['sofa_resp', 'sofa_coag', 'sofa_liver', 
                       'sofa_cardio', 'sofa_cns', 'sofa_renal']
    available_components = [c for c in sofa_components if c in data.columns]
    
    # 检查是否有带 _comp 后缀的列
    if not available_components:
        for c in sofa_components:
            comp_col = c + '_comp'
            if comp_col in data.columns:
                data = data.rename(columns={comp_col: c})
                available_components.append(c)
    
    fig = go.Figure()
    
    if available_components:
        if stacked:
            # 堆叠面积图
            for component in available_components:
                fig.add_trace(go.Scatter(
                    x=time_data,
                    y=data[component],
                    mode='lines',
                    name=SOFA_LABELS.get(component, component),
                    line=dict(width=0.5, color=SOFA_COLORS.get(component)),
                    stackgroup='sofa',
                    fillcolor=SOFA_COLORS.get(component),
                    hovertemplate=f'{SOFA_LABELS.get(component, component)}: %{{y}}<extra></extra>',
                ))
        else:
            # 多线图
            for component in available_components:
                fig.add_trace(go.Scatter(
                    x=time_data,
                    y=data[component],
                    mode='lines+markers',
                    name=SOFA_LABELS.get(component, component),
                    line=dict(color=SOFA_COLORS.get(component), width=2),
                    marker=dict(size=4),
                ))
    
    # 添加总分线
    if show_total and 'sofa' in data.columns:
        fig.add_trace(go.Scatter(
            x=time_data,
            y=data['sofa'],
            mode='lines+markers',
            name='Total SOFA',
            line=dict(color='black', width=3, dash='dot'),
            marker=dict(size=8, symbol='diamond'),
        ))
    
    # 设置标题
    if title is None:
        title = "SOFA Score Breakdown"
        if patient_id is not None:
            title = f"{title} - Patient {patient_id}"
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title=create_time_axis_label('hours'),
        yaxis_title='SOFA Score',
        height=height,
        width=width,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
        ),
        hovermode='x unified',
    )
    
    # 设置 Y 轴范围
    fig.update_yaxes(range=[0, 24], dtick=4)
    
    return fig


def plot_sofa_trajectory(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    patient_ids: Optional[List[Any]] = None,
    title: Optional[str] = None,
    show_mean: bool = True,
    show_ci: bool = True,
    height: int = 500,
    width: int = 1000,
) -> 'go.Figure':
    """绘制多患者 SOFA 轨迹图。
    
    显示多个患者的 SOFA 评分轨迹，可选显示均值和置信区间。
    
    Args:
        data: SOFA 数据
        patient_ids: 要显示的患者ID列表
        title: 图表标题
        show_mean: 是否显示均值线
        show_ci: 是否显示置信区间
        height: 图表高度
        width: 图表宽度
        
    Returns:
        Plotly Figure 对象
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Install with: pip install plotly")
    
    # 准备数据
    if isinstance(data, dict) and 'sofa' in data:
        df = data['sofa']
    else:
        df = data
    
    if isinstance(df, pd.Series):
        df = df.reset_index()
    
    # 检测 ID 列
    id_candidates = ['stay_id', 'hadm_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid']
    id_col = None
    for col in id_candidates:
        if col in df.columns:
            id_col = col
            break
    
    # 确定时间列和值列
    time_col = 'time' if 'time' in df.columns else 'index'
    value_col = 'sofa' if 'sofa' in df.columns else df.columns[-1]
    
    fig = go.Figure()
    
    if id_col is not None:
        # 多患者数据
        unique_ids = df[id_col].unique()
        if patient_ids is not None:
            unique_ids = [pid for pid in patient_ids if pid in unique_ids]
        
        # 绘制个体轨迹
        for pid in unique_ids[:50]:  # 限制最多显示50个患者
            patient_data = df[df[id_col] == pid]
            fig.add_trace(go.Scatter(
                x=patient_data[time_col],
                y=patient_data[value_col],
                mode='lines',
                name=f'Patient {pid}',
                line=dict(width=1, color='lightgray'),
                opacity=0.5,
                showlegend=False,
                hoverinfo='skip',
            ))
        
        # 计算均值和置信区间
        if show_mean or show_ci:
            grouped = df.groupby(time_col)[value_col]
            mean_values = grouped.mean()
            std_values = grouped.std()
            
            if show_ci:
                # 95% 置信区间
                upper = mean_values + 1.96 * std_values / np.sqrt(grouped.count())
                lower = mean_values - 1.96 * std_values / np.sqrt(grouped.count())
                
                fig.add_trace(go.Scatter(
                    x=mean_values.index.tolist() + mean_values.index.tolist()[::-1],
                    y=upper.tolist() + lower.tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(31, 119, 180, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% CI',
                    showlegend=True,
                ))
            
            if show_mean:
                fig.add_trace(go.Scatter(
                    x=mean_values.index,
                    y=mean_values.values,
                    mode='lines+markers',
                    name='Mean SOFA',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=6),
                ))
    else:
        # 单患者数据
        fig.add_trace(go.Scatter(
            x=df[time_col] if time_col in df.columns else df.index,
            y=df[value_col],
            mode='lines+markers',
            name='SOFA Score',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6),
        ))
    
    # 设置标题
    if title is None:
        title = "SOFA Score Trajectory"
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title=create_time_axis_label('hours'),
        yaxis_title='SOFA Score',
        height=height,
        width=width,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
        ),
    )
    
    fig.update_yaxes(range=[0, 24], dtick=4)
    
    return fig


def plot_sepsis_timeline(
    sofa_data: pd.DataFrame,
    sepsis_data: Optional[pd.DataFrame] = None,
    patient_id: Optional[Any] = None,
    title: Optional[str] = None,
    show_threshold: bool = True,
    height: int = 500,
    width: int = 1000,
) -> 'go.Figure':
    """绘制 Sepsis-3 事件时间线。
    
    显示 SOFA 评分变化并标注 Sepsis-3 诊断事件。
    
    Args:
        sofa_data: SOFA 评分数据
        sepsis_data: Sepsis-3 诊断数据（包含 sep3 事件时间）
        patient_id: 患者ID
        title: 图表标题
        show_threshold: 是否显示 SOFA≥2 阈值线
        height: 图表高度
        width: 图表宽度
        
    Returns:
        Plotly Figure 对象
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Install with: pip install plotly")
    
    # 准备 SOFA 数据
    sofa_df = prepare_timeseries_data(sofa_data, patient_id, 'sofa')
    
    # 确定列名
    time_col = 'time' if 'time' in sofa_df.columns else 'index'
    value_col = 'sofa' if 'sofa' in sofa_df.columns else sofa_df.columns[-1]
    
    fig = go.Figure()
    
    # 添加 SOFA 曲线
    time_data = sofa_df[time_col] if time_col in sofa_df.columns else sofa_df.index
    
    fig.add_trace(go.Scatter(
        x=time_data,
        y=sofa_df[value_col],
        mode='lines+markers',
        name='SOFA Score',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)',
    ))
    
    # 添加阈值线
    if show_threshold:
        fig.add_hline(
            y=2,
            line_dash='dash',
            line_color='red',
            annotation_text='SOFA ≥ 2 (Sepsis Threshold)',
            annotation_position='top right',
        )
    
    # 标注 Sepsis-3 事件
    if sepsis_data is not None and len(sepsis_data) > 0:
        sep_df = prepare_timeseries_data(sepsis_data, patient_id)
        
        if 'sep3' in sep_df.columns:
            sep_events = sep_df[sep_df['sep3'].fillna(False)]
        else:
            sep_events = sep_df
        
        if len(sep_events) > 0:
            sep_time = sep_events[time_col].iloc[0] if time_col in sep_events.columns else sep_events.index[0]
            
            # 获取该时刻的 SOFA 值
            closest_idx = (sofa_df[time_col] - sep_time).abs().idxmin() if time_col in sofa_df.columns else 0
            sep_sofa = sofa_df.loc[closest_idx, value_col]
            
            # 添加垂直线
            fig.add_vline(
                x=sep_time,
                line_dash='dot',
                line_color='red',
                line_width=2,
            )
            
            # 添加标注
            fig.add_annotation(
                x=sep_time,
                y=sep_sofa,
                text=f"Sepsis-3 Onset<br>Time: {sep_time:.1f}h",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowcolor='red',
                font=dict(color='red', size=12),
                bgcolor='white',
                bordercolor='red',
            )
    
    # 设置标题
    if title is None:
        title = "Sepsis-3 Timeline"
        if patient_id is not None:
            title = f"{title} - Patient {patient_id}"
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title=create_time_axis_label('hours'),
        yaxis_title='SOFA Score',
        height=height,
        width=width,
        showlegend=True,
    )
    
    fig.update_yaxes(range=[0, max(24, sofa_df[value_col].max() + 2)], dtick=4)
    
    return fig
