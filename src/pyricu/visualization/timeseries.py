"""时序数据可视化。

提供时序数据的各种可视化图表，包括单概念时间线、多概念面板图、
药物给药甘特图和实验室检查热力图。
"""

from typing import Optional, Union, List, Dict, Any, Tuple

import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from .utils import (
    get_concept_label,
    get_concept_color,
    prepare_timeseries_data,
    create_time_axis_label,
    PYRICU_COLORS,
)


def plot_timeline(
    data: Union[pd.DataFrame, pd.Series],
    patient_id: Optional[Any] = None,
    value_col: Optional[str] = None,
    title: Optional[str] = None,
    show_markers: bool = True,
    highlight_abnormal: bool = False,
    normal_range: Optional[Tuple[float, float]] = None,
    color: Optional[str] = None,
    height: int = 400,
    width: int = 900,
) -> 'go.Figure':
    """绘制单概念时序图。
    
    Args:
        data: 包含时序数据的 DataFrame 或 Series
        patient_id: 患者ID（如果数据包含多个患者）
        value_col: 值列名称
        title: 图表标题
        show_markers: 是否显示数据点标记
        highlight_abnormal: 是否高亮异常值区域
        normal_range: 正常值范围 (min, max)
        color: 线条颜色
        height: 图表高度
        width: 图表宽度
        
    Returns:
        Plotly Figure 对象
        
    Example:
        >>> data = load_concepts(data_path, concepts=['hr'], patient_ids=[10001])
        >>> fig = plot_timeline(data['hr'], patient_id=10001, title="Heart Rate")
        >>> fig.show()
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Install with: pip install plotly")
    
    # 准备数据
    df = prepare_timeseries_data(data, patient_id, value_col)
    
    # 确定值列
    if value_col is None:
        # 找到第一个非ID、非时间的列
        exclude_cols = ['time', 'index', 'stay_id', 'hadm_id', 'subject_id', 
                       'icustay_id', 'patientunitstayid', 'admissionid', 'patientid']
        for col in df.columns:
            if col not in exclude_cols:
                value_col = col
                break
    
    if value_col is None or value_col not in df.columns:
        raise ValueError(f"Cannot find value column in data. Columns: {list(df.columns)}")
    
    # 设置颜色
    if color is None:
        color = get_concept_color(value_col)
    
    # 设置标题
    if title is None:
        title = get_concept_label(value_col)
        if patient_id is not None:
            title = f"{title} - Patient {patient_id}"
    
    # 创建图表
    fig = go.Figure()
    
    # 添加正常范围区域
    if highlight_abnormal and normal_range is not None:
        fig.add_hrect(
            y0=normal_range[0],
            y1=normal_range[1],
            fillcolor='green',
            opacity=0.1,
            line_width=0,
            annotation_text='Normal Range',
            annotation_position='top left',
        )
    
    # 添加时序线
    mode = 'lines+markers' if show_markers else 'lines'
    fig.add_trace(go.Scatter(
        x=df['time'] if 'time' in df.columns else df.index,
        y=df[value_col],
        mode=mode,
        name=get_concept_label(value_col),
        line=dict(color=color, width=2),
        marker=dict(size=6, color=color),
        hovertemplate='Time: %{x:.1f}h<br>Value: %{y:.2f}<extra></extra>',
    ))
    
    # 更新布局
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title=create_time_axis_label('hours'),
        yaxis_title=get_concept_label(value_col),
        height=height,
        width=width,
        showlegend=False,
        hovermode='x unified',
    )
    
    return fig


def plot_vitals_panel(
    data: Dict[str, pd.DataFrame],
    patient_id: Optional[Any] = None,
    concepts: Optional[List[str]] = None,
    title: Optional[str] = None,
    highlight_abnormal: bool = False,
    shared_xaxis: bool = True,
    height_per_row: int = 200,
    width: int = 1000,
) -> 'go.Figure':
    """绘制多概念面板图（生命体征仪表板）。
    
    Args:
        data: 概念数据字典 {concept_name: DataFrame}
        patient_id: 患者ID
        concepts: 要显示的概念列表，默认显示所有
        title: 图表标题
        highlight_abnormal: 是否高亮异常值
        shared_xaxis: 是否共享X轴
        height_per_row: 每行高度
        width: 图表宽度
        
    Returns:
        Plotly Figure 对象
        
    Example:
        >>> data = load_concepts(data_path, 
        ...     concepts=['hr', 'sbp', 'temp', 'resp', 'spo2'],
        ...     patient_ids=[10001]
        ... )
        >>> fig = plot_vitals_panel(data, patient_id=10001)
        >>> fig.show()
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Install with: pip install plotly")
    
    # 确定要显示的概念
    if concepts is None:
        concepts = list(data.keys())
    
    # 过滤有效数据
    valid_concepts = []
    for c in concepts:
        if c in data and data[c] is not None:
            df = data[c]
            if isinstance(df, pd.Series):
                df = df.reset_index()
            if len(df) > 0:
                valid_concepts.append(c)
    
    if not valid_concepts:
        raise ValueError("No valid concept data to plot")
    
    n_concepts = len(valid_concepts)
    
    # 创建子图
    fig = make_subplots(
        rows=n_concepts,
        cols=1,
        shared_xaxes=shared_xaxis,
        vertical_spacing=0.05,
        subplot_titles=[get_concept_label(c) for c in valid_concepts],
    )
    
    # 正常范围定义
    normal_ranges = {
        'hr': (60, 100),
        'sbp': (90, 140),
        'dbp': (60, 90),
        'map': (70, 100),
        'temp': (36.0, 37.5),
        'resp': (12, 20),
        'spo2': (94, 100),
    }
    
    # 添加每个概念的数据
    for i, concept in enumerate(valid_concepts, 1):
        df = prepare_timeseries_data(data[concept], patient_id, concept)
        
        if concept not in df.columns:
            # 尝试找到值列
            exclude = ['time', 'index', 'stay_id', 'hadm_id', 'subject_id']
            for col in df.columns:
                if col not in exclude:
                    df = df.rename(columns={col: concept})
                    break
        
        color = get_concept_color(concept)
        
        # 添加正常范围
        if highlight_abnormal and concept in normal_ranges:
            low, high = normal_ranges[concept]
            fig.add_hrect(
                y0=low, y1=high,
                fillcolor='green', opacity=0.1,
                line_width=0,
                row=i, col=1,
            )
        
        # 添加时序线
        time_data = df['time'] if 'time' in df.columns else df.index
        value_data = df[concept] if concept in df.columns else df.iloc[:, -1]
        
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=value_data,
                mode='lines+markers',
                name=concept,
                line=dict(color=color, width=1.5),
                marker=dict(size=4, color=color),
                showlegend=False,
            ),
            row=i, col=1,
        )
    
    # 设置标题
    if title is None:
        title = "Vital Signs Panel"
        if patient_id is not None:
            title = f"{title} - Patient {patient_id}"
    
    # 更新布局
    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=height_per_row * n_concepts + 100,
        width=width,
        showlegend=False,
    )
    
    # 更新X轴标签（只在最后一行显示）
    fig.update_xaxes(title_text=create_time_axis_label('hours'), row=n_concepts, col=1)
    
    return fig


def plot_medications_gantt(
    data: Dict[str, pd.DataFrame],
    patient_id: Optional[Any] = None,
    medications: Optional[List[str]] = None,
    title: Optional[str] = None,
    height: int = 400,
    width: int = 1000,
) -> 'go.Figure':
    """绘制药物给药甘特图。
    
    显示各种药物的给药时间段。
    
    Args:
        data: 药物数据字典，每个值应包含开始和结束时间
        patient_id: 患者ID
        medications: 要显示的药物列表
        title: 图表标题
        height: 图表高度
        width: 图表宽度
        
    Returns:
        Plotly Figure 对象
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Install with: pip install plotly")
    
    # 确定要显示的药物
    if medications is None:
        # 默认显示常见血管活性药物
        medications = ['norepi_rate', 'dopa_rate', 'dobu_rate', 'epi_rate', 'adh_rate']
        medications = [m for m in medications if m in data]
    
    if not medications:
        raise ValueError("No medication data to plot")
    
    fig = go.Figure()
    
    # 药物颜色
    med_colors = {
        'norepi_rate': '#e41a1c',
        'dopa_rate': '#377eb8',
        'dobu_rate': '#4daf4a',
        'epi_rate': '#984ea3',
        'adh_rate': '#ff7f00',
    }
    
    for idx, med in enumerate(medications):
        if med not in data or data[med] is None:
            continue
            
        df = prepare_timeseries_data(data[med], patient_id, med)
        
        if med not in df.columns:
            for col in df.columns:
                if 'rate' in col.lower() or col not in ['time', 'index']:
                    df = df.rename(columns={col: med})
                    break
        
        if med not in df.columns or len(df) == 0:
            continue
        
        # 过滤有值的时间点
        df = df[df[med] > 0].copy()
        
        if len(df) == 0:
            continue
        
        time_data = df['time'] if 'time' in df.columns else df.index
        
        color = med_colors.get(med, PYRICU_COLORS['primary'])
        
        # 添加标记
        fig.add_trace(go.Scatter(
            x=time_data,
            y=[idx] * len(time_data),
            mode='markers',
            name=get_concept_label(med),
            marker=dict(
                size=df[med] * 50 + 5,  # 大小反映剂量
                color=color,
                opacity=0.7,
            ),
            hovertemplate=f'{get_concept_label(med)}<br>' +
                         'Time: %{x:.1f}h<br>' +
                         'Rate: %{customdata:.3f}<extra></extra>',
            customdata=df[med],
        ))
    
    # 设置Y轴
    fig.update_yaxes(
        tickmode='array',
        tickvals=list(range(len(medications))),
        ticktext=[get_concept_label(m) for m in medications],
    )
    
    # 设置标题
    if title is None:
        title = "Medication Administration"
        if patient_id is not None:
            title = f"{title} - Patient {patient_id}"
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title=create_time_axis_label('hours'),
        yaxis_title='Medication',
        height=height,
        width=width,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    
    return fig


def plot_lab_heatmap(
    data: Dict[str, pd.DataFrame],
    patient_id: Optional[Any] = None,
    labs: Optional[List[str]] = None,
    title: Optional[str] = None,
    normalize: bool = True,
    height: int = 500,
    width: int = 1000,
) -> 'go.Figure':
    """绘制实验室检查热力图。
    
    显示多个实验室指标随时间的变化热力图。
    
    Args:
        data: 实验室数据字典
        patient_id: 患者ID
        labs: 要显示的实验室指标列表
        title: 图表标题
        normalize: 是否标准化值到0-1范围
        height: 图表高度
        width: 图表宽度
        
    Returns:
        Plotly Figure 对象
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Install with: pip install plotly")
    
    # 确定要显示的实验室指标
    if labs is None:
        labs = ['wbc', 'plt', 'crea', 'bili', 'lac', 'ph', 'pco2', 'po2']
        labs = [l for l in labs if l in data]
    
    if not labs:
        raise ValueError("No lab data to plot")
    
    # 准备热力图数据
    all_times = set()
    lab_data = {}
    
    for lab in labs:
        if lab not in data or data[lab] is None:
            continue
            
        df = prepare_timeseries_data(data[lab], patient_id, lab)
        
        if lab not in df.columns:
            for col in df.columns:
                if col not in ['time', 'index']:
                    df = df.rename(columns={col: lab})
                    break
        
        if lab not in df.columns:
            continue
        
        time_col = 'time' if 'time' in df.columns else df.index.name or 'index'
        times = df[time_col] if time_col in df.columns else df.index
        
        lab_data[lab] = pd.Series(df[lab].values, index=times)
        all_times.update(times)
    
    if not lab_data:
        raise ValueError("No valid lab data after processing")
    
    # 创建时间索引
    all_times = sorted(all_times)
    
    # 创建热力图矩阵
    matrix = []
    y_labels = []
    
    for lab in labs:
        if lab not in lab_data:
            continue
            
        series = lab_data[lab]
        row = []
        for t in all_times:
            if t in series.index:
                row.append(series[t])
            else:
                row.append(np.nan)
        
        # 标准化
        if normalize:
            row = np.array(row)
            valid = ~np.isnan(row)
            if valid.any():
                min_val, max_val = np.nanmin(row), np.nanmax(row)
                if max_val > min_val:
                    row = (row - min_val) / (max_val - min_val)
        
        matrix.append(row)
        y_labels.append(get_concept_label(lab))
    
    # 创建热力图
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=all_times,
        y=y_labels,
        colorscale='RdYlBu_r',
        hoverongaps=False,
        hovertemplate='Time: %{x:.1f}h<br>%{y}<br>Value: %{z:.2f}<extra></extra>',
    ))
    
    # 设置标题
    if title is None:
        title = "Laboratory Results Heatmap"
        if patient_id is not None:
            title = f"{title} - Patient {patient_id}"
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title=create_time_axis_label('hours'),
        height=height,
        width=width,
    )
    
    return fig
