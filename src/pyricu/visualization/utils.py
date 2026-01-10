"""可视化工具函数。

提供统一主题设置、图表保存等通用功能。
"""

from typing import Optional, Union, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# PyRICU 默认配色方案
PYRICU_COLORS = {
    'primary': '#1f77b4',      # 蓝色
    'secondary': '#ff7f0e',    # 橙色
    'success': '#2ca02c',      # 绿色
    'danger': '#d62728',       # 红色
    'warning': '#ffbb33',      # 黄色
    'info': '#17a2b8',         # 青色
    'light': '#f8f9fa',
    'dark': '#343a40',
    
    # 器官系统颜色 (SOFA)
    'respiratory': '#e41a1c',
    'coagulation': '#377eb8',
    'liver': '#4daf4a',
    'cardiovascular': '#984ea3',
    'cns': '#ff7f00',
    'renal': '#a65628',
    
    # 生命体征颜色
    'hr': '#e41a1c',
    'sbp': '#377eb8',
    'dbp': '#4daf4a',
    'map': '#984ea3',
    'temp': '#ff7f00',
    'resp': '#a65628',
    'spo2': '#f781bf',
}

# 概念显示名称
CONCEPT_LABELS = {
    'hr': 'Heart Rate (bpm)',
    'sbp': 'Systolic BP (mmHg)',
    'dbp': 'Diastolic BP (mmHg)',
    'map': 'Mean Arterial Pressure (mmHg)',
    'temp': 'Temperature (°C)',
    'resp': 'Respiratory Rate (/min)',
    'spo2': 'SpO2 (%)',
    'fio2': 'FiO2 (%)',
    'gcs': 'Glasgow Coma Scale',
    'sofa': 'SOFA Score',
    'sofa_resp': 'SOFA Respiratory',
    'sofa_coag': 'SOFA Coagulation',
    'sofa_liver': 'SOFA Liver',
    'sofa_cardio': 'SOFA Cardiovascular',
    'sofa_cns': 'SOFA CNS',
    'sofa_renal': 'SOFA Renal',
    'crea': 'Creatinine (mg/dL)',
    'bili': 'Bilirubin (mg/dL)',
    'plt': 'Platelets (×10³/μL)',
    'wbc': 'WBC (×10³/μL)',
    'lac': 'Lactate (mmol/L)',
    'pafi': 'PaO2/FiO2 Ratio',
    'urine24': 'Urine Output (mL/24h)',
    'norepi_rate': 'Norepinephrine (mcg/kg/min)',
    'dopa_rate': 'Dopamine (mcg/kg/min)',
    'dobu_rate': 'Dobutamine (mcg/kg/min)',
    'epi_rate': 'Epinephrine (mcg/kg/min)',
}


def setup_theme(theme: str = 'pyricu') -> None:
    """设置 Plotly 主题。
    
    Args:
        theme: 主题名称 ('pyricu', 'plotly', 'plotly_white', 'plotly_dark')
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Install with: pip install plotly")
    
    if theme == 'pyricu':
        # 自定义 PyRICU 主题
        pio.templates['pyricu'] = go.layout.Template(
            layout=go.Layout(
                font=dict(family='Arial, sans-serif', size=12),
                title=dict(font=dict(size=16, color='#333')),
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    showgrid=True,
                    gridcolor='#eee',
                    showline=True,
                    linecolor='#ccc',
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#eee',
                    showline=True,
                    linecolor='#ccc',
                ),
                colorway=[
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                    '#bcbd22', '#17becf'
                ],
                legend=dict(
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#ccc',
                    borderwidth=1,
                ),
            )
        )
        pio.templates.default = 'pyricu'
    else:
        pio.templates.default = theme


def save_figure(
    fig: 'go.Figure',
    filepath: Union[str, Path],
    format: Optional[str] = None,
    width: int = 1200,
    height: int = 800,
    scale: float = 2.0,
) -> None:
    """保存图表到文件。
    
    Args:
        fig: Plotly Figure 对象
        filepath: 保存路径
        format: 文件格式 (png, svg, pdf, html)，默认从扩展名推断
        width: 图片宽度 (像素)
        height: 图片高度 (像素)
        scale: 缩放比例 (用于高分辨率输出)
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Install with: pip install plotly")
    
    filepath = Path(filepath)
    
    if format is None:
        format = filepath.suffix.lstrip('.').lower()
    
    if format == 'html':
        fig.write_html(filepath, include_plotlyjs='cdn')
    elif format in ('png', 'svg', 'pdf', 'jpeg', 'webp'):
        try:
            fig.write_image(
                filepath,
                format=format,
                width=width,
                height=height,
                scale=scale,
            )
        except ValueError as e:
            if 'kaleido' in str(e).lower():
                raise ImportError(
                    "kaleido is required for static image export. "
                    "Install with: pip install kaleido"
                ) from e
            raise
    else:
        raise ValueError(f"Unsupported format: {format}")


def get_concept_label(concept: str) -> str:
    """获取概念的显示标签。"""
    return CONCEPT_LABELS.get(concept, concept.replace('_', ' ').title())


def get_concept_color(concept: str) -> str:
    """获取概念的颜色。"""
    return PYRICU_COLORS.get(concept, PYRICU_COLORS['primary'])


def detect_id_col(df: pd.DataFrame) -> Optional[str]:
    """检测 DataFrame 中的 ID 列。"""
    id_candidates = [
        'stay_id', 'hadm_id', 'subject_id', 'icustay_id',
        'patientunitstayid', 'admissionid', 'patientid'
    ]
    for col in id_candidates:
        if col in df.columns:
            return col
    return None


def detect_time_col(df: pd.DataFrame) -> Optional[str]:
    """检测 DataFrame 中的时间列。"""
    time_candidates = ['time', 'charttime', 'index_var', 'starttime', 'endtime']
    for col in time_candidates:
        if col in df.columns:
            return col
    return None


def prepare_timeseries_data(
    data: Union[pd.DataFrame, pd.Series],
    patient_id: Optional[Any] = None,
    value_col: Optional[str] = None,
) -> pd.DataFrame:
    """准备时序数据用于可视化。
    
    Args:
        data: 输入数据
        patient_id: 可选的患者ID过滤
        value_col: 值列名称
        
    Returns:
        标准化的 DataFrame，包含 time 和 value 列
    """
    # 如果是 Series，转换为 DataFrame
    if isinstance(data, pd.Series):
        df = data.reset_index()
        if value_col is None:
            value_col = data.name or 'value'
    else:
        df = data.copy()
    
    # 检测 ID 列
    id_col = detect_id_col(df)
    
    # 按患者过滤
    if patient_id is not None and id_col is not None:
        df = df[df[id_col] == patient_id].copy()
    
    # 检测时间列
    time_col = detect_time_col(df)
    if time_col is None and 'index' in df.columns:
        time_col = 'index'
    
    # 标准化列名
    if time_col and time_col != 'time':
        df = df.rename(columns={time_col: 'time'})
    
    # 确保时间列是数值类型
    if 'time' in df.columns:
        if pd.api.types.is_timedelta64_dtype(df['time']):
            df['time'] = df['time'].dt.total_seconds() / 3600  # 转换为小时
    
    return df


def create_time_axis_label(unit: str = 'hours') -> str:
    """创建时间轴标签。"""
    labels = {
        'hours': 'Time (hours from ICU admission)',
        'days': 'Time (days from ICU admission)',
        'minutes': 'Time (minutes from ICU admission)',
    }
    return labels.get(unit, f'Time ({unit})')
