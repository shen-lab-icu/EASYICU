"""队列级可视化。

提供数据质量评估、群体统计和跨数据库对比的可视化图表。
"""

from typing import Optional, Union, List, Dict, Any

import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from .utils import (
    get_concept_label,
    get_concept_color,
    PYRICU_COLORS,
    CONCEPT_LABELS,
)


def plot_missing_heatmap(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    concepts: Optional[List[str]] = None,
    title: Optional[str] = None,
    height: int = 600,
    width: int = 1000,
    colorscale: str = 'RdYlGn_r',
) -> 'go.Figure':
    """绘制缺失值热力图。
    
    显示各 concept 在不同时间窗口的缺失率。
    
    Args:
        data: 特征数据，可以是 DataFrame 或 dict
        concepts: 要显示的 concept 列表
        title: 图表标题
        height: 图表高度
        width: 图表宽度
        colorscale: 颜色方案
        
    Returns:
        Plotly Figure 对象
        
    Example:
        >>> features = load_concepts(data_path, concepts=['hr', 'map', 'temp'])
        >>> fig = plot_missing_heatmap(features)
        >>> fig.show()
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Install with: pip install plotly")
    
    # 整合数据
    if isinstance(data, dict):
        missing_rates = {}
        for concept, df in data.items():
            if concepts is not None and concept not in concepts:
                continue
            if isinstance(df, pd.DataFrame):
                # 计算整体缺失率
                if len(df.columns) > 0:
                    value_cols = [c for c in df.columns if c not in ['stay_id', 'hadm_id', 'time', 'index']]
                    if value_cols:
                        missing_rates[concept] = df[value_cols[0]].isna().mean() * 100
                    else:
                        missing_rates[concept] = 0
            elif isinstance(df, pd.Series):
                missing_rates[concept] = df.isna().mean() * 100
        
        # 创建简单条形图
        concepts_list = list(missing_rates.keys())
        rates = list(missing_rates.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=concepts_list,
                y=rates,
                marker_color=[get_concept_color(c) for c in concepts_list],
                text=[f'{r:.1f}%' for r in rates],
                textposition='outside',
            )
        ])
        
        fig.update_layout(
            title=dict(text=title or "Missing Rate by Concept", x=0.5),
            xaxis_title="Concept",
            yaxis_title="Missing Rate (%)",
            height=height,
            width=width,
        )
        fig.update_yaxes(range=[0, 100])
        
        return fig
    
    else:
        # DataFrame 输入 - 按列计算缺失率
        df = data
        
        # 识别 ID 和时间列
        id_cols = ['stay_id', 'hadm_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid']
        time_cols = ['time', 'index', 'charttime']
        
        value_cols = [c for c in df.columns if c not in id_cols + time_cols]
        
        if concepts is not None:
            value_cols = [c for c in value_cols if c in concepts]
        
        # 计算每列缺失率
        missing_rates = {col: df[col].isna().mean() * 100 for col in value_cols}
        
        # 排序
        sorted_cols = sorted(missing_rates.keys(), key=lambda x: missing_rates[x], reverse=True)
        rates = [missing_rates[c] for c in sorted_cols]
        
        fig = go.Figure(data=[
            go.Bar(
                x=sorted_cols,
                y=rates,
                marker_color=[get_concept_color(c) for c in sorted_cols],
                text=[f'{r:.1f}%' for r in rates],
                textposition='outside',
            )
        ])
        
        fig.update_layout(
            title=dict(text=title or "Missing Rate by Column", x=0.5),
            xaxis_title="Column",
            yaxis_title="Missing Rate (%)",
            height=height,
            width=width,
            xaxis_tickangle=-45,
        )
        fig.update_yaxes(range=[0, 100])
        
        return fig


def plot_concept_distribution(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    concept: str,
    by_database: bool = False,
    title: Optional[str] = None,
    bins: int = 50,
    height: int = 500,
    width: int = 800,
) -> 'go.Figure':
    """绘制 concept 数值分布图。
    
    显示指定 concept 的分布直方图或密度图。
    
    Args:
        data: 特征数据
        concept: concept 名称
        by_database: 是否按数据库分组
        title: 图表标题
        bins: 直方图分箱数
        height: 图表高度
        width: 图表宽度
        
    Returns:
        Plotly Figure 对象
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Install with: pip install plotly")
    
    # 提取数据
    if isinstance(data, dict):
        if concept in data:
            df = data[concept]
        else:
            raise ValueError(f"Concept '{concept}' not found in data")
    else:
        df = data
        if concept not in df.columns:
            raise ValueError(f"Concept '{concept}' not found in columns")
    
    # 获取值列
    if isinstance(df, pd.DataFrame):
        if concept in df.columns:
            values = df[concept].dropna()
        else:
            # 找最后一个非 ID 列
            value_cols = [c for c in df.columns if c not in ['stay_id', 'hadm_id', 'time', 'index']]
            if value_cols:
                values = df[value_cols[-1]].dropna()
            else:
                values = df.iloc[:, -1].dropna()
    else:
        values = df.dropna()
    
    fig = go.Figure()
    
    if by_database and 'database' in df.columns if isinstance(df, pd.DataFrame) else False:
        # 按数据库分组
        databases = df['database'].unique()
        colors = px.colors.qualitative.Set1
        
        for i, db in enumerate(databases):
            db_values = df[df['database'] == db][concept].dropna()
            fig.add_trace(go.Histogram(
                x=db_values,
                name=db.upper(),
                opacity=0.7,
                marker_color=colors[i % len(colors)],
                nbinsx=bins,
            ))
        
        fig.update_layout(barmode='overlay')
    else:
        # 单一分布
        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=bins,
            marker_color=get_concept_color(concept),
            opacity=0.8,
            name=get_concept_label(concept),
        ))
    
    # 添加统计信息
    mean_val = values.mean()
    median_val = values.median()
    
    fig.add_vline(x=mean_val, line_dash='dash', line_color='red',
                  annotation_text=f'Mean: {mean_val:.2f}')
    fig.add_vline(x=median_val, line_dash='dot', line_color='blue',
                  annotation_text=f'Median: {median_val:.2f}')
    
    # 设置标题
    if title is None:
        title = f"Distribution of {get_concept_label(concept)}"
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title=get_concept_label(concept),
        yaxis_title="Count",
        height=height,
        width=width,
        showlegend=True,
    )
    
    return fig


def plot_database_comparison(
    data: Dict[str, Dict[str, pd.DataFrame]],
    concepts: Optional[List[str]] = None,
    metric: str = 'mean',
    title: Optional[str] = None,
    height: int = 600,
    width: int = 1000,
) -> 'go.Figure':
    """绘制跨数据库对比图。
    
    显示不同 ICU 数据库中各 concept 的统计对比。
    
    Args:
        data: 嵌套字典，格式为 {database: {concept: DataFrame}}
        concepts: 要比较的 concept 列表
        metric: 对比指标 ('mean', 'median', 'missing_rate', 'std')
        title: 图表标题
        height: 图表高度
        width: 图表宽度
        
    Returns:
        Plotly Figure 对象
        
    Example:
        >>> miiv_data = load_concepts(miiv_path, concepts=['hr', 'map'])
        >>> eicu_data = load_concepts(eicu_path, concepts=['hr', 'map'])
        >>> combined = {'miiv': miiv_data, 'eicu': eicu_data}
        >>> fig = plot_database_comparison(combined, metric='mean')
        >>> fig.show()
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Install with: pip install plotly")
    
    databases = list(data.keys())
    
    # 收集所有 concepts
    all_concepts = set()
    for db_data in data.values():
        all_concepts.update(db_data.keys())
    
    if concepts is not None:
        all_concepts = [c for c in concepts if c in all_concepts]
    else:
        all_concepts = sorted(all_concepts)
    
    # 计算每个数据库每个 concept 的指标
    results = {db: {} for db in databases}
    
    for db, db_data in data.items():
        for concept in all_concepts:
            if concept not in db_data:
                results[db][concept] = np.nan
                continue
            
            df = db_data[concept]
            
            # 提取值
            if isinstance(df, pd.DataFrame):
                value_cols = [c for c in df.columns if c not in ['stay_id', 'hadm_id', 'time', 'index']]
                if concept in df.columns:
                    values = df[concept].dropna()
                elif value_cols:
                    values = df[value_cols[-1]].dropna()
                else:
                    values = pd.Series()
            else:
                values = df.dropna()
            
            # 计算指标
            if len(values) == 0:
                results[db][concept] = np.nan
            elif metric == 'mean':
                results[db][concept] = values.mean()
            elif metric == 'median':
                results[db][concept] = values.median()
            elif metric == 'std':
                results[db][concept] = values.std()
            elif metric == 'missing_rate':
                total_len = len(df) if isinstance(df, (pd.DataFrame, pd.Series)) else 0
                results[db][concept] = (1 - len(values) / total_len) * 100 if total_len > 0 else 0
    
    fig = go.Figure()
    
    # 数据库颜色
    db_colors = {
        'miiv': '#1f77b4',
        'mimic': '#1f77b4',
        'eicu': '#ff7f0e',
        'aumc': '#2ca02c',
        'hirid': '#d62728',
    }
    
    for db in databases:
        values = [results[db].get(c, np.nan) for c in all_concepts]
        labels = [get_concept_label(c) for c in all_concepts]
        
        fig.add_trace(go.Bar(
            x=labels,
            y=values,
            name=db.upper(),
            marker_color=db_colors.get(db.lower(), '#9467bd'),
        ))
    
    # 设置标题
    metric_labels = {
        'mean': 'Mean Value',
        'median': 'Median Value',
        'std': 'Standard Deviation',
        'missing_rate': 'Missing Rate (%)',
    }
    
    if title is None:
        title = f"Database Comparison - {metric_labels.get(metric, metric)}"
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Concept",
        yaxis_title=metric_labels.get(metric, metric),
        height=height,
        width=width,
        barmode='group',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
        ),
        xaxis_tickangle=-45,
    )
    
    return fig


def plot_cohort_summary(
    data: Union[pd.DataFrame, Dict[str, Any]],
    title: Optional[str] = None,
    height: int = 800,
    width: int = 1200,
) -> 'go.Figure':
    """绘制队列综合摘要图。
    
    显示患者数量、住院时长、年龄分布等基本统计信息。
    
    Args:
        data: 队列数据或摘要统计
        title: 图表标题
        height: 图表高度
        width: 图表宽度
        
    Returns:
        Plotly Figure 对象
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Install with: pip install plotly")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Age Distribution', 
            'Length of Stay (Days)',
            'SOFA Score Distribution', 
            'Mortality by SOFA Quartile'
        ),
        specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
               [{'type': 'histogram'}, {'type': 'bar'}]],
    )
    
    if isinstance(data, pd.DataFrame):
        df = data
        
        # Age distribution
        if 'age' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['age'].dropna(), nbinsx=30, 
                           marker_color='#1f77b4', name='Age'),
                row=1, col=1
            )
        
        # Length of stay
        los_cols = ['los_icu', 'los', 'length_of_stay']
        for col in los_cols:
            if col in df.columns:
                fig.add_trace(
                    go.Histogram(x=df[col].dropna(), nbinsx=30,
                               marker_color='#ff7f0e', name='LOS'),
                    row=1, col=2
                )
                break
        
        # SOFA distribution
        if 'sofa' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['sofa'].dropna(), nbinsx=24,
                           marker_color='#2ca02c', name='SOFA'),
                row=2, col=1
            )
        
        # Mortality by SOFA quartile
        if 'sofa' in df.columns and any(c in df.columns for c in ['death', 'mortality', 'hospital_expire_flag']):
            mort_col = next(c for c in ['death', 'mortality', 'hospital_expire_flag'] if c in df.columns)
            
            df['sofa_quartile'] = pd.qcut(df['sofa'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
            mort_by_q = df.groupby('sofa_quartile')[mort_col].mean() * 100
            
            fig.add_trace(
                go.Bar(x=mort_by_q.index.tolist(), y=mort_by_q.values,
                      marker_color='#d62728', name='Mortality %'),
                row=2, col=2
            )
    
    # 设置标题
    if title is None:
        title = "Cohort Summary"
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=height,
        width=width,
        showlegend=False,
    )
    
    # 更新轴标签
    fig.update_xaxes(title_text="Age (years)", row=1, col=1)
    fig.update_xaxes(title_text="Days", row=1, col=2)
    fig.update_xaxes(title_text="SOFA Score", row=2, col=1)
    fig.update_xaxes(title_text="SOFA Quartile", row=2, col=2)
    
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Mortality (%)", row=2, col=2)
    
    return fig
