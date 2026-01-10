"""患者级可视化。

提供单个患者的综合仪表盘和报告生成功能。
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
    get_concept_label,
    get_concept_color,
    prepare_timeseries_data,
    create_time_axis_label,
    PYRICU_COLORS,
    CONCEPT_LABELS,
)

from .timeseries import plot_vitals_panel, plot_medications_gantt, plot_lab_heatmap
from .scores import plot_sofa_breakdown, plot_sepsis_timeline


class PatientDashboard:
    """患者综合仪表盘类。
    
    提供单个患者的多维度可视化视图，包括：
    - 生命体征面板
    - 用药时间线
    - 实验室检查热图
    - SOFA 评分分解
    - Sepsis-3 事件标注
    
    Attributes:
        patient_id: 患者标识符
        data: 所有 concept 数据的字典
        database: 数据来源数据库名称
        
    Example:
        >>> dashboard = PatientDashboard(patient_id=10001, database='miiv')
        >>> dashboard.load_data(data_dict)
        >>> fig = dashboard.render_full_dashboard()
        >>> fig.show()
    """
    
    def __init__(
        self,
        patient_id: Any,
        database: str = 'miiv',
    ):
        """初始化患者仪表盘。
        
        Args:
            patient_id: 患者标识符
            database: 数据库名称
        """
        self.patient_id = patient_id
        self.database = database
        self.data: Dict[str, pd.DataFrame] = {}
        self._id_col: Optional[str] = None
    
    def load_data(
        self,
        data: Dict[str, pd.DataFrame],
        id_col: Optional[str] = None,
    ) -> 'PatientDashboard':
        """加载患者数据。
        
        Args:
            data: concept 数据字典
            id_col: ID 列名称
            
        Returns:
            self，支持链式调用
        """
        self.data = data
        self._id_col = id_col
        
        # 自动检测 ID 列
        if id_col is None:
            id_candidates = ['stay_id', 'hadm_id', 'icustay_id', 
                           'patientunitstayid', 'admissionid', 'patientid']
            for concept_df in data.values():
                if isinstance(concept_df, pd.DataFrame):
                    for col in id_candidates:
                        if col in concept_df.columns:
                            self._id_col = col
                            break
                if self._id_col:
                    break
        
        return self
    
    def get_patient_data(self, concept: str) -> Optional[pd.DataFrame]:
        """获取指定患者的单个 concept 数据。
        
        Args:
            concept: concept 名称
            
        Returns:
            患者数据 DataFrame，如果不存在返回 None
        """
        if concept not in self.data:
            return None
        
        df = self.data[concept]
        
        if self._id_col and self._id_col in df.columns:
            return df[df[self._id_col] == self.patient_id].copy()
        
        return df
    
    def render_vitals(
        self,
        concepts: Optional[List[str]] = None,
        **kwargs,
    ) -> 'go.Figure':
        """渲染生命体征面板。
        
        Args:
            concepts: 要显示的 concept 列表
            **kwargs: 传递给 plot_vitals_panel 的参数
            
        Returns:
            Plotly Figure 对象
        """
        if concepts is None:
            # 默认生命体征
            concepts = ['hr', 'map', 'sbp', 'dbp', 'resp', 'temp', 'spo2']
        
        # 收集数据
        vitals_data = {}
        for concept in concepts:
            patient_df = self.get_patient_data(concept)
            if patient_df is not None and len(patient_df) > 0:
                vitals_data[concept] = patient_df
        
        if not vitals_data:
            raise ValueError(f"No vitals data available for patient {self.patient_id}")
        
        kwargs.setdefault('title', f'Vitals - Patient {self.patient_id}')
        return plot_vitals_panel(vitals_data, **kwargs)
    
    def render_medications(
        self,
        concepts: Optional[List[str]] = None,
        **kwargs,
    ) -> 'go.Figure':
        """渲染用药时间线。
        
        Args:
            concepts: 要显示的 concept 列表
            **kwargs: 传递给 plot_medications_gantt 的参数
            
        Returns:
            Plotly Figure 对象
        """
        if concepts is None:
            # 默认药物
            concepts = ['norepi_rate', 'epi_rate', 'dopa_rate', 'dobu_rate', 
                       'vaso_rate', 'ins', 'hep']
        
        # 收集数据
        med_data = {}
        for concept in concepts:
            patient_df = self.get_patient_data(concept)
            if patient_df is not None and len(patient_df) > 0:
                med_data[concept] = patient_df
        
        if not med_data:
            raise ValueError(f"No medication data available for patient {self.patient_id}")
        
        kwargs.setdefault('title', f'Medications - Patient {self.patient_id}')
        return plot_medications_gantt(med_data, **kwargs)
    
    def render_labs(
        self,
        concepts: Optional[List[str]] = None,
        **kwargs,
    ) -> 'go.Figure':
        """渲染实验室检查热图。
        
        Args:
            concepts: 要显示的 concept 列表
            **kwargs: 传递给 plot_lab_heatmap 的参数
            
        Returns:
            Plotly Figure 对象
        """
        if concepts is None:
            # 默认实验室检查
            concepts = ['bili', 'crea', 'lac', 'plt', 'wbc', 'hgb', 
                       'pao2', 'paco2', 'ph', 'glucose', 'sodium', 'potassium']
        
        # 收集数据
        lab_data = {}
        for concept in concepts:
            patient_df = self.get_patient_data(concept)
            if patient_df is not None and len(patient_df) > 0:
                lab_data[concept] = patient_df
        
        if not lab_data:
            raise ValueError(f"No lab data available for patient {self.patient_id}")
        
        kwargs.setdefault('title', f'Lab Results - Patient {self.patient_id}')
        return plot_lab_heatmap(lab_data, **kwargs)
    
    def render_sofa(self, **kwargs) -> 'go.Figure':
        """渲染 SOFA 评分分解图。
        
        Args:
            **kwargs: 传递给 plot_sofa_breakdown 的参数
            
        Returns:
            Plotly Figure 对象
        """
        sofa_data = self.get_patient_data('sofa')
        
        if sofa_data is None or len(sofa_data) == 0:
            raise ValueError(f"No SOFA data available for patient {self.patient_id}")
        
        kwargs.setdefault('title', f'SOFA Score - Patient {self.patient_id}')
        return plot_sofa_breakdown(sofa_data, patient_id=self.patient_id, **kwargs)
    
    def render_sepsis(self, **kwargs) -> 'go.Figure':
        """渲染 Sepsis-3 时间线。
        
        Args:
            **kwargs: 传递给 plot_sepsis_timeline 的参数
            
        Returns:
            Plotly Figure 对象
        """
        sofa_data = self.get_patient_data('sofa')
        sep3_data = self.get_patient_data('sep3')
        
        if sofa_data is None or len(sofa_data) == 0:
            raise ValueError(f"No SOFA data available for patient {self.patient_id}")
        
        kwargs.setdefault('title', f'Sepsis-3 Timeline - Patient {self.patient_id}')
        return plot_sepsis_timeline(sofa_data, sep3_data, patient_id=self.patient_id, **kwargs)
    
    def render_full_dashboard(
        self,
        height: int = 1800,
        width: int = 1400,
    ) -> 'go.Figure':
        """渲染完整仪表盘。
        
        包含所有可用的可视化面板。
        
        Args:
            height: 图表高度
            width: 图表宽度
            
        Returns:
            Plotly Figure 对象
        """
        if not HAS_PLOTLY:
            raise ImportError("plotly is required. Install with: pip install plotly")
        
        # 确定有哪些数据可用
        has_vitals = any(self.get_patient_data(c) is not None 
                        for c in ['hr', 'map', 'sbp', 'resp', 'temp'])
        has_meds = any(self.get_patient_data(c) is not None 
                      for c in ['norepi_rate', 'epi_rate', 'dopa_rate', 'vaso_rate'])
        has_labs = any(self.get_patient_data(c) is not None 
                      for c in ['bili', 'crea', 'lac', 'plt'])
        has_sofa = self.get_patient_data('sofa') is not None
        
        # 计算需要多少行
        n_rows = sum([has_vitals, has_meds, has_labs, has_sofa])
        if n_rows == 0:
            raise ValueError(f"No data available for patient {self.patient_id}")
        
        # 创建子图
        row_heights = []
        subplot_titles = []
        
        if has_vitals:
            row_heights.append(0.3)
            subplot_titles.append('Vital Signs')
        if has_meds:
            row_heights.append(0.2)
            subplot_titles.append('Medications')
        if has_labs:
            row_heights.append(0.25)
            subplot_titles.append('Lab Results')
        if has_sofa:
            row_heights.append(0.25)
            subplot_titles.append('SOFA Score')
        
        fig = make_subplots(
            rows=n_rows, cols=1,
            subplot_titles=subplot_titles,
            row_heights=row_heights,
            vertical_spacing=0.08,
        )
        
        current_row = 1
        
        # 添加生命体征
        if has_vitals:
            vitals_concepts = ['hr', 'map', 'sbp', 'resp', 'temp', 'spo2']
            for concept in vitals_concepts:
                patient_df = self.get_patient_data(concept)
                if patient_df is not None and len(patient_df) > 0:
                    time_col = 'time' if 'time' in patient_df.columns else patient_df.index
                    if isinstance(time_col, str):
                        time_data = patient_df[time_col]
                    else:
                        time_data = patient_df.index
                    
                    value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=time_data,
                            y=patient_df[value_col],
                            mode='lines+markers',
                            name=get_concept_label(concept),
                            line=dict(color=get_concept_color(concept)),
                            marker=dict(size=3),
                        ),
                        row=current_row, col=1
                    )
            current_row += 1
        
        # 添加用药 (简化版)
        if has_meds:
            med_concepts = ['norepi_rate', 'epi_rate', 'dopa_rate', 'dobu_rate', 'vaso_rate']
            for concept in med_concepts:
                patient_df = self.get_patient_data(concept)
                if patient_df is not None and len(patient_df) > 0:
                    time_col = 'time' if 'time' in patient_df.columns else patient_df.index
                    if isinstance(time_col, str):
                        time_data = patient_df[time_col]
                    else:
                        time_data = patient_df.index
                    
                    value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=time_data,
                            y=patient_df[value_col],
                            mode='lines',
                            name=get_concept_label(concept),
                            line=dict(color=get_concept_color(concept)),
                            fill='tozeroy',
                        ),
                        row=current_row, col=1
                    )
            current_row += 1
        
        # 添加实验室检查 (简化版)
        if has_labs:
            lab_concepts = ['bili', 'crea', 'lac', 'plt', 'wbc']
            for concept in lab_concepts:
                patient_df = self.get_patient_data(concept)
                if patient_df is not None and len(patient_df) > 0:
                    time_col = 'time' if 'time' in patient_df.columns else patient_df.index
                    if isinstance(time_col, str):
                        time_data = patient_df[time_col]
                    else:
                        time_data = patient_df.index
                    
                    value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=time_data,
                            y=patient_df[value_col],
                            mode='lines+markers',
                            name=get_concept_label(concept),
                            line=dict(color=get_concept_color(concept)),
                            marker=dict(size=4),
                        ),
                        row=current_row, col=1
                    )
            current_row += 1
        
        # 添加 SOFA
        if has_sofa:
            sofa_df = self.get_patient_data('sofa')
            time_col = 'time' if 'time' in sofa_df.columns else sofa_df.index
            if isinstance(time_col, str):
                time_data = sofa_df[time_col]
            else:
                time_data = sofa_df.index
            
            value_col = 'sofa' if 'sofa' in sofa_df.columns else sofa_df.columns[-1]
            
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=sofa_df[value_col],
                    mode='lines+markers',
                    name='SOFA Score',
                    line=dict(color='#d62728', width=2),
                    marker=dict(size=6),
                    fill='tozeroy',
                    fillcolor='rgba(214, 39, 40, 0.1)',
                ),
                row=current_row, col=1
            )
        
        # 设置布局
        fig.update_layout(
            title=dict(
                text=f"Patient Dashboard - ID: {self.patient_id} ({self.database.upper()})",
                x=0.5,
                font=dict(size=18),
            ),
            height=height,
            width=width,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
            ),
            hovermode='x unified',
        )
        
        # 更新 X 轴标签
        for i in range(1, n_rows + 1):
            fig.update_xaxes(title_text=create_time_axis_label('hours'), row=i, col=1)
        
        return fig


def render_patient_report(
    patient_id: Any,
    data: Dict[str, pd.DataFrame],
    database: str = 'miiv',
    output_format: str = 'html',
    output_path: Optional[str] = None,
) -> Union['go.Figure', str]:
    """生成单患者综合报告。
    
    便捷函数，快速生成患者仪表盘。
    
    Args:
        patient_id: 患者标识符
        data: concept 数据字典
        database: 数据库名称
        output_format: 输出格式 ('html', 'png', 'figure')
        output_path: 输出文件路径
        
    Returns:
        如果 output_format='figure'，返回 Figure 对象
        否则返回输出文件路径
    """
    dashboard = PatientDashboard(patient_id=patient_id, database=database)
    dashboard.load_data(data)
    
    fig = dashboard.render_full_dashboard()
    
    if output_format == 'figure':
        return fig
    
    if output_path is None:
        output_path = f'patient_{patient_id}_report.{output_format}'
    
    if output_format == 'html':
        fig.write_html(output_path)
    elif output_format == 'png':
        fig.write_image(output_path)
    elif output_format == 'pdf':
        fig.write_image(output_path)
    
    return output_path
