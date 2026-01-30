"""
队列对比可视化模块 - 用于ICU患者队列的对比分析和可视化

支持的可视化类型:
1. 人口统计学对比
   - 年龄分布对比 (箱线图/直方图)
   - 性别分布对比 (饼图/条形图)
   - 住院时长对比
   
2. 临床指标对比
   - 特征分布对比
   - 生命体征时间序列对比
   - SOFA评分对比
   
3. 结局对比
   - 死亡率对比
   - 住院时长分布

用法示例:
    >>> from pyricu.cohort_visualization import CohortVisualizer
    >>> 
    >>> # 创建可视化器
    >>> viz = CohortVisualizer(database='miiv', data_path='/path/to/data')
    >>> 
    >>> # Sepsis vs 非Sepsis对比
    >>> fig = viz.compare_demographics(
    ...     group1_ids=sepsis_ids, group1_name='Sepsis',
    ...     group2_ids=non_sepsis_ids, group2_name='Non-Sepsis'
    ... )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# 可选依赖
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


@dataclass
class CohortGroup:
    """队列组定义"""
    patient_ids: List[int]
    name: str
    color: Optional[str] = None
    
    def __len__(self) -> int:
        return len(self.patient_ids)


class CohortVisualizer:
    """
    队列对比可视化器
    
    提供多种可视化方法用于对比不同患者队列
    """
    
    # 默认颜色方案
    DEFAULT_COLORS = [
        '#636EFA',  # 蓝色
        '#EF553B',  # 红色
        '#00CC96',  # 绿色
        '#AB63FA',  # 紫色
        '#FFA15A',  # 橙色
        '#19D3F3',  # 青色
        '#FF6692',  # 粉色
        '#B6E880',  # 浅绿
    ]
    
    def __init__(
        self,
        database: str = 'miiv',
        data_path: Optional[Union[str, Path]] = None,
        language: str = 'zh',  # 'zh' 或 'en'
    ):
        self.database = database
        self.data_path = Path(data_path) if data_path else None
        self.language = language
        
        # 缓存
        self._demographics_cache: Optional[pd.DataFrame] = None
        
        if not HAS_PLOTLY:
            logger.warning("Plotly未安装，部分可视化功能不可用。安装: pip install plotly")
    
    def _get_demographics(self) -> pd.DataFrame:
        """获取人口统计学数据（带缓存）"""
        if self._demographics_cache is not None:
            return self._demographics_cache
        
        from .patient_filter import PatientFilter
        pf = PatientFilter(database=self.database, data_path=self.data_path)
        self._demographics_cache = pf._load_demographics()
        return self._demographics_cache
    
    def _tr(self, key: str) -> str:
        """翻译（根据语言设置）"""
        translations = {
            'age': {'zh': '年龄', 'en': 'Age'},
            'gender': {'zh': '性别', 'en': 'Gender'},
            'male': {'zh': '男性', 'en': 'Male'},
            'female': {'zh': '女性', 'en': 'Female'},
            'los': {'zh': '住院时长', 'en': 'Length of Stay'},
            'hours': {'zh': '小时', 'en': 'hours'},
            'patients': {'zh': '患者数', 'en': 'Patients'},
            'count': {'zh': '计数', 'en': 'Count'},
            'percentage': {'zh': '百分比', 'en': 'Percentage'},
            'survived': {'zh': '存活', 'en': 'Survived'},
            'deceased': {'zh': '死亡', 'en': 'Deceased'},
            'mortality': {'zh': '死亡率', 'en': 'Mortality Rate'},
            'mean': {'zh': '均值', 'en': 'Mean'},
            'median': {'zh': '中位数', 'en': 'Median'},
            'distribution': {'zh': '分布', 'en': 'Distribution'},
            'comparison': {'zh': '对比', 'en': 'Comparison'},
            'cohort': {'zh': '队列', 'en': 'Cohort'},
            'group': {'zh': '组别', 'en': 'Group'},
            'value': {'zh': '值', 'en': 'Value'},
            'time': {'zh': '时间', 'en': 'Time'},
            'feature': {'zh': '特征', 'en': 'Feature'},
        }
        return translations.get(key, {}).get(self.language, key)
    
    def _compute_kde(self, data: np.ndarray, n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算核密度估计 (KDE)
        
        Args:
            data: 输入数据
            n_points: 输出点数
        
        Returns:
            (x坐标, 密度值) 元组
        """
        from scipy import stats
        
        data = data[~np.isnan(data)]
        if len(data) < 2:
            return np.array([0]), np.array([0])
        
        # 使用Scott's rule确定带宽
        kde = stats.gaussian_kde(data, bw_method='scott')
        
        # 生成x轴点
        x_min, x_max = data.min(), data.max()
        padding = (x_max - x_min) * 0.1
        x_grid = np.linspace(x_min - padding, x_max + padding, n_points)
        
        # 计算密度
        density = kde(x_grid)
        
        return x_grid, density
    
    def compare_demographics(
        self,
        group1_ids: List[int],
        group2_ids: List[int],
        group1_name: str = 'Group 1',
        group2_name: str = 'Group 2',
        show_age: bool = True,
        show_gender: bool = True,
        show_los: bool = True,
        show_mortality: bool = True,
        los_percentile_cap: float = 95,  # 住院时长截断百分位数
    ) -> 'go.Figure':
        """
        对比两个队列的人口统计学特征（使用密度图）
        
        Args:
            group1_ids: 第一组患者ID
            group2_ids: 第二组患者ID
            group1_name: 第一组名称
            group2_name: 第二组名称
            show_age: 显示年龄对比
            show_gender: 显示性别对比
            show_los: 显示住院时长对比
            show_mortality: 显示死亡率对比
            los_percentile_cap: 住院时长x轴截断的百分位数（默认95%）
        
        Returns:
            Plotly Figure对象
        """
        if not HAS_PLOTLY:
            raise ImportError("需要安装Plotly: pip install plotly")
        
        df = self._get_demographics()
        df1 = df[df['patient_id'].isin(group1_ids)]
        df2 = df[df['patient_id'].isin(group2_ids)]
        
        # 计算需要显示的子图数量
        num_plots = sum([show_age, show_gender, show_los, show_mortality])
        cols = min(2, num_plots)
        rows = (num_plots + cols - 1) // cols
        
        # 创建子图布局
        subplot_titles = []
        if show_age:
            subplot_titles.append(f"{self._tr('age')}{self._tr('distribution')}")
        if show_gender:
            subplot_titles.append(f"{self._tr('gender')}{self._tr('distribution')}")
        if show_los:
            subplot_titles.append(f"{self._tr('los')}{self._tr('distribution')}")
        if show_mortality:
            subplot_titles.append(f"{self._tr('mortality')}{self._tr('comparison')}")
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{"type": "xy"}] * cols for _ in range(rows)],
            horizontal_spacing=0.12,
            vertical_spacing=0.15,
        )
        
        plot_idx = 0
        
        # 1. 年龄分布对比（密度图）
        if show_age and 'age' in df1.columns:
            row, col = plot_idx // cols + 1, plot_idx % cols + 1
            
            # 计算KDE
            age1 = df1['age'].dropna().values
            age2 = df2['age'].dropna().values
            
            x1, y1 = self._compute_kde(age1)
            x2, y2 = self._compute_kde(age2)
            
            # 添加密度曲线
            fig.add_trace(
                go.Scatter(
                    x=x1, y=y1,
                    name=group1_name,
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color=self.DEFAULT_COLORS[0], width=2),
                    fillcolor=f'rgba(99, 110, 250, 0.3)',
                ),
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter(
                    x=x2, y=y2,
                    name=group2_name,
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color=self.DEFAULT_COLORS[1], width=2),
                    fillcolor=f'rgba(239, 85, 59, 0.3)',
                ),
                row=row, col=col
            )
            
            # 添加均值竖线
            mean1, mean2 = age1.mean(), age2.mean()
            fig.add_vline(x=mean1, line_dash="dash", line_color=self.DEFAULT_COLORS[0], 
                         row=row, col=col, annotation_text=f"μ={mean1:.1f}")
            fig.add_vline(x=mean2, line_dash="dash", line_color=self.DEFAULT_COLORS[1],
                         row=row, col=col, annotation_text=f"μ={mean2:.1f}")
            
            fig.update_xaxes(title_text=self._tr('age'), row=row, col=col)
            fig.update_yaxes(title_text="Density", row=row, col=col)
            plot_idx += 1
        
        # 2. 性别分布对比（百分比条形图）
        if show_gender and 'gender' in df1.columns:
            row, col = plot_idx // cols + 1, plot_idx % cols + 1
            
            # 计算百分比
            g1_male_pct = (df1['gender'] == 'M').mean() * 100
            g1_female_pct = (df1['gender'] == 'F').mean() * 100
            g2_male_pct = (df2['gender'] == 'M').mean() * 100
            g2_female_pct = (df2['gender'] == 'F').mean() * 100
            
            fig.add_trace(
                go.Bar(
                    x=[self._tr('male'), self._tr('female')],
                    y=[g1_male_pct, g1_female_pct],
                    name=group1_name,
                    marker_color=self.DEFAULT_COLORS[0],
                    text=[f"{g1_male_pct:.1f}%", f"{g1_female_pct:.1f}%"],
                    textposition='outside',
                ),
                row=row, col=col
            )
            fig.add_trace(
                go.Bar(
                    x=[self._tr('male'), self._tr('female')],
                    y=[g2_male_pct, g2_female_pct],
                    name=group2_name,
                    marker_color=self.DEFAULT_COLORS[1],
                    text=[f"{g2_male_pct:.1f}%", f"{g2_female_pct:.1f}%"],
                    textposition='outside',
                ),
                row=row, col=col
            )
            fig.update_yaxes(title_text=f"{self._tr('percentage')} (%)", row=row, col=col, range=[0, 100])
            plot_idx += 1
        
        # 3. 住院时长分布（密度图，限制x轴范围）
        if show_los and 'los_hours' in df1.columns:
            row, col = plot_idx // cols + 1, plot_idx % cols + 1
            
            los1 = df1['los_hours'].dropna().values
            los2 = df2['los_hours'].dropna().values
            
            # 计算截断值（使用两组合并数据的百分位数）
            all_los = np.concatenate([los1, los2])
            los_cap = np.percentile(all_los, los_percentile_cap)
            
            # 对数据进行截断以计算KDE（但不修改原始数据）
            los1_capped = los1[los1 <= los_cap]
            los2_capped = los2[los2 <= los_cap]
            
            if len(los1_capped) > 1 and len(los2_capped) > 1:
                x1, y1 = self._compute_kde(los1_capped)
                x2, y2 = self._compute_kde(los2_capped)
                
                # 添加密度曲线
                fig.add_trace(
                    go.Scatter(
                        x=x1, y=y1,
                        name=group1_name,
                        mode='lines',
                        fill='tozeroy',
                        line=dict(color=self.DEFAULT_COLORS[0], width=2),
                        fillcolor=f'rgba(99, 110, 250, 0.3)',
                        showlegend=False,
                    ),
                    row=row, col=col
                )
                fig.add_trace(
                    go.Scatter(
                        x=x2, y=y2,
                        name=group2_name,
                        mode='lines',
                        fill='tozeroy',
                        line=dict(color=self.DEFAULT_COLORS[1], width=2),
                        fillcolor=f'rgba(239, 85, 59, 0.3)',
                        showlegend=False,
                    ),
                    row=row, col=col
                )
                
                # 添加中位数竖线
                med1, med2 = np.median(los1), np.median(los2)
                fig.add_vline(x=med1, line_dash="dash", line_color=self.DEFAULT_COLORS[0],
                             row=row, col=col, annotation_text=f"M={med1:.0f}h")
                fig.add_vline(x=med2, line_dash="dash", line_color=self.DEFAULT_COLORS[1],
                             row=row, col=col, annotation_text=f"M={med2:.0f}h")
            
            # 限制x轴范围
            fig.update_xaxes(
                title_text=f"{self._tr('los')} ({self._tr('hours')})", 
                row=row, col=col,
                range=[0, los_cap]
            )
            fig.update_yaxes(title_text="Density", row=row, col=col)
            plot_idx += 1
        
        # 4. 死亡率对比
        if show_mortality and 'survived' in df1.columns:
            row, col = plot_idx // cols + 1, plot_idx % cols + 1
            
            g1_mortality = (1 - df1['survived'].mean()) * 100
            g2_mortality = (1 - df2['survived'].mean()) * 100
            
            fig.add_trace(
                go.Bar(
                    x=[group1_name, group2_name],
                    y=[g1_mortality, g2_mortality],
                    marker_color=[self.DEFAULT_COLORS[0], self.DEFAULT_COLORS[1]],
                    text=[f"{g1_mortality:.1f}%", f"{g2_mortality:.1f}%"],
                    textposition='outside',
                    showlegend=False,
                ),
                row=row, col=col
            )
            fig.update_yaxes(
                title_text=f"{self._tr('mortality')} (%)", 
                row=row, col=col,
                range=[0, max(g1_mortality, g2_mortality) * 1.3]  # 留出空间显示文字
            )
            plot_idx += 1
        
        fig.update_layout(
            title=f"{self._tr('cohort')}{self._tr('comparison')}: {group1_name} vs {group2_name}",
            showlegend=True,
            height=400 * rows,
            barmode='group',  # 分组条形图
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
        )
        
        return fig
    
    def compare_feature_distribution(
        self,
        feature_data: pd.DataFrame,
        feature_name: str,
        group1_ids: List[int],
        group2_ids: List[int],
        group1_name: str = 'Group 1',
        group2_name: str = 'Group 2',
        id_column: str = 'patient_id',
        chart_type: str = 'box',  # 'box', 'histogram', 'violin'
    ) -> 'go.Figure':
        """
        对比两个队列的特定特征分布
        
        Args:
            feature_data: 包含特征数据的DataFrame
            feature_name: 特征名称（列名）
            group1_ids: 第一组患者ID
            group2_ids: 第二组患者ID
            group1_name: 第一组名称
            group2_name: 第二组名称
            id_column: ID列名
            chart_type: 图表类型 ('box', 'histogram', 'violin')
        
        Returns:
            Plotly Figure对象
        """
        if not HAS_PLOTLY:
            raise ImportError("需要安装Plotly: pip install plotly")
        
        # 分组
        df1 = feature_data[feature_data[id_column].isin(group1_ids)]
        df2 = feature_data[feature_data[id_column].isin(group2_ids)]
        
        fig = go.Figure()
        
        if chart_type == 'box':
            fig.add_trace(go.Box(
                y=df1[feature_name], name=group1_name,
                marker_color=self.DEFAULT_COLORS[0]
            ))
            fig.add_trace(go.Box(
                y=df2[feature_name], name=group2_name,
                marker_color=self.DEFAULT_COLORS[1]
            ))
        
        elif chart_type == 'histogram':
            fig.add_trace(go.Histogram(
                x=df1[feature_name], name=group1_name,
                marker_color=self.DEFAULT_COLORS[0], opacity=0.7
            ))
            fig.add_trace(go.Histogram(
                x=df2[feature_name], name=group2_name,
                marker_color=self.DEFAULT_COLORS[1], opacity=0.7
            ))
            fig.update_layout(barmode='overlay')
        
        elif chart_type == 'violin':
            fig.add_trace(go.Violin(
                y=df1[feature_name], name=group1_name,
                marker_color=self.DEFAULT_COLORS[0], box_visible=True
            ))
            fig.add_trace(go.Violin(
                y=df2[feature_name], name=group2_name,
                marker_color=self.DEFAULT_COLORS[1], box_visible=True
            ))
        
        fig.update_layout(
            title=f"{feature_name} {self._tr('distribution')}{self._tr('comparison')}",
            yaxis_title=feature_name,
            showlegend=True,
        )
        
        return fig
    
    def compare_time_series(
        self,
        data: pd.DataFrame,
        value_column: str,
        group1_ids: List[int],
        group2_ids: List[int],
        group1_name: str = 'Group 1',
        group2_name: str = 'Group 2',
        time_column: str = 'time',
        id_column: str = 'patient_id',
        agg_func: str = 'mean',  # 'mean', 'median'
        show_ci: bool = True,    # 显示置信区间
        max_hours: Optional[int] = None,
    ) -> 'go.Figure':
        """
        对比两个队列的时间序列变化
        
        Args:
            data: 包含时间序列数据的DataFrame
            value_column: 值列名
            group1_ids: 第一组患者ID
            group2_ids: 第二组患者ID
            time_column: 时间列名
            id_column: ID列名
            agg_func: 聚合函数
            show_ci: 是否显示置信区间
            max_hours: 最大显示小时数
        
        Returns:
            Plotly Figure对象
        """
        if not HAS_PLOTLY:
            raise ImportError("需要安装Plotly: pip install plotly")
        
        df1 = data[data[id_column].isin(group1_ids)].copy()
        df2 = data[data[id_column].isin(group2_ids)].copy()
        
        if max_hours is not None:
            df1 = df1[df1[time_column] <= max_hours]
            df2 = df2[df2[time_column] <= max_hours]
        
        # 按时间聚合
        def aggregate_group(df, name):
            if agg_func == 'mean':
                agg = df.groupby(time_column)[value_column].agg(['mean', 'std', 'count'])
            else:
                agg = df.groupby(time_column)[value_column].agg(['median', 'std', 'count'])
                agg.columns = ['mean', 'std', 'count']
            
            agg = agg.reset_index()
            # 计算95%置信区间
            agg['ci'] = 1.96 * agg['std'] / np.sqrt(agg['count'])
            return agg
        
        agg1 = aggregate_group(df1, group1_name)
        agg2 = aggregate_group(df2, group2_name)
        
        fig = go.Figure()
        
        # Group 1
        fig.add_trace(go.Scatter(
            x=agg1[time_column], y=agg1['mean'],
            name=group1_name,
            mode='lines',
            line=dict(color=self.DEFAULT_COLORS[0])
        ))
        
        if show_ci:
            fig.add_trace(go.Scatter(
                x=list(agg1[time_column]) + list(agg1[time_column][::-1]),
                y=list(agg1['mean'] + agg1['ci']) + list((agg1['mean'] - agg1['ci'])[::-1]),
                fill='toself',
                fillcolor=f'rgba(99, 110, 250, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{group1_name} 95% CI',
                showlegend=False
            ))
        
        # Group 2
        fig.add_trace(go.Scatter(
            x=agg2[time_column], y=agg2['mean'],
            name=group2_name,
            mode='lines',
            line=dict(color=self.DEFAULT_COLORS[1])
        ))
        
        if show_ci:
            fig.add_trace(go.Scatter(
                x=list(agg2[time_column]) + list(agg2[time_column][::-1]),
                y=list(agg2['mean'] + agg2['ci']) + list((agg2['mean'] - agg2['ci'])[::-1]),
                fill='toself',
                fillcolor=f'rgba(239, 85, 59, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{group2_name} 95% CI',
                showlegend=False
            ))
        
        fig.update_layout(
            title=f"{value_column} {self._tr('time')} Series {self._tr('comparison')}",
            xaxis_title=f"{self._tr('time')} ({self._tr('hours')})",
            yaxis_title=value_column,
            showlegend=True,
        )
        
        return fig
    
    def create_summary_table(
        self,
        group1_ids: List[int],
        group2_ids: List[int],
        group1_name: str = 'Group 1',
        group2_name: str = 'Group 2',
        show_pvalue: bool = True,
    ) -> pd.DataFrame:
        """
        创建TableOne风格的队列对比汇总表（带p值）
        
        Args:
            group1_ids: 第一组患者ID
            group2_ids: 第二组患者ID
            group1_name: 第一组名称
            group2_name: 第二组名称
            show_pvalue: 是否显示p值
        
        Returns:
            对比汇总DataFrame，包含p值列
        """
        from scipy import stats
        
        df = self._get_demographics()
        df1 = df[df['patient_id'].isin(group1_ids)]
        df2 = df[df['patient_id'].isin(group2_ids)]
        
        rows = []
        
        def format_pvalue(p: float) -> str:
            """格式化p值"""
            if p < 0.001:
                return '<0.001'
            elif p < 0.01:
                return f'{p:.3f}'
            else:
                return f'{p:.2f}'
        
        # 患者数
        row = {
            self._tr('feature'): self._tr('patients'),
            group1_name: f"n={len(df1)}",
            group2_name: f"n={len(df2)}",
        }
        if show_pvalue:
            row['P-value'] = '-'
        rows.append(row)
        
        # 年龄 (连续变量，使用Mann-Whitney U检验，因为年龄可能不正态)
        if 'age' in df1.columns:
            age1, age2 = df1['age'].dropna(), df2['age'].dropna()
            _, p_age = stats.mannwhitneyu(age1, age2, alternative='two-sided')
            
            row = {
                self._tr('feature'): f"{self._tr('age')} ({self._tr('mean')}±SD)",
                group1_name: f"{age1.mean():.1f} ± {age1.std():.1f}",
                group2_name: f"{age2.mean():.1f} ± {age2.std():.1f}",
            }
            if show_pvalue:
                row['P-value'] = format_pvalue(p_age)
            rows.append(row)
            
            # 年龄中位数 [IQR]
            row = {
                self._tr('feature'): f"  {self._tr('median')} [IQR]",
                group1_name: f"{age1.median():.1f} [{age1.quantile(0.25):.1f}-{age1.quantile(0.75):.1f}]",
                group2_name: f"{age2.median():.1f} [{age2.quantile(0.25):.1f}-{age2.quantile(0.75):.1f}]",
            }
            if show_pvalue:
                row['P-value'] = ''
            rows.append(row)
        
        # 性别 (分类变量，使用卡方检验)
        if 'gender' in df1.columns:
            g1_male = (df1['gender'] == 'M').sum()
            g1_female = (df1['gender'] == 'F').sum()
            g2_male = (df2['gender'] == 'M').sum()
            g2_female = (df2['gender'] == 'F').sum()
            
            # 卡方检验
            contingency = [[g1_male, g1_female], [g2_male, g2_female]]
            _, p_gender, _, _ = stats.chi2_contingency(contingency)
            
            g1_male_pct = g1_male / len(df1) * 100 if len(df1) > 0 else 0
            g2_male_pct = g2_male / len(df2) * 100 if len(df2) > 0 else 0
            
            row = {
                self._tr('feature'): f"{self._tr('male')}, n (%)",
                group1_name: f"{g1_male} ({g1_male_pct:.1f}%)",
                group2_name: f"{g2_male} ({g2_male_pct:.1f}%)",
            }
            if show_pvalue:
                row['P-value'] = format_pvalue(p_gender)
            rows.append(row)
        
        # 住院时长 (连续变量，使用Mann-Whitney U检验)
        if 'los_hours' in df1.columns:
            los1, los2 = df1['los_hours'].dropna(), df2['los_hours'].dropna()
            _, p_los = stats.mannwhitneyu(los1, los2, alternative='two-sided')
            
            # 转换为天数更直观
            los1_days = los1 / 24
            los2_days = los2 / 24
            
            row = {
                self._tr('feature'): f"{self._tr('los')} ({self._tr('median')}, days) [IQR]",
                group1_name: f"{los1_days.median():.1f} [{los1_days.quantile(0.25):.1f}-{los1_days.quantile(0.75):.1f}]",
                group2_name: f"{los2_days.median():.1f} [{los2_days.quantile(0.25):.1f}-{los2_days.quantile(0.75):.1f}]",
            }
            if show_pvalue:
                row['P-value'] = format_pvalue(p_los)
            rows.append(row)
        
        # 死亡率 (分类变量，使用卡方检验)
        if 'survived' in df1.columns:
            g1_deceased = (df1['survived'] == 0).sum()
            g1_survived = (df1['survived'] == 1).sum()
            g2_deceased = (df2['survived'] == 0).sum()
            g2_survived = (df2['survived'] == 1).sum()
            
            # 卡方检验
            contingency = [[g1_deceased, g1_survived], [g2_deceased, g2_survived]]
            _, p_mort, _, _ = stats.chi2_contingency(contingency)
            
            g1_mort_pct = g1_deceased / len(df1) * 100 if len(df1) > 0 else 0
            g2_mort_pct = g2_deceased / len(df2) * 100 if len(df2) > 0 else 0
            
            row = {
                self._tr('feature'): f"{self._tr('mortality')}, n (%)",
                group1_name: f"{g1_deceased} ({g1_mort_pct:.1f}%)",
                group2_name: f"{g2_deceased} ({g2_mort_pct:.1f}%)",
            }
            if show_pvalue:
                row['P-value'] = format_pvalue(p_mort)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def compare_multi_group(
        self,
        groups: List[CohortGroup],
        feature: str = 'age',
        chart_type: str = 'box',
    ) -> 'go.Figure':
        """
        对比多个队列组
        
        Args:
            groups: CohortGroup列表
            feature: 要对比的特征 ('age', 'los_hours', 'survived')
            chart_type: 图表类型
        
        Returns:
            Plotly Figure对象
        """
        if not HAS_PLOTLY:
            raise ImportError("需要安装Plotly: pip install plotly")
        
        df = self._get_demographics()
        fig = go.Figure()
        
        for i, group in enumerate(groups):
            group_df = df[df['patient_id'].isin(group.patient_ids)]
            color = group.color or self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)]
            
            if chart_type == 'box':
                fig.add_trace(go.Box(
                    y=group_df[feature],
                    name=group.name,
                    marker_color=color
                ))
            elif chart_type == 'violin':
                fig.add_trace(go.Violin(
                    y=group_df[feature],
                    name=group.name,
                    marker_color=color,
                    box_visible=True
                ))
        
        fig.update_layout(
            title=f"{feature} {self._tr('distribution')}{self._tr('comparison')}",
            yaxis_title=feature,
            showlegend=True,
        )
        
        return fig


def create_sepsis_comparison(
    database: str = 'miiv',
    data_path: Optional[Union[str, Path]] = None,
    language: str = 'zh',
) -> Dict[str, Any]:
    """
    便捷函数：创建Sepsis vs 非Sepsis对比可视化
    
    Args:
        database: 数据库类型
        data_path: 数据路径
        language: 语言 ('zh' 或 'en')
    
    Returns:
        包含可视化对象和统计表的字典
    """
    from .patient_filter import filter_patients
    
    # 筛选队列
    sepsis_ids = filter_patients(
        database=database,
        data_path=data_path,
        has_sepsis=True
    )
    non_sepsis_ids = filter_patients(
        database=database,
        data_path=data_path,
        has_sepsis=False
    )
    
    # 创建可视化
    viz = CohortVisualizer(database=database, data_path=data_path, language=language)
    
    fig = viz.compare_demographics(
        group1_ids=sepsis_ids,
        group2_ids=non_sepsis_ids,
        group1_name='Sepsis' if language == 'en' else '脓毒症',
        group2_name='Non-Sepsis' if language == 'en' else '非脓毒症',
    )
    
    summary_table = viz.create_summary_table(
        group1_ids=sepsis_ids,
        group2_ids=non_sepsis_ids,
        group1_name='Sepsis' if language == 'en' else '脓毒症',
        group2_name='Non-Sepsis' if language == 'en' else '非脓毒症',
    )
    
    return {
        'figure': fig,
        'summary_table': summary_table,
        'sepsis_count': len(sepsis_ids),
        'non_sepsis_count': len(non_sepsis_ids),
    }


def create_survival_comparison(
    database: str = 'miiv',
    data_path: Optional[Union[str, Path]] = None,
    language: str = 'zh',
) -> Dict[str, Any]:
    """
    便捷函数：创建存活 vs 死亡对比可视化
    
    Args:
        database: 数据库类型
        data_path: 数据路径
        language: 语言 ('zh' 或 'en')
    
    Returns:
        包含可视化对象和统计表的字典
    """
    from .patient_filter import filter_patients
    
    # 筛选队列
    survived_ids = filter_patients(
        database=database,
        data_path=data_path,
        survived=True
    )
    deceased_ids = filter_patients(
        database=database,
        data_path=data_path,
        survived=False
    )
    
    # 创建可视化
    viz = CohortVisualizer(database=database, data_path=data_path, language=language)
    
    fig = viz.compare_demographics(
        group1_ids=survived_ids,
        group2_ids=deceased_ids,
        group1_name='Survived' if language == 'en' else '存活',
        group2_name='Deceased' if language == 'en' else '死亡',
        show_mortality=False,  # 不显示死亡率（分组本身就是按存活分的）
    )
    
    summary_table = viz.create_summary_table(
        group1_ids=survived_ids,
        group2_ids=deceased_ids,
        group1_name='Survived' if language == 'en' else '存活',
        group2_name='Deceased' if language == 'en' else '死亡',
    )
    
    return {
        'figure': fig,
        'summary_table': summary_table,
        'survived_count': len(survived_ids),
        'deceased_count': len(deceased_ids),
    }


# ============================================================
# 多数据库特征分布对比类
# ============================================================

class MultiDatabaseDistribution:
    """
    多数据库特征分布对比可视化器
    
    生成类似论文中的多变量密度分布对比图，展示不同ICU数据库间的特征分布差异。
    """
    
    # 数据库颜色配置
    DB_COLORS = {
        'aumc': '#1f77b4',      # 蓝色 - Amsterdam
        'eicu': '#ff7f0e',      # 橙色 - eICU
        'miiv': '#2ca02c',      # 绿色 - MIMIC
        'hirid': '#d62728',     # 红色 - HiRID
    }
    
    DB_LABELS = {
        'aumc': 'Amsterdam',
        'eicu': 'eICU',
        'miiv': 'MIMIC-IV',
        'hirid': 'HiRID',
    }
    
    # 特征配置（名称、单位、合理范围）
    FEATURE_CONFIG = {
        # 生命体征
        'hr': {'name': 'Heart Rate', 'unit': '/min', 'range': (30, 200)},
        'sbp': {'name': 'Systolic BP', 'unit': 'mmHg', 'range': (40, 220)},
        'dbp': {'name': 'Diastolic BP', 'unit': 'mmHg', 'range': (20, 140)},
        'map': {'name': 'Mean Arterial Pressure', 'unit': 'mmHg', 'range': (30, 160)},
        'resp': {'name': 'Respiratory Rate', 'unit': '/min', 'range': (5, 50)},
        'temp': {'name': 'Temperature', 'unit': '°C', 'range': (34, 42)},
        'o2sat': {'name': 'SpO2', 'unit': '%', 'range': (70, 100)},
        # 实验室
        'glu': {'name': 'Glucose', 'unit': 'mg/dL', 'range': (40, 500)},
        'na': {'name': 'Sodium', 'unit': 'mmol/L', 'range': (120, 160)},
        'k': {'name': 'Potassium', 'unit': 'mmol/L', 'range': (2.5, 7.0)},
        'crea': {'name': 'Creatinine', 'unit': 'mg/dL', 'range': (0.2, 15)},
        'bili': {'name': 'Bilirubin', 'unit': 'mg/dL', 'range': (0, 30)},
        'lact': {'name': 'Lactate', 'unit': 'mmol/L', 'range': (0, 15)},
        'hgb': {'name': 'Hemoglobin', 'unit': 'g/dL', 'range': (5, 18)},
        'plt': {'name': 'Platelets', 'unit': '×10⁹/L', 'range': (0, 600)},
        'wbc': {'name': 'WBC', 'unit': '×10⁹/L', 'range': (0, 50)},
        # 血气
        'ph': {'name': 'pH', 'unit': '', 'range': (6.8, 7.8)},
        'po2': {'name': 'PaO2', 'unit': 'mmHg', 'range': (30, 400)},
        'pco2': {'name': 'PaCO2', 'unit': 'mmHg', 'range': (15, 100)},
        'fio2': {'name': 'FiO2', 'unit': '%', 'range': (21, 100)},
        # 其他
        'gcs': {'name': 'GCS', 'unit': '', 'range': (3, 15)},
        'urine': {'name': 'Urine Output', 'unit': 'mL/h', 'range': (0, 500)},
    }
    
    def __init__(self, data_root: str = '/home/zhuhb/icudb', language: str = 'en'):
        self.data_root = Path(data_root)
        self.language = language
        self._data_cache: Dict[str, pd.DataFrame] = {}
    
    def _get_db_path(self, db: str) -> Path:
        """获取数据库路径"""
        db_paths = {
            'miiv': 'mimiciv/3.1',
            'eicu': 'eicu/2.0.1',
            'aumc': 'aumc/1.0.2',
            'hirid': 'hirid/1.1.1',
        }
        return self.data_root / db_paths.get(db, db)
    
    def load_feature_data(
        self,
        concepts: List[str],
        databases: List[str],
        max_patients: int = 500,
        sample_size: int = 5000,
    ) -> Dict[str, pd.DataFrame]:
        """
        加载多数据库特征数据
        
        Returns:
            {database: DataFrame with columns [concept, value]}
        """
        from pyricu import load_concepts
        
        result = {}
        
        for db in databases:
            db_path = self._get_db_path(db)
            if not db_path.exists():
                continue
            
            all_data = []
            for concept in concepts:
                try:
                    df = load_concepts(
                        concepts=[concept],
                        database=db,
                        data_path=str(db_path),
                        max_patients=max_patients,
                        verbose=False,
                    )
                    
                    if df is not None and not df.empty and concept in df.columns:
                        values = df[concept].dropna()
                        if len(values) > sample_size:
                            values = values.sample(n=sample_size, random_state=42)
                        
                        if len(values) > 0:
                            all_data.append(pd.DataFrame({
                                'concept': concept,
                                'value': values.values
                            }))
                except Exception:
                    continue
            
            if all_data:
                result[db] = pd.concat(all_data, ignore_index=True)
        
        return result
    
    def _compute_kde(
        self, 
        values: np.ndarray, 
        x_range: Tuple[float, float],
        n_points: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算核密度估计"""
        from scipy import stats
        
        if len(values) < 10:
            return np.array([]), np.array([])
        
        # 移除异常值
        q1, q99 = np.percentile(values, [1, 99])
        values = values[(values >= q1) & (values <= q99)]
        
        if len(values) < 10:
            return np.array([]), np.array([])
        
        try:
            kde = stats.gaussian_kde(values, bw_method='scott')
            x = np.linspace(x_range[0], x_range[1], n_points)
            y = kde(x)
            return x, y
        except Exception:
            return np.array([]), np.array([])
    
    def create_distribution_grid(
        self,
        data: Dict[str, pd.DataFrame],
        concepts: List[str],
        cols: int = 5,
    ) -> 'go.Figure':
        """
        创建多特征分布网格图
        
        Args:
            data: {database: DataFrame} 数据字典
            concepts: 特征列表
            cols: 列数
        
        Returns:
            Plotly Figure
        """
        if not HAS_PLOTLY:
            raise ImportError("需要安装Plotly")
        
        n_features = len(concepts)
        rows = (n_features + cols - 1) // cols
        
        # 获取特征标题
        titles = [self.FEATURE_CONFIG.get(c, {}).get('name', c) for c in concepts]
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=titles,
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
        )
        
        for idx, concept in enumerate(concepts):
            row = idx // cols + 1
            col = idx % cols + 1
            config = self.FEATURE_CONFIG.get(concept, {'range': (0, 100), 'unit': ''})
            x_range = config.get('range', (0, 100))
            
            for db, df in data.items():
                concept_data = df[df['concept'] == concept]['value'].values
                if len(concept_data) < 10:
                    continue
                
                x, y = self._compute_kde(concept_data, x_range)
                if len(x) == 0:
                    continue
                
                # 转换颜色为RGBA
                color = self.DB_COLORS.get(db, '#888888')
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='lines',
                        name=self.DB_LABELS.get(db, db),
                        line=dict(color=color, width=2),
                        fill='tozeroy',
                        fillcolor=f'rgba({r},{g},{b},0.25)',
                        showlegend=(idx == 0),
                        legendgroup=db,
                    ),
                    row=row,
                    col=col,
                )
            
            # 设置轴标签
            fig.update_xaxes(
                title_text=config.get('unit', ''),
                title_font_size=9,
                row=row,
                col=col,
            )
            fig.update_yaxes(
                title_text='Density' if col == 1 else '',
                title_font_size=9,
                row=row,
                col=col,
            )
        
        fig.update_layout(
            height=240 * rows,
            title_text="Multi-Database Feature Distribution Comparison",
            title_x=0.5,
            title_y=0.98,  # 🔧 FIX: 将标题位置上移
            title_font_size=16,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.06,  # 🔧 FIX: 将图例位置上移，与标题保持距离
                xanchor="center",
                x=0.5,
            ),
            margin=dict(t=120, b=30, l=40, r=20),  # 🔧 FIX: 增加顶部margin给标题和图例更多空间
        )
        
        # 更新子图标题字体
        for annotation in fig.layout.annotations:
            annotation.font.size = 10
        
        return fig
    
    def create_single_feature_comparison(
        self,
        data: Dict[str, pd.DataFrame],
        concept: str,
    ) -> Tuple['go.Figure', pd.DataFrame]:
        """创建单特征详细对比"""
        if not HAS_PLOTLY:
            raise ImportError("需要安装Plotly")
        
        config = self.FEATURE_CONFIG.get(concept, {'range': (0, 100), 'unit': '', 'name': concept})
        x_range = config.get('range', (0, 100))
        
        fig = go.Figure()
        stats_data = []
        
        for db, df in data.items():
            concept_data = df[df['concept'] == concept]['value'].values
            if len(concept_data) < 10:
                continue
            
            # 统计信息
            stats_data.append({
                'Database': self.DB_LABELS.get(db, db),
                'N': len(concept_data),
                'Mean': np.mean(concept_data),
                'Std': np.std(concept_data),
                'Median': np.median(concept_data),
                'Q25': np.percentile(concept_data, 25),
                'Q75': np.percentile(concept_data, 75),
            })
            
            x, y = self._compute_kde(concept_data, x_range)
            if len(x) == 0:
                continue
            
            color = self.DB_COLORS.get(db, '#888888')
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    name=self.DB_LABELS.get(db, db),
                    line=dict(color=color, width=2.5),
                    fill='tozeroy',
                    fillcolor=f'rgba({r},{g},{b},0.2)',
                )
            )
        
        fig.update_layout(
            title=config.get('name', concept),
            xaxis_title=config.get('unit', ''),
            yaxis_title='Density',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        
        return fig, pd.DataFrame(stats_data)
