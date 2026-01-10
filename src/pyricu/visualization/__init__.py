"""PyRICU Visualization Module.

本地化ICU数据可视化工具包，用于时序数据、评分系统和队列分析的可视化。

所有数据由用户自行下载到本地，所有计算在用户本地机器执行。

Usage:
    >>> from pyricu.visualization import plot_timeline, plot_sofa_breakdown
    >>> from pyricu import load_concepts
    >>> 
    >>> # 加载数据
    >>> data = load_concepts(
    ...     data_path='/path/to/miiv',
    ...     concepts=['hr', 'sbp', 'sofa'],
    ...     patient_ids=[10001, 10002]
    ... )
    >>> 
    >>> # 可视化
    >>> plot_timeline(data['hr'], patient_id=10001, title="Heart Rate")
    >>> plot_sofa_breakdown(data['sofa'], patient_id=10001)
"""

from typing import TYPE_CHECKING

# 检查可选依赖
_HAS_PLOTLY = False
_HAS_MATPLOTLIB = False

try:
    import plotly
    _HAS_PLOTLY = True
except ImportError:
    pass

try:
    import matplotlib
    _HAS_MATPLOTLIB = True
except ImportError:
    pass


def _check_plotly():
    """检查 plotly 是否可用"""
    if not _HAS_PLOTLY:
        raise ImportError(
            "plotly is required for visualization. "
            "Install with: pip install pyricu[viz]"
        )


# 延迟导入，避免在没有依赖时报错
if TYPE_CHECKING:
    from .timeseries import (
        plot_timeline,
        plot_vitals_panel,
        plot_medications_gantt,
        plot_lab_heatmap,
    )
    from .scores import (
        plot_sofa_breakdown,
        plot_sofa_trajectory,
        plot_sepsis_timeline,
    )
    from .cohort import (
        plot_missing_heatmap,
        plot_concept_distribution,
        plot_database_comparison,
    )
    from .patient import PatientDashboard
    from .utils import setup_theme, save_figure


def __getattr__(name):
    """延迟导入可视化函数"""
    _check_plotly()
    
    if name in ('plot_timeline', 'plot_vitals_panel', 'plot_medications_gantt', 'plot_lab_heatmap'):
        from . import timeseries
        return getattr(timeseries, name)
    
    elif name in ('plot_sofa_breakdown', 'plot_sofa_trajectory', 'plot_sepsis_timeline'):
        from . import scores
        return getattr(scores, name)
    
    elif name in ('plot_missing_heatmap', 'plot_concept_distribution', 'plot_database_comparison', 'plot_cohort_summary'):
        from . import cohort
        return getattr(cohort, name)
    
    elif name in ('PatientDashboard', 'render_patient_report'):
        from . import patient
        return getattr(patient, name)
    
    elif name in ('setup_theme', 'save_figure'):
        from . import utils
        return getattr(utils, name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # timeseries
    'plot_timeline',
    'plot_vitals_panel',
    'plot_medications_gantt',
    'plot_lab_heatmap',
    # scores
    'plot_sofa_breakdown',
    'plot_sofa_trajectory',
    'plot_sepsis_timeline',
    # cohort
    'plot_missing_heatmap',
    'plot_concept_distribution',
    'plot_database_comparison',
    'plot_cohort_summary',
    # patient
    'PatientDashboard',
    'render_patient_report',
    # utils
    'setup_theme',
    'save_figure',
]
