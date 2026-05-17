"""多数据库连续特征分布差异可视化

生成类似论文中的多变量密度分布对比图，展示不同ICU数据库间的特征分布差异。
支持 MIIV, eICU, AUMC, HiRID 四个数据库的对比。
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import List, Dict, Tuple
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="ICU Database Feature Distribution Comparison",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 数据库颜色配置 - 与参考图一致
DB_COLORS = {
    'aumc': '#1f77b4',      # 蓝色 - Amsterdam
    'eicu': '#ff7f0e',      # 橙色 - eICU
    'miiv': '#2ca02c',      # 绿色 - MIMIC
    'hirid': '#d62728',     # 红色 - HiRID
}

DB_LABELS = {
    'aumc': 'Amsterdam (AUMC)',
    'eicu': 'eICU-CRD',
    'miiv': 'MIMIC-IV',
    'hirid': 'HiRID',
}

# 特征分组和显示配置
FEATURE_GROUPS = {
    "Vital Signs": {
        'hr': {'name': 'Heart rate', 'unit': '/min', 'range': (30, 200)},
        'sbp': {'name': 'Invasive systolic arterial pressure', 'unit': 'mmHg', 'range': (40, 220)},
        'dbp': {'name': 'Invasive diastolic arterial pressure', 'unit': 'mmHg', 'range': (20, 140)},
        'map': {'name': 'Invasive mean arterial pressure', 'unit': 'mmHg', 'range': (30, 160)},
        'nbps': {'name': 'Non-invasive systolic arterial pressure', 'unit': 'mmHg', 'range': (60, 220)},
        'nbpd': {'name': 'Non-invasive diastolic arterial pressure', 'unit': 'mmHg', 'range': (20, 140)},
        'nbpm': {'name': 'Non-invasive mean arterial pressure', 'unit': 'mmHg', 'range': (30, 160)},
        'o2sat': {'name': 'Oxygen saturation in Arterial blood', 'unit': '%', 'range': (70, 100)},
        'temp': {'name': 'Core body temperature', 'unit': 'Cel', 'range': (34, 42)},
        'resp': {'name': 'Respiratory rate', 'unit': '/min', 'range': (5, 50)},
    },
    "Respiratory": {
        'tv': {'name': 'Expiratory tidal volume', 'unit': 'mL/kg', 'range': (0, 1000)},
        'pplat': {'name': 'Plateau pressure', 'unit': 'cmH2O', 'range': (0, 50)},
        'vent': {'name': 'Ventilator rate', 'unit': '/min', 'range': (0, 40)},
        'tv_set': {'name': 'Tidal volume setting', 'unit': 'mL/kg', 'range': (200, 800)},
        'fio2': {'name': 'Inspired oxygen concentration', 'unit': '%', 'range': (21, 100)},
        'peep': {'name': 'Positive end expiratory pressure setting', 'unit': 'cmH2O', 'range': (0, 25)},
    },
    "Laboratory - Metabolic": {
        'lact': {'name': 'Lactate [Mass/volume] in Arterial blood', 'unit': 'mg/mL', 'range': (0, 15)},
        'glu': {'name': 'Glucose [Moles/volume] in Serum or Plasma', 'unit': 'mg/dL', 'range': (40, 500)},
        'mg': {'name': 'Magnesium [Moles/volume] in Blood', 'unit': 'mmol/L', 'range': (0.5, 2.0)},
        'na': {'name': 'Sodium [Moles/volume] in Blood', 'unit': 'mmol/L', 'range': (120, 160)},
        'crea': {'name': 'Creatinine [Moles/volume] in Blood', 'unit': 'umol/L', 'range': (20, 500)},
        'ca': {'name': 'Calcium [Moles/volume] in Blood', 'unit': 'umol/L', 'range': (1.5, 3.0)},
        'cl': {'name': 'Chloride [Moles/volume] in Blood', 'unit': '%', 'range': (80, 130)},
        'k': {'name': 'Potassium [Moles/volume] in Blood', 'unit': 'mmol/L', 'range': (2.5, 7.0)},
    },
    "Laboratory - Coagulation": {
        'ptt': {'name': 'aPTT in Blood by Coagulation assay', 'unit': 's', 'range': (15, 150)},
        'bili': {'name': 'Bilirubin.total [Moles/volume] in Serum or Plasma', 'unit': 'umol/L', 'range': (0, 200)},
        'alt': {'name': 'Alanine aminotransferase [Enzymatic activity/volume]', 'unit': 'U/L', 'range': (0, 2500)},
        'ast': {'name': 'Aspartate aminotransferase [Enzymatic activity/volume]', 'unit': 'U/L', 'range': (0, 3500)},
        'alp': {'name': 'Alkaline phosphatase [Enzymatic activity/volume]', 'unit': 'U/L', 'range': (0, 600)},
    },
    "Laboratory - Other": {
        'alb': {'name': 'Albumin [Mass/volume] in Serum or Plasma', 'unit': 'g/L', 'range': (10, 50)},
        'phos': {'name': 'Phosphate [Moles/volume] in Blood', 'unit': 'mg/dL', 'range': (0.5, 3.0)},
        'bicar': {'name': 'Bicarbonate [Moles/volume] in Arterial blood', 'unit': 'mmol/L', 'range': (10, 40)},
        'bun': {'name': 'Urea [Moles/volume] in Venous blood', 'unit': 'mg/dL', 'range': (0, 150)},
    },
    "Blood Gas": {
        'ph': {'name': 'pH of Arterial blood', 'unit': 'pH', 'range': (6.8, 7.8)},
        'po2': {'name': 'Oxygen [Partial pressure] in Arterial blood', 'unit': 'pH', 'range': (30, 400)},
        'pco2': {'name': 'Carbon dioxide [Partial pressure] in Arterial blood', 'unit': 'pH', 'range': (15, 100)},
    },
    "Hematology": {
        'hgb': {'name': 'Hemoglobin [Mass/volume] in Blood', 'unit': 'g/dL', 'range': (5, 18)},
        'wbc': {'name': 'Leukocytes [#/volume] in Blood', 'unit': 'U', 'range': (0, 50)},
        'plt': {'name': 'Platelets [#/volume] in Blood', 'unit': 'U', 'range': (0, 600)},
    },
    "Urine & Consciousness": {
        'urine': {'name': 'Hourly urine volume', 'unit': 'mL/h', 'range': (0, 600)},
        'gcs': {'name': 'Glasgow coma score', 'unit': '', 'range': (3, 15)},
        'egcs': {'name': 'Glasgow Coma Score eye opening subscore', 'unit': '', 'range': (1, 4)},
        'vgcs': {'name': 'Glasgow Coma Score verbal response subscore', 'unit': '', 'range': (1, 5)},
        'mgcs': {'name': 'Glasgow Coma Score motor response subscore', 'unit': '', 'range': (1, 6)},
    },
}


def get_flat_features() -> Dict[str, dict]:
    """获取扁平化的特征字典"""
    flat = {}
    for group, features in FEATURE_GROUPS.items():
        for code, config in features.items():
            flat[code] = {**config, 'group': group}
    return flat


@st.cache_data(ttl=3600)
def load_feature_data(
    data_path: str,
    concepts: List[str],
    databases: List[str],
    max_patients: int = 1000,
    sample_per_patient: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    从各数据库加载特征数据
    
    Args:
        data_path: ICU数据根目录
        concepts: 要加载的概念列表
        databases: 要加载的数据库列表
        max_patients: 每个数据库最大患者数
        sample_per_patient: 每个患者采样的记录数
    
    Returns:
        字典 {database: DataFrame with columns [concept, value]}
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from easyicu import load_concepts
    
    result = {}
    
    for db in databases:
        db_path = _get_db_path(data_path, db)
        if not db_path or not os.path.exists(db_path):
            st.warning(f"数据库 {db} 路径不存在: {db_path}")
            continue
            
        try:
            all_data = []
            for concept in concepts:
                try:
                    df = load_concepts(
                        concepts=[concept],
                        database=db,
                        data_path=db_path,
                        max_patients=max_patients,
                        verbose=False,
                    )
                    
                    if df is not None and not df.empty and concept in df.columns:
                        # 采样以减少数据量
                        if len(df) > max_patients * sample_per_patient:
                            df = df.sample(n=max_patients * sample_per_patient, random_state=42)
                        
                        values = df[concept].dropna()
                        if len(values) > 0:
                            all_data.append(pd.DataFrame({
                                'concept': concept,
                                'value': values.values
                            }))
                except Exception as e:
                    st.write(f"  ⚠️ {concept}: {str(e)[:50]}")
                    continue
            
            if all_data:
                result[db] = pd.concat(all_data, ignore_index=True)
                
        except Exception as e:
            st.error(f"加载 {db} 失败: {e}")
    
    return result


def _get_db_path(base_path: str, db: str) -> str:
    """根据数据库名获取完整路径"""
    db_paths = {
        'miiv': 'mimiciv/3.1',
        'eicu': 'eicu/2.0.1',
        'aumc': 'aumc/1.0.2',
        'hirid': 'hirid/1.1.1',
    }
    return os.path.join(base_path, db_paths.get(db, db))


def compute_kde(
    values: np.ndarray, 
    x_range: Tuple[float, float],
    n_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算核密度估计"""
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


def create_distribution_subplot(
    data: Dict[str, pd.DataFrame],
    concept: str,
    config: dict,
    row: int,
    col: int,
    fig: go.Figure,
    show_legend: bool = False,
) -> None:
    """为单个特征创建分布子图"""
    
    x_range = config.get('range', (0, 100))
    
    for db, df in data.items():
        concept_data = df[df['concept'] == concept]['value'].values
        if len(concept_data) < 10:
            continue
            
        x, y = compute_kde(concept_data, x_range)
        if len(x) == 0:
            continue
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=DB_LABELS.get(db, db),
                line=dict(color=DB_COLORS.get(db, '#888888'), width=2),
                fill='tozeroy',
                fillcolor=f"rgba{tuple(list(int(DB_COLORS.get(db, '#888888').lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.3])}",
                showlegend=show_legend,
                legendgroup=db,
            ),
            row=row,
            col=col,
        )
    
    # 设置子图标题和轴标签
    fig.update_xaxes(
        title_text=config.get('unit', ''),
        title_font_size=10,
        row=row,
        col=col,
    )
    fig.update_yaxes(
        title_text='Density' if col == 1 else '',
        title_font_size=10,
        row=row,
        col=col,
    )


def create_full_distribution_figure(
    data: Dict[str, pd.DataFrame],
    selected_features: List[str],
    cols: int = 5,
) -> go.Figure:
    """创建完整的多特征分布图"""
    
    flat_features = get_flat_features()
    n_features = len(selected_features)
    rows = (n_features + cols - 1) // cols
    
    # 获取特征标题
    titles = [flat_features.get(f, {}).get('name', f) for f in selected_features]
    
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )
    
    for idx, feature in enumerate(selected_features):
        row = idx // cols + 1
        col = idx % cols + 1
        config = flat_features.get(feature, {'range': (0, 100), 'unit': ''})
        
        # 只在第一个子图显示图例
        show_legend = (idx == 0)
        
        create_distribution_subplot(data, feature, config, row, col, fig, show_legend)
    
    # 更新整体布局
    fig.update_layout(
        height=280 * rows,
        width=1400,
        title=dict(
            text="Multi-Database Feature Distribution Comparison",
            x=0.5,
            xanchor="center",
            y=0.985,
            yanchor="top",
            font=dict(size=20),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        margin=dict(t=140, b=50, l=50, r=50),
    )
    
    # 更新子图标题字体大小
    for annotation in fig.layout.annotations:
        annotation.font.size = 11
    
    return fig


def create_single_feature_comparison(
    data: Dict[str, pd.DataFrame],
    concept: str,
    config: dict,
) -> go.Figure:
    """创建单特征详细对比图"""
    
    fig = go.Figure()
    x_range = config.get('range', (0, 100))
    
    stats_data = []
    
    for db, df in data.items():
        concept_data = df[df['concept'] == concept]['value'].values
        if len(concept_data) < 10:
            continue
        
        # 计算统计信息
        stats_data.append({
            'Database': DB_LABELS.get(db, db),
            'N': len(concept_data),
            'Mean': np.mean(concept_data),
            'Std': np.std(concept_data),
            'Median': np.median(concept_data),
            'Q25': np.percentile(concept_data, 25),
            'Q75': np.percentile(concept_data, 75),
            'Min': np.min(concept_data),
            'Max': np.max(concept_data),
        })
        
        x, y = compute_kde(concept_data, x_range)
        if len(x) == 0:
            continue
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=DB_LABELS.get(db, db),
                line=dict(color=DB_COLORS.get(db, '#888888'), width=2.5),
                fill='tozeroy',
                fillcolor=f"rgba{tuple(list(int(DB_COLORS.get(db, '#888888').lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.25])}",
            )
        )
    
    fig.update_layout(
        title=f"{config.get('name', concept)}",
        xaxis_title=config.get('unit', ''),
        yaxis_title='Density',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )
    
    return fig, pd.DataFrame(stats_data)


# ==================== Streamlit UI ====================

def main():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .stats-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">📊 ICU Database Feature Distribution</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-database continuous feature distribution comparison across MIMIC-IV, eICU, AUMC, and HiRID</p>', unsafe_allow_html=True)
    
    # 侧边栏配置
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # 数据路径
        data_path = st.text_input(
            "📁 ICU Data Root Path",
            value=os.environ.get('EASYICU_DATA_PATH', '/home/zhuhb/icudb'),
            help="Root directory containing all ICU databases",
        )
        
        st.divider()
        
        # 数据库选择
        st.markdown("### 🏥 Databases")
        selected_dbs = []
        for db, label in DB_LABELS.items():
            color = DB_COLORS[db]
            if st.checkbox(
                f":{color[1:]} [{label}]",
                value=True,
                key=f"db_{db}",
            ):
                selected_dbs.append(db)
        
        st.divider()
        
        # 特征组选择
        st.markdown("### 📋 Feature Groups")
        selected_groups = st.multiselect(
            "Select groups",
            list(FEATURE_GROUPS.keys()),
            default=["Vital Signs", "Laboratory - Metabolic"],
        )
        
        # 获取选中组的所有特征
        available_features = []
        for group in selected_groups:
            available_features.extend(list(FEATURE_GROUPS[group].keys()))
        
        # 特征选择
        if available_features:
            selected_features = st.multiselect(
                "Select specific features",
                available_features,
                default=available_features[:10],
            )
        else:
            selected_features = []
        
        st.divider()
        
        # 采样配置
        st.markdown("### 🎯 Sampling")
        max_patients = st.slider(
            "Max patients per database",
            min_value=100,
            max_value=5000,
            value=500,
            step=100,
        )
        
        sample_per_patient = st.slider(
            "Samples per patient",
            min_value=1,
            max_value=50,
            value=10,
        )
        
        st.divider()
        
        # 可视化配置
        st.markdown("### 📐 Layout")
        n_cols = st.slider("Columns", min_value=3, max_value=6, value=5)
        
        load_button = st.button("🚀 Load Data & Generate", type="primary", use_container_width=True)
    
    # 主区域
    if not selected_dbs:
        st.warning("⚠️ Please select at least one database")
        return
    
    if not selected_features:
        st.warning("⚠️ Please select at least one feature")
        return
    
    # 标签页
    tab1, tab2, tab3 = st.tabs(["📊 Distribution Grid", "🔍 Single Feature Detail", "📈 Summary Statistics"])
    
    with tab1:
        if load_button or 'feature_data' in st.session_state:
            if load_button:
                with st.spinner("Loading data from databases..."):
                    st.session_state.feature_data = load_feature_data(
                        data_path,
                        selected_features,
                        selected_dbs,
                        max_patients,
                        sample_per_patient,
                    )
            
            if 'feature_data' in st.session_state and st.session_state.feature_data:
                data = st.session_state.feature_data
                
                # 显示加载的数据量
                cols = st.columns(len(data))
                for i, (db, df) in enumerate(data.items()):
                    with cols[i]:
                        st.metric(
                            label=DB_LABELS.get(db, db),
                            value=f"{len(df):,}",
                            delta="records",
                        )
                
                # 生成分布图
                fig = create_full_distribution_figure(data, selected_features, n_cols)
                st.plotly_chart(fig, use_container_width=True)
                
                # 下载按钮
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    # 导出为HTML
                    html_bytes = fig.to_html().encode()
                    st.download_button(
                        "📥 Download as HTML",
                        html_bytes,
                        file_name="feature_distribution.html",
                        mime="text/html",
                    )
            else:
                st.info("👆 Click 'Load Data & Generate' to create the distribution plot")
        else:
            st.info("👆 Click 'Load Data & Generate' in the sidebar to start")
    
    with tab2:
        if 'feature_data' in st.session_state and st.session_state.feature_data:
            data = st.session_state.feature_data
            flat_features = get_flat_features()
            
            selected_single = st.selectbox(
                "Select a feature for detailed view",
                selected_features,
                format_func=lambda x: f"{x} - {flat_features.get(x, {}).get('name', x)}",
            )
            
            if selected_single:
                config = flat_features.get(selected_single, {'range': (0, 100), 'unit': ''})
                fig, stats_df = create_single_feature_comparison(data, selected_single, config)
                
                st.plotly_chart(fig, use_container_width=True)
                
                if not stats_df.empty:
                    st.markdown("### 📊 Statistics Summary")
                    st.dataframe(
                        stats_df.style.format({
                            'Mean': '{:.2f}',
                            'Std': '{:.2f}',
                            'Median': '{:.2f}',
                            'Q25': '{:.2f}',
                            'Q75': '{:.2f}',
                            'Min': '{:.2f}',
                            'Max': '{:.2f}',
                        }),
                        use_container_width=True,
                    )
        else:
            st.info("Please load data first from the Distribution Grid tab")
    
    with tab3:
        if 'feature_data' in st.session_state and st.session_state.feature_data:
            data = st.session_state.feature_data
            flat_features = get_flat_features()
            
            st.markdown("### 📈 Cross-Database Summary Statistics")
            
            all_stats = []
            for feature in selected_features:
                config = flat_features.get(feature, {})
                for db, df in data.items():
                    concept_data = df[df['concept'] == feature]['value'].values
                    if len(concept_data) > 0:
                        all_stats.append({
                            'Feature': config.get('name', feature),
                            'Code': feature,
                            'Database': DB_LABELS.get(db, db),
                            'N': len(concept_data),
                            'Mean': np.mean(concept_data),
                            'Std': np.std(concept_data),
                            'Median': np.median(concept_data),
                        })
            
            if all_stats:
                stats_df = pd.DataFrame(all_stats)
                
                # 创建透视表
                pivot_mean = stats_df.pivot(index='Code', columns='Database', values='Mean')
                pivot_n = stats_df.pivot(index='Code', columns='Database', values='N')
                
                st.markdown("#### Mean Values by Database")
                st.dataframe(pivot_mean.style.format('{:.2f}'), use_container_width=True)
                
                st.markdown("#### Sample Sizes by Database")
                st.dataframe(pivot_n.style.format('{:,.0f}'), use_container_width=True)
                
                # 🔧 FIX: 导出时使用 utf-8-sig 编码并替换特殊字符
                export_stats = stats_df.copy()
                for col in export_stats.columns:
                    if export_stats[col].dtype == 'object':
                        export_stats[col] = export_stats[col].astype(str).str.replace('±', '+/-', regex=False)
                        export_stats[col] = export_stats[col].str.replace('≥', '>=', regex=False)
                        export_stats[col] = export_stats[col].str.replace('≤', '<=', regex=False)
                csv = export_stats.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    "📥 Download Full Statistics (CSV)",
                    csv,
                    file_name="feature_statistics.csv",
                    mime="text/csv",
                )
        else:
            st.info("Please load data first from the Distribution Grid tab")


if __name__ == "__main__":
    main()
