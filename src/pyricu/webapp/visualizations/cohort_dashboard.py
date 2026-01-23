"""ICU队列多维度可视化仪表板

提供类似VIEWER的多视角队列展示，包括：
- 人口统计学分布
- 临床路径分析  
- 病例复杂度热力图
- 患者时间线图表
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="ICU Cohort Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 数据库配置
DB_COLORS = {
    'aumc': '#1f77b4',
    'eicu': '#ff7f0e', 
    'miiv': '#2ca02c',
    'hirid': '#d62728',
}

DB_LABELS = {
    'aumc': 'Amsterdam (AUMC)',
    'eicu': 'eICU-CRD',
    'miiv': 'MIMIC-IV',
    'hirid': 'HiRID',
}

# 颜色方案
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'info': '#06b6d4',
}

# 诊断/药物类别颜色
CATEGORY_COLORS = px.colors.qualitative.Set3


def _get_db_path(base_path: str, db: str) -> str:
    """根据数据库名获取完整路径"""
    db_paths = {
        'miiv': 'mimiciv/3.1',
        'eicu': 'eicu/2.0.1', 
        'aumc': 'aumc/1.0.2',
        'hirid': 'hirid/1.1.1',
    }
    return os.path.join(base_path, db_paths.get(db, db))


@st.cache_data(ttl=3600)
def load_cohort_data(data_path: str, database: str, max_patients: int = 1000) -> Dict[str, pd.DataFrame]:
    """
    加载队列基本数据
    
    Returns:
        包含 demographics, diagnoses, medications, vitals, outcomes 的字典
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from pyricu import load_concepts
    from pyricu.base import BaseICULoader
    
    db_path = _get_db_path(data_path, database)
    
    result = {}
    
    try:
        loader = BaseICULoader(database=database, data_path=db_path)
        
        # 1. 加载人口统计学数据
        try:
            if database == 'miiv':
                patients_table = loader.datasource.load_table('patients', verbose=False)
                admissions_table = loader.datasource.load_table('admissions', verbose=False)
                icustays_table = loader.datasource.load_table('icustays', verbose=False)
                
                if patients_table is not None and admissions_table is not None:
                    patients_df = patients_table.data.copy()
                    admissions_df = admissions_table.data.copy()
                    icustays_df = icustays_table.data.copy() if icustays_table else pd.DataFrame()
                    
                    # 合并数据
                    demo = admissions_df.merge(patients_df, on='subject_id', how='left')
                    if not icustays_df.empty:
                        demo = demo.merge(icustays_df[['hadm_id', 'stay_id', 'los']], on='hadm_id', how='left')
                    
                    if max_patients and len(demo) > max_patients:
                        demo = demo.head(max_patients)
                    
                    result['demographics'] = demo
                    
            elif database == 'eicu':
                patient_table = loader.datasource.load_table('patient', verbose=False)
                if patient_table is not None:
                    demo = patient_table.data.copy()
                    if max_patients and len(demo) > max_patients:
                        demo = demo.head(max_patients)
                    result['demographics'] = demo
                    
            elif database == 'aumc':
                admissions_table = loader.datasource.load_table('admissions', verbose=False)
                if admissions_table is not None:
                    demo = admissions_table.data.copy()
                    if max_patients and len(demo) > max_patients:
                        demo = demo.head(max_patients)
                    result['demographics'] = demo
                    
            elif database == 'hirid':
                general_table = loader.datasource.load_table('general', verbose=False)
                if general_table is not None:
                    demo = general_table.data.copy()
                    if max_patients and len(demo) > max_patients:
                        demo = demo.head(max_patients)
                    result['demographics'] = demo
                    
        except Exception as e:
            st.warning(f"加载人口统计学数据失败: {e}")
        
        # 2. 加载诊断数据
        try:
            if database == 'miiv':
                diag_table = loader.datasource.load_table('diagnoses_icd', verbose=False)
                if diag_table is not None:
                    result['diagnoses'] = diag_table.data.head(max_patients * 10)
            elif database == 'eicu':
                diag_table = loader.datasource.load_table('diagnosis', verbose=False)
                if diag_table is not None:
                    result['diagnoses'] = diag_table.data.head(max_patients * 10)
        except Exception as e:
            pass
            
        # 3. 加载生命体征概览
        try:
            vitals = load_concepts(
                concepts=['hr', 'sbp', 'temp', 'resp', 'o2sat'],
                database=database,
                data_path=db_path,
                max_patients=min(max_patients, 500),
                verbose=False,
            )
            if vitals is not None and not vitals.empty:
                result['vitals'] = vitals
        except Exception as e:
            pass
            
        # 4. 加载SOFA评分
        try:
            sofa = load_concepts(
                concepts=['sofa'],
                database=database,
                data_path=db_path,
                max_patients=min(max_patients, 200),
                verbose=False,
            )
            if sofa is not None and not sofa.empty:
                result['outcomes'] = sofa
        except Exception as e:
            pass
            
    except Exception as e:
        st.error(f"加载 {database} 数据失败: {e}")
    
    return result


def create_population_overview(demo: pd.DataFrame, database: str) -> go.Figure:
    """创建人口统计学概览图"""
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Age Distribution', 'Gender Distribution', 'ICU Length of Stay',
                       'Admission Type', 'Mortality', 'Time Trend'),
        specs=[[{'type': 'histogram'}, {'type': 'pie'}, {'type': 'histogram'}],
               [{'type': 'bar'}, {'type': 'pie'}, {'type': 'scatter'}]],
    )
    
    # 1. 年龄分布
    age_col = None
    for col in ['anchor_age', 'age', 'agegroup', 'admissionyeargroup']:
        if col in demo.columns:
            age_col = col
            break
    
    if age_col and age_col in demo.columns:
        age_data = demo[age_col].dropna()
        if len(age_data) > 0:
            # 处理年龄组
            if age_data.dtype == 'object':
                age_counts = age_data.value_counts()
                fig.add_trace(
                    go.Bar(x=age_counts.index, y=age_counts.values, marker_color=COLORS['primary']),
                    row=1, col=1
                )
            else:
                fig.add_trace(
                    go.Histogram(x=age_data, nbinsx=20, marker_color=COLORS['primary']),
                    row=1, col=1
                )
    
    # 2. 性别分布
    gender_col = None
    for col in ['gender', 'sex']:
        if col in demo.columns:
            gender_col = col
            break
    
    if gender_col:
        gender_counts = demo[gender_col].value_counts()
        fig.add_trace(
            go.Pie(
                labels=gender_counts.index,
                values=gender_counts.values,
                marker_colors=[COLORS['primary'], COLORS['secondary']],
                hole=0.4,
            ),
            row=1, col=2
        )
    
    # 3. ICU住院时长
    los_col = None
    for col in ['los', 'unitdischargeoffset', 'lengthofstay']:
        if col in demo.columns:
            los_col = col
            break
    
    if los_col:
        los_data = demo[los_col].dropna()
        if los_col == 'unitdischargeoffset':
            los_data = los_data / (60 * 24)  # 转换为天
        if len(los_data) > 0:
            fig.add_trace(
                go.Histogram(x=los_data[los_data <= los_data.quantile(0.95)], 
                           nbinsx=30, marker_color=COLORS['success']),
                row=1, col=3
            )
    
    # 4. 入院类型
    admit_col = None
    for col in ['admission_type', 'unitadmitsource', 'origin']:
        if col in demo.columns:
            admit_col = col
            break
    
    if admit_col:
        admit_counts = demo[admit_col].value_counts().head(10)
        fig.add_trace(
            go.Bar(
                x=admit_counts.values,
                y=admit_counts.index,
                orientation='h',
                marker_color=COLORS['info'],
            ),
            row=2, col=1
        )
    
    # 5. 死亡率
    death_col = None
    for col in ['hospital_expire_flag', 'unitdischargestatus', 'death', 'discharge']:
        if col in demo.columns:
            death_col = col
            break
    
    if death_col:
        if demo[death_col].dtype in ['int64', 'float64']:
            death_counts = demo[death_col].value_counts()
            labels = ['Survived', 'Expired'] if len(death_counts) == 2 else death_counts.index.tolist()
        else:
            death_counts = demo[death_col].value_counts()
            labels = death_counts.index.tolist()
        
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=death_counts.values,
                marker_colors=[COLORS['success'], COLORS['danger']],
                hole=0.4,
            ),
            row=2, col=2
        )
    
    # 6. 时间趋势
    time_col = None
    for col in ['admittime', 'hospitaladmittime', 'admissiontime', 'admissionyeargroup']:
        if col in demo.columns:
            time_col = col
            break
    
    if time_col:
        time_data = demo[time_col].dropna()
        if time_data.dtype == 'object' and 'year' in time_col.lower():
            time_counts = time_data.value_counts().sort_index()
            fig.add_trace(
                go.Scatter(x=time_counts.index, y=time_counts.values, 
                          mode='lines+markers', marker_color=COLORS['warning']),
                row=2, col=3
            )
        elif pd.api.types.is_datetime64_any_dtype(time_data):
            time_data = pd.to_datetime(time_data)
            monthly = time_data.dt.to_period('M').value_counts().sort_index()
            fig.add_trace(
                go.Scatter(x=monthly.index.astype(str), y=monthly.values,
                          mode='lines+markers', marker_color=COLORS['warning']),
                row=2, col=3
            )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text=f"Population Overview - {DB_LABELS.get(database, database)}",
        title_x=0.5,
    )
    
    return fig


def create_diagnosis_donut(diagnoses: pd.DataFrame, database: str) -> go.Figure:
    """创建诊断分布甜甜圈图"""
    
    # 根据数据库确定诊断列
    diag_col = None
    for col in ['icd_code', 'diagnosisstring', 'icd9code']:
        if col in diagnoses.columns:
            diag_col = col
            break
    
    if diag_col is None:
        return go.Figure()
    
    # 获取诊断类别（取前缀）
    diag_data = diagnoses[diag_col].dropna().astype(str)
    if database == 'eicu':
        # eICU诊断是层级结构，取第一级
        categories = diag_data.str.split('|').str[0]
    else:
        # ICD编码取前3位
        categories = diag_data.str[:3]
    
    cat_counts = categories.value_counts().head(12)
    
    fig = go.Figure(data=[go.Pie(
        labels=cat_counts.index,
        values=cat_counts.values,
        hole=0.5,
        marker_colors=CATEGORY_COLORS,
        textinfo='percent+label',
        textposition='outside',
    )])
    
    fig.update_layout(
        title_text="Diagnosis Distribution",
        title_x=0.5,
        height=450,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
        ),
    )
    
    return fig


def create_complexity_heatmap(demo: pd.DataFrame, database: str) -> go.Figure:
    """创建病例复杂度热力图"""
    
    # 确定分组列
    los_col = None
    for col in ['los', 'unitdischargeoffset', 'lengthofstay']:
        if col in demo.columns:
            los_col = col
            break
    
    age_col = None
    for col in ['anchor_age', 'age']:
        if col in demo.columns:
            age_col = col
            break
    
    if los_col is None or age_col is None:
        # 使用模拟数据
        complexity_matrix = np.random.randint(0, 100, (6, 8))
        age_groups = ['18-30', '30-40', '40-50', '50-60', '60-70', '70+']
        los_groups = ['0-1', '1-2', '2-3', '3-5', '5-7', '7-10', '10-14', '14+']
    else:
        # 创建分组
        los_data = demo[los_col].dropna()
        age_data = demo[age_col].dropna()
        
        if los_col == 'unitdischargeoffset':
            los_data = los_data / (60 * 24)
        
        # 年龄分组
        age_bins = [0, 30, 40, 50, 60, 70, 100]
        age_labels = ['18-30', '30-40', '40-50', '50-60', '60-70', '70+']
        
        # LOS分组
        los_bins = [0, 1, 2, 3, 5, 7, 10, 14, 1000]
        los_labels = ['0-1', '1-2', '2-3', '3-5', '5-7', '7-10', '10-14', '14+']
        
        # 创建交叉表
        temp_df = pd.DataFrame({
            'age_group': pd.cut(age_data, bins=age_bins, labels=age_labels),
            'los_group': pd.cut(los_data, bins=los_bins, labels=los_labels),
        }).dropna()
        
        if len(temp_df) > 0:
            cross_tab = pd.crosstab(temp_df['age_group'], temp_df['los_group'])
            complexity_matrix = cross_tab.values
            age_groups = cross_tab.index.tolist()
            los_groups = cross_tab.columns.tolist()
        else:
            complexity_matrix = np.random.randint(0, 100, (6, 8))
            age_groups = age_labels
            los_groups = los_labels
    
    # 创建热力图
    fig = go.Figure(data=go.Heatmap(
        z=complexity_matrix,
        x=los_groups,
        y=age_groups,
        colorscale='RdYlGn_r',
        text=complexity_matrix,
        texttemplate='%{text}',
        textfont={'size': 12},
        hovertemplate='Age: %{y}<br>LOS: %{x} days<br>Count: %{z}<extra></extra>',
    ))
    
    fig.update_layout(
        title_text="Caseload Complexity (Age vs LOS)",
        title_x=0.5,
        xaxis_title="Length of Stay (days)",
        yaxis_title="Age Group",
        height=400,
    )
    
    return fig


def create_vital_timeline(vitals: pd.DataFrame, patient_id: int, database: str) -> go.Figure:
    """创建单个患者的生命体征时间线"""
    
    # 确定ID列
    id_col = 'stay_id'
    if database == 'eicu':
        id_col = 'patientunitstayid'
    elif database == 'aumc':
        id_col = 'admissionid'
    elif database == 'hirid':
        id_col = 'patientid'
    
    # 确定时间列
    time_col = 'charttime'
    for col in ['charttime', 'observationoffset', 'measuredat', 'datetime']:
        if col in vitals.columns:
            time_col = col
            break
    
    if id_col not in vitals.columns:
        return go.Figure()
    
    # 筛选患者数据
    patient_data = vitals[vitals[id_col] == patient_id].copy()
    
    if patient_data.empty:
        return go.Figure()
    
    # 创建子图
    vital_cols = [c for c in ['hr', 'sbp', 'temp', 'resp', 'o2sat'] if c in patient_data.columns]
    
    if not vital_cols:
        return go.Figure()
    
    fig = make_subplots(
        rows=len(vital_cols), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=vital_cols,
    )
    
    colors = [COLORS['danger'], COLORS['primary'], COLORS['warning'], 
              COLORS['success'], COLORS['info']]
    
    for i, col in enumerate(vital_cols):
        col_data = patient_data[[time_col, col]].dropna()
        if not col_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=col_data[time_col],
                    y=col_data[col],
                    mode='lines+markers',
                    name=col.upper(),
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4),
                ),
                row=i+1, col=1
            )
    
    fig.update_layout(
        height=100 + 120 * len(vital_cols),
        title_text=f"Patient Timeline - ID: {patient_id}",
        title_x=0.5,
        showlegend=False,
    )
    
    return fig


def create_sofa_distribution(outcomes: pd.DataFrame, database: str) -> go.Figure:
    """创建SOFA评分分布图"""
    
    if 'sofa' not in outcomes.columns:
        return go.Figure()
    
    sofa_data = outcomes['sofa'].dropna()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('SOFA Score Distribution', 'SOFA Score Over Time'),
        specs=[[{'type': 'histogram'}, {'type': 'scatter'}]],
    )
    
    # 直方图
    fig.add_trace(
        go.Histogram(x=sofa_data, nbinsx=24, marker_color=COLORS['danger']),
        row=1, col=1
    )
    
    # 时间趋势（取前1000个点）
    time_col = 'charttime'
    for col in ['charttime', 'observationoffset', 'measuredat']:
        if col in outcomes.columns:
            time_col = col
            break
    
    sample = outcomes[[time_col, 'sofa']].dropna().head(1000)
    if not sample.empty:
        fig.add_trace(
            go.Scatter(
                x=sample[time_col],
                y=sample['sofa'],
                mode='markers',
                marker=dict(size=4, color=COLORS['danger'], opacity=0.5),
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        height=350,
        title_text=f"SOFA Score Analysis - {DB_LABELS.get(database, database)}",
        title_x=0.5,
        showlegend=False,
    )
    
    return fig


def create_summary_metrics(data: Dict[str, pd.DataFrame], database: str) -> Dict:
    """计算汇总指标"""
    
    metrics = {
        'total_patients': 0,
        'avg_age': None,
        'male_pct': None,
        'avg_los': None,
        'mortality_rate': None,
        'avg_sofa': None,
    }
    
    if 'demographics' in data:
        demo = data['demographics']
        metrics['total_patients'] = len(demo)
        
        # 年龄
        for col in ['anchor_age', 'age']:
            if col in demo.columns:
                age_data = pd.to_numeric(demo[col], errors='coerce')
                metrics['avg_age'] = age_data.mean()
                break
        
        # 性别
        for col in ['gender', 'sex']:
            if col in demo.columns:
                gender_counts = demo[col].value_counts(normalize=True)
                for g in ['M', 'Male', 'MALE', 'm']:
                    if g in gender_counts.index:
                        metrics['male_pct'] = gender_counts[g] * 100
                        break
                break
        
        # LOS
        for col in ['los', 'unitdischargeoffset', 'lengthofstay']:
            if col in demo.columns:
                los = pd.to_numeric(demo[col], errors='coerce')
                if col == 'unitdischargeoffset':
                    los = los / (60 * 24)
                metrics['avg_los'] = los.mean()
                break
        
        # 死亡率
        for col in ['hospital_expire_flag', 'unitdischargestatus']:
            if col in demo.columns:
                if demo[col].dtype in ['int64', 'float64']:
                    metrics['mortality_rate'] = demo[col].mean() * 100
                break
    
    if 'outcomes' in data and 'sofa' in data['outcomes'].columns:
        metrics['avg_sofa'] = data['outcomes']['sofa'].mean()
    
    return metrics


# ==================== Streamlit UI ====================

def main():
    # 自定义样式
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #667eea;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #666;
            margin-top: 5px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🏥 ICU Cohort Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-dimensional visualization of ICU patient cohorts</p>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        data_path = st.text_input(
            "📁 ICU Data Root Path",
            value=os.environ.get('RICU_DATA_PATH', '/home/zhuhb/icudb'),
        )
        
        st.divider()
        
        database = st.selectbox(
            "🏥 Select Database",
            options=['miiv', 'eicu', 'aumc', 'hirid'],
            format_func=lambda x: DB_LABELS.get(x, x),
        )
        
        max_patients = st.slider(
            "Max Patients",
            min_value=100,
            max_value=5000,
            value=500,
            step=100,
        )
        
        st.divider()
        
        load_button = st.button("🚀 Load Cohort Data", type="primary", use_container_width=True)
    
    # 主区域
    if load_button or 'cohort_data' in st.session_state:
        if load_button:
            with st.spinner(f"Loading {DB_LABELS.get(database, database)} data..."):
                st.session_state.cohort_data = load_cohort_data(data_path, database, max_patients)
                st.session_state.current_db = database
        
        if 'cohort_data' in st.session_state and st.session_state.cohort_data:
            data = st.session_state.cohort_data
            db = st.session_state.get('current_db', database)
            
            # 顶部指标卡片
            metrics = create_summary_metrics(data, db)
            
            cols = st.columns(6)
            metric_items = [
                ("👥 Patients", metrics['total_patients'], None, "{:,}"),
                ("📅 Avg Age", metrics['avg_age'], "years", "{:.1f}"),
                ("👨 Male %", metrics['male_pct'], "%", "{:.1f}"),
                ("🏥 Avg LOS", metrics['avg_los'], "days", "{:.1f}"),
                ("💀 Mortality", metrics['mortality_rate'], "%", "{:.1f}"),
                ("📊 Avg SOFA", metrics['avg_sofa'], "", "{:.1f}"),
            ]
            
            for i, (label, value, unit, fmt) in enumerate(metric_items):
                with cols[i]:
                    if value is not None:
                        display_val = fmt.format(value)
                        if unit:
                            display_val += f" {unit}"
                    else:
                        display_val = "N/A"
                    st.metric(label, display_val)
            
            st.divider()
            
            # 标签页
            tab1, tab2, tab3, tab4 = st.tabs([
                "👥 Population Overview", 
                "🩺 Clinical Analysis",
                "📊 Complexity Heatmap",
                "📈 Patient Timeline"
            ])
            
            with tab1:
                if 'demographics' in data:
                    fig = create_population_overview(data['demographics'], db)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Demographics data not available")
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'diagnoses' in data:
                        fig = create_diagnosis_donut(data['diagnoses'], db)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Diagnosis data not available")
                
                with col2:
                    if 'outcomes' in data:
                        fig = create_sofa_distribution(data['outcomes'], db)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("SOFA data not available")
            
            with tab3:
                if 'demographics' in data:
                    fig = create_complexity_heatmap(data['demographics'], db)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **Interpretation:**
                    - 热力图展示了年龄组与ICU住院时长的交叉分布
                    - 颜色越深（红色）表示该类别的患者数量越多
                    - 可用于识别高复杂度患者群体
                    """)
                else:
                    st.warning("Demographics data not available")
            
            with tab4:
                if 'vitals' in data:
                    vitals = data['vitals']
                    
                    # 确定ID列
                    id_col = 'stay_id'
                    if db == 'eicu':
                        id_col = 'patientunitstayid'
                    elif db == 'aumc':
                        id_col = 'admissionid'
                    elif db == 'hirid':
                        id_col = 'patientid'
                    
                    if id_col in vitals.columns:
                        patient_ids = vitals[id_col].unique()[:50]  # 最多显示50个
                        
                        selected_patient = st.selectbox(
                            "Select Patient ID",
                            options=patient_ids,
                        )
                        
                        if selected_patient:
                            fig = create_vital_timeline(vitals, selected_patient, db)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"ID column {id_col} not found in vitals data")
                else:
                    st.info("Vitals data not available - click Load to fetch data")
        else:
            st.info("👈 Click 'Load Cohort Data' in the sidebar to start")
    else:
        # 初始状态显示说明
        st.markdown("""
        ### 🎯 Dashboard Features
        
        This dashboard provides comprehensive visualization of ICU patient cohorts:
        
        1. **👥 Population Overview** - Demographics distribution including age, gender, LOS, and mortality
        2. **🩺 Clinical Analysis** - Diagnosis patterns and SOFA score distributions  
        3. **📊 Complexity Heatmap** - Age vs LOS matrix showing caseload complexity
        4. **📈 Patient Timeline** - Individual patient vital signs over time
        
        ### 🏥 Supported Databases
        
        - **MIMIC-IV** - Beth Israel Deaconess Medical Center (Boston)
        - **eICU-CRD** - Multi-center US ICU database
        - **Amsterdam UMC** - Dutch academic medical center
        - **HiRID** - Bern University Hospital (Switzerland)
        
        👈 **Select a database and click 'Load Cohort Data' to begin**
        """)


if __name__ == "__main__":
    main()
