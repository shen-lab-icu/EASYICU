"""PyRICU Streamlit 主应用。

本地 ICU 数据分析和可视化平台。
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import os

# 🚀 性能优化：禁用自动缓存清除，保持表缓存在多次加载间复用
os.environ['PYRICU_AUTO_CLEAR_CACHE'] = 'False'

# ============ 内存管理配置 ============
def get_system_memory_gb() -> float:
    """获取系统总内存（GB）"""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 8.0  # 默认假设 8GB

def get_available_memory_gb() -> float:
    """获取当前可用内存（GB）"""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return 4.0  # 默认假设 4GB 可用

# 系统内存信息
SYSTEM_MEMORY_GB = get_system_memory_gb()
# 默认内存限制：系统内存的 50%，但不超过 16GB，不低于 4GB
DEFAULT_MEMORY_LIMIT_GB = max(4, min(16, SYSTEM_MEMORY_GB * 0.5))

# ============ 低内存模式配置 ============
LOW_MEMORY_MODE = os.environ.get('PYRICU_LOW_MEMORY', '0') == '1'
WORKERS = int(os.environ.get('PYRICU_WORKERS', '0')) or None  # 0 表示自动

if LOW_MEMORY_MODE:
    # 低内存模式下减少缓存和并行度
    os.environ['PYRICU_CHUNK_SIZE'] = '50000'  # 更小的块大小
    os.environ['PYRICU_MAX_CACHE_SIZE'] = '100'  # 减少缓存表数量
    if WORKERS is None:
        WORKERS = 2  # 默认减少到 2 个线程
    DEFAULT_MEMORY_LIMIT_GB = min(DEFAULT_MEMORY_LIMIT_GB, 4)  # 低内存模式限制到 4GB

if WORKERS:
    os.environ['PYRICU_WORKERS'] = str(WORKERS)

# 页面配置
st.set_page_config(
    page_title="PyRICU Data Explorer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 侧边栏宽度调整（加宽以提高可见性）
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        min-width: 380px;
        max-width: 420px;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 380px;
    }
</style>
""", unsafe_allow_html=True)

# 自定义 CSS - 同时兼容深色和浅色主题
st.markdown("""
<style>
    /* 减少页面顶部留白 */
    .block-container {
        padding-top: 0.5rem !important;
        margin-top: 0 !important;
    }
    header[data-testid="stHeader"] {
        height: 0 !important;
        min-height: 0 !important;
        visibility: hidden !important;
    }
    
    /* 顶部 Tabs 标签样式 - 更大更显眼 */
    div[data-baseweb="tab-list"] {
        gap: 8px !important;
        margin-top: 0 !important;
        padding-top: 0 !important;
        background: linear-gradient(180deg, rgba(31,119,180,0.05), transparent) !important;
        padding: 8px !important;
        border-radius: 12px !important;
    }
    div[data-baseweb="tab-list"] button {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        padding: 14px 24px !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    div[data-baseweb="tab-list"] button:hover {
        background: rgba(31,119,180,0.15) !important;
    }
    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
    }
    div[data-baseweb="tab-list"] button p {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }
    
    /* 主题色彩 - 更现代的配色 */
    :root {
        --primary-color: #667eea;
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --success-color: #10b981;
        --success-gradient: linear-gradient(135deg, #10b981 0%, #059669 100%);
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --info-color: #06b6d4;
        --card-bg-light: #ffffff;
        --card-bg-dark: rgba(30, 35, 45, 0.95);
        --text-primary-light: #1e1e1e;
        --text-primary-dark: #e0e0e0;
        --text-secondary-light: #555;
        --text-secondary-dark: #aaa;
    }
    
    /* 主标题 - 现代渐变 */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-top: 0;
        margin-bottom: 0.5rem;
        text-align: center;
        letter-spacing: -0.5px;
    }
    
    /* 副标题 - 自适应主题 */
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 1.5rem;
        text-align: center;
        font-weight: 400;
    }
    @media (prefers-color-scheme: dark) {
        .sub-header { color: #aaa; }
    }
    
    /* 卡片样式 - 自适应主题 + 现代设计 */
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f5f7fa);
        border-radius: 16px;
        padding: 1.4rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(102,126,234,0.1);
        border: 1px solid rgba(102,126,234,0.1);
        transition: all 0.3s ease;
        color: #1e1e1e;
    }
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background: linear-gradient(145deg, rgba(40,45,60,0.95), rgba(30,35,50,0.95));
            border: 1px solid rgba(102,126,234,0.2);
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            color: #e0e0e0;
        }
    }
    [data-testid="stAppViewContainer"][data-theme="dark"] .metric-card {
        background: linear-gradient(145deg, rgba(40,45,60,0.95), rgba(30,35,50,0.95));
        border: 1px solid rgba(102,126,234,0.2);
        color: #e0e0e0;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.2);
        border-color: rgba(102,126,234,0.3);
    }
    
    /* 功能卡片 - 自适应主题 + 现代设计 */
    .feature-card {
        background: linear-gradient(145deg, #ffffff, #f8f9ff);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(102,126,234,0.15);
        margin: 0.5rem 0;
        color: #333;
        transition: all 0.3s ease;
    }
    @media (prefers-color-scheme: dark) {
        .feature-card {
            background: linear-gradient(145deg, rgba(40,45,60,0.95), rgba(30,35,50,0.95));
            border: 1px solid rgba(102,126,234,0.2);
            color: #e0e0e0;
        }
    }
    [data-testid="stAppViewContainer"][data-theme="dark"] .feature-card {
        background: linear-gradient(145deg, rgba(40,45,60,0.95), rgba(30,35,50,0.95));
        border: 1px solid rgba(102,126,234,0.2);
        color: #e0e0e0;
    }
    .feature-card:hover {
        border-color: #667eea;
        box-shadow: 0 8px 25px rgba(102,126,234,0.25);
        transform: translateY(-2px);
    }
    .feature-card h4 {
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    .feature-card ol, .feature-card li {
        color: inherit;
    }
    .feature-card p {
        color: #666;
    }
    @media (prefers-color-scheme: dark) {
        .feature-card p { color: #aaa; }
    }
    
    /* 移除旧的 Tab 样式，已在上方定义 */
    
    /* 成功/警告框 - 自适应主题 */
    .success-box {
        background: rgba(40, 167, 69, 0.15);
        border-left: 4px solid #28a745;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 10px 0;
        color: #155724;
    }
    @media (prefers-color-scheme: dark) {
        .success-box { color: #a3d9a5; }
    }
    .warning-box {
        background: rgba(255, 193, 7, 0.15);
        border-left: 4px solid #ffc107;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 10px 0;
        color: #856404;
    }
    @media (prefers-color-scheme: dark) {
        .warning-box { color: #ffe69c; }
    }
    .info-box {
        background: rgba(23, 162, 184, 0.15);
        border-left: 4px solid #17a2b8;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 10px 0;
        color: #0c5460;
    }
    @media (prefers-color-scheme: dark) {
        .info-box { color: #8dd3e0; }
    }
    
    /* 分隔线 */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #1f77b4, transparent);
        margin: 1.5rem 0;
    }
    
    /* 统计数字 */
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
    }
    @media (prefers-color-scheme: dark) {
        .stat-number { color: #4fc3f7; }
    }
    .stat-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
    }
    @media (prefers-color-scheme: dark) {
        .stat-label { color: #aaa; }
    }
    
    /* 患者信息卡片 - 自适应主题 */
    .patient-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        border: 2px solid #e0e0e0;
        margin-bottom: 1rem;
        color: #333;
    }
    @media (prefers-color-scheme: dark) {
        .patient-card {
            background: rgba(30, 40, 50, 0.9);
            border: 2px solid rgba(255,255,255,0.15);
            color: #e0e0e0;
        }
    }
    .patient-card.critical {
        border-color: #dc3545;
        background: rgba(220, 53, 69, 0.1);
    }
    .patient-card.warning {
        border-color: #ffc107;
        background: rgba(255, 193, 7, 0.1);
    }
    .patient-card.stable {
        border-color: #28a745;
        background: rgba(40, 167, 69, 0.1);
    }
    
    /* 图表容器 */
    .chart-container {
        background: rgba(30, 40, 50, 0.8);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        margin: 0.5rem 0;
    }
    
    /* 侧边栏样式 - 移除背景覆盖 */
    [data-testid="stSidebar"] .stButton button {
        background: linear-gradient(135deg, #1f77b4, #2980b9);
        color: white;
        border: none;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: linear-gradient(135deg, #2980b9, #1f77b4);
    }
    
    /* 进度条 */
    .progress-bar {
        height: 8px;
        background: #e9ecef;
        border-radius: 4px;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        border-radius: 4px;
        transition: width 0.3s;
    }
    
    /* 数据表格优化 */
    .dataframe {
        border-radius: 8px !important;
        overflow: hidden;
    }
    
    /* 加宽侧边栏 */
    [data-testid="stSidebar"] {
        min-width: 450px !important;
        max-width: 550px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        min-width: 450px !important;
        max-width: 550px !important;
    }
    
    /* SOFA2 亮点徽章 */
    .sofa2-badge {
        background: linear-gradient(135deg, #ff6b6b, #ffa500);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 8px;
    }
    
    /* 新功能高亮卡片 - 白底黑字，更清晰 */
    .highlight-card {
        background: #ffffff;
        border: 2px solid #1f77b4;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #333;
    }
    .highlight-card h4 {
        color: #1f77b4;
        margin-bottom: 0.8rem;
    }
    .highlight-card p, .highlight-card li {
        color: #555;
    }
    .highlight-card b {
        color: #1f77b4;
    }
    @media (prefers-color-scheme: dark) {
        .highlight-card {
            background: #1e2a3a;
            color: #e0e0e0;
        }
        .highlight-card p, .highlight-card li {
            color: #bbb;
        }
    }
</style>
""", unsafe_allow_html=True)


# 数据字典定义 - 特征缩写及其含义
CONCEPT_DICTIONARY = {
    # 生命体征
    'hr': ('Heart Rate', '心率', 'bpm'),
    'map': ('Mean Arterial Pressure', '平均动脉压', 'mmHg'),
    'sbp': ('Systolic Blood Pressure', '收缩压', 'mmHg'),
    'dbp': ('Diastolic Blood Pressure', '舒张压', 'mmHg'),
    'temp': ('Temperature', '体温', '°C'),
    'etco2': ('End-Tidal CO2', '呼气末二氧化碳', 'mmHg'),
    'resp': ('Respiratory Rate', '呼吸频率', 'breaths/min'),
    
    # 呼吸系统
    'pafi': ('PaO2/FiO2 Ratio', '氧合指数', 'mmHg'),
    'safi': ('SpO2/FiO2 Ratio', '脉氧/吸氧比', ''),
    'supp_o2': ('Supplemental Oxygen', '辅助吸氧', 'boolean'),
    'vent_ind': ('Mechanical Ventilation Indicator', '机械通气指示', 'boolean'),
    'o2sat': ('Oxygen Saturation (SpO2)', '血氧饱和度', '%'),
    'sao2': ('Arterial Oxygen Saturation', '动脉血氧饱和度', '%'),
    'mech_vent': ('Mechanical Ventilation', '机械通气', 'boolean'),
    'ett_gcs': ('Endotracheal Tube + GCS', '气管插管GCS', ''),
    'fio2': ('Fraction of Inspired Oxygen', '吸入氧浓度', '%'),
    
    # 血气分析
    'be': ('Base Excess', '碱剩余', 'mEq/L'),
    'cai': ('Ionized Calcium', '离子钙', 'mmol/L'),
    'hbco': ('Carboxyhemoglobin', '碳氧血红蛋白', '%'),
    'lact': ('Lactate', '乳酸', 'mmol/L'),
    'methb': ('Methemoglobin', '高铁血红蛋白', '%'),
    'pco2': ('Partial Pressure of CO2', '二氧化碳分压', 'mmHg'),
    'ph': ('Blood pH', '血液pH值', ''),
    'po2': ('Partial Pressure of O2', '氧分压', 'mmHg'),
    'tco2': ('Total CO2', '总二氧化碳', 'mEq/L'),
    
    # 实验室检查
    'alb': ('Albumin', '白蛋白', 'g/dL'),
    'alp': ('Alkaline Phosphatase', '碱性磷酸酶', 'IU/L'),
    'alt': ('Alanine Aminotransferase', '谷丙转氨酶', 'IU/L'),
    'ast': ('Aspartate Aminotransferase', '谷草转氨酶', 'IU/L'),
    'bicar': ('Bicarbonate', '碳酸氢根', 'mEq/L'),
    'bili': ('Total Bilirubin', '总胆红素', 'mg/dL'),
    'bili_dir': ('Direct Bilirubin', '直接胆红素', 'mg/dL'),
    'bun': ('Blood Urea Nitrogen', '血尿素氮', 'mg/dL'),
    'ca': ('Calcium', '钙', 'mg/dL'),
    'ck': ('Creatine Kinase', '肌酸激酶', 'IU/L'),
    'ckmb': ('CK-MB', '肌酸激酶同工酶', 'ng/mL'),
    'cl': ('Chloride', '氯', 'mEq/L'),
    'crea': ('Creatinine', '肌酐', 'mg/dL'),
    'crp': ('C-Reactive Protein', 'C反应蛋白', 'mg/L'),
    'glu': ('Glucose', '血糖', 'mg/dL'),
    'k': ('Potassium', '钾', 'mEq/L'),
    'mg': ('Magnesium', '镁', 'mg/dL'),
    'na': ('Sodium', '钠', 'mEq/L'),
    'phos': ('Phosphorus', '磷', 'mg/dL'),
    'tnt': ('Troponin T', '肌钙蛋白T', 'ng/mL'),
    
    # 血液学
    'bnd': ('Band Neutrophils', '杆状核中性粒细胞', '%'),
    'esr': ('Erythrocyte Sedimentation Rate', '红细胞沉降率', 'mm/hr'),
    'fgn': ('Fibrinogen', '纤维蛋白原', 'mg/dL'),
    'hgb': ('Hemoglobin', '血红蛋白', 'g/dL'),
    'inr_pt': ('INR (Prothrombin Time)', '国际标准化比值', ''),
    'lymph': ('Lymphocytes', '淋巴细胞', '%'),
    'mch': ('Mean Corpuscular Hemoglobin', '平均红细胞血红蛋白含量', 'pg'),
    'mchc': ('Mean Corpuscular Hemoglobin Concentration', '平均红细胞血红蛋白浓度', 'g/dL'),
    'mcv': ('Mean Corpuscular Volume', '平均红细胞体积', 'fL'),
    'neut': ('Neutrophils', '中性粒细胞', '%'),
    'plt': ('Platelets', '血小板', '×10³/μL'),
    'ptt': ('Partial Thromboplastin Time', '部分凝血活酶时间', 'sec'),
    'wbc': ('White Blood Cells', '白细胞', '×10³/μL'),
    
    # 药物治疗
    'abx': ('Antibiotics', '抗生素使用', 'boolean'),
    'adh_rate': ('Vasopressin Rate', '血管加压素速率', 'units/min'),
    'cort': ('Corticosteroids', '糖皮质激素', 'boolean'),
    'dex': ('Dexmedetomidine', '右美托咪定', 'mcg/kg/hr'),
    'dobu_dur': ('Dobutamine Duration', '多巴酚丁胺持续时间', 'hours'),
    'dobu_rate': ('Dobutamine Rate', '多巴酚丁胺速率', 'mcg/kg/min'),
    'dobu60': ('Dobutamine >60min', '多巴酚丁胺>60分钟', 'boolean'),
    'epi_dur': ('Epinephrine Duration', '肾上腺素持续时间', 'hours'),
    'epi_rate': ('Epinephrine Rate', '肾上腺素速率', 'mcg/kg/min'),
    'ins': ('Insulin', '胰岛素', 'units/hr'),
    'norepi_dur': ('Norepinephrine Duration', '去甲肾上腺素持续时间', 'hours'),
    'norepi_equiv': ('Norepinephrine Equivalent', '去甲肾上腺素当量', 'mcg/kg/min'),
    'norepi_rate': ('Norepinephrine Rate', '去甲肾上腺素速率', 'mcg/kg/min'),
    'vaso_ind': ('Vasopressor Indicator', '血管活性药物指示', 'boolean'),
    
    # 尿量
    'urine': ('Urine Output', '尿量', 'mL'),
    'urine24': ('24h Urine Output', '24小时尿量', 'mL/24h'),
    
    # 神经系统
    'avpu': ('AVPU Scale', 'AVPU意识评分', ''),
    'egcs': ('Eye Component of GCS', 'GCS眼睛评分', ''),
    'gcs': ('Glasgow Coma Scale', '格拉斯哥昏迷评分', ''),
    'mgcs': ('Motor Component of GCS', 'GCS运动评分', ''),
    'rass': ('Richmond Agitation-Sedation Scale', 'RASS镇静评分', ''),
    'tgcs': ('Total GCS', 'GCS总分', ''),
    'vgcs': ('Verbal Component of GCS', 'GCS语言评分', ''),
    
    # 人口统计
    'age': ('Age', '年龄', 'years'),
    'bmi': ('Body Mass Index', '体重指数', 'kg/m²'),
    'height': ('Height', '身高', 'cm'),
    'sex': ('Sex', '性别', ''),
    'weight': ('Weight', '体重', 'kg'),
    
    # SOFA-1 评分
    'sofa': ('SOFA Score (Total)', 'SOFA总分', '0-24'),
    'sofa_resp': ('SOFA Respiratory', 'SOFA呼吸评分', '0-4'),
    'sofa_coag': ('SOFA Coagulation', 'SOFA凝血评分', '0-4'),
    'sofa_liver': ('SOFA Liver', 'SOFA肝脏评分', '0-4'),
    'sofa_cardio': ('SOFA Cardiovascular', 'SOFA心血管评分', '0-4'),
    'sofa_cns': ('SOFA Central Nervous System', 'SOFA神经评分', '0-4'),
    'sofa_renal': ('SOFA Renal', 'SOFA肾脏评分', '0-4'),
    'qsofa': ('Quick SOFA', '快速SOFA评分', '0-3'),
    'sirs': ('SIRS Criteria', 'SIRS标准', '0-4'),
    'mews': ('Modified Early Warning Score', '改良早期预警评分', '0-14'),
    'news': ('National Early Warning Score', '国家早期预警评分', '0-20'),
    'death': ('In-hospital Mortality', '院内死亡', 'boolean'),
    'los_icu': ('ICU Length of Stay', 'ICU住院时长', 'days'),
    'los_hosp': ('Hospital Length of Stay', '住院时长', 'days'),
    
    # SOFA-2 评分 (2025年新标准)
    'sofa2': ('SOFA-2 Score (Total)', 'SOFA-2总分 (2025新标准)', '0-24'),
    'sofa2_resp': ('SOFA-2 Respiratory', 'SOFA-2呼吸评分', '0-4'),
    'sofa2_coag': ('SOFA-2 Coagulation', 'SOFA-2凝血评分', '0-4'),
    'sofa2_liver': ('SOFA-2 Liver', 'SOFA-2肝脏评分', '0-4'),
    'sofa2_cardio': ('SOFA-2 Cardiovascular', 'SOFA-2心血管评分', '0-4'),
    'sofa2_cns': ('SOFA-2 Central Nervous System', 'SOFA-2神经评分', '0-4'),
    'sofa2_renal': ('SOFA-2 Renal', 'SOFA-2肾脏评分', '0-4'),
    
    # Sepsis 诊断
    'sep3': ('Sepsis-3 Diagnosis (Default)', 'Sepsis-3诊断 (默认)', 'boolean'),
    'sep3_sofa1': ('Sepsis-3 (SOFA-1 based)', 'Sepsis-3诊断 (基于传统SOFA)', 'boolean'),
    'sep3_sofa2': ('Sepsis-3 (SOFA-2 based)', 'Sepsis-3诊断 (基于SOFA-2, 2025新标准)', 'boolean'),
    'sepsis_sofa2': ('Sepsis (SOFA-2 based)', 'Sepsis诊断 (基于SOFA-2)', 'boolean'),
    'susp_inf': ('Suspected Infection', '疑似感染', 'boolean'),
    'infection_icd': ('ICD Infection Diagnosis', 'ICD感染诊断 (Angus标准)', 'boolean'),
    
    # 呼吸支持 (扩展)
    'spo2': ('Peripheral Oxygen Saturation', '脉搏血氧饱和度', '%'),
    'vent_start': ('Ventilation Start Time', '通气开始时间', 'datetime'),
    'vent_end': ('Ventilation End Time', '通气结束时间', 'datetime'),
    'ecmo': ('ECMO in Use', 'ECMO使用中', 'boolean'),
    'ecmo_indication': ('ECMO Indication', 'ECMO适应症 (呼吸/心血管)', ''),
    
    # 血液学 (扩展)
    'basos': ('Basophils', '嗜碱性粒细胞', '%'),
    'eos': ('Eosinophils', '嗜酸性粒细胞', '%'),
    'hba1c': ('Hemoglobin A1C', '糖化血红蛋白', '%'),
    'hct': ('Hematocrit', '红细胞压积', '%'),
    'pt': ('Prothrombin Time', '凝血酶原时间', 'sec'),
    'rbc': ('Red Blood Cell Count', '红细胞计数', '×10⁶/μL'),
    'rdw': ('Red Cell Distribution Width', '红细胞分布宽度', '%'),
    
    # 生化 (扩展)
    'tri': ('Troponin I', '肌钙蛋白I', 'ng/mL'),
    'bicarb': ('Bicarbonate (alias)', '碳酸氢根 (别名)', 'mEq/L'),
    'potassium': ('Potassium (alias)', '钾 (别名)', 'mEq/L'),
    
    # 药物 (扩展)
    'dopa_rate': ('Dopamine Rate', '多巴胺速率', 'mcg/kg/min'),
    'dopa_dur': ('Dopamine Duration', '多巴胺持续时间', 'hours'),
    'dopa60': ('Dopamine >60min', '多巴胺>60分钟', 'boolean'),
    'norepi60': ('Norepinephrine >60min', '去甲肾上腺素>60分钟', 'boolean'),
    'epi60': ('Epinephrine >60min', '肾上腺素>60分钟', 'boolean'),
    'phn_rate': ('Phenylephrine Rate', '去氧肾上腺素速率', 'mcg/kg/min'),
    
    # 肾脏
    'rrt': ('Renal Replacement Therapy', '肾脏替代治疗', 'boolean'),
    'rrt_criteria': ('RRT Criteria Met', '满足RRT标准', 'boolean'),
    
    # 神经 (扩展)
    'sedated_gcs': ('GCS Before Sedation', '镇静前GCS', ''),
    
    # 心血管 (扩展)
    'mech_circ_support': ('Mechanical Circulatory Support', '机械循环支持 (IABP/LVAD/Impella)', 'boolean'),
    
    # 人口统计 (扩展)
    'adm': ('Admission Type', '入院类型', ''),
    
    # 微生物
    'samp': ('Body Fluid Sampling', '体液采样', 'boolean'),
}

# 特征详细描述（英文和中文）
CONCEPT_DESCRIPTIONS = {
    # SOFA-2
    'sofa2': ('Total SOFA-2 score (2025 new standard), sum of 6 organ systems', 'SOFA-2总分（2025年新标准），6个器官系统评分之和'),
    'sofa2_resp': ('Respiratory component: PaO2/FiO2 or SpO2/FiO2 ratio with ventilation status', '呼吸系统评分：基于氧合指数和通气状态'),
    'sofa2_coag': ('Coagulation component: platelet count', '凝血系统评分：基于血小板计数'),
    'sofa2_liver': ('Liver component: bilirubin level', '肝脏评分：基于胆红素水平'),
    'sofa2_cardio': ('Cardiovascular component: MAP and vasopressor requirements', '心血管评分：基于平均动脉压和血管活性药物'),
    'sofa2_cns': ('Neurological component: GCS score', '神经系统评分：基于格拉斯哥昏迷评分'),
    'sofa2_renal': ('Renal component: creatinine and urine output', '肾脏评分：基于肌酐和尿量'),
    
    # Sepsis
    'sep3_sofa2': ('Sepsis-3 diagnosis based on SOFA-2 (≥2 point increase + suspected infection)', '基于SOFA-2的Sepsis-3诊断（SOFA≥2分上升+疑似感染）'),
    'sep3_sofa1': ('Sepsis-3 diagnosis based on traditional SOFA-1', '基于传统SOFA-1的Sepsis-3诊断'),
    'susp_inf': ('Suspected infection based on antibiotic + culture criteria', '基于抗生素+培养标准的疑似感染'),
    'infection_icd': ('Infection diagnosis based on Angus ICD criteria', '基于Angus ICD标准的感染诊断'),
    
    # Vitals
    'hr': ('Heart rate in beats per minute', '每分钟心跳次数'),
    'map': ('Mean arterial pressure = (SBP + 2×DBP) / 3', '平均动脉压 = (收缩压 + 2×舒张压) / 3'),
    'sbp': ('Systolic blood pressure (peak pressure during heartbeat)', '收缩压（心脏收缩时的最高压力）'),
    'dbp': ('Diastolic blood pressure (pressure between heartbeats)', '舒张压（心脏舒张时的最低压力）'),
    'temp': ('Body temperature in Celsius', '体温（摄氏度）'),
    'resp': ('Respiratory rate (breaths per minute)', '呼吸频率（每分钟呼吸次数）'),
    
    # Respiratory
    'pafi': ('PaO2/FiO2 ratio - key oxygenation index', '氧合指数，反映肺部气体交换功能'),
    'safi': ('SpO2/FiO2 ratio - non-invasive oxygenation estimate', '脉氧/吸氧比，非侵入性氧合评估'),
    'fio2': ('Fraction of inspired oxygen (21-100%)', '吸入氧浓度（21-100%）'),
    'vent_ind': ('Indicates if patient is on mechanical ventilation', '患者是否接受机械通气'),
    
    # Blood gas
    'lact': ('Lactate - marker of tissue hypoperfusion', '乳酸 - 组织低灌注标志物'),
    'ph': ('Blood acidity/alkalinity (normal 7.35-7.45)', '血液酸碱度（正常7.35-7.45）'),
    'pco2': ('Partial pressure of CO2 in blood', '血液中二氧化碳分压'),
    'po2': ('Partial pressure of O2 in blood', '血液中氧分压'),
    
    # Labs
    'crea': ('Creatinine - kidney function marker', '肌酐 - 肾功能标志物'),
    'bili': ('Total bilirubin - liver function marker', '总胆红素 - 肝功能标志物'),
    'plt': ('Platelet count - coagulation marker', '血小板计数 - 凝血功能标志物'),
    'wbc': ('White blood cell count - infection/inflammation marker', '白细胞计数 - 感染/炎症标志物'),
    
    # Vasopressors
    'norepi_rate': ('Norepinephrine infusion rate (weight-adjusted)', '去甲肾上腺素输注速率（体重校正）'),
    'norepi_equiv': ('Norepinephrine equivalent dose (standardized vasopressor dose)', '去甲肾上腺素当量（标准化血管活性药物剂量）'),
    'vaso_ind': ('Indicates any vasopressor use', '是否使用任何血管活性药物'),
    
    # Neurological
    'gcs': ('Glasgow Coma Scale total score (3-15)', '格拉斯哥昏迷评分总分（3-15分）'),
    
    # Outcomes
    'death': ('In-hospital mortality (0=survived, 1=died)', '院内死亡（0=存活，1=死亡）'),
    'los_icu': ('ICU length of stay in days', 'ICU住院时长（天）'),
    'los_hosp': ('Hospital length of stay in days', '总住院时长（天）'),
}

# 全局特征分组定义 - 供侧边栏和数据字典共用
# 使用英文key，并提供双语显示名称
CONCEPT_GROUPS_INTERNAL = {
    'sofa2_score': ['sofa2', 'sofa2_resp', 'sofa2_coag', 'sofa2_liver', 'sofa2_cardio', 'sofa2_cns', 'sofa2_renal'],
    'sofa1_score': ['sofa', 'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal'],
    'sepsis3_sofa2': ['sep3_sofa2', 'susp_inf', 'infection_icd', 'samp'],
    'sepsis3_sofa1': ['sep3_sofa1', 'susp_inf', 'infection_icd', 'samp'],
    'vitals': ['hr', 'map', 'sbp', 'dbp', 'temp', 'etco2', 'spo2', 'resp'],
    'respiratory': ['pafi', 'safi', 'fio2', 'supp_o2', 'vent_ind', 'vent_start', 'vent_end', 'o2sat', 'sao2', 'mech_vent', 'ett_gcs', 'ecmo', 'ecmo_indication'],
    'blood_gas': ['be', 'cai', 'hbco', 'lact', 'methb', 'pco2', 'ph', 'po2', 'tco2'],
    'chemistry': ['alb', 'alp', 'alt', 'ast', 'bicar', 'bili', 'bili_dir', 'bun', 'ca', 'ck', 'ckmb', 'cl', 'crea', 'crp', 'glu', 'k', 'mg', 'na', 'phos', 'tnt', 'tri'],
    'hematology': ['bnd', 'basos', 'eos', 'esr', 'fgn', 'hba1c', 'hct', 'hgb', 'inr_pt', 'lymph', 'mch', 'mchc', 'mcv', 'neut', 'plt', 'pt', 'ptt', 'rbc', 'rdw', 'wbc'],
    'vasopressors': ['norepi_rate', 'norepi_dur', 'norepi_equiv', 'norepi60', 'epi_rate', 'epi_dur', 'epi60', 'dopa_rate', 'dopa_dur', 'dopa60', 'dobu_rate', 'dobu_dur', 'dobu60', 'adh_rate', 'phn_rate', 'vaso_ind'],
    'medications': ['abx', 'cort', 'dex', 'ins'],
    'renal': ['urine', 'urine24', 'rrt', 'rrt_criteria'],
    'neurological': ['avpu', 'egcs', 'gcs', 'mgcs', 'rass', 'tgcs', 'vgcs', 'sedated_gcs'],
    'circulatory': ['mech_circ_support'],
    'demographics': ['age', 'bmi', 'height', 'sex', 'weight', 'adm'],
    'other_scores': ['qsofa', 'sirs', 'mews', 'news'],
    'outcome': ['death', 'los_icu', 'los_hosp'],
}

# 双语显示名称映射（优化：更清晰的命名区分评分vs诊断）
CONCEPT_GROUP_NAMES = {
    'sofa2_score': ('⭐ SOFA-2 Scores (2025 New - 7 items)', '⭐ SOFA-2 评分 (2025新标准 - 7项)'),
    'sofa1_score': ('📊 SOFA-1 Scores (Traditional - 7 items)', '📊 SOFA-1 评分 (传统 - 7项)'),
    'sepsis3_sofa2': ('🦠 Sepsis-3 Diagnosis (SOFA-2)', '🦠 Sepsis-3 诊断 (基于SOFA-2)'),
    'sepsis3_sofa1': ('🦠 Sepsis-3 Diagnosis (SOFA-1)', '🦠 Sepsis-3 诊断 (基于SOFA-1)'),
    'vitals': ('❤️ Vital Signs', '❤️ 生命体征'),
    'respiratory': ('🫁 Respiratory Support', '🫁 呼吸支持'),
    'blood_gas': ('🩸 Blood Gas Analysis', '🩸 血气分析'),
    'chemistry': ('🧪 Lab - Chemistry', '🧪 实验室-生化'),
    'hematology': ('🔬 Lab - Hematology', '🔬 实验室-血液学'),
    'vasopressors': ('💉 Vasopressors', '💉 血管活性药物'),
    'medications': ('💊 Other Medications', '💊 其他药物'),
    'renal': ('🚰 Renal & Urine Output', '🚰 肾脏与尿量'),
    'neurological': ('🧠 Neurological', '🧠 神经系统'),
    'circulatory': ('🫀 Circulatory Support', '🫀 循环支持'),
    'demographics': ('👤 Demographics', '👤 人口统计'),
    'other_scores': ('📈 Other Scores', '📈 其他评分'),
    'outcome': ('🎯 Outcome', '🎯 结局'),
}

def get_concept_groups():
    """根据当前语言返回带正确显示名称的特征分组。"""
    lang = st.session_state.get('language', 'en')
    result = {}
    for key, concepts in CONCEPT_GROUPS_INTERNAL.items():
        en_name, zh_name = CONCEPT_GROUP_NAMES.get(key, (key, key))
        display_name = en_name if lang == 'en' else zh_name
        result[display_name] = concepts
    return result

# 保持向后兼容的CONCEPT_GROUPS（默认中文）
CONCEPT_GROUPS = {
    "⭐ SOFA-2 评分 (2025新标准)": ['sofa2', 'sofa2_resp', 'sofa2_coag', 'sofa2_liver', 'sofa2_cardio', 'sofa2_cns', 'sofa2_renal'],
    "⭐ Sepsis-3 诊断 (基于SOFA-2)": ['sep3_sofa2', 'susp_inf', 'infection_icd', 'samp'],
    "Sepsis-3 诊断 (基于SOFA-1)": ['sep3_sofa1', 'susp_inf', 'infection_icd', 'samp'],
    "生命体征 (vitals)": ['hr', 'map', 'sbp', 'dbp', 'temp', 'etco2', 'spo2', 'resp'],
    "呼吸支持 (respiratory)": ['pafi', 'safi', 'fio2', 'supp_o2', 'vent_ind', 'vent_start', 'vent_end', 'o2sat', 'sao2', 'mech_vent', 'ett_gcs', 'ecmo', 'ecmo_indication'],
    "血气分析 (blood gas)": ['be', 'cai', 'hbco', 'lact', 'methb', 'pco2', 'ph', 'po2', 'tco2'],
    "实验室-生化 (chemistry)": ['alb', 'alp', 'alt', 'ast', 'bicar', 'bili', 'bili_dir', 'bun', 'ca', 'ck', 'ckmb', 'cl', 'crea', 'crp', 'glu', 'k', 'mg', 'na', 'phos', 'tnt', 'tri'],
    "实验室-血液学 (hematology)": ['bnd', 'basos', 'eos', 'esr', 'fgn', 'hba1c', 'hct', 'hgb', 'inr_pt', 'lymph', 'mch', 'mchc', 'mcv', 'neut', 'plt', 'pt', 'ptt', 'rbc', 'rdw', 'wbc'],
    "血管活性药物 (vasopressors)": ['norepi_rate', 'norepi_dur', 'norepi_equiv', 'norepi60', 'epi_rate', 'epi_dur', 'epi60', 'dopa_rate', 'dopa_dur', 'dopa60', 'dobu_rate', 'dobu_dur', 'dobu60', 'adh_rate', 'phn_rate', 'vaso_ind'],
    "其他药物 (medications)": ['abx', 'cort', 'dex', 'ins'],
    "肾脏与尿量 (renal)": ['urine', 'urine24', 'rrt', 'rrt_criteria'],
    "神经系统 (neurological)": ['avpu', 'egcs', 'gcs', 'mgcs', 'rass', 'tgcs', 'vgcs', 'sedated_gcs'],
    "循环支持 (circulatory)": ['mech_circ_support'],
    "人口统计 (demographics)": ['age', 'bmi', 'height', 'sex', 'weight', 'adm'],
    "SOFA-1 评分 (传统)": ['sofa', 'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal'],
    "其他评分 (scores)": ['qsofa', 'sirs', 'mews', 'news'],
    "结局 (outcome)": ['death', 'los_icu', 'los_hosp'],
}


def render_data_dictionary():
    """Render data dictionary (aligned with sidebar groups)."""
    lang = st.session_state.get('language', 'en')
    
    # 双语标题
    title = "### 📖 Data Dictionary" if lang == 'en' else "### 📖 数据字典"
    st.markdown(title)
    
    caption = "Feature abbreviations, English names, Chinese meanings, and units (aligned with module categories)" if lang == 'en' else "每个特征的缩写、英文名称、中文含义及单位（与左侧模块分类一致）"
    st.caption(caption)
    
    # 获取双语分组
    concept_groups = get_concept_groups()
    
    # 使用 tabs 或 expanders 来展示
    all_label = "All" if lang == 'en' else "全部"
    select_label = "Select Category" if lang == 'en' else "选择类别查看"
    
    selected_category = st.selectbox(
        select_label,
        options=[all_label] + list(concept_groups.keys()),
        index=0,
        key="dict_category_select"
    )
    
    if selected_category == all_label:
        # 显示所有类别
        for cat_name, concepts in concept_groups.items():
            feat_label = "features" if lang == 'en' else "个特征"
            with st.expander(f"📁 {cat_name} ({len(concepts)} {feat_label})", expanded=False):
                _render_category_table(concepts, lang)
    else:
        # 只显示选中的类别
        st.markdown(f"#### {selected_category}")
        _render_category_table(concept_groups[selected_category], lang)


def _render_category_table(concepts, lang='en'):
    """Render feature table for a single category with detailed descriptions."""
    rows = []
    for concept in concepts:
        if concept in CONCEPT_DICTIONARY:
            eng_name, chn_name, unit = CONCEPT_DICTIONARY[concept]
            # 获取详细描述
            if concept in CONCEPT_DESCRIPTIONS:
                eng_desc, chn_desc = CONCEPT_DESCRIPTIONS[concept]
            else:
                eng_desc, chn_desc = '', ''
            
            if lang == 'en':
                rows.append({
                    'Abbr': concept,
                    'Full Name': eng_name,
                    'Description': eng_desc if eng_desc else chn_name,
                    'Unit': unit if unit else '-'
                })
            else:
                rows.append({
                    '缩写': concept,
                    '全名': eng_name,
                    '详细说明': chn_desc if chn_desc else chn_name,
                    '单位': unit if unit else '-'
                })
    
    if rows:
        df = pd.DataFrame(rows)
        if lang == 'en':
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Abbr': st.column_config.TextColumn('Abbr', width='small'),
                    'Full Name': st.column_config.TextColumn('Full Name', width='medium'),
                    'Description': st.column_config.TextColumn('Description', width='large'),
                    'Unit': st.column_config.TextColumn('Unit', width='small'),
                }
            )
        else:
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    '缩写': st.column_config.TextColumn('缩写', width='small'),
                    '全名': st.column_config.TextColumn('全名', width='medium'),
                    '详细说明': st.column_config.TextColumn('详细说明', width='large'),
                    '单位': st.column_config.TextColumn('单位', width='small'),
                }
            )


def check_data_status(data_path: str, database: str) -> dict:
    """检查数据目录的状态，返回文件统计信息。"""
    from pathlib import Path
    
    path = Path(data_path)
    result = {
        'ready': False,
        'parquet_count': 0,
        'csv_count': 0,
        'csv_files': [],
        'parquet_files': [],
        'missing_tables': [],
    }
    
    # 统计 parquet 文件（包括分片目录）
    parquet_files = list(path.glob('*.parquet'))
    # 检查分片目录（如 chartevents/1.parquet）
    for subdir in path.iterdir():
        if subdir.is_dir():
            shard_files = list(subdir.glob('[0-9]*.parquet'))
            if shard_files:
                result['parquet_count'] += 1
                result['parquet_files'].append(subdir.name)
    
    result['parquet_count'] += len(parquet_files)
    result['parquet_files'].extend([f.stem for f in parquet_files])
    
    # 统计 CSV 文件
    csv_files = list(path.glob('*.csv')) + list(path.glob('*.csv.gz'))
    result['csv_count'] = len(csv_files)
    result['csv_files'] = [f.name for f in csv_files]
    
    # 检查是否有足够的 parquet 文件（至少需要一些核心表）
    core_tables = {
        'miiv': ['icustays', 'patients', 'admissions'],
        'eicu': ['patient', 'apachepatientresult'],
        'aumc': ['admissions', 'drugitems'],
        'hirid': ['general_table', 'observations'],
    }
    
    required = core_tables.get(database, [])
    found = set(f.lower() for f in result['parquet_files'])
    
    # 如果有 parquet 文件，检查核心表是否存在
    if result['parquet_count'] > 0:
        missing = [t for t in required if t not in found]
        if len(missing) <= 1:  # 允许缺少1个核心表
            result['ready'] = True
        else:
            result['missing_tables'] = missing
    
    return result


def convert_data_with_progress(data_path: str, database: str):
    """带进度条的数据转换功能。"""
    import time
    
    lang = st.session_state.get('language', 'en')
    
    conv_title = "🔄 Data Conversion" if lang == 'en' else "🔄 数据转换"
    st.markdown(f"### {conv_title}")
    
    warn_msg = "⚠️ **Note**: Converting large datasets may take a long time (30min~2hrs), please be patient." if lang == 'en' else "⚠️ **注意**：转换大型数据集可能需要较长时间（30分钟~2小时），请耐心等待。"
    st.warning(warn_msg)
    
    info_msg = "💡 Do not close the page during conversion. After completion, data will be stored in Parquet format for faster loading." if lang == 'en' else "💡 转换过程中请勿关闭页面。转换完成后，数据将以 Parquet 格式存储，后续加载速度将大幅提升。"
    st.info(info_msg)
    
    try:
        from pyricu.data_converter import DataConverter
        
        converter = DataConverter(data_path, database=database, verbose=True)
        
        # 获取需要转换的文件列表
        csv_files = converter._get_csv_files()
        total_files = len(csv_files)
        
        if total_files == 0:
            err_msg = "No CSV files found to convert" if lang == 'en' else "未找到需要转换的 CSV 文件"
            st.error(err_msg)
            return
        
        detect_msg = f"📊 Detected **{total_files}** CSV files to convert" if lang == 'en' else f"📊 共检测到 **{total_files}** 个 CSV 文件需要转换"
        st.markdown(detect_msg)
        
        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        details_container = st.container()
        
        converted = 0
        skipped = 0
        failed = 0
        
        for idx, csv_file in enumerate(csv_files):
            file_name = csv_file.name
            file_size_mb = csv_file.stat().st_size / (1024 * 1024)
            
            # 更新状态
            processing_msg = f"**Processing**: `{file_name}` ({file_size_mb:.1f} MB) [{idx+1}/{total_files}]" if lang == 'en' else f"**正在处理**: `{file_name}` ({file_size_mb:.1f} MB) [{idx+1}/{total_files}]"
            status_text.markdown(processing_msg)
            
            # 检查是否需要转换
            needs_conversion, reason = converter._is_conversion_needed(csv_file)
            
            if not needs_conversion:
                skipped += 1
                with details_container:
                    skip_msg = f"⏭️ Skipped: {file_name} ({reason})" if lang == 'en' else f"⏭️ 跳过: {file_name} ({reason})"
                    st.caption(skip_msg)
            else:
                try:
                    # 执行转换 - 使用正确的方法名
                    converter._convert_file(csv_file)
                    converted += 1
                    with details_container:
                        done_msg = f"✅ Done: {file_name}" if lang == 'en' else f"✅ 完成: {file_name}"
                        st.caption(done_msg)
                except Exception as e:
                    failed += 1
                    with details_container:
                        fail_msg = f"❌ Failed: {file_name} - {str(e)[:50]}" if lang == 'en' else f"❌ 失败: {file_name} - {str(e)[:50]}"
                        st.caption(fail_msg)
            
            # 更新进度
            progress = (idx + 1) / total_files
            progress_bar.progress(progress)
        
        # 转换完成
        progress_bar.progress(1.0)
        status_text.empty()
        
        if lang == 'en':
            summary = f"""
            ✅ **Conversion Complete!**
            - Successfully converted: {converted} files
            - Already existed/skipped: {skipped} files
            - Failed: {failed} files
            """
        else:
            summary = f"""
            ✅ **转换完成！**
            - 成功转换: {converted} 个文件
            - 已存在跳过: {skipped} 个文件
            - 转换失败: {failed} 个文件
            """
        st.success(summary)
        
        if failed == 0:
            st.balloons()
            all_done_msg = "🎉 All data converted successfully, you can now load the data!" if lang == 'en' else "🎉 所有数据已转换完成，现在可以加载数据了！"
            st.info(all_done_msg)
        else:
            partial_msg = "Some files failed to convert, but you can still try loading the converted data." if lang == 'en' else "部分文件转换失败，但您仍可以尝试加载已转换的数据。"
            st.warning(partial_msg)
            
    except ImportError:
        import_err = "Data converter module not installed. Please ensure the full pyricu package is installed." if lang == 'en' else "数据转换模块未安装。请确保已安装完整的 pyricu 包。"
        st.error(import_err)
    except Exception as e:
        conv_err = f"Conversion error: {str(e)}" if lang == 'en' else f"转换过程出错: {str(e)}"
        st.error(conv_err)


# ============ 🚀 智能硬件检测与动态并行配置 ============

def get_system_resources():
    """检测系统硬件资源。
    
    Returns:
        dict: 包含 cpu_count, memory_gb, recommended_workers, recommended_backend
    """
    import os
    import psutil
    
    # CPU 核心数
    cpu_count = os.cpu_count() or 4
    
    # 可用内存 (GB)
    try:
        mem_info = psutil.virtual_memory()
        total_memory_gb = mem_info.total / (1024 ** 3)
        available_memory_gb = mem_info.available / (1024 ** 3)
    except:
        total_memory_gb = 8  # 默认假设 8GB
        available_memory_gb = 4
    
    # 根据硬件资源计算推荐的并行配置
    # 规则：
    # - 每个 worker 大约需要 2GB 内存用于处理 ICU 数据
    # - 不超过 CPU 核心数的 75%（保留系统响应能力）
    # - 最大不超过 64 个 workers（避免过度并行的开销）
    
    max_workers_by_memory = int(available_memory_gb / 2)  # 每 worker 约 2GB
    max_workers_by_cpu = int(cpu_count * 0.75)  # 使用 75% 的 CPU
    
    recommended_workers = min(max_workers_by_memory, max_workers_by_cpu, 64)
    recommended_workers = max(recommended_workers, 1)  # 至少 1 个
    
    # 根据配置选择后端
    # - 高核心数(>16)且内存充足(>32GB): 使用 loky 进程池获得更好的 GIL 规避
    # - 中等配置: 使用 thread 线程池，开销更小
    if cpu_count >= 16 and total_memory_gb >= 32:
        recommended_backend = "loky"
    else:
        recommended_backend = "thread"
    
    return {
        'cpu_count': cpu_count,
        'total_memory_gb': round(total_memory_gb, 1),
        'available_memory_gb': round(available_memory_gb, 1),
        'recommended_workers': recommended_workers,
        'recommended_backend': recommended_backend,
    }


def get_optimal_parallel_config(num_patients: int = None, task_type: str = 'load'):
    """根据系统资源和任务规模返回最优的并行配置。
    
    Args:
        num_patients: 要处理的患者数量，None 表示未知/全量
        task_type: 任务类型 ('load', 'export', 'preview')
    
    Returns:
        tuple: (parallel_workers, parallel_backend)
    """
    resources = get_system_resources()
    base_workers = resources['recommended_workers']
    backend = resources['recommended_backend']
    
    # 根据任务类型调整
    if task_type == 'preview':
        # 预览只需少量数据，不需要太多并行
        workers = min(base_workers, 4)
        backend = "thread"  # 预览用线程更快启动
    elif task_type == 'load':
        # 数据加载根据患者数量调整
        if num_patients is None or num_patients >= 50000:
            workers = base_workers  # 全量使用推荐配置
        elif num_patients >= 10000:
            workers = min(base_workers, max(8, base_workers // 2))
        elif num_patients >= 2000:
            workers = min(base_workers, 4)
        else:
            workers = 1  # 少量患者不需要并行
    elif task_type == 'export':
        # 导出任务可以使用更多资源
        workers = base_workers
    else:
        workers = min(base_workers, 8)
    
    # Streamlit webapp 环境下，线程通常更安全
    # 只有在明确高配置环境下才使用进程池
    if backend == "loky" and task_type != 'export':
        backend = "thread"  # webapp 中优先使用线程
    
    return workers, backend


def init_session_state():
    """初始化 session state。"""
    if 'data_path' not in st.session_state:
        st.session_state.data_path = None
    if 'database' not in st.session_state:
        st.session_state.database = 'miiv'
    if 'loaded_concepts' not in st.session_state:
        st.session_state.loaded_concepts = {}
    if 'patient_ids' not in st.session_state:
        st.session_state.patient_ids = []
    if 'all_patient_count' not in st.session_state:
        st.session_state.all_patient_count = 0
    if 'selected_patient' not in st.session_state:
        st.session_state.selected_patient = None
    if 'use_mock_data' not in st.session_state:
        st.session_state.use_mock_data = False
    if 'id_col' not in st.session_state:
        st.session_state.id_col = 'stay_id'
    # 新增：用于简化流程的状态
    if 'selected_concepts' not in st.session_state:
        st.session_state.selected_concepts = []
    if 'export_completed' not in st.session_state:
        st.session_state.export_completed = False
    if 'mock_params' not in st.session_state:
        st.session_state.mock_params = {'n_patients': 10, 'hours': 72}
    if 'trigger_export' not in st.session_state:
        st.session_state.trigger_export = False
    if 'export_format' not in st.session_state:
        st.session_state.export_format = 'Parquet'  # 默认Parquet
    if 'export_path' not in st.session_state:
        st.session_state.export_path = os.path.expanduser('~/pyricu_export')
    if 'path_validated' not in st.session_state:
        st.session_state.path_validated = False
    if 'language' not in st.session_state:
        st.session_state.language = 'en'  # 默认英文
    # 🚀 性能优化：患者数量限制（默认0表示全量加载，可设为具体数字如5000来限制）
    if 'patient_limit' not in st.session_state:
        st.session_state.patient_limit = 0  # 默认全量
    if 'available_patient_ids' not in st.session_state:
        st.session_state.available_patient_ids = None


# ============ 国际化文本 ============
TEXTS = {
    'en': {
        'app_title': '🏥 PyRICU Data Explorer',
        'app_subtitle': 'Local ICU Data Analytics Platform',
        'select_mode': '🎯 Select Mode',
        'mode_extract': '💾 Data Extraction (New Data)',
        'mode_viz': '📊 Quick Visualization (Existing Data)',
        'step1': 'Step 1: Data Source',
        'step2': 'Step 2: Cohort Selection',
        'step3': 'Step 3: Select Features',
        'step4': 'Step 4: Export Data',
        'demo_mode': '🎭 Demo Mode',
        'real_data': '📁 Real Data',
        'demo_mode_desc': 'System generates simulated ICU data',
        'select_database': 'Select Database',
        'data_path': 'Data Path',
        'validate_path': '✅ Validate Path',
        'path_valid': '✅ Path Valid',
        'path_invalid': '❌ Path Invalid',
        'feature_groups': 'Feature Groups',
        'export_path': 'Export Path',
        'export_format': 'Export Format',
        'export_data': '💾 Export Data',
        'quick_viz': '📈 Quick Visualization',
        'load_data': '🔍 Load Data',
        'loading': 'Loading...',
        'data_loaded': '✅ Data Loaded',
        'features_loaded': 'features loaded',
        'patients_loaded': 'patients loaded',
        'select_tables': 'Select Tables to Load',
        'found_files': 'Found {n} data files',
        'no_files': 'No data files found in this directory',
        'dir_not_exist': 'Directory does not exist',
        'data_dir': '📁 Data Directory',
        'file_list': '📋 File List',
        'loaded_data': '📊 Loaded Data',
        'view_features': 'View Feature List',
        'load_hint': '💡 Select a data directory and load data to start visualization',
        'home': '📚 Tutorial',
        'timeseries': '📈 Time Series',
        'patient_view': '🏥 Patient View',
        'data_quality': '📊 Data Quality',
        'cohort_compare': '📊 Cohort Comparison',
        'ready': '🎉 Ready!',
        'ready_desc': 'Data loaded, you can start exploring.',
        'database': 'Database',
        'features': 'Features',
        'patients': 'Patients',
        'status': 'Status',
        'start_analysis': '🚀 Start Analysis',
        'select_tab': 'Select a tab above to explore data:',
        'data_summary': '📋 Data Summary',
        'n_patients': 'Number of Patients',
        'n_hours': 'Data Duration (hours)',
        'current_task': '📍 Current Task',
        'configure_source': 'Configure Data Source',
        'select_features': 'Select Features',
        'export_or_preview': 'Export Data or Load Preview',
        'data_dict': '📖 Data Dictionary',
        'view_desc': 'View Feature Descriptions',
    },
    'zh': {
        'app_title': '🏥 PyRICU 数据探索器',
        'app_subtitle': '本地 ICU 数据分析与可视化平台',
        'select_mode': '🎯 选择操作模式',
        'mode_extract': '💾 数据提取导出（新数据）',
        'mode_viz': '📊 快速可视化（已有数据）',
        'step1': '步骤1: 数据源',
        'step2': '步骤2: 队列筛选',
        'step3': '步骤3: 选择特征',
        'step4': '步骤4: 导出数据',
        'demo_mode': '🎭 演示模式',
        'real_data': '📁 真实数据',
        'demo_mode_desc': '系统生成模拟ICU数据供体验',
        'select_database': '选择数据库',
        'data_path': '数据路径',
        'validate_path': '✅ 验证路径',
        'path_valid': '✅ 路径有效',
        'path_invalid': '❌ 路径无效',
        'feature_groups': '特征分组',
        'export_path': '导出路径',
        'export_format': '导出格式',
        'export_data': '💾 导出数据',
        'quick_viz': '📈 快速可视化',
        'load_data': '🔍 加载数据',
        'loading': '加载中...',
        'data_loaded': '✅ 数据已加载',
        'features_loaded': '个特征已加载',
        'patients_loaded': '个患者已加载',
        'select_tables': '选择要加载的表格',
        'found_files': '发现 {n} 个数据文件',
        'no_files': '该目录下没有找到数据文件',
        'dir_not_exist': '目录不存在',
        'data_dir': '📁 数据目录',
        'file_list': '📋 文件列表',
        'loaded_data': '📊 已加载数据',
        'view_features': '查看特征列表',
        'load_hint': '💡 选择数据目录并加载数据后，即可在右侧进行可视化分析',
        'home': '📚 教程',
        'timeseries': '📈 时序分析',
        'patient_view': '🏥 患者视图',
        'data_quality': '📊 数据质量',
        'cohort_compare': '📊 队列对比',
        'ready': '🎉 准备就绪！',
        'ready_desc': '数据已加载，您可以开始探索分析了。',
        'database': '数据库',
        'features': '特征',
        'patients': '患者',
        'status': '状态',
        'start_analysis': '🚀 开始分析',
        'select_tab': '选择上方的标签页开始探索数据：',
        'data_summary': '📋 数据摘要',
        'n_patients': '患者数量',
        'n_hours': '数据时长(小时)',
        'current_task': '📍 当前任务',
        'configure_source': '配置数据源',
        'select_features': '选择特征',
        'export_or_preview': '导出数据或加载预览',
        'data_dict': '📖 数据字典',
        'view_desc': '查看特征说明',
    }
}

def get_text(key: str) -> str:
    """根据当前语言获取文本。"""
    lang = st.session_state.get('language', 'en')
    return TEXTS.get(lang, TEXTS['en']).get(key, key)


def validate_database_path(data_path: str, database: str) -> dict:
    """
    验证数据路径是否包含指定数据库所需的文件。
    严格检查每个模块所需的所有表。
    
    返回:
        dict: {'valid': bool, 'message': str, 'suggestion': str (可选)}
    """
    path = Path(data_path)
    lang = st.session_state.get('language', 'en')
    
    # 各数据库需要的核心表（Parquet格式）- 包括分片目录
    # 分为必需表和可选表
    required_parquet_tables = {
        'miiv': {
            'core': ['icustays', 'patients', 'admissions'],  # 核心ID表
            'clinical': ['chartevents', 'labevents', 'inputevents', 'outputevents'],  # 临床数据
            'medication': ['prescriptions', 'ingredientevents'],  # 药物数据
            'other': ['procedureevents', 'd_items', 'd_labitems'],  # 其他
        },
        'eicu': {
            'core': ['patient'],
            'clinical': ['vitalperiodic', 'lab', 'nursecharting'],
            'medication': ['infusiondrug', 'medication'],
        },
        'aumc': {
            'core': ['admissions'],
            'clinical': ['numericitems', 'listitems'],
            'medication': ['drugitems'],
        },
        'hirid': {
            'core': ['general'],  # ricu uses 'general' not 'general_table'
            'clinical': ['observations', 'ordinal'],
            'medication': ['pharma'],  # ricu uses 'pharma' not 'pharma_records'
        },
    }
    
    # 各数据库需要的核心表（CSV/GZ格式 - 原始文件）
    required_csv_files = {
        'miiv': ['icustays.csv', 'chartevents.csv', 'labevents.csv', 'prescriptions.csv', 'inputevents.csv'],
        'eicu': ['patient.csv', 'vitalPeriodic.csv', 'lab.csv'],
        'aumc': ['admissions.csv', 'numericitems.csv', 'drugitems.csv'],
        'hirid': ['general_table.csv', 'pharma_records.csv'],
    }
    
    db_name = {
        'miiv': 'MIMIC-IV', 'eicu': 'eICU-CRD',
        'aumc': 'AmsterdamUMCdb', 'hirid': 'HiRID'
    }.get(database, database.upper())
    
    # 检查Parquet文件和分片目录
    parquet_files = list(path.rglob('*.parquet'))
    parquet_names = set(f.name.lower().replace('.parquet', '') for f in parquet_files)
    
    # 检查分片目录（如 chartevents/1.parquet）
    parquet_dirs = set()
    for pf in parquet_files:
        try:
            if pf.parent != path:
                rel = pf.parent.relative_to(path)
                # 如果是 xxx/1.parquet 格式，记录 xxx
                if pf.stem.isdigit():
                    parquet_dirs.add(pf.parent.name.lower())
        except ValueError:
            pass
    
    # 合并所有找到的表（单文件和分片目录）
    all_found = parquet_names | parquet_dirs
    
    # 检查各类别的表
    db_tables = required_parquet_tables.get(database, {})
    found_tables = []
    missing_tables = []
    missing_by_category = {}
    
    for category, tables in db_tables.items():
        for table in tables:
            if table.lower() in all_found:
                found_tables.append(table)
            else:
                missing_tables.append(table)
                if category not in missing_by_category:
                    missing_by_category[category] = []
                missing_by_category[category].append(table)
    
    total_required = sum(len(tables) for tables in db_tables.values())
    
    # 如果全部找到
    if len(missing_tables) == 0:
        msg = f'✅ {db_name}: All {total_required} required tables found ({len(parquet_files)} Parquet files)' if lang == 'en' else f'✅ {db_name}: 所有 {total_required} 个必需表已找到 ({len(parquet_files)} 个 Parquet 文件)'
        return {
            'valid': True,
            'message': msg
        }
    
    # 核心表缺失是严重问题
    core_missing = missing_by_category.get('core', [])
    if core_missing:
        missing_str = ', '.join(core_missing)
        if lang == 'en':
            msg = f'❌ {db_name}: Missing core tables: {missing_str}'
            sug = f'💡 Core tables are required. Please ensure data is properly converted.'
        else:
            msg = f'❌ {db_name}: 缺少核心表: {missing_str}'
            sug = f'💡 核心表是必需的，请确保数据已正确转换。'
        return {
            'valid': False,
            'message': msg,
            'suggestion': sug,
            'can_convert': True,
            'csv_path': str(path),
            'missing_tables': missing_tables,
        }
    
    # 部分表缺失（非核心）
    if len(found_tables) > 0:
        missing_str = ', '.join(missing_tables[:5])
        if len(missing_tables) > 5:
            missing_str += f' (+{len(missing_tables)-5} more)'
        if lang == 'en':
            msg = f'⚠️ {db_name}: Found {len(found_tables)}/{total_required} tables, missing: {missing_str}'
            sug = f'💡 Click "Convert to Parquet" to convert missing tables'
        else:
            msg = f'⚠️ {db_name}: 找到 {len(found_tables)}/{total_required} 个表，缺少: {missing_str}'
            sug = f'💡 点击「转换为Parquet」转换缺失的表'
        return {
            'valid': False,
            'message': msg,
            'suggestion': sug,
            'can_convert': True,
            'csv_path': str(path),
            'missing_tables': missing_tables,
        }
    
    # 检查是否存在 CSV 文件（可能需要转换）
    csv_files = list(path.rglob('*.csv')) + list(path.rglob('*.csv.gz'))
    csv_names = [f.name.lower().replace('.gz', '') for f in csv_files]
    
    required_csvs = required_csv_files.get(database, [])
    found_csvs = []
    for req in required_csvs:
        if req.lower() in csv_names:
            found_csvs.append(req)
    
    if len(found_csvs) >= len(required_csvs) // 2:
        # 找到 CSV 文件但没有 Parquet - 需要转换
        msg = f'⚠️ Found {db_name} raw CSV files ({len(csv_files)} files), need to convert to Parquet' if lang == 'en' else f'⚠️ 找到 {db_name} 原始 CSV 文件 ({len(csv_files)} 个)，需要转换为 Parquet 格式'
        sug = '💡 Click "Convert to Parquet" button below to convert all files' if lang == 'en' else '💡 点击下方「转换为Parquet」按钮转换所有文件'
        return {
            'valid': False,
            'message': msg,
            'suggestion': sug,
            'can_convert': True,
            'csv_path': str(path)
        }
    
    # 检查是否是子目录结构
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    subdir_names = [d.name.lower() for d in subdirs]
    
    # 检查常见的子目录结构
    expected_subdirs = {
        'miiv': ['hosp', 'icu', 'ed'],
        'eicu': ['eicu-crd'],
        'aumc': ['amsterdamumc'],
        'hirid': ['hirid'],
    }
    
    for expected in expected_subdirs.get(database, []):
        if expected.lower() in subdir_names:
            # 找到预期子目录
            lang = st.session_state.get('language', 'en')
            msg = f'⚠️ Detected {db_name} directory structure, but data may be in subdirectory' if lang == 'en' else f'⚠️ 检测到 {db_name} 目录结构，但数据可能在子目录中'
            sug = f'💡 Try path: {path / expected}' if lang == 'en' else f'💡 请尝试路径: {path / expected}'
            return {
                'valid': False,
                'message': msg,
                'suggestion': sug
            }
    
    # 完全找不到相关文件
    lang = st.session_state.get('language', 'en')
    msg = f'❌ Required data files for {db_name} not found in this path' if lang == 'en' else f'❌ 在此路径下未找到 {db_name} 所需的数据文件'
    sug = '💡 Please verify: 1) Path is correct 2) Database type matches 3) Data is downloaded' if lang == 'en' else '💡 请确认: 1) 路径是否正确 2) 数据库类型是否匹配 3) 数据是否已下载'
    return {
        'valid': False,
        'message': msg,
        'suggestion': sug
    }


def generate_mock_data(n_patients=10, hours=72):
    """生成模拟 ICU 数据用于演示。"""
    data = {}
    patient_ids = list(range(10001, 10001 + n_patients))
    
    np.random.seed(42)
    time_points = np.arange(0, hours, 1)

    # 1. 预先确定患者 Sepsis 状态和发病时间，用于联动 SOFA
    patient_sepsis_meta = {}
    for pid in patient_ids:
        # 30% 概率患 sepsis
        is_septic = np.random.random() < 0.3
        # 发病时间随机分布在 10h ~ hours-10h 之间
        onset = np.random.choice(range(10, max(11, hours-10))) if is_septic else -999
        
        # 确定感染窗口 (samp time)
        samp_time = -1
        if is_septic:
            # 采样时间通常在发病前后
            samp_time = onset + np.random.randint(-4, 4)
            samp_time = max(0, min(hours-1, samp_time))
            
        patient_sepsis_meta[pid] = {
            'is_septic': is_septic,
            'onset': onset,
            'samp_time': samp_time
        }
    
    # 心率
    hr_records = []
    for pid in patient_ids:
        base_hr = np.random.uniform(70, 90)
        # 如果 septic, 心率在发病后升高
        meta = patient_sepsis_meta[pid]
        
        for t in time_points:
            hr = base_hr + np.sin(t / 6) * 10 + np.random.normal(0, 5)
            if meta['is_septic'] and t >= meta['onset']:
                hr += 20 # 发病后心率增加
                
            hr_records.append({'stay_id': pid, 'time': t, 'hr': max(40, min(150, hr))})
    data['hr'] = pd.DataFrame(hr_records)
    
    # MAP
    map_records = []
    for pid in patient_ids:
        base_map = np.random.uniform(65, 85)
        meta = patient_sepsis_meta[pid]
        
        for t in time_points:
            map_val = base_map + np.cos(t / 8) * 8 + np.random.normal(0, 4)
            if meta['is_septic'] and t >= meta['onset']:
                map_val -= 15 # 发病后血压下降
                
            map_records.append({'stay_id': pid, 'time': t, 'map': max(40, min(120, map_val))})
    data['map'] = pd.DataFrame(map_records)
    
    # SBP
    sbp_records = []
    for pid in patient_ids:
        base_sbp = np.random.uniform(110, 140)
        meta = patient_sepsis_meta[pid]
        
        for t in time_points:
            sbp_val = base_sbp + np.sin(t / 5) * 15 + np.random.normal(0, 8)
            if meta['is_septic'] and t >= meta['onset']:
                sbp_val -= 20
                
            sbp_records.append({'stay_id': pid, 'time': t, 'sbp': max(70, min(200, sbp_val))})
    data['sbp'] = pd.DataFrame(sbp_records)
    
    # 体温
    temp_records = []
    for pid in patient_ids:
        base_temp = np.random.uniform(36.5, 37.5)
        meta = patient_sepsis_meta[pid]
        
        for t in time_points[::4]:
            temp_val = base_temp + np.random.normal(0, 0.3)
            # 随机发热
            if np.random.random() < 0.1:
                temp_val += 1.5
            # Sepsis 发热
            if meta['is_septic'] and t >= meta['onset']:
                 temp_val += 1.2
                 
            temp_records.append({'stay_id': pid, 'time': t, 'temp': max(35, min(41, temp_val))})
    data['temp'] = pd.DataFrame(temp_records)
    
    # 呼吸
    resp_records = []
    for pid in patient_ids:
        base_resp = np.random.uniform(14, 18)
        meta = patient_sepsis_meta[pid]
        
        for t in time_points:
            resp_val = base_resp + np.random.normal(0, 2)
            if meta['is_septic'] and t >= meta['onset']:
                resp_val += 8
                
            resp_records.append({'stay_id': pid, 'time': t, 'resp': max(8, min(40, resp_val))})
    data['resp'] = pd.DataFrame(resp_records)
    
    # SpO2
    spo2_records = []
    for pid in patient_ids:
        for t in time_points:
            spo2_val = 97 + np.random.normal(0, 2)
            if np.random.random() < 0.05:
                spo2_val -= 10
            spo2_records.append({'stay_id': pid, 'time': t, 'spo2': max(80, min(100, spo2_val))})
    data['spo2'] = pd.DataFrame(spo2_records)
    
    # SOFA
    sofa_records = []
    for pid in patient_ids:
        meta = patient_sepsis_meta[pid]
        
        for t in time_points[::6]: # 模拟每6小时评分
            # 基础分布
            probs = [0.6, 0.3, 0.1, 0.0, 0.0] 
            
            # 如果是 sepsis 患者且处于发病期，概率向高分偏移
            if meta['is_septic'] and t >= meta['onset']:
                probs = [0.1, 0.2, 0.3, 0.25, 0.15]
                
            sofa_resp = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa_coag = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa_liver = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa_cardio = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa_cns = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa_renal = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa_total = sofa_resp + sofa_coag + sofa_liver + sofa_cardio + sofa_cns + sofa_renal
            
            sofa_records.append({
                'stay_id': pid, 'time': t, 'sofa': sofa_total,
                'sofa_resp': sofa_resp, 'sofa_coag': sofa_coag, 'sofa_liver': sofa_liver,
                'sofa_cardio': sofa_cardio, 'sofa_cns': sofa_cns, 'sofa_renal': sofa_renal,
            })
    data['sofa'] = pd.DataFrame(sofa_records)
    
    # 肌酐
    crea_records = []
    for pid in patient_ids:
        base_crea = np.random.uniform(0.8, 1.2)
        for t in time_points[::8]:
            crea_val = base_crea + np.random.normal(0, 0.2)
            crea_records.append({'stay_id': pid, 'time': t, 'crea': max(0.3, crea_val)})
    data['crea'] = pd.DataFrame(crea_records)
    
    # 胆红素
    bili_records = []
    for pid in patient_ids:
        base_bili = np.random.uniform(0.5, 1.5)
        for t in time_points[::12]:
            bili_val = base_bili + np.random.normal(0, 0.3)
            bili_records.append({'stay_id': pid, 'time': t, 'bili': max(0.1, bili_val)})
    data['bili'] = pd.DataFrame(bili_records)
    
    # 乳酸
    lac_records = []
    for pid in patient_ids:
        base_lac = np.random.uniform(1.0, 2.0)
        meta = patient_sepsis_meta[pid]
        
        for t in time_points[::6]:
            lac_val = base_lac + np.random.normal(0, 0.5)
            if meta['is_septic'] and t >= meta['onset']:
                lac_val += 3.0 # 乳酸升高
                
            lac_records.append({'stay_id': pid, 'time': t, 'lac': max(0.5, lac_val)})
    data['lac'] = pd.DataFrame(lac_records)
    
    # 血小板
    plt_records = []
    for pid in patient_ids:
        base_plt = np.random.uniform(150, 300)
        for t in time_points[::12]:
            plt_val = base_plt + np.random.normal(0, 30)
            plt_records.append({'stay_id': pid, 'time': t, 'plt': max(10, plt_val)})
    data['plt'] = pd.DataFrame(plt_records)
    
    # 去甲肾上腺素
    norepi_records = []
    for pid in patient_ids:
        for t in time_points:
            if 12 <= t <= 48 and np.random.random() < 0.6:
                rate = np.random.uniform(0.05, 0.3)
                norepi_records.append({'stay_id': pid, 'time': t, 'norepi_rate': rate})
    data['norepi_rate'] = pd.DataFrame(norepi_records) if norepi_records else pd.DataFrame(
        columns=['stay_id', 'time', 'norepi_rate'])
    
    # SOFA-2 评分 (2025新标准)
    sofa2_records = []
    for pid in patient_ids:
        meta = patient_sepsis_meta[pid]
        
        for t in time_points[::6]:
            # 基础分布
            probs = [0.55, 0.3, 0.1, 0.05, 0.0]
            if meta['is_septic'] and t >= meta['onset']:
                probs = [0.1, 0.2, 0.3, 0.25, 0.15]
                
            sofa2_resp = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa2_coag = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa2_liver = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa2_cardio = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa2_cns = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa2_renal = np.random.choice([0, 1, 2, 3, 4], p=probs)
            sofa2_total = sofa2_resp + sofa2_coag + sofa2_liver + sofa2_cardio + sofa2_cns + sofa2_renal
            
            sofa2_records.append({
                'stay_id': pid, 'time': t, 'sofa2': sofa2_total,
                'sofa2_resp': sofa2_resp, 'sofa2_coag': sofa2_coag, 'sofa2_liver': sofa2_liver,
                'sofa2_cardio': sofa2_cardio, 'sofa2_cns': sofa2_cns, 'sofa2_renal': sofa2_renal,
            })
    data['sofa2'] = pd.DataFrame(sofa2_records)
    # 添加各组件到 data
    sofa2_df = data['sofa2']
    data['sofa2_resp'] = sofa2_df[['stay_id', 'time', 'sofa2_resp']].copy()
    data['sofa2_coag'] = sofa2_df[['stay_id', 'time', 'sofa2_coag']].copy()
    data['sofa2_liver'] = sofa2_df[['stay_id', 'time', 'sofa2_liver']].copy()
    data['sofa2_cardio'] = sofa2_df[['stay_id', 'time', 'sofa2_cardio']].copy()
    data['sofa2_cns'] = sofa2_df[['stay_id', 'time', 'sofa2_cns']].copy()
    data['sofa2_renal'] = sofa2_df[['stay_id', 'time', 'sofa2_renal']].copy()
    
    # Sepsis-3 诊断数据 (严格基于 SOFA 变化)
    sep3_sofa2_records = []
    
    # 先把 sofa2 转换为 (stay_id, time) 索引以便查询
    sofa2_lookup = data['sofa2'].set_index(['stay_id', 'time'])['sofa2']
    
    for pid in patient_ids:
        meta = patient_sepsis_meta[pid]
        
        # 1. 疑似感染 susp_inf
        # 定义: 在发病前24h到发病后72h为“疑似感染窗口”
        # 这里简化：只要是septic患者，我们在 onset 之后标记疑似
        # samp 仅在具体的采样点为 1
        
        for t in time_points:
            # 默认值
            susp_inf_val = 0
            samp_val = 0
            sep3_val = 0
            infection_icd_val = 0
            
            if meta['is_septic']:
                # infection_icd 通常是静态诊断，整程为1
                infection_icd_val = 1
                
                # samp: 仅在采样点有值 (稀疏)
                if t == meta['samp_time']:
                    samp_val = 1
                    
                # susp_inf: 模拟有一个 suspicions window
                # 此处设为：samp_time 前24h 到 后72h
                samp_t = meta['samp_time']
                if samp_t >= 0 and (samp_t - 24 <= t <= samp_t + 72):
                    susp_inf_val = 1
                    
                # Sepsis-3: susp_inf=1 AND (current_sofa >= 2)
                # 真实标准是 delta_sofa >= 2，假设 baseline=0，则 absolute>=2
                try:
                    # 注意：sofa数据是每6小时采样的，中间时间点可能在dataframe里没有
                    # 这里通过 lookup 查找最近的有效值，或者因为我们的 time_points 包含所有点
                    # 但 sofa2_df 是 dense 的吗？ generate logic 是 time_points[::6]
                    # 我们需要插值或对齐。为简化 mock，我们在生成 sofa2 时只生成了部分点
                    # 但 export 要求对齐。这里我们简单处理：若无法查到准确sofa，则沿用上一个
                    pass 
                except:
                    pass
            
            sep3_sofa2_records.append({
                'stay_id': pid,
                'time': t,
                'sep3_sofa2': 0, # 稍后计算
                'susp_inf': susp_inf_val,
                'infection_icd': infection_icd_val,
                'samp': samp_val,
            })
            
    # 计算 Sep3 结果 (需要合并 sofa 数据)
    sep3_df = pd.DataFrame(sep3_sofa2_records)
    
    # 将 Sep3 表和 SOFA2 表合并计算最终 Sep3 状态
    # SOFA2 是每6小时一点，我们先 forward fill 到每小时
    sofa2_full = pd.DataFrame({'stay_id': patient_ids, 'key': 1}).merge(pd.DataFrame({'time': time_points, 'key': 1}), on='key').drop(columns=['key'])
    sofa2_source = data['sofa2'][['stay_id', 'time', 'sofa2']]
    sofa2_interpolated = sofa2_full.merge(sofa2_source, on=['stay_id', 'time'], how='left')
    sofa2_interpolated['sofa2'] = sofa2_interpolated.groupby('stay_id')['sofa2'].ffill().fillna(0)
    
    # 合并
    sep3_final = sep3_df.merge(sofa2_interpolated, on=['stay_id', 'time'], how='left')
    
    # 应用 Sepsis3 规则: susp_inf == 1 AND sofa2 >= 2
    sep3_final['sep3_sofa2'] = ((sep3_final['susp_inf'] == 1) & (sep3_final['sofa2'] >= 2)).astype(int)
    
    # 更新到 data
    data['sep3_sofa2'] = sep3_final[['stay_id', 'time', 'sep3_sofa2', 'susp_inf', 'infection_icd', 'samp']]
    data['susp_inf'] = sep3_final[['stay_id', 'time', 'susp_inf']]
    data['infection_icd'] = sep3_final[['stay_id', 'time', 'infection_icd']]
    data['samp'] = sep3_final[['stay_id', 'time', 'samp']]
    
    # Sepsis-3 (SOFA-1) 同理
    sofa1_source = data['sofa'][['stay_id', 'time', 'sofa']]
    sofa1_interpolated = sofa2_full.merge(sofa1_source, on=['stay_id', 'time'], how='left')
    sofa1_interpolated['sofa'] = sofa1_interpolated.groupby('stay_id')['sofa'].ffill().fillna(0)
    
    sep3_sofa1_final = sep3_final[['stay_id', 'time', 'susp_inf']].merge(sofa1_interpolated, on=['stay_id', 'time'], how='left')
    sep3_sofa1_final['sep3_sofa1'] = ((sep3_sofa1_final['susp_inf'] == 1) & (sep3_sofa1_final['sofa'] >= 2)).astype(int)
    
    data['sep3_sofa1'] = sep3_sofa1_final[['stay_id', 'time', 'sep3_sofa1']]
    
    # 添加 SOFA-1 各组件到 data
    sofa_df = data['sofa']
    data['sofa_resp'] = sofa_df[['stay_id', 'time', 'sofa_resp']].copy()
    data['sofa_coag'] = sofa_df[['stay_id', 'time', 'sofa_coag']].copy()
    data['sofa_liver'] = sofa_df[['stay_id', 'time', 'sofa_liver']].copy()
    data['sofa_cardio'] = sofa_df[['stay_id', 'time', 'sofa_cardio']].copy()
    data['sofa_cns'] = sofa_df[['stay_id', 'time', 'sofa_cns']].copy()
    data['sofa_renal'] = sofa_df[['stay_id', 'time', 'sofa_renal']].copy()
    
    # ============ 补充更多常用概念 ============
    
    # DBP (舒张压)
    dbp_records = []
    for pid in patient_ids:
        base_dbp = np.random.uniform(60, 80)
        for t in time_points:
            dbp_val = base_dbp + np.sin(t / 5) * 8 + np.random.normal(0, 5)
            dbp_records.append({'stay_id': pid, 'time': t, 'dbp': max(40, min(110, dbp_val))})
    data['dbp'] = pd.DataFrame(dbp_records)
    
    # GCS (格拉斯哥昏迷评分)
    gcs_records = []
    for pid in patient_ids:
        base_gcs = np.random.choice([15, 14, 13, 12, 10, 8], p=[0.5, 0.2, 0.1, 0.08, 0.07, 0.05])
        for t in time_points[::4]:
            gcs_val = base_gcs + np.random.choice([-1, 0, 0, 0, 1], p=[0.1, 0.3, 0.3, 0.2, 0.1])
            gcs_records.append({'stay_id': pid, 'time': t, 'gcs': max(3, min(15, gcs_val))})
    data['gcs'] = pd.DataFrame(gcs_records)
    
    # 血气分析：pH, pco2, po2, lact
    ph_records = []
    pco2_records = []
    po2_records = []
    for pid in patient_ids:
        base_ph = np.random.uniform(7.35, 7.45)
        base_pco2 = np.random.uniform(35, 45)
        base_po2 = np.random.uniform(80, 100)
        for t in time_points[::6]:
            ph_records.append({'stay_id': pid, 'time': t, 'ph': base_ph + np.random.normal(0, 0.03)})
            pco2_records.append({'stay_id': pid, 'time': t, 'pco2': base_pco2 + np.random.normal(0, 3)})
            po2_records.append({'stay_id': pid, 'time': t, 'po2': max(60, base_po2 + np.random.normal(0, 10))})
    data['ph'] = pd.DataFrame(ph_records)
    data['pco2'] = pd.DataFrame(pco2_records)
    data['po2'] = pd.DataFrame(po2_records)
    # lact 已经作为 lac 存在，添加别名
    data['lact'] = data['lac'].rename(columns={'lac': 'lact'}).copy() if 'lac' in data else pd.DataFrame()
    
    # 呼吸系统：pafi, fio2, vent_ind
    pafi_records = []
    fio2_records = []
    vent_ind_records = []
    for pid in patient_ids:
        base_fio2 = np.random.choice([0.21, 0.3, 0.4, 0.5], p=[0.4, 0.3, 0.2, 0.1])
        for t in time_points[::4]:
            fio2_val = base_fio2 + np.random.uniform(-0.05, 0.05)
            fio2_val = max(0.21, min(1.0, fio2_val))
            po2_val = 80 + np.random.normal(0, 15)
            pafi_val = po2_val / fio2_val
            vent = 1 if fio2_val > 0.3 else 0
            pafi_records.append({'stay_id': pid, 'time': t, 'pafi': pafi_val})
            fio2_records.append({'stay_id': pid, 'time': t, 'fio2': fio2_val * 100})  # 转为百分比
            vent_ind_records.append({'stay_id': pid, 'time': t, 'vent_ind': vent})
    data['pafi'] = pd.DataFrame(pafi_records)
    data['fio2'] = pd.DataFrame(fio2_records)
    data['vent_ind'] = pd.DataFrame(vent_ind_records)
    
    # 尿量
    urine_records = []
    for pid in patient_ids:
        for t in time_points:
            urine_val = np.random.uniform(30, 100)
            urine_records.append({'stay_id': pid, 'time': t, 'urine': urine_val})
    data['urine'] = pd.DataFrame(urine_records)
    
    # WBC (白细胞)
    wbc_records = []
    for pid in patient_ids:
        base_wbc = np.random.uniform(6, 12)
        for t in time_points[::12]:
            wbc_val = base_wbc + np.random.normal(0, 2)
            wbc_records.append({'stay_id': pid, 'time': t, 'wbc': max(1, wbc_val)})
    data['wbc'] = pd.DataFrame(wbc_records)
    
    # 结局数据
    death_records = []
    los_icu_records = []
    for pid in patient_ids:
        death = 1 if np.random.random() < 0.15 else 0
        los_icu = np.random.uniform(1, 14)
        death_records.append({'stay_id': pid, 'death': death})
        los_icu_records.append({'stay_id': pid, 'los_icu': los_icu})
    data['death'] = pd.DataFrame(death_records)
    data['los_icu'] = pd.DataFrame(los_icu_records)
    
    # 人口统计
    age_records = []
    weight_records = []
    for pid in patient_ids:
        age_records.append({'stay_id': pid, 'age': np.random.uniform(40, 85)})
        weight_records.append({'stay_id': pid, 'weight': np.random.uniform(50, 100)})
    data['age'] = pd.DataFrame(age_records)
    data['weight'] = pd.DataFrame(weight_records)
    
    # 其他评分
    qsofa_records = []
    sirs_records = []
    for pid in patient_ids:
        for t in time_points[::6]:
            qsofa_records.append({'stay_id': pid, 'time': t, 'qsofa': np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])})
            sirs_records.append({'stay_id': pid, 'time': t, 'sirs': np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.25, 0.25, 0.2, 0.1])})
    data['qsofa'] = pd.DataFrame(qsofa_records)
    data['sirs'] = pd.DataFrame(sirs_records)
    
    # 药物：抗生素使用
    abx_records = []
    for pid in patient_ids:
        abx_records.append({'stay_id': pid, 'abx': 1 if np.random.random() < 0.7 else 0})
    data['abx'] = pd.DataFrame(abx_records)
    
    return data, patient_ids


def render_visualization_mode():
    """渲染快速可视化模式的侧边栏内容。"""
    st.markdown(f"### {get_text('quick_viz')}")
    hint_text = "Load data from exported files for interactive analysis" if st.session_state.language == 'en' else "从已导出的数据加载并进行交互式分析"
    st.caption(hint_text)
    
    # 数据目录选择
    # 优先使用 last_export_dir（导出后记录的实际路径），其次是 export_path
    if st.session_state.get('last_export_dir'):
        default_path = st.session_state.get('last_export_dir')
    else:
        default_path = st.session_state.get('export_path', os.path.expanduser('~/pyricu_export/miiv'))
    
    data_dir = st.text_input(
        get_text('data_dir'),
        value=default_path,
        placeholder="Select exported data directory" if st.session_state.language == 'en' else "选择已导出数据的目录",
        key="viz_data_dir",
        help="Directory containing exported CSV/Parquet/Excel files" if st.session_state.language == 'en' else "包含已导出的 CSV/Parquet/Excel 文件的目录"
    )
    
    # 添加路径检查按钮
    check_btn = "🔍 Check Path" if st.session_state.language == 'en' else "🔍 检查路径"
    if st.button(check_btn, key="check_viz_path", use_container_width=True):
        if data_dir:
            if Path(data_dir).exists():
                files = list(Path(data_dir).glob('*.csv')) + list(Path(data_dir).glob('*.parquet')) + list(Path(data_dir).glob('*.xlsx'))
                if files:
                    ok_msg = f"✅ Path valid! Found {len(files)} data files" if st.session_state.language == 'en' else f"✅ 路径有效！发现 {len(files)} 个数据文件"
                    st.success(ok_msg)
                else:
                    warn_msg = "⚠️ Directory exists but no data files found" if st.session_state.language == 'en' else "⚠️ 目录存在但未找到数据文件"
                    st.warning(warn_msg)
            else:
                err_msg = "❌ Path does not exist" if st.session_state.language == 'en' else "❌ 路径不存在"
                st.error(err_msg)
        else:
            warn_msg = "⚠️ Please enter a path first" if st.session_state.language == 'en' else "⚠️ 请先输入路径"
            st.warning(warn_msg)
    
    if data_dir and Path(data_dir).exists():
        # 扫描可用文件
        available_files = list(Path(data_dir).glob('*.csv')) + \
                          list(Path(data_dir).glob('*.parquet')) + \
                          list(Path(data_dir).glob('*.xlsx'))
        
        if available_files:
            file_names = [f.stem for f in available_files]
            found_msg = f"✅ Found {len(available_files)} data files" if st.session_state.language == 'en' else f"✅ 发现 {len(available_files)} 个数据文件"
            st.success(found_msg)
            
            # 让用户选择要加载的表格
            select_label = "Select Tables to Load" if st.session_state.language == 'en' else "选择要加载的表格"
            select_help = "Select tables to load for visualization (max 3 recommended)" if st.session_state.language == 'en' else "选择要加载到可视化的表格（建议不超过3个以保证流畅性）"
            selected_files = st.multiselect(
                select_label,
                options=file_names,
                default=file_names[:3] if len(file_names) <= 5 else file_names[:2],
                help=select_help,
                key="viz_selected_files"
            )
            
            if selected_files:
                selected_msg = f"{len(selected_files)} tables selected" if st.session_state.language == 'en' else f"已选 {len(selected_files)} 个表格"
                st.caption(selected_msg)
                
                # 患者数量选择器
                st.markdown("---")
                patient_limit_label = "Patients to Load" if st.session_state.language == 'en' else "加载患者数量"
                
                # 使用 selectbox 代替 slider，提供预设选项和"全部"选项
                patient_options = [50, 100, 200, 500, -1]  # -1 表示全部
                option_labels = {
                    50: "50 (Fast)" if st.session_state.language == 'en' else "50 (快速)",
                    100: "100 (Recommended)" if st.session_state.language == 'en' else "100 (推荐)",
                    200: "200 (Slow)" if st.session_state.language == 'en' else "200 (较慢)",
                    500: "500 (Very Slow)" if st.session_state.language == 'en' else "500 (很慢)",
                    -1: "🔓 All (May Lag!)" if st.session_state.language == 'en' else "🔓 全部 (可能卡顿！)"
                }
                
                selected_option = st.selectbox(
                    patient_limit_label,
                    options=patient_options,
                    index=1,  # 默认选择100
                    format_func=lambda x: option_labels[x],
                    key="viz_max_patients"
                )
                
                # 根据选择显示警告
                if selected_option == -1:
                    all_warn = "⚠️ Loading ALL patients may cause UI lag or crash for large datasets!" if st.session_state.language == 'en' else "⚠️ 加载全部患者可能导致界面卡顿甚至崩溃！大数据集请谨慎使用"
                    st.warning(all_warn)
                    max_patients = None  # None 表示不限制
                elif selected_option >= 200:
                    perf_warn = "⚠️ High patient count may cause slow performance" if st.session_state.language == 'en' else "⚠️ 患者数较多，性能可能下降"
                    st.warning(perf_warn)
                    max_patients = selected_option
                else:
                    max_patients = selected_option
                
                st.markdown("---")
                
                # 显示加载状态
                is_loaded = len(st.session_state.loaded_concepts) > 0
                if is_loaded:
                    loaded_msg = f"📊 {len(st.session_state.loaded_concepts)} features, {len(st.session_state.patient_ids)} patients loaded" if st.session_state.language == 'en' else f"📊 已加载 {len(st.session_state.loaded_concepts)} 个特征，{len(st.session_state.patient_ids)} 个患者"
                    st.info(loaded_msg)
                
                if st.button(get_text('load_data'), type="primary", use_container_width=True):
                    loading_msg = "Loading data..." if st.session_state.language == 'en' else "正在加载数据..."
                    with st.spinner(loading_msg):
                        load_from_exported(data_dir, selected_files=selected_files, max_patients=max_patients)
                    st.rerun()
            else:
                st.button(get_text('load_data'), type="primary", use_container_width=True, disabled=True)
                warn_msg = "⚠️ Please select at least one table" if st.session_state.language == 'en' else "⚠️ 请选择至少一个表格"
                st.caption(warn_msg)
            
            # 显示文件预览
            with st.expander(get_text('file_list'), expanded=False):
                for f in available_files[:10]:
                    st.caption(f"• {f.name}")
                if len(available_files) > 10:
                    more_msg = f"... {len(available_files)} files total" if st.session_state.language == 'en' else f"... 共 {len(available_files)} 个文件"
                    st.caption(more_msg)
        else:
            st.warning(get_text('no_files'))
            format_msg = "Supported formats: CSV, Parquet, Excel" if st.session_state.language == 'en' else "支持格式：CSV、Parquet、Excel"
            st.caption(format_msg)
    elif data_dir:
        st.error(get_text('dir_not_exist'))
        check_msg = "Please check if the path is correct" if st.session_state.language == 'en' else "请检查路径是否正确"
        st.caption(check_msg)
    
    st.markdown("---")
    
    # 显示已加载数据的状态
    if len(st.session_state.loaded_concepts) > 0:
        st.markdown(f"### {get_text('loaded_data')}")
        feat_msg = f"✅ {len(st.session_state.loaded_concepts)} features" if st.session_state.language == 'en' else f"✅ {len(st.session_state.loaded_concepts)} 个特征"
        pat_msg = f"✅ {len(st.session_state.patient_ids)} patients" if st.session_state.language == 'en' else f"✅ {len(st.session_state.patient_ids)} 个患者"
        st.success(feat_msg)
        st.success(pat_msg)
        
        with st.expander(get_text('view_features'), expanded=False):
            for concept in sorted(st.session_state.loaded_concepts.keys()):
                st.caption(f"• {concept}")
    else:
        st.info(get_text('load_hint'))


def render_sidebar():
    """渲染侧边栏 - 简化版：选择 → 导出，无需加载到内存。"""
    # 使用双语特征分组
    concept_groups = get_concept_groups()
    
    # 所有可用的 concepts 列表（用于自定义选择）
    all_available_concepts = sorted(set(c for group_concepts in concept_groups.values() for c in group_concepts))
    
    with st.sidebar:
        st.markdown(f"## {get_text('app_title')}")
        
        # 显示系统资源状态
        available_mem = get_available_memory_gb()
        if available_mem < 2:
            st.warning(f"⚠️ Low memory: {available_mem:.1f}GB" if st.session_state.get('language') == 'en' else f"⚠️ 内存不足: {available_mem:.1f}GB")
        elif LOW_MEMORY_MODE:
            st.info("💾 Low Memory Mode" if st.session_state.get('language') == 'en' else "💾 低内存模式")
        
        # 语言切换 - 更紧凑的布局
        lang = st.selectbox(
            "🌐 Language",
            options=['EN', 'ZH'],
            index=0 if st.session_state.language == 'en' else 1,
            key="lang_select",
        )
        if (lang == 'EN' and st.session_state.language != 'en') or \
           (lang == 'ZH' and st.session_state.language != 'zh'):
            st.session_state.language = 'en' if lang == 'EN' else 'zh'
            st.rerun()
        
        st.markdown("---")
        
        # ============ 快捷入口：两个并列模式 ============
        st.markdown(f"**{get_text('select_mode')}**")
        
        # 初始化模式状态
        if 'app_mode' not in st.session_state:
            st.session_state.app_mode = 'extract'  # 默认为数据提取模式
        
        # 自定义样式的模式选择按钮
        extract_selected = st.session_state.app_mode == 'extract'
        viz_selected = st.session_state.app_mode == 'viz'
        
        # 定义选中和未选中的样式
        if st.session_state.language == 'en':
            extract_label = "📤 Data Extraction"
            viz_label = "📊 Quick Visualization"
        else:
            extract_label = "📤 数据提取导出"
            viz_label = "📊 快速可视化"
        
        # 使用HTML渲染漂亮的模式切换按钮 - 更明显的样式区分
        st.markdown("""
        <style>
        .mode-btn-container {
            display: flex;
            gap: 8px;
            margin: 10px 0;
        }
        .mode-btn {
            flex: 1;
            padding: 16px 12px;
            border-radius: 10px;
            text-align: center;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }
        .mode-btn-active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.5) !important;
            transform: scale(1.02);
        }
        .mode-btn-inactive {
            background: #f8f9fa !important;
            color: #666 !important;
            border: 2px dashed #ccc !important;
            opacity: 0.7;
        }
        .mode-btn-inactive:hover {
            background: #e8eaee !important;
            border-color: #999 !important;
            opacity: 1;
        }
        /* 更强的样式覆盖：选中状态 */
        div[data-testid="stHorizontalBlock"] div[data-testid="column"]:first-child button[kind="primary"],
        div[data-testid="stHorizontalBlock"] div[data-testid="column"]:last-child button[kind="primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: 3px solid #667eea !important;
            box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6) !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            transform: scale(1.03);
            animation: pulse-selected 2s infinite;
        }
        @keyframes pulse-selected {
            0%, 100% { box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6); }
            50% { box-shadow: 0 8px 35px rgba(102, 126, 234, 0.8); }
        }
        /* 更明显的未选中样式 */
        div[data-testid="stHorizontalBlock"] div[data-testid="column"]:first-child button[kind="secondary"],
        div[data-testid="stHorizontalBlock"] div[data-testid="column"]:last-child button[kind="secondary"] {
            background: #f8f9fa !important;
            color: #888 !important;
            border: 2px dashed #ccc !important;
            opacity: 0.65;
            font-weight: 500 !important;
        }
        div[data-testid="stHorizontalBlock"] div[data-testid="column"] button[kind="secondary"]:hover {
            opacity: 1;
            border-color: #667eea !important;
            background: #f0f0ff !important;
            color: #667eea !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 显示当前选中模式的指示器
        current_mode_indicator = f"🎯 **{'Data Extraction' if extract_selected else 'Quick Visualization'}** mode active" if st.session_state.language == 'en' else f"🎯 当前模式: **{'数据提取导出' if extract_selected else '快速可视化'}**"
        st.markdown(current_mode_indicator)
        
        # 使用两列放置按钮 - 所有模式都用按钮，确保可点击
        mode_cols = st.columns(2)
        
        with mode_cols[0]:
            # 数据提取按钮 - 总是可点击
            btn_type = "primary" if extract_selected else "secondary"
            if st.button(extract_label, key="btn_mode_extract", use_container_width=True, type=btn_type):
                if not extract_selected:
                    st.session_state.app_mode = 'extract'
                    # 切换模式时清空已加载数据和相关状态
                    st.session_state.loaded_concepts = {}
                    st.session_state.patient_ids = []
                    st.session_state.selected_patient = None
                    st.session_state.concept_dataframes = {}
                    st.rerun()
        
        with mode_cols[1]:
            # 快速可视化按钮 - 总是可点击
            btn_type = "primary" if viz_selected else "secondary"
            if st.button(viz_label, key="btn_mode_viz", use_container_width=True, type=btn_type):
                if not viz_selected:
                    st.session_state.app_mode = 'viz'
                    # 切换模式时清空已加载数据和相关状态
                    st.session_state.loaded_concepts = {}
                    st.session_state.patient_ids = []
                    st.session_state.selected_patient = None
                    st.session_state.concept_dataframes = {}
                    st.rerun()
        
        # 根据选择设置mode变量
        mode = get_text('mode_viz') if st.session_state.app_mode == 'viz' else get_text('mode_extract')
        
        st.markdown("---")
        
        # ============ 根据模式显示不同内容 ============
        if mode == get_text('mode_viz'):
            # 快速可视化模式 - 直接从已导出的数据加载
            render_visualization_mode()
            return
        
        # ============ 数据提取导出模式 ============
        # ============ 步骤1: 数据源选择 ============
        st.markdown(f"### 📊 {get_text('step1')}")
        
        # 数据模式选择
        demo_label = "🎭 Demo Mode" if st.session_state.language == 'en' else "🎭 演示模式"
        real_label = "📁 Real Data" if st.session_state.language == 'en' else "📁 真实数据"
        data_source_help = "Demo mode uses simulated data; Real data mode exports from local ICU databases" if st.session_state.language == 'en' else "演示模式使用模拟数据；真实数据模式从本地ICU数据库导出"
        data_mode = st.radio(
            "Select Data Source" if st.session_state.language == 'en' else "选择数据来源",
            options=[demo_label, real_label],
            index=0 if st.session_state.use_mock_data else 1,
            help=data_source_help,
            label_visibility="collapsed"
        )
        
        use_mock = data_mode == demo_label
        st.session_state.use_mock_data = use_mock
        
        if use_mock:
            demo_title = "✨ Demo Mode" if st.session_state.language == 'en' else "✨ 演示模式"
            demo_desc = "System generates simulated ICU data for exploration" if st.session_state.language == 'en' else "系统生成模拟ICU数据供体验"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f77b4, #2ca02c); 
                        padding: 10px 14px; border-radius: 8px; color: white; margin: 8px 0;">
                <b>{demo_title}</b><br>
                <small>{demo_desc}</small>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.database = 'mock'
            
            # 模拟数据参数
            n_patients_label = "Number of Patients" if st.session_state.language == 'en' else "患者数量"
            hours_label = "Data Duration (hours)" if st.session_state.language == 'en' else "数据时长(小时)"
            n_patients = st.slider(n_patients_label, 5, 50, 10)
            hours = st.slider(hours_label, 24, 168, 72)
            st.session_state.mock_params = {'n_patients': n_patients, 'hours': hours}
            
        else:
            # 真实数据模式
            db_label = "Select Database" if st.session_state.language == 'en' else "选择数据库"
            database = st.selectbox(
                db_label,
                options=['miiv', 'eicu', 'aumc', 'hirid'],
                index=0,
                format_func=lambda x: {
                    'miiv': 'MIMIC-IV', 'eicu': 'eICU-CRD', 
                    'aumc': 'AmsterdamUMCdb', 'hirid': 'HiRID'
                }.get(x, x)
            )
            st.session_state.database = database
            
            default_path = "/home/1_publicData/icu_databases/mimiciv/3.1/" if database == 'miiv' else ""
            path_label = "Data Path" if st.session_state.language == 'en' else "数据路径"
            data_path = st.text_input(
                path_label,
                value=st.session_state.data_path or default_path,
                placeholder=f"/path/to/{database}"
            )
            
            # 验证按钮
            validate_btn = "🔍 Validate Data Path" if st.session_state.language == 'en' else "🔍 验证数据路径"
            if st.button(validate_btn, width="stretch", key="validate_path"):
                if not data_path:
                    err_msg = "❌ Please enter data path" if st.session_state.language == 'en' else "❌ 请输入数据路径"
                    st.error(err_msg)
                elif not Path(data_path).exists():
                    err_msg = "❌ Path does not exist" if st.session_state.language == 'en' else "❌ 路径不存在"
                    st.error(err_msg)
                else:
                    # 检查数据库所需文件
                    validation_result = validate_database_path(data_path, database)
                    st.session_state.last_validation = validation_result
                    st.session_state.last_validated_path = data_path
                    
                    if validation_result['valid']:
                        st.session_state.data_path = data_path
                        st.session_state.path_validated = True
                        st.success(f"✅ {validation_result['message']}")
                    else:
                        st.session_state.path_validated = False
                        st.error(validation_result['message'])
                        if validation_result.get('suggestion'):
                            st.info(validation_result['suggestion'])
            
            # 显示当前验证状态和转换按钮
            last_validation = st.session_state.get('last_validation', {})
            last_path = st.session_state.get('last_validated_path', '')
            
            if st.session_state.get('path_validated') and st.session_state.data_path == data_path:
                validated_msg = "✅ Path validated" if st.session_state.language == 'en' else "✅ 路径已验证"
                st.success(validated_msg)
            elif last_validation.get('can_convert') and last_path == data_path:
                # 显示转换按钮
                convert_btn = "🔄 Convert to Parquet" if st.session_state.language == 'en' else "🔄 转换为Parquet"
                if st.button(convert_btn, width="stretch", type="primary", key="convert_csv"):
                    st.session_state.show_convert_dialog = True
                    st.session_state.convert_source_path = data_path
                    st.rerun()
                csv_hint = "💡 Or click below to use raw CSV (slower)" if st.session_state.language == 'en' else "💡 或点击下方使用原始CSV（较慢）"
                st.caption(csv_hint)
                use_csv_btn = "📂 Use Raw CSV Data" if st.session_state.language == 'en' else "📂 使用原始CSV数据"
                if st.button(use_csv_btn, width="stretch", key="use_csv"):
                    st.session_state.data_path = data_path
                    st.session_state.path_validated = True
                    csv_ok_msg = "✅ Will use CSV format (slower loading)" if st.session_state.language == 'en' else "✅ 将使用CSV格式（加载较慢）"
                    st.success(csv_ok_msg)
                    st.rerun()
            elif data_path and Path(data_path).exists():
                validate_hint = "💡 Click the button above to validate data format" if st.session_state.language == 'en' else "💡 点击上方按钮验证数据格式"
                st.caption(validate_hint)
        
        st.markdown("---")
        
        # ============ 步骤2: 队列筛选（新增） ============
        step2_cohort_title = "Step 2: Cohort Selection" if st.session_state.language == 'en' else "步骤2: 队列筛选"
        st.markdown(f"### 👥 {step2_cohort_title}")
        
        # 初始化队列筛选的 session state
        if 'cohort_filter' not in st.session_state:
            st.session_state.cohort_filter = {
                'age_min': None,
                'age_max': None,
                'first_icu_stay': None,
                'los_min': None,
                'los_max': None,
                'gender': None,
                'survived': None,
                'has_sepsis': None,
            }
        if 'cohort_enabled' not in st.session_state:
            st.session_state.cohort_enabled = False
        if 'filtered_patient_count' not in st.session_state:
            st.session_state.filtered_patient_count = None
        
        # 启用队列筛选开关
        cohort_toggle_label = "Enable Cohort Filtering" if st.session_state.language == 'en' else "启用队列筛选"
        cohort_help = "Filter patients by demographics and clinical criteria" if st.session_state.language == 'en' else "根据人口统计学和临床标准筛选患者"
        cohort_enabled = st.toggle(cohort_toggle_label, value=st.session_state.cohort_enabled, help=cohort_help)
        st.session_state.cohort_enabled = cohort_enabled
        
        if cohort_enabled:
            # 年龄筛选
            age_label = "🎂 Age Range" if st.session_state.language == 'en' else "🎂 年龄范围"
            with st.expander(age_label, expanded=True):
                age_col1, age_col2 = st.columns(2)
                with age_col1:
                    age_min_label = "Min Age" if st.session_state.language == 'en' else "最小年龄"
                    age_min = st.number_input(
                        age_min_label, min_value=0, max_value=120, 
                        value=18 if st.session_state.cohort_filter['age_min'] is None else int(st.session_state.cohort_filter['age_min']),
                        key="cohort_age_min"
                    )
                    if age_min > 0:
                        st.session_state.cohort_filter['age_min'] = age_min
                    else:
                        st.session_state.cohort_filter['age_min'] = None
                with age_col2:
                    age_max_label = "Max Age" if st.session_state.language == 'en' else "最大年龄"
                    age_max = st.number_input(
                        age_max_label, min_value=0, max_value=120, 
                        value=100 if st.session_state.cohort_filter['age_max'] is None else int(st.session_state.cohort_filter['age_max']),
                        key="cohort_age_max"
                    )
                    if age_max < 120:
                        st.session_state.cohort_filter['age_max'] = age_max
                    else:
                        st.session_state.cohort_filter['age_max'] = None
            
            # 首次入ICU筛选
            first_icu_label = "🏥 First ICU Stay Only" if st.session_state.language == 'en' else "🏥 仅首次入ICU"
            first_icu_options = {
                'any': 'Any' if st.session_state.language == 'en' else '不限',
                'yes': 'Yes (First ICU only)' if st.session_state.language == 'en' else '是（仅首次）',
                'no': 'No (Readmissions only)' if st.session_state.language == 'en' else '否（仅再入院）',
            }
            first_icu_val = st.radio(
                first_icu_label,
                options=list(first_icu_options.keys()),
                format_func=lambda x: first_icu_options[x],
                index=0,
                horizontal=True,
                key="cohort_first_icu"
            )
            if first_icu_val == 'yes':
                st.session_state.cohort_filter['first_icu_stay'] = True
            elif first_icu_val == 'no':
                st.session_state.cohort_filter['first_icu_stay'] = False
            else:
                st.session_state.cohort_filter['first_icu_stay'] = None
            
            # 住院时长筛选（只需要最短时长，默认24小时）
            los_label = "⏱️ Min ICU Stay (hours)" if st.session_state.language == 'en' else "⏱️ 最短住院时长（小时）"
            los_help = "Minimum ICU stay duration to include patients (default 24h)" if st.session_state.language == 'en' else "纳入患者的最短ICU住院时长（默认24小时）"
            los_min = st.number_input(
                los_label, min_value=0, max_value=10000, value=24,
                help=los_help,
                key="cohort_los_min"
            )
            st.session_state.cohort_filter['los_min'] = los_min if los_min > 0 else None
            st.session_state.cohort_filter['los_max'] = None  # 不再使用max
            
            # 性别筛选
            gender_label = "👤 Gender" if st.session_state.language == 'en' else "👤 性别"
            gender_options = {
                'any': 'Any' if st.session_state.language == 'en' else '不限',
                'M': 'Male' if st.session_state.language == 'en' else '男性',
                'F': 'Female' if st.session_state.language == 'en' else '女性',
            }
            gender_val = st.radio(
                gender_label,
                options=list(gender_options.keys()),
                format_func=lambda x: gender_options[x],
                index=0,
                horizontal=True,
                key="cohort_gender"
            )
            st.session_state.cohort_filter['gender'] = gender_val if gender_val != 'any' else None
            
            # 存活状态筛选
            survival_label = "💚 Survival Status" if st.session_state.language == 'en' else "💚 存活状态"
            survival_options = {
                'any': 'Any' if st.session_state.language == 'en' else '不限',
                'survived': 'Survived' if st.session_state.language == 'en' else '存活',
                'deceased': 'Deceased' if st.session_state.language == 'en' else '死亡',
            }
            survival_val = st.radio(
                survival_label,
                options=list(survival_options.keys()),
                format_func=lambda x: survival_options[x],
                index=0,
                horizontal=True,
                key="cohort_survival"
            )
            if survival_val == 'survived':
                st.session_state.cohort_filter['survived'] = True
            elif survival_val == 'deceased':
                st.session_state.cohort_filter['survived'] = False
            else:
                st.session_state.cohort_filter['survived'] = None
            
            # Sepsis筛选
            sepsis_label = "🦠 Sepsis Diagnosis" if st.session_state.language == 'en' else "🦠 脓毒症诊断"
            sepsis_options = {
                'any': 'Any' if st.session_state.language == 'en' else '不限',
                'yes': 'Has Sepsis' if st.session_state.language == 'en' else '有脓毒症',
                'no': 'No Sepsis' if st.session_state.language == 'en' else '无脓毒症',
            }
            sepsis_val = st.radio(
                sepsis_label,
                options=list(sepsis_options.keys()),
                format_func=lambda x: sepsis_options[x],
                index=0,
                horizontal=True,
                key="cohort_sepsis"
            )
            if sepsis_val == 'yes':
                st.session_state.cohort_filter['has_sepsis'] = True
            elif sepsis_val == 'no':
                st.session_state.cohort_filter['has_sepsis'] = False
            else:
                st.session_state.cohort_filter['has_sepsis'] = None
            
            # 显示当前筛选条件摘要
            filter_summary = []
            cf = st.session_state.cohort_filter
            if cf['age_min'] is not None or cf['age_max'] is not None:
                age_range = f"{cf['age_min'] or 0}-{cf['age_max'] or '∞'}"
                filter_summary.append(f"Age: {age_range}" if st.session_state.language == 'en' else f"年龄: {age_range}")
            if cf['first_icu_stay'] is not None:
                filter_summary.append(f"First ICU: {'Yes' if cf['first_icu_stay'] else 'No'}" if st.session_state.language == 'en' else f"首次入ICU: {'是' if cf['first_icu_stay'] else '否'}")
            if cf['gender'] is not None:
                filter_summary.append(f"Gender: {cf['gender']}" if st.session_state.language == 'en' else f"性别: {'男' if cf['gender']=='M' else '女'}")
            if cf['survived'] is not None:
                filter_summary.append(f"Survived: {'Yes' if cf['survived'] else 'No'}" if st.session_state.language == 'en' else f"存活: {'是' if cf['survived'] else '否'}")
            if cf['has_sepsis'] is not None:
                filter_summary.append(f"Sepsis: {'Yes' if cf['has_sepsis'] else 'No'}" if st.session_state.language == 'en' else f"脓毒症: {'是' if cf['has_sepsis'] else '否'}")
            
            if filter_summary:
                summary_text = " | ".join(filter_summary)
                st.info(f"📋 {summary_text}")
            else:
                no_filter_msg = "No filters applied (will load all patients)" if st.session_state.language == 'en' else "未设置筛选条件（将加载所有患者）"
                st.caption(no_filter_msg)
        else:
            # 队列筛选禁用时的提示
            disabled_msg = "💡 Enable cohort filtering to select specific patient populations" if st.session_state.language == 'en' else "💡 启用队列筛选可选择特定患者人群"
            st.caption(disabled_msg)
        
        st.markdown("---")
        
        # ============ 步骤3: Concept 选择 ============
        step3_title = "Step 3: Select Features" if st.session_state.language == 'en' else "步骤3: 选择特征"
        st.markdown(f"### 🔧 {step3_title}")
        
        # 初始化 session state
        if 'concept_checkboxes' not in st.session_state:
            st.session_state.concept_checkboxes = {}
        if 'selected_groups' not in st.session_state:
            st.session_state.selected_groups = []
        
        selected_concepts = []
        
        # 使用 multiselect 管理类别选择
        valid_defaults = [g for g in st.session_state.selected_groups if g in concept_groups]
        
        cat_label = "Select Feature Categories" if st.session_state.language == 'en' else "选择特征类别"
        cat_help = "Multi-select, click × to remove" if st.session_state.language == 'en' else "可多选，点击 × 删除"
        cat_placeholder = "Click to select..." if st.session_state.language == 'en' else "点击选择..."
        
        current_selection = st.multiselect(
            cat_label,
            options=list(concept_groups.keys()),
            default=valid_defaults,
            help=cat_help,
            placeholder=cat_placeholder
        )
        
        # 检测变化并更新
        if current_selection != st.session_state.selected_groups:
            added_groups = set(current_selection) - set(st.session_state.selected_groups)
            for grp in added_groups:
                for concept in concept_groups.get(grp, []):
                    st.session_state.concept_checkboxes[concept] = True
            
            removed_groups = set(st.session_state.selected_groups) - set(current_selection)
            for grp in removed_groups:
                for concept in concept_groups.get(grp, []):
                    if concept in st.session_state.concept_checkboxes:
                        del st.session_state.concept_checkboxes[concept]
            
            st.session_state.selected_groups = current_selection
            st.rerun()
        
        # 显示已选类别的详细特征配置
        if st.session_state.selected_groups:
            import hashlib
            
            detail_label = "🎯 Feature Detail Configuration" if st.session_state.language == 'en' else "🎯 特征详细配置"
            with st.expander(detail_label, expanded=True):
                for group_name in st.session_state.selected_groups:
                    if group_name not in concept_groups:
                        continue
                    key_hash = hashlib.md5(group_name.encode()).hexdigest()[:8]
                    
                    st.markdown(f"**{group_name}**")
                    group_concepts = concept_groups.get(group_name, [])
                    cols = st.columns(3)
                    for cidx, concept in enumerate(group_concepts):
                        with cols[cidx % 3]:
                            default_val = st.session_state.concept_checkboxes.get(concept, True)
                            checked = st.checkbox(concept, value=default_val, key=f"cb_{key_hash}_{concept}")
                            st.session_state.concept_checkboxes[concept] = checked
                    st.markdown("---")
            
            # 收集所有选中的 concepts
            for group_name in st.session_state.selected_groups:
                for concept in concept_groups.get(group_name, []):
                    if st.session_state.concept_checkboxes.get(concept, True):
                        selected_concepts.append(concept)
            
            selected_concepts = list(set(selected_concepts))
            selected_msg = f"✅ {len(selected_concepts)} features selected" if st.session_state.language == 'en' else f"✅ 已选 {len(selected_concepts)} 个特征"
            st.success(selected_msg)
        
        st.session_state.selected_concepts = selected_concepts
        
        st.markdown("---")
        
        # ============ 步骤4: 直接导出 ============
        step4_title = "Step 4: Export Data" if st.session_state.language == 'en' else "步骤4: 导出数据"
        st.markdown(f"### 💾 {step4_title}")
        
        # 导出路径配置 - 实时根据数据库显示子目录
        base_export_path = os.path.expanduser('~/pyricu_export')
        db_name = st.session_state.get('database', 'mock')
        default_export_path = str(Path(base_export_path) / db_name)
        
        export_path = st.text_input(
            "Export Path" if st.session_state.language == 'en' else "导出路径",
            value=default_export_path,
            placeholder="Select export directory" if st.session_state.language == 'en' else "选择导出目录",
            help=(f"Data will be exported to this directory (Current database: {db_name.upper()})" if st.session_state.language == 'en' else f"数据将导出到此目录（当前数据库: {db_name.upper()}）")
        )
        st.session_state.export_path = export_path
        
        # 检查路径并提供创建选项
        if export_path:
            if Path(export_path).exists():
                path_ok_msg = "✅ Path valid" if st.session_state.language == 'en' else "✅ 路径有效"
                st.success(path_ok_msg)
            else:
                col_create, col_info = st.columns([1, 2])
                with col_create:
                    create_btn = "📁 Create Directory" if st.session_state.language == 'en' else "📁 创建目录"
                    if st.button(create_btn, key="create_export_dir"):
                        try:
                            Path(export_path).mkdir(parents=True, exist_ok=True)
                            ok_msg = "✅ Directory created" if st.session_state.language == 'en' else "✅ 目录已创建"
                            st.success(ok_msg)
                            st.rerun()
                        except Exception as e:
                            err_msg = f"Creation failed: {e}" if st.session_state.language == 'en' else f"创建失败: {e}"
                            st.error(err_msg)
                with col_info:
                    not_exist_msg = "Path does not exist" if st.session_state.language == 'en' else "路径不存在"
                    st.caption(not_exist_msg)
        
        # 导出格式选择（优先Parquet）
        format_label = "Export Format" if st.session_state.language == 'en' else "导出格式"
        format_help = "Parquet format is smaller and faster to load, recommended" if st.session_state.language == 'en' else "Parquet格式体积小、加载快，推荐使用"
        export_format = st.selectbox(
            format_label,
            options=['Parquet', 'CSV', 'Excel'],
            index=0,
            help=format_help
        )
        st.session_state.export_format = export_format
        
        # 🚀 患者数量限制（性能优化选项）
        limit_label = "Patient Limit" if st.session_state.language == 'en' else "患者数量限制"
        limit_help = "Limit number of patients to speed up loading. 0 = no limit (full data, may be slow)" if st.session_state.language == 'en' else "限制加载的患者数量以加速。0 = 不限制（全量数据，可能较慢）"
        patient_limit_options = [0, 1000, 5000, 10000, 20000, 50000]
        patient_limit_labels = {
            0: "All patients (slower)" if st.session_state.language == 'en' else "全部患者（较慢）",
            1000: "1,000",
            5000: "5,000", 
            10000: "10,000",
            20000: "20,000",
            50000: "50,000"
        }
        current_limit = st.session_state.get('patient_limit', 0)
        if current_limit not in patient_limit_options:
            current_limit = 0
        patient_limit = st.selectbox(
            limit_label,
            options=patient_limit_options,
            index=patient_limit_options.index(current_limit),
            format_func=lambda x: patient_limit_labels.get(x, str(x)),
            help=limit_help
        )
        st.session_state.patient_limit = patient_limit
        
        # 导出按钮
        can_export = (use_mock or (st.session_state.data_path and Path(st.session_state.data_path).exists())) and selected_concepts and export_path and Path(export_path).exists()
        
        export_btn = "📥 Export Data" if st.session_state.language == 'en' else "📥 导出数据"
        if can_export:
            if st.button(export_btn, type="primary", width="stretch"):
                st.session_state.trigger_export = True
                st.session_state.export_completed = False
                st.rerun()
        else:
            st.button(export_btn, type="primary", width="stretch", disabled=True)
            if not selected_concepts:
                feat_warn = "⚠️ Please select features first" if st.session_state.language == 'en' else "⚠️ 请先选择特征"
                st.caption(feat_warn)
            elif not use_mock and not st.session_state.data_path:
                path_warn = "⚠️ Please set data path first" if st.session_state.language == 'en' else "⚠️ 请先设置数据路径"
                st.caption(path_warn)
        
        # ============ 系统资源信息 ============
        st.markdown("---")
        resources = get_system_resources()
        perf_title = "⚡ Performance" if st.session_state.language == 'en' else "⚡ 性能配置"
        with st.expander(perf_title, expanded=False):
            if st.session_state.language == 'en':
                st.markdown(f"""
                **System Resources:**
                - 🖥️ CPU: {resources['cpu_count']} cores
                - 💾 RAM: {resources['total_memory_gb']} GB total
                - 📊 Available: {resources['available_memory_gb']} GB
                
                **Auto-optimized:**
                - Workers: {resources['recommended_workers']}
                - Backend: {resources['recommended_backend']}
                """)
            else:
                st.markdown(f"""
                **系统资源:**
                - 🖥️ CPU: {resources['cpu_count']} 核心
                - 💾 内存: {resources['total_memory_gb']} GB 总计
                - 📊 可用: {resources['available_memory_gb']} GB
                
                **自动优化配置:**
                - 并行数: {resources['recommended_workers']}
                - 后端: {resources['recommended_backend']}
                """)


def load_from_exported(export_dir: str, max_patients: int = 100, selected_files: list = None):
    """从已导出的数据文件加载数据（限制患者数用于快速预览）。
    
    从宽表中提取每个特征列，使其可以单独选择和可视化。
    
    Args:
        export_dir: 导出目录路径
        max_patients: 最大患者数限制（默认100）
        selected_files: 要加载的文件名列表（不含扩展名），None表示全部加载
    """
    try:
        import time
        load_start = time.time()
        
        export_path = Path(export_dir)
        raw_data = {}  # 原始文件数据
        
        # ID列和时间列，不作为特征
        id_candidates = ['stay_id', 'hadm_id', 'icustay_id', 
                        'patientunitstayid', 'admissionid', 'patientid']
        time_candidates = ['time', 'charttime', 'starttime', 'endtime', 
                          'datetime', 'timestamp', 'index']
        exclude_cols = set(id_candidates + time_candidates)
        
        # 扫描并加载选中的数据文件
        for file in export_path.iterdir():
            file_stem = file.stem
            
            # 如果指定了文件列表，只加载选中的
            if selected_files is not None and file_stem not in selected_files:
                continue
            
            if file.suffix == '.csv':
                df = pd.read_csv(file)
                raw_data[file_stem] = df
            elif file.suffix == '.parquet':
                df = pd.read_parquet(file)
                raw_data[file_stem] = df
            elif file.suffix == '.xlsx':
                df = pd.read_excel(file)
                raw_data[file_stem] = df
        
        if not raw_data:
            lang = st.session_state.get('language', 'en')
            warn_msg = "⚠️ No valid data files found" if lang == 'en' else "⚠️ 未找到有效的数据文件"
            st.warning(warn_msg)
            return
        
        # 从宽表中提取每个特征列作为单独的concept
        data = {}
        
        # 找到ID列和时间列
        id_col_found = 'stay_id'
        time_col_found = 'time'
        
        for file_name, df in raw_data.items():
            if isinstance(df, pd.DataFrame):
                # 找ID列
                for col in id_candidates:
                    if col in df.columns:
                        id_col_found = col
                        break
                # 找时间列
                for col in time_candidates:
                    if col in df.columns:
                        time_col_found = col
                        break
                break
        
        # 从每个宽表中提取特征列
        for file_name, df in raw_data.items():
            if isinstance(df, pd.DataFrame):
                # 获取特征列（排除ID列和时间列）
                feature_cols = [c for c in df.columns if c not in exclude_cols]
                
                # 为每个特征创建单独的DataFrame
                for feat_col in feature_cols:
                    # 保留ID列、时间列和该特征列
                    keep_cols = []
                    if id_col_found in df.columns:
                        keep_cols.append(id_col_found)
                    if time_col_found in df.columns:
                        keep_cols.append(time_col_found)
                    keep_cols.append(feat_col)
                    
                    feat_df = df[keep_cols].copy()
                    # 重命名特征列为标准名
                    feat_df = feat_df.rename(columns={feat_col: feat_col})
                    data[feat_col] = feat_df
        
        # 获取患者列表
        patient_ids = set()
        
        for concept_df in data.values():
            if isinstance(concept_df, pd.DataFrame):
                if id_col_found in concept_df.columns:
                    patient_ids.update(concept_df[id_col_found].unique())
        
        all_patient_count = len(patient_ids)
        
        # 限制患者数用于可视化预览（max_patients=None 表示加载全部）
        if max_patients is None or max_patients <= 0:
            preview_patient_ids = sorted(list(patient_ids))
            is_limited = False
        else:
            preview_patient_ids = sorted(list(patient_ids))[:max_patients]
            is_limited = all_patient_count > max_patients
        
        # 筛选数据只保留限制的患者
        filtered_data = {}
        for concept_name, df in data.items():
            if isinstance(df, pd.DataFrame) and id_col_found in df.columns:
                filtered_df = df[df[id_col_found].isin(preview_patient_ids)]
                if len(filtered_df) > 0:
                    filtered_data[concept_name] = filtered_df
            else:
                filtered_data[concept_name] = df
        
        st.session_state.loaded_concepts = filtered_data
        st.session_state.patient_ids = preview_patient_ids
        st.session_state.all_patient_count = all_patient_count
        st.session_state.id_col = id_col_found
        
        load_elapsed = time.time() - load_start
        
        # 显示提示信息
        lang = st.session_state.get('language', 'en')
        if lang == 'en':
            st.success(f"✅ Loaded {len(filtered_data)} features, {len(preview_patient_ids)}/{all_patient_count} patients ({load_elapsed:.1f}s)")
            if is_limited:
                st.info(f"💡 For better performance, preview is limited to {max_patients} patients. Full data has been exported to disk.")
        else:
            st.success(f"✅ 已加载 {len(filtered_data)} 个特征，{len(preview_patient_ids)}/{all_patient_count} 个患者 ({load_elapsed:.1f}秒)")
            if is_limited:
                st.info(f"💡 为保证流畅性，可视化预览仅加载前 {max_patients} 个患者。完整数据已导出到磁盘，可使用Python/R进行完整分析。")
        
    except Exception as e:
        lang = st.session_state.get('language', 'en')
        err_msg = f"Loading failed: {e}" if lang == 'en' else f"加载失败: {e}"
        st.error(err_msg)


def load_data():
    """Load data with parallel acceleration support - optimized batch loading."""
    lang = st.session_state.get('language', 'en')
    
    if not st.session_state.data_path:
        err_msg = "Please set data path first" if lang == 'en' else "请先设置数据路径"
        st.error(err_msg)
        return
    
    if not st.session_state.selected_concepts:
        err_msg = "Please select at least one concept" if lang == 'en' else "请选择至少一个 Concept"
        st.error(err_msg)
        return
    
    # 显示加载提示
    n_selected = len(st.session_state.selected_concepts)
    if lang == 'en':
        st.info(f"⏳ Loading {n_selected} features in batch mode, please wait...")
        spinner_msg = "Batch loading data, please wait..."
    else:
        st.info(f"⏳ 批量加载 {n_selected} 个特征数据，请稍候...")
        spinner_msg = "正在批量加载数据，请稍候..."
    
    with st.spinner(spinner_msg):
        try:
            # 动态导入以避免循环导入
            from pyricu import load_concepts
            import time
            import os
            
            concepts_list = st.session_state.selected_concepts
            n_concepts = len(concepts_list)
            
            load_start = time.time()
            
            # 🚀 优化：真正的批量加载 - 一次调用加载所有concepts
            # 🚀 性能优化：参照 extract_baseline_features.py 的配置
            # 关键：使用 patient_ids 限制加载的患者范围（默认0表示全量）
            patient_limit = st.session_state.get('patient_limit', 0)
            
            # 获取可用的患者ID列表（如果有缓存就使用缓存）
            patient_ids_filter = None
            if patient_limit and patient_limit > 0:
                # 尝试从 icustays 获取患者ID
                try:
                    data_path = Path(st.session_state.data_path)
                    database = st.session_state.get('database', 'miiv')
                    
                    # 根据数据库类型确定 ID 列名
                    id_col_map = {
                        'miiv': 'stay_id',
                        'eicu': 'patientunitstayid', 
                        'aumc': 'admissionid',
                        'hirid': 'patientid'
                    }
                    id_col = id_col_map.get(database, 'stay_id')
                    
                    # 读取 icustays 获取患者ID
                    icustays_files = ['icustays.parquet', 'patient.parquet', 'admissions.parquet']
                    for f in icustays_files:
                        fp = data_path / f
                        if fp.exists():
                            icustays_df = pd.read_parquet(fp, columns=[id_col] if id_col else None)
                            if id_col in icustays_df.columns:
                                all_patient_ids = icustays_df[id_col].unique().tolist()
                                # 限制患者数量
                                if len(all_patient_ids) > patient_limit:
                                    sample_ids = all_patient_ids[:patient_limit]
                                else:
                                    sample_ids = all_patient_ids
                                patient_ids_filter = {id_col: sample_ids}
                                break
                except Exception:
                    pass  # 无法获取患者ID，不使用过滤
            
            # 🚀 智能并行配置：根据系统资源和患者数量动态调整
            num_patients = len(patient_ids_filter.get(id_col, [])) if patient_ids_filter else None
            parallel_workers, parallel_backend = get_optimal_parallel_config(num_patients, task_type='load')
            
            try:
                # 🔧 逐个加载概念，跳过不可用的（某些概念在特定数据库中没有数据源配置）
                data = {}
                failed_concepts = []
                
                for i, concept in enumerate(concepts_list):
                    try:
                        load_kwargs = {
                            'data_path': st.session_state.data_path,
                            'database': st.session_state.get('database'),
                            'concepts': [concept],
                            'verbose': False,
                            'merge': False,
                            'concept_workers': 1,
                            'parallel_workers': parallel_workers,
                            'parallel_backend': parallel_backend,
                        }
                        if patient_ids_filter:
                            load_kwargs['patient_ids'] = patient_ids_filter
                        
                        result = load_concepts(**load_kwargs)
                        
                        # 处理返回结果（可能是 dict 或 DataFrame）
                        if isinstance(result, dict):
                            for cname, df in result.items():
                                # 🔧 处理各种返回类型（ICUTable, ConceptFrame等）
                                if hasattr(df, 'to_pandas'):
                                    df = df.to_pandas()
                                elif hasattr(df, 'dataframe'):
                                    df = df.dataframe()
                                elif hasattr(df, 'data') and isinstance(df.data, pd.DataFrame):
                                    df = df.data
                                
                                if isinstance(df, pd.DataFrame) and len(df) > 0:
                                    data[cname] = df
                                elif isinstance(df, pd.Series):
                                    data[cname] = df.to_frame().reset_index()
                        elif isinstance(result, pd.DataFrame):
                            # 单概念加载返回 DataFrame
                            if len(result) > 0:
                                data[concept] = result
                    except Exception:
                        failed_concepts.append(concept)
                        continue  # 跳过失败的概念，继续加载其他的
                
                if failed_concepts:
                    skip_msg = f"⚠️ Skipped {len(failed_concepts)} unavailable: {', '.join(failed_concepts[:5])}" if lang == 'en' else f"⚠️ 跳过 {len(failed_concepts)} 个不可用: {', '.join(failed_concepts[:5])}"
                    st.warning(skip_msg)
                    
            except Exception as batch_err:
                # 加载完全失败
                batch_err_msg = f"⚠️ Loading failed: {batch_err}" if lang == 'en' else f"⚠️ 加载失败: {batch_err}"
                st.warning(batch_err_msg)
                data = {}
            
            load_elapsed = time.time() - load_start
            
            if not data:
                warn_msg = "⚠️ Failed to load any data, please check data path and concept selection" if lang == 'en' else "⚠️ 未能加载任何数据，请检查数据路径和 Concept 选择"
                st.warning(warn_msg)
                return
            
            st.session_state.loaded_concepts = data
            
            # 获取患者列表 - 统计所有患者数，但UI选择器限制显示数量
            patient_ids = set()
            id_candidates = ['stay_id', 'hadm_id', 'icustay_id', 
                           'patientunitstayid', 'admissionid', 'patientid']
            
            for concept_df in data.values():
                if isinstance(concept_df, pd.DataFrame):
                    for col in id_candidates:
                        if col in concept_df.columns:
                            patient_ids.update(concept_df[col].unique())
                            break
            
            # 保存完整患者列表用于统计，UI选择器用截断列表
            all_patient_ids = sorted(list(patient_ids))
            st.session_state.all_patient_count = len(all_patient_ids)  # 保存真实患者数
            st.session_state.patient_ids = all_patient_ids[:5000]  # UI选择器限制5000个
            
            if lang == 'en':
                st.success(f"✅ Loaded {len(data)} concepts, {len(all_patient_ids)} patients ({load_elapsed:.1f}s)")
            else:
                st.success(f"✅ 成功加载 {len(data)} 个 Concepts，{len(all_patient_ids)} 个患者 ({load_elapsed:.1f}秒)")
            
        except Exception as e:
            err_msg = f"Loading failed: {e}" if lang == 'en' else f"加载失败: {e}"
            st.error(err_msg)


def load_data_for_preview(max_patients: int = 50):
    """Load limited data for preview visualization (memory-friendly version)."""
    lang = st.session_state.get('language', 'en')
    
    if not st.session_state.data_path:
        err_msg = "Please set data path first" if lang == 'en' else "请先设置数据路径"
        st.error(err_msg)
        return
    
    selected = st.session_state.get('selected_concepts', [])
    if not selected:
        err_msg = "Please select at least one feature" if lang == 'en' else "请选择至少一个特征"
        st.error(err_msg)
        return
    
    try:
        from pyricu import load_concepts
        import time
        
        load_start = time.time()
        data = {}
        
        # 只加载前5个concept作为预览
        preview_concepts = selected[:5]
        
        # 🚀 性能优化：参照 extract_baseline_features.py
        # 预览只加载少量患者（max_patients 个）
        patient_ids_filter = None
        id_col = 'stay_id'
        try:
            data_path = Path(st.session_state.data_path)
            database = st.session_state.get('database', 'miiv')
            id_col_map = {'miiv': 'stay_id', 'eicu': 'patientunitstayid', 'aumc': 'admissionid', 'hirid': 'patientid'}
            id_col = id_col_map.get(database, 'stay_id')
            
            for f in ['icustays.parquet', 'patient.parquet', 'admissions.parquet']:
                fp = data_path / f
                if fp.exists():
                    icustays_df = pd.read_parquet(fp, columns=[id_col] if id_col else None)
                    if id_col in icustays_df.columns:
                        # 预览只需要 max_patients 个患者
                        sample_ids = icustays_df[id_col].unique().tolist()[:max_patients]
                        patient_ids_filter = {id_col: sample_ids}
                        break
        except Exception:
            pass
        
        try:
            load_kwargs = {
                'data_path': st.session_state.data_path,
                'database': st.session_state.get('database'),
                'concepts': preview_concepts,
                'verbose': False,
                'merge': False,
                'concept_workers': 1,
                'parallel_workers': 1,  # 预览数据少，不需要并行
                'parallel_backend': "thread",
            }
            if patient_ids_filter:
                load_kwargs['patient_ids'] = patient_ids_filter
            
            result = load_concepts(**load_kwargs)
            
            if isinstance(result, dict):
                for concept, df in result.items():
                    # 🔧 处理各种返回类型
                    if hasattr(df, 'to_pandas'):
                        df = df.to_pandas()
                    elif hasattr(df, 'dataframe'):
                        df = df.dataframe()
                    elif hasattr(df, 'data') and isinstance(df.data, pd.DataFrame):
                        df = df.data
                    
                    if isinstance(df, pd.DataFrame) and len(df) > 0:
                        data[concept] = df
                    elif isinstance(df, pd.Series):
                        data[concept] = df.to_frame().reset_index()
            elif isinstance(result, pd.DataFrame):
                # 单概念加载返回 DataFrame
                if len(result) > 0:
                    data[preview_concepts[0]] = result
        except Exception:
            # 批量失败，回退到逐个加载
            for concept in preview_concepts:
                try:
                    df = load_concepts(
                        data_path=st.session_state.data_path,
                        database=st.session_state.get('database'),
                        concepts=[concept],
                        verbose=False,
                        merge=True,
                    )
                    if hasattr(df, 'to_pandas'):
                        df = df.to_pandas()
                    elif hasattr(df, 'dataframe'):
                        df = df.dataframe()
                    if isinstance(df, pd.DataFrame) and len(df) > 0:
                        data[concept] = df
                except Exception:
                    pass
        
        if not data:
            lang = st.session_state.get('language', 'en')
            warn_msg = "⚠️ Failed to load any data" if lang == 'en' else "⚠️ 未能加载任何数据"
            st.warning(warn_msg)
            return
        
        # 获取患者列表并限制数量
        patient_ids = set()
        id_col_found = 'stay_id'
        id_candidates = ['stay_id', 'hadm_id', 'icustay_id', 
                       'patientunitstayid', 'admissionid', 'patientid']
        
        for concept_df in data.values():
            if isinstance(concept_df, pd.DataFrame):
                for col in id_candidates:
                    if col in concept_df.columns:
                        patient_ids.update(concept_df[col].unique())
                        id_col_found = col
                        break
        
        all_patient_count = len(patient_ids)
        preview_patient_ids = sorted(list(patient_ids))[:max_patients]
        
        # 筛选数据只保留限制的患者
        filtered_data = {}
        for concept_name, df in data.items():
            if isinstance(df, pd.DataFrame) and id_col_found in df.columns:
                filtered_df = df[df[id_col_found].isin(preview_patient_ids)]
                if len(filtered_df) > 0:
                    filtered_data[concept_name] = filtered_df
            else:
                filtered_data[concept_name] = df
        
        st.session_state.loaded_concepts = filtered_data
        st.session_state.patient_ids = preview_patient_ids
        st.session_state.all_patient_count = all_patient_count
        st.session_state.id_col = id_col_found
        
        load_elapsed = time.time() - load_start
        
        lang = st.session_state.get('language', 'en')
        if lang == 'en':
            st.success(f"✅ Preview data loaded: {len(filtered_data)} features, {len(preview_patient_ids)}/{all_patient_count} patients ({load_elapsed:.1f}s)")
            if all_patient_count > max_patients:
                st.info(f"💡 For better performance, visualization is limited to {max_patients} patients. Export data first for full analysis with Python/R.")
        else:
            st.success(f"✅ 预览数据已加载：{len(filtered_data)} 个特征，{len(preview_patient_ids)}/{all_patient_count} 个患者 ({load_elapsed:.1f}秒)")
            if all_patient_count > max_patients:
                st.info(f"💡 为保证流畅性，可视化仅加载前 {max_patients} 个患者。建议先导出数据，再用Python/R工具进行完整分析。")
        
    except Exception as e:
        lang = st.session_state.get('language', 'en')
        err_msg = f"Loading failed: {e}" if lang == 'en' else f"加载失败: {e}"
        st.error(err_msg)


def render_data_overview():
    """渲染已加载数据的概览页面。"""
    lang = st.session_state.language
    
    # 标题已经在main()中渲染，这里不再重复
    
    # 准备就绪提示
    ready_title = "🎉 Ready!" if lang == 'en' else "🎉 准备就绪！"
    ready_desc = "Data loaded, you can start exploring." if lang == 'en' else "数据已加载，您可以开始探索分析了。"
    st.markdown(f"## {ready_title}")
    st.markdown(ready_desc)
    
    # 状态概览
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        db_display = "🎭 DEMO" if st.session_state.use_mock_data else st.session_state.database.upper()
        db_label = "Database" if lang == 'en' else "数据库"
        st.markdown(f'''
        <div class="metric-card">
            <div class="stat-label">{db_label}</div>
            <div class="stat-number" style="font-size:1.8rem">{db_display}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        n_concepts = len(st.session_state.loaded_concepts)
        feat_label = "Features" if lang == 'en' else "已加载特征"
        st.markdown(f'''
        <div class="metric-card">
            <div class="stat-label">{feat_label}</div>
            <div class="stat-number">{n_concepts}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        # 优先从已加载数据中计算实际患者数
        n_patients = 0
        if st.session_state.loaded_concepts:
            # 从加载的数据中提取实际患者数
            all_ids = set()
            id_col = st.session_state.get('id_col', 'stay_id')
            for df in st.session_state.loaded_concepts.values():
                if isinstance(df, pd.DataFrame) and id_col in df.columns:
                    all_ids.update(df[id_col].unique())
            n_patients = len(all_ids) if all_ids else len(st.session_state.patient_ids)
        else:
            n_patients = len(st.session_state.patient_ids)
        
        pat_label = "Patients" if lang == 'en' else "患者数量"
        st.markdown(f'''
        <div class="metric-card">
            <div class="stat-label">{pat_label}</div>
            <div class="stat-number">{n_patients:,}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        status_label = "Status" if lang == 'en' else "数据状态"
        st.markdown(f'''
        <div class="metric-card">
            <div class="stat-label">{status_label}</div>
            <div class="stat-number" style="color:#28a745">✅ {"Ready" if lang == 'en' else "就绪"}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # 快捷导航
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    start_label = "🚀 Start Analysis" if lang == 'en' else "🚀 开始分析"
    tab_hint = "Select a tab above to explore data:" if lang == 'en' else "选择上方的标签页开始探索数据："
    st.markdown(f"### {start_label}")
    st.markdown(tab_hint)
    
    if lang == 'en':
        features = [
            ("📈", "Time Series", "Interactive time series visualization, single/multi-patient comparison"),
            ("🏥", "Patient View", "Single patient multi-dimensional dashboard"),
            ("📊", "Data Quality", "Missing rate analysis and data distribution statistics"),
        ]
    else:
        features = [
            ("📈", "时序分析", "交互式时间序列可视化，支持单患者/多患者比较"),
            ("🏥", "患者视图", "单患者多维度仪表盘，全景了解患者状态"),
            ("📊", "数据质量", "缺失率分析与数据分布统计"),
        ]
    
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i]:
            st.markdown(f'''
            <div class="feature-card" style="text-align:center;min-height:120px">
                <div style="font-size:2rem">{icon}</div>
                <div style="font-weight:600;color:#4fc3f7">{title}</div>
                <div style="font-size:0.85rem;color:#aaa">{desc}</div>
            </div>
            ''', unsafe_allow_html=True)
    
    # 数据摘要
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    summary_label = "📋 Data Summary" if lang == 'en' else "📋 数据摘要"
    st.markdown(f"### {summary_label}")
    
    concept_stats = []
    for name, df in st.session_state.loaded_concepts.items():
        if isinstance(df, pd.DataFrame):
            n_records = len(df)
            n_pts = df[st.session_state.id_col].nunique() if st.session_state.id_col in df.columns else 0
            concept_stats.append({
                'Feature' if lang == 'en' else 'Concept': name,
                'Records' if lang == 'en' else '记录数': f"{n_records:,}",
                'Patients' if lang == 'en' else '患者数': n_pts,
            })
    
    if concept_stats:
        st.dataframe(pd.DataFrame(concept_stats), use_container_width=True, hide_index=True)


def render_home():
    """渲染首页 - 引导式教程，根据用户进度动态显示。"""
    lang = st.session_state.language
    
    # 如果已加载数据，直接显示数据概览
    if len(st.session_state.loaded_concepts) > 0:
        render_data_overview()
        return
    
    # 标题已经在main()中渲染，这里不再重复
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # 获取当前模式 - 使用app_mode（'extract'或'viz'）
    current_mode = st.session_state.get('app_mode', 'extract')
    is_viz_mode = current_mode == 'viz'
    
    if is_viz_mode:
        # ============ 快速可视化模式教程 ============
        render_home_viz_mode(lang)
    else:
        # ============ 数据提取导出模式教程 ============
        render_home_extract_mode(lang)


def render_home_viz_mode(lang):
    """渲染快速可视化模式的首页教程。"""
    # 进度指示器
    col1, col2 = st.columns(2)
    
    # 检查状态
    viz_dir = st.session_state.get('viz_data_dir', '')
    has_files = False
    if viz_dir and Path(viz_dir).exists():
        files = list(Path(viz_dir).glob('*.csv')) + list(Path(viz_dir).glob('*.parquet')) + list(Path(viz_dir).glob('*.xlsx'))
        has_files = len(files) > 0
    
    step1_done = has_files
    step2_done = len(st.session_state.loaded_concepts) > 0
    
    done_text = "✅ Done" if lang == 'en' else "✅ 完成"
    in_progress_text = "🔵 In Progress" if lang == 'en' else "🔵 进行中"
    waiting_text = "⏳ Waiting" if lang == 'en' else "⏳ 等待"
    
    with col1:
        status = done_text if step1_done else in_progress_text
        color = "#28a745" if step1_done else "#ffc107"
        step_label = "Step 1" if lang == 'en' else "步骤 1"
        step_desc = "Select Data Directory" if lang == 'en' else "选择数据目录"
        st.markdown(f'''
        <div class="metric-card" style="border-left: 4px solid {color}">
            <div class="stat-label">{step_label}</div>
            <div style="font-weight:600">{step_desc}</div>
            <div style="color:{color};font-size:0.9rem">{status}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        if step1_done:
            status = done_text if step2_done else in_progress_text
            color = "#28a745" if step2_done else "#ffc107"
        else:
            status = waiting_text
            color = "#6c757d"
        step_label = "Step 2" if lang == 'en' else "步骤 2"
        step_desc = "Load & Visualize" if lang == 'en' else "加载并可视化"
        st.markdown(f'''
        <div class="metric-card" style="border-left: 4px solid {color}">
            <div class="stat-label">{step_label}</div>
            <div style="font-weight:600">{step_desc}</div>
            <div style="color:{color};font-size:0.9rem">{status}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # 教程内容
    if not step1_done:
        task_header = "📍 Current Task: Select Data Directory" if lang == 'en' else "📍 当前任务：选择数据目录"
        st.markdown(f"## {task_header}")
        
        if lang == 'en':
            st.markdown('''
            <div class="highlight-card">
                <h4>👈 Please specify the data directory in the left sidebar</h4>
                <p style="color:#ccc; margin-bottom:12px">
                    Quick Visualization mode loads data from previously exported files:
                </p>
                <ul style="color:#bbb; font-size:0.9rem;">
                    <li>Enter the path to the directory containing exported data files</li>
                    <li>Supported formats: <b>CSV, Parquet, Excel</b></li>
                    <li>If you haven't exported data yet, switch to "Data Extraction" mode first</li>
                </ul>
                <p style="color:#ffa500; margin-top:12px;">
                    <b>💡 Tip:</b> Default path is <code>~/pyricu_export/miiv</code>
                </p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="highlight-card">
                <h4>👈 请在左侧边栏指定数据目录</h4>
                <p style="color:#ccc; margin-bottom:12px">
                    快速可视化模式从已导出的文件加载数据：
                </p>
                <ul style="color:#bbb; font-size:0.9rem;">
                    <li>输入包含已导出数据文件的目录路径</li>
                    <li>支持的格式：<b>CSV、Parquet、Excel</b></li>
                    <li>如果您还没有导出过数据，请先切换到「数据提取导出」模式</li>
                </ul>
                <p style="color:#ffa500; margin-top:12px;">
                    <b>💡 提示：</b> 默认路径是 <code>~/pyricu_export/miiv</code>
                </p>
            </div>
            ''', unsafe_allow_html=True)
    else:
        task_header = "📍 Current Task: Load Data" if lang == 'en' else "📍 当前任务：加载数据"
        st.markdown(f"## {task_header}")
        
        if lang == 'en':
            st.markdown('''
            <div class="highlight-card">
                <h4>👈 Click "Load Data" in the left sidebar</h4>
                <p style="color:#ccc; margin-bottom:12px">
                    Data files found! You can now:
                </p>
                <ul style="color:#bbb; font-size:0.9rem;">
                    <li>Select specific tables to load (recommended ≤ 3 for best performance)</li>
                    <li>Click <b>"Load Data"</b> button to load into memory</li>
                    <li>After loading, use the tabs above to explore and visualize</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="highlight-card">
                <h4>👈 在左侧边栏点击「加载数据」</h4>
                <p style="color:#ccc; margin-bottom:12px">
                    已发现数据文件！您现在可以：
                </p>
                <ul style="color:#bbb; font-size:0.9rem;">
                    <li>选择要加载的表格（建议不超过3个以保证流畅性）</li>
                    <li>点击 <b>「加载数据」</b> 按钮将数据加载到内存</li>
                    <li>加载完成后，使用上方的标签页进行探索和可视化</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)
    
    # 功能预览
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    preview_title = "🎯 What You Can Do After Loading" if lang == 'en' else "🎯 加载后可用功能"
    st.markdown(f"### {preview_title}")
    
    if lang == 'en':
        features = [
            ("📈", "Time Series", "Interactive time series visualization"),
            ("🏥", "Patient View", "Single patient dashboard"),
            ("📊", "Data Quality", "Missing rate & distribution analysis"),
        ]
    else:
        features = [
            ("📈", "时序分析", "交互式时间序列可视化"),
            ("🏥", "患者视图", "单患者多维仪表盘"),
            ("📊", "数据质量", "缺失率与分布分析"),
        ]
    
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i]:
            st.markdown(f'''
            <div class="feature-card" style="text-align:center;min-height:100px">
                <div style="font-size:2rem">{icon}</div>
                <div style="font-weight:600;color:#4fc3f7">{title}</div>
                <div style="font-size:0.85rem;color:#aaa">{desc}</div>
            </div>
            ''', unsafe_allow_html=True)


def render_home_extract_mode(lang):
    """渲染数据提取导出模式的首页教程。"""
    
    # ============ 固定导航栏 - 使用sticky定位 ============
    nav_labels = [
        ("📋 " + ("Progress" if lang == 'en' else "进度"), "progress"),
        ("📍 " + ("Guide" if lang == 'en' else "引导"), "guide"),
        ("📖 " + ("Dictionary" if lang == 'en' else "数据字典"), "dictionary"),
    ]
    
    # 使用sticky定位的导航栏，更现代的渐变色
    nav_links = " ".join([f'<a href="#{anchor}" style="color:white;text-decoration:none;padding:10px 24px;background:rgba(255,255,255,0.2);border-radius:25px;font-size:1rem;font-weight:600;margin:0 8px;transition:all 0.3s;backdrop-filter:blur(10px);" onmouseover="this.style.background=\'rgba(255,255,255,0.35)\'" onmouseout="this.style.background=\'rgba(255,255,255,0.2)\'">{label}</a>' for label, anchor in nav_labels])
    st.markdown(f'''
    <div style="
        position: sticky;
        top: 0;
        z-index: 999;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 14px 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(102,126,234,0.4);
    ">{nav_links}</div>
    ''', unsafe_allow_html=True)
    
    # 计算当前步骤完成状态
    step1_done = st.session_state.use_mock_data or (st.session_state.data_path and Path(st.session_state.data_path).exists())
    step2_done = len(st.session_state.get('selected_concepts', [])) > 0
    step3_done = st.session_state.get('export_completed', False) or len(st.session_state.loaded_concepts) > 0
    
    # ============ 进度指示器 ============
    # 添加锚点和大标题
    st.markdown('<div id="progress"></div>', unsafe_allow_html=True)
    progress_title = "📋 Progress" if lang == 'en' else "📋 进度"
    st.markdown(f'<h2 style="background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;border-bottom:3px solid #667eea;padding-bottom:10px;margin-top:10px;font-size:1.6rem;">{progress_title}</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    # 状态文本
    done_text = "✅ Done" if lang == 'en' else "✅ 完成"
    in_progress_text = "🔵 In Progress" if lang == 'en' else "🔵 进行中"
    waiting_text = "⏳ Waiting" if lang == 'en' else "⏳ 等待"
    
    with col1:
        status = done_text if step1_done else in_progress_text
        color = "#28a745" if step1_done else "#ffc107"
        step_label = "Step 1" if lang == 'en' else "步骤 1"
        step_desc = "Data Source" if lang == 'en' else "配置数据源"
        st.markdown(f'''
        <div class="metric-card" style="border-left: 4px solid {color}">
            <div class="stat-label">{step_label}</div>
            <div style="font-weight:600">{step_desc}</div>
            <div style="color:{color};font-size:0.9rem">{status}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        if step1_done:
            status = done_text if step2_done else in_progress_text
            color = "#28a745" if step2_done else "#ffc107"
        else:
            status = waiting_text
            color = "#6c757d"
        step_label = "Step 2" if lang == 'en' else "步骤 2"
        step_desc = "Select Features" if lang == 'en' else "选择特征"
        st.markdown(f'''
        <div class="metric-card" style="border-left: 4px solid {color}">
            <div class="stat-label">{step_label}</div>
            <div style="font-weight:600">{step_desc}</div>
            <div style="color:{color};font-size:0.9rem">{status}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        if step1_done and step2_done:
            status = done_text if step3_done else in_progress_text
            color = "#28a745" if step3_done else "#ffc107"
        else:
            status = waiting_text
            color = "#6c757d"
        step_label = "Step 3" if lang == 'en' else "步骤 3"
        step_desc = "Export/Preview" if lang == 'en' else "导出/预览"
        st.markdown(f'''
        <div class="metric-card" style="border-left: 4px solid {color}">
            <div class="stat-label">{step_label}</div>
            <div style="font-weight:600">{step_desc}</div>
            <div style="color:{color};font-size:0.9rem">{status}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # ============ 动态引导内容 ============
    # 添加引导锚点和大标题
    st.markdown('<div id="guide"></div>', unsafe_allow_html=True)
    guide_title = "📍 Guide" if lang == 'en' else "📍 引导"
    st.markdown(f'<h2 style="background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;border-bottom:3px solid #667eea;padding-bottom:10px;margin-top:10px;font-size:1.6rem;">{guide_title}</h2>', unsafe_allow_html=True)
    
    if not step1_done:
        # 步骤1引导：配置数据源
        task_hint = "👉 Configure Data Source" if lang == 'en' else "👉 配置数据源"
        st.markdown(f"**{task_hint}**")
        
        if lang == 'en':
            st.markdown('''
            <div class="highlight-card">
                <h4>👈 Please configure data source in the left sidebar</h4>
                <p><b>🎭 Demo Mode</b> - No data needed, auto-generates simulated ICU data for learning</p>
                <p><b>📊 Real Data</b> - Supports MIMIC-IV, eICU, AUMC, HiRID (local processing, secure)</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="highlight-card">
                <h4>👈 请在左侧边栏完成数据源配置</h4>
                <p><b>🎭 演示模式</b> - 无需数据，自动生成模拟ICU数据，适合学习体验</p>
                <p><b>📊 真实数据</b> - 支持MIMIC-IV、eICU、AUMC、HiRID（本地处理，安全可靠）</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # 快速开始按钮
        quick_start_title = "⚡ Quick Start" if lang == 'en' else "⚡ 快速开始"
        st.markdown(f"### {quick_start_title}")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            demo_btn = "🎭 Enable Demo Mode" if lang == 'en' else "🎭 一键启用演示模式"
            if st.button(demo_btn, type="primary", width="stretch", key="quick_demo"):
                st.session_state.use_mock_data = True
                st.session_state.database = 'mock'
                success_msg = "✅ Demo mode enabled! Please continue to select features." if lang == 'en' else "✅ 演示模式已启用！请继续选择特征。"
                st.success(success_msg)
                st.rerun()
        
    elif not step2_done:
        # 步骤2引导：选择特征
        task_hint = "👉 Select Analysis Features" if lang == 'en' else "👉 选择分析特征"
        st.markdown(f"**{task_hint}**")
        
        # 显示当前数据源状态
        if st.session_state.use_mock_data:
            source_info = "🎭 **Demo Mode**" if lang == 'en' else "🎭 **演示模式**"
        else:
            source_info = f"📊 **Real Data** - `{st.session_state.data_path}`" if lang == 'en' else f"📊 **真实数据** - `{st.session_state.data_path}`"
        source_label = "**Current Data Source**" if lang == 'en' else "**当前数据源**"
        st.markdown(f"{source_label}: {source_info}")
        
        if lang == 'en':
            st.markdown('''
            <div class="highlight-card">
                <h4>👈 Please select features to analyze in the left sidebar</h4>
                <p style="color:#ccc; margin-bottom:12px">
                    PyRICU provides 130+ ICU features, organized by category. You can:
                </p>
                <ul style="color:#bbb; font-size:0.9rem;">
                    <li><b>Select by group</b>: Expand a group, select entire group or individual features</li>
                    <li><b>Use presets</b>: Click "SOFA-2 Features" or "Common Features" for quick selection</li>
                    <li><b>Custom combination</b>: Combine freely based on research needs</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="highlight-card">
                <h4>👈 请在左侧边栏选择要分析的特征</h4>
                <p style="color:#ccc; margin-bottom:12px">
                    PyRICU 提供 130+ ICU 特征，已按类别分组。您可以：
                </p>
                <ul style="color:#bbb; font-size:0.9rem;">
                    <li><b>按分组选择</b>：展开某个分组，选择整组或单个特征</li>
                    <li><b>使用预设</b>：点击「SOFA-2特征」或「常用特征」快速选择</li>
                    <li><b>自定义组合</b>：根据研究需求自由组合</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)
        
        # ⭐ SOFA-2 亮点介绍
        sofa_title = "🌟 Recommended Feature" if lang == 'en' else "🌟 推荐特色功能"
        st.markdown(f"### {sofa_title}")
        if lang == 'en':
            st.markdown('''
            <div class="feature-card" style="border-left:4px solid #ffa500">
                <h4>SOFA-2 Scoring System (October 2025 JAMA New Standard)</h4>
                <p style="color:#ccc; margin-bottom:12px">
                    PyRICU is the <b>first open-source ICU data analysis toolkit implementing SOFA-2</b>.
                    Based on the latest consensus published in JAMA Network Open in October 2025.
                </p>
                <div style="display:flex; gap:20px; flex-wrap:wrap;">
                    <div style="flex:1; min-width:200px;">
                        <b style="color:#ffa500">📊 SOFA-2 Key Improvements:</b>
                        <ul style="color:#bbb; font-size:0.9rem; margin-top:6px;">
                            <li>Respiratory: Added ECMO, HFNC, NIV support recognition</li>
                            <li>Cardiovascular: Integrated norepinephrine + epinephrine dosing</li>
                            <li>Renal: Added RRT automatic 4-point rule</li>
                            <li>Neurological: Added delirium treatment recognition</li>
                        </ul>
                    </div>
                    <div style="flex:1; min-width:200px;">
                        <b style="color:#ffa500">💡 Quick Start:</b>
                        <ul style="color:#bbb; font-size:0.9rem; margin-top:6px;">
                            <li>Click "🔥 SOFA-2 Features" preset on the left</li>
                            <li>Auto-selects all SOFA-2 related features</li>
                            <li>Features marked with ⭐ are SOFA-2 exclusive</li>
                        </ul>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="feature-card" style="border-left:4px solid #ffa500">
                <h4>SOFA-2 评分系统（2025年10月 JAMA 新标准）</h4>
                <p style="color:#ccc; margin-bottom:12px">
                    PyRICU 是<b>首个实现 SOFA-2 评分</b>的开源 ICU 数据分析工具包。
                    基于 2025 年 JAMA Network Open 发布的最新共识进行了重大更新。
                </p>
                <div style="display:flex; gap:20px; flex-wrap:wrap;">
                    <div style="flex:1; min-width:200px;">
                        <b style="color:#ffa500">📊 SOFA-2 主要改进：</b>
                        <ul style="color:#bbb; font-size:0.9rem; margin-top:6px;">
                            <li>呼吸评分：新增 ECMO、HFNC、NIV 支持识别</li>
                            <li>心血管评分：整合去甲肾+肾上腺素剂量</li>
                            <li>肾脏评分：新增 RRT 自动4分规则</li>
                            <li>神经评分：新增谵妄治疗识别</li>
                        </ul>
                    </div>
                    <div style="flex:1; min-width:200px;">
                        <b style="color:#ffa500">💡 快速体验：</b>
                        <ul style="color:#bbb; font-size:0.9rem; margin-top:6px;">
                            <li>点击左侧「🔥 SOFA-2 特征」预设</li>
                            <li>自动选择所有 SOFA-2 相关特征</li>
                            <li>标有 ⭐ 的是 SOFA-2 专属特征</li>
                        </ul>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
    elif not step3_done:
        # 步骤3引导：导出或预览
        task_hint = "👉 Export Data or Load Preview" if lang == 'en' else "👉 导出数据或加载预览"
        st.markdown(f"**{task_hint}**")
        
        # 显示当前选择摘要
        selected = st.session_state.get('selected_concepts', [])
        if st.session_state.use_mock_data:
            source_info = "🎭 Demo Mode" if lang == 'en' else "🎭 演示模式"
        else:
            source_info = f"📊 {st.session_state.data_path}"
        
        source_label = "Data Source" if lang == 'en' else "数据源"
        feat_label = "Selected Features" if lang == 'en' else "已选特征"
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'''
            <div class="metric-card">
                <div class="stat-label">{source_label}</div>
                <div style="font-weight:600">{source_info}</div>
            </div>
            ''', unsafe_allow_html=True)
        with col2:
            st.markdown(f'''
            <div class="metric-card">
                <div class="stat-label">{feat_label}</div>
                <div class="stat-number">{len(selected)}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        if lang == 'en':
            st.markdown('''
            <div class="highlight-card">
                <h4>👈 Please select next action in the left sidebar</h4>
                <p style="color:#ccc; margin-bottom:12px">
                    You have completed data source and feature configuration. Now you can:
                </p>
                <div style="display:flex; gap:20px; flex-wrap:wrap;">
                    <div style="flex:1; min-width:250px;">
                        <b style="color:#28a745">📥 Direct Export (Recommended for low-memory devices)</b>
                        <ul style="color:#bbb; font-size:0.9rem; margin-top:6px;">
                            <li>Select export format (CSV/Parquet/Excel)</li>
                            <li>Click "Export Data" to save directly to disk</li>
                            <li>Uses no memory, suitable for large datasets</li>
                        </ul>
                    </div>
                    <div style="flex:1; min-width:250px;">
                        <b style="color:#4fc3f7">🔍 Load Preview Data</b>
                        <ul style="color:#bbb; font-size:0.9rem; margin-top:6px;">
                            <li>Load small amount of data to memory</li>
                            <li>Use interactive visualization analysis</li>
                            <li>Suitable for data exploration and quality checks</li>
                        </ul>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="highlight-card">
                <h4>👈 请在左侧边栏选择下一步操作</h4>
                <p style="color:#ccc; margin-bottom:12px">
                    您已完成数据源和特征配置，现在可以：
                </p>
                <div style="display:flex; gap:20px; flex-wrap:wrap;">
                    <div style="flex:1; min-width:250px;">
                        <b style="color:#28a745">📥 直接导出（推荐低内存设备）</b>
                        <ul style="color:#bbb; font-size:0.9rem; margin-top:6px;">
                            <li>选择导出格式（CSV/Parquet/Excel）</li>
                            <li>点击「导出数据」直接保存到磁盘</li>
                            <li>不占用内存，适合大数据集</li>
                        </ul>
                    </div>
                    <div style="flex:1; min-width:250px;">
                        <b style="color:#4fc3f7">🔍 加载预览数据</b>
                        <ul style="color:#bbb; font-size:0.9rem; margin-top:6px;">
                            <li>加载少量数据到内存</li>
                            <li>使用交互式可视化分析</li>
                            <li>适合数据探索和质量检查</li>
                        </ul>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # 快速操作
        quick_action_title = "⚡ Quick Actions" if lang == 'en' else "⚡ 快速操作"
        st.markdown(f"### {quick_action_title}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.use_mock_data:
                gen_btn = "🔍 Generate Mock Data & Preview" if lang == 'en' else "🔍 生成模拟数据并预览"
                if st.button(gen_btn, type="primary", width="stretch"):
                    spin_msg = "Generating..." if lang == 'en' else "生成中..."
                    with st.spinner(spin_msg):
                        params = st.session_state.get('mock_params', {'n_patients': 10, 'hours': 72})
                        data, patient_ids = generate_mock_data(**params)
                        st.session_state.loaded_concepts = data
                        st.session_state.patient_ids = patient_ids
                        st.session_state.id_col = 'stay_id'
                        success_msg = "✅ Mock data generated!" if lang == 'en' else "✅ 模拟数据已生成！"
                        st.success(success_msg)
                    st.rerun()
            else:
                load_btn = "🔍 Load Preview Data" if lang == 'en' else "🔍 加载预览数据"
                if st.button(load_btn, type="secondary", width="stretch"):
                    load_data_for_preview()
                    st.rerun()
        
        with col2:
            hint_msg = "_Or switch to 'Data Export' tab for full export_" if lang == 'en' else "_或切换到「数据导出」标签页进行完整导出_"
            st.markdown(hint_msg)
    
    else:
        # 所有步骤完成 - 显示数据摘要和导航
        ready_title = "🎉 Ready!" if lang == 'en' else "🎉 准备就绪！"
        ready_desc = "Data loaded, you can start exploring and analyzing." if lang == 'en' else "数据已加载，您可以开始探索分析了。"
        st.success(f"**{ready_title}** {ready_desc}")
        
        # 状态概览
        col1, col2, col3, col4 = st.columns(4)
        
        db_label = "Database" if lang == 'en' else "数据库"
        feat_label = "Loaded Features" if lang == 'en' else "已加载特征"
        patient_label = "Patients" if lang == 'en' else "患者数量"
        status_label = "Status" if lang == 'en' else "数据状态"
        ready_status = "✅ Ready" if lang == 'en' else "✅ 就绪"
        
        with col1:
            db_display = "🎭 DEMO" if st.session_state.use_mock_data else st.session_state.database.upper()
            st.markdown(f'''
            <div class="metric-card">
                <div class="stat-label">{db_label}</div>
                <div class="stat-number" style="font-size:1.8rem">{db_display}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            n_concepts = len(st.session_state.loaded_concepts)
            st.markdown(f'''
            <div class="metric-card">
                <div class="stat-label">{feat_label}</div>
                <div class="stat-number">{n_concepts}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            # 优先从已加载数据中计算实际患者数
            n_patients = 0
            if st.session_state.loaded_concepts:
                all_ids = set()
                id_col = st.session_state.get('id_col', 'stay_id')
                for df in st.session_state.loaded_concepts.values():
                    if isinstance(df, pd.DataFrame) and id_col in df.columns:
                        all_ids.update(df[id_col].unique())
                n_patients = len(all_ids) if all_ids else len(st.session_state.patient_ids)
            else:
                n_patients = len(st.session_state.patient_ids)
            
            st.markdown(f'''
            <div class="metric-card">
                <div class="stat-label">{patient_label}</div>
                <div class="stat-number">{n_patients:,}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'''
            <div class="metric-card">
                <div class="stat-label">{status_label}</div>
                <div class="stat-number" style="color:#28a745">{ready_status}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        # 快捷导航
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        start_title = "### 🚀 Start Analysis" if lang == 'en' else "### 🚀 开始分析"
        st.markdown(start_title)
        start_desc = "Select a tab above to start exploring data:" if lang == 'en' else "选择上方的标签页开始探索数据："
        st.markdown(start_desc)
        
        if lang == 'en':
            features = [
                ("📈", "Time Series", "Interactive time series visualization with single/multi-patient comparison"),
                ("🏥", "Patient View", "Multi-dimensional patient dashboard for comprehensive status overview"),
                ("📊", "Data Quality", "Missing rate analysis and data distribution statistics"),
            ]
        else:
            features = [
                ("📈", "时序分析", "交互式时间序列可视化，支持单患者/多患者比较"),
                ("🏥", "患者视图", "单患者多维度仪表盘，全景了解患者状态"),
                ("📊", "数据质量", "缺失率分析与数据分布统计"),
            ]
        
        cols = st.columns(3)
        for i, (icon, title, desc) in enumerate(features):
            with cols[i]:
                st.markdown(f'''
                <div class="feature-card" style="text-align:center;min-height:120px">
                    <div style="font-size:2rem">{icon}</div>
                    <div style="font-weight:600;color:#4fc3f7">{title}</div>
                    <div style="font-size:0.85rem;color:#aaa">{desc}</div>
                </div>
                ''', unsafe_allow_html=True)
        
        # 数据摘要
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        summary_title = "### 📋 Data Summary" if lang == 'en' else "### 📋 数据摘要"
        st.markdown(summary_title)
        
        records_col = "Records" if lang == 'en' else "记录数"
        patients_col = "Patients" if lang == 'en' else "患者数"
        
        concept_stats = []
        for name, df in st.session_state.loaded_concepts.items():
            if isinstance(df, pd.DataFrame):
                n_records = len(df)
                n_pts = df[st.session_state.id_col].nunique() if st.session_state.id_col in df.columns else 0
                concept_stats.append({
                    'Concept': name,
                    records_col: f"{n_records:,}",
                    patients_col: n_pts,
                })
        
        if concept_stats:
            stats_df = pd.DataFrame(concept_stats)
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
        
        # 快捷操作
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            regen_label = "🔄 Regenerate Data" if lang == 'en' else "🔄 重新生成数据"
            regen_spinner = "Regenerating..." if lang == 'en' else "重新生成中..."
            if st.button(regen_label, width="stretch", key="regen_home"):
                with st.spinner(regen_spinner):
                    data, patient_ids = generate_mock_data(n_patients=10, hours=72)
                    st.session_state.loaded_concepts = data
                    st.session_state.patient_ids = patient_ids
                st.rerun()
        
        with col2:
            clear_label = "🗑️ Clear Data" if lang == 'en' else "🗑️ 清空数据"
            if st.button(clear_label, width="stretch", key="clear_home"):
                st.session_state.loaded_concepts = {}
                st.session_state.patient_ids = []
                st.session_state.export_completed = False
                st.rerun()
    
    # ============ 数据字典展示 ============
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    # 添加字典锚点和大标题
    st.markdown('<div id="dictionary"></div>', unsafe_allow_html=True)
    dict_header = "📖 Data Dictionary" if lang == 'en' else "📖 数据字典"
    st.markdown(f'<h2 style="background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;border-bottom:3px solid #667eea;padding-bottom:10px;margin-top:10px;font-size:1.6rem;">{dict_header}</h2>', unsafe_allow_html=True)
    render_home_data_dictionary(lang)
    
    # 页脚信息
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if lang == 'en':
        st.markdown('''
        <div style="text-align:center;color:#aaa;font-size:0.85rem">
            <p>🏥 PyRICU - Python Re-Implementation of RICU | 
            📦 <a href="https://github.com/your-repo/pyricu" style="color:#4fc3f7">GitHub</a> | 
            📖 <a href="#" style="color:#4fc3f7">Docs</a></p>
            <p>All data processing is done locally, no data is uploaded to any server 🔒</p>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div style="text-align:center;color:#aaa;font-size:0.85rem">
            <p>🏥 PyRICU - Python Re-Implementation of RICU | 
            📦 <a href="https://github.com/your-repo/pyricu" style="color:#4fc3f7">GitHub</a> | 
            📖 <a href="#" style="color:#4fc3f7">文档</a></p>
            <p>所有数据处理均在本地完成，不会上传到任何服务器 🔒</p>
        </div>
        ''', unsafe_allow_html=True)


def render_home_data_dictionary(lang):
    """在首页渲染完整的数据字典。"""
    dict_title = "📖 Complete Data Dictionary" if lang == 'en' else "📖 完整数据字典"
    
    with st.expander(dict_title, expanded=True):
        dict_intro = "PyRICU provides 130+ ICU clinical features, organized by category. Click each category to view detailed descriptions." if lang == 'en' else "PyRICU 提供 130+ ICU 临床特征，按类别组织。点击各类别查看详细说明。"
        st.caption(dict_intro)
        
        # 获取分组
        concept_groups = get_concept_groups()
        
        # 使用 tabs 展示各分类
        group_names = list(concept_groups.keys())
        tabs = st.tabs(group_names[:8])  # 前8个分类
        
        for i, tab in enumerate(tabs):
            with tab:
                group_name = group_names[i]
                concepts = concept_groups[group_name]
                _render_home_dict_table(concepts, lang)
        
        # 其余分类用expander
        if len(group_names) > 8:
            more_title = "📂 More Categories" if lang == 'en' else "📂 更多类别"
            st.markdown(f"#### {more_title}")
            for group_name in group_names[8:]:
                feat_text = "features" if lang == 'en' else "个特征"
                with st.expander(f"{group_name} ({len(concept_groups[group_name])} {feat_text})"):
                    _render_home_dict_table(concept_groups[group_name], lang)


def _render_home_dict_table(concepts, lang):
    """为首页数据字典渲染表格。"""
    rows = []
    for concept in concepts:
        if concept in CONCEPT_DICTIONARY:
            eng_name, chn_name, unit = CONCEPT_DICTIONARY[concept]
            # 获取详细描述
            if concept in CONCEPT_DESCRIPTIONS:
                eng_desc, chn_desc = CONCEPT_DESCRIPTIONS[concept]
            else:
                eng_desc, chn_desc = eng_name, chn_name  # 用名称作为默认描述
            
            if lang == 'en':
                rows.append({
                    'Code': concept,
                    'Full Name': eng_name,
                    'Description': eng_desc,
                    'Unit': unit if unit else '-'
                })
            else:
                rows.append({
                    '代码': concept,
                    '全称': eng_name,
                    '说明': chn_desc,
                    '单位': unit if unit else '-'
                })
    
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True, height=300)


def render_timeseries_page():
    """渲染时序分析页面。"""
    lang = st.session_state.get('language', 'en')
    
    page_title = "📈 Time Series Analysis" if lang == 'en' else "📈 时序数据分析"
    st.markdown(f"## {page_title}")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    if len(st.session_state.loaded_concepts) == 0:
        if lang == 'en':
            st.markdown('''
            <div class="info-box">
                <strong>👈 Please load data from the sidebar first</strong><br>
                💡 Tip: Click "Enable Demo Mode" on homepage for quick start
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="info-box">
                <strong>👈 请先在侧边栏加载数据</strong><br>
                💡 提示：点击首页「一键体验演示模式」快速开始
            </div>
            ''', unsafe_allow_html=True)
        return
    
    # Concept 选择区域
    available_concepts = list(st.session_state.loaded_concepts.keys())
    
    # 分析模式选择
    mode_label = "Analysis Mode" if lang == 'en' else "分析模式"
    mode_single = "Single Patient" if lang == 'en' else "单患者分析"
    mode_multi = "Multi-Patient Comparison" if lang == 'en' else "多患者比较"
    analysis_mode = st.radio(
        mode_label,
        options=[mode_single, mode_multi],
        horizontal=True,
        key="ts_mode"
    )
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    if analysis_mode == mode_single:
        # 顶部控制面板
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            concept_label = "📋 Select Concept" if lang == 'en' else "📋 选择 Concept"
            concept_help = "Select data type to visualize" if lang == 'en' else "选择要可视化的数据类型"
            selected_concept = st.selectbox(
                concept_label,
                options=available_concepts,
                key="ts_concept",
                help=concept_help
            )
        
        with col2:
            if st.session_state.patient_ids:
                patient_label = "👤 Select Patient" if lang == 'en' else "👤 选择患者"
                patient_id = st.selectbox(
                    patient_label,
                    options=st.session_state.patient_ids[:100],
                    key="ts_patient"
                )
            else:
                patient_id = None
                no_patient_msg = "No patients found" if lang == 'en' else "未找到患者"
                st.warning(no_patient_msg)
        
        with col3:
            chart_label = "📊 Chart Type" if lang == 'en' else "📊 图表类型"
            line_opt = "Line Chart" if lang == 'en' else "折线图"
            scatter_opt = "Scatter Plot" if lang == 'en' else "散点图"
            area_opt = "Area Chart" if lang == 'en' else "面积图"
            chart_type = st.selectbox(
                chart_label,
                options=[line_opt, scatter_opt, area_opt],
                key="ts_chart_type"
            )
        
        with col4:
            show_stats_label = "Show Statistics" if lang == 'en' else "显示统计"
            show_stats = st.checkbox(show_stats_label, value=True, key="ts_show_stats")
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # 主图表区域
        if selected_concept and patient_id:
            df = st.session_state.loaded_concepts[selected_concept]
            
            # 确保是 DataFrame
            if not isinstance(df, pd.DataFrame):
                format_warn = f"Data format not supported: {type(df).__name__}" if lang == 'en' else f"数据格式不支持: {type(df).__name__}"
                st.warning(format_warn)
                return
            
            # 过滤数据
            id_col = st.session_state.id_col
            if id_col and id_col in df.columns:
                patient_df = df[df[id_col] == patient_id].copy()
            else:
                patient_df = df.copy()
            
            # 显示图表
            if len(patient_df) > 0:
                try:
                    import plotly.express as px
                    import plotly.graph_objects as go
                    
                    # 确定数值列
                    numeric_cols = patient_df.select_dtypes(include=['number']).columns
                    # 排除ID列和所有可能的时间列
                    exclude_cols = ['stay_id', 'hadm_id', 'icustay_id', 'index', 'time', 
                                   'charttime', 'starttime', 'endtime', 'datetime', 'timestamp',
                                   'patientunitstayid', 'admissionid', 'patientid']
                    value_cols = [c for c in numeric_cols if c not in exclude_cols]
                    
                    # 检测时间列 - 支持多种命名
                    time_candidates = ['time', 'charttime', 'starttime', 'endtime', 'datetime', 'timestamp']
                    time_col = None
                    for tc in time_candidates:
                        if tc in patient_df.columns:
                            time_col = tc
                            break
                    
                    if value_cols:
                        value_col = value_cols[0]
                        
                        if time_col:
                            # 根据图表类型创建图表
                            line_type = "Line Chart" if lang == 'en' else "折线图"
                            scatter_type = "Scatter Plot" if lang == 'en' else "散点图"
                            patient_label = "Patient" if lang == 'en' else "患者"
                            chart_title = f"📈 {selected_concept.upper()} - {patient_label} {patient_id}"
                            
                            if chart_type == line_type:
                                fig = px.line(
                                    patient_df, x=time_col, y=value_col,
                                    title=chart_title,
                                    markers=True
                                )
                            elif chart_type == scatter_type:
                                fig = px.scatter(
                                    patient_df, x=time_col, y=value_col,
                                    title=chart_title,
                                    size_max=10
                                )
                            else:  # 面积图
                                fig = px.area(
                                    patient_df, x=time_col, y=value_col,
                                    title=chart_title
                                )
                            
                            # 美化图表
                            time_label = "Time (hours)" if lang == 'en' else "时间 (小时)"
                            fig.update_layout(
                                template="plotly_white",
                                hovermode="x unified",
                                xaxis_title=time_label,
                                yaxis_title=value_col.upper(),
                                font=dict(size=12),
                                title_font_size=16,
                                showlegend=False,
                                margin=dict(l=50, r=30, t=50, b=50),
                            )
                            fig.update_traces(
                                line=dict(width=2, color='#1f77b4'),
                                marker=dict(size=6)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # 🔧 只有数值没有时间列（静态数据/单点数据）
                            st.info("ℹ️ Static value (No time series data)" if lang == 'en' else "ℹ️ 静态数值（无时间序列数据）")
                            if len(patient_df) == 1:
                                val = patient_df[value_col].iloc[0]
                                st.metric(label=value_col.upper(), value=f"{val}")
                            else:
                                st.dataframe(patient_df[[value_col]], use_container_width=True)

                        # 显示统计信息
                        if show_stats:
                            stat_title = "#### 📊 Statistical Summary" if lang == 'en' else "#### 📊 统计摘要"
                            st.markdown(stat_title)
                            stat_cols = st.columns(5)
                            values = patient_df[value_col]
                            if lang == 'en':
                                stats = [
                                    ("Min", f"{values.min():.2f}", "📉"),
                                    ("Max", f"{values.max():.2f}", "📈"),
                                    ("Mean", f"{values.mean():.2f}", "📊"),
                                    ("Std Dev", f"{values.std():.2f}", "📐"),
                                    ("Records", f"{len(values)}", "📝"),
                                ]
                            else:
                                stats = [
                                    ("最小值", f"{values.min():.2f}", "📉"),
                                    ("最大值", f"{values.max():.2f}", "📈"),
                                    ("平均值", f"{values.mean():.2f}", "📊"),
                                    ("标准差", f"{values.std():.2f}", "📐"),
                                    ("记录数", f"{len(values)}", "📝"),
                                ]
                            for i, (label, value, icon) in enumerate(stats):
                                with stat_cols[i]:
                                    st.metric(f"{icon} {label}", value)
                    else:
                        warn_msg = "Data missing numeric value columns" if lang == 'en' else "数据中缺少数值列"
                        st.warning(warn_msg)
                        st.dataframe(patient_df.head(20), use_container_width=True)
                        
                except Exception as e:
                    err_msg = f"Chart rendering failed: {e}" if lang == 'en' else f"图表渲染失败: {e}"
                    st.warning(err_msg)
                    if 'time' in patient_df.columns:
                        chart_df = patient_df.set_index('time')
                        value_cols = [c for c in chart_df.columns if c not in [id_col]]
                        if value_cols:
                            st.line_chart(chart_df[value_cols[0]])
            else:
                no_data_msg = f"ℹ️ No {selected_concept} data for patient {patient_id}" if lang == 'en' else f"ℹ️ 患者 {patient_id} 无 {selected_concept} 数据"
                st.info(no_data_msg)
        
        # 数据表格预览
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        preview_label = "📋 Data Table Preview" if lang == 'en' else "📋 数据表格预览"
        with st.expander(preview_label, expanded=False):
            if selected_concept in st.session_state.loaded_concepts:
                df = st.session_state.loaded_concepts[selected_concept]
                if isinstance(df, pd.DataFrame):
                    if patient_id:
                        id_col = st.session_state.id_col
                        if id_col in df.columns:
                            df = df[df[id_col] == patient_id]
                    st.dataframe(df.head(50), use_container_width=True, hide_index=True)
                else:
                    format_msg = "Data format does not support preview" if lang == 'en' else "数据格式不支持预览"
                    st.info(format_msg)
    
    else:  # 多患者比较模式
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            concept_label = "📋 Select Concept" if lang == 'en' else "📋 选择 Concept"
            selected_concept = st.selectbox(
                concept_label,
                options=available_concepts,
                key="ts_concept_multi"
            )
        
        with col2:
            if st.session_state.patient_ids:
                compare_label = "👥 Select patients to compare (max 5)" if lang == 'en' else "👥 选择要比较的患者 (最多5个)"
                compare_patients = st.multiselect(
                    compare_label,
                    options=st.session_state.patient_ids[:50],
                    default=st.session_state.patient_ids[:3],
                    max_selections=5,
                    key="ts_compare_patients"
                )
            else:
                compare_patients = []
        
        with col3:
            normalize = st.checkbox("归一化比较", value=False, key="ts_normalize",
                                   help="将数值归一化到0-1范围便于比较")
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        if selected_concept and compare_patients:
            try:
                import plotly.graph_objects as go
                
                df = st.session_state.loaded_concepts[selected_concept]
                
                # 确保是 DataFrame
                if not isinstance(df, pd.DataFrame):
                    format_warn = f"Data format not supported for multi-patient comparison: {type(df).__name__}" if lang == 'en' else f"数据格式不支持多患者比较: {type(df).__name__}"
                    st.warning(format_warn)
                    return
                
                id_col = st.session_state.id_col
                
                # 确定数值列
                numeric_cols = df.select_dtypes(include=['number']).columns
                # 排除ID列和所有可能的时间列
                exclude_cols = ['stay_id', 'hadm_id', 'icustay_id', 'index', 'time',
                               'charttime', 'starttime', 'endtime', 'datetime', 'timestamp',
                               'patientunitstayid', 'admissionid', 'patientid']
                value_cols = [c for c in numeric_cols if c not in exclude_cols]
                
                # 检测时间列
                time_candidates = ['time', 'charttime', 'starttime', 'endtime', 'datetime', 'timestamp']
                time_col = None
                for tc in time_candidates:
                    if tc in df.columns:
                        time_col = tc
                        break
                
                if value_cols and time_col and id_col in df.columns:
                    value_col = value_cols[0]
                    
                    fig = go.Figure()
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                    
                    comparison_stats = []
                    
                    for i, pid in enumerate(compare_patients):
                        patient_df = df[df[id_col] == pid].sort_values(time_col)
                        
                        if len(patient_df) > 0:
                            y_values = patient_df[value_col].values
                            
                            # 归一化
                            if normalize and len(y_values) > 0:
                                y_min, y_max = y_values.min(), y_values.max()
                                if y_max > y_min:
                                    y_values = (y_values - y_min) / (y_max - y_min)
                            
                            fig.add_trace(go.Scatter(
                                x=patient_df[time_col],
                                y=y_values,
                                mode='lines+markers',
                                name=f"患者 {pid}",
                                line=dict(color=colors[i % len(colors)], width=2),
                                marker=dict(size=4)
                            ))
                            
                            comparison_stats.append({
                                '患者': pid,
                                '平均值': f"{patient_df[value_col].mean():.2f}",
                                '最大值': f"{patient_df[value_col].max():.2f}",
                                '最小值': f"{patient_df[value_col].min():.2f}",
                                '记录数': len(patient_df)
                            })
                    
                    fig.update_layout(
                        template="plotly_white",
                        title=f"📊 {selected_concept.upper()} 多患者比较",
                        xaxis_title="时间 (小时)",
                        yaxis_title=f"{value_col}" + (" (归一化)" if normalize else ""),
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                        height=450,
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 比较统计表
                    if comparison_stats:
                        compare_stats_title = "#### 📊 Comparison Statistics" if lang == 'en' else "#### 📊 比较统计"
                        st.markdown(compare_stats_title)
                        st.dataframe(pd.DataFrame(comparison_stats), use_container_width=True, hide_index=True)
                else:
                    format_warn = "Data format not supported for multi-patient comparison" if lang == 'en' else "数据格式不支持多患者比较"
                    st.warning(format_warn)
                    
            except Exception as e:
                err_msg = f"Comparison chart rendering failed: {e}" if lang == 'en' else f"比较图表渲染失败: {e}"
                st.error(err_msg)


def render_patient_page():
    """渲染患者视图页面。"""
    lang = st.session_state.get('language', 'en')
    
    page_title = "🏥 Patient Overview" if lang == 'en' else "🏥 患者综合视图"
    st.markdown(f"## {page_title}")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    if len(st.session_state.loaded_concepts) == 0:
        if lang == 'en':
            st.markdown('''
            <div class="info-box">
                <strong>👈 Please load data from the sidebar first</strong><br>
                💡 Tip: Select "Demo Mode" to quickly explore all features
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="info-box">
                <strong>👈 请先在侧边栏加载数据</strong><br>
                💡 提示：勾选「使用模拟数据」可快速体验所有功能
            </div>
            ''', unsafe_allow_html=True)
        return
    
    if not st.session_state.patient_ids:
        warn_msg = "⚠️ No patient data found" if lang == 'en' else "⚠️ 未找到患者数据"
        st.warning(warn_msg)
        return
    
    # 患者选择面板
    select_title = "🎛️ Patient Selection" if lang == 'en' else "🎛️ 患者选择"
    st.markdown(f"### {select_title}")
    
    # 快速导航按钮
    first_btn = "⏮️ First" if lang == 'en' else "⏮️ 首位"
    prev_btn = "⬅️ Previous" if lang == 'en' else "⬅️ 上一位"
    next_btn = "➡️ Next" if lang == 'en' else "➡️ 下一位"
    last_btn = "⏭️ Last" if lang == 'en' else "⏭️ 末位"
    rand_btn = "🎲 Random" if lang == 'en' else "🎲 随机"
    first_help = "Jump to first patient" if lang == 'en' else "跳转到第一位患者"
    prev_help = "Previous patient" if lang == 'en' else "上一位患者"
    next_help = "Next patient" if lang == 'en' else "下一位患者"
    last_help = "Jump to last patient" if lang == 'en' else "跳转到最后一位患者"
    rand_help = "Random select a patient" if lang == 'en' else "随机选择一位患者"
    
    nav_cols = st.columns(6)
    with nav_cols[0]:
        if st.button(first_btn, width="stretch", help=first_help):
            st.session_state.patient_view_id = st.session_state.patient_ids[0]
            st.rerun()
    with nav_cols[1]:
        if st.button(prev_btn, width="stretch", help=prev_help):
            current_idx = st.session_state.patient_ids.index(st.session_state.get('patient_view_id', st.session_state.patient_ids[0]))
            if current_idx > 0:
                st.session_state.patient_view_id = st.session_state.patient_ids[current_idx - 1]
                st.rerun()
    with nav_cols[2]:
        if st.button(next_btn, width="stretch", help=next_help):
            current_idx = st.session_state.patient_ids.index(st.session_state.get('patient_view_id', st.session_state.patient_ids[0]))
            if current_idx < len(st.session_state.patient_ids) - 1:
                st.session_state.patient_view_id = st.session_state.patient_ids[current_idx + 1]
                st.rerun()
    with nav_cols[3]:
        if st.button(last_btn, width="stretch", help=last_help):
            st.session_state.patient_view_id = st.session_state.patient_ids[-1]
            st.rerun()
    with nav_cols[4]:
        if st.button(rand_btn, width="stretch", help=rand_help):
            import random
            st.session_state.patient_view_id = random.choice(st.session_state.patient_ids)
            st.rerun()
    with nav_cols[5]:
        # 显示当前位置
        current_idx = st.session_state.patient_ids.index(st.session_state.get('patient_view_id', st.session_state.patient_ids[0]))
        st.markdown(f"<div style='text-align:center;padding:0.5rem;background:rgba(30,40,50,0.6);border-radius:4px'>{current_idx + 1}/{len(st.session_state.patient_ids)}</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        pat_id_label = "👤 Patient ID" if lang == 'en' else "👤 患者 ID"
        patient_id = st.selectbox(
            pat_id_label,
            options=st.session_state.patient_ids[:100],
            key="patient_view_id"
        )
    
    with col2:
        view_label = "📋 View Mode" if lang == 'en' else "📋 显示模式"
        view_options = ["Dashboard", "Category View", "Data Table"] if lang == 'en' else ["综合仪表盘", "分类视图", "数据表格"]
        view_mode = st.selectbox(
            view_label,
            options=view_options,
            key="patient_view_mode"
        )
    
    with col3:
        # 数据概览 - 显示更详细的可用数据信息
        id_col = st.session_state.id_col
        available_concepts = [k for k, v in st.session_state.loaded_concepts.items() 
                             if isinstance(v, pd.DataFrame) and id_col in v.columns 
                             and patient_id in v[id_col].values]
        n_concepts = len(available_concepts)
        
        # 统计各类别数据
        vitals_list = ['hr', 'map', 'sbp', 'dbp', 'resp', 'temp', 'spo2']
        labs_list = ['bili', 'crea', 'lac', 'plt', 'wbc', 'hgb', 'inr_pt', 'ptt']
        scores_list = ['sofa', 'sofa2', 'qsofa', 'sirs', 'gcs', 'sep3_sofa1', 'sep3_sofa2']
        
        n_vitals = len([c for c in available_concepts if c in vitals_list])
        n_labs = len([c for c in available_concepts if c in labs_list])
        n_scores = len([c for c in available_concepts if c in scores_list])
        
        data_label = "Available Data" if lang == 'en' else "可用数据"
        st.markdown(f'''
        <div class="metric-card" style="padding:0.5rem 1rem">
            <div class="stat-label">{data_label}</div>
            <div style="display:flex;gap:1rem;font-size:0.9rem">
                <span>📊 {n_concepts} total</span>
                <span>❤️ {n_vitals} vitals</span>
                <span>🧪 {n_labs} labs</span>
                <span>📈 {n_scores} scores</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # 判断视图模式
    dashboard_mode = "Dashboard" if lang == 'en' else "综合仪表盘"
    category_mode = "Category View" if lang == 'en' else "分类视图"
    table_mode = "Data Table" if lang == 'en' else "数据表格"
    
    if patient_id:
        st.session_state.selected_patient = patient_id
        id_col = st.session_state.id_col
        
        if view_mode == dashboard_mode:
            # 自定义综合仪表盘
            dash_title = "### 📊 Dashboard" if lang == 'en' else "### 📊 综合仪表盘"
            st.markdown(dash_title)
            
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # 收集所有生命体征数据
                vitals = ['hr', 'map', 'sbp', 'resp', 'spo2']
                vitals_data = {}
                time_candidates = ['time', 'charttime', 'starttime', 'endtime', 'datetime', 'timestamp']
                
                for v in vitals:
                    if v in st.session_state.loaded_concepts:
                        df = st.session_state.loaded_concepts[v]
                        if isinstance(df, pd.DataFrame) and id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                            if len(patient_df) > 0:
                                # 检测时间列
                                time_col = None
                                for tc in time_candidates:
                                    if tc in patient_df.columns:
                                        time_col = tc
                                        break
                                if time_col:
                                    vitals_data[v] = (patient_df, time_col)
                
                if vitals_data:
                    # 创建多行子图
                    n_vitals = len(vitals_data)
                    fig = make_subplots(
                        rows=n_vitals, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=[v.upper() for v in vitals_data.keys()]
                    )
                    
                    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
                    
                    for i, (name, (df, time_col)) in enumerate(vitals_data.items(), 1):
                        value_col = name if name in df.columns else df.columns[-1]
                        fig.add_trace(
                            go.Scatter(
                                x=df[time_col], y=df[value_col],
                                mode='lines+markers',
                                name=name.upper(),
                                line=dict(color=colors[(i-1) % len(colors)], width=2),
                                marker=dict(size=4)
                            ),
                            row=i, col=1
                        )
                    
                    vitals_title = f"Patient {patient_id} Vital Signs Trend" if lang == 'en' else f"患者 {patient_id} 生命体征趋势"
                    fig.update_layout(
                        height=150 * n_vitals + 100,
                        template="plotly_white",
                        showlegend=False,
                        title_text=vitals_title,
                        title_font_size=16,
                        margin=dict(l=50, r=30, t=60, b=50),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    no_vitals = "ℹ️ No vital signs data available" if lang == 'en' else "ℹ️ 无可用的生命体征数据"
                    st.info(no_vitals)
                
                # SOFA 评分趋势
                if 'sofa' in st.session_state.loaded_concepts:
                    sofa_df = st.session_state.loaded_concepts['sofa']
                    if isinstance(sofa_df, pd.DataFrame) and id_col in sofa_df.columns:
                        patient_sofa = sofa_df[sofa_df[id_col] == patient_id]
                        # 检测时间列
                        sofa_time_col = None
                        for tc in time_candidates:
                            if tc in patient_sofa.columns:
                                sofa_time_col = tc
                                break
                        
                        if len(patient_sofa) > 0 and sofa_time_col:
                            sofa_trend = "#### 📈 SOFA Score Trend" if lang == 'en' else "#### 📈 SOFA 评分趋势"
                            st.markdown(sofa_trend)
                            
                            # SOFA 分解堆叠图
                            sofa_components = ['sofa_resp', 'sofa_coag', 'sofa_liver', 
                                             'sofa_cardio', 'sofa_cns', 'sofa_renal']
                            available_components = [c for c in sofa_components if c in patient_sofa.columns]
                            
                            if available_components:
                                fig = go.Figure()
                                colors = ['#ff6b6b', '#feca57', '#48dbfb', '#ff9ff3', '#54a0ff', '#5f27cd']
                                
                                for i, comp in enumerate(available_components):
                                    fig.add_trace(go.Bar(
                                        x=patient_sofa[sofa_time_col],
                                        y=patient_sofa[comp],
                                        name=comp.replace('sofa_', '').upper(),
                                        marker_color=colors[i]
                                    ))
                                
                                time_label = "Time" if lang == 'en' else "时间"
                                score_label = "SOFA Score" if lang == 'en' else "SOFA 分数"
                                fig.update_layout(
                                    barmode='stack',
                                    template="plotly_white",
                                    height=350,
                                    xaxis_title=time_label,
                                    yaxis_title=score_label,
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                
                # ============ SOFA-1 vs SOFA-2 对比图表 ============
                has_sofa1 = 'sofa' in st.session_state.loaded_concepts
                has_sofa2 = 'sofa2' in st.session_state.loaded_concepts
                
                if has_sofa1 and has_sofa2:
                    compare_title = "#### 🔄 SOFA-1 vs SOFA-2 Comparison" if lang == 'en' else "#### 🔄 SOFA-1 与 SOFA-2 对比"
                    st.markdown(compare_title)
                    
                    sofa1_df = st.session_state.loaded_concepts['sofa']
                    sofa2_df = st.session_state.loaded_concepts['sofa2']
                    
                    # 获取患者数据
                    if isinstance(sofa1_df, pd.DataFrame) and id_col in sofa1_df.columns:
                        patient_sofa1 = sofa1_df[sofa1_df[id_col] == patient_id].copy()
                    else:
                        patient_sofa1 = pd.DataFrame()
                    
                    if isinstance(sofa2_df, pd.DataFrame) and id_col in sofa2_df.columns:
                        patient_sofa2 = sofa2_df[sofa2_df[id_col] == patient_id].copy()
                    else:
                        patient_sofa2 = pd.DataFrame()
                    
                    if len(patient_sofa1) > 0 and len(patient_sofa2) > 0:
                        # 检测时间列
                        time_col1 = None
                        time_col2 = None
                        for tc in time_candidates:
                            if tc in patient_sofa1.columns and time_col1 is None:
                                time_col1 = tc
                            if tc in patient_sofa2.columns and time_col2 is None:
                                time_col2 = tc
                        
                        if time_col1 and time_col2:
                            # 1. 总分对比折线图
                            total_compare = "**Total Score Comparison**" if lang == 'en' else "**总分对比**"
                            st.markdown(total_compare)
                            
                            fig_total = go.Figure()
                            
                            # SOFA-1 总分
                            if 'sofa' in patient_sofa1.columns:
                                fig_total.add_trace(go.Scatter(
                                    x=patient_sofa1[time_col1],
                                    y=patient_sofa1['sofa'],
                                    mode='lines+markers',
                                    name='SOFA-1 (Traditional)',
                                    line=dict(color='#1f77b4', width=3),
                                    marker=dict(size=8)
                                ))
                            
                            # SOFA-2 总分
                            if 'sofa2' in patient_sofa2.columns:
                                fig_total.add_trace(go.Scatter(
                                    x=patient_sofa2[time_col2],
                                    y=patient_sofa2['sofa2'],
                                    mode='lines+markers',
                                    name='SOFA-2 (2025 New)',
                                    line=dict(color='#ff7f0e', width=3, dash='dash'),
                                    marker=dict(size=8, symbol='diamond')
                                ))
                            
                            time_label = "Time (hours from ICU admission)" if lang == 'en' else "时间 (ICU入院后小时)"
                            score_label = "Total SOFA Score" if lang == 'en' else "SOFA 总分"
                            fig_total.update_layout(
                                template="plotly_white",
                                height=300,
                                xaxis_title=time_label,
                                yaxis_title=score_label,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig_total, use_container_width=True)
                            
                            # 2. 子器官评分对比（6个子图）
                            organ_compare = "**Organ-specific Score Comparison**" if lang == 'en' else "**各器官评分对比**"
                            st.markdown(organ_compare)
                            
                            # 定义器官映射
                            organ_pairs = [
                                ('sofa_resp', 'sofa2_resp', 'Respiratory', '呼吸'),
                                ('sofa_coag', 'sofa2_coag', 'Coagulation', '凝血'),
                                ('sofa_liver', 'sofa2_liver', 'Liver', '肝脏'),
                                ('sofa_cardio', 'sofa2_cardio', 'Cardiovascular', '心血管'),
                                ('sofa_cns', 'sofa2_cns', 'Neurological', '神经'),
                                ('sofa_renal', 'sofa2_renal', 'Renal', '肾脏'),
                            ]
                            
                            # 🔧 检查器官评分列是否存在于各自的 DataFrame 中
                            # 如果不存在，尝试从其他加载的 concepts 中获取
                            def get_organ_data(patient_df, organ_col, time_col, loaded_concepts, id_col, patient_id):
                                """获取器官评分数据，优先从 sofa/sofa2 DataFrame，否则从单独加载的 concept"""
                                try:
                                    if organ_col in patient_df.columns and time_col in patient_df.columns:
                                        return patient_df[[time_col, organ_col]].copy()
                                    # 尝试从单独加载的 concept 获取
                                    if organ_col in loaded_concepts:
                                        organ_df = loaded_concepts[organ_col]
                                        if isinstance(organ_df, pd.DataFrame) and id_col in organ_df.columns:
                                            patient_organ = organ_df[organ_df[id_col] == patient_id].copy()
                                            if len(patient_organ) > 0 and organ_col in patient_organ.columns:
                                                # 找时间列
                                                for tc in ['time', 'charttime', 'starttime']:
                                                    if tc in patient_organ.columns:
                                                        return patient_organ[[tc, organ_col]].rename(columns={tc: time_col})
                                except Exception:
                                    pass
                                return None
                            
                            # 创建 2x3 子图
                            from plotly.subplots import make_subplots
                            
                            fig_organs = make_subplots(
                                rows=2, cols=3,
                                subplot_titles=[p[2] if lang == 'en' else p[3] for p in organ_pairs],
                                vertical_spacing=0.15,
                                horizontal_spacing=0.08
                            )
                            
                            has_any_data = False
                            for idx, (sofa1_col, sofa2_col, en_name, zh_name) in enumerate(organ_pairs):
                                row = idx // 3 + 1
                                col = idx % 3 + 1
                                
                                # SOFA-1 器官评分
                                sofa1_organ = get_organ_data(patient_sofa1, sofa1_col, time_col1, 
                                                            st.session_state.loaded_concepts, id_col, patient_id)
                                if sofa1_organ is not None and len(sofa1_organ) > 0:
                                    has_any_data = True
                                    fig_organs.add_trace(
                                        go.Scatter(
                                            x=sofa1_organ[time_col1],
                                            y=sofa1_organ[sofa1_col],
                                            mode='lines+markers',
                                            name='SOFA-1' if idx == 0 else None,
                                            legendgroup='sofa1',
                                            showlegend=(idx == 0),
                                            line=dict(color='#1f77b4', width=2),
                                            marker=dict(size=5)
                                        ),
                                        row=row, col=col
                                    )
                                
                                # SOFA-2 器官评分
                                sofa2_organ = get_organ_data(patient_sofa2, sofa2_col, time_col2,
                                                            st.session_state.loaded_concepts, id_col, patient_id)
                                if sofa2_organ is not None and len(sofa2_organ) > 0:
                                    has_any_data = True
                                    fig_organs.add_trace(
                                        go.Scatter(
                                            x=sofa2_organ[time_col2],
                                            y=sofa2_organ[sofa2_col],
                                            mode='lines+markers',
                                            name='SOFA-2' if idx == 0 else None,
                                            legendgroup='sofa2',
                                            showlegend=(idx == 0),
                                            line=dict(color='#ff7f0e', width=2, dash='dash'),
                                            marker=dict(size=5, symbol='diamond')
                                        ),
                                        row=row, col=col
                                    )
                            
                            if has_any_data:
                                fig_organs.update_layout(
                                    height=500,
                                    template="plotly_white",
                                    legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
                                    hovermode='x unified'
                                )
                                
                                # 更新 y 轴范围 (0-4)
                                for i in range(1, 7):
                                    fig_organs.update_yaxes(range=[0, 4.5], row=(i-1)//3+1, col=(i-1)%3+1)
                                
                                st.plotly_chart(fig_organs, use_container_width=True)
                            else:
                                no_organ_msg = "ℹ️ Organ-specific scores not available in current data. Load individual organ concepts (e.g., sofa_resp, sofa2_resp) to see detailed comparison." if lang == 'en' else "ℹ️ 当前数据中无法获取器官子评分。请加载单独的器官概念（如 sofa_resp, sofa2_resp）以查看详细对比。"
                                st.info(no_organ_msg)
                            
                            # 3. 差异分析表格
                            diff_title = "**Score Difference (SOFA-2 - SOFA-1)**" if lang == 'en' else "**评分差异 (SOFA-2 - SOFA-1)**"
                            st.markdown(diff_title)
                            
                            # 计算最新时间点的差异
                            latest_sofa1 = patient_sofa1.iloc[-1] if len(patient_sofa1) > 0 else {}
                            latest_sofa2 = patient_sofa2.iloc[-1] if len(patient_sofa2) > 0 else {}
                            
                            diff_data = []
                            for sofa1_col, sofa2_col, en_name, zh_name in organ_pairs:
                                val1 = latest_sofa1.get(sofa1_col, 0) if isinstance(latest_sofa1, dict) or hasattr(latest_sofa1, 'get') else (latest_sofa1[sofa1_col] if sofa1_col in latest_sofa1.index else 0)
                                val2 = latest_sofa2.get(sofa2_col, 0) if isinstance(latest_sofa2, dict) or hasattr(latest_sofa2, 'get') else (latest_sofa2[sofa2_col] if sofa2_col in latest_sofa2.index else 0)
                                diff = val2 - val1
                                organ_name = en_name if lang == 'en' else zh_name
                                diff_data.append({
                                    'Organ' if lang == 'en' else '器官': organ_name,
                                    'SOFA-1': int(val1),
                                    'SOFA-2': int(val2),
                                    'Diff' if lang == 'en' else '差异': int(diff)
                                })
                            
                            # 总分差异
                            total1 = latest_sofa1.get('sofa', 0) if isinstance(latest_sofa1, dict) or hasattr(latest_sofa1, 'get') else (latest_sofa1['sofa'] if 'sofa' in latest_sofa1.index else 0)
                            total2 = latest_sofa2.get('sofa2', 0) if isinstance(latest_sofa2, dict) or hasattr(latest_sofa2, 'get') else (latest_sofa2['sofa2'] if 'sofa2' in latest_sofa2.index else 0)
                            diff_data.append({
                                'Organ' if lang == 'en' else '器官': '**Total**' if lang == 'en' else '**总分**',
                                'SOFA-1': int(total1),
                                'SOFA-2': int(total2),
                                'Diff' if lang == 'en' else '差异': int(total2 - total1)
                            })
                            
                            diff_df = pd.DataFrame(diff_data)
                            st.dataframe(diff_df, use_container_width=True, hide_index=True)
                    else:
                        no_compare = "ℹ️ Need both SOFA-1 and SOFA-2 data for comparison" if lang == 'en' else "ℹ️ 需要同时有 SOFA-1 和 SOFA-2 数据才能对比"
                        st.info(no_compare)
                
                # Dashboard 快速摘要面板
                summary_title = "#### 📋 Quick Summary" if lang == 'en' else "#### 📋 快速摘要"
                st.markdown(summary_title)
                
                summary_cols = st.columns(4)
                
                # Sepsis 状态
                with summary_cols[0]:
                    sepsis_status = "Not loaded ⚪" if lang == 'en' else "未加载 ⚪"
                    sepsis_color = "#6c757d"
                    
                    found_sep = False
                    if 'sep3_sofa2' in st.session_state.loaded_concepts:
                        sep_df = st.session_state.loaded_concepts['sep3_sofa2']
                        concept_key = 'sep3_sofa2'
                        found_sep = True
                    elif 'sep3_sofa1' in st.session_state.loaded_concepts:
                        sep_df = st.session_state.loaded_concepts['sep3_sofa1']
                        concept_key = 'sep3_sofa1'
                        found_sep = True
                    
                    if found_sep:
                        sepsis_status = "Unknown"
                        if isinstance(sep_df, pd.DataFrame) and id_col in sep_df.columns:
                            patient_sep = sep_df[sep_df[id_col] == patient_id]
                            if len(patient_sep) > 0 and concept_key in patient_sep.columns:
                                if patient_sep[concept_key].max() == 1:
                                    sepsis_status = "Sepsis ⚠️" if lang == 'en' else "脓毒症 ⚠️"
                                    sepsis_color = "#dc3545"
                                else:
                                    sepsis_status = "No Sepsis ✅" if lang == 'en' else "无脓毒症 ✅"
                                    sepsis_color = "#28a745"
                            else:
                                sepsis_status = "No Records" if lang == 'en' else "无记录"

                    st.markdown(f"**Sepsis-3**" if lang == 'en' else f"**脓毒症-3**")
                    st.markdown(f"<span style='color:{sepsis_color};font-weight:bold'>{sepsis_status}</span>", unsafe_allow_html=True)
                
                # 机械通气
                with summary_cols[1]:
                    vent_status = "Not loaded ⚪" if lang == 'en' else "未加载 ⚪"
                    vent_concepts = ['vent_ind', 'mech_vent', 'vent_start']
                    
                    # 检查是否有相关 concept 被加载
                    found_vent = any(c in st.session_state.loaded_concepts for c in vent_concepts)
                    
                    if found_vent:
                        vent_status = "Unknown"
                        if 'vent_ind' in st.session_state.loaded_concepts:
                            vent_df = st.session_state.loaded_concepts['vent_ind']
                            if isinstance(vent_df, pd.DataFrame) and id_col in vent_df.columns:
                                patient_vent = vent_df[vent_df[id_col] == patient_id]
                                if len(patient_vent) > 0 and 'vent_ind' in patient_vent.columns:
                                    vent_status = "Yes ✅" if patient_vent['vent_ind'].max() == 1 else "No ❌"
                                else:
                                    vent_status = "No Records" if lang == 'en' else "无记录"
                    
                    st.markdown(f"**Mechanical Vent**" if lang == 'en' else f"**机械通气**")
                    st.markdown(vent_status)
                
                # 血管活性药物
                with summary_cols[2]:
                    vaso_status = "Not loaded ⚪" if lang == 'en' else "未加载 ⚪"
                    vaso_concepts = ['norepi_rate', 'epi_rate', 'dopa_rate', 'vaso_ind']
                    
                    found_vaso = any(c in st.session_state.loaded_concepts for c in vaso_concepts)
                    
                    if found_vaso:
                        vaso_status = "No ❌"
                        for vc in vaso_concepts:
                            if vc in st.session_state.loaded_concepts:
                                vdf = st.session_state.loaded_concepts[vc]
                                if isinstance(vdf, pd.DataFrame) and id_col in vdf.columns:
                                    pvdf = vdf[vdf[id_col] == patient_id]
                                    if len(pvdf) > 0:
                                        val_col = vc if vc in pvdf.columns else pvdf.columns[-1]
                                        if pvdf[val_col].max() > 0:
                                            vaso_status = "Yes ✅"
                                            break
                    
                    st.markdown(f"**Vasopressors**" if lang == 'en' else f"**血管活性药**")
                    st.markdown(vaso_status)
                
                # GCS
                with summary_cols[3]:
                    gcs_val = "Not loaded" if lang == 'en' else "未加载"
                    gcs_color = "#6c757d"
                    
                    if 'gcs' in st.session_state.loaded_concepts:
                        gcs_val = "N/A"
                        gcs_df = st.session_state.loaded_concepts['gcs']
                        if isinstance(gcs_df, pd.DataFrame) and id_col in gcs_df.columns:
                            patient_gcs = gcs_df[gcs_df[id_col] == patient_id]
                            if len(patient_gcs) > 0 and 'gcs' in patient_gcs.columns:
                                val = patient_gcs['gcs'].iloc[-1]
                                gcs_color = "#28a745" if val >= 13 else ("#ffc107" if val >= 9 else "#dc3545")
                                gcs_val = f"{val:.0f}"
                            else:
                                gcs_val = "No Records" if lang == 'en' else "无记录"
                    # 尝试从 sofa_cns 推断
                    elif 'sofa_cns' in st.session_state.loaded_concepts or 'sofa2_cns' in st.session_state.loaded_concepts:
                        cns_col = 'sofa_cns' if 'sofa_cns' in st.session_state.loaded_concepts else 'sofa2_cns'
                        cns_df = st.session_state.loaded_concepts[cns_col]
                        if isinstance(cns_df, pd.DataFrame) and id_col in cns_df.columns:
                            patient_cns = cns_df[cns_df[id_col] == patient_id]
                            if len(patient_cns) > 0 and cns_col in patient_cns.columns:
                                cns_score = patient_cns[cns_col].iloc[-1]
                                # 0:15, 1:13-14, 2:10-12, 3:6-9, 4:<6
                                if cns_score == 0: gcs_val, gcs_color = "15 (est)", "#28a745"
                                elif cns_score == 1: gcs_val, gcs_color = "13-14 (est)", "#28a745"
                                elif cns_score == 2: gcs_val, gcs_color = "10-12 (est)", "#ffc107"
                                elif cns_score == 3: gcs_val, gcs_color = "6-9 (est)", "#dc3545"
                                elif cns_score == 4: gcs_val, gcs_color = "<6 (est)", "#dc3545"
                    
                    st.markdown("**GCS**")
                    st.markdown(f"<span style='color:{gcs_color};font-weight:bold;font-size:1.2rem'>{gcs_val}</span>", unsafe_allow_html=True)
                            
            except Exception as e:
                err_msg = f"Dashboard rendering failed: {e}" if lang == 'en' else f"综合仪表盘渲染失败: {e}"
                st.warning(err_msg)
                switch_msg = "Please try switching to 'Category View'" if lang == 'en' else "请尝试切换到「分类视图」"
                st.info(switch_msg)
        
        elif view_mode == category_mode:
            # 时间列候选（提前定义，避免UnboundLocalError）
            time_candidates = ['time', 'charttime', 'starttime', 'endtime', 'datetime', 'timestamp']
            
            # 生命体征
            vitals_title = "### ❤️ Vital Signs" if lang == 'en' else "### ❤️ 生命体征"
            st.markdown(vitals_title)
            vitals = ['hr', 'map', 'sbp', 'resp', 'temp', 'spo2']
            vitals_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                          if k in vitals and isinstance(v, pd.DataFrame)}
            
            if vitals_data:
                cols = st.columns(min(3, len(vitals_data)))
                
                for i, (concept, df) in enumerate(vitals_data.items()):
                    with cols[i % 3]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df
                        
                        if len(patient_df) > 0:
                            # 显示最新值
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            latest_val = patient_df[value_col].iloc[-1]
                            st.metric(concept.upper(), f"{latest_val:.1f}")
                            
                            # 小型趋势图 - 检测时间列
                            time_col = None
                            for tc in time_candidates:
                                if tc in patient_df.columns:
                                    time_col = tc
                                    break
                            if time_col:
                                st.line_chart(patient_df.set_index(time_col)[value_col], height=120)
            else:
                no_vitals = "ℹ️ No vital signs data available" if lang == 'en' else "ℹ️ 无可用的生命体征数据"
                st.info(no_vitals)
            
            # SOFA/SOFA2 评分
            sofa_concepts = ['sofa', 'sofa2']
            sofa_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                        if k in sofa_concepts and isinstance(v, pd.DataFrame)}
            
            if sofa_data:
                sofa_title = "### 📊 SOFA Score" if lang == 'en' else "### 📊 SOFA 评分"
                st.markdown(sofa_title)
                
                for sofa_key, sofa_df in sofa_data.items():
                    if id_col in sofa_df.columns:
                        patient_sofa = sofa_df[sofa_df[id_col] == patient_id]
                    else:
                        patient_sofa = sofa_df
                    
                    if len(patient_sofa) > 0:
                        latest = patient_sofa.iloc[-1]
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            sofa_val = latest.get(sofa_key, 0)
                            sofa_color = "#28a745" if sofa_val < 6 else ("#ffc107" if sofa_val < 10 else "#dc3545")
                            label = f"Latest {sofa_key.upper()}" if lang == 'en' else f"最新 {sofa_key.upper()}"
                            st.markdown(f'''
                            <div class="metric-card" style="text-align:center">
                                <div class="stat-label">{label}</div>
                                <div class="stat-number" style="color:{sofa_color}">{sofa_val}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        with col2:
                            sofa_time_col = None
                            for tc in time_candidates:
                                if tc in patient_sofa.columns:
                                    sofa_time_col = tc
                                    break
                            if sofa_key in patient_sofa.columns and sofa_time_col:
                                st.line_chart(patient_sofa.set_index(sofa_time_col)[sofa_key], height=150)
            
            # Sepsis-3 诊断状态
            sepsis_concepts = ['sep3_sofa1', 'sep3_sofa2', 'susp_inf', 'infection_icd']
            sepsis_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                          if k in sepsis_concepts and isinstance(v, pd.DataFrame)}
            
            if sepsis_data:
                sepsis_title = "### 🦠 Sepsis-3 Status" if lang == 'en' else "### 🦠 Sepsis-3 诊断"
                st.markdown(sepsis_title)
                cols = st.columns(len(sepsis_data))
                for i, (concept, df) in enumerate(sepsis_data.items()):
                    with cols[i]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df
                        
                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            val = patient_df[value_col].iloc[-1] if len(patient_df) > 0 else 0
                            if val == 1:
                                st.markdown(f"✅ **{concept}**: Yes" if lang == 'en' else f"✅ **{concept}**: 是")
                            else:
                                st.markdown(f"❌ **{concept}**: No" if lang == 'en' else f"❌ **{concept}**: 否")
            
            # 实验室检查 - 扩展更多指标
            labs = ['bili', 'crea', 'lac', 'lact', 'plt', 'wbc', 'hgb', 'hct', 'inr_pt', 'ptt', 'alb', 'glu', 'na', 'k', 'cl', 'bun']
            labs_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                        if k in labs and isinstance(v, pd.DataFrame)}
            
            if labs_data:
                labs_title = "### 🧪 Laboratory Tests" if lang == 'en' else "### 🧪 实验室检查"
                st.markdown(labs_title)
                cols = st.columns(min(4, len(labs_data)))
                for i, (concept, df) in enumerate(labs_data.items()):
                    with cols[i % 4]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df
                        
                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            st.metric(
                                label=concept.upper(),
                                value=f"{patient_df[value_col].iloc[-1]:.2f}",
                                delta=f"{patient_df[value_col].iloc[-1] - patient_df[value_col].iloc[0]:.2f}" if len(patient_df) > 1 else None
                            )
                            lab_time_col = None
                            for tc in time_candidates:
                                if tc in patient_df.columns:
                                    lab_time_col = tc
                                    break
                            if lab_time_col:
                                st.line_chart(patient_df.set_index(lab_time_col)[value_col], height=120)
            
            # 血气分析
            blood_gas = ['ph', 'pco2', 'po2', 'pafi', 'safi', 'be', 'hco3', 'bicar', 'fio2']
            bg_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                      if k in blood_gas and isinstance(v, pd.DataFrame)}
            
            if bg_data:
                bg_title = "### 🩸 Blood Gas Analysis" if lang == 'en' else "### 🩸 血气分析"
                st.markdown(bg_title)
                cols = st.columns(min(4, len(bg_data)))
                for i, (concept, df) in enumerate(bg_data.items()):
                    with cols[i % 4]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df
                        
                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            st.metric(label=concept.upper(), value=f"{patient_df[value_col].iloc[-1]:.2f}")
            
            # 血管活性药物
            vasopressors = ['norepi_rate', 'epi_rate', 'dopa_rate', 'dobu_rate', 'adh_rate', 'phn_rate', 'vaso_ind']
            vaso_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                        if k in vasopressors and isinstance(v, pd.DataFrame)}
            
            if vaso_data:
                vaso_title = "### 💉 Vasopressors" if lang == 'en' else "### 💉 血管活性药物"
                st.markdown(vaso_title)
                cols = st.columns(min(4, len(vaso_data)))
                for i, (concept, df) in enumerate(vaso_data.items()):
                    with cols[i % 4]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df
                        
                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            if concept == 'vaso_ind':
                                val = patient_df[value_col].max()
                                st.markdown(f"**{concept}**: {'Yes ✅' if val == 1 else 'No ❌'}")
                            else:
                                st.metric(label=concept.upper(), value=f"{patient_df[value_col].iloc[-1]:.3f}")
                                vaso_time_col = None
                                for tc in time_candidates:
                                    if tc in patient_df.columns:
                                        vaso_time_col = tc
                                        break
                                if vaso_time_col:
                                    st.line_chart(patient_df.set_index(vaso_time_col)[value_col], height=100)
            
            # 呼吸支持
            resp_support = ['vent_ind', 'fio2', 'spo2', 'pafi', 'safi', 'resp']
            resp_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                        if k in resp_support and isinstance(v, pd.DataFrame) and k not in bg_data}  # 避免重复
            
            if resp_data:
                resp_title = "### 🫁 Respiratory Support" if lang == 'en' else "### 🫁 呼吸支持"
                st.markdown(resp_title)
                cols = st.columns(min(4, len(resp_data)))
                for i, (concept, df) in enumerate(resp_data.items()):
                    with cols[i % 4]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df
                        
                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            if concept == 'vent_ind':
                                val = patient_df[value_col].max()
                                st.markdown(f"**Mechanical Vent**: {'Yes ✅' if val == 1 else 'No ❌'}" if lang == 'en' else f"**机械通气**: {'是 ✅' if val == 1 else '否 ❌'}")
                            else:
                                st.metric(label=concept.upper(), value=f"{patient_df[value_col].iloc[-1]:.1f}")
            
            # 神经系统
            neuro = ['gcs', 'egcs', 'mgcs', 'vgcs', 'rass', 'avpu']
            neuro_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                         if k in neuro and isinstance(v, pd.DataFrame)}
            
            if neuro_data:
                neuro_title = "### 🧠 Neurological" if lang == 'en' else "### 🧠 神经系统"
                st.markdown(neuro_title)
                cols = st.columns(min(4, len(neuro_data)))
                for i, (concept, df) in enumerate(neuro_data.items()):
                    with cols[i % 4]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df
                        
                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            val = patient_df[value_col].iloc[-1]
                            # GCS 颜色编码
                            if concept == 'gcs':
                                color = "#28a745" if val >= 13 else ("#ffc107" if val >= 9 else "#dc3545")
                                st.markdown(f"<div style='color:{color};font-size:1.5rem;font-weight:bold'>GCS: {val:.0f}</div>", unsafe_allow_html=True)
                            else:
                                st.metric(label=concept.upper(), value=f"{val:.0f}")
            
            # 肾脏功能
            renal = ['urine', 'urine24', 'crea', 'bun', 'rrt']
            renal_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                         if k in renal and isinstance(v, pd.DataFrame) and k not in labs_data}
            
            if renal_data:
                renal_title = "### 🚰 Renal Function" if lang == 'en' else "### 🚰 肾脏功能"
                st.markdown(renal_title)
                cols = st.columns(min(4, len(renal_data)))
                for i, (concept, df) in enumerate(renal_data.items()):
                    with cols[i % 4]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df
                        
                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            if concept == 'rrt':
                                val = patient_df[value_col].max()
                                st.markdown(f"**RRT**: {'Yes ✅' if val == 1 else 'No ❌'}")
                            else:
                                st.metric(label=concept.upper(), value=f"{patient_df[value_col].iloc[-1]:.1f}")
            
            # 其他评分
            other_scores = ['qsofa', 'sirs', 'mews', 'news']
            score_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                         if k in other_scores and isinstance(v, pd.DataFrame)}
            
            if score_data:
                score_title = "### 📈 Other Scores" if lang == 'en' else "### 📈 其他评分"
                st.markdown(score_title)
                cols = st.columns(min(4, len(score_data)))
                for i, (concept, df) in enumerate(score_data.items()):
                    with cols[i % 4]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df
                        
                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            st.metric(label=concept.upper(), value=f"{patient_df[value_col].iloc[-1]:.0f}")
        
        elif view_mode == table_mode:
            table_title = "### 📋 Patient Data Table" if lang == 'en' else "### 📋 患者数据表格"
            st.markdown(table_title)
            for concept, df in st.session_state.loaded_concepts.items():
                if id_col in df.columns:
                    patient_df = df[df[id_col] == patient_id]
                else:
                    patient_df = df
                
                if len(patient_df) > 0:
                    records_label = "records" if lang == 'en' else "条记录"
                    with st.expander(f"{concept} ({len(patient_df)} {records_label})", expanded=False):
                        st.dataframe(patient_df, use_container_width=True)


def render_quality_page():
    """渲染数据质量页面。"""
    lang = st.session_state.get('language', 'en')
    
    page_title = "📊 Data Quality Assessment" if lang == 'en' else "📊 数据质量评估"
    st.markdown(f"## {page_title}")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    if len(st.session_state.loaded_concepts) == 0:
        if lang == 'en':
            st.markdown('''
            <div class="info-box">
                <strong>👈 Please load data from the sidebar first</strong><br>
                💡 Tip: Select "Demo Mode" to quickly explore all features
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="info-box">
                <strong>👈 请先在侧边栏加载数据</strong><br>
                💡 提示：勾选「使用模拟数据」可快速体验所有功能
            </div>
            ''', unsafe_allow_html=True)
        return
    
    # 总体质量评分
    quality_title = "🎯 Quality Score" if lang == 'en' else "🎯 质量评分"
    st.markdown(f"### {quality_title}")
    
    total_records = 0
    total_missing = 0
    quality_data = []
    
    for concept, df in st.session_state.loaded_concepts.items():
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            numeric_cols = df.select_dtypes(include=['number']).columns
            # 排除ID列和所有可能的时间列，只保留真正的数值列
            exclude_cols = ['stay_id', 'hadm_id', 'icustay_id', 'time', 'index',
                           'charttime', 'starttime', 'endtime', 'datetime', 'timestamp',
                           'patientunitstayid', 'admissionid', 'patientid']
            value_cols = [c for c in numeric_cols if c not in exclude_cols]
            
            n_records = len(df)
            n_patients = df[st.session_state.id_col].nunique() if st.session_state.id_col in df.columns else 0
            
            # 计算 NA 缺失率
            na_rate = df[value_cols].isna().mean().mean() * 100 if value_cols else 0
            
            # 计算数据覆盖率：每个患者的记录数 / 理论记录数（假设每小时1条）
            if n_patients > 0 and value_cols:
                records_per_patient = n_records / n_patients
                # 假设ICU住院平均72小时，每小时1条生命体征
                expected_records = 72 if concept in ['hr', 'map', 'sbp', 'resp', 'spo2', 'temp'] else 24
                coverage_rate = min(100, (records_per_patient / expected_records) * 100)
                # 综合缺失率 = NA缺失 + (100 - 覆盖率) * 0.3
                missing_rate = na_rate + (100 - coverage_rate) * 0.3 if na_rate == 0 else na_rate
            else:
                missing_rate = na_rate
            
            total_records += n_records
            total_missing += n_records * (missing_rate / 100)
            
            # 质量等级
            if lang == 'en':
                if missing_rate < 5:
                    quality = "🟢 Excellent"
                elif missing_rate < 15:
                    quality = "🟡 Good"
                elif missing_rate < 30:
                    quality = "🟠 Fair"
                else:
                    quality = "🔴 Poor"
            else:
                if missing_rate < 5:
                    quality = "🟢 优秀"
                elif missing_rate < 15:
                    quality = "🟡 良好"
                elif missing_rate < 30:
                    quality = "🟠 一般"
                else:
                    quality = "🔴 较差"
            
            records_col = "Records" if lang == 'en' else "记录数"
            patients_col = "Patients" if lang == 'en' else "患者数"
            missing_col = "Missing %" if lang == 'en' else "缺失率"
            quality_col = "Quality" if lang == 'en' else "质量"
            
            quality_data.append({
                'Concept': concept,
                records_col: f"{n_records:,}",
                patients_col: n_patients,
                missing_col: f"{missing_rate:.1f}%",
                quality_col: quality,
            })
    
    # 总体质量评分卡片
    overall_missing = (total_missing / total_records * 100) if total_records > 0 else 0
    quality_score = max(0, 100 - overall_missing * 2)
    
    col1, col2, col3, col4 = st.columns(4)
    
    score_label = "Quality Score" if lang == 'en' else "质量评分"
    records_label = "Total Records" if lang == 'en' else "总记录数"
    missing_label = "Avg Missing %" if lang == 'en' else "平均缺失率"
    items_label = "Data Items" if lang == 'en' else "数据项数"
    
    with col1:
        score_color = "#28a745" if quality_score >= 80 else ("#ffc107" if quality_score >= 60 else "#dc3545")
        st.markdown(f'''
        <div class="metric-card" style="text-align:center">
            <div class="stat-label">{score_label}</div>
            <div class="stat-number" style="color:{score_color}">{quality_score:.0f}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card" style="text-align:center">
            <div class="stat-label">{records_label}</div>
            <div class="stat-number">{total_records:,}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card" style="text-align:center">
            <div class="stat-label">{missing_label}</div>
            <div class="stat-number" style="font-size:1.5rem">{overall_missing:.1f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="metric-card" style="text-align:center">
            <div class="stat-label">{items_label}</div>
            <div class="stat-number">{len(quality_data)}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # 详细数据表
    detail_title = "### 📋 Detailed Quality Report" if lang == 'en' else "### 📋 详细质量报告"
    st.markdown(detail_title)
    
    if quality_data:
        quality_df = pd.DataFrame(quality_data)
        st.dataframe(
            quality_df, 
            use_container_width=True, 
            hide_index=True,
        )
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # 可视化分析
    tab1_label = "📊 Missing Rate Chart" if lang == 'en' else "📊 缺失率图表"
    tab2_label = "📈 Value Distribution" if lang == 'en' else "📈 数值分布"
    tab3_label = "⏱️ Time Coverage" if lang == 'en' else "⏱️ 时间覆盖"
    tab1, tab2, tab3 = st.tabs([tab1_label, tab2_label, tab3_label])
    
    with tab1:
        # 缺失率条形图
        try:
            import plotly.express as px
            
            missing_data = []
            for concept, df in st.session_state.loaded_concepts.items():
                if isinstance(df, pd.DataFrame):
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    # 排除ID列和所有可能的时间列
                    exclude_cols = ['stay_id', 'hadm_id', 'icustay_id', 'time', 'index',
                                   'charttime', 'starttime', 'endtime', 'datetime', 'timestamp',
                                   'patientunitstayid', 'admissionid', 'patientid']
                    value_cols = [c for c in numeric_cols if c not in exclude_cols]
                    if value_cols:
                        # 1. 计算 NA 缺失率
                        na_rate = df[value_cols].isna().mean().mean() * 100
                        
                        # 2. 计算覆盖率调整 (与上方详情表逻辑保持一致)
                        n_records = len(df)
                        n_patients = df[st.session_state.id_col].nunique() if st.session_state.id_col in df.columns else 0
                        final_missing_rate = na_rate
                        
                        if n_patients > 0:
                            records_per_patient = n_records / n_patients
                            expected_records = 72 if concept in ['hr', 'map', 'sbp', 'resp', 'spo2', 'temp'] else 24
                            coverage_rate = min(100, (records_per_patient / expected_records) * 100)
                            # 如果 NA 率为 0，则主要反映覆盖率不足
                            final_missing_rate = na_rate + (100 - coverage_rate) * 0.3 if na_rate == 0 else na_rate

                        missing_rate_label = "Missing Rate (%)" if lang == 'en' else "空值比例 (%)"
                        records_label_2 = "Records" if lang == 'en' else "记录数"
                        
                        missing_data.append({
                            'Concept': concept, 
                            missing_rate_label: final_missing_rate,
                            records_label_2: len(df)
                        })
            
            if missing_data:
                missing_df = pd.DataFrame(missing_data)
                missing_rate_col = "Missing Rate (%)" if lang == 'en' else "空值比例 (%)"
                missing_df = missing_df.sort_values(missing_rate_col, ascending=True)
                
                # 检查是否全是0
                if missing_df[missing_rate_col].sum() == 0:
                    # 所有数据无缺失，显示成功信息
                    good_msg = "✅ Excellent data quality: No missing values in numeric columns" if lang == 'en' else "✅ 数据质量良好：所有数值列均无空值 (NA/NaN)"
                    st.success(good_msg)
                    
                    # 显示概念列表
                    concepts_loaded = f"**Loaded Concepts ({len(missing_df)} total):**" if lang == 'en' else f"**已加载概念 ({len(missing_df)} 个)：**"
                    st.markdown(concepts_loaded)
                    concept_list = ", ".join(missing_df['Concept'].tolist())
                    st.write(concept_list)
                else:
                    # 有缺失值，绘制条形图
                    chart_title = '📉 Missing Rate Analysis by Concept' if lang == 'en' else '📉 各 Concept 空值比例分析'
                    fig = px.bar(
                        missing_df, x=missing_rate_col, y='Concept',
                        orientation='h',
                        title=chart_title,
                        color=missing_rate_col,
                        color_continuous_scale=['#28a745', '#ffc107', '#dc3545'],
                        hover_data=[records_label_2 if lang == 'en' else '记录数']
                    )
                    fig.update_layout(
                        template="plotly_white",
                        height=max(300, len(missing_data) * 40),
                        showlegend=False,
                        yaxis_title="",
                        margin=dict(l=100, r=30, t=50, b=50),
                    )
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            err_msg = f"Chart rendering failed: {e}" if lang == 'en' else f"图表渲染失败: {e}"
            st.warning(err_msg)
    
    with tab2:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            select_concept_label = "Select Concept" if lang == 'en' else "选择 Concept"
            concept = st.selectbox(
                select_concept_label,
                options=list(st.session_state.loaded_concepts.keys()),
                key="quality_concept"
            )
        
        with col2:
            if concept:
                df = st.session_state.loaded_concepts[concept]
                
                if isinstance(df, pd.DataFrame):
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    non_id_cols = [c for c in numeric_cols if c not in ['stay_id', 'hadm_id', 'time', 'index']]
                    
                    if non_id_cols:
                        try:
                            import plotly.express as px
                            import plotly.graph_objects as go
                            
                            value_col = non_id_cols[0]
                            
                            dist_title = f"📊 {concept.upper()} Value Distribution" if lang == 'en' else f"📊 {concept.upper()} 数值分布"
                            fig = px.histogram(
                                df, x=value_col, nbins=50,
                                title=dist_title,
                                marginal="box"
                            )
                            fig.update_layout(
                                template="plotly_white",
                                height=400,
                                showlegend=False,
                            )
                            fig.update_traces(marker_color='#1f77b4')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 统计摘要
                            summary_label = "**Statistical Summary:**" if lang == 'en' else "**统计摘要:**"
                            st.markdown(summary_label)
                            col_a, col_b, col_c, col_d, col_e = st.columns(5)
                            col_a.metric("Min", f"{df[value_col].min():.2f}")
                            col_b.metric("Max", f"{df[value_col].max():.2f}")
                            col_c.metric("Mean", f"{df[value_col].mean():.2f}")
                            col_d.metric("Median", f"{df[value_col].median():.2f}")
                            col_e.metric("Std", f"{df[value_col].std():.2f}")
                            
                        except Exception as e:
                            err_msg = f"Distribution chart rendering failed: {e}" if lang == 'en' else f"分布图渲染失败: {e}"
                            st.warning(err_msg)
    
    with tab3:
        time_coverage = []
        for concept, df in st.session_state.loaded_concepts.items():
            if isinstance(df, pd.DataFrame) and 'time' in df.columns:
                min_time = df['time'].min()
                max_time = df['time'].max()
                time_span = max_time - min_time
                
                # 计算平均采样间隔
                if st.session_state.id_col in df.columns:
                    avg_interval = df.groupby(st.session_state.id_col)['time'].apply(
                        lambda x: x.diff().mean() if len(x) > 1 else 0
                    ).mean()
                else:
                    avg_interval = 0
                
                start_label = "Start Time" if lang == 'en' else "起始时间"
                end_label = "End Time" if lang == 'en' else "结束时间"
                span_label = "Time Span" if lang == 'en' else "时间跨度"
                interval_label = "Avg Interval" if lang == 'en' else "平均间隔"
                
                time_coverage.append({
                    'Concept': concept,
                    start_label: f"{min_time:.1f}h",
                    end_label: f"{max_time:.1f}h",
                    span_label: f"{time_span:.1f}h",
                    interval_label: f"{avg_interval:.2f}h" if avg_interval > 0 else "-",
                })
        
        if time_coverage:
            coverage_df = pd.DataFrame(time_coverage)
            st.dataframe(
                coverage_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Concept": st.column_config.TextColumn("📋 Concept"),
                    "起始时间": st.column_config.TextColumn("⏰ 起始"),
                    "结束时间": st.column_config.TextColumn("⏰ 结束"),
                    "时间跨度": st.column_config.TextColumn("📏 跨度"),
                    "平均间隔": st.column_config.TextColumn("⏱️ 间隔"),
                }
            )


def render_cohort_comparison_page():
    """渲染队列对比可视化页面 - 基于侧边栏筛选的患者进行分组对比。"""
    lang = st.session_state.get('language', 'en')
    
    page_title = "📊 Cohort Comparison" if lang == 'en' else "📊 队列对比分析"
    st.markdown(f"## {page_title}")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # 检查是否有数据路径
    data_path = st.session_state.get('data_path')
    database = st.session_state.get('database', 'miiv')
    
    if not data_path or not Path(data_path).exists():
        if lang == 'en':
            st.warning("👈 Please configure data source in sidebar first (Step 1)")
        else:
            st.warning("👈 请先在侧边栏配置数据源（步骤1）")
        return
    
    # 检查是否已经加载了数据
    loaded_concepts = st.session_state.get('loaded_concepts', [])
    patient_ids = st.session_state.get('patient_ids', [])
    all_patient_count = st.session_state.get('all_patient_count', 0)
    
    # 优先使用特征数据中的患者ID（更准确）
    if 'concept_results' in st.session_state and st.session_state.concept_results:
        # 从加载的数据中提取实际患者ID
        actual_patient_ids = set()
        for concept_name, df in st.session_state.concept_results.items():
            if df is not None and 'stay_id' in df.columns:
                actual_patient_ids.update(df['stay_id'].unique())
        if actual_patient_ids:
            patient_ids = list(actual_patient_ids)
            all_patient_count = len(patient_ids)
    
    if not patient_ids or all_patient_count == 0:
        if lang == 'en':
            st.info("""
            **📋 How to use Cohort Comparison:**
            
            1. Go to sidebar **Step 1** to configure data source
            2. Enable **Step 2: Cohort Selection** to filter patients  
            3. Select features in **Step 3** and click **Load Data**
            4. Return here to compare patient subgroups
            
            The comparison will be based on patients you loaded in the Data Viewer tab.
            """)
        else:
            st.info("""
            **📋 队列对比使用说明：**
            
            1. 在侧边栏**步骤1**配置数据源
            2. 启用**步骤2：队列筛选**来筛选患者
            3. 在**步骤3**选择特征并点击**加载数据**
            4. 返回此页面进行分组对比
            
            对比将基于您在数据查看器中加载的患者进行。
            """)
        return
    
    # 显示当前数据状态
    if lang == 'en':
        st.success(f"✅ Working with **{all_patient_count:,}** patients from your loaded data")
    else:
        st.success(f"✅ 基于已加载的 **{all_patient_count:,}** 名患者进行对比分析")
    
    # 显示当前筛选条件（如果有）
    cohort_enabled = st.session_state.get('cohort_enabled', False)
    if cohort_enabled:
        cf = st.session_state.get('cohort_filter', {})
        filter_parts = []
        if cf.get('age_min') is not None or cf.get('age_max') is not None:
            age_str = f"Age: {cf.get('age_min', 0)}-{cf.get('age_max', '∞')}" if lang == 'en' else f"年龄: {cf.get('age_min', 0)}-{cf.get('age_max', '∞')}"
            filter_parts.append(age_str)
        if cf.get('first_icu_stay') is not None:
            icu_str = f"First ICU: {'Yes' if cf['first_icu_stay'] else 'No'}" if lang == 'en' else f"首次入ICU: {'是' if cf['first_icu_stay'] else '否'}"
            filter_parts.append(icu_str)
        if cf.get('los_min') is not None:
            los_str = f"LOS ≥ {cf['los_min']}h" if lang == 'en' else f"住院≥{cf['los_min']}h"
            filter_parts.append(los_str)
        if filter_parts:
            filter_info = " | ".join(filter_parts)
            if lang == 'en':
                st.caption(f"📋 Current filters: {filter_info}")
            else:
                st.caption(f"📋 当前筛选条件: {filter_info}")
    
    st.markdown("---")
    
    # 对比模式选择
    compare_mode_label = "Select Comparison Mode" if lang == 'en' else "选择对比模式"
    compare_options = {
        'survival': ('💀 Survived vs Deceased' if lang == 'en' else '💀 存活 vs 死亡'),
        'age': ('👴 Age Groups' if lang == 'en' else '👴 年龄分组'),
        'gender': ('👫 Male vs Female' if lang == 'en' else '👫 男性 vs 女性'),
        'los': ('🏥 Short vs Long Stay' if lang == 'en' else '🏥 短住院 vs 长住院'),
    }
    
    compare_mode = st.radio(
        compare_mode_label,
        options=list(compare_options.keys()),
        format_func=lambda x: compare_options[x],
        horizontal=True
    )
    
    st.markdown("---")
    
    try:
        from pyricu.cohort_visualization import CohortVisualizer
        from pyricu.patient_filter import PatientFilter
        
        viz = CohortVisualizer(database=database, data_path=data_path, language=lang)
        
        # 获取人口统计学数据用于分组
        pf = PatientFilter(database=database, data_path=data_path)
        demographics_df = pf._load_demographics()
        
        # 只保留当前加载的患者
        base_df = demographics_df[demographics_df['patient_id'].isin(patient_ids)]
        
        if len(base_df) == 0:
            if lang == 'en':
                st.warning("No demographic data available for loaded patients.")
            else:
                st.warning("无法获取已加载患者的人口统计学数据。")
            return
        
        group1_ids = []
        group2_ids = []
        group1_name = ""
        group2_name = ""
        show_mortality = True
        
        if compare_mode == 'survival':
            # 存活 vs 死亡
            if 'survived' not in base_df.columns:
                if lang == 'en':
                    st.warning("Survival data not available in demographics.")
                else:
                    st.warning("人口统计学数据中没有存活状态信息。")
                return
            
            survived_df = base_df[base_df['survived'] == 1]
            deceased_df = base_df[base_df['survived'] == 0]
            
            group1_ids = survived_df['patient_id'].tolist()
            group2_ids = deceased_df['patient_id'].tolist()
            group1_name = 'Survived' if lang == 'en' else '存活'
            group2_name = 'Deceased' if lang == 'en' else '死亡'
            show_mortality = False  # 分组本身就是按存活分的
            
        elif compare_mode == 'age':
            # 年龄分组
            age_threshold = st.slider(
                "Age Threshold" if lang == 'en' else "年龄阈值",
                min_value=30, max_value=90, value=65, step=5
            )
            
            young_df = base_df[base_df['age'] < age_threshold]
            old_df = base_df[base_df['age'] >= age_threshold]
            
            group1_ids = young_df['patient_id'].tolist()
            group2_ids = old_df['patient_id'].tolist()
            group1_name = f'Age < {age_threshold}' if lang == 'en' else f'年龄 < {age_threshold}'
            group2_name = f'Age ≥ {age_threshold}' if lang == 'en' else f'年龄 ≥ {age_threshold}'
            
        elif compare_mode == 'gender':
            # 性别分组
            if 'gender' not in base_df.columns:
                if lang == 'en':
                    st.warning("Gender data not available in demographics.")
                else:
                    st.warning("人口统计学数据中没有性别信息。")
                return
            
            male_df = base_df[base_df['gender'] == 'M']
            female_df = base_df[base_df['gender'] == 'F']
            
            group1_ids = male_df['patient_id'].tolist()
            group2_ids = female_df['patient_id'].tolist()
            group1_name = 'Male' if lang == 'en' else '男性'
            group2_name = 'Female' if lang == 'en' else '女性'
            
        elif compare_mode == 'los':
            # 住院时长分组
            if 'los_hours' not in base_df.columns:
                if lang == 'en':
                    st.warning("Length of stay data not available in demographics.")
                else:
                    st.warning("人口统计学数据中没有住院时长信息。")
                return
            
            # 使用中位数作为阈值
            median_los = base_df['los_hours'].median()
            los_threshold = st.slider(
                "LOS Threshold (hours)" if lang == 'en' else "住院时长阈值（小时）",
                min_value=24, max_value=int(min(500, base_df['los_hours'].quantile(0.95))),
                value=int(median_los), step=12
            )
            
            short_df = base_df[base_df['los_hours'] < los_threshold]
            long_df = base_df[base_df['los_hours'] >= los_threshold]
            
            group1_ids = short_df['patient_id'].tolist()
            group2_ids = long_df['patient_id'].tolist()
            group1_name = f'LOS < {los_threshold}h' if lang == 'en' else f'住院 < {los_threshold}h'
            group2_name = f'LOS ≥ {los_threshold}h' if lang == 'en' else f'住院 ≥ {los_threshold}h'
        
        # 显示分组统计
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(group1_name, f"{len(group1_ids):,}")
        with col2:
            st.metric(group2_name, f"{len(group2_ids):,}")
        with col3:
            total = len(group1_ids) + len(group2_ids)
            pct1 = len(group1_ids) / total * 100 if total > 0 else 0
            ratio_label = "Ratio" if lang == 'en' else "比例"
            st.metric(ratio_label, f"{pct1:.1f}% / {100-pct1:.1f}%")
        
        if len(group1_ids) == 0 or len(group2_ids) == 0:
            if lang == 'en':
                st.warning("One of the groups has no patients. Please adjust the criteria.")
            else:
                st.warning("其中一个分组没有患者，请调整分组条件。")
            return
        
        # 创建对比可视化
        st.markdown("---")
        viz_title = "📊 Demographics Comparison" if lang == 'en' else "📊 人口统计学对比"
        st.markdown(f"### {viz_title}")
        
        fig = viz.compare_demographics(
            group1_ids=group1_ids,
            group2_ids=group2_ids,
            group1_name=group1_name,
            group2_name=group2_name,
            show_mortality=show_mortality
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 统计表格 (TableOne风格)
        summary_title = "📋 Baseline Characteristics (TableOne)" if lang == 'en' else "📋 基线特征对比 (TableOne)"
        st.markdown(f"### {summary_title}")
        summary_df = viz.create_summary_table(
            group1_ids=group1_ids,
            group2_ids=group2_ids,
            group1_name=group1_name,
            group2_name=group2_name,
            show_pvalue=True
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # 添加统计说明
        if lang == 'en':
            stats_note = "**Statistical Methods:** Mann-Whitney U test for continuous variables, Chi-square test for categorical variables."
        else:
            stats_note = "**统计方法：** 连续变量使用Mann-Whitney U检验，分类变量使用卡方检验。"
        st.caption(stats_note)
        
    except ImportError as e:
        if lang == 'en':
            st.error(f"Required modules not available: {e}")
        else:
            st.error(f"缺少必要模块: {e}")
    except Exception as e:
        if lang == 'en':
            st.error(f"Error in cohort comparison: {e}")
        else:
            st.error(f"队列对比出错: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_convert_dialog():
    """Render CSV to Parquet conversion dialog."""
    lang = st.session_state.get('language', 'en')
    source_path = st.session_state.get('convert_source_path', '')
    
    dialog_title = "## 🔄 CSV to Parquet Conversion" if lang == 'en' else "## 🔄 CSV 转换为 Parquet"
    st.markdown(dialog_title)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    source_info = f"📁 Source directory: `{source_path}`" if lang == 'en' else f"📁 源目录: `{source_path}`"
    st.info(source_info)
    
    # 显示系统内存信息
    available_mem = get_available_memory_gb()
    mem_info = f"💻 System: {SYSTEM_MEMORY_GB:.1f}GB total, {available_mem:.1f}GB available" if lang == 'en' else f"💻 系统内存: 共 {SYSTEM_MEMORY_GB:.1f}GB，可用 {available_mem:.1f}GB"
    st.caption(mem_info)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 目标目录（默认同目录）
        target_label = "Parquet Output Directory" if lang == 'en' else "Parquet输出目录"
        target_help = "Converted Parquet files will be saved to this directory" if lang == 'en' else "转换后的Parquet文件将保存到此目录"
        target_path = st.text_input(
            target_label,
            value=source_path,
            help=target_help
        )
    
    with col2:
        # 内存限制选项
        mem_label = "Memory Limit (GB)" if lang == 'en' else "内存限制 (GB)"
        mem_help = "Maximum memory to use during conversion. Lower = slower but safer. Default: 8GB" if lang == 'en' else "转换时使用的最大内存。数值越低越安全但更慢。默认: 8GB"
        
        # 初始化 session state
        if 'convert_memory_limit' not in st.session_state:
            st.session_state.convert_memory_limit = min(8, DEFAULT_MEMORY_LIMIT_GB)
        
        memory_limit = st.slider(
            mem_label,
            min_value=2,
            max_value=min(32, int(SYSTEM_MEMORY_GB)),
            value=int(st.session_state.convert_memory_limit),
            step=1,
            help=mem_help
        )
        st.session_state.convert_memory_limit = memory_limit
    
    with col3:
        # 转换选项
        st.markdown("&nbsp;")  # 对齐
        overwrite_label = "Overwrite existing" if lang == 'en' else "覆盖已存在文件"
        overwrite = st.checkbox(overwrite_label, value=False)
    
    # 根据内存限制计算推荐的块大小
    chunk_size = _calculate_chunk_size(memory_limit)
    chunk_info = f"📊 Chunk size: {chunk_size:,} rows (based on {memory_limit}GB limit)" if lang == 'en' else f"📊 分块大小: {chunk_size:,} 行（基于 {memory_limit}GB 限制）"
    st.caption(chunk_info)
    
    # 扫描可转换文件
    if source_path and Path(source_path).exists():
        csv_files = list(Path(source_path).rglob('*.csv')) + list(Path(source_path).rglob('*.csv.gz'))
        found_msg = f"**Found {len(csv_files)} CSV files to convert**" if lang == 'en' else f"**发现 {len(csv_files)} 个CSV文件可转换**"
        st.markdown(found_msg)
        
        view_label = "View file list" if lang == 'en' else "查看文件列表"
        with st.expander(view_label, expanded=False):
            for f in csv_files[:20]:
                size_mb = f.stat().st_size / (1024 * 1024)
                st.caption(f"• {f.name} ({size_mb:.1f} MB)")
            if len(csv_files) > 20:
                more_msg = f"... and {len(csv_files) - 20} more files" if lang == 'en' else f"... 及其他 {len(csv_files) - 20} 个文件"
                st.caption(more_msg)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        start_label = "🚀 Start Conversion" if lang == 'en' else "🚀 开始转换"
        if st.button(start_label, type="primary", width="stretch"):
            if not target_path or not Path(target_path).exists():
                err_msg = "❌ Please set a valid output directory" if lang == 'en' else "❌ 请设置有效的输出目录"
                st.error(err_msg)
            else:
                # 不使用 spinner，直接显示进度
                st.info("🔄 Starting conversion..." if lang == 'en' else "🔄 开始转换...")
                
                # 使用用户设置的内存限制
                mem_limit = st.session_state.get('convert_memory_limit', 8)
                success, failed = convert_csv_to_parquet(source_path, target_path, overwrite, memory_limit_gb=mem_limit)
                
                if success > 0:
                    success_msg = f"✅ Successfully converted {success} files" if lang == 'en' else f"✅ 成功转换 {success} 个文件"
                    st.success(success_msg)
                    st.session_state.path_validated = True
                    st.session_state.data_path = target_path
                if failed > 0:
                    fail_msg = f"⚠️ {failed} files failed to convert" if lang == 'en' else f"⚠️ {failed} 个文件转换失败"
                    st.warning(fail_msg)
                    
                st.session_state.show_convert_dialog = False
                st.rerun()
    
    with col2:
        cancel_label = "❌ Cancel" if lang == 'en' else "❌ 取消"
        if st.button(cancel_label, width="stretch"):
            st.session_state.show_convert_dialog = False
            st.rerun()
    
    with col3:
        use_csv_label = "📂 Use Original CSV" if lang == 'en' else "📂 使用原始CSV"
        if st.button(use_csv_label, width="stretch"):
            st.session_state.data_path = source_path
            st.session_state.path_validated = True
            st.session_state.show_convert_dialog = False
            csv_info = "Will use CSV format (slower loading)" if lang == 'en' else "将使用CSV格式（加载较慢）"
            st.info(csv_info)
            st.rerun()


def _calculate_chunk_size(memory_limit_gb: int) -> int:
    """根据内存限制计算合适的分块大小。
    
    假设每行平均约 1KB 内存占用，预留 50% 内存给其他操作。
    """
    # 每GB内存大约可处理 500,000 行（保守估计）
    rows_per_gb = 500_000
    # 使用 50% 的内存限制用于数据加载
    chunk_size = int(memory_limit_gb * rows_per_gb * 0.5)
    # 限制在合理范围内
    return max(50_000, min(5_000_000, chunk_size))


def convert_csv_to_parquet(source_dir: str, target_dir: str, overwrite: bool = False, memory_limit_gb: int = 8) -> tuple:
    """将目录下的CSV文件转换为Parquet格式。
    
    使用 DataConverter 类进行专业转换，支持大表分片。
    
    Args:
        source_dir: 源目录
        target_dir: 目标目录
        overwrite: 是否覆盖已存在的文件
        memory_limit_gb: 内存限制（GB）
    """
    import gc
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 根据内存限制计算块大小
    chunk_size = _calculate_chunk_size(memory_limit_gb)
    
    # 尝试使用专业的 DataConverter
    try:
        from pyricu.data_converter import DataConverter
        
        # 检测数据库类型
        database = _detect_database_type(source_path)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        memory_text = st.empty()
        
        status_text.info(f"🔍 Detecting database type: {database.upper() if database else 'Unknown'}")
        
        # 创建转换器
        converter = DataConverter(
            data_path=source_path,
            database=database,
            chunk_size=chunk_size,
            parallel_workers=max(1, min(4, WORKERS or 2)),  # 限制并行数
            verbose=True,
        )
        
        # 获取需要转换的文件
        csv_files = converter._get_csv_files()
        
        if not csv_files:
            status_text.warning("⚠️ No CSV files found to convert")
            return 0, 0
        
        status_text.info(f"📊 Found {len(csv_files)} CSV files to convert")
        
        success = 0
        failed = 0
        skipped = 0
        
        for idx, csv_file in enumerate(csv_files):
            try:
                # 显示内存状态
                current_mem = get_available_memory_gb()
                memory_text.caption(f"💾 Available memory: {current_mem:.1f} GB")
                
                # 检查是否需要转换
                needs_convert, reason = converter._is_conversion_needed(csv_file)
                
                if not needs_convert and not overwrite:
                    status_text.caption(f"⏭️ Skip: {csv_file.name} ({reason})")
                    skipped += 1
                    progress_bar.progress((idx + 1) / len(csv_files))
                    continue
                
                file_size_mb = csv_file.stat().st_size / (1024 * 1024)
                status_text.markdown(f"**Converting**: `{csv_file.name}` ({file_size_mb:.1f}MB) ({idx+1}/{len(csv_files)})")
                
                # 使用 DataConverter 的转换方法（支持分片）
                result = converter._convert_file(csv_file)
                
                if result.get('status') == 'completed':
                    shards = result.get('shards', 0)
                    rows = result.get('row_count', 0)
                    if shards > 0:
                        status_text.caption(f"✅ {csv_file.name}: {rows:,} rows → {shards} shards")
                    else:
                        status_text.caption(f"✅ {csv_file.name}: {rows:,} rows")
                    success += 1
                else:
                    failed += 1
                    status_text.caption(f"❌ {csv_file.name}: {result.get('error', 'Unknown error')}")
                
                gc.collect()
                
            except Exception as e:
                failed += 1
                status_text.caption(f"❌ Failed: {csv_file.name} - {str(e)[:100]}")
                gc.collect()
            
            progress_bar.progress((idx + 1) / len(csv_files))
        
        progress_bar.progress(1.0)
        
        if skipped > 0:
            status_text.info(f"📊 Completed: {success} converted, {skipped} skipped, {failed} failed")
        else:
            status_text.empty()
        
        memory_text.empty()
        gc.collect()
        
        return success + skipped, failed
        
    except ImportError:
        # 回退到简单转换
        return _simple_convert_csv_to_parquet(source_dir, target_dir, overwrite, memory_limit_gb)
    except Exception as e:
        # 捕获所有其他错误并显示
        st.error(f"❌ Conversion error: {str(e)}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())
        return 0, 1


def _detect_database_type(path: Path) -> str:
    """检测数据库类型"""
    path_str = str(path).lower()
    
    if 'eicu' in path_str:
        return 'eicu'
    elif 'miiv' in path_str or 'mimic' in path_str:
        return 'miiv'
    elif 'aumc' in path_str or 'amsterdam' in path_str:
        return 'aumc'
    elif 'hirid' in path_str:
        return 'hirid'
    
    # 尝试从文件名检测
    files = list(path.rglob('*.csv')) + list(path.rglob('*.csv.gz'))
    file_names = [f.name.lower() for f in files]
    
    if any('patient.csv' in f for f in file_names):
        return 'eicu'
    elif any('icustays.csv' in f for f in file_names):
        return 'miiv'
    elif any('admissions.csv' in f and 'numericitems.csv' in ' '.join(file_names) for f in file_names):
        return 'aumc'
    
    return 'unknown'


def _simple_convert_csv_to_parquet(source_dir: str, target_dir: str, overwrite: bool = False, memory_limit_gb: int = 8) -> tuple:
    """简单的 CSV 转 Parquet（回退方案）"""
    import gc
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    csv_files = list(source_path.rglob('*.csv')) + list(source_path.rglob('*.csv.gz'))
    csv_files.sort(key=lambda f: f.stat().st_size)
    
    chunk_size = _calculate_chunk_size(memory_limit_gb)
    large_file_threshold = 100 * 1024 * 1024
    
    success = 0
    failed = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    memory_text = st.empty()
    
    for idx, csv_file in enumerate(csv_files):
        try:
            current_mem = get_available_memory_gb()
            memory_text.caption(f"💾 Available memory: {current_mem:.1f} GB")
            
            rel_path = csv_file.relative_to(source_path)
            parquet_name = rel_path.stem.replace('.csv', '') + '.parquet'
            parquet_file = target_path / rel_path.parent / parquet_name
            
            if parquet_file.exists() and not overwrite:
                status_text.caption(f"⏭️ Skip: {csv_file.name} (exists)")
                success += 1  # 跳过的也算成功
                progress_bar.progress((idx + 1) / len(csv_files))
                continue
            
            parquet_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_size = csv_file.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            status_text.markdown(f"**Converting**: `{csv_file.name}` ({file_size_mb:.1f}MB) ({idx+1}/{len(csv_files)})")
            
            if file_size > large_file_threshold:
                _convert_large_csv(csv_file, parquet_file, chunk_size)
            else:
                df = pd.read_csv(csv_file, low_memory=True)
                df.to_parquet(parquet_file, index=False)
                del df
            
            success += 1
            gc.collect()
            
        except Exception as e:
            failed += 1
            status_text.caption(f"❌ Failed: {csv_file.name} - {str(e)[:50]}")
            gc.collect()
        
        progress_bar.progress((idx + 1) / len(csv_files))
    
    progress_bar.progress(1.0)
    status_text.empty()
    memory_text.empty()
    gc.collect()
    
    return success, failed


def _convert_large_csv(csv_file: Path, parquet_file: Path, chunk_size: int):
    """分块转换大型CSV文件为Parquet。
    
    使用 PyArrow 的增量写入方式，避免一次性加载全部数据到内存。
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import gc
    
    # 使用分块读取
    chunks = pd.read_csv(csv_file, chunksize=chunk_size, low_memory=True)
    
    writer = None
    total_rows = 0
    
    try:
        for chunk in chunks:
            table = pa.Table.from_pandas(chunk)
            
            if writer is None:
                # 首次写入，创建 ParquetWriter
                writer = pq.ParquetWriter(str(parquet_file), table.schema)
            
            writer.write_table(table)
            total_rows += len(chunk)
            
            # 释放内存
            del chunk
            del table
            gc.collect()
            
    finally:
        if writer:
            writer.close()


def _generate_cohort_prefix() -> str:
    """根据队列筛选条件生成文件名前缀。
    
    Returns:
        筛选条件前缀字符串，如 "age18-80_firstICU_los24h"，无筛选则返回空字符串
    """
    if not st.session_state.get('cohort_enabled', False):
        return ""
    
    cf = st.session_state.get('cohort_filter', {})
    parts = []
    
    # 年龄
    age_min = cf.get('age_min')
    age_max = cf.get('age_max')
    if age_min is not None or age_max is not None:
        age_str = f"age{int(age_min) if age_min else 0}-{int(age_max) if age_max else 'inf'}"
        parts.append(age_str)
    
    # 首次入ICU
    first_icu = cf.get('first_icu_stay')
    if first_icu is True:
        parts.append("firstICU")
    elif first_icu is False:
        parts.append("readmit")
    
    # 住院时长
    los_min = cf.get('los_min')
    if los_min is not None and los_min > 0:
        parts.append(f"los{int(los_min)}h")
    
    # 性别
    gender = cf.get('gender')
    if gender is not None:
        parts.append(f"sex{gender}")
    
    # 存活状态
    survived = cf.get('survived')
    if survived is True:
        parts.append("survived")
    elif survived is False:
        parts.append("deceased")
    
    # Sepsis
    has_sepsis = cf.get('has_sepsis')
    if has_sepsis is True:
        parts.append("sepsis")
    elif has_sepsis is False:
        parts.append("noSepsis")
    
    return "_".join(parts)


def execute_sidebar_export():
    """执行侧边栏触发的数据导出（直接导出到本地目录，带进度条）。"""
    from datetime import datetime
    
    lang = st.session_state.get('language', 'en')
    export_path = st.session_state.get('export_path', '')
    export_format = st.session_state.get('export_format', 'Parquet').lower()
    selected_concepts = st.session_state.get('selected_concepts', [])
    use_mock = st.session_state.use_mock_data
    
    if not export_path or not Path(export_path).exists():
        err_msg = "❌ Please set a valid export path first" if lang == 'en' else "❌ 请先设置有效的导出路径"
        st.error(err_msg)
        return
    
    if not selected_concepts:
        err_msg = "❌ Please select features to export first" if lang == 'en' else "❌ 请先选择要导出的特征"
        st.error(err_msg)
        return
    
    try:
        export_title = "📤 Export Progress" if lang == 'en' else "📤 导出进度"
        st.markdown(f"### {export_title}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 直接使用用户设置的导出路径（已包含数据库子目录）
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        total_concepts = len(selected_concepts)
        
        # 创建进度条和状态显示
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        if use_mock:
            # 生成模拟数据并导出
            gen_msg = "**Generating mock data...**" if lang == 'en' else "**正在生成模拟数据...**"
            status_text.markdown(gen_msg)
            params = st.session_state.get('mock_params', {'n_patients': 10, 'hours': 72})
            all_mock_data, patient_ids = generate_mock_data(**params)
            
            
            # 🔧 根据用户选择的 concepts 过滤数据
            data = {}
            for concept in selected_concepts:
                if concept in all_mock_data:
                    data[concept] = all_mock_data[concept]
            
            
            # 显示加载情况
            loaded_count = len(data)
            if loaded_count < len(selected_concepts):
                missing = [c for c in selected_concepts if c not in all_mock_data]
                skip_msg = f"⚠️ {len(missing)} concepts not in mock data: {', '.join(missing[:5])}" if lang == 'en' else f"⚠️ 模拟数据中不存在 {len(missing)} 个概念: {', '.join(missing[:5])}"
                st.warning(skip_msg)
            
            progress_bar.progress(0.3)
        else:
            # 加载真实数据并导出（批量并行加载）
            from pyricu import load_concepts
            import os
            
            # 批量并行加载所有特征
            batch_msg = f"**Loading {total_concepts} features (batch mode)...**" if lang == 'en' else f"**批量加载 {total_concepts} 个特征...**"
            status_text.markdown(batch_msg)
            
            # 🚀 性能优化：参照 extract_baseline_features.py 的配置
            patient_limit = st.session_state.get('patient_limit', 0)  # 导出默认不限制
            
            # 获取患者ID过滤器
            patient_ids_filter = None
            id_col = 'stay_id'
            if patient_limit and patient_limit > 0:
                try:
                    data_path = Path(st.session_state.data_path)
                    database = st.session_state.get('database', 'miiv')
                    id_col_map = {'miiv': 'stay_id', 'eicu': 'patientunitstayid', 'aumc': 'admissionid', 'hirid': 'patientid'}
                    id_col = id_col_map.get(database, 'stay_id')
                    
                    for f in ['icustays.parquet', 'patient.parquet', 'admissions.parquet']:
                        fp = data_path / f
                        if fp.exists():
                            icustays_df = pd.read_parquet(fp, columns=[id_col] if id_col else None)
                            if id_col in icustays_df.columns:
                                all_ids = icustays_df[id_col].unique().tolist()
                                sample_ids = all_ids[:patient_limit] if len(all_ids) > patient_limit else all_ids
                                patient_ids_filter = {id_col: sample_ids}
                                break
                except Exception:
                    pass
            
            # 🚀 智能并行配置：根据系统资源和患者数量动态调整
            num_patients = len(patient_ids_filter.get(id_col, [])) if patient_ids_filter else None
            parallel_workers, parallel_backend = get_optimal_parallel_config(num_patients, task_type='export')
            
            # 显示系统资源信息（调试用）
            resources = get_system_resources()
            perf_msg = f"🚀 System: {resources['cpu_count']} cores, {resources['total_memory_gb']}GB RAM → Using {parallel_workers} workers ({parallel_backend})" if lang == 'en' else f"🚀 系统: {resources['cpu_count']} 核心, {resources['total_memory_gb']}GB 内存 → 使用 {parallel_workers} 并行 ({parallel_backend})"
            st.info(perf_msg)
            
            try:
                # 🔧 逐个加载概念，跳过不可用的（某些概念在特定数据库中没有数据源配置）
                data = {}
                failed_concepts = []
                
                for i, concept in enumerate(selected_concepts):
                    try:
                        load_kwargs = {
                            'data_path': st.session_state.data_path,
                            'database': st.session_state.get('database'),
                            'concepts': [concept],
                            'verbose': False,
                            'merge': False,
                            'concept_workers': 1,
                            'parallel_workers': parallel_workers,
                            'parallel_backend': parallel_backend,
                        }
                        if patient_ids_filter:
                            load_kwargs['patient_ids'] = patient_ids_filter
                        
                        result = load_concepts(**load_kwargs)
                        
                        # 处理返回结果（可能是 dict 或 DataFrame）
                        if isinstance(result, dict):
                            for cname, df in result.items():
                                # 🔧 处理各种返回类型
                                if hasattr(df, 'to_pandas'):
                                    df = df.to_pandas()
                                elif hasattr(df, 'dataframe'):
                                    df = df.dataframe()
                                elif hasattr(df, 'data') and isinstance(df.data, pd.DataFrame):
                                    df = df.data
                                
                                if isinstance(df, pd.DataFrame) and len(df) > 0:
                                    data[cname] = df
                                elif isinstance(df, pd.Series):
                                    data[cname] = df.to_frame().reset_index()
                        elif isinstance(result, pd.DataFrame):
                            # 单概念加载返回 DataFrame
                            if len(result) > 0:
                                data[concept] = result
                        
                        # 更新进度
                        progress_bar.progress(0.1 + 0.4 * (i + 1) / total_concepts)
                        
                    except Exception as e:
                        failed_concepts.append(concept)
                        continue  # 跳过失败的概念，继续加载其他的
                
                progress_bar.progress(0.5)
                if failed_concepts:
                    skip_msg = f"⚠️ Skipped {len(failed_concepts)} unavailable: {', '.join(failed_concepts[:5])}" if lang == 'en' else f"⚠️ 跳过 {len(failed_concepts)} 个不可用: {', '.join(failed_concepts[:5])}"
                    st.warning(skip_msg)
                loaded_msg = f"✅ Loaded {len(data)}/{total_concepts} features" if lang == 'en' else f"✅ 已加载 {len(data)}/{total_concepts} 个特征"
                status_text.markdown(loaded_msg)
                
            except Exception as e:
                warn_msg = f"⚠️ Batch loading failed: {e}" if lang == 'en' else f"⚠️ 批量加载失败: {e}"
                st.warning(warn_msg)
                data = {}
        
        # 按模块分组导出（将同一分组的特征合并为宽表）
        merge_msg = "**Merging and exporting by module...**" if lang == 'en' else "**正在按模块合并导出...**"
        status_text.markdown(merge_msg)
        
        # 反向映射：concept -> group_key（英文key用于文件名）
        concept_to_group = {}
        
        # 🔧 智能调整分组优先级
        # 默认使用 定义顺序，但如果检测到用户只使用了 SOFA-1 相关的 Sepsis
        # 则调整优先级，确保共享概念被归类到 Sepsis-3 (SOFA-1) 组
        group_priority = list(CONCEPT_GROUPS_INTERNAL.keys())
        loaded_keys = set(data.keys())
        if 'sep3_sofa1' in loaded_keys and 'sep3_sofa2' not in loaded_keys:
            # Sepsis-3 SOFA-1 存在但 SOFA-2 不存在 => 优先使用 SOFA-1 组
            if 'sepsis3_sofa1' in group_priority and 'sepsis3_sofa2' in group_priority:
                # 交换位置或重建列表，让 sofa1 排在 sofa2 前面
                group_priority.remove('sepsis3_sofa1')
                idx_sofa2 = group_priority.index('sepsis3_sofa2')
                group_priority.insert(idx_sofa2, 'sepsis3_sofa1')
        
        for group_key in group_priority:
            concepts = CONCEPT_GROUPS_INTERNAL[group_key]
            for c in concepts:
                if c not in concept_to_group:  # 优先使用第一个分组
                    concept_to_group[c] = group_key
        
        # 按分组聚合数据
        grouped_data = {}
        for concept_name, df in data.items():
            if not isinstance(df, pd.DataFrame) or len(df) == 0:
                continue
            
            group_key = concept_to_group.get(concept_name, 'other')
            
            if group_key not in grouped_data:
                grouped_data[group_key] = {}
            
            grouped_data[group_key][concept_name] = df
        
        # 导出合并后的分组数据（宽表格式）
        total_groups = len(grouped_data)
        for idx, (group_name, concept_dfs) in enumerate(grouped_data.items()):
            export_group_msg = f"**Exporting**: `{group_name}` ({idx+1}/{total_groups})" if lang == 'en' else f"**正在导出**: `{group_name}` ({idx+1}/{total_groups})"
            status_text.markdown(export_group_msg)
            
            # 将同一分组的所有 concept 合并为宽表
            # 找到共同的 ID 列和时间列
            id_candidates = ['stay_id', 'hadm_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid']
            time_candidates = ['time', 'charttime', 'starttime', 'endtime', 'itemtime']
            
            # 🔧 先统一所有 DataFrame 的时间列名称
            # 不同概念可能使用不同的时间列名（charttime, starttime等）
            # 注意：PyRICU 的时间是相对于 ICU 入院的小时数，不是 datetime
            unified_time_col = 'charttime'  # 统一使用 charttime 作为时间列名
            normalized_concept_dfs = {}
            for cname, cdf in concept_dfs.items():
                cdf = cdf.copy()
                
                # 检查是否已经有统一的时间列
                if unified_time_col in cdf.columns:
                    # 删除其他时间列以避免重复
                    other_time_cols = [tc for tc in time_candidates if tc in cdf.columns and tc != unified_time_col]
                    if other_time_cols:
                        cdf = cdf.drop(columns=other_time_cols)
                else:
                    # 找到当前 DataFrame 的第一个时间列并重命名
                    for tc in time_candidates:
                        if tc in cdf.columns:
                            cdf = cdf.rename(columns={tc: unified_time_col})
                            # 删除其他时间列
                            other_time_cols = [t for t in time_candidates if t in cdf.columns and t != unified_time_col]
                            if other_time_cols:
                                cdf = cdf.drop(columns=other_time_cols)
                            break
                
                # 🔧 不再强制转换时间列类型，保持原始的小时数格式
                # PyRICU 的时间是相对于 ICU 入院的小时数（0, 1, 2, 3...）
                
                normalized_concept_dfs[cname] = cdf
            concept_dfs = normalized_concept_dfs
            
            # 确定这个分组的主键列
            merge_cols = []
            id_col = None
            time_col = None
            
            first_df = list(concept_dfs.values())[0]
            for col in id_candidates:
                if col in first_df.columns:
                    id_col = col
                    merge_cols.append(col)
                    break
            for col in time_candidates:
                if col in first_df.columns:
                    time_col = col
                    merge_cols.append(col)
                    break
            
            if not merge_cols:
                # 没有共同的合并键，简单拼接
                all_dfs = []
                for cname, cdf in concept_dfs.items():
                    cdf = cdf.copy()
                    cdf['_concept'] = cname
                    all_dfs.append(cdf)
                merged_df = pd.concat(all_dfs, ignore_index=True)
            else:
                # 使用 merge 创建宽表
                merged_df = None
                for concept_name, df in concept_dfs.items():
                    # 🔧 确保当前 df 包含所有 merge_cols
                    # 如果缺少某列，跳过合并该概念（改为追加）
                    missing_cols = [c for c in merge_cols if c not in df.columns]
                    if missing_cols:
                        # 该概念缺少合并列，作为独立数据追加
                        if merged_df is None:
                            merged_df = df.copy()
                            # 重命名值列
                            value_cols = [c for c in df.columns if c not in merge_cols]
                            if len(value_cols) == 1:
                                merged_df = merged_df.rename(columns={value_cols[0]: concept_name})
                        else:
                            # 作为独立行追加
                            df_copy = df.copy()
                            df_copy['_concept'] = concept_name
                            merged_df = pd.concat([merged_df, df_copy], ignore_index=True)
                        continue
                    
                    # 只保留合并键和当前 concept 的值列
                    # 🔧 删除非核心列（如 valueuom 等元数据列）
                    metadata_cols = ['valueuom', 'unit', 'units', 'category', 'type']
                    cols_to_drop = [c for c in df.columns if c in metadata_cols]
                    if cols_to_drop:
                        df = df.drop(columns=cols_to_drop)
                    
                    value_cols = [c for c in df.columns if c not in merge_cols]
                    
                    # 如果只有一个值列，用 concept 名重命名
                    if len(value_cols) == 1:
                        df = df.rename(columns={value_cols[0]: concept_name})
                    elif len(value_cols) > 1:
                        # 多个值列，添加前缀
                        rename_map = {c: f"{concept_name}_{c}" for c in value_cols if c != concept_name}
                        df = df.rename(columns=rename_map)
                    
                    if merged_df is None:
                        merged_df = df
                    else:
                        # 外连接合并
                        merged_df = pd.merge(merged_df, df, on=merge_cols, how='outer')
            
            if merged_df is None or len(merged_df) == 0:
                continue
            
            # 生成文件名：[筛选条件前缀_]模块名_特征1_特征2_...
            concept_names = list(concept_dfs.keys())
            # 限制特征名长度，避免文件名过长
            if len(concept_names) <= 5:
                concepts_suffix = '_'.join(concept_names)
            else:
                concepts_suffix = '_'.join(concept_names[:4]) + f'_etc{len(concept_names)}'
            
            # 🚀 添加队列筛选条件前缀
            cohort_prefix = _generate_cohort_prefix()
            
            # 清理文件名中的特殊字符
            if cohort_prefix:
                safe_filename = f"{cohort_prefix}_{group_name}_{concepts_suffix}".replace('/', '_').replace('\\', '_')
            else:
                safe_filename = f"{group_name}_{concepts_suffix}".replace('/', '_').replace('\\', '_')
            # 限制文件名总长度
            if len(safe_filename) > 150:
                safe_filename = safe_filename[:150]
            
            if export_format == 'csv':
                file_path = export_dir / f"{safe_filename}.csv"
                merged_df.to_csv(file_path, index=False)
            elif export_format == 'parquet':
                file_path = export_dir / f"{safe_filename}.parquet"
                merged_df.to_parquet(file_path, index=False)
            elif export_format == 'excel':
                file_path = export_dir / f"{safe_filename}.xlsx"
                merged_df.to_excel(file_path, index=False)
            else:
                file_path = export_dir / f"{safe_filename}.parquet"
                merged_df.to_parquet(file_path, index=False)
            
            exported_files.append(str(file_path))
            
            # 更新导出进度（从50%到100%）
            if use_mock:
                progress_bar.progress(0.3 + 0.7 * (idx + 1) / total_groups)
            else:
                progress_bar.progress(0.5 + 0.5 * (idx + 1) / total_groups)
        
        # 完成
        progress_bar.progress(1.0)
        status_text.empty()
        
        if exported_files:
            st.session_state.export_completed = True
            st.session_state.last_export_dir = str(export_dir)  # 保存实际导出目录
            success_msg = f"✅ Successfully exported {len(exported_files)} files to `{export_dir}`" if lang == 'en' else f"✅ 成功导出 {len(exported_files)} 个文件到 `{export_dir}`"
            st.success(success_msg)
            
            # 显示导出的文件列表
            view_files_label = "📁 View Exported Files" if lang == 'en' else "📁 查看导出文件"
            with st.expander(view_files_label, expanded=True):
                for f in exported_files[:10]:
                    st.caption(f"• {Path(f).name}")
                if len(exported_files) > 10:
                    more_msg = f"... and {len(exported_files) - 10} more files" if lang == 'en' else f"... 及其他 {len(exported_files) - 10} 个文件"
                    st.caption(more_msg)
        else:
            no_data_msg = "⚠️ No data was exported" if lang == 'en' else "⚠️ 没有数据被导出"
            st.warning(no_data_msg)
                
    except Exception as e:
        fail_msg = f"❌ Export failed: {e}" if lang == 'en' else f"❌ 导出失败: {e}"
        st.error(fail_msg)


def render_export_page():
    """渲染数据导出页面。"""
    lang = st.session_state.get('language', 'en')
    export_title = "💾 Data Export" if lang == 'en' else "💾 数据导出"
    st.markdown(f"## {export_title}")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    if len(st.session_state.loaded_concepts) == 0:
        if lang == 'en':
            st.markdown('''
            <div class="info-box">
                <strong>👈 Please load data from the sidebar first</strong><br>
                💡 Tip: Select "Demo Mode" to quickly explore all features
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="info-box">
                <strong>👈 请先在侧边栏加载数据</strong><br>
                💡 提示：勾选「使用模拟数据」可快速体验所有功能
            </div>
            ''', unsafe_allow_html=True)
        return
    
    # 快速导出面板
    quick_title = "⚡ Quick Export" if lang == 'en' else "⚡ 快速导出"
    st.markdown(f"### {quick_title}")
    quick_cols = st.columns(4)
    
    import io
    from datetime import datetime
    
    with quick_cols[0]:
        # 一键导出所有CSV
        df_list = [df.assign(concept=name) for name, df in st.session_state.loaded_concepts.items() 
                   if isinstance(df, pd.DataFrame) and len(df) > 0]
        if df_list:
            all_data = pd.concat(df_list, ignore_index=True)
            csv_all = all_data.to_csv(index=False)
            all_csv_label = "📄 All CSV" if lang == 'en' else "📄 全部CSV"
            all_csv_help = "Export all data as CSV" if lang == 'en' else "一键导出所有数据为CSV"
            st.download_button(
                label=all_csv_label,
                data=csv_all,
                file_name=f"pyricu_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width="stretch",
                help=all_csv_help
            )
        else:
            no_data_label = "📄 No Data" if lang == 'en' else "📄 无数据"
            st.button(no_data_label, disabled=True, width="stretch")
    
    with quick_cols[1]:
        # 当前选中患者
        if st.session_state.get('selected_patient'):
            patient_id = st.session_state.selected_patient
            patient_data = {}
            for name, df in st.session_state.loaded_concepts.items():
                if isinstance(df, pd.DataFrame) and st.session_state.id_col in df.columns:
                    patient_df = df[df[st.session_state.id_col] == patient_id]
                    if len(patient_df) > 0:
                        patient_data[name] = patient_df
            
            if patient_data:
                patient_combined = pd.concat(
                    [df.assign(concept=name) for name, df in patient_data.items()],
                    ignore_index=True
                )
                patient_csv = patient_combined.to_csv(index=False)
                st.download_button(
                    label=f"👤 患者{patient_id}",
                    data=patient_csv,
                    file_name=f"patient_{patient_id}_{datetime.now().strftime('%H%M%S')}.csv",
                    mime="text/csv",
                    width="stretch",
                    help=f"Export all data for patient {patient_id}" if lang == 'en' else f"导出患者 {patient_id} 的所有数据"
                )
            else:
                no_pat = "👤 No Patient" if lang == 'en' else "👤 无患者"
                st.button(no_pat, disabled=True, width="stretch")
        else:
            no_sel = "👤 No Selection" if lang == 'en' else "👤 未选患者"
            no_sel_help = "Please select a patient in Patient View first" if lang == 'en' else "请先在患者视图中选择一位患者"
            st.button(no_sel, disabled=True, width="stretch", help=no_sel_help)
    
    with quick_cols[2]:
        # 生命体征快速导出
        vitals = ['hr', 'map', 'sbp', 'resp', 'spo2', 'temp']
        vitals_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                      if k in vitals and isinstance(v, pd.DataFrame) and len(v) > 0}
        if vitals_data:
            vitals_combined = pd.concat(
                [df.assign(concept=name) for name, df in vitals_data.items()],
                ignore_index=True
            )
            vitals_csv = vitals_combined.to_csv(index=False)
            vitals_label = "💓 Vitals" if lang == 'en' else "💓 生命体征"
            vitals_help = "Export all vital signs data" if lang == 'en' else "导出所有生命体征数据"
            st.download_button(
                label=vitals_label,
                data=vitals_csv,
                file_name=f"vitals_{datetime.now().strftime('%H%M%S')}.csv",
                mime="text/csv",
                width="stretch",
                help=vitals_help
            )
        else:
            no_vitals = "💓 No Vitals" if lang == 'en' else "💓 无体征数据"
            st.button(no_vitals, disabled=True, width="stretch")
    
    with quick_cols[3]:
        # 实验室数据快速导出
        labs = ['bili', 'crea', 'plt', 'lac', 'wbc', 'hgb']
        labs_data = {k: v for k, v in st.session_state.loaded_concepts.items() 
                    if k in labs and isinstance(v, pd.DataFrame) and len(v) > 0}
        if labs_data:
            labs_combined = pd.concat(
                [df.assign(concept=name) for name, df in labs_data.items()],
                ignore_index=True
            )
            labs_csv = labs_combined.to_csv(index=False)
            labs_label = "🧪 Labs" if lang == 'en' else "🧪 实验室"
            labs_help = "Export all laboratory data" if lang == 'en' else "导出所有实验室数据"
            st.download_button(
                label=labs_label,
                data=labs_csv,
                file_name=f"labs_{datetime.now().strftime('%H%M%S')}.csv",
                mime="text/csv",
                width="stretch",
                help=labs_help
            )
        else:
            no_labs = "🧪 No Labs Data" if lang == 'en' else "🧪 无实验室数据"
            st.button(no_labs, disabled=True, width="stretch")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # 导出配置面板
    custom_title = "### 🎛️ Custom Export" if lang == 'en' else "### 🎛️ 自定义导出"
    st.markdown(custom_title)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        concepts_label = "📋 Select Concepts" if lang == 'en' else "📋 选择 Concepts"
        concepts_help = "Select data types to export" if lang == 'en' else "选择要导出的数据类型"
        concepts_to_export = st.multiselect(
            concepts_label,
            options=list(st.session_state.loaded_concepts.keys()),
            default=list(st.session_state.loaded_concepts.keys()),
            help=concepts_help
        )
    
    with col2:
        format_label = "📁 Export Format" if lang == 'en' else "📁 导出格式"
        format_help = "CSV: Universal format\nExcel: Multi-sheet support\nParquet: Efficient storage" if lang == 'en' else "CSV: 通用格式\nExcel: 支持多Sheet\nParquet: 高效存储"
        export_format = st.selectbox(
            format_label,
            options=['CSV', 'Excel', 'Parquet'],
            help=format_help
        )
        
        format_icons = {'CSV': '📄', 'Excel': '📊', 'Parquet': '⚡'}
        selected_text = "Selected" if lang == 'en' else "已选择"
        st.markdown(f"<small>{format_icons.get(export_format, '')} {selected_text} {export_format}</small>", unsafe_allow_html=True)
    
    with col3:
        merge_label = "📦 Merge Mode" if lang == 'en' else "📦 合并模式"
        merge_options = ['Separate Files', 'Merge Into One'] if lang == 'en' else ['分开保存', '合并为一个文件']
        merge_help = "Separate: One file per Concept\nMerge: All data in one file" if lang == 'en' else "分开: 每个Concept一个文件\n合并: 所有数据合并"
        merge_mode = st.selectbox(
            merge_label,
            options=merge_options,
            help=merge_help
        )
    
    # 高级选项
    adv_label = "⚙️ Advanced Options" if lang == 'en' else "⚙️ 高级选项"
    with st.expander(adv_label, expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            filter_label = "Filter by Patient" if lang == 'en' else "按患者过滤"
            filter_patient = st.checkbox(filter_label, value=False)
            if filter_patient and st.session_state.patient_ids:
                select_patients_label = "Select Patients" if lang == 'en' else "选择患者"
                selected_patients = st.multiselect(
                    select_patients_label,
                    options=st.session_state.patient_ids[:100],
                    default=st.session_state.patient_ids[:5]
                )
            else:
                selected_patients = None
        
        with col2:
            index_label = "Include Row Index" if lang == 'en' else "包含行索引"
            include_index = st.checkbox(index_label, value=False)
            timestamp_label = "Add Timestamp to Filename" if lang == 'en' else "文件名添加时间戳"
            add_timestamp = st.checkbox(timestamp_label, value=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # 导出预览
    preview_title = "### 📋 Export Preview" if lang == 'en' else "### 📋 导出预览"
    st.markdown(preview_title)
    
    preview_data = {}
    total_rows = 0
    total_cols = 0
    
    for name in concepts_to_export:
        df = st.session_state.loaded_concepts[name]
        
        # 确保是 DataFrame
        if not isinstance(df, pd.DataFrame):
            continue
        
        if selected_patients and st.session_state.id_col in df.columns:
            df = df[df[st.session_state.id_col].isin(selected_patients)]
        
        preview_data[name] = df
        total_rows += len(df)
        total_cols = max(total_cols, len(df.columns))
    
    # 预览统计卡片
    col1, col2, col3, col4 = st.columns(4)
    
    total_records_label = "Total Records" if lang == 'en' else "总记录数"
    est_size_label = "Est. Size" if lang == 'en' else "预估大小"
    format_label_2 = "Format" if lang == 'en' else "格式"
    
    with col1:
        st.markdown(f'''
        <div class="metric-card" style="text-align:center">
            <div class="stat-label">Concepts</div>
            <div class="stat-number">{len(concepts_to_export)}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card" style="text-align:center">
            <div class="stat-label">{total_records_label}</div>
            <div class="stat-number">{total_rows:,}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        # 估算文件大小
        est_size = total_rows * total_cols * 10 / 1024  # 粗略估算 KB
        size_str = f"{est_size:.0f} KB" if est_size < 1024 else f"{est_size/1024:.1f} MB"
        st.markdown(f'''
        <div class="metric-card" style="text-align:center">
            <div class="stat-label">{est_size_label}</div>
            <div class="stat-number" style="font-size:1.5rem">{size_str}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="metric-card" style="text-align:center">
            <div class="stat-label">{format_label_2}</div>
            <div class="stat-number" style="font-size:1.5rem">{export_format}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # 数据预览表格
    if concepts_to_export:
        preview_exp_label = "👁️ Preview Data" if lang == 'en' else "👁️ 预览数据"
        with st.expander(preview_exp_label, expanded=False):
            select_preview_label = "Select Preview" if lang == 'en' else "选择预览"
            preview_concept = st.selectbox(select_preview_label, concepts_to_export)
            if preview_concept in preview_data:
                st.dataframe(preview_data[preview_concept].head(20), use_container_width=True, hide_index=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # 导出按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        export_btn_label = "📥 Export Data" if lang == 'en' else "📥 导出数据"
        spinner_text = "Preparing export..." if lang == 'en' else "正在准备导出..."
        merge_single = "Merge Into One" if lang == 'en' else "合并为一个文件"
        
        if st.button(export_btn_label, type="primary", width="stretch"):
            with st.spinner(spinner_text):
                import io
                from datetime import datetime
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""
                
                try:
                    filename_base = f"pyricu_export_{timestamp}" if timestamp else "pyricu_export"
                    
                    if export_format == 'CSV':
                        if merge_mode == merge_single:
                            combined = pd.concat(
                                [df.assign(concept=name) for name, df in preview_data.items()],
                                ignore_index=True
                            )
                            csv = combined.to_csv(index=include_index)
                            dl_csv = "⬇️ Download CSV" if lang == 'en' else "⬇️ 下载 CSV"
                            st.download_button(
                                label=dl_csv,
                                data=csv,
                                file_name=f"{filename_base}.csv",
                                mime="text/csv",
                            )
                        else:
                            # 分开保存 - 创建 ZIP
                            import zipfile
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                                for name, df in preview_data.items():
                                    csv_data = df.to_csv(index=include_index)
                                    zf.writestr(f"{name}.csv", csv_data)
                            
                            dl_zip = "⬇️ Download ZIP (Multiple CSVs)" if lang == 'en' else "⬇️ 下载 ZIP (多个CSV)"
                            st.download_button(
                                label=dl_zip,
                                data=zip_buffer.getvalue(),
                                file_name=f"{filename_base}.zip",
                                mime="application/zip",
                            )
                    
                    elif export_format == 'Excel':
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            if merge_mode == merge_single:
                                combined = pd.concat(
                                    [df.assign(concept=name) for name, df in preview_data.items()],
                                    ignore_index=True
                                )
                                combined.to_excel(writer, sheet_name='all_data', index=include_index)
                            else:
                                for name, df in preview_data.items():
                                    sheet_name = name[:31]  # Excel sheet name limit
                                    df.to_excel(writer, sheet_name=sheet_name, index=include_index)
                        
                        dl_excel = "⬇️ Download Excel" if lang == 'en' else "⬇️ 下载 Excel"
                        st.download_button(
                            label=dl_excel,
                            data=output.getvalue(),
                            file_name=f"{filename_base}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    
                    elif export_format == 'Parquet':
                        combined = pd.concat(
                            [df.assign(concept=name) for name, df in preview_data.items()],
                            ignore_index=True
                        )
                        output = io.BytesIO()
                        combined.to_parquet(output, index=include_index)
                        dl_parquet = "⬇️ Download Parquet" if lang == 'en' else "⬇️ 下载 Parquet"
                        st.download_button(
                            label=dl_parquet,
                            data=output.getvalue(),
                            file_name=f"{filename_base}.parquet",
                            mime="application/octet-stream",
                        )
                    
                    success_msg = "✅ Export ready! Click the button above to download" if lang == 'en' else "✅ 导出准备完成！点击上方按钮下载"
                    st.markdown(f'''
                    <div class="success-box">
                        {success_msg}
                    </div>
                    ''', unsafe_allow_html=True)
                    
                except Exception as e:
                    err_msg = f"❌ Export failed: {e}" if lang == 'en' else f"❌ 导出失败: {e}"
                    st.error(err_msg)


def main():
    """主函数。"""
    init_session_state()
    render_sidebar()
    
    # 处理侧边栏触发的导出
    if st.session_state.get('trigger_export', False):
        st.session_state.trigger_export = False
        execute_sidebar_export()
    
    # 处理CSV转换对话框
    if st.session_state.get('show_convert_dialog', False):
        render_convert_dialog()
    
    # ============ 顶部标题（放在导航栏上方） ============
    lang = st.session_state.get('language', 'en')
    if lang == 'en':
        st.markdown('<div class="main-header">🏥 PyRICU Data Explorer</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Local ICU Data Analytics Platform</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="main-header">🏥 PyRICU 数据探索器</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">本地 ICU 数据分析与可视化平台</div>', unsafe_allow_html=True)
    
    # 主页面标签（数据导出已移至左侧边栏）
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        get_text('home'),
        get_text('timeseries'), 
        get_text('patient_view'),
        get_text('data_quality'),
        get_text('cohort_compare'),
    ])
    
    with tab1:
        render_home()
    
    with tab2:
        render_timeseries_page()
    
    with tab3:
        render_patient_page()
    
    with tab4:
        render_quality_page()
    
    with tab5:
        render_cohort_comparison_page()
    
    # 底部状态栏
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    footer_cols = st.columns([2, 2, 1])
    
    with footer_cols[0]:
        if st.session_state.language == 'en':
            data_status = "✅ Data Loaded" if len(st.session_state.loaded_concepts) > 0 else "⏳ No Data"
            patients_label = "Patients"
        else:
            data_status = "✅ 数据已加载" if len(st.session_state.loaded_concepts) > 0 else "⏳ 未加载数据"
            patients_label = "患者"
        n_concepts = len(st.session_state.loaded_concepts)
        n_patients = len(st.session_state.patient_ids) if st.session_state.patient_ids else 0
        st.markdown(
            f"<small style='color:#888'>{data_status} | 📋 {n_concepts} Concepts | 👥 {n_patients} {patients_label}</small>",
            unsafe_allow_html=True
        )
    
    with footer_cols[1]:
        if st.session_state.get('selected_patient'):
            patient_label = "Current Patient" if st.session_state.language == 'en' else "当前患者"
            st.markdown(
                f"<small style='color:#888'>🎯 {patient_label}: {st.session_state.selected_patient}</small>",
                unsafe_allow_html=True
            )
    
    with footer_cols[2]:
        # 帮助按钮
        help_btn_text = "❓ Help" if st.session_state.language == 'en' else "❓ 帮助"
        with st.popover(help_btn_text):
            if st.session_state.language == 'en':
                st.markdown("""
                ### 🚀 Quick Start
                
                **1. Load Data**
                - Check "Demo Mode" in sidebar for quick exploration
                - Or upload real Parquet/CSV files
                
                **2. Browse & Analyze**
                - 📈 **Time Series**: View metric trends, multi-patient comparison
                - 🏥 **Patient View**: Comprehensive single patient data
                - 📊 **Data Quality**: Assess data completeness
                
                **3. Export Data**
                - ⚡ Quick Export: One-click export common data
                - 🎛️ Custom: Select format and filter conditions
                
                ---
                
                💡 **Tips**: 
                - Homepage has "Quick Experience" button
                - Patient view supports quick navigation
                - Multi-patient comparison can normalize data
                """)
            else:
                st.markdown("""
                ### 🚀 快速上手
                
                **1. 加载数据**
                - 侧边栏勾选「使用模拟数据」快速体验
                - 或上传真实 Parquet/CSV 文件
                
                **2. 浏览分析**
                - 📈 **时序分析**: 查看指标趋势，支持多患者比较
                - 🏥 **患者视图**: 综合查看单个患者数据
                - 📊 **数据质量**: 评估数据完整性
                
                **3. 导出数据**
                - ⚡ 快速导出: 一键导出常用数据
                - 🎛️ 自定义: 选择格式和筛选条件
                
                ---
                
                💡 **提示**: 
                - 首页有「一键体验」按钮
                - 患者视图支持快速导航
                - 多患者比较可归一化数据
                """)


if __name__ == "__main__":
    main()
