"""PyRICU Webapp 配置模块。

包含国际化文本、概念分组、常量等配置。
"""

import streamlit as st
from typing import Dict, List, Any


# ============ 国际化文本 ============
TEXTS: Dict[str, Dict[str, str]] = {
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
        'quick_visualization': '📊 Quick Visualization',
        'cohort_compare': '📊 Cohort Analysis',
        'sub_data_table': '📋 Data Tables',
        'sub_timeseries': '📈 Time Series',
        'sub_patient_view': '🏥 Patient View',
        'sub_data_quality': '📊 Data Quality',
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
        'quick_visualization': '📊 快速可视化',
        'cohort_compare': '📊 队列分析',
        'sub_data_table': '📋 数据大表',
        'sub_timeseries': '📈 时序分析',
        'sub_patient_view': '🏥 患者视图',
        'sub_data_quality': '📊 数据质量',
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
    """根据当前语言获取文本。
    
    Args:
        key: 文本键名
        
    Returns:
        对应语言的文本，未找到则返回键名
    """
    lang = st.session_state.get('language', 'en')
    return TEXTS.get(lang, TEXTS['en']).get(key, key)


# ============ 概念分组配置 ============
def get_concept_groups() -> Dict[str, Dict[str, Any]]:
    """获取概念分组配置。
    
    Returns:
        概念分组字典，包含中英文名称和概念列表
    """
    return {
        'vitals': {
            'name_en': '💓 Vital Signs',
            'name_zh': '💓 生命体征',
            'concepts': ['hr', 'sbp', 'dbp', 'map', 'resp', 'temp', 'spo2']
        },
        'labs_basic': {
            'name_en': '🧪 Basic Labs',
            'name_zh': '🧪 基础实验室',
            'concepts': ['bili', 'crea', 'glu', 'k', 'na', 'phos', 'alb']
        },
        'blood_gas': {
            'name_en': '🫁 Blood Gas',
            'name_zh': '🫁 血气分析',
            'concepts': ['po2', 'pco2', 'ph', 'o2sat', 'fio2', 'sao2']
        },
        'hematology': {
            'name_en': '🩸 Hematology',
            'name_zh': '🩸 血液学',
            'concepts': ['hgb', 'plt', 'wbc', 'inr_pt', 'ptt']
        },
        'neurological': {
            'name_en': '🧠 Neurological',
            'name_zh': '🧠 神经系统',
            'concepts': ['gcs', 'tgcs', 'avpu']
        },
        'demographics': {
            'name_en': '👤 Demographics',
            'name_zh': '👤 人口统计',
            'concepts': ['weight', 'height', 'bmi', 'age', 'sex']
        },
        'urine': {
            'name_en': '💧 Urine Output',
            'name_zh': '💧 尿量',
            'concepts': ['urine', 'urine24']
        },
        'vasopressors': {
            'name_en': '💉 Vasopressors',
            'name_zh': '💉 血管活性药物',
            'concepts': ['norepi_rate', 'epi_rate', 'dopa_rate', 'dobu_rate', 'vaso_rate']
        },
        'outcome': {
            'name_en': '📊 Outcomes',
            'name_zh': '📊 结局指标',
            'concepts': ['death', 'los_icu', 'los_hosp', 'abx']
        },
        'sofa': {
            'name_en': '📈 SOFA Scores',
            'name_zh': '📈 SOFA评分',
            'concepts': ['sofa', 'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal']
        },
        'sofa2': {
            'name_en': '📈 SOFA2 Scores',
            'name_zh': '📈 SOFA2评分',
            'concepts': ['sofa2', 'sofa2_resp', 'sofa2_coag', 'sofa2_liver', 'sofa2_cardio', 'sofa2_cns', 'sofa2_renal']
        },
        'sepsis': {
            'name_en': '🦠 Sepsis',
            'name_zh': '🦠 脓毒症',
            'concepts': ['sep3', 'susp_inf', 'sep3_sofa2']
        },
        'ventilator': {
            'name_en': '🫁 Ventilator',
            'name_zh': '🫁 呼吸机',
            'concepts': ['peep', 'tidal_vol', 'pip', 'plateau_pres', 'minute_vol', 'vent_rate', 'driving_pres']
        },
        'ratios': {
            'name_en': '📐 Ratios',
            'name_zh': '📐 比值',
            'concepts': ['pafi', 'safi']
        },
    }


# ============ 数据库配置 ============
DATABASE_NAMES: Dict[str, str] = {
    'miiv': 'MIMIC-IV',
    'eicu': 'eICU-CRD',
    'aumc': 'AmsterdamUMCdb',
    'hirid': 'HiRID',
    'mimic': 'MIMIC-III',
    'sic': 'SICdb',
}


# ============ 图表配置 ============
CHART_COLORS: Dict[str, str] = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'info': '#06b6d4',
}


# ============ 必需表配置 ============
REQUIRED_TABLES: Dict[str, Dict[str, List[str]]] = {
    'miiv': {
        'required': ['patients', 'admissions', 'icustays'],
        'optional': ['chartevents', 'labevents', 'inputevents', 'outputevents'],
    },
    'eicu': {
        'required': ['patient'],
        'optional': ['vitalperiodic', 'vitalaperiodic', 'lab', 'nursecharting'],
    },
    'aumc': {
        'required': ['admissions'],
        'optional': ['numericitems', 'listitems', 'drugitems'],
    },
    'hirid': {
        'required': ['general_table'],
        'optional': ['observations', 'pharma'],
    },
    'mimic': {
        'required': ['patients', 'admissions', 'icustays'],
        'optional': ['chartevents', 'labevents'],
    },
    'sic': {
        'required': ['cases'],
        'optional': ['data_float_h', 'laboratory', 'medication'],
    },
}
