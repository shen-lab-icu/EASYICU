"""PyRICU Webapp Session State 管理模块。

集中管理 Streamlit session state 的初始化和访问。
"""

import os
import streamlit as st
from typing import Dict, Any


def init_session_state():
    """初始化 session state。
    
    在应用启动时调用，确保所有必需的状态变量都已初始化。
    """
    # 🆕 入口模式：'none' (入口页), 'demo' (演示模式), 'real' (真实数据模式)
    if 'entry_mode' not in st.session_state:
        st.session_state.entry_mode = 'none'
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
        st.session_state.mock_params = {'n_patients': 100, 'hours': 72}
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
    # 🚀 性能优化：患者数量限制
    # 全量 MIIV 约 5万患者/4000万行，加载需 ~50s；100患者约2s
    # 🔧 FIX 2025-01-28: 默认全量加载（0=不限制），满足大多数用户需求
    if 'patient_limit' not in st.session_state:
        st.session_state.patient_limit = 0  # 默认全量加载
    if 'available_patient_ids' not in st.session_state:
        st.session_state.available_patient_ids = None
    # 🆕 步骤确认状态
    if 'step1_confirmed' not in st.session_state:
        st.session_state.step1_confirmed = False
    if 'step2_confirmed' not in st.session_state:
        st.session_state.step2_confirmed = False


def get_mock_params_with_cohort() -> Dict[str, Any]:
    """获取完整的 mock_params，包含最新的 cohort_filter。
    
    由于 Streamlit 的渲染顺序，Step 1 (数据源) 在 Step 2 (队列筛选) 之前执行，
    所以 mock_params 中的 cohort_filter 可能不是最新的。
    
    此函数确保在调用 generate_mock_data 时使用最新的 cohort_filter。
    
    Returns:
        包含最新 cohort_filter 的 mock_params 字典
    """
    params = st.session_state.get('mock_params', {'n_patients': 100, 'hours': 72}).copy()
    
    # 如果启用了队列筛选，添加最新的 cohort_filter
    if st.session_state.get('cohort_enabled', False):
        cohort_filter = st.session_state.get('cohort_filter', None)
        if cohort_filter:
            params['cohort_filter'] = cohort_filter
    
    return params


def get_state(key: str, default: Any = None) -> Any:
    """安全获取 session state 值。
    
    Args:
        key: 状态键名
        default: 默认值
        
    Returns:
        状态值或默认值
    """
    return st.session_state.get(key, default)


def set_state(key: str, value: Any) -> None:
    """设置 session state 值。
    
    Args:
        key: 状态键名
        value: 状态值
    """
    st.session_state[key] = value


def clear_loaded_data() -> None:
    """清除已加载的数据状态。"""
    st.session_state.loaded_concepts = {}
    st.session_state.patient_ids = []
    st.session_state.all_patient_count = 0
    st.session_state.selected_patient = None


def get_id_column() -> str:
    """获取当前数据库的 ID 列名。
    
    Returns:
        ID 列名（如 stay_id, patientunitstayid 等）
    """
    db = st.session_state.get('database', 'miiv')
    id_col_map = {
        'miiv': 'stay_id',
        'eicu': 'patientunitstayid',
        'aumc': 'admissionid',
        'hirid': 'patientid',
        'mimic': 'icustay_id',
        'sic': 'CaseID',
    }
    return id_col_map.get(db, 'stay_id')


def is_demo_mode() -> bool:
    """检查是否处于演示模式。
    
    Returns:
        True 如果是演示模式
    """
    return st.session_state.get('entry_mode') == 'demo'


def is_real_data_mode() -> bool:
    """检查是否处于真实数据模式。
    
    Returns:
        True 如果是真实数据模式
    """
    return st.session_state.get('entry_mode') == 'real'


def get_current_language() -> str:
    """获取当前语言设置。
    
    Returns:
        语言代码（'en' 或 'zh'）
    """
    return st.session_state.get('language', 'en')
