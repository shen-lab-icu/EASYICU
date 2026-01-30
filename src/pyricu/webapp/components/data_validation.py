"""
数据验证和检查功能模块。

包含:
- validate_database_path: 验证数据库路径
- check_data_status: 检查数据目录状态
- find_database_path: 查找数据库路径

所有验证函数都不依赖 Streamlit UI，可独立使用。
"""

from pathlib import Path
from typing import Dict, List, Optional, Any


# 各数据库需要的核心表（Parquet格式）
REQUIRED_PARQUET_TABLES = {
    'miiv': {
        'core': ['icustays', 'patients', 'admissions'],
        'clinical': ['chartevents', 'labevents', 'outputevents'],
        'medication': ['prescriptions', 'inputevents'],
    },
    'eicu': {
        'core': ['patient', 'apachepatientresult'],
        'clinical': ['vitalperiodic', 'vitalaperiodic', 'lab', 'nursecharting'],
        'medication': ['medication', 'infusiondrug'],
    },
    'aumc': {
        'core': ['admissions', 'drugitems'],
        'clinical': ['numericitems', 'listitems', 'freetextitems'],
        'medication': ['procedureorderitems'],
    },
    'hirid': {
        'core': ['general_table'],
        'clinical': ['observations'],
        'medication': ['pharma_records'],
    },
    'mimic': {  # MIMIC-III
        'core': ['icustays', 'patients', 'admissions'],
        'clinical': ['chartevents', 'labevents', 'outputevents'],
        'medication': ['prescriptions', 'inputevents_cv', 'inputevents_mv'],
    },
    'sic': {  # SICdb
        'core': ['cases'],
        'clinical': ['data_float_h', 'laboratory'],
        'medication': ['medication'],
    },
}

# 各数据库需要的核心表（CSV/GZ格式 - 原始文件）
REQUIRED_CSV_FILES = {
    'miiv': ['icustays.csv', 'chartevents.csv', 'labevents.csv', 'prescriptions.csv', 'inputevents.csv'],
    'eicu': ['patient.csv', 'vitalPeriodic.csv', 'lab.csv'],
    'aumc': ['admissions.csv', 'numericitems.csv', 'drugitems.csv'],
    'hirid': ['general_table.csv', 'pharma_records.csv'],
    'mimic': ['icustays.csv', 'chartevents.csv', 'labevents.csv', 'prescriptions.csv'],
    'sic': ['cases.csv', 'data_float_h.csv', 'laboratory.csv', 'medication.csv'],
}

# 数据库名称映射
DATABASE_NAMES = {
    'miiv': 'MIMIC-IV',
    'eicu': 'eICU-CRD',
    'aumc': 'AmsterdamUMCdb',
    'hirid': 'HiRID',
    'mimic': 'MIMIC-III',
    'sic': 'SICdb',
}

# 数据库版本目录模式
DATABASE_PATTERNS = {
    'miiv': ['mimiciv', 'mimic-iv', 'mimic_iv', 'miiv'],
    'eicu': ['eicu', 'eicu-crd', 'eicu_crd'],
    'aumc': ['aumc', 'amsterdamumc', 'amsterdamumcdb'],
    'hirid': ['hirid'],
    'mimic': ['mimiciii', 'mimic-iii', 'mimic_iii', 'mimic3'],
    'sic': ['sicdb', 'sic', 'sic_db'],
}


def check_data_status(data_path: str, database: str) -> Dict[str, Any]:
    """检查数据目录的状态，返回文件统计信息。
    
    Args:
        data_path: 数据目录路径
        database: 数据库类型
        
    Returns:
        dict: 包含以下键:
            - ready: bool - 数据是否就绪
            - parquet_count: int - Parquet 文件数量
            - csv_count: int - CSV 文件数量
            - csv_files: list - CSV 文件名列表
            - parquet_files: list - Parquet 文件/目录名列表
            - missing_tables: list - 缺失的核心表
    """
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
        'mimic': ['icustays', 'patients', 'admissions'],
        'sic': ['cases'],
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


def validate_database_path(data_path: str, database: str, lang: str = 'en') -> Dict[str, Any]:
    """验证数据库目录是否包含所需的表。
    
    Args:
        data_path: 数据目录路径
        database: 数据库类型
        lang: 语言 ('en' 或 'zh')
        
    Returns:
        dict: 包含以下键:
            - valid: bool - 路径是否有效
            - message: str - 验证结果消息
            - parquet_count: int - 可选，Parquet 文件数量
            - bucket_count: int - 可选，分桶目录数量
    """
    path = Path(data_path)
    
    if not path.exists():
        msg = f"❌ Path does not exist: {data_path}" if lang == 'en' else f"❌ 路径不存在: {data_path}"
        return {'valid': False, 'message': msg}
    
    if not path.is_dir():
        msg = f"❌ Not a directory: {data_path}" if lang == 'en' else f"❌ 不是目录: {data_path}"
        return {'valid': False, 'message': msg}
    
    db_name = DATABASE_NAMES.get(database, database.upper())
    
    # 检查Parquet文件和分片目录
    parquet_files = list(path.rglob('*.parquet'))
    parquet_names = set(f.name.lower().replace('.parquet', '') for f in parquet_files)
    
    # 对于某些数据库（如 HiRID），某些核心表可能是 CSV 格式
    csv_files = list(path.glob('*.csv'))
    csv_names = set(f.name.lower().replace('.csv', '') for f in csv_files)
    
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
    
    # 检查分桶目录（如 chartevents_bucket/bucket_id=*/data.parquet）
    bucket_dirs = set()
    for subdir in path.iterdir():
        if subdir.is_dir() and subdir.name.endswith('_bucket'):
            # 检查是否有 parquet 文件
            bucket_parquets = list(subdir.rglob('*.parquet'))
            if bucket_parquets:
                # 去掉 _bucket 后缀得到表名
                table_name = subdir.name[:-7]  # remove '_bucket'
                bucket_dirs.add(table_name.lower())
    
    # 合并所有找到的表（单文件、分片目录、分桶目录、CSV文件）
    all_found = parquet_names | parquet_dirs | bucket_dirs | csv_names
    
    # HiRID 特殊处理：pharma_bucket → pharma_records
    if database == 'hirid' and 'pharma' in all_found:
        all_found.add('pharma_records')
    
    # 检查各类别的表
    db_tables = REQUIRED_PARQUET_TABLES.get(database, {})
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
        bucket_info = f", {len(bucket_dirs)} bucketed" if bucket_dirs else ""
        msg = f'✅ {db_name}: All {total_required} required tables found ({len(parquet_files)} Parquet files{bucket_info})' if lang == 'en' else f'✅ {db_name}: 所有 {total_required} 个必需表已找到 ({len(parquet_files)} 个 Parquet 文件{bucket_info})'
        return {
            'valid': True,
            'message': msg,
            'parquet_count': len(parquet_files),
            'bucket_count': len(bucket_dirs),
        }
    
    # 核心表缺失是严重问题
    core_missing = missing_by_category.get('core', [])
    if core_missing:
        missing_str = ', '.join(core_missing)
        if lang == 'en':
            msg = f'❌ {db_name}: Missing core tables: {missing_str}. Please convert CSV to Parquet first.'
        else:
            msg = f'❌ {db_name}: 缺少核心表: {missing_str}。请先将 CSV 转换为 Parquet。'
        return {'valid': False, 'message': msg}
    
    # 临床表缺失是警告
    clinical_missing = missing_by_category.get('clinical', [])
    medication_missing = missing_by_category.get('medication', [])
    
    if clinical_missing or medication_missing:
        all_missing = clinical_missing + medication_missing
        missing_str = ', '.join(all_missing[:3])
        more_text = f' +{len(all_missing) - 3} more' if len(all_missing) > 3 else ''
        bucket_info = f", {len(bucket_dirs)} bucketed" if bucket_dirs else ""
        
        if lang == 'en':
            msg = f'⚠️ {db_name}: Found {len(found_tables)}/{total_required} tables ({len(parquet_files)} Parquet{bucket_info}). Missing optional: {missing_str}{more_text}'
        else:
            msg = f'⚠️ {db_name}: 找到 {len(found_tables)}/{total_required} 个表 ({len(parquet_files)} 个 Parquet{bucket_info})。缺少可选表: {missing_str}{more_text}'
        
        return {
            'valid': True,
            'message': msg,
            'parquet_count': len(parquet_files),
            'bucket_count': len(bucket_dirs),
        }
    
    # 默认情况
    bucket_info = f", {len(bucket_dirs)} bucketed" if bucket_dirs else ""
    msg = f'✅ {db_name}: {len(found_tables)} tables found ({len(parquet_files)} Parquet{bucket_info})' if lang == 'en' else f'✅ {db_name}: 找到 {len(found_tables)} 个表 ({len(parquet_files)} 个 Parquet{bucket_info})'
    return {
        'valid': True,
        'message': msg,
        'parquet_count': len(parquet_files),
        'bucket_count': len(bucket_dirs),
    }


def find_database_path(root: str, db_name: str) -> Optional[str]:
    """在根目录下自动查找数据库目录。
    
    Args:
        root: 根目录路径
        db_name: 数据库名称 ('miiv', 'eicu' 等)
        
    Returns:
        找到的数据库路径，未找到返回 None
    """
    root_path = Path(root)
    if not root_path.exists():
        return None
    
    patterns = DATABASE_PATTERNS.get(db_name, [db_name])
    
    # 首先检查直接子目录
    for subdir in root_path.iterdir():
        if not subdir.is_dir():
            continue
        
        subdir_lower = subdir.name.lower()
        
        # 检查是否匹配数据库模式
        for pattern in patterns:
            if pattern in subdir_lower:
                # 检查是否有版本子目录（如 2.2, 3.1）
                version_dirs = [d for d in subdir.iterdir() 
                               if d.is_dir() and d.name[0].isdigit()]
                if version_dirs:
                    # 使用最新版本
                    latest = sorted(version_dirs, 
                                   key=lambda x: x.name, 
                                   reverse=True)[0]
                    return str(latest)
                else:
                    # 没有版本目录，直接使用
                    return str(subdir)
    
    return None


def get_database_info(database: str) -> Dict[str, Any]:
    """获取数据库的元信息。
    
    Args:
        database: 数据库名称
        
    Returns:
        dict: 包含 name, id_col, required_tables 等信息
    """
    id_col_map = {
        'miiv': 'stay_id',
        'eicu': 'patientunitstayid',
        'aumc': 'admissionid',
        'hirid': 'patientid',
        'mimic': 'icustay_id',
        'sic': 'CaseID',
    }
    
    return {
        'name': DATABASE_NAMES.get(database, database.upper()),
        'id_col': id_col_map.get(database, 'stay_id'),
        'required_tables': REQUIRED_PARQUET_TABLES.get(database, {}),
        'patterns': DATABASE_PATTERNS.get(database, [database]),
    }
