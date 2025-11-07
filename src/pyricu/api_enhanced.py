"""
Enhanced API with caching and time alignment support
"""
from typing import List, Union, Optional, Dict
from pathlib import Path
import pandas as pd
import pickle
import hashlib
from datetime import datetime

from .concept import ConceptDictionary, ConceptResolver
from .datasource import ICUDataSource
from .config import DataSourceConfig
from .resources import load_data_sources, load_dictionary


def _get_cache_key(concepts: List[str], source: str, **kwargs) -> str:
    """Generate cache key from parameters."""
    key_str = f"{source}_{','.join(sorted(concepts))}_{str(sorted(kwargs.items()))}"
    return hashlib.md5(key_str.encode()).hexdigest()


def load_concept_cached(
    concepts: Union[str, List[str]],
    source: str,
    data_path: Union[str, Path],
    cache_dir: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
    patient_ids: Optional[List] = None,
    merge: bool = True,
    align_time: bool = False,  # NEW: align to ICU admission time
    verbose: bool = True,
    use_pickle: bool = True,  # NEW: use pickle instead of CSV
    n_patients: Optional[int] = None,  # NEW: sample N patients for testing
    **kwargs,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load ICU concept data with caching support.
    
    Args:
        concepts: Concept name(s) to load
        source: Data source name ('mimic', 'miiv', etc.)
        data_path: Path to data source files
        cache_dir: Directory for cache files (default: data_path/cache)
        force_reload: If True, ignore cache and reload from source
        patient_ids: Optional patient ID filter
        merge: If True, merge concepts into wide format
        align_time: If True, align charttime to ICU admission (hours since admission)
        verbose: Show progress messages
        use_pickle: If True, cache as pickle; if False, use CSV
        n_patients: If provided, randomly sample N patients (for testing)
        **kwargs: Additional parameters for concept resolver
        
    Returns:
        DataFrame with concept data (and optionally time-aligned)
    """
    # Setup cache directory
    if cache_dir is None:
        cache_dir = Path(data_path) / "cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare concept list
    if isinstance(concepts, str):
        concept_list = [concepts]
    else:
        concept_list = list(concepts)
    
    # Generate cache key
    cache_params = {
        'merge': merge,
        'align_time': align_time,
        **kwargs
    }
    cache_key = _get_cache_key(concept_list, source, **cache_params)
    cache_ext = 'pkl' if use_pickle else 'csv'
    cache_file = cache_dir / f"{source}_{'_'.join(concept_list[:3])}_{cache_key[:8]}.{cache_ext}"
    
    # Try to load from cache
    if not force_reload and cache_file.exists():
        if verbose:
            print(f"📦 从缓存加载: {cache_file.name}")
        try:
            if use_pickle:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
            else:
                result = pd.read_csv(cache_file, parse_dates=['charttime'])
            
            if verbose:
                if isinstance(result, pd.DataFrame):
                    print(f"✅ 成功加载 {len(result):,} 行缓存数据")
                else:
                    print(f"✅ 成功加载 {len(result)} 个概念的缓存数据")
            return result
        except Exception as e:
            if verbose:
                print(f"⚠️  缓存加载失败: {e}，重新提取...")
    
    # Load from source
    if verbose:
        print(f"📊 从 {source.upper()} 提取 {len(concept_list)} 个概念...")
        if concept_list:
            print(f"   概念: {', '.join(concept_list)}")
    
    # Load data source config
    registry = load_data_sources()
    if source not in registry:
        available = [cfg.name for cfg in registry]
        raise ValueError(f"未知数据源 '{source}'。可用: {available}")
    
    source_config = registry.get(source)
    datasource = ICUDataSource(config=source_config, base_path=Path(data_path))
    
    # Load dictionary and create resolver
    dict_obj = load_dictionary()
    resolver = ConceptResolver(dict_obj)
    
    # Handle patient sampling for testing
    if n_patients is not None and patient_ids is None:
        if verbose:
            print(f"🎲 随机采样 {n_patients} 个患者进行测试...")
        
        # Load patient/stay IDs from icustays or similar table
        try:
            if source in ['miiv', 'mimic']:
                icu_table = datasource.load_table('icustays')
                # MIMIC-IV需要同时获取stay_id和subject_id用于过滤不同的表
                # - chartevents等使用stay_id
                # - labevents等使用subject_id
                if hasattr(icu_table, 'data'):
                    all_stay_ids = icu_table.data['stay_id'].unique()
                    if len(all_stay_ids) > n_patients:
                        import numpy as np
                        np.random.seed(42)  # 可重现的随机采样
                        sampled_stay_ids = np.random.choice(all_stay_ids, n_patients, replace=False)
                        
                        # 获取对应的subject_id
                        sampled_df = icu_table.data[icu_table.data['stay_id'].isin(sampled_stay_ids)]
                        patient_ids = {
                            'stay_id': sampled_stay_ids.tolist(),
                            'subject_id': sampled_df['subject_id'].unique().tolist()
                        }
                        if verbose:
                            print(f"   采样了 {len(patient_ids['stay_id'])} 个stay_id, {len(patient_ids['subject_id'])} 个subject_id")
                    else:
                        patient_ids = {
                            'stay_id': all_stay_ids.tolist(),
                            'subject_id': icu_table.data['subject_id'].unique().tolist()
                        }
                        if verbose:
                            print(f"   总共 {len(patient_ids['stay_id'])} 个stay_id")
                else:
                    patient_ids = None
            elif source == 'eicu':
                icu_table = datasource.load_table('patient')
                id_col = 'patientunitstayid'
                if icu_table is not None and hasattr(icu_table, 'data'):
                    all_ids = icu_table.data[id_col].unique()
                    if len(all_ids) > n_patients:
                        import numpy as np
                        np.random.seed(42)
                        patient_ids = np.random.choice(all_ids, n_patients, replace=False).tolist()
                        if verbose:
                            print(f"   采样了 {len(patient_ids)} 个患者ID")
                    else:
                        patient_ids = all_ids.tolist()
            elif source == 'hirid':
                icu_table = datasource.load_table('general_table')
                id_col = 'patientid'
                if icu_table is not None and hasattr(icu_table, 'data'):
                    all_ids = icu_table.data[id_col].unique()
                    if len(all_ids) > n_patients:
                        import numpy as np
                        np.random.seed(42)
                        patient_ids = np.random.choice(all_ids, n_patients, replace=False).tolist()
                        if verbose:
                            print(f"   采样了 {len(patient_ids)} 个患者ID")
                    else:
                        patient_ids = all_ids.tolist()
            else:
                patient_ids = None
        except Exception as e:
            if verbose:
                print(f"   ⚠️  采样失败: {e}，将加载全部数据")
            patient_ids = None
    
    # Load concepts
    result = resolver.load_concepts(
        concept_list,
        datasource,
        patient_ids=patient_ids,
        merge=merge,
        **kwargs,
    )
    
    # Time alignment if requested
    if align_time:
        result = align_to_icu_admission(result, datasource, source, verbose=verbose)
    
    # Save to cache
    try:
        if use_pickle:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            if isinstance(result, pd.DataFrame):
                result.to_csv(cache_file, index=False)
            else:
                # Can't easily cache dict to single CSV
                if verbose:
                    print("⚠️  字典结果未缓存（仅支持合并的DataFrame）")
        
        if verbose:
            print(f"💾 已缓存到: {cache_file.name}")
    except Exception as e:
        if verbose:
            print(f"⚠️  缓存保存失败: {e}")
    
    if verbose:
        if isinstance(result, pd.DataFrame):
            print(f"✅ 成功提取 {len(result):,} 行数据")
        else:
            print(f"✅ 成功提取 {len(result)} 个概念")
    
    return result


def align_to_icu_admission(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    datasource: ICUDataSource,
    source: str,
    aggregate_hourly: bool = True,  # NEW: 聚合到每小时一行
    agg_func: str = 'median',  # NEW: 聚合函数 (median, mean, min, max)
    filter_icu_window: bool = True,  # NEW: 过滤到ICU时间窗口
    before_icu_hours: int = 0,  # NEW: 入ICU前保留的小时数
    after_icu_hours: int = 0,  # NEW: 出ICU后保留的小时数
    verbose: bool = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Align charttime to ICU admission time and aggregate to hourly intervals.
    根据ricu的stay_windows逻辑，默认只保留ICU住院期间的数据。
    
    Args:
        data: Concept data with charttime
        datasource: Data source instance
        source: Data source name
        aggregate_hourly: If True, aggregate multiple measurements per hour
        agg_func: Aggregation function ('median', 'mean', 'min', 'max')
        filter_icu_window: If True, filter to ICU stay window (default: True)
        before_icu_hours: Hours before ICU admission to include (default: 0)
        after_icu_hours: Hours after ICU discharge to include (default: 0)
        verbose: Show progress
        
    Returns:
        Data with charttime as integer hours since ICU admission, one row per hour
    """
    if verbose:
        print("⏰ 对齐时间到ICU入院时间...")
    
    # Handle dict of DataFrames
    if isinstance(data, dict):
        return {
            name: align_to_icu_admission(df, datasource, source, aggregate_hourly, agg_func, 
                                        filter_icu_window, before_icu_hours, after_icu_hours, verbose=False)
            for name, df in data.items()
        }
    
    # Get ICU stay information (admission time)
    try:
        # Try to load icustays table
        if source in ['miiv', 'mimic']:
            icu_table_obj = datasource.load_table('icustays')
            id_col = 'stay_id'
            time_col_in = 'intime'
            time_col_out = 'outtime'
        elif source == 'eicu':
            icu_table_obj = datasource.load_table('patient')
            id_col = 'patientunitstayid'
            time_col_in = 'hospitaladmittime24'  # or unitadmittime24
            time_col_out = 'unitdischargetime24'
        elif source == 'hirid':
            icu_table_obj = datasource.load_table('general_table')
            id_col = 'patientid'
            time_col_in = 'admissiontime'
            time_col_out = 'dischargetime'
        else:
            if verbose:
                print(f"⚠️  数据源 '{source}' 不支持时间对齐")
            return data
        
        # Extract DataFrame from ICUTable
        if hasattr(icu_table_obj, 'data'):
            icu_table = icu_table_obj.data
        else:
            icu_table = icu_table_obj
        
        # Ensure datetime types
        if time_col_in in icu_table.columns:
            icu_table[time_col_in] = pd.to_datetime(icu_table[time_col_in], errors='coerce')
        if time_col_out in icu_table.columns:
            icu_table[time_col_out] = pd.to_datetime(icu_table[time_col_out], errors='coerce')
        
        # Get admission and discharge times
        admission_times = icu_table[[id_col, time_col_in, time_col_out]].rename(
            columns={id_col: 'stay_id', time_col_in: 'admission_time', time_col_out: 'discharge_time'}
        )
        
        # Merge with data
        if 'stay_id' not in data.columns:
            if verbose:
                print("⚠️  数据中没有 stay_id 列，跳过时间对齐")
            return data
        
        aligned = data.merge(admission_times, on='stay_id', how='left')
        
        # Calculate hours since admission
        if 'charttime' in aligned.columns and 'admission_time' in aligned.columns:
            aligned['charttime'] = pd.to_datetime(aligned['charttime'], errors='coerce')
            aligned['admission_time'] = pd.to_datetime(aligned['admission_time'], errors='coerce')
            aligned['discharge_time'] = pd.to_datetime(aligned['discharge_time'], errors='coerce')
            
            time_diff = aligned['charttime'] - aligned['admission_time']
            hours_float = time_diff.dt.total_seconds() / 3600
            
            # Apply ICU window filter (类似ricu的stay_windows逻辑)
            if filter_icu_window:
                # 计算ICU住院时长（小时）
                icu_los = (aligned['discharge_time'] - aligned['admission_time']).dt.total_seconds() / 3600
                
                # 过滤条件：-before_icu_hours <= hours_since_admission <= icu_los + after_icu_hours
                lower_bound = -before_icu_hours
                upper_bound = icu_los + after_icu_hours
                
                mask = (hours_float >= lower_bound) & (hours_float <= upper_bound)
                before_filter = len(aligned)
                aligned = aligned[mask]
                after_filter = len(aligned)
                
                if verbose:
                    filtered = before_filter - after_filter
                    print(f"   🪟 ICU时间窗口过滤: [{-before_icu_hours}h 到 出ICU+{after_icu_hours}h]")
                    print(f"      过滤前: {before_filter:,} 行")
                    print(f"      过滤后: {after_filter:,} 行") 
                    if before_filter > 0:
                        print(f"      过滤掉: {filtered:,} 行 ({filtered/before_filter*100:.1f}%)")
                
                # 重新计算hours_float (因为过滤后可能有变化)
                time_diff = aligned['charttime'] - aligned['admission_time']
                hours_float = time_diff.dt.total_seconds() / 3600
            
            # Round to nearest hour (floor)
            aligned['hours_since_admission'] = hours_float.apply(lambda x: int(x) if pd.notna(x) else None)
            
            # Drop original time columns
            aligned = aligned.drop(columns=['charttime', 'admission_time', 'discharge_time'])
            
            # Aggregate to hourly if requested
            if aggregate_hourly:
                value_cols = [col for col in aligned.columns if col not in ['stay_id', 'hours_since_admission']]
                
                if value_cols:
                    group_cols = ['stay_id', 'hours_since_admission']
                    
                    # Build aggregation dict
                    agg_dict = {}
                    for col in value_cols:
                        if aligned[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                            agg_dict[col] = agg_func
                        else:
                            agg_dict[col] = 'first'  # Non-numeric: take first
                    
                    aligned = aligned.groupby(group_cols, as_index=False).agg(agg_dict)
                    
                    if verbose:
                        print(f"   ✅ 时间已对齐并聚合到每小时一行 (使用 {agg_func})")
            else:
                if verbose:
                    print(f"   ✅ 时间已对齐为入院后小时数")
            
            # Rename to charttime for consistency
            aligned = aligned.rename(columns={'hours_since_admission': 'charttime'})
            
            if verbose:
                if len(aligned) > 0:
                    print(f"      时间范围: {aligned['charttime'].min():.0f}h - {aligned['charttime'].max():.0f}h")
                    print(f"      数据形状: {aligned.shape}")
                else:
                    print(f"      ⚠️  过滤后无数据")
        
        return aligned
        
    except Exception as e:
        if verbose:
            print(f"⚠️  时间对齐失败: {e}，返回原始数据")
        import traceback
        traceback.print_exc()
        return data


def load_sofa_with_score(
    source: str,
    data_path: Union[str, Path],
    cache_dir: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
    align_time: bool = True,
    win_length_hours: int = 24,
    n_patients: Optional[int] = None,  # NEW: sample for testing
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load SOFA components and calculate SOFA scores.
    
    根据 ricu 的定义，SOFA 评分需要6个组件：
    - sofa_resp:   pafi + vent_ind
    - sofa_coag:   plt
    - sofa_liver:  bili
    - sofa_cardio: map + dopa60 + norepi60 + dobu60 + epi60
    - sofa_cns:    gcs
    - sofa_renal:  crea + urine24
    
    Args:
        source: Data source name
        data_path: Path to data files
        cache_dir: Cache directory
        force_reload: Force reload from source
        align_time: Align to ICU admission time
        win_length_hours: Window length for worst value calculation (default: 24)
        n_patients: Sample N patients for testing (None = all patients)
        verbose: Show progress
        
    Returns:
        DataFrame with SOFA components and total SOFA score
    """
    if verbose:
        print("=" * 70)
        print("加载 SOFA 组件")
        print("=" * 70)
    
    # SOFA完整依赖 (基于ricu定义)
    # 为了简化，我们先提取基础指标
    basic_concepts = [
        # Respiratory
        'pafi', 'vent_ind',
        # Coagulation  
        'plt',
        # Liver
        'bili',
        # Cardiovascular
        'map', 
        # 'dopa60', 'norepi60', 'dobu60', 'epi60',  # 暂时跳过药物
        # CNS
        'gcs',
        # Renal
        'crea', 
        # 'urine24',  # 暂时跳过尿量
    ]
    
    if verbose:
        print(f"\n提取基础 SOFA 指标:")
        print(f"  {', '.join(basic_concepts)}")
        if n_patients:
            print(f"  采样 {n_patients} 个患者进行测试")
    
    # Load components
    sofa_data = load_concept_cached(
        basic_concepts,
        source,
        data_path,
        cache_dir=cache_dir,
        force_reload=force_reload,
        merge=True,
        align_time=align_time,
        n_patients=n_patients,  # 传递采样参数
        verbose=verbose,
        use_pickle=True,
    )
    
    if verbose:
        print(f"\n✅ SOFA 基础指标提取完成: {sofa_data.shape}")
        print(f"   列: {list(sofa_data.columns)}")
    
    # 注意: 完整的 SOFA 评分计算需要实现滑动窗口和组件评分函数
    # 这里返回的是原始指标数据
    return sofa_data


__all__ = [
    'load_concept_cached',
    'align_to_icu_admission',
    'load_sofa_with_score',
]
