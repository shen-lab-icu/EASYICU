"""
Subprocess worker functions for EasyICU webapp.

These functions MUST be in a separate module from app.py so that
Windows `multiprocessing.Process(start_method='spawn')` can import them
WITHOUT importing Streamlit. app.py has module-level `import streamlit`
and `st.set_page_config()` calls that crash in non-Streamlit subprocesses.

On Linux, `multiprocessing` uses 'fork' (copies parent memory), so this
is not an issue. On Windows, 'spawn' creates a fresh Python interpreter
that must import the module containing the target function.
"""


def _subprocess_load_module(concepts, database, data_path, patient_ids_filter,
                            batch_size, output_dir):
    """在子进程中加载一个模块的概念，结果写入 parquet 文件。

    子进程退出后 OS 完整回收所有内存（包括 pymalloc arena 碎片），
    彻底解决跨模块碎片累积导致的 RSS 膨胀问题。
    """
    import os, sys, json
    os.environ.setdefault('EASYICU_DATA_PATH', os.environ.get('EASYICU_DATA_PATH', ''))
    os.environ.setdefault('EASYICU_DUCKDB_THREADS', '4')
    os.environ.setdefault('EASYICU_DUCKDB_MEMORY_LIMIT', '2GB')
    _src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '')
    if _src not in sys.path:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import pandas as pd
    from easyicu import load_concepts as _lc

    kwargs = dict(
        data_path=data_path, database=database,
        concepts=concepts, verbose=False, merge=False, concept_workers=1,
    )
    if patient_ids_filter:
        kwargs['patient_ids'] = patient_ids_filter
    if batch_size:
        kwargs['batch_size'] = batch_size

    result = _lc(**kwargs)

    saved = {}
    if isinstance(result, dict):
        for c, df in result.items():
            if hasattr(df, 'data') and isinstance(df.data, pd.DataFrame):
                df = df.data
            elif hasattr(df, 'to_pandas'):
                df = df.to_pandas()
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                path = os.path.join(output_dir, f"{c}.parquet")
                df.to_parquet(path, index=False)
                saved[c] = path

    with open(os.path.join(output_dir, '_manifest.json'), 'w') as f:
        json.dump(saved, f)


def _subprocess_load_and_export_module(concepts, database, data_path,
                                       patient_ids_filter, batch_size,
                                       export_dir, export_format, group_name,
                                       cohort_exclude_ids, overwrite,
                                       cohort_suffix, dep_concepts_to_cache,
                                       deps_cache_dir):
    """在子进程中完成 load + merge + export 全部工作。

    主进程不接触任何 DataFrame，彻底消除 pymalloc arena 碎片在主进程中的累积。
    子进程退出后 OS 完整回收所有内存。

    返回: 通过 _manifest.json 传回元数据 (exported_file, patient_ids, concepts, rows, empty_concepts)
    """
    import os, sys, json, shutil
    os.environ.setdefault('EASYICU_DATA_PATH', os.environ.get('EASYICU_DATA_PATH', ''))
    os.environ.setdefault('EASYICU_DUCKDB_THREADS', '4')
    os.environ.setdefault('EASYICU_DUCKDB_MEMORY_LIMIT', '2GB')
    _src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '')
    if _src not in sys.path:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import numpy as np
    import pandas as pd
    from easyicu import load_concepts as _lc

    # ── 1. 加载概念 ──
    kwargs = dict(
        data_path=data_path, database=database,
        concepts=concepts, verbose=False, merge=False, concept_workers=1,
    )
    if patient_ids_filter:
        kwargs['patient_ids'] = patient_ids_filter
    if batch_size:
        kwargs['batch_size'] = batch_size

    result = _lc(**kwargs)

    concept_dfs = {}
    if isinstance(result, dict):
        for c, df in result.items():
            if hasattr(df, 'data') and isinstance(df.data, pd.DataFrame):
                df = df.data
            elif hasattr(df, 'to_pandas'):
                df = df.to_pandas()
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                concept_dfs[c] = df

    # 缓存特殊概念依赖的 parquet（供 AKI/CircFailure 子进程复用）
    if deps_cache_dir and dep_concepts_to_cache:
        os.makedirs(deps_cache_dir, exist_ok=True)
        for cname, cdf in concept_dfs.items():
            if cname in dep_concepts_to_cache:
                try:
                    cdf.to_parquet(os.path.join(deps_cache_dir, f"{cname}.parquet"), index=False)
                except Exception:
                    pass

    empty_concepts = [c for c in concepts if c not in concept_dfs]

    # ── 2. 应用 cohort filter ──
    if cohort_exclude_ids:
        id_cands = ['stay_id', 'hadm_id', 'icustay_id', 'patientunitstayid',
                     'admissionid', 'patientid', 'CaseID']
        for cname, df in list(concept_dfs.items()):
            for idc in id_cands:
                if idc in df.columns:
                    concept_dfs[cname] = df[~df[idc].isin(cohort_exclude_ids)].copy()
                    break

    if not concept_dfs:
        # 无数据 → 写空 manifest
        with open(os.path.join(export_dir, f'_manifest_{group_name}.json'), 'w') as f:
            json.dump({'exported_file': None, 'patient_ids': [], 'concepts': [],
                       'rows': 0, 'empty_concepts': empty_concepts}, f)
        return

    # ── 3. 合并宽表 (从 _export_module_to_disk 提取的核心逻辑) ──
    id_candidates = ['stay_id', 'hadm_id', 'icustay_id', 'patientunitstayid',
                     'admissionid', 'patientid', 'CaseID']
    time_candidates = ['time', 'charttime', 'starttime', 'start', 'endtime',
                       'itemtime', 'datetime', 'Offset', 'measuredat_minutes', 'measuredat',
                       'givenat', 'enteredentryat']
    unified_time_col = 'charttime'

    # 统一时间列名称
    for cname in list(concept_dfs.keys()):
        cdf = concept_dfs[cname].copy()
        if unified_time_col in cdf.columns:
            other_time_cols = [tc for tc in time_candidates
                               if tc in cdf.columns and tc != unified_time_col]
            if other_time_cols:
                cdf = cdf.drop(columns=other_time_cols)
        else:
            for tc in time_candidates:
                if tc in cdf.columns:
                    cdf = cdf.rename(columns={tc: unified_time_col})
                    other_time_cols = [t for t in time_candidates
                                       if t in cdf.columns and t != unified_time_col]
                    if other_time_cols:
                        cdf = cdf.drop(columns=other_time_cols)
                    break
        concept_dfs[cname] = cdf

    # 确定主键列
    merge_cols = []
    _id_col = None
    _time_col = None
    potential_id_cols = set()
    potential_time_cols = set()
    for cdf in concept_dfs.values():
        for col in id_candidates:
            if col in cdf.columns:
                potential_id_cols.add(col)
                break
        for col in time_candidates:
            if col in cdf.columns:
                potential_time_cols.add(col)
                break
    for col in id_candidates:
        if col in potential_id_cols:
            _id_col = col
            merge_cols.append(col)
            break
    for col in time_candidates:
        if col in potential_time_cols:
            _time_col = col
            merge_cols.append(col)
            break

    if not merge_cols:
        all_dfs = []
        for cname, cdf in concept_dfs.items():
            cdf = cdf.copy()
            cdf['_concept'] = cname
            all_dfs.append(cdf)
        merged_df = pd.concat(all_dfs, ignore_index=True)
    else:
        all_concept_dfs = []
        for concept_name, df in concept_dfs.items():
            if _id_col and _id_col not in df.columns:
                continue
            metadata_cols = ['valueuom', 'unit', 'units', 'category', 'type',
                            'dur_var', 'entertime']  # dur_var/entertime: WinTbl 内部列，不导出
            cols_to_drop = [c for c in df.columns if c in metadata_cols]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
            value_cols = [c for c in df.columns if c not in merge_cols]
            df_to_add = df.copy()
            if len(value_cols) == 1:
                df_to_add = df_to_add.rename(columns={value_cols[0]: concept_name})
            elif len(value_cols) > 1:
                if concept_name in value_cols:
                    keep_val_cols = [concept_name]
                else:
                    keep_val_cols = value_cols
                cols_to_keep = merge_cols + keep_val_cols
                df_to_add = df_to_add[[c for c in cols_to_keep if c in df_to_add.columns]]
                remaining_val_cols = [c for c in df_to_add.columns if c not in merge_cols]
                if len(remaining_val_cols) == 1 and remaining_val_cols[0] != concept_name:
                    df_to_add = df_to_add.rename(columns={remaining_val_cols[0]: concept_name})
                elif len(remaining_val_cols) > 1:
                    rename_map = {}
                    for c in remaining_val_cols:
                        if c != concept_name and not c.startswith(f"{concept_name}_"):
                            rename_map[c] = f"{concept_name}_{c}"
                    if rename_map:
                        df_to_add = df_to_add.rename(columns=rename_map)
            for mc in merge_cols:
                if mc not in df_to_add.columns:
                    if mc in {'charttime', 'time', 'starttime', 'endtime', 'itemtime'}:
                        df_to_add[mc] = 0.0
                    else:
                        df_to_add[mc] = np.nan
            keep_cols = merge_cols + [c for c in df_to_add.columns if c not in merge_cols]
            all_concept_dfs.append(df_to_add[keep_cols])

        if len(all_concept_dfs) == 0:
            merged_df = pd.DataFrame(columns=merge_cols + list(concept_dfs.keys()))
        elif len(all_concept_dfs) == 1:
            merged_df = all_concept_dfs[0]
        else:
            # 确保 merge_cols 数值类型一致
            time_related_cols = {'charttime', 'time', 'starttime', 'endtime', 'itemtime'}
            id_related_cols = set(id_candidates)
            for i, df in enumerate(all_concept_dfs):
                for col in merge_cols:
                    if col in df.columns:
                        col_dtype = df[col].dtype
                        if col in time_related_cols:
                            if col_dtype == 'object' or not pd.api.types.is_numeric_dtype(col_dtype):
                                all_concept_dfs[i][col] = pd.to_numeric(df[col], errors='coerce')
                        elif col in id_related_cols:
                            if col_dtype == 'object':
                                all_concept_dfs[i][col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                            elif pd.api.types.is_numeric_dtype(col_dtype):
                                all_concept_dfs[i][col] = df[col].astype('Int64')
                        else:
                            if col_dtype == 'object':
                                all_concept_dfs[i][col] = pd.to_numeric(df[col], errors='coerce')

            for i, df in enumerate(all_concept_dfs):
                for col in merge_cols:
                    if col in time_related_cols and pd.api.types.is_float_dtype(df[col]):
                        all_concept_dfs[i][col] = df[col].round(2)

            total_rows_sum = sum(len(df) for df in all_concept_dfs)
            use_fast_path = (total_rows_sum < 2_000_000)

            if use_fast_path:
                try:
                    processed_dfs = []
                    static_dfs = []
                    _empty_local = []
                    for df in all_concept_dfs:
                        df_temp = df.copy()
                        val_cols = [c for c in df_temp.columns if c not in merge_cols]
                        if not val_cols:
                            continue
                        is_static = False
                        if _time_col and _time_col in df_temp.columns:
                            if df_temp[_time_col].isna().all():
                                is_static = True
                        if is_static:
                            if _id_col and _id_col in df_temp.columns:
                                static_cols = [_id_col] + val_cols
                                static_df = df_temp[static_cols].drop_duplicates(subset=[_id_col], keep='last')
                                static_dfs.append(static_df)
                        else:
                            df_temp = df_temp.drop_duplicates(subset=merge_cols, keep='last')
                            for value_col in val_cols:
                                if len(df_temp) == 0:
                                    _empty_local.append(value_col)
                                    continue
                                single_val_df = df_temp[merge_cols + [value_col]].copy()
                                single_val_df['_concept'] = str(value_col)
                                single_val_df['_value'] = single_val_df[value_col]
                                single_val_df.drop(columns=[value_col], inplace=True)
                                processed_dfs.append(single_val_df)

                    if not processed_dfs and not static_dfs:
                        merged_df = pd.DataFrame(columns=merge_cols + list(concept_dfs.keys()))
                    else:
                        if processed_dfs:
                            stacked = pd.concat(processed_dfs, ignore_index=True)
                            merged_df = stacked.pivot_table(
                                index=merge_cols, columns='_concept',
                                values='_value', aggfunc='first'
                            ).reset_index()
                            for ec in _empty_local:
                                if ec not in merged_df.columns:
                                    merged_df[ec] = np.nan
                        else:
                            merged_df = None
                        if static_dfs:
                            from functools import reduce
                            static_merged = reduce(
                                lambda left, right: pd.merge(left, right, on=_id_col, how='outer'),
                                static_dfs
                            )
                            if merged_df is not None and _id_col in merged_df.columns:
                                merged_df = pd.merge(merged_df, static_merged, on=_id_col, how='left')
                            else:
                                merged_df = static_merged
                except Exception:
                    use_fast_path = False

            if not use_fast_path:
                if len(all_concept_dfs) > 10:
                    _batch_sz = 5
                    batches = []
                    for i in range(0, len(all_concept_dfs), _batch_sz):
                        batch = all_concept_dfs[i:i+_batch_sz]
                        from functools import reduce
                        try:
                            batch_merged = reduce(
                                lambda left, right: pd.merge(left, right, on=merge_cols, how='outer'),
                                batch
                            )
                            if len(batch_merged) > 0:
                                batch_merged = batch_merged.drop_duplicates(subset=merge_cols)
                            batches.append(batch_merged)
                        except Exception:
                            continue
                    if not batches:
                        merged_df = pd.DataFrame(columns=merge_cols + list(concept_dfs.keys()))
                    else:
                        merged_df = reduce(
                            lambda left, right: pd.merge(left, right, on=merge_cols, how='outer'),
                            batches
                        )
                else:
                    from functools import reduce
                    merged_df = reduce(
                        lambda left, right: pd.merge(left, right, on=merge_cols, how='outer'),
                        all_concept_dfs
                    )
                if merged_df is not None and len(merged_df) > 0:
                    merged_df = merged_df.drop_duplicates(subset=merge_cols)

    if merged_df is None or len(merged_df) == 0:
        merged_df = pd.DataFrame(columns=merge_cols + list(concept_dfs.keys()))

    # ── 4. 生成文件名并写入 ──
    concept_names_sorted = sorted(list(concept_dfs.keys()))
    if len(concept_names_sorted) <= 5:
        concepts_suffix = '_'.join(concept_names_sorted)
    else:
        concepts_suffix = '_'.join(concept_names_sorted[:4]) + f'_etc{len(concept_names_sorted)}'
    if cohort_suffix:
        safe_filename = f"{group_name}_{concepts_suffix}_{cohort_suffix}".replace('/', '_').replace('\\', '_')
    else:
        safe_filename = f"{group_name}_{concepts_suffix}".replace('/', '_').replace('\\', '_')
    if len(safe_filename) > 150:
        safe_filename = safe_filename[:150]

    if export_format == 'csv':
        file_path = os.path.join(export_dir, f"{safe_filename}.csv")
    elif export_format == 'parquet':
        file_path = os.path.join(export_dir, f"{safe_filename}.parquet")
    elif export_format == 'excel':
        file_path = os.path.join(export_dir, f"{safe_filename}.xlsx")
    else:
        file_path = os.path.join(export_dir, f"{safe_filename}.parquet")

    # 覆盖模式: 删除旧文件
    if overwrite:
        import glob
        for ext in ['.parquet', '.csv', '.xlsx']:
            for old_file in glob.glob(os.path.join(export_dir, f"{group_name}_*{ext}")):
                try:
                    os.unlink(old_file)
                except Exception:
                    pass

    # 收集患者ID
    patient_ids_list = []
    if len(merged_df) > 0:
        for _idc in id_candidates:
            if _idc in merged_df.columns:
                patient_ids_list = merged_df[_idc].dropna().unique().tolist()
                break

    # 写入文件
    if export_format == 'csv':
        merged_df.to_csv(file_path, index=False, encoding='utf-8-sig')
    elif export_format == 'parquet':
        for col in merged_df.columns:
            if merged_df[col].dtype == object:
                numeric_vals = pd.to_numeric(merged_df[col], errors='coerce')
                orig_valid = merged_df[col].notna().sum()
                if orig_valid > 0 and numeric_vals.notna().sum() >= orig_valid * 0.5:
                    merged_df[col] = numeric_vals
                else:
                    merged_df[col] = merged_df[col].astype(str)
        merged_df.to_parquet(file_path, index=False)
    elif export_format == 'excel':
        merged_df.to_excel(file_path, index=False)
    else:
        for col in merged_df.columns:
            if merged_df[col].dtype == object:
                numeric_vals = pd.to_numeric(merged_df[col], errors='coerce')
                orig_valid = merged_df[col].notna().sum()
                if orig_valid > 0 and numeric_vals.notna().sum() >= orig_valid * 0.5:
                    merged_df[col] = numeric_vals
                else:
                    merged_df[col] = merged_df[col].astype(str)
        merged_df.to_parquet(file_path, index=False)

    n_rows = len(merged_df)

    # ── 5. 写 manifest (metadata only) ──
    manifest = {
        'exported_file': file_path,
        'patient_ids': [int(x) if not np.isnan(x) else None for x in patient_ids_list] if patient_ids_list else [],
        'concepts': list(concept_dfs.keys()),
        'rows': n_rows,
        'empty_concepts': empty_concepts,
    }
    with open(os.path.join(export_dir, f'_manifest_{group_name}.json'), 'w') as f:
        json.dump(manifest, f)


def _subprocess_load_special(concepts, database, data_path, patient_ids_filter,
                             max_patients, output_dir, preloaded_parquet_dir=None):
    """在子进程中加载特殊概念（AKI, CircFailure 等），结果写入 parquet。

    优化: 优先从 preloaded_parquet_dir 读取已导出的依赖概念 parquet 文件，
    避免重复从数据库加载。仅缺失的依赖概念才从数据库加载。
    """
    import os, sys, json
    _src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '')
    if _src not in sys.path:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault('EASYICU_DUCKDB_THREADS', '4')
    os.environ.setdefault('EASYICU_DUCKDB_MEMORY_LIMIT', '2GB')

    import pandas as pd
    from easyicu.api import load_concepts as _lc

    # 收集 AKI 和 CircFailure 需要的全部依赖概念
    aki_deps = ['crea', 'urine', 'weight', 'rrt']
    circ_deps = ['lact', 'map', 'norepi_rate', 'epi_rate', 'dobu_rate', 'dopa_rate']

    has_aki = any(c in concepts for c in ['aki', 'aki_stage', 'aki_stage_creat',
                  'aki_stage_uo', 'aki_stage_rrt', 'uo_rt_6hr', 'uo_rt_12hr',
                  'uo_rt_24hr', 'creat_low_past_48hr', 'creat_low_past_7day'])
    has_circ = any(c in concepts for c in ['circ_failure', 'circ_event'])

    all_deps = []
    if has_aki:
        all_deps.extend(aki_deps)
    if has_circ:
        all_deps.extend(circ_deps)
    all_deps = list(dict.fromkeys(all_deps))  # dedupe preserving order

    # 1) 优先从已导出的 parquet 文件读取依赖概念（零数据库开销）
    preloaded = {}
    if preloaded_parquet_dir and os.path.isdir(preloaded_parquet_dir):
        for dep in all_deps:
            pq_path = os.path.join(preloaded_parquet_dir, f"{dep}.parquet")
            if os.path.exists(pq_path):
                try:
                    df = pd.read_parquet(pq_path)
                    if not df.empty:
                        preloaded[dep] = df
                except Exception:
                    pass

    # 2) 仅加载缺失的依赖概念（从数据库）
    missing_deps = [d for d in all_deps if d not in preloaded]
    if missing_deps:
        load_kwargs = dict(
            data_path=data_path, database=database,
            concepts=missing_deps, verbose=False, merge=False, concept_workers=1,
        )
        if patient_ids_filter:
            load_kwargs['patient_ids'] = patient_ids_filter
        if max_patients:
            load_kwargs['max_patients'] = max_patients
        try:
            result = _lc(**load_kwargs)
            if isinstance(result, dict):
                for c, df in result.items():
                    if hasattr(df, 'data'):
                        df = df.data
                    if hasattr(df, 'to_pandas'):
                        df = df.to_pandas()
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        preloaded[c] = df
        except Exception:
            pass  # fallback: AKI/CircFailure will load individually

    # 统一 preloaded 中所有概念的时间列名为 'charttime'
    _time_aliases_pre = ['starttime', 'measuredat_minutes', 'measuredat', 'datetime',
                         'observationoffset', 'Offset', 'start',
                         'givenat', 'enteredentryat']
    for _cname in list(preloaded.keys()):
        _cdf = preloaded[_cname]
        if 'charttime' not in _cdf.columns:
            for _alias in _time_aliases_pre:
                if _alias in _cdf.columns:
                    preloaded[_cname] = _cdf.rename(columns={_alias: 'charttime'})
                    break

    saved = {}

    if has_aki:
        try:
            from easyicu.kdigo_aki import load_kdigo_aki
            aki_kwargs = dict(database=database, data_path=data_path, verbose=False,
                            preloaded_data=preloaded)
            if patient_ids_filter:
                id_col = list(patient_ids_filter.keys())[0]
                aki_kwargs['patient_ids'] = patient_ids_filter[id_col]
            if max_patients:
                aki_kwargs['max_patients'] = max_patients
            aki_df = load_kdigo_aki(**aki_kwargs)
            if isinstance(aki_df, pd.DataFrame) and not aki_df.empty:
                for c in concepts:
                    if c in ['aki', 'aki_stage', 'aki_stage_creat', 'aki_stage_uo',
                             'aki_stage_rrt', 'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
                             'creat_low_past_48hr', 'creat_low_past_7day']:
                        path = os.path.join(output_dir, f"{c}.parquet")
                        aki_df.to_parquet(path, index=False)
                        saved[c] = path
        except Exception as e:
            print(f"AKI loading failed: {e}")

    if has_circ:
        try:
            from easyicu.circ_failure import load_circ_failure
            circ_kwargs = dict(database=database, data_path=data_path, verbose=False,
                             preloaded_data=preloaded)
            if patient_ids_filter:
                id_col = list(patient_ids_filter.keys())[0]
                circ_kwargs['patient_ids'] = patient_ids_filter[id_col]
            if max_patients:
                circ_kwargs['max_patients'] = max_patients
            circ_df = load_circ_failure(**circ_kwargs)
            if isinstance(circ_df, pd.DataFrame) and not circ_df.empty:
                for c in concepts:
                    if c in ['circ_failure', 'circ_event']:
                        path = os.path.join(output_dir, f"{c}.parquet")
                        circ_df.to_parquet(path, index=False)
                        saved[c] = path
        except Exception as e:
            print(f"CircFailure loading failed: {e}")

    # Sep3 从缓存读取 SOFA，不重新计算
    sep_concepts = [c for c in concepts if c in ('sep3_sofa1', 'sep3_sofa2')]
    if sep_concepts:
        try:
            _sep3_load_list = ['susp_inf']
            if 'sep3_sofa1' in sep_concepts:
                _sep3_load_list.append('sofa')
            if 'sep3_sofa2' in sep_concepts:
                _sep3_load_list.append('sofa2')
            _sep3_load_list = list(dict.fromkeys(_sep3_load_list))

            # 从缓存读取
            sep_dfs = {}
            _time_aliases_all = ['charttime', 'time', 'starttime', 'datetime',
                                 'measuredat_minutes', 'measuredat', 'Offset']
            for _dep in _sep3_load_list:
                _cached = None
                if _dep in preloaded:
                    _cached = preloaded[_dep]
                elif preloaded_parquet_dir:
                    _pq = os.path.join(preloaded_parquet_dir, f"{_dep}.parquet")
                    if os.path.exists(_pq):
                        try:
                            _cached = pd.read_parquet(_pq)
                        except Exception:
                            pass
                if _cached is not None and len(_cached) > 0:
                    for _ta in _time_aliases_all:
                        if _ta in _cached.columns and _ta != 'charttime':
                            _cached = _cached.rename(columns={_ta: 'charttime'})
                            break
                    sep_dfs[_dep] = _cached

            _have_all_cache = all(d in sep_dfs for d in _sep3_load_list)

            if _have_all_cache:
                import numpy as np
                print(f"Sep3: using cached data ({', '.join(f'{k}={len(v)}rows' for k,v in sep_dfs.items())})")
                susp_df = sep_dfs['susp_inf']
                _id_col_s = None
                for _ic in ['stay_id', 'patientunitstayid', 'admissionid', 'patientid', 'icustay_id', 'CaseID']:
                    if _ic in susp_df.columns:
                        _id_col_s = _ic
                        break
                _time_col_s = 'charttime' if 'charttime' in susp_df.columns else None

                if _id_col_s and _time_col_s:
                    merged = susp_df.copy()
                    for _sn in ['sofa', 'sofa2']:
                        if _sn in sep_dfs:
                            _sdf = sep_dfs[_sn]
                            _vcols = [c for c in _sdf.columns if c not in [_id_col_s, 'charttime', 'valueuom', 'unit']]
                            if _vcols:
                                _keep = [c for c in [_id_col_s, 'charttime'] + _vcols if c in _sdf.columns]
                                _sdf = _sdf[_keep].copy()
                                if _sn not in _sdf.columns and len(_vcols) == 1:
                                    _sdf = _sdf.rename(columns={_vcols[0]: _sn})
                                _mc = [c for c in [_id_col_s, 'charttime'] if c in _sdf.columns]
                                if _mc:
                                    merged = pd.merge(merged, _sdf, on=_mc, how='left')

                    _susp_v = 'susp_inf'
                    if _susp_v not in merged.columns:
                        _sc_cands = [c for c in merged.columns if c not in [_id_col_s, 'charttime', 'sofa', 'sofa2']]
                        if _sc_cands:
                            merged = merged.rename(columns={_sc_cands[0]: 'susp_inf'})

                    if 'susp_inf' in merged.columns:
                        _sm = merged['susp_inf'].fillna(0).astype(bool)
                        _rc = [_id_col_s, 'charttime']
                        if 'sep3_sofa1' in sep_concepts and 'sofa' in merged.columns:
                            _s1 = _sm & (merged['sofa'].fillna(0) >= 2)
                            _sep1_df = merged[_s1][_rc].copy()
                            _sep1_df['sep3_sofa1'] = True
                            if not _sep1_df.empty:
                                path = os.path.join(output_dir, 'sep3_sofa1.parquet')
                                _sep1_df.to_parquet(path, index=False)
                                saved['sep3_sofa1'] = path
                        if 'sep3_sofa2' in sep_concepts and 'sofa2' in merged.columns:
                            _s2 = _sm & (merged['sofa2'].fillna(0) >= 2)
                            _sep2_df = merged[_s2][_rc].copy()
                            _sep2_df['sep3_sofa2'] = True
                            if not _sep2_df.empty:
                                path = os.path.join(output_dir, 'sep3_sofa2.parquet')
                                _sep2_df.to_parquet(path, index=False)
                                saved['sep3_sofa2'] = path
            else:
                # Fallback: batch load
                _missing = [d for d in _sep3_load_list if d not in sep_dfs]
                if _missing:
                    _batch_sz = 5000
                    _pid_list = None
                    if patient_ids_filter:
                        _pid_list = list(next(iter(patient_ids_filter.values())))
                    elif max_patients:
                        _pid_list = None  # let load_concepts handle max_patients

                    _load_kw = dict(data_path=data_path, database=database,
                                    concepts=_missing, verbose=False, merge=False, concept_workers=1)
                    if _pid_list:
                        _id_col_k = list(patient_ids_filter.keys())[0]
                        _total_p = len(_pid_list)
                        for _bi in range(0, _total_p, _batch_sz):
                            _batch_pids = _pid_list[_bi:_bi + _batch_sz]
                            _load_kw_b = dict(_load_kw)
                            _load_kw_b['patient_ids'] = {_id_col_k: _batch_pids}
                            try:
                                _br = _lc(**_load_kw_b)
                                if isinstance(_br, dict):
                                    for _bk, _bv in _br.items():
                                        if hasattr(_bv, 'data'):
                                            _bv = _bv.data
                                        if isinstance(_bv, pd.DataFrame) and not _bv.empty:
                                            if _bk in sep_dfs:
                                                sep_dfs[_bk] = pd.concat([sep_dfs[_bk], _bv], ignore_index=True)
                                            else:
                                                sep_dfs[_bk] = _bv
                            except Exception:
                                pass
                    else:
                        if max_patients:
                            _load_kw['max_patients'] = max_patients
                        try:
                            _br = _lc(**_load_kw)
                            if isinstance(_br, dict):
                                for _bk, _bv in _br.items():
                                    if hasattr(_bv, 'data'):
                                        _bv = _bv.data
                                    if isinstance(_bv, pd.DataFrame) and not _bv.empty:
                                        sep_dfs[_bk] = _bv
                        except Exception:
                            pass
        except Exception as e:
            print(f"Sep3 loading failed: {e}")
            import traceback
            traceback.print_exc()

    with open(os.path.join(output_dir, '_manifest.json'), 'w') as f:
        json.dump(saved, f)
