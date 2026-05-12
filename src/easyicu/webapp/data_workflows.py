"""Data loading, validation, and cohort filtering workflows for the EasyICU webapp."""

from __future__ import annotations

from typing import Any

from easyicu.webapp.services import count_unique_concepts, normalize_column_name


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to extracted data workflows."""
    protected = {'check_data_status', 'convert_data_with_progress', 'apply_cohort_filter', 'validate_database_path', 'load_from_exported', 'load_data', 'load_data_for_preview', "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def check_data_status(data_path: str, database: str, app_context: dict[str, Any] | None = None) -> dict:
    """检查数据目录的状态，返回文件统计信息。"""
    if app_context is not None:
        _install_app_context(app_context)
    
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
    parquet_files = [f for f in path.glob('*.parquet') if not f.name.startswith('.')]
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
    csv_files = [f for f in list(path.glob('*.csv')) + list(path.glob('*.csv.gz')) if not f.name.startswith('.')]
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


def convert_data_with_progress(data_path: str, database: str, app_context: dict[str, Any] | None = None):
    """带进度条的数据转换功能。"""
    if app_context is not None:
        _install_app_context(app_context)
    
    import time

    lang = st.session_state.get('language', 'en')

    conv_title = "🔄 Data Conversion" if lang == 'en' else "🔄 数据转换"
    st.markdown(f"### {conv_title}")

    warn_msg = "⚠️ **Note**: Converting large datasets may take a long time (30min~2hrs), please be patient." if lang == 'en' else "⚠️ **注意**：转换大型数据集可能需要较长时间（30分钟~2小时），请耐心等待。"
    st.warning(warn_msg)

    info_msg = "💡 Using DuckDB for memory-efficient conversion. Large tables will be bucket-partitioned automatically." if lang == 'en' else "💡 使用 DuckDB 进行内存安全转换，大表将自动进行分桶优化。"
    st.info(info_msg)

    # 定义需要分桶转换的大表
    BUCKET_TABLES = {
        'miiv': {
            'chartevents': ('itemid', 100),
            'labevents': ('itemid', 100),
            'inputevents': ('itemid', 50),
        },
        'eicu': {
            'nursecharting': ('nursingchartcelltypevalname', 30),  # 按字符串hash
            'lab': ('labname', 50),
        },
        'aumc': {
            'numericitems': ('itemid', 100),
            'listitems': ('itemid', 50),
        },
        'hirid': {
            'observations': ('variableid', 100),
            'pharma': ('pharmaid', 50),
        },
        'mimic': {
            'chartevents': ('itemid', 100),
            'labevents': ('itemid', 100),
        },
        'sic': {
            'data_float_h': ('dataid', 50),
            'laboratory': ('laboratoryid', 50),
        },
    }

    try:
        from easyicu.duckdb_converter import DuckDBConverter
        from easyicu.bucket_converter import convert_to_buckets, BucketConfig
        import gc

        # 自动检测可用内存，预留 3GB 给 OS/Python
        from easyicu.memory_manager import get_available_memory_mb
        _avail_gb = get_available_memory_mb() / 1024
        _duckdb_mem_gb = max(2.0, _avail_gb - 3.0)

        converter = DuckDBConverter(
            data_path=data_path,
            memory_limit_gb=_duckdb_mem_gb,
            verbose=True
        )

        # 获取需要转换的文件列表
        csv_files = converter._find_csv_files()
        total_files = len(csv_files)

        if total_files == 0:
            err_msg = "No CSV files found to convert" if lang == 'en' else "未找到需要转换的 CSV 文件"
            st.error(err_msg)
            return

        # 分类文件：大表用分桶，小表用普通转换
        bucket_tables_config = BUCKET_TABLES.get(database, {})
        bucket_files = []
        normal_files = []

        for csv_file in csv_files:
            stem = csv_file.stem.lower().replace('.csv', '')
            if stem in bucket_tables_config:
                bucket_files.append((csv_file, bucket_tables_config[stem]))
            else:
                normal_files.append(csv_file)

        # 按文件大小升序处理：先快速完成小文件给用户反馈，把整段大文件留到最后单独跑
        def _size_key(p):
            try:
                return p.stat().st_size
            except OSError:
                return 0
        normal_files.sort(key=_size_key)
        bucket_files.sort(key=lambda pair: _size_key(pair[0]))

        detect_msg = f"📊 Detected **{len(normal_files)}** normal + **{len(bucket_files)}** large tables" if lang == 'en' else f"📊 共检测到 **{len(normal_files)}** 个普通表 + **{len(bucket_files)}** 个大表"
        st.markdown(detect_msg)

        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        details_container = st.container()

        converted = 0
        skipped = 0
        failed = 0
        total = len(normal_files) + len(bucket_files)
        current = 0

        # 1. 先转换普通表
        for csv_file in normal_files:
            current += 1
            file_name = csv_file.name
            file_size_mb = csv_file.stat().st_size / (1024 * 1024)

            processing_msg = f"**Processing**: `{file_name}` ({file_size_mb:.1f} MB) [{current}/{total}]" if lang == 'en' else f"**正在处理**: `{file_name}` ({file_size_mb:.1f} MB) [{current}/{total}]"
            status_text.markdown(processing_msg)

            parquet_path = converter._get_parquet_path(csv_file)
            if parquet_path.exists():
                skipped += 1
                with details_container:
                    exists_msg = "exists" if lang == 'en' else "已存在"
                    st.caption(f"⏭️ {file_name} ({exists_msg})")
            else:
                try:
                    result = converter.convert_file(csv_file)
                    if result['status'] == 'success':
                        converted += 1
                        with details_container:
                            rows_label = "rows" if lang == 'en' else "行"
                            st.caption(f"✅ {file_name}: {result['row_count']:,} {rows_label}")
                    else:
                        failed += 1
                        with details_container:
                            st.caption(f"❌ {file_name}: {result.get('error', 'unknown')[:40]}")
                except Exception as e:
                    failed += 1
                    with details_container:
                        st.caption(f"❌ {file_name}: {str(e)[:40]}")

            progress_bar.progress(current / total)
            gc.collect()

        # 2. 分桶转换大表
        for csv_file, (partition_col, num_buckets) in bucket_files:
            current += 1
            file_name = csv_file.name
            file_size_mb = csv_file.stat().st_size / (1024 * 1024)
            stem = csv_file.stem.lower().replace('.csv', '')

            processing_msg = f"**Bucketing**: `{file_name}` ({file_size_mb:.1f} MB) → {num_buckets} buckets [{current}/{total}]" if lang == 'en' else f"**分桶转换**: `{file_name}` ({file_size_mb:.1f} MB) → {num_buckets} 个桶 [{current}/{total}]"
            status_text.markdown(processing_msg)

            # 检查分桶目录是否真正完成（通过 _COMPLETE 标记），无标记则视为不完整需重做
            bucket_dir = csv_file.parent / f"{stem}_bucket"
            sentinel = bucket_dir / '_COMPLETE'
            if bucket_dir.exists() and sentinel.exists():
                skipped += 1
                with details_container:
                    bucket_exists_msg = "bucket exists" if lang == 'en' else "分桶目录已存在"
                    st.caption(f"⏭️ {file_name} ({bucket_exists_msg})")
            else:
                # 旧目录无 _COMPLETE 标记时清理再重做，避免半成品蒙混过关
                if bucket_dir.exists():
                    import shutil as _shutil
                    _shutil.rmtree(bucket_dir, ignore_errors=True)
                try:
                    # AUMC 使用 latin-1 编码（µmol 等特殊字符）
                    _encoding = 'latin-1' if database == 'aumc' else None
                    # 内存留空 → BucketConfig.__post_init__ 通过 _memory_tier 自适应
                    config = BucketConfig(
                        num_buckets=num_buckets,
                        partition_col=partition_col,
                        encoding=_encoding,
                    )
                    result = convert_to_buckets(
                        source_path=csv_file,
                        output_dir=bucket_dir,
                        config=config,
                        overwrite=True
                    )
                    if result.success:
                        converted += 1
                        with details_container:
                            bucket_label = "buckets" if lang == 'en' else "个桶"
                            rows_label = "rows" if lang == 'en' else "行"
                            st.caption(f"✅ {file_name} → {result.num_buckets} {bucket_label}, {result.total_rows:,} {rows_label}")
                    else:
                        failed += 1
                        with details_container:
                            st.caption(f"❌ {file_name}: {result.error[:40] if result.error else 'unknown'}")
                except Exception as e:
                    failed += 1
                    with details_container:
                        st.caption(f"❌ {file_name}: {str(e)[:40]}")

            progress_bar.progress(current / total)
            gc.collect()

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

    except ImportError as e:
        import_err = f"Data converter module not installed: {e}" if lang == 'en' else f"数据转换模块未安装: {e}"
        st.error(import_err)
    except Exception as e:
        conv_err = f"Conversion error: {str(e)}" if lang == 'en' else f"转换过程出错: {str(e)}"
        st.error(conv_err)


def apply_cohort_filter(data_path, database, candidate_ids=None, app_context: dict[str, Any] | None = None):
    """
    Apply cohort filters from st.session_state to real patient data.

    Reads ICU metadata tables (icustays, patients, admissions) and filters
    patient IDs based on the active cohort criteria (age, first_icu_stay,
    los_min, gender, survived, disease cohort, ICD keywords).

    Args:
        data_path: Path to the database directory (e.g. /home/zhuhb/icudb/mimiciv/3.1)
        database: Database name ('miiv', 'eicu', 'aumc', 'hirid', 'mimic', 'sic')
        candidate_ids: Optional pre-filtered list of IDs to further filter

    Returns:
        dict with keys: id_col, filtered_ids, total_before, total_after, filter_details
        or None if cohort filtering is disabled / no active filter
    """
    if app_context is not None:
        _install_app_context(app_context)
    
    # Check if filtering is enabled
    if not st.session_state.get('cohort_enabled', False):
        return None

    cf = st.session_state.get('cohort_filter', {})
    if not cf:
        return None

    # Check if any filter is actually active
    has_active = (
        cf.get('age_min') is not None or
        cf.get('age_max') is not None or
        cf.get('first_icu_stay') is not None or
        cf.get('los_min') is not None or
        cf.get('gender') is not None or
        cf.get('survived') is not None or
        cf.get('disease_cohort') not in (None, '', 'none') or
        bool(str(cf.get('icd_include_query', cf.get('icd_query', ''))).strip()) or
        bool(str(cf.get('icd_exclude_query', '')).strip())
    )
    if not has_active:
        return None

    data_path = Path(data_path)

    # Database-specific configuration
    DB_META = {
        'miiv': {
            'id_col': 'stay_id', 'subject_col': 'subject_id',
            'icu_table': 'icustays.parquet', 'patient_table': 'patients.parquet',
            'admission_table': 'admissions.parquet',
        },
        'eicu': {
            'id_col': 'patientunitstayid', 'subject_col': 'uniquepid',
            'icu_table': 'patient.parquet', 'patient_table': None,
            'admission_table': None,
        },
        'aumc': {
            'id_col': 'admissionid', 'subject_col': 'patientid',
            'icu_table': 'admissions.parquet', 'patient_table': None,
            'admission_table': None,
        },
        'hirid': {
            'id_col': 'patientid', 'subject_col': 'patientid',
            'icu_table': 'general.parquet', 'patient_table': None,
            'admission_table': None,
            'icu_table_fallback': 'general_table.csv',  # fallback if parquet missing
        },
        'mimic': {
            'id_col': 'icustay_id', 'subject_col': 'subject_id',
            'icu_table': 'icustays.parquet', 'patient_table': 'patients.parquet',
            'admission_table': 'admissions.parquet',
        },
        'sic': {
            'id_col': 'CaseID', 'subject_col': 'PatientID',
            'icu_table': 'cases.parquet', 'patient_table': None,
            'admission_table': None,
        },
    }

    meta = DB_META.get(database)
    if not meta:
        print(f"[COHORT] Unknown database: {database}")
        return None

    id_col = meta['id_col']
    subject_col = meta['subject_col']

    # Load ICU stays table
    icu_path = data_path / meta['icu_table']
    if not icu_path.exists():
        # Try fallback path (e.g., HiRID general_table.csv)
        fallback = meta.get('icu_table_fallback')
        if fallback:
            icu_path = data_path / fallback
        if not icu_path.exists():
            print(f"[COHORT] ICU table not found: {icu_path}")
            return None

    if str(icu_path).endswith('.csv') or str(icu_path).endswith('.csv.gz'):
        icu_df = pd.read_csv(icu_path)
    else:
        icu_df = pd.read_parquet(icu_path)

    # Normalize column names to lowercase for comparison (except for SICdb)
    if database != 'sic':
        icu_df.columns = [c.lower() for c in icu_df.columns]
        id_col_lower = id_col.lower()
        subject_col_lower = subject_col.lower()
    else:
        id_col_lower = id_col
        subject_col_lower = subject_col

    # Load optional tables
    patient_df = None
    admission_df = None
    if meta.get('patient_table'):
        pt_path = data_path / meta['patient_table']
        if pt_path.exists():
            patient_df = pd.read_parquet(pt_path)
            if database != 'sic':
                patient_df.columns = [c.lower() for c in patient_df.columns]
    if meta.get('admission_table'):
        adm_path = data_path / meta['admission_table']
        if adm_path.exists():
            admission_df = pd.read_parquet(adm_path)
            if database != 'sic':
                admission_df.columns = [c.lower() for c in admission_df.columns]

    # Start with all IDs
    if candidate_ids is not None:
        mask = icu_df[id_col_lower].isin(candidate_ids)
        icu_df = icu_df[mask].copy()
        icu_df = icu_df.reset_index(drop=True)  # reset index so merge-based filters align

    total_before = len(icu_df)
    keep_mask = pd.Series(True, index=icu_df.index)
    filter_details = []  # list of (label_en, label_cn, excluded_count)

    # ---------- Age Filter ----------
    if cf.get('age_min') is not None or cf.get('age_max') is not None:
        age_series = _get_age_series(icu_df, database, patient_df, admission_df,
                                     id_col_lower, subject_col_lower)
        if age_series is not None:
            before_count = keep_mask.sum()
            age_valid = age_series.notna()
            if cf.get('age_min') is not None:
                keep_mask &= age_valid & (age_series >= cf['age_min'])
            if cf.get('age_max') is not None:
                keep_mask &= age_valid & (age_series <= cf['age_max'])
            excluded = int(before_count - keep_mask.sum())
            age_range = f"{cf.get('age_min', 0)}-{cf.get('age_max', '∞')}"
            filter_details.append((f"Age {age_range}", f"年龄 {age_range}", excluded))

    # ---------- First ICU Stay Filter ----------
    if cf.get('first_icu_stay') is not None:
        first_mask = _get_first_icu_mask(icu_df, database, id_col_lower, subject_col_lower)
        if first_mask is not None:
            before_count = keep_mask.sum()
            first_valid = first_mask.notna()
            if cf['first_icu_stay']:
                keep_mask &= first_valid & first_mask.fillna(False).astype(bool)
            else:
                keep_mask &= first_valid & ~first_mask.fillna(True).astype(bool)
            excluded = int(before_count - keep_mask.sum())
            en_label = "First ICU stay only" if cf['first_icu_stay'] else "Non-first ICU stay only"
            cn_label = "仅首次ICU入住" if cf['first_icu_stay'] else "仅非首次ICU入住"
            filter_details.append((en_label, cn_label, excluded))

    # ---------- Min LOS Filter ----------
    if cf.get('los_min') is not None:
        los_series = _get_los_hours_series(icu_df, database)
        if los_series is not None:
            before_count = keep_mask.sum()
            keep_mask &= los_series.notna() & (los_series >= cf['los_min'])
            excluded = int(before_count - keep_mask.sum())
            filter_details.append((f"LOS ≥ {cf['los_min']}h", f"住院时长 ≥ {cf['los_min']}h", excluded))

    # ---------- Gender Filter ----------
    if cf.get('gender') is not None:
        sex_series = _get_sex_series(icu_df, database, patient_df,
                                     id_col_lower, subject_col_lower)
        if sex_series is not None:
            before_count = keep_mask.sum()
            keep_mask &= sex_series.notna() & (sex_series == cf['gender'])
            excluded = int(before_count - keep_mask.sum())
            gender_en = "Male" if cf['gender'] == 'M' else "Female"
            gender_cn = "男性" if cf['gender'] == 'M' else "女性"
            filter_details.append((f"{gender_en} only", f"仅{gender_cn}", excluded))

    # ---------- Survival Filter ----------
    if cf.get('survived') is not None:
        death_series = _get_death_series(icu_df, database, patient_df, admission_df,
                                         id_col_lower, subject_col_lower)
        if death_series is not None:
            before_count = keep_mask.sum()
            death_valid = death_series.notna()
            death_bool = pd.array(death_series.fillna(False), dtype=bool)
            if cf['survived']:
                keep_mask &= death_valid & ~death_bool  # survived = known not dead
            else:
                keep_mask &= death_valid & death_bool   # deceased = known dead
            excluded = int(before_count - keep_mask.sum())
            en_label = "Survived only" if cf['survived'] else "Deceased only"
            cn_label = "仅存活" if cf['survived'] else "仅死亡"
            filter_details.append((en_label, cn_label, excluded))

    # ---------- Sepsis / ICD pre-filter ----------
    disease_cohort = cf.get('disease_cohort')
    disease_cfg = DISEASE_COHORT_CONFIG.get(disease_cohort or 'none', {})
    if disease_cohort == 'sepsis':
        try:
            from easyicu.patient_filter import PatientFilter
            before_count = keep_mask.sum()
            pf = PatientFilter(database=database, data_path=data_path, verbose=False)
            sepsis_ids = pf._get_sepsis_patients()
            if sepsis_ids:
                keep_mask &= icu_df[id_col_lower].isin(sepsis_ids)
            else:
                keep_mask &= False
            excluded = int(before_count - keep_mask.sum())
            filter_details.append(("Sepsis-3 / sepsis cohort", "脓毒症队列", excluded))
        except Exception as e:
            print(f"[COHORT] Sepsis pre-filter skipped ({database}): {e}")

    icd_template_tokens = disease_cfg.get('icd_tokens', [])
    if disease_cohort not in (None, '', 'none', 'sepsis') and icd_template_tokens and database in ICD_FILTER_DATABASES:
        before_count = keep_mask.sum()
        matched_ids = _match_ids_by_icd_tokens(data_path, database, icu_df, id_col_lower, icd_template_tokens)
        if matched_ids:
            keep_mask &= icu_df[id_col_lower].isin(matched_ids)
        else:
            keep_mask &= False
        excluded = int(before_count - keep_mask.sum())
        filter_details.append((disease_cfg.get('label_en', 'Disease cohort'),
                               disease_cfg.get('label_zh', '疾病队列'),
                               excluded))

    icd_include_tokens = _split_query_tokens(cf.get('icd_include_query', cf.get('icd_query', '')))
    if icd_include_tokens and database in ICD_FILTER_DATABASES:
        before_count = keep_mask.sum()
        try:
            matched_ids = _match_ids_by_icd_tokens(data_path, database, icu_df, id_col_lower, icd_include_tokens)
            if matched_ids:
                keep_mask &= icu_df[id_col_lower].isin(matched_ids)
            else:
                keep_mask &= False
            excluded = int(before_count - keep_mask.sum())
            filter_details.append((f"ICD include ({', '.join(icd_include_tokens)})",
                                   f"ICD 包含 ({', '.join(icd_include_tokens)})",
                                   excluded))
        except Exception as e:
            print(f"[COHORT] ICD include filter skipped ({database}): {e}")

    icd_exclude_tokens = _split_query_tokens(cf.get('icd_exclude_query', ''))
    if icd_exclude_tokens and database in ICD_FILTER_DATABASES:
        before_count = keep_mask.sum()
        try:
            matched_ids = _match_ids_by_icd_tokens(data_path, database, icu_df, id_col_lower, icd_exclude_tokens)
            if matched_ids:
                keep_mask &= ~icu_df[id_col_lower].isin(matched_ids)
            excluded = int(before_count - keep_mask.sum())
            filter_details.append((f"ICD exclude ({', '.join(icd_exclude_tokens)})",
                                   f"ICD 排除 ({', '.join(icd_exclude_tokens)})",
                                   excluded))
        except Exception as e:
            print(f"[COHORT] ICD exclude filter skipped ({database}): {e}")

    filtered_ids = icu_df.loc[keep_mask, id_col_lower].unique().tolist()
    total_after = len(filtered_ids)

    pct = total_after / total_before * 100 if total_before > 0 else 0
    print(f"[COHORT] {database}: {total_before} → {total_after} patients ({pct:.1f}% retained)")

    return {
        'id_col': id_col,    # original case (e.g. CaseID)
        'filtered_ids': filtered_ids,
        'total_before': total_before,
        'total_after': total_after,
        'filter_details': filter_details,
    }


def validate_database_path(data_path: str, database: str, app_context: dict[str, Any] | None = None) -> dict:
    """
    验证数据路径是否包含指定数据库所需的文件。
    严格检查每个模块所需的所有表。

    返回:
        dict: {'valid': bool, 'message': str, 'suggestion': str (可选)}
    """
    if app_context is not None:
        _install_app_context(app_context)
    
    path = Path(data_path)
    resolver = globals().get('find_database_path')
    if callable(resolver):
        try:
            resolved_path = Path(resolver(str(path), database))
            if resolved_path.exists():
                path = resolved_path
        except Exception:
            pass

    lang = st.session_state.get('language', 'en')
    download_info = _get_database_download_info(database, lang)

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
    required_csv_files = {
        'miiv': ['icustays.csv', 'chartevents.csv', 'labevents.csv', 'prescriptions.csv', 'inputevents.csv'],
        'eicu': ['patient.csv', 'vitalPeriodic.csv', 'lab.csv'],
        'aumc': ['admissions.csv', 'numericitems.csv', 'drugitems.csv'],
        'hirid': ['general_table.csv', 'pharma_records.csv'],
        'mimic': ['icustays.csv', 'chartevents.csv', 'labevents.csv', 'prescriptions.csv'],
        'sic': ['cases.csv', 'data_float_h.csv', 'laboratory.csv', 'medication.csv'],
    }

    db_name = {
        'miiv': 'MIMIC-IV', 'eicu': 'eICU-CRD',
        'aumc': 'AmsterdamUMCdb', 'hirid': 'HiRID',
        'mimic': 'MIMIC-III', 'sic': 'SICdb'
    }.get(database, database.upper())

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

    # 分别统计 Parquet 可用和仅 CSV 可用的表
    parquet_available = parquet_names | parquet_dirs | bucket_dirs

    # HiRID 特殊处理：别名映射
    if database == 'hirid':
        if 'pharma' in parquet_available:
            parquet_available.add('pharma_records')
        # general.parquet (R ricu convention) 或 general_table.parquet (直接转换)
        if 'general' in parquet_available or 'general_table' in parquet_available:
            parquet_available.add('general_table')
    if database == 'hirid' and 'pharma' in csv_names:
        csv_names.add('pharma_records')

    # 检查各类别的表
    db_tables = required_parquet_tables.get(database, {})
    found_tables = []       # Parquet 可用
    csv_only_tables = []    # 仅 CSV 可用（无 Parquet）
    missing_tables = []     # 完全缺失
    missing_by_category = {}

    for category, tables in db_tables.items():
        for table in tables:
            tl = table.lower()
            if tl in parquet_available:
                found_tables.append(table)
            elif tl in csv_names:
                csv_only_tables.append(table)
            else:
                missing_tables.append(table)
                if category not in missing_by_category:
                    missing_by_category[category] = []
                missing_by_category[category].append(table)

    total_required = sum(len(tables) for tables in db_tables.values())

    # 如果所有表都有 Parquet（最佳情况）
    if len(missing_tables) == 0 and len(csv_only_tables) == 0:
        bucket_info = f", {len(bucket_dirs)} bucketed" if bucket_dirs else ""
        msg = f'✅ {db_name}: All {total_required} required tables found ({len(parquet_files)} Parquet files{bucket_info})' if lang == 'en' else f'✅ {db_name}: 所有 {total_required} 个必需表已找到 ({len(parquet_files)} 个 Parquet 文件{bucket_info})'
        return {
            'valid': True,
            'message': msg
        }

    # 有表仅通过 CSV 找到（无 Parquet）→ 需要转换
    if len(missing_tables) == 0 and len(csv_only_tables) > 0:
        csv_list = ', '.join(csv_only_tables[:5])
        if len(csv_only_tables) > 5:
            csv_list += f' (+{len(csv_only_tables)-5} more)'
        if lang == 'en':
            msg = f'⚠️ {db_name}: {len(csv_only_tables)} tables only found as CSV (no Parquet): {csv_list}. Click Validate & Setup to auto-convert.'
            sug = '💡 Click "Validate & Setup" to auto-convert CSV files'
        else:
            msg = f'⚠️ {db_name}: {len(csv_only_tables)} 个表仅有 CSV 格式（无 Parquet）: {csv_list}。点击「验证并设置」自动转换。'
            sug = '💡 点击「验证并设置」自动转换 CSV 文件'
        return {
            'valid': False,
            'message': msg,
            'suggestion': sug,
            'can_convert': True,
            'csv_path': str(path),
            'download_url': download_info['url'] if download_info else None,
            'download_label': download_info['label'] if download_info else None,
            'download_note': download_info['note'] if download_info else None,
        }

    # 核心表缺失是严重问题
    core_missing = missing_by_category.get('core', [])
    if core_missing:
        missing_str = ', '.join(core_missing)
        csv_hint = ""
        if csv_only_tables:
            csv_hint_tables = ', '.join(csv_only_tables[:3])
            csv_hint = f" ({csv_hint_tables} only as CSV)" if lang == 'en' else f"（{csv_hint_tables} 仅有CSV格式）"
        if lang == 'en':
            msg = f'❌ {db_name}: Missing core tables: {missing_str}{csv_hint}'
            sug = f'💡 Core tables are required. Please ensure data is properly converted.'
        else:
            msg = f'❌ {db_name}: 缺少核心表: {missing_str}{csv_hint}'
            sug = f'💡 核心表是必需的，请确保数据已正确转换。'
        return {
            'valid': False,
            'message': msg,
            'suggestion': sug,
            'can_convert': True,
            'csv_path': str(path),
            'missing_tables': missing_tables + csv_only_tables,
            'download_url': download_info['url'] if download_info else None,
            'download_label': download_info['label'] if download_info else None,
            'download_note': download_info['note'] if download_info else None,
        }

    # 部分表缺失（非核心）或仅有 CSV
    all_need_convert = missing_tables + csv_only_tables
    if len(found_tables) > 0 or len(csv_only_tables) > 0:
        parts = []
        if missing_tables:
            parts.append(', '.join(missing_tables[:3]) + (' (missing)' if lang == 'en' else '（缺失）'))
        if csv_only_tables:
            parts.append(', '.join(csv_only_tables[:3]) + (' (CSV only)' if lang == 'en' else '（仅CSV）'))
        detail_str = '; '.join(parts)
        if lang == 'en':
            msg = f'⚠️ {db_name}: {len(found_tables)}/{total_required} tables ready (Parquet), need conversion: {detail_str}'
            sug = f'💡 Click "Validate & Setup" to auto-convert missing/CSV tables'
        else:
            msg = f'⚠️ {db_name}: {len(found_tables)}/{total_required} 个表就绪（Parquet），需要转换: {detail_str}'
            sug = f'💡 点击「验证并设置」自动转换缺失或CSV格式的表'
        return {
            'valid': False,
            'message': msg,
            'suggestion': sug,
            'can_convert': True,
            'csv_path': str(path),
            'missing_tables': all_need_convert,
            'download_url': download_info['url'] if download_info else None,
            'download_label': download_info['label'] if download_info else None,
            'download_note': download_info['note'] if download_info else None,
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
        sug = '💡 Click "Validate & Setup" to auto-convert' if lang == 'en' else '💡 点击「验证并设置」自动转换'
        return {
            'valid': False,
            'message': msg,
            'suggestion': sug,
            'can_convert': True,
            'csv_path': str(path),
            'download_url': download_info['url'] if download_info else None,
            'download_label': download_info['label'] if download_info else None,
            'download_note': download_info['note'] if download_info else None,
        }

    # HiRID 特殊检测：可能只有 tar.gz 归档文件（尚未解压）
    if database == 'hirid':
        tar_gz_files = list(path.glob('*.tar.gz'))
        raw_stage = path / 'raw_stage'
        if raw_stage.exists():
            tar_gz_files.extend(raw_stage.glob('*.tar.gz'))
        hirid_archives = [f.name for f in tar_gz_files if any(
            kw in f.name.lower() for kw in ['observation', 'pharma', 'reference']
        )]
        if hirid_archives:
            archive_list = ', '.join(hirid_archives[:5])
            msg = (f'⚠️ {db_name}: Found archives ({archive_list}) that need extraction and conversion' if lang == 'en' else
                   f'⚠️ {db_name}: 发现归档文件 ({archive_list}) 需要解压和转换')
            sug = '💡 Click "Validate & Setup" to auto-extract and convert' if lang == 'en' else '💡 点击「验证并设置」自动解压和转换'
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
                'suggestion': sug,
                'download_url': download_info['url'] if download_info else None,
                'download_label': download_info['label'] if download_info else None,
                'download_note': download_info['note'] if download_info else None,
            }

    # 完全找不到相关文件
    lang = st.session_state.get('language', 'en')
    msg = f'❌ Required data files for {db_name} not found in this path' if lang == 'en' else f'❌ 在此路径下未找到 {db_name} 所需的数据文件'
    sug = '💡 Please verify: 1) Path is correct 2) Database type matches 3) Data is downloaded' if lang == 'en' else '💡 请确认: 1) 路径是否正确 2) 数据库类型是否匹配 3) 数据是否已下载'
    return {
        'valid': False,
        'message': msg,
        'suggestion': sug,
        'download_url': download_info['url'] if download_info else None,
        'download_label': download_info['label'] if download_info else None,
        'download_note': download_info['note'] if download_info else None,
    }


def load_from_exported(export_dir: str, max_patients: int = 50, selected_files: list = None, app_context: dict[str, Any] | None = None):
    """从已导出的数据文件加载数据（限制患者数用于快速预览）。

    从宽表中提取每个特征列，使其可以单独选择和可视化。
    对 Parquet 文件使用 PyArrow 行级过滤，避免全量读入内存。

    Args:
        export_dir: 导出目录路径
        max_patients: 最大患者数限制（默认50）
        selected_files: 要加载的文件名列表（不含扩展名），None表示全部加载
    """
    if app_context is not None:
        _install_app_context(app_context)
    
    try:
        import time
        load_start = time.time()

        export_path = Path(export_dir)
        raw_data = {}  # 原始文件数据
        read_failures: List[Dict[str, Any]] = []

        # ID列和时间列，不作为特征
        id_candidates = ['stay_id', 'hadm_id', 'icustay_id',
                        'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
        time_candidates = ['time', 'charttime', 'starttime', 'endtime',
                          'datetime', 'timestamp', 'index', 'Offset',
                          'measuredat_minutes', 'measuredat']
        exclude_cols = set(id_candidates + time_candidates)

        # --- Phase 0: 收集要加载的文件 ---
        target_files = []  # [(Path, suffix)]
        for file in export_path.iterdir():
            file_stem = file.stem
            if selected_files is not None and file_stem not in selected_files:
                continue
            if file.suffix in ('.csv', '.parquet', '.xlsx'):
                target_files.append(file)

        # --- Phase 1: 对 Parquet 文件预扫描，确定 ID 列并采样患者 ---
        # 只读一个小文件的 ID 列（<0.1s），避免全量读取 19 个大文件
        _sampled_ids = None  # set | None
        _id_col_for_filter = None
        _all_ids_count = 0  # 真实总患者数
        _need_filter = max_patients is not None and max_patients > 0

        if _need_filter:
            parquet_files = [f for f in target_files if f.suffix == '.parquet']
            if parquet_files:
                try:
                    import pyarrow.parquet as pq
                    # 用第一个 parquet 的 schema 确定 ID 列
                    _schema = pq.read_schema(parquet_files[0])
                    _schema_names = _schema.names
                    for _idc in id_candidates:
                        if _idc in _schema_names:
                            _id_col_for_filter = _idc
                            break
                    if _id_col_for_filter:
                        # 优先扫 demographics/outcome（小且包含全部患者）
                        _priority_names = {'demographics', 'outcome', 'circulatory'}
                        _scan_file = None
                        for _pf in parquet_files:
                            if _pf.stem in _priority_names:
                                _scan_file = _pf
                                break
                        if _scan_file is None:
                            # 没找到优先文件，选最小的
                            _scan_file = min(parquet_files, key=lambda f: f.stat().st_size)

                        _id_tbl = pq.read_table(_scan_file, columns=[_id_col_for_filter])
                        _all_ids = set(_id_tbl.column(_id_col_for_filter).to_pylist())
                        _all_ids_count = len(_all_ids)

                        if len(_all_ids) > max_patients:
                            import random
                            _sampled_ids = set(random.sample(sorted(_all_ids), max_patients))
                        else:
                            _sampled_ids = _all_ids  # 不需要裁剪
                except Exception:
                    pass  # 降级到旧路径（全量读取+后过滤）

        # --- Phase 2: 按文件类型高效加载 ---
        for file in target_files:
            file_stem = file.stem
            try:
                if file.suffix == '.parquet':
                    if _sampled_ids is not None and _id_col_for_filter:
                        import pyarrow.parquet as pq
                        _filters = [(_id_col_for_filter, "in", sorted(_sampled_ids))]
                        try:
                            tbl = pq.read_table(file, filters=_filters)
                            raw_data[file_stem] = tbl.to_pandas()
                        except Exception:
                            # Fallback for older PyArrow / edge-case parquet layouts.
                            tbl = pq.read_table(file)
                            if _id_col_for_filter in tbl.column_names:
                                import pyarrow.compute as pc
                                import pyarrow as pa

                                mask = pc.is_in(
                                    tbl.column(_id_col_for_filter),
                                    value_set=pa.array(sorted(_sampled_ids)),
                                )
                                tbl = tbl.filter(mask)
                            raw_data[file_stem] = tbl.to_pandas()
                    else:
                        raw_data[file_stem] = pd.read_parquet(file)
                elif file.suffix == '.csv':
                    if _need_filter and _sampled_ids is not None and _id_col_for_filter:
                        # CSV: 先读 header 确认 ID 列存在，再分块过滤
                        _hdr = pd.read_csv(file, nrows=0)
                        if _id_col_for_filter in _hdr.columns:
                            _chunks = []
                            for _chunk in pd.read_csv(file, chunksize=50000):
                                _chunks.append(_chunk[_chunk[_id_col_for_filter].isin(_sampled_ids)])
                            raw_data[file_stem] = pd.concat(_chunks, ignore_index=True) if _chunks else pd.DataFrame()
                        else:
                            raw_data[file_stem] = pd.read_csv(file)
                    else:
                        raw_data[file_stem] = pd.read_csv(file)
                elif file.suffix == '.xlsx':
                    raw_data[file_stem] = pd.read_excel(file)
            except Exception as _read_err:
                print(f'[load_from_exported] Failed to read {file.name}: {_read_err}')
                read_failures.append({
                    'file': file.name,
                    'suffix': file.suffix,
                    'error': str(_read_err),
                    'created_by': _get_parquet_created_by(file) if file.suffix == '.parquet' else None,
                })
                continue

        if not raw_data:
            lang = st.session_state.get('language', 'en')
            warn_msg = _build_export_read_failure_warning(target_files, read_failures, lang=lang)
            st.warning(warn_msg)
            return

        # 从宽表中提取每个特征列作为单独的concept
        data = {}

        # 首先确定全局ID列（用于患者筛选）
        id_col_found = 'stay_id'
        for file_name, df in raw_data.items():
            if isinstance(df, pd.DataFrame):
                for col in id_candidates:
                    if col in df.columns:
                        id_col_found = col
                        break
                break

        # 从每个宽表中提取特征列
        # 注意：每个文件可能有不同的时间列，需要单独检测
        # 🔧 2026-02-12: 添加列名规范化和去重逻辑
        for file_name, df in raw_data.items():
            if isinstance(df, pd.DataFrame):
                # 为当前文件找时间列（每个文件单独检测）
                file_time_col = None
                for col in time_candidates:
                    if col in df.columns:
                        file_time_col = col
                        break

                # 获取特征列（排除ID列、时间列和元数据列如_concept）
                meta_cols = {'_concept'}
                feature_cols = [c for c in df.columns if c not in exclude_cols and c not in meta_cols]

                # 为每个特征创建单独的DataFrame
                for feat_col in feature_cols:
                    # 🔧 规范化列名（去重）
                    normalized_col = normalize_column_name(feat_col)

                    # 如果规范化后的列名已存在，跳过（保留第一个遇到的）
                    if normalized_col in data:
                        continue

                    # 保留ID列、该文件的时间列和该特征列
                    keep_cols = []
                    if id_col_found in df.columns:
                        keep_cols.append(id_col_found)
                    if file_time_col and file_time_col in df.columns:
                        keep_cols.append(file_time_col)
                    keep_cols.append(feat_col)

                    feat_df = df[keep_cols].copy()
                    # 🔧 重命名特征列为规范化后的名称
                    if feat_col != normalized_col:
                        feat_df = feat_df.rename(columns={feat_col: normalized_col})
                    data[normalized_col] = feat_df

        # 获取患者列表
        patient_ids = set()

        for concept_df in data.values():
            if isinstance(concept_df, pd.DataFrame):
                if id_col_found in concept_df.columns:
                    patient_ids.update(concept_df[id_col_found].unique())

        # 如果预扫描阶段已采样过，真实总患者数来自预扫描
        if _all_ids_count > 0:
            all_patient_count = _all_ids_count
        else:
            all_patient_count = len(patient_ids)

        # 限制患者数用于可视化预览（max_patients=None 表示加载全部）
        # 若 parquet 已在读取阶段预过滤，这里不再需要二次裁剪
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
                # 即使DataFrame为空也保留，确保特征数量一致
                filtered_data[concept_name] = filtered_df
            else:
                # 对于没有ID列的DataFrame（如静态指标），直接保留
                filtered_data[concept_name] = df

        st.session_state.loaded_concepts = filtered_data
        st.session_state.loaded_data_origin = 'exported_files'
        st.session_state.patient_ids = preview_patient_ids
        st.session_state.all_patient_count = all_patient_count
        st.session_state.id_col = id_col_found

        # 🔧 FIX (2026-02-03): 设置 selected_concepts 以便侧边栏的导出按钮可用
        st.session_state.selected_concepts = list(filtered_data.keys())

        # 🔧 FIX (2026-02-12): 规范化后每列就是一个概念，直接统计列数
        # 由于在加载时已经去重，这里直接使用 len(filtered_data)
        unique_concept_count = len(filtered_data)

        # 🔧 FIX (2026-02-03): Load Data后重置导出触发状态，避免白屏
        # 注意：不应该重置 export_completed，因为 Quick Visualization 的 Load Data
        # 是独立于侧边栏数据提取器的功能，不应该影响导出完成状态
        st.session_state.trigger_export = False
        st.session_state['_exporting_in_progress'] = False
        # 清理跳过/覆盖模块状态（这些是导出过程中的临时状态，可以安全清理）
        if '_skipped_modules' in st.session_state:
            del st.session_state['_skipped_modules']
        if '_overwrite_modules' in st.session_state:
            del st.session_state['_overwrite_modules']
        if '_viz_import_export_auto_trigger' in st.session_state:
            del st.session_state['_viz_import_export_auto_trigger']

        load_elapsed = time.time() - load_start

        # 显示提示信息
        # 🔧 FIX (2026-02-12): 规范化后 concepts = columns (已去重)
        lang = st.session_state.get('language', 'en')
        if lang == 'en':
            st.success(f"✅ Loaded {unique_concept_count} concepts, {len(preview_patient_ids)}/{all_patient_count} patients ({load_elapsed:.1f}s)")
            if is_limited:
                st.info(f"💡 For better performance, preview is limited to {max_patients} patients. Full data has been exported to disk.")
        else:
            st.success(f"✅ 已加载 {unique_concept_count} 个概念，{len(preview_patient_ids)}/{all_patient_count} 个患者 ({load_elapsed:.1f}秒)")
            if is_limited:
                st.info(f"💡 为保证流畅性，可视化预览仅加载前 {max_patients} 个患者。完整数据已导出到磁盘，可使用Python/R进行完整分析。")

    except Exception as e:
        lang = st.session_state.get('language', 'en')
        err_msg = f"Loading failed: {e}" if lang == 'en' else f"加载失败: {e}"
        st.error(err_msg)


def load_data(app_context: dict[str, Any] | None = None):
    """Load data with parallel acceleration support - optimized batch loading."""
    if app_context is not None:
        _install_app_context(app_context)
    
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
            from easyicu import load_concepts
            import time
            import os

            concepts_list = st.session_state.selected_concepts
            n_concepts = len(concepts_list)

            load_start = time.time()

            # 🚀 优化：真正的批量加载 - 一次调用加载所有concepts
            # 🚀 性能优化：参照 extract_baseline_features.py 的配置
            patient_limit = st.session_state.get('patient_limit', 0)

            # 获取数据库信息
            data_path = Path(st.session_state.data_path)
            database = st.session_state.get('database', 'miiv')
            id_col_map = {
                'miiv': 'stay_id',
                'eicu': 'patientunitstayid',
                'aumc': 'admissionid',
                'hirid': 'patientid',
                'mimic': 'icustay_id',
                'sic': 'CaseID'
            }
            id_col = id_col_map.get(database, 'stay_id')
            patient_ids_filter = None

            # 👥 先从数据库选patient_limit个患者，再对这些患者做人群筛选
            try:
                # Step 1: 先选patient_limit个患者作为候选集
                candidate_ids = None
                if patient_limit and patient_limit > 0:
                    try:
                        icustays_files = _get_patient_id_table_files(database)
                        for f in icustays_files:
                            fp = data_path / f
                            if fp.exists():
                                icustays_df = pd.read_parquet(fp, columns=[id_col] if id_col else None)
                                if id_col in icustays_df.columns:
                                    all_patient_ids = icustays_df[id_col].unique().tolist()
                                    candidate_ids = _sample_patient_ids_random(all_patient_ids, patient_limit)
                                    break
                    except Exception:
                        pass

                # Step 2: 在候选集上应用人群筛选
                data_path_for_cohort = st.session_state.data_path
                database_for_cohort = st.session_state.get('database', 'miiv')
                cohort_result = apply_cohort_filter(data_path_for_cohort, database_for_cohort, candidate_ids=candidate_ids)
                if cohort_result is not None:
                    cohort_id_col = cohort_result['id_col']
                    filtered_ids = cohort_result['filtered_ids']
                    id_col = cohort_id_col
                    patient_ids_filter = {id_col: filtered_ids}
                    # Save cohort stats for completion message
                    n_before = len(candidate_ids) if candidate_ids else 0
                    n_after = len(filtered_ids)
                    st.session_state['_cohort_stats'] = {
                        'before': n_before, 'after': n_after, 'excluded': n_before - n_after,
                        'filter_details': cohort_result.get('filter_details', []),
                    }
                    # 🚫 Zero patients after cohort filter — abort extraction
                    if n_after == 0:
                        lang = st.session_state.get('language', 'en')
                        details_parts = []
                        for fd in cohort_result.get('filter_details', []):
                            label = fd[0] if lang == 'en' else fd[1]
                            details_parts.append(f"{label} (excluded {fd[2]})")
                        details_str = ", ".join(details_parts) if details_parts else ""
                        if lang == 'en':
                            msg = f"No patients meet the cohort criteria. {n_before} candidates were all excluded."
                            if details_str:
                                msg += f" Filters applied: {details_str}."
                            msg += " Please adjust your cohort filters in Step 2 and try again."
                        else:
                            msg = f"没有患者满足队列筛选条件。{n_before} 名候选患者全部被排除。"
                            if details_str:
                                msg += f" 已应用的筛选条件: {details_str}。"
                            msg += " 请在步骤2中调整筛选条件后重试。"
                        st.error(msg)
                        return
                elif candidate_ids is not None:
                    # No cohort filter active, use the candidate set directly
                    patient_ids_filter = {id_col: candidate_ids}
                    st.session_state['_cohort_stats'] = None
                else:
                    st.session_state['_cohort_stats'] = None
            except Exception as _cohort_err:
                print(f"[COHORT] Error in load_data: {_cohort_err}")
                # Fallback: just apply patient_limit without cohort filter
                if patient_limit and patient_limit > 0:
                    try:
                        icustays_files = _get_patient_id_table_files(database)
                        for f in icustays_files:
                            fp = data_path / f
                            if fp.exists():
                                icustays_df = pd.read_parquet(fp, columns=[id_col] if id_col else None)
                                if id_col in icustays_df.columns:
                                    all_patient_ids = icustays_df[id_col].unique().tolist()
                                    sample_ids = _sample_patient_ids_random(all_patient_ids, patient_limit)
                                    patient_ids_filter = {id_col: sample_ids}
                                    break
                    except Exception:
                        pass

            # �🚀 智能并行配置：根据系统资源和患者数量动态调整
            num_patients = len(patient_ids_filter.get(id_col, [])) if patient_ids_filter else None
            parallel_workers, parallel_backend = get_optimal_parallel_config(num_patients, task_type='load')

            try:
                # 🔧 按模块分组加载概念，显示进度条，跳过不可用/超时的概念
                data = {}
                failed_concepts = []
                empty_concepts = []  # 🆕 跟踪返回空结果的概念

                # 🚀 FIX 2026-02: 按模块分组加载 + 进度条 + 单模块超时
                # 解决全量AUMC (21K患者) 加载10+分钟用户感觉"卡住"的问题
                import signal

                # 将概念按模块分组
                concept_to_module = {}
                for mod_key, mod_concepts in CONCEPT_GROUPS_INTERNAL.items():
                    for c in mod_concepts:
                        concept_to_module[c] = mod_key

                module_concept_map = {}
                special_concepts_to_load = []
                for c in concepts_list:
                    if c in SPECIAL_CONCEPTS:
                        special_concepts_to_load.append(c)
                        continue
                    mod = concept_to_module.get(c, '_other')
                    if mod not in module_concept_map:
                        module_concept_map[mod] = []
                    module_concept_map[mod].append(c)

                # 模块加载优先级：快的先加载
                MODULE_LOAD_ORDER = [
                    'vitals', 'demographics', 'outcome',
                    'chemistry', 'hematology', 'blood_gas',
                    'medications', 'ventilator', 'respiratory',
                    'vasopressors', 'renal', 'neurological',
                    'other_scores', 'circulatory',
                    'sofa1_score', 'sofa2_score',
                    'sepsis3_sofa1', 'sepsis3_sofa2', 'sepsis_shared',
                    '_other',
                ]
                ordered_modules = [m for m in MODULE_LOAD_ORDER if m in module_concept_map]
                for m in module_concept_map:
                    if m not in ordered_modules:
                        ordered_modules.append(m)

                total_modules = len(ordered_modules) + (1 if special_concepts_to_load else 0)
                progress_placeholder = st.empty()
                status_placeholder = st.empty()

                # 单模块超时限制 (秒) - 防止单个模块阻塞整体流程
                MODULE_TIMEOUT = 300  # 5分钟

                def _process_load_result(result, concept_names):
                    """处理 load_concepts 返回结果（复用逻辑）"""
                    if isinstance(result, dict):
                        for cname, df in result.items():
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
                            else:
                                empty_concepts.append(cname)
                    elif isinstance(result, pd.DataFrame):
                        if len(result) > 0:
                            for concept in concept_names:
                                data[concept] = result
                        else:
                            empty_concepts.extend(concept_names)
                    for c in concept_names:
                        if c not in data and c not in empty_concepts:
                            empty_concepts.append(c)

                for mod_idx, mod_key in enumerate(ordered_modules):
                    if check_cancelled():
                        _handle_export_cancel()
                        return
                    mod_concepts = module_concept_map[mod_key]
                    mod_display = CONCEPT_GROUP_NAMES.get(mod_key, (mod_key, mod_key))
                    mod_name = mod_display[1] if lang != 'en' else mod_display[0]

                    elapsed = time.time() - load_start
                    progress_pct = (mod_idx) / total_modules
                    progress_placeholder.progress(progress_pct, text=f"{'Loading' if lang == 'en' else '正在加载'} {mod_name} ({mod_idx+1}/{total_modules}) - {elapsed:.0f}s")
                    status_placeholder.caption(f"📦 {mod_name}: {', '.join(mod_concepts[:5])}{'...' if len(mod_concepts) > 5 else ''}")

                    mod_start = time.time()
                    try:
                        load_kwargs = {
                            'data_path': st.session_state.data_path,
                            'database': st.session_state.get('database'),
                            'concepts': mod_concepts,
                            'verbose': False,
                            'merge': False,
                            'concept_workers': 1,
                            'parallel_workers': parallel_workers,
                            'parallel_backend': parallel_backend,
                        }
                        if patient_ids_filter:
                            load_kwargs['patient_ids'] = patient_ids_filter

                        # 🔧 FIX (2026-02-20): 使用后台线程执行，主线程每2秒发送UI更新保活WebSocket
                        # 解决全量患者导出时长时间无UI更新导致 "Connection timed out" 的问题
                        _thr_result = {'done': False, 'result': None, 'error': None}

                        def _load_in_thread(fn, kw, holder):
                            try:
                                holder['result'] = fn(**kw)
                            except Exception as _e:
                                holder['error'] = _e
                            finally:
                                holder['done'] = True

                        _t = threading.Thread(
                            target=_load_in_thread,
                            args=(load_concepts, load_kwargs, _thr_result),
                            daemon=True,
                        )
                        _t.start()

                        # 主线程每2秒发送UI更新 → 保活WebSocket连接
                        _ka_tick = 0
                        _spinners = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']
                        while not _thr_result['done']:
                            time.sleep(2)
                            _ka_tick += 1
                            _mod_elapsed = time.time() - mod_start
                            _total_elapsed = time.time() - load_start
                            _sp = _spinners[_ka_tick % 10]
                            if lang == 'en':
                                _ka_msg = f"{_sp} **Loading {mod_name}** ({len(mod_concepts)} concepts, module {mod_idx+1}/{total_modules}, elapsed {_total_elapsed:.0f}s, module {_mod_elapsed:.0f}s)"
                            else:
                                _ka_msg = f"{_sp} **正在加载 {mod_name}** ({len(mod_concepts)} 个概念, 模块 {mod_idx+1}/{total_modules}, 已用 {_total_elapsed:.0f}s, 本模块 {_mod_elapsed:.0f}s)"
                            status_placeholder.caption(_ka_msg)

                        _t.join(timeout=5)

                        if _thr_result['error']:
                            raise _thr_result['error']

                        _process_load_result(_thr_result['result'], mod_concepts)

                    except Exception as mod_e:
                        # 模块加载失败，逐个概念回退
                        for concept in mod_concepts:
                            try:
                                single_kwargs = {
                                    'data_path': st.session_state.data_path,
                                    'database': st.session_state.get('database'),
                                    'concepts': [concept],
                                    'verbose': False,
                                    'merge': False,
                                    'concept_workers': 1,
                                }
                                single_kwargs.update(_get_sepsis_runtime_options())
                                if patient_ids_filter:
                                    single_kwargs['patient_ids'] = patient_ids_filter
                                result = load_concepts(**single_kwargs)
                                _process_load_result(result, [concept])
                            except Exception:
                                failed_concepts.append(concept)
                                continue

                    progress_placeholder.progress((mod_idx + 1) / total_modules, text=f"✅ {mod_name} ({mod_idx+1}/{total_modules})")

                # 加载特殊概念 (AKI, circ_failure, sep3) — 也用 threading 保活
                if special_concepts_to_load:
                    progress_placeholder.progress((len(ordered_modules)) / total_modules, text=f"{'Loading special concepts...' if lang == 'en' else '正在加载特殊概念...'}")
                    try:
                        _sp_result = {'done': False, 'result': None, 'error': None}

                        def _load_special_thread(fn, kw, holder):
                            try:
                                holder['result'] = fn(**kw)
                            except Exception as _e:
                                holder['error'] = _e
                            finally:
                                holder['done'] = True

                        _sp_kwargs = dict(
                            concepts=special_concepts_to_load,
                            database=st.session_state.get('database', 'miiv'),
                            data_path=st.session_state.data_path,
                            patient_ids=patient_ids_filter,
                            max_patients=patient_limit if patient_limit and patient_limit > 0 else None,
                            verbose=False,
                        )
                        _sp_kwargs.update(_get_sepsis_runtime_options())

                        _sp_t = threading.Thread(
                            target=_load_special_thread,
                            args=(load_special_concepts, _sp_kwargs, _sp_result),
                            daemon=True,
                        )
                        _sp_t.start()

                        _sp_tick = 0
                        _sp_start = time.time()
                        while not _sp_result['done']:
                            time.sleep(2)
                            _sp_tick += 1
                            _sp_elapsed = time.time() - _sp_start
                            _sp_char = _spinners[_sp_tick % 10]
                            _sp_msg = f"{_sp_char} {'Loading special concepts' if lang == 'en' else '正在加载特殊概念'}... {_sp_elapsed:.0f}s"
                            status_placeholder.caption(_sp_msg)

                        _sp_t.join(timeout=5)

                        if _sp_result['error']:
                            raise _sp_result['error']

                        special_data = _sp_result['result']
                        for cname, df in special_data.items():
                            if isinstance(df, pd.DataFrame) and not df.empty:
                                data[cname] = df
                        failed_special = [c for c in special_concepts_to_load if c not in data]
                        failed_concepts.extend(failed_special)
                    except Exception:
                        failed_concepts.extend(special_concepts_to_load)

                # 完成进度条
                progress_placeholder.progress(1.0, text=f"✅ {'Done!' if lang == 'en' else '加载完成!'}")
                status_placeholder.empty()

                if failed_concepts:
                    skip_msg = f"⚠️ Skipped {len(failed_concepts)} unavailable: {', '.join(failed_concepts[:5])}" if lang == 'en' else f"⚠️ 跳过 {len(failed_concepts)} 个不可用: {', '.join(failed_concepts[:5])}"
                    st.warning(skip_msg)

                # 🆕 显示空结果概念提示
                if empty_concepts:
                    empty_msg = f"ℹ️ {len(empty_concepts)} concepts returned empty (not configured or no data): {', '.join(empty_concepts[:8])}" if lang == 'en' else f"ℹ️ {len(empty_concepts)} 个概念返回空结果（未配置或无数据）: {', '.join(empty_concepts[:8])}"
                    st.info(empty_msg)

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

            # 🔧 POST-FILTER: Remove patients whose cohort-critical features are None
            database = st.session_state.get('database', 'miiv')
            data = _post_filter_cohort_data(data, database)

            st.session_state.loaded_concepts = data
            st.session_state.loaded_data_origin = 'quick_load'

            # 获取患者列表 - 统计所有患者数，但UI选择器限制显示数量
            patient_ids = set()
            id_candidates = ['stay_id', 'hadm_id', 'icustay_id',
                           'patientunitstayid', 'admissionid', 'patientid', 'CaseID']

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


def load_data_for_preview(max_patients: int = 50, app_context: dict[str, Any] | None = None):
    """Load limited data for preview visualization (memory-friendly version)."""
    if app_context is not None:
        _install_app_context(app_context)
    
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
        from easyicu import load_concepts
        import time

        load_start = time.time()
        data = {}

        # 只加载前5个concept作为预览
        preview_concepts = selected[:5]

        # 先选max_patients个患者，再对这些患者做人群筛选
        patient_ids_filter = None
        id_col = 'stay_id'
        try:
            data_path = Path(st.session_state.data_path)
            database = st.session_state.get('database', 'miiv')
            id_col_map = {'miiv': 'stay_id', 'eicu': 'patientunitstayid', 'aumc': 'admissionid', 'hirid': 'patientid', 'mimic': 'icustay_id', 'sic': 'CaseID'}
            id_col = id_col_map.get(database, 'stay_id')

            # Step 1: 先选max_patients个患者作为候选集
            candidate_ids = None
            for f in _get_patient_id_table_files(database):
                fp = data_path / f
                if fp.exists():
                    icustays_df = pd.read_parquet(fp, columns=[id_col] if id_col else None)
                    if id_col in icustays_df.columns:
                        candidate_ids = _sample_patient_ids_random(icustays_df[id_col].unique().tolist(), max_patients)
                        break

            # Step 2: 在候选集上应用人群筛选
            try:
                data_path_for_cohort = st.session_state.data_path
                database_for_cohort = st.session_state.get('database', 'miiv')
                cohort_result = apply_cohort_filter(data_path_for_cohort, database_for_cohort, candidate_ids=candidate_ids)
                if cohort_result is not None:
                    cohort_id_col = cohort_result['id_col']
                    filtered_ids = cohort_result['filtered_ids']
                    id_col = cohort_id_col
                    patient_ids_filter = {id_col: filtered_ids}
                    # 🚫 Zero patients after cohort filter — abort preview
                    if len(filtered_ids) == 0:
                        lang = st.session_state.get('language', 'en')
                        n_before = len(candidate_ids) if candidate_ids else 0
                        if lang == 'en':
                            st.error(f"No patients meet the cohort criteria. {n_before} candidates were all excluded. Please adjust your cohort filters in Step 2.")
                        else:
                            st.error(f"没有患者满足队列筛选条件。{n_before} 名候选患者全部被排除。请在步骤2中调整筛选条件。")
                        return
                elif candidate_ids is not None:
                    patient_ids_filter = {id_col: candidate_ids}
            except Exception as _cohort_err:
                print(f"[COHORT] Error in load_data_for_preview: {_cohort_err}")
                if candidate_ids is not None:
                    patient_ids_filter = {id_col: candidate_ids}
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
            load_kwargs.update(_get_sepsis_runtime_options())
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

                    # 保留所有特征，包括空DataFrame（确保特征数量一致）
                    if isinstance(df, pd.DataFrame):
                        data[concept] = df
                    elif isinstance(df, pd.Series):
                        data[concept] = df.to_frame().reset_index()
            elif isinstance(result, pd.DataFrame):
                # 单概念加载返回 DataFrame（即使为空也保留）
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
                        **_get_sepsis_runtime_options(),
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

        # 🔧 POST-FILTER: Remove patients whose cohort-critical features are None
        database = st.session_state.get('database', 'miiv')
        data = _post_filter_cohort_data(data, database)

        # 获取患者列表并限制数量
        patient_ids = set()
        id_col_found = 'stay_id'
        id_candidates = ['stay_id', 'hadm_id', 'icustay_id',
                       'patientunitstayid', 'admissionid', 'patientid', 'CaseID']

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
                # 保留所有特征，包括空DataFrame（确保特征数量一致）
                filtered_data[concept_name] = filtered_df
            else:
                # 对于没有ID列的DataFrame（如静态指标），直接保留
                filtered_data[concept_name] = df

        st.session_state.loaded_concepts = filtered_data
        st.session_state.loaded_data_origin = 'quick_preview'
        st.session_state.patient_ids = preview_patient_ids
        st.session_state.all_patient_count = all_patient_count
        st.session_state.id_col = id_col_found

        load_elapsed = time.time() - load_start

        # 🔧 FIX (2026-02-04): 统计唯一概念数
        unique_concept_count = count_unique_concepts(list(filtered_data.keys()))

        lang = st.session_state.get('language', 'en')
        if lang == 'en':
            st.success(f"✅ Preview data loaded: {unique_concept_count} concepts ({len(filtered_data)} columns), {len(preview_patient_ids)}/{all_patient_count} patients ({load_elapsed:.1f}s)")
            if all_patient_count > max_patients:
                st.info(f"💡 For better performance, visualization is limited to {max_patients} patients. Export data first for full analysis with Python/R.")
        else:
            st.success(f"✅ 预览数据已加载：{unique_concept_count} 个概念（{len(filtered_data)} 列），{len(preview_patient_ids)}/{all_patient_count} 个患者 ({load_elapsed:.1f}秒)")
            if all_patient_count > max_patients:
                st.info(f"💡 为保证流畅性，可视化仅加载前 {max_patients} 个患者。建议先导出数据，再用Python/R工具进行完整分析。")

    except Exception as e:
        lang = st.session_state.get('language', 'en')
        err_msg = f"Loading failed: {e}" if lang == 'en' else f"加载失败: {e}"
        st.error(err_msg)
