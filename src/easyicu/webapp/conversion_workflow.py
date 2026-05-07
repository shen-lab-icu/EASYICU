"""CSV and database conversion workflows for the EasyICU webapp."""

from __future__ import annotations

from typing import Any


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to extracted workflows."""
    protected = {'render_convert_dialog', 'convert_csv_to_parquet', '_convert_hirid_data', "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def render_convert_dialog(app_context: dict[str, Any] | None = None):
    """Render CSV to Parquet conversion dialog."""
    if app_context is not None:
        _install_app_context(app_context)
    

    lang = st.session_state.get('language', 'en')
    source_path = st.session_state.get('convert_source_path', '')

    dialog_title = "## 🔄 CSV to Parquet Conversion" if lang == 'en' else "## 🔄 CSV 转换为 Parquet"
    st.markdown(dialog_title)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    source_info = f"📁 Source directory: `{source_path}`" if lang == 'en' else f"📁 源目录: `{source_path}`"
    st.info(source_info)

    col1, col2 = st.columns(2)

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
        # 转换选项
        overwrite_label = "Overwrite existing Parquet files" if lang == 'en' else "覆盖已存在的Parquet文件"
        overwrite = st.checkbox(overwrite_label, value=False)
        optimize_label = "Optimize buckets for repeated extraction" if lang == 'en' else "优化分桶以便反复提取"
        optimize_help = (
            "Keeps the default conversion fast when off. When on, bucketed large tables are sorted by source id, "
            "which can improve repeated feature extraction but may make conversion slower."
            if lang == 'en'
            else
            "关闭时保持默认快速转换；开启后会按来源ID整理大表分桶，可能降低转换速度，但更利于后续反复提取。"
        )
        extraction_optimized_buckets = st.checkbox(
            optimize_label,
            value=False,
            help=optimize_help,
        )

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

    col1, col2 = st.columns([1, 1])

    with col1:
        start_label = "🚀 Start Conversion" if lang == 'en' else "🚀 开始转换"
        if st.button(start_label, type="primary", use_container_width=True):
            if not target_path or not Path(target_path).exists():
                err_msg = "❌ Please set a valid output directory" if lang == 'en' else "❌ 请设置有效的输出目录"
                st.error(err_msg)
            else:
                spinner_msg = "Converting..." if lang == 'en' else "正在转换..."
                with st.spinner(spinner_msg):
                    success, failed = convert_csv_to_parquet(
                        source_path,
                        target_path,
                        overwrite,
                        extraction_optimized_buckets=extraction_optimized_buckets,
                    )

                # 转换完成后自动重新验证
                _database = st.session_state.get('database', 'miiv')
                _revalidation = validate_database_path(target_path, _database)

                if _revalidation['valid']:
                    # 全部就绪：转换成功 + 验证通过
                    st.session_state.path_validated = True
                    st.session_state.data_path = target_path
                    st.session_state.last_validation = _revalidation
                    st.session_state.last_validated_path = target_path
                    st.session_state.show_convert_dialog = False
                    _done_msg = (f"✅ Setup complete! Converted {success} items, all data validated."
                                 if lang == 'en' else
                                 f"✅ 设置完成！已转换 {success} 项，数据验证通过。")
                    st.success(_done_msg)
                    st.balloons()
                    import time as _t; _t.sleep(1.5)
                    st.rerun()
                elif success > 0:
                    # 部分完成：有转换但验证未完全通过
                    st.session_state.last_validation = _revalidation
                    st.session_state.last_validated_path = target_path
                    _partial_msg = (f"⚠️ Converted {success} items, but some data still needs attention."
                                    if lang == 'en' else
                                    f"⚠️ 已转换 {success} 项，但部分数据仍需处理。")
                    st.warning(_partial_msg)
                    st.error(_revalidation['message'])
                elif failed > 0:
                    fail_msg = (f"⚠️ {failed} files failed to convert. Please check error messages above."
                                if lang == 'en' else
                                f"⚠️ {failed} 个文件转换失败，请查看上方错误信息。")
                    st.warning(fail_msg)
                else:
                    no_files_msg = ("⚠️ No files were converted. Please check your data path."
                                    if lang == 'en' else
                                    "⚠️ 没有文件被转换，请检查数据路径。")
                    st.warning(no_files_msg)

    with col2:
        cancel_label = "❌ Cancel" if lang == 'en' else "❌ 取消"
        if st.button(cancel_label, use_container_width=True):
            st.session_state.show_convert_dialog = False
            st.rerun()


def convert_csv_to_parquet(
    source_dir: str,
    target_dir: str,
    overwrite: bool = False,
    app_context: dict[str, Any] | None = None,
    extraction_optimized_buckets: bool = False,
) -> tuple:
    """将目录下的CSV文件转换为Parquet格式。

    大表自动使用分桶转换，普通表使用 DuckDB 直接转换。
    HiRID 特殊处理：已经是 parquet 格式，只需分桶转换。
    """
    if app_context is not None:
        _install_app_context(app_context)
    
    import gc
    import time

    # 获取数据库类型
    database = st.session_state.get('database', 'miiv')

    # HiRID 特殊处理：数据已经是 parquet 格式，只需分桶
    if database == 'hirid':
        return _convert_hirid_data(
            source_dir,
            target_dir,
            overwrite,
            extraction_optimized_buckets=extraction_optimized_buckets,
        )

    # 定义需要分桶转换的大表
    BUCKET_TABLES = {
        'miiv': {
            'chartevents': ('itemid', 100),
            'labevents': ('itemid', 100),
            'inputevents': ('itemid', 50),
        },
        'eicu': {
            'nursecharting': ('nursingchartcelltypevalname', 30),
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
        import time
    except ImportError as e:
        st.error(f"Converter not available: {e}")
        return 0, 0

    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # 收集 CSV 文件并去重：同一表名的 .csv 和 .csv.gz 只保留一个（优先 .csv.gz 更小更快）
    _all_csvs = list(source_path.rglob('*.csv')) + list(source_path.rglob('*.csv.gz'))
    _csv_by_stem = {}  # stem → file, 优先 .gz
    for f in _all_csvs:
        stem = f.stem.lower().replace('.csv', '')  # .csv.gz → .csv → stem
        key = (str(f.parent), stem)
        if key not in _csv_by_stem or f.suffix.lower() == '.gz':
            _csv_by_stem[key] = f
    csv_files = list(_csv_by_stem.values())

    # 分类文件：大表用分桶，小表用普通转换
    bucket_tables_config = BUCKET_TABLES.get(database, {})
    bucket_files = []
    normal_files = []

    # 计算总大小用于预估时间
    total_size_mb = 0
    for csv_file in csv_files:
        stem = csv_file.stem.lower().replace('.csv', '')
        file_size = csv_file.stat().st_size / (1024 * 1024)
        total_size_mb += file_size
        if stem in bucket_tables_config:
            bucket_files.append((csv_file, bucket_tables_config[stem]))
        else:
            normal_files.append(csv_file)

    success = 0
    failed = 0
    total = len(normal_files) + len(bucket_files)
    current = 0
    processed_size_mb = 0

    progress_bar = st.progress(0)
    status_text = st.empty()
    eta_text = st.empty()
    details = st.container()

    # 转换速度跟踪
    start_time = time.time()

    def update_eta(processed_mb: float, elapsed_seconds: float):
        """更新预估剩余时间"""
        if elapsed_seconds > 0 and processed_mb > 0:
            speed_mb_per_sec = processed_mb / elapsed_seconds
            remaining_mb = total_size_mb - processed_mb
            if speed_mb_per_sec > 0:
                eta_seconds = remaining_mb / speed_mb_per_sec
                if eta_seconds < 60:
                    eta_str = f"{eta_seconds:.0f}s"
                elif eta_seconds < 3600:
                    eta_str = f"{eta_seconds/60:.1f}min"
                else:
                    eta_str = f"{eta_seconds/3600:.1f}h"
                eta_text.markdown(f"⏱️ **Speed**: {speed_mb_per_sec:.1f} MB/s | **ETA**: {eta_str} | **Total**: {total_size_mb:.0f} MB")

    # 创建 DuckDB 转换器（根据可用内存自动配置）
    try:
        from easyicu.memory_manager import get_available_memory_mb
        _avail_gb = get_available_memory_mb() / 1024
        # 使用可用内存的 60%，最低 2GB，最高 24GB
        _mem_limit_gb = max(2.0, min(_avail_gb * 0.6, 24.0))
    except Exception:
        _mem_limit_gb = 8.0  # 保守默认值
    converter = DuckDBConverter(
        data_path=str(source_path),
        memory_limit_gb=_mem_limit_gb,
        verbose=False
    )

    # 1. 转换普通表
    for csv_file in normal_files:
        current += 1
        file_size_mb = csv_file.stat().st_size / (1024 * 1024)
        try:
            rel_path = csv_file.relative_to(source_path)
            # 小写化文件名以确保下游一致性（MIMIC-III 使用大写 ADMISSIONS.csv.gz）
            parquet_name = rel_path.stem.replace('.csv', '').lower() + '.parquet'
            parquet_file = target_path / rel_path.parent / parquet_name

            if parquet_file.exists() and not overwrite:
                with details:
                    st.caption(f"⏭️ {csv_file.name} (exists)")
                processed_size_mb += file_size_mb
                progress_bar.progress(current / total)
                update_eta(processed_size_mb, time.time() - start_time)
                continue

            parquet_file.parent.mkdir(parents=True, exist_ok=True)

            status_text.markdown(f"**Converting**: `{csv_file.name}` ({file_size_mb:.1f}MB) [{current}/{total}]")

            result = converter.convert_file(csv_file)

            # DuckDBConverter 使用原始文件名（可能大写），重命名为小写
            if result['status'] == 'success' and result.get('output_path'):
                _out = Path(result['output_path'])
                if _out.exists() and _out.name != parquet_name:
                    _out.rename(_out.parent / parquet_name)

            processed_size_mb += file_size_mb

            if result['status'] == 'success':
                success += 1
                with details:
                    st.caption(f"✅ {csv_file.name}: {result['row_count']:,} rows")
            else:
                failed += 1
                with details:
                    st.caption(f"❌ {csv_file.name}: {result.get('error', 'unknown')[:40]}")

            gc.collect()
            update_eta(processed_size_mb, time.time() - start_time)

        except Exception as e:
            failed += 1
            processed_size_mb += file_size_mb
            with details:
                st.caption(f"❌ {csv_file.name}: {str(e)[:40]}")

        progress_bar.progress(current / total)

    # 2. 分桶转换大表
    for csv_file, (partition_col, num_buckets) in bucket_files:
        current += 1
        stem = csv_file.stem.lower().replace('.csv', '')
        bucket_dir = target_path / f"{stem}_bucket"
        file_size_mb = csv_file.stat().st_size / (1024 * 1024)

        try:
            # 检查分桶是否已完整完成（通过 _COMPLETE 标记文件）
            sentinel = bucket_dir / '_COMPLETE'
            if bucket_dir.exists() and sentinel.exists() and not overwrite:
                with details:
                    st.caption(f"⏭️ {csv_file.name} (bucket complete)")
                processed_size_mb += file_size_mb
                progress_bar.progress(current / total)
                update_eta(processed_size_mb, time.time() - start_time)
                continue
            # 如果目录存在但无 _COMPLETE 标记，说明上次转换不完整，清理后重新转换
            if bucket_dir.exists() and not sentinel.exists():
                import shutil
                with details:
                    st.caption(f"🔄 {csv_file.name} (incomplete, re-converting...)")
                shutil.rmtree(bucket_dir)

            status_text.markdown(f"**Bucketing**: `{csv_file.name}` ({file_size_mb:.1f}MB) → {num_buckets} buckets [{current}/{total}]")

            skip_sorting = not extraction_optimized_buckets
            if extraction_optimized_buckets:
                with details:
                    st.caption(
                        "🔎 Bucket extraction profile enabled: sorting rows by source id"
                        if lang == 'en'
                        else "🔎 已启用提取优化分桶：按来源ID整理分桶内数据"
                    )

            config = BucketConfig(
                num_buckets=num_buckets,
                partition_col=partition_col,
                memory_limit=f'{_mem_limit_gb:.0f}GB',
                threads=0,  # 自动检测CPU核心数
                row_group_size=1_000_000,
                compression='zstd',
                skip_sorting=skip_sorting,
            )
            result = convert_to_buckets(
                source_path=csv_file,
                output_dir=bucket_dir,
                config=config,
                overwrite=overwrite
            )

            processed_size_mb += file_size_mb

            if result.success:
                success += 1
                with details:
                    st.caption(f"✅ {csv_file.name} → {result.num_buckets} buckets, {result.total_rows:,} rows")
            else:
                failed += 1
                with details:
                    st.caption(f"❌ {csv_file.name}: {result.error[:40] if result.error else 'unknown'}")

            gc.collect()
            update_eta(processed_size_mb, time.time() - start_time)

        except Exception as e:
            failed += 1
            processed_size_mb += file_size_mb
            with details:
                st.caption(f"❌ {csv_file.name}: {str(e)[:40]}")

        progress_bar.progress(current / total)

    # 完成后显示总耗时
    total_time = time.time() - start_time
    if total_time < 60:
        time_str = f"{total_time:.1f}s"
    elif total_time < 3600:
        time_str = f"{total_time/60:.1f}min"
    else:
        time_str = f"{total_time/3600:.1f}h"

    progress_bar.progress(1.0)
    status_text.empty()
    eta_text.markdown(f"✅ **Completed** in {time_str} | **Avg Speed**: {total_size_mb/total_time:.1f} MB/s")

    return success, failed


def _convert_hirid_data(
    source_dir: str,
    target_dir: str,
    overwrite: bool = False,
    app_context: dict[str, Any] | None = None,
    extraction_optimized_buckets: bool = False,
) -> tuple:
    """HiRID 一站式转换：解压 → 重命名 parquet shards → CSV→parquet → 分桶。

    完整处理流程（用户只需点一次）：
    1. 解压 raw_stage/*.tar.gz + reference_data.tar.gz
    2. 重命名 observation_tables/parquet/part-N → observations/N.parquet
    3. general_table.csv → general.parquet（以及其他小 CSV 文件）
    4. observations/ → observations_bucket/ (variableid, 100桶)
    5. pharma/ → pharma_bucket/ (pharmaid, 50桶)
    """
    if app_context is not None:
        _install_app_context(app_context)
    
    import time

    lang = st.session_state.get('language', 'en')

    try:
        from easyicu.bucket_converter import (
            convert_parquet_directory_to_buckets
        )
        from easyicu.duckdb_converter import DuckDBConverter
    except ImportError as e:
        st.error(f"Converter not available: {e}")
        return 0, 0

    source_path = Path(source_dir)

    progress_bar = st.progress(0)
    status_text = st.empty()
    details = st.container()

    success = 0
    failed = 0
    start_time = time.time()

    # ============================================================
    # 阶段 1: 解压 tar.gz 文件
    # ============================================================
    status_text.markdown("**Phase 1/4**: Extracting archives..." if lang == 'en' else "**阶段 1/4**: 解压归档文件...")

    # 解压 reference_data.tar.gz → general_table.csv
    reference_tar = source_path / 'reference_data.tar.gz'
    general_csv = source_path / 'general_table.csv'
    if reference_tar.exists() and not general_csv.exists():
        try:
            import tarfile
            with st.spinner("Extracting reference_data.tar.gz..." if lang == 'en' else "正在解压 reference_data.tar.gz..."):
                with tarfile.open(reference_tar, 'r:gz') as tar:
                    tar.extractall(path=source_path)
            with details:
                st.caption("✅ reference_data.tar.gz → general_table.csv")
        except Exception as e:
            with details:
                st.caption(f"❌ reference_data.tar.gz: {e}")
            failed += 1

    # 解压 observation_tables + pharma_records
    raw_stage = source_path / 'raw_stage'
    archives = [
        ('observation_tables_parquet.tar.gz', 'observation_tables', 'observations'),
        ('pharma_records_parquet.tar.gz', 'pharma_records', 'pharma'),
    ]
    for archive_name, extract_dir, target_dir_name in archives:
        archive_path = raw_stage / archive_name if raw_stage.exists() else source_path / archive_name
        if not archive_path.exists():
            continue
        # 如果已有重命名后的 shard 目录（如 observations/），跳过
        shard_dir = source_path / target_dir_name
        if shard_dir.is_dir() and list(shard_dir.glob('[0-9]*.parquet')):
            with details:
                st.caption(f"⏭️ {archive_name} (already extracted → {target_dir_name}/)")
            continue
        # 如果解压目录已存在，跳过解压
        extracted_dir = source_path / extract_dir
        if not (extracted_dir.is_dir() and any(extracted_dir.rglob('*.parquet'))):
            try:
                import tarfile
                with st.spinner(f"Extracting {archive_name}..." if lang == 'en' else f"正在解压 {archive_name}..."):
                    with tarfile.open(archive_path, 'r:gz') as tar:
                        tar.extractall(path=source_path)
                with details:
                    st.caption(f"✅ {archive_name}")
            except Exception as e:
                with details:
                    st.caption(f"❌ {archive_name}: {e}")
                failed += 1
                continue

        # 重命名 part-N.parquet → N.parquet (R ricu 格式)
        parquet_subdir = extracted_dir / 'parquet'
        if parquet_subdir.is_dir():
            src_files = sorted(parquet_subdir.glob('part-*.parquet'))
        elif extracted_dir.is_dir():
            src_files = sorted(extracted_dir.glob('part-*.parquet'))
        else:
            src_files = []

        if src_files and not (shard_dir.is_dir() and list(shard_dir.glob('[0-9]*.parquet'))):
            import shutil
            shard_dir.mkdir(parents=True, exist_ok=True)
            for f in src_files:
                try:
                    idx = int(f.stem.replace('part-', '')) + 1  # 1-based
                    dst = shard_dir / f'{idx}.parquet'
                    if not dst.exists():
                        shutil.copy2(f, dst)
                except (ValueError, OSError):
                    pass
            with details:
                st.caption(f"✅ Renamed {len(src_files)} shards → {target_dir_name}/")

    progress_bar.progress(0.2)

    # ============================================================
    # 阶段 2: CSV → Parquet (小表)
    # ============================================================
    status_text.markdown("**Phase 2/4**: Converting CSV → Parquet..." if lang == 'en' else "**阶段 2/4**: CSV 转换为 Parquet...")

    csv_files = list(source_path.glob('*.csv'))
    converter = DuckDBConverter(data_path=str(source_path), memory_limit_gb=8.0, verbose=False)

    for csv_file in csv_files:
        parquet_name = csv_file.stem + '.parquet'
        parquet_file = source_path / parquet_name
        if parquet_file.exists() and not overwrite:
            continue
        try:
            result = converter.convert_file(csv_file)
            if result['status'] == 'success':
                success += 1
                with details:
                    st.caption(f"✅ {csv_file.name} → {parquet_name} ({result['row_count']:,} rows)")
            else:
                failed += 1
                with details:
                    st.caption(f"❌ {csv_file.name}: {result.get('error', 'unknown')[:60]}")
        except Exception as e:
            failed += 1
            with details:
                st.caption(f"❌ {csv_file.name}: {str(e)[:60]}")

    progress_bar.progress(0.4)

    # ============================================================
    # 阶段 3+4: Parquet shard 目录 → 分桶
    # ============================================================
    obs_dir = None
    pharma_dir = None

    obs_candidates = [
        source_path / 'observations',
        source_path / 'observations' / 'parquet',
        source_path / 'observation_tables',
        source_path / 'observation_tables' / 'parquet',
    ]
    for cand in obs_candidates:
        if cand.exists() and cand.is_dir():
            if list(cand.glob('*.parquet')):
                obs_dir = cand
                break

    # 可能的 pharma 目录位置
    pharma_candidates = [
        source_path / 'pharma',
        source_path / 'pharma' / 'parquet',
        source_path / 'pharma_records',
        source_path / 'pharma_records' / 'parquet',
    ]
    for cand in pharma_candidates:
        if cand.exists() and cand.is_dir():
            if list(cand.glob('*.parquet')):
                pharma_dir = cand
                break

    # 检查是否有数据可分桶
    if not obs_dir and not pharma_dir:
        st.error("❌ No observation/pharma parquet directories found after extraction. "
                 "Please check your HiRID data." if lang == 'en' else
                 "❌ 解压后未找到 observation/pharma parquet 目录，请检查 HiRID 数据。")
        return success, max(1, failed)

    obs_bucket_dir = source_path / 'observations_bucket'
    pharma_bucket_dir = source_path / 'pharma_bucket'

    tasks = []
    if obs_dir:
        tasks.append(('observations', obs_dir, obs_bucket_dir, 'variableid', 100))
    if pharma_dir:
        tasks.append(('pharma', pharma_dir, pharma_bucket_dir, 'pharmaid', 50))

    total = len(tasks)

    for idx, (name, src_dir, bucket_dir, partition_col, num_buckets) in enumerate(tasks):
        _pct = 0.6 + 0.4 * idx / max(len(tasks), 1)
        progress_bar.progress(_pct)
        status_text.markdown(f"**Phase 3/4**: Bucketing `{name}` → {num_buckets} buckets..." if lang == 'en' else f"**阶段 3/4**: 分桶 `{name}` → {num_buckets} 个桶...")

        try:
            sentinel = bucket_dir / '_COMPLETE'
            source_newer = False
            if sentinel.exists():
                try:
                    sentinel_mtime = sentinel.stat().st_mtime
                    source_newer = any(p.stat().st_mtime > sentinel_mtime for p in src_dir.glob('*.parquet'))
                except OSError:
                    source_newer = True
            if bucket_dir.exists() and sentinel.exists() and not source_newer and not overwrite:
                with details:
                    st.caption(f"⏭️ {name} (bucket complete, skipped)" if lang == 'en' else f"⏭️ {name} (分桶已完成，跳过)")
                success += 1
                continue
            if bucket_dir.exists() and not overwrite:
                import shutil
                with details:
                    reason = "source updated" if source_newer else "incomplete"
                    st.caption(f"🔄 {name} ({reason}, re-converting...)" if lang == 'en' else f"🔄 {name} (分桶未完成或源已更新，重新转换...)")
                shutil.rmtree(bucket_dir)

            skip_sorting = not extraction_optimized_buckets
            if extraction_optimized_buckets:
                with details:
                    st.caption(
                        f"🔎 {name}: extraction-optimized bucket sorting enabled"
                        if lang == 'en'
                        else f"🔎 {name}: 已启用提取优化分桶排序"
                    )

            result = convert_parquet_directory_to_buckets(
                source_dir=src_dir,
                output_dir=bucket_dir,
                partition_col=partition_col,
                num_buckets=num_buckets,
                overwrite=overwrite,
                skip_sorting=skip_sorting,
            )

            if result.success:
                success += 1
                with details:
                    st.caption(f"✅ {name} → {result.num_buckets} buckets, {result.total_rows:,} rows")
            else:
                failed += 1
                with details:
                    st.caption(f"❌ {name}: {result.error[:60] if result.error else 'unknown'}")
        except Exception as e:
            failed += 1
            with details:
                st.caption(f"❌ {name}: {str(e)[:60]}")

    total_time = time.time() - start_time
    progress_bar.progress(1.0)
    status_text.empty()

    if success > 0:
        st.success(f"✅ HiRID setup completed in {total_time:.1f}s ({success} steps)" if lang == 'en' else f"✅ HiRID 设置完成，耗时 {total_time:.1f}秒 ({success} 个步骤)")

    return success, failed
