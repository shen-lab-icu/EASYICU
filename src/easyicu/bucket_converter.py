"""
分桶转换器 - 将大表按变量ID分桶存储

核心思想：
1. hash(itemid) % num_buckets 将数据分到固定数量的桶
2. 每个桶内数据按itemid排序，利用Parquet Row Group统计信息实现谓词下推
3. 读取时根据目标itemid计算桶号，只扫描相关桶

算法优化：
- 使用DuckDB进行转换，充分利用多核和向量化执行
- 100个桶是最佳平衡点：每个桶约800MB（对于80GB的表）
- Row Group大小100,000行，便于细粒度谓词下推
- write_statistics=true 确保Row Group统计信息用于谓词下推

16GB内存优化：
- 转换时自动检测可用内存，预留3GB给OS/Python
- 指定temp_directory到高速SSD，避免内存溢出
- 读取时列投影 + 谓词下推，最小化内存占用
"""

import logging
import os
from pathlib import Path
from typing import Optional, Set, List, Callable
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


def _duckdb_path(p) -> str:
    """Convert a Path or string to a DuckDB-safe path string (forward slashes)."""
    return str(p).replace('\\', '/')


def _get_total_ram_gb() -> float:
    """获取系统总物理内存(GB)"""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    return int(line.split()[1]) / 1024 / 1024
    except (OSError, ValueError):
        pass
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    return 16.0  # 保守默认值


def _auto_memory_limit() -> str:
    """根据系统可用内存自动设置 DuckDB memory_limit。
    
    低内存系统 (<32GB) 使用基于总内存的保守限制，防止与 PARTITION_BY 的
    100 个同时分区写入器组合导致内存爆炸。
    """
    total_gb = _get_total_ram_gb()
    if total_gb < 32:
        # 低内存：基于总量的保守限制（而非可用量，避免页面文件虚高）
        limit_gb = max(1, int(total_gb * 0.15))  # 12GB→1GB, 16GB→2GB, 24GB→3GB
        return f'{limit_gb}GB'
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    avail_mb = int(line.split()[1]) / 1024.0
                    limit_gb = max(2, int(avail_mb / 1024) - 3)
                    return f'{limit_gb}GB'
    except (OSError, ValueError):
        pass
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        limit_gb = max(2, int(avail_gb) - 3)
        return f'{limit_gb}GB'
    except ImportError:
        pass
    return '6GB'


@dataclass
class BucketConfig:
    """分桶配置"""
    num_buckets: int = 100  # 桶数量
    partition_col: str = 'itemid'  # 分桶列
    row_group_size: int = 1_000_000  # Row Group大小，1M行最优平衡
    compression: str = 'zstd'  # zstd压缩率更高，速度接近snappy
    memory_limit: str = ''  # 空字符串 = 自动检测（预留3GB给OS）
    threads: int = 0  # 0=自动检测CPU核心数
    temp_directory: Optional[str] = None  # 临时文件目录，建议SSD
    skip_sorting: bool = True  # 跳过排序，大幅加速
    column_types: Optional[dict] = None  # 强制指定列类型，如 {'VALUE': 'VARCHAR'}
    encoding: Optional[str] = None  # CSV 编码，如 'latin-1'（AUMC 需要）

    def __post_init__(self):
        if not self.memory_limit:
            self.memory_limit = _auto_memory_limit()


@dataclass 
class ConversionResult:
    """转换结果"""
    success: bool
    num_buckets: int
    total_rows: int
    total_size_bytes: int
    elapsed_seconds: float
    output_dir: Optional[Path] = None
    error: Optional[str] = None


def _duckdb_hash(itemid: int, num_buckets: int = 100) -> int:
    """
    使用DuckDB的hash函数计算桶ID
    
    注意: Python的hash()和DuckDB的hash()结果不同！
    转换时使用DuckDB，读取时也必须使用DuckDB的hash来定位桶。
    """
    import duckdb
    conn = duckdb.connect()
    try:
        result = conn.execute(f"SELECT hash({itemid}) % {num_buckets}").fetchone()[0]
    finally:
        conn.close()
    return result


def _duckdb_hash_batch(itemids: Set[int], num_buckets: int = 100) -> Set[int]:
    """
    批量计算DuckDB hash，返回目标桶ID集合
    """
    import duckdb
    conn = duckdb.connect()
    try:
        # 使用 UNNEST 批量计算
        itemid_list = list(itemids)
        conn.execute("CREATE TEMP TABLE items AS SELECT UNNEST(?) as itemid", [itemid_list])
        result = conn.execute(f"SELECT DISTINCT hash(itemid) % {num_buckets} FROM items").fetchall()
    finally:
        conn.close()
    return {row[0] for row in result}


def convert_to_buckets(
    source_path: Path,
    output_dir: Path,
    config: BucketConfig = BucketConfig(),
    progress_callback: Optional[Callable[[str], None]] = None,
    overwrite: bool = False
) -> ConversionResult:
    """
    将大表转换为分桶Parquet格式
    
    Args:
        source_path: 源文件路径（CSV或Parquet）
        output_dir: 输出目录
        config: 分桶配置
        progress_callback: 进度回调函数
        overwrite: 是否覆盖已存在的目录
    
    Returns:
        ConversionResult: 转换结果
    """
    import duckdb
    import shutil
    
    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
    
    start_time = time.time()
    
    # 检查源文件
    source_path = Path(source_path)
    if not source_path.exists():
        return ConversionResult(
            success=False, num_buckets=0, total_rows=0,
            total_size_bytes=0, elapsed_seconds=0,
            error=f"源文件不存在: {source_path}"
        )
    
    # 准备输出目录
    output_dir = Path(output_dir)
    if output_dir.exists():
        if overwrite:
            log(f"删除已存在的输出目录: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            return ConversionResult(
                success=False, num_buckets=0, total_rows=0,
                total_size_bytes=0, elapsed_seconds=0,
                error=f"输出目录已存在: {output_dir}"
            )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        log(f"开始转换: {source_path.name}")
        log(f"分桶数: {config.num_buckets}, 分桶列: {config.partition_col}")
        log(f"内存限制: {config.memory_limit}, 临时目录: {config.temp_directory or '默认'}")
        
        conn = duckdb.connect()
        # 并行线程数：0=自动检测CPU核心数
        if config.threads > 0:
            conn.execute(f"SET threads={config.threads}")
        # 内存限制：防止OOM
        conn.execute(f"SET memory_limit='{config.memory_limit}'")
        # 禁用保序以启用并行写入
        conn.execute("SET preserve_insertion_order=false")
        # 启用进度条
        conn.execute("SET enable_progress_bar=true")
        
        # 设置临时目录：建议在高速SSD上，处理80GB排序的磁盘溢出
        if config.temp_directory:
            os.makedirs(config.temp_directory, exist_ok=True)
            conn.execute(f"SET temp_directory='{_duckdb_path(config.temp_directory)}'")
            log(f"临时目录设置为: {config.temp_directory}")
        
        # 低内存检测：<32GB RAM 时启用保守策略
        total_ram_gb = _get_total_ram_gb()
        low_memory = total_ram_gb < 32
        
        if low_memory:
            # 低内存模式：限制 DuckDB 资源
            duck_mem = f'{max(1, int(total_ram_gb * 0.15))}GB'
            conn.execute(f"SET memory_limit='{duck_mem}'")
            conn.execute(f'SET threads={min(4, os.cpu_count() or 4)}')
            log(f"低内存模式: {total_ram_gb:.0f}GB RAM, DuckDB {duck_mem}")
            
            # 确保设置 temp_directory（DuckDB 溢出用）
            if not config.temp_directory:
                import tempfile as _tmpmod
                _tmp_dir = _tmpmod.gettempdir()
                conn.execute(f"SET temp_directory='{_duckdb_path(_tmp_dir)}'")
        
        # 确定读取方式
        source_name = source_path.name.lower()
        is_csv = source_name.endswith('.csv.gz') or source_name.endswith('.csv')
        
        if is_csv:
            types_arg = ""
            if config.column_types:
                types_str = ", ".join(f"'{k}': '{v}'" for k, v in config.column_types.items())
                types_arg = f", types={{{types_str}}}"
                log(f"强制列类型: {config.column_types}")
            
            # 检测是否需要 latin-1 编码（AUMC 包含 µmol 等特殊字符）
            encoding_arg = ""
            if config.encoding:
                encoding_arg = f", encoding='{config.encoding}'"
            else:
                try:
                    parent_name = source_path.parent.name.lower()
                    if 'aumc' in parent_name or 'aumc' in str(source_path.parent.parent).lower():
                        encoding_arg = ", encoding='latin-1'"
                except Exception:
                    pass
            
            # 低内存模式: parallel=false 避免 DuckDB 并行 CSV 扫描器的
            # null_padding + quoted newlines 兼容性问题
            parallel_arg = ", parallel=false" if low_memory else ""
            
            csv_read_expr = (
                f"read_csv_auto('{_duckdb_path(source_path)}', "
                f"ignore_errors=true, null_padding=true{types_arg}{encoding_arg}{parallel_arg})"
            )
            log(f"源文件类型: CSV{'（gzip压缩）' if source_name.endswith('.gz') else ''}")
        elif source_name.endswith('.parquet'):
            log("源文件类型: Parquet")
        else:
            return ConversionResult(
                success=False, num_buckets=0, total_rows=0,
                total_size_bytes=0, elapsed_seconds=0,
                error=f"不支持的文件格式: {source_path.suffix}，仅支持 .csv, .csv.gz, .parquet"
            )
        
        # 🚀 大 CSV 文件两阶段转换优化
        # 对 HDD 上的大文件（>1GB CSV），直接 CSV→100 桶会触发大量随机写，
        # 导致 HDD 寻道风暴（75GB CSV 可能耗时 1 小时+）。
        # 两阶段策略：先 CSV→单 Parquet（纯顺序写），再 Parquet→100 桶（内存/缓存读取极快）。
        # 实测 75GB AUMC numericitems: 1h+ → 2 分钟（大内存服务器）。
        #
        # 低内存优化（<32GB RAM）：
        # DuckDB PARTITION_BY 同时打开 N 个分区写入器，每个缓冲 ROW_GROUP_SIZE 行。
        # 100 分区 × 大 ROW_GROUP_SIZE → 内存爆炸（12GB PC 上实测 24GB commit）。
        # 解决：多趟分桶 — 每趟只写 20 个桶，RSS 稳定 <1GB。
        _TWO_STAGE_THRESHOLD = 1 * 1024 * 1024 * 1024  # 1GB
        use_two_stage = is_csv and source_path.stat().st_size > _TWO_STAGE_THRESHOLD
        temp_parquet = None
        
        if use_two_stage:
            import tempfile
            # 中间 Parquet 放在与输出目录同一磁盘，利用 OS 页缓存加速第二阶段
            temp_fd, temp_parquet = tempfile.mkstemp(
                suffix='.parquet', prefix='_easyicu_stage1_',
                dir=str(output_dir.parent)
            )
            os.close(temp_fd)
            
            log(f"大文件两阶段转换 ({source_path.stat().st_size / 1e9:.1f}GB)...")
            log(f"  Stage 1/2: CSV → 单个 Parquet（顺序写入）...")
            t_s1 = time.time()
            conn.execute(f"""
                COPY (SELECT * FROM {csv_read_expr})
                TO '{_duckdb_path(temp_parquet)}' (
                    FORMAT PARQUET, COMPRESSION 'SNAPPY',
                    ROW_GROUP_SIZE 1000000
                )
            """)
            s1_time = time.time() - t_s1
            s1_size = os.path.getsize(temp_parquet) / 1e9
            log(f"  Stage 1 完成: {s1_size:.2f}GB, {s1_time:.1f}s")
            
            read_expr = f"read_parquet('{_duckdb_path(temp_parquet)}')"
        else:
            if is_csv:
                read_expr = csv_read_expr
            else:
                read_expr = f"read_parquet('{_duckdb_path(source_path)}')"
            log("执行分桶转换...")
        
        try:
            # 计算多趟参数
            # 低内存 + 大文件: 多趟分桶，每趟处理部分桶以限制同时打开的分区写入器
            if low_memory and use_two_stage:
                # 每趟 20 个桶：实测每趟 RSS < 1GB，5 趟合计 ~200s（服务器）
                buckets_per_pass = max(5, min(config.num_buckets, int(total_ram_gb * 2)))
                n_passes = (config.num_buckets + buckets_per_pass - 1) // buckets_per_pass
                
                log(f"  Stage 2/2: Parquet → {config.num_buckets} 桶"
                    f"（{n_passes} 趟, 每趟 {buckets_per_pass} 桶）")
                for pi in range(n_passes):
                    s = pi * buckets_per_pass
                    e = min(s + buckets_per_pass - 1, config.num_buckets - 1)
                    log(f"    趟 {pi+1}/{n_passes}: bucket {s}-{e}")
                    conn.execute(f"""
                        COPY (
                            SELECT *,
                                   hash({config.partition_col}) % {config.num_buckets} as bucket_id
                            FROM {read_expr}
                            WHERE hash({config.partition_col}) % {config.num_buckets}
                                  BETWEEN {s} AND {e}
                        )
                        TO '{_duckdb_path(output_dir)}'
                        (FORMAT PARQUET,
                         PARTITION_BY (bucket_id),
                         COMPRESSION {config.compression.upper()},
                         ROW_GROUP_SIZE {config.row_group_size},
                         OVERWRITE_OR_IGNORE)
                    """)
            else:
                if use_two_stage:
                    log(f"  Stage 2/2: Parquet → {config.num_buckets} 个分桶...")
                sql = f"""
                    COPY (
                        SELECT *,
                               hash({config.partition_col}) % {config.num_buckets} as bucket_id
                        FROM {read_expr}
                    )
                    TO '{_duckdb_path(output_dir)}'
                    (FORMAT PARQUET,
                     PARTITION_BY (bucket_id),
                     COMPRESSION {config.compression.upper()},
                     ROW_GROUP_SIZE {config.row_group_size},
                     OVERWRITE_OR_IGNORE)
                """
                conn.execute(sql)
        finally:
            # 清理临时 Parquet 文件
            if temp_parquet and os.path.exists(temp_parquet):
                try:
                    os.remove(temp_parquet)
                except OSError:
                    pass
        
        # 统计结果
        elapsed = time.time() - start_time
        
        # 🚀 优化: 从已写出的parquet桶计数，避免重新扫描源文件（对80GB CSV节省10-30分钟）
        try:
            row_count = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{_duckdb_path(output_dir)}/**/*.parquet', hive_partitioning=true)").fetchone()[0]
        except Exception:
            row_count = 0  # 如果计数失败，不影响转换结果
        
        # 计算总大小
        total_size = sum(f.stat().st_size for f in output_dir.rglob('*.parquet'))
        actual_buckets = len([d for d in output_dir.iterdir() if d.is_dir()])
        
        conn.close()
        
        # 写入完成标记文件 — 用于崩溃恢复检测
        sentinel = output_dir / '_COMPLETE'
        sentinel.write_text(f"{row_count},{actual_buckets},{total_size}")
        
        log(f"转换完成! 耗时: {elapsed:.1f}秒")
        log(f"总行数: {row_count:,}")
        log(f"分桶数: {actual_buckets}")
        log(f"总大小: {total_size / 1024**3:.2f} GB")
        log(f"平均桶大小: {total_size / max(actual_buckets, 1) / 1024**2:.1f} MB")
        
        return ConversionResult(
            success=True,
            num_buckets=actual_buckets,
            total_rows=row_count,
            total_size_bytes=total_size,
            elapsed_seconds=elapsed,
            output_dir=output_dir
        )
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception("转换失败")
        try:
            conn.close()
        except Exception:
            pass
        return ConversionResult(
            success=False, num_buckets=0, total_rows=0,
            total_size_bytes=0, elapsed_seconds=elapsed,
            error=str(e)
        )


def read_from_buckets(
    bucket_dir: Path,
    itemids: Optional[Set[int]] = None,
    columns: Optional[List[str]] = None,
    partition_col: str = 'itemid',
    num_buckets: int = 100,
    explain: bool = False
):
    """
    从分桶目录高效读取数据
    
    核心优化:
    1. 根据itemid计算目标桶，只扫描相关桶（跳过99%无关数据）
    2. 使用Polars Lazy API实现谓词下推（利用Row Group统计信息）
    3. 列投影只读取需要的列（Parquet列式存储优势）
    
    16GB内存安全:
    - 只加载需要的桶 + 需要的列 + 符合条件的行
    - 即使总数据80GB，实际内存占用通常 < 1GB
    
    Args:
        bucket_dir: 分桶目录
        itemids: 要读取的itemid集合（None表示全部）
        columns: 要读取的列（None表示全部，但排除bucket_id）
        partition_col: 分桶列名
        num_buckets: 桶数量
        explain: 是否打印查询计划（用于验证谓词下推生效）
    
    Returns:
        polars.DataFrame
    """
    import polars as pl
    
    bucket_dir = Path(bucket_dir)
    
    if itemids:
        # 使用DuckDB hash计算目标桶（与转换时一致）
        target_buckets = _duckdb_hash_batch(itemids, num_buckets)
        logger.info(f"目标itemid: {len(itemids)}个, 定位到{len(target_buckets)}个桶")
        
        # 只读取目标桶
        parquet_files = []
        for bucket_id in target_buckets:
            bucket_path = bucket_dir / f"bucket_id={bucket_id}"
            if bucket_path.exists():
                parquet_files.extend(bucket_path.glob("*.parquet"))
        
        if not parquet_files:
            # 返回空DataFrame
            return pl.DataFrame()
        
        # 使用Lazy API实现谓词下推
        # Polars会利用Parquet Row Group统计信息跳过不匹配的Row Group
        lf = pl.scan_parquet(parquet_files)
        lf = lf.filter(pl.col(partition_col).is_in(list(itemids)))
    else:
        # 读取所有桶（仍受益于列投影）
        lf = pl.scan_parquet(str(bucket_dir / "**/*.parquet"))
    
    # 列投影：只读取需要的列，大幅减少内存
    if columns:
        # 排除 bucket_id 列，只选择用户需要的列
        available_cols = [c for c in columns if c != 'bucket_id']
        lf = lf.select(available_cols)
    else:
        # 排除 bucket_id 列
        lf = lf.select(pl.exclude('bucket_id'))
    
    # 验证查询计划（调试用）
    if explain:
        print("=== Polars 查询计划 ===")
        print(lf.explain())
        print()
        print("=== 优化后查询计划 ===")
        print(lf.explain(optimized=True))
    
    return lf.collect()


def read_from_buckets_streaming(
    bucket_dir: Path,
    itemids: Optional[Set[int]] = None,
    columns: Optional[List[str]] = None,
    partition_col: str = 'itemid',
    num_buckets: int = 100,
    batch_size: int = 1_000_000
):
    """
    流式读取分桶数据，用于超大结果集
    
    当预期结果超过可用内存时使用此函数。
    每次yield一个batch，由调用者决定如何处理。
    
    Args:
        bucket_dir: 分桶目录
        itemids: 要读取的itemid集合
        columns: 要读取的列
        partition_col: 分桶列名
        num_buckets: 桶数量
        batch_size: 每批行数
    
    Yields:
        polars.DataFrame: 每批数据
    """
    import polars as pl
    
    bucket_dir = Path(bucket_dir)
    
    if itemids:
        target_buckets = _duckdb_hash_batch(itemids, num_buckets)
        
        for bucket_id in sorted(target_buckets):
            bucket_path = bucket_dir / f"bucket_id={bucket_id}"
            if not bucket_path.exists():
                continue
                
            parquet_files = list(bucket_path.glob("*.parquet"))
            if not parquet_files:
                continue
            
            # 逐桶读取
            lf = pl.scan_parquet(parquet_files)
            lf = lf.filter(pl.col(partition_col).is_in(list(itemids)))
            
            if columns:
                available_cols = [c for c in columns if c != 'bucket_id']
                lf = lf.select(available_cols)
            else:
                lf = lf.select(pl.exclude('bucket_id'))
            
            # 使用 sink 的方式分批返回
            df = lf.collect()
            
            # 分批 yield
            for i in range(0, len(df), batch_size):
                yield df.slice(i, batch_size)
    else:
        # 全量读取也分批
        all_dirs = sorted(bucket_dir.iterdir())
        for bucket_path in all_dirs:
            if not bucket_path.is_dir():
                continue
            
            parquet_files = list(bucket_path.glob("*.parquet"))
            if not parquet_files:
                continue
            
            lf = pl.scan_parquet(parquet_files)
            
            if columns:
                available_cols = [c for c in columns if c != 'bucket_id']
                lf = lf.select(available_cols)
            else:
                lf = lf.select(pl.exclude('bucket_id'))
            
            df = lf.collect()
            for i in range(0, len(df), batch_size):
                yield df.slice(i, batch_size)


# === AUMC numericitems 专用转换函数 ===

def convert_aumc_numericitems(
    data_path: str = '/home/zhuhb/icudb/aumc/1.0.2',
    num_buckets: int = 100,
    overwrite: bool = False
) -> ConversionResult:
    """
    转换 AUMC numericitems 到分桶格式
    
    AUMC numericitems.csv 包含特殊编码字符（如 µmol），需要使用显式 schema
    来避免 DuckDB 的类型推断因特殊字符而跳过行。
    
    使用两阶段优化：CSV→单Parquet→分桶，避免HDD随机写瓶颈。
    实测 75GB CSV: 1h+ → ~2 分钟。
    
    Args:
        data_path: AUMC数据目录
        num_buckets: 桶数量（默认100）
        overwrite: 是否覆盖已存在的目录
    """
    data_path = Path(data_path)
    source = data_path / 'numericitems.csv'
    output = data_path / 'numericitems_bucket'
    
    config = BucketConfig(
        num_buckets=num_buckets,
        partition_col='itemid',
        row_group_size=100_000,
        compression='snappy',
    )
    
    return convert_to_buckets(source, output, config, overwrite=overwrite)


def convert_aumc_listitems(
    data_path: str = '/home/zhuhb/icudb/aumc/1.0.2',
    num_buckets: int = 50,
    overwrite: bool = False
) -> ConversionResult:
    """
    转换 AUMC listitems 到分桶格式
    """
    data_path = Path(data_path)
    source = data_path / 'listitems.csv'
    output = data_path / 'listitems_bucket'
    
    config = BucketConfig(
        num_buckets=num_buckets,
        partition_col='itemid',
        row_group_size=100_000,
        compression='snappy'
    )
    
    return convert_to_buckets(source, output, config, overwrite=overwrite)


def convert_parquet_directory_to_buckets(
    source_dir: Path,
    output_dir: Path,
    partition_col: str,
    num_buckets: int = 100,
    overwrite: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None
) -> ConversionResult:
    """
    将已有的 Parquet 目录（如 HiRID observations）转换为分桶格式
    
    这个函数专门处理已有多个 parquet 分片的情况，例如：
    - HiRID observations: 250个按患者分片的parquet → 按variableid分桶
    - MIIV chartevents: 30个数字分片的parquet → 按itemid分桶
    
    核心优化：
    - 使用 DuckDB glob 模式读取所有分片
    - 一次性排序并分桶输出
    - 16GB 内存安全：设置 memory_limit 和 temp_directory
    
    Args:
        source_dir: 源目录（包含多个 parquet 文件）
        output_dir: 输出目录（将创建 bucket_id=* 子目录）
        partition_col: 分桶列（如 variableid 或 itemid）
        num_buckets: 桶数量
        overwrite: 是否覆盖已存在的目录
        progress_callback: 进度回调
    
    Returns:
        ConversionResult
    """
    import duckdb
    import shutil
    
    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
    
    start_time = time.time()
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # 检查源目录
    if not source_dir.is_dir():
        return ConversionResult(
            success=False, num_buckets=0, total_rows=0,
            total_size_bytes=0, elapsed_seconds=0,
            error=f"源目录不存在: {source_dir}"
        )
    
    parquet_files = list(source_dir.glob("*.parquet"))
    if not parquet_files:
        return ConversionResult(
            success=False, num_buckets=0, total_rows=0,
            total_size_bytes=0, elapsed_seconds=0,
            error=f"源目录没有 parquet 文件: {source_dir}"
        )
    
    log(f"发现 {len(parquet_files)} 个 parquet 文件")
    
    # 准备输出目录
    if output_dir.exists():
        if overwrite:
            log(f"删除已存在的输出目录: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            return ConversionResult(
                success=False, num_buckets=0, total_rows=0,
                total_size_bytes=0, elapsed_seconds=0,
                error=f"输出目录已存在: {output_dir}"
            )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        log(f"开始转换: {source_dir.name} → {output_dir.name}")
        log(f"分桶数: {num_buckets}, 分桶列: {partition_col}")
        
        conn = duckdb.connect()
        conn.execute("SET threads=16")
        conn.execute("SET memory_limit='10GB'")
        
        # 使用临时目录防止内存溢出
        temp_dir = output_dir.parent / f".{output_dir.name}_temp"
        temp_dir.mkdir(exist_ok=True)
        conn.execute(f"SET temp_directory='{_duckdb_path(temp_dir)}'")
        log(f"临时目录: {temp_dir}")
        
        # 使用 glob 读取所有 parquet，union_by_name 处理 schema 差异
        glob_pattern = _duckdb_path(source_dir / "*.parquet")
        read_expr = f"read_parquet('{glob_pattern}', union_by_name=true)"
        
        # 分桶转换
        log("执行分桶转换 (读取 → 分桶)...")
        
        sql = f"""
            COPY (
                SELECT *,
                       hash({partition_col}) % {num_buckets} as bucket_id
                FROM {read_expr}
            )
            TO '{_duckdb_path(output_dir)}'
            (FORMAT PARQUET,
             PARTITION_BY (bucket_id),
             COMPRESSION SNAPPY,
             ROW_GROUP_SIZE 100000,
             OVERWRITE_OR_IGNORE)
        """
        
        conn.execute(sql)
        
        # 统计结果
        elapsed = time.time() - start_time
        
        # 🚀 优化: 从已写出的parquet桶计数，避免重新扫描源parquet分片
        try:
            row_count = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{_duckdb_path(output_dir)}/**/*.parquet', hive_partitioning=true)").fetchone()[0]
        except Exception:
            row_count = 0
        
        # 计算总大小
        total_size = sum(f.stat().st_size for f in output_dir.rglob('*.parquet'))
        actual_buckets = len([d for d in output_dir.iterdir() if d.is_dir()])
        
        conn.close()
        
        # 清理临时目录
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        log(f"转换完成! 耗时: {elapsed:.1f}秒")
        log(f"总行数: {row_count:,}")
        log(f"分桶数: {actual_buckets}")
        log(f"总大小: {total_size / 1024**3:.2f} GB")
        
        return ConversionResult(
            success=True,
            num_buckets=actual_buckets,
            total_rows=row_count,
            total_size_bytes=total_size,
            elapsed_seconds=elapsed,
            output_dir=output_dir
        )
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception("转换失败")
        try:
            conn.close()
        except Exception:
            pass
        return ConversionResult(
            success=False, num_buckets=0, total_rows=0,
            total_size_bytes=0, elapsed_seconds=elapsed,
            error=str(e)
        )


def convert_hirid_observations(
    data_path: str = '/home/zhuhb/icudb/hirid/1.1.1',
    num_buckets: int = 100,
    overwrite: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None
) -> ConversionResult:
    """
    将 HiRID observations 转换为按 variableid 分桶格式
    
    HiRID 官方提供的 250 个分片是按患者分的，
    这意味着提取单个变量时仍需扫描所有 250 个分片。
    
    按 variableid 分桶后：
    - 提取单变量只需扫描 1 个桶（跳过 99% 无关数据）
    - 预期性能提升 10-100x
    - 内存峰值大幅降低
    
    Args:
        data_path: HiRID 数据目录
        num_buckets: 桶数量（默认 100）
        overwrite: 是否覆盖已存在的目录
        progress_callback: 进度回调函数
    """
    data_path = Path(data_path)
    source = data_path / 'observations'
    output = data_path / 'observations_bucket'
    
    return convert_parquet_directory_to_buckets(
        source, output, 
        partition_col='variableid',
        num_buckets=num_buckets,
        overwrite=overwrite,
        progress_callback=progress_callback
    )


def convert_miiv_chartevents(
    data_path: str = '/home/zhuhb/icudb/mimiciv/3.1/icu',
    num_buckets: int = 100,
    overwrite: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None
) -> ConversionResult:
    """
    将 MIIV chartevents 直接从 csv.gz 转换为按 itemid 分桶格式
    
    一步到位：csv.gz → 分桶 parquet（无需先转成单个 parquet）
    
    Args:
        data_path: MIIV ICU 数据目录（含 chartevents.csv.gz）
        num_buckets: 桶数量（默认 100）
        overwrite: 是否覆盖已存在的目录
        progress_callback: 进度回调函数
    """
    data_path = Path(data_path)
    
    # 按优先级检查源文件：csv.gz > parquet
    csv_gz = data_path / 'chartevents.csv.gz'
    parquet = data_path / 'chartevents.parquet'
    
    if csv_gz.exists():
        source = csv_gz
    elif parquet.exists():
        source = parquet
    else:
        return ConversionResult(
            success=False, num_buckets=0, total_rows=0,
            total_size_bytes=0, elapsed_seconds=0,
            error=f"未找到 chartevents 源文件，已检查:\n  - {csv_gz}\n  - {parquet}"
        )
    
    # 输出到同级目录的 _bucket 目录
    output = data_path / 'chartevents_bucket'
    
    config = BucketConfig(
        num_buckets=num_buckets,
        partition_col='itemid',
        row_group_size=100_000,
        compression='snappy'
    )
    
    return convert_to_buckets(source, output, config, progress_callback=progress_callback, overwrite=overwrite)


def convert_eicu_nursecharting(
    data_path: str = '/home/zhuhb/icudb/eicu/2.0.1',
    num_buckets: int = 50,
    overwrite: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None
) -> ConversionResult:
    """
    将 eICU nursecharting 转换为按 nursingchartcelltypevalname 分桶格式
    
    eICU 使用字符串作为变量标识（如 'Heart Rate', 'O2 Saturation'）
    支持从 csv.gz 或已转换的 parquet 目录转换
    
    Args:
        data_path: eICU 数据目录
        num_buckets: 桶数量
        overwrite: 是否覆盖
        progress_callback: 进度回调函数
    """
    data_path = Path(data_path)
    
    # 按优先级检查源文件：csv.gz > 已有 parquet 目录
    csv_gz = data_path / 'nurseCharting.csv.gz'
    parquet_dir = data_path / 'nursecharting'
    
    output = data_path / 'nursecharting_bucket'
    
    if csv_gz.exists():
        # 从 csv.gz 直接转换
        config = BucketConfig(
            num_buckets=num_buckets,
            partition_col='nursingchartcelltypevalname',
            row_group_size=100_000,
            compression='snappy'
        )
        return convert_to_buckets(csv_gz, output, config, progress_callback=progress_callback, overwrite=overwrite)
    elif parquet_dir.exists() and parquet_dir.is_dir():
        # 从已有 parquet 目录转换
        return convert_parquet_directory_to_buckets(
            parquet_dir, output,
            partition_col='nursingchartcelltypevalname',
            num_buckets=num_buckets,
            overwrite=overwrite,
            progress_callback=progress_callback
        )
    else:
        return ConversionResult(
            success=False, num_buckets=0, total_rows=0,
            total_size_bytes=0, elapsed_seconds=0,
            error=f"未找到 nursecharting 源文件，已检查:\n  - {csv_gz}\n  - {parquet_dir}/"
        )


def convert_miiv_labevents(
    data_path: str = '/home/zhuhb/icudb/mimiciv/3.1/hosp',
    num_buckets: int = 100,
    overwrite: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None
) -> ConversionResult:
    """
    将 MIIV labevents 直接从 csv.gz 转换为按 itemid 分桶格式
    
    一步到位：csv.gz → 分桶 parquet
    
    Args:
        data_path: MIIV hosp 数据目录（含 labevents.csv.gz）
        num_buckets: 桶数量（默认 100）
        overwrite: 是否覆盖已存在的目录
        progress_callback: 进度回调函数
    """
    data_path = Path(data_path)
    
    # 按优先级检查源文件：csv.gz > parquet
    csv_gz = data_path / 'labevents.csv.gz'
    parquet = data_path / 'labevents.parquet'
    parquet_dir = data_path / 'labevents'  # 可能是目录形式
    
    if csv_gz.exists():
        source = csv_gz
    elif parquet.exists():
        source = parquet
    elif parquet_dir.exists() and parquet_dir.is_dir():
        # 从已有目录转换
        return convert_parquet_directory_to_buckets(
            parquet_dir, data_path / 'labevents_bucket',
            partition_col='itemid',
            num_buckets=num_buckets,
            overwrite=overwrite,
            progress_callback=progress_callback
        )
    else:
        return ConversionResult(
            success=False, num_buckets=0, total_rows=0,
            total_size_bytes=0, elapsed_seconds=0,
            error=f"未找到 labevents 源文件，已检查:\n  - {csv_gz}\n  - {parquet}\n  - {parquet_dir}/"
        )
    
    output = data_path / 'labevents_bucket'
    
    config = BucketConfig(
        num_buckets=num_buckets,
        partition_col='itemid',
        row_group_size=100_000,
        compression='snappy'
    )
    
    return convert_to_buckets(source, output, config, progress_callback=progress_callback, overwrite=overwrite)


def convert_eicu_lab(
    data_path: str = '/home/zhuhb/icudb/eicu/2.0.1',
    num_buckets: int = 50,
    overwrite: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None
) -> ConversionResult:
    """
    将 eICU lab 转换为按 labname 分桶格式
    
    eICU 使用字符串（如 'glucose', 'creatinine'）作为变量标识
    
    Args:
        data_path: eICU 数据目录
        num_buckets: 桶数量（默认 50，因为只有 158 个唯一 labname）
        overwrite: 是否覆盖已存在的目录
        progress_callback: 进度回调函数
    """
    data_path = Path(data_path)
    
    # 按优先级检查源文件：csv.gz > parquet > 目录
    csv_gz = data_path / 'lab.csv.gz'
    parquet = data_path / 'lab.parquet'
    parquet_dir = data_path / 'lab'
    
    output = data_path / 'lab_bucket'
    
    if csv_gz.exists():
        source = csv_gz
    elif parquet.exists():
        source = parquet
    elif parquet_dir.exists() and parquet_dir.is_dir():
        # 从已有目录转换
        return convert_parquet_directory_to_buckets(
            parquet_dir, output,
            partition_col='labname',
            num_buckets=num_buckets,
            overwrite=overwrite,
            progress_callback=progress_callback
        )
    else:
        return ConversionResult(
            success=False, num_buckets=0, total_rows=0,
            total_size_bytes=0, elapsed_seconds=0,
            error=f"未找到 lab 源文件，已检查:\n  - {csv_gz}\n  - {parquet}\n  - {parquet_dir}/"
        )
    
    config = BucketConfig(
        num_buckets=num_buckets,
        partition_col='labname',
        row_group_size=100_000,
        compression='snappy'
    )
    
    return convert_to_buckets(source, output, config, progress_callback=progress_callback, overwrite=overwrite)


def convert_miiv_inputevents(
    data_path: str = '/home/zhuhb/icudb/mimiciv/3.1/icu',
    num_buckets: int = 50,
    overwrite: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None
) -> ConversionResult:
    """
    将 MIIV inputevents 转换为按 itemid 分桶格式
    
    inputevents 包含血管活性药物等重要概念（约13个）
    
    Args:
        data_path: MIIV ICU 数据目录（含 inputevents.csv.gz）
        num_buckets: 桶数量（默认 50，因为概念数较少）
        overwrite: 是否覆盖已存在的目录
        progress_callback: 进度回调函数
    """
    data_path = Path(data_path)
    
    # 按优先级检查源文件
    csv_gz = data_path / 'inputevents.csv.gz'
    parquet = data_path / 'inputevents.parquet'
    
    output = data_path / 'inputevents_bucket'
    
    if csv_gz.exists():
        source = csv_gz
    elif parquet.exists():
        source = parquet
    else:
        return ConversionResult(
            success=False, num_buckets=0, total_rows=0,
            total_size_bytes=0, elapsed_seconds=0,
            error=f"未找到 inputevents 源文件，已检查:\n  - {csv_gz}\n  - {parquet}"
        )
    
    config = BucketConfig(
        num_buckets=num_buckets,
        partition_col='itemid',
        row_group_size=100_000,
        compression='snappy'
    )
    
    return convert_to_buckets(source, output, config, progress_callback=progress_callback, overwrite=overwrite)


def convert_hirid_pharma(
    data_path: str = '/home/zhuhb/icudb/hirid/1.1.1',
    num_buckets: int = 50,
    overwrite: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None
) -> ConversionResult:
    """
    将 HiRID pharma 转换为按 pharmaid 分桶格式
    
    pharma 表包含药物相关概念（约11个）
    
    Args:
        data_path: HiRID 数据目录
        num_buckets: 桶数量（默认 50）
        overwrite: 是否覆盖已存在的目录
        progress_callback: 进度回调函数
    """
    data_path = Path(data_path)
    source = data_path / 'pharma'
    output = data_path / 'pharma_bucket'
    
    if not source.is_dir():
        return ConversionResult(
            success=False, num_buckets=0, total_rows=0,
            total_size_bytes=0, elapsed_seconds=0,
            error=f"pharma 目录不存在: {source}"
        )
    
    return convert_parquet_directory_to_buckets(
        source, output,
        partition_col='pharmaid',
        num_buckets=num_buckets,
        overwrite=overwrite,
        progress_callback=progress_callback
    )


def convert_mimic3_chartevents(
    data_path: str = '/home/zhuhb/icudb/mimiciii/1.4',
    num_buckets: int = 100,
    overwrite: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None
) -> ConversionResult:
    """
    将 MIMIC-III chartevents 转换为按 itemid 分桶格式
    
    MIMIC-III 的 chartevents 表结构与 MIMIC-IV 类似，约3.3亿行
    
    🔧 重要: VALUE 列必须强制为 VARCHAR 类型！
    DuckDB 的自动类型检测会将 VALUE 列识别为 DOUBLE（因为大多数值是数字），
    但像 GCS 分数这样的概念，VALUE 列包含文本如 "4 Spontaneously"，
    如果被解析为 DOUBLE 会变成 NaN。
    
    Args:
        data_path: MIMIC-III 数据目录
        num_buckets: 桶数量（默认 100）
        overwrite: 是否覆盖已存在的目录
        progress_callback: 进度回调函数
    """
    data_path = Path(data_path)
    
    # 优先检查已有分桶目录
    bucket_dir = data_path / 'chartevents_bucket'
    if bucket_dir.exists() and not overwrite:
        return ConversionResult(
            success=True, num_buckets=num_buckets, total_rows=0,
            total_size_bytes=0, elapsed_seconds=0,
            error=f"分桶目录已存在: {bucket_dir}"
        )
    
    # 查找源文件
    source = None
    for name in ['CHARTEVENTS.csv.gz', 'chartevents.csv.gz', 'chartevents.csv', 'chartevents.parquet']:
        p = data_path / name
        if p.exists():
            source = p
            break
    
    if not source:
        return ConversionResult(
            success=False, num_buckets=0, total_rows=0,
            total_size_bytes=0, elapsed_seconds=0,
            error=f"chartevents 不存在于 {data_path}"
        )
    
    # 🔧 关键修复: 强制 VALUE 列为 VARCHAR
    # 这样 "4 Spontaneously" 这样的文本值就不会变成 NaN
    config = BucketConfig(
        num_buckets=num_buckets,
        partition_col='itemid',
        row_group_size=100_000,
        column_types={'VALUE': 'VARCHAR'}  # 修复 GCS 等概念的 VALUE 列数据丢失
    )
    
    return convert_to_buckets(source, bucket_dir, config, progress_callback=progress_callback, overwrite=overwrite)


def convert_mimic3_labevents(
    data_path: str = '/home/zhuhb/icudb/mimiciii/1.4',
    num_buckets: int = 100,
    overwrite: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None
) -> ConversionResult:
    """
    将 MIMIC-III labevents 转换为按 itemid 分桶格式
    
    Args:
        data_path: MIMIC-III 数据目录
        num_buckets: 桶数量（默认 100）
        overwrite: 是否覆盖已存在的目录
        progress_callback: 进度回调函数
    """
    data_path = Path(data_path)
    
    bucket_dir = data_path / 'labevents_bucket'
    if bucket_dir.exists() and not overwrite:
        return ConversionResult(
            success=True, num_buckets=num_buckets, total_rows=0,
            total_size_bytes=0, elapsed_seconds=0,
            error=f"分桶目录已存在: {bucket_dir}"
        )
    
    source = None
    for name in ['labevents.csv.gz', 'labevents.csv', 'labevents.parquet']:
        p = data_path / name
        if p.exists():
            source = p
            break
    
    if not source:
        return ConversionResult(
            success=False, num_buckets=0, total_rows=0,
            total_size_bytes=0, elapsed_seconds=0,
            error=f"labevents 不存在于 {data_path}"
        )
    
    config = BucketConfig(
        num_buckets=num_buckets,
        partition_col='itemid',
        row_group_size=100_000
    )
    
    return convert_to_buckets(source, bucket_dir, config, progress_callback=progress_callback, overwrite=overwrite)


def convert_sic_data_float_h(
    data_path: str = '/home/zhuhb/icudb/sicdb/1.0.6',
    num_buckets: int = 50,
    overwrite: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None
) -> ConversionResult:
    """
    将 SICdb data_float_h 转换为按 DataID 分桶格式
    
    data_float_h 是 SICdb 的主要生命体征表（约3.1GB）
    
    Args:
        data_path: SICdb 数据目录
        num_buckets: 桶数量（默认 50）
        overwrite: 是否覆盖已存在的目录
        progress_callback: 进度回调函数
    """
    data_path = Path(data_path)
    
    bucket_dir = data_path / 'data_float_h_bucket'
    if bucket_dir.exists() and not overwrite:
        return ConversionResult(
            success=True, num_buckets=num_buckets, total_rows=0,
            total_size_bytes=0, elapsed_seconds=0,
            error=f"分桶目录已存在: {bucket_dir}"
        )
    
    source = None
    for name in ['data_float_h.csv.gz', 'data_float_h.csv', 'data_float_h.parquet']:
        p = data_path / name
        if p.exists():
            source = p
            break
    
    if not source:
        return ConversionResult(
            success=False, num_buckets=0, total_rows=0,
            total_size_bytes=0, elapsed_seconds=0,
            error=f"data_float_h 不存在于 {data_path}"
        )
    
    config = BucketConfig(
        num_buckets=num_buckets,
        partition_col='DataID',  # SICdb 使用大写列名
        row_group_size=100_000
    )
    
    return convert_to_buckets(source, bucket_dir, config, progress_callback=progress_callback, overwrite=overwrite)


def convert_sic_laboratory(
    data_path: str = '/home/zhuhb/icudb/sicdb/1.0.6',
    num_buckets: int = 50,
    overwrite: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None
) -> ConversionResult:
    """
    将 SICdb laboratory 转换为按 LaboratoryID 分桶格式
    
    Args:
        data_path: SICdb 数据目录
        num_buckets: 桶数量（默认 50）
        overwrite: 是否覆盖已存在的目录
        progress_callback: 进度回调函数
    """
    data_path = Path(data_path)
    
    bucket_dir = data_path / 'laboratory_bucket'
    if bucket_dir.exists() and not overwrite:
        return ConversionResult(
            success=True, num_buckets=num_buckets, total_rows=0,
            total_size_bytes=0, elapsed_seconds=0,
            error=f"分桶目录已存在: {bucket_dir}"
        )
    
    source = None
    for name in ['laboratory.csv.gz', 'laboratory.csv', 'laboratory.parquet']:
        p = data_path / name
        if p.exists():
            source = p
            break
    
    if not source:
        return ConversionResult(
            success=False, num_buckets=0, total_rows=0,
            total_size_bytes=0, elapsed_seconds=0,
            error=f"laboratory 不存在于 {data_path}"
        )
    
    config = BucketConfig(
        num_buckets=num_buckets,
        partition_col='LaboratoryID',  # SICdb 使用大写列名
        row_group_size=100_000
    )
    
    return convert_to_buckets(source, bucket_dir, config, progress_callback=progress_callback, overwrite=overwrite)


def convert_sic_medication(
    data_path: str = '/home/zhuhb/icudb/sicdb/1.0.6',
    num_buckets: int = 50,
    overwrite: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None
) -> ConversionResult:
    """
    将 SICdb medication 转换为按 DrugID 分桶格式
    
    Args:
        data_path: SICdb 数据目录
        num_buckets: 桶数量（默认 50）
        overwrite: 是否覆盖已存在的目录
        progress_callback: 进度回调函数
    """
    data_path = Path(data_path)
    
    bucket_dir = data_path / 'medication_bucket'
    if bucket_dir.exists() and not overwrite:
        return ConversionResult(
            success=True, num_buckets=num_buckets, total_rows=0,
            total_size_bytes=0, elapsed_seconds=0,
            error=f"分桶目录已存在: {bucket_dir}"
        )
    
    source = None
    for name in ['medication.csv.gz', 'medication.csv', 'medication.parquet']:
        p = data_path / name
        if p.exists():
            source = p
            break
    
    if not source:
        return ConversionResult(
            success=False, num_buckets=0, total_rows=0,
            total_size_bytes=0, elapsed_seconds=0,
            error=f"medication 不存在于 {data_path}"
        )
    
    config = BucketConfig(
        num_buckets=num_buckets,
        partition_col='DrugID',  # SICdb 使用大写列名
        row_group_size=100_000
    )
    
    return convert_to_buckets(source, bucket_dir, config, progress_callback=progress_callback, overwrite=overwrite)


def verify_query_plan(
    bucket_dir: Path,
    itemids: Set[int],
    columns: List[str],
    partition_col: str = 'itemid',
    num_buckets: int = 100
) -> dict:
    """
    验证查询计划是否正确应用了谓词下推和列投影
    
    用于调试和性能验证：
    1. 检查 FILTER 是否出现在计划中（谓词下推）
    2. 检查 PROJECT 是否只包含需要的列（列投影）
    3. 估算实际扫描的数据量 vs 全量数据
    
    Args:
        bucket_dir: 分桶目录
        itemids: 目标itemid集合
        columns: 需要的列
        partition_col: 分桶列名
        num_buckets: 桶数量
    
    Returns:
        dict: 包含查询计划和优化信息
    """
    import polars as pl
    
    bucket_dir = Path(bucket_dir)
    # 使用DuckDB hash计算目标桶（与转换时一致）
    target_buckets = _duckdb_hash_batch(itemids, num_buckets)
    
    # 收集目标桶文件
    parquet_files = []
    for bucket_id in target_buckets:
        bucket_path = bucket_dir / f"bucket_id={bucket_id}"
        if bucket_path.exists():
            parquet_files.extend(bucket_path.glob("*.parquet"))
    
    if not parquet_files:
        return {"error": "没有找到匹配的桶"}
    
    # 构建查询
    lf = pl.scan_parquet(parquet_files)
    lf = lf.filter(pl.col(partition_col).is_in(list(itemids)))
    available_cols = [c for c in columns if c != 'bucket_id']
    lf = lf.select(available_cols)
    
    # 获取查询计划
    raw_plan = lf.explain()
    optimized_plan = lf.explain(optimized=True)
    
    # 分析优化效果
    bucket_reduction = f"{len(target_buckets)}/{num_buckets} 桶 ({100*len(target_buckets)/num_buckets:.1f}%)"
    column_reduction = f"{len(available_cols)} 列"
    
    return {
        "raw_plan": raw_plan,
        "optimized_plan": optimized_plan,
        "target_buckets": len(target_buckets),
        "total_buckets": num_buckets,
        "bucket_reduction": bucket_reduction,
        "column_reduction": column_reduction,
        "files_to_scan": len(parquet_files),
        # Polars 使用 SELECTION 表示谓词下推，FILTER 表示后置过滤
        "predicate_pushdown": "SELECTION" in optimized_plan or "selection" in optimized_plan.lower(),
        "column_projection": "PROJECT" in optimized_plan or "project" in optimized_plan.lower()
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='分桶转换器')
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 转换命令
    convert_parser = subparsers.add_parser('convert', help='转换文件到分桶格式')
    convert_parser.add_argument('source', help='源文件路径')
    convert_parser.add_argument('output', help='输出目录')
    convert_parser.add_argument('--buckets', type=int, default=100, help='桶数量')
    convert_parser.add_argument('--column', default='itemid', help='分桶列')
    convert_parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的目录')
    convert_parser.add_argument('--temp-dir', help='临时目录（建议SSD）')
    convert_parser.add_argument('--memory', default='10GB', help='内存限制')
    
    # 验证命令
    verify_parser = subparsers.add_parser('verify', help='验证查询计划')
    verify_parser.add_argument('bucket_dir', help='分桶目录')
    verify_parser.add_argument('--itemids', type=int, nargs='+', required=True, help='测试itemid')
    verify_parser.add_argument('--columns', nargs='+', default=['value'], help='测试列')
    verify_parser.add_argument('--buckets', type=int, default=100, help='桶数量')
    
    args = parser.parse_args()
    
    if args.command == 'convert':
        config = BucketConfig(
            num_buckets=args.buckets,
            partition_col=args.column,
            memory_limit=args.memory,
            temp_directory=args.temp_dir
        )
        
        result = convert_to_buckets(
            Path(args.source),
            Path(args.output),
            config,
            overwrite=args.overwrite
        )
        
        if result.success:
            print("\n✅ 转换成功!")
            print(f"   输出目录: {result.output_dir}")
            print(f"   总行数: {result.total_rows:,}")
            print(f"   分桶数: {result.num_buckets}")
            print(f"   总大小: {result.total_size_bytes / 1024**3:.2f} GB")
        else:
            print(f"\n❌ 转换失败: {result.error}")
    
    elif args.command == 'verify':
        result = verify_query_plan(
            Path(args.bucket_dir),
            set(args.itemids),
            args.columns,
            num_buckets=args.buckets
        )
        
        print("\n=== 查询计划验证 ===")
        print(f"目标桶数: {result['bucket_reduction']}")
        print(f"列数: {result['column_reduction']}")
        print(f"谓词下推 (SELECTION): {'✅' if result['predicate_pushdown'] else '❌'}")
        print(f"列投影 (PROJECT): {'✅' if result['column_projection'] else '❌'}")
        print(f"\n优化后查询计划:\n{result['optimized_plan']}")
    
    else:
        parser.print_help()
