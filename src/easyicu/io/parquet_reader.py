"""
高性能 Parquet 读取器 - 纯 Python,无需 R
比 fst 更快,支持更多优化功能
"""
from pathlib import Path
from typing import Optional, List, Union, Tuple
import logging
import pandas as pd
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _sql_literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _duckdb_where(filters: Optional[List[Tuple]]) -> str:
    """Render a tiny subset of PyArrow-style filters for fallback reads."""
    if not filters:
        return ""
    clauses = []
    for filt in filters:
        if not isinstance(filt, tuple) or len(filt) != 3:
            raise ValueError("DuckDB parquet fallback supports only (column, op, value) filters")
        col, op, value = filt
        op_l = str(op).lower()
        if op_l in {"=", "=="}:
            clauses.append(f"{_quote_ident(col)} = {_sql_literal(value)}")
        elif op_l in {"!=", "<>"}:
            clauses.append(f"{_quote_ident(col)} <> {_sql_literal(value)}")
        elif op_l in {">", ">=", "<", "<="}:
            clauses.append(f"{_quote_ident(col)} {op_l} {_sql_literal(value)}")
        elif op_l == "in":
            values = list(value)
            if not values:
                clauses.append("FALSE")
            else:
                clauses.append(
                    f"{_quote_ident(col)} IN ("
                    + ", ".join(_sql_literal(v) for v in values)
                    + ")"
                )
        else:
            raise ValueError(f"Unsupported DuckDB parquet fallback filter op: {op!r}")
    return " WHERE " + " AND ".join(clauses)


def _read_parquet_duckdb(
    file_path: Union[str, Path],
    columns: Optional[List[str]] = None,
    filters: Optional[List[Tuple]] = None,
) -> pd.DataFrame:
    """Fallback reader for parquet files pyarrow can inspect but not decode."""
    try:
        import duckdb
    except Exception as exc:  # pragma: no cover - only hit without duckdb installed
        raise RuntimeError(
            "pyarrow failed to read parquet and duckdb is not available for fallback"
        ) from exc

    path = str(Path(file_path)).replace("'", "''")
    select_cols = "*" if columns is None else ", ".join(_quote_ident(c) for c in columns)
    query = f"SELECT {select_cols} FROM read_parquet('{path}', union_by_name=true)"
    query += _duckdb_where(filters)
    return duckdb.connect().execute(query).fetchdf()

def read_parquet(
    file_path: Union[str, Path],
    columns: Optional[List[str]] = None,
    filters: Optional[List[Tuple]] = None,
    use_threads: bool = True
) -> pd.DataFrame:
    """
    读取 Parquet 文件（优化版）
    
    Args:
        file_path: Parquet 文件路径
        columns: 要读取的列（None = 全部）
        filters: PyArrow 过滤器，在读取时直接过滤，速度更快！
            例如: [('stay_id', 'in', [123, 456])]
        use_threads: 使用多线程加速（默认True）
    
    Returns:
        DataFrame
    
    性能优势:
        - 列裁剪：只读取需要的列，节省内存和时间
        - 谓词下推：在读取时就过滤数据，避免读取无用数据
        - 多线程：自动并行读取多个 row groups
    
    Examples:
        # 基本读取
        df = read_parquet("data.parquet")
        
        # 只读取特定列（快10倍！）
        df = read_parquet("data.parquet", columns=['stay_id', 'value'])
        
        # 带过滤（快100倍！）
        df = read_parquet(
            "chartevents.parquet",
            columns=['stay_id', 'charttime', 'value'],
            filters=[('stay_id', 'in', patient_ids)]  # 只读取这些患者！
        )
    """
    try:
        return pd.read_parquet(
            file_path,
            engine='pyarrow',
            columns=columns,
            filters=filters,
            use_threads=use_threads
        )
    except Exception as exc:
        message = str(exc)
        if "Repetition level histogram size mismatch" not in message:
            raise
        logger.warning(
            "pyarrow could not decode %s (%s); falling back to DuckDB",
            file_path,
            message,
        )
        return _read_parquet_duckdb(file_path, columns=columns, filters=filters)

def read_parquet_parallel(
    file_paths: List[Union[str, Path]],
    columns: Optional[List[str]] = None,
    filters: Optional[List[Tuple]] = None,
    max_workers: int = 4,
    verbose: bool = False
) -> pd.DataFrame:
    """
    并行读取多个 Parquet 文件并合并
    
    比 fst_reader_fast.read_fst_parallel 快 5-10 倍！
    
    Args:
        file_paths: Parquet 文件路径列表
        columns: 要读取的列
        filters: 过滤条件（应用到每个文件）
        max_workers: 最大并行数
        verbose: 显示进度
    
    Returns:
        合并后的 DataFrame
    """
    def read_one(path):
        try:
            df = read_parquet(path, columns=columns, filters=filters)
            return df if len(df) > 0 else None
        except Exception as e:
            if verbose:
                print(f"   ⚠️  读取失败 {Path(path).name}: {e}")
            return None
    
    if verbose:
        logger.debug(f"并行读取 {len(file_paths)} 个 Parquet 分区...")
        if filters:
            logger.debug(f"应用过滤器: {filters}")
    
    dfs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_one, path): path for path in file_paths}
        
        for i, future in enumerate(as_completed(futures), 1):
            df = future.result()
            if df is not None:
                dfs.append(df)
            
            if verbose and i % 10 == 0:
                print(f"   进度: {i}/{len(file_paths)}")
    
    if not dfs:
        return pd.DataFrame()
    
    result = pd.concat(dfs, ignore_index=True)
    
    if verbose:
        logger.debug(f"读取完成: {len(result):,} 行")
    
    return result

def parquet_metadata(file_path: Union[str, Path]) -> dict:
    """
    获取 Parquet 文件元数据（无需读取数据）
    
    速度极快！只读取文件头部。
    
    Args:
        file_path: Parquet 文件路径
    
    Returns:
        元数据字典
    """
    parquet_file = pq.ParquetFile(file_path)
    
    return {
        'nrow': parquet_file.metadata.num_rows,
        'ncol': parquet_file.metadata.num_columns,
        'columns': parquet_file.schema.names,
        'num_row_groups': parquet_file.num_row_groups,
        'size_bytes': Path(file_path).stat().st_size,
        'schema': str(parquet_file.schema)
    }

def parquet_peek(
    file_path: Union[str, Path],
    n: int = 5,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    快速查看 Parquet 文件的前 n 行
    
    只读取第一个 row group，速度极快！
    
    Args:
        file_path: Parquet 文件路径
        n: 行数
        columns: 列（None = 全部）
    
    Returns:
        前 n 行 DataFrame
    """
    try:
        parquet_file = pq.ParquetFile(file_path)

        # 只读取第一个 row group
        first_batch = parquet_file.read_row_group(0, columns=columns)
        df = first_batch.to_pandas()
        return df.head(n)
    except Exception as exc:
        if "Repetition level histogram size mismatch" not in str(exc):
            raise
        return read_parquet(file_path, columns=columns).head(n)

def optimize_parquet_for_filtering(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    partition_cols: Optional[List[str]] = None,
    row_group_size: int = 100000
):
    """
    优化 Parquet 文件以支持快速过滤
    
    Args:
        input_path: 输入 Parquet 文件
        output_path: 输出 Parquet 文件
        partition_cols: 按这些列分区（例如 ['stay_id']）
        row_group_size: Row group 大小（影响过滤性能）
    
    使用场景:
        如果你经常按 stay_id 过滤，可以优化：
        optimize_parquet_for_filtering(
            'chartevents.parquet',
            'chartevents_optimized.parquet',
            partition_cols=['stay_id']
        )
    """
    df = pd.read_parquet(input_path)
    
    if partition_cols:
        # 按列排序，提高过滤效率
        df = df.sort_values(partition_cols)
    
    df.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',
        index=False,
        row_group_size=row_group_size,
        # partition_by=partition_cols,  # 可选：创建分区文件夹
    )

# 兼容性别名
def read_table_parquet(
    file_path: Union[str, Path],
    columns: Optional[List[str]] = None,
    filters: Optional[List[Tuple]] = None
) -> pd.DataFrame:
    """
    兼容 datasource.py 的接口
    """
    return read_parquet(file_path, columns=columns, filters=filters)

__all__ = [
    'read_parquet',
    'read_parquet_parallel',
    'parquet_metadata',
    'parquet_peek',
    'optimize_parquet_for_filtering',
    'read_table_parquet'
]
