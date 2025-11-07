"""
Fast FST file reader using R directly without CSV conversion.
Optimized for large files by reading directly into Python.
"""
import subprocess
import tempfile
import io
from pathlib import Path
from typing import Optional, Union, List
import pandas as pd


def read_fst_fast(file_path: Union[str, Path], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fast FST reading using R's fst package.
    
    First tries to use feather format (requires R arrow package).
    Falls back to CSV if arrow is not available.
    
    Args:
        file_path: Path to the FST file
        columns: Optional list of column names to read (None = all columns)
        
    Returns:
        DataFrame with the data
    """
    file_path = Path(file_path)
    
    # First try feather (faster but requires arrow)
    try:
        return _read_fst_via_feather(file_path, columns)
    except RuntimeError as e:
        if "arrow" in str(e).lower():
            # Fallback to CSV
            return _read_fst_via_csv(file_path, columns)
        raise


def _read_fst_via_feather(file_path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Read FST via feather format (requires R arrow package)."""
    with tempfile.NamedTemporaryFile(suffix='.feather', delete=False) as tmp:
        feather_path = tmp.name
    
    try:
        if columns:
            cols_arg = f"columns = c({', '.join(repr(c) for c in columns)})"
        else:
            cols_arg = ""
        
        r_script = f"""
        library(fst)
        library(arrow)
        data <- read_fst("{file_path}", {cols_arg})
        write_feather(data, "{feather_path}")
        """
        
        result = subprocess.run(
            ["R", "--vanilla", "--slave", "-e", r_script],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"R conversion failed: {result.stderr}")
        
        df = pd.read_feather(feather_path)
        return df
    finally:
        Path(feather_path).unlink(missing_ok=True)


def _read_fst_via_csv(file_path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Read FST via CSV format (slower but no extra R packages needed)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        csv_path = tmp.name
    
    try:
        if columns:
            cols_arg = f"columns = c({', '.join(repr(c) for c in columns)})"
        else:
            cols_arg = ""
        
        r_script = f"""
        library(fst)
        data <- read_fst("{file_path}", {cols_arg})
        write.csv(data, "{csv_path}", row.names = FALSE)
        """
        
        result = subprocess.run(
            ["R", "--vanilla", "--slave", "-e", r_script],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"R conversion failed: {result.stderr}")
        
        df = pd.read_csv(csv_path)
        return df
    finally:
        Path(csv_path).unlink(missing_ok=True)


def read_fst_direct_rpy2(file_path: Union[str, Path], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Direct FST reading using rpy2 with optimized settings.
    This is the fastest method but requires rpy2 to work properly.
    
    Args:
        file_path: Path to the FST file
        columns: Optional list of column names to read
        
    Returns:
        DataFrame with the data
    """
    try:
        from rpy2 import robjects
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        
        # Activate pandas conversion
        pandas2ri.activate()
        
        # Import fst package
        base = importr('base')
        fst = importr('fst')
        
        # Read FST file
        if columns:
            r_cols = robjects.StrVector(columns)
            r_df = fst.read_fst(str(file_path), columns=r_cols)
        else:
            r_df = fst.read_fst(str(file_path))
        
        # Convert to pandas
        df = pandas2ri.rpy2py(r_df)
        
        pandas2ri.deactivate()
        return df
        
    except Exception as e:
        raise RuntimeError(f"rpy2 direct reading failed: {e}")


def read_fst_parallel(
    file_paths: List[Union[str, Path]], 
    columns: Optional[List[str]] = None,
    max_workers: int = 4,
    verbose: bool = False,
    patient_ids_filter: Optional[tuple] = None
) -> pd.DataFrame:
    """
    Read multiple FST files in parallel and concatenate.
    
    Args:
        file_paths: List of FST file paths
        columns: Optional list of columns to read
        max_workers: Maximum number of parallel workers
        verbose: Show progress information
        patient_ids_filter: Optional (column_name, set_of_ids) for filtering during read
        
    Returns:
        Concatenated DataFrame
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def read_one(path):
        try:
            df = read_fst_fast(path, columns)
            # Apply patient ID filter immediately after reading
            if patient_ids_filter is not None and df is not None and len(df) > 0:
                id_column, patient_ids = patient_ids_filter
                if id_column in df.columns:
                    df = df[df[id_column].isin(patient_ids)]
            return df
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return None
    
    if verbose and len(file_paths) > 5:
        print(f"   正在并行读取 {len(file_paths)} 个分区文件...")
        if patient_ids_filter is not None:
            id_col, ids = patient_ids_filter
            print(f"   ⚡ 读取时立即过滤 {len(ids)} 个患者 (列: {id_col})")
    
    dfs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_one, path): path for path in file_paths}
        
        total = len(futures)
        completed = 0
        for future in as_completed(futures):
            df = future.result()
            if df is not None and len(df) > 0:  # Skip empty DataFrames
                dfs.append(df)
            completed += 1
            if verbose and len(file_paths) > 5 and completed % 5 == 0:
                print(f"   进度: {completed}/{total} 个分区已读取...")
    
    if not dfs:
        raise RuntimeError("No files were successfully read")
    
    if verbose and len(file_paths) > 5:
        total_rows = sum(len(df) for df in dfs)
        print(f"   完成读取 ({len(dfs)} 个分区, 共 {total_rows:,} 行), 正在合并数据...")
    
    # 优化合并：如果有很多DataFrame，分批合并
    if len(dfs) > 10:
        # 分批合并，避免一次性合并太多数据
        batch_size = 10
        result = dfs[0]
        for i in range(1, len(dfs), batch_size):
            batch = dfs[i:i+batch_size]
            if verbose and len(file_paths) > 5:
                print(f"   合并进度: {min(i+batch_size, len(dfs))}/{len(dfs)} 个分区...")
            result = pd.concat([result] + batch, ignore_index=True)
        return result
    else:
        return pd.concat(dfs, ignore_index=True)


def check_fst_file_info(file_path: Union[str, Path]) -> dict:
    """
    Get FST file metadata without reading all data.
    
    Args:
        file_path: Path to FST file
        
    Returns:
        Dictionary with file info (nrow, ncol, column names)
    """
    r_script = f"""
    library(fst)
    meta <- fst::metadata_fst("{file_path}")
    cat(meta$nrOfRows, "\n")
    cat(length(meta$columnNames), "\n")
    cat(paste(meta$columnNames, collapse=","), "\n")
    """
    
    result = subprocess.run(
        ["R", "--vanilla", "--slave", "-e", r_script],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get FST metadata: {result.stderr}")
    
    lines = result.stdout.strip().split('\n')
    return {
        'nrow': int(lines[0]),
        'ncol': int(lines[1]),
        'columns': lines[2].split(',') if len(lines) > 2 else []
    }
