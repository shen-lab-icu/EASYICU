"""
FST file reader - wrapper around fst_reader_fast.
Provides a simpler interface for reading FST files.
"""
from pathlib import Path
from typing import Optional, List, Union
import pandas as pd

from .fst_reader_fast import read_fst_fast, check_fst_file_info


def read_fst(file_path: Union[str, Path], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Read an FST file into a pandas DataFrame.
    
    This is a simple wrapper around read_fst_fast for compatibility.
    
    Args:
        file_path: Path to the FST file
        columns: Optional list of column names to read (None = all columns)
        
    Returns:
        DataFrame with the data
        
    Examples:
        >>> df = read_fst("data.fst")
        >>> df = read_fst("data.fst", columns=["id", "value"])
    """
    return read_fst_fast(file_path, columns=columns)


def fst_metadata(file_path: Union[str, Path]) -> dict:
    """
    Get metadata from an FST file without reading all data.
    
    Args:
        file_path: Path to the FST file
        
    Returns:
        Dictionary with file metadata (nrow, ncol, columns)
        
    Examples:
        >>> info = fst_metadata("data.fst")
        >>> print(f"Rows: {info['nrow']}, Columns: {info['ncol']}")
    """
    return check_fst_file_info(file_path)


__all__ = ['read_fst', 'fst_metadata']
