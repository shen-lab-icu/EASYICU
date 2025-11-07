"""Data export utilities (R ricu utils-export.R).

Provides functions for exporting ICU data in various formats,
including PhysioNet Sepsis Challenge PSV format.
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd
import numpy as np


def write_psv(
    data: pd.DataFrame,
    output_dir: Union[str, Path],
    id_col: str,
    index_col: Optional[str] = None,
    na_rows: Optional[bool] = None,
) -> None:
    """Write data in PSV format (PhysioNet Sepsis Challenge).
    
    Writes data to pipe-separated value files, with one file per patient ID.
    Files are named with patient IDs (e.g., 'p001.psv').
    
    Args:
        data: DataFrame to export
        output_dir: Directory to write files to
        id_col: Column containing patient IDs
        index_col: Time index column (will be converted to hours if datetime)
        na_rows: If True, fill gaps with NaN; if False, remove all-NA rows;
                 if None, write as-is
                 
    Examples:
        >>> df = pd.DataFrame({
        ...     'patient_id': [1, 1, 2, 2],
        ...     'time': pd.date_range('2020-01-01', periods=4, freq='H'),
        ...     'hr': [80, 85, 90, 95],
        ...     'sbp': [120, 125, 130, 135]
        ... })
        >>> write_psv(df, 'output_dir', 'patient_id', 'time')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = data.copy()
    
    # Handle na_rows option
    if na_rows is True:
        # Fill gaps (would need fill_gaps function)
        from .ts_utils import fill_gaps
        data = fill_gaps(data, [id_col], index_col, pd.Timedelta(hours=1))
    elif na_rows is False:
        # Remove rows where all data columns are NA
        data_cols = [c for c in data.columns if c not in [id_col, index_col]]
        data = data.dropna(subset=data_cols, how='all')
    
    # Convert time to hours if datetime
    if index_col and pd.api.types.is_datetime64_any_dtype(data[index_col]):
        # Convert to hours from start
        for patient_id, group in data.groupby(id_col):
            min_time = group[index_col].min()
            data.loc[data[id_col] == patient_id, index_col] = \
                (group[index_col] - min_time).dt.total_seconds() / 3600
    
    # Split by patient and write files
    for patient_id, group in data.groupby(id_col):
        # Remove ID column from output
        output_data = group.drop(columns=[id_col])
        
        # Write to PSV file
        filename = output_dir / f"p{patient_id}.psv"
        output_data.to_csv(filename, sep='|', index=False, na_rep='NaN')
    
    print(f"Wrote {data[id_col].nunique()} PSV files to {output_dir}")


def read_psv(
    input_dir: Union[str, Path],
    id_col: str = "stay_id",
    index_col: Optional[str] = None,
) -> pd.DataFrame:
    """Read data from PSV format files (R ricu read_psv).
    
    Reads pipe-separated value files from a directory and combines them
    into a single DataFrame, extracting patient IDs from filenames.
    
    Args:
        input_dir: Directory containing PSV files
        id_col: Name for patient ID column
        index_col: Name for time index column (will be converted to timedelta)
        
    Returns:
        Combined DataFrame with data from all patients
        
    Examples:
        >>> df = read_psv('input_dir', 'patient_id', 'time')
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    
    # Find all PSV files
    psv_files = list(input_dir.glob("*.psv"))
    
    if not psv_files:
        raise ValueError(f"No PSV files found in {input_dir}")
    
    all_data = []
    
    for filepath in psv_files:
        # Extract patient ID from filename (e.g., 'p001.psv' -> 1)
        patient_id = filepath.stem
        if patient_id.startswith('p'):
            patient_id = int(patient_id[1:])
        else:
            patient_id = int(patient_id)
        
        # Read file
        df = pd.read_csv(filepath, sep='|', na_values='NaN')
        df[id_col] = patient_id
        
        all_data.append(df)
    
    # Combine all data
    result = pd.concat(all_data, ignore_index=True)
    
    # Convert index to timedelta if specified
    if index_col and index_col in result.columns:
        result[index_col] = pd.to_timedelta(result[index_col], unit='h')
    
    return result


def export_wide_format(
    data: pd.DataFrame,
    output_file: Union[str, Path],
    id_cols: list,
    index_col: str,
    sep: str = ',',
) -> None:
    """Export data in wide format.
    
    Args:
        data: DataFrame to export
        output_file: Output file path
        id_cols: ID columns
        index_col: Time index column
        sep: Delimiter (default: comma)
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    data.to_csv(output_file, sep=sep, index=False, na_rep='NA')
    print(f"Exported to {output_file}")


def export_long_format(
    data: pd.DataFrame,
    output_file: Union[str, Path],
    id_cols: list,
    index_col: str,
    value_vars: Optional[list] = None,
    sep: str = ',',
) -> None:
    """Export data in long (melted) format.
    
    Converts wide format to long format where each measurement is a row.
    
    Args:
        data: DataFrame to export
        output_file: Output file path
        id_cols: ID columns
        index_col: Time index column
        value_vars: Variables to melt (if None, all except id/index)
        sep: Delimiter (default: comma)
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine value variables
    if value_vars is None:
        value_vars = [c for c in data.columns if c not in id_cols + [index_col]]
    
    # Melt to long format
    long_data = data.melt(
        id_vars=id_cols + [index_col],
        value_vars=value_vars,
        var_name='variable',
        value_name='value'
    )
    
    # Remove NA values
    long_data = long_data.dropna(subset=['value'])
    
    long_data.to_csv(output_file, sep=sep, index=False, na_rep='NA')
    print(f"Exported to {output_file} (long format)")


def export_summary(
    data: pd.DataFrame,
    output_file: Union[str, Path],
    id_cols: list,
    numeric_cols: Optional[list] = None,
) -> None:
    """Export summary statistics.
    
    Args:
        data: DataFrame to summarize
        output_file: Output file path
        id_cols: ID columns
        numeric_cols: Numeric columns to summarize (if None, auto-detect)
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect numeric columns
    if numeric_cols is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in id_cols]
    
    # Calculate summary statistics
    summary = data[numeric_cols].describe()
    
    summary.to_csv(output_file)
    print(f"Exported summary to {output_file}")


def export_cohort_info(
    data: pd.DataFrame,
    output_file: Union[str, Path],
    id_cols: list,
    index_col: Optional[str] = None,
) -> None:
    """Export cohort information (patient counts, time ranges, etc.).
    
    Args:
        data: DataFrame with patient data
        output_file: Output file path
        id_cols: ID columns
        index_col: Time index column (optional)
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    info = {}
    
    # Basic counts
    info['n_patients'] = data[id_cols[0]].nunique()
    info['n_observations'] = len(data)
    
    # Time range if available
    if index_col and index_col in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[index_col]):
            info['start_time'] = data[index_col].min()
            info['end_time'] = data[index_col].max()
            info['duration'] = info['end_time'] - info['start_time']
    
    # Variable counts
    info['n_variables'] = len([c for c in data.columns if c not in id_cols + [index_col]])
    
    # Missing data
    total_cells = data.shape[0] * data.shape[1]
    missing_cells = data.isna().sum().sum()
    info['missing_pct'] = (missing_cells / total_cells) * 100
    
    # Convert to DataFrame and save
    info_df = pd.DataFrame([info])
    info_df.to_csv(output_file, index=False)
    print(f"Exported cohort info to {output_file}")


# ============================================================================
# Additional export formats
# ============================================================================

def export_parquet(
    data: pd.DataFrame,
    output_file: Union[str, Path],
    compression: str = 'snappy',
) -> None:
    """Export data to Parquet format (R ricu export_parquet).
    
    Args:
        data: DataFrame to export
        output_file: Output file path
        compression: Compression method ('snappy', 'gzip', 'brotli', 'none')
        
    Examples:
        >>> export_parquet(df, 'output.parquet', compression='snappy')
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required for Parquet export. Install with: pip install pyarrow")
    
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    data.to_parquet(output_file, compression=compression)
    print(f"Exported to {output_file} (Parquet format)")


def export_feather(
    data: pd.DataFrame,
    output_file: Union[str, Path],
    compression: str = 'lz4',
) -> None:
    """Export data to Feather format (R ricu export_feather).
    
    Args:
        data: DataFrame to export
        output_file: Output file path
        compression: Compression method ('lz4', 'zstd', 'uncompressed')
        
    Examples:
        >>> export_feather(df, 'output.feather')
    """
    try:
        import pyarrow.feather as feather
    except ImportError:
        raise ImportError("pyarrow is required for Feather export. Install with: pip install pyarrow")
    
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    data.to_feather(output_file, compression=compression)
    print(f"Exported to {output_file} (Feather format)")


def export_json(
    data: pd.DataFrame,
    output_file: Union[str, Path],
    orient: str = 'records',
    indent: Optional[int] = 2,
) -> None:
    """Export data to JSON format (R ricu export_json).
    
    Args:
        data: DataFrame to export
        output_file: Output file path
        orient: JSON orientation ('records', 'index', 'columns', 'values', 'table')
        indent: Indentation level (None for compact)
        
    Examples:
        >>> export_json(df, 'output.json', orient='records')
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    data.to_json(output_file, orient=orient, indent=indent, date_format='iso')
    print(f"Exported to {output_file} (JSON format)")


def export_data(
    data: pd.DataFrame,
    output_file: Union[str, Path],
    format: str = 'auto',
    **kwargs
) -> None:
    """Export data with auto-detected format (R ricu export_data).
    
    Automatically detects format from file extension and exports accordingly.
    
    Args:
        data: DataFrame to export
        output_file: Output file path
        format: Format override ('auto', 'csv', 'parquet', 'feather', 'json', 'psv')
        **kwargs: Additional arguments passed to specific export function
        
    Examples:
        >>> export_data(df, 'output.csv')
        >>> export_data(df, 'output.parquet')
        >>> export_data(df, 'output.feather')
    """
    output_file = Path(output_file)
    
    # Auto-detect format from extension
    if format == 'auto':
        suffix = output_file.suffix.lower()
        format_map = {
            '.csv': 'csv',
            '.parquet': 'parquet',
            '.pq': 'parquet',
            '.feather': 'feather',
            '.ftr': 'feather',
            '.json': 'json',
            '.psv': 'psv',
        }
        format = format_map.get(suffix, 'csv')
    
    # Export with appropriate function
    if format == 'csv':
        sep = kwargs.get('sep', ',')
        data.to_csv(output_file, sep=sep, index=False, na_rep='NA')
        print(f"Exported to {output_file} (CSV format)")
    
    elif format == 'psv':
        id_col = kwargs.get('id_col')
        index_col = kwargs.get('index_col')
        if id_col is None:
            raise ValueError("id_col must be specified for PSV format")
        # PSV writes multiple files
        write_psv(data, output_file.parent, id_col, index_col, kwargs.get('na_rows'))
    
    elif format == 'parquet':
        export_parquet(data, output_file, kwargs.get('compression', 'snappy'))
    
    elif format == 'feather':
        export_feather(data, output_file, kwargs.get('compression', 'lz4'))
    
    elif format == 'json':
        export_json(data, output_file, kwargs.get('orient', 'records'), kwargs.get('indent', 2))
    
    else:
        raise ValueError(f"Unknown format: {format}")


def data_quality_report(
    data: pd.DataFrame,
    output_file: Union[str, Path],
    id_cols: Optional[list] = None,
    index_col: Optional[str] = None,
) -> pd.DataFrame:
    """Generate data quality report (R ricu data_quality_report).
    
    Creates a comprehensive report on data completeness, ranges, and quality.
    
    Args:
        data: DataFrame to analyze
        output_file: Output file path for report
        id_cols: ID columns
        index_col: Time index column
        
    Returns:
        DataFrame with quality metrics
        
    Examples:
        >>> report = data_quality_report(df, 'quality_report.csv', ['patient_id'], 'time')
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    id_cols = id_cols or []
    meta_cols = id_cols + ([index_col] if index_col else [])
    data_cols = [c for c in data.columns if c not in meta_cols]
    
    quality_metrics = []
    
    for col in data_cols:
        metrics = {
            'variable': col,
            'type': str(data[col].dtype),
            'count': len(data),
            'missing': data[col].isna().sum(),
            'missing_pct': (data[col].isna().sum() / len(data)) * 100,
            'unique': data[col].nunique(),
        }
        
        # Numeric stats
        if pd.api.types.is_numeric_dtype(data[col]):
            metrics.update({
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'q25': data[col].quantile(0.25),
                'median': data[col].median(),
                'q75': data[col].quantile(0.75),
                'max': data[col].max(),
            })
        
        quality_metrics.append(metrics)
    
    report_df = pd.DataFrame(quality_metrics)
    report_df.to_csv(output_file, index=False)
    
    print(f"Data quality report saved to {output_file}")
    
    return report_df
