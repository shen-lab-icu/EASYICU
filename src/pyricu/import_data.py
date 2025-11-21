"""Data import utilities for converting raw CSV files to efficient formats.

This module handles importing downloaded ICU data, converting CSV files to
Parquet format for efficient access, handling large files with chunking,
and managing partitioned tables.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from .config import DataSourceConfig, TableConfig

LOGGER = logging.getLogger(__name__)

def import_table(
    table_cfg: TableConfig,
    data_dir: Path,
    *,
    chunk_size: int = 10_000_000,
    force: bool = False,
    verbose: bool = True,
) -> None:
    """Import a single table from CSV to Parquet format.

    Args:
        table_cfg: Table configuration
        data_dir: Directory containing the raw CSV files
        chunk_size: Number of rows to process at once for large files
        force: If True, re-import even if Parquet file exists
        verbose: If True, print progress information
    """
    data_dir = Path(data_dir)
    table_name = table_cfg.name

    # Determine output file path
    parquet_file = data_dir / f"{table_name}.parquet"

    if parquet_file.exists() and not force:
        if verbose:
            LOGGER.info(f"Table {table_name} already imported, skipping")
        return

    # Get source CSV file(s)
    csv_files = []
    for file_entry in table_cfg.files:
        file_path = file_entry.get("path") or file_entry.get("file")
        if file_path:
            csv_files.append(data_dir / file_path)

    if not csv_files:
        LOGGER.warning(f"No source files found for table {table_name}")
        return

    # Check if files exist
    missing = [f for f in csv_files if not f.exists()]
    if missing:
        LOGGER.error(f"Missing source files: {missing}")
        return

    if verbose:
        LOGGER.info(f"Importing table {table_name} from {len(csv_files)} file(s)")

    try:
        # Read and concatenate CSV files
        if len(csv_files) == 1:
            df = _import_single_csv(csv_files[0], table_cfg, chunk_size, verbose)
        else:
            dfs = []
            for csv_file in csv_files:
                chunk_df = _import_single_csv(csv_file, table_cfg, chunk_size, verbose)
                dfs.append(chunk_df)
            df = pd.concat(dfs, ignore_index=True)

        # Write to Parquet with high compression
        df.to_parquet(
            parquet_file,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

        if verbose:
            LOGGER.info(
                f"Successfully imported {table_name}: {len(df)} rows, "
                f"{len(df.columns)} columns"
            )

    except Exception as e:
        LOGGER.error(f"Failed to import table {table_name}: {e}")
        if parquet_file.exists():
            parquet_file.unlink()
        raise

def _import_single_csv(
    csv_file: Path,
    table_cfg: TableConfig,
    chunk_size: int,
    verbose: bool,
) -> pd.DataFrame:
    """Import a single CSV file with optional chunking for large files.

    Args:
        csv_file: Path to CSV file
        table_cfg: Table configuration
        chunk_size: Number of rows per chunk
        verbose: If True, print progress

    Returns:
        Imported DataFrame
    """
    # Prepare dtype mapping from column configuration
    dtype_map = {}
    if table_cfg.columns:
        for col_name, col_info in table_cfg.columns.items():
            if isinstance(col_info, dict) and "type" in col_info:
                dtype_map[col_name] = _map_dtype(col_info["type"])

    # Identify datetime columns
    parse_dates = []
    defaults = table_cfg.defaults
    if defaults.index_var:
        parse_dates.append(defaults.index_var)
    if defaults.time_vars:
        parse_dates.extend(defaults.time_vars)

    # Remove duplicates
    parse_dates = list(set(parse_dates))

    if verbose:
        LOGGER.info(f"Reading {csv_file.name}...")

    # Check file size to determine if chunking is needed
    file_size = csv_file.stat().st_size
    use_chunks = file_size > 500_000_000  # 500 MB threshold

    if use_chunks:
        # Read in chunks for large files
        chunks = []
        reader = pd.read_csv(
            csv_file,
            dtype=dtype_map,
            parse_dates=parse_dates,
            chunksize=chunk_size,
            low_memory=False,
        )
        for i, chunk in enumerate(reader):
            chunks.append(chunk)
            if verbose and (i + 1) % 10 == 0:
                LOGGER.info(f"  Processed {(i + 1) * chunk_size:,} rows...")
        
        df = pd.concat(chunks, ignore_index=True)
    else:
        # Read entire file at once for smaller files
        df = pd.read_csv(
            csv_file,
            dtype=dtype_map,
            parse_dates=parse_dates,
            low_memory=False,
        )

    return df

def _map_dtype(type_str: str) -> Optional[str]:
    """Map configuration type strings to pandas dtypes.

    Args:
        type_str: Type string from configuration

    Returns:
        Pandas dtype string or None
    """
    type_map = {
        "integer": "Int64",
        "numeric": "float64",
        "character": "str",
        "logical": "boolean",
    }
    return type_map.get(type_str.lower())

def import_src(
    config: DataSourceConfig,
    data_dir: Path,
    *,
    tables: Optional[Sequence[str]] = None,
    force: bool = False,
    verbose: bool = True,
    cleanup: bool = False,
) -> None:
    """Import all tables for a data source.

    Args:
        config: Data source configuration
        data_dir: Directory containing raw CSV files
        tables: List of table names to import; None imports all
        force: If True, re-import existing tables
        verbose: If True, print progress information
        cleanup: If True, delete CSV files after successful import
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    data_dir = Path(data_dir)

    # Determine which tables to import
    if tables is None:
        tables = list(config.tables.keys())

    # Filter out already imported tables if not forcing
    if not force:
        existing = []
        for table_name in tables:
            parquet_file = data_dir / f"{table_name}.parquet"
            if parquet_file.exists():
                existing.append(table_name)
        
        tables = [t for t in tables if t not in existing]
        if existing and verbose:
            LOGGER.info(f"Skipping {len(existing)} already imported tables")

    if not tables:
        LOGGER.info("All requested tables have already been imported")
        return

    if verbose:
        LOGGER.info(f"Importing {len(tables)} table(s) for {config.name}")

    # Import each table
    failed = []
    for table_name in tables:
        try:
            table_cfg = config.get_table(table_name)
            import_table(table_cfg, data_dir, force=force, verbose=verbose)
        except Exception as e:
            LOGGER.error(f"Failed to import table {table_name}: {e}")
            failed.append(table_name)

    if failed:
        LOGGER.warning(f"Failed to import {len(failed)} tables: {', '.join(failed)}")
    elif verbose:
        LOGGER.info(f"Successfully imported all {len(tables)} tables")

    # Cleanup CSV files if requested
    if cleanup and not failed:
        for table_name in tables:
            table_cfg = config.get_table(table_name)
            for file_entry in table_cfg.files:
                file_path = file_entry.get("path") or file_entry.get("file")
                if file_path:
                    csv_file = data_dir / file_path
                    if csv_file.exists():
                        csv_file.unlink()
                        if verbose:
                            LOGGER.info(f"Removed {csv_file.name}")

def import_sources(
    source_names: Iterable[str],
    registry,
    data_dirs: Sequence[Path | str],
    **kwargs,
) -> None:
    """Import multiple data sources.

    Args:
        source_names: List of data source names
        registry: Registry containing data source configurations
        data_dirs: Directories corresponding to each source
        **kwargs: Additional arguments passed to import_src
    """
    for source_name, data_dir in zip(source_names, data_dirs):
        try:
            config = registry.get(source_name)
            import_src(config, Path(data_dir), **kwargs)
        except Exception as e:
            LOGGER.error(f"Failed to import {source_name}: {e}")
