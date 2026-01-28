"""
DuckDB-based Data Converter for pyricu

Memory-efficient CSV to Parquet conversion using DuckDB's streaming capabilities.
Designed to work within 12GB RAM constraint (16GB absolute max).

Key advantages over pandas-based conversion:
- Strict memory control via DuckDB's memory_limit setting
- Native CSV parsing without loading entire file into memory
- Streaming writes to Parquet
- Better handling of large compressed files

Usage:
    from pyricu.duckdb_converter import DuckDBConverter
    
    converter = DuckDBConverter('/path/to/database', memory_limit_gb=6)
    converter.convert_all()
"""

from __future__ import annotations

import os
import gc
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class DuckDBConverter:
    """Memory-efficient data converter using DuckDB.
    
    Converts CSV/CSV.GZ files to Parquet format with strict memory control.
    Target: Work within 12GB RAM, 16GB absolute maximum.
    """
    
    # Tables that should be converted (prioritize large tables)
    PRIORITY_TABLES = {
        'miiv': ['chartevents', 'labevents', 'inputevents', 'outputevents', 'emar', 'emar_detail', 
                 'procedureevents', 'datetimeevents', 'ingredientevents', 'pharmacy'],
        'eicu': ['nursecharting', 'vitalperiodic', 'vitalaperiodic', 'lab', 'medication',
                 'infusiondrug', 'respiratorycharting', 'microlab'],
        'aumc': ['numericitems', 'listitems', 'drugitems', 'freetextitems', 'procedureorderitems'],
        'hirid': ['observations', 'pharma', 'ordersentry'],
        'mimic': ['chartevents', 'labevents', 'inputevents_mv', 'inputevents_cv', 
                  'outputevents', 'procedureevents_mv', 'datetimeevents'],
        'sic': ['data_float_h', 'laboratory', 'medication', 'cases'],
    }
    
    def __init__(
        self, 
        data_path: str,
        memory_limit_gb: float = 6.0,
        temp_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        """Initialize DuckDB converter.
        
        Args:
            data_path: Path to database directory (e.g., /path/to/mimiciv/3.1)
            memory_limit_gb: Maximum memory DuckDB can use (default 6GB for safety)
            temp_dir: Directory for temporary files (default: system temp)
            verbose: Whether to print progress messages
        """
        self.data_path = Path(data_path)
        self.memory_limit_gb = memory_limit_gb
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.verbose = verbose
        
        # Detect database type from path
        self.db_type = self._detect_database_type()
        
        if self.verbose:
            logger.info(f"DuckDBConverter initialized:")
            logger.info(f"  Data path: {self.data_path}")
            logger.info(f"  Database type: {self.db_type}")
            logger.info(f"  Memory limit: {self.memory_limit_gb}GB")
    
    def _detect_database_type(self) -> str:
        """Detect database type from path."""
        path_str = str(self.data_path).lower()
        
        if 'mimiciv' in path_str or 'mimic-iv' in path_str:
            return 'miiv'
        elif 'mimiciii' in path_str or 'mimic-iii' in path_str:
            return 'mimic'
        elif 'eicu' in path_str:
            return 'eicu'
        elif 'aumc' in path_str or 'amsterdam' in path_str:
            return 'aumc'
        elif 'hirid' in path_str:
            return 'hirid'
        elif 'sicdb' in path_str or 'sic' in path_str:
            return 'sic'
        else:
            return 'unknown'
    
    # HiRID 参考文件和非数据表（不需要转换）
    HIRID_SKIP_FILES = {
        'hirid_variable_reference_preprocessed',
        'hirid_variable_reference',
        'hirid_outcome_imputation_parameters',
        'general_table',
        'apache_patient_result',
    }
    
    def _find_csv_files(self) -> List[Path]:
        """Find all CSV and CSV.GZ files in the data directory."""
        csv_files = []
        
        for ext in ['*.csv', '*.csv.gz', '*.CSV', '*.CSV.GZ']:
            csv_files.extend(self.data_path.glob(ext))
        
        # Also check common subdirectories
        for subdir in ['hosp', 'icu', 'core', 'mimic_core', 'mimic_hosp', 'mimic_icu']:
            subpath = self.data_path / subdir
            if subpath.exists():
                for ext in ['*.csv', '*.csv.gz', '*.CSV', '*.CSV.GZ']:
                    csv_files.extend(subpath.glob(ext))
        
        # 过滤掉不需要转换的文件（如 HiRID 参考文件）
        filtered_files = []
        for f in csv_files:
            # 获取文件名（去掉所有扩展名）
            stem = f.stem
            if stem.endswith('.csv'):
                stem = stem[:-4]
            stem_lower = stem.lower()
            
            # 跳过 HiRID 参考文件
            if self.db_type == 'hirid' and stem_lower in self.HIRID_SKIP_FILES:
                if self.verbose:
                    logger.info(f"  ⏭️ Skipping reference file: {f.name}")
                continue
            
            filtered_files.append(f)
        
        return sorted(filtered_files)
    
    def _get_parquet_path(self, csv_path: Path) -> Path:
        """Get the output parquet path for a CSV file."""
        # Remove .gz extension if present
        stem = csv_path.stem
        if stem.endswith('.csv'):
            stem = stem[:-4]
        elif csv_path.suffix.lower() == '.csv':
            stem = csv_path.stem
        
        return csv_path.parent / f"{stem}.parquet"
    
    def _decompress_gz_if_needed(self, csv_path: Path) -> Path:
        """Check if decompression is needed.
        
        DuckDB can read .gz files directly with read_csv's compression='gzip' option.
        We no longer need to decompress to temp files, which saves memory and disk space.
        """
        # DuckDB handles .gz files directly - no decompression needed!
        return csv_path
    
    def convert_file(self, csv_path: Path) -> Dict[str, Any]:
        """Convert a single CSV file to Parquet using DuckDB.
        
        This is the core memory-efficient conversion function.
        Uses DuckDB's COPY command for streaming write.
        """
        import duckdb
        
        result = {
            'file': csv_path.name,
            'status': 'pending',
            'row_count': 0,
            'error': None,
        }
        
        parquet_path = self._get_parquet_path(csv_path)
        temp_csv_path = None
        
        try:
            # Decompress if needed
            source_path = self._decompress_gz_if_needed(csv_path)
            if source_path != csv_path:
                temp_csv_path = source_path
            
            # Create DuckDB connection with optimized settings
            con = duckdb.connect(':memory:')
            con.execute(f"SET memory_limit = '{self.memory_limit_gb}GB'")
            # 不设置线程数，让DuckDB自动检测CPU核心数
            con.execute("SET preserve_insertion_order = false")  # 允许并行写入
            
            # Read CSV and write to Parquet in one streaming operation
            # DuckDB handles this efficiently without loading entire file
            if self.verbose:
                logger.info(f"  Converting {csv_path.name} -> {parquet_path.name}...")
            
            # Use COPY for memory-efficient streaming
            # First, create a view of the CSV file
            escaped_path = str(source_path).replace("'", "''")
            escaped_output = str(parquet_path).replace("'", "''")
            
            # Read CSV with auto-detection (DuckDB handles .gz automatically)
            # Detect if file is gzipped
            is_gzipped = str(source_path).lower().endswith('.gz')
            compression_opt = ", compression='gzip'" if is_gzipped else ""
            
            # AUMC 使用 latin-1 编码（含有特殊字符如 °C）
            encoding_opt = ", encoding='latin-1'" if self.db_type == 'aumc' else ""
            
            # 优化参数：
            # - ZSTD压缩比snappy高30%，速度接近
            # - ROW_GROUP_SIZE=1000000 减少写入次数
            # - sample_size=100000 平衡速度和类型推断准确性（避免扫描80GB文件）
            con.execute(f"""
                COPY (
                    SELECT * FROM read_csv('{escaped_path}', 
                        auto_detect=true, 
                        header=true,
                        sample_size=100000,
                        ignore_errors=true,
                        null_padding=true,
                        all_varchar=false
                        {compression_opt}
                        {encoding_opt}
                    )
                ) TO '{escaped_output}' (
                    FORMAT PARQUET, 
                    COMPRESSION 'ZSTD',
                    ROW_GROUP_SIZE 1000000
                )
            """)
            
            # Get row count from parquet metadata (faster than COUNT(*))
            row_count = con.execute(f"""
                SELECT COUNT(*) FROM parquet_scan('{escaped_output}')
            """).fetchone()[0]
            
            con.close()
            
            result['status'] = 'success'
            result['row_count'] = row_count
            result['output_path'] = str(parquet_path)
            
            if self.verbose:
                logger.info(f"  ✅ Converted {csv_path.name}: {row_count:,} rows")
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            logger.error(f"  ❌ Failed to convert {csv_path.name}: {e}")
        
        finally:
            # Clean up temp file if created
            if temp_csv_path and temp_csv_path.exists():
                try:
                    temp_csv_path.unlink()
                except:
                    pass
            
            # Force garbage collection
            gc.collect()
        
        return result
    
    def convert_all(self, skip_existing: bool = True) -> List[Dict[str, Any]]:
        """Convert all CSV files in the data directory.
        
        Args:
            skip_existing: Skip files that already have parquet versions
            
        Returns:
            List of conversion results
        """
        csv_files = self._find_csv_files()
        
        if not csv_files:
            logger.warning(f"No CSV files found in {self.data_path}")
            return []
        
        if self.verbose:
            logger.info(f"Found {len(csv_files)} CSV files to process")
        
        results = []
        
        for csv_path in csv_files:
            parquet_path = self._get_parquet_path(csv_path)
            
            # Skip if parquet already exists
            if skip_existing and parquet_path.exists():
                if self.verbose:
                    logger.info(f"  ⏭️ Skipping {csv_path.name} (parquet exists)")
                results.append({
                    'file': csv_path.name,
                    'status': 'skipped',
                    'reason': 'parquet_exists',
                })
                continue
            
            # Convert the file
            result = self.convert_file(csv_path)
            results.append(result)
            
            # Force GC between files
            gc.collect()
        
        # Summary
        success = sum(1 for r in results if r['status'] == 'success')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        failed = sum(1 for r in results if r['status'] == 'error')
        
        if self.verbose:
            logger.info(f"\nConversion complete: {success} success, {skipped} skipped, {failed} failed")
        
        return results
    
    def convert_priority_tables(self) -> List[Dict[str, Any]]:
        """Convert only priority (large) tables for the detected database type."""
        priority_list = self.PRIORITY_TABLES.get(self.db_type, [])
        
        if not priority_list:
            logger.warning(f"No priority tables defined for {self.db_type}")
            return self.convert_all()
        
        csv_files = self._find_csv_files()
        priority_files = []
        
        for csv_path in csv_files:
            stem = csv_path.stem.lower()
            if stem.endswith('.csv'):
                stem = stem[:-4]
            
            if stem in priority_list:
                priority_files.append(csv_path)
        
        if not priority_files:
            logger.warning(f"No priority tables found in {self.data_path}")
            return []
        
        if self.verbose:
            logger.info(f"Converting {len(priority_files)} priority tables...")
        
        results = []
        for csv_path in priority_files:
            result = self.convert_file(csv_path)
            results.append(result)
            gc.collect()
        
        return results


def convert_with_duckdb(
    data_path: str,
    memory_limit_gb: float = 6.0,
    priority_only: bool = False,
    skip_existing: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """High-level function to convert CSV files using DuckDB.
    
    Args:
        data_path: Path to database directory
        memory_limit_gb: Maximum memory for DuckDB (default 6GB)
        priority_only: Only convert large/priority tables
        skip_existing: Skip files with existing parquet versions
        verbose: Print progress messages
        
    Returns:
        List of conversion results
    """
    converter = DuckDBConverter(
        data_path=data_path,
        memory_limit_gb=memory_limit_gb,
        verbose=verbose,
    )
    
    if priority_only:
        return converter.convert_priority_tables()
    else:
        return converter.convert_all(skip_existing=skip_existing)


if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='DuckDB-based CSV to Parquet converter')
    parser.add_argument('data_path', help='Path to database directory')
    parser.add_argument('--memory-limit', type=float, default=6.0, 
                       help='Memory limit in GB (default: 6)')
    parser.add_argument('--priority-only', action='store_true',
                       help='Only convert priority tables')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing parquet files')
    
    args = parser.parse_args()
    
    results = convert_with_duckdb(
        data_path=args.data_path,
        memory_limit_gb=args.memory_limit,
        priority_only=args.priority_only,
        skip_existing=not args.force,
    )
    
    print(f"\n{'='*60}")
    print("Conversion Results:")
    for r in results:
        status = r['status']
        if status == 'success':
            print(f"  ✅ {r['file']}: {r['row_count']:,} rows")
        elif status == 'skipped':
            print(f"  ⏭️ {r['file']}: {r.get('reason', 'skipped')}")
        else:
            print(f"  ❌ {r['file']}: {r.get('error', 'unknown error')}")
