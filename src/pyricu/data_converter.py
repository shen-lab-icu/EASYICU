"""
Data Converter Module for pyricu

This module provides utilities to convert CSV/CSV.GZ files to Parquet format
for faster data loading and reduced memory usage.

Usage:
    from pyricu.data_converter import DataConverter
    
    # Check and convert all tables for a database
    converter = DataConverter('/path/to/eicu/2.0.1')
    converter.ensure_parquet_ready()
    
    # Or use the CLI
    # python -m pyricu.data_converter /path/to/eicu/2.0.1
"""

from __future__ import annotations

import os
import gzip
import tarfile
import logging
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import pandas as pd

logger = logging.getLogger(__name__)


# Partitioning configuration for large tables (matching ricu's data-sources.json)
# Format: {database: {table_name: {"col": partition_column, "breaks": [breakpoints]}}}
PARTITIONING_CONFIG = {
    "eicu": {
        "nursecharting": {
            "col": "patientunitstayid",
            "breaks": [514528, 1037072, 1453997, 1775421, 2499831, 2937948, 3213286]
        },
        "vitalperiodic": {
            "col": "patientunitstayid",
            "breaks": [514528, 1037072, 1453997, 1775421, 2499831, 2937948, 3213286]
        },
    },
    "miiv": {
        "labevents": {
            "col": "itemid", 
            "breaks": [50868, 50902, 50943, 50983, 51146, 51248, 51256, 51279, 51491]
        },
        "poe": {
            "col": "subject_id",
            "breaks": [12017899, 13999829, 15979442, 17994364]
        },
        "chartevents": {
            "col": "itemid",
            "breaks": [220048, 220059, 220181, 220228, 220615, 223782, 223835, 223905, 223962, 223990, 
                       224015, 224055, 224082, 224093, 224328, 224650, 224701, 224850, 225072, 226104, 
                       227240, 227467, 227950, 227960, 228004, 228397, 228594, 228924, 229124]
        },
    },
    "aumc": {
        "listitems": {
            "col": "itemid",
            "breaks": [12290]
        },
        "numericitems": {
            "col": "itemid",
            "breaks": [6641, 6642, 6643, 6664, 6666, 6667, 6669, 6672, 6673, 6675, 6707, 6709, 
                       8874, 12270, 12275, 12278, 12281, 12286, 12303, 12561, 12576, 12804, 14841]
        },
    },
    "hirid": {
        "observations": {
            "col": "variableid",
            "breaks": [110, 120, 200, 210, 211, 300, 620, 2010, 2610, 3110, 4000, 5685, 15001565, 30005075]
        },
        "pharma": {
            "col": "pharmaid",
            "breaks": [431]
        },
    },
}


class ConversionStatus:
    """Tracks the conversion status of data files."""
    
    PENDING = "pending"
    CONVERTING = "converting"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # Already in parquet format


class DataConverter:
    """
    Converts CSV/CSV.GZ files to Parquet format for faster loading.
    
    Features:
    - Automatic detection of CSV/CSV.GZ files
    - Parallel conversion for multiple files
    - Integrity verification (row count, checksum)
    - Progress tracking and resumable conversion
    - Handles large files with chunked reading
    - ID-based partitioning matching ricu's logic
    - Memory-efficient streaming for ultra-large tables
    """
    
    # Default chunk size for reading large CSV files (rows)
    # 1M rows per chunk for high-performance conversion
    # Typical usage: ~3-5GB memory for large files
    DEFAULT_CHUNK_SIZE = 1_000_000
    
    # Status file name to track conversion progress
    STATUS_FILE = ".pyricu_conversion_status.json"
    
    # Common encodings to try
    ENCODINGS = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    # Memory threshold for buffer flush (in rows per partition)
    PARTITION_BUFFER_THRESHOLD = 2_000_000
    
    def __init__(
        self,
        data_path: str | Path,
        database: Optional[str] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        parallel_workers: int = 4,
        verbose: bool = True,
    ):
        """
        Initialize the data converter.
        
        Args:
            data_path: Path to the database directory containing CSV files
            database: Database type (auto-detected if None)
            chunk_size: Number of rows to read at a time for large files
            parallel_workers: Number of parallel conversion workers
            verbose: Enable verbose logging
        """
        self.data_path = Path(data_path)
        self.database = database or self._detect_database()
        self.chunk_size = chunk_size
        self.parallel_workers = parallel_workers
        self.verbose = verbose
        
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")
        
        self._status: Dict[str, Dict[str, Any]] = {}
        self._load_status()
    
    def _detect_database(self) -> str:
        """Detect database type from directory structure."""
        path_str = str(self.data_path).lower()
        
        if 'eicu' in path_str:
            return 'eicu'
        elif 'miiv' in path_str or 'mimic' in path_str:
            return 'miiv'
        elif 'aumc' in path_str:
            return 'aumc'
        elif 'hirid' in path_str:
            return 'hirid'
        
        # Try to detect from files
        files = list(self.data_path.glob('*.csv*'))
        file_names = [f.name.lower() for f in files]
        
        if any('patient.csv' in f for f in file_names):
            return 'eicu'
        elif any('admissions.csv' in f for f in file_names):
            return 'miiv'
        
        return 'unknown'
    
    def _extract_hirid_archives(self) -> List[str]:
        """
        Extract HiRID tar.gz archives if they exist.
        
        HiRID directory structure:
        - raw_stage/
            - observation_tables_csv.tar.gz OR observation_tables_parquet.tar.gz
            - pharma_records_csv.tar.gz OR pharma_records_parquet.tar.gz
        - reference_data.tar.gz
        
        After extraction, converts HiRID parquet shards (part-N.parquet) to
        numbered format (N.parquet) in the appropriate directories.
        
        Returns:
            List of extracted archive names
        """
        extracted = []
        
        # Check for HiRID-specific archives
        # Format: (archive_path, extraction_marker, is_parquet, target_dir)
        hirid_archives = [
            (self.data_path / 'reference_data.tar.gz', 'general_table.csv', False, None),
            (self.data_path / 'raw_stage' / 'observation_tables_parquet.tar.gz', 'observation_tables', True, 'observations'),
            (self.data_path / 'raw_stage' / 'observation_tables_csv.tar.gz', 'observation_tables', False, None),
            (self.data_path / 'raw_stage' / 'pharma_records_parquet.tar.gz', 'pharma_records', True, 'pharma'),
            (self.data_path / 'raw_stage' / 'pharma_records_csv.tar.gz', 'pharma_records', False, None),
        ]
        
        for archive_path, marker, is_parquet, ricu_target in hirid_archives:
            if not archive_path.exists():
                continue
            
            # Check if Parquet shards already exist (skip extraction)
            if ricu_target:
                ricu_dir = self.data_path / ricu_target
                if ricu_dir.is_dir() and list(ricu_dir.glob('[0-9]*.parquet')):
                    logger.info(f"Skipping {archive_path.name} - Parquet shards already exist in {ricu_target}/")
                    continue
                
            # Check if already extracted
            if is_parquet:
                marker_path = self.data_path / marker
            else:
                marker_path = self.data_path / marker if '.' not in marker else self.data_path / marker
            
            # Skip CSV if parquet version is already extracted
            if not is_parquet:
                parquet_marker = self.data_path / marker
                if parquet_marker.is_dir():
                    parquet_subdir = parquet_marker / 'parquet'
                    if parquet_subdir.is_dir() and any(parquet_subdir.glob('*.parquet')):
                        logger.info(f"Skipping {archive_path.name} - parquet version already extracted")
                        continue
            
            # Check if extraction is needed
            needs_extract = True
            if marker_path.exists():
                if marker_path.is_dir():
                    if any(marker_path.iterdir()):
                        needs_extract = False
                else:
                    needs_extract = False
            
            if needs_extract:
                logger.info(f"Extracting {archive_path.name}...")
                try:
                    with tarfile.open(archive_path, 'r:gz') as tar:
                        tar.extractall(path=self.data_path)
                    extracted.append(archive_path.name)
                    logger.info(f"Extracted {archive_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to extract {archive_path.name}: {e}")
                    continue
            
            # Convert HiRID parquet shards to ricu format if needed
            if is_parquet and ricu_target:
                self._convert_hirid_parquet_to_ricu(marker, ricu_target)
        
        return extracted
    
    def _convert_hirid_parquet_to_ricu(self, source_dir_name: str, ricu_target: str) -> None:
        """
        Convert HiRID parquet shards (part-N.parquet) to numbered format (N.parquet).
        
        HiRID original format: observation_tables/parquet/part-0.parquet, part-1.parquet, ...
        Target format: observations/1.parquet, 2.parquet, ...
        
        Args:
            source_dir_name: Source directory name (e.g., 'observation_tables')
            ricu_target: Target directory name (e.g., 'observations')
        """
        import shutil
        
        source_parquet_dir = self.data_path / source_dir_name / 'parquet'
        ricu_dir = self.data_path / ricu_target
        
        if not source_parquet_dir.is_dir():
            logger.debug(f"No parquet directory found at {source_parquet_dir}")
            return
        
        # Check if Parquet shards already exist
        if ricu_dir.is_dir() and list(ricu_dir.glob('[0-9]*.parquet')):
            logger.info(f"Parquet shards already exist in {ricu_target}/, skipping conversion")
            return
        
        # Find all part-N.parquet files
        part_files = sorted(source_parquet_dir.glob('part-*.parquet'), 
                           key=lambda f: int(f.stem.split('-')[1]))
        
        if not part_files:
            logger.debug(f"No part-*.parquet files found in {source_parquet_dir}")
            return
        
        logger.info(f"Converting {len(part_files)} HiRID parquet shards to {ricu_target}/")
        
        # Create target directory
        ricu_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy and rename files: part-0.parquet -> 1.parquet, part-1.parquet -> 2.parquet, etc.
        for part_file in part_files:
            # Extract part number (part-0 -> 0, part-1 -> 1, etc.)
            part_num = int(part_file.stem.split('-')[1])
            # ricu uses 1-based numbering
            ricu_num = part_num + 1
            target_file = ricu_dir / f"{ricu_num}.parquet"
            
            if not target_file.exists():
                shutil.copy2(part_file, target_file)
                logger.debug(f"Copied {part_file.name} -> {target_file.name}")
        
        logger.info(f"✅ Converted {len(part_files)} shards to {ricu_target}/")
    
    def _load_status(self) -> None:
        """Load conversion status from file."""
        status_file = self.data_path / self.STATUS_FILE
        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    self._status = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load status file: {e}")
                self._status = {}
    
    def _save_status(self) -> None:
        """Save conversion status to file."""
        status_file = self.data_path / self.STATUS_FILE
        try:
            with open(status_file, 'w') as f:
                json.dump(self._status, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save status file: {e}")
    
    def _get_csv_files(self) -> List[Path]:
        """Get all CSV/CSV.GZ files in the data directory (including subdirs).
        
        Deduplicates files with the same table name, preferring:
        1. Larger files (likely full data, not demo)
        2. Files closer to root directory
        
        Filters out:
        1. CSV shards inside directories that already have ricu parquet shards
        2. part-*.csv files that belong to already-converted observation tables
        3. cache directory files (pyricu internal cache files)
        4. demo directory files
        
        For HiRID: looks in raw_stage/ subdirectory if needed.
        """
        csv_files = []
        
        # Directories to exclude (case-insensitive)
        excluded_dirs = {'cache', 'demo', '__pycache__', '.git', 'imputed_stage', 'merged_stage'}
        
        # For HiRID, also search in raw_stage subdirectory
        search_paths = [self.data_path]
        if self.database == 'hirid':
            raw_stage = self.data_path / 'raw_stage'
            if raw_stage.is_dir():
                search_paths.append(raw_stage)
            # Also search in observation_tables and pharma_records
            for subdir in ['observation_tables', 'pharma_records']:
                subdir_path = self.data_path / subdir
                if subdir_path.is_dir():
                    search_paths.append(subdir_path)
        
        # Find all CSV and CSV.GZ files recursively
        for search_path in search_paths:
            for pattern in ['*.csv', '*.csv.gz', '*.CSV', '*.CSV.GZ']:
                for f in search_path.rglob(pattern):
                    # Check if any parent directory is in the excluded list
                    try:
                        parts_lower = [p.lower() for p in f.relative_to(self.data_path).parts[:-1]]
                    except ValueError:
                        parts_lower = [p.lower() for p in f.parts[:-1]]
                    if any(part in excluded_dirs for part in parts_lower):
                        continue
                    csv_files.append(f)
        
        # Filter out CSV shards that have already been converted to ricu parquet shards
        # These are typically in subdirectories like observation_tables/csv/part-*.csv
        # but ricu has already converted them to observations/*.parquet
        filtered_files = []
        
        # Patterns for CSV shard files that should be skipped if ricu parquet exists
        shard_patterns = {
            # HiRID: observation_tables/csv/part-*.csv -> observations/*.parquet
            'observation_tables': 'observations',
            'pharma_records': 'pharma',
            # AUMC: numericitems_split/num_*.csv -> numericitems/*.parquet
            'numericitems_split': 'numericitems',
            'listitems_split': 'listitems',
        }
        
        for f in csv_files:
            skip = False
            
            # Check if this is a shard CSV file (part-N.csv or num_NN.csv pattern)
            fname = f.name.lower()
            is_shard_csv = (
                fname.startswith('part-') or 
                fname.startswith('num_') or
                (fname[:2].isdigit() and fname.endswith('.csv'))
            )
            
            if is_shard_csv:
                # Check parent directory to see if ricu parquet shards exist
                parent_name = f.parent.name.lower()
                grandparent = f.parent.parent
                
                # Map CSV shard directory to ricu parquet shard directory
                for csv_dir, parquet_dir in shard_patterns.items():
                    if csv_dir in str(f.parent).lower():
                        # Check if ricu parquet shards exist
                        ricu_shard_dir = self.data_path / parquet_dir
                        if ricu_shard_dir.is_dir():
                            parquet_shards = list(ricu_shard_dir.glob('[0-9]*.parquet'))
                            if parquet_shards:
                                skip = True
                                break
                
                # Also check if parent directory name matches a known shard dir with parquet
                if not skip and parent_name in ['csv', 'data', 'numericitems_split', 'listitems_split']:
                    # Get the actual table directory (grandparent or parent)
                    table_dir_name = f.parent.name.lower()
                    # Common mappings
                    mappings = {
                        'observation_tables': 'observations',
                        'pharma_records': 'pharma',
                        'numericitems_split': 'numericitems',
                        'listitems_split': 'listitems',
                    }
                    target_dir = mappings.get(table_dir_name, table_dir_name)
                    ricu_shard_dir = self.data_path / target_dir
                    if ricu_shard_dir.is_dir() and ricu_shard_dir != grandparent:
                        parquet_shards = list(ricu_shard_dir.glob('[0-9]*.parquet'))
                        if parquet_shards:
                            skip = True
            
            if not skip:
                filtered_files.append(f)
        
        # Deduplicate by table name (keep largest file for each table)
        table_files: Dict[str, Path] = {}
        for f in filtered_files:
            table_name = self._get_table_name_from_path(f)
            if table_name not in table_files:
                table_files[table_name] = f
            else:
                # Prefer larger file (full data vs demo)
                existing = table_files[table_name]
                if f.stat().st_size > existing.stat().st_size:
                    table_files[table_name] = f
        
        # Get unique files and sort by size
        unique_files = list(table_files.values())
        unique_files.sort(key=lambda f: f.stat().st_size)
        
        return unique_files
    
    # ricu table name mappings (CSV name -> Parquet name)
    # Some databases use different names for CSV vs Parquet files
    RICU_TABLE_NAME_MAP = {
        # HiRID: original CSV uses _table suffix, ricu parquet doesn't
        'general_table': 'general',
        'pharma_records': 'pharma',
        'observation_tables': 'observations',
    }
    
    def _get_table_name_from_path(self, csv_path: Path) -> str:
        """Extract table name from CSV path (without extension)."""
        name = csv_path.name
        if name.endswith('.csv.gz'):
            name = name[:-7]
        elif name.endswith('.csv'):
            name = name[:-4]
        elif name.endswith('.CSV.GZ'):
            name = name[:-7]
        elif name.endswith('.CSV'):
            name = name[:-4]
        return name.lower()
    
    def _get_ricu_table_name(self, csv_path: Path) -> str:
        """Get the ricu-style table name (may differ from CSV name)."""
        csv_name = self._get_table_name_from_path(csv_path)
        return self.RICU_TABLE_NAME_MAP.get(csv_name, csv_name)
    
    def _get_parquet_path(self, csv_path: Path) -> Path:
        """Get the corresponding parquet path for a CSV file.
        
        ricu style: parquet files are in root directory, not preserving subdirectory structure.
        Uses ricu table name mapping for databases like HiRID.
        """
        name = self._get_ricu_table_name(csv_path)
        # ricu puts parquet files in root directory
        return self.data_path / f"{name}.parquet"
    
    def _get_parquet_path_with_subdir(self, csv_path: Path) -> Path:
        """Get parquet path preserving subdirectory structure (alternative location)."""
        name = self._get_ricu_table_name(csv_path)
        try:
            rel_parent = csv_path.parent.relative_to(self.data_path)
        except ValueError:
            rel_parent = Path()
        return self.data_path / rel_parent / f"{name}.parquet"
    
    def _get_shard_dir(self, csv_path: Path) -> Path:
        """Get the shard directory path for a large CSV file.
        
        ricu style: shard directories are in root directory (e.g., chartevents/1.parquet).
        Uses ricu table name mapping.
        """
        table_name = self._get_ricu_table_name(csv_path)
        # ricu puts shard directories in root directory
        return self.data_path / table_name
    
    def _get_shard_dir_with_subdir(self, csv_path: Path) -> Path:
        """Get shard directory preserving subdirectory structure (alternative location)."""
        table_name = self._get_ricu_table_name(csv_path)
        try:
            rel_parent = csv_path.parent.relative_to(self.data_path)
        except ValueError:
            rel_parent = Path()
        return self.data_path / rel_parent / table_name
    
    def _has_valid_shards(self, csv_path: Path) -> Tuple[bool, int]:
        """
        Check if valid sharded parquet files exist for a CSV.
        Checks both root directory, subdirectory locations, and bucket directories.
        
        Returns:
            (has_shards, shard_count)
        """
        table_name = self._get_ricu_table_name(csv_path)
        
        # Check for bucket directories first (e.g., nursecharting_bucket/bucket_id=*)
        bucket_dir = self.data_path / f"{table_name}_bucket"
        if bucket_dir.is_dir():
            bucket_subdirs = list(bucket_dir.glob("bucket_id=*"))
            if bucket_subdirs:
                # Count non-empty buckets
                non_empty = sum(1 for d in bucket_subdirs if list(d.glob("*.parquet")))
                if non_empty > 0:
                    return True, non_empty
        
        # Check both possible shard directory locations
        for shard_dir in [self._get_shard_dir(csv_path), self._get_shard_dir_with_subdir(csv_path)]:
            if not shard_dir.is_dir():
                continue
            
            # Count parquet shards (1.parquet, 2.parquet, etc.)
            shard_files = list(shard_dir.glob('[0-9]*.parquet'))
            if not shard_files:
                continue
            
            # Verify shards are numbered sequentially from 1
            shard_nums = sorted([int(f.stem) for f in shard_files if f.stem.isdigit()])
            if not shard_nums or shard_nums[0] != 1:
                continue
            
            # Check for gaps in sequence
            expected = list(range(1, len(shard_nums) + 1))
            if shard_nums != expected:
                continue
            
            return True, len(shard_nums)
        
        return False, 0
    
    def _is_conversion_needed(self, csv_path: Path) -> Tuple[bool, str]:
        """
        Check if conversion is needed for a CSV file.
        
        Handles both single parquet files and sharded directories.
        Checks both root directory and subdirectory locations.
        
        Returns:
            (needs_conversion, reason)
        """
        # First check for sharded directory (for large files)
        has_shards, shard_count = self._has_valid_shards(csv_path)
        if has_shards:
            # Check if CSV is newer than shards
            shard_dir = self._get_shard_dir(csv_path)
            if not shard_dir.is_dir():
                shard_dir = self._get_shard_dir_with_subdir(csv_path)
            
            csv_mtime = csv_path.stat().st_mtime
            
            # Check any shard file's mtime
            first_shard = shard_dir / "1.parquet"
            if first_shard.exists():
                shard_mtime = first_shard.stat().st_mtime
                if csv_mtime > shard_mtime:
                    return True, "CSV is newer than shards"
            
            return False, f"sharded ({shard_count} files)"
        
        # Check if single parquet file exists (check both locations)
        parquet_path = self._get_parquet_path(csv_path)
        parquet_path_subdir = self._get_parquet_path_with_subdir(csv_path)
        
        existing_parquet = None
        if parquet_path.exists():
            existing_parquet = parquet_path
        elif parquet_path_subdir.exists():
            existing_parquet = parquet_path_subdir
        
        if existing_parquet is None:
            return True, "parquet file does not exist"
        
        # Check if CSV is newer than parquet
        csv_mtime = csv_path.stat().st_mtime
        parquet_mtime = existing_parquet.stat().st_mtime
        
        if csv_mtime > parquet_mtime:
            return True, "CSV is newer than parquet"
        
        # Check status file for previous conversion
        file_key = csv_path.name
        if file_key in self._status:
            status = self._status[file_key]
            if status.get('status') == ConversionStatus.COMPLETED:
                # Verify row count using pyarrow metadata (no memory overhead)
                stored_rows = status.get('row_count', 0)
                try:
                    import pyarrow.parquet as pq_reader
                    # Only read metadata, not actual data
                    parquet_file = pq_reader.ParquetFile(parquet_path)
                    actual_rows = parquet_file.metadata.num_rows
                    if actual_rows == stored_rows:
                        return False, "already converted and verified"
                except Exception:
                    return True, "parquet file corrupted"
        
        return False, "parquet exists and is up to date"
    
    def _detect_encoding(self, csv_path: Path) -> str:
        """Detect the correct encoding for a CSV file.
        
        Uses a quick approach: read raw bytes and check for encoding errors,
        then verify with pandas on a sample.
        
        Special handling for known databases:
        - AUMC uses latin1 encoding (Dutch medical data)
        """
        is_gzipped = csv_path.name.endswith('.gz')
        
        # Check if this is AUMC database (uses latin1 encoding)
        path_lower = str(csv_path).lower()
        if 'aumc' in path_lower:
            # AUMC uses latin1 encoding - verify it works
            try:
                if is_gzipped:
                    pd.read_csv(csv_path, encoding='latin1', compression='gzip', nrows=100)
                else:
                    pd.read_csv(csv_path, encoding='latin1', nrows=100)
                return 'latin1'
            except:
                pass  # Fall through to normal detection
        
        # Quick byte-level check for non-gzipped files
        if not is_gzipped:
            # Read a sample of raw bytes from different parts of the file
            file_size = csv_path.stat().st_size
            samples = []
            with open(csv_path, 'rb') as f:
                # Read beginning
                samples.append(f.read(50000))
                # Read middle
                if file_size > 100000:
                    f.seek(file_size // 2)
                    samples.append(f.read(50000))
                # Read near end
                if file_size > 200000:
                    f.seek(max(0, file_size - 50000))
                    samples.append(f.read(50000))
            
            sample_bytes = b''.join(samples)
            
            for encoding in self.ENCODINGS:
                try:
                    sample_bytes.decode(encoding)
                    # Verify with pandas on first 1000 rows
                    try:
                        pd.read_csv(csv_path, encoding=encoding, nrows=1000)
                        return encoding
                    except:
                        continue
                except (UnicodeDecodeError, LookupError):
                    continue
        else:
            # For gzipped files, try reading samples with pandas
            # Use only 1000 rows to minimize memory
            for encoding in self.ENCODINGS:
                try:
                    sample_df = pd.read_csv(csv_path, encoding=encoding, compression='gzip', nrows=1000)
                    del sample_df
                    return encoding
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    if 'codec' in str(e).lower() or 'encode' in str(e).lower():
                        continue
                    return encoding  # Non-encoding error, use this encoding
        
        # Fallback to utf-8 with errors='replace'
        return 'utf-8-replace'
    
    def _read_csv_with_encoding(self, csv_path: Path, **kwargs) -> pd.DataFrame:
        """Read CSV file with automatic encoding detection.
        
        For chunked reading (chunksize in kwargs), detects encoding first
        to avoid errors during iteration.
        
        Uses optimized settings for memory-efficient reading.
        """
        is_gzipped = csv_path.name.endswith('.gz') or csv_path.name.endswith('.GZ')
        
        # Detect encoding first
        encoding = self._detect_encoding(csv_path)
        
        # Base read arguments - optimized for memory efficiency
        read_args = {
            'on_bad_lines': 'warn',  # Don't fail on bad lines
            'low_memory': True,  # Force low memory mode
        }
        read_args.update(kwargs)
        
        # Handle special utf-8-replace fallback
        if encoding == 'utf-8-replace':
            read_args['encoding'] = 'utf-8'
            read_args['encoding_errors'] = 'replace'
        else:
            read_args['encoding'] = encoding
        
        if is_gzipped:
            read_args['compression'] = 'gzip'
        
        return pd.read_csv(csv_path, **read_args)
    
    # Threshold for sharding large files (1GB compressed)
    # Only very large tables like chartevents need sharding
    SHARD_THRESHOLD_MB = 1000
    # Number of rows per shard - 25M rows to match ~200-250MB parquet files
    ROWS_PER_SHARD = 25_000_000  # 25M rows per shard
    
    # Known problematic columns that have mixed types
    # These columns often contain mixed numeric/string/bytes data
    MIXED_TYPE_COLUMNS = {
        # MIMIC-IV
        'pharmacy': ['lockout_interval', 'one_hr_max', 'doses_per_24_hrs', 
                     'duration', 'duration_interval', 'expiration_value'],
        'prescriptions': ['dose_val_rx', 'form_val_disp', 'doses_per_24_hrs'],
        'emar': ['dose_due', 'dose_given'],
        'emar_detail': ['dose_due', 'dose_given', 'completion_interval'],
        # eICU
        'infusiondrug': ['drugrate', 'infusionrate', 'drugamount', 'volumeoffluid'],
        'medication': ['dosage', 'loadingdose', 'frequency'],
        'respiratorycare': ['airwaysize', 'airwayposition', 'cuffpressure', 
                            'apneaparms', 'lowexhaledminvol', 'potentialblockvalve',
                            'lowexhaledtv', 'aboression', 'highpeakpress',
                            'lowpeakpress', 'exhaledmvtime', 'highexhaledmv'],
        'respiratorycharting': ['respchartvalue', 'respchartvaluelabel'],
        # AUMC - all object columns should be converted to string
        'admissions': ['destination', 'origin'],
        'drugitems': ['ordercategoryname', 'doserateunit', 'doseunitid', 'doserateunitid'],
        'freetextitems': ['value'],
        'listitems': ['value'],
        'numericitems': ['value', 'unit', 'registeredby'],
        'procedureorderitems': ['ordercategoryname'],
        'processitems': ['item'],
    }
    
    def _fix_mixed_type_columns(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """Fix columns with mixed types by converting to string.
        
        Some tables have columns with mixed bytes/float/string types
        that cause parquet conversion to fail.
        
        Args:
            df: DataFrame to fix
            filename: Original filename (for identifying known problematic columns)
            
        Returns:
            Fixed DataFrame
        """
        # Get table name from filename
        table_name = filename.lower()
        for ext in ['.csv.gz', '.csv']:
            if table_name.endswith(ext):
                table_name = table_name[:-len(ext)]
                break
        
        # Check for known problematic columns
        known_cols = self.MIXED_TYPE_COLUMNS.get(table_name, [])
        for col in known_cols:
            if col in df.columns:
                try:
                    # Convert to string, handling bytes and other types
                    df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) and not isinstance(x, str) else x)
                except Exception:
                    df[col] = df[col].astype(str)
        
        # Aggressively convert ALL object columns to string to avoid mixed type issues
        # This is safer for parquet export
        for col in df.select_dtypes(include=['object']).columns:
            if col not in known_cols:
                try:
                    # Check if column has any non-string values
                    sample = df[col].dropna().head(100)
                    has_non_string = False
                    if len(sample) > 0:
                        for val in sample:
                            if not isinstance(val, str):
                                has_non_string = True
                                break
                    
                    if has_non_string:
                        df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) and not isinstance(x, str) else x)
                except Exception:
                    # If any error, force convert to string
                    df[col] = df[col].astype(str)
        
        return df
    
    def _get_table_name(self, csv_path: Path) -> str:
        """Extract table name from CSV path (without extension)."""
        name = csv_path.name
        if name.endswith('.csv.gz'):
            name = name[:-7]
        elif name.endswith('.csv'):
            name = name[:-4]
        elif name.endswith('.CSV.GZ'):
            name = name[:-7]
        elif name.endswith('.CSV'):
            name = name[:-4]
        return name.lower()  # Normalize to lowercase like ricu
    
    def _should_shard(self, csv_path: Path) -> bool:
        """Determine if a file should be sharded.
        
        A file should be sharded if:
        1. It's defined in PARTITIONING_CONFIG for ID-based partitioning, OR
        2. The compressed file size exceeds SHARD_THRESHOLD_MB (for row-based partitioning)
        
        This ensures large files like emar.csv.gz (774MB) are also sharded.
        """
        table_name = self._get_table_name(csv_path)
        
        # Check if table has ID-based partitioning config
        partition_config = self._get_partitioning_config(table_name)
        if partition_config is not None:
            return True
        
        # Also shard large files even without partition config
        file_size_mb = csv_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.SHARD_THRESHOLD_MB:
            return True
        
        return False
    
    def _convert_file(self, csv_path: Path) -> Dict[str, Any]:
        """
        Convert a single CSV file to Parquet.
        
        For large files (>500MB compressed), creates a sharded directory structure
        like ricu: tablename/1.parquet, 2.parquet, etc.
        
        Returns:
            Conversion result dictionary
        """
        file_key = csv_path.name
        table_name = self._get_table_name(csv_path)
        
        result = {
            'file': file_key,
            'table': table_name,
            'status': ConversionStatus.PENDING,
            'row_count': 0,
            'shards': 0,
            'error': None,
        }
        
        try:
            # Update status
            result['status'] = ConversionStatus.CONVERTING
            self._status[file_key] = result.copy()
            self._save_status()
            
            file_size_mb = csv_path.stat().st_size / (1024 * 1024)
            should_shard = self._should_shard(csv_path)
            
            if self.verbose:
                shard_note = " (will be sharded)" if should_shard else ""
                logger.info(f"Converting {file_key} ({file_size_mb:.1f} MB){shard_note}...")
            
            if should_shard:
                # Sharded conversion for large files
                result = self._convert_file_sharded(csv_path, result)
            else:
                # Single file conversion for smaller files
                result = self._convert_file_single(csv_path, result)
            
        except Exception as e:
            result['status'] = ConversionStatus.FAILED
            result['error'] = str(e)
            logger.error(f"  ❌ Failed to convert {file_key}: {e}")
        
        # Update and save status
        self._status[file_key] = result
        self._save_status()
        
        return result
    
    def _convert_file_single(self, csv_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a smaller CSV file to a single parquet file.
        
        Uses streaming write for large files to avoid memory issues.
        Handles mixed-type columns by converting them to string before parquet export.
        """
        import gc
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        parquet_path = self._get_parquet_path(csv_path)
        
        # Check file size to decide on streaming vs direct write
        file_size = csv_path.stat().st_size
        use_streaming = file_size > 50 * 1024 * 1024  # > 50MB use streaming
        
        if use_streaming:
            # Use streaming write for larger files
            total_rows = 0
            writer = None
            chunk_iter = None
            reference_schema = None
            
            # Pre-infer stable schema for files with potential type issues
            file_size_mb = file_size / (1024 * 1024)
            if file_size_mb > 100:
                try:
                    reference_schema = self._infer_stable_schema(csv_path, sample_chunks=3)
                    if self.verbose:
                        logger.info(f"  Pre-inferred stable schema with {len(reference_schema)} columns")
                except Exception as e:
                    logger.warning(f"  Failed to pre-infer schema: {e}")
            
            try:
                chunk_iter = self._read_csv_with_encoding(
                    csv_path, 
                    chunksize=self.chunk_size,
                    low_memory=True,
                )
                
                for i, chunk in enumerate(chunk_iter):
                    # Fix mixed-type columns in each chunk
                    chunk = self._fix_mixed_type_columns(chunk, csv_path.name)
                    
                    # Convert to PyArrow table
                    table = pa.Table.from_pandas(chunk, preserve_index=False)
                    
                    # Initialize writer on first chunk
                    if writer is None:
                        if reference_schema is None:
                            reference_schema = table.schema
                        writer = pq.ParquetWriter(parquet_path, reference_schema, compression='snappy')
                    
                    # Normalize schema if different from reference
                    if table.schema != reference_schema:
                        table = self._normalize_schema(table, reference_schema)
                    
                    writer.write_table(table)
                    total_rows += len(chunk)
                    
                    if self.verbose and (i + 1) % 50 == 0:
                        logger.info(f"  Written {total_rows:,} rows...")
                    
                    del chunk, table
                    # GC less frequently for better performance
                    if (i + 1) % 100 == 0:
                        gc.collect()
                
                if writer is not None:
                    writer.close()
                    
            except Exception as e:
                if writer is not None:
                    writer.close()
                raise
            finally:
                # Clean up iterator
                if chunk_iter is not None:
                    if hasattr(chunk_iter, 'close'):
                        chunk_iter.close()
                    del chunk_iter
                gc.collect()
            
            result['row_count'] = total_rows
            
        else:
            # Read entire file at once for small files
            df = self._read_csv_with_encoding(csv_path, low_memory=True)
            
            # Fix mixed-type columns before parquet export
            df = self._fix_mixed_type_columns(df, csv_path.name)
            
            # Convert to parquet with error handling
            try:
                df.to_parquet(parquet_path, index=False, engine='pyarrow')
            except Exception as e:
                if 'Expected bytes' in str(e) or 'object' in str(e).lower():
                    # Convert all object columns to string
                    logger.warning(f"  ⚠️ Converting object columns to string for {csv_path.name}")
                    for col in df.select_dtypes(include=['object']).columns:
                        df[col] = df[col].astype(str)
                    df.to_parquet(parquet_path, index=False, engine='pyarrow')
                else:
                    raise
            
            result['row_count'] = len(df)
        
        result['status'] = ConversionStatus.COMPLETED
        result['shards'] = 0
        
        if self.verbose:
            logger.info(f"  ✅ Converted {result['file']}: {result['row_count']:,} rows")
        
        return result
    
    def _get_partitioning_config(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get partitioning configuration for a table from ricu config."""
        db_config = PARTITIONING_CONFIG.get(self.database, {})
        return db_config.get(table_name.lower())
    
    def _assign_partition(self, value, breaks: List) -> int:
        """
        Assign a value to a partition based on breakpoints.
        
        Matches ricu's logic: partition 1 for values <= breaks[0],
        partition 2 for breaks[0] < value <= breaks[1], etc.
        """
        import bisect
        # bisect_right returns the insertion point, which is 0-indexed
        # We add 1 to get 1-indexed partition numbers
        return bisect.bisect_right(breaks, value) + 1
    
    def _convert_file_sharded(self, csv_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a large CSV file to sharded parquet files in a directory.
        
        Uses ricu's ID-based partitioning logic:
        - If partitioning config exists: partition by ID column using breakpoints
        - Otherwise: partition by row count
        
        Creates: tablename/1.parquet, 2.parquet, etc. (like ricu)
        Preserves subdirectory structure (e.g., icu/chartevents/).
        """
        table_name = self._get_table_name(csv_path)
        shard_dir = self._get_shard_dir(csv_path)  # Uses preserved subdirectory structure
        
        # Create shard directory
        shard_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for ricu-style partitioning config
        partition_config = self._get_partitioning_config(table_name)
        
        # ID-based partitioning matches ricu's output format exactly.
        # Now that memory is not an issue, enable by default for tables with config.
        # Set PYRICU_USE_ID_PARTITIONING=0 to disable if needed.
        use_id_partitioning = os.environ.get('PYRICU_USE_ID_PARTITIONING', '1') == '1'
        
        if partition_config and use_id_partitioning:
            # Use ID-based partitioning (ricu style) - opens many writers, uses more memory
            if self.verbose:
                logger.info(f"  Using ID-based partitioning (may use more memory)")
            result = self._convert_with_id_partitioning(
                csv_path, shard_dir, partition_config, result
            )
        else:
            # Use row-count based partitioning - one writer at a time, memory efficient
            result = self._convert_with_row_partitioning(
                csv_path, shard_dir, result
            )
        
        return result
    
    def _convert_with_id_partitioning(
        self, 
        csv_path: Path, 
        shard_dir: Path, 
        partition_config: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert using ID-based partitioning (matching ricu's logic).
        
        Partitions data based on breakpoints in a specific column.
        Uses memory-efficient streaming - writes directly to partition files
        without accumulating data in memory.
        """
        import gc
        import bisect
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        partition_col = partition_config['col']
        breaks = partition_config['breaks']
        if not isinstance(breaks, list):
            breaks = [breaks]
        
        n_partitions = len(breaks) + 1
        
        if self.verbose:
            logger.info(f"  Using ID-based partitioning on '{partition_col}' with {n_partitions} partitions (streaming mode)")
        
        # Use PyArrow ParquetWriter for each partition - true streaming write
        partition_writers: Dict[int, pq.ParquetWriter] = {}
        partition_total_rows: Dict[int, int] = {i: 0 for i in range(1, n_partitions + 1)}
        
        total_rows = 0
        reference_schema = None
        
        # Pre-infer stable schema for large files
        file_size_mb = csv_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:
            try:
                # Use only 3 chunks to minimize memory overhead
                reference_schema = self._infer_stable_schema(csv_path, sample_chunks=3)
                if self.verbose:
                    logger.info(f"  Pre-inferred stable schema with {len(reference_schema)} columns")
                # Force GC after schema inference
                gc.collect()
            except Exception as e:
                logger.warning(f"  Failed to pre-infer schema: {e}")
        
        def get_partition_path(part_num: int) -> Path:
            return shard_dir / f"{part_num}.parquet"
        
        def write_to_partition(part_num: int, df: pd.DataFrame):
            """Write DataFrame directly to partition file using streaming."""
            nonlocal reference_schema
            
            if len(df) == 0:
                return
            
            # Convert to PyArrow table
            table = pa.Table.from_pandas(df, preserve_index=False)
            
            # Use first table's schema as reference if not pre-inferred
            if reference_schema is None:
                reference_schema = table.schema
            
            # Normalize schema if different
            if table.schema != reference_schema:
                table = self._normalize_schema(table, reference_schema)
            
            # Initialize writer if needed
            if part_num not in partition_writers:
                shard_path = get_partition_path(part_num)
                partition_writers[part_num] = pq.ParquetWriter(
                    shard_path, 
                    reference_schema,
                    compression='snappy'
                )
            
            # Write the batch
            partition_writers[part_num].write_table(table)
            partition_total_rows[part_num] += len(df)
        
        def close_all_writers():
            """Close all partition writers."""
            for part_num, writer in partition_writers.items():
                try:
                    writer.close()
                    if self.verbose:
                        logger.info(f"  📁 Wrote partition {part_num}: {partition_total_rows[part_num]:,} rows")
                except Exception as e:
                    logger.warning(f"  ⚠️ Error closing partition {part_num}: {e}")
        
        chunk_iter = None
        try:
            # Read in chunks and stream to partitions
            chunk_iter = self._read_csv_with_encoding(
                csv_path, 
                chunksize=self.chunk_size,
                low_memory=True,
            )
            
            for chunk_num, chunk in enumerate(chunk_iter):
                chunk_rows = len(chunk)
                total_rows += chunk_rows
                
                if self.verbose and (chunk_num + 1) % 50 == 0:
                    logger.info(f"  Read {total_rows:,} rows...")
                
                # Fix mixed-type columns before conversion
                chunk = self._fix_mixed_type_columns(chunk, csv_path.name)
                
                # Assign each row to a partition
                if partition_col not in chunk.columns:
                    logger.warning(f"  Partition column '{partition_col}' not found, using row-based partitioning")
                    close_all_writers()
                    # Clean up partial files
                    for part_num in range(1, n_partitions + 1):
                        try:
                            get_partition_path(part_num).unlink()
                        except:
                            pass
                    return self._convert_with_row_partitioning(csv_path, shard_dir, result)
                
                # Vectorized partition assignment
                col_values = chunk[partition_col].values
                chunk['_partition'] = [bisect.bisect_right(breaks, v) + 1 for v in col_values]
                
                # Write each partition's data directly to file
                for part_num in range(1, n_partitions + 1):
                    part_chunk = chunk[chunk['_partition'] == part_num].drop(columns=['_partition'])
                    if len(part_chunk) > 0:
                        write_to_partition(part_num, part_chunk)
                        del part_chunk
                
                # Clear chunk reference and collect garbage more frequently on large files
                del chunk
                if (chunk_num + 1) % 20 == 0:
                    gc.collect()
            
            # Close all writers
            close_all_writers()
            gc.collect()
            
            result['status'] = ConversionStatus.COMPLETED
            result['row_count'] = total_rows
            result['shards'] = n_partitions
            result['shard_dir'] = str(shard_dir)
            result['partition_col'] = partition_col
            result['partition_breaks'] = breaks
            
            if self.verbose:
                logger.info(f"  ✅ Converted {result['file']}: {total_rows:,} rows in {n_partitions} partitions")
            
        except Exception as e:
            # Make sure to close writers on error
            close_all_writers()
            raise
        finally:
            # Clean up iterator to release file handles and memory
            if chunk_iter is not None:
                if hasattr(chunk_iter, 'close'):
                    chunk_iter.close()
                del chunk_iter
            gc.collect()
        
        return result
    
    def _normalize_schema(self, table: 'pa.Table', reference_schema: 'pa.Schema') -> 'pa.Table':
        """
        Normalize a PyArrow table to match a reference schema.
        
        Handles cases where chunks may have different inferred types (e.g., null vs string).
        Casts columns to match the reference schema.
        """
        import pyarrow as pa
        
        new_columns = []
        for i, field in enumerate(reference_schema):
            col_name = field.name
            ref_type = field.type
            table_field = table.schema.field(col_name)
            table_type = table_field.type
            
            col = table.column(col_name)
            
            # If types differ, we need to cast
            if table_type != ref_type:
                # Handle null type -> actual type casting
                if pa.types.is_null(table_type):
                    # Create array of nulls with correct type
                    null_array = pa.nulls(len(col), type=ref_type)
                    new_columns.append(null_array)
                elif pa.types.is_null(ref_type):
                    # Reference was null but we now have real type - use string
                    try:
                        new_columns.append(col.cast(pa.string(), safe=False))
                    except:
                        new_columns.append(col)
                else:
                    # Try to cast to reference type
                    try:
                        new_columns.append(col.cast(ref_type, safe=False))
                    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
                        # If cast fails, convert both to string
                        if pa.types.is_string(ref_type) or pa.types.is_large_string(ref_type):
                            try:
                                new_columns.append(col.cast(pa.string(), safe=False))
                            except:
                                # Last resort: convert via Python
                                arr = col.to_pylist()
                                str_arr = [str(x) if x is not None else None for x in arr]
                                new_columns.append(pa.array(str_arr, type=pa.string()))
                        else:
                            # For numeric types that fail, try string
                            try:
                                new_columns.append(col.cast(pa.string(), safe=False))
                            except:
                                new_columns.append(col)
            else:
                new_columns.append(col)
        
        return pa.Table.from_arrays(new_columns, schema=reference_schema)
    
    def _infer_stable_schema(self, csv_path: Path, sample_chunks: int = 3) -> 'pa.Schema':
        """
        Infer a stable schema by reading multiple chunks and merging types.
        
        This prevents schema mismatch errors when some chunks have null columns.
        Uses minimal memory by only keeping schema objects, not data.
        """
        import gc
        import pyarrow as pa
        
        schemas = []
        chunk_iter = self._read_csv_with_encoding(
            csv_path,
            chunksize=self.chunk_size,
            low_memory=True,
        )
        
        try:
            # Read first few chunks to build stable schema
            for i, chunk in enumerate(chunk_iter):
                if i >= sample_chunks:
                    break
                chunk = self._fix_mixed_type_columns(chunk, csv_path.name)
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                schemas.append(table.schema)
                # Explicitly delete to free memory immediately
                del chunk
                del table
        finally:
            # Close the iterator to release file handle and buffers
            if hasattr(chunk_iter, 'close'):
                chunk_iter.close()
            del chunk_iter
            gc.collect()
        
        if not schemas:
            raise ValueError(f"No data to infer schema from {csv_path}")
        
        # Merge schemas - prefer string types for maximum compatibility
        merged_fields = []
        for i, field in enumerate(schemas[0]):
            best_type = field.type
            for schema in schemas[1:]:
                other_type = schema.field(i).type
                # Prefer non-null type
                if pa.types.is_null(best_type) and not pa.types.is_null(other_type):
                    best_type = other_type
                # Handle type conflicts - prefer string for safety
                elif not pa.types.is_null(other_type) and other_type != best_type:
                    # If either is string, use string (most flexible)
                    if pa.types.is_string(other_type) or pa.types.is_large_string(other_type):
                        best_type = pa.string()
                    elif pa.types.is_string(best_type) or pa.types.is_large_string(best_type):
                        best_type = pa.string()
                    # If one is double and one is int, use double
                    elif pa.types.is_floating(other_type) and pa.types.is_integer(best_type):
                        best_type = other_type
                    elif pa.types.is_floating(best_type) and pa.types.is_integer(other_type):
                        pass  # keep best_type (floating)
                    else:
                        # For any other conflict, use string as safest option
                        best_type = pa.string()
            merged_fields.append(pa.field(field.name, best_type))
        
        return pa.schema(merged_fields)
    
    def _convert_with_row_partitioning(
        self,
        csv_path: Path,
        shard_dir: Path,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert using row-count based partitioning (fallback method).
        Uses streaming write to avoid memory accumulation.
        """
        import gc
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        total_rows = 0
        shard_num = 1
        current_writer = None
        current_shard_rows = 0
        reference_schema = None
        
        # Infer stable schema upfront for large files
        file_size_mb = csv_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:  # For large files, pre-scan for stable schema
            try:
                # Use only 3 chunks to minimize memory overhead
                reference_schema = self._infer_stable_schema(csv_path, sample_chunks=3)
                if self.verbose:
                    logger.info(f"  Pre-inferred stable schema with {len(reference_schema)} columns")
                # Force GC after schema inference
                gc.collect()
            except Exception as e:
                logger.warning(f"  Failed to pre-infer schema: {e}")
        
        def start_new_shard():
            nonlocal shard_num, current_writer, current_shard_rows
            if current_writer is not None:
                current_writer.close()
                if self.verbose:
                    logger.info(f"  📁 Wrote shard {shard_num}: {current_shard_rows:,} rows")
                shard_num += 1
            current_shard_rows = 0
            current_writer = None  # Will be initialized on first write
        
        chunk_iter = None
        try:
            # Read and write in chunks
            chunk_iter = self._read_csv_with_encoding(
                csv_path, 
                chunksize=self.chunk_size,
                low_memory=True,
            )
            
            for chunk in chunk_iter:
                chunk_len = len(chunk)
                total_rows += chunk_len
                
                # Fix mixed-type columns before conversion
                chunk = self._fix_mixed_type_columns(chunk, csv_path.name)
                
                # Convert to PyArrow table
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                
                # Initialize writer if needed (use reference schema if available)
                if current_writer is None:
                    if reference_schema is None:
                        reference_schema = table.schema
                    shard_path = shard_dir / f"{shard_num}.parquet"
                    current_writer = pq.ParquetWriter(shard_path, reference_schema, compression='snappy')
                
                # Normalize table to match reference schema
                if table.schema != reference_schema:
                    table = self._normalize_schema(table, reference_schema)
                
                current_writer.write_table(table)
                
                current_shard_rows += chunk_len
                del chunk, table
                
                # Start new shard when reaching threshold
                if current_shard_rows >= self.ROWS_PER_SHARD:
                    start_new_shard()
                    gc.collect()  # GC after closing each shard
                
                # Periodic garbage collection - more frequent (every 500K rows)
                if total_rows % 500_000 == 0:
                    gc.collect()
        
        finally:
            # Clean up iterator to release file handles and memory
            if chunk_iter is not None:
                if hasattr(chunk_iter, 'close'):
                    chunk_iter.close()
                del chunk_iter
            gc.collect()
        
        # Close final shard
        if current_writer is not None:
            current_writer.close()
            if self.verbose:
                logger.info(f"  📁 Wrote shard {shard_num}: {current_shard_rows:,} rows")
        
        result['status'] = ConversionStatus.COMPLETED
        result['row_count'] = total_rows
        result['shards'] = shard_num
        result['shard_dir'] = str(shard_dir)
        
        if self.verbose:
            logger.info(f"  ✅ Converted {result['file']}: {total_rows:,} rows in {shard_num} shards")
        
        return result
    
    def get_conversion_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current conversion status for all files.
        
        Returns:
            Dictionary mapping file names to their status
        """
        status = {}
        csv_files = self._get_csv_files()
        
        for csv_path in csv_files:
            file_key = csv_path.name
            needs_conversion, reason = self._is_conversion_needed(csv_path)
            
            if not needs_conversion:
                status[file_key] = {
                    'status': ConversionStatus.SKIPPED,
                    'reason': reason,
                }
            elif file_key in self._status:
                status[file_key] = self._status[file_key]
            else:
                status[file_key] = {
                    'status': ConversionStatus.PENDING,
                    'reason': reason,
                }
        
        return status
    
    def convert_all(self, force: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Convert all CSV files to Parquet.
        
        Args:
            force: Force reconversion even if parquet exists
            
        Returns:
            Dictionary of conversion results
        """
        csv_files = self._get_csv_files()
        
        if not csv_files:
            if self.verbose:
                logger.info(f"No CSV files found in {self.data_path}")
            return {}
        
        # Filter files that need conversion
        files_to_convert = []
        for csv_path in csv_files:
            if force:
                files_to_convert.append(csv_path)
            else:
                needs_conversion, reason = self._is_conversion_needed(csv_path)
                if needs_conversion:
                    files_to_convert.append(csv_path)
        
        if not files_to_convert:
            if self.verbose:
                logger.info(f"All {len(csv_files)} files are already converted")
            return self.get_conversion_status()
        
        if self.verbose:
            logger.info(f"Converting {len(files_to_convert)} of {len(csv_files)} files...")
        
        results = {}
        
        # Use parallel conversion for multiple files
        if len(files_to_convert) > 1 and self.parallel_workers > 1:
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                future_map = {
                    executor.submit(self._convert_file, csv_path): csv_path
                    for csv_path in files_to_convert
                }
                
                for future in as_completed(future_map):
                    csv_path = future_map[future]
                    try:
                        result = future.result()
                        results[csv_path.name] = result
                    except Exception as e:
                        results[csv_path.name] = {
                            'status': ConversionStatus.FAILED,
                            'error': str(e),
                        }
        else:
            # Sequential conversion
            for csv_path in files_to_convert:
                result = self._convert_file(csv_path)
                results[csv_path.name] = result
        
        return results
    
    def is_ready(self) -> Tuple[bool, List[str]]:
        """
        Check if all data files are ready (converted to parquet).
        
        Returns:
            (is_ready, list of missing/failed files)
        """
        csv_files = self._get_csv_files()
        missing_or_failed = []
        
        for csv_path in csv_files:
            needs_conversion, reason = self._is_conversion_needed(csv_path)
            if needs_conversion:
                missing_or_failed.append(f"{csv_path.name}: {reason}")
        
        return len(missing_or_failed) == 0, missing_or_failed
    
    def ensure_parquet_ready(self, auto_convert: bool = True) -> bool:
        """
        Ensure all parquet files are ready for loading.
        
        Args:
            auto_convert: Automatically convert missing files
            
        Returns:
            True if all files are ready, False otherwise
        """
        # For HiRID, extract tar.gz archives first
        if self.database == 'hirid':
            try:
                extracted = self._extract_hirid_archives()
                if extracted and self.verbose:
                    logger.info(f"📦 Extracted HiRID archives: {', '.join(extracted)}")
            except Exception as e:
                logger.warning(f"HiRID archive extraction failed: {e}")
        
        is_ready, missing = self.is_ready()
        
        if is_ready:
            if self.verbose:
                logger.info(f"✅ All data files are ready in {self.data_path}")
            return True
        
        if not auto_convert:
            logger.warning(f"❌ {len(missing)} files need conversion:")
            for msg in missing[:10]:
                logger.warning(f"  - {msg}")
            if len(missing) > 10:
                logger.warning(f"  ... and {len(missing) - 10} more")
            return False
        
        # Auto-convert
        if self.verbose:
            logger.info(f"🔄 Converting {len(missing)} files to parquet...")
        
        results = self.convert_all()
        
        # Check results
        failed = [name for name, r in results.items() if r.get('status') == ConversionStatus.FAILED]
        
        if failed:
            logger.error(f"❌ {len(failed)} files failed to convert:")
            for name in failed[:5]:
                error = results[name].get('error', 'Unknown error')
                logger.error(f"  - {name}: {error}")
            return False
        
        if self.verbose:
            logger.info(f"✅ Successfully converted all files")
        
        return True
    
    def get_table_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all tables (files) in the database.
        
        Returns:
            Dictionary with table information
        """
        info = {}
        
        # Check parquet files
        for pq_path in self.data_path.glob('*.parquet'):
            name = pq_path.stem
            try:
                # Read just the metadata
                df = pd.read_parquet(pq_path)
                info[name] = {
                    'format': 'parquet',
                    'path': str(pq_path),
                    'rows': len(df),
                    'columns': list(df.columns),
                    'size_mb': pq_path.stat().st_size / (1024 * 1024),
                }
            except Exception as e:
                info[name] = {
                    'format': 'parquet',
                    'path': str(pq_path),
                    'error': str(e),
                }
        
        # Check for unconverted CSV files
        for csv_path in self._get_csv_files():
            name = csv_path.stem
            if name.endswith('.csv'):
                name = name[:-4]
            
            if name not in info:
                info[name] = {
                    'format': 'csv',
                    'path': str(csv_path),
                    'size_mb': csv_path.stat().st_size / (1024 * 1024),
                    'needs_conversion': True,
                }
        
        return info


def ensure_database_ready(
    data_path: str | Path,
    database: Optional[str] = None,
    auto_convert: bool = True,
    verbose: bool = True,
) -> bool:
    """
    Convenience function to ensure a database is ready for use.
    
    Args:
        data_path: Path to the database directory
        database: Database type (auto-detected if None)
        auto_convert: Automatically convert CSV files to parquet
        verbose: Enable verbose logging
        
    Returns:
        True if database is ready, False otherwise
    """
    converter = DataConverter(
        data_path=data_path,
        database=database,
        verbose=verbose,
    )
    return converter.ensure_parquet_ready(auto_convert=auto_convert)


def main():
    """CLI entry point for data conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert ICU database CSV files to Parquet format"
    )
    parser.add_argument(
        "data_path",
        help="Path to the database directory containing CSV files",
    )
    parser.add_argument(
        "-d", "--database",
        help="Database type (auto-detected if not specified)",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force reconversion even if parquet exists",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of parallel conversion workers (default: 4)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show conversion status without converting",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show table information",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )
    
    try:
        converter = DataConverter(
            data_path=args.data_path,
            database=args.database,
            parallel_workers=args.workers,
            verbose=not args.quiet,
        )
        
        if args.status:
            print(f"\n📊 Conversion Status for {args.data_path}")
            print("=" * 60)
            status = converter.get_conversion_status()
            for name, info in sorted(status.items()):
                status_str = info.get('status', 'unknown')
                if status_str == ConversionStatus.SKIPPED:
                    print(f"  ✅ {name}: already converted")
                elif status_str == ConversionStatus.COMPLETED:
                    print(f"  ✅ {name}: converted ({info.get('row_count', 0):,} rows)")
                elif status_str == ConversionStatus.FAILED:
                    print(f"  ❌ {name}: failed - {info.get('error', 'unknown error')}")
                else:
                    print(f"  ⏳ {name}: pending ({info.get('reason', '')})")
            return
        
        if args.info:
            print(f"\n📋 Table Information for {args.data_path}")
            print("=" * 60)
            info = converter.get_table_info()
            for name, table_info in sorted(info.items()):
                fmt = table_info.get('format', 'unknown')
                size = table_info.get('size_mb', 0)
                rows = table_info.get('rows', 'N/A')
                cols = len(table_info.get('columns', []))
                print(f"  {name}")
                print(f"    Format: {fmt}, Size: {size:.1f} MB, Rows: {rows}, Columns: {cols}")
            return
        
        # Perform conversion
        print(f"\n🔄 Converting database: {args.data_path}")
        print(f"   Database type: {converter.database}")
        print("=" * 60)
        
        results = converter.convert_all(force=args.force)
        
        # Summary
        completed = sum(1 for r in results.values() if r.get('status') == ConversionStatus.COMPLETED)
        failed = sum(1 for r in results.values() if r.get('status') == ConversionStatus.FAILED)
        
        print("\n" + "=" * 60)
        print(f"✅ Completed: {completed}")
        if failed:
            print(f"❌ Failed: {failed}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
