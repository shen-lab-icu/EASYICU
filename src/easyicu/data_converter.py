"""
Data Converter Module for easyicu

This module provides utilities to convert CSV/CSV.GZ files to Parquet format
for faster data loading and reduced memory usage.

Usage:
    from easyicu.data_converter import DataConverter
    
    # Check and convert all tables for a database
    converter = DataConverter('/path/to/eicu/2.0.1')
    converter.ensure_parquet_ready()
    
    # Or use the CLI
    # python -m easyicu.data_converter /path/to/eicu/2.0.1
"""

from __future__ import annotations

import os
import tarfile
import logging
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import pyarrow

logger = logging.getLogger(__name__)


def _is_hidden_sidecar(path: Path) -> bool:
    """Return True for macOS AppleDouble or other hidden sidecar files."""
    name = path.name
    return name.startswith("._") or name.startswith(".")


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


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
    - Memory-efficient DuckDB-based conversion (primary method)
    - Fallback to pandas for problematic files
    - Integrity verification (row count, checksum)
    - Progress tracking and resumable conversion
    - ID-based partitioning matching ricu's logic
    - Memory limit: designed for 12GB RAM systems
    """
    
    # Default chunk size for reading large CSV files (rows)
    # 50K rows per chunk for memory efficiency
    # Typical usage: ~200-400MB memory per chunk
    DEFAULT_CHUNK_SIZE = 50_000
    
    # Status file name to track conversion progress
    STATUS_FILE = ".easyicu_conversion_status.json"
    CONVERSION_MANIFEST_FILE = "conversion_manifest.json"
    
    # Common encodings to try
    ENCODINGS = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']

    # pandas.read_csv default NA strings — used by the Arrow CSV path so its
    # string-column null handling matches the legacy pandas converter.
    _PANDAS_NA_VALUES = [
        '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
        '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None',
        'n/a', 'nan', 'null',
    ]
    
    # Memory threshold for buffer flush (in rows per partition)
    PARTITION_BUFFER_THRESHOLD = 500_000
    
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

        # Parquet output codec. zstd produces ~20-25% smaller files than
        # snappy at similar speed; on a slow mount fewer bytes means faster
        # conversion *writes* AND faster downstream extraction *reads*.
        # Override with EASYICU_PARQUET_COMPRESSION (e.g. 'snappy').
        self.parquet_compression = os.environ.get(
            'EASYICU_PARQUET_COMPRESSION', 'zstd'
        ).lower()

        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")
        
        self._status: Dict[str, Dict[str, Any]] = {}
        # convert_all() runs _convert_file() on a ThreadPoolExecutor; this
        # reentrant lock serialises mutations of _status and the JSON write
        # so concurrent workers cannot corrupt .easyicu_conversion_status.json
        # or trip json.dump's "dictionary changed size during iteration".
        self._status_lock = threading.RLock()
        self._load_status()

        # 🚀 perf A1/A2: cache bucket-shard presence at init so
        # `_has_valid_shards` does not re-walk `<table>_bucket/bucket_id=*/*.parquet`
        # for every CSV file. On slow filesystems (macfuse / network mounts)
        # the per-call recursive readdir over 100 bucket sub-dirs dominated
        # status checks; one shot at startup is two orders of magnitude
        # cheaper. Lazy-init: defer the scan until first lookup.
        self._bucket_dir_cache: Optional[Dict[str, int]] = None
        self._shard_dir_cache: Optional[Dict[str, int]] = None

    def _scan_bucket_dirs(self) -> Dict[str, int]:
        """One-shot scan of `<table>_bucket` directories. Returns
        ``{table_name: non_empty_bucket_count}``. Cached.
        """
        if self._bucket_dir_cache is not None:
            return self._bucket_dir_cache
        cache: Dict[str, int] = {}
        try:
            for entry in self.data_path.iterdir():
                if not entry.is_dir() or not entry.name.endswith("_bucket"):
                    continue
                table_name = entry.name[: -len("_bucket")]
                try:
                    bucket_subdirs = [d for d in entry.iterdir() if d.is_dir() and d.name.startswith("bucket_id=")]
                except OSError:
                    continue
                non_empty = 0
                for d in bucket_subdirs:
                    try:
                        # 🚀 perf A1/A2 (footer guard): require at least one
                        # bucket file to pass the parquet-magic check, not
                        # just exist. Same rationale as the shard-dir scan
                        # above — partial / truncated files must not be
                        # accepted as "already converted".
                        parquet_files = [p for p in d.iterdir() if p.suffix == ".parquet"]
                        if any(self._has_parquet_footer(p) for p in parquet_files):
                            non_empty += 1
                    except OSError:
                        pass
                if non_empty > 0:
                    cache[table_name] = non_empty
        except OSError:
            pass
        self._bucket_dir_cache = cache
        return cache

    def _scan_shard_dirs(self) -> Dict[str, int]:
        """One-shot scan of `<table>/N.parquet` shard directories. Returns
        ``{table_name: sequential_shard_count}``. Cached.

        🚀 perf A1/A2 (footer guard): we also verify each shard's parquet
        magic bytes at the very end of the file (4 bytes ``PAR1``). A
        previous conversion killed mid-write left ``vitalperiodic/1.parquet``
        present-but-truncated; the original scan only checked filename
        existence and the corrupt shard slipped through as "converted",
        which silently broke the entire reconvert flow. The footer
        check is one ``open + seek + read(8)`` per file — negligible.
        """
        if self._shard_dir_cache is not None:
            return self._shard_dir_cache
        cache: Dict[str, int] = {}
        try:
            for entry in self.data_path.iterdir():
                if not entry.is_dir():
                    continue
                try:
                    shard_paths: Dict[int, Path] = {}
                    for p in entry.iterdir():
                        if p.suffix == ".parquet" and p.stem.isdigit():
                            shard_paths[int(p.stem)] = p
                    if not shard_paths:
                        continue
                    shard_nums = sorted(shard_paths.keys())
                    if shard_nums[0] != 1 or shard_nums != list(range(1, len(shard_nums) + 1)):
                        continue
                    if not all(self._has_parquet_footer(shard_paths[n]) for n in shard_nums):
                        # Treat as not-converted so the caller redrives
                        # the conversion instead of trusting truncated shards.
                        continue
                    cache[entry.name] = len(shard_nums)
                except OSError:
                    continue
        except OSError:
            pass
        self._shard_dir_cache = cache
        return cache

    @staticmethod
    def _has_parquet_footer(path: Path) -> bool:
        """Cheap parquet integrity check — last 4 bytes must be ``PAR1``.

        Avoids a full pyarrow metadata read; one syscall per shard.
        Catches truncated files from killed converters / disconnected mounts.
        """
        try:
            sz = path.stat().st_size
            if sz < 8:
                return False
            with path.open("rb") as f:
                f.seek(-4, 2)
                return f.read(4) == b"PAR1"
        except OSError:
            return False

    def _invalidate_dir_caches(self) -> None:
        """Drop bucket/shard dir caches. Call after writing new shards/buckets."""
        self._bucket_dir_cache = None
        self._shard_dir_cache = None

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
        
        for archive_path, marker, is_parquet, target_dir_name in hirid_archives:
            if not archive_path.exists():
                continue
            
            # Check if Parquet shards already exist (skip extraction)
            if target_dir_name:
                target_dir = self.data_path / target_dir_name
                if target_dir.is_dir() and list(target_dir.glob('[0-9]*.parquet')):
                    logger.info(f"Skipping {archive_path.name} - Parquet shards already exist in {target_dir_name}/")
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
            if is_parquet and target_dir_name:
                self._convert_hirid_parquet(marker, target_dir_name)
        
        return extracted
    
    def _convert_hirid_parquet(self, source_dir_name: str, target_dir_name: str) -> None:
        """
        Convert HiRID parquet shards (part-N.parquet) to numbered format (N.parquet).
        
        HiRID original format: observation_tables/parquet/part-0.parquet, part-1.parquet, ...
        Target format: observations/1.parquet, 2.parquet, ...
        
        Args:
            source_dir_name: Source directory name (e.g., 'observation_tables')
            target_dir_name: Target directory name (e.g., 'observations')
        """
        import shutil
        
        source_parquet_dir = self.data_path / source_dir_name / 'parquet'
        target_dir = self.data_path / target_dir_name
        
        if not source_parquet_dir.is_dir():
            logger.debug(f"No parquet directory found at {source_parquet_dir}")
            return
        
        # Check if Parquet shards already exist
        if target_dir.is_dir() and list(target_dir.glob('[0-9]*.parquet')):
            logger.info(f"Parquet shards already exist in {target_dir_name}/, skipping conversion")
            return
        
        # Find all part-N.parquet files
        part_files = sorted(source_parquet_dir.glob('part-*.parquet'), 
                           key=lambda f: int(f.stem.split('-')[1]))
        
        if not part_files:
            logger.debug(f"No part-*.parquet files found in {source_parquet_dir}")
            return
        
        logger.info(f"Converting {len(part_files)} HiRID parquet shards to {target_dir_name}/")
        
        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy and rename files: part-0.parquet -> 1.parquet, part-1.parquet -> 2.parquet, etc.
        for part_file in part_files:
            # Extract part number (part-0 -> 0, part-1 -> 1, etc.)
            part_num = int(part_file.stem.split('-')[1])
            # ricu uses 1-based numbering
            shard_num = part_num + 1
            target_file = target_dir / f"{shard_num}.parquet"
            
            if not target_file.exists():
                shutil.copy2(part_file, target_file)
                logger.debug(f"Copied {part_file.name} -> {target_file.name}")
        
        logger.info(f"✅ Converted {len(part_files)} shards to {target_dir_name}/")
    
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
        """Persist conversion status atomically.

        Called concurrently from ThreadPoolExecutor workers. The lock keeps
        json serialisation from iterating ``self._status`` while another
        worker mutates it; the temp-file + ``os.replace`` ensures the on-disk
        file is always a complete JSON document even if the process dies
        mid-write.
        """
        status_file = self.data_path / self.STATUS_FILE
        try:
            with self._status_lock:
                payload = json.dumps(self._status, indent=2)
            tmp_file = status_file.with_name(
                f"{status_file.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            with open(tmp_file, 'w') as f:
                f.write(payload)
            os.replace(tmp_file, status_file)
        except Exception as e:
            logger.warning(f"Could not save status file: {e}")

    def _record_status(self, file_key: str, result: Dict[str, Any]) -> None:
        """Atomically record one file's status and flush to disk.

        Holds the lock across both the dict mutation and the serialise step
        so a concurrent worker can never observe (or persist) a torn state.
        """
        with self._status_lock:
            self._status[file_key] = result
            self._save_status()
    
    def _get_csv_files(self) -> List[Path]:
        """Get all CSV/CSV.GZ files in the data directory (including subdirs).
        
        Deduplicates files with the same table name, preferring:
        1. Larger files (likely full data, not demo)
        2. Files closer to root directory
        
        Filters out:
        1. CSV shards inside directories that already have ricu parquet shards
        2. part-*.csv files that belong to already-converted observation tables
        3. cache directory files (easyicu internal cache files)
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
        
        # 🚀 perf A2 (extension): os.walk with directory pruning so we
        # never descend into `<table>_bucket/` or sharded shard dirs
        # full of parquet files. The previous implementation called
        # `rglob` 4 times (one per CSV pattern) over the entire tree,
        # touching 10k+ parquet entries in chartevents_bucket purely to
        # discard them — the dominant cost of startup on slow mounts and
        # the reason `mimic-iii "All N files are already converted"`
        # printed instantly but the process kept running for minutes.
        csv_suffixes = ('.csv', '.csv.gz')
        for search_path in search_paths:
            for root, dirs, files in os.walk(search_path, topdown=True):
                # Prune excluded subtrees in place (os.walk respects this
                # only when topdown=True).
                pruned: List[str] = []
                for d in dirs:
                    dl = d.lower()
                    if dl in excluded_dirs:
                        continue
                    # Skip `<table>_bucket/` shard layouts entirely —
                    # they only ever contain parquet, not CSV.
                    if dl.endswith('_bucket'):
                        continue
                    pruned.append(d)
                dirs[:] = pruned

                for name in files:
                    lower = name.lower()
                    if not lower.endswith(csv_suffixes):
                        continue
                    f = Path(root) / name
                    if _is_hidden_sidecar(f):
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
                        shard_dir = self.data_path / parquet_dir
                        if shard_dir.is_dir():
                            parquet_shards = list(shard_dir.glob('[0-9]*.parquet'))
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
                    shard_dir = self.data_path / target_dir
                    if shard_dir.is_dir() and shard_dir != grandparent:
                        parquet_shards = list(shard_dir.glob('[0-9]*.parquet'))
                        if parquet_shards:
                            skip = True
            
            if not skip:
                filtered_files.append(f)
        
        # 🚀 perf A3: stat each file ONCE — the original code called
        # ``f.stat().st_size`` three times per file (dedup compare + final
        # sort), which on slow filesystems is the dominant cost of file
        # enumeration. Read size once, carry it on a tuple.
        sized: List[Tuple[Path, int]] = []
        for f in filtered_files:
            try:
                size = f.stat().st_size
            except OSError:
                size = 0
            sized.append((f, size))

        table_files: Dict[str, Tuple[Path, int]] = {}
        for f, size in sized:
            table_name = self._get_table_name_from_path(f)
            existing = table_files.get(table_name)
            if existing is None or size > existing[1]:
                table_files[table_name] = (f, size)

        unique = list(table_files.values())
        unique.sort(key=lambda pair: pair[1])
        return [p for p, _ in unique]
    
    # ricu table name mappings (CSV name -> Parquet name)
    # Some databases use different names for CSV vs Parquet files
    TABLE_NAME_MAP = {
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
    
    def _get_table_name(self, csv_path: Path) -> str:
        """Get the ricu-style table name (may differ from CSV name)."""
        csv_name = self._get_table_name_from_path(csv_path)
        return self.TABLE_NAME_MAP.get(csv_name, csv_name)
    
    def _get_parquet_path(self, csv_path: Path) -> Path:
        """Get the corresponding parquet path for a CSV file.
        
        ricu style: parquet files are in root directory, not preserving subdirectory structure.
        Uses ricu table name mapping for databases like HiRID.
        """
        name = self._get_table_name(csv_path)
        # ricu puts parquet files in root directory
        return self.data_path / f"{name}.parquet"
    
    def _get_parquet_path_with_subdir(self, csv_path: Path) -> Path:
        """Get parquet path preserving subdirectory structure (alternative location)."""
        name = self._get_table_name(csv_path)
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
        table_name = self._get_table_name(csv_path)
        # ricu puts shard directories in root directory
        return self.data_path / table_name
    
    def _get_shard_dir_with_subdir(self, csv_path: Path) -> Path:
        """Get shard directory preserving subdirectory structure (alternative location)."""
        table_name = self._get_table_name(csv_path)
        try:
            rel_parent = csv_path.parent.relative_to(self.data_path)
        except ValueError:
            rel_parent = Path()
        return self.data_path / rel_parent / table_name
    
    def _has_valid_shards(self, csv_path: Path) -> Tuple[bool, int]:
        """
        Check if valid sharded parquet files exist for a CSV.

        🚀 perf A1/A2: uses cached one-shot scans (`_scan_bucket_dirs`,
        `_scan_shard_dirs`) so this method is O(1) per call after the
        first invocation, instead of recursively walking the bucket tree
        for every CSV (which was the dominant cost on slow filesystems —
        the cause of the mimic-iii startup hang on macfuse).

        Returns:
            (has_shards, shard_count)
        """
        table_name = self._get_table_name(csv_path)

        # Bucket dirs (e.g., chartevents_bucket/bucket_id=*/*.parquet) win first.
        bucket_cache = self._scan_bucket_dirs()
        if table_name in bucket_cache:
            return True, bucket_cache[table_name]

        # Sequential shards (e.g., vitalPeriodic/1.parquet ... N.parquet).
        # Check both the table-name directory and the same-name-with-subdir
        # path used by `_get_shard_dir_with_subdir`.
        shard_cache = self._scan_shard_dirs()
        if table_name in shard_cache:
            return True, shard_cache[table_name]

        # Fall back to the slow per-call paths only for the subdir variant
        # (e.g., icu/chartevents/) which the cache does not enumerate.
        subdir_shard_dir = self._get_shard_dir_with_subdir(csv_path)
        if subdir_shard_dir.is_dir():
            try:
                shard_nums: List[int] = []
                for p in subdir_shard_dir.iterdir():
                    if p.suffix == ".parquet" and p.stem.isdigit():
                        shard_nums.append(int(p.stem))
            except OSError:
                shard_nums = []
            if shard_nums:
                shard_nums.sort()
                if shard_nums[0] == 1 and shard_nums == list(range(1, len(shard_nums) + 1)):
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
            except Exception:
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
                    except Exception:
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

        🚀 perf Y: when the encoding is utf-8 (the common case) and the
        caller is doing a streaming chunked read, dispatch to the
        ``pyarrow.csv`` streaming reader instead of ``pandas.read_csv``.
        PyArrow's CSV parser is C++-vectorised and benchmarked ~3× faster
        on representative eicu/mimic files (see
        ``scripts/bench_csv_reader.py``). The returned iterator yields
        pandas DataFrames so downstream code (`_fix_mixed_type_columns`,
        partition assignment, parquet write) is unchanged.

        Set ``EASYICU_CSV_READER=pandas`` to force the legacy path.
        """
        is_gzipped = csv_path.name.endswith('.gz') or csv_path.name.endswith('.GZ')

        # Detect encoding first
        encoding = self._detect_encoding(csv_path)

        # 🚀 perf Y (rolled back to opt-in): the pyarrow CSV streaming
        # reader is 3× faster in isolation (see
        # ``scripts/bench_csv_reader.py``) but during real conversion the
        # downstream sort_values / groupby / parquet write are pandas
        # operations, so every chunk pays an extra Arrow→pandas
        # conversion. On eicu vitalPeriodic this made conversion ~5×
        # SLOWER end-to-end than the pandas reader. Keeping the code
        # path available behind the env flag so a future refactor can
        # stay in pyarrow end-to-end (then it would be a real win).
        chunksize = kwargs.get('chunksize')
        prefer_arrow = (
            chunksize is not None
            and os.environ.get('EASYICU_CSV_READER', 'pandas').lower() == 'pyarrow'
            and encoding in ('utf-8', 'utf-8-replace', 'ascii')
        )
        if prefer_arrow:
            try:
                return self._read_csv_arrow_chunks(csv_path, int(chunksize), is_gzipped)
            except Exception as exc:
                logger.info(
                    "  ⚠️ pyarrow CSV reader failed on %s (%s), falling back to pandas",
                    csv_path.name, type(exc).__name__,
                )

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

    def _read_csv_arrow_chunks(self, csv_path: Path, chunksize: int, is_gzipped: bool):
        """Yield pandas DataFrames of ~``chunksize`` rows each via
        pyarrow.csv streaming reader.

        PyArrow's reader delivers RecordBatches sized by ``block_size``
        bytes (~10 MB by default), which usually contains many more
        rows than our preferred chunksize. We re-batch into accumulator
        slices roughly matching the caller's chunksize so downstream
        partition assignment + write_table stays in a memory-friendly
        regime.
        """
        import pyarrow.csv as pa_csv
        import pyarrow as pa

        # Aim for ~10 MB parse blocks; pyarrow defaults are similar but
        # pinning here keeps row counts predictable.
        block_size = max(8 * 1024 * 1024, chunksize * 200)
        read_opts = pa_csv.ReadOptions(block_size=block_size)
        parse_opts = pa_csv.ParseOptions(invalid_row_handler=lambda row: 'skip')
        convert_opts = pa_csv.ConvertOptions(strings_can_be_null=True)

        def _iterator():
            with pa_csv.open_csv(
                str(csv_path),
                read_options=read_opts,
                parse_options=parse_opts,
                convert_options=convert_opts,
            ) as reader:
                accumulator: list = []
                accumulated_rows = 0
                for batch in reader:
                    accumulator.append(batch)
                    accumulated_rows += batch.num_rows
                    if accumulated_rows >= chunksize:
                        tbl = pa.Table.from_batches(accumulator)
                        yield tbl.to_pandas()
                        accumulator.clear()
                        accumulated_rows = 0
                if accumulator:
                    tbl = pa.Table.from_batches(accumulator)
                    yield tbl.to_pandas()

        return _iterator()
    
    # Threshold for sharding large files (50MB compressed for memory safety)
    # Reduced from 1GB to prevent OOM on typical systems
    SHARD_THRESHOLD_MB = 50
    # Number of rows per shard - 5M rows to match ~50-100MB parquet files
    ROWS_PER_SHARD = 5_000_000  # 5M rows per shard (reduced from 25M)
    
    # Known problematic columns that have mixed types
    # These columns often contain mixed numeric/string/bytes data
    MIXED_TYPE_COLUMNS = {
        # MIMIC-III / MIMIC-IV — chartevents.value mixes numeric vitals with
        # GCS-component text ('Spontaneously' / 'Oriented'); type inference
        # that picks a numeric type silently drops every text row, losing
        # 6+ neurological concepts. Pin it to string.
        'chartevents': ['value'],
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
            self._record_status(file_key, result.copy())
            
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
        self._record_status(file_key, result)
        
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
                        writer = pq.ParquetWriter(parquet_path, reference_schema, compression=self.parquet_compression)
                    
                    # Normalize schema if different from reference
                    if table.schema != reference_schema:
                        table = self._normalize_schema(table, reference_schema)
                    
                    writer.write_table(table)
                    total_rows += len(chunk)
                    
                    if self.verbose and (i + 1) % 20 == 0:
                        logger.info(f"  Written {total_rows:,} rows...")
                    
                    # Aggressive memory cleanup for Windows
                    del chunk, table
                    # GC every 10 chunks for memory safety
                    if (i + 1) % 10 == 0:
                        gc.collect()
                
                if writer is not None:
                    writer.close()
                    
            except Exception:
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
                df.to_parquet(parquet_path, index=False, engine='pyarrow', compression=self.parquet_compression)
            except Exception as e:
                if 'Expected bytes' in str(e) or 'object' in str(e).lower():
                    # Convert all object columns to string
                    logger.warning(f"  ⚠️ Converting object columns to string for {csv_path.name}")
                    for col in df.select_dtypes(include=['object']).columns:
                        df[col] = df[col].astype(str)
                    df.to_parquet(parquet_path, index=False, engine='pyarrow', compression=self.parquet_compression)
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
        # Set EASYICU_USE_ID_PARTITIONING=0 to disable if needed.
        use_id_partitioning = os.environ.get('EASYICU_USE_ID_PARTITIONING', '1') == '1'
        
        if partition_config and use_id_partitioning:
            # Use ID-based partitioning (ricu style) - opens many writers, uses more memory
            if self.verbose:
                logger.info("  Using ID-based partitioning (may use more memory)")
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
        """Dispatch ID-partitioned conversion: end-to-end Arrow, pandas fallback.

        The Arrow path keeps the whole pipeline (CSV parse → partition assign →
        parquet write) in pyarrow, avoiding the per-chunk Arrow↔pandas round
        trips that made the earlier naive pyarrow reader ~5× slower end-to-end.
        On any failure (non-UTF8, unexpected schema drift, etc.) it falls back
        to the proven pandas implementation.
        """
        if self._arrow_csv_enabled(csv_path):
            try:
                return self._convert_with_id_partitioning_arrow(
                    csv_path, shard_dir, partition_config, result
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "  ⚠️ Arrow id-partition path failed for %s (%s: %s); "
                    "falling back to pandas",
                    csv_path.name, type(exc).__name__, exc,
                )
                self._wipe_shard_dir(shard_dir)
        return self._convert_with_id_partitioning_pandas(
            csv_path, shard_dir, partition_config, result
        )

    def _convert_with_id_partitioning_pandas(
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
                    compression=self.parquet_compression
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
                
                if self.verbose and (chunk_num + 1) % 10 == 0:
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
                        except Exception:
                            pass
                    return self._convert_with_row_partitioning(csv_path, shard_dir, result)
                
                # 🚀 perf A5: actually-vectorized partition assignment.
                # The original code claimed vectorization but ran a Python
                # list comprehension over `col_values`; np.searchsorted is
                # C-vectorized and 10-100× faster on million-row chunks.
                col_values = chunk[partition_col].values
                _np_breaks = np.asarray(breaks)
                chunk['_partition'] = np.searchsorted(_np_breaks, col_values, side='right') + 1

                # 🚀 perf Z (eicu wide-table cohort speedup): sort each chunk
                # by partition_col before writing. Each parquet row group
                # then carries a narrow [min, max] zone-map on partition_col,
                # so DuckDB can skip row groups that don't intersect a
                # cohort filter (`patientunitstayid IN (...)`). On a typical
                # 200-patient cohort this prunes ~70-95% of row groups
                # within each shard — much bigger win than re-partitioning.
                # Cost: pandas sort_values on a 1M-row chunk is sub-second.
                chunk = chunk.sort_values(
                    [partition_col], kind='mergesort'
                )

                # 🚀 perf A6: single groupby pass instead of N boolean
                # masks + N drops. Each `chunk[mask].drop(...)` allocated
                # a new DataFrame; on 5M-row chunks with 8 partitions that
                # was 8× the necessary memory churn.
                for part_num, part_chunk in chunk.groupby('_partition', sort=False):
                    if len(part_chunk) == 0:
                        continue
                    part_chunk = part_chunk.drop(columns=['_partition'])
                    write_to_partition(int(part_num), part_chunk)
                    del part_chunk

                # 🚀 perf A7: drop the per-chunk `gc.collect()`. CPython
                # already reclaims when refcounts hit zero (the `del chunk`
                # below is sufficient); calling gc every chunk inserts a
                # 20–100 ms STW pause that compounded on macfuse runs.
                del chunk
            
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
            
        except Exception:
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
    
    def _normalize_schema(self, table: "pyarrow.Table", reference_schema: "pyarrow.Schema") -> "pyarrow.Table":
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
                    except Exception:
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
                            except Exception:
                                # Last resort: convert via Python
                                arr = col.to_pylist()
                                str_arr = [str(x) if x is not None else None for x in arr]
                                new_columns.append(pa.array(str_arr, type=pa.string()))
                        else:
                            # For numeric types that fail, try string
                            try:
                                new_columns.append(col.cast(pa.string(), safe=False))
                            except Exception:
                                new_columns.append(col)
            else:
                new_columns.append(col)
        
        return pa.Table.from_arrays(new_columns, schema=reference_schema)
    
    def _infer_stable_schema(self, csv_path: Path, sample_chunks: int = 3) -> "pyarrow.Schema":
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
        """Dispatch row-count sharded conversion: Arrow path, pandas fallback."""
        if self._arrow_csv_enabled(csv_path):
            try:
                return self._convert_with_row_partitioning_arrow(
                    csv_path, shard_dir, result
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "  ⚠️ Arrow row-shard path failed for %s (%s: %s); "
                    "falling back to pandas",
                    csv_path.name, type(exc).__name__, exc,
                )
                self._wipe_shard_dir(shard_dir)
        return self._convert_with_row_partitioning_pandas(csv_path, shard_dir, result)

    def _convert_with_row_partitioning_pandas(
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
                    current_writer = pq.ParquetWriter(shard_path, reference_schema, compression=self.parquet_compression)
                
                # Normalize table to match reference schema
                if table.schema != reference_schema:
                    table = self._normalize_schema(table, reference_schema)
                
                current_writer.write_table(table)
                
                current_shard_rows += chunk_len
                del chunk, table

                # Start new shard when reaching threshold
                if current_shard_rows >= self.ROWS_PER_SHARD:
                    start_new_shard()
                    # 🚀 perf A7: keep one gc.collect at shard boundaries
                    # (cheap, infrequent) but drop the per-chunk call below.
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

    # ------------------------------------------------------------------
    # End-to-end Arrow conversion path
    #
    # The legacy pandas path parses CSV with pandas.read_csv and round-trips
    # every chunk Arrow→pandas→Arrow. On gzip-compressed databases (eicu,
    # mimic-iv) CSV parsing dominates and pandas is ~3× slower than
    # pyarrow.csv. Keeping the whole pipeline in pyarrow removes both the
    # slow parser and the round trips. The pandas methods above remain as a
    # robustness fallback for non-UTF8 files or unexpected schema drift.
    # ------------------------------------------------------------------

    def _arrow_csv_enabled(self, csv_path: Path) -> bool:
        """Whether the end-to-end Arrow CSV path may be used for this file."""
        if os.environ.get('EASYICU_CSV_READER', '').lower() == 'pandas':
            return False
        try:
            enc = self._detect_encoding(csv_path)
        except Exception:  # noqa: BLE001
            return False
        # pyarrow.csv assumes UTF-8/ASCII; other encodings go to pandas.
        return enc in ('utf-8', 'ascii', 'utf-8-replace')

    def _wipe_shard_dir(self, shard_dir: Path) -> None:
        """Delete parquet shards left behind by a failed Arrow attempt."""
        try:
            for p in shard_dir.glob('*.parquet'):
                try:
                    p.unlink()
                except OSError:
                    pass
        except Exception:  # noqa: BLE001
            pass

    def _open_arrow_csv(self, csv_path: Path, table_name: str):
        """Open a pyarrow streaming CSV reader (.gz handled transparently).

        Known mixed-type columns are pinned to string up front via
        ConvertOptions.column_types, which replaces the pandas-era
        ``_fix_mixed_type_columns`` post-processing.
        """
        import pyarrow as pa
        import pyarrow.csv as pa_csv

        col_types = {
            col: pa.string()
            for col in self.MIXED_TYPE_COLUMNS.get(table_name, [])
        }
        read_opts = pa_csv.ReadOptions(block_size=16 * 1024 * 1024)
        parse_opts = pa_csv.ParseOptions(invalid_row_handler=lambda row: 'skip')
        convert_opts = pa_csv.ConvertOptions(
            strings_can_be_null=True,
            column_types=col_types or None,
            # Match pandas.read_csv's default NA strings so string columns
            # get the same null treatment as the legacy pandas converter
            # (otherwise values like 'NA'/'None'/'null' stay as text and
            # diverge from existing prepared parquet).
            null_values=self._PANDAS_NA_VALUES,
        )
        return pa_csv.open_csv(
            str(csv_path),
            read_options=read_opts,
            parse_options=parse_opts,
            convert_options=convert_opts,
        )

    def _threaded_batch_iter(self, reader, queue_size: int = 4):
        """Yield record batches from *reader* via a background producer thread.

        The CSV read/decompress/parse (``read_next_batch``) and the parquet
        encode/compress/write the caller does both release the GIL in
        pyarrow's C++ layer, so running the reader on its own thread overlaps
        the two phases — conversion wall time drops toward ``max(read, write)``
        instead of ``read + write``.
        """
        import threading
        import queue as _queue

        q: "_queue.Queue" = _queue.Queue(maxsize=queue_size)
        sentinel = object()
        err: list = []
        stop = threading.Event()

        def _produce():
            try:
                while not stop.is_set():
                    try:
                        batch = reader.read_next_batch()
                    except StopIteration:
                        break
                    # timed put so an early consumer exit can't deadlock us
                    while not stop.is_set():
                        try:
                            q.put(batch, timeout=0.5)
                            break
                        except _queue.Full:
                            continue
            except Exception as exc:  # noqa: BLE001
                err.append(exc)
            finally:
                try:
                    q.put(sentinel, timeout=0.5)
                except _queue.Full:
                    pass

        thread = threading.Thread(target=_produce, daemon=True)
        thread.start()
        try:
            while True:
                batch = q.get()
                if batch is sentinel:
                    break
                yield batch
        finally:
            stop.set()
            # unblock a producer parked on a full queue
            try:
                while True:
                    q.get_nowait()
            except _queue.Empty:
                pass
            thread.join(timeout=10)
        if err:
            raise err[0]

    def _convert_with_id_partitioning_arrow(
        self,
        csv_path: Path,
        shard_dir: Path,
        partition_config: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """ID-partitioned conversion kept entirely in pyarrow."""
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        partition_col = partition_config['col']
        breaks = partition_config['breaks']
        if not isinstance(breaks, list):
            breaks = [breaks]
        n_partitions = len(breaks) + 1
        table_name = self._get_table_name(csv_path)

        if self.verbose:
            logger.info(
                f"  Using ID-based partitioning on '{partition_col}' with "
                f"{n_partitions} partitions (Arrow streaming)"
            )

        reader = self._open_arrow_csv(csv_path, table_name)
        schema = reader.schema
        if partition_col not in schema.names:
            try:
                reader.close()
            except Exception:  # noqa: BLE001
                pass
            raise ValueError(
                f"partition column '{partition_col}' not in CSV header"
            )

        writers: Dict[int, "pq.ParquetWriter"] = {}
        part_rows: Dict[int, int] = {i: 0 for i in range(1, n_partitions + 1)}
        total_rows = 0
        # ~1M-row working tables: big enough to amortise per-table overhead,
        # small enough that 8 partition slices stay well within memory.
        batch_target = 1_000_000
        pending: list = []
        pending_rows = 0

        def flush(tbl: "pa.Table") -> None:
            nonlocal total_rows
            if tbl.num_rows == 0:
                return
            col = tbl.column(partition_col)
            part_idx = None
            for b in breaks:
                ge = pc.cast(pc.greater_equal(col, b), pa.int32())
                part_idx = ge if part_idx is None else pc.add(part_idx, ge)
            part_idx = pc.add(part_idx, pa.scalar(1, pa.int32()))
            # Null ids (should not occur for an ID column) -> last partition,
            # so rows are never silently dropped.
            part_idx = pc.fill_null(part_idx, pa.scalar(n_partitions, pa.int32()))
            tagged = tbl.append_column('__part', part_idx)
            for p in range(1, n_partitions + 1):
                sub = tagged.filter(pc.equal(tagged.column('__part'), p))
                if sub.num_rows == 0:
                    continue
                # Drop helper col, then sort by the partition column so each
                # parquet row group carries a narrow zone-map (cohort filters
                # can skip row groups) — parity with the pandas path.
                sub = sub.drop(['__part']).sort_by(partition_col)
                if p not in writers:
                    writers[p] = pq.ParquetWriter(
                        shard_dir / f"{p}.parquet", schema, compression=self.parquet_compression
                    )
                writers[p].write_table(sub)
                part_rows[p] += sub.num_rows
            total_rows += tbl.num_rows

        try:
            for batch in self._threaded_batch_iter(reader):
                pending.append(batch)
                pending_rows += batch.num_rows
                if pending_rows >= batch_target:
                    flush(pa.Table.from_batches(pending, schema))
                    pending = []
                    pending_rows = 0
                    if self.verbose:
                        logger.info(f"  Read {total_rows:,} rows...")
            if pending:
                flush(pa.Table.from_batches(pending, schema))
        finally:
            for p, w in writers.items():
                try:
                    w.close()
                    if self.verbose:
                        logger.info(
                            f"  📁 Wrote partition {p}: {part_rows[p]:,} rows"
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"  ⚠️ Error closing partition {p}: {exc}")
            try:
                reader.close()
            except Exception:  # noqa: BLE001
                pass

        result['status'] = ConversionStatus.COMPLETED
        result['row_count'] = total_rows
        result['shards'] = n_partitions
        result['shard_dir'] = str(shard_dir)
        result['partition_col'] = partition_col
        result['partition_breaks'] = breaks
        if self.verbose:
            logger.info(
                f"  ✅ Converted {result['file']}: {total_rows:,} rows "
                f"in {n_partitions} partitions (Arrow)"
            )
        return result

    def _convert_with_row_partitioning_arrow(
        self,
        csv_path: Path,
        shard_dir: Path,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Row-count sharded conversion kept entirely in pyarrow."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        table_name = self._get_table_name(csv_path)
        reader = self._open_arrow_csv(csv_path, table_name)
        schema = reader.schema

        total_rows = 0
        shards_written = 0
        writer = None
        shard_rows = 0
        pending: list = []
        pending_rows = 0
        flush_target = 500_000

        def write_pending() -> None:
            nonlocal writer, shard_rows, pending, pending_rows
            if not pending:
                return
            tbl = pa.Table.from_batches(pending, schema)
            pending = []
            pending_rows = 0
            if writer is None:
                writer = pq.ParquetWriter(
                    shard_dir / f"{shards_written + 1}.parquet",
                    schema, compression=self.parquet_compression,
                )
            writer.write_table(tbl)
            shard_rows += tbl.num_rows

        try:
            for batch in self._threaded_batch_iter(reader):
                pending.append(batch)
                pending_rows += batch.num_rows
                total_rows += batch.num_rows
                if pending_rows >= flush_target:
                    write_pending()
                if shard_rows >= self.ROWS_PER_SHARD:
                    writer.close()
                    shards_written += 1
                    if self.verbose:
                        logger.info(
                            f"  📁 Wrote shard {shards_written}: "
                            f"{shard_rows:,} rows"
                        )
                    writer = None
                    shard_rows = 0
            write_pending()
        finally:
            if writer is not None:
                try:
                    writer.close()
                    shards_written += 1
                    if self.verbose:
                        logger.info(
                            f"  📁 Wrote shard {shards_written}: "
                            f"{shard_rows:,} rows"
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"  ⚠️ Error closing shard: {exc}")
            try:
                reader.close()
            except Exception:  # noqa: BLE001
                pass

        result['status'] = ConversionStatus.COMPLETED
        result['row_count'] = total_rows
        result['shards'] = max(shards_written, 1)
        result['shard_dir'] = str(shard_dir)
        if self.verbose:
            logger.info(
                f"  ✅ Converted {result['file']}: {total_rows:,} rows "
                f"in {shards_written} shards (Arrow)"
            )
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
    
    def convert_all(
        self,
        force: bool = False,
        *,
        write_manifest: bool = True,
        evidence_root: Optional[str | Path] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Convert all CSV files to Parquet.

        Args:
            force: Force reconversion even if parquet exists
            write_manifest: Write ``conversion_manifest.json`` after status resolution
            evidence_root: Optional research-agent run/work directory. When provided,
                the manifest is also registered in an EvidenceStore.
            progress_callback: Optional callable invoked once per converted file
                with ``{'file', 'current', 'total', 'status', 'result'}``. Lets a
                UI render per-file progress without re-implementing the loop.

        Returns:
            Dictionary of conversion results
        """
        # HiRID ships its bulk tables as tar.gz archives; extract them so
        # _get_csv_files / sharding see the unpacked data. Idempotent — skips
        # when shards already exist.
        if self.database == 'hirid':
            try:
                extracted = self._extract_hirid_archives()
                if extracted and self.verbose:
                    logger.info(f"📦 Extracted HiRID archives: {', '.join(extracted)}")
            except Exception as e:
                logger.warning(f"HiRID archive extraction failed: {e}")

        csv_files = self._get_csv_files()
        
        if not csv_files:
            if self.verbose:
                logger.info(f"No CSV files found in {self.data_path}")
            if write_manifest:
                self.write_conversion_manifest({}, evidence_root=evidence_root)
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
            results = self.get_conversion_status()
            if write_manifest:
                self.write_conversion_manifest(results, evidence_root=evidence_root)
            return results
        
        if self.verbose:
            logger.info(f"Converting {len(files_to_convert)} of {len(csv_files)} files...")
        
        results = {}
        total = len(files_to_convert)

        def _emit(csv_path: Path, result: Dict[str, Any], done: int) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback({
                    'file': csv_path.name,
                    'current': done,
                    'total': total,
                    'status': result.get('status'),
                    'result': result,
                })
            except Exception as e:  # noqa: BLE001 — UI callback must not abort conversion
                logger.warning(f"progress_callback raised: {e}")

        # Use parallel conversion for multiple files
        if len(files_to_convert) > 1 and self.parallel_workers > 1:
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                future_map = {
                    executor.submit(self._convert_file, csv_path): csv_path
                    for csv_path in files_to_convert
                }

                done = 0
                for future in as_completed(future_map):
                    csv_path = future_map[future]
                    try:
                        result = future.result()
                        results[csv_path.name] = result
                    except Exception as e:
                        result = {
                            'status': ConversionStatus.FAILED,
                            'error': str(e),
                        }
                        results[csv_path.name] = result
                    done += 1
                    _emit(csv_path, result, done)
        else:
            # Sequential conversion
            for done, csv_path in enumerate(files_to_convert, start=1):
                result = self._convert_file(csv_path)
                results[csv_path.name] = result
                _emit(csv_path, result, done)
        if write_manifest:
            self.write_conversion_manifest(results, evidence_root=evidence_root)
        # 🚀 perf A1/A2: drop bucket/shard cache so a subsequent call sees
        # the new shards/buckets written by this run.
        self._invalidate_dir_caches()
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
    
    def ensure_parquet_ready(
        self,
        auto_convert: bool = True,
        *,
        evidence_root: Optional[str | Path] = None,
    ) -> bool:
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
            self.write_conversion_manifest(
                self.get_conversion_status(),
                evidence_root=evidence_root,
            )
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
        
        results = self.convert_all(evidence_root=evidence_root)
        
        # Check results
        failed = [name for name, r in results.items() if r.get('status') == ConversionStatus.FAILED]
        
        if failed:
            logger.error(f"❌ {len(failed)} files failed to convert:")
            for name in failed[:5]:
                error = results[name].get('error', 'Unknown error')
                logger.error(f"  - {name}: {error}")
            return False
        
        if self.verbose:
            logger.info("✅ Successfully converted all files")
        
        return True

    def write_conversion_manifest(
        self,
        results: Optional[Dict[str, Dict[str, Any]]] = None,
        *,
        evidence_root: Optional[str | Path] = None,
    ) -> Path:
        """Write and optionally evidence-bind a conversion manifest.

        The manifest links raw CSV inputs, parquet outputs, status metadata,
        and SHA-256 hashes so downstream research-agent runs can cite the
        upstream standardisation step without exposing database-specific SQL.
        """

        results = dict(results or self.get_conversion_status())
        # 完整 SHA256 需把每个输入/输出文件再整读一遍——在慢速挂载上代价极高。
        # 仅当本次 manifest 要做 research-agent 证据绑定时才计算密码学哈希；
        # 普通转换用 size+mtime 指纹即可。
        hash_files = evidence_root is not None
        manifest = {
            "schema_version": "easyicu.conversion_manifest/1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "database": self.database,
            "data_path": str(self.data_path),
            "converter": {
                "chunk_size": self.chunk_size,
                "parallel_workers": self.parallel_workers,
                "status_file": self.STATUS_FILE,
                "partitioning_tables": sorted(PARTITIONING_CONFIG.get(self.database, {})),
            },
            "quirks": {
                "hidden_sidecar_files_ignored": True,
                "large_tables_may_be_sharded": True,
            },
            "tables": [
                self._conversion_manifest_entry(file_name, result, hash_files=hash_files)
                for file_name, result in sorted(results.items())
            ],
        }
        path = self.data_path / self.CONVERSION_MANIFEST_FILE
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        if evidence_root is not None:
            try:
                from easyicu.research_agent.evidence import EvidenceStore

                store = EvidenceStore(Path(evidence_root))
                store.register_file(
                    kind="log",
                    description=(
                        "EasyICU data conversion manifest linking raw inputs, "
                        "parquet outputs, conversion status, and SHA-256 hashes."
                    ),
                    source_path=path,
                    aliases=["conversion_manifest", f"conversion_manifest_{self.database}"],
                    producer="easyicu.data_converter",
                    generation_mode="deterministic_conversion_manifest",
                    metadata={
                        "database": self.database,
                        "data_path": str(self.data_path),
                    },
                )
            except Exception as exc:
                logger.warning(f"Failed to register conversion manifest evidence: {exc}")
        return path

    def _conversion_manifest_entry(
        self,
        file_name: str,
        result: Dict[str, Any],
        *,
        hash_files: bool = True,
    ) -> Dict[str, Any]:
        csv_path = self._find_csv_by_name(file_name)
        outputs = self._converted_outputs_for_result(csv_path, result)
        return {
            "file": file_name,
            "table": result.get("table") or (self._get_table_name(csv_path) if csv_path else None),
            "status": result.get("status"),
            "reason": result.get("reason"),
            "error": result.get("error"),
            "row_count": result.get("row_count"),
            "shards": result.get("shards"),
            "partition_col": result.get("partition_col"),
            "partition_breaks": result.get("partition_breaks"),
            "input": (
                self._file_manifest_record(csv_path, hash_files=hash_files)
                if csv_path else None
            ),
            "outputs": [
                self._file_manifest_record(path, hash_files=hash_files)
                for path in outputs
            ],
        }

    def _find_csv_by_name(self, file_name: str) -> Optional[Path]:
        for csv_path in self._get_csv_files():
            if csv_path.name == file_name:
                return csv_path
        candidate = self.data_path / file_name
        return candidate if candidate.exists() else None

    def _converted_outputs_for_result(
        self,
        csv_path: Optional[Path],
        result: Dict[str, Any],
    ) -> List[Path]:
        shard_dir = result.get("shard_dir")
        if shard_dir:
            path = Path(str(shard_dir))
            if path.exists():
                return sorted(path.glob("*.parquet"))
        if csv_path is None:
            return []
        candidates = [
            self._get_parquet_path(csv_path),
            self._get_parquet_path_with_subdir(csv_path),
        ]
        return [path for path in candidates if path.exists()]

    def _file_manifest_record(
        self, path: Optional[Path], *, hash_files: bool = True
    ) -> Dict[str, Any]:
        if path is None:
            return {"path": None, "exists": False}
        exists = path.exists()
        record: Dict[str, Any] = {
            "path": str(path),
            "relative_path": (
                str(path.relative_to(self.data_path))
                if exists and _is_relative_to(path, self.data_path)
                else str(path)
            ),
            "exists": exists,
        }
        if exists and path.is_file():
            stat = path.stat()
            record["size_bytes"] = stat.st_size
            if hash_files:
                record["sha256"] = _sha256_file(path)
            else:
                # 慢速存储上跳过整文件 SHA256（要把所有输入/输出再整读一遍盘，
                # 在 macfuse 挂载上能让转换耗时翻倍）。改用 size+mtime 廉价指纹；
                # 完整 SHA256 仅在 research-agent 证据绑定（传入 evidence_root）时计算。
                record["mtime_ns"] = stat.st_mtime_ns
        return record
    
    def get_table_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all tables (files) in the database.
        
        Returns:
            Dictionary with table information
        """
        info = {}
        
        # Check parquet files
        for pq_path in self.data_path.glob('*.parquet'):
            if _is_hidden_sidecar(pq_path):
                continue
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
