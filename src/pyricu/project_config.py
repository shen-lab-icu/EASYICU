"""Project-level configuration for pyricu.

This module provides centralized configuration for:
- Data paths (production data)
- Default patient lists
- Common constants

Separates project/testing configuration from data source configuration (config.py).
"""

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

# ============================================================================
# Environment Variables
# ============================================================================

# Allow override via environment variables
ENV_PROD_DATA = os.getenv('PYRICU_PROD_DATA')
ENV_PROJECT_ROOT = os.getenv('PYRICU_PROJECT_ROOT')

# ============================================================================
# Path Configuration
# ============================================================================

# Project root directory
if ENV_PROJECT_ROOT:
    PROJECT_ROOT = Path(ENV_PROJECT_ROOT)
else:
    # Default: parent directory of pyricu/src/pyricu
    PROJECT_ROOT = Path(__file__).parent.parent.parent

# Production data paths - use original raw data only
if ENV_PROD_DATA:
    PRODUCTION_DATA_PATH = Path(ENV_PROD_DATA)
else:
    PRODUCTION_DATA_PATH = Path("/home/1_publicData/icu_databases/mimiciv/3.1")

PRODUCTION_DATA_EICU = Path("/home/1_publicData/icu_databases/eicu/2.0.1")
PRODUCTION_DATA_AUMC = Path("/home/1_publicData/icu_databases/aumc/1.0.2")
PRODUCTION_DATA_HIRID = Path("/home/1_publicData/icu_databases/hirid/1.1.1")

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = PROJECT_ROOT / ".cache"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [OUTPUT_DIR, CACHE_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Default Patient IDs  
# ============================================================================

# Default patient IDs extracted from production data
# MIMIC-IV: stay_ids with rich SOFA-2 features
DEFAULT_PATIENTS_MIIV = [30005000, 30009597, 30017005]

# eICU: patientunitstayid with RRT and vasopressor data
DEFAULT_PATIENTS_EICU = [1589473, 2506331, 2683425, 2785291, 3146941]

# AUMC: admissionid with RRT, ECMO and vasopressor data
DEFAULT_PATIENTS_AUMC = [11, 3441, 53]

# HiRID: patientid (to be extracted from production data)
DEFAULT_PATIENTS_HIRID: Optional[List[int]] = None

# Default (MIMIC-IV)
DEFAULT_PATIENTS = DEFAULT_PATIENTS_MIIV

# Single patient for quick debugging
DEBUG_PATIENT_MIIV = [30017005]
DEBUG_PATIENT_EICU = [2572404]
DEBUG_PATIENT = DEBUG_PATIENT_MIIV

def _load_unique_ids(file_path: Path, column: str, limit: int) -> List[int]:
    if not file_path.exists():
        return []

    series = pd.read_parquet(file_path, columns=[column])[column].dropna()
    ids: List[int] = []
    for value in series:
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            continue
        if candidate not in ids:
            ids.append(candidate)
            if len(ids) >= limit:
                break
    return ids

def _load_ids_for_database(database: str, data_path: Path, limit: int) -> List[int]:
    if database == "aumc":
        return _load_unique_ids(data_path / "admissions.parquet", "admissionid", limit)
    if database == "hirid":
        return _load_unique_ids(data_path / "general.parquet", "patientid", limit)
    return []

# ============================================================================
# Data Source Configuration
# ============================================================================

# Default data source name
DEFAULT_SOURCE = "miiv"

# Supported data sources
SUPPORTED_SOURCES = ["miiv", "eicu", "aumc", "hirid"]

# ============================================================================
# Concept Configuration
# ============================================================================

# Common vital sign concepts
VITAL_CONCEPTS = ["hr", "sbp", "dbp", "temp", "resp", "spo2"]

# Common lab concepts
LAB_CONCEPTS = ["wbc", "hgb", "plt", "crea", "bili", "lact"]

# SOFA component concepts
SOFA_CONCEPTS = ["sofa", "resp", "coag", "liver", "cardio", "cns", "renal"]

# SOFA2 component concepts (2025 definitions)
SOFA2_CONCEPTS = ["sofa2", "resp2", "coag2", "liver2", "cardio2", "cns2", "renal2"]

# Sepsis-3 concepts
SEPSIS3_CONCEPTS = ["susp_inf", "sofa", "sep3"]

# Sepsis-2 concepts (2025 definitions)
SEPSIS2_CONCEPTS = ["susp_inf", "sofa2", "sep2"]

# ============================================================================
# Time Configuration
# ============================================================================

# Default time windows (in hours)
DEFAULT_TIME_WINDOW = 24  # 24 hours
DEFAULT_INTERVAL = 1  # 1 hour intervals

# SOFA time window
SOFA_TIME_WINDOW = 24  # 24 hours from ICU admission

# Sepsis time window
SEPSIS_TIME_WINDOW = 48  # 48 hours for infection suspicion

# ============================================================================
# Logging Configuration
# ============================================================================

# Default log level
LOG_LEVEL = os.getenv('PYRICU_LOG_LEVEL', 'INFO')

# Enable verbose output
VERBOSE = os.getenv('PYRICU_VERBOSE', 'True').lower() in ('true', '1', 'yes')

# Enable progress bars
SHOW_PROGRESS = os.getenv('PYRICU_PROGRESS', 'True').lower() in ('true', '1', 'yes')

# ============================================================================
# Performance Configuration
# ============================================================================

# Enable caching
ENABLE_CACHE = True

# Cache size limit (MB) - optimized for 16GB systems
CACHE_SIZE_LIMIT = int(os.getenv('PYRICU_CACHE_SIZE_MB', '0'))  # 0 = unlimited cache by default

# Auto-clear cache on startup - helps prevent stale data issues
# 🚀 性能优化：默认禁用自动缓存清除，避免每次导入都清除缓存
# 如需清除缓存，可设置环境变量 PYRICU_AUTO_CLEAR_CACHE=True 或调用 clear_cache()
AUTO_CLEAR_CACHE = os.getenv('PYRICU_AUTO_CLEAR_CACHE', 'False').lower() in ('true', '1', 'yes')

# Number of parallel workers - optimized for 16GB systems
MAX_WORKERS = int(os.getenv('PYRICU_MAX_WORKERS', '999'))  # Reduced from 4 to 2 for memory efficiency

# Chunk size for batch processing - optimized for 16GB systems
CHUNK_SIZE_STR = os.getenv('PYRICU_CHUNK_SIZE', 'None')
CHUNK_SIZE = int(CHUNK_SIZE_STR) if CHUNK_SIZE_STR != 'None' else None  # Reduced from 10000 to 5000

# Memory management settings
MEMORY_SAFETY_THRESHOLD = 0.8  # Use 80% of total memory as safety limit
MEMORY_WARNING_THRESHOLD = 0.7  # Warning at 70% memory usage
MEMORY_CLEANUP_THRESHOLD = 0.85  # Automatic cleanup at 85% memory usage

# Performance optimization flags
AUTO_OPTIMIZE_DTYPES = os.getenv('PYRICU_AUTO_OPTIMIZE_DTYPES', 'True').lower() in ('true', '1', 'yes')
ENABLE_MEMORY_MONITORING = os.getenv('PYRICU_ENABLE_MEMORY_MONITORING', 'True').lower() in ('true', '1', 'yes')
USE_CHUNKED_LOADING = os.getenv('PYRICU_USE_CHUNKED_LOADING', 'auto')  # auto, always, never

# ============================================================================
# Helper Functions
# ============================================================================

def get_data_path(source: str = "production", database: str = "miiv") -> Path:
    """Get data path for specified database.
    
    Args:
        source: Data source type (only 'production' is supported)
        database: Database name ('miiv', 'eicu', 'hirid', 'aumc')
        
    Returns:
        Path to data directory
        
    Examples:
        >>> get_data_path('production', 'miiv')
        PosixPath('/home/1_publicData/icu_databases/mimiciv/3.1')
        >>> get_data_path('production', 'eicu')
        PosixPath('/home/1_publicData/icu_databases/eicu/2.0.1')
    """
    if source != "production":
        raise ValueError(f"Only 'production' data source is supported, got: {source}")
        
    if database == "miiv":
        return PRODUCTION_DATA_PATH
    elif database == "eicu":
        return PRODUCTION_DATA_EICU
    elif database == "aumc":
        return PRODUCTION_DATA_AUMC
    elif database == "hirid":
        return PRODUCTION_DATA_HIRID
    else:
        raise ValueError(f"Unknown database: {database}")

def get_patient_ids(patient_set: str = "default", database: str = "miiv", data_path: Path = None) -> List[int]:
    """Get patient IDs for specified patient set and database.
    
    Args:
        patient_set: Patient set name ('default', 'debug', or number like '50')
        database: Database name ('miiv', 'eicu', 'hirid', 'aumc')
        data_path: Optional data path (will use get_data_path if not provided)
        
    Returns:
        List of patient subject_ids or stay_ids
        
    Examples:
        >>> get_patient_ids('default', 'miiv')
        [30005000, 30009597, 30017005]
        >>> get_patient_ids('debug', 'eicu')
        [2572404]
    """
    if data_path is None:
        data_path = get_data_path('production', database)
    else:
        data_path = Path(data_path)

    if patient_set == "default":
        if database == "miiv":
            return DEFAULT_PATIENTS_MIIV
        elif database == "eicu":
            return DEFAULT_PATIENTS_EICU
        elif database == "aumc":
            global DEFAULT_PATIENTS_AUMC
            if DEFAULT_PATIENTS_AUMC is None:
                DEFAULT_PATIENTS_AUMC = _load_ids_for_database(database, data_path, 3)
            if not DEFAULT_PATIENTS_AUMC:
                raise ValueError("Unable to load default AUMC admission IDs")
            return DEFAULT_PATIENTS_AUMC
        elif database == "hirid":
            global DEFAULT_PATIENTS_HIRID
            if DEFAULT_PATIENTS_HIRID is None:
                DEFAULT_PATIENTS_HIRID = _load_ids_for_database(database, data_path, 3)
            if not DEFAULT_PATIENTS_HIRID:
                raise ValueError("Unable to load default HiRID patient IDs")
            return DEFAULT_PATIENTS_HIRID
        else:
            return DEFAULT_PATIENTS
    elif patient_set == "debug":
        if database == "miiv":
            return DEBUG_PATIENT_MIIV
        elif database == "eicu":
            return DEBUG_PATIENT_EICU
        elif database in {"aumc", "hirid"}:
            ids = _load_ids_for_database(database, data_path, 1)
            if not ids:
                raise ValueError(f"Unable to load debug IDs for '{database}'")
            return ids
        else:
            return DEBUG_PATIENT
    elif patient_set.isdigit():
        # Load N patients from production data
        limit = int(patient_set)
        ids = _load_ids_for_database(database, data_path, limit)
        if not ids:
            raise ValueError(f"Unable to load {limit} patients for database '{database}'")
        return ids
    else:
        raise ValueError(f"Unknown patient set: {patient_set}")

def get_concepts(concept_group: str) -> List[str]:
    """Get concept list for specified group.
    
    Args:
        concept_group: Concept group name ('vitals', 'labs', 'sofa', 'sepsis3')
        
    Returns:
        List of concept names
        
    Examples:
        >>> get_concepts('vitals')
        ['hr', 'sbp', 'dbp', 'temp', 'resp', 'spo2']
        >>> get_concepts('sofa')
        ['sofa', 'resp', 'coag', 'liver', 'cardio', 'cns', 'renal']
    """
    concept_map = {
        'vitals': VITAL_CONCEPTS,
        'labs': LAB_CONCEPTS,
        'sofa': SOFA_CONCEPTS,
        'sofa2': SOFA2_CONCEPTS,
        'sepsis3': SEPSIS3_CONCEPTS,
        'sepsis2': SEPSIS2_CONCEPTS,
    }
    
    if concept_group not in concept_map:
        raise ValueError(f"Unknown concept group: {concept_group}")
    
    return concept_map[concept_group]

def print_config() -> None:
    """Print current configuration (for debugging).
    
    Examples:
        >>> print_config()
        📋 Pyricu Project Configuration
        ================================
        Project Root: /home/zhuhb/project/ricu_to_python/pyricu
        Production Data (MIMIC-IV): /home/1_publicData/icu_databases/mimiciv/3.1
        ...
    """
    print("📋 Pyricu Project Configuration")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Production Data (MIMIC-IV): {PRODUCTION_DATA_PATH}")
    print(f"Production Data (eICU): {PRODUCTION_DATA_EICU}")
    print(f"Production Data (AUMC): {PRODUCTION_DATA_AUMC}")
    print(f"Production Data (HiRID): {PRODUCTION_DATA_HIRID}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"Cache Dir: {CACHE_DIR}")
    print(f"Logs Dir: {LOGS_DIR}")
    print()
    print(f"Default Patients (MIMIC-IV): {len(DEFAULT_PATIENTS_MIIV)} patients")
    print(f"Default Patients (eICU): {len(DEFAULT_PATIENTS_EICU)} patients")
    print(f"Default Source: {DEFAULT_SOURCE}")
    print()
    print(f"Cache Enabled: {ENABLE_CACHE}")
    print(f"Max Workers: {MAX_WORKERS}")
    print(f"Chunk Size: {CHUNK_SIZE}")
    print(f"Verbose: {VERBOSE}")
    print("=" * 50)

# ============================================================================
# Validation
# ============================================================================

def validate_paths() -> bool:
    """Validate that production data paths exist.
    
    Returns:
        True if at least one production path exists, False otherwise
        
    Examples:
        >>> if validate_paths():
        ...     print("Production paths available")
    """
    valid = False
    
    # Check production data paths
    for db_name, db_path in [
        ("MIMIC-IV", PRODUCTION_DATA_PATH),
        ("eICU", PRODUCTION_DATA_EICU),
        ("AUMC", PRODUCTION_DATA_AUMC),
        ("HiRID", PRODUCTION_DATA_HIRID),
    ]:
        if db_path.exists():
            valid = True
        elif VERBOSE:
            print(f"ℹ️  Info: {db_name} production data path not found: {db_path}")
    
    return valid

# Auto-validate on import
if VERBOSE:
    validate_paths()

# ============================================================================
# Export all configuration
# ============================================================================

__all__ = [
    # Paths
    'PROJECT_ROOT',
    'PRODUCTION_DATA_PATH',
    'PRODUCTION_DATA_EICU',
    'PRODUCTION_DATA_AUMC',
    'PRODUCTION_DATA_HIRID',
    'OUTPUT_DIR',
    'CACHE_DIR',
    'LOGS_DIR',
    
    # Patient IDs
    'DEFAULT_PATIENTS',
    'DEFAULT_PATIENTS_MIIV',
    'DEFAULT_PATIENTS_EICU',
    'DEFAULT_PATIENTS_AUMC',
    'DEFAULT_PATIENTS_HIRID',
    'DEBUG_PATIENT',
    'DEBUG_PATIENT_MIIV',
    'DEBUG_PATIENT_EICU',
    
    # Data sources
    'DEFAULT_SOURCE',
    'SUPPORTED_SOURCES',
    
    # Concepts
    'VITAL_CONCEPTS',
    'LAB_CONCEPTS',
    'SOFA_CONCEPTS',
    'SOFA2_CONCEPTS',
    'SEPSIS3_CONCEPTS',
    'SEPSIS2_CONCEPTS',
    
    # Time configuration
    'DEFAULT_TIME_WINDOW',
    'DEFAULT_INTERVAL',
    'SOFA_TIME_WINDOW',
    'SEPSIS_TIME_WINDOW',
    
    # Logging
    'LOG_LEVEL',
    'VERBOSE',
    'SHOW_PROGRESS',
    
    # Performance
    'ENABLE_CACHE',
    'CACHE_SIZE_LIMIT',
    'MAX_WORKERS',
    'CHUNK_SIZE',
    'MEMORY_SAFETY_THRESHOLD',
    'MEMORY_WARNING_THRESHOLD',
    'MEMORY_CLEANUP_THRESHOLD',
    'AUTO_OPTIMIZE_DTYPES',
    'ENABLE_MEMORY_MONITORING',
    'USE_CHUNKED_LOADING',
    
    # Helper functions
    'get_data_path',
    'get_patient_ids',
    'get_concepts',
    'print_config',
    'validate_paths',
]
