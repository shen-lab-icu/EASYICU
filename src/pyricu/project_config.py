"""Project-level configuration for pyricu testing and demos.

This module provides centralized configuration for:
- Data paths (test data, production data)
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
ENV_TEST_DATA = os.getenv('PYRICU_TEST_DATA')
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

# Test data directories
if ENV_TEST_DATA:
    TEST_DATA_PATH = Path(ENV_TEST_DATA)
else:
    TEST_DATA_PATH = PROJECT_ROOT / "test_data_miiv"

TEST_DATA_MINIMAL = PROJECT_ROOT / "test_data_minimal"
TEST_DATA_MIIV = PROJECT_ROOT / "test_data_miiv"
TEST_DATA_EICU = PROJECT_ROOT / "test_data_eicu"
TEST_DATA_AUMC = PROJECT_ROOT / "test_data_aumc"
TEST_DATA_HIRID = PROJECT_ROOT / "test_data_hirid"

# Production data path
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

# Standard 3-patient test set (used in most tests)
# MIMIC-IV: stay_ids from test_data_miiv
# 选择有RRT+血管加压药+谵妄评估的患者以便测试SOFA-2特征
DEFAULT_TEST_PATIENTS_MIIV = [30005000, 30009597, 30017005]

# eICU: patientunitstayid from test_data_eicu
# 选择有RRT和血管加压药数据的患者以便测试SOFA-2特征
# 患者245906数据最丰富(RRT 13条, 血管加压药 17条)
DEFAULT_TEST_PATIENTS_EICU = [243334, 245906, 249329, 251510, 257542]

# AUMC: admissionid from test_data_aumc
# 选择有RRT、ECMO和血管加压药数据的患者以便测试SOFA-2特征
# 患者11: RRT数据（CVVH）
# 患者3441: ECMO数据（VA ECMO，296条记录）
# 患者53: RRT数据（CVVH）
DEFAULT_TEST_PATIENTS_AUMC = [11, 3441, 53]

# Default (MIMIC-IV)
DEFAULT_TEST_PATIENTS = DEFAULT_TEST_PATIENTS_MIIV
DEFAULT_TEST_PATIENTS_HIRID: Optional[List[int]] = None

# Extended 50-patient test set (for comprehensive validation)
# Note: Automatically loaded from test data - use get_patient_ids('50patients')
DEFAULT_50_PATIENTS = None  # Will be loaded dynamically

# Single patient for quick debugging
DEBUG_PATIENT_MIIV = [30017005]  # 修正为实际存在的患者ID
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
AUTO_CLEAR_CACHE = os.getenv('PYRICU_AUTO_CLEAR_CACHE', 'True').lower() in ('true', '1', 'yes')

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

def get_data_path(source: str = "test", database: str = "miiv") -> Path:
    """Get data path for specified source and database.
    
    Args:
        source: Data source type ('test', 'test_minimal', 'production')
        database: Database name ('miiv', 'eicu', 'hirid', 'aumc')
        
    Returns:
        Path to data directory
        
    Examples:
        >>> get_data_path('test', 'miiv')
        PosixPath('/home/zhuhb/project/ricu_to_python/pyricu/test_data_50patients')
        >>> get_data_path('test', 'eicu')
        PosixPath('/home/zhuhb/project/ricu_to_python/pyricu/test_data_eicu')
        >>> get_data_path('production', 'miiv')
        PosixPath('/home/1_publicData/icu_databases/mimiciv/3.1')
    """
    if source == "test":
        if database == "miiv":
            return TEST_DATA_PATH
        elif database == "eicu":
            return TEST_DATA_EICU
        elif database == "aumc":
            return TEST_DATA_AUMC
        elif database == "hirid":
            return TEST_DATA_HIRID
        else:
            raise ValueError(f"Test data for database '{database}' not configured")
    elif source == "test_minimal":
        return TEST_DATA_MINIMAL
    elif source == "production":
        if database == "miiv":
            return PRODUCTION_DATA_PATH
        elif database == "eicu":
            return PRODUCTION_DATA_EICU
        elif database == "aumc":
            return PRODUCTION_DATA_AUMC
        elif database == "hirid":
            return PRODUCTION_DATA_HIRID
        else:
            raise ValueError(f"Production data for database '{database}' not configured")
    else:
        raise ValueError(f"Unknown data source: {source}")


def get_patient_ids(patient_set: str = "default", database: str = "miiv", data_path: Path = None) -> List[int]:
    """Get patient IDs for specified test set and database.
    
    Args:
        patient_set: Patient set name ('default', '50patients', 'debug')
        database: Database name ('miiv', 'eicu', 'hirid', 'aumc')
        data_path: Optional data path (will use get_data_path if not provided)
        
    Returns:
        List of patient subject_ids or stay_ids
        
    Examples:
        >>> get_patient_ids('default', 'miiv')
        [34807493, 33987268, 35044219]
        >>> get_patient_ids('debug', 'eicu')
        [2572404]
    """
    if data_path is None:
        data_path = get_data_path('test', database)
    else:
        data_path = Path(data_path)

    if patient_set == "default":
        if database == "miiv":
            return DEFAULT_TEST_PATIENTS_MIIV
        if database == "eicu":
            return DEFAULT_TEST_PATIENTS_EICU
        if database == "aumc":
            global DEFAULT_TEST_PATIENTS_AUMC
            if DEFAULT_TEST_PATIENTS_AUMC is None:
                DEFAULT_TEST_PATIENTS_AUMC = _load_ids_for_database(database, data_path, 3)
            if not DEFAULT_TEST_PATIENTS_AUMC:
                raise ValueError("Unable to load default AUMC admission IDs")
            return DEFAULT_TEST_PATIENTS_AUMC
        if database == "hirid":
            global DEFAULT_TEST_PATIENTS_HIRID
            if DEFAULT_TEST_PATIENTS_HIRID is None:
                DEFAULT_TEST_PATIENTS_HIRID = _load_ids_for_database(database, data_path, 3)
            if not DEFAULT_TEST_PATIENTS_HIRID:
                raise ValueError("Unable to load default HiRID patient IDs")
            return DEFAULT_TEST_PATIENTS_HIRID
        return DEFAULT_TEST_PATIENTS
    elif patient_set == "50patients":
        if database in {"aumc", "hirid"}:
            ids = _load_ids_for_database(database, data_path, 50)
            if not ids:
                raise ValueError(f"Unable to load 50 patients for database '{database}'")
            return ids
        if DEFAULT_50_PATIENTS is None:
            try:
                # 使用新的统一API替代quickstart
                from pyricu.base import BaseICULoader
                loader = BaseICULoader(database=database, data_path=data_path, verbose=False)
                # 简化版：返回默认患者列表，因为具体加载需要更多上下文
                if database == "miiv":
                    return DEFAULT_TEST_PATIENTS_MIIV
                if database == "eicu":
                    return DEFAULT_TEST_PATIENTS_EICU
                return DEFAULT_TEST_PATIENTS
            except Exception:
                if database == "miiv":
                    return DEFAULT_TEST_PATIENTS_MIIV
                if database == "eicu":
                    return DEFAULT_TEST_PATIENTS_EICU
                return DEFAULT_TEST_PATIENTS
        return DEFAULT_50_PATIENTS
    elif patient_set == "debug":
        if database == "miiv":
            return DEBUG_PATIENT_MIIV
        if database == "eicu":
            return DEBUG_PATIENT_EICU
        if database in {"aumc", "hirid"}:
            ids = _load_ids_for_database(database, data_path, 1)
            if not ids:
                raise ValueError(f"Unable to load debug IDs for '{database}'")
            return ids
        return DEBUG_PATIENT
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
        Test Data: /home/zhuhb/project/ricu_to_python/pyricu/test_data_50patients
        Production Data: /home/1_publicData/icu_databases/mimiciv/3.1
        ...
    """
    print("📋 Pyricu Project Configuration")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Test Data: {TEST_DATA_PATH}")
    print(f"Production Data: {PRODUCTION_DATA_PATH}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"Cache Dir: {CACHE_DIR}")
    print(f"Logs Dir: {LOGS_DIR}")
    print()
    print(f"Default Patients: {len(DEFAULT_TEST_PATIENTS)} patients")
    if DEFAULT_50_PATIENTS is not None:
        print(f"50-Patient Set: {len(DEFAULT_50_PATIENTS)} patients")
    else:
        print(f"50-Patient Set: Dynamic (loaded on demand)")
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
    """Validate that critical paths exist.
    
    Returns:
        True if all paths exist, False otherwise
        
    Examples:
        >>> if validate_paths():
        ...     print("All paths valid")
    """
    valid = True
    
    # Check test data path
    if not TEST_DATA_PATH.exists():
        print(f"⚠️  Warning: Test data path not found: {TEST_DATA_PATH}")
        valid = False
    
    # Check production data path (optional)
    if not PRODUCTION_DATA_PATH.exists():
        print(f"ℹ️  Info: Production data path not found: {PRODUCTION_DATA_PATH}")
        # Don't mark as invalid since production data is optional
    
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
    'TEST_DATA_PATH',
    'TEST_DATA_MINIMAL',
    'TEST_DATA_EICU',
    'TEST_DATA_AUMC',
    'TEST_DATA_HIRID',
    'PRODUCTION_DATA_PATH',
    'PRODUCTION_DATA_EICU',
    'PRODUCTION_DATA_AUMC',
    'PRODUCTION_DATA_HIRID',
    'OUTPUT_DIR',
    'CACHE_DIR',
    'LOGS_DIR',
    
    # Patient IDs
    'DEFAULT_TEST_PATIENTS',
    'DEFAULT_TEST_PATIENTS_MIIV',
    'DEFAULT_TEST_PATIENTS_EICU',
    'DEFAULT_TEST_PATIENTS_AUMC',
    'DEFAULT_TEST_PATIENTS_HIRID',
    'DEFAULT_50_PATIENTS',
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
