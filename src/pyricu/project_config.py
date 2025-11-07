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
    TEST_DATA_PATH = PROJECT_ROOT / "test_data_50patients"

TEST_DATA_MINIMAL = PROJECT_ROOT / "test_data_minimal"

# Production data path
if ENV_PROD_DATA:
    PRODUCTION_DATA_PATH = Path(ENV_PROD_DATA)
else:
    PRODUCTION_DATA_PATH = Path("/home/1_publicData/icu_databases/mimiciv/3.1")

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
# Note: These are stay_ids from test_data_50patients
DEFAULT_TEST_PATIENTS = [34807493, 33987268, 35044219]

# Extended 50-patient test set (for comprehensive validation)
# Note: Automatically loaded from test data - use get_patient_ids('50patients')
DEFAULT_50_PATIENTS = None  # Will be loaded dynamically

# Single patient for quick debugging
DEBUG_PATIENT = [34807493]

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

# Cache size limit (MB)
CACHE_SIZE_LIMIT = 1000  # 1 GB

# Number of parallel workers
MAX_WORKERS = int(os.getenv('PYRICU_MAX_WORKERS', '4'))

# Chunk size for batch processing
CHUNK_SIZE = int(os.getenv('PYRICU_CHUNK_SIZE', '10000'))

# ============================================================================
# Helper Functions
# ============================================================================

def get_data_path(source: str = "test") -> Path:
    """Get data path for specified source.
    
    Args:
        source: Data source type ('test', 'test_minimal', 'production')
        
    Returns:
        Path to data directory
        
    Examples:
        >>> get_data_path('test')
        PosixPath('/home/zhuhb/project/ricu_to_python/pyricu/test_data_50patients')
        >>> get_data_path('production')
        PosixPath('/home/1_publicData/icu_databases/mimiciv/3.1')
    """
    if source == "test":
        return TEST_DATA_PATH
    elif source == "test_minimal":
        return TEST_DATA_MINIMAL
    elif source == "production":
        return PRODUCTION_DATA_PATH
    else:
        raise ValueError(f"Unknown data source: {source}")


def get_patient_ids(patient_set: str = "default") -> List[int]:
    """Get patient IDs for specified test set.
    
    Args:
        patient_set: Patient set name ('default', '50patients', 'debug')
        
    Returns:
        List of patient subject_ids or stay_ids
        
    Examples:
        >>> get_patient_ids('default')
        [34807493, 33987268, 35044219]
        >>> get_patient_ids('debug')
        [34807493]
    """
    if patient_set == "default":
        return DEFAULT_TEST_PATIENTS
    elif patient_set == "50patients":
        # Dynamically load from test data
        if DEFAULT_50_PATIENTS is None:
            try:
                from pyricu.quickstart import get_patient_ids as load_patient_ids
                return load_patient_ids(TEST_DATA_PATH, max_patients=50, database='miiv')
            except:
                # Fallback to default
                return DEFAULT_TEST_PATIENTS
        return DEFAULT_50_PATIENTS
    elif patient_set == "debug":
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
    'PRODUCTION_DATA_PATH',
    'OUTPUT_DIR',
    'CACHE_DIR',
    'LOGS_DIR',
    
    # Patient IDs
    'DEFAULT_TEST_PATIENTS',
    'DEFAULT_50_PATIENTS',
    'DEBUG_PATIENT',
    
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
    
    # Helper functions
    'get_data_path',
    'get_patient_ids',
    'get_concepts',
    'print_config',
    'validate_paths',
]
