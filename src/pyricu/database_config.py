"""Database-specific configuration constants.

This module centralizes all database-specific constants such as column names,
thresholds, and conversion factors to avoid hardcoding throughout the codebase.
"""

from typing import Dict, List
import pandas as pd

# ============================================================================
# Database Metadata
# ============================================================================

SUPPORTED_DATABASES = ['miiv', 'eicu', 'aumc', 'hirid', 'sic']

DATABASE_FULL_NAMES = {
    'miiv': 'MIMIC-IV',
    'eicu': 'eICU Collaborative Research Database',
    'aumc': 'AmsterdamUMCdb',
    'hirid': 'HiRID (High Time Resolution ICU Dataset)',
    'sic': 'SICdb (Surgical Intensive Care Database)',
}

# ============================================================================
# Column Names by Database
# ============================================================================

# Primary ID columns (patient/stay identifiers)
ID_COLUMNS: Dict[str, List[str]] = {
    'miiv': ['stay_id', 'subject_id', 'hadm_id'],
    'eicu': ['patientunitstayid', 'patientid'],
    'aumc': ['admissionid', 'patientid'],
    'hirid': ['patientid'],
    'sic': ['patientid', 'icustay_id'],
}

# Primary time columns
TIME_COLUMNS: Dict[str, str] = {
    'miiv': 'charttime',
    'eicu': 'observationoffset',
    'aumc': 'measuredat',
    'hirid': 'datetime',
    'sic': 'datetime',
}

# Alternative time columns (start/end times for intervals)
START_TIME_COLUMNS: Dict[str, str] = {
    'miiv': 'starttime',
    'eicu': 'drugstartoffset',
    'aumc': 'start',
    'hirid': 'datetime',
    'sic': 'datetime',
}

END_TIME_COLUMNS: Dict[str, str] = {
    'miiv': 'endtime',
    'eicu': 'drugstopoffset',
    'aumc': 'stop',
    'hirid': 'datetime',
    'sic': 'datetime',
}

# Value columns (generic measurement values)
VALUE_COLUMNS: Dict[str, str] = {
    'miiv': 'valuenum',
    'eicu': 'value',
    'aumc': 'value',
    'hirid': 'value',
    'sic': 'value',
}

# Unit columns (measurement units)
UNIT_COLUMNS: Dict[str, str] = {
    'miiv': 'valueuom',
    'eicu': 'unit',
    'aumc': 'unit',
    'hirid': 'unit',
    'sic': 'unit',
}

# ============================================================================
# Time-related Constants
# ============================================================================

# Time format for each database
TIME_IS_OFFSET: Dict[str, bool] = {
    'miiv': False,  # Uses datetime
    'eicu': True,   # Uses minutes from admission
    'aumc': False,  # Uses datetime
    'hirid': False, # Uses datetime
    'sic': False,   # Uses datetime
}

# Threshold for detecting pre-ICU records (minutes)
PRE_ICU_TIME_THRESHOLD = -1000  # For MIMIC-IV, records before -1000 min are pre-ICU

# Default time filtering (include records >= this value)
DEFAULT_MIN_TIME = 0  # ICU admission time

# ============================================================================
# Unit Conversion Constants
# ============================================================================

# FiO2 unit detection thresholds
# FiO2 can be stored as fraction (0-1) or percentage (21-100)
FIO2_FRACTION_THRESHOLD = 1.0  # Values <= 1 are likely fractions
FIO2_PERCENTAGE_RANGE = (21, 100)  # Valid percentage range

# FiO2 default value (room air)
FIO2_ROOM_AIR = 21.0  # Percentage

# Temperature conversion
TEMP_FAHRENHEIT_THRESHOLD = 50.0  # Temps > 50 are likely Fahrenheit
TEMP_CELSIUS_RANGE = (30, 45)  # Valid Celsius range for body temperature

# Vasopressor rate units (target: μg/kg/min)
VASO_RATE_UNIT_TARGET = 'mcg/kg/min'

# Weight units
WEIGHT_KG_RANGE = (20, 300)  # Valid weight range in kg
WEIGHT_LB_THRESHOLD = 300  # Weights > 300 are likely lbs

# ============================================================================
# SOFA Score Constants
# ============================================================================

# SOFA respiratory thresholds (PaO2/FiO2)
SOFA_RESP_THRESHOLDS = {
    4: 100,
    3: 200,
    2: 300,
    1: 400,
}

# SOFA coagulation thresholds (platelets ×10³/mm³)
SOFA_COAG_THRESHOLDS = {
    4: 20,
    3: 50,
    2: 100,
    1: 150,
}

# SOFA liver thresholds (bilirubin mg/dL)
SOFA_LIVER_THRESHOLDS = {
    4: 12.0,
    3: 6.0,
    2: 2.0,
    1: 1.2,
}

# SOFA cardiovascular thresholds (vasopressor doses in μg/kg/min)
SOFA_CARDIO_THRESHOLDS = {
    'dopamine': {4: 15, 3: 5, 2: 0},
    'epinephrine': {4: 0.1, 3: 0, 2: float('-inf')},
    'norepinephrine': {4: 0.1, 3: 0, 2: float('-inf')},
    'dobutamine': {2: 0},
}

# SOFA CNS thresholds (GCS)
SOFA_CNS_THRESHOLDS = {
    4: (3, 5),
    3: (6, 9),
    2: (10, 12),
    1: (13, 14),
    0: (15, 15),
}

# SOFA renal thresholds
SOFA_RENAL_CREA_THRESHOLDS = {  # Creatinine mg/dL
    4: 5.0,
    3: 3.5,
    2: 2.0,
    1: 1.2,
}

SOFA_RENAL_URINE_THRESHOLDS = {  # Urine output mL/day
    4: 200,
    3: 500,
}

# ============================================================================
# SOFA-2 Constants (2025 version)
# ============================================================================

# SOFA-2 respiratory thresholds (adjusted)
SOFA2_RESP_PF_THRESHOLDS = {
    4: 75,
    3: 150,
    2: 225,
    1: 300,
}

# SOFA-2 SaFi thresholds (SpO2/FiO2)
SOFA2_RESP_SF_THRESHOLDS = {
    4: 120,
    3: 200,
    2: 250,
    1: 300,
}

# SOFA-2 SpO2 validity threshold
SOFA2_SPO2_VALID_MAX = 98  # Only use SaFi if SpO2 < 98%

# SOFA-2 coagulation thresholds (relaxed)
SOFA2_COAG_THRESHOLDS = {
    4: 50,
    3: 80,
    2: 100,
    1: 150,
}

# SOFA-2 liver thresholds (relaxed 1-point)
SOFA2_LIVER_THRESHOLDS = {
    4: 12.0,
    3: 6.0,
    2: 3.0,
    1: 1.2,  # Relaxed from 1.9 in SOFA-1
}

# SOFA-2 cardiovascular thresholds (combined catecholamines)
SOFA2_CARDIO_COMBINED_THRESHOLDS = {
    4: 0.4,  # NE+Epi > 0.4
    3: 0.2,  # NE+Epi 0.2-0.4
    2: 0,    # Any dose
}

SOFA2_CARDIO_DOPAMINE_THRESHOLDS = {  # Only if NE+Epi unavailable
    4: 40,
    3: 20,
    2: 0,
}

# SOFA-2 renal urine output thresholds (duration-based)
SOFA2_RENAL_UO_THRESHOLDS = {
    1: {'duration_hours': 6, 'rate_ml_kg_h': 0.5},   # <0.5 for 6-12h
    2: {'duration_hours': 12, 'rate_ml_kg_h': 0.5},  # <0.5 for ≥12h
    3: {'duration_hours': 24, 'rate_ml_kg_h': 0.3},  # <0.3 for ≥24h
}

# SOFA-2 RRT criteria thresholds
SOFA2_RRT_CRITERIA = {
    'potassium_mmol_l': 6.0,
    'ph': 7.20,
    'bicarbonate_mmol_l': 12.0,
}

# ============================================================================
# Window Duration Constants
# ============================================================================

# Default window durations for various calculations
WINDOW_DURATIONS = {
    'sofa_default': pd.Timedelta(hours=24),
    'pafi_match': pd.Timedelta(hours=2),
    'vent_match': pd.Timedelta(hours=6),
    'vent_min_length': pd.Timedelta(minutes=30),
    'gcs_valid': pd.Timedelta(hours=6),
    'urine_24h': pd.Timedelta(hours=24),
    'urine_min': pd.Timedelta(hours=12),
    'vaso_max_gap': pd.Timedelta(minutes=5),
    'vaso_min_dur': pd.Timedelta(hours=1),
}

# Interval creation constants
INTERVAL_DEFAULTS = {
    'overhang': pd.Timedelta(hours=1),  # Default duration if no next measurement
    'max_len': pd.Timedelta(hours=6),   # Maximum interval length
}

# Duration padding for HiRID ventilation
HIRID_VENT_PADDING = pd.Timedelta(hours=4)
HIRID_VENT_CAP = pd.Timedelta(hours=12)

# ============================================================================
# Database-specific FiO2 Behavior
# ============================================================================

FIO2_DATABASE_BEHAVIOR = {
    'miiv': {
        'format': 'auto',  # Auto-detect (both fraction and percentage exist)
        'needs_conversion': True,
        'convert_threshold': 1.0,
    },
    'eicu': {
        'format': 'percentage',  # Always percentage (21-100)
        'needs_conversion': False,
    },
    'aumc': {
        'format': 'percentage',  # percent_as_numeric conversion
        'needs_conversion': False,
    },
    'hirid': {
        'format': 'auto',
        'needs_conversion': True,
        'convert_threshold': 1.0,
    },
    'sic': {
        'format': 'auto',
        'needs_conversion': True,
        'convert_threshold': 1.0,
    },
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_primary_id_column(database: str) -> str:
    """Get primary ID column for a database.
    
    Args:
        database: Database name
        
    Returns:
        Primary ID column name
    """
    if database not in ID_COLUMNS:
        raise ValueError(f"Unknown database: {database}")
    return ID_COLUMNS[database][0]

def get_all_id_columns(database: str) -> List[str]:
    """Get all ID columns for a database.
    
    Args:
        database: Database name
        
    Returns:
        List of ID column names
    """
    if database not in ID_COLUMNS:
        raise ValueError(f"Unknown database: {database}")
    return ID_COLUMNS[database].copy()

def get_time_column(database: str) -> str:
    """Get primary time column for a database.
    
    Args:
        database: Database name
        
    Returns:
        Time column name
    """
    if database not in TIME_COLUMNS:
        raise ValueError(f"Unknown database: {database}")
    return TIME_COLUMNS[database]

def is_offset_based(database: str) -> bool:
    """Check if database uses time offsets instead of absolute timestamps.
    
    Args:
        database: Database name
        
    Returns:
        True if database uses offsets (e.g., eICU)
    """
    return TIME_IS_OFFSET.get(database, False)

def get_fio2_behavior(database: str) -> Dict:
    """Get FiO2 handling behavior for a database.
    
    Args:
        database: Database name
        
    Returns:
        Dictionary with format and conversion information
    """
    return FIO2_DATABASE_BEHAVIOR.get(database, {
        'format': 'auto',
        'needs_conversion': True,
        'convert_threshold': 1.0,
    })
