"""
Feature module configuration for pyricu.

This module defines the mapping between ricu CSV exports and pyricu concepts.
Used by verification scripts to compare pyricu output with ricu reference data.

Extracted from feature_compare.py to provide a clean, shared configuration.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class FeatureModule:
    """Metadata describing a ricu CSV <-> pyricu concept family."""

    name: str
    ricu_file: str
    concepts: List[str]
    id_column: str = "stay_id"
    time_column: Optional[str] = "charttime"
    description: str = ""


# ============================================================================
# Module Definitions
# ============================================================================

MODULES: List[FeatureModule] = [
    FeatureModule(
        name="outcome",
        ricu_file="{db}_outcome.csv",
        concepts=[
            "sofa",
            "sofa_resp",
            "sofa_coag",
            "sofa_liver",
            "sofa_cardio",
            "sofa_cns",
            "sofa_renal",
            "qsofa",
            "sirs",
            "death",
            "los_icu",
        ],
        time_column="index_var",
        description="SOFA outcome timeline",
    ),
    FeatureModule(
        name="resp",
        ricu_file="{db}_resp.csv",
        concepts=[
            "pafi",
            "safi",
            "resp",
            "supp_o2",
            "vent_ind",
            "o2sat",
            "sao2",
            "mech_vent",
            "ett_gcs",
            # Note: fio2 is NOT in ricu's resp CSV, it's in the blood module
        ],
        description="Respiratory chain (PaFi/SaFi/Vent indicators)",
    ),
    FeatureModule(
        name="blood",
        ricu_file="{db}_blood.csv",
        concepts=["be", "cai", "fio2", "hbco", "lact", "methb", "pco2", "ph", "po2", "tco2"],
        description="Arterial blood gas",
    ),
    FeatureModule(
        name="lab",
        ricu_file="{db}_lab.csv",
        concepts=[
            "alb",
            "alp",
            "alt",
            "ast",
            "bicar",
            "bili",
            "bili_dir",
            "bun",
            "ca",
            "ck",
            "ckmb",
            "cl",
            "crea",
            "crp",
            "glu",
            "k",
            "mg",
            "na",
            "phos",
            "tnt",
        ],
        description="General chemistry",
    ),
    FeatureModule(
        name="hematology",
        ricu_file="{db}_hematology.csv",
        concepts=["bnd", "esr", "fgn", "hgb", "inr_pt", "lymph", "mch", "mchc", "mcv", "neut", "plt", "ptt", "wbc"],
        description="Heme/Coag profile",
    ),
    FeatureModule(
        name="med",
        ricu_file="{db}_med.csv",
        concepts=[
            "abx",
            "adh_rate",
            "cort",
            "dex",
            "dobu_dur",
            "dobu_rate",
            "dobu60",
            "epi_dur",
            "epi_rate",
            "ins",
            "norepi_dur",
            "norepi_equiv",
            "norepi_rate",
            "vaso_ind",
        ],
        time_column="starttime",
        description="Vasopressors / medications",
    ),
    FeatureModule(
        name="vital",
        ricu_file="{db}_vital.csv",
        concepts=["dbp", "etco2", "hr", "map", "sbp", "temp"],
        description="Bedside vitals",
    ),
    FeatureModule(
        name="output",
        ricu_file="{db}_output.csv",
        concepts=["urine", "urine24"],
        description="Urine output",
    ),
    FeatureModule(
        name="neu",
        ricu_file="{db}_neu.csv",
        concepts=["avpu", "egcs", "gcs", "mgcs", "rass", "vgcs"],
        description="Neurologic assessments",
    ),
    FeatureModule(
        name="demo",
        ricu_file="{db}_demo.csv",
        concepts=["age", "bmi", "height", "sex", "weight"],
        time_column=None,
        description="Demographics",
    ),
]

# Module lookup by name
MODULES_BY_NAME: Dict[str, FeatureModule] = {m.name: m for m in MODULES}

# All concept names across all modules
ALL_CONCEPTS: List[str] = [c for m in MODULES for c in m.concepts]

# ============================================================================
# SOFA Component Dependencies
# ============================================================================

SOFA_COMPONENT_DEPENDENCIES: Dict[str, List[str]] = {
    "sofa_resp": ["pafi", "safi", "vent_ind", "supp_o2", "fio2", "resp", "o2sat", "sao2"],
    "sofa_coag": ["plt", "inr_pt", "ptt"],
    "sofa_liver": ["bili", "bili_dir", "alp", "ast", "alt"],
    "sofa_cardio": ["map", "vaso_ind", "norepi_rate", "dobu_rate", "epi_rate", "adh_rate"],
    "sofa_cns": ["gcs", "mgcs", "rass", "avpu"],
    "sofa_renal": ["crea", "bun", "urine", "urine24"],
}

# ============================================================================
# Supported Databases
# ============================================================================

SUPPORTED_DATABASES = ("miiv", "eicu", "aumc", "hirid")

# ID column names for each database
DATABASE_ID_COLUMNS: Dict[str, str] = {
    "miiv": "stay_id",
    "eicu": "patientunitstayid",
    "aumc": "admissionid",
    "hirid": "patientid",
}

# Common ID column candidates (for auto-detection)
ID_CANDIDATES = ["stay_id", "subject_id", "patientunitstayid", "admissionid", "patientid"]

# Common time column candidates (for auto-detection)
TIME_CANDIDATES = ["charttime", "index_var", "starttime", "time", "observationoffset", "labresultoffset"]
