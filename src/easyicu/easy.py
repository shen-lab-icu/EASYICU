"""Deprecated convenience wrappers retained for the EasyICU 1.x series.

Use the top-level :mod:`easyicu` API instead. This module is a small,
call-time-warning compatibility shim and is scheduled for removal in 2.0.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from .api import load_concepts

__all__ = [
    "load_vitals",
    "load_labs",
    "load_sofa_score",
    "load_sepsis",
    "load_custom",
    "quick_summary",
]

_REMOVAL_MESSAGE = (
    "easyicu.easy.{name}() is deprecated and will be removed in EasyICU 2.0; "
    "use {replacement} instead."
)


def _warn(name: str, replacement: str) -> None:
    warnings.warn(
        _REMOVAL_MESSAGE.format(name=name, replacement=replacement),
        DeprecationWarning,
        stacklevel=2,
    )


def _load(
    name: str,
    replacement: str,
    concepts: Union[str, List[str]],
    *,
    data_path: Union[str, Path],
    patient_ids: Optional[List[int]],
    database: str,
    interval_hours: float,
    **kwargs,
) -> pd.DataFrame:
    _warn(name, replacement)
    return load_concepts(
        concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=pd.Timedelta(hours=interval_hours),
        verbose=False,
        **kwargs,
    )


def load_vitals(
    data_path: Union[str, Path],
    patient_ids: Optional[List[int]] = None,
    database: str = "miiv",
    interval_hours: float = 1.0,
    concepts: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Deprecated wrapper for loading a legacy vital-sign bundle."""
    return _load(
        "load_vitals",
        "easyicu.load_vitals()",
        concepts or ["hr", "sbp", "dbp", "resp", "temp", "spo2"],
        data_path=data_path,
        patient_ids=patient_ids,
        database=database,
        interval_hours=interval_hours,
    )


def load_labs(
    data_path: Union[str, Path],
    patient_ids: Optional[List[int]] = None,
    database: str = "miiv",
    interval_hours: float = 6.0,
    concepts: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Deprecated wrapper for loading a legacy laboratory bundle."""
    return _load(
        "load_labs",
        "easyicu.load_labs()",
        concepts or ["wbc", "hgb", "plt", "crea", "bili", "lact"],
        data_path=data_path,
        patient_ids=patient_ids,
        database=database,
        interval_hours=interval_hours,
    )


def load_sofa_score(
    data_path: Union[str, Path],
    patient_ids: Optional[List[int]] = None,
    database: str = "miiv",
    interval_hours: float = 1.0,
    keep_components: bool = True,
) -> pd.DataFrame:
    """Deprecated wrapper for :func:`easyicu.load_sofa`."""
    return _load(
        "load_sofa_score",
        "easyicu.load_sofa()",
        "sofa",
        data_path=data_path,
        patient_ids=patient_ids,
        database=database,
        interval_hours=interval_hours,
        keep_components=keep_components,
    )


def load_sepsis(
    data_path: Union[str, Path],
    patient_ids: Optional[List[int]] = None,
    database: str = "miiv",
    interval_hours: float = 1.0,
    definition: str = "sepsis3",
) -> pd.DataFrame:
    """Deprecated wrapper for :func:`easyicu.load_sepsis3`."""
    concept_by_definition = {"sepsis3": "sep3", "sepsis2": "sep2"}
    try:
        concept = concept_by_definition[definition]
    except KeyError as exc:
        raise ValueError(f"Unknown sepsis definition: {definition}") from exc
    return _load(
        "load_sepsis",
        "easyicu.load_sepsis3()",
        concept,
        data_path=data_path,
        patient_ids=patient_ids,
        database=database,
        interval_hours=interval_hours,
    )


def load_custom(
    data_path: Union[str, Path],
    concepts: Union[str, List[str]],
    patient_ids: Optional[List[int]] = None,
    database: str = "miiv",
    interval_hours: float = 1.0,
) -> pd.DataFrame:
    """Deprecated wrapper for :func:`easyicu.load_concepts`."""
    return _load(
        "load_custom",
        "easyicu.load_concepts()",
        concepts,
        data_path=data_path,
        patient_ids=patient_ids,
        database=database,
        interval_hours=interval_hours,
    )


def quick_summary(
    data_path: Union[str, Path],
    patient_ids: Optional[List[int]] = None,
    database: str = "miiv",
) -> dict:
    """Return the legacy best-effort summary plus explicit failure metadata."""
    _warn("quick_summary", "explicit easyicu.load_*() calls")
    errors: dict[str, str] = {}
    loaded: dict[str, pd.DataFrame] = {}
    for label, concepts in {
        "vitals": ["hr"],
        "labs": ["wbc"],
        "sofa": "sofa",
        "sepsis": "sep3",
    }.items():
        try:
            loaded[label] = load_concepts(
                concepts,
                patient_ids=patient_ids,
                database=database,
                data_path=data_path,
                verbose=False,
            )
        except Exception as exc:  # legacy summary is intentionally best effort
            errors[label] = type(exc).__name__
            loaded[label] = pd.DataFrame()

    vitals, labs = loaded["vitals"], loaded["labs"]
    sofa, sepsis = loaded["sofa"], loaded["sepsis"]
    return {
        "patients": len(patient_ids) if patient_ids else "all",
        "vitals_records": len(vitals),
        "lab_records": len(labs),
        "sofa_mean": sofa["sofa"].mean() if "sofa" in sofa.columns else None,
        "sepsis_positive": sepsis["sep3"].sum() if "sep3" in sepsis.columns else 0,
        "errors": errors,
    }
