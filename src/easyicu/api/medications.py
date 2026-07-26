"""Medication-domain loading and merge contracts.

This module owns medication catalog selection and partial-result semantics.
The public :mod:`easyicu.api` facade injects the canonical concept loader and
validator so legacy monkeypatching and call signatures remain compatible.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import pandas as pd


MEDICATION_GROUPS: Mapping[str, tuple[str, ...]] = {
    "vasopressors": (
        "adh_rate",
        "dobu_dur",
        "dobu_rate",
        "dobu60",
        "dopa_dur",
        "dopa_rate",
        "epi_dur",
        "epi_rate",
        "norepi_dur",
        "norepi_equiv",
        "norepi_rate",
        "phn_rate",
        "vaso_ind",
    ),
    "sedation": (
        "propofol",
        "propofol_rate",
        "midazolam",
        "midazolam_rate",
        "dexmedetomidine",
        "lorazepam",
        "ketamine",
    ),
    "analgesia": ("fentanyl", "fentanyl_rate", "morphine"),
    "neuromuscular": ("rocuronium", "vecuronium", "cisatracurium"),
    "antibiotics": ("abx", "vancomycin", "meropenem"),
    "cardiac": ("amiodarone", "milrinone"),
    "diuretics": ("furosemide", "mannitol"),
    "anticoagulation": ("heparin", "warfarin", "apixaban", "enoxaparin"),
    "antiplatelet": ("aspirin",),
    "endocrine": ("cort", "ins", "insulin"),
    "vasodilators": ("nitroglycerin",),
    "gi_prophylaxis": ("pantoprazole",),
    "electrolytes": ("calcium_iv", "potassium_iv", "magnesium_iv", "bicarbonate"),
    "colloids_blood": ("albumin_iv", "packed_rbc", "ffp", "platelets"),
    "neurology": ("levetiracetam",),
    "gi": ("octreotide",),
    "reversal": ("neostigmine",),
    "corticosteroids": ("cort", "dexamethasone"),
    "other": ("dex", "dextrose50"),
}

LEGACY_MEDICATION_CONCEPTS: tuple[str, ...] = (
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
)


class MedicationLoadError(RuntimeError):
    """Raised when a medication bundle would otherwise return partial data."""

    def __init__(self, report: Dict[str, object]):
        self.report = report
        failures = report.get("failed", {})
        failed_names = ", ".join(sorted(failures)) if isinstance(failures, dict) else ""
        super().__init__(
            "Medication loading was incomplete"
            + (f" for: {failed_names}" if failed_names else "")
            + ". Pass allow_partial=True only when a partial result is intentional."
        )


class MedicationMergeError(ValueError):
    """Raised when independently loaded medication frames cannot be merged safely."""


def select_medication_concepts(
    *,
    groups: Optional[Union[str, Sequence[str]]],
    include_new: bool,
) -> List[str]:
    """Resolve a stable, duplicate-free medication concept list."""
    if groups is not None:
        requested_groups = [groups] if isinstance(groups, str) else list(groups)
        unknown = set(requested_groups) - set(MEDICATION_GROUPS)
        if unknown:
            raise ValueError(
                f"Unknown medication group(s): {sorted(unknown)}. "
                f"Valid groups: {sorted(MEDICATION_GROUPS)}"
            )
        source_groups = [MEDICATION_GROUPS[group] for group in requested_groups]
    elif include_new:
        source_groups = list(MEDICATION_GROUPS.values())
    else:
        return list(LEGACY_MEDICATION_CONCEPTS)

    concepts: List[str] = []
    seen: set[str] = set()
    for group_concepts in source_groups:
        for concept in group_concepts:
            if concept not in seen:
                concepts.append(concept)
                seen.add(concept)
    return concepts


def merge_medication_frames(
    frames: List[pd.DataFrame],
    concepts: List[str],
) -> pd.DataFrame:
    """Merge concept frames without heuristic many-to-many multiplication."""
    if not frames:
        return pd.DataFrame()

    id_candidates = (
        "stay_id",
        "icustay_id",
        "subject_id",
        "hadm_id",
        "patientunitstayid",
        "admissionid",
        "patientid",
        "patient_id",
        "CaseID",
        "caseid",
    )
    time_candidates = (
        "charttime",
        "starttime",
        "endtime",
        "time",
        "datetime",
        "timestamp",
        "observationoffset",
        "chartoffset",
        "eventtime",
        "realtime",
    )

    merged = frames[0].copy()
    merged_concepts = [concepts[0]]
    for concept, frame in zip(concepts[1:], frames[1:]):
        shared_ids = [
            column
            for column in id_candidates
            if column in merged.columns and column in frame.columns
        ]
        shared_times = [
            column
            for column in time_candidates
            if column in merged.columns and column in frame.columns
        ]
        merge_columns = shared_ids + shared_times
        if not shared_ids:
            raise MedicationMergeError(
                "Cannot safely merge medication concepts "
                f"{merged_concepts!r} and {concept!r}: no shared patient/stay ID column."
            )
        if merged.duplicated(merge_columns).any() or frame.duplicated(
            merge_columns
        ).any():
            raise MedicationMergeError(
                "Cannot safely merge medication concepts "
                f"{merged_concepts!r} and {concept!r}: merge keys {merge_columns!r} "
                "are not unique, so an outer merge could multiply rows."
            )

        merged = pd.merge(
            merged,
            frame,
            on=merge_columns,
            how="outer",
            validate="one_to_one",
        )
        merged_concepts.append(concept)
    return merged


def load_medications_impl(
    *,
    load_concepts_fn: Callable[..., pd.DataFrame],
    validate_concepts_fn: Callable[[List[str], bool], List[str]],
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = "1h",
    win_length: Union[str, pd.Timedelta] = "24h",
    verbose: bool = False,
    groups: Optional[Union[str, List[str]]] = None,
    include_new: bool = True,
    allow_partial: bool = False,
) -> pd.DataFrame:
    """Load a medication bundle through injected concept-domain services."""
    if verbose:
        print("💊 加载药物治疗数据...")

    concepts = select_medication_concepts(groups=groups, include_new=include_new)
    available_concepts = validate_concepts_fn(concepts, verbose)
    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    results: List[pd.DataFrame] = []
    loaded_concepts: List[str] = []
    failed_concepts: Dict[str, Dict[str, str]] = {}
    for concept in available_concepts:
        try:
            frame = load_concepts_fn(
                concepts=[concept],
                patient_ids=patient_ids,
                database=database,
                data_path=data_path,
                interval=interval,
                win_length=win_length,
                merge=True,
                verbose=False,
            )
            if frame is not None and not frame.empty:
                results.append(frame)
                loaded_concepts.append(concept)
            else:
                failed_concepts[concept] = {"reason": "empty_result"}
        except Exception as exc:
            failed_concepts[concept] = {
                "reason": "load_error",
                "error_type": type(exc).__name__,
            }

    report: Dict[str, object] = {
        "requested": list(concepts),
        "validated": list(available_concepts),
        "loaded": list(loaded_concepts),
        "failed": failed_concepts,
    }
    if failed_concepts and not allow_partial:
        raise MedicationLoadError(report)
    if failed_concepts:
        warnings.warn(
            "Medication loading returned an explicitly allowed partial result; "
            f"failed concepts: {sorted(failed_concepts)}",
            RuntimeWarning,
            stacklevel=3,
        )

    if not results:
        if verbose:
            print("  ❌ 没有成功加载的概念")
        empty = pd.DataFrame()
        empty.attrs["easyicu_medication_load_report"] = report
        return empty

    if verbose:
        print(f"  ✅ 成功加载 {len(loaded_concepts)} 个概念: {loaded_concepts}")
    merged = merge_medication_frames(results, loaded_concepts)
    merged.attrs["easyicu_medication_load_report"] = report
    return merged


__all__ = [
    "LEGACY_MEDICATION_CONCEPTS",
    "MEDICATION_GROUPS",
    "MedicationLoadError",
    "MedicationMergeError",
    "load_medications_impl",
    "merge_medication_frames",
    "select_medication_concepts",
]
