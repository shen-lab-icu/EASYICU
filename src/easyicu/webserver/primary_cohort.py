"""Canonical execution-level primary-cohort contract.

Owner: Data Extraction primary-cohort semantics.
Public contract: normalize every StudyContext cohort spelling to the exact
population, admission, diagnosis, phenotype-window, and sampling axes that
can change the executed queue.  Persistence, Copilot consent, workflow gates,
and Data Extraction consume this owner instead of maintaining field rosters.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from easyicu.webserver.input_validation import parse_bool


SCHEMA_VERSION = "easyicu.normalized-primary-cohort-scope/1"
DEFAULT_OBSERVATION_WINDOW_HOURS = 24 * 30
#: Largest diagnosis roster Data Extraction will execute as one predicate.
#: Exceeding it is refused, never trimmed: a silently shortened roster runs a
#: different cohort than the researcher stated, and nothing downstream can
#: tell that it happened.
MAX_DIAGNOSIS_TOKENS = 64
SUPPORTED_COHORT_PRESETS: Tuple[str, ...] = (
    "adult_all",
    "adult_first",
    "aki",
    "all_icu",
    "icd",
    "respiratory",
    "sepsis3",
    "vasopressor",
    "ventilation",
)
CONCEPT_DERIVED_PRESETS = frozenset(
    {"aki", "respiratory", "sepsis3", "vasopressor", "ventilation"}
)
ADMISSION_ELIGIBILITY_FIELDS: Tuple[str, ...] = (
    "age_min",
    "age_max",
    "min_icu_los_hours",
    "exclude_readmissions",
)
DIAGNOSIS_ELIGIBILITY_FIELDS: Tuple[str, ...] = (
    "icd_enabled",
    "icd_include",
    "icd_exclude",
    "include_diagnoses",
    "exclude_diagnoses",
)


class PrimaryCohortContractError(ValueError):
    """Typed failure for a cohort that Data Extraction cannot execute."""

    def __init__(self, code: str, detail: Optional[Mapping[str, Any]] = None) -> None:
        self.code = str(code)
        self.detail = dict(detail or {})
        super().__init__(self.code)


def normalize_preset(value: Any) -> str:
    preset = str(value or "").strip().lower()
    if preset not in SUPPORTED_COHORT_PRESETS:
        raise PrimaryCohortContractError(
            "unsupported_cohort_preset",
            {"preset": preset, "supported": list(SUPPORTED_COHORT_PRESETS)},
        )
    return preset


def _coerce_int(
    value: Any,
    default: int,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    try:
        result = int(float(value))
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def normalize_diagnosis_tokens(raw: Any) -> Tuple[str, ...]:
    """Normalize the diagnosis aliases exactly once for execution and consent."""

    tokens: list[str] = []
    values = raw if isinstance(raw, (list, tuple, set)) else [raw]
    for value in values:
        text = str(value or "").upper().replace("，", ",").replace("；", ";")
        for part in re.split(r"[\s,;]+", text):
            token = part.strip().replace(".", "")
            if not token:
                continue
            if "-" in token:
                start, end = [item.strip() for item in token.split("-", 1)]
                if (
                    len(start) == len(end)
                    and len(start) >= 2
                    and start[:-2] == end[:-2]
                    and start[-2:].isdigit()
                    and end[-2:].isdigit()
                ):
                    prefix = start[:-2]
                    lower, upper = int(start[-2:]), int(end[-2:])
                    if 0 <= upper - lower <= 50:
                        tokens.extend(
                            f"{prefix}{index:02d}"
                            for index in range(lower, upper + 1)
                        )
                        continue
            tokens.append(token)
    unique = tuple(dict.fromkeys(tokens))
    if len(unique) > MAX_DIAGNOSIS_TOKENS:
        # A stated range expands here: "I20-I52, J09-J18, E10-E50" is three
        # criteria and 84 codes. Trimming to the cap dropped E31-E50 from the
        # executed queue with no error and no finding, so the cohort that ran
        # was not the cohort that was stated.
        raise PrimaryCohortContractError(
            "diagnosis_filter_too_large",
            {
                "token_count": len(unique),
                "max_tokens": MAX_DIAGNOSIS_TOKENS,
                "dropped_tokens": list(unique[MAX_DIAGNOSIS_TOKENS:]),
            },
        )
    return unique


def normalize_execution_cohort(cohort: Any) -> Dict[str, Any]:
    """Return the flat cohort contract consumed by Data Extraction."""

    raw = dict(cohort) if isinstance(cohort, Mapping) else {}
    # Metadata such as a label, review note, or sampling cap must not silently
    # choose an adult/first-stay population. Admission restrictions are neutral
    # unless their own axis or an explicit legacy preset states them.
    preset = normalize_preset(raw.get("preset") or "all_icu")
    include = normalize_diagnosis_tokens(
        raw.get("icd_include")
        or raw.get("include_diagnoses")
        or raw.get("include")
        or ""
    )
    exclude = normalize_diagnosis_tokens(
        raw.get("icd_exclude")
        or raw.get("exclude_diagnoses")
        or raw.get("exclude")
        or ""
    )
    try:
        # A structured include/exclude criterion is itself an executable
        # diagnosis predicate.  Requiring a second alias-specific boolean
        # would let StudyContext and Data Extraction disagree about the same
        # cohort definition.
        icd_enabled = (
            parse_bool(raw.get("icd_enabled"), default=False)
            or preset == "icd"
            or bool(include or exclude)
        )
        exclude_readmissions = parse_bool(
            raw.get("exclude_readmissions"),
            default=preset == "adult_first",
        )
    except ValueError as exc:
        raise PrimaryCohortContractError(
            "invalid_cohort_boolean",
            {"fields": ["icd_enabled", "exclude_readmissions"]},
        ) from exc
    if not icd_enabled:
        include = ()
        exclude = ()
    if preset == "icd" and not include and not exclude:
        raise PrimaryCohortContractError("empty_icd_filter", {"preset": preset})

    age_min = _coerce_int(
        raw.get("age_min"),
        18 if preset == "adult_first" else 0,
        0,
        120,
    )
    age_max = _coerce_int(raw.get("age_max"), 100, 0, 120)
    if age_min > age_max:
        age_min, age_max = age_max, age_min
    if preset == "adult_first":
        age_min = max(18, age_min)
    min_los = _coerce_int(raw.get("min_icu_los_hours"), 0, 0, 24 * 30)
    observation_window = _coerce_int(
        raw.get("observation_window_hours"),
        DEFAULT_OBSERVATION_WINDOW_HOURS,
        1,
        DEFAULT_OBSERVATION_WINDOW_HOURS,
    )
    return {
        "preset": preset,
        "age_min": age_min,
        "age_max": age_max,
        "min_icu_los_hours": min_los,
        "observation_window_hours": observation_window,
        "exclude_readmissions": exclude_readmissions,
        "icd_enabled": icd_enabled,
        "icd_include": list(include),
        "icd_exclude": list(exclude),
    }


def _canonical_mapping(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    decoded = json.loads(encoded)
    return decoded if isinstance(decoded, dict) else {}


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value


@dataclass(frozen=True)
class NormalizedPrimaryCohortScope:
    """Immutable scientific coordinates for the executed primary queue."""

    population: Mapping[str, Any]
    admission_eligibility: Mapping[str, Any]
    diagnosis_eligibility: Mapping[str, Any]
    phenotype_window: Mapping[str, Any]
    sampling: Mapping[str, Any]
    selection_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "population": _thaw_value(self.population),
            "admission_eligibility": _thaw_value(self.admission_eligibility),
            "diagnosis_eligibility": _thaw_value(self.diagnosis_eligibility),
            "phenotype_window": _thaw_value(self.phenotype_window),
            "sampling": _thaw_value(self.sampling),
            "selection_mode": self.selection_mode,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def stated_fields(self) -> Tuple[str, ...]:
        fields: list[str] = []
        if self.population.get("kind") != "all_icu":
            fields.append("population")
        admission = self.admission_eligibility
        if int(admission.get("minimum_age_years") or 0) > 0:
            fields.append("admission_eligibility.minimum_age_years")
        if int(admission.get("maximum_age_years") or 100) < 100:
            fields.append("admission_eligibility.maximum_age_years")
        if int(admission.get("minimum_icu_duration_hours") or 0) > 0:
            fields.append("admission_eligibility.minimum_icu_duration_hours")
        if admission.get("repeated_admission_policy") == "first_icu_admission_only":
            fields.append("admission_eligibility.repeated_admission_policy")
        diagnosis = self.diagnosis_eligibility
        if diagnosis.get("enabled"):
            if diagnosis.get("include"):
                fields.append("diagnosis_eligibility.include")
            if diagnosis.get("exclude"):
                fields.append("diagnosis_eligibility.exclude")
        if self.phenotype_window:
            fields.append("phenotype_window")
        if self.sampling.get("status") == "capped":
            fields.append("sampling.max_patients")
        return tuple(fields)


def normalize_primary_cohort_scope(
    cohort: Any,
    *,
    max_patients: Any = None,
) -> NormalizedPrimaryCohortScope:
    """Compile the canonical denominator contract from one StudyContext cohort."""

    raw = dict(cohort) if isinstance(cohort, Mapping) else {}
    execution = normalize_execution_cohort(raw)
    preset = str(execution["preset"])
    if preset in {"adult_first", "adult_all"}:
        population = {"kind": "all_icu", "definition": "all_bound_icu_stays"}
    elif preset in CONCEPT_DERIVED_PRESETS:
        population = {"kind": "concept_derived", "definition": preset}
    elif preset == "icd":
        population = {"kind": "diagnosis_defined", "definition": "icd"}
    else:
        population = {"kind": "all_icu", "definition": "all_bound_icu_stays"}

    repeated_policy = (
        "first_icu_admission_only"
        if execution["exclude_readmissions"]
        else "all_icu_admissions"
    )
    admission = {
        "minimum_age_years": int(execution["age_min"]),
        "maximum_age_years": int(execution["age_max"]),
        "minimum_icu_duration_hours": int(execution["min_icu_los_hours"]),
        "repeated_admission_policy": repeated_policy,
    }
    diagnosis = {
        "enabled": bool(execution["icd_enabled"]),
        "include": tuple(execution["icd_include"]),
        "exclude": tuple(execution["icd_exclude"]),
    }
    phenotype: Dict[str, Any] = {}
    if preset in CONCEPT_DERIVED_PRESETS:
        phenotype = {
            "definition": preset,
            "observation_window_hours": int(execution["observation_window_hours"]),
        }
        if preset == "sepsis3" and isinstance(raw.get("sepsis_definition"), Mapping):
            phenotype["definition_parameters"] = _canonical_mapping(
                raw.get("sepsis_definition")
            )

    cap_source = raw.get("max_patients") if max_patients is None else max_patients
    cap = _coerce_int(cap_source, 0, 0, None)
    sampling = (
        {"status": "capped", "max_patients": cap}
        if cap
        else {"status": "uncapped", "max_patients": None}
    )
    predicate_filtered = bool(
        population["kind"] != "all_icu"
        or admission["minimum_age_years"] > 0
        or admission["maximum_age_years"] < 100
        or admission["minimum_icu_duration_hours"] > 0
        or repeated_policy == "first_icu_admission_only"
        or diagnosis["enabled"]
    )
    return NormalizedPrimaryCohortScope(
        population=_freeze_value(population),
        admission_eligibility=_freeze_value(admission),
        diagnosis_eligibility=_freeze_value(diagnosis),
        phenotype_window=_freeze_value(phenotype),
        sampling=_freeze_value(sampling),
        selection_mode="predicate_filtered" if predicate_filtered else "all_input_rows",
    )


__all__ = [
    "ADMISSION_ELIGIBILITY_FIELDS",
    "CONCEPT_DERIVED_PRESETS",
    "DEFAULT_OBSERVATION_WINDOW_HOURS",
    "DIAGNOSIS_ELIGIBILITY_FIELDS",
    "MAX_DIAGNOSIS_TOKENS",
    "NormalizedPrimaryCohortScope",
    "PrimaryCohortContractError",
    "SCHEMA_VERSION",
    "SUPPORTED_COHORT_PRESETS",
    "normalize_diagnosis_tokens",
    "normalize_execution_cohort",
    "normalize_preset",
    "normalize_primary_cohort_scope",
]
