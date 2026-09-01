"""Versioned source-native AKI phenotype profiles.

The canonical EasyICU phenotype remains :func:`kdigo_stages`.  This module
adds explicitly selected compatibility profiles for database-maintainer or
study-author implementations.  A source-native profile never replaces the
canonical columns and never silently turns an unavailable implementation into
an EasyICU-defined AKI stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import json
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from .kdigo_aki import (
    KDIGOComponentSchemaError,
    _detect_id_col,
    _detect_time_col,
    _detect_value_col,
    _resolve_time_unit,
    _strict_numeric_component,
    kdigo_creatinine,
    kdigo_stages,
    kdigo_uo,
)


_REGISTRY_FILE = "aki-profile-registry.json"
_PROFILE_PREFIX = "aki_source_native"
_REFERENCE_PROFILE_ID = "MIT_LCP_KDIGO_REFERENCE_PORT_V1"

RENAL_AKI_BUNDLE_OUTPUTS = (
    "aki_reference",
    "aki_stage_reference",
    "aki_severe_reference",
    "aki_stage_creat_reference",
    "aki_stage_uo_reference",
    "aki_stage_rrt_reference",
    "aki_stage_reference_smoothed_6h",
    "aki_reference_profile",
    "aki_reference_status",
    "aki_reference_authority",
    "aki_reference_reliability_grade",
    "aki_reference_fidelity",
    "aki_reference_rrt_scope",
    "aki_reference_uses_future",
    "aki_source_native",
    "aki_stage_source_native",
    "aki_severe_source_native",
    "aki_stage_creat_source_native",
    "aki_stage_uo_source_native",
    "aki_stage_crrt_source_native",
    "aki_stage_source_native_smoothed",
    "aki_source_native_profile",
    "aki_source_native_status",
    "aki_source_native_authority",
    "aki_source_native_reliability_grade",
    "aki_source_native_fidelity",
    "aki_source_native_ascertainment",
    "aki_source_native_reason",
    "aki_source_native_time_scale",
    "aki_source_native_uses_future",
    "kidney_observation_window_coverage",
    "creatinine_evidence_status",
    "creatinine_evidence_reason",
    "urine_evidence_status",
    "rrt_evidence_status",
    "creat_low_past_48hr",
    "creat_low_past_7day",
    "creat_baseline_n_48h",
    "creat_baseline_n_7d",
    "creat_baseline_source",
    "creat_pre_icu_history_observed",
    "uo_rt_6hr",
    "uo_rt_12hr",
    "uo_rt_24hr",
)


class AKIProfileError(ValueError):
    """Base error for an invalid source-native profile request."""


class AKIProfilePrerequisiteError(AKIProfileError):
    """Raised when an exact source-native profile lacks required inputs."""


@dataclass(frozen=True)
class AKIProfile:
    """One immutable profile definition from the packaged registry."""

    profile_id: str
    database: str
    authority_level: str
    reliability_grade: str
    fidelity: str
    output_kind: str
    recommended_role: str
    payload: Mapping[str, Any]


def load_aki_profile_registry() -> Mapping[str, Any]:
    """Return the packaged, versioned AKI profile registry."""

    resource = resources.files("easyicu.data").joinpath(_REGISTRY_FILE)
    payload = json.loads(resource.read_text(encoding="utf8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("profiles"), dict):
        raise RuntimeError("Packaged AKI profile registry is malformed")
    return payload


def list_aki_profiles(database: Optional[str] = None) -> tuple[AKIProfile, ...]:
    """List profiles, optionally restricted to a database or alias."""

    profiles = tuple(
        _profile_from_payload(profile_id, payload)
        for profile_id, payload in load_aki_profile_registry()["profiles"].items()
    )
    if database is None:
        return profiles
    needle = _normalize_database(database)
    return tuple(
        profile
        for profile in profiles
        if profile.database == "all"
        or needle
        in {
            _normalize_database(profile.database),
            *{
                _normalize_database(alias)
                for alias in profile.payload.get("aliases", [])
            },
        }
    )


def get_aki_profile(profile_id: str) -> AKIProfile:
    """Resolve one profile by its stable identifier."""

    payload = load_aki_profile_registry()["profiles"].get(profile_id)
    if payload is None:
        known = ", ".join(sorted(load_aki_profile_registry()["profiles"]))
        raise KeyError(f"Unknown AKI profile {profile_id!r}; known profiles: {known}")
    return _profile_from_payload(profile_id, payload)


def default_source_native_profile(database: str) -> AKIProfile:
    """Return the single non-harmonized profile registered for ``database``."""

    candidates = tuple(
        profile for profile in list_aki_profiles(database) if profile.database != "all"
    )
    if len(candidates) != 1:
        raise AKIProfileError(
            f"Expected one source-native profile for {database!r}, found "
            f"{[profile.profile_id for profile in candidates]!r}"
        )
    return candidates[0]


def apply_source_native_aki(
    database: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """Apply the frozen source-native profile registered for ``database``.

    This convenience entry point is intentionally explicit about the requested
    database.  It does not replace :func:`kdigo_stages`, alter EasyICU's
    canonical columns, or fall back to the harmonized phenotype when the
    source-native implementation is unavailable.
    """

    profile = default_source_native_profile(database)
    return apply_aki_profile(profile.profile_id, **kwargs)


def apply_reference_aki(**kwargs: Any) -> pd.DataFrame:
    """Apply the registry-selected cross-database public reference profile."""

    profile_id = str(load_aki_profile_registry()["default_profile"])
    if profile_id != _REFERENCE_PROFILE_ID:
        raise AKIProfileError(
            "The registry default is not the supported public reference profile: "
            f"{profile_id!r}"
        )
    return apply_aki_profile(profile_id, **kwargs)


def build_renal_aki_bundle(
    database: str,
    *,
    crea_df: Optional[pd.DataFrame] = None,
    urine_df: Optional[pd.DataFrame] = None,
    weight_df: Optional[pd.DataFrame] = None,
    rrt_df: Optional[pd.DataFrame] = None,
    crrt_df: Optional[pd.DataFrame] = None,
    id_col: Optional[str] = None,
    time_col: Optional[str] = None,
    crea_col: str = "crea",
    urine_col: str = "urine",
    weight_col: str = "weight",
    urine_source_is_rate: bool = False,
    time_unit: Optional[str] = None,
    interval: Optional[pd.Timedelta] = None,
    observation_window_coverage: Optional[Mapping[Any, str]] = None,
    rrt_source_complete: bool = False,
) -> pd.DataFrame:
    """Build the renal export's reference, quality, and native AKI layers.

    Future-looking case-level native profiles (AUMC and SICdb) remain in the
    explicit profile API but are not broadcast onto a dynamic hourly table.
    This prevents a first-168-hour endpoint from silently becoming an early
    predictor.  Their profile and non-embedding reason are still exported.
    """

    normalized_database = _normalize_database(database)
    common_kwargs = {
        "crea_df": crea_df,
        "urine_df": urine_df,
        "weight_df": weight_df,
        "rrt_df": rrt_df,
        "id_col": id_col,
        "time_col": time_col,
        "crea_col": crea_col,
        "urine_col": urine_col,
        "weight_col": weight_col,
        "time_unit": time_unit,
    }
    reference = apply_reference_aki(**common_kwargs)
    resolved_id, resolved_time = _component_keys(
        (crea_df, urine_df, rrt_df), id_col, time_col
    )

    # Keep evidence receipts, but do not publish the historical strict disease
    # label or its assessable/non-assessable cohort split.
    quality = kdigo_stages(
        crea_df=crea_df,
        urine_df=urine_df,
        weight_df=weight_df,
        rrt_df=rrt_df,
        id_col=resolved_id,
        time_col=resolved_time,
        crea_col=crea_col,
        urine_col=urine_col,
        weight_col=weight_col,
        urine_source_is_rate=urine_source_is_rate,
        time_unit=time_unit,
        interval=interval,
        observation_window_coverage=observation_window_coverage,
        rrt_source_complete=rrt_source_complete,
    )
    quality_columns = {
        "observation_window_coverage": "kidney_observation_window_coverage",
        "creatinine_ascertainment": "creatinine_evidence_status",
        "creatinine_ascertainment_reason": "creatinine_evidence_reason",
        "urine_ascertainment": "urine_evidence_status",
        "rrt_ascertainment": "rrt_evidence_status",
        "creat_low_past_48hr": "creat_low_past_48hr",
        "creat_low_past_7day": "creat_low_past_7day",
        "creat_baseline_n_48h": "creat_baseline_n_48h",
        "creat_baseline_n_7d": "creat_baseline_n_7d",
        "creat_baseline_source": "creat_baseline_source",
        "creat_pre_icu_history_observed": "creat_pre_icu_history_observed",
        "uo_rt_6hr": "uo_rt_6hr",
        "uo_rt_12hr": "uo_rt_12hr",
        "uo_rt_24hr": "uo_rt_24hr",
    }
    available_quality = [
        column for column in quality_columns if column in quality.columns
    ]
    result = reference.merge(
        quality[[resolved_id, resolved_time, *available_quality]].rename(
            columns=quality_columns
        ),
        on=[resolved_id, resolved_time],
        how="outer",
        validate="one_to_one",
        sort=False,
    )

    native_profile = default_source_native_profile(normalized_database)
    if normalized_database in {"aumc", "sic"}:
        result["aki_stage_source_native"] = pd.Series(
            pd.NA, index=result.index, dtype="Int64"
        )
        result["aki_source_native"] = pd.Series(
            pd.NA, index=result.index, dtype="boolean"
        )
        result["aki_source_native_profile"] = native_profile.profile_id
        result["aki_source_native_status"] = (
            "case_level_future_endpoint_not_embedded_in_dynamic_renal"
        )
        result["aki_source_native_authority"] = native_profile.authority_level
        result["aki_source_native_reliability_grade"] = (
            native_profile.reliability_grade
        )
        result["aki_source_native_fidelity"] = native_profile.fidelity
        result["aki_source_native_ascertainment"] = "not_embedded"
        result["aki_source_native_reason"] = (
            "source-native profile remains available through apply_aki_profile; "
            "it is not repeated across earlier hourly rows"
        )
    else:
        native_kwargs = dict(common_kwargs)
        if normalized_database == "miiv":
            native_kwargs["rrt_df"] = crrt_df
            if crrt_df is None:
                native = _unavailable_profile_frame(
                    crea_df,
                    urine_df,
                    rrt_df,
                    id_col=resolved_id,
                    time_col=resolved_time,
                    reason="mimic_iv_crrt_mode_source_unavailable",
                )
                native = _stamp_profile(
                    native,
                    native_profile,
                    status="not_evaluable_required_crrt_source_missing",
                    ascertainment="indeterminate",
                )
            else:
                native = apply_source_native_aki(
                    normalized_database, **native_kwargs
                )
        else:
            native = apply_source_native_aki(
                normalized_database, **native_kwargs
            )
        result = result.merge(
            native,
            on=[resolved_id, resolved_time],
            how="outer",
            validate="one_to_one",
            sort=False,
        )
        for column in (
            "aki_source_native_profile",
            "aki_source_native_status",
            "aki_source_native_authority",
            "aki_source_native_reliability_grade",
            "aki_source_native_fidelity",
        ):
            if column in result:
                result[column] = result[column].ffill().bfill()

    result["aki_source_native_time_scale"] = native_profile.output_kind
    result["aki_source_native_uses_future"] = bool(
        native_profile.payload["uses_future_information"]
    )
    if "aki_stage_source_native" not in result:
        result["aki_stage_source_native"] = pd.Series(
            pd.NA, index=result.index, dtype="Int64"
        )
    native_stage = result["aki_stage_source_native"].astype("Int64")
    result["aki_severe_source_native"] = (
        native_stage.ge(2).where(native_stage.notna()).astype("boolean")
    )
    # The profile adapters carry internal compatibility columns.  The renal
    # module is a versioned physical contract, so never leak those columns (or
    # the deprecated strict phenotype) into current exports.
    keep = [
        resolved_id,
        resolved_time,
        *[column for column in RENAL_AKI_BUNDLE_OUTPUTS if column in result],
    ]
    return (
        result[keep]
        .sort_values([resolved_id, resolved_time], kind="stable")
        .reset_index(drop=True)
    )


def load_renal_aki_bundle(
    database: str,
    *,
    data_path: Optional[str] = None,
    patient_ids: Optional[list[Any]] = None,
    max_patients: Optional[int] = None,
    verbose: bool = True,
    preloaded_data: Optional[Mapping[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Load normalized inputs and build the current renal AKI contract.

    ``load_kdigo_aki`` remains available for historical reproduction.  New
    physical renal exports call this loader and therefore cannot silently
    inherit the deprecated strict phenotype columns.
    """

    from easyicu.api import load_concepts

    preloaded = dict(preloaded_data or {})
    aliases = {
        "kdigo_creatinine_input": ("kdigo_creatinine_input", "crea"),
        "kdigo_urine_input": ("kdigo_urine_input", "urine"),
        "weight": ("weight",),
        "acute_rrt_input": ("acute_rrt_input", "rrt"),
        "crrt_mode_input": ("crrt_mode_input", "crrt"),
    }

    def load_component(name: str) -> Optional[pd.DataFrame]:
        for alias in aliases[name]:
            frame = preloaded.get(alias)
            if isinstance(frame, pd.DataFrame):
                return frame
        try:
            frame = load_concepts(
                concepts=[name],
                database=database,
                data_path=data_path,
                patient_ids=patient_ids,
                max_patients=max_patients,
                verbose=verbose,
            )
        except (KeyError, FileNotFoundError, ValueError):
            if name == "crrt_mode_input":
                return None
            raise
        return frame if isinstance(frame, pd.DataFrame) else None

    crea_df = load_component("kdigo_creatinine_input")
    urine_df = load_component("kdigo_urine_input")
    weight_df = load_component("weight")
    rrt_df = load_component("acute_rrt_input")
    crrt_df = load_component("crrt_mode_input")

    def rename_value(
        frame: Optional[pd.DataFrame], source: str, target: str
    ) -> Optional[pd.DataFrame]:
        if frame is None or target in frame or source not in frame:
            return frame
        return frame.rename(columns={source: target})

    crea_df = rename_value(crea_df, "kdigo_creatinine_input", "crea")
    urine_df = rename_value(urine_df, "kdigo_urine_input", "urine")
    rrt_df = rename_value(rrt_df, "acute_rrt_input", "rrt")
    crrt_df = rename_value(crrt_df, "crrt_mode_input", "rrt")

    return build_renal_aki_bundle(
        database=database,
        crea_df=crea_df,
        urine_df=urine_df,
        weight_df=weight_df,
        rrt_df=rrt_df,
        crrt_df=crrt_df,
        urine_source_is_rate=_normalize_database(database) == "hirid",
        time_unit="hours",
        interval=pd.Timedelta(hours=1),
        rrt_source_complete=rrt_df is not None,
    )


def apply_aki_profile(
    profile_id: str,
    *,
    crea_df: Optional[pd.DataFrame] = None,
    urine_df: Optional[pd.DataFrame] = None,
    weight_df: Optional[pd.DataFrame] = None,
    rrt_df: Optional[pd.DataFrame] = None,
    native_endpoint_df: Optional[pd.DataFrame] = None,
    id_col: Optional[str] = None,
    time_col: Optional[str] = None,
    crea_col: str = "crea",
    urine_col: str = "urine",
    weight_col: str = "weight",
    time_unit: Optional[str] = None,
    urine_source_is_rate: bool = False,
    interval: Optional[pd.Timedelta] = None,
    observation_window_coverage: Optional[Mapping[Any, str]] = None,
    rrt_source_complete: bool = False,
    strict_prerequisites: bool = False,
) -> pd.DataFrame:
    """Apply a frozen canonical or source-native AKI profile.

    Parameters use the normalized EasyICU component boundary.  HiRID's
    published profile additionally requires the author-produced endpoint table
    because the public HiRID 1.1.1 release does not contain all HiRID-II
    auxiliaries.  With ``strict_prerequisites=False`` that profile returns an
    explicit unavailable receipt rather than inventing an approximate stage.
    """

    profile = get_aki_profile(profile_id)
    kwargs = {
        "crea_df": crea_df,
        "urine_df": urine_df,
        "weight_df": weight_df,
        "rrt_df": rrt_df,
        "id_col": id_col,
        "time_col": time_col,
        "crea_col": crea_col,
        "urine_col": urine_col,
        "weight_col": weight_col,
        "time_unit": time_unit,
    }

    if profile_id == _REFERENCE_PROFILE_ID:
        result = _mimic_iv_profile(**kwargs).rename(
            columns={
                "aki_stage_creat": "aki_stage_creat_reference",
                "aki_stage_uo": "aki_stage_uo_reference",
                "aki_stage_crrt": "aki_stage_rrt_reference",
                "aki_stage_source_native": "aki_stage_reference",
                "aki_source_native": "aki_reference",
                "aki_stage_source_native_smoothed": (
                    "aki_stage_reference_smoothed_6h"
                ),
                "aki_source_native_ascertainment": (
                    "aki_reference_ascertainment"
                ),
            }
        )
        result["aki_severe_reference"] = result["aki_stage_reference"].ge(2)
        result["aki_reference_profile"] = profile.profile_id
        result["aki_reference_status"] = "evaluated_semantic_port"
        result["aki_reference_authority"] = profile.authority_level
        result["aki_reference_reliability_grade"] = profile.reliability_grade
        result["aki_reference_fidelity"] = profile.fidelity
        result["aki_reference_rrt_scope"] = (
            "all_active_rrt_cross_database_port"
        )
        result["aki_reference_uses_future"] = bool(
            profile.payload["uses_future_information"]
        )
        return result
    if profile_id == "EASYICU_KDIGO_STRICT_PRIOR_V1":
        result = kdigo_stages(
            crea_df=crea_df,
            urine_df=urine_df,
            weight_df=weight_df,
            rrt_df=rrt_df,
            id_col=id_col,
            time_col=time_col,
            crea_col=crea_col,
            urine_col=urine_col,
            weight_col=weight_col,
            urine_source_is_rate=urine_source_is_rate,
            time_unit=time_unit,
            interval=interval,
            observation_window_coverage=observation_window_coverage,
            rrt_source_complete=rrt_source_complete,
        )
        result = result.copy()
        result["aki_stage_canonical"] = result["aki_stage"]
        result["aki_canonical"] = result["aki"]
        result["aki_canonical_profile"] = profile.profile_id
        result["aki_canonical_status"] = "evaluated"
        result["aki_canonical_authority"] = profile.authority_level
        result["aki_canonical_reliability_grade"] = profile.reliability_grade
        result["aki_canonical_ascertainment"] = result["aki_ascertainment"]
        return result
    if profile.database == "miiv":
        result = _mimic_iv_profile(**kwargs)
    elif profile.database == "mimic":
        result = _mimic_iii_profile(**kwargs)
    elif profile.database == "sic":
        result = _sicdb_profile(**kwargs)
    elif profile.database == "aumc":
        result = _aumc_profile(**kwargs)
    elif profile.database == "eicu":
        result = _eicu_components_profile(**kwargs)
    elif profile.database == "hirid":
        if native_endpoint_df is None or native_endpoint_df.empty:
            if strict_prerequisites:
                raise AKIProfilePrerequisiteError(
                    "HIRID_AKI_EWS_2024_BTAE212 requires the author-produced "
                    "HiRID-II endpoint table and publication-only auxiliaries"
                )
            result = _unavailable_profile_frame(
                crea_df,
                urine_df,
                rrt_df,
                id_col=id_col,
                time_col=time_col,
                reason="required_hirid_ii_author_endpoint_unavailable",
            )
            return _stamp_profile(
                result,
                profile,
                status="not_evaluable_required_source_missing",
                ascertainment="indeterminate",
            )
        result = _hirid_author_endpoint_profile(
            native_endpoint_df, id_col=id_col, time_col=time_col
        )
    else:
        raise AKIProfileError(f"No implementation is registered for {profile_id}")

    status = "components_only" if profile.database == "eicu" else "evaluated"
    return _stamp_profile(result, profile, status=status)


def compare_aki_profiles(
    canonical: pd.DataFrame,
    source_native: pd.DataFrame,
    *,
    id_col: str,
    time_col: Optional[str] = None,
) -> pd.DataFrame:
    """Build an explicit canonical-versus-native comparison table."""

    keys = [id_col] + ([time_col] if time_col is not None else [])
    canonical_stage = _first_existing(canonical, "aki_stage_canonical", "aki_stage")
    native_stage = _first_existing(
        source_native, "aki_stage_source_native", "aki_stage"
    )
    left = canonical[keys].copy()
    left["aki_stage_canonical"] = canonical[canonical_stage].astype("Int64")
    right = source_native[keys].copy()
    right["aki_stage_source_native"] = source_native[native_stage].astype("Int64")
    for column in (
        "aki_source_native_profile",
        "aki_source_native_status",
        "aki_source_native_ascertainment",
    ):
        if column in source_native:
            right[column] = source_native[column]
    result = left.merge(right, on=keys, how="outer")
    comparable = (
        result[["aki_stage_canonical", "aki_stage_source_native"]].notna().all(axis=1)
    )
    result["stage_comparable"] = comparable
    result["stage_agreement"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result.loc[comparable, "stage_agreement"] = (
        result.loc[comparable, "aki_stage_canonical"]
        == result.loc[comparable, "aki_stage_source_native"]
    )
    result["stage_difference_native_minus_canonical"] = (
        result["aki_stage_source_native"] - result["aki_stage_canonical"]
    ).astype("Int64")
    return result


def _profile_from_payload(profile_id: str, payload: Mapping[str, Any]) -> AKIProfile:
    return AKIProfile(
        profile_id=profile_id,
        database=str(payload["database"]),
        authority_level=str(payload["authority_level"]),
        reliability_grade=str(payload["reliability_grade"]),
        fidelity=str(payload["fidelity"]),
        output_kind=str(payload["output_kind"]),
        recommended_role=str(payload["recommended_role"]),
        payload=payload,
    )


def _normalize_database(database: str) -> str:
    aliases = {
        "mimiciii": "mimic",
        "mimic_iii": "mimic",
        "mimic-iii": "mimic",
        "mimiciv": "miiv",
        "mimic_iv": "miiv",
        "mimic-iv": "miiv",
        "sicdb": "sic",
        "amsterdamumcdb": "aumc",
    }
    normalized = database.strip().lower()
    return aliases.get(normalized, normalized)


def _first_existing(frame: pd.DataFrame, *columns: str) -> str:
    for column in columns:
        if column in frame:
            return column
    raise AKIProfileError(f"None of {columns!r} exist in the supplied table")


def _stamp_profile(
    frame: pd.DataFrame,
    profile: AKIProfile,
    *,
    status: str,
    ascertainment: Optional[str] = None,
    ascertainment_column: Optional[str] = None,
) -> pd.DataFrame:
    result = frame.copy()
    if "aki_stage" in result and "aki_stage_source_native" not in result:
        result = result.rename(columns={"aki_stage": "aki_stage_source_native"})
    if "aki" in result and "aki_source_native" not in result:
        result = result.rename(columns={"aki": "aki_source_native"})
    result = result.rename(
        columns={
            "aki_stage_creat": "aki_stage_creat_source_native",
            "aki_stage_uo": "aki_stage_uo_source_native",
            "aki_stage_crrt": "aki_stage_crrt_source_native",
        }
    )
    result[f"{_PROFILE_PREFIX}_profile"] = profile.profile_id
    result[f"{_PROFILE_PREFIX}_status"] = status
    result[f"{_PROFILE_PREFIX}_authority"] = profile.authority_level
    result[f"{_PROFILE_PREFIX}_reliability_grade"] = profile.reliability_grade
    result[f"{_PROFILE_PREFIX}_fidelity"] = profile.fidelity
    if ascertainment_column and ascertainment_column in result:
        result[f"{_PROFILE_PREFIX}_ascertainment"] = result[ascertainment_column]
    elif f"{_PROFILE_PREFIX}_ascertainment" not in result:
        result[f"{_PROFILE_PREFIX}_ascertainment"] = (
            ascertainment or "observed_component_engineering_stage"
        )
    return result


def _component_keys(
    frames: tuple[Optional[pd.DataFrame], ...],
    id_col: Optional[str],
    time_col: Optional[str],
) -> tuple[str, str]:
    anchor = next(
        (
            frame
            for frame in frames
            if isinstance(frame, pd.DataFrame) and not frame.empty
        ),
        None,
    )
    if anchor is None:
        raise AKIProfilePrerequisiteError(
            "At least one non-empty component is required"
        )
    resolved_id = _detect_id_col(anchor, id_col)
    resolved_time = _detect_time_col(anchor, time_col)
    if resolved_id is None or resolved_time is None:
        raise KDIGOComponentSchemaError(
            component="source_native_aki",
            reason_code="source_native_keys_unresolved",
            message="Could not resolve source-native AKI ID/time columns",
        )
    return resolved_id, resolved_time


def _time_hours(
    series: pd.Series, time_col: str, time_unit: Optional[str]
) -> pd.Series:
    unit = _resolve_time_unit(series, time_col, time_unit)
    if unit == "datetime":
        return (series - series.min()) / pd.Timedelta(hours=1)
    if unit == "timedelta":
        return series / pd.Timedelta(hours=1)
    scale = {"seconds": 3600.0, "minutes": 60.0, "hours": 1.0}[unit]
    return pd.to_numeric(series, errors="raise").astype(float) / scale


def _native_spine(
    frames: tuple[Optional[pd.DataFrame], ...], id_col: str, time_col: str
) -> pd.DataFrame:
    keys = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        source_id = _detect_id_col(frame, id_col)
        source_time = _detect_time_col(frame, time_col)
        if source_id is None or source_time is None:
            continue
        keys.append(
            frame[[source_id, source_time]].rename(
                columns={source_id: id_col, source_time: time_col}
            )
        )
    if not keys:
        return pd.DataFrame(columns=[id_col, time_col])
    return (
        pd.concat(keys, ignore_index=True)
        .dropna()
        .drop_duplicates()
        .sort_values([id_col, time_col], kind="stable")
        .reset_index(drop=True)
    )


def _mimic_iv_profile(
    *,
    crea_df: Optional[pd.DataFrame],
    urine_df: Optional[pd.DataFrame],
    weight_df: Optional[pd.DataFrame],
    rrt_df: Optional[pd.DataFrame],
    id_col: Optional[str],
    time_col: Optional[str],
    crea_col: str,
    urine_col: str,
    weight_col: str,
    time_unit: Optional[str],
) -> pd.DataFrame:
    resolved_id, resolved_time = _component_keys(
        (crea_df, urine_df, rrt_df), id_col, time_col
    )
    spine = _native_spine((crea_df, urine_df, rrt_df), resolved_id, resolved_time)
    result = spine.copy()
    if crea_df is not None and not crea_df.empty:
        creat = kdigo_creatinine(
            crea_df,
            id_col=resolved_id,
            time_col=resolved_time,
            value_col=crea_col,
            time_unit=time_unit,
        )
        result = result.merge(
            creat[[resolved_id, resolved_time, "aki_stage_creat"]],
            on=[resolved_id, resolved_time],
            how="left",
        )
    else:
        result["aki_stage_creat"] = pd.NA
    if (
        urine_df is not None
        and not urine_df.empty
        and weight_df is not None
        and not weight_df.empty
    ):
        uo = kdigo_uo(
            urine_df,
            weight_df,
            id_col=resolved_id,
            time_col=resolved_time,
            urine_col=urine_col,
            weight_col=weight_col,
            time_unit=time_unit,
        )
        result = result.merge(
            uo[[resolved_id, resolved_time, "aki_stage_uo"]],
            on=[resolved_id, resolved_time],
            how="left",
        )
    else:
        result["aki_stage_uo"] = pd.NA
    result["aki_stage_crrt"] = _event_stage_at_exact_rows(
        result, rrt_df, resolved_id, resolved_time
    )
    result["aki_stage_source_native"] = (
        result[["aki_stage_creat", "aki_stage_uo", "aki_stage_crrt"]]
        .fillna(0)
        .max(axis=1)
        .astype("Int64")
    )
    result["aki_source_native"] = result["aki_stage_source_native"].gt(0)
    result["aki_stage_source_native_smoothed"] = _rolling_stage_max(
        result,
        resolved_id,
        resolved_time,
        stage_col="aki_stage_source_native",
        hours=6.0,
        time_unit=time_unit,
    )
    result[f"{_PROFILE_PREFIX}_ascertainment"] = np.where(
        result["aki_stage_source_native"].gt(0),
        "positive",
        "component_negative_or_unobserved_coalesced",
    )
    return result


def _mimic_iii_profile(
    *,
    crea_df: Optional[pd.DataFrame],
    urine_df: Optional[pd.DataFrame],
    weight_df: Optional[pd.DataFrame],
    rrt_df: Optional[pd.DataFrame],
    id_col: Optional[str],
    time_col: Optional[str],
    crea_col: str,
    urine_col: str,
    weight_col: str,
    time_unit: Optional[str],
) -> pd.DataFrame:
    del rrt_df
    resolved_id, resolved_time = _component_keys((crea_df, urine_df), id_col, time_col)
    filtered_crea = crea_df
    if crea_df is not None and not crea_df.empty:
        filtered_crea = crea_df.copy()
        hours = _time_hours(filtered_crea[resolved_time], resolved_time, time_unit)
        if not pd.api.types.is_datetime64_any_dtype(filtered_crea[resolved_time]):
            filtered_crea = filtered_crea.loc[hours.between(-6.0, 162.0)].copy()
    spine = _native_spine((filtered_crea, urine_df), resolved_id, resolved_time)
    result = spine.copy()
    if filtered_crea is not None and not filtered_crea.empty:
        creat = kdigo_creatinine(
            filtered_crea,
            id_col=resolved_id,
            time_col=resolved_time,
            value_col=crea_col,
            time_unit=time_unit,
        )
        result = result.merge(
            creat[[resolved_id, resolved_time, "aki_stage_creat"]],
            on=[resolved_id, resolved_time],
            how="left",
        )
    else:
        result["aki_stage_creat"] = pd.NA
    if (
        urine_df is not None
        and not urine_df.empty
        and weight_df is not None
        and not weight_df.empty
    ):
        uo = _mimic_iii_uo(
            urine_df,
            weight_df,
            id_col=resolved_id,
            time_col=resolved_time,
            urine_col=urine_col,
            weight_col=weight_col,
            time_unit=time_unit,
        )
        result = result.merge(uo, on=[resolved_id, resolved_time], how="left")
    else:
        result["aki_stage_uo"] = pd.NA
    result["aki_stage_source_native"] = (
        result[["aki_stage_creat", "aki_stage_uo"]]
        .fillna(0)
        .max(axis=1)
        .astype("Int64")
    )
    result["aki_source_native"] = result["aki_stage_source_native"].gt(0)
    result[f"{_PROFILE_PREFIX}_ascertainment"] = np.where(
        result["aki_stage_source_native"].gt(0),
        "positive",
        "component_negative_or_unobserved_coalesced",
    )
    return result


def _mimic_iii_uo(
    urine_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    urine_col: str,
    weight_col: str,
    time_unit: Optional[str],
) -> pd.DataFrame:
    frame = urine_df[[id_col, time_col, urine_col]].copy()
    frame[urine_col] = _strict_numeric_component(
        frame[urine_col],
        component="mimic_iii_urine",
        reason_code="mimic_iii_urine_non_numeric",
        field_name="urine",
    )
    frame = frame.dropna().sort_values([id_col, time_col]).reset_index(drop=True)
    weights = _one_weight_per_entity(weight_df, id_col, weight_col)
    frame = frame.merge(weights, on=id_col, how="left")
    output = []
    for _, group in frame.groupby(id_col, sort=False):
        group = group.copy()
        group["_hours"] = _time_hours(group[time_col], time_col, time_unit).to_numpy()
        times = group["_hours"].to_numpy(float)
        values = group[urine_col].to_numpy(float)
        patient_weight = group[weight_col].to_numpy(float)
        stages = np.zeros(len(group), dtype=int)
        for row_index, current in enumerate(times):
            windows = {}
            for label, width in (("6", 5.0), ("12", 11.0), ("24", 23.0)):
                mask = (times <= current) & (times >= current - width)
                if not mask.any() or not np.isfinite(patient_weight[row_index]):
                    windows[label] = (np.nan, np.nan)
                    continue
                duration = current - times[mask].min() + 1.0
                rate = values[mask].sum() / patient_weight[row_index] / duration
                windows[label] = (rate, duration)
            rate6, duration6 = windows["6"]
            rate12, duration12 = windows["12"]
            rate24, duration24 = windows["24"]
            stage = 0
            if duration6 >= 2 and rate6 < 0.5:
                stage = 1
            if duration12 >= 5 and rate12 < 0.5:
                stage = 2
            if (duration24 >= 11 and rate24 < 0.3) or (duration12 >= 5 and rate12 == 0):
                stage = 3
            stages[row_index] = stage
        group["aki_stage_uo"] = pd.Series(stages, index=group.index, dtype="Int64")
        output.append(group[[id_col, time_col, "aki_stage_uo"]])
    return pd.concat(output, ignore_index=True)


def _event_stage_at_exact_rows(
    result: pd.DataFrame,
    rrt_df: Optional[pd.DataFrame],
    id_col: str,
    time_col: str,
) -> pd.Series:
    stage = pd.Series(pd.NA, index=result.index, dtype="Int64")
    if rrt_df is None or rrt_df.empty:
        return stage
    source_id = _detect_id_col(rrt_df, id_col)
    source_time = _detect_time_col(rrt_df, time_col)
    value_col = _detect_value_col(rrt_df, "rrt")
    if source_id is None or source_time is None or value_col is None:
        raise AKIProfileError("RRT source has no ID/time/value contract")
    active = rrt_df.loc[
        rrt_df[value_col].fillna(False).astype(bool), [source_id, source_time]
    ]
    active_keys = set(map(tuple, active.to_numpy()))
    for index, row in result[[id_col, time_col]].iterrows():
        if (row[id_col], row[time_col]) in active_keys:
            stage.loc[index] = 3
    return stage


def _rolling_stage_max(
    frame: pd.DataFrame,
    id_col: str,
    time_col: str,
    *,
    stage_col: str,
    hours: float,
    time_unit: Optional[str],
) -> pd.Series:
    output = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    for _, index in frame.groupby(id_col, sort=False).groups.items():
        group = frame.loc[index].sort_values(time_col)
        numeric_time = _time_hours(group[time_col], time_col, time_unit).to_numpy()
        stages = group[stage_col].fillna(0).to_numpy(int)
        maxima = [
            stages[(numeric_time >= current - hours) & (numeric_time <= current)].max()
            for current in numeric_time
        ]
        output.loc[group.index] = pd.array(maxima, dtype="Int64")
    return output


def _one_weight_per_entity(
    weight_df: pd.DataFrame, id_col: str, weight_col: str
) -> pd.DataFrame:
    source_id = _detect_id_col(weight_df, id_col)
    source_value = (
        weight_col
        if weight_col in weight_df
        else _detect_value_col(weight_df, weight_col)
    )
    if source_id is None or source_value is None:
        raise AKIProfileError("Weight source has no ID/value contract")
    frame = weight_df[[source_id, source_value]].rename(
        columns={source_id: id_col, source_value: weight_col}
    )
    frame[weight_col] = pd.to_numeric(frame[weight_col], errors="coerce")
    frame = frame.loc[frame[weight_col].gt(0)].drop_duplicates()
    conflicts = frame.groupby(id_col)[weight_col].nunique().gt(1)
    if conflicts.any():
        raise AKIProfileError(
            "Source-native adapter requires a pre-resolved weight when multiple "
            "weight values exist for one entity"
        )
    return frame.drop_duplicates(id_col)


def _sicdb_profile(
    *,
    crea_df: Optional[pd.DataFrame],
    urine_df: Optional[pd.DataFrame],
    weight_df: Optional[pd.DataFrame],
    rrt_df: Optional[pd.DataFrame],
    id_col: Optional[str],
    time_col: Optional[str],
    crea_col: str,
    urine_col: str,
    weight_col: str,
    time_unit: Optional[str],
) -> pd.DataFrame:
    del weight_df, weight_col
    resolved_id, resolved_time = _component_keys(
        (crea_df, urine_df, rrt_df), id_col, time_col
    )
    entity_ids = sorted(
        set().union(
            *(
                set(frame[_detect_id_col(frame, resolved_id)].dropna())
                for frame in (crea_df, urine_df, rrt_df)
                if frame is not None and not frame.empty
            )
        )
    )
    rows = []
    for entity_id in entity_ids:
        stage = 0
        creat_assessable = False
        if crea_df is not None and not crea_df.empty:
            c = crea_df.loc[
                crea_df[resolved_id] == entity_id, [resolved_time, crea_col]
            ].copy()
            c["_hours"] = _time_hours(c[resolved_time], resolved_time, time_unit)
            c[crea_col] = pd.to_numeric(c[crea_col], errors="coerce")
            c = c.dropna().sort_values("_hours")
            baseline_rows = c.loc[(c["_hours"] > -24) & (c["_hours"] < 168)]
            baseline = (
                baseline_rows.iloc[0][crea_col] if not baseline_rows.empty else np.nan
            )
            max48 = c.loc[(c["_hours"] >= 0) & (c["_hours"] < 48), crea_col].max()
            max168 = c.loc[(c["_hours"] >= 0) & (c["_hours"] < 168), crea_col].max()
            if np.isfinite(baseline) and np.isfinite(max168):
                creat_assessable = True
                if np.isfinite(max48) and max48 > baseline + 0.3:
                    stage = max(stage, 1)
                if max168 > baseline * 1.5:
                    stage = max(stage, 1)
                if max168 > baseline * 2:
                    stage = max(stage, 2)
                if max168 > baseline * 3:
                    stage = max(stage, 3)
                if max168 > 4 and max168 - baseline > 0.5:
                    stage = max(stage, 3)
        urine_stage, urine_assessable = _sicdb_urine_stage(
            urine_df,
            entity_id,
            id_col=resolved_id,
            time_col=resolved_time,
            urine_col=urine_col,
            time_unit=time_unit,
        )
        stage = max(stage, urine_stage)
        rrt_positive = _entity_has_event(
            rrt_df, entity_id, resolved_id, resolved_time, time_unit, 0.0, 168.0
        )
        if rrt_positive:
            stage = 3
        rows.append(
            {
                resolved_id: entity_id,
                "aki_stage_source_native": stage,
                "aki_source_native": stage > 0,
                "aki_stage_creat_assessable_source_native": creat_assessable,
                "aki_stage_uo_assessable_source_native": urine_assessable,
                "aki_stage_rrt_positive_source_native": rrt_positive,
                f"{_PROFILE_PREFIX}_ascertainment": (
                    "positive"
                    if stage > 0
                    else "stage_zero_including_unobserved_components"
                ),
            }
        )
    return pd.DataFrame(rows)


def _sicdb_urine_stage(
    urine_df: Optional[pd.DataFrame],
    entity_id: Any,
    *,
    id_col: str,
    time_col: str,
    urine_col: str,
    time_unit: Optional[str],
) -> tuple[int, bool]:
    if urine_df is None or urine_df.empty:
        return 0, False
    frame = urine_df.loc[urine_df[id_col] == entity_id, [time_col, urine_col]].copy()
    frame["_hours"] = np.floor(
        _time_hours(frame[time_col], time_col, time_unit)
    ).astype(int)
    frame[urine_col] = pd.to_numeric(frame[urine_col], errors="coerce")
    values = dict(
        frame.loc[(frame["_hours"] >= 0) & (frame["_hours"] < 168)]
        .dropna()[["_hours", urine_col]]
        .itertuples(index=False, name=None)
    )
    if not values:
        return 0, False
    stage = 0
    for hour in range(0, 168):
        for width, threshold, candidate in (
            (6, 0.5, 1),
            (12, 0.5, 2),
            (24, 0.3, 3),
        ):
            if hour <= width or hour - width not in values or hour not in values:
                continue
            observed = [values[h] for h in range(hour - width, hour + 1) if h in values]
            if observed and sum(observed) / (width + 1) / 70.0 < threshold:
                stage = max(stage, candidate)
    return stage, True


def _entity_has_event(
    frame: Optional[pd.DataFrame],
    entity_id: Any,
    id_col: str,
    time_col: str,
    time_unit: Optional[str],
    start: float,
    stop: float,
) -> bool:
    if frame is None or frame.empty:
        return False
    value_col = _detect_value_col(frame, "rrt")
    if value_col is None:
        return False
    subset = frame.loc[frame[id_col] == entity_id].copy()
    hours = _time_hours(subset[time_col], time_col, time_unit)
    active = subset[value_col].fillna(False).astype(bool)
    return bool((active & hours.ge(start) & hours.lt(stop)).any())


def _aumc_profile(
    *,
    crea_df: Optional[pd.DataFrame],
    urine_df: Optional[pd.DataFrame],
    weight_df: Optional[pd.DataFrame],
    rrt_df: Optional[pd.DataFrame],
    id_col: Optional[str],
    time_col: Optional[str],
    crea_col: str,
    urine_col: str,
    weight_col: str,
    time_unit: Optional[str],
) -> pd.DataFrame:
    del urine_df, weight_df, rrt_df, urine_col, weight_col
    if crea_df is None or crea_df.empty:
        raise AKIProfilePrerequisiteError("AUMC legacy profile requires creatinine")
    resolved_id, resolved_time = _component_keys((crea_df,), id_col, time_col)
    rows = []
    for entity_id, group in crea_df.groupby(resolved_id, sort=False):
        group = group[[resolved_time, crea_col]].copy()
        group["_hours"] = _time_hours(group[resolved_time], resolved_time, time_unit)
        group[crea_col] = pd.to_numeric(group[crea_col], errors="coerce")
        group = group.dropna()
        baseline = group.loc[
            (group["_hours"] > -(365 * 24)) & (group["_hours"] < 24), crea_col
        ].min()
        maximum = group.loc[
            (group["_hours"] > 0) & (group["_hours"] < 168), crea_col
        ].max()
        assessable = bool(np.isfinite(baseline) and np.isfinite(maximum))
        positive = bool(
            assessable
            and (
                (baseline > 0 and maximum / baseline > 3)
                or (maximum >= 354 / 88.4 and maximum - baseline >= 44 / 88.4)
            )
        )
        rows.append(
            {
                resolved_id: entity_id,
                "acute_renal_failure_source_native": positive,
                "aki_binary_source_native": positive,
                "aki_stage_source_native": 3 if positive else 0,
                f"{_PROFILE_PREFIX}_ascertainment": (
                    "positive"
                    if positive
                    else "negative"
                    if assessable
                    else "unobserved_coalesced_negative"
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["aki_stage_source_native"] = result["aki_stage_source_native"].astype(
        "Int64"
    )
    return result


def _eicu_components_profile(
    *,
    crea_df: Optional[pd.DataFrame],
    urine_df: Optional[pd.DataFrame],
    weight_df: Optional[pd.DataFrame],
    rrt_df: Optional[pd.DataFrame],
    id_col: Optional[str],
    time_col: Optional[str],
    crea_col: str,
    urine_col: str,
    weight_col: str,
    time_unit: Optional[str],
) -> pd.DataFrame:
    del crea_df, weight_df, rrt_df, crea_col, weight_col, time_unit
    if urine_df is None or urine_df.empty:
        raise AKIProfilePrerequisiteError("eICU component profile requires urine data")
    resolved_id, resolved_time = _component_keys((urine_df,), id_col, time_col)
    source_value = (
        urine_col
        if urine_col in urine_df
        else _detect_value_col(urine_df, "cellvaluenumeric")
    )
    if source_value is None:
        raise AKIProfileError("eICU urine component has no numeric value column")
    result = urine_df.copy()
    if "cellpath" in result:
        result = result.loc[
            result["cellpath"].astype(str).str.startswith("I&O|Output (ml)")
        ].copy()
    result["urine_output_source_native"] = pd.to_numeric(
        result[source_value], errors="coerce"
    )
    result["aki_stage_source_native"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )
    result[f"{_PROFILE_PREFIX}_ascertainment"] = "not_available_components_only"
    keep = [
        resolved_id,
        resolved_time,
        "urine_output_source_native",
        "aki_stage_source_native",
        f"{_PROFILE_PREFIX}_ascertainment",
    ]
    return result[keep].reset_index(drop=True)


def _hirid_author_endpoint_profile(
    endpoint: pd.DataFrame,
    *,
    id_col: Optional[str],
    time_col: Optional[str],
) -> pd.DataFrame:
    resolved_id = _detect_id_col(endpoint, id_col)
    resolved_time = _detect_time_col(endpoint, time_col)
    if resolved_id is None or resolved_time is None:
        raise AKIProfileError("HiRID author endpoint has no ID/time columns")
    result = endpoint.copy()
    if "endpoint_status" in result:
        raw = result["endpoint_status"].astype("string").str.lower()
        numeric = pd.to_numeric(raw, errors="coerce")
        result["aki_stage_source_native"] = numeric.where(
            numeric.between(0, 3), pd.NA
        ).astype("Int64")
        result[f"{_PROFILE_PREFIX}_ascertainment"] = np.where(
            result["aki_stage_source_native"].notna(), "observed", "indeterminate"
        )
    elif all(column in result for column in ("geq1", "geq2", "geq3")):
        stage = pd.Series(pd.NA, index=result.index, dtype="Int64")
        known = result[["geq1", "geq2", "geq3"]].isin([0, 1]).all(axis=1)
        stage.loc[known] = 0
        stage.loc[result["geq1"].eq(1)] = 1
        stage.loc[result["geq2"].eq(1)] = 2
        stage.loc[result["geq3"].eq(1)] = 3
        result["aki_stage_source_native"] = stage
        result[f"{_PROFILE_PREFIX}_ascertainment"] = np.where(
            stage.notna(), "observed", "indeterminate"
        )
    else:
        raise AKIProfileError(
            "HiRID author endpoint requires endpoint_status or geq1/geq2/geq3"
        )
    result["aki_source_native"] = (
        result["aki_stage_source_native"].gt(0).astype("boolean")
    )
    return result


def _unavailable_profile_frame(
    crea_df: Optional[pd.DataFrame],
    urine_df: Optional[pd.DataFrame],
    rrt_df: Optional[pd.DataFrame],
    *,
    id_col: Optional[str],
    time_col: Optional[str],
    reason: str,
) -> pd.DataFrame:
    frames = (crea_df, urine_df, rrt_df)
    try:
        resolved_id, resolved_time = _component_keys(frames, id_col, time_col)
        result = _native_spine(frames, resolved_id, resolved_time)
    except AKIProfilePrerequisiteError:
        resolved_id = id_col or "stay_id"
        resolved_time = time_col or "charttime"
        result = pd.DataFrame(columns=[resolved_id, resolved_time])
    result["aki_stage_source_native"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )
    result[f"{_PROFILE_PREFIX}_reason"] = reason
    return result
