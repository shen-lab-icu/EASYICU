"""Aggregate, path-free execution readiness for a registered data package.

This owner proves only pre-analysis capabilities: a typed eligibility
denominator, source-bound patient grouping, and the time/duration coordinates
required by prespecified sensitivities.  It never returns patient rows, private
mapping paths, event counts, rates, comparisons, or effect estimates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from easyicu.research_agent.intake.export_package import (
    ExportPackageError,
    read_exported_concept,
)
from easyicu.webserver import raw_source_authority, source_identity_authority


def _read_review_concept(source_path: str, concept_id: str) -> Optional[pd.DataFrame]:
    try:
        return read_exported_concept(Path(source_path), concept_id)
    except (ExportPackageError, KeyError, OSError, ValueError):
        return None


def _cohort_eligibility_review(
    study: Mapping[str, Any],
    *,
    source_path: str,
    registered_denominator: int,
) -> dict[str, Any]:
    cohort = study.get("cohort")
    cohort = cohort if isinstance(cohort, Mapping) else {}
    age_min = cohort.get("age_min")
    age_max = cohort.get("age_max")
    if age_min is None and age_max is None:
        return {
            "status": "ready",
            "count": registered_denominator,
            "basis": "registered_export_all_rows",
            "missing_age_count": 0,
        }

    age = _read_review_concept(source_path, "age")
    if age is None or not {"stay_id", "age"}.issubset(age.columns):
        return {
            "status": "unavailable",
            "count": None,
            "basis": "typed_age_eligibility",
            "reason_code": "cohort_age_eligibility_unavailable",
        }
    one = age[["stay_id", "age"]].drop_duplicates(subset=["stay_id"]).copy()
    values = pd.to_numeric(one["age"], errors="coerce")
    if len(one) != registered_denominator:
        return {
            "status": "unavailable",
            "count": None,
            "basis": "typed_age_eligibility",
            "reason_code": "cohort_age_denominator_mismatch",
        }
    eligible = values.notna()
    if age_min is not None:
        eligible &= values >= float(age_min)
    if age_max is not None:
        eligible &= values <= float(age_max)
    return {
        "status": "ready",
        "count": int(eligible.sum()),
        "basis": "typed_age_eligibility",
        "age_min": float(age_min) if age_min is not None else None,
        "age_max": float(age_max) if age_max is not None else None,
        "missing_age_count": int(values.isna().sum()),
        "excluded_by_age_count": int((~eligible & values.notna()).sum()),
    }


def _patient_grouping_review(
    study: Mapping[str, Any], *, source_path: str
) -> dict[str, Any]:
    source = study.get("data_source")
    source = source if isinstance(source, Mapping) else {}
    try:
        binding = source_identity_authority.resolve_patient_grouping_authority(
            export_path=source_path,
            database=str(source.get("database") or ""),
        )
    except source_identity_authority.PatientGroupingAuthorityError as exc:
        return {
            "status": "invalid",
            "reason_code": exc.code,
            "details": dict(exc.details),
        }
    if binding is None:
        return {
            "status": "unavailable",
            "reason_code": "patient_grouping_authority_unavailable",
        }
    coordinates = dict(binding.authority_coordinates)
    return {
        "status": "ready",
        "output_identity_column": binding.output_identity_column,
        "group_derivation": str(coordinates.get("grouping_derivation") or ""),
        "authority_ref": str(coordinates.get("authority_ref") or ""),
        "export_manifest_sha256": str(
            coordinates.get("export_manifest_sha256") or ""
        ),
        "mapping_sha256": binding.mapping_sha256,
        "provider_visible_values": False,
    }


def _concept_time_capability(
    source_path: str, concept_id: str, *, event_status: bool = False
) -> dict[str, Any]:
    frame = _read_review_concept(source_path, concept_id)
    if frame is None or concept_id not in frame.columns:
        return {
            "status": "unavailable",
            "source_concept": concept_id,
            "reason_code": "runtime_concept_unavailable",
        }
    has_time = "charttime" in frame.columns
    event_time_complete = None
    if event_status and has_time:
        values = frame[concept_id]
        positive = values.fillna(False).astype(bool)
        event_time_complete = bool(frame.loc[positive, "charttime"].notna().all())
    return {
        "status": "ready" if has_time else "unavailable",
        "source_concept": concept_id,
        "canonical_time_available": has_time,
        **(
            {"event_time_complete_for_recorded_events": event_time_complete}
            if event_time_complete is not None
            else {}
        ),
    }


def _hospital_mortality_followup_capability(
    study: Mapping[str, Any], *, source_path: str
) -> dict[str, Any]:
    """Return the path-free MIMIC-IV hospital-mortality follow-up contract.

    The prepared export's ``death`` flag and ``los_icu`` are deliberately not
    evidence for an in-hospital survival axis.  A legacy export earns this
    capability only when the host has resolved its exact raw-source authority.
    That resolver has already validated table digests and schemas; this public
    review retains only its path-free receipt and typed output coordinates.
    """

    source = study.get("data_source")
    source = source if isinstance(source, Mapping) else {}
    try:
        binding = raw_source_authority.resolve_raw_mimic_iv_source_binding(
            export_path=source_path,
            database=str(source.get("database") or ""),
        )
    except raw_source_authority.RawSourceAuthorityError as exc:
        return {
            "status": "invalid",
            "reason_code": "time_varying_hospital_mortality_followup_contract_unavailable",
            "source_authority_reason_code": exc.code,
        }
    if binding is None:
        return {
            "status": "unavailable",
            "reason_code": "time_varying_hospital_mortality_followup_contract_unavailable",
            "source_authority_reason_code": "raw_source_authority_unavailable",
        }

    receipt = binding.public_receipt()
    contract = receipt.get("hospital_mortality_followup")
    contract = contract if isinstance(contract, Mapping) else {}
    expected = {
        "event_time_column": "death_time_hours",
        "observation_duration_column": "hospital_followup_time_hours",
        "unit": "hours",
        "materializer": "mimic_iv_hospital_death_or_discharge_censor",
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        return {
            "status": "invalid",
            "reason_code": "time_varying_hospital_mortality_followup_contract_unavailable",
            "source_authority_reason_code": "raw_source_authority_contract_invalid",
        }
    return {
        "status": "ready",
        **expected,
        "source_authority": receipt,
    }


def _runtime_readiness_review(
    study: Mapping[str, Any],
    *,
    source_path: str,
    catalog_by_id: Mapping[str, Any],
) -> dict[str, Any]:
    execution = study.get("execution_concepts")
    execution = execution if isinstance(execution, Mapping) else {}
    exposure = str(execution.get("primary_exposure") or "").strip()
    outcome = str(execution.get("outcome") or "").strip()
    sensitivities = study.get("sensitivity_specs")
    sensitivities = sensitivities if isinstance(sensitivities, list) else []
    strategies = {
        str(row.get("strategy") or "")
        for row in sensitivities
        if isinstance(row, Mapping)
    }

    grouping = _patient_grouping_review(study, source_path=source_path)
    time_varying_requested = "time_varying" in strategies
    timing_sensitive_survival_requested = bool(
        {"landmark", "time_varying"}.intersection(strategies)
    )
    time_varying_followup_reason = (
        "time_varying_hospital_mortality_followup_contract_unavailable"
        if outcome == "death"
        else "time_varying_outcome_followup_contract_unavailable"
    )
    hospital_mortality_followup: Optional[dict[str, Any]] = None
    if outcome == "death" and timing_sensitive_survival_requested:
        hospital_mortality_followup = _hospital_mortality_followup_capability(
            study, source_path=source_path
        )
        if hospital_mortality_followup["status"] == "ready":
            event_time = {
                "status": "ready",
                "source_concept": "death",
                "materialized_column": hospital_mortality_followup[
                    "event_time_column"
                ],
                "derivation": hospital_mortality_followup["materializer"],
            }
            observation_duration = {
                "status": "ready",
                "source_concept": "hospital_mortality_followup",
                "materialized_column": hospital_mortality_followup[
                    "observation_duration_column"
                ],
                "unit": hospital_mortality_followup["unit"],
                "derivation": hospital_mortality_followup["materializer"],
            }
        else:
            # A generic event ``charttime`` says nothing about censoring.  The
            # export must not degrade a hospital endpoint to an ICU discharge
            # endpoint merely because one of its tables has timestamps.
            event_time = {
                "status": "unavailable",
                "source_concept": "death",
                "reason_code": time_varying_followup_reason,
                "source_authority_reason_code": hospital_mortality_followup[
                    "source_authority_reason_code"
                ],
            }
            observation_duration = {
                "status": "unavailable",
                "source_concept": None,
                "unit": None,
                "reason_code": time_varying_followup_reason,
                "source_authority_reason_code": hospital_mortality_followup[
                    "source_authority_reason_code"
                ],
            }
    elif time_varying_requested:
        # No generic outcome is currently bound to a verified source-specific
        # event/censor pair.  Keep the failure at the runtime boundary rather
        # than accepting a nearby timestamp as a survival contract.
        event_time = {
            "status": "unavailable",
            "source_concept": outcome,
            "reason_code": time_varying_followup_reason,
        }
        observation_duration = {
            "status": "unavailable",
            "source_concept": None,
            "unit": None,
            "reason_code": time_varying_followup_reason,
        }
    else:
        event_time = (
            _concept_time_capability(source_path, outcome, event_status=True)
            if outcome
            else {
                "status": "unavailable",
                "reason_code": "runtime_outcome_required",
            }
        )
        duration_ready = "los_icu" in catalog_by_id
        observation_duration = {
            "status": "ready" if duration_ready else "unavailable",
            "source_concept": "los_icu",
            "unit": "days",
        }
    exposure_time = (
        _concept_time_capability(source_path, exposure)
        if exposure
        else {
            "status": "unavailable",
            "reason_code": "runtime_exposure_required",
        }
    )
    readmission_ready = "icu_readmission" in catalog_by_id
    required_findings: list[str] = []
    design = study.get("analysis_design")
    design = design if isinstance(design, Mapping) else {}
    if design.get("variance_estimator") == "cluster_robust" and grouping.get(
        "status"
    ) != "ready":
        required_findings.append(
            str(grouping.get("reason_code") or "patient_grouping_authority_unavailable")
        )
    if "landmark" in strategies:
        if event_time.get("status") != "ready":
            required_findings.append("landmark_outcome_event_time_unavailable")
        if observation_duration.get("status") != "ready":
            required_findings.append("landmark_observation_duration_unavailable")
        if exposure_time.get("status") != "ready":
            required_findings.append("landmark_exposure_time_unavailable")
    if time_varying_requested:
        from easyicu.research_agent.planning.sensitivity_authority import normalize_prespecified_sensitivities

        try:
            requested = [spec for spec in normalize_prespecified_sensitivities(sensitivities)
                         if spec.strategy == "time_varying"]
            contract_ready = len(requested) == 1 and requested[0].time_varying_execution is not None
        except ValueError:
            contract_ready = False
        if not contract_ready:
            required_findings.append("time_varying_runtime_unavailable")
        if event_time.get("status") != "ready":
            required_findings.append(time_varying_followup_reason)
    if "non_readmission_restriction" in strategies and not readmission_ready:
        required_findings.append("non_readmission_indicator_unavailable")

    outcome_event_time = dict(event_time)
    if (
        event_time.get("status") == "ready"
        and outcome
        and "materialized_column" not in outcome_event_time
    ):
        outcome_event_time.update(
            {
                "materialized_column": f"{outcome}_time",
                "derivation": "owner_materializer_first_recorded_event_time",
            }
        )
    result = {
        "status": "blocked" if required_findings else "ready",
        "patient_grouping": grouping,
        "outcome_event_time": outcome_event_time,
        "primary_exposure_time": exposure_time,
        "observation_duration": observation_duration,
        "readmission_indicator": {
            "status": "ready" if readmission_ready else "unavailable",
            "source_concept": "icu_readmission",
            "first_patient_stay_authority": False,
        },
        "required_findings": required_findings,
    }
    if hospital_mortality_followup is not None:
        result["hospital_mortality_followup"] = hospital_mortality_followup
    return result


def build_data_package_execution_readiness(
    study: Mapping[str, Any],
    *,
    source_path: str,
    catalog_by_id: Mapping[str, Any],
    registered_denominator: int,
) -> dict[str, Any]:
    """Compile the path-free pre-analysis execution capability receipt."""

    return {
        "eligible_denominator": _cohort_eligibility_review(
            study,
            source_path=source_path,
            registered_denominator=registered_denominator,
        ),
        "runtime_readiness": _runtime_readiness_review(
            study,
            source_path=source_path,
            catalog_by_id=catalog_by_id,
        ),
    }


__all__ = ["build_data_package_execution_readiness"]
