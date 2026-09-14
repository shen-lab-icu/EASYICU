"""Web adapter for the reviewed two-group RMST contrast."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.contracts.rmst import RMSTSpec
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)

from .scientific_runtime_projection import (
    WebScientificRuntimeProjection,
    WebScientificRuntimeProjectionError,
)
from .study_contexts import primary_cohort_selection_mode


def rmst_specification(
    specs: Sequence[PrespecifiedSensitivitySpec],
) -> tuple[str, RMSTSpec] | None:
    matching = [spec for spec in specs if spec.strategy == "restricted_mean_survival"]
    if not matching:
        return None
    if len(matching) != 1 or matching[0].rmst_execution is None:
        raise WebScientificRuntimeProjectionError(
            "web_rmst_specification_incomplete",
            "The agent must close exactly one complete RMST time/event/group/horizon specification.",
            details={"owner": "plan_agent", "human_question_required": False},
        )
    return matching[0].spec_id, matching[0].rmst_execution


def compile_rmst_runtime_projection(
    *,
    study: Mapping[str, Any],
    sensitivity_specs: Sequence[PrespecifiedSensitivitySpec],
    primary_exposure: str | None,
    primary_exposure_source: str | None,
    target_outcome: str | None,
    declared_covariates: Sequence[str],
    covariate_operationalizations: Mapping[str, str],
    target_is_event_status: bool,
    universe_path: Path,
    scientific_configuration_sha256: str,
    literature_citation_keys: Sequence[str] = (),
    direct_comparator_literature_keys: Sequence[str] = (),
    dependence: Any = None,
) -> WebScientificRuntimeProjection | None:
    selected = rmst_specification(sensitivity_specs)
    if selected is None:
        return None
    spec_id, specification = selected
    selection_mode = primary_cohort_selection_mode(study)
    if selection_mode != "all_input_rows":
        raise WebScientificRuntimeProjectionError(
            "web_rmst_filtered_cohort_not_bound",
            "The RMST owner requires a source-bound all-input-row cohort; additional cohort predicates need their own verified projection.",
            details={"owner": "source_runtime", "selection_mode": selection_mode},
        )
    if (
        not primary_exposure
        or primary_exposure_source != specification.group_column
        or not target_outcome
        or target_outcome != specification.event_column
        or not target_is_event_status
        or specification.event_code != 1.0
    ):
        raise WebScientificRuntimeProjectionError(
            "web_rmst_study_binding_mismatch",
            "The RMST contract differs from the study exposure grouping, event column or event coding.",
            details={"owner": "plan_agent", "human_question_required": False},
        )
    identity_column = str(
        getattr(dependence, "group_source", None) or "patient_stay_id"
    )
    required_columns = {
        identity_column,
        specification.time_column,
        specification.event_column,
        specification.group_column,
    }
    try:
        import pyarrow.parquet as pq

        schema_columns = set(pq.read_schema(universe_path).names)
    except Exception as exc:  # noqa: BLE001 - retyped at the Web owner boundary
        raise WebScientificRuntimeProjectionError(
            "web_rmst_schema_unavailable",
            "The materialized cohort schema could not be read for RMST binding.",
            details={"owner": "source_runtime", "reason": str(exc)[:500]},
        ) from exc
    missing_columns = sorted(required_columns - schema_columns)
    if missing_columns:
        raise WebScientificRuntimeProjectionError(
            "web_rmst_columns_missing",
            "The materialized cohort lacks columns required by the reviewed RMST contract.",
            details={"owner": "source_runtime", "missing_columns": missing_columns},
        )
    reference, comparator = specification.group_levels
    authority_payload = {
        "schema_version": "easyicu.rmst_runtime_authority/1",
        "authority_kind": "restricted_mean_survival_difference",
        "protocol_content_sha256": scientific_configuration_sha256,
        "specification": specification.model_dump(mode="json"),
        "sensitivity_spec_id": spec_id,
        "identity_column": identity_column,
        "primary_cohort_selection_mode": selection_mode,
        "development_execution_only_allowed": True,
        "plan_method": "rmst",
        "plan_intent": (
            f"Estimate the prespecified {specification.tau:g}-{specification.time_unit} "
            f"restricted mean survival difference for {reference} versus {comparator} "
            f"using the reviewed Kaplan-Meier kernel. Descriptive analysis only; "
            "no causal or publication claim."
        ),
        "plan_outputs": ["table:rmst_summary", "log:rmst_runtime_receipt"],
    }
    authority = build_current_case_scientific_runtime_authority(authority_payload)
    payload = authority.model_dump(mode="json")
    return WebScientificRuntimeProjection(
        authority=payload,
        projection_sha256=canonical_sha256(payload),
        analysis_only_execution=True,
    )


__all__ = ["compile_rmst_runtime_projection", "rmst_specification"]
