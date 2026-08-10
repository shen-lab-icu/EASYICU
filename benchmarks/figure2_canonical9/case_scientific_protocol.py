"""Typed, digestable E2/H2/H3 scientific protocols for Canonical9.

This benchmark-local module owns the case-specific clinical and methods
coordinates that must not leak into shared Planner prompts or generic KnowHow
cards.  The JSON files are review inputs, not attestations or run authority.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ScientificCaseProtocolError(ValueError):
    """A tracked case protocol is missing, malformed, or assigned to the wrong task."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProtocolCitation(_StrictFrozenModel):
    citation_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")
    title: str = Field(min_length=1, max_length=400)
    year: int = Field(ge=1900, le=2100)
    url: str = Field(pattern=r"^https://")
    doi: str | None = Field(default=None, max_length=160)


class E2ScientificProtocol(_StrictFrozenModel):
    schema_version: Literal["easyicu.figure2_e2_scientific_protocol/1"]
    task_id: Literal["e2_lactate_mortality"]
    protocol_version: str
    review_status: Literal["human_attestation_pending"]
    literature_search_cutoff: Literal["2026-08-09"]
    time_zero: Literal["icu_admission"]
    exposure_window_hours: tuple[Literal[0], Literal[24]]
    exposure: Literal["maximum_valid_typed_lactate"]
    exposure_units: Literal["mmol/L"]
    primary_population: str
    primary_estimand: str
    adjustment_set: tuple[str, ...]
    primary_method: str
    full_cohort_measurement_audit: tuple[str, ...]
    exposure_opportunity_audit: tuple[str, ...]
    reportability_rule: str
    forbidden_interpretations: tuple[str, ...]
    citations: tuple[ProtocolCitation, ...]

    @model_validator(mode="after")
    def _measurement_and_current_guideline_are_explicit(self) -> "E2ScientificProtocol":
        if not self.adjustment_set:
            raise ValueError("E2 adjustment_set must be prespecified")
        if not {"measured_fraction", "unmeasured_fraction"}.issubset(
            self.full_cohort_measurement_audit
        ):
            raise ValueError("E2 must retain the measured/unmeasured denominator")
        if "ssc_adult_2026" not in {item.citation_id for item in self.citations}:
            raise ValueError("E2 must cite the current 2026 adult SSC guideline")
        return self


class MedicationCaptureContract(_StrictFrozenModel):
    source: Literal["mimic_iv_inputevents_derived_typed_vasopressor"]
    audited_window_hours: tuple[Literal[0], Literal[24]]
    positive_record_semantics: str
    absent_record_semantics: Literal["no_recorded_administration_not_verified_non_use"]
    verified_non_use_available: Literal[False]
    binary_control_arm_authorized: Literal[False]
    causal_contrast_authorized: Literal[False]
    decision: Literal["fail_closed"]
    reason_code: Literal["H2_VERIFIED_NON_USE_UNAVAILABLE"]


class TargetTrialCoordinates(_StrictFrozenModel):
    eligible_population: str
    treatment_strategies: tuple[str, str]
    time_zero: Literal["icu_admission"]
    grace_period_hours: Literal[24]
    grace_period_method: Literal["clone_censor_weight_if_source_contract_becomes_valid"]
    followup: str
    outcome: str
    estimand: str
    censoring_and_competing_events: str
    baseline_adjustment_timing: Literal["at_or_before_icu_admission_time_zero"]
    baseline_adjustment_variables: tuple[str, ...]
    grace_period_time_varying_variables: tuple[str, ...]
    post_time_zero_variable_role: Literal[
        "time_varying_information_for_prespecified_grace_period_adherence_or_censoring_model_only"
    ]
    estimation_method: Literal[
        "clone_censor_weight_with_stabilized_inverse_probability_censoring_weights"
    ]
    positivity_interval: tuple[float, float]
    weight_truncation_percentiles: tuple[float, float]
    balance_threshold_absolute_smd: float
    positivity_failure_action: Literal["fail_closed_no_effect_estimate"]
    sensitivity_analyses: tuple[str, ...]

    @model_validator(mode="after")
    def _closed_target_trial_rules(self) -> "TargetTrialCoordinates":
        if not self.baseline_adjustment_variables:
            raise ValueError("H2 baseline adjustment variables must be prespecified")
        if not self.grace_period_time_varying_variables:
            raise ValueError("H2 grace-period time-varying variables must be prespecified")
        if self.positivity_interval != (0.05, 0.95):
            raise ValueError("H2 positivity interval must be frozen at [0.05, 0.95]")
        if self.weight_truncation_percentiles != (1.0, 99.0):
            raise ValueError("H2 weight truncation must be frozen at [1, 99]")
        if self.balance_threshold_absolute_smd != 0.1:
            raise ValueError("H2 balance threshold must be absolute SMD <= 0.1")
        return self


class H2ScientificProtocol(_StrictFrozenModel):
    schema_version: Literal["easyicu.figure2_h2_scientific_protocol/1"]
    task_id: Literal["h2_vasopressor_causal"]
    protocol_version: str
    review_status: Literal["human_attestation_pending"]
    literature_search_cutoff: Literal["2026-08-09"]
    current_source_capture: MedicationCaptureContract
    intended_target_trial: TargetTrialCoordinates
    current_scientific_result: Literal[
        "treatment_contrast_not_identifiable_from_available_capture_contract"
    ]
    reportability_rule: str
    forbidden_actions: tuple[str, ...]
    citations: tuple[ProtocolCitation, ...]

    @model_validator(mode="after")
    def _current_source_cannot_open_a_control_arm(self) -> "H2ScientificProtocol":
        citation_ids = {item.citation_id for item in self.citations}
        required = {"target_statement_2025", "ssc_adult_2026", "mimic_iv_inputevents"}
        if not required.issubset(citation_ids):
            raise ValueError("H2 is missing current methods, clinical, or source evidence")
        if not self.forbidden_actions:
            raise ValueError("H2 must declare forbidden absence-as-nonuse actions")
        return self


class HistoricalH3Protocol(_StrictFrozenModel):
    protocol_version: str
    model_family: Literal["diagonal_gaussian_mixture"]
    selected_n_clusters: Literal[6]
    n_resamples: Literal[100]
    sample_fraction: Literal[0.8]
    stability_metric: Literal["mean_adjusted_rand_index"]
    minimum_mean_stability: Literal[0.7]
    observed_mean_stability: Literal[0.5357]
    decision: Literal["terminal_failed_closed"]


class H3Representation(_StrictFrozenModel):
    population: str
    time_zero: Literal["icu_admission"]
    window_hours: tuple[Literal[0], Literal[72]]
    grid_width_hours: Literal[12]
    aggregation: Literal["max"]
    features: tuple[str, ...]
    scaling: Literal["coordinate_wise_z_score_using_observed_values"]
    minimum_observed_sofa2_windows: Literal[2]
    missingness_method: Literal["observed_data_likelihood_no_zero_or_locf_imputation"]
    trailing_missingness_policy: Literal[
        "retain_and_report_discharge_death_and_measurement_support"
    ]


class H3SelectionAndStability(_StrictFrozenModel):
    model_family: Literal["observed_data_diagonal_gaussian_mixture"]
    candidate_cluster_counts: tuple[int, ...]
    cluster_number_criterion: Literal["minimum_bic"]
    outcome_blind_selection: Literal[True]
    minimum_cluster_fraction: float
    cluster_size_failure_action: Literal["no_stable_solution_no_alternate_k"]
    resampling_method: Literal["subsample_without_replacement"]
    n_resamples: Literal[100]
    sample_fraction: Literal[0.8]
    base_seed: Literal[1729]
    stability_metric: Literal["mean_adjusted_rand_index"]
    minimum_successful_resamples: Literal[100]
    minimum_mean_stability: Literal[0.7]
    stability_failure_action: Literal["no_stable_solution_no_post_hoc_rescue"]

    @model_validator(mode="after")
    def _candidate_and_size_rules_are_closed(self) -> "H3SelectionAndStability":
        if self.candidate_cluster_counts != (2, 3, 4, 5, 6):
            raise ValueError("H3 candidate k set must be the frozen ordered 2-6 set")
        if self.minimum_cluster_fraction != 0.05:
            raise ValueError("H3 minimum cluster fraction must be 0.05")
        return self


class H3ScientificProtocol(_StrictFrozenModel):
    schema_version: Literal["easyicu.figure2_h3_scientific_protocol/1"]
    task_id: Literal["h3_trajectory_clustering"]
    protocol_version: str
    review_status: Literal["human_attestation_pending"]
    literature_search_cutoff: Literal["2026-08-09"]
    supersedes_terminal_protocol: HistoricalH3Protocol
    representation: H3Representation
    selection_and_stability: H3SelectionAndStability
    outcome_use: Literal[
        "descriptive_after_frozen_assignment_never_for_model_or_k_selection"
    ]
    external_reproducibility_rule: str
    reportability_rule: str
    forbidden_actions: tuple[str, ...]
    citations: tuple[ProtocolCitation, ...]

    @model_validator(mode="after")
    def _redesign_preserves_a_no_solution_result(self) -> "H3ScientificProtocol":
        if "no stable phenotype solution" not in self.reportability_rule.lower():
            raise ValueError("H3 must make no stable solution a formal result")
        if "wong_multicenter_phenotypes_2026" not in {
            item.citation_id for item in self.citations
        }:
            raise ValueError("H3 must include current multicenter reproducibility evidence")
        return self


ScientificCaseProtocol = Annotated[
    Union[E2ScientificProtocol, H2ScientificProtocol, H3ScientificProtocol],
    Field(discriminator="task_id"),
]


_PROTOCOL_FILENAMES = {
    "e2_lactate_mortality": "e2_lactate_mortality_20260809.json",
    "h2_vasopressor_causal": "h2_vasopressor_causal_20260809.json",
    "h3_trajectory_clustering": "h3_trajectory_clustering_20260809.json",
}
_PROTOCOL_MODELS = {
    "e2_lactate_mortality": E2ScientificProtocol,
    "h2_vasopressor_causal": H2ScientificProtocol,
    "h3_trajectory_clustering": H3ScientificProtocol,
}


def default_case_protocol_path(task_id: str) -> Path:
    """Return the tracked review-input path for one case-scoped protocol."""

    try:
        filename = _PROTOCOL_FILENAMES[task_id]
    except KeyError as exc:
        raise ScientificCaseProtocolError(
            f"SCIENTIFIC_CASE_PROTOCOL_UNKNOWN_TASK: {task_id}"
        ) from exc
    return Path(__file__).with_name("protocols") / filename


def load_case_scientific_protocol(
    path: Path,
    *,
    expected_task_id: str,
) -> E2ScientificProtocol | H2ScientificProtocol | H3ScientificProtocol:
    """Strict-load a case protocol and assign failures to this owner module."""

    try:
        model = _PROTOCOL_MODELS[expected_task_id]
    except KeyError as exc:
        raise ScientificCaseProtocolError(
            f"SCIENTIFIC_CASE_PROTOCOL_UNKNOWN_TASK: {expected_task_id}"
        ) from exc
    try:
        protocol = model.model_validate_json(path.read_bytes(), strict=True)
    except Exception as exc:  # noqa: BLE001
        raise ScientificCaseProtocolError(
            f"SCIENTIFIC_CASE_PROTOCOL_INVALID: {expected_task_id}: {exc}"
        ) from exc
    if protocol.task_id != expected_task_id:
        raise ScientificCaseProtocolError(
            "SCIENTIFIC_CASE_PROTOCOL_TASK_MISMATCH: "
            f"expected={expected_task_id} observed={protocol.task_id}"
        )
    return protocol


def case_protocol_content_sha256(protocol: BaseModel) -> str:
    """Hash normalized protocol content independently of JSON whitespace."""

    raw = json.dumps(
        protocol.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_default_case_protocol(
    task_id: str,
) -> E2ScientificProtocol | H2ScientificProtocol | H3ScientificProtocol:
    return load_case_scientific_protocol(
        default_case_protocol_path(task_id),
        expected_task_id=task_id,
    )


__all__ = [
    "E2ScientificProtocol",
    "H2ScientificProtocol",
    "H3ScientificProtocol",
    "ScientificCaseProtocolError",
    "case_protocol_content_sha256",
    "default_case_protocol_path",
    "load_case_scientific_protocol",
    "load_default_case_protocol",
]
