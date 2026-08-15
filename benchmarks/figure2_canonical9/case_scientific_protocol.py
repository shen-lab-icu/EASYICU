"""Typed, digestable E2/H2/H3 scientific protocols for Canonical9.

This benchmark-local module owns the case-specific clinical and methods
coordinates that must not leak into shared Planner prompts or generic KnowHow
cards.  The JSON files are review inputs, not attestations or run authority.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from easyicu.research_agent.schema import TrajectoryStabilitySpec
from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    build_trajectory_scientific_runtime_authority,
)


class ScientificCaseProtocolError(ValueError):
    """A tracked case protocol is missing, malformed, or assigned to the wrong task."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class ProtocolCitation(_StrictFrozenModel):
    citation_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")
    title: str = Field(min_length=1, max_length=400)
    year: int = Field(ge=1900, le=2100)
    url: str = Field(pattern=r"^https://")
    doi: str | None = Field(default=None, max_length=160)


class E2LandmarkPrimary(_StrictFrozenModel):
    landmark_hours: Literal[24]
    population_rule: Literal[
        "alive_and_under_icu_observation_at_24h_with_valid_0_24h_lactate"
    ]
    exposure_opportunity: Literal["complete_icu_0_24h_window"]
    followup_start: Literal["after_24h_landmark"]
    outcome: Literal["post_landmark_in_hospital_death"]
    interpretation: Literal["descriptive_prognostic_association_not_causal"]


class E2PrimaryModel(_StrictFrozenModel):
    family: Literal["logistic_regression"]
    exposure_form: Literal["restricted_cubic_spline"]
    knot_quantiles: tuple[float, float, float]
    reference: Literal["median_lactate_in_primary_population"]
    interval: Literal["95_percent_confidence_interval"]
    headline_output: Literal["adjusted_odds_curve_and_prespecified_contrasts"]
    linear_sensitivity: Literal["per_1_mmol_l_linear_term"]

    @model_validator(mode="after")
    def _frozen_spline(self) -> "E2PrimaryModel":
        if self.knot_quantiles != (0.10, 0.50, 0.90):
            raise ValueError("E2 primary RCS knots must be frozen at 10/50/90 percentiles")
        return self


class E2ScientificProtocol(_StrictFrozenModel):
    schema_version: Literal["easyicu.figure2_e2_scientific_protocol/2"]
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
    primary_landmark: E2LandmarkPrimary
    adjustment_set: tuple[str, ...]
    primary_model: E2PrimaryModel
    secondary_descriptive_sensitivity: str
    full_cohort_measurement_audit: tuple[str, ...]
    exposure_opportunity_audit: tuple[str, ...]
    reportability_rule: str
    forbidden_interpretations: tuple[str, ...]
    citations: tuple[ProtocolCitation, ...]

    @model_validator(mode="after")
    def _measurement_and_current_guideline_are_explicit(self) -> "E2ScientificProtocol":
        if self.adjustment_set != ("age", "sex", "charlson"):
            raise ValueError("E2 adjustment_set must be frozen as age/sex/charlson")
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
    pre_icu_treatment_history_authority: Literal[False]
    initiator_status_authorized: Literal[False]
    prevalent_user_status_authorized: Literal[False]
    verified_non_use_available: Literal[False]
    binary_control_arm_authorized: Literal[False]
    causal_contrast_authorized: Literal[False]
    decision: Literal["fail_closed"]
    reason_code: Literal["H2_VERIFIED_NON_USE_UNAVAILABLE"]


class H2FutureUnblockContract(_StrictFrozenModel):
    status: Literal[
        "future_design_only_not_executable_under_current_materialization"
    ]
    required_source_coverage: tuple[str, ...]
    coverage_unit: Literal["per_icu_stay"]
    pre_icu_lookback_must_be_prespecified: Literal[True]
    must_distinguish: tuple[
        Literal["true_initiator"],
        Literal["prevalent_user"],
        Literal["verified_non_user"],
    ]
    new_clinical_and_methods_review_required: Literal[True]

    @model_validator(mode="after")
    def _coverage_is_concrete(self) -> "H2FutureUnblockContract":
        if len(self.required_source_coverage) < 3:
            raise ValueError("H2 future unblock contract needs concrete source coverage")
        return self


class H2ScientificProtocol(_StrictFrozenModel):
    schema_version: Literal["easyicu.figure2_h2_scientific_protocol/2"]
    task_id: Literal["h2_vasopressor_causal"]
    protocol_version: str
    review_status: Literal["human_attestation_pending"]
    literature_search_cutoff: Literal["2026-08-09"]
    current_source_capture: MedicationCaptureContract
    current_formal_scope: Literal[
        "source_specific_fail_closed_feasibility_only_no_effect_estimation"
    ]
    future_unblock_contract: H2FutureUnblockContract
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
    descriptive_only_features: tuple[str, ...]
    scaling: Literal[
        "pooled_coordinate_wise_z_score_using_owner_available_values"
    ]
    scaling_ddof: Literal[0]
    scaling_zero_variance_action: Literal["fail_closed"]
    minimum_available_sofa2_windows: Literal[2]
    concept_missingness_method: Literal[
        "respect_sofa2_concept_owner_missingness_and_locf_semantics"
    ]
    evidence_states: tuple[
        Literal["direct_observed"],
        Literal["owner_locf_available"],
        Literal["unavailable"],
    ]
    owner_locf_policy: Literal["include_with_explicit_audit"]
    unavailable_policy: Literal["exclude_from_observed_data_likelihood"]
    clustering_stage_imputation: Literal["none"]
    trailing_missingness_policy: Literal[
        "retain_and_report_discharge_death_and_measurement_support"
    ]


class H3SelectionAndStability(_StrictFrozenModel):
    model_family: Literal["observed_data_diagonal_gaussian_mixture"]
    candidate_cluster_counts: tuple[int, ...]
    cluster_number_criterion: Literal["minimum_bic"]
    candidate_boundary_action: Literal[
        "fail_closed_if_minimum_bic_is_at_upper_boundary"
    ]
    candidate_boundary_reason_code: Literal["H3_NO_INTERIOR_BIC_OPTIMUM"]
    candidate_fit_base_seed: Literal[1729]
    candidate_fit_max_iter: Literal[200]
    candidate_fit_tolerance: Literal[1e-6]
    candidate_fit_regularization: Literal[1e-6]
    bic_sample_size: Literal["frozen_population_rows"]
    bic_parameter_count: Literal[
        "mixture_weights_k_minus_1_plus_2_k_per_coordinate"
    ]
    bic_tie_break: Literal["smaller_k"]
    outcome_blind_selection: Literal[True]
    minimum_cluster_fraction: float
    minimum_cluster_fraction_reason_code: Literal[
        "H3_MINIMUM_CLUSTER_FRACTION_NOT_MET"
    ]
    cluster_size_failure_action: Literal["no_stable_solution_no_alternate_k"]
    resampling_method: Literal["subsample_without_replacement"]
    n_resamples: Literal[100]
    sample_fraction: Literal[0.8]
    base_seed: Literal[1729]
    stability_metric: Literal["mean_adjusted_rand_index"]
    minimum_successful_resamples: Literal[100]
    refit_failure_action: Literal[
        "numerical_engine_failure_not_scientific_instability"
    ]
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
    schema_version: Literal["easyicu.figure2_h3_scientific_protocol/2"]
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


class RuntimeScientificProjection(_StrictFrozenModel):
    """The one human-reviewable projection consumed by Canonical9 runtime.

    It contains the exact normalized protocol bytes plus the deterministic
    Agent-facing rendering.  Human review and the launcher bind the digest of
    this object, so editing projection code or Agent-visible wording after
    sign-off invalidates run authority even when the protocol version is not
    changed.
    """

    schema_version: Literal["easyicu.figure2_runtime_scientific_projection/1"]
    task_id: Literal[
        "e2_lactate_mortality",
        "h2_vasopressor_causal",
        "h3_trajectory_clustering",
    ]
    protocol_version: str
    protocol_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_protocol_json: str = Field(min_length=2)
    agent_visible_required_outputs: tuple[str, ...]
    agent_visible_guardrails: tuple[str, ...]
    deterministic_execution_contract: dict[str, Any] | None
    runtime_projection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _exact_content_and_digest(self) -> "RuntimeScientificProjection":
        if (
            hashlib.sha256(self.canonical_protocol_json.encode("utf-8")).hexdigest()
            != self.protocol_content_sha256
        ):
            raise ValueError("runtime projection protocol content digest mismatch")
        body = self.model_dump(mode="json", exclude={"runtime_projection_sha256"})
        if hashlib.sha256(_canonical_json_bytes(body)).hexdigest() != (
            self.runtime_projection_sha256
        ):
            raise ValueError("runtime scientific projection digest mismatch")
        if not self.agent_visible_required_outputs or not self.agent_visible_guardrails:
            raise ValueError("runtime projection must be visible to the Agent")
        if self.deterministic_execution_contract is None:
            raise ValueError("every governed case requires a deterministic contract")
        expected_schema = {
            "e2_lactate_mortality": "easyicu.landmark_spline_runtime_authority/1",
            "h2_vasopressor_causal": "easyicu.source_feasibility_runtime_authority/1",
            "h3_trajectory_clustering": (
                "easyicu.trajectory_scientific_runtime_authority/1"
            ),
        }[self.task_id]
        if self.deterministic_execution_contract.get("schema_version") != (
            expected_schema
        ):
            raise ValueError("runtime deterministic contract kind does not match task")
        return self


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

    return hashlib.sha256(
        _canonical_json_bytes(protocol.model_dump(mode="json"))
    ).hexdigest()


def _h3_deterministic_execution_contract(
    protocol: H3ScientificProtocol,
) -> dict[str, Any]:
    """Compile the typed H3 protocol into the shared case-neutral executor schema."""

    representation = protocol.representation
    selection = protocol.selection_and_stability
    columns = tuple(
        f"{concept}__h{start}_{start + representation.grid_width_hours}"
        for concept in representation.features
        for start in range(
            representation.window_hours[0],
            representation.window_hours[1],
            representation.grid_width_hours,
        )
    )
    stability_spec = TrajectoryStabilitySpec(
        n_resamples=selection.n_resamples,
        sample_fraction=selection.sample_fraction,
        base_seed=selection.base_seed,
        minimum_successful_resamples=selection.minimum_successful_resamples,
        refit_max_iter=selection.candidate_fit_max_iter,
        refit_tolerance=selection.candidate_fit_tolerance,
        refit_regularization=selection.candidate_fit_regularization,
        minimum_mean_stability=selection.minimum_mean_stability,
        decision_mode="minimum_mean_threshold",
    )
    return build_trajectory_scientific_runtime_authority(
        {
            "schema_version": "easyicu.trajectory_scientific_runtime_authority/1",
            "protocol_content_sha256": case_protocol_content_sha256(protocol),
            "coordinate_concepts": list(representation.features),
            "descriptive_only_concepts": list(
                representation.descriptive_only_features
            ),
            "window_start_hours": representation.window_hours[0],
            "window_end_hours": representation.window_hours[1],
            "grid_width_hours": representation.grid_width_hours,
            "aggregation": representation.aggregation,
            "representation_columns": list(columns),
            "minimum_available_windows": (
                representation.minimum_available_sofa2_windows
            ),
            "coordinate_scaling": {
                "method": "pooled_coordinate_wise_z_score",
                "ddof": representation.scaling_ddof,
                "observed_value_policy": "direct_or_owner_locf_available",
                "missing_value_policy": (
                    "preserve_missing_exclude_from_likelihood"
                ),
                "zero_variance_action": (
                    representation.scaling_zero_variance_action
                ),
            },
            "evidence_state_policy": {
                "direct_observed": "include",
                "owner_locf_available": "include_and_audit",
                "unavailable": "exclude",
                "additional_clustering_stage_imputation": (
                    representation.clustering_stage_imputation
                ),
            },
            "representation_plan_method": (
                "signed_fixed_window_trajectory_representation"
            ),
            "representation_plan_intent": (
                "Build the digest-bound fixed-window trajectory representation "
                "exactly as declared by the scientific runtime authority."
            ),
            "representation_plan_inputs": [],
            "representation_required_outputs": [
                "artifact:trajectory_representation",
                "table:trajectory_membership",
                "manifest:trajectory_representation_schema",
            ],
            "model_family": "latent_class_diagonal_gaussian_mixture",
            "fit_method": "observed_data_em_diagonal_gaussian_mixture",
            "covariance_type": "diag",
            "candidate_cluster_counts": list(selection.candidate_cluster_counts),
            "selection_criterion": "bic",
            "selection_rule": "minimum",
            "candidate_fit_base_seed": selection.candidate_fit_base_seed,
            "candidate_fit_max_iter": selection.candidate_fit_max_iter,
            "candidate_fit_tolerance": selection.candidate_fit_tolerance,
            "candidate_fit_regularization": selection.candidate_fit_regularization,
            "bic_sample_size": selection.bic_sample_size,
            "bic_parameter_count": selection.bic_parameter_count,
            "bic_tie_break": selection.bic_tie_break,
            "upper_boundary_action": (
                "fail_closed_if_selected_at_upper_boundary"
            ),
            "upper_boundary_reason_code": (
                selection.candidate_boundary_reason_code
            ),
            "minimum_cluster_fraction": selection.minimum_cluster_fraction,
            "minimum_cluster_fraction_reason_code": (
                selection.minimum_cluster_fraction_reason_code
            ),
            "stability_spec": stability_spec.model_dump(mode="json"),
        }
    ).model_dump(mode="json")


def _e2_deterministic_execution_contract(
    protocol: E2ScientificProtocol,
) -> dict[str, Any]:
    model = protocol.primary_model
    return build_current_case_scientific_runtime_authority(
        {
            "schema_version": "easyicu.landmark_spline_runtime_authority/1",
            "authority_kind": "landmark_spline_association",
            "protocol_content_sha256": case_protocol_content_sha256(protocol),
            "plan_method": "signed_landmark_restricted_cubic_spline",
            "plan_intent": (
                "Execute the signed 24-hour landmark restricted-cubic-spline "
                "association and its prespecified linear sensitivity."
            ),
            "plan_outputs": [
                "table:e2_landmark_rcs_curve",
                "table:e2_landmark_rcs_contrasts",
                "table:e2_linear_sensitivity",
                "log:e2_scientific_runtime_receipt",
            ],
            "exposure_column": "lact_max",
            "outcome_column": "death",
            "outcome_time_column": "death_time",
            "observation_duration_column": "los_icu",
            "observation_duration_unit": "days",
            "landmark_hours": protocol.primary_landmark.landmark_hours,
            "required_adjustment_columns": ["age", "sex", "charlson_first"],
            "categorical_adjustment_columns": ["sex"],
            "spline_knot_quantiles": list(model.knot_quantiles),
            "spline_reference": "median_in_primary_population",
            "curve_quantile_range": [
                model.knot_quantiles[0],
                model.knot_quantiles[2],
            ],
            "curve_points": 41,
            "linear_sensitivity_per_unit": 1.0,
            "interpretation": protocol.primary_landmark.interpretation,
        }
    ).model_dump(mode="json")


def _h2_deterministic_execution_contract(
    protocol: H2ScientificProtocol,
) -> dict[str, Any]:
    capture = protocol.current_source_capture
    return build_current_case_scientific_runtime_authority(
        {
            "schema_version": "easyicu.source_feasibility_runtime_authority/1",
            "authority_kind": "source_feasibility_fail_closed",
            "protocol_content_sha256": case_protocol_content_sha256(protocol),
            "plan_method": "signed_source_feasibility_fail_closed",
            "plan_intent": (
                "Emit the signed source-specific feasibility result without "
                "constructing a treatment contrast or effect estimate."
            ),
            "plan_outputs": [
                "table:h2_source_feasibility",
                "log:h2_scientific_runtime_receipt",
            ],
            "source": capture.source,
            "audited_window_hours": list(capture.audited_window_hours),
            "decision": capture.decision,
            "reason_code": capture.reason_code,
            "verified_non_use_available": capture.verified_non_use_available,
            "binary_control_arm_authorized": capture.binary_control_arm_authorized,
            "causal_contrast_authorized": capture.causal_contrast_authorized,
            "forbidden_plan_tokens": [
                "propensity_score_matching",
                "psm",
                "iptw",
                "inverse_probability_weighting",
                "effect_estimate",
                "causal_effect",
                "control_arm",
            ],
            "future_design_authorized": False,
        }
    ).model_dump(mode="json")


def _projection_agent_content(
    protocol: E2ScientificProtocol | H2ScientificProtocol | H3ScientificProtocol,
    execution_contract: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Render only typed protocol fields; never maintain a second science source."""

    if isinstance(protocol, E2ScientificProtocol):
        landmark = protocol.primary_landmark
        model = protocol.primary_model
        rule_ref = (
            "scientific_runtime_contract:"
            + str(execution_contract["execution_contract_sha256"])
        )
        return (
            tuple(execution_contract["plan_outputs"]),
            (
                f"Primary population rule={landmark.population_rule}; landmark={landmark.landmark_hours}h; "
                f"follow-up={landmark.followup_start}; outcome={landmark.outcome}.",
                f"Primary model={model.family}/{model.exposure_form}; knot quantiles="
                f"{list(model.knot_quantiles)}; reference={model.reference}; linear model is sensitivity only.",
                "Do not treat observation duration or lactate measurement count as ordinary baseline covariates; "
                "report the variable-opportunity measured-subset analysis only as a secondary descriptive sensitivity.",
                "The estimand is descriptive/prognostic and must never be described as causal.",
                "The only primary owner must use method="
                f"{execution_contract['plan_method']}, intent="
                f"{execution_contract['plan_intent']!r}, and icu_rule_refs must "
                f"contain {rule_ref}.",
            ),
        )
    if isinstance(protocol, H2ScientificProtocol):
        capture = protocol.current_source_capture
        unblock = protocol.future_unblock_contract
        rule_ref = (
            "scientific_runtime_contract:"
            + str(execution_contract["execution_contract_sha256"])
        )
        return (
            tuple(execution_contract["plan_outputs"]),
            (
                f"Current source={capture.source} has pre_icu_treatment_history_authority="
                f"{str(capture.pre_icu_treatment_history_authority).lower()}, verified_non_use_available="
                f"{str(capture.verified_non_use_available).lower()}, and causal_contrast_authorized="
                f"{str(capture.causal_contrast_authorized).lower()}.",
                f"Return {capture.reason_code}; do not construct a control arm, PSM/IPTW, or effect estimate.",
                f"The only current unblock route is new per-stay source coverage satisfying "
                f"{list(unblock.required_source_coverage)} followed by new clinical and methods review.",
                "The future target-trial design is non-authorizing and must not be executed under the current materialization.",
                "Declare no primary effect step. The sole current-result owner must "
                f"use method={execution_contract['plan_method']}, intent="
                f"{execution_contract['plan_intent']!r}, and icu_rule_refs must "
                f"contain {rule_ref}.",
            ),
        )
    representation = protocol.representation
    selection = protocol.selection_and_stability
    rule_ref = (
        "scientific_runtime_contract:"
        + str(execution_contract["execution_contract_sha256"])
    )
    return (
        (
            "receipt-aware trajectory representation and scaling manifest",
            "frozen candidate-k BIC selection ledger with interior-optimum decision",
            "100-resample stability audit separating engine failure from scientific instability",
        ),
        (
            f"Clustering coordinates are exactly {list(representation.features)}; descriptive-only features "
            f"{list(representation.descriptive_only_features)} must not enter the model matrix.",
            f"Respect evidence states {list(representation.evidence_states)} and SOFA-2 concept-owner missingness; "
            "perform no additional clustering-stage imputation and exclude unavailable values from likelihood.",
            f"Apply {representation.scaling} with ddof={representation.scaling_ddof}; record centers, scales, "
            "mask policy, and a digest-bound scaling manifest.",
            f"Evaluate candidate k={list(selection.candidate_cluster_counts)} by minimum BIC; if the minimum is "
            f"at the upper boundary, fail closed with {selection.candidate_boundary_reason_code}.",
            "Emit the exact signed representation columns in this order: "
            f"{execution_contract['representation_columns']}; the deterministic runtime "
            "will reject any substituted feature, time bin, or descriptive-only coordinate.",
            "Classify a refit numerical/engine failure separately from a completed stability analysis whose mean ARI is below threshold.",
            "The representation owner must use method="
            f"{execution_contract['representation_plan_method']}, intent="
            f"{execution_contract['representation_plan_intent']!r}, exact inputs="
            f"{execution_contract['representation_plan_inputs']}, and icu_rule_refs "
            f"must contain {rule_ref}.",
        ),
    )


def build_runtime_scientific_projection(
    protocol: E2ScientificProtocol | H2ScientificProtocol | H3ScientificProtocol,
) -> RuntimeScientificProjection:
    """Compile the signed protocol into its sole deterministic runtime projection."""

    protocol_payload = protocol.model_dump(mode="json")
    canonical_protocol_json = _canonical_json_bytes(protocol_payload).decode("utf-8")
    if isinstance(protocol, E2ScientificProtocol):
        execution_contract = _e2_deterministic_execution_contract(protocol)
    elif isinstance(protocol, H2ScientificProtocol):
        execution_contract = _h2_deterministic_execution_contract(protocol)
    else:
        execution_contract = _h3_deterministic_execution_contract(protocol)
    required_outputs, guardrails = _projection_agent_content(
        protocol, execution_contract
    )
    body = {
        "schema_version": "easyicu.figure2_runtime_scientific_projection/1",
        "task_id": protocol.task_id,
        "protocol_version": protocol.protocol_version,
        "protocol_content_sha256": hashlib.sha256(
            canonical_protocol_json.encode("utf-8")
        ).hexdigest(),
        "canonical_protocol_json": canonical_protocol_json,
        "agent_visible_required_outputs": required_outputs,
        "agent_visible_guardrails": guardrails,
        "deterministic_execution_contract": execution_contract,
    }
    return RuntimeScientificProjection(
        **body,
        runtime_projection_sha256=hashlib.sha256(
            _canonical_json_bytes(body)
        ).hexdigest(),
    )


def load_runtime_scientific_projection(
    value: RuntimeScientificProjection | Mapping[str, Any],
) -> RuntimeScientificProjection:
    """Strictly validate a JSONL/runtime projection supplied by a caller."""

    if isinstance(value, RuntimeScientificProjection):
        return value
    # Validate through JSON so JSON arrays may populate frozen tuple fields while
    # scalar/object types remain strict.  This is the representation present in a
    # decoded benchmark JSONL row.
    return RuntimeScientificProjection.model_validate_json(
        _canonical_json_bytes(dict(value)),
        strict=True,
    )


def load_default_case_protocol(
    task_id: str,
) -> E2ScientificProtocol | H2ScientificProtocol | H3ScientificProtocol:
    return load_case_scientific_protocol(
        default_case_protocol_path(task_id),
        expected_task_id=task_id,
    )


__all__ = [
    "RuntimeScientificProjection",
    "E2ScientificProtocol",
    "H2ScientificProtocol",
    "H3ScientificProtocol",
    "ScientificCaseProtocolError",
    "build_runtime_scientific_projection",
    "case_protocol_content_sha256",
    "default_case_protocol_path",
    "load_case_scientific_protocol",
    "load_default_case_protocol",
    "load_runtime_scientific_projection",
]
