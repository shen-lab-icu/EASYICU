"""Digest-bound scientific authority for one fixed-window trajectory run.

This module owns the small, case-neutral contract between a caller-reviewed
trajectory protocol and the deterministic representation, candidate-selection,
and stability executors.  The caller chooses the concepts and numerical design;
the shared engine only validates and executes the immutable projection.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..schema import AnalysisPlan, TrajectoryStabilitySpec
from .plan_contract import (
    OBSERVED_DATA_DIAG_GMM_METHOD,
    trajectory_step_roles,
)


class TrajectoryScientificAuthorityError(ValueError):
    """A plan or artifact drifted from its caller-reviewed trajectory contract."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _normalise(value: Any) -> str:
    return "_".join(
        part
        for part in "".join(
            char.lower() if char.isalnum() else " " for char in str(value or "")
        ).split()
        if part
    )


class CoordinateScalingAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    method: Literal["pooled_coordinate_wise_z_score"]
    ddof: Literal[0]
    observed_value_policy: Literal["direct_or_owner_locf_available"]
    missing_value_policy: Literal["preserve_missing_exclude_from_likelihood"]
    zero_variance_action: Literal["fail_closed"]


class EvidenceStateAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    direct_observed: Literal["include"]
    owner_locf_available: Literal["include_and_audit"]
    unavailable: Literal["exclude"]
    additional_clustering_stage_imputation: Literal["none"]


class TrajectoryScientificRuntimeAuthority(BaseModel):
    """Immutable scientific inputs consumed by all three trajectory owners."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["easyicu.trajectory_scientific_runtime_authority/1"]
    protocol_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    coordinate_concepts: tuple[str, ...]
    descriptive_only_concepts: tuple[str, ...]
    window_start_hours: int
    window_end_hours: int
    grid_width_hours: int = Field(gt=0)
    aggregation: Literal["max"]
    representation_columns: tuple[str, ...]
    minimum_available_windows: int = Field(ge=1)
    coordinate_scaling: CoordinateScalingAuthority
    evidence_state_policy: EvidenceStateAuthority
    representation_plan_method: Literal[
        "signed_fixed_window_trajectory_representation"
    ]
    representation_plan_intent: str
    representation_plan_inputs: tuple[str, ...]
    representation_required_outputs: tuple[str, ...]
    model_family: Literal["latent_class_diagonal_gaussian_mixture"]
    fit_method: Literal["observed_data_em_diagonal_gaussian_mixture"]
    covariance_type: Literal["diag"]
    candidate_cluster_counts: tuple[int, ...]
    selection_criterion: Literal["bic"]
    selection_rule: Literal["minimum"]
    candidate_fit_base_seed: int = Field(ge=0, le=2_147_483_647)
    candidate_fit_max_iter: int = Field(ge=10, le=10_000)
    candidate_fit_tolerance: float = Field(gt=0.0, le=0.1)
    candidate_fit_regularization: float = Field(gt=0.0, le=1.0)
    bic_sample_size: Literal["frozen_population_rows"]
    bic_parameter_count: Literal["mixture_weights_k_minus_1_plus_2_k_per_coordinate"]
    bic_tie_break: Literal["smaller_k"]
    upper_boundary_action: Literal["fail_closed_if_selected_at_upper_boundary"]
    upper_boundary_reason_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,79}$")
    minimum_cluster_fraction: float = Field(gt=0.0, lt=1.0)
    minimum_cluster_fraction_reason_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,79}$")
    stability_spec: TrajectoryStabilitySpec
    execution_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _closed_contract(self) -> "TrajectoryScientificRuntimeAuthority":
        if len(self.coordinate_concepts) < 2 or len(
            set(self.coordinate_concepts)
        ) != len(self.coordinate_concepts):
            raise ValueError("trajectory coordinate concepts must be unique")
        if set(self.coordinate_concepts) & set(self.descriptive_only_concepts):
            raise ValueError("descriptive-only concepts cannot be model coordinates")
        if self.window_end_hours <= self.window_start_hours:
            raise ValueError("trajectory window must have positive width")
        span = self.window_end_hours - self.window_start_hours
        if span % self.grid_width_hours:
            raise ValueError("trajectory window must divide into a complete fixed grid")
        expected_columns = tuple(
            f"{concept}__h{start}_{start + self.grid_width_hours}"
            for concept in self.coordinate_concepts
            for start in range(
                self.window_start_hours,
                self.window_end_hours,
                self.grid_width_hours,
            )
        )
        if self.representation_columns != expected_columns:
            raise ValueError("representation columns drifted from concepts/fixed grid")
        if (
            len(self.candidate_cluster_counts) < 2
            or len(set(self.candidate_cluster_counts))
            != len(self.candidate_cluster_counts)
            or tuple(sorted(self.candidate_cluster_counts))
            != self.candidate_cluster_counts
            or self.candidate_cluster_counts[0] < 2
        ):
            raise ValueError(
                "candidate cluster counts must be a unique increasing grid"
            )
        body = self.model_dump(mode="json", exclude={"execution_contract_sha256"})
        if hashlib.sha256(_canonical_bytes(body)).hexdigest() != (
            self.execution_contract_sha256
        ):
            raise ValueError("trajectory scientific execution-contract digest mismatch")
        return self

    @property
    def scaling_payload(self) -> dict[str, Any]:
        return self.coordinate_scaling.model_dump(mode="json")

    @property
    def evidence_payload(self) -> dict[str, Any]:
        return self.evidence_state_policy.model_dump(mode="json")

    @property
    def plan_rule_ref(self) -> str:
        return f"scientific_runtime_contract:{self.execution_contract_sha256}"

    def validate_plan(self, plan: AnalysisPlan) -> None:
        owners: dict[str, list[Any]] = {
            role: [step for step in plan.steps if role in trajectory_step_roles(step)]
            for role in ("representation", "candidate_selection", "stability_freeze")
        }
        missing_or_ambiguous = {
            role: [step.step_id for step in steps]
            for role, steps in owners.items()
            if len(steps) != 1
        }
        if missing_or_ambiguous:
            raise TrajectoryScientificAuthorityError(
                "signed trajectory authority requires one owner per execution role: "
                f"{missing_or_ambiguous}"
            )
        representation = owners["representation"][0]
        candidate = owners["candidate_selection"][0]
        stability = owners["stability_freeze"][0]
        order = {step.step_id: index for index, step in enumerate(plan.steps)}
        if not (
            order[representation.step_id]
            < order[candidate.step_id]
            < order[stability.step_id]
        ):
            raise TrajectoryScientificAuthorityError(
                "signed trajectory execution roles are not ordered representation -> "
                "candidate selection -> stability"
            )
        representation_issues: list[str] = []
        if representation.method != self.representation_plan_method:
            representation_issues.append("method")
        if representation.intent != self.representation_plan_intent:
            representation_issues.append("intent")
        if tuple(representation.inputs) != self.representation_plan_inputs:
            representation_issues.append("inputs")
        if not set(self.representation_required_outputs).issubset(
            representation.expected_outputs
        ):
            representation_issues.append("expected_outputs")
        if self.plan_rule_ref not in set(representation.icu_rule_refs):
            representation_issues.append("scientific_runtime_contract")
        if representation_issues:
            raise TrajectoryScientificAuthorityError(
                "trajectory representation plan drifted from signed authority: "
                + ", ".join(representation_issues)
            )
        if _normalise(candidate.method) != OBSERVED_DATA_DIAG_GMM_METHOD:
            raise TrajectoryScientificAuthorityError(
                "candidate-selection method drifted from signed observed-data GMM"
            )
        observed_spec = stability.trajectory_stability_spec
        if observed_spec is None or observed_spec.model_dump(mode="json") != (
            self.stability_spec.model_dump(mode="json")
        ):
            raise TrajectoryScientificAuthorityError(
                "trajectory stability design drifted from signed execution contract"
            )

    def validate_representation_schema(self, schema: Mapping[str, Any]) -> None:
        issues: list[str] = []
        family = schema.get("observation_family")
        if not isinstance(family, list) or tuple(family) != self.coordinate_concepts:
            issues.append("observation_family")
        for field in ("observation_columns", "representation_columns"):
            value = schema.get(field)
            if (
                not isinstance(value, list)
                or tuple(value) != self.representation_columns
            ):
                issues.append(field)
        if schema.get("min_observed_windows") != self.minimum_available_windows:
            issues.append("min_observed_windows")
        if schema.get("coordinate_scaling") != self.scaling_payload:
            issues.append("coordinate_scaling")
        if schema.get("evidence_state_policy") != self.evidence_payload:
            issues.append("evidence_state_policy")
        expected_window = {
            "start_hours": self.window_start_hours,
            "end_hours": self.window_end_hours,
            "grid_width_hours": self.grid_width_hours,
            "aggregation": self.aggregation,
        }
        if schema.get("source_window_contract") != expected_window:
            issues.append("source_window_contract")
        binding = schema.get("scientific_runtime_authority")
        expected_binding = {
            "schema_version": self.schema_version,
            "protocol_content_sha256": self.protocol_content_sha256,
            "execution_contract_sha256": self.execution_contract_sha256,
        }
        if binding != expected_binding:
            issues.append("scientific_runtime_authority")
        if issues:
            raise TrajectoryScientificAuthorityError(
                "trajectory representation drifted from signed authority: "
                + ", ".join(issues)
            )

    def validate_selection(self, payload: Mapping[str, Any]) -> None:
        candidates = payload.get("candidates")
        candidate_k = (
            tuple(item.get("n_clusters") for item in candidates)
            if isinstance(candidates, list)
            and all(isinstance(item, Mapping) for item in candidates)
            else ()
        )
        expected = {
            "criterion": self.selection_criterion,
            "selection_rule": self.selection_rule,
            "direction": "minimize",
            "candidate_range_boundary_rule": self.upper_boundary_action,
            "candidate_range_boundary_reason_code": self.upper_boundary_reason_code,
        }
        issues = [
            field for field, value in expected.items() if payload.get(field) != value
        ]
        if candidate_k != self.candidate_cluster_counts:
            issues.append("candidate_cluster_counts")
        if issues:
            raise TrajectoryScientificAuthorityError(
                "cluster selection drifted from signed authority: " + ", ".join(issues)
            )


def load_trajectory_scientific_runtime_authority(
    value: TrajectoryScientificRuntimeAuthority | Mapping[str, Any],
) -> TrajectoryScientificRuntimeAuthority:
    if isinstance(value, TrajectoryScientificRuntimeAuthority):
        return value

    def plain(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): plain(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [plain(child) for child in item]
        return item

    return TrajectoryScientificRuntimeAuthority.model_validate_json(
        json.dumps(plain(value), sort_keys=True), strict=True
    )


def build_trajectory_scientific_runtime_authority(
    value: Mapping[str, Any],
) -> TrajectoryScientificRuntimeAuthority:
    """Seal caller-compiled fields into one self-digesting immutable contract."""

    body = dict(value)
    if "execution_contract_sha256" in body:
        raise ValueError("execution contract digest is host-generated")
    body["execution_contract_sha256"] = hashlib.sha256(
        _canonical_bytes(body)
    ).hexdigest()
    return load_trajectory_scientific_runtime_authority(body)


__all__ = [
    "CoordinateScalingAuthority",
    "EvidenceStateAuthority",
    "TrajectoryScientificAuthorityError",
    "TrajectoryScientificRuntimeAuthority",
    "build_trajectory_scientific_runtime_authority",
    "load_trajectory_scientific_runtime_authority",
]
