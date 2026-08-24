"""Typed validation for the deterministic cross-sectional phenotyping owner."""

from __future__ import annotations

import math
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .capability_ids import PHENOTYPING_ANALYSIS_KIND

_SELECTION_RULE = "maximum_silhouette_then_lower_k"


class PhenotypingCandidateReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_k: int = Field(ge=2)
    silhouette: float = Field(ge=-1, le=1)
    inertia: float = Field(ge=0)
    selected: bool
    selection_rule: Literal["maximum_silhouette_then_lower_k"]

    @model_validator(mode="after")
    def _finite_metrics(self) -> "PhenotypingCandidateReceipt":
        if not math.isfinite(self.silhouette) or not math.isfinite(self.inertia):
            raise ValueError("phenotyping candidate metrics must be finite")
        return self


class PhenotypingRuntimeReceipt(BaseModel):
    """Evidence for one bounded exploratory cluster solution, not a phenotype fact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["easyicu.cross_sectional_phenotyping_runtime_receipt/1"]
    analysis_kind: Literal["cross_sectional_phenotyping"]
    owner: Literal[
        "easyicu.research_agent.execution.runners.cross_sectional_phenotyping_executor"
    ]
    n_rows: int = Field(ge=20)
    feature_roster: list[str] = Field(min_length=2)
    preprocessing: Literal["median_imputation_then_standard_scaling"]
    clustering_method: Literal["minibatch_kmeans"]
    random_seed: Literal[1729]
    candidates: list[PhenotypingCandidateReceipt] = Field(min_length=1)
    selected_n_clusters: int = Field(ge=2)
    selected_silhouette_score: float = Field(ge=-1, le=1)
    cluster_counts: dict[str, int]
    source_cohort_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phenotype_profiles_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phenotype_assignments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome_used_for_fit: Literal[False]
    downstream_outcome_use: Literal["descriptive_only"]
    causal_entity_claim_authorized: Literal[False]
    external_reproducibility_established: Literal[False]
    paper_authorization_allowed: Literal[False]

    @model_validator(mode="after")
    def _coherent_solution(self) -> "PhenotypingRuntimeReceipt":
        if len(set(self.feature_roster)) != len(self.feature_roster):
            raise ValueError("phenotyping feature roster must be unique")
        if any(not value.strip() for value in self.feature_roster):
            raise ValueError("phenotyping feature roster contains an empty name")
        if not math.isfinite(self.selected_silhouette_score):
            raise ValueError("selected silhouette score must be finite")
        candidate_ks = [row.candidate_k for row in self.candidates]
        if len(set(candidate_ks)) != len(candidate_ks):
            raise ValueError("phenotyping candidate k values must be unique")
        selected = [row for row in self.candidates if row.selected]
        if len(selected) != 1:
            raise ValueError("phenotyping receipt requires exactly one selected k")
        expected = max(
            self.candidates,
            key=lambda row: (row.silhouette, -row.candidate_k),
        )
        if selected[0].candidate_k != expected.candidate_k:
            raise ValueError("selected k does not maximize the sealed criterion")
        if self.selected_n_clusters != selected[0].candidate_k:
            raise ValueError("selected cluster count disagrees with candidate grid")
        if not math.isclose(
            self.selected_silhouette_score,
            selected[0].silhouette,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("selected silhouette disagrees with candidate grid")
        expected_labels = {str(value) for value in range(self.selected_n_clusters)}
        if set(self.cluster_counts) != expected_labels:
            raise ValueError("cluster counts do not cover the selected labels")
        if any(value <= 0 for value in self.cluster_counts.values()):
            raise ValueError("every selected cluster must contain rows")
        if sum(self.cluster_counts.values()) != self.n_rows:
            raise ValueError("cluster counts do not sum to the fitted population")
        return self


def phenotyping_runtime_receipt_valid(summary: Any) -> bool:
    if not isinstance(summary, Mapping) or summary.get("status") != "ok":
        return False
    try:
        PhenotypingRuntimeReceipt.model_validate(
            summary.get("scientific_runtime_receipt")
        )
    except Exception:
        return False
    return True


def _summaries(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    values: list[Mapping[str, Any]] = []
    for record in records:
        summary = record.get("step_summary")
        if isinstance(summary, Mapping):
            values.append(summary)
    return values


def phenotyping_runtime_bundle_errors(
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Validate primary, k-selection, and stability as one scientific bundle."""

    summaries = _summaries(records)
    primary = [
        summary
        for summary in summaries
        if summary.get("deterministic_standard_analysis")
        == PHENOTYPING_ANALYSIS_KIND
        and "scientific_runtime_receipt" in summary
    ]
    if len(primary) != 1:
        return [
            "phenotyping validator requires exactly one sealed primary runtime receipt"
        ]
    try:
        receipt = PhenotypingRuntimeReceipt.model_validate(
            primary[0].get("scientific_runtime_receipt")
        )
    except Exception as exc:
        return [f"phenotyping primary runtime receipt is invalid: {exc}"]

    selections = [
        summary
        for summary in summaries
        if summary.get("deterministic_standard_analysis")
        == PHENOTYPING_ANALYSIS_KIND
        and summary.get("method")
        == "deterministic_cross_sectional_phenotyping_diagnostic"
        and isinstance(summary.get("cluster_selection"), Mapping)
    ]
    stabilities = [
        summary
        for summary in summaries
        if summary.get("deterministic_standard_analysis")
        == PHENOTYPING_ANALYSIS_KIND
        and summary.get("method")
        == "deterministic_cross_sectional_phenotyping_diagnostic"
        and isinstance(summary.get("cluster_stability"), Mapping)
    ]
    errors: list[str] = []
    if len(selections) != 1:
        errors.append(
            "phenotyping validator requires exactly one deterministic k-selection replay"
        )
    if len(stabilities) != 1:
        errors.append(
            "phenotyping validator requires exactly one deterministic stability replay"
        )
    if errors:
        return errors

    selection = selections[0]["cluster_selection"]
    try:
        selected_k = int(selection.get("selected_n_clusters"))
        candidates = [
            PhenotypingCandidateReceipt.model_validate(row)
            for row in selection.get("candidates", [])
        ]
    except Exception as exc:
        errors.append(f"phenotyping k-selection replay is invalid: {exc}")
    else:
        if selection.get("criterion") != "silhouette_score":
            errors.append("phenotyping k-selection criterion drifted")
        if selection.get("selection_rule") != _SELECTION_RULE:
            errors.append("phenotyping k-selection rule drifted")
        if selected_k != receipt.selected_n_clusters:
            errors.append("phenotyping k-selection disagrees with the primary receipt")
        candidate_grid_replayed = len(candidates) == len(receipt.candidates) and all(
            observed.candidate_k == expected.candidate_k
            and observed.selected is expected.selected
            and observed.selection_rule == expected.selection_rule
            and math.isclose(
                observed.silhouette,
                expected.silhouette,
                rel_tol=1e-5,
                abs_tol=1e-8,
            )
            and math.isclose(
                observed.inertia,
                expected.inertia,
                rel_tol=1e-5,
                abs_tol=1e-6,
            )
            for observed, expected in zip(candidates, receipt.candidates)
        )
        if not candidate_grid_replayed:
            errors.append("phenotyping k-selection candidate grid was not replayed")

    stability = stabilities[0]["cluster_stability"]
    try:
        stability_k = int(stability.get("selected_n_clusters"))
        n_resamples = int(stability.get("n_resamples"))
        mean_ari = float(stability.get("mean_adjusted_rand_index"))
        replicates = list(stability.get("replicates") or [])
        ari_values = [float(row["adjusted_rand_index"]) for row in replicates]
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"phenotyping stability replay is invalid: {exc}")
    else:
        if stability_k != receipt.selected_n_clusters:
            errors.append("phenotyping stability replay used a different k")
        if n_resamples < 5 or n_resamples != len(replicates):
            errors.append("phenotyping stability replay has insufficient resamples")
        if not ari_values or not all(
            math.isfinite(value) and -1 <= value <= 1 for value in ari_values
        ):
            errors.append("phenotyping stability replay contains invalid ARI values")
        elif not math.isfinite(mean_ari) or not math.isclose(
            mean_ari,
            sum(ari_values) / len(ari_values),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            errors.append("phenotyping stability mean does not match its resamples")
    return errors


__all__ = [
    "PhenotypingCandidateReceipt",
    "PhenotypingRuntimeReceipt",
    "phenotyping_runtime_bundle_errors",
    "phenotyping_runtime_receipt_valid",
]
