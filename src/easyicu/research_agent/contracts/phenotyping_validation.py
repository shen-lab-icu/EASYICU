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


class PhenotypingCompleteCaseReceipt(BaseModel):
    """One locked missingness sensitivity for the cluster solution.

    The metric is label-invariant agreement between the primary assignments
    and a complete-case refit on the same rows.  Its interval is a deterministic
    paired bootstrap of those two assignment vectors; it is not an interval for
    biological or external reproducibility.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "easyicu.cross_sectional_phenotyping_complete_case_receipt/1"
    ]
    spec_id: str = Field(min_length=1)
    axis: Literal["missing"]
    missing_strategy: Literal["complete_case"]
    complete_case_variables: list[str] = Field(min_length=1)
    primary_feature_roster: list[str] = Field(min_length=2)
    n_total: int = Field(ge=20)
    n_complete: int = Field(ge=20)
    primary_selected_n_clusters: int = Field(ge=2)
    complete_case_selected_n_clusters: int = Field(ge=2)
    complete_case_candidates: list[PhenotypingCandidateReceipt] = Field(min_length=1)
    comparison_metric: Literal["adjusted_rand_index"]
    point_estimate: float = Field(ge=-1, le=1)
    ci_low: float = Field(ge=-1, le=1)
    ci_high: float = Field(ge=-1, le=1)
    standard_error: float = Field(ge=0)
    interval_method: Literal["paired_assignment_bootstrap_percentile_95"]
    n_bootstrap: int = Field(ge=100)
    random_seed: Literal[1729]
    primary_preprocessing: Literal["median_imputation_then_standard_scaling"]
    sensitivity_preprocessing: Literal["complete_case_then_standard_scaling"]
    clustering_method: Literal["minibatch_kmeans"]
    table_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome_used_for_fit: Literal[False]
    causal_entity_claim_authorized: Literal[False]
    paper_authorization_allowed: Literal[False]

    @model_validator(mode="after")
    def _coherent_sensitivity(self) -> "PhenotypingCompleteCaseReceipt":
        if len(set(self.complete_case_variables)) != len(
            self.complete_case_variables
        ):
            raise ValueError("complete-case variables must be unique")
        if len(set(self.primary_feature_roster)) != len(self.primary_feature_roster):
            raise ValueError("primary feature roster must be unique")
        if not set(self.complete_case_variables).issubset(
            self.primary_feature_roster
        ):
            raise ValueError("complete-case variables must belong to the primary roster")
        if self.n_complete > self.n_total:
            raise ValueError("complete-case population exceeds the primary population")
        selected = [row for row in self.complete_case_candidates if row.selected]
        if len(selected) != 1:
            raise ValueError("complete-case sensitivity requires one selected k")
        expected = max(
            self.complete_case_candidates,
            key=lambda row: (row.silhouette, -row.candidate_k),
        )
        if selected[0].candidate_k != expected.candidate_k:
            raise ValueError("complete-case selected k does not maximize silhouette")
        if self.complete_case_selected_n_clusters != selected[0].candidate_k:
            raise ValueError("complete-case selected k disagrees with its candidate grid")
        if not all(
            math.isfinite(value)
            for value in (
                self.point_estimate,
                self.ci_low,
                self.ci_high,
                self.standard_error,
            )
        ):
            raise ValueError("complete-case agreement metrics must be finite")
        if self.ci_low > self.ci_high:
            raise ValueError("complete-case agreement interval is reversed")
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
    complete_case_sensitivities: list[PhenotypingCompleteCaseReceipt] = Field(
        default_factory=list
    )
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
        sensitivity_ids = [row.spec_id for row in self.complete_case_sensitivities]
        if len(sensitivity_ids) != len(set(sensitivity_ids)):
            raise ValueError("complete-case sensitivity spec ids must be unique")
        if any(
            row.primary_feature_roster != self.feature_roster
            or row.n_total != self.n_rows
            or row.primary_selected_n_clusters != self.selected_n_clusters
            for row in self.complete_case_sensitivities
        ):
            raise ValueError(
                "complete-case sensitivity disagrees with the primary phenotype receipt"
            )
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

    robustness_rows = {
        str(row.get("spec_id") or ""): row
        for row in (primary[0].get("robustness_rows") or [])
        if isinstance(row, Mapping)
    }
    for sensitivity in receipt.complete_case_sensitivities:
        row = robustness_rows.get(sensitivity.spec_id)
        if row is None:
            errors = [
                "phenotyping complete-case receipt lacks its robustness panel row"
            ]
            break
        try:
            row_matches = (
                row.get("axis") == "missing"
                and bool(row.get("converged"))
                and int(row.get("n")) == sensitivity.n_complete
                and math.isclose(
                    float(row.get("point_estimate")),
                    sensitivity.point_estimate,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(row.get("ci_low")),
                    sensitivity.ci_low,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(row.get("ci_high")),
                    sensitivity.ci_high,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            )
        except (TypeError, ValueError):
            row_matches = False
        if not row_matches:
            errors = [
                "phenotyping complete-case receipt disagrees with its robustness row"
            ]
            break
    else:
        errors = []

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
    "PhenotypingCompleteCaseReceipt",
    "PhenotypingRuntimeReceipt",
    "phenotyping_runtime_bundle_errors",
    "phenotyping_runtime_receipt_valid",
]
