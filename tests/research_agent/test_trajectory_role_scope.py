from __future__ import annotations

import inspect
import json
from pathlib import Path

from easyicu.research_agent.contracts.declared_product import (
    declared_product_contract_findings,
)
from easyicu.research_agent.gates import contract as contract_gate
from easyicu.research_agent.gates.method_compatibility import (
    detect_forbidden_pattern_usage,
)
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    ResearchContext,
)
from easyicu.research_agent.trajectory.plan_contract import (
    trajectory_role_code_contract,
)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Discover trajectory phenotypes.",
        cohort=CohortDescriptor(
            cohort_name="test", database="test", n_patients=10, n_stays=10
        ),
        variables=[],
    )


def _representation_step() -> AnalysisStep:
    return AnalysisStep(
        step_id="representation",
        intent="Build the agent-selected trajectory representation.",
        method="missingness_aware_rank_preserving_functional_representation",
        expected_outputs=[
            "artifact:trajectory_representation",
            "table:trajectory_membership",
        ],
    )


def test_representation_code_cannot_fit_or_select_clusters():
    code = """
from sklearn.cluster import MiniBatchKMeans
model = MiniBatchKMeans(n_clusters=3).fit_predict(X)
pd.DataFrame(rows).to_csv(out_dir / 'cluster_selection.csv')
"""

    violations = detect_forbidden_pattern_usage(
        code,
        _context(),
        _representation_step(),
    )

    assert any(item["kind"] == "trajectory_role_scope" for item in violations)


def test_representation_summary_cannot_hide_downstream_products_as_diagnostics():
    findings = declared_product_contract_findings(
        step=_representation_step(),
        step_summary={
            "status": "ok",
            "outputs": {
                "trajectory_representation": "trajectory_representation.parquet",
                "trajectory_membership": "trajectory_membership.csv",
            },
            "diagnostic_files": {
                "cluster_selection": "cluster_selection.csv",
                "cluster_stability": "cluster_stability.csv",
                "cluster_profile": "cluster_profile.csv",
            },
        },
        effect_method_authorized=False,
    )

    assert any(
        finding.detail.get("kind") == "trajectory_role_product_out_of_scope"
        for finding in findings
    )


def test_representation_audits_and_source_companions_remain_allowed():
    findings = declared_product_contract_findings(
        step=_representation_step(),
        step_summary={
            "status": "ok",
            "outputs": {
                "trajectory_representation": "trajectory_representation.parquet",
                "trajectory_membership": "trajectory_membership.csv",
            },
            "diagnostic_files": {
                "feature_quality_audit": "feature_quality_audit.csv",
                "input_reconciliation": "input_reconciliation.csv",
            },
        },
        effect_method_authorized=False,
    )

    assert not any(finding.severity == "error" for finding in findings)


def _candidate_step() -> AnalysisStep:
    return AnalysisStep(
        step_id="candidate_selection",
        intent="Compare agent-selected candidate cluster solutions.",
        method="model_based_latent_class_clustering",
        expected_outputs=[
            "artifact:candidate_cluster_models",
            "manifest:cluster_selection",
        ],
    )


def test_candidate_role_rejects_explicit_failed_selection_payload():
    findings = declared_product_contract_findings(
        step=_candidate_step(),
        step_summary={
            "status": "ok",
            "cluster_selection": {
                "criterion": "bic",
                "selection_rule": "minimum",
                "direction": "minimize",
                "selected_n_clusters": None,
                "candidates": [],
                "rationale": "Candidate fitting failed.",
            },
        },
        effect_method_authorized=False,
    )

    assert any(
        finding.detail.get("kind") == "trajectory_candidate_selection_invalid"
        for finding in findings
    )


def test_candidate_role_accepts_replayable_agent_selection():
    selection = {
        "criterion": "bic",
        "selection_rule": "minimum",
        "direction": "minimize",
        "selected_n_clusters": 3,
        "candidates": [
            {"n_clusters": 2, "criterion_value": 120.0},
            {"n_clusters": 3, "criterion_value": 100.0},
        ],
        "rationale": "Selected the finite minimum BIC.",
        "candidate_range_boundary_rule": "allow_upper_boundary",
        "candidate_range_boundary_reason_code": None,
    }
    findings = declared_product_contract_findings(
        step=_candidate_step(),
        step_summary={
            "status": "ok",
            "cluster_selection": selection,
            "n_clusters": 3,
        },
        effect_method_authorized=False,
    )

    assert not any(finding.severity == "error" for finding in findings)


def test_candidate_only_role_rejects_bootstrap_stability_output():
    findings = declared_product_contract_findings(
        step=_candidate_step(),
        step_summary={
            "status": "ok",
            "output_files": [
                "candidate_cluster_models.json",
                "cluster_selection.json",
                "bootstrap_stability.csv",
            ],
            "cluster_selection": {
                "criterion": "bic",
                "selection_rule": "minimum",
                "direction": "minimize",
                "selected_n_clusters": 2,
                "candidates": [
                    {"n_clusters": 2, "criterion_value": 10.0},
                    {"n_clusters": 3, "criterion_value": 12.0},
                ],
                "rationale": "The minimum finite BIC was selected.",
            },
        },
        effect_method_authorized=False,
    )

    assert any(
        finding.detail.get("kind") == "trajectory_role_product_out_of_scope"
        and "stability_freeze"
        in finding.detail.get("unauthorized_products_by_role", {})
        for finding in findings
    )


def test_candidate_code_contract_consumes_one_upstream_coordinate_layer():
    step = _candidate_step().model_copy(
        update={
            "inputs": [
                "artifact:trajectory_representation",
                "table:scaling_summary",
            ]
        }
    )

    contract = trajectory_role_code_contract(context=_context(), step=step)

    assert "scaled_representation_column" in contract
    assert "do not reapply cohort, anchor, or observed-window eligibility" in contract
    assert "cluster_selection.json is a closed selection-only manifest" in contract
    assert "Do not add role, id_column, clustering_method, model_family" in contract
    assert "Copy id_column into the cluster-selection manifest" not in contract
    assert "candidate_cluster_solution_schema.json" in contract
    assert "easyicu.candidate_cluster_solution_schema/2" in contract
    assert "observed_data_em_diagonal_gaussian_mixture" in contract
    assert "median/zero imputation" in contract
    assert "do not run bootstrap" in contract
    assert "do not write cluster profiles" in contract


def test_general_clustering_does_not_receive_fixed_trajectory_code_contract():
    step = AnalysisStep(
        step_id="cluster_discovery",
        intent="Select and characterize first-24-hour stay-level clusters.",
        method="unsupervised_clustering",
        expected_outputs=[
            "manifest:cluster_selection",
            "artifact:cluster_assignments",
            "table:cluster_characteristics",
        ],
    )

    assert (
        trajectory_role_code_contract(
            context=_context(),
            step=step,
            applies=False,
        )
        == ""
    )


def test_general_clustering_is_not_judged_by_fixed_trajectory_result_contract():
    step = AnalysisStep(
        step_id="cluster_discovery",
        intent="Select and characterize first-24-hour stay-level clusters.",
        method="unsupervised_clustering",
        expected_outputs=[
            "manifest:cluster_selection",
            "artifact:cluster_assignments",
            "table:cluster_characteristics",
        ],
    )

    findings = declared_product_contract_findings(
        step=step,
        step_summary={
            "status": "ok",
            "output_files": [
                "cluster_selection.json",
                "cluster_assignments.csv",
                "cluster_characteristics.csv",
            ],
        },
        effect_method_authorized=False,
        trajectory_role_contract_applies=False,
    )

    assert not any(
        finding.validator in {"trajectory_role_scope", "trajectory_role_result"}
        for finding in findings
    )


def test_shared_step_gate_passes_one_compiled_trajectory_applicability_decision():
    source = inspect.getsource(contract_gate._step_deterministic_contract_findings)

    assert "trajectory_plan_contract_applies(" in source
    assert "trajectory_role_contract_applies=" in source


def test_representation_role_rejects_unversioned_prose_schema(tmp_path: Path):
    step = _representation_step().model_copy(
        update={
            "expected_outputs": [
                "artifact:trajectory_representation",
                "table:trajectory_membership",
                "manifest:trajectory_representation_schema",
            ]
        }
    )
    (tmp_path / "trajectory_representation_schema.json").write_text(
        json.dumps(
            {
                "id_column": "encounter_key",
                "observation_family": "organ_state",
                "observation_columns": ["state_t0", "state_t1"],
                "profile_columns": ["state_t0", "state_t1"],
                "representation_columns": ["state_t0", "state_t1"],
                "frozen_population_n": 10,
                "representation_sha256": "a" * 64,
                "anchor_provenance": "fixed columns from cohort",
                "trailing_na_policy": "preserve missingness",
            }
        ),
        encoding="utf-8",
    )

    findings = declared_product_contract_findings(
        step=step,
        step_summary={"status": "ok"},
        effect_method_authorized=False,
        out_dir=tmp_path,
    )

    finding = next(
        item
        for item in findings
        if item.detail.get("kind") == "trajectory_representation_schema_incomplete"
    )
    assert "schema_version is missing or unsupported" in finding.detail["issues"]
    assert any("trailing_na_policy" in issue for issue in finding.detail["issues"])


def test_candidate_role_rejects_imputed_sklearn_schema_for_observed_data_method(
    tmp_path: Path,
):
    step = AnalysisStep(
        step_id="candidate_selection",
        intent="Fit a prespecified candidate family.",
        method="observed_data_diagonal_gaussian_mixture_candidate_selection",
        inputs=["artifact:trajectory_representation"],
        expected_outputs=[
            "artifact:candidate_cluster_assignments",
            "manifest:cluster_selection",
            "manifest:candidate_cluster_solution_schema",
        ],
    )
    (tmp_path / "candidate_cluster_solution_schema.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu.candidate_cluster_solution_schema/2",
                "id_column": "encounter_key",
                "representation_columns": ["state_t0", "state_t1"],
                "model_family": "diagonal_gaussian_mixture",
                "fit_method": "GaussianMixture",
                "covariance_type": "diag",
                "selected_n_clusters": 2,
                "selected_model_id": "candidate_2",
                "assignment_column": "cluster",
                "criterion": "BIC",
                "selection_rule": "minimum",
                "direction": "minimize",
                "selected_criterion_value": 10.0,
                "representation_schema_sha256": "a" * 64,
                "candidate_assignments_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    selection = {
        "criterion": "BIC",
        "selection_rule": "minimum",
        "direction": "minimize",
        "selected_n_clusters": 2,
        "candidates": [
            {"n_clusters": 2, "criterion_value": 10.0},
            {"n_clusters": 3, "criterion_value": 12.0},
        ],
        "rationale": "Minimum finite BIC.",
        "candidate_range_boundary_rule": "allow_upper_boundary",
        "candidate_range_boundary_reason_code": None,
    }

    findings = declared_product_contract_findings(
        step=step,
        step_summary={"status": "ok", "cluster_selection": selection},
        effect_method_authorized=False,
        out_dir=tmp_path,
    )

    finding = next(
        item
        for item in findings
        if item.detail.get("kind") == "trajectory_candidate_schema_incomplete"
    )
    assert any("model_family" in issue for issue in finding.detail["issues"])
    assert any("fit_method" in issue for issue in finding.detail["issues"])


def test_stability_code_contract_reuses_frozen_candidate_solution_only():
    step = AnalysisStep(
        step_id="stability_freeze",
        intent="Assess resampling stability and freeze the selected solution.",
        method="model_based_clustering_with_bootstrap_stability",
        inputs=[
            "artifact:trajectory_representation",
            "artifact:candidate_cluster_models",
            "artifact:candidate_cluster_assignments",
            "manifest:cluster_selection",
        ],
        expected_outputs=[
            "manifest:trajectory_missingness_policy",
            "table:cluster_assignments",
            "table:cluster_stability",
            "table:cluster_stability_assignments",
        ],
    )

    contract = trajectory_role_code_contract(context=_context(), step=step)

    assert "EASYICU_RESOLVED_INPUTS_JSON" in contract
    assert "typed schema manifests as authoritative" in contract
    assert "Apply each schema to its own contract" in contract
    assert "Never require representation coordinates" in contract
    assert "exact model family, fit_method" in contract
    assert "Do not substitute a complete-data estimator" in contract
    assert "do not read COHORT_PARQUET" in contract
    assert "exact representation_columns in the same order" in contract
    assert "copy the selected candidate labels" in contract
    assert "exactly one shared column" in contract
    assert "complete and unique in both tables" in contract
    assert "full identifier sets are equal" in contract
    assert "Never select an identifier by its name" in contract
    assert "accept only one fitted candidate-model record" in contract
    assert "evidence_id plus selected_n_clusters" in contract
    assert "Fail closed if zero or multiple records match" in contract
    assert "every fitted candidate record" in contract
    assert "one identical normalized method family" in contract
    assert "Never infer the method from" in contract
    assert "Do not compare candidate k values" in contract
    assert "same method and same k" in contract
    assert "at least two genuinely distinct resamples/refits" in contract
    assert "no candidate-selection table, cluster sizes, profiles" in contract


def test_stability_summary_cannot_claim_characterization_products():
    step = AnalysisStep(
        step_id="stability_freeze",
        intent="Assess resampling stability and freeze the selected solution.",
        method="model_based_clustering_with_bootstrap_stability",
        expected_outputs=[
            "manifest:trajectory_missingness_policy",
            "table:cluster_assignments",
            "table:cluster_stability",
            "table:cluster_stability_assignments",
        ],
    )

    findings = declared_product_contract_findings(
        step=step,
        step_summary={
            "status": "ok",
            "output_files": [
                "trajectory_missingness_policy.json",
                "cluster_assignments.csv",
                "cluster_stability.csv",
                "cluster_stability_assignments.csv",
                "cluster_sizes.csv",
                "trajectory_profiles.csv",
            ],
        },
        effect_method_authorized=False,
    )

    assert any(
        finding.detail.get("kind") == "trajectory_role_product_out_of_scope"
        and "characterization"
        in finding.detail.get("unauthorized_products_by_role", {})
        for finding in findings
    )
