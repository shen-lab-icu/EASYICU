from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.trajectory_scientific_candidate_executor import (
    run_trajectory_scientific_candidate_selection,
)
from easyicu.research_agent.execution.runners.trajectory_scientific_representation_executor import (
    run_trajectory_scientific_representation,
)
from easyicu.research_agent.execution.runners.trajectory_stability_executor import (
    run_trajectory_stability,
)
from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    CohortDescriptor,
    ResearchContext,
    TrajectoryStabilitySpec,
)
from easyicu.research_agent.trajectory.plan_contract import (
    evaluate_trajectory_plan_dag,
    trajectory_step_roles,
)
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    TrajectoryScientificAuthorityError,
    build_trajectory_scientific_runtime_authority,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(run_dir: Path, path: Path, evidence_id: str) -> dict[str, str]:
    return {
        "relative_path": str(path.relative_to(run_dir)),
        "sha256": _sha256(path),
        "evidence_id": evidence_id,
    }


def _authority():
    columns = (
        "sofa2_resp__h0_12",
        "sofa2_resp__h12_24",
        "lact__h0_12",
        "lact__h12_24",
    )

    spec = TrajectoryStabilitySpec(
        n_resamples=2,
        sample_fraction=0.75,
        base_seed=1729,
        minimum_successful_resamples=2,
        refit_max_iter=500,
        refit_tolerance=1e-5,
        refit_regularization=1e-6,
        minimum_mean_stability=0.0,
        decision_mode="minimum_mean_threshold",
    )
    return build_trajectory_scientific_runtime_authority(
        {
            "schema_version": "easyicu.trajectory_scientific_runtime_authority/1",
            "protocol_content_sha256": "1" * 64,
            "coordinate_concepts": ["sofa2_resp", "lact"],
            "descriptive_only_concepts": ["sofa2"],
            "window_start_hours": 0,
            "window_end_hours": 24,
            "grid_width_hours": 12,
            "aggregation": "max",
            "representation_columns": list(columns),
            "minimum_available_windows": 2,
            "coordinate_scaling": {
                "method": "pooled_coordinate_wise_z_score",
                "ddof": 0,
                "observed_value_policy": "direct_or_owner_locf_available",
                "missing_value_policy": "preserve_missing_exclude_from_likelihood",
                "zero_variance_action": "fail_closed",
            },
            "evidence_state_policy": {
                "direct_observed": "include",
                "owner_locf_available": "include_and_audit",
                "unavailable": "exclude",
                "additional_clustering_stage_imputation": "none",
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
            "candidate_cluster_counts": [2, 3, 4],
            "selection_criterion": "bic",
            "selection_rule": "minimum",
            "candidate_fit_base_seed": 1729,
            "candidate_fit_max_iter": 500,
            "candidate_fit_tolerance": 1e-5,
            "candidate_fit_regularization": 1e-6,
            "bic_sample_size": "frozen_population_rows",
            "bic_parameter_count": (
                "mixture_weights_k_minus_1_plus_2_k_per_coordinate"
            ),
            "bic_tie_break": "smaller_k",
            "upper_boundary_action": "fail_closed_if_selected_at_upper_boundary",
            "upper_boundary_reason_code": "H3_NO_INTERIOR_BIC_OPTIMUM",
            "minimum_cluster_fraction": 0.05,
            "minimum_cluster_fraction_reason_code": (
                "H3_MINIMUM_CLUSTER_FRACTION_NOT_MET"
            ),
            "stability_spec": spec.model_dump(mode="json"),
        }
    )


def _signed_plan(authority) -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "research_question": "Assess fixed-window trajectory phenotypes.",
            "analysis_type": "trajectory_clustering",
            "steps": [
                {
                    "step_id": "01_representation",
                    "planned_analysis_role": "auxiliary",
                    "intent": authority.representation_plan_intent,
                    "inputs": list(authority.representation_plan_inputs),
                    "expected_outputs": [
                        *authority.representation_required_outputs,
                        "table:feature_availability",
                    ],
                    "method": authority.representation_plan_method,
                    "icu_rule_refs": [authority.plan_rule_ref],
                },
                {
                    "step_id": "02_candidates",
                    "planned_analysis_role": "primary",
                    "intent": "Fit every signed candidate and select by BIC.",
                    "inputs": [
                        "artifact:trajectory_representation",
                        "manifest:trajectory_representation_schema",
                    ],
                    "expected_outputs": [
                        "artifact:candidate_cluster_assignments",
                        "manifest:cluster_selection",
                        "manifest:candidate_cluster_solution_schema",
                    ],
                    "method": (
                        "observed_data_diagonal_gaussian_mixture_candidate_selection"
                    ),
                    "icu_rule_refs": [],
                },
                {
                    "step_id": "03_stability",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Execute the signed stability design.",
                    "inputs": [
                        "artifact:trajectory_representation",
                        "artifact:candidate_cluster_assignments",
                        "manifest:cluster_selection",
                        "manifest:trajectory_representation_schema",
                        "manifest:candidate_cluster_solution_schema",
                    ],
                    "expected_outputs": [
                        "artifact:stability_freeze",
                        "artifact:cluster_assignments",
                        "manifest:cluster_stability_spec",
                        "manifest:trajectory_missingness_policy",
                        "table:cluster_assignments",
                        "table:cluster_stability",
                        "table:cluster_stability_assignments",
                        "table:cluster_assignment_provenance",
                    ],
                    "method": "trajectory_cluster_stability",
                    "icu_rule_refs": [],
                    "trajectory_stability_spec": authority.stability_spec.model_dump(
                        mode="json"
                    ),
                },
            ],
        }
    )


def test_signed_plan_rejects_representation_inputs_and_intent_drift() -> None:
    authority = _authority()
    plan = _signed_plan(authority)
    authority.validate_plan(plan)

    representation = plan.steps[0].model_copy(
        update={
            "inputs": ["sofa2", "sofa2_resp"],
            "intent": "Use total plus one component, 24h means, and zero imputation.",
        }
    )
    drifted = plan.model_copy(update={"steps": [representation, *plan.steps[1:]]})
    with pytest.raises(ValueError, match="representation plan drifted"):
        authority.validate_plan(drifted)


def test_signed_trajectory_contract_owns_all_four_real_execution_steps() -> None:
    authority = _authority()
    plan = authority.development_execution_only_plan(
        research_question="Assess fixed-window trajectory phenotypes."
    )
    expected_kinds = (
        "trajectory_signed_representation",
        "trajectory_signed_candidate_selection",
        "trajectory_cluster_stability",
        "trajectory_selection_diagnostic_figure",
    )
    for step, expected_kind in zip(plan.steps, expected_kinds):
        selected = select_standard_executor(
            step,
            plan=plan,
            trajectory_scientific_runtime_authority=authority,
            scientific_runtime_projection_sha256="2" * 64,
        )
        assert selected is not None
        assert selected.analysis_kind == expected_kind
        if expected_kind != "trajectory_selection_diagnostic_figure":
            assert authority.execution_contract_sha256 in selected.code


def test_signed_trajectory_authority_projects_and_rebinds_execution_only_plan() -> None:
    authority = _authority()
    plan = authority.development_execution_only_plan(
        research_question="Assess fixed-window trajectory phenotypes."
    )
    authority.validate_plan(plan)
    assert [step.step_id for step in plan.steps] == list(
        authority.development_execution_step_ids
    )
    assert [step.method for step in plan.steps] == [
        authority.representation_plan_method,
        "observed_data_diagonal_gaussian_mixture_candidate_selection",
        "trajectory_cluster_stability_characterization",
        "signed_trajectory_selection_diagnostic_figure",
    ]
    assert "manifest:trajectory_window_manifest" in plan.steps[0].expected_outputs
    assert "table:trajectory_candidate_selection" in plan.steps[1].expected_outputs
    assert plan.steps[-1].expected_outputs == [
        "figure:trajectory_selection_diagnostics"
    ]
    assert trajectory_step_roles(plan.steps[2]) == frozenset(
        {"stability_freeze", "characterization"}
    )
    context = ResearchContext(
        research_question=plan.research_question,
        cohort=CohortDescriptor(
            cohort_name="signed-long-trajectory",
            database="test",
            n_patients=120,
            n_stays=120,
            id_columns=["stay_id"],
        ),
        variables=[],
        target_outcome="death",
    )
    evaluation = evaluate_trajectory_plan_dag(
        plan=plan,
        context=context,
        long_trajectory_bound=True,
    )
    assert [
        finding
        for finding in evaluation.findings
        if finding.severity == "error"
    ] == []

    authorities = ScientificRuntimeAuthorities(
        trajectory=authority,
        current_case=None,
    )
    projected = authorities.development_execution_only_plan(
        research_question=plan.research_question
    )
    assert projected is not None
    projected_plan, projected_finding = projected
    authority.validate_plan(projected_plan)
    assert projected_finding.detail["reason_code"] == (
        "trajectory_development_execution_only_authority_compiled"
    )

    generic_step = plan.steps[-1].model_copy(
        update={
            "step_id": "03_generic_article_figure",
            "method": "visualization",
            "intent": "Add a generic article figure.",
            "expected_outputs": ["figure:generic_article_figure"],
            "trajectory_stability_spec": None,
        }
    )
    rebound, rebound_findings = authorities.bind_plan(
        plan.model_copy(update={"steps": [*plan.steps, generic_step]})
    )
    authority.validate_plan(rebound)
    assert len(rebound.steps) == 4
    assert rebound_findings[0].detail["reason_code"] == (
        "trajectory_development_execution_only_authority_compiled"
    )

    legacy_checkpoint = plan.model_copy(update={"steps": plan.steps[:3]})
    assert authority.is_development_execution_only_plan(legacy_checkpoint)
    upgraded, upgrade_findings = authorities.bind_plan(legacy_checkpoint)
    authority.validate_plan(upgraded)
    assert len(upgraded.steps) == 4
    assert upgraded.steps[-1].step_id == (
        "03_authority_compiled_trajectory_selection_figure"
    )
    assert upgrade_findings[0].detail["reason_code"] == (
        "trajectory_development_execution_only_authority_compiled"
    )

    host_capped_prefix = plan.model_copy(update={"steps": plan.steps[:2]})
    assert authority.is_development_execution_only_plan(host_capped_prefix)
    rebuilt, rebuild_findings = authorities.bind_plan(host_capped_prefix)
    authority.validate_plan(rebuilt)
    assert tuple(step.step_id for step in rebuilt.steps) == (
        "00_authority_compiled_trajectory_representation",
        "01_authority_compiled_trajectory_candidates",
        "02_authority_compiled_trajectory_stability",
        "03_authority_compiled_trajectory_selection_figure",
    )
    assert rebuild_findings[0].detail["reason_code"] == (
        "trajectory_development_execution_only_authority_compiled"
    )


def test_signed_representation_excludes_owner_unavailable_zero(tmp_path: Path) -> None:
    authority = _authority()
    rows = []
    for stay_id in range(1, 9):
        for charttime, resp, lact in ((0.0, 0.0, 1.0), (12.0, 2.0, 2.0)):
            rows.extend(
                [
                    {
                        "stay_id": stay_id,
                        "charttime": charttime,
                        "concept": "sofa2_resp",
                        "value_num": resp,
                        "value_str": str(resp),
                        "evidence_state": "direct_observed",
                        "owner_observed": 1,
                        "owner_available": 1,
                    },
                    {
                        "stay_id": stay_id,
                        "charttime": charttime,
                        "concept": "lact",
                        "value_num": lact,
                        "value_str": str(lact),
                        "evidence_state": "direct_observed",
                        "owner_observed": 1,
                        "owner_available": 1,
                    },
                ]
            )
        rows.append(
            {
                "stay_id": stay_id,
                "charttime": 6.0,
                "concept": "sofa2_resp",
                "value_num": 99.0,
                "value_str": "99",
                "evidence_state": "unavailable",
                "owner_observed": 0,
                "owner_available": 0,
            }
        )
    for concept in ("sofa2_resp", "lact"):
        rows.append(
            {
                "stay_id": 9,
                "charttime": 0.0,
                "concept": concept,
                "value_num": None,
                "value_str": None,
                "evidence_state": "unavailable",
                "owner_observed": 0,
                "owner_available": 0,
            }
        )
    trajectory_path = tmp_path / "trajectory.parquet"
    pd.DataFrame(rows).to_parquet(trajectory_path, index=False)
    out_dir = tmp_path / "representation"

    summary = run_trajectory_scientific_representation(
        authority=authority,
        runtime_projection_sha256="2" * 64,
        trajectory_path=trajectory_path,
        out_dir=out_dir,
    )

    assert summary["status"] == "ok"
    representation = pd.read_parquet(out_dir / "trajectory_representation.parquet")
    assert representation["sofa2_resp__h0_12"].max() == 0.0
    membership = pd.read_csv(out_dir / "trajectory_membership.csv")
    unavailable = membership.loc[membership["stay_id"].eq(9)].iloc[0]
    assert not bool(unavailable["included_in_clustering"])
    flow = dict(
        pd.read_csv(out_dir / "cohort_flow.csv")[["metric", "n"]].itertuples(
            index=False, name=None
        )
    )
    assert flow == {
        "input_cohort": 9,
        "meets_min_observed_windows": 8,
        "excluded_insufficient_windows": 1,
        "included_in_clustering": 8,
    }
    schema = json.loads(
        (out_dir / "trajectory_representation_schema.json").read_text("utf-8")
    )
    authority.validate_representation_schema(schema)
    window_manifest = json.loads(
        (out_dir / "trajectory_window_manifest.json").read_text("utf-8")
    )
    assert window_manifest["panel_product"] == "artifact:trajectory_representation"
    assert [item["family"] for item in window_manifest["families"]] == list(
        authority.coordinate_concepts
    )
    assert summary["output_files"]["manifest:trajectory_window_manifest"] == (
        "trajectory_window_manifest.json"
    )


def test_signed_candidate_and_stability_share_one_scaling_and_selection_contract(
    tmp_path: Path,
) -> None:
    authority = _authority()
    run_dir = tmp_path
    upstream = run_dir / "upstream"
    upstream.mkdir()
    rng = np.random.default_rng(44)
    labels = np.repeat([0, 1, 2], 40)
    centers = np.asarray(
        [
            [-6.0, -4.0, -2.0, -1.0],
            [0.0, 0.0, 0.0, 0.0],
            [6.0, 4.0, 2.0, 1.0],
        ]
    )
    matrix = centers[labels] + rng.normal(0.0, 0.2, size=(len(labels), 4))
    representation = pd.DataFrame(
        matrix, columns=list(authority.representation_columns)
    )
    representation.insert(0, "stay_id", np.arange(1, len(labels) + 1))
    representation_path = upstream / "trajectory_representation.parquet"
    representation.to_parquet(representation_path, index=False)
    schema = {
        "schema_version": "easyicu.trajectory_representation_schema/2",
        "id_column": "stay_id",
        "observation_family": list(authority.coordinate_concepts),
        "observation_columns": list(authority.representation_columns),
        "min_observed_windows": authority.minimum_available_windows,
        "profile_columns": list(authority.representation_columns),
        "profile_summary_statistic": "mean",
        "time_axis": "relative_hours",
        "anchor": "icu_admission",
        "anchor_provenance": "task_contract",
        "anchor_source": "signed_runtime_scientific_projection",
        "source_window_contract": {
            "start_hours": 0,
            "end_hours": 24,
            "grid_width_hours": 12,
            "aggregation": "max",
        },
        "trailing_na_policy": {
            "zero_imputation": False,
            "eligibility_uses_observed_window_count": True,
            "profile_summaries_ignore_missing": True,
        },
        "coordinate_scaling": authority.scaling_payload,
        "evidence_state_policy": authority.evidence_payload,
        "representation_columns": list(authority.representation_columns),
        "frozen_population_n": len(representation),
        "representation_sha256": _sha256(representation_path),
        "scientific_runtime_authority": {
            "schema_version": authority.schema_version,
            "protocol_content_sha256": authority.protocol_content_sha256,
            "execution_contract_sha256": authority.execution_contract_sha256,
        },
        "runtime_projection_sha256": "2" * 64,
    }
    schema_path = upstream / "trajectory_representation_schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    candidate_inputs = {
        "inputs": {
            "artifact:trajectory_representation": _binding(
                run_dir, representation_path, "signed-representation"
            ),
            "manifest:trajectory_representation_schema": _binding(
                run_dir, schema_path, "signed-representation-schema"
            ),
        }
    }
    candidate_out = run_dir / "candidate"

    candidate_summary = run_trajectory_scientific_candidate_selection(
        authority=authority,
        runtime_projection_sha256="2" * 64,
        out_dir=candidate_out,
        run_dir=run_dir,
        resolved_inputs=candidate_inputs,
    )

    assert candidate_summary["status"] == "ok", candidate_summary
    assert candidate_summary["n_clusters"] == 3
    assert (
        candidate_summary["output_files"]["table:trajectory_candidate_selection"]
        == "trajectory_candidate_selection.csv"
    )
    selection_table = pd.read_csv(candidate_out / "trajectory_candidate_selection.csv")
    assert list(selection_table.columns) == [
        "n_clusters",
        "bic",
        "aic",
        "final_log_likelihood",
        "parameter_count",
        "selected",
        "aic_minimum",
        "upper_boundary",
        "scientific_status",
        "reason_code",
        "reportable_result",
    ]
    assert np.isfinite(selection_table[["bic", "aic"]].to_numpy()).all()
    assert selection_table["aic_minimum"].sum() == 1
    assert candidate_summary["diagnostic_criteria"] == ["bic", "aic"]
    stability_inputs = {
        "inputs": {
            "artifact:trajectory_representation": _binding(
                run_dir, representation_path, "signed-representation"
            ),
            "manifest:trajectory_representation_schema": _binding(
                run_dir, schema_path, "signed-representation-schema"
            ),
            "artifact:candidate_cluster_assignments": _binding(
                run_dir,
                candidate_out / "candidate_cluster_assignments.csv",
                "signed-candidate-assignments",
            ),
            "manifest:cluster_selection": _binding(
                run_dir,
                candidate_out / "cluster_selection.json",
                "signed-cluster-selection",
            ),
            "manifest:candidate_cluster_solution_schema": _binding(
                run_dir,
                candidate_out / "candidate_cluster_solution_schema.json",
                "signed-candidate-schema",
            ),
        }
    }
    stability_out = run_dir / "stability"

    stability_summary = run_trajectory_stability(
        spec=authority.stability_spec,
        out_dir=stability_out,
        run_dir=run_dir,
        resolved_inputs=stability_inputs,
        scientific_runtime_authority=authority,
        runtime_projection_sha256="2" * 64,
        include_characterization=True,
    )

    assert stability_summary["status"] == "ok", stability_summary
    assert (
        stability_summary["coordinate_scaling_sha256"]
        == (candidate_summary["coordinate_scaling_sha256"])
    )
    profiles = pd.read_csv(stability_out / "trajectory_profiles.csv")
    sizes = pd.read_csv(stability_out / "cluster_sizes.csv")
    assert set(profiles.columns) == {
        "cluster",
        "source_column",
        "window_start_hours",
        "window_end_hours",
        "summary_statistic",
        "value",
        "n_observed",
    }
    assert len(profiles) == 3 * len(authority.representation_columns)
    assert sizes["n"].sum() == len(representation)

    rejected_out = run_dir / "rejected_candidate"
    rejected_out.mkdir()
    rejected_assignments = pd.DataFrame(
        {
            "stay_id": representation["stay_id"],
            "candidate_cluster": np.arange(len(representation)) % 4,
        }
    )
    rejected_assignments_path = rejected_out / "candidate_cluster_assignments.csv"
    rejected_assignments.to_csv(rejected_assignments_path, index=False)
    rejected_selection = json.loads(
        (candidate_out / "cluster_selection.json").read_text("utf-8")
    )
    upper_k = max(authority.candidate_cluster_counts)
    rejected_selection["selected_n_clusters"] = upper_k
    for candidate in rejected_selection["candidates"]:
        candidate["criterion_value"] = (
            0.0 if candidate["n_clusters"] == upper_k else 100.0
        )
    rejected_selection_path = rejected_out / "cluster_selection.json"
    rejected_selection_path.write_text(json.dumps(rejected_selection), encoding="utf-8")
    rejected_schema = json.loads(
        (candidate_out / "candidate_cluster_solution_schema.json").read_text("utf-8")
    )
    rejected_schema.update(
        {
            "selected_n_clusters": upper_k,
            "selected_model_id": f"signed-observed-data-diag-gmm-k{upper_k}",
            "selected_criterion_value": 0.0,
            "candidate_assignments_sha256": _sha256(rejected_assignments_path),
            "scientific_selection_status": "failed_closed",
            "stability_authorized": False,
            "scientific_selection_reason_code": authority.upper_boundary_reason_code,
        }
    )
    rejected_schema_path = rejected_out / "candidate_cluster_solution_schema.json"
    rejected_schema_path.write_text(json.dumps(rejected_schema), encoding="utf-8")
    rejected_inputs = json.loads(json.dumps(stability_inputs))
    rejected_inputs["inputs"]["artifact:candidate_cluster_assignments"] = _binding(
        run_dir, rejected_assignments_path, "rejected-candidate-assignments"
    )
    rejected_inputs["inputs"]["manifest:cluster_selection"] = _binding(
        run_dir, rejected_selection_path, "rejected-cluster-selection"
    )
    rejected_inputs["inputs"][
        "manifest:candidate_cluster_solution_schema"
    ] = _binding(run_dir, rejected_schema_path, "rejected-candidate-schema")
    rejected_stability_out = run_dir / "rejected_stability"
    rejected_summary = run_trajectory_stability(
        spec=authority.stability_spec,
        out_dir=rejected_stability_out,
        run_dir=run_dir,
        resolved_inputs=rejected_inputs,
        scientific_runtime_authority=authority,
        runtime_projection_sha256="2" * 64,
        include_characterization=True,
    )
    assert rejected_summary["status"] == "ok"
    assert rejected_summary["scientific_status"] == "failed_closed"
    assert rejected_summary["stability_refits_executed"] == 0
    assert pd.read_csv(rejected_stability_out / "cluster_assignments.csv").empty
    assert pd.read_csv(rejected_stability_out / "trajectory_profiles.csv").empty

    selection_path = candidate_out / "cluster_selection.json"
    selection = json.loads(selection_path.read_text("utf-8"))
    upper_selection = json.loads(json.dumps(selection))
    upper_k = max(authority.candidate_cluster_counts)
    upper_selection["selected_n_clusters"] = upper_k
    for candidate in upper_selection["candidates"]:
        candidate["criterion_value"] = (
            0.0 if candidate["n_clusters"] == upper_k else 100.0
        )
    with pytest.raises(
        TrajectoryScientificAuthorityError,
        match="candidate range does not contain an interior optimum",
    ):
        authority.validate_selection(upper_selection)

    selection["criterion"] = "aic"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    stability_inputs["inputs"]["manifest:cluster_selection"] = _binding(
        run_dir, selection_path, "tampered-cluster-selection"
    )

    tampered = run_trajectory_stability(
        spec=authority.stability_spec,
        out_dir=run_dir / "tampered_stability",
        run_dir=run_dir,
        resolved_inputs=stability_inputs,
        scientific_runtime_authority=authority,
        runtime_projection_sha256="2" * 64,
    )

    assert tampered["status"] == "failed_closed"
    assert tampered["reason_code"] == "TRAJECTORY_STABILITY_CONTRACT_INVALID"
    assert "signed authority" in " ".join(tampered["errors"])
