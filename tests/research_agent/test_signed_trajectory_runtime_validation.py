from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.contracts.capability_ids import (
    SIGNED_TRAJECTORY_PHENOTYPING_CAPABILITY_ID,
)
from easyicu.research_agent.planning.capability_registry import (
    assess_scientific_capability,
)
from easyicu.research_agent.reporting.readiness import _compute_readiness_gates
from easyicu.research_agent.schema import (
    AnalysisPlan,
    CohortDescriptor,
    ResearchContext,
    TrajectoryStabilitySpec,
)
from easyicu.research_agent.trajectory.runtime_validation import (
    signed_trajectory_runtime_bundle_errors,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan() -> AnalysisPlan:
    rule = "scientific_runtime_contract:" + "a" * 64
    stability = TrajectoryStabilitySpec(
        n_resamples=100,
        sample_fraction=0.8,
        minimum_successful_resamples=100,
    ).model_dump(mode="json")
    return AnalysisPlan.model_validate(
        {
            "research_question": "Assess fixed-window trajectory phenotypes.",
            "analysis_type": "trajectory_clustering",
            "steps": [
                {
                    "step_id": "01_representation",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Build the signed representation.",
                    "inputs": [],
                    "expected_outputs": [
                        "artifact:trajectory_representation",
                        "manifest:trajectory_representation_schema",
                    ],
                    "method": "signed_fixed_window_trajectory_representation",
                    "icu_rule_refs": [rule],
                },
                {
                    "step_id": "02_candidates",
                    "planned_analysis_role": "primary",
                    "intent": "Select from the signed candidate grid.",
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
                    "icu_rule_refs": [rule],
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
                    "expected_outputs": ["artifact:stability_freeze"],
                    "method": "trajectory_cluster_stability_characterization",
                    "icu_rule_refs": [rule],
                    "trajectory_stability_spec": stability,
                },
                {
                    "step_id": "04_figure",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Render the signed selection diagnostic.",
                    "inputs": [
                        "table:trajectory_candidate_selection",
                        "table:feature_availability",
                    ],
                    "expected_outputs": ["figure:trajectory_selection_diagnostics"],
                    "method": "signed_trajectory_selection_diagnostic_figure",
                    "icu_rule_refs": [rule],
                },
            ],
        }
    )


def _records(tmp_path: Path, plan: AnalysisPlan) -> list[dict]:
    protocol = "b" * 64
    runtime = "c" * 64
    authority = {
        "schema_version": "easyicu.trajectory_scientific_runtime_authority/1",
        "protocol_content_sha256": protocol,
        "execution_contract_sha256": "a" * 64,
        "runtime_projection_sha256": runtime,
    }
    rep_dir = tmp_path / "steps" / plan.steps[0].step_id / "outputs"
    candidate_dir = tmp_path / "steps" / plan.steps[1].step_id / "outputs"
    rep_dir.mkdir(parents=True)
    candidate_dir.mkdir(parents=True)
    rep_schema = rep_dir / "trajectory_representation_schema.json"
    rep_schema.write_text(json.dumps({"schema_version": "test"}), encoding="utf-8")
    selection = {
        "criterion": "bic",
        "direction": "minimize",
        "selection_rule": "minimum",
        "selected_n_clusters": 4,
        "candidates": [
            {"n_clusters": 2, "criterion_value": 300.0},
            {"n_clusters": 3, "criterion_value": 200.0},
            {"n_clusters": 4, "criterion_value": 100.0},
        ],
    }
    selection_path = candidate_dir / "cluster_selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    candidate_schema = {
        "stability_authorized": False,
        "scientific_selection_reason_code": "NO_INTERIOR_OPTIMUM",
    }
    candidate_schema_path = candidate_dir / "candidate_solution_schema.json"
    candidate_schema_path.write_text(
        json.dumps(candidate_schema), encoding="utf-8"
    )
    rep_summary = {
        "status": "ok",
        "eligible_n": 120,
        "observation_family": ["resp", "lact"],
        "representation_columns": [
            "resp__h0_12",
            "resp__h12_24",
            "lact__h0_12",
            "lact__h12_24",
        ],
        "runtime_projection_sha256": runtime,
        "scientific_runtime_authority": {
            key: value for key, value in authority.items() if key != "runtime_projection_sha256"
        },
        "output_files": {
            "manifest:trajectory_representation_schema": rep_schema.name
        },
    }
    candidate_summary = {
        "status": "ok",
        "n_clusters": 4,
        "cluster_selection": selection,
        "scientific_status": "failed_closed",
        "stability_authorized": False,
        "reason_code": "NO_INTERIOR_OPTIMUM",
        "reportable_result": "no_interior_solution_in_prespecified_candidate_range",
        "candidate_solution_schema_sha256": _sha256(candidate_schema_path),
        "scientific_runtime_authority": authority,
        "input_bindings": [
            {
                "input_key": "manifest:trajectory_representation_schema",
                "sha256": _sha256(rep_schema),
            }
        ],
        "output_files": {
            "manifest:cluster_selection": selection_path.name,
            "manifest:candidate_cluster_solution_schema": candidate_schema_path.name,
        },
    }
    stability_summary = {
        "status": "ok",
        "scientific_status": "failed_closed",
        "reason_code": "NO_INTERIOR_OPTIMUM",
        "freeze_status": "not_frozen_candidate_selection_failed_closed",
        "stability_refits_executed": 0,
        "reportable_result": "no_stable_phenotype_solution",
        "outcome_binding_received_by_executor": False,
        "outcome_bindings_received": [],
        "scientific_runtime_authority": authority,
        "input_bindings": [
            {
                "input_key": "manifest:candidate_cluster_solution_schema",
                "sha256": _sha256(candidate_schema_path),
            },
            {
                "input_key": "manifest:cluster_selection",
                "sha256": _sha256(selection_path),
            },
        ],
    }
    figure_summary = {
        "status": "ok",
        "scientific_status": "failed_closed",
        "reason_code": "NO_INTERIOR_OPTIMUM",
    }
    summaries = (rep_summary, candidate_summary, stability_summary, figure_summary)
    kinds = (
        "trajectory_signed_representation",
        "trajectory_signed_candidate_selection",
        "trajectory_cluster_stability",
        "trajectory_selection_diagnostic_figure",
    )
    return [
        {
            "step_id": step.step_id,
            "status": "ok",
            "deterministic_standard_analysis": kind,
            "step_summary": summary,
        }
        for step, kind, summary in zip(plan.steps, kinds, summaries, strict=True)
    ]


def test_signed_boundary_optimum_is_a_validated_non_solution(tmp_path: Path) -> None:
    plan = _plan()
    records = _records(tmp_path, plan)
    assert signed_trajectory_runtime_bundle_errors(
        plan=plan, records=records, run_dir=tmp_path
    ) == []
    context = ResearchContext(
        research_question=plan.research_question,
        variables=[],
        cohort=CohortDescriptor(
            cohort_name="trajectory", database="synthetic", n_patients=120, n_stays=120
        ),
    )
    assessment = assess_scientific_capability(
        analysis_type=plan.analysis_type,
        context=context,
        plan=plan,
    )
    assert assessment.capability_id == SIGNED_TRAJECTORY_PHENOTYPING_CAPABILITY_ID
    assert assessment.claim_ceiling == "reportable"
    gates = _compute_readiness_gates(
        context=context,
        plan=plan,
        per_step_records=records,
        findings=[],
        evidence=EvidenceStore(tmp_path),
        run_dir=tmp_path,
        manuscript_path=tmp_path / "manuscript.md",
        stop_after_analysis=True,
    )
    assert gates["execution_complete"] is True
    assert gates["analysis_validated"] is True
    assert gates["paper_authorized"] is False


def test_signed_trajectory_non_solution_tampering_fails_closed(tmp_path: Path) -> None:
    plan = _plan()
    records = _records(tmp_path, plan)
    tampered = deepcopy(records)
    tampered[2]["step_summary"]["outcome_binding_received_by_executor"] = True
    errors = signed_trajectory_runtime_bundle_errors(
        plan=plan, records=tampered, run_dir=tmp_path
    )
    assert errors == ["signed trajectory failed-closed decision is incoherent"]
