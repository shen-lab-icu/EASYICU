from __future__ import annotations

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.pipeline import _migrate_resume_trajectory_products
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    FixedWindowTrajectoryMetadata,
    ResearchContext,
    VariableRole,
)
from easyicu.research_agent.trajectory_plan_contract import (
    evaluate_trajectory_plan_dag,
)


def _context() -> ResearchContext:
    variables = [
        ConceptDescriptor(
            name=f"score_h{start}_{end}",
            role=VariableRole.ORDINAL_SCORE,
            dtype="float64",
            is_ordinal=True,
            fixed_window_trajectory=FixedWindowTrajectoryMetadata(
                family="score",
                window_start_hours=float(start),
                window_end_hours=float(end),
                window_width_hours=float(end - start),
                source_scale="ordinal",
                representation_kind="fractional_window_summary",
                observed_fractional_values=True,
            ),
        )
        for start, end in ((0, 6), (6, 12))
    ]
    return ResearchContext(
        research_question="Discover fixed-window phenotypes.",
        cohort=CohortDescriptor(
            cohort_name="test", database="test", n_patients=10, n_stays=10
        ),
        variables=variables,
    )


def _legacy_plan() -> AnalysisPlan:
    windows = ["score_h0_6", "score_h6_12"]
    return AnalysisPlan(
        research_question="Discover fixed-window phenotypes.",
        analysis_type="trajectory_clustering",
        revision=2,
        steps=[
            AnalysisStep(
                step_id="representation",
                intent="Build a trajectory representation.",
                method="missingness_aware_trajectory_representation",
                inputs=windows,
                expected_outputs=["artifact:trajectory_representation"],
            ),
            AnalysisStep(
                step_id="cluster",
                intent="Select and freeze a stable clustering solution.",
                method="latent_class_clustering_with_bootstrap_stability",
                inputs=["artifact:trajectory_representation"],
                expected_outputs=[
                    "artifact:candidate_cluster_fits",
                    "manifest:cluster_selection",
                    "artifact:cluster_assignments",
                    "table:cluster_stability",
                ],
            ),
            AnalysisStep(
                step_id="characterize",
                intent="Describe the frozen clusters.",
                method="descriptive_cluster_characterization",
                inputs=["artifact:cluster_assignments"],
                expected_outputs=["table:cluster_characteristics"],
            ),
        ],
    )


def _split_plan_with_stability_role_drift() -> AnalysisPlan:
    windows = ["score_h0_6", "score_h6_12"]
    return AnalysisPlan(
        research_question="Discover fixed-window phenotypes.",
        analysis_type="trajectory_clustering",
        revision=4,
        steps=[
            AnalysisStep(
                step_id="representation",
                intent="Build the frozen trajectory representation.",
                method="missingness_aware_trajectory_representation",
                inputs=windows,
                expected_outputs=[
                    "artifact:trajectory_representation",
                    "table:trajectory_membership",
                ],
            ),
            AnalysisStep(
                step_id="candidate_selection",
                intent="Compare candidate clustering solutions.",
                method="model_based_clustering",
                inputs=["artifact:trajectory_representation"],
                expected_outputs=[
                    "artifact:candidate_cluster_models",
                    "artifact:candidate_cluster_assignments",
                    "manifest:cluster_selection",
                ],
            ),
            AnalysisStep(
                step_id="stability",
                intent="Refit resamples and freeze the selected solution.",
                method="model_based_clustering_with_bootstrap_stability",
                inputs=[
                    "artifact:trajectory_representation",
                    "artifact:candidate_cluster_models",
                    "artifact:candidate_cluster_assignments",
                ],
                expected_outputs=[
                    "artifact:stability_freeze",
                    "artifact:cluster_assignments",
                    "table:cluster_number_selection",
                    "table:cluster_stability",
                    "table:cluster_sizes",
                    "manifest:trajectory_missingness_policy",
                    "table:cluster_assignments",
                    "table:cluster_stability_assignments",
                ],
            ),
            AnalysisStep(
                step_id="characterize",
                intent="Describe the frozen clusters.",
                method="descriptive_cluster_characterization",
                inputs=[
                    "artifact:cluster_assignments",
                    "artifact:stability_freeze",
                    "table:cluster_sizes",
                ],
                expected_outputs=[
                    "table:trajectory_profiles",
                    "table:cluster_sizes",
                ],
            ),
        ],
    )


def test_resume_writes_schema_only_trajectory_revision(tmp_path):
    evidence = EvidenceStore(tmp_path)

    migrated, path, findings = _migrate_resume_trajectory_products(
        plan=_legacy_plan(),
        context=_context(),
        run_dir=tmp_path,
        evidence=evidence,
        prompt_version="test",
        llm_signature="mock",
    )

    assert path == tmp_path / "analysis_plan_revision_3.json"
    assert path.is_file()
    assert migrated.revision == 3
    assert [step.step_id for step in migrated.steps] == [
        "representation",
        "cluster",
        "characterize",
    ]
    representation = migrated.steps[0]
    cluster = migrated.steps[1]
    characterize = migrated.steps[2]
    assert "table:trajectory_membership" in representation.expected_outputs
    assert (
        "manifest:trajectory_representation_schema"
        in representation.expected_outputs
    )
    assert "manifest:trajectory_missingness_policy" in cluster.expected_outputs
    assert "manifest:candidate_cluster_solution_schema" in cluster.expected_outputs
    assert "table:cluster_stability_assignments" in cluster.expected_outputs
    assert "table:trajectory_profiles" in characterize.expected_outputs
    assert evidence.get("analysis_plan_revision_3") is not None
    assert any(
        finding.detail.get("kind") == "trajectory_canonical_products_added"
        for finding in findings
    )


def test_current_trajectory_schema_is_not_revised_again(tmp_path):
    evidence = EvidenceStore(tmp_path)
    first, _, _ = _migrate_resume_trajectory_products(
        plan=_legacy_plan(),
        context=_context(),
        run_dir=tmp_path,
        evidence=evidence,
        prompt_version="test",
        llm_signature="mock",
    )

    unchanged, path, findings = _migrate_resume_trajectory_products(
        plan=first,
        context=_context(),
        run_dir=tmp_path,
        evidence=evidence,
        prompt_version="test",
        llm_signature="mock",
    )

    assert unchanged == first
    assert path is None
    assert findings == []


def test_resume_migrates_redundant_split_role_outputs_before_execution(tmp_path):
    evidence = EvidenceStore(tmp_path)

    migrated, path, findings = _migrate_resume_trajectory_products(
        plan=_split_plan_with_stability_role_drift(),
        context=_context(),
        run_dir=tmp_path,
        evidence=evidence,
        prompt_version="test",
        llm_signature="mock",
    )

    assert path == tmp_path / "analysis_plan_revision_5.json"
    stability = next(step for step in migrated.steps if step.step_id == "stability")
    characterize = next(
        step for step in migrated.steps if step.step_id == "characterize"
    )
    assert "table:cluster_number_selection" not in stability.expected_outputs
    assert "table:cluster_sizes" not in stability.expected_outputs
    assert "manifest:cluster_selection" in stability.inputs
    assert "manifest:trajectory_representation_schema" in stability.inputs
    assert "manifest:candidate_cluster_solution_schema" in stability.inputs
    candidate = next(
        step for step in migrated.steps if step.step_id == "candidate_selection"
    )
    assert "manifest:trajectory_representation_schema" in candidate.inputs
    assert "table:cluster_sizes" not in characterize.inputs
    assert evaluate_trajectory_plan_dag(
        plan=migrated,
        context=_context(),
    ).findings == ()
    assert any(
        finding.detail.get("kind")
        == "trajectory_redundant_split_role_outputs_removed"
        for finding in findings
    )
