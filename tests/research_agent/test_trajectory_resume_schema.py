from __future__ import annotations

import json

import pandas as pd

from easyicu.research_agent.evidence import EvidenceStore
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    FixedWindowTrajectoryMetadata,
    ResearchContext,
    TemporalConstraint,
    VariableRole,
)
from easyicu.research_agent.trajectory_resume_schema import (
    materialize_legacy_trajectory_replay_schemas,
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
            cohort_name="test", database="test", n_patients=3, n_stays=3
        ),
        variables=variables,
        temporal_constraints=[
            TemporalConstraint(
                raw_text="from unit admission",
                relation="relative_to_anchor",
                anchor_event="unit_admission",
                executable_repr="relative_to_anchor|anchor=unit_admission",
            )
        ],
    )


def _plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Discover fixed-window phenotypes.",
        analysis_type="trajectory_clustering",
        steps=[
            AnalysisStep(
                step_id="representation",
                intent="Build a trajectory representation.",
                method="missingness_aware_trajectory_representation",
                inputs=["score_h0_6", "score_h6_12"],
                expected_outputs=[
                    "artifact:trajectory_representation",
                    "table:trajectory_membership",
                    "manifest:trajectory_representation_schema",
                ],
            ),
            AnalysisStep(
                step_id="candidate",
                intent="Compare candidate clustering solutions.",
                method="model_based_clustering",
                inputs=[
                    "artifact:trajectory_representation",
                    "manifest:trajectory_representation_schema",
                ],
                expected_outputs=[
                    "artifact:candidate_cluster_models",
                    "artifact:candidate_cluster_assignments",
                    "manifest:cluster_selection",
                    "manifest:candidate_cluster_solution_schema",
                ],
            ),
            AnalysisStep(
                step_id="stability",
                intent="Refit resamples and freeze the selected solution.",
                method="model_based_clustering_with_bootstrap_stability",
                inputs=[
                    "artifact:trajectory_representation",
                    "manifest:trajectory_representation_schema",
                    "artifact:candidate_cluster_models",
                    "artifact:candidate_cluster_assignments",
                    "manifest:cluster_selection",
                    "manifest:candidate_cluster_solution_schema",
                ],
                expected_outputs=[
                    "artifact:stability_freeze",
                    "manifest:trajectory_missingness_policy",
                    "table:cluster_assignments",
                    "table:cluster_stability",
                    "table:cluster_stability_assignments",
                ],
            ),
            AnalysisStep(
                step_id="characterize",
                intent="Describe the frozen clusters.",
                method="descriptive_cluster_characterization",
                inputs=["artifact:stability_freeze"],
                expected_outputs=[
                    "table:trajectory_profiles",
                    "table:cluster_sizes",
                ],
            ),
        ],
    )


def _legacy_records(tmp_path, *, conflicting_id: bool = False):
    evidence = EvidenceStore(tmp_path)
    source_dir = tmp_path / "legacy"
    source_dir.mkdir()
    representation_path = source_dir / "trajectory_representation.csv"
    pd.DataFrame(
        {
            "subject_key": [101, 102, 103],
            "observed__x0": [True, True, True],
            "observed__x1": [True, True, False],
            "z0": [0.1, 0.2, 0.3],
            "z1": [1.1, 1.2, None],
            "profile__score": [0.6, 0.7, 0.3],
        }
    ).to_csv(representation_path, index=False)
    representation_record = evidence.register_file(
        kind="table",
        description="Legacy trajectory representation.",
        source_path=representation_path,
        produced_by_step="representation",
    )
    membership_path = source_dir / "trajectory_membership.csv"
    pd.DataFrame(
        {
            "subject_key": [101, 102, 103, 104],
            "observed_window_count": [2, 2, 1, 0],
            "meets_min_observed_windows": [True, True, True, False],
            "included_in_clustering": [True, True, True, False],
        }
    ).to_csv(membership_path, index=False)
    membership_record = evidence.register_file(
        kind="table",
        description="Legacy trajectory membership.",
        source_path=membership_path,
        produced_by_step="representation",
    )
    representation_summary = {
        "status": "completed",
        "id_column": "other_key" if conflicting_id else None,
        "trajectory_source": {"source_id_column": "subject_key"},
        "scaled_representation_columns": ["z0", "z1"],
        "missingness_indicator_columns": ["observed__x0", "observed__x1"],
        "observation_family": "score",
        "ordered_observation_columns": ["score_h0_6", "score_h6_12"],
        "min_observed_windows": 1,
        "profile_columns": ["profile__score"],
        "profile_window_columns": {"score": ["z0", "z1"]},
        "profile_summary_statistic": "median",
        "time_axis": "relative_hours",
        "anchor": "unit_admission",
        "anchor_provenance": "agent-declared fixed-window alignment",
        "anchor_source": "unit_admission",
        "trailing_na_policy": {"zero_imputation": False},
        "missingness_policy": {
            "no_zero_imputation": True,
            "no_value_imputation": True,
        },
        "representation_row_n": 3,
        "output_files": ["trajectory_representation.csv"],
    }
    representation_summary_path = source_dir / "representation_step_summary.json"
    representation_summary_path.write_text(
        json.dumps(representation_summary), encoding="utf-8"
    )
    representation_summary_record = evidence.register_file(
        kind="statistic",
        description="Legacy representation step summary.",
        source_path=representation_summary_path,
        produced_by_step="representation",
    )

    models_path = source_dir / "candidate_cluster_models.json"
    models_path.write_text(
        json.dumps(
            {
                "model_family": "diagonal_gaussian_mixture",
                "representation_columns": ["z0", "z1"],
                "models": [
                    {
                        "n_clusters": 2,
                        "fit_status": "fitted",
                        "converged": True,
                        "model_family": "diagonal_gaussian_mixture",
                        "fit_method": "complete_data_em_gaussian_mixture",
                        "covariance_type": "diag",
                        "criterion_value": 10.0,
                    },
                    {
                        "n_clusters": 3,
                        "fit_status": "fitted",
                        "converged": True,
                        "model_family": "diagonal_gaussian_mixture",
                        "fit_method": "complete_data_em_gaussian_mixture",
                        "covariance_type": "diag",
                        "criterion_value": 12.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    assignments_path = source_dir / "candidate_cluster_assignments.csv"
    pd.DataFrame(
        {
            "subject_key": [101, 102, 103],
            "cluster_label_k2": [0, 1, 0],
            "max_posterior_k2": [0.9, 0.8, 0.7],
            "assignment_available_k2": [True, True, True],
            "cluster_label_k3": [0, 1, 2],
            "max_posterior_k3": [0.8, 0.7, 0.6],
            "assignment_available_k3": [True, True, True],
        }
    ).to_csv(assignments_path, index=False)
    selection_path = source_dir / "cluster_selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "criterion": "bic",
                "selection_rule": "minimum",
                "direction": "minimize",
                "selected_n_clusters": 2,
                "candidates": [
                    {"n_clusters": 2, "criterion_value": 10.0},
                    {"n_clusters": 3, "criterion_value": 12.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    candidate_evidence = [
        evidence.register_file(
            kind="log",
            description=path.stem,
            source_path=path,
            produced_by_step="candidate",
        )
        for path in (models_path, assignments_path, selection_path)
    ]
    candidate_summary = {
        "status": "completed",
        "clustering_method": "diagonal Gaussian mixture",
        "cluster_selection": json.loads(selection_path.read_text(encoding="utf-8")),
        "output_files": [path.name for path in (models_path, assignments_path, selection_path)],
    }
    candidate_summary_path = source_dir / "candidate_step_summary.json"
    candidate_summary_path.write_text(
        json.dumps(candidate_summary), encoding="utf-8"
    )
    candidate_summary_record = evidence.register_file(
        kind="statistic",
        description="Legacy candidate step summary.",
        source_path=candidate_summary_path,
        produced_by_step="candidate",
    )
    records = [
        {
            "step_id": "representation",
            "status": "ok",
            "evidence_ids": [
                representation_record.evidence_id,
                membership_record.evidence_id,
                representation_summary_record.evidence_id,
            ],
            "step_summary_evidence_id": representation_summary_record.evidence_id,
            "step_summary": representation_summary,
        },
        {
            "step_id": "candidate",
            "status": "ok",
            "evidence_ids": [
                *[record.evidence_id for record in candidate_evidence],
                candidate_summary_record.evidence_id,
            ],
            "step_summary_evidence_id": candidate_summary_record.evidence_id,
            "step_summary": candidate_summary,
        },
    ]
    return evidence, records


def test_resume_materializes_digest_bound_replay_schemas(tmp_path):
    evidence, records = _legacy_records(tmp_path)

    findings = materialize_legacy_trajectory_replay_schemas(
        plan=_plan(),
        context=_context(),
        run_dir=tmp_path,
        evidence=evidence,
        per_step_records=records,
        prompt_pack_version="test",
    )

    assert not any(finding.severity == "error" for finding in findings), [
        finding.detail for finding in findings
    ]
    representation_schema = json.loads(
        (tmp_path / "resume_migrations/representation/trajectory_representation_schema.json")
        .read_text(encoding="utf-8")
    )
    candidate_schema = json.loads(
        (tmp_path / "resume_migrations/candidate/candidate_cluster_solution_schema.json")
        .read_text(encoding="utf-8")
    )
    assert representation_schema["id_column"] == "subject_key"
    assert representation_schema["representation_columns"] == ["z0", "z1"]
    assert representation_schema["frozen_population_n"] == 3
    assert candidate_schema["selected_n_clusters"] == 2
    assert candidate_schema["schema_version"].endswith("/2")
    assert candidate_schema["assignment_column"] == "cluster_label_k2"
    assert candidate_schema["model_family"] == "diagonal_gaussian_mixture"
    assert candidate_schema["fit_method"] == "complete_data_em_gaussian_mixture"
    assert candidate_schema["selected_model_id"].endswith("::n_clusters_2")
    assert next(
        record for record in records if record["step_id"] == "representation"
    )["resume_schema_migrations"]
    assert next(record for record in records if record["step_id"] == "candidate")[
        "resume_schema_migrations"
    ]


def test_resume_schema_migration_fails_closed_on_conflicting_explicit_id(tmp_path):
    evidence, records = _legacy_records(tmp_path, conflicting_id=True)

    findings = materialize_legacy_trajectory_replay_schemas(
        plan=_plan(),
        context=_context(),
        run_dir=tmp_path,
        evidence=evidence,
        per_step_records=records,
        prompt_pack_version="test",
    )

    assert any(finding.severity == "error" for finding in findings)
    assert not (
        tmp_path / "resume_migrations/representation/trajectory_representation_schema.json"
    ).exists()


def test_resume_from_candidate_materializes_representation_schema_first(tmp_path):
    evidence, records = _legacy_records(tmp_path)

    findings = materialize_legacy_trajectory_replay_schemas(
        plan=_plan(),
        context=_context(),
        run_dir=tmp_path,
        evidence=evidence,
        per_step_records=records[:1],
        prompt_pack_version="test",
    )

    assert not any(finding.severity == "error" for finding in findings)
    assert (
        tmp_path
        / "resume_migrations/representation/trajectory_representation_schema.json"
    ).is_file()
    assert not (
        tmp_path
        / "resume_migrations/candidate/candidate_cluster_solution_schema.json"
    ).exists()


def test_resume_schema_rejects_tampered_summary_evidence(tmp_path):
    evidence, records = _legacy_records(tmp_path)
    summary_record = evidence.get(records[0]["step_summary_evidence_id"])
    assert summary_record is not None
    (tmp_path / summary_record.relative_path).write_text("{}", encoding="utf-8")

    findings = materialize_legacy_trajectory_replay_schemas(
        plan=_plan(),
        context=_context(),
        run_dir=tmp_path,
        evidence=evidence,
        per_step_records=records,
        prompt_pack_version="test",
    )

    assert any(finding.severity == "error" for finding in findings)
    assert "digest" in str(findings[0].detail.get("reason") or "")


def test_resume_schema_rejects_duplicate_current_schema_evidence(tmp_path):
    evidence, records = _legacy_records(tmp_path)
    first = materialize_legacy_trajectory_replay_schemas(
        plan=_plan(),
        context=_context(),
        run_dir=tmp_path,
        evidence=evidence,
        per_step_records=records,
        prompt_pack_version="test",
    )
    assert not any(finding.severity == "error" for finding in first)
    duplicate_source = tmp_path / "duplicate/trajectory_representation_schema.json"
    duplicate_source.parent.mkdir()
    duplicate_source.write_text('{"schema_version":"duplicate"}', encoding="utf-8")
    duplicate = evidence.register_file(
        kind="log",
        description="Conflicting replay schema.",
        source_path=duplicate_source,
        produced_by_step="representation",
    )
    representation_record = next(
        record for record in records if record["step_id"] == "representation"
    )
    representation_record["evidence_ids"].append(duplicate.evidence_id)

    findings = materialize_legacy_trajectory_replay_schemas(
        plan=_plan(),
        context=_context(),
        run_dir=tmp_path,
        evidence=evidence,
        per_step_records=records,
        prompt_pack_version="test",
    )

    assert any(finding.severity == "error" for finding in findings)
    assert "multiple current" in str(findings[0].detail.get("reason") or "")
