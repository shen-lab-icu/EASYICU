"""Planner ownership gates for the standard trajectory-stability executor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent.evidence import EvidenceStore
from easyicu.research_agent.pipeline import (
    _migrate_legacy_resume_trajectory_stability_spec,
)
from easyicu.research_agent.pipeline_execute import _plan_signature
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    TrajectoryStabilitySpec,
)
from easyicu.research_agent.trajectory_plan_contract import (
    STABILITY_EXECUTOR_INPUTS,
    STABILITY_EXECUTOR_OUTPUTS,
)
from easyicu.research_agent.trajectory_stability_executor import (
    trajectory_stability_executor_owns_step,
)


def _stability_spec(*, base_seed: int = 1729) -> TrajectoryStabilitySpec:
    return TrajectoryStabilitySpec(
        resampling_method="subsample_without_replacement",
        n_resamples=3,
        sample_fraction=0.8,
        sample_fraction_rounding="floor",
        base_seed=base_seed,
        seed_derivation="numpy_seedsequence_spawn_uint32_v1",
        cross_resample_membership="distinct_membership_required",
        stability_metric="adjusted_rand_index",
        stability_aggregation="mean",
        metric_label_source="raw_refit_labels_label_invariant",
        evaluation_scope="sampled_overlap",
        label_alignment="hungarian_maximum_overlap",
        label_alignment_reference="frozen_candidate_assignments",
        label_alignment_tie_break=(
            "minimum_rank_distance_then_lexicographic_v1"
        ),
        final_assignment_policy="copy_selected_candidate_labels",
        minimum_successful_resamples=3,
        failed_refit_policy="record_once_no_retry",
        refit_engine="easyicu_observed_data_diag_gmm_v1",
        refit_initialization="random_balanced_assignments",
        refit_max_iter=60,
        refit_tolerance=1e-4,
        refit_regularization=1e-6,
        decision_mode="report_only",
        threshold_failure_action="fail_closed_require_planner_revision",
    )


def _candidate_step() -> AnalysisStep:
    return AnalysisStep(
        step_id="candidate",
        intent="Compare the agent-planned candidate clustering solutions.",
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
    )


def _stability_step(
    *,
    spec: TrajectoryStabilitySpec | None = None,
    exact_contract: bool = False,
    extra_outputs: tuple[str, ...] = (),
) -> AnalysisStep:
    inputs = (
        sorted(STABILITY_EXECUTOR_INPUTS)
        if exact_contract
        else [
            "artifact:trajectory_representation",
            "artifact:candidate_cluster_models",
            "artifact:candidate_cluster_assignments",
            "manifest:cluster_selection",
            "manifest:trajectory_representation_schema",
            "manifest:candidate_cluster_solution_schema",
        ]
    )
    outputs = (
        sorted(STABILITY_EXECUTOR_OUTPUTS)
        if exact_contract
        else [
            "artifact:stability_freeze",
            "artifact:cluster_assignments",
            "manifest:trajectory_missingness_policy",
            "table:cluster_assignments",
            "table:cluster_stability",
            "table:cluster_stability_assignments",
            "table:cluster_assignment_provenance",
        ]
    )
    return AnalysisStep(
        step_id="stability",
        intent="Assess stability and freeze only the selected candidate solution.",
        method=(
            "trajectory_cluster_stability"
            if spec is not None
            else "model_based_clustering_with_bootstrap_stability"
        ),
        inputs=inputs,
        expected_outputs=[*outputs, *extra_outputs],
        trajectory_stability_spec=spec,
    )


def _plan(
    *,
    spec: TrajectoryStabilitySpec | None = None,
    exact_contract: bool = False,
    extra_stability_outputs: tuple[str, ...] = (),
) -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Discover and validate longitudinal ICU phenotypes.",
        analysis_type="trajectory_clustering",
        revision=1,
        steps=[
            _candidate_step(),
            _stability_step(
                spec=spec,
                exact_contract=exact_contract,
                extra_outputs=extra_stability_outputs,
            ),
        ],
    )


def _evidence_and_resume_state(
    run_dir: Path,
    *,
    supported_schema: bool = True,
    latest_candidate_status: str = "ok",
    stability_status: str | None = None,
    tamper_candidate_schema: bool = False,
) -> tuple[EvidenceStore, dict]:
    source_dir = run_dir / "source"
    source_dir.mkdir(parents=True)
    evidence = EvidenceStore(run_dir)

    representation_path = source_dir / "trajectory_representation.csv"
    representation_path.write_text(
        "opaque_id,coordinate_a,coordinate_b\n"
        "a,0.0,1.0\nb,1.0,0.0\nc,2.0,3.0\n"
        "d,3.0,2.0\ne,4.0,5.0\nf,5.0,4.0\n",
        encoding="utf-8",
    )
    representation = evidence.register_file(
        kind="table",
        description="Digest-bound trajectory representation.",
        source_path=representation_path,
        produced_by_step="representation",
    )
    membership_path = source_dir / "trajectory_membership.csv"
    membership_path.write_text(
        "opaque_id,included_in_clustering\n"
        "a,true\nb,true\nc,true\nd,true\ne,true\nf,true\n",
        encoding="utf-8",
    )
    membership = evidence.register_file(
        kind="table",
        description="Digest-bound trajectory membership.",
        source_path=membership_path,
        produced_by_step="representation",
    )

    representation_schema_path = source_dir / "trajectory_representation_schema.json"
    representation_schema_path.write_text(
        json.dumps(
            {
                "schema_version": "easyicu.trajectory_representation_schema/1",
                "id_column": "opaque_id",
                "representation_columns": ["coordinate_a", "coordinate_b"],
                "frozen_population_n": 6,
                "observation_family": "opaque_signal_family",
                "observation_columns": ["marker_h0_6", "marker_h6_12"],
                "min_observed_windows": 2,
                "profile_columns": ["coordinate_a", "coordinate_b"],
                "profile_summary_statistic": "mean",
                "time_axis": "relative_hours",
                "anchor": "index_event",
                "anchor_provenance": "agent_declared",
                "anchor_source": "synthetic_contract_fixture",
                "trailing_na_policy": {
                    "zero_imputation": False,
                    "eligibility_uses_observed_window_count": True,
                    "profile_summaries_ignore_missing": True,
                },
                "membership_evidence_id": membership.evidence_id,
                "membership_sha256": membership.sha256,
                "representation_evidence_id": representation.evidence_id,
                "representation_sha256": representation.sha256,
            }
        ),
        encoding="utf-8",
    )
    representation_schema = evidence.register_file(
        kind="log",
        description="Digest-bound trajectory representation schema.",
        source_path=representation_schema_path,
        produced_by_step="representation",
    )

    candidate_assignments_path = source_dir / "candidate_cluster_assignments.csv"
    candidate_assignments_path.write_text(
        "opaque_id,chosen_partition\n"
        "a,0\nb,0\nc,1\nd,1\ne,2\nf,2\n",
        encoding="utf-8",
    )
    candidate_assignments = evidence.register_file(
        kind="table",
        description="Digest-bound candidate cluster assignments.",
        source_path=candidate_assignments_path,
        produced_by_step="candidate",
    )
    candidate_models_path = source_dir / "candidate_cluster_models.json"
    candidate_models_path.write_text("{}", encoding="utf-8")
    candidate_models = evidence.register_file(
        kind="log",
        description="Digest-bound candidate models.",
        source_path=candidate_models_path,
        produced_by_step="candidate",
    )
    cluster_selection_path = source_dir / "cluster_selection.json"
    cluster_selection_path.write_text("{}", encoding="utf-8")
    cluster_selection = evidence.register_file(
        kind="log",
        description="Digest-bound cluster selection.",
        source_path=cluster_selection_path,
        produced_by_step="candidate",
    )

    candidate_schema_path = source_dir / "candidate_cluster_solution_schema.json"
    candidate_schema_path.write_text(
        json.dumps(
            {
                "schema_version": "easyicu.candidate_cluster_solution_schema/2",
                "representation_schema_evidence_id": (
                    representation_schema.evidence_id
                ),
                "id_column": "opaque_id",
                "representation_columns": ["coordinate_a", "coordinate_b"],
                "model_family": (
                    "latent_class_diagonal_gaussian_mixture"
                    if supported_schema
                    else "kmeans"
                ),
                "fit_method": (
                    "observed_data_em_diagonal_gaussian_mixture"
                    if supported_schema
                    else "lloyd_kmeans"
                ),
                "covariance_type": "diag" if supported_schema else "none",
                "selected_n_clusters": 3,
                "selected_model_id": "opaque-model-k3",
                "assignment_column": "chosen_partition",
                "candidate_models_evidence_id": candidate_models.evidence_id,
                "candidate_assignments_evidence_id": (
                    candidate_assignments.evidence_id
                ),
                "cluster_selection_evidence_id": cluster_selection.evidence_id,
                "criterion": "bic",
                "selection_rule": "minimum",
                "direction": "minimize",
                "selected_criterion_value": 100.0,
            }
        ),
        encoding="utf-8",
    )
    candidate_schema = evidence.register_file(
        kind="log",
        description="Digest-bound candidate cluster solution schema.",
        source_path=candidate_schema_path,
        produced_by_step="candidate",
    )
    if tamper_candidate_schema:
        (run_dir / candidate_schema.relative_path).write_text(
            json.dumps({"tampered": True}), encoding="utf-8"
        )

    candidate_ok_record = {
        "step_id": "candidate",
        "status": "ok",
        "evidence_ids": [
            candidate_schema.evidence_id,
            candidate_assignments.evidence_id,
            candidate_models.evidence_id,
            cluster_selection.evidence_id,
        ],
    }
    records = [candidate_ok_record]
    if latest_candidate_status != "ok":
        records.append(
            {
                "step_id": "candidate",
                "status": latest_candidate_status,
                "evidence_ids": [candidate_schema.evidence_id],
            }
        )
    if stability_status is not None:
        records.append({"step_id": "stability", "status": stability_status})
    return evidence, {"per_step_records": records}


def _packet_json() -> str:
    return json.dumps(
        {
            "steps": [
                {
                    "step_id": "stability",
                    "method": "trajectory_cluster_stability",
                    "inputs": sorted(STABILITY_EXECUTOR_INPUTS),
                    "expected_outputs": sorted(STABILITY_EXECUTOR_OUTPUTS),
                    "trajectory_stability_spec": _stability_spec().model_dump(
                        mode="json"
                    ),
                }
            ]
        }
    )


def test_legacy_resume_adds_planner_spec_only_from_verified_supported_candidate(
    tmp_path: Path,
) -> None:
    plan = _plan()
    evidence, resume_state = _evidence_and_resume_state(tmp_path)
    calls = []

    class PacketLLM:
        name = "trajectory-stability-packet-llm"

        def complete(self, messages, *, max_tokens=4096, temperature=0.1):
            calls.append((messages, max_tokens, temperature))
            return _packet_json()

    revised, revision_path, target_ids = (
        _migrate_legacy_resume_trajectory_stability_spec(
            plan=plan,
            run_dir=tmp_path,
            resume_state=resume_state,
            resume_from_step_id=None,
            role_resolver=lambda role: PacketLLM() if role == "planner" else None,
            evidence=evidence,
            prompt_version="test-pack",
            llm_signature="test-planner",
        )
    )

    assert len(calls) == 1
    assert target_ids == ("stability",)
    assert revision_path == tmp_path / "analysis_plan_revision_2.json"
    assert revision_path.exists()
    stability = revised.steps[1]
    assert stability.inputs == sorted(STABILITY_EXECUTOR_INPUTS)
    assert stability.expected_outputs == sorted(STABILITY_EXECUTOR_OUTPUTS)
    assert stability.trajectory_stability_spec == _stability_spec()
    assert trajectory_stability_executor_owns_step(stability, plan=revised)
    record = evidence.get("analysis_plan_revision_2")
    assert record is not None
    assert record.metadata["reason"] == "legacy_missing_trajectory_stability_spec"
    assert record.metadata["retrospective_development_migration"] is True


@pytest.mark.parametrize(
    ("latest_candidate_status", "stability_status", "tamper_candidate_schema"),
    [
        ("contract_failed", None, False),
        ("ok", "ok", False),
        ("ok", None, True),
    ],
)
def test_legacy_resume_does_not_add_spec_without_current_verified_open_target(
    tmp_path: Path,
    latest_candidate_status: str,
    stability_status: str | None,
    tamper_candidate_schema: bool,
) -> None:
    plan = _plan()
    evidence, resume_state = _evidence_and_resume_state(
        tmp_path,
        latest_candidate_status=latest_candidate_status,
        stability_status=stability_status,
        tamper_candidate_schema=tamper_candidate_schema,
    )

    unchanged, revision_path, target_ids = (
        _migrate_legacy_resume_trajectory_stability_spec(
            plan=plan,
            run_dir=tmp_path,
            resume_state=resume_state,
            resume_from_step_id=None,
            role_resolver=lambda _role: (_ for _ in ()).throw(
                AssertionError("an ineligible legacy checkpoint must not call the LLM")
            ),
            evidence=evidence,
            prompt_version="test-pack",
            llm_signature="test-planner",
        )
    )

    assert unchanged is plan
    assert revision_path is None
    assert target_ids == ()
    assert unchanged.steps[1].trajectory_stability_spec is None


def test_resume_cut_at_candidate_does_not_call_stability_migration_llm(
    tmp_path: Path,
) -> None:
    plan = _plan()
    evidence, resume_state = _evidence_and_resume_state(tmp_path)

    unchanged, revision_path, target_ids = (
        _migrate_legacy_resume_trajectory_stability_spec(
            plan=plan,
            run_dir=tmp_path,
            resume_state=resume_state,
            resume_from_step_id="candidate",
            role_resolver=lambda _role: (_ for _ in ()).throw(
                AssertionError("the cut candidate is no longer completed")
            ),
            evidence=evidence,
            prompt_version="test-pack",
            llm_signature="test-planner",
        )
    )

    assert unchanged is plan
    assert revision_path is None
    assert target_ids == ()


def test_unsupported_candidate_schema_preserves_agent_coder_path(
    tmp_path: Path,
) -> None:
    plan = _plan(exact_contract=True)
    evidence, resume_state = _evidence_and_resume_state(
        tmp_path, supported_schema=False
    )

    unchanged, revision_path, target_ids = (
        _migrate_legacy_resume_trajectory_stability_spec(
            plan=plan,
            run_dir=tmp_path,
            resume_state=resume_state,
            resume_from_step_id=None,
            role_resolver=lambda _role: (_ for _ in ()).throw(
                AssertionError("unsupported fits must not be coerced into the runner")
            ),
            evidence=evidence,
            prompt_version="test-pack",
            llm_signature="test-planner",
        )
    )

    stability = unchanged.steps[1]
    assert unchanged is plan
    assert revision_path is None
    assert target_ids == ()
    assert stability.trajectory_stability_spec is None
    assert not trajectory_stability_executor_owns_step(stability, plan=unchanged)


def test_standard_stability_route_requires_closed_independent_owner() -> None:
    valid = _plan(spec=_stability_spec(), exact_contract=True)
    assert trajectory_stability_executor_owns_step(valid.steps[1], plan=valid)

    no_spec = _plan(exact_contract=True)
    assert not trajectory_stability_executor_owns_step(
        no_spec.steps[1], plan=no_spec
    )

    extra_outcome = _plan(
        spec=_stability_spec(),
        exact_contract=True,
        extra_stability_outputs=("table:outcome_by_cluster",),
    )
    assert not trajectory_stability_executor_owns_step(
        extra_outcome.steps[1], plan=extra_outcome
    )

    wrong_method = _plan(spec=_stability_spec(), exact_contract=True)
    wrong_method.steps[1].method = "cluster_robust_regression"
    assert not trajectory_stability_executor_owns_step(
        wrong_method.steps[1], plan=wrong_method
    )

    monolithic_step = AnalysisStep(
        step_id="candidate_and_stability",
        intent="Select candidates and assess stability in one scientific step.",
        method="model_based_clustering_with_bootstrap_stability",
        inputs=sorted(STABILITY_EXECUTOR_INPUTS),
        expected_outputs=sorted(
            STABILITY_EXECUTOR_OUTPUTS
            | {
                "artifact:candidate_cluster_models",
                "artifact:candidate_cluster_assignments",
                "manifest:cluster_selection",
                "manifest:candidate_cluster_solution_schema",
            }
        ),
        trajectory_stability_spec=_stability_spec(),
    )
    monolithic = AnalysisPlan(
        research_question="Discover and validate longitudinal ICU phenotypes.",
        analysis_type="trajectory_clustering",
        steps=[monolithic_step],
    )
    assert not trajectory_stability_executor_owns_step(
        monolithic_step, plan=monolithic
    )


def test_plan_signature_changes_when_stability_spec_changes() -> None:
    first = _plan(spec=_stability_spec(base_seed=1729), exact_contract=True)
    same = _plan(spec=_stability_spec(base_seed=1729), exact_contract=True)
    changed = _plan(spec=_stability_spec(base_seed=1730), exact_contract=True)

    assert _plan_signature(first) == _plan_signature(same)
    assert _plan_signature(first) != _plan_signature(changed)
