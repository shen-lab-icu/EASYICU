from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.contracts.runtime import RunResult
from easyicu.research_agent.schema import AnalysisPlan
from easyicu.research_agent.trajectory.plan_contract import (
    STABILITY_EXECUTOR_INPUTS,
    STABILITY_EXECUTOR_OUTPUTS,
)
from easyicu.research_agent.execution.runners.trajectory_stability_executor import (
    run_trajectory_stability,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_exact_resolved_inputs(
    resolved_inputs_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load exact host bindings and return truthful consumption receipts."""

    resolved = json.loads(resolved_inputs_path.read_text(encoding="utf-8"))
    raw_bindings = resolved.get("inputs")
    assert isinstance(raw_bindings, dict)
    loaded: dict[str, Any] = {}
    receipts: list[dict[str, Any]] = []
    bindings: dict[str, dict[str, Any]] = {}
    for input_key, raw_binding in sorted(raw_bindings.items()):
        assert isinstance(raw_binding, dict)
        binding = dict(raw_binding)
        path = Path(str(binding["absolute_path"]))
        assert path.is_file()
        assert _sha256(path) == binding["sha256"]
        receipt = {
            "input_key": input_key,
            "evidence_id": binding["evidence_id"],
            "sha256": binding["sha256"],
            "loaded": True,
        }
        if path.suffix.lower() == ".csv":
            value = pd.read_csv(path)
            receipt["row_count"] = int(len(value))
        elif path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            raise AssertionError(f"unsupported controlled fixture input: {path}")
        loaded[input_key] = value
        receipts.append(receipt)
        bindings[input_key] = binding
    return loaded, receipts, bindings


def _stability_spec() -> dict[str, Any]:
    return {
        "resampling_method": "subsample_without_replacement",
        "n_resamples": 2,
        "sample_fraction": 0.75,
        "sample_size": None,
        "sample_fraction_rounding": "floor",
        "base_seed": 31_415,
        "seed_derivation": "numpy_seedsequence_spawn_uint32_v1",
        "cross_resample_membership": "distinct_membership_required",
        "stability_metric": "adjusted_rand_index",
        "stability_aggregation": "mean",
        "metric_label_source": "raw_refit_labels_label_invariant",
        "evaluation_scope": "sampled_overlap",
        "label_alignment": "hungarian_maximum_overlap",
        "label_alignment_reference": "frozen_candidate_assignments",
        "label_alignment_tie_break": ("minimum_rank_distance_then_lexicographic_v1"),
        "final_assignment_policy": "copy_selected_candidate_labels",
        "minimum_successful_resamples": 2,
        "failed_refit_policy": "record_once_no_retry",
        "refit_engine": "easyicu_observed_data_diag_gmm_v1",
        "refit_initialization": "random_balanced_assignments",
        "refit_max_iter": 500,
        "refit_tolerance": 1e-5,
        "refit_regularization": 1e-6,
        "minimum_mean_stability": None,
        "decision_mode": "report_only",
        "threshold_failure_action": "fail_closed_require_planner_revision",
    }


def _trajectory_plan() -> dict[str, Any]:
    return {
        "research_question": "Assess fixed-window trajectory phenotypes.",
        "analysis_type": "trajectory_clustering",
        "steps": [
            {
                "step_id": "01_representation",
                "planned_analysis_role": "auxiliary",
                "intent": "Build the agent-selected trajectory representation.",
                "inputs": ["marker_h0_6", "marker_h6_12"],
                "expected_outputs": [
                    "artifact:trajectory_representation",
                    "table:trajectory_membership",
                    "table:feature_availability",
                    "table:feature_missingness_heatmap",
                    "manifest:trajectory_representation_schema",
                ],
                "method": "missingness_aware_trajectory_representation",
                "icu_rule_refs": [],
                "model_requirements": [],
                "trajectory_stability_spec": None,
            },
            {
                "step_id": "02_candidates",
                "planned_analysis_role": "primary",
                "intent": "Fit and select the agent-planned candidate solution.",
                "inputs": [
                    "artifact:trajectory_representation",
                    "manifest:trajectory_representation_schema",
                ],
                "expected_outputs": [
                    "artifact:candidate_cluster_models",
                    "artifact:candidate_cluster_assignments",
                    "table:embedding_plot",
                    "manifest:cluster_selection",
                    "manifest:candidate_cluster_solution_schema",
                ],
                "method": "latent_class_trajectory_clustering",
                "icu_rule_refs": [],
                "model_requirements": [],
                "trajectory_stability_spec": None,
            },
            {
                "step_id": "03_stability",
                "planned_analysis_role": "auxiliary",
                "intent": "Execute the planner-owned stability design.",
                "inputs": sorted(STABILITY_EXECUTOR_INPUTS),
                "expected_outputs": sorted(STABILITY_EXECUTOR_OUTPUTS),
                "method": "trajectory_cluster_stability",
                "icu_rule_refs": [],
                "model_requirements": [],
                "trajectory_stability_spec": _stability_spec(),
            },
            {
                "step_id": "04_characterization",
                "planned_analysis_role": "secondary",
                "intent": "Describe the frozen groups without causal claims.",
                "inputs": ["artifact:cluster_assignments"],
                "expected_outputs": [
                    "table:trajectory_profiles",
                    "table:phenotype_characteristics",
                    "table:cluster_sizes",
                    "table:descriptive_result",
                ],
                "method": "descriptive_cluster_characterization",
                "icu_rule_refs": [],
                "model_requirements": [],
                "trajectory_stability_spec": None,
            },
            {
                "step_id": "05_robustness",
                "planned_analysis_role": "sensitivity",
                "intent": (
                    "Re-evaluate the frozen trajectory solution in complete cases "
                    "without changing the primary clustering assignment."
                ),
                "inputs": ["artifact:cluster_assignments"],
                "expected_outputs": [
                    "table:robustness_matrix",
                    "statistic:robustness_summary",
                ],
                "method": "robustness_sensitivity",
                "icu_rule_refs": [],
                "model_requirements": [],
                "trajectory_stability_spec": None,
            },
        ],
        "robustness_specs": [
            {
                "spec_id": "complete_case_trajectory_inputs",
                "axis": "missing",
                "description": (
                    "Repeat the frozen characterization among stays with complete "
                    "values for the planner-selected trajectory inputs."
                ),
                # A complete-case spec must name the variables whose
                # completeness defines the set; the host will not infer them.
                # This fixture predates that requirement and was scripting a
                # plan no real Planner emits any more.
                "missing_override": {
                    "strategy": "complete_case",
                    "variables": ["sofa_max", "death"],
                },
            }
        ],
        "rationale": "Exercise the typed supporting stability calculator.",
    }


def _PlanAndCoderLLM():
    """Compose the reviewed prompt router instead of subclassing a mock."""

    from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient

    return PatternScriptedMockLLMClient(
        [
            ("ICU-AWARE RESEARCH PLAN", [json.dumps(_trajectory_plan())]),
            ("WRITE THE PYTHON CODE", ["value = 1\n"] * 8),
            ("REPAIR THE PYTHON CODE", ["value = 2\n"]),
            (
                "INTERPRET THE RESULTS",
                ["The registered supporting products were reviewed."] * 8,
            ),
            (
                "EVERY FINDING MUST INCLUDE",
                [json.dumps({"findings": []})] * 8,
            ),
        ],
        default="{}",
    )


def _captured_prompts(llm, marker: str) -> list[str]:
    folded = marker.casefold()
    return [
        next(
            (
                message.content
                for message in reversed(messages)
                if message.role == "user"
            ),
            "",
        )
        for messages, _kwargs in llm.calls
        if folded
        in "\n".join(str(message.content or "") for message in messages).casefold()
    ]


class _HybridTrajectoryRunner:
    """Use real typed stability computation between controlled fixture steps."""

    def __init__(self, *, workdir: Path) -> None:
        self.workdir = Path(workdir)
        self.calls: list[str] = []
        self.candidate_labels: dict[str, str] = {}
        self.downstream_resolved_assignments = False
        self.exact_binding_consumption: dict[str, set[str]] = {}

    @staticmethod
    def validate_runtime_capabilities() -> tuple[str, ...]:
        """Declare the immutable packages exercised by this controlled runner."""

        return ("numpy", "pandas", "scipy", "sklearn", "statsmodels")

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_representation(self, out_dir: Path) -> None:
        rng = np.random.default_rng(702)
        labels = np.repeat([0, 1], 60)
        centers = np.asarray([[-5.0, -2.5], [5.0, 2.5]])
        matrix = centers[labels] + rng.normal(0.0, 0.15, size=(len(labels), 2))
        identifiers = [f"opaque-{index:03d}" for index in range(len(labels))]
        representation = pd.DataFrame(
            {
                "opaque_id": identifiers,
                "coordinate_a": matrix[:, 0],
                "coordinate_b": matrix[:, 1],
            }
        )
        representation_path = out_dir / "trajectory_representation.csv"
        representation.to_csv(representation_path, index=False)
        membership_path = out_dir / "trajectory_membership.csv"
        membership = pd.DataFrame(
            {
                "opaque_id": identifiers,
                "observed_window_count": [2] * len(identifiers),
                "meets_min_observed_windows": [True] * len(identifiers),
                "included_in_clustering": [True] * len(identifiers),
                "exclusion_reason": [""] * len(identifiers),
            }
        )
        membership.to_csv(membership_path, index=False)
        pd.DataFrame(
            {
                "feature": ["marker_h0_6", "marker_h6_12"],
                "available_n": [len(membership), len(membership)],
            }
        ).to_csv(out_dir / "feature_availability.csv", index=False)
        pd.DataFrame(
            {
                "feature": ["marker_h0_6", "marker_h6_12"],
                "missing_n": [0, 0],
                "missing_fraction": [0.0, 0.0],
            }
        ).to_csv(out_dir / "feature_missingness_heatmap.csv", index=False)

        representation_sha = _sha256(representation_path)
        self._write_json(
            out_dir / "trajectory_representation_schema.json",
            {
                "schema_version": "easyicu.trajectory_representation_schema/2",
                "id_column": "opaque_id",
                "representation_columns": ["coordinate_a", "coordinate_b"],
                "frozen_population_n": len(representation),
                "observation_family": "opaque_signal_family",
                "observation_columns": ["marker_h0_6", "marker_h6_12"],
                "min_observed_windows": 2,
                "profile_columns": ["coordinate_a", "coordinate_b"],
                "profile_summary_statistic": "mean",
                "time_axis": "relative_hours",
                "anchor": "index_event",
                "anchor_provenance": "agent_declared",
                "anchor_source": "synthetic_contract_fixture",
                "membership_evidence_id": (
                    f"table_trajectory_membership_{_sha256(membership_path)[:8]}"
                ),
                "membership_sha256": _sha256(membership_path),
                "trailing_na_policy": {
                    "zero_imputation": False,
                    "eligibility_uses_observed_window_count": True,
                    "profile_summaries_ignore_missing": True,
                },
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
                "representation_sha256": representation_sha,
            },
        )
        self._write_json(
            out_dir / "step_summary.json",
            {
                "status": "ok",
                "analysis_family": "trajectory_clustering",
                "representation_row_n": len(representation),
                "output_files": {
                    "artifact:trajectory_representation": (
                        "trajectory_representation.csv"
                    ),
                    "table:trajectory_membership": "trajectory_membership.csv",
                    "table:feature_availability": "feature_availability.csv",
                    "table:feature_missingness_heatmap": (
                        "feature_missingness_heatmap.csv"
                    ),
                    "manifest:trajectory_representation_schema": (
                        "trajectory_representation_schema.json"
                    ),
                },
            },
        )

    def _write_candidates(
        self,
        out_dir: Path,
        *,
        resolved_inputs_path: Path,
    ) -> None:
        loaded, receipts, bindings = _load_exact_resolved_inputs(resolved_inputs_path)
        expected_keys = {
            "artifact:trajectory_representation",
            "manifest:trajectory_representation_schema",
        }
        assert set(loaded) == expected_keys
        representation = loaded["artifact:trajectory_representation"]
        representation_schema = loaded["manifest:trajectory_representation_schema"]
        assert isinstance(representation, pd.DataFrame)
        assert isinstance(representation_schema, dict)
        self.exact_binding_consumption["02_candidates"] = set(loaded)
        identifiers = representation["opaque_id"].tolist()
        labels = ["group::100"] * 60 + ["group::200"] * 60
        assignments = pd.DataFrame(
            {"opaque_id": identifiers, "chosen_partition": labels}
        ).sample(frac=1.0, random_state=19)
        assignments_path = out_dir / "candidate_cluster_assignments.csv"
        assignments.to_csv(assignments_path, index=False)
        self.candidate_labels = assignments.set_index("opaque_id")[
            "chosen_partition"
        ].to_dict()
        representation.merge(assignments, on="opaque_id", how="inner").to_csv(
            out_dir / "embedding_plot.csv", index=False
        )

        candidate_models_path = out_dir / "candidate_cluster_models.csv"
        pd.DataFrame(
            {
                "model_id": ["candidate-k2", "candidate-k3"],
                "n_clusters": [2, 3],
                "criterion_value": [100.0, 120.0],
            }
        ).to_csv(candidate_models_path, index=False)
        selection = {
            "criterion": "bic",
            "selection_rule": "minimum",
            "direction": "minimize",
            "selected_n_clusters": 2,
            "candidates": [
                {"n_clusters": 2, "criterion_value": 100.0},
                {"n_clusters": 3, "criterion_value": 120.0},
            ],
            "rationale": "The prespecified BIC rule selected k=2.",
        }
        cluster_selection_path = out_dir / "cluster_selection.json"
        self._write_json(cluster_selection_path, selection)

        assignment_sha = _sha256(assignments_path)
        self._write_json(
            out_dir / "candidate_cluster_solution_schema.json",
            {
                "schema_version": "easyicu.candidate_cluster_solution_schema/2",
                "id_column": "opaque_id",
                "representation_columns": ["coordinate_a", "coordinate_b"],
                "model_family": "latent_class_diagonal_gaussian_mixture",
                "fit_method": "observed_data_em_diagonal_gaussian_mixture",
                "covariance_type": "diag",
                "selected_n_clusters": 2,
                "selected_model_id": "candidate-k2",
                "assignment_column": "chosen_partition",
                "candidate_models_evidence_id": (
                    "table_candidate_cluster_models_"
                    f"{_sha256(candidate_models_path)[:8]}"
                ),
                "cluster_selection_evidence_id": (
                    f"log_cluster_selection_{_sha256(cluster_selection_path)[:8]}"
                ),
                "criterion": "bic",
                "selection_rule": "minimum",
                "direction": "minimize",
                "selected_criterion_value": 100.0,
                "representation_schema_sha256": (
                    bindings["manifest:trajectory_representation_schema"]["sha256"]
                ),
                "candidate_assignments_sha256": assignment_sha,
                "coordinate_scaling": representation_schema[
                    "coordinate_scaling"
                ],
            },
        )
        self._write_json(
            out_dir / "step_summary.json",
            {
                "status": "ok",
                "analysis_family": "trajectory_clustering",
                "n_clusters": 2,
                "clustering_method": ("latent_class_diagonal_gaussian_mixture"),
                "cluster_selection": selection,
                "input_bindings": receipts,
                "output_files": {
                    "artifact:candidate_cluster_models": (
                        "candidate_cluster_models.csv"
                    ),
                    "artifact:candidate_cluster_assignments": (
                        "candidate_cluster_assignments.csv"
                    ),
                    "table:embedding_plot": "embedding_plot.csv",
                    "manifest:cluster_selection": "cluster_selection.json",
                    "manifest:candidate_cluster_solution_schema": (
                        "candidate_cluster_solution_schema.json"
                    ),
                },
            },
        )

    def _write_characterization(
        self,
        out_dir: Path,
        *,
        resolved_inputs_path: Path,
    ) -> None:
        loaded, receipts, _bindings = _load_exact_resolved_inputs(resolved_inputs_path)
        assert set(loaded) == {"artifact:cluster_assignments"}
        assignments = loaded["artifact:cluster_assignments"]
        assert isinstance(assignments, pd.DataFrame)
        assert assignments.set_index("opaque_id")["cluster"].to_dict() == (
            self.candidate_labels
        )
        self.downstream_resolved_assignments = True
        self.exact_binding_consumption["04_characterization"] = set(loaded)

        sizes = assignments.groupby("cluster", as_index=False).size()
        sizes["proportion"] = sizes["size"] / len(assignments)
        sizes.to_csv(out_dir / "cluster_sizes.csv", index=False)
        sizes.rename(columns={"size": "n"}).to_csv(
            out_dir / "phenotype_characteristics.csv", index=False
        )
        sizes.rename(columns={"size": "n"}).to_csv(
            out_dir / "trajectory_profiles.csv", index=False
        )
        sizes.rename(columns={"size": "n"}).to_csv(
            out_dir / "descriptive_result.csv", index=False
        )
        self._write_json(
            out_dir / "step_summary.json",
            {
                "status": "ok",
                "analysis_family": "trajectory_clustering",
                "n_clusters": 2,
                "clustering_method": ("latent_class_diagonal_gaussian_mixture"),
                "input_bindings": receipts,
                "output_files": {
                    "table:trajectory_profiles": "trajectory_profiles.csv",
                    "table:phenotype_characteristics": (
                        "phenotype_characteristics.csv"
                    ),
                    "table:cluster_sizes": "cluster_sizes.csv",
                    "table:descriptive_result": "descriptive_result.csv",
                },
            },
        )

    def run(
        self,
        *,
        step_id: str,
        code: str,
        resolved_inputs_path: Path | None = None,
    ) -> RunResult:
        self.calls.append(step_id)
        cwd = self.workdir / "steps" / step_id
        out_dir = cwd / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        script_path = cwd / "analysis.py"
        script_path.write_text(code, encoding="utf-8")
        (cwd / "run.log").write_text("controlled runner\n", encoding="utf-8")

        if step_id == "01_representation":
            self._write_representation(out_dir)
        elif step_id == "02_candidates":
            assert resolved_inputs_path is not None
            self._write_candidates(
                out_dir,
                resolved_inputs_path=resolved_inputs_path,
            )
        elif step_id == "03_stability":
            assert "run_trajectory_stability" in code
            assert resolved_inputs_path is not None
            run_trajectory_stability(
                spec=_stability_spec(),
                out_dir=out_dir,
                run_dir=self.workdir,
                resolved_inputs=resolved_inputs_path,
            )
        elif step_id == "04_characterization":
            assert resolved_inputs_path is not None
            self._write_characterization(
                out_dir,
                resolved_inputs_path=resolved_inputs_path,
            )
        else:
            raise AssertionError(f"unexpected step: {step_id}")

        return RunResult(
            step_id=step_id,
            script_path=script_path,
            cwd=cwd,
            out_dir=out_dir,
            stdout="",
            stderr="",
            returncode=0,
            duration_seconds=0.01,
            artefacts=sorted(path for path in out_dir.iterdir() if path.is_file()),
            effective_isolation="controlled_test",
        )


def _disable_unrelated_audits(monkeypatch: pytest.MonkeyPatch) -> None:
    from easyicu.research_agent.execution import phase as pipeline_execute

    monkeypatch.setattr(pipeline_execute, "trajectory_bundle_findings", lambda **_: [])
    monkeypatch.setattr(
        pipeline_execute.RuntimeSupervisor,
        "critique_step",
        lambda _self, *, state, **_kwargs: state,
    )
    for validator in (
        pipeline_execute.StatisticalValidator,
        pipeline_execute.ClinicalConstraintValidator,
        pipeline_execute.StatisticalGuard,
        pipeline_execute.FigureContractQualityValidator,
        pipeline_execute.FigureSourceDataValidator,
        pipeline_execute.CrossStepCohortLockValidator,
        pipeline_execute.CrossStepRegisteredOutputValidator,
        pipeline_execute.CrossStepReconciliationTraceValidator,
        pipeline_execute.CrossStepSourceStatusValidator,
        pipeline_execute.StepSummaryFractionValidator,
        pipeline_execute.PrimaryModelContractValidator,
    ):
        monkeypatch.setattr(validator, "audit", lambda _self, **_kwargs: [])


def test_typed_trajectory_stability_success_is_evidence_bound_and_continues(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.execution import phase as pipeline_execute
    from easyicu.research_agent.authority.runtime_artifacts import (
        verified_run_evidence_path,
    )

    _disable_unrelated_audits(monkeypatch)
    llm = _PlanAndCoderLLM()
    runner_records: list[tuple[float, _HybridTrajectoryRunner]] = []
    runners_by_timeout: dict[float, _HybridTrajectoryRunner] = {}

    def runner_factory(*, workdir, timeout_seconds, **_kwargs):
        timeout = float(timeout_seconds)
        runner = runners_by_timeout.get(timeout)
        if runner is None:
            runner = _HybridTrajectoryRunner(workdir=Path(workdir))
            runners_by_timeout[timeout] = runner
            runner_records.append((timeout, runner))
        return runner

    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        timeout_seconds=17.0,
        standard_executor_timeout_seconds=1_234.0,
        runner_factory=runner_factory,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        enable_replanning=False,
        enable_deterministic_code_fallback=True,
        enable_deterministic_runner_repair=False,
        max_code_repair_attempts=2,
    )
    cohort = pd.DataFrame(
        {
            "stay_id": list(range(1, 25)),
            "marker_h0_6": np.linspace(-1.0, 1.0, 24),
            "marker_h6_12": np.linspace(-0.5, 1.5, 24),
            "death": [0, 1] * 12,
        }
    )
    result = pipeline.run(
        question="Assess fixed-window trajectory phenotypes.",
        cohort=cohort,
        cohort_name="trajectory_stability_success",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="04_characterization",
        stop_after_analysis=True,
    )

    run_dir = Path(result.workdir)
    partial = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    latest = {record["step_id"]: record for record in partial["per_step_records"]}
    stability_record = latest["03_stability"]
    downstream_record = latest["04_characterization"]
    ordinary_runners = [runner for timeout, runner in runner_records if timeout == 17.0]
    standard_runners = [
        runner for timeout, runner in runner_records if timeout == 1_234.0
    ]
    ordinary_calls = [
        step_id for runner in ordinary_runners for step_id in runner.calls
    ]
    standard_calls = [
        step_id for runner in standard_runners for step_id in runner.calls
    ]
    candidate_runner = next(
        runner for runner in ordinary_runners if runner.candidate_labels
    )

    assert stability_record["status"] == "ok"
    assert stability_record["deterministic_standard_analysis"] == (
        "trajectory_cluster_stability"
    )
    assert stability_record["generation_mode"] == "deterministic_standard"
    assert not [
        finding
        for finding in stability_record.get("contract_findings", [])
        if finding.get("severity") == "error"
    ]
    assert downstream_record["status"] == "ok"
    assert stability_record["execution_timeout_seconds"] == 1_234.0
    assert downstream_record["execution_timeout_seconds"] == 17.0
    assert standard_calls == ["03_stability"]
    assert "03_stability" not in ordinary_calls
    assert "04_characterization" in ordinary_calls
    assert any(runner.downstream_resolved_assignments for runner in ordinary_runners)
    assert any(
        runner.exact_binding_consumption.get("02_candidates")
        == {
            "artifact:trajectory_representation",
            "manifest:trajectory_representation_schema",
        }
        for runner in ordinary_runners
    )
    assert any(
        runner.exact_binding_consumption.get("04_characterization")
        == {"artifact:cluster_assignments"}
        for runner in ordinary_runners
    )
    standard_script = (run_dir / "steps" / "03_stability" / "analysis.py").read_text(
        encoding="utf-8"
    )
    for native_thread_env in (
        "VECLIB_MAXIMUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert f"os.environ['{native_thread_env}'] = '1'" in standard_script

    stability_coder_marker = "03_stability"
    assert all(
        stability_coder_marker not in prompt
        for prompt in _captured_prompts(llm, "WRITE THE PYTHON CODE")
    )
    assert _captured_prompts(llm, "REPAIR THE PYTHON CODE") == []

    evidence = EvidenceStore(run_dir)
    script_records = [
        record
        for record in evidence.records()
        if record.produced_by_step == "03_stability" and record.kind == "code"
    ]
    assert len(script_records) == 1
    script_record = script_records[0]
    assert script_record.producer == "standard_executor"
    assert script_record.generation_mode == "deterministic_standard"
    plan = AnalysisPlan.model_validate(
        json.loads((run_dir / "analysis_plan.json").read_text(encoding="utf-8"))
    )
    for product in sorted(STABILITY_EXECUTOR_OUTPUTS):
        ref, failure = pipeline_execute._resolve_typed_input_evidence(
            input_name=product,
            plan=plan,
            evidence_records=evidence.records(),
            per_step_records=partial["per_step_records"],
            run_dir=run_dir,
        )
        assert failure is None, (product, failure)
        assert ref is not None
        record = evidence.get(ref.evidence_id)
        assert record is not None
        assert verified_run_evidence_path(run_dir, record) is not None

    final_assignments = pd.read_csv(
        run_dir / "steps" / "03_stability" / "outputs" / "cluster_assignments.csv"
    )
    assert final_assignments.set_index("opaque_id")["cluster"].to_dict() == (
        candidate_runner.candidate_labels
    )

    summary_evidence_id = stability_record["step_summary_evidence_id"]
    stability_claims = [
        claim for claim in evidence.numeric_claims() if claim.step_id == "03_stability"
    ]
    assert stability_claims
    assert {claim.evidence_id for claim in stability_claims} == {summary_evidence_id}
    claim_fields = {claim.source_field for claim in stability_claims}
    assert {
        "selected_n_clusters",
        "n_successful_resamples",
        "mean_adjusted_rand_index",
    } <= claim_fields
