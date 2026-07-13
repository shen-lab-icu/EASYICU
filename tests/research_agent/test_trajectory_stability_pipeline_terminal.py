from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from easyicu.research_agent.evidence import EvidenceStore
from easyicu.research_agent.runner import RunResult
from easyicu.research_agent.schema import ValidationFinding
from easyicu.research_agent.trajectory_plan_contract import (
    STABILITY_EXECUTOR_INPUTS,
    STABILITY_EXECUTOR_OUTPUTS,
)

_TRIVIAL_CODE = "value = 1\n"


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
        "refit_max_iter": 100,
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
                "intent": "Build the agent-selected trajectory representation.",
                "inputs": ["marker_h0_6", "marker_h6_12"],
                "expected_outputs": [
                    "artifact:trajectory_representation",
                    "table:trajectory_membership",
                    "manifest:trajectory_representation_schema",
                ],
                "method": "missingness_aware_trajectory_representation",
                "icu_rule_refs": [],
                "model_requirements": [],
                "trajectory_stability_spec": None,
            },
            {
                "step_id": "02_candidates",
                "intent": "Fit and select the agent-planned candidate solution.",
                "inputs": [
                    "artifact:trajectory_representation",
                    "manifest:trajectory_representation_schema",
                ],
                "expected_outputs": [
                    "artifact:candidate_cluster_models",
                    "artifact:candidate_cluster_assignments",
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
                "intent": "Describe the frozen groups without causal claims.",
                "inputs": ["artifact:cluster_assignments"],
                "expected_outputs": [
                    "table:trajectory_profiles",
                    "table:cluster_sizes",
                ],
                "method": "descriptive_cluster_characterization",
                "icu_rule_refs": [],
                "model_requirements": [],
                "trajectory_stability_spec": None,
            },
        ],
        "rationale": "Exercise the typed supporting stability calculator.",
    }


class _TerminalPlanLLM:
    name = "trajectory-terminal-sentinel-test"

    def __init__(self) -> None:
        self.repair_calls = 0
        self.write_calls = 0

    def complete(self, messages, *, max_tokens=2048, temperature=0.2):
        del max_tokens, temperature
        user = next(
            (
                message.content
                for message in reversed(messages)
                if message.role == "user"
            ),
            "",
        )
        upper = user.upper()
        if "ICU-AWARE RESEARCH PLAN" in upper:
            return json.dumps(_trajectory_plan())
        if "REPAIR THE PYTHON CODE" in upper:
            self.repair_calls += 1
            return _TRIVIAL_CODE
        if "WRITE THE PYTHON CODE" in upper:
            self.write_calls += 1
            return _TRIVIAL_CODE
        if "INTERPRET THE RESULTS" in upper:
            return "The registered supporting products were reviewed."
        return "{}"


class _TerminalRunner:
    def __init__(
        self,
        *,
        workdir: Path,
        stability_mode: str,
        timeout_seconds: float,
    ) -> None:
        self.workdir = Path(workdir)
        self.stability_mode = stability_mode
        self.timeout_seconds = float(timeout_seconds)
        self.calls: list[str] = []

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_upstream_outputs(self, step_id: str, out_dir: Path) -> None:
        if step_id == "01_representation":
            pd.DataFrame(
                {
                    "opaque_id": ["a", "b", "c", "d"],
                    "coordinate_a": [0.0, 0.1, 2.0, 2.1],
                    "coordinate_b": [0.2, 0.0, 2.2, 2.0],
                }
            ).to_csv(out_dir / "trajectory_representation.csv", index=False)
            pd.DataFrame(
                {
                    "opaque_id": ["a", "b", "c", "d"],
                    "observed_window_count": [2, 2, 2, 2],
                    "meets_min_observed_windows": [True] * 4,
                    "included_in_clustering": [True] * 4,
                    "exclusion_reason": [""] * 4,
                }
            ).to_csv(out_dir / "trajectory_membership.csv", index=False)
            self._write_json(
                out_dir / "trajectory_representation_schema.json",
                {"schema_version": "test-representation-schema"},
            )
            output_files = {
                "artifact:trajectory_representation": "trajectory_representation.csv",
                "table:trajectory_membership": "trajectory_membership.csv",
                "manifest:trajectory_representation_schema": (
                    "trajectory_representation_schema.json"
                ),
            }
        elif step_id == "02_candidates":
            pd.DataFrame({"model_id": ["candidate-2"]}).to_csv(
                out_dir / "candidate_cluster_models.csv", index=False
            )
            pd.DataFrame(
                {
                    "opaque_id": ["a", "b", "c", "d"],
                    "chosen_partition": [0, 0, 1, 1],
                }
            ).to_csv(out_dir / "candidate_cluster_assignments.csv", index=False)
            self._write_json(
                out_dir / "cluster_selection.json",
                {"selected_n_clusters": 2},
            )
            self._write_json(
                out_dir / "candidate_cluster_solution_schema.json",
                {"schema_version": "test-candidate-schema"},
            )
            output_files = {
                "artifact:candidate_cluster_models": "candidate_cluster_models.csv",
                "artifact:candidate_cluster_assignments": (
                    "candidate_cluster_assignments.csv"
                ),
                "manifest:cluster_selection": "cluster_selection.json",
                "manifest:candidate_cluster_solution_schema": (
                    "candidate_cluster_solution_schema.json"
                ),
            }
        else:
            raise AssertionError(f"unexpected upstream step {step_id}")
        self._write_json(
            out_dir / "step_summary.json",
            {"status": "ok", "output_files": output_files},
        )

    def _write_terminal_outputs(self, out_dir: Path) -> None:
        if self.stability_mode == "empty":
            return
        pd.DataFrame(
            {
                "resample_id": ["failed-resample"],
                "adjusted_rand_index": [0.125],
            }
        ).to_csv(out_dir / "cluster_stability.csv", index=False)
        self._write_json(
            out_dir / "cluster_stability_refit_attempts.json",
            {
                "planned_n_resamples": 2,
                "attempted_n_resamples": 1,
                "successful_n_resamples": 0,
            },
        )
        self._write_json(
            out_dir / "step_summary.json",
            {
                "status": (
                    "ok"
                    if self.stability_mode == "ok_contract_error"
                    else "failed_closed"
                ),
                "mean_adjusted_rand_index": 0.125,
                "n_successful_resamples": 0,
                "errors": ["synthetic standard-executor failure"],
            },
        )

    def run(
        self,
        *,
        step_id: str,
        code: str,
        resolved_inputs_path: Path | None = None,
    ) -> RunResult:
        del resolved_inputs_path
        self.calls.append(step_id)
        cwd = self.workdir / "steps" / step_id
        out_dir = cwd / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        script_path = cwd / "analysis.py"
        script_path.write_text(code, encoding="utf-8")
        (cwd / "run.log").write_text("synthetic runner\n", encoding="utf-8")
        if (
            step_id == "01_representation"
            and self.stability_mode == "unsafe_ordinary_timeout"
        ):
            (out_dir / "untrusted_partial.csv").write_text(
                "value\n1\n", encoding="utf-8"
            )
            return RunResult(
                step_id=step_id,
                script_path=script_path,
                cwd=cwd,
                out_dir=out_dir,
                stdout="",
                stderr="synthetic unconfirmed teardown",
                returncode=-1,
                duration_seconds=self.timeout_seconds,
                artefacts=[],
                timed_out=True,
                effective_isolation="synthetic_test",
                outputs_safe_to_collect=False,
            )
        if step_id == "03_stability" and self.stability_mode in {
            "runtime_timeout",
            "unsafe_runtime_timeout",
        }:
            # Simulate interruption after the standard executor has streamed
            # assignments but before its atomic publish/cleanup boundary.
            pd.DataFrame(
                {"opaque_id": ["a"], "resample_id": [0], "cluster": [1]}
            ).to_csv(
                out_dir / ".cluster_stability_assignments.pending.csv",
                index=False,
            )
            return RunResult(
                step_id=step_id,
                script_path=script_path,
                cwd=cwd,
                out_dir=out_dir,
                stdout="",
                stderr="synthetic timeout",
                returncode=-1,
                duration_seconds=self.timeout_seconds,
                artefacts=[],
                timed_out=True,
                effective_isolation="synthetic_test",
                outputs_safe_to_collect=(
                    self.stability_mode != "unsafe_runtime_timeout"
                ),
            )
        if step_id == "03_stability":
            self._write_terminal_outputs(out_dir)
        else:
            self._write_upstream_outputs(step_id, out_dir)
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
            effective_isolation="synthetic_test",
        )


def _disable_unrelated_step_audits(monkeypatch: pytest.MonkeyPatch) -> None:
    from easyicu.research_agent import pipeline_execute

    monkeypatch.setattr(pipeline_execute, "_step_contract_findings", lambda **_: [])
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


def _run_terminal_case(
    *,
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stability_mode: str,
):
    _disable_unrelated_step_audits(monkeypatch)
    if stability_mode == "preexecution_concept_error":
        from easyicu.research_agent import pipeline_execute

        original_audit = pipeline_execute.ConceptUsageAuditor.audit

        def concept_audit(self, *, context, script_text, step):
            if step.step_id == "03_stability":
                return [
                    ValidationFinding(
                        validator="concept_usage",
                        severity="error",
                        message="synthetic trusted-adapter concept failure",
                    )
                ]
            return original_audit(
                self,
                context=context,
                script_text=script_text,
                step=step,
            )

        monkeypatch.setattr(
            pipeline_execute.ConceptUsageAuditor,
            "audit",
            concept_audit,
        )
    if stability_mode == "ok_contract_error":
        from easyicu.research_agent import pipeline_execute

        monkeypatch.setattr(
            pipeline_execute,
            "_step_contract_findings",
            lambda **kwargs: (
                [
                    ValidationFinding(
                        validator="step_contract",
                        severity="error",
                        message="synthetic standard-output contract failure",
                    )
                ]
                if kwargs["step"].step_id == "03_stability"
                else []
            ),
        )
    llm = _TerminalPlanLLM()
    runner_holder: dict[str, _TerminalRunner] = {}

    def runner_factory(*, workdir, timeout_seconds, **_kwargs):
        runner = _TerminalRunner(
            workdir=Path(workdir),
            stability_mode=stability_mode,
            timeout_seconds=timeout_seconds,
        )
        runner_holder["runner"] = runner
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
        enable_deterministic_runner_repair=True,
        max_code_repair_attempts=2,
    )
    cohort = pd.DataFrame(
        {
            "stay_id": list(range(1, 13)),
            "marker_h0_6": [value / 10 for value in range(12)],
            "marker_h6_12": [value / 11 for value in range(12)],
            "death": [0, 1] * 6,
        }
    )
    result = pipeline.run(
        question="Assess fixed-window trajectory phenotypes.",
        cohort=cohort,
        cohort_name="trajectory_terminal_sentinel",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="03_stability",
        stop_after_analysis=True,
    )
    return result, llm, runner_holder["runner"]


@pytest.mark.parametrize(
    ("stability_mode", "expected_reason", "expected_runner_calls"),
    [
        ("failed_closed", "executor_reported_failed_closed", 1),
        ("empty", "missing_executor_outputs", 1),
        ("ok_contract_error", "executor_output_contract_failed", 1),
        ("runtime_timeout", "executor_runtime_failure", 1),
        ("unsafe_runtime_timeout", "executor_runtime_failure", 1),
        ("preexecution_concept_error", "preexecution_concept_gate_failed", 0),
    ],
)
def test_trajectory_stability_terminal_failures_never_enter_repair_or_fallback(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stability_mode: str,
    expected_reason: str,
    expected_runner_calls: int,
) -> None:
    result, llm, runner = _run_terminal_case(
        ra=ra,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        stability_mode=stability_mode,
    )
    run_dir = Path(result.workdir)
    partial = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    stability_record = next(
        record
        for record in reversed(partial["per_step_records"])
        if record.get("step_id") == "03_stability"
    )

    assert stability_record["status"] == "deterministic_standard_blocked"
    assert stability_record["diagnostic_only"] is True
    assert stability_record["standard_executor_terminal_reason"] == expected_reason
    assert llm.repair_calls == 0
    assert runner.calls.count("03_stability") == expected_runner_calls
    if expected_runner_calls:
        assert runner.timeout_seconds == 1_234.0
        assert stability_record["execution_timeout_seconds"] == 1_234.0
    if stability_mode == "unsafe_runtime_timeout":
        assert stability_record["outputs_safe_to_collect"] is False
    assert "04_characterization" not in runner.calls

    pending_path = (
        run_dir
        / "steps"
        / "03_stability"
        / "outputs"
        / ".cluster_stability_assignments.pending.csv"
    )
    if stability_mode == "unsafe_runtime_timeout":
        assert pending_path.exists()
    else:
        assert not pending_path.exists()

    evidence = EvidenceStore(run_dir)
    assert not any(
        record.relative_path.endswith(
            ".cluster_stability_assignments.pending.csv"
        )
        for record in evidence.records()
    )
    assert [
        claim for claim in evidence.numeric_claims() if claim.step_id == "03_stability"
    ] == []

    if stability_mode == "failed_closed":
        output_names = {
            "step_summary.json",
            "cluster_stability.csv",
            "cluster_stability_refit_attempts.json",
        }
        output_records = [
            record
            for record in evidence.records()
            if record.produced_by_step == "03_stability"
            and any(record.relative_path.endswith(name) for name in output_names)
        ]
        assert len(output_records) == len(output_names)
        assert all(
            record.metadata.get("diagnostic_only") is True for record in output_records
        )


def test_standard_executor_pending_file_is_never_registered_during_cleanup_race(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent import pipeline_execute

    # Model a writer that recreates its private stream after unlink.  Evidence
    # safety must come from an explicit deny-list at enumeration/registration,
    # not from winning a filesystem timing race.
    monkeypatch.setattr(
        pipeline_execute,
        "_remove_standard_executor_pending_artifacts",
        lambda _out_dir: None,
    )
    result, _llm, _runner = _run_terminal_case(
        ra=ra,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        stability_mode="runtime_timeout",
    )
    run_dir = Path(result.workdir)
    pending_path = (
        run_dir
        / "steps"
        / "03_stability"
        / "outputs"
        / ".cluster_stability_assignments.pending.csv"
    )

    assert pending_path.exists()
    evidence = EvidenceStore(run_dir)
    assert not any(
        record.relative_path.endswith(pending_path.name)
        for record in evidence.records()
    )


def test_unsafe_runner_mount_short_circuits_before_any_output_probe(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent import pipeline_execute

    real_figure_probe = pipeline_execute._has_figure_exports

    def forbidden_output_probe(out_dir: Path) -> bool:
        if out_dir.parent.name == "03_stability":
            raise AssertionError("unsafe runner output mount must not be inspected")
        return real_figure_probe(out_dir)

    monkeypatch.setattr(
        pipeline_execute,
        "_has_figure_exports",
        forbidden_output_probe,
    )
    result, llm, runner = _run_terminal_case(
        ra=ra,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        stability_mode="unsafe_runtime_timeout",
    )
    run_dir = Path(result.workdir)
    partial = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    stability_record = next(
        record
        for record in reversed(partial["per_step_records"])
        if record.get("step_id") == "03_stability"
    )

    assert stability_record["status"] == "deterministic_standard_blocked"
    assert stability_record["outputs_safe_to_collect"] is False
    assert stability_record["runner_output_safety_reason"] == (
        "runner_output_teardown_unconfirmed"
    )
    assert llm.repair_calls == 0
    assert runner.calls.count("03_stability") == 1
    assert [
        claim
        for claim in EvidenceStore(run_dir).numeric_claims()
        if claim.step_id == "03_stability"
    ] == []


def test_ordinary_unsafe_runner_mount_never_enters_repair_or_evidence(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent import pipeline_execute

    def forbidden_output_probe(_out_dir: Path) -> bool:
        raise AssertionError("unsafe runner output mount must not be inspected")

    monkeypatch.setattr(
        pipeline_execute,
        "_has_figure_exports",
        forbidden_output_probe,
    )
    result, llm, runner = _run_terminal_case(
        ra=ra,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        stability_mode="unsafe_ordinary_timeout",
    )
    run_dir = Path(result.workdir)
    partial = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    failed_record = next(
        record
        for record in reversed(partial["per_step_records"])
        if record.get("step_id") == "01_representation"
    )

    assert failed_record["status"] == "execution_failed"
    assert failed_record["outputs_safe_to_collect"] is False
    assert failed_record["runner_output_safety_reason"] == (
        "runner_output_teardown_unconfirmed"
    )
    assert llm.repair_calls == 0
    assert runner.calls == ["01_representation"]
    assert not any(
        record.relative_path.endswith("untrusted_partial.csv")
        for record in EvidenceStore(run_dir).records()
    )
