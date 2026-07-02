"""Resume-from-partial-run + manifest streaming (T2.4).

These tests pin two contracts:

1. **Manifest streaming.** The pipeline must flush
   ``manifest_partial.json`` after every step so a crash mid-loop
   leaves a usable resume sentinel.
2. **Resume from partial.** Running the pipeline a second time with
   ``resume_run_id=<the previous run_id>`` must:

   - reuse the same ``run_dir``,
   - skip steps whose prior status is ``"ok"``,
   - re-execute steps that are missing from the partial manifest,
   - end with a final ``manifest.json`` that the rest of the
     pipeline (manuscript, latex, report) treats normally.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tools.run_research_agent_bench import (
    _resolve_resume_run_id,
    _run_ehrflowbench_jsonl,
)
from easyicu.research_agent.pipeline import (
    _load_compatible_resume_plan,
    _load_resume_state,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _run_full(ra, synthetic_cohort, workdir: Path):
    pipeline = ra.ResearchAgentPipeline(workdir=workdir, llm=ra.MockLLMClient())
    return pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="resume_test",
        database="synthetic",
        target_outcome="death",
    )


def _write_bench_resume_checkpoint(run_dir: Path, *, complete: bool = False) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis_plan.json").write_text(
        json.dumps({"steps": []}), encoding="utf-8"
    )
    (run_dir / "manifest_partial.json").write_text(
        json.dumps({"per_step_records": []}), encoding="utf-8"
    )
    if complete:
        (run_dir / "run_status.json").write_text(
            json.dumps({"gates": {"execution_complete": True}}),
            encoding="utf-8",
        )


def test_bench_runner_explicit_resume_id_wins_over_auto_discovery(tmp_path: Path):
    selected = tmp_path / "run_20260701T000000_selected"
    auto_latest = tmp_path / "run_20260701T999999_auto"
    _write_bench_resume_checkpoint(selected)
    _write_bench_resume_checkpoint(auto_latest)

    assert (
        _resolve_resume_run_id(
            workdir=tmp_path,
            reuse_existing=True,
            resume_run_id=selected.name,
        )
        == selected.name
    )


def test_bench_runner_auto_resume_ignores_complete_runs(tmp_path: Path):
    interrupted = tmp_path / "run_20260701T010000_interrupted"
    complete_latest = tmp_path / "run_20260701T999999_complete"
    _write_bench_resume_checkpoint(interrupted)
    _write_bench_resume_checkpoint(complete_latest, complete=True)

    assert (
        _resolve_resume_run_id(
            workdir=tmp_path,
            reuse_existing=True,
            resume_run_id=None,
        )
        == interrupted.name
    )


def test_bench_runner_explicit_resume_requires_locked_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run_20260701T000000_missing"
    run_dir.mkdir()

    with pytest.raises(SystemExit, match="analysis_plan.json"):
        _resolve_resume_run_id(
            workdir=tmp_path,
            reuse_existing=False,
            resume_run_id=run_dir.name,
        )


def test_load_resume_state_rejects_corrupt_partial_manifest(tmp_path: Path):
    run_dir = tmp_path / "run_corrupt"
    run_dir.mkdir()
    (run_dir / "manifest_partial.json").write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt checkpoint"):
        _load_resume_state(run_dir)


def test_bench_runner_resume_id_rejects_paths(tmp_path: Path):
    with pytest.raises(SystemExit, match="not a path"):
        _resolve_resume_run_id(
            workdir=tmp_path,
            reuse_existing=False,
            resume_run_id="../run_20260701T000000_bad",
        )


def test_bench_runner_ehrflow_resume_requires_single_row(tmp_path: Path):
    jsonl_path = tmp_path / "items.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"key": "E1"}),
                json.dumps({"key": "E2"}),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="one-row EHRFlowBench JSONL"):
        _run_ehrflowbench_jsonl(
            jsonl_path=jsonl_path,
            out_root=tmp_path / "out",
            seed=7,
            arms=["naive"],
            resume_run_id="run_20260701T000000_selected",
        )


def test_resume_prefers_latest_compatible_plan_revision(tmp_path: Path):
    run_dir = tmp_path / "run_20260701T000000_revision"
    run_dir.mkdir()
    original = AnalysisPlan(
        research_question="Resume E1.",
        steps=[
            AnalysisStep(
                step_id="01_cohort",
                intent="Define cohort.",
                expected_outputs=["table:cohort"],
            ),
            AnalysisStep(
                step_id="02_table",
                intent="Completed table.",
                expected_outputs=["table:table"],
            ),
            AnalysisStep(
                step_id="05_sensitivity_figure",
                intent="Render the publication figure(s) declared by step '05_sensitivity'.",
                expected_outputs=["figure:sensitivity"],
            ),
        ],
    )
    revision = AnalysisPlan(
        research_question="Resume E1.",
        revision=2,
        steps=[
            AnalysisStep(
                step_id="00_probe",
                intent="Probe.",
                expected_outputs=["table:probe"],
            ),
            AnalysisStep(
                step_id="01_cohort",
                intent="Define cohort.",
                expected_outputs=["table:cohort"],
            ),
            AnalysisStep(
                step_id="02_table",
                intent="Completed table.",
                expected_outputs=["table:table"],
            ),
            AnalysisStep(
                step_id="05_sensitivity",
                intent="Run sensitivity.",
                expected_outputs=["table:sensitivity"],
            ),
            AnalysisStep(
                step_id="05_sensitivity_figure",
                intent="Render the publication figure(s) declared by step '05_sensitivity'.",
                expected_outputs=["figure:sensitivity"],
            ),
        ],
    )
    (run_dir / "analysis_plan.json").write_text(
        original.model_dump_json(indent=2), encoding="utf-8"
    )
    (run_dir / "analysis_plan_revision_2.json").write_text(
        revision.model_dump_json(indent=2), encoding="utf-8"
    )
    resume_state = {
        "plan_path": "analysis_plan.json",
        "per_step_records": [
            {"step_id": "00_probe", "status": "ok"},
            {"step_id": "01_cohort", "status": "ok"},
            {"step_id": "02_table", "status": "ok"},
        ],
    }

    plan, path = _load_compatible_resume_plan(
        run_dir=run_dir,
        resume_state=resume_state,
    )

    assert path == run_dir / "analysis_plan_revision_2.json"
    assert [step.step_id for step in plan.steps][-2:] == [
        "05_sensitivity",
        "05_sensitivity_figure",
    ]


def test_partial_manifest_is_written_after_run(ra, synthetic_cohort, tmp_path: Path):
    result = _run_full(ra, synthetic_cohort, tmp_path)
    run_dir = Path(result.workdir)
    partial = run_dir / "manifest_partial.json"
    assert partial.exists(), "manifest_partial.json must be written during the run"

    data = json.loads(partial.read_text(encoding="utf-8"))
    assert data["run_id"] == result.run_id
    assert data["schema_version"].startswith("easyicu.research_manifest_partial")
    # Every step in per_step_records should have status ok after a clean run.
    statuses = [r.get("status") for r in data.get("per_step_records", [])]
    assert statuses, "no step records persisted in partial manifest"
    assert all(s == "ok" for s in statuses), statuses


def test_partial_manifest_checkpoints_executed_step_before_interpretation(
    ra, synthetic_cohort, tmp_path: Path
):
    class InterruptingAnalyzerLLM(ra.MockLLMClient):
        name = "interrupting-analyzer"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            if "INTERPRET THE RESULTS" in user.upper():
                raise KeyboardInterrupt("simulate interruption after runner outputs")
            return super().complete(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=InterruptingAnalyzerLLM(),
        enable_literature=False,
    )
    with pytest.raises(KeyboardInterrupt):
        pipeline.run(
            question="Is admission SOFA-2 associated with ICU mortality?",
            cohort=synthetic_cohort,
            cohort_name="resume_test",
            database="synthetic",
            target_outcome="death",
        )

    run_dirs = sorted(tmp_path.glob("run_*"))
    assert run_dirs
    partial = json.loads((run_dirs[-1] / "manifest_partial.json").read_text(encoding="utf-8"))
    records = partial.get("per_step_records") or []
    pending = [
        record
        for record in records
        if record.get("status") == "executed_pending_review"
    ]
    assert pending, records
    assert pending[-1].get("review_pending") is True
    assert pending[-1].get("step_summary"), pending[-1]
    assert pending[-1].get("evidence_ids"), pending[-1]


def test_resume_skips_completed_steps(ra, synthetic_cohort, tmp_path: Path):
    """A second invocation with ``resume_run_id`` should re-use the same
    workdir and add no new step records — every step is already ok."""
    first = _run_full(ra, synthetic_cohort, tmp_path)
    run_dir = Path(first.workdir)
    partial_before = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    assert all(r.get("status") == "ok" for r in partial_before["per_step_records"])
    n_records_before = len(partial_before["per_step_records"])
    n_evidence_before = len(partial_before["evidence"])

    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    second = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="resume_test",
        database="synthetic",
        target_outcome="death",
        resume_run_id=first.run_id,
    )
    assert second.run_id == first.run_id, "resume must reuse the same run_id"
    assert second.workdir == first.workdir, "resume must reuse the same workdir"

    partial_after = json.loads((run_dir / "manifest_partial.json").read_text(encoding="utf-8"))
    statuses_after = [r.get("status") for r in partial_after["per_step_records"]]
    # No new step record should have been added, every step was already ok.
    assert len(partial_after["per_step_records"]) == n_records_before, statuses_after
    # Evidence count may grow by a constant (literature/manuscript/latex
    # are re-emitted on resume); the *step-bound* evidence does not grow.
    assert len(partial_after["evidence"]) >= n_evidence_before


def test_resume_reruns_missing_step(ra, synthetic_cohort, tmp_path: Path):
    """Doctor the partial manifest to drop the last step, then resume —
    the dropped step must be re-executed and ``per_step_records`` must
    grow by exactly one entry."""
    first = _run_full(ra, synthetic_cohort, tmp_path)
    run_dir = Path(first.workdir)
    partial_path = run_dir / "manifest_partial.json"
    partial = json.loads(partial_path.read_text(encoding="utf-8"))

    records = partial["per_step_records"]
    assert len(records) >= 2, "need ≥2 steps to test partial resume"
    drop_index = next(
        (
            i for i, record in enumerate(records)
            if record.get("step_id") == "04_primary_association"
        ),
        len(records) - 1,
    )
    dropped = records.pop(drop_index)
    partial_path.write_text(
        json.dumps(partial, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    second = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="resume_test",
        database="synthetic",
        target_outcome="death",
        resume_run_id=first.run_id,
    )
    assert second.run_id == first.run_id

    partial_after = json.loads(partial_path.read_text(encoding="utf-8"))
    new_step_ids = [r["step_id"] for r in partial_after["per_step_records"]]
    assert dropped["step_id"] in new_step_ids, (
        f"dropped step {dropped['step_id']!r} was not re-executed; new ids: {new_step_ids}"
    )


def test_resume_from_completed_step_can_stop_after_that_step(
    ra, synthetic_cohort, tmp_path: Path
):
    """A reviewed prior plan can be continued from an arbitrary completed step.

    This covers the interactive workflow where a user approves upstream plan
    work, then asks to rerun one downstream step without manually editing the
    checkpoint manifest.
    """
    first = _run_full(ra, synthetic_cohort, tmp_path)
    run_dir = Path(first.workdir)
    plan = json.loads((run_dir / "analysis_plan.json").read_text(encoding="utf-8"))
    plan_order = {
        step["step_id"]: idx for idx, step in enumerate(plan.get("steps") or [])
    }
    assert "04_primary_association" in plan_order
    stop_index = plan_order["04_primary_association"]
    partial_before = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    assert any(
        plan_order.get(record.get("step_id"), -1) > stop_index
        for record in partial_before["per_step_records"]
    ), "full run should have completed steps after the selected resume point"

    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    second = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="resume_test",
        database="synthetic",
        target_outcome="death",
        resume_run_id=first.run_id,
        resume_from_step_id="04_primary_association",
        stop_after_step_id="04_primary_association",
        stop_after_analysis=True,
    )
    assert second.run_id == first.run_id

    partial_after = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    step_ids_after = [record["step_id"] for record in partial_after["per_step_records"]]
    assert "04_primary_association" in step_ids_after
    assert all(
        plan_order.get(step_id, -1) <= stop_index for step_id in step_ids_after
    ), step_ids_after
    resume_findings = [
        finding
        for finding in partial_after["findings"]
        if finding.get("validator") == "resume"
    ]
    assert resume_findings
    dropped = resume_findings[-1]["detail"]["dropped_completed_step_ids"]
    assert "04_primary_association" in dropped
    assert not any(
        finding.get("validator") == "manuscript_gate"
        and finding.get("severity") == "error"
        for finding in partial_after["findings"]
    )


def test_resume_from_step_reuses_prior_code_when_coder_fails(
    ra, tmp_path: Path
):
    """An explicit step resume may reuse prior code evidence if coder is down."""

    class SingleStepLLM:
        name = "single-step-llm"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            upper = user.upper()
            if "ICU-AWARE RESEARCH PLAN" in upper:
                return json.dumps({
                    "research_question": "Is SOFA associated with ICU mortality?",
                    "steps": [{
                        "step_id": "04_primary_association",
                        "intent": "Estimate SOFA and ICU mortality association.",
                        "inputs": ["sofa2", "death"],
                        "expected_outputs": [
                            "table:primary_association",
                            "statistic:primary_or",
                        ],
                        "method": "descriptive",
                        "icu_rule_refs": ["aggregation_rule_for"],
                    }],
                    "rationale": "single-step resume code reuse test",
                })
            if "WRITE THE PYTHON CODE" in upper:
                return """
import json
import os
import pandas as pd

df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
summary = {
    "predictor": "sofa2",
    "n": int(len(df)),
    "sofa2_median": float(df["sofa2"].median()),
    "mortality_rate": float(df["death"].mean()),
    "primary_or": 1.42,
    "primary_or_ci_low": 1.10,
    "primary_or_ci_high": 1.83,
    "model_converged": True,
    "statistic:primary_or": {
        "value": 1.42,
        "ci_low": 1.10,
        "ci_high": 1.83,
    },
}
pd.DataFrame([summary]).to_csv(os.path.join(out, "primary_association.csv"), index=False)
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
"""
            if "INTERPRET THE RESULTS" in upper:
                return "The primary table is available {evidence:primary_association}."
            if "MANUSCRIPT SCAFFOLD" in upper:
                return "# Title\n\n## Results\n\nThe table is available {evidence:primary_association}."
            return "{}"

    class FailingCoderLLM(SingleStepLLM):
        name = "failing-coder-llm"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            if "WRITE THE PYTHON CODE" in user.upper():
                raise RuntimeError("simulated coder outage")
            return super().complete(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    cohort = pd.DataFrame({
        "stay_id": [1, 2, 3, 4],
        "sofa2": [0, 1, 3, 6],
        "death": [1, 0, 0, 1],
    })
    first_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=SingleStepLLM(),
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
    )
    first = first_pipeline.run(
        question="Is SOFA associated with ICU mortality?",
        cohort=cohort,
        cohort_name="resume_code_reuse_test",
        database="synthetic",
        target_outcome="death",
        stop_after_analysis=True,
    )
    run_dir = Path(first.workdir)
    partial_path = run_dir / "manifest_partial.json"
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    bad_code_path = run_dir / "evidence" / "code_bad__analysis.py"
    bad_code_path.write_text("{}", encoding="utf-8")
    partial["evidence"].append({
        "evidence_id": "code_bad",
        "kind": "code",
        "description": "Malformed code evidence that must not be reused.",
        "relative_path": "evidence/code_bad__analysis.py",
        "sha256": "bad",
        "produced_by_step": "04_primary_association",
        "inputs": [],
        "script_evidence_id": None,
        "producer": "coder",
        "generation_mode": "llm",
        "finding_severity": None,
        "finding_messages": [],
        "metadata": {},
        "created_at": "2026-07-01T00:00:00Z",
    })
    partial.setdefault("findings", []).extend([
        {
            "validator": "step_contract",
            "severity": "error",
            "message": (
                "stale pre-resume error for step "
                "04_primary_association should be cleared"
            ),
            "detail": {"step_id": "04_primary_association"},
        },
        {
            "validator": "manuscript_gate",
            "severity": "error",
            "message": "stale pre-resume gate error should be cleared",
            "detail": {
                "failed_steps": [
                    {"step_id": "04_primary_association", "status": "failed"}
                ]
            },
        },
    ])
    partial_path.write_text(
        json.dumps(partial, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    second_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=FailingCoderLLM(),
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
    )
    second = second_pipeline.run(
        question="Is SOFA associated with ICU mortality?",
        cohort=cohort,
        cohort_name="resume_code_reuse_test",
        database="synthetic",
        target_outcome="death",
        resume_run_id=first.run_id,
        resume_from_step_id="04_primary_association",
        stop_after_step_id="04_primary_association",
        stop_after_analysis=True,
    )
    assert second.run_id == first.run_id

    partial = json.loads(
        (Path(second.workdir) / "manifest_partial.json").read_text(encoding="utf-8")
    )
    records = [
        record
        for record in partial["per_step_records"]
        if record.get("step_id") == "04_primary_association"
    ]
    assert records[-1]["status"] == "ok"
    assert records[-1]["generation_mode"] == "resumed_code_reuse"
    assert records[-1]["resumed_code_evidence_id"]
    assert records[-1]["resumed_code_evidence_id"] != "code_bad"
    assert not any(
        "stale pre-resume" in finding.get("message", "")
        for finding in partial["findings"]
    )
    assert any(
        finding.get("validator") == "coder"
        and "reused prior agent-generated code" in finding.get("message", "")
        for finding in partial["findings"]
    )


def test_resume_reuses_locked_plan_instead_of_replanning(
    ra, synthetic_cohort, tmp_path: Path
):
    """Resume must reuse the prior run's ``analysis_plan.json`` rather than
    re-running the planner.

    A non-deterministic hosted planner returns a *different* plan on resume,
    whose step_ids no longer match the completed-step skip set — so the
    "resume" would silently re-run the whole analysis under new names. Pin
    that the locked plan is reused: the resumed planner output is ignored and
    the saved step_ids are unchanged.
    """
    first = _run_full(ra, synthetic_cohort, tmp_path)
    run_dir = Path(first.workdir)
    plan_path = run_dir / "analysis_plan.json"
    step_ids_before = [s["step_id"] for s in json.loads(
        plan_path.read_text(encoding="utf-8"))["steps"]]
    assert step_ids_before, "first run produced no plan steps"

    class DifferentPlanLLM:
        """Planner here returns a plan with a step_id that must never appear
        if the locked plan is reused."""

        name = "different-plan-llm"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            user = next(
                (m.content for m in reversed(messages) if m.role == "user"), ""
            )
            upper = user.upper()
            if "ICU-AWARE RESEARCH PLAN" in upper:
                return json.dumps({
                    "research_question": "Is admission SOFA-2 associated with ICU mortality?",
                    "steps": [{
                        "step_id": "88_resume_should_ignore_this",
                        "intent": "This plan must be ignored on resume.",
                        "inputs": ["sofa2", "death"],
                        "expected_outputs": ["table:ignored"],
                        "method": "descriptive",
                        "icu_rule_refs": ["aggregation_rule_for"],
                    }],
                    "rationale": "resume must not use this plan",
                })
            if "INTERPRET THE RESULTS" in upper:
                return "Reused-plan interpretation {evidence:primary_association}."
            if "MANUSCRIPT SCAFFOLD" in upper:
                return "# Title\n\n## Results\n\nReused {evidence:primary_association}."
            return "{}"

    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=DifferentPlanLLM())
    second = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="resume_test",
        database="synthetic",
        target_outcome="death",
        resume_run_id=first.run_id,
    )
    assert second.run_id == first.run_id

    step_ids_after = [s["step_id"] for s in json.loads(
        plan_path.read_text(encoding="utf-8"))["steps"]]
    assert step_ids_after == step_ids_before, (
        "resume re-planned instead of reusing the locked plan: "
        f"{step_ids_before} -> {step_ids_after}"
    )
    assert "88_resume_should_ignore_this" not in step_ids_after

    manifest = json.loads(Path(second.manifest_path).read_text(encoding="utf-8"))
    assert any(
        (f.get("detail") or {}).get("generation_mode") == "resumed"
        for f in manifest["findings"]
    ), "no 'resumed' planner finding recorded"


def test_resume_to_nonexistent_run_id_starts_fresh_directory(ra, synthetic_cohort,
                                                             tmp_path: Path):
    """Passing a resume_run_id that has no prior run_dir should still
    work — the pipeline creates the directory and runs everything from
    scratch (the partial manifest is just absent)."""
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    result = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="resume_test",
        database="synthetic",
        target_outcome="death",
        resume_run_id="run_does_not_exist_yet",
    )
    assert result.run_id == "run_does_not_exist_yet"
    assert (Path(result.workdir) / "manifest.json").exists()
    assert (Path(result.workdir) / "manifest_partial.json").exists()


def test_final_manifest_keeps_step_records_for_metered_hosted_stub(ra, tmp_path: Path):
    """The final manifest must keep per-step resume records outside Mock paths.

    The real hosted path is wrapped in ``MeteredClient`` when cost
    tracking is enabled. This stub exercises that routing without a
    network call and pins the final ``manifest.json`` contract that
    paper/provenance tooling reads.
    """

    class HostedStubLLM:
        name = "hosted-stub"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            upper = user.upper()
            if "ICU-AWARE RESEARCH PLAN" in upper:
                return json.dumps({
                    "research_question": "Is SOFA associated with ICU mortality?",
                    "steps": [{
                        "step_id": "04_primary_association",
                        "intent": "Estimate SOFA and ICU mortality association.",
                        "inputs": ["sofa2", "death"],
                        "expected_outputs": ["table:primary_association"],
                        "method": "descriptive",
                        "icu_rule_refs": ["aggregation_rule_for"],
                    }],
                    "rationale": "single-step hosted-stub resume test",
                })
            if "WRITE THE PYTHON CODE" in upper:
                return """
import json
import os
import pandas as pd

df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
summary = {
    "predictor": "sofa2",
    "n": int(len(df)),
    "sofa2_median": float(df["sofa2"].median()),
    "mortality_rate": float(df["death"].mean()),
}
pd.DataFrame([summary]).to_csv(os.path.join(out, "primary_association.csv"), index=False)
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
"""
            if "INTERPRET THE RESULTS" in upper:
                return "The primary association table is available {evidence:primary_association}."
            if "MANUSCRIPT SCAFFOLD" in upper:
                return "# Title\n\n## Results\n\nThe table is available {evidence:primary_association}."
            return "{}"

    cohort = pd.DataFrame({
        "stay_id": [1, 2, 3, 4],
        "sofa2": [0, 1, 3, 6],
        "death": [1, 0, 0, 1],
    })
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=HostedStubLLM(),
        enable_cost_tracking=True,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
    )
    result = pipeline.run(
        question="Is SOFA associated with ICU mortality?",
        cohort=cohort,
        cohort_name="hosted_stub_resume_test",
        database="synthetic",
        target_outcome="death",
    )

    run_dir = Path(result.workdir)
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    partial = json.loads((run_dir / "manifest_partial.json").read_text(encoding="utf-8"))

    assert manifest["per_step_records"], "final manifest dropped per-step records"
    assert manifest["per_step_records"] == partial["per_step_records"]
    assert manifest["per_step_records"][0]["status"] == "ok"
    assert manifest["cost_records"], "hosted-stub path should be metered"
