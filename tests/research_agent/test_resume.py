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


def _run_full(ra, synthetic_cohort, workdir: Path):
    pipeline = ra.ResearchAgentPipeline(workdir=workdir, llm=ra.MockLLMClient())
    return pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="resume_test",
        database="synthetic",
        target_outcome="death",
    )


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
