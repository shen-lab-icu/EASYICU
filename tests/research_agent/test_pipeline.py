"""End-to-end pipeline test with the synthetic SOFA cohort.

This is the integration test the ROADMAP's "mock pipeline must always
pass" rule rests on. If this regresses, the demo (and any reviewer
clicking "run") gets a broken artefact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def test_pipeline_end_to_end_synthetic_cohort(ra, synthetic_cohort, tmp_path: Path):
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    result = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="synthetic_test_cohort",
        database="synthetic",
        target_outcome="death",
        cross_database_validation=["miiv", "eicu"],
    )
    # 1) The result paths exist and are populated.
    paths = result.as_paths()
    for k, p in paths.items():
        assert Path(p).exists(), f"missing {k}: {p}"

    run_dir = Path(result.workdir)

    # 2) Manifest and evidence were written.
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["evidence"], "manifest has no registered evidence"
    kinds = {e["kind"] for e in manifest["evidence"]}
    assert {"code", "log", "table", "figure", "statistic"} <= kinds, (
        f"evidence kinds incomplete: {kinds}"
    )
    # at least 6 artefacts as required by the roadmap
    assert len(manifest["evidence"]) >= 6, manifest["evidence"]

    # 3) The bound manuscript should have ZERO ``[evidence missing: …]``
    #    lines (T1.2 acceptance criterion).
    bound = (run_dir / "manuscript_scaffold_bound.md").read_text(encoding="utf-8")
    assert "[evidence missing:" not in bound, (
        "bound manuscript contains unresolved evidence placeholders:\n" + bound
    )

    # 4) The SOFA-zero anomaly should appear in at least one step_summary.json.
    summaries = list(run_dir.rglob("step_summary.json"))
    assert summaries, "no step_summary.json was produced"
    flagged = False
    for ssj in summaries:
        try:
            data = json.loads(ssj.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("sofa_zero_anomaly"):
            flagged = True
            break
    assert flagged, "synthetic cohort SOFA2==0 anomaly was not detected"

    # 5) The manifest's findings should mention the anomaly.
    finding_msgs = " ".join(f.get("message", "") for f in manifest["findings"])
    assert "non-monotonic" in finding_msgs.lower() or "exceeds" in finding_msgs.lower(), (
        f"validator did not surface the SOFA-zero anomaly:\n{finding_msgs}"
    )


def test_pipeline_with_clinical_skill(ra, synthetic_cohort, tmp_path: Path):
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    result = pipeline.run(
        cohort=synthetic_cohort,
        cohort_name="synthetic_skill_cohort",
        database="synthetic",
        skill="sofa_mortality",
    )
    assert result.evidence_count > 0
    # The skill plan must include a sofa_zero audit step.
    plan = json.loads(Path(result.plan_path).read_text(encoding="utf-8"))
    step_ids = [s["step_id"] for s in plan["steps"]]
    assert any("sofa_zero" in sid for sid in step_ids)


def test_pipeline_falls_back_when_planner_returns_empty(ra, synthetic_cohort, tmp_path: Path):
    class EmptyPlanner:
        name = "empty-planner"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            return ""

    router = ra.LLMRouter(default=ra.MockLLMClient(), planner=EmptyPlanner())
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=router)
    result = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="planner_fallback_test",
        database="synthetic",
        target_outcome="death",
    )
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert any(f["validator"] == "planner" for f in manifest["findings"])
    assert Path(result.plan_path).exists()


def test_pipeline_repairs_failed_generated_code(ra, tmp_path: Path):
    """A real-LLM style traceback should trigger one coder repair pass."""

    class RepairLLM:
        name = "repair-llm"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            upper = user.upper()
            if "ICU-AWARE RESEARCH PLAN" in upper:
                return json.dumps({
                    "research_question": "Does age describe ICU mortality?",
                    "steps": [{
                        "step_id": "01_table_one",
                        "intent": "Write a compact cohort table.",
                        "inputs": ["age", "death"],
                        "expected_outputs": ["table:table_one"],
                        "method": "descriptive",
                        "icu_rule_refs": ["aggregation_rule_for"],
                    }],
                    "rationale": "minimal repair test",
                })
            if "REPAIR THE PYTHON CODE" in upper:
                return """
import json
import os
import pandas as pd

df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
pd.DataFrame({"n": [int(len(df))]}).to_csv(os.path.join(out, "table_one.csv"), index=False)
summary = {"n": int(len(df))}
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
print(json.dumps(summary))
"""
            if "WRITE THE PYTHON CODE" in upper:
                return "raise KeyError('intentional first pass failure')\n"
            if "INTERPRET THE RESULTS" in upper:
                return "The cohort table was produced {evidence:table_one}."
            if "MANUSCRIPT SCAFFOLD" in upper:
                return "# Title\n\n## Results\n\nThe cohort table was produced {evidence:table_one}.\n\n(left to the human author)"
            return "{}"

    cohort = pd.DataFrame({"age": [50, 60, 70], "death": [0, 1, 0]})
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=RepairLLM(),
        enable_literature=False,
    )
    result = pipeline.run(
        question="Does age describe ICU mortality?",
        cohort=cohort,
        cohort_name="repair_test",
        database="synthetic",
        target_outcome="death",
    )

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    partial = json.loads((Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8"))
    assert partial["per_step_records"][0]["status"] == "ok"
    assert partial["per_step_records"][0]["code_repair_attempts"] == 1
    assert not [
        f for f in manifest["findings"]
        if f["severity"] == "error" and f["validator"] == "runner"
    ]


def test_pipeline_repairs_concept_audit_violation(ra, tmp_path: Path):
    """A generated mean-of-SOFA script should be repaired before execution."""

    class ConceptRepairLLM:
        name = "concept-repair-llm"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            upper = user.upper()
            if "ICU-AWARE RESEARCH PLAN" in upper:
                return json.dumps({
                    "research_question": "Does SOFA describe ICU mortality?",
                    "steps": [{
                        "step_id": "04_primary_association",
                        "intent": "Assess SOFA-2 and mortality.",
                        "inputs": ["sofa2", "death"],
                        "expected_outputs": ["table:primary_association"],
                        "method": "regression",
                        "icu_rule_refs": ["aggregation_rule_for"],
                    }],
                    "rationale": "minimal concept repair test",
                })
            if "REPAIR THE PYTHON CODE" in upper:
                return """
import json
import os
import pandas as pd

df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
pd.DataFrame({
    "variable": ["sofa2"],
    "median": [float(df["sofa2"].median())],
}).to_csv(os.path.join(out, "primary_association.csv"), index=False)
summary = {"predictor": "sofa2", "sofa2_median": float(df["sofa2"].median())}
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
print(json.dumps(summary))
"""
            if "WRITE THE PYTHON CODE" in upper:
                return """
import os
import pandas as pd
df = pd.read_parquet(os.environ["COHORT_PARQUET"])
bad = df["sofa2"].mean()
pd.DataFrame({"bad": [bad]}).to_csv(os.path.join(os.environ["STEP_OUT_DIR"], "primary_association.csv"), index=False)
"""
            if "INTERPRET THE RESULTS" in upper:
                return "The repaired table was produced {evidence:primary_association_table}."
            if "MANUSCRIPT SCAFFOLD" in upper:
                return "# Title\n\n## Results\n\nThe repaired table was produced {evidence:primary_association_table}.\n\n(left to the human author)"
            return "{}"

    cohort = pd.DataFrame({"sofa2": [0, 1, 3, 4], "death": [1, 0, 0, 1]})
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=ConceptRepairLLM(),
        enable_literature=False,
    )
    result = pipeline.run(
        question="Does SOFA describe ICU mortality?",
        cohort=cohort,
        cohort_name="concept_repair_test",
        database="synthetic",
        target_outcome="death",
    )

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    partial = json.loads((Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8"))
    assert partial["per_step_records"][0]["status"] == "ok"
    assert partial["per_step_records"][0]["concept_repair_attempts"] == 1
    assert not [
        f for f in manifest["findings"]
        if f["severity"] == "error" and f["validator"] == "concept_usage_auditor"
    ]


def test_pipeline_falls_back_to_deterministic_code_after_repair_failure(
    ra, tmp_path: Path
):
    """If hosted-model code and its repair both fail, use mock-safe code."""

    class FallbackLLM:
        name = "fallback-llm"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            upper = user.upper()
            if "ICU-AWARE RESEARCH PLAN" in upper:
                return json.dumps({
                    "research_question": "Is SOFA associated with ICU mortality?",
                    "steps": [{
                        "step_id": "01_table_one",
                        "intent": "Produce a Table 1 cohort summary.",
                        "inputs": ["sofa2", "death"],
                        "expected_outputs": ["table:table_one"],
                        "method": "descriptive",
                        "icu_rule_refs": ["aggregation_rule_for"],
                    }],
                    "rationale": "minimal fallback test",
                })
            if "WRITE THE PYTHON CODE" in upper or "REPAIR THE PYTHON CODE" in upper:
                return "raise RuntimeError('still broken')\n"
            if "INTERPRET THE RESULTS" in upper:
                return "The fallback table was produced {evidence:table_one}."
            if "MANUSCRIPT SCAFFOLD" in upper:
                return "# Title\n\n## Results\n\nThe fallback table was produced {evidence:table_one}.\n\n(left to the human author)"
            return "{}"

    cohort = pd.DataFrame({"sofa2": [0, 1, 3, 4], "death": [1, 0, 0, 1]})
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=FallbackLLM(),
        enable_literature=False,
    )
    result = pipeline.run(
        question="Is SOFA associated with ICU mortality?",
        cohort=cohort,
        cohort_name="fallback_test",
        database="synthetic",
        target_outcome="death",
    )

    partial = json.loads((Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8"))
    record = partial["per_step_records"][0]
    assert record["status"] == "ok"
    assert record["deterministic_code_fallback"] == "execution_failure"


def test_pipeline_falls_back_when_repair_model_call_fails(ra, tmp_path: Path):
    """A provider 429 during repair should not strand the whole step."""

    class RepairRaisesLLM:
        name = "repair-raises-llm"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            upper = user.upper()
            if "ICU-AWARE RESEARCH PLAN" in upper:
                return json.dumps({
                    "research_question": "Is SOFA associated with ICU mortality?",
                    "steps": [{
                        "step_id": "01_table_one",
                        "intent": "Produce a Table 1 cohort summary.",
                        "inputs": ["sofa2", "death"],
                        "expected_outputs": ["table:table_one"],
                        "method": "descriptive",
                        "icu_rule_refs": ["aggregation_rule_for"],
                    }],
                    "rationale": "minimal repair-failure fallback test",
                })
            if "REPAIR THE PYTHON CODE" in upper:
                raise RuntimeError("provider rate limited")
            if "WRITE THE PYTHON CODE" in upper:
                return "raise RuntimeError('broken first draft')\n"
            if "INTERPRET THE RESULTS" in upper:
                return "The fallback table was produced {evidence:table_one}."
            if "MANUSCRIPT SCAFFOLD" in upper:
                return "# Title\n\n## Results\n\nThe fallback table was produced {evidence:table_one}.\n\n(left to the human author)"
            return "{}"

    cohort = pd.DataFrame({"sofa2": [0, 1, 3, 4], "death": [1, 0, 0, 1]})
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=RepairRaisesLLM(),
        enable_literature=False,
    )
    result = pipeline.run(
        question="Is SOFA associated with ICU mortality?",
        cohort=cohort,
        cohort_name="repair_raises_test",
        database="synthetic",
        target_outcome="death",
    )

    partial = json.loads((Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8"))
    record = partial["per_step_records"][0]
    assert record["status"] == "ok"
    assert record["deterministic_code_fallback"] == "repair_failed"


def test_pipeline_falls_back_when_successful_script_writes_no_artefacts(
    ra, tmp_path: Path
):
    """Exit-code 0 with an empty output dir is not a usable analysis step."""

    class NoArtefactLLM:
        name = "no-artefact-llm"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            upper = user.upper()
            if "ICU-AWARE RESEARCH PLAN" in upper:
                return json.dumps({
                    "research_question": "Is SOFA associated with ICU mortality?",
                    "steps": [{
                        "step_id": "01_table_one",
                        "intent": "Produce a Table 1 cohort summary.",
                        "inputs": ["sofa2", "death"],
                        "expected_outputs": ["table:table_one"],
                        "method": "descriptive",
                        "icu_rule_refs": ["aggregation_rule_for"],
                    }],
                    "rationale": "minimal no-artefact fallback test",
                })
            if "WRITE THE PYTHON CODE" in upper:
                return "print('I forgot to write outputs')\n"
            if "INTERPRET THE RESULTS" in upper:
                return "The fallback table was produced {evidence:table_one}."
            if "MANUSCRIPT SCAFFOLD" in upper:
                return "# Title\n\n## Results\n\nThe fallback table was produced {evidence:table_one}.\n\n(left to the human author)"
            return "{}"

    cohort = pd.DataFrame({"sofa2": [0, 1, 3, 4], "death": [1, 0, 0, 1]})
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=NoArtefactLLM(),
        enable_literature=False,
    )
    result = pipeline.run(
        question="Is SOFA associated with ICU mortality?",
        cohort=cohort,
        cohort_name="no_artefact_test",
        database="synthetic",
        target_outcome="death",
    )

    partial = json.loads((Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8"))
    record = partial["per_step_records"][0]
    assert record["status"] == "ok"
    assert record["deterministic_code_fallback"] == "no_artefacts"


def test_composite_audit_outputs_get_primary_association_alias(ra):
    from easyicu.research_agent.pipeline import _semantic_aliases_for

    step = ra.schema.AnalysisStep(
        step_id="04_composite_audit",
        intent="Check stratum-level component balance.",
    )
    aliases = _semantic_aliases_for(step, Path("step_summary.json"))
    assert "primary_association" in aliases


def test_pipeline_can_pause_after_analysis_phase(ra, synthetic_cohort, tmp_path: Path):
    events = []
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    result = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="synthetic_analysis_only",
        database="synthetic",
        target_outcome="death",
        stop_after_analysis=True,
        progress_callback=events.append,
    )

    run_dir = Path(result.workdir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "paused_after_analysis" in manifest["notes"]
    assert not (run_dir / "manuscript_scaffold.tex").exists()
    assert not (run_dir / "literature_bundle.json").exists()
    bound = (run_dir / "manuscript_scaffold_bound.md").read_text(encoding="utf-8")
    assert "stopped after the analysis phase" in bound
    report = (run_dir / "results_report.md").read_text(encoding="utf-8")
    assert "PAUSED AFTER ANALYSIS" in report
    assert any(e.get("stage") == "step" and e.get("status") == "complete" for e in events)
    assert any(e.get("stage") == "pause" and e.get("status") == "paused" for e in events)


def test_mock_planner_honours_sofa2_when_sofa_is_also_present(ra, synthetic_cohort, tmp_path: Path):
    """A SOFA-2 question must not silently fall back to a legacy ``sofa`` column."""
    cohort = synthetic_cohort.copy()
    cohort.insert(3, "sofa", cohort["sofa2"])

    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    result = pipeline.run(
        question="Is early SOFA-2 associated with ICU mortality?",
        cohort=cohort,
        cohort_name="synthetic_sofa_and_sofa2",
        database="synthetic",
        target_outcome="death",
    )

    plan = json.loads(Path(result.plan_path).read_text(encoding="utf-8"))
    by_id = {step["step_id"]: step for step in plan["steps"]}
    assert by_id["04_primary_association"]["inputs"][:2] == ["sofa2", "death"]
    assert by_id["05_sofa_zero_audit"]["inputs"][:2] == ["sofa2", "death"]


def test_mock_planner_maps_clinical_phrases_to_expected_predictors(ra, tmp_path: Path):
    """Clinical wording such as KDIGO stage / vasopressor should not fall back to age."""
    cases = [
        (
            "Is peak first-24h KDIGO AKI stage associated with ICU mortality?",
            "kdigo_stage",
            pd.DataFrame({
                "stay_id": range(1, 81),
                "age": [60 + (i % 20) for i in range(80)],
                "kdigo_stage": [i % 4 for i in range(80)],
                "creat": [0.8 + 0.2 * (i % 4) for i in range(80)],
                "death": [1 if i % 5 == 0 else 0 for i in range(80)],
            }),
        ),
        (
            "Is any-vasopressor exposure within the first 24h associated with ICU mortality?",
            "vaso",
            pd.DataFrame({
                "stay_id": range(1, 81),
                "age": [60 + (i % 20) for i in range(80)],
                "vaso": [i % 2 for i in range(80)],
                "death": [1 if i % 5 == 0 else 0 for i in range(80)],
            }),
        ),
    ]

    for question, predictor, cohort in cases:
        pipeline = ra.ResearchAgentPipeline(workdir=tmp_path / predictor, llm=ra.MockLLMClient())
        result = pipeline.run(
            question=question,
            cohort=cohort,
            cohort_name=f"{predictor}_phrase_test",
            database="synthetic",
            target_outcome="death",
        )
        plan = json.loads(Path(result.plan_path).read_text(encoding="utf-8"))
        by_id = {step["step_id"]: step for step in plan["steps"]}
        assert by_id["04_primary_association"]["inputs"][:2] == [predictor, "death"]
