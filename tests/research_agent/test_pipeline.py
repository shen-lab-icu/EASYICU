"""End-to-end pipeline test with the synthetic SOFA cohort.

This is the integration test the ROADMAP's "mock pipeline must always
pass" rule rests on. If this regresses, the demo (and any reviewer
clicking "run") gets a broken artefact.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def _step_record_by_id(records, step_id: str):
    for record in records:
        if record.get("step_id") == step_id:
            return record
    raise AssertionError(f"step record {step_id!r} not found in {records!r}")


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
    assert manifest["used_mock_llm"] is True
    assert manifest["prompt_pack_version"].startswith("easyicu-research-agent-prompts/")
    assert manifest["prompt_pack_files"]
    assert manifest["concept_dict_path"] == "easyicu/data/concept-dict.json"
    assert len(manifest["concept_dict_sha"]) == 64
    assert set(manifest["concept_dict_sha"]) <= set("0123456789abcdef")
    kinds = {e["kind"] for e in manifest["evidence"]}
    assert {"code", "log", "table", "figure", "statistic"} <= kinds, (
        f"evidence kinds incomplete: {kinds}"
    )
    # at least 6 artefacts as required by the roadmap
    assert len(manifest["evidence"]) >= 6, manifest["evidence"]
    evidence_ids = {e["evidence_id"] for e in manifest["evidence"]}
    assert {
        "clinical_semantics_resolution",
        "data_extraction_request",
        "data_extraction_result",
        "hypothesis_blueprint",
        "preplan_literature_bundle",
        "publication_figure_contract",
        "publication_figure_skill_summary",
        "publication_figure_svg",
        "manuscript_critique",
    } <= evidence_ids
    assert (run_dir / "hypothesis_blueprint.json").exists()

    # 3) The bound manuscript should have ZERO ``[evidence missing: …]``
    #    lines (T1.2 acceptance criterion).
    bound = (run_dir / "manuscript_scaffold_bound.md").read_text(encoding="utf-8")
    assert "[evidence missing:" not in bound, (
        "bound manuscript contains unresolved evidence placeholders:\n" + bound
    )
    partial = json.loads((run_dir / "manifest_partial.json").read_text(encoding="utf-8"))
    assert partial["runtime_state"]["analysis_family"]

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


def test_pipeline_stops_when_hypothesis_blueprint_is_blocked(
    ra,
    tmp_path: Path,
    monkeypatch,
):
    cohort_path = tmp_path / "cohort_no_outcome.parquet"
    pd.DataFrame({
        "stay_id": [1, 2, 3],
        "sofa2": [0, 2, 5],
    }).to_parquet(cohort_path)

    import easyicu.research_agent.pipeline as pipeline_module

    class BlockedBlueprintAgent:
        def run(self, *, context, literature):
            return ra.schema.HypothesisBlueprint(
                research_question=context.research_question,
                hypothesis="Target outcome is unavailable.",
                hypothesis_type="feasibility",
                feasible_variables=["sofa2"],
                missing_variables=["target_outcome"],
                stepwise_plan=["Resolve target outcome before modeling."],
                feasibility_status="blocked",
                domain_gate_notes=["No target outcome is available."],
            )

    monkeypatch.setattr(
        pipeline_module,
        "HypothesisBlueprintAgent",
        BlockedBlueprintAgent,
    )

    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=ra.MockLLMClient(),
        submission_profile_name="npj_dm",
        submission_profile_version="20260527",
        submission_profile_locked_at="2026-05-27T00:00:00Z",
    )
    result = pipeline.run(
        question="Describe admission SOFA-2 signal in this ICU cohort.",
        cohort=cohort_path,
        cohort_name="blocked_blueprint_cohort",
        database="synthetic",
    )

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["notes"] == "aborted: hypothesis_blueprint_blocked"
    assert manifest["submission_profile_name"] == "npj_dm"
    assert manifest["submission_profile_version"] == "20260527"
    assert manifest["submission_profile_locked_at"] == "2026-05-27T00:00:00Z"
    assert result.plan_path == ""
    assert any(
        finding["validator"] == "hypothesis_blueprint"
        and finding["severity"] == "error"
        for finding in manifest["findings"]
    )
    assert any(e["evidence_id"] == "hypothesis_blueprint" for e in manifest["evidence"])


def test_pipeline_run_async(ra, synthetic_cohort, tmp_path: Path):
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())

    async def _run():
        return await pipeline.run_async(
            question="Is admission SOFA-2 associated with ICU mortality?",
            cohort=synthetic_cohort,
            cohort_name="synthetic_async_cohort",
            database="synthetic",
            target_outcome="death",
        )

    result = asyncio.run(_run())
    assert Path(result.manifest_path).exists()


def test_pipeline_falls_back_when_planner_returns_empty(ra, synthetic_cohort, tmp_path: Path):
    class EmptyPlanner:
        name = "empty-planner"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            return ""

    router = ra.LLMRouter(default=ra.MockLLMClient(), planner=EmptyPlanner())
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=router,
        enable_deterministic_planner_fallback=True,
    )
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


def test_remove_tbd_sentences_strips_placeholder_results(ra):
    from easyicu.research_agent.pipeline import _remove_tbd_sentences

    bound = (
        "## Results\n\n"
        "The cohort comprised 800 patients {evidence:table_one}. "
        "Median age was [TBD] years {evidence:table_one}. "
        "SOFA-2 was associated with mortality (OR 1.17) {evidence:primary_association}.\n"
    )

    cleaned, removed = _remove_tbd_sentences(bound)

    assert "Median age was [TBD] years" not in cleaned
    assert "The cohort comprised 800 patients" in cleaned
    assert "SOFA-2 was associated with mortality" in cleaned
    assert removed == ["Median age was [TBD] years {evidence:table_one}."]


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
        # Disable deterministic runner repair so LLM repair path is exercised
        enable_deterministic_runner_repair=False,
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
    record = _step_record_by_id(partial["per_step_records"], "01_table_one")
    assert record["status"] == "ok"
    assert record["code_repair_attempts"] == 1
    assert not [
        f for f in manifest["findings"]
        if f["severity"] == "error" and f["validator"] == "runner"
    ]


def test_promote_prior_publication_bundle_copies_real_figure_exports(tmp_path: Path):
    from easyicu.research_agent.pipeline import _promote_prior_publication_bundle

    run_dir = tmp_path / "run"
    source_dir = run_dir / "steps" / "05_primary_association" / "outputs"
    target_dir = run_dir / "steps" / "06_publication_figure_generation" / "outputs"
    source_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)

    (source_dir / "primary_association_curve.png").write_bytes(b"png")
    (source_dir / "primary_association_curve.svg").write_text("<svg><text>A</text></svg>", encoding="utf-8")
    (source_dir / "primary_association_curve.pdf").write_bytes(b"%PDF-1.4")
    (source_dir / "primary_association_curve.tiff").write_bytes(b"TIFF")
    (source_dir / "primary_association_curve.figure_contract.json").write_text(
        json.dumps({"figure_id": "primary_association_curve"}),
        encoding="utf-8",
    )

    repair = _promote_prior_publication_bundle(
        run_dir=run_dir,
        current_step_id="06_publication_figure_generation",
        out_dir=target_dir,
    )

    assert repair == "publication_bundle_promote_v1"
    assert (target_dir / "publication_figure.png").exists()
    assert (target_dir / "publication_figure.svg").exists()
    assert (target_dir / "publication_figure.pdf").exists()
    assert (target_dir / "publication_figure.tiff").exists()
    assert (target_dir / "publication_figure.figure_contract.json").exists()
    summary = json.loads((target_dir / "step_summary.json").read_text(encoding="utf-8"))
    assert summary["publication_figure_rescue"]["mode"] == "promotion"
    assert summary["figure_path"] == "publication_figure.pdf"
    assert sorted(summary["figure_files"]) == [
        "publication_figure.pdf",
        "publication_figure.png",
        "publication_figure.svg",
        "publication_figure.tiff",
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
summary = {
    "predictor": "sofa2",
    "sofa2_median": float(df["sofa2"].median()),
    "primary_or": 1.0,
}
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
    record = _step_record_by_id(partial["per_step_records"], "04_primary_association")
    assert record["status"] == "ok"
    assert record["concept_repair_attempts"] == 1
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
        enable_deterministic_code_fallback=True,
        # Disable deterministic runner repair so code fallback path is exercised
        enable_deterministic_runner_repair=False,
    )
    result = pipeline.run(
        question="Is SOFA associated with ICU mortality?",
        cohort=cohort,
        cohort_name="fallback_test",
        database="synthetic",
        target_outcome="death",
    )

    partial = json.loads((Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8"))
    record = _step_record_by_id(partial["per_step_records"], "01_table_one")
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
        enable_deterministic_code_fallback=True,
        # Disable deterministic runner repair so the code fallback path is exercised
        enable_deterministic_runner_repair=False,
    )
    result = pipeline.run(
        question="Is SOFA associated with ICU mortality?",
        cohort=cohort,
        cohort_name="repair_raises_test",
        database="synthetic",
        target_outcome="death",
    )

    partial = json.loads((Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8"))
    record = _step_record_by_id(partial["per_step_records"], "01_table_one")
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
        enable_deterministic_code_fallback=True,
    )
    result = pipeline.run(
        question="Is SOFA associated with ICU mortality?",
        cohort=cohort,
        cohort_name="no_artefact_test",
        database="synthetic",
        target_outcome="death",
    )

    partial = json.loads((Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8"))
    record = _step_record_by_id(partial["per_step_records"], "01_table_one")
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


def test_association_model_outputs_get_primary_association_alias(ra):
    from easyicu.research_agent.pipeline import _semantic_aliases_for

    step = ra.schema.AnalysisStep(
        step_id="03_association_model",
        intent="Fit adjusted association model.",
    )
    aliases = _semantic_aliases_for(step, Path("step_summary.json"))
    assert "primary_association" in aliases
    assert "association_model" in aliases


def test_stratified_mortality_outputs_get_outcome_rate_alias(ra):
    from easyicu.research_agent.pipeline import _semantic_aliases_for

    step = ra.schema.AnalysisStep(
        step_id="01_stratified_mortality",
        intent="Estimate mortality across severity strata.",
    )
    aliases = _semantic_aliases_for(step, Path("step_summary.json"))
    assert "stratified_mortality" in aliases
    assert "outcome_rate" in aliases
    assert "primary_association" in aliases


def test_stratified_incidence_outputs_get_outcome_rate_alias(ra):
    from easyicu.research_agent.pipeline import _semantic_aliases_for

    step = ra.schema.AnalysisStep(
        step_id="01_stratified_incidence",
        intent="Estimate mortality across SOFA-2 strata.",
    )
    aliases = _semantic_aliases_for(step, Path("step_summary.json"))
    assert "stratified_mortality" in aliases
    assert "outcome_rate" in aliases
    assert "primary_association" in aliases


def test_sofa2_mortality_figure_gets_base_alias(ra):
    from easyicu.research_agent.pipeline import _semantic_aliases_for

    step = ra.schema.AnalysisStep(
        step_id="02_mortality_figure",
        intent="Plot mortality by SOFA-2 stratum.",
    )
    aliases = _semantic_aliases_for(step, Path("mortality_by_sofa2_stratum.png"))
    assert "mortality_by_sofa2_stratum" in aliases
    assert "figure_mortality_by_sofa2_stratum" in aliases


def test_correlation_analysis_outputs_get_primary_association_alias(ra):
    from easyicu.research_agent.pipeline import _semantic_aliases_for

    step = ra.schema.AnalysisStep(
        step_id="02_correlation_analysis",
        intent="Compute Spearman component-total correlations.",
    )
    aliases = _semantic_aliases_for(step, Path("step_summary.json"))
    assert "primary_association" in aliases
    assert "correlation_summary" in aliases
    assert "spearman_correlation" in aliases


def test_correlation_heatmap_gets_base_alias(ra):
    from easyicu.research_agent.pipeline import _semantic_aliases_for

    step = ra.schema.AnalysisStep(
        step_id="03_visualization",
        intent="Render correlation heatmap.",
    )
    aliases = _semantic_aliases_for(step, Path("sofa2_correlation_heatmap.png"))
    assert "sofa2_correlation_heatmap" in aliases
    assert "correlation_figure" in aliases


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


def test_pipeline_run_requires_explicit_llm(tmp_path: Path):
    import easyicu.research_agent as ra

    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path)
    cohort = pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]})
    with pytest.raises(ValueError, match="requires an explicit `llm=`"):
        pipeline.run(
            question="Does death exist?",
            cohort=cohort,
            cohort_name="missing_llm",
            database="synthetic",
            target_outcome="death",
        )


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
        # The mock planner emits ``04_primary_association`` for plain
        # association questions, but ``_normalise_plan_for_family`` may
        # consolidate it into a canonical ``02_treatment_exposure_bias_association``
        # step when the question matches the bias_audit family (e.g. the
        # vasopressor case). What we actually want to pin is that the
        # primary predictor wired into the association step matches the
        # clinical phrasing — irrespective of the consolidated step id.
        assoc_steps = [
            step
            for step in plan["steps"]
            if "association" in step["step_id"]
            and not step["step_id"].endswith("_figure")
        ]
        assert assoc_steps, plan["steps"]
        primary_assoc = assoc_steps[0]
        assert primary_assoc["inputs"][:2] == [predictor, "death"], primary_assoc


def test_mock_planner_skips_table_one_for_minimal_association_question(ra, tmp_path: Path):
    """A narrow association question should not force a cohort-summary step."""
    cohort = pd.DataFrame({
        "stay_id": range(1, 81),
        "gcs": [15 - (i % 6) for i in range(80)],
        "death": [1 if i % 7 == 0 else 0 for i in range(80)],
    })

    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    result = pipeline.run(
        question="Is GCS associated with ICU mortality?",
        cohort=cohort,
        cohort_name="minimal_gcs_mortality",
        database="synthetic",
        target_outcome="death",
    )

    plan = json.loads(Path(result.plan_path).read_text(encoding="utf-8"))
    step_ids = [step["step_id"] for step in plan["steps"]]
    assert "01_table_one" not in step_ids
    assert "04_primary_association" in step_ids


def test_mock_planner_uses_quality_only_plan_when_question_is_data_audit(ra, tmp_path: Path):
    """Data-quality questions should not silently expand into effect-estimation steps."""
    cohort = pd.DataFrame({
        "stay_id": range(1, 61),
        "bili": [None if i % 5 == 0 else 0.8 + 0.1 * (i % 4) for i in range(60)],
        "vaso": [None if i % 4 == 0 else int(i % 3 == 0) for i in range(60)],
        "death": [1 if i % 6 == 0 else 0 for i in range(60)],
    })

    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    result = pipeline.run(
        question="Audit bilirubin and vasopressor measurement completeness in this ICU cohort.",
        cohort=cohort,
        cohort_name="quality_only_audit",
        database="synthetic",
        target_outcome="death",
    )

    plan = json.loads(Path(result.plan_path).read_text(encoding="utf-8"))
    step_ids = [step["step_id"] for step in plan["steps"]]
    # The data-quality skill emits a single ``03_missingness_audit`` step.
    # The plan-contract guard may split it into a table step and an
    # appended figure-only follow-up (``03_missingness_audit_figure``)
    # when the source step declares both ``table:`` and ``figure:``
    # outputs. Either shape is acceptable here as long as no foreign
    # effect-estimation step is introduced.
    assert step_ids[0] == "03_missingness_audit", step_ids
    assert all(
        sid.startswith("03_missingness_audit") for sid in step_ids
    ), step_ids


def test_pipeline_replicate_writes_cross_database_comparison(ra, tmp_path: Path):
    cohorts = {
        "miiv": pd.DataFrame({
            "stay_id": range(1, 31),
            "age": [60 + (i % 8) for i in range(30)],
            "sofa2": [i % 6 for i in range(30)],
            "death": [1 if i % 5 == 0 else 0 for i in range(30)],
        }),
        "eicu": pd.DataFrame({
            "stay_id": range(1, 31),
            "age": [58 + (i % 9) for i in range(30)],
            "sofa2": [i % 5 for i in range(30)],
            "death": [1 if i % 6 == 0 else 0 for i in range(30)],
        }),
    }
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    result = pipeline.replicate(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohorts=cohorts,
        target_outcome="death",
    )
    csv_path = Path(result["comparison_csv"])
    md_path = Path(result["comparison_md"])
    summary_csv_path = Path(result["summary_csv"])
    summary_md_path = Path(result["summary_md"])
    validation_report_path = Path(result["validation_report"])
    assert csv_path.exists()
    assert md_path.exists()
    assert summary_csv_path.exists()
    assert summary_md_path.exists()
    assert validation_report_path.exists()
    df = pd.read_csv(csv_path)
    assert set(df["database"]) == {"miiv", "eicu"}
    summary_df = pd.read_csv(summary_csv_path)
    assert set(summary_df["database"]) == {"miiv", "eicu"}


def test_plan_contract_does_not_relabel_covariate_as_primary_bias_audit(ra):
    from easyicu.research_agent import pipeline as pipeline_mod

    ctx = ra.build_research_context(
        research_question=(
            "Is biomarker X associated with ICU mortality after adjustment "
            "for age, sex, MAP, and vasopressor exposure?"
        ),
        cohort=pd.DataFrame({
            "stay_id": [1, 2, 3, 4],
            "biomarker_x": [1.0, 2.0, 3.0, 4.0],
            "map_min_24h": [70, 65, 60, 55],
            "vaso_any_24h": [0, 0, 1, 1],
            "death": [0, 0, 1, 1],
        }),
        cohort_name="c",
        database="miiv",
        target_outcome="death",
    )
    plan = ra.schema.AnalysisPlan(
        research_question=ctx.research_question,
        steps=[
            ra.schema.AnalysisStep(
                step_id="03_biomarker_mortality_association",
                intent=(
                    "Model biomarker_x and death with age, sex, MAP, "
                    "and vasopressor exposure as covariates."
                ),
                inputs=["biomarker_x", "death", "vaso_any_24h"],
                expected_outputs=["table:primary_association", "statistic:primary_or"],
                method="logistic_regression",
            )
        ],
    )

    revised, findings = pipeline_mod._enforce_advanced_plan_contract(
        plan=plan,
        context=ctx,
    )

    assert not findings
    assert revised.steps[0].step_id == "03_biomarker_mortality_association"


def test_pipeline_auto_enables_llm_concept_audit_for_non_mock_llm(ra, tmp_path: Path):
    class _TextLLM:
        name = "real-ish"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            return '{"findings":[]}'

    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=_TextLLM())
    assert pipeline._enable_llm_concept_audit is True


def test_pipeline_keeps_llm_concept_audit_off_for_mock_default(ra, tmp_path: Path):
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    assert pipeline._enable_llm_concept_audit is False


def test_pipeline_probe_can_trigger_replanning(ra, tmp_path: Path):
    class ReplanningLLM:
        name = "replanning-llm"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            upper = user.upper()
            if "REVISE THE ICU-AWARE RESEARCH PLAN" in upper:
                return json.dumps({
                    "research_question": "Audit then model mortality.",
                    "steps": [
                        {
                            "step_id": "03_missingness_audit",
                            "intent": "Audit missingness before modelling.",
                            "inputs": ["lact", "death"],
                            "expected_outputs": ["table:missingness"],
                            "method": "missingness_audit",
                            "icu_rule_refs": ["aggregation_rule_for"],
                        },
                        {
                            "step_id": "04_primary_association",
                            "intent": "Model lactate and mortality.",
                            "inputs": ["lact", "death"],
                            "expected_outputs": ["table:primary_association"],
                            "method": "regression",
                            "icu_rule_refs": ["aggregation_rule_for"],
                        },
                    ],
                    "rationale": "Probe revealed substantial missingness; audit first.",
                    "revision": 2,
                })
            if "ICU-AWARE RESEARCH PLAN" in upper:
                return json.dumps({
                    "research_question": "Audit then model mortality.",
                    "steps": [
                        {
                            "step_id": "04_primary_association",
                            "intent": "Model lactate and mortality.",
                            "inputs": ["lact", "death"],
                            "expected_outputs": ["table:primary_association"],
                            "method": "regression",
                            "icu_rule_refs": ["aggregation_rule_for"],
                        }
                    ],
                    "rationale": "Initial one-step plan.",
                    "revision": 1,
                })
            if "WRITE THE PYTHON CODE" in upper:
                if "03_missingness_audit" in upper:
                    return """
import json, os, pandas as pd
df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
pd.DataFrame({"variable": ["lact"], "fraction_missing": [float(df["lact"].isna().mean())]}).to_csv(os.path.join(out, "missingness.csv"), index=False)
summary = {"variable": "lact", "fraction_missing": float(df["lact"].isna().mean())}
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
"""
                return """
import json, os, pandas as pd
df = pd.read_parquet(os.environ["COHORT_PARQUET"]).dropna(subset=["lact", "death"])
out = os.environ["STEP_OUT_DIR"]
pd.DataFrame({"variable": ["lact"], "odds_ratio": [1.2]}).to_csv(os.path.join(out, "primary_association.csv"), index=False)
summary = {"predictor": "lact", "primary_or": 1.2}
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
"""
            if "INTERPRET THE RESULTS" in upper:
                return "See {evidence:primary_association}."
            if "MANUSCRIPT SCAFFOLD" in upper:
                return "# Title\n\n## Results\n\nSee {evidence:primary_association}.\n\n(left to the human author)"
            return "{}"

    cohort = pd.DataFrame({
        "stay_id": range(1, 41),
        "lact": [None if i % 3 == 0 else 1.0 + (i % 5) for i in range(40)],
        "death": [1 if i % 7 == 0 else 0 for i in range(40)],
    })
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=ReplanningLLM(),
        enable_literature=False,
        enable_probe_step=True,
        enable_replanning=True,
    )
    result = pipeline.run(
        question="Audit then model mortality.",
        cohort=cohort,
        cohort_name="replan_case",
        database="synthetic",
        target_outcome="death",
    )
    run_dir = Path(result.workdir)
    partial = json.loads((run_dir / "manifest_partial.json").read_text(encoding="utf-8"))
    step_ids = [rec["step_id"] for rec in partial["per_step_records"]]
    assert "00_probe" in step_ids
    assert "03_missingness_audit" in step_ids
    assert (run_dir / "analysis_plan_revision_2.json").exists()


def test_deterministic_runner_repair_patches_statsmodels_dtype_failure(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
import numpy as np
import pandas as pd
import statsmodels.api as sm
X = pd.DataFrame({"age": ["50", "60"], "sex_M": [True, False]})
y = pd.Series([0, 1])
res = sm.Logit(y, X).fit(disp=0)
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="Pandas data cast to numpy dtype of object",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "dtype_coerce_v1"
    assert "_easyicu_runner_repair_v1" in patched
    assert "sm.Logit(*_easyicu_runner_repair_v1(X, y))" in patched
    namespace = {}
    exec(patched, namespace)
    result = namespace["res"]
    assert len(result.params) == 2


def test_deterministic_runner_repair_patches_arbitrary_statsmodels_assignment(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
import numpy as np
import pandas as pd
import statsmodels.api as sm
cc_df = pd.DataFrame({"death": [0, 1], "lactate": ["1.0", "2.0"], "sex_M": [True, False]})
cc_model = sm.Logit(cc_df["death"], sm.add_constant(cc_df[["lactate", "sex_M"]]))
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="Pandas data cast to numpy dtype of object",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "dtype_coerce_v1"
    assert "cc_model = sm.Logit(*_easyicu_runner_repair_v1(" in patched


def test_deterministic_runner_repair_reapplies_dtype_after_coder_rewrite(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
import numpy as np
import pandas as pd
import statsmodels.api as sm

def _fit_logistic(X, y):
    X = X.apply(pd.to_numeric, errors="coerce")
    X = sm.add_constant(X, has_constant="add")
    y = pd.to_numeric(y, errors="coerce")
    data = pd.concat([y, X], axis=1).dropna()
    y_clean = data[y.name]
    X_clean = data.drop(columns=[y.name])
    model = sm.Logit(y_clean, X_clean)
    return model.fit(disp=0)

df = pd.DataFrame({
    "death": [0, 1, 0, 1, 0, 1],
    "age": ["50", "60", "55", "65", "45", "70"],
    "sex_M": [True, False, True, False, True, False],
})
result = _fit_logistic(df[["age", "sex_M"]], df["death"])
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="Pandas data cast to numpy dtype of object",
        previous_repair="dtype_coerce_v1",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "dtype_coerce_v1"
    assert "model = sm.Logit(*_easyicu_runner_repair_v1(X_clean, y_clean))" in patched
    assert "_easyicu_runner_repair_v1" in patched


def test_deterministic_runner_repair_adds_missing_os_import(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
value = os.environ["COHORT_PARQUET"]
print(value)
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="NameError: name 'os' is not defined",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "missing_os_import_v1"
    assert patched.startswith("import os\n")


def test_deterministic_runner_repair_strips_python_prefix(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = "pythonimport os\nvalue = os.environ['COHORT_PARQUET']\n"
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="SyntaxError: invalid syntax",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "strip_python_prefix_v1"
    assert patched.startswith("import os\n")


def test_deterministic_runner_repair_replaces_unclosed_table_one(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
import pandas as pd
df = pd.read_parquet(cohort_path)
df.to_csv("table_one.csv")
summary = {
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="SyntaxError: '{' was never closed",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "table_one_descriptive_repair_v1"
    assert "table_one.csv" in patched
    assert "step_summary.json" in patched
    assert 'os.environ["COHORT_PARQUET"]' in patched


def test_deterministic_runner_repair_updates_proportion_confint_nobs(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
from statsmodels.stats.proportion import proportion_confint
ci_lower, ci_upper = proportion_confint(count=events, n=len(df), method="wilson")
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="TypeError: proportion_confint() got an unexpected keyword argument 'n'",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "proportion_confint_nobs_keyword_v1"
    assert "nobs=len(df)" in patched


def test_deterministic_runner_repair_flattens_matplotlib_xerr(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
xerr_lower = np.array([or_estimate - or_lower])
xerr_upper = np.array([or_upper - or_estimate])
ax.errorbar([or_estimate], [0], xerr=np.array([[xerr_lower], [xerr_upper]]), fmt='o')
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log=(
            "ValueError: 'xerr' (shape: (2, 1, 1)) must be a scalar "
            "or a 1D or (2, n) array-like whose shape matches 'x'"
        ),
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "matplotlib_errorbar_xerr_shape_v1"
    assert "xerr=np.vstack([np.ravel(xerr_lower), np.ravel(xerr_upper)])" in patched


def test_deterministic_runner_repair_downgrades_bad_publication_contract(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
# Create figure contract with panels
figure_contract = make_figure_contract(
    figure_id="robustness_analysis",
    core_claim="Comparison of missing-data strategies"
)

# Add panel to the contract
figure_contract["panels"].append({
    "panel_id": "forest_plot",
    "title": "Odds Ratio by Missing Data Strategy",
    "role": "main",
    "claim": "Lactate odds ratio estimates are stable",
    "evidence_ids": ["robustness_summary"]
})

# Create main panel (forest plot of odds ratios)
fig, ax = plt.subplots()
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="ValueError: panels are required",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "publication_contract_optional_v1"
    assert "figure_contract = None" in patched
    assert 'figure_contract["panels"]' not in patched
    assert "# Create main panel" in patched


def test_deterministic_runner_repair_downgrades_bad_publication_contract_without_comment(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
figure_contract = make_figure_contract(
    figure_id="fig_trajectory_clustering",
    core_claim="Clusters show distinct mortality outcomes"
)

panel_a_data = cluster_means.reset_index()
figure_contract["panels"].append({
    "panel_id": "A",
    "title": "Cluster Profiles",
    "role": "profile_plot",
    "claim": "Cluster profiles differ",
    "evidence_ids": ["cluster_profile_data"]
})

fig, ax = plt.subplots()
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="ValueError: panels are required",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "publication_contract_optional_v1"
    assert "figure_contract = None" in patched
    assert 'figure_contract["panels"]' not in patched
    assert "panel_a_data = cluster_means.reset_index()" in patched
    assert "fig, ax = plt.subplots()" in patched


def test_promote_sibling_figure_exports_normalizes_outputs_stem(ra, tmp_path: Path):
    from easyicu.research_agent.pipeline import _promote_sibling_figure_exports

    step_dir = tmp_path / "steps" / "03_figure"
    out_dir = step_dir / "outputs"
    out_dir.mkdir(parents=True)
    (out_dir / "step_summary.json").write_text(
        json.dumps({"artifact_type": "figure"}), encoding="utf-8"
    )
    (step_dir / "outputs.png").write_bytes(b"png")
    (step_dir / "outputs.svg").write_text("<svg/>", encoding="utf-8")
    (step_dir / "outputs.figure_contract.json").write_text(
        json.dumps({"figure_id": "sofa_mortality_by_stratum"}),
        encoding="utf-8",
    )

    name = _promote_sibling_figure_exports(out_dir=out_dir)

    assert name == "sibling_figure_exports_promote_v1"
    assert (out_dir / "publication_figure.png").exists()
    assert (out_dir / "publication_figure.svg").exists()
    assert (out_dir / "publication_figure.figure_contract.json").exists()
    summary = json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))
    assert summary["figure_path"] == "publication_figure.png"
    assert sorted(summary["figure_files"]) == [
        "publication_figure.png",
        "publication_figure.svg",
    ]
    assert summary["publication_figure_rescue"]["mode"] == "sibling_outputs_stem"


def test_deterministic_runner_repair_restores_shadowed_json_module(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
import json

def json(value):
    return value

with open("step_summary.json", "w") as f:
    json.dump({"ok": True}, f, indent=2)
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="AttributeError: 'function' object has no attribute 'dump'",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "restore_shadowed_json_module_v1"
    assert "json = importlib.import_module('json')" in patched
    assert "    import importlib\n    json = importlib.import_module('json')\n    json.dump(" in patched


def test_deterministic_runner_repair_dedupes_outcome_before_unique(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
primary_predictor = "vaso_any_24h"
outcome = "death"
covariates = ["age", "sex"]
required_cols = [primary_predictor, outcome] + covariates
model_df = df[required_cols + [outcome]].copy()
if not set(model_df[outcome].unique()).issubset({0, 1}):
    raise ValueError("Outcome variable must be binary")
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="AttributeError: 'DataFrame' object has no attribute 'unique'",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "dedupe_required_cols_outcome_v1"
    assert "list(dict.fromkeys(required_cols + [outcome]))" in patched


def test_deterministic_runner_repair_encodes_sex_before_robustness_nan_check(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
covariates = ['sex', 'map_min_24h']
if len(cc_df) > 0:
    y_cc = cc_df[outcome_var]
    X_cc = cc_df[[predictor_var] + covariates]
    X_cc = X_cc.apply(pd.to_numeric, errors='coerce')
    y_cc = y_cc.astype(float)
    if not np.isfinite(X_cc.to_numpy()).all() or not np.isfinite(y_cc.to_numpy()).all():
        results['complete_case']['error'] = "exog contains inf or nans"
if len(mi_df) > 0:
    y_mi = mi_df[outcome_var]
    X_mi = mi_df[[predictor_var, 'lactate_missing_24h'] + covariates]
    X_mi = X_mi.apply(pd.to_numeric, errors='coerce')
    y_mi = y_mi.astype(float)
if len(rv_df) > 0:
    y_rv = rv_df[outcome_var]
    X_rv = rv_df[covariates]
    X_rv = X_rv.apply(pd.to_numeric, errors='coerce')
    y_rv = y_rv.astype(float)
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="Complete-case model failed: exog contains inf or nans",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "robustness_encode_sex_before_numeric_checks_v1"
    assert "if 'sex' in X_cc.columns:" in patched
    assert "valid_cc_idx = X_cc.dropna().index.intersection(y_cc.dropna().index)" in patched
    assert "    if 'sex' in X_mi.columns:" in patched


def test_deterministic_summary_repair_adds_complete_case_or_fallback(ra):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    code = """
predictor_var = 'lactate_max_24h'
step_summary = {
    'statistic:primary_or': None,
    'statistic:complete_case_n': 450,
}
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "statistic:primary_or": None,
            "statistic:complete_case_n": 450,
            "note": "Complete-case model failed: exog contains inf or nans",
        },
        previous_repair="sex_numeric_coercion_before_dropna_v1",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "robustness_complete_case_or_fallback_v1"
    assert "_easyicu_complete_case_or_fallback_v1" in patched
    compile(patched, "<patched>", "exec")


def test_deterministic_runner_repair_replaces_hallucinated_figure_utils_import(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
from easyicu.research_output.figure_utils import make_figure_contract, save_publication_figure

contract = make_figure_contract(
    figure_id="cluster_profiles",
    core_claim="Clusters have distinct physiology.",
)
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="ModuleNotFoundError: No module named 'easyicu.research_output'",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "replace_hallucinated_figure_utils_import_v1"
    assert "easyicu.research_agent.publication_figures" in patched
    assert "easyicu.research_output.figure_utils" not in patched


def test_deterministic_runner_repair_filters_x_cols_after_dummy_encoding(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
model_df = pd.get_dummies(model_df, columns=["sex"], drop_first=True)
x_cols = ["lactate_max_24h", "age", "sex", "map_min_24h"]
dummy_cols = [col for col in model_df.columns if col.startswith("sex_")]
x_cols.extend(dummy_cols)
X = model_df[x_cols].copy()
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="KeyError: \"['sex'] not in index\"",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "filter_x_cols_after_dummy_encoding_v1"
    assert "x_cols = [col for col in x_cols if col in model_df.columns]" in patched


def test_deterministic_runner_repair_filters_x_cols_before_dropna_after_dummy_encoding(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
sex_dummies = pd.get_dummies(model_df["sex"], prefix="sex", drop_first=True)
model_df = pd.concat([model_df.drop("sex", axis=1), sex_dummies], axis=1)
x_cols = [primary_predictor] + covariates.copy()
model_df = model_df.dropna(subset=x_cols + [outcome])
X = model_df[x_cols].astype(float)
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="KeyError: ['sex']",
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "filter_x_cols_before_dropna_after_dummy_encoding_v1"
    assert "x_cols = [col for col in x_cols if col in model_df.columns]" in patched
    assert patched.index("x_cols = [col for col in x_cols if col in model_df.columns]") < patched.index("dropna")


def test_deterministic_runner_repair_filters_generic_dropna_after_dummy_encoding(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
model_df = pd.get_dummies(model_df, columns=['sex'], drop_first=True)
x_cols = [primary_predictor] + covariates
dummy_cols = [col for col in model_df.columns if col.startswith('sex_')]
x_cols.extend(dummy_cols)
model_df = model_df.apply(pd.to_numeric, errors="coerce")
model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[outcome_col] + x_cols)
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="KeyError: ['sex']",
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "filter_x_cols_before_dropna_after_dummy_encoding_v1"
    assert "x_cols = [col for col in x_cols if col in model_df.columns]" in patched
    assert patched.index("x_cols = [col for col in x_cols if col in model_df.columns]") < patched.index("dropna")


def test_deterministic_runner_repair_uses_df_for_missing_indicator_source(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
df = pd.read_parquet(cohort_path)
model_df = df[all_vars].copy()
creat_missing = [col for col in creat_cols if col in df.columns]
model_df['creat_missing'] = model_df[creat_missing].isnull().any(axis=1).astype(int)
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log='KeyError: "None of [Index([\'creat_max_24h\', \'creat_median_24h\'], dtype=\'object\')] are in the [columns]"',
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "missing_indicator_source_df_v1"
    assert "model_df['creat_missing'] = df[creat_missing].isnull().any(axis=1).astype(int)" in patched


def test_deterministic_runner_repair_restores_outcome_in_all_vars_subset(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
outcome_col = 'death'
primary_predictor = 'kdigo_stage_max_24h'
covariates = ['age', 'sex', 'sofa2_renal_max_24h', 'vaso_any_24h']
all_vars = [primary_predictor] + covariates
cc_df = df[all_vars].dropna()
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log='KeyError: "[\'death\'] not in index"',
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "include_outcome_in_all_vars_v1"
    assert "all_vars = [outcome_col, primary_predictor] + covariates" in patched


def test_deterministic_runner_repair_restores_predictor_and_sex_in_robustness_script(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
predictor_col = lactate_col
covariates = ["age", "sex", "sofa2_max_24h"]
def fit_logistic_model(X, y):
    return None
model_df = df[all_vars].copy()
cc_X = cc_df[covariates]
mi_X = mi_df[covariates + ['lactate_missing']]
rv_X = rv_df[covariates]
ax.errorbar(x_pos, ors, yerr=[yerr_lower, yerr_upper], fmt='o')
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'",
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "robustness_predictor_design_and_plot_v1"
    assert "model_df['sex'] = model_df['sex'].astype(str).str.lower().isin(['m', 'male']).astype(float)" in patched
    assert "cc_X = cc_df[[predictor_col] + covariates]" in patched
    assert "mi_X = mi_df[[predictor_col] + covariates + ['lactate_missing']]" in patched
    assert "rv_X = rv_df[[predictor_col] + covariates]" in patched
    assert "plot_rows = [" in patched


def test_deterministic_runner_repair_stabilizes_predictor_col_robustness_template(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
outcome_col = "death"
predictor_col = lactate_col
covariates = ["age", "sex", "los_icu", "sofa2_max_24h", "map_min_24h", "vaso_any_24h", "bili_max_24h", "bili_n_24h", "creat_max_24h"]
model_df = df[[outcome_col, predictor_col] + covariates].copy()
model_df["lactate_missing"] = model_df[predictor_col].isnull().astype(int)
complete_case_df = model_df.dropna(subset=[predictor_col])
missing_indicator_df = model_df.copy()
reduced_variable_df = model_df.drop(columns=[predictor_col]).copy()
X_cc = sm.add_constant(complete_case_df[covariates], has_constant="add")
X_mi = sm.add_constant(missing_indicator_df[covariates + ["lactate_missing"]], has_constant="add")
X_rv = sm.add_constant(reduced_variable_df[covariates], has_constant="add")
X_rv = X_rv.drop(columns=[predictor_col])
lci = [row_cc["or_lower"], row_mi["or_lower"], row_rv["or_lower"]]
uci = [row_cc["or_upper"], row_mi["or_upper"], row_rv["or_upper"]]
ax.errorbar(x_pos, ors, yerr=[yerr_lower, yerr_upper], fmt='o')
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'",
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "robustness_predictor_design_and_plot_v1"
    assert "model_df['sex'] = model_df['sex'].astype(str).str.lower().isin(['m', 'male']).astype(float)" in patched
    assert "reduced_covariates = [c for c in covariates if model_df[c].isna().mean() <= 0.2]" in patched
    assert "complete_case_df = model_df.dropna(subset=[outcome_col, predictor_col] + covariates)" in patched
    assert "missing_indicator_df[predictor_col] = missing_indicator_df[predictor_col].fillna(0)" in patched
    assert "missing_indicator_df = missing_indicator_df.dropna(subset=[outcome_col] + covariates)" in patched
    assert "reduced_variable_df = model_df[[outcome_col, predictor_col] + reduced_covariates].dropna().copy()" in patched
    assert 'X_cc = sm.add_constant(complete_case_df[[predictor_col] + covariates], has_constant="add")' in patched
    assert 'X_mi = sm.add_constant(missing_indicator_df[[predictor_col] + covariates + ["lactate_missing"]], has_constant="add")' in patched
    assert 'X_rv = sm.add_constant(reduced_variable_df[[predictor_col] + reduced_covariates], has_constant="add")' in patched
    assert "plot_rows = [" in patched
    assert "if len(x_pos):" in patched


def test_deterministic_runner_repair_preserves_indentation_for_robustness_patch(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
def main():
    outcome_col = "death"
    predictor_col = lactate_col
    covariates = ["age", "sex"]
    model_df = df[[outcome_col, predictor_col] + covariates].copy()
    model_df["lactate_missing"] = model_df[predictor_col].isnull().astype(int)
    complete_case_df = model_df.dropna(subset=[predictor_col])
    missing_indicator_df = model_df.copy()
    reduced_variable_df = model_df.drop(columns=[predictor_col]).copy()
    X_cc = sm.add_constant(complete_case_df[covariates], has_constant="add")
    X_mi = sm.add_constant(missing_indicator_df[covariates + ["lactate_missing"]], has_constant="add")
    X_rv = sm.add_constant(reduced_variable_df[covariates], has_constant="add")
    X_rv = X_rv.drop(columns=[predictor_col])
    ax.errorbar(x_pos, ors, yerr=[yerr_lower, yerr_upper], fmt='o')
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'",
    )

    assert repaired is not None
    _name, patched = repaired
    assert "\n    if 'sex' in model_df.columns:" in patched
    assert "\n    reduced_covariates = [c for c in covariates if model_df[c].isna().mean() <= 0.2]" in patched


def test_deterministic_runner_repair_handles_undefined_primary_predictor_summary(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
summary = {
    "primary_predictor": primary_predictor if primary_predictor else None,
}
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="NameError: name 'primary_predictor' is not defined",
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "primary_predictor_safe_summary_lookup_v1"
    assert "locals().get('predictor_col')" in patched


def test_deterministic_runner_repair_replaces_broken_outcome_incidence(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
STEP_NAME = "outcome_incidence"
final_model = ols("logit(" + 
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="SyntaxError: invalid syntax. Perhaps you forgot a comma?",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "outcome_incidence_descriptive_repair_v1"
    assert "outcome_incidence.csv" in patched
    assert "outcome_rate.json" in patched
    assert "proportion_confint" in patched


def test_deterministic_runner_repair_rewrites_broken_prediction_split_script(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
from sklearn.model_selection import train_test_split
figure_contract = FigureContract(
    figure_id=1,
    panels=1,
    panels=['split'],
    figure_id=1,
)
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="SyntaxError: keyword argument repeated: figure_id",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "prediction_split_minimal_v1"
    assert "train_test_split(" in patched
    assert '"split_strategy": "stratified_random"' in patched


def test_deterministic_runner_repair_preserves_categorical_prediction_columns(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
for col in predictors:
    if col in data:
        data[col] = pd.to_numeric(data[col], errors="coerce")
categorical_features = ["sex"]
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log=(
            "ValueError: Found array with 0 feature(s) (shape=(800, 0)) while a minimum "
            "of 1 is required. OneHotEncoder failed after categorical branch was emptied."
        ),
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "prediction_preserve_categorical_before_ohe_v1"
    assert 'col not in ["sex"]' in patched
    assert 'data["sex"] = data["sex"].astype("string")' in patched


def test_deterministic_runner_repair_injects_logreg_imputation(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[:, 1]
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="ValueError: Input X contains NaN. LogisticRegression does not accept missing values encoded as NaN natively.",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "logreg_impute_v1"
    assert "_easyicu_logreg_impute_v1" in patched
    assert "X_test = _easyicu_logreg_impute_v1(X_test)" in patched


def test_deterministic_runner_repair_fixes_prediction_calibration_import(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
from sklearn.metrics import roc_auc_score, brier_score_loss, calibration_curve
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log=(
            "ImportError: cannot import name 'calibration_curve' from "
            "'sklearn.metrics'"
        ),
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "prediction_calibration_import_fix_v1"
    assert "from sklearn.metrics import roc_auc_score, brier_score_loss" in patched
    assert "from sklearn.calibration import calibration_curve" in patched


def test_deterministic_runner_repair_retries_prediction_calibration_import(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
from sklearn.metrics import roc_auc_score, brier_score_loss, calibration_curve
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log=(
            "ImportError: cannot import name 'calibration_curve' from "
            "'sklearn.metrics'"
        ),
        previous_repair="prediction_calibration_import_fix_v1",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "prediction_calibration_import_fix_v1"
    assert "from sklearn.calibration import calibration_curve" in patched


def test_deterministic_runner_repair_regularizes_singular_logit(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
import statsmodels.api as sm
model = sm.Logit(y, X)
result = model.fit(disp=0, method='newton')
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="numpy.linalg.LinAlgError: Singular matrix",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "logit_regularized_fit_v1"
    assert "_easyicu_safe_logit_fit_v1" in patched
    assert "result = _easyicu_safe_logit_fit_v1(model)" in patched


def test_deterministic_runner_repair_falls_back_to_sklearn_after_regularized_singular(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
required_cols = ['lactate_max_24h', 'death', 'age', 'sex', 'map_min_24h', 'vaso_any_24h']
model = sm.Logit(y, X)
result = _easyicu_safe_logit_fit_v1(model)
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="numpy.linalg.LinAlgError: Singular matrix",
        previous_repair="logit_regularized_fit_v1",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "shock_primary_assoc_sklearn_v1"
    assert "LogisticRegression(" in patched
    assert "logistic_regression_sklearn_bootstrap" in patched


def test_deterministic_runner_repair_promotes_publication_bundle_script(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
from easyicu.research_agent.publication_figures import make_figure_contract
pub_style = apply_publication_style()
save_publication_figure(figure_contract, fig)
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="NameError: name 'apply_publication_style' is not defined",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "publication_bundle_promote_script_v1"
    assert 'target_stem = "publication_figure"' in patched
    assert "publication_figure_rescue" in patched


def test_deterministic_runner_repair_swaps_csv_reader_for_parquet(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
import pandas as pd
df = pd.read_csv(cohort_path, encoding='utf-8')
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 7",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "cohort_csv_to_parquet_v1"
    assert "pd.read_parquet(cohort_path)" in patched


def test_deterministic_runner_repair_swaps_env_csv_reader_for_parquet(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
import os
import pandas as pd
cohort = pd.read_csv(os.environ['COHORT_PARQUET'])
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 7",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "cohort_csv_to_parquet_v1"
    assert "pd.read_parquet(os.environ['COHORT_PARQUET'])" in patched


def test_deterministic_runner_repair_falls_back_for_age_tertile_observed(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
import pandas as pd
df["age_tertile"] = pd.qcut(df["age"], q=3, labels=["1st", "2nd", "3rd"], duplicates='drop', observed=True)
summary = {"mortality_rate": df["death"].mean()}
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="TypeError: qcut() got an unexpected keyword argument 'observed'",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "age_stratified_mortality_dependency_free_v1"
    assert "age_tertile_mortality.csv" in patched
    compile(patched, "<patched>", "exec")


def test_deterministic_runner_repair_falls_back_for_norepi_optional_deps(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
import statsmodels.api as sm
predictor = "norepi_equiv_max_24h"
summary = {"primary_or": None}
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="ModuleNotFoundError: No module named 'statsmodels'",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "norepinephrine_dose_response_dependency_free_v1"
    assert "norepinephrine_quartile_mortality.csv" in patched
    assert "statistic:primary_or" in patched
    compile(patched, "<patched>", "exec")


def test_deterministic_runner_repair_sanitizes_numpy_json_keys(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
import json
summary = {np.int64(1): {"count": np.int64(2)}}
with open("step_summary.json", "w") as f:
    json.dump(summary, f)
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="TypeError: keys must be str, int, float, bool or None, not int64",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "json_dump_numpy_key_sanitizer_v1"
    assert "_easyicu_json_sanitize_v1" in patched
    compile(patched, "<patched>", "exec")


def test_step_contract_accepts_figure_file_dicts(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.schema.AnalysisStep(
        step_id="02_mortality_figure",
        intent="Create a publication-ready figure.",
        inputs=[],
        expected_outputs=["figure:mortality"],
        method="visualization",
    )
    findings = _step_contract_findings(
        step=step,
        step_summary={"figure_files": [{"path": "/tmp/mortality.png"}]},
    )
    assert not [f for f in findings if f.severity == "error"]


def test_deterministic_runner_repair_restores_primary_predictor_in_logit_design(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
import pandas as pd
import statsmodels.api as sm
df = pd.read_parquet(cohort_path)
model_df = df[['lactate_max_24h', 'map_min_24h', 'vaso_any_24h', 'age', 'sex', 'death']].copy()
model_df = pd.get_dummies(model_df, columns=['sex'], drop_first=True)
y = model_df['death'].astype(float)
X = model_df[['map_min_24h', 'vaso_any_24h', 'age'] + [col for col in model_df.columns if col.startswith('sex_')]].astype(float)
X = sm.add_constant(X, has_constant='add')
try:
    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=0)
    coef_table = result.conf_int()
    lactate_or = coef_table.loc['lactate_max_24h', 'or']
except Exception as e:
    print(f"Error fitting logistic regression: {e}")
step_summary = {"n_total_stays": int(n_total), "odds_ratio": lactate_or}
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log=(
            "Error fitting logistic regression: 'lactate_max_24h'\n"
            "NameError: name 'n_total' is not defined"
        ),
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "primary_predictor_omitted_from_design_v1"
    assert "X = model_df[['lactate_max_24h'," in patched
    assert "n_total = int(len(df))" in patched
    assert "lactate_or = None" in patched


def test_deterministic_runner_repair_restores_primary_predictor_with_indented_x_line(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
def main():
    model_df = df[['lactate_max_24h', 'map_min_24h', 'vaso_any_24h', 'age', 'sex', 'death']].copy()
    y = model_df['death'].astype(float)
    X = model_df[['map_min_24h', 'vaso_any_24h', 'age', 'sex']].copy()
    X = sm.add_constant(X, has_constant='add')
    coef_table = result.conf_int()
    lactate_or = coef_table.loc['lactate_max_24h', 'or']
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="Error fitting logistic regression: 'lactate_max_24h'",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "primary_predictor_omitted_from_design_v1"
    assert "X = model_df[['lactate_max_24h'," in patched


def test_deterministic_runner_repair_fixes_stringified_binary_outcome_key(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
table_one_data.append({"variable": "In-hospital Mortality", "type": "binary", "count": summary["outcomes"]["death"]["counts"][1],
                      "pct": summary["outcomes"]["death"]["pct"][1]})
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="KeyError: 1",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "table_one_binary_key_string_v1"
    assert '.get("1", summary["outcomes"]["death"]["counts"].get(1, 0))' in patched
    assert '.get("1", summary["outcomes"]["death"]["pct"].get(1, 0.0))' in patched


def test_step_contract_findings_flag_missing_primary_association_estimate(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="03_primary_association_model",
        intent="Estimate the adjusted association between lactate and mortality.",
        expected_outputs=["statistic:adjusted_or_ci"],
    )
    findings = _step_contract_findings(
        step=step,
        step_summary={
            "primary_predictor": "lactate_max_24h",
            "sample_size": 785,
            "estimate": None,
            "ci_lower": None,
            "ci_upper": None,
            "p_value": None,
            "skipped": "No valid lactate_max_24h data",
        },
    )
    assert findings
    assert findings[0].validator == "step_contract"
    assert findings[0].severity == "error"
    assert "primary association estimate" in findings[0].message


def test_step_contract_findings_accepts_nested_primary_association_estimate(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="03_primary_association_model",
        intent="Estimate the adjusted association between lactate and mortality.",
        expected_outputs=["statistic:adjusted_or_ci"],
    )
    findings = _step_contract_findings(
        step=step,
        step_summary={
            "primary_predictor": "lactate_max_24h",
            "statistic": {
                "adjusted_or_ci": {
                    "estimate": 1.21,
                    "ci_lower": 1.10,
                    "ci_upper": 1.33,
                }
            },
        },
    )
    assert findings == []


def test_step_contract_findings_accepts_nested_primary_association_or(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="03_association_model",
        intent="Estimate the adjusted association between lactate and mortality.",
        expected_outputs=["statistic:adjusted_or_ci"],
    )
    findings = _step_contract_findings(
        step=step,
        step_summary={
            "predictor": "lactate_max_24h",
            "statistic": {
                "adjusted_or_ci": {
                    "or": 1.21,
                    "ci_lower": 1.10,
                    "ci_upper": 1.33,
                }
            },
        },
    )
    assert findings == []


def test_step_contract_findings_accepts_predictor_named_or_key(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="03_association_model",
        intent="Fit lactate mortality model.",
        expected_outputs=["odds_ratio", "confidence_interval"],
    )

    findings = _step_contract_findings(
        step=step,
        step_summary={"lactate_max_24h_or": 1.21},
    )

    assert findings == []


def test_step_contract_findings_accepts_primary_association_estimate_key(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="02_lactate_map_vaso_mortality_association",
        intent="Estimate a lactate/MAP/vasopressor mortality association.",
        expected_outputs=["statistic:primary_association"],
    )

    findings = _step_contract_findings(
        step=step,
        step_summary={"primary_association_estimate": 1.42},
    )

    assert findings == []


def test_step_contract_findings_does_not_require_or_for_data_quality_association_table(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="01_sofa_zero_audit",
        intent="Audit SOFA-zero co-occurrence and missingness.",
        expected_outputs=[
            "table:sofa_zero_component_distribution",
            "table:sofa_zero_associations",
            "log:missingness_summary",
        ],
    )

    findings = _step_contract_findings(
        step=step,
        step_summary={
            "sofa_zero_count": 41,
            "guardrail_warning": True,
            "primary_association_estimate": None,
        },
    )

    assert findings == []


def test_step_contract_findings_does_not_treat_association_calibration_as_prediction(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="03_association_model",
        intent="Estimate adjusted vasopressor association with mortality.",
        expected_outputs=["statistic:adjusted_or_ci", "figure:association_model_calibration"],
    )

    findings = _step_contract_findings(
        step=step,
        step_summary={"statistic": {"adjusted_or": 1.68}},
    )

    assert findings == []


def test_render_writer_evidence_digest_prefers_machine_scalars(ra):
    from easyicu.research_agent.pipeline import _render_writer_evidence_digest

    digest = _render_writer_evidence_digest(
        [
            {
                "step_id": "03_primary_association_model",
                "status": "contract_failed",
                "step_summary": {
                    "primary_predictor": "lactate_max_24h",
                    "target_outcome": "death",
                    "sample_size": 785,
                    "estimate": None,
                    "ci_lower": None,
                    "ci_upper": None,
                    "skipped": "No valid lactate_max_24h data",
                    "missingness": {"lactate_max_24h": {"count": 335}},
                },
            }
        ]
    )
    assert "- 03_primary_association_model [contract_failed]" in digest
    assert '"sample_size": 785' in digest
    assert '"primary_predictor": "lactate_max_24h"' in digest
    assert '"skipped": "No valid lactate_max_24h data"' in digest
    assert "missingness" not in digest


def test_render_writer_evidence_digest_flattens_nested_statistics(ra):
    from easyicu.research_agent.pipeline import _render_writer_evidence_digest

    digest = _render_writer_evidence_digest(
        [
            {
                "step_id": "03_primary_association_model",
                "status": "ok",
                "step_summary": {
                    "primary_predictor": "lactate_max_24h",
                    "statistic": {
                        "adjusted_or_ci": {
                            "estimate": 1.216,
                            "ci_lower": 1.109,
                            "ci_upper": 1.333,
                            "p_value": 2.9e-05,
                        }
                    },
                },
            }
        ]
    )
    assert '"estimate": 1.216' in digest
    assert '"ci_lower": 1.109' in digest
    assert '"ci_upper": 1.333' in digest
    assert '"p_value": 2.9e-05' in digest


def test_step_contract_repair_guidance_flags_missing_primary_predictor_in_x(ra):
    from easyicu.research_agent.pipeline import _step_contract_repair_guidance

    step = ra.AnalysisStep(
        step_id="04_primary_association_model",
        intent="Estimate lactate association.",
        expected_outputs=["statistic:adjusted_or_ci"],
    )
    guidance = _step_contract_repair_guidance(
        step=step,
        step_summary={
            "predictor": "lactate_max_24h",
            "statistic": {"adjusted_or_ci": {"estimate": None}},
            "notes": "Model fitting failed: 'lactate_max_24h'",
        },
        code="""
X = model_df[['age', 'map_min_24h']].astype(float)
result = sm.Logit(y, X).fit()
coef = result.params['lactate_max_24h']
""",
    )
    assert "lactate_max_24h" in guidance
    assert "X.columns" in guidance


def test_step_contract_findings_accepts_cv_prediction_metrics(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="03_model_training",
        intent="Train and evaluate mortality prediction model.",
        expected_outputs=["statistic:cv_auroc_mean", "statistic:brier_score"],
    )

    findings = _step_contract_findings(
        step=step,
        step_summary={"statistic:cv_auroc_mean": 0.71, "statistic:brier_score": 0.09},
    )

    assert findings == []


def test_step_contract_findings_accepts_suffixed_prediction_metrics(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="03_model_evaluation",
        intent="Evaluate mortality prediction robustness.",
        expected_outputs=["statistic:auroc", "statistic:brier_score"],
    )

    findings = _step_contract_findings(
        step=step,
        step_summary={
            "statistic:auroc_complete_case": 0.61,
            "statistic:auroc_missing_indicator": 0.73,
            "statistic:brier_score_complete_case": 0.11,
            "statistic:calibration_slope_missing_indicator": 0.94,
        },
    )

    assert findings == []


def test_step_contract_findings_requires_metrics_for_prediction_training_intent(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="03_model_training",
        intent="Train a mortality prediction model with 5-fold cross-validation.",
        expected_outputs=["model:trained_prediction_model"],
    )

    findings = _step_contract_findings(
        step=step,
        step_summary={
            "cv_auroc_mean": None,
            "brier_score": None,
            "error": "could not convert string to float: 'M'",
        },
    )

    assert any("AUROC" in finding.message for finding in findings)
    assert any("Brier" in finding.message for finding in findings)


def test_step_contract_findings_accepts_prefixed_clustering_metrics(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="01_trajectory_clustering",
        intent="Cluster shock physiology.",
        expected_outputs=[
            "statistic:silhouette_score",
            "statistic:cluster_count",
            "table:cluster_characteristics",
            "log:clustering_process",
        ],
    )

    findings = _step_contract_findings(
        step=step,
        step_summary={
            "statistic:silhouette_score": 0.46,
            "statistic:cluster_count": 2,
        },
    )

    assert findings == []


def test_step_contract_findings_accepts_complete_case_lactate_or_alias(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="03_complete_case_robustness",
        intent="Fit robustness logistic regression models.",
        expected_outputs=[
            "statistic:primary_or",
            "statistic:complete_case_n",
            "table:robustness_summary",
        ],
    )

    findings = _step_contract_findings(
        step=step,
        step_summary={
            "statistic:lactate_or_complete_case": 1.14,
            "table:complete_case_robustness_summary": [
                {"strategy": "Complete-case", "n": 450},
            ],
        },
    )

    assert findings == []


def test_step_contract_findings_requires_figure_path_for_figure_output(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="03_association_model",
        intent="Estimate lactate association with a publication-ready figure.",
        expected_outputs=["table:adjusted_association", "figure:lactate_mortality_plot"],
    )

    findings = _step_contract_findings(
        step=step,
        step_summary={
            "estimate": 1.21,
            "ci_lower": 1.10,
            "ci_upper": 1.33,
        },
    )

    assert any("figure artifact" in finding.message for finding in findings)


def test_step_contract_findings_accepts_figure_path_for_figure_output(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="03_association_model",
        intent="Estimate lactate association with a publication-ready figure.",
        expected_outputs=["table:adjusted_association", "figure:lactate_mortality_plot"],
    )

    findings = _step_contract_findings(
        step=step,
        step_summary={
            "estimate": 1.21,
            "ci_lower": 1.10,
            "ci_upper": 1.33,
            "figure_path": "outputs/lactate_mortality_plot.png",
        },
    )

    assert findings == []


def test_step_contract_findings_accepts_figure_files_list_for_figure_output(ra):
    """Regression: list-valued ``figure_files`` must satisfy the figure contract.

    The coder prompt explicitly recommends ``figure_files`` as one of the
    acceptable keys, but ``_flatten_scalar_dict`` drops lists, so the validator
    used to ignore well-formed list-valued figure manifests and falsely flag
    ``contract_failed`` even when the agent had produced PNG/SVG outputs.
    """
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="01_table_one",
        intent="Build a publication-ready Table 1 figure for descriptive output.",
        expected_outputs=["table:table_one", "figure:table_one_summary"],
    )

    findings = _step_contract_findings(
        step=step,
        step_summary={
            "n_rows": 1000,
            "mortality_rate": 9.6,
            "figure_files": [
                "publication_ready_figure.png",
                "publication_ready_figure.svg",
            ],
        },
    )

    assert findings == [], (
        "list-valued figure_files should satisfy the figure contract; "
        f"unexpected findings: {findings}"
    )


def test_step_contract_findings_rejects_empty_figure_files_list(ra):
    """An empty figure list must still trigger the missing-figure contract."""
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="01_table_one",
        intent="Build a publication-ready Table 1 figure for descriptive output.",
        expected_outputs=["figure:table_one_summary"],
    )

    findings = _step_contract_findings(
        step=step,
        step_summary={
            "figure_files": [],
            "non_figure_strings": ["table_one.csv"],
        },
    )

    assert any("figure artifact" in finding.message for finding in findings)


def test_split_table_and_figure_outputs_in_plan_splits_mixed_step(ra):
    """A single step declaring both ``table:`` and ``figure:`` outputs is split
    into a table-only step and an appended figure-only follow-up so that the
    coder can target each artefact independently.
    """
    from easyicu.research_agent.pipeline import (
        _split_table_and_figure_outputs_in_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="Build Table 1 and a figure.",
        steps=[
            AnalysisStep(
                step_id="01_table_one",
                intent="Compute Table 1 plus its visual.",
                inputs=["age", "sex"],
                expected_outputs=["table:table_one", "figure:table_one_visual"],
                method="descriptive",
            ),
        ],
    )

    revised, findings = _split_table_and_figure_outputs_in_plan(plan=plan)

    assert [s.step_id for s in revised.steps] == [
        "01_table_one",
        "01_table_one_figure",
    ]
    table_step = revised.steps[0]
    figure_step = revised.steps[1]
    assert table_step.expected_outputs == ["table:table_one"]
    assert figure_step.expected_outputs == ["figure:table_one_visual"]
    assert findings and findings[0].severity == "warning"
    assert "01_table_one_figure" in findings[0].message


def test_split_table_and_figure_outputs_in_plan_no_op_when_pure_steps(ra):
    """Steps that are figure-only or table-only are left untouched."""
    from easyicu.research_agent.pipeline import (
        _split_table_and_figure_outputs_in_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="Pure plan.",
        steps=[
            AnalysisStep(
                step_id="01_table",
                intent="Just a table.",
                expected_outputs=["table:t"],
            ),
            AnalysisStep(
                step_id="02_figure",
                intent="Just a figure.",
                expected_outputs=["figure:f"],
            ),
        ],
    )
    revised, findings = _split_table_and_figure_outputs_in_plan(plan=plan)
    assert revised is plan
    assert findings == []


def test_split_table_and_figure_outputs_in_plan_no_op_for_advanced_self_contained_step(ra):
    from easyicu.research_agent.pipeline import (
        _split_table_and_figure_outputs_in_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="Fit robustness models and draw a figure.",
        steps=[
            AnalysisStep(
                step_id="03_complete_case_robustness",
                intent="Fit robustness models and write the robustness figure.",
                expected_outputs=[
                    "statistic:primary_or",
                    "table:robustness_summary",
                    "figure:robustness_plot",
                ],
                method="association_robustness",
            ),
        ],
    )

    revised, findings = _split_table_and_figure_outputs_in_plan(plan=plan)

    assert revised is plan
    assert findings == []


def test_ensure_publication_figure_step_appends_when_missing(ra):
    """Naive arms occasionally emit single-step plans without a figure even
    though the research question asks for a publication-ready figure. The
    plan-contract guard appends a fallback step in that case.
    """
    from easyicu.research_agent.pipeline import (
        _ensure_publication_figure_step_in_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    class _Ctx:
        research_question = (
            "Build a Table 1 of demographics, severity, missingness, and a "
            "concise publication-ready figure."
        )

    plan = AnalysisPlan(
        research_question=_Ctx.research_question,
        steps=[
            AnalysisStep(
                step_id="01_table_one",
                intent="Compute Table 1.",
                expected_outputs=["table:table_one", "statistic:n_rows"],
            ),
        ],
    )

    revised, findings = _ensure_publication_figure_step_in_plan(
        plan=plan,
        context=_Ctx(),
    )

    assert len(revised.steps) == 2
    appended = revised.steps[-1]
    assert appended.step_id.endswith("publication_figure_fallback")
    assert any("figure" in out for out in appended.expected_outputs)
    assert findings and findings[0].severity == "warning"
    assert "fallback figure step" in findings[0].message


def test_ensure_publication_figure_step_no_op_when_figure_exists(ra):
    """If the plan already produces a figure, the guard is a no-op."""
    from easyicu.research_agent.pipeline import (
        _ensure_publication_figure_step_in_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    class _Ctx:
        research_question = "Produce a publication-ready figure of strata."

    plan = AnalysisPlan(
        research_question=_Ctx.research_question,
        steps=[
            AnalysisStep(
                step_id="01_strata",
                intent="Compute strata.",
                expected_outputs=["table:strata"],
            ),
            AnalysisStep(
                step_id="02_figure",
                intent="Render the figure.",
                expected_outputs=["figure:strata_overview"],
            ),
        ],
    )
    revised, findings = _ensure_publication_figure_step_in_plan(
        plan=plan,
        context=_Ctx(),
    )
    assert revised is plan or len(revised.steps) == 2
    assert findings == []


def test_ensure_publication_figure_step_no_op_when_question_does_not_request_figure(ra):
    """If the question doesn't ask for a figure, the guard stays out of the way."""
    from easyicu.research_agent.pipeline import (
        _ensure_publication_figure_step_in_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    class _Ctx:
        research_question = "List demographics counts for the cohort."

    plan = AnalysisPlan(
        research_question=_Ctx.research_question,
        steps=[
            AnalysisStep(
                step_id="01_counts",
                intent="Compute counts.",
                expected_outputs=["table:counts"],
            ),
        ],
    )
    revised, findings = _ensure_publication_figure_step_in_plan(
        plan=plan,
        context=_Ctx(),
    )
    assert len(revised.steps) == 1
    assert findings == []


def test_deterministic_runner_repair_injects_undefined_helper_stub(ra):
    """Regression: NameError for an undefined helper triggers stub injection."""
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = (
        "import json\n"
        "data = {\"a\": 1}\n"
        "with open('out.json', 'w') as f:\n"
        "    json.dump(data, f, default=to_json_serializable)\n"
    )
    run_log = (
        "Traceback (most recent call last):\n"
        "  File \"analysis.py\", line 4, in <module>\n"
        "NameError: name 'to_json_serializable' is not defined\n"
    )

    result = _deterministic_runner_repair(code=code, run_log=run_log)
    assert result is not None
    repair_name, repaired = result
    assert repair_name == "undefined_helper_stub_to_json_serializable_v1"
    assert "def to_json_serializable" in repaired
    # The original code is preserved beneath the stub.
    assert "json.dump(data, f, default=to_json_serializable)" in repaired
    # Idempotent: feeding the repaired code back produces no further repair.
    second = _deterministic_runner_repair(
        code=repaired, run_log=run_log, previous_repair=repair_name
    )
    assert second is None or second[0] != repair_name


def test_deterministic_runner_repair_skips_helper_already_defined(ra):
    """If the helper IS defined, the stub injection must not fire."""
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = (
        "def to_json_serializable(value):\n"
        "    return value\n"
        "\n"
        "import json\n"
        "json.dump({}, open('out.json','w'), default=to_json_serializable)\n"
    )
    run_log = "NameError: name 'to_json_serializable' is not defined"
    result = _deterministic_runner_repair(code=code, run_log=run_log)
    # NameError is not consistent with the code (helper IS defined), so
    # the runner should not inject another stub.
    assert result is None or result[0] != "undefined_helper_stub_to_json_serializable_v1"


def test_demote_unresolved_evidence_placeholders_rewrites_to_html_comment(ra):
    """Regression: ``[evidence missing: …]`` markers must be demoted to
    HTML comments for clean rendering while the source still preserves trace.
    """
    from easyicu.research_agent.pipeline import (
        _demote_unresolved_evidence_placeholders,
    )

    bound = (
        "# Results\n\n"
        "The mortality rate was [evidence missing: mortality_rate] across "
        "the cohort and the SOFA-2 stratum mortality table is "
        "[stratum_table](evidence/x.csv).\n"
        "Sensitivity analysis [evidence missing: sensitivity_table] is "
        "pending.\n"
    )

    rewritten, demoted = _demote_unresolved_evidence_placeholders(bound)

    assert "[evidence missing:" not in rewritten
    assert "<!-- evidence missing: mortality_rate -->" in rewritten
    assert "<!-- evidence missing: sensitivity_table -->" in rewritten
    # bound link to a real evidence id must survive untouched
    assert "[stratum_table](evidence/x.csv)" in rewritten
    assert demoted == ["mortality_rate", "sensitivity_table"]


def test_manuscript_numeric_auditor_blocks_auroc_drift(ra):
    from easyicu.research_agent.pipeline import _audit_manuscript_numeric_claims

    bound = (
        "The model achieved an AUROC of 0.82 (95% CI: 0.79-0.85) "
        "[model_performance](evidence/model_performance.csv).\n"
    )
    findings = _audit_manuscript_numeric_claims(
        bound,
        per_step_records=[
            {
                "step_id": "01_model_training",
                "status": "ok",
                "step_summary": {
                    "statistic:auroc": 0.7769836515783012,
                    "statistic:brier_score": 0.1793673693714194,
                },
            }
        ],
    )

    messages = [finding.message for finding in findings]
    assert any("AUROC claim" in message for message in messages)
    assert any("confidence interval" in message for message in messages)
    assert {finding.severity for finding in findings} == {"error"}


def test_manuscript_numeric_auditor_allows_normal_rounding(ra):
    from easyicu.research_agent.pipeline import _audit_manuscript_numeric_claims

    bound = (
        "The model achieved an AUROC of 0.78 and Brier score of 0.18 "
        "[model_performance](evidence/model_performance.csv). "
        "The baseline prevalence was 9.6% [model_performance](evidence/model_performance.csv).\n"
    )
    findings = _audit_manuscript_numeric_claims(
        bound,
        per_step_records=[
            {
                "step_id": "01_model_training",
                "status": "ok",
                "step_summary": {
                    "statistic:auroc": 0.7769836515783012,
                    "statistic:brier_score": 0.1793673693714194,
                    "statistic:baseline_prevalence": 0.096,
                },
            }
        ],
    )

    assert findings == []


def test_manuscript_numeric_auditor_ignores_ci_percent_near_outcome_phrase(ra):
    """Regression: '95% confidence interval' must not be read as a 0.95
    prevalence claim when an outcome phrase ('death'/'mortality') appears
    earlier in the same sentence.

    This reproduces the false positive that blocked otherwise-valid
    manuscripts across the gpt-5.4 and gpt-5.5 canonical batches
    (2026-05-20): the lazy proximity window bound the '95%' from
    '... odds of ICU death ... and a 95% confidence interval ...' to the
    'death' phrase and flagged 'prevalence claim 0.95 does not match
    registered baseline prevalence 0.0373'.
    """
    from easyicu.research_agent.pipeline import _audit_manuscript_numeric_claims

    bound = (
        "ICU mortality was 3.7% "
        "[probe](evidence/probe.csv). "
        "Higher early SOFA-2 severity was associated with higher odds of ICU "
        "death, with an odds ratio of 1.39 per modeled one-point increase and "
        "a 95% confidence interval from 1.19 to 1.63 "
        "[assoc](evidence/assoc.csv).\n"
    )
    findings = _audit_manuscript_numeric_claims(
        bound,
        per_step_records=[
            {
                "step_id": "00_probe",
                "status": "ok",
                "step_summary": {"statistic:baseline_prevalence": 0.037},
            }
        ],
    )

    messages = [finding.message for finding in findings]
    assert not any("prevalence claim 0.95" in m for m in messages), messages
    # The genuine 3.7% mortality claim still matches the registered 0.037 and
    # must NOT be flagged either.
    assert not any("prevalence claim" in m for m in messages), messages


def test_repair_common_writer_placeholders_prediction_fallbacks(ra, tmp_path: Path):
    from easyicu.research_agent.evidence import EvidenceStore
    from easyicu.research_agent.pipeline import _repair_common_writer_placeholders

    store = EvidenceStore(tmp_path)
    context_path = tmp_path / "research_context.json"
    context_path.write_text("{}", encoding="utf-8")
    store.register_file(
        kind="log",
        description="ResearchContext",
        source_path=context_path,
        evidence_id="research_context",
        producer="test",
    )
    summary_path = tmp_path / "step_summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    store.register_file(
        kind="statistic",
        description="Model summary",
        source_path=summary_path,
        evidence_id="statistic_step_summary_model",
        producer="test",
        aliases=["01_model_training"],
    )

    scaffold = (
        "The cohort had 1,000 stays {evidence:table_one}. "
        "Mortality was 9.6% {evidence:outcome_rate}. "
        "Performance was evaluated {evidence:primary_association}."
    )
    repaired, repairs = _repair_common_writer_placeholders(
        scaffold,
        context=SimpleNamespace(
            research_question="Build a mortality prediction workflow with AUROC."
        ),
        evidence=store,
    )

    assert "{evidence:table_one}" not in repaired
    assert "{evidence:outcome_rate}" not in repaired
    assert "{evidence:primary_association}" not in repaired
    assert "{evidence:research_context}" in repaired
    assert "{evidence:01_model_training}" in repaired
    assert ("table_one", "research_context") in repairs


def test_execution_gate_and_parent_figure_dependency_helpers(ra):
    from easyicu.research_agent.pipeline import (
        _execution_gate_status,
        _parent_step_id_for_figure_step,
    )

    plan = ra.AnalysisPlan(
        research_question="Build a mortality prediction model.",
        steps=[
            ra.AnalysisStep(
                step_id="01_model_training",
                intent="Train and evaluate model.",
                expected_outputs=["table:model_performance"],
            ),
            ra.AnalysisStep(
                step_id="01_model_training_figure",
                intent="Render the publication figure(s) declared by step '01_model_training'.",
                expected_outputs=["figure:discrimination_calibration"],
            ),
        ],
    )

    assert _parent_step_id_for_figure_step(plan.steps[1]) == "01_model_training"
    gate = _execution_gate_status(
        plan=plan,
        per_step_records=[
            {"step_id": "01_model_training", "status": "execution_failed"},
            {
                "step_id": "01_model_training_figure",
                "status": "skipped_dependency_failed",
                "diagnostic_only": True,
            },
        ],
    )

    assert gate["execution_complete"] is False
    assert gate["failed_steps"] == [
        {"step_id": "01_model_training", "status": "execution_failed"},
        {"step_id": "01_model_training_figure", "status": "skipped_dependency_failed"},
    ]


def test_readiness_artifacts_fail_closed_without_manuscript_ready(ra, tmp_path: Path):
    from easyicu.research_agent.evidence import EvidenceStore
    from easyicu.research_agent.pipeline import _write_readiness_artifacts
    from easyicu.research_agent.schema import ValidationFinding

    context = ra.ResearchContext(
        research_question="Build a mortality prediction model.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="01_model_training",
                intent="Train model.",
                expected_outputs=["table:model_performance"],
            )
        ],
    )
    evidence = EvidenceStore(tmp_path)
    bound_path = tmp_path / "manuscript_scaffold_bound.md"
    bound_path.write_text(
        "# Manuscript scaffold not generated\n\nStrict fail-closed policy blocked drafting.\n",
        encoding="utf-8",
    )

    gates, artifact_paths = _write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=[
            ValidationFinding(
                validator="runner",
                severity="error",
                message="Step 01_model_training failed.",
            )
        ],
        per_step_records=[
            {"step_id": "01_model_training", "status": "execution_failed"}
        ],
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=bound_path,
        stop_after_analysis=False,
    )

    assert gates["manuscript_ready"] is False
    assert gates["execution_complete"] is False
    assert not (tmp_path / "manuscript_ready.md").exists()
    for filename in (
        "run_status.json",
        "claim_ledger.csv",
        "evidence_audit.json",
        "numeric_audit.json",
        "author_review_note.md",
    ):
        assert (tmp_path / filename).exists()
    assert "manuscript_ready" not in artifact_paths
    run_status = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    assert run_status["status"] == "diagnostic_only"


def test_readiness_artifacts_emit_manuscript_ready_only_after_gates_pass(ra, tmp_path: Path):
    from easyicu.research_agent.evidence import EvidenceStore
    from easyicu.research_agent.pipeline import _write_readiness_artifacts

    context = ra.ResearchContext(
        research_question="Estimate mortality risk.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="01_model_training",
                intent="Train model.",
                expected_outputs=["table:model_performance"],
            )
        ],
    )
    evidence = EvidenceStore(tmp_path)
    bound_path = tmp_path / "manuscript_scaffold_bound.md"
    bound_path.write_text(
        "The model achieved an AUROC of 0.78 [model_performance](evidence/model.csv).\n",
        encoding="utf-8",
    )

    gates, artifact_paths = _write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=[],
        per_step_records=[
            {
                "step_id": "01_model_training",
                "status": "ok",
                "step_summary": {"statistic:auroc": 0.776},
            }
        ],
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=bound_path,
        stop_after_analysis=False,
    )

    assert gates["manuscript_ready"] is True
    assert gates["publication_ready"] is False
    assert artifact_paths["manuscript_ready"] == "manuscript_ready.md"
    assert (tmp_path / "manuscript_ready.md").read_text(
        encoding="utf-8"
    ) == bound_path.read_text(encoding="utf-8")


def test_publication_bundle_ready_groups_hash_suffixed_exports_under_one_stem(
    ra, tmp_path: Path
):
    from easyicu.research_agent.evidence import EvidenceStore
    from easyicu.research_agent.pipeline import _publication_figure_bundle_ready

    evidence = EvidenceStore(tmp_path)
    for suffix in ("svg", "png", "pdf", "tiff"):
        path = tmp_path / f"easyicu_publication_figure.{suffix}"
        path.write_text("x", encoding="utf-8")
        evidence.register_file(
            kind="figure",
            description="Publication figure export.",
            source_path=path,
            evidence_id=f"publication_figure_{suffix}",
            producer="publication_figure_skill",
            generation_mode="deterministic_figure_skill",
            metadata={"figure_role": "publication_figure"},
        )

    readiness = _publication_figure_bundle_ready(evidence=evidence, run_dir=tmp_path)

    assert readiness["publication_figure_bundle_ready"] is True
    assert readiness["publication_ready_stems"] == ["easyicu_publication_figure"]


def test_salvage_minimal_contract_step_summary_from_table_one_csv(
    ra, tmp_path: Path
):
    from easyicu.research_agent.pipeline import _salvage_minimal_contract_step_summary

    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True)
    (out_dir / "step_summary.json").write_text("{}", encoding="utf-8")
    pd.DataFrame(
        [
            {"variable": "age", "median": 65.0},
            {"variable": "sofa2", "median": 7.0},
        ]
    ).to_csv(out_dir / "table_one.csv", index=False)

    salvaged = _salvage_minimal_contract_step_summary(
        step=ra.AnalysisStep(
            step_id="01_table_one",
            intent="Summarise baseline characteristics.",
            expected_outputs=["table:table_one"],
        ),
        out_dir=out_dir,
    )

    assert salvaged is True
    payload = json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))
    assert payload["table_one_path"] == "table_one.csv"
    assert payload["n_rows"] == 2
    assert payload["variables_reported"] == ["age", "sofa2"]


def test_metric_claim_extractors_ignore_evidence_link_ids(ra):
    from easyicu.research_agent.pipeline import (
        _extract_metric_claims,
        _extract_percent_claims_near,
    )

    text = (
        "Model performance was assessed using AUROC, Brier score, and baseline prevalence "
        "[01_model_training](evidence/statistic_step_summary.json). "
        "The model demonstrated an AUROC of 0.70 [01_model_training](evidence/x.json) "
        "and a Brier score of 0.20 [01_model_training](evidence/x.json). "
        "The incidence of ICU death was 9.6% [00_probe](evidence/probe.csv)."
    )

    assert _extract_metric_claims(text, r"\b(?:AUROC|AUC)\b") == [0.70]
    assert _extract_metric_claims(text, r"\bBrier(?: score)?\b") == [0.20]
    assert _extract_percent_claims_near(
        text, r"\b(?:mortality|death|outcome incidence)\b"
    ) == [0.096]


def test_critic_accepts_bound_markdown_evidence_links_but_blocks_hidden_missing_refs(ra):
    critic = ra.CriticAgent()

    clean = critic.review_manuscript(
        scaffold=(
            "The cohort comprised 1,000 ICU stays "
            "[research_context](evidence/research_context__research_context.json \"sha256=abc\"). "
            "The model achieved an AUROC of 0.78 "
            "[01_model_training](evidence/statistic_step_summary.json \"sha256=def\")."
        ),
        available_evidence_ids=["research_context", "01_model_training"],
    )

    assert clean.status == "pass"
    assert clean.unsupported_claims == []
    assert clean.missing_evidence_refs == []

    blocked = critic.review_manuscript(
        scaffold=(
            "The cohort comprised 1,000 ICU stays "
            "[research_context](evidence/research_context__research_context.json \"sha256=abc\"). "
            "<!-- evidence missing: table_one -->"
        ),
        available_evidence_ids=["research_context"],
    )

    assert blocked.status == "blocked"
    assert blocked.missing_evidence_refs == ["table_one"]


def test_pipeline_removed_unsupported_sentences_do_not_block_final_manuscript(
    ra, synthetic_cohort, tmp_path: Path, monkeypatch
):
    def fake_manuscript_run(self, *, context, evidence_ids, evidence_digest=None):
        return (
            "The model's performance was consistent across folds, indicating robustness.\n\n"
            "Baseline characteristics are summarised in Table 1 {evidence:research_context}.\n"
        )

    monkeypatch.setattr(ra.ManuscriptAgent, "run", fake_manuscript_run)

    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    result = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="synthetic_test_cohort",
        database="synthetic",
        target_outcome="death",
    )

    run_dir = Path(result.workdir)
    critique = json.loads((run_dir / "manuscript_critique.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    run_status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    filtered = (run_dir / "manuscript_scaffold_filtered.md").read_text(encoding="utf-8")

    assert critique["status"] == "pass"
    assert critique["unsupported_claims"] == []
    assert "performance was consistent" not in filtered
    assert run_status["gates"]["evidence_complete"] is True
    assert not any(
        finding["validator"] == "critic_agent" and "manuscript" in finding["message"]
        for finding in manifest["findings"]
    )


def test_evidence_filter_removes_unquantified_performance_claims(ra, tmp_path: Path):
    from easyicu.research_agent.evidence import EvidenceStore

    store = EvidenceStore(tmp_path)
    scaffold = (
        "The model's performance was consistent across folds, indicating robustness.\n"
        "Methods text can remain here.\n"
    )

    filtered, removed = store.enforce_evidence_bound_scaffold(scaffold)

    assert "performance was consistent" not in filtered
    assert any("performance was consistent" in sentence for sentence in removed)
    assert "Methods text can remain here." in filtered


def test_demote_unresolved_evidence_placeholders_no_op_when_clean(ra):
    from easyicu.research_agent.pipeline import (
        _demote_unresolved_evidence_placeholders,
    )

    text = "All values are bound: [outcome_rate](evidence/o.json).\n"
    rewritten, demoted = _demote_unresolved_evidence_placeholders(text)
    assert rewritten == text
    assert demoted == []


def test_extract_missing_index_columns_parses_keyerror_message(ra):
    """Regression: pandas KeyError must yield the exact missing column list."""
    from easyicu.research_agent.pipeline import _extract_missing_index_columns

    run_log = (
        "Traceback (most recent call last):\n"
        "  File \"analysis.py\", line 142, in <module>\n"
        "    model_df = df[[outcome_col, predictor_col] + covariates].copy()\n"
        "KeyError: \"['age', 'sex', 'map_min_24h', 'vaso_any_24h'] not in index\""
    )

    cols = _extract_missing_index_columns(run_log)
    assert cols == ["age", "sex", "map_min_24h", "vaso_any_24h"]


def test_extract_missing_index_columns_empty_when_no_keyerror(ra):
    from easyicu.research_agent.pipeline import _extract_missing_index_columns

    assert _extract_missing_index_columns("") == []
    assert _extract_missing_index_columns("ValueError: bad things") == []


def test_strip_columns_from_list_literals_removes_known_missing_entries(ra):
    """List literals containing only missing column strings are pruned."""
    from easyicu.research_agent.pipeline import _strip_columns_from_list_literals

    code = (
        "predictor_col = 'lact_max_24h'\n"
        "covariates = [\"age\", \"sex\", \"map_min_24h\", \"vaso_any_24h\"]\n"
        "X = model_df[[predictor_col] + covariates]\n"
    )

    repaired = _strip_columns_from_list_literals(
        code, ["age", "sex", "map_min_24h", "vaso_any_24h"]
    )

    assert "covariates = []" in repaired
    # Mixed expression like ``[predictor_col] + covariates`` must be left
    # alone because it contains non-literal elements.
    assert "[[predictor_col] + covariates]" in repaired


def test_strip_columns_from_list_literals_preserves_known_columns(ra):
    """Only the missing entries are removed; known columns survive."""
    from easyicu.research_agent.pipeline import _strip_columns_from_list_literals

    code = "covariates = ['age', 'lact_max_24h', 'unknown_col']\n"

    repaired = _strip_columns_from_list_literals(
        code, ["age", "unknown_col"]
    )

    assert "covariates = ['lact_max_24h']" in repaired


def test_strip_columns_from_list_literals_noop_when_columns_absent(ra):
    """No-op when the missing columns don't appear in any list literal."""
    from easyicu.research_agent.pipeline import _strip_columns_from_list_literals

    code = "covariates = ['lact_max_24h']\n"
    repaired = _strip_columns_from_list_literals(code, ["age", "sex"])
    assert repaired == code


def test_deterministic_runner_repair_fixes_column_hallucination(ra):
    """End-to-end: agent-emitted column-hallucination KeyError is auto-patched."""
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = (
        "import pandas as pd\n"
        "df = pd.read_parquet('cohort.parquet')\n"
        "outcome_col = 'death'\n"
        "predictor_col = 'lact_max_24h'\n"
        "covariates = [\"age\", \"sex\", \"map_min_24h\", \"vaso_any_24h\"]\n"
        "model_df = df[[outcome_col, predictor_col] + covariates].copy()\n"
    )
    run_log = (
        "Traceback (most recent call last):\n"
        "  File \"analysis.py\", line 6, in <module>\n"
        "KeyError: \"['age', 'sex', 'map_min_24h', 'vaso_any_24h'] not in index\""
    )

    result = _deterministic_runner_repair(code=code, run_log=run_log)

    assert result is not None, "runner repair should fire on column hallucination"
    repair_name, repaired = result
    assert repair_name == "strip_unknown_cols_from_list_literals_v1"
    assert "covariates = []" in repaired
    # The previous_repair guard prevents infinite repair loops.
    second = _deterministic_runner_repair(
        code=repaired, run_log=run_log, previous_repair=repair_name
    )
    assert second is None or second[0] != repair_name


def test_preserve_figure_steps_after_replan_re_attaches_dropped_figure_step(ra):
    """Regression: Replanner must not silently drop figure-producing steps.

    qwen3-coder-30b under naive arms (no ICU context) often returns a
    revised plan that rationalises away the figure step after the probe
    summary. Task contracts still require the figure artefact, so the
    pipeline must re-attach any dropped step whose ``expected_outputs``
    declare a figure/plot output.
    """
    from easyicu.research_agent.pipeline import (
        _preserve_figure_steps_after_replan,
        _step_produces_figure,
    )

    fig_step = ra.AnalysisStep(
        step_id="02_summary_figure",
        intent="Render publication-ready figure for the table-one summary.",
        expected_outputs=["figure:table_one_summary"],
    )
    table_step = ra.AnalysisStep(
        step_id="01_table_one",
        intent="Build descriptive Table 1.",
        expected_outputs=["table:table_one"],
    )
    current = ra.AnalysisPlan(
        research_question="describe the cohort",
        steps=[table_step, fig_step],
    )
    revised = ra.AnalysisPlan(
        research_question="describe the cohort",
        steps=[table_step],
        revision=2,
    )

    assert _step_produces_figure(fig_step) is True
    assert _step_produces_figure(table_step) is False

    preserved, findings = _preserve_figure_steps_after_replan(
        current=current,
        revised=revised,
    )

    preserved_ids = [s.step_id for s in preserved.steps]
    assert "02_summary_figure" in preserved_ids, (
        "dropped figure step must be re-attached to revised plan; "
        f"got steps={preserved_ids}"
    )
    assert any(
        f.severity == "warning" and "figure-producing" in f.message
        for f in findings
    )


def test_preserve_figure_steps_after_replan_no_op_when_figure_kept(ra):
    """No-op when the replanner kept all figure steps."""
    from easyicu.research_agent.pipeline import _preserve_figure_steps_after_replan

    fig_step = ra.AnalysisStep(
        step_id="02_summary_figure",
        intent="Render summary figure.",
        expected_outputs=["figure:table_one_summary"],
    )
    table_step = ra.AnalysisStep(
        step_id="01_table_one",
        intent="Build Table 1.",
        expected_outputs=["table:table_one"],
    )
    current = ra.AnalysisPlan(
        research_question="describe the cohort",
        steps=[table_step, fig_step],
    )
    revised = ra.AnalysisPlan(
        research_question="describe the cohort",
        steps=[table_step, fig_step],
        revision=2,
    )

    preserved, findings = _preserve_figure_steps_after_replan(
        current=current,
        revised=revised,
    )

    assert findings == []
    assert [s.step_id for s in preserved.steps] == [
        "01_table_one",
        "02_summary_figure",
    ]


def test_step_contract_repair_guidance_for_prediction_categorical_passthrough(ra):
    from easyicu.research_agent.pipeline import _step_contract_repair_guidance

    step = ra.AnalysisStep(
        step_id="03_model_training",
        intent="Train mortality prediction model.",
        expected_outputs=["statistic:auroc", "statistic:brier_score"],
    )

    guidance = _step_contract_repair_guidance(
        step=step,
        step_summary={
            "auroc": None,
            "brier_score": None,
            "error": "could not convert string to float: 'M'",
        },
        code="categorical_transformer = Pipeline([('onehot', 'passthrough')])",
    )

    assert "OneHotEncoder" in guidance
    assert "categorical variable reached a numeric estimator" in guidance


def test_step_contract_repair_guidance_for_empty_sex_coercion(ra):
    from easyicu.research_agent.pipeline import _step_contract_repair_guidance

    step = ra.AnalysisStep(
        step_id="03_association_model",
        intent="Fit lactate mortality model.",
        expected_outputs=["odds_ratio"],
    )

    guidance = _step_contract_repair_guidance(
        step=step,
        step_summary={
            "n_total": 0,
            "primary_predictor": "lactate_max_24h",
            "odds_ratio": None,
            "skipped": "zero-size array to reduction operation maximum which has no identity",
        },
        code="model_df = model_df.apply(pd.to_numeric, errors='coerce')\nmodel_df = pd.get_dummies(model_df, columns=['sex'])",
    )

    assert "dummy-encoding `sex` first" in guidance
    assert "numeric odds ratio" in guidance


def test_step_contract_repair_guidance_for_object_dtype_logit(ra):
    from easyicu.research_agent.pipeline import _step_contract_repair_guidance

    step = ra.AnalysisStep(
        step_id="04_primary_association_model",
        intent="Fit lactate mortality model with figure output.",
        expected_outputs=["figure:lactate_mortality_plot", "odds_ratio"],
    )

    guidance = _step_contract_repair_guidance(
        step=step,
        step_summary={
            "primary_predictor": "lactate_max_24h",
            "or_estimate": None,
            "model_notes": "Model fitting failed: Pandas data cast to numpy dtype of object.",
        },
        code="X = pd.get_dummies(X, columns=['sex'], drop_first=True)\nmodel = sm.Logit(y, X)",
    )

    assert "object-dtype design matrix" in guidance
    assert "X.astype(float)" in guidance
    assert "non-null odds ratio" in guidance


def test_step_contract_repair_guidance_requires_figure_recording(ra):
    from easyicu.research_agent.pipeline import _step_contract_repair_guidance

    step = ra.AnalysisStep(
        step_id="03_association_model",
        intent="Estimate lactate association.",
        expected_outputs=["table:adjusted_association", "figure:lactate_mortality_plot"],
    )

    guidance = _step_contract_repair_guidance(
        step=step,
        step_summary={"estimate": 1.21, "ci_lower": 1.10, "ci_upper": 1.33},
        code="fig.savefig('lactate.png')",
    )

    assert "figure output" in guidance
    assert "figure_path" in guidance


def test_step_contract_repair_guidance_for_clustering_contract(ra):
    from easyicu.research_agent.pipeline import _step_contract_repair_guidance

    step = ra.AnalysisStep(
        step_id="01_trajectory_clustering",
        intent="Cluster shock physiology.",
        expected_outputs=["statistic:silhouette_score", "table:cluster_characteristics"],
    )

    guidance = _step_contract_repair_guidance(
        step=step,
        step_summary={"error": "silhouette_score missing"},
        code="labels = kmeans.fit_predict(X)",
    )

    assert "silhouette_score" in guidance
    assert "cluster_characteristics.csv" in guidance
    assert "self-contained" in guidance


def test_semantic_aliases_include_step_id_and_prediction_aliases(ra, tmp_path: Path):
    from easyicu.research_agent.pipeline import _semantic_aliases_for

    step = ra.AnalysisStep(
        step_id="01_model_training",
        intent="Train mortality prediction model.",
        expected_outputs=["statistic:auroc", "statistic:brier_score"],
    )

    aliases = _semantic_aliases_for(step, tmp_path / "step_summary.json")

    assert "01_model_training" in aliases
    assert "model_training" in aliases
    assert "model_performance" in aliases
    assert "prediction_performance" in aliases
    assert "baseline_prevalence" in aliases
    assert "primary_association" not in aliases
    assert "outcome_rate" not in aliases


def test_infer_generic_v15_fallback_key_skips_prediction_tasks(ra):
    from easyicu.research_agent.pipeline import _infer_generic_v15_fallback_key

    key = _infer_generic_v15_fallback_key(
        code=(
            "from sklearn.model_selection import StratifiedKFold\n"
            "y_pred = model.predict_proba(X)[:, 1]\n"
            "roc_auc_score(y, y_pred)\n"
            "brier_score_loss(y, y_pred)\n"
            "lactate_max_24h\n"
        ),
        diagnostic_text="Prediction model for t07_mortality_prediction_auroc calibration plot",
    )
    assert key is None


def test_semantic_aliases_bind_cohort_summary_outcome_rate(ra, tmp_path: Path):
    from easyicu.research_agent.pipeline import _semantic_aliases_for

    step = ra.AnalysisStep(
        step_id="01_cohort_summary",
        intent="Summarize cohort mortality and missingness.",
        expected_outputs=["table:table_one", "statistic:mortality_rate"],
    )

    aliases = _semantic_aliases_for(step, tmp_path / "step_summary.json")

    assert "cohort_summary" in aliases
    assert "outcome_rate" in aliases
    assert "mortality_rate" in aliases


def test_semantic_aliases_bind_t10_missing_data_audit_outcome_rate(ra, tmp_path: Path):
    from easyicu.research_agent.pipeline import _semantic_aliases_for

    step = ra.AnalysisStep(
        step_id="01_missing_data_audit",
        intent="Audit missingness and event rate.",
        expected_outputs=["table:missingness_summary", "statistic:event_rate"],
    )

    aliases = _semantic_aliases_for(step, tmp_path / "step_summary.json")

    assert "missingness" in aliases
    assert "outcome_rate" in aliases


def test_semantic_aliases_include_clustering_aliases(ra, tmp_path: Path):
    from easyicu.research_agent.pipeline import _semantic_aliases_for

    step = ra.AnalysisStep(
        step_id="01_trajectory_clustering",
        intent="Cluster shock physiology.",
        expected_outputs=["table:cluster_characteristics", "statistic:silhouette_score"],
    )

    aliases = _semantic_aliases_for(step, tmp_path / "step_summary.json")

    assert "table_one" in aliases
    assert "cluster_summary" in aliases
    assert "cluster_characteristics" in aliases
    assert "cluster_mortality" in aliases


def test_semantic_aliases_bind_kdigo_sensitivity_to_primary_association(ra, tmp_path: Path):
    from easyicu.research_agent.pipeline import _semantic_aliases_for

    step = ra.AnalysisStep(
        step_id="06_sensitivity_reduced_vars",
        intent="Fit reduced-variable KDIGO sensitivity model.",
        expected_outputs=["statistic:adjusted_or_ci"],
    )

    aliases = _semantic_aliases_for(step, tmp_path / "step_summary.json")

    assert "primary_association" in aliases
    assert "kdigo_sensitivity" in aliases


def test_semantic_aliases_bind_sofa_zero_audit_outcome_rate(ra, tmp_path: Path):
    from easyicu.research_agent.pipeline import _semantic_aliases_for

    step = ra.AnalysisStep(
        step_id="01_sofa_zero_audit",
        intent="Audit SOFA-zero mortality and missingness.",
        expected_outputs=["statistic:sofa_zero_count", "statistic:mortality_rate"],
    )

    aliases = _semantic_aliases_for(step, tmp_path / "step_summary.json")

    assert "sofa_zero_count" in aliases
    assert "outcome_rate" in aliases


def test_advanced_plan_contract_collapses_prediction_to_one_step(ra):
    from easyicu.research_agent.pipeline import _enforce_advanced_plan_contract
    from easyicu.research_agent.schema import UserPreferences

    ctx = ra.ResearchContext(
        research_question="Build a mortality prediction model.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
        user_preferences=UserPreferences(inferred_analysis_family="prediction_model"),
    )
    plan = ra.AnalysisPlan(
        research_question=ctx.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="01_train_model",
                intent="Train model.",
                inputs=["age", "sex"],
                expected_outputs=["model:trained_model"],
            ),
            ra.AnalysisStep(
                step_id="02_evaluate_auroc",
                intent="Evaluate AUROC from prior predictions.",
                inputs=["01_train_model"],
                expected_outputs=["statistic:auroc"],
            ),
        ],
    )

    revised, findings = _enforce_advanced_plan_contract(plan=plan, context=ctx)

    assert [step.step_id for step in revised.steps] == ["01_model_training"]
    assert "statistic:auroc" in revised.steps[0].expected_outputs
    assert "statistic:brier_score" in revised.steps[0].expected_outputs
    assert findings and findings[0].validator == "plan_contract"


def test_deterministic_summary_repair_restores_primary_predictor_after_soft_failure(ra):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    code = """
import pandas as pd
import statsmodels.api as sm
df = pd.read_parquet(cohort_path)
model_df = df[['lactate_max_24h', 'map_min_24h', 'vaso_any_24h', 'age', 'sex', 'death']].copy()
y = model_df['death'].astype(float)
X = model_df[['map_min_24h', 'vaso_any_24h', 'age', 'sex']].astype(float)
X = sm.add_constant(X, has_constant='add')
try:
    model = sm.Logit(y, X)
    result = model.fit(disp=0)
    coef = result.params['lactate_max_24h']
    conf_int = result.conf_int().loc['lactate_max_24h']
    p_value = result.pvalues['lactate_max_24h']
except Exception as e:
    step_summary = {'error': str(e)}
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "primary_predictor": "lactate_max_24h",
            "estimate": None,
            "error": "'lactate_max_24h'",
        },
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "primary_predictor_omitted_from_design_v1"
    assert "X = model_df[['lactate_max_24h'," in patched


def test_deterministic_summary_repair_dedupes_predictor_design_matrix(ra):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    code = """
model_df = pd.get_dummies(model_df, columns=categorical_cols, drop_first=True)
x_cols = [predictor_col] + [col for col in model_df.columns if col != outcome_col]
X = model_df[x_cols]
X = sm.add_constant(X, has_constant="add")
try:
    model = sm.Logit(y, X)
except Exception as e:
    pass
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "predictor": "lactate_max_24h",
            "statistic": {"adjusted_or_ci": {"or": None}},
        },
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "dedupe_predictor_numeric_design_v1"
    assert "col not in [outcome_col, predictor_col]" in patched
    assert '.apply(pd.to_numeric, errors="coerce").astype(float)' in patched


def test_deterministic_summary_repair_preserves_categorical_sex_before_dropna(ra):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    code = """
model_df = df[required_cols].copy()
model_df = model_df.apply(pd.to_numeric, errors="coerce")
model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
model_df = model_df.dropna(subset=['lactate_max_24h'])
X = model_df[['lactate_max_24h', 'age', 'sex', 'map_min_24h', 'vaso_any_24h']].astype(float)
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "primary_predictor": "lactate_max_24h",
            "statistic:adjusted_or": None,
            "skipped": "No valid data after dropping lactate missing rows",
        },
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "sex_numeric_coercion_before_dropna_v1"
    assert "model_df['sex'] = model_df['sex'].astype(str).str.lower().isin(['m', 'male']).astype(float)" in patched
    assert "if col != 'sex':" in patched
    compile(patched, "<patched>", "exec")


def test_deterministic_summary_repair_handles_insufficient_data_message(ra):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    code = """
model_df = df[['lactate_max_24h', 'map_min_24h', 'vaso_any_24h', 'age', 'sex', 'death']].copy()
model_df = model_df.apply(pd.to_numeric, errors="coerce")
model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "primary_predictor": "lactate_max_24h",
            "estimate": None,
            "skipped": "Insufficient data for regression analysis",
        },
        previous_repair="primary_predictor_omitted_from_design_v1",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "sex_numeric_coercion_before_dropna_v1"
    assert "isin(['m', 'male'])" in patched


def test_deterministic_summary_repair_handles_null_model_summary_with_sex(ra):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    code = """
covariates = ['age', 'sex']
model_df = df[[outcome_col, primary_predictor] + covariates].copy()
model_df = model_df.apply(pd.to_numeric, errors="coerce")
model_df = model_df.replace([np.inf, -np.inf], np.nan)
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "primary_predictor": "lactate_max_24h",
            "complete_case_n": None,
            "lactate_or": None,
        },
        previous_repair="primary_predictor_omitted_from_design_v1",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "sex_numeric_coercion_before_dropna_v1"
    assert "model_df['sex'] = model_df['sex'].astype(str)" in patched
    assert "if col != 'sex':" in patched


def test_deterministic_summary_repair_infers_predictor_from_code_for_sex_coercion(ra):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    code = """
primary_predictor = 'kdigo_stage_max_24h'
covariates = ['age', 'sex', 'sofa2_renal_max_24h', 'vaso_any_24h']
all_vars = [primary_predictor] + covariates + [outcome_col]
model_df = df[all_vars].copy()
model_df = model_df.apply(pd.to_numeric, errors="coerce")
model_df = model_df.replace([np.inf, -np.inf], np.nan)
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "statistic:primary_or": None,
            "statistic:complete_case_n": 0,
        },
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "sex_numeric_coercion_before_dropna_v1"
    assert "model_df['sex'] = model_df['sex'].astype(str)" in patched
    assert "if col != 'sex':" in patched


def test_deterministic_summary_repair_encodes_raw_sex_for_logit_without_predictor(ra):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    code = """
outcome_col = "death"
lactate_col = "lactate_max_24h"
predictor_col = lactate_col
covariates = ["age", "sex", "los_icu"]
model_df = df[[outcome_col, predictor_col] + covariates].copy()
y_complete = model_df[outcome_col]
X_complete = sm.add_constant(model_df[covariates + [predictor_col]], has_constant="add")
complete_case_model = sm.Logit(y_complete, X_complete).fit(disp=0)
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "statistic:primary_or": None,
            "statistic:complete_case_n": 217,
            "model:complete_case_model": None,
        },
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "sex_binary_encode_for_logit_v1"
    assert "model_df['sex'] = model_df['sex'].astype(str).str.lower().isin(['m', 'male']).astype(float)" in patched
    assert "model_df[col] = pd.to_numeric(model_df[col], errors=\"coerce\")" in patched


def test_deterministic_summary_repair_reuses_dtype_repair_for_soft_failure(ra):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    code = """
import statsmodels.api as sm
X = model_df[x_cols]
y = model_df[outcome_col]
model = sm.Logit(y, X)
result = model.fit(disp=0)
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "primary_predictor": "lactate_max_24h",
            "results": {
                "complete-case": {
                    "or_estimate": None,
                    "error": "Pandas data cast to numpy dtype of object. Check input data with np.asarray(data).",
                }
            },
        },
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "dtype_coerce_v1"
    assert "_easyicu_runner_repair_v1" in patched


def test_deterministic_summary_repair_patches_nested_logit_helper_dtype(ra):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    code = '''
import pandas as pd
import statsmodels.api as sm

def _fit_logistic(X, y):
    X = X.apply(pd.to_numeric, errors="coerce")
    X = sm.add_constant(X, has_constant="add")
    y = pd.to_numeric(y, errors="coerce")
    data = pd.concat([y, X], axis=1).dropna()
    y_clean = data[y.name]
    X_clean = data.drop(columns=[y.name])
    model = sm.Logit(y_clean, X_clean)
    return model.fit(disp=False)
'''
    repaired = _deterministic_summary_repair(
        code=code,
        previous_repair="dtype_coerce_v1",
        step_summary={
            "model:logistic_regression_complete_case": {
                "converged": False,
                "error": "Pandas data cast to numpy dtype of object. Check input data with np.asarray(data).",
            },
        },
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "statsmodels_helper_design_float_v1"
    assert 'X = X.apply(pd.to_numeric, errors="coerce").astype(float)' in patched
    assert 'X_clean = data.drop(columns=[y.name]).apply(pd.to_numeric, errors="coerce").astype(float)' in patched


def test_deterministic_summary_repair_casts_dummy_design_for_null_logit(ra):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    code = """
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X_final = sm.add_constant(X_encoded, has_constant="add")
model = sm.Logit(y, X_final)
result = model.fit(disp=0)
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "predictor": "lactate_max_24h",
            "estimate": None,
            "error_message": "Unknown error",
        },
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "statsmodels_dummy_design_float_v1"
    assert 'X_encoded.apply(pd.to_numeric, errors="coerce").astype(float)' in patched


def test_deterministic_summary_repair_handles_formula_dummy_name_error(ra):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    code = """
import pandas as pd
from statsmodels.formula.api import logit
df = pd.read_parquet(cohort_path)
df = pd.get_dummies(df, columns=['sex'], drop_first=True)
try:
    model = logit('death ~ sofa2 + age + sex_F', data=df).fit()
except Exception as e:
    step_summary = {
        'primary_predictor': 'sofa2',
        'estimate': None,
        'error': str(e),
    }
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "primary_predictor": "sofa2",
            "estimate": None,
            "error": "Error evaluating factor: NameError: name 'sex_F' is not defined",
        },
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "formula_dummy_name_fallback_v1"
    assert "_easyicu_primary_association_fallback_v1" in patched
    assert '"primary_or"' in patched
    compile(patched, "<patched>", "exec")


def test_deterministic_summary_repair_restores_predictor_in_helper_design(ra):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    code = """
def compute_or_ci(df, predictor, outcome_col, covariates):
    model_df = df[[predictor] + covariates + [outcome_col]].copy()
    model_df = model_df.dropna()
    y = model_df[outcome_col].astype(float)
    X = model_df[covariates].astype(float)
    X = sm.add_constant(X, has_constant="add")
    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=0)
    if predictor in result.params.index:
        return np.exp(result.params[predictor]), result.conf_int().loc[predictor, 0], result.conf_int().loc[predictor, 1]
    return None, None, None
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "statistic:primary_or": None,
            "manifest:robustness_analysis_manifest": {
                "primary_predictor": "lactate_max_24h"
            },
        },
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "primary_predictor_omitted_from_design_v1"
    assert "design_cols = [predictor] + [col for col in covariates if col != predictor]" in patched
    assert 'model_df[design_cols].apply(pd.to_numeric, errors="coerce").astype(float)' in patched


def test_deterministic_summary_repair_stabilizes_robustness_missingness_models(ra):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    code = """
outcome_col = 'death'
primary_predictor = 'lactate_max_24h'
covariates = ['age', 'sex', 'los_icu', 'sofa2_max_24h', 'map_min_24h', 'vaso_any_24h', 'bili_max_24h', 'bili_n_24h', 'creat_max_24h']
model_df = df[[outcome_col, primary_predictor] + covariates].copy()
if 'sex' in model_df.columns:
    model_df['sex'] = model_df['sex'].astype(str).str.lower().isin(['m', 'male']).astype(float)
for col in model_df.columns:
    if col != 'sex':
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
model_df = model_df.replace([np.inf, -np.inf], np.nan)
cc_df = model_df.dropna(subset=[primary_predictor])
mi_df = model_df.copy()
mi_df['lactate_missing'] = mi_df[primary_predictor].isna().astype(int)
rv_df = model_df.dropna(subset=[primary_predictor])
cc_X = sm.add_constant(cc_df[[primary_predictor] + covariates], has_constant="add")
mi_X = sm.add_constant(mi_df[[primary_predictor, 'lactate_missing'] + covariates], has_constant="add")
rv_X = sm.add_constant(rv_df[covariates], has_constant="add")
cc_result = sm.Logit(cc_df[outcome_col], cc_X).fit(disp=0)
mi_result = sm.Logit(mi_df[outcome_col], mi_X).fit(disp=0)
rv_result = sm.Logit(rv_df[outcome_col], rv_X).fit(disp=0)
for strategy, or_est, or_lower, or_upper, n, event_rate in [
    ('Complete-case', cc_or, cc_or_lower, cc_or_upper, cc_n, cc_event_rate),
    ('Missing-indicator', mi_or, mi_or_lower, mi_or_upper, len(mi_df), mi_df[outcome_col].mean()),
    ('Reduced-variable', rv_or, rv_or_lower, rv_or_upper, len(rv_df), rv_df[outcome_col].mean())
]:
    pass
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "statistic:primary_or": None,
            "statistic:complete_case_n": 450,
            "model:complete_case_model": None,
            "model:missing_indicator_model": None,
            "model:reduced_variable_model": None,
        },
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "robustness_missingness_contract_v1"
    assert "reduced_covariates = [c for c in covariates if model_df[c].isna().mean() <= 0.2]" in patched
    assert "cc_df = model_df.dropna(subset=[outcome_col, primary_predictor] + covariates)" in patched
    assert "mi_df[primary_predictor] = mi_df[primary_predictor].fillna(0)" in patched
    assert "mi_df = mi_df.dropna(subset=[outcome_col] + covariates)" in patched
    assert "rv_df = model_df[[outcome_col, primary_predictor] + reduced_covariates].dropna()" in patched
    assert 'rv_X = sm.add_constant(rv_df[[primary_predictor] + reduced_covariates], has_constant="add")' in patched


def test_deterministic_summary_repair_allows_generic_unknown_error(ra):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    code = """
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X_final = sm.add_constant(X_encoded, has_constant="add")
model = sm.Logit(y, X_final)
result = model.fit(disp=0)
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "predictor": "lactate_max_24h",
            "estimate": None,
            "error_message": "Unknown error",
        },
    )

    assert repaired is not None
    name, _patched = repaired
    assert name == "statsmodels_dummy_design_float_v1"


def test_deterministic_summary_repair_casts_bool_before_simple_imputer(ra):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    code = """
X_sklearn = model_df[x_cols].copy()
pipeline.fit(X_sklearn, y_sklearn)
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "statistic": {"adjusted_or": None},
            "skipped": [
                "SimpleImputer does not support data with dtype bool. Please provide numeric data."
            ],
        },
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "sklearn_bool_imputer_cast_v1"
    assert "select_dtypes(include=['bool'])" in patched
    assert "astype(int)" in patched


def test_deterministic_runner_repair_filters_missing_dummy_columns(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
model_df = pd.get_dummies(model_df, columns=["sex"], drop_first=True)
x_cols = [predictor_col] + ["age", "sex"]
model_df_subset = model_df[[outcome_col] + x_cols].copy()
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="KeyError: \"['sex'] not in index\"",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "filter_x_cols_after_dummy_encoding_v1"
    assert "x_cols = [col for col in x_cols if col in model_df.columns]" in patched


def test_step_contract_findings_accepts_textual_or_summary(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings
    from easyicu.research_agent.schema import AnalysisStep

    step = AnalysisStep(
        step_id="04_primary_association_model",
        intent="Fit logistic regression for lactate association.",
        expected_outputs=["summary:primary_association"],
    )
    findings = _step_contract_findings(
        step=step,
        step_summary={
            "summary": {
                "notes": [
                    "Association estimate with lactate: OR=1.219 (95% CI 1.116-1.332)."
                ]
            }
        },
    )
    assert [f for f in findings if f.severity == "error"] == []


def test_advanced_plan_contract_normalizes_robustness_steps(ra):
    from easyicu.research_agent.pipeline import _enforce_advanced_plan_contract
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        CohortDescriptor,
        ResearchContext,
        UserPreferences,
    )

    plan = AnalysisPlan(
        research_question="Compare lactate missing-data strategies.",
        steps=[
            AnalysisStep(
                step_id="01_missingness",
                intent="Summarize missingness.",
                expected_outputs=["table:missingness"],
            ),
            AnalysisStep(
                step_id="03_model_fitting_complete_case",
                intent="Fit complete-case logistic regression.",
                expected_outputs=["model:complete_case_model"],
            ),
            AnalysisStep(
                step_id="04_robustness_figure",
                intent="Generate robustness figure from model outputs.",
                expected_outputs=["figure:robustness_plot"],
            ),
        ],
    )
    context = ResearchContext(
        research_question="Compare lactate missing-data strategies.",
        cohort=CohortDescriptor(
            cohort_name="cohort",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
        target_outcome="death",
        user_preferences=UserPreferences(inferred_analysis_family="robustness"),
    )

    revised, findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    step_ids = [step.step_id for step in revised.steps]
    assert step_ids == ["01_missingness", "03_complete_case_robustness"]
    assert any("statistic:primary_or" in step.expected_outputs for step in revised.steps)
    assert findings and findings[0].validator == "plan_contract"


def test_advanced_plan_contract_normalizes_bias_audit_steps(ra):
    from easyicu.research_agent.pipeline import _enforce_advanced_plan_contract
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        CohortDescriptor,
        ResearchContext,
        UserPreferences,
    )

    plan = AnalysisPlan(
        research_question="Estimate vasopressor association with mortality and audit selection bias.",
        steps=[
            AnalysisStep(
                step_id="01_cohort_summary",
                intent="Summarize cohort and vasopressor exposure.",
                expected_outputs=["table:cohort_summary"],
            ),
            AnalysisStep(
                step_id="02_outcome_incidence",
                intent="Report mortality incidence.",
                expected_outputs=["statistic:mortality_rate"],
            ),
            AnalysisStep(
                step_id="03_missingness_audit",
                intent="Audit norepinephrine-equivalent missingness.",
                expected_outputs=["table:missingness_profile"],
            ),
        ],
    )
    context = ResearchContext(
        research_question="Estimate vasopressor association with mortality and audit selection bias.",
        cohort=CohortDescriptor(
            cohort_name="cohort",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
        target_outcome="death",
        user_preferences=UserPreferences(inferred_analysis_family="bias_audit"),
    )

    revised, findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    assert [step.step_id for step in revised.steps] == ["02_treatment_exposure_bias_association"]
    assert "statistic:primary_or" in revised.steps[0].expected_outputs
    assert "statistic:selection_bias_warning" in revised.steps[0].expected_outputs
    assert findings and findings[0].validator == "plan_contract"


def test_advanced_plan_contract_does_not_rewrite_sofa_data_quality_audit(ra):
    from easyicu.research_agent.pipeline import _enforce_advanced_plan_contract
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        CohortDescriptor,
        ResearchContext,
        UserPreferences,
    )

    plan = AnalysisPlan(
        research_question=(
            "Audit whether SOFA-2 equal to zero co-occurs with high lactate, "
            "low MAP, vasopressor exposure, or mortality."
        ),
        steps=[
            AnalysisStep(
                step_id="01_sofa_zero_audit",
                intent="Audit SOFA zero rows and missing components.",
                expected_outputs=["statistic:sofa_zero_count", "table:sofa_zero_audit"],
            )
        ],
    )
    context = ResearchContext(
        research_question=plan.research_question,
        cohort=CohortDescriptor(
            cohort_name="cohort",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
        target_outcome="death",
        user_preferences=UserPreferences(inferred_analysis_family="data_quality_audit"),
    )

    revised, findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    assert [step.step_id for step in revised.steps] == ["01_sofa_zero_audit"]
    assert findings == []


def test_advanced_plan_contract_infers_robustness_without_user_preferences(ra):
    from easyicu.research_agent.pipeline import _enforce_advanced_plan_contract
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        CohortDescriptor,
        ResearchContext,
    )

    plan = AnalysisPlan(
        research_question="Compare complete-case, missing-indicator, and reduced-variable lactate models.",
        steps=[
            AnalysisStep(
                step_id="01_robustness_analysis",
                intent="Compare complete-case and missing-indicator robustness strategies.",
                expected_outputs=["table:robustness_summary", "figure:robustness_figure"],
            ),
        ],
    )
    context = ResearchContext(
        research_question="Compare complete-case, missing-indicator, and reduced-variable lactate models.",
        cohort=CohortDescriptor(
            cohort_name="cohort",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
        target_outcome="death",
        user_preferences=None,
    )

    revised, findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    assert [step.step_id for step in revised.steps] == ["03_complete_case_robustness"]
    assert "statistic:primary_or" in revised.steps[0].expected_outputs
    assert "statistic:complete_case_n" in revised.steps[0].expected_outputs
    assert findings and findings[0].validator == "plan_contract"


def test_salvage_stdout_json_step_summary(ra, tmp_path: Path):
    from easyicu.research_agent.pipeline import _salvage_stdout_json_step_summary
    from easyicu.research_agent.runner import RunResult

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    result = RunResult(
        step_id="01_outcome_incidence",
        script_path=tmp_path / "analysis.py",
        cwd=tmp_path,
        out_dir=out_dir,
        stdout='prefix\n{"statistic:outcome_incidence": 0.096, "sample_size": 1000}\n',
        stderr="",
        returncode=0,
        duration_seconds=0.1,
    )

    assert _salvage_stdout_json_step_summary(result) is True
    saved = json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))
    assert saved["statistic:outcome_incidence"] == 0.096


def test_salvage_named_json_step_summary(ra, tmp_path: Path):
    from easyicu.research_agent.pipeline import _salvage_named_json_step_summary
    from easyicu.research_agent.runner import RunResult

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    summary_json = out_dir / "sofa_zero_artefact_audit_summary.json"
    summary_json.write_text(
        json.dumps({"sofa_zero_count": 41, "guardrail_warning": True}),
        encoding="utf-8",
    )
    result = RunResult(
        step_id="01_sofa_zero_audit",
        script_path=tmp_path / "analysis.py",
        cwd=tmp_path,
        out_dir=out_dir,
        stdout="",
        stderr="",
        returncode=0,
        duration_seconds=0.1,
        artefacts=[summary_json],
    )

    assert _salvage_named_json_step_summary(result) is True
    saved = json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))
    assert saved["sofa_zero_count"] == 41
    assert saved["guardrail_warning"] is True


def test_pipeline_run_from_spec_writes_runtime_artifacts(ra, tmp_path: Path):
    df = pd.DataFrame({
        "stay_id": [1, 2, 3, 4],
        "age": [60.0, 72.0, 55.0, 80.0],
        "sofa2": [0, 1, 5, 8],
        "death": [0, 0, 1, 1],
    })
    cohort_path = tmp_path / "cohort.parquet"
    df.to_parquet(cohort_path, index=False)
    spec = ra.ExperimentSpec(
        question="Analyze whether admission SOFA-2 is associated with ICU mortality.",
        cohort=ra.CohortInputSpec(
            cohort=str(cohort_path),
            cohort_name="demo",
            database="synthetic",
            target_outcome="death",
            user_preferences={"inferred_analysis_family": "association_study"},
        ),
        runtime=ra.RuntimeSpec(
            workdir=str(tmp_path / "runs"),
            stop_after_analysis=True,
            enable_literature=False,
            enable_visual_qa=False,
            enable_latex=False,
        ),
    )
    pipe = ra.ResearchAgentPipeline(
        workdir=spec.runtime.workdir,
        llm=ra.MockLLMClient(),
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
    )
    result = pipe.run_from_spec(spec)
    run_dir = Path(result.workdir)
    assert (run_dir / "experiment_spec.yaml").exists()
    assert (run_dir / "workflow_graph.json").exists()
    assert (run_dir / "execution_replay.json").exists()
    assert (run_dir / "audit_log.jsonl").exists()


def test_generic_creatinine_fallback_writes_clustering_contract(ra, tmp_path: Path, monkeypatch):
    from easyicu.research_agent.pipeline import _generic_v15_task_fallback_code

    cohort = pd.DataFrame({
        "creat_max_24h": [1.0, 1.2, 1.5, 2.2, 2.5, 3.0],
        "creat_median_24h": [1.0, 1.0, 1.1, 1.4, 1.6, 1.8],
        "kdigo_stage_max_24h": [0, 0, 1, 1, 2, 3],
        "sofa2_renal_max_24h": [0, 1, 1, 2, 3, 4],
    })
    cohort_path = tmp_path / "cohort.parquet"
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    cohort.to_parquet(cohort_path, index=False)
    monkeypatch.setenv("COHORT_PARQUET", str(cohort_path))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))

    code = _generic_v15_task_fallback_code("creatinine")
    assert code is not None
    compile(code, "<creatinine_fallback>", "exec")
    exec(code, {"__name__": "__main__"})

    summary = json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))
    assert summary["cluster_count"] >= 2
    assert -1 <= summary["silhouette_score"] <= 1
    assert summary["statistic:cluster_count"] == summary["cluster_count"]
    assert summary["statistic:silhouette_score"] == summary["silhouette_score"]
    assert summary["cluster_characteristics"]
    assert summary["cluster_mortality"]
    assert summary["manifest:clustering_methodology"]["sklearn_required"] is False
    for name in [
        "cluster_characteristics.csv",
        "cluster_mortality.csv",
        "clustering_methodology.json",
        "clustering_algorithm_details.json",
        "clustering_visualization.png",
        "clustering_visualization.svg",
    ]:
        assert (out_dir / name).exists()


def test_generic_clustering_fallback_writes_tables_and_figures(ra, tmp_path: Path, monkeypatch):
    from easyicu.research_agent.pipeline import (
        _generic_v15_task_fallback_code,
        _infer_generic_v15_fallback_key,
    )

    cohort = pd.DataFrame({
        "stay_id": [1, 2, 3, 4, 5, 6],
        "sofa2_max_24h": [1, 2, 4, 7, 9, 11],
        "lactate_max_24h": [1.1, 1.4, 2.8, 3.1, 4.5, 5.0],
        "map_min_24h": [82, 78, 70, 62, 55, 51],
        "hr_max_24h": [88, 92, 104, 115, 122, 130],
        "creat_max_24h": [0.8, 1.0, 1.3, 1.8, 2.1, 2.6],
        "vaso_any_24h": [0, 0, 1, 1, 1, 1],
        "death": [0, 0, 0, 1, 1, 1],
    })
    cohort_path = tmp_path / "cohort.parquet"
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    cohort.to_parquet(cohort_path, index=False)
    monkeypatch.setenv("COHORT_PARQUET", str(cohort_path))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))

    assert _infer_generic_v15_fallback_key(
        "Generate cluster_characteristics and cluster_mortality for SOFA/lactate clustering",
        "KeyError: cluster_labels",
    ) == "clustering"

    code = _generic_v15_task_fallback_code("clustering")
    assert code is not None
    compile(code, "<clustering_fallback>", "exec")
    exec(code, {"__name__": "__main__"})

    summary = json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))
    assert summary["cluster_count"] >= 2
    assert {"sofa2", "lactate", "map", "heart_rate", "creatinine", "vasopressor"} <= set(
        summary["features_used"]
    )
    assert summary["cluster_characteristics"]
    assert summary["cluster_mortality"]
    assert summary["table:cluster_characteristics"].endswith("cluster_characteristics.csv")
    assert summary["table:cluster_mortality"].endswith("cluster_mortality.csv")
    assert summary["figure:clustering_visualization"].endswith("clustering_visualization.png")
    for name in [
        "cluster_labels.csv",
        "cluster_characteristics.csv",
        "cluster_mortality.csv",
        "clustering_methodology.json",
        "clustering_algorithm_details.json",
        "clustering_visualization.png",
        "clustering_visualization.svg",
        "cluster_profiles.png",
        "cluster_profiles.svg",
    ]:
        assert (out_dir / name).exists()
