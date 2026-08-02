"""End-to-end pipeline test with the synthetic SOFA cohort.

This is the integration test the ROADMAP's "mock pipeline must always
pass" rule rests on. If this regresses, the demo (and any reviewer
clicking "run") gets a broken artefact.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient


def _step_record_by_id(records, step_id: str):
    for record in records:
        if record.get("step_id") == step_id:
            return record
    raise AssertionError(f"step record {step_id!r} not found in {records!r}")


def _empty_custom_llm_response(user_prompt: str) -> str:
    """Return the schema-valid empty response for the active test prompt."""

    upper = str(user_prompt or "").upper()
    if "EVERY FINDING MUST INCLUDE" in upper and "RETURN JSON ONLY" in upper:
        return json.dumps({"findings": []})
    return "{}"


def _disable_article_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep focused lifecycle fixtures independent of manuscript completeness."""

    import easyicu.research_agent.agents.core as agent_core
    import easyicu.research_agent.pipeline as pipeline_module
    from easyicu.research_agent.agents.core import PlannerAgent

    original_run = PlannerAgent.run

    def run_without_article_contract(self, context, **kwargs):
        kwargs["enforce_article_contract"] = False
        return original_run(self, context, **kwargs)

    monkeypatch.setattr(PlannerAgent, "run", run_without_article_contract)
    monkeypatch.setattr(
        agent_core,
        "_validate_required_primary_result",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_enforce_advanced_plan_contract",
        lambda *, plan, context, **_kwargs: (plan, []),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_ensure_publication_figure_step_in_plan",
        lambda *, plan, context, force: (plan, []),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_ensure_audit_panel_step_in_plan",
        lambda *, plan, context, **_kwargs: (plan, []),
    )


def _stable_plan_rules(plan: str):
    """Return initial and probe-replan routes for a fixed focused test plan."""

    return [
        ("Produce an ICU-AWARE RESEARCH PLAN as JSON", [plan] * 8),
        ("REVISE THE ICU-AWARE RESEARCH PLAN", [plan] * 8),
    ]


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
    assert {
        "code",
        "log",
        "table",
        "figure",
        "statistic",
    } <= kinds, f"evidence kinds incomplete: {kinds}"
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
    plan = json.loads(Path(result.plan_path).read_text(encoding="utf-8"))
    plan_by_id = {step["step_id"]: step for step in plan["steps"]}
    parent = plan_by_id["04_primary_association"]
    figure_child = plan_by_id["04_primary_association_figure"]
    assert parent["method"] in {"association_analysis", "logistic_regression"}
    assert figure_child["method"] == "visualization"
    assert figure_child["inputs"] == [
        output
        for output in parent["expected_outputs"]
        if output.startswith(
            ("table:", "statistic:", "artifact:", "dataset:", "model:")
        )
    ]
    assert figure_child["inputs"]

    # 3) The bound manuscript should have ZERO ``[evidence missing: …]``
    #    lines (T1.2 acceptance criterion).
    bound = (run_dir / "manuscript_scaffold_bound.md").read_text(encoding="utf-8")
    assert "[evidence missing:" not in bound, (
        "bound manuscript contains unresolved evidence placeholders:\n" + bound
    )
    partial = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    assert partial["runtime_state"]["analysis_family"]
    ok_step_records = [
        record
        for record in manifest["per_step_records"]
        if record.get("step_id") != "00_probe" and record.get("status") == "ok"
    ]
    assert ok_step_records
    for record in ok_step_records:
        assert record.get("evidence_ids"), record
        assert record["interpretation_evidence_id"] in record["evidence_ids"]

    # 4) The deterministic probe should expose score completeness without
    # peeking at outcome-stratum rates or auto-generating a SOFA-specific audit.
    probe_paths = list(run_dir.rglob("probe_summary.json"))
    assert probe_paths, "no probe_summary.json was produced"
    probe = json.loads(probe_paths[0].read_text(encoding="utf-8"))
    assert "outcome_rate" not in probe
    completeness = probe["score_completeness"]
    assert completeness
    sofa_report = next(item for item in completeness if item["variable"] == "sofa2")
    assert sofa_report["completeness"]["n_low_completeness"] > 0


def test_strict_evidence_failure_writes_structured_diagnostic_package(
    ra,
    synthetic_cohort,
    tmp_path: Path,
    monkeypatch,
):
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=ra.MockLLMClient(),
        enable_literature=False,
        evidence_enforcement_mode="strict",
    )

    def _raise_strict_failure(**kwargs):
        raise ra.EvidenceEnforcementError(
            "Manuscript contains 1 numeric value not traceable.",
            detail={"untraced": ["9.99"]},
        )

    monkeypatch.setattr(pipeline, "_run_write_phase", _raise_strict_failure)

    result = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="strict_failure_test",
        database="synthetic",
        target_outcome="death",
    )
    run_dir = Path(result.workdir)
    run_status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    bound = Path(result.manuscript_path).read_text(encoding="utf-8")

    assert "STRICT evidence enforcement failed" in bound
    assert run_status["gates"]["numeric_error_count"] == 1
    assert run_status["gates"]["manuscript_generated"] is False
    assert any(
        finding["validator"] == "manuscript_numeric_auditor"
        and finding["severity"] == "error"
        for finding in manifest["findings"]
    )


def test_writer_failure_does_not_pass_empty_manuscript(
    ra,
    synthetic_cohort,
    tmp_path: Path,
    monkeypatch,
):
    def _raise_writer_failure(self, *, context, evidence_ids, evidence_digest=None):
        raise RuntimeError("local writer endpoint unavailable")

    monkeypatch.setattr(ra.ManuscriptAgent, "run", _raise_writer_failure)

    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=ra.MockLLMClient(),
        enable_literature=False,
    )
    result = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="writer_failure_test",
        database="synthetic",
        target_outcome="death",
    )

    run_dir = Path(result.workdir)
    bound = (run_dir / "manuscript_scaffold_bound.md").read_text(encoding="utf-8")
    critique = json.loads(
        (run_dir / "manuscript_critique.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    run_status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))

    assert "Manuscript scaffold not generated" in bound
    assert critique["status"] == "blocked"
    assert run_status["gates"]["manuscript_generated"] is False
    assert run_status["gates"]["manuscript_ready"] is False
    validators = {finding["validator"] for finding in manifest["findings"]}
    assert "writer_agent" in validators
    assert "evidence_bound_writer" in validators


def test_pipeline_with_clinical_skill(ra, synthetic_cohort, tmp_path: Path):
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    # No case-specific skill ships in the registry anymore; the caller supplies
    # a case-neutral ClinicalSkill object and the mechanism drives the pipeline.
    skill = ra.ClinicalSkill(
        key="generic_exposure_outcome",
        name="Generic exposure → outcome association",
        description="Case-neutral skill for the pipeline smoke test.",
        research_question_template="Is sofa2 associated with death in {database}?",
        target_outcome="death",
        primary_predictor="sofa2",
        expected_variables=["age", "sex", "sofa2", "death"],
    )
    result = pipeline.run(
        cohort=synthetic_cohort,
        cohort_name="synthetic_skill_cohort",
        database="synthetic",
        skill=skill,
    )
    assert result.evidence_count > 0
    plan = json.loads(Path(result.plan_path).read_text(encoding="utf-8"))
    assert plan["steps"], "the skill plan produced no steps"


def test_pipeline_stops_when_hypothesis_blueprint_is_blocked(
    ra,
    tmp_path: Path,
    monkeypatch,
):
    cohort_path = tmp_path / "cohort_no_outcome.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "sofa2": [0, 2, 5],
        }
    ).to_parquet(cohort_path)

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


def test_pipeline_falls_back_when_planner_returns_empty(
    ra, synthetic_cohort, tmp_path: Path
):
    empty_planner = PatternScriptedMockLLMClient(
        [
            ("Produce an ICU-AWARE RESEARCH PLAN as JSON", [""] * 8),
            ("REVISE THE ICU-AWARE RESEARCH PLAN", [""] * 8),
        ],
        default="",
    )
    router = ra.LLMRouter(default=ra.MockLLMClient(), planner=empty_planner)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=router,
        enable_deterministic_planner_fallback=True,
        max_code_repair_attempts=0,
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


def test_pipeline_recovers_when_planner_parses_to_zero_steps(
    ra, synthetic_cohort, tmp_path: Path, monkeypatch
):
    """A planner that returns *valid JSON with 0 steps* (E1 20260611 v4-flash)
    must not run an empty plan: retry the planner, recover on a non-empty reply.
    """
    _disable_article_contract(monkeypatch)
    valid_plan = json.dumps(
        {
            "research_question": "Is admission SOFA-2 associated with ICU mortality?",
            "steps": [
                {
                    "step_id": "01_assoc",
                    "planned_analysis_role": "primary",
                    "intent": "Fit the adjusted association model.",
                    "expected_outputs": ["table:or_table"],
                    "method": "logit",
                }
            ],
            "rationale": "single-step plan",
        }
    )

    flaky = PatternScriptedMockLLMClient(
        [
            (
                "Produce an ICU-AWARE RESEARCH PLAN as JSON",
                ['{"research_question": "q", "steps": []}', valid_plan, valid_plan],
            ),
            ("REVISE THE ICU-AWARE RESEARCH PLAN", [valid_plan] * 8),
        ],
        default=valid_plan,
    )
    router = ra.LLMRouter(default=ra.MockLLMClient(), planner=flaky)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=router,
        enable_deterministic_planner_fallback=True,
        max_code_repair_attempts=0,
    )
    result = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="planner_zero_step_retry",
        database="synthetic",
        target_outcome="death",
    )
    assert len(flaky.calls) >= 2  # the retry happened
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert any(
        f["validator"] == "planner"
        and "retry" in (f.get("detail") or {}).get("generation_mode", "")
        for f in manifest["findings"]
    )


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


def test_pipeline_repairs_failed_generated_code(ra, tmp_path: Path, monkeypatch):
    """A real-LLM style traceback should trigger one coder repair pass."""

    _disable_article_contract(monkeypatch)
    plan = json.dumps(
        {
            "research_question": "Does age describe ICU mortality?",
            "steps": [
                {
                    "step_id": "01_table_one",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Write a compact cohort table.",
                    "inputs": ["death"],
                    "expected_outputs": ["table:table_one"],
                    "method": "descriptive",
                    "icu_rule_refs": ["aggregation_rule_for"],
                }
            ],
            "rationale": "minimal repair test",
        }
    )
    repaired_code = """
import json
import os
import pandas as pd

df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
pd.DataFrame({"n": [int(len(df))]}).to_csv(os.path.join(out, "table_one.csv"), index=False)
summary = {
    "n": int(len(df)),
    "output_files": {"table:table_one": "table_one.csv"},
}
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
print(json.dumps(summary))
"""
    llm = PatternScriptedMockLLMClient(
        [
            *_stable_plan_rules(plan),
            (
                "WRITE THE PYTHON CODE FOR STEP",
                ["raise KeyError('intentional first pass failure')\n"] * 8,
            ),
            ("REPAIR THE PYTHON CODE FOR STEP", [repaired_code] * 8),
            (
                "INTERPRET THE RESULTS OF STEP",
                ["The cohort table was produced {evidence:table_one}."] * 8,
            ),
            (
                "WRITE A MANUSCRIPT SCAFFOLD",
                [
                    "# Title\n\n## Results\n\nThe cohort table was produced "
                    "{evidence:table_one}.\n\n(left to the human author)"
                ]
                * 8,
            ),
        ]
    )

    cohort = pd.DataFrame({"age": [50, 60, 70], "death": [0, 1, 0]})
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
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
    partial = json.loads(
        (Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8")
    )
    record = _step_record_by_id(partial["per_step_records"], "01_table_one")
    assert record["status"] == "ok"
    assert record["code_repair_attempts"] == 1
    assert not [
        f
        for f in manifest["findings"]
        if f["severity"] == "error" and f["validator"] == "runner"
    ]


def test_pipeline_repairs_cross_step_source_status_denominator_drift(
    ra, tmp_path: Path, monkeypatch
):
    """A later Table 1 must preserve a prior explicit source-status lock."""

    _disable_article_contract(monkeypatch)

    def source_lock_script() -> str:
        return """
import json
import os
import pandas as pd

out = os.environ["STEP_OUT_DIR"]
pd.DataFrame({"status": ["valid", "no_source"], "n": [3, 2]}).to_csv(
    os.path.join(out, "source_status_lock.csv"), index=False
)
summary = {
    "missingness": {
        "source_status_counts": {
            "adult_analytic_cohort": {
                "lab_max": {
                    "valid_observed_level_or_value": 3,
                    "no_recorded_source_or_observation": 2,
                    "measured_or_observed_source_with_summary_missing": 0,
                    "contradictory_or_invalid_source_summary": 0,
                }
            }
        }
    }
}
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
print(json.dumps(summary))
"""

    def table_one_script(valid_n: int) -> str:
        return f"""
import json
import os
import pandas as pd

out = os.environ["STEP_OUT_DIR"]
pd.DataFrame({{"variable": ["lab_max"], "n_nonmissing": [{valid_n}]}}).to_csv(
    os.path.join(out, "table_one.csv"), index=False
)
summary = {{
    "measurement_status": {{
        "source_summary": "lab_max",
        "counts": [
            {{"category": "Observed valid", "count": {valid_n}}},
            {{"category": "No source", "count": {5 - valid_n}}},
            {{"category": "Measured but summary missing", "count": 0}},
            {{"category": "Invalid summary", "count": 0}},
            {{"category": "Contradictory status", "count": 0}},
        ],
    }},
    "output_files": {{"table:table_one": "table_one.csv"}},
}}
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
print(json.dumps(summary))
"""

    plan = json.dumps(
        {
            "research_question": "Keep measured lab counts consistent.",
            "steps": [
                {
                    "step_id": "01_source_status_audit",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Record the source-status denominator lock.",
                    "inputs": ["lab_max"],
                    "expected_outputs": [],
                    "method": "descriptive",
                    "icu_rule_refs": ["aggregation_rule_for"],
                },
                {
                    "step_id": "02_table_one",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Build Table 1 without redefining lab validity.",
                    "inputs": ["lab_max"],
                    "expected_outputs": ["table:table_one"],
                    "method": "descriptive",
                    "icu_rule_refs": ["aggregation_rule_for"],
                },
            ],
            "rationale": "minimal cross-step contract repair test",
        }
    )
    llm = PatternScriptedMockLLMClient(
        [
            *_stable_plan_rules(plan),
            (
                "WRITE THE PYTHON CODE FOR STEP 01_SOURCE_STATUS_AUDIT",
                [source_lock_script()] * 8,
            ),
            (
                "WRITE THE PYTHON CODE FOR STEP 02_TABLE_ONE",
                [table_one_script(1)] * 8,
            ),
            (
                "REPAIR THE PYTHON CODE FOR STEP 02_TABLE_ONE",
                [table_one_script(3)] * 8,
            ),
            (
                "INTERPRET THE RESULTS OF STEP",
                ["The requested descriptive output was produced."] * 8,
            ),
            (
                "WRITE A MANUSCRIPT SCAFFOLD",
                ["# Title\n\n## Results\n\nDescriptive outputs were produced.\n"] * 8,
            ),
        ]
    )

    cohort = pd.DataFrame(
        {"lab_max": [1.0, 2.0, 3.0, np.nan, np.nan], "death": [0, 1, 0, 0, 1]}
    )
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        enable_literature=False,
        enable_deterministic_runner_repair=False,
        max_code_repair_attempts=1,
    )
    result = pipeline.run(
        question="Keep measured lab counts consistent.",
        cohort=cohort,
        cohort_name="cross_step_source_status_test",
        database="synthetic",
        target_outcome="death",
    )

    partial = json.loads(
        (Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8")
    )
    record = _step_record_by_id(partial["per_step_records"], "02_table_one")
    assert record["status"] == "ok"
    assert record["code_repair_attempts"] == 1
    assert record["step_summary"]["measurement_status"]["counts"][0]["count"] == 3
    assert not [
        finding
        for finding in record["contract_findings"]
        if finding["severity"] == "error"
        and finding["validator"] == "cross_step_source_status"
    ]


def test_pipeline_repairs_fixed_cohort_drift_in_current_step(
    ra, tmp_path: Path, monkeypatch
):
    """An explicit fixed-cohort promise routes N drift to local code repair."""

    _disable_article_contract(monkeypatch)

    def summary_script(*, cohort_n: int, field: str) -> str:
        return f"""
import json
import os

out = os.environ["STEP_OUT_DIR"]
summary = {{"status": "completed", "{field}": {cohort_n}}}
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
print(json.dumps(summary))
"""

    plan = json.dumps(
        {
            "research_question": "Keep the analytic cohort fixed.",
            "steps": [
                {
                    "step_id": "01_cohort_lock",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Record the completed analytic cohort.",
                    # No ranged raw input: the script below only counts rows, so
                    # declaring one would owe a plausibility receipt it never
                    # computes -- and the step would be blocked before it ever
                    # reached the fixed-cohort repair this test is about.
                    "inputs": ["death"],
                    "expected_outputs": [],
                    "method": "descriptive",
                    "icu_rule_refs": [],
                },
                {
                    "step_id": "02_reconcile",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Keep the completed cohort fixed while reconciling outputs.",
                    "inputs": ["death"],
                    "expected_outputs": [],
                    "method": "data_quality_audit",
                    "icu_rule_refs": [],
                },
            ],
            "rationale": "minimal fixed-cohort repair test",
        }
    )
    llm = PatternScriptedMockLLMClient(
        [
            *_stable_plan_rules(plan),
            (
                "WRITE THE PYTHON CODE FOR STEP 01_COHORT_LOCK",
                [summary_script(cohort_n=5, field="n_total")] * 8,
            ),
            (
                "WRITE THE PYTHON CODE FOR STEP 02_RECONCILE",
                [summary_script(cohort_n=4, field="n_final_cohort")] * 8,
            ),
            (
                "REPAIR THE PYTHON CODE FOR STEP 02_RECONCILE",
                [summary_script(cohort_n=5, field="n_final_cohort")] * 8,
            ),
            (
                "INTERPRET THE RESULTS OF STEP",
                ["The fixed-cohort reconciliation was completed."] * 8,
            ),
            (
                "WRITE A MANUSCRIPT SCAFFOLD",
                ["# Title\n\n## Results\n\nThe cohort was retained.\n"] * 8,
            ),
        ]
    )

    cohort = pd.DataFrame({"age": [40, 50, 60, 70, 80], "death": [0, 1, 0, 1, 0]})
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        enable_literature=False,
        enable_deterministic_runner_repair=False,
        max_code_repair_attempts=1,
    )
    result = pipeline.run(
        question="Keep the analytic cohort fixed.",
        cohort=cohort,
        cohort_name="cross_step_cohort_lock_test",
        database="synthetic",
        target_outcome="death",
    )

    partial = json.loads(
        (Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8")
    )
    record = _step_record_by_id(partial["per_step_records"], "02_reconcile")
    assert record["status"] == "ok"
    assert record["code_repair_attempts"] == 1
    assert record["step_summary"]["n_final_cohort"] == 5
    assert not [
        finding
        for finding in record["contract_findings"]
        if finding["severity"] == "error"
        and finding["validator"] == "cross_step_cohort_lock"
    ]


def test_runtime_crash_after_contract_repair_gets_its_own_repair_budget(
    ra, tmp_path: Path, monkeypatch
):
    """A contract repair that *introduces* a crash must not strand the step.

    Reproduces the E1 20260611 deepseek-v4-flash failure: the primary
    association step ran (rc=0) but failed the effect-size contract; the
    contract repair then produced code that computed the OR and wrote its
    table but crashed on a trailing line (``AttributeError`` on a renamed
    column). With ``max_code_repair_attempts=1`` the single *shared* budget
    was consumed by the contract repair, so the runtime crash fail-closed
    even though the analysis was otherwise valid. Runtime crashes now carry
    their own repair budget, so the traceback gets a repair pass.
    """

    _disable_article_contract(monkeypatch)
    plan = json.dumps(
        {
            "research_question": "Is the exposure associated with mortality?",
            "steps": [
                {
                    "step_id": "05_primary_association",
                    "planned_analysis_role": "primary",
                    "intent": "Estimate the adjusted odds ratio for the exposure.",
                    # See 01_cohort_lock above: the scripts here fabricate the
                    # estimate table outright, so a ranged input would block the
                    # step before the crash-after-contract-repair path is reached.
                    "inputs": ["death"],
                    "expected_outputs": ["statistic:primary_association"],
                    "method": "logistic_regression",
                    "icu_rule_refs": ["aggregation_rule_for"],
                }
            ],
            "rationale": "minimal contract-then-crash repair test",
        }
    )
    initial_code = (
        "import json, os\n"
        "import pandas as pd\n"
        "out = os.environ['STEP_OUT_DIR']\n"
        "pd.DataFrame({'n': [3]})"
        ".to_csv(os.path.join(out, 'primary_association.csv'), index=False)\n"
        "summary = {'n': 3}\n"
        "with open(os.path.join(out, 'step_summary.json'), 'w') as f:\n"
        "    json.dump(summary, f)\n"
        "print(json.dumps(summary))\n"
    )
    crashing_repair = (
        "import os\n"
        "import pandas as pd\n"
        "out = os.environ['STEP_OUT_DIR']\n"
        "pd.DataFrame({'predictor': ['exposure'], 'odds_ratio': [1.47]})"
        ".to_csv(os.path.join(out, 'primary_association.csv'), index=False)\n"
        "class _Row:\n    pass\n"
        "row = _Row()\n"
        "_ = row.log_odds  # AttributeError after the OR is written\n"
    )
    runtime_repair = (
        "import json, os\n"
        "import pandas as pd\n"
        "out = os.environ['STEP_OUT_DIR']\n"
        "pd.DataFrame({'predictor': ['exposure'], 'odds_ratio': [1.47]})"
        ".to_csv(os.path.join(out, 'primary_association.csv'), index=False)\n"
        "with open(os.path.join(out, 'primary_association.json'), 'w') as f:\n"
        "    json.dump({'name': 'primary_association', 'estimate': 1.47, "
        "'effect_scale': 'odds_ratio'}, f)\n"
        "summary = {'primary_or': 1.47, 'odds_ratio': 1.47, "
        "'primary_predictor': 'exposure', "
        "'output_files': "
        "{'statistic:primary_association': 'primary_association.json'}}\n"
        "with open(os.path.join(out, 'step_summary.json'), 'w') as f:\n"
        "    json.dump(summary, f)\n"
        "print(json.dumps(summary))\n"
    )
    llm = PatternScriptedMockLLMClient(
        [
            *_stable_plan_rules(plan),
            ("WRITE THE PYTHON CODE FOR STEP", [initial_code] * 8),
            (
                "REPAIR THE PYTHON CODE FOR STEP",
                [crashing_repair, crashing_repair, runtime_repair, runtime_repair],
            ),
            (
                "INTERPRET THE RESULTS OF STEP",
                [
                    "The adjusted odds ratio was estimated "
                    "{evidence:primary_association}."
                ]
                * 8,
            ),
            (
                "WRITE A MANUSCRIPT SCAFFOLD",
                [
                    "# Title\n\n## Results\n\nThe adjusted odds ratio was estimated "
                    "{evidence:primary_association}.\n\n(left to the human author)"
                ]
                * 8,
            ),
        ]
    )

    cohort = pd.DataFrame({"age": [50, 60, 70], "death": [0, 1, 0]})
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        enable_literature=False,
        # Pin the tight single-attempt budget so the test exercises the exact
        # starvation case: under the old shared counter the contract repair
        # consumed the only attempt and the crash fail-closed. The runtime
        # crash must get its own attempt.
        max_code_repair_attempts=1,
        # Force the LLM repair path so the runtime crash exercises the budget,
        # not a deterministic pattern repair.
        enable_deterministic_runner_repair=False,
    )
    result = pipeline.run(
        question="Is the exposure associated with mortality?",
        cohort=cohort,
        cohort_name="contract_then_crash_test",
        database="synthetic",
        target_outcome="death",
    )

    partial = json.loads(
        (Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8")
    )
    record = _step_record_by_id(partial["per_step_records"], "05_primary_association")
    # The step recovered (no fail-closed) and used a contract repair *and* a
    # separate runtime repair.
    assert record["status"] == "ok", json.dumps(record, indent=2, sort_keys=True)
    assert record["code_repair_attempts"] == 2
    assert record.get("runtime_repair_attempts") == 1
    assert record["step_llm_repair_attempts"] == 2
    assert record["step_llm_repair_classes"] == ["contract", "runtime"]


def test_method_substitution_contract_repair_is_blocked_when_budget_is_zero(
    ra,
    tmp_path: Path,
    monkeypatch,
):
    _disable_article_contract(monkeypatch)
    plan = json.dumps(
        {
            "research_question": (
                "Estimate the adjusted association between Sepsis-3 and mortality."
            ),
            "steps": [
                {
                    "step_id": "05_primary_association",
                    "planned_analysis_role": "primary",
                    "intent": (
                        "Estimate the adjusted odds ratio for Sepsis-3 and mortality."
                    ),
                    # The overadjustment rule reads the *covariates the summary
                    # reports*, not the declared inputs; the script fabricates its
                    # table and reads no column, so declaring the ranged raw
                    # inputs would only block it before the rule can fire.
                    "inputs": ["sepsis3", "death"],
                    "expected_outputs": ["statistic:primary_association"],
                    # `logistic` alone is not one of the effect-method heads that
                    # grant effect authority, and without that authority the
                    # overadjustment auditor never runs -- so the step would fail
                    # the product contract instead of the rule under test.
                    "method": "logistic_regression",
                    "icu_rule_refs": ["no_overadjustment_for_exposure_constituents"],
                }
            ],
            "rationale": "minimal deterministic overadjustment repair test",
        }
    )
    overadjusted_code = """
import json
import os
import pandas as pd

out = os.environ["STEP_OUT_DIR"]
x_cols = [
    "sepsis3",
    "age_per_10y",
    "map_min_per_10mmhg",
    "map_min_missing_indicator",
]
x_cols = list(dict.fromkeys(x_cols))

pd.DataFrame({
    "term": x_cols,
    "estimate": [1.0] * len(x_cols),
    "odds_ratio": [1.0] * len(x_cols),
}).to_csv(os.path.join(out, "adjusted_association_death.csv"), index=False)

summary = {
    "primary_predictor": "sepsis3",
    "primary_or": 1.0,
    "odds_ratio": 1.0,
    "primary_adjusted_association": {
        "term": "sepsis3",
        "estimate": 1.0,
        "effect_scale": "adjusted odds ratio",
        "covariates": x_cols,
    },
}
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
print(json.dumps(summary))
"""
    llm = PatternScriptedMockLLMClient(
        [
            *_stable_plan_rules(plan),
            ("WRITE THE PYTHON CODE FOR STEP", [overadjusted_code] * 8),
            (
                "REPAIR THE PYTHON CODE FOR STEP",
                [AssertionError("LLM repair should not be called")],
            ),
            (
                "INTERPRET THE RESULTS OF STEP",
                [
                    "The adjusted odds ratio was estimated "
                    "{evidence:adjusted_association_death}."
                ]
                * 8,
            ),
            (
                "WRITE A MANUSCRIPT SCAFFOLD",
                ["# Title\n\n## Results\n\nAnalysis stopped after execution."] * 8,
            ),
        ]
    )

    cohort = pd.DataFrame(
        {
            "sepsis3": [0, 1, 0, 1],
            "death": [0, 1, 0, 1],
            "age": [50, 60, 70, 80],
            "map_min": [72, 61, 80, 58],
        }
    )
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        enable_literature=False,
        max_code_repair_attempts=0,
    )

    result = pipeline.run(
        question="Is Sepsis-3 associated with mortality?",
        cohort=cohort,
        cohort_name="overadjustment_budget_zero_test",
        database="synthetic",
        target_outcome="death",
        primary_exposure="sepsis3",
        stop_after_analysis=True,
    )

    run_dir = Path(result.workdir)
    partial = json.loads((run_dir / "manifest_partial.json").read_text("utf-8"))
    record = _step_record_by_id(partial["per_step_records"], "05_primary_association")
    covariates = record["step_summary"]["primary_adjusted_association"]["covariates"]
    assert record["status"] == "contract_failed"
    assert record.get("runner_repair") is None
    assert record.get("code_repair_attempts", 0) == 0
    assert "map_min_per_10mmhg" in covariates
    assert "map_min_missing_indicator" in covariates
    assert [f for f in record.get("contract_findings", []) if f["severity"] == "error"]

    repair_ledger = json.loads(
        (run_dir / "repairs_applied.json").read_text(encoding="utf-8")
    )
    blocked = [
        item
        for item in repair_ledger["repairs"]
        if item["repair_id"] == "drop_overadjustment_covariates_v1"
    ]
    assert blocked
    assert blocked[-1]["outcome"] == "blocked_by_automatic_repair_policy"


def test_generic_association_figure_coder_failure_fails_closed(
    ra, tmp_path: Path, monkeypatch
):
    _disable_article_contract(monkeypatch)
    plan = json.dumps(
        {
            "research_question": (
                "Estimate an association and render its publication figure."
            ),
            "steps": [
                {
                    "step_id": "03_primary_association",
                    "planned_analysis_role": "primary",
                    "intent": "Estimate the adjusted odds ratio.",
                    "inputs": ["sepsis3", "death"],
                    "expected_outputs": ["statistic:primary_or"],
                    "method": "logistic_regression",
                    "icu_rule_refs": [],
                },
                {
                    "step_id": "03_primary_association_figure",
                    "planned_analysis_role": "auxiliary",
                    "intent": (
                        "Render the publication figure(s) declared by "
                        "step '03_primary_association'."
                    ),
                    "inputs": ["statistic:primary_or"],
                    "expected_outputs": ["figure:publication_figure"],
                    "method": "publication_figure_generation",
                    "icu_rule_refs": [],
                },
            ],
            "rationale": "minimal figure rescue test",
        }
    )
    primary_code = """
import json
import os
import pandas as pd

out = os.environ["STEP_OUT_DIR"]
pd.DataFrame({
    "term": ["sepsis3", "age_per_10y"],
    "reader_label": ["Sepsis-3 positive vs negative", "Age, per 10 years"],
    "effect_scale": ["adjusted odds ratio", "adjusted odds ratio"],
    "estimate": [1.24, 1.08],
    "ci_low": [1.11, 1.02],
    "ci_high": [1.38, 1.15],
    "p_value": [0.001, 0.02],
}).to_csv(os.path.join(out, "adjusted_association_death.csv"), index=False)
summary = {
    "primary_predictor": "sepsis3",
    "primary_or": 1.24,
    "primary_adjusted_association": {
        "term": "sepsis3",
        "estimate": 1.24,
        "ci_low": 1.11,
        "ci_high": 1.38,
        "effect_scale": "adjusted odds ratio",
        "covariates": ["sepsis3", "age_per_10y"],
    },
}
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
print(json.dumps(summary))
"""
    llm = PatternScriptedMockLLMClient(
        [
            *_stable_plan_rules(plan),
            (
                "WRITE THE PYTHON CODE FOR STEP",
                [
                    primary_code,
                    RuntimeError("simulated local LLM outage for figure coder"),
                ],
            ),
            (
                "INTERPRET THE RESULTS OF STEP",
                ["The step completed with registered evidence."] * 8,
            ),
            (
                "WRITE A MANUSCRIPT SCAFFOLD",
                ["# Title\n\n## Results\n\nAnalysis stopped after execution."] * 8,
            ),
        ]
    )

    cohort = pd.DataFrame(
        {
            "sepsis3": [0, 1, 0, 1],
            "death": [0, 1, 0, 1],
            "age": [50, 60, 70, 80],
        }
    )
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        enable_literature=False,
        enable_llm_concept_audit=False,
    )

    result = pipeline.run(
        question="Is Sepsis-3 associated with mortality?",
        cohort=cohort,
        cohort_name="figure_rescue_test",
        database="synthetic",
        target_outcome="death",
        primary_exposure="sepsis3",
        stop_after_analysis=True,
    )

    run_dir = Path(result.workdir)
    out_dir = run_dir / "steps" / "03_primary_association_figure" / "outputs"
    partial = json.loads((run_dir / "manifest_partial.json").read_text("utf-8"))
    record = _step_record_by_id(
        partial["per_step_records"], "03_primary_association_figure"
    )
    parent_record = _step_record_by_id(
        partial["per_step_records"], "03_primary_association"
    )
    assert parent_record["status"] == "ok"
    assert record["status"] == "coder_failed"
    assert "deterministic_code_fallback" not in record
    # One call creates the parent analysis; at least one later call still gives
    # the agent its declared figure step before failing closed on the outage.
    code_calls = [
        messages
        for messages, _kwargs in llm.calls
        if any(
            message.role == "user"
            and "WRITE THE PYTHON CODE FOR STEP" in message.content.upper()
            for message in messages
        )
    ]
    assert len(code_calls) >= 2
    assert not out_dir.exists() or not list(out_dir.glob("publication_figure*"))


@pytest.mark.parametrize(
    ("failing_status", "expected_code_calls", "error_pattern"),
    [
        (
            "initial_generation_pending",
            0,
            "reservation could not be checkpointed",
        ),
        (
            "candidate_checkpointed",
            1,
            "candidate authority could not be checkpointed",
        ),
    ],
)
def test_initial_authority_checkpoint_io_failure_never_enters_code_fallback(
    ra,
    tmp_path: Path,
    monkeypatch,
    failing_status: str,
    expected_code_calls: int,
    error_pattern: str,
):
    import re

    from easyicu.research_agent.execution import phase as pipeline_execute
    from easyicu.research_agent.authority.step_runtime import (
        StepAuthorityRuntimeError,
    )

    _disable_article_contract(monkeypatch)
    plan = json.dumps(
        {
            "research_question": "Summarize the locked ICU cohort.",
            "steps": [
                {
                    "step_id": "01_summary",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Produce the declared cohort summary.",
                    "inputs": ["stay_id"],
                    "expected_outputs": ["table:cohort_summary"],
                    "method": "descriptive_summary",
                    "icu_rule_refs": [],
                }
            ],
            "rationale": "checkpoint integrity regression",
        }
    )
    valid_code = """
import json
import os
import pandas as pd

df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
pd.DataFrame({"n": [len(df)]}).to_csv(
    os.path.join(out, "cohort_summary.csv"), index=False
)
summary = {
    "n": len(df),
    "output_files": [
        {"kind": "table", "name": "cohort_summary", "path": "cohort_summary.csv"}
    ],
}
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle)
"""
    llm = PatternScriptedMockLLMClient(
        [
            *_stable_plan_rules(plan),
            ("WRITE THE PYTHON CODE FOR STEP", [valid_code] * 8),
        ]
    )

    original_write = pipeline_execute.write_run_checkpoint

    def fail_target_checkpoint(path, payload):  # noqa: ANN001, ANN202
        statuses = {
            str(record.get("status") or "")
            for record in payload.get("per_step_records", [])
            if isinstance(record, dict)
        }
        if failing_status in statuses:
            raise OSError(f"simulated {failing_status} checkpoint failure")
        return original_write(path, payload)

    monkeypatch.setattr(
        pipeline_execute,
        "write_run_checkpoint",
        fail_target_checkpoint,
    )
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
    )

    result = pipeline.run(
        question="Summarize the locked ICU cohort.",
        cohort=pd.DataFrame({"stay_id": [1, 2]}),
        cohort_name="checkpoint_failure_test",
        database="synthetic",
        stop_after_analysis=True,
    )

    # ``run`` no longer re-raises: 89b04b1 made an unexpected step exception end
    # the run fail-closed instead of escaping it, so the run stays sealable and
    # the traceback is persisted. That retired the transport this test used, not
    # the properties it protected, so both are asserted at their current owners.
    #
    # Property 1 -- the failure is *recorded*, with its typed reason, against the
    # step that raised. A generic message here would mean the run is sealable but
    # undiagnosable.
    workdir = Path(str(result.workdir))
    manifest = json.loads((workdir / "manifest.json").read_text(encoding="utf-8"))
    records = {
        record["step_id"]: record
        for record in manifest["per_step_records"]
        if isinstance(record, dict) and record.get("step_id")
    }
    assert records["01_summary"]["status"] == "execution_raised"
    recorded_error = str(records["01_summary"].get("error") or "")
    assert StepAuthorityRuntimeError.__name__ in recorded_error, recorded_error
    assert re.search(error_pattern, recorded_error), recorded_error

    # Property 2 -- the run is floored, so nothing downstream can read a
    # checkpoint-failed run as a result. ``manuscript_path`` is still written
    # (a scaffold always is); ``status`` is what says it carries no finding.
    status = json.loads((workdir / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "diagnostic_only"
    assert status["strict_fail_closed"] is True
    assert status["gates"]["execution_complete"] is False
    assert status["gates"]["completed_step_count"] == 0

    code_calls = [
        messages
        for messages, _kwargs in llm.calls
        if any(
            message.role == "user"
            and "WRITE THE PYTHON CODE FOR STEP" in message.content.upper()
            for message in messages
        )
    ]
    assert len(code_calls) == expected_code_calls


def test_promote_prior_publication_bundle_copies_real_figure_exports(tmp_path: Path):
    from easyicu.research_agent.pipeline import _promote_prior_publication_bundle

    run_dir = tmp_path / "run"
    source_dir = run_dir / "steps" / "05_primary_association" / "outputs"
    target_dir = run_dir / "steps" / "06_publication_figure_generation" / "outputs"
    source_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)

    (source_dir / "primary_association_curve.png").write_bytes(b"png")
    (source_dir / "primary_association_curve.svg").write_text(
        "<svg><text>A</text></svg>", encoding="utf-8"
    )
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


def test_split_figure_step_never_promotes_an_unrelated_prior_bundle(
    tmp_path: Path,
):
    from easyicu.research_agent.pipeline import _promote_prior_publication_bundle

    run_dir = tmp_path / "run"
    prior = run_dir / "steps" / "01_cohort_flow_figure" / "outputs"
    parent = run_dir / "steps" / "04_absolute_risk_context" / "outputs"
    target = run_dir / "steps" / "04_absolute_risk_context_figure" / "outputs"
    prior.mkdir(parents=True)
    parent.mkdir(parents=True)
    target.mkdir(parents=True)
    (prior / "publication_figure.png").write_bytes(b"cohort-flow")
    (prior / "publication_figure.figure_contract.json").write_text(
        json.dumps(
            {
                "figure_id": "publication_figure",
                "panels": [
                    {"panel_id": "A", "role": "overview"},
                    {"panel_id": "B", "role": "audit"},
                ],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"estimate_type": ["outcome_risk"]}).to_csv(
        parent / "exposure_outcome_summary.csv", index=False
    )

    repair = _promote_prior_publication_bundle(
        run_dir=run_dir,
        current_step_id="04_absolute_risk_context_figure",
        out_dir=target,
    )

    assert repair is None
    assert not (target / "publication_figure.png").exists()


def test_promote_prior_publication_bundle_filters_roles_for_primary_results(
    tmp_path: Path,
):
    from easyicu.research_agent.pipeline import _promote_prior_publication_bundle

    run_dir = tmp_path / "run"
    cohort_dir = run_dir / "steps" / "02_cohort_overlap_figure" / "outputs"
    target_dir = run_dir / "steps" / "03_primary_results_figure" / "outputs"
    cohort_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)

    (cohort_dir / "publication_figure.png").write_bytes(b"cohort")
    (cohort_dir / "publication_figure.figure_contract.json").write_text(
        json.dumps(
            {
                "figure_id": "publication_figure",
                "panels": [
                    {"panel_id": "A", "role": "overview"},
                    {"panel_id": "B", "role": "audit"},
                ],
            }
        ),
        encoding="utf-8",
    )

    repair = _promote_prior_publication_bundle(
        run_dir=run_dir,
        current_step_id="03_primary_results_figure",
        out_dir=target_dir,
        required_roles=("descriptive_result", "primary_estimand"),
    )
    assert repair is None
    assert not (target_dir / "publication_figure.png").exists()

    association_dir = run_dir / "steps" / "03_primary_association_figure" / "outputs"
    association_dir.mkdir(parents=True)
    (association_dir / "publication_figure.png").write_bytes(b"association")
    (association_dir / "publication_figure.figure_contract.json").write_text(
        json.dumps(
            {
                "figure_id": "publication_figure",
                "panels": [
                    {"panel_id": "A", "role": "primary_estimand"},
                ],
            }
        ),
        encoding="utf-8",
    )

    repair = _promote_prior_publication_bundle(
        run_dir=run_dir,
        current_step_id="03_primary_results_figure",
        out_dir=target_dir,
        required_roles=("descriptive_result", "primary_estimand"),
    )
    assert repair == "publication_bundle_promote_v1"
    assert (target_dir / "publication_figure.png").read_bytes() == b"association"


def test_pipeline_does_not_block_or_repair_advisory_ordinal_mean(
    ra, tmp_path: Path, monkeypatch
):
    """Impartiality: a script that computes ``.mean()`` of an ordinal SOFA
    score (here inside a generic describe-style summary that ALSO reports the
    level distribution) must NOT hard-block or trigger an auto-repair that
    imposes median/max over the agent's choice. The caution is recorded as a
    WARNING and the step completes. This is the regression guard for the M1
    false-degradation, where a single helper-level ``.mean()`` dragged an
    otherwise-correct ordinal analysis down to ``diagnostic_only``.
    """

    # `sofa2` carries a plausibility range, and unlike the fallback fixtures
    # below both scripts here really do read it -- so the step genuinely owes a
    # flag-only receipt.  Both drafts are wrapped in the host's own receipt
    # block (the same one the offline mock provider appends) rather than a
    # hand-copied literal, so the fixture cannot drift away from the contract it
    # is meant to satisfy.
    from easyicu.research_agent.providers.mocks import _with_mock_plausibility_receipt

    _disable_article_contract(monkeypatch)
    plan = json.dumps(
        {
            "research_question": "Does SOFA describe ICU mortality?",
            "steps": [
                {
                    "step_id": "04_primary_association",
                    "planned_analysis_role": "primary",
                    "intent": "Assess SOFA-2 and mortality.",
                    "inputs": ["sofa2", "death"],
                    "expected_outputs": ["table:primary_association"],
                    "method": "regression",
                    "icu_rule_refs": ["aggregation_rule_for"],
                }
            ],
            "rationale": "minimal advisory-ordinal-mean test",
        }
    )
    repaired_code = """
import json
import os
from pathlib import Path

import pandas as pd

df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
out_dir = Path(out)
pd.DataFrame({
    "variable": ["sofa2"],
    "median": [float(df["sofa2"].median())],
}).to_csv(os.path.join(out, "primary_association.csv"), index=False)
summary = {
    "predictor": "sofa2",
    "sofa2_median": float(df["sofa2"].median()),
    "primary_or": 1.0,
    "output_files": {"table:primary_association": "primary_association.csv"},
}
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
print(json.dumps(summary))
"""
    initial_code = """
import json
import os
from pathlib import Path

import pandas as pd
df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
out_dir = Path(out)

levels = df["sofa2"].value_counts().sort_index()
pd.DataFrame({
    "sofa2_level": levels.index,
    "n": levels.values,
}).to_csv(os.path.join(out, "primary_association.csv"), index=False)
summary = {
    "predictor": "sofa2",
    "sofa2_level_distribution": {int(k): int(v) for k, v in levels.items()},
    "sofa2_supplementary_mean": float(df["sofa2"].mean()),
    "primary_or": 1.0,
    "output_files": {"table:primary_association": "primary_association.csv"},
}
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
print(json.dumps(summary))
"""
    initial_code = _with_mock_plausibility_receipt(initial_code)
    repaired_code = _with_mock_plausibility_receipt(repaired_code)
    llm = PatternScriptedMockLLMClient(
        [
            *_stable_plan_rules(plan),
            ("WRITE THE PYTHON CODE FOR STEP", [initial_code] * 8),
            ("REPAIR THE PYTHON CODE FOR STEP", [repaired_code] * 8),
            (
                "INTERPRET THE RESULTS OF STEP",
                [
                    "The repaired table was produced "
                    "{evidence:primary_association_table}."
                ]
                * 8,
            ),
            (
                "WRITE A MANUSCRIPT SCAFFOLD",
                [
                    "# Title\n\n## Results\n\nThe repaired table was produced "
                    "{evidence:primary_association_table}.\n\n(left to the human author)"
                ]
                * 8,
            ),
        ]
    )

    cohort = pd.DataFrame({"sofa2": [0, 1, 3, 4], "death": [1, 0, 0, 1]})
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
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
    partial = json.loads(
        (Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8")
    )
    record = _step_record_by_id(partial["per_step_records"], "04_primary_association")
    # The step completes without being blocked and without an auto-repair that
    # would impose a different aggregation over the agent's choice.
    assert record["status"] == "ok"
    assert record.get("concept_repair_attempts", 0) == 0
    assert record.get("status") != "blocked_by_concept_audit"
    # No forbidden-aggregation finding is escalated to a blocking error...
    assert not [
        f
        for f in manifest["findings"]
        if f["severity"] == "error" and f["validator"] == "concept_usage_auditor"
    ]
    # ...but the advisory caution is still surfaced for the reviewer.
    assert any(
        f["validator"] == "concept_usage_auditor"
        and f["severity"] == "warning"
        and ("ordinal" in f["message"].lower() or "sofa" in f["message"].lower())
        for f in record.get("usage_findings", [])
    ), record.get("usage_findings")


def test_pipeline_falls_back_to_deterministic_code_after_repair_failure(
    ra, tmp_path: Path, monkeypatch
):
    """If hosted-model code and its repair both fail, use mock-safe code."""

    _disable_article_contract(monkeypatch)
    plan = json.dumps(
        {
            "research_question": "Is SOFA associated with ICU mortality?",
            "steps": [
                {
                    "step_id": "01_table_one",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Produce a Table 1 cohort summary.",
                    # The draft below raises immediately and reads nothing, so a
                    # ranged raw input here would be refused by the plausibility
                    # preflight and the step would never reach the *runtime*
                    # failure whose repair this test is about.
                    "inputs": ["death"],
                    "expected_outputs": ["table:table_one"],
                    "method": "descriptive",
                    "icu_rule_refs": ["aggregation_rule_for"],
                }
            ],
            "rationale": "minimal fallback test",
        }
    )
    broken_code = "import os\nraise RuntimeError('still broken')\n"
    llm = PatternScriptedMockLLMClient(
        [
            *_stable_plan_rules(plan),
            ("WRITE THE PYTHON CODE FOR STEP", [broken_code] * 8),
            ("REPAIR THE PYTHON CODE FOR STEP", [broken_code] * 8),
            (
                "INTERPRET THE RESULTS OF STEP",
                ["The fallback table was produced {evidence:table_one}."] * 8,
            ),
            (
                "WRITE A MANUSCRIPT SCAFFOLD",
                [
                    "# Title\n\n## Results\n\nThe fallback table was produced "
                    "{evidence:table_one}.\n\n(left to the human author)"
                ]
                * 8,
            ),
        ]
    )

    cohort = pd.DataFrame({"sofa2": [0, 1, 3, 4], "death": [1, 0, 0, 1]})
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
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

    partial = json.loads(
        (Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8")
    )
    record = _step_record_by_id(partial["per_step_records"], "01_table_one")
    assert record["status"] == "ok"
    # The repair provider returned the identical known-broken script. The
    # repair layer now detects that no-op without rerunning it, so the causal
    # fallback reason is the failed repair rather than another execution.
    assert record["deterministic_code_fallback"] == "repair_failed"


def test_pipeline_falls_back_when_repair_model_call_fails(
    ra, tmp_path: Path, monkeypatch
):
    """A provider 429 during repair should not strand the whole step."""

    _disable_article_contract(monkeypatch)
    plan = json.dumps(
        {
            "research_question": "Is SOFA associated with ICU mortality?",
            "steps": [
                {
                    "step_id": "01_table_one",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Produce a Table 1 cohort summary.",
                    "inputs": ["death"],
                    "expected_outputs": ["table:table_one"],
                    "method": "descriptive",
                    "icu_rule_refs": ["aggregation_rule_for"],
                }
            ],
            "rationale": "minimal repair-failure fallback test",
        }
    )
    llm = PatternScriptedMockLLMClient(
        [
            *_stable_plan_rules(plan),
            (
                "WRITE THE PYTHON CODE FOR STEP",
                ["raise RuntimeError('broken first draft')\n"] * 8,
            ),
            (
                "REPAIR THE PYTHON CODE FOR STEP",
                [RuntimeError("provider rate limited")],
            ),
            (
                "INTERPRET THE RESULTS OF STEP",
                ["The fallback table was produced {evidence:table_one}."] * 8,
            ),
            (
                "WRITE A MANUSCRIPT SCAFFOLD",
                [
                    "# Title\n\n## Results\n\nThe fallback table was produced "
                    "{evidence:table_one}.\n\n(left to the human author)"
                ]
                * 8,
            ),
        ]
    )

    cohort = pd.DataFrame({"sofa2": [0, 1, 3, 4], "death": [1, 0, 0, 1]})
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
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

    partial = json.loads(
        (Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8")
    )
    record = _step_record_by_id(partial["per_step_records"], "01_table_one")
    assert record["status"] == "ok"
    assert record["deterministic_code_fallback"] == "repair_failed"


def test_pipeline_falls_back_when_successful_script_writes_no_artefacts(
    ra, tmp_path: Path, monkeypatch
):
    """Exit-code 0 with an empty output dir is not a usable analysis step."""

    _disable_article_contract(monkeypatch)
    plan = json.dumps(
        {
            "research_question": "Is SOFA associated with ICU mortality?",
            "steps": [
                {
                    "step_id": "01_table_one",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Produce a Table 1 cohort summary.",
                    "inputs": ["death"],
                    "expected_outputs": ["table:table_one"],
                    "method": "descriptive",
                    "icu_rule_refs": ["aggregation_rule_for"],
                }
            ],
            "rationale": "minimal no-artefact fallback test",
        }
    )
    llm = PatternScriptedMockLLMClient(
        [
            *_stable_plan_rules(plan),
            (
                "WRITE THE PYTHON CODE FOR STEP",
                ["print('I forgot to write outputs')\n"] * 8,
            ),
            (
                "INTERPRET THE RESULTS OF STEP",
                ["The fallback table was produced {evidence:table_one}."] * 8,
            ),
            (
                "WRITE A MANUSCRIPT SCAFFOLD",
                [
                    "# Title\n\n## Results\n\nThe fallback table was produced "
                    "{evidence:table_one}.\n\n(left to the human author)"
                ]
                * 8,
            ),
        ]
    )

    cohort = pd.DataFrame({"sofa2": [0, 1, 3, 4], "death": [1, 0, 0, 1]})
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
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

    partial = json.loads(
        (Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8")
    )
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
    assert any(
        e.get("stage") == "step" and e.get("status") == "complete" for e in events
    )
    assert any(
        e.get("stage") == "pause" and e.get("status") == "paused" for e in events
    )


def test_mock_planner_honours_sofa2_when_sofa_is_also_present(
    ra, synthetic_cohort, tmp_path: Path
):
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
    assert "05_outcome_stratification" not in by_id


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
            pd.DataFrame(
                {
                    "stay_id": range(1, 81),
                    "age": [60 + (i % 20) for i in range(80)],
                    "kdigo_stage": [i % 4 for i in range(80)],
                    "creat": [0.8 + 0.2 * (i % 4) for i in range(80)],
                    "death": [1 if i % 5 == 0 else 0 for i in range(80)],
                }
            ),
        ),
        (
            "Is any-vasopressor exposure within the first 24h associated with ICU mortality?",
            "vaso",
            pd.DataFrame(
                {
                    "stay_id": range(1, 81),
                    "age": [60 + (i % 20) for i in range(80)],
                    "vaso": [i % 2 for i in range(80)],
                    "death": [1 if i % 5 == 0 else 0 for i in range(80)],
                }
            ),
        ),
    ]

    for question, predictor, cohort in cases:
        pipeline = ra.ResearchAgentPipeline(
            workdir=tmp_path / predictor, llm=ra.MockLLMClient()
        )
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


def test_mock_planner_skips_table_one_for_minimal_association_question(
    ra, tmp_path: Path
):
    """A narrow association question should not force a cohort-summary step."""
    cohort = pd.DataFrame(
        {
            "stay_id": range(1, 81),
            "gcs": [15 - (i % 6) for i in range(80)],
            "death": [1 if i % 7 == 0 else 0 for i in range(80)],
        }
    )

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


def test_mock_planner_uses_quality_only_plan_when_question_is_data_audit(
    ra, tmp_path: Path
):
    """Data-quality questions should not silently expand into effect-estimation steps."""
    cohort = pd.DataFrame(
        {
            "stay_id": range(1, 61),
            "bili": [None if i % 5 == 0 else 0.8 + 0.1 * (i % 4) for i in range(60)],
            "vaso": [None if i % 4 == 0 else int(i % 3 == 0) for i in range(60)],
            "death": [1 if i % 6 == 0 else 0 for i in range(60)],
        }
    )

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
    assert all(sid.startswith("03_missingness_audit") for sid in step_ids), step_ids


def test_pipeline_replicate_writes_cross_database_comparison(ra, tmp_path: Path):
    cohorts = {
        "miiv": pd.DataFrame(
            {
                "stay_id": range(1, 31),
                "age": [60 + (i % 8) for i in range(30)],
                "sofa2": [i % 6 for i in range(30)],
                "death": [1 if i % 5 == 0 else 0 for i in range(30)],
            }
        ),
        "eicu": pd.DataFrame(
            {
                "stay_id": range(1, 31),
                "age": [58 + (i % 9) for i in range(30)],
                "sofa2": [i % 5 for i in range(30)],
                "death": [1 if i % 6 == 0 else 0 for i in range(30)],
            }
        ),
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
        cohort=pd.DataFrame(
            {
                "stay_id": [1, 2, 3, 4],
                "biomarker_x": [1.0, 2.0, 3.0, 4.0],
                "map_min_24h": [70, 65, 60, 55],
                "vaso_any_24h": [0, 0, 1, 1],
                "death": [0, 0, 1, 1],
            }
        ),
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


def test_survival_analysis_type_drives_km_figure_contract(ra):
    """A declared Cox owner receives contracts without losing agent ownership."""
    from easyicu.research_agent import pipeline as pipeline_mod

    ctx = ra.build_research_context(
        research_question=(
            "Time-to-event survival of 28-day mortality with Cox proportional "
            "hazards stratified by SOFA-2 band."
        ),
        cohort=pd.DataFrame(
            {
                "stay_id": [1, 2, 3, 4],
                "sofa2_band": [1, 2, 3, 4],
                "followup_days": [5, 28, 12, 3],
                "death": [0, 1, 0, 1],
            }
        ),
        cohort_name="c",
        database="miiv",
        target_outcome="death",
    )
    plan = ra.schema.AnalysisPlan(
        research_question=ctx.research_question,
        analysis_type="survival",
        steps=[
            ra.schema.AnalysisStep(
                step_id="03_cox_model",
                intent="Fit a Cox proportional-hazards model for 28-day mortality.",
                inputs=["sofa2_band", "followup_days", "death"],
                expected_outputs=["table:hr"],
                method="cox_proportional_hazards",
            )
        ],
    )

    revised, findings = pipeline_mod._enforce_advanced_plan_contract(
        plan=plan,
        context=ctx,
    )

    assert any(f.detail.get("family") == "survival" for f in findings), findings
    assert [step.step_id for step in revised.steps] == ["03_cox_model"]
    surv_step = revised.steps[0]
    assert surv_step.method == "cox_proportional_hazards"
    assert "figure:survival_curves" in surv_step.expected_outputs
    assert "table:cox_summary" in surv_step.expected_outputs


def test_normalise_contract_family_bridges_registry_keys(ra):
    """The alias layer must map registry analysis_type keys onto contract
    buckets so the stamped plan.analysis_type drives figure enforcement."""
    from easyicu.research_agent import plan_utils

    assert plan_utils._normalise_contract_family("survival") == "survival"
    assert (
        plan_utils._normalise_contract_family("trajectory_clustering") == "clustering"
    )
    # Result-bearing families that own their bucket pass through identically.
    for key in (
        "dynamic_prediction",
        "causal_inference",
        "treatment_response",
        "validation",
    ):
        assert plan_utils._normalise_contract_family(key) == key
    # Families without a figure/metric contract fall back to the heuristic.
    assert plan_utils._normalise_contract_family("association_study") == ""
    assert plan_utils._normalise_contract_family("descriptive_epidemiology") == ""
    assert plan_utils._normalise_contract_family("multimodal") == ""
    assert plan_utils._normalise_contract_family(None) == ""
    # Legacy contract-bucket words still pass through unchanged.
    assert plan_utils._normalise_contract_family("clustering") == "clustering"


@pytest.mark.parametrize(
    "analysis_type, question, step_intent, method, figure_tag",
    [
        (
            "dynamic_prediction",
            "Time-updated dynamic prediction of deterioration over rolling horizons.",
            "Build a time-varying landmark prediction at each horizon.",
            "landmark_model",
            "figure:time_varying_discrimination",
        ),
        (
            "causal_inference",
            "Causal effect of early vasopressors using propensity-score IPW.",
            "Estimate IPW propensity-weighted causal effect with covariate balance.",
            "ipw",
            "figure:covariate_balance",
        ),
        (
            "treatment_response",
            "Heterogeneous treatment response: who are the responders to steroids?",
            "Test effect modification and summarize responder subgroups.",
            "interaction_model",
            "figure:subgroup_forest",
        ),
        (
            "validation",
            "External validation and transportability of the SOFA score.",
            "Externally validate score discrimination and calibration.",
            "external_validation",
            "figure:external_validation",
        ),
    ],
)
def test_specific_analysis_types_drive_their_own_figure_contract(
    ra, analysis_type, question, step_intent, method, figure_tag
):
    """A family label cannot synthesize a missing scientific method owner."""
    from easyicu.research_agent import pipeline as pipeline_mod

    ctx = ra.build_research_context(
        research_question=question,
        cohort=pd.DataFrame(
            {
                "stay_id": [1, 2, 3, 4],
                "exposure": [0, 1, 0, 1],
                "death": [0, 1, 0, 1],
            }
        ),
        cohort_name="c",
        database="miiv",
        target_outcome="death",
    )
    plan = ra.schema.AnalysisPlan(
        research_question=question,
        analysis_type=analysis_type,
        steps=[
            ra.schema.AnalysisStep(
                step_id="03_step",
                intent=step_intent,
                inputs=["exposure", "death"],
                expected_outputs=["table:x"],
                method=method,
            )
        ],
    )

    revised, findings = pipeline_mod._enforce_advanced_plan_contract(
        plan=plan, context=ctx
    )

    assert revised == plan
    assert [step.step_id for step in revised.steps] == ["03_step"]
    assert revised.steps[0].method == method
    assert any(
        f.detail.get("family") == analysis_type
        and f.detail.get("missing_structured_owner") is True
        for f in findings
    ), findings
    all_outputs = [o for s in revised.steps for o in s.expected_outputs]
    assert figure_tag not in all_outputs, all_outputs


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


def test_pipeline_probe_can_trigger_replanning(ra, tmp_path: Path, monkeypatch):
    _disable_article_contract(monkeypatch)
    initial_plan = json.dumps(
        {
            "research_question": "Audit then model mortality.",
            "steps": [
                {
                    "step_id": "04_primary_association",
                    "planned_analysis_role": "primary",
                    "intent": "Model lactate and mortality.",
                    "inputs": ["lact", "death"],
                    "expected_outputs": ["table:primary_association"],
                    "method": "regression",
                    "icu_rule_refs": ["aggregation_rule_for"],
                }
            ],
            "rationale": "Initial one-step plan.",
            "revision": 1,
        }
    )
    revised_plan = json.dumps(
        {
            "research_question": "Audit then model mortality.",
            "steps": [
                {
                    "step_id": "00_probe",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Probe distributions before execution.",
                    "inputs": [],
                    "expected_outputs": [],
                    "method": None,
                    "icu_rule_refs": [],
                },
                {
                    "step_id": "03_missingness_audit",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Audit missingness before modelling.",
                    "inputs": ["lact", "death"],
                    "expected_outputs": ["table:missingness"],
                    "method": "missingness_audit",
                    "icu_rule_refs": ["aggregation_rule_for"],
                },
                {
                    "step_id": "04_primary_association",
                    "planned_analysis_role": "primary",
                    "intent": "Model lactate and mortality.",
                    "inputs": ["lact", "death"],
                    "expected_outputs": ["table:primary_association"],
                    "method": "regression",
                    "icu_rule_refs": ["aggregation_rule_for"],
                },
            ],
            "rationale": "Probe revealed substantial missingness; audit first.",
            "revision": 2,
        }
    )
    missingness_code = """
import json, os, pandas as pd
df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
pd.DataFrame({"variable": ["lact"], "fraction_missing": [float(df["lact"].isna().mean())]}).to_csv(os.path.join(out, "missingness.csv"), index=False)
summary = {"variable": "lact", "fraction_missing": float(df["lact"].isna().mean())}
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
"""
    association_code = """
import json, os, pandas as pd
df = pd.read_parquet(os.environ["COHORT_PARQUET"]).dropna(subset=["lact", "death"])
out = os.environ["STEP_OUT_DIR"]
pd.DataFrame({"variable": ["lact"], "odds_ratio": [1.2]}).to_csv(os.path.join(out, "primary_association.csv"), index=False)
summary = {"predictor": "lact", "primary_or": 1.2}
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
"""
    llm = PatternScriptedMockLLMClient(
        [
            (
                "Produce an ICU-AWARE RESEARCH PLAN as JSON",
                [initial_plan] * 8,
            ),
            ("REVISE THE ICU-AWARE RESEARCH PLAN", [revised_plan] * 8),
            (
                "WRITE THE PYTHON CODE FOR STEP 03_MISSINGNESS_AUDIT",
                [missingness_code] * 8,
            ),
            (
                "WRITE THE PYTHON CODE FOR STEP 04_PRIMARY_ASSOCIATION",
                [association_code] * 8,
            ),
            (
                "INTERPRET THE RESULTS OF STEP",
                ["See {evidence:primary_association}."] * 8,
            ),
            (
                "WRITE A MANUSCRIPT SCAFFOLD",
                [
                    "# Title\n\n## Results\n\nSee "
                    "{evidence:primary_association}.\n\n(left to the human author)"
                ]
                * 8,
            ),
        ]
    )

    cohort = pd.DataFrame(
        {
            "stay_id": range(1, 41),
            "lact": [None if i % 3 == 0 else 1.0 + (i % 5) for i in range(40)],
            "death": [1 if i % 7 == 0 else 0 for i in range(40)],
        }
    )
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
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
    partial = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    step_ids = [rec["step_id"] for rec in partial["per_step_records"]]
    assert "00_probe" in step_ids
    probe_records = [
        rec for rec in partial["per_step_records"] if rec["step_id"] == "00_probe"
    ]
    assert len(probe_records) == 1
    assert probe_records[0]["status"] == "ok"
    assert probe_records[0]["generation_mode"] == "deterministic_probe"
    coder_prompts = [
        message.content
        for messages, _kwargs in llm.calls
        for message in messages
        if message.role == "user"
        and "WRITE THE PYTHON CODE FOR STEP" in message.content.upper()
    ]
    assert not any("00_probe" in prompt for prompt in coder_prompts)
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


def test_deterministic_runner_repair_preserves_statsmodels_constructor_kwargs(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
import numpy as np
import pandas as pd
import statsmodels.api as sm
X = pd.DataFrame({"age": ["50", "60", "55", "65"], "sex_M": [True, False, True, False]})
y = pd.Series([0, 1, 0, 1])
model = sm.Logit(y, X, missing="raise")
result = model.fit(disp=0)
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="Pandas data cast to numpy dtype of object",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "dtype_coerce_v1"
    assert (
        'model = sm.Logit(*_easyicu_runner_repair_v1(X, y), missing="raise")' in patched
    )
    compile(patched, "<patched>", "exec")


def test_deterministic_runner_repair_aligns_statsmodels_endog_exog_indices(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
import numpy as np
import pandas as pd
import statsmodels.api as sm
df = pd.DataFrame(
    {
        "death": [0, 1, 0, 1, 0, 1],
        "lactate": [1.0, 2.2, 1.4, 2.8, 1.7, 3.1],
        "age": [50, 60, 55, 65, 58, 70],
    },
    index=[10, 12, 14, 16, 18, 20],
)
y = df["death"].astype(float)
X = df[["lactate", "age"]].reset_index(drop=True).astype(float)
X = sm.add_constant(X, has_constant="add")
model = sm.OLS(y, X)
result = model.fit()
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="ValueError: The indices for endog and exog are not aligned",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "statsmodels_endog_exog_index_align_v1"
    assert "_easyicu_statsmodels_align_index_v1(X, y)" in patched
    namespace = {}
    exec(patched, namespace)
    result = namespace["result"]
    assert len(result.params) == 3


def test_deterministic_summary_repair_aligns_indices_after_dtype_repair(ra):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    code = """
import numpy as np
import pandas as pd
import statsmodels.api as sm
df = pd.DataFrame(
    {
        "death": [0, 1, 0, 1, 0, 1],
        "lactate": [1.0, 2.2, 1.4, 2.8, 1.7, 3.1],
        "age": [50, 60, 55, 65, 58, 70],
    },
    index=[101, 103, 107, 109, 113, 127],
)
y = df["death"].astype(float)
X = df[["lactate", "age"]].reset_index(drop=True).astype(float)
X = sm.add_constant(X, has_constant="add")
model = sm.OLS(y, X)
result = model.fit()
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "primary_predictor": "lactate",
            "primary_or": None,
            "fit_error": "The indices for endog and exog are not aligned",
        },
        previous_repair="dtype_coerce_v1",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "statsmodels_endog_exog_index_align_v1"
    namespace = {}
    exec(patched, namespace)
    assert len(namespace["result"].params) == 3


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


def test_deterministic_runner_repair_filters_statsmodels_conf_int_rows(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
import numpy as np
import pandas as pd

class FakeResult:
    params = pd.Series(
        [0.0, 0.2, 0.4],
        index=["const", "sofa2_1", "sofa2_2"],
    )

    def conf_int(self):
        return pd.DataFrame(
            {0: [-0.1, 0.1, 0.3], 1: [0.1, 0.3, 0.5]},
            index=["const", "sofa2_1", "sofa2_2"],
        )

model_result = FakeResult()
coef_series = model_result.params.filter(like="sofa2_")
conf_int = model_result.conf_int().filter(like="sofa2_")
lower = np.exp(conf_int.iloc[:, 0])
upper = np.exp(conf_int.iloc[:, 1])
or_vals = np.exp(coef_series)
plot_df = pd.DataFrame({
    "or": or_vals.values,
    "ci_low": lower.values,
    "ci_high": upper.values,
})
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="IndexError: single positional indexer is out-of-bounds",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "statsmodels_conf_int_filter_axis_v1"
    assert '.index.astype(str).str.contains("sofa2_", regex=False)' in patched
    namespace = {}
    exec(patched, namespace)
    plot_df = namespace["plot_df"]
    assert list(plot_df["or"].round(6)) == list(np.exp([0.2, 0.4]).round(6))
    assert len(plot_df) == 2


def test_deterministic_runner_repair_conf_int_filter_is_case_neutral(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
import numpy as np
import pandas as pd

class FakeResult:
    def conf_int(self):
        return pd.DataFrame(
            {0: [-0.4, -0.2, 0.3], 1: [-0.1, 0.1, 0.7]},
            index=["const", "lact_early", "lact_late"],
        )

result = FakeResult()
ci = result.conf_int().filter(like="lact_")
lower = np.exp(ci.iloc[:, 0])
upper = np.exp(ci.iloc[:, 1])
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="IndexError: single positional indexer is out-of-bounds",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "statsmodels_conf_int_filter_axis_v1"
    assert '.index.astype(str).str.contains("lact_", regex=False)' in patched
    namespace = {}
    exec(patched, namespace)
    assert list(namespace["lower"].round(6)) == list(np.exp([-0.2, 0.3]).round(6))
    assert list(namespace["upper"].round(6)) == list(np.exp([0.1, 0.7]).round(6))


def test_deterministic_runner_repair_materializes_analysis_cohort_before_required_cols(
    ra,
    tmp_path: Path,
    monkeypatch,
):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
import json
import os
from pathlib import Path

import pandas as pd


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


cohort_path = os.environ["COHORT_PARQUET"]
out_dir = Path(os.environ["STEP_OUT_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(cohort_path)
required_cols = ["sofa2_admission", "analysis_cohort", "death", "age", "sex", "weight"]
missing_cols = [c for c in required_cols if c not in df.columns]
step_summary = {"step": "association"}
if missing_cols:
    step_summary["skipped"] = {
        "reason": "required columns missing",
        "missing_columns": missing_cols,
    }
    write_json(out_dir / "step_summary.json", step_summary)
    raise SystemExit

data = df[required_cols].copy()
data["analysis_cohort"] = data["analysis_cohort"].astype("category")
step_summary["status"] = "completed"
step_summary["categories"] = [str(x) for x in data["analysis_cohort"].cat.categories]
write_json(out_dir / "step_summary.json", step_summary)
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log='{"skipped": {"reason": "required columns missing", "missing_columns": ["analysis_cohort"]}}',
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "derived_analysis_cohort_materialization_v1"
    assert "materialize generated analysis strata" in patched

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "sofa2_admission": [0, 1, 2],
            "death": [0, 1, 0],
            "age": [60, 70, 80],
            "sex": ["F", "M", "F"],
            "weight": [60.0, 75.0, 81.0],
        }
    ).to_parquet(cohort_path)
    out_dir = tmp_path / "out"
    monkeypatch.setenv("COHORT_PARQUET", str(cohort_path))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    exec(compile(patched, "<patched>", "exec"), {})

    summary = json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["categories"] == ["0", "1", "2"]


def test_deterministic_runner_repair_analysis_cohort_source_is_case_neutral(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
import pandas as pd
df = pd.read_parquet(cohort_path)
required_cols = ["lactate_max_24h", "analysis_cohort", "death"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise SystemExit
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log='{"missing_columns": ["analysis_cohort"]}',
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "derived_analysis_cohort_materialization_v1"
    assert "'lactate_max_24h' in df.columns" in patched
    assert "sofa2" not in patched


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


def test_deterministic_runner_repair_downgrades_bad_publication_contract_without_comment(
    ra,
):
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
    assert (
        "    import importlib\n    json = importlib.import_module('json')\n    json.dump("
        in patched
    )


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
    X_mi = mi_df[[predictor_var, 'creatinine_missing_24h'] + covariates]
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
    assert (
        "valid_cc_idx = X_cc.dropna().index.intersection(y_cc.dropna().index)"
        in patched
    )
    assert "    if 'sex' in X_mi.columns:" in patched


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
    assert "easyicu.research_agent.figures.publication" in patched
    assert "easyicu.research_output.figure_utils" not in patched


def test_deterministic_runner_repair_does_not_strip_real_host_module(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
from easyicu.research_agent.methods.descriptive_inputs import (
    strict_numeric_input,
)
from easyicu.research_agent.methods.table_one import build_grouped_table_one

print(build_grouped_table_one)
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log=(
            "ModuleNotFoundError: No module named "
            "'easyicu.research_agent.methods.table_one'"
        ),
    )
    assert repaired is None


def test_deterministic_runner_repair_inserts_stub_after_parenthesized_import(ra):
    import ast

    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
from easyicu.research_agent.methods.descriptive_inputs import (
    strict_numeric_input,
)
from easyicu.fake_table_sdk import fake_table

print(fake_table)
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="ModuleNotFoundError: No module named 'easyicu.fake_table_sdk'",
    )
    assert repaired is not None
    _repair_id, patched = repaired
    ast.parse(patched)
    assert patched.index("strict_numeric_input,") < patched.index("def fake_table")


def test_deterministic_runner_repair_inserts_stub_before_late_import(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
from pathlib import Path
from easyicu.fake_table_sdk import fake_table

print(fake_table)
import json
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="ModuleNotFoundError: No module named 'easyicu.fake_table_sdk'",
    )
    assert repaired is not None
    _repair_id, patched = repaired
    assert patched.index("def fake_table") < patched.index("print(fake_table)")


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


def test_deterministic_runner_repair_filters_x_cols_before_dropna_after_dummy_encoding(
    ra,
):
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
    assert patched.index(
        "x_cols = [col for col in x_cols if col in model_df.columns]"
    ) < patched.index("dropna")


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
    assert patched.index(
        "x_cols = [col for col in x_cols if col in model_df.columns]"
    ) < patched.index("dropna")


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
        run_log="KeyError: \"None of [Index(['creat_max_24h', 'creat_median_24h'], dtype='object')] are in the [columns]\"",
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "missing_indicator_source_df_v1"
    assert (
        "model_df['creat_missing'] = df[creat_missing].isnull().any(axis=1).astype(int)"
        in patched
    )


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
        run_log="KeyError: \"['death'] not in index\"",
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "include_outcome_in_all_vars_v1"
    assert "all_vars = [outcome_col, primary_predictor] + covariates" in patched


def test_deterministic_runner_repair_restores_predictor_and_sex_in_robustness_script(
    ra,
):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
predictor_col = creatinine_col
covariates = ["age", "sex", "sofa2_max_24h"]
def fit_logistic_model(X, y):
    return None
model_df = df[all_vars].copy()
cc_X = cc_df[covariates]
mi_X = mi_df[covariates + ['creatinine_missing']]
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
    assert (
        "model_df['sex'] = model_df['sex'].astype(str).str.lower().isin(['m', 'male']).astype(float)"
        in patched
    )
    assert "cc_X = cc_df[[predictor_col] + covariates]" in patched
    assert (
        "mi_X = mi_df[[predictor_col] + covariates + ['creatinine_missing']]" in patched
    )
    assert "rv_X = rv_df[[predictor_col] + covariates]" in patched
    assert "plot_rows = [" in patched


def test_deterministic_runner_repair_stabilizes_predictor_col_robustness_template(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
outcome_col = "readmission_30d"
predictor_col = creatinine_col
covariates = ["age", "sex", "los_icu", "sofa2_max_24h", "map_min_24h", "vaso_any_24h", "bili_max_24h", "bili_n_24h", "creat_max_24h"]
model_df = df[[outcome_col, predictor_col] + covariates].copy()
model_df["creatinine_missing"] = model_df[predictor_col].isnull().astype(int)
complete_case_df = model_df.dropna(subset=[predictor_col])
missing_indicator_df = model_df.copy()
reduced_variable_df = model_df.drop(columns=[predictor_col]).copy()
X_cc = sm.add_constant(complete_case_df[covariates], has_constant="add")
X_mi = sm.add_constant(missing_indicator_df[covariates + ["creatinine_missing"]], has_constant="add")
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
    assert (
        "model_df['sex'] = model_df['sex'].astype(str).str.lower().isin(['m', 'male']).astype(float)"
        in patched
    )
    assert (
        "reduced_covariates = [c for c in covariates if model_df[c].isna().mean() <= 0.2]"
        in patched
    )
    assert (
        "complete_case_df = model_df.dropna(subset=[outcome_col, predictor_col] + covariates)"
        in patched
    )
    assert (
        "missing_indicator_df[predictor_col] = missing_indicator_df[predictor_col].fillna(0)"
        in patched
    )
    assert (
        "missing_indicator_df = missing_indicator_df.dropna(subset=[outcome_col] + covariates)"
        in patched
    )
    assert (
        "reduced_variable_df = model_df[[outcome_col, predictor_col] + reduced_covariates].dropna().copy()"
        in patched
    )
    assert (
        'X_cc = sm.add_constant(complete_case_df[[predictor_col] + covariates], has_constant="add")'
        in patched
    )
    assert (
        'X_mi = sm.add_constant(missing_indicator_df[[predictor_col] + covariates + ["creatinine_missing"]], has_constant="add")'
        in patched
    )
    assert (
        'X_rv = sm.add_constant(reduced_variable_df[[predictor_col] + reduced_covariates], has_constant="add")'
        in patched
    )
    assert "plot_rows = [" in patched
    assert "if len(x_pos):" in patched


def test_deterministic_runner_repair_preserves_indentation_for_robustness_patch(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
def main():
    outcome_col = "readmission_30d"
    predictor_col = creatinine_col
    covariates = ["age", "sex"]
    model_df = df[[outcome_col, predictor_col] + covariates].copy()
    model_df["creatinine_missing"] = model_df[predictor_col].isnull().astype(int)
    complete_case_df = model_df.dropna(subset=[predictor_col])
    missing_indicator_df = model_df.copy()
    reduced_variable_df = model_df.drop(columns=[predictor_col]).copy()
    X_cc = sm.add_constant(complete_case_df[covariates], has_constant="add")
    X_mi = sm.add_constant(missing_indicator_df[covariates + ["creatinine_missing"]], has_constant="add")
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
    assert (
        "\n    reduced_covariates = [c for c in covariates if model_df[c].isna().mean() <= 0.2]"
        in patched
    )


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
            "ImportError: cannot import name 'calibration_curve' from 'sklearn.metrics'"
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
            "ImportError: cannot import name 'calibration_curve' from 'sklearn.metrics'"
        ),
        previous_repair="prediction_calibration_import_fix_v1",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "prediction_calibration_import_fix_v1"
    assert "from sklearn.calibration import calibration_curve" in patched


def test_deterministic_runner_repair_rank_reduces_singular_logit(ra):
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
    assert name == "rank_safe_statsmodels_design_v1"
    assert "_easyicu_rank_safe_design_v1" in patched
    assert "model = sm.GLM(y, X, family=sm.families.Binomial())" in patched
    assert "_easyicu_safe_exp_v1" in patched


def test_deterministic_summary_repair_rank_reduces_nested_primary_model(ra):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    code = """
import numpy as np
import pandas as pd
import statsmodels.api as sm

def run_model():
    np.random.seed(42)
    n = 80
    exposure_col = "exposure"
    exposure = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    age = np.linspace(40, 80, n)
    logits = -2.0 + 0.9 * exposure + 0.01 * (age - 60)
    probabilities = 1 / (1 + np.exp(-logits))
    y = pd.Series(np.random.binomial(1, probabilities), name="death")
    X = pd.DataFrame({"exposure": exposure, "age": age, "age_dup": age})
    X = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y, X)
    return model.fit(disp=0, maxiter=200)

result = run_model()
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "analysis_family": "cohort_definition_sensitivity",
            "primary_exposure": "exposure",
            "target_outcome": "death",
            "primary_model": {
                "outcome": "death",
                "exposure": "exposure",
                "odds_ratio": None,
                "or_ci_low": None,
                "or_ci_high": None,
                "converged": False,
                "notes": "Model fit failed: LinAlgError: Singular matrix",
            },
        },
        analysis_family="cohort_definition_sensitivity",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "rank_safe_statsmodels_design_v1"
    namespace = {}
    exec(patched, namespace)
    result = namespace["result"]
    assert "exposure" in result.params.index
    assert "age_dup" not in result.params.index


def test_deterministic_summary_repair_prioritizes_failed_primary_over_sensitivity(ra):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    code = """
import statsmodels.api as sm
fit_method = "statsmodels_logit_mle"
interval_method = "profile_normal"
ci_low = beta - 1.96 * se
reported_interval_method = "wald_95_percent"
result = sm.Logit(y, X).fit(disp=False, maxiter=200)
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "model_contracts": [
                {
                    "analysis_role": "primary",
                    "fit_status": "not_fitted",
                    "fit_failure_reason": "Singular matrix",
                },
                {
                    "analysis_role": "sensitivity",
                    "fit_status": "fitted",
                },
            ],
            "sensitivity_result": {"odds_ratio": 1.60799},
        },
        analysis_family="cohort_definition_sensitivity",
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "rank_safe_statsmodels_design_v1"
    assert "_easyicu_rank_safe_design_v1" in patched
    assert (
        "result = sm.GLM(y, X, family=sm.families.Binomial()).fit("
        "disp=False, maxiter=200)"
    ) in patched
    assert 'fit_method = "statsmodels_glm_binomial_irls_rank_safe"' in patched
    assert 'interval_method = "wald_95_percent"' in patched


def test_deterministic_runner_repair_promotes_publication_bundle_script(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
from easyicu.research_agent.figures.publication import make_figure_contract
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


def test_deterministic_runner_repair_removes_qcut_observed_without_case_fallback(ra):
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
    assert name == "remove_pandas_cut_observed_keyword_v1"
    assert "age_tertile_mortality.csv" not in patched
    assert "observed=True" not in patched
    compile(patched, "<patched>", "exec")


def test_deterministic_runner_repair_does_not_default_to_norepi_case_fallback(ra):
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
    assert repaired is None


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


def test_step_contract_blocks_unauthorized_cohort_redefinition_in_qc_step(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.schema.AnalysisStep(
        step_id="04_exposure_derivation_qc",
        intent="Audit the ordered exposure and component consistency.",
        inputs=["stage_max"],
        expected_outputs=["table:stage_distribution"],
        method="ordinal_exposure_derivation_and_quality_control",
    )
    findings = _step_contract_findings(
        step=step,
        step_summary={
            "status": "completed",
            "analysis_family": "cohort_definition_sensitivity",
            "cohort_definition": {
                "current_step_is_cohort_definition_sensitivity": True
            },
        },
    )
    assert any(
        f.severity == "error"
        and f.detail.get("kind") == "unauthorized_cohort_redefinition"
        for f in findings
    )


def test_step_contract_allows_declared_primary_cohort_definition(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.schema.AnalysisStep(
        step_id="01_primary_cohort_flow",
        intent="Define the analysis cohort and report attrition.",
        inputs=["age"],
        expected_outputs=["table:cohort_flow"],
        method="cohort_definition",
    )
    findings = _step_contract_findings(
        step=step,
        step_summary={
            "status": "completed",
            "analysis_family": "cohort_definition_sensitivity",
            "n_final_cohort": 100,
        },
    )
    assert not [
        f
        for f in findings
        if f.detail.get("kind") == "unauthorized_cohort_redefinition"
    ]


def test_step_contract_does_not_duplicate_host_measurement_provenance_gate(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.schema.AnalysisStep(
        step_id="04_exposure_derivation_qc",
        intent="Audit ordered component summaries and source flags.",
        inputs=["stage_max", "stage_measured"],
        expected_outputs=["table:stage_component_qc"],
        method="ordinal_exposure_derivation_and_quality_control",
    )

    findings = _step_contract_findings(
        step=step,
        step_summary={"status": "completed", "component_qc": {}},
    )

    assert not [
        finding
        for finding in findings
        if str(finding.detail.get("kind") or "").startswith(
            "component_count_consistency"
        )
    ]


def test_deterministic_runner_repair_restores_primary_predictor_in_logit_design(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
import pandas as pd
import statsmodels.api as sm
df = pd.read_parquet(cohort_path)
model_df = df[['creatinine_max_24h', 'map_min_24h', 'vaso_any_24h', 'age', 'sex', 'readmission_30d']].copy()
model_df = pd.get_dummies(model_df, columns=['sex'], drop_first=True)
y = model_df['readmission_30d'].astype(float)
X = model_df[['map_min_24h', 'vaso_any_24h', 'age'] + [col for col in model_df.columns if col.startswith('sex_')]].astype(float)
X = sm.add_constant(X, has_constant='add')
try:
    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=0)
    coef_table = result.conf_int()
    primary_or = coef_table.loc['creatinine_max_24h', 'or']
except Exception as e:
    print(f"Error fitting logistic regression: {e}")
step_summary = {"n_total_stays": int(n_total), "odds_ratio": primary_or}
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log=(
            "Error fitting logistic regression: 'creatinine_max_24h'\n"
            "NameError: name 'n_total' is not defined"
        ),
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "primary_predictor_omitted_from_design_v1"
    assert "X = model_df[['creatinine_max_24h'," in patched
    assert "n_total = int(len(df))" in patched


def test_deterministic_runner_repair_restores_primary_predictor_with_indented_x_line(
    ra,
):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
def main():
    model_df = df[['creatinine_max_24h', 'map_min_24h', 'vaso_any_24h', 'age', 'sex', 'readmission_30d']].copy()
    y = model_df['readmission_30d'].astype(float)
    X = model_df[['map_min_24h', 'vaso_any_24h', 'age', 'sex']].copy()
    X = sm.add_constant(X, has_constant='add')
    coef_table = result.conf_int()
    primary_or = coef_table.loc['creatinine_max_24h', 'or']
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="Error fitting logistic regression: 'creatinine_max_24h'",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "primary_predictor_omitted_from_design_v1"
    assert "X = model_df[['creatinine_max_24h'," in patched


def test_deterministic_runner_repair_fixes_stringified_binary_outcome_key(ra):
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = """
table_one_data.append({"variable": "30-day readmission", "type": "binary", "count": summary["outcomes"]["readmission_30d"]["counts"][1],
                      "pct": summary["outcomes"]["readmission_30d"]["pct"][1]})
"""
    repaired = _deterministic_runner_repair(
        code=code,
        run_log="KeyError: 1",
    )
    assert repaired is not None
    name, patched = repaired
    assert name == "table_one_binary_key_string_v1"
    assert (
        '.get("1", summary["outcomes"]["readmission_30d"]["counts"].get(1, 0))'
        in patched
    )
    assert (
        '.get("1", summary["outcomes"]["readmission_30d"]["pct"].get(1, 0.0))'
        in patched
    )


def test_step_contract_findings_flag_missing_primary_association_estimate(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="03_primary_association_model",
        method="adjusted_logistic_regression",
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
        method="adjusted_logistic_regression",
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
        method="adjusted_logistic_regression",
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


def test_step_contract_findings_accepts_nested_adjusted_sofa_odds_ratio_dict(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="04_association_model",
        method="adjusted_logistic_regression",
        intent="Estimate adjusted association between admission SOFA-2 and ICU mortality.",
        expected_outputs=["statistic:primary_or", "table:adjusted_odds_ratio_sofa"],
    )
    findings = _step_contract_findings(
        step=step,
        step_summary={
            "primary": {
                "n": 470,
                "event_count": 43,
                "reference_sofa_level": 1.0,
                "adjusted_odds_ratio_sofa": {
                    "sofa2_0.0": 2.0253510259937584,
                    "sofa2_2.0": 2.0132350215619796,
                },
                "adjusted_odds_ratio_sofa_ci95": {
                    "sofa2_0.0": {
                        "low": 0.545606642458396,
                        "high": 7.518322650932099,
                    },
                    "sofa2_2.0": {
                        "low": 0.4295700581569013,
                        "high": 9.43528343067909,
                    },
                },
            },
        },
    )

    assert findings == []


def test_step_contract_rejects_blocked_step_summary(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    findings = _step_contract_findings(
        step=ra.AnalysisStep(
            step_id="04_reconciliation",
            intent="Reconcile registered descriptive evidence.",
            expected_outputs=["table:reconciliation"],
        ),
        step_summary={
            "status": "blocked",
            "blocking_reason": "No structured source input was available.",
        },
    )

    assert any(
        finding.severity == "error"
        and "cannot be recorded as a successful completed step" in finding.message
        for finding in findings
    )


def test_step_contract_findings_rejects_nested_ci_without_effect_value(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="04_association_model",
        method="adjusted_logistic_regression",
        intent="Estimate adjusted association between admission SOFA-2 and ICU mortality.",
        expected_outputs=["statistic:primary_or", "table:adjusted_odds_ratio_sofa"],
    )
    findings = _step_contract_findings(
        step=step,
        step_summary={
            "primary": {
                "n": 470,
                "adjusted_odds_ratio_sofa_ci95": {
                    "sofa2_0.0": {
                        "low": 0.545606642458396,
                        "high": 7.518322650932099,
                    },
                },
            },
        },
    )

    assert findings
    assert findings[0].validator == "step_contract"
    assert findings[0].severity == "error"


def test_step_contract_findings_accepts_predictor_named_or_key(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="03_association_model",
        method="adjusted_logistic_regression",
        intent="Fit lactate mortality model.",
        expected_outputs=["odds_ratio", "confidence_interval"],
    )

    findings = _step_contract_findings(
        step=step,
        step_summary={"creatinine_max_24h_or": 1.21},
    )

    assert findings == []


def test_step_contract_findings_accepts_primary_association_estimate_key(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="02_lactate_map_vaso_mortality_association",
        method="adjusted_logistic_regression",
        intent="Estimate a lactate/MAP/vasopressor mortality association.",
        expected_outputs=["statistic:primary_association"],
    )

    findings = _step_contract_findings(
        step=step,
        step_summary={"primary_association_estimate": 1.42},
    )

    assert findings == []


def test_step_contract_findings_does_not_require_or_for_data_quality_association_table(
    ra,
):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="01_component_completeness_qc",
        intent="Check composite-score component completeness.",
        expected_outputs=[
            "table:component_completeness",
            "table:missing_component_distribution",
            "log:missingness_summary",
        ],
    )

    findings = _step_contract_findings(
        step=step,
        step_summary={
            "low_completeness_count": 41,
            "guardrail_warning": True,
            "primary_association_estimate": None,
        },
    )

    assert findings == []


def test_step_contract_findings_does_not_treat_association_calibration_as_prediction(
    ra,
):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="03_association_model",
        method="adjusted_logistic_regression",
        intent="Estimate adjusted vasopressor association with mortality.",
        expected_outputs=[
            "statistic:adjusted_or_ci",
            "figure:association_model_calibration",
        ],
    )

    findings = _step_contract_findings(
        step=step,
        step_summary={"statistic": {"adjusted_or": 1.68}},
    )

    assert findings == []


def test_render_writer_evidence_digest_hides_failed_step_scalars(ra):
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
    assert "  {}" in digest
    assert '"sample_size": 785' not in digest
    assert '"primary_predictor": "lactate_max_24h"' not in digest
    assert '"skipped": "No valid lactate_max_24h data"' not in digest
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
        method="adjusted_logistic_regression",
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
        method="prediction_model_evaluation",
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
        method="prediction_model_evaluation",
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


def test_step_contract_findings_requires_declared_prediction_metrics(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="03_model_training",
        method="prediction_model_evaluation",
        intent="Train a mortality prediction model with 5-fold cross-validation.",
        expected_outputs=[
            "model:trained_prediction_model",
            "statistic:auroc",
            "statistic:brier_score",
        ],
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
        method="kmeans_clustering",
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
            "cluster_selection": {
                "criterion": "silhouette_score",
                "selection_rule": "maximum",
                "direction": "maximize",
                "selected_n_clusters": 2,
                "candidates": [
                    {"n_clusters": 1, "criterion_value": 0.0},
                    {"n_clusters": 2, "criterion_value": 0.46},
                ],
                "rationale": "Maximum among evaluated candidates.",
            },
        },
    )

    assert findings == []


def test_step_contract_findings_accepts_complete_case_primary_or_alias(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    step = ra.AnalysisStep(
        step_id="03_complete_case_robustness",
        method="logistic_regression",
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
            "statistic:primary_or_complete_case": 1.14,
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
        expected_outputs=[
            "table:adjusted_association",
            "figure:lactate_mortality_plot",
        ],
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
        method="adjusted_logistic_regression",
        intent="Estimate lactate association with a publication-ready figure.",
        expected_outputs=[
            "table:adjusted_association",
            "figure:lactate_mortality_plot",
        ],
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
    assert figure_step.inputs == ["table:table_one"]
    assert figure_step.method == "visualization"
    assert [
        contract.model_dump(mode="json")
        for contract in figure_step.input_consumption_contracts
    ] == [
        {
            "schema_version": "easyicu.artifact_consumption/1",
            "input_key": "table:table_one",
            "mode": "all_rows",
            "role_column": None,
            "expected_roles": [],
        }
    ]
    assert findings and findings[0].severity == "warning"
    assert "01_table_one_figure" in findings[0].message


def test_split_rehomes_figure_to_sole_exact_typed_source_producer(ra):
    from easyicu.research_agent.pipeline import (
        _split_table_and_figure_outputs_in_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="Report cohort flow and baseline characteristics.",
        steps=[
            AnalysisStep(
                step_id="02_cohort",
                intent="Create the cohort accounting table.",
                expected_outputs=["table:cohort_flow", "table:cohort_diagnostics"],
                method="cohort_definition_and_attrition",
            ),
            AnalysisStep(
                step_id="05_table_one",
                intent="Create Table 1 and the planned cohort-flow figure.",
                expected_outputs=["table:table_one", "figure:cohort_flow"],
                method="descriptive_baseline_characteristics",
            ),
        ],
    )

    revised, findings = _split_table_and_figure_outputs_in_plan(plan=plan)

    assert [step.step_id for step in revised.steps] == [
        "02_cohort",
        "02_cohort_figure",
        "05_table_one",
    ]
    assert revised.steps[0].expected_outputs == [
        "table:cohort_flow",
        "table:cohort_diagnostics",
    ]
    assert revised.steps[1].inputs == ["table:cohort_flow"]
    assert "table:cohort_diagnostics" not in revised.steps[1].inputs
    assert revised.steps[1].expected_outputs == ["figure:cohort_flow"]
    assert revised.steps[2].expected_outputs == ["table:table_one"]
    assert any(
        finding.detail.get("reason") == "figure_exact_typed_source_rehome"
        for finding in findings
    )


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


def test_split_table_and_figure_outputs_keeps_figure_with_log_sidecar(ra):
    from easyicu.research_agent.pipeline import (
        _split_table_and_figure_outputs_in_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="Render one publication figure and its process log.",
        steps=[
            AnalysisStep(
                step_id="05_publication_figure",
                intent="Render the already planned publication figure.",
                inputs=["table:primary_result"],
                expected_outputs=[
                    "figure:primary_result",
                    "log:rendering_process",
                ],
                method="publication_figure",
            )
        ],
    )

    revised, findings = _split_table_and_figure_outputs_in_plan(plan=plan)

    assert revised is plan
    assert findings == []


@pytest.mark.parametrize(
    "source_output",
    [
        "statistic:primary_effect",
        "model:adjusted_model",
        "artifact:primary_effect",
        "dataset:primary_effect",
    ],
)
def test_split_table_and_figure_outputs_requires_replayable_parent_table(
    ra,
    source_output,
):
    from easyicu.research_agent.pipeline import (
        _split_table_and_figure_outputs_in_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="Estimate a result and render its planned figure.",
        steps=[
            AnalysisStep(
                step_id="04_primary_result",
                intent="Estimate the result and render its planned figure.",
                expected_outputs=[source_output, "figure:primary_result"],
                method="logistic_regression",
            )
        ],
    )

    revised, findings = _split_table_and_figure_outputs_in_plan(plan=plan)

    assert revised is plan
    assert findings == []


def test_split_effect_figure_requires_effect_bearing_parent_table(ra):
    from easyicu.research_agent.pipeline import (
        _split_table_and_figure_outputs_in_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="Estimate a primary effect and render its forest plot.",
        steps=[
            AnalysisStep(
                step_id="04_primary_effect",
                intent="Estimate the primary effect and render its forest plot.",
                expected_outputs=[
                    "table:cohort_summary",
                    "statistic:primary_effect",
                    "figure:primary_effect_forest",
                ],
                method="logistic_regression",
            )
        ],
    )

    revised, findings = _split_table_and_figure_outputs_in_plan(plan=plan)

    assert revised is plan
    assert findings == []


def test_split_effect_figure_requires_bound_table_to_prove_figure_scale(ra):
    from easyicu.research_agent.pipeline import (
        _split_table_and_figure_outputs_in_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="Estimate an association and render its odds ratio.",
        steps=[
            AnalysisStep(
                step_id="04_primary_association",
                intent="Estimate and render the primary association.",
                expected_outputs=[
                    "table:association_estimates",
                    "figure:primary_or_forest",
                ],
                method="logistic_regression",
            )
        ],
    )

    revised, findings = _split_table_and_figure_outputs_in_plan(plan=plan)

    assert revised is plan
    assert findings == []


def test_split_effect_figure_when_exact_bound_table_proves_figure_scale(ra):
    from easyicu.research_agent.pipeline import (
        _split_table_and_figure_outputs_in_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="Estimate an odds ratio and render its forest plot.",
        steps=[
            AnalysisStep(
                step_id="04_primary_association",
                intent="Estimate and render the primary odds ratio.",
                expected_outputs=[
                    "table:primary_or",
                    "figure:primary_or_forest",
                ],
                method="logistic_regression",
            )
        ],
    )

    revised, findings = _split_table_and_figure_outputs_in_plan(plan=plan)

    assert [step.step_id for step in revised.steps] == [
        "04_primary_association",
        "04_primary_association_figure",
    ]
    assert revised.steps[1].inputs == ["table:primary_or"]
    assert findings


def test_split_generic_primary_adjusted_effect_from_planner_model_roster(ra):
    from easyicu.research_agent.pipeline import (
        _split_table_and_figure_outputs_in_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="Estimate the Planner-owned primary adjusted effect.",
        steps=[
            AnalysisStep(
                step_id="05_primary_adjusted_association",
                intent="Fit and render the Planner-owned adjusted model roster.",
                expected_outputs=[
                    "table:adjusted_association_estimates",
                    "artifact:primary_model_specification",
                    "figure:primary_adjusted_effect",
                ],
                method="adjusted_association_models",
                model_requirements=[
                    {
                        "requirement_id": "primary_death_model",
                        "outcome": "death",
                        "outcome_type": "binary",
                        "method_family": "logistic_regression",
                        "exposure_source": "exposure",
                        "analysis_role": "primary",
                        "analysis_set": "complete_case",
                        "required_for_step_success": True,
                    }
                ],
            )
        ],
    )

    revised, findings = _split_table_and_figure_outputs_in_plan(plan=plan)

    assert [step.step_id for step in revised.steps] == [
        "05_primary_adjusted_association",
        "05_primary_adjusted_association_figure",
    ]
    assert revised.steps[1].inputs == [
        "table:adjusted_association_estimates",
    ]
    assert findings


@pytest.mark.parametrize(
    "method",
    [
        "association_robustness with prespecified complete-case analysis",
        "bias_audit_association with negative controls",
        "clustering with kmeans",
    ],
)
def test_splitter_respects_non_primary_method_head_with_rider(ra, method):
    from easyicu.research_agent.pipeline import (
        _split_table_and_figure_outputs_in_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="Run the planned method and its display.",
        steps=[
            AnalysisStep(
                step_id="03_method",
                intent="Run the method and render its planned display.",
                expected_outputs=["table:results", "figure:results_overview"],
                method=method,
            )
        ],
    )

    revised, findings = _split_table_and_figure_outputs_in_plan(plan=plan)

    assert revised is plan
    assert findings == []


def test_split_table_and_figure_requires_unique_typed_parent_product(ra):
    from easyicu.research_agent.pipeline import (
        _split_table_and_figure_outputs_in_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="Render the primary association.",
        steps=[
            AnalysisStep(
                step_id="01_primary",
                intent="Estimate and render the primary association.",
                expected_outputs=[
                    "table:association_estimates",
                    "figure:association_forest",
                ],
            ),
            AnalysisStep(
                step_id="02_sensitivity",
                intent="Estimate a sensitivity association.",
                expected_outputs=["table:association_estimates"],
            ),
        ],
    )

    revised, findings = _split_table_and_figure_outputs_in_plan(plan=plan)

    assert revised is plan
    assert findings == []


def test_split_table_and_figure_does_not_create_duplicate_child_id(ra):
    from easyicu.research_agent.pipeline import (
        _split_table_and_figure_outputs_in_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="Render the primary table.",
        steps=[
            AnalysisStep(
                step_id="01_primary",
                intent="Create and render the primary table.",
                expected_outputs=["table:summary", "figure:summary"],
            ),
            AnalysisStep(
                step_id="01_primary_figure",
                intent="Existing agent-planned renderer.",
                inputs=["table:summary"],
                expected_outputs=["figure:summary"],
                method="visualization",
            ),
        ],
    )

    revised, findings = _split_table_and_figure_outputs_in_plan(plan=plan)

    assert revised is not plan
    assert any(
        finding.detail.get("reason") == "visualization_all_rows_consumption_default"
        for finding in findings
    )
    assert [step.step_id for step in revised.steps].count("01_primary_figure") == 1
    assert revised.steps[1].input_consumption_contracts[0].mode == "all_rows"


def test_split_table_and_figure_outputs_in_plan_no_op_for_advanced_self_contained_step(
    ra,
):
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


def test_plan_cap_drops_figure_when_its_typed_source_closure_exceeds_cap(ra):
    from easyicu.research_agent.pipeline import (
        _cap_plan_preserving_figure_steps,
        _ensure_publication_figure_step_in_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    class _Ctx:
        research_question = "Build tables and a concise publication-ready figure."

    plan = AnalysisPlan(
        research_question=_Ctx.research_question,
        steps=[
            AnalysisStep(
                step_id=f"{idx:02d}_table",
                intent=f"Compute table {idx}.",
                expected_outputs=[f"table:t{idx}"],
            )
            for idx in range(1, 5)
        ],
    )
    with_figure, _ = _ensure_publication_figure_step_in_plan(
        plan=plan,
        context=_Ctx(),
    )

    capped, findings = _cap_plan_preserving_figure_steps(plan=with_figure, cap=4)

    assert len(capped.steps) == 4
    assert not any(
        "publication_figure_fallback" in step.step_id for step in capped.steps
    )
    assert {output for step in capped.steps for output in step.expected_outputs} == {
        "table:t1",
        "table:t2",
        "table:t3",
        "table:t4",
    }
    assert findings
    assert findings[0].detail["dependency_displaced_figure_step_ids"]


def test_plan_cap_preserves_figure_source_parent_pair(ra):
    """Preserving a split figure step must not displace its source step.

    Regression from E1: max_total_steps=6 kept
    ``05_sensitivity_comparison_figure`` by replacing
    ``05_sensitivity_comparison``, leaving a rendering-only step with no
    upstream source table.
    """
    from easyicu.research_agent.pipeline import _cap_plan_preserving_figure_steps
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="E1 Sepsis-3 benchmark.",
        steps=[
            AnalysisStep(
                step_id="01_cohort",
                intent="Define the cohort.",
                expected_outputs=["table:cohort"],
            ),
            AnalysisStep(
                step_id="02_table_one",
                intent="Build Table 1.",
                expected_outputs=["table:table_one"],
            ),
            AnalysisStep(
                step_id="03_missingness",
                intent="Audit missingness.",
                expected_outputs=["table:missing"],
            ),
            AnalysisStep(
                step_id="04_primary_model",
                intent="Fit the primary model.",
                expected_outputs=["table:primary", "statistic:primary_or"],
            ),
            AnalysisStep(
                step_id="04_primary_model_figure",
                intent="Render the publication figure(s) declared by step '04_primary_model'.",
                expected_outputs=["figure:effect_estimate_forest"],
            ),
            AnalysisStep(
                step_id="05_sensitivity_comparison",
                intent="Run sensitivity analyses.",
                expected_outputs=["table:sensitivity", "statistic:robustness_or"],
            ),
            AnalysisStep(
                step_id="05_sensitivity_comparison_figure",
                intent=(
                    "Render the publication figure(s) declared by step "
                    "'05_sensitivity_comparison'."
                ),
                expected_outputs=["figure:sensitivity_forest"],
            ),
        ],
    )

    capped, findings = _cap_plan_preserving_figure_steps(plan=plan, cap=6)
    step_ids = [step.step_id for step in capped.steps]

    assert len(step_ids) == 6
    assert "05_sensitivity_comparison" in step_ids
    assert "05_sensitivity_comparison_figure" in step_ids
    assert step_ids.index("05_sensitivity_comparison") < step_ids.index(
        "05_sensitivity_comparison_figure"
    )
    assert "03_missingness" not in step_ids
    assert findings
    assert findings[0].detail["preserved_figure_step_ids"] == [
        "05_sensitivity_comparison_figure"
    ]


def test_plan_cap_does_not_displace_protected_completed_steps(ra):
    from easyicu.research_agent.pipeline import _cap_plan_preserving_figure_steps
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="Replan with completed upstream steps.",
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
                step_id="02_table_one",
                intent="Completed table one.",
                expected_outputs=["table:table_one"],
            ),
            AnalysisStep(
                step_id="03_missingness",
                intent="Audit missingness.",
                expected_outputs=["table:missingness"],
            ),
            AnalysisStep(
                step_id="04_primary_model",
                intent="Fit primary model.",
                expected_outputs=["table:primary"],
            ),
            AnalysisStep(
                step_id="04_primary_model_figure",
                intent="Render the publication figure(s) declared by step '04_primary_model'.",
                expected_outputs=["figure:primary"],
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

    capped, findings = _cap_plan_preserving_figure_steps(
        plan=plan,
        cap=6,
        protected_step_ids=["00_probe", "01_cohort", "02_table_one"],
    )
    step_ids = [step.step_id for step in capped.steps]

    assert {"00_probe", "01_cohort", "02_table_one"} <= set(step_ids)
    assert "05_sensitivity" in step_ids
    assert "05_sensitivity_figure" in step_ids
    assert "04_primary_model_figure" not in step_ids
    assert findings[0].detail["protected_step_ids"] == [
        "00_probe",
        "01_cohort",
        "02_table_one",
    ]


def test_plan_cap_makes_room_for_late_primary_anchor_without_exceeding_cap(ra):
    from easyicu.research_agent.pipeline import _cap_plan_preserving_figure_steps
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="Generic adjusted ICU association.",
        steps=[
            AnalysisStep(
                step_id=f"{idx:02d}_repair",
                intent=f"Repair supporting evidence {idx}.",
                method="evidence_repair",
            )
            for idx in range(1, 5)
        ]
        + [
            AnalysisStep(
                step_id="05_primary_adjusted_model",
                planned_analysis_role="primary",
                intent="Fit the primary adjusted model.",
                method="logistic_regression",
                expected_outputs=["table:primary_estimate"],
            )
        ],
    )

    capped, findings = _cap_plan_preserving_figure_steps(plan=plan, cap=4)
    step_ids = [step.step_id for step in capped.steps]

    assert len(step_ids) == 4
    assert "05_primary_adjusted_model" in step_ids
    assert findings
    assert "05_primary_adjusted_model" in findings[0].detail["protected_step_ids"]


def test_deterministic_runner_repair_injects_undefined_helper_stub(ra):
    """Regression: NameError for an undefined helper triggers stub injection."""
    from easyicu.research_agent.pipeline import _deterministic_runner_repair

    code = (
        "import json\n"
        'data = {"a": 1}\n'
        "with open('out.json', 'w') as f:\n"
        "    json.dump(data, f, default=to_json_serializable)\n"
    )
    run_log = (
        "Traceback (most recent call last):\n"
        '  File "analysis.py", line 4, in <module>\n'
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
    assert (
        result is None or result[0] != "undefined_helper_stub_to_json_serializable_v1"
    )


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


def test_manuscript_numeric_auditor_ignores_between_estimate_delta(ra):
    """Regression (M2): a difference/tolerance number near the metric word
    ('AUROC estimates differing by less than 0.001') is a delta between two
    estimates, not a reported point estimate, and must not be bound to the
    registered AUROC as a 0.001 claim. The genuine 0.827/0.83 claims in the
    same manuscript must still be checked and pass."""
    from easyicu.research_agent.pipeline import _audit_manuscript_numeric_claims

    bound = (
        "Discrimination was good, with an AUROC of 0.827 "
        "[model_performance](evidence/model_performance.csv). The two registered "
        "summaries agreed closely, with AUROC estimates differing by less than 0.001 "
        "and Brier scores differing by less than 0.001 "
        "[model_performance](evidence/model_performance.csv). A Brier score of 0.172 "
        "was reported [model_performance](evidence/model_performance.csv).\n"
    )
    findings = _audit_manuscript_numeric_claims(
        bound,
        per_step_records=[
            {
                "step_id": "01_model_training",
                "status": "ok",
                "step_summary": {
                    "statistic:auroc": 0.8267455907381426,
                    "statistic:brier_score": 0.1716274488483539,
                },
            }
        ],
    )

    assert findings == []


def test_manuscript_numeric_auditor_does_not_treat_ordinary_prose_as_delta(ra):
    from easyicu.research_agent.pipeline import _audit_manuscript_numeric_claims

    for sentence in (
        "Within the derivation cohort, the model achieved an AUROC of 0.92.",
        "The AUROC differed by site and was 0.92 in the derivation cohort.",
    ):
        findings = _audit_manuscript_numeric_claims(
            sentence + " [model](evidence/model.csv).\n",
            per_step_records=[
                {
                    "step_id": "01_model_training",
                    "status": "ok",
                    "step_summary": {"statistic:auroc": 0.766},
                }
            ],
        )

        assert any("AUROC claim" in finding.message for finding in findings)


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


def test_manuscript_numeric_auditor_ignores_stratum_specific_mortality_rate(ra):
    from easyicu.research_agent.pipeline import _audit_manuscript_numeric_claims

    bound = (
        "The overall cohort mortality was 9.4% [probe](evidence/probe.csv). "
        "The zero-score stratum had a lower mortality rate of 5.6% "
        "[strata](evidence/strata.csv).\n"
    )
    findings = _audit_manuscript_numeric_claims(
        bound,
        per_step_records=[
            {
                "step_id": "00_probe",
                "status": "ok",
                "step_summary": {"statistic:baseline_prevalence": 0.094},
            }
        ],
    )

    messages = [finding.message for finding in findings]
    assert not any("prevalence claim 0.056" in m for m in messages), messages


def test_manuscript_numeric_auditor_still_flags_overall_prevalence_mismatch(ra):
    from easyicu.research_agent.pipeline import _audit_manuscript_numeric_claims

    bound = "The overall cohort mortality was 5.6% [probe](evidence/probe.csv).\n"
    findings = _audit_manuscript_numeric_claims(
        bound,
        per_step_records=[
            {
                "step_id": "00_probe",
                "status": "ok",
                "step_summary": {"statistic:baseline_prevalence": 0.094},
            }
        ],
    )

    assert any("prevalence claim 0.056" in finding.message for finding in findings)


def test_manuscript_numeric_auditor_accepts_any_registered_step_auroc(ra):
    """Regression (N3, 2026-06-14): a prediction run registers more than one
    AUROC — the primary model step (0.868) and a feature-eligibility / audit
    step (0.812). The writer correctly cites the primary model's 0.868 in the
    Results headline and the audit step's 0.812 in a robustness sentence, each
    against its own evidence id. The auditor previously collapsed to the FIRST
    registered value (0.812, because that step's summary came first) and flagged
    the correctly-cited 0.868 as 'does not match registered AUROC 0.812',
    blocking an honest manuscript. Both values are registered, so neither is a
    hallucination and the audit must stay silent."""
    from easyicu.research_agent.pipeline import _audit_manuscript_numeric_claims

    bound = (
        "Discrimination in the development workflow reached an AUROC of 0.868 "
        "and a Brier score of 0.141 "
        "[01_model_training](evidence/model_training.json). In a feature-"
        "eligibility audit, the AUROC was 0.812 with a Brier score of 0.164 "
        "[01_feature_eligibility](evidence/feature_eligibility.json).\n"
    )
    findings = _audit_manuscript_numeric_claims(
        bound,
        per_step_records=[
            {
                "step_id": "01_feature_eligibility_range_audit",
                "status": "ok",
                "step_summary": {
                    "statistic:auroc": 0.8117303969867427,
                    "statistic:brier_score": 0.1635597216840789,
                },
            },
            {
                "step_id": "01_model_training",
                "status": "ok",
                "step_summary": {
                    "statistic:auroc": 0.8682892909365074,
                    "statistic:brier_score": 0.14099160242679873,
                    "statistic:baseline_prevalence": 0.10021385165893837,
                },
            },
        ],
    )

    assert findings == [], [f.message for f in findings]


def test_manuscript_numeric_auditor_ignores_footnote_provenance_stepid_digit(ra):
    """Regression (M2, 2026-06-14): the binder auto-appends footnote-definition
    lines like ``[^claim_2]: value=0.830918; field=metrics.auroc;
    evidence=statistic_step_summary_1c8c8ff2; display=0.831``. The metric field
    name ``metrics.auroc`` followed by the content-addressed step id's leading
    digit (``_1c8c8ff2``) was parsed as a spurious AUROC claim of ``1``, which
    then "did not match registered 0.831" and blocked the manuscript — an
    intermittent false positive that fired only when the sha started with a
    digit. The footnote line is machine provenance and must not be scanned; a
    real AUROC is always a decimal anyway."""
    from easyicu.research_agent.pipeline import _audit_manuscript_numeric_claims

    bound = (
        "On the held-out set the model achieved an AUROC of 0.83 "
        "[01_model_training](evidence/m.json).\n\n"
        "[^claim_2]: value=0.830918; step=01_model_training_figure; "
        "field=metrics.auroc; evidence=statistic_step_summary_1c8c8ff2; "
        "display=0.831\n"
    )
    findings = _audit_manuscript_numeric_claims(
        bound,
        per_step_records=[
            {
                "step_id": "01_model_training",
                "status": "ok",
                "step_summary": {
                    "statistic:auroc": 0.831,
                    "statistic:brier_score": 0.169,
                },
            }
        ],
    )

    assert findings == [], [f.message for f in findings]


def test_manuscript_numeric_auditor_flags_value_matching_no_registered_step(ra):
    """Guard for the match-any relaxation: a manuscript AUROC that matches
    NONE of the registered per-step values (here 0.95 against {0.812, 0.868})
    is still a hallucination and must be flagged. Match-any must not become
    match-nothing."""
    from easyicu.research_agent.pipeline import _audit_manuscript_numeric_claims

    bound = (
        "The model achieved an AUROC of 0.95 "
        "[01_model_training](evidence/model_training.json).\n"
    )
    findings = _audit_manuscript_numeric_claims(
        bound,
        per_step_records=[
            {
                "step_id": "01_feature_eligibility_range_audit",
                "status": "ok",
                "step_summary": {"statistic:auroc": 0.8117303969867427},
            },
            {
                "step_id": "01_model_training",
                "status": "ok",
                "step_summary": {"statistic:auroc": 0.8682892909365074},
            },
        ],
    )

    assert any("AUROC claim 0.95" in f.message for f in findings), [
        f.message for f in findings
    ]


def test_repair_common_writer_placeholders_prediction_fallbacks(ra, tmp_path: Path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
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


def test_prediction_placeholder_repair_does_not_create_outcome_rate_for_continuous_target(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
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
    context = ra.schema.ResearchContext(
        research_question="Build an ICU length-of-stay prediction model.",
        cohort=ra.schema.CohortDescriptor(
            cohort_name="c",
            database="synthetic",
            n_patients=10,
            n_stays=10,
            outcome_columns=["los_icu"],
        ),
        variables=[
            ra.schema.ConceptDescriptor(
                name="los_icu",
                role="outcome",
                dtype="float64",
                description="Continuous length-of-stay outcome.",
                source_concept="length_of_stay",
            )
        ],
        target_outcome="los_icu",
    )
    scaffold = (
        "The cohort summary is {evidence:table_one}. "
        "The endpoint rate was {evidence:outcome_rate}. "
        "Performance was evaluated {evidence:primary_association}."
    )

    repaired, repairs = _repair_common_writer_placeholders(
        scaffold,
        context=context,
        evidence=store,
    )

    assert "{evidence:table_one}" not in repaired
    assert "{evidence:outcome_rate}" in repaired
    assert "{evidence:primary_association}" not in repaired
    assert ("outcome_rate", "01_model_training") not in repairs


def test_repair_common_writer_citation_omissions_for_methods_sentences(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _repair_common_writer_citation_omissions

    store = EvidenceStore(tmp_path)
    for evidence_id in (
        "01_define_cohort_and_derive_sepsis3",
        "04_primary_adjusted_association_model",
    ):
        path = tmp_path / f"{evidence_id}.json"
        path.write_text("{}", encoding="utf-8")
        store.register_file(
            kind="statistic",
            description=evidence_id,
            source_path=path,
            evidence_id=evidence_id,
            producer="test",
        )

    scaffold = (
        "The primary predictor was binary Sepsis-3 status, derived from the "
        "available source columns.\n"
        "The primary association was estimated with logistic regression because "
        "the outcome was binary.\n"
    )

    repaired, repairs = _repair_common_writer_citation_omissions(
        scaffold,
        evidence=store,
    )

    assert "{evidence:01_define_cohort_and_derive_sepsis3}" in repaired
    assert "{evidence:04_primary_adjusted_association_model}" in repaired
    assert [item["evidence_id"] for item in repairs] == [
        "01_define_cohort_and_derive_sepsis3",
        "04_primary_adjusted_association_model",
    ]


def test_repair_common_writer_citation_omissions_handles_mixed_cited_paragraph(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _repair_common_writer_citation_omissions

    store = EvidenceStore(tmp_path)
    for evidence_id in (
        "01_define_cohort_and_derive_sepsis3",
        "04_primary_adjusted_association_model",
    ):
        path = tmp_path / f"{evidence_id}.json"
        path.write_text("{}", encoding="utf-8")
        store.register_file(
            kind="statistic",
            description=evidence_id,
            source_path=path,
            evidence_id=evidence_id,
            producer="test",
        )

    scaffold = (
        "The prior sentence is already cited {evidence:04_primary_adjusted_association_model}. "
        "The objective was to estimate Sepsis-3 prevalence and evaluate the "
        "association with in-hospital death after adjustment. "
        "The key exposure was derived from sep3_sofa2 fields.\n"
    )

    repaired, repairs = _repair_common_writer_citation_omissions(
        scaffold,
        evidence=store,
    )

    assert repaired.count("{evidence:04_primary_adjusted_association_model}") == 2
    assert "{evidence:01_define_cohort_and_derive_sepsis3}" in repaired
    assert [item["evidence_id"] for item in repairs] == [
        "04_primary_adjusted_association_model",
        "01_define_cohort_and_derive_sepsis3",
    ]


def test_repair_common_writer_citation_omissions_fails_closed_without_evidence(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _repair_common_writer_citation_omissions

    store = EvidenceStore(tmp_path)
    scaffold = (
        "The primary association was estimated with logistic regression because "
        "the outcome was binary.\n"
    )

    repaired, repairs = _repair_common_writer_citation_omissions(
        scaffold,
        evidence=store,
    )

    assert repaired == scaffold.rstrip()
    assert repairs == []


def test_repair_common_writer_citation_omissions_skips_manuscript_metadata(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _repair_common_writer_citation_omissions

    store = EvidenceStore(tmp_path)
    table_path = tmp_path / "table_one.csv"
    table_path.write_text("variable,value\nage,64\n", encoding="utf-8")
    store.register_file(
        kind="table",
        description="Table 1",
        source_path=table_path,
        evidence_id="table_one",
        producer="test",
    )
    scaffold = (
        "**Keywords:** Sepsis-3, intensive care unit, in-hospital mortality.\n"
        "## Data and code availability\n"
        "The cohort, generated scripts, SHA-256 evidence store, reproducibility "
        "envelope, STROBE checklist, and supplementary tables are released "
        "alongside this manuscript.\n"
    )

    repaired, repairs = _repair_common_writer_citation_omissions(
        scaffold,
        evidence=store,
    )

    assert repaired == scaffold.rstrip()
    assert repairs == []
    assert "{evidence:" not in repaired


def test_repair_common_writer_citation_omissions_does_not_launder_numeric_results(
    ra,
    tmp_path: Path,
):
    """A sentence reporting a numeric result must not be tagged to a Methods step.

    Regression guard for the narrow numeric-result detector: a Methods sentence
    that merely names a numbered concept (Sepsis-3) is still repaired, but a
    sentence that reports an effect estimate value is left to the fail-closed
    result-sentence filter instead of being laundered with a Methods citation.
    """

    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _repair_common_writer_citation_omissions

    store = EvidenceStore(tmp_path)
    for evidence_id in (
        "01_define_cohort_and_derive_sepsis3",
        "04_primary_adjusted_association_model",
    ):
        path = tmp_path / f"{evidence_id}.json"
        path.write_text("{}", encoding="utf-8")
        store.register_file(
            kind="statistic",
            description=evidence_id,
            source_path=path,
            evidence_id=evidence_id,
            producer="test",
        )

    scaffold = (
        "The primary predictor was binary Sepsis-3 status, derived from the "
        "available source columns.\n"
        "The adjusted odds ratio was 1.42 (95% CI 1.10-1.80).\n"
    )

    repaired, repairs = _repair_common_writer_citation_omissions(
        scaffold,
        evidence=store,
    )

    # The numbered-concept Methods sentence IS repaired.
    assert "{evidence:01_define_cohort_and_derive_sepsis3}" in repaired
    # The numeric-result sentence is NOT given any Methods citation.
    result_line = [ln for ln in repaired.splitlines() if "odds ratio was 1.42" in ln][0]
    assert "{evidence:" not in result_line
    assert [item["evidence_id"] for item in repairs] == [
        "01_define_cohort_and_derive_sepsis3"
    ]


def test_apply_writer_evidence_repair_decisions_cites_or_drops_without_rewriting(ra):
    from easyicu.research_agent.reporting.manuscript_post import (
        _apply_writer_evidence_repair_decisions,
    )

    supported = "Sepsis is clinically important."
    unsupported = "No estimate was available for reporting."
    scaffold = f"## Introduction\n\n{supported} {unsupported}\n"

    repaired, applied = _apply_writer_evidence_repair_decisions(
        scaffold,
        missing_sentences=[supported, unsupported],
        decisions=[
            {
                "index": 0,
                "action": "cite",
                "evidence_ids": ["literature_prisma"],
            },
            {"index": 1, "action": "drop", "evidence_ids": []},
        ],
    )

    assert "Sepsis is clinically important {evidence:literature_prisma}." in repaired
    assert unsupported not in repaired
    assert [item["action"] for item in applied] == ["cite", "drop"]


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
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
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


def _evidence_bound_demo_manuscript() -> str:
    return (
        "The manuscript reports the adjusted association, denominator audit, "
        "and sensitivity context using registered source evidence "
        "[model_evidence](evidence/model.csv).\n"
    )


def test_readiness_artifacts_reject_writer_failure_text(ra, tmp_path: Path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
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
        "(writer failed: Error code: 401 - invalid proxy api key.)",
        encoding="utf-8",
    )

    gates, artifact_paths = _write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=[],
        per_step_records=[{"step_id": "01_model_training", "status": "ok"}],
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=bound_path,
        stop_after_analysis=False,
    )

    assert gates["execution_complete"] is True
    assert gates["manuscript_text_ready"] is False
    assert gates["manuscript_generated"] is False
    assert gates["manuscript_ready"] is False
    assert gates["publication_ready"] is False
    assert any("writer/runtime failure" in e for e in gates["manuscript_text_errors"])
    assert "manuscript_ready" not in artifact_paths
    assert not (tmp_path / "manuscript_ready.md").exists()
    review_note = (tmp_path / "author_review_note.md").read_text(encoding="utf-8")
    assert "Manuscript text gate" in review_note
    assert "No blocking gate failures were detected" not in review_note


def test_readiness_artifacts_reject_unresolved_manifest_comments(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _write_readiness_artifacts

    context = ra.ResearchContext(
        research_question="Estimate whether Sepsis-3 is associated with mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="01_table_one",
                intent="Summarise baseline characteristics.",
                expected_outputs=["table:table_one"],
            ),
            ra.AnalysisStep(
                step_id="02_model",
                intent="Fit adjusted association model.",
                expected_outputs=["table:adjusted_association"],
            ),
            ra.AnalysisStep(
                step_id="03_sensitivity",
                intent="Audit sensitivity and denominator robustness.",
                expected_outputs=["figure:sensitivity_audit"],
            ),
        ],
    )
    evidence = EvidenceStore(tmp_path)
    current_evidence = _register_complete_display_suite_for_readiness(
        evidence,
        tmp_path,
        table_step_id="01_table_one",
        publication_source_step_id="02_model",
    )
    bound_path = tmp_path / "manuscript_scaffold_bound.md"
    bound_path.write_text(
        _evidence_bound_demo_manuscript() + "\nThe estimate is linked to evidence "
        "[model_evidence](evidence/model.csv)<!-- warning: see manifest -->.\n"
        + "A remaining claim is also linked "
        "[model_evidence](evidence/model.csv)<!-- error: see manifest -->.\n",
        encoding="utf-8",
    )

    gates, artifact_paths = _write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=[],
        per_step_records=[
            {
                "step_id": "01_table_one",
                "status": "ok",
                "evidence_ids": current_evidence["01_table_one"],
            },
            {
                "step_id": "02_model",
                "status": "ok",
                "evidence_ids": current_evidence["02_model"],
            },
            {"step_id": "03_sensitivity", "status": "ok"},
        ],
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=bound_path,
        stop_after_analysis=False,
    )

    assert gates["execution_complete"] is True
    assert gates["publication_figure_bundle_ready"] is True
    assert gates["manuscript_text_ready"] is False
    assert gates["manuscript_manifest_warning_count"] == 1
    assert gates["manuscript_manifest_error_count"] == 1
    assert any("manifest warning" in err for err in gates["manuscript_text_errors"])
    assert any("manifest error" in err for err in gates["manuscript_text_errors"])
    assert gates["manuscript_ready"] is False
    assert gates["publication_ready"] is False
    assert "manuscript_ready" not in artifact_paths
    assert not (tmp_path / "manuscript_ready.md").exists()


def test_readiness_artifacts_emit_manuscript_ready_only_after_gates_pass(
    ra, tmp_path: Path
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
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
    bound_path.write_text(_evidence_bound_demo_manuscript(), encoding="utf-8")

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


def test_readiness_artifacts_reject_untraced_numeric_marker(ra, tmp_path: Path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
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
        _evidence_bound_demo_manuscript()
        + "\nThe writer added 999 <!-- UNTRACED:999 -->.\n",
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

    assert gates["execution_complete"] is True
    assert gates["numeric_verified"] is False
    assert gates["manuscript_ready"] is False
    assert gates["numeric_error_count"] == 1
    assert "manuscript_ready" not in artifact_paths
    assert not (tmp_path / "manuscript_ready.md").exists()


def _register_publication_bundle_for_readiness(
    evidence,
    tmp_path: Path,
    *,
    contract: dict,
    source_step_id: str | None = None,
) -> str:
    from easyicu.research_agent.figures.publication import (
        PUBLICATION_FIGURE_SKILL_POLICY_VERSION,
    )

    out = tmp_path / "publication_figures"
    out.mkdir(parents=True, exist_ok=True)
    contract_path = out / "easyicu_publication_figure.figure_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    source_path = out / "publication_figure_source_data.csv"
    source_path.write_text("term,estimate\nsepsis3,1.14\n", encoding="utf-8")
    source_record = evidence.register_file(
        kind="table",
        description="Publication figure source data.",
        source_path=source_path,
        evidence_id="publication_figure_source_data",
        produced_by_step=source_step_id,
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
    )
    source_metadata = {
        "figure_skill_policy_version": PUBLICATION_FIGURE_SKILL_POLICY_VERSION,
        "source_evidence_id": source_record.evidence_id,
        "source_evidence_ids": [source_record.evidence_id],
        "source_evidence_sha256": {source_record.evidence_id: source_record.sha256},
    }
    evidence.register_file(
        kind="log",
        description="Publication figure contract.",
        source_path=contract_path,
        evidence_id="publication_figure_contract",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
        metadata=source_metadata,
    )
    for suffix in ("svg", "png"):
        path = out / f"easyicu_publication_figure.{suffix}"
        path.write_text("figure", encoding="utf-8")
        evidence.register_file(
            kind="figure",
            description=f"Publication figure export ({suffix}).",
            source_path=path,
            evidence_id=f"publication_figure_{suffix}",
            producer="publication_figure_skill",
            generation_mode="deterministic_figure_skill",
            metadata={
                "figure_role": "publication_figure",
                "figure_contract": "publication_figure_contract",
                **source_metadata,
            },
        )
    return source_record.evidence_id


def _write_publication_skill_summary(
    evidence,
    *,
    version: int,
    audit_findings: list[dict],
) -> Path:
    suffix = "" if version == 1 else f"_v{version}"
    source_ids = [
        record.evidence_id
        for record in evidence.records()
        if str(record.evidence_id).startswith("publication_figure_source_")
    ]
    figure_ids = [
        record.evidence_id
        for record in evidence.records()
        if record.kind == "figure"
        and str(record.evidence_id).startswith("publication_figure_")
    ]
    contract_id = next(
        (
            record.evidence_id
            for record in evidence.records()
            if str(record.evidence_id).startswith("publication_figure_contract")
        ),
        None,
    )
    record = evidence.register_json(
        kind="log",
        description="PublicationFigureSkill summary.",
        payload={
            "generated": True,
            "source_evidence_ids": source_ids,
            "figure_evidence_ids": figure_ids,
            "contract_evidence_id": contract_id,
            "audit_findings": audit_findings,
        },
        filename="publication_figure_skill_summary.json",
        evidence_id=f"publication_figure_skill_summary{suffix}",
        producer="PublicationFigureSkill",
        generation_mode="deterministic_figure_skill",
    )
    return evidence.root / record.relative_path


def _register_complete_display_suite_for_readiness(
    evidence,
    tmp_path: Path,
    *,
    table_step_id: str | None = None,
    publication_source_step_id: str | None = None,
) -> dict[str, list[str]]:
    provenance_path = tmp_path / "provenance_sources.json"
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": "easyicu.provenance_sources/1",
                "records": [
                    {
                        "relative_path": "cohort.parquet",
                        "sha256": "a" * 64,
                        "skipped_reason": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence.register_file(
        kind="log",
        description="Raw/cohort source provenance.",
        source_path=provenance_path,
        evidence_id="provenance_sources",
        producer="pipeline",
        generation_mode="system",
    )
    table_one_path = tmp_path / "table_one.csv"
    table_one_path.write_text("variable,value\nage,64\n", encoding="utf-8")
    table_record = evidence.register_file(
        kind="table",
        description="Table 1 baseline cohort characteristics.",
        source_path=table_one_path,
        evidence_id="table_table_one",
        produced_by_step=table_step_id,
        producer="coder",
        generation_mode="llm_code",
    )
    primary_path = tmp_path / "primary_estimand.csv"
    primary_path.write_text(
        "term,estimate,ci_low,ci_high\nsepsis3,1.14,1.02,1.28\n",
        encoding="utf-8",
    )
    primary_record = evidence.register_file(
        kind="table",
        description="Planner-owned primary adjusted estimate.",
        source_path=primary_path,
        evidence_id="table_primary_estimand",
        produced_by_step=publication_source_step_id,
        producer="coder",
        generation_mode="llm_code",
    )
    publication_source_id = _register_publication_bundle_for_readiness(
        evidence,
        tmp_path,
        source_step_id=publication_source_step_id,
        contract={
            "figure_id": "easyicu_publication_figure",
            "core_claim": "Absolute risk, primary effect, data quality, and sensitivity audit are shown.",
            "panels": [
                {
                    "panel_id": "A",
                    "title": "Absolute outcome risk",
                    "role": "descriptive_result",
                    "chart_type": "dot_interval_absolute_risk",
                    "claim": "Exposure prevalence and absolute outcome risk are shown before adjusted estimates.",
                },
                {
                    "panel_id": "B",
                    "title": "Adjusted odds-ratio estimate",
                    "role": "relationship",
                    "chart_type": "forest",
                    "claim": "The primary effect estimate is drawn from source data.",
                },
                {
                    "panel_id": "C",
                    "title": "Missingness and measurement availability",
                    "role": "data_quality",
                    "chart_type": "availability_panel",
                    "claim": "Missingness and measurement availability are shown with source-data denominators.",
                },
                {
                    "panel_id": "D",
                    "title": "Sensitivity and denominator audit",
                    "role": "robustness",
                    "chart_type": "specification_grid",
                    "claim": "Robustness and denominator context are shown together.",
                },
            ],
        },
    )
    bound: dict[str, list[str]] = {}
    if table_step_id:
        bound.setdefault(table_step_id, []).append(table_record.evidence_id)
    if publication_source_step_id:
        bound.setdefault(publication_source_step_id, []).extend(
            [primary_record.evidence_id, publication_source_id]
        )
    return bound


def _authoritative_readiness_records(
    plan,
    evidence_ids_by_step: dict[str, list[str]],
) -> list[dict]:
    """Build the same doubly-bound role records produced by the live pipeline."""

    return [
        {
            "step_id": step.step_id,
            "status": "ok",
            "planned_analysis_role": step.planned_analysis_role,
            "analysis_request": {"step": step.model_dump(mode="json")},
            "evidence_ids": list(evidence_ids_by_step.get(step.step_id, [])),
        }
        for step in plan.steps
    ]


def test_readiness_publication_ready_requires_article_display_suite(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _write_readiness_artifacts

    context = ra.ResearchContext(
        research_question="Estimate whether Sepsis-3 is associated with mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="01_table_one",
                intent="Summarise baseline characteristics.",
                expected_outputs=["table:table_one"],
            ),
            ra.AnalysisStep(
                step_id="02_model",
                intent="Fit adjusted association model.",
                expected_outputs=["table:adjusted_association"],
            ),
        ],
    )
    evidence = EvidenceStore(tmp_path)
    publication_source_id = _register_publication_bundle_for_readiness(
        evidence,
        tmp_path,
        source_step_id="02_model",
        contract={
            "figure_id": "easyicu_publication_figure",
            "core_claim": "Adjusted association estimate.",
            "panels": [
                {
                    "panel_id": "A",
                    "title": "Adjusted association",
                    "role": "relationship",
                    "claim": "The adjusted odds ratio is shown.",
                }
            ],
        },
    )
    bound_path = tmp_path / "manuscript_scaffold_bound.md"
    bound_path.write_text(_evidence_bound_demo_manuscript(), encoding="utf-8")

    gates, artifact_paths = _write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=[],
        per_step_records=[
            {"step_id": "01_table_one", "status": "ok"},
            {
                "step_id": "02_model",
                "status": "ok",
                "evidence_ids": [publication_source_id],
            },
        ],
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=bound_path,
        stop_after_analysis=False,
    )

    assert gates["publication_figure_bundle_ready"] is True
    assert gates["manuscript_ready"] is True
    assert gates["display_suite_complete"] is False
    assert gates["publication_ready"] is False
    assert "display_suite_audit" in artifact_paths
    assert "article_contract_audit" in artifact_paths
    assert gates["article_contract_complete"] is False
    assert "baseline_context" in gates["article_missing_artifact_roles"]
    assert "data_quality" in gates["article_missing_artifact_roles"]
    assert any("Table 1" in err for err in gates["display_suite_errors"])
    assert any("fewer than two panels" in err for err in gates["display_suite_errors"])


def test_readiness_publication_ready_accepts_complete_display_suite(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _write_readiness_artifacts

    context = ra.ResearchContext(
        research_question="Estimate whether Sepsis-3 is associated with mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="01_table_one",
                intent="Summarise baseline characteristics.",
                expected_outputs=["table:table_one"],
            ),
            ra.AnalysisStep(
                step_id="02_model",
                intent="Fit adjusted association model.",
                expected_outputs=["table:primary_estimand"],
                planned_analysis_role="primary",
            ),
            ra.AnalysisStep(
                step_id="03_sensitivity",
                intent="Audit sensitivity and denominator robustness.",
                expected_outputs=["figure:sensitivity_audit"],
            ),
        ],
    )
    evidence = EvidenceStore(tmp_path)
    bound_evidence = _register_complete_display_suite_for_readiness(
        evidence,
        tmp_path,
        table_step_id="01_table_one",
        publication_source_step_id="02_model",
    )
    bound_path = tmp_path / "manuscript_scaffold_bound.md"
    bound_path.write_text(_evidence_bound_demo_manuscript(), encoding="utf-8")

    gates, artifact_paths = _write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=[],
        per_step_records=_authoritative_readiness_records(plan, bound_evidence),
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=bound_path,
        stop_after_analysis=False,
    )

    assert gates["display_suite_complete"] is True
    assert gates["article_contract_complete"] is True
    assert gates["article_figure_strategy_complete"] is True
    assert gates["display_table_one_present"] is True
    assert gates["display_contract_panel_count"] == 4
    assert gates["display_absolute_risk_visual_present"] is True
    assert "dot_interval_absolute_risk" in gates["display_chart_types"]
    assert gates["publication_ready"] is True
    assert (tmp_path / artifact_paths["display_suite_audit"]).exists()
    assert (tmp_path / artifact_paths["article_contract_audit"]).exists()
    assert (tmp_path / artifact_paths["article_figure_strategy_audit"]).exists()


def test_display_suite_keeps_step_contracts_supporting_not_primary(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _write_readiness_artifacts

    context = ra.ResearchContext(
        research_question="Estimate whether Sepsis-3 is associated with mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="01_table_one",
                intent="Summarise baseline characteristics.",
                expected_outputs=["table:table_one"],
            ),
            ra.AnalysisStep(
                step_id="02_model",
                intent="Fit adjusted association model.",
                expected_outputs=["table:adjusted_association"],
            ),
            ra.AnalysisStep(
                step_id="03_supporting",
                intent="Render supporting sensitivity and data-quality figures.",
                expected_outputs=["figure:supporting_sensitivity"],
            ),
        ],
    )
    evidence = EvidenceStore(tmp_path)
    table_one_path = tmp_path / "table_one.csv"
    table_one_path.write_text("variable,value\nage,64\n", encoding="utf-8")
    evidence.register_file(
        kind="table",
        description="Table 1 baseline cohort characteristics.",
        source_path=table_one_path,
        evidence_id="table_table_one",
        producer="coder",
        generation_mode="llm_code",
    )
    _register_publication_bundle_for_readiness(
        evidence,
        tmp_path,
        contract={
            "figure_id": "easyicu_publication_figure",
            "core_claim": "Adjusted association estimate.",
            "panels": [
                {
                    "panel_id": "A",
                    "title": "Adjusted association",
                    "role": "primary_estimand",
                    "chart_type": "forest",
                    "claim": "The adjusted odds ratio is shown.",
                }
            ],
        },
    )
    support_dir = tmp_path / "steps" / "03_supporting" / "outputs"
    support_dir.mkdir(parents=True)
    (support_dir / "supporting_sensitivity.png").write_text(
        "supporting figure",
        encoding="utf-8",
    )
    (support_dir / "supporting_sensitivity.figure_contract.json").write_text(
        json.dumps(
            {
                "figure_id": "supporting_sensitivity",
                "core_claim": (
                    "Supporting absolute-risk, missingness, and sensitivity "
                    "panels are available for supplement review."
                ),
                "panels": [
                    {
                        "panel_id": "B",
                        "title": "Exposure prevalence and absolute outcome risk",
                        "role": "descriptive_result",
                        "chart_type": "dot_interval_absolute_risk",
                        "claim": (
                            "Exposure prevalence and absolute outcome risk "
                            "are shown before adjusted estimates."
                        ),
                    },
                    {
                        "panel_id": "C",
                        "title": "Missingness and measurement availability",
                        "role": "data_quality",
                        "chart_type": "availability_panel",
                        "claim": "Missingness and measurement quality are visible.",
                    },
                    {
                        "panel_id": "D",
                        "title": "Sensitivity and denominator audit",
                        "role": "robustness",
                        "chart_type": "specification_grid",
                        "claim": "Sensitivity and denominator audit context are shown.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    bound_path = tmp_path / "manuscript_scaffold_bound.md"
    bound_path.write_text(_evidence_bound_demo_manuscript(), encoding="utf-8")

    gates, artifact_paths = _write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=[],
        per_step_records=[
            {"step_id": "01_table_one", "status": "ok"},
            {"step_id": "02_model", "status": "ok"},
            {"step_id": "03_supporting", "status": "ok"},
        ],
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=bound_path,
        stop_after_analysis=False,
    )

    assert gates["display_suite_complete"] is False
    assert gates["publication_ready"] is False
    assert gates["display_contract_panel_count"] == 4
    assert gates["display_primary_publication_panel_count"] == 1
    assert gates["display_supporting_panel_count"] == 3
    assert gates["display_primary_publication_contract_paths"] == [
        "publication_figures/easyicu_publication_figure.figure_contract.json"
    ]
    assert gates["display_supporting_figure_contract_paths"] == [
        "steps/03_supporting/outputs/supporting_sensitivity.figure_contract.json"
    ]
    assert gates["display_absolute_risk_visual_present"] is True
    assert gates["display_primary_publication_absolute_risk_visual_present"] is False
    assert gates["display_supporting_absolute_risk_visual_present"] is True
    assert any(
        "Primary publication figure exposes fewer" in err
        for err in gates["display_suite_errors"]
    )
    assert any(
        "Primary publication figure lacks panel-role" in err
        for err in gates["display_suite_errors"]
    )
    assert any(
        "Primary association figure lacks" in err
        for err in gates["display_suite_errors"]
    )

    display_audit = json.loads(
        (tmp_path / artifact_paths["display_suite_audit"]).read_text(encoding="utf-8")
    )
    assert display_audit["schema_version"] == "easyicu.display_suite_audit/2"
    assert display_audit["primary_publication_panel_count"] == 1
    assert display_audit["supporting_panel_count"] == 3
    assert display_audit["primary_publication_contract_paths"] == [
        "publication_figures/easyicu_publication_figure.figure_contract.json"
    ]
    assert artifact_paths["review_artifacts"] == "review_artifacts.json"
    assert artifact_paths["figure_gallery"] == "figure_gallery.json"
    run_status = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    canonical_outputs = run_status["canonical_outputs"]
    assert (
        canonical_outputs["primary_publication_figure"]
        == "publication_figures/easyicu_publication_figure.png"
    )
    assert (
        canonical_outputs["primary_publication_figure_contract"]
        == "publication_figures/easyicu_publication_figure.figure_contract.json"
    )
    assert (
        canonical_outputs["primary_publication_figure_png"]
        == "publication_figures/easyicu_publication_figure.png"
    )
    assert (
        canonical_outputs["primary_publication_figure_svg"]
        == "publication_figures/easyicu_publication_figure.svg"
    )
    assert "steps/03_supporting/outputs/supporting_sensitivity.png" not in set(
        canonical_outputs.values()
    )

    review_artifacts = json.loads(
        (tmp_path / "review_artifacts.json").read_text(encoding="utf-8")
    )
    assert review_artifacts["schema_version"] == "easyicu.review_artifacts/1"
    assert (
        review_artifacts["policy"][
            "supporting_step_figures_are_not_canonical_main_figures"
        ]
        is True
    )
    assert review_artifacts["primary_publication_figures"][0]["tier"] == (
        "primary_publication"
    )
    assert review_artifacts["primary_publication_figures"][0]["relative_path"] == (
        "publication_figures/easyicu_publication_figure.png"
    )
    assert (
        review_artifacts["primary_publication_figures"][0]["review_recommendation"]
        == "review_first"
    )
    assert review_artifacts["supporting_figures"][0]["tier"] == "supporting_step"
    assert review_artifacts["supporting_figures"][0]["relative_path"] == (
        "steps/03_supporting/outputs/supporting_sensitivity.png"
    )
    assert review_artifacts["supporting_figures"][0]["review_recommendation"] == (
        "supporting_context_not_primary"
    )
    assert review_artifacts["primary_publication_figures"][0]["data_url"].startswith(
        "data:image/png;base64,"
    )
    assert "data_url" not in review_artifacts["supporting_figures"][0]

    figure_gallery = json.loads(
        (tmp_path / "figure_gallery.json").read_text(encoding="utf-8")
    )
    assert figure_gallery["kind"] == "figure_gallery"
    assert figure_gallery["primary_count"] == 1
    assert figure_gallery["supporting_count"] == 1
    assert figure_gallery["figures"][0]["tier"] == "primary_publication"
    assert figure_gallery["figures"][0]["data_url"].startswith("data:image/png;base64,")
    assert figure_gallery["figures"][1]["tier"] == "supporting_step"
    assert "data_url" not in figure_gallery["figures"][1]
    author_review = (tmp_path / "author_review_note.md").read_text(encoding="utf-8")
    assert "primary_publication_figure_contracts: `1`" in author_review
    assert "supporting_figure_contracts: `1`" in author_review


def test_review_gallery_archives_covered_and_duplicate_supporting_figures(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _write_readiness_artifacts

    def write_support_contract(
        step_id: str,
        stem: str,
        *,
        figure_id: str,
        roles: list[str],
    ) -> None:
        out = tmp_path / "steps" / step_id / "outputs"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{stem}.png").write_text("figure", encoding="utf-8")
        (out / f"{stem}.figure_contract.json").write_text(
            json.dumps(
                {
                    "figure_id": figure_id,
                    "core_claim": f"{figure_id} supporting display.",
                    "panels": [
                        {
                            "panel_id": chr(65 + index),
                            "title": role.replace("_", " ").title(),
                            "role": role,
                            "chart_type": "dot_interval",
                            "claim": f"{role} is shown.",
                        }
                        for index, role in enumerate(roles)
                    ],
                }
            ),
            encoding="utf-8",
        )

    context = ra.ResearchContext(
        research_question="Estimate whether Sepsis-3 is associated with mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="01_table_one",
                intent="Summarise baseline characteristics.",
                expected_outputs=["table:table_one"],
            ),
            ra.AnalysisStep(
                step_id="02_model",
                intent="Fit adjusted association model.",
                expected_outputs=["table:adjusted_association"],
            ),
            ra.AnalysisStep(
                step_id="03_old_primary_render",
                intent="Render an older primary-like step figure.",
                expected_outputs=["figure:publication_figure"],
            ),
            ra.AnalysisStep(
                step_id="04_supporting_missingness",
                intent="Render a supporting missingness figure.",
                expected_outputs=["figure:missingness_measurement_panel"],
            ),
            ra.AnalysisStep(
                step_id="05_duplicate_missingness",
                intent="Render a duplicate supporting missingness figure.",
                expected_outputs=["figure:missingness_measurement_panel"],
            ),
            ra.AnalysisStep(
                step_id="06_sensitivity",
                intent="Render a robustness figure.",
                expected_outputs=["figure:sensitivity_forest"],
            ),
        ],
    )
    evidence = EvidenceStore(tmp_path)
    bound_evidence = _register_complete_display_suite_for_readiness(
        evidence,
        tmp_path,
        table_step_id="01_table_one",
        publication_source_step_id="02_model",
    )
    write_support_contract(
        "03_old_primary_render",
        "publication_figure",
        figure_id="publication_figure",
        roles=["descriptive_result", "relationship"],
    )
    write_support_contract(
        "04_supporting_missingness",
        "missingness_measurement_panel",
        figure_id="missingness_measurement_panel",
        roles=["audit"],
    )
    write_support_contract(
        "05_duplicate_missingness",
        "missingness_measurement_panel",
        figure_id="missingness_measurement_panel",
        roles=["audit"],
    )
    write_support_contract(
        "06_sensitivity",
        "sensitivity_forest",
        figure_id="sensitivity_forest",
        roles=["robustness"],
    )
    bound_path = tmp_path / "manuscript_scaffold_bound.md"
    bound_path.write_text(_evidence_bound_demo_manuscript(), encoding="utf-8")

    gates, artifact_paths = _write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=[],
        per_step_records=[
            {
                "step_id": "01_table_one",
                "status": "ok",
                "evidence_ids": bound_evidence["01_table_one"],
            },
            {
                "step_id": "02_model",
                "status": "ok",
                "evidence_ids": bound_evidence["02_model"],
            },
            {"step_id": "03_old_primary_render", "status": "ok"},
            {"step_id": "04_supporting_missingness", "status": "ok"},
            {"step_id": "05_duplicate_missingness", "status": "ok"},
            {"step_id": "06_sensitivity", "status": "ok"},
        ],
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=bound_path,
        stop_after_analysis=False,
    )

    assert gates["display_suite_complete"] is True
    review_artifacts = json.loads(
        (tmp_path / artifact_paths["review_artifacts"]).read_text(encoding="utf-8")
    )
    visible_paths = [
        row["relative_path"] for row in review_artifacts["supporting_figures"]
    ]
    assert visible_paths == [
        "steps/04_supporting_missingness/outputs/missingness_measurement_panel.png",
        "steps/06_sensitivity/outputs/sensitivity_forest.png",
    ]
    archived = review_artifacts["archived_supporting_figures"]
    assert {row["archive_reason"] for row in archived} == {
        "covered_by_primary_publication_figure",
        "duplicate_supporting_figure_id",
    }
    assert all(row["status"] == "archived_supporting" for row in archived)

    figure_gallery = json.loads(
        (tmp_path / artifact_paths["figure_gallery"]).read_text(encoding="utf-8")
    )
    assert figure_gallery["primary_count"] == 1
    assert figure_gallery["supporting_count"] == 2
    assert figure_gallery["archived_supporting_count"] == 2
    assert [row["relative_path"] for row in figure_gallery["figures"]] == [
        "publication_figures/easyicu_publication_figure.png",
        "steps/04_supporting_missingness/outputs/missingness_measurement_panel.png",
        "steps/06_sensitivity/outputs/sensitivity_forest.png",
    ]
    assert all(
        "covered_by_primary_publication_figure" != row.get("archive_reason")
        for row in figure_gallery["figures"]
    )


def test_article_figure_strategy_rejects_sparse_primary_publication_figure(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.planning.figure_strategy import (
        summarize_article_figure_strategy_coverage,
    )

    context = ra.ResearchContext(
        research_question="Estimate whether Sepsis-3 is associated with mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
        target_outcome="death",
    )
    publication_dir = tmp_path / "publication_figures"
    publication_dir.mkdir(parents=True)
    (publication_dir / "easyicu_publication_figure.figure_contract.json").write_text(
        json.dumps(
            {
                "figure_id": "easyicu_publication_figure",
                "core_claim": "Prevalence and the primary adjusted estimate are shown.",
                "panels": [
                    {
                        "panel_id": "A",
                        "title": "Prevalence and absolute outcome risk",
                        "role": "descriptive_result",
                        "chart_type": "dot_interval_absolute_risk",
                        "claim": "Exposure prevalence and absolute outcome risk are shown before adjusted estimates.",
                    },
                    {
                        "panel_id": "B",
                        "title": "Primary adjusted association",
                        "role": "primary_estimand",
                        "chart_type": "forest",
                        "claim": "The adjusted odds ratio and confidence interval are shown.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    support_dir = tmp_path / "steps" / "05_sensitivity" / "outputs"
    support_dir.mkdir(parents=True)
    (support_dir / "supporting_sensitivity.figure_contract.json").write_text(
        json.dumps(
            {
                "figure_id": "supporting_sensitivity",
                "core_claim": "Supporting data-quality and robustness panels are shown.",
                "panels": [
                    {
                        "panel_id": "C",
                        "title": "Sensitivity and denominator audit",
                        "role": "robustness",
                        "chart_type": "specification_grid",
                        "claim": "Alternative definitions and denominators are shown.",
                    },
                    {
                        "panel_id": "D",
                        "title": "Missingness and measurement availability",
                        "role": "data_quality",
                        "chart_type": "availability_panel",
                        "claim": "Missingness and measurement availability are visible.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    status = summarize_article_figure_strategy_coverage(
        context=context,
        run_dir=tmp_path,
    )

    assert set(status["article_figure_strategy_covered_roles"]) == {
        "data_quality",
        "descriptive_result",
        "primary_estimand",
        "robustness",
    }
    assert status["article_figure_strategy_primary_publication_roles"] == [
        "descriptive_result",
        "primary_estimand",
    ]
    assert status["article_figure_strategy_complete"] is False
    assert any(
        "Primary publication figure covers fewer required visual roles" in err
        for err in status["article_figure_strategy_errors"]
    )


def test_association_display_suite_rejects_generic_chart_only_bundle(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _write_readiness_artifacts

    context = ra.ResearchContext(
        research_question="Estimate whether Sepsis-3 is associated with mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="01_table_one",
                intent="Summarise baseline characteristics.",
                expected_outputs=["table:table_one"],
            ),
            ra.AnalysisStep(
                step_id="02_model",
                intent="Fit adjusted association model.",
                expected_outputs=["table:adjusted_association"],
            ),
            ra.AnalysisStep(
                step_id="03_figures",
                intent="Render forest, bar, and heatmap result panels.",
                expected_outputs=["figure:publication_figure"],
            ),
        ],
    )
    evidence = EvidenceStore(tmp_path)
    table_one_path = tmp_path / "table_one.csv"
    table_one_path.write_text("variable,value\nage,64\n", encoding="utf-8")
    evidence.register_file(
        kind="table",
        description="Table 1 baseline cohort characteristics.",
        source_path=table_one_path,
        evidence_id="table_table_one",
        producer="coder",
        generation_mode="llm_code",
    )
    _register_publication_bundle_for_readiness(
        evidence,
        tmp_path,
        contract={
            "figure_id": "easyicu_publication_figure",
            "core_claim": "Generic audited result panels.",
            "panels": [
                {
                    "panel_id": "A",
                    "title": "Adjusted association forest",
                    "role": "relationship",
                    "chart_type": "forest",
                    "claim": "Adjusted odds ratios are shown.",
                },
                {
                    "panel_id": "B",
                    "title": "Denominator bar chart",
                    "role": "audit",
                    "chart_type": "bar",
                    "claim": "Denominators are audited.",
                },
                {
                    "panel_id": "C",
                    "title": "Overlap heatmap",
                    "role": "robustness",
                    "chart_type": "heatmap",
                    "claim": "Definition overlap is shown.",
                },
            ],
        },
    )
    bound_path = tmp_path / "manuscript_scaffold_bound.md"
    bound_path.write_text(_evidence_bound_demo_manuscript(), encoding="utf-8")

    gates, _ = _write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=[],
        per_step_records=[
            {"step_id": "01_table_one", "status": "ok"},
            {"step_id": "02_model", "status": "ok"},
            {"step_id": "03_figures", "status": "ok"},
        ],
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=bound_path,
        stop_after_analysis=False,
    )

    assert gates["display_suite_complete"] is False
    assert gates["publication_ready"] is False
    assert gates["display_chart_types"] == ["bar", "forest", "heatmap"]
    assert gates["display_absolute_risk_visual_present"] is False
    assert any(
        "lacks a visual prevalence" in err for err in gates["display_suite_errors"]
    )
    assert any(
        "generic bar/forest/heatmap" in err for err in gates["display_suite_errors"]
    )


def test_association_display_suite_rejects_risk_difference_without_absolute_risk_context(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _write_readiness_artifacts

    context = ra.ResearchContext(
        research_question="Estimate whether Sepsis-3 is associated with mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="01_table_one",
                intent="Summarise baseline characteristics.",
                expected_outputs=["table:table_one"],
            ),
            ra.AnalysisStep(
                step_id="02_model",
                intent="Fit adjusted association model.",
                expected_outputs=["table:adjusted_association"],
            ),
            ra.AnalysisStep(
                step_id="03_sensitivity",
                intent="Render adjusted estimate, risk difference, and denominator audit.",
                expected_outputs=["figure:publication_figure"],
            ),
        ],
    )
    evidence = EvidenceStore(tmp_path)
    table_one_path = tmp_path / "table_one.csv"
    table_one_path.write_text("variable,value\nage,64\n", encoding="utf-8")
    evidence.register_file(
        kind="table",
        description="Table 1 baseline cohort characteristics.",
        source_path=table_one_path,
        evidence_id="table_table_one",
        producer="coder",
        generation_mode="llm_code",
    )
    _register_publication_bundle_for_readiness(
        evidence,
        tmp_path,
        contract={
            "figure_id": "easyicu_publication_figure",
            "core_claim": "Adjusted association, risk difference, and denominator audit.",
            "panels": [
                {
                    "panel_id": "A",
                    "title": "Adjusted odds-ratio estimate",
                    "role": "relationship",
                    "chart_type": "forest",
                    "claim": "The adjusted odds ratio is shown.",
                },
                {
                    "panel_id": "B",
                    "title": "Risk difference sensitivity",
                    "role": "relationship",
                    "chart_type": "dot_interval",
                    "claim": "Risk-difference sensitivity estimates are shown.",
                },
                {
                    "panel_id": "C",
                    "title": "Denominator audit",
                    "role": "audit",
                    "chart_type": "bar",
                    "claim": "Analytic sample sizes are shown.",
                },
            ],
        },
    )
    bound_path = tmp_path / "manuscript_scaffold_bound.md"
    bound_path.write_text(_evidence_bound_demo_manuscript(), encoding="utf-8")

    gates, _ = _write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=[],
        per_step_records=[
            {"step_id": "01_table_one", "status": "ok"},
            {"step_id": "02_model", "status": "ok"},
            {"step_id": "03_sensitivity", "status": "ok"},
        ],
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=bound_path,
        stop_after_analysis=False,
    )

    assert gates["display_suite_complete"] is False
    assert gates["publication_ready"] is False
    assert gates["display_absolute_risk_visual_present"] is False
    assert "dot_interval" in gates["display_chart_types"]
    assert gates["article_figure_strategy_complete"] is False
    assert any(
        "risk-difference sensitivity panels alone" in err
        for err in gates["display_suite_errors"]
    )
    assert any(
        "descriptive_result" in err for err in gates["article_figure_strategy_errors"]
    )


def test_readiness_supersedes_stale_publication_figure_contract_quality_error(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _write_readiness_artifacts
    from easyicu.research_agent.schema import ValidationFinding

    context = ra.ResearchContext(
        research_question="Estimate whether Sepsis-3 is associated with mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="01_table_one",
                intent="Summarise baseline characteristics.",
                expected_outputs=["table:table_one"],
            ),
            ra.AnalysisStep(
                step_id="02_model",
                intent="Fit adjusted association model.",
                expected_outputs=["table:primary_estimand"],
                planned_analysis_role="primary",
            ),
            ra.AnalysisStep(
                step_id="03_sensitivity",
                intent="Audit sensitivity and denominator robustness.",
                expected_outputs=["figure:sensitivity_audit"],
            ),
        ],
    )
    evidence = EvidenceStore(tmp_path)
    current_evidence = _register_complete_display_suite_for_readiness(
        evidence,
        tmp_path,
        table_step_id="01_table_one",
        publication_source_step_id="02_model",
    )
    _write_publication_skill_summary(
        evidence,
        version=2,
        audit_findings=[
            {
                "validator": "figure_contract_quality",
                "severity": "warning",
                "message": "Publication figure contract now has required panel roles.",
            }
        ],
    )
    bound_path = tmp_path / "manuscript_scaffold_bound.md"
    bound_path.write_text(_evidence_bound_demo_manuscript(), encoding="utf-8")

    gates, _ = _write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=[
            ValidationFinding(
                validator="figure_contract_quality",
                severity="error",
                message=(
                    "Publication figure 'easyicu_publication_figure' lacks "
                    "required panel-role diversity."
                ),
            )
        ],
        per_step_records=_authoritative_readiness_records(plan, current_evidence),
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=bound_path,
        stop_after_analysis=False,
    )

    assert gates["analysis_validated"] is True
    assert gates["analysis_error_count"] == 0
    assert gates["superseded_error_count"] == 1
    assert gates["publication_ready"] is True


def test_author_review_note_marks_superseded_publication_export_error_nonblocking(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _write_readiness_artifacts
    from easyicu.research_agent.schema import ValidationFinding

    context = ra.ResearchContext(
        research_question="Estimate whether Sepsis-3 is associated with mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="01_table_one",
                intent="Summarise baseline characteristics.",
                expected_outputs=["table:table_one"],
            ),
            ra.AnalysisStep(
                step_id="02_model",
                intent="Fit adjusted association model.",
                expected_outputs=["table:primary_estimand"],
                planned_analysis_role="primary",
            ),
            ra.AnalysisStep(
                step_id="03_sensitivity",
                intent="Audit sensitivity and denominator robustness.",
                expected_outputs=["figure:sensitivity_audit"],
            ),
        ],
    )
    evidence = EvidenceStore(tmp_path)
    current_evidence = _register_complete_display_suite_for_readiness(
        evidence,
        tmp_path,
        table_step_id="01_table_one",
        publication_source_step_id="02_model",
    )
    _write_publication_skill_summary(evidence, version=2, audit_findings=[])
    stale_error = ValidationFinding(
        validator="publication_figure_export",
        severity="error",
        message=(
            "SVG figure 'easyicu_publication_figure.svg' has overlapping text "
            "elements; multi-panel labels need more spacing."
        ),
    )
    bound_path = tmp_path / "manuscript_scaffold_bound.md"
    bound_path.write_text(_evidence_bound_demo_manuscript(), encoding="utf-8")

    gates, _ = _write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=[stale_error],
        per_step_records=_authoritative_readiness_records(plan, current_evidence),
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=bound_path,
        stop_after_analysis=False,
    )

    assert gates["superseded_error_count"] == 1
    assert gates["publication_figure_bundle_ready"] is True
    assert gates["publication_ready"] is True
    author_review = (tmp_path / "author_review_note.md").read_text(encoding="utf-8")
    blocking_section = author_review.split("## Superseded findings", maxsplit=1)[0]
    assert "`publication_figure_export`" not in blocking_section
    assert "## Superseded findings" in author_review
    assert "`publication_figure_export`" in author_review


def test_readiness_keeps_current_publication_figure_export_error_active(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _write_readiness_artifacts
    from easyicu.research_agent.schema import ValidationFinding

    context = ra.ResearchContext(
        research_question="Estimate whether Sepsis-3 is associated with mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="01_table_one",
                intent="Summarise baseline characteristics.",
                expected_outputs=["table:table_one"],
            ),
            ra.AnalysisStep(
                step_id="02_model",
                intent="Fit adjusted association model.",
                expected_outputs=["table:adjusted_association"],
            ),
            ra.AnalysisStep(
                step_id="03_sensitivity",
                intent="Audit sensitivity and denominator robustness.",
                expected_outputs=["figure:sensitivity_audit"],
            ),
        ],
    )
    evidence = EvidenceStore(tmp_path)
    _register_complete_display_suite_for_readiness(evidence, tmp_path)
    current_error = {
        "validator": "publication_figure_export",
        "severity": "error",
        "message": (
            "SVG figure 'easyicu_publication_figure.svg' has overlapping text "
            "elements; multi-panel labels need more spacing."
        ),
    }
    _write_publication_skill_summary(
        evidence,
        version=2,
        audit_findings=[current_error],
    )
    bound_path = tmp_path / "manuscript_scaffold_bound.md"
    bound_path.write_text(_evidence_bound_demo_manuscript(), encoding="utf-8")

    gates, _ = _write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=[ValidationFinding(**current_error)],
        per_step_records=[
            {"step_id": "01_table_one", "status": "ok"},
            {"step_id": "02_model", "status": "ok"},
            {"step_id": "03_sensitivity", "status": "ok"},
        ],
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=bound_path,
        stop_after_analysis=False,
    )

    assert gates["analysis_validated"] is False
    assert gates["analysis_error_count"] == 1
    assert gates["superseded_error_count"] == 0
    assert gates["publication_figure_bundle_ready"] is False
    assert gates["publication_figure_visual_qa_passed"] is False
    assert gates["publication_figure_visual_qa_error_count"] == 1
    assert gates["publication_ready"] is False


def test_readiness_supersedes_stale_strict_writer_error_after_clean_bound_manuscript(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _write_readiness_artifacts
    from easyicu.research_agent.schema import ValidationFinding

    context = ra.ResearchContext(
        research_question="Estimate whether Sepsis-3 is associated with mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="01_table_one",
                intent="Summarise baseline characteristics.",
                expected_outputs=["table:table_one"],
            ),
            ra.AnalysisStep(
                step_id="02_model",
                intent="Fit adjusted association model.",
                expected_outputs=["table:primary_estimand"],
                planned_analysis_role="primary",
            ),
            ra.AnalysisStep(
                step_id="03_sensitivity",
                intent="Audit sensitivity and denominator robustness.",
                expected_outputs=["figure:sensitivity_audit"],
            ),
        ],
    )
    evidence = EvidenceStore(tmp_path)
    current_evidence = _register_complete_display_suite_for_readiness(
        evidence,
        tmp_path,
        table_step_id="01_table_one",
        publication_source_step_id="02_model",
    )
    bound_path = tmp_path / "manuscript_scaffold_bound.md"
    bound_path.write_text(_evidence_bound_demo_manuscript(), encoding="utf-8")

    gates, _ = _write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=[
            ValidationFinding(
                validator="evidence_bound_writer",
                severity="error",
                message=(
                    "STRICT evidence enforcement blocked manuscript generation: "
                    "STRICT evidence mode: 2 result-like sentence(s) without "
                    "{evidence:<id>} placeholders."
                ),
            )
        ],
        per_step_records=_authoritative_readiness_records(plan, current_evidence),
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=bound_path,
        stop_after_analysis=False,
    )

    assert gates["evidence_complete"] is True
    assert gates["superseded_error_count"] == 1
    assert gates["evidence_error_count"] == 0
    assert gates["publication_ready"] is True


def test_readiness_supersedes_stale_numeric_error_after_clean_bound_manuscript(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _write_readiness_artifacts
    from easyicu.research_agent.schema import ValidationFinding

    context = ra.ResearchContext(
        research_question="Estimate whether Sepsis-3 is associated with mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="01_table_one",
                intent="Summarise baseline characteristics.",
                expected_outputs=["table:table_one"],
            ),
            ra.AnalysisStep(
                step_id="02_model",
                intent="Fit adjusted association model.",
                expected_outputs=["table:primary_estimand"],
                planned_analysis_role="primary",
            ),
            ra.AnalysisStep(
                step_id="03_sensitivity",
                intent="Audit sensitivity and denominator robustness.",
                expected_outputs=["figure:sensitivity_audit"],
            ),
        ],
    )
    evidence = EvidenceStore(tmp_path)
    current_evidence = _register_complete_display_suite_for_readiness(
        evidence,
        tmp_path,
        table_step_id="01_table_one",
        publication_source_step_id="02_model",
    )
    bound_path = tmp_path / "manuscript_scaffold_bound.md"
    bound_path.write_text(
        "The adjusted association used 10 stays[^claim_1].\n\n"
        "[^claim_1]: value=10; step=02_model; field=n_final_model; evidence=e_model\n",
        encoding="utf-8",
    )

    gates, _ = _write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=[
            ValidationFinding(
                validator="manuscript_numeric_auditor",
                severity="error",
                message=(
                    "STRICT evidence enforcement blocked manuscript generation: "
                    "Manuscript contains 1 numeric value(s) not traceable to any "
                    "registered claim (STRICT mode)."
                ),
            )
        ],
        per_step_records=_authoritative_readiness_records(plan, current_evidence),
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=bound_path,
        stop_after_analysis=False,
    )

    assert gates["numeric_verified"] is True
    assert gates["superseded_error_count"] == 1
    assert gates["numeric_error_count"] == 0
    assert gates["publication_ready"] is True


def test_readiness_supersedes_stale_critic_error_after_passed_current_critique(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _write_readiness_artifacts
    from easyicu.research_agent.schema import ValidationFinding

    context = ra.ResearchContext(
        research_question="Estimate whether Sepsis-3 is associated with mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="01_table_one",
                intent="Summarise baseline characteristics.",
                expected_outputs=["table:table_one"],
            ),
            ra.AnalysisStep(
                step_id="02_model",
                intent="Fit adjusted association model.",
                expected_outputs=["table:primary_estimand"],
                planned_analysis_role="primary",
            ),
            ra.AnalysisStep(
                step_id="03_sensitivity",
                intent="Audit sensitivity and denominator robustness.",
                expected_outputs=["figure:sensitivity_audit"],
            ),
        ],
    )
    evidence = EvidenceStore(tmp_path)
    current_evidence = _register_complete_display_suite_for_readiness(
        evidence,
        tmp_path,
        table_step_id="01_table_one",
        publication_source_step_id="02_model",
    )
    bound_path = tmp_path / "manuscript_scaffold_bound.md"
    bound_path.write_text(_evidence_bound_demo_manuscript(), encoding="utf-8")
    (tmp_path / "manuscript_critique.json").write_text(
        '{"status":"pass","concerns":[],"unsupported_claims":[]}',
        encoding="utf-8",
    )

    gates, _ = _write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=[
            ValidationFinding(
                validator="critic_agent",
                severity="error",
                message=(
                    "CriticAgent marked manuscript as needs_revision: "
                    "Some result-like sentences were filtered or remain unsupported."
                ),
            )
        ],
        per_step_records=_authoritative_readiness_records(plan, current_evidence),
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=bound_path,
        stop_after_analysis=False,
    )

    assert gates["evidence_complete"] is True
    assert gates["evidence_error_count"] == 0
    assert gates["superseded_error_count"] == 1
    assert gates["publication_ready"] is True


def _readiness_fixture_for_manifest_caveats(ra, tmp_path: Path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.schema import ValidationFinding

    context = ra.ResearchContext(
        research_question="Estimate whether Sepsis-3 is associated with mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="01_table_one",
                intent="Summarise baseline characteristics.",
                expected_outputs=["table:table_one"],
            ),
            ra.AnalysisStep(
                step_id="02_model",
                intent="Fit adjusted association model.",
                expected_outputs=["table:primary_estimand"],
                planned_analysis_role="primary",
            ),
            ra.AnalysisStep(
                step_id="03_sensitivity",
                intent="Audit sensitivity and denominator robustness.",
                expected_outputs=["figure:sensitivity_audit"],
            ),
        ],
    )
    evidence = EvidenceStore(tmp_path)
    current_evidence = _register_complete_display_suite_for_readiness(
        evidence,
        tmp_path,
        table_step_id="01_table_one",
        publication_source_step_id="02_model",
    )
    caveat_finding = ValidationFinding(
        validator="evidence_bound_writer",
        severity="error",
        message=(
            "Bound manuscript cites evidence records with unresolved "
            "manifest caveats: 0 error and 4 warning comment(s)."
        ),
    )
    per_step_records = _authoritative_readiness_records(plan, current_evidence)
    return context, plan, evidence, caveat_finding, per_step_records


def test_readiness_supersedes_stale_manifest_caveat_error_when_current_bound_is_clean(
    ra,
    tmp_path: Path,
):
    """A caveat-count error from an earlier writer pass must retire when the
    latest bound manuscript carries no manifest-caveat comments (e.g. a
    resume whose rewrite cites only caveat-free records)."""
    from easyicu.research_agent.pipeline import _write_readiness_artifacts

    context, plan, evidence, caveat_finding, per_step_records = (
        _readiness_fixture_for_manifest_caveats(ra, tmp_path)
    )
    bound_path = tmp_path / "manuscript_scaffold_bound.md"
    bound_path.write_text(_evidence_bound_demo_manuscript(), encoding="utf-8")

    gates, _ = _write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=[caveat_finding],
        per_step_records=per_step_records,
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=bound_path,
        stop_after_analysis=False,
    )

    assert gates["evidence_complete"] is True
    assert gates["evidence_error_count"] == 0
    assert gates["superseded_error_count"] == 1
    assert gates["publication_ready"] is True


def test_readiness_keeps_manifest_caveat_error_when_current_bound_still_caveated(
    ra,
    tmp_path: Path,
):
    """Fail closed: while the latest bound manuscript still carries a
    manifest-caveat comment, the caveat-count error stays active."""
    from easyicu.research_agent.pipeline import _write_readiness_artifacts

    context, plan, evidence, caveat_finding, per_step_records = (
        _readiness_fixture_for_manifest_caveats(ra, tmp_path)
    )
    bound_path = tmp_path / "manuscript_scaffold_bound.md"
    bound_path.write_text(
        _evidence_bound_demo_manuscript()
        + "\nThe exposure derivation caveat remains open "
        "[table_one](evidence/table_one.csv)<!-- warning: see manifest -->.\n",
        encoding="utf-8",
    )

    gates, _ = _write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=[caveat_finding],
        per_step_records=per_step_records,
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=bound_path,
        stop_after_analysis=False,
    )

    assert gates["evidence_complete"] is False
    assert gates["evidence_error_count"] == 1
    assert gates["superseded_error_count"] == 0
    assert gates["manuscript_ready"] is False


def test_readiness_artifacts_block_outcome_leak_after_blocked_gate(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _write_readiness_artifacts

    context = ra.ResearchContext(
        research_question="Audit outcome linkage after a definition-sensitivity gate.",
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
                step_id="04_outcome_gate",
                intent="Check whether outcome linkage is authorized.",
                expected_outputs=["table:outcome_gate"],
            )
        ],
    )
    step_out = tmp_path / "steps" / "04_outcome_gate" / "outputs"
    step_out.mkdir(parents=True)
    blocked_summary = {
        "step_id": "04_outcome_gate",
        "primary_analysis_authorized": False,
        "grouped_death_analysis_executed": False,
        "target_outcome": "death",
    }
    (step_out / "step_summary.json").write_text(
        json.dumps(blocked_summary),
        encoding="utf-8",
    )
    evidence = EvidenceStore(tmp_path)
    bound_path = tmp_path / "manuscript_scaffold_bound.md"
    bound_path.write_text(
        "The exploratory association with death was near-null between definition "
        "groups [outcome_gate](evidence/outcome_gate.csv).\n",
        encoding="utf-8",
    )

    gates, artifact_paths = _write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=[],
        per_step_records=[
            {
                "step_id": "04_outcome_gate",
                "status": "ok",
                "step_summary": blocked_summary,
            }
        ],
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=bound_path,
        stop_after_analysis=False,
    )

    assert gates["execution_complete"] is True
    assert gates["blocked_outcome_step_ids"] == ["04_outcome_gate"]
    assert gates["blocked_outcome_not_leaked"] is False
    assert gates["analysis_validated"] is False
    assert gates["manuscript_ready"] is False
    assert gates["publication_ready"] is False
    assert "manuscript_ready" not in artifact_paths
    assert not (tmp_path / "manuscript_ready.md").exists()
    claim_ledger = (tmp_path / "claim_ledger.csv").read_text(encoding="utf-8")
    assert "blocked_outcome_leak" in claim_ledger
    run_status = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    assert run_status["status"] == "analysis_only"


def test_publication_bundle_ready_groups_hash_suffixed_exports_under_one_stem(
    ra, tmp_path: Path
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _publication_figure_bundle_ready
    from easyicu.research_agent.figures.publication import (
        PUBLICATION_FIGURE_SKILL_POLICY_VERSION,
    )

    evidence = EvidenceStore(tmp_path)
    contract_path = tmp_path / "easyicu_publication_figure.figure_contract.json"
    contract_path.write_text("{}", encoding="utf-8")
    source_path = tmp_path / "publication_figure_source.csv"
    source_path.write_text("x,y\n1,2\n", encoding="utf-8")
    source_record = evidence.register_file(
        kind="table",
        description="Publication figure source data.",
        source_path=source_path,
        evidence_id="publication_figure_source_demo",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
    )
    source_metadata = {
        "figure_skill_policy_version": PUBLICATION_FIGURE_SKILL_POLICY_VERSION,
        "source_evidence_id": source_record.evidence_id,
        "source_evidence_ids": [source_record.evidence_id],
        "source_evidence_sha256": {source_record.evidence_id: source_record.sha256},
    }
    evidence.register_file(
        kind="log",
        description="Publication figure contract.",
        source_path=contract_path,
        evidence_id="publication_figure_contract",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
        metadata=source_metadata,
    )
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
            metadata={"figure_role": "publication_figure", **source_metadata},
        )

    readiness = _publication_figure_bundle_ready(evidence=evidence, run_dir=tmp_path)

    assert readiness["publication_figure_bundle_ready"] is True
    assert readiness["publication_ready_stems"] == ["easyicu_publication_figure"]
    assert readiness["publication_figure_contract_ready"] is True
    assert readiness["publication_figure_source_data_ready"] is True


def test_publication_bundle_requires_sources_from_current_checkpoint(
    ra, tmp_path: Path
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _publication_figure_bundle_ready
    from easyicu.research_agent.figures.publication import (
        PUBLICATION_FIGURE_SKILL_POLICY_VERSION,
    )

    evidence = EvidenceStore(tmp_path)
    source_path = tmp_path / "current_source.csv"
    source_path.write_text("x,y\n1,2\n", encoding="utf-8")
    source_record = evidence.register_file(
        kind="table",
        description="Current publication source.",
        source_path=source_path,
        evidence_id="current_source",
        produced_by_step="02_model",
        producer="coder",
        generation_mode="llm",
    )
    metadata = {
        "figure_skill_policy_version": PUBLICATION_FIGURE_SKILL_POLICY_VERSION,
        "source_evidence_id": source_record.evidence_id,
        "source_evidence_ids": [source_record.evidence_id],
        "source_evidence_sha256": {source_record.evidence_id: source_record.sha256},
    }
    contract_path = tmp_path / "easyicu_publication_figure.figure_contract.json"
    contract_path.write_text("{}", encoding="utf-8")
    evidence.register_file(
        kind="log",
        description="Publication figure contract.",
        source_path=contract_path,
        evidence_id="publication_figure_contract",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
        metadata=metadata,
    )
    for suffix in ("svg", "png"):
        path = tmp_path / f"easyicu_publication_figure.{suffix}"
        path.write_text("x", encoding="utf-8")
        evidence.register_file(
            kind="figure",
            description="Publication figure export.",
            source_path=path,
            evidence_id=f"publication_figure_{suffix}",
            producer="publication_figure_skill",
            generation_mode="deterministic_figure_skill",
            metadata={"figure_role": "publication_figure", **metadata},
        )

    current = [
        {
            "step_id": "02_model",
            "status": "ok",
            "evidence_ids": [source_record.evidence_id],
        }
    ]
    assert (
        _publication_figure_bundle_ready(
            evidence=evidence,
            run_dir=tmp_path,
            per_step_records=current,
        )["publication_figure_bundle_ready"]
        is True
    )

    superseded = [
        *current,
        {"step_id": "02_model", "status": "contract_failed"},
    ]
    stale = _publication_figure_bundle_ready(
        evidence=evidence,
        run_dir=tmp_path,
        per_step_records=superseded,
    )
    assert stale["publication_figure_bundle_ready"] is False
    assert stale["publication_figure_contract_ready"] is False
    assert stale["publication_ready_stems"] == []


def test_publication_bundle_ready_rejects_outdated_figure_policy(ra, tmp_path: Path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _publication_figure_bundle_ready

    evidence = EvidenceStore(tmp_path)
    contract_path = tmp_path / "easyicu_publication_figure.figure_contract.json"
    contract_path.write_text("{}", encoding="utf-8")
    source_path = tmp_path / "publication_figure_source.csv"
    source_path.write_text("x,y\n1,2\n", encoding="utf-8")
    source_record = evidence.register_file(
        kind="table",
        description="Publication figure source data.",
        source_path=source_path,
        evidence_id="publication_figure_source_demo",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
    )
    outdated_metadata = {
        "source_evidence_id": source_record.evidence_id,
        "source_evidence_ids": [source_record.evidence_id],
        "source_evidence_sha256": {source_record.evidence_id: source_record.sha256},
    }
    evidence.register_file(
        kind="log",
        description="Publication figure contract.",
        source_path=contract_path,
        evidence_id="publication_figure_contract",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
        metadata=outdated_metadata,
    )
    for suffix in ("svg", "png"):
        path = tmp_path / f"easyicu_publication_figure.{suffix}"
        path.write_text("x", encoding="utf-8")
        evidence.register_file(
            kind="figure",
            description="Publication figure export.",
            source_path=path,
            evidence_id=f"publication_figure_{suffix}",
            producer="publication_figure_skill",
            generation_mode="deterministic_figure_skill",
            metadata={"figure_role": "publication_figure", **outdated_metadata},
        )

    readiness = _publication_figure_bundle_ready(evidence=evidence, run_dir=tmp_path)

    assert readiness["publication_figure_bundle_ready"] is False
    assert readiness["publication_ready_stems"] == []


def test_publication_bundle_ready_rejects_stale_publication_skill_exports(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _publication_figure_bundle_ready

    evidence = EvidenceStore(tmp_path)
    contract_path = tmp_path / "easyicu_publication_figure.figure_contract.json"
    contract_path.write_text("{}", encoding="utf-8")
    source_path = tmp_path / "publication_figure_source.csv"
    source_path.write_text("x,y\n1,2\n", encoding="utf-8")
    evidence.register_file(
        kind="table",
        description="Publication figure source data.",
        source_path=source_path,
        evidence_id="publication_figure_source_demo",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
    )
    stale_metadata = {
        "source_evidence_id": "publication_figure_source_demo",
        "source_evidence_ids": ["publication_figure_source_demo"],
    }
    evidence.register_file(
        kind="log",
        description="Publication figure contract.",
        source_path=contract_path,
        evidence_id="publication_figure_contract",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
        metadata=stale_metadata,
    )
    for suffix in ("svg", "png"):
        path = tmp_path / f"easyicu_publication_figure.{suffix}"
        path.write_text("x", encoding="utf-8")
        evidence.register_file(
            kind="figure",
            description="Publication figure export.",
            source_path=path,
            evidence_id=f"publication_figure_{suffix}",
            producer="publication_figure_skill",
            generation_mode="deterministic_figure_skill",
            metadata={"figure_role": "publication_figure", **stale_metadata},
        )

    readiness = _publication_figure_bundle_ready(evidence=evidence, run_dir=tmp_path)

    assert readiness["publication_figure_bundle_ready"] is False
    assert readiness["publication_ready_stems"] == []


def test_publication_bundle_ready_rejects_uncontracted_forest_plot_png_svg(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _publication_figure_bundle_ready

    evidence = EvidenceStore(tmp_path)
    for suffix in ("png", "svg"):
        path = tmp_path / f"forest_plot.{suffix}"
        path.write_text("x", encoding="utf-8")
        evidence.register_file(
            kind="figure",
            description="Forest plot for the association model.",
            source_path=path,
            evidence_id=f"figure_forest_plot_{suffix}",
            producer="coder",
            generation_mode="llm_code",
        )

    readiness = _publication_figure_bundle_ready(evidence=evidence, run_dir=tmp_path)

    assert readiness["publication_figure_bundle_ready"] is False
    assert readiness["publication_ready_stems"] == []


def test_publication_bundle_ready_blocks_visual_qa_errors(ra, tmp_path: Path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _publication_figure_bundle_ready
    from easyicu.research_agent.figures.publication import (
        PUBLICATION_FIGURE_SKILL_POLICY_VERSION,
    )
    from easyicu.research_agent.schema import ValidationFinding

    evidence = EvidenceStore(tmp_path)
    contract_path = tmp_path / "easyicu_publication_figure.figure_contract.json"
    contract_path.write_text("{}", encoding="utf-8")
    source_path = tmp_path / "publication_figure_source.csv"
    source_path.write_text("x,y\n1,2\n", encoding="utf-8")
    source_record = evidence.register_file(
        kind="table",
        description="Publication figure source data.",
        source_path=source_path,
        evidence_id="publication_figure_source_demo",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
    )
    source_metadata = {
        "figure_skill_policy_version": PUBLICATION_FIGURE_SKILL_POLICY_VERSION,
        "source_evidence_id": source_record.evidence_id,
        "source_evidence_ids": [source_record.evidence_id],
        "source_evidence_sha256": {source_record.evidence_id: source_record.sha256},
    }
    evidence.register_file(
        kind="log",
        description="Publication figure contract.",
        source_path=contract_path,
        evidence_id="publication_figure_contract",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
        metadata=source_metadata,
    )
    for suffix in ("svg", "png"):
        path = tmp_path / f"easyicu_publication_figure.{suffix}"
        path.write_text("x", encoding="utf-8")
        evidence.register_file(
            kind="figure",
            description="Publication figure export.",
            source_path=path,
            evidence_id=f"publication_figure_{suffix}",
            producer="publication_figure_skill",
            generation_mode="deterministic_figure_skill",
            metadata={
                "figure_role": "publication_figure",
                "figure_contract": "publication_figure_contract",
                **source_metadata,
            },
        )

    readiness = _publication_figure_bundle_ready(
        evidence=evidence,
        run_dir=tmp_path,
        findings=[
            ValidationFinding(
                validator="visual_qa",
                severity="error",
                message="Could not open figure 'easyicu_publication_figure.png'",
            )
        ],
    )

    assert readiness["publication_figure_bundle_ready"] is False
    assert readiness["publication_figure_visual_qa_passed"] is False


def test_publication_bundle_ready_blocks_publication_export_visual_errors(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.pipeline import _publication_figure_bundle_ready
    from easyicu.research_agent.schema import ValidationFinding

    evidence = EvidenceStore(tmp_path)
    _register_publication_bundle_for_readiness(
        evidence,
        tmp_path,
        contract={
            "figure_id": "easyicu_publication_figure",
            "core_claim": "Publication figure is exported.",
            "panels": [
                {
                    "panel_id": "A",
                    "title": "Primary adjusted association",
                    "role": "primary_estimand",
                    "chart_type": "forest",
                    "claim": "The adjusted association is shown.",
                }
            ],
        },
    )

    readiness = _publication_figure_bundle_ready(
        evidence=evidence,
        run_dir=tmp_path,
        findings=[
            ValidationFinding(
                validator="publication_figure_export",
                severity="error",
                message=(
                    "SVG figure 'easyicu_publication_figure.svg' has overlapping "
                    "text elements; multi-panel labels, annotations or axis text "
                    "need more spacing."
                ),
            )
        ],
    )

    assert readiness["publication_figure_bundle_ready"] is False
    assert readiness["publication_figure_visual_qa_passed"] is False
    assert readiness["publication_figure_visual_qa_error_count"] == 1
    assert readiness["publication_figure_visual_qa_errors"] == [
        {
            "validator": "publication_figure_export",
            "message": (
                "SVG figure 'easyicu_publication_figure.svg' has overlapping text "
                "elements; multi-panel labels, annotations or axis text need more "
                "spacing."
            ),
        }
    ]


def test_salvage_minimal_contract_step_summary_from_table_one_csv(ra, tmp_path: Path):
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


def test_critic_accepts_bound_markdown_evidence_links_but_blocks_hidden_missing_refs(
    ra,
):
    critic = ra.CriticAgent()

    clean = critic.review_manuscript(
        scaffold=(
            "The cohort comprised 1,000 ICU stays "
            '[research_context](evidence/research_context__research_context.json "sha256=abc"). '
            "The model achieved an AUROC of 0.78 "
            '[01_model_training](evidence/statistic_step_summary.json "sha256=def").'
        ),
        available_evidence_ids=["research_context", "01_model_training"],
    )

    assert clean.status == "pass"
    assert clean.unsupported_claims == []
    assert clean.missing_evidence_refs == []
    assert clean.suggested_repairs == []

    blocked = critic.review_manuscript(
        scaffold=(
            "The cohort comprised 1,000 ICU stays "
            '[research_context](evidence/research_context__research_context.json "sha256=abc"). '
            "<!-- evidence missing: table_one -->"
        ),
        available_evidence_ids=["research_context"],
    )

    assert blocked.status == "blocked"
    assert blocked.missing_evidence_refs == ["table_one"]
    assert blocked.suggested_repairs


def test_step_critic_does_not_recommend_repairs_after_a_clean_pass(ra):
    critic = ra.CriticAgent()
    step = ra.AnalysisStep(
        step_id="cohort_summary",
        intent="Describe the locked cohort.",
        expected_outputs=["table:cohort_summary"],
        method="descriptive_summary",
    )

    clean = critic.review_step(
        step=step,
        step_summary={"status": "ok", "n_rows": 12},
        evidence_refs=[ra.EvidenceRef(evidence_id="table_cohort_summary")],
        findings=[],
    )
    assert clean.status == "pass"
    assert clean.suggested_repairs == []

    blocked = critic.review_step(
        step=step,
        step_summary={"status": "ok", "n_rows": 12},
        evidence_refs=[],
        findings=[],
    )
    assert blocked.status == "blocked"
    assert blocked.suggested_repairs


def test_critic_does_not_flag_footnote_provenance_block(ra):
    # Regression (E2): the numeric binder appends a machine-provenance footnote
    # block (``[^claim_N]: value=...; step=...; evidence=<step>``). When a claim
    # binds to a step-level virtual evidence the footnote uses a plaintext
    # ``evidence=robustness_panel`` token (no ``](evidence/)`` link), carries
    # numbers + claimy words (auroc/brier), and has no end punctuation, so the
    # support check mis-read the whole block as one unsupported result sentence
    # and falsely set manuscript_ready=False. Footnote definition lines must be
    # excluded — the block PROVES the claims are bound.
    critic = ra.CriticAgent()
    scaffold = (
        "## Results\n"
        "Peak lactate was associated with death with a point estimate of 1.006[^claim_1] "
        '[primary_association](evidence/step_summary.json "sha256=abc").\n\n'
        "[^claim_1]: value=1.00643; step=robustness_panel; field=primary_point_estimate; evidence=robustness_panel "
        "[^claim_5]: value=0.788645; step=03_complete_case_robustness; field=auroc; evidence=robustness_panel "
        "[^claim_6]: value=0.0914; step=03_complete_case_robustness; field=brier_score; evidence=robustness_panel\n"
    )
    critique = critic.review_manuscript(
        scaffold=scaffold, available_evidence_ids=["primary_association"]
    )
    assert critique.status == "pass"
    assert critique.unsupported_claims == []

    # A genuine unsupported result sentence in the BODY is still caught.
    bad = (
        "## Results\n"
        "The model achieved an AUROC of 0.91 in the cohort and showed strong calibration.\n"
    )
    flagged = critic.review_manuscript(scaffold=bad, available_evidence_ids=[])
    assert flagged.status == "needs_revision"
    assert flagged.unsupported_claims


def test_critic_ignores_manuscript_metadata_sections(ra):
    critic = ra.CriticAgent()
    scaffold = (
        "## Results\n"
        "The cohort comprised 500 stays [cohort](evidence/cohort.json).\n\n"
        "## Data and code availability\n"
        "The cohort, generated scripts, SHA-256 evidence store, STROBE checklist, "
        "and supplementary tables are released alongside this manuscript.\n\n"
        "## Funding\n"
        "Funding information was not available to the analysis agent and should "
        "be completed by the authors before journal submission.\n"
    )

    critique = critic.review_manuscript(
        scaffold=scaffold,
        available_evidence_ids=["cohort"],
    )

    assert critique.status == "pass"
    assert critique.unsupported_claims == []


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
    critique = json.loads(
        (run_dir / "manuscript_critique.json").read_text(encoding="utf-8")
    )
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


def test_manuscript_critic_ignores_markdown_title_and_background_framing(ra):
    critic = ra.CriticAgent()
    scaffold = (
        "# Retrospective cohort study of 500 miiv ICU admissions reveals "
        "admission SOFA-2 score was associated with increased ICU mortality\n\n"
        "## Abstract\n\n"
        "**Background:**\n"
        "Clarifying this association could inform early risk stratification.\n\n"
        "**Results:**\n"
        "The cohort comprised 500 stays [cohort](evidence/cohort.json)."
    )

    critique = critic.review_manuscript(
        scaffold=scaffold,
        available_evidence_ids=["cohort"],
    )

    assert critique.status == "pass"
    assert critique.unsupported_claims == []


def test_evidence_filter_removes_unquantified_performance_claims(ra, tmp_path: Path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore

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
        '  File "analysis.py", line 142, in <module>\n'
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
        'covariates = ["age", "sex", "map_min_24h", "vaso_any_24h"]\n'
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

    repaired = _strip_columns_from_list_literals(code, ["age", "unknown_col"])

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
        'covariates = ["age", "sex", "map_min_24h", "vaso_any_24h"]\n'
        "model_df = df[[outcome_col, predictor_col] + covariates].copy()\n"
    )
    run_log = (
        "Traceback (most recent call last):\n"
        '  File "analysis.py", line 6, in <module>\n'
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
        f.severity == "warning" and "figure-producing" in f.message for f in findings
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


def test_preserve_figure_steps_after_replan_restores_exact_parent_products(ra):
    """An echoed pre-split parent must not strand the preserved render child."""
    from easyicu.research_agent.pipeline import (
        _preserve_figure_steps_after_replan,
    )

    current_parent = ra.AnalysisStep(
        step_id="01_model_training",
        intent="Fit the agent-selected prediction model.",
        method="prediction_model",
        expected_outputs=[
            "statistic:auroc",
            "table:model_performance",
            "table:roc_curve",
        ],
    )
    current_figure = ra.AnalysisStep(
        step_id="01_model_training_figure",
        intent=(
            "Render the publication figure declared by step " "'01_model_training'."
        ),
        method="visualization",
        inputs=["table:model_performance", "table:roc_curve"],
        expected_outputs=["figure:discrimination_calibration"],
    )
    current = ra.AnalysisPlan(
        research_question="build a prediction model",
        steps=[current_parent, current_figure],
    )
    # The replanner echoes the original parent shape and drops the host-split
    # child. It did not choose a different method or producer.
    revised = ra.AnalysisPlan(
        research_question="build a prediction model",
        steps=[
            current_parent.model_copy(update={"expected_outputs": ["statistic:auroc"]})
        ],
        revision=2,
    )

    preserved, findings = _preserve_figure_steps_after_replan(
        current=current,
        revised=revised,
    )

    by_id = {step.step_id: step for step in preserved.steps}
    assert by_id["01_model_training"].expected_outputs == [
        "statistic:auroc",
        "table:model_performance",
        "table:roc_curve",
    ]
    assert "01_model_training_figure" in by_id
    assert any(
        (finding.detail or {}).get("reason")
        == "preserved_figure_parent_output_contract"
        for finding in findings
    )


def test_preserve_figure_steps_after_replan_does_not_invent_missing_parent(ra):
    """A dropped producer remains a typed-DAG error; preservation cannot guess."""
    from easyicu.research_agent.pipeline import (
        _preserve_figure_steps_after_replan,
    )

    current_parent = ra.AnalysisStep(
        step_id="01_model_training",
        intent="Fit the agent-selected prediction model.",
        expected_outputs=["table:model_performance"],
    )
    current_figure = ra.AnalysisStep(
        step_id="01_model_training_figure",
        intent=(
            "Render the publication figure declared by step " "'01_model_training'."
        ),
        method="visualization",
        inputs=["table:model_performance"],
        expected_outputs=["figure:discrimination_calibration"],
    )
    current = ra.AnalysisPlan(
        research_question="build a prediction model",
        steps=[current_parent, current_figure],
    )
    revised = ra.AnalysisPlan(
        research_question="build a prediction model",
        steps=[
            ra.AnalysisStep(
                step_id="02_other",
                intent="Retain an unrelated descriptive step.",
                expected_outputs=[],
            )
        ],
        revision=2,
    )

    preserved, findings = _preserve_figure_steps_after_replan(
        current=current,
        revised=revised,
    )

    assert [step.step_id for step in preserved.steps] == [
        "02_other",
        "01_model_training_figure",
    ]
    assert all(
        (finding.detail or {}).get("reason")
        != "preserved_figure_parent_output_contract"
        for finding in findings
    )


def test_step_contract_repair_guidance_for_prediction_categorical_passthrough(ra):
    from easyicu.research_agent.pipeline import _step_contract_repair_guidance

    step = ra.AnalysisStep(
        step_id="03_model_training",
        method="prediction_model_evaluation",
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
        expected_outputs=[
            "table:adjusted_association",
            "figure:lactate_mortality_plot",
        ],
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
        method="kmeans_clustering",
        expected_outputs=[
            "statistic:cluster_count",
            "manifest:cluster_selection",
            "table:cluster_characteristics",
        ],
    )

    guidance = _step_contract_repair_guidance(
        step=step,
        step_summary={"error": "cluster selection evidence missing"},
        code="labels = kmeans.fit_predict(X)",
    )

    assert "full `cluster_selection`" in guidance
    assert "cluster_characteristics.csv" in guidance
    assert "self-contained" in guidance


def test_step_contract_repair_guidance_preserves_assignment_model_roster(ra):
    from easyicu.research_agent.pipeline import _step_contract_repair_guidance

    step = ra.AnalysisStep(
        step_id="balance",
        intent="Diagnose the Planner-owned assignment models.",
        inputs=["artifact:assignment_model"],
        expected_outputs=["artifact:balance_diagnostics"],
        method="positivity_and_balance_diagnostics",
    )
    guidance = _step_contract_repair_guidance(
        step=step,
        step_summary={
            "diagnostic_status": "not_computable",
            "skipped_reason": "model roster requires an explicit binding",
        },
        code="raise RuntimeError('model roster requires an explicit binding')",
        input_bindings={
            "artifact:assignment_model": {
                "product_contract": {
                    "models": [
                        {
                            "model_id": "declared-a",
                            "analysis_set": "set-a",
                            "fit_status": "fitted",
                            "propensity_score_column": "ps_a",
                        },
                        {
                            "model_id": "declared-b",
                            "analysis_set": "set-b",
                            "fit_status": "fitted",
                            "propensity_score_column": "ps_b",
                        },
                    ]
                }
            }
        },
    )

    assert "Planner-owned model roster" in guidance
    assert "every fitted roster entry" in guidance
    assert "declared-a" in guidance and "declared-b" in guidance
    assert "do not choose the first row" in guidance


def test_semantic_aliases_include_step_id_and_prediction_aliases(ra, tmp_path: Path):
    from easyicu.research_agent.pipeline import _semantic_aliases_for

    step = ra.AnalysisStep(
        step_id="01_model_training",
        method="prediction_model_evaluation",
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
        method="kmeans_clustering",
        expected_outputs=[
            "table:cluster_characteristics",
            "statistic:silhouette_score",
        ],
    )

    aliases = _semantic_aliases_for(step, tmp_path / "step_summary.json")

    assert "table_one" in aliases
    assert "cluster_summary" in aliases
    assert "cluster_characteristics" in aliases
    assert "cluster_mortality" in aliases


def test_semantic_aliases_bind_kdigo_sensitivity_to_primary_association(
    ra, tmp_path: Path
):
    from easyicu.research_agent.pipeline import _semantic_aliases_for

    step = ra.AnalysisStep(
        step_id="06_sensitivity_reduced_vars",
        intent="Fit reduced-variable KDIGO sensitivity model.",
        expected_outputs=["statistic:adjusted_or_ci"],
    )

    aliases = _semantic_aliases_for(step, tmp_path / "step_summary.json")

    assert "primary_association" in aliases
    assert "kdigo_sensitivity" in aliases


def test_semantic_aliases_do_not_bind_primary_association_for_robustness_without_effect(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.pipeline import _semantic_aliases_for

    summary = tmp_path / "step_summary.json"
    summary.write_text(
        '{"missingness_strategy": null, "note": "no effect estimate"}',
        encoding="utf-8",
    )
    step = ra.AnalysisStep(
        step_id="02b_missingness_handling",
        intent="Finalize robustness missingness strategy.",
        expected_outputs=["log:robustness_missingness"],
    )

    aliases = _semantic_aliases_for(step, summary)

    assert "robustness_summary" in aliases
    assert "primary_association" not in aliases


def test_semantic_aliases_bind_stratified_mortality_table(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.pipeline import _semantic_aliases_for

    step = ra.AnalysisStep(
        step_id="01_stratified_mortality",
        intent="Report explicitly requested stratified mortality.",
        expected_outputs=["table:stratified_mortality"],
    )

    aliases = _semantic_aliases_for(
        step,
        tmp_path / "stratified_mortality_incidence.csv",
    )

    assert "stratified_mortality" in aliases
    assert "outcome_rate" in aliases


def test_semantic_aliases_bind_report_mortality_rate_to_outcome_rate(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.pipeline import _semantic_aliases_for

    summary = tmp_path / "step_summary.json"
    summary.write_text('{"n_rows": 500, "mortality_rate": 0.094}', encoding="utf-8")
    step = ra.AnalysisStep(
        step_id="07_report",
        intent="Write final report tables.",
        expected_outputs=["table:table_one"],
    )

    aliases = _semantic_aliases_for(step, summary)

    assert "outcome_rate" in aliases
    assert "mortality_rate" in aliases
    assert "cohort_summary" in aliases


def test_advanced_plan_contract_preserves_prediction_evaluation_boundary(ra):
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

    assert [step.step_id for step in revised.steps] == [
        "01_train_model",
        "02_evaluate_auroc",
    ]
    assert revised == plan
    assert findings and findings[0].validator == "plan_contract"
    assert findings[0].detail.get("missing_structured_owner") is True


def test_figure_detection_uses_structured_artifacts_and_word_boundaries(ra):
    from easyicu.research_agent.plan_utils import (
        _research_question_implies_figure,
        _step_expects_figure,
    )

    assert not _step_expects_figure(
        ra.AnalysisStep(
            step_id="01_configuration",
            intent="Configure the model.",
            method="configure_model",
            expected_outputs=["table:model_configuration"],
        )
    )
    assert _step_expects_figure(
        ra.AnalysisStep(
            step_id="02_results",
            intent="Render results.",
            method="visualization",
            expected_outputs=["figure:primary_results"],
        )
    )
    assert not _research_question_implies_figure(
        "Configure ordinal exposure categories before fitting the model."
    )
    assert _research_question_implies_figure(
        "Report the result as a figure or a source-backed table."
    )


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
    assert (
        "model_df['sex'] = model_df['sex'].astype(str).str.lower().isin(['m', 'male']).astype(float)"
        in patched
    )
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
            "primary_predictor": "creatinine_max_24h",
            "complete_case_n": None,
            "primary_or": None,
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
    assert (
        "model_df['sex'] = model_df['sex'].astype(str).str.lower().isin(['m', 'male']).astype(float)"
        in patched
    )
    assert 'model_df[col] = pd.to_numeric(model_df[col], errors="coerce")' in patched


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

    code = """
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
"""
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
    assert (
        'X_clean = data.drop(columns=[y.name]).apply(pd.to_numeric, errors="coerce").astype(float)'
        in patched
    )


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


def test_formula_dummy_name_error_is_left_to_agent_repair(ra):
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

    assert repaired is None


def test_deterministic_summary_repair_leaves_singular_ordinal_model_to_agent(
    ra,
):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    code = """
import os
import json
import pandas as pd
import statsmodels.api as sm
import patsy

cohort_path = os.environ["COHORT_PARQUET"]
out_dir = os.environ["STEP_OUT_DIR"]
df = pd.read_parquet(cohort_path)
formula = "death ~ C(sofa2_admission) + age + weight + C(sex)"
y, X = patsy.dmatrices(formula, data=df, return_type="dataframe")
try:
    model = sm.Logit(y, X.astype(float)).fit(disp=False)
except Exception as exc:
    step_summary = {
        "model": {
            "type": "logistic_regression",
            "predictors": list(X.columns),
            "converged": False,
            "error": "Singular matrix",
        },
        "primary_association_estimate": {
            "variable": None,
            "odds_ratio": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": None,
        },
    }
    with open(os.path.join(out_dir, "step_summary.json"), "w", encoding="utf-8") as f:
        json.dump(step_summary, f)
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "model": {
                "predictors": [
                    "Intercept",
                    "C(sofa2_admission)[T.1]",
                    "C(sofa2_admission)[T.2]",
                    "C(sofa2_admission)[T.3]",
                    "C(sofa2_admission)[T.4]",
                ],
                "converged": False,
                "error": "Singular matrix",
            },
            "primary_association_estimate": {
                "variable": None,
                "odds_ratio": None,
                "ci_low": None,
                "ci_high": None,
                "p_value": None,
            },
        },
    )

    assert repaired is None


def test_singular_ordinal_primary_model_does_not_inject_fallback_outputs(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    cohort = pd.DataFrame(
        {
            "sofa2_admission": [0, 1, 2, 3, 4, 5] * 20,
            "death": [0, 0, 1, 0, 1, 1] * 10 + [0, 1, 0, 1, 0, 1] * 10,
            "age": [50 + (i % 23) for i in range(120)],
            "sex": ["F", "M", "F", "M", "F", "M"] * 20,
            "weight": [60 + ((i * 7) % 35) for i in range(120)],
        }
    )
    cohort_path = tmp_path / "cohort.parquet"
    out_dir = tmp_path / "step"
    out_dir.mkdir()
    cohort.to_parquet(cohort_path, index=False)
    monkeypatch.setenv("COHORT_PARQUET", str(cohort_path))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))

    code = """
import os
import json
with open(os.path.join(os.environ["STEP_OUT_DIR"], "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump({
        "model": {
            "predictors": ["Intercept", "C(sofa2_admission)[T.1]", "C(sofa2_admission)[T.2]", "age"],
            "converged": False,
            "error": "Singular matrix"
        },
        "primary_association_estimate": {
            "variable": None,
            "odds_ratio": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": None
        }
    }, f)
"""
    repaired = _deterministic_summary_repair(
        code=code + "\n# formula reference: death ~ C(sofa2_admission) + age",
        step_summary={
            "model": {
                "predictors": [
                    "Intercept",
                    "C(sofa2_admission)[T.1]",
                    "C(sofa2_admission)[T.2]",
                    "C(sofa2_admission)[T.3]",
                ],
                "converged": False,
                "error": "Singular matrix",
            },
            "primary_association_estimate": {
                "variable": None,
                "odds_ratio": None,
                "ci_low": None,
                "ci_high": None,
                "p_value": None,
            },
        },
    )
    assert repaired is None
    assert not (out_dir / "step_summary.json").exists()
    assert not (out_dir / "association_results.csv").exists()


def test_summary_repair_does_not_substitute_glm_for_mle_retvals_failure(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A caught GLM API failure must return to agent repair, not a new model."""
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    cohort = pd.DataFrame(
        {
            "sofa2_admission": [0, 1, 2, 3, 4, 5] * 20,
            "death": [0, 0, 1, 0, 1, 1] * 10 + [0, 1, 0, 1, 0, 1] * 10,
            "age": [50 + (i % 23) for i in range(120)],
            "sex": ["F", "M", "F", "M", "F", "M"] * 20,
            "weight": [60 + ((i * 7) % 35) for i in range(120)],
        }
    )
    cohort_path = tmp_path / "cohort.parquet"
    out_dir = tmp_path / "step"
    out_dir.mkdir()
    cohort.to_parquet(cohort_path, index=False)
    monkeypatch.setenv("COHORT_PARQUET", str(cohort_path))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))

    code = """
import os
import json
import pandas as pd
import statsmodels.api as sm

out_dir = os.environ["STEP_OUT_DIR"]
df = pd.read_parquet(os.environ["COHORT_PARQUET"])
required_cols = ["sofa2_admission", "death"]
df = df[required_cols].dropna().copy()
df["sofa2_admission"] = pd.to_numeric(df["sofa2_admission"], errors="coerce")
df = df.dropna(subset=["sofa2_admission"])
X = sm.add_constant(pd.get_dummies(df["sofa2_admission"], prefix="sofa2").astype(float))
y = df["death"].astype(float)
step_summary = {"skipped": [], "derived_claims": []}
try:
    result = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    converged = result.mle_retvals.get("converged", True)
except Exception as exc:
    step_summary["skipped"].append({"reason": "model_fit_error", "error": str(exc)})
    converged = False
step_summary["primary_odds_ratio"] = None
step_summary["converged"] = converged
step_summary["n_observations"] = int(len(df))
with open(os.path.join(out_dir, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(step_summary, f)
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "skipped": [
                {
                    "reason": "model_fit_error",
                    "error": "'GLMResults' object has no attribute 'mle_retvals'",
                }
            ],
            "primary_odds_ratio": None,
            "converged": False,
            "n_observations": 120,
        },
    )

    assert repaired is None
    assert not (out_dir / "step_summary.json").exists()
    assert not (out_dir / "association_results.csv").exists()


def test_summary_repair_does_not_substitute_logit_for_null_primary_or(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A null Logit result remains an agent-owned repair/fail-closed outcome."""

    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    n = 160
    cohort = pd.DataFrame(
        {
            "sofa2_admission": [i % 8 for i in range(n)],
            "death": [1 if (i % 8) >= 5 or i % 19 == 0 else 0 for i in range(n)],
            "age": [45 + (i % 30) for i in range(n)],
            "sex": ["F", "M", "F", "M"] * (n // 4),
            "bmi": [20.0 + (i % 11) * 0.7 for i in range(n)],
            "weight": [55.0 + (i % 25) for i in range(n)],
        }
    )
    cohort_path = tmp_path / "cohort.parquet"
    out_dir = tmp_path / "step"
    out_dir.mkdir()
    cohort.to_parquet(cohort_path, index=False)
    monkeypatch.setenv("COHORT_PARQUET", str(cohort_path))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))

    code = """
import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

out_dir = os.environ["STEP_OUT_DIR"]
df = pd.read_parquet(os.environ["COHORT_PARQUET"])
required_vars = ["sofa2_admission", "age", "sex", "bmi", "weight", "death"]
cc_mask = df[required_vars].notnull().all(axis=1)
df_cc = df.loc[cc_mask].copy()
y = df_cc["death"].astype(int)
df_cc["sex"] = df_cc["sex"].astype("category")
sex_dummies = pd.get_dummies(df_cc["sex"], drop_first=True, prefix="sex")
X = pd.concat(
    [
        df_cc[["sofa2_admission", "age", "bmi", "weight"]].astype(float),
        sex_dummies,
    ],
    axis=1,
)
X = sm.add_constant(X, has_constant="add")
step_summary = {"skipped": [], "derived_claims": []}
primary_row = {}
try:
    model = sm.Logit(y, X).fit(disp=False)
    coef = model.params["sofa2_admission"]
    se = model.bse["sofa2_admission"]
    primary_row = {
        "or": np.exp(coef),
        "ci_low": np.exp(coef - 1.96 * se),
        "ci_high": np.exp(coef + 1.96 * se),
        "pvalue": model.pvalues["sofa2_admission"],
    }
except Exception as exc:
    primary_row = {
        "error": str(exc),
    }
pd.DataFrame([primary_row]).to_csv(
    os.path.join(out_dir, "logistic_regression_results.csv"),
    index=False,
)
step_summary.update(
    {
        "n_total": int(df.shape[0]),
        "n_complete_case": int(df_cc.shape[0]),
        "outcome": "ICU mortality (death)",
        "primary_predictor": "sofa2_admission (ordinal, per-point)",
        "primary_or": float(primary_row.get("or")) if "or" in primary_row else None,
        "primary_ci_low": float(primary_row.get("ci_low")) if "ci_low" in primary_row else None,
        "primary_ci_high": float(primary_row.get("ci_high")) if "ci_high" in primary_row else None,
        "primary_pvalue": float(primary_row.get("pvalue")) if "pvalue" in primary_row else None,
    }
)
with open(os.path.join(out_dir, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(step_summary, f)
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "n_total": 160,
            "n_complete_case": 160,
            "outcome": "ICU mortality (death)",
            "primary_predictor": "sofa2_admission (ordinal, per-point)",
            "primary_or": None,
            "primary_ci_low": None,
            "primary_ci_high": None,
            "primary_pvalue": None,
        },
    )

    assert repaired is None
    assert not (out_dir / "step_summary.json").exists()
    assert not (out_dir / "logistic_regression_results.csv").exists()


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
    assert (
        "design_cols = [predictor] + [col for col in covariates if col != predictor]"
        in patched
    )
    assert (
        'model_df[design_cols].apply(pd.to_numeric, errors="coerce").astype(float)'
        in patched
    )


def test_deterministic_summary_repair_stabilizes_robustness_missingness_models(ra):
    from easyicu.research_agent.pipeline import _deterministic_summary_repair

    code = """
outcome_col = 'readmission_30d'
primary_predictor = 'creatinine_max_24h'
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
mi_df['creatinine_missing'] = mi_df[primary_predictor].isna().astype(int)
rv_df = model_df.dropna(subset=[primary_predictor])
cc_X = sm.add_constant(cc_df[[primary_predictor] + covariates], has_constant="add")
mi_X = sm.add_constant(mi_df[[primary_predictor, 'creatinine_missing'] + covariates], has_constant="add")
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
    assert (
        "reduced_covariates = [c for c in covariates if model_df[c].isna().mean() <= 0.2]"
        in patched
    )
    assert (
        "cc_df = model_df.dropna(subset=[outcome_col, primary_predictor] + covariates)"
        in patched
    )
    assert "mi_df[primary_predictor] = mi_df[primary_predictor].fillna(0)" in patched
    assert "mi_df = mi_df.dropna(subset=[outcome_col] + covariates)" in patched
    assert (
        "rv_df = model_df[[outcome_col, primary_predictor] + reduced_covariates].dropna()"
        in patched
    )
    assert (
        'rv_X = sm.add_constant(rv_df[[primary_predictor] + reduced_covariates], has_constant="add")'
        in patched
    )


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
        method="adjusted_logistic_regression",
        intent="Fit logistic regression for lactate association.",
        expected_outputs=["statistic:primary_association_estimate"],
    )
    findings = _step_contract_findings(
        step=step,
        step_summary={
            "output_files": {
                "statistic:primary_association_estimate": 1.219,
            },
            "summary": {
                "notes": [
                    "Association estimate with lactate: OR=1.219 (95% CI 1.116-1.332)."
                ]
            },
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

    assert [step.step_id for step in revised.steps] == [
        "01_missingness",
        "03_model_fitting_complete_case",
        "04_robustness_figure",
    ]
    assert revised == plan
    assert findings[0].detail.get("missing_structured_owner") is True


def _survival_plan_with_only_association_steps(AnalysisPlan, AnalysisStep):
    """A locked-survival plan the LLM wrote as a pure association study."""
    return AnalysisPlan(
        research_question=(
            "Estimate the association between mechanical ventilation and 28-day "
            "mortality respecting exposure timing and censoring."
        ),
        analysis_type="survival",
        steps=[
            AnalysisStep(
                step_id="01_cohort_timezero_attrition",
                intent="Define time zero, follow-up, and cohort attrition.",
                method="descriptive",
                expected_outputs=["table:cohort_attrition"],
            ),
            AnalysisStep(
                step_id="03_table_one",
                intent="Baseline characteristics.",
                method="descriptive",
                expected_outputs=["table:table_one"],
            ),
            AnalysisStep(
                step_id="05_primary_landmark_association_model",
                intent="Fit adjusted association model for the landmark outcome.",
                method="association_study",
                expected_outputs=["table:adjusted_association_estimates"],
            ),
            AnalysisStep(
                step_id="05_primary_landmark_association_model_figure",
                intent="Forest plot of adjusted effects.",
                method="association_study",
                expected_outputs=["figure:effect_forest"],
            ),
            AnalysisStep(
                step_id="06_sensitivity_and_diagnostics",
                intent="Sensitivity and model diagnostics.",
                method="association_study",
                expected_outputs=["table:robustness_results"],
            ),
        ],
    )


def test_advanced_plan_contract_does_not_choose_survival_method_for_agent(ra):
    """A family mismatch is surfaced without rewriting the agent's science."""
    from easyicu.research_agent.pipeline import _enforce_advanced_plan_contract
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        CohortDescriptor,
        ResearchContext,
    )

    plan = _survival_plan_with_only_association_steps(AnalysisPlan, AnalysisStep)
    context = ResearchContext(
        research_question=plan.research_question,
        cohort=CohortDescriptor(
            cohort_name="cohort", database="synthetic", n_patients=10, n_stays=10
        ),
        variables=[],
        target_outcome="death",
        primary_exposure="vent",
    )

    revised, findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    assert revised == plan
    assert findings and findings[0].validator == "plan_contract"
    assert findings[0].detail.get("missing_structured_owner") is True
    assert findings[0].detail.get("preserved_step_ids") == [
        step.step_id for step in plan.steps
    ]


def test_advanced_plan_contract_never_converts_primary_model_cohort(ra):
    from easyicu.research_agent.pipeline import _enforce_advanced_plan_contract

    context = ra.ResearchContext(
        research_question="Estimate survival while preserving the planned owners.",
        cohort=ra.CohortDescriptor(
            cohort_name="cohort", database="synthetic", n_patients=50, n_stays=50
        ),
        variables=[],
        primary_exposure="exposure",
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        analysis_type="survival",
        steps=[
            ra.AnalysisStep(
                step_id="01_primary_model_cohort",
                intent=(
                    "Prepare the primary model cohort for survival analysis and "
                    "report attrition."
                ),
                method="cohort_definition",
                expected_outputs=["table:cohort_attrition"],
            ),
            ra.AnalysisStep(
                step_id="05_primary_association",
                intent="Estimate the prespecified adjusted association.",
                method="mixed_effects_regression",
                expected_outputs=["table:association_estimates"],
            ),
        ],
    )

    revised, findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    assert revised == plan
    assert [step.method for step in revised.steps] == [
        "cohort_definition",
        "mixed_effects_regression",
    ]
    assert findings[0].detail.get("missing_structured_owner") is True


def test_advanced_plan_contract_preserves_survival_cohort_owner_boundary(ra):
    from easyicu.research_agent.pipeline import _enforce_advanced_plan_contract

    context = ra.ResearchContext(
        research_question="Estimate time-to-event survival with Cox regression.",
        cohort=ra.CohortDescriptor(
            cohort_name="cohort", database="synthetic", n_patients=20, n_stays=20
        ),
        variables=[],
        primary_exposure="exposure",
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        analysis_type="survival",
        steps=[
            ra.AnalysisStep(
                step_id="01_survival_cohort",
                intent="Define the survival cohort and report attrition.",
                method="cohort_definition",
                expected_outputs=["table:cohort_attrition"],
            ),
            ra.AnalysisStep(
                step_id="05_primary_cox",
                intent="Fit the prespecified Cox proportional-hazards model.",
                method="cox_proportional_hazards",
                expected_outputs=["table:hr"],
            ),
        ],
    )

    revised, _findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    assert [step.step_id for step in revised.steps] == [
        "01_survival_cohort",
        "05_primary_cox",
    ]
    assert revised.steps[0].method == "cohort_definition"
    assert revised.steps[0].expected_outputs == ["table:cohort_attrition"]
    assert revised.steps[1].method == "cox_proportional_hazards"
    assert "table:cox_summary" in revised.steps[1].expected_outputs


def test_advanced_plan_contract_preserves_explicit_kmeans_method(ra):
    from easyicu.research_agent.pipeline import _enforce_advanced_plan_contract

    context = ra.ResearchContext(
        research_question="Discover longitudinal phenotypes with KMeans clustering.",
        cohort=ra.CohortDescriptor(
            cohort_name="cohort", database="synthetic", n_patients=20, n_stays=20
        ),
        variables=[],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        analysis_type="trajectory_clustering",
        steps=[
            ra.AnalysisStep(
                step_id="05_kmeans_phenotyping",
                intent="Discover trajectory phenotypes with KMeans.",
                method="kmeans_clustering",
                expected_outputs=[
                    "table:cluster_assignments",
                    "table:cluster_characteristics",
                ],
            )
        ],
    )

    revised, _findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    assert [step.step_id for step in revised.steps] == ["05_kmeans_phenotyping"]
    assert revised.steps[0].method == "kmeans_clustering"
    assert "manifest:cluster_selection" in revised.steps[0].expected_outputs
    assert "statistic:silhouette_score" not in revised.steps[0].expected_outputs


def test_clustering_contract_does_not_invent_mortality_characterization(ra):
    from easyicu.research_agent.pipeline import _enforce_advanced_plan_contract

    context = ra.ResearchContext(
        research_question="Discover longitudinal phenotypes without outcome analysis.",
        cohort=ra.CohortDescriptor(
            cohort_name="cohort", database="synthetic", n_patients=20, n_stays=20
        ),
        variables=[],
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        analysis_type="trajectory_clustering",
        steps=[
            ra.AnalysisStep(
                step_id="05_gmm_phenotyping",
                intent="Discover trajectory phenotypes with a Gaussian mixture model.",
                method="gaussian_mixture_model",
                expected_outputs=[
                    "table:cluster_assignments",
                    "statistic:cluster_count",
                ],
            )
        ],
    )

    revised, _findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    outputs = revised.steps[0].expected_outputs
    assert "table:cluster_mortality" not in outputs
    assert "table:outcome_by_cluster" not in outputs


def test_advanced_plan_contract_leaves_pure_association_family_alone(ra):
    """A genuine association study is NOT force-converted (upgrade-only guard)."""
    from easyicu.research_agent.pipeline import _enforce_advanced_plan_contract
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        CohortDescriptor,
        ResearchContext,
        UserPreferences,
    )

    plan = AnalysisPlan(
        research_question="Is Sepsis-3 associated with mortality after adjustment?",
        steps=[
            AnalysisStep(
                step_id="01_cohort",
                intent="Cohort attrition.",
                method="descriptive",
                expected_outputs=["table:cohort_attrition"],
            ),
            AnalysisStep(
                step_id="02_primary_association_model",
                intent="Adjusted logistic association.",
                method="association_study",
                expected_outputs=["table:adjusted_association_estimates"],
            ),
        ],
    )
    context = ResearchContext(
        research_question=plan.research_question,
        cohort=CohortDescriptor(
            cohort_name="cohort", database="synthetic", n_patients=10, n_stays=10
        ),
        variables=[],
        target_outcome="death",
        primary_exposure="sepsis3",
        user_preferences=UserPreferences(inferred_analysis_family="association_study"),
    )

    revised, findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    step_ids = [s.step_id for s in revised.steps]
    assert "01_survival_analysis" not in step_ids
    assert step_ids == ["01_cohort", "02_primary_association_model"]
    assert not any(
        f.detail and f.detail.get("converted_from_association") for f in findings
    )


def test_advanced_plan_contract_does_not_rewrite_cluster_robust_association(ra):
    from easyicu.research_agent.pipeline import _enforce_advanced_plan_contract

    context = ra.ResearchContext(
        research_question=(
            "Estimate the mortality association using mixed effects with "
            "cluster-robust SE and hospital-level clustering."
        ),
        cohort=ra.CohortDescriptor(
            cohort_name="cohort", database="synthetic", n_patients=50, n_stays=50
        ),
        variables=[],
        primary_exposure="exposure",
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        analysis_type="association_study",
        steps=[
            ra.AnalysisStep(
                step_id="05_primary_association",
                intent=context.research_question,
                method="mixed_effects_regression",
                expected_outputs=["table:association_estimates"],
            )
        ],
    )

    revised, findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    assert [step.step_id for step in revised.steps] == ["05_primary_association"]
    assert revised.steps[0].method == "mixed_effects_regression"
    assert not any(item.detail.get("family") == "clustering" for item in findings)


def test_terminal_publication_repair_replan_skip_requires_satisfied_bundle(
    ra, tmp_path: Path
):
    from easyicu.research_agent.execution.phase import (
        _terminal_publication_repair_replan_skip_detail,
    )

    completed_step_id = "03_primary_results_display_contract_audit"
    repair_step_id = "03_primary_results_publication_figure_repair"
    outputs_dir = tmp_path / "steps" / completed_step_id / "outputs"
    outputs_dir.mkdir(parents=True)
    (outputs_dir / "publication_figure.png").write_bytes(b"png")
    (outputs_dir / "publication_figure_source_data.csv").write_text(
        "term,or,ci_low,ci_high\nsepsis3,1.2,1.1,1.3\n",
        encoding="utf-8",
    )
    (outputs_dir / "publication_figure.figure_contract.json").write_text(
        json.dumps(
            {
                "figure_id": "publication_figure",
                "panels": [
                    {"panel_id": "A", "role": "descriptive_result"},
                    {"panel_id": "B", "role": "primary_estimand"},
                ],
                "export_formats": ["png"],
                "source_data": ["publication_figure_source_data.csv"],
            }
        ),
        encoding="utf-8",
    )

    plan = ra.AnalysisPlan(
        research_question="Estimate prevalence and adjusted association.",
        steps=[
            ra.AnalysisStep(
                step_id=completed_step_id,
                intent="Audit and repair the primary results display contract.",
                expected_outputs=["figure:publication_figure"],
            ),
            ra.AnalysisStep(
                step_id=repair_step_id,
                intent=(
                    "Rendering-only repair from upstream results; produce the "
                    "publication figure without re-analysing the cohort."
                ),
                method="rendering_only_repair_from_primary_results",
                expected_outputs=["figure:adjusted_effect_forest_repair"],
            ),
        ],
    )

    detail = _terminal_publication_repair_replan_skip_detail(
        plan=plan,
        completed_records=[{"step_id": completed_step_id, "status": "ok"}],
        run_dir=tmp_path,
    )

    assert detail is not None
    assert detail["remaining_step_ids"] == [repair_step_id]
    assert detail["satisfied_by_step_id"] == completed_step_id

    plan_with_analysis_remaining = ra.AnalysisPlan(
        research_question=plan.research_question,
        steps=[
            *plan.steps,
            ra.AnalysisStep(
                step_id="04_new_model",
                intent="Fit an additional model.",
                expected_outputs=["table:model"],
            ),
        ],
    )
    assert (
        _terminal_publication_repair_replan_skip_detail(
            plan=plan_with_analysis_remaining,
            completed_records=[{"step_id": completed_step_id, "status": "ok"}],
            run_dir=tmp_path,
        )
        is None
    )


def test_advanced_plan_contract_preserves_article_level_robustness_suite(ra):
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
            "Estimate Sepsis-3 prevalence and adjusted mortality association "
            "with visible attrition, missingness, and robustness."
        ),
        steps=[
            AnalysisStep(
                step_id="01_primary_cohort_and_exposure_definition",
                intent="Define cohort eligibility, attrition, and Sepsis-3 exposure.",
                expected_outputs=["table:cohort_attrition", "derived_variable:sepsis3"],
            ),
            AnalysisStep(
                step_id="02_table_one_and_missingness",
                intent="Render Table 1 baseline characteristics and missingness audit.",
                expected_outputs=[
                    "table:table_one",
                    "table:missingness_measurement_audit",
                ],
            ),
            AnalysisStep(
                step_id="03_primary_adjusted_association",
                intent="Fit adjusted association model and report odds ratio.",
                expected_outputs=["table:adjusted_association_primary"],
            ),
            AnalysisStep(
                step_id="04_robustness_grid",
                intent="Run complete-case and alternative-definition sensitivity analyses.",
                expected_outputs=["figure:robustness_grid"],
            ),
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
        primary_exposure="sepsis3",
        user_preferences=UserPreferences(inferred_analysis_family="robustness"),
    )

    revised, findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    assert [step.step_id for step in revised.steps] == [
        "01_primary_cohort_and_exposure_definition",
        "02_table_one_and_missingness",
        "03_primary_adjusted_association",
        "04_robustness_grid",
    ]
    robustness_step = revised.steps[-1]
    assert robustness_step.expected_outputs == ["figure:robustness_grid"]
    assert revised == plan
    assert findings[0].detail.get("missing_structured_owner") is True


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

    assert [step.step_id for step in revised.steps] == [
        "01_cohort_summary",
        "02_outcome_incidence",
        "03_missingness_audit",
    ]
    assert revised == plan
    assert findings[0].detail.get("missing_structured_owner") is True


def test_advanced_plan_contract_does_not_rewrite_component_data_quality_audit(ra):
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
            "Audit whether composite-score rows have enough measured components "
            "before any outcome model is fit."
        ),
        steps=[
            AnalysisStep(
                step_id="01_component_completeness_qc",
                intent="Check composite-score component completeness.",
                expected_outputs=[
                    "statistic:low_completeness_count",
                    "table:component_completeness",
                ],
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

    assert [step.step_id for step in revised.steps] == ["01_component_completeness_qc"]
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
                expected_outputs=[
                    "table:robustness_summary",
                    "figure:robustness_figure",
                ],
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

    assert [step.step_id for step in revised.steps] == ["01_robustness_analysis"]
    assert revised.steps[0].expected_outputs == [
        "table:robustness_summary",
        "figure:robustness_figure",
    ]
    assert findings == []


def test_salvage_stdout_json_step_summary(ra, tmp_path: Path):
    from easyicu.research_agent.pipeline import _salvage_stdout_json_step_summary
    from easyicu.research_agent.contracts.runtime import RunResult

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
    from easyicu.research_agent.contracts.runtime import RunResult

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    summary_json = out_dir / "component_completeness_qc_summary.json"
    summary_json.write_text(
        json.dumps({"low_completeness_count": 41, "guardrail_warning": True}),
        encoding="utf-8",
    )
    result = RunResult(
        step_id="01_component_completeness_qc",
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
    assert saved["low_completeness_count"] == 41
    assert saved["guardrail_warning"] is True


def test_pipeline_run_from_spec_writes_runtime_artifacts(ra, tmp_path: Path):
    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "age": [60.0, 72.0, 55.0, 80.0],
            "sofa2": [0, 1, 5, 8],
            "death": [0, 0, 1, 1],
        }
    )
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
