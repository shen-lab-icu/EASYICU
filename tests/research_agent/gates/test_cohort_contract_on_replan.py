"""Execute-phase re-check of the structured-纳排 contract.

Regression for the E1 deepseek-v4-flash run12 finding
-----------------------------------------------------
The plan-phase contract (``pipeline._run_plan_phase`` →
``_cohort_definition_contract_findings``) only inspects the *initial* plan.
For non-deterministic providers that initial plan is commonly a 0-step
shell, and the real executing plan — which always carried a
``01_cohort_definition`` step but left ``plan.cohort`` structurally empty —
is grown by the **replanner** during execution. So the contract was
bypassed: every downstream step silently ran on the unfiltered universe
while each generated step re-applied 纳排 inconsistently (run12 ran the
primary regression by re-filtering the universe in its own code instead of
on a framework-enforced cohort).

``run_execute_phase`` now re-checks the contract against the plan that
actually executes (at execute-start and after every substantive replan) and
surfaces an auditable ``cohort_contract`` error — without aborting the run,
so a defensible manuscript is still produced.
"""

from __future__ import annotations

import json
from pathlib import Path


def test_replanner_grown_cohort_step_with_empty_definition_is_flagged(
    ra, synthetic_cohort, tmp_path: Path, monkeypatch
):
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient

    analysis_step = AnalysisStep(
        step_id="02_missingness",
        planned_analysis_role="auxiliary",
        intent="Summarize cohort missingness.",
        inputs=["age", "death"],
        expected_outputs=["table:missingness"],
        method="missingness_audit",
    )
    initial = AnalysisPlan(
        research_question="Is admission SOFA-2 associated with ICU mortality?",
        steps=[analysis_step],
    )
    revised = initial.model_copy(
        update={
            "steps": [
                AnalysisStep(
                    step_id="01_cohort_definition",
                    intent=(
                        "Define the adult ICU analysis cohort: age >= 18 "
                        "and ICU LoS >= 1 day; report the attrition."
                    ),
                    inputs=[],
                    expected_outputs=["table:analysis_cohort"],
                    method="cohort_definition",
                ),
                analysis_step,
            ],
            "revision": initial.revision + 1,
        }
    )
    llm = PatternScriptedMockLLMClient(
        [
            (
                "ICU-AWARE RESEARCH PLAN",
                [initial.model_dump_json(indent=2)] * 4,
            ),
            ("PROBE SUMMARY:", [revised.model_dump_json(indent=2)] * 4),
        ],
        contextual_default=True,
    )
    from easyicu.research_agent.agents.core import PlannerAgent

    original_planner_run = PlannerAgent.run

    def run_without_article_suite(self, context, **kwargs):
        kwargs["enforce_article_contract"] = False
        return original_planner_run(self, context, **kwargs)

    monkeypatch.setattr(PlannerAgent, "run", run_without_article_suite)
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=llm)
    result = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="cohort_contract_replan",
        database="synthetic",
        target_outcome="death",
    )

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    findings = manifest["findings"]

    contract_errors = [
        f
        for f in findings
        if f.get("validator") == "cohort_contract" and f.get("severity") == "error"
    ]
    assert contract_errors, (
        "replanner grew a cohort-definition step with an empty structured "
        "cohort, but no cohort_contract error was surfaced — the execute-phase "
        "re-check did not fire"
    )
    # The re-check tags the stage so an auditor can tell it apart from the
    # plan-phase contract.
    assert any(
        (f.get("detail") or {}).get("stage") == "execute" for f in contract_errors
    ), "cohort_contract error is not tagged as an execute-phase re-check"

    # The newer typed-product gate is deliberately fail-closed: an empty cohort
    # definition cannot legitimately realise table:analysis_cohort. Downstream
    # diagnostic steps may still run, but the missing product must remain red.
    records = manifest.get("per_step_records", [])
    cohort_records = [r for r in records if r.get("step_id") == "01_cohort_definition"]
    assert cohort_records and cohort_records[-1].get("status") == "contract_failed"
    assert any(r.get("status") == "ok" for r in records if r not in cohort_records)
