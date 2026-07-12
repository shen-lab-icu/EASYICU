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
import re
from pathlib import Path


def _replan_prompt(user: str) -> bool:
    upper = user.upper()
    return "PROBE SUMMARY:" in upper and "CURRENT PLAN:" in upper


def test_replanner_grown_cohort_step_with_empty_definition_is_flagged(
    ra, synthetic_cohort, tmp_path: Path
):
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    class ReplanAddsCohortStepLLM(ra.MockLLMClient):
        """Initial plan carries no cohort-definition step (plan-phase contract
        passes); the replanner grows a ``01_cohort_definition`` step but leaves
        ``plan.cohort`` empty — the run12 shape. Everything else delegates to
        the deterministic mock so the run still completes."""

        def complete(self, messages, **kwargs):
            user = next(
                (m.content for m in reversed(messages) if m.role == "user"), ""
            )
            if _replan_prompt(user):
                match = re.search(
                    r"CURRENT PLAN:\n(\{.*?\})\n\nPROBE SUMMARY:",
                    user,
                    flags=re.DOTALL,
                )
                current = (
                    AnalysisPlan.model_validate_json(match.group(1))
                    if match
                    else AnalysisPlan(research_question="q", steps=[])
                )
                steps = list(current.steps)
                if not any("cohort_def" in (s.step_id or "") for s in steps):
                    steps.insert(
                        0,
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
                    )
                revised = current.model_copy(
                    update={"steps": steps, "revision": current.revision + 1}
                )
                # plan.cohort stays empty: the 纳排 lives only in step prose.
                return revised.model_dump_json(indent=2)
            return super().complete(messages, **kwargs)

    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path, llm=ReplanAddsCohortStepLLM()
    )
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

    # Fail-open, not fail-closed: the contract is auditable but the run still
    # finishes and produces a manuscript.
    statuses = [r.get("status") for r in manifest.get("per_step_records", [])]
    assert statuses and all(s == "ok" for s in statuses), statuses
