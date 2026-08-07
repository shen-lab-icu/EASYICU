"""A step whose declared model is incomplete is refused, not handed to the Coder.

``execution/phase.py`` said it in its own comment: "A step no deterministic
owner claims falls to the stochastic Coder silently."  For a step the host has
an owner for, and where the owner is only waiting on a field the Planner left
null, that silence is a fail-open at a declaration boundary -- it produces the
paper's primary result from a model nobody declared, by the actor whose
accumulated repair guidance records it going wrong (``sex`` numeric-coerced
before dummy encoding, object-dtype design matrices, contracts "satisfied" with
a null estimate).

The plan-time gate asks for the field first and spends a forced replan on the
answer.  Reaching execution still under-declared means that replan did not
supply it.  Owner decision 2026-07-30: refuse the step.

Measured over the recorded plans, 694 distinct step shapes:

    no owner + a declaration gap      8   <- the branch under test
    an owner claimed + a gap          0   <- the false block it must not do

The second row is why ``test_a_claimed_step_is_never_blocked_by_another_owners_gap``
builds its case by hand and says so: the semantics are locked, but reality has
not produced one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.execution.owner_declaration import (
    execution_declaration_refusal,
)
from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient
from easyicu.research_agent.schema import AnalysisStep, PlannedModelRequirement

_SUMMARY_CODE = """
import json
import os
import pandas as pd

df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
summary = {
    "n": int(len(df)),
    "output_files": {"table:cohort_summary": "cohort_summary.csv"},
}
pd.DataFrame([summary]).to_csv(os.path.join(out, "cohort_summary.csv"), index=False)
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
"""


def _under_declared_association_step() -> dict:
    """The real shape: "use an adjusted model", and no adjustment set.

    ``covariates`` is omitted rather than set to ``[]`` -- an empty list is a
    declaration of an unadjusted model, which is a different statement and one
    the owner would happily execute.
    """

    return {
        "step_id": "02_primary_adjusted_association",
        "planned_analysis_role": "primary",
        "intent": "Estimate the adjusted association between value and death.",
        "inputs": ["value", "age", "sex", "death"],
        "expected_outputs": ["table:adjusted_association_estimates"],
        "method": "adjusted_association_models",
        "icu_rule_refs": [],
        "model_requirements": [
            {
                "requirement_id": "primary_value_death_logistic",
                "outcome": "death",
                "outcome_type": "binary",
                "method_family": "binary_logistic_regression",
                "exposure_source": "value",
                "analysis_role": "primary",
                "analysis_set": "complete_case",
            }
        ],
    }


_PLAN_RESPONSE = json.dumps(
    {
        "research_question": "Is value associated with death?",
        "steps": [
            {
                "step_id": "01_summary",
                "planned_analysis_role": "auxiliary",
                "intent": "Produce a descriptive cohort summary.",
                "inputs": ["stay_id", "value"],
                "expected_outputs": ["table:cohort_summary"],
                "method": "descriptive_summary",
                "icu_rule_refs": [],
            },
            _under_declared_association_step(),
        ],
        "rationale": "under-declared primary model regression",
    }
)


def _scripted_llm() -> PatternScriptedMockLLMClient:
    """The Planner does not fix it when asked -- which is the case under test.

    Replying to the replan with the same plan is not a strawman: it is what 74
    of 81 recorded declarations of this product actually look like after the
    plan-time gate has had its turn.
    """

    return PatternScriptedMockLLMClient(
        [
            ("Produce an ICU-AWARE RESEARCH PLAN as JSON", [_PLAN_RESPONSE]),
            ("Replanning contract", [_PLAN_RESPONSE, _PLAN_RESPONSE, _PLAN_RESPONSE]),
            ("WRITE THE PYTHON CODE", [_SUMMARY_CODE, _SUMMARY_CODE]),
            ("INTERPRET THE RESULTS", ["Summary {evidence:cohort_summary}."]),
            (
                "MANUSCRIPT SCAFFOLD",
                ["# Title\n\n## Results\n\nSummary {evidence:cohort_summary}."],
            ),
        ]
    )


def _coder_calls_for(llm: PatternScriptedMockLLMClient, step_id: str) -> int:
    """How many times the Coder was asked to write code mentioning this step."""

    total = 0
    for messages, _kwargs in llm.calls:
        text = "\n".join(str(message.content or "") for message in messages)
        if "WRITE THE PYTHON CODE" in text.upper() and step_id in text:
            total += 1
    return total


@pytest.fixture()
def result(ra, tmp_path: Path, monkeypatch):
    from easyicu.research_agent.agents.core import PlannerAgent

    original_run = PlannerAgent.run

    def run_without_article_contract(self, context, **kwargs):
        kwargs["enforce_article_contract"] = False
        return original_run(self, context, **kwargs)

    monkeypatch.setattr(PlannerAgent, "run", run_without_article_contract)
    llm = _scripted_llm()
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
        max_code_repair_attempts=1,
    )
    outcome = pipeline.run(
        question="Is value associated with death?",
        cohort=pd.DataFrame(
            {
                "stay_id": [1, 2, 3, 4, 5, 6],
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "age": [60, 70, 55, 80, 65, 75],
                "sex": ["M", "F", "M", "F", "M", "F"],
                "death": [0, 1, 0, 1, 0, 1],
            }
        ),
        cohort_name="under_declared_test",
        database="synthetic",
        target_outcome="death",
        stop_after_analysis=True,
    )
    return outcome, llm


def _record(result_tuple, step_id: str) -> dict:
    outcome, _llm = result_tuple
    partial = json.loads(
        (Path(outcome.workdir) / "manifest_partial.json").read_text(encoding="utf-8")
    )
    return next(
        item for item in partial["per_step_records"] if item["step_id"] == step_id
    )


# ---------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------


def test_the_under_declared_step_is_blocked(result):
    record = _record(result, "02_primary_adjusted_association")
    assert record["status"] == "blocked_owner_declaration_incomplete"


def test_the_coder_is_never_asked_to_write_the_blocked_step(result):
    """The load-bearing assertion.

    A status string can be set anywhere.  What the change is for is that the
    stochastic actor never gets the step -- so this counts the actual prompts.
    """

    _outcome, llm = result
    assert _coder_calls_for(llm, "02_primary_adjusted_association") == 0
    # ...and the check is not vacuous: a step with no gap still reaches it.
    assert _coder_calls_for(llm, "01_summary") >= 1


def test_the_refusal_names_the_field_that_was_left_undeclared(result):
    record = _record(result, "02_primary_adjusted_association")
    missing = record["owner_declaration_missing"]
    assert missing == {
        "adjusted_association_estimates": ["model_requirements[0].covariates"]
    }


def test_blocking_one_step_does_not_kill_the_others(result):
    """Per step, not per run.

    The sibling plan-DAG blocks set ``steps_to_run = []``; a step the host
    merely cannot claim must not take the descriptive steps down with it.
    """

    assert _record(result, "01_summary")["status"] == "ok"


# ---------------------------------------------------------------------------
# Negative controls: what must NOT be blocked
# ---------------------------------------------------------------------------


def _trace_for(step: AnalysisStep):
    class _Plan:
        steps = (step,)

    trace: list = []
    chosen = select_standard_executor(step, plan=_Plan(), trace=trace)
    return chosen, tuple(c for c in trace if c.missing_declarations)


def test_a_fully_declared_step_reports_no_gap_and_is_claimed():
    """The positive control for the branch's own input.

    If a completely declared step still reported a gap, the refusal would fire
    on work the host can do -- so this asserts the trace, not just the outcome.
    """

    step = AnalysisStep(
        step_id="02_primary_adjusted_association",
        planned_analysis_role="primary",
        intent="Estimate the adjusted association.",
        inputs=["cohort:analysis_set", "value", "age", "sex", "death"],
        expected_outputs=["table:adjusted_association_estimates"],
        method="adjusted_association_models",
        model_requirements=[
            PlannedModelRequirement(
                requirement_id="primary_value_death_logistic",
                outcome="death",
                outcome_type="binary",
                method_family="binary_logistic_regression",
                exposure_source="value",
                analysis_role="primary",
                analysis_set="complete_case",
                covariates=["age", "sex"],
            )
        ],
    )
    chosen, gaps = _trace_for(step)
    assert gaps == (), gaps
    assert chosen is not None


def test_a_step_no_owner_supports_reports_no_gap():
    """A wrong-shape decline is not a declaration gap.

    No owner exists for this analysis, so the Coder path is correct and
    refusing it would stop work nothing else can do.  ``incomplete_declaration``
    and ``wrong_shape`` being different verdicts is what keeps those apart.
    """

    step = AnalysisStep(
        step_id="03_unsupported",
        planned_analysis_role="primary",
        intent="Fit a mixed-effects model the host does not implement.",
        inputs=["value", "death"],
        expected_outputs=["table:mixed_effects_estimates"],
        method="mixed_effects_models",
        icu_rule_refs=[],
    )
    _chosen, gaps = _trace_for(step)
    assert gaps == ()


class _Candidate:
    """The one field ``execution_declaration_refusal`` reads off a trace entry."""

    def __init__(self, analysis_kind: str, missing: tuple[str, ...]):
        self.analysis_kind = analysis_kind
        self.missing_declarations = missing


def test_a_claimed_step_is_never_refused_over_another_owners_gap():
    """CONSTRUCTED, not observed: 0 of 694 recorded step shapes look like this.

    Kept because the failure it prevents is a wrong block of a step that was
    about to be computed correctly, and nothing in the selector forbids two
    owners answering that way.  The first draft of this test asserted
    ``not (executor is None and gaps)`` against local variables -- a tautology
    that would have passed with the production code deleted.  Calling the real
    function is the difference between locking the rule and describing it.
    """

    gap = _Candidate("some_other_owner", ("model_requirements[0].covariates",))
    assert execution_declaration_refusal(claimed_by=object(), trace=[gap]) == ()


def test_an_unclaimed_step_with_a_gap_is_refused():
    gap = _Candidate("adjusted_association_estimates", ("model_requirements",))
    assert execution_declaration_refusal(claimed_by=None, trace=[gap]) == (gap,)


def test_an_unclaimed_step_without_a_gap_is_not_refused():
    """No owner and no gap: nothing is waiting on a field, the Coder is right."""

    wrong_shape = _Candidate("adjusted_association_estimates", ())
    assert execution_declaration_refusal(claimed_by=None, trace=[wrong_shape]) == ()


def test_every_owner_still_waiting_is_named_not_just_the_first():
    """The refusal message lists them, so dropping one hides a required field."""

    first = _Candidate("owner_a", ("spec.alpha",))
    second = _Candidate("owner_b", ("spec.beta",))
    assert execution_declaration_refusal(
        claimed_by=None, trace=[first, _Candidate("owner_c", ()), second]
    ) == (first, second)
