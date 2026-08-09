"""A timeout must end the step, not start a repair loop.

``classify_runtime_failure`` returning the right object proves only that the
classifier is right. What costs a run is the execution phase: whether it acts on
the classification, or hands the Coder a log with no traceback and asks it to
fix code that was correct but slow. That question can only be answered by
running a step whose runner times out, so these tests do that.

The negative control matters as much as the timeout case. A step that genuinely
crashes must still buy its repairs; a change that terminated every failure would
pass every assertion below except that one.

Every assertion here was checked against a mutant that restores the pre-fix
behaviour (a timeout returning ``None`` from the classifier). One earlier
candidate — "the killed script is not executed a second time" — survived that
mutant, because the repair loop is stopped by something else before it re-runs
anything. It was removed rather than kept as decoration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.provider_budget import (
    provider_call_budget_receipt_path,
)
from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient

TIMEOUT_SECONDS = 321.0

_SCRIPT = """
import json
import os
import pandas as pd

out = os.environ["STEP_OUT_DIR"]
summary = {"n": 3, "output_files": {"table:cohort_summary": "cohort_summary.csv"}}
pd.DataFrame([summary]).to_csv(os.path.join(out, "cohort_summary.csv"), index=False)
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle)
"""


def _plan() -> str:
    return json.dumps(
        {
            "research_question": "Summarize the ICU cohort.",
            "steps": [
                {
                    "step_id": "01_summary",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Produce a descriptive cohort summary.",
                    "inputs": ["stay_id"],
                    "expected_outputs": ["table:cohort_summary"],
                    "method": "descriptive_summary",
                    "icu_rule_refs": [],
                }
            ],
            "rationale": "execution-phase timeout regression",
        }
    )


def _prompt_text(messages) -> str:
    return "\n".join(str(message.content or "") for message in messages)


def _call_count(client, marker: str) -> int:
    folded = marker.casefold()
    return sum(
        folded in _prompt_text(messages).casefold() for messages, _ in client.calls
    )


def _isolate_article_suite_contract(monkeypatch) -> None:
    from easyicu.research_agent.agents.core import PlannerAgent

    original_run = PlannerAgent.run

    def run_without_article_suite(self, context, **kwargs):
        kwargs["enforce_article_contract"] = False
        return original_run(self, context, **kwargs)

    monkeypatch.setattr(PlannerAgent, "run", run_without_article_suite)


def _run_with_failing_runner(*, ra, tmp_path: Path, monkeypatch, timed_out: bool):
    """Execute one step whose runner always fails, timing out or crashing."""

    _isolate_article_suite_contract(monkeypatch)
    from easyicu.research_agent.contracts.runtime import RunResult

    runner_calls: list[str] = []

    class FailingRunner:
        network_policy = "none"
        authority_identity_sha256 = "1" * 64

        def __init__(self, *, workdir: Path, timeout_seconds: float) -> None:
            self.workdir = Path(workdir)
            self.timeout_seconds = timeout_seconds

        @staticmethod
        def validate_runtime_capabilities() -> tuple[str, ...]:
            return ("pandas",)

        def run(self, *, step_id, code, resolved_inputs_path=None):
            del resolved_inputs_path
            runner_calls.append(step_id)
            step_dir = self.workdir / "steps" / step_id
            out_dir = step_dir / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            script_path = step_dir / "analysis.py"
            script_path.write_text(code, encoding="utf-8")
            log_path = step_dir / "run.log"
            # A killed script leaves whatever prefix it had written. A crashed
            # one leaves a traceback. The classifier must read the kill, not
            # the text.
            log_path.write_text(
                (
                    "loading cohort...\n"
                    if timed_out
                    else "Traceback (most recent call last):\n"
                    "NameError: name 'model_frame' is not defined\n"
                ),
                encoding="utf-8",
            )
            return RunResult(
                step_id=step_id,
                script_path=script_path,
                cwd=step_dir,
                out_dir=out_dir,
                stdout="",
                stderr="synthetic failure",
                returncode=-9 if timed_out else 1,
                duration_seconds=self.timeout_seconds if timed_out else 0.02,
                artefacts=[],
                timed_out=timed_out,
                effective_isolation="controlled_test",
                runner_log_path=log_path,
            )

    def runner_factory(*, workdir, timeout_seconds=TIMEOUT_SECONDS, **_kwargs):
        return FailingRunner(workdir=Path(workdir), timeout_seconds=timeout_seconds)

    llm = PatternScriptedMockLLMClient(
        [
            ("ICU-AWARE RESEARCH PLAN", [_plan()] * 4),
            ("WRITE THE PYTHON CODE", [f"```python\n{_SCRIPT}```"] * 8),
            # Distinguishable from the original: an identical candidate is
            # short-circuited, which would make "the runner ran once" true for
            # a reason that has nothing to do with the timeout.
            ("REPAIR THE PYTHON CODE", [f"```python\n# repaired\n{_SCRIPT}```"] * 8),
            (
                "CONSERVATIVE ICU CONCEPT-USE AUDITOR",
                [json.dumps({"findings": []})] * 8,
            ),
            ("INTERPRET THE RESULTS", ["The cohort summary is available."] * 8),
        ]
    )
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        runner_factory=runner_factory,
        timeout_seconds=TIMEOUT_SECONDS,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
        enable_probe_step=False,
        enable_replanning=False,
    )
    result = pipeline.run(
        question="Summarize the ICU cohort.",
        cohort=pd.DataFrame({"stay_id": [1, 2, 3], "death": [0, 1, 0]}),
        cohort_name="execution_timeout_test",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )
    run_dir = Path(result.workdir)
    partial = json.loads((run_dir / "manifest_partial.json").read_text("utf-8"))
    records = [
        record
        for record in partial["per_step_records"]
        if record.get("step_id") == "01_summary"
    ]
    return llm, runner_calls, run_dir, partial, records


@pytest.fixture(scope="module")
def timeout_run(ra, tmp_path_factory, request):
    """One real run whose only step is killed by the wall clock."""

    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)
    return _run_with_failing_runner(
        ra=ra,
        tmp_path=tmp_path_factory.mktemp("timeout"),
        monkeypatch=monkeypatch,
        timed_out=True,
    )


def test_a_timed_out_step_never_reaches_the_coder(timeout_run) -> None:
    """The repairer would be reading a log with no traceback in it."""

    llm, _runner_calls, _run_dir, _partial, _records = timeout_run
    assert _call_count(llm, "REPAIR THE PYTHON CODE") == 0


def test_the_step_ends_terminally_with_the_clock_that_ended_it(timeout_run) -> None:
    _llm, _runner_calls, _run_dir, _partial, records = timeout_run
    assert records, "the run must record the step it killed"
    record = records[-1]
    assert record["runtime_failure_class"] == "execution_timeout"
    assert record["runtime_repair_route"] == "fail_closed"
    assert record["llm_repair_used"] is False
    assert record["timed_out"] is True
    assert record["execution_timeout_seconds"] == TIMEOUT_SECONDS
    assert record["diagnostic_only"] is True


def test_the_timeout_costs_no_repair_budget(timeout_run) -> None:
    """A step that failed for want of time must not also have been charged."""

    _llm, _runner_calls, run_dir, _partial, _records = timeout_run
    receipt_path = provider_call_budget_receipt_path(run_dir, step_id="01_summary")
    assert receipt_path.exists(), "the step must have written a provider receipt"
    receipt = json.loads(receipt_path.read_text("utf-8"))
    categories = [str(item) for item in (receipt.get("categories") or [])]
    # Not `.get("category_history")`: that key does not exist, so reading it
    # returns [] and the assertion below holds for every possible run.
    assert categories, "the receipt must record what the step actually spent"
    assert not [name for name in categories if "repair" in name], categories


def test_the_run_reports_which_clock_it_hit(timeout_run) -> None:
    _llm, _runner_calls, _run_dir, partial, _records = timeout_run
    timeouts = [
        finding
        for finding in partial.get("findings") or []
        if finding.get("validator") == "runtime_execution_timeout"
    ]
    assert timeouts, "an operator choosing a bigger budget needs the limit"
    assert timeouts[0]["detail"]["timeout_seconds"] == TIMEOUT_SECONDS


def test_a_crashed_step_still_buys_its_repairs(ra, tmp_path: Path, monkeypatch) -> None:
    """The control that keeps the fix from being 'terminate everything'.

    A NameError is exactly what the repair loop exists for. If this passes
    only because nothing repairs any more, the change is not narrower — it is
    just broken in the other direction.
    """

    llm, _runner_calls, _run_dir, _partial, records = _run_with_failing_runner(
        ra=ra, tmp_path=tmp_path, monkeypatch=monkeypatch, timed_out=False
    )
    assert _call_count(llm, "REPAIR THE PYTHON CODE") > 0
    record = records[-1]
    assert record.get("runtime_failure_class") != "execution_timeout"
    assert record.get("runtime_repair_route") != "fail_closed"
