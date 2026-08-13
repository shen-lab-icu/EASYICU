"""The concept audit is the last gate -- and it could be refused unheard.

The per-step LLM-repair allowance was one flat pool of 2, drawn on by six
different repair classes. Five of them (runtime, contract, compatibility,
visual, critic_resume) are discovered BEFORE the concept audit, which reads the
code only after it has run. So a step could spend its whole allowance answering
an earlier failure and then be blocked at a gate it was never given one chance
to answer.

MEASURED over every recorded run (1,552 step records): 80 steps ended
``blocked_by_concept_audit``. **30 of them -- across 7 of the 9 tasks -- had
spent ZERO concept-class repairs.** The pool had gone to ``runtime`` (14),
``contract`` (6 + 3 mixed), ``compatibility`` (2), ``visual`` (1) and others.
28 of those 30 still had unspent provider calls (3 to 7 of 9).

A concept repair is worth attempting: of the 171 steps that ever spent one,
65 (38 %) finished ``ok``.

The reserve is ADDITIVE. The strict alternative -- holding one of the existing
two back for the terminal gate -- was measured and rejected: 40 steps that
currently finish ``ok`` spent two non-concept repairs and would have lost one.

This is not a new idea in the codebase. The provider-call budget one level down
already reserves its final slot for the terminal gate
(``reserved_final_category = "concept_audit"``). The logical allowance simply
never had the same rule.
"""

from __future__ import annotations

import ast
import collections
import hashlib
import inspect
import json
import pathlib

import pytest

from easyicu.research_agent.authority.provider_budget import StepProviderCallBudget
from easyicu.research_agent.repairs.coordination import (
    TERMINAL_GATE_REPAIR_CLASSES,
    TERMINAL_STAGE_REPAIR_CLASSES,
    StepRepairBudget,
)

_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")
STEP_ID = "06_fit_and_evaluate_prediction_model"


def _budget(tmp_path, *, max_llm: int = 2, provider_limit: int = 9):
    provider = StepProviderCallBudget(
        provider_limit,
        step_id=STEP_ID,
        receipt_path=tmp_path / "receipt.json",
        reserved_final_category="concept_audit",
    )
    step_record: dict = {}
    return (
        provider,
        step_record,
        StepRepairBudget(
            provider_budget=provider,
            step_record=step_record,
            max_llm_repairs=max_llm,
            provider_receipt_relative_path=".runtime/provider_call_budgets/x.json",
        ),
    )


def _spend(budget, provider, repair_class: str, *, category: str = "repair") -> bool:
    """Consume one repair AND settle its transport, as a real attempt does.

    A reservation stays pending until its result is durable; leaving it pending
    makes the next reservation fail closed on resume grounds, which is a
    different rule from the one under test.
    """

    if not budget.consume(repair_class):
        return False
    attempt_id = budget.llm_repair_attempts
    provider.consume(category)
    provider.complete_logical_repair_transport(
        attempt_id=attempt_id,
        mode="minimal_patch",
        after_code_sha256=hashlib.sha256(
            f"# repaired {attempt_id}\n".encode("utf-8")
        ).hexdigest(),
    )
    return True


# ---------------------------------------------------------------------------
# The recorded death, reproduced
# ---------------------------------------------------------------------------


def test_two_runtime_repairs_no_longer_starve_the_concept_gate(tmp_path):
    """m2's step 06, exactly: classes ``('runtime', 'runtime')``, then blocked.

    Its manifest recorded ``concept_repair_attempts: 0`` with 5 provider calls
    unspent -- the concept auditor's finding (missing ``sex`` silently encoded
    as the reference category) never got one attempt.
    """

    provider, record, budget = _budget(tmp_path)

    assert _spend(budget, provider, "runtime")
    assert _spend(budget, provider, "runtime")

    # Every earlier class is now correctly refused: the pool is spent.
    assert not budget.available("runtime")
    assert not budget.available("contract")
    assert not budget.available("visual")
    assert not budget.available()

    # The terminal gate is not.
    assert budget.available("concept")
    assert _spend(budget, provider, "concept")
    assert record["step_llm_repair_classes"] == ["runtime", "runtime", "concept"]
    # And the record says WHY it shows 3 attempts against a budget of 2.
    assert record["step_llm_repair_budget"] == 2
    assert record["step_llm_repair_terminal_gate_reserve"] == ["concept"]


def test_a_step_that_never_needed_the_reserve_does_not_claim_one(tmp_path):
    provider, record, budget = _budget(tmp_path)

    assert _spend(budget, provider, "runtime")
    assert _spend(budget, provider, "concept")
    assert "step_llm_repair_terminal_gate_reserve" not in record


def test_the_reserve_is_one_attempt_and_not_a_second_pool(tmp_path):
    provider, _record, budget = _budget(tmp_path)

    assert _spend(budget, provider, "runtime")
    assert _spend(budget, provider, "runtime")
    assert _spend(budget, provider, "concept")
    assert not budget.available("concept")
    assert not budget.available("post_mutation_concept")
    assert not budget.consume("concept")


def test_a_step_that_already_answered_the_gate_gets_no_extra(tmp_path):
    """Spending a concept repair first must not buy a third general one."""

    provider, _record, budget = _budget(tmp_path)

    assert _spend(budget, provider, "concept")
    assert _spend(budget, provider, "runtime")
    assert not budget.available("runtime")
    assert not budget.available("concept")
    assert budget.effective_limit("concept") == 2


def test_the_two_terminal_classes_share_one_reserve(tmp_path):
    """``post_mutation_concept`` is the same gate, seen after a mutation."""

    provider, _record, budget = _budget(tmp_path)

    assert _spend(budget, provider, "contract")
    assert _spend(budget, provider, "contract")
    assert budget.available("post_mutation_concept")
    assert _spend(budget, provider, "post_mutation_concept")
    assert not budget.available("concept")


# ---------------------------------------------------------------------------
# It costs the steps that currently succeed nothing
# ---------------------------------------------------------------------------


def test_a_pre_execution_class_still_sees_exactly_the_configured_pool(tmp_path):
    """The steps that finish ok on two ordinary repairs are untouched.

    RENAMED. ``runtime`` used to be listed here as "an earlier class". It is
    not: execution is the LAST thing that happens, and a traceback cannot exist
    until every pre-execution gate has already had its chance at the pool. It
    now has a reserve of its own, so it is asserted with the other terminal
    stages below instead.
    """

    provider, _record, budget = _budget(tmp_path)

    assert budget.effective_limit("contract") == 2
    assert budget.effective_limit("visual") == 2
    assert budget.effective_limit("compatibility") == 2
    assert budget.effective_limit("critic_resume") == 2
    assert budget.effective_limit(None) == 2
    assert budget.effective_limit("") == 2

    assert _spend(budget, provider, "visual")
    assert _spend(budget, provider, "runtime")
    assert not budget.available("runtime")


def test_the_provider_budget_still_binds(tmp_path):
    """The reserve is logical only; it cannot conjure a provider call.

    The provider budget holds its own final slot for the concept AUDIT, so a
    step with no affordable non-audit call is refused even at the terminal gate
    -- which is right: the audit itself must still be payable.
    """

    provider, record, budget = _budget(tmp_path, provider_limit=3)
    provider.consume("initial_generation")
    provider.consume("runtime_repair")

    assert not budget.available("concept")
    assert record.get("step_provider_call_repair_unavailable") is True


# ---------------------------------------------------------------------------
# It is reachable: the real call sites ask with the class
# ---------------------------------------------------------------------------


def test_the_execute_phase_asks_the_gate_by_name():
    """A class-blind guard would refuse before ``consume`` was ever reached.

    Several call sites raise ``AssertionError`` if ``consume`` returns False
    after their own availability check passed, so the guard and the consume
    must agree on the class. This walks the module rather than grepping: a
    source-text test would survive the argument being dropped from one site.
    """

    from easyicu.research_agent.execution import phase as execution_phase
    from easyicu.research_agent.execution import concept_repair

    trees = (
        ast.parse(inspect.getsource(execution_phase)),
        ast.parse(inspect.getsource(concept_repair.run_concept_repair_loop)),
    )
    asked: collections.Counter = collections.Counter()
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            if function_name not in {
                "_llm_repair_budget_available",
                "_logical_llm_repair_budget_available",
                "repair_budget_available",
                "logical_budget_available",
            }:
                continue
            if node.args and isinstance(node.args[0], ast.Constant):
                asked[str(node.args[0].value)] += 1

    for terminal_class in sorted(TERMINAL_GATE_REPAIR_CLASSES):
        assert asked[terminal_class] >= 1, (
            f"no guard asks the budget about {terminal_class!r}; the terminal "
            f"gate would be refused before it could spend its reserve: {asked}"
        )


def test_the_concept_repair_is_gated_on_that_same_availability_check():
    """The premise of the whole change, read from the source.

    In m2's record ``concept_repair_attempts`` was 0 while
    ``provider_call_remaining`` was 5, so neither the per-class attempt cap nor
    the provider budget stopped it -- the logical pool did.
    """

    from easyicu.research_agent.execution import phase as execution_phase
    from easyicu.research_agent.execution import concept_repair

    source = inspect.getsource(execution_phase) + inspect.getsource(
        concept_repair.run_concept_repair_loop
    )
    assert 'or not services.repair_budget_available("concept")' in source


# ---------------------------------------------------------------------------
# The corpus record that motivated it
# ---------------------------------------------------------------------------


def _blocked_records():
    for path in _CORPUS.glob("batch_*/*/aware/run_*/manifest.json"):
        try:
            manifest = json.loads(path.read_text())
        except Exception:  # noqa: BLE001 - a malformed manifest is not the subject
            continue
        for record in manifest.get("per_step_records", []):
            if record.get("status") == "blocked_by_concept_audit":
                yield path.relative_to(_CORPUS).parts[1], record


def test_the_recorded_starvation_is_broad_and_not_one_task():
    """A fix keyed to one task would not be a fix. This spans 7 of the 9."""

    if not _CORPUS.exists():
        pytest.skip("recorded run corpus is not mounted")

    starved = [
        (task, record)
        for task, record in _blocked_records()
        if not any(
            str(item).strip() in TERMINAL_GATE_REPAIR_CLASSES
            for item in record.get("step_llm_repair_classes") or ()
        )
    ]
    if not starved:
        pytest.skip("no recorded run carries a concept-audit block")

    tasks = {task for task, _ in starved}
    assert len(tasks) >= 5, tasks
    # And they were refused with provider calls still unspent -- the reserve
    # has something to spend.
    with_headroom = [
        record
        for _task, record in starved
        if int(record.get("step_provider_call_remaining") or 0) > 0
    ]
    assert len(with_headroom) >= len(starved) // 2


# ---------------------------------------------------------------------------
# Execution is a terminal stage too, and it has its own reserve
# ---------------------------------------------------------------------------


def test_a_traceback_can_still_be_answered_after_the_gates_took_the_pool(tmp_path):
    """m2 05/06, verify31: classes ('contract','contract'), then KeyError.

    That step died ``execution_failed`` on ``KeyError: 'row_count'`` at line 52
    -- a one-line defect, the most specific repair signal the pipeline
    produces -- with SIX provider calls unspent and ``runtime_repair_attempts``
    of zero, because both repairs had gone to the contract gate before the
    script ever ran.

    MEASURED: 20 of the 89 recorded ``execution_failed`` steps, across 7 of the
    9 tasks, died with zero runtime repairs and calls remaining. Of the 211
    steps that ever spent a runtime repair, 90 (43 %) finished ok.
    """

    provider, record, budget = _budget(tmp_path)

    assert _spend(budget, provider, "contract")
    assert _spend(budget, provider, "contract")

    assert not budget.available("contract")
    assert not budget.available("visual")
    assert budget.available("runtime")
    assert _spend(budget, provider, "runtime")
    assert record["step_llm_repair_terminal_gate_reserve"] == ["runtime"]


def test_the_two_reserves_are_independent(tmp_path):
    """Spending the audit's reserve must not cost execution its own.

    They answer different questions at different moments, and a step can
    genuinely need both: the gate refuses the script, the repaired script then
    crashes.
    """

    provider, record, budget = _budget(tmp_path)

    assert _spend(budget, provider, "contract")
    assert _spend(budget, provider, "contract")
    assert _spend(budget, provider, "concept")
    assert budget.available("runtime")
    assert _spend(budget, provider, "runtime")

    assert record["step_llm_repair_terminal_gate_reserve"] == ["concept", "runtime"]
    # And neither stage gets a second.
    assert not budget.available("runtime")
    assert not budget.available("post_mutation_concept")


def test_each_stage_reserve_is_granted_once_only(tmp_path):
    provider, _record, budget = _budget(tmp_path)

    assert _spend(budget, provider, "runtime")
    assert _spend(budget, provider, "runtime")
    # runtime is already paid, so the pool is genuinely gone for it
    assert not budget.available("runtime")
    assert budget.available("concept")
    assert _spend(budget, provider, "concept")
    assert not budget.available("concept")
    assert not budget.available("runtime")


def test_the_stages_are_disjoint_and_cover_the_two_that_run_last():
    """A class in two stages would grant two reserves for one failure."""

    seen: set[str] = set()
    for stage in TERMINAL_STAGE_REPAIR_CLASSES:
        assert not (seen & stage), stage
        seen |= set(stage)
    assert TERMINAL_GATE_REPAIR_CLASSES in TERMINAL_STAGE_REPAIR_CLASSES
    assert "runtime" in seen
    # Everything discovered BEFORE the script runs stays on the shared pool.
    for earlier in ("contract", "compatibility", "visual", "critic_resume"):
        assert earlier not in seen


def test_the_execute_phase_asks_the_budget_about_runtime_too():
    """A class-blind guard refuses before ``consume`` is ever reached."""

    import inspect

    from easyicu.research_agent.execution import phase as execution_phase

    tree = ast.parse(inspect.getsource(execution_phase))
    asked: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in {
            "_llm_repair_budget_available",
            "_logical_llm_repair_budget_available",
        }:
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            asked.add(str(node.args[0].value))

    assert "runtime" in asked, asked
