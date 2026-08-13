"""The step budget is funded in logical attempts and spent in HTTP requests.

``orchestration/config.py`` certifies, at construction, that a step can pay for
the repairs the same configuration promises::

    1 initial generation
      + max_code_repair_attempts        (3)
      + max_step_llm_repair_attempts    (2)
      + 1 reserved concept audit        = 7        granted: 9

and its own docstring says why the check exists:

    "A step that exhausts its provider budget mid-repair fails, and it fails
     the way a scientifically broken step fails -- so the run reports an
     analysis problem that is really an accounting one."

Every term of that sum is a *logical* attempt: one ask, one answer.  The
counter is charged somewhere else entirely --
``provider_budget.py::_ActiveProviderCall.consume_transport_attempt``::

    self._transport_attempts += 1
    already_reserved = self._transport_attempts == 1
    if not already_reserved:
        self.budget.consume(self.category)

-- so every HTTP retry of the SAME request spends one of the repairs the
guard just certified.  With ``--transport-max-attempts 8`` a single flaky
generation can eat 8 of the 9 calls before the script has run once.

MEASURED on the recorded corpus, from the runs' own budget receipts (599 of
them): 13 steps were charged more provider calls than they made logical
generation attempts.  The worst is the one that motivated this file --

    h1_ventilation_survival / 04_event_censoring_audit
        logical generations: 2      charged: 5
        code repairs taken:  0      (the configuration promises 3)
        used 8 of 9, and the 9th is reserved for the concept audit
        outcome: execution_failed

That step died on a host helper raising ``ValueError`` -- a textbook repairable
traceback -- with zero of its three repairs spent.  Two more steps
(h1 ``07_cox_summary_figure``, m1 ``04_absolute_risk_context``) ended at
exactly 9 of 9.

The charge is not what bounds retries.  ``providers/llm.py`` owns that::

    manual_attempts = 1 + max(0, int(getattr(self, "_max_retries", 8)))
    for attempt in range(manual_attempts):

and nothing else decides whether to attempt.  (A second check did consult
the step allowance around the *sleep*; once the allowance stopped cancelling
retries it survived only to suppress the backoff of a retry that was going to
happen anyway, so it was removed with this change.)  The run/batch hard stop
is consumed separately
and first, by ``consume_active_transport_attempt`` itself.  So the step budget
contributes no bound that is not already held by two other owners -- it only
converts an infrastructure failure into a scientific one, which is precisely
what the guard beside it exists to prevent.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from easyicu.research_agent.agents.coder_generation import (
    MAX_INITIAL_GENERATION_ATTEMPTS,
)
from easyicu.research_agent.authority.provider_budget import (
    StepProviderCallBudget,
    complete_with_provider_budget,
    consume_active_transport_attempt,
)
from easyicu.research_agent.orchestration.config import (
    step_provider_call_entitlement,
)

_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


def _budget(limit: int = 9) -> StepProviderCallBudget:
    return StepProviderCallBudget(limit, step_id="04_event_censoring_audit")


def _call_retrying(budget: StepProviderCallBudget, *, retries: int) -> None:
    """One logical call whose transport fails ``retries`` times, then succeeds.

    This is the exact shape ``providers/llm.py`` produces: a single
    ``complete_with_provider_budget`` wrapping a loop that calls
    ``consume_active_transport_attempt`` once per HTTP attempt.
    """

    def _transport() -> str:
        for _ in range(retries + 1):
            consume_active_transport_attempt()
        return "ok"

    complete_with_provider_budget(
        budget=budget, category="initial_generation", call=_transport
    )


def test_one_request_retried_four_times_costs_four_repairs():
    """The defect, at its smallest: same question asked once, charged 4 times."""

    budget = _budget()
    _call_retrying(budget, retries=3)

    assert budget.used == 1, (
        "a single logical call that the transport retried consumed "
        f"{budget.used} of the step's {budget.limit} provider calls"
    )


def test_the_repairs_the_configuration_promised_survive_a_flaky_transport():
    """End state, in the units the guard is written in.

    The guard funds 1 generation + 3 code repairs + 2 LLM repairs + 1 audit.
    After a generation whose transport retried, all three code repairs must
    still be affordable -- that is the whole content of the certificate.

    Seven retries is not a worst case invented for this test: the launcher
    passes ``--transport-max-attempts 8``, so eight HTTP attempts for one
    question is the configured maximum, and h1's recorded step spent five on
    generation alone before its script had run once.
    """

    budget = _budget()
    _call_retrying(budget, retries=7)

    for index in range(3):
        assert budget.can_consume("runtime_repair_patch"), (
            f"code repair {index + 1} of 3 is unaffordable after one retried "
            f"generation: used {budget.used} of {budget.limit}"
        )
        budget.consume("runtime_repair_patch")


def test_the_entitlement_counts_the_generation_policy_the_host_declares():
    """The guard's first term is a literal 1; the generation layer says 2.

    ``coder_generation.py`` declares ``MAX_INITIAL_GENERATION_ATTEMPTS = 2``
    -- "at most one audited regeneration" -- so even the host's own documented
    happy path may spend two where the sum funds one.
    """

    entitlement = step_provider_call_entitlement(
        max_code_repair_attempts=3,
        max_step_llm_repair_attempts=2,
        llm_concept_audit_enabled=True,
    )

    assert MAX_INITIAL_GENERATION_ATTEMPTS == 2
    assert entitlement == MAX_INITIAL_GENERATION_ATTEMPTS + 3 + 2 + 1, entitlement


def test_a_first_attempt_is_still_charged():
    """Removing the retry charge must not stop charging the call itself."""

    budget = _budget()
    _call_retrying(budget, retries=0)

    assert budget.used == 1
    assert budget.categories == ("initial_generation",)


def test_two_logical_calls_are_two_charges():
    """Distinct asks stay distinct: this is what the budget is for."""

    budget = _budget()
    _call_retrying(budget, retries=5)
    _call_retrying(budget, retries=5)

    assert budget.used == 2, budget.categories


def test_the_transport_retry_bound_lives_elsewhere_and_is_unchanged():
    """Proof that dropping the charge weakens no bound.

    ``providers/llm.py`` bounds retries with its own ``manual_attempts`` loop,
    which this change does not touch. No part of the retry path consults the
    step allowance any more -- neither to attempt nor to back off.
    """

    import inspect

    from easyicu.research_agent.providers import llm

    source = inspect.getsource(llm)
    assert (
        'manual_attempts = 1 + max(0, int(getattr(self, "_max_retries", 8)))'
        in source
    )
    assert "for attempt in range(manual_attempts):" in source
    assert "active_provider_retry_available" not in source

    # And the backoff is decided by the attempt count alone.
    helper = source[
        source.index("def _sleep_before_retry(") : source.index(
            "for attempt in range(manual_attempts):"
        )
    ]
    condition = [
        line.strip() for line in helper.splitlines() if line.strip().startswith("if ")
    ]
    assert condition == ["if attempt_index + 1 < manual_attempts:"], condition


def test_the_run_wide_stop_loss_is_still_charged_for_every_attempt(monkeypatch):
    """The independent ceiling must keep counting every real request.

    This is the bound that replaces nothing -- it was always the one doing the
    work -- so it must still fire once per HTTP attempt, retries included, and
    it must remain the value the caller receives back.
    """

    from easyicu.research_agent.authority import provider_hard_stop

    charged: list[int] = []
    monkeypatch.setattr(
        provider_hard_stop,
        "consume_active_provider_hard_stop_attempt",
        lambda: (charged.append(1), 123.0)[1],
    )

    budget = _budget()
    _call_retrying(budget, retries=7)

    assert len(charged) == 8, charged
    assert budget.used == 1


def test_the_recorded_receipts_show_the_overcharge():
    """Re-measures the corpus rather than restating it."""

    if not _CORPUS.exists():
        pytest.skip("recorded run corpus is not mounted")

    receipts = overcharged = 0
    worst = None
    for path in _CORPUS.glob(
        "batch_*/*/aware/run_*/.runtime/provider_call_budgets/*.json"
    ):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        categories = record.get("categories") or []
        if not categories:
            continue
        receipts += 1
        logical = len(record.get("initial_generations") or [])
        charged = sum(1 for item in categories if item == "initial_generation")
        if charged > max(1, logical):
            overcharged += 1
            excess = charged - max(1, logical)
            if worst is None or excess > worst[0]:
                worst = (excess, str(record.get("step_id")), logical, charged)

    if not receipts:
        pytest.skip("no recorded budget receipt could be parsed")
    assert receipts > 100, receipts
    assert overcharged > 0, "no recorded step was charged beyond its logical asks"
    assert worst is not None and worst[0] >= 3, worst
