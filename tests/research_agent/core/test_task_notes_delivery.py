"""What a task ``notes`` string actually reaches.

`ResearchContext.notes` is where a study rule lands when it is not one of the
typed fields — M2's "split on the patient prefix, not the stay id", H2's
exposure-missingness semantics. Two separate readings of the source say it is
delivered to the Coder:

* ``_coder_relevant_notes`` exists and returns the notes unchanged, and
* ``CoderAgent.repair`` computes ``include_scientific_authority`` from a set of
  scientific repair reasons and passes both to the repair-context renderer.

Neither reading is delivery. ``format_repair_authority_context`` opens with
``del include_scientific_authority, user_notes``. These tests measure the bytes
an agent hands to a client, so the gap cannot be re-argued from either side.

They pin current behaviour, not desired behaviour: feeding free-form user text
straight into a provider prompt is an egress and prompt-hygiene decision, not a
bug fix. If a typed task protocol is built later, these tests are what tells
you the delivery actually changed.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.agents.core import (
    CoderAgent,
    PlannerAgent,
    _coder_relevant_notes,
)
from easyicu.research_agent.providers.mocks import ExternalCaptureMockLLMClient
from easyicu.research_agent.repairs.reasons import RepairPromptAuthority, RepairReason
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.research_context.repair_prompt import (
    format_repair_authority_context,
)
from easyicu.research_agent.schema import AnalysisStep

SENTINEL = "SENTINELNOTE_PATIENT_SPLIT"
NOTES = (
    f"{SENTINEL} Derive the patient group from patient_stay_id by taking the "
    "prefix before ':s'; never split on the full patient_stay_id."
)
SCRIPT = "import os\nprint(os.environ['COHORT_PARQUET'])\n"


@pytest.fixture()
def ctx() -> Any:
    rng = np.random.default_rng(0)
    n = 40
    cohort = pd.DataFrame(
        {
            "patient_stay_id": [f"p{i // 2:03d}:s{i % 2}" for i in range(n)],
            "age": rng.integers(40, 85, n).astype(float),
            "death": rng.integers(0, 2, n),
        }
    )
    authority = build_research_context(
        cohort=cohort,
        research_question="Predict in-hospital mortality.",
        cohort_name="probe",
        database="miiv",
        target_outcome="death",
        id_columns=["patient_stay_id"],
        notes=NOTES,
    )
    return getattr(authority, "context", authority)


@pytest.fixture()
def step() -> AnalysisStep:
    return AnalysisStep(
        step_id="03_prediction",
        intent="Fit a patient-level split model.",
        expected_outputs=["auroc_table"],
        method="logistic_regression",
    )


def _sent_bytes(calls: Sequence[Any]) -> str:
    return "\n".join(
        str(message.content) for messages, _kwargs in calls for message in messages
    )


def test_the_context_really_does_carry_the_note(ctx: Any) -> None:
    """Rules out the trivial explanation that the note was dropped at intake."""

    assert SENTINEL in (ctx.notes or "")


def test_the_planner_decides_the_study_without_the_note(ctx: Any) -> None:
    """The Planner picks the design; a rule it never reads cannot constrain it."""

    prompt = "\n".join(m.content for m in PlannerAgent.request_messages(ctx))
    assert prompt, "planner prompt must not be empty for this check to mean anything"
    assert SENTINEL not in prompt


def test_the_coder_writes_the_first_script_without_the_note(
    ctx: Any, step: AnalysisStep
) -> None:
    capture = ExternalCaptureMockLLMClient([f"```python\n{SCRIPT}```"] * 4)
    CoderAgent(capture).run(context=ctx, step=step)

    assert capture.calls, "the probe must observe a real generation call"
    assert SENTINEL not in _sent_bytes(capture.calls)


@pytest.mark.parametrize(
    "reason",
    [
        RepairReason.UNDEFINED_HELPER.value,
        # The scientific reasons are the ones `CoderAgent.repair` tests for
        # before deciding to attach notes at all.
        RepairReason.SCIENTIFIC_SEMANTICS_VIOLATION.value,
        RepairReason.ROW_ALIGNMENT_UNVERIFIED.value,
    ],
)
def test_no_repair_reason_delivers_the_note(
    ctx: Any, step: AnalysisStep, reason: str
) -> None:
    """Including the scientific reasons that select the notes channel."""

    authority = RepairPromptAuthority.create(typed_ticket=[{"reason": reason}])
    capture = ExternalCaptureMockLLMClient([f"```python\n{SCRIPT}```"] * 6)
    CoderAgent(capture).repair(
        context=ctx,
        step=step,
        code=SCRIPT,
        run_log="Traceback (most recent call last): NameError",
        repair_authority=authority,
        current_repair_authority=authority,
    )

    assert capture.calls, "the probe must observe a real repair call"
    assert SENTINEL not in _sent_bytes(capture.calls)


def test_the_notes_filter_passes_what_the_renderer_then_discards() -> None:
    """Name the seam, so a future reader is not misled the way this one was.

    The filter returns the note, so anyone reading `_coder_relevant_notes`
    concludes the Coder gets it. The renderer that receives it is where the
    channel actually ends.
    """

    assert _coder_relevant_notes(NOTES) == NOTES.strip()

    context_without_notes = build_research_context(
        cohort=pd.DataFrame({"patient_stay_id": ["p0:s0"], "death": [0]}),
        research_question="Predict in-hospital mortality.",
        cohort_name="probe",
        database="miiv",
        target_outcome="death",
        id_columns=["patient_stay_id"],
    )
    inner = getattr(context_without_notes, "context", context_without_notes)

    scientific_context = format_repair_authority_context(
        inner, include_scientific_authority=True, user_notes=NOTES
    )
    mechanical_context = format_repair_authority_context(
        inner, include_scientific_authority=False, user_notes=""
    )
    assert SENTINEL not in scientific_context
    assert SENTINEL not in mechanical_context
    assert len(mechanical_context.encode("utf-8")) < len(
        scientific_context.encode("utf-8")
    )
