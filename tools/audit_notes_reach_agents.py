"""Offline probe: where can a task ``notes`` string actually influence a run?

Zero provider calls. Builds a real ResearchContext carrying a sentinel note
that mirrors M2's patient-grouping rule, then drives the REAL agents with a
capturing client and inspects the exact bytes each stage would send.

Why the capturing client matters: an earlier version of this probe asked
``_coder_relevant_notes(ctx.notes)`` whether the Coder would see the note and
reported "REACHES". That function is a filter, not a delivery channel — it is
called only from ``CoderAgent.repair``, never from ``CoderAgent.run``. Reading
the helper answered "would this note survive the filter", not "does the Coder
receive it", and the two answers differ for every initial generation. Every
verdict below is now taken from messages an agent actually handed to a client.

Measured result: no path delivers the note. ``CoderAgent.repair`` computes
``include_scientific_authority`` from a set of scientific repair reasons and
passes it, with the filtered notes, to
``research_context.repair_prompt.format_repair_authority_context`` — which
opens with ``del include_scientific_authority, user_notes``. The gating logic
runs; its result is discarded. Reading either side alone suggests a working
conditional channel, which is why this probe measures the bytes instead.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from easyicu.research_agent.agents.core import CoderAgent, PlannerAgent
from easyicu.research_agent.providers.factory import provider_transport_destination
from easyicu.research_agent.providers.mocks import ExternalCaptureMockLLMClient
from easyicu.research_agent.repairs.reasons import RepairPromptAuthority, RepairReason
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.schema import AnalysisStep

SENTINEL = "SENTINELNOTE_M2"
NOTES = (
    f"{SENTINEL} For the required patient-level train/test split, derive the "
    "patient group from patient_stay_id by taking the prefix before ':s'. Never "
    "split directly on the full patient_stay_id because it is unique per ICU stay."
)

SCRIPT = (
    "import os\n"
    "import pandas as pd\n"
    "df = pd.read_parquet(os.environ['COHORT_PARQUET'])\n"
    "print(len(df))\n"
)


def _build_context() -> Any:
    rng = np.random.default_rng(0)
    n = 60
    cohort = pd.DataFrame(
        {
            "patient_stay_id": [f"p{i // 2:03d}:s{i % 2}" for i in range(n)],
            "age": rng.integers(40, 85, n).astype(float),
            "hr_max": rng.normal(95, 12, n),
            "lact_max": rng.normal(2.4, 0.8, n),
            "death": rng.integers(0, 2, n),
        }
    )
    authority = build_research_context(
        cohort=cohort,
        research_question="Predict in-hospital mortality from first-24h physiology.",
        cohort_name="probe",
        database="miiv",
        target_outcome="death",
        id_columns=["patient_stay_id"],
        notes=NOTES,
    )
    return getattr(authority, "context", authority)


def _step() -> AnalysisStep:
    return AnalysisStep(
        step_id="03_prediction",
        intent="Fit a patient-level split logistic model for in-hospital mortality.",
        expected_outputs=["auroc_table"],
        method="logistic_regression",
    )


def _sentinel_in(messages: Sequence[Any]) -> bool:
    return SENTINEL in "\n".join(str(m.content) for m in messages)


def _report(label: str, calls: Sequence[Any], *, note: str = "") -> bool:
    """Report whether any prompt this stage actually sent carried the note."""

    suffix = (" — " + note) if note else ""
    if not calls:
        print(f"  {label:<34} NO PROVIDER CALL{suffix}")
        return False
    reached = any(_sentinel_in(messages) for messages, _kwargs in calls)
    chars = sum(
        len("\n".join(str(m.content) for m in messages)) for messages, _ in calls
    )
    verdict = "REACHES" if reached else "DOES NOT REACH"
    print(
        f"  {label:<34} {verdict:<15} " f"({len(calls)} call(s), {chars} chars){suffix}"
    )
    return reached


def main() -> None:
    ctx = _build_context()
    step = _step()
    print(f"context.notes carries {SENTINEL}:", SENTINEL in (ctx.notes or ""))

    results: dict[str, bool] = {}

    # ---- 1. Planner: inspect the exact request messages it would send. ----
    print("\n--- initial planning ---")
    planner_messages = PlannerAgent.request_messages(ctx)
    results["Planner initial prompt"] = _report(
        "Planner initial prompt", [(planner_messages, {})]
    )

    # ---- 2. Coder initial generation: drive the real agent. ----
    print("\n--- initial code generation ---")
    capture = ExternalCaptureMockLLMClient([f"```python\n{SCRIPT}```"] * 4)
    print("  measured transport:", provider_transport_destination(capture))
    agent = CoderAgent(capture)
    before = len(capture.calls)
    try:
        agent.run(context=ctx, step=step)
    except Exception as exc:  # noqa: BLE001 - a probe reports, it does not fail
        print(f"  (run raised {type(exc).__name__}: {exc})")
    results["Coder initial generation"] = _report(
        "Coder initial generation", capture.calls[before:]
    )

    # ---- 3. Repair: the notes channel is gated on the repair reason. ----
    print("\n--- repair ---")
    for label, reason, note in (
        (
            "Coder repair (mechanical reason)",
            RepairReason.UNDEFINED_HELPER.value,
            "notes withheld unless the reason is scientific",
        ),
        (
            "Coder repair (scientific reason)",
            RepairReason.SCIENTIFIC_SEMANTICS_VIOLATION.value,
            "",
        ),
    ):
        authority = RepairPromptAuthority.create(typed_ticket=[{"reason": reason}])
        capture = ExternalCaptureMockLLMClient([f"```python\n{SCRIPT}```"] * 6)
        agent = CoderAgent(capture)
        before = len(capture.calls)
        try:
            agent.repair(
                context=ctx,
                step=step,
                code=SCRIPT,
                run_log="Traceback (most recent call last): NameError",
                repair_authority=authority,
                current_repair_authority=authority,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ({label} raised {type(exc).__name__}: {exc})")
        results[label] = _report(label, capture.calls[before:], note=note)

    print("\n=== VERDICT ===")
    for label, reached in results.items():
        print(f"  {label:<34} {'REACHES' if reached else 'DOES NOT REACH'}")
    if not any(results.values()):
        print(
            "\nNo stage delivers the note. A study rule recorded only in `notes` —\n"
            "M2's patient-level split, H2's exposure-missingness semantics — is not\n"
            "something the agent gets late; it is something the agent never gets."
        )


if __name__ == "__main__":
    main()
