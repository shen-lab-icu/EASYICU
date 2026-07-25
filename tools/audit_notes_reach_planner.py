"""Offline probe: does the materialization spec's `notes` reach the Planner prompt?

Zero provider calls. Builds a real ResearchContext with a sentinel notes string
that mirrors M2's patient-grouping rule, then inspects the EXACT bytes the
PlannerAgent would send, and compares against the Coder's notes channel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from easyicu.research_agent.research_context.builder import build_research_context

SENTINEL = (
    "SENTINELNOTE_M2 For the required patient-level train/test split, derive the "
    "patient group from patient_stay_id by taking the prefix before ':s'. Never "
    "split directly on the full patient_stay_id because it is unique per ICU stay."
)

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
    notes=SENTINEL,
)
ctx = getattr(authority, "context", authority)

print("context.notes set          :", bool(ctx.notes))
print("context.notes contains tag :", "SENTINELNOTE_M2" in (ctx.notes or ""))

# ---- 1. outbound-safe payload (the projection every outbound prompt uses) ----
from easyicu.research_agent.research_context.outbound import (  # noqa: E402
    format_outbound_safe_context,
    outbound_safe_context_payload,
)

payload = outbound_safe_context_payload(ctx)
rendered_outbound = format_outbound_safe_context(ctx)
print("\n--- outbound_safe_context ---")
print("payload top-level keys     :", sorted(payload))
print("'notes' key present        :", "notes" in payload)
print("sentinel in rendered bytes :", "SENTINELNOTE_M2" in rendered_outbound)

# ---- 2. the EXACT planner request messages ----
from easyicu.research_agent.agents.core import PlannerAgent  # noqa: E402

messages = PlannerAgent.request_messages(ctx)
planner_bytes = "\n".join(m.content for m in messages)
print("\n--- PlannerAgent.request_messages ---")
print("n messages                 :", len(messages))
print("total chars                :", len(planner_bytes))
print("sentinel in planner prompt :", "SENTINELNOTE_M2" in planner_bytes)
print("'patient_stay_id' appears  :", "patient_stay_id" in planner_bytes)
print("':s' grouping rule appears :", "prefix before" in planner_bytes)

# ---- 3. the Coder's notes channel ----
from easyicu.research_agent.agents.core import _coder_relevant_notes  # noqa: E402

coder_notes = _coder_relevant_notes(ctx.notes)
print("\n--- Coder notes channel ---")
print("sentinel in coder notes    :", "SENTINELNOTE_M2" in coder_notes)

print("\n=== VERDICT ===")
print(
    "notes -> Planner :",
    "REACHES" if "SENTINELNOTE_M2" in planner_bytes else "DOES NOT REACH",
)
print(
    "notes -> Coder   :",
    "REACHES" if "SENTINELNOTE_M2" in coder_notes else "DOES NOT REACH",
)
