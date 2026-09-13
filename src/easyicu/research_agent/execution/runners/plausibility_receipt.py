"""Execution-side entry points for the host-rendered plausibility receipt.

The generated source, its exact-tail reader, and the scope-guided renderer live
in ``authority/plausibility_receipt_code``; this module keeps the execution
API stable for the runners, phase and candidate loop that already import it.
"""

from __future__ import annotations

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...authority.plausibility_receipt_code import (
    render_host_plausibility_receipt_tail,
    render_standard_plausibility_receipt_code,
    verified_host_plausibility_receipt_region,
)

__all__ = [
    "host_plausibility_receipt_injected",
    "render_host_plausibility_receipt_tail",
    "render_standard_plausibility_receipt_code",
    "verified_host_plausibility_receipt_region",
]


def host_plausibility_receipt_injected(
    code: str,
    *,
    scope: FlagOnlyPlausibilityScope | None,
    already_satisfied: bool,
) -> str:
    """Return ``code`` with the host's own receipt appended, when it is owed.

    MEASURED over every recorded run, ``flag_only_plausibility_obligation`` is
    the single largest pre-execution blocker: 37 findings across 32 distinct
    steps in 8 of the 9 tasks, 53 % of all mechanical-preflight findings. The
    obligation is mechanical -- read each declared column's bounds from the
    sealed manifest, count what falls outside, file the counts under one exact
    key -- and the host renders it correctly for its own executors. Only
    agent-authored steps must hand-write it, and they get it wrong: h2's
    causal step spent BOTH of its LLM repairs on this one message, with five
    provider calls still unspent, and died anyway.

    The alternative considered and rejected was a host helper the agent calls.
    It fails on the decisive point: it still depends on the agent REMEMBERING
    to call it, which is the exact thing that fails 37 times. This module's own
    docstring gives the second reason -- the comparisons are rendered into the
    source so the static gate can verify the code that will actually run, which
    an imported helper defeats.

    Injection happens before the concept audit, so the approved digest and the
    executed digest both cover the assembled script and their identity is
    preserved by construction.
    """

    body = str(code or "")
    if scope is None or not scope.expected_columns or already_satisfied:
        return body
    if not body.strip():
        return body

    return body.rstrip() + "\n\n" + render_host_plausibility_receipt_tail(scope) + "\n"
