"""What the Planner is told when its declared family replaced its contract.

The article contract shown to the Planner is compiled from the family the host
inferred; the one that judges its plan, from the family the plan declared.
``agents.planner`` appends this note to a rejection when the two differ.
"""

from __future__ import annotations

from typing import Any


def describe_article_contract_family_switch(*, shown: Any, judged: Any) -> str:
    """Say that the declared family replaced the published contract, and publish it.

    The article contract shown to the Planner is compiled from the family the
    host *inferred* from the research context. The contract that judges the
    plan is recompiled from the family the plan *declared*. When those differ,
    every required role is decided by a document the Planner never saw.

    Measured on 2026-07-29: E1's context infers ``survival``
    (``diagnostics``/``survival_effect``/``temporal_absolute_risk``), and the
    Planner declared ``association_study`` -- the right label for a binary
    in-hospital-mortality outcome, and what the previously accepted plan
    declared. It was then judged on ``primary_estimand`` and ``robustness``,
    which had never been published to it, and one attempt was told to produce
    ``table:survival_curve`` for a binary outcome. Five attempts, five distinct
    violations, nothing executed.

    Which side is right is genuinely open -- the inference read a time-to-event
    *sensitivity* as the whole design, and the Planner was arguably closer --
    so this does not pick a winner. It removes the part that is indefensible
    either way: discovering a contract one missing role per paid attempt. The
    judging family's whole required set is stated here so a single retry can
    satisfy it, or reconsider the declaration knowing what it costs.

    Returns ``""`` when the families agree, so the ordinary rejection is not
    padded with a switch that did not happen.
    """

    shown_family = str(getattr(shown, "source_analysis_type", "") or "")
    judged_family = str(getattr(judged, "source_analysis_type", "") or "")
    if not shown_family or not judged_family or shown_family == judged_family:
        return ""
    required = ", ".join(str(role) for role in judged.required_roles)
    return (
        f" NOTE: the article contract you were shown was compiled for "
        f"analysis_type={shown_family}; your plan declares "
        f"analysis_type={judged_family}, which REPLACED it. The full required "
        f"role set for {judged_family} is: {required}. Either cover all of "
        f"them, or declare analysis_type={shown_family} and cover the contract "
        "you were shown -- do not alternate between the two."
    )


__all__ = ["describe_article_contract_family_switch"]
