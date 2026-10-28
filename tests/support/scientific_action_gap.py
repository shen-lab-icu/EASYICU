"""Synthetic future-method gap; independent of which real methods ship."""

from easyicu.research_agent.planning.scientific_action_catalog import ScientificActionResolution


def scientific_action_gap():
    return ScientificActionResolution(
        status="alternative", requested_action_id="prediction.future_method",
        selected_action_ids=(), alternative_action_ids=("prediction.internal_validation",),
        missing_requirements=("reviewed execution owner",),
        issue_code="scientific_action_requires_user_choice", requires_user_confirmation=True,
        detail="The synthetic future method is unavailable; internal validation is a different workflow requiring explicit user choice.",
    )
