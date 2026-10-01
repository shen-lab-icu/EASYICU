"""A reviewed survival design replans on the signed suite with its accepted Table 1.

After review compiles a proposed survival design, the next candidate is planned
on the signed landmark survival suite and keeps the reviewed candidate's
Table 1 rows.  Two host owners still read the signed suite as if it had no
Table 1 and no design of its own:
- the family-spec request refused any accepted roster for a sealed suite
  before the Provider call, although the suite describes its adjustment
  columns by exposure status;
- the plan review looked for those rows only in a plan Table 1 step, and read
  the compiled landmark design the suite executes as a user-required
  sensitivity analysis left protocol-only.
Synthetic study only (renal replacement therapy and 90-day mortality); zero
patient rows.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
from easyicu.research_agent.planning.family_spec import FamilySpecError
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from tests.support.survival_sealed import (
    accepting as _accepting,
    bound_survival_plan as _bound_plan,
    sealed_request as _request,
    sealed_survival as _sealed,
)

BASELINE = "ACCEPTED_BASELINE_CONTENT_MISSING"
PROTOCOL_ONLY = "REQUIRED_SENSITIVITY_IS_PROTOCOL_ONLY"


def _codes(context, plan, authority):
    review = build_plan_scientific_review(
        context=context, plan=plan, require_reportable_capability=True, runtime_authority=authority,
    )
    return review, {finding.code for finding in review.findings if finding.severity == "blocker"}


def test_the_reviewed_table_one_survives_the_sealed_replan_to_an_approvable_plan(tmp_path):
    context, authority = _sealed(tmp_path)
    assert authority.table_one_columns == ("age", "sex")
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)
    accepting = _accepting(context, ["age", "sex"])

    request = _request(accepting, authorities)
    assert request.sealed_suite is not None and request.sealed_suite.adjustment_columns == ["age", "sex"]
    plan = _bound_plan(accepting, authorities)
    suite = next(step for step in plan.steps if step.method == "signed_landmark_survival_suite")
    assert authority.table_one_product in suite.expected_outputs
    assert not any(step.table_one_spec is not None for step in plan.steps)

    review, blockers = _codes(accepting, plan, authority)
    assert BASELINE not in blockers and PROTOCOL_ONLY not in blockers
    assert review.approval_allowed is True


def test_a_row_or_grouping_the_sealed_table_does_not_describe_is_refused_before_the_provider(tmp_path):
    context, authority = _sealed(tmp_path)
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)

    with pytest.raises(FamilySpecError) as caught:
        _request(_accepting(context, ["age", "weight"]), authorities)
    assert caught.value.reason_code == "family_spec_accepted_baseline_unsatisfiable"
    assert "'weight'" in str(caught.value) and "'age'" not in str(caught.value).split("not ")[-1]

    with pytest.raises(FamilySpecError) as caught:
        _request(_accepting(context, ["age"], group="sex"), authorities)
    assert caught.value.reason_code == "family_spec_accepted_baseline_grouping_unsupported"
    # An ungrouped roster of sealed columns is kept.
    assert _request(_accepting(context, ["sex"], group=None), authorities).sealed_suite is not None


def test_the_review_credits_only_the_design_and_table_the_signed_suite_owns(tmp_path):
    other = {
        "spec_id": "user_landmark_48h", "axis": "timing", "strategy": "landmark", "landmark_hours": 48.0,
        "require_alive_at_landmark": True, "exclude_negative_event_times": True,
        "observation_duration_variable": "followup_days_90d", "observation_duration_unit": "days",
    }
    context, authority = _sealed(tmp_path, review_only_specs=(other,))
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)
    accepting = _accepting(context, ["age", "sex"])
    plan = _bound_plan(accepting, authorities)

    review, blockers = _codes(accepting, plan, authority)
    # A second landmark the suite does not run is still a missing sensitivity.
    protocol_only = next(finding for finding in review.findings if finding.code == PROTOCOL_ONLY)
    assert "user_landmark_48h" in protocol_only.message
    assert "agent_plan_survival_landmark_24h" not in protocol_only.message
    # Without the signed authority nothing proves the suite's Table 1 roster.
    _review, unsigned = _codes(accepting, plan, None)
    assert BASELINE in unsigned
