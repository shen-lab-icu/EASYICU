"""The signed survival suite projects its claims into the Results the Writer writes.

The suite's manuscript projection anchored its restricted-mean claim at
"Primary association" and its interval-specific claim at "Sensitivity and
subgroup analyses".  A survival plan requires neither subsection: the shared
Results vocabulary names its primary subsection "Survival results", and the
interval model belongs to the suite's own non-PH policy, not to a sensitivity
step.  The projection therefore failed after a complete execution and the
Writer's calls.  The interval sentence also said the PH assumption was
rejected whatever the suite observed.  Synthetic study and seeded synthetic
rows only (renal replacement therapy and 90-day mortality).
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.contracts.manuscript_result_structure import (
    PRIMARY_RESULT_HEADINGS_BY_FAMILY,
)
from easyicu.research_agent.execution.runners.landmark_survival_executor import (
    build_survival_manuscript_projection,
)
from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
from easyicu.research_agent.reporting.manuscript_projection import (
    ManuscriptProjectionError,
    project_owner_issued_manuscript_claims,
)
from easyicu.research_agent.reporting.manuscript_result_structure import required_result_subsections
from tests.support.survival_sealed import (
    bound_survival_plan,
    run_signed_suite,
    sealed_survival,
    synthetic_survival_rows,
)

REJECTED = "were retained because the proportional-hazards assumption was rejected"
NOT_REJECTED = "although the proportional-hazards assumption was not rejected"


def _writer_scaffold(headings) -> str:
    """The Abstract label and the Results subsections the Writer is required to write."""

    return (
        "## Abstract\n\n**Results:** The Writer summarised the suite.\n\n## Results\n\n"
        + "".join(f"### {heading}\n\nThe Writer's prose.\n\n" for heading in headings)
    )


def _record(summary) -> dict:
    return {
        "step_summary": summary,
        "generation_mode": "deterministic_standard",
        "step_summary_evidence_id": "statistic_step_summary_primary_survival_suite",
    }


def test_the_suite_claims_land_in_the_survival_results_the_plan_requires(tmp_path):
    context, authority = sealed_survival(tmp_path)
    plan = bound_survival_plan(context, ScientificRuntimeAuthorities(trajectory=None, current_case=authority))
    headings = required_result_subsections(plan)
    survival = PRIMARY_RESULT_HEADINGS_BY_FAMILY["survival"]
    assert survival in headings
    assert "Primary association" not in headings and "Sensitivity and subgroup analyses" not in headings

    summary = run_signed_suite(authority, synthetic_survival_rows(), tmp_path / "out")
    block = summary["reportable_survival_results"]
    projected, repairs = project_owner_issued_manuscript_claims(
        _writer_scaffold(headings), per_step_records=[_record(summary)],
    )

    assert {(item["claim_id"], item["target_label"]) for item in repairs} == {
        ("primary_rmst_contrast", "Results"),
        ("primary_rmst_contrast", survival),
        ("time_varying_association_intervals", "Results"),
        ("time_varying_association_intervals", survival),
    }
    # The closing sentence states the PH outcome the suite observed.
    rejected = not block["constant_hazard_ratio_authorized"]
    assert (REJECTED in projected) is rejected
    assert (NOT_REJECTED in projected) is (not rejected)


@pytest.mark.parametrize("rejected", [True, False])
def test_the_interval_sentence_follows_the_observed_ph_outcome(rejected):
    projection = build_survival_manuscript_projection(interval_count=2, proportional_hazards_rejected=rejected)
    closing = projection["claims"][1]["fragments"][-1]["text"]
    assert (REJECTED in closing) is rejected
    assert (NOT_REJECTED in closing) is (not rejected)
    labels = {target["label"] for claim in projection["claims"] for target in claim["targets"]}
    assert labels == {"Results", PRIMARY_RESULT_HEADINGS_BY_FAMILY["survival"]}


def test_a_heading_the_plan_does_not_require_still_fails_closed(tmp_path):
    context, authority = sealed_survival(tmp_path)
    plan = bound_survival_plan(context, ScientificRuntimeAuthorities(trajectory=None, current_case=authority))
    summary = run_signed_suite(authority, synthetic_survival_rows(), tmp_path / "out")
    for claim in summary["reportable_survival_results"]["manuscript_projection"]["claims"]:
        for target in claim["targets"]:
            if target["kind"] == "markdown_heading":
                target["label"] = "Primary association"

    with pytest.raises(ManuscriptProjectionError, match="target is absent: markdown_heading:Primary association"):
        project_owner_issued_manuscript_claims(
            _writer_scaffold(required_result_subsections(plan)), per_step_records=[_record(summary)],
        )
