"""A prediction needs patient groups its source can give.

A static prediction splits development and validation stays by patient
(``contracts.patient_grouping_need``).  While a patient's repeated ICU stays
are possible, a source that cannot group patients left that step to fail at
execution, after approval.  The review now refuses such a plan before
approval, reading the status the planning context states of its source; a
context that states none has only its own grouping authority.  Synthetic
contexts only.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.contracts.patient_grouping_need import (
    PATIENT_GROUPING_AUTHORITY_ERROR_KEY,
    PATIENT_GROUPING_STATUS_KEY,
)
from easyicu.research_agent.intake.materialized_metadata import (
    FIRST_ICU_STAY_RESTRICTION_SCHEMA,
)
from easyicu.research_agent.planning.dependence_authority import repeat_units_possible
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    plan_revision_blocker_codes,
    prediction_patient_grouping_findings,
)
from easyicu.research_agent.schema import ResearchContext

from .family_spec_fixtures import (
    _prediction_context,
    _prediction_payload,
    _request,
    _run,
)

CODE = "PREDICTION_PATIENT_GROUPING_UNAVAILABLE"
NOT_CARRIED = "PREDICTION_PATIENT_GROUPING_NOT_CARRIED_BY_TRAJECTORY"
FEATURES = ["age", "sex", "hr_max", "lactate_max", "map_min"]


def _template_plan(context: ResearchContext):
    request = _request(context, cohort_mode=None)
    _llm, result = _run(
        context,
        [json.dumps(_prediction_payload(request, features=FEATURES))],
        required_primary_cohort_selection_mode=None,
    )
    return result.output


def _with_provenance(context: ResearchContext, **entries) -> ResearchContext:
    provenance = {**(context.cohort.provenance or {}), **entries}
    return context.model_copy(
        update={"cohort": context.cohort.model_copy(update={"provenance": provenance})}
    )


def _stated(context: ResearchContext, status: str, error: str | None = None) -> ResearchContext:
    entries = {PATIENT_GROUPING_STATUS_KEY: status}
    if error is not None:
        entries[PATIENT_GROUPING_AUTHORITY_ERROR_KEY] = error
    return _with_provenance(context, **entries)


@pytest.mark.parametrize(
    ("status", "error", "code", "reason", "remedy"),
    [
        (
            "source_has_none", None, CODE,
            "the host knows no verified patient grouping for this source",
            "Register the source's patient identity",
        ),
        (
            "not_carried_by_trajectory", None, NOT_CARRIED,
            "not carried by the design read from each stay's long trajectory",
            "Plan the study without a design read from each stay's long trajectory",
        ),
        (
            "authority_invalid", "patient_grouping_mapping_digest_mismatch", CODE,
            "fails its checks (patient_grouping_mapping_digest_mismatch)",
            "Repair the source's patient identity authority",
        ),
    ],
)
def test_a_source_that_cannot_group_patients_is_refused(status, error, code, reason, remedy) -> None:
    context = _prediction_context()
    plan = _template_plan(context)

    [finding] = prediction_patient_grouping_findings(_stated(context, status, error), plan)

    # A grouping the declared design does not carry has its own code: its
    # remedy is the design, not the source.
    assert (finding.code, finding.severity) == (code, "blocker")
    assert finding.remediation_route == "runtime_capability"
    assert "Step 'primary_performance' splits or clusters stays by patient" in finding.message
    assert reason in finding.message
    assert finding.remediation.startswith(remedy)
    assert "first ICU stay" in finding.remediation


@pytest.mark.parametrize("status", ["bound", "available_unbound"])
def test_a_source_that_can_group_patients_passes(status) -> None:
    context = _prediction_context()

    assert prediction_patient_grouping_findings(
        _stated(context, status), _template_plan(context)
    ) == []


def test_a_context_that_states_no_status_reads_its_own_grouping() -> None:
    context = _prediction_context()
    plan = _template_plan(context)

    [finding] = prediction_patient_grouping_findings(context, plan)
    assert "the planning context binds no patient grouping" in finding.message
    grouped = _with_provenance(
        context.model_copy(
            update={"cohort": context.cohort.model_copy(update={"id_columns": ["patient_stay_id"]})}
        ),
        replacement_row_identity={
            "output_identity_column": "patient_stay_id",
            "patient_group_derivation": {"algorithm": "prefix_before_:s", "delimiter": ":s"},
            "mapping_file_sha256": "a" * 64,
        },
    )
    assert prediction_patient_grouping_findings(grouped, plan) == []
    # Repeated stays the counts show are grouped by that same authority.
    repeated = grouped.model_copy(
        update={"cohort": grouped.cohort.model_copy(update={"n_patients": 5, "n_stays": 8})}
    )
    assert repeat_units_possible(repeated)
    assert prediction_patient_grouping_findings(repeated, plan) == []


def test_first_icu_stays_need_no_grouping() -> None:
    context = _prediction_context()
    first_stays = _with_provenance(
        _stated(context, "source_has_none"),
        first_icu_stay_restriction={
            "schema_version": FIRST_ICU_STAY_RESTRICTION_SCHEMA,
            "coordinate_sha256": "c" * 64,
        },
    )

    assert prediction_patient_grouping_findings(first_stays, _template_plan(context)) == []


def test_a_plan_without_a_step_that_needs_groups_is_not_asked() -> None:
    context = _prediction_context()
    plan = _template_plan(context)
    unsplit = plan.model_copy(
        update={"steps": [step for step in plan.steps if step.step_id != "primary_performance"]}
    )

    assert prediction_patient_grouping_findings(_stated(context, "source_has_none"), unsplit) == []


def test_the_review_refuses_approval_and_a_futile_plan_retry() -> None:
    context = _stated(_prediction_context(), "source_has_none")

    review = build_plan_scientific_review(context=context, plan=_template_plan(context))

    assert CODE in {finding.code for finding in review.findings}
    assert not review.approval_allowed
    # Another Planner turn cannot give the source patient groups.
    assert CODE in plan_revision_blocker_codes(list(review.findings))
