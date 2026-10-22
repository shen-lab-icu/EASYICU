"""A question changed after its results asks for a new plan.

On the hosted pilot a clinician read the results of "in-hospital mortality
by sex", then asked in the same conversation for mortality by age group.
The study's question changed, but the workflow kept the finished run as the
current analysis and offered only its interpretation: no plan for the new
question could be started.  A completed analysis whose plan answered an
earlier configuration is now history, and the plan step offers a fresh plan,
as for a superseded plan review.  Synthetic study only.
"""

from __future__ import annotations

from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot


def _study(question: str) -> dict:
    return {
        "id": "study-changed-question",
        "revision": 3,
        "question": question,
        "data_source": {"path": "/private/prepared/source", "database": "eicu"},
        "cohort": {"label": "ICU stays"},
        "modules": [],
        "outcome": "",
        "analysis_goal": "",
        "time_window": {},
        "export_format": "",
        "confirmations": {"extraction_completed": True},
    }


_ANSWERED = _study("In-hospital mortality of men and women among all ICU stays.")


def _snapshot(study: dict):
    return build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_id": "run-sex-mortality",
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "analysis_only",
            "scientific_configuration_sha256": study_context_owner.scientific_configuration_sha256(
                _ANSWERED
            ),
            "gate_checks": {
                "execution_complete": True,
                "analysis_validated": True,
                "numeric_verified": True,
                "evidence_complete": True,
                "manuscript_ready": True,
            },
            "artifact_names": [
                "agent_plan.json",
                "evidence_ledger.json",
                "result_tables.json",
                "figure_gallery.json",
                "manuscript_draft.json",
                "source_run_manifest.json",
            ],
        },
    )


def test_the_results_of_the_question_asked_are_interpreted() -> None:
    snapshot = _snapshot(_ANSWERED)

    by_id = {row.id: row for row in snapshot.stages}
    assert snapshot.current_stage == "interpretation"
    assert by_id["analysis"].reason_code == "validated_analysis_ready"


def test_a_new_question_after_the_results_offers_a_new_plan() -> None:
    snapshot = _snapshot(_study("In-hospital mortality below and above age 65 among all ICU stays."))

    by_id = {row.id: row for row in snapshot.stages}
    assert snapshot.current_stage == "plan"
    assert (by_id["plan"].status, by_id["plan"].reason_code) == ("ready", "plan_configuration_superseded")
    assert by_id["analysis"].status == "blocked"
    assert by_id["interpretation"].status == "blocked"
    assert by_id["manuscript"].status == "blocked"
