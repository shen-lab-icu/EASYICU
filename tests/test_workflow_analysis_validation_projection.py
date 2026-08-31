from __future__ import annotations

from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot


def test_completed_numeric_outputs_project_validation_repair_without_replanning() -> None:
    study = {
        "id": "study-validation-repair",
        "revision": 7,
        "question": "Is lactate associated with in-hospital mortality?",
        "data_source": {"path": "/private/prepared/source", "database": "miiv"},
        "cohort": {"label": "ICU patients"},
        "modules": [],
        "outcome": "",
        "analysis_goal": "",
        "time_window": {},
        "export_format": "",
        "confirmations": {},
    }
    digest = study_context_owner.scientific_configuration_sha256(study)

    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=False,
        active_job=None,
        latest_run={
            "run_id": "run-analysis-validation-open",
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "gate_reason": "research_agent_pipeline_failed_closed",
            "gate_checks": {
                "execution_complete": True,
                "analysis_validated": False,
                "evidence_complete": True,
                "numeric_verified": True,
            },
            "run_status": "analysis_only",
            "scientific_configuration_sha256": digest,
            "artifact_names": [
                "agent_plan.json",
                "evidence_ledger.json",
                "result_tables.json",
                "figure_gallery.json",
                "source_run_manifest.json",
            ],
        },
    )

    by_id = {row.id: row for row in snapshot.stages}
    assert by_id["setup"].status == "complete"
    assert by_id["extraction"].status == "complete"
    assert by_id["plan"].status == "complete"
    assert by_id["analysis"].status == "review_required"
    assert by_id["analysis"].reason_code == "analysis_outputs_require_validation"
    assert by_id["interpretation"].status == "review_required"
    assert by_id["manuscript"].status == "blocked"
    assert snapshot.current_stage == "analysis"
    assert snapshot.next_action_code == "analysis_outputs_require_validation"
    assert snapshot.analysis_validation_retry_available is True
