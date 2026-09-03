"""Analysis-stage visibility contracts for Pi Copilot job projections."""

from easyicu.webserver.pi_copilot.projections import project_job


def test_validated_analysis_only_results_remain_visible_before_manuscript_numeric_gate() -> None:
    projected = project_job(
        {
            "id": "job-analysis-only",
            "kind": "agent-run",
            "status": "done",
            "result": {
                "run_id": "run-analysis-only",
                "gate": {
                    "status": "blocked",
                    "reason": "research_agent_pipeline_failed_closed",
                    "reportable": False,
                    "checks": [
                        {"id": "execution_complete", "passed": True},
                        {"id": "analysis_validated", "passed": True},
                        {"id": "numeric_verified", "passed": False},
                        {"id": "evidence_complete", "passed": False},
                    ],
                },
                "artifacts": [
                    {"name": "result_tables.json", "sha256": "a" * 64, "bytes": 40},
                    {"name": "figure_gallery.json", "sha256": "b" * 64, "bytes": 50},
                ],
            },
        }
    )

    assert projected["analysis_results_available"] is True
    assert projected["analysis_validated"] is True
    assert projected["numeric_verified"] is False
    assert projected["reportable"] is False
    assert [row["artifact"] for row in projected["artifact_refs"]] == [
        "result_tables.json",
        "figure_gallery.json",
    ]
