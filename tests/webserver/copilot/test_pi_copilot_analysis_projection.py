"""Analysis-stage visibility contracts for Pi Copilot job projections."""

from easyicu.webserver.pi_copilot.projections import project_job
from easyicu.webserver.agent_pipeline_runs import _gate_from_axes, _readiness_axes


def test_optional_render_failure_keeps_results_and_limitation_visible(tmp_path):
    import json

    limitation = {"step_id": "presentation", "status": "failed", "message": "可选图表未完成，分析结果仍可查看。"}
    (tmp_path / "run_status.json").write_text(json.dumps({"gates": {
        "execution_complete": True, "analysis_validated": True,
        "evidence_complete": False, "numeric_verified": False,
        "optional_render_failures": [limitation],
    }}))
    gate = _gate_from_axes(_readiness_axes(tmp_path), pending=False)
    projection = project_job({"id": "synthetic-job", "kind": "agent-run", "status": "done", "result": {
        "run_id": "synthetic-run", "gate": gate,
        "artifacts": [{"name": "result_tables.json", "sha256": "a" * 64}],
    }})
    assert projection["analysis_results_available"] is True
    assert projection["artifact_refs"][0]["artifact"] == "result_tables.json"
    assert projection["limitations"] == [{"step_id": "presentation", "message": limitation["message"]}]
    assert projection["reportable"] is False


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
