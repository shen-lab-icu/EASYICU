"""Small synthetic job receipts through the real conversation replay owner."""

from easyicu.webserver.jobs import Job
from easyicu.webserver.pi_copilot.projections import project_job


def archive(replay, *, job_id="job-1", project="project-1", conversation="conversation-1",
            kind="agent-run", status="done", result=None, error=None):
    replay.record_host_action(
        session_id=conversation, project_id=project, action_id=job_id,
        action_code="generate_plan", child_job_id=job_id,
    )
    job = Job(job_id, kind)
    job.finish(status, result=result, error=error)
    projected = project_job(job.snapshot())
    replay.archive_child_job(session_id=conversation, project_id=project, job=projected)
    return projected


def result_ready(run_id="run-1"):
    return {
        "run_id": run_id,
        "gate": {"status": "analysis_only", "checks": [
            {"id": "execution_complete", "passed": True},
            {"id": "analysis_validated", "passed": True},
        ]},
        "artifacts": [{"name": "result_tables.json", "sha256": "a" * 64}],
    }
