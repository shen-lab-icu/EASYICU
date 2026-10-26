"""Run the normal extraction transaction inline in a sequential batch slot."""
from pathlib import Path

from . import dataio, study_contexts
from .pi_copilot.extraction_handoff import submit_study_extraction
from .research_multi_contracts import MultiDatabaseError


def prepare_source(study, source, job):
    from .routes.jobs import submit_extraction

    runners = []

    def capture(kind, runner):
        # The extraction submission owner finishes its ordinary revision/source
        # checks and handoff before opening the runner's start gate. The batch
        # then runs it in this slot, with no thread, parallel job or poller.
        runners.append(runner)
        return job

    registered = source if dataio.prepared_export_manifest_path(Path(source["path"])) is not None else None
    transaction = submit_study_extraction(study=study, registered_source=registered,
        submit=lambda body: submit_extraction(body, submit=capture))
    if transaction.reused:
        return study, source
    if len(runners) != 1:
        raise MultiDatabaseError("multi_database_preparation_interrupted")
    result = runners[0](job)
    if job.is_cancel_requested():
        raise MultiDatabaseError("multi_database_cancelled")
    refreshed = study_contexts.get_context(study["id"])
    selected = (refreshed or {}).get("data_source") or {}
    if not result.get("study_context_rebound") or not selected.get("path"):
        raise MultiDatabaseError("multi_database_extraction_unbound")
    dataio.validate_research_pipeline_source(selected["path"], database=selected["database"])
    return refreshed, selected
