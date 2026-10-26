"""Deterministic append references carried by the normal durable job dispatcher.

There is no provider task to recover: the append owner executes in the job's
process under its file lock, without launching a host child process. An
interrupted execution is refused; only a recorded approval before execution
can continue. Dispatch and study admission remain with the shared owners.
"""
from pathlib import Path

from . import study_contexts
from .research_append_control import check_cancel, stop_append
from .research_append_run import append_run_for_id

SCHEMA = "easyicu.result-append-dispatch/1"


def stage_append_approval(*, run_id, study, reviewer, note):
    """Retain the submitted human decision before the durable queue accepts it."""
    from .research_append_run import approve_append_run

    child = append_run_for_id(run_id)
    if child is None or child.record["study"]["id"] != study["id"]:
        raise ValueError("The addition belongs to another study.")
    return approve_append_run(child.wrapper, decision="approved", reviewer=reviewer, note=note,
                              current_study=study, record_only=True)


def prepare_append_descriptor(*, study, project_root, run_id):
    from .research_auto_resume import _owned_wrapper

    child = append_run_for_id(run_id)
    if child is None or child.record["study"]["id"] != study["id"]:
        raise ValueError("The addition belongs to another study.")
    descriptor = dict(schema_version=SCHEMA, study_context_id=study["id"], project_root=str(Path(project_root).resolve()),
        scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(study),
        execution_resume_source_run_id=run_id, append_record_sha256=child.record["record_sha256"], external_llm_opt_in=True)
    _owned_wrapper(descriptor, {"wrapper_dir": str(child.wrapper)})
    return descriptor


def recover_append_runner(descriptor, job):
    from . import settings
    from .research_auto_resume import _owned_wrapper

    def checked(bound_job):
        check_cancel(bound_job)
        if (descriptor.get("schema_version") != SCHEMA or descriptor.get("external_llm_opt_in") is not True
                or settings.load_settings().get("ai_enabled") is not True):
            raise ValueError("The addition no longer has permission to run.")
        current = study_contexts.get_context(descriptor["study_context_id"])
        child = append_run_for_id(descriptor["execution_resume_source_run_id"])
        if (current is None or child is None or child.record["study"]["id"] != current["id"]
                or child.record["record_sha256"] != descriptor["append_record_sha256"]
                or study_contexts.scientific_configuration_sha256(current) != descriptor["scientific_configuration_sha256"]):
            raise ValueError("The saved addition or study changed before restart.")
        _owned_wrapper(descriptor, {"wrapper_dir": str(child.wrapper)})
        if child.checkpoint.state == "executing":
            # A fit interrupted after it started must never be run twice.
            from .agent_review_recovery import _locked
            from .research_append_run import load_append_run

            with _locked(child.wrapper / "append.lock"):
                child = load_append_run(child.wrapper)
                if child.checkpoint.state == "executing":
                    stop_append(child, code="result_append_execution_failed")
            raise ValueError("An interrupted addition needs a new review before another execution.")
        return current

    checked(job)

    def resume(bound_job):
        from .agent_pipeline_runs import resume_research_pipeline

        return resume_research_pipeline(run_id=descriptor["execution_resume_source_run_id"],
            study_context_id=descriptor["study_context_id"], decision="approved", reviewer="", note="", job=bound_job,
            current_study_context=checked(bound_job), require_recorded_decision=True)

    return resume
