"""Read-only projection of a pending preparation request for its bound run.

This is deliberately not an authority or a producer. Until the producer can
prove exactly what it consumes, a request is shown without an approve action.
"""
from pathlib import Path
from typing import Any, Mapping
from datetime import datetime, timezone
import json

from pydantic import ValidationError

from easyicu.research_agent.authority.filesystem import AnchoredDirectory, AuthorityFilesystemError
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_sha256


def run_artifact_disclosure_safe_scope(scope: Mapping[str, Any]) -> bool:
    from easyicu.webserver.run_artifact_disclosure import scan_browser_projection
    return bool(scan_browser_projection(dict(scope)).get("passed"))


def _candidate_projection(*, run: Mapping[str, Any], study_context_id: str,
                          checkpoint: RawPreparationCheckpoint, scope: Mapping[str, Any]) -> dict[str, Any] | None:
    """Project metadata without granting preparation or analysis.

    Source identity verification may hash the full raw patient file.
    """
    from easyicu.webserver import agent_pipeline_runs, first_observed_preparation, run_artifact_disclosure, study_contexts
    from easyicu.webserver.raw_preparation_dependencies import (
        _declared_typed_export_columns, declared_native_export_inventory, first_observed_candidate_preparation_preview,
    )

    try:
        pending, study = agent_pipeline_runs.raw_preparation_review_owner(run["run_id"], study_context_id)
        if Path(pending.run_dir) != Path(str(run["project_dir"])):
            return None
        expected_study = study_contexts.scientific_configuration_sha256(study)
        current = study_contexts.get_context(study_context_id)
        if not current or study_contexts.scientific_configuration_sha256(current) != expected_study:
            return None
        preview = first_observed_candidate_preparation_preview(
            run_dir=Path(pending.run_dir), pending=pending, checkpoint=checkpoint)
        try:
            verified_scope, export_path = first_observed_preparation._host_inputs(
                run_dir=Path(pending.run_dir), request=checkpoint.request)
            if verified_scope != dict(scope):
                raise ValueError("candidate_preparation_scope_changed")
            preview["declared_export_inventory"] = declared_native_export_inventory(
                export_path=export_path, expected_manifest_sha256=scope["export_manifest_sha256"],
                expected_database=scope["database"])
            columns = _declared_typed_export_columns(
                export_path=export_path, inventory=preview["declared_export_inventory"])
            dependencies = preview["declared_plan_dependencies"]
            available = {column["source_concept"] for item in columns["files"]
                         for column in item["declared_columns"] if column["role"] in {"value", "event_status"}}
            missing = sorted(set(dependencies["required_source_concepts"]) - available)
            candidate_scope = {"schema_version": "easyicu.candidate-research-read-scope/1",
                "first_observed_scope": dict(scope), "plan_dependencies": dependencies,
                "population_needs": preview["population_needs"], "context_windows": preview["context_windows"],
                "manifest_inventory": preview["declared_export_inventory"], "typed_columns": columns,
                "undeclared_source_concepts": missing,
                "blockers": sorted(set(dependencies["blockers"] + [
                    "candidate_reader_projection_unverified", "candidate_host_action_authority_unavailable",
                    *(["candidate_required_source_concepts_unavailable"] if missing else []),
                ])), "complete": False, "preparation_authorized": False, "analysis_authorized": False}
            preview["candidate_read_scope_sha256"] = canonical_sha256(candidate_scope)
            preview["candidate_read_scope"] = candidate_scope
        except (AuthorityFilesystemError, OSError, KeyError, ValueError, TypeError, RuntimeError):
            pass  # No roster is safer than an invented or partial roster.
        current = study_contexts.get_context(study_context_id)
        if (current and study_contexts.scientific_configuration_sha256(current) == expected_study
                and run_artifact_disclosure.scan_browser_projection(preview).get("passed")):
            return preview
    except (AuthorityFilesystemError, OSError, KeyError, ValueError, TypeError, RuntimeError):
        pass
    return None


def pending_preparation_projection(run: Mapping[str, Any] | None, *, study_context_id: str = "") -> Mapping[str, Any] | None:
    if not run or not run.get("run_id") or not run.get("project_dir"):
        return None
    try:
        with AnchoredDirectory.open(Path(str(run["project_dir"]))) as directory:
            names = ("raw_preparation_checkpoint.json", "sic_raw_preparation_checkpoint.json")
            selected = [name for name in names if not directory.is_absent(name)]
            # Conflicting requests cannot be resolved by choosing one.
            if len(selected) != 1:
                return None
            payload = directory.read_bytes(selected[0], max_bytes=1024 * 1024)
            directory.assert_still_selected()
        checkpoint = RawPreparationCheckpoint.model_validate_json(payload)
        if checkpoint.state == "approved" and study_context_id:
            scope = checkpoint.request.payload.get("first_observed_icu_preparation_scope")
            if isinstance(scope, Mapping) and scope.get("adapter_id") == "eicu-demo-first-observed-single-hospital-adult/1":
                result = dict(_approved_first_observed_projection(checkpoint, scope, run=run, study_context_id=study_context_id))
                if result.get("status") == "first_observed_complete":
                    preview = _candidate_projection(run=run, study_context_id=study_context_id, checkpoint=checkpoint, scope=scope)
                    if preview is not None and run_artifact_disclosure_safe_scope(scope):
                        result.update(scope=dict(scope), candidate_preparation_preview=preview)
                return result
        if checkpoint.state != "pending":
            return None
        expires = datetime.fromisoformat(checkpoint.expires_at.replace("Z", "+00:00"))
        if expires.tzinfo is None or datetime.now(timezone.utc) >= expires:
            return None
        scope = [value for key, value in checkpoint.request.payload.items() if key != "plan_review_authority"]
        if len(scope) != 1 or not isinstance(scope[0], Mapping):
            return None
        result = {"run_id": str(run["run_id"]), "checkpoint_sha256": checkpoint.checkpoint_sha256,
                  "summary": str(checkpoint.request.summary)[:4000],
                  "scope": dict(scope[0]), "status": "awaiting_consumption_closure", "approvable": False}
        # A pending first-observed checkpoint grants no patient-file reads.
        # The saved disclosure remains visible; discovery waits for approval.
        return result
    except (AuthorityFilesystemError, OSError, ValidationError, ValueError, TypeError):
        return None


def _approved_first_observed_projection(checkpoint: RawPreparationCheckpoint, scope: Mapping[str, Any],
                                        *, run: Mapping[str, Any], study_context_id: str) -> Mapping[str, Any]:
    """Project verified status only, without creating a job or reading patient rows."""
    from easyicu import state_paths
    from easyicu.webserver.durable_research import current_service
    from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding
    from easyicu.webserver.raw_preparation_durable import inspect_admitted_first_observed

    unavailable = {"status": "first_observed_unavailable", "analysis_authorized": False}
    body = checkpoint.model_dump(mode="json", exclude={"checkpoint_sha256"})
    body.update(state="pending", decision=None)
    nonce = canonical_sha256({"run_id": str(run["run_id"]), "study_context_id": study_context_id,
                              "checkpoint_sha256": canonical_sha256(body)})
    try:
        service = current_service()
        if service is None or service.manager._journal is None:
            return unavailable
        root = state_paths.state_root() / "private-preparation-decisions"
        with AnchoredDirectory.open(root) as directory:
            receipt = json.loads(directory.read_bytes(nonce + ".json", max_bytes=64 * 1024))
            directory.assert_still_selected()
        binding = RawPreparationHostBinding(project_id=receipt["project_id"], session_id=receipt["session_id"],
            study_context_id=study_context_id, run_id=str(run["run_id"]), nonce=nonce)
        job_id = receipt["job_id"]
        command = service.store.get(job_id)
        if command is None:
            return unavailable
        job = service.manager.get(job_id)
        if job is not None and job.snapshot().get("status") == "done":
            from easyicu.databases.research_plan_requirements import _constraints
            from easyicu.research_agent.research_context.export_selection import SourceIdentity
            from easyicu.research_agent.research_context.typed import parse_research_context_json
            from easyicu.webserver.first_observed_run_guard import resolve_host_prepared_first_observed_coordinate

            with AnchoredDirectory.open(Path(str(run["project_dir"]))) as directory:
                raw_context = directory.read_bytes("research_context.json", max_bytes=5 * 1024 * 1024)
                directory.assert_still_selected()
            context = parse_research_context_json(raw_context.decode("utf-8"))
            source = SourceIdentity.from_record(_constraints(context).get("source_identity"))
            resolve_host_prepared_first_observed_coordinate(
                preparation_run_id=str(run["run_id"]), study_context_id=study_context_id,
                project_id=binding.project_id, session_id=binding.session_id, source_identity=source,
                criterion=scope["criterion"], export_manifest_sha256=scope["export_manifest_sha256"])
            return {"status": "first_observed_complete", "analysis_authorized": False}
        result = inspect_admitted_first_observed(job_id=job_id, job=job,
            checkpoint=checkpoint, binding=binding, scope=scope, command=command,
            store_path=service.store.path.absolute(), journal_path=service.manager._journal.path.absolute())
        return {"status": "first_observed_" + result["preparation_status"],
                "analysis_authorized": False}
    except (AuthorityFilesystemError, OSError, ValidationError, KeyError, ValueError, TypeError, RuntimeError):
        return unavailable
