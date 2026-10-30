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
                return _approved_first_observed_projection(checkpoint, scope, run=run, study_context_id=study_context_id)
        if checkpoint.state != "pending":
            return None
        expires = datetime.fromisoformat(checkpoint.expires_at.replace("Z", "+00:00"))
        if expires.tzinfo is None or datetime.now(timezone.utc) >= expires:
            return None
        scope = [value for key, value in checkpoint.request.payload.items() if key != "plan_review_authority"]
        if len(scope) != 1 or not isinstance(scope[0], Mapping):
            return None
        return {"run_id": str(run["run_id"]), "checkpoint_sha256": checkpoint.checkpoint_sha256,
                "summary": str(checkpoint.request.summary)[:4000],
                "scope": dict(scope[0]), "status": "awaiting_consumption_closure", "approvable": False}
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
        result = inspect_admitted_first_observed(job_id=job_id, job=service.manager.get(job_id),
            checkpoint=checkpoint, binding=binding, scope=scope, command=command,
            store_path=service.store.path.absolute(), journal_path=service.manager._journal.path.absolute())
        return {"status": "first_observed_" + result["preparation_status"],
                "analysis_authorized": False}
    except (AuthorityFilesystemError, OSError, ValidationError, KeyError, ValueError, TypeError, RuntimeError):
        return unavailable
