"""Read-only projection of a pending preparation request for its bound run.

This is deliberately not an authority or a producer. Until the producer can
prove exactly what it consumes, a request is shown without an approve action.
"""
from pathlib import Path
from typing import Any, Mapping
from datetime import datetime, timezone

from pydantic import ValidationError

from easyicu.research_agent.authority.filesystem import AnchoredDirectory, AuthorityFilesystemError
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint


def pending_preparation_projection(run: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
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
