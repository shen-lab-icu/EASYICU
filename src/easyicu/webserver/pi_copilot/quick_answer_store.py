"""Private quick-answer proposals and public conversation attachments."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.research_agent.research_context.typed import parse_research_context
from easyicu.webserver.quick_answer.contracts import QuickAnswer, QuickProposal, QuickReading, QuickStop
from easyicu.webserver.quick_answer.evidence import project_quick_answer
from easyicu.webserver.quick_answer.source import PreparedSource

from .contracts import PiCopilotError
from .locking import exclusive_file_lock


class QuickQuestionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reading: QuickReading


def error(code: str) -> PiCopilotError:
    return PiCopilotError(code, code, status_code=409)


def source_record(source: PreparedSource | None) -> dict | None:
    if source is None:
        return None
    return {
        "path": str(source.path), "context": source.context.model_dump(mode="json"),
        "identity": source.identity.to_dict(), "unit_column": source.unit_column,
        "patient_column": source.patient_column, "analysis_unit": source.analysis_unit,
        "data_sha256": source.data_sha256, "binding": source.binding,
    }


def stored_source(row: dict) -> PreparedSource:
    # Preserve the original digests; rebinding here would bless changed bytes.
    return PreparedSource(**{
        **row, "path": Path(row["path"]), "context": parse_research_context(row["context"]),
        "identity": SourceIdentity.from_record(row["identity"]),
    })


def public_attachment(row: dict) -> dict:
    return {
        "quick_answer_id": row["quick_answer_id"], "message_job_id": row["message_job_id"],
        "proposal": QuickProposal.model_validate(row["proposal"]).model_dump(mode="json"),
        "answer": QuickAnswer.model_validate(row["answer"]).model_dump(mode="json") if row.get("answer") else None,
        "study_context_id": row.get("handoff_study_context_id"),
    }


class QuickAnswerStore:
    def __init__(self, root: Path):
        self.root = root

    def directory(self, session_id: str) -> Path:
        return self.root / hashlib.sha256(session_id.encode()).hexdigest()

    def lock(self, session_id: str):
        return exclusive_file_lock(self.directory(session_id) / ".lock", code="quick_answer_busy")

    def path(self, session_id: str, quick_answer_id: str) -> Path:
        if not re.fullmatch(r"qa_[a-f0-9]{32}", quick_answer_id):
            raise error("quick_answer_not_found")
        return self.directory(session_id) / quick_answer_id / "record.json"

    def read(self, session_id: str, project_id: str, quick_answer_id: str) -> dict:
        path = self.path(session_id, quick_answer_id)
        try:
            if path.is_symlink() or path.stat().st_size > 2 * 1024 * 1024:
                raise error("quick_answer_invalid")
            row = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise error("quick_answer_not_found") from exc
        if (row.get("session_id"), row.get("project_id"), row.get("quick_answer_id")) != (
            session_id, project_id, quick_answer_id,
        ):
            raise error("quick_answer_scope_mismatch")
        return row

    def write(self, row: dict) -> None:
        path = self.path(row["session_id"], row["quick_answer_id"])
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(row, ensure_ascii=False, allow_nan=False))
        temporary.chmod(0o600)
        temporary.replace(path)

    def public(self, row: dict) -> dict:
        attachment = public_attachment(row)
        answer = attachment["answer"]
        if answer and answer["status"] == "ok":
            proposal = QuickProposal.model_validate(row["proposal"])
            try:
                coordinate = answer["rows"][0]["evidence"]
                verified = project_quick_answer(
                    run_dir=self.path(row["session_id"], row["quick_answer_id"]).parent / "execution",
                    proposal=proposal, evidence_id=coordinate["evidence_id"], sha256=coordinate["sha256"],
                )
                if verified.model_dump(mode="json") != answer:
                    raise QuickStop("quick_evidence_binding_failed")
            except (QuickStop, OSError, ValueError, IndexError):
                verified = QuickAnswer(status="stopped", reason="quick_evidence_binding_failed",
                                       source=proposal.expand.source, expand=proposal.expand)
            attachment["answer"] = verified.model_dump(mode="json")
        return attachment

    def attachments(self, session_id: str, project_id: str, message_ids: set[str]) -> list[dict]:
        # Read only attachments on the active replay page; edited branches do
        # not regain approval authority on reload.
        rows = []
        for path in self.directory(session_id).glob("qa_*/record.json"):
            row = self.read(session_id, project_id, path.parent.name)
            if row["message_job_id"] in message_ids or row["quick_answer_id"] in message_ids:
                rows.append(self.public(row))
        return rows
