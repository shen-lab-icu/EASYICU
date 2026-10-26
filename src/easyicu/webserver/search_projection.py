"""Allowlisted text projections from host metadata, with no table cell indexing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator

from .search_contracts import SearchDocument, text_value
from .pi_copilot.user_visible_text import project_user_turn_text, sanitize_user_visible_text


def messages(rows: list, project: str, conversation: str, source: str) -> Iterator[SearchDocument]:
    occurrences: dict[str, int] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("role") not in {"user", "assistant"}:
            continue
        text = text_value(row.get("text"))
        content = row.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text += "\n".join(text_value(part.get("text")) for part in content
                              if isinstance(part, dict) and part.get("type") == "text")
        reply = row.get("reply")
        if isinstance(reply, dict):
            text += "\n" + "\n".join(text_value(reply.get(lang)) for lang in ("en", "zh"))
        for marker in ("\n\n[EASYICU_INTERNAL_RESPONSE_LANGUAGE_V1]\n",
                       "\n\n[EASYICU_CURRENT_TURN_OWNER_CONTEXT_V1]\n", "\n\n[EASYICU_HOST_TRANSITION_V1]\n"):
            text = text.split(marker, 1)[0]
        text = (sanitize_user_visible_text(text) if row["role"] == "assistant" else project_user_turn_text(text))
        if not text.strip():
            continue
        identity = text_value(row.get("entry_id") or row.get("id"))[:160]
        target = dict(kind="conversation", project_id=project, conversation_id=conversation, source=source)
        if not identity:
            # Old Guided/Page Guide writers have no message ids. The fingerprint
            # survives truncation of preceding messages; occurrence separates twins.
            digest = hashlib.sha256(json.dumps([row.get("role"), row.get("created_at"), text],
                                               ensure_ascii=False).encode()).hexdigest()[:24]
            occurrence = occurrences.get(digest, 0)
            occurrences[digest] = occurrence + 1
            identity = f"legacy-{digest}-{occurrence}"
            target.update(message_index=index, created_at=row.get("created_at"))
        target["message_id"] = identity
        yield SearchDocument(project, "message", "", text_value(text), target)


def transcript(entries: list, project: str, conversation: str) -> Iterator[SearchDocument]:
    # Pi persists a tree. Only the current leaf's branch can be opened by the UI.
    by_id = {row["id"]: row for row in entries if isinstance(row.get("id"), str) and row.get("type") != "session"}
    branch, seen = [], set()
    leaf = next(reversed(by_id), None)
    while isinstance(leaf, str) and leaf in by_id and leaf not in seen:
        seen.add(leaf)
        row = by_id[leaf]
        if row.get("type") == "message" and isinstance(row.get("message"), dict):
            branch.append({**row["message"], "entry_id": leaf})
        leaf = row.get("parentId")
    yield from messages(list(reversed(branch)), project, conversation, "pi")


def run_documents(payload: dict, project: str, run: str, name: str, wrapper: str) -> Iterator[SearchDocument]:
    target = dict(kind="run", project_id=project, run_id=run, run_ref=wrapper)
    if name == "agent_plan.json":
        parts = [text_value(payload.get(key)) for key in ("research_question", "summary", "title")]
        steps = payload.get("steps")
        parts.extend(text_value(step.get("intent")) for step in (steps[:200] if isinstance(steps, list) else [])
                     if isinstance(step, dict))
        yield SearchDocument(project, "plan", text_value(payload.get("title")), text_value("\n".join(parts)),
                             {**target, "panel": "plan"})
    elif name == "run_context.json":
        text = "\n".join(text_value(payload.get(key)) for key in ("question", "answer", "summary"))
        yield SearchDocument(project, "run", "", text_value(text), {**target, "panel": "results"})
    elif name == "manuscript_draft.json":
        draft = payload.get("draft") if isinstance(payload.get("draft"), dict) else payload
        text = "\n".join(text_value(draft.get(key)) for key in ("answer", "abstract", "results", "summary", "markdown_preview"))
        yield SearchDocument(project, "answer", text_value(draft.get("title")), text_value(text),
                             {**target, "panel": "results"})
    elif name == "result_tables.json":
        tables = payload.get("tables")
        for index, table in enumerate(tables[:200] if isinstance(tables, list) else []):
            if not isinstance(table, dict):
                continue
            # The host's result-table projection contains aggregate tables only.
            # Reject unexpected identifier columns even in a corrupt projection.
            headers = table.get("headers") or []
            if not isinstance(headers, list):
                continue
            if any(str(h).casefold() in {"stay_id", "subject_id", "hadm_id", "patient_id"} for h in headers):
                continue
            filename = text_value(table.get("name"))
            if not filename or "/" in filename or "\\" in filename:
                continue
            title = text_value(table.get("title") or table.get("label") or table.get("description"))
            file_id = "result_tables.json#" + str(table.get("evidence_id") or index)[:160]
            yield SearchDocument(project, "file", title, filename,
                                 {**target, "kind": "file", "file_id": file_id, "artifact": name})
