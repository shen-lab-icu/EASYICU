"""Small synthetic persisted projects shared by search owner and HTTP tests."""

import hashlib
import json

import pytest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path))
    monkeypatch.delenv("EASYICU_HOSTED", raising=False)
    state = tmp_path / ".easyicu"
    projects = tmp_path / "easyicu/projects"
    drafts = []
    for pid, title in (("orchard", "果园研究 Orchard"), ("meadow", "Meadow study")):
        folder = projects / pid
        folder.mkdir(parents=True)
        drafts.append(dict(id=pid, title=title, project_dir=str(folder)))
    write_json(state / "webserver_guided_drafts.json", {"drafts": drafts, "sessions": []})
    transcript = state / "pi-agent/sessions/orchard.jsonl"
    transcript.parent.mkdir(parents=True)
    entries = [
        {"type": "session", "id": "sdk-orchard", "version": 3},
        {"type": "message", "id": "u1", "parentId": None,
         "message": {"role": "user", "content": [{"type": "text", "text": "比较果树生长差异"}]}},
        {"type": "message", "id": "a1", "parentId": "u1",
         "message": {"role": "assistant", "content": [
             {"type": "thinking", "thinking": "privatehiddenword"},
             {"type": "text", "text": "🌱 Growth improved in the orchard."}]}},
    ]
    transcript.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n")
    write_json(state / "pi_copilot_sessions.json", {"sessions": [
        dict(session_id="c1", project_id="orchard", title="Growth discussion", pi_session_file=str(transcript)),
    ]})
    write_json(state / "pi_project_authority.json", {"bindings": [
        dict(project_id="orchard", study_context_id="study-orchard"),
    ]})
    key = hashlib.sha256(b"study-orchard").hexdigest()[:32]
    run = state / "pi-agent/workspace" / key / "study-orchard/run_first"
    write_json(run / "run_context.json", dict(run_id="r1", study_id="study-orchard", question="Tree growth"))
    write_json(run / "agent_plan.json", dict(research_question="Tree growth", title="Seasonal design",
                                              summary="Compare spring growth", steps=[{"intent": "Estimate variation"}]))
    write_json(run / "manuscript_draft.json", {"draft": {"title": "Orchard report", "results": "Canopy increased"}})
    write_json(run / "result_tables.json", {"tables": [
        {"evidence_id": "table1", "name": "growth_estimates.csv", "title": "树木生长估计",
         "headers": ["group", "estimate"], "rows": [["rowsecretword", 12]]},
    ]})
    (run / "manuscript_scaffold.pdf").write_bytes(b"not a real PDF: file name only")
    return state, projects, run, transcript
