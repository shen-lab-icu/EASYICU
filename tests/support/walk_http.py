"""Synthetic HTTP snapshots for the walk tool; no server or patient data."""

from tools.walk.runner import Options


class SyntheticAPI:
    """Small, row-poisoned snapshots matching the public HTTP adapters."""

    def __init__(self):
        self.calls = []
        self.sessions = {}
        self.approved = False
        self.question = ""

    def request(self, method, path, body=None, *, params=None, timeout=30):
        self.calls.append((method, path, body, params))
        if path == "/api/workspaces/registry":
            return {"sources": [{"id": "demo", "label": "Synthetic", "database": "sic", "ok": True}]}
        if path == "/api/capabilities":
            return {"capabilities": {"method_skills": {"items": [
                {"id": "simple-study", "layer": "research_workflow", "action_ids": ["summary"], "enabled": True},
            ], "components": [], "planned_method_count": 1}}}
        if path == "/api/capabilities/research-workflows":
            return {"items": [{"id": "simple-study", "steps": [{"module_id": "summary"}]}]}
        if path.endswith("/simple-study/package"):
            return {"files": [{"path": "examples/public_demo.md", "content": (
                "# Worked example: demo\n\nPublic source: demo.\n\n"
                "Describe synthetic ICU admission measurements?\n\n## Plan proposed for your review\n\nAdults.\n"
            )}]}
        if path == "/api/guided/drafts":
            return {"ok": True, "draft": {"id": "project-walk", "title": body["title"]}}
        if path.endswith("/projects/initialize"):
            return {"ok": True}
        if path == "/api/copilot/pi/sessions":
            sid = f"session-{len(self.sessions)}"
            self.sessions[sid] = {"session_id": sid, "project_id": "project-walk", "binding": {"study_context_id": "study-walk"}}
            return {"ok": True, "session": self.sessions[sid]}
        if path.endswith("/mentions/resolve"):
            return {"reference": {"kind": "database", "source": "registered_source", "id": "demo", "digest": "a" * 64}}
        if path.endswith("/message"):
            self.question = body["message"]
            return {"ok": True}
        if path.endswith("/data-source-authorization"):
            assert body["source_id"] == "demo"
            return {"ok": True}
        if path.endswith("/workflow"):
            return {
                "workflow": {"current_stage": "plan", "next_action_code": "plan_review_required",
                    "plan_execution_ready": True, "plan_review_summary": {"status": "pass", "run_id": "run-walk"}},
                "host_decisions": {"plan_review": {"family": "plan_review", "run_id": "run-walk",
                    "scientific_configuration_sha256": "a" * 64, "review_authority_sha256": "b" * 64}},
                "latest_run": {"present": True, "run_id": "run-walk", "gate_status": "pass",
                    "artifact_refs": [{"artifact": "report.md"}], "execution_complete": self.approved},
                "runs": [{"run_id": "run-walk", "plan_available": True}],
            }
        if path == "/api/jobs/agent-run-review":
            self.approved = True
            return {"job_id": "review-job"}
        if path == "/api/jobs/review-job":
            return {"status": "done", "result": {"patient_id": "NEVER-SAVE-THIS"}}
        if path.startswith("/api/copilot/pi/sessions/"):
            sid = path.rsplit("/", 1)[-1]
            return {"session": {**self.sessions[sid], "transcript": [{"text": "patient_id=NEVER-SAVE-THIS"}]}}
        raise AssertionError((method, path))


def walk_options(tmp_path, **kwargs):
    return Options(source="demo", out=tmp_path / "walk", only=("workflows",), max_workflows=2, **kwargs)


class PhaseAPI(SyntheticAPI):
    def __init__(self):
        super().__init__()
        self.drafts = 0
        self.executed = set()
        self.deleted = set()

    def request(self, method, path, body=None, *, params=None, timeout=30):
        if path == "/api/health":
            return {"status": "ok"}
        if path == "/api/settings":
            return {"about": {"version": "0.1.0", "commit": "123abcdef", "runner_image_tag": "easyicu-runner:v1"},
                    "api_key": "NEVER-SAVE-THIS"}
        if path == "/api/copilot/pi/status":
            return {"runtime": {"status": "ready", "shell_ready": True, "configuration": {"connection_verified": True}}}
        if path == "/api/catalog":
            return {"groupConcepts": {"basic": ["age"], "labs": ["lact", "bili"]}}
        if path == "/api/data-map":
            return {"basis": "packaged_metadata", "concept_total": 3, "concept_next_offset": None,
                    "databases": [{"database": "sic", "source_total": 1, "studies": [], "concepts": [
                        {"concept": "age", "status": "substituted"}, {"concept": "lact", "status": "available"},
                        {"concept": "bili", "status": "missing"}]}]}
        if path.endswith("/six-database-cohort-extraction/projection"):
            return {"databases": [{"database": "sic", "modules": {f"module_{i}": {} for i in range(19)}}]}
        if path == "/api/ideas/mine":
            assert body["metadata_only"] is True and body["allow_network"] is False
            return {"idea_ledger": [{"prior_art": {"status": "not_checked"}, "feasibility": {"tier": "T1_metadata_only"}}]}
        if path.endswith("/artifacts/literature_evidence.json"):
            return {"payload": {"status": "searched", "search": {"search_conducted": True,
                "prisma": {"identified": 4, "screened": 3, "included": 2}, "sources_returning": ["pubmed"]},
                "review": {"role_counts": {"direct_comparator": 1}, "shortfalls": [{"role": "design_analogue"}]},
                "patient_id": "NEVER-SAVE-THIS", "citations": [{"abstract": "NEVER-SAVE-THIS"}]}}
        if path.endswith("/skills"):
            return {"items": [{"source": "builtin", "id": "simple-study", "version": "1", "enabled": True}]}
        if path.endswith("/draft-skills"):
            assert body["attached_skills"][0]["id"] == "simple-study"
            return {"attached_skills": body["attached_skills"]}
        if path.endswith("/mentions"):
            return {"items": [{"kind": "file", "id": "walk-note"}]}
        if path.endswith("/mentions/resolve") and body["kind"] == "file":
            return {"reference": {"kind": "file", "id": "walk-note", "digest": "f" * 64}}
        if path.endswith("/draft-mentions"):
            assert body["mentions"][0]["kind"] == "file"
            return {"mentions": body["mentions"]}
        if path.endswith(("/rename", "/delete")):
            sid = path.split("/")[-2]
            assert self.sessions[sid]["project_id"] == body["project_id"]
            self.calls.append((method, path, body, params))
            if path.endswith("/delete"):
                assert body["confirm"] is True
                self.deleted.add(sid)
            return {"ok": True}
        result = super().request(method, path, body, params=params, timeout=timeout)
        if path == "/api/guided/drafts":
            self.drafts += 1
            result["draft"]["id"] = f"project-walk-{self.drafts}"
        if path == "/api/copilot/pi/sessions":
            session = result["session"]
            session["project_id"] = body["project_id"]
            session["binding"] = {"study_context_id": "study-" + body["project_id"], "study_revision": 1}
        if path.endswith("/workflow"):
            pid = path.split("/")[-2]
            rid = "run-" + pid
            result["project_id"] = pid
            result["workflow"]["plan_review_summary"]["run_id"] = rid
            result["host_decisions"]["plan_review"]["run_id"] = rid
            result["latest_run"]["run_id"] = rid
            result["latest_run"]["execution_complete"] = rid in self.executed
        if path == "/api/jobs/agent-run-review":
            self.executed.add(body["run_id"])
        return result


