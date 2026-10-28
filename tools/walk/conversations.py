"""Conversation gates, reviewed execution and project-scoped interaction probes."""

from __future__ import annotations

import re
import uuid

from .contracts import Status, WalkStop
from .http import HTTPFailure
from .privacy import token
from .report import run_metrics


def _identity(value):
    if token(value) == "not_reported" or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,159}", value):
        raise WalkStop(Status.REFUSED, "server_identity_invalid")
    return value


def example_question(package: dict) -> str | None:
    """Read the English question from the public worked-example package."""
    files = {row.get("path"): row.get("content") for row in package.get("files", []) if isinstance(row, dict)}
    content = files.get("examples/public_demo.md")
    if not isinstance(content, str):
        return None
    prefix = content.split("## Plan proposed for your review", 1)[0]
    paragraphs = [part.strip() for part in prefix.split("\n\n") if part.strip()]
    question = next((part for part in reversed(paragraphs) if not part.startswith(("#", "Public source:"))), None)
    return question if question and len(question) <= 8000 else None


def plan_metrics(snapshot: dict) -> dict:
    workflow = snapshot.get("workflow") or {}
    review = workflow.get("plan_review_summary") or {}
    questions = review.get("authorization_questions") or []
    decision = workflow.get("plan_decision")
    pending = len(questions) + (1 if decision and not questions else 0)
    return {"plan_valid": review.get("status") in {"pass", "analysis_only", "ready_for_approval"}
            if review else None,
            "plan_review_status": token(review.get("status")),
            "pending_design_choices": pending,
            "plan_execution_ready": workflow.get("plan_execution_ready") is True,
            "current_stage": token(workflow.get("current_stage")),
            "refusal_codes": [token(code) for code in review.get("finding_codes", [])]}


class ConversationWalk:
    def _save_item(self, key, values):
        self.store.state["items"].setdefault(key, {}).update(values)
        self.store.flush()

    def _state(self, key):
        return self.store.state["items"].get(key, {})

    def _session(self, key):
        state = self._state(key)
        sid, pid = _identity(state["session_id"]), _identity(state["project_id"])
        payload = self.request("GET", f"/api/copilot/pi/sessions/{sid}", params={
            "project_id": pid, "transcript_limit": 1, "replay_limit": 1})
        session = payload.get("session") or {}
        if session.get("project_id") != pid or session.get("session_id") != sid:
            raise WalkStop(Status.REFUSED, "foreign_project_refused")
        binding = session.get("binding") or {}
        if state.get("study_context_id") and binding.get("study_context_id") != state["study_context_id"]:
            raise WalkStop(Status.REFUSED, "foreign_study_refused")
        return session

    def _session_request(self, key, method, suffix, body=None, *, params=None):
        state = self._state(key)
        sid, pid = _identity(state["session_id"]), _identity(state["project_id"])
        if body is not None:
            body = {**body, "project_id": pid}
        else:
            params = {**(params or {}), "project_id": pid}
        return self.request(method, f"/api/copilot/pi/sessions/{sid}/{suffix}", body, params=params)

    def _conversation(self, key):
        state = self._state(key)
        if state.get("deleted"):
            raise WalkStop(Status.SKIPPED, "walk_conversation_already_deleted")
        if state.get("session_id"):
            return self._session(key)
        if not state.get("project_id"):
            title = f"EasyICU walk {uuid.uuid4().hex[:12]}"
            payload = self.request("POST", "/api/guided/drafts", {
                "title": title, "folder_slug": title.lower().replace(" ", "-"),
                "data_mode": "real", "source": self.source(),
            })
            draft = payload.get("draft") or {}
            if payload.get("ok") is not True or draft.get("title") != title:
                raise WalkStop(Status.REFUSED, "walk_project_creation_refused")
            pid = _identity(draft.get("id"))
            self._save_item(key, {"project_id": pid, "project_title": title})
        state = self._state(key)
        self.request("POST", "/api/copilot/pi/projects/initialize", {
            "project_id": state["project_id"], "title": state["project_title"], "confirm_initialization": True})
        payload = self.request("POST", "/api/copilot/pi/sessions", {
            "project_id": state["project_id"], "title": "EasyICU walk example",
            "agent_mode": "research", "language": "en", "external_llm_opt_in": True})
        session = payload.get("session") or {}
        if session.get("project_id") != state["project_id"]:
            raise WalkStop(Status.REFUSED, "foreign_project_refused")
        sid = _identity(session.get("session_id"))
        study = _identity((session.get("binding") or {}).get("study_context_id"))
        self._save_item(key, {"session_id": sid, "study_context_id": study})
        return session

    def _bind_source(self, key):
        if not self._state(key).get("source_bound"):
            source = self.source()
            self._session_request(key, "POST", "data-source-authorization", {
                "action": "use_registered_source", "database": source["database"], "source_id": source["id"]})
            self._save_item(key, {"source_bound": True})
        resolved = self._session_request(key, "POST", "mentions/resolve", {
            "kind": "database", "source": "registered_source", "id": self.source()["id"]})
        ref = resolved.get("reference") or {}
        if (ref.get("kind") != "database" or ref.get("source") != "registered_source"
                or ref.get("id") != self.source()["id"]):
            raise WalkStop(Status.REFUSED, "source_mention_mismatch")
        return ref

    def _snapshot(self, key):
        pid = _identity(self._state(key)["project_id"])
        payload = self.request("GET", f"/api/copilot/pi/projects/{pid}/workflow")
        if payload.get("project_id", pid) != pid:
            raise WalkStop(Status.REFUSED, "foreign_project_refused")
        return payload

    def _wait_message(self, key):
        while True:
            session = self._session(key)
            if not session.get("active_message_job_id") and not session.get("streaming"):
                if session.get("last_turn_status") in {"failed", "cancelled", "interrupted"}:
                    raise WalkStop(Status.FAILED, "conversation_turn_failed")
                return session
            self.pause()

    def _send(self, key, message, *, mentions=None, intent=None, attachments=None):
        body = {"message": message, "allowed_actions": ["configure", "literature"], "message_origin": "typed"}
        if mentions is not None:
            body["mentions"] = mentions
        if attachments is not None:
            body["attached_skills"] = attachments
        if intent:
            body["turn_intent"] = intent
        result = self._session_request(key, "POST", "message", body)
        job_id = result.get("job_id")
        if job_id:
            self._save_item(key, {"message_job_id": _identity(job_id)})
        return result

    def _host_action(self, key, snapshot, code, family):
        decision = (snapshot.get("host_decisions") or {}).get(family)
        if not isinstance(decision, dict) or decision.get("family") != family:
            raise WalkStop(Status.REFUSED, "current_host_decision_required", plan_metrics(snapshot))
        state = self._state(key)
        return {"project_id": state["project_id"], "session_id": state["session_id"],
                "action_code": code, "decision": decision}

    def _start_plan(self, key, snapshot):
        state = self._state(key)
        session = self._session(key)
        study = _identity(state["study_context_id"])
        context = self.request("GET", f"/api/study-contexts/{study}").get("context") or {}
        if context.get("id") != study:
            raise WalkStop(Status.REFUSED, "foreign_study_refused")
        source = context.get("data_source") or {}
        if source.get("database") != self.source()["database"] or not source.get("path"):
            raise WalkStop(Status.REFUSED, "selected_source_not_bound")
        provider = session.get("research_provider") or {}
        result = self.request("POST", "/api/jobs/agent-run", {
            "study_id": study, "study_context_id": study, "path": source["path"],
            "question": context.get("question"), "run_type": "full", "engine": "research_agent_pipeline",
            "candidate_plan_only": True, "planner_start_mode": "fresh", "literature_search_authorized": True,
            "external_llm_opt_in": True, "llm_provider": provider.get("provider", ""),
            "credential_source": provider.get("credential_source", ""),
            "host_action": self._host_action(key, snapshot, "generate_plan", "plan_transition"),
        })
        self._save_item(key, {"plan_job_id": _identity(result.get("job_id"))})

    def _plan(self, example):
        key = example["id"]
        self._conversation(key)
        if not self._state(key).get("question_sent"):
            question = example.get("example_question")
            if not isinstance(question, str) or not question.strip():
                package = self.request("GET", f"/api/capabilities/method-skills/{key}/package")
                question = example_question(package)
            if not question:
                raise WalkStop(Status.SKIPPED, "runnable_example_not_exposed")
            mention = self._bind_source(key)
            self._send(key, question + "\nPrepare a candidate plan and retrieve supporting literature for review.",
                       mentions=[mention], intent="implement_scientific_question")
            self._save_item(key, {"question_sent": True})
        self._wait_message(key)
        while True:
            snapshot = self._snapshot(key)
            workflow = snapshot.get("workflow") or {}
            metrics = plan_metrics(snapshot)
            active = snapshot.get("active_job") or {}
            if active.get("status") in {"queued", "running"}:
                self.pause()
                continue
            offer = (snapshot.get("host_decisions") or {}).get("plan_transition")
            needs_plan = workflow.get("next_action_code") in {
                "provider_ready_to_generate_plan", "target_trial_plan_ready", "plan_configuration_superseded",
                "failed_pipeline_requires_fresh_plan", "plan_scientific_changes_required",
            }
            if offer and needs_plan and not metrics["pending_design_choices"] and not self._state(key).get("plan_job_id"):
                self._start_plan(key, snapshot)
                continue
            if metrics["pending_design_choices"] or workflow.get("plan_review_summary"):
                return snapshot
            if any(row.get("plan_available") for row in snapshot.get("runs", [])):
                return snapshot
            code = token(workflow.get("next_action_code"), "candidate_plan_not_available")
            raise WalkStop(Status.REFUSED, code, metrics)

    def _literature(self, key, snapshot):
        state = self._state(key)
        run = snapshot.get("latest_run") or {}
        rid = run.get("run_id") or (snapshot.get("workflow", {}).get("plan_review_summary") or {}).get("run_id")
        if not rid:
            raise WalkStop(Status.SKIPPED, "literature_receipt_not_exposed")
        try:
            payload = self.request("GET", f"/api/copilot/pi/projects/{_identity(state['project_id'])}"
                                   f"/runs/{_identity(rid)}/artifacts/literature_evidence.json")
        except HTTPFailure as exc:
            if exc.status == 404:
                raise WalkStop(Status.SKIPPED, "literature_receipt_not_exposed") from None
            raise
        literature = payload.get("payload") or {}
        search = literature.get("search") or {}
        prisma = search.get("prisma") or {}
        review = literature.get("review") or {}
        counts = {"retrieved": prisma.get("identified"), "screened": prisma.get("screened"),
                  "included": prisma.get("included"), "sources": dict(source_presence(search)),
                  "source_count": len(search.get("sources_returning", [])),
                  "role_coverage": review.get("role_counts") or {},
                  "shortfall_count": len(review.get("shortfalls") or []),
                  "shortfalls": [row.get("role") for row in review.get("shortfalls", [])],
                  "status": token(literature.get("status")),
                  "unavailable": ["retrieval_flow_not_exposed"] if not prisma else []}
        if search.get("search_conducted") is not True:
            raise WalkStop(Status.SKIPPED, "literature_retrieval_not_conducted", counts)
        return counts

    def _execute(self, key, snapshot):
        metrics = plan_metrics(snapshot)
        state = self._state(key)
        if not self.options.approve_test_runs:
            raise WalkStop(Status.SKIPPED, "approval_flag_required", metrics)
        if metrics["pending_design_choices"]:
            raise WalkStop(Status.REFUSED, "design_choice_pending", metrics)
        if not state.get("execution_job_id"):
            # Fetch the offer immediately before approval; stale receipts cannot authorize it.
            snapshot = self._snapshot(key)
            action = self._host_action(key, snapshot, "execute_plan", "plan_review")
            rid = _identity(action["decision"].get("run_id"))
            session = self._session(key)
            bound_run = (session.get("binding") or {}).get("run_id")
            if bound_run and bound_run != rid:
                raise WalkStop(Status.REFUSED, "review_run_binding_mismatch", metrics)
            result = self.request("POST", "/api/jobs/agent-run-review", {
                "run_id": rid, "study_context_id": state["study_context_id"],
                "decision": "approved", "external_llm_opt_in": True, "host_action": action})
            self._save_item(key, {"execution_job_id": _identity(result.get("job_id")), "run_id": rid})
        job_id = _identity(self._state(key)["execution_job_id"])
        try:
            while True:
                job = self.request("GET", f"/api/jobs/{job_id}")
                if job.get("status") not in {"queued", "running"}:
                    break
                self.pause()
        except TimeoutError:
            raise WalkStop(Status.TIMED_OUT, "phase_timeout", {**metrics, "run_outcome": "timed_out"}) from None
        snapshot = self._snapshot(key)
        metrics.update(run_metrics(snapshot))
        metrics["run_outcome"] = token(job.get("status"))
        if job.get("status") != "done":
            raise WalkStop(Status.FAILED, "execution_job_failed", metrics)
        if metrics["gate_status"] == "blocked":
            raise WalkStop(Status.REFUSED, metrics["gate_reason_code"], metrics)
        if not metrics["execution_complete"]:
            raise WalkStop(Status.FAILED, "execution_not_complete", metrics)
        return metrics

    def walk_example(self, example, *, phase):
        snapshot = self._plan(example)
        return self._literature(example["id"], snapshot) if phase == "literature" else self._execute(example["id"], snapshot)

    def interaction_items(self):
        key = "walk-interactions"

        def attach():
            self._conversation(key)
            menu = self._session_request(key, "GET", "skills").get("items", [])
            selected = next((row for row in menu if row.get("enabled") and row.get("source") == "builtin"), None)
            if selected is None:
                raise WalkStop(Status.SKIPPED, "attachable_skill_not_exposed")
            ref = {k: selected[k] for k in ("source", "id", "version", "digest") if selected.get(k) is not None}
            self._session_request(key, "PUT", "draft-skills", {"attached_skills": [ref]})
            return {"attached_skill_count": 1, "attached": True}

        def mention():
            self._conversation(key)
            menu = self._session_request(key, "GET", "mentions", params={"kind": "file", "limit": 1})
            rows = menu.get("items") or []
            if rows:
                resolved = self._session_request(key, "POST", "mentions/resolve", {"kind": "file", "id": rows[0]["id"]})
                ref = resolved.get("reference")
                if not isinstance(ref, dict) or ref.get("kind") != "file":
                    raise WalkStop(Status.FAILED, "material_mention_invalid")
            else:
                ref = self._bind_source(key)
            self._session_request(key, "PUT", "draft-mentions", {"mentions": [ref]})
            return {"mention_count": 1, "resolved": True,
                    "scope": "project_file_mention" if rows else "registered_source_mention",
                    "unavailable": [] if rows else ["project_file_not_exposed"]}

        def task_table():
            self._conversation(key)
            snapshot = self._snapshot(key)
            return {"task_count": len(snapshot.get("runs") or []), "scope": "project_workflow_api"}

        def rename_delete(action):
            self._conversation(key)
            body = {"title": "EasyICU walk renamed"} if action == "rename" else {"confirm": True}
            self._session_request(key, "POST", action, body)
            if action == "delete":
                self._save_item(key, {"deleted": True})
            return {"renamed" if action == "rename" else "removed": True}

        self.item("interactions", "composer-skill-attach", attach)
        self.item("interactions", "material-mention", mention)
        self.item("interactions", "task-table", task_table)
        self.item("interactions", "finished-run-followup", self._followup, execution_policy=True)
        self.item("interactions", "design-choice-rerun", self._design_rerun, execution_policy=True)
        self.item("interactions", "rename-conversation", lambda: rename_delete("rename"))
        self.item("interactions", "delete-conversation", lambda: rename_delete("delete"))

    def _followup(self):
        for key, state in self.store.state["items"].items():
            if not state.get("execution_job_id") or state.get("deleted"):
                continue
            snapshot = self._snapshot(key)
            if (snapshot.get("latest_run") or {}).get("execution_complete") is True:
                self._session(key)
                self._send(key, "Explain the completed run's limitations using aggregate evidence only.")
                self._wait_message(key)
                return {"status": "replied"}
        raise WalkStop(Status.SKIPPED, "finished_walk_run_required")

    def _design_rerun(self):
        if not self.options.approve_test_runs:
            raise WalkStop(Status.SKIPPED, "approval_flag_required")
        for key, state in list(self.store.state["items"].items()):
            if not state.get("question_sent") or state.get("deleted"):
                continue
            snapshot = self._snapshot(key)
            decision = (snapshot.get("workflow") or {}).get("plan_decision") or {}
            recommended = decision.get("option_id") or decision.get("recommended_option_id")
            options = decision.get("options") or []
            if not recommended or options and not any(row.get("id") == recommended for row in options):
                continue
            session = self._session(key)
            binding = session.get("binding") or {}
            self._session_request(key, "POST", "plan-decision-selection", {
                "decision_code": decision["decision_code"], "option_id": recommended,
                "expected_revision": binding.get("study_revision"),
                "run_id": binding.get("run_id") or decision.get("source_run_id"),
                **({"card_sha256": decision["card_sha256"]} if decision.get("card_sha256") else {}),
            })
            self._save_item(key, {"plan_job_id": "", "execution_job_id": "", "run_id": ""})
            snapshot = self._plan({"id": key})
            return self._execute(key, snapshot)
        raise WalkStop(Status.SKIPPED, "recommended_design_choice_not_exposed")


def source_presence(search):
    """Only source presence is exposed; no source-specific record totals are inferred."""
    return [(token(source), 1) for source in search.get("sources_returning", [])]
