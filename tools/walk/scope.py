"""Closed HTTP route set and walk-owned project/session mutation boundary."""

import re

from .contracts import Status, WalkStop

_READ_ONLY = frozenset({
    "/api/health", "/api/settings", "/api/copilot/pi/status", "/api/workspaces/registry",
    "/api/capabilities", "/api/capabilities/research-workflows", "/api/catalog", "/api/data-map",
    "/api/capabilities/method-skills/six-database-cohort-extraction/projection",
})


def check_request(method, path, body, params, state):
    """No settings writes, arbitrary artifact downloads or foreign resources."""
    items = list(state["items"].values())
    body, params = body or {}, params or {}
    projects = {row.get("project_id") for row in items if row.get("project_id")}
    if method == "GET" and (path in _READ_ONLY or re.fullmatch(
            r"/api/capabilities/method-skills/[A-Za-z0-9_.-]+/package", path)):
        return
    if method == "POST" and path == "/api/guided/drafts":
        if (re.fullmatch(r"EasyICU walk [a-f0-9]{12}", body.get("title", ""))
                and set(body) <= {"title", "folder_slug", "data_mode", "source"}):
            return
    if method == "POST" and path in {"/api/copilot/pi/projects/initialize", "/api/copilot/pi/sessions"}:
        if body.get("project_id") in projects:
            return
        raise WalkStop(Status.REFUSED, "walk_scope_refused")
    session = re.fullmatch(r"/api/copilot/pi/sessions/([^/]+)(?:/(.+))?", path)
    if session:
        sid, suffix = session.groups()
        owner = next((row for row in items if row.get("session_id") == sid), {})
        project = (params if method == "GET" else body).get("project_id")
        if not owner or project != owner.get("project_id"):
            raise WalkStop(Status.REFUSED, "walk_scope_refused")
        if (method == "GET" and suffix in {None, "skills", "mentions"}
                or method == "PUT" and suffix in {"draft-skills", "draft-mentions"}
                or method == "POST" and suffix in {"data-source-authorization", "mentions/resolve", "message",
                                                    "plan-decision-selection", "rename", "delete"}):
            return
    project = re.fullmatch(r"/api/copilot/pi/projects/([^/]+)/(.*)", path)
    if project and method == "GET":
        pid, suffix = project.groups()
        if pid not in projects:
            raise WalkStop(Status.REFUSED, "walk_scope_refused")
        if suffix == "workflow" or re.fullmatch(r"runs/[A-Za-z0-9_.-]+/artifacts/literature_evidence\.json", suffix):
            return
    if method == "GET" and path.startswith("/api/study-contexts/"):
        if any(path == f"/api/study-contexts/{row.get('study_context_id')}" for row in items):
            return
    if method == "GET" and path.startswith("/api/jobs/"):
        if any(path == f"/api/jobs/{row.get(key)}" for row in items
               for key in ("message_job_id", "plan_job_id", "execution_job_id") if row.get(key)):
            return
    if method == "POST" and path in {"/api/jobs/agent-run", "/api/jobs/agent-run-review"}:
        action = body.get("host_action") or {}
        if any(row.get("session_id") and row.get("project_id") == action.get("project_id")
               and row["session_id"] == action.get("session_id")
               and row.get("study_context_id") == body.get("study_context_id") for row in items):
            return
        raise WalkStop(Status.REFUSED, "walk_scope_refused")
    if method == "POST" and path == "/api/ideas/mine" and body.get("metadata_only") is True and body.get("allow_network") is False:
        return
    raise WalkStop(Status.REFUSED, "unsafe_api_route")
