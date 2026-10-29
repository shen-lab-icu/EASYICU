"""Direct contract-owner validation; no ASGI/TestClient or live server needed."""

from dataclasses import replace

from easyicu.research_agent.method_skills import method_skill_catalog, method_skill_package
from easyicu.webserver.host_action_contracts import HostActionRequest
from easyicu.webserver.host_action_jobs import check_review_request
from easyicu.webserver.research_workflows import research_workflow_catalog
from easyicu.webserver.routes.pi_copilot import (
    PiDataSourceAuthorizationRequest, PiMessageRequest, PiProjectInitializeRequest,
    PiSessionCreateRequest, PiSessionDeleteRequest, PiSessionRenameRequest,
)
from easyicu.webserver.routes.mentions import DraftMentionsRequest, ResolveMentionRequest
from easyicu.webserver.routes.skill_attachments import DraftSkillsRequest
from tools.walk.contracts import PHASES
from tools.walk.conversations import example_question
from tools.walk.runner import WalkRunner
from tests.support.walk_http import PhaseAPI, walk_options


def test_generated_requests_validate_against_current_http_contract_owners(tmp_path):
    validated = []

    class ValidatingAPI(PhaseAPI):
        def request(self, method, path, body=None, **kwargs):
            models = {
                "/api/copilot/pi/sessions": PiSessionCreateRequest,
                "/api/copilot/pi/projects/initialize": PiProjectInitializeRequest,
                "data-source-authorization": PiDataSourceAuthorizationRequest,
                "mentions/resolve": ResolveMentionRequest,
                "message": PiMessageRequest,
                "rename": PiSessionRenameRequest,
                "delete": PiSessionDeleteRequest,
                "draft-skills": DraftSkillsRequest,
                "draft-mentions": DraftMentionsRequest,
            }
            if method in {"POST", "PUT"}:
                model = models.get(path) or next((value for suffix, value in models.items() if path.endswith("/" + suffix)), None)
                if model:
                    model.model_validate(body)
                    validated.append(model)
                if path == "/api/jobs/agent-run-review":
                    check_review_request(HostActionRequest.model_validate(body["host_action"]), body)
                    validated.append(HostActionRequest)
            return super().request(method, path, body, **kwargs)

    options = replace(walk_options(tmp_path, approve_test_runs=True), only=PHASES)
    report = WalkRunner(options, api=ValidatingAPI()).run()
    assert not report["status_counts"].get("failed"), report
    assert {PiSessionCreateRequest, PiMessageRequest, PiDataSourceAuthorizationRequest,
            ResolveMentionRequest, DraftSkillsRequest, DraftMentionsRequest,
            PiSessionRenameRequest, PiSessionDeleteRequest, HostActionRequest} <= set(validated)


def test_current_public_workflow_packages_supply_their_real_example_questions():
    workflows = research_workflow_catalog(enabled=True)["items"]
    assert len(workflows) >= 2
    for row in workflows:
        package = method_skill_package(row["id"], enabled=True)
        question = example_question(package)
        assert question and len(question) < 8000, row["id"]
        assert not question.startswith(("#", "Public source:"))
        assert "proposed teaching plan" not in question


def test_inventory_reads_the_current_public_catalogue_owner_shapes(tmp_path):
    class CatalogueAPI(PhaseAPI):
        def request(self, method, path, body=None, **kwargs):
            if path == "/api/capabilities":
                return {"capabilities": {"method_skills": method_skill_catalog(enabled=True)}}
            if path == "/api/capabilities/research-workflows":
                return research_workflow_catalog(enabled=True)
            return super().request(method, path, body, **kwargs)

    report = WalkRunner(replace(walk_options(tmp_path), only=("catalogue",)), api=CatalogueAPI()).run()
    row = report["items"][-1]
    assert row["status"] == "passed", [
        item for item in row["metrics"]["catalogue_items"] if item["owner_status"] == "not_reported"]
    assert row["metrics"]["unreported_owner_count"] == 0
    assert all(item["implementation"] != "not_reported" for item in row["metrics"]["catalogue_items"])
