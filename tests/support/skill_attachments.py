"""Synthetic personal skill attachment service fixtures."""
import pytest
from easyicu.extensions import ExtensionRegistry
from easyicu.webserver.pi_copilot.service import PiCopilotService
from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeGateway, study_state as study_state


def _install(registry, body="Use concise sentences.", **kwargs):
    return registry.install_skill(
        "---\nname: synthetic-helper\ndescription: Synthetic helper\n---\n" + body,
        **kwargs,
    )

@pytest.fixture
def registry(tmp_path):
    return ExtensionRegistry(tmp_path / "extensions")

@pytest.fixture
def service(tmp_path, registry, study_state):
    instance = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=FakeGateway(), extension_registry=registry)
    yield instance
    instance.close()

def _session(service):
    return service.create_session(project_id="synthetic-project", external_llm_opt_in=True)["session"]
