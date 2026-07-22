"""Provider protocol layering and canonical import boundaries."""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

import pytest

import easyicu.research_agent as research_agent
from easyicu.research_agent import providers
from easyicu.research_agent.providers import llm, mocks as llm_mocks
from easyicu.research_agent.providers import protocol


def test_canonical_llm_protocol_exports_are_identical() -> None:
    assert llm.LLMClient is protocol.LLMClient
    assert llm.LLMMessage is protocol.LLMMessage
    assert research_agent.LLMClient is protocol.LLMClient
    assert research_agent.MockLLMClient is llm_mocks.MockLLMClient


def test_mock_annotations_resolve_without_importing_back_from_llm() -> None:
    hints = get_type_hints(llm_mocks.MockLLMClient.complete)
    assert hints["messages"] == get_type_hints(protocol.LLMClient.complete)["messages"]
    assert llm_mocks.MockLLMClient().complete(
        [protocol.LLMMessage(role="user", content="hello")]
    )


def test_mock_subclass_cannot_inherit_offline_authority() -> None:
    class InterruptingAnalyzer(llm_mocks.MockLLMClient):
        name = "interrupting-analyzer"

    with pytest.raises(ValueError, match="external LLM transport is disabled"):
        InterruptingAnalyzer()


def test_provider_package_keeps_factory_import_lazy() -> None:
    path = Path(inspect.getsourcefile(providers))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_level_imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    assert not any(node.module == "factory" for node in top_level_imports)


def test_provider_factory_legacy_exports_remain_available() -> None:
    assert set(providers.__all__) == {
        "DEFAULT_OPENAI_BASE_URL",
        "DEFAULT_OPENROUTER_BASE_URL",
        "LOCAL_OPENAI_DUMMY_API_KEY",
        "ProviderConfigurationError",
        "build_provider_client",
        "is_loopback_openai_base_url",
        "resolve_provider_base_url",
    }
    assert callable(providers.build_provider_client)
    assert callable(providers.resolve_provider_base_url)
    assert providers.DEFAULT_OPENAI_BASE_URL.startswith("https://")


def test_protocol_first_import_does_not_load_concrete_llm_or_factory() -> None:
    code = """
import sys
from easyicu.research_agent.providers.protocol import LLMMessage
assert 'easyicu.research_agent.providers.llm' not in sys.modules
assert 'easyicu.research_agent.providers.factory' not in sys.modules
assert LLMMessage('user', 'ok').content == 'ok'
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_production_provider_import_does_not_load_mock_module() -> None:
    code = """
import sys
from easyicu.research_agent.providers.llm import OpenAIClient, llm_is_mockish
assert 'easyicu.research_agent.providers.mocks' not in sys.modules
assert OpenAIClient.__module__ == 'easyicu.research_agent.providers.llm'
assert not llm_is_mockish(OpenAIClient.__new__(OpenAIClient))
assert 'easyicu.research_agent.providers.mocks' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_mockish_classification_uses_only_registered_offline_graphs(ra) -> None:
    from easyicu.research_agent.providers.llm import FallbackLLMClient, llm_is_mockish
    from easyicu.research_agent.providers.mocks import MockLLMClient

    class DuckMockRouter:
        name = "mock-router"
        _model = "definitely-mock"

        def __init__(self, child):  # noqa: ANN001
            self.child = child

        def for_role(self, _role):  # noqa: ANN001
            return self.child

        def iter_clients(self):
            return iter([self.child])

        def complete(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            return "mock"

    child = MockLLMClient()
    assert llm_is_mockish(child) is True
    assert llm_is_mockish(FallbackLLMClient(child)) is True
    assert llm_is_mockish(DuckMockRouter(child)) is False


def test_protocol_has_no_concrete_provider_or_pipeline_dependency() -> None:
    path = Path(inspect.getsourcefile(protocol))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not imported & {"llm", "llm_mocks", "pipeline", "schema"}


def test_production_entrypoints_cannot_construct_openai_client_outside_factory():
    root = Path(__file__).resolve().parents[2]
    targets = [
        root / "src" / "easyicu" / "research_agent",
        root / "tools",
        root / "scripts",
        root / "examples",
    ]
    violations: list[str] = []
    for target in targets:
        for path in target.rglob("*.py"):
            if path.name == "factory.py" or path.parts[-2:] == (
                "providers",
                "llm.py",
            ):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr if isinstance(func, ast.Attribute) else ""
                )
                if name == "OpenAIClient":
                    violations.append(f"{path.relative_to(root)}:{node.lineno}")
    assert violations == []


def test_all_external_entry_surfaces_route_through_provider_factory():
    root = Path(__file__).resolve().parents[2]
    paths = (
        "src/easyicu/research_agent/cli.py",
        "src/easyicu/research_agent/replication_cli.py",
        "src/easyicu/research_agent/mcp_server.py",
        "src/easyicu/research_agent/evaluation/tier2_jury.py",
        "tools/run_research_agent_bench.py",
        "tools/run_openrouter_fullflow_validation.py",
    )
    for relative in paths:
        source = (root / relative).read_text(encoding="utf-8")
        assert "build_provider_client(" in source, relative


def test_production_prompt_calls_use_the_authorized_delivery_boundary() -> None:
    root = Path(__file__).resolve().parents[2]
    targets = [root / name for name in ("src", "tools", "scripts", "examples")]
    allowed_internal = {
        "src/easyicu/research_agent/providers/factory.py",
        "src/easyicu/research_agent/providers/llm.py",
        "src/easyicu/research_agent/providers/cost.py",
        "src/easyicu/research_agent/replication/envelope.py",
    }
    violations: list[str] = []
    for target in targets:
        for path in target.rglob("*.py"):
            relative = path.relative_to(root).as_posix()
            if relative in allowed_internal:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"complete", "complete_with_images"}
                ):
                    violations.append(f"{relative}:{node.lineno}:{node.func.attr}")
    assert violations == []


def test_provider_trust_registration_is_confined_to_reviewed_owners() -> None:
    root = Path(__file__).resolve().parents[2]
    targets = [root / name for name in ("src", "tools", "scripts", "examples")]
    allowed = {
        "_register_provider_wrapper": {
            "src/easyicu/research_agent/providers/llm.py",
            "src/easyicu/research_agent/providers/cost.py",
            "src/easyicu/research_agent/replication/envelope.py",
            "tools/run_research_know_how_planner_ab.py",
        },
        "register_offline_test_client": {
            "src/easyicu/research_agent/providers/mocks.py",
            "src/easyicu/research_agent/discovery/idea_mining_data_first_route.py",
        },
    }
    violations: list[str] = []
    for target in targets:
        for path in target.rglob("*.py"):
            relative = path.relative_to(root).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr if isinstance(func, ast.Attribute) else ""
                )
                if name in allowed and relative not in allowed[name]:
                    violations.append(f"{relative}:{node.lineno}:{name}")
    assert violations == []
