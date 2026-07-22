"""Provider protocol layering and canonical import boundaries."""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

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


def test_renamed_mock_subclass_remains_explicitly_offline() -> None:
    class InterruptingAnalyzer(llm_mocks.MockLLMClient):
        name = "interrupting-analyzer"

    assert llm.llm_is_mockish(InterruptingAnalyzer())


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
    targets = [root / "src" / "easyicu" / "research_agent", root / "tools"]
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
                    else func.attr
                    if isinstance(func, ast.Attribute)
                    else ""
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
