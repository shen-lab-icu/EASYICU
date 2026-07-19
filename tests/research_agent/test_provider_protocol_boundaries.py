"""Provider protocol layering and legacy import compatibility."""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

import easyicu.research_agent as research_agent
from easyicu.research_agent import llm, llm_mocks, providers
from easyicu.research_agent.providers import protocol


def test_legacy_llm_protocol_exports_are_identical() -> None:
    assert llm.LLMClient is protocol.LLMClient
    assert llm.LLMMessage is protocol.LLMMessage
    assert research_agent.LLMClient is protocol.LLMClient


def test_mock_annotations_resolve_without_importing_back_from_llm() -> None:
    hints = get_type_hints(llm_mocks.MockLLMClient.complete)
    assert hints["messages"] == get_type_hints(protocol.LLMClient.complete)["messages"]
    assert llm_mocks.MockLLMClient().complete(
        [protocol.LLMMessage(role="user", content="hello")]
    )


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
assert 'easyicu.research_agent.llm' not in sys.modules
assert 'easyicu.research_agent.providers.factory' not in sys.modules
assert LLMMessage('user', 'ok').content == 'ok'
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
