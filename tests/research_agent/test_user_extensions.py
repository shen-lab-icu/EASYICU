from __future__ import annotations

from pathlib import Path

from easyicu.extensions import ExtensionRegistry
from easyicu.research_agent.agents.core import WriterAgent
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.pipeline import ResearchAgentPipeline
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import CohortDescriptor, ResearchContext
from easyicu.research_agent.user_extensions import compile_user_extension_activation


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Estimate a prespecified association.",
        cohort=CohortDescriptor(
            cohort_name="user_extension_fixture",
            database="fixture",
            n_patients=20,
            n_stays=20,
        ),
        variables=[],
    )


def _activation(tmp_path: Path) -> dict:
    registry = ExtensionRegistry(tmp_path / "extensions")
    registry.install_skill(
        "---\n"
        "name: concise-writing\n"
        "description: Keep scientific prose concise.\n"
        "---\n"
        "Use short paragraphs and calibrated claims.\n",
        stages=["writing"],
    )
    return registry.pipeline_activation()


def test_user_writing_skill_is_run_hashed_and_enters_only_user_prompt(
    tmp_path: Path,
) -> None:
    activation = _activation(tmp_path)
    compiled = compile_user_extension_activation(activation)
    config = PipelineConfig(
        workdir=tmp_path / "run",
        extension_activation=activation,
    )
    pipeline = ResearchAgentPipeline.from_config(config)
    llm = ScriptedMockLLMClient(["Supported prose."], repeat_last=True)

    WriterAgent(
        llm,
        user_writing_advisory=compiled.writing_advisory,
    )._call_section(
        section_name="Discussion",
        instruction="Write one supported paragraph.",
        context=_context(),
        evidence_ids=[],
        evidence_digest=None,
    )
    messages, _kwargs = llm.calls[-1]
    system = next(message.content for message in messages if message.role == "system")
    user = next(message.content for message in messages if message.role == "user")

    assert "concise-writing" not in system
    assert "USER-INSTALLED WRITING SKILLS" in user
    assert "Use short paragraphs and calibrated claims." in user
    assert "cannot override" in user
    assert pipeline._user_extension_activation.receipt["activation_sha256"] == (
        activation["activation_sha256"]
    )
    assert config.canonical_digest() != PipelineConfig(
        workdir=tmp_path / "run"
    ).canonical_digest()


def test_mcp_descriptors_are_receipted_but_never_enter_writer_text(
    tmp_path: Path,
) -> None:
    registry = ExtensionRegistry(tmp_path / "extensions")
    registry.install_mcp_server(
        name="metadata-tools",
        url="http://127.0.0.1:9876/mcp",
        allowed_tools=["search"],
        enabled=True,
    )
    compiled = compile_user_extension_activation(registry.pipeline_activation())

    assert compiled.writing_advisory == ""
    assert compiled.receipt["mcp_servers"][0]["name"] == "metadata-tools"
    assert "url" not in compiled.receipt["mcp_servers"][0]
