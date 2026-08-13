from __future__ import annotations

from pathlib import Path

from easyicu.research_agent.agents.core import WriterAgent
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.pipeline import ResearchAgentPipeline
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.prompts import (
    load_prompt_pack,
    prompt_pack_files,
)
from easyicu.research_agent.publication_skills import (
    NATURE_FIGURE_SKILL_ID,
    NATURE_WRITING_SKILL_ID,
    PUBLICATION_SKILLS,
    compile_publication_skill_activation,
    publication_skill_flags_from_settings,
)
from easyicu.research_agent.schema import CohortDescriptor, ResearchContext


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Estimate a prespecified association.",
        cohort=CohortDescriptor(
            cohort_name="publication_skill_fixture",
            database="fixture",
            n_patients=20,
            n_stays=20,
        ),
        variables=[],
    )


def _system_prompt(*, enabled: bool) -> str:
    llm = ScriptedMockLLMClient(["Supported prose."], repeat_last=True)
    WriterAgent(llm, nature_writing_enabled=enabled)._call_section(
        section_name="Discussion",
        instruction="Write one supported paragraph.",
        context=_context(),
        evidence_ids=[],
        evidence_digest=None,
    )
    messages, _kwargs = llm.calls[-1]
    return next(message.content for message in messages if message.role == "system")


def test_publication_registry_has_two_default_case_neutral_skills() -> None:
    assert [skill.skill_id for skill in PUBLICATION_SKILLS] == [
        NATURE_FIGURE_SKILL_ID,
        NATURE_WRITING_SKILL_ID,
    ]
    assert all(skill.default_enabled for skill in PUBLICATION_SKILLS)
    rendered = "\n".join(
        str(value)
        for skill in PUBLICATION_SKILLS
        for value in skill.to_dict().values()
    ).casefold()
    for case_specific_token in ("mimic", "sofa", "sepsis", "figure 2"):
        assert case_specific_token not in rendered


def test_publication_activation_is_deterministic_and_auditable() -> None:
    first = compile_publication_skill_activation().to_dict()
    second = compile_publication_skill_activation().to_dict()

    assert first == second
    assert first["active_skill_ids"] == [
        NATURE_FIGURE_SKILL_ID,
        NATURE_WRITING_SKILL_ID,
    ]
    assert len(first["activation_sha256"]) == 64

    writing_off = compile_publication_skill_activation(
        nature_writing_enabled=False
    ).to_dict()
    assert writing_off["active_skill_ids"] == [NATURE_FIGURE_SKILL_ID]
    assert writing_off["inactive_skill_ids"] == [NATURE_WRITING_SKILL_ID]
    assert writing_off["activation_sha256"] != first["activation_sha256"]


def test_web_master_and_per_skill_switches_resolve_fail_closed() -> None:
    assert publication_skill_flags_from_settings({}) == {
        "nature_figure_enabled": True,
        "nature_writing_enabled": True,
    }
    assert publication_skill_flags_from_settings(
        {
            "science_skills_enabled": True,
            "nature_figure_skill_enabled": False,
            "nature_writing_skill_enabled": True,
        }
    ) == {
        "nature_figure_enabled": False,
        "nature_writing_enabled": True,
    }
    assert publication_skill_flags_from_settings(
        {
            "science_skills_enabled": False,
            "nature_figure_skill_enabled": True,
            "nature_writing_skill_enabled": True,
        }
    ) == {
        "nature_figure_enabled": False,
        "nature_writing_enabled": False,
    }


def test_nature_writing_prompt_is_default_on_and_can_be_unplugged() -> None:
    enabled = _system_prompt(enabled=True)
    disabled = _system_prompt(enabled=False)

    marker = "NATURE-STYLE EVIDENCE WRITING SKILL"
    assert marker in enabled
    assert "Never invent or silently complete" in enabled
    assert marker not in disabled


def test_nature_writing_contract_is_versioned_in_prompt_pack() -> None:
    prompt_pack = load_prompt_pack()
    files = prompt_pack_files()

    assert "nature_writing" in prompt_pack
    assert any(path.endswith("/nature_writing.txt") for path in files)
    assert all(len(digest) == 64 for digest in files.values())


def test_pipeline_defaults_bind_both_publication_skills(tmp_path: Path) -> None:
    default_config = PipelineConfig(workdir=tmp_path / "default")
    default_pipeline = ResearchAgentPipeline.from_config(default_config)
    assert default_pipeline._enable_publication_figure_skill is True
    assert default_pipeline._enable_nature_writing_skill is True

    disabled_config = PipelineConfig(
        workdir=tmp_path / "disabled",
        enable_publication_figure_skill=False,
        enable_nature_writing_skill=False,
    )
    disabled_pipeline = ResearchAgentPipeline.from_config(disabled_config)
    assert disabled_pipeline._enable_publication_figure_skill is False
    assert disabled_pipeline._enable_nature_writing_skill is False
