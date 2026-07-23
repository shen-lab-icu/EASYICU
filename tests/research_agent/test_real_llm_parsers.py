"""Parser-robustness tests for the real-LLM smoke path (T1.3).

Free-tier OpenRouter models (gemini-2.0-flash-exp, llama-3.x, etc.)
routinely:

* wrap JSON in a ```json fence with prose before/after,
* prefix Python code with "Sure! Here is the script:" and ```python,
* surround the manuscript markdown with ```markdown … ```,
* mention an opening brace inside a string literal that earlier
  parsers miscounted.

This module pins the agent helpers' tolerance for those quirks so a
regression in ``_strip_code_fence`` / ``_first_json_block`` shows up
without anyone having to spend a real LLM call.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient


def _load_agents_helpers(ra):
    """Load parser helpers from their canonical implementation module."""
    return importlib.import_module(ra.__name__ + ".agents.core")


def test_strip_code_fence_handles_leading_prose(ra):
    helpers = _load_agents_helpers(ra)
    raw = (
        "Sure! Here is the analysis plan you asked for:\n\n"
        "```json\n"
        '{"research_question": "x", "steps": []}\n'
        "```\n\n"
        "Let me know if you need more steps."
    )
    out = helpers._strip_code_fence(raw)
    assert out.strip().startswith("{")
    assert "Sure!" not in out
    assert "Let me know" not in out


def test_strip_code_fence_python_block(ra):
    helpers = _load_agents_helpers(ra)
    raw = (
        "Here you go:\n```python\n"
        "import pandas as pd\n"
        "df = pd.read_parquet('x')\n"
        "```"
    )
    out = helpers._strip_code_fence(raw).strip()
    assert out.startswith("import pandas")


def test_strip_code_fence_markdown_block(ra):
    helpers = _load_agents_helpers(ra)
    raw = "Here is the manuscript:\n```markdown\n# Title\n\nBody.\n```\n"
    out = helpers._strip_code_fence(raw)
    assert "# Title" in out
    assert "Here is" not in out


def test_strip_code_fence_no_fence_passthrough(ra):
    helpers = _load_agents_helpers(ra)
    raw = '{"a": 1}'
    assert helpers._strip_code_fence(raw) == raw


def test_first_json_block_skips_braces_in_strings(ra):
    """Earlier parsers miscounted braces inside string literals; pin the fix."""
    helpers = _load_agents_helpers(ra)
    raw = (
        'Some prose before. {"intent": "the {evidence:foo} placeholder is required",'
        ' "steps": [{"step_id": "01"}]}'
    )
    block = helpers._first_json_block(raw)
    assert block is not None
    import json as _json

    parsed = _json.loads(block)
    assert parsed["steps"][0]["step_id"] == "01"


def test_planner_parse_recovers_fenced_json(ra):
    """End-to-end: PlannerAgent._parse must accept fenced JSON."""
    raw = (
        "Sure, here's the plan:\n```json\n"
        '{"research_question": "Is sofa2 -> death?", "steps":'
        ' [{"step_id":"01_table_one","planned_analysis_role":"auxiliary",'
        '"intent":"t1","inputs":[],"expected_outputs":[]}]}\n'
        "```"
    )
    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="Is sofa2 -> death?",
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="d", n_patients=1, n_stays=1
        ),
        variables=[],
    )

    class _DummyLLM:
        name = "dummy"

        def complete(self, messages, **kwargs):
            return raw

    from easyicu.research_agent.agents.core import PlannerAgent

    plan = PlannerAgent(_DummyLLM())._parse(raw, ctx)
    assert plan.steps and plan.steps[0].step_id == "01_table_one"


def test_planner_parse_preserves_declared_display_labels(ra):
    raw = (
        '{"research_question":"Estimate an association.",'
        '"display_labels":{"death":"In-hospital mortality",'
        '"primary":"Primary analysis"},'
        '"steps":[{"step_id":"01_model",'
        '"planned_analysis_role":"primary","intent":"fit",'
        '"inputs":[],"expected_outputs":["statistic:adjusted_effect"]}]}'
    )
    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="Estimate an association.",
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="d", n_patients=1, n_stays=1
        ),
        variables=[],
    )

    class _DummyLLM:
        name = "dummy"

        def complete(self, messages, **kwargs):
            return raw

    from easyicu.research_agent.agents.core import PlannerAgent

    plan = PlannerAgent(_DummyLLM())._parse(raw, ctx)

    assert plan.display_labels == {
        "death": "In-hospital mortality",
        "primary": "Primary analysis",
    }


def test_planner_parse_drops_extra_step_fields(ra):
    raw = (
        '{"research_question": "Is sofa2 -> death?", "extra": "drop me", "steps":'
        ' [{"step_id":"06_cross_database","planned_analysis_role":"auxiliary",'
        '"intent":"protocol","inputs":[], '
        '"expected_outputs":[],"note":"external cohort unavailable"}]}'
    )
    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="Is sofa2 -> death?",
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="d", n_patients=1, n_stays=1
        ),
        variables=[],
    )

    class _DummyLLM:
        name = "dummy"

        def complete(self, messages, **kwargs):
            return raw

    from easyicu.research_agent.agents.core import PlannerAgent

    plan = PlannerAgent(_DummyLLM())._parse(raw, ctx)
    assert plan.steps[0].step_id == "06_cross_database"
    assert not hasattr(plan.steps[0], "note")


def test_planner_uses_enough_completion_budget(ra):
    """Reasoning models can spend part of max_tokens before final JSON."""
    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="Is sofa2 -> death?",
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="d", n_patients=1, n_stays=1
        ),
        variables=[],
    )

    from easyicu.research_agent.agents.core import PlannerAgent

    llm = ScriptedMockLLMClient(
        [
            '{"research_question": "Is sofa2 -> death?", "steps":'
            ' [{"step_id":"01_table_one","planned_analysis_role":"auxiliary",'
            '"intent":"t1","inputs":[],"expected_outputs":[]}]}'
        ]
    )
    PlannerAgent(llm).run(ctx)
    assert llm.calls[0][1]["max_tokens"] >= 4096


def test_planner_retries_dictionary_concept_absent_from_sealed_typed_input(
    tmp_path,
):
    """A legal global concept is not executable authority for every cohort."""

    from easyicu.research_agent.agents.core import PlannerAgent
    from tests.research_agent.test_materialized_column_metadata import (
        _build_v2_context,
    )

    context = _build_v2_context(tmp_path)

    def response(concept_id: str) -> str:
        return (
            '{"research_question":"Describe the sealed cohort.",'
            '"analysis_type":"descriptive_epidemiology",'
            '"cohort":{"name":"primary","inclusion":[{'
            f'"concept_id":"{concept_id}",'
            '"time_window":{"anchor":"icu_admission",'
            '"start_offset_hours":0,"end_offset_hours":24},'
            '"aggregation":"max","op":"not_missing","value":null}],'
            '"exclusion":[]},'
            '"steps":[{"step_id":"01_define_cohort",'
            '"planned_analysis_role":"auxiliary",'
            '"intent":"Materialize the declared analysis cohort.",'
            '"inputs":["stay_id","lact_max"],'
            '"expected_outputs":["artifact:analysis_cohort"],'
            '"method":"cohort_definition"}]}'
        )

    llm = ScriptedMockLLMClient([response("hr"), response("lact")])
    plan = PlannerAgent(llm).run(context)

    assert len(llm.calls) == 2
    assert plan.cohort is not None
    assert plan.cohort.inclusion[0].concept_id == "lact"
    feedback = llm.calls[1][0][-1].content
    assert "not executable against this sealed input" in feedback
    assert "lact_max" in feedback


def test_cohort_concept_allowlist_includes_sofa2_overlay() -> None:
    from easyicu.research_agent.planning.cohort_contract import known_concept_ids

    assert {"sofa2", "sep3_sofa2", "sofa2_resp"} <= known_concept_ids()


def test_openai_client_passes_provider_extra_body(ra, monkeypatch):
    """OpenRouter reasoning controls must reach the SDK request."""
    calls = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            calls["create"] = kwargs
            message = types.SimpleNamespace(content="ok")
            choice = types.SimpleNamespace(message=message, finish_reason="stop")
            usage = types.SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )
            return types.SimpleNamespace(choices=[choice], usage=usage)

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            calls["client"] = kwargs
            self.chat = types.SimpleNamespace(
                completions=_FakeCompletions(),
            )

    monkeypatch.setitem(
        sys.modules, "openai", types.SimpleNamespace(OpenAI=_FakeOpenAI)
    )

    from easyicu.research_agent.providers.factory import authorize_provider_client
    from easyicu.research_agent.providers.llm import LLMMessage, OpenAIClient

    extra_body = {"reasoning": {"effort": "none", "exclude": True}}
    client = OpenAIClient(
        model="z-ai/glm-4.5-air:free",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        extra_body=extra_body,
    )
    authorize_provider_client(
        client,
        provider="openai",
        model="z-ai/glm-4.5-air:free",
        base_url="https://openrouter.ai/api/v1",
        destination="external",
        environment={"EASYICU_ALLOW_EXTERNAL_LLM": "1"},
    )
    assert client.complete([LLMMessage(role="user", content="hi")]) == "ok"
    assert calls["create"]["extra_body"] == extra_body


def test_writer_strips_markdown_fence(ra, tmp_path: Path):
    """If the LLM wraps the manuscript in ```markdown, the binder must
    still see raw markdown so it can locate ``{evidence:*}``."""
    raw = "```markdown\n# Title\n\nCohort: {evidence:table_one}.\n```"

    from easyicu.research_agent.agents.core import WriterAgent

    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="x",
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="d", n_patients=1, n_stays=1
        ),
        variables=[],
    )
    out = WriterAgent(ScriptedMockLLMClient([raw], repeat_last=True)).run(
        context=ctx, evidence_ids=["table_one"]
    )
    # The fence must be stripped so the binder regex matches.
    assert "{evidence:table_one}" in out
    assert "```markdown" not in out


def test_writer_language_prompt_preserves_evidence_ids(ra):
    """The Chinese writer mode should ask for zh prose but keep evidence ids ASCII."""
    from easyicu.research_agent.agents.core import WriterAgent

    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="x",
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="d", n_patients=1, n_stays=1
        ),
        variables=[],
    )

    llm = ScriptedMockLLMClient(
        ["# 标题\n\n结果：12 例 {evidence:table_one}。\n"],
        repeat_last=True,
    )
    out = WriterAgent(llm, language="zh").run(
        context=ctx,
        evidence_ids=["table_one"],
    )

    prompts = "\n".join(
        message.content for messages, _kwargs in llm.calls for message in messages
    )
    assert "Simplified Chinese" in prompts
    assert "do not translate evidence ids" in prompts
    assert "{evidence:table_one}" in out


def test_writer_prompt_discourages_tbd_and_manifest_narration(ra):
    # The writer contract (writer.txt → _WRITER_GUIDE) lands in the
    # *system* message of every per-section LLM call. Capture the full
    # joined prompt across every section so we can assert on contract
    # text regardless of which section was last called.
    from easyicu.research_agent.agents.core import WriterAgent

    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="x",
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="d", n_patients=1, n_stays=1
        ),
        variables=[],
    )

    llm = ScriptedMockLLMClient(
        [
            "# Title\n\n## Results\n\nBaseline characteristics are summarised "
            "in Table 1 {evidence:table_one}.\n"
        ],
        repeat_last=True,
    )
    out = WriterAgent(llm).run(context=ctx, evidence_ids=["table_one"])

    captured = {"system": "", "user": ""}
    for messages, _kwargs in llm.calls:
        for message in messages:
            captured[message.role] += message.content + "\n"

    # Writer contract assertions land in the system prompt.
    assert "`[TBD]`" in captured["system"]
    assert "warning: see manifest" in captured["system"]
    assert (
        "Only cite `table_one`, `outcome_rate`, or `primary_association`"
        in captured["system"]
    )
    # Writer contract should reference `model_performance` as a fallback
    # baseline source for prediction tasks. Exact wording has shifted; we
    # assert on the alias token rather than a specific sentence.
    assert "`model_performance`" in captured["system"]
    assert "Use exactly single braces" in captured["user"]
    assert "mechanisms, strengths, or limitations" in captured["user"]
    assert "Each conclusion sentence must cite" in captured["user"]
    assert "TBD by author" not in captured["user"]
    assert "Funding information was not available" in captured["user"]
    # The dummy LLM's stock response should land in the bound output.
    assert "{evidence:table_one}" in out


def test_openrouter_reasoning_extra_body_skips_gpt_oss(ra):
    from easyicu.research_agent.providers.llm import openrouter_reasoning_extra_body

    assert openrouter_reasoning_extra_body("openai/gpt-oss-120b:free") is None
    assert openrouter_reasoning_extra_body("z-ai/glm-4.5-air:free") == {
        "reasoning": {"effort": "none", "exclude": True}
    }
