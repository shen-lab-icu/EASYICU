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


def _load_agents_helpers(ra):
    """The helpers are private to ``agents`` — load the submodule directly."""
    return importlib.import_module(ra.__name__ + ".agents")


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
        ' [{"step_id":"01_table_one","intent":"t1","inputs":[],"expected_outputs":[]}]}\n'
        "```"
    )
    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="Is sofa2 -> death?",
        cohort=schema.CohortDescriptor(cohort_name="c", database="d", n_patients=1, n_stays=1),
        variables=[],
    )

    class _DummyLLM:
        name = "dummy"

        def complete(self, messages, **kwargs):
            return raw

    from easyicu.research_agent.agents import PlannerAgent
    plan = PlannerAgent(_DummyLLM())._parse(raw, ctx)
    assert plan.steps and plan.steps[0].step_id == "01_table_one"


def test_planner_parse_drops_extra_step_fields(ra):
    raw = (
        '{"research_question": "Is sofa2 -> death?", "extra": "drop me", "steps":'
        ' [{"step_id":"06_cross_database","intent":"protocol","inputs":[],'
        '"expected_outputs":[],"note":"external cohort unavailable"}]}'
    )
    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="Is sofa2 -> death?",
        cohort=schema.CohortDescriptor(cohort_name="c", database="d", n_patients=1, n_stays=1),
        variables=[],
    )

    class _DummyLLM:
        name = "dummy"

        def complete(self, messages, **kwargs):
            return raw

    from easyicu.research_agent.agents import PlannerAgent
    plan = PlannerAgent(_DummyLLM())._parse(raw, ctx)
    assert plan.steps[0].step_id == "06_cross_database"
    assert not hasattr(plan.steps[0], "note")


def test_planner_uses_enough_completion_budget(ra):
    """Reasoning models can spend part of max_tokens before final JSON."""
    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="Is sofa2 -> death?",
        cohort=schema.CohortDescriptor(cohort_name="c", database="d", n_patients=1, n_stays=1),
        variables=[],
    )

    class _CapturingLLM:
        name = "dummy"

        def __init__(self):
            self.kwargs = None

        def complete(self, messages, **kwargs):
            self.kwargs = kwargs
            return (
                '{"research_question": "Is sofa2 -> death?", "steps":'
                ' [{"step_id":"01_table_one","intent":"t1","inputs":[],"expected_outputs":[]}]}'
            )

    from easyicu.research_agent.agents import PlannerAgent
    llm = _CapturingLLM()
    PlannerAgent(llm).run(ctx)
    assert llm.kwargs["max_tokens"] >= 4096


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

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_FakeOpenAI))

    from easyicu.research_agent.llm import LLMMessage, OpenAIClient

    extra_body = {"reasoning": {"effort": "none", "exclude": True}}
    client = OpenAIClient(
        model="z-ai/glm-4.5-air:free",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        extra_body=extra_body,
    )
    assert client.complete([LLMMessage(role="user", content="hi")]) == "ok"
    assert calls["create"]["extra_body"] == extra_body


def test_writer_strips_markdown_fence(ra, tmp_path: Path):
    """If the LLM wraps the manuscript in ```markdown, the binder must
    still see raw markdown so it can locate ``{evidence:*}``."""
    raw = "```markdown\n# Title\n\nCohort: {evidence:table_one}.\n```"

    class _DummyLLM:
        name = "dummy"

        def complete(self, messages, **kwargs):
            return raw

    from easyicu.research_agent.agents import WriterAgent
    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="x",
        cohort=schema.CohortDescriptor(cohort_name="c", database="d", n_patients=1, n_stays=1),
        variables=[],
    )
    out = WriterAgent(_DummyLLM()).run(context=ctx, evidence_ids=["table_one"])
    # The fence must be stripped so the binder regex matches.
    assert "{evidence:table_one}" in out
    assert "```markdown" not in out


def test_writer_language_prompt_preserves_evidence_ids(ra):
    """The Chinese writer mode should ask for zh prose but keep evidence ids ASCII."""
    captured = {}

    class _DummyLLM:
        name = "dummy"

        def complete(self, messages, **kwargs):
            captured["prompt"] = messages[-1].content
            return "# 标题\n\n结果：12 例 {evidence:table_one}。\n"

    from easyicu.research_agent.agents import WriterAgent
    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="x",
        cohort=schema.CohortDescriptor(cohort_name="c", database="d", n_patients=1, n_stays=1),
        variables=[],
    )

    out = WriterAgent(_DummyLLM(), language="zh").run(
        context=ctx,
        evidence_ids=["table_one"],
    )

    assert "Simplified Chinese" in captured["prompt"]
    assert "do not translate evidence ids" in captured["prompt"]
    assert "{evidence:table_one}" in out
