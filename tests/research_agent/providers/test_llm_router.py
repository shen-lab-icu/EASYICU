"""Per-agent LLM router (T2.3).

These tests pin three contracts:

1. :class:`LLMRouter` constructs sensibly, routes by role, and falls
   back to the default client.
2. The pipeline correctly threads role-bound clients into each agent
   so a planner-only model is invoked only for planning, etc.
3. Backwards compatibility: a plain :class:`LLMClient` (no
   ``for_role`` method) still works exactly as before.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _spy_llm(ra, name: str, *, context=None):
    """Return one exact registered built-in mock with a test display label."""
    from easyicu.research_agent.providers.mocks import MockLLMClient

    client = MockLLMClient(context=context)
    client.spy_name = name
    return client


def _pipeline_context(synthetic_cohort):
    from easyicu.research_agent.research_context.builder import (
        build_research_context,
    )

    return build_research_context(
        research_question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="router_test",
        database="synthetic",
        target_outcome="death",
    )


def _isolate_article_suite_contract(monkeypatch) -> None:
    from easyicu.research_agent.agents.core import PlannerAgent

    original_run = PlannerAgent.run

    def run_without_article_suite(self, context, **kwargs):
        kwargs["enforce_article_contract"] = False
        return original_run(self, context, **kwargs)

    monkeypatch.setattr(PlannerAgent, "run", run_without_article_suite)


def _spy_prompts(spy) -> list[str]:
    return [
        next(
            (
                message.content
                for message in reversed(messages)
                if message.role == "user"
            ),
            "",
        )
        for messages, _kwargs in spy.calls
    ]


# ---------------------------------------------------------------------------
# Constructor + routing
# ---------------------------------------------------------------------------


def test_router_requires_at_least_one_client(ra):
    with pytest.raises(ValueError):
        ra.LLMRouter()


def test_router_for_role_returns_role_specific_client(ra):
    default = _spy_llm(ra, "default")
    planner = _spy_llm(ra, "planner")
    coder = _spy_llm(ra, "coder")
    router = ra.LLMRouter(default=default, planner=planner, coder=coder)

    assert router.for_role("planner") is planner
    assert router.for_role("coder") is coder
    # analyzer / writer / literature / repair have no per-role client → fall back.
    assert router.for_role("analyzer") is default
    assert router.for_role("writer") is default
    assert router.for_role("literature") is default
    assert router.for_role("repair") is default


def test_router_can_isolate_repair_from_initial_coder(ra):
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    coder = ScriptedMockLLMClient(["coder"])
    repair = ScriptedMockLLMClient(["repair"])
    router = ra.LLMRouter(default=coder, coder=coder, repair=repair)

    assert router.for_role("coder") is coder
    assert router.for_role("repair") is repair


def test_router_for_role_unknown_role(ra):
    router = ra.LLMRouter(default=_spy_llm(ra, "d"))
    with pytest.raises(KeyError):
        router.for_role("not_a_role")


def test_router_no_default_partial_roles_raises(ra):
    """If a role isn't configured and no default is set, asking for it raises."""
    router = ra.LLMRouter(planner=_spy_llm(ra, "p"))
    assert router.for_role("planner").spy_name == "p"
    with pytest.raises(KeyError):
        router.for_role("coder")


def test_router_iter_clients_dedupes(ra):
    shared = _spy_llm(ra, "shared")
    other = _spy_llm(ra, "other")
    router = ra.LLMRouter(default=shared, planner=shared, coder=other, writer=shared)
    seen = list(router.iter_clients())
    assert shared in seen and other in seen
    assert len(seen) == 2  # deduped by id()


def test_router_complete_routes_to_default(ra):
    default = _spy_llm(ra, "default")
    router = ra.LLMRouter(default=default, planner=_spy_llm(ra, "p"))
    from easyicu.research_agent.providers.llm import LLMMessage

    out = router.complete([LLMMessage(role="user", content="hello")])
    assert isinstance(out, str)
    assert _spy_prompts(default) == ["hello"]


def test_router_complete_without_default_raises(ra):
    router = ra.LLMRouter(planner=_spy_llm(ra, "p"))
    from easyicu.research_agent.providers.llm import LLMMessage

    with pytest.raises(RuntimeError):
        router.complete([LLMMessage(role="user", content="hello")])


# ---------------------------------------------------------------------------
# resolve_role_client helper
# ---------------------------------------------------------------------------


def test_resolve_role_client_with_router(ra):
    from easyicu.research_agent.providers.llm import resolve_role_client

    planner = _spy_llm(ra, "planner")
    default = _spy_llm(ra, "default")
    router = ra.LLMRouter(default=default, planner=planner)
    assert resolve_role_client(router, "planner") is planner
    assert resolve_role_client(router, "coder") is default


def test_resolve_role_client_with_plain_client(ra):
    from easyicu.research_agent.providers.llm import resolve_role_client

    plain = _spy_llm(ra, "plain")
    # No ``for_role`` → always returns the same client (legacy semantics).
    assert resolve_role_client(plain, "planner") is plain
    assert resolve_role_client(plain, "coder") is plain
    assert resolve_role_client(plain, "writer") is plain


def test_resolve_role_client_with_none(ra):
    from easyicu.research_agent.providers.llm import resolve_role_client

    assert resolve_role_client(None, "planner") is None


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_pipeline_with_router_routes_each_agent_to_its_role(
    ra, synthetic_cohort, tmp_path: Path, monkeypatch
):
    """End-to-end: a router with distinct planner/coder/writer/analyzer
    spies should route each agent's calls to the matching client."""
    _isolate_article_suite_contract(monkeypatch)
    context = _pipeline_context(synthetic_cohort)
    planner = _spy_llm(ra, "planner", context=context)
    coder = _spy_llm(ra, "coder", context=context)
    analyzer = _spy_llm(ra, "analyzer", context=context)
    writer = _spy_llm(ra, "writer", context=context)
    router = ra.LLMRouter(
        default=_spy_llm(ra, "default", context=context),
        planner=planner,
        coder=coder,
        analyzer=analyzer,
        writer=writer,
    )

    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=router)
    result = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="router_test",
        database="synthetic",
        target_outcome="death",
        force_writer_probe=True,
    )
    assert result.evidence_count > 0

    def _names(spy):
        return [prompt[:80] for prompt in _spy_prompts(spy)]

    assert planner.calls, f"planner spy not called: {_names(coder)}"
    assert coder.calls, f"coder spy not called: {_names(planner)}"
    # The explicit writer probe must use the writer route. Analyzer is
    # fail-stop dependent and is legitimately absent when execution blocks.
    assert writer.calls, "writer spy not called"
    if analyzer.calls:
        assert any(
            "INTERPRET THE RESULTS" in prompt.upper()
            for prompt in _spy_prompts(analyzer)
        )
    else:
        assert not any(
            "INTERPRET THE RESULTS" in prompt.upper()
            for spy in (planner, coder, writer)
            for prompt in _spy_prompts(spy)
        )

    # Sanity: planner prompt should contain the planner anchor phrase,
    # coder prompt the code anchor — i.e. routing isn't crossed.
    assert any(
        "ICU-AWARE RESEARCH PLAN" in prompt.upper() for prompt in _spy_prompts(planner)
    )
    assert any(
        "WRITE THE PYTHON CODE" in prompt.upper() for prompt in _spy_prompts(coder)
    )


def test_pipeline_backwards_compat_plain_client(
    ra, synthetic_cohort, tmp_path: Path, monkeypatch
):
    """Passing a plain MockLLMClient (no for_role) must keep working."""
    _isolate_article_suite_contract(monkeypatch)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=ra.MockLLMClient(context=_pipeline_context(synthetic_cohort)),
    )
    result = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="legacy_compat",
        database="synthetic",
        target_outcome="death",
    )
    assert result.evidence_count > 0
