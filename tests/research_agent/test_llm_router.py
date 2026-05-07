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


def _spy_llm(ra, name: str):
    """A tiny stub LLM that records every ``complete`` call's prompt."""
    from easyicu.research_agent.llm import MockLLMClient

    class _Spy(MockLLMClient):
        def __init__(self):
            super().__init__(context=None)
            self.calls = []
            self.spy_name = name

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            self.calls.append({
                "name": self.spy_name,
                "user_prompt": next(
                    (m.content for m in reversed(messages) if m.role == "user"), ""),
            })
            return super().complete(messages, max_tokens=max_tokens, temperature=temperature)

    return _Spy()


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
    # analyzer / writer / literature have no per-role client → fall back.
    assert router.for_role("analyzer") is default
    assert router.for_role("writer") is default
    assert router.for_role("literature") is default


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
    from easyicu.research_agent.llm import LLMMessage
    out = router.complete([LLMMessage(role="user", content="hello")])
    assert isinstance(out, str)
    assert any(c["name"] == "default" for c in default.calls)


def test_router_complete_without_default_raises(ra):
    router = ra.LLMRouter(planner=_spy_llm(ra, "p"))
    from easyicu.research_agent.llm import LLMMessage
    with pytest.raises(RuntimeError):
        router.complete([LLMMessage(role="user", content="hello")])


# ---------------------------------------------------------------------------
# resolve_role_client helper
# ---------------------------------------------------------------------------


def test_resolve_role_client_with_router(ra):
    from easyicu.research_agent.llm import resolve_role_client
    planner = _spy_llm(ra, "planner")
    default = _spy_llm(ra, "default")
    router = ra.LLMRouter(default=default, planner=planner)
    assert resolve_role_client(router, "planner") is planner
    assert resolve_role_client(router, "coder") is default


def test_resolve_role_client_with_plain_client(ra):
    from easyicu.research_agent.llm import resolve_role_client
    plain = _spy_llm(ra, "plain")
    # No ``for_role`` → always returns the same client (legacy semantics).
    assert resolve_role_client(plain, "planner") is plain
    assert resolve_role_client(plain, "coder") is plain
    assert resolve_role_client(plain, "writer") is plain


def test_resolve_role_client_with_none(ra):
    from easyicu.research_agent.llm import resolve_role_client
    assert resolve_role_client(None, "planner") is None


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_pipeline_with_router_routes_each_agent_to_its_role(ra, synthetic_cohort,
                                                            tmp_path: Path):
    """End-to-end: a router with distinct planner/coder/writer/analyzer
    spies should route each agent's calls to the matching client."""
    planner = _spy_llm(ra, "planner")
    coder = _spy_llm(ra, "coder")
    analyzer = _spy_llm(ra, "analyzer")
    writer = _spy_llm(ra, "writer")
    router = ra.LLMRouter(
        default=_spy_llm(ra, "default"),
        planner=planner, coder=coder, analyzer=analyzer, writer=writer,
    )

    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=router)
    result = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="router_test",
        database="synthetic",
        target_outcome="death",
    )
    assert result.evidence_count > 0

    def _names(spy):
        return [c["user_prompt"][:80] for c in spy.calls]

    assert planner.calls, f"planner spy not called: {_names(coder)}"
    assert coder.calls, f"coder spy not called: {_names(planner)}"
    # writer / analyzer should have been called too
    assert writer.calls, "writer spy not called"
    assert analyzer.calls, "analyzer spy not called"

    # Sanity: planner prompt should contain the planner anchor phrase,
    # coder prompt the code anchor — i.e. routing isn't crossed.
    assert any("ICU-AWARE RESEARCH PLAN" in c["user_prompt"].upper() for c in planner.calls)
    assert any("WRITE THE PYTHON CODE" in c["user_prompt"].upper() for c in coder.calls)


def test_pipeline_backwards_compat_plain_client(ra, synthetic_cohort, tmp_path: Path):
    """Passing a plain MockLLMClient (no for_role) must keep working."""
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    result = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="legacy_compat",
        database="synthetic",
        target_outcome="death",
    )
    assert result.evidence_count > 0
