"""Invariant: NO engine can bypass NumericClaim binding.

These canaries make concern #2 explicit and permanent. A local coding-agent
CLI (Codex / Claude Code) or any future "stronger" brain may be plugged in as
the manuscript producer, but the value-level evidence gate
(`bind_numeric_values` in STRICT mode) does not care which engine wrote the
text — every printed number must trace to a registered claim or be rejected.

If someone later adds an engine-specific "trusted" path that lets a provider's
numbers skip binding, one of these tests must fail.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Fake "engines": each satisfies the LLMClient.complete(...) -> str contract and
# returns a manuscript string. They stand in for mock / API / CLI brains.
# ---------------------------------------------------------------------------

class _FixedTextEngine:
    """A minimal LLMClient whose completion is a fixed manuscript string."""

    def __init__(self, name: str, manuscript: str) -> None:
        self.name = name
        self._manuscript = manuscript

    def complete(self, messages, **_kwargs) -> str:  # noqa: ANN001
        return self._manuscript


def _cli_engine_returning(manuscript: str, monkeypatch):
    """A *real* CLIAgentLLMClient whose subprocess is patched to print text.

    This proves the CLI-agent path's output is subject to the same gate — not
    a hand-rolled stub.
    """
    import shutil
    import subprocess

    from easyicu.research_agent.llm import CLIAgentLLMClient

    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/" + cmd)
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kw: SimpleNamespace(returncode=0, stdout=manuscript, stderr=""),
    )
    return CLIAgentLLMClient(backend="codex")


_HALLUCINATED = "The observed odds ratio was 1.42, but the engine invented 999."
_CLEAN = "The observed odds ratio was 1.42 across the cohort."


def _store_with_or(ra, tmp_path: Path, mode: str):
    store = ra.EvidenceStore(root=tmp_path, enforcement_mode=mode)
    store.register_step_summary_numerics(
        step_id="assoc",
        evidence_id="evid_assoc",
        summary={"primary_or": 1.42},
    )
    return store


@pytest.mark.parametrize("engine_name", ["mock-like", "api-like", "codex-cli"])
def test_hallucinated_number_blocked_in_strict_for_every_engine(
    ra, tmp_path: Path, monkeypatch, engine_name: str
):
    from easyicu.research_agent.llm import LLMMessage
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    if engine_name == "codex-cli":
        engine = _cli_engine_returning(_HALLUCINATED, monkeypatch)
    else:
        engine = _FixedTextEngine(engine_name, _HALLUCINATED)

    # The engine "writes" the manuscript — exactly how a real writer role calls it.
    manuscript = engine.complete([LLMMessage(role="user", content="write results")])

    store = _store_with_or(ra, tmp_path, "strict")
    with pytest.raises(ra.EvidenceEnforcementError) as exc_info:
        bind_numeric_values(manuscript, evidence=store)
    assert exc_info.value.detail["untraced"] == ["999"]


@pytest.mark.parametrize("engine_name", ["mock-like", "api-like", "codex-cli"])
def test_registered_number_passes_for_every_engine(
    ra, tmp_path: Path, monkeypatch, engine_name: str
):
    from easyicu.research_agent.llm import LLMMessage
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    if engine_name == "codex-cli":
        engine = _cli_engine_returning(_CLEAN, monkeypatch)
    else:
        engine = _FixedTextEngine(engine_name, _CLEAN)

    manuscript = engine.complete([LLMMessage(role="user", content="write results")])

    store = _store_with_or(ra, tmp_path, "strict")
    bound, binding_map, untraced = bind_numeric_values(manuscript, evidence=store)
    assert untraced == []
    assert any(c.source_field == "primary_or" for c in binding_map.values())


def test_invariant_is_documented_at_the_boundary():
    """The engine-agnostic invariant must stay written where it is enforced."""
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    doc = bind_numeric_values.__doc__ or ""
    assert "Engine-agnostic provenance invariant" in doc
    assert "bypass this gate" in doc
