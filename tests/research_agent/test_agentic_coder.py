"""Unit tests for the altitude-2a agentic coder.

Fully offline: the CLI subprocess and PATH lookup are patched. The key
invariant under test is that delegation returns the CLI-authored *script*
(which the instrumented runtime later executes + evidence-binds), and that it
degrades to the wrapped LLM coder whenever the CLI is unavailable.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import easyicu.research_agent.agentic_coder as ac_mod
from easyicu.research_agent.agentic_coder import (
    AgenticCoderAgent,
    maybe_wrap_coder,
)


class _FakeCoder:
    """Stand-in for CoderAgent: records whether the fallback path was taken."""

    def __init__(self):
        self.run_called = False
        self.repair_called = False

    def run(self, *, context, step):  # noqa: ANN001
        self.run_called = True
        return "# fallback script from LLM coder\n"

    def repair(self, *, context, step, code, run_log, attempt):  # noqa: ANN001
        self.repair_called = True
        return code + "\n# repaired\n"


def _step():
    return SimpleNamespace(
        step_id="assoc",
        intent="association",
        inputs={},
        expected_outputs={},
        method=None,
    )


def _ctx(ra):
    return ra.ResearchContext(
        research_question="Association between X and mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo", database="synthetic", n_stays=10, n_patients=10
        ),
        variables=[],
    )


def _make_cli_available(monkeypatch, backend="codex"):
    monkeypatch.setattr(ac_mod, "cli_backend_available", lambda b: b == backend)


# ---------------------------------------------------------------------------
# Degradation: CLI absent -> wrapped LLM coder
# ---------------------------------------------------------------------------

def test_falls_back_when_cli_unavailable(ra, monkeypatch):
    monkeypatch.setattr(ac_mod, "cli_backend_available", lambda b: False)
    fallback = _FakeCoder()
    agent = AgenticCoderAgent(fallback, backend="codex")
    out = agent.run(context=ra.MockLLMClient().context if False else None, step=_step())
    assert fallback.run_called
    assert agent.last_delegation_used is False
    assert "fallback script" in out


def test_falls_back_when_cli_writes_no_script(ra, monkeypatch):
    _make_cli_available(monkeypatch)
    # subprocess "succeeds" but never creates analysis.py
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="done", stderr=""),
    )
    fallback = _FakeCoder()
    agent = AgenticCoderAgent(fallback, backend="codex")
    out = agent.run(context=_ctx(ra), step=_step())
    assert fallback.run_called
    assert agent.last_delegation_used is False
    assert "fallback script" in out


# ---------------------------------------------------------------------------
# Delegation: CLI authors a script -> we return THAT script (not numbers)
# ---------------------------------------------------------------------------

def test_returns_cli_authored_script(ra, monkeypatch):
    _make_cli_available(monkeypatch)

    authored = "import pandas as pd\nprint('analysis')\n"

    def _fake_run(argv, **kwargs):
        # Emulate the CLI writing the final script into its workdir (cwd).
        Path(kwargs["cwd"], "analysis.py").write_text(authored, encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ran ok, result=42", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    # No method-compatibility violations for this snippet.
    import easyicu.research_agent.method_compatibility as mc
    monkeypatch.setattr(
        mc,
        "detect_forbidden_pattern_usage",
        lambda code, ctx, step=None: [],
    )

    fallback = _FakeCoder()
    agent = AgenticCoderAgent(fallback, backend="codex")
    out = agent.run(context=_ctx(ra), step=_step())

    assert agent.last_delegation_used is True
    assert fallback.run_called is False
    # We return the authored SCRIPT, never the CLI's printed "result=42".
    assert out.strip() == authored.strip()
    assert "result=42" not in out


def test_cohort_env_is_passed_through_to_subprocess(ra, monkeypatch):
    _make_cli_available(monkeypatch)
    monkeypatch.setenv("COHORT_PARQUET", "/tmp/cohort.parquet")
    captured = {}

    def _fake_run(argv, **kwargs):
        captured["env"] = kwargs.get("env", {})
        Path(kwargs["cwd"], "analysis.py").write_text("x=1\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    import easyicu.research_agent.method_compatibility as mc
    monkeypatch.setattr(
        mc,
        "detect_forbidden_pattern_usage",
        lambda code, ctx, step=None: [],
    )

    AgenticCoderAgent(_FakeCoder(), backend="codex").run(context=_ctx(ra), step=_step())
    assert captured["env"].get("COHORT_PARQUET") == "/tmp/cohort.parquet"


def test_agentic_prompt_forbids_undeclared_figures(ra):
    prompt = AgenticCoderAgent(_FakeCoder(), backend="codex")._build_prompt(
        _ctx(ra), _step()
    )

    assert "DECLARED OUTPUT SCOPE (binding)" in prompt
    assert "declares no figure product" in prompt
    assert "Do not render, save, or register figures" in prompt


def test_compatibility_violation_routes_through_fallback_repair(ra, monkeypatch):
    _make_cli_available(monkeypatch)

    def _fake_run(argv, **kwargs):
        Path(kwargs["cwd"], "analysis.py").write_text("bad = 1\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    import easyicu.research_agent.method_compatibility as mc
    calls = {"n": 0}

    def _detect(code, ctx, step=None):
        # Violate once, then pass after repair appended its marker.
        if "# repaired" in code:
            return []
        calls["n"] += 1
        return [{"pattern": "kmeans_on_ordinal"}]

    monkeypatch.setattr(mc, "detect_forbidden_pattern_usage", _detect)
    monkeypatch.setattr(mc, "format_violation_message", lambda v: "violation!")

    fallback = _FakeCoder()
    out = AgenticCoderAgent(fallback, backend="codex").run(context=_ctx(ra), step=_step())
    assert fallback.repair_called is True
    assert "# repaired" in out


# ---------------------------------------------------------------------------
# maybe_wrap_coder opt-in gate (default OFF)
# ---------------------------------------------------------------------------

def test_maybe_wrap_off_by_default():
    coder = _FakeCoder()
    assert maybe_wrap_coder(coder, env={}) is coder


def test_maybe_wrap_ignores_unknown_backend():
    coder = _FakeCoder()
    assert maybe_wrap_coder(coder, env={"EASYICU_AGENTIC_CODER_BACKEND": "gpt"}) is coder


def test_maybe_wrap_enables_for_codex():
    coder = _FakeCoder()
    wrapped = maybe_wrap_coder(coder, env={"EASYICU_AGENTIC_CODER_BACKEND": "codex"})
    assert isinstance(wrapped, AgenticCoderAgent)
    assert wrapped.backend == "codex"
    assert wrapped.fallback is coder
