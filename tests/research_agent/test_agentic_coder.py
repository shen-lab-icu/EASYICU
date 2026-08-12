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

import easyicu.research_agent.agents.agentic_coder as ac_mod
from easyicu.research_agent.agents.agentic_coder import (
    AgenticCoderAgent,
    maybe_wrap_coder,
)
from easyicu.research_agent.authority.coder_authority import HostCoderAuthority


class _FakeCoder:
    """Stand-in for CoderAgent: records whether the fallback path was taken."""

    def __init__(self):
        self.run_called = False
        self.repair_called = False
        self.run_kwargs = {}

    def run(self, *, context, step, **kwargs):  # noqa: ANN001
        self.run_called = True
        self.run_kwargs = dict(kwargs)
        return "# fallback script from LLM coder\n"

    def repair(self, *, context, step, code, run_log, attempt):  # noqa: ANN001
        self.repair_called = True
        return code + "\n# repaired\n"


class _FakeBudget:
    def __init__(self, status=None):
        self.status = status

    def initial_generation_resume_status(self):
        return self.status


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
    monkeypatch.setenv("EASYICU_ALLOW_EXTERNAL_LLM", "1")


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
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="done", stderr=""),
    )
    fallback = _FakeCoder()
    agent = AgenticCoderAgent(fallback, backend="codex")
    out = agent.run(context=_ctx(ra), step=_step())
    assert fallback.run_called
    assert agent.last_delegation_used is False
    assert "fallback script" in out


def test_empty_cli_fallback_preserves_provider_budget(ra, monkeypatch):
    _make_cli_available(monkeypatch)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    budget = _FakeBudget()
    fallback = _FakeCoder()

    AgenticCoderAgent(fallback, backend="codex").run(
        context=_ctx(ra),
        step=_step(),
        provider_budget=budget,
    )

    assert fallback.run_kwargs["provider_budget"] is budget


def test_host_authority_forces_receipt_aware_fallback_without_losing_budget(
    ra, monkeypatch
):
    _make_cli_available(monkeypatch)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("host authority must not reach the direct CLI")
        ),
    )
    budget = _FakeBudget()
    authority = HostCoderAuthority().append("exact host binding")
    fallback = _FakeCoder()

    AgenticCoderAgent(fallback, backend="codex").run(
        context=_ctx(ra),
        step=_step(),
        provider_budget=budget,
        host_authority=authority,
    )

    assert fallback.run_called
    assert fallback.run_kwargs["provider_budget"] is budget
    assert fallback.run_kwargs["host_authority"] == authority


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
    import easyicu.research_agent.gates.method_compatibility as mc

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


def test_none_capsule_hooks_do_not_disable_cli_delegation(ra, monkeypatch):
    _make_cli_available(monkeypatch)
    authored = "print('agentic')\n"

    def _fake_run(argv, **kwargs):
        Path(kwargs["cwd"], "analysis.py").write_text(authored, encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    import easyicu.research_agent.gates.method_compatibility as mc

    monkeypatch.setattr(
        mc,
        "detect_forbidden_pattern_usage",
        lambda code, ctx, step=None: [],
    )
    fallback = _FakeCoder()
    agent = AgenticCoderAgent(fallback, backend="codex")

    out = agent.run(
        context=_ctx(ra),
        step=_step(),
        provider_budget=_FakeBudget(),
        initial_generation_binding=None,
        persist_candidate=None,
        on_initial_reserved=None,
        on_initial_candidate=None,
        reserve_compatibility_repair=None,
        on_repair_candidate=None,
    )

    assert out.strip() == authored.strip()
    assert agent.last_delegation_used is True
    assert fallback.run_called is False


def test_unpaid_fallback_reservation_cannot_switch_to_cli(ra, monkeypatch):
    _make_cli_available(monkeypatch)
    fallback = _FakeCoder()
    budget = _FakeBudget("unpaid_pending")
    agent = AgenticCoderAgent(fallback, backend="codex")
    monkeypatch.setattr(
        agent,
        "_delegate",
        lambda context, step: (_ for _ in ()).throw(
            AssertionError("CLI must not replace a pending provider reservation")
        ),
    )

    agent.run(
        context=_ctx(ra),
        step=_step(),
        provider_budget=budget,
        initial_generation_binding={"schema_version": "test"},
    )

    assert fallback.run_called is True
    assert fallback.run_kwargs["provider_budget"] is budget
    assert fallback.run_kwargs["initial_generation_binding"] == {
        "schema_version": "test"
    }


def test_capsule_transport_uses_receipt_aware_fallback_before_cli(ra, monkeypatch):
    _make_cli_available(monkeypatch)
    fallback = _FakeCoder()
    budget = _FakeBudget("absent")
    agent = AgenticCoderAgent(fallback, backend="codex")
    monkeypatch.setattr(
        agent,
        "_delegate",
        lambda context, step: (_ for _ in ()).throw(
            AssertionError("untracked CLI must not run in capsule mode")
        ),
    )

    agent.run(
        context=_ctx(ra),
        step=_step(),
        provider_budget=budget,
        initial_generation_binding={"schema_version": "test"},
    )

    assert fallback.run_called is True
    assert fallback.run_kwargs["provider_budget"] is budget
    assert agent.last_delegation_used is False


def test_cohort_env_is_passed_through_to_subprocess(ra, monkeypatch):
    _make_cli_available(monkeypatch)
    monkeypatch.setenv("COHORT_PARQUET", "/tmp/cohort.parquet")
    captured = {}

    def _fake_run(argv, **kwargs):
        captured["env"] = kwargs.get("env", {})
        Path(kwargs["cwd"], "analysis.py").write_text("x=1\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    import easyicu.research_agent.gates.method_compatibility as mc

    monkeypatch.setattr(
        mc,
        "detect_forbidden_pattern_usage",
        lambda code, ctx, step=None: [],
    )

    AgenticCoderAgent(_FakeCoder(), backend="codex").run(context=_ctx(ra), step=_step())
    assert captured["env"].get("COHORT_PARQUET") == "/tmp/cohort.parquet"


def test_agentic_coder_does_not_delegate_without_external_opt_in(ra, monkeypatch):
    monkeypatch.setattr(ac_mod, "cli_backend_available", lambda backend: True)
    monkeypatch.delenv("EASYICU_ALLOW_EXTERNAL_LLM", raising=False)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unauthorized standalone CLI must not launch")
        ),
    )
    fallback = _FakeCoder()

    result = AgenticCoderAgent(fallback, backend="codex").run(
        context=_ctx(ra), step=_step()
    )

    assert fallback.run_called is True
    assert "fallback script" in result


def test_agentic_coder_drops_unrelated_parent_secrets(ra, monkeypatch):
    _make_cli_available(monkeypatch)
    monkeypatch.setenv("COHORT_PARQUET", "/tmp/cohort.parquet")
    monkeypatch.setenv("OPENAI_API_KEY", "required-backend-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-leak")
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-leak")
    monkeypatch.setenv("DATABASE_URL", "postgresql://must-not-leak")
    captured = {}

    def _fake_run(argv, **kwargs):
        captured["env"] = kwargs["env"]
        Path(kwargs["cwd"], "analysis.py").write_text("x=1\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    import easyicu.research_agent.gates.method_compatibility as mc

    monkeypatch.setattr(
        mc,
        "detect_forbidden_pattern_usage",
        lambda code, ctx, step=None: [],
    )

    AgenticCoderAgent(_FakeCoder(), backend="codex").run(
        context=_ctx(ra), step=_step()
    )

    assert captured["env"]["COHORT_PARQUET"] == "/tmp/cohort.parquet"
    assert captured["env"]["OPENAI_API_KEY"] == "required-backend-secret"
    assert "AWS_SECRET_ACCESS_KEY" not in captured["env"]
    assert "GITHUB_TOKEN" not in captured["env"]
    assert "DATABASE_URL" not in captured["env"]


def test_agentic_prompt_forbids_undeclared_figures(ra):
    prompt = AgenticCoderAgent(_FakeCoder(), backend="codex")._build_prompt(
        _ctx(ra), _step()
    )

    assert "DECLARED OUTPUT SCOPE (binding)" in prompt
    assert "declares no figure product" in prompt
    assert "Do not render, save, or register figures" in prompt


def test_compatibility_violation_defers_to_central_repair_gate(ra, monkeypatch):
    _make_cli_available(monkeypatch)

    def _fake_run(argv, **kwargs):
        Path(kwargs["cwd"], "analysis.py").write_text("bad = 1\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    import easyicu.research_agent.gates.method_compatibility as mc

    calls = {"n": 0}

    def _detect(code, ctx, step=None):
        calls["n"] += 1
        return [{"pattern": "kmeans_on_ordinal"}]

    monkeypatch.setattr(mc, "detect_forbidden_pattern_usage", _detect)
    fallback = _FakeCoder()
    agent = AgenticCoderAgent(fallback, backend="codex")
    out = agent.run(context=_ctx(ra), step=_step())
    assert fallback.repair_called is False
    assert out == "bad = 1"
    assert agent.last_compatibility_violations == [{"pattern": "kmeans_on_ordinal"}]


# ---------------------------------------------------------------------------
# maybe_wrap_coder opt-in gate (default OFF)
# ---------------------------------------------------------------------------


def test_maybe_wrap_off_by_default():
    coder = _FakeCoder()
    assert maybe_wrap_coder(coder, env={}) is coder


def test_maybe_wrap_ignores_unknown_backend():
    coder = _FakeCoder()
    assert (
        maybe_wrap_coder(coder, env={"EASYICU_AGENTIC_CODER_BACKEND": "gpt"}) is coder
    )


def test_maybe_wrap_enables_for_codex():
    coder = _FakeCoder()
    wrapped = maybe_wrap_coder(coder, env={"EASYICU_AGENTIC_CODER_BACKEND": "codex"})
    assert isinstance(wrapped, AgenticCoderAgent)
    assert wrapped.backend == "codex"
    assert wrapped.fallback is coder
