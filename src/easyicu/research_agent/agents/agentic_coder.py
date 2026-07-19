"""Altitude-2a coder — delegate script *authoring + self-repair* to a local
coding-agent CLI (Codex / Claude Code), but keep execution + evidence binding
house-owned.

This is the step beyond altitude-1 (CLI as a plain ``complete()`` brain). Here
the CLI runs its own write -> run -> fix loop in a sandbox: it can actually
execute the analysis against the cohort parquet (read via ``COHORT_PARQUET``),
see the real error, and repair — which the LLM ``CoderAgent`` cannot do because
it never runs code. The CLI writes the final working script to a known file,
which we read back and hand to the instrumented runtime unchanged.

**The provenance invariant is preserved by construction.** We return only the
*script*; we never trust the numbers the CLI printed during its own run. The
runtime re-executes that script in the instrumented namespace and binds every
value as a ``NumericClaim`` exactly as with the LLM coder. So a stronger engine
gives us better *code*, not unverified results. See the engine-agnostic gate in
``manuscript_post.bind_numeric_values``.

**Opt-in and degrading.** Nothing constructs this by default. When wired in
(env ``EASYICU_AGENTIC_CODER_BACKEND`` at the single ``pipeline_execute``
construction site) it falls back to the wrapped LLM ``CoderAgent`` whenever the
CLI backend is unavailable or fails to produce a script — the same capability
ladder as :func:`easyicu.research_agent.providers.llm.build_llm_client`.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List

from ..providers.llm import cli_backend_available
from ..schema import AnalysisStep, ResearchContext
from ..authority.coder_authority import HostCoderAuthority

# The filename the CLI is told to save its final, working script as. We read
# this file back rather than parsing the CLI's chat output for a code fence —
# deterministic and unambiguous.
_SCRIPT_NAME = "analysis.py"

_DEFAULT_TIMEOUT_SECONDS = 600.0


class AgenticCoderAgent:
    """Coder that delegates authoring to a local CLI agent, returning a script."""

    def __init__(
        self,
        fallback: Any,
        *,
        backend: str = "codex",
        cohort_env: str = "COHORT_PARQUET",
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.fallback = fallback
        self.backend = str(backend or "").strip().lower()
        self.cohort_env = cohort_env
        self.timeout = float(timeout)
        # Mirror CoderAgent's public attributes so callers that inspect the
        # coder (compatibility telemetry) keep working when this is swapped in.
        self.last_compatibility_violations: List[dict] = []
        self.last_compatibility_repair_attempts: int = 0
        self.last_delegation_used: bool = False

    # -- prompt -------------------------------------------------------------
    def _build_prompt(self, context: ResearchContext, step: AnalysisStep) -> str:
        from .core import (
            _declared_output_scope_contract,
            _format_context,
            _typed_input_scope_contract,
        )
        from ..trajectory_contract import trajectory_phenotyping_code_contract
        from ..trajectory_plan_contract import trajectory_role_code_contract

        return (
            f"You are authoring ONE self-contained Python analysis script for "
            f"step {step.step_id} of an ICU research pipeline.\n\n"
            f"Step intent: {step.intent}\n"
            f"Step inputs: {step.inputs}\n"
            f"Expected outputs: {step.expected_outputs}\n"
            f"Method: {step.method or '(unspecified — choose conservatively)'}\n\n"
            "RULES:\n"
            f"- Read the cohort from os.environ['{self.cohort_env}'] (a parquet "
            "file). Do NOT inline or fabricate data.\n"
            "- Do NOT hardcode any result value. Every reported number must be "
            "computed from the cohort at run time.\n"
            f"- Write the final, working script to a file named '{_SCRIPT_NAME}' "
            "in the current directory, then run it and fix any errors until it "
            "executes cleanly and produces the expected outputs.\n"
            "- The script must be runnable standalone with `python "
            f"{_SCRIPT_NAME}`; keep all imports inside it.\n\n"
            + _declared_output_scope_contract(step)
            + _typed_input_scope_contract(step)
            + "\n"
            "RESEARCH CONTEXT:\n"
            + _format_context(context)
            + trajectory_phenotyping_code_contract(context=context, step=step)
            + trajectory_role_code_contract(context=context, step=step)
        )

    def _argv(self, workdir: str) -> List[str]:
        # codex needs workspace-write to author + run the script in workdir; the
        # cohort parquet is only read, so it stays outside this writable dir.
        if self.backend == "codex":
            return [
                "codex",
                "exec",
                "--sandbox",
                "workspace-write",
                "--skip-git-repo-check",
                "--color",
                "never",
                "-C",
                workdir,
            ]
        if self.backend == "claude":
            return [
                "claude",
                "-p",
                "--output-format",
                "text",
                "--permission-mode",
                "acceptEdits",
                "--add-dir",
                workdir,
            ]
        raise ValueError(f"Unsupported agentic coder backend: {self.backend!r}")

    # -- main ---------------------------------------------------------------
    def run(
        self,
        *,
        context: ResearchContext,
        step: AnalysisStep,
        **authority_hooks: Any,
    ) -> str:
        self.last_delegation_used = False
        provider_budget = authority_hooks.pop("provider_budget", None)
        fallback_kwargs = dict(authority_hooks)
        if provider_budget is not None:
            fallback_kwargs["provider_budget"] = provider_budget
        host_authority = fallback_kwargs.get("host_authority")
        if (
            isinstance(host_authority, HostCoderAuthority)
            and host_authority.attachments
        ):
            return self.fallback.run(
                context=context,
                step=step,
                **fallback_kwargs,
            )
        if provider_budget is not None:
            # The local CLI is not yet a receipt-aware provider transport.  In
            # a capsule-backed pipeline, delegating before the durable initial
            # reservation would make a crash repeat the external CLI attempt,
            # while an empty CLI result followed by the fallback would record
            # only one of two real attempts.  Keep standalone delegation
            # available, but route authority-bound generation through the
            # receipt-aware fallback until the CLI has its own adapter.
            if authority_hooks.get("initial_generation_binding") is not None:
                return self.fallback.run(
                    context=context,
                    step=step,
                    **fallback_kwargs,
                )
            if provider_budget.initial_generation_resume_status() == ("unpaid_pending"):
                return self.fallback.run(
                    context=context,
                    step=step,
                    **fallback_kwargs,
                )
        if not cli_backend_available(self.backend):
            return self.fallback.run(
                context=context,
                step=step,
                **fallback_kwargs,
            )

        script = self._delegate(context, step)
        if not script:
            # CLI produced nothing usable — degrade to the LLM coder.
            return self.fallback.run(
                context=context,
                step=step,
                **fallback_kwargs,
            )

        self.last_delegation_used = True
        return self._enforce_compatibility(context, step, script)

    def _delegate(self, context: ResearchContext, step: AnalysisStep) -> str:
        prompt = self._build_prompt(context, step)
        with tempfile.TemporaryDirectory(prefix="easyicu-agentic-coder-") as workdir:
            env = dict(os.environ)  # pass COHORT_PARQUET et al. through unchanged
            try:
                subprocess.run(
                    self._argv(workdir),
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=workdir,
                    env=env,
                )
            except (subprocess.TimeoutExpired, OSError):
                return ""
            script_path = Path(workdir) / _SCRIPT_NAME
            if not script_path.is_file():
                return ""
            try:
                code = script_path.read_text(encoding="utf-8")
            except OSError:
                return ""
        from .core import _strip_code_fence

        return _strip_code_fence(code.strip())

    def repair(self, **kwargs: Any) -> str:
        """Keep pipeline repair accounting on the wrapped Coder transport."""

        return self.fallback.repair(**kwargs)

    def _enforce_compatibility(
        self, context: ResearchContext, step: AnalysisStep, code: str
    ) -> str:
        """Record violations; the central repair coordinator owns any repair."""
        from ..gates.method_compatibility import detect_forbidden_pattern_usage

        self.last_compatibility_repair_attempts = 0
        self.last_compatibility_violations = detect_forbidden_pattern_usage(
            code,
            context,
            step,
        )
        return code


def maybe_wrap_coder(
    coder: Any, *, env: "os._Environ[str] | dict[str, str] | None" = None
) -> Any:
    """Return an :class:`AgenticCoderAgent` wrapping ``coder`` iff opted in.

    Controlled by ``EASYICU_AGENTIC_CODER_BACKEND`` (``codex`` / ``claude``).
    When unset, returns ``coder`` unchanged, so the default pipeline behaviour
    and every existing test are untouched. When set but the CLI is not
    installed, the wrapper still degrades to ``coder`` at call time.
    """
    source = os.environ if env is None else env
    backend = str(source.get("EASYICU_AGENTIC_CODER_BACKEND", "") or "").strip().lower()
    if backend not in {"codex", "claude"}:
        return coder
    return AgenticCoderAgent(coder, backend=backend)
