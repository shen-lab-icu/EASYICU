"""Local coding-agent CLI as an OpenAI-compatible chat backend.

This is the EasyICU analogue of open-design's "drive a local coding-agent CLI"
idea (its Agentic Model Router). Instead of an API key + HTTP endpoint, it
shells out to a pre-authenticated ``claude`` or ``codex`` CLI that the user
already has installed, and treats the CLI's printed answer as a single chat
completion.

Safety posture — this matches the EasyICU advisory / human-confirmed
invariants documented in ``CLAUDE.md``:

- The CLI is used as a **text generator only**. It runs in print / exec mode,
  it is **not** granted write/execute tool permissions, and it runs with a
  read-only sandbox in a throwaway temp cwd. Nothing it does can touch the
  repo, the prepared data, or the evidence store.
- Every write that changes execution state still flows through the classic
  engine (``copilot_engine.run_copilot_step``). This client only ever returns
  advisory text, exactly like the OpenAI-compatible path it stands in for.
- It is still an **external model call** (the CLI talks to Anthropic / OpenAI
  servers), so it stays behind the sidebar AI opt-in gate just like every
  other real provider — it is NOT exempt the way ``MockLLMClient`` is.

The classes below intentionally mimic the small slice of the ``openai`` client
surface that ``llm_chat`` relies on:

- ``client.chat.completions.create(model=, messages=, stream=, ...)``
- ``client.with_options(timeout=, max_retries=)`` (returns a configured copy)
- non-stream result: ``resp.choices[0].message.content``
- stream result: an iterable of chunks with ``chunk.choices[0].delta.content``
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence


# Provider keys registered in ``llm_config.PROVIDERS``. Anything starting with
# this prefix is routed to a CLI backend instead of an HTTP client.
CLI_PROVIDER_PREFIX = "cli_"

# provider key -> CLI executable name.
_CLI_COMMANDS: dict[str, str] = {
    "cli_claude": "claude",
    "cli_codex": "codex",
}

# Model values that mean "let the CLI pick its own default model".
_DEFAULT_MODEL_SENTINELS = {"", "default", "local-cli"}

_DEFAULT_TIMEOUT_SECONDS = 180.0


class CLIAgentError(RuntimeError):
    """Raised when a CLI backend is unavailable or fails to produce output."""


def is_cli_provider(provider: str) -> bool:
    """Return True when *provider* should be served by a local CLI backend."""
    return str(provider or "").startswith(CLI_PROVIDER_PREFIX)


def cli_command_for(provider: str) -> str | None:
    """Return the CLI executable name for *provider*, or None if unknown."""
    return _CLI_COMMANDS.get(str(provider or ""))


def cli_available(provider: str) -> bool:
    """Return True when the CLI backing *provider* is installed on PATH."""
    command = cli_command_for(provider)
    return bool(command and shutil.which(command))


def _flatten_messages(messages: Sequence[Mapping[str, object]]) -> tuple[str, str]:
    """Split chat messages into (system_text, conversation_text).

    The CLIs take a single prompt string, so the multi-turn transcript is
    flattened. System content is returned separately so callers that support a
    dedicated system flag can use it; the rest is a labelled transcript.
    """
    system_parts: list[str] = []
    convo_parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role") or "user").strip().lower()
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            convo_parts.append(f"Assistant:\n{content}")
        else:
            convo_parts.append(f"User:\n{content}")
    return "\n\n".join(system_parts), "\n\n".join(convo_parts)


def _resolve_model(model: object) -> str | None:
    """Return a CLI model override, or None to use the CLI default."""
    text = str(model or "").strip()
    if text.lower() in _DEFAULT_MODEL_SENTINELS:
        return None
    return text


def _build_argv(provider: str, command: str, model: object, system: str, cwd: str) -> list[str]:
    """Build the non-interactive CLI argv for a text-only completion."""
    resolved_model = _resolve_model(model)
    if provider == "cli_claude":
        argv = [command, "-p", "--output-format", "text"]
        if resolved_model:
            argv += ["--model", resolved_model]
        if system:
            # Keep the model in plain-answer mode; the system text is an
            # extra instruction, not a tool grant.
            argv += ["--append-system-prompt", system]
        # No tool permissions are granted: print mode + default permission
        # mode means any tool call that needs approval is auto-denied
        # (there is no interactive approver), so this stays a text generator.
        return argv
    if provider == "cli_codex":
        argv = [command, "exec", "--sandbox", "read-only", "--skip-git-repo-check",
                "--color", "never", "-C", cwd]
        if resolved_model:
            argv += ["-m", resolved_model]
        return argv
    raise CLIAgentError(f"Unsupported CLI provider: {provider!r}")


def _run_cli(provider: str, model: object, messages: Sequence[Mapping[str, object]],
             timeout: float) -> str:
    """Invoke the CLI once and return its printed text answer."""
    command = cli_command_for(provider)
    if not command or not shutil.which(command):
        raise CLIAgentError(
            f"The '{command or provider}' CLI is not installed or not on PATH."
        )
    system, conversation = _flatten_messages(messages)
    # Run in a throwaway dir so the agent can never see or touch the repo.
    with tempfile.TemporaryDirectory(prefix="easyicu-cli-agent-") as cwd:
        argv = _build_argv(provider, command, model, system, cwd)
        if provider == "cli_codex":
            # codex has no system flag; fold system into the prompt.
            prompt = f"{system}\n\n{conversation}".strip() if system else conversation
        else:
            prompt = conversation
        try:
            proc = subprocess.run(
                argv,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - timing
            raise CLIAgentError(
                f"{command} timed out after {timeout:.0f}s."
            ) from exc
        except OSError as exc:  # pragma: no cover - env specific
            raise CLIAgentError(f"Failed to launch {command}: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise CLIAgentError(
            f"{command} exited with code {proc.returncode}: {detail[:500]}"
        )
    text = (proc.stdout or "").strip()
    if not text:
        raise CLIAgentError(f"{command} returned an empty response.")
    return text


# ---------------------------------------------------------------------------
# OpenAI-shaped response objects
# ---------------------------------------------------------------------------

@dataclass
class _Message:
    content: str
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Completion:
    choices: list[_Choice]


@dataclass
class _Delta:
    content: str | None = None
    reasoning_content: None = None


@dataclass
class _ChunkChoice:
    delta: _Delta


@dataclass
class _Chunk:
    choices: list[_ChunkChoice]


def _stream_chunks(text: str) -> Iterable[_Chunk]:
    """Yield the answer as word-sized chunks for a progressive UI feel."""
    parts = text.split(" ")
    for idx, part in enumerate(parts):
        token = part if idx == 0 else " " + part
        yield _Chunk(choices=[_ChunkChoice(delta=_Delta(content=token))])


# ---------------------------------------------------------------------------
# OpenAI-shaped client
# ---------------------------------------------------------------------------

class _Completions:
    def __init__(self, client: "CLIAgentClient") -> None:
        self._client = client

    def create(self, *, model: object = None,
               messages: Sequence[Mapping[str, object]] | None = None,
               stream: bool = False, **_ignored: object):
        text = _run_cli(
            self._client.provider,
            model,
            messages or [],
            timeout=self._client.timeout,
        )
        if stream:
            return _stream_chunks(text)
        return _Completion(choices=[_Choice(message=_Message(content=text))])


class _Chat:
    def __init__(self, client: "CLIAgentClient") -> None:
        self.completions = _Completions(client)


@dataclass
class CLIAgentClient:
    """Minimal OpenAI-compatible client backed by a local coding-agent CLI."""

    provider: str
    timeout: float = _DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        self.chat = _Chat(self)

    def with_options(self, *, timeout: float | None = None,
                     max_retries: int | None = None, **_ignored: object) -> "CLIAgentClient":
        """Return a copy with overridden options (mirrors openai's client)."""
        new_timeout = float(timeout) if timeout is not None else self.timeout
        return replace(self, timeout=new_timeout)
