"""Test-only isolation for research-agent runner backend selection."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


def _simulate_host_backend_probe(
    monkeypatch: pytest.MonkeyPatch,
    runner_module,
    *,
    docker_ready: bool,
) -> None:
    monkeypatch.setattr(runner_module.sys, "platform", "darwin")

    def fake_which(name: str):
        if name == "docker":
            return "/usr/local/bin/docker" if docker_ready else None
        if name == "sandbox-exec":
            return "/usr/bin/sandbox-exec"
        return None

    monkeypatch.setattr(runner_module.shutil, "which", fake_which)
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="sha256:" + "a" * 64 + "\n",
            stderr="",
        ),
    )


@pytest.mark.parametrize(
    ("docker_ready", "production_choice"),
    [(False, "subprocess"), (True, "docker")],
)
def test_ordinary_pipeline_backend_does_not_follow_docker_availability(
    ra,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    docker_ready: bool,
    production_choice: str,
) -> None:
    import easyicu.research_agent.execution.runner as runner_module
    import easyicu.research_agent.pipeline as pipeline_module

    _simulate_host_backend_probe(
        monkeypatch,
        runner_module,
        docker_ready=docker_ready,
    )

    # The production probe still responds to the real host state.
    assert (
        runner_module.select_safe_runner_kind(image="easyicu:test") == production_choice
    )
    # Ordinary Pipeline tests are fixed to one deterministic backend.
    assert pipeline_module.select_safe_runner_kind(image="easyicu:test") == "subprocess"
    assert os.environ["EASYICU_ALLOW_UNSAFE_HOST_FALLBACK"] == "1"
    # Test isolation must not rewrite the production-facing Pipeline default.
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path / "work")
    assert pipeline._runner_kind == "auto"


@pytest.mark.parametrize(
    "research_agent_runner_backend",
    ["auto"],
    indirect=True,
)
def test_auto_backend_opt_in_preserves_the_production_probe(
    research_agent_runner_backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.research_agent.execution.runner as runner_module
    import easyicu.research_agent.pipeline as pipeline_module

    _simulate_host_backend_probe(monkeypatch, runner_module, docker_ready=True)

    assert research_agent_runner_backend == "auto"
    assert pipeline_module.select_safe_runner_kind(image="easyicu:test") == "docker"


@pytest.mark.parametrize(
    "research_agent_runner_backend",
    ["docker"],
    indirect=True,
)
def test_docker_backend_opt_in_is_not_overridden(
    research_agent_runner_backend: str,
) -> None:
    import easyicu.research_agent.pipeline as pipeline_module

    assert research_agent_runner_backend == "docker"
    assert pipeline_module.select_safe_runner_kind(image="easyicu:test") == "docker"
