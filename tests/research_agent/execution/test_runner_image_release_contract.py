"""Reference image and runtime preflight release contracts."""
from pathlib import Path

import pytest


def test_runner_release_base_is_digest_pinned_and_matches_lockfile() -> None:
    root = Path(__file__).resolve().parents[3]
    image_root = root / "src/easyicu/research_agent/runner_image"
    dockerfile = (image_root / "Dockerfile").read_text(encoding="utf-8")
    lock = dict(
        line.split("=", 1)
        for line in (image_root / "base-image.lock").read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    pinned = f'{lock["image"]}@{lock["digest"]}'

    assert f"ARG BASE_IMAGE={pinned}" in dockerfile
    assert lock["digest"].startswith("sha256:")
    assert len(lock["digest"]) == 71



def test_runner_image_workflow_builds_smokes_and_generates_sbom() -> None:
    root = Path(__file__).resolve().parents[3]
    workflow = (root / ".github/workflows/research_runner_image_ci.yml").read_text(
        encoding="utf-8"
    )

    assert "docker build" in workflow
    assert "--network none" in workflow
    assert "--read-only" in workflow
    assert "--cap-drop ALL" in workflow
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in workflow
    assert "anchore/sbom-action@3ad7283483fc7af8ff2b4ea19663c2d5ca935e26" in workflow
    # Every input the Dockerfile copies triggers the gate, in push and PR.
    for path in ('"src/easyicu/**"', '"pyproject.toml"', '"README.md"'):
        assert workflow.count(path) == 2, path
    assert "tests/research_agent/authority/test_execution_kernel_identity.py" in workflow
    assert "format: cyclonedx-json" in workflow



def test_runtime_preflight_failure_spends_zero_llm_calls(
    ra,
    synthetic_cohort,
    tmp_path: Path,
):
    class MissingRuntime:
        def validate_runtime_capabilities(self):
            raise RuntimeError("required runtime package is missing")

    # Use an exact reviewed offline fixture. Subclassing MockLLMClient is no
    # longer an authorization mechanism, and this preflight must not need one.
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    llm = ScriptedMockLLMClient(["{}"])
    pipe = ra.ResearchAgentPipeline(
        workdir=tmp_path / "ra",
        llm=llm,
        runner_factory=lambda **_kwargs: MissingRuntime(),
    )

    with pytest.raises(RuntimeError, match="required runtime package is missing"):
        pipe.run(
            question="Is exposure associated with mortality?",
            cohort=synthetic_cohort,
            cohort_name="runtime_preflight",
            database="miiv",
            target_outcome="death",
        )

    assert llm.calls == []


@pytest.mark.parametrize("message", ["permission denied: /private/host.sock", "context not found", "server returned an invalid response", ""])
def test_unknown_docker_failures_do_not_claim_that_an_image_is_missing(message):
    from easyicu.research_agent.execution.runner import _classify_docker_failure

    assert _classify_docker_failure(message, "") == "docker_probe_failed"
