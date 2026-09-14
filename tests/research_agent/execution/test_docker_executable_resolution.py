"""Docker executable discovery and explicit-wrapper ownership."""

from pathlib import Path

import pandas as pd
import pytest


def _make_cohort(tmp_path: Path) -> Path:
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}).to_parquet(
        cohort, index=False
    )
    return cohort


def test_constructor_finds_docker_outside_a_short_service_path(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A launchd-hosted service can omit standard install locations from PATH."""

    import easyicu.research_agent.execution.runner as runner_mod
    from easyicu.research_agent.execution import docker_locality

    local = tmp_path / "homebrew-bin"
    local.mkdir()
    binary = local / "docker"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)

    monkeypatch.setattr(runner_mod.shutil, "which", lambda _n: None)
    monkeypatch.setattr(docker_locality, "LOCAL_DOCKER_DIRS", (local,))

    runner = ra.DockerRunner(
        workdir=tmp_path / "run", cohort_parquet=_make_cohort(tmp_path)
    )

    assert runner.docker_executable == str(binary)


def test_explicit_docker_path_is_never_replaced_by_another_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A caller that named a binary gets that binary, or a clear refusal."""

    import easyicu.research_agent.execution.runner as runner_mod
    from easyicu.research_agent.execution import docker_locality

    local = tmp_path / "homebrew-bin"
    local.mkdir()
    other = local / "docker"
    other.write_text("#!/bin/sh\n", encoding="utf-8")
    other.chmod(0o755)

    monkeypatch.setattr(runner_mod.shutil, "which", lambda _n: None)
    monkeypatch.setattr(docker_locality, "LOCAL_DOCKER_DIRS", (local,))

    assert docker_locality.resolve_docker_executable(str(tmp_path / "nope")) is None


def test_explicit_docker_wrapper_is_found_before_other_command_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from easyicu.research_agent.execution import docker_locality

    first = tmp_path / "first-bin"
    second = tmp_path / "second-bin"
    first.mkdir()
    second.mkdir()
    wrapper = second / "docker-approved-wrapper"
    for binary in (first / "docker", first / "podman", wrapper):
        binary.write_text("#!/bin/sh\n", encoding="utf-8")
        binary.chmod(0o755)
    monkeypatch.setattr(docker_locality.shutil, "which", lambda _name: None)
    monkeypatch.setattr(docker_locality, "LOCAL_DOCKER_DIRS", (first, second))

    assert docker_locality.resolve_docker_executable(wrapper.name) == str(wrapper)


@pytest.mark.parametrize("selection", ["argument", "environment"])
def test_missing_explicit_docker_wrapper_blocks_constructor_and_preflight(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: str
):
    from easyicu.research_agent.execution import docker_locality
    from easyicu.research_agent.execution import runner as runner_module

    binary = tmp_path / "docker"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setattr(docker_locality.shutil, "which", lambda _name: None)
    monkeypatch.setattr(docker_locality, "LOCAL_DOCKER_DIRS", (tmp_path,))
    kwargs = {}
    if selection == "environment":
        monkeypatch.setenv("EASYICU_DOCKER_EXECUTABLE", "docker-approved-wrapper")
    else:
        kwargs["docker_executable"] = "docker-approved-wrapper"

    with pytest.raises(FileNotFoundError, match="docker-approved-wrapper"):
        ra.DockerRunner(
            workdir=tmp_path / "run", cohort_parquet=_make_cohort(tmp_path), **kwargs
        )

    def unexpected_probe(*_args, **_kwargs):
        pytest.fail("An unavailable explicit wrapper must not probe another runtime")

    monkeypatch.setattr(runner_module, "_run_with_bounded_output", unexpected_probe)
    availability = runner_module.probe_runner_availability(kind="docker", **kwargs)
    assert availability.available is False
    assert availability.reason_code == "docker_executable_missing"
