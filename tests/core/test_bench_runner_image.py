"""A benchmark launch must name its runner image, and the rule must be in-repo.

The launcher script lives outside any git repository, so the guard that used to
be written there was protected by nothing.  These tests own the rule instead.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from tools.bench_runner_image import (
    ENV_VAR,
    RunnerImageError,
    main,
    resolve_required_runner_image,
)


def test_an_unset_image_refuses_the_launch() -> None:
    with pytest.raises(RunnerImageError) as excinfo:
        resolve_required_runner_image({}, verify_present=False)

    assert excinfo.value.reason_code == "runner_image_not_declared"
    assert ENV_VAR in str(excinfo.value)


@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_a_blank_image_is_not_a_declaration(value: str) -> None:
    """An empty variable is how a shell passes "I forgot", not a choice."""

    with pytest.raises(RunnerImageError) as excinfo:
        resolve_required_runner_image({ENV_VAR: value}, verify_present=False)

    assert excinfo.value.reason_code == "runner_image_not_declared"


def test_a_declared_image_is_returned_verbatim() -> None:
    image = resolve_required_runner_image(
        {ENV_VAR: "  easyicu-research-agent:dev-abc1234  "},
        verify_present=False,
    )

    assert image == "easyicu-research-agent:dev-abc1234"


def test_an_image_this_host_does_not_have_is_refused(tmp_path: Path) -> None:
    """A typo must fail at launch, not deep inside the run."""

    fake_docker = tmp_path / "docker"
    fake_docker.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake_docker.chmod(0o755)

    with pytest.raises(RunnerImageError) as excinfo:
        resolve_required_runner_image(
            {ENV_VAR: "easyicu-research-agent:no-such-tag"},
            docker=str(fake_docker),
        )

    assert excinfo.value.reason_code == "runner_image_not_present"


def test_a_present_image_passes_verification(tmp_path: Path) -> None:
    fake_docker = tmp_path / "docker"
    fake_docker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_docker.chmod(0o755)

    image = resolve_required_runner_image(
        {ENV_VAR: "easyicu-research-agent:dev-abc1234"},
        docker=str(fake_docker),
    )

    assert image == "easyicu-research-agent:dev-abc1234"


def test_an_unusable_docker_does_not_turn_into_a_false_refusal(
    tmp_path: Path,
) -> None:
    """Verification is a courtesy; the explicit-naming rule is the guarantee.

    If docker cannot be consulted we must not invent a refusal for an image
    that may well exist -- the runtime will fail loudly on its own.  What we
    must never do is let the *unset* case through, which the tests above pin.
    """

    missing = tmp_path / "not-a-real-docker"

    image = resolve_required_runner_image(
        {ENV_VAR: "easyicu-research-agent:dev-abc1234"},
        docker=str(missing),
    )

    assert image == "easyicu-research-agent:dev-abc1234"


def test_the_module_entrypoint_exits_two_when_undeclared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The launcher shells out to this; the exit status is its contract."""

    monkeypatch.delenv(ENV_VAR, raising=False)

    assert main([]) == 2


def test_the_entrypoint_prints_the_image_when_declared(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    fake_docker = tmp_path / "docker"
    fake_docker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_docker.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv(ENV_VAR, "easyicu-research-agent:dev-abc1234")

    assert main([]) == 0
    assert capsys.readouterr().out.strip() == "easyicu-research-agent:dev-abc1234"


def test_run_as_a_subprocess_refuses_without_the_variable() -> None:
    """End-to-end: the exact invocation a shell launcher makes."""

    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "tools.bench_runner_image"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(repo_root)},
        timeout=120,
        check=False,
    )

    assert completed.returncode == 2
    assert "runner_image_not_declared" in completed.stderr
