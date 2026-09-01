"""A missing container runtime must be said early, and said by name.

Whether the mandated backend can run generated code is a static fact a bounded
probe settles in seconds. It used to be discovered only inside the pipeline --
after the provider was authorized, the data foundation was built and the cohort
was materialized -- so a stopped daemon cost roughly a minute and a half of real
provider spend before saying anything. What it then said was "The governed
Research Agent operation failed." with an empty ``exception_types``: the word
"Docker" appeared nowhere between the daemon's own error and the screen.

Requiring the container is deliberate -- results without an image digest are not
submission-grade -- so the repair is not to start it automatically or to fall
back to the host. It is to ask before spending, and to keep the answer
attributable.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.execution import runner as runner_module
from easyicu.webserver import agent_pipeline_runs, research_launch_runtime


# One realistic daemon-down stderr, host socket path and all. Nothing derived
# from it may reach a persisted diagnostic.
_DAEMON_DOWN_STDERR = (
    "Cannot connect to the Docker daemon at "
    "unix:///Users/someone/.colima/default/docker.sock. "
    "Is the docker daemon running?"
)
_SOCKET_PATH = "/Users/someone/.colima/default/docker.sock"


def _unavailable(reason_code: str) -> runner_module.RunnerAvailability:
    return runner_module.RunnerAvailability(
        kind="docker",
        available=False,
        image="easyicu-research-agent:1.0.0",
        reason_code=reason_code,
    )


def _stub_probe(monkeypatch: pytest.MonkeyPatch, availability):
    """Replace the live probe and record what the caller asked about."""

    asked: list[str] = []

    def probe(kind, **_kwargs):
        asked.append(kind)
        if callable(availability):
            return availability(kind)
        return availability

    monkeypatch.setattr(runner_module, "probe_runner_availability", probe)
    return asked


def test_a_stopped_daemon_is_refused_before_the_run_spends_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = _stub_probe(monkeypatch, _unavailable("docker_daemon_unreachable"))

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        research_launch_runtime._require_execution_runtime(
            budget_mode="full_reviewed",
            runner_image="easyicu-research-agent:1.0.0",
        )

    assert asked == ["docker"]
    assert exc.value.code == "research_pipeline_execution_runtime_unavailable"
    assert exc.value.details["reason_code"] == "docker_daemon_unreachable"
    assert exc.value.details["runner_kind"] == "docker"
    assert exc.value.details["owner"] == "easyicu.research_agent.execution.runner"
    # The researcher is told what to do, not merely that something failed.
    assert "Start Docker" in str(exc.value)


def test_a_ready_runtime_is_not_an_obstacle(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_probe(
        monkeypatch,
        runner_module.RunnerAvailability(
            kind="docker", available=True, image="easyicu-research-agent:1.0.0"
        ),
    )

    research_launch_runtime._require_execution_runtime(
        budget_mode="full_reviewed",
        runner_image="easyicu-research-agent:1.0.0",
    )


def test_a_planner_only_launch_never_asks_for_a_container_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planning generates no code, so planning must stay startable without one.

    A gate that refused every launch would trade a late failure for a machine
    that cannot plan at all.
    """

    asked = _stub_probe(monkeypatch, _unavailable("docker_daemon_unreachable"))

    for budget_mode in ("planner_canary", "", "unknown_mode"):
        research_launch_runtime._require_execution_runtime(
            budget_mode=budget_mode,
            runner_image="easyicu-research-agent:1.0.0",
        )

    assert asked == []


def test_the_preflight_guards_exactly_the_profiles_the_run_can_select() -> None:
    """Guarding a profile the run never selects protects nothing.

    Which of the two literature variants a run picks is decided inside the run,
    so both are read from the same mapping the run itself uses.
    """

    from easyicu.research_agent.orchestration.profiles import get_submission_profile

    canary = {
        agent_pipeline_runs._submission_profile_ref(
            budget_mode="planner_canary", live_pubmed=flag
        )
        for flag in (False, True)
    }
    reviewed = {
        agent_pipeline_runs._submission_profile_ref(
            budget_mode="full_reviewed", live_pubmed=flag
        )
        for flag in (False, True)
    }

    assert len(canary) == 2 and len(reviewed) == 2
    assert not canary & reviewed
    # The split the preflight keys off is the profiles' own declaration.
    assert all(get_submission_profile(ref).planner_only for ref in canary)
    assert not any(get_submission_profile(ref).planner_only for ref in reviewed)
    assert {get_submission_profile(ref).requires_runner for ref in reviewed} == {
        "docker"
    }


def test_the_run_reads_the_profile_mapping_from_one_place() -> None:
    """Two copies of the mapping is how the preflight drifts from the run."""

    source = Path("src/easyicu/webserver/agent_pipeline_runs.py").read_text(
        encoding="utf-8"
    )
    assert "submission_profile_ref = _submission_profile_ref(" in source
    # The profile refs are named only inside the shared resolver, so the run
    # cannot quietly grow a second copy of the mapping.
    _, _, after_factory = source.partition("def make_research_pipeline_run_runner(")
    assert after_factory
    assert "CURRENT_E1_REVIEWED_DEMO_DEV_PROFILE_REF" not in after_factory
    assert "CURRENT_E1_PLANNER_CANARY_DEV_PROFILE_REF" not in after_factory


def test_a_runtime_that_dies_mid_run_is_attributable_not_anonymous() -> None:
    """The late failure still happens on a host that stops Docker mid-run.

    When it does, it must name its owner instead of collapsing into the generic
    execution-failure code the science shares with every other defect.
    """

    error = runner_module.ExecutionRuntimeUnavailableError(
        _unavailable("docker_daemon_unreachable")
    )

    assert agent_pipeline_runs._safe_pipeline_typed_failure(error) == {
        "owner": "easyicu.execution.runtime_v1",
        "reason_code": "docker_daemon_unreachable",
        "runner_kind": "docker",
    }
    assert (
        agent_pipeline_runs._pipeline_failure_code(error)
        == "research_pipeline_execution_runtime_unavailable"
    )
    # ...and the same code the launch preflight raises, so one cause has one name.
    assert (
        type(error).__name__ in agent_pipeline_runs._SAFE_PIPELINE_EXCEPTION_TYPES
    )


def test_the_persisted_diagnostic_records_the_type_and_not_the_socket_path(
    tmp_path: Path,
) -> None:
    """Attributable, still leak-closed.

    ``exception_types`` was empty because a bare ``RuntimeError`` is not on the
    safe list -- correctly so. The repair is a typed exception, not a wider list
    and not persisted exception text: the daemon's own wording names a host
    path and must not survive the boundary.
    """

    reason = runner_module._classify_docker_failure(_DAEMON_DOWN_STDERR, "")
    error = runner_module.ExecutionRuntimeUnavailableError(_unavailable(reason))

    relative = agent_pipeline_runs._write_pipeline_failure_diagnostic(
        wrapper_dir=tmp_path,
        exc=error,
        code="research_pipeline_execution_runtime_unavailable",
    )
    payload = json.loads((tmp_path / str(relative)).read_text(encoding="utf-8"))

    assert payload["exception_types"] == ["ExecutionRuntimeUnavailableError"]
    assert payload["typed_failure"]["reason_code"] == "docker_daemon_unreachable"
    assert payload["message"] == "The governed Research Agent operation failed."
    serialized = json.dumps(payload)
    assert _SOCKET_PATH not in serialized
    assert "colima" not in serialized


def test_a_stopped_daemon_and_a_missing_image_are_not_the_same_problem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """They need different fixes, and only the daemon's text separates them."""

    monkeypatch.setattr(runner_module.shutil, "which", lambda _name: "/usr/bin/docker")

    def probe_returning(stderr: str):
        return lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr=stderr
        )

    monkeypatch.setattr(
        runner_module, "_run_with_bounded_output", probe_returning(_DAEMON_DOWN_STDERR)
    )
    down = runner_module.probe_runner_availability("docker", image="easyicu:test")

    monkeypatch.setattr(
        runner_module,
        "_run_with_bounded_output",
        probe_returning("Error response from daemon: No such image: easyicu:test"),
    )
    absent = runner_module.probe_runner_availability("docker", image="easyicu:test")

    assert down.available is False and absent.available is False
    assert down.reason_code == "docker_daemon_unreachable"
    assert absent.reason_code == "docker_image_missing"
    # The classification is the only thing kept; the wording is discarded.
    assert _SOCKET_PATH not in repr(down)


def test_a_missing_docker_executable_is_reported_without_a_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_module.shutil, "which", lambda _name: None)

    def fail(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("no probe is possible without an executable")

    monkeypatch.setattr(runner_module, "_run_with_bounded_output", fail)

    availability = runner_module.probe_runner_availability("docker")
    assert availability.available is False
    assert availability.reason_code == "docker_executable_missing"


def test_the_web_projection_mirrors_the_owner_contract() -> None:
    """A reason code the owner adds must not reach the boundary unprojected."""

    assert (
        agent_pipeline_runs._EXECUTION_RUNTIME_DIAGNOSTIC_OWNER
        == runner_module.EXECUTION_RUNTIME_DIAGNOSTIC_OWNER
    )
    assert (
        agent_pipeline_runs._SAFE_RUNNER_UNAVAILABLE_REASONS
        == runner_module.RUNNER_UNAVAILABLE_REASON_CODES
    )
    # Every closed reason carries a fix, not just a name.
    for reason_code in runner_module.RUNNER_UNAVAILABLE_REASON_CODES:
        remediation = runner_module.runner_unavailable_remediation(reason_code)
        assert remediation and "not usable" not in remediation


def test_an_unknown_reason_code_is_refused_by_the_projection() -> None:
    """Exception attributes are mutable; the boundary re-validates them."""

    error = runner_module.ExecutionRuntimeUnavailableError(
        _unavailable("docker_daemon_unreachable")
    )
    error.easyicu_safe_diagnostic = {
        "owner": "easyicu.execution.runtime_v1",
        "reason_code": "daemon said /Users/someone/.colima/docker.sock",
        "runner_kind": "docker",
    }

    assert agent_pipeline_runs._safe_pipeline_typed_failure(error) == {}


def test_a_resume_after_approval_also_names_the_runtime(monkeypatch) -> None:
    """Executing the approved plan is where the container is used for real.

    A host that stops Docker between planning and approval fails there, and
    "The governed Research Agent run could not resume after plan review." tells
    the researcher nothing they can act on.
    """

    from easyicu.research_agent.orchestration.workflow import HumanReviewPending

    error = runner_module.ExecutionRuntimeUnavailableError(
        _unavailable("docker_daemon_unreachable")
    )
    assert not isinstance(error, HumanReviewPending)

    source = Path("src/easyicu/webserver/agent_pipeline_runs.py").read_text(
        encoding="utf-8"
    )
    _, _, resume = source.partition("def resume_research_pipeline(")
    assert resume
    # The resume handler asks the same owner the launch preflight does, so one
    # cause keeps one name on both entry points.
    assert "runtime_unavailable = (" in resume
    assert '"research_pipeline_execution_runtime_unavailable"' in resume
    assert '"research_pipeline_review_resume_failed"' in resume
