"""Offline contracts for the opt-in laptop acceptance driver."""

from __future__ import annotations

import json
import io
import logging
import os
from contextlib import nullcontext
from http.client import IncompleteRead
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest
import yaml

from tools import windows_laptop_acceptance as acceptance


ROOT = Path(__file__).resolve().parents[2]


def test_windows_workflow_uses_the_desktop_build_and_only_publishes_aggregates() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/windows_laptop_acceptance.yml").read_text("utf-8"),
    )
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"workflow_dispatch"}
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"memory_pressure_gb", "ref"}
    assert inputs["memory_pressure_gb"]["default"] == 8
    job = workflow["jobs"]["acceptance"]
    assert job["runs-on"] == "windows-latest"
    steps = job["steps"]
    checkout = next(step for step in steps if "actions/checkout@" in step.get("uses", ""))
    assert checkout["with"]["ref"] == "${{ inputs.ref || github.ref }}"
    setup = next(step for step in steps if "actions/setup-python@" in step.get("uses", ""))
    assert setup["with"]["python-version"] == "3.11"
    commands = "\n".join(step.get("run", "") for step in steps)
    assert "python desktop/scripts/build_windows.py" in commands
    assert "pip install" not in commands
    assert "--desktop-backend" in commands
    assert "--desktop-first-result" in commands
    assert "-I tools/windows_laptop_acceptance.py" in commands
    assert "Start-Process" in commands and "/S /D=$installDir" in commands
    assert "uninstall.exe" in commands
    assert 'secrets.' not in json.dumps(workflow)
    assert '$env:PATH = "$env:SystemRoot\\System32;$env:SystemRoot"' in commands
    assert "--cache-key" in commands
    assert "--memory-pressure-gb" in commands
    assert "${{ inputs.memory_pressure_gb }}" not in commands
    cache = next(step for step in steps if "actions/cache@" in step.get("uses", ""))
    assert "steps.demo-cache.outputs.key" in cache["with"]["key"]
    upload = next(step for step in steps if "actions/upload-artifact@" in step.get("uses", ""))
    assert upload["if"] == "${{ always() }}"
    assert upload["with"]["path"].splitlines() == [
        "acceptance-results/acceptance.json",
        "acceptance-results/summary.md",
    ]


@pytest.mark.parametrize("docker_reason", ["docker_executable_missing", "docker_daemon_unreachable"])
def test_desktop_boundary_uses_host_profile_and_never_claims_a_result(tmp_path, monkeypatch, docker_reason):
    from easyicu.research_agent.execution import runner
    from easyicu.research_agent.orchestration.profiles import DESKTOP_FIRST_RESULT
    from easyicu.webserver import dataio

    probes, bindings = [], []

    def availability(kind, **kwargs):
        probes.append(kind)
        assert kwargs.get("workdir") is None
        return SimpleNamespace(available=kind == "host", kind=kind, image="", reason_code="" if kind == "host" else docker_reason)

    monkeypatch.setenv("EASYICU_DEPLOYMENT", "desktop")
    monkeypatch.setattr(runner, "probe_runner_availability", availability)
    def validate(path, **kwargs):
        bindings.append((path, kwargs))
        return {"ok": True, "path": "private", "binding": {"files": [{"relative_path": "private"}]}}
    monkeypatch.setattr(dataio, "validate_research_pipeline_source", validate)
    result = acceptance.probe_desktop_first_result([tmp_path / "miiv", tmp_path / "eicu"])
    assert probes == ["docker", "docker", "host"]
    assert bindings == [(str(tmp_path / "eicu"), {"database": "eicu_demo"})]
    assert result["profile_ref"] == DESKTOP_FIRST_RESULT.ref
    assert result["runner_kind"] == "host"
    assert result["docker_reason_code"] == docker_reason
    assert result["deterministic_analysis_without_docker"] == "yes"
    assert result["source_validated"] is True
    assert result["first_result"]["status"] == "blocked"
    assert result["first_result"]["reason_code"] == "desktop_first_result_model_replay_unavailable"
    assert result["first_result"]["evidence_bound_result"] is False
    assert result["first_result"]["seconds_to_first_result"] is None
    assert result["first_result"]["budget_met"] is None
    assert result["analysis_executed"] is False
    assert result["provider_calls"] == 0
    assert "private" not in json.dumps(result)


def test_desktop_boundary_rejects_a_docker_run_and_a_missing_demo(tmp_path, monkeypatch):
    from easyicu.research_agent.execution import runner
    from easyicu.webserver import research_launch_runtime

    monkeypatch.setattr(runner, "probe_runner_availability", lambda *a, **kw: SimpleNamespace(reason_code="", available=True))
    monkeypatch.setattr(research_launch_runtime, "_require_execution_runtime", lambda **kw: "docker")
    with pytest.raises(acceptance.AcceptanceFailure, match="desktop_host_runner_required"):
        acceptance.probe_desktop_first_result([tmp_path / "a", tmp_path / "b"])
    monkeypatch.setattr(research_launch_runtime, "_require_execution_runtime", lambda **kw: "host")
    with pytest.raises(acceptance.AcceptanceFailure, match="demo_exports_unavailable_for_research"):
        acceptance.probe_desktop_first_result([])


@pytest.mark.parametrize("elapsed,within", [(30, True), (901, False)])
def test_blocked_m1_records_elapsed_time_without_passing_the_budget(tmp_path, monkeypatch, elapsed, within):
    work = tmp_path / "EasyICU 数据"
    monkeypatch.setattr(acceptance, "prepare_dataset", lambda source, paths, *a, **kw: paths.export)
    monkeypatch.setattr(acceptance, "probe_web", lambda *a: None)
    monkeypatch.setattr(acceptance, "probe_desktop_first_result", lambda exports: {"first_result": {
        "status": "blocked", "reason_code": "desktop_first_result_model_replay_unavailable",
        "evidence_bound_result": False, "budget_met": None, "seconds_to_first_result": None,
    }})
    ticks = iter([0, elapsed])
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: next(ticks))
    report = acceptance.run_acceptance(work, work / "cache", tmp_path / "receipts",
                                       memory_pressure_gb=0, desktop_first_result=True)
    assert report["status"] == "blocked"
    result = report["research"]["first_result"]
    assert result["elapsed_to_boundary_seconds"] == elapsed
    assert result["budget_seconds"] == 900
    assert result["boundary_within_budget"] is within
    assert result["budget_met"] is None
    assert "M1: **blocked**" in (tmp_path / "receipts/summary.md").read_text()


def test_installed_backend_uses_its_entrypoint_and_private_session(tmp_path, monkeypatch):
    socket = SimpleNamespace(bind=lambda address: None, getsockname=lambda: ("127.0.0.1", 4321))
    monkeypatch.setattr(acceptance.socket, "socket", lambda: nullcontext(socket))
    commands = []
    def popen(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(pid=42)
    monkeypatch.setattr(acceptance.subprocess, "Popen", popen)
    monkeypatch.setattr(acceptance, "stop_process_tree", lambda *args: None)
    backend = tmp_path / "installed/resources/backend/easyicu-backend.exe"
    server = acceptance.LocalWebServer(tmp_path, desktop_backend=backend, startup_timeout=0)
    with pytest.raises(acceptance.AcceptanceFailure, match="webserver_health_timeout"):
        with server:
            pass
    command, options = commands[0]
    assert command[0] == str(backend)
    assert "-m" not in command
    assert command[command.index("--state-dir") + 1] == str(tmp_path / "home")
    assert command[command.index("--parent-pid") + 1] == str(os.getpid())
    token = options["env"]["EASYICU_DESKTOP_SESSION_TOKEN"]
    assert len(token) >= 32
    assert token not in " ".join(command)
    assert server.headers == {"X-EasyICU-Desktop-Token": token}
    assert server.log.closed


def test_failed_export_preserves_metrics_without_registering_partial_files(tmp_path, monkeypatch):
    from easyicu.webserver import dataio, sources
    source, paths, _, _ = _fake_export_owners(tmp_path, monkeypatch)
    def export(**kwargs):
        def fail(job):
            raise dataio.ExportCohortError("synthetic_export_failure")
        return fail
    monkeypatch.setattr(dataio, "make_export_runner", export)
    monkeypatch.setattr(sources, "register_source", lambda *a, **kw: pytest.fail("partial export registered"))
    phases = {}
    with pytest.raises(dataio.ExportCohortError):
        acceptance.prepare_dataset(source, paths, phases, fetch=lambda *a: None)
    assert phases["convert"]["status"] == "passed"
    assert phases["export"]["status"] == "failed"
    assert phases["export"]["failure"]["reason_code"] == "synthetic_export_failure"
    assert phases["export"]["rows"] == 2


@pytest.mark.parametrize("same", [True, False])
def test_desktop_checks_bind_the_installed_sidecar_to_the_wheel(tmp_path, monkeypatch, same):
    from easyicu.research_agent.execution import kernel_identity
    backend = tmp_path / "resources/backend/easyicu-backend.exe"
    backend.parent.mkdir(parents=True)
    backend.touch()
    calls = []
    def identity(path):
        calls.append(path)
        return "same" if same or len(calls) == 1 else "different"
    monkeypatch.setattr(kernel_identity, "build_execution_kernel_identity", identity)
    if same:
        assert acceptance.verify_desktop_build(backend) == {"installed_backend": True, "kernel_identity_matches": True}
    else:
        with pytest.raises(acceptance.AcceptanceFailure, match="desktop_kernel_identity_mismatch"):
            acceptance.verify_desktop_build(backend)
    assert calls[1] == backend.parent / "_internal/easyicu"
    backend.unlink()
    with pytest.raises(acceptance.AcceptanceFailure, match="desktop_backend_missing"):
        acceptance.verify_desktop_build(backend)


def test_desktop_token_reaches_only_the_loopback_request_header(monkeypatch):
    requests = []
    class Response:
        code = 200
        def read(self):
            return b'{"ok":true}'
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    def open_request(request, **kwargs):
        requests.append(request)
        return Response()
    monkeypatch.setattr(acceptance, "build_opener", lambda *args: SimpleNamespace(open=open_request))
    token = "a" * 64
    server = acceptance.LocalWebServer(Path("."), desktop_backend=Path("backend.exe"))
    server.headers = {"X-EasyICU-Desktop-Token": token}
    server.base_url = "http://127.0.0.1:4321"
    assert server.request("GET", "/api/catalog", None) == (200, {"ok": True})
    assert dict(requests[0].header_items())["X-easyicu-desktop-token"] == token
    assert token not in requests[0].full_url
    assert requests[0].data is None


def test_desktop_mode_restores_environment_and_simulates_an_absent_docker(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYICU_DEPLOYMENT", "server")
    monkeypatch.setenv("EASYICU_DOCKER_EXECUTABLE", "previous-docker")
    with acceptance.laptop_environment(tmp_path, tmp_path / "cache", desktop_first_result=True):
        assert os.environ["EASYICU_DEPLOYMENT"] == "desktop"
        assert os.environ["EASYICU_DOCKER_EXECUTABLE"] == str(tmp_path / "docker-not-installed/docker.exe")
    assert os.environ["EASYICU_DEPLOYMENT"] == "server"
    assert os.environ["EASYICU_DOCKER_EXECUTABLE"] == "previous-docker"


def test_cli_returns_nonzero_for_an_unproven_m1(monkeypatch):
    calls = []
    def run(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "blocked"}
    monkeypatch.setattr(acceptance, "run_acceptance", run)
    assert acceptance.main(["--desktop-first-result", "--desktop-backend", "installed.exe"]) == 2
    assert calls[0]["desktop_first_result"] is True
    assert calls[0]["desktop_backend"] == Path("installed.exe")


def test_launch_failure_receipt_preserves_the_runtime_reason():
    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
    exc = ResearchPipelineRunError(
        "research_pipeline_execution_runtime_unavailable", "private diagnostics",
        details={"reason_code": "host_runtime_unavailable", "path": "private"},
    )
    assert acceptance.failure_receipt(exc) == {
        "reason_code": "host_runtime_unavailable", "exception_type": "ResearchPipelineRunError",
    }


def test_cache_key_tracks_both_official_versions_and_digests() -> None:
    from dataclasses import replace
    from easyicu.webserver.demo_source_contracts import SOURCES

    key = acceptance.demo_cache_key(SOURCES)
    assert all(source.version in key for source in SOURCES)
    assert key != acceptance.demo_cache_key((replace(SOURCES[0], version="99"), SOURCES[1]))
    assert key != acceptance.demo_cache_key((replace(SOURCES[0], archive_sha256="a" * 64), SOURCES[1]))


@pytest.mark.parametrize("error", [URLError("temporary"), HTTPError("", 503, "", {}, None), IncompleteRead(b"partial", 42)])
def test_download_retries_transient_errors_three_times(error: Exception) -> None:
    attempts = []
    sleeps = []

    def download():
        attempts.append(1)
        if len(attempts) < 3:
            raise error
        return "checksum", True

    assert acceptance.retry_download(download, sleep=sleeps.append) == ("checksum", True)
    assert len(attempts) == 3
    assert len(sleeps) == 2


@pytest.mark.parametrize("error", [ValueError("invalid archive"), HTTPError("", 403, "", {}, None)])
def test_download_does_not_retry_permanent_failures(error: Exception) -> None:
    attempts = []

    def download():
        attempts.append(1)
        raise error

    with pytest.raises(type(error)):
        acceptance.retry_download(download, sleep=lambda _: None)
    assert len(attempts) == 1


def test_download_stops_after_three_failed_attempts() -> None:
    attempts = []

    def download():
        attempts.append(1)
        raise URLError("still unavailable")

    with pytest.raises(URLError):
        acceptance.retry_download(download, sleep=lambda _: None)
    assert len(attempts) == 3


def test_zero_pressure_starts_no_helper(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(acceptance.subprocess, "Popen", lambda *a, **kw: pytest.fail("helper started"))
    with acceptance.MemoryPressure(0, tmp_path) as pressure:
        assert pressure.pid is None
        assert pressure.report["requested_bytes"] == 0
        assert pressure.report["before"]["total_bytes"] > 0
        assert pressure.report["after"]["available_bytes"] > 0
    assert pressure.report["released"]["available_bytes"] > 0


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_pressure_is_rejected(value: float, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        acceptance.MemoryPressure(value, tmp_path)


def test_receipt_drops_rows_ids_paths_and_exception_text(tmp_path: Path) -> None:
    class ProductFailure(Exception):
        code = "synthetic_product_failure"

    report = {"status": "failed", "failures": [acceptance.failure_receipt(ProductFailure("patient 987 /private/data"))]}
    acceptance.write_reports(report, tmp_path)
    payload = (tmp_path / "acceptance.json").read_text("utf-8")
    summary = (tmp_path / "summary.md").read_text("utf-8")
    assert json.loads(payload)["failures"][0]["reason_code"] == "synthetic_product_failure"
    assert "987" not in payload + summary
    assert "/private" not in payload + summary


@pytest.mark.parametrize("path", ["plain", "with space", "数据"])
def test_acceptance_requires_a_space_and_chinese_path(tmp_path: Path, path: str) -> None:
    with pytest.raises(acceptance.AcceptanceFailure, match="laptop_path_required"):
        acceptance.validate_work_dir(tmp_path / path)


def test_read_failures_are_recorded_without_response_bodies(tmp_path: Path) -> None:
    calls = []
    records = []

    def request(method, path, body):
        calls.append((method, path, body))
        return 503, {"patient_id": 987, "error": "private"}

    with pytest.raises(acceptance.AcceptanceFailure, match="web_read_failed"):
        acceptance.probe_read_paths(request, [tmp_path / "export"], records)
    assert calls
    assert records[0]["status_code"] == 503
    assert "987" not in json.dumps(records)
    assert "private" not in json.dumps(records)


def test_tracebacks_fail_acceptance_even_when_http_succeeds(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text("Traceback (most recent call last):\nprivate row\n", encoding="utf-8")
    with pytest.raises(acceptance.AcceptanceFailure, match="server_traceback"):
        acceptance.check_server_log(log)


def test_phase_peak_includes_children_and_excludes_the_pressure_helper(monkeypatch) -> None:
    def process(pid, rss):
        return SimpleNamespace(pid=pid, memory_info=lambda: SimpleNamespace(rss=rss))

    root = process(1, 100)
    root.children = lambda recursive: [process(2, 200), process(3, 8000)]
    monkeypatch.setattr(acceptance.psutil, "Process", lambda: root)
    with acceptance.PhaseMeasurement(excluded_pids=(3,)) as measurement:
        pass
    assert measurement.report["peak_working_set_bytes"] == 300
    assert measurement.report["sample_interval_seconds"] == 1
    assert measurement.report["status"] == "passed"


def test_pressure_helper_accepts_windows_newlines_and_always_releases(tmp_path: Path, monkeypatch) -> None:
    process = SimpleNamespace(
        pid=42, stdin=io.BytesIO(), stdout=io.BytesIO(b"ready\r\n"), wait=lambda **kw: 0,
    )
    monkeypatch.setattr(acceptance.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(acceptance.psutil, "Process", lambda pid: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100)))
    with pytest.raises(RuntimeError, match="phase failed"), acceptance.MemoryPressure(1 / 1024, tmp_path) as pressure:
        assert pressure.pid == 42
        raise RuntimeError("phase failed")
    assert process.stdin.closed
    assert process.stdout.closed
    assert pressure.report["released"]["total_bytes"] > 0


def test_failed_pressure_start_reaps_the_helper(tmp_path: Path, monkeypatch) -> None:
    process = SimpleNamespace(
        pid=42, stdin=io.BytesIO(), stdout=io.BytesIO(b""), wait=lambda **kw: 0,
    )
    monkeypatch.setattr(acceptance.subprocess, "Popen", lambda *a, **kw: process)
    with pytest.raises(acceptance.AcceptanceFailure, match="memory_pressure_helper_failed"):
        with acceptance.MemoryPressure(1 / 1024, tmp_path):
            pytest.fail("failed helper accepted")
    assert process.stdin.closed and process.stdout.closed


def test_real_pressure_helper_touches_resident_pages_and_exits(tmp_path: Path) -> None:
    size = 64 * 1024**2
    with acceptance.MemoryPressure(size / acceptance.GIB, tmp_path) as pressure:
        pid = pressure.pid
        assert pressure.report["helper_working_set_bytes"] >= size
        assert acceptance.psutil.pid_exists(pid)
    assert not acceptance.psutil.pid_exists(pid)


def test_pressure_rejects_allocations_without_headroom(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(acceptance, "memory_snapshot", lambda: {"total_bytes": 16 * acceptance.GIB, "available_bytes": acceptance.GIB})
    monkeypatch.setattr(acceptance.subprocess, "Popen", lambda *a, **kw: pytest.fail("unsafe allocation"))
    with pytest.raises(acceptance.AcceptanceFailure, match="memory_pressure_insufficient_headroom"):
        with acceptance.MemoryPressure(8, tmp_path):
            pass


def test_default_pressure_must_reach_the_laptop_free_memory_target(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(acceptance, "memory_snapshot", lambda: {"total_bytes": 32 * acceptance.GIB, "available_bytes": 24 * acceptance.GIB})
    process = SimpleNamespace(pid=42, stdin=io.BytesIO(), stdout=io.BytesIO(b"ready\n"), wait=lambda **kw: 0)
    monkeypatch.setattr(acceptance.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(acceptance.psutil, "Process", lambda pid: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100)))
    with pytest.raises(acceptance.AcceptanceFailure, match="memory_pressure_target_not_reached"):
        with acceptance.MemoryPressure(8, tmp_path):
            pass
    assert process.stdin.closed and process.stdout.closed


def test_read_probes_use_registered_paths_and_let_the_owner_select_one_stay(tmp_path: Path) -> None:
    calls, records = [], []

    def request(method, path, body):
        calls.append((method, path, body))
        return 200, {"ok": True, "dict": {"hr": {}}}

    acceptance.probe_read_paths(request, [tmp_path / "miiv", tmp_path / "eicu"], records)
    assert len(records) == 7
    assert {record["read"] for record in records} == {"catalog", "dictionary", "cohort_1", "cohort_2", "patient_1", "patient_2", "crossdb"}
    assert all(record["passed"] for record in records)
    for method, path, body in calls:
        if path == "/api/patient-review/drilldown":
            assert method == "POST"
            assert set(body) == {"source_path"}
    assert calls[-1][2] == {"paths": [str(tmp_path / "miiv"), str(tmp_path / "eicu")]}


def test_a_dictionary_error_on_http_200_still_fails() -> None:
    records = []
    with pytest.raises(acceptance.AcceptanceFailure, match="web_read_failed"):
        acceptance.probe_read_paths(lambda *args: (200, {"ok": False}), [], records)
    assert not any(record["passed"] for record in records)


@pytest.mark.parametrize("reason_code", ["docker_executable_missing", "docker_daemon_unreachable"])
def test_research_uses_the_real_launch_gate_without_running_analysis(monkeypatch, reason_code: str) -> None:
    from easyicu.webserver import research_launch_runtime
    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

    calls = []

    def unavailable(**kwargs):
        calls.append(kwargs)
        raise ResearchPipelineRunError(
            "research_pipeline_execution_runtime_unavailable", "private host path",
            details={"reason_code": reason_code},
        )

    monkeypatch.setattr(research_launch_runtime, "_require_execution_runtime", unavailable)
    result = acceptance.probe_research()
    assert calls == [{"budget_mode": "full_reviewed", "runner_image": ""}]
    assert result["reason_code"] == reason_code
    assert result["available"] is False
    assert result["deterministic_analysis_without_docker"] == "no"
    assert result["local_preflight_without_docker"] == "yes"
    assert result["analysis_executed"] is False
    assert "private host" not in json.dumps(result)


def _fake_export_owners(tmp_path: Path, monkeypatch):
    """Driver-unit boundary; the separate synthetic test uses unmodified owners."""
    from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL
    from easyicu.webserver import dataio, sources
    from easyicu.webserver.demo_source_contracts import SOURCES
    from tests.support.windows_laptop_demo import isolated_demo_paths

    paths = isolated_demo_paths(tmp_path, SOURCES[0])
    paths.raw.mkdir(parents=True)
    paths.export.mkdir()
    (paths.export / "_manifest.json").write_text('{"schema_version":"easyicu_native_export_v2"}')
    calls = []
    plans = {name: {"mode": "one_shot", "batch_size": 2, "reason_code": "measured_profile_fast_path"} for name, concepts in CONCEPT_GROUPS_INTERNAL.items() if concepts}
    result = {
        "manifest": "_manifest.json", "file_count": len(plans), "total_rows": 2 * len(plans),
        "module_resource_plans": plans,
        "files": [{"module": name, "file": "private", "rows": 2} for name in plans],
    }

    def export(**kwargs):
        calls.append(kwargs)
        return lambda job: result

    monkeypatch.setattr(dataio, "make_convert_runner", lambda *args: lambda job: {"failed": 0, "ready_for_analysis": True})
    monkeypatch.setattr(dataio, "make_export_runner", export)
    monkeypatch.setattr(acceptance, "parquet_inventory", lambda root: {"file_count": 1, "rows": 2})
    monkeypatch.setattr(sources, "register_source", lambda *args, **kwargs: {"ok": True})
    return SOURCES[0], paths, calls, result


def test_exports_use_all_modules_all_stays_and_owner_defaults(tmp_path: Path, monkeypatch) -> None:
    from easyicu.webserver import dataio

    source, paths, calls, result = _fake_export_owners(tmp_path, monkeypatch)
    phases = {}
    acceptance.prepare_dataset(source, paths, phases, fetch=lambda *args: None)
    kwargs = calls[0]
    assert kwargs["modules"] is None and kwargs["concepts"] is None
    assert kwargs["max_patients"] is None
    assert kwargs["cohort"] == dataio.normalize_export_cohort_contract(None)
    assert kwargs["export_format"] == "parquet"
    assert kwargs["include_feature_definitions"] is True
    assert kwargs["merge"] is False
    assert phases["export"]["rows"] == result["total_rows"]
    for metrics in phases["export"]["modules"].values():
        assert metrics == {"mode": "one_shot", "batch_size": 2, "reason_code": "measured_profile_fast_path", "file_count": 1, "rows": 2}
    assert "private" not in json.dumps(phases)


@pytest.mark.parametrize("defect", ["modules", "schema", "cancelled"])
def test_incomplete_native_exports_fail_before_registration(tmp_path: Path, monkeypatch, defect: str) -> None:
    from easyicu.webserver import sources

    source, paths, _, result = _fake_export_owners(tmp_path, monkeypatch)
    if defect == "modules":
        result["module_resource_plans"].pop(next(iter(result["module_resource_plans"])))
    elif defect == "schema":
        (paths.export / "_manifest.json").write_text('{"schema_version":"legacy"}')
    else:
        result["cancelled_at"] = "modules"
    monkeypatch.setattr(sources, "register_source", lambda *args, **kwargs: pytest.fail("registered partial output"))
    phases = {}
    with pytest.raises(acceptance.AcceptanceFailure):
        acceptance.prepare_dataset(source, paths, phases, fetch=lambda *args: None)
    assert phases["export"]["status"] == "failed"


@pytest.mark.parametrize("failure", [{"failed": 1}, {"failed": 0, "ready_for_analysis": False}])
def test_incomplete_conversion_does_not_export(tmp_path: Path, monkeypatch, failure) -> None:
    from easyicu.webserver import dataio

    source, paths, calls, _ = _fake_export_owners(tmp_path, monkeypatch)
    monkeypatch.setattr(dataio, "make_convert_runner", lambda *args: lambda job: failure)
    with pytest.raises(acceptance.AcceptanceFailure, match="source_conversion_quality_incomplete"):
        acceptance.prepare_dataset(source, paths, {}, fetch=lambda *args: None)
    assert calls == []


def test_existing_log_handlers_cannot_publish_owner_diagnostics(tmp_path: Path) -> None:
    console = io.StringIO()
    logger = logging.getLogger()
    handler = logging.StreamHandler(console)
    logger.addHandler(handler)
    try:
        with (tmp_path / "owners.log").open("w") as log, acceptance.private_owner_logs(log):
            logging.warning("private synthetic diagnostic")
            print("private synthetic stdout")
        assert console.getvalue() == ""
        assert "private synthetic diagnostic" in (tmp_path / "owners.log").read_text()
        assert "private synthetic stdout" in (tmp_path / "owners.log").read_text()
        assert handler in logger.handlers
    finally:
        logger.removeHandler(handler)
        handler.close()


def test_failure_receipts_survive_an_unexpected_research_preflight_error(tmp_path: Path, monkeypatch) -> None:
    work = tmp_path / "EasyICU 数据"
    original_dir = Path.cwd()
    original_state = os.environ.get("EASYICU_HOME")
    fetches = []

    def fetch(source, paths, job):
        assert Path.cwd() == work
        assert os.environ["EASYICU_HOME"] == str(work / "home")
        fetches.append(source.database)
        raise acceptance.AcceptanceFailure("synthetic_download_unavailable")

    def failed_research():
        raise ValueError("must not appear in receipts")

    monkeypatch.setattr(acceptance, "probe_research", failed_research)
    monkeypatch.setattr(acceptance, "probe_web", lambda *args: pytest.fail("partial demo used"))
    report = acceptance.run_acceptance(work, work / "cache", tmp_path / "receipt", memory_pressure_gb=0, fetch=fetch)
    assert report["status"] == "failed"
    assert len(fetches) == 2
    assert report["research"]["status"] == "failed"
    assert report["memory"]["released"]["total_bytes"] > 0
    assert Path.cwd() == original_dir
    assert os.environ.get("EASYICU_HOME") == original_state
    assert (tmp_path / "receipt" / "summary.md").is_file()
    assert "must not appear" not in (tmp_path / "receipt" / "acceptance.json").read_text()


def test_health_timeout_closes_the_log_and_stops_the_wrapper(tmp_path: Path, monkeypatch) -> None:
    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def bind(self, address):
            assert address == ("127.0.0.1", 0)

        def getsockname(self):
            return "127.0.0.1", 4321

    process = SimpleNamespace(pid=42)
    commands, stopped = [], []

    def popen(command, **kwargs):
        commands.append(command)
        assert kwargs["cwd"] == tmp_path
        return process

    monkeypatch.setattr(acceptance.socket, "socket", Socket)
    monkeypatch.setattr(acceptance.subprocess, "Popen", popen)
    monkeypatch.setattr(acceptance, "stop_process_tree", lambda proc, children: stopped.append(proc))
    server = acceptance.LocalWebServer(tmp_path, startup_timeout=0)
    with pytest.raises(acceptance.AcceptanceFailure, match="webserver_health_timeout"):
        with server:
            pytest.fail("unhealthy server accepted")
    assert commands[0][1:4] == ["-m", "easyicu.webserver", "run"]
    assert stopped == [process]
    assert server.log.closed


def test_server_process_creation_failure_closes_the_log(tmp_path: Path, monkeypatch) -> None:
    socket = SimpleNamespace(bind=lambda address: None, getsockname=lambda: ("127.0.0.1", 4321))
    monkeypatch.setattr(acceptance.socket, "socket", lambda: nullcontext(socket))

    def popen(*args, **kwargs):
        raise OSError("synthetic process creation failure")

    monkeypatch.setattr(acceptance.subprocess, "Popen", popen)
    server = acceptance.LocalWebServer(tmp_path)
    with pytest.raises(OSError):
        with server:
            pytest.fail("failed server entered")
    assert server.log.closed


def test_stop_reaps_both_a_python_wrapper_and_its_child() -> None:
    script = """import subprocess, sys
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
print(child.pid, flush=True)
try:
    sys.stdin.read(1)
except KeyboardInterrupt:
    pass
finally:
    child.wait()
"""
    process = acceptance.subprocess.Popen(
        [acceptance.sys.executable, "-c", script], stdin=acceptance.subprocess.PIPE,
        stdout=acceptance.subprocess.PIPE, stderr=acceptance.subprocess.DEVNULL,
        creationflags=acceptance.subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    child_pid = int(process.stdout.readline())
    child = acceptance.psutil.Process(child_pid)
    try:
        acceptance.stop_process_tree(process, [child])
        assert process.poll() is not None
        assert not child.is_running()
    finally:
        if child.is_running():
            child.kill()
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdin.close()
        process.stdout.close()

