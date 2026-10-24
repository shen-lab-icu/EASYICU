#!/usr/bin/env python3
"""Opt-in public-demo laptop acceptance; receipts contain aggregates only.

Install with ``python scripts/launch_easyicu.py install``, then use that
launcher's runtime Python to run this file. No credentialed input, custom URL,
Docker installation, provider call, or change to production policy is needed.
The dictionary screen reads /api/catalog; its dictionary is checked separately.
Local logs and exports stay in the work directory and are never artifacts.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from http.client import IncompleteRead
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

import psutil

GIB = 1024**3
SAMPLE_SECONDS = 1.0


class AcceptanceFailure(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


def failure_receipt(exc: Exception) -> dict[str, str]:
    """Keep typed codes, never exception text, paths, URLs, rows or identifiers."""
    detail = getattr(exc, "detail", None)
    code = getattr(exc, "reason_code", None) or getattr(exc, "code", None)
    if not code and isinstance(detail, dict):
        code = detail.get("reason_code") or detail.get("error")
    if not isinstance(code, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,100}", code):
        code = "acceptance_phase_failed"
    return {"reason_code": code, "exception_type": type(exc).__name__}


def demo_cache_key(sources: Iterable[Any]) -> str:
    sources = tuple(sources)
    identity = [(source.database, source.version, source.archive_sha256) for source in sources]
    digest = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
    versions = "-".join(f"{source.database}-{source.version}" for source in sources)
    return f"official-demo-archives-v1-{versions}-{digest}"


def retry_download(operation: Callable, *, sleep: Callable = time.sleep) -> Any:
    """Three attempts for transient transport failures; archive/security errors fail."""
    for attempt in range(3):
        try:
            return operation()
        except (URLError, TimeoutError, ConnectionError, IncompleteRead) as exc:
            if isinstance(exc, HTTPError) and exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                raise
            if attempt == 2:
                raise
            sleep(2**attempt)


def memory_snapshot() -> dict[str, int]:
    memory = psutil.virtual_memory()
    return {"total_bytes": int(memory.total), "available_bytes": int(memory.available)}


def stop_process_tree(process: subprocess.Popen, known_children: Iterable[Any] = ()) -> None:
    """Ask uvicorn to shut down, then reap the wrapper and every captured child."""
    children = {child.pid: child for child in known_children}
    try:
        children.update({child.pid: child for child in psutil.Process(process.pid).children(recursive=True)})
    except psutil.Error:
        pass
    if process.poll() is None:
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                for child in children.values():
                    try:
                        child.send_signal(signal.SIGINT)
                    except psutil.Error:
                        pass
                process.send_signal(signal.SIGINT)
        except (OSError, ValueError):
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
    alive = [child for child in children.values() if child.is_running()]
    for child in alive:
        try:
            child.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(alive, timeout=5)
    for child in alive:
        try:
            child.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=5)
    if process.poll() is None:
        process.kill()
    process.wait(timeout=5)


def hold_memory(size: int) -> None:
    """A separate resident, touched allocation; stdin EOF releases every block."""
    blocks = []
    remaining = size
    while remaining:
        length = min(16 * 1024**2, remaining)
        blocks.append(bytearray(b"x") * length)
        remaining -= length
    print("ready", flush=True)
    sys.stdin.buffer.read(1)


class MemoryPressure:
    def __init__(self, gb: float, work_dir: Path) -> None:
        if not math.isfinite(gb) or gb < 0:
            raise ValueError("memory pressure must be finite and nonnegative")
        self.size = int(gb * GIB)
        self.work_dir = work_dir
        self.process: subprocess.Popen | None = None
        self.report: dict[str, Any] = {"requested_bytes": self.size}

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    def __enter__(self) -> MemoryPressure:
        self.report["before"] = memory_snapshot()
        try:
            if self.size:
                if self.size + 512 * 1024**2 > self.report["before"]["available_bytes"]:
                    raise AcceptanceFailure("memory_pressure_insufficient_headroom")
                self.process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--hold-memory-bytes", str(self.size)],
                    cwd=self.work_dir,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                )
                ready = []
                reader = threading.Thread(target=lambda: ready.append(self.process.stdout.readline()), daemon=True)
                reader.start()
                reader.join(timeout=120)
                if reader.is_alive() or not ready or ready[0].strip() != b"ready":
                    raise AcceptanceFailure("memory_pressure_helper_failed")
                self.report["helper_working_set_bytes"] = psutil.Process(self.process.pid).memory_info().rss
            self.report["after"] = memory_snapshot()
            self.report["under_8_gib_available"] = self.report["after"]["available_bytes"] < 8 * GIB
            if self.size >= 8 * GIB and not self.report["under_8_gib_available"]:
                raise AcceptanceFailure("memory_pressure_target_not_reached")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_: Any) -> None:
        if self.process is not None:
            if self.process.stdin is not None:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                stop_process_tree(self.process)
            if self.process.stdout is not None:
                self.process.stdout.close()
        self.report["released"] = memory_snapshot()


class PhaseMeasurement:
    """Sum resident working sets of this driver and descendants once per second.

    Windows psutil RSS is working set. Summed RSS on POSIX can count shared
    pages twice. The intentional pressure helper is excluded from phase peaks
    and reported separately, so an 8 GiB helper cannot hide product usage.
    """
    def __init__(self, *, excluded_pids: Iterable[int] = ()) -> None:
        self.excluded = set(excluded_pids)
        self.report: dict[str, Any] = {
            "seconds": 0.0, "peak_working_set_bytes": 0,
            "sample_interval_seconds": SAMPLE_SECONDS, "file_count": 0, "rows": 0,
        }
        self.done = threading.Event()

    def sample(self) -> None:
        root = psutil.Process()
        processes = [root, *root.children(recursive=True)]
        total = 0
        for process in processes:
            if process.pid in self.excluded:
                continue
            try:
                total += process.memory_info().rss
            except psutil.Error:
                pass
        self.report["peak_working_set_bytes"] = max(self.report["peak_working_set_bytes"], total)

    def _watch(self) -> None:
        while not self.done.wait(SAMPLE_SECONDS):
            self.sample()

    def __enter__(self) -> PhaseMeasurement:
        self.started = time.perf_counter()
        self.sample()
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.done.set()
        self.thread.join(timeout=2)
        self.sample()
        self.report["seconds"] = round(time.perf_counter() - self.started, 4)
        self.report["status"] = "passed" if exc is None else "failed"
        if exc is not None:
            self.report["failure"] = failure_receipt(exc)


@contextmanager
def phase(name: str, phases: dict, excluded_pids: Iterable[int] = ()):
    measurement = PhaseMeasurement(excluded_pids=excluded_pids)
    phases[name] = measurement.report
    with measurement:
        yield measurement.report


class AggregateJob:
    """Job protocol without retaining events, which can include local paths."""
    cancel_requested = False

    def emit(self, event: dict) -> bool:
        return True


def parquet_inventory(root: Path) -> dict[str, int]:
    import pyarrow.parquet as pq

    count, rows = 0, 0
    for path in root.rglob("*.parquet"):
        with pq.ParquetFile(path) as file:
            rows += file.metadata.num_rows
        count += 1
    return {"file_count": count, "rows": rows}


def prepare_dataset(source: Any, paths: Any, phases: dict, *, excluded_pids=(), fetch=None) -> Path:
    """Exercise the web conversion/export owners, without overriding their rules."""
    from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL
    from easyicu.webserver import dataio, sources
    from easyicu.webserver import demo_source_storage as storage

    job = AggregateJob()
    with phase("download_extract", phases, excluded_pids) as metrics:
        if fetch is None:
            digest, reused = retry_download(lambda: storage.download_archive(source, paths, job))
            storage.extract_archive(source, paths, digest, job)
            metrics["cache_hit"] = reused
        else:
            # Tests supply only synthetic raw tables at the download boundary.
            fetch(source, paths, job)
            metrics["cache_hit"] = False
        metrics["file_count"] = sum(path.is_file() for path in paths.raw.rglob("*"))

    with phase("convert", phases, excluded_pids) as metrics:
        result = dataio.make_convert_runner(str(paths.raw), source.database)(job)
        metrics.update(parquet_inventory(paths.raw))
        for key in ("converted", "skipped", "failed", "bad_rows_skipped"):
            metrics[key] = int(result.get(key) or 0)
        if result.get("failed") or result.get("ready_for_analysis") is False or not metrics["file_count"]:
            raise AcceptanceFailure("source_conversion_quality_incomplete")

    with phase("export", phases, excluded_pids) as metrics:
        cohort = dataio.normalize_export_cohort_contract(None)
        metrics["cohort"] = cohort
        try:
            result = dataio.make_export_runner(
                data_path=str(paths.raw), database=source.database,
                modules=None, concepts=None, export_format="parquet", merge=False,
                out_dir=str(paths.export), create_run_subdir=False, max_patients=None,
                cohort=cohort, include_feature_definitions=True,
            )(job)
        except Exception:
            # A failed owner can leave partial files. Count them without ever
            # registering them or manufacturing a success manifest.
            metrics.update(parquet_inventory(paths.export))
            raise
        if result.get("cancelled_at") or not result.get("manifest"):
            raise AcceptanceFailure("native_export_incomplete")
        manifest = json.loads((paths.export / result["manifest"]).read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "easyicu_native_export_v2":
            raise AcceptanceFailure("native_export_schema_mismatch")
        plans = result.get("module_resource_plans") or {}
        expected = {name for name, concepts in CONCEPT_GROUPS_INTERNAL.items() if concepts}
        if set(plans) != expected:
            raise AcceptanceFailure("native_export_modules_incomplete")
        metrics["modules"] = {
            name: {
                "mode": plan["mode"], "batch_size": int(plan["batch_size"]),
                "reason_code": plan["reason_code"],
                "file_count": sum(item.get("module") == name for item in result["files"]),
                "rows": sum(int(item.get("rows") or 0) for item in result["files"] if item.get("module") == name),
            }
            for name, plan in plans.items()
        }
        metrics.update(file_count=int(result["file_count"]), rows=int(result["total_rows"]))
        if not metrics["file_count"] or not metrics["rows"]:
            raise AcceptanceFailure("native_export_empty")

    with phase("register", phases, excluded_pids):
        result = sources.register_source(str(paths.export), label=source.title, active=True, crossdb=True)
        if not result.get("ok"):
            raise AcceptanceFailure("demo_registration_failed")
    return paths.export


def request_json(base_url: str, method: str, path: str, body: dict | None, *, timeout=120) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(base_url + path, data=data, method=method, headers={"Content-Type": "application/json"})
    # A clinician's proxy variables should not send loopback payloads elsewhere.
    opener = build_opener(ProxyHandler({}))
    try:
        response = opener.open(request, timeout=timeout)  # noqa: S310 - fixed loopback origin.
    except HTTPError as exc:
        response = exc
    with response:
        status = response.code
        payload = response.read()
    try:
        return status, json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return status, {}


def probe_read_paths(request: Callable, exports: list[Path], records: list[dict]) -> None:
    checks = [
        ("catalog", "GET", "/api/catalog", None),
        ("dictionary", "GET", "/api/catalog", None),
    ]
    for index, export in enumerate(exports, start=1):
        checks.extend([
            (f"cohort_{index}", "POST", "/api/cohort-review/summary", {"source_path": str(export)}),
            # With no entity_ref the real owner selects the first demo stay.
            (f"patient_{index}", "POST", "/api/patient-review/drilldown", {"source_path": str(export)}),
        ])
    checks.append(("crossdb", "POST", "/api/crossdb-review/summary", {"paths": [str(path) for path in exports]}))
    failed = False
    for name, method, path, body in checks:
        started = time.perf_counter()
        record = {"read": name, "status_code": None}
        try:
            status, payload = request(method, path, body)
            record["status_code"] = status
            valid = isinstance(payload, dict) and payload.get("ok") is not False
            if name in {"catalog", "dictionary"}:
                valid = valid and bool(payload.get("dict"))
            record["passed"] = 200 <= status < 300 and valid
        except (OSError, ValueError) as exc:
            record.update(passed=False, failure=failure_receipt(exc))
        record["seconds"] = round(time.perf_counter() - started, 4)
        records.append(record)
        failed |= not record["passed"]
    if failed:
        raise AcceptanceFailure("web_read_failed")


def check_server_log(path: Path) -> None:
    with path.open(encoding="utf-8", errors="replace") as log:
        if any("traceback (most recent call last)" in line.lower() for line in log):
            raise AcceptanceFailure("server_traceback")


class LocalWebServer:
    def __init__(self, work_dir: Path, *, startup_timeout: float = 120) -> None:
        self.work_dir = work_dir
        self.startup_timeout = startup_timeout
        self.log_path = work_dir / "webserver.log"
        self.children: dict[int, Any] = {}

    def request(self, method, path, body):
        return request_json(self.base_url, method, path, body)

    def __enter__(self) -> LocalWebServer:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self.log = self.log_path.open("w", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "easyicu.webserver", "run", "--host", "127.0.0.1", "--port", str(port)],
                cwd=self.work_dir, stdout=self.log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
        except BaseException:
            self.log.close()
            raise
        try:
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                try:
                    self.children.update({child.pid: child for child in psutil.Process(self.process.pid).children(recursive=True)})
                except psutil.Error:
                    pass
                if self.process.poll() is not None:
                    raise AcceptanceFailure("webserver_exited_before_health")
                try:
                    status, _ = request_json(self.base_url, "GET", "/api/health", None, timeout=2)
                    if status == 200:
                        return self
                except OSError:
                    pass
                time.sleep(1)
            raise AcceptanceFailure("webserver_health_timeout")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_: Any) -> None:
        try:
            stop_process_tree(self.process, self.children.values())
        finally:
            self.log.close()


def probe_web(work_dir: Path, exports: list[Path], metrics: dict) -> None:
    metrics["reads"] = []
    try:
        with LocalWebServer(work_dir) as server:
            probe_read_paths(server.request, exports, metrics["reads"])
    finally:
        log = work_dir / "webserver.log"
        if log.exists():
            try:
                check_server_log(log)
            except AcceptanceFailure:
                metrics["traceback_found"] = True
                raise
        metrics["traceback_found"] = False


def probe_research() -> dict:
    from easyicu.research_agent.orchestration.profiles import get_submission_profile
    from easyicu.webserver import agent_runs
    from easyicu.webserver.research_launch_runtime import _require_execution_runtime, _submission_profile_ref
    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

    profiles = [get_submission_profile(_submission_profile_ref(budget_mode="full_reviewed", live_pubmed=value)) for value in (False, True)]
    docker_required = any(not profile.planner_only and profile.requires_runner == "docker" for profile in profiles)
    result = {
        "owner": "easyicu.webserver.research_launch_runtime._require_execution_runtime",
        "budget_mode": "full_reviewed", "available": True, "reason_code": "ready",
        # Governed deterministic execution uses the same mandated runner.
        "deterministic_analysis_without_docker": "no" if docker_required else "yes",
        "deterministic_analysis_owner": "easyicu.webserver.research_launch_runtime._require_execution_runtime",
    }
    try:
        # No workdir: this only inspects availability; it cannot run a container.
        _require_execution_runtime(budget_mode="full_reviewed", runner_image="")
    except ResearchPipelineRunError as exc:
        result.update(available=False, reason_code=exc.details.get("reason_code"), code=exc.code)
    provider = agent_runs.resolve_agent_provider_config(
        run_type="preflight", llm_provider="mock", external_llm_opt_in=False, ai_enabled=False, environ={},
    )
    result["local_preflight_without_docker"] = "yes" if not provider.get("external") else "no"
    result["local_preflight_owner"] = "easyicu.webserver.agent_runs.resolve_agent_provider_config"
    result["local_preflight_scope"] = "export_summary_and_quality_audit"
    result["analysis_executed"] = False
    return result


def validate_work_dir(work_dir: Path) -> None:
    path = str(work_dir)
    if " " not in path or not re.search(r"[\u4e00-\u9fff]", path):
        raise AcceptanceFailure("laptop_path_required")


def write_reports(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "acceptance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    lines = [f"Windows laptop acceptance: **{report['status']}**", ""]
    memory = report.get("memory", {})
    if memory:
        lines.append(f"Memory pressure: {memory['requested_bytes'] / GIB:g} GiB; phase peaks exclude this helper.")
        for name in ("before", "after"):
            if name in memory:
                lines.append(f"{name.capitalize()}: {memory[name]['available_bytes'] / GIB:.2f} GiB available / {memory[name]['total_bytes'] / GIB:.2f} GiB total.")
    lines.extend(["", "| Demo / phase | Status | Seconds | Peak GiB | Files | Rows |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for dataset in report.get("datasets", []):
        for name, metrics in dataset["phases"].items():
            lines.append(f"| {dataset['database']} {dataset['version']} / {name} | {metrics['status']} | {metrics['seconds']:.2f} | {metrics['peak_working_set_bytes'] / GIB:.3f} | {metrics['file_count']} | {metrics['rows']} |")
    for read in report.get("web", {}).get("reads", []):
        lines.append(f"\n{read['read']}: HTTP {read['status_code']}, {read['seconds']:.3f} s.")
    research = report.get("research")
    if research:
        code = research.get("reason_code") or (research.get("failure") or {}).get("reason_code")
        lines.append(f"\nResearch preflight: {code}; governed deterministic analysis without Docker: {research.get('deterministic_analysis_without_docker', 'not_completed')}; local summary preflight: {research.get('local_preflight_without_docker', 'not_completed')}.")
    for failure in report.get("failures", []):
        lines.append(f"\nFailure: `{failure['reason_code']}` ({failure['exception_type']}).")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@contextmanager
def laptop_environment(work_dir: Path, cache_dir: Path):
    """Keep CLI state in the test folder and restore embedding callers' state."""
    values = {
        "EASYICU_HOME": str(work_dir / "home"),
        "EASYICU_RUNTIME_DIR": str(work_dir / "runtime"),
        "EASYICU_DEMO_CACHE_DIR": str(cache_dir),
        "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1", "EASYICU_VERBOSE": "0",
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
    }
    previous = {key: os.environ.get(key) for key in values}
    previous_dir = Path.cwd()
    try:
        os.environ.update(values)
        os.chdir(work_dir)
        yield
    finally:
        os.chdir(previous_dir)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def private_owner_logs(log):
    """Existing logging handlers must not send local diagnostics to CI output."""
    root = logging.getLogger()
    previous_handlers, previous_level = root.handlers[:], root.level
    handler = logging.StreamHandler(log)
    root.handlers = [handler]
    try:
        with redirect_stdout(log), redirect_stderr(log):
            yield
    finally:
        root.handlers = previous_handlers
        root.setLevel(previous_level)
        handler.close()


def run_acceptance(work_dir: Path, cache_dir: Path, output_dir: Path, *, memory_pressure_gb=8, fetch=None) -> dict:
    report: dict[str, Any] = {
        "schema_version": "easyicu_laptop_acceptance_v1", "status": "failed",
        "platform": sys.platform, "python": sys.version.split()[0],
        "started_utc": datetime.now(timezone.utc).isoformat(), "datasets": [], "failures": [],
    }
    output_dir = output_dir.resolve()
    try:
        work_dir, cache_dir = work_dir.resolve(), cache_dir.resolve()
        validate_work_dir(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        with laptop_environment(work_dir, cache_dir):
            # Set the isolated state root before importing owners with path constants.
            from easyicu.webserver import demo_source_storage as storage
            from easyicu.webserver.demo_source_contracts import SOURCES

            report["demo_cache_key"] = demo_cache_key(SOURCES)
            pressure = MemoryPressure(memory_pressure_gb, work_dir)
            report["memory"] = pressure.report
            with pressure, (work_dir / "owners.log").open("w", encoding="utf-8") as log, private_owner_logs(log):
                excluded = (pressure.pid,) if pressure.pid else ()
                exports = []
                for source in SOURCES:
                    dataset = {"database": source.database, "version": source.version, "phases": {}}
                    report["datasets"].append(dataset)
                    try:
                        exports.append(prepare_dataset(source, storage.source_paths(source), dataset["phases"], excluded_pids=excluded, fetch=fetch))
                    except Exception as exc:
                        report["failures"].append(failure_receipt(exc))
                try:
                    with phase("web", report, excluded) as metrics:
                        if len(exports) != len(SOURCES):
                            raise AcceptanceFailure("demo_exports_unavailable_for_web")
                        probe_web(work_dir, exports, metrics)
                except Exception as exc:
                    report["failures"].append(failure_receipt(exc))
                try:
                    with phase("research", report, excluded) as metrics:
                        metrics.update(probe_research())
                except Exception as exc:
                    report["failures"].append(failure_receipt(exc))
        report["status"] = "failed" if report["failures"] else "passed"
    except Exception as exc:
        report["failures"].append(failure_receipt(exc))
    finally:
        write_reports(report, output_dir)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-pressure-gb", type=float, default=8)
    parser.add_argument("--work-dir", type=Path, default=Path.cwd() / "EasyICU 数据")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "acceptance-results")
    parser.add_argument("--cache-key", action="store_true")
    parser.add_argument("--hold-memory-bytes", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.hold_memory_bytes is not None:
        hold_memory(args.hold_memory_bytes)
        return 0
    if args.cache_key:
        from easyicu.webserver.demo_source_contracts import SOURCES
        print(demo_cache_key(SOURCES))
        return 0
    report = run_acceptance(
        args.work_dir, args.cache_dir or args.work_dir / "demo-cache", args.output_dir,
        memory_pressure_gb=args.memory_pressure_gb,
    )
    print(f"Laptop acceptance: {report['status']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
