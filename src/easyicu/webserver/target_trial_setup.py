"""The host's compile job for the target trial a causal study states.

Owner
-----
Copilot states the trial and its population from the host's menus
(``easyicu_state_target_trial``).  This module checks the statement, starts
one compile job for the study and keeps what the job compiled:

* the job acquires the analysis universe a run of the study acquires for the
  trial -- the same concepts, covariates summarized over ``[0, T0)`` and the
  treatment's onsets read over ``[0, T0 + G)`` -- through Data Extraction's
  acquisition owner, with no model choosing a concept
  (``research_launch_scientific.target_trial_launch_inputs``);
* it builds the research context a run builds on that universe
  (``agent_pipeline_runs.target_trial_context_arguments``), compiles the
  population at the trial's time zero and the trial on it, keeps the record
  by its digest (``target_trial_records``) and states the trial in the study
  without an approval (``study_contexts.bind_target_trial_design``);
* a stop before the record is typed and says what comes next: a database v1
  does not emulate trials in, a study whose setup a run would refuse (its
  launch's own code is kept), data the acquisition cannot read, a population
  that does not compile, a study that changed while the job ran.

A prepared package keeps every row of each stay it holds: its recorded
observation window selects stays only for a concept-derived cohort
(``dataio.export_rows_decided_by``), so the trial's windows are read from the
package whatever window it was extracted over.

One compile runs at a time.  The study's job pointer holds the study while it
runs, as an extraction's does.  The request and the result are kept with the
job, so the conversation and the card can say what was asked and why it did
not compile.  Nothing here calls a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import resource
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from pydantic import BaseModel, ValidationError

from easyicu.databases import normalize_database_key
from easyicu.research_agent.planning.analysis_types import canonical_analysis_family
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.planning.target_trial_spec import TargetTrialSpec
from easyicu.webserver import dataio, jobs, study_contexts
from easyicu.webserver.target_trial_records import keep_target_trial_record, records_root

#: The Copilot tool that states the trial; the conversation's tool catalog lists it.
TARGET_TRIAL_STATE_TOOL = "easyicu_state_target_trial"
TARGET_TRIAL_COMPILE_JOB_KIND = "target-trial-compile"
TARGET_TRIAL_COMPILE_JOB_SCHEMA_VERSION = "easyicu.target_trial_compile_job/1"
#: The tool result that hands the conversation its compile job.
TARGET_TRIAL_COMPILE_SUBMITTED = "easyicu_target_trial_compile_submitted"
#: Why the host does not start a compile.  Stable codes.
TARGET_TRIAL_SETUP_REFUSALS = (
    "target_trial_statement_invalid",
    "target_trial_family_mismatch",
    "target_trial_export_required",
    "target_trial_compile_busy",
)
#: Why a compile stopped before its record: what comes next, not an error.
#: Stable codes.
TARGET_TRIAL_COMPILE_STOPS = (
    "target_trial_database_out_of_scope",
    "target_trial_study_not_ready",
    "target_trial_data_unavailable",
    "target_trial_population_not_compiled",
    "target_trial_study_changed",
)
#: A compile that failed unexpectedly; the job keeps the lower layer's code.
TARGET_TRIAL_COMPILE_FAILED = "target_trial_compile_failed"
#: A compile job the host no longer runs that left no result: the host
#: stopped while it ran.  Stable.
TARGET_TRIAL_COMPILE_INTERRUPTED = "target_trial_compile_interrupted"
#: The databases v1 emulates trials in.
TARGET_TRIAL_DATABASES = frozenset({"miiv"})
#: A study bound to a database v1 does not emulate trials in.  Stable.
TARGET_TRIAL_DATABASE_OUT_OF_SCOPE = "target_trial_database_out_of_scope"
_CAUSAL_FAMILY = "causal_inference"
_OWNER = "easyicu.webserver.target_trial_setup"
_MAX_JOB_RECORD_BYTES = 256 * 1024
_JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CODE = re.compile(r"[a-z][a-z0-9_]{2,120}")
#: A lower layer's code or exception type, as a failed compile keeps it.
_CAUSE = re.compile(r"[A-Za-z][A-Za-z0-9_.]{1,120}")
_STATEMENT_FIELDS = (("spec", TargetTrialSpec), ("population_spec", PopulationSpec))
#: One compile at a time: it reads a study's data as an extraction does.
_COMPILE_GATE = threading.BoundedSemaphore(1)
_RECORD_LOCK = threading.Lock()


class TargetTrialSetupError(RuntimeError):
    """The host does not start a compile; the statement stays the conversation's."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 409,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.details = dict(details or {})

    def tool_result(self) -> dict[str, Any]:
        """The blocked result the conversation's tool call shows for this refusal."""

        return {
            "status": "blocked",
            "code": self.code,
            "summary": str(self),
            "owner": _OWNER,
            "details": dict(self.details),
        }


class TargetTrialCompileFailure(RuntimeError):
    """A compile that failed unexpectedly: a stable code, with the cause kept."""

    code = "target_trial_compile_failed"

    def __init__(self, cause_code: str) -> None:
        super().__init__(f"The compile failed unexpectedly ({cause_code}).")
        self.cause_code = cause_code


@dataclass(frozen=True)
class TargetTrialStatement:
    """The trial and the population Copilot stated, as the host's models read them."""

    spec: TargetTrialSpec
    population_spec: PopulationSpec

    def request(self) -> dict[str, Any]:
        body = {
            "spec": self.spec.model_dump(mode="json"),
            "population_spec": self.population_spec.model_dump(mode="json"),
        }
        raw = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return {**body, "request_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest()}


def read_target_trial_statement(params: Mapping[str, Any]) -> TargetTrialStatement:
    """The statement, or the field paths the host's models refuse in it."""

    parsed: dict[str, BaseModel] = {}
    fields: list[str] = []
    for key, model in _STATEMENT_FIELDS:
        try:
            parsed[key] = model.model_validate(params.get(key))
        except ValidationError as exc:
            for error in exc.errors(include_url=False, include_input=False):
                location = [str(part) for part in error.get("loc") or ()]
                fields.append(".".join([key, *location]))
    if fields:
        raise TargetTrialSetupError(
            "target_trial_statement_invalid",
            "The stated trial or population breaks the host's menus; restate the "
            "fields named.",
            status_code=422,
            details={"fields": sorted(dict.fromkeys(fields))[:16]},
        )
    return TargetTrialStatement(
        spec=parsed["spec"],  # type: ignore[arg-type]
        population_spec=parsed["population_spec"],  # type: ignore[arg-type]
    )


def _study_key(study_id: str) -> str:
    return hashlib.sha256(str(study_id).encode("utf-8")).hexdigest()[:24]


def _jobs_dir(study_id: str) -> Path:
    return records_root() / _study_key(study_id) / "jobs"


def _job_record_path(study_id: str, job_id: str, created_ns: int) -> Path:
    # Named by when the job started: the newest name is the latest job.
    return _jobs_dir(study_id) / f"{created_ns:020d}-{job_id}.json"


def _write_job_record(path: Path, record: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if len(encoded) > _MAX_JOB_RECORD_BYTES:
        raise ValueError("a compile job record exceeds its bounded size")
    with _RECORD_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(path.parent),
            prefix=".target-trial-job-",
            suffix=".tmp",
            delete=False,
        )
        temporary = Path(handle.name)
        try:
            with handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _read_job_record(path: Path) -> Optional[dict[str, Any]]:
    """One job's record as the host kept it, or ``None`` for anything else."""

    try:
        if path.stat().st_size > _MAX_JOB_RECORD_BYTES:
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (
        isinstance(record, dict)
        and record.get("schema_version") == TARGET_TRIAL_COMPILE_JOB_SCHEMA_VERSION
        and path.stem.partition("-")[2] == record.get("job_id")
    ):
        return record
    return None


def _job_paths(study_id: str) -> list[Path]:
    directory = _jobs_dir(study_id)
    return sorted(directory.glob("*.json")) if directory.is_dir() else []


def latest_target_trial_compile(study_id: str) -> Optional[dict[str, Any]]:
    """The study's latest compile job.

    ``{job_id, status, reason_code, compile_sha256, detail, missing_concepts,
    cause_code}``: ``status`` is ``running`` while the job runs, then the
    result's ``compiled`` or ``stopped``, ``failed`` for an unexpected error,
    and ``interrupted`` (:data:`TARGET_TRIAL_COMPILE_INTERRUPTED`) for a job
    this process no longer runs that left no result.  ``missing_concepts``
    names what a stop for unavailable data could not read, and ``cause_code``
    the lower layer's code of a failure: bounded codes, never a value.
    """

    latest = next(
        (
            record
            for record in map(_read_job_record, reversed(_job_paths(study_id)))
            if record is not None
        ),
        None,
    )
    if latest is None:
        return None
    result = latest.get("result") if isinstance(latest.get("result"), Mapping) else None
    found: Mapping[str, Any] = result or {}
    if result is not None:
        status = str(result.get("status") or "failed")
        reason, detail = result.get("reason_code"), result.get("detail")
    else:
        job = jobs.MANAGER.get(str(latest["job_id"]))
        running = job is not None and job.status == "running"
        status = "running" if running else "interrupted"
        reason, detail = (
            (None, None)
            if running
            else (
                TARGET_TRIAL_COMPILE_INTERRUPTED,
                "The compile job ended without a result; state the trial again.",
            )
        )
    cause = str(found.get("cause_code") or "")
    return {
        "job_id": str(latest["job_id"]),
        "status": status,
        "reason_code": reason,
        "compile_sha256": found.get("compile_sha256"),
        "detail": detail,
        "missing_concepts": _bounded_codes(found.get("missing_concepts")),
        "cause_code": cause if _CAUSE.fullmatch(cause) else None,
    }


def target_trial_job_record(study_id: str, job_id: str) -> Optional[dict[str, Any]]:
    """What one compile job was asked and what it found, as the host kept it."""

    if not _JOB_ID.fullmatch(str(job_id or "")):
        return None
    directory = _jobs_dir(study_id)
    for path in sorted(directory.glob(f"*-{job_id}.json")) if directory.is_dir() else ():
        record = _read_job_record(path)
        if record is not None:
            return record
    return None


def target_trial_family_declared(study: Mapping[str, Any]) -> bool:
    design = study.get("analysis_design")
    if not isinstance(design, Mapping):
        return False
    return canonical_analysis_family(design.get("analysis_family")) == _CAUSAL_FAMILY


def _bound_source(study: Mapping[str, Any]) -> Mapping[str, Any]:
    source = study.get("data_source")
    return source if isinstance(source, Mapping) else {}


def _study_database(study: Mapping[str, Any]) -> str:
    """The bound database's registry key, as the launch reads it."""

    raw = str(_bound_source(study).get("database") or "").strip()
    if not raw:
        return ""
    try:
        return normalize_database_key(raw)
    except KeyError:
        # A database the registry does not know is no trial database either;
        # the launch refuses it by its own code.
        return raw.lower()


def target_trial_database_gap(study: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """Why v1 emulates no trial in the study's bound database, or ``None``.

    A study bound to no database has no gap yet: the compile refuses it when
    it is bound.  The compile refuses a bound database with the same code.
    """

    database = _study_database(study)
    if not database or database in TARGET_TRIAL_DATABASES:
        return None
    return {
        "code": TARGET_TRIAL_DATABASE_OUT_OF_SCOPE,
        "database": database,
        "supported_databases": sorted(TARGET_TRIAL_DATABASES),
    }


def _bound_export(study: Mapping[str, Any]) -> tuple[str, str]:
    path = str(_bound_source(study).get("path") or "").strip()
    database = _study_database(study)
    if not path or dataio.read_prepared_export_manifest(path) is None:
        raise TargetTrialSetupError(
            "target_trial_export_required",
            "The study has no prepared data package to compile the trial on; "
            "extract its data first.",
        )
    return path, database


def _stopped(code: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"status": "stopped", "reason_code": code, "detail": detail, **extra}


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Bytes on macOS, kilobytes on Linux.
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


def _launch_refused(exc: Any) -> dict[str, Any]:
    """The stop for a launch a run of the study would refuse the same way."""

    concept = exc.details.get("concept_id")
    if concept:
        return _stopped(
            "target_trial_data_unavailable",
            "The study's data modules do not provide a concept the trial reads.",
            missing_concepts=[str(concept)],
            cause_code=exc.code,
        )
    return _stopped(
        "target_trial_study_not_ready",
        "A run of this study would refuse to start; complete its setup first.",
        cause_code=exc.code,
    )


def _compile(
    job: Any,
    *,
    study: Mapping[str, Any],
    statement: TargetTrialStatement,
    export_path: str,
    database: str,
    revision: int,
) -> dict[str, Any]:
    """Acquire, build the context, compile and keep; or the typed stop on the way.

    The launch, the acquisition and the research context are the ones a run
    of the study prepares once the trial is approved
    (``prepare_target_trial_launch``, ``target_trial_research_context``), so
    the run compiles the approved record again on its own context.
    """

    study_id = str(study.get("id"))
    if database not in TARGET_TRIAL_DATABASES:
        return _stopped(
            TARGET_TRIAL_DATABASE_OUT_OF_SCOPE,
            f"Target trials are emulated in {', '.join(sorted(TARGET_TRIAL_DATABASES))} "
            f"in this version, not in {database or 'this database'}.",
        )
    from easyicu.research_agent.planning.population_compile import compile_population
    from easyicu.research_agent.planning.target_trial_compile import (
        compile_target_trial,
        target_trial_context_endpoint,
    )
    from easyicu.research_agent.planning.target_trial_configuration import (
        TargetTrialCompileRecord,
    )
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    # The run's owner: imported here, as the Copilot's service imports this
    # module and the run's module imports the Copilot's contracts.
    from easyicu.webserver.agent_pipeline_runs import (
        acquire_target_trial_universe,
        target_trial_research_context,
    )
    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
    from easyicu.webserver.research_pipeline_run_preparation import (
        prepare_target_trial_launch,
    )

    try:
        prepared = prepare_target_trial_launch(
            export_path=export_path,
            study=study,
            spec=statement.spec,
            population_spec=statement.population_spec,
        )
    except ResearchPipelineRunError as exc:
        return _launch_refused(exc)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="easyicu-target-trial-") as scratch:
        try:
            acquisition = acquire_target_trial_universe(
                prepared,
                export_path=export_path,
                # The host selects every concept: a call to it is a defect.
                llm=ScriptedMockLLMClient([]),
                output_dir=Path(scratch) / "pipeline_input",
            )
        except ResearchPipelineRunError as exc:
            return _launch_refused(exc)
        acquisition_seconds = round(time.monotonic() - started, 1)
        if acquisition.blocked or acquisition.universe_path is None:
            return _stopped(
                "target_trial_data_unavailable",
                "The data package does not provide what the trial reads.",
                missing_concepts=sorted(str(c) for c in acquisition.missing_concepts or ()),
            )
        compiling = time.monotonic()
        try:
            # The endpoint the signed suite binds onto the run's context.
            context = target_trial_research_context(
                prepared,
                acquisition,
                export_path=export_path,
                endpoint=target_trial_context_endpoint(statement.spec),
            )
        except ResearchPipelineRunError as exc:
            return _launch_refused(exc)
        time_zero = statement.spec.time_zero.hours_after_icu_admission
        try:
            population = compile_population(
                statement.population_spec, context, time_zero_hours=time_zero
            )
        except ValueError:
            return _stopped(
                "target_trial_population_not_compiled",
                "The stated population does not compile at the trial's time zero.",
            )
        compiled = compile_target_trial(statement.spec, context, population=population)
    kept = TargetTrialCompileRecord.of(compiled, statement.population_spec)
    keep_target_trial_record(study_id, kept)
    try:
        study_contexts.bind_target_trial_design(
            study_id, kept.design(), expected_revision=revision
        )
    except study_contexts.StudyContextError as exc:
        if exc.detail.get("error") != "study_context_revision_conflict":
            raise
        return _stopped(
            "target_trial_study_changed",
            "The study changed while the trial compiled; state the trial again.",
        )
    return {
        "status": "compiled",
        "reason_code": None,
        "detail": (
            "The trial compiled; its card lists what to confirm."
            if kept.approvable
            else "The trial compiled; its card lists what holds approval."
        ),
        "compile_sha256": kept.compile_sha256,
        "approvable": kept.approvable,
        "metrics": {
            "acquisition_seconds": acquisition_seconds,
            "compile_seconds": round(time.monotonic() - compiling, 1),
            "peak_rss_mb": _peak_rss_mb(),
        },
    }


def _startable(study: Mapping[str, Any]) -> tuple[str, str]:
    """The study's bound export and database, or why no compile starts for it."""

    if not target_trial_family_declared(study):
        raise TargetTrialSetupError(
            "target_trial_family_mismatch",
            "Only a study whose design is causal inference states a target trial.",
        )
    export_path, database = _bound_export(study)
    if str(study.get("active_job_id") or "").strip():
        raise TargetTrialSetupError(
            "study_job_running", "The study already runs a job; wait for it."
        )
    return export_path, database


def _compile_busy() -> TargetTrialSetupError:
    return TargetTrialSetupError(
        "target_trial_compile_busy",
        "Another target trial is compiling; state this one when it finishes.",
    )


def check_target_trial_statement(
    study: Mapping[str, Any], params: Mapping[str, Any]
) -> TargetTrialStatement:
    """The statement a compile can start from, or why the host refuses it.

    Every refusal comes before the conversation spends its grant: a study of
    another family, a statement the host's menus refuse, a study without a
    prepared data package, a study whose own job runs, a compile already
    running.  :func:`submit_target_trial_compile` checks again as it starts.
    """

    if not target_trial_family_declared(study):
        _startable(study)
    statement = read_target_trial_statement(params)
    _startable(study)
    if not _COMPILE_GATE.acquire(blocking=False):
        raise _compile_busy()
    _COMPILE_GATE.release()
    return statement


def submit_target_trial_compile(
    study: Mapping[str, Any], statement: TargetTrialStatement
) -> Any:
    """Start the study's compile job; the job states the trial when it compiles.

    Refused before any job: a study of another family, a study without a
    prepared data package, a study whose own job runs, a compile already
    running, a study changed since it was read.
    """

    study_id = str(study.get("id") or "").strip()
    export_path, database = _startable(study)
    if not _COMPILE_GATE.acquire(blocking=False):
        raise _compile_busy()
    start_gate = threading.Event()
    started: dict[str, Any] = {}
    request = statement.request()
    # The compile moves no lifecycle stage: clearing its pointer keeps them.
    stage = str(study.get("current_stage") or "plan")
    route = str(study.get("last_route") or "entry")

    def runner(job: Any) -> dict[str, Any]:
        try:
            start_gate.wait()
            if "error" in started:
                raise RuntimeError(f"target_trial_compile_start_blocked:{started['error']}")
            path, record = started["path"], started["record"]
            try:
                result = _compile(
                    job,
                    study=started["study"],
                    statement=statement,
                    export_path=export_path,
                    database=database,
                    revision=int(started["study"].get("revision") or 0),
                )
            except Exception as exc:
                cause = str(getattr(exc, "code", "") or type(exc).__name__)[:120]
                failed = {
                    "status": "failed",
                    "reason_code": TARGET_TRIAL_COMPILE_FAILED,
                    "detail": "The compile failed unexpectedly; state the trial again.",
                    "cause_code": cause,
                }
                _write_job_record(path, {**record, "result": failed})
                raise TargetTrialCompileFailure(cause) from exc
            _write_job_record(path, {**record, "result": result})
            return {"target_trial_compile": result}
        finally:
            _COMPILE_GATE.release()
            try:
                study_contexts.clear_active_job_if(
                    study_id, job.id, current_stage=stage, last_route=route
                )
            except Exception:
                pass

    try:
        job = jobs.MANAGER.submit(TARGET_TRIAL_COMPILE_JOB_KIND, runner)
    except jobs.JobCapacityError as exc:
        _COMPILE_GATE.release()
        raise TargetTrialSetupError(
            "job_capacity_exceeded",
            "Wait for a running local job to finish before stating the trial again.",
            status_code=429,
            details={"running": exc.running, "max_running": exc.max_running},
        ) from exc
    except BaseException:
        _COMPILE_GATE.release()
        raise
    try:
        started["study"] = study_contexts.handoff_context(
            study_id,
            active_job_id=job.id,
            expected_revision=int(study.get("revision") or 0),
        )
        # Kept before the tool answers, so the card shows the compile at once.
        started["path"] = _job_record_path(study_id, job.id, time.time_ns())
        started["record"] = {
            "schema_version": TARGET_TRIAL_COMPILE_JOB_SCHEMA_VERSION,
            "job_id": job.id,
            "created_at": time.time(),
            "request": request,
            "result": None,
        }
        _write_job_record(started["path"], started["record"])
    except study_contexts.StudyContextError as exc:
        started["error"] = str(exc.detail.get("error") or "study_context_sync_failed")
    except (OSError, ValueError):
        started["error"] = TARGET_TRIAL_COMPILE_FAILED
    finally:
        start_gate.set()
    if "error" in started:
        raise TargetTrialSetupError(
            started["error"], "The compile could not start; state the trial again."
        )
    return job


def submitted_tool_result(study: Mapping[str, Any], job: Any) -> dict[str, Any]:
    """The tool result that hands the conversation its compile job."""

    return {
        "status": "ok",
        "code": TARGET_TRIAL_COMPILE_SUBMITTED,
        "summary": (
            f"Started EasyICU target trial compile job {job.id}. When it compiles, "
            "the study's card lists what to confirm; nothing is approved until the "
            "researcher approves the card."
        ),
        "owner": _OWNER,
        "details": {
            "job_id": job.id,
            "kind": job.kind,
            "status": job.status,
            "study_context_id": str(study.get("id") or ""),
        },
    }


def _bounded_codes(values: Any, limit: int = 32) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    codes = [str(value) for value in values if _CODE.fullmatch(str(value))]
    return codes[:limit]


def _hours(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def project_target_trial_compile(raw: Any) -> Optional[dict[str, Any]]:
    """A compile job's result as the conversation shows it: codes and numbers only."""

    if not isinstance(raw, Mapping) or raw.get("status") not in {
        "compiled",
        "stopped",
        "failed",
    }:
        return None
    reason = str(raw.get("reason_code") or "")
    projected: dict[str, Any] = {
        "status": raw["status"],
        "reason_code": reason if _CODE.fullmatch(reason) else None,
        "detail": str(raw.get("detail") or "")[:400],
    }
    digest = str(raw.get("compile_sha256") or "")
    if _SHA256.fullmatch(digest):
        projected["compile_sha256"] = digest
    if isinstance(raw.get("approvable"), bool):
        projected["approvable"] = raw["approvable"]
    if "missing_concepts" in raw:
        projected["missing_concepts"] = _bounded_codes(raw.get("missing_concepts"))
    cause = str(raw.get("cause_code") or "")
    if _CAUSE.fullmatch(cause):
        projected["cause_code"] = cause
    metrics = raw.get("metrics")
    if isinstance(metrics, Mapping):
        projected["metrics"] = {
            key: value
            for key in ("acquisition_seconds", "compile_seconds", "peak_rss_mb")
            if (value := _hours(metrics.get(key))) is not None
        }
    return projected


def _inline_refs(node: Any, definitions: Mapping[str, Any], seen: tuple[str, ...]) -> Any:
    if isinstance(node, list):
        return [_inline_refs(item, definitions, seen) for item in node]
    if not isinstance(node, Mapping):
        return node
    reference = node.get("$ref")
    if isinstance(reference, str):
        name = reference.rsplit("/", 1)[-1]
        if name in seen or name not in definitions:
            raise ValueError(f"the statement schema cannot inline {reference}")
        inlined = _inline_refs(definitions[name], definitions, (*seen, name))
        siblings = {key: value for key, value in node.items() if key != "$ref"}
        return {**inlined, **_inline_refs(siblings, definitions, seen)}
    # A discriminator's mapping names the definitions inlined here; the
    # alternatives' own constant tags keep them apart.
    return {
        key: _inline_refs(value, definitions, seen)
        for key, value in node.items()
        if key not in {"$defs", "discriminator"}
    }


def target_trial_statement_schema() -> dict[str, Any]:
    """The tool's arguments, ``{spec, population_spec}``, as one JSON schema.

    Generated from the host's own models, with every reference inlined, so
    the conversation's tool and the host read the same menus.
    """

    properties: dict[str, Any] = {}
    for key, model in _STATEMENT_FIELDS:
        schema = model.model_json_schema(mode="validation")
        properties[key] = _inline_refs(schema, schema.get("$defs") or {}, ())
    return {
        "type": "object",
        "properties": properties,
        "required": [key for key, _ in _STATEMENT_FIELDS],
        "additionalProperties": False,
    }


__all__ = [
    "TARGET_TRIAL_COMPILE_FAILED",
    "TARGET_TRIAL_COMPILE_INTERRUPTED",
    "TARGET_TRIAL_COMPILE_JOB_KIND",
    "TARGET_TRIAL_COMPILE_JOB_SCHEMA_VERSION",
    "TARGET_TRIAL_COMPILE_STOPS",
    "TARGET_TRIAL_COMPILE_SUBMITTED",
    "TARGET_TRIAL_DATABASES",
    "TARGET_TRIAL_DATABASE_OUT_OF_SCOPE",
    "TARGET_TRIAL_SETUP_REFUSALS",
    "TARGET_TRIAL_STATE_TOOL",
    "TargetTrialCompileFailure",
    "TargetTrialSetupError",
    "TargetTrialStatement",
    "check_target_trial_statement",
    "latest_target_trial_compile",
    "project_target_trial_compile",
    "read_target_trial_statement",
    "submit_target_trial_compile",
    "submitted_tool_result",
    "target_trial_database_gap",
    "target_trial_family_declared",
    "target_trial_job_record",
    "target_trial_statement_schema",
]
