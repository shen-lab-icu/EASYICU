"""Path-free reading of one pipeline run's actions and executed steps.

Owner: the run-level facts behind the Copilot execution trace.  Three
read-only contracts, all addressed inside an already-resolved run directory:

* :func:`trace_action` names one Research Agent progress/audit event by a
  closed action code.  It reads only the host's fixed message envelopes and
  typed detail; free prose and diagnostic detail never cross.  An event the
  host did not write in a known envelope has no action.
* :func:`trace_timings_projection` is the audit-log timing projection the
  job result carries for a finished run: lifecycle coordinates plus the
  typed action facts above.
* :func:`project_step_detail` reads one executed plan step for the browser:
  the code that ran, how it ran, and the files it produced.  The caller
  supplies an opaque step id; the owner never accepts a client path.

Runner output (``run.log``) stays local.  The provider boundary sends an
external model only a host-built diagnostic envelope
(``research_agent.authority.diagnostic_envelope``): no owner redacts runner
output for disclosure, so the browser receives no log text either.  Output
files are listed by name; only registered, digest-verified PNG figures are
embedded, and tables open through the existing evidence-preview owner.
Row-level formats (parquet and the like) are never named or served.
"""

from __future__ import annotations

from easyicu.runtime.platform_compat import open_nofollow

import base64
import hashlib
import json
import math
import os
import re
import stat
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from easyicu.webserver import run_artifact_disclosure

SCHEMA_VERSION = "easyicu.web-run-step-detail/1"
MAX_CODE_BYTES = 64 * 1024
MAX_THUMBNAIL_BYTES = 420_000
MAX_THUMBNAILS_TOTAL_BYTES = 1_400_000
MAX_OUTPUT_FILES = 24
MAX_AUDIT_BYTES = 8_000_000
MAX_INDEX_BYTES = 4 * 1024 * 1024
MAX_SUMMARY_BYTES = 256 * 1024
MAX_HASH_BYTES = 16 * 1024 * 1024
#: An ICU database's stays fit well below this; a larger count is not one.
MAX_STAY_COUNT = 10_000_000

#: One path component: no separator, never ``.`` or ``..``.
STEP_ID_PATTERN = r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,159}"
_STEP_ID = re.compile(STEP_ID_PATTERN)
_RUN_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,159}")
_CODE = re.compile(r"[a-z][a-z0-9_]{0,79}")
_FILE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,159}")
_EVIDENCE_ID = re.compile(r"[A-Za-z0-9_.-]{1,160}")
_SHA256 = re.compile(r"[a-f0-9]{64}")
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: Keys a classified event may carry; every value is a closed code or a
#: bounded integer.  Transport owners forward exactly these keys.
TRACE_ACTION_KEYS = (
    "action",
    "executor",
    "run_kind",
    "repair_kind",
    "attempt",
    "group_variable_count",
    "plan_step_count",
    "stay_count",
    "variable_count",
)
#: Facts a finished run's projection attaches to particular actions.
TRACE_FACT_KEYS = (
    "query_count",
    "record_count",
    "queries",
    "source_label",
    "source_database",
    "group_variable_label",
    "outcome_label",
)

_S = rf"(?P<step>{STEP_ID_PATTERN})"
# The executor's fixed plan-step envelopes; their text is safe to forward.
_STEP_ENVELOPE = re.compile(
    rf"(?:Step \d+/\d+ (?:started|complete|failed contract checks|failed Critic review): "
    rf"|Skipped step already completed by pre-execution: "
    rf"|Skipped completed step from prior run: ){_S}\."
)
_STEP_EVENT = re.compile(
    rf"Step \d+/\d+ (?P<verb>started|complete|failed contract checks|failed Critic review): {_S}\."
)
_PRECOMPLETED = re.compile(rf"Skipped step already completed by pre-execution: {_S}\.")
_REUSED = re.compile(rf"Skipped completed step from prior run: {_S}\.")
_DEPENDENCY_SKIP = re.compile(rf"Skipped {_S}; required step {STEP_ID_PATTERN} failed\.")
_FAILED_CLOSED = re.compile(rf"Step {_S} failed closed: provider-call receipt invalid\.")
_GENERATING = re.compile(rf"Generating analysis script for {_S}\.")
_FALLBACK = re.compile(rf"Using deterministic fallback script for {_S}\.")
_DETERMINISTIC_RUNNER = re.compile(
    rf"Using deterministic (?P<name>absolute-risk context|robustness|"
    rf"missingness/measurement audit) runner for {_S}\."
)
_RUN = re.compile(
    rf"Running (?P<label>generated script|resumed script|fallback script|"
    rf"standard executor script) for {_S}\."
)
_REPAIR = re.compile(
    rf"Repairing (?P<kind>failed script|contract violation|"
    rf"post-mutation concept violation|figure layout) for {_S}\."
)
_REPAIR_RETRY = re.compile(
    rf"Repair attempt did not yield usable code for {_S} \(attempt (?P<attempt>\d{{1,3}})\): "
    r"[A-Za-z_][A-Za-z0-9_]{0,79}; retrying the repair without re-executing unchanged code\."
)
_DETERMINISTIC_REPAIR = re.compile(
    rf"Applied deterministic (?P<kind>summary|contract|runner) repair for {_S}: "
    r"[A-Za-z0-9_.-]{1,80}\."
)
_REPAIR_FAILED = re.compile(
    rf"(?:Repair failed|Contract repair failed|Concept-audit repair failed|Coder failed|"
    rf"Visual QA repair failed) for {_S}\."
)
_EXECUTOR_MESSAGE = re.compile(rf"(?P<message>[A-Z][^\n]{{0,238}}) for {_S}\.")
_PLANNING = re.compile(
    r"(?P<unit>Study structure|Cohort and analysis rules|Current executable plan step|Plan draft) "
    r"(?:validation attempt (?P<started>\d{1,3})/(?P<started_total>\d{1,3}) started"
    r"|attempt (?P<attempt>\d{1,3})/(?P<total>\d{1,3}) "
    r"(?P<verdict>did not satisfy its contract(?:; retrying)?|passed validation))\."
)
_EXPOSURE_GROUPS = re.compile(r"Exposure groups staged: (?P<names>[^\n]{0,600})\.")
_PLAN_READY = re.compile(r"Analysis plan ready with (?P<count>\d{1,4}) step\(s\)\.")
_FIXED_ACTIONS = {
    ("context", "Research context built."): "context_built",
    ("hypothesis", "Building pre-plan literature and hypothesis blueprint."): "literature_build",
    ("hypothesis", "Literature is deferred to a full article run."): "literature_deferred",
    ("cohort", "Cohort materialised to parquet."): "cohort_materialized",
    ("cohort", "Verified immutable staged cohort for resume."): "cohort_verified",
    ("cohort", "Trajectory staged with exact input authority."): "trajectory_staged",
    ("audit", "Initial cohort audit passed."): "cohort_audit",
    ("runtime", "Execution runtime validated before planning."): "runtime_validated",
    ("visual_qa", "Auditing generated figures."): "figure_audit",
}
_RUN_KINDS = {
    "generated script": "generated",
    "resumed script": "resumed",
    "fallback script": "fallback",
    "standard executor script": "standard",
}
_REPAIR_KINDS = {
    "failed script": "runtime",
    "contract violation": "contract",
    "post-mutation concept violation": "concept",
    "figure layout": "figure_layout",
}
_PLANNING_UNITS = {
    "Study structure": "structure",
    "Cohort and analysis rules": "rules",
    "Current executable plan step": "step",
    "Plan draft": "plan",
}
_ERROR_STATUSES = frozenset({"error", "failed", "blocked"})


class RunStepDetailError(ValueError):
    """Stable, owner-attributable refusal of a step-detail request."""

    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _bounded_int(value: Any, upper: int = 10_000) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if 0 <= number <= upper else None


@lru_cache(maxsize=1)
def _standard_executor_messages() -> Dict[str, str]:
    """Map each registered executor's fixed progress sentence to its key.

    The registry owns both; an exact lookup reads its own vocabulary rather
    than inferring a method from prose.  Computed sentences are skipped.
    """

    try:
        from easyicu.research_agent.execution.runners.selection import (
            STANDARD_EXECUTORS,
        )
    except Exception:  # noqa: BLE001 - a reading aid never blocks the trace
        return {}
    messages: Dict[str, str] = {}
    for executor in STANDARD_EXECUTORS.executors:
        message = executor.progress_message
        if isinstance(message, str) and message and _CODE.fullmatch(executor.key):
            messages[message] = executor.key
    return messages


def trace_action(
    phase: Any, message: Any, detail: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Return the closed action facts of one progress event, or ``{}``."""

    stage = str(phase or "").strip()
    text = str(message or "").strip()
    extra = detail if isinstance(detail, Mapping) else {}
    facts: Dict[str, Any] = {}
    match: Optional[re.Match[str]] = None

    def step_bound(action: str, found: re.Match[str], **more: Any) -> Dict[str, Any]:
        row = {"action": action, "step_id": found.group("step"), **more}
        declared = str(extra.get("step_id") or "")
        if declared and declared != row["step_id"]:
            return {}
        return row

    if stage == "coder":
        if match := _GENERATING.fullmatch(text):
            facts = step_bound("code_generation", match)
        elif match := _FALLBACK.fullmatch(text):
            facts = step_bound("fallback_code", match)
        elif match := _DETERMINISTIC_RUNNER.fullmatch(text):
            executor = re.sub(r"[^a-z]+", "_", match.group("name")).strip("_")
            facts = step_bound("standard_executor", match, executor=executor)
        elif match := _REPAIR.fullmatch(text):
            facts = step_bound("repair", match, repair_kind=_REPAIR_KINDS[match.group("kind")])
        elif match := _REPAIR_RETRY.fullmatch(text):
            facts = step_bound("repair_retry", match, attempt=int(match.group("attempt")))
        elif match := _REPAIR_FAILED.fullmatch(text):
            facts = step_bound("repair_failed", match)
        elif match := _EXECUTOR_MESSAGE.fullmatch(text):
            key = _standard_executor_messages().get(match.group("message"))
            if key:
                facts = step_bound("standard_executor", match, executor=key)
    elif stage == "runner":
        if match := _RUN.fullmatch(text):
            facts = step_bound("run", match, run_kind=_RUN_KINDS[match.group("label")])
    elif stage == "visual_qa":
        if match := _REPAIR.fullmatch(text):
            facts = step_bound("repair", match, repair_kind=_REPAIR_KINDS[match.group("kind")])
        elif match := _REPAIR_FAILED.fullmatch(text):
            facts = step_bound("repair_failed", match)
    elif stage == "runner_repair":
        if match := _DETERMINISTIC_REPAIR.fullmatch(text):
            facts = step_bound("deterministic_repair", match, repair_kind=match.group("kind"))
    elif stage == "step":
        if match := _DEPENDENCY_SKIP.fullmatch(text):
            facts = step_bound("step_skipped", match)
        elif match := _FAILED_CLOSED.fullmatch(text):
            facts = step_bound("step_error", match)
    elif stage == "planning":
        if match := _PLANNING.fullmatch(text):
            started = match.group("started") is not None
            verdict = str(match.group("verdict") or "")
            facts = {
                "action": "plan_validation",
                "planning_unit": _PLANNING_UNITS[match.group("unit")],
                "retry_phase": "started" if started
                else "accepted" if verdict == "passed validation" else "rejected",
                "current": int(match.group("started") or match.group("attempt")),
                "total": int(match.group("started_total") or match.group("total")),
            }
    elif stage == "context" and (match := _EXPOSURE_GROUPS.fullmatch(text)):
        names = [item for item in match.group("names").split(", ") if item.strip()]
        facts = {"action": "exposure_groups_staged", "group_variable_count": len(names)}
    elif stage == "plan" and (match := _PLAN_READY.fullmatch(text)):
        facts = {"action": "plan_ready", "plan_step_count": int(match.group("count"))}
    if not facts and (stage, text) in _FIXED_ACTIONS:
        facts = {"action": _FIXED_ACTIONS[(stage, text)]}
    if facts.get("action") == "context_built":
        # The research context's own counts: the ICU stays it holds and the
        # variables it built, bounded integers from the event's typed detail.
        for key, source in (("stay_count", "n_stays"), ("variable_count", "n_variables")):
            count = _bounded_int(extra.get(source), MAX_STAY_COUNT)
            if count is not None:
                facts[key] = count
    if not facts and stage in {"runner", "coder", "contract", "audit", "visual_qa"}:
        # A failure envelope names its step only through typed detail.
        status = str(extra.get("status") or "")
        step_id = str(extra.get("step_id") or "")
        if status in _ERROR_STATUSES and _STEP_ID.fullmatch(step_id):
            facts = {"action": "step_error", "step_id": step_id}
    if facts.get("action") in {"repair", "run"} and "attempt" not in facts:
        attempt = _bounded_int(
            extra.get("visual_repair_attempts")
            if facts.get("repair_kind") == "figure_layout"
            else extra.get("repair_attempts"),
            1_000,
        )
        if attempt:
            facts["attempt"] = attempt
    return {
        key: value
        for key, value in facts.items()
        if value is not None
        and (
            isinstance(value, int)
            or (key == "step_id" and _STEP_ID.fullmatch(str(value)))
            or _CODE.fullmatch(str(value))
        )
    }


def _open_regular(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = open_nofollow(path, flags)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise OSError("not a regular file")
    return descriptor


def _read_bounded(path: Path, limit: int) -> Optional[bytes]:
    """Read a regular, non-symlink file of at most ``limit`` bytes."""

    try:
        descriptor = _open_regular(path)
    except OSError:
        return None
    try:
        if os.fstat(descriptor).st_size > limit:
            return None
        chunks: List[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        return raw if len(raw) <= limit else None
    finally:
        os.close(descriptor)


def _read_prefix_and_digest(path: Path, cap: int) -> Optional[Tuple[bytes, int, str]]:
    """Return the first ``cap`` bytes, the size and the whole-file digest."""

    try:
        descriptor = _open_regular(path)
    except OSError:
        return None
    try:
        size = os.fstat(descriptor).st_size
        if size > MAX_HASH_BYTES:
            return None
        digest = hashlib.sha256()
        prefix = bytearray()
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if len(prefix) < cap:
                prefix.extend(chunk[: cap - len(prefix)])
        return bytes(prefix), size, digest.hexdigest()
    finally:
        os.close(descriptor)


def _child(root: Path, *parts: str) -> Optional[Path]:
    """Resolve fixed children of ``root`` without following any symlink."""

    current = root
    for part in parts:
        current = current / part
        try:
            if stat.S_ISLNK(os.lstat(current).st_mode):
                return None
        except OSError:
            return None
    return current


def _read_json(path: Optional[Path], limit: int) -> Any:
    if path is None:
        return None
    raw = _read_bounded(path, limit)
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _literature_facts(run_dir: Path) -> Dict[str, Any]:
    bundle = _read_json(_child(run_dir, "preplan_literature_bundle.json"), MAX_INDEX_BYTES)
    if not isinstance(bundle, Mapping):
        return {}
    provenance = bundle.get("search_provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    searches = provenance.get("search_queries")
    searches = searches if isinstance(searches, Mapping) else {}
    queries = [
        " ".join(str(query).split())[:200]
        for values in searches.values()
        if isinstance(values, list)
        for query in values
        if str(query or "").strip()
    ]
    citations = bundle.get("citations")
    facts: Dict[str, Any] = {"query_count": len(queries)}
    if isinstance(citations, list):
        facts["record_count"] = len(citations)
    shown = [
        query
        for query in queries[:4]
        if run_artifact_disclosure.scan_browser_projection({"query": query})["passed"]
    ]
    if shown:
        facts["queries"] = shown
    return facts


def _source_facts(source: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(source, Mapping):
        return {}
    facts = {
        "source_label": " ".join(str(source.get("label") or "").split())[:160],
        "source_database": str(source.get("database") or "").strip()[:64],
    }
    return {
        key: value
        for key, value in facts.items()
        if value and run_artifact_disclosure.scan_browser_projection({key: value})["passed"]
    }


def _reader_label_pair(labels: Mapping[str, Any]) -> Dict[str, str]:
    shown: Dict[str, str] = {}
    for language in ("zh", "en"):
        value = labels.get(language)
        if not isinstance(value, str):
            continue
        text = " ".join(value.split())[:240]
        if text and run_artifact_disclosure.scan_browser_projection({"label": text})["passed"]:
            shown[language] = text
    return shown


def _run_reader_facts(
    run_dir: Path, *, plan: Optional[Mapping[str, Any]] = None,
    step_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Reader names from the grouping owner and recorded primary outcome.

    The current plan's digest-bound authority supplies its display label;
    the context identifies the primary endpoint. No column name is a label.
    """
    from easyicu.research_agent.orchestration.exposure_group_labels import (
        recorded_exposure_group_labels,
    )
    from easyicu.webserver.literature_projection import load_current_plan_authority
    from easyicu.webserver.pi_copilot.exposure_group_notes import project_exposure_groups

    if plan is None:
        plan = load_current_plan_authority(run_dir)
    context = _read_json(_child(run_dir, "research_context.json"), MAX_INDEX_BYTES)
    context = context if isinstance(context, Mapping) else {}
    projected = project_exposure_groups(recorded_exposure_group_labels(run_dir, plan=plan))
    groupings = (projected or {}).get("groupings") or []
    variable = context.get("primary_exposure")
    if step_id:
        steps = plan.get("steps")
        step = next((item for item in steps or [] if isinstance(item, Mapping)
                     and item.get("step_id") == step_id), {})
        inputs = step.get("inputs")
        candidates = [row for row in groupings if row["variable"] in (
            inputs if isinstance(inputs, list) else []
        )]
        if len(candidates) == 1:
            variable = candidates[0]["variable"]
    elif variable is None and len(groupings) == 1:
        variable = groupings[0]["variable"]
    grouping = next((row for row in groupings if row["variable"] == variable), {})
    facts: Dict[str, Any] = {}
    group_label = _reader_label_pair({
        "zh": f"{grouping['concept_label_zh']}分组" if grouping.get("concept_label_zh") else "",
        "en": f"{grouping['concept_label_en']} groups" if grouping.get("concept_label_en") else "",
    })
    if group_label:
        facts["group_variable_label"] = group_label
    labels = plan.get("display_labels")
    outcome = context.get("target_outcome")
    label = labels.get(outcome) if isinstance(labels, Mapping) and isinstance(outcome, str) else None
    if isinstance(label, str) and label.strip():
        # One recorded display label serves both languages unless the context
        # also records an English description for this same endpoint.
        english = label
        if re.search(r"[㐀-鿿]", label):
            variables = context.get("variables")
            for item in variables if isinstance(variables, list) else []:
                if isinstance(item, Mapping) and item.get("name") == outcome:
                    description = item.get("description")
                    if isinstance(description, str) and description.strip() and not re.search(r"[㐀-鿿]", description):
                        english = description
        outcome_label = _reader_label_pair({"zh": label, "en": english})
        if outcome_label:
            facts["outcome_label"] = outcome_label
    return facts


def trace_timings_projection(
    run_dir: Optional[Path], *, source: Optional[Mapping[str, Any]] = None,
    plan: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Read only audit lifecycle coordinates and typed action facts.

    Audit events are free prose.  Only the executor's fixed step envelopes
    are forwarded as text; every other event is its phase, status and the
    closed facts of :func:`trace_action`, never its message or detail.
    """

    if run_dir is None:
        return []
    path = _child(Path(run_dir), "audit_log.jsonl")
    if path is None:
        return []
    raw = _read_bounded(path, MAX_AUDIT_BYTES)
    if raw is None:
        return []
    rows: List[Dict[str, Any]] = []
    literature: Optional[Dict[str, Any]] = None
    reader_facts = _run_reader_facts(Path(run_dir), plan=plan)
    step_reader_facts: Dict[str, Dict[str, Any]] = {}
    for line in raw.decode("utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                continue
            timestamp = datetime.fromisoformat(str(event.get("timestamp", "")))
            if timestamp.tzinfo is None:
                continue
        except (ValueError, TypeError):
            continue
        phase = str(event.get("phase") or "")
        if not _CODE.fullmatch(phase):
            continue
        status = str(event.get("status") or "running")
        if not _CODE.fullmatch(status):
            status = "running"
        message = str(event.get("event") or "")
        row: Dict[str, Any] = {
            "timestamp": timestamp.isoformat(),
            "phase": phase,
            "event": message if _STEP_ENVELOPE.fullmatch(message) else "",
            "status": status,
        }
        step_id = str(event.get("step_id") or "")
        if _STEP_ID.fullmatch(step_id):
            row["step_id"] = step_id
        detail = event.get("detail")
        detail = dict(detail) if isinstance(detail, Mapping) else {}
        detail.update({"status": status, "step_id": row.get("step_id", "")})
        facts = trace_action(phase, message, detail)
        if facts.get("step_id") and "step_id" not in row:
            row["step_id"] = facts["step_id"]
        row.update({key: facts[key] for key in facts if key != "step_id"})
        if clean_step := row.get("step_id"):
            if clean_step not in step_reader_facts:
                step_reader_facts[clean_step] = _run_reader_facts(
                    Path(run_dir), plan=plan, step_id=clean_step,
                )
            row.update(step_reader_facts[clean_step])
        elif facts.get("action") == "exposure_groups_staged":
            row.update(reader_facts)
        if facts.get("action") == "literature_build":
            if literature is None:
                literature = _literature_facts(Path(run_dir))
            row.update(literature)
        elif facts.get("action") == "context_built":
            row.update(_source_facts(source))
        rows.append(row)
    return rows


def pipeline_run_dir(project_dir: str, run_id: str) -> Path:
    """Return the concrete pipeline run a wrapper declares for ``run_id``.

    Wrapper-level artefacts are copied up for result browsing while the
    evidence registry, audit log and step directories stay in the concrete
    run.  An undeclared, escaping or symlinked child keeps the wrapper.
    """

    wrapper = Path(project_dir).expanduser().resolve()
    clean_run = str(run_id or "").strip()
    if not _RUN_ID.fullmatch(clean_run):
        return wrapper
    pipeline_root = (wrapper / "pipeline").resolve()
    declared = pipeline_root / clean_run
    if (
        not declared.is_symlink()
        and declared.is_dir()
        and declared.resolve().parent == pipeline_root
    ):
        return declared.resolve()
    return wrapper


def _evidence_records(run_dir: Path) -> List[Mapping[str, Any]]:
    records = _read_json(_child(run_dir, "evidence", "evidence_index.json"), MAX_INDEX_BYTES)
    return [row for row in records if isinstance(row, Mapping)] if isinstance(records, list) else []


def _record_file(run_dir: Path, record: Mapping[str, Any]) -> Optional[Path]:
    parts = str(record.get("relative_path") or "").replace("\\", "/").split("/")
    if (
        len(parts) != 2
        or parts[0] != "evidence"
        or not _FILE_NAME.fullmatch(parts[1].split("__", 1)[-1])
        or ".." in parts[1]
    ):
        return None
    return _child(run_dir, *parts)


def _record_name(record: Mapping[str, Any]) -> str:
    name = str(record.get("relative_path") or "").replace("\\", "/").rsplit("/", 1)[-1]
    prefix = f"{record.get('evidence_id') or ''}__"
    return name[len(prefix):] if name.startswith(prefix) else name


def _registered(record: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    evidence_id = str(record.get("evidence_id") or "")
    digest = str(record.get("sha256") or "").strip().lower()
    if _EVIDENCE_ID.fullmatch(evidence_id) and _SHA256.fullmatch(digest):
        return evidence_id, digest
    return None


def _step_state(rows: Iterable[Mapping[str, Any]]) -> str:
    state = ""
    for row in rows:
        event = str(row.get("event") or "")
        action = str(row.get("action") or "")
        if match := _STEP_EVENT.fullmatch(event):
            verb = match.group("verb")
            state = "running" if verb == "started" else "complete" if verb == "complete" else "failed"
        elif _PRECOMPLETED.fullmatch(event):
            state = "precompleted"
        elif _REUSED.fullmatch(event):
            state = "complete"
        elif action == "step_skipped":
            state = "skipped"
        elif action in {"step_error", "repair_failed"}:
            state = "failed"
        elif not state:
            state = "running"
    return state


_STANDARD_MODES = frozenset(
    {"deterministic_standard", "deterministic_skill", "deterministic_probe", "system"}
)


def _execution(
    rows: List[Mapping[str, Any]],
    code_record: Optional[Mapping[str, Any]],
    summary: Mapping[str, Any],
    state: str,
) -> Dict[str, str]:
    mode = ""
    executor = ""
    for row in rows:
        action = row.get("action")
        if action == "standard_executor":
            mode, executor = "standard", str(row.get("executor") or executor)
        elif action == "code_generation" and mode != "standard":
            mode = "generated"
        elif action == "fallback_code":
            mode = "fallback"
        elif action == "run" and not mode:
            mode = {"standard": "standard", "fallback": "fallback"}.get(
                str(row.get("run_kind") or ""), "generated"
            )
    generation = str((code_record or {}).get("generation_mode") or "")
    if generation:
        mode = (
            "standard" if generation in _STANDARD_MODES
            else "fallback" if generation == "fallback"
            else "generated"
        )
    declared = str(summary.get("deterministic_standard_analysis") or "")
    if not executor and mode in {"standard", ""} and _CODE.fullmatch(declared):
        mode, executor = "standard", declared
    if not mode and state == "precompleted":
        mode = "host"
    return {"mode": mode or "unknown", "executor": executor}


# Absolute host paths, written by a tool or a person, become a fixed token.
_HOST_PATH_TOKEN = re.compile(
    r"(?:file://)?(?:(?<![A-Za-z0-9])/(?:Users|home|private|tmp|var|etc|opt|Volumes)/"
    r"|\b[A-Za-z]:\\)[^\s'\"`,;)\]}]*",
    re.I,
)


def _code_projection(
    run_dir: Path, step_dir: Optional[Path], record: Optional[Mapping[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], str]:
    from easyicu.research_agent.authority.secret_redaction import redact_text_secrets

    path: Optional[Path] = None
    expected = ""
    name = "analysis.py"
    if record is not None and (registered := _registered(record)):
        path = _record_file(run_dir, record)
        expected = registered[1]
        name = _record_name(record) or name
    if path is None and step_dir is not None:
        path = _child(step_dir, "analysis.py")
        expected = ""
        name = "analysis.py"
    read = _read_prefix_and_digest(path, MAX_CODE_BYTES) if path is not None else None
    if read is None:
        return None, "not_written"
    prefix, size, digest = read
    if expected and digest != expected:
        return None, "digest_mismatch"
    truncated = size > len(prefix)
    if truncated and b"\n" in prefix:
        prefix = prefix[: prefix.rfind(b"\n") + 1]
    try:
        text = prefix.decode("utf-8", errors="ignore" if truncated else "strict")
    except UnicodeDecodeError:
        return None, "encoding_unsupported"
    text, redacted = _HOST_PATH_TOKEN.subn("<host path>", text)
    text = redact_text_secrets(text)
    if _HOST_PATH_TOKEN.search(text):
        return None, "host_path_detected"
    language = {".py": "python", ".r": "r", ".sql": "sql", ".sh": "shell"}.get(
        Path(name).suffix.lower(), "text"
    )
    return {
        "display_name": name if _FILE_NAME.fullmatch(name) else "analysis.py",
        "language": language,
        "text": text,
        "line_count": len(text.splitlines()),
        "bytes": size,
        "truncated": truncated,
        "redacted_paths": redacted,
    }, "available"


_OUTPUT_KINDS = {
    ".png": "figure",
    ".svg": "figure",
    ".pdf": "figure",
    ".csv": "table",
    ".json": "record",
    ".txt": "document",
    ".md": "document",
    ".html": "document",
    ".tex": "document",
}
#: Registered kinds the existing evidence-preview owner can open.
_PREVIEW_RECORD_KINDS = {
    "table": frozenset({"table"}),
    "record": frozenset({"statistic", "log", "table"}),
}


def _outputs(
    run_dir: Path,
    step_dir: Optional[Path],
    step_id: str,
    records: List[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    outputs_dir = _child(step_dir, "outputs") if step_dir is not None else None
    if outputs_dir is None or not outputs_dir.is_dir():
        return [], 0, 0
    by_name: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        if str(record.get("produced_by_step") or "") == step_id:
            by_name.setdefault(_record_name(record), []).append(record)
    try:
        # Apply clinical priority before the cap, so runner records cannot
        # displace the summary or result tables from the bounded list.
        entries = sorted(os.scandir(outputs_dir), key=lambda entry: (
            entry.name != "step_summary.json",
            entry.name.startswith("runner_") or ".lock." in entry.name,
            entry.name,
        ))
    except OSError:
        return [], 0, 0
    listed: List[Dict[str, Any]] = []
    withheld = 0
    more = 0
    thumbnail_budget = MAX_THUMBNAILS_TOTAL_BYTES
    for entry in entries:
        try:
            regular = entry.is_file(follow_symlinks=False)
            size = entry.stat(follow_symlinks=False).st_size
        except OSError:
            continue
        if not regular:
            continue
        name = entry.name
        kind = _OUTPUT_KINDS.get(Path(name).suffix.lower())
        if kind is None or not _FILE_NAME.fullmatch(name) or name == "table_one_private_checkpoint.json":
            # Row-level and unknown formats are never named or served.
            withheld += 1
            continue
        if len(listed) >= MAX_OUTPUT_FILES:
            more += 1
            continue
        if name == "step_summary.json":
            kind = "summary"
        row: Dict[str, Any] = {"name": name, "kind": kind, "bytes": size}
        matches = by_name.get(name, [])
        record = matches[0] if len(matches) == 1 else None
        registered = _registered(record) if record is not None else None
        current = (
            _read_prefix_and_digest(Path(entry.path), 0) if registered is not None else None
        )
        verified = bool(registered and current and current[2] == registered[1])
        record_kind = str((record or {}).get("kind") or "").lower()
        preview_kinds = _PREVIEW_RECORD_KINDS.get("record" if kind == "summary" else kind)
        if verified and preview_kinds and record_kind in preview_kinds:
            row["evidence"] = {"id": registered[0], "sha256": registered[1]}
        if (
            verified
            and kind == "figure"
            and name.lower().endswith(".png")
            and record_kind == "figure"
            and size <= min(MAX_THUMBNAIL_BYTES, thumbnail_budget)
        ):
            raw = _read_bounded(Path(entry.path), MAX_THUMBNAIL_BYTES)
            if raw is not None and raw.startswith(_PNG_MAGIC) and (
                hashlib.sha256(raw).hexdigest() == registered[1]
            ):
                row["thumbnail"] = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
                thumbnail_budget -= len(raw)
        listed.append(row)
    return listed, withheld, more


def _headline(summary: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    value = summary.get("headline")
    if not isinstance(value, Mapping):
        return None
    headline = {
        language: " ".join(str(value.get(language) or "").split())[:240]
        for language in ("en", "zh")
    }
    headline = {key: text for key, text in headline.items() if text}
    if not headline or not run_artifact_disclosure.scan_browser_projection(
        {"headline": headline}
    )["passed"]:
        return None
    return headline


def _planned_steps(run_dir: Path) -> List[Mapping[str, Any]]:
    plan = _read_json(_child(run_dir, "analysis_plan.json"), MAX_INDEX_BYTES)
    steps = plan.get("steps") if isinstance(plan, Mapping) else None
    return [step for step in steps or [] if isinstance(step, Mapping)]


def _planned_step_ids(run_dir: Path) -> set[str]:
    return {str(step.get("step_id") or "") for step in _planned_steps(run_dir)}


#: Plan inputs with a typed prefix (``artifact:``, ``table:`` ...) name a
#: product or evidence, not a variable.
_NON_VARIABLE_INPUT = re.compile(r"^[a-z_]+:")


def _step_inputs(
    run_dir: Path,
    step_id: str,
    run_rows: List[Mapping[str, Any]],
    source: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """What one step read, from receipts: source, stays, variables, plan place.

    The source is the run context's label; the stay count is the research
    context's own (``context_built``); the variable count and the step's place
    are read from the executable plan.  Each fact is absent when no receipt
    records it -- nothing is inferred.
    """

    facts: Dict[str, Any] = {
        **_source_facts(source), **_run_reader_facts(run_dir, step_id=step_id),
    }
    for row in run_rows:
        if row.get("action") == "context_built" and isinstance(row.get("stay_count"), int):
            facts["stay_count"] = row["stay_count"]
    steps = _planned_steps(run_dir)
    ids = [str(step.get("step_id") or "") for step in steps]
    if step_id in ids:
        planned = steps[ids.index(step_id)]
        facts["step_index"] = ids.index(step_id) + 1
        facts["step_total"] = len(ids)
        inputs = planned.get("inputs")
        variables = {
            item.strip() for item in (inputs if isinstance(inputs, list) else [])
            if isinstance(item, str) and item.strip()
            and not _NON_VARIABLE_INPUT.match(item.strip())
        }
        # A step whose plan names no variable says nothing about what it read.
        if variables:
            facts["variable_count"] = len(variables)
    return facts


def _step_timing(rows: Iterable[Mapping[str, Any]]) -> Dict[str, str]:
    """The step's own start and end, from its fixed audit envelopes only."""

    started = ""
    ended = ""
    for row in rows:
        event = str(row.get("event") or "")
        stamp = str(row.get("timestamp") or "")
        if not stamp:
            continue
        if match := _STEP_EVENT.fullmatch(event):
            if match.group("verb") == "started":
                started = started or stamp
            else:
                ended = stamp
        elif _PRECOMPLETED.fullmatch(event) or _REUSED.fullmatch(event):
            ended = stamp
    timing = {"started_at": started, "ended_at": ended}
    return {key: value for key, value in timing.items() if value}


def wrapper_source(project_dir: str) -> Optional[Dict[str, Any]]:
    """The data source a run's wrapper context names, for the step's Input."""

    if not str(project_dir or "").strip():
        return None
    try:
        wrapper = Path(str(project_dir)).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    context = _read_json(_child(wrapper, "run_context.json"), MAX_SUMMARY_BYTES)
    source = context.get("source") if isinstance(context, Mapping) else None
    return dict(source) if isinstance(source, Mapping) else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _receipt_count(value: Any) -> Optional[int]:
    return value if type(value) is int and 0 <= value <= MAX_STAY_COUNT else None


def _receipt_text(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip() or len(value) > 240:
        return None
    return value if run_artifact_disclosure.scan_browser_projection({"text": value})["passed"] else None


def _receipt_name(value: Any) -> Optional[str]:
    text = _receipt_text(value)
    return text if text and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,159}", text) else None


def _verified_receipt(
    root: Path, records: List[Mapping[str, Any]], name: str, step: Optional[str] = None,
) -> Mapping[str, Any]:
    """Only a unique, current registered receipt, never an arbitrary JSON file."""
    matches = [r for r in records if _record_name(r) == name
               and r.get("produced_by_step") == step
               and r.get("kind") == ("statistic" if step else "log")]
    if len(matches) != 1 or not (identity := _registered(matches[0])):
        return {}
    path = _record_file(root, matches[0])
    raw = _read_bounded(path, MAX_SUMMARY_BYTES) if path else None
    original = _child(root, "steps", step, "outputs", name) if step else _child(root, name)
    current = _read_bounded(original, MAX_SUMMARY_BYTES) if original else None
    if raw is None or raw != current or hashlib.sha256(raw).hexdigest() != identity[1]:
        return {}
    try:
        value = _mapping(json.loads(raw))
    except (ValueError, UnicodeDecodeError):
        return {}
    if value.get("run_id", root.name) != root.name or (step and value.get("step_id", step) != step):
        return {}
    return value


def _receipt_reference(root: Path, reference: Any, prefix: str) -> Mapping[str, Any]:
    """Allow only the two immutable metadata families in this run's root.

    Do not use the materialized-cohort loader: that owner opens row data.
    Neither a client path nor any private checkpoint is an admitted reference.
    """
    ref = _mapping(reference)
    digest = str(ref.get("sha256") or "")
    if not _SHA256.fullmatch(digest) or ref.get("file") != f"{prefix}.sha256-{digest}.json":
        return {}
    path = _child(root, ref["file"])
    raw = _read_bounded(path, MAX_SUMMARY_BYTES) if path else None
    if raw is None or type(ref.get("size")) is not int or len(raw) != ref["size"]:
        return {}
    if hashlib.sha256(raw).hexdigest() != digest:
        return {}
    try:
        return _mapping(json.loads(raw))
    except (ValueError, UnicodeDecodeError):
        return {}


def _receipt_concepts(sidecar: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    concepts: List[Dict[str, Any]] = []
    omitted = 0
    for file in _list(sidecar.get("files")):
        for column, binding in _mapping(_mapping(file).get("columns")).items():
            entry = _mapping(binding)
            metadata = _mapping(entry.get("metadata"))
            if not _receipt_name(column) or not metadata:
                continue
            window = _mapping(entry.get("derivation_window"))
            bounds = [window.get(key) for key in ("start_hours", "end_hours")]
            timed = all(type(n) in {float, int} and math.isfinite(n) for n in bounds)
            for source in _list(metadata.get("source_lineage")) or [{}]:
                if len(concepts) >= 32:
                    omitted += 1
                    continue
                lineage = _mapping(source)
                concepts.append({
                    "column": column, "concept": _receipt_name(metadata.get("source_concept")),
                    "table": _receipt_name(lineage.get("table")),
                    "selector": _receipt_name(lineage.get("selector_variable")),
                    "items": [text for item in _list(lineage.get("item_ids"))[:16]
                              if (text := _receipt_text(item))],
                    "selector_regex": _receipt_text(lineage.get("selector_regex")),
                    "value_column": _receipt_name(lineage.get("value_variable")),
                    "time_column": _receipt_name(lineage.get("time_variable")),
                    "unit_column": _receipt_name(lineage.get("unit_variable")),
                    "unit": _receipt_text(metadata.get("canonical_unit")),
                    "aggregation": _receipt_name(metadata.get("aggregation")),
                    "transform": _receipt_text(entry.get("representation_transform")),
                    "callback": _receipt_text(lineage.get("callback")),
                    "window": {"origin": _receipt_name(window.get("origin")),
                               "start_hours": bounds[0], "end_hours": bounds[1]} if timed else None,
                })
    return concepts, omitted


def _receipt_step_counts(
    root: Path, records: List[Mapping[str, Any]], run_rows: List[Mapping[str, Any]],
    authority: Mapping[str, Any], analysis_unit: Optional[str],
) -> List[Dict[str, Any]]:
    counts = []
    for index, planned in enumerate(_planned_steps(root)[:24], 1):
        step = str(planned.get("step_id") or "")
        if not _STEP_ID.fullmatch(step):
            continue
        state = _step_state(r for r in run_rows if r.get("step_id") == step)
        if state in {"running", "not_started", "skipped"}:
            continue
        summary = _verified_receipt(root, records, "step_summary.json", step)
        if not summary:
            continue
        binding_key = summary.get("cohort_input_key") or summary.get("typed_cohort_input")
        inputs = [_mapping(row) for row in _list(summary.get("input_bindings"))
                  if _mapping(row).get("loaded") is True
                  and (not binding_key or _mapping(row).get("input_key") == binding_key)]
        before = _receipt_count(summary.get("n_source"))
        if "n_source" not in summary and len(inputs) == 1:
            before = _receipt_count(inputs[0].get("row_count"))
        model = "n_total" in summary and bool(summary.get("estimator_kind") or summary.get("model_contracts"))
        after_key = next((key for key in ("n_total", "n_analysis_cohort", "analysis_cohort_n", "cohort_n")
                          if key in summary), "")
        unit = summary.get("analysis_unit")
        landmark = _mapping(summary.get("landmark_runtime_receipt"))
        if (landmark.get("schema_version") == "easyicu.landmark_categorical_cohort_runtime_receipt/1"
                and landmark.get("source_sha256") == authority.get("cohort_sha256")):
            unit = analysis_unit
        unit = unit if isinstance(unit, str) and unit in {"icu_stay", "patient"} else "observations"
        counts.append({"step_id": step, "index": index, "before": before,
                       "before_unit": unit if "n_source" in summary else "rows",
                       "after": _receipt_count(summary.get(after_key)), "unit": unit,
                       "events": _receipt_count(summary.get("n_events")), "model": model})
    return counts


def _extraction_record(
    root: Path, records: List[Mapping[str, Any]], run_rows: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Bounded, allowlisted facts from the registered run input and step receipts.

    The capsule's current authority (not its earlier scientific-input snapshot)
    binds the sidecar. Null coordinates remain null; no dictionary is consulted.
    Patient grouping counts describe the source universe, never a model cohort.
    """
    capsule = _verified_receipt(root, records, "run_input_capsule.json")
    authority = _receipt_reference(root, capsule.get("materialized_cohort_authority_ref"), "cohort_authority")
    if (capsule.get("schema_version") != "easyicu.run_input_capsule/2"
            or authority.get("schema_version") != "easyicu.materialized_cohort_authority/1"
            or not capsule.get("cohort_sha256")
            or authority.get("cohort_sha256") != capsule["cohort_sha256"]):
        authority = {}
    sidecar = _receipt_reference(root, authority.get("column_metadata"), "cohort_column_metadata")
    if sidecar.get("schema_version") != "easyicu.column_metadata_sidecar/1":
        sidecar = {}
    provenance = _mapping(authority.get("semantic_provenance"))
    followup = _mapping(_mapping(provenance.get("hospital_followup_materialization")).get("followup"))
    unit = followup.get("analysis_unit")
    unit = unit if isinstance(unit, str) and unit in {"icu_stay", "patient"} else None
    replacement = _mapping(provenance.get("replacement_row_identity"))
    grouping = _mapping(replacement.get("authority_coordinates"))
    identifiers = []
    for name, role in ((authority.get("identity_column"), "analysis_key"),
                       (replacement.get("mapping_stay_column"), "stay_key"),
                       (replacement.get("mapping_patient_column"), "patient_group")):
        if clean := _receipt_name(name):
            identifiers.append({"name": clean, "role": role})
    table = _receipt_name(grouping.get("identity_table"))
    column = _receipt_name(grouping.get("patient_identifier_column"))
    if table and column:
        identifiers.append({"name": f"{table}.{column}", "role": "patient_source"})
    concepts, omitted = _receipt_concepts(sidecar)
    steps = _receipt_step_counts(root, records, run_rows, authority, unit)
    omitted_steps = max(0, len(_planned_steps(root)) - 24)
    extracted = _receipt_count(provenance.get("n_stays_extracted"))
    retained = _receipt_count(provenance.get("n_stays_after_inclusion_exclusion"))
    complete = (
        bool(concepts and unit and identifiers and steps and not omitted and not omitted_steps)
        and all(row.get(key) is not None for row in concepts
                for key in ("concept", "table", "value_column", "time_column", "unit_column", "unit", "window"))
        and all(row["before"] is not None and row["after"] is not None for row in steps)
        and extracted is not None and retained is not None
    )
    return {
        "status": "recorded" if complete else "partial" if authority or steps else "unrecorded",
        "analysis_unit": unit, "identifiers": identifiers, "concepts": concepts,
        "omitted_concepts": omitted, "omitted_steps": omitted_steps, "steps": steps,
        "extracted": extracted, "retained": retained,
        "source_patient_count": _receipt_count(grouping.get("patients")),
        "source_stay_count": _receipt_count(grouping.get("stays")),
    }


def project_step_detail(
    run_dir: Path, step_id: str, *, source: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Return one executed step's code, execution and outputs, path-free.

    ``source`` is the run context's data source (label and database); only
    its browser-safe label and database code reach the step's inputs.
    """

    clean = str(step_id or "").strip()
    if not _STEP_ID.fullmatch(clean):
        raise RunStepDetailError(
            "run_step_id_invalid", "The step id is not a valid step coordinate.", 400
        )
    root = Path(run_dir).resolve()
    step_dir = _child(root, "steps", clean)
    if step_dir is not None and not step_dir.is_dir():
        step_dir = None
    run_rows = trace_timings_projection(root)
    rows = [row for row in run_rows if row.get("step_id") == clean]
    if step_dir is None and not rows and clean not in _planned_step_ids(root):
        raise RunStepDetailError(
            "run_step_not_found", "The step does not belong to this run.", 404
        )
    records = _evidence_records(root)
    code_records = [
        record for record in records
        if str(record.get("kind") or "").lower() == "code"
        and str(record.get("produced_by_step") or "") == clean
    ]
    # Several registered scripts mean a repaired step: the last one ran.
    code_record = code_records[-1] if code_records else None
    state = _step_state(rows)
    summary_path = _child(step_dir, "outputs", "step_summary.json") if step_dir else None
    summary = _read_json(summary_path, MAX_SUMMARY_BYTES)
    summary = summary if isinstance(summary, Mapping) else {}
    if not state:
        state = "complete" if summary else "not_started" if step_dir is None else "unknown"
    code, code_status = _code_projection(root, step_dir, code_record)
    finished = state in {"complete", "failed", "precompleted"}
    outputs, withheld, more = (
        _outputs(root, step_dir, clean, records) if finished else ([], 0, 0)
    )
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step_id": clean,
        "state": state,
        "execution": _execution(rows, code_record, summary, state),
        "code": code,
        "code_status": code_status,
        # No owner redacts runner output for disclosure; it stays local.
        "log": {"shown": False, "reason": "runner_output_stays_local"},
        "outputs": outputs,
        "withheld_output_count": withheld,
        "more_output_count": more,
        "inputs": _step_inputs(root, clean, run_rows, source),
        "extraction_record": _extraction_record(root, records, run_rows),
        "timing": _step_timing(rows),
    }
    headline = _headline(summary) if finished else None
    if headline:
        payload["headline"] = headline
    scanned = {
        **payload,
        "code": {key: value for key, value in (code or {}).items() if key != "text"},
        "outputs": [
            {key: value for key, value in row.items() if key != "thumbnail"}
            for row in outputs
        ],
    }
    if not (
        run_artifact_disclosure.scan_browser_projection({"step": scanned})["passed"]
        and run_artifact_disclosure.scan_artifact_payloads({"step": scanned})["passed"]
    ):
        raise RunStepDetailError(
            "run_step_privacy_blocked",
            "The step detail was withheld by the EasyICU privacy scan.",
            409,
        )
    return payload


__all__ = [
    "MAX_CODE_BYTES",
    "RunStepDetailError",
    "SCHEMA_VERSION",
    "STEP_ID_PATTERN",
    "TRACE_ACTION_KEYS",
    "TRACE_FACT_KEYS",
    "pipeline_run_dir",
    "project_step_detail",
    "trace_action",
    "trace_timings_projection",
    "wrapper_source",
]
