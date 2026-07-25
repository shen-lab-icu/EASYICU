"""Authorization and disclosure policy for the MCP tool surface.

The research agent's own prompts never carry raw records: everything an
external model sees is built by ``outbound_safe_context_payload()``, a
deny-by-default projection that keeps variable definitions, cohort sizes,
units, time windows and aggregate statistics and drops the rest.

MCP is a second front door into the same data, and it did not inherit that
boundary: ``research_agent.load_concepts`` returned ``frame.head(n).to_dict()``
— patient ids, timestamps and concept values — to whatever client called it,
with no cap on ``n``, and let the caller name any ``data_path``, ``workdir``
and ``output_path`` on the host filesystem.

This module supplies the three controls that were missing:

* **scopes** — tools declare what authority they need, and patient-level
  disclosure is a scope of its own that is off unless the operator grants it;
* **path confinement** — every filesystem argument must resolve inside a root
  configured at startup;
* **disclosure projection** — the default frame summary is shape, dtypes,
  missingness and (above a small-cell floor) aggregate statistics, with
  identifier and timestamp columns excluded.

Everything is configured by environment variable so a stdio server mounted by
a desktop client can be locked down without a config file.
"""

from __future__ import annotations

import hashlib
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

#: Authority to read study/concept metadata, skills and manifests.
SCOPE_METADATA = "metadata"
#: Authority to start a full pipeline run.
SCOPE_RUN_PIPELINE = "run_pipeline"
#: Authority to receive patient-level rows in a response.
SCOPE_READ_PATIENT_DATA = "read_patient_data"
#: Authority to write extraction output to the host filesystem.
SCOPE_WRITE_ARTIFACTS = "write_artifacts"
#: Authority to register artefacts in the EvidenceStore.
SCOPE_BIND_EVIDENCE = "bind_evidence"
#: Authority to receive the *internal* ResearchContext / manifest shape rather
#: than the deny-by-default outbound projection every other agent path uses.
SCOPE_READ_INTERNAL_CONTEXT = "read_internal_context"

ALL_SCOPES: FrozenSet[str] = frozenset(
    {
        SCOPE_METADATA,
        SCOPE_RUN_PIPELINE,
        SCOPE_READ_PATIENT_DATA,
        SCOPE_WRITE_ARTIFACTS,
        SCOPE_BIND_EVIDENCE,
        SCOPE_READ_INTERNAL_CONTEXT,
    }
)

#: Everything except the two disclosure scopes. Extraction still runs, writes
#: its parquet and registers evidence; what changes is that the *response*
#: carries shape and aggregate statistics instead of rows, and study metadata
#: arrives as the same outbound-safe projection the Planner prompt gets.
DEFAULT_SCOPES: FrozenSet[str] = ALL_SCOPES - {
    SCOPE_READ_PATIENT_DATA,
    SCOPE_READ_INTERNAL_CONTEXT,
}

MCP_SCOPES_ENV = "EASYICU_MCP_SCOPES"
MCP_ALLOW_PATIENT_DATA_ENV = "EASYICU_MCP_ALLOW_PATIENT_DATA"
MCP_ALLOWED_ROOTS_ENV = "EASYICU_MCP_ALLOWED_ROOTS"
MCP_ALLOW_IDENTIFIER_COLUMNS_ENV = "EASYICU_MCP_ALLOW_IDENTIFIER_COLUMNS"
MCP_PATIENT_DATA_TOKEN_ENV = "EASYICU_MCP_PATIENT_DATA_TOKEN"
#: Server-owned directory for the patient-data access trail. The audit must not
#: depend on a caller-supplied ``workdir``: a client that simply omits it would
#: otherwise receive rows with no record that it did.
MCP_AUDIT_ROOT_ENV = "EASYICU_MCP_AUDIT_ROOT"

#: Hard ceiling on preview rows even when patient-level disclosure is granted.
MAX_PREVIEW_ROWS = 20

#: Below this row count, per-column aggregate statistics are withheld: a min,
#: max or mean over three rows is close to disclosing the rows themselves.
MIN_ROWS_FOR_AGGREGATE_STATS = 20

#: The same floor applied *per column*. A frame-level row count is not enough:
#: a 100-row extract whose rare assay has one non-missing value would otherwise
#: report ``count=1, mean=min=max=<that patient's value>``, which discloses the
#: row the frame-level floor was meant to protect.
MIN_NON_MISSING_FOR_COLUMN_STATS = 20

_IDENTIFIER_COLUMN = re.compile(
    r"(?i)^(?:subject_id|hadm_id|stay_id|icustay_id|patientunitstayid|"
    r"patienthealthsystemstayid|admissionid|patientid|caseid|person_id|"
    r"encounter_id|uniquepid|.*_id)$"
)
_TIME_COLUMN = re.compile(
    r"(?i)(?:^|_)(?:time|date|datetime|dttm|timestamp|charttime|storetime|"
    r"intime|outtime|starttime|endtime|deathtime|dob|dod|admittime|dischtime)"
    r"(?:$|_)"
)


class MCPAuthorizationError(PermissionError):
    """Raised when a tool call exceeds the granted MCP scopes."""


class MCPPathError(ValueError):
    """Raised when a filesystem argument escapes the configured roots."""


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


#: Per-request narrowing of the process scopes. The HTTP/SSE transport uses
#: this to withhold patient-level disclosure from a caller that authenticated
#: with the general bearer token but did not present the separate patient-data
#: token — including on a loopback bind, where the general token is optional.
_SCOPE_OVERRIDE: ContextVar[Optional[FrozenSet[str]]] = ContextVar(
    "easyicu_mcp_scope_override", default=None
)


@contextmanager
def scope_override(scopes: FrozenSet[str]) -> Iterator[None]:
    """Narrow the granted scopes for the duration of one request."""

    token = _SCOPE_OVERRIDE.set(frozenset(scopes))
    try:
        yield
    finally:
        _SCOPE_OVERRIDE.reset(token)


def process_scopes() -> FrozenSet[str]:
    """Return the scopes the server process was started with.

    ``EASYICU_MCP_SCOPES`` (comma-separated) is the explicit form.
    ``EASYICU_MCP_ALLOW_PATIENT_DATA=1`` is the shorthand for adding
    :data:`SCOPE_READ_PATIENT_DATA` to the default set.
    """

    raw = str(os.environ.get(MCP_SCOPES_ENV, "") or "").strip()
    if raw:
        requested = {part.strip() for part in raw.split(",") if part.strip()}
        unknown = requested - ALL_SCOPES
        if unknown:
            raise ValueError(
                f"unknown {MCP_SCOPES_ENV} value(s): {sorted(unknown)}; "
                f"known scopes are {sorted(ALL_SCOPES)}"
            )
        scopes = frozenset(requested)
    else:
        scopes = DEFAULT_SCOPES
    if _env_flag(MCP_ALLOW_PATIENT_DATA_ENV):
        scopes = scopes | {SCOPE_READ_PATIENT_DATA}
    return scopes


def granted_scopes() -> FrozenSet[str]:
    """Return the scopes in force for the current request."""

    override = _SCOPE_OVERRIDE.get()
    if override is not None:
        return override & process_scopes()
    return process_scopes()


def require_scope(scope: str, *, tool: str) -> None:
    """Raise unless ``scope`` is granted."""

    if scope not in granted_scopes():
        raise MCPAuthorizationError(
            f"tool {tool!r} requires the {scope!r} MCP scope, which this server "
            f"does not grant. Set {MCP_SCOPES_ENV} (comma-separated) or, for "
            f"patient-level disclosure, {MCP_ALLOW_PATIENT_DATA_ENV}=1."
        )


def allowed_roots() -> Tuple[Path, ...]:
    """Return the configured filesystem roots for MCP path arguments.

    Defaults to the working directory the server was started in. An ICU
    database on another volume is reachable only after the operator names it
    in ``EASYICU_MCP_ALLOWED_ROOTS``.
    """

    raw = str(os.environ.get(MCP_ALLOWED_ROOTS_ENV, "") or "")
    roots: List[Path] = []
    for part in raw.split(os.pathsep):
        candidate = part.strip()
        if not candidate:
            continue
        roots.append(Path(candidate).expanduser().resolve())
    return tuple(roots) if roots else (Path.cwd().resolve(),)


def resolve_within_roots(value: Any, *, field: str) -> Path:
    """Resolve ``value`` and require it to sit inside an allowed root.

    The path need not exist yet — extraction output is created later — but
    every existing component is resolved first, so a symlink cannot be used to
    hop outside a root.
    """

    raw = str(value or "").strip()
    if not raw:
        raise MCPPathError(f"{field} must be a non-empty path")
    candidate = Path(raw).expanduser()
    # Resolve the deepest existing ancestor so symlinked parents are followed,
    # then re-attach the not-yet-created tail.
    existing = candidate
    tail: List[str] = []
    while not existing.exists() and existing != existing.parent:
        tail.append(existing.name)
        existing = existing.parent
    resolved = existing.resolve()
    for part in reversed(tail):
        resolved = resolved / part
    if ".." in resolved.parts:
        raise MCPPathError(f"{field} must not contain '..'")

    roots = allowed_roots()
    for root in roots:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return resolved
    raise MCPPathError(
        f"{field}={raw!r} resolves outside the configured MCP roots "
        f"{[str(r) for r in roots]}. Set {MCP_ALLOWED_ROOTS_ENV} "
        f"({os.pathsep}-separated) at server startup to allow it."
    )


def path_digest(value: Any) -> str:
    """Hash a path for audit records so the raw layout is not disclosed."""

    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def is_identifier_column(name: str) -> bool:
    return bool(_IDENTIFIER_COLUMN.match(str(name)))


def is_time_column(name: str, dtype: Any = None) -> bool:
    if dtype is not None and "datetime" in str(dtype).lower():
        return True
    return bool(_TIME_COLUMN.search(str(name)))


@dataclass(frozen=True)
class DisclosurePolicy:
    """What a single tool response may reveal about a frame."""

    patient_data: bool
    preview_rows: int
    include_identifier_columns: bool

    @classmethod
    def current(cls, requested_preview_rows: Any = None) -> "DisclosurePolicy":
        patient_data = SCOPE_READ_PATIENT_DATA in granted_scopes()
        try:
            requested = int(requested_preview_rows or 0)
        except (TypeError, ValueError):
            requested = 0
        rows = max(0, min(requested, MAX_PREVIEW_ROWS)) if patient_data else 0
        return cls(
            patient_data=patient_data,
            preview_rows=rows,
            include_identifier_columns=(
                patient_data and _env_flag(MCP_ALLOW_IDENTIFIER_COLUMNS_ENV)
            ),
        )


def summarise_frame(frame: Any, *, policy: DisclosurePolicy) -> Dict[str, Any]:
    """Project a frame down to what ``policy`` permits.

    Without :data:`SCOPE_READ_PATIENT_DATA` the result is shape, column
    definitions, dtypes, per-column missingness and — only above the
    small-cell floor — aggregate statistics for non-identifier, non-time
    numeric columns. No row ever leaves.
    """

    if not hasattr(frame, "shape") or not hasattr(frame, "columns"):
        return {"type": type(frame).__name__, "repr": repr(frame)[:500]}

    columns = [str(c) for c in frame.columns]
    dtypes = {str(c): str(t) for c, t in frame.dtypes.items()}
    rows = int(frame.shape[0])
    identifier_columns = [c for c in columns if is_identifier_column(c)]
    time_columns = [c for c in columns if is_time_column(c, dtypes.get(c))]

    summary: Dict[str, Any] = {
        "type": type(frame).__name__,
        "rows": rows,
        "columns": columns,
        "dtypes": dtypes,
        "identifier_columns": identifier_columns,
        "time_columns": time_columns,
        "missing_fraction": _missing_fraction(frame, columns),
        "disclosure": {
            "patient_data": policy.patient_data,
            "preview_rows": policy.preview_rows,
        },
    }

    excluded = set(identifier_columns) | set(time_columns)
    summary["aggregate_statistics"] = _aggregate_statistics(
        frame, columns=[c for c in columns if c not in excluded], rows=rows
    )

    if not policy.patient_data or policy.preview_rows <= 0:
        summary["preview"] = []
        summary["preview_withheld_reason"] = (
            "patient-level rows are not disclosed over MCP; grant the "
            f"{SCOPE_READ_PATIENT_DATA!r} scope to receive a capped preview"
        )
        return summary

    preview_frame = frame.head(policy.preview_rows)
    if not policy.include_identifier_columns and excluded:
        keep = [c for c in columns if c not in excluded]
        preview_frame = preview_frame[keep]
        summary["preview_redacted_columns"] = sorted(excluded)
    summary["preview"] = preview_frame.to_dict(orient="records")
    return summary


def _missing_fraction(frame: Any, columns: Sequence[str]) -> Dict[str, float]:
    rows = int(frame.shape[0])
    if rows <= 0:
        return {str(column): 0.0 for column in columns}
    fractions: Dict[str, float] = {}
    for column in columns:
        try:
            missing = int(frame[column].isna().sum())
        except Exception:  # pragma: no cover - exotic dtypes
            continue
        fractions[str(column)] = round(missing / rows, 6)
    return fractions


def _aggregate_statistics(
    frame: Any, *, columns: Sequence[str], rows: int
) -> Dict[str, Any]:
    if rows < MIN_ROWS_FOR_AGGREGATE_STATS:
        return {
            "withheld": True,
            "reason": (
                f"fewer than {MIN_ROWS_FOR_AGGREGATE_STATS} rows; aggregate "
                "statistics over a small cell can disclose the rows themselves"
            ),
        }
    stats: Dict[str, Any] = {}
    for column in columns:
        try:
            series = frame[column]
            if "float" not in str(series.dtype) and "int" not in str(series.dtype):
                continue
            non_missing = int(series.notna().sum())
            if non_missing < MIN_NON_MISSING_FOR_COLUMN_STATS:
                # min/max/mean over a handful of contributing rows reproduces
                # those rows.  Report only the suppressed cell size, which is
                # already visible through ``missing_fraction``.
                stats[str(column)] = {
                    "withheld": True,
                    "non_missing_count": non_missing,
                    "reason": (
                        f"fewer than {MIN_NON_MISSING_FOR_COLUMN_STATS} "
                        "non-missing values in this column"
                    ),
                }
                continue
            described = series.describe()
        except Exception:  # pragma: no cover - exotic dtypes
            continue
        stats[str(column)] = {
            key: (None if _is_nan(value) else float(value))
            for key, value in described.items()
            if key in {"count", "mean", "std", "min", "25%", "50%", "75%", "max"}
        }
    return stats


def _is_nan(value: Any) -> bool:
    try:
        return value != value  # noqa: PLR0124 - NaN self-inequality
    except Exception:  # pragma: no cover
        return False


def patient_data_audit_root() -> Path:
    """Return the server-owned directory the patient-data trail is written to.

    Deliberately independent of the caller's ``workdir``: the audit answers
    "who read patient rows from this host", so a client must not be able to
    redirect it, and must not be able to suppress it by leaving ``workdir``
    unset. Defaults to the first configured root so a stdio server started
    with no extra configuration still keeps a trail.
    """

    configured = (os.environ.get(MCP_AUDIT_ROOT_ENV) or "").strip()
    root = Path(configured).expanduser() if configured else allowed_roots()[0]
    return (root / ".easyicu_mcp_audit").resolve()


def patient_data_audit_payload(
    *,
    tool: str,
    concepts: Sequence[str],
    database: Any,
    data_path: Any,
    patient_ids: Any,
    frame_summaries: Mapping[str, Any],
    output_paths: Sequence[Path],
    policy: DisclosurePolicy,
) -> Dict[str, Any]:
    """Build the PHI-free audit record for one MCP extraction call."""

    if isinstance(patient_ids, (list, tuple, set)):
        n_requested: Optional[int] = len(patient_ids)
    else:
        n_requested = None
    return {
        "schema": "easyicu.mcp_patient_data_access/1",
        "tool": tool,
        "caller": "mcp_client",
        "concepts": [str(c) for c in concepts],
        "database": str(database) if database else None,
        "data_path_sha256": path_digest(data_path) if data_path else None,
        "requested_patient_ids": n_requested,
        "returned_rows": {
            str(name): int(summary.get("rows") or 0)
            for name, summary in frame_summaries.items()
            if isinstance(summary, Mapping)
        },
        # The actual number of rows this response carried, not the configured
        # cap: a request capped at 20 that returned 3 must not be audited as 20,
        # and a capped request over an empty frame must not look like a leak.
        "disclosed_patient_rows": sum(
            len(summary.get("preview") or ())
            for summary in frame_summaries.values()
            if isinstance(summary, Mapping)
        ),
        "disclosed_patient_rows_cap": (
            policy.preview_rows if policy.patient_data else 0
        ),
        "output_path_sha256": [path_digest(p) for p in output_paths],
        "granted_scopes": sorted(granted_scopes()),
    }


__all__ = [
    "ALL_SCOPES",
    "DEFAULT_SCOPES",
    "MAX_PREVIEW_ROWS",
    "MCP_ALLOWED_ROOTS_ENV",
    "MCP_ALLOW_IDENTIFIER_COLUMNS_ENV",
    "MCP_ALLOW_PATIENT_DATA_ENV",
    "MCP_AUDIT_ROOT_ENV",
    "MCP_PATIENT_DATA_TOKEN_ENV",
    "MCP_SCOPES_ENV",
    "MIN_NON_MISSING_FOR_COLUMN_STATS",
    "MIN_ROWS_FOR_AGGREGATE_STATS",
    "SCOPE_BIND_EVIDENCE",
    "SCOPE_METADATA",
    "SCOPE_READ_INTERNAL_CONTEXT",
    "SCOPE_READ_PATIENT_DATA",
    "SCOPE_RUN_PIPELINE",
    "SCOPE_WRITE_ARTIFACTS",
    "patient_data_audit_root",
    "DisclosurePolicy",
    "MCPAuthorizationError",
    "MCPPathError",
    "allowed_roots",
    "granted_scopes",
    "is_identifier_column",
    "is_time_column",
    "patient_data_audit_payload",
    "path_digest",
    "process_scopes",
    "require_scope",
    "scope_override",
    "resolve_within_roots",
    "summarise_frame",
]
