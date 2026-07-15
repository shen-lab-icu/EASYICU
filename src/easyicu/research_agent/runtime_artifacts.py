"""Runtime artifacts for audit logging, workflow graphs, and replay."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .architecture import SystemLayer
from .schema import AnalysisPlan, ResearchContext, ValidationFinding


def _record_field(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def verified_run_evidence_path(
    run_dir: str | Path,
    record: Any,
) -> Optional[Path]:
    """Return the digest-verified evidence file for ``record``.

    Evidence metadata is untrusted input at every reader boundary.  A record is
    therefore usable only when its path is relative to the run's canonical
    ``evidence`` directory, no path component is a symbolic link, the target is
    a regular file, and its bytes still match the registered SHA-256 digest.
    ``None`` is the fail-closed result for malformed, stale, or escaped records.
    """

    relative_text = str(_record_field(record, "relative_path") or "").strip()
    expected_digest = str(_record_field(record, "sha256") or "").strip().lower()
    relative = Path(relative_text)
    if (
        not relative_text
        or "\x00" in relative_text
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or len(expected_digest) != 64
        or any(char not in "0123456789abcdef" for char in expected_digest)
    ):
        return None

    run_root = Path(run_dir).expanduser().resolve()
    evidence_root = run_root / "evidence"
    try:
        if evidence_root.is_symlink() or not evidence_root.is_dir():
            return None
        evidence_root_resolved = evidence_root.resolve(strict=True)
    except OSError:
        return None

    parts = relative.parts
    if parts and parts[0] == "evidence":
        parts = parts[1:]
    if not parts:
        return None
    candidate = evidence_root.joinpath(*parts)
    current = evidence_root
    try:
        for part in parts:
            current = current / part
            if current.is_symlink():
                return None
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(evidence_root_resolved)
        if not stat.S_ISREG(candidate.stat(follow_symlinks=False).st_mode):
            return None
        if _sha256_of_file(candidate) != expected_digest:
            return None
    except (FileNotFoundError, OSError, ValueError):
        return None
    return candidate


def current_step_records(
    per_step_records: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    """Return the latest outer record for each step.

    Evidence blobs and step directories are intentionally append-only across a
    resume.  The per-step ledger is therefore the execution authority: a later
    failed/blocked checkpoint supersedes an earlier ``status="ok"`` checkpoint
    for the same step without deleting its historical artifacts.
    """

    latest_by_step: Dict[str, Mapping[str, Any]] = {}
    for record in per_step_records or []:
        if not isinstance(record, Mapping):
            continue
        step_id = str(record.get("step_id") or "").strip()
        if step_id:
            latest_by_step[step_id] = record
    return list(latest_by_step.values())


def current_successful_step_records(
    per_step_records: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    """Latest per-step records whose outer execution status is exactly OK."""

    return [
        record
        for record in current_step_records(per_step_records)
        if str(record.get("status") or "").strip().lower() == "ok"
    ]


def current_successful_step_ids(
    per_step_records: Sequence[Mapping[str, Any]],
) -> set[str]:
    """Step ids currently authorised to contribute scientific artifacts."""

    return {
        str(record.get("step_id") or "").strip()
        for record in current_successful_step_records(per_step_records)
        if str(record.get("step_id") or "").strip()
    }


def active_step_evidence_ids(
    per_step_records: Sequence[Mapping[str, Any]],
) -> set[str]:
    """Evidence ids referenced by the latest successful step checkpoints."""

    return {
        evidence_id
        for evidence_ids in active_step_evidence_ids_by_step(per_step_records).values()
        for evidence_id in evidence_ids
    }


def active_step_evidence_ids_by_step(
    per_step_records: Sequence[Mapping[str, Any]],
) -> Dict[str, set[str]]:
    """Current evidence authority keyed by its exact producing step.

    A flat set is insufficient for claim publication because a retired claim
    from one step could otherwise borrow an active evidence id from another.
    """

    return {
        str(record.get("step_id") or "").strip(): {
            str(evidence_id)
            for evidence_id in (record.get("evidence_ids") or [])
            if str(evidence_id).strip()
        }
        for record in current_successful_step_records(per_step_records)
        if str(record.get("step_id") or "").strip()
    }


def run_level_evidence_matches_claim_owner(
    *,
    claim_step_id: str,
    evidence_id: str,
) -> bool:
    """Bind a run-level numeric claim to its exact semantic record family."""

    owner = str(claim_step_id or "").strip()
    candidate = str(evidence_id or "").strip()
    if not owner or not candidate:
        return False
    if candidate == owner:
        return True
    version = candidate.removeprefix(f"{owner}_v")
    return version != candidate and version.isdigit() and int(version) >= 2


def current_evidence_records(
    evidence_records: Sequence[Any],
    per_step_records: Optional[Sequence[Mapping[str, Any]]],
) -> List[Any]:
    """Filter step-produced evidence to the current successful checkpoints.

    Run-level records have no ``produced_by_step`` and remain visible.  When no
    step ledger is supplied (legacy readers/tests), preserve the historical
    behaviour and return every record.
    """

    if per_step_records is None:
        return list(evidence_records)
    active_ids_by_step = {
        str(record.get("step_id") or "").strip(): {
            str(evidence_id)
            for evidence_id in (record.get("evidence_ids") or [])
            if str(evidence_id).strip()
        }
        for record in current_successful_step_records(per_step_records)
        if str(record.get("step_id") or "").strip()
    }
    current: List[Any] = []
    for record in evidence_records:
        if isinstance(record, Mapping):
            produced_by_step = record.get("produced_by_step")
            evidence_id = record.get("evidence_id")
        else:
            produced_by_step = getattr(record, "produced_by_step", None)
            evidence_id = getattr(record, "evidence_id", None)
        producer = str(produced_by_step or "").strip()
        if not producer or str(evidence_id or "") in active_ids_by_step.get(
            producer, set()
        ):
            current.append(record)
    return current


class RunArtifactAuthorityError(ValueError):
    """A run checkpoint exists but cannot safely establish current authority."""


@contextmanager
def _checkpoint_write_lock(run_dir: Path):
    """Serialize checkpoint sequence allocation across local resume processes."""

    import fcntl

    lock_path = run_dir / ".manifest.checkpoint.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _checkpoint_sequence(payload: Mapping[str, Any]) -> Optional[int]:
    value = payload.get("checkpoint_sequence")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _next_checkpoint_sequence(run_dir: Path) -> int:
    sequences: List[int] = []
    for name in ("manifest_partial.json", "manifest.json"):
        path = run_dir / name
        if not path.exists():
            continue
        payload = _read_run_checkpoint(path, newest=True)
        sequence = _checkpoint_sequence(payload)
        if sequence is not None:
            sequences.append(sequence)
    return max(sequences, default=0) + 1


def write_run_checkpoint(
    path: str | Path,
    payload: Mapping[str, Any],
) -> int:
    """Atomically persist one monotonically sequenced run checkpoint.

    The temporary file lives beside the destination so ``os.replace`` is an
    atomic same-filesystem operation.  A small advisory lock makes sequence
    allocation and replacement indivisible across concurrent local resumes.
    """

    destination = Path(path)
    run_dir = destination.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    with _checkpoint_write_lock(run_dir):
        sequence = _next_checkpoint_sequence(run_dir)
        body = dict(payload)
        body["checkpoint_sequence"] = sequence
        encoded = json.dumps(
            body,
            indent=2,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=run_dir,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            directory_fd = os.open(run_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return sequence


def _read_run_checkpoint(path: Path, *, newest: bool) -> Dict[str, Any]:
    position = "newest " if newest else ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RunArtifactAuthorityError(
            f"Run {position}checkpoint {path.name} is corrupt or unreadable; "
            "refusing to fall back to older artifact authority."
        ) from exc
    if not isinstance(payload, dict):
        raise RunArtifactAuthorityError(
            f"Run {position}checkpoint {path.name} is corrupt: expected a JSON "
            "object; refusing to fall back to older artifact authority."
        )
    return payload


def load_run_artifact_authority(
    run_dir: str | Path,
) -> Optional[Dict[str, Any]]:
    """Load the newest checkpoint that declares a per-step execution ledger.

    The most recently modified manifest is the authority candidate; a live
    resumed partial can therefore supersede an older final manifest, while a
    stale partial cannot mask a later final failure.  The candidate is selected
    *before* parsing so a corrupt or unreadable newest checkpoint fails closed
    instead of replaying an older success or masquerading as a legacy run.  A
    mapping that explicitly contains
    ``per_step_records`` is modern authority even when that field is empty or
    malformed; callers must not fall back to append-only step directories in
    that case.  ``None`` therefore means *legacy, no ledger field anywhere*, not
    merely "no currently successful steps".  A run with no manifest at all,
    or only a valid legacy manifest that never declared a ledger, retains the
    historical ``None`` compatibility signal.
    """

    root = Path(run_dir)
    candidates: List[tuple[int, int, Path]] = []
    for name in ("manifest_partial.json", "manifest.json"):
        path = root / name
        try:
            modified_ns = path.stat().st_mtime_ns
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RunArtifactAuthorityError(
                f"Run checkpoint {path.name} cannot be inspected; refusing to "
                "fall back to another artifact authority."
            ) from exc
        # Prefer the actually newest durable checkpoint.  Partial wins only on
        # a timestamp tie, preserving the live-resume convention without
        # letting a stale partial mask a later final failure.
        partial_tiebreak = 1 if name == "manifest_partial.json" else 0
        candidates.append((modified_ns, partial_tiebreak, path))
    if not candidates:
        return None

    ordered = sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True)
    newest_path = ordered[0][2]
    newest_payload = _read_run_checkpoint(newest_path, newest=True)
    parsed: List[tuple[int, int, Path, Dict[str, Any]]] = [
        (*ordered[0], newest_payload)
    ]
    for modified_ns, tiebreak, older_path in ordered[1:]:
        try:
            older_payload = _read_run_checkpoint(older_path, newest=False)
        except RunArtifactAuthorityError:
            # An unreadable older checkpoint cannot supersede a readable newer
            # one.  It remains historical corruption, not current authority.
            continue
        parsed.append((modified_ns, tiebreak, older_path, older_payload))

    newest_sequence = _checkpoint_sequence(newest_payload)
    sequenced = [
        (sequence, modified_ns, tiebreak, path, payload)
        for modified_ns, tiebreak, path, payload in parsed
        if (sequence := _checkpoint_sequence(payload)) is not None
    ]
    if newest_sequence is None and sequenced:
        raise RunArtifactAuthorityError(
            f"Run newest checkpoint {newest_path.name} lacks a monotonic "
            "checkpoint_sequence while an older sequenced checkpoint exists; "
            "refusing ambiguous authority rollback."
        )
    if sequenced:
        _, _, _, authority_path, authority_payload = max(
            sequenced,
            key=lambda item: (item[0], item[1], item[2]),
        )
    else:
        authority_path, authority_payload = newest_path, newest_payload
    if "per_step_records" in authority_payload:
        return authority_payload

    # A single valid pre-ledger manifest is a genuine legacy run.  When an
    # older modern ledger also exists, however, the newest ledger-less object
    # is a damaged checkpoint boundary, not permission to replay old success.
    for _, _, older_path, older_payload in parsed:
        if older_path == authority_path:
            continue
        if "per_step_records" in older_payload:
            raise RunArtifactAuthorityError(
                f"Run newest checkpoint {authority_path.name} does not declare "
                "per_step_records; refusing to replay the older checkpoint "
                f"{older_path.name}."
            )
    return None


def current_run_evidence_records(
    run_dir: str | Path,
    *,
    per_step_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Optional[List[Mapping[str, Any]]]:
    """Evidence authorised by the run's current successful checkpoints.

    Returns ``None`` only for a legacy run with no per-step ledger and no
    explicit ledger supplied by the caller.  An empty list is authoritative for
    a modern run with no active evidence, preserving the crucial distinction
    between "nothing active" and "scan the filesystem as a fallback".
    """

    authority = load_run_artifact_authority(run_dir)
    if authority is None:
        if per_step_records is None:
            return None
        evidence_records: Sequence[Any] = []
    else:
        raw_evidence = authority.get("evidence")
        evidence_records = raw_evidence if isinstance(raw_evidence, list) else []
    if per_step_records is None:
        raw_records = (authority or {}).get("per_step_records")
        per_step_records = raw_records if isinstance(raw_records, list) else []
    return [
        record
        for record in current_evidence_records(evidence_records, per_step_records)
        if isinstance(record, Mapping)
    ]


def current_run_evidence_paths(
    run_dir: str | Path,
    *,
    per_step_records: Optional[Sequence[Mapping[str, Any]]] = None,
    evidence_ids: Optional[Iterable[str]] = None,
) -> Optional[List[Path]]:
    """Existing, run-contained paths for current authorised evidence records.

    Like :func:`current_run_evidence_records`, ``None`` is the legacy signal;
    ``[]`` is a modern run with no authorised evidence.  Paths escaping the run
    directory are ignored so manifest data cannot widen a reader's scope.
    """

    records = current_run_evidence_records(
        run_dir,
        per_step_records=per_step_records,
    )
    if records is None:
        return None
    allowed_ids = (
        {str(evidence_id) for evidence_id in evidence_ids}
        if evidence_ids is not None
        else None
    )
    paths: List[Path] = []
    seen: set[Path] = set()
    for record in records:
        if (
            allowed_ids is not None
            and str(record.get("evidence_id") or "") not in allowed_ids
        ):
            continue
        path = verified_run_evidence_path(run_dir, record)
        if path is not None and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _git_field(args: List[str], *, preserve_empty: bool = False) -> Optional[str]:
    """Run a short read-only git command, returning stripped stdout or None.

    Never raises: any failure (git missing, not a repo, timeout) yields
    None so code-version capture degrades gracefully in shipped installs.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    value = (out.stdout or "").strip()
    if value or preserve_empty:
        return value
    return None


def capture_code_version() -> Optional[Dict[str, Any]]:
    """Capture the code identity (git sha/branch/dirty + package version).

    Ties a run manifest back to the exact source that produced it, which the
    reproducibility claim depends on. All fields are best-effort: a shipped
    wheel with no git checkout still records ``package_version``; a git
    checkout with no installed metadata still records the sha. Returns None
    only when BOTH sources fail.
    """
    sha = _git_field(["rev-parse", "HEAD"])
    branch = _git_field(["rev-parse", "--abbrev-ref", "HEAD"])
    # --porcelain status: any output => working tree has uncommitted changes.
    # An empty successful porcelain response is authoritative evidence of a
    # clean checkout, not the same state as git being unavailable.
    status = _git_field(["status", "--porcelain"], preserve_empty=True)
    dirty = None if status is None else bool(status)

    package_version: Optional[str] = None
    try:
        import easyicu

        package_version = getattr(easyicu, "__version__", None)
        if package_version is None:
            from importlib.metadata import version as _pkg_version

            package_version = _pkg_version("easyicu")
    except Exception:
        package_version = None

    if sha is None and package_version is None:
        return None
    return {
        "git_sha": sha,
        "git_branch": branch,
        "git_dirty": dirty,
        "package_version": package_version,
    }


class AuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    phase: str
    event: str
    status: str = "running"
    step_id: Optional[str] = None
    detail: Dict[str, Any] = Field(default_factory=dict)


class AuditLogger:
    """Append-only JSONL audit log for runtime supervision."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        *,
        phase: str,
        event: str,
        status: str = "running",
        step_id: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        record = AuditEvent(
            phase=phase,
            event=event,
            status=status,
            step_id=step_id,
            detail=dict(detail or {}),
        )
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")
        return record


class WorkflowNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    label: str
    layer: SystemLayer
    kind: str
    status: str = "pending"
    detail: Dict[str, Any] = Field(default_factory=dict)


class WorkflowEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    relation: str


class WorkflowGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "easyicu.research_workflow_graph/1"
    run_id: str
    nodes: List[WorkflowNode] = Field(default_factory=list)
    edges: List[WorkflowEdge] = Field(default_factory=list)


class ReplayStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    status: str
    generation_mode: Optional[str] = None
    returncode: Optional[int] = None
    evidence_ids: List[str] = Field(default_factory=list)
    interpretation_evidence_id: Optional[str] = None


class ExecutionReplayBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "easyicu.research_execution_replay/1"
    run_id: str
    cohort_sha256: str
    llm_signature: str
    prompt_pack_version: Optional[str] = None
    context_path: str
    plan_path: str
    steps: List[ReplayStep] = Field(default_factory=list)
    finding_count: int = 0
    evidence_ids: List[str] = Field(default_factory=list)


def build_workflow_graph(
    *,
    run_id: str,
    context: ResearchContext,
    plan: AnalysisPlan,
    per_step_records: Sequence[Dict[str, Any]],
    paused_after_analysis: bool,
) -> WorkflowGraph:
    graph = WorkflowGraph(run_id=run_id)
    graph.nodes.extend(
        [
            WorkflowNode(
                node_id="context",
                label=f"Cohort context ({context.cohort.database})",
                layer=SystemLayer.ICU_DATA_FOUNDATION,
                kind="context",
                status="ok",
                detail={"n_stays": context.cohort.n_stays},
            ),
            WorkflowNode(
                node_id="plan",
                label="Planner / RuntimeSupervisor",
                layer=SystemLayer.AGENT_ORCHESTRATION,
                kind="plan",
                status="ok",
                detail={"n_steps": len(plan.steps)},
            ),
            WorkflowNode(
                node_id="evidence",
                label="Evidence-bound runtime",
                layer=SystemLayer.SAFE_ANALYTICAL_RUNTIME,
                kind="evidence_store",
                status="ok",
            ),
            WorkflowNode(
                node_id="manuscript" if not paused_after_analysis else "analysis_pause",
                label=(
                    "Scientific discovery / manuscript output"
                    if not paused_after_analysis
                    else "Pause after analysis"
                ),
                layer=SystemLayer.SCIENTIFIC_DISCOVERY,
                kind="write" if not paused_after_analysis else "pause",
                status="paused" if paused_after_analysis else "ok",
            ),
        ]
    )
    graph.edges.extend(
        [
            WorkflowEdge(source="context", target="plan", relation="grounds"),
            WorkflowEdge(source="plan", target="evidence", relation="executes_through"),
            WorkflowEdge(
                source="evidence",
                target="manuscript" if not paused_after_analysis else "analysis_pause",
                relation="binds",
            ),
        ]
    )
    for rec in per_step_records:
        step_id = str(rec.get("step_id") or "")
        if not step_id:
            continue
        graph.nodes.append(
            WorkflowNode(
                node_id=step_id,
                label=step_id,
                layer=(
                    SystemLayer.AGENT_ORCHESTRATION
                    if step_id != "00_probe"
                    else SystemLayer.ICU_DATA_FOUNDATION
                ),
                kind="step",
                status=str(rec.get("status") or "unknown"),
                detail={
                    "intent": rec.get("intent"),
                    "generation_mode": rec.get("generation_mode"),
                },
            )
        )
        graph.edges.append(
            WorkflowEdge(source="plan", target=step_id, relation="contains")
        )
        graph.edges.append(
            WorkflowEdge(source=step_id, target="evidence", relation="registers")
        )
    return graph


def render_workflow_graph_mermaid(graph: WorkflowGraph) -> str:
    lines = ["```mermaid", "flowchart TD"]
    for node in graph.nodes:
        label = node.label.replace('"', "'")
        lines.append(f'    {node.node_id}["{label}\\n{node.status}"]')
    for edge in graph.edges:
        rel = edge.relation.replace('"', "'")
        lines.append(f'    {edge.source} -->|"{rel}"| {edge.target}')
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def build_execution_replay(
    *,
    run_id: str,
    cohort_path: Path,
    context_path: str,
    plan_path: str,
    llm_signature: str,
    prompt_pack_version: Optional[str],
    per_step_records: Sequence[Dict[str, Any]],
    findings: Sequence[ValidationFinding],
    evidence_ids: Iterable[str],
) -> ExecutionReplayBundle:
    return ExecutionReplayBundle(
        run_id=run_id,
        cohort_sha256=_sha256_of_file(cohort_path),
        llm_signature=llm_signature,
        prompt_pack_version=prompt_pack_version,
        context_path=context_path,
        plan_path=plan_path,
        steps=[
            ReplayStep(
                step_id=str(rec.get("step_id") or ""),
                status=str(rec.get("status") or "unknown"),
                generation_mode=(
                    str(rec.get("generation_mode"))
                    if rec.get("generation_mode") is not None
                    else None
                ),
                returncode=(
                    int(rec["returncode"])
                    if rec.get("returncode") is not None
                    else None
                ),
                evidence_ids=[str(x) for x in rec.get("evidence_ids", []) or []],
                interpretation_evidence_id=(
                    str(rec.get("interpretation_evidence_id"))
                    if rec.get("interpretation_evidence_id") is not None
                    else None
                ),
            )
            for rec in per_step_records
            if rec.get("step_id")
        ],
        finding_count=len(list(findings)),
        evidence_ids=list(evidence_ids),
    )


def _sha256_of_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_artifact(path: str | Path, payload: BaseModel | Dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = (
        payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    )
    out.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return out


__all__ = [
    "AuditEvent",
    "AuditLogger",
    "WorkflowNode",
    "WorkflowEdge",
    "WorkflowGraph",
    "ExecutionReplayBundle",
    "current_step_records",
    "current_successful_step_records",
    "current_successful_step_ids",
    "active_step_evidence_ids",
    "active_step_evidence_ids_by_step",
    "run_level_evidence_matches_claim_owner",
    "current_evidence_records",
    "verified_run_evidence_path",
    "RunArtifactAuthorityError",
    "write_run_checkpoint",
    "load_run_artifact_authority",
    "current_run_evidence_records",
    "current_run_evidence_paths",
    "build_workflow_graph",
    "render_workflow_graph_mermaid",
    "build_execution_replay",
    "write_json_artifact",
]
