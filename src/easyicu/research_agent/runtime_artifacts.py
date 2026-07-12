"""Runtime artifacts for audit logging, workflow graphs, and replay."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .architecture import SystemLayer
from .schema import AnalysisPlan, ResearchContext, ValidationFinding


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
        str(evidence_id)
        for record in current_successful_step_records(per_step_records)
        for evidence_id in (record.get("evidence_ids") or [])
        if str(evidence_id).strip()
    }


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
    active_ids = active_step_evidence_ids(per_step_records)
    current: List[Any] = []
    for record in evidence_records:
        if isinstance(record, Mapping):
            produced_by_step = record.get("produced_by_step")
            evidence_id = record.get("evidence_id")
        else:
            produced_by_step = getattr(record, "produced_by_step", None)
            evidence_id = getattr(record, "evidence_id", None)
        if (
            not str(produced_by_step or "").strip()
            or str(evidence_id or "") in active_ids
        ):
            current.append(record)
    return current


def load_run_artifact_authority(
    run_dir: str | Path,
) -> Optional[Dict[str, Any]]:
    """Load the newest checkpoint that declares a per-step execution ledger.

    ``manifest_partial.json`` is preferred because a resumed run can update it
    before a new final manifest is written.  A mapping that explicitly contains
    ``per_step_records`` is modern authority even when that field is empty or
    malformed; callers must not fall back to append-only step directories in
    that case.  ``None`` therefore means *legacy, no ledger field anywhere*, not
    merely "no currently successful steps".
    """

    root = Path(run_dir)
    for name in ("manifest_partial.json", "manifest.json"):
        path = root / name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(payload, dict) and "per_step_records" in payload:
            return payload
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
    root = Path(run_dir).resolve()
    paths: List[Path] = []
    seen: set[Path] = set()
    for record in records:
        if (
            allowed_ids is not None
            and str(record.get("evidence_id") or "") not in allowed_ids
        ):
            continue
        relative_path = str(record.get("relative_path") or "").strip()
        if not relative_path:
            continue
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if path.is_file() and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _git_field(args: List[str]) -> Optional[str]:
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
    return (out.stdout or "").strip() or None


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
    status = _git_field(["status", "--porcelain"])
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
    "current_evidence_records",
    "load_run_artifact_authority",
    "current_run_evidence_records",
    "current_run_evidence_paths",
    "build_workflow_graph",
    "render_workflow_graph_mermaid",
    "build_execution_replay",
    "write_json_artifact",
]
