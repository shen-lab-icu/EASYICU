"""Runtime artifacts for audit logging, workflow graphs, and replay."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .architecture import SystemLayer
from .schema import AnalysisPlan, ResearchContext, ValidationFinding


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
                label="Scientific discovery / manuscript output" if not paused_after_analysis else "Pause after analysis",
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
                layer=SystemLayer.AGENT_ORCHESTRATION if step_id != "00_probe" else SystemLayer.ICU_DATA_FOUNDATION,
                kind="step",
                status=str(rec.get("status") or "unknown"),
                detail={
                    "intent": rec.get("intent"),
                    "generation_mode": rec.get("generation_mode"),
                },
            )
        )
        graph.edges.append(WorkflowEdge(source="plan", target=step_id, relation="contains"))
        graph.edges.append(WorkflowEdge(source=step_id, target="evidence", relation="registers"))
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
                generation_mode=(str(rec.get("generation_mode")) if rec.get("generation_mode") is not None else None),
                returncode=(int(rec["returncode"]) if rec.get("returncode") is not None else None),
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
    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out


__all__ = [
    "AuditEvent",
    "AuditLogger",
    "WorkflowNode",
    "WorkflowEdge",
    "WorkflowGraph",
    "ExecutionReplayBundle",
    "build_workflow_graph",
    "render_workflow_graph_mermaid",
    "build_execution_replay",
    "write_json_artifact",
]
