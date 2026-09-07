"""Typed owner for Web Research Agent run records and their directory layout."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Union

from easyicu.research_agent.contracts.frozen_payload import freeze_payload, thaw_payload


@dataclass(frozen=True)
class RunDirectory:
    """One server-owned Web run wrapper and its fixed child layout."""

    path: Path

    @classmethod
    def resolve(cls, value: object) -> Optional["RunDirectory"]:
        text = str(value or "").strip()
        if not text:
            return None
        return cls(Path(text).expanduser().resolve())

    @classmethod
    def create(cls, project_root: Path, study_id: object, job_id: object) -> "RunDirectory":
        study = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(study_id or "study")).strip("-.")
        study = study[:96] or "study"
        return cls(project_root.expanduser().resolve() / study / f"run_{job_id}")

    def artifact(self, name: str) -> Path:
        if not name or Path(name).name != name:
            raise ValueError("run_artifact_name_invalid")
        return self.path / name

    @property
    def pipeline_root(self) -> Path:
        return self.path / "pipeline"

    def pipeline_run(self, run_id: str) -> Path:
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("pipeline_run_id_invalid")
        return self.pipeline_root / run_id


@dataclass(frozen=True)
class RunGate:
    status: Optional[str]
    reason: Optional[str]
    reportable: bool
    draft_unlocked: bool
    checks: tuple[Mapping[str, Any], ...]
    payload: Mapping[str, Any]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RunGate":
        checks = payload.get("checks")
        rows = checks if isinstance(checks, Sequence) and not isinstance(checks, str) else ()
        return cls(
            status=str(payload["status"]) if payload.get("status") is not None else None,
            reason=str(payload["reason"]) if payload.get("reason") is not None else None,
            reportable=bool(payload.get("reportable")),
            draft_unlocked=bool(payload.get("draft_unlocked")),
            checks=tuple(freeze_payload(row) for row in rows if isinstance(row, Mapping)),
            payload=freeze_payload(payload),
        )

    def to_dict(self) -> Dict[str, Any]:
        return thaw_payload(self.payload)


@dataclass(frozen=True)
class RunReadiness:
    status: str
    signable: bool
    signed: bool
    signoff_stale: bool
    reportable: bool
    draft_unlocked: bool
    gate_status: Optional[str]
    gate_reason: Optional[str]
    checks_total: int
    checks_passed: int
    non_human_failures: tuple[str, ...]
    human_signoff_passed_in_gate: bool
    required_confirmations: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RunReadiness":
        return cls(
            status=str(payload.get("status") or "blocked"),
            signable=bool(payload.get("signable")),
            signed=bool(payload.get("signed")),
            signoff_stale=bool(payload.get("signoff_stale")),
            reportable=bool(payload.get("reportable")),
            draft_unlocked=bool(payload.get("draft_unlocked")),
            gate_status=(str(payload["gate_status"]) if payload.get("gate_status") is not None else None),
            gate_reason=(str(payload["gate_reason"]) if payload.get("gate_reason") is not None else None),
            checks_total=int(payload.get("checks_total") or 0),
            checks_passed=int(payload.get("checks_passed") or 0),
            non_human_failures=tuple(str(item) for item in payload.get("non_human_failures") or ()),
            human_signoff_passed_in_gate=bool(payload.get("human_signoff_passed_in_gate")),
            required_confirmations=tuple(str(item) for item in payload.get("required_confirmations") or ()),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "signable": self.signable,
            "signed": self.signed,
            "signoff_stale": self.signoff_stale,
            "reportable": self.reportable,
            "draft_unlocked": self.draft_unlocked,
            "gate_status": self.gate_status,
            "gate_reason": self.gate_reason,
            "checks_total": self.checks_total,
            "checks_passed": self.checks_passed,
            "non_human_failures": list(self.non_human_failures),
            "human_signoff_passed_in_gate": self.human_signoff_passed_in_gate,
            "required_confirmations": list(self.required_confirmations),
        }


@dataclass(frozen=True)
class RunArtifact:
    name: str
    path: str
    relative_path: str
    bytes: int
    sha256: str
    kind: str
    summary: Any = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RunArtifact":
        return cls(
            name=str(payload.get("name") or ""),
            path=str(payload.get("path") or ""),
            relative_path=str(payload.get("relative_path") or ""),
            bytes=int(payload.get("bytes") or 0),
            sha256=str(payload.get("sha256") or ""),
            kind=str(payload.get("kind") or ""),
            summary=freeze_payload(payload.get("summary")),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "name": self.name,
            "path": self.path,
            "relative_path": self.relative_path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "kind": self.kind,
        }
        if self.summary is not None:
            payload["summary"] = thaw_payload(self.summary)
        return payload


@dataclass(frozen=True)
class RunSignoff:
    run_id: Optional[str]
    reviewer: Optional[str]
    signed_at: Optional[str]
    status: Optional[str]
    payload: Mapping[str, Any]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RunSignoff":
        return cls(
            run_id=str(payload["run_id"]) if payload.get("run_id") is not None else None,
            reviewer=str(payload["reviewer"]) if payload.get("reviewer") is not None else None,
            signed_at=str(payload["signed_at"]) if payload.get("signed_at") is not None else None,
            status=str(payload["status"]) if payload.get("status") is not None else None,
            payload=freeze_payload(payload),
        )

    def to_dict(self) -> Dict[str, Any]:
        return thaw_payload(self.payload)


@dataclass(frozen=True)
class RunRecord:
    ok: Literal[True]
    directory: RunDirectory
    run_id: Optional[str]
    run_type: str
    study_id: Optional[str]
    scientific_configuration_sha256: Optional[str]
    mode: Optional[str]
    engine: Optional[str]
    gate: RunGate
    readiness: RunReadiness
    signoff_stale: bool
    signoff_integrity: Mapping[str, Any]
    signoff: Optional[RunSignoff]
    artifacts: tuple[RunArtifact, ...]
    artifact_payloads: Mapping[str, Mapping[str, Any]]

    @property
    def signed(self) -> bool:
        return self.signoff is not None and bool(self.signoff.payload)

    @classmethod
    def build(
        cls,
        *,
        directory: RunDirectory,
        run_context: Mapping[str, Any],
        ledger: Mapping[str, Any],
        gate_payload: Mapping[str, Any],
        readiness_payload: Mapping[str, Any],
        signoff_payload: Optional[Mapping[str, Any]],
        signoff_integrity: Mapping[str, Any],
        artifacts: Sequence[Mapping[str, Any]],
        artifact_payloads: Mapping[str, Mapping[str, Any]],
    ) -> "RunRecord":
        return cls(
            ok=True,
            directory=directory,
            run_id=(str(run_context.get("run_id") or ledger.get("run_id")) if run_context.get("run_id") or ledger.get("run_id") else None),
            run_type=str(ledger.get("run_type") or "preflight"),
            study_id=str(run_context["study_id"]) if run_context.get("study_id") is not None else None,
            scientific_configuration_sha256=(str(run_context["scientific_configuration_sha256"]) if run_context.get("scientific_configuration_sha256") is not None else None),
            mode=str(run_context["mode"]) if run_context.get("mode") is not None else None,
            engine=str(run_context["engine"]) if run_context.get("engine") is not None else None,
            gate=RunGate.from_payload(gate_payload),
            readiness=RunReadiness.from_payload(readiness_payload),
            signoff_stale=bool(signoff_integrity.get("signoff_stale")),
            signoff_integrity=freeze_payload(signoff_integrity),
            signoff=(RunSignoff.from_payload(signoff_payload) if signoff_payload is not None else None),
            artifacts=tuple(RunArtifact.from_payload(item) for item in artifacts),
            artifact_payloads=freeze_payload(artifact_payloads),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "project_dir": str(self.directory.path),
            "run_id": self.run_id,
            "run_type": self.run_type,
            "study_id": self.study_id,
            "scientific_configuration_sha256": self.scientific_configuration_sha256,
            "mode": self.mode,
            "engine": self.engine,
            "gate": self.gate.to_dict(),
            "readiness": self.readiness.to_dict(),
            "signed": self.signed,
            "signoff_stale": self.signoff_stale,
            "signoff_integrity": thaw_payload(self.signoff_integrity),
            "signoff": self.signoff.to_dict() if self.signoff is not None else None,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "artifact_payloads": thaw_payload(self.artifact_payloads),
        }


@dataclass(frozen=True)
class RunRecordReadError:
    ok: Literal[False]
    error: str
    directory: Optional[RunDirectory] = None
    artifact: Optional[str] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"ok": False, "error": self.error}
        if self.directory is not None:
            payload["project_dir"] = str(self.directory.path)
        if self.artifact is not None:
            payload["artifact"] = self.artifact
        if self.message is not None:
            payload["message"] = self.message
        return payload


RunRecordReadResult = Union[RunRecord, RunRecordReadError]


def read_error(payload: Mapping[str, Any], directory: Optional[RunDirectory]) -> RunRecordReadError:
    return RunRecordReadError(
        ok=False,
        error=str(payload.get("error") or "run_record_unavailable"),
        directory=directory,
        artifact=str(payload["artifact"]) if payload.get("artifact") is not None else None,
        message=str(payload["message"]) if payload.get("message") is not None else None,
    )
