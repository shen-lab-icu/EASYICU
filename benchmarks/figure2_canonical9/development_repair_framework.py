"""Case-scoped, no-provider repair readiness for the Canonical9 protocol.

This module deliberately does *not* attempt to repair a scientific question,
materialize a cohort, or invoke an LLM.  It makes the boundary between four
kinds of work executable:

* typed input / provenance authority;
* a case-specific human protocol decision;
* a scientific redesign after a valid negative result; and
* a reproduced, case-neutral engine defect.

The policy is stored beside the Canonical9 benchmark material, rather than in a
shared Agent prompt.  A report from this module is therefore a development
checklist, never an authorization to launch a paper-facing run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Iterable, Mapping

from .evaluator.input_binding_v2 import (
    BlockedCanonicalTaskBinding,
    ReadyCanonicalTaskBinding,
    load_canonical_run_input_bindings,
)
from .evaluator.rubric_v1 import FIGURE2_TASK_IDS

REPAIR_PROTOCOL_SCHEMA = "easyicu.figure2_development_repair_protocol/1"
REPAIR_PROTOCOL_REF = "figure2_canonical9/development_repair/20260722-v1"
_MAX_PROTOCOL_BYTES = 256 * 1024

_VALID_WORK_KINDS = frozenset(
    {
        "typed_input_authority",
        "case_protocol",
        "scientific_redesign",
        "engine_regression",
    }
)
_HUMAN_OWNED_KINDS = frozenset({"typed_input_authority", "case_protocol"})


class DevelopmentRepairProtocolError(ValueError):
    """Raised when the repository-owned repair protocol is malformed."""


@dataclasses.dataclass(frozen=True)
class RepairRequirement:
    """One specific gate for a benchmark task.

    ``auto_action`` is intentionally false for authority, protocol, and science
    work.  Only a separately reproduced ``engine_regression`` may be fixed in
    shared runtime code, and even that remains blocked until its regression test
    passes.  This prevents a repair loop from fabricating scientific authority.
    """

    code: str
    work_kind: str
    resolution: str
    auto_action: bool

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RepairRequirement":
        expected = {"code", "work_kind", "resolution", "auto_action"}
        if set(raw) != expected:
            raise DevelopmentRepairProtocolError(
                "repair requirement keys must exactly match the schema"
            )
        code = raw["code"]
        work_kind = raw["work_kind"]
        resolution = raw["resolution"]
        auto_action = raw["auto_action"]
        if (
            not isinstance(code, str)
            or not code
            or not code.replace("_", "").isalnum()
            or code.upper() != code
        ):
            raise DevelopmentRepairProtocolError("repair requirement code is invalid")
        if work_kind not in _VALID_WORK_KINDS:
            raise DevelopmentRepairProtocolError(
                "repair requirement work_kind is invalid"
            )
        if not isinstance(resolution, str) or not resolution.strip():
            raise DevelopmentRepairProtocolError(
                "repair requirement resolution is invalid"
            )
        if type(auto_action) is not bool:
            raise DevelopmentRepairProtocolError(
                "repair requirement auto_action is invalid"
            )
        if work_kind in _HUMAN_OWNED_KINDS and auto_action:
            raise DevelopmentRepairProtocolError(
                "authority and protocol requirements must not auto-act"
            )
        return cls(code, work_kind, resolution, auto_action)

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class TaskRepairProtocol:
    task_id: str
    requirements: tuple[RepairRequirement, ...]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TaskRepairProtocol":
        if set(raw) != {"task_id", "requirements"}:
            raise DevelopmentRepairProtocolError(
                "task repair protocol keys must exactly match the schema"
            )
        task_id = raw["task_id"]
        requirements = raw["requirements"]
        if task_id not in FIGURE2_TASK_IDS:
            raise DevelopmentRepairProtocolError(
                "task repair protocol has unknown task"
            )
        if not isinstance(requirements, list) or not requirements:
            raise DevelopmentRepairProtocolError(
                "task repair protocol needs requirements"
            )
        parsed = tuple(RepairRequirement.from_dict(item) for item in requirements)
        codes = tuple(item.code for item in parsed)
        if codes != tuple(sorted(codes)) or len(codes) != len(set(codes)):
            raise DevelopmentRepairProtocolError(
                "repair requirement codes must be sorted and unique"
            )
        return cls(task_id, parsed)


@dataclasses.dataclass(frozen=True)
class TaskRepairReadiness:
    task_id: str
    state: str
    input_binding_blockers: tuple[str, ...]
    requirements: tuple[RepairRequirement, ...]

    @property
    def launch_ready(self) -> bool:
        """A report never upgrades input authority; only records readiness."""

        return not self.input_binding_blockers and not self.requirements

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "state": self.state,
            "launch_ready": self.launch_ready,
            "input_binding_blockers": list(self.input_binding_blockers),
            "requirements": [item.to_dict() for item in self.requirements],
        }


def _protocol_path() -> Path:
    return Path(__file__).with_name("development_repair_protocol_v1.json")


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_object(path: Path) -> dict[str, Any]:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_PROTOCOL_BYTES:
            raise DevelopmentRepairProtocolError(
                "repair protocol must be a small regular file"
            )
        raw = b""
        while len(raw) <= _MAX_PROTOCOL_BYTES:
            part = os.read(descriptor, 64 * 1024)
            if not part:
                break
            raw += part
        if len(raw) > _MAX_PROTOCOL_BYTES:
            raise DevelopmentRepairProtocolError("repair protocol exceeds size limit")
    finally:
        if descriptor is not None:
            os.close(descriptor)

    def _duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DevelopmentRepairProtocolError(
                    f"repair protocol has duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DevelopmentRepairProtocolError(
            "repair protocol is not valid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise DevelopmentRepairProtocolError("repair protocol root must be an object")
    if raw != _canonical_json_bytes(value) + b"\n":
        raise DevelopmentRepairProtocolError("repair protocol is not canonical JSON")
    return value


def load_development_repair_protocol(
    path: Path | str | None = None,
) -> tuple[tuple[TaskRepairProtocol, ...], str]:
    """Load the exact-nine, case-scoped repair contract and its byte digest."""

    source = Path(path) if path is not None else _protocol_path()
    raw = _strict_json_object(source)
    expected = {"schema_version", "protocol_ref", "tasks"}
    if set(raw) != expected:
        raise DevelopmentRepairProtocolError("repair protocol keys do not match schema")
    if raw["schema_version"] != REPAIR_PROTOCOL_SCHEMA:
        raise DevelopmentRepairProtocolError("repair protocol schema is unsupported")
    if raw["protocol_ref"] != REPAIR_PROTOCOL_REF:
        raise DevelopmentRepairProtocolError("repair protocol ref is unsupported")
    if not isinstance(raw["tasks"], list):
        raise DevelopmentRepairProtocolError("repair protocol tasks must be a list")
    tasks = tuple(TaskRepairProtocol.from_dict(item) for item in raw["tasks"])
    if tuple(item.task_id for item in tasks) != tuple(FIGURE2_TASK_IDS):
        raise DevelopmentRepairProtocolError(
            "repair protocol must contain exact Canonical9 task order"
        )
    return tasks, hashlib.sha256(_canonical_json_bytes(raw) + b"\n").hexdigest()


def evaluate_development_repair_readiness(
    *, protocol_path: Path | str | None = None
) -> tuple[tuple[TaskRepairReadiness, ...], str, str]:
    """Combine immutable per-task repair requirements with live input gating.

    Returns ``(tasks, repair_protocol_sha256, input_binding_sha256)``.  It is a
    pure, offline report: no provider, data export, materializer, or runner is
    touched.  In particular, a ``ready`` input binding does not erase pending
    scientific requirements and a repair plan never makes a blocked binding ready.
    """

    protocols, protocol_digest = load_development_repair_protocol(protocol_path)
    bindings, binding_digest = load_canonical_run_input_bindings()
    by_id = {item.task_id: item for item in bindings.tasks}
    report: list[TaskRepairReadiness] = []
    for protocol in protocols:
        binding = by_id[protocol.task_id]
        if isinstance(binding, BlockedCanonicalTaskBinding):
            blockers = tuple(binding.blockers)
            state = "blocked"
        elif isinstance(binding, ReadyCanonicalTaskBinding):
            blockers = ()
            state = "ready_input_pending_repair"
        else:  # pragma: no cover - discriminated model makes this impossible
            raise DevelopmentRepairProtocolError("unknown canonical input binding")
        report.append(
            TaskRepairReadiness(
                task_id=protocol.task_id,
                state=state,
                input_binding_blockers=blockers,
                requirements=protocol.requirements,
            )
        )
    return tuple(report), protocol_digest, binding_digest


def render_development_repair_report(
    rows: Iterable[TaskRepairReadiness],
    *,
    repair_protocol_sha256: str,
    input_binding_sha256: str,
) -> dict[str, object]:
    """Render a deterministic report suitable for a task log or preflight."""

    items = tuple(rows)
    if tuple(item.task_id for item in items) != tuple(FIGURE2_TASK_IDS):
        raise DevelopmentRepairProtocolError("report rows must retain Canonical9 order")
    return {
        "schema_version": "easyicu.figure2_development_repair_report/1",
        "repair_protocol_sha256": repair_protocol_sha256,
        "canonical_input_binding_sha256": input_binding_sha256,
        "real_run_authorized": False,
        "note": (
            "Development repair readiness only. A real run additionally requires "
            "the P4 production-input authority and a separately confirmed operator "
            "freeze declaration."
        ),
        "tasks": [item.to_dict() for item in items],
    }


__all__ = [
    "DevelopmentRepairProtocolError",
    "REPAIR_PROTOCOL_REF",
    "REPAIR_PROTOCOL_SCHEMA",
    "RepairRequirement",
    "TaskRepairProtocol",
    "TaskRepairReadiness",
    "evaluate_development_repair_readiness",
    "load_development_repair_protocol",
    "render_development_repair_report",
]
