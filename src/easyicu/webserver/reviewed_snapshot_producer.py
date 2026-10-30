"""Host-only boundary for reviewed, closed-dependency snapshot producers.

No production adapters are registered yet. A review and a sandbox alone do
not establish a complete read set; unknown adapters fail before consuming a
decision. Registration is an internal code decision, never a request parameter.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Protocol

from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_json
from easyicu.research_agent.execution.raw_snapshot_sandbox import run_snapshot_worker
from easyicu.webserver.raw_case_snapshot import stage_reviewed_source_files, verify_staged_source_files
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding, consume_host_review, verify_host_review


@dataclass(frozen=True)
class ClosedWorkerSpec:
    source: Path
    source_identity_sha256: str
    files: tuple[str, ...]
    worker: Path
    python: Path
    runtime_roots: tuple[Path, ...]


class ClosedSourceAdapter(Protocol):
    """Trusted host implementation; selection cannot come from a client plugin."""

    def validate_scope(self, scope: Mapping[str, Any]) -> ClosedWorkerSpec:
        """Validate source, plan, producer, configuration and complete read scope without reading source bytes."""

    def verify_source(self, scope: Mapping[str, Any], spec: ClosedWorkerSpec) -> None:
        """After consumption, verify the registered source and its approved roster."""

    def arguments(self, snapshot: Path, output: Path) -> tuple[str, ...]:
        """Provide arguments for the immutable producer selected by the adapter."""

    def verify_output(self, scope: Mapping[str, Any], spec: ClosedWorkerSpec, output: Path) -> None:
        """Validate the exact output roster, schema and content before handoff."""


# Populated only by audited in-tree adapters. No current adapter establishes
# closure for real ICU data or a production user review route.
_CLOSED_ADAPTERS: dict[str, ClosedSourceAdapter] = {}


def require_reviewable_scope(scope: Mapping[str, Any]) -> ClosedWorkerSpec:
    """Validate a candidate without consuming a grant or reading source data."""
    checked = json.loads(canonical_json(dict(scope)))
    adapter_id = checked.get("adapter_id")
    adapter = _CLOSED_ADAPTERS.get(adapter_id) if isinstance(adapter_id, str) else None
    if adapter is None:
        raise ValueError("raw_preparation_read_closure_unverified")
    return adapter.validate_scope(checked)


def run_reviewed_snapshot_producer(
    *, checkpoint: RawPreparationCheckpoint, binding: RawPreparationHostBinding,
    scope: Mapping[str, Any], host_work_root: Path, timeout: float = 120,
):
    """Verify and consume one host decision, then execute a closed producer.

    This is not a research-analysis grant. Host-owned work roots and approved
    runtime directories are caller responsibilities; the receipt does not assert
    a point-in-time source snapshot or the files actually consumed.
    """
    scope = json.loads(canonical_json(dict(scope)))
    adapter_id = scope.get("adapter_id")
    adapter = _CLOSED_ADAPTERS.get(adapter_id) if isinstance(adapter_id, str) else None
    if adapter is None:
        raise ValueError("raw_preparation_read_closure_unverified")
    coordinates = binding.model_dump(mode="python")
    verify_host_review(checkpoint=checkpoint, scope=scope, **coordinates)
    spec = require_reviewable_scope(scope)
    consume_host_review(checkpoint=checkpoint, scope=scope, **coordinates)
    adapter.verify_source(scope, spec)
    root = Path(tempfile.mkdtemp(prefix="reviewed-producer-", dir=host_work_root))
    snapshot, output, scratch = root / "snapshot", root / "output", root / "scratch"
    output.mkdir(mode=0o700)
    scratch.mkdir(mode=0o700)
    receipt = stage_reviewed_source_files(source_root=spec.source, staging_root=snapshot,
                                          relative_files=spec.files,
                                          source_identity_sha256=spec.source_identity_sha256)
    result = run_snapshot_worker(snapshot=snapshot, output=output, python=spec.python,
                                 worker=spec.worker, runtime_roots=spec.runtime_roots,
                                 forbidden_source=spec.source,
                                 scratch=scratch,
                                 arguments=adapter.arguments(snapshot, output), timeout=timeout)
    verify_staged_source_files(snapshot, receipt)
    if result.returncode:
        raise ValueError("raw_preparation_worker_failed")
    adapter.verify_output(scope, spec, output)
    return result, receipt, output
