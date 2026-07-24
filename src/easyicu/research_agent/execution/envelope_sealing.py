"""Single final-gate compiler seam for canonical step-result migration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Literal, Mapping

from easyicu.research_agent.schema import AnalysisStep

from ..contracts.result_envelope import (
    StepResultEnvelope,
    normalize_step_result_shadow,
    verify_step_result_envelope,
)
from .runner import DockerRunner

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SealedStepResultEnvelopeSnapshot:
    """Non-authoritative compiler result returned with final gate findings."""

    envelope: StepResultEnvelope | None
    error_code: (
        Literal[
            "sealed_envelope_compile_failed",
            "sealed_envelope_digest_invalid",
        ]
        | None
    ) = None

    @property
    def ready(self) -> bool:
        return self.envelope is not None and self.error_code is None


def compile_sealed_step_result_shadow(
    *,
    step: AnalysisStep,
    step_summary: Dict[str, Any],
    output_dir: Path,
    run_dir: Path,
    resolved_input_bindings: Mapping[str, Mapping[str, Any]] | None = None,
    current_status: str | None = None,
) -> SealedStepResultEnvelopeSnapshot:
    """Compile one post-repair output view without granting paper authority.

    ``resolved_input_bindings`` is the host-owned result of typed-input
    resolution.  This seam rechecks path structure and containment before
    projecting opaque references, but deliberately does not hash large input
    files again.  The final gate's integrity validators own byte-level
    verification; duplicating that I/O here would make canonicalization scale
    with cohort size.
    """

    raw_summary_path = output_dir / "step_summary.json"
    try:
        authorized_path_refs: dict[str, str] = {}
        run_root = run_dir.resolve(strict=True)
        for raw_binding in (resolved_input_bindings or {}).values():
            binding = dict(raw_binding)
            evidence_id = str(binding.get("evidence_id") or "").strip()
            declared_sha256 = str(binding.get("sha256") or "").strip()
            relative = PurePosixPath(str(binding.get("relative_path") or ""))
            absolute = Path(str(binding.get("absolute_path") or ""))
            if (
                not evidence_id
                or not _SHA256_RE.fullmatch(declared_sha256)
                or relative.is_absolute()
                or not relative.parts
                or any(part in {"", ".", ".."} for part in relative.parts)
                or not absolute.is_absolute()
                or absolute.is_symlink()
            ):
                continue
            resolved = absolute.resolve(strict=True)
            resolved.relative_to(run_root)
            expected = run_root.joinpath(*relative.parts).resolve(strict=True)
            if resolved != expected or not resolved.is_file():
                continue
            opaque_ref = f"evidence:{evidence_id}@sha256:{declared_sha256}"
            authorized_path_refs[str(resolved)] = opaque_ref
            authorized_path_refs[
                (PurePosixPath(DockerRunner.CONTAINER_RUN_ROOT) / relative).as_posix()
            ] = opaque_ref
        envelope = normalize_step_result_shadow(
            step_id=step.step_id,
            step_summary=step_summary,
            output_dir=output_dir,
            status=current_status,
            planned_analysis_role=step.planned_analysis_role,
            raw_summary_artifact_bytes=(
                raw_summary_path.read_bytes() if raw_summary_path.is_file() else None
            ),
            container_output_roots=(DockerRunner.CONTAINER_OUTPUT_ROOT,),
            authorized_path_refs=authorized_path_refs,
        )
    except Exception:
        return SealedStepResultEnvelopeSnapshot(
            envelope=None,
            error_code="sealed_envelope_compile_failed",
        )
    if not verify_step_result_envelope(envelope):
        return SealedStepResultEnvelopeSnapshot(
            envelope=None,
            error_code="sealed_envelope_digest_invalid",
        )
    return SealedStepResultEnvelopeSnapshot(envelope=envelope)


__all__ = [
    "SealedStepResultEnvelopeSnapshot",
    "compile_sealed_step_result_shadow",
]
