"""Explicit image revisions for an already approved development execution.

The scientific approval remains bound to its original configuration. This
separate host receipt proves the sole infrastructure change instead of
rewriting that approval or presenting the new configuration as the old one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..canonical_json import canonical_sha256
from .config import PipelineConfig
from .human_review_checkpoint import (
    HumanReviewCheckpointError,
    completed_review_authorizes_exact_retry,
    checkpoint_path as human_review_checkpoint_path,
    load_checkpoint,
)
from .profiles import is_paper_facing_profile


@dataclass(frozen=True)
class ExecutionRuntimeRevision:
    """A credential-free source configuration, bound to one completed review."""

    approved_config_json: str
    approved_config_sha256: str
    target_config_sha256: str
    checkpoint_sha256: str
    run_id: str

    def validate(self, *, config: PipelineConfig, run_dir: Path):
        approved = PipelineConfig.from_recovery_payload(
            json.loads(self.approved_config_json),
            expected_digest=self.approved_config_sha256,
        )
        before, after = approved.recovery_payload(), config.recovery_payload()
        changed = {
            key
            for key in before.keys() | after.keys()
            if before.get(key) != after.get(key)
        }
        if (
            changed != {"runner_image"}
            or config.canonical_digest() != self.target_config_sha256
            or approved.runner_kind != "docker"
            or approved.runner_network != "none"
            or approved.expected_runner_image_digest is not None
            or is_paper_facing_profile(approved.submission_profile_name)
            or not approved.require_human_plan_review
            or not config.runner_image
        ):
            raise HumanReviewCheckpointError(
                "runtime revision must change only an unpinned development Docker image"
            )
        if Path(run_dir).resolve() != (Path(approved.workdir) / self.run_id).resolve():
            raise HumanReviewCheckpointError("runtime revision selected another run")
        path = human_review_checkpoint_path(run_dir)
        checkpoint = load_checkpoint(path, require_pending=False)
        if (
            checkpoint.checkpoint_sha256 != self.checkpoint_sha256
            or checkpoint.run_id != self.run_id
        ):
            raise HumanReviewCheckpointError(
                "runtime revision source checkpoint changed"
            )
        if not completed_review_authorizes_exact_retry(
            path,
            pipeline_config_sha256=approved.canonical_digest(),
            run_input_capsule_sha256=checkpoint.run_input_capsule_sha256,
            plan_payload=checkpoint.plan_handoff["plan"],
        ):
            raise HumanReviewCheckpointError(
                "runtime revision requires a completed execution approval"
            )
        return checkpoint

    def authorize_and_record(
        self,
        *,
        config: PipelineConfig,
        run_dir: Path,
        plan_payload: Mapping[str, Any],
        run_input_capsule_sha256: str,
        runtime_bundle: Mapping[str, Any] | None,
        runtime_capabilities: Sequence[str],
        evidence: Any,
    ) -> None:
        checkpoint = self.validate(config=config, run_dir=run_dir)
        if (
            canonical_sha256(plan_payload)
            != canonical_sha256(checkpoint.plan_handoff["plan"])
            or run_input_capsule_sha256 != checkpoint.run_input_capsule_sha256
        ):
            raise HumanReviewCheckpointError(
                "runtime revision changed the reviewed plan or input capsule"
            )
        provenance = (runtime_bundle or {}).get("provenance")
        if (
            (runtime_bundle or {}).get("schema") != "easyicu.docker_runtime_preflight/3"
            or not isinstance(provenance, Mapping)
            or provenance.get("runtime") != "docker"
            or provenance.get("image_reference") != config.runner_image
            or provenance.get("network") != "none"
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(provenance.get("image_id") or "")
            )
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(provenance.get("execution_kernel_identity_sha256") or ""),
            )
            or not set(checkpoint.runtime_capabilities).issubset(runtime_capabilities)
        ):
            raise HumanReviewCheckpointError(
                "runtime revision lacks a matching validated Docker kernel"
            )
        payload = {
            "schema_version": "easyicu.execution_runtime_revision/1",
            "authority_kind": "development_environment_revision",
            "paper_authority": False,
            "run_id": self.run_id,
            "approved_checkpoint_sha256": self.checkpoint_sha256,
            "approved_config_sha256": self.approved_config_sha256,
            "target_config_sha256": self.target_config_sha256,
            "approved_config": json.loads(self.approved_config_json),
            "changed_fields": ["runner_image"],
            "target_image_reference": config.runner_image,
            "validated_runtime_bundle": dict(runtime_bundle or {}),
            "run_input_capsule_sha256": run_input_capsule_sha256,
            "reviewed_plan_sha256": canonical_sha256(plan_payload),
            "decision_set_sha256": checkpoint.consumed_decision_sha256,
            "execution_start_receipt_sha256": checkpoint.execution_start_receipt_sha256,
        }
        digest = canonical_sha256(payload)
        path = Path(run_dir) / f"execution_runtime_revision_{digest}.json"
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        try:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(text)
        except FileExistsError:
            if path.is_symlink() or path.read_text(encoding="utf-8") != text:
                raise HumanReviewCheckpointError("runtime revision receipt changed")
        evidence.register_file(
            kind="log",
            description="Explicit development execution image revision; original approval retained.",
            source_path=path,
            evidence_id=f"execution_runtime_revision_{digest}",
            producer="pipeline",
            generation_mode="system",
            metadata={
                "approved_config_sha256": self.approved_config_sha256,
                "target_config_sha256": self.target_config_sha256,
                "paper_authority": False,
            },
        )


def prepare_execution_runtime_revision(
    *,
    approved_config: PipelineConfig,
    runner_image: str,
    run_dir: Path,
) -> tuple[PipelineConfig, ExecutionRuntimeRevision | None]:
    """Select the host's current image without rebuilding the scientific plan."""
    if not runner_image or approved_config.runner_image == runner_image:
        return approved_config, None
    checkpoint = load_checkpoint(
        human_review_checkpoint_path(run_dir), require_pending=False
    )
    payload = approved_config.recovery_payload()
    updated = PipelineConfig(**{**payload, "runner_image": runner_image})
    revision = ExecutionRuntimeRevision(
        approved_config_json=json.dumps(payload, sort_keys=True),
        approved_config_sha256=approved_config.canonical_digest(),
        target_config_sha256=updated.canonical_digest(),
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        run_id=checkpoint.run_id,
    )
    revision.validate(config=updated, run_dir=run_dir)
    return updated, revision
