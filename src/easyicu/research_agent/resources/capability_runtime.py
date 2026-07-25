"""Production orchestration for reviewable analytical capabilities.

This module owns the request -> human approval -> rebuilt image transition.
It never installs packages and never calls a provider.  The main pipeline only
asks it for a run-identity coordinate and a pre-Planner decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from ..authority.evidence_store import EvidenceStore
from ..orchestration.profiles import require_profile_capability_workflow_setting
from ..schema import ValidationFinding
from .capability import (
    CapabilityActivation,
    CapabilityApproval,
    CapabilityRequest,
    runtime_snapshot_sha256,
    verify_capability_activation,
    write_capability_activation,
    write_capability_approval,
    write_capability_request,
)
from .schema import ResourceDescriptor


@dataclass
class CapabilityWorkflowRuntime:
    """Profile-bound capability state for one pipeline construction."""

    enabled: bool
    profile_ref: str
    expected_image_digest: str | None
    request: CapabilityRequest | None
    approval: CapabilityApproval | None
    activation: CapabilityActivation | None
    approved_resources: tuple[ResourceDescriptor, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        enabled: bool,
        profile_name: str | None,
        profile_version: str | None,
        expected_image_digest: str | None,
        request: Optional[Mapping[str, object]],
        approval: Optional[Mapping[str, object]],
        activation: Optional[Mapping[str, object]],
    ) -> "CapabilityWorkflowRuntime":
        require_profile_capability_workflow_setting(
            name=profile_name,
            version=profile_version,
            enabled=bool(enabled),
            expected_runner_image_digest=expected_image_digest,
        )
        parsed_request = (
            CapabilityRequest.model_validate(request) if request is not None else None
        )
        parsed_approval = (
            CapabilityApproval.model_validate(approval)
            if approval is not None
            else None
        )
        parsed_activation = (
            CapabilityActivation.model_validate(activation)
            if activation is not None
            else None
        )
        coordinates = (parsed_request, parsed_approval, parsed_activation)
        if not enabled and any(value is not None for value in coordinates):
            raise ValueError(
                "Capability coordinates require the workflow to be enabled"
            )
        if parsed_activation is not None and (
            parsed_request is None or parsed_approval is None
        ):
            raise ValueError("Capability activation requires request and approval")
        if parsed_approval is not None and parsed_activation is None:
            raise ValueError("Capability approval requires a new-run activation")
        return cls(
            enabled=bool(enabled),
            profile_ref=f"{profile_name}/{profile_version}",
            expected_image_digest=expected_image_digest,
            request=parsed_request,
            approval=parsed_approval,
            activation=parsed_activation,
        )

    def scientific_coordinate(self) -> dict[str, object] | None:
        request = self.request
        if request is None:
            return None
        coordinate: dict[str, object] = {
            "schema": "easyicu.capability_run_coordinate/1",
            "request_id": request.request_id,
            "request_sha256": request.sha256,
            "runtime_install_allowed": False,
        }
        if self.approval is not None:
            coordinate["approval_sha256"] = self.approval.sha256
        if self.activation is None:
            coordinate["status"] = "review_required"
            return coordinate
        coordinate.update(
            {
                "activation_id": self.activation.activation_id,
                "activation_sha256": self.activation.sha256,
                "target_profile_ref": self.activation.target_profile_ref,
                "image_digest": self.activation.image_digest,
                "new_run_required": True,
            }
        )
        return coordinate

    def prepare(
        self,
        *,
        run_dir: Path,
        evidence: EvidenceStore,
        runtime_import_names: tuple[str, ...],
        runtime_bundle: Mapping[str, object] | None,
        is_resume: bool,
    ) -> ValidationFinding | None:
        """Persist a request or validate activation before Planner spend."""

        request = self.request
        if not self.enabled or request is None:
            return None
        request_path = Path(run_dir) / "capability" / "request.json"
        write_capability_request(request_path, request)
        self._register(
            evidence=evidence,
            evidence_id="capability_request",
            path=request_path,
            description="Digest-bound, non-executable analytical capability request.",
        )
        if self.activation is None or self.approval is None:
            if request.import_name in set(runtime_import_names):
                raise ValueError(
                    "Installed software must be registered instead of entering "
                    "the missing-capability review path"
                )
            if request.runtime_snapshot_sha256 != runtime_snapshot_sha256(
                runtime_import_names
            ):
                raise ValueError(
                    "Capability request was built against a different runtime snapshot"
                )
            return ValidationFinding(
                validator="capability_workflow",
                severity="error",
                message=(
                    "Required analytical software is unavailable. The host saved "
                    "a reviewable request and stopped before Planner/Coder; runtime "
                    "installation and package guessing are forbidden."
                ),
                evidence_ids=["capability_request"],
                detail={
                    "reason": "capability_review_required",
                    "request_id": request.request_id,
                    "import_name": request.import_name,
                    "runtime_install_allowed": False,
                    "provider_calls": 0,
                },
            )
        actual_image = str((runtime_bundle or {}).get("image_id") or "") or None
        resource = verify_capability_activation(
            request=request,
            approval=self.approval,
            activation=self.activation,
            current_profile_ref=self.profile_ref,
            expected_image_digest=self.expected_image_digest,
            actual_image_digest=actual_image,
            runtime_import_names=runtime_import_names,
            is_resume=is_resume,
        )
        approval_path = Path(run_dir) / "capability" / "approval.json"
        activation_path = Path(run_dir) / "capability" / "activation.json"
        write_capability_approval(approval_path, self.approval)
        write_capability_activation(activation_path, self.activation)
        self._register(
            evidence=evidence,
            evidence_id="capability_approval",
            path=approval_path,
            description="Human approval bound to validation tests and immutable image.",
        )
        self._register(
            evidence=evidence,
            evidence_id="capability_activation",
            path=activation_path,
            description="New-run activation bound to profile and image digest.",
        )
        self.approved_resources = (resource,)
        return None

    @staticmethod
    def _register(
        *,
        evidence: EvidenceStore,
        evidence_id: str,
        path: Path,
        description: str,
    ) -> None:
        if evidence.get(evidence_id) is not None:
            return
        evidence.register_file(
            kind="log",
            description=description,
            source_path=path,
            evidence_id=evidence_id,
            producer="capability_workflow",
            generation_mode="system",
        )


__all__ = ["CapabilityWorkflowRuntime"]
