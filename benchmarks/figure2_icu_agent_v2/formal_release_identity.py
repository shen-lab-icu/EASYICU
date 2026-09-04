"""Single owner for the Figure 2 formal source and registration identity."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

REGISTERED_SOURCE_PATHS: Mapping[str, Path] = MappingProxyType(
    {
        "validator_sha256": PACKAGE_ROOT / "design_v2_1.py",
        "validator_test_sha256": (
            REPO_ROOT / "tests/benchmarks/figure2_icu_agent_v2/test_design_v2_1.py"
        ),
        "formal_authority_sha256": PACKAGE_ROOT / "formal_authority.py",
        "formal_authority_test_sha256": (
            REPO_ROOT
            / "tests/benchmarks/figure2_icu_agent_v2/test_formal_runtime_v2_1.py"
        ),
        "formal_provider_gate_sha256": PACKAGE_ROOT / "formal_provider_gate.py",
        "provider_hard_stop_client_sha256": (
            REPO_ROOT / "src/easyicu/research_agent/providers/hard_stop.py"
        ),
        "provider_hard_stop_ledger_sha256": (
            REPO_ROOT / "src/easyicu/research_agent/authority/provider_hard_stop.py"
        ),
        "provider_hard_stop_test_sha256": (
            REPO_ROOT
            / "tests/research_agent/providers/test_provider_hard_stop.py"
        ),
        "formal_collaborator_adapter_sha256": (
            PACKAGE_ROOT / "formal_collaborator_adapter.py"
        ),
        "pipeline_services_sha256": (
            REPO_ROOT / "src/easyicu/research_agent/orchestration/services.py"
        ),
        "pipeline_services_test_sha256": (
            REPO_ROOT
            / "tests/research_agent/integration/test_pipeline_config_contract.py"
        ),
        "formal_easyicu_runner_sha256": PACKAGE_ROOT / "formal_easyicu_runner.py",
        "formal_generic_runner_sha256": PACKAGE_ROOT / "formal_generic_runner.py",
        "generic_harness_sha256": PACKAGE_ROOT / "generic_code_agent_harness.py",
        "easyicu_review_adapter_sha256": (
            PACKAGE_ROOT / "easyicu_review_bundle_adapter.py"
        ),
        "review_bundle_writer_sha256": PACKAGE_ROOT / "review_bundle_writer.py",
        "immutable_publication_sha256": (
            PACKAGE_ROOT / "immutable_publication.py"
        ),
        "review_bundle_normalizer_sha256": (
            PACKAGE_ROOT / "review_bundle_normalizer.py"
        ),
        "review_bundle_semantics_sha256": (
            PACKAGE_ROOT / "review_bundle_semantics.py"
        ),
        "formal_scheduler_sha256": PACKAGE_ROOT / "formal_scheduler.py",
        "formal_trajectory_lifecycle_sha256": (
            PACKAGE_ROOT / "formal_trajectory_lifecycle.py"
        ),
        "formal_release_identity_sha256": (
            PACKAGE_ROOT / "formal_release_identity.py"
        ),
        "multi_host_acceptance_sha256": PACKAGE_ROOT / "multi_host_acceptance.py",
        "blinded_evaluator_sha256": PACKAGE_ROOT / "blinded_evaluator.py",
        "formal_implementation_owner_test_sha256": (
            REPO_ROOT
            / "tests/benchmarks/figure2_icu_agent_v2/"
            "test_formal_implementation_owners.py"
        ),
    }
)

IMPLEMENTATION_OWNER_PATHS: Mapping[str, Path] = MappingProxyType(
    {
        "provider_gate": PACKAGE_ROOT / "formal_provider_gate.py",
        "provider_hard_stop_client": (
            REPO_ROOT / "src/easyicu/research_agent/providers/hard_stop.py"
        ),
        "provider_hard_stop_ledger": (
            REPO_ROOT / "src/easyicu/research_agent/authority/provider_hard_stop.py"
        ),
        "formal_collaborator_adapter": (
            PACKAGE_ROOT / "formal_collaborator_adapter.py"
        ),
        "pipeline_services": (
            REPO_ROOT / "src/easyicu/research_agent/orchestration/services.py"
        ),
        "easyicu_formal_runner": PACKAGE_ROOT / "formal_easyicu_runner.py",
        "generic_formal_runner": PACKAGE_ROOT / "formal_generic_runner.py",
        "generic_harness": PACKAGE_ROOT / "generic_code_agent_harness.py",
        "easyicu_review_adapter": PACKAGE_ROOT / "easyicu_review_bundle_adapter.py",
        "review_bundle_writer": PACKAGE_ROOT / "review_bundle_writer.py",
        "immutable_publication": PACKAGE_ROOT / "immutable_publication.py",
        "shared_review_semantics": PACKAGE_ROOT / "review_bundle_semantics.py",
        "review_normalizer": PACKAGE_ROOT / "review_bundle_normalizer.py",
        "formal_scheduler": PACKAGE_ROOT / "formal_scheduler.py",
        "formal_trajectory_lifecycle": (
            PACKAGE_ROOT / "formal_trajectory_lifecycle.py"
        ),
        "formal_release_identity": PACKAGE_ROOT / "formal_release_identity.py",
        "multi_host_acceptance": PACKAGE_ROOT / "multi_host_acceptance.py",
        "blinded_evaluator": PACKAGE_ROOT / "blinded_evaluator.py",
    }
)

REGISTRATION_METADATA_FIELDS = (
    "registry_name",
    "immutable_registration_id",
    "registration_timestamp_utc",
    "embargo_or_public_status",
    "package_sha256",
    "protocol_sha256",
    "design_commit",
    "annotated_tag",
    "registrant_identity",
    "trusted_authority_signer_identity",
    "trusted_authority_ed25519_public_key_base64",
    "amendment_policy_acknowledged",
)


class FormalReleaseIdentityError(ValueError):
    """Registered source identity is missing, unreadable, or mismatched."""

    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}")


def required_registration_fields() -> tuple[str, ...]:
    """Return the complete, deterministic external registration field set."""

    source_fields = tuple(REGISTERED_SOURCE_PATHS)
    metadata_before_sources = REGISTRATION_METADATA_FIELDS[:6]
    metadata_after_sources = REGISTRATION_METADATA_FIELDS[6:]
    return (*metadata_before_sources, *source_fields, *metadata_after_sources)


def registered_source_digests() -> dict[str, str]:
    """Hash every formal source owned by the current release identity."""

    digests: dict[str, str] = {}
    for field, path in REGISTERED_SOURCE_PATHS.items():
        try:
            digests[field] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise FormalReleaseIdentityError(
                "FORMAL_AUTHORITY_REGISTERED_SOURCE_UNREADABLE",
                str(path),
            ) from exc
    return digests


def registered_implementation_owners() -> dict[str, str]:
    """Return repository-relative paths for every formal implementation owner."""

    return {
        owner: str(path.relative_to(REPO_ROOT))
        for owner, path in IMPLEMENTATION_OWNER_PATHS.items()
    }


def validate_registered_source_identity(registration: Mapping[str, Any]) -> None:
    """Match registered source digests to the exact current formal implementation."""

    for field, actual in registered_source_digests().items():
        signed = registration.get(field)
        if not isinstance(signed, str) or not _SHA256_RE.fullmatch(signed):
            raise FormalReleaseIdentityError(
                "FORMAL_AUTHORITY_DIGEST_INVALID",
                f"registration.{field}",
            )
        if signed != actual:
            raise FormalReleaseIdentityError(
                "FORMAL_AUTHORITY_REGISTERED_SOURCE_MISMATCH",
                field,
            )


__all__ = [
    "FormalReleaseIdentityError",
    "IMPLEMENTATION_OWNER_PATHS",
    "REGISTERED_SOURCE_PATHS",
    "registered_implementation_owners",
    "registered_source_digests",
    "required_registration_fields",
    "validate_registered_source_identity",
]
