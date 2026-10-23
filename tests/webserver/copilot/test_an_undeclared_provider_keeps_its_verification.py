"""An undeclared provider keeps the verification it had before declarations.

Adding the data-handling declaration keys must not invalidate every verification
receipt written before them: hosted users cannot re-verify, so an upgraded
instance would lose its model. A declaration is a different authority and does
need a fresh verification.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from easyicu.research_agent.providers.client_trust import DataHandling, InferenceLocation
from easyicu.webserver.pi_copilot.provider_config import PiProviderConfig


def _config(**declaration: object) -> PiProviderConfig:
    return PiProviderConfig(
        provider="openai-compatible",
        api_key="sk-test-not-a-secret",
        base_url="https://models.example.test/v1",
        model="example-model",
        api_transport="openai-completions",
        **declaration,
    )


def _fingerprint_before_declarations(config: PiProviderConfig) -> str:
    """The fingerprint a verification receipt carried before the declaration keys."""

    environment = {
        "EASYICU_PI_PROVIDER": config.provider,
        "EASYICU_PI_API_KEY": config.api_key,
        "EASYICU_PI_BASE_URL": config.base_url,
        "EASYICU_PI_MODEL": config.model,
        "EASYICU_PI_API": config.api_transport,
    }
    canonical = json.dumps(
        {
            "schema": "easyicu.pi-provider-config-fingerprint/2",
            "environment": environment,
            "openai_auth_header": config.openai_auth_header.value,
            "verification_contract": "model-catalog+inference-probe/1",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_an_undeclared_connection_matches_its_earlier_receipt() -> None:
    config = _config()
    assert config.fingerprint() == _fingerprint_before_declarations(config)


@pytest.mark.parametrize(
    "declaration",
    [
        {"data_handling": DataHandling.ZERO_RETENTION_DECLARED},
        {"data_handling": DataHandling.ON_DEVICE},
        {"inference_location": InferenceLocation.ON_DEVICE},
    ],
)
def test_a_declaration_needs_a_new_verification(declaration: dict[str, object]) -> None:
    undeclared = _config()
    declared = _config(**declaration)
    assert declared.fingerprint() != undeclared.fingerprint()
    assert declared.fingerprint() != _fingerprint_before_declarations(undeclared)


def test_withdrawing_a_declaration_restores_the_earlier_receipt() -> None:
    declared = _config(data_handling=DataHandling.ZERO_RETENTION_DECLARED)
    withdrawn = _config(data_handling=DataHandling.UNSPECIFIED)
    assert declared.fingerprint() != withdrawn.fingerprint()
    assert withdrawn.fingerprint() == _fingerprint_before_declarations(withdrawn)


def test_zero_retention_and_on_device_are_distinct_authorities() -> None:
    zero_retention = _config(data_handling=DataHandling.ZERO_RETENTION_DECLARED)
    on_device = _config(data_handling=DataHandling.ON_DEVICE)
    assert zero_retention.fingerprint() != on_device.fingerprint()
