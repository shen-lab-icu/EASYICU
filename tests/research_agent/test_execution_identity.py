from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from easyicu.research_agent.authority.execution_identity import (
    ExecutionIdentity,
    ExpectedExecutionIdentity,
    execution_identity_for_pipeline,
)
from easyicu.research_agent.providers.factory import (
    ProviderAuthorization,
    build_provider_client,
    provider_authorization_for_configuration,
)


def _provider_authorization(*, model: str = "model-a") -> dict[str, Any]:
    authorization = ProviderAuthorization.create(
        provider="openai",
        model=model,
        base_url="https://provider.example/v1",
        destination="external",
        authorization_mode="operator_env",
    )
    return {
        "schema_version": "easyicu.provider_authorization_manifest/1",
        "clients": [
            {
                "provider": authorization.provider,
                "model": authorization.model,
                "base_url": authorization.base_url,
                "destination": authorization.destination,
                "authorization_mode": authorization.authorization_mode,
                "authorization_sha256": authorization.authorization_sha256,
            }
        ],
    }


def _identity(**overrides: Any) -> ExecutionIdentity:
    coordinates = {
        "submission_profile_name": "profile",
        "submission_profile_version": "v1",
        "runner": "docker",
        "runner_image_digest": "sha256:" + "a" * 64,
        "network_policy": "none",
        "provider_authorization": _provider_authorization(),
        "llm_seed": 17,
        "data_seed": 7,
        "input_authority_sha256": "e" * 64,
        "host_runner_authorized": False,
        "code_version": {"git_sha": "b" * 40, "git_dirty": False},
    }
    coordinates.update(overrides)
    if "provider_client" in overrides:
        coordinates.pop("provider_authorization", None)
    return ExecutionIdentity.create(**coordinates)


def test_every_execution_coordinate_changes_content_identity() -> None:
    baseline = _identity()
    variants = (
        _identity(submission_profile_version="v2"),
        _identity(runner="subprocess"),
        _identity(runner_image_digest="sha256:" + "c" * 64),
        _identity(network_policy="bridge"),
        _identity(provider_authorization=_provider_authorization(model="model-b")),
        _identity(llm_seed=18),
        _identity(data_seed=8),
        _identity(input_authority_sha256="f" * 64),
        _identity(host_runner_authorized=True),
        _identity(code_version={"git_sha": "d" * 40, "git_dirty": False}),
    )
    assert (
        len({baseline.identity_sha256, *(row.identity_sha256 for row in variants)})
        == 11
    )


def test_identity_tampering_is_rejected() -> None:
    payload = _identity().model_dump(mode="json")
    payload["network_policy"] = "bridge"

    with pytest.raises(
        ValidationError,
        match="execution environment identity digest mismatch",
    ):
        ExecutionIdentity.model_validate(payload, strict=True)


def test_expected_identity_freeze_rejects_tampering() -> None:
    frozen = ExpectedExecutionIdentity.create(_identity())
    payload = frozen.model_dump(mode="json")
    payload["expected_identity_sha256"] = "0" * 64

    with pytest.raises(
        ValidationError,
        match="expected execution environment identity digest",
    ):
        ExpectedExecutionIdentity.model_validate(payload, strict=True)


def test_frozen_environment_is_shared_but_reuse_identity_binds_input() -> None:
    first = _identity(data_seed=7, input_authority_sha256="e" * 64)
    second = _identity(data_seed=8, input_authority_sha256="f" * 64)
    frozen = ExpectedExecutionIdentity.create(first)

    assert first.environment_identity_sha256 == second.environment_identity_sha256
    assert first.identity_sha256 != second.identity_sha256
    assert frozen.expected_identity_sha256 == second.environment_identity_sha256


def test_preflight_provider_coordinates_match_the_constructed_client() -> None:
    from easyicu.research_agent.providers.llm import OpenAIClient

    environment = {"OPENAI_BASE_URL": "http://127.0.0.1:8317/v1"}
    client = build_provider_client(
        provider="openai",
        model="local-model",
        request_timeout=60.0,
        title="EasyICU test",
        client_cls=OpenAIClient,
        environment=environment,
    )
    common = {
        "submission_profile_name": "profile",
        "submission_profile_version": "v1",
        "runner": "docker",
        "runner_image_digest": "sha256:" + "a" * 64,
        "network_policy": "none",
        "llm_seed": 17,
        "data_seed": 7,
        "input_authority_sha256": "e" * 64,
        "code_version": {"git_sha": "b" * 40, "git_dirty": False},
    }
    actual = ExecutionIdentity.create(provider_client=client, **common)
    preflight = ExecutionIdentity.create(
        provider_authorization=provider_authorization_for_configuration(
            provider="openai",
            model="local-model",
            environment=environment,
        ),
        **common,
    )

    assert actual.identity_sha256 == preflight.identity_sha256


def test_unmanaged_custom_provider_can_never_receive_paper_authority() -> None:
    class CustomForwarder:
        name = "custom-forwarder"

    identity = _identity(provider_client=CustomForwarder())

    assert identity.provider_authorization["clients"][0] == {
        "provider": "custom-forwarder",
        "model": "",
        "base_url": "",
        "destination": "external",
        "authorization_mode": "unmanaged",
        "authorization_sha256": "",
    }
    assert identity.paper_eligible is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"runner": "subprocess"},
        {"runner_image_digest": None},
        {"network_policy": "bridge"},
        {"host_runner_authorized": True},
        {"code_version": {"git_sha": "b" * 40, "git_dirty": True}},
        {"submission_profile_name": None, "submission_profile_version": None},
        {"input_authority_sha256": None},
    ],
)
def test_paper_eligibility_is_deny_by_default(overrides: dict[str, Any]) -> None:
    assert _identity(**overrides).paper_eligible is False


def test_frozen_environment_cannot_authorize_unbound_input() -> None:
    bound = _identity()
    unbound = _identity(input_authority_sha256=None)
    frozen = ExpectedExecutionIdentity.create(unbound)

    assert frozen.expected_identity_sha256 == unbound.environment_identity_sha256
    assert unbound.paper_eligible is False
    assert unbound.environment_identity_sha256 != bound.environment_identity_sha256
    assert unbound.identity_sha256 != bound.identity_sha256


def _pipeline_with_runtime_image(
    *,
    actual: str,
    expected: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        _execution_identity=None,
        _submission_profile_name="profile",
        _submission_profile_version="v1",
        _runner_kind="docker",
        _expected_runner_image_digest=expected,
        _validated_runtime_bundle={
            "schema": "easyicu.docker_runtime_preflight/2",
            "provenance": {"image_id": actual},
        },
        _runner_network="none",
        _llm=None,
        _llm_seed=None,
        _config=SimpleNamespace(
            execution_data_seed=7,
            execution_input_authority_sha256="e" * 64,
        ),
        _host_runner_authorized=False,
    )


def test_pipeline_identity_binds_the_validated_runtime_image() -> None:
    actual = "sha256:" + "a" * 64

    identity = execution_identity_for_pipeline(
        _pipeline_with_runtime_image(actual=actual)
    )

    assert identity.runner_image_digest == actual


def test_pipeline_identity_rejects_expected_and_validated_image_mismatch() -> None:
    with pytest.raises(ValueError, match="differs from the expected image"):
        execution_identity_for_pipeline(
            _pipeline_with_runtime_image(
                actual="sha256:" + "a" * 64,
                expected="sha256:" + "b" * 64,
            )
        )
