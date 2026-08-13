"""Canonical provider construction and cross-entrypoint security regressions."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

_PROVIDER_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "EASYICU_ALLOW_EXTERNAL_LLM",
    "EASYICU_OPENAI_AUTH_HEADER",
    "EASYICU_TRUST_LOOPBACK_PROXY_KEY",
)


def _mutated_openai_complete(self, *_args, **_kwargs):
    self._attack_callback_calls += 1
    return "must not run"


def _mutated_openai_complete_with_usage(self, *_args, **_kwargs):
    self._attack_callback_calls += 1
    return "must not run", None


def _mutated_openai_rebuild(self):
    from easyicu.research_agent.providers.factory import (
        _refresh_reviewed_transport_dispatch,
    )

    self._client = self._malicious_transport
    _refresh_reviewed_transport_dispatch(self)


def _constructed_local_openai(monkeypatch, *, completions=None, max_retries=0):
    from easyicu.research_agent.providers.llm import OpenAIClient

    transport = SimpleNamespace(
        chat=SimpleNamespace(completions=completions or SimpleNamespace())
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=lambda **_kwargs: transport),
    )
    client = OpenAIClient(
        model="model",
        api_key="non-secret-test-key",
        base_url="http://127.0.0.1:8787/v1",
        request_timeout=1.0,
        max_retries=max_retries,
    )
    return client, transport


@pytest.fixture(autouse=True)
def _clean_provider_environment(monkeypatch):
    for name in _PROVIDER_ENV_KEYS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://localhost:8787/v1", True),
        ("https://LOCALHOST./v1", True),
        ("http://127.0.0.1:8787/v1", True),
        ("http://127.42.0.9/v1", True),
        ("http://[::1]:8787/v1", True),
        ("http://0.0.0.0:8787/v1", False),
        ("http://localhost.example/v1", False),
        ("https://attacker.example/collect?next=localhost", False),
        ("http://localhost@attacker.example/v1", False),
        ("http://attacker.example@127.0.0.1/v1", False),
        ("http://127.0.0.1/v1?forward=https://attacker.example", False),
        ("http://127.0.0.1/v1#attacker", False),
        ("http://127.0.0.1/attacker/v1", False),
        ("ftp://127.0.0.1/v1", False),
        ("http://127.0.0.1:not-a-port/v1", False),
        ("127.0.0.1:8787/v1", False),
        (None, False),
    ],
)
def test_loopback_url_classification_is_parsed_and_strict(ra, base_url, expected):
    from easyicu.research_agent.providers import is_loopback_openai_base_url

    assert is_loopback_openai_base_url(base_url) is expected


def _install_entrypoint_recorders(monkeypatch, ra):
    import easyicu.research_agent.mcp_server as mcp
    import tools.run_discovery_to_manuscript as discovery
    import tools.run_research_agent_bench as benchmark

    seen: dict[str, dict[str, Any]] = {}

    def recorder(label: str):
        class RecordingClient:
            def __init__(self, **kwargs):
                seen[label] = kwargs

        return RecordingClient

    monkeypatch.setattr(mcp, "OpenAIClient", recorder("mcp"))
    monkeypatch.setattr(discovery, "OpenAIClient", recorder("discovery"))
    monkeypatch.setattr(ra, "OpenAIClient", recorder("benchmark"))
    return mcp, discovery, benchmark, seen


def _build_all_three(mcp, discovery, benchmark, *, provider: str):
    model = "openai/gpt-oss-120b:free"
    timeout = 17.0
    mcp_client, mcp_error = mcp._build_run_llm(
        {
            "provider": provider,
            "model": model,
            "request_timeout": timeout,
        }
    )
    assert mcp_error is None
    assert mcp_client is not None
    discovery._build_data_foundation_llm(
        provider=provider,
        model=model,
        request_timeout=timeout,
    )
    benchmark._make_llm(
        provider=provider,
        model=model,
        request_timeout=timeout,
    )


def test_openrouter_contract_is_consistent_across_all_three_entries(ra, monkeypatch):
    mcp, discovery, benchmark, seen = _install_entrypoint_recorders(monkeypatch, ra)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://router.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-openai-secret")
    monkeypatch.setenv("EASYICU_ALLOW_EXTERNAL_LLM", "1")

    _build_all_three(mcp, discovery, benchmark, provider="openrouter")

    expected_titles = {
        "mcp": "EasyICU research-agent MCP",
        "discovery": "EasyICU discovery-to-manuscript",
        "benchmark": "EasyICU research-agent benchmark",
    }
    assert set(seen) == set(expected_titles)
    for entry, kwargs in seen.items():
        assert kwargs["api_key"] == "router-secret"
        assert kwargs["base_url"] == "https://router.example/v1"
        assert kwargs["model"] == "openai/gpt-oss-120b:free"
        assert kwargs["request_timeout"] == 17.0
        assert kwargs["extra_headers"] == {
            "HTTP-Referer": "https://github.com/shen-lab-icu/easyicu",
            "X-Title": expected_titles[entry],
        }


def test_loopback_openai_never_forwards_paid_secrets_from_any_entry(ra, monkeypatch):
    from easyicu.research_agent.providers import LOCAL_OPENAI_DUMMY_API_KEY

    mcp, discovery, benchmark, seen = _install_entrypoint_recorders(monkeypatch, ra)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8787/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "paid-openai-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "paid-openrouter-secret")

    _build_all_three(mcp, discovery, benchmark, provider="openai")

    assert set(seen) == {"mcp", "discovery", "benchmark"}
    for kwargs in seen.values():
        assert kwargs["base_url"] == "http://127.0.0.1:8787/v1"
        assert kwargs["api_key"] == LOCAL_OPENAI_DUMMY_API_KEY
        assert kwargs["api_key"] not in {
            "paid-openai-secret",
            "paid-openrouter-secret",
        }


def test_loopback_opt_in_forwards_real_openai_key_to_trusted_proxy(ra, monkeypatch):
    # A TRUSTED authenticating loopback proxy (e.g. Codex Tools on :8787) that
    # validates the client key needs the real key. With the explicit opt-in set,
    # the factory forwards OPENAI_API_KEY instead of the dummy so the proxy does
    # not 401. This is off by default (see the security test above).
    mcp, discovery, benchmark, seen = _install_entrypoint_recorders(monkeypatch, ra)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8787/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-proxy-key")
    monkeypatch.setenv("EASYICU_TRUST_LOOPBACK_PROXY_KEY", "1")

    _build_all_three(mcp, discovery, benchmark, provider="openai")

    assert set(seen) == {"mcp", "discovery", "benchmark"}
    for kwargs in seen.values():
        assert kwargs["base_url"] == "http://127.0.0.1:8787/v1"
        assert kwargs["api_key"] == "sk-real-proxy-key"


def test_trusted_loopback_x_api_key_contract_reaches_every_entrypoint(ra, monkeypatch):
    mcp, discovery, benchmark, seen = _install_entrypoint_recorders(monkeypatch, ra)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8317/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "local-proxy-key")
    monkeypatch.setenv("EASYICU_TRUST_LOOPBACK_PROXY_KEY", "1")
    monkeypatch.setenv("EASYICU_OPENAI_AUTH_HEADER", "x-api-key")

    _build_all_three(mcp, discovery, benchmark, provider="openai")

    assert set(seen) == {"mcp", "discovery", "benchmark"}
    for kwargs in seen.values():
        assert kwargs["api_key"] == "local-proxy-key"
        assert kwargs["extra_headers"] == {"x-api-key": "local-proxy-key"}


def test_x_api_key_contract_is_rejected_for_external_or_request_owned_endpoint(
    ra, monkeypatch
):
    from easyicu.research_agent.providers.factory import (
        OPENAI_AUTH_HEADER_NOT_AUTHORIZED,
        ProviderConfigurationError,
        build_provider_client,
    )

    class Recorder:
        def __init__(self, **_kwargs):
            raise AssertionError("rejected auth contract must not construct a client")

    external = {
        "OPENAI_BASE_URL": "https://api.example.test/v1",
        "OPENAI_API_KEY": "external-key",
        "EASYICU_ALLOW_EXTERNAL_LLM": "1",
        "EASYICU_TRUST_LOOPBACK_PROXY_KEY": "1",
        "EASYICU_OPENAI_AUTH_HEADER": "x-api-key",
    }
    with pytest.raises(ProviderConfigurationError) as external_error:
        build_provider_client(
            provider="openai",
            model="model",
            request_timeout=1,
            title="test",
            client_cls=Recorder,
            environment=external,
        )
    assert external_error.value.issue == OPENAI_AUTH_HEADER_NOT_AUTHORIZED

    loopback = {
        "OPENAI_API_KEY": "loopback-key",
        "EASYICU_TRUST_LOOPBACK_PROXY_KEY": "1",
        "EASYICU_OPENAI_AUTH_HEADER": "x-api-key",
    }
    with pytest.raises(ProviderConfigurationError) as override_error:
        build_provider_client(
            provider="openai",
            model="model",
            request_timeout=1,
            title="test",
            client_cls=Recorder,
            environment=loopback,
            base_url_override="http://127.0.0.1:9999/v1",
        )
    assert override_error.value.issue == OPENAI_AUTH_HEADER_NOT_AUTHORIZED


def test_loopback_opt_in_does_not_forward_real_key_to_per_request_override(
    ra, monkeypatch
):
    # Regression (security): the opt-in trusts ONE operator-configured proxy
    # (server-owned OPENAI_BASE_URL). An untrusted per-request ``base_url``
    # override can name ANY loopback port -- including a listener a local caller
    # controls -- so it must NEVER receive the real key even with the opt-in set.
    # Otherwise a local caller could steer the paid secret to a port it owns and
    # harvest it from the Authorization header.
    import easyicu.research_agent.mcp_server as mcp
    from easyicu.research_agent.providers import LOCAL_OPENAI_DUMMY_API_KEY

    constructed: list[dict[str, Any]] = []
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-proxy-key")
    monkeypatch.setenv("EASYICU_TRUST_LOOPBACK_PROXY_KEY", "1")
    monkeypatch.setattr(
        mcp, "OpenAIClient", lambda **kwargs: constructed.append(kwargs)
    )

    client, error = mcp._build_run_llm(
        {
            "provider": "openai",
            "model": "model",
            # Attacker-chosen loopback port supplied via the per-request override.
            "base_url": "http://127.0.0.1:9999/v1",
        }
    )

    assert error is None
    assert constructed, "client should still be constructed for a loopback override"
    assert constructed[0]["base_url"] == "http://127.0.0.1:9999/v1"
    assert constructed[0]["api_key"] == LOCAL_OPENAI_DUMMY_API_KEY
    assert constructed[0]["api_key"] != "sk-real-proxy-key"


def test_loopback_opt_in_without_key_still_uses_dummy(ra, monkeypatch):
    # Opt-in set but no real key present (true no-auth vLLM): still the dummy,
    # never an empty/None credential.
    from easyicu.research_agent.providers import LOCAL_OPENAI_DUMMY_API_KEY

    mcp, discovery, benchmark, seen = _install_entrypoint_recorders(monkeypatch, ra)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8787/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EASYICU_TRUST_LOOPBACK_PROXY_KEY", "1")

    _build_all_three(mcp, discovery, benchmark, provider="openai")

    for kwargs in seen.values():
        assert kwargs["api_key"] == LOCAL_OPENAI_DUMMY_API_KEY


def test_benchmark_adaptive_reasoning_is_explicit_and_role_scoped(ra, monkeypatch):
    import tools.run_research_agent_bench as benchmark
    from easyicu.research_agent.providers.factory import (
        provider_authorization_manifest,
    )

    constructed = []

    class _Transport:
        def __init__(self, **kwargs):
            constructed.append(dict(kwargs))
            self.chat = SimpleNamespace(completions=SimpleNamespace())

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=lambda **kwargs: _Transport(**kwargs)),
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8317/v1")

    router = benchmark._make_llm(
        provider="openai",
        model="gpt-5.6-luna",
        request_timeout=17.0,
        reasoning_effort_profile="adaptive_v1",
    )

    assert router.for_role("planner") is router.for_role("coder")
    assert router.for_role("analyzer") is router.for_role("writer")
    assert router.for_role("repair") is not router.for_role("coder")
    assert router.for_role("planner")._extra_body == {"reasoning": {"effort": "medium"}}
    assert router.for_role("analyzer")._extra_body == {"reasoning": {"effort": "low"}}
    assert router.for_role("repair")._extra_body == {"reasoning": {"effort": "high"}}
    assert (
        provider_authorization_manifest(router)["reasoning_effort_profile"]
        == "adaptive_v1"
    )
    expected_identity = benchmark._benchmark_execution_identity(
        {},
        provider="openai",
        model="gpt-5.6-luna",
        reasoning_effort_profile="adaptive_v1",
    )
    actual_identity = benchmark._benchmark_execution_identity({}, router)
    assert actual_identity.identity_sha256 == expected_identity.identity_sha256
    assert len(constructed) == 3


def test_reasoning_extra_body_mutation_is_rejected_before_transport(ra, monkeypatch):
    from easyicu.research_agent.providers.factory import (
        ProviderConfigurationError,
        authorized_complete,
    )
    from easyicu.research_agent.providers.llm import OpenAIClient
    from easyicu.research_agent.providers.protocol import LLMMessage

    calls = []
    completions = SimpleNamespace(create=lambda **kwargs: calls.append(kwargs))
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(
            OpenAI=lambda **_kwargs: SimpleNamespace(
                chat=SimpleNamespace(completions=completions)
            )
        ),
    )
    client = OpenAIClient(
        model="model",
        api_key="non-secret-test-key",
        base_url="http://127.0.0.1:8317/v1",
        request_timeout=1.0,
        max_retries=0,
        extra_body={"reasoning": {"effort": "medium"}},
    )
    client._extra_body["reasoning"]["effort"] = "low"

    with pytest.raises(ProviderConfigurationError):
        authorized_complete(
            client,
            [LLMMessage(role="user", content="must not leave")],
        )
    assert calls == []


@pytest.mark.parametrize(
    ("provider", "available_key", "missing_key"),
    [
        ("openai", "OPENROUTER_API_KEY", "OPENAI_API_KEY"),
        ("openrouter", "OPENAI_API_KEY", "OPENROUTER_API_KEY"),
    ],
)
def test_provider_keys_are_not_interchangeable_across_entrypoints(
    ra,
    monkeypatch,
    provider,
    available_key,
    missing_key,
):
    mcp, discovery, benchmark, seen = _install_entrypoint_recorders(monkeypatch, ra)
    monkeypatch.setenv(available_key, "wrong-provider-secret")
    if provider == "openai":
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    _client, error = mcp._build_run_llm(
        {"provider": provider, "model": "model", "request_timeout": 3.0}
    )
    assert error["error_code"] == "llm_configuration_required"
    assert missing_key in error["error"]

    expected_cli_error = f"{missing_key} is required for --provider {provider}"
    with pytest.raises(SystemExit, match=expected_cli_error):
        discovery._build_data_foundation_llm(
            provider=provider,
            model="model",
            request_timeout=3.0,
        )
    with pytest.raises(SystemExit, match=expected_cli_error):
        benchmark._make_llm(
            provider=provider,
            model="model",
            request_timeout=3.0,
        )
    assert seen == {}


def test_mcp_rejects_wildcard_bind_address_as_per_call_loopback(ra, monkeypatch):
    import easyicu.research_agent.mcp_server as mcp

    constructed = []
    monkeypatch.setenv("OPENAI_API_KEY", "paid-openai-secret")
    monkeypatch.setattr(
        mcp,
        "OpenAIClient",
        lambda **kwargs: constructed.append(kwargs),
    )

    client, error = mcp._build_run_llm(
        {
            "provider": "openai",
            "model": "model",
            "base_url": "http://0.0.0.0:8787/v1",
        }
    )

    assert client is None
    assert error["error_code"] == "llm_configuration_invalid"
    assert "loopback" in error["error"]
    assert constructed == []


def test_external_provider_requires_explicit_operator_authorization(ra, monkeypatch):
    import easyicu.research_agent.mcp_server as mcp

    monkeypatch.setenv("OPENAI_API_KEY", "configured-but-not-authorized")
    client, error = mcp._build_run_llm(
        {"provider": "openai", "model": "gpt-test", "request_timeout": 5}
    )

    assert client is None
    assert error == {
        "error": (
            "configuration_error: external LLM transport is disabled; set "
            "EASYICU_ALLOW_EXTERNAL_LLM=1 only after approving the exact "
            "provider endpoint and outbound data policy"
        ),
        "error_code": "llm_external_transport_not_authorized",
    }


def test_factory_authorization_records_exact_nonsecret_endpoint():
    from easyicu.research_agent.providers.factory import (
        provider_authorization_for_configuration,
    )

    payload = provider_authorization_for_configuration(
        provider="openrouter",
        model="provider/model",
        environment={
            "OPENROUTER_API_KEY": "secret-never-persisted",
            "OPENROUTER_BASE_URL": "https://router.example/v1",
            "EASYICU_ALLOW_EXTERNAL_LLM": "1",
        },
    )

    assert payload["schema_version"] == "easyicu.provider_authorization_manifest/2"
    assert payload["reasoning_effort_profile"] == "provider_default"
    assert payload["clients"] == [
        {
            "provider": "openrouter",
            "model": "provider/model",
            "base_url": "https://router.example/v1",
            "destination": "external",
            "authorization_mode": "operator_env",
            "authorization_sha256": payload["clients"][0]["authorization_sha256"],
        }
    ]
    assert "secret-never-persisted" not in str(payload)


def test_factory_constructor_seam_does_not_authorize_custom_transport():
    from easyicu.research_agent.providers.factory import (
        ProviderConfigurationError,
        authorized_complete,
        build_provider_client,
        provider_authorization_manifest,
    )
    from easyicu.research_agent.providers.protocol import LLMMessage

    class CustomForwarder:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs
            self.calls = 0

        def complete(self, *_args, **_kwargs):
            self.calls += 1
            return "leaked"

    client = build_provider_client(
        provider="openai",
        model="model",
        request_timeout=10,
        title="test",
        client_cls=CustomForwarder,
        environment={"OPENAI_BASE_URL": "http://127.0.0.1:8317/v1"},
    )

    with pytest.raises(ProviderConfigurationError):
        authorized_complete(client, [LLMMessage(role="user", content="secret")])
    assert client.calls == 0
    assert (
        provider_authorization_manifest(client)["clients"][0]["authorization_mode"]
        == "unmanaged"
    )


def test_unknown_provider_is_unmanaged_external_and_never_local_exempt():
    from easyicu.research_agent.providers.factory import (
        provider_authorization_manifest,
        provider_transport_destination,
    )

    class CustomForwarder:
        name = "custom-forwarder"

    client = CustomForwarder()

    assert provider_transport_destination(client) == "external"
    assert provider_authorization_manifest(client) == {
        "schema_version": "easyicu.provider_authorization_manifest/2",
        "reasoning_effort_profile": "provider_default",
        "clients": [
            {
                "provider": "custom-forwarder",
                "model": "",
                "base_url": "",
                "destination": "external",
                "authorization_mode": "unmanaged",
                "authorization_sha256": "",
            }
        ],
    }


def test_unmanaged_custom_planner_is_rejected_before_complete(ra, monkeypatch):
    from easyicu.research_agent.agents import PlannerAgent
    from easyicu.research_agent.providers.factory import (
        EXTERNAL_LLM_NOT_AUTHORIZED,
        ProviderConfigurationError,
    )

    class CustomForwarder:
        name = "custom-forwarder"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("an unmanaged provider must receive no prompt")

    # The operator opt-in authorizes factory-bound external endpoints; it must
    # not bless an unmanaged custom adapter with no endpoint authority.
    monkeypatch.setenv("EASYICU_ALLOW_EXTERNAL_LLM", "1")
    client = CustomForwarder()
    context = ra.ResearchContext(
        research_question="Test provider authorization.",
        cohort=ra.CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
    )
    with pytest.raises(ProviderConfigurationError) as exc_info:
        PlannerAgent(client).run(context)

    assert exc_info.value.issue == EXTERNAL_LLM_NOT_AUTHORIZED
    assert client.calls == 0


def test_forged_mock_marker_never_authorizes_a_custom_client():
    from easyicu.research_agent.providers.factory import (
        ProviderConfigurationError,
        authorized_complete,
    )
    from easyicu.research_agent.providers.protocol import LLMMessage

    class ForgedMock:
        __easyicu_mock_client__ = True

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, *_args, **_kwargs):
            self.calls += 1
            return "leaked"

    client = ForgedMock()
    with pytest.raises(ProviderConfigurationError):
        authorized_complete(client, [LLMMessage(role="user", content="secret")])
    assert client.calls == 0


def test_custom_client_cannot_self_register_as_offline():
    from easyicu.research_agent.providers.factory import (
        EXTERNAL_LLM_NOT_AUTHORIZED,
        ProviderConfigurationError,
        register_offline_test_client,
    )

    class CustomClient:
        pass

    with pytest.raises(ProviderConfigurationError) as exc_info:
        register_offline_test_client(CustomClient())

    assert exc_info.value.issue == EXTERNAL_LLM_NOT_AUTHORIZED


def test_registered_mock_instance_method_override_is_rejected_before_callback():
    from easyicu.research_agent.providers.factory import (
        EXTERNAL_LLM_NOT_AUTHORIZED,
        ProviderConfigurationError,
        authorized_complete,
    )
    from easyicu.research_agent.providers.mocks import MockLLMClient
    from easyicu.research_agent.providers.protocol import LLMMessage

    client = MockLLMClient()
    calls = 0

    def custom_callback(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return "must not run"

    client.complete = custom_callback
    with pytest.raises(ProviderConfigurationError) as exc_info:
        authorized_complete(
            client,
            [LLMMessage(role="user", content="must remain local")],
        )

    assert exc_info.value.issue == EXTERNAL_LLM_NOT_AUTHORIZED
    assert calls == 0


def test_registered_mock_class_method_mutation_is_rejected_before_callback(
    monkeypatch,
):
    from easyicu.research_agent.providers.factory import (
        EXTERNAL_LLM_NOT_AUTHORIZED,
        ProviderConfigurationError,
        authorized_complete,
    )
    from easyicu.research_agent.providers.mocks import MockLLMClient
    from easyicu.research_agent.providers.protocol import LLMMessage

    client = MockLLMClient()
    calls = 0

    def custom_callback(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return "must not run"

    monkeypatch.setattr(MockLLMClient, "complete", custom_callback)
    with pytest.raises(ProviderConfigurationError) as exc_info:
        authorized_complete(
            client,
            [LLMMessage(role="user", content="must remain local")],
        )

    assert exc_info.value.issue == EXTERNAL_LLM_NOT_AUTHORIZED
    assert calls == 0


def test_registered_vision_mock_image_method_override_is_rejected_before_callback():
    from easyicu.research_agent.providers.factory import (
        EXTERNAL_LLM_NOT_AUTHORIZED,
        ProviderConfigurationError,
        authorized_complete_with_images,
    )
    from easyicu.research_agent.providers.mocks import ScriptedVisionMockLLMClient

    client = ScriptedVisionMockLLMClient(["unused"])
    calls = 0

    def custom_callback(**_kwargs):
        nonlocal calls
        calls += 1
        return "must not run"

    client.complete_with_images = custom_callback
    with pytest.raises(ProviderConfigurationError) as exc_info:
        authorized_complete_with_images(
            client,
            messages=[],
            images=[],
        )

    assert exc_info.value.issue == EXTERNAL_LLM_NOT_AUTHORIZED
    assert calls == 0


def test_unconstructed_exact_openai_object_cannot_gain_local_authority():
    from easyicu.research_agent.providers.factory import (
        EXTERNAL_LLM_NOT_AUTHORIZED,
        ProviderConfigurationError,
        authorize_provider_client,
    )
    from easyicu.research_agent.providers.llm import OpenAIClient

    client = object.__new__(OpenAIClient)
    client._model = "model"
    client._resolved_base_url = "http://127.0.0.1:8787/v1"
    calls = 0

    def custom_callback(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return "must not run"

    client.complete = custom_callback
    with pytest.raises(ProviderConfigurationError) as exc_info:
        authorize_provider_client(
            client,
            provider="openai",
            model="model",
            base_url="http://127.0.0.1:8787/v1",
            destination="local",
            environment={},
        )

    assert exc_info.value.issue == EXTERNAL_LLM_NOT_AUTHORIZED
    assert calls == 0


def test_authorized_openai_rejects_replaced_transport_before_callback(monkeypatch):
    from easyicu.research_agent.providers.factory import (
        EXTERNAL_LLM_NOT_AUTHORIZED,
        ProviderConfigurationError,
        authorized_complete,
    )
    from easyicu.research_agent.providers.protocol import LLMMessage

    client, _transport = _constructed_local_openai(monkeypatch)
    calls = 0

    class _MaliciousCompletions:
        def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            return SimpleNamespace()

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=_MaliciousCompletions())
    )
    with pytest.raises(ProviderConfigurationError) as exc_info:
        authorized_complete(client, [LLMMessage(role="user", content="secret")])

    assert exc_info.value.issue == EXTERNAL_LLM_NOT_AUTHORIZED
    assert calls == 0


def test_authorized_openai_rejects_getattribute_dispatch_mutation_before_callback(
    monkeypatch,
):
    from easyicu.research_agent.providers.factory import (
        EXTERNAL_LLM_NOT_AUTHORIZED,
        ProviderConfigurationError,
        authorized_complete,
    )
    from easyicu.research_agent.providers.llm import OpenAIClient
    from easyicu.research_agent.providers.protocol import LLMMessage

    client, _transport = _constructed_local_openai(monkeypatch)
    calls = 0
    original = OpenAIClient.__getattribute__

    def malicious_getattribute(self, name):
        if name == "complete":

            def callback(*_args, **_kwargs):
                nonlocal calls
                calls += 1
                return "must not run"

            return callback
        return original(self, name)

    monkeypatch.setattr(OpenAIClient, "__getattribute__", malicious_getattribute)
    with pytest.raises(ProviderConfigurationError) as exc_info:
        authorized_complete(client, [LLMMessage(role="user", content="secret")])

    assert exc_info.value.issue == EXTERNAL_LLM_NOT_AUTHORIZED
    assert calls == 0


def test_authorized_openai_rejects_in_place_complete_code_mutation(monkeypatch):
    from easyicu.research_agent.providers.factory import (
        EXTERNAL_LLM_NOT_AUTHORIZED,
        ProviderConfigurationError,
        authorized_complete,
    )
    from easyicu.research_agent.providers.llm import OpenAIClient
    from easyicu.research_agent.providers.protocol import LLMMessage

    client, _transport = _constructed_local_openai(monkeypatch)
    client._attack_callback_calls = 0
    original_code = OpenAIClient.complete.__code__
    try:
        OpenAIClient.complete.__code__ = _mutated_openai_complete.__code__
        with pytest.raises(ProviderConfigurationError) as exc_info:
            authorized_complete(client, [LLMMessage(role="user", content="secret")])
    finally:
        OpenAIClient.complete.__code__ = original_code

    assert exc_info.value.issue == EXTERNAL_LLM_NOT_AUTHORIZED
    assert client._attack_callback_calls == 0


def test_authorized_openai_rejects_in_place_complete_with_usage_code_mutation(
    monkeypatch,
):
    from easyicu.research_agent.providers.factory import (
        EXTERNAL_LLM_NOT_AUTHORIZED,
        ProviderConfigurationError,
        authorized_complete,
    )
    from easyicu.research_agent.providers.llm import OpenAIClient
    from easyicu.research_agent.providers.protocol import LLMMessage

    client, _transport = _constructed_local_openai(monkeypatch)
    client._attack_callback_calls = 0
    original_code = OpenAIClient.complete_with_usage.__code__
    try:
        OpenAIClient.complete_with_usage.__code__ = (
            _mutated_openai_complete_with_usage.__code__
        )
        with pytest.raises(ProviderConfigurationError) as exc_info:
            authorized_complete(client, [LLMMessage(role="user", content="secret")])
    finally:
        OpenAIClient.complete_with_usage.__code__ = original_code

    assert exc_info.value.issue == EXTERNAL_LLM_NOT_AUTHORIZED
    assert client._attack_callback_calls == 0


def test_malicious_rebuild_cannot_refresh_transport_authority(monkeypatch):
    from easyicu.research_agent.providers.factory import (
        EXTERNAL_LLM_NOT_AUTHORIZED,
        ProviderConfigurationError,
        authorized_complete,
    )
    from easyicu.research_agent.providers.llm import OpenAIClient
    from easyicu.research_agent.providers.protocol import LLMMessage

    client, _transport = _constructed_local_openai(monkeypatch)
    calls = 0

    class _MaliciousCompletions:
        def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="must not run"),
                    )
                ],
                usage=None,
            )

    client._malicious_transport = SimpleNamespace(
        chat=SimpleNamespace(completions=_MaliciousCompletions())
    )
    original_code = OpenAIClient._rebuild_openai_client.__code__
    try:
        OpenAIClient._rebuild_openai_client.__code__ = _mutated_openai_rebuild.__code__
        with pytest.raises(ProviderConfigurationError) as refresh_exc:
            client._rebuild_openai_client()
    finally:
        OpenAIClient._rebuild_openai_client.__code__ = original_code

    assert refresh_exc.value.issue == EXTERNAL_LLM_NOT_AUTHORIZED
    with pytest.raises(ProviderConfigurationError) as delivery_exc:
        authorized_complete(client, [LLMMessage(role="user", content="secret")])
    assert delivery_exc.value.issue == EXTERNAL_LLM_NOT_AUTHORIZED
    assert calls == 0


def test_reviewed_openai_rebuild_can_rotate_transport(monkeypatch):
    from easyicu.research_agent.providers.factory import authorized_complete
    from easyicu.research_agent.providers.protocol import LLMMessage

    client, original_transport = _constructed_local_openai(monkeypatch)
    calls = 0

    class _ReviewedCompletions:
        def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="reviewed rebuild"),
                    )
                ],
                usage=None,
            )

    replacement = SimpleNamespace(
        chat=SimpleNamespace(completions=_ReviewedCompletions())
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=lambda **_kwargs: replacement),
    )

    client._rebuild_openai_client()

    assert client._client is replacement
    assert client._client is not original_transport
    assert (
        authorized_complete(client, [LLMMessage(role="user", content="safe")])
        == "reviewed rebuild"
    )
    assert calls == 1


def test_remote_openai_transport_cannot_be_authorized_as_local():
    from easyicu.research_agent.providers.factory import (
        EXTERNAL_LLM_NOT_AUTHORIZED,
        ProviderConfigurationError,
        authorize_provider_client,
    )
    from easyicu.research_agent.providers.llm import OpenAIClient

    client = object.__new__(OpenAIClient)
    client._model = "model"
    client._resolved_base_url = "https://attacker.example/v1"

    with pytest.raises(ProviderConfigurationError) as exc_info:
        authorize_provider_client(
            client,
            provider="openai",
            model="model",
            base_url="https://attacker.example/v1",
            destination="local",
            environment={},
        )

    assert exc_info.value.issue == EXTERNAL_LLM_NOT_AUTHORIZED


@pytest.mark.parametrize(
    ("attribute", "mutated_value"),
    [
        ("_resolved_base_url", "https://attacker.example/v1"),
        ("_model", "mutated-model"),
        ("_completion_token_parameter", "max_completion_tokens"),
    ],
)
def test_registered_openai_transport_mutation_is_rejected_before_delivery(
    attribute: str,
    mutated_value: str,
) -> None:
    from easyicu.research_agent.providers.factory import (
        EXTERNAL_LLM_NOT_AUTHORIZED,
        ProviderConfigurationError,
        authorized_complete,
        build_provider_client,
    )
    from easyicu.research_agent.providers.llm import OpenAIClient
    from easyicu.research_agent.providers.protocol import LLMMessage

    client = build_provider_client(
        provider="openai",
        model="model",
        request_timeout=1.0,
        title="provider mutation test",
        client_cls=OpenAIClient,
        environment={"OPENAI_BASE_URL": "http://127.0.0.1:8787/v1"},
    )
    setattr(client, attribute, mutated_value)

    with pytest.raises(ProviderConfigurationError) as exc_info:
        authorized_complete(
            client,
            [LLMMessage(role="user", content="must remain local")],
        )

    assert exc_info.value.issue == EXTERNAL_LLM_NOT_AUTHORIZED


def test_forged_provider_authorization_attribute_never_authorizes_client():
    from easyicu.research_agent.providers.factory import (
        ProviderAuthorization,
        ProviderConfigurationError,
        authorized_complete,
    )
    from easyicu.research_agent.providers.protocol import LLMMessage

    class ForgedTransport:
        def __init__(self) -> None:
            self.calls = 0
            self.__easyicu_provider_authorization__ = ProviderAuthorization.create(
                provider="openai",
                model="forged",
                base_url="http://127.0.0.1:8787/v1",
                destination="local",
                authorization_mode="local_exempt",
            )

        def complete(self, *_args, **_kwargs):
            self.calls += 1
            return "leaked"

    client = ForgedTransport()
    with pytest.raises(ProviderConfigurationError):
        authorized_complete(client, [LLMMessage(role="user", content="secret")])
    assert client.calls == 0


@pytest.mark.parametrize("surface", ["inner", "clients", "iterator"])
def test_unregistered_top_level_wrapper_cannot_inherit_child_trust(surface: str):
    from easyicu.research_agent.providers.factory import (
        ProviderConfigurationError,
        authorized_complete,
    )
    from easyicu.research_agent.providers.mocks import MockLLMClient
    from easyicu.research_agent.providers.protocol import LLMMessage

    child = MockLLMClient()

    class MaliciousWrapper:
        def __init__(self) -> None:
            self.calls = 0
            if surface == "inner":
                self._inner = child
            elif surface == "clients":
                self._clients = [child]

        def iter_clients(self):
            return iter([child]) if surface == "iterator" else iter([])

        def complete(self, *_args, **_kwargs):
            self.calls += 1
            return "leaked"

    wrapper = MaliciousWrapper()
    with pytest.raises(ProviderConfigurationError):
        authorized_complete(wrapper, [LLMMessage(role="user", content="secret")])
    assert wrapper.calls == 0


def test_registered_wrapper_rejects_mutated_child_graph_before_delivery():
    from easyicu.research_agent.providers.factory import (
        ProviderConfigurationError,
        authorized_complete,
    )
    from easyicu.research_agent.providers.llm import FallbackLLMClient
    from easyicu.research_agent.providers.mocks import MockLLMClient
    from easyicu.research_agent.providers.protocol import LLMMessage

    trusted = MockLLMClient()
    trusted_calls = 0
    original_complete = trusted.complete

    def tracked_complete(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal trusted_calls
        trusted_calls += 1
        return original_complete(*args, **kwargs)

    trusted.complete = tracked_complete

    class UnmanagedChild:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, *_args, **_kwargs):
            self.calls += 1
            return "leaked"

    unmanaged = UnmanagedChild()
    wrapper = FallbackLLMClient(trusted)
    wrapper._clients.append(unmanaged)

    with pytest.raises(ProviderConfigurationError):
        authorized_complete(wrapper, [LLMMessage(role="user", content="secret")])
    assert unmanaged.calls == 0
    assert trusted_calls == 0


@pytest.mark.parametrize("wrapper_kind", ["fallback", "router"])
def test_unmanaged_provider_graph_child_is_rejected_before_any_call(
    wrapper_kind: str,
) -> None:
    from easyicu.research_agent.providers.factory import (
        EXTERNAL_LLM_NOT_AUTHORIZED,
        ProviderConfigurationError,
        authorized_complete,
    )
    from easyicu.research_agent.providers.llm import FallbackLLMClient, LLMRouter
    from easyicu.research_agent.providers.mocks import MockLLMClient
    from easyicu.research_agent.providers.protocol import LLMMessage

    class CustomForwarder:
        name = "custom-forwarder"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, *_args, **_kwargs):
            self.calls += 1
            return "must-not-run"

    custom = CustomForwarder()
    mock = MockLLMClient()
    client = (
        FallbackLLMClient(mock, custom)
        if wrapper_kind == "fallback"
        else LLMRouter(default=mock, planner=custom)
    )
    with pytest.raises(ProviderConfigurationError) as exc_info:
        authorized_complete(
            client,
            [LLMMessage(role="user", content="must not leave host")],
        )

    assert exc_info.value.issue == EXTERNAL_LLM_NOT_AUTHORIZED
    assert custom.calls == 0
