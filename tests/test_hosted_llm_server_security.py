from __future__ import annotations

import ipaddress
import json

import pytest
import requests
from fastapi import Request
from fastapi.testclient import TestClient

from easyicu import hosted_llm_server as hosted


def _upstream_response(status: int, payload: dict) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.headers["Content-Type"] = "application/json"
    response._content = json.dumps(payload).encode("utf-8")  # noqa: SLF001
    response.url = "https://upstream.example/v1/chat/completions"
    return response


def _request(*, peer: str, forwarded_for: str | None = None) -> Request:
    headers = []
    if forwarded_for:
        headers.append((b"x-forwarded-for", forwarded_for.encode("ascii")))
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/",
            "raw_path": b"/",
            "query_string": b"",
            "headers": headers,
            "client": (peer, 50000),
            "server": ("testserver", 80),
        }
    )


def test_hosted_relay_defaults_to_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EASYICU_HOSTED_HOST", raising=False)

    args = hosted.build_parser().parse_args([])

    assert args.host == "127.0.0.1"


def test_hosted_relay_fails_closed_without_auth_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hosted, "OPENROUTER_API_KEY", "test-upstream-key")
    monkeypatch.setattr(hosted, "HOSTED_SERVER_TOKEN", "")
    monkeypatch.setattr(hosted, "HOSTED_ALLOW_UNAUTHENTICATED_LOCAL", False)

    response = TestClient(hosted.app).post(
        "/v1/chat/completions",
        json={"model": "hosted-default", "messages": []},
    )

    assert response.status_code == 503
    assert "authentication is not configured" in response.json()["detail"]


def test_hosted_relay_requires_token_and_rejects_unlisted_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hosted, "OPENROUTER_API_KEY", "test-upstream-key")
    monkeypatch.setattr(hosted, "HOSTED_SERVER_TOKEN", "shared-test-token")
    monkeypatch.setattr(hosted, "HOSTED_ALLOWED_MODELS", set())
    client = TestClient(hosted.app)

    unauthorized = client.post(
        "/v1/chat/completions",
        json={"model": "hosted-default", "messages": []},
    )
    unlisted = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer shared-test-token"},
        json={"model": "attacker/expensive-model", "messages": []},
    )

    assert unauthorized.status_code == 401
    assert unlisted.status_code == 400
    assert unlisted.json()["detail"] == "Requested model is not allowed."


def test_hosted_relay_reports_the_actual_model_after_a_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful fallback must be visible to the research-agent client."""

    monkeypatch.setattr(hosted, "OPENROUTER_API_KEY", "test-upstream-key")
    monkeypatch.setattr(hosted, "HOSTED_SERVER_TOKEN", "shared-test-token")
    monkeypatch.setattr(hosted, "HOSTED_FALLBACK_MODELS", ("fallback-model",))
    monkeypatch.setattr(
        hosted,
        "MODEL_ALIASES",
        {"hosted-default": "configured-model"},
    )
    attempted_models: list[str] = []

    def fake_post(_url, *, json, **_kwargs):  # noqa: ANN001
        attempted_models.append(json["model"])
        if len(attempted_models) == 1:
            return _upstream_response(
                503,
                {"error": {"message": "upstream rate limited"}},
            )
        return _upstream_response(
            200,
            {
                "model": "provider/served-model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "OK"},
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        )

    monkeypatch.setattr(hosted.requests, "post", fake_post)
    response = TestClient(hosted.app).post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer shared-test-token"},
        json={"model": "hosted-default", "messages": []},
    )

    assert response.status_code == 200
    assert attempted_models == ["configured-model", "fallback-model"]
    payload = response.json()
    assert payload["model"] == "provider/served-model"
    assert payload["easyicu_model_provenance"] == {
        "schema_version": "easyicu.hosted_model_provenance/1",
        "requested_model": "configured-model",
        "attempted_model": "fallback-model",
        "upstream_reported_model": "provider/served-model",
        "fallback_used": True,
    }


def test_hosted_relay_ignores_forwarded_ip_without_trusted_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    req = _request(peer="198.51.100.7", forwarded_for="203.0.113.8")
    monkeypatch.setattr(hosted, "HOSTED_TRUSTED_PROXY_NETWORKS", ())

    assert hosted._client_ip(req) == "198.51.100.7"

    monkeypatch.setattr(
        hosted,
        "HOSTED_TRUSTED_PROXY_NETWORKS",
        (ipaddress.ip_network("198.51.100.7/32"),),
    )
    assert hosted._client_ip(req) == "203.0.113.8"


def test_hosted_relay_strips_trusted_proxy_chain_from_right(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hosted,
        "HOSTED_TRUSTED_PROXY_NETWORKS",
        (ipaddress.ip_network("10.0.0.0/8"),),
    )
    request_with_forged_prefix = _request(
        peer="10.0.0.1",
        forwarded_for="198.51.100.99, 203.0.113.50, 10.0.0.2",
    )
    all_trusted = _request(
        peer="10.0.0.1",
        forwarded_for="10.0.0.3, 10.0.0.2",
    )

    assert hosted._client_ip(request_with_forged_prefix) == "203.0.113.50"
    assert hosted._client_ip(all_trusted) == "10.0.0.3"


def test_hosted_relay_wildcards_require_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hosted, "HOSTED_SERVER_TOKEN", "token")
    monkeypatch.setattr(hosted, "_RAW_ALLOWED_ORIGINS", ["*"])
    monkeypatch.setattr(hosted, "HOSTED_ALLOW_WILDCARD_ORIGIN", False)

    with pytest.raises(RuntimeError, match="Wildcard CORS"):
        hosted._validate_security_configuration("127.0.0.1")


def test_hosted_relay_unauthed_development_is_loopback_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hosted, "HOSTED_SERVER_TOKEN", "")
    monkeypatch.setattr(hosted, "HOSTED_ALLOW_UNAUTHENTICATED_LOCAL", True)
    monkeypatch.setattr(hosted, "_RAW_ALLOWED_ORIGINS", [])
    monkeypatch.setattr(hosted, "_RAW_ALLOWED_HOSTS", [])

    hosted._validate_security_configuration("127.0.0.1")
    with pytest.raises(RuntimeError, match="SERVER_TOKEN"):
        hosted._validate_security_configuration("0.0.0.0")


def test_hosted_relay_rejects_untrusted_host() -> None:
    response = TestClient(hosted.app, base_url="http://relay.attacker.test").get(
        "/health"
    )

    assert response.status_code == 400


def test_hosted_relay_accepts_bracketed_ipv6_loopback_host() -> None:
    response = TestClient(
        hosted.app,
        client=("::1", 50000),
    ).get("/health", headers={"Host": "[::1]:8787"})

    assert response.status_code == 200
