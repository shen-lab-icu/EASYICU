"""Shared local-host security contract for credential-bearing provider URLs."""

from __future__ import annotations

from urllib.parse import urlsplit

from easyicu.outbound_url_security import (
    OutboundUrlSecurityError,
    validate_outbound_http_endpoint,
)

_TRUSTED_PROVIDER_HOSTNAMES = frozenset(
    {
        "api.anthropic.com",
        "api.openai.com",
        "generativelanguage.googleapis.com",
    }
)

# D-P2-4: RFC 2606 documentation hosts. The custom-provider preset shows an
# example address as placeholder text only; the frontend refuses to submit an
# empty or example.* address, and this backend gate rejects it a second time
# so a crafted POST can never trade a pasted API key for a probe to a
# stand-in host.
_EXAMPLE_RESERVED_ROOTS = frozenset(
    {
        "example.com",
        "example.org",
        "example.net",
        "example.edu",
    }
)


class ProviderUrlSecurityError(ValueError):
    """A provider endpoint was rejected before receiving a credential."""

    def __init__(self, reason: str) -> None:
        self.reason = str(reason or "rejected")
        super().__init__(self.reason)


def _hostname_of(base_url: str) -> str:
    """Return the lower-cased hostname without DNS, userinfo, or port."""

    try:
        return (urlsplit(str(base_url or "")).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def _is_example_host(hostname: str) -> bool:
    host = str(hostname or "").lower().rstrip(".")
    return any(
        host == root or host.endswith("." + root) for root in _EXAMPLE_RESERVED_ROOTS
    )


def validate_credential_endpoint(base_url: str) -> str:
    """Refuse a destination this host must not send an API credential to.

    Plaintext HTTP is permitted only for loopback services. Redirect handling
    remains the caller's responsibility and must stay disabled.
    """

    # D-P2-4: reject the documentation placeholder. This runs AFTER the
    # outbound policy check on purpose: structural violations
    # (credentials-in-URL, plaintext, metadata hosts) keep their specific
    # reason codes, and a clean public-https placeholder is still refused
    # here so no API key is ever sent to example.*.
    try:
        validated = validate_outbound_http_endpoint(
            base_url,
            proxy_fake_ip_https_hosts=_TRUSTED_PROVIDER_HOSTNAMES,
        )
    except OutboundUrlSecurityError as exc:
        raise ProviderUrlSecurityError(exc.reason) from exc
    if _is_example_host(_hostname_of(validated)):
        raise ProviderUrlSecurityError("example_placeholder_not_allowed")
    return validated


__all__ = ["ProviderUrlSecurityError", "validate_credential_endpoint"]
