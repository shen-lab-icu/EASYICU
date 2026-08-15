"""Shared local-host security contract for credential-bearing provider URLs."""

from __future__ import annotations

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


class ProviderUrlSecurityError(ValueError):
    """A provider endpoint was rejected before receiving a credential."""

    def __init__(self, reason: str) -> None:
        self.reason = str(reason or "rejected")
        super().__init__(self.reason)


def validate_credential_endpoint(base_url: str) -> str:
    """Refuse a destination this host must not send an API credential to.

    Plaintext HTTP is permitted only for loopback services. Redirect handling
    remains the caller's responsibility and must stay disabled.
    """

    try:
        return validate_outbound_http_endpoint(
            base_url,
            proxy_fake_ip_https_hosts=_TRUSTED_PROVIDER_HOSTNAMES,
        )
    except OutboundUrlSecurityError as exc:
        raise ProviderUrlSecurityError(exc.reason) from exc


__all__ = ["ProviderUrlSecurityError", "validate_credential_endpoint"]
