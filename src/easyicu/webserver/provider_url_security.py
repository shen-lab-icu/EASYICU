"""Shared local-host security contract for credential-bearing provider URLs."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

_METADATA_HOSTNAMES = frozenset(
    {
        "metadata",
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
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

    text = str(base_url or "").strip()
    if not text:
        raise ProviderUrlSecurityError("missing")
    parsed = urlsplit(text)

    def _refuse(reason: str) -> None:
        raise ProviderUrlSecurityError(reason)

    if parsed.scheme not in {"http", "https"}:
        _refuse("scheme_not_http")
    if parsed.username or parsed.password:
        _refuse("credentials_in_url")
    if parsed.query or parsed.fragment:
        _refuse("query_or_fragment_in_url")
    host = (parsed.hostname or "").strip()
    if not host:
        _refuse("no_host")
    if host.lower() in _METADATA_HOSTNAMES:
        _refuse("metadata_host")

    try:
        resolved = socket.getaddrinfo(
            host,
            parsed.port or None,
            proto=socket.IPPROTO_TCP,
        )
    except OSError:
        _refuse("host_does_not_resolve")
        return text  # pragma: no cover - _refuse always raises
    addresses = {ipaddress.ip_address(info[4][0]) for info in resolved}
    if not addresses:
        _refuse("host_does_not_resolve")

    loopback_only = all(address.is_loopback for address in addresses)
    for address in addresses:
        if address.is_loopback:
            continue
        if address.is_link_local or address.is_reserved or address.is_multicast:
            _refuse("link_local_or_reserved_address")
        if address.is_private:
            _refuse("private_address")
    if parsed.scheme == "http" and not loopback_only:
        _refuse("plaintext_to_non_loopback")
    return text


__all__ = ["ProviderUrlSecurityError", "validate_credential_endpoint"]
