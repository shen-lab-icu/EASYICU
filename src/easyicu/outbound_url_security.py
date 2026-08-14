"""Shared SSRF and transport policy for configured outbound HTTP endpoints."""

from __future__ import annotations

import ipaddress
import socket
from typing import Iterable
from urllib.parse import urlsplit

_METADATA_HOSTNAMES = frozenset(
    {"metadata", "metadata.google.internal", "metadata.goog", "instance-data"}
)
_PROXY_FAKE_IP_NETWORK = ipaddress.ip_network("198.18.0.0/15")


class OutboundUrlSecurityError(ValueError):
    """A configured outbound endpoint violates the shared host policy."""

    def __init__(self, reason: str) -> None:
        self.reason = str(reason or "rejected")
        super().__init__(self.reason)


def validate_outbound_http_endpoint(
    url: str,
    *,
    proxy_fake_ip_https_hosts: Iterable[str] = (),
) -> str:
    """Allow public HTTPS and loopback HTTP while rejecting SSRF targets.

    The optional fake-IP exception is intentionally hostname-scoped.  It
    supports certificate-validated official providers behind a Clash-style
    proxy without weakening custom endpoint validation.
    """

    text = str(url or "").strip()

    def _refuse(reason: str) -> None:
        raise OutboundUrlSecurityError(reason)

    if not text:
        _refuse("missing")
    parsed = urlsplit(text)
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
        return text  # pragma: no cover
    addresses = {ipaddress.ip_address(info[4][0]) for info in resolved}
    if not addresses:
        _refuse("host_does_not_resolve")

    loopback_only = all(address.is_loopback for address in addresses)
    if any(address.is_loopback for address in addresses) and not loopback_only:
        # A hostname that spans loopback and public space has no single network
        # trust class.  Accepting it would let the resolver choose the public
        # answer during validation and the loopback answer during connect.
        _refuse("mixed_address_scope")
    proxy_exception = (
        parsed.scheme == "https"
        and host.lower() in {str(item).lower() for item in proxy_fake_ip_https_hosts}
    )
    for address in addresses:
        if address.is_loopback:
            continue
        if proxy_exception and address in _PROXY_FAKE_IP_NETWORK:
            continue
        if address.is_link_local or address.is_reserved or address.is_multicast:
            _refuse("link_local_or_reserved_address")
        if address.is_private:
            _refuse("private_address")
    if parsed.scheme == "http" and not loopback_only:
        _refuse("plaintext_to_non_loopback")
    return text


__all__ = ["OutboundUrlSecurityError", "validate_outbound_http_endpoint"]
