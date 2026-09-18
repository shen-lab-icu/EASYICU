"""Shared SSRF and transport policy for configured outbound HTTP endpoints.

Validation-vs-connect separation (TOCTOU) limitation
----------------------------------------------------
:func:`validate_outbound_http_endpoint` validates the DNS resolution observed
at call time. It does **not** pin the transport: a hostname whose DNS answer
changes (or resolves differently per-connection, e.g. round-robin / split
horizon) may connect to an address other than the one validated here. This is
an inherent validate-then-connect separation, not a stable connection guarantee.

High-risk deployments should therefore pin the validated address: resolve the
hostname, call this validator with ``pinned_ip`` set to the intended address,
and then connect to that pinned IP directly (preserving the original hostname
for TLS SNI/Host verification at the call site). The ``pinned_ip`` parameter
is per-call state only (no shared mutable state), so callers may pass it
through threads safely; each thread validates and pins its own connection.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Iterable, Optional, Tuple
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
    pinned_ip: str | None = None,
) -> str:
    """Allow public HTTPS and loopback HTTP while rejecting SSRF targets.

    The optional fake-IP exception is intentionally hostname-scoped.  It
    supports certificate-validated official providers behind a Clash-style
    proxy without weakening custom endpoint validation.

    Args:
        url: Candidate outbound endpoint.
        proxy_fake_ip_https_hosts: Hostnames allowed to resolve into the
            Clash-style fake-IP range (198.18.0.0/15) over HTTPS.
        pinned_ip: Optional already-validated IP string the caller intends to
            connect to. When given, it must be one of the addresses resolved
            for ``url`` right now and must itself satisfy the network policy;
            otherwise validation fails. This narrows (but does not fully
            close) the DNS validation-vs-connect TOCTOU window: the caller is
            still responsible for actually connecting to ``pinned_ip`` (with
            correct SNI/Host handling) instead of re-resolving the hostname.

    .. note::
        Validation and connection are separate steps. Re-resolving the
        hostname at connect time may yield a different address than the one
        validated here. High-risk deployments must pass ``pinned_ip`` and
        connect to the pinned address.
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
        if not address.is_global:
            # ``is_private`` does not cover every non-routable class. In
            # particular, IPv4 shared address space (100.64.0.0/10, CGNAT) is
            # neither private nor global according to ``ipaddress``.
            _refuse("non_global_address")
    if parsed.scheme == "http" and not loopback_only:
        _refuse("plaintext_to_non_loopback")
    if pinned_ip is not None:
        try:
            pinned = ipaddress.ip_address(str(pinned_ip).strip())
        except ValueError:
            _refuse("pinned_ip_invalid")
            raise  # pragma: no cover - _refuse always raises
        if pinned not in addresses:
            # The pinned address is not what DNS returns right now: either
            # stale, mistyped, or a TOCTOU race. Fail closed.
            _refuse("pinned_ip_mismatch")
    return text


def validated_http_endpoint_with_pin(
    url: str,
    *,
    proxy_fake_ip_https_hosts: Iterable[str] = (),
    pinned_ip: str | None = None,
) -> Tuple[str, Optional[str]]:
    """Validate like :func:`validate_outbound_http_endpoint` and additionally
    return a ``(connect_url, host_header)`` pair from the SAME resolution.

    A single ``getaddrinfo`` backs both the policy check and the returned
    address, so there is no resolve#1-vs-resolve#2 gap inside this call:

    * literal-IP hosts → ``(url, None)`` (no DNS involved);
    * loopback-HTTP DNS names → ``(url with the validated IP, original
      Host header)``: the caller must connect to ``connect_url`` (no
      re-resolution happens) and send ``host_header``. This closes the DNS
      rebinding window for plaintext loopback endpoints;
    * ``https`` DNS names → ``(url, None)``: rewriting to a literal IP would
      break TLS SNI/certificate verification, so the hostname URL is kept.
      Residual risk is TLS-constrained (a rebinding target without a valid
      certificate for the original hostname fails the handshake).

    ``pinned_ip``, when given, must be one of the addresses resolved right
    now (same rule as the validator); otherwise validation fails.
    """

    text = str(url or "").strip()
    validated = validate_outbound_http_endpoint(
        text,
        proxy_fake_ip_https_hosts=proxy_fake_ip_https_hosts,
        pinned_ip=pinned_ip,
    )
    parsed = urlsplit(validated)
    host = (parsed.hostname or "").strip()
    try:
        ipaddress.ip_address(host)
        return validated, None
    except ValueError:
        pass
    if parsed.scheme != "http":
        return validated, None
    infos = socket.getaddrinfo(
        host, parsed.port or 80, proto=socket.IPPROTO_TCP
    )
    candidates = [info[4][0] for info in infos]
    # Prefer IPv4: every candidate here is already policy-clean (loopback for
    # http), and IPv4 maximizes server compatibility (::1-only listeners are
    # rare; a v4-only server is common). Deterministic, documented bias.
    candidates.sort(key=lambda ip: (":" in ip, ip))
    if pinned_ip is not None:
        wanted = str(ipaddress.ip_address(str(pinned_ip).strip()))
        if wanted not in candidates:
            raise OutboundUrlSecurityError("pinned_ip_mismatch")
        chosen = wanted
    else:
        chosen = candidates[0]
    # Re-check the finally chosen address: DNS may legally differ between the
    # validator's resolution above and this one (round-robin); only a still
    # policy-clean address may be used for the connection.
    validate_outbound_http_endpoint(
        validated, proxy_fake_ip_https_hosts=proxy_fake_ip_https_hosts,
        pinned_ip=chosen,
    )
    port = parsed.port or 80
    host_header = host if port == 80 else f"{host}:{port}"
    ip_host = f"[{chosen}]" if ":" in chosen else chosen
    rebuilt = parsed._replace(
        netloc=f"{ip_host}:{parsed.port}" if parsed.port else ip_host
    )
    return rebuilt.geturl(), host_header


__all__ = ["OutboundUrlSecurityError", "validate_outbound_http_endpoint", "validated_http_endpoint_with_pin"]
