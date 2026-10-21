"""Explicit gateway configuration, independent of instance/provider settings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


def https_origin(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
            or any(ch.isspace() for ch in value)):
        raise ValueError("public_origin must be an HTTPS origin")
    port = parsed.port
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    return f"https://{host}" + (f":{port}" if port and port != 443 else "")


@dataclass(frozen=True)
class GatewayConfig:
    public_origin: str
    users_file: Path
    secret_file: Path
    cert_file: Path
    key_file: Path
    host: str = "127.0.0.1"
    port: int = 8443

    def __post_init__(self) -> None:
        object.__setattr__(self, "public_origin", https_origin(self.public_origin))
        if not self.host or type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("host and port must identify a listening address")
        paths = [self.users_file, self.secret_file, self.cert_file, self.key_file]
        if any(not path.is_absolute() for path in paths) or len(set(paths)) != len(paths):
            raise ValueError("gateway file paths must be distinct absolute paths")


def load_config(path: Path) -> GatewayConfig:
    path = path.expanduser().absolute()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("gateway config must be an object")
    for key in ("users_file", "secret_file", "cert_file", "key_file"):
        value = Path(payload[key]).expanduser()
        # Do not resolve symlinks: the private-file owner must reject them.
        payload[key] = value if value.is_absolute() else path.parent / value
    return GatewayConfig(**payload)
