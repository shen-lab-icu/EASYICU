"""Internal hash binding between a metadata retry journal and Provider slot."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_REQUEST_SHA256: ContextVar[str | None] = ContextVar("metadata_request_sha256", default=None)


def current_metadata_request_sha256() -> str | None:
    return _REQUEST_SHA256.get()


@contextmanager
def metadata_request_binding(value: str) -> Iterator[None]:
    token = _REQUEST_SHA256.set(value)
    try:
        yield
    finally:
        _REQUEST_SHA256.reset(token)
