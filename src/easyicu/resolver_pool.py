"""Per-call resolver checkout for a loader shared across threads (B18).

A ``ConceptResolver`` runs one top-level ``load_concepts`` call at a time; a
call from another context is refused with ``ResolverBusyError``. A loader that
concurrent callers share (the ``api.concepts`` global loader cache) therefore
lends each call ONE resolver from a bounded pool it owns:

* the call opens a scope (:func:`resolver_call_scope`), a ``ContextVar`` that
  binds the resolvers it checked out;
* the first load in the scope checks one out (:func:`call_resolver`): an idle
  resolver is reused (the loader's own resolver first, so sequential callers
  and ``keep_cache`` keep their caches); with none idle and fewer than N
  members a new one is created; with N members the caller BLOCKS on the pool
  condition until a call returns one. No timeouts, no re-routing;
* a nested load in the same context (or a ``contextvars.copy_context`` worker)
  reaches the same scope and so the call's own resolver, where resolver
  nesting applies; anything that re-enters that resolver from another context
  still gets ``ResolverBusyError``. The pool is never consulted for it;
* the scope returns its resolvers on every exit path (``finally``).

Each member keeps the resolver's own bounded caches and revalidation, so the
pool can hold up to N times the per-resolver cache cap. N defaults to
:data:`DEFAULT_POOL_SIZE`; ``EASYICU_RESOLVER_POOL_SIZE`` sets it (a positive
integer) and run metadata records it.
"""

from __future__ import annotations

import contextvars
import os
import threading
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

__all__ = [
    "DEFAULT_POOL_SIZE",
    "POOL_SIZE_ENV",
    "ResolverPool",
    "call_resolver",
    "resolver_call_scope",
    "resolver_pool_size",
]

DEFAULT_POOL_SIZE = 4
POOL_SIZE_ENV = "EASYICU_RESOLVER_POOL_SIZE"


def resolver_pool_size() -> int:
    """The configured pool size N (refuses anything but a positive integer)."""
    raw = os.environ.get(POOL_SIZE_ENV, "").strip()
    if not raw:
        return DEFAULT_POOL_SIZE
    try:
        size = int(raw)
    except ValueError:
        raise ValueError(f"{POOL_SIZE_ENV} must be a positive integer, got {raw!r}") from None
    if size < 1:
        raise ValueError(f"{POOL_SIZE_ENV} must be a positive integer, got {raw!r}")
    return size


class ResolverPool:
    """At most ``size`` resolvers; a checkout blocks on the pool when all are out."""

    def __init__(
        self,
        primary,
        factory: Callable[[], object],
        *,
        size: Optional[int] = None,
        on_create: Optional[Callable[[object], None]] = None,
    ) -> None:
        self.size = resolver_pool_size() if size is None else int(size)
        if self.size < 1:
            raise ValueError(f"resolver pool size must be positive, got {self.size}")
        self._primary = primary
        self._factory = factory
        self._on_create = on_create
        self._members: list = [primary]
        self._idle: list = [primary]
        self._condition = threading.Condition()

    def checkout(self):
        with self._condition:
            while True:
                if self._idle:
                    index = next(
                        (i for i, r in enumerate(self._idle) if r is self._primary),
                        len(self._idle) - 1,
                    )
                    return self._idle.pop(index)
                if len(self._members) < self.size:
                    resolver = self._factory()
                    if self._on_create is not None:
                        self._on_create(resolver)
                    self._members.append(resolver)
                    return resolver
                self._condition.wait()

    def checkin(self, resolver) -> None:
        with self._condition:
            if not any(member is resolver for member in self._members):
                raise RuntimeError("resolver_pool_foreign: this resolver is not a member of the pool")
            if any(idle is resolver for idle in self._idle):
                raise RuntimeError("resolver_pool_double_return: this resolver is already idle")
            self._idle.append(resolver)
            self._condition.notify()

    def members(self) -> tuple:
        with self._condition:
            return tuple(self._members)

    def stats(self) -> dict:
        with self._condition:
            return {"size": self.size, "members": len(self._members), "idle": len(self._idle)}


class _CallHolds:
    __slots__ = ("lock", "held")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.held: list = []  # (pool, resolver) pairs, at most one per pool


_CALL_HOLDS: contextvars.ContextVar[Optional[_CallHolds]] = contextvars.ContextVar(
    "easyicu_call_resolvers", default=None
)


@contextmanager
def resolver_call_scope() -> Iterator[None]:
    """Bind one call's checked-out resolvers; nested scopes join the outer one."""
    if _CALL_HOLDS.get() is not None:
        yield
        return
    holds = _CallHolds()
    token = _CALL_HOLDS.set(holds)
    try:
        yield
    finally:
        _CALL_HOLDS.reset(token)
        with holds.lock:
            held, holds.held = holds.held, []
        for pool, resolver in held:
            pool.checkin(resolver)


def call_resolver(pool: ResolverPool):
    """The resolver this call holds from ``pool``, checking one out the first time."""
    holds = _CALL_HOLDS.get()
    if holds is None:
        raise RuntimeError("call_resolver needs an open resolver_call_scope")
    with holds.lock:
        for owner, resolver in holds.held:
            if owner is pool:
                return resolver
        resolver = pool.checkout()
        holds.held.append((pool, resolver))
        return resolver
