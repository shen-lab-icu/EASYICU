"""Per-call resolver checkout for a loader shared across threads (B18, D27).

A ``ConceptResolver`` runs one top-level ``load_concepts`` call at a time; a
call from another context is refused with ``ResolverBusyError``. A loader that
concurrent callers share (the ``api.concepts`` global loader cache) therefore
lends each call ONE resolver from a bounded pool it owns.

Call scope. ``resolver_call_scope`` opens a scope (bound in a ``ContextVar``)
whose owner is the call. A nested load in the same context, or a
``contextvars.copy_context`` worker, enters the same scope as another user.
The scope closes when its owner exits, and its leases go back to the pool
only after the LAST user has left. A copied context that tries to enter a
closed scope (it outlived its call) is refused with ``ResolverCallClosed``:
it still names the old call, so starting a new call from it silently would
be wrong (D27-SCOPE-LIFETIME).

Leases. The pool records a lease as ``slot.owner = scope``, a single store
under the pool's lock; nothing else marks a resolver as taken. Cleanup
releases every slot whose owner is the scope, so a lease interrupted between
the checkout and the caller seeing it is still found and returned
(D27-CHECKOUT-CANCEL). A released resolver whose own call is still running
(a joined worker) becomes idle when that call ends (``defer_until_idle``).

Waiting. With all N members leased a checkout waits on the pool's
``Condition`` for at most the configured wait, then raises the typed
``ResolverPoolExhausted`` (a call that waits on a plain thread which itself
calls ``load_concepts`` can otherwise never finish).

Each member keeps the resolver's own bounded caches and revalidation, so the
cached data can reach N times the per-resolver cache cap (the cap is not a
bound on process memory). N defaults to :data:`DEFAULT_POOL_SIZE` and the
wait to :data:`DEFAULT_POOL_WAIT_SECONDS`; both are set by environment
variables and recorded in run metadata.
"""

from __future__ import annotations

import contextvars
import math
import os
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

from .resolver_errors import ResolverCallClosed, ResolverPoolExhausted

__all__ = [
    "DEFAULT_POOL_SIZE",
    "DEFAULT_POOL_WAIT_SECONDS",
    "POOL_SIZE_ENV",
    "POOL_WAIT_ENV",
    "ResolverPool",
    "call_resolver",
    "adopt_resolver_pool_config",
    "resolver_call_scope",
    "resolver_pool_config",
    "resolver_pool_size",
    "resolver_pool_wait_seconds",
]

DEFAULT_POOL_SIZE = 4
POOL_SIZE_ENV = "EASYICU_RESOLVER_POOL_SIZE"
DEFAULT_POOL_WAIT_SECONDS = 600.0
POOL_WAIT_ENV = "EASYICU_RESOLVER_POOL_WAIT_SECONDS"


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


def resolver_pool_wait_seconds() -> float:
    """The configured longest pool wait (a finite, positive number of seconds)."""
    raw = os.environ.get(POOL_WAIT_ENV, "").strip()
    if not raw:
        return DEFAULT_POOL_WAIT_SECONDS
    try:
        seconds = float(raw)
    except ValueError:
        raise ValueError(f"{POOL_WAIT_ENV} must be a positive number of seconds, got {raw!r}") from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{POOL_WAIT_ENV} must be a positive number of seconds, got {raw!r}")
    return seconds


_CONFIG_LOCK = threading.Lock()
_CONFIG: Optional[tuple] = None


def resolver_pool_config() -> dict:
    """This process's pool size and wait, frozen at the first use.

    Every pool created without explicit values uses them, and run metadata
    records them, so a later change of the environment cannot make the
    record disagree with the pools that served the run. Once frozen, a read
    takes no lock (one tuple read); only the first freeze is locked, and a
    forked child gets a fresh lock (D38-FORK-LOCK).
    """
    global _CONFIG
    frozen = _CONFIG
    if frozen is None:
        with _CONFIG_LOCK:
            if _CONFIG is None:
                _CONFIG = (resolver_pool_size(), resolver_pool_wait_seconds())
            frozen = _CONFIG
    size, wait_seconds = frozen
    return {"size": size, "wait_seconds": wait_seconds}


def adopt_resolver_pool_config(config) -> dict:
    """Freeze a worker process to its parent's pool configuration.

    Extraction passes the parent's frozen values to every module worker, so
    the run's recorded configuration is the one each worker used. A worker
    already frozen to different values (it read its own environment first)
    is refused.
    """
    global _CONFIG
    wanted = (int(config["size"]), float(config["wait_seconds"]))
    if wanted[0] < 1 or not math.isfinite(wanted[1]) or wanted[1] <= 0:
        raise ValueError(f"resolver pool configuration out of range: {config!r}")
    with _CONFIG_LOCK:
        if _CONFIG is None:
            _CONFIG = wanted
        elif _CONFIG != wanted:
            raise RuntimeError(
                f"resolver_pool_config_mismatch: this worker already uses {_CONFIG}, "
                f"its run uses {wanted}"
            )
    return resolver_pool_config()


def _fresh_config_lock_in_child() -> None:
    # A lock held by another thread at fork stays locked forever in the child
    # (its owner thread does not exist there). Keep the frozen tuple.
    global _CONFIG_LOCK
    _CONFIG_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_fresh_config_lock_in_child)


def _reset_resolver_pool_config() -> None:
    """Forget the frozen configuration (tests that change the environment)."""
    global _CONFIG
    with _CONFIG_LOCK:
        _CONFIG = None


class _Slot:
    __slots__ = ("resolver", "owner", "returning_owner")

    def __init__(self, resolver, owner=None) -> None:
        self.resolver = resolver
        self.owner = owner  # None = idle; a scope = leased; _RETURNING = waiting for its call
        self.returning_owner = None


_RETURNING = object()


class ResolverPool:
    """At most ``size`` resolvers, each leased to at most one call scope."""

    def __init__(
        self,
        primary,
        factory: Callable[[], object],
        *,
        size: Optional[int] = None,
        wait_seconds: Optional[float] = None,
        on_create: Optional[Callable[[object], None]] = None,
        defer_until_idle: Optional[Callable[[object, Callable[[], None]], bool]] = None,
        is_idle: Optional[Callable[[object], bool]] = None,
    ) -> None:
        config = resolver_pool_config() if size is None or wait_seconds is None else None
        self.size = config["size"] if size is None else int(size)
        if self.size < 1:
            raise ValueError(f"resolver pool size must be positive, got {self.size}")
        self.wait_seconds = config["wait_seconds"] if wait_seconds is None else float(wait_seconds)
        if not math.isfinite(self.wait_seconds) or self.wait_seconds <= 0:
            raise ValueError(f"resolver pool wait must be positive, got {self.wait_seconds}")
        self._primary = primary
        self._factory = factory
        self._on_create = on_create
        self._defer_until_idle = defer_until_idle
        self._is_idle = is_idle
        self._slots: list = [_Slot(primary)]
        self._condition = threading.Condition()

    def _held_by(self, owner):
        for slot in self._slots:
            if slot.owner is owner:
                return slot.resolver
        return None

    def checkout(self, owner=None):
        """Lease one resolver to ``owner`` (idempotent per owner); wait if all are leased.

        Without an owner (manual use) the lease is anonymous and only
        :meth:`checkin` returns it.
        """
        if owner is None:
            owner = object()
        elif owner is _RETURNING:
            raise ValueError("a lease needs an owner")
        deadline = time.monotonic() + self.wait_seconds
        with self._condition:
            while True:
                held = self._held_by(owner)
                if held is not None:
                    return held
                self._reclaim_finished()
                free = [slot for slot in self._slots if slot.owner is None]
                if free:
                    slot = next((s for s in free if s.resolver is self._primary), free[-1])
                    slot.owner = owner  # the lease: one store, found again by owner
                    return slot.resolver
                if len(self._slots) < self.size:
                    resolver = self._factory()
                    if self._on_create is not None:
                        self._on_create(resolver)
                    self._slots.append(_Slot(resolver, owner))  # the lease: one append
                    return resolver
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ResolverPoolExhausted(self.size, self.wait_seconds)
                self._condition.wait(remaining)

    def release(self, owner) -> None:
        """Return every resolver leased to ``owner``; one still running returns when its call ends."""
        with self._condition:
            for slot in self._slots:
                if slot.owner is owner:
                    self._return_slot(slot, owner)
            self._condition.notify_all()

    def _return_slot(self, slot, owner) -> None:
        # Under the condition. The deferral is registered before the slot is
        # marked, and ``_returned`` accepts either mark, so an interrupt in
        # between cannot strand the slot.
        slot.returning_owner = owner
        if self._defer_until_idle is not None and self._defer_until_idle(
            slot.resolver, lambda slot=slot, owner=owner: self._returned(slot, owner)
        ):
            slot.owner = _RETURNING
        else:
            slot.owner = None
            slot.returning_owner = None

    def _reclaim_finished(self) -> None:
        """Recover interrupted scope releases and interrupted idle callbacks."""
        for slot in self._slots:
            if slot.owner is _RETURNING:
                if self._is_idle is not None and self._is_idle(slot.resolver):
                    slot.owner = None
                    slot.returning_owner = None
                    self._condition.notify_all()
                continue
            finished = getattr(slot.owner, "finished", None)
            if finished is not None and finished():
                self._return_slot(slot, slot.owner)

    def _returned(self, slot, owner) -> None:
        with self._condition:
            # Recovery can make the resolver available before this callback runs.
            # An old callback must not release a newer scope's pending return.
            if (slot.owner is _RETURNING and slot.returning_owner is owner) or slot.owner is owner:
                slot.owner = None
                slot.returning_owner = None
                self._condition.notify_all()

    def checkin(self, resolver) -> None:
        """Return one resolver leased outside a call scope (tests, manual use)."""
        with self._condition:
            slot = next((s for s in self._slots if s.resolver is resolver), None)
            if slot is None:
                raise RuntimeError("resolver_pool_foreign: this resolver is not a member of the pool")
            if slot.owner is None:
                raise RuntimeError("resolver_pool_double_return: this resolver is already idle")
            slot.owner = None
            slot.returning_owner = None
            self._condition.notify_all()

    def members(self) -> tuple:
        with self._condition:
            return tuple(slot.resolver for slot in self._slots)

    def stats(self) -> dict:
        with self._condition:
            self._reclaim_finished()
            return {
                "size": self.size,
                "members": len(self._slots),
                "idle": sum(slot.owner is None for slot in self._slots),
            }


class _CallScope:
    """One call's leases; closed when its owner exits, released after its last user.

    ``finished`` (closed, no users left) is what makes its leases returnable;
    ``released`` is set only after every pool has taken them back. The
    release is idempotent, so an interrupted one is resumed by a later
    ``leave`` or by the pool itself (``ResolverPool`` reclaims the leases of
    a finished scope, D35-RELEASE-CANCEL).
    """

    __slots__ = ("lock", "owner", "users", "closed", "released", "pools")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.owner = object()
        self.users = {self.owner}  # entry tokens; a set add/discard is one operation
        self.closed = False
        self.released = False
        self.pools: list = []

    def finished(self) -> bool:
        return self.closed and not self.users

    def leave(self, entry) -> None:
        with self.lock:
            self.users.discard(entry)
            if entry is self.owner:
                self.closed = True
            if not self.finished() or self.released:
                return
            pools = list(self.pools)
        for pool in pools:
            pool.release(self)  # idempotent: returns only what is still leased to us
        with self.lock:
            self.released = True


_CALL_SCOPE: contextvars.ContextVar[Optional[_CallScope]] = contextvars.ContextVar(
    "easyicu_call_scope", default=None
)


@contextmanager
def resolver_call_scope() -> Iterator[None]:
    """Open this call's scope, or enter the open scope of the call this context belongs to."""
    scope = _CALL_SCOPE.get()
    if scope is not None:
        entry = object()
        try:
            with scope.lock:
                if scope.closed:
                    raise ResolverCallClosed("copied context")
                scope.users.add(entry)
            yield
        finally:
            scope.leave(entry)  # no-op unless the entry was added
        return
    scope = _CallScope()
    try:
        _CALL_SCOPE.set(scope)
        yield
    finally:
        try:
            _CALL_SCOPE.set(None)  # this branch only runs with no scope bound
        finally:
            scope.leave(scope.owner)


def call_resolver(pool: ResolverPool):
    """The resolver this call leases from ``pool``, leasing one the first time."""
    scope = _CALL_SCOPE.get()
    if scope is None:
        raise RuntimeError("call_resolver needs an open resolver_call_scope")
    with scope.lock:
        if scope.released:
            raise ResolverCallClosed("call scope")
        if not any(known is pool for known in scope.pools):
            scope.pools.append(pool)  # registered before the lease, so cleanup finds it
        return pool.checkout(scope)
