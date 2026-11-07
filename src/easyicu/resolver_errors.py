"""Typed refusals of the per-call resolver guard and pool (B18-REENTRY-DEADLOCK, D27).

A ``ConceptResolver`` keeps per-call state (depth, batch scope, request
caches), so one top-level ``load_concepts`` runs on it at a time. A second
top-level call from another context, whether a concurrent caller sharing the
resolver or a plain thread-pool worker that a callback started without
propagating its context, is refused at once instead of waiting: waiting
could never end when the call holding the resolver waits for that worker.
"""

from __future__ import annotations


class ResolverBusyError(RuntimeError):
    """A top-level ``load_concepts`` on a resolver whose call runs in another context."""

    reason = "resolver_busy"

    def __init__(self, resolver_id: int):
        self.resolver_id = int(resolver_id)
        super().__init__(
            "resolver_busy: this ConceptResolver is running a load_concepts call in "
            "another context; use a separate resolver per concurrent caller, or run "
            "work that re-enters it with contextvars.copy_context().run so it joins "
            "the running call"
        )

    def __reduce__(self):  # crosses process boundaries with its fields
        return (type(self), (self.resolver_id,))


class ResolverCallClosed(RuntimeError):
    """Work in a copied context that outlived the ``load_concepts`` call it belonged to.

    A ``contextvars.copy_context`` worker joins its call (same resolver, same
    unit). Once that call has finished, the copied context can no longer load
    through it: the call's resolver may already serve another request. This
    is refused rather than silently started as a new call, because the stale
    context still names the old call's resolver and unit.
    """

    reason = "resolver_call_closed"

    def __init__(self, scope: str):
        self.scope = str(scope)
        super().__init__(
            f"resolver_call_closed: this {self.scope} belongs to a load_concepts call that "
            "has already finished; work in a copied context must finish inside its call "
            "(join the worker before the call returns), or start a new call from a fresh "
            "context"
        )

    def __reduce__(self):
        return (type(self), (self.scope,))


class ResolverPoolExhausted(RuntimeError):
    """No pooled resolver came free within the configured wait."""

    reason = "resolver_pool_exhausted"

    def __init__(self, size: int, wait_seconds: float):
        self.size = int(size)
        self.wait_seconds = float(wait_seconds)
        super().__init__(
            f"resolver_pool_exhausted: all {self.size} resolvers of this loader stayed in "
            f"use for {self.wait_seconds:g} s; raise EASYICU_RESOLVER_POOL_SIZE or "
            "EASYICU_RESOLVER_POOL_WAIT_SECONDS, or check for a call that waits on a "
            "plain thread which itself calls load_concepts"
        )

    def __reduce__(self):
        return (type(self), (self.size, self.wait_seconds))
