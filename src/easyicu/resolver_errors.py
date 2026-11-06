"""Typed refusal of a concurrent top-level call on a resolver already in use (B18-REENTRY-DEADLOCK).

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
