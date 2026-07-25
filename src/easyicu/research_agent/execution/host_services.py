"""Host-owned services consumed by the execute control plane.

The execute phase is intentionally implemented outside :mod:`pipeline`, but it
still needs a small set of host functions whose public compatibility surface is
owned by that module.  Importing those functions back from ``pipeline`` created
the final control-plane import cycle.  These immutable dependency objects make
that ownership explicit without copying the implementations or moving scientific
decisions into the executor.

``ResearchAgentPipeline`` constructs a fresh object for every execute-phase
call.  Consequently a test or embedding application that monkeypatches one of
the legacy pipeline helpers before a run still supplies the current helper,
rather than a stale function captured at module-import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class PublicationFigureAuthorityServices:
    """Host functions that authorize one deterministic figure adapter."""

    distribution_availability_step_matches_parent: Callable[..., bool]
    sealed_renderer_step_matches_parent: Callable[..., bool]
    sealed_renderer_parent_digest_seal: Callable[..., Any]
    deterministic_repair_id_for_upstream: Callable[..., Any]


@dataclass(frozen=True)
class ExecutePhaseServices:
    """Immutable host-service snapshot for one execute-phase invocation."""

    build_probe_summary: Callable[..., Any]
    deterministic_figure_family_supported_for_upstream: Callable[..., bool]
    promote_prior_publication_bundle: Callable[..., Any]
    promote_sibling_figure_exports: Callable[..., Any]
    render_publication_bundle_from_prior_outputs_for_step: Callable[..., Any]
    semantic_aliases_for: Callable[..., Any]
    publication_figure_authority: PublicationFigureAuthorityServices


class ExecutePhaseHost(Protocol):
    """Structural host contract required by :func:`run_execute_phase`.

    The executor also reads configuration attributes and invokes existing agent
    factories on the host.  ``__getattr__`` keeps those intentionally broad
    collaborator reads typed as ``Any`` while the cycle-breaking service seam is
    explicit and testable.
    """

    def _execute_phase_services(self) -> ExecutePhaseServices: ...

    def __getattr__(self, name: str) -> Any: ...


__all__ = [
    "ExecutePhaseHost",
    "ExecutePhaseServices",
    "PublicationFigureAuthorityServices",
]
