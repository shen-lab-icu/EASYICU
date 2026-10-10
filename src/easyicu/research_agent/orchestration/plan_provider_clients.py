"""The Provider clients a run's plan phase calls, wrapped once for the phase.

Owner
-----
Every role client the plan phase resolves is wrapped in one order: the
reproducibility envelope records the call, the hard stop reserves every raw
transport retry before delivery, and the meter receives usage from that same
call for the run manifest.  The plan phase builds these clients before its
first Provider call -- the exposure-grouping request, when the host plans
groupings, comes before the outline -- so no call it makes escapes the
envelope, the hard stop or the meter.  A resumed review handoff restores its
own clients (``ResearchAgentPipeline._restore_role_handoff``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from ..providers.cost import CostMeter, metered_role_resolver
from ..providers.hard_stop import HardStopClient
from ..providers.llm import resolve_role_client
from ..replication.envelope import ReproEnvelope, envelope_role_resolver


@dataclass(frozen=True)
class PlanProviderClients:
    """Each role's wrapped client, and the records its calls are kept in."""

    role_resolver: Callable[[str], Any]
    cost_meter: Optional[CostMeter]
    repro_envelope: Optional[ReproEnvelope]


class _RoleResolverShim:
    name = "role_resolver_shim"

    def __init__(self, resolver: Callable[[str], Any]) -> None:
        self._resolver = resolver

    def for_role(self, role: str) -> Any:
        return self._resolver(role)

    def complete(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover
        raise RuntimeError("RoleResolverShim is a dispatcher; call for_role() first.")


def plan_provider_clients(
    *,
    llm: Any,
    run_id: str,
    run_dir: Path,
    reproducibility_envelope: bool,
    llm_seed: Optional[int],
    envelope_include_previews: bool,
    provider_hard_stop: Any,
    cost_tracking: bool,
    cost_price_table: Optional[Mapping[str, Any]],
) -> PlanProviderClients:
    """Wrap ``llm``'s role clients: envelope, then hard stop, then meter."""

    repro_envelope: Optional[ReproEnvelope] = None
    if reproducibility_envelope:
        repro_envelope = ReproEnvelope(
            run_id=run_id,
            seed=llm_seed,
            include_previews=envelope_include_previews,
        )
    if repro_envelope is not None:
        base_role_resolver = envelope_role_resolver(
            llm,
            repro_envelope,
            seed=llm_seed,
        )
    else:

        def base_role_resolver(role: str) -> Any:
            return resolve_role_client(llm, role)

    if provider_hard_stop is not None:

        def stopped_role_resolver(role: str) -> Any:
            base = base_role_resolver(role)
            if base is None or isinstance(base, HardStopClient):
                return base
            return HardStopClient(base, role=role, task=provider_hard_stop)

    else:
        stopped_role_resolver = base_role_resolver

    if not cost_tracking:
        return PlanProviderClients(
            role_resolver=stopped_role_resolver,
            cost_meter=None,
            repro_envelope=repro_envelope,
        )
    cost_meter = (
        CostMeter(
            price_table=dict(cost_price_table) if cost_price_table else None,
            runtime_dir=run_dir / ".runtime",
        )
        if cost_price_table is not None
        else CostMeter(runtime_dir=run_dir / ".runtime")
    )
    return PlanProviderClients(
        role_resolver=metered_role_resolver(
            _RoleResolverShim(stopped_role_resolver), cost_meter
        ),
        cost_meter=cost_meter,
        repro_envelope=repro_envelope,
    )


__all__ = ["PlanProviderClients", "plan_provider_clients"]
