"""DoWhy adapter for graph identification, not causal-effect promotion.

EasyICU retains ownership of target-trial semantics, time zero, clinical
variable meanings and temporal legality.  This adapter asks DoWhy only whether
the already-declared graph identifies a causal estimand.  It neither chooses an
estimator nor upgrades the causal capability from ``analysis_only``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

from .runtime import probe_external_adapter


@dataclass(frozen=True)
class DoWhyIdentificationContract:
    """Coordinates supplied by the approved causal plan, never inferred here."""

    treatment: str
    outcome: str
    causal_graph: str
    observed_common_causes: Tuple[str, ...] = ()
    identifier: Literal["auto", "id-algorithm"] = "auto"

    def __post_init__(self) -> None:
        treatment = str(self.treatment).strip()
        outcome = str(self.outcome).strip()
        graph = str(self.causal_graph).strip()
        causes = tuple(str(cause).strip() for cause in self.observed_common_causes)
        object.__setattr__(self, "treatment", treatment)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "causal_graph", graph)
        object.__setattr__(self, "observed_common_causes", causes)
        values = {
            "treatment": treatment,
            "outcome": outcome,
            "causal_graph": graph,
        }
        for field, value in values.items():
            if not str(value).strip():
                raise ValueError(f"DoWhy identification contract requires {field}")
        if treatment == outcome:
            raise ValueError("DoWhy treatment and outcome must differ")
        if any(not cause for cause in causes) or len(causes) != len(set(causes)):
            raise ValueError("DoWhy observed common causes must be unique and nonblank")
        if self.treatment in causes or self.outcome in causes:
            raise ValueError("DoWhy common causes cannot repeat treatment or outcome")


@dataclass(frozen=True)
class DoWhyIdentificationReceipt:
    """Evidence-ready identification result without an effect estimate."""

    status: Literal[
        "identified",
        "not_identified",
        "input_contract_failed",
        "adapter_unavailable",
    ]
    adapter_version: Optional[str]
    issue_code: Optional[str]
    treatment: str
    outcome: str
    graph_sha256: str
    identifier: str
    declared_assumptions_sha256: str
    identified_estimand_type: Optional[str] = None
    identified_estimand_sha256: Optional[str] = None
    identification_routes: Tuple[str, ...] = ()
    assumption_fingerprints: Tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "easyicu.dowhy_identification_receipt/2",
            "status": self.status,
            "adapter_version": self.adapter_version,
            "issue_code": self.issue_code,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "graph_sha256": self.graph_sha256,
            "identifier": self.identifier,
            "declared_assumptions_sha256": self.declared_assumptions_sha256,
            "identified_estimand_type": self.identified_estimand_type,
            "identified_estimand_sha256": self.identified_estimand_sha256,
            "identification_routes": list(self.identification_routes),
            "assumption_fingerprints": list(self.assumption_fingerprints),
            "estimand_normalization": "whitespace_collapse_address_redaction_v1",
            "effect_estimate_present": False,
        }


def _graph_sha256(graph: str) -> str:
    return hashlib.sha256(graph.encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalise_estimand_text(value: Any) -> str:
    collapsed = " ".join(str(value).split())
    return re.sub(r"0x[0-9a-fA-F]+", "0xADDRESS", collapsed)


def _declared_assumptions_sha256(
    contract: DoWhyIdentificationContract,
) -> str:
    payload = {
        "treatment": contract.treatment,
        "outcome": contract.outcome,
        "graph_sha256": _graph_sha256(contract.causal_graph),
        "observed_common_causes": sorted(contract.observed_common_causes),
        "identifier": contract.identifier,
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _identified_estimand_receipt(
    identified_estimand: Any,
) -> tuple[Optional[str], Optional[str], Tuple[str, ...], Tuple[str, ...]]:
    if identified_estimand is None:
        return None, None, (), ()

    estimand_type = type(identified_estimand).__name__
    estimand_sha256 = _sha256_text(
        _normalise_estimand_text(identified_estimand)
    )
    routes: list[str] = []
    assumptions: list[str] = []
    estimands = getattr(identified_estimand, "estimands", None)
    if isinstance(estimands, Mapping):
        for route, route_estimand in sorted(
            estimands.items(), key=lambda item: str(item[0])
        ):
            if route_estimand is None:
                continue
            routes.append(str(route))
            route_assumptions = (
                route_estimand.get("assumptions")
                if isinstance(route_estimand, Mapping)
                else getattr(route_estimand, "assumptions", None)
            )
            if isinstance(route_assumptions, Mapping):
                assumptions.extend(
                    f"{key}:{value}"
                    for key, value in sorted(
                        route_assumptions.items(), key=lambda item: str(item[0])
                    )
                )
            elif isinstance(route_assumptions, (list, tuple, set, frozenset)):
                assumptions.extend(str(value) for value in route_assumptions)
            elif route_assumptions is not None:
                assumptions.append(str(route_assumptions))

    assumption_fingerprints = tuple(
        sorted({_sha256_text(_normalise_estimand_text(value)) for value in assumptions})
    )
    return estimand_type, estimand_sha256, tuple(routes), assumption_fingerprints


def identify_declared_causal_effect(
    dataframe: Any,
    contract: DoWhyIdentificationContract,
) -> DoWhyIdentificationReceipt:
    """Run only DoWhy's declared-graph identification operation.

    This returns no effect size.  An estimator, refuters and their clinical
    interpretation must be implemented as a separate, validated capability
    before causal results can become publication-reportable.
    """

    graph_sha256 = _graph_sha256(contract.causal_graph)
    declared_assumptions_sha256 = _declared_assumptions_sha256(contract)
    try:
        available_columns = set(str(column) for column in dataframe.columns)
    except (AttributeError, TypeError):
        available_columns = set()
    required_columns = {
        contract.treatment,
        contract.outcome,
        *contract.observed_common_causes,
    }
    if not required_columns.issubset(available_columns):
        return DoWhyIdentificationReceipt(
            status="input_contract_failed",
            adapter_version=None,
            issue_code="causal_identification_input_columns_unresolved",
            treatment=contract.treatment,
            outcome=contract.outcome,
            graph_sha256=graph_sha256,
            identifier=contract.identifier,
            declared_assumptions_sha256=declared_assumptions_sha256,
        )
    runtime = probe_external_adapter("dowhy_identification_v1")
    if not runtime.available:
        return DoWhyIdentificationReceipt(
            status="adapter_unavailable",
            adapter_version=runtime.installed_version,
            issue_code=runtime.issue_code,
            treatment=contract.treatment,
            outcome=contract.outcome,
            graph_sha256=graph_sha256,
            identifier=contract.identifier,
            declared_assumptions_sha256=declared_assumptions_sha256,
        )
    dowhy = importlib.import_module("dowhy")
    try:
        model = dowhy.CausalModel(
            data=dataframe,
            treatment=contract.treatment,
            outcome=contract.outcome,
            graph=contract.causal_graph,
            common_causes=list(contract.observed_common_causes),
        )
        kwargs = (
            {"method_name": "id-algorithm"}
            if contract.identifier == "id-algorithm"
            else {}
        )
        identified_estimand = model.identify_effect(**kwargs)
    except (TypeError, ValueError):
        return DoWhyIdentificationReceipt(
            status="input_contract_failed",
            adapter_version=runtime.installed_version,
            issue_code="causal_identification_engine_rejected_contract",
            treatment=contract.treatment,
            outcome=contract.outcome,
            graph_sha256=graph_sha256,
            identifier=contract.identifier,
            declared_assumptions_sha256=declared_assumptions_sha256,
        )
    (
        identified_estimand_type,
        identified_estimand_sha256,
        identification_routes,
        assumption_fingerprints,
    ) = _identified_estimand_receipt(identified_estimand)
    return DoWhyIdentificationReceipt(
        status="identified" if identified_estimand is not None else "not_identified",
        adapter_version=runtime.installed_version,
        issue_code=(None if identified_estimand is not None else "causal_effect_not_identified"),
        treatment=contract.treatment,
        outcome=contract.outcome,
        graph_sha256=graph_sha256,
        identifier=contract.identifier,
        declared_assumptions_sha256=declared_assumptions_sha256,
        identified_estimand_type=identified_estimand_type,
        identified_estimand_sha256=identified_estimand_sha256,
        identification_routes=identification_routes,
        assumption_fingerprints=assumption_fingerprints,
    )


__all__ = [
    "DoWhyIdentificationContract",
    "DoWhyIdentificationReceipt",
    "identify_declared_causal_effect",
]
