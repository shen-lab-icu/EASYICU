"""DoWhy adapter for graph identification, not causal-effect promotion.

EasyICU retains ownership of target-trial semantics, time zero, clinical
variable meanings and temporal legality.  This adapter asks DoWhy only whether
the already-declared graph identifies a causal estimand.  It neither chooses an
estimator nor upgrades the causal capability from ``analysis_only``.
"""

from __future__ import annotations

import hashlib
import importlib
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
        values = {
            "treatment": self.treatment,
            "outcome": self.outcome,
            "causal_graph": self.causal_graph,
        }
        for field, value in values.items():
            if not str(value).strip():
                raise ValueError(f"DoWhy identification contract requires {field}")
        if self.treatment == self.outcome:
            raise ValueError("DoWhy treatment and outcome must differ")
        causes = tuple(cause.strip() for cause in self.observed_common_causes)
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

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "easyicu.dowhy_identification_receipt/1",
            "status": self.status,
            "adapter_version": self.adapter_version,
            "issue_code": self.issue_code,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "graph_sha256": self.graph_sha256,
            "identifier": self.identifier,
            "effect_estimate_present": False,
        }


def _graph_sha256(graph: str) -> str:
    return hashlib.sha256(graph.encode("utf-8")).hexdigest()


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
        )
    runtime = probe_external_adapter("dowhy_identification_v1")
    if not runtime.available:
        return DoWhyIdentificationReceipt(
            status="adapter_unavailable",
            adapter_version=None,
            issue_code=runtime.issue_code,
            treatment=contract.treatment,
            outcome=contract.outcome,
            graph_sha256=graph_sha256,
            identifier=contract.identifier,
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
        )
    return DoWhyIdentificationReceipt(
        status="identified" if identified_estimand is not None else "not_identified",
        adapter_version=runtime.installed_version,
        issue_code=(None if identified_estimand is not None else "causal_effect_not_identified"),
        treatment=contract.treatment,
        outcome=contract.outcome,
        graph_sha256=graph_sha256,
        identifier=contract.identifier,
    )


__all__ = [
    "DoWhyIdentificationContract",
    "DoWhyIdentificationReceipt",
    "identify_declared_causal_effect",
]
