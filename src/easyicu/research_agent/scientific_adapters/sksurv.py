"""scikit-survival adapter for declared competing-risks CIF curves.

The adapter is deliberately limited to non-parametric cumulative incidence.
It will not present a cause-naive Cox model as a competing-risks result, infer
event meanings, or select a time origin.  Those coordinates are all explicit
in the caller's typed plan and future family result contract.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

import numpy as np

from .runtime import probe_external_adapter


@dataclass(frozen=True)
class CompetingRisksCIFContract:
    """Declared endpoint coding for one cumulative-incidence calculation."""

    time_column: str
    event_column: str
    event_of_interest: int
    competing_event_codes: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.time_column.strip() or not self.event_column.strip():
            raise ValueError("competing-risks CIF requires time and event columns")
        if self.time_column == self.event_column:
            raise ValueError("competing-risks time and event columns must differ")
        if self.event_of_interest <= 0:
            raise ValueError("competing-risks event of interest must be positive")
        if not self.competing_event_codes:
            raise ValueError("competing-risks CIF requires declared competing events")
        if any(code <= 0 for code in self.competing_event_codes):
            raise ValueError("competing-risks event codes must be positive")
        if self.event_of_interest in self.competing_event_codes:
            raise ValueError("event of interest cannot also be a competing event")
        if len(self.competing_event_codes) != len(set(self.competing_event_codes)):
            raise ValueError("competing-risks event codes must be unique")


@dataclass(frozen=True)
class CompetingRisksCIFResult:
    """Portable curve result; persistence still belongs to EvidenceStore."""

    status: Literal["estimated", "input_contract_failed", "adapter_unavailable"]
    adapter_version: Optional[str]
    issue_code: Optional[str]
    event_of_interest: int
    event_code_mapping: Tuple[Tuple[int, int], ...]
    n_observations: int
    n_events_of_interest: int
    time: Tuple[float, ...] = ()
    cumulative_incidence: Tuple[float, ...] = ()

    def to_rows(self) -> list[dict[str, object]]:
        """Return a typed table ready for a caller-owned evidence artifact."""

        return [
            {
                "time": time,
                "cumulative_incidence": incidence,
                "event_of_interest": self.event_of_interest,
            }
            for time, incidence in zip(self.time, self.cumulative_incidence)
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "easyicu.sksurv_competing_risks_cif/1",
            "status": self.status,
            "adapter_version": self.adapter_version,
            "issue_code": self.issue_code,
            "event_of_interest": self.event_of_interest,
            "event_code_mapping": [list(pair) for pair in self.event_code_mapping],
            "n_observations": self.n_observations,
            "n_events_of_interest": self.n_events_of_interest,
            "curve_points": len(self.time),
        }


def _input_failure(
    contract: CompetingRisksCIFContract,
    *,
    issue_code: str,
) -> CompetingRisksCIFResult:
    return CompetingRisksCIFResult(
        status="input_contract_failed",
        adapter_version=None,
        issue_code=issue_code,
        event_of_interest=contract.event_of_interest,
        event_code_mapping=(),
        n_observations=0,
        n_events_of_interest=0,
    )


def estimate_declared_cumulative_incidence(
    dataframe: Any,
    contract: CompetingRisksCIFContract,
) -> CompetingRisksCIFResult:
    """Estimate CIF only when every event code is explicit and observable."""

    try:
        times = np.asarray(dataframe[contract.time_column], dtype=float)
        source_events = np.asarray(dataframe[contract.event_column], dtype=float)
    except (KeyError, TypeError, ValueError):
        return _input_failure(
            contract,
            issue_code="competing_risk_input_columns_unresolved",
        )
    if (
        times.ndim != 1
        or source_events.ndim != 1
        or len(times) == 0
        or len(times) != len(source_events)
        or not np.isfinite(times).all()
        or not np.isfinite(source_events).all()
        or (times < 0).any()
    ):
        return _input_failure(
            contract,
            issue_code="competing_risk_time_or_event_values_invalid",
        )
    if not np.equal(source_events, np.rint(source_events)).all():
        return _input_failure(
            contract,
            issue_code="competing_risk_event_codes_not_integral",
        )
    event_codes = source_events.astype(int)
    if (event_codes < 0).any():
        return _input_failure(
            contract,
            issue_code="competing_risk_event_codes_invalid",
        )
    declared_risks = (contract.event_of_interest, *contract.competing_event_codes)
    observed_positive = set(event_codes[event_codes > 0])
    if observed_positive != set(declared_risks):
        return _input_failure(
            contract,
            issue_code="competing_risk_event_codes_do_not_match_contract",
        )

    # scikit-survival requires consecutive 1..n risk labels.  Remapping is
    # mechanical and receipt-bound, never an inference about the event type.
    ordered_source_codes = tuple(sorted(declared_risks))
    code_mapping = {
        source_code: runtime_code
        for runtime_code, source_code in enumerate(ordered_source_codes, start=1)
    }
    runtime_events = np.asarray(
        [code_mapping.get(code, 0) for code in event_codes],
        dtype=int,
    )
    runtime = probe_external_adapter("sksurv_competing_risks_cif_v1")
    mapping_receipt = tuple(sorted(code_mapping.items()))
    n_interest = int((event_codes == contract.event_of_interest).sum())
    if not runtime.available:
        return CompetingRisksCIFResult(
            status="adapter_unavailable",
            adapter_version=None,
            issue_code=runtime.issue_code,
            event_of_interest=contract.event_of_interest,
            event_code_mapping=mapping_receipt,
            n_observations=len(times),
            n_events_of_interest=n_interest,
        )
    nonparametric = importlib.import_module("sksurv.nonparametric")
    try:
        output = nonparametric.cumulative_incidence_competing_risks(
            runtime_events,
            times,
            conf_type=None,
        )
        curve_time, all_risk_incidence = output[:2]
    except ValueError:
        return _input_failure(
            contract,
            issue_code="competing_risk_estimator_rejected_declared_input",
        )
    runtime_interest_code = code_mapping[contract.event_of_interest]
    curve = np.asarray(all_risk_incidence[runtime_interest_code], dtype=float)
    return CompetingRisksCIFResult(
        status="estimated",
        adapter_version=runtime.installed_version,
        issue_code=None,
        event_of_interest=contract.event_of_interest,
        event_code_mapping=mapping_receipt,
        n_observations=len(times),
        n_events_of_interest=n_interest,
        time=tuple(float(value) for value in curve_time),
        cumulative_incidence=tuple(float(value) for value in curve),
    )


__all__ = [
    "CompetingRisksCIFContract",
    "CompetingRisksCIFResult",
    "estimate_declared_cumulative_incidence",
]
