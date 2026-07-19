"""[Layer 5: Evaluation & Submission Scaffold] Cross-model concordance audit.

This module does not run frontier LLMs and does not adjudicate primary
estimates. It compares already-produced plans and robustness panels so a later
multi-backend pilot can report plan-level and panel-primary concordance without
inventing new analysis logic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from pydantic import BaseModel

from ..robustness_panel import RobustnessPanel
from ..schema import AnalysisPlan


@dataclass(frozen=True)
class BackendIdentity:
    name: str
    llm_provider: str
    llm_model: str
    prompt_pack_version: str
    env_overrides: Dict[str, str]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BackendIdentity":
        return cls(
            name=str(data.get("name") or "").strip(),
            llm_provider=str(data.get("llm_provider") or "mock").strip(),
            llm_model=str(data.get("llm_model") or "mock").strip(),
            prompt_pack_version=str(data.get("prompt_pack_version") or "unknown").strip(),
            env_overrides={
                str(k): str(v) for k, v in dict(data.get("env_overrides") or {}).items()
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FieldDisagreement:
    field_path: str
    values_by_backend: Dict[str, Any]
    all_agree: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_path": self.field_path,
            "values_by_backend": _json_ready(self.values_by_backend),
            "all_agree": self.all_agree,
        }


@dataclass(frozen=True)
class PlanConcordance:
    backends: tuple[BackendIdentity, ...]
    field_disagreements: tuple[FieldDisagreement, ...]
    overall_agreement_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backends": [b.to_dict() for b in self.backends],
            "field_disagreements": [d.to_dict() for d in self.field_disagreements],
            "overall_agreement_rate": self.overall_agreement_rate,
        }


@dataclass(frozen=True)
class PanelPrimaryConcordance:
    backends: tuple[BackendIdentity, ...]
    primary_estimates_by_backend: Dict[str, Optional[float]]
    primary_ci_low_by_backend: Dict[str, Optional[float]]
    primary_ci_high_by_backend: Dict[str, Optional[float]]
    range_low: Optional[float]
    range_high: Optional[float]
    backends_concordant: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backends": [b.to_dict() for b in self.backends],
            "primary_estimates_by_backend": self.primary_estimates_by_backend,
            "primary_ci_low_by_backend": self.primary_ci_low_by_backend,
            "primary_ci_high_by_backend": self.primary_ci_high_by_backend,
            "range_low": self.range_low,
            "range_high": self.range_high,
            "backends_concordant": self.backends_concordant,
        }


def compare_plans(
    plans: Mapping[str, AnalysisPlan],
    *,
    backends: Optional[Mapping[str, BackendIdentity]] = None,
) -> PlanConcordance:
    """Compare selected audit-facing fields across backend plans."""

    if len(plans) < 2:
        raise ValueError("cross-model plan concordance requires at least two plans")
    backend_tuple = _backend_tuple(plans, backends)
    field_values = {
        field_path: {
            backend_name: _json_ready(value)
            for backend_name, value in values_by_backend.items()
        }
        for field_path, values_by_backend in _collect_plan_fields(plans).items()
    }
    disagreements = []
    agreed = 0
    for field_path in sorted(field_values):
        values_by_backend = field_values[field_path]
        all_agree = _all_values_equal(values_by_backend.values())
        agreed += int(all_agree)
        disagreements.append(
            FieldDisagreement(
                field_path=field_path,
                values_by_backend=values_by_backend,
                all_agree=all_agree,
            )
        )
    rate = agreed / len(disagreements) if disagreements else 1.0
    return PlanConcordance(
        backends=backend_tuple,
        field_disagreements=tuple(disagreements),
        overall_agreement_rate=rate,
    )


def compare_panel_primaries(
    panels: Mapping[str, RobustnessPanel],
    *,
    backends: Optional[Mapping[str, BackendIdentity]] = None,
) -> PanelPrimaryConcordance:
    """Compare the primary robustness-panel row across backends."""

    if len(panels) < 2:
        raise ValueError("cross-model panel concordance requires at least two panels")
    backend_tuple = _backend_tuple(panels, backends)
    estimates: Dict[str, Optional[float]] = {}
    lows: Dict[str, Optional[float]] = {}
    highs: Dict[str, Optional[float]] = {}
    for name, panel in panels.items():
        row = next((r for r in panel.rows if r.spec_id == panel.primary_spec_id), None)
        estimates[name] = row.point_estimate if row is not None else None
        lows[name] = row.ci_low if row is not None else None
        highs[name] = row.ci_high if row is not None else None
    finite_lows = [v for v in lows.values() if v is not None]
    finite_highs = [v for v in highs.values() if v is not None]
    return PanelPrimaryConcordance(
        backends=backend_tuple,
        primary_estimates_by_backend=estimates,
        primary_ci_low_by_backend=lows,
        primary_ci_high_by_backend=highs,
        range_low=min(finite_lows) if finite_lows else None,
        range_high=max(finite_highs) if finite_highs else None,
        backends_concordant=_all_intervals_overlap(lows, highs),
    )


def write_cross_model_report(
    run_dir: Path | str,
    plan_conc: PlanConcordance,
    panel_conc: Optional[PanelPrimaryConcordance] = None,
) -> Path:
    """Write ``cross_model_report.json`` and return its path."""

    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cross_model_report.json"
    payload: Dict[str, Any] = {
        "plan_concordance": plan_conc.to_dict(),
        "panel_primary_concordance": (
            panel_conc.to_dict() if panel_conc is not None else None
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _backend_tuple(
    keyed: Mapping[str, Any],
    backends: Optional[Mapping[str, BackendIdentity]],
) -> tuple[BackendIdentity, ...]:
    if backends:
        return tuple(backends[name] for name in keyed)
    return tuple(
        BackendIdentity(
            name=name,
            llm_provider="unknown",
            llm_model="unknown",
            prompt_pack_version="unknown",
            env_overrides={},
        )
        for name in keyed
    )


def _collect_plan_fields(
    plans: Mapping[str, AnalysisPlan],
) -> Dict[str, Dict[str, Any]]:
    fields: Dict[str, Dict[str, Any]] = {}
    for backend_name, plan in plans.items():
        values = _plan_field_values(plan)
        for field_path, value in values.items():
            fields.setdefault(field_path, {})[backend_name] = value
    return fields


def _plan_field_values(plan: AnalysisPlan) -> Dict[str, Any]:
    cohort = plan.cohort
    values: Dict[str, Any] = {}
    if cohort is None:
        values["cohort"] = None
        return values
    values["cohort.derived_from_named"] = cohort.derived_from_named
    values["cohort.inclusion.concept_ids"] = sorted(
        p.concept_id for p in cohort.inclusion
    )
    values["cohort.inclusion.time_windows"] = sorted(
        (p.concept_id, p.time_window.anchor, p.time_window.start_offset_hours, p.time_window.end_offset_hours)
        for p in cohort.inclusion
    )
    values["cohort.inclusion.aggregations"] = sorted(
        (p.concept_id, p.aggregation) for p in cohort.inclusion
    )
    values["cohort.exclusion.concept_ids"] = sorted(
        p.concept_id for p in cohort.exclusion
    )
    values["cohort.exclusion.time_windows"] = sorted(
        (p.concept_id, p.time_window.anchor, p.time_window.start_offset_hours, p.time_window.end_offset_hours)
        for p in cohort.exclusion
    )
    values["cohort.exclusion.aggregations"] = sorted(
        (p.concept_id, p.aggregation) for p in cohort.exclusion
    )
    values["robustness.outcome_overrides"] = sorted(
        (s.spec_id, _json_ready(s.outcome_override)) for s in plan.robustness_specs
    )
    values["robustness.missing_overrides"] = sorted(
        (s.spec_id, _json_ready(s.missing_override)) for s in plan.robustness_specs
    )
    return values


def _all_values_equal(values: Sequence[Any] | Any) -> bool:
    as_list = list(values)
    if not as_list:
        return True
    first = as_list[0]
    return all(value == first for value in as_list[1:])


def _all_intervals_overlap(
    lows: Mapping[str, Optional[float]],
    highs: Mapping[str, Optional[float]],
) -> bool:
    names = list(lows)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            left_low, left_high = lows[left], highs[left]
            right_low, right_high = lows[right], highs[right]
            if None in {left_low, left_high, right_low, right_high}:
                return False
            if left_high < right_low or right_high < left_low:  # type: ignore[operator]
                return False
    return True


def _json_ready(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


__all__ = [
    "BackendIdentity",
    "FieldDisagreement",
    "PanelPrimaryConcordance",
    "PlanConcordance",
    "compare_panel_primaries",
    "compare_plans",
    "write_cross_model_report",
]
