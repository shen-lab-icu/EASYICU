"""Deterministic replay of plan-locked cohort membership predicates.

This is a leaf statistical kernel: it recomputes membership counts for the
locked robustness cohort variants from the pre-filter universe, without
choosing an estimand, exposure, outcome, or cohort. It lives in
``robustness/`` (not ``execution/``) because both the read-only contract gate
and the auxiliary robustness runner consume it — a gate must never depend on
the execution layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..cohort.schema import build_cohort
from .estimators import _data_with_predicate_aliases
from .panel import PRIMARY_SPEC_ID, RobustnessSpec

__all__ = [
    "replay_locked_memberships",
]


def _identifier_column(data: Any) -> Optional[str]:
    columns = set(str(column) for column in getattr(data, "columns", []))
    return next(
        (
            column
            for column in ("stay_id", "icustay_id", "icu_stay_id")
            if column in columns
        ),
        None,
    )


def _membership_audit(
    *,
    specs: Sequence[RobustnessSpec],
    cohort: Any,
    universe: Any,
    context: Any = None,
    exposure: str = "",
) -> List[Dict[str, Any]]:
    if cohort is None:
        return []
    primary_n = int(len(cohort))
    universe_n = int(len(universe)) if universe is not None else primary_n
    rows: List[Dict[str, Any]] = [
        {
            "spec_id": PRIMARY_SPEC_ID,
            "axis": "primary",
            "membership_source": "analysis_cohort",
            "universe_n": universe_n,
            "primary_membership_n": primary_n,
            "variant_membership_n": primary_n,
            "overlap_n": primary_n,
            "inflow_n": 0,
            "outflow_n": 0,
            "membership_delta_n": 0,
            "membership_executable": True,
            "notes": "Primary membership from COHORT_PARQUET.",
        }
    ]
    id_col = _identifier_column(cohort)
    primary_ids = set(cohort[id_col].dropna()) if id_col else None
    for spec in specs:
        if spec.axis != "cohort":
            rows.append(
                {
                    "spec_id": spec.spec_id,
                    "axis": spec.axis,
                    "membership_source": "analysis_cohort",
                    "universe_n": universe_n,
                    "primary_membership_n": primary_n,
                    "variant_membership_n": primary_n,
                    "overlap_n": primary_n,
                    "inflow_n": 0,
                    "outflow_n": 0,
                    "membership_delta_n": 0,
                    "membership_executable": True,
                    "notes": "This axis retains the primary cohort membership.",
                }
            )
            continue
        if universe is None:
            rows.append(
                {
                    "spec_id": spec.spec_id,
                    "axis": spec.axis,
                    "membership_source": "unavailable",
                    "universe_n": None,
                    "primary_membership_n": primary_n,
                    "variant_membership_n": None,
                    "overlap_n": None,
                    "inflow_n": None,
                    "outflow_n": None,
                    "membership_delta_n": None,
                    "membership_executable": False,
                    "notes": (
                        "Cohort membership variant was not executed because the "
                        "pre-filter universe was unavailable."
                    ),
                }
            )
            continue
        try:
            if spec.cohort_override is None:
                raise ValueError("cohort_override is missing")
            data_for_filter = _data_with_predicate_aliases(
                data=universe,
                cohort_definition=spec.cohort_override,
                exposure=exposure,
                context=context,
            )
            variant = build_cohort(spec.cohort_override, data=data_for_filter)
            variant_n = int(len(variant))
            variant_id_col = _identifier_column(variant)
            if primary_ids is not None and variant_id_col == id_col:
                variant_ids = set(variant[id_col].dropna())
                inflow_n = len(variant_ids - primary_ids)
                outflow_n = len(primary_ids - variant_ids)
                overlap_n = primary_n - outflow_n
            else:
                inflow_n = None
                outflow_n = None
                overlap_n = None
            rows.append(
                {
                    "spec_id": spec.spec_id,
                    "axis": spec.axis,
                    "membership_source": "universe",
                    "universe_n": universe_n,
                    "primary_membership_n": primary_n,
                    "variant_membership_n": variant_n,
                    "overlap_n": overlap_n,
                    "inflow_n": inflow_n,
                    "outflow_n": outflow_n,
                    "membership_delta_n": variant_n - primary_n,
                    "membership_executable": True,
                    "notes": "Membership recomputed from the locked cohort override.",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "spec_id": spec.spec_id,
                    "axis": spec.axis,
                    "membership_source": "universe",
                    "universe_n": universe_n,
                    "primary_membership_n": primary_n,
                    "variant_membership_n": None,
                    "overlap_n": None,
                    "inflow_n": None,
                    "outflow_n": None,
                    "membership_delta_n": None,
                    "membership_executable": False,
                    "notes": f"Locked cohort override was not executable: {exc}",
                }
            )
    return rows


def replay_locked_memberships(
    *,
    specs: Sequence[RobustnessSpec],
    cohort: Any,
    universe: Any,
    context: Any = None,
    exposure: str = "",
) -> List[Dict[str, Any]]:
    """Replay plan-locked membership predicates without choosing an estimand."""

    return _membership_audit(
        specs=specs,
        cohort=cohort,
        universe=universe,
        context=context,
        exposure=exposure,
    )
