"""Owner: the browser projection of a static prediction run's primary performance.

The host-owned static prediction executor (``contracts/prediction_execution``)
writes one row of ``table:model_performance`` for the plan's primary
prediction step.  This owner projects that row, and only that row, as a typed
block for the run answer.  The row is bound as the kernel's typed binding
binds a table product:

- the plan declares exactly one producer of the product, and it is the
  primary step of the static prediction action;
- the producer's latest ledger record is successful, and its typed
  ``output_files`` names the product's file;
- the evidence record is one of that record's evidence ids, produced by the
  step, registered as ``<evidence_id>__<file>`` and digest-verified.

No other table is read and no column is chosen by a name pattern.  A run where
any link is missing, or whose row is not a single complete row, gets no block.

Authority is the host's: the gate status the projection already computed and
the readiness axis ``paper_authorized``.  The executor's self-declared
authority columns are not copied.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, get_args

from easyicu.research_agent.authority.runtime_artifacts import (
    current_step_records,
    verified_run_evidence_path,
)
from easyicu.research_agent.contracts.prediction_execution import (
    PREDICTION_PERFORMANCE_PRODUCT,
    PREDICTION_PRIMARY_ACTION,
    STATIC_PREDICTION_ACTION_OUTPUTS,
)
from easyicu.research_agent.contracts.prediction_validation import CalibrationStatus

PREDICTION_PERFORMANCE_SCHEMA_VERSION = "easyicu.web-prediction-performance/1"
#: The analysis types (``planning/analysis_types.py``) of prediction studies.
PREDICTION_ANALYSIS_TYPES = frozenset({"prediction_model", "dynamic_prediction"})
_MAX_PERFORMANCE_BYTES = 1_500_000
_CALIBRATION_STATUSES = frozenset(get_args(CalibrationStatus))


def _plan_steps(plan: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    steps = plan.get("steps") if isinstance(plan, Mapping) else None
    return [step for step in steps or [] if isinstance(step, Mapping)]


def _step_id(step: Mapping[str, Any]) -> str:
    return str(step.get("step_id") or "").strip()


def prediction_result_step_ids(plan: Mapping[str, Any]) -> FrozenSet[str]:
    """The steps of a prediction study that declare a static prediction action."""

    if not isinstance(plan, Mapping):
        return frozenset()
    if str(plan.get("analysis_type") or "") not in PREDICTION_ANALYSIS_TYPES:
        return frozenset()
    return frozenset(
        _step_id(step)
        for step in _plan_steps(plan)
        if _step_id(step)
        and str(step.get("scientific_action_id") or "")
        in STATIC_PREDICTION_ACTION_OUTPUTS
    )


def _primary_producer(plan: Mapping[str, Any]) -> Optional[str]:
    """The one step declaring the product, when it is the primary prediction."""

    declaring = [
        step
        for step in _plan_steps(plan)
        if PREDICTION_PERFORMANCE_PRODUCT
        in [str(value or "").strip() for value in step.get("expected_outputs") or []]
    ]
    if len(declaring) != 1:
        return None
    step = declaring[0]
    if (
        step.get("planned_analysis_role") != "primary"
        or str(step.get("scientific_action_id") or "") != PREDICTION_PRIMARY_ACTION
    ):
        return None
    return _step_id(step) or None


def _producer_record(run_dir: Path, step_id: str) -> Optional[Mapping[str, Any]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    records = (
        manifest.get("per_step_records") if isinstance(manifest, Mapping) else None
    )
    latest = {
        str(record.get("step_id") or "").strip(): record
        for record in current_step_records(records if isinstance(records, list) else [])
    }
    record = latest.get(step_id)
    if record is None or str(record.get("status") or "").strip().lower() != "ok":
        return None
    return record


def _declared_filename(record: Mapping[str, Any]) -> Optional[str]:
    summary = record.get("step_summary")
    files = summary.get("output_files") if isinstance(summary, Mapping) else None
    value = (
        files.get(PREDICTION_PERFORMANCE_PRODUCT)
        if isinstance(files, Mapping)
        else None
    )
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value.strip()).name or None


def _bound_evidence(
    run_dir: Path, record: Mapping[str, Any], step_id: str, filename: str
) -> Optional[tuple[str, Path]]:
    active = {
        str(value).strip()
        for value in record.get("evidence_ids") or []
        if str(value).strip()
    }
    index = json.loads(
        (run_dir / "evidence" / "evidence_index.json").read_text(encoding="utf-8")
    )
    candidates = []
    for item in index if isinstance(index, list) else []:
        if not isinstance(item, Mapping):
            continue
        evidence_id = str(item.get("evidence_id") or "").strip()
        if (
            evidence_id not in active
            or str(item.get("produced_by_step") or "").strip() != step_id
            or str(item.get("kind") or "") != "table"
        ):
            continue
        path = verified_run_evidence_path(run_dir, item)
        # The evidence store registers a step file as ``<evidence_id>__<file>``.
        if path is None or path.name != f"{evidence_id}__{filename}":
            continue
        candidates.append((evidence_id, path))
    return candidates[0] if len(candidates) == 1 else None


def _single_row(path: Path) -> Optional[Dict[str, str]]:
    if path.stat().st_size > _MAX_PERFORMANCE_BYTES:
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if len(rows) == 1 else None


def _number(
    row: Mapping[str, Any],
    name: str,
    *,
    low: Optional[float] = None,
    high: Optional[float] = None,
) -> Optional[float]:
    try:
        value = float(str(row.get(name) or "").strip())
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    if (low is not None and value < low) or (high is not None and value > high):
        return None
    return value


def _count(row: Mapping[str, Any], name: str) -> Optional[int]:
    value = _number(row, name, low=0)
    if value is None or not value.is_integer():
        return None
    return int(value)


def _performance_fields(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The row's values, or None when a required value is absent or invalid."""

    fields: Dict[str, Any] = {
        "model": str(row.get("model") or "").strip()[:80],
        "development_n": _count(row, "development_n"),
        "validation_n": _count(row, "validation_n"),
        "validation_subject_n": _count(row, "validation_subject_n"),
        "validation_event_n": _count(row, "validation_event_n"),
        "validation_event_rate": _number(row, "validation_event_rate", low=0, high=1),
        "auroc": _number(row, "auroc", low=0, high=1),
        "auroc_ci_low": _number(row, "auroc_ci_low", low=0, high=1),
        "auroc_ci_high": _number(row, "auroc_ci_high", low=0, high=1),
        "auroc_ci_method": str(row.get("auroc_ci_method") or "").strip()[:80],
        "average_precision": _number(row, "average_precision", low=0, high=1),
        "brier_score": _number(row, "brier_score", low=0, high=1),
        "calibration_status": str(row.get("calibration_status") or "").strip(),
    }
    if any(value in (None, "") for value in fields.values()):
        return None
    if fields["calibration_status"] not in _CALIBRATION_STATUSES:
        return None
    if not fields["auroc_ci_low"] <= fields["auroc"] <= fields["auroc_ci_high"]:
        return None
    # The validation set's patients and events cannot outnumber its rows.
    if not (
        0 < fields["validation_subject_n"] <= fields["validation_n"]
        and fields["validation_event_n"] <= fields["validation_n"]
    ):
        return None
    intercept = _number(row, "calibration_intercept")
    slope = _number(row, "calibration_slope")
    if fields["calibration_status"] == "estimated":
        if intercept is None or slope is None:
            return None
    else:
        intercept = slope = None
    repeated_n = _count(row, "repeated_split_n")
    repeated_mean = _number(row, "repeated_split_auroc_mean", low=0, high=1)
    repeated_sd = _number(row, "repeated_split_auroc_sd", low=0)
    return {
        **fields,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "repeated_split_n": repeated_n,
        "repeated_split_auroc_mean": repeated_mean if repeated_n else None,
        "repeated_split_auroc_sd": repeated_sd if repeated_n else None,
    }


def prediction_performance_projection(
    run_dir: Path,
    *,
    plan: Mapping[str, Any],
    gate_status: str,
    paper_authorized: bool,
) -> Optional[Dict[str, Any]]:
    """The primary model's registered performance, or None when it cannot be bound."""

    step_id = _primary_producer(plan if isinstance(plan, Mapping) else {})
    if step_id is None:
        return None
    try:
        record = _producer_record(run_dir, step_id)
        filename = _declared_filename(record) if record is not None else None
        bound = (
            _bound_evidence(run_dir, record, step_id, filename)
            if record is not None and filename
            else None
        )
        row = _single_row(bound[1]) if bound is not None else None
    except (OSError, UnicodeDecodeError, ValueError, csv.Error):
        return None
    fields = _performance_fields(row) if row is not None else None
    if bound is None or fields is None:
        return None
    return {
        "schema_version": PREDICTION_PERFORMANCE_SCHEMA_VERSION,
        "product": PREDICTION_PERFORMANCE_PRODUCT,
        "step_id": step_id,
        "evidence_id": bound[0],
        "authority_scope": str(gate_status or ""),
        "paper_authorization_allowed": paper_authorized is True,
        **fields,
    }


__all__ = [
    "PREDICTION_ANALYSIS_TYPES",
    "PREDICTION_PERFORMANCE_SCHEMA_VERSION",
    "prediction_performance_projection",
    "prediction_result_step_ids",
]
