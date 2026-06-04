"""[Layer 4: Evidence & Provenance] Pre-specified robustness panel.

The robustness panel is a disclosure artefact, not a discovery engine.
Alternative specifications are locked before execution and the final panel
summarises their estimates without asking an LLM to decide whether the
findings are surprising, important, or worth promoting.

Non-goals:
- Do not use the panel to ensemble or replace the primary estimate.
- Do not let an LLM judge which variant is worth discussing.
- Do not promote robustness variants into primary analyses without replanning.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

from .cohort_schema import CohortDefinition, coerce_cohort_definition

RobustnessAxis = Literal["cohort", "missing", "outcome"]

MIN_AXIS_COUNTS: Dict[str, int] = {"cohort": 3, "missing": 2, "outcome": 2}

PRIMARY_SPEC_ID = "primary"
LOCK_FILENAME = "robustness_specs_locked.json"
PANEL_FILENAME = "robustness_panel.json"


@dataclass(frozen=True)
class RobustnessSpec:
    spec_id: str
    axis: RobustnessAxis
    description: str
    cohort_override: Optional[CohortDefinition] = None
    missing_override: Optional[Dict[str, Any]] = None
    outcome_override: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "axis": self.axis,
            "description": self.description,
            "cohort_override": (
                self.cohort_override.to_dict() if self.cohort_override is not None else None
            ),
            "missing_override": self.missing_override,
            "outcome_override": self.outcome_override,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobustnessSpec":
        return cls(
            spec_id=str(data.get("spec_id") or "").strip(),
            axis=str(data.get("axis") or "").strip(),  # type: ignore[arg-type]
            description=str(data.get("description") or "").strip(),
            cohort_override=coerce_cohort_definition(data.get("cohort_override")),
            missing_override=_dict_or_none(data.get("missing_override")),
            outcome_override=_dict_or_none(data.get("outcome_override")),
        )


@dataclass(frozen=True)
class RobustnessPanelRow:
    spec_id: str
    axis: str
    n: int
    point_estimate: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    se: Optional[float]
    evidence_id: str
    converged: bool
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobustnessPanelRow":
        return cls(
            spec_id=str(data.get("spec_id") or "").strip(),
            axis=str(data.get("axis") or "").strip(),
            n=int(data.get("n") or 0),
            point_estimate=_optional_float(data.get("point_estimate")),
            ci_low=_optional_float(data.get("ci_low")),
            ci_high=_optional_float(data.get("ci_high")),
            se=_optional_float(data.get("se")),
            evidence_id=str(data.get("evidence_id") or "").strip(),
            converged=bool(data.get("converged")),
            notes=str(data.get("notes") or ""),
        )


@dataclass(frozen=True)
class RobustnessPanel:
    primary_spec_id: str
    rows: tuple[RobustnessPanelRow, ...]
    range_low: Optional[float]
    range_high: Optional[float]
    n_variants: int
    locked_at: str

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[RobustnessPanelRow],
        *,
        primary_spec_id: str = PRIMARY_SPEC_ID,
        locked_at: Optional[str] = None,
    ) -> "RobustnessPanel":
        row_tuple = tuple(rows)
        converged_lows = [
            r.ci_low for r in row_tuple if r.converged and r.ci_low is not None
        ]
        converged_highs = [
            r.ci_high for r in row_tuple if r.converged and r.ci_high is not None
        ]
        return cls(
            primary_spec_id=primary_spec_id,
            rows=row_tuple,
            range_low=min(converged_lows) if converged_lows else None,
            range_high=max(converged_highs) if converged_highs else None,
            n_variants=sum(1 for r in row_tuple if r.spec_id != primary_spec_id),
            locked_at=locked_at or datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> Dict[str, Any]:
        primary = next(
            (row for row in self.rows if row.spec_id == self.primary_spec_id),
            None,
        )
        return {
            "primary_spec_id": self.primary_spec_id,
            "rows": [row.to_dict() for row in self.rows],
            "row_count": len(self.rows),
            "primary_point_estimate": (
                primary.point_estimate if primary is not None else None
            ),
            "range_low": self.range_low,
            "range_high": self.range_high,
            "n_variants": self.n_variants,
            "locked_at": self.locked_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobustnessPanel":
        return cls.from_rows(
            [RobustnessPanelRow.from_dict(r) for r in data.get("rows", []) or []],
            primary_spec_id=str(data.get("primary_spec_id") or PRIMARY_SPEC_ID),
            locked_at=str(data.get("locked_at") or datetime.now(timezone.utc).isoformat()),
        )


class RobustnessPlanError(ValueError):
    """Raised when robustness specifications are missing or not locked."""


def default_robustness_specs() -> List[RobustnessSpec]:
    """Return case-neutral fallback specs used when a planner omits the field."""

    return [
        RobustnessSpec(
            spec_id="alt_cohort_author_defined_1",
            axis="cohort",
            description=(
                "Caller-defined alternative cohort 1. The framework fallback is "
                "case-neutral and carries no built-in clinical predicate."
            ),
            cohort_override=CohortDefinition(name="alt_cohort_author_defined_1"),
        ),
        RobustnessSpec(
            spec_id="alt_cohort_author_defined_2",
            axis="cohort",
            description=(
                "Caller-defined alternative cohort 2. Register concrete predicates "
                "in the case protocol before paper-facing runs."
            ),
            cohort_override=CohortDefinition(name="alt_cohort_author_defined_2"),
        ),
        RobustnessSpec(
            spec_id="alt_cohort_author_defined_3",
            axis="cohort",
            description=(
                "Caller-defined alternative cohort 3. This placeholder keeps the "
                "audit path active without hard-coding a benchmark case."
            ),
            cohort_override=CohortDefinition(name="alt_cohort_author_defined_3"),
        ),
        RobustnessSpec(
            spec_id="alt_missing_complete_case",
            axis="missing",
            description="Restrict the model to complete cases for required variables.",
            missing_override={"strategy": "complete_case"},
        ),
        RobustnessSpec(
            spec_id="alt_missing_median_impute",
            axis="missing",
            description="Use median imputation for continuous predictors.",
            missing_override={"strategy": "median_imputation"},
        ),
        RobustnessSpec(
            spec_id="alt_outcome_author_defined_1",
            axis="outcome",
            description=(
                "Caller-defined alternative outcome 1. Register the concrete "
                "endpoint or estimand in the case protocol before paper-facing "
                "runs; the fallback carries no built-in clinical outcome."
            ),
            outcome_override={"target": "author_defined_outcome_1"},
        ),
        RobustnessSpec(
            spec_id="alt_outcome_author_defined_2",
            axis="outcome",
            description=(
                "Caller-defined alternative outcome 2. This placeholder keeps "
                "the robustness contract active without substituting a built-in "
                "endpoint, time horizon, or event definition."
            ),
            outcome_override={"target": "author_defined_outcome_2"},
        ),
    ]


def validate_robustness_specs(specs: Sequence[RobustnessSpec]) -> None:
    counts = {axis: 0 for axis in MIN_AXIS_COUNTS}
    seen_ids: set[str] = set()
    problems: List[str] = []
    for spec in specs:
        if not spec.spec_id:
            problems.append("spec_id must be non-empty")
        if spec.spec_id in seen_ids:
            problems.append(f"duplicate spec_id: {spec.spec_id}")
        seen_ids.add(spec.spec_id)
        if spec.axis not in counts:
            problems.append(f"unknown robustness axis for {spec.spec_id}: {spec.axis}")
            continue
        counts[spec.axis] += 1
    for axis, minimum in MIN_AXIS_COUNTS.items():
        if counts[axis] < minimum:
            problems.append(
                f"robustness_specs require at least {minimum} {axis} axis spec(s); "
                f"got {counts[axis]}"
            )
    if problems:
        raise RobustnessPlanError("; ".join(problems))


def ensure_robustness_specs(plan: Any) -> Any:
    specs = list(getattr(plan, "robustness_specs", []) or [])
    if specs:
        validate_robustness_specs(specs)
        return plan
    return plan.model_copy(update={"robustness_specs": default_robustness_specs()})


def write_locked_robustness_specs(
    *,
    run_dir: Path,
    plan: Any,
    evidence: Any,
    prompt_pack_version: Optional[str],
    llm_signature: str,
) -> Path:
    specs = list(getattr(plan, "robustness_specs", []) or [])
    validate_robustness_specs(specs)
    payload = {
        "schema_version": "easyicu.robustness_specs/1",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": robustness_specs_sha(specs),
        "specs": [spec.to_dict() for spec in specs],
    }
    path = run_dir / LOCK_FILENAME
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if evidence.get("robustness_specs_locked") is None:
        evidence.register_file(
            kind="log",
            description="Pre-specified robustness specifications locked after planning.",
            source_path=path,
            evidence_id="robustness_specs_locked",
            aliases=["robustness_specs_locked"],
            producer="planner",
            generation_mode="system",
            prompt_pack_version=prompt_pack_version,
            metadata={"llm_signature": llm_signature},
        )
    return path


def assert_robustness_specs_locked(*, run_dir: Path, plan: Any) -> None:
    specs = list(getattr(plan, "robustness_specs", []) or [])
    if not specs:
        return
    path = run_dir / LOCK_FILENAME
    if not path.exists():
        raise RobustnessPlanError("robustness_specs plan locked file is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RobustnessPlanError(f"robustness_specs lock is unreadable: {exc}") from exc
    expected = str(payload.get("spec_sha256") or "")
    observed = robustness_specs_sha(specs)
    if observed != expected:
        raise RobustnessPlanError(
            "robustness_specs changed after plan lock; execute phase refuses "
            "to run an unlocked robustness panel"
        )


def load_locked_robustness_specs(run_dir: Path) -> List[RobustnessSpec]:
    """Load plan-time robustness specs from ``robustness_specs_locked.json``.

    Late replanning can replace the active ``AnalysisPlan`` object and drop the
    `robustness_specs` attribute. The locked file is the durable pre-specified
    panel contract, so execute finalisation can safely fall back to it.
    """

    path = Path(run_dir) / LOCK_FILENAME
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RobustnessPlanError(
            f"robustness_specs lock is unreadable: {exc}"
        ) from exc
    raw_specs = payload.get("specs") or []
    if not isinstance(raw_specs, list):
        raise RobustnessPlanError("robustness_specs lock has invalid specs payload")
    return [
        RobustnessSpec.from_dict(spec)
        for spec in raw_specs
        if isinstance(spec, dict)
    ]


def robustness_specs_for_execution(*, run_dir: Path, plan: Any) -> List[RobustnessSpec]:
    """Return active plan specs, falling back to the plan-time lock."""

    specs = list(getattr(plan, "robustness_specs", []) or [])
    if specs:
        return specs
    return load_locked_robustness_specs(run_dir)


def robustness_specs_sha(specs: Sequence[RobustnessSpec]) -> str:
    payload = [spec.to_dict() for spec in specs]
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_robustness_panel_from_records(
    *,
    specs: Sequence[RobustnessSpec],
    per_step_records: Sequence[Dict[str, Any]],
    adapter_rows: Optional[Sequence[RobustnessPanelRow]] = None,
    locked_at: Optional[str] = None,
) -> RobustnessPanel:
    rows: List[RobustnessPanelRow] = []
    for row in adapter_rows or []:
        rows.append(row)
    existing = {row.spec_id for row in rows}
    primary = _primary_row_from_records(per_step_records)
    if primary is not None and primary.spec_id not in existing:
        rows.append(primary)
        existing.add(primary.spec_id)
    for row in _declared_rows_from_records(per_step_records):
        if row.spec_id in existing:
            continue
        rows.append(row)
        existing.add(row.spec_id)

    for spec in specs:
        if spec.spec_id in existing:
            continue
        rows.append(
            RobustnessPanelRow(
                spec_id=spec.spec_id,
                axis=spec.axis,
                n=0,
                point_estimate=None,
                ci_low=None,
                ci_high=None,
                se=None,
                evidence_id="",
                converged=False,
                notes=(
                    "No executable variant estimate was emitted for this "
                    "pre-specified spec."
                ),
            )
        )
    if not rows:
        rows.append(
            RobustnessPanelRow(
                spec_id=PRIMARY_SPEC_ID,
                axis="primary",
                n=0,
                point_estimate=None,
                ci_low=None,
                ci_high=None,
                se=None,
                evidence_id="",
                converged=False,
                notes="No primary estimate was available.",
            )
        )
    return RobustnessPanel.from_rows(rows, locked_at=locked_at)


def write_robustness_panel(
    *,
    run_dir: Path,
    panel: RobustnessPanel,
    evidence: Any,
    prompt_pack_version: Optional[str],
) -> Path:
    path = run_dir / PANEL_FILENAME
    path.write_text(
        json.dumps(panel.to_dict(), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    record = evidence.register_file(
        kind="statistic",
        description="Pre-specified robustness panel summary.",
        source_path=path,
        evidence_id="robustness_panel",
        aliases=["robustness_panel"],
        producer="pipeline",
        generation_mode="system",
        prompt_pack_version=prompt_pack_version,
        on_sha_change="new_id",
    )
    evidence.register_step_summary_numerics(
        step_id="robustness_panel",
        evidence_id=record.evidence_id,
        summary=numeric_digest_for_panel(panel),
        max_leaves=None,
    )
    return path


def load_robustness_panel(path: Path) -> Optional[RobustnessPanel]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return RobustnessPanel.from_dict(payload)


def numeric_digest_for_panel(panel: RobustnessPanel) -> Dict[str, Any]:
    primary = next((row for row in panel.rows if row.spec_id == panel.primary_spec_id), None)
    digest: Dict[str, Any] = {}
    seen_values: set[tuple[float, int]] = set()

    def _add_unique(key: str, value: Any) -> None:
        if not isinstance(value, (int, float)):
            return
        fvalue = float(value)
        # Collapse duplicate display-equivalent panel values so the manuscript
        # numeric binder does not see five indistinguishable claims for the
        # same point estimate (primary, worst-by-axis, and multiple rows).
        # Keeping the first claim preserves a stable panel-level source while
        # the full row list remains available in robustness_panel.json.
        normalized = (round(fvalue, 12), 0 if isinstance(value, int) else 1)
        if normalized in seen_values:
            return
        seen_values.add(normalized)
        digest[key] = value

    _add_unique("n_variants", panel.n_variants)
    if panel.n_variants > 0 and panel.range_low is not None:
        _add_unique("range_low", panel.range_low)
    if panel.n_variants > 0 and panel.range_high is not None:
        _add_unique("range_high", panel.range_high)
    if primary is not None:
        _add_unique("primary_n", primary.n)
        _add_unique("primary_point_estimate", primary.point_estimate)
        _add_unique("primary_ci_low", primary.ci_low)
        _add_unique("primary_ci_high", primary.ci_high)
    for axis, row in worst_rows_by_axis(panel).items():
        _add_unique(f"worst_{axis}_point_estimate", row.point_estimate)
    return {k: v for k, v in digest.items() if isinstance(v, (int, float))}


def worst_rows_by_axis(panel: RobustnessPanel) -> Dict[str, RobustnessPanelRow]:
    selected: Dict[str, RobustnessPanelRow] = {}
    for row in panel.rows:
        if row.spec_id == panel.primary_spec_id or not row.converged:
            continue
        if row.point_estimate is None:
            continue
        current = selected.get(row.axis)
        if current is None or abs(row.point_estimate) < abs(current.point_estimate or 0.0):
            selected[row.axis] = row
    return selected


def _primary_row_from_records(
    per_step_records: Sequence[Dict[str, Any]]
) -> Optional[RobustnessPanelRow]:
    for record in per_step_records:
        summary = record.get("step_summary")
        if not isinstance(summary, dict):
            continue
        estimate = _first_float(
            summary,
            ("primary_or", "adjusted_or", "odds_ratio", "estimate", "auroc"),
        )
        if estimate is None:
            continue
        ci_low = _first_float(
            summary,
            ("primary_ci_low", "ci_lower", "overall_ci_low"),
        )
        ci_high = _first_float(
            summary,
            ("primary_ci_high", "ci_upper", "overall_ci_high"),
        )
        ci_pair = summary.get("primary_or_ci")
        if isinstance(ci_pair, (list, tuple)) and len(ci_pair) >= 2:
            ci_low = _optional_float(ci_pair[0])
            ci_high = _optional_float(ci_pair[1])
        return RobustnessPanelRow(
            spec_id=PRIMARY_SPEC_ID,
            axis="primary",
            n=int(_first_float(summary, ("n", "n_total", "sample_size")) or 0),
            point_estimate=estimate,
            ci_low=ci_low,
            ci_high=ci_high,
            se=_first_float(summary, ("se", "primary_or_se")),
            evidence_id=str(record.get("step_summary_evidence_id") or ""),
            converged=True,
            notes="Primary analysis estimate.",
        )
    return None


def _declared_rows_from_records(
    per_step_records: Sequence[Dict[str, Any]]
) -> Iterable[RobustnessPanelRow]:
    for record in per_step_records:
        summary = record.get("step_summary")
        if not isinstance(summary, dict):
            continue
        candidates = summary.get("robustness_rows")
        if candidates is None and isinstance(summary.get("robustness_panel"), dict):
            candidates = summary["robustness_panel"].get("rows")
        if not isinstance(candidates, list):
            continue
        for raw in candidates:
            if not isinstance(raw, dict):
                continue
            data = dict(raw)
            data.setdefault("evidence_id", str(record.get("step_summary_evidence_id") or ""))
            yield RobustnessPanelRow.from_dict(data)


def _dict_or_none(value: Any) -> Optional[Dict[str, Any]]:
    return value if isinstance(value, dict) else None


def _safe_key(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.strip().lower())


def _optional_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _first_float(summary: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key in summary:
            value = _optional_float(summary.get(key))
            if value is not None:
                return value
    return None


__all__ = [
    "MIN_AXIS_COUNTS",
    "PANEL_FILENAME",
    "PRIMARY_SPEC_ID",
    "RobustnessPanel",
    "RobustnessPanelRow",
    "RobustnessPlanError",
    "RobustnessSpec",
    "assert_robustness_specs_locked",
    "build_robustness_panel_from_records",
    "default_robustness_specs",
    "ensure_robustness_specs",
    "load_locked_robustness_specs",
    "load_robustness_panel",
    "numeric_digest_for_panel",
    "robustness_specs_for_execution",
    "validate_robustness_specs",
    "worst_rows_by_axis",
    "write_locked_robustness_specs",
    "write_robustness_panel",
]
