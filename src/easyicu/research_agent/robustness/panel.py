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
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..authority.lock_contract import (
    LockAuthorityError,
    assert_lock_matches_evidence_anchor,
    rehydrate_timestamp_only_legacy_lock,
)
from ..planning.cohort_contract import CohortDefinition
from ..planning.robustness_contract import (
    MIN_AXIS_COUNTS,
    RobustnessAxis,
    RobustnessPlanError,
    RobustnessSpec,
    validate_robustness_specs,
)

PRIMARY_SPEC_ID = "primary"
LOCK_FILENAME = "robustness_specs_locked.json"
PANEL_FILENAME = "robustness_panel.json"


def _assert_lock_matches_evidence_anchor(*, run_dir: Path, lock_path: Path) -> bool:
    """Verify a modern lock against its immutable plan-time evidence copy.

    Older fixture/legacy runs can predate ``evidence_index.json``; their
    self-hash remains the compatibility boundary.  Once an evidence index
    exists, however, the lock must match the exact bytes registered by the
    planner.  This prevents a post-lock rewrite from self-authorising merely by
    recomputing ``spec_sha256`` inside the same mutable JSON file.
    """

    try:
        return assert_lock_matches_evidence_anchor(
            run_dir=run_dir,
            lock_path=lock_path,
            evidence_id="robustness_specs_locked",
            label="robustness specification lock",
        )
    except LockAuthorityError as exc:
        raise RobustnessPlanError(str(exc)) from exc


def _successful_step_records(
    per_step_records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return latest-per-step records with an explicit successful status."""

    # Imported lazily because runtime_artifacts owns schema models; this module
    # is a runtime consumer of the pure planning contract, not its owner.
    from ..authority.runtime_artifacts import current_successful_step_records

    return [
        record
        for record in current_successful_step_records(per_step_records)
        if isinstance(record, dict)
    ]


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
        claimable_rows = [row for row in row_tuple if _row_has_claimable_estimate(row)]
        converged_lows = [r.ci_low for r in claimable_rows if r.ci_low is not None]
        converged_highs = [r.ci_high for r in claimable_rows if r.ci_high is not None]
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
    path = run_dir / LOCK_FILENAME
    if path.exists():
        try:
            repair = rehydrate_timestamp_only_legacy_lock(
                run_dir=run_dir,
                lock_path=path,
                evidence_id="robustness_specs_locked",
                label="robustness specification lock",
            )
        except LockAuthorityError as exc:
            raise RobustnessPlanError(str(exc)) from exc
        if repair is not None and evidence.get(
            "robustness_lock_resume_rehydration"
        ) is None:
            evidence.register_json(
                kind="log",
                description=(
                    "Resume compatibility repair: restored the robustness lock "
                    "from its verified plan-time evidence anchor after a legacy "
                    "timestamp-only rewrite."
                ),
                payload=repair,
                filename="robustness_lock_resume_rehydration.json",
                evidence_id="robustness_lock_resume_rehydration",
                producer="planner",
                generation_mode="system",
                prompt_pack_version=prompt_pack_version,
                metadata={"llm_signature": llm_signature},
            )
        locked_specs = load_locked_robustness_specs(run_dir)
        validate_robustness_specs(locked_specs)
        if specs and robustness_specs_sha(specs) != robustness_specs_sha(
            locked_specs
        ):
            raise RobustnessPlanError(
                "robustness_specs changed after plan lock; refusing to overwrite "
                "the pre-specified execution contract"
            )
        if evidence.get("robustness_specs_locked") is None:
            evidence.register_file(
                kind="log",
                description=(
                    "Pre-specified robustness specifications locked after planning."
                ),
                source_path=path,
                evidence_id="robustness_specs_locked",
                aliases=["robustness_specs_locked"],
                producer="planner",
                generation_mode="system",
                prompt_pack_version=prompt_pack_version,
                metadata={"llm_signature": llm_signature, "lock_reused": True},
            )
        return path
    if not specs:
        specs = default_robustness_specs()
    validate_robustness_specs(specs)
    payload = {
        "schema_version": "easyicu.robustness_specs/1",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": robustness_specs_sha(specs),
        "specs": [spec.to_dict() for spec in specs],
    }
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
    path = run_dir / LOCK_FILENAME
    if not path.exists() and not specs:
        return
    if not path.exists():
        raise RobustnessPlanError("robustness_specs plan locked file is missing")
    locked_specs = load_locked_robustness_specs(run_dir)
    if specs and robustness_specs_sha(specs) != robustness_specs_sha(locked_specs):
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
    if path.is_symlink() or not path.is_file():
        raise RobustnessPlanError("robustness_specs lock must be a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RobustnessPlanError(
            f"robustness_specs lock is unreadable: {exc}"
        ) from exc
    raw_specs = payload.get("specs") or []
    if not isinstance(raw_specs, list) or not raw_specs:
        raise RobustnessPlanError("robustness_specs lock has invalid specs payload")
    if any(not isinstance(spec, dict) for spec in raw_specs):
        raise RobustnessPlanError("robustness_specs lock has invalid spec entries")
    specs = [
        RobustnessSpec.from_dict(spec)
        for spec in raw_specs
    ]
    validate_robustness_specs(specs)
    expected_sha = str(payload.get("spec_sha256") or "").strip()
    observed_sha = robustness_specs_sha(specs)
    if not expected_sha or expected_sha != observed_sha:
        raise RobustnessPlanError("robustness specification lock hash mismatch")
    _assert_lock_matches_evidence_anchor(run_dir=Path(run_dir), lock_path=path)
    return specs


def robustness_specs_for_execution(*, run_dir: Path, plan: Any) -> List[RobustnessSpec]:
    """Return the validated plan-time lock as the sole execution contract."""

    active_specs = list(getattr(plan, "robustness_specs", []) or [])
    locked_specs = load_locked_robustness_specs(run_dir)
    if active_specs:
        if not locked_specs:
            raise RobustnessPlanError("robustness_specs plan locked file is missing")
        if robustness_specs_sha(active_specs) != robustness_specs_sha(locked_specs):
            raise RobustnessPlanError(
                "robustness_specs changed after plan lock; execute phase refuses "
                "to run an unlocked robustness panel"
            )
    return locked_specs


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
    successful_records = _successful_step_records(per_step_records)
    locked_axes = {spec.spec_id: spec.axis for spec in specs}
    rows: List[RobustnessPanelRow] = []
    existing: set[str] = set()
    primary = _primary_row_from_records(successful_records)
    if primary is not None:
        rows.append(primary)
        existing.add(primary.spec_id)
    for row in _declared_rows_from_records(successful_records):
        # The canonical primary row requires a complete typed effect contract;
        # free-form panel rows cannot replace it. Variant rows, however, are the
        # agent/step-owned scientific products and outrank auxiliary refits.
        if row.spec_id == PRIMARY_SPEC_ID:
            continue
        if locked_axes.get(row.spec_id) != row.axis:
            continue
        if row.spec_id in existing:
            continue
        rows.append(row)
        existing.add(row.spec_id)
    for row in adapter_rows or []:
        # Auxiliary fitting may fill only pre-specified non-primary variants.
        # The primary row must come from a complete successful step contract.
        if row.spec_id == PRIMARY_SPEC_ID:
            continue
        if locked_axes.get(row.spec_id) != row.axis:
            continue
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
    lock_path = Path(run_dir) / LOCK_FILENAME
    non_primary_rows = [
        row for row in panel.rows if row.spec_id != PRIMARY_SPEC_ID
    ]
    if non_primary_rows and not lock_path.exists():
        raise RobustnessPlanError(
            "non-primary robustness panel rows require a verified plan-time lock"
        )
    if lock_path.exists():
        locked_axes = {
            spec.spec_id: spec.axis for spec in load_locked_robustness_specs(run_dir)
        }
        if non_primary_rows and not _assert_lock_matches_evidence_anchor(
            run_dir=Path(run_dir),
            lock_path=lock_path,
        ):
            raise RobustnessPlanError(
                "non-primary robustness panel rows require a verified plan-time lock"
            )
        invalid_rows = [
            row.spec_id
            for row in panel.rows
            if row.spec_id != PRIMARY_SPEC_ID
            and locked_axes.get(row.spec_id) != row.axis
        ]
        if invalid_rows:
            raise RobustnessPlanError(
                "robustness panel contains rows outside the plan-time lock: "
                + ", ".join(sorted(set(invalid_rows)))
            )
    _assert_claimable_panel_rows_match_evidence(
        run_dir=Path(run_dir),
        panel=panel,
        evidence=evidence,
    )
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
    primary_is_claimable = bool(
        primary is not None
        and primary.converged
        and primary.n > 0
        and primary.evidence_id
        and primary.point_estimate is not None
        and primary.ci_low is not None
        and primary.ci_high is not None
        and all(
            math.isfinite(float(value))
            for value in (
                primary.point_estimate,
                primary.ci_low,
                primary.ci_high,
            )
        )
    )
    if primary_is_claimable and primary is not None:
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
        if row.spec_id == panel.primary_spec_id or not _row_has_claimable_estimate(row):
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
    from .primary_effect import (
        _extract_primary_effect_payload_from_records,
        _primary_effect_payload_is_complete,
    )

    payload = _extract_primary_effect_payload_from_records(
        _successful_step_records(per_step_records)
    )
    if not _primary_effect_payload_is_complete(payload):
        return None
    assert isinstance(payload, dict)
    sample_size = payload.get("sample_size")
    return RobustnessPanelRow(
        spec_id=PRIMARY_SPEC_ID,
        axis="primary",
        n=int(sample_size) if isinstance(sample_size, int) else 0,
        point_estimate=_optional_float(payload.get("primary_or")),
        ci_low=_optional_float(payload.get("primary_ci_low")),
        ci_high=_optional_float(payload.get("primary_ci_high")),
        se=None,
        evidence_id=str(payload.get("evidence_id") or ""),
        converged=True,
        notes="Primary analysis estimate from step_summary.",
    )


def _declared_rows_from_records(
    per_step_records: Sequence[Dict[str, Any]]
) -> Iterable[RobustnessPanelRow]:
    for record in _successful_step_records(per_step_records):
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
            # Numeric authority is the current digest-bound step summary that
            # contains this row.  A free-form row must not redirect its values to
            # an unrelated evidence id such as a replay log.
            data["evidence_id"] = str(
                record.get("step_summary_evidence_id") or ""
            )
            yield RobustnessPanelRow.from_dict(data)


def _row_has_claimable_estimate(row: RobustnessPanelRow) -> bool:
    return bool(
        row.converged
        and row.n > 0
        and row.evidence_id
        and row.point_estimate is not None
        and row.ci_low is not None
        and row.ci_high is not None
        and all(
            math.isfinite(float(value))
            for value in (row.point_estimate, row.ci_low, row.ci_high)
        )
        and float(row.ci_low) <= float(row.ci_high)
    )


def _same_number(left: Any, right: Any) -> bool:
    left_value = _optional_float(left)
    right_value = _optional_float(right)
    return bool(
        left_value is not None
        and right_value is not None
        and math.isclose(left_value, right_value, rel_tol=1e-9, abs_tol=1e-12)
    )


def _row_matches_summary_payload(
    row: RobustnessPanelRow,
    payload: Dict[str, Any],
) -> bool:
    if row.spec_id == PRIMARY_SPEC_ID:
        from .primary_effect import (
            _extract_primary_effect_payload_from_summary,
            _primary_effect_payload_is_complete,
        )

        candidate = _extract_primary_effect_payload_from_summary(
            payload,
            path=None,
            preferred_predictor=None,
        )
        candidate.pop("_score", None)
        if not _primary_effect_payload_is_complete(candidate):
            return False
        return bool(
            int(float(candidate.get("sample_size") or 0)) == row.n
            and _same_number(candidate.get("primary_or"), row.point_estimate)
            and _same_number(candidate.get("primary_ci_low"), row.ci_low)
            and _same_number(candidate.get("primary_ci_high"), row.ci_high)
        )

    candidates = payload.get("robustness_rows")
    if candidates is None and isinstance(payload.get("robustness_panel"), dict):
        candidates = payload["robustness_panel"].get("rows")
    if not isinstance(candidates, list):
        return False
    matching = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and str(candidate.get("spec_id") or "") == row.spec_id
        and str(candidate.get("axis") or "") == row.axis
    ]
    if len(matching) != 1:
        return False
    candidate = matching[0]
    return bool(
        bool(candidate.get("converged"))
        and int(candidate.get("n") or 0) == row.n
        and _same_number(candidate.get("point_estimate"), row.point_estimate)
        and _same_number(candidate.get("ci_low"), row.ci_low)
        and _same_number(candidate.get("ci_high"), row.ci_high)
    )


def _assert_claimable_panel_rows_match_evidence(
    *,
    run_dir: Path,
    panel: RobustnessPanel,
    evidence: Any,
) -> None:
    records = {
        str(record.evidence_id): record
        for record in evidence.records()
        if str(getattr(record, "evidence_id", "")).strip()
    }
    aliases = evidence.aliases()
    from ..authority.runtime_artifacts import verified_run_evidence_path

    for row in panel.rows:
        if row.converged and not _row_has_claimable_estimate(row):
            raise RobustnessPlanError(
                f"robustness panel row {row.spec_id!r} has an incomplete claim contract"
            )
        if not _row_has_claimable_estimate(row):
            continue
        canonical_id = (
            row.evidence_id
            if row.evidence_id in records
            else str(aliases.get(row.evidence_id) or "")
        )
        record = records.get(canonical_id)
        if record is None:
            raise RobustnessPlanError(
                f"robustness panel row {row.spec_id!r} references nonexistent evidence"
            )
        evidence_path = verified_run_evidence_path(run_dir, record)
        if evidence_path is None:
            raise RobustnessPlanError(
                f"robustness panel row {row.spec_id!r} references stale evidence"
            )
        try:
            payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RobustnessPlanError(
                f"robustness panel row {row.spec_id!r} evidence is not a JSON summary"
            ) from exc
        if not isinstance(payload, dict) or not _row_matches_summary_payload(row, payload):
            raise RobustnessPlanError(
                f"robustness panel row {row.spec_id!r} disagrees with its evidence summary"
            )


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
