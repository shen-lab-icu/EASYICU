"""Resume/checkpoint policy helpers for the research-agent execute phase."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .contracts import ValidationFinding
from .schema import AnalysisPlan, AnalysisStep


@dataclass
class ResumeApplication:
    """Filtered state loaded from a previous partial manifest."""

    per_step_records: List[Dict[str, Any]]
    resumed_step_ids: Set[str]
    findings: List[ValidationFinding]
    probe_summary: Dict[str, Any]


class ResumeController:
    """Small policy object for explicit/implicit resume decisions.

    The execute loop should decide *when* to run a step. This object decides
    which prior records are still valid, which stale findings must be dropped,
    and whether an explicitly resumed step may reuse previous generated code.
    """

    def __init__(
        self,
        *,
        plan: AnalysisPlan,
        run_dir: Path,
        resume_state: Optional[Dict[str, Any]],
        resume_from_step_id: Optional[str] = None,
        stop_after_step_id: Optional[str] = None,
    ) -> None:
        self.plan = plan
        self.run_dir = Path(run_dir)
        self.resume_state = resume_state
        self.resume_from_step_id = (resume_from_step_id or "").strip() or None
        self.stop_after_step_id = (stop_after_step_id or "").strip() or None
        self._initial_step_order = {s.step_id: i for i, s in enumerate(plan.steps)}
        self._validate_initial_request()

    def _validate_initial_request(self) -> None:
        for label, step_id in (
            ("resume_from_step_id", self.resume_from_step_id),
            ("stop_after_step_id", self.stop_after_step_id),
        ):
            if step_id and self._initial_step_index(step_id) is None:
                raise ValueError(
                    f"{label}={step_id!r} is not in the active analysis plan."
                )

    def _initial_step_index(self, step_id: str) -> Optional[int]:
        if step_id == "00_probe":
            return -1
        return self._initial_step_order.get(step_id)

    def _resume_cut_index(self) -> Optional[int]:
        if not self.resume_from_step_id:
            return None
        return self._initial_step_index(self.resume_from_step_id)

    def _rerun_step_ids(self) -> Set[str]:
        cut_index = self._resume_cut_index()
        if cut_index is None:
            return set()
        step_ids: Set[str] = set()
        if cut_index <= -1:
            step_ids.add("00_probe")
        step_ids.update(
            step.step_id
            for step in self.plan.steps
            if (
                (idx := self._initial_step_index(step.step_id)) is not None
                and idx >= cut_index
            )
        )
        return step_ids

    def apply(self) -> ResumeApplication:
        if self.resume_state is None:
            return ResumeApplication([], set(), [], {})
        try:
            return self._apply()
        except Exception as exc:
            return ResumeApplication(
                [],
                set(),
                [
                    ValidationFinding(
                        validator="resume",
                        severity="error",
                        message=(
                            "Resume state could not be applied safely; prior "
                            "checkpoint records were ignored instead of being "
                            "silently trusted."
                        ),
                        detail={
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:300],
                        },
                    )
                ],
                {},
            )

    def _apply(self) -> ResumeApplication:
        prior_records = [
            rec
            for rec in (self.resume_state or {}).get("per_step_records", []) or []
            if isinstance(rec, dict) and rec.get("step_id")
        ]
        prior_ok_step_ids = {
            str(rec["step_id"]) for rec in prior_records if rec.get("status") == "ok"
        }
        rerun_step_ids = self._rerun_step_ids()
        cut_index = self._resume_cut_index()
        dropped_step_ids: Set[str] = set()
        per_step_records: List[Dict[str, Any]] = []
        resumed_step_ids: Set[str] = set()
        probe_summary: Dict[str, Any] = {}

        for rec in prior_records:
            if rec.get("status") != "ok":
                continue
            step_id = str(rec["step_id"])
            if cut_index is not None:
                idx = self._initial_step_index(step_id)
                if idx is None or idx >= cut_index:
                    dropped_step_ids.add(step_id)
                    continue
            per_step_records.append(rec)
            resumed_step_ids.add(step_id)
            if step_id == "00_probe" and isinstance(rec.get("step_summary"), dict):
                probe_summary = rec["step_summary"]

        findings: List[ValidationFinding] = []
        for payload in (self.resume_state or {}).get("findings", []) or []:
            try:
                finding = ValidationFinding.model_validate(payload)
            except Exception:
                continue
            if self._finding_mentions_step(finding, rerun_step_ids):
                continue
            if finding.validator == "cohort_auditor":
                continue
            if finding.validator == "runner":
                msg = finding.message or ""
                if any(step_id in msg for step_id in prior_ok_step_ids):
                    continue
            findings.append(finding)

        if self.resume_from_step_id:
            findings.append(
                ValidationFinding(
                    validator="resume",
                    severity="info",
                    message=(
                        "Resume forced from requested step "
                        f"{self.resume_from_step_id}; completed records at that "
                        "step and later were ignored."
                    ),
                    detail={
                        "resume_from_step_id": self.resume_from_step_id,
                        "dropped_completed_step_ids": sorted(dropped_step_ids),
                        "cleared_finding_step_ids": sorted(rerun_step_ids),
                    },
                )
            )

        return ResumeApplication(
            per_step_records=per_step_records,
            resumed_step_ids=resumed_step_ids,
            findings=findings,
            probe_summary=probe_summary,
        )

    @staticmethod
    def _finding_mentions_step(
        finding: ValidationFinding,
        step_ids: Set[str],
    ) -> bool:
        if not step_ids:
            return False
        detail = finding.detail if isinstance(finding.detail, dict) else {}
        if detail.get("step_id") in step_ids:
            return True
        haystack = f"{finding.validator} {finding.message} {detail!r}"
        return any(step_id in haystack for step_id in step_ids)

    def prior_code_for_step(
        self,
        step_id: str,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Return prior agent-generated code for an explicitly resumed step."""

        if (
            not self.resume_from_step_id
            or step_id != self.resume_from_step_id
            or self.resume_state is None
        ):
            return None
        for payload in reversed(self.resume_state.get("evidence", []) or []):
            if not isinstance(payload, dict):
                continue
            if (
                payload.get("kind") != "code"
                or payload.get("produced_by_step") != step_id
            ):
                continue
            relative_path = str(payload.get("relative_path") or "")
            if not relative_path:
                continue
            source_path = _resolve_resume_evidence_path(self.run_dir, relative_path)
            if source_path is None:
                continue
            if not source_path.exists():
                continue
            try:
                prior_code = source_path.read_text(encoding="utf-8")
            except OSError:
                continue
            if not _looks_like_generated_python(prior_code):
                continue
            return prior_code, dict(payload)
        return None

    def remaining_steps(
        self,
        *,
        plan: AnalysisPlan,
        executed_step_ids: Set[str],
    ) -> List[AnalysisStep]:
        return [
            step
            for step in plan.steps
            if step.step_id not in executed_step_ids
            and self.within_requested_stop(plan=plan, step=step)
        ]

    def within_requested_stop(
        self,
        *,
        plan: AnalysisPlan,
        step: AnalysisStep,
    ) -> bool:
        stop_index = self.stop_index_for_plan(plan)
        if stop_index is None:
            return True
        step_order = {s.step_id: i for i, s in enumerate(plan.steps)}
        idx = step_order.get(step.step_id)
        return idx is not None and idx <= stop_index

    def stop_index_for_plan(self, plan: AnalysisPlan) -> Optional[int]:
        if self.stop_after_step_id is None:
            return None
        if self.stop_after_step_id == "00_probe":
            return -1
        step_order = {s.step_id: i for i, s in enumerate(plan.steps)}
        idx = step_order.get(self.stop_after_step_id)
        if idx is None:
            raise ValueError(
                f"stop_after_step_id={self.stop_after_step_id!r} is not in "
                "the current analysis plan."
            )
        return idx


def _resolve_resume_evidence_path(run_dir: Path, relative_path: str) -> Optional[Path]:
    candidate_rel = Path(relative_path)
    if candidate_rel.is_absolute() or ".." in candidate_rel.parts:
        return None
    try:
        root = run_dir.resolve()
        candidate = (root / candidate_rel).resolve()
        candidate.relative_to(root)
    except Exception:
        return None
    return candidate


def _looks_like_generated_python(code: str) -> bool:
    stripped = code.strip()
    if not stripped or stripped in {"{}", "[]", "null", "None"}:
        return False
    return any(
        marker in stripped
        for marker in (
            "import ",
            "from ",
            "def ",
            "os.environ",
            "pd.",
            "STEP_OUT_DIR",
            "COHORT_PARQUET",
        )
    )


def upsert_step_record(
    records: List[Dict[str, Any]],
    record: Dict[str, Any],
    *,
    replace_statuses: Optional[Set[str]] = None,
) -> None:
    step_id = record.get("step_id")
    if not step_id:
        records.append(record)
        return
    for idx, existing in enumerate(records):
        if existing.get("step_id") != step_id:
            continue
        if replace_statuses is not None and existing.get("status") not in replace_statuses:
            continue
        records[idx] = record
        return
    records.append(record)
