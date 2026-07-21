"""Resume/checkpoint policy helpers for the research-agent execute phase."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..repairs.patch import looks_like_executable_python
from ..contracts.runtime import ValidationFinding
from ..authority.evidence_snapshot import load_current_evidence_snapshot
from ..authority.runtime_artifacts import (
    current_step_records,
    verified_run_evidence_path,
)
from ..schema import AnalysisPlan, AnalysisStep


@dataclass
class ResumeApplication:
    """Filtered state loaded from a previous partial manifest."""

    per_step_records: List[Dict[str, Any]]
    resumed_step_ids: Set[str]
    findings: List[ValidationFinding]
    probe_summary: Dict[str, Any]
    audit_history: List[Dict[str, Any]] = field(default_factory=list)
    dropped_step_ids: Set[str] = field(default_factory=set)


@dataclass(frozen=True)
class QuarantinedConceptDraft:
    """Unexecuted code retained only so a failed concept repair can resume."""

    code: str
    sha256: str
    relative_path: str
    findings: Tuple[Dict[str, Any], ...]


_QUARANTINE_SCHEMA = "easyicu.quarantined_concept_draft/1"
_QUARANTINE_DIRNAME = ".quarantine"
_QUARANTINE_CODE_NAME = "concept_draft.py"
_QUARANTINE_META_NAME = "concept_draft.json"
_ROOT_AGENT_CODE_GENERATION_MODES = frozenset({"llm", "repaired", "runner_repaired"})
_AGENT_CODE_GENERATION_MODES = frozenset(
    {*_ROOT_AGENT_CODE_GENERATION_MODES, "resumed_code_reuse"}
)
_SHA256_HEX_LENGTH = 64


def _agent_origin_generation_mode(payload: Dict[str, Any]) -> Optional[str]:
    """Resolve a code record to a non-resumed agent origin."""

    generation_mode = str(payload.get("generation_mode") or "")
    if generation_mode in _ROOT_AGENT_CODE_GENERATION_MODES:
        return generation_mode
    if generation_mode != "resumed_code_reuse":
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    resumed_from = str(metadata.get("resumed_from_generation_mode") or "")
    if resumed_from in _ROOT_AGENT_CODE_GENERATION_MODES:
        return resumed_from
    return None


def _safe_step_component(step_id: str) -> str:
    text = str(step_id or "")
    if (
        not text
        or text in {".", ".."}
        or "\x00" in text
        or "/" in text
        or "\\" in text
        or Path(text).is_absolute()
        or Path(text).name != text
    ):
        raise ValueError("step_id must be a single safe path component")
    return text


def _quarantine_paths(run_dir: Path, step_id: str) -> Tuple[Path, Path, Path]:
    safe_step_id = _safe_step_component(step_id)
    root = Path(run_dir).expanduser().resolve()
    steps_dir = root / "steps"
    step_dir = steps_dir / safe_step_id
    quarantine_dir = step_dir / _QUARANTINE_DIRNAME
    code_path = quarantine_dir / _QUARANTINE_CODE_NAME
    meta_path = quarantine_dir / _QUARANTINE_META_NAME
    for label, path in (
        ("steps directory", steps_dir),
        ("step directory", step_dir),
        ("quarantine directory", quarantine_dir),
        ("quarantine code", code_path),
        ("quarantine metadata", meta_path),
    ):
        if path.is_symlink():
            raise ValueError(f"{label} must not be a symbolic link")
    return (
        quarantine_dir,
        code_path,
        meta_path,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temp_name = handle.name
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def store_quarantined_concept_draft(
    *,
    run_dir: Path,
    step_id: str,
    code: str,
    findings: List[Dict[str, Any]],
) -> QuarantinedConceptDraft:
    """Atomically retain rejected code outside the ordinary evidence store."""

    quarantine_dir, code_path, meta_path = _quarantine_paths(run_dir, step_id)
    if not _looks_like_generated_python(code):
        raise ValueError("quarantined concept draft is not recognisable Python")
    error_findings: Tuple[Dict[str, Any], ...] = tuple(
        ValidationFinding.model_validate(finding).model_dump(mode="json")
        for finding in findings
        if isinstance(finding, dict) and finding.get("severity") == "error"
    )
    if not error_findings:
        raise ValueError("quarantined concept draft requires an error finding")
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    relative_path = str(code_path.relative_to(Path(run_dir).expanduser().resolve()))
    payload = {
        "schema_version": _QUARANTINE_SCHEMA,
        "step_id": step_id,
        "state": "unexecuted",
        "requires_repair": True,
        "sha256": digest,
        "relative_path": relative_path,
        "findings": list(error_findings),
    }
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    # Re-check after creating missing parents so a pre-existing or concurrently
    # swapped symlink is never used as the write destination.
    quarantine_dir, code_path, meta_path = _quarantine_paths(run_dir, step_id)
    # Publish metadata last. A crash after the code write leaves no valid
    # checkpoint; a reader never trusts code without a matching digest record.
    _atomic_write_text(code_path, code)
    _atomic_write_text(
        meta_path,
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
    )
    return QuarantinedConceptDraft(
        code=code,
        sha256=digest,
        relative_path=relative_path,
        findings=error_findings,
    )


def load_quarantined_concept_draft(
    *, run_dir: Path, step_id: str
) -> Optional[QuarantinedConceptDraft]:
    """Load a valid rejected draft; malformed or tampered checkpoints fail closed."""

    try:
        _quarantine_dir, code_path, meta_path = _quarantine_paths(run_dir, step_id)
    except ValueError:
        return None
    if not code_path.is_file() or not meta_path.is_file():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        code = code_path.read_text(encoding="utf-8")
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    root = Path(run_dir).expanduser().resolve()
    expected_relative_path = str(code_path.relative_to(root))
    if (
        payload.get("schema_version") != _QUARANTINE_SCHEMA
        or payload.get("step_id") != step_id
        or payload.get("state") != "unexecuted"
        or payload.get("requires_repair") is not True
        or payload.get("relative_path") != expected_relative_path
        or not _looks_like_generated_python(code)
    ):
        return None
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if payload.get("sha256") != digest:
        return None
    raw_findings = payload.get("findings")
    if not isinstance(raw_findings, list):
        return None
    try:
        error_findings = tuple(
            ValidationFinding.model_validate(finding).model_dump(mode="json")
            for finding in raw_findings
            if isinstance(finding, dict) and finding.get("severity") == "error"
        )
    except (TypeError, ValueError):
        return None
    if not error_findings or len(error_findings) != len(raw_findings):
        return None
    return QuarantinedConceptDraft(
        code=code,
        sha256=digest,
        relative_path=expected_relative_path,
        findings=error_findings,
    )


def clear_quarantined_concept_draft(*, run_dir: Path, step_id: str) -> None:
    """Remove a draft once its material repair passes the concept gate."""

    quarantine_dir, _code_path, _meta_path = _quarantine_paths(run_dir, step_id)
    if not quarantine_dir.exists():
        return
    try:
        shutil.rmtree(quarantine_dir)
    except OSError as exc:
        raise ValueError("quarantine directory could not be removed safely") from exc
    if quarantine_dir.exists() or quarantine_dir.is_symlink():
        raise ValueError("quarantine directory remained after removal")


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
        prior_history = [
            rec
            for rec in (self.resume_state or {}).get("per_step_records", []) or []
            if isinstance(rec, dict) and rec.get("step_id")
        ]
        saved_attempt_history = [
            dict(rec)
            for rec in (self.resume_state or {}).get("step_attempt_history", []) or []
            if isinstance(rec, dict) and rec.get("step_id")
        ]
        audit_history = list(saved_attempt_history)
        for record in prior_history:
            if record not in audit_history:
                audit_history.append(dict(record))
        # The checkpoint ledger is append-only, but resume authority is not:
        # only the newest outer record for a step may be reused.  In
        # particular, an old ``ok`` must not survive a later contract failure.
        prior_records = [dict(rec) for rec in current_step_records(prior_history)]
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
            detail = finding.detail if isinstance(finding.detail, dict) else {}
            # Plan-DAG findings describe a particular saved plan revision, not
            # an immutable run event. Resume loads the newest digest-verified
            # compatible plan and the execute phase re-evaluates its trajectory
            # DAG immediately. Carrying pending/errors from an older revision
            # can otherwise block a now-valid plan before the selected step.
            if finding.validator in {"plan_contract_pending", "plan_typed_dag"}:
                continue
            if finding.validator == "plan_contract" and str(
                detail.get("kind") or ""
            ).startswith("trajectory_"):
                continue
            if self._finding_mentions_step(finding, rerun_step_ids):
                continue
            if finding.validator == "cohort_auditor":
                continue
            if finding.validator == "runner":
                if self._finding_mentions_step(finding, prior_ok_step_ids):
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
            audit_history=audit_history,
            dropped_step_ids=set(dropped_step_ids),
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
        return any(
            re.search(
                rf"(?<![A-Za-z0-9_.-]){re.escape(step_id)}(?![A-Za-z0-9_.-])",
                haystack,
            )
            is not None
            for step_id in step_ids
        )

    def prior_code_for_step(
        self,
        step_id: str,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Return eligible evidence-bound code for a resumed step.

        Modern runs may reuse only code selected by the full-state evidence
        authority. Legacy runs without that selector retain the historical
        manifest/index fallback so old checkpoints can still resume. Mutable
        manifest copies can never re-authorize code omitted from a modern
        authority generation.

        Explicit resume preserves its historical selected-step behaviour.  On
        implicit resume, code is merely offered when the newest outer record
        for this step is ``contract_failed``.  The execute phase remains the
        authority for the stronger one-shot, digest, input, and scientific-
        signature checks before any candidate can bypass initial generation.
        """

        explicitly_selected = (
            self.resume_from_step_id is not None and step_id == self.resume_from_step_id
        )
        latest_records = current_step_records(
            [
                record
                for record in (
                    (self.resume_state or {}).get("per_step_records", []) or []
                )
                if isinstance(record, dict) and record.get("step_id")
            ]
        )
        implicitly_failed_contract = self.resume_from_step_id is None and any(
            str(record.get("step_id") or "") == step_id
            and str(record.get("status") or "").strip().lower() == "contract_failed"
            for record in latest_records
        )
        if not explicitly_selected and not implicitly_failed_contract:
            return None

        snapshot = load_current_evidence_snapshot(self.run_dir)
        if snapshot.generation is not None or snapshot.source in {
            "root_marker_legacy",
            "root_marker_legacy_prepared",
        }:
            # Once a store has a selected authority, mutable manifest copies
            # cannot re-authorize code that is absent from that authority.
            payloads: List[Any] = list(snapshot.records)
        else:
            payloads = list((self.resume_state or {}).get("evidence", []) or [])
            for path in (self.run_dir / "manifest_partial.json",):
                if not path.is_file():
                    continue
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(loaded, dict):
                    records = loaded.get("evidence", []) or []
                elif isinstance(loaded, list):
                    records = loaded
                else:
                    records = []
                if isinstance(records, list):
                    payloads.extend(records)
            payloads.extend(snapshot.records)

        for payload in reversed(payloads):
            if not isinstance(payload, dict):
                continue
            if (
                payload.get("kind") != "code"
                or payload.get("produced_by_step") != step_id
            ):
                continue
            generation_mode = str(payload.get("generation_mode") or "")
            if (
                generation_mode not in _AGENT_CODE_GENERATION_MODES
                or _agent_origin_generation_mode(payload) is None
            ):
                continue
            expected_sha256 = str(payload.get("sha256") or "").lower()
            if len(expected_sha256) != _SHA256_HEX_LENGTH or any(
                char not in "0123456789abcdef" for char in expected_sha256
            ):
                continue
            relative_path = str(payload.get("relative_path") or "")
            if not relative_path:
                continue
            source_path = verified_run_evidence_path(self.run_dir, payload)
            if source_path is None:
                continue
            try:
                raw_code = source_path.read_bytes()
                prior_code = raw_code.decode("utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if not _looks_like_generated_python(prior_code):
                continue
            return prior_code, dict(payload)
        return None

    def prior_negative_critic_report_for_step(
        self,
        step_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the latest fail-closed Critic report for a selected rerun.

        The resume application intentionally drops the selected failed record
        from the live ledger.  Keep its structured Critic feedback available as
        repair input so the next attempt improves the prior Agent script instead
        of blindly generating an unrelated replacement.
        """

        if not self.resume_from_step_id or step_id != self.resume_from_step_id:
            return None
        records = [
            record
            for record in (self.resume_state or {}).get("per_step_records", []) or []
            if isinstance(record, dict) and record.get("step_id") == step_id
        ]
        for record in reversed(records):
            if str(record.get("status") or "") != "critic_failed":
                continue
            report = record.get("critique_report")
            if not isinstance(report, dict):
                continue
            if str(report.get("status") or "") not in {"needs_revision", "blocked"}:
                continue
            return dict(report)
        return None

    def quarantined_concept_draft_for_step(
        self,
        step_id: str,
    ) -> Optional[QuarantinedConceptDraft]:
        """Return pending rejected code only for the explicitly resumed step.

        This is intentionally separate from :meth:`prior_code_for_step`: the
        caller must force a fresh repair and all audits before runner execution.
        """

        if not self.resume_from_step_id or step_id != self.resume_from_step_id:
            return None
        records = [
            record
            for record in (
                (self.resume_state or {}).get("step_attempt_history")
                or (self.resume_state or {}).get("per_step_records")
                or []
            )
            if isinstance(record, dict) and record.get("step_id") == step_id
        ]
        latest = next(
            (
                record
                for record in current_step_records(records)
                if str(record.get("step_id") or "") == step_id
            ),
            None,
        )
        if (
            not isinstance(latest, dict)
            or latest.get("quarantined_requires_repair") is not True
        ):
            return None
        draft = load_quarantined_concept_draft(
            run_dir=self.run_dir,
            step_id=step_id,
        )
        if draft is None:
            return None
        if (
            str(latest.get("quarantined_draft_sha256") or "") != draft.sha256
            or str(latest.get("quarantined_draft_relative_path") or "")
            != draft.relative_path
        ):
            return None
        return draft

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
        step_order = {s.step_id: i for i, s in enumerate(plan.steps)}
        idx = step_order.get(step.step_id)
        if idx is None:
            return False
        # ``resume_from_step_id`` is a real lower execution bound, not merely a
        # finding-cleanup hint. Earlier incomplete/supporting steps are outside
        # an explicitly targeted repair window and must not consume model calls.
        start_index = self._resume_cut_index()
        if start_index is not None and idx < start_index:
            return False
        stop_index = self.stop_index_for_plan(plan)
        return stop_index is None or idx <= stop_index

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
        evidence_root = root / "evidence"
        if evidence_root.is_symlink():
            return None
        resolved_evidence_root = evidence_root.resolve()
        resolved_evidence_root.relative_to(root)
        candidate = (root / candidate_rel).resolve()
        candidate.relative_to(root)
        candidate.relative_to(resolved_evidence_root)
    except Exception:
        return None
    return candidate


def _looks_like_generated_python(code: str) -> bool:
    return looks_like_executable_python(code)


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
        if (
            replace_statuses is not None
            and existing.get("status") not in replace_statuses
        ):
            continue
        records[idx] = record
        return
    records.append(record)
