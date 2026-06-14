"""Run readiness gates and report rendering.

Extracted from :mod:`easyicu.research_agent.pipeline` so the fail-closed
readiness logic (run_status.json, evidence_audit.json, numeric_audit.json,
claim_ledger.csv, author_review_note.md, manuscript_ready.md) and the
human-readable results_report.md rendering can be reasoned about,
tested, and re-used without pulling the full ``ResearchAgentPipeline``
class into memory.

Public entry points
-------------------
* :func:`write_readiness_artifacts` — compute the fail-closed gates,
  write the canonical audit JSON / CSV artefacts, register them on the
  :class:`EvidenceStore`, and return ``(gates, artifact_paths)``.
* :func:`render_report` — render the human-readable run summary
  ``results_report.md``.
* :func:`execution_gate_status` — pure helper that decides whether
  the analysis-execution phase reached completion (called both during
  the execute loop and during readiness derivation).

Internal helpers
----------------
``_count_missing_evidence_markers``, ``_publication_figure_bundle_ready``,
``_compute_readiness_gates``, ``_extract_claim_ledger_rows`` and
``_render_author_review_note`` keep leading underscores because they
have no external callers — they are implementation details of the two
public entry points above.

All functions are stateless: they take their inputs as keyword args
(``context``, ``plan``, ``findings``, ``per_step_records``,
``evidence``, ``run_dir``, ``manuscript_path`` ...) and write/read
the canonical run-directory layout. Nothing here touches the
``ResearchAgentPipeline`` instance.
"""

from __future__ import annotations

import csv
import json
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .evidence import EvidenceStore
from .schema import AnalysisPlan, ResearchContext, ValidationFinding


def execution_gate_status(
    *,
    plan: Optional[AnalysisPlan],
    per_step_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if plan is None:
        return {
            "execution_complete": False,
            "required_step_count": 0,
            "completed_step_count": 0,
            "missing_steps": [],
            "failed_steps": [{"step_id": "plan", "status": "not_available"}],
        }
    required_step_ids = [s.step_id for s in (plan.steps if plan is not None else [])]
    # 🔧 2026-05-16: previously this dict excluded `00_probe` records on the
    # assumption the planner never lists it. But the planner *does* sometimes
    # include `00_probe` in the published plan, and a deterministic probe
    # record is appended to per_step_records with status='ok' from
    # pipeline_execute._build_probe_summary. Excluding it here made the gate
    # mis-report `00_probe` as a permanently-missing required step. Surfacing
    # the deterministic probe record fixes the false negative.
    status_by_step = {
        str(record.get("step_id")): str(record.get("status") or "")
        for record in per_step_records
        if record.get("step_id")
    }
    missing_steps = [step_id for step_id in required_step_ids if step_id not in status_by_step]
    failed_steps = [
        {"step_id": step_id, "status": status_by_step.get(step_id)}
        for step_id in required_step_ids
        if step_id in status_by_step and status_by_step.get(step_id) != "ok"
    ]
    return {
        "execution_complete": not missing_steps and not failed_steps,
        "required_step_count": len(required_step_ids),
        "completed_step_count": sum(
            1 for step_id in required_step_ids if status_by_step.get(step_id) == "ok"
        ),
        "missing_steps": missing_steps,
        "failed_steps": failed_steps,
    }


def _count_missing_evidence_markers(text: str) -> int:
    return len(
        re.findall(
            r"(?:\[evidence missing:\s*[^\]]+\]|<!--\s*evidence missing:\s*[^>]+-->)",
            text or "",
            flags=re.IGNORECASE,
        )
    )


def _publication_figure_bundle_ready(
    *,
    evidence: EvidenceStore,
    run_dir: Path,
    findings: Optional[Sequence[ValidationFinding]] = None,
) -> Dict[str, Any]:
    stems: Dict[str, set[str]] = {}
    source_ready = False
    contract_ready = evidence.get("publication_figure_contract") is not None
    visual_errors = [
        finding
        for finding in (findings or [])
        if finding.severity == "error"
        and finding.validator in {"visual_qa", "vlm_visual_qa"}
    ]
    for record in evidence.records():
        metadata = record.metadata or {}
        if record.evidence_id.startswith("publication_figure_source_"):
            source_ready = True
        is_contract_record = record.evidence_id == "publication_figure_contract"
        if record.kind != "figure":
            if is_contract_record and (
                metadata.get("source_evidence_id")
                or metadata.get("source_evidence_ids")
                or metadata.get("source_data")
            ):
                source_ready = True
            continue
        role = str(metadata.get("figure_role") or "").lower()
        haystack = " ".join(
            [
                record.evidence_id,
                record.description,
                record.relative_path,
                role,
            ]
        ).lower()
        is_explicit_publication = (
            role == "publication_figure"
            or record.evidence_id.startswith("publication_figure_")
            or "publication_figure" in haystack
        )
        if not is_explicit_publication:
            continue
        if (
            metadata.get("source_evidence_id")
            or metadata.get("source_evidence_ids")
            or metadata.get("source_data")
        ):
            source_ready = True
        path = run_dir / record.relative_path
        stem = path.with_suffix("").name.split("__", 1)[-1]
        stems.setdefault(stem, set()).add(path.suffix.lower().lstrip("."))
    ready_stems = [
        stem
        for stem, suffixes in stems.items()
        if {"svg", "png"} <= suffixes
        or {"svg", "png", "pdf", "tiff"} <= suffixes
        or {"svg", "png", "pdf", "tif"} <= suffixes
    ]
    visual_qa_passed = not visual_errors
    contract_complete = contract_ready and source_ready and visual_qa_passed
    if not contract_complete:
        ready_stems = []
    return {
        "publication_figure_bundle_ready": bool(ready_stems) and contract_complete,
        "publication_figure_stems": sorted(stems),
        "publication_ready_stems": sorted(ready_stems),
        "publication_figure_contract_ready": contract_ready,
        "publication_figure_source_data_ready": source_ready,
        "publication_figure_visual_qa_passed": visual_qa_passed,
    }


_STEP_ID_IN_MESSAGE_PATTERNS = (
    # Matches the in-message tokens written by every pipeline_execute
    # ValidationFinding site that references a specific step. See
    # ``_step_id_referenced_in_finding`` for the full taxonomy.
    re.compile(r"\bfor step\s+([A-Za-z0-9_./-]+)"),
    re.compile(r"\bstep\s+([A-Za-z0-9_./-]+)\s+(?:was|failed|skipped|blocked)"),
)


def _step_id_referenced_in_finding(finding: ValidationFinding) -> Optional[str]:
    """Return the step_id a finding ties to, if any.

    Priority order:

    1. ``finding.detail["step_id"]`` (preferred — set by new
       step-tied finding sites going forward).
    2. Regex scan of ``finding.message`` for the canonical
       ``"for step <id>"`` / ``"step <id> failed"`` phrasings used
       across :mod:`pipeline_execute`. This catches the existing
       ~14 historical ValidationFinding sites without requiring a
       sweeping rewrite at every emit point.

    Returns ``None`` when no step_id reference is found, which means
    the finding is global (e.g. "no manuscript generated") and should
    NOT be superseded by per-step success.
    """
    if finding.detail and isinstance(finding.detail.get("step_id"), str):
        return finding.detail["step_id"]
    message = finding.message or ""
    for pattern in _STEP_ID_IN_MESSAGE_PATTERNS:
        m = pattern.search(message)
        if m:
            return m.group(1)
    return None


def _successful_step_ids(per_step_records: Sequence[Dict[str, Any]]) -> set:
    """Step ids whose FINAL recorded status was ``"ok"``.

    Used to identify findings that have been *superseded* by a
    successful retry / resume / deterministic-fallback / replanned
    re-execution. The general rule: if a step ultimately finished
    cleanly, its earlier failure findings (e.g. a 502 from the
    original pre-resume invocation) should not count against the
    final readiness gates — they remain in the manifest for the
    audit trail but are not treated as errors at the report stage.
    """
    return {
        str(rec.get("step_id"))
        for rec in per_step_records
        if isinstance(rec, dict)
        and rec.get("status") == "ok"
        and rec.get("step_id")
    }


def _step_ids_in_records(per_step_records: Sequence[Dict[str, Any]]) -> set:
    """Every step_id that has any per_step_record entry (ok or not).

    A finding whose referenced step_id is NOT in this set must
    come from a step that was *replanned away* — i.e. the replanner
    dropped the failing step and substituted a different step that
    later ran instead. The original failure is no longer part of
    the plan-of-record and shouldn't drag the readiness gate down
    when the final-plan execution otherwise completed cleanly. This
    is the third recognised supersession axis (see
    :func:`_partition_findings_by_supersession`).
    """
    return {
        str(rec.get("step_id"))
        for rec in per_step_records
        if isinstance(rec, dict) and rec.get("step_id")
    }


_GATE_STATE_SUPERSESSION_PATTERNS = (
    # Common "we skipped X because gate Y did not pass" / "X was not
    # produced because Y failed" findings emitted when a gate was
    # transiently False during the run. If the gate is now True at
    # report time, the finding is stale and should not count.
    (
        "manuscript_gate",
        "execution gate did not pass",
        "execution_complete",
    ),
    (
        "manuscript_gate",
        "manuscript generation skipped",
        "execution_complete",
    ),
)


def _is_gate_state_superseded(
    finding: ValidationFinding,
    *,
    gate_state: Dict[str, bool],
) -> bool:
    """True when the finding documents a transient gate-state failure
    that the latest gate snapshot has now resolved.

    For example, ``manuscript_gate`` emits a finding when the writer
    refuses to draft a manuscript because the execution gate was False
    at that moment. If the writer later runs successfully on resume
    and execution_complete becomes True, the original "skipped because
    execution gate did not pass" message is stale: nothing skipped in
    the *final* state of the run.
    """
    validator = finding.validator or ""
    message = (finding.message or "").lower()
    for v_match, msg_substr, gate_key in _GATE_STATE_SUPERSESSION_PATTERNS:
        if v_match == validator and msg_substr.lower() in message:
            if gate_state.get(gate_key):
                return True
    return False


def _partition_findings_by_supersession(
    findings: Sequence[ValidationFinding],
    *,
    success_step_ids: set,
    known_step_ids: Optional[set] = None,
    gate_state: Optional[Dict[str, bool]] = None,
) -> tuple[List[ValidationFinding], List[ValidationFinding]]:
    """Split findings into (active, superseded).

    A finding is *superseded* when one of:

    1. It carries a step_id reference (via ``detail.step_id`` or a
       recognised message pattern) and that step_id is in
       ``success_step_ids`` — the step ultimately succeeded so its
       earlier failure findings are stale.
    2. It references a step_id that is no longer in the plan-of-
       record (i.e. not in ``known_step_ids``). This is the
       "replanned away" axis: the replanner dropped the failing
       step and the substitute step ran instead. The original
       failure is detached from the final plan and shouldn't count
       against the final gate. ``known_step_ids`` is the union of
       every step_id present in ``per_step_records`` (ok or not).
       When ``known_step_ids`` is None, this axis is skipped.
    3. It documents a transient gate-state failure (e.g.,
       ``manuscript_gate`` complaining the execution gate had not
       passed) and the corresponding gate is now True in
       ``gate_state``.

    The classification is purely deterministic — same inputs always
    yield the same partition. The superseded set is returned
    alongside the active set so the manifest can record both for
    audit traceability.
    """
    gate_state = gate_state or {}
    active: List[ValidationFinding] = []
    superseded: List[ValidationFinding] = []
    for f in findings:
        sid = _step_id_referenced_in_finding(f)
        if sid:
            if sid in success_step_ids:
                superseded.append(f)
                continue
            if known_step_ids is not None and sid not in known_step_ids:
                # Step was replanned away — its failure is no longer
                # part of the plan-of-record.
                superseded.append(f)
                continue
        if _is_gate_state_superseded(f, gate_state=gate_state):
            superseded.append(f)
            continue
        active.append(f)
    return active, superseded


def _compute_readiness_gates(
    *,
    plan: Optional[AnalysisPlan],
    per_step_records: Sequence[Dict[str, Any]],
    findings: Sequence[ValidationFinding],
    evidence: EvidenceStore,
    run_dir: Path,
    manuscript_path: Path,
    stop_after_analysis: bool,
    writer_probe_mode: bool = False,
    writer_probe_failed_steps: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    execution = execution_gate_status(plan=plan, per_step_records=per_step_records)
    manuscript_text = ""
    if manuscript_path.exists():
        try:
            manuscript_text = manuscript_path.read_text(encoding="utf-8")
        except Exception:
            manuscript_text = ""
    missing_evidence_count = _count_missing_evidence_markers(manuscript_text)
    # General supersession rule: if a step eventually succeeded
    # (status="ok" in per_step_records), any earlier ValidationFinding
    # tied to that step_id (e.g. the original 502 / coder_failed
    # finding from before a successful resume) should not count
    # toward the readiness gates. The full finding list is preserved
    # in the manifest so the audit trail still shows the failure +
    # recovery sequence.
    success_step_ids = _successful_step_ids(per_step_records)
    known_step_ids = _step_ids_in_records(per_step_records)
    # Compute current gate state once so the gate-state supersession
    # rule sees the final values, not the transient mid-run snapshot
    # the finding was emitted under.
    current_gate_state: Dict[str, bool] = {
        "execution_complete": bool(execution.get("execution_complete")),
    }
    active_findings, superseded_findings = _partition_findings_by_supersession(
        findings,
        success_step_ids=success_step_ids,
        known_step_ids=known_step_ids,
        gate_state=current_gate_state,
    )
    numeric_errors = [
        f.message
        for f in active_findings
        if f.severity == "error" and f.validator == "manuscript_numeric_auditor"
    ]
    evidence_errors = [
        f.message
        for f in active_findings
        if f.severity == "error" and f.validator in {"evidence_bound_writer", "critic_agent"}
    ]
    non_manuscript_errors = [
        f.message
        for f in active_findings
        if f.severity == "error"
        and f.validator
        not in {
            "manuscript_numeric_auditor",
            "evidence_bound_writer",
            "critic_agent",
        }
    ]
    manuscript_generated = (
        not writer_probe_mode
        and manuscript_path.exists()
        and "Manuscript scaffold not generated" not in manuscript_text[:300]
        and not stop_after_analysis
    )
    evidence_complete = manuscript_generated and missing_evidence_count == 0 and not evidence_errors
    numeric_verified = manuscript_generated and not numeric_errors
    analysis_validated = execution["execution_complete"] and not non_manuscript_errors
    manuscript_ready = (
        execution["execution_complete"]
        and evidence_complete
        and numeric_verified
        and analysis_validated
    )
    publication = _publication_figure_bundle_ready(
        evidence=evidence,
        run_dir=run_dir,
        findings=active_findings,
    )
    return {
        **execution,
        "evidence_complete": evidence_complete,
        "numeric_verified": numeric_verified,
        "analysis_validated": analysis_validated,
        "manuscript_ready": manuscript_ready,
        "publication_ready": manuscript_ready
        and publication["publication_figure_bundle_ready"],
        "manuscript_generated": manuscript_generated,
        "writer_probe_mode": bool(writer_probe_mode),
        "writer_probe_failed_steps": list(writer_probe_failed_steps or []),
        "missing_evidence_count": missing_evidence_count,
        "numeric_error_count": len(numeric_errors),
        "evidence_error_count": len(evidence_errors),
        "analysis_error_count": len(non_manuscript_errors),
        "numeric_errors": numeric_errors,
        "evidence_errors": evidence_errors,
        "analysis_errors": non_manuscript_errors,
        # Audit-trail surface for the supersession rule (see
        # _partition_findings_by_supersession). Reviewers can inspect
        # which findings the readiness gate ignored because the
        # underlying step ultimately succeeded.
        "superseded_error_count": sum(1 for f in superseded_findings if f.severity == "error"),
        "superseded_errors": [
            {"validator": f.validator, "message": f.message}
            for f in superseded_findings
            if f.severity == "error"
        ],
        **publication,
    }


def write_readiness_artifacts(
    *,
    context: ResearchContext,
    plan: Optional[AnalysisPlan],
    findings: Sequence[ValidationFinding],
    per_step_records: Sequence[Dict[str, Any]],
    evidence: EvidenceStore,
    run_dir: Path,
    manuscript_path: Path,
    stop_after_analysis: bool,
    writer_probe_mode: bool = False,
    writer_probe_failed_steps: Optional[Sequence[str]] = None,
) -> tuple[Dict[str, Any], Dict[str, str]]:
    gates = _compute_readiness_gates(
        plan=plan,
        per_step_records=per_step_records,
        findings=findings,
        evidence=evidence,
        run_dir=run_dir,
        manuscript_path=manuscript_path,
        stop_after_analysis=stop_after_analysis,
        writer_probe_mode=writer_probe_mode,
        writer_probe_failed_steps=writer_probe_failed_steps,
    )
    status = (
        "publication_ready"
        if gates["publication_ready"]
        else "manuscript_ready"
        if gates["manuscript_ready"]
        else "analysis_only"
        if gates["execution_complete"]
        else "diagnostic_only"
    )

    artifact_paths: Dict[str, str] = {}

    run_status_path = run_dir / "run_status.json"
    run_status_payload = {
        "schema_version": "easyicu.run_status/1",
        "status": status,
        "strict_fail_closed": True,
        "writer_probe_mode": bool(writer_probe_mode),
        "writer_probe_failed_steps": list(writer_probe_failed_steps or []),
        "research_question": context.research_question,
        "gates": gates,
        "canonical_outputs": {},
    }
    run_status_path.write_text(
        json.dumps(run_status_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    artifact_paths["run_status"] = str(run_status_path.relative_to(run_dir))

    evidence_audit_path = run_dir / "evidence_audit.json"
    evidence_records = evidence.records()
    kinds: Dict[str, int] = {}
    for rec in evidence_records:
        kinds[rec.kind] = kinds.get(rec.kind, 0) + 1
    evidence_audit_payload = {
        "schema_version": "easyicu.evidence_audit/1",
        "evidence_count": len(evidence_records),
        "kinds": kinds,
        "missing_evidence_count": gates["missing_evidence_count"],
        "evidence_complete": gates["evidence_complete"],
        "manuscript_path": str(manuscript_path.relative_to(run_dir))
        if manuscript_path.exists()
        else None,
    }
    evidence_audit_path.write_text(
        json.dumps(evidence_audit_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    artifact_paths["evidence_audit"] = str(evidence_audit_path.relative_to(run_dir))

    numeric_audit_path = run_dir / "numeric_audit.json"
    numeric_audit_payload = {
        "schema_version": "easyicu.numeric_audit/1",
        "numeric_verified": gates["numeric_verified"],
        "numeric_error_count": gates["numeric_error_count"],
        "numeric_errors": gates["numeric_errors"],
    }
    numeric_audit_path.write_text(
        json.dumps(numeric_audit_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    artifact_paths["numeric_audit"] = str(numeric_audit_path.relative_to(run_dir))

    claim_ledger_path = run_dir / "claim_ledger.csv"
    claim_rows = _extract_claim_ledger_rows(manuscript_path=manuscript_path, gates=gates)
    with claim_ledger_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "claim_id",
                "claim_text",
                "evidence_refs",
                "status",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(claim_rows)
    artifact_paths["claim_ledger"] = str(claim_ledger_path.relative_to(run_dir))

    author_review_path = run_dir / "author_review_note.md"
    author_review_path.write_text(
        _render_author_review_note(
            status=status,
            gates=gates,
            findings=findings,
            per_step_records=per_step_records,
        ),
        encoding="utf-8",
    )
    artifact_paths["author_review_note"] = str(author_review_path.relative_to(run_dir))

    manuscript_ready_path = run_dir / "manuscript_ready.md"
    if gates["manuscript_ready"] and manuscript_path.exists():
        manuscript_ready_path.write_text(
            manuscript_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        artifact_paths["manuscript_ready"] = str(
            manuscript_ready_path.relative_to(run_dir)
        )
    elif manuscript_ready_path.exists():
        manuscript_ready_path.unlink()

    run_status_payload["canonical_outputs"] = artifact_paths
    run_status_path.write_text(
        json.dumps(run_status_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    registrations = [
        (
            "run_status",
            "log",
            "Fail-closed run readiness gate summary.",
            run_status_path,
        ),
        (
            "evidence_audit",
            "statistic",
            "Evidence completeness audit for manuscript gating.",
            evidence_audit_path,
        ),
        (
            "numeric_audit",
            "statistic",
            "Numeric-claim audit for manuscript gating.",
            numeric_audit_path,
        ),
        (
            "claim_ledger",
            "table",
            "Ledger of manuscript claims and evidence links.",
            claim_ledger_path,
        ),
        (
            "author_review_note",
            "log",
            "Human-readable fail-closed review note for the run.",
            author_review_path,
        ),
    ]
    if gates["manuscript_ready"] and manuscript_ready_path.exists():
        registrations.append(
            (
                "manuscript_ready",
                "log",
                "Formal manuscript-ready markdown, emitted only after readiness gates pass.",
                manuscript_ready_path,
            )
        )
    for evidence_id, kind, description, path in registrations:
        if evidence.get(evidence_id) is None:
            evidence.register_file(
                kind=kind,
                description=description,
                source_path=path,
                evidence_id=evidence_id,
                aliases=[evidence_id],
                producer="pipeline",
                generation_mode="system",
            )

    return gates, artifact_paths


def _extract_claim_ledger_rows(
    *, manuscript_path: Path, gates: Dict[str, Any]
) -> List[Dict[str, str]]:
    if not manuscript_path.exists():
        return [
            {
                "claim_id": "claim_000",
                "claim_text": "",
                "evidence_refs": "",
                "status": "not_generated",
                "note": "No manuscript file was produced.",
            }
        ]
    text = manuscript_path.read_text(encoding="utf-8", errors="replace")
    if "Manuscript scaffold not generated" in text[:300]:
        return [
            {
                "claim_id": "claim_000",
                "claim_text": "Formal manuscript was not generated.",
                "evidence_refs": "",
                "status": "diagnostic_only",
                "note": "Strict fail-closed gate blocked writer output.",
            }
        ]
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("|"):
            continue
        evidence_refs = re.findall(r"\[([^\]]+)\]\((?:evidence/)?[^)]+\)", stripped)
        missing = _count_missing_evidence_markers(stripped)
        if not evidence_refs and not missing:
            continue
        rows.append(
            {
                "claim_id": f"claim_{len(rows) + 1:03d}",
                "claim_text": re.sub(r"\s+", " ", stripped)[:1000],
                "evidence_refs": ";".join(evidence_refs),
                "status": "missing_evidence" if missing else "bound",
                "note": "" if not missing else "Unresolved evidence marker present.",
            }
        )
    if not rows:
        rows.append(
            {
                "claim_id": "claim_000",
                "claim_text": "",
                "evidence_refs": "",
                "status": "empty",
                "note": (
                    "No evidence-bound claims were detected; manuscript_ready="
                    + str(bool(gates.get("manuscript_ready")))
                ),
            }
        )
    return rows


def _render_author_review_note(
    *,
    status: str,
    gates: Dict[str, Any],
    findings: Sequence[ValidationFinding],
    per_step_records: Sequence[Dict[str, Any]],
) -> str:
    lines = [
        "# Author review note",
        "",
        f"- Status: `{status}`",
        f"- execution_complete: `{gates['execution_complete']}`",
        f"- evidence_complete: `{gates['evidence_complete']}`",
        f"- numeric_verified: `{gates['numeric_verified']}`",
        f"- analysis_validated: `{gates['analysis_validated']}`",
        f"- manuscript_ready: `{gates['manuscript_ready']}`",
        f"- publication_ready: `{gates['publication_ready']}`",
        "",
    ]
    failed_steps = gates.get("failed_steps") or []
    missing_steps = gates.get("missing_steps") or []
    if failed_steps or missing_steps:
        lines.extend(["## Blocking step issues", ""])
        for item in failed_steps:
            lines.append(f"- `{item.get('step_id')}` status `{item.get('status')}`")
        for step_id in missing_steps:
            lines.append(f"- `{step_id}` missing execution record")
        lines.append("")
    error_findings = [f for f in findings if f.severity == "error"]
    if error_findings:
        lines.extend(["## Blocking findings", ""])
        for finding in error_findings:
            lines.append(f"- `{finding.validator}`: {finding.message}")
        lines.append("")
    if not error_findings and not failed_steps and not missing_steps:
        lines.extend(
            [
                "## Review",
                "",
                "No blocking gate failures were detected. Use `manuscript_ready.md` "
                "as the formal draft if present.",
                "",
            ]
        )
    lines.extend(["## Step status", ""])
    for record in per_step_records:
        step_id = record.get("step_id", "")
        if step_id == "00_probe":
            continue
        lines.append(f"- `{step_id}`: `{record.get('status', '?')}`")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def render_report(
    *,
    context: ResearchContext,
    plan: Optional[AnalysisPlan],
    findings: Sequence[ValidationFinding],
    per_step_records: Sequence[Dict[str, Any]],
    evidence: EvidenceStore,
    aborted_reason: Optional[str] = None,
    paused_after_analysis: bool = False,
    readiness: Optional[Dict[str, Any]] = None,
) -> str:
    parts: List[str] = []
    parts.append("# Research-agent results report")
    parts.append("")
    parts.append(f"- Research question: {context.research_question}")
    parts.append(f"- Cohort: {context.cohort.cohort_name} ({context.cohort.database})")
    parts.append(
        f"- Stays: {context.cohort.n_stays:,} / Patients: {context.cohort.n_patients:,}"
    )
    if context.target_outcome:
        parts.append(f"- Target outcome: {context.target_outcome}")
    if context.cross_database_validation:
        parts.append(
            "- Cross-database replication: "
            + ", ".join(context.cross_database_validation)
        )
    parts.append("")

    if aborted_reason:
        parts.append(f"## Status: ABORTED ({aborted_reason})")
        parts.append("")
    elif paused_after_analysis:
        parts.append("## Status: PAUSED AFTER ANALYSIS")
        parts.append("")
        parts.append(
            "The run intentionally stopped before literature retrieval, "
            "manuscript drafting and LaTeX export. Review the registered "
            "tables, figures, statistics and findings before drafting the article."
        )
        parts.append("")
    elif readiness:
        status = (
            "PUBLICATION READY"
            if readiness.get("publication_ready")
            else "MANUSCRIPT READY"
            if readiness.get("manuscript_ready")
            else "DIAGNOSTIC ONLY"
        )
        parts.append(f"## Status: {status}")
        parts.append("")
        parts.append(
            "- Gates: execution_complete={execution_complete}, "
            "evidence_complete={evidence_complete}, "
            "numeric_verified={numeric_verified}, "
            "analysis_validated={analysis_validated}, "
            "manuscript_ready={manuscript_ready}, "
            "publication_ready={publication_ready}".format(**readiness)
        )
        parts.append("")

    if plan:
        parts.append("## Plan")
        parts.append("")
        for s in plan.steps:
            parts.append(f"- **{s.step_id}** — {s.intent}")
        parts.append("")

    if per_step_records:
        parts.append("## Step outcomes")
        parts.append("")
        for r in per_step_records:
            parts.append(
                f"- **{r['step_id']}** — status: `{r.get('status', '?')}`"
                + (f" (rc={r['returncode']})" if "returncode" in r else "")
            )
        parts.append("")

    parts.append("## Findings")
    parts.append("")
    if not findings:
        parts.append("- (no findings recorded)")
    else:
        for f in findings:
            parts.append(f"- `{f.severity}` [{f.validator}] {f.message}")
    parts.append("")

    parts.append("## Evidence (registered artefacts)")
    parts.append("")
    parts.append("| evidence_id | kind | description | sha256 (head) | path |")
    parts.append("|---|---|---|---|---|")
    pipe_escape = "\\|"
    for r in evidence.records():
        desc = r.description.replace("|", pipe_escape)
        parts.append(
            f"| `{r.evidence_id}` | {r.kind} | "
            f"{desc} | `{r.sha256[:10]}…` | `{r.relative_path}` |"
        )
    parts.append("")
    parts.append(
        textwrap.dedent(
            """
        ---
        Generated by `easyicu.research_agent.ResearchAgentPipeline`. Every entry
        in the Evidence table is reproducible: rerun the script identified by
        `script_evidence_id` in the manifest, hash the output, and confirm it
        matches the `sha256` recorded here.
    """
        ).strip()
    )
    return "\n".join(parts) + "\n"


__all__ = [
    "execution_gate_status",
    "render_report",
    "write_readiness_artifacts",
]
