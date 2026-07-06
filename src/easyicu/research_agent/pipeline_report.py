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

from .article_contract import (
    article_contract_audit_payload,
    summarize_article_contract_coverage,
)
from .display_suite import summarize_display_suite_status
from .evidence import EvidenceStore
from .figure_strategy import summarize_article_figure_strategy_coverage
from .publication_figures import PUBLICATION_FIGURE_SKILL_POLICY_VERSION
from .review_artifacts import build_review_artifact_payloads
from .runtime_artifacts import capture_code_version
from .schema import AnalysisPlan, ResearchContext, ValidationFinding


def _figure_steps_satisfied_by_repair(run_dir: Path) -> set:
    """Figure step_ids whose figure a successful rendering-only repair produced.

    A ``*_figure`` step can fail (its own runner emitted no exports) yet still be
    salvaged by a later ``*_figure_repair`` step that renders the figure into its
    OWN outputs dir. That repair step is not a required plan step, so the gate
    would otherwise still count the original figure step as ``execution_failed``.
    We credit it: a rendering-only repair step (``status == ok`` with a real
    rendered figure on disk) whose ``parent_step`` is ``P`` satisfies the figure
    step ``P + "_figure"``. Matching is exact via ``parent_step`` — no fuzzy
    token overlap — so an unrelated repair can never mask a genuine failure.
    """

    satisfied: set = set()
    steps_dir = run_dir / "steps"
    if not steps_dir.is_dir():
        return satisfied
    for summary in steps_dir.glob("*/outputs/step_summary.json"):
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if (
            not payload.get("rendering_only")
            or str(payload.get("status") or "") != "ok"
        ):
            continue
        parent = str(payload.get("parent_step") or "").strip()
        if not parent:
            continue
        outputs_dir = summary.parent
        has_rendered_figure = any(
            any(outputs_dir.glob(f"*{ext}"))
            for ext in (".png", ".svg", ".pdf", ".tiff")
        )
        if has_rendered_figure:
            satisfied.add(parent + "_figure")
    return satisfied


def execution_gate_status(
    *,
    plan: Optional[AnalysisPlan],
    per_step_records: Sequence[Dict[str, Any]],
    run_dir: Optional[Path] = None,
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
    missing_steps = [
        step_id for step_id in required_step_ids if step_id not in status_by_step
    ]
    # Credit ``*_figure`` steps whose figure a later rendering-only repair step
    # actually produced: the figure exists on disk, so a hard fail-close here
    # would be a false negative (the deliverable is present). Only figure steps
    # matched EXACTLY by a repair's ``parent_step`` are credited; every other
    # failure still blocks the gate. ``run_dir=None`` preserves legacy behaviour.
    repaired_figures: set = (
        _figure_steps_satisfied_by_repair(run_dir) if run_dir is not None else set()
    )

    def _step_ok(step_id: str) -> bool:
        return status_by_step.get(step_id) == "ok" or (
            step_id in repaired_figures and str(step_id).endswith("_figure")
        )

    failed_steps = [
        {"step_id": step_id, "status": status_by_step.get(step_id)}
        for step_id in required_step_ids
        if step_id in status_by_step and not _step_ok(step_id)
    ]
    return {
        "execution_complete": not missing_steps and not failed_steps,
        "required_step_count": len(required_step_ids),
        "completed_step_count": sum(
            1 for step_id in required_step_ids if _step_ok(step_id)
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


_OUTCOME_ENDPOINT_RE = re.compile(
    r"\b(?:death|mortality|survival|endpoint|event|outcome)s?\b",
    flags=re.IGNORECASE,
)
_SPECIFIC_OUTCOME_RE = re.compile(
    r"\b(?:death|mortality|survival)\b",
    flags=re.IGNORECASE,
)
_OUTCOME_INFERENCE_RE = re.compile(
    r"\b(?:association|contrast|comparison|difference|effect|estimate|"
    r"point estimate|odds ratio|hazard ratio|risk ratio|relative risk|"
    r"protective|harmful|near[- ]?null|equivalence|prognostic|"
    r"separation|vary|varies|varied|ranging)\b",
    flags=re.IGNORECASE,
)
_OUTCOME_GROUP_RE = re.compile(
    r"\b(?:death|mortality|outcome|endpoint|event)s?\b.{0,80}"
    r"\b(?:across|between|by)\b.{0,60}\b(?:group|subgroup|definition)s?\b",
    flags=re.IGNORECASE,
)
_OUTCOME_SAFE_BLOCK_RE = re.compile(
    r"\b(?:blocked|withheld|not authorized|not authorised|not performed|"
    r"not executed|not analysed|not analyzed|not inferred|cannot be inferred|"
    r"must not be inferred|no outcome claim|no death claim|no mortality claim|"
    r"before any|prior to any)\b",
    flags=re.IGNORECASE,
)


def _payload_mentions_outcome(payload: Any) -> bool:
    return bool(_OUTCOME_ENDPOINT_RE.search(json.dumps(payload, ensure_ascii=False)))


def _step_summary_blocks_outcome(payload: Dict[str, Any]) -> bool:
    if payload.get("grouped_death_analysis_executed") is False:
        return True
    if payload.get("exploratory_group_death_tabulation_authorized") is False:
        return True
    if payload.get(
        "primary_analysis_authorized"
    ) is False and _payload_mentions_outcome(payload):
        return True
    if payload.get("analysis_executed") is False:
        dumped = json.dumps(payload, ensure_ascii=False).lower()
        return "blocked" in dumped and _payload_mentions_outcome(payload)
    return False


def _blocked_outcome_step_ids(run_dir: Path) -> List[str]:
    blocked: set[str] = set()
    for path in sorted(run_dir.glob("steps/*/outputs/step_summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and _step_summary_blocks_outcome(payload):
            blocked.add(path.parents[1].name)
    for path in sorted(run_dir.glob("steps/*/outputs/*gate*.csv")):
        try:
            with path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        except Exception:
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_text = json.dumps(row, ensure_ascii=False).lower()
            false_authorization = any(
                str(value).strip().lower() == "false"
                and any(token in str(key).lower() for token in ("author", "execut"))
                for key, value in row.items()
            )
            if (
                str(row.get("status", "")).lower() == "blocked"
                and _OUTCOME_ENDPOINT_RE.search(row_text)
                and (false_authorization or "blocked" in row_text)
            ):
                blocked.add(path.parents[1].name)
                break
    return sorted(blocked)


def _blocked_outcome_manuscript_leaks(manuscript_text: str) -> List[str]:
    leaks: List[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", manuscript_text or ""):
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if not sentence:
            continue
        natural = re.sub(r"<!--.*?-->", "", sentence)
        natural = re.sub(r"\[[^\]]+\]\([^)]+\)", "", natural)
        natural = re.sub(r"\{\[[^\]]+\]\([^)]+\)\}", "", natural)
        natural = re.sub(r"\s+", " ", natural).strip()
        if _OUTCOME_SAFE_BLOCK_RE.search(natural):
            continue
        has_specific_endpoint = bool(_SPECIFIC_OUTCOME_RE.search(natural))
        has_generic_endpoint = bool(_OUTCOME_ENDPOINT_RE.search(natural))
        if not has_generic_endpoint:
            continue
        inference = bool(_OUTCOME_INFERENCE_RE.search(natural))
        grouped = bool(_OUTCOME_GROUP_RE.search(natural))
        if grouped or (has_specific_endpoint and inference):
            leaks.append(natural[:600])
    return leaks


def _record_artifact_basename(record: Any) -> str:
    return Path(str(record.relative_path)).name.split("__", 1)[-1]


def _source_fingerprints_match(
    evidence: EvidenceStore, metadata: Dict[str, Any]
) -> bool:
    source_ids = metadata.get("source_evidence_ids")
    if isinstance(source_ids, str):
        ids = [source_ids]
    elif isinstance(source_ids, (list, tuple, set)):
        ids = [str(eid) for eid in source_ids if str(eid)]
    else:
        ids = []
    single = metadata.get("source_evidence_id")
    if single and str(single) not in ids:
        ids.append(str(single))
    fingerprints = metadata.get("source_evidence_sha256")
    if not ids or not isinstance(fingerprints, dict) or not fingerprints:
        return False
    for evidence_id in ids:
        source = evidence.get(evidence_id)
        if source is None or fingerprints.get(evidence_id) != source.sha256:
            return False
    return True


def _publication_figure_policy_matches(metadata: Dict[str, Any]) -> bool:
    return (
        metadata.get("figure_skill_policy_version")
        == PUBLICATION_FIGURE_SKILL_POLICY_VERSION
    )


def _run_level_publication_skill_record(record: Any) -> bool:
    return record.producer == "publication_figure_skill" and _record_artifact_basename(
        record
    ).startswith("easyicu_publication_figure.")


_PUBLICATION_FIGURE_VISUAL_ERROR_VALIDATORS = {
    "publication_figure_export",
    "visual_qa",
    "vlm_visual_qa",
}


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
        and finding.validator in _PUBLICATION_FIGURE_VISUAL_ERROR_VALIDATORS
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
        if _run_level_publication_skill_record(record) and (
            not _source_fingerprints_match(evidence, metadata)
            or not _publication_figure_policy_matches(metadata)
        ):
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
        "publication_figure_visual_qa_error_count": len(visual_errors),
        "publication_figure_visual_qa_errors": [
            {"validator": finding.validator, "message": finding.message}
            for finding in visual_errors
        ],
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
        if isinstance(rec, dict) and rec.get("status") == "ok" and rec.get("step_id")
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
    (
        "evidence_bound_writer",
        "strict evidence enforcement blocked manuscript generation",
        "manuscript_bound_clean",
    ),
    (
        "manuscript_numeric_auditor",
        "strict evidence enforcement blocked manuscript generation",
        "manuscript_numeric_bound_clean",
    ),
    (
        "critic_agent",
        "criticagent marked manuscript",
        "manuscript_critique_passed",
    ),
    # A caveat-count finding cannot tell which writer pass it came
    # from, so an earlier pass's "cites records with unresolved
    # manifest caveats" error survives a later clean rewrite (e.g. a
    # resume whose new draft cites only caveat-free records). Gate it
    # on the CURRENT bound text: if the latest manuscript carries no
    # `<!-- warning|error: see manifest -->` comments, the finding is
    # stale; if the latest text still has caveats, it stays active.
    (
        "evidence_bound_writer",
        "unresolved manifest caveats",
        "manuscript_manifest_caveats_clean",
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


def _manuscript_numeric_bound_clean(manuscript_text: str) -> bool:
    """Return true when the latest bound manuscript has no numeric gap markers."""
    if not manuscript_text:
        return False
    if "Manuscript scaffold not generated" in manuscript_text[:300]:
        return False
    return (
        "<!-- UNTRACED:" not in manuscript_text
        and "<!-- AMBIGUOUS:" not in manuscript_text
        and not _MANIFEST_COMMENT_RE.search(manuscript_text)
    )


_WRITER_FAILURE_RE = re.compile(
    r"\b(?:writer failed|error code|invalid proxy api key|api key|"
    r"connection error|rate limit|authentication)\b",
    flags=re.IGNORECASE,
)

_MANIFEST_COMMENT_RE = re.compile(
    r"<!--\s*(?P<level>warning|error)\s*:\s*see manifest\s*-->",
    flags=re.IGNORECASE,
)


def _manuscript_text_status(manuscript_text: str) -> Dict[str, Any]:
    """Validate that a manuscript file contains a real evidence-bound draft.

    Readiness used to treat any non-placeholder manuscript path as generated.
    That allowed a one-line writer exception such as ``(writer failed: ...)`` or
    an empty bound file to pass downstream gates. The check intentionally stays
    case-neutral: it validates draft substance and evidence binding, not any
    particular benchmark topic, variable, or figure.
    """

    text = str(manuscript_text or "")
    stripped = text.strip()
    errors: List[str] = []
    if not stripped:
        errors.append("manuscript draft is empty")
    head = stripped[:600]
    if "Manuscript scaffold not generated" in head:
        errors.append("manuscript scaffold was not generated")
    if _WRITER_FAILURE_RE.search(head):
        errors.append("manuscript draft contains a writer/runtime failure message")
    word_count = len(re.findall(r"[A-Za-z][A-Za-z0-9-]*", stripped))
    if word_count < 8:
        errors.append("manuscript draft has too little prose content")
    if not re.search(r"(?:\]\(evidence/|\{evidence:|\[\^claim_)", stripped):
        errors.append("manuscript draft has no evidence-bound claim links")
    manifest_comment_counts = {"warning": 0, "error": 0}
    for match in _MANIFEST_COMMENT_RE.finditer(stripped):
        level = match.group("level").lower()
        manifest_comment_counts[level] = manifest_comment_counts.get(level, 0) + 1
    if manifest_comment_counts["error"]:
        errors.append(
            "manuscript draft contains "
            f"{manifest_comment_counts['error']} unresolved manifest error comment(s)"
        )
    if manifest_comment_counts["warning"]:
        errors.append(
            "manuscript draft contains "
            f"{manifest_comment_counts['warning']} unresolved manifest warning comment(s)"
        )
    return {
        "manuscript_text_ready": not errors,
        "manuscript_text_errors": errors,
        "manuscript_word_count": word_count,
        "manuscript_manifest_warning_count": manifest_comment_counts["warning"],
        "manuscript_manifest_error_count": manifest_comment_counts["error"],
    }


_PUBLICATION_FIGURE_SUMMARY_RE = re.compile(
    r"^publication_figure_skill_summary(?:_v(?P<version>\d+))?"
    r"__publication_figure_skill_summary\.json$"
)


def _publication_figure_summary_sort_key(path: Path) -> tuple[int, str]:
    match = _PUBLICATION_FIGURE_SUMMARY_RE.match(path.name)
    if not match:
        return (0, path.name)
    version = int(match.group("version") or "1")
    return (version, path.name)


def _latest_publication_figure_audit_status(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Return the latest versioned publication-figure audit summary, if any."""
    candidates = sorted(
        (run_dir / "evidence").glob(
            "publication_figure_skill_summary*__publication_figure_skill_summary.json"
        ),
        key=_publication_figure_summary_sort_key,
    )
    for path in reversed(candidates):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        findings = payload.get("audit_findings")
        if not isinstance(findings, list):
            continue
        errors = [
            item
            for item in findings
            if isinstance(item, dict) and item.get("severity") == "error"
        ]
        return {
            "path": str(path),
            "error_count": len(errors),
            "errors": errors,
        }
    return None


def _finding_references_publication_figure(finding: ValidationFinding) -> bool:
    detail_text = ""
    if finding.detail:
        try:
            detail_text = json.dumps(finding.detail, ensure_ascii=False, default=str)
        except Exception:
            detail_text = str(finding.detail)
    haystack = " ".join(
        [
            finding.validator or "",
            finding.message or "",
            " ".join(finding.evidence_ids or []),
            detail_text,
        ]
    ).lower()
    return "publication_figure" in haystack or "easyicu_publication_figure" in haystack


def _is_publication_figure_audit_superseded(
    finding: ValidationFinding,
    *,
    latest_publication_audit: Optional[Dict[str, Any]],
) -> bool:
    """True when a newer publication-figure audit has resolved an older error."""
    if not latest_publication_audit:
        return False
    if latest_publication_audit.get("error_count") != 0:
        return False
    if finding.severity != "error":
        return False
    if finding.validator not in {
        "publication_figure_export",
        "figure_contract_quality",
        "visual_qa",
        "vlm_visual_qa",
    }:
        return False
    return _finding_references_publication_figure(finding)


# ---------------------------------------------------------------------------
# Primary-result plausibility gate ("table == reality").
#
# The value-level numeric auditor verifies manuscript-number == table-number,
# but never table-number == reality. A generated primary table can be
# internally consistent yet physically impossible — e.g. a positional column
# swap that makes the Cox "event" column the sum of ages, yielding 4.6M events
# for 73k stays and a 0h median follow-up. The manuscript then faithfully
# transcribes the garbage and every value-level gate passes. This gate flags
# ONLY values that cannot occur for ANY question (never a question-specific
# direction, threshold, or magnitude), keeping shared gates case-neutral.
# ---------------------------------------------------------------------------
_PLAUSIBILITY_EVENT_KEYS = (
    "events",
    "n_events",
    "n_events_model",
    "num_events",
    "event_count",
)
_PLAUSIBILITY_N_KEYS = (
    "n",
    "n_model",
    "n_analysis",
    "n_analytic",
    "modeled_analytic_n",
    "n_complete_case",
    "n_complete_case_primary_model",
    "n_primary_complete_case",
    "n_stays",
    "n_patients",
    "n_obs",
    "n_full",
)
_PLAUSIBILITY_RATIO_KEYS = ("hazard_ratio", "odds_ratio", "risk_ratio")
_PLAUSIBILITY_RATE_KEYS = (
    "event_rate",
    "outcome_rate",
    "death_rate",
    "mortality_rate",
)
_PLAUSIBILITY_RESULT_MARKERS = (
    "hazard_ratio",
    "odds_ratio",
    "risk_ratio",
    "estimate",
    "point_estimate",
    "p_value",
    "pvalue",
    "log_hazard_ratio",
)
_PLAUSIBILITY_RESULT_CSVS = (
    "cox_summary.csv",
    "cox_model.csv",
    "adjusted_cox_model.csv",
    "hazard_ratio.csv",
    "adjusted_association.csv",
    "association_model_summary.csv",
    "crude_vs_adjusted_association.csv",
)


def _plausibility_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        x = float(value)
    elif isinstance(value, str):
        try:
            x = float(value.strip())
        except (ValueError, AttributeError):
            return None
    else:
        return None
    # finite only (reject NaN / +-inf without importing math)
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return x


def _plausibility_first(
    mapping: Dict[str, Any], keys: Sequence[str]
) -> Optional[float]:
    for key in keys:
        if key in mapping:
            num = _plausibility_number(mapping[key])
            if num is not None:
                return num
    return None


def _plausibility_errors_for_row(where: str, row: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    # events <= n — guarded to model-result rows to avoid flagging unrelated
    # count dicts that merely happen to carry both keys.
    if any(marker in row for marker in _PLAUSIBILITY_RESULT_MARKERS):
        events = _plausibility_first(row, _PLAUSIBILITY_EVENT_KEYS)
        n = _plausibility_first(row, _PLAUSIBILITY_N_KEYS)
        if events is not None and n is not None and n > 0 and events > n:
            errs.append(
                f"{where}: implausible primary result — {int(events)} events "
                f"exceed {int(n)} analysis units; an event count cannot exceed "
                "the sample (a corrupted/column-swapped result table)."
            )
    rate = _plausibility_first(row, _PLAUSIBILITY_RATE_KEYS)
    if rate is not None and (rate < 0.0 or rate > 1.0):
        errs.append(
            f"{where}: implausible event rate {rate} (a proportion must be "
            "within [0, 1])."
        )
    ratio = _plausibility_first(row, _PLAUSIBILITY_RATIO_KEYS)
    if ratio is not None and ratio <= 0.0:
        errs.append(
            f"{where}: implausible ratio estimate {ratio} (a hazard/odds/risk "
            "ratio must be > 0)."
        )
    lo = _plausibility_first(row, ("ci_low",))
    hi = _plausibility_first(row, ("ci_high",))
    if lo is not None and hi is not None and lo > hi:
        errs.append(
            f"{where}: inverted confidence interval (ci_low {lo} > ci_high {hi})."
        )
    return errs


def _plausibility_walk(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _plausibility_walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _plausibility_walk(item)


def primary_result_plausibility_errors(run_dir: Path) -> List[str]:
    """Return case-neutral ``table == reality`` violations in primary artefacts.

    Scans every step's ``step_summary.json`` (recursively) and the known
    primary-result CSVs for values that are physically impossible for any
    question. Returns an empty list for a healthy run; a non-empty list is a
    fail-closed analysis error so the run cannot reach ``manuscript_ready``.
    """

    errors: List[str] = []
    seen: set = set()
    steps_dir = run_dir / "steps"
    if not steps_dir.is_dir():
        return errors

    def _add(new_errors: List[str]) -> None:
        for err in new_errors:
            if err not in seen:
                seen.add(err)
                errors.append(err)

    for summary in sorted(steps_dir.glob("*/outputs/step_summary.json")):
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
        except Exception:
            continue
        label = summary.parent.parent.name
        for mapping in _plausibility_walk(payload):
            _add(_plausibility_errors_for_row(label, mapping))

    for outputs_dir in sorted(steps_dir.glob("*/outputs")):
        for name in _PLAUSIBILITY_RESULT_CSVS:
            path = outputs_dir / name
            if not path.exists():
                continue
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        _add(
                            _plausibility_errors_for_row(
                                f"{outputs_dir.parent.name}/{name}", dict(row)
                            )
                        )
            except Exception:
                continue
    return errors


# --- primary survival estimand integrity -------------------------------------
# The PRIMARY time-to-event estimate must come from the deterministic Cox runner
# (reproducible, correct exposure, no positional column swaps). When an LLM coder
# produces it instead, the estimate is unverified and can silently fabricate an
# implausible Cox model, including positional column swaps where non-event values
# are interpreted as events. This gate fails closed on that path regardless of
# *why* the deterministic runner did not fire (disabled flag, unmet preflight,
# already-consumed fallback). It is case-neutral: it only fires for plans that
# declare a survival / time-to-event PRIMARY step, so association / prediction /
# clustering questions are untouched.
_SURVIVAL_PRIMARY_METHODS = ("survival_analysis", "time_to_event", "cox", "cox_ph")
# Keys the deterministic Cox runner always writes into its step_summary; together
# they fingerprint that step_summary as runner-produced (see deterministic_survival).
_DETERMINISTIC_SURVIVAL_MARKERS = ("fit_engine", "adjustment_source")
# Keys that show a step actually REPORTED a survival estimate, so an empty or
# prep-only survival step is not misread as a fabricated result.
_SURVIVAL_RESULT_KEYS = (
    "hazard_ratio",
    "primary_model",
    "cox_terms",
    "log_hazard_ratio",
)


def _is_survival_method_step(step: Any) -> bool:
    """True for a survival / time-to-event analysis step (not a figure step).

    Deliberately does NOT apply the execution predicate's ``sensitivity`` +
    ``definition`` exclusion: that heuristic wrongly excludes a PRIMARY survival
    step whose intent merely *mentions* sensitivity analyses, and excluding it
    here would hide exactly the LLM-coded result this gate must
    catch. A genuine cohort-definition-sensitivity step is filtered out later by
    the result-key check (it never emits a primary Cox ``hazard_ratio`` /
    ``primary_model``).
    """
    method = str(getattr(step, "method", "") or "").lower()
    if method not in _SURVIVAL_PRIMARY_METHODS:
        return False
    step_id = str(getattr(step, "step_id", "") or "").lower()
    return "figure" not in step_id


def _survival_summary_is_deterministic(payload: Dict[str, Any]) -> bool:
    return payload.get(
        "deterministic_standard_analysis"
    ) == "survival_primary_cox" or all(
        key in payload for key in _DETERMINISTIC_SURVIVAL_MARKERS
    )


def primary_survival_estimate_integrity_errors(
    plan: Optional[AnalysisPlan], run_dir: Optional[Path]
) -> List[str]:
    """Fail closed if a survival estimand did not come from the runner.

    Returns ``[]`` for any question without a survival / time-to-event step and
    for a survival design where the primary Cox estimate carries the
    deterministic runner fingerprint. When a survival design produced a primary
    Cox estimate (``hazard_ratio`` / ``primary_model``) but NO survival result
    step carries the fingerprint, the estimand is an unverified LLM-coded result
    and this returns a fail-closed analysis error.
    """
    if plan is None or run_dir is None:
        return []
    result_steps: List[str] = []
    any_deterministic = False
    for step in getattr(plan, "steps", None) or []:
        if not _is_survival_method_step(step):
            continue
        step_id = str(getattr(step, "step_id", "") or "")
        summary_path = run_dir / "steps" / step_id / "outputs" / "step_summary.json"
        if not summary_path.exists():
            # A missing summary is the execution gate's concern (missing/failed
            # step); this gate only judges a step that DID produce a summary.
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if _survival_summary_is_deterministic(payload):
            any_deterministic = True
            continue
        if any(key in payload for key in _SURVIVAL_RESULT_KEYS):
            result_steps.append(step_id)
    if result_steps and not any_deterministic:
        joined = ", ".join(sorted(result_steps))
        return [
            "primary survival estimand was not produced by the deterministic Cox "
            f"runner (steps {joined} reported a Cox estimate with no runner "
            "fingerprint -- unverified LLM-coded survival result; fail closed)"
        ]
    return []


def _partition_findings_by_supersession(
    findings: Sequence[ValidationFinding],
    *,
    success_step_ids: set,
    known_step_ids: Optional[set] = None,
    gate_state: Optional[Dict[str, bool]] = None,
    latest_publication_audit: Optional[Dict[str, Any]] = None,
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
    4. It is an older publication-figure audit error and the latest
       versioned publication-figure skill summary has no current
       errors. This keeps repaired figure exports from being blocked
       by stale resume-era QA findings while preserving the historical
       finding in the audit trail.

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
        if _is_publication_figure_audit_superseded(
            f,
            latest_publication_audit=latest_publication_audit,
        ):
            superseded.append(f)
            continue
        active.append(f)
    return active, superseded


def _compute_readiness_gates(
    *,
    context: ResearchContext,
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
    execution = execution_gate_status(
        plan=plan, per_step_records=per_step_records, run_dir=run_dir
    )
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
        "manuscript_bound_clean": bool(
            manuscript_text
            and "Manuscript scaffold not generated" not in manuscript_text[:300]
            and missing_evidence_count == 0
            and not stop_after_analysis
            and not writer_probe_mode
        ),
        "manuscript_numeric_bound_clean": bool(
            manuscript_text
            and _manuscript_numeric_bound_clean(manuscript_text)
            and not stop_after_analysis
            and not writer_probe_mode
        ),
        "manuscript_manifest_caveats_clean": bool(
            manuscript_text
            and "Manuscript scaffold not generated" not in manuscript_text[:300]
            and not _MANIFEST_COMMENT_RE.search(manuscript_text)
            and not stop_after_analysis
            and not writer_probe_mode
        ),
        "manuscript_critique_passed": False,
    }
    critique_path = run_dir / "manuscript_critique.json"
    if critique_path.exists():
        try:
            critique_payload = json.loads(critique_path.read_text(encoding="utf-8"))
        except Exception:
            critique_payload = {}
        current_gate_state["manuscript_critique_passed"] = (
            isinstance(critique_payload, dict)
            and critique_payload.get("status") == "pass"
        )
    latest_publication_audit = _latest_publication_figure_audit_status(run_dir)
    active_findings, superseded_findings = _partition_findings_by_supersession(
        findings,
        success_step_ids=success_step_ids,
        known_step_ids=known_step_ids,
        gate_state=current_gate_state,
        latest_publication_audit=latest_publication_audit,
    )
    numeric_errors = [
        f.message
        for f in active_findings
        if f.severity == "error" and f.validator == "manuscript_numeric_auditor"
    ]
    evidence_errors = [
        f.message
        for f in active_findings
        if f.severity == "error"
        and f.validator in {"evidence_bound_writer", "critic_agent"}
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
    blocked_outcome_steps = _blocked_outcome_step_ids(run_dir)
    blocked_outcome_leaks = (
        _blocked_outcome_manuscript_leaks(manuscript_text)
        if blocked_outcome_steps
        else []
    )
    blocked_outcome_errors = [
        "blocked outcome gate leaked into manuscript: " + leak
        for leak in blocked_outcome_leaks
    ]
    manuscript_text_gate = _manuscript_text_status(manuscript_text)
    manuscript_generated = (
        not writer_probe_mode
        and manuscript_path.exists()
        and manuscript_text_gate["manuscript_text_ready"]
        and not stop_after_analysis
    )
    evidence_complete = (
        manuscript_generated and missing_evidence_count == 0 and not evidence_errors
    )
    numeric_verified = manuscript_generated and not numeric_errors
    # `table == reality` gate: a physically-impossible primary result (e.g. a
    # column-swapped Cox table with more events than patients) fails closed even
    # though the value-level numeric auditor — which only checks
    # manuscript-number == table-number — would pass it.
    plausibility_errors = primary_result_plausibility_errors(run_dir)
    # Integrity: a survival PRIMARY estimand must come from the deterministic Cox
    # runner, never an LLM coder that may silently swap columns.
    survival_integrity_errors = primary_survival_estimate_integrity_errors(
        plan, run_dir
    )
    analysis_errors = (
        non_manuscript_errors
        + blocked_outcome_errors
        + plausibility_errors
        + survival_integrity_errors
    )
    analysis_validated = execution["execution_complete"] and not analysis_errors
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
    display_suite = summarize_display_suite_status(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=run_dir,
        publication=publication,
    )
    article_contract = summarize_article_contract_coverage(
        context=context,
        plan=plan,
        evidence_records=evidence.records(),
        per_step_records=per_step_records,
        run_dir=run_dir,
    )
    figure_strategy = summarize_article_figure_strategy_coverage(
        context=context,
        run_dir=run_dir,
    )
    return {
        **execution,
        "evidence_complete": evidence_complete,
        "numeric_verified": numeric_verified,
        "analysis_validated": analysis_validated,
        "manuscript_ready": manuscript_ready,
        "publication_ready": manuscript_ready
        and publication["publication_figure_bundle_ready"]
        and display_suite["display_suite_complete"]
        and article_contract["article_contract_complete"]
        and figure_strategy["article_figure_strategy_complete"],
        "manuscript_generated": manuscript_generated,
        **manuscript_text_gate,
        "writer_probe_mode": bool(writer_probe_mode),
        "writer_probe_failed_steps": list(writer_probe_failed_steps or []),
        "missing_evidence_count": missing_evidence_count,
        "numeric_error_count": len(numeric_errors),
        "evidence_error_count": len(evidence_errors),
        "analysis_error_count": len(analysis_errors),
        "numeric_errors": numeric_errors,
        "evidence_errors": evidence_errors,
        "analysis_errors": analysis_errors,
        "blocked_outcome_step_ids": blocked_outcome_steps,
        "blocked_outcome_not_leaked": not blocked_outcome_leaks,
        "blocked_outcome_leak_count": len(blocked_outcome_leaks),
        "blocked_outcome_leaks": blocked_outcome_leaks,
        # Audit-trail surface for the supersession rule (see
        # _partition_findings_by_supersession). Reviewers can inspect
        # which findings the readiness gate ignored because the
        # underlying step ultimately succeeded.
        "superseded_error_count": sum(
            1 for f in superseded_findings if f.severity == "error"
        ),
        "superseded_errors": [
            {"validator": f.validator, "message": f.message}
            for f in superseded_findings
            if f.severity == "error"
        ],
        **publication,
        **display_suite,
        **article_contract,
        **figure_strategy,
    }


def _count_writer_attempts(run_dir: Path) -> Optional[int]:
    """Count writer drafting passes from the run's audit_log.jsonl.

    Each writer pass emits a ``"Drafting manuscript scaffold."`` event; on
    a resumed run these accumulate, so the count is a cheap fragility proxy
    (attempts-to-ready). Returns None when the audit log is absent — older
    runs, or a run that failed before the writer phase.
    """
    audit_path = run_dir / "audit_log.jsonl"
    if not audit_path.exists():
        return None
    count = 0
    try:
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            if str(event.get("event", "")).startswith("Drafting manuscript scaffold"):
                count += 1
    except Exception:
        return None
    return count


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
        context=context,
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
    # Attempts-to-ready: how many writer passes this run needed before the
    # gates were satisfied. Derived from the always-on audit_log.jsonl event
    # stream (no separate event artifact) so the gate story is quantitative
    # — a run that took 4 writer passes is more fragile than one that took 1.
    gates["writer_attempt_count"] = _count_writer_attempts(run_dir)
    status = (
        "publication_ready"
        if gates["publication_ready"]
        else (
            "manuscript_ready"
            if gates["manuscript_ready"]
            else "analysis_only" if gates["execution_complete"] else "diagnostic_only"
        )
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
        # Code identity for quick access without opening the full manifest;
        # the authoritative copy lives in manifest.json's ``code_version``.
        "code_version": capture_code_version(),
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
        "manuscript_path": (
            str(manuscript_path.relative_to(run_dir))
            if manuscript_path.exists()
            else None
        ),
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

    display_suite_path = run_dir / "display_suite_audit.json"
    display_suite_payload = {
        "schema_version": "easyicu.display_suite_audit/2",
        "display_suite_complete": gates["display_suite_complete"],
        "table_count": gates["display_table_count"],
        "figure_contract_count": gates["display_figure_contract_count"],
        "result_figure_contract_count": gates["display_result_figure_contract_count"],
        "primary_publication_figure_contract_count": gates[
            "display_primary_publication_figure_contract_count"
        ],
        "supporting_figure_contract_count": gates[
            "display_supporting_figure_contract_count"
        ],
        "other_figure_contract_count": gates["display_other_figure_contract_count"],
        "primary_publication_contract_paths": gates[
            "display_primary_publication_contract_paths"
        ],
        "supporting_figure_contract_paths": gates[
            "display_supporting_figure_contract_paths"
        ],
        "other_figure_contract_paths": gates["display_other_figure_contract_paths"],
        "contract_panel_count": gates["display_contract_panel_count"],
        "primary_publication_panel_count": gates[
            "display_primary_publication_panel_count"
        ],
        "supporting_panel_count": gates["display_supporting_panel_count"],
        "contract_role_count": gates["display_contract_role_count"],
        "primary_publication_role_count": gates[
            "display_primary_publication_role_count"
        ],
        "supporting_role_count": gates["display_supporting_role_count"],
        "chart_types": gates["display_chart_types"],
        "primary_publication_chart_types": gates[
            "display_primary_publication_chart_types"
        ],
        "supporting_chart_types": gates["display_supporting_chart_types"],
        "absolute_risk_visual_present": gates["display_absolute_risk_visual_present"],
        "primary_publication_absolute_risk_visual_present": gates[
            "display_primary_publication_absolute_risk_visual_present"
        ],
        "supporting_absolute_risk_visual_present": gates[
            "display_supporting_absolute_risk_visual_present"
        ],
        "primary_publication_result_figure_contract_count": gates[
            "display_primary_publication_result_figure_contract_count"
        ],
        "supporting_result_figure_contract_count": gates[
            "display_supporting_result_figure_contract_count"
        ],
        "categories": gates["display_categories"],
        "table_one_expected": gates["display_table_one_expected"],
        "table_one_present": gates["display_table_one_present"],
        "audit_context_present": gates["display_audit_context_present"],
        "errors": gates["display_suite_errors"],
    }
    display_suite_path.write_text(
        json.dumps(display_suite_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    artifact_paths["display_suite_audit"] = str(display_suite_path.relative_to(run_dir))

    article_contract_path = run_dir / "article_contract_audit.json"
    article_contract_payload = article_contract_audit_payload(gates)
    article_contract_path.write_text(
        json.dumps(
            article_contract_payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    artifact_paths["article_contract_audit"] = str(
        article_contract_path.relative_to(run_dir)
    )

    figure_strategy_path = run_dir / "article_figure_strategy_audit.json"
    figure_strategy_payload = {
        "schema_version": gates["article_figure_strategy_audit_schema_version"],
        "article_figure_strategy_complete": gates["article_figure_strategy_complete"],
        "analysis_family": gates["article_figure_strategy_family"],
        "archetype": gates["article_figure_strategy_archetype"],
        "hero_role": gates["article_figure_strategy_hero_role"],
        "required_roles": gates["article_figure_strategy_required_roles"],
        "covered_roles": gates["article_figure_strategy_covered_roles"],
        "missing_roles": gates["article_figure_strategy_missing_roles"],
        "chart_types": gates["article_figure_strategy_chart_types"],
        "primary_publication_roles": gates[
            "article_figure_strategy_primary_publication_roles"
        ],
        "primary_publication_chart_types": gates[
            "article_figure_strategy_primary_publication_chart_types"
        ],
        "primary_publication_panel_count": gates[
            "article_figure_strategy_primary_publication_panel_count"
        ],
        "primary_publication_minimum_required_role_count": gates[
            "article_figure_strategy_primary_publication_minimum_required_role_count"
        ],
        "primary_publication_role_panels": gates[
            "article_figure_strategy_primary_publication_role_panels"
        ],
        "minimum_distinct_chart_types": gates[
            "article_figure_strategy_minimum_distinct_chart_types"
        ],
        "role_panels": gates["article_figure_strategy_role_panels"],
        "errors": gates["article_figure_strategy_errors"],
        "strategy": gates["article_figure_strategy"],
    }
    figure_strategy_path.write_text(
        json.dumps(
            figure_strategy_payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    artifact_paths["article_figure_strategy_audit"] = str(
        figure_strategy_path.relative_to(run_dir)
    )

    claim_ledger_path = run_dir / "claim_ledger.csv"
    claim_rows = _extract_claim_ledger_rows(
        manuscript_path=manuscript_path, gates=gates
    )
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

    review_payload, figure_gallery_payload, canonical_figure_paths = (
        build_review_artifact_payloads(run_dir=run_dir, gates=gates)
    )
    review_artifacts_path = run_dir / "review_artifacts.json"
    review_artifacts_path.write_text(
        json.dumps(review_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    artifact_paths["review_artifacts"] = str(review_artifacts_path.relative_to(run_dir))
    figure_gallery_path = run_dir / "figure_gallery.json"
    figure_gallery_path.write_text(
        json.dumps(
            figure_gallery_payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    artifact_paths["figure_gallery"] = str(figure_gallery_path.relative_to(run_dir))

    run_status_payload["canonical_outputs"] = {
        **artifact_paths,
        **canonical_figure_paths,
    }
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
            "display_suite_audit",
            "statistic",
            "Article display-suite coverage audit for publication gating.",
            display_suite_path,
        ),
        (
            "claim_ledger",
            "table",
            "Ledger of manuscript claims and evidence links.",
            claim_ledger_path,
        ),
        (
            "article_figure_strategy_audit",
            "log",
            "Article figure-strategy audit for publication gating.",
            figure_strategy_path,
        ),
        (
            "review_artifacts",
            "log",
            "Reviewer-facing artifact manifest with primary and supporting figure tiers.",
            review_artifacts_path,
        ),
        (
            "figure_gallery",
            "log",
            "Reviewer-facing figure gallery with primary and supporting figure tiers.",
            figure_gallery_path,
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
    for leak in gates.get("blocked_outcome_leaks") or []:
        rows.append(
            {
                "claim_id": f"claim_{len(rows) + 1:03d}",
                "claim_text": re.sub(r"\s+", " ", str(leak))[:1000],
                "evidence_refs": "",
                "status": "blocked_outcome_leak",
                "note": "Outcome linkage was blocked by a run gate, so this manuscript inference is not authorized.",
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
        f"- display_suite_complete: `{gates.get('display_suite_complete')}`",
        "- primary_publication_figure_contracts: "
        f"`{gates.get('display_primary_publication_figure_contract_count')}`",
        "- supporting_figure_contracts: "
        f"`{gates.get('display_supporting_figure_contract_count')}`",
        f"- publication_ready: `{gates['publication_ready']}`",
        "",
    ]
    superseded_error_keys = {
        (str(item.get("validator") or ""), str(item.get("message") or ""))
        for item in (gates.get("superseded_errors") or [])
        if isinstance(item, dict)
    }
    failed_steps = gates.get("failed_steps") or []
    missing_steps = gates.get("missing_steps") or []
    if failed_steps or missing_steps:
        lines.extend(["## Blocking step issues", ""])
        for item in failed_steps:
            lines.append(f"- `{item.get('step_id')}` status `{item.get('status')}`")
        for step_id in missing_steps:
            lines.append(f"- `{step_id}` missing execution record")
    if gates.get("blocked_outcome_step_ids") or gates.get("blocked_outcome_leaks"):
        lines.extend(["", "## Blocked outcome gate", ""])
        for step_id in gates.get("blocked_outcome_step_ids") or []:
            lines.append(f"- `{step_id}` blocked outcome linkage or tabulation.")
        for leak in gates.get("blocked_outcome_leaks") or []:
            lines.append(f"- Manuscript leak: {str(leak)[:240]}")
        lines.append("")
    if gates.get("display_suite_errors"):
        lines.extend(["", "## Display suite gate", ""])
        for error in gates.get("display_suite_errors") or []:
            lines.append(f"- {error}")
        lines.append("")
    if gates.get("manuscript_text_errors"):
        lines.extend(["", "## Manuscript text gate", ""])
        for error in gates.get("manuscript_text_errors") or []:
            lines.append(f"- {error}")
        lines.append("")
    error_findings = [
        f
        for f in findings
        if f.severity == "error"
        and (str(f.validator or ""), str(f.message or "")) not in superseded_error_keys
    ]
    superseded_error_findings = [
        f
        for f in findings
        if f.severity == "error"
        and (str(f.validator or ""), str(f.message or "")) in superseded_error_keys
    ]
    if error_findings:
        lines.extend(["## Blocking findings", ""])
        for finding in error_findings:
            lines.append(f"- `{finding.validator}`: {finding.message}")
        lines.append("")
    if superseded_error_findings:
        lines.extend(
            [
                "## Superseded findings",
                "",
                "These findings are retained in the manifest audit trail but do "
                "not block the current readiness gates.",
                "",
            ]
        )
        for finding in superseded_error_findings:
            lines.append(f"- `{finding.validator}`: {finding.message}")
        lines.append("")
    manuscript_text_errors = gates.get("manuscript_text_errors") or []
    if (
        not error_findings
        and not failed_steps
        and not missing_steps
        and not manuscript_text_errors
    ):
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
            else (
                "MANUSCRIPT READY"
                if readiness.get("manuscript_ready")
                else "DIAGNOSTIC ONLY"
            )
        )
        parts.append(f"## Status: {status}")
        parts.append("")
        parts.append(
            "- Gates: execution_complete={execution_complete}, "
            "evidence_complete={evidence_complete}, "
            "numeric_verified={numeric_verified}, "
            "analysis_validated={analysis_validated}, "
            "manuscript_ready={manuscript_ready}, "
            "display_suite_complete={display_suite_complete}, "
            "publication_ready={publication_ready}".format(**readiness)
        )
        parts.append("")
        primary_contracts = (
            readiness.get("display_primary_publication_contract_paths") or []
        )
        supporting_contracts = (
            readiness.get("display_supporting_figure_contract_paths") or []
        )
        if primary_contracts or supporting_contracts:
            parts.append("## Figure display tiers")
            parts.append("")
            if primary_contracts:
                parts.append("- Primary publication contracts:")
                parts.extend(f"  - `{path}`" for path in primary_contracts)
            else:
                parts.append("- Primary publication contracts: none")
            if supporting_contracts:
                parts.append("- Supporting step contracts:")
                parts.extend(f"  - `{path}`" for path in supporting_contracts)
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
    parts.append(textwrap.dedent("""
        ---
        Generated by `easyicu.research_agent.ResearchAgentPipeline`. Every entry
        in the Evidence table is reproducible: rerun the script identified by
        `script_evidence_id` in the manifest, hash the output, and confirm it
        matches the `sha256` recorded here.
    """).strip())
    return "\n".join(parts) + "\n"


__all__ = [
    "execution_gate_status",
    "render_report",
    "write_readiness_artifacts",
]
