"""[Layer 4: Evidence & Provenance / Layer 2: LLM Orchestration]
Write phase for the EasyICU research-agent pipeline.

This module is callable as ``run_write_phase(pipeline, ...)``. It reads
configuration and collaborators from the pipeline instance, matching the
``pipeline_execute.py`` free-function pattern, and returns the existing
``_WritePhaseResult`` boundary object.

Boundary contract: consumes ``_PlanPhaseResult`` + ``_ExecutePhaseResult``
and emits ``_WritePhaseResult``. The dataclasses live in ``contracts.py`` so
all phase modules share one handoff vocabulary.
"""

from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .agents import CriticAgent, ManuscriptAgent
from .audits.manuscript_claims import audit_manuscript_numeric_claims
from .bibtex import render_bibtex
from .causal_audit import run_causal_audit
from .contracts import (
    ValidationFinding,
    _ExecutePhaseResult,
    _PlanPhaseResult,
    _WritePhaseResult,
)
from .evidence import (
    EvidenceEnforcementError,
    EvidenceEnforcementMode,
    sha256_of_file,
)
from .figure_skill import PublicationFigureSkill
from .latex import scaffold_to_latex
from .literature import LiteratureAgent, LiteratureBundle
from .llm import MockLLMClient
from .manuscript_post import (
    bind_numeric_values,
    enforce_writer_claim_language,
    _demote_unresolved_evidence_placeholders,
    _remove_tbd_sentences,
    _repair_common_writer_citation_omissions,
    _repair_common_writer_placeholders,
)
from .pipeline_report import execution_gate_status
from .pipeline_writer_aux import (
    _preferred_writer_evidence_names,
    _render_writer_evidence_digest,
    _render_writer_evidence_digest_v2,
)
from .replication.notebook import (
    NotebookStep,
    build_notebook,
    build_requirements_lockfile,
    write_notebook,
)
from .reporting.reporting_checklist import (
    build_internal_phenotype_checklist,
    build_strobe_checklist,
    build_tripod_ai_checklist,
    choose_checklist,
)
from .reporting.reviewer import run_reviewer_round
from .schema import CritiqueReport, EvidenceRef, ManuscriptDraftPacket
from .side_findings import collect_side_findings
from .visual_qa import VLMVisualQAAdapter, VisualQAAuditor
from .reporting.pdf_render import render_pdf_for_run


class RuntimeProvenanceMismatchError(RuntimeError):
    """Docker steps disagree about the immutable execution environment."""


def _validated_runtime_lock(run_dir: Path) -> Optional[Path]:
    """Return one lock only when every Docker step has the same snapshot."""

    lock_paths = sorted(run_dir.glob("steps/*/outputs/runner_requirements.lock.txt"))
    provenance_paths = sorted(run_dir.glob("steps/*/outputs/runner_provenance.json"))
    if not lock_paths and not provenance_paths:
        return None
    lock_by_parent = {path.parent.resolve(): path for path in lock_paths}
    provenance_by_parent = {path.parent.resolve(): path for path in provenance_paths}
    if set(lock_by_parent) != set(provenance_by_parent):
        raise RuntimeProvenanceMismatchError(
            "Docker runtime provenance is incomplete: every step output must "
            "contain both runner_requirements.lock.txt and runner_provenance.json"
        )

    reference_lock: Optional[bytes] = None
    reference_provenance: Optional[str] = None
    for parent in sorted(lock_by_parent, key=str):
        lock_path = lock_by_parent[parent]
        provenance_path = provenance_by_parent[parent]
        lock_bytes = lock_path.read_bytes()
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeProvenanceMismatchError(
                f"Invalid Docker runtime provenance: {provenance_path}"
            ) from exc
        if not isinstance(provenance, dict):
            raise RuntimeProvenanceMismatchError(
                f"Docker runtime provenance must be an object: {provenance_path}"
            )
        expected_lock_sha = provenance.get("requirements_sha256")
        actual_lock_sha = hashlib.sha256(lock_bytes).hexdigest()
        if expected_lock_sha != actual_lock_sha:
            raise RuntimeProvenanceMismatchError(
                f"Docker runtime lock hash mismatch for {lock_path}"
            )
        canonical_provenance = json.dumps(
            provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        if reference_lock is None:
            reference_lock = lock_bytes
            reference_provenance = canonical_provenance
        elif (
            lock_bytes != reference_lock or canonical_provenance != reference_provenance
        ):
            raise RuntimeProvenanceMismatchError(
                "Docker steps used inconsistent image provenance or dependency locks"
            )
    return lock_paths[0]


def _assert_registered_runtime_lock_matches(evidence: Any, lockfile_path: Path) -> None:
    existing = evidence.get("requirements_lockfile")
    if existing is not None and existing.sha256 != sha256_of_file(lockfile_path):
        raise RuntimeProvenanceMismatchError(
            "Resume produced a requirements.lock.txt that differs from the "
            "already registered requirements_lockfile evidence"
        )


def _failed_step_labels_from_execution_gate(
    execution_gate: Dict[str, Any],
) -> Tuple[str, ...]:
    """Return human-readable failed/missing step labels for probe banners."""
    labels: List[str] = []
    for step_id in execution_gate.get("missing_steps") or []:
        labels.append(f"{step_id} (missing)")
    for item in execution_gate.get("failed_steps") or []:
        if isinstance(item, dict):
            step_id = item.get("step_id") or "unknown"
            status = item.get("status") or "failed"
            labels.append(f"{step_id} ({status})")
        else:
            labels.append(str(item))
    return tuple(labels)


def _writer_probe_banner(failed_steps: Sequence[str]) -> str:
    failed_text = ", ".join(failed_steps) if failed_steps else "unknown"
    return (
        "> ⚠️ DIAGNOSTIC PROBE ONLY — execution gate did not pass.\n"
        "> This draft was forced through for engineering triage. It is NOT\n"
        "> a reproducible analytic result, MUST NOT be cited, and SHOULD NOT\n"
        "> be used as a manuscript scaffold for publication.\n"
        f"> Failed steps: {failed_text}"
    )


_MANIFEST_COMMENT_RE = re.compile(
    r"<!--\s*(?P<level>warning|error)\s*:\s*see manifest\s*-->",
    flags=re.I,
)


def _manifest_comment_counts(text: str) -> Dict[str, int]:
    counts = {"warning": 0, "error": 0}
    for match in _MANIFEST_COMMENT_RE.finditer(text or ""):
        level = match.group("level").lower()
        counts[level] = counts.get(level, 0) + 1
    return counts


def _has_substantive_manuscript_text(text: str) -> bool:
    stripped = re.sub(r"<!--.*?-->", "", text or "", flags=re.S).strip()
    if not stripped:
        return False
    prose_lines = [
        line.strip()
        for line in stripped.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "[", "|", "<!--"))
    ]
    return any(re.search(r"[A-Za-z0-9]", line) for line in prose_lines)


def _blocked_manuscript_critique(
    reason: str,
    *,
    unsupported_claims: Sequence[str] = (),
    suggested_repairs: Sequence[str] = (),
) -> CritiqueReport:
    """Build an explicit non-passing review when no valid review is possible."""

    return CritiqueReport(
        status="blocked",
        reviewer="PipelineCritiqueFailSafe",
        concerns=[reason],
        unsupported_claims=list(unsupported_claims),
        suggested_repairs=list(suggested_repairs)
        or ["Resolve the blocking condition and rerun manuscript review."],
    )


def _review_manuscript_with_fail_safe(
    critic: CriticAgent,
    *,
    scaffold: str,
    available_evidence_ids: Sequence[str],
) -> Tuple[CritiqueReport, Optional[str]]:
    """Review a manuscript without ever converting a critic failure to pass."""

    try:
        return (
            critic.review_manuscript(
                scaffold=scaffold,
                available_evidence_ids=available_evidence_ids,
            ),
            None,
        )
    except Exception as exc:
        exception_type = type(exc).__name__
        return (
            _blocked_manuscript_critique(
                "CriticAgent could not complete manuscript review "
                f"({exception_type}); no passing review decision is available.",
                suggested_repairs=[
                    "Restore the manuscript critic and rerun review before treating "
                    "the manuscript as ready."
                ],
            ),
            exception_type,
        )


def _persist_manuscript_critique(
    *,
    critique: CritiqueReport,
    run_dir: Path,
    evidence: Any,
    producer: str,
) -> Path:
    """Persist the critique contract on success, failure, and skipped writes."""

    critique_path = run_dir / "manuscript_critique.json"
    critique_path.write_text(critique.model_dump_json(indent=2), encoding="utf-8")
    if evidence.get("manuscript_critique") is None:
        evidence.register_file(
            kind="log",
            description=(
                "Structured manuscript critique or explicit blocked fail-safe decision."
            ),
            source_path=critique_path,
            evidence_id="manuscript_critique",
            producer=producer,
            generation_mode="system",
        )
    return critique_path


def run_write_phase(
    pipeline,
    *,
    plan_result: _PlanPhaseResult,
    execute_result: _ExecutePhaseResult,
    run_dir: Path,
    run_id: str,
    stop_after_analysis: bool,
    manuscript_title: Optional[str],
    manuscript_authors: Optional[Sequence[str]],
    run_language: str,
    emit_progress: Callable[..., None],
    force_writer_probe: bool = False,
) -> _WritePhaseResult:
    """Draft manuscript-facing outputs after analysis is complete."""
    context = plan_result.context
    agent_context = plan_result.agent_context
    evidence = plan_result.evidence
    findings = plan_result.findings
    role_resolver = plan_result.role_resolver
    prompt_version = plan_result.prompt_version
    runtime_state = execute_result.runtime_state
    per_step_records = execute_result.per_step_records
    critic = CriticAgent(role_resolver("analyzer"))

    def blocked_write_result(bound_path: Path, reason: str) -> _WritePhaseResult:
        critique = _blocked_manuscript_critique(reason)
        _persist_manuscript_critique(
            critique=critique,
            run_dir=run_dir,
            evidence=evidence,
            producer="pipeline",
        )
        return _WritePhaseResult(
            literature=None,
            bound_path=bound_path,
            manuscript_critique=critique,
        )

    execution_gate = execution_gate_status(
        plan=execute_result.plan,
        per_step_records=per_step_records,
    )
    writer_probe_mode = (
        bool(force_writer_probe) and not execution_gate["execution_complete"]
    )
    writer_probe_failed_steps = _failed_step_labels_from_execution_gate(execution_gate)
    if not execution_gate["execution_complete"]:
        if writer_probe_mode:
            findings.append(
                ValidationFinding(
                    validator="manuscript_gate",
                    severity="warning",
                    message=(
                        "Diagnostic writer probe forced manuscript drafting even "
                        "though the execution gate did not pass. This output is "
                        "for engineering triage only and must not be cited."
                    ),
                    detail={
                        **execution_gate,
                        "writer_probe_mode": True,
                        "writer_probe_failed_steps": list(writer_probe_failed_steps),
                    },
                )
            )
        elif stop_after_analysis:
            findings.append(
                ValidationFinding(
                    validator="manuscript_gate",
                    severity="info",
                    message=(
                        "Manuscript generation skipped after a planned analysis "
                        "pause before the execution gate completed."
                    ),
                    detail={
                        **execution_gate,
                        "planned_pause": True,
                    },
                )
            )
            emit_progress(
                "pause",
                "Analysis paused before all planned steps completed; manuscript generation skipped.",
                status="paused",
                run_id=run_id,
            )
            bound_path = run_dir / "manuscript_scaffold_bound.md"
            bound_path.write_text(
                "# Manuscript scaffold not generated\n\n"
                "This run stopped after a requested analysis checkpoint before "
                "all planned analysis steps completed. Review the completed "
                "step outputs and resume from the next step when ready.\n",
                encoding="utf-8",
            )
            return blocked_write_result(
                bound_path,
                "Manuscript review was not run because execution paused before "
                "all planned analysis steps completed.",
            )
        else:
            findings.append(
                ValidationFinding(
                    validator="manuscript_gate",
                    severity="error",
                    message=(
                        "Formal manuscript generation skipped because the execution "
                        "gate did not pass. Review author_review_note.md and the "
                        "diagnostic artefacts before rerunning."
                    ),
                    detail=execution_gate,
                )
            )
            bound_path = run_dir / "manuscript_scaffold_bound.md"
            bound_path.write_text(
                "# Manuscript scaffold not generated\n\n"
                "Strict fail-closed policy blocked manuscript drafting because "
                "one or more required analysis steps did not complete successfully.\n\n"
                "Review `author_review_note.md`, `run_status.json`, "
                "`evidence_audit.json`, `numeric_audit.json`, and "
                "`claim_ledger.csv` for the diagnostic record.\n",
                encoding="utf-8",
            )
            return blocked_write_result(
                bound_path,
                "Manuscript review was not run because the analysis execution "
                "gate did not pass.",
            )

    if stop_after_analysis:
        emit_progress(
            "pause",
            "Analysis phase complete; manuscript generation skipped by user setting.",
            status="paused",
            run_id=run_id,
        )
        bound_path = run_dir / "manuscript_scaffold_bound.md"
        bound_path.write_text(
            "# Manuscript scaffold not generated\n\n"
            "This run stopped after the analysis phase. Review the "
            "`results_report.md`, tables, figures and manifest, then "
            "rerun with manuscript drafting enabled when the analysis "
            "is ready.\n",
            encoding="utf-8",
        )
        return blocked_write_result(
            bound_path,
            "Manuscript review was not run because manuscript drafting was "
            "explicitly paused after analysis.",
        )

    literature: Optional[LiteratureBundle] = None
    if pipeline._enable_publication_figure_skill:
        try:
            emit_progress(
                "figure",
                "Rendering manuscript-facing publication figure bundle from registered evidence.",
                run_id=run_id,
            )
            figure_result = PublicationFigureSkill().run(
                context=context,
                plan=execute_result.plan,
                evidence=evidence,
                run_dir=run_dir,
                prompt_pack_version=prompt_version,
            )
            findings.extend(figure_result.findings)
            if pipeline._enable_visual_qa and figure_result.figure_evidence_ids:
                fig_paths = []
                for evidence_id in figure_result.figure_evidence_ids:
                    record = evidence.get(evidence_id)
                    if record is not None:
                        fig_paths.append(run_dir / record.relative_path)
                if fig_paths:
                    vlm_adapter = pipeline._visual_qa_adapter
                    if vlm_adapter is None and pipeline._enable_vlm_visual_qa:
                        client = pipeline._vlm_client or role_resolver("analyzer")
                        if client is not None:
                            vlm_adapter = VLMVisualQAAdapter(client)
                    publication_visual_findings = VisualQAAuditor(
                        vlm_adapter=vlm_adapter
                    ).audit(figure_paths=fig_paths)
                    # See the final-pass demotion above: layout-style
                    # visual_qa errors raised on the publication
                    # bundle are cosmetic and must not block
                    # acceptance after the per-step repair budget
                    # has been exhausted upstream.
                    findings.extend(
                        (
                            finding.model_copy(update={"severity": "warning"})
                            if finding.severity == "error"
                            else finding
                        )
                        for finding in publication_visual_findings
                    )
        except Exception as exc:
            findings.append(
                ValidationFinding(
                    validator="publication_figure_skill",
                    severity="warning",
                    message=f"Publication figure skill failed; writer will use existing evidence only: {exc}",
                )
            )

    if pipeline._enable_literature:
        try:
            emit_progress(
                "literature",
                "Building literature bundle for manuscript drafting.",
                run_id=run_id,
            )
            lit_client = role_resolver("literature")
            if hasattr(lit_client, "_inner") and isinstance(
                getattr(lit_client, "_inner", None), MockLLMClient
            ):
                lit_client = None
            if isinstance(lit_client, MockLLMClient):
                lit_client = None
            pubmed_client = None
            if pipeline._enable_pubmed:
                from .literature import PubMedLiteratureClient

                pubmed_client = PubMedLiteratureClient(
                    email=pipeline._pubmed_email,
                    api_key=pipeline._pubmed_api_key,
                )
            tavily_client = None
            if pipeline._enable_tavily:
                from .literature import TavilyLiteratureClient

                tavily_client = TavilyLiteratureClient(
                    api_key=pipeline._tavily_api_key,
                    include_domains=pipeline._tavily_include_domains,
                    exclude_domains=pipeline._tavily_exclude_domains,
                )
            literature = LiteratureAgent(
                lit_client,
                enable_pubmed=pipeline._enable_pubmed,
                pubmed_client=pubmed_client,
                enable_tavily=pipeline._enable_tavily,
                tavily_client=tavily_client,
                tavily_retmax=pipeline._tavily_retmax,
            ).run(agent_context)
            lit_path = run_dir / "literature_bundle.json"
            lit_path.write_text(literature.model_dump_json(indent=2), encoding="utf-8")
            if evidence.get("literature_bundle") is None:
                evidence.register_file(
                    kind="log",
                    description="LiteratureBundle (citation registry for this run).",
                    source_path=lit_path,
                    evidence_id="literature_bundle",
                    producer="literature",
                    generation_mode=(
                        "llm" if lit_client is not None else "deterministic_skill"
                    ),
                    prompt_pack_version=prompt_version,
                    metadata={
                        "enable_pubmed": pipeline._enable_pubmed,
                        "enable_tavily": pipeline._enable_tavily,
                    },
                )
            # O21 — PRISMA 2020 counts. Registered as a separate
            # evidence id so the manuscript can cite
            # ``{evidence:literature_prisma}`` without pulling the
            # whole citation table into the binder.
            if literature.prisma is not None:
                prisma_path = run_dir / "literature_prisma.json"
                prisma_md_path = run_dir / "literature_prisma.md"
                prisma_path.write_text(
                    json.dumps(
                        {
                            "research_question": literature.research_question,
                            "prisma": literature.prisma,
                        },
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )
                p = literature.prisma
                prisma_md = (
                    "# PRISMA 2020 flow (O21)\n\n"
                    f"- Records identified: **{p.get('identified', 0)}**\n"
                    f"- Duplicates removed: **{p.get('duplicates_removed', 0)}**\n"
                    f"- Records screened: **{p.get('screened', 0)}**\n"
                    f"- Records eligible: **{p.get('eligible', 0)}**\n"
                    f"- Records included in review: **{p.get('included', 0)}**\n"
                )
                prisma_md_path.write_text(prisma_md, encoding="utf-8")
                if evidence.get("literature_prisma") is None:
                    evidence.register_file(
                        kind="statistic",
                        description=(
                            "PRISMA 2020 flow counts for the literature search (O21)."
                        ),
                        source_path=prisma_path,
                        evidence_id="literature_prisma",
                        producer="literature",
                        generation_mode="system",
                    )
                if evidence.get("literature_prisma_summary") is None:
                    evidence.register_file(
                        kind="log",
                        description="Human-readable PRISMA flow summary (O21).",
                        source_path=prisma_md_path,
                        evidence_id="literature_prisma_summary",
                        producer="literature",
                        generation_mode="system",
                    )
        except Exception as exc:
            findings.append(
                ValidationFinding(
                    validator="literature_agent",
                    severity="warning",
                    message=f"Literature agent failed: {exc}",
                )
            )

    # The evidence store is append-only across resume attempts. Freeze one
    # digest-verified view after the optional literature inputs are registered
    # and use it for every analysis-facing writer consumer; otherwise an old
    # failed figure, code file, or statistic can leak back through a reader
    # that walks ``evidence.records``.
    current_verified_evidence_records = evidence.current_verified_records(
        per_step_records
    )
    current_evidence_names = evidence.current_resolvable_names(per_step_records)
    preferred_writer_evidence_names = _preferred_writer_evidence_names(
        evidence,
        per_step_records,
    )
    emit_progress(
        "writer",
        "Drafting manuscript scaffold.",
        run_id=run_id,
    )
    writer = ManuscriptAgent(role_resolver("writer"), language=run_language)
    manuscript_packet: Optional[ManuscriptDraftPacket] = None
    if runtime_state.semantics is not None:
        manuscript_packet = writer.build_packet(
            context=context,
            semantics=runtime_state.semantics,
            evidence_refs=[
                EvidenceRef(
                    evidence_id=record.evidence_id,
                    kind=record.kind,
                    description=record.description,
                    relative_path=record.relative_path,
                )
                for record in current_verified_evidence_records
            ],
            findings=[
                f.message for f in findings if f.severity in {"warning", "error"}
            ],
            caveats=list(runtime_state.semantics.safety_guardrails),
        )
        packet_path = run_dir / "manuscript_packet.json"
        packet_path.write_text(
            manuscript_packet.model_dump_json(indent=2),
            encoding="utf-8",
        )
        if evidence.get("manuscript_packet") is None:
            evidence.register_file(
                kind="log",
                description="Typed manuscript draft packet passed into the manuscript agent.",
                source_path=packet_path,
                evidence_id="manuscript_packet",
                producer="manuscript_agent",
                generation_mode="system",
                prompt_pack_version=prompt_version,
            )
    writer_error_message: Optional[str] = None
    try:
        if pipeline._writer_digest_widened:
            writer_evidence_digest = _render_writer_evidence_digest_v2(
                per_step_records,
                context=context,
                run_dir=run_dir,
                evidence=evidence,
                secondary_cap_per_step=pipeline._writer_digest_secondary_cap_per_step,
            )
        else:
            writer_evidence_digest = _render_writer_evidence_digest(
                context=context,
                run_dir=run_dir,
                per_step_records=per_step_records,
            )
        writer_digest_path = run_dir / "writer_evidence_digest.md"
        writer_digest_path.write_text(writer_evidence_digest, encoding="utf-8")
        if evidence.get("writer_evidence_digest") is None:
            evidence.register_file(
                kind="log",
                description=(
                    "Writer evidence digest "
                    f"({'v2 widened' if pipeline._writer_digest_widened else 'v1 primary-only'})."
                ),
                source_path=writer_digest_path,
                evidence_id="writer_evidence_digest",
                producer="pipeline",
                generation_mode="system",
                metadata={
                    "writer_digest_widened": bool(pipeline._writer_digest_widened),
                    "writer_digest_secondary_cap_per_step": int(
                        pipeline._writer_digest_secondary_cap_per_step
                    ),
                },
            )
        scaffold = writer.run(
            context=agent_context,
            evidence_ids=preferred_writer_evidence_names,
            evidence_digest=writer_evidence_digest,
        )
    except Exception as exc:
        writer_error_message = f"{type(exc).__name__}: {exc}"
        scaffold = ""
        findings.append(
            ValidationFinding(
                validator="writer_agent",
                severity="error",
                message=(
                    "WriterAgent failed before producing a manuscript scaffold: "
                    f"{writer_error_message}"
                ),
                detail={
                    "exception_type": type(exc).__name__,
                    "writer_digest_widened": bool(pipeline._writer_digest_widened),
                },
            )
        )
    scaffold, placeholder_repairs = _repair_common_writer_placeholders(
        scaffold,
        context=context,
        evidence=evidence,
        allowed_evidence_names=current_evidence_names,
    )
    if placeholder_repairs:
        findings.append(
            ValidationFinding(
                validator="evidence_bound_writer",
                severity="warning",
                message=(
                    "Repaired common manuscript evidence placeholder(s): "
                    + ", ".join(f"{old}->{new}" for old, new in placeholder_repairs)
                ),
                detail={
                    "repairs": [
                        {"from": old, "to": new} for old, new in placeholder_repairs
                    ]
                },
            )
        )
    scaffold, citation_repairs = _repair_common_writer_citation_omissions(
        scaffold,
        evidence=evidence,
        allowed_evidence_names=current_evidence_names,
    )
    if citation_repairs:
        findings.append(
            ValidationFinding(
                validator="evidence_bound_writer",
                severity="warning",
                message=(
                    "Repaired common uncited manuscript methods sentence(s): "
                    f"{len(citation_repairs)} citation(s) appended."
                ),
                detail={"citation_repairs": citation_repairs},
            )
        )
    scaffold_path = run_dir / "manuscript_scaffold.md"
    scaffold_path.write_text(scaffold, encoding="utf-8")
    if evidence.get("manuscript_scaffold_raw") is None:
        evidence.register_file(
            kind="log",
            description="Manuscript scaffold (raw, with {evidence:*} placeholders).",
            source_path=scaffold_path,
            evidence_id="manuscript_scaffold_raw",
            producer="writer",
            generation_mode="llm",
            prompt_pack_version=prompt_version,
        )

    evidence_bound_scaffold, removed_sentences = (
        evidence.enforce_evidence_bound_scaffold(scaffold)
    )
    if removed_sentences:
        findings.append(
            ValidationFinding(
                validator="evidence_bound_writer",
                severity="warning",
                message=(
                    f"Filtered {len(removed_sentences)} result-like sentence(s) without evidence placeholders before manuscript binding."
                ),
                detail={"removed_sentences": removed_sentences},
            )
        )
        filtered_path = run_dir / "manuscript_scaffold_filtered.md"
        filtered_path.write_text(evidence_bound_scaffold, encoding="utf-8")
        if evidence.get("manuscript_scaffold_filtered") is None:
            evidence.register_file(
                kind="log",
                description="Manuscript scaffold after evidence-bound filtering.",
                source_path=filtered_path,
                evidence_id="manuscript_scaffold_filtered",
                producer="pipeline",
                generation_mode="system",
            )

    bound_unfiltered = evidence.bind_manuscript(
        evidence_bound_scaffold,
        per_step_records=per_step_records,
    )
    bound, demoted_missing_ids = _demote_unresolved_evidence_placeholders(
        bound_unfiltered
    )
    bound, removed_tbd_sentences = _remove_tbd_sentences(bound)
    if (
        removed_tbd_sentences
        and pipeline._evidence_enforcement_mode is EvidenceEnforcementMode.STRICT
    ):
        raise EvidenceEnforcementError(
            f"STRICT evidence mode: writer emitted {len(removed_tbd_sentences)} "
            f"sentence(s) containing [TBD]/[TODO]/[TK] placeholder(s). "
            f"The bound manuscript must not carry unresolved writer "
            f"placeholders before submission.",
            detail={"tbd_sentences": removed_tbd_sentences},
        )
    side_findings = collect_side_findings(per_step_records)
    bound, language_guard_detail = enforce_writer_claim_language(
        bound,
        enforcement_mode=pipeline._evidence_enforcement_mode,
        side_findings=side_findings,
    )
    if language_guard_detail:
        findings.append(
            ValidationFinding(
                validator="manuscript_language_guard",
                severity="warning",
                message=(
                    "Annotated post-hoc claim language or side-finding leakage "
                    "in SOFT evidence mode."
                ),
                detail=language_guard_detail,
            )
        )
    manifest_comment_counts = _manifest_comment_counts(bound)
    manifest_comment_total = sum(manifest_comment_counts.values())
    if manifest_comment_total:
        findings.append(
            ValidationFinding(
                validator="evidence_bound_writer",
                severity="error",
                message=(
                    "Bound manuscript cites evidence records with unresolved "
                    f"manifest caveats: {manifest_comment_counts['error']} error "
                    f"and {manifest_comment_counts['warning']} warning comment(s)."
                ),
                detail={"manifest_comment_counts": manifest_comment_counts},
            )
        )
    manuscript_output_blockers: List[str] = []
    if writer_error_message:
        manuscript_output_blockers.append(
            "WriterAgent failed before producing a manuscript scaffold."
        )
    if not _has_substantive_manuscript_text(bound):
        manuscript_output_blockers.append(
            "Bound manuscript has no substantive evidence-bound prose after filtering."
        )
    if manuscript_output_blockers:
        findings.append(
            ValidationFinding(
                validator="evidence_bound_writer",
                severity="error",
                message=(
                    "Bound manuscript is empty or non-substantive after writer "
                    "execution and evidence filtering."
                ),
                detail={"blockers": manuscript_output_blockers},
            )
        )
        bound = (
            "# Manuscript scaffold not generated\n\n"
            "The manuscript writer failed or produced no substantive "
            "evidence-bound prose for this run. See manifest findings for the "
            "writer and evidence-binding errors.\n"
        )
    # Value-level provenance binding: attach a footnote next to every
    # numeric value in the manuscript pointing to the exact step /
    # field / evidence id that produced it. STRICT mode raises when a
    # number cannot be traced; SOFT mode marks it inline so reviewers
    # see the gap without breaking the bound output.
    bound, numeric_binding_map, untraced_numerics = bind_numeric_values(
        bound,
        evidence=evidence,
        enforcement_mode=pipeline._evidence_enforcement_mode,
        per_step_records=per_step_records,
    )
    bound_evidence_id = (
        "manuscript_scaffold_writer_probe"
        if writer_probe_mode
        else "manuscript_scaffold_bound"
    )
    bound_path = run_dir / (
        "manuscript_scaffold_writer_probe.md"
        if writer_probe_mode
        else "manuscript_scaffold_bound.md"
    )
    if writer_probe_mode:
        bound = _writer_probe_banner(writer_probe_failed_steps) + "\n\n" + bound
    bound_path.write_text(bound, encoding="utf-8")
    if evidence.get(bound_evidence_id) is None:
        evidence.register_file(
            kind="log",
            description=(
                "Diagnostic writer-probe manuscript scaffold forced past "
                "a failed execution gate."
                if writer_probe_mode
                else "Manuscript scaffold with evidence ids resolved to file links + sha256."
            ),
            source_path=bound_path,
            evidence_id=bound_evidence_id,
            producer="pipeline",
            generation_mode="system",
            metadata=(
                {
                    "writer_probe_mode": bool(writer_probe_mode),
                    "writer_probe_failed_steps": list(writer_probe_failed_steps),
                }
                if writer_probe_mode
                else None
            ),
        )
    if demoted_missing_ids:
        unfiltered_path = run_dir / "manuscript_scaffold_bound_unfiltered.md"
        unfiltered_path.write_text(bound_unfiltered, encoding="utf-8")
        if evidence.get("manuscript_scaffold_bound_unfiltered") is None:
            evidence.register_file(
                kind="log",
                description=(
                    "Manuscript scaffold prior to demoting unresolved "
                    "[evidence missing: …] placeholders to HTML comments."
                ),
                source_path=unfiltered_path,
                evidence_id="manuscript_scaffold_bound_unfiltered",
                producer="pipeline",
                generation_mode="system",
            )
        findings.append(
            ValidationFinding(
                validator="evidence_bound_writer",
                severity="warning",
                message=(
                    f"Demoted {len(demoted_missing_ids)} unresolved "
                    f"[evidence missing: …] placeholder(s) to HTML "
                    f"comments so the manuscript still renders cleanly; "
                    f"see manuscript_scaffold_bound_unfiltered.md for "
                    f"the original."
                ),
                detail={"missing_evidence_ids": sorted(set(demoted_missing_ids))},
            )
        )
    if removed_tbd_sentences:
        findings.append(
            ValidationFinding(
                validator="evidence_bound_writer",
                severity="warning",
                message=(
                    f"Removed {len(removed_tbd_sentences)} sentence(s) containing "
                    "[TBD] from the bound manuscript; the writer must omit "
                    "unsupported values instead of leaving placeholders."
                ),
                detail={"removed_sentences": removed_tbd_sentences},
            )
        )
    manuscript_numeric_findings = audit_manuscript_numeric_claims(
        bound,
        per_step_records=per_step_records,
    )
    findings.extend(manuscript_numeric_findings)

    manuscript_critique, critic_review_error = _review_manuscript_with_fail_safe(
        critic,
        scaffold=bound,
        available_evidence_ids=current_evidence_names,
    )
    if manuscript_output_blockers:
        manuscript_critique = manuscript_critique.model_copy(
            update={
                "status": "blocked",
                "unsupported_claims": list(manuscript_critique.unsupported_claims)
                + manuscript_output_blockers,
                "concerns": list(manuscript_critique.concerns)
                + [
                    "The writer did not produce a usable evidence-bound manuscript; "
                    "the run must remain analysis-only until writer output is regenerated."
                ],
            }
        )
    if manifest_comment_total:
        manuscript_critique = manuscript_critique.model_copy(
            update={
                "status": "blocked",
                "unsupported_claims": list(manuscript_critique.unsupported_claims)
                + [
                    "Bound manuscript contains unresolved manifest caveat comments "
                    f"({manifest_comment_counts['error']} error, "
                    f"{manifest_comment_counts['warning']} warning)."
                ],
                "concerns": list(manuscript_critique.concerns)
                + [
                    "Manuscript cites records that are not manuscript-facing clean "
                    "because their evidence records carry active warning/error caveats."
                ],
            }
        )
    if manuscript_numeric_findings:
        manuscript_critique = manuscript_critique.model_copy(
            update={
                "status": "blocked",
                "unsupported_claims": list(manuscript_critique.unsupported_claims)
                + [finding.message for finding in manuscript_numeric_findings],
                "concerns": list(manuscript_critique.concerns)
                + [
                    "Manuscript numeric claims disagree with registered step_summary values."
                ],
            }
        )
    _persist_manuscript_critique(
        critique=manuscript_critique,
        run_dir=run_dir,
        evidence=evidence,
        producer="pipeline" if critic_review_error else "critic",
    )
    if critic_review_error is not None:
        findings.append(
            ValidationFinding(
                validator="critic_agent",
                severity="error",
                message=(
                    "CriticAgent failed during manuscript review; a blocked "
                    "fail-safe critique was persisted instead of a passing result."
                ),
                evidence_ids=["manuscript_critique"],
                detail={"exception_type": critic_review_error},
            )
        )
    elif manuscript_critique.status in {"needs_revision", "blocked"}:
        findings.append(
            ValidationFinding(
                validator="critic_agent",
                severity="error",
                message=(
                    f"CriticAgent marked manuscript as {manuscript_critique.status}: "
                    + "; ".join(
                        manuscript_critique.concerns
                        or manuscript_critique.suggested_repairs
                        or ["review required"]
                    )
                ),
                evidence_ids=["manuscript_critique"],
            )
        )

    if pipeline._enable_latex:
        try:
            emit_progress(
                "latex",
                "Rendering LaTeX and BibTeX scaffold.",
                run_id=run_id,
            )
            bib_basename = "manuscript_scaffold"
            # Collect registered figure paths for auto-embedding.
            fig_paths_for_latex: List[Tuple[str, str]] = []
            for rec in current_verified_evidence_records:
                if rec.kind != "figure":
                    continue
                # Prefer PNG for LaTeX compatibility; SVG needs
                # inkscape or svg package.
                if rec.relative_path.endswith((".png", ".pdf", ".tiff")):
                    fig_paths_for_latex.append(
                        (rec.evidence_id, "evidence/" + rec.relative_path)
                    )
            tex = scaffold_to_latex(
                markdown=bound,
                title=manuscript_title
                or f"EasyICU research-agent: {context.research_question}",
                authors=manuscript_authors or ["EasyICU research-agent"],
                bibliography=literature,
                bibliography_basename=bib_basename,
                venue_template=pipeline._latex_venue_template,
                figure_paths=fig_paths_for_latex or None,
            )
            tex_path = run_dir / "manuscript_scaffold.tex"
            tex_path.write_text(tex, encoding="utf-8")
            if evidence.get("manuscript_scaffold_tex") is None:
                evidence.register_file(
                    kind="log",
                    description="LaTeX manuscript scaffold generated from the bound markdown.",
                    source_path=tex_path,
                    evidence_id="manuscript_scaffold_tex",
                    producer="pipeline",
                    generation_mode="system",
                )
            if literature is not None and getattr(literature, "citations", None):
                bib = render_bibtex(literature)
                bib_path = run_dir / f"{bib_basename}.bib"
                bib_path.write_text(bib, encoding="utf-8")
                if evidence.get("manuscript_bibliography") is None:
                    evidence.register_file(
                        kind="log",
                        description="BibTeX file rendered from the literature bundle.",
                        source_path=bib_path,
                        evidence_id="manuscript_bibliography",
                        producer="pipeline",
                        generation_mode="system",
                    )
            # Optional: compile the .tex to PDF on disk so users
            # can open ``manuscript_scaffold.pdf`` directly. Off
            # by default because not every environment has a
            # LaTeX install.
            if pipeline._enable_pdf_render:
                bib_full = (
                    run_dir / f"{bib_basename}.bib"
                    if (run_dir / f"{bib_basename}.bib").exists()
                    else None
                )
                pdf_result = render_pdf_for_run(
                    tex_path=tex_path,
                    bib_path=bib_full,
                    output_dir=run_dir,
                )
                if pdf_result.success and pdf_result.pdf_path is not None:
                    if evidence.get("manuscript_scaffold_pdf") is None:
                        evidence.register_file(
                            kind="log",
                            description=(
                                f"Compiled manuscript PDF "
                                f"(engine={pdf_result.engine})."
                            ),
                            source_path=pdf_result.pdf_path,
                            evidence_id="manuscript_scaffold_pdf",
                            producer="pipeline",
                            generation_mode="system",
                        )
                    findings.append(
                        ValidationFinding(
                            validator="pdf_render",
                            severity="info",
                            message=(
                                f"Rendered manuscript PDF via " f"{pdf_result.engine}."
                            ),
                            evidence_ids=["manuscript_scaffold_pdf"],
                        )
                    )
                else:
                    findings.append(
                        ValidationFinding(
                            validator="pdf_render",
                            severity="warning",
                            message=(
                                "PDF render failed or no LaTeX "
                                "engine found: " + "; ".join(pdf_result.notes)
                            ),
                        )
                    )
        except Exception as exc:
            findings.append(
                ValidationFinding(
                    validator="latex_export",
                    severity="warning",
                    message=f"LaTeX export failed: {exc}",
                )
            )

    # O18 — Causal audit. Run last in the write phase so the
    # bound manuscript (post binding, post filtering) is what gets
    # scanned. Associational-effect-with-causal-language is a
    # warning; causal_overclaimed-with-causal-language is an
    # error.
    if pipeline._enable_causal_audit:
        try:
            bound_text = bound_path.read_text(encoding="utf-8")
        except Exception:
            bound_text = ""
        causal_report = run_causal_audit(
            evidence_records=current_verified_evidence_records,
            run_dir=run_dir,
            bound_manuscript=bound_text,
        )
        causal_json = run_dir / "causal_audit_report.json"
        causal_md = run_dir / "causal_audit_report.md"
        causal_report.write_json(causal_json)
        causal_report.write_markdown(causal_md)
        if evidence.get("causal_audit_report") is None:
            evidence.register_file(
                kind="statistic",
                description=(
                    "Causal-claim audit (O18): effect labels "
                    "(associational / causal_explicit / "
                    "causal_overclaimed) and causal-language hits."
                ),
                source_path=causal_json,
                evidence_id="causal_audit_report",
                producer="pipeline",
                generation_mode="system",
            )
        if evidence.get("causal_audit_summary") is None:
            evidence.register_file(
                kind="log",
                description="Human-readable causal-audit summary (O18).",
                source_path=causal_md,
                evidence_id="causal_audit_summary",
                producer="pipeline",
                generation_mode="system",
            )
        summary = causal_report.summary()
        if summary["n_effects_labelled"] > 0 or summary["n_language_errors"] > 0:
            findings.append(
                ValidationFinding(
                    validator="causal_audit",
                    severity="info",
                    message=(
                        f"Labelled {summary['n_effects_labelled']} effect(s); "
                        f"{summary['n_associational']} associational, "
                        f"{summary['n_causal_explicit']} causal_explicit, "
                        f"{summary['n_causal_overclaimed']} causal_overclaimed."
                    ),
                    evidence_ids=["causal_audit_report"],
                    detail=summary,
                )
            )
        for hit in causal_report.language_hits:
            findings.append(
                ValidationFinding(
                    validator="causal_audit",
                    severity=hit.severity,
                    message=(
                        f"Causal language over {hit.strength} pattern "
                        f"`{hit.pattern}` cited "
                        f"{hit.linked_effect_labels or 'no labelled effect'}."
                    ),
                    evidence_ids=list(hit.linked_evidence_ids)
                    + ["causal_audit_report"],
                    detail={"sentence": hit.sentence[:280]},
                )
            )

    # Methodological-rigor gate: does the analysis METHOD match the locked
    # study-design family? Catches a survival question answered with a static
    # odds ratio, discrimination without calibration, clusters without
    # stability, and a complete-case estimate under material missingness. The
    # causal family is deliberately delegated to the causal audit above; this
    # covers the families it does not. Advisory and best-effort: findings feed
    # the readiness gate but a rigor-audit exception never crashes the write
    # phase.
    try:
        from .methodological_rigor import (
            MethodologicalRigorAuditor,
            extract_method_signals,
        )

        rigor_findings = MethodologicalRigorAuditor().audit(
            context=context,
            evidence=evidence,
            evidence_records=current_verified_evidence_records,
        )
        signals = extract_method_signals(
            context,
            evidence,
            evidence_records=current_verified_evidence_records,
        )
        if evidence.get("methodological_rigor_report") is None:
            evidence.register_json(
                kind="statistic",
                description=(
                    "Methodological-rigor audit: does the analysis method match "
                    "the locked study-design family?"
                ),
                payload={
                    "family": signals.family,
                    "signals": dict(signals.__dict__),
                    "findings": [f.model_dump(mode="json") for f in rigor_findings],
                },
                filename="methodological_rigor_report.json",
                evidence_id="methodological_rigor_report",
                producer="pipeline",
                generation_mode="system",
            )
        findings += rigor_findings
    except Exception as exc:  # pragma: no cover - defensive
        findings.append(
            ValidationFinding(
                validator="methodological_rigor",
                severity="info",
                message=f"Methodological-rigor audit skipped: {exc}",
            )
        )

    # O16 — Reporting-guideline checklist. Writes STROBE (always)
    # and TRIPOD+AI (when the analysis family looks like a
    # prediction / validation study). Findings are emitted at
    # ``info`` severity by default so the paper can still be
    # produced; reviewers see the coverage number and decide.
    if pipeline._enable_reporting_checklist:
        try:
            bound_text = bound_path.read_text(encoding="utf-8")
        except Exception:
            bound_text = ""
        if pipeline._reporting_checklist_names is not None:
            wanted = tuple(n.lower() for n in pipeline._reporting_checklist_names)
        else:
            analysis_family = (
                (context.user_preferences.inferred_analysis_family or "")
                if getattr(context, "user_preferences", None)
                else ""
            )
            wanted = choose_checklist(analysis_family)
        checklist_reports = []
        if "strobe" in wanted:
            checklist_reports.append(
                (
                    "strobe",
                    build_strobe_checklist(
                        evidence_records=current_verified_evidence_records,
                        bound_manuscript=bound_text,
                        task_kind=getattr(pipeline, "_benchmark_task_kind", None),
                    ),
                )
            )
        if "tripod_ai" in wanted or "tripod+ai" in wanted:
            checklist_reports.append(
                (
                    "tripod_ai",
                    build_tripod_ai_checklist(
                        evidence_records=current_verified_evidence_records,
                        bound_manuscript=bound_text,
                    ),
                )
            )
        if "internal_phenotype" in wanted:
            checklist_reports.append(
                (
                    "internal_phenotype",
                    build_internal_phenotype_checklist(
                        evidence_records=current_verified_evidence_records,
                        bound_manuscript=bound_text,
                        task_kind=getattr(pipeline, "_benchmark_task_kind", None),
                    ),
                )
            )
        for key, report in checklist_reports:
            md_path = run_dir / f"reporting_checklist_{key}.md"
            json_path = run_dir / f"reporting_checklist_{key}.json"
            md_path.write_text(report.to_markdown(), encoding="utf-8")
            json_path.write_text(
                json.dumps(report.to_json(), indent=2, default=str),
                encoding="utf-8",
            )
            md_evid_id = f"reporting_checklist_{key}"
            json_evid_id = f"reporting_checklist_{key}_json"
            if evidence.get(md_evid_id) is None:
                evidence.register_file(
                    kind="log",
                    description=(
                        f"Auto-filled {report.name} reporting checklist (O16)."
                    ),
                    source_path=md_path,
                    evidence_id=md_evid_id,
                    producer="pipeline",
                    generation_mode="system",
                )
            if evidence.get(json_evid_id) is None:
                evidence.register_file(
                    kind="log",
                    description=(
                        f"Structured {report.name} reporting checklist (O16)."
                    ),
                    source_path=json_path,
                    evidence_id=json_evid_id,
                    producer="pipeline",
                    generation_mode="system",
                )
            summary = report.summary()
            findings.append(
                ValidationFinding(
                    validator="reporting_checklist",
                    severity="info",
                    message=(
                        f"{report.name} coverage {summary['coverage']:.0%} "
                        f"({summary['n_addressed']} addressed, "
                        f"{summary['n_partial']} partial, "
                        f"{summary['n_open']} open, "
                        f"{summary['n_not_applicable']} n/a)."
                    ),
                    evidence_ids=[md_evid_id],
                    detail=summary,
                )
            )
            # Promote to warning only if coverage < 50 %; reviewers
            # care about Methods completeness, not every cell.
            if summary["coverage"] < 0.5:
                findings.append(
                    ValidationFinding(
                        validator="reporting_checklist",
                        severity="warning",
                        message=(
                            f"{report.name} reporting coverage below 50 %; "
                            "expect reviewer pushback on Methods completeness."
                        ),
                        evidence_ids=[md_evid_id],
                        detail=summary,
                    )
                )

    # O15 — Simulated three-role reviewer round. Runs after the
    # deterministic gates so each reviewer reads the latest
    # findings (multiple-testing, causal-audit, checklist). The
    # output is not a validator; it is a reviewer-facing note
    # bundle that the manuscript author / responsible clinician
    # uses to tighten the draft before submission.
    if pipeline._enable_reviewer_round:
        reviewer_report = run_reviewer_round(
            evidence_records=current_verified_evidence_records,
            findings=findings,
            round_index=0,
        )
        reviewer_md = run_dir / "reviewer_report.md"
        reviewer_json = run_dir / "reviewer_report.json"
        reviewer_md.write_text(reviewer_report.to_markdown(), encoding="utf-8")
        reviewer_json.write_text(
            json.dumps(reviewer_report.to_json(), indent=2, default=str),
            encoding="utf-8",
        )
        if evidence.get("reviewer_report") is None:
            evidence.register_file(
                kind="log",
                description=(
                    "Three-role simulated reviewer report (O15): "
                    "statistician / clinician / methodologist."
                ),
                source_path=reviewer_md,
                evidence_id="reviewer_report",
                producer="pipeline",
                generation_mode="system",
            )
        if evidence.get("reviewer_report_json") is None:
            evidence.register_file(
                kind="log",
                description="Structured reviewer report (O15).",
                source_path=reviewer_json,
                evidence_id="reviewer_report_json",
                producer="pipeline",
                generation_mode="system",
            )
        summary = reviewer_report.summary()
        rec = summary["aggregated_recommendation"]
        severity = {
            "accept": "info",
            "minor_revision": "info",
            "major_revision": "warning",
            "reject": "error",
        }.get(rec, "info")
        findings.append(
            ValidationFinding(
                validator="reviewer_round",
                severity=severity,
                message=(
                    f"Simulated reviewers returned `{rec}` "
                    f"(info={summary['counts'].get('info',0)}, "
                    f"minor={summary['counts'].get('minor',0)}, "
                    f"major={summary['counts'].get('major',0)}, "
                    f"reject={summary['counts'].get('reject',0)})."
                ),
                evidence_ids=["reviewer_report"],
                detail=summary,
            )
        )

    # O26 — Notebook + lockfile. Concatenates every per-step
    # generated script in plan order into a single runnable
    # ``run.ipynb`` and captures the interpreter's installed
    # packages in ``requirements.lock.txt``. Runs regardless of
    # reviewer / checklist flags so the reproducibility artefacts
    # are always present.
    captured_runtime_lock = _validated_runtime_lock(run_dir)
    try:
        notebook_steps: List[NotebookStep] = []
        intent_by_id = {s.step_id: s.intent for s in plan_result.plan.steps}
        # Preserve plan order: iterate plan, pick first 'code'
        # evidence per step.
        code_records_by_step: Dict[str, Any] = {}
        for rec in current_verified_evidence_records:
            if rec.kind != "code":
                continue
            step_id = rec.produced_by_step or ""
            if step_id and step_id not in code_records_by_step:
                code_records_by_step[step_id] = rec
        for step in plan_result.plan.steps:
            rec = code_records_by_step.get(step.step_id)
            if rec is None:
                continue
            candidates = [
                run_dir / "evidence" / rec.relative_path,
                run_dir / rec.relative_path,
            ]
            path = next((p for p in candidates if p.exists()), None)
            if path is None:
                continue
            try:
                code_text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            notebook_steps.append(
                NotebookStep(
                    step_id=step.step_id,
                    intent=intent_by_id.get(step.step_id, step.intent),
                    code=code_text,
                )
            )
        if notebook_steps:
            notebook = build_notebook(
                research_question=plan_result.context.research_question,
                cohort_relative_path="cohort.parquet",
                steps=notebook_steps,
            )
            nb_path = run_dir / "run.ipynb"
            write_notebook(nb_path, notebook)
            if evidence.get("run_notebook") is None:
                evidence.register_file(
                    kind="code",
                    description=(
                        "Auto-generated Jupyter notebook re-running "
                        "every plan step top-to-bottom (O26)."
                    ),
                    source_path=nb_path,
                    evidence_id="run_notebook",
                    producer="pipeline",
                    generation_mode="system",
                )
        lockfile_path = run_dir / "requirements.lock.txt"
        lockfile_path.write_text(
            build_requirements_lockfile(captured_runtime_lock), encoding="utf-8"
        )
        _assert_registered_runtime_lock_matches(evidence, lockfile_path)
        existing_lock_record = evidence.get("requirements_lockfile")
        if existing_lock_record is None:
            evidence.register_file(
                kind="log",
                description=(
                    "Execution-runtime requirements lockfile captured "
                    "at run time (O26)."
                ),
                source_path=lockfile_path,
                evidence_id="requirements_lockfile",
                producer="pipeline",
                generation_mode="system",
                metadata={
                    "runtime_source": (
                        "docker_runner"
                        if captured_runtime_lock is not None
                        else "host_interpreter"
                    )
                },
            )
    except RuntimeProvenanceMismatchError:
        raise
    except Exception as exc:
        findings.append(
            ValidationFinding(
                validator="repro_artifacts",
                severity="warning",
                message=(
                    f"Failed to build run.ipynb / lockfile: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        )

    return _WritePhaseResult(
        literature=literature,
        bound_path=bound_path,
        manuscript_packet=manuscript_packet,
        manuscript_critique=manuscript_critique,
        writer_probe_mode=writer_probe_mode,
        writer_probe_failed_steps=tuple(writer_probe_failed_steps),
    )
