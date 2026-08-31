"""[Layer 4: Evidence & Provenance / Layer 2: LLM Orchestration]
Write phase for the EasyICU research-agent pipeline.

This module is callable as ``run_write_phase(pipeline, ...)``. It reads
configuration and collaborators from the pipeline instance, matching the
``execution/phase.py`` free-function pattern, and returns the existing
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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..agents.core import CriticAgent, ManuscriptAgent
from ..audits.manuscript_claims import audit_manuscript_numeric_claims
from ..audits.envelope_consumers import RegisteredOutputEnvelopeConsumer
from .bibtex import render_bibtex
from ..review.causal_audit import run_causal_audit
from ..contracts.runtime import (
    ValidationFinding,
    _ExecutePhaseResult,
    _PlanPhaseResult,
    _WritePhaseResult,
)
from ..authority.evidence_store import (
    EvidenceEnforcementError,
    EvidenceEnforcementMode,
    evidence_artifact_basename_stem,
    sha256_of_file,
)
from ..authority.manuscript_claim_policy import (
    missing_scientific_claims_in_results,
    place_scientific_claim_tokens_in_results,
)
from ..authority.runtime_artifacts import (
    current_step_records,
    verified_run_evidence_path,
)
from ..figures.skill import PublicationFigureSkill
from ..publication_skills import compile_publication_skill_activation
from .latex import scaffold_to_latex
from .manuscript_literature import (
    audit_manuscript_literature,
    remove_sentences_with_unknown_literature_keys,
    repair_evidence_ids_mistyped_as_literature,
    repair_missing_context_section_citations,
    repair_missing_methods_method_citation,
    render_writer_literature_digest,
)
from .manuscript_quality import (
    ManuscriptQualityFinding,
    audit_manuscript_quality,
    expected_manuscript_display_labels,
    render_reader_manuscript,
)
from .administrative_authority import load_manuscript_administrative_authority
from .manuscript_provenance import (
    ManuscriptProvenanceError,
    build_manuscript_provenance,
)
from .manuscript_projection import project_owner_issued_manuscript_claims
from .novelty_positioning import build_unsigned_novelty_positioning_packet
from ..literature import LiteratureAgent, LiteratureBundle, manuscript_citable_keys
from ..orchestration.profiles import is_paper_facing_profile
from ..providers.mocks import MockLLMClient
from ..providers.prompt_budget import budgeted_vlm_client
from ..providers.structured_retry import StructuredResponseFailure
from .manuscript_post import (
    _apply_writer_evidence_repair_decisions,
    _writer_repair_target_span,
    bind_numeric_values,
    drop_untraceable_numeric_sentences,
    enforce_writer_claim_language,
    _demote_unresolved_evidence_placeholders,
    _remove_tbd_sentences,
    _remove_unregistered_evidence_placeholders,
    _repair_common_writer_citation_omissions,
    _repair_common_writer_placeholders,
    repair_miscited_numeric_citations,
    repair_single_variant_robustness_metric_prose,
)
from .readiness import _is_cosmetic_visual_error, execution_gate_status
from .writer_evidence import (
    _preferred_writer_evidence_names,
    _render_writer_evidence_digest,
    _render_writer_evidence_digest_v2,
)
from .writer_evidence_repair import decide_writer_evidence_repairs
from ..replication.notebook import (
    NotebookStep,
    build_notebook,
    build_requirements_lockfile,
    write_notebook,
)
from .reporting_checklist import (
    build_internal_phenotype_checklist,
    build_strobe_checklist,
    build_tripod_ai_checklist,
    choose_checklist,
)
from .reviewer import run_reviewer_round
from ..schema import CritiqueReport, EvidenceRef, ManuscriptDraftPacket
from .side_findings import collect_side_findings
from ..robustness.panel import load_robustness_panel
from ..gates.figure_egress import (
    FigureEgressReceiptError,
    register_figure_egress_receipt,
)
from ..gates.visual_qa import VLMVisualQAAdapter, VisualQAAuditor
from .pdf_render import render_pdf_for_run


class RuntimeProvenanceMismatchError(RuntimeError):
    """Docker steps disagree about the immutable execution environment."""


def _latex_figure_paths(
    evidence_records: Sequence[Any],
) -> Tuple[List[Tuple[str, str]], Tuple[str, ...]]:
    """Choose one LaTeX-safe export for each registered logical figure.

    Figure evidence intentionally registers several publication exports of the
    same plot.  Embedding every export duplicated figures in the manuscript and
    let TIFF files reach engines that cannot determine their bounding box.  The
    PDF renderer owns a single review document, so it selects one compile-safe
    representative per logical figure: PDF first, then PNG.

    Grouping uses the evidence store's own ``<evidence_id>__<filename>`` reader.
    Splitting on the first ``__`` here would corrupt the key whenever an
    evidence id ends in ``_``, and because each export of one figure carries its
    own id, a corrupted key silently reinstates the duplicate embedding this
    selection exists to prevent.

    Returns the selected ``(evidence_id, relative_path)`` pairs plus the ids of
    registered figures that own no compile-safe export, so the caller can report
    the omission instead of letting a figure disappear from the document.
    """

    priority = {".pdf": 0, ".png": 1}
    selected: Dict[str, Tuple[int, int, str, str]] = {}
    unrepresented: Dict[str, str] = {}
    for index, record in enumerate(evidence_records):
        if getattr(record, "kind", None) != "figure":
            continue
        relative_path = str(getattr(record, "relative_path", "") or "").replace(
            "\\", "/"
        )
        if not relative_path:
            continue
        evidence_id = str(getattr(record, "evidence_id", "") or "")
        logical_key = evidence_artifact_basename_stem(
            Path(relative_path), evidence_id
        )
        suffix = Path(relative_path).suffix.lower()
        if suffix not in priority:
            unrepresented.setdefault(logical_key, evidence_id or logical_key)
            continue
        candidate = (
            priority[suffix],
            index,
            evidence_id or logical_key,
            relative_path,
        )
        current = selected.get(logical_key)
        if current is None or candidate[:2] < current[:2]:
            selected[logical_key] = candidate
    chosen = [
        (evidence_id, relative_path)
        for _priority, _index, evidence_id, relative_path in sorted(
            selected.values(), key=lambda row: row[1]
        )
    ]
    omitted = tuple(
        identifier
        for logical_key, identifier in unrepresented.items()
        if logical_key not in selected
    )
    return chosen, omitted


def _deterministically_drop_rejected_writer_sentences(
    scaffold: str,
    rejected_sentences: Sequence[str],
) -> tuple[str, List[Dict[str, object]]]:
    """Remove STRICT-rejected prose without depending on mutable offsets.

    Rejected sentence strings can overlap (for example, a labelled sentence
    and its unlabelled body) or repeat across manuscript sections. Applying
    one indexed edit at a time can therefore remove the target of a later
    edit. The fallback is intentionally more conservative: longest exact
    rejected strings are removed first, and every remaining exact occurrence
    of the same rejected prose is removed. The unchanged STRICT gate still
    revalidates the result immediately afterwards.
    """

    sentences = [str(sentence).strip() for sentence in rejected_sentences]
    repaired = scaffold
    for sentence in sorted(set(sentences), key=lambda value: (-len(value), value)):
        if not sentence:
            continue
        while (span := _writer_repair_target_span(repaired, sentence)) is not None:
            repaired = repaired[: span[0]] + repaired[span[1] :]
    applied = [
        {
            "index": index,
            "action": "drop",
            "evidence_ids": [],
            "sentence": sentence[:500],
        }
        for index, sentence in enumerate(sentences)
    ]
    return repaired, applied


def _repair_rejected_writer_sentences(
    scaffold: str,
    *,
    llm: Any,
    evidence_ids: Sequence[str],
    evidence_digest: Optional[str],
    rejected_sentences: Sequence[str],
    scientific_claims: Dict[str, str],
    claim_required_sentences: Sequence[str],
    allowed_claim_refs: Sequence[str],
    language: str,
) -> tuple[str, List[Dict[str, object]], Optional[Dict[str, Any]]]:
    """Apply bounded model decisions or deterministically drop rejected prose.

    The optional model pass can only choose cite, exact host claim, or drop.
    Invalid structured output or an internally inconsistent application must
    never abort an otherwise valid analysis run. Dropping every sentence the
    unchanged STRICT gate already rejected is the conservative host-owned
    fallback: it cannot add a number, citation, or scientific interpretation.

    Provider transport, refusal, and budget failures deliberately propagate;
    they are not validation failures and must retain their owning boundary.
    """

    fallback_detail: Optional[Dict[str, Any]] = None
    original_scaffold = scaffold
    try:
        repair_decisions = decide_writer_evidence_repairs(
            llm,
            evidence_ids=evidence_ids,
            evidence_digest=evidence_digest,
            missing_sentences=rejected_sentences,
            scientific_claims=scientific_claims,
            claim_required_sentences=claim_required_sentences,
            language=language,
        )
        repaired, applied = _apply_writer_evidence_repair_decisions(
            original_scaffold,
            missing_sentences=rejected_sentences,
            decisions=repair_decisions,
            allowed_evidence_ids=evidence_ids,
            allowed_claim_refs=allowed_claim_refs,
        )
    except (StructuredResponseFailure, ValueError) as exc:
        repaired, applied = _deterministically_drop_rejected_writer_sentences(
            original_scaffold,
            rejected_sentences,
        )
        raw_attempts = getattr(exc, "easyicu_structured_attempt_metadata", [])
        safe_attempts = [dict(item) for item in raw_attempts if isinstance(item, dict)][
            :4
        ]
        fallback_detail = {
            "reason_code": "writer_evidence_repair_deterministic_drop",
            "exception_type": type(exc).__name__,
            "rejected_sentence_count": len(rejected_sentences),
            "structured_attempts": safe_attempts,
        }
    return repaired, applied, fallback_detail


def _drop_residual_strict_writer_sentences(
    scaffold: str,
    *,
    enforce_scaffold: Callable[[str], object],
) -> tuple[str, List[Dict[str, object]], Optional[Dict[str, Any]]]:
    """Revalidate one bounded repair and drop any prose STRICT still rejects.

    A model-selected evidence citation is not scientific-claim authority.  In
    particular, appending ``{evidence:*}`` to numeric or interpretive prose
    does not make that sentence legal under the manuscript grammar.  Run the
    unchanged owner gate immediately after the bounded repair and remove only
    the exact residual sentences it names.  A second failure propagates, so
    this helper cannot turn an unknown enforcement defect into a manuscript.
    """

    try:
        enforce_scaffold(scaffold)
    except EvidenceEnforcementError as exc:
        detail = exc.detail or {}
        raw_results = detail.get("removed_sentences", [])
        raw_claims = detail.get("unsupported_scientific_claim_sentences", [])
        result_sentences = (
            [str(value).strip() for value in raw_results if str(value).strip()]
            if isinstance(raw_results, list)
            else []
        )
        claim_sentences = (
            [str(value).strip() for value in raw_claims if str(value).strip()]
            if isinstance(raw_claims, list)
            else []
        )
        rejected = [*result_sentences, *claim_sentences]
        if not rejected:
            raise
        drop_decisions = [
            {"index": index, "action": "drop", "evidence_ids": []}
            for index in range(len(rejected))
        ]
        cleaned, applied = _apply_writer_evidence_repair_decisions(
            scaffold,
            missing_sentences=rejected,
            decisions=drop_decisions,
            allowed_claim_refs=(),
        )
        # Fail closed if anything outside the exact first-gate rejection set
        # remains invalid.  The normal bind stage will enforce the same gate
        # again, but this local check keeps the repair boundary attributable.
        enforce_scaffold(cleaned)
        return (
            cleaned,
            applied,
            {
                "reason_code": "writer_evidence_repair_residual_strict_drop",
                "rejected_sentence_count": len(rejected),
                "result_sentence_count": len(result_sentences),
                "scientific_claim_sentence_count": len(claim_sentences),
            },
        )
    return scaffold, [], None


_DEVELOPMENT_MUTABLE_PROVENANCE_FIELDS = frozenset(
    {
        "image_reference",
        "image_id",
        "repo_digests",
        "requirements_sha256",
        "execution_kernel_identity_sha256",
        "execution_kernel_source_sha256",
        "execution_kernel_files_sha256",
        "execution_kernel_file_count",
    }
)


def demote_cosmetic_publication_visual_findings(
    findings: Sequence[ValidationFinding],
) -> List[ValidationFinding]:
    """Demote only the cosmetic visual errors on the publication bundle.

    This used to demote *every* ``error`` from the final publication audit,
    and it ran before readiness — so ``readiness._is_cosmetic_visual_error``,
    which exists precisely to separate a text-spacing nit from a blank,
    cropped, unreadable or numerically inconsistent figure, never saw the
    original severity. A genuinely broken publication figure reached the
    readiness gate as a warning and could no longer block ``manuscript_ready``.

    Reusing the readiness predicate keeps one rule instead of two.
    """

    return [
        (
            finding.model_copy(update={"severity": "warning"})
            if _is_cosmetic_visual_error(finding)
            else finding
        )
        for finding in findings
    ]


def _runtime_dependency_rows(lock_bytes: bytes) -> Tuple[str, ...]:
    """Return only installable dependency pins from a runner lock."""

    return tuple(
        line.strip()
        for line in lock_bytes.decode("utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _runtime_snapshot_entries(run_dir: Path) -> List[Dict[str, Any]]:
    """Load and self-validate every per-step Docker runtime snapshot."""

    lock_paths = sorted(run_dir.glob("steps/*/outputs/runner_requirements.lock.txt"))
    provenance_paths = sorted(run_dir.glob("steps/*/outputs/runner_provenance.json"))
    if not lock_paths and not provenance_paths:
        return []
    lock_by_parent = {path.parent.resolve(): path for path in lock_paths}
    provenance_by_parent = {path.parent.resolve(): path for path in provenance_paths}
    if set(lock_by_parent) != set(provenance_by_parent):
        raise RuntimeProvenanceMismatchError(
            "Docker runtime provenance is incomplete: every step output must "
            "contain both runner_requirements.lock.txt and runner_provenance.json"
        )

    entries: List[Dict[str, Any]] = []
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
        if provenance.get("runtime") != "docker" or provenance.get("network") != "none":
            raise RuntimeProvenanceMismatchError(
                f"Unsafe Docker runtime provenance for {provenance_path}"
            )
        canonical_provenance = json.dumps(
            provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        entries.append(
            {
                "step_id": lock_path.parents[1].name,
                "lock_path": lock_path,
                "lock_bytes": lock_bytes,
                "lock_sha256": actual_lock_sha,
                "dependency_rows": _runtime_dependency_rows(lock_bytes),
                "provenance_path": provenance_path,
                "provenance": provenance,
                "canonical_provenance": canonical_provenance,
                "provenance_sha256": hashlib.sha256(
                    canonical_provenance.encode("utf-8")
                ).hexdigest(),
            }
        )
    return entries


def _validated_runtime_lock(
    run_dir: Path,
    *,
    allow_development_lineage: bool = False,
) -> Optional[Path]:
    """Return one runner lock after enforcing paper or development provenance.

    Paper-facing/default runs require byte-identical locks and provenance for
    every step. An explicitly non-paper development diagnostic may span images
    while framework fixes are tested, but only when package pins and all
    non-source runtime controls remain identical. Image and execution-kernel
    source fingerprints may change because they identify the framework build,
    not the installed dependency set or sandbox policy. The newest step lock
    is then selected for the diagnostic notebook and the full lineage is
    recorded separately by :func:`_write_development_runtime_lineage`.
    """

    entries = _runtime_snapshot_entries(run_dir)
    if not entries:
        return None

    reference = entries[0]
    reference_stable_provenance = {
        key: value
        for key, value in reference["provenance"].items()
        if key not in _DEVELOPMENT_MUTABLE_PROVENANCE_FIELDS
    }
    development_mismatch = False
    for entry in entries[1:]:
        exact_match = (
            entry["lock_bytes"] == reference["lock_bytes"]
            and entry["canonical_provenance"] == reference["canonical_provenance"]
        )
        if exact_match:
            continue
        if not allow_development_lineage:
            raise RuntimeProvenanceMismatchError(
                "Docker steps used inconsistent image provenance or dependency locks"
            )
        development_mismatch = True
        stable_provenance = {
            key: value
            for key, value in entry["provenance"].items()
            if key not in _DEVELOPMENT_MUTABLE_PROVENANCE_FIELDS
        }
        if (
            entry["dependency_rows"] != reference["dependency_rows"]
            or stable_provenance != reference_stable_provenance
        ):
            raise RuntimeProvenanceMismatchError(
                "Development resume changed dependency pins or immutable Docker "
                "runtime controls"
            )
    return entries[-1]["lock_path"] if development_mismatch else entries[0]["lock_path"]


def _write_development_runtime_lineage(run_dir: Path) -> Path:
    """Persist the exact multi-image lineage for a non-paper diagnostic run."""

    entries = _runtime_snapshot_entries(run_dir)
    if not entries:
        raise RuntimeProvenanceMismatchError(
            "Development runtime lineage requires Docker step snapshots"
        )
    selected = entries[-1]
    payload = {
        "schema_version": "easyicu.development_runtime_lineage/1",
        "paper_authority": False,
        "diagnostic_only": True,
        "mixed_runtime_snapshots": len(
            {entry["provenance_sha256"] for entry in entries}
        )
        > 1,
        "selected_notebook_lock": str(selected["lock_path"].relative_to(run_dir)),
        "selected_notebook_lock_sha256": selected["lock_sha256"],
        "steps": [
            {
                "step_id": entry["step_id"],
                "lock_path": str(entry["lock_path"].relative_to(run_dir)),
                "lock_sha256": entry["lock_sha256"],
                "dependency_rows_sha256": hashlib.sha256(
                    "\n".join(entry["dependency_rows"]).encode("utf-8")
                ).hexdigest(),
                "provenance_path": str(entry["provenance_path"].relative_to(run_dir)),
                "provenance_sha256": entry["provenance_sha256"],
                "provenance": entry["provenance"],
            }
            for entry in entries
        ],
    }
    lineage_path = run_dir / "development_runtime_lineage.json"
    lineage_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return lineage_path


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


def _persist_manuscript_quality_artifacts(
    *,
    bound: str,
    bound_evidence_id: str,
    run_dir: Path,
    evidence: Any,
    findings: List[ValidationFinding],
    expected_display_labels: Sequence[str] = (),
) -> tuple[ManuscriptQualityFinding, ...]:
    """Persist a non-authoritative reader view and its deterministic audit."""

    audit = audit_manuscript_quality(
        bound,
        expected_display_labels=expected_display_labels,
    )
    quality_audit_path = run_dir / "manuscript_quality_audit.json"
    quality_audit_path.write_text(
        json.dumps(
            audit.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    reader_path = run_dir / "manuscript_reader.md"
    reader_path.write_text(render_reader_manuscript(bound), encoding="utf-8")
    if evidence.get("manuscript_quality_audit") is None:
        evidence.register_file(
            kind="log",
            description=(
                "Deterministic reader-facing manuscript structure, terminology, "
                "and cross-section consistency audit."
            ),
            source_path=quality_audit_path,
            evidence_id="manuscript_quality_audit",
            producer="pipeline",
            generation_mode="system",
        )
    if evidence.get("manuscript_reader") is None:
        evidence.register_file(
            kind="log",
            description=(
                "Non-authoritative reader view with audit links and numeric claim "
                "footnotes removed; the bound manuscript remains authoritative."
            ),
            source_path=reader_path,
            evidence_id="manuscript_reader",
            producer="pipeline",
            generation_mode="system",
            metadata={
                "authoritative_manuscript": False,
                "source_evidence_id": bound_evidence_id,
                "source_sha256": audit.source_sha256,
            },
        )
    errors = tuple(item for item in audit.findings if item.severity == "error")
    if errors:
        findings.append(
            ValidationFinding(
                validator="manuscript_quality",
                severity="error",
                message=(
                    "Deterministic manuscript quality audit requires changes: "
                    + "; ".join(f"{item.code} ({item.section})" for item in errors[:8])
                ),
                evidence_ids=["manuscript_quality_audit", "manuscript_reader"],
                detail=audit.to_dict(),
            )
        )
    return errors


def _persist_manuscript_provenance_artifact(
    *,
    bound: str,
    numeric_binding_map: Mapping[str, Any],
    run_dir: Path,
    evidence: Any,
    findings: List[ValidationFinding],
) -> None:
    """Persist the path-free number -> JSON -> code/data reader contract."""

    try:
        payload = build_manuscript_provenance(
            manuscript=bound,
            evidence=evidence,
            binding_map=numeric_binding_map,
        )
    except ManuscriptProvenanceError as exc:
        findings.append(
            ValidationFinding(
                validator="manuscript_provenance",
                severity="error",
                message=(
                    f"Interactive manuscript provenance could not be verified: {exc}"
                ),
            )
        )
        return
    path = run_dir / "manuscript_provenance.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if evidence.get("manuscript_provenance") is None:
        evidence.register_file(
            kind="log",
            description=(
                "Path-free interactive manuscript reader provenance: every bound "
                "number maps to its JSON field and registered code/data artefacts."
            ),
            source_path=path,
            evidence_id="manuscript_provenance",
            producer="pipeline",
            generation_mode="system",
            metadata={
                "schema_version": payload["schema_version"],
                "manuscript_sha256": payload["manuscript_sha256"],
                "claim_ceiling": payload["claim_ceiling"],
            },
        )


@dataclass(frozen=True)
class _DraftStageResult:
    """Immutable handoff from manuscript generation to evidence binding."""

    current_verified_evidence_records: Sequence[Any]
    current_evidence_names: Sequence[str]
    manuscript_packet: Optional[ManuscriptDraftPacket]
    writer_error_message: Optional[str]
    scaffold: str


@dataclass(frozen=True)
class _BindingStageResult:
    """Immutable handoff from manuscript binding to publication audits."""

    bound: str
    bound_path: Path
    manuscript_critique: CritiqueReport


def _activate_publication_figure(
    pipeline: Any,
    *,
    execute_result: _ExecutePhaseResult,
    context: Any,
    evidence: Any,
    findings: List[ValidationFinding],
    role_resolver: Callable[[str], Any],
    prompt_version: str,
    run_dir: Path,
    run_id: str,
    emit_progress: Callable[..., None],
) -> None:
    """Build the deterministic publication figure after successful execution.

    Figure promotion is an analysis-output suffix, not manuscript drafting. It
    must therefore run before a requested ``stop_after_analysis`` pause. This
    keeps the pause provider-free while still closing the source-backed article
    display bundle from already registered evidence.
    """
    publication_skill_activation = compile_publication_skill_activation(
        nature_figure_enabled=pipeline._enable_publication_figure_skill,
        nature_writing_enabled=pipeline._enable_nature_writing_skill,
    )
    activation_payload = publication_skill_activation.to_dict()
    if evidence.get("publication_skill_activation") is None:
        evidence.register_json(
            kind="log",
            description=(
                "Run-bound activation receipt for the built-in Nature Figure "
                "and Nature Writing publication skills."
            ),
            payload=activation_payload,
            filename="publication_skill_activation.json",
            evidence_id="publication_skill_activation",
            aliases=["publication_skill_activation"],
            producer="publication_skill_registry",
            generation_mode="deterministic_skill",
            prompt_pack_version=prompt_version,
            metadata={
                "active_skill_ids": activation_payload["active_skill_ids"],
                "activation_sha256": activation_payload["activation_sha256"],
            },
        )
    if pipeline._enable_publication_figure_skill:
        try:
            emit_progress(
                "figure",
                "Nature Figure is rendering a manuscript-facing bundle from registered evidence.",
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
                    egress_policy = None
                    if vlm_adapter is None and pipeline._enable_vlm_visual_qa:
                        # An injected client is still a consumer of the
                        # role, so it gets the same envelope; `or` used to let
                        # it past unwrapped and unattributed.
                        client = budgeted_vlm_client(
                            pipeline, role_resolver, "vlm_visual_qa"
                        )
                        if client is not None:
                            egress_policy = pipeline._figure_egress_policy(
                                evidence=evidence, run_dir=run_dir
                            )
                            vlm_adapter = VLMVisualQAAdapter(
                                client, egress_policy=egress_policy
                            )
                    if egress_policy is not None:
                        # Phase 1 of the egress record, written *before* any
                        # byte can leave. If the upload succeeds and the host
                        # then dies, the run still carries the intent and its
                        # authority; a completed receipt that is missing is
                        # then a detectable gap rather than silence.
                        register_figure_egress_receipt(
                            policy=egress_policy,
                            evidence=evidence,
                            run_dir=run_dir,
                            phase="intent",
                        )
                    try:
                        publication_visual_findings = VisualQAAuditor(
                            vlm_adapter=vlm_adapter
                        ).audit(figure_paths=fig_paths)
                    finally:
                        # Phase 2 runs even when visual QA raised, because the
                        # upload may already have happened. Failure to write it
                        # is not demotable: an unrecorded egress is exactly the
                        # state this receipt exists to make impossible.
                        if egress_policy is not None:
                            register_figure_egress_receipt(
                                policy=egress_policy,
                                evidence=evidence,
                                run_dir=run_dir,
                                phase="completed",
                            )
                    findings.extend(
                        demote_cosmetic_publication_visual_findings(
                            publication_visual_findings
                        )
                    )
        except FigureEgressReceiptError:
            # Deliberately not caught by the blanket handler below: the run
            # sent (or may have sent) image bytes off the host and cannot say
            # so in its own evidence.
            raise
        except Exception as exc:
            findings.append(
                ValidationFinding(
                    validator="publication_figure_skill",
                    severity="warning",
                    message=(
                        "Publication figure skill failed; downstream reporting "
                        f"will use existing evidence only: {exc}"
                    ),
                )
            )


def _activate_publication_inputs(
    pipeline: Any,
    *,
    plan_result: _PlanPhaseResult,
    agent_context: Any,
    evidence: Any,
    findings: List[ValidationFinding],
    role_resolver: Callable[[str], Any],
    prompt_version: str,
    run_dir: Path,
    run_id: str,
    emit_progress: Callable[..., None],
) -> Optional[LiteratureBundle]:
    """Produce the run-bound literature bundle for manuscript drafting."""
    literature: Optional[LiteratureBundle] = plan_result.preplan_literature

    if pipeline._enable_literature and literature is None:
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
                from ..literature import PubMedLiteratureClient

                pubmed_client = PubMedLiteratureClient(
                    email=pipeline._pubmed_email,
                    api_key=pipeline._pubmed_api_key,
                )
            tavily_client = None
            if pipeline._enable_tavily:
                from ..literature import TavilyLiteratureClient

                tavily_client = TavilyLiteratureClient(
                    api_key=pipeline._tavily_api_key,
                    include_domains=pipeline._tavily_include_domains,
                    exclude_domains=pipeline._tavily_exclude_domains,
                )
            literature = LiteratureAgent(
                lit_client,
                bound_seed=pipeline._bound_preplan_literature,
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
            # The artifact is always written; what changed is that it stops
            # claiming a search happened when none did.  A run with no retrieval
            # source enabled used to publish "identified 4 ... included 4",
            # which reads as a systematic search that found four papers rather
            # than four preset references passing through untouched.
            prisma_path = run_dir / "literature_prisma.json"
            prisma_md_path = run_dir / "literature_prisma.md"
            provenance = literature.search_provenance
            prisma_path.write_text(
                json.dumps(
                    {
                        "research_question": literature.research_question,
                        "prisma": literature.prisma,
                        "search_provenance": (
                            provenance.model_dump() if provenance is not None else None
                        ),
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            if literature.prisma is not None:
                p = literature.prisma
                prisma_md = (
                    "# PRISMA 2020 flow (O21)\n\n"
                    f"- Records identified: **{p.get('identified', 0)}**\n"
                    f"- Duplicates removed: **{p.get('duplicates_removed', 0)}**\n"
                    f"- Records screened: **{p.get('screened', 0)}**\n"
                    f"- Records eligible: **{p.get('eligible', 0)}**\n"
                    f"- Records included in review: **{p.get('included', 0)}**\n"
                )
            else:
                sources = ", ".join(provenance.sources_enabled) if provenance else ""
                prisma_md = (
                    "# Literature provenance (O21)\n\n"
                    "**No PRISMA flow is reported: no literature search was "
                    "conducted for this run.**\n\n"
                    f"- Retrieval sources enabled: **{sources or 'none'}**\n"
                    "- Curated references carried through: "
                    f"**{provenance.curated_seed_count if provenance else 0}**\n\n"
                    "A PRISMA flow describes screening and selection. Reporting "
                    "one for a preset reference list would overstate what was "
                    "done.\n"
                )
            prisma_md_path.write_text(prisma_md, encoding="utf-8")
            if evidence.get("literature_prisma") is None:
                evidence.register_file(
                    kind="statistic",
                    description=(
                        "Literature search provenance, with PRISMA 2020 flow "
                        "counts when a retrieval source actually ran (O21)."
                    ),
                    source_path=prisma_path,
                    evidence_id="literature_prisma",
                    producer="literature",
                    generation_mode="system",
                )
            if evidence.get("literature_prisma_summary") is None:
                evidence.register_file(
                    kind="log",
                    description="Human-readable literature provenance summary (O21).",
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
    return literature


def _writer_execution_checkpoint_sha256(
    records: Sequence[Dict[str, Any]],
) -> str:
    """Digest the final record for every execution step."""

    payload = current_step_records(records)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verified_resume_writer_scaffold(
    *,
    resume_state: Optional[Dict[str, Any]],
    evidence: Any,
    run_dir: Path,
    per_step_records: Sequence[Dict[str, Any]],
) -> Optional[tuple[str, Dict[str, Any]]]:
    """Return the prior Writer draft only for an unchanged execution ledger.

    A report-only resume should not pay for or introduce a second free-form
    manuscript when every analysis checkpoint is unchanged. Reuse is denied
    when the prior step ledger differs, the registered raw draft is absent, or
    its immutable EvidenceStore copy fails path/digest verification. The
    caller still runs every current host-owned manuscript gate.
    """

    if not isinstance(resume_state, dict):
        return None
    prior_records = resume_state.get("per_step_records")
    if not isinstance(prior_records, list) or not prior_records:
        return None
    current_records = list(per_step_records)
    if not current_records:
        return None
    prior_digest = _writer_execution_checkpoint_sha256(prior_records)
    current_digest = _writer_execution_checkpoint_sha256(current_records)
    if prior_digest != current_digest:
        return None
    from .manuscript_sections import manuscript_writer_contract_sha256

    current_contract_sha256 = manuscript_writer_contract_sha256()
    records = [
        record
        for record in evidence.current_verified_records(per_step_records)
        if record.evidence_id == "manuscript_scaffold_raw"
        or (record.metadata or {}).get("resume_supersedes") == "manuscript_scaffold_raw"
    ]
    record = next(
        (
            candidate
            for candidate in reversed(records)
            if (candidate.metadata or {}).get("writer_contract_sha256")
            == current_contract_sha256
        ),
        None,
    )
    if record is None:
        return None
    verified_path = verified_run_evidence_path(run_dir, record)
    if verified_path is None:
        return None
    try:
        scaffold = verified_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    if not scaffold.strip():
        return None
    return scaffold, {
        "reason_code": "verified_prior_writer_scaffold_reused",
        "source_evidence_id": str(record.evidence_id),
        "source_sha256": str(record.sha256),
        "execution_checkpoint_sha256": current_digest,
        "source_relative_path": str(record.relative_path),
        "writer_contract_sha256": current_contract_sha256,
    }


def _verified_resume_writer_scaffold_for_quality_migration(
    *,
    resume_state: Optional[Dict[str, Any]],
    evidence: Any,
    run_dir: Path,
    per_step_records: Sequence[Dict[str, Any]],
) -> Optional[tuple[str, Dict[str, Any]]]:
    """Return the newest verified older-contract scaffold for targeted repair."""

    if not isinstance(resume_state, dict):
        return None
    prior_records = resume_state.get("per_step_records")
    if not isinstance(prior_records, list) or not prior_records:
        return None
    current_records = list(per_step_records)
    if not current_records:
        return None
    prior_digest = _writer_execution_checkpoint_sha256(prior_records)
    current_digest = _writer_execution_checkpoint_sha256(current_records)
    if prior_digest != current_digest:
        return None

    from .manuscript_sections import manuscript_writer_contract_sha256

    current_contract_sha256 = manuscript_writer_contract_sha256()
    candidates = [
        record
        for record in evidence.current_verified_records(per_step_records)
        if record.evidence_id == "manuscript_scaffold_raw"
        or (record.metadata or {}).get("resume_supersedes") == "manuscript_scaffold_raw"
    ]
    for record in reversed(candidates):
        prior_contract = str(
            (record.metadata or {}).get("writer_contract_sha256") or ""
        )
        if prior_contract == current_contract_sha256:
            continue
        verified_path = verified_run_evidence_path(run_dir, record)
        if verified_path is None:
            continue
        try:
            scaffold = verified_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if not scaffold.strip():
            continue
        return scaffold, {
            "reason_code": "verified_prior_writer_scaffold_quality_migration",
            "source_evidence_id": str(record.evidence_id),
            "source_sha256": str(record.sha256),
            "execution_checkpoint_sha256": current_digest,
            "source_relative_path": str(record.relative_path),
            "source_writer_contract_sha256": prior_contract or None,
            "target_writer_contract_sha256": current_contract_sha256,
        }
    return None


def _render_or_resume_writer_scaffold(
    *,
    writer: Any,
    resume_state: Optional[Dict[str, Any]],
    evidence: Any,
    run_dir: Path,
    per_step_records: Sequence[Dict[str, Any]],
    execute_result: _ExecutePhaseResult,
    literature: Optional[LiteratureBundle],
    agent_context: Any,
    preferred_evidence_names: Sequence[str],
    writer_evidence_digest: str,
    findings: List[ValidationFinding],
) -> str:
    """Select exact reuse, bounded section migration, or a fresh Writer draft."""

    resume_scaffold = _verified_resume_writer_scaffold(
        resume_state=resume_state,
        evidence=evidence,
        run_dir=run_dir,
        per_step_records=per_step_records,
    )
    if resume_scaffold is not None:
        scaffold, resume_detail = resume_scaffold
        findings.append(
            ValidationFinding(
                validator="writer_resume",
                severity="info",
                message=(
                    "Reused the digest-verified prior Writer scaffold because "
                    "the current execution checkpoint is unchanged. Current "
                    "evidence, numeric, literature, and critique gates still "
                    "revalidate the manuscript."
                ),
                evidence_ids=["manuscript_scaffold_raw"],
                detail=resume_detail,
            )
        )
        return scaffold

    administrative_authority = load_manuscript_administrative_authority(run_dir)
    literature_digest = render_writer_literature_digest(
        literature,
        plan=execute_result.plan,
    )
    migration_scaffold = _verified_resume_writer_scaffold_for_quality_migration(
        resume_state=resume_state,
        evidence=evidence,
        run_dir=run_dir,
        per_step_records=per_step_records,
    )
    if migration_scaffold is None:
        return writer.run(
            context=agent_context,
            evidence_ids=preferred_evidence_names,
            evidence_digest=writer_evidence_digest,
            literature_digest=literature_digest,
            administrative_authority=administrative_authority,
        )

    prior_scaffold, migration_detail = migration_scaffold
    try:
        scaffold, repaired_section_keys = writer.repair_existing(
            prior_scaffold,
            context=agent_context,
            evidence_ids=preferred_evidence_names,
            evidence_digest=writer_evidence_digest,
            literature_digest=literature_digest,
            administrative_authority=administrative_authority,
        )
    except Exception as exc:
        findings.append(
            ValidationFinding(
                validator="writer_resume",
                severity="error",
                message=(
                    "Writer quality migration failed; the last digest-verified "
                    "non-empty scaffold was preserved and remains unpromoted."
                ),
                evidence_ids=[migration_detail["source_evidence_id"]],
                detail={
                    **migration_detail,
                    "reason_code": "WRITER_QUALITY_MIGRATION_FAILED_PRIOR_PRESERVED",
                    "exception_type": type(exc).__name__,
                },
            )
        )
        return prior_scaffold
    findings.append(
        ValidationFinding(
            validator="writer_resume",
            severity="warning",
            message=(
                "Migrated the digest-verified prior Writer scaffold by "
                "regenerating only deterministic error-owning sections under "
                "the current Writer contract."
            ),
            evidence_ids=[migration_detail["source_evidence_id"]],
            detail={
                **migration_detail,
                "repaired_section_keys": list(repaired_section_keys),
            },
        )
    )
    return scaffold


def _rehydrate_step_numeric_authority(
    *,
    pipeline: Any,
    evidence: Any,
    per_step_records: Sequence[Dict[str, Any]],
) -> None:
    """Idempotently reapply the current headline-priority numeric cap."""

    max_leaves = (
        pipeline._max_numeric_claims_per_step
        if pipeline._max_numeric_claims_per_step > 0
        else None
    )
    for record in current_step_records(per_step_records):
        summary = record.get("step_summary")
        evidence_id = record.get("step_summary_evidence_id")
        step_id = record.get("step_id")
        if isinstance(summary, dict) and evidence_id and step_id:
            evidence.register_step_summary_numerics(
                step_id=str(step_id),
                evidence_id=str(evidence_id),
                summary=summary,
                max_leaves=max_leaves,
            )


def _repair_robustness_reader_prose(
    *,
    scaffold: str,
    run_dir: Path,
    findings: List[ValidationFinding],
) -> str:
    """Render a single robustness variant as its estimate and interval."""

    repaired, repairs = repair_single_variant_robustness_metric_prose(
        scaffold,
        panel=load_robustness_panel(run_dir / "robustness_panel.json"),
    )
    if repairs:
        findings.append(
            ValidationFinding(
                validator="manuscript_robustness_prose",
                severity="warning",
                message=(
                    "Replaced a one-variant robustness envelope with the "
                    "registered point estimate and confidence interval."
                ),
                detail={"repairs": repairs},
            )
        )
    return repaired


def _project_and_report_owner_manuscript_claims(
    scaffold: str,
    per_step_records: Sequence[Dict[str, Any]],
    findings: List[ValidationFinding],
) -> str:
    """Project typed owner claims before the unchanged STRICT gates rerun."""

    projected, repairs = project_owner_issued_manuscript_claims(
        scaffold,
        per_step_records=current_step_records(per_step_records),
    )
    if repairs:
        findings.append(
            ValidationFinding(
                validator="evidence_bound_writer",
                severity="warning",
                message=(
                    "Projected deterministic owner-issued manuscript claim(s) "
                    "omitted by Writer; the unchanged STRICT numeric and evidence "
                    "gates revalidate them."
                ),
                detail={"repairs": repairs},
            )
        )
    return projected


def _draft_manuscript(
    pipeline: Any,
    *,
    context: Any,
    agent_context: Any,
    evidence: Any,
    findings: List[ValidationFinding],
    literature: Optional[LiteratureBundle],
    per_step_records: Sequence[Dict[str, Any]],
    resume_state: Optional[Dict[str, Any]],
    prompt_version: str,
    role_resolver: Callable[[str], Any],
    runtime_state: Any,
    execute_result: _ExecutePhaseResult,
    run_dir: Path,
    run_id: str,
    run_language: str,
    emit_progress: Callable[..., None],
) -> _DraftStageResult:
    """Generate and minimally repair the evidence-aware manuscript scaffold."""
    _rehydrate_step_numeric_authority(
        pipeline=pipeline,
        evidence=evidence,
        per_step_records=per_step_records,
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
    # Produce a digest-bound appraisal surface before the Writer runs.  It is
    # deliberately unsigned and leaves comparator/difference cells blank; an
    # abstract search hit cannot authorize the Agent to declare its own work
    # novel.  Existing independently reviewed packets are never overwritten.
    novelty_path = run_dir / "novelty_positioning_audit.json"
    if not novelty_path.exists():
        novelty_packet = build_unsigned_novelty_positioning_packet(
            context=context,
            plan=execute_result.plan,
            literature=literature,
        )
        novelty_path.write_text(
            novelty_packet.model_dump_json(indent=2),
            encoding="utf-8",
        )
    if evidence.get("novelty_positioning_audit") is None:
        evidence.register_file(
            kind="log",
            description=(
                "Unsigned source-bound novelty comparison packet for independent "
                "clinical and methods appraisal."
            ),
            source_path=novelty_path,
            evidence_id="novelty_positioning_audit",
            producer="pipeline",
            generation_mode="system",
        )
    emit_progress(
        "writer",
        "Drafting manuscript scaffold.",
        run_id=run_id,
    )
    writer = ManuscriptAgent(
        role_resolver("writer"),
        language=run_language,
        nature_writing_enabled=pipeline._enable_nature_writing_skill,
        user_writing_advisory=(pipeline._user_extension_activation.writing_advisory),
    )
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
        writer_authority_records = (
            RegisteredOutputEnvelopeConsumer().authoritative_writer_records(
                current_step_records(per_step_records),
                evidence_store=evidence,
            )
        )
        if pipeline._writer_digest_widened:
            writer_evidence_digest = _render_writer_evidence_digest_v2(
                writer_authority_records,
                context=context,
                run_dir=run_dir,
                evidence=evidence,
                secondary_cap_per_step=pipeline._writer_digest_secondary_cap_per_step,
            )
        else:
            writer_evidence_digest = _render_writer_evidence_digest(
                context=context,
                run_dir=run_dir,
                per_step_records=writer_authority_records,
                evidence=evidence,
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
                    "step_result_envelope_authority": True,
                    "writer_digest_secondary_cap_per_step": int(
                        pipeline._writer_digest_secondary_cap_per_step
                    ),
                },
            )
        scaffold = _render_or_resume_writer_scaffold(
            writer=writer,
            resume_state=resume_state,
            evidence=evidence,
            run_dir=run_dir,
            per_step_records=per_step_records,
            execute_result=execute_result,
            literature=literature,
            agent_context=agent_context,
            preferred_evidence_names=preferred_writer_evidence_names,
            writer_evidence_digest=writer_evidence_digest,
            findings=findings,
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
    scaffold = _repair_robustness_reader_prose(
        scaffold=scaffold,
        run_dir=run_dir,
        findings=findings,
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
    scaffold = _project_and_report_owner_manuscript_claims(
        scaffold,
        per_step_records,
        findings,
    )
    scaffold, removed_unregistered_placeholders = (
        _remove_unregistered_evidence_placeholders(
            scaffold,
            allowed_evidence_ids=current_evidence_names,
        )
    )
    if removed_unregistered_placeholders:
        findings.append(
            ValidationFinding(
                validator="evidence_bound_writer",
                severity="warning",
                message=(
                    "Removed unregistered manuscript evidence placeholder(s) "
                    "before the unchanged STRICT sentence and binding gates."
                ),
                detail={
                    "reason_code": "unregistered_evidence_placeholder_removed",
                    "evidence_ids": removed_unregistered_placeholders,
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
    scaffold, miscitation_repairs = repair_miscited_numeric_citations(
        scaffold,
        evidence=evidence,
    )
    if miscitation_repairs:
        findings.append(
            ValidationFinding(
                validator="manuscript_numeric_auditor",
                severity="warning",
                message=(
                    "Appended the owning step's citation to "
                    f"{len(miscitation_repairs)} number(s) cited to a step that "
                    "registered no such value; the writer's own citation was "
                    "kept."
                ),
                detail={"miscitation_repairs": miscitation_repairs},
            )
        )
    scaffold, method_citation_repair = repair_missing_methods_method_citation(
        scaffold,
        literature,
        plan=execute_result.plan,
    )
    if method_citation_repair is not None:
        findings.append(
            ValidationFinding(
                validator="manuscript_literature",
                severity="warning",
                message=(
                    "Restored one omitted Methods citation from the exact "
                    "Planner-bound reporting authority."
                ),
                detail={"repair": method_citation_repair},
            )
        )
    scaffold, context_citation_repairs = repair_missing_context_section_citations(
        scaffold,
        literature,
    )
    if context_citation_repairs:
        findings.append(
            ValidationFinding(
                validator="manuscript_literature",
                severity="warning",
                message=(
                    "Restored neutral section citation(s) from the exact "
                    "run-bound contextual literature authority."
                ),
                detail={"repairs": context_citation_repairs},
            )
        )
    if (
        scaffold
        and pipeline._evidence_enforcement_mode is EvidenceEnforcementMode.STRICT
    ):
        strict_missing_sentences: List[str] = []
        strict_scientific_claim_sentences: List[str] = []
        try:
            evidence.enforce_evidence_bound_scaffold(scaffold)
        except EvidenceEnforcementError as exc:
            raw_missing = (exc.detail or {}).get("removed_sentences", [])
            if isinstance(raw_missing, list):
                strict_missing_sentences = [
                    str(sentence).strip()
                    for sentence in raw_missing
                    if str(sentence).strip()
                ]
            raw_claims = (exc.detail or {}).get(
                "unsupported_scientific_claim_sentences", []
            )
            if isinstance(raw_claims, list):
                strict_scientific_claim_sentences = [
                    str(sentence).strip()
                    for sentence in raw_claims
                    if str(sentence).strip()
                ]
        # Keep duplicate sentences: the same unsupported prose can appear in
        # Abstract, Results and Conclusion, and each occurrence must be removed
        # or replaced before the unchanged STRICT gate can pass.
        rejected_sentences = [
            *strict_missing_sentences,
            *strict_scientific_claim_sentences,
        ]
        if rejected_sentences:
            authoritative_claims = evidence.authoritative_scientific_claims(
                per_step_records
            )
            claim_text_by_ref = {
                claim.claim_ref: claim.render_text() for claim in authoritative_claims
            }
            (
                scaffold,
                applied_evidence_repairs,
                repair_fallback_detail,
            ) = _repair_rejected_writer_sentences(
                scaffold,
                llm=writer.llm,
                evidence_ids=preferred_writer_evidence_names,
                evidence_digest=writer_evidence_digest,
                rejected_sentences=rejected_sentences,
                scientific_claims=claim_text_by_ref,
                claim_required_sentences=strict_scientific_claim_sentences,
                allowed_claim_refs=tuple(claim_text_by_ref),
                language=run_language,
            )
            (
                scaffold,
                residual_strict_drops,
                residual_strict_drop_detail,
            ) = _drop_residual_strict_writer_sentences(
                scaffold,
                enforce_scaffold=evidence.enforce_evidence_bound_scaffold,
            )
            repair_message_prefix = (
                "Applied deterministic drop fallback after an invalid bounded "
                "writer evidence repair decision for "
                if repair_fallback_detail is not None
                else "Applied one bounded writer evidence repair pass to "
            )
            residual_message = (
                " "
                f"STRICT still rejected {len(residual_strict_drops)} cited or "
                "unsupported sentence(s), which the host removed before "
                "binding."
                if residual_strict_drops
                else ""
            )
            findings.append(
                ValidationFinding(
                    validator="evidence_bound_writer",
                    severity="warning",
                    message=(
                        repair_message_prefix
                        + f"{len(applied_evidence_repairs)} sentence(s); the unchanged "
                        "STRICT gate revalidated the result." + residual_message
                    ),
                    detail={
                        "evidence_repairs": applied_evidence_repairs,
                        "residual_strict_drops": residual_strict_drops,
                        **(
                            {"fallback": repair_fallback_detail}
                            if repair_fallback_detail is not None
                            else {}
                        ),
                        **(
                            {"residual_drop": residual_strict_drop_detail}
                            if residual_strict_drop_detail is not None
                            else {}
                        ),
                    },
                )
            )
    authoritative_claims = evidence.authoritative_scientific_claims(per_step_records)
    claim_placement = place_scientific_claim_tokens_in_results(
        scaffold,
        claims=authoritative_claims,
    )
    scaffold = claim_placement.scaffold
    if claim_placement.inserted_claim_refs:
        findings.append(
            ValidationFinding(
                validator="manuscript_result_sufficiency",
                severity="warning",
                message=(
                    "Inserted host-authorized scientific claim token(s) omitted "
                    "from the Results section by the Writer."
                ),
                detail={
                    "inserted_claim_refs": list(claim_placement.inserted_claim_refs)
                },
            )
        )
    if claim_placement.missing_claim_refs:
        findings.append(
            ValidationFinding(
                validator="manuscript_result_sufficiency",
                severity="error",
                message=(
                    "The manuscript has no Results section in which to place "
                    "host-authorized scientific claims."
                ),
                detail={"missing_claim_refs": list(claim_placement.missing_claim_refs)},
            )
        )
    from .manuscript_quality import repair_reader_structure_from_existing_prose

    scaffold, structural_repairs = repair_reader_structure_from_existing_prose(scaffold)
    if structural_repairs:
        if pipeline._evidence_enforcement_mode is EvidenceEnforcementMode.STRICT:
            evidence.enforce_evidence_bound_scaffold(scaffold)
        findings.append(
            ValidationFinding(
                validator="manuscript_quality",
                severity="warning",
                message=(
                    "Restored reader structure using only existing "
                    "evidence-bound manuscript prose."
                ),
                detail={"repairs": list(structural_repairs)},
            )
        )
    scaffold_path = run_dir / "manuscript_scaffold.md"
    scaffold_path.write_text(scaffold, encoding="utf-8")
    if evidence.get("manuscript_scaffold_raw") is None or scaffold.strip():
        from .manuscript_sections import manuscript_writer_contract_sha256

        evidence.register_file(
            kind="log",
            description="Manuscript scaffold (raw, with {evidence:*} placeholders).",
            source_path=scaffold_path,
            evidence_id="manuscript_scaffold_raw",
            producer="writer",
            generation_mode="llm",
            prompt_pack_version=prompt_version,
            metadata={
                "writer_contract_sha256": manuscript_writer_contract_sha256(),
            },
            on_sha_change="new_id",
        )
    return _DraftStageResult(
        current_verified_evidence_records=current_verified_evidence_records,
        current_evidence_names=current_evidence_names,
        manuscript_packet=manuscript_packet,
        writer_error_message=writer_error_message,
        scaffold=scaffold,
    )


def _bind_and_review_manuscript(
    pipeline: Any,
    *,
    critic: CriticAgent,
    evidence: Any,
    findings: List[ValidationFinding],
    literature: Optional[LiteratureBundle],
    per_step_records: Sequence[Dict[str, Any]],
    current_evidence_names: Sequence[str],
    scaffold: str,
    writer_error_message: Optional[str],
    writer_probe_mode: bool,
    writer_probe_failed_steps: Sequence[str],
    run_dir: Path,
) -> _BindingStageResult:
    """Bind manuscript claims to current evidence and persist the critique."""
    scaffold, mistyped_literature_repairs = repair_evidence_ids_mistyped_as_literature(
        scaffold,
        literature,
        evidence_ids=tuple(current_evidence_names),
    )
    if mistyped_literature_repairs:
        findings.append(
            ValidationFinding(
                validator="manuscript_literature",
                severity="warning",
                message=(
                    "Removed evidence id(s) mistyped as literature keys: "
                    + ", ".join(mistyped_literature_repairs)
                ),
                detail={"evidence_ids": mistyped_literature_repairs},
            )
        )
    scaffold, removed_unknown_keys, removed_unknown_sentences = (
        remove_sentences_with_unknown_literature_keys(scaffold, literature)
    )
    if removed_unknown_keys:
        findings.append(
            ValidationFinding(
                validator="manuscript_literature",
                severity="warning",
                message=(
                    "Deleted sentence(s) that cited keys outside the exact "
                    "run-bound literature bundle; no replacement source was inferred."
                ),
                detail={
                    "unknown_keys": removed_unknown_keys,
                    "removed_sentence_count": removed_unknown_sentences,
                },
            )
        )
    manuscript_literature_audit = audit_manuscript_literature(scaffold, literature)
    manuscript_literature_path = run_dir / "manuscript_literature_audit.json"
    manuscript_literature_path.write_text(
        manuscript_literature_audit.model_dump_json(indent=2), encoding="utf-8"
    )
    if evidence.get("manuscript_literature_audit") is None:
        evidence.register_file(
            kind="log",
            description="Exact run-bound manuscript literature citation audit.",
            source_path=manuscript_literature_path,
            evidence_id="manuscript_literature_audit",
            producer="pipeline",
            generation_mode="system",
        )
    if manuscript_literature_audit.status != "pass":
        findings.append(
            ValidationFinding(
                validator="manuscript_literature",
                severity="error",
                message=manuscript_literature_audit.message,
                evidence_ids=["manuscript_literature_audit"],
                detail=manuscript_literature_audit.model_dump(mode="json"),
            )
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
    removed_numeric_sentences: List[Dict[str, Any]] = []
    if pipeline._evidence_enforcement_mode is EvidenceEnforcementMode.STRICT:
        bound, removed_numeric_sentences = drop_untraceable_numeric_sentences(
            bound,
            evidence=evidence,
            per_step_records=per_step_records,
        )
    if removed_numeric_sentences:
        numeric_filtered_path = run_dir / "manuscript_scaffold_numeric_filtered.md"
        numeric_filtered_path.write_text(bound, encoding="utf-8")
        if evidence.get("manuscript_scaffold_numeric_filtered") is None:
            evidence.register_file(
                kind="log",
                description=(
                    "Manuscript scaffold after deterministic removal of "
                    "sentences rejected by strict numeric provenance binding."
                ),
                source_path=numeric_filtered_path,
                evidence_id="manuscript_scaffold_numeric_filtered",
                producer="pipeline",
                generation_mode="system",
            )
        findings.append(
            ValidationFinding(
                validator="manuscript_numeric_auditor",
                severity="warning",
                message=(
                    "Removed "
                    f"{len(removed_numeric_sentences)} sentence(s) rejected by "
                    "the unchanged STRICT numeric provenance gate; the full "
                    "manuscript is revalidated after this deterministic filter."
                ),
                evidence_ids=["manuscript_scaffold_numeric_filtered"],
                detail={"removed_sentences": removed_numeric_sentences},
            )
        )
    from .manuscript_quality import repair_reader_structure_from_existing_prose

    bound, post_filter_structural_repairs = repair_reader_structure_from_existing_prose(
        bound
    )
    if post_filter_structural_repairs:
        findings.append(
            ValidationFinding(
                validator="manuscript_quality",
                severity="warning",
                message=(
                    "Restored post-filter reader structure using only prose "
                    "that already survived the evidence and numeric gates."
                ),
                detail={"repairs": list(post_filter_structural_repairs)},
            )
        )
    authoritative_claims = evidence.authoritative_scientific_claims(per_step_records)
    missing_result_claims = missing_scientific_claims_in_results(
        bound,
        claims=authoritative_claims,
    )
    if missing_result_claims:
        findings.append(
            ValidationFinding(
                validator="manuscript_result_sufficiency",
                severity="error",
                message=(
                    "Final evidence/numeric filtering removed or failed to bind "
                    "host-authorized scientific claim(s) from the Results section."
                ),
                detail={"missing_claim_refs": list(missing_result_claims)},
            )
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
    numeric_binding_findings: List[ValidationFinding] = []
    if untraced_numerics:
        numeric_binding_findings.append(
            ValidationFinding(
                validator="manuscript_numeric_auditor",
                severity="error",
                message=(
                    "Bound manuscript contains numeric values that are not "
                    "traceable to registered evidence."
                ),
                detail={"untraced_numerics": list(untraced_numerics)},
            )
        )
        findings.extend(numeric_binding_findings)
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
    manuscript_value_findings = audit_manuscript_numeric_claims(
        bound,
        per_step_records=per_step_records,
    )
    findings.extend(manuscript_value_findings)
    manuscript_numeric_findings = [
        *numeric_binding_findings,
        *manuscript_value_findings,
    ]

    manuscript_quality_errors = _persist_manuscript_quality_artifacts(
        bound=bound,
        bound_evidence_id=bound_evidence_id,
        run_dir=run_dir,
        evidence=evidence,
        findings=findings,
        expected_display_labels=expected_manuscript_display_labels(
            current_evidence_names
        ),
    )
    if not writer_probe_mode:
        _persist_manuscript_provenance_artifact(
            bound=bound,
            numeric_binding_map=numeric_binding_map,
            run_dir=run_dir,
            evidence=evidence,
            findings=findings,
        )

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
    if manuscript_quality_errors:
        manuscript_critique = manuscript_critique.model_copy(
            update={
                "status": "blocked",
                "concerns": list(manuscript_critique.concerns)
                + [
                    "The deterministic manuscript quality audit found structural, "
                    "terminology, or cross-section consistency errors."
                ],
                "suggested_repairs": list(manuscript_critique.suggested_repairs)
                + [item.message for item in manuscript_quality_errors[:8]],
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
    return _BindingStageResult(
        bound=bound,
        bound_path=bound_path,
        manuscript_critique=manuscript_critique,
    )


def _publish_and_audit_manuscript(
    pipeline: Any,
    *,
    bound: str,
    bound_path: Path,
    context: Any,
    current_verified_evidence_records: Sequence[Any],
    evidence: Any,
    findings: List[ValidationFinding],
    literature: Optional[LiteratureBundle],
    manuscript_authors: Optional[Sequence[str]],
    manuscript_title: Optional[str],
    per_step_records: Sequence[Dict[str, Any]],
    repro_envelope: Any,
    run_dir: Path,
    run_id: str,
    emit_progress: Callable[..., None],
) -> None:
    """Render publication artifacts and run post-binding scientific audits."""
    if pipeline._enable_latex:
        try:
            emit_progress(
                "latex",
                "Rendering LaTeX and BibTeX scaffold.",
                run_id=run_id,
            )
            bib_basename = "manuscript_scaffold"
            # EvidenceRecord paths are already run-root-relative (normally
            # ``evidence/<file>``).  Select exactly one compile-safe export per
            # logical figure; publication TIFF remains registered for release
            # but is not a LaTeX input.
            fig_paths_for_latex, figures_without_latex_export = (
                _latex_figure_paths(current_verified_evidence_records)
            )
            if figures_without_latex_export:
                findings.append(
                    ValidationFinding(
                        validator="latex_figure_selection",
                        severity="warning",
                        message=(
                            "Registered figures own no LaTeX-safe (PDF/PNG) "
                            "export and were omitted from the review document: "
                            + ", ".join(figures_without_latex_export)
                        ),
                        evidence_ids=list(figures_without_latex_export),
                    )
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
                draft_watermark=pipeline._latex_draft_watermark,
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
            if manuscript_citable_keys(literature):
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
                    draft_watermark=pipeline._latex_draft_watermark,
                )
                if pdf_result.success and pdf_result.pdf_path is not None:
                    if evidence.get("manuscript_scaffold_pdf") is None:
                        evidence.register_file(
                            kind="log",
                            description=(
                                f"Compiled manuscript PDF (engine={pdf_result.engine})."
                            ),
                            source_path=pdf_result.pdf_path,
                            evidence_id="manuscript_scaffold_pdf",
                            producer="pipeline",
                            generation_mode="system",
                        )
                    if (
                        pdf_result.receipt_path is not None
                        and evidence.get("manuscript_pdf_receipt") is None
                    ):
                        evidence.register_file(
                            kind="log",
                            description=(
                                "Digest-bound receipt for the sandboxed manuscript "
                                "PDF render."
                            ),
                            source_path=pdf_result.receipt_path,
                            evidence_id="manuscript_pdf_receipt",
                            producer="pipeline",
                            generation_mode="system",
                        )
                    findings.append(
                        ValidationFinding(
                            validator="pdf_render",
                            severity="info",
                            message=(
                                f"Rendered manuscript PDF via {pdf_result.engine}."
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
        from ..review.methodological_rigor import (
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
        # The checklist is a coverage report over what this run PRODUCED, not
        # an analysis-facing writer consumer, so it takes its own view here
        # rather than reading the one frozen near the top of this function.
        # Everything the checklist asks about -- the bound scaffold, the causal
        # audit -- is registered by this phase AFTER that freeze, so the frozen
        # view made it report artefacts as "awaiting" that the same run had
        # already bound, and cost a fully executed run its reporting-coverage
        # score. Same digest-verified call, so an old or superseded record still
        # cannot leak in; only the moment it is taken differs.
        checklist_evidence_records = evidence.current_verified_records(per_step_records)
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
                        evidence_records=checklist_evidence_records,
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
                        evidence_records=checklist_evidence_records,
                        bound_manuscript=bound_text,
                    ),
                )
            )
        if "internal_phenotype" in wanted:
            checklist_reports.append(
                (
                    "internal_phenotype",
                    build_internal_phenotype_checklist(
                        evidence_records=checklist_evidence_records,
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
        _register_reproducibility_envelope_for_review(
            repro_envelope=repro_envelope,
            evidence=evidence,
            run_dir=run_dir,
        )
        reviewer_evidence_records = evidence.current_verified_records(per_step_records)
        reviewer_report = run_reviewer_round(
            evidence_records=reviewer_evidence_records,
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
                    f"(info={summary['counts'].get('info', 0)}, "
                    f"minor={summary['counts'].get('minor', 0)}, "
                    f"major={summary['counts'].get('major', 0)}, "
                    f"reject={summary['counts'].get('reject', 0)})."
                ),
                evidence_ids=["reviewer_report"],
                detail=summary,
            )
        )


def _register_reproducibility_envelope_for_review(
    *,
    repro_envelope: Any,
    evidence: Any,
    run_dir: Path,
) -> None:
    """Publish the completed call envelope before the reviewer reads evidence.

    Planning and manuscript generation are the stochastic stages.  By this
    point both are complete; the remaining review and finalisation code is
    deterministic.  Registering the envelope here prevents the reviewer from
    reporting it missing merely because finalisation has not run yet.
    """

    if repro_envelope is None:
        return
    envelope_path = run_dir / "reproducibility_envelope.json"
    repro_envelope.to_disk(envelope_path)
    evidence.register_file(
        kind="log",
        description=(
            "LLM reproducibility envelope (O20): per-call prompt/response "
            "sha256, requested seed, temperature, provider/model, and a "
            "PHI-safe environment snapshot."
        ),
        source_path=envelope_path,
        evidence_id="reproducibility_envelope",
        producer="pipeline",
        generation_mode="system",
        on_sha_change="new_id",
    )


def _development_runtime_lineage_allowed(pipeline: Any) -> bool:
    """Allow mixed-image lineage only for explicitly non-paper coordinates."""

    if bool(getattr(pipeline, "_development_diagnostic", False)):
        return True
    submission_profile_name = str(
        getattr(pipeline, "_submission_profile_name", "") or ""
    ).strip()
    return bool(
        submission_profile_name
        and not is_paper_facing_profile(submission_profile_name)
    )


def _write_reproducibility_artifacts(
    pipeline: Any,
    *,
    plan_result: _PlanPhaseResult,
    evidence: Any,
    findings: List[ValidationFinding],
    current_verified_evidence_records: Sequence[Any],
    run_dir: Path,
) -> None:
    """Persist the notebook, dependency lock, and development lineage."""
    # O26 — Notebook + lockfile. Concatenates every per-step
    # generated script in plan order into a single runnable
    # ``run.ipynb`` and captures the interpreter's installed
    # packages in ``requirements.lock.txt``. Runs regardless of
    # reviewer / checklist flags so the reproducibility artefacts
    # are always present.
    development_lineage_allowed = _development_runtime_lineage_allowed(pipeline)
    captured_runtime_lock = _validated_runtime_lock(
        run_dir,
        allow_development_lineage=development_lineage_allowed,
    )
    try:
        if development_lineage_allowed and captured_runtime_lock is not None:
            lineage_path = _write_development_runtime_lineage(run_dir)
            lineage_record = evidence.register_file(
                kind="log",
                description=(
                    "Exact per-step Docker image and lock lineage for an "
                    "explicitly non-paper development diagnostic."
                ),
                source_path=lineage_path,
                evidence_id="development_runtime_lineage",
                producer="pipeline",
                generation_mode="system",
                metadata={
                    "paper_authority": False,
                    "diagnostic_only": True,
                },
                on_sha_change="new_id",
            )
            findings.append(
                ValidationFinding(
                    validator="development_runtime_lineage",
                    severity="warning",
                    message=(
                        "This explicitly non-paper diagnostic resumed across "
                        "framework images. Exact per-step runtime lineage was "
                        "retained; a fresh single-image run is required for "
                        "paper authority."
                    ),
                    evidence_ids=[lineage_record.evidence_id],
                    detail={
                        "paper_authority": False,
                        "diagnostic_only": True,
                    },
                )
            )
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
        if not development_lineage_allowed:
            _assert_registered_runtime_lock_matches(evidence, lockfile_path)
        existing_lock_record = evidence.get("requirements_lockfile")
        if existing_lock_record is None or development_lineage_allowed:
            evidence.register_file(
                kind="log",
                description=(
                    "Execution-runtime requirements lockfile captured at run "
                    "time (O26)."
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
                    ),
                    "paper_authority": not development_lineage_allowed,
                    "diagnostic_only": development_lineage_allowed,
                },
                on_sha_change=(
                    "new_id" if development_lineage_allowed else "raise"
                ),
            )
    except RuntimeProvenanceMismatchError:
        raise
    except Exception as exc:
        findings.append(
            ValidationFinding(
                validator="repro_artifacts",
                severity="warning",
                message=(
                    f"Failed to build run.ipynb / lockfile: {type(exc).__name__}: {exc}"
                ),
            )
        )


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
    user_extension_receipt = pipeline._user_extension_activation.receipt
    if evidence.get("user_extension_activation") is None:
        evidence.register_json(
            kind="log",
            description=(
                "Run-bound, path-free activation receipt for user-installed "
                "Skills and MCP descriptors. MCP output is not scientific evidence."
            ),
            payload=user_extension_receipt,
            filename="user_extension_activation.json",
            evidence_id="user_extension_activation",
            aliases=["user_extension_activation"],
            producer="easyicu.extensions",
            generation_mode="system",
            prompt_pack_version=prompt_version,
            metadata={
                "activation_sha256": user_extension_receipt["activation_sha256"],
                "active_skill_count": len(user_extension_receipt["skills"]),
                "active_mcp_server_count": len(user_extension_receipt["mcp_servers"]),
                "scientific_evidence_authority": False,
            },
        )

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

    _activate_publication_figure(
        pipeline,
        execute_result=execute_result,
        context=context,
        evidence=evidence,
        findings=findings,
        role_resolver=role_resolver,
        prompt_version=prompt_version,
        run_dir=run_dir,
        run_id=run_id,
        emit_progress=emit_progress,
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

    literature = _activate_publication_inputs(
        pipeline,
        plan_result=plan_result,
        agent_context=agent_context,
        evidence=evidence,
        findings=findings,
        role_resolver=role_resolver,
        prompt_version=prompt_version,
        run_dir=run_dir,
        run_id=run_id,
        emit_progress=emit_progress,
    )

    draft = _draft_manuscript(
        pipeline,
        context=context,
        agent_context=agent_context,
        evidence=evidence,
        findings=findings,
        literature=literature,
        per_step_records=per_step_records,
        resume_state=plan_result.resume_state,
        prompt_version=prompt_version,
        role_resolver=role_resolver,
        runtime_state=runtime_state,
        execute_result=execute_result,
        run_dir=run_dir,
        run_id=run_id,
        run_language=run_language,
        emit_progress=emit_progress,
    )
    current_verified_evidence_records = draft.current_verified_evidence_records
    manuscript_packet = draft.manuscript_packet

    binding = _bind_and_review_manuscript(
        pipeline,
        critic=critic,
        evidence=evidence,
        findings=findings,
        literature=literature,
        per_step_records=per_step_records,
        current_evidence_names=draft.current_evidence_names,
        scaffold=draft.scaffold,
        writer_error_message=draft.writer_error_message,
        writer_probe_mode=writer_probe_mode,
        writer_probe_failed_steps=writer_probe_failed_steps,
        run_dir=run_dir,
    )
    bound_path = binding.bound_path
    manuscript_critique = binding.manuscript_critique

    _publish_and_audit_manuscript(
        pipeline,
        bound=binding.bound,
        bound_path=bound_path,
        context=context,
        current_verified_evidence_records=current_verified_evidence_records,
        evidence=evidence,
        findings=findings,
        literature=literature,
        manuscript_authors=manuscript_authors,
        manuscript_title=manuscript_title,
        per_step_records=per_step_records,
        repro_envelope=plan_result.repro_envelope,
        run_dir=run_dir,
        run_id=run_id,
        emit_progress=emit_progress,
    )

    _write_reproducibility_artifacts(
        pipeline,
        plan_result=plan_result,
        evidence=evidence,
        findings=findings,
        current_verified_evidence_records=current_verified_evidence_records,
        run_dir=run_dir,
    )

    return _WritePhaseResult(
        literature=literature,
        bound_path=bound_path,
        manuscript_packet=manuscript_packet,
        manuscript_critique=manuscript_critique,
        writer_probe_mode=writer_probe_mode,
        writer_probe_failed_steps=tuple(writer_probe_failed_steps),
    )
