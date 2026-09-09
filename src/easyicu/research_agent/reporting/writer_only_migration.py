"""Report-only migration of a sealed run's existing manuscript.

The owner boundary is intentionally narrow: this module reads a prior Writer
draft plus already-materialized context, literature and aggregate evidence. It
does not import the pipeline, planner, executor, coder, figure, or EvidenceStore
and it never writes into the sealed source run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Optional, Sequence
import uuid

from ..authority.evidence_snapshot import load_current_evidence_snapshot
from ..authority.manuscript_claim_policy import (
    expand_scientific_claim_tokens,
    filter_evidence_bound_scaffold,
)
from ..authority.scientific_claim_registry import load_registered_scientific_claims
from ..authority.manuscript_method_facts import load_manuscript_method_facts
from ..literature import LiteratureBundle
from ..research_context.typed import ResearchContextAuthority, parse_research_context_json
from ..schema import AnalysisPlan, EvidenceRecord
from .administrative_authority import (
    ManuscriptAdministrativeAuthority,
    load_manuscript_administrative_authority,
)
from .manuscript_literature import (
    ManuscriptLiteratureAudit,
    audit_manuscript_literature,
    remove_sentences_with_unknown_literature_keys,
    repair_missing_context_section_citations,
    repair_missing_methods_method_citation,
    render_writer_literature_digest,
)
from .manuscript_quality import (
    ManuscriptQualityAudit,
    audit_manuscript_quality,
    expected_manuscript_display_labels,
    repair_reader_structure_from_existing_prose,
    repair_registered_display_callouts,
    remove_empty_optional_subsections,
    render_reader_manuscript,
)
from .manuscript_sections import (
    completed_section_repair_candidate,
    manuscript_writer_contract_sha256,
    quality_repair_section_keys,
    quality_repair_section_errors,
)
from .manuscript_quality import repair_section_opening_connectors
from .manuscript_labels import recorded_definition_section_errors, source_bound_manuscript_labels
from .manuscript_baseline import baseline_reporting_mentions
from .manuscript_surface import deduplicate_claim_paragraphs, repair_filtered_section_openers
from .manuscript_method_facts import place_manuscript_method_facts
from .descriptive_report_facts import (
    DescriptiveReportFact, place_descriptive_report_facts, place_primary_result_summaries,
)


WRITER_ONLY_MIGRATION_SCHEMA = "easyicu.writer_only_manuscript_migration/1"
_EVIDENCE_TOKEN = re.compile(r"\{evidence:([^{}\s]+)\}")
_EVIDENCE_GROUP = re.compile(
    r"(?<!\{)\{evidence:([A-Za-z0-9][A-Za-z0-9_.-]*"
    r"(?:\s*[,;]\s*(?:evidence:)?[A-Za-z0-9][A-Za-z0-9_.-]*)+)\}(?!\})"
)
_INPUT_NAMES = (
    "manuscript_scaffold.md",
    "research_context.json",
    "analysis_plan.json",
    "preplan_literature_bundle.json",
    "writer_evidence_digest.md",
)


class WriterOnlyMigrationError(RuntimeError):
    """A report-only migration failed its owner contract."""

    def __init__(self, *, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


@dataclass(frozen=True)
class PreparedWriterOnlyMigration:
    source_run_dir: Path
    original_source_manuscript: str
    source_manuscript: str
    migration_draft_path: Optional[Path]
    migration_draft_sha256: str
    context: ResearchContextAuthority
    plan: Optional[AnalysisPlan]
    literature: LiteratureBundle
    literature_digest: str
    evidence_digest: str
    evidence_ids: tuple[str, ...]
    expected_display_labels: tuple[str, ...]
    administrative_authority: Optional[ManuscriptAdministrativeAuthority]
    source_hashes: Mapping[str, str]
    source_quality_audit: ManuscriptQualityAudit
    source_literature_audit: ManuscriptLiteratureAudit
    planned_section_keys: tuple[str, ...]
    removed_unknown_literature_keys: tuple[str, ...]
    removed_unknown_literature_sentences: int
    plan_validation_status: str = "validated"
    plan_validation_error_sha256: str = ""
    host_result_facts: tuple[DescriptiveReportFact, ...] = ()
    evidence_digest_origin: str = "saved_writer_input"


@dataclass(frozen=True)
class WriterOnlyMigrationResult:
    manuscript: str
    reader_manuscript: str
    repaired_section_keys: tuple[str, ...]
    quality_audit: ManuscriptQualityAudit
    literature_audit: ManuscriptLiteratureAudit
    deterministic_literature_repairs: tuple[Mapping[str, Any], ...]
    authority_repaired_section_keys: tuple[str, ...]
    authority_filtered_section_keys: tuple[str, ...]
    removed_unresolved_evidence_refs: tuple[str, ...]
    removed_unresolved_evidence_token_count: int
    normalized_claim_token_count: int
    abstract_conclusion_boundary_repaired: bool


@dataclass(frozen=True)
class _ReadOnlyAuthority:
    records: tuple[EvidenceRecord, ...]
    aliases: Mapping[str, str]
    claims_by_ref: Mapping[str, Any]


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_regular(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_INPUT_UNAVAILABLE",
            detail=f"required regular input is unavailable: {path.name}",
        )
    raw = path.read_bytes()
    if not raw.strip():
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_INPUT_EMPTY",
            detail=f"required input is empty: {path.name}",
        )
    return raw


def _input_hashes(run_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in _INPUT_NAMES:
        hashes[name] = _sha256(_read_regular(run_dir / name))
    authority = run_dir / "authorities" / "manuscript_administrative_authority.json"
    if authority.exists():
        hashes[str(authority.relative_to(run_dir))] = _sha256(_read_regular(authority))
    return hashes


def _evidence_ids(run_dir: Path, manuscript: str, digest: str) -> tuple[str, ...]:
    ids = set(_EVIDENCE_TOKEN.findall(manuscript))
    ids.update(_EVIDENCE_TOKEN.findall(digest))
    aliases_path = run_dir / "evidence" / "evidence_aliases.json"
    if aliases_path.is_file() and not aliases_path.is_symlink():
        try:
            aliases = json.loads(aliases_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            aliases = None
        if isinstance(aliases, dict):
            for label in ("table_one", "publication_figure_contract"):
                if label in aliases:
                    ids.add(label)
    return tuple(sorted(ids))


def _read_only_authority(run_dir: Path) -> _ReadOnlyAuthority:
    try:
        snapshot = load_current_evidence_snapshot(run_dir)
        records = tuple(EvidenceRecord.model_validate(item) for item in snapshot.records)
        claims = load_registered_scientific_claims(root=run_dir, records=records)
    except Exception as exc:
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_EVIDENCE_AUTHORITY_INVALID",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    return _ReadOnlyAuthority(
        records=records,
        aliases=dict(snapshot.aliases),
        claims_by_ref={claim.claim_ref: claim for claim in claims},
    )


def _claim_reader_view(run_dir: Path, manuscript: str) -> str:
    """Audit the rendered verified claims while retaining the canonical tokens."""
    if "{claim" not in manuscript:
        return manuscript
    authority = _read_only_authority(run_dir)
    expanded = expand_scientific_claim_tokens(
        manuscript, resolve_claim=authority.claims_by_ref.get,
        current_evidence_ids={record.evidence_id for record in authority.records},
    )
    if expanded.missing_claim_refs or expanded.malformed_sentences:
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_SCIENTIFIC_CLAIM_BINDING_FAILED",
            detail="Reader validation requires complete, current registered claims.",
        )
    return expanded.scaffold


def _section_key_for_excerpt(manuscript: str, excerpt: str) -> Optional[str]:
    matches = list(re.finditer(r"^##\s+([^\n]+?)\s*$", manuscript, flags=re.M))
    keys = {
        "title": "title",
        "abstract": "abstract",
        "introduction": "introduction",
        "methods": "methods",
        "results": "results",
        "discussion": "discussion",
        "limitations": "limitations",
        "conclusion": "conclusion",
    }
    for index, match in enumerate(matches):
        key = keys.get(match.group(1).strip().casefold())
        if key is None:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(manuscript)
        if excerpt in manuscript[match.start() : end]:
            return key
    return None


def _normalize_registered_evidence_groups(manuscript: str, authority: _ReadOnlyAuthority) -> str:
    """Canonicalize only a closed list of exact registered citation ids."""

    records_by_id = {record.evidence_id: record for record in authority.records}

    def resolve_evidence(ref: str) -> bool:
        evidence_id = authority.aliases.get(ref, ref)
        return evidence_id in records_by_id

    # The model may group citations using bibliography punctuation. Split only
    # exact, registered identifiers; unknown or malformed groups stay rejected.
    # This changes citation syntax, never the prose or the source of a value.
    def normalize_group(match: re.Match[str]) -> str:
        refs = [part.strip().removeprefix("evidence:")
                for part in re.split(r"[,;]", match.group(1))]
        if not all(resolve_evidence(ref) for ref in refs):
            return match.group(0)
        return " ".join(f"{{evidence:{ref}}}" for ref in dict.fromkeys(refs))

    return _EVIDENCE_GROUP.sub(normalize_group, manuscript)


def _claim_policy_projection(
    run_dir: Path,
    manuscript: str,
) -> tuple[str, dict[str, tuple[str, ...]]]:
    authority = _read_only_authority(run_dir)
    records_by_id = {record.evidence_id: record for record in authority.records}

    def resolve_evidence(ref: str) -> bool:
        return authority.aliases.get(ref, ref) in records_by_id

    manuscript = _normalize_registered_evidence_groups(manuscript, authority)
    manuscript, _ = _normalize_claim_token_sentences(manuscript)
    facts = load_manuscript_method_facts(root=run_dir, records=authority.records)
    manuscript, _ = place_manuscript_method_facts(manuscript, facts)
    filtered = filter_evidence_bound_scaffold(
        manuscript,
        resolve_claim=authority.claims_by_ref.get,
        resolve_evidence=resolve_evidence,
        method_facts=facts,
    )
    cleaned = repair_filtered_section_openers(filtered.scaffold, before_filter=manuscript)
    cleaned = deduplicate_claim_paragraphs(cleaned)
    rejected = tuple(
        dict.fromkeys(
            (
                *filtered.removed_result_sentences,
                *filtered.unsupported_scientific_claim_sentences,
            )
        )
    )
    by_section: dict[str, list[str]] = {}
    for excerpt in rejected:
        key = _section_key_for_excerpt(manuscript, excerpt)
        if key is None:
            raise WriterOnlyMigrationError(
                code="WRITER_ONLY_AUTHORITY_OWNER_UNRESOLVED",
                detail=_sha256(excerpt.encode("utf-8")),
            )
        by_section.setdefault(key, []).append(excerpt[:500])
    return cleaned, {
        key: tuple(values) for key, values in by_section.items()
    }


def _remove_unresolved_evidence_tokens(
    run_dir: Path,
    manuscript: str,
) -> tuple[str, tuple[str, ...], int]:
    """Remove unresolved tokens before sentence-level authority filtering."""

    authority = _read_only_authority(run_dir)
    manuscript = _normalize_registered_evidence_groups(manuscript, authority)
    records_by_id = {record.evidence_id for record in authority.records}
    removed: list[str] = []

    def replace(match: re.Match[str]) -> str:
        ref = match.group(1)
        evidence_id = authority.aliases.get(ref, ref)
        if evidence_id in records_by_id:
            return match.group(0)
        removed.append(ref)
        return ""

    cleaned = _EVIDENCE_TOKEN.sub(replace, manuscript)
    cleaned = re.sub(r"[ \t]+(?=\n)", "", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return cleaned, tuple(dict.fromkeys(removed)), len(removed)


def _normalize_claim_token_sentences(manuscript: str) -> tuple[str, int]:
    """Place each exact claim token on its own paragraph before filtering."""

    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"\n\n{match.group(1)}\n\n"

    normalized = re.sub(
        r"\s*(\{claim:[^{}\s]+\})\s*",
        replace,
        manuscript,
    )
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip() + "\n", count


def _repair_abstract_conclusion_boundary(
    manuscript: str,
    literature: LiteratureBundle,
    *,
    rejected_sentences: Sequence[str],
) -> tuple[str, bool]:
    """Repair only a rejected conclusion, never unrelated abstract findings."""

    available = {record.key for record in literature.citations}
    causal_key = next(
        (key for key in ("strobe_2007", "record_2015") if key in available),
        None,
    )
    validation_key = next(
        (key for key in ("record_2015", "strobe_2007") if key in available),
        None,
    )
    if causal_key is None or validation_key is None:
        return manuscript, False
    abstract_match = re.search(
        r"(^##\s+Abstract\s*$)(.*?)(?=^##\s+|\Z)",
        manuscript,
        flags=re.M | re.S,
    )
    if abstract_match is None or "**Conclusions:**" not in abstract_match.group(2):
        return manuscript, False
    conclusion = abstract_match.group(2).split("**Conclusions:**", 1)[1]
    if not any(excerpt and excerpt in conclusion for excerpt in rejected_sentences):
        return manuscript, False
    replacement = (
        "**Conclusions:** Because this was an observational analysis, the "
        f"estimates do not establish causation [@{causal_key}]. Independent "
        "validation in other cohorts is required before broader interpretation "
        f"[@{validation_key}]."
    )
    repaired_abstract = re.sub(
        r"\*\*Conclusions:\*\*.*\Z",
        replacement,
        abstract_match.group(2).strip(),
        count=1,
        flags=re.S,
    )
    repaired = (
        manuscript[: abstract_match.start()]
        + abstract_match.group(1)
        + "\n\n"
        + repaired_abstract
        + "\n\n"
        + manuscript[abstract_match.end() :]
    )
    return repaired, repaired != manuscript


def prepare_writer_only_migration(
    run_dir: Path,
    *,
    migration_draft: Optional[Path] = None,
    host_verified_evidence_digest: Optional[str] = None,
) -> PreparedWriterOnlyMigration:
    """Load and audit one sealed run without changing it."""

    source = Path(run_dir).expanduser().resolve(strict=True)
    if not source.is_dir():
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_SOURCE_NOT_DIRECTORY",
            detail=str(source),
        )
    hashes = _input_hashes(source)
    original_manuscript = (source / "manuscript_scaffold.md").read_text(
        encoding="utf-8"
    )
    draft_path: Optional[Path] = None
    manuscript = original_manuscript
    if migration_draft is not None:
        draft_path = Path(migration_draft).expanduser().resolve(strict=True)
        try:
            draft_path.relative_to(source)
        except ValueError:
            pass
        else:
            raise WriterOnlyMigrationError(
                code="WRITER_ONLY_MIGRATION_DRAFT_INSIDE_SOURCE",
                detail=str(draft_path),
            )
        manuscript = _read_regular(draft_path).decode("utf-8")
    evidence_digest = (
        host_verified_evidence_digest if host_verified_evidence_digest is not None
        else (source / "writer_evidence_digest.md").read_text(encoding="utf-8")
    )
    try:
        context = parse_research_context_json(
            (source / "research_context.json").read_bytes()
        )
        literature = LiteratureBundle.model_validate_json(
            (source / "preplan_literature_bundle.json").read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_TYPED_INPUT_INVALID",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    raw_plan = (source / "analysis_plan.json").read_text(encoding="utf-8")
    plan_validation_status = "validated"
    plan_validation_error_sha256 = ""
    try:
        plan: Optional[AnalysisPlan] = AnalysisPlan.model_validate_json(raw_plan)
    except Exception as exc:
        try:
            raw_plan_payload = json.loads(raw_plan)
        except json.JSONDecodeError as parse_exc:
            raise WriterOnlyMigrationError(
                code="WRITER_ONLY_TYPED_INPUT_INVALID",
                detail=f"{type(parse_exc).__name__}: {parse_exc}",
            ) from parse_exc
        if not isinstance(raw_plan_payload, dict):
            raise WriterOnlyMigrationError(
                code="WRITER_ONLY_TYPED_INPUT_INVALID",
                detail="analysis_plan.json must contain a JSON object",
            ) from exc
        plan = None
        plan_validation_status = "legacy_schema_incompatible_report_only"
        plan_validation_error_sha256 = _sha256(
            f"{type(exc).__name__}: {exc}".encode("utf-8")
        )
    manuscript, unknown_keys, removed_sentences = (
        remove_sentences_with_unknown_literature_keys(
            manuscript,
            literature,
        )
    )
    manuscript, context_citation_repairs = repair_missing_context_section_citations(
        manuscript,
        literature,
    )
    manuscript, methods_citation_repair = repair_missing_methods_method_citation(
        manuscript,
        literature,
        plan=plan,
    )
    ids = _evidence_ids(source, manuscript, evidence_digest)
    labels = expected_manuscript_display_labels(ids)
    manuscript, _structural_repairs = repair_reader_structure_from_existing_prose(
        manuscript
    )
    manuscript, _display_repairs = repair_registered_display_callouts(
        manuscript,
        expected_display_labels=labels,
    )
    source_quality = audit_manuscript_quality(
        manuscript,
        analysis_plan=plan,
        expected_display_labels=labels,
        expected_baseline_mentions=baseline_reporting_mentions(context, plan.display_labels if plan else None),
    )
    source_literature = audit_manuscript_literature(manuscript, literature)
    return PreparedWriterOnlyMigration(
        source_run_dir=source,
        original_source_manuscript=original_manuscript,
        source_manuscript=manuscript,
        migration_draft_path=draft_path,
        migration_draft_sha256=_sha256(manuscript.encode("utf-8")),
        context=context,
        plan=plan,
        literature=literature,
        literature_digest=render_writer_literature_digest(literature, plan=plan),
        evidence_digest=evidence_digest,
        evidence_ids=ids,
        expected_display_labels=labels,
        administrative_authority=load_manuscript_administrative_authority(source),
        source_hashes=hashes,
        source_quality_audit=source_quality,
        source_literature_audit=source_literature,
        planned_section_keys=quality_repair_section_keys(
            manuscript,
            analysis_plan=plan,
            expected_display_labels=labels,
            expected_baseline_mentions=baseline_reporting_mentions(context, plan.display_labels if plan else None),
        ),
        removed_unknown_literature_keys=tuple(unknown_keys),
        removed_unknown_literature_sentences=int(removed_sentences),
        plan_validation_status=plan_validation_status,
        plan_validation_error_sha256=plan_validation_error_sha256,
        evidence_digest_origin=("verified_current_output_envelopes"
                                if host_verified_evidence_digest is not None else "saved_writer_input"),
    )


def writer_only_preflight_payload(
    prepared: PreparedWriterOnlyMigration,
) -> dict[str, Any]:
    """Render the zero-Provider repair plan."""

    return {
        "schema_version": WRITER_ONLY_MIGRATION_SCHEMA,
        "mode": "preflight",
        "source_run_dir": str(prepared.source_run_dir),
        "source_hashes": dict(prepared.source_hashes),
        "writer_evidence_digest_sha256": _sha256(prepared.evidence_digest.encode("utf-8")),
        "writer_evidence_digest_origin": prepared.evidence_digest_origin,
        "writer_contract_sha256": manuscript_writer_contract_sha256(),
        "migration_draft_path": (
            str(prepared.migration_draft_path)
            if prepared.migration_draft_path is not None
            else None
        ),
        "migration_draft_sha256": prepared.migration_draft_sha256,
        "source_quality_status": prepared.source_quality_audit.status,
        "source_quality_findings": [
            asdict(finding)
            for finding in prepared.source_quality_audit.findings
        ],
        "source_literature_status": prepared.source_literature_audit.status,
        "plan_validation_status": prepared.plan_validation_status,
        "plan_validation_error_sha256": prepared.plan_validation_error_sha256,
        "planned_section_keys": list(prepared.planned_section_keys),
        "removed_unknown_literature_keys": list(
            prepared.removed_unknown_literature_keys
        ),
        "removed_unknown_literature_sentences": (
            prepared.removed_unknown_literature_sentences
        ),
        "expected_display_labels": list(prepared.expected_display_labels),
        "host_result_fact_count": len(prepared.host_result_facts),
        "provider_calls": 0,
        "forbidden_roles": ["planner", "executor", "coder", "figure"],
        "claim_ceiling": "analysis_only",
        "publication_authorized": False,
    }


def repair_writer_only(
    prepared: PreparedWriterOnlyMigration,
    *,
    writer: Any,
) -> WriterOnlyMigrationResult:
    """Repair only deterministic Writer section owners in memory."""

    if (
        prepared.plan is None
        and prepared.plan_validation_status
        == "legacy_schema_incompatible_report_only"
        and prepared.planned_section_keys
    ):
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_LEGACY_PLAN_REPAIR_FORBIDDEN",
            detail=", ".join(prepared.planned_section_keys),
        )

    source_manuscript = prepared.source_manuscript
    reader_labels = source_bound_manuscript_labels(
        prepared.context, prepared.plan.display_labels if prepared.plan else {},
        language=getattr(writer, "language", "en"),
    )
    if prepared.host_result_facts:
        # Compile the mechanical core before deciding which prose still needs
        # a model. These facts are verified outputs, not hand-written answers.
        source_manuscript, _ = _claim_policy_projection(
            prepared.source_run_dir, source_manuscript,
        )
        source_manuscript = place_descriptive_report_facts(source_manuscript, prepared.host_result_facts)
        source_manuscript = place_primary_result_summaries(source_manuscript, prepared.host_result_facts)
        source_manuscript, _ = repair_registered_display_callouts(
            source_manuscript, expected_display_labels=prepared.expected_display_labels,
        )
        source_manuscript = remove_empty_optional_subsections(source_manuscript)
    try:
        manuscript, repaired_keys = writer.repair_existing(
            source_manuscript,
            analysis_plan=prepared.plan,
            context=prepared.context,
            evidence_ids=prepared.evidence_ids,
            evidence_digest=prepared.evidence_digest,
            literature_digest=prepared.literature_digest,
            reader_display_labels=reader_labels,
            administrative_authority=prepared.administrative_authority,
        )
    except WriterOnlyMigrationError:
        # Preserve replay drift identity; it must never become a reusable
        # shorter cache prefix on the next report recovery.
        raise
    except Exception as exc:
        candidate = completed_section_repair_candidate(exc)
        if candidate is None:
            raise WriterOnlyMigrationError(
                code="WRITER_ONLY_REPAIR_FAILED_PRIOR_PRESERVED",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        manuscript, repaired_keys = candidate
    if not str(manuscript or "").strip():
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_REPAIR_EMPTY_PRIOR_PRESERVED",
            detail="Writer returned an empty manuscript",
        )
    if _input_hashes(prepared.source_run_dir) != dict(prepared.source_hashes):
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_SOURCE_CHANGED_DURING_REPAIR",
            detail="one or more sealed source inputs changed during migration",
        )
    manuscript, unknown_keys, removed_sentences = (
        remove_sentences_with_unknown_literature_keys(
            manuscript,
            prepared.literature,
        )
    )
    manuscript, context_repairs = repair_missing_context_section_citations(
        manuscript,
        prepared.literature,
    )
    manuscript, method_repair = repair_missing_methods_method_citation(
        manuscript,
        prepared.literature,
        plan=prepared.plan,
    )
    deterministic_literature_repairs: list[Mapping[str, Any]] = [
        {
            "kind": "removed_unknown_literature_sentences",
            "keys": list(unknown_keys),
            "sentence_count": int(removed_sentences),
        }
    ] if unknown_keys else []
    deterministic_literature_repairs.extend(context_repairs)
    if method_repair is not None:
        deterministic_literature_repairs.append(
            {"kind": "methods_reporting_citation", **method_repair}
        )
    (
        manuscript,
        removed_unresolved_evidence_refs,
        removed_unresolved_evidence_token_count,
    ) = _remove_unresolved_evidence_tokens(
        prepared.source_run_dir,
        manuscript,
    )
    manuscript, normalized_claim_token_count = _normalize_claim_token_sentences(
        manuscript
    )
    abstract_conclusion_boundary_repaired = False
    authority_repaired: list[str] = []
    authority_filtered: list[str] = []
    _, initial_errors = _claim_policy_projection(prepared.source_run_dir, manuscript)
    if "abstract" in initial_errors:
        manuscript, abstract_conclusion_boundary_repaired = _repair_abstract_conclusion_boundary(
            manuscript, prepared.literature, rejected_sentences=initial_errors["abstract"],
        )
    # Two model repair passes, each followed by a real validation. A successful
    # final repair must not fall through a for/else and be reported exhausted.
    # Deterministic normalization and callout placement consume no model pass.
    for _attempt in range(3):
        canonical, section_errors = _claim_policy_projection(
            prepared.source_run_dir,
            manuscript,
        )
        canonical = place_descriptive_report_facts(canonical, prepared.host_result_facts)
        canonical = place_primary_result_summaries(canonical, prepared.host_result_facts)
        canonical, _ = repair_registered_display_callouts(
            canonical, expected_display_labels=prepared.expected_display_labels,
        )
        canonical = remove_empty_optional_subsections(canonical)
        canonical = repair_section_opening_connectors(canonical)
        reader_view = _claim_reader_view(prepared.source_run_dir, canonical)
        canonical_quality = audit_manuscript_quality(
            reader_view,
            analysis_plan=prepared.plan,
            expected_primary_result_facts=prepared.host_result_facts,
            expected_display_labels=prepared.expected_display_labels,
            expected_baseline_mentions=baseline_reporting_mentions(prepared.context, prepared.plan.display_labels if prepared.plan else None),
        )
        canonical_literature = audit_manuscript_literature(
            reader_view,
            prepared.literature,
        )
        if prepared.literature.citations and not canonical_literature.direct_comparator_keys_available:
            boundary = (
                "The literature assembled for this report did not include a verified direct comparator; "
                "a like-for-like comparison with published estimates is not supported by the available sources "
                "{evidence:preplan_literature_bundle}."
            )
            if boundary not in canonical:
                canonical = re.sub(r"(?ms)(^## Limitations\s*\n.*?)(?=^## |\Z)",
                                   lambda match: match.group(1).rstrip() + "\n\n" + boundary + "\n\n",
                                   canonical, count=1)
        recording_errors = recorded_definition_section_errors(reader_view, prepared.context)
        if (
            canonical_quality.status == "pass"
            and canonical_literature.status == "pass"
            and not recording_errors
        ):
            manuscript = canonical
            authority_filtered.extend(
                key for key in section_errors if key not in authority_filtered
            )
            break
        if _attempt == 2:
            raise WriterOnlyMigrationError(
                code="WRITER_ONLY_AUTHORITY_REPAIR_EXHAUSTED_PRIOR_PRESERVED",
                detail=", ".join(sorted(section_errors)),
            )
        repair_errors = quality_repair_section_errors(
            reader_view,
            analysis_plan=prepared.plan,
            expected_primary_result_facts=prepared.host_result_facts,
            expected_display_labels=prepared.expected_display_labels,
            expected_baseline_mentions=baseline_reporting_mentions(
                prepared.context, prepared.plan.display_labels if prepared.plan else None,
            ),
        )
        repair_errors.update(recording_errors)
        for key in (
            *canonical_literature.missing_required_citation_sections,
            *canonical_literature.direct_comparator_sections_missing,
            *(("methods",) if canonical_literature.methods_method_source_missing else ()),
        ):
            repair_errors[str(key).lower()] = (canonical_literature.message,)
        if not repair_errors:
            raise WriterOnlyMigrationError(
                code="WRITER_ONLY_REPAIR_OWNER_UNRESOLVED",
                detail="The canonical report failed a gate with no section-owned repair.",
            )
        # Reject unsupported sentences once. Do not rewrite an otherwise valid
        # section merely because its unsupported optional prose was removed.
        repair_errors = {
            key: (*details, *section_errors.get(key, ()))
            for key, details in repair_errors.items()
        }
        authority_filtered.extend(
            key for key in section_errors if key not in authority_filtered
        )
        repair_sections = getattr(writer, "repair_sections", None)
        if not callable(repair_sections):
            raise WriterOnlyMigrationError(
                code="WRITER_ONLY_AUTHORITY_REPAIR_UNAVAILABLE",
                detail=", ".join(sorted(repair_errors)),
            )
        try:
            manuscript, repaired_authority_keys = repair_sections(
                canonical,
                analysis_plan=prepared.plan,
                section_errors=repair_errors,
                context=prepared.context,
                evidence_ids=prepared.evidence_ids,
                evidence_digest=prepared.evidence_digest,
                literature_digest=prepared.literature_digest,
                reader_display_labels=reader_labels,
                administrative_authority=prepared.administrative_authority,
            )
        except WriterOnlyMigrationError:
            raise
        except Exception as exc:
            candidate = completed_section_repair_candidate(
                exc, expected_section_keys=tuple(repair_errors),
            )
            if candidate is None:
                raise WriterOnlyMigrationError(
                    code="WRITER_ONLY_AUTHORITY_REPAIR_FAILED_PRIOR_PRESERVED",
                    detail=f"{type(exc).__name__}: {exc}",
                ) from exc
            manuscript, repaired_authority_keys = candidate
        for key in repaired_authority_keys:
            if key not in authority_repaired:
                authority_repaired.append(key)
    reader_view = _claim_reader_view(prepared.source_run_dir, manuscript)
    quality = audit_manuscript_quality(
        reader_view,
        analysis_plan=prepared.plan,
        expected_primary_result_facts=prepared.host_result_facts,
        expected_display_labels=prepared.expected_display_labels,
        expected_baseline_mentions=baseline_reporting_mentions(prepared.context, prepared.plan.display_labels if prepared.plan else None),
    )
    if quality.status != "pass":
        codes = sorted({finding.code for finding in quality.findings})
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_QUALITY_AUDIT_FAILED_PRIOR_PRESERVED",
            detail=", ".join(codes),
        )
    literature = audit_manuscript_literature(reader_view, prepared.literature)
    if literature.status != "pass":
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_LITERATURE_AUDIT_FAILED_PRIOR_PRESERVED",
            detail=literature.message,
        )
    return WriterOnlyMigrationResult(
        manuscript=manuscript,
        reader_manuscript=render_reader_manuscript(reader_view),
        repaired_section_keys=tuple(repaired_keys),
        quality_audit=quality,
        literature_audit=literature,
        deterministic_literature_repairs=tuple(deterministic_literature_repairs),
        authority_repaired_section_keys=tuple(authority_repaired),
        authority_filtered_section_keys=tuple(authority_filtered),
        removed_unresolved_evidence_refs=removed_unresolved_evidence_refs,
        removed_unresolved_evidence_token_count=(
            removed_unresolved_evidence_token_count
        ),
        normalized_claim_token_count=normalized_claim_token_count,
        abstract_conclusion_boundary_repaired=(
            abstract_conclusion_boundary_repaired
        ),
    )


def _bind_and_copy_evidence(
    prepared: PreparedWriterOnlyMigration,
    manuscript: str,
    *,
    output_dir: Path,
) -> tuple[str, tuple[str, ...]]:
    authority = _read_only_authority(prepared.source_run_dir)
    records_by_id = {record.evidence_id: record for record in authority.records}
    expanded = expand_scientific_claim_tokens(
        manuscript,
        resolve_claim=authority.claims_by_ref.get,
        current_evidence_ids=set(records_by_id),
    )
    if expanded.missing_claim_refs or expanded.malformed_sentences:
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_SCIENTIFIC_CLAIM_BINDING_FAILED",
            detail=(
                f"missing={list(expanded.missing_claim_refs)}; "
                f"malformed={len(expanded.malformed_sentences)}"
            ),
        )
    copied: list[str] = []

    def replace(match: re.Match[str]) -> str:
        ref = match.group(1)
        evidence_id = authority.aliases.get(ref, ref)
        record = records_by_id.get(evidence_id)
        if record is None:
            raise WriterOnlyMigrationError(
                code="WRITER_ONLY_EVIDENCE_REFERENCE_UNRESOLVED",
                detail=ref,
            )
        source_path = (prepared.source_run_dir / record.relative_path).resolve()
        try:
            source_path.relative_to(prepared.source_run_dir)
        except ValueError as exc:
            raise WriterOnlyMigrationError(
                code="WRITER_ONLY_EVIDENCE_PATH_ESCAPE",
                detail=record.evidence_id,
            ) from exc
        if not source_path.is_file() or source_path.is_symlink():
            raise WriterOnlyMigrationError(
                code="WRITER_ONLY_EVIDENCE_FILE_UNAVAILABLE",
                detail=record.evidence_id,
            )
        if _sha256(source_path.read_bytes()) != record.sha256:
            raise WriterOnlyMigrationError(
                code="WRITER_ONLY_EVIDENCE_DIGEST_DRIFT",
                detail=record.evidence_id,
            )
        destination = output_dir / "evidence" / source_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source_path, destination)
        if _sha256(destination.read_bytes()) != record.sha256:
            raise WriterOnlyMigrationError(
                code="WRITER_ONLY_EVIDENCE_COPY_DRIFT",
                detail=record.evidence_id,
            )
        relative = destination.relative_to(output_dir).as_posix()
        copied.append(relative)
        return f'[{ref}]({relative} "sha256={record.sha256[:8]}")'

    bound = _EVIDENCE_TOKEN.sub(replace, expanded.scaffold)
    return bound, tuple(dict.fromkeys(copied))


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short writer-only artifact write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def publish_writer_only_result(
    prepared: PreparedWriterOnlyMigration,
    result: WriterOnlyMigrationResult,
    *,
    output_dir: Path,
    provider: str,
    model: str,
    provider_summary: Mapping[str, Any],
    provider_ledger: str,
) -> dict[str, Any]:
    """Publish one successful migration outside the sealed run."""

    output = Path(output_dir).expanduser().resolve()
    try:
        output.relative_to(prepared.source_run_dir)
    except ValueError:
        pass
    else:
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_OUTPUT_INSIDE_SOURCE",
            detail=str(output),
        )
    output.mkdir(parents=True, exist_ok=True)
    bound_manuscript, copied_evidence = _bind_and_copy_evidence(
        prepared,
        result.manuscript,
        output_dir=output,
    )
    bound_quality = audit_manuscript_quality(
        bound_manuscript,
        analysis_plan=prepared.plan,
        expected_display_labels=prepared.expected_display_labels,
        expected_baseline_mentions=baseline_reporting_mentions(prepared.context, prepared.plan.display_labels if prepared.plan else None),
    )
    if bound_quality.status != "pass":
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_BOUND_QUALITY_AUDIT_FAILED",
            detail=", ".join(sorted({item.code for item in bound_quality.findings})),
        )
    bound_literature = audit_manuscript_literature(
        bound_manuscript,
        prepared.literature,
    )
    if bound_literature.status != "pass":
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_BOUND_LITERATURE_AUDIT_FAILED",
            detail=bound_literature.message,
        )
    _atomic_write(output / "manuscript_scaffold.md", result.manuscript.encode("utf-8"))
    _atomic_write(output / "writer_evidence_digest.md", prepared.evidence_digest.encode("utf-8"))
    _atomic_write(
        output / "manuscript_bound.md",
        bound_manuscript.encode("utf-8"),
    )
    _atomic_write(
        output / "manuscript_reader.md",
        render_reader_manuscript(bound_manuscript).encode("utf-8"),
    )
    quality_payload = result.quality_audit.to_dict()
    literature_payload = result.literature_audit.model_dump(mode="json")
    _atomic_write(
        output / "manuscript_quality_audit.json",
        (json.dumps(quality_payload, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    _atomic_write(
        output / "manuscript_literature_audit.json",
        (json.dumps(literature_payload, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    _atomic_write(
        output / "cost_summary.json",
        (json.dumps(dict(provider_summary), ensure_ascii=False, indent=2) + "\n").encode(),
    )
    receipt = {
        "schema_version": WRITER_ONLY_MIGRATION_SCHEMA,
        "status": "pass",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(prepared.source_run_dir),
        "source_hashes": dict(prepared.source_hashes),
        "writer_evidence_digest_sha256": _sha256(prepared.evidence_digest.encode("utf-8")),
        "writer_evidence_digest_origin": prepared.evidence_digest_origin,
        "plan_validation_status": prepared.plan_validation_status,
        "plan_validation_error_sha256": prepared.plan_validation_error_sha256,
        "source_manuscript_sha256": _sha256(
            prepared.original_source_manuscript.encode("utf-8")
        ),
        "migration_input_manuscript_sha256": _sha256(
            prepared.source_manuscript.encode("utf-8")
        ),
        "migration_draft_path": (
            str(prepared.migration_draft_path)
            if prepared.migration_draft_path is not None
            else None
        ),
        "output_manuscript_sha256": _sha256(result.manuscript.encode("utf-8")),
        "output_bound_manuscript_sha256": _sha256(
            bound_manuscript.encode("utf-8")
        ),
        "planned_section_keys": list(prepared.planned_section_keys),
        "host_result_facts": [asdict(fact) for fact in prepared.host_result_facts],
        "repaired_section_keys": list(result.repaired_section_keys),
        "authority_repaired_section_keys": list(
            result.authority_repaired_section_keys
        ),
        "authority_filtered_section_keys": list(
            result.authority_filtered_section_keys
        ),
        "removed_unresolved_evidence_refs": list(
            result.removed_unresolved_evidence_refs
        ),
        "removed_unresolved_evidence_token_count": (
            result.removed_unresolved_evidence_token_count
        ),
        "normalized_claim_token_count": result.normalized_claim_token_count,
        "abstract_conclusion_boundary_repaired": (
            result.abstract_conclusion_boundary_repaired
        ),
        "removed_unknown_literature_keys": list(
            prepared.removed_unknown_literature_keys
        ),
        "removed_unknown_literature_sentences": (
            prepared.removed_unknown_literature_sentences
        ),
        "deterministic_literature_repairs": list(
            result.deterministic_literature_repairs
        ),
        "copied_evidence_files": list(copied_evidence),
        "quality_status": result.quality_audit.status,
        "literature_status": result.literature_audit.status,
        "provider": provider,
        "model": model,
        "provider_summary": dict(provider_summary),
        "provider_ledger": provider_ledger,
        "roles_used": ["writer"] if provider_summary.get("n_calls") else [],
        "forbidden_roles": ["planner", "executor", "coder", "figure"],
        "analysis_steps_executed": 0,
        "source_run_modified": False,
        "claim_ceiling": "analysis_only",
        "publication_authorized": False,
    }
    _atomic_write(
        output / "writer_only_migration_receipt.json",
        (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return receipt


def publish_writer_only_failure(
    prepared: PreparedWriterOnlyMigration,
    *,
    output_dir: Path,
    error: BaseException,
    provider: str,
    model: str,
    provider_summary: Mapping[str, Any],
    provider_ledger: str,
) -> dict[str, Any]:
    """Write a PHI-safe failure receipt without publishing a replacement draft."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    code = (
        error.code
        if isinstance(error, WriterOnlyMigrationError)
        else "WRITER_ONLY_UNEXPECTED_FAILURE_PRIOR_PRESERVED"
    )
    receipt = {
        "schema_version": WRITER_ONLY_MIGRATION_SCHEMA,
        "status": "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(prepared.source_run_dir),
        "source_hashes": dict(prepared.source_hashes),
        "reason_code": code,
        "exception_type": type(error).__name__,
        "error_sha256": _sha256(str(error).encode("utf-8")),
        "provider": provider,
        "model": model,
        "provider_summary": dict(provider_summary),
        "provider_ledger": provider_ledger,
        "replacement_published": False,
        "source_run_modified": False,
        "claim_ceiling": "analysis_only",
        "publication_authorized": False,
    }
    _atomic_write(
        output / "writer_only_migration_receipt.json",
        (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return receipt


__all__ = [
    "PreparedWriterOnlyMigration",
    "WRITER_ONLY_MIGRATION_SCHEMA",
    "WriterOnlyMigrationError",
    "WriterOnlyMigrationResult",
    "prepare_writer_only_migration",
    "publish_writer_only_failure",
    "publish_writer_only_result",
    "repair_writer_only",
    "writer_only_preflight_payload",
]
