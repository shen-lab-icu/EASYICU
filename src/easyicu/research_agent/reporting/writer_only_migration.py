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
from typing import Any, Mapping, Optional
import uuid

from ..literature import LiteratureBundle
from ..research_context.typed import ResearchContextAuthority, parse_research_context_json
from ..schema import AnalysisPlan
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
    render_reader_manuscript,
)
from .manuscript_sections import quality_repair_section_keys


WRITER_ONLY_MIGRATION_SCHEMA = "easyicu.writer_only_manuscript_migration/1"
_EVIDENCE_TOKEN = re.compile(r"\{evidence:([^{}\s]+)\}")
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
    context: ResearchContextAuthority
    plan: AnalysisPlan
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


@dataclass(frozen=True)
class WriterOnlyMigrationResult:
    manuscript: str
    reader_manuscript: str
    repaired_section_keys: tuple[str, ...]
    quality_audit: ManuscriptQualityAudit
    literature_audit: ManuscriptLiteratureAudit
    deterministic_literature_repairs: tuple[Mapping[str, Any], ...]


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


def prepare_writer_only_migration(run_dir: Path) -> PreparedWriterOnlyMigration:
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
    evidence_digest = (source / "writer_evidence_digest.md").read_text(
        encoding="utf-8"
    )
    try:
        context = parse_research_context_json(
            (source / "research_context.json").read_bytes()
        )
        plan = AnalysisPlan.model_validate_json(
            (source / "analysis_plan.json").read_text(encoding="utf-8")
        )
        literature = LiteratureBundle.model_validate_json(
            (source / "preplan_literature_bundle.json").read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_TYPED_INPUT_INVALID",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    manuscript, unknown_keys, removed_sentences = (
        remove_sentences_with_unknown_literature_keys(
            original_manuscript,
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
    source_quality = audit_manuscript_quality(
        manuscript,
        expected_display_labels=labels,
    )
    source_literature = audit_manuscript_literature(manuscript, literature)
    return PreparedWriterOnlyMigration(
        source_run_dir=source,
        original_source_manuscript=original_manuscript,
        source_manuscript=manuscript,
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
            expected_display_labels=labels,
        ),
        removed_unknown_literature_keys=tuple(unknown_keys),
        removed_unknown_literature_sentences=int(removed_sentences),
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
        "source_quality_status": prepared.source_quality_audit.status,
        "source_quality_findings": [
            asdict(finding)
            for finding in prepared.source_quality_audit.findings
        ],
        "source_literature_status": prepared.source_literature_audit.status,
        "planned_section_keys": list(prepared.planned_section_keys),
        "removed_unknown_literature_keys": list(
            prepared.removed_unknown_literature_keys
        ),
        "removed_unknown_literature_sentences": (
            prepared.removed_unknown_literature_sentences
        ),
        "expected_display_labels": list(prepared.expected_display_labels),
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

    try:
        manuscript, repaired_keys = writer.repair_existing(
            prepared.source_manuscript,
            context=prepared.context,
            evidence_ids=prepared.evidence_ids,
            evidence_digest=prepared.evidence_digest,
            literature_digest=prepared.literature_digest,
            administrative_authority=prepared.administrative_authority,
        )
    except Exception as exc:
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_REPAIR_FAILED_PRIOR_PRESERVED",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
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
    quality = audit_manuscript_quality(
        manuscript,
        expected_display_labels=prepared.expected_display_labels,
    )
    if quality.status != "pass":
        codes = sorted({finding.code for finding in quality.findings})
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_QUALITY_AUDIT_FAILED_PRIOR_PRESERVED",
            detail=", ".join(codes),
        )
    literature = audit_manuscript_literature(manuscript, prepared.literature)
    if literature.status != "pass":
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_LITERATURE_AUDIT_FAILED_PRIOR_PRESERVED",
            detail=literature.message,
        )
    return WriterOnlyMigrationResult(
        manuscript=manuscript,
        reader_manuscript=render_reader_manuscript(manuscript),
        repaired_section_keys=tuple(repaired_keys),
        quality_audit=quality,
        literature_audit=literature,
        deterministic_literature_repairs=tuple(deterministic_literature_repairs),
    )


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
    _atomic_write(output / "manuscript_scaffold.md", result.manuscript.encode("utf-8"))
    _atomic_write(
        output / "manuscript_reader.md",
        result.reader_manuscript.encode("utf-8"),
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
        "source_manuscript_sha256": _sha256(
            prepared.original_source_manuscript.encode("utf-8")
        ),
        "migration_input_manuscript_sha256": _sha256(
            prepared.source_manuscript.encode("utf-8")
        ),
        "output_manuscript_sha256": _sha256(result.manuscript.encode("utf-8")),
        "planned_section_keys": list(prepared.planned_section_keys),
        "repaired_section_keys": list(result.repaired_section_keys),
        "removed_unknown_literature_keys": list(
            prepared.removed_unknown_literature_keys
        ),
        "removed_unknown_literature_sentences": (
            prepared.removed_unknown_literature_sentences
        ),
        "deterministic_literature_repairs": list(
            result.deterministic_literature_repairs
        ),
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
