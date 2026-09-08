"""Web-owned report-only revisions of approved, completed analysis.

Uses the normal submission/provider authority, but never constructs a pipeline
or probes Docker. Revisions and their failures live outside the sealed source
run; the original scientific gates are not rewritten or promoted.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from easyicu.research_agent.authority.provider_hard_stop import (
    ProviderHardStopLimits,
    validate_provider_transport_reservation_capacity,
)
from easyicu.research_agent.reporting.registered_report_inputs import (
    bind_registered_report_numbers,
    build_registered_report_reader,
    prepare_registered_report_repair,
)
from easyicu.research_agent.reporting.writer_only_migration import (
    WriterOnlyMigrationError,
    publish_writer_only_failure,
    publish_writer_only_result,
    repair_writer_only,
    writer_only_preflight_payload,
)
from easyicu.research_agent.reporting.manuscript_quality import render_reader_manuscript
from easyicu.webserver import agent_pipeline_runs as pipeline_owner
from easyicu.webserver import provider_adapter, run_artifact_disclosure, study_contexts
from easyicu.webserver import dataio
from easyicu.webserver.report_revision_export import build_revision_figure_gallery, export_revision_pdf
from easyicu.research_agent.reporting.revision_figures import build_revision_figure_bundle
from easyicu.webserver.report_revision_replay import load_failed_writer_replay


def _source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise WriterOnlyMigrationError(
                code="WRITER_ONLY_SOURCE_SYMLINK",
                detail="Source contains a symbolic link.",
            )
        if path.is_file():
            relative = str(path.relative_to(root)).encode()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _report_only_limits(limits: ProviderHardStopLimits) -> ProviderHardStopLimits:
    """Narrow the approved budget; a report revision cannot expand it."""
    narrowed = replace(
        limits,
        max_provider_attempts_per_run=min(6, limits.max_provider_attempts_per_run),
        max_provider_attempts_per_batch=min(6, limits.max_provider_attempts_per_batch),
        max_wall_clock_seconds_per_task=min(600, limits.max_wall_clock_seconds_per_task),
    )
    # Preserve approved token ceilings: transports without an enforceable
    # output cap need the existing conservative completion reservation.
    validate_provider_transport_reservation_capacity(narrowed)
    return narrowed


def _current_revision_input(target) -> tuple[Path | None, str | None]:
    """Resume the verified current report, retaining its editorial changes."""
    wrapper = target.wrapper_dir
    path = wrapper / "manuscript_provenance.json"
    if not path.exists():
        return None, None
    raw = path.read_bytes()
    provenance = json.loads(raw)
    revision = provenance.get("report_revision")
    if not revision:
        return None, None
    ledger = json.loads((wrapper / "evidence_ledger.json").read_text())
    identifier = str(revision.get("revision_id") or "")
    if (path.is_symlink() or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", identifier)
        or revision.get("schema_version") != "easyicu.web-report-revision/1"
        or revision.get("status") != "pass" or revision.get("source_run_id") != target.pipeline_run_id
        or revision.get("claim_ceiling") != "analysis_only" or revision.get("publication_authorized") is not False
        or revision.get("output_sha256") != provenance.get("manuscript_sha256")
        or not any(row.get("name") == path.name and row.get("sha256") == hashlib.sha256(raw).hexdigest()
                   for row in ledger["artifacts"])):
        raise WriterOnlyMigrationError(code="WRITER_ONLY_CURRENT_REVISION_CHANGED", detail="Current report binding is invalid.")
    root = wrapper / "report_revisions" / identifier
    receipt_path = root / "writer_only_migration_receipt.json"
    bound_path = root / "manuscript_bound.md"
    if (root.parent.is_symlink() or root.is_symlink() or receipt_path.is_symlink() or bound_path.is_symlink()
        or hashlib.sha256(receipt_path.read_bytes()).hexdigest() != revision.get("receipt_sha256")
        or hashlib.sha256(bound_path.read_bytes()).hexdigest() != revision.get("output_sha256")):
        raise WriterOnlyMigrationError(code="WRITER_ONLY_CURRENT_REVISION_CHANGED", detail="Current report files changed.")
    receipt = json.loads(receipt_path.read_text())
    canonical = revision.get("canonical_sha256")
    draft = root / ("manuscript_canonical.md" if canonical else "manuscript_scaffold.md")
    expected = canonical or receipt.get("output_manuscript_sha256")
    if draft.is_symlink() or hashlib.sha256(draft.read_bytes()).hexdigest() != expected:
        raise WriterOnlyMigrationError(code="WRITER_ONLY_CURRENT_REVISION_CHANGED", detail="Current editable report changed.")
    return draft, identifier


def make_report_only_run_runner(
    *,
    study_context,
    project_root,
    provider,
    provider_environment,
    credential_source,
    execution_resume_source_run_id,
    budget_mode,
    export_path,
    **other: Any,
):
    """Resolve approved source before job creation; revalidate at job start."""

    if budget_mode != "full_reviewed" or any(other.values()):
        raise pipeline_owner.ResearchPipelineRunError(
            "report_only_scope_invalid",
            "Report repair requires an exact reviewed run and no planning amendments.",
        )
    limits = _report_only_limits(
        provider_adapter.web_research_agent_hard_stop_limits(budget_mode)
    )
    target = pipeline_owner._resolve_execution_resume_wrapper(
        study=study_context,
        project_root=project_root,
        source_run_id=execution_resume_source_run_id,
    )
    seed = pipeline_owner.load_recovery_seed(target.wrapper_dir)
    if seed is None:
        raise pipeline_owner.ResearchPipelineRunError(
            "report_only_recovery_required",
            "The exact approved configuration is unavailable.",
        )
    dataio.validate_research_pipeline_source(
        export_path,
        database=(study_context.get("data_source") or {}).get("database"),
        expected_binding=seed.prepared_package_binding,
    )
    approved_config = pipeline_owner._validated_execution_retry_config(
        current_config=None,
        target=target,
        recovery_seed=seed,
        current_scientific_digest=study_contexts.scientific_configuration_sha256(
            study_context
        ),
        prepared_package_binding=seed.prepared_package_binding,
    )
    run_dir = target.wrapper_dir / "pipeline" / target.pipeline_run_id
    draft_path, parent_revision = _current_revision_input(target)
    prepare_registered_report_repair(run_dir, migration_draft=draft_path)
    source_digest = _source_fingerprint(run_dir)

    def runner(job):
        from easyicu.research_agent.agents.reporting import WriterAgent
        from easyicu.research_agent.authority.provider_hard_stop import (
            ProviderHardStopLedger,
        )
        from easyicu.research_agent.providers.cost import CostMeter, MeteredClient
        from easyicu.research_agent.providers.hard_stop import HardStopClient

        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", str(job.id)):
            raise ValueError("Invalid report revision identifier")
        current = study_contexts.get_context(study_context["id"])
        if (
            study_contexts.scientific_configuration_sha256(current)
            != seed.scientific_configuration_sha256
        ):
            raise WriterOnlyMigrationError(
                code="WRITER_ONLY_STUDY_CHANGED",
                detail="Scientific setup changed before repair.",
            )
        if _source_fingerprint(run_dir) != source_digest:
            raise WriterOnlyMigrationError(
                code="WRITER_ONLY_SOURCE_CHANGED",
                detail="Sealed source changed before repair.",
            )
        if _current_revision_input(target) != (draft_path, parent_revision):
            raise WriterOnlyMigrationError(code="WRITER_ONLY_CURRENT_REVISION_CHANGED", detail="Current report changed before repair.")
        prepared = prepare_registered_report_repair(run_dir, migration_draft=draft_path)
        output = target.wrapper_dir / "report_revisions" / str(job.id)
        if output.parent.is_symlink():
            raise ValueError("Report revision parent must not be a symbolic link")
        output.mkdir(parents=True, exist_ok=False)
        pipeline_owner._write_json(
            output / "preflight.json", writer_only_preflight_payload(prepared)
        )
        ledger_path = output / "runtime" / "provider_hard_stop.json"
        ledger = ProviderHardStopLedger(
            path=ledger_path,
            task_ids=[str(job.id)],
            limits=limits,
            batch_id="report-" + str(job.id),
            declaration_sha256=seed.scientific_configuration_sha256,
        )
        task = ledger.start_task(str(job.id))
        meter = CostMeter(runtime_dir=output / "runtime")
        public_provider = {key: provider.get(key) for key in ("provider", "model")}
        replay = load_failed_writer_replay(target.wrapper_dir, prepared)
        try:
            pipeline_owner._progress(
                job,
                step="report_repair",
                label="Repairing only the manuscript from sealed aggregate evidence; no analysis is being executed",
            )
            client, _ = provider_adapter.build_research_agent_provider_client(
                dict(provider),
                request_timeout=task.cap_timeout(180),
                request_hard_timeout=task.cap_timeout(240),
                environ=provider_environment,
            )
            metered = MeteredClient(
                HardStopClient(client, role="writer", task=task),
                role="writer",
                meter=meter,
                model_override=provider.get("model"),
            )

            class RecordingWriter(WriterAgent):
                attempt = 0

                def _call_section(self, **kwargs):
                    self.attempt += 1
                    section = str(kwargs["section_name"])
                    pipeline_owner._progress(
                        job,
                        step="report_repair",
                        label=f"Repairing report section: {section}",
                    )
                    text = replay.section(section_name=section, instruction=kwargs["instruction"]) if replay else super()._call_section(**kwargs)
                    pipeline_owner._write_json(
                        output
                        / "runtime"
                        / f"writer_candidate_{self.attempt:02d}.json",
                        {"section": section, "instruction": kwargs["instruction"], "text": text,
                         "replayed_from_revision": replay.revision_id if replay else None},
                    )
                    return text

            writer = RecordingWriter(
                metered,
                language=approved_config.manuscript_language,
                nature_writing_enabled=approved_config.enable_nature_writing_skill,
            )
            result = repair_writer_only(prepared, writer=writer)
            canonical = result.manuscript.encode("utf-8")
            (output / "manuscript_canonical.md").write_bytes(canonical)
            numbered, binding_count = bind_registered_report_numbers(
                run_dir, result.manuscript
            )
            result = replace(
                result,
                manuscript=numbered,
                reader_manuscript=render_reader_manuscript(numbered),
            )
            if _source_fingerprint(run_dir) != source_digest:
                raise WriterOnlyMigrationError(
                    code="WRITER_ONLY_SOURCE_CHANGED",
                    detail="Source changed during repair.",
                )
            receipt = publish_writer_only_result(
                prepared,
                result,
                output_dir=output,
                provider=str(provider.get("provider") or ""),
                model=str(provider.get("model") or ""),
                provider_summary=meter.summary(
                    hard_stop_accounting=task.accounting_summary()
                ),
                provider_ledger=str(ledger_path),
            )
            revision = {
                "schema_version": "easyicu.web-report-revision/1",
                "revision_id": str(job.id),
                "source_run_id": target.pipeline_run_id,
                "status": "pass",
                "source_fingerprint": source_digest,
                "receipt_sha256": hashlib.sha256(
                    (output / "writer_only_migration_receipt.json").read_bytes()
                ).hexdigest(),
                "output_sha256": receipt["output_bound_manuscript_sha256"],
                "numeric_binding_count": binding_count,
                "parent_revision_id": parent_revision,
                "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
                "replayed_from_revision": replay.revision_id if replay else None,
                "replayed_section_count": replay.cursor if replay else 0,
                "analysis_steps_executed": 0,
                "claim_ceiling": "analysis_only",
                "publication_authorized": False,
            }
            provenance = build_registered_report_reader(
                run_dir, (output / "manuscript_bound.md").read_text(encoding="utf-8"),
            )
            figure_bundle = build_revision_figure_bundle(prepared=prepared, output=output)
            revision["figure_revision"] = {
                "receipt_path": "figures/figure_revision_receipt.json",
                "receipt_sha256": figure_bundle.receipt_sha256,
            }
            revision["pdf_artifact"] = export_revision_pdf(
                prepared=prepared, output=output, reader=result.reader_manuscript,
                revision=revision, figure_bundle=figure_bundle,
            )
            pipeline_owner._write_json(output / "manuscript_provenance.json", provenance)
            if _source_fingerprint(run_dir) != source_digest:
                raise WriterOnlyMigrationError(code="WRITER_ONLY_SOURCE_CHANGED", detail="Source analysis changed during report export.")
            if _current_revision_input(target) != (draft_path, parent_revision):
                raise WriterOnlyMigrationError(code="WRITER_ONLY_CURRENT_REVISION_CHANGED", detail="Current report changed during repair.")
            projected = _project_revision(
                target,
                study_context,
                result.reader_manuscript,
                revision,
                public_provider,
                provenance=provenance,
                pdf_path=output / "pdf" / "manuscript_revision.pdf",
                figure_gallery=build_revision_figure_gallery(prepared, figure_bundle=figure_bundle),
            )
            task.finish(score={"report_quality": "pass", "publication_authorized": False})
            return projected
        except BaseException as exc:
            task.finish(error=type(exc).__name__)
            publish_writer_only_failure(
                prepared,
                output_dir=output,
                error=WriterOnlyMigrationError(
                    code=getattr(exc, "code", "WRITER_ONLY_REPAIR_FAILED"),
                    detail=type(exc).__name__,
                ),
                provider=str(provider.get("provider") or ""),
                model=str(provider.get("model") or ""),
                provider_summary=meter.summary(
                    hard_stop_accounting=task.accounting_summary()
                ),
                provider_ledger=str(ledger_path),
            )
            raise

    return runner


def _project_revision(
    target, study, reader: str, revision: dict, provider: dict, *, provenance: dict,
    pdf_path: Path | None = None,
    figure_gallery: dict | None = None,
) -> dict:
    """Replace only the mutable Web draft projection, never source run gates."""

    wrapper = target.wrapper_dir
    if (
        provenance.get("schema_version") != "easyicu.manuscript-provenance/1"
        or provenance.get("manuscript_sha256") != revision.get("output_sha256")
        or provenance.get("claim_ceiling") != "analysis_only"
        or provenance.get("publication_authorized") is not False
    ):
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_READER_BINDING_FAILED", detail="Reader does not match this revision.",
        )
    ledger = json.loads((wrapper / "evidence_ledger.json").read_text())
    provenance = {**provenance, "report_revision": revision}
    gallery = next((row for row in ledger["artifacts"] if row.get("name") == "figure_gallery.json"), None)
    if figure_gallery is not None:
        gallery_sha = hashlib.sha256(json.dumps(
            figure_gallery, ensure_ascii=False, indent=2, sort_keys=True,
        ).encode("utf-8")).hexdigest()
        provenance["figure_gallery_artifact"] = {"name": "figure_gallery.json", "sha256": gallery_sha}
    elif gallery:
        provenance["figure_gallery_artifact"] = {"name": "figure_gallery.json", "sha256": gallery["sha256"]}
    draft = json.loads((wrapper / "manuscript_draft.json").read_text())
    draft.update(
        status="report_revision_quality_pass_analysis_only",
        markdown_preview=reader,
        source="registered_report_only_revision",
        report_revision=revision,
        claims=[],
        sentences=[],
        reader=provenance,
    )
    payloads = {"manuscript_draft.json": draft, "manuscript_provenance.json": provenance}
    if figure_gallery is not None:
        payloads["figure_gallery.json"] = figure_gallery
    pdf_bytes = None
    if pdf_path is not None:
        pdf = revision.get("pdf_artifact") or {}
        pdf_bytes = pdf_path.read_bytes()
        if (
            pdf_path.is_symlink() or (wrapper / "manuscript_revision.pdf").is_symlink()
            or pdf.get("name") != "manuscript_revision.pdf"
            or pdf.get("revision_id") != revision.get("revision_id")
            or pdf.get("manuscript_sha256") != revision.get("output_sha256")
            or hashlib.sha256(pdf_bytes).hexdigest() != pdf.get("sha256")
            or not pdf_bytes.startswith(b"%PDF-") or len(pdf_bytes) > 16 * 1024 * 1024
        ):
            raise WriterOnlyMigrationError(code="WRITER_ONLY_PDF_BINDING_FAILED", detail="PDF does not match this revision.")
    elif revision.get("pdf_artifact"):
        raise WriterOnlyMigrationError(code="WRITER_ONLY_PDF_BINDING_FAILED", detail="Revision PDF is missing.")
    if not run_artifact_disclosure.scan_browser_projection(payloads)["passed"]:
        raise WriterOnlyMigrationError(
            code="WRITER_ONLY_PROJECTION_PRIVACY_FAILED",
            detail="Report preview withheld.",
        )
    # Rebuild only reader projections; every scientific gate stays as it
    # was. The revision is a separately identified report, not a new analysis.
    for name, payload in payloads.items():
        pipeline_owner._write_json(wrapper / name, payload)
    names = set(payloads)
    if pdf_bytes is not None:
        (wrapper / "manuscript_revision.pdf").write_bytes(pdf_bytes)
        names.add("manuscript_revision.pdf")
    ledger["artifacts"] = [
        row for row in ledger["artifacts"] if row.get("name") not in names
    ] + [
        pipeline_owner._artifact_record(wrapper / name) for name in sorted(names)
    ]
    pipeline_owner._write_json(wrapper / "evidence_ledger.json", ledger)
    gate = json.loads((wrapper / "quality_gate.json").read_text())["gate"]
    return {
        "run_id": target.pipeline_run_id,
        "study_id": study["id"],
        "mode": "research_agent_pipeline",
        "run_type": "full",
        "engine": "easyicu.research_agent.pipeline",
        "project_dir": str(wrapper),
        "gate": gate,
        "provider": provider,
        "report_revision": revision,
        "artifacts": [
            *ledger["artifacts"],
            pipeline_owner._artifact_record(wrapper / "evidence_ledger.json"),
        ],
        "human_review_pending": False,
        "pending_reviews": [],
        "uploads": 0,
    }
